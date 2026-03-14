"""
hybrid_search.py

Hybrid retrieval combining:

1. Dense search (CLIP + LanceDB)
2. Keyword search (BM25)
3. Multi-query retrieval
4. Reciprocal Rank Fusion (RRF)
5. Classifier-prioritized chapter boosting
6. Taxonomy boosting
7. Cross-encoder reranking

This sits on top of vector_search.py without modifying it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from retrieval.taxonomy_router import TaxonomyRouter
from retrieval.vector_search import vector_search
from retrieval.reranker import CrossEncoderReranker


# =========================
# Config
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks" / "chunks.jsonl"

RRF_K = 60
PER_LIST_K = 25
BM25_TOP_K = 25
FUSION_TOP_K = 20
BM25_MIN_SCORE = 0.1

CLASSIFIER_CHAPTER_BOOST = 0.08
CLASSIFIER_SUBHEADING_BOOST = 0.04
LEXICAL_CHAPTER_BOOST = 0.015
LEXICAL_SUBHEADING_BOOST = 0.02
QUERY_TERM_METADATA_BOOST = 0.03

reranker = CrossEncoderReranker()
taxonomy_router = TaxonomyRouter()


# =========================
# Load chunks once
# =========================

all_chunks: List[Dict[str, Any]] = []
chunk_lookup: Dict[str, Dict[str, Any]] = {}

if not CHUNKS_FILE.exists():
    raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        if row.get("type") != "text":
            continue

        text = (row.get("text") or "").strip()
        if not text:
            continue

        cid = row["chunk_id"]
        meta = row.get("metadata", {}) or {}

        item = {
            "id": cid,
            "text": text,
            "chapter": meta.get("chapter"),
            "subheading": meta.get("subheading"),
            "source": meta.get("source"),
        }

        all_chunks.append(item)
        chunk_lookup[cid] = item


# =========================
# BM25 index once
# =========================

def _tokenize(text: str) -> List[str]:
    return (text or "").lower().split()


tokenized_docs = [_tokenize(c["text"]) for c in all_chunks]
bm25 = BM25Okapi(tokenized_docs)


# =========================
# Helpers
# =========================

def _rrf(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def _dedupe_queries(
    original_query: str,
    enhanced_queries: Optional[List[str]],
    keywords: Optional[List[str]],
) -> tuple[list[str], list[str]]:
    dense_queries = [original_query] + (enhanced_queries or [])
    dense_queries = [q.strip() for q in dense_queries if q and q.strip()]
    dense_queries = list(dict.fromkeys(dense_queries))

    sparse_queries = dense_queries + (keywords or [])
    sparse_queries = [q.strip() for q in sparse_queries if q and q.strip()]
    sparse_queries = list(dict.fromkeys(sparse_queries))

    return dense_queries, sparse_queries


def _distance_to_confidence(d: Optional[float]) -> int:
    if d is None:
        return 10

    d = float(d)

    if d <= 0.18:
        return 90
    if d <= 0.24:
        return 80
    if d <= 0.30:
        return 70
    if d <= 0.38:
        return 55
    if d <= 0.48:
        return 40
    return 25


def _ordered_unique(items: List[str]) -> List[str]:
    out = []
    seen = set()

    for item in items:
        cleaned = (item or "").strip()
        if not cleaned:
            continue

        key = cleaned.lower()
        if key in seen:
            continue

        seen.add(key)
        out.append(cleaned)

    return out


# =========================
# Main Hybrid Search
# =========================

def hybrid_search(
    original_query: str,
    enhanced_queries: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    image_path: Optional[Path] = None,
    image_context_text: Optional[str] = None,
    classifier_chapter_hints: Optional[List[str]] = None,
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    Returns hybrid-retrieved and reranked text results.

    Output format is intentionally close to vector_search.py so orchestrator
    changes remain small.
    """

    dense_queries, sparse_queries = _dedupe_queries(
        original_query=original_query,
        enhanced_queries=enhanced_queries,
        keywords=keywords,
    )

    router_query = " ".join(
        [original_query]
        + (keywords or [])
        + ([image_context_text] if image_context_text else [])
    )

    lexical_predicted_chapters = taxonomy_router.predict_chapters(router_query)
    classifier_chapter_hints = _ordered_unique(classifier_chapter_hints or [])

    combined_chapters = _ordered_unique(
        classifier_chapter_hints + lexical_predicted_chapters
    )

    # Accumulate fusion scores
    fusion_scores: Dict[str, float] = defaultdict(float)

    # Track best dense distance globally
    best_dense_text_distance: Optional[float] = None
    best_dense_chunk: Optional[Dict[str, Any]] = None

    # Keep some image results from the main/original dense search only
    image_results: List[Dict[str, Any]] = []

    # -------------------------
    # 1) Dense retrieval
    # -------------------------
    for i, q in enumerate(dense_queries):
        dense_out = vector_search(
            original_query=q,
            enhanced_queries=None,
            image_path=image_path if i == 0 else None,
            image_context_text=image_context_text if i == 0 else None,
            top_k=PER_LIST_K,
            overfetch_k=40,
        )

        dense_results = dense_out.get("text_results", [])
        if i == 0:
            image_results = dense_out.get("image_results", [])

        d = dense_out.get("retrieval_distance")
        if d is not None:
            d = float(d)
            if best_dense_text_distance is None or d < best_dense_text_distance:
                best_dense_text_distance = d

        for rank, chunk in enumerate(dense_results[:PER_LIST_K], start=1):
            cid = chunk["id"]
            fusion_scores[cid] += _rrf(rank)

            dist = chunk.get("distance")
            if dist is not None:
                try:
                    dist = float(dist)
                    if best_dense_chunk is None or dist < float(best_dense_chunk["distance"]):
                        best_dense_chunk = {
                            "id": cid,
                            "distance": dist,
                            "chapter": chunk.get("chapter"),
                            "subheading": chunk.get("subheading"),
                        }
                except Exception:
                    pass

    # -------------------------
    # 2) Sparse retrieval (BM25)
    # -------------------------
    for q in sparse_queries:
        tokens = _tokenize(q)
        if not tokens:
            continue

        scores = bm25.get_scores(tokens)
        ranked_idx = np.argsort(scores)[::-1]

        sparse_rank = 0
        for idx in ranked_idx:
            score = float(scores[idx])

            if score < BM25_MIN_SCORE:
                continue

            cid = all_chunks[idx]["id"]
            sparse_rank += 1
            fusion_scores[cid] += _rrf(sparse_rank)

            if sparse_rank >= BM25_TOP_K:
                break

    # -------------------------
    # 2.5) Classifier chapter hint boost (priority)
    # -------------------------
    classifier_hinted_subs: List[str] = []
    for ch in classifier_chapter_hints:
        classifier_hinted_subs.extend(taxonomy_router.taxonomy.get(ch, []))

    classifier_hinted_subs = _ordered_unique(classifier_hinted_subs)

    for cid in fusion_scores.keys():
        chunk = chunk_lookup.get(cid)
        if not chunk:
            continue

        chapter = (chunk.get("chapter") or "").strip()
        subheading = (chunk.get("subheading") or "").strip()

        if chapter in classifier_chapter_hints:
            fusion_scores[cid] += CLASSIFIER_CHAPTER_BOOST

        if subheading and subheading in classifier_hinted_subs:
            fusion_scores[cid] += CLASSIFIER_SUBHEADING_BOOST

    # -------------------------
    # 3) Lexical taxonomy boost
    # -------------------------
    if lexical_predicted_chapters:
        lexical_predicted_subs: List[str] = []

        for ch in lexical_predicted_chapters:
            subs = taxonomy_router.taxonomy.get(ch, [])
            lexical_predicted_subs.extend(subs)

        lexical_predicted_subs = _ordered_unique(lexical_predicted_subs)

        for cid in fusion_scores.keys():
            chunk = chunk_lookup.get(cid)
            if not chunk:
                continue

            chapter = chunk.get("chapter")
            subheading = chunk.get("subheading")

            if chapter in lexical_predicted_chapters:
                fusion_scores[cid] += LEXICAL_CHAPTER_BOOST

            if subheading and subheading in lexical_predicted_subs:
                fusion_scores[cid] += LEXICAL_SUBHEADING_BOOST

    # -------------------------
    # 3.5) Query-term metadata boost
    # -------------------------
    query_terms = set(
        re.findall(r"[a-z]+", " ".join(keywords or []) + " " + original_query.lower())
    )

    for cid in fusion_scores.keys():
        chunk = chunk_lookup.get(cid)
        if not chunk:
            continue

        chapter = (chunk.get("chapter") or "").lower()
        subheading = (chunk.get("subheading") or "").lower()

        meta_text = f"{chapter} {subheading}"

        if any(term in meta_text for term in query_terms):
            fusion_scores[cid] += QUERY_TERM_METADATA_BOOST

    # -------------------------
    # 4) Fusion candidate selection
    # -------------------------
    ranked_ids = sorted(
        fusion_scores.keys(),
        key=lambda cid: fusion_scores[cid],
        reverse=True
    )

    candidates: List[Dict[str, Any]] = []
    for cid in ranked_ids[:FUSION_TOP_K]:
        chunk = chunk_lookup.get(cid)
        if not chunk:
            continue

        candidate = {
            "id": chunk["id"],
            "text": chunk["text"],
            "chapter": chunk.get("chapter"),
            "subheading": chunk.get("subheading"),
            "distance": None,
            "modality": "text",
            "fusion_score": fusion_scores[cid],
        }
        candidates.append(candidate)

    # Fill dense distances where possible using the original dense query results first
    dense_out_main = vector_search(
        original_query=original_query,
        enhanced_queries=enhanced_queries or [],
        image_path=image_path,
        image_context_text=image_context_text,
        top_k=max(top_k, PER_LIST_K),
        overfetch_k=40,
    )

    dense_map = {
        c["id"]: c
        for c in dense_out_main.get("text_results", [])
    }

    for c in candidates:
        dense_chunk = dense_map.get(c["id"])
        if dense_chunk:
            c["distance"] = dense_chunk.get("distance")
            if dense_chunk.get("rerank_score") is not None:
                c["rerank_score"] = dense_chunk.get("rerank_score")

    # -------------------------
    # 5) Cross-encoder rerank
    # -------------------------
    rerank_query = original_query.strip()
    if image_context_text and image_context_text.strip():
        rerank_query = (rerank_query + "\n" + image_context_text.strip()).strip()

    reranked_text = reranker.rerank(
        query=rerank_query,
        candidates=candidates,
        text_key="text",
        top_k=top_k,
    )

    top_rerank_score = None
    if reranked_text:
        top_rerank_score = reranked_text[0].get("rerank_score")

    retrieval_confidence = _distance_to_confidence(best_dense_text_distance)

    return {
        "text_results": reranked_text,
        "image_results": image_results[:top_k],
        "retrieval_distance": best_dense_text_distance,
        "retrieval_confidence": retrieval_confidence,
        "did_rerank": True,
        "top_rerank_score": top_rerank_score,
        "best_dense_text_distance": best_dense_text_distance,
        "best_dense_chunk": best_dense_chunk,
        "predicted_chapters": combined_chapters,
        "classifier_chapter_hints": classifier_chapter_hints,
        "lexical_predicted_chapters": lexical_predicted_chapters,
        "fusion_debug": {
            "dense_queries": dense_queries,
            "sparse_queries": sparse_queries,
            "ranked_ids": ranked_ids[:FUSION_TOP_K],
        },
    }


# =========================
# Quick test
# =========================

if __name__ == "__main__":
    result = hybrid_search(
        original_query="What does crossed arms mean during negotiation?",
        enhanced_queries=[
            "Arm barrier signals during negotiation",
            "Interpreting crossed arms in a negotiation context",
            "Crossed arms as a defensive or closed-off signal in business discussions",
            "The meaning of arm crossing in power dynamics during negotiation",
        ],
        keywords=[
            "crossed arms",
            "negotiation",
            "arm barrier signals",
            "body language",
            "defensive signals",
            "closed-off posture",
        ],
        classifier_chapter_hints=["ARM SIGNALS"],
        image_path=None,
        image_context_text=None,
        top_k=8,
    )

    print("Top text:", len(result["text_results"]))
    print("Top images:", len(result["image_results"]))
    print("retrieval_distance:", result["retrieval_distance"])
    print("retrieval_confidence:", result["retrieval_confidence"])
    print("predicted_chapters:", result["predicted_chapters"])
    print("classifier_chapter_hints:", result["classifier_chapter_hints"])
    print("lexical_predicted_chapters:", result["lexical_predicted_chapters"])
    print("best_dense_chunk:", result["best_dense_chunk"])