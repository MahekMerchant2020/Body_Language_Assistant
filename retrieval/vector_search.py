"""
vector_search.py

Multimodal vector search over LanceDB using:

1) CLIP for dense retrieval (text + image)
2) Cross-encoder reranker for text chunks (TEXT ONLY)
3) Optional MMR diversity to reduce redundancy
4) Distance -> retrieval_confidence calibration
5) Conditional rerank (only if dense retrieval seems weak)
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import lancedb
from PIL import Image
from transformers import CLIPModel, CLIPTokenizerFast, CLIPProcessor

from retrieval.reranker import CrossEncoderReranker


# =========================
# Config
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

LANCEDB_DIR = BASE_DIR / "lancedb_store"
LANCEDB_TABLE = "multimodal_pdf"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TEXT_WEIGHT = 0.6
IMAGE_WEIGHT = 0.4

# Dense overfetch for reranking/MMR
DEFAULT_OVERFETCH_K = 40

# Conditional rerank: if dense best distance is already strong, skip rerank
RERANK_IF_DISTANCE_ABOVE = 0.30  # tune on your data

# MMR: how many diverse candidates to keep before reranking
MMR_ENABLED = True
MMR_KEEP = 16
MMR_LAMBDA = 0.7  # higher -> more relevance, lower -> more diversity


# =========================
# Load Models Once
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = CLIPTokenizerFast.from_pretrained(CLIP_MODEL_NAME)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
clip_model.eval()

reranker = CrossEncoderReranker(model_name=RERANK_MODEL)


# =========================
# Utilities
# =========================

def l2_normalize(v: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / denom


def _as_tensor(output: Any) -> torch.Tensor:
    """
    Transformers version compatibility:
    Sometimes model.get_*_features returns Tensor,
    sometimes a ModelOutput with pooler_output or last_hidden_state.
    """
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        # fallback: mean pool tokens
        return output.last_hidden_state.mean(dim=1)
    raise TypeError(f"Unexpected CLIP output type: {type(output)}")


def embed_text(text: str) -> np.ndarray:
    inputs = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = clip_model.get_text_features(**inputs)
        feats = _as_tensor(output)

    feats = feats.to(dtype=torch.float32).cpu().numpy()
    feats = l2_normalize(feats)
    return feats[0]


def embed_image(image_path: Path) -> np.ndarray:
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    inputs = processor(images=[img], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = clip_model.get_image_features(**inputs)
        feats = _as_tensor(output)

    feats = feats.to(dtype=torch.float32).cpu().numpy()
    feats = l2_normalize(feats)
    return feats[0]


def fuse_vectors(text_vec: Optional[np.ndarray], image_vec: Optional[np.ndarray]) -> np.ndarray:
    if text_vec is not None and image_vec is not None:
        fused = TEXT_WEIGHT * text_vec + IMAGE_WEIGHT * image_vec
        norm = np.linalg.norm(fused)
        return fused / (norm + 1e-12)
    if text_vec is not None:
        return text_vec
    if image_vec is not None:
        return image_vec
    raise ValueError("No vectors to fuse.")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def mmr_select(
    query_vec: np.ndarray,
    candidates: List[Dict[str, Any]],
    candidate_vecs: List[np.ndarray],
    k: int,
    lambda_mult: float = 0.7,
) -> List[int]:
    """
    MMR: select a diverse subset of candidates.
    maximize λ * sim(q, d) - (1-λ) * max_sim(d, selected)
    """
    if not candidates or k <= 0:
        return []

    rel = [cosine_sim(query_vec, v) for v in candidate_vecs]
    selected: List[int] = []
    remaining = list(range(len(candidates)))

    first = int(np.argmax(rel))
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < k:
        best_idx = None
        best_score = -1e9

        for i in remaining:
            max_div = max(cosine_sim(candidate_vecs[i], candidate_vecs[j]) for j in selected)
            score = lambda_mult * rel[i] - (1 - lambda_mult) * max_div
            if score > best_score:
                best_score = score
                best_idx = i

        selected.append(best_idx)  # type: ignore[arg-type]
        remaining.remove(best_idx)  # type: ignore[arg-type]

    return selected


def distance_to_confidence(d: Optional[float]) -> int:
    """
    Heuristic calibration: lower distance => higher confidence.
    Tune breakpoints from your debug logs.
    """
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


def _clean_text(s: Any, max_chars: int = 3000) -> str:
    t = (s or "").strip()
    if len(t) > max_chars:
        t = t[:max_chars].rstrip() + "…"
    return t


# =========================
# Main Search
# =========================

def vector_search(
    original_query: Optional[str] = None,
    enhanced_queries: Optional[List[str]] = None,
    image_path: Optional[Path] = None,
    image_context_text: Optional[str] = None,   # from image_interpreter (full string, not caption)
    top_k: int = 8,
    overfetch_k: int = DEFAULT_OVERFETCH_K,
) -> Dict[str, Any]:

    if not LANCEDB_DIR.exists():
        raise RuntimeError("LanceDB store not found. Run ingestion first.")

    db = lancedb.connect(str(LANCEDB_DIR))
    table = db.open_table(LANCEDB_TABLE)

    # =========================
    # Build Text Vector (RAG fusion)
    # =========================

    text_vec = None

    if original_query:
        orig_vec = embed_text(original_query)

        if enhanced_queries:
            enh = [q for q in enhanced_queries if q and q.strip()]
            enhanced_vecs = [embed_text(q) for q in enh]
            enhanced_mean = np.mean(enhanced_vecs, axis=0)
            enhanced_mean /= (np.linalg.norm(enhanced_mean) + 1e-12)

            text_vec = 0.5 * orig_vec + 0.5 * enhanced_mean
            text_vec /= (np.linalg.norm(text_vec) + 1e-12)
        else:
            text_vec = orig_vec

    # Blend in image interpretation text (NOT caption) if provided
    if image_context_text and image_context_text.strip():
        context_vec = embed_text(image_context_text)
        if text_vec is None:
            text_vec = context_vec
        else:
            mixed = 0.7 * text_vec + 0.3 * context_vec
            text_vec = mixed / (np.linalg.norm(mixed) + 1e-12)

    image_vec = embed_image(image_path) if image_path else None

    final_vec = fuse_vectors(text_vec, image_vec)

    # =========================
    # Dense Retrieval (overfetch)
    # =========================

    raw = (
        table.search(final_vec.tolist())
        .limit(overfetch_k)
        .to_list()
    )

    text_candidates: List[Dict[str, Any]] = []
    image_candidates: List[Dict[str, Any]] = []

    for r in raw:
        modality = r.get("modality")
        item = {
            "id": r["id"],
            "distance": float(r["_distance"]),  # lower is better
            "text": r.get("text"),
            "chapter": r.get("chapter"),
            "subheading": r.get("subheading"),
            "image_path": r.get("image_path"),
            "modality": modality,
        }

        if modality == "text":
            # only keep candidates with usable text
            if (item["text"] or "").strip():
                item["text"] = _clean_text(item["text"])
                text_candidates.append(item)

        elif modality == "image":
            image_candidates.append(item)

    # Best dense distance (text)
    best_dense_text_distance = min([c["distance"] for c in text_candidates], default=None)

    # =========================
    # Optional MMR diversity (TEXT candidates)
    # =========================

    if MMR_ENABLED and text_candidates and (text_vec is not None or image_vec is not None):
        # Use final_vec for relevance, and CLIP-text embeddings of candidate texts for diversity
        # (small N, fast enough)
        cand_vecs = [embed_text(c["text"]) for c in text_candidates]
        keep_n = min(MMR_KEEP, len(text_candidates))
        mmr_idx = mmr_select(
            query_vec=final_vec,
            candidates=text_candidates,
            candidate_vecs=cand_vecs,
            k=keep_n,
            lambda_mult=MMR_LAMBDA,
        )
        text_candidates = [text_candidates[i] for i in mmr_idx]

    # =========================
    # Cross-Encoder Rerank (TEXT ONLY, conditional)
    # =========================

    combined_query = (original_query or "").strip()
    if image_context_text and image_context_text.strip():
        combined_query = (combined_query + "\n" + image_context_text).strip()

    # If no user text at all, still rerank with image_context_text as query
    if not combined_query and image_context_text:
        combined_query = image_context_text.strip()

    did_rerank = False
    reranked_text: List[Dict[str, Any]] = []

    should_rerank = (
        bool(combined_query)
        and bool(text_candidates)
        and (best_dense_text_distance is None or best_dense_text_distance > RERANK_IF_DISTANCE_ABOVE)
    )

    if should_rerank:
        reranked_text = reranker.rerank(
            query=combined_query,
            candidates=text_candidates,
            text_key="text",
            top_k=top_k,
        )
        did_rerank = True
    else:
        # Keep dense ordering
        text_candidates.sort(key=lambda x: x["distance"])
        reranked_text = text_candidates[:top_k]
        did_rerank = False

    # Images stay CLIP-ranked
    image_candidates.sort(key=lambda x: x["distance"])
    top_images = image_candidates[:top_k]

    # =========================
    # Retrieval Confidence
    # =========================

    retrieval_distance = reranked_text[0]["distance"] if reranked_text else None
    retrieval_confidence = distance_to_confidence(retrieval_distance)
    top_rerank_score = reranked_text[0].get("rerank_score")

    return {
        "text_results": reranked_text,
        "image_results": top_images,
        "retrieval_distance": retrieval_distance,
        "retrieval_confidence": retrieval_confidence,
        "did_rerank": did_rerank,
        "top_rerank_score": top_rerank_score,
        "best_dense_text_distance": best_dense_text_distance,
    }


# =========================
# Quick Test
# =========================

if __name__ == "__main__":
    out = vector_search(
        original_query="What does crossed arms mean during negotiation?",
        enhanced_queries=["Interpret folded arms in business context"],
        image_path=None,
        image_context_text=None,
        top_k=8,
        overfetch_k=40,
    )

    print("Top text:", len(out["text_results"]))
    print("Top images:", len(out["image_results"]))
    print("retrieval_distance:", out["retrieval_distance"])
    print("retrieval_confidence:", out["retrieval_confidence"])
    print("did_rerank:", out["did_rerank"])