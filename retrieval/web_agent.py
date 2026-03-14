"""
web_agent.py

Agentic web fallback for body-language questions when book retrieval is weak.

Features:
1) Credibility scoring
2) Domain filtering
3) Citation highlighting
4) Unified output schema aligned with answer_generator.py
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field


# =========================
# Config
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=BASE_DIR / ".env")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
WEB_AGENT_MODEL = os.getenv("WEB_AGENT_MODEL", "google/gemini-2.5-flash-lite")

if not TAVILY_API_KEY:
    raise EnvironmentError(
        "❌ TAVILY_API_KEY not found.\n"
        f"Add it to: {BASE_DIR / '.env'}\n"
        "Example:\n"
        "TAVILY_API_KEY=your_key_here"
    )

if not OPENROUTER_API_KEY:
    raise EnvironmentError(
        "❌ OPENROUTER_API_KEY not found.\n"
        f"Add it to: {BASE_DIR / '.env'}\n"
        "Example:\n"
        "OPENROUTER_API_KEY=your_key_here"
    )

TAVILY_SEARCH_URL = os.getenv("TAVILY_SEARCH_URL", "https://api.tavily.com/search")

DEFAULT_MAX_RESULTS = 8
DEFAULT_CONTEXT_CHARS = 7000

DEFAULT_DENY_DOMAINS = {
    "pinterest.com",
    "quora.com",
    "towardsdatascience.com",
    "scribd.com",
    "slideshare.net",
    "coursehero.com",
    "brainly.com",
    "chegg.com",
    "pdfcoffee.com",
    "docplayer.net",
    "tiktok.com",
    "instagram.com",
}


# =========================
# Output Schema
# =========================

class AnswerOutput(BaseModel):
    answer: str = Field(..., description="Direct helpful answer to the user.")
    retrieval_confidence: int = Field(..., ge=0, le=100, description="0-100 retrieval confidence.")
    reasoning: str = Field(..., description="Brief justification tied to sources.")
    source: str = Field(..., description="Attribution string built from actual sources.")
    answerability_confidence: int = Field(..., ge=0, le=100, description="0-100 answerability confidence.")
    needs_web_fallback: bool = Field(..., description="Always false for final web answers.")


# =========================
# Internal Models
# =========================

@dataclass
class WebSource:
    cid: str
    title: str
    url: str
    domain: str
    snippet: str
    tavily_score: float
    credibility: float
    blended_score: float


# =========================
# Utilities
# =========================

_CIT_RE = re.compile(r"\[C(\d+)\]")
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _safe_json_parse(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    t = _JSON_FENCE_RE.sub("", t).strip()

    if "```" in t:
        parts = [p.strip() for p in t.split("```") if p.strip()]
        candidates = [p for p in parts if p.lstrip().startswith("{") and p.rstrip().endswith("}")]
        t = max(candidates, key=len) if candidates else max(parts, key=len)

    return json.loads(t)


def _fallback_wrap_non_json(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()

    if not text:
        text = "I couldn’t generate a reliable web answer for this question."

    return {
        "answer": text,
        "retrieval_confidence": 30,
        "reasoning": "The web model did not return valid JSON, so the raw response was wrapped as a fallback answer.",
        "source": "",
        "answerability_confidence": 30,
        "needs_web_fallback": False,
    }


def _coerce_confidence(val: Any) -> int:
    if val is None:
        return 50

    if isinstance(val, int):
        return max(0, min(100, val))

    if isinstance(val, float):
        if 0.0 <= val <= 1.0:
            return max(0, min(100, int(round(val * 100))))
        return max(0, min(100, int(round(val))))

    if isinstance(val, str):
        s = val.strip().replace(" ", "")
        if s.endswith("%"):
            num = s[:-1]
            try:
                f = float(num)
            except Exception:
                return 50
            return max(0, min(100, int(round(f))))
        try:
            f = float(s)
            if 0.0 <= f <= 1.0:
                return max(0, min(100, int(round(f * 100))))
            return max(0, min(100, int(round(f))))
        except Exception:
            return 50

    return 50


def _extract_domain(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    u = url.split("//", 1)[-1]
    domain = u.split("/", 1)[0].lower()
    domain = domain.split(":", 1)[0]
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _domain_credibility(domain: str) -> float:
    d = (domain or "").lower()
    if not d:
        return 0.3

    if d.endswith(".gov") or d.endswith(".mil"):
        return 0.98
    if d.endswith(".edu") or d.endswith(".ac.uk") or ".edu." in d or ".ac." in d:
        return 0.92

    strong = (
        "nature.com",
        "sciencemag.org",
        "cell.com",
        "nejm.org",
        "jamanetwork.com",
        "bmj.com",
        "thelancet.com",
        "springer.com",
        "sciencedirect.com",
        "elsevier.com",
        "tandfonline.com",
        "wiley.com",
        "onlinelibrary.wiley.com",
        "sagepub.com",
        "cambridge.org",
        "oup.com",
        "academic.oup.com",
        "ieee.org",
        "acm.org",
        "dl.acm.org",
        "arxiv.org",
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "cochranelibrary.com",
        "psycnet.apa.org",
        "apa.org",
    )
    if any(s in d for s in strong):
        if "arxiv.org" in d:
            return 0.82
        return 0.90

    reputable = (
        "bbc.co.uk",
        "reuters.com",
        "apnews.com",
        "theguardian.com",
        "nytimes.com",
        "washingtonpost.com",
        "nationalgeographic.com",
        "britannica.com",
    )
    if any(s in d for s in reputable):
        return 0.78

    if d.endswith(".org"):
        return 0.70

    return 0.55


def _passes_domain_filters(domain: str, allow_domains: Optional[List[str]], deny_domains: Optional[List[str]]) -> bool:
    d = (domain or "").lower()
    if not d:
        return False

    deny = set(x.lower() for x in (deny_domains or []))
    if d in deny:
        return False
    if any(d.endswith("." + dd) for dd in deny):
        return False

    if allow_domains:
        allow = [x.lower() for x in allow_domains]
        return (d in allow) or any(d.endswith("." + a) for a in allow)

    return True


def _blend_scores(relevance: float, credibility: float, w_rel: float = 0.65, w_cred: float = 0.35) -> float:
    r = max(0.0, min(1.0, relevance))
    c = max(0.0, min(1.0, credibility))
    return w_rel * r + w_cred * c


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _build_web_context(sources: List[WebSource], max_chars: int = DEFAULT_CONTEXT_CHARS) -> str:
    parts: List[str] = []
    total = 0

    for src in sources:
        block = (
            f"[{src.cid}] TITLE: {src.title}\n"
            f"DOMAIN: {src.domain}\n"
            f"URL: {src.url}\n"
            f"SNIPPET: {src.snippet}\n"
        )
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block) + 1

    return "\n".join(parts).strip()


def _sources_to_string(used_sources: List[WebSource], max_items: int = 4) -> str:
    items: List[str] = []
    for s in used_sources[:max_items]:
        title = s.title or "Untitled"
        dom = s.domain or _extract_domain(s.url)
        if dom:
            items.append(f"Web — {title} ({dom})")
        else:
            items.append(f"Web — {title}")
    return "; ".join(items) if items else "Web"


def _find_used_cids(answer: str, reasoning: str) -> List[str]:
    found = set()
    for txt in (answer or "", reasoning or ""):
        for m in _CIT_RE.finditer(txt):
            found.add(f"C{m.group(1)}")

    def _cid_key(x: str) -> int:
        try:
            return int(x[1:])
        except Exception:
            return 9999

    return sorted(found, key=_cid_key)


# =========================
# Tavily Search
# =========================

def tavily_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[Dict[str, Any]]:
    query = (query or "").strip()[:400]

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        resp = requests.post(TAVILY_SEARCH_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", []) or []
    except requests.exceptions.HTTPError as e:
        print("⚠️ Tavily HTTP error:", e)
        return []
    except Exception as e:
        print("⚠️ Tavily request failed:", e)
        return []


def build_ranked_sources(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    allow_domains: Optional[List[str]] = None,
    deny_domains: Optional[List[str]] = None,
    min_credibility: float = 0.55,
) -> List[WebSource]:
    deny = set(DEFAULT_DENY_DOMAINS)
    if deny_domains:
        deny.update(x.lower() for x in deny_domains)

    raw = tavily_search(query, max_results=max_results)
    if not raw:
        return []

    sources: List[WebSource] = []
    for r in raw:
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        snippet = (r.get("content") or r.get("snippet") or "").strip()
        domain = _extract_domain(url)

        if not _passes_domain_filters(domain, allow_domains, list(deny)):
            continue

        cred = _domain_credibility(domain)
        if cred < min_credibility:
            continue

        rel = r.get("score", 0.7)
        try:
            rel = float(rel)
        except Exception:
            rel = 0.7
        if rel > 1.0:
            rel = 1.0 / (1.0 + rel)

        blended = _blend_scores(rel, cred, w_rel=0.55, w_cred=0.45)

        sources.append(
            WebSource(
                cid=f"C{len(sources) + 1}",
                title=_truncate(title, 160) if title else "Untitled",
                url=url,
                domain=domain,
                snippet=_truncate(snippet, 800) if snippet else "",
                tavily_score=rel,
                credibility=cred,
                blended_score=blended,
            )
        )

    sources.sort(key=lambda s: s.blended_score, reverse=True)

    for i, s in enumerate(sources, start=1):
        s.cid = f"C{i}"

    return sources


# =========================
# OpenRouter LLM Call
# =========================

def openrouter_chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    url = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": WEB_AGENT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""


# =========================
# Main Web Agent
# =========================

def web_fallback_answer(
    user_query: str,
    enhanced_queries: Optional[List[str]] = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    allow_domains: Optional[List[str]] = None,
    deny_domains: Optional[List[str]] = None,
    min_credibility: float = 0.65,
) -> AnswerOutput:
    query_pool = [user_query] + (enhanced_queries or [])
    query_pool = [q.strip()[:200] for q in query_pool if q and q.strip()]

    unique_queries = []
    seen_tokens = []

    for q in query_pool:
        tokens = set(q.lower().split())
        if not any(len(tokens & s) / max(len(tokens), 1) > 0.7 for s in seen_tokens):
            unique_queries.append(q)
            seen_tokens.append(tokens)

    query_pool = unique_queries[:3]

    by_url: Dict[str, WebSource] = {}
    for q in query_pool:
        sources = build_ranked_sources(
            q,
            max_results=max_results,
            allow_domains=allow_domains,
            deny_domains=deny_domains,
            min_credibility=min_credibility,
        )
        for s in sources:
            if not s.url:
                continue
            if s.url not in by_url or s.blended_score > by_url[s.url].blended_score:
                by_url[s.url] = s

    merged = sorted(by_url.values(), key=lambda s: s.blended_score, reverse=True)[:max_results]

    if not merged:
        return AnswerOutput(
            answer="I couldn’t find reliable web sources for this question right now.",
            retrieval_confidence=25,
            reasoning="Web search returned no high-credibility sources after filtering.",
            source="Web",
            answerability_confidence=25,
            needs_web_fallback=False,
        )

    web_context = _build_web_context(merged)

    schema_hint = {
        "answer": "string (MUST include citations like [C1] inline)",
        "confidence": "integer 0-100",
        "reasoning": "string (brief; MUST include citations like [C1])",
        "source": "string (you may leave blank; system will override)",
    }

    system_msg = {
        "role": "system",
        "content": (
            "You are a careful research assistant.\n"
            "Answer the user's question using ONLY the WEB CONTEXT.\n"
            "Rules:\n"
            "1) DO NOT invent facts not supported by WEB CONTEXT.\n"
            "2) You MUST cite evidence inline using the bracket format: [C1], [C2], etc.\n"
            "3) Put citations in BOTH the answer and reasoning where relevant.\n"
            "4) Output MUST be valid JSON only. No markdown. No ``` fences.\n"
            "5) If the evidence is weak or incomplete, still return valid JSON with a cautious answer.\n"
            f"Return JSON with this exact shape: {json.dumps(schema_hint)}\n"
        ),
    }

    human_msg = {
        "role": "user",
        "content": (
            f"USER QUESTION:\n{user_query}\n\n"
            f"WEB CONTEXT:\n{web_context}\n"
        ),
    }

    raw = openrouter_chat([system_msg, human_msg], temperature=0.0)

    try:
        data = _safe_json_parse(raw)
    except Exception:
        data = _fallback_wrap_non_json(raw)

    web_conf = _coerce_confidence(data.get("confidence"))

    # If fallback wrapper already produced unified schema, keep it.
    if "retrieval_confidence" in data and "answerability_confidence" in data:
        out = AnswerOutput(**data)
    else:
        try:
            out = AnswerOutput(
                answer=(data.get("answer") or "").strip(),
                retrieval_confidence=web_conf,
                reasoning=(data.get("reasoning") or "").strip(),
                source="",
                answerability_confidence=web_conf,
                needs_web_fallback=False,
            )
        except Exception as e:
            raise ValueError(
                f"❌ Web agent output failed schema validation.\nRaw:\n{raw}\n\nParsed:\n{data}\n\nError: {e}"
            )

    used_cids = _find_used_cids(out.answer, out.reasoning)

    if not used_cids:
        out.retrieval_confidence = min(out.retrieval_confidence, 40)
        out.answerability_confidence = min(out.answerability_confidence, 40)
        out.reasoning = (out.reasoning or "").strip() + " (No inline citations were provided.)"
        out.answer = (out.answer or "").strip() + f" [{merged[0].cid}]"
        used_cids = [merged[0].cid]

    cid_to_src = {s.cid: s for s in merged}
    valid_cids = [c for c in used_cids if c in cid_to_src]

    if not valid_cids:
        valid_cids = [merged[0].cid]

    used_sources = [cid_to_src[c] for c in valid_cids]

    out.source = _sources_to_string(used_sources)

    if merged:
        top_blended = merged[0].blended_score
        if top_blended < 0.60:
            out.retrieval_confidence = min(out.retrieval_confidence, 65)
            out.answerability_confidence = min(out.answerability_confidence, 65)

    if len(used_sources) == 1 and used_sources[0].credibility < 0.75:
        out.retrieval_confidence = min(out.retrieval_confidence, 75)
        out.answerability_confidence = min(out.answerability_confidence, 75)

    if len(used_sources) < 2 and out.retrieval_confidence > 85:
        out.retrieval_confidence = 80
        out.answerability_confidence = 80

    return out


# =========================
# Quick test
# =========================

if __name__ == "__main__":
    q = "Latest research on microexpressions accuracy?"
    enhanced = [
        "recent studies on micro-expression recognition accuracy",
        "microexpression detection model accuracy CASME II",
    ]
    result = web_fallback_answer(
        user_query=q,
        enhanced_queries=enhanced,
        max_results=8,
        allow_domains=None,
        deny_domains=["reddit.com"],
        min_credibility=0.55,
    )
    print(result.model_dump())