"""
answer_generator.py

Generates final answers using an LLM (OpenRouter -> Gemini 2.5 Flash Lite),
grounded in retrieved context.

Separates:
- retrieval_confidence (from retrieval stage)
- answerability_confidence (LLM answer quality)

Signals when web fallback is needed.
"""

from __future__ import annotations

import os
import json
import re
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# =========================
# Config
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "google/gemini-2.5-flash-lite")

if not OPENROUTER_API_KEY:
    raise EnvironmentError(
        f"❌ OPENROUTER_API_KEY not found in {BASE_DIR / '.env'}"
    )

_CHUNK_ID_RE = re.compile(r"chunk_(\d+)", re.IGNORECASE)


# =========================
# Output Schema
# =========================

class AnswerOutput(BaseModel):
    answer: str
    retrieval_confidence: int
    reasoning: str
    source: str
    answerability_confidence: int
    needs_web_fallback: bool


# =========================
# Utilities
# =========================

def _safe_json_parse(text: str) -> Dict[str, Any]:

    t = (text or "").strip()

    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?", "", t, flags=re.IGNORECASE).strip("`").strip()

    if "```" in t:
        parts = t.split("```")
        t = max(parts, key=len).strip()

    return json.loads(t)


def _normalize_confidence(value: Any) -> int:

    try:

        if value is None:
            return 50

        if isinstance(value, int):
            return max(0, min(100, value))

        if isinstance(value, float):

            if math.isnan(value) or math.isinf(value):
                return 50

            if 0 <= value <= 1:
                return int(round(value * 100))

            return int(round(value))

        if isinstance(value, str):

            s = value.strip().replace("%", "")
            f = float(s)

            if math.isnan(f) or math.isinf(f):
                return 50

            if 0 <= f <= 1:
                return int(round(f * 100))

            return int(round(f))

    except Exception:
        return 50

    return 50


def _stabilize_answerability(raw: int) -> int:
    """
    LLMs often exaggerate confidence.
    This calibration makes the score more realistic.
    """

    if raw >= 90:
        return 85

    if raw >= 80:
        return 80

    if raw >= 70:
        return 70

    if raw >= 60:
        return 60

    if raw >= 50:
        return 50

    return raw


def _format_book_context(chunks: List[Dict[str, Any]], max_chars: int = 7000) -> str:

    parts = []
    total = 0

    for c in chunks:

        chapter = c.get("chapter") or "Unknown chapter"
        subheading = c.get("subheading") or "No subheading"
        cid = c.get("id") or "unknown_chunk"
        distance = c.get("distance") or c.get("score")

        header = f"[CONTEXT] chunk_id={cid} | chapter={chapter} | subheading={subheading}"

        if distance is not None:
            try:
                header += f" | distance={float(distance):.4f}"
            except Exception:
                pass

        body = (c.get("text") or "").strip()

        if not body:
            continue

        block = header + "\n" + body

        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n\n".join(parts)


def _source_from_book(chunks: List[Dict[str, Any]], max_items: int = 3) -> str:

    seen = set()
    sources = []

    for c in chunks:

        chapter = (c.get("chapter") or "").strip()
        subheading = (c.get("subheading") or "").strip()

        key = (chapter, subheading)

        if key in seen:
            continue

        seen.add(key)

        if chapter and subheading:
            sources.append(f"Book — Chapter: {chapter}, Subheading: {subheading}")

        elif chapter:
            sources.append(f"Book — Chapter: {chapter}")

        if len(sources) >= max_items:
            break

    return "; ".join(sources) if sources else "Book — The Definitive Book of Body Language"


def _extract_used_chunk_ids(reasoning: str) -> List[str]:

    if not reasoning:
        return []

    matches = _CHUNK_ID_RE.findall(reasoning)

    return [f"chunk_{m}" for m in matches]


# =========================
# Main Generator
# =========================

def generate_answer(
    user_query: str,
    book_chunks: List[Dict[str, Any]],
    retrieval_confidence: int,
    web_snippets: Optional[List[Dict[str, Any]]] = None,
    image_context_text: Optional[str] = None,
    mode: Literal["book_only", "web_only", "hybrid"] = "book_only",
) -> AnswerOutput:

    web_snippets = web_snippets or []

    book_context = _format_book_context(book_chunks)

    schema_hint = {
        "answer": "non-empty string",
        "answerability_confidence": "integer 0-100",
        "reasoning": "non-empty string",
    }

    system = SystemMessage(
        content=(
            "You are a body language analysis assistant.\n\n"
            "You may receive:\n"
            "- Image interpretation (descriptions of posture and gestures)\n"
            "- Book context explaining body language meanings\n\n"
            "Your task:\n"
            "1) Use the IMAGE INTERPRETATION to identify visible body language cues.\n"
            "2) Use the BOOK CONTEXT to interpret what those cues typically mean.\n"
            "3) Combine both sources to produce a clear explanation.\n\n"
            "Important rules:\n"
            "- Do NOT say you cannot see the image.\n"
            "- The image has already been interpreted into text.\n"
            "- Do not repeat the interpretation verbatim — explain what the cues suggest.\n"
            "- Prefer psychological meaning over simple description.\n"
            "- If the book context provides a useful partial answer, do not assign an extremely low answerability score.\n"
            "- Do not invent citations.\n\n"
            "Output valid JSON only. The answer and reasoning must not be empty.\n"
        )
    )

    human = HumanMessage(
        content=(
            f"USER QUESTION:\n{user_query}\n\n"
            f"IMAGE INTERPRETATION:\n{image_context_text or '(none)'}\n\n"
            f"BOOK CONTEXT:\n{book_context or '(none)'}"
        )
    )

    llm = ChatOpenAI(
        model=ANSWER_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.0,
    )

    response = llm.invoke([system, human])

    raw = (response.content or "").strip()

    try:

        data = _safe_json_parse(raw)

        answerability_raw = _normalize_confidence(
            data.get("answerability_confidence")
        )

        answerability = _stabilize_answerability(answerability_raw)

        out = AnswerOutput(
            answer=data.get("answer", ""),
            reasoning=data.get("reasoning", ""),
            source="",
            retrieval_confidence=_normalize_confidence(retrieval_confidence),
            answerability_confidence=answerability,
            needs_web_fallback=False,
        )

        # Guard against blank model output
        if not out.answer.strip():
            out.answer = (
                "I could not generate a grounded answer from the available context."
            )
            out.answerability_confidence = min(out.answerability_confidence, 25)
            out.needs_web_fallback = True

        if not out.reasoning.strip():
            out.reasoning = (
                "The generated response was empty, so the answer was treated as insufficient."
            )

    except Exception as e:

        raise ValueError(
            f"❌ Failed to parse AnswerOutput JSON.\nRaw:\n{raw}\n\nError: {e}"
        )

    # =========================
    # Retrieval Calibration
    # =========================

    if book_chunks:

        top = book_chunks[0]

        try:

            top_distance = float(
                top.get("distance") if top.get("distance") is not None
                else top.get("score", 1.0)
            )

        except Exception:
            top_distance = 1.0

        if top_distance > 0.40:
            out.answerability_confidence = min(out.answerability_confidence, 50)

        if top_distance > 0.55:
            out.answerability_confidence = min(out.answerability_confidence, 30)

    # If image interpretation exists, treat it as valid grounding
    if image_context_text and image_context_text.strip():
        out.answerability_confidence = max(out.answerability_confidence, 60)
        
    # =========================
    # Source Grounding
    # =========================

    used_chunk_ids = _extract_used_chunk_ids(out.reasoning)

    if used_chunk_ids:

        grounded_chunks = [
            c for c in book_chunks
            if c.get("id") in used_chunk_ids
        ]

        if grounded_chunks:
            out.source = _source_from_book(grounded_chunks)

        else:
            out.source = _source_from_book(book_chunks[:1])

    else:
        out.source = _source_from_book(book_chunks[:1])

    # =========================
    # Web Fallback Decision
    # =========================

    needs_web = False

    if mode == "book_only":

        if out.retrieval_confidence < 40:
            needs_web = True

        if out.answerability_confidence < 30:
            needs_web = True

    out.needs_web_fallback = needs_web

    return out