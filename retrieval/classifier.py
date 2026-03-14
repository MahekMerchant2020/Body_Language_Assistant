# retrieval/classifier.py
"""
classifier.py

LangChain-based relevance + routing classifier.

Routes user query to:
- "book"   -> LanceDB multimodal RAG (body language + confidence high)
- "web"    -> Web fallback (body language but book likely insufficient / general)
- "refuse" -> Not body-language related / out of scope

Also returns:
- chapter_hints -> ordered list of up to 2 exact chapter names from the book

Uses:
- LangChain Chat model (OpenRouter recommended)
- PydanticOutputParser
- Extra sanitation fallback for markdown-fenced JSON

ENV (in BASE_DIR/.env):
- OPENROUTER_API_KEY=...
Optional:
- OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
- CLASSIFIER_MODEL=google/gemini-2.5-flash-lite
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


# =========================
# Base directory + env
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=BASE_DIR / ".env")

TAXONOMY_PATH = BASE_DIR / "data/processed/processed_text/book_taxonomy.json"


def load_book_taxonomy_text() -> str:
    if not TAXONOMY_PATH.exists():
        return ""

    try:
        with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
            taxonomy = json.load(f)
    except Exception:
        return ""

    lines = []
    for chapter, subheadings in taxonomy.items():
        lines.append(chapter)
        for sub in subheadings:
            lines.append(f"  - {sub}")

    return "\n".join(lines).strip()


def load_book_chapters() -> List[str]:
    if not TAXONOMY_PATH.exists():
        return []

    try:
        with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
            taxonomy = json.load(f)
    except Exception:
        return []

    return list(taxonomy.keys())


BOOK_TAXONOMY_TEXT = load_book_taxonomy_text()
BOOK_CHAPTERS = load_book_chapters()
BOOK_CHAPTERS_TEXT = "\n".join(BOOK_CHAPTERS)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "google/gemini-2.5-flash-lite")

if not OPENROUTER_API_KEY:
    raise EnvironmentError(
        "❌ OPENROUTER_API_KEY not found.\n"
        f"Add it to: {BASE_DIR / '.env'}\n"
        "Example:\n"
        "OPENROUTER_API_KEY=your_key_here"
    )


# =========================
# Output schema
# =========================

Route = Literal["book", "web", "refuse"]


class ClassificationOutput(BaseModel):
    route: Route = Field(
        ...,
        description="Routing decision: 'book', 'web', or 'refuse'.",
    )
    reason: str = Field(
        ...,
        description="Short explanation of why this route was chosen.",
    )
    chapter_hints: List[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of up to 2 exact chapter names from BOOK_CHAPTERS_TEXT "
            "that are most relevant if route='book'. Otherwise empty list."
        ),
    )


# =========================
# LLM + prompt + parser
# =========================

parser = PydanticOutputParser(pydantic_object=ClassificationOutput)

SYSTEM_RULES = """
You are a strict router for a body-language-only assistant.

The assistant answers questions using a single body language book.

Book topics include:

{BOOK_TAXONOMY_TEXT}

Valid chapter names for chapter_hints are ONLY these chapter titles:

{BOOK_CHAPTERS_TEXT}

Routing rules:

route="book"
Use when the query is about body language and likely answerable from the book.

route="web"
Use when the query is about body language but likely requires:
- scientific research
- latest trends on social media
- statistics
- modern psychology studies
- medical explanations
- content not typically in a body language book

route="refuse"
Use when the query is unrelated to body language.

Additional rule for chapter_hints:
- If route="book", return up to 2 items.
- Every item in chapter_hints must be chosen ONLY from the valid chapter titles above.
- Do NOT return subheadings.
- If no chapter is clear, return [].
- If route is "web" or "refuse", return [].

Return ONLY valid JSON.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_RULES),
        (
            "human",
            "User query: {query}\n"
            "Has image input: {has_image}\n\n"
            "{format_instructions}",
        ),
    ]
)

llm = ChatOpenAI(
    model=CLASSIFIER_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    temperature=0,
)


# =========================
# Robust parsing helpers
# =========================

def _strip_markdown_fences(text: str) -> str:
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text.strip()


def _extract_first_json_object(text: str) -> str:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else text


def _validate_chapter_hints(out: ClassificationOutput) -> ClassificationOutput:
    valid_chapters = set(BOOK_CHAPTERS)
    cleaned = [c for c in out.chapter_hints if c in valid_chapters]

    # web/refuse should never carry chapter hints
    if out.route in {"web", "refuse"}:
        cleaned = []

    out.chapter_hints = cleaned[:2]
    return out


def _parse_with_fallback(raw: str) -> ClassificationOutput:
    try:
        out = parser.parse(raw)
    except Exception:
        cleaned = _extract_first_json_object(_strip_markdown_fences(raw))
        data = json.loads(cleaned)
        out = ClassificationOutput(**data)

    return _validate_chapter_hints(out)


# =========================
# Public API
# =========================

def classify_query(query: str, has_image: bool = False) -> ClassificationOutput:
    """
    Classify a user query into {book, web, refuse}.
    """
    chain = prompt | llm
    resp = chain.invoke(
        {
            "query": query,
            "has_image": str(bool(has_image)),
            "BOOK_TAXONOMY_TEXT": BOOK_TAXONOMY_TEXT,
            "BOOK_CHAPTERS_TEXT": BOOK_CHAPTERS_TEXT,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    content = getattr(resp, "content", "") or ""
    if not content.strip():
        return ClassificationOutput(
            route="refuse",
            reason="Empty classifier output; defaulting to refuse for safety.",
            chapter_hints=[],
        )

    try:
        return _parse_with_fallback(content)
    except Exception as e:
        raise ValueError(
            f"❌ Failed to parse classifier output:\n{content}\n\nError: {e}"
        ) from e


# =========================
# CLI test
# =========================

if __name__ == "__main__":
    tests = [
        ("What does crossed arms mean during negotiation?", False),
        ("Explain Newton's laws of motion.", False),
        ("Can you analyze this person's posture in the image?", True),
        ("Latest research on microexpressions accuracy?", False),
    ]

    for q, img in tests:
        out = classify_query(q, has_image=img)
        print("\nQ:", q)
        print(out.model_dump())