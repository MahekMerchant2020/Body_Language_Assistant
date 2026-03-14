"""
query_enhancer.py

Generates enhanced query variants for:

- LanceDB vector search (RAG fusion)
- Tavily web search
- Keyword reinforcement

Uses:
- Gemini 2.5 Flash Lite (via OpenRouter)
- LangChain
- Pydantic structured output

Outputs:
{
    original: str,
    enhanced_queries: List[str],
    keywords: List[str]
}
"""

from pathlib import Path
from typing import List
import os
import json

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from retrieval.taxonomy_router import TaxonomyRouter

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


# =========================
# Configuration
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]  # OutskillGenAI_Capstone/
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

BOOK_TAXONOMY_TEXT = load_book_taxonomy_text()
taxonomy_router = TaxonomyRouter()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
ENHANCER_MODEL = os.getenv("ENHANCER_MODEL", "google/gemini-2.5-flash-lite")

if not OPENROUTER_API_KEY:
    raise EnvironmentError(
        "❌ OPENROUTER_API_KEY not found.\n"
        f"Add it to: {BASE_DIR / '.env'}\n"
        "Example:\n"
        "OPENROUTER_API_KEY=your_key_here"
    )


# =========================
# Output Schema
# =========================

class QueryEnhancementOutput(BaseModel):
    original: str = Field(description="Original user query.")
    enhanced_queries: List[str] = Field(
        description="2–4 semantically enriched query variants."
    )
    keywords: List[str] = Field(
        description="Strong body-language related keywords."
    )


parser = PydanticOutputParser(pydantic_object=QueryEnhancementOutput)


# =========================
# LLM Setup
# =========================

llm = ChatOpenAI(
    model=ENHANCER_MODEL,
    temperature=0.2,  # slight creativity but controlled
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a query enhancer for a body language RAG system.

The knowledge base is a book organized into chapters and subtopics.

Book structure:
{BOOK_TAXONOMY_TEXT}

Your goals:
1. Rewrite the query using chapter terminology likely used in the book if applicable.
2. Expand the query for semantic retrieval.
3. Generate 2–4 alternative queries.
4. Extract strong keywords (3–8).
5. If the query is general (e.g., first impression), generate queries using broader body language concepts like posture, eye contact, handshake, confidence signals.
6. Prefer chapter terminology when relevant.

Example:
"impress my girlfriend"
→ "courtship body language signals"
→ "female attraction cues"

Rules:
- Do NOT answer the question.
- Do NOT invent facts.
- Keep meaning faithful.

Return JSON only.
"""
        ),
        (
            "user",
            """
User Query:
{query}

{format_instructions}
"""
        ),
    ]
)

chain = prompt | llm | parser

# =========================
# Main Function
# =========================

def enhance_query(query: str) -> QueryEnhancementOutput:
    """
    Enhance user query for improved vector search + web search.
    """

    result = chain.invoke(
        {
            "query": query,
            "BOOK_TAXONOMY_TEXT": BOOK_TAXONOMY_TEXT,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    return result


# =========================
# Test Block
# =========================

if __name__ == "__main__":

    test_queries = [
        "What does crossed arms mean during negotiation?",
        "Analyze this person's posture in the image.",
        "Latest research on microexpressions accuracy."
    ]

    for q in test_queries:
        print("\nQ:", q)
        out = enhance_query(q)
        print(out.model_dump())