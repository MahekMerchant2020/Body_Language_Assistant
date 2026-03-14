"""
Retrieval package for the Body Language Assistant.

This package contains all components responsible for:
- query classification
- hybrid vector + metadata retrieval
- web fallback search
- image interpretation
- answer generation
- orchestration of the RAG pipeline
"""

__all__ = [
    "classifier",
    "query_enhancer",
    "reranker"
    "hybrid_search",
    "vector_search",
    "answer_generator",
    "web_agent",
    "image_interpreter",
    "taxonomy_router",
    "orchestrator",
]