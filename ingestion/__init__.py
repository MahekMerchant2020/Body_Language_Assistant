"""
Ingestion package for the Body Language Assistant.

Responsible for building the multimodal knowledge base from
The Definitive Book of Body Language.

Components typically include:
- document loading
- image extraction
- text + image chunking
- embedding generation
- LanceDB indexing
"""

__all__ = [
    "pdf_loader",
    "structured_text_cleaner",
    "text_embedder",
    "image_preprocessor",
    "image_embedder"
    "chunker",
    "index_builder",
]