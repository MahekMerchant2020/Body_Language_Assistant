"""
index_builder.py

End-to-end ingestion pipeline for a PDF (LanceDB local):
- Extract structured spans + images
- Preprocess images
- Clean structured text → cleaned_text.txt
- Chunk cleaned text (chapter/subheading/image linking already handled in chunker.py)
- Embed text + images locally using CLIP (text_embedder.py + image_embedder.py)
- Store everything into local LanceDB for multimodal / cross-modal retrieval

Notes:
- LanceDB is embedded (no server). It stores a local folder on disk.
- For CLIP, text + image embeddings share the same vector space, so we store ONE vector column.
- Cross-modal key conventions (IMPORTANT):
  - Text payload key:  metadata.linked_image_ids
  - Image payload key: metadata.linked_text_ids
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Iterable

import lancedb

from ingestion.pdf_loader import PDFLoader
from ingestion.structured_text_cleaner import StructuredTextCleaner
from ingestion.image_preprocessor import preprocess_all_images
from ingestion.chunker import run_chunking
from ingestion.text_embedder import embed_chunks
from ingestion.image_embedder import embed_images

# =========================
# Config
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]  # OutskillGenAI_Capstone/
PDF_FILE = BASE_DIR / "data/raw/raw_pdf/The_Definitive_Book_of_Body_Language.pdf"

TEXT_EMBEDDINGS_FILE = BASE_DIR / "embeddings/text_embeddings/text_embeddings.jsonl"
IMAGE_EMBEDDINGS_FILE = BASE_DIR / "embeddings/image_embeddings/image_embeddings.jsonl"

# LanceDB local folder + table name
LANCEDB_DIR = BASE_DIR / "lancedb_store"
LANCEDB_TABLE = "multimodal_pdf"

# Batch size for table.add()
BATCH_SIZE = 256


# =========================
# Helpers
# =========================

def load_jsonl(file_path: Path) -> List[dict]:
    if not file_path.exists():
        raise FileNotFoundError(f"❌ JSONL file not found: {file_path}")

    data: List[dict] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def normalize_vector(vec: Any) -> List[float]:
    """
    Ensure the embedding vector is a plain Python list[float].
    """
    if vec is None:
        return []
    if isinstance(vec, list):
        return vec
    # torch / numpy fallback
    try:
        return vec.tolist()
    except Exception:
        try:
            return list(vec)
        except Exception:
            return []


def _get_meta_list(meta: dict, key: str) -> List[Any]:
    v = meta.get(key, [])
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def build_rows_for_lancedb(
    text_embeddings: List[dict],
    image_embeddings: List[dict],
) -> List[Dict[str, Any]]:
    """
    Convert JSONL embedding records into LanceDB rows.
    IMPORTANT: Avoid None for string fields so Arrow doesn't infer NullType.
    """
    rows: List[Dict[str, Any]] = []

    def s(x: Any) -> str:
        return "" if x is None else str(x)

    def slist(x: Any) -> List[str]:
        if not x:
            return []
        return [str(v) for v in x]

    # Text rows
    for rec in text_embeddings:
        meta = rec.get("metadata", {}) or {}
        rows.append({
            "id": s(rec.get("id")),
            "modality": "text",
            "vector": normalize_vector(rec.get("vector")),
            "text": s(meta.get("text")),
            "image_path": "",  # <-- never None
            "chapter": s(meta.get("chapter")),
            "subheading": s(meta.get("subheading")),
            "source": s(meta.get("source")),
            "linked_image_ids": slist(meta.get("linked_image_ids")),
            "linked_text_chunk_ids": slist(meta.get("linked_text_chunk_ids") or meta.get("linked_text_ids")),
        })

    # Image rows
    for rec in image_embeddings:
        meta = rec.get("metadata", {}) or {}
        rows.append({
            "id": s(rec.get("id")),
            "modality": "image",
            "vector": normalize_vector(rec.get("vector")),
            "text": "",  # <-- never None
            "image_path": s(meta.get("image_path")),
            "chapter": s(meta.get("chapter")),
            "subheading": s(meta.get("subheading")),
            "source": s(meta.get("source")),
            "linked_image_ids": slist(meta.get("linked_image_ids")),
            "linked_text_chunk_ids": slist(meta.get("linked_text_chunk_ids") or meta.get("linked_text_ids")),
        })

    return rows


def chunked(iterable: List[dict], size: int) -> Iterable[List[dict]]:
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def upsert_lancedb(rows: List[Dict[str, Any]]):
    """
    Writes all rows into LanceDB (overwrite table each run).
    Creates table from the first batch (schema inference),
    then adds remaining batches.
    """
    if not rows:
        raise RuntimeError("❌ No rows provided to LanceDB.")

    LANCEDB_DIR.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(LANCEDB_DIR))

    # Overwrite table each run
    if LANCEDB_TABLE in db.list_tables():   # <-- replace table_names() usage
        db.drop_table(LANCEDB_TABLE)

    batches = list(chunked(rows, BATCH_SIZE))
    first_batch = batches[0]

    # ✅ Create table from first batch so schema is inferred
    table = db.create_table(LANCEDB_TABLE, data=first_batch, mode="overwrite")

    total = len(first_batch)
    print(f"[INFO] LanceDB inserted {total}/{len(rows)} rows")

    # Add remaining batches
    for batch in batches[1:]:
        table.add(batch)
        total += len(batch)
        print(f"[INFO] LanceDB inserted {total}/{len(rows)} rows")

    print(f"[DONE] LanceDB table '{LANCEDB_TABLE}' ready at: {LANCEDB_DIR}")
    

# =========================
# Pipeline
# =========================

def main():
    # Basic file checks
    if not PDF_FILE.exists():
        raise FileNotFoundError(f"❌ PDF not found: {PDF_FILE}")

    print("[STEP 1] Extracting structured text + images from PDF...")
    pdf_loader = PDFLoader(str(PDF_FILE))
    pdf_loader.load()

    print("[STEP 2] Preprocessing images...")
    preprocess_all_images()

    print("[STEP 3] Cleaning structured text...")
    cleaner = StructuredTextCleaner()
    cleaned_path = cleaner.clean()
    print(f"[INFO] Cleaned text saved to {cleaned_path}")

    print("[STEP 4] Chunking cleaned text...")
    run_chunking()

    print("[STEP 5] Embedding text chunks (CLIP local)...")
    embed_chunks()

    print("[STEP 6] Embedding images (CLIP local)...")
    embed_images()

    print("[STEP 7] Loading embeddings JSONL...")
    text_embeddings = load_jsonl(TEXT_EMBEDDINGS_FILE)
    image_embeddings = load_jsonl(IMAGE_EMBEDDINGS_FILE)

    if not text_embeddings and not image_embeddings:
        raise RuntimeError("❌ No embeddings found. Check text_embedder.py / image_embedder.py outputs.")

    print(f"[INFO] Loaded {len(text_embeddings)} text embeddings and {len(image_embeddings)} image embeddings")

    print("[STEP 8] Building LanceDB rows...")
    rows = build_rows_for_lancedb(text_embeddings, image_embeddings)

    # Sanity check: vectors must exist
    rows = [r for r in rows if r.get("vector")]
    if not rows:
        raise RuntimeError("❌ All rows had empty vectors. Check embedding outputs.")

    # Sanity check: vector dims should be consistent
    first_dim = len(rows[0]["vector"])
    bad = [r["id"] for r in rows if len(r["vector"]) != first_dim]
    if bad:
        raise RuntimeError(
            f"❌ Vector dimension mismatch. Expected dim={first_dim} but got mismatches for {len(bad)} rows "
            f"(e.g., {bad[:5]}). Ensure both text and image embedders use the same CLIP model."
        )

    print(f"[INFO] Rows to write into LanceDB: {len(rows)} (vector_dim={first_dim})")

    print("[STEP 9] Writing to LanceDB (batched)...")
    upsert_lancedb(rows)

    print("[DONE] PDF ingestion completed successfully (LanceDB)!")


if __name__ == "__main__":
    main()
