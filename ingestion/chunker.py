"""
chunker.py

Production-grade structure-aware, token-based recursive chunking.

Hierarchy:
Chapter → Subheading → Paragraph → Token-recursive split

Rules:
1. Never link images across chapter boundaries.
2. If no subheading exists, link at chapter level.
3. Treat chapter as strongest semantic container.
4. Reset alignment state on new chapter.
"""

import json
import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# Configuration
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_TEXT_DIR = BASE_DIR / "data" / "processed" / "processed_text"
CHUNKS_DIR = BASE_DIR / "data" / "processed" / "chunks"

MAX_CHUNK_TOKENS = 400
CHUNK_OVERLAP = 50

IMAGE_REF_PATTERN = re.compile(r"\[IMAGE_REF: (.*?)\]")
CHAPTER_NUMBER_PATTERN = re.compile(r"^Chapter\s+\d+", re.IGNORECASE)

# ✅ TRUE token-based recursive splitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=MAX_CHUNK_TOKENS,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        " ",
        ""
    ]
)

def is_noise_heading(text: str) -> bool:
    """
    Detect OCR / PDF noise headings like 'I', 'J', etc.
    """
    t = text.strip()

    if len(t) <= 1:
        return True

    # single roman numerals often appear as artifacts
    if t in {"I", "V", "X", "J"}:
        return True

    return False

# =========================
# Core Chunking
# =========================

def chunk_from_cleaned_text(text: str):

    lines = text.split("\n")

    chunks = []
    chunk_counter = 0

    current_chapter = None
    current_subheading = None
    paragraph_buffer = []

    # image alignment state (chapter-scoped only)
    pending_image_id = None

    def flush_paragraph():
        nonlocal chunk_counter, paragraph_buffer, pending_image_id

        if not paragraph_buffer:
            return

        paragraph_text = " ".join(paragraph_buffer).strip()
        paragraph_buffer = []

        if not paragraph_text:
            return

        split_chunks = text_splitter.split_text(paragraph_text)

        for chunk_text in split_chunks:

            chunk_text = chunk_text.strip()

            # ✅ Clean leading punctuation artifacts
            chunk_text = re.sub(r"^[\.\,\;\:\s]+", "", chunk_text)

            if not chunk_text:
                continue

            image_ids = []

            # Attach pending image (forward linking)
            if pending_image_id:
                image_ids.append(pending_image_id)
                pending_image_id = None  # reset after attaching

            chunk_counter += 1
            chunks.append({
                "chunk_id": f"chunk_{chunk_counter:05d}",
                "chunk_index": chunk_counter,
                "type": "text",
                "text": chunk_text,
                "metadata": {
                    "chapter": current_chapter,
                    "subheading": current_subheading,
                    "image_ids": image_ids,
                    "source": "cleaned_text.txt"
                }
            })

    for line in lines:
        stripped = line.strip()

        # Empty line → paragraph boundary
        if not stripped:
            flush_paragraph()
            continue

        # Skip page markers
        if stripped.startswith("--- Page"):
            flush_paragraph()
            continue

        # =========================
        # IMAGE REFERENCE
        # =========================
        if stripped.startswith("[IMAGE_REF:"):
            flush_paragraph()

            image_id_match = IMAGE_REF_PATTERN.findall(stripped)
            image_id = None

            if image_id_match:
                image_id = (
                    image_id_match[0]
                    .replace(".jpeg", "")
                    .replace(".jpg", "")
                    .replace(".png", "")
                )

            # 🔒 Never link across chapters
            if current_chapter is None:
                # Chapter not yet set → standalone image
                image_metadata_chapter = None
            else:
                image_metadata_chapter = current_chapter

            # Attach image to previous text chunk (same chapter only)
            if (
                image_id
                and chunks
                and chunks[-1]["type"] == "text"
                and chunks[-1]["metadata"]["chapter"] == current_chapter
            ):
                chunks[-1]["metadata"]["image_ids"].append(image_id)

            # Store for forward linking (same chapter only)
            if image_id and current_chapter:
                pending_image_id = image_id
            else:
                pending_image_id = None

            chunk_counter += 1
            chunks.append({
                "chunk_id": f"chunk_{chunk_counter:05d}",
                "chunk_index": chunk_counter,
                "type": "image",
                "text": stripped,
                "metadata": {
                    "chapter": image_metadata_chapter,
                    "subheading": current_subheading,
                    "image_ids": [image_id] if image_id else [],
                    "source": "cleaned_text.txt"
                }
            })

            continue

        # =========================
        # CHAPTER NUMBER
        # =========================
        if CHAPTER_NUMBER_PATTERN.match(stripped):
            flush_paragraph()

            # 🔒 Reset all alignment state (Rule 4)
            current_chapter = None
            current_subheading = None
            pending_image_id = None

            continue

        # =========================
        # CHAPTER TITLE
        # =========================
        if stripped.startswith("<CHAPTER_TITLE>"):
            flush_paragraph()

            candidate = stripped.replace("<CHAPTER_TITLE>", "").strip()

            # Ignore noise chapters like "I", "J"
            if is_noise_heading(candidate):
                continue

            # Chapter is strongest container
            current_chapter = candidate
            current_subheading = None
            pending_image_id = None  # 🔒 reset alignment

            continue

        # =========================
        # SUBHEADING
        # =========================
        if stripped.startswith("<SUBHEADING>"):
            flush_paragraph()

            candidate = stripped.replace("<SUBHEADING>", "").strip()

            # Ignore noise subheadings
            if is_noise_heading(candidate):
                continue

            current_subheading = candidate

            continue

        # Otherwise → accumulate paragraph text
        paragraph_buffer.append(stripped)

    # Flush last paragraph
    flush_paragraph()

    return chunks


def save_book_taxonomy(chunks, output_dir):

    taxonomy = {}

    for c in chunks:

        if c.get("type") != "text":
            continue

        meta = c.get("metadata", {}) or {}

        chapter = (meta.get("chapter") or "").strip()
        subheading = (meta.get("subheading") or "").strip()

        # Skip noise chapters
        if len(chapter) <= 1:
            continue

        # Clean noise subheadings
        if len(subheading) <= 1:
            subheading = ""

        if chapter not in taxonomy:
            taxonomy[chapter] = []

        if subheading and subheading not in taxonomy[chapter]:
            taxonomy[chapter].append(subheading)

    out_path = Path(output_dir) / "book_taxonomy.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    print(f"Saved taxonomy to {out_path}")


# =========================
# Entry Point
# =========================

def run_chunking():
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = CHUNKS_DIR / "chunks.jsonl"

    input_file = PROCESSED_TEXT_DIR / "cleaned_text.txt"

    if not input_file.exists():
        print("❌ cleaned_text.txt not found.")
        return

    text = input_file.read_text(encoding="utf-8")

    if not text.strip():
        print("⚠️ cleaned_text.txt is empty.")
        return

    chunks = chunk_from_cleaned_text(text)

    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"[INFO] Stored {len(chunks)} chunks in {output_file}")

    save_book_taxonomy(chunks = chunks, output_dir = PROCESSED_TEXT_DIR)


if __name__ == "__main__":
    run_chunking()
