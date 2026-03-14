"""
text_embedder.py

Embeds text chunks locally using CLIP:
openai/clip-vit-base-patch32

- Loads chunks
- Cleans image placeholders
- Filters linked_image_ids to only include processed images
- Embeds text locally (no HF API)
- Saves embeddings + metadata for LanceDB ingestion
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import CLIPTokenizerFast, CLIPModel

# =========================
# Configuration
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=BASE_DIR / ".env")  # optional; safe to keep

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

CHUNKS_FILE = BASE_DIR / "data/processed/chunks/chunks.jsonl"
PROCESSED_IMAGES_DIR = BASE_DIR / "data/processed/processed_images"

OUTPUT_DIR = BASE_DIR / "embeddings/text_embeddings"
OUTPUT_FILE = OUTPUT_DIR / "text_embeddings.jsonl"

BATCH_SIZE = 32
MAX_TOKENS = 77  # CLIP text limit

# =========================
# Helpers
# =========================

def load_chunks() -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"[INFO] Loaded {len(chunks)} text chunks")
    return chunks


def get_valid_image_ids() -> set:
    # processed images are stored as .png in your pipeline
    return {p.stem for p in PROCESSED_IMAGES_DIR.glob("*.png")}


def clean_text_for_embedding(text: str) -> str:
    # Remove image markers like [IMAGE_REF: img_page_012_03.png]
    text = re.sub(r"\[IMAGE_REF: .*?\]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def l2_normalize(v: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / denom


# =========================
# Core Logic
# =========================

def embed_chunks():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"❌ Chunks file not found: {CHUNKS_FILE}")

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Load CLIP
    tokenizer = CLIPTokenizerFast.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    model.eval()

    chunks = load_chunks()
    valid_images = get_valid_image_ids()

    # Prepare items to embed (skip empties)
    embed_items = []
    for chunk in chunks:
        raw_text = chunk.get("text", "")
        clean_text = clean_text_for_embedding(raw_text)
        if not clean_text:
            continue
        embed_items.append((chunk, clean_text))

    print(f"[INFO] Chunks to embed (non-empty after cleaning): {len(embed_items)}")

    embedded_count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
        for start in tqdm(range(0, len(embed_items), BATCH_SIZE), desc="Embedding text"):
            batch = embed_items[start:start + BATCH_SIZE]
            batch_chunks = [c for (c, _) in batch]
            batch_texts = [t for (_, t) in batch]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                out = model.get_text_features(**inputs)
                # Transformers version differences:
                # - sometimes returns a Tensor
                # - sometimes returns a ModelOutput with pooler_output
                if hasattr(out, "pooler_output"):
                    feats = out.pooler_output
                else:
                    feats = out
                feats = feats.to(dtype=torch.float32).cpu().numpy()

            # Normalize for cosine similarity in vector DB
            feats = l2_normalize(feats)

            # Write JSONL records
            for chunk, text_str, vec in zip(batch_chunks, batch_texts, feats):
                metadata = chunk.get("metadata", {})

                linked_images = metadata.get("linked_image_ids", [])
                linked_images = [img_id for img_id in linked_images if img_id in valid_images]

                record = {
                    "id": chunk["chunk_id"],
                    "vector": vec.tolist(),
                    "metadata": {
                        **metadata,
                        "linked_image_ids": linked_images,
                        "modality": "text",
                        "text": text_str,
                    },
                }

                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                embedded_count += 1

    print(f"[INFO] Successfully embedded {embedded_count} chunks")
    print(f"[INFO] Text embeddings saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    embed_chunks()
