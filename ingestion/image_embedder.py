"""
image_embedder.py

Embeds images locally using CLIP:
openai/clip-vit-base-patch32

Writes JSONL records for vector DB ingestion.

✅ Consistent with index_builder.py convention:
- text metadata uses:   linked_image_ids
- image metadata uses:  linked_text_ids
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# =========================
# Configuration
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]  # OutskillGenAI_Capstone/
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

IMAGE_DIR = BASE_DIR / "data/processed/processed_images"
OUTPUT_DIR = BASE_DIR / "embeddings/image_embeddings"
OUTPUT_FILE = OUTPUT_DIR / "image_embeddings.jsonl"

# For building image -> text links
TEXT_CHUNKS_FILE = BASE_DIR / "data/processed/chunks/chunks.jsonl"

BATCH_SIZE = 16  # images are heavier than text on CPU


# =========================
# Helpers
# =========================

def l2_normalize(v: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / denom


def build_image_to_text_map() -> Dict[str, List[str]]:
    """
    Map image_id (stem) -> list of text chunk_ids that reference it.
    Reads from chunks.jsonl (correct source of linked_image_ids).
    """
    mapping: Dict[str, List[str]] = {}

    if not TEXT_CHUNKS_FILE.exists():
        print(f"[WARN] chunks.jsonl not found, cannot build image↔text links: {TEXT_CHUNKS_FILE}")
        return mapping

    with open(TEXT_CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue

            linked_images = chunk.get("metadata", {}).get("linked_image_ids", []) or []
            for img_id in linked_images:
                mapping.setdefault(img_id, []).append(chunk_id)

    return mapping


def load_image_rgb(path: Path) -> Image.Image:
    """Load image safely and force RGB for CLIP input."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# =========================
# Core Logic
# =========================

def embed_images():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(IMAGE_DIR.glob("*.png")))
    if len(image_files) == 0:
        print(f"[WARN] No processed images found in: {IMAGE_DIR}")
        print("[WARN] Skipping image embedding step (nothing to embed).")
        # Still create an empty jsonl so downstream code won't crash if it expects the file
        OUTPUT_FILE.write_text("", encoding="utf-8")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    model.eval()

    print(f"[INFO] Found {len(image_files)} images")

    # Build cross-modal links: image_id -> [chunk_id, ...]
    image_to_text = build_image_to_text_map()

    embedded_count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
        for start in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Embedding images"):
            batch_paths = image_files[start:start + BATCH_SIZE]
            batch_imgs = [load_image_rgb(p) for p in batch_paths]

            inputs = processor(images=batch_imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                feats = model.get_image_features(**inputs)

                # Some transformer versions return a ModelOutput; handle both.
                if hasattr(feats, "pooler_output"):
                    feats = feats.pooler_output

                feats = feats.to(dtype=torch.float32).cpu().numpy()

            feats = l2_normalize(feats)

            for img_path, vec in zip(batch_paths, feats):
                img_id = img_path.stem
                linked_text_ids = image_to_text.get(img_id, [])

                record = {
                    "id": img_id,
                    "vector": vec.tolist(),
                    "metadata": {
                        "modality": "image",
                        "image_path": str(img_path),
                        # ✅ Consistent key expected by index_builder.py
                        "linked_text_ids": linked_text_ids,
                    },
                }

                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                embedded_count += 1

    print(f"[INFO] Successfully embedded {embedded_count} images")
    print(f"[INFO] Image embeddings saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    embed_images()
