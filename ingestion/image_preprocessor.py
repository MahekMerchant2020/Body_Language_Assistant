"""
image_preprocessor.py

Preprocesses raw images extracted from PDFs so they are suitable for
multimodal embedding models such as:
nvidia/omni-embed-nemotron-3b

Steps performed:
1. Load image
2. Fix orientation (EXIF)
3. Convert to RGB
4. Resize + center-crop to square
5. Normalize pixel values
6. Save processed image
"""

import os
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

# =========================
# Configuration
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]  # OutskillGenAI_Capstone/

RAW_IMAGE_DIR = BASE_DIR / "data/raw/raw_images"
PROCESSED_IMAGE_DIR = BASE_DIR / "data/processed/processed_images"

# Safe default for vision embedders
TARGET_IMAGE_SIZE = 224  # 224x224 is widely supported
OUTPUT_FORMAT = "PNG"    # Lossless and consistent


# =========================
# Utility Functions
# =========================

def normalize_image(img: Image.Image) -> Image.Image:
    """
    Normalize image pixel values to [0, 255] after float processing.
    (Actual tensor normalization will be done at embedding time.)
    """
    img = img.convert("RGB")  # force RGB before numpy

    img_array = np.asarray(img).astype(np.float32)
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    return Image.fromarray(img_array, mode="RGB")  # force RGB output


def resize_and_pad(img: Image.Image, size: int) -> Image.Image:
    """
    Resize while maintaining aspect ratio,
    then pad to square instead of cropping.
    """

    img.thumbnail((size, size), Image.BICUBIC)

    new_img = Image.new("RGB", (size, size), (255, 255, 255))  # white background

    paste_x = (size - img.width) // 2
    paste_y = (size - img.height) // 2

    new_img.paste(img, (paste_x, paste_y))

    return new_img


# =========================
# Core Processing Logic
# =========================

def preprocess_image(image_path: Path, output_dir: Path):
    """
    Preprocess a single image and save it to output_dir.
    """
    try:
        with Image.open(image_path) as img:
            # 1. Fix EXIF orientation (important for PDFs)
            img = ImageOps.exif_transpose(img)

            # 2. Convert to RGB (removes alpha & grayscale issues)
            img = img.convert("L").convert("RGB")

            # 3. Resize and center-crop
            img = resize_and_pad(img, TARGET_IMAGE_SIZE)

            # 4. Normalize
            img = normalize_image(img)

            # 5. Save processed image
            output_path = output_dir / f"{image_path.stem}.png"
            img = img.convert("RGB")
            img.save(output_path, format=OUTPUT_FORMAT)

    except Exception as e:
        print(f"[ERROR] Failed to process {image_path.name}: {e}")


def preprocess_all_images():
    """
    Preprocess all images in RAW_IMAGE_DIR.
    """
    PROCESSED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    image_files = [
        p for p in RAW_IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]

    print(f"[INFO] Found {len(image_files)} raw images")

    for img_path in image_files:
        preprocess_image(img_path, PROCESSED_IMAGE_DIR)

    print("[INFO] Image preprocessing complete")


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    preprocess_all_images()
