import json
import fitz
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parents[1]


class PDFLoader:
    """
    Extracts structured text spans (with font metadata) and images from PDF.
    """

    def __init__(
        self,
        pdf_path: str,
        raw_text_dir: str = "data/raw/raw_text",
        raw_image_dir: str = "data/raw/raw_images",
    ):
        self.pdf_path = BASE_DIR / pdf_path
        self.raw_text_dir = BASE_DIR / raw_text_dir
        self.raw_image_dir = BASE_DIR / raw_image_dir

        self.raw_text_dir.mkdir(parents=True, exist_ok=True)
        self.raw_image_dir.mkdir(parents=True, exist_ok=True)
    

    def clear_directory(self, directory_path: Path):
        """
        Deletes all files and subdirectories inside the given directory
        but keeps the directory itself.
        """
        if not directory_path.exists():
            return
        
        for item in directory_path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    # --------------------------------------------------
    # TEXT + IMAGE EXTRACTION (STRUCTURED)
    # --------------------------------------------------

    def extract_text_and_images(self):
        doc = fitz.open(self.pdf_path)

        structured_output = []
        image_count = 0

        for page_num, page in enumerate(doc, start=1):

            structured_output.append({
                "type": "page_marker",
                "page": page_num,
                "text": f"--- Page {page_num} ---"
            })

            blocks = page.get_text("dict")["blocks"]

            # Sort blocks top-to-bottom, then left-to-right
            blocks = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))

            for block_index, block in enumerate(blocks):

                # ------------------------
                # TEXT BLOCK
                # ------------------------
                if block["type"] == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):

                            span_text = span.get("text", "").strip()
                            if not span_text:
                                continue

                            structured_output.append({
                                "type": "text",
                                "page": page_num,
                                "text": span_text,
                                "font": span.get("font"),
                                "size": span.get("size"),
                                "flags": span.get("flags"),
                            })

                # ------------------------
                # IMAGE BLOCK
                # ------------------------
                elif block["type"] == 1:
                    image_bytes = block.get("image")
                    if not image_bytes:
                        continue

                    image_ext = block.get("ext", "png")
                    image_name = f"img_page_{page_num:03d}_{block_index:02d}.{image_ext}"
                    image_path = self.raw_image_dir / image_name

                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)

                    structured_output.append({
                        "type": "image",
                        "page": page_num,
                        "image_name": image_name
                    })

                    image_count += 1

        doc.close()

        output_path = self.raw_text_dir / "raw_structured.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(structured_output, f, indent=2)

        return output_path, image_count

    # --------------------------------------------------
    # MAIN LOAD FUNCTION
    # --------------------------------------------------

    def load(self):

        # 🔥 Clear image folders before extraction
        self.clear_directory(self.raw_image_dir)

        processed_image_dir = BASE_DIR / "data" / "processed" / "processed_images"
        self.clear_directory(processed_image_dir)

        text_path, num_images = self.extract_text_and_images()

        return {
            "raw_text_path": str(text_path),
            "raw_image_dir": str(self.raw_image_dir),
            "num_images": num_images,
        }
