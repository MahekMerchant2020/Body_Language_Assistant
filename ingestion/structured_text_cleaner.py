import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parents[1]


class StructuredTextCleaner:
    """
    Cleans structured span-level JSON from PDFLoader.
    """

    def __init__(
        self,
        input_json_path="data/raw/raw_text/raw_structured.json",
        output_path="data/processed/processed_text/cleaned_text.txt",
    ):
        self.input_json_path = BASE_DIR / input_json_path
        self.output_path = BASE_DIR / output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # LOAD JSON
    # --------------------------------------------------

    def load_json(self):
        with open(self.input_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # --------------------------------------------------
    # CLEAN STRUCTURE
    # --------------------------------------------------

    def clean(self):
        data = self.load_json()

        # Collect font sizes from text spans only
        sizes = [
            round(item["size"])
            for item in data
            if item["type"] == "text"
        ]

        if not sizes:
            print("⚠ No text spans found.")
            self.output_path.write_text("")
            return self.output_path

        # Determine body font size (most common)
        body_font_size = Counter(sizes).most_common(1)[0][0]

        output_lines = []
        current_page = None

        for item in data:

            if item["type"] == "page_marker":
                current_page = item["page"]
                output_lines.append(f"\n--- Page {current_page} ---\n")
                continue

            if item["type"] == "image":
                output_lines.append(f"[IMAGE_REF: {item['image_name']}]")
                continue

            if item["type"] != "text":
                continue

            text = item["text"].strip()
            size = round(item["size"])
            font = item["font"]
            flags = item["flags"]

            if not text:
                continue

            # Skip page number footers (small numeric text)
            if text.isdigit() and size < body_font_size:
                continue

            # Skip running headers (italic small text)
            if "Italic" in font and size < body_font_size:
                continue

            is_bold = "Bold" in font or (flags & 16)

            # Chapter title (very large bold)
            if is_bold and size >= body_font_size + 6:
                output_lines.append(f"\n<CHAPTER_TITLE> {text}\n")
                continue

            # Subheading (medium bold)
            if is_bold and size > body_font_size:
                output_lines.append(f"\n<SUBHEADING> {text}\n")
                continue

            # Normal paragraph text
            output_lines.append(text)

        # --------------------------------------------------
        # MERGE CONSECUTIVE CHAPTER & SUBHEADING TAGS
        # --------------------------------------------------
        
        merged_lines = []
        i = 0
        
        while i < len(output_lines):
            line = output_lines[i].strip()
            
            # Merge consecutive CHAPTER_TITLE tags
            if line.startswith("<CHAPTER_TITLE>"):
                combined_text = line.replace("<CHAPTER_TITLE>", "").strip()
                
                j = i + 1               
                while j < len(output_lines):
                    next_line = output_lines[j].strip()
                    if next_line.startswith("<CHAPTER_TITLE>"):
                        combined_text += " " + next_line.replace("<CHAPTER_TITLE>", "").strip()
                        j += 1
                    else:
                        break
                
                merged_lines.append(f"\n<CHAPTER_TITLE> {combined_text}\n")
                i = j
                continue
            
            # Merge consecutive SUBHEADING tags
            if line.startswith("<SUBHEADING>"):
                combined_text = line.replace("<SUBHEADING>", "").strip()
                
                j = i + 1
                while j < len(output_lines):
                    next_line = output_lines[j].strip()
                    if next_line.startswith("<SUBHEADING>"):
                        combined_text += " " + next_line.replace("<SUBHEADING>", "").strip()
                        j += 1
                    else:
                        break
                
                merged_lines.append(f"\n<SUBHEADING> {combined_text}\n")
                i = j
                continue
            
            # Everything else remains unchanged
            merged_lines.append(output_lines[i])
            i += 1
        
        cleaned_text = "\n".join(merged_lines).strip()
        
        self.output_path.write_text(cleaned_text, encoding="utf-8")
        print("✅ Cleaned text length:", len(cleaned_text))
        
        return self.output_path

