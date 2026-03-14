import json
import re
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = BASE_DIR / "data/processed/processed_text/book_taxonomy.json"


class TaxonomyRouter:

    def __init__(self):

        with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
            self.taxonomy = json.load(f)

    # ----------------------------
    # Token normalization
    # ----------------------------

    def normalize_tokens(self, text: str):

        text = text.lower()

        text = re.sub(r"[^a-z0-9\s]", " ", text)

        tokens = text.split()

        return set(tokens)

    # ----------------------------
    # Chapter prediction
    # ----------------------------

    def predict_chapters(self, query: str) -> List[str]:

        q_tokens = self.normalize_tokens(query)

        scores = {}

        for chapter, subs in self.taxonomy.items():

            score = 0

            chap_tokens = self.normalize_tokens(chapter)

            overlap = len(q_tokens & chap_tokens)

            score += overlap * 2

            for sub in subs:

                sub_tokens = self.normalize_tokens(sub)

                overlap = len(q_tokens & sub_tokens)

                score += overlap

            scores[chapter] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [c for c, s in ranked[:2] if s > 0]