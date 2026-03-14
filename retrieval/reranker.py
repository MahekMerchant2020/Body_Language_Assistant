# retrieval/reranker.py

from __future__ import annotations

from typing import List, Dict, Any, Optional
import math
import torch
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Text-only reranker using a cross-encoder.

    - Higher score = more relevant
    - Reranks only text chunks
    - NaN/Inf-safe
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 256,   # keep this <= 512; 256 is faster + often enough for chunks
        force_fp32: bool = True, # strongly recommended to avoid NaNs
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # CrossEncoder supports max_length in newer sentence-transformers
        # If your version doesn't, it will still work, just ignore it.
        try:
            self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        except TypeError:
            self.model = CrossEncoder(model_name, device=device)
            # fallback: set attribute if present
            if hasattr(self.model, "max_length"):
                self.model.max_length = max_length

        self.batch_size = batch_size
        self.device = device

        if force_fp32:
            # Ensure model weights are float32 (helps prevent NaN on some setups)
            try:
                self.model.model = self.model.model.to(dtype=torch.float32)
            except Exception:
                pass

    @staticmethod
    def _sanitize_score(x: Any) -> float:
        """
        Convert to float and remove NaN/Inf.
        Use a very low score for invalid values so they sink in ranking.
        """
        try:
            f = float(x)
        except Exception:
            return -1e9

        if math.isnan(f) or math.isinf(f):
            return -1e9

        return f

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        text_key: str = "text",
        top_k: Optional[int] = None,
        normalize_scores: bool = False,
    ) -> List[Dict[str, Any]]:

        if not query or not query.strip() or not candidates:
            return candidates

        pairs = []
        filtered = []

        for c in candidates:
            text = (c.get(text_key) or "").strip()
            if not text:
                continue
            filtered.append(c)
            pairs.append((query, text))

        if not pairs:
            return candidates

        # Predict in batches
        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        scores = [self._sanitize_score(s) for s in raw_scores]

        # Optional normalization (0–1) AFTER sanitization
        if normalize_scores and scores:
            min_s = min(scores)
            max_s = max(scores)
            denom = (max_s - min_s) + 1e-12
            scores = [(s - min_s) / denom for s in scores]

        # Attach rerank_score
        for c, s in zip(filtered, scores):
            c["rerank_score"] = float(s)

        # Sort descending (higher = better)
        filtered.sort(key=lambda x: x.get("rerank_score", -1e9), reverse=True)

        if top_k is not None:
            filtered = filtered[:top_k]

        return filtered