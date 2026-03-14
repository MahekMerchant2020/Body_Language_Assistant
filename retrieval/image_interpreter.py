"""
image_interpreter.py

Uses a multimodal LLM (via OpenRouter) to interpret a user-provided image for
body-language relevant cues, then returns a schema-stable JSON result.

Key features:
- Works with OpenRouter (Gemini 2.5 Flash Lite by default)
- Robust JSON extraction (handles ```json fences)
- Normalizes outputs so Pydantic validation is stable even if the LLM returns strings instead of lists
- Produces per-person observations where possible

Expected .env:
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1   (optional)
IMAGE_INTERPRETER_MODEL=google/gemini-2.5-flash-lite (optional)
"""

import os
import re
import json
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# =========================
# Config
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]  # OutskillGenAI_Capstone/
load_dotenv(dotenv_path=BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
IMAGE_INTERPRETER_MODEL = os.getenv("IMAGE_INTERPRETER_MODEL", "google/gemini-2.5-flash-lite")

if not OPENROUTER_API_KEY:
    raise EnvironmentError(
        "❌ OPENROUTER_API_KEY not found.\n"
        f"Add it to: {BASE_DIR / '.env'}\n"
        "Example:\n"
        "OPENROUTER_API_KEY=your_key_here"
    )

# =========================
# Schema
# =========================

class PersonCue(BaseModel):
    person: str
    description: str


class ImageInterpretation(BaseModel):
    posture: List[PersonCue] = Field(default_factory=list)
    gesture_cluster: List[PersonCue] = Field(default_factory=list)
    tension_signals: List[PersonCue] = Field(default_factory=list)
    possible_interpretation: str
    confidence: int = Field(ge=0, le=100)
    short_caption: str = Field(default="")

# =========================
# Utilities
# =========================

def _image_to_data_url(image_path: Path) -> str:
    ext = image_path.suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    mime = f"image/{ext}" if ext in {"png", "jpeg", "webp"} else "image/png"

    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _strip_code_fences(s: str) -> str:
    # Removes ```json ... ``` or ``` ... ```
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _extract_json_block(text: str) -> str:
    """
    Tries to reliably extract the first JSON object from a model response.
    Handles code fences and extra commentary.
    """
    cleaned = _strip_code_fences(text)

    # If it's already valid JSON, return
    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass

    # Fallback: find first {...} block (greedy but workable)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1].strip()
        return candidate

    return cleaned  # let caller error with context

def normalize_to_list(value: Any) -> List[Dict[str, str]]:
    """
    Ensures posture / gesture_cluster / tension_signals are always:
      List[{"person": str, "description": str}]
    """
    if isinstance(value, list):
        # allow list of dicts, or list of strings (normalize)
        out: List[Dict[str, str]] = []
        for item in value:
            if isinstance(item, dict) and "description" in item:
                out.append({
                    "person": str(item.get("person", "unspecified")),
                    "description": str(item.get("description", "")).strip(),
                })
            elif isinstance(item, str):
                out.append({"person": "unspecified", "description": item.strip()})
        return [x for x in out if x.get("description")]

    if isinstance(value, str):
        v = value.strip()
        if not v:
            return []
        return [{"person": "unspecified", "description": v}]

    return []

def normalize_image_output(data: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(data or {})
    data["posture"] = normalize_to_list(data.get("posture"))
    data["gesture_cluster"] = normalize_to_list(data.get("gesture_cluster"))
    data["tension_signals"] = normalize_to_list(data.get("tension_signals"))

    # Ensure required fields exist
    if "possible_interpretation" not in data or not isinstance(data["possible_interpretation"], str):
        data["possible_interpretation"] = ""

    # Confidence sometimes comes as string; coerce safely
    conf = data.get("confidence", 0)
    try:
        conf_int = int(conf)
    except Exception:
        conf_int = 0
    data["confidence"] = max(0, min(100, conf_int))

    return data

# =========================
# LLM Call
# =========================

def _openrouter_chat(messages: List[Dict[str, Any]], temperature: float = 0.2) -> str:
    url = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": IMAGE_INTERPRETER_MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()

    # OpenAI-compatible shape
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected OpenRouter response format: {data}")

# =========================
# Public API
# =========================

def interpret_image(
    image_path: str | Path,
    user_query: Optional[str] = None,
) -> ImageInterpretation:
    """
    Returns a structured interpretation of body-language relevant cues in the image.
    """
    path = Path(image_path)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"❌ Image not found: {path}")

    data_url = _image_to_data_url(path)

    system = (
        "You are a careful body-language assistant.\n"
        "Only describe what is visible (posture, gestures, orientation, spacing). "
        "Avoid identity guessing or sensitive attribute inference.\n"
        "Output ONLY valid JSON that matches the requested schema. No markdown fences."
    )

    schema_hint = {
        "posture": [{"person": "string", "description": "string"}],
        "gesture_cluster": [{"person": "string", "description": "string"}],
        "tension_signals": [{"person": "string", "description": "string"}],
        "possible_interpretation": "string",
        "confidence": 0
    }

    prompt = (
        "Analyze the image for body-language cues.\n"
        "Return JSON with fields:\n"
        "- posture: list of {person, description}\n"
        "- gesture_cluster: list of {person, description}\n"
        "- tension_signals: list of {person, description}\n"
        "- possible_interpretation: a grounded summary\n"
        "- confidence: integer 0-100\n\n"
        f"JSON schema example (do not copy, just follow shape):\n{json.dumps(schema_hint)}\n"
    )

    if user_query:
        prompt += f"\nUser question/context: {user_query}\n"

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    content = _openrouter_chat(messages)

    json_text = _extract_json_block(content)
    try:
        raw = json.loads(json_text)
    except Exception as e:
        raise ValueError(
            f"❌ Failed to parse image interpretation JSON.\n\nRaw output:\n{content}\n\n"
            f"Extracted candidate:\n{json_text}\n\nError: {e}"
        )

    normalized = normalize_image_output(raw)
    try:
        short_caption = build_short_caption(normalized)
        normalized["short_caption"] = short_caption
        interpretation = ImageInterpretation(**normalized)
        return interpretation
    except Exception as e:
        raise ValueError(
            f"❌ Parsed JSON but failed schema validation.\n\nNormalized:\n{json.dumps(normalized, indent=2)}\n\n"
            f"Original model output:\n{content}\n\nError: {e}"
        )

def build_short_caption(data: dict) -> str:
    """
    Builds a compact semantic caption from structured interpretation.
    Used for fallback vector search.
    """

    parts = []

    for item in data.get("posture", [])[:2]:
        parts.append(item["description"])

    for item in data.get("gesture_cluster", [])[:2]:
        parts.append(item["description"])

    summary = data.get("possible_interpretation", "")

    caption = ". ".join(parts)
    if summary:
        caption += f". Overall: {summary}"

    return caption.strip()

# =========================
# Quick local test
# =========================

if __name__ == "__main__":
    test_path = BASE_DIR / "test.jpg"
    if test_path.exists():
        result = interpret_image(test_path, user_query="Analyze this posture.")
        print(result.model_dump())
    else:
        print(f"[WARN] Test image not found at: {test_path}")