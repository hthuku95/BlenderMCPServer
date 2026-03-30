"""
Vision tools — image analysis via Claude AND Gemini vision APIs.

Strategy:
  - analyse_reference_image(): uses Gemini (gemini-2.0-flash) as primary,
    falls back to Claude (claude-sonnet-4-6) if Gemini fails or key is absent.
  - compare_render_to_reference(): uses Claude as primary (stronger at
    multi-image comparison with structured JSON output), falls back to Gemini.

This dual-provider pattern maximises availability and lets the two models
cross-check each other's analysis.

Used by:
  - VisionAgent: analyse reference images → structured scene parameters
  - QA Agent:    compare a rendered frame vs the reference → correction dict
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _encode_image(path_or_url: str) -> tuple[str, str]:
    """Return (media_type, base64_data) for a local file or http(s) URL."""
    if path_or_url.startswith(("http://", "https://")):
        import httpx
        resp = httpx.get(path_or_url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "image/jpeg").split(";")[0]
        return ctype, base64.standard_b64encode(resp.content).decode()

    data = Path(path_or_url).read_bytes()
    suffix = Path(path_or_url).suffix.lower()
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")
    return media_type, base64.standard_b64encode(data).decode()


def _extract_json(raw: str) -> dict | None:
    """Strip markdown fences and parse JSON. Returns None on failure."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        # parts[1] is the fenced block
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Provider backends
# ---------------------------------------------------------------------------

def _gemini_vision_single(prompt: str, image_path_or_url: str) -> str:
    """Call Gemini with one image. Returns raw text response."""
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash-lite")

    media_type, b64 = _encode_image(image_path_or_url)
    image_part = {"mime_type": media_type, "data": b64}

    response = model.generate_content([image_part, prompt])
    return response.text


def _gemini_vision_two(prompt: str, image1: str, image2: str) -> str:
    """Call Gemini with two images (for render vs reference comparison)."""
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash-lite")

    mt1, b1 = _encode_image(image1)
    mt2, b2 = _encode_image(image2)

    response = model.generate_content([
        {"mime_type": mt1, "data": b1},
        {"mime_type": mt2, "data": b2},
        prompt,
    ])
    return response.text


def _claude_vision_single(prompt: str, image_path_or_url: str) -> str:
    """Call Claude vision with one image."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    media_type, b64 = _encode_image(image_path_or_url)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return response.content[0].text


def _claude_vision_two(prompt: str, image1: str, image2: str) -> str:
    """Call Claude vision with two images."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    mt1, b1 = _encode_image(image1)
    mt2, b2 = _encode_image(image2)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mt1, "data": b1}},
                {"type": "image", "source": {"type": "base64", "media_type": mt2, "data": b2}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_reference_image(image_path_or_url: str) -> dict:
    """
    Analyse a reference image and extract structured scene parameters for Blender.

    Primary: Gemini (gemini-2.0-flash) — fast and cost-effective.
    Fallback: Claude (claude-sonnet-4-6) — used if Gemini key absent or call fails.

    Returns a dict with keys:
        dominant_colors: list[str]   — hex colour codes (up to 5)
        background_type: str         — "solid" | "gradient" | "hdri" | "texture"
        lighting_type: str           — "studio" | "natural" | "dramatic" | "neon"
        camera_angle: str            — "front" | "top" | "isometric" | "orbit" | "close_up"
        mood: str                    — "cinematic" | "minimal" | "energetic" | "calm" | "dark"
        key_objects: list[str]       — main scene elements to reproduce
        blender_reference_mode: int  — 1 (Image overlay), 2 (Camera BG), or 3 (World HDRI)
        notes: str                   — free-text advice for the bpy script generator
        _provider: str               — "gemini" or "claude" (which model answered)
    """
    prompt = """Analyse this reference image for 3D scene reproduction in Blender.
Return ONLY valid JSON with these fields:
{
  "dominant_colors": ["#rrggbb", ...],
  "background_type": "solid|gradient|hdri|texture",
  "lighting_type": "studio|natural|dramatic|neon",
  "camera_angle": "front|top|isometric|orbit|close_up",
  "mood": "cinematic|minimal|energetic|calm|dark",
  "key_objects": ["object1", "object2"],
  "blender_reference_mode": 1,
  "notes": "free text advice for the script generator"
}
blender_reference_mode guide:
  1 = Simple image: flat design, 2D, logo, UI screenshot
  2 = Photo/realistic: product shot, portrait, real scene
  3 = Environment/landscape: outdoor, sky, HDRI-worthy scene
"""

    _FALLBACK = {
        "dominant_colors": ["#1a1a2e"],
        "background_type": "gradient",
        "lighting_type": "studio",
        "camera_angle": "front",
        "mood": "cinematic",
        "key_objects": [],
        "blender_reference_mode": 2,
        "notes": "",
        "_provider": "fallback",
    }

    # Try Gemini first
    if os.environ.get("GEMINI_API_KEY"):
        try:
            raw = _gemini_vision_single(prompt, image_path_or_url)
            result = _extract_json(raw)
            if result:
                result["_provider"] = "gemini"
                return result
        except Exception:
            pass  # fall through to Claude

    # Try Claude
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            raw = _claude_vision_single(prompt, image_path_or_url)
            result = _extract_json(raw)
            if result:
                result["_provider"] = "claude"
                return result
        except Exception:
            pass

    return _FALLBACK


def compare_render_to_reference(
    render_path: str,
    reference_path_or_url: str,
    prompt_context: str = "",
) -> dict:
    """
    QA comparison: compare a rendered frame against the reference image.

    Primary: Claude (claude-sonnet-4-6) — stronger at structured multi-image comparison.
    Fallback: Gemini (gemini-2.0-flash) — used if Claude key absent or call fails.

    Returns a dict with keys:
        match_score: float          — 0.0–1.0
        approved: bool              — True if match_score >= 0.70
        corrections: dict
            lighting_correction: str | None
            color_correction: str | None
            composition_correction: str | None
            object_correction: str | None
        notes: str
        _provider: str              — "claude" or "gemini"
    """
    context_str = f"\nOriginal prompt context: {prompt_context}" if prompt_context else ""
    prompt = f"""Compare these two images: [IMAGE 1 = REFERENCE], [IMAGE 2 = RENDER].{context_str}

Return ONLY valid JSON:
{{
  "match_score": 0.0-1.0,
  "approved": true|false,
  "corrections": {{
    "lighting_correction": "description or null",
    "color_correction": "description or null",
    "composition_correction": "description or null",
    "object_correction": "description or null"
  }},
  "notes": "summary"
}}

approved = true if match_score >= 0.70.
Be specific in corrections — they will be fed back into a Blender script generator."""

    _FALLBACK = {
        "match_score": 0.5,
        "approved": False,
        "corrections": {},
        "notes": "Vision comparison unavailable",
        "_provider": "fallback",
    }

    # Try Claude first (better multi-image structured output)
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            raw = _claude_vision_two(prompt, reference_path_or_url, render_path)
            result = _extract_json(raw)
            if result:
                result.setdefault("approved", result.get("match_score", 0) >= 0.70)
                result["_provider"] = "claude"
                return result
        except Exception:
            pass

    # Fallback to Gemini
    if os.environ.get("GEMINI_API_KEY"):
        try:
            raw = _gemini_vision_two(prompt, reference_path_or_url, render_path)
            result = _extract_json(raw)
            if result:
                result.setdefault("approved", result.get("match_score", 0) >= 0.70)
                result["_provider"] = "gemini"
                return result
        except Exception:
            pass

    return _FALLBACK
