"""
Vision tools — analyse images via Claude claude-sonnet-4-6 vision API.

Used by:
  - VisionAgent: analyse reference images → structured scene parameters
  - QA Agent:    compare a rendered frame vs the reference → correction dict

Both functions return structured dicts (not raw text) so the pipeline can
act on them programmatically.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path


def _encode_image(path_or_url: str) -> tuple[str, str]:
    """
    Return (media_type, base64_data) for an image file or a data-URI.
    If path_or_url starts with http/https, download it first.
    """
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


def _claude_vision(prompt: str, image_path_or_url: str, model: str = "claude-sonnet-4-6") -> str:
    """Call Claude vision with a single image and a text prompt."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    media_type, b64 = _encode_image(image_path_or_url)

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return response.content[0].text


def analyse_reference_image(image_path_or_url: str) -> dict:
    """
    Analyse a reference image and extract structured scene parameters for Blender.

    Returns a dict with keys:
        dominant_colors: list[str]   — hex colour codes (up to 5)
        background_type: str         — "solid" | "gradient" | "hdri" | "texture"
        lighting_type: str           — "studio" | "natural" | "dramatic" | "neon"
        camera_angle: str            — "front" | "top" | "isometric" | "orbit" | "close_up"
        mood: str                    — "cinematic" | "minimal" | "energetic" | "calm" | "dark"
        key_objects: list[str]       — main scene elements to reproduce
        blender_reference_mode: int  — 1 (Image overlay), 2 (Camera BG), or 3 (World HDRI)
        notes: str                   — free-text advice for the bpy script generator
    """
    prompt = """Analyse this reference image for 3D scene reproduction in Blender.
Return ONLY valid JSON with these fields:
{
  "dominant_colors": ["#rrggbb", ...],        // up to 5 hex codes
  "background_type": "solid|gradient|hdri|texture",
  "lighting_type": "studio|natural|dramatic|neon",
  "camera_angle": "front|top|isometric|orbit|close_up",
  "mood": "cinematic|minimal|energetic|calm|dark",
  "key_objects": ["object1", "object2"],      // main scene elements
  "blender_reference_mode": 1,                // 1=Image overlay, 2=Camera BG, 3=World HDRI
  "notes": "free text advice for the script generator"
}
blender_reference_mode guide:
  1 = Simple image: flat design, 2D, logo, UI screenshot → Mode 1 (Empty/Image overlay)
  2 = Photo/realistic: product shot, portrait, real scene → Mode 2 (Camera Background)
  3 = Environment/landscape: outdoor, sky, HDRI-worthy scene → Mode 3 (World/HDRI)
"""
    raw = _claude_vision(prompt, image_path_or_url)

    # Extract JSON from the response
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # fallback defaults
        return {
            "dominant_colors": ["#1a1a2e"],
            "background_type": "gradient",
            "lighting_type": "studio",
            "camera_angle": "front",
            "mood": "cinematic",
            "key_objects": [],
            "blender_reference_mode": 2,
            "notes": raw,
        }


def compare_render_to_reference(
    render_path: str,
    reference_path_or_url: str,
    prompt_context: str = "",
) -> dict:
    """
    QA comparison: analyse a rendered frame against the reference image.

    Returns a dict with keys:
        match_score: float          — 0.0–1.0 (1.0 = perfect match)
        approved: bool              — True if match_score >= 0.70
        corrections: dict           — keyed adjustments to apply in the next render pass:
            lighting_correction: str | None
            color_correction: str | None
            composition_correction: str | None
            object_correction: str | None
        notes: str                  — human-readable QA summary
    """
    # Build a two-image prompt
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    ref_media_type, ref_b64 = _encode_image(reference_path_or_url)
    render_media_type, render_b64 = _encode_image(render_path)

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

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": ref_media_type, "data": ref_b64},
                    },
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": render_media_type, "data": render_b64},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "match_score": 0.5,
            "approved": False,
            "corrections": {},
            "notes": raw,
        }

    # Ensure approved key is consistent with score
    result.setdefault("approved", result.get("match_score", 0) >= 0.70)
    return result
