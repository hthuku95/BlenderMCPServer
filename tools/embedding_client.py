"""
Multimodal embedding client — gemini-embedding-2 only.

gemini-embedding-2 is Google's first natively multimodal embedding model (GA April 2026).
Accepts text, images, video, audio, and PDFs as input. Maps all modalities into a
single unified embedding space (3072-dimensional, configurable via output_dimensionality).

Provider fallback:
  - Primary: Gemini Embedding 2 (gemini-embedding-2, multimodal)
  - If GEMINI_API_KEY is missing: raise RuntimeError (no text-only fallback permitted)

Usage:
    from tools.embedding_client import generate_embedding, generate_multimodal_embedding

    # Text only
    vec = generate_embedding("Describe this video", dimension=768)

    # Multimodal (text + image)
    with open("frame.png", "rb") as f:
        vec = generate_multimodal_embedding(
            text="A person walking",
            image_bytes=f.read(),
            dimension=768,
        )
"""

from __future__ import annotations

import os
from typing import Any


_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
_DEFAULT_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))


def _get_client():
    """Return a google-genai client using Gemini API key."""
    api_key = (
        os.getenv("VIDEO_GEMINI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "No Gemini API key found (VIDEO_GEMINI_API_KEY / GEMINI_API_KEY). "
            "gemini-embedding-2 requires a Gemini API key. "
            "No text-only embedding fallback is permitted."
        )
    from google import genai as google_genai
    return google_genai.Client(api_key=api_key)


def generate_embedding(
    text: str,
    dimension: int | None = None,
) -> list[float]:
    """
    Generate a text embedding using gemini-embedding-2.

    Args:
        text: Input text (up to 8192 tokens).
        dimension: Output dimensionality (128–3072). Defaults to EMBEDDING_DIMENSION env
                   (768) or 768 if not set.

    Returns:
        A list[float] of length `dimension`.

    Raises:
        RuntimeError: If no Gemini API key is configured.
    """
    client = _get_client()
    dim = dimension or _DEFAULT_DIMENSION

    from google.genai import types

    result = client.models.embed_content(
        model=_EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=dim),
    )
    return result.embeddings[0].values


def generate_multimodal_embedding(
    text: str | None = None,
    image_bytes: bytes | None = None,
    image_mime: str = "image/png",
    video_bytes: bytes | None = None,
    video_mime: str = "video/mp4",
    audio_bytes: bytes | None = None,
    audio_mime: str = "audio/wav",
    pdf_bytes: bytes | None = None,
    dimension: int | None = None,
) -> list[float]:
    """
    Generate a multimodal embedding from text + optional media.

    gemini-embedding-2 accepts interleaved text, images, video, audio, and PDF
    in a single request and produces one aggregated embedding. If you need
    separate embeddings per input, use the Batch API.

    Args:
        text: Optional text description.
        image_bytes: Raw image bytes (PNG/JPEG, max 6 images total per call).
        image_mime: MIME type of image bytes.
        video_bytes: Raw video bytes (MP4/MOV, max 120 seconds).
        video_mime: MIME type of video bytes.
        audio_bytes: Raw audio bytes (MP3/WAV, max 180 seconds).
        audio_mime: MIME type of audio bytes.
        pdf_bytes: Raw PDF bytes (max 6 pages).
        dimension: Output dimensionality (128–3072). Default 768.

    Returns:
        A list[float] of length `dimension`.

    Raises:
        RuntimeError: If no Gemini API key is configured.
        ValueError: If no input content is provided.
    """
    client = _get_client()
    dim = dimension or _DEFAULT_DIMENSION

    from google.genai import types

    parts: list[Any] = []

    if text:
        parts.append(text)

    if image_bytes:
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime))

    if video_bytes:
        parts.append(types.Part.from_bytes(data=video_bytes, mime_type=video_mime))

    if audio_bytes:
        parts.append(types.Part.from_bytes(data=audio_bytes, mime_type=audio_mime))

    if pdf_bytes:
        parts.append(types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))

    if not parts:
        raise ValueError("At least one input (text, image, video, audio, or PDF) is required")

    result = client.models.embed_content(
        model=_EMBEDDING_MODEL,
        contents=parts,
        config=types.EmbedContentConfig(output_dimensionality=dim),
    )
    return result.embeddings[0].values
