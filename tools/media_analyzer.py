"""
media_analyzer.py — Video analysis via Gemini File API using a dedicated API key.

This module is intentionally isolated from the main LLM client so it uses
BLENDER_GEMINI_API_KEY (or VIDEO_GEMINI_API_KEY as fallback), keeping video
analysis quota completely separate from text generation quota.

The analyze_video_for_clips() function accepts a video URL (typically an R2
presigned URL or YouTube URL), uploads/references it via the Gemini File API,
runs a structured viral-moments analysis, and returns the same JSON schema that
the Rust VideoAnalysis struct expects.

Called by the /api/analyze-video REST endpoint in server.py.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import httpx

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Priority: dedicated blender key → video key → generic key
def _get_api_key() -> str | None:
    return (
        os.getenv("BLENDER_GEMINI_API_KEY")
        or os.getenv("VIDEO_GEMINI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )


def _make_prompt(clips_requested: int, min_duration: float, max_duration: float, factors: list[str]) -> str:
    factor_hint = ", ".join(factors) if factors else "humor, surprise, emotion, action, insight"
    return f"""Analyze this video and identify the most viral, engaging moments for YouTube Shorts / TikTok.

Requirements:
- Find exactly {clips_requested} moments (or fewer if the video is short)
- Each moment must be between {min_duration:.0f}s and {max_duration:.0f}s long
- Focus on: {factor_hint}

Respond with ONLY valid JSON (no markdown, no code blocks):
{{
  "video_summary": "<2-3 sentence summary of the full video>",
  "content_type": "<one of: gaming, tech, education, entertainment, sports, music, vlog, podcast, other>",
  "overall_quality": <float 0.0-1.0, how good the video is for clipping>,
  "viral_moments": [
    {{
      "start_sec": <float — start timestamp in seconds>,
      "end_sec": <float — end timestamp in seconds>,
      "title": "<short engaging title, max 60 chars>",
      "hook": "<first sentence/hook to grab attention>",
      "quality_score": <float 0.0-1.0>,
      "viral_factors": ["<factor1>", "<factor2>"],
      "thumbnail_sec": <float — best frame for thumbnail, within start_sec..end_sec>,
      "reason": "<why this moment is engaging>"
    }}
  ]
}}

Order viral_moments by quality_score descending."""


async def analyze_video_for_clips(
    video_url: str,
    clips_requested: int = 3,
    min_duration: float = 30.0,
    max_duration: float = 90.0,
    high_performing_factors: list[str] | None = None,
) -> dict[str, Any]:
    """
    Analyze a video URL for viral clip moments.

    Supports:
    - YouTube URLs: analyzed via Gemini's native YouTube understanding (fileData + fileUri)
    - Direct video URLs (R2 presigned, etc.): downloaded to a temp file then uploaded via
      Gemini File API before analysis

    Returns a dict matching the Rust VideoAnalysis struct schema.
    """
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("No Gemini API key configured (BLENDER_GEMINI_API_KEY, VIDEO_GEMINI_API_KEY, or GEMINI_API_KEY)")

    factors = high_performing_factors or []
    prompt = _make_prompt(clips_requested, min_duration, max_duration, factors)

    is_youtube = "youtube.com" in video_url or "youtu.be" in video_url

    if is_youtube:
        result = await _analyze_youtube_url(api_key, video_url, prompt)
    else:
        result = await _analyze_direct_url(api_key, video_url, prompt)

    return result


async def _analyze_youtube_url(api_key: str, video_url: str, prompt: str) -> dict[str, Any]:
    """Use Gemini's native YouTube URL understanding (no download needed)."""
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "fileData": {
                            "mimeType": "video/mp4",
                            "fileUri": video_url,
                        }
                    },
                    {"text": prompt},
                ],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.3,
        },
    }
    return await _call_gemini(api_key, payload)


async def _analyze_direct_url(api_key: str, video_url: str, prompt: str) -> dict[str, Any]:
    """Download video to temp file, upload via Gemini File API, then analyze."""
    # Download video
    async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
        resp = await client.get(video_url)
        resp.raise_for_status()
        video_bytes = resp.content

    # Upload to Gemini File API
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        file_uri = await _upload_to_gemini_file_api(api_key, tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Analyze uploaded file
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "fileData": {
                            "mimeType": "video/mp4",
                            "fileUri": file_uri,
                        }
                    },
                    {"text": prompt},
                ],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.3,
        },
    }
    return await _call_gemini(api_key, payload)


async def _upload_to_gemini_file_api(api_key: str, file_path: str) -> str:
    """Upload a local video file to the Gemini File API and return the file URI."""
    upload_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={api_key}"

    file_size = os.path.getsize(file_path)
    headers = {
        "X-Goog-Upload-Protocol": "multipart",
        "Content-Type": "multipart/related; boundary=boundary123",
    }

    with open(file_path, "rb") as f:
        video_bytes = f.read()

    metadata = json.dumps({"file": {"display_name": "video_for_analysis"}}).encode()
    body = (
        b"--boundary123\r\n"
        b"Content-Type: application/json\r\n\r\n"
        + metadata
        + b"\r\n--boundary123\r\n"
        b"Content-Type: video/mp4\r\n\r\n"
        + video_bytes
        + b"\r\n--boundary123--"
    )

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(upload_url, content=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    file_uri = data.get("file", {}).get("uri")
    if not file_uri:
        raise ValueError(f"Gemini File API did not return a URI. Response: {data}")

    # Wait for file to be ACTIVE (processing takes a few seconds)
    await _wait_for_file_active(api_key, file_uri)
    return file_uri


async def _wait_for_file_active(api_key: str, file_uri: str, max_attempts: int = 20) -> None:
    """Poll the Gemini File API until the file state is ACTIVE."""
    import asyncio

    # Extract file name from URI: "https://.../files/abc123" → "files/abc123"
    file_name = "/".join(file_uri.split("/")[-2:])
    status_url = f"{GEMINI_API_BASE}/{file_name}?key={api_key}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        for _ in range(max_attempts):
            resp = await client.get(status_url)
            resp.raise_for_status()
            state = resp.json().get("state", "")
            if state == "ACTIVE":
                return
            if state == "FAILED":
                raise ValueError("Gemini File API reported FAILED state for uploaded file")
            await asyncio.sleep(3)

    raise TimeoutError("Gemini File API file did not become ACTIVE within timeout")


async def _call_gemini(api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST to Gemini generateContent and parse the VideoAnalysis JSON response."""
    url = f"{GEMINI_API_BASE}/models/gemini-2.5-flash:generateContent?key={api_key}"

    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(url, json=payload)

    if resp.status_code == 429:
        raise RuntimeError(f"Gemini 429 rate limit: {resp.text}")
    resp.raise_for_status()

    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected Gemini response shape: {data}") from exc

    # Strip markdown fences if model included them despite responseMimeType
    cleaned = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini returned non-JSON text: {cleaned[:500]}") from exc
