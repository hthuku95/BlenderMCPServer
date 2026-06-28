"""
Video quality review — sends full video to multimodal models for native understanding.

Fallback chain (per rule #4: Ollama is always default):
  1. Ollama/Gemma 4 12B (self-hosted, free — no API key, no rate limits)
  2. Gemini (File API upload + native video understanding — quota-limited)
  3. Claude (direct URL video understanding — billing)

For Ollama, extracts evenly-spaced keyframes since Ollama's API accepts images
(video not yet natively supported in /api/chat). For Gemini and Claude, sends
the full video file natively.

Each provider uses its env-var-configured model from llm_client.py.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import subprocess
import tempfile

import httpx

from tools.llm_client import _OLLAMA_BASE_URL, _OLLAMA_MODEL, _GEMINI_MODEL, _CLAUDE_MODEL

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

_QA_PROMPT = """You are a professional video quality reviewer. Analyze the rendered video against the original creative brief.

Original brief: {brief}

Return ONLY valid JSON (no markdown):
{{
  "quality_score": <float 0.0-1.0, overall quality>,
  "brief_match_score": <float 0.0-1.0, how well the video matches the brief>,
  "technical_issues": ["issue1", "issue2"],
  "visual_quality": "<poor|fair|good|excellent>",
  "composition_feedback": "<feedback on framing, layout, camera work>",
  "pacing_feedback": "<feedback on timing, rhythm, duration>",
  "suggested_improvements": ["improvement1", "improvement2"],
  "summary": "<2-3 sentence overall assessment>"
}}

Be critical and specific. Quality_score of <0.6 means it needs a re-render.
Focus on: composition, lighting (if Blender), clarity (if Manim), pacing, and alignment with the brief."""


async def review_video(
    video_url: str,
    brief: str,
) -> dict:
    """Review a rendered video against its creative brief.

    Fallback chain:
      1. Ollama/Gemma 4 12B (self-hosted, free — default, always tried first)
      2. Gemini (File API upload + native video understanding)
      3. Claude (direct URL video understanding)

    Args:
        video_url: Presigned R2 URL or any direct video URL.
        brief: The original creative brief/prompt used to generate the video.

    Returns:
        Structured QA review dict with quality_score, brief_match_score,
        technical_issues, visual_quality, composition_feedback,
        pacing_feedback, suggested_improvements, summary.
    """
    errors: list[str] = []

    # --- 1. Try Ollama/Gemma 4 12B (free, self-hosted, always default) ---
    try:
        return await _review_with_ollama(video_url, brief)
    except Exception as exc:
        msg = f"Ollama review failed: {exc}"
        logger.warning("video_review: %s", msg)
        errors.append(msg)

    # --- 2. Try Gemini (quota-limited, fallback) ---
    gemini_key = _get_gemini_api_key()
    if gemini_key:
        try:
            return await _review_with_gemini(video_url, brief, gemini_key)
        except Exception as exc:
            msg = f"Gemini review failed: {exc}"
            logger.warning("video_review: %s", msg)
            errors.append(msg)

    # --- 3. Try Claude (billing, last resort) ---
    claude_key = os.getenv("ANTHROPIC_API_KEY")
    if claude_key:
        try:
            return await _review_with_claude(video_url, brief, claude_key)
        except Exception as exc:
            msg = f"Claude review failed: {exc}"
            logger.warning("video_review: %s", msg)
            errors.append(msg)

    # --- All providers failed ---
    return {
        "quality_score": 0.0,
        "brief_match_score": 0.0,
        "technical_issues": errors,
        "visual_quality": "unknown",
        "composition_feedback": "",
        "pacing_feedback": "",
        "suggested_improvements": ["Re-run review after fixing pipeline"],
        "summary": f"All providers failed: {'; '.join(errors)}",
        "_error": "; ".join(errors),
    }


# ---------------------------------------------------------------------------
# 1. Ollama provider (Gemma 4 12B — free, self-hosted, always default)
# ---------------------------------------------------------------------------


async def _review_with_ollama(video_url: str, brief: str) -> dict:
    """Review video using Ollama/Gemma 4 12B.

    Downloads the video, extracts 5 evenly-spaced frames via FFmpeg,
    base64-encodes them, and sends them to Ollama's /api/chat as images.
    """
    temp_video = None
    try:
        async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
            resp = await client.get(video_url)
            resp.raise_for_status()
            video_bytes = resp.content

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            temp_video = tmp.name

        frames_base64 = _extract_frames(temp_video, num_frames=5)

        prompt = _QA_PROMPT.format(brief=brief)

        # Build Ollama /api/chat payload with image frames
        messages = [
            {
                "role": "user",
                "content": (
                    "I will send you 5 evenly-spaced frames from a rendered video. "
                    "Please analyze the video quality based on these frames.\n\n"
                    + prompt
                ),
            }
        ]

        for i, b64 in enumerate(frames_base64):
            messages[0]["content"] += f"\n\nFrame {i+1}/5:"
            messages.append({
                "role": "user",
                "content": "",
                "images": [b64],
            })

        payload = {
            "model": _OLLAMA_MODEL,
            "messages": messages,
            "options": {"think": False},
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{_OLLAMA_BASE_URL}/api/chat",
                headers={"Content-Type": "application/json"},
                json=payload,
            )

        if resp.status_code >= 400:
            raise RuntimeError(
                f"Ollama ({_OLLAMA_MODEL}) error {resp.status_code}: {resp.text[:500]}"
            )

        data = resp.json()
        content = (data.get("message") or {}).get("content") or ""
        if not content:
            content = (data.get("message") or {}).get("thinking") or ""
        if not content:
            raise RuntimeError(f"Ollama returned empty: {data}")

        return _parse_json_response(content, f"ollama:{_OLLAMA_MODEL}")

    finally:
        if temp_video and os.path.exists(temp_video):
            try:
                os.unlink(temp_video)
            except OSError:
                pass


def _extract_frames(video_path: str, num_frames: int = 5) -> list[str]:
    """Extract evenly-spaced keyframes from a video as base64-encoded JPEGs using FFmpeg."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        duration = float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, OSError) as exc:
        raise RuntimeError(f"Failed to get video duration: {exc}") from exc

    if duration <= 0:
        raise RuntimeError(f"Invalid video duration: {duration}")

    frames: list[str] = []
    for i in range(num_frames):
        timestamp = duration * (i + 0.5) / num_frames
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", str(timestamp),
                    "-i", video_path,
                    "-vframes", "1",
                    "-q:v", "2",
                    "-f", "image2pipe",
                    "-vcodec", "mjpeg",
                    "pipe:1",
                ],
                capture_output=True, timeout=30,
            )
            if proc.returncode != 0 or not proc.stdout:
                raise RuntimeError(f"ffmpeg frame extraction failed: {proc.stderr.decode(errors='replace')[:200]}")
            frames.append(base64.b64encode(proc.stdout).decode("ascii"))
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"ffmpeg timed out at timestamp {timestamp}") from exc

    return frames


# ---------------------------------------------------------------------------
# 2. Gemini provider (File API upload + native video understanding)
# ---------------------------------------------------------------------------


def _get_gemini_api_key() -> str | None:
    return (
        os.getenv("BLENDER_GEMINI_API_KEY")
        or os.getenv("VIDEO_GEMINI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )


async def _review_with_gemini(video_url: str, brief: str, api_key: str) -> dict:
    """Download video, upload to Gemini File API, then review."""
    temp_path = None
    try:
        async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
            resp = await client.get(video_url)
            resp.raise_for_status()
            video_bytes = resp.content

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            temp_path = tmp.name

        file_uri = await _upload_to_gemini_file_api(api_key, temp_path)

        prompt = _QA_PROMPT.format(brief=brief)
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"fileData": {"mimeType": "video/mp4", "fileUri": file_uri}},
                        {"text": prompt},
                    ],
                }
            ],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3},
        }

        return await _call_gemini(api_key, payload, _GEMINI_MODEL)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


async def _upload_to_gemini_file_api(api_key: str, file_path: str) -> str:
    upload_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={api_key}"
    headers = {
        "X-Goog-Upload-Protocol": "multipart",
        "Content-Type": "multipart/related; boundary=boundary123",
    }

    with open(file_path, "rb") as f:
        video_bytes = f.read()

    metadata = json.dumps({"file": {"display_name": "video_for_review"}}).encode()
    body = (
        b"--boundary123\r\n"
        b"Content-Type: application/json\r\n\r\n" + metadata +
        b"\r\n--boundary123\r\n"
        b"Content-Type: video/mp4\r\n\r\n" + video_bytes +
        b"\r\n--boundary123--"
    )

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(upload_url, content=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    file_uri = data.get("file", {}).get("uri")
    if not file_uri:
        raise ValueError(f"Gemini File API did not return a URI. Response: {data}")

    await _wait_for_file_active(api_key, file_uri)
    return file_uri


async def _wait_for_file_active(api_key: str, file_uri: str, max_attempts: int = 30) -> None:
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
                raise ValueError("Gemini File API reported FAILED state for uploaded video")
            await asyncio.sleep(3)

    raise TimeoutError("Gemini File API video did not become ACTIVE within timeout")


async def _call_gemini(api_key: str, payload: dict, model: str) -> dict:
    url = f"{GEMINI_API_BASE}/models/{model}:generateContent?key={api_key}"

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

    return _parse_json_response(text, f"gemini:{model}")


# ---------------------------------------------------------------------------
# 3. Claude provider (direct URL, no upload needed)
# ---------------------------------------------------------------------------


async def _review_with_claude(video_url: str, brief: str, api_key: str) -> dict:
    """Send video directly to Claude via URL — no upload needed."""
    import anthropic

    prompt = _QA_PROMPT.format(brief=brief)
    client = anthropic.AsyncAnthropic(api_key=api_key)

    resp = await client.messages.create(
        model=_CLAUDE_MODEL,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "source": {"type": "url", "url": video_url},
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    text = resp.content[0].text if resp.content else ""
    return _parse_json_response(text, f"claude:{_CLAUDE_MODEL}")


# ---------------------------------------------------------------------------
# Shared JSON parser
# ---------------------------------------------------------------------------


def _parse_json_response(raw_text: str, provider: str) -> dict:
    """Parse JSON from model output, with default fallbacks."""
    cleaned = raw_text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{provider} returned non-JSON: {cleaned[:500]}") from exc

    parsed.setdefault("quality_score", 0.0)
    parsed.setdefault("brief_match_score", 0.0)
    parsed.setdefault("technical_issues", [])
    parsed.setdefault("visual_quality", "unknown")
    parsed.setdefault("composition_feedback", "")
    parsed.setdefault("pacing_feedback", "")
    parsed.setdefault("suggested_improvements", [])
    parsed.setdefault("summary", "")
    parsed["_provider"] = provider
    return parsed
