"""
Video quality review — embedding-based, no frame extraction.

Pipeline:
  1. Gemini generative model (File API) watches full video natively → analysis text
  2. gemini-embedding-2: embed(analysis_text) vs embed(brief) → cosine similarity = brief_match_score
  3. FFprobe → technical metadata (duration, resolution, fps, codec, bitrate)
  4. Ollama/Gemma 4 (text-only, no video) → final structured review JSON
  5. Fallback: parse Gemini's output directly
  6. Last resort: Claude direct URL video review

No frame extraction anywhere. Video is sent natively to Gemini File API.
Ollama is the default LLM provider (rule #4) — it only sees text here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import subprocess
import tempfile

import httpx

from tools.llm_client import _OLLAMA_BASE_URL, _OLLAMA_MODEL, _GEMINI_MODEL, _CLAUDE_MODEL

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

_GEMINI_ANALYSIS_PROMPT = """You are a professional video quality reviewer. Analyze this rendered video carefully.

Original creative brief: {brief}

First, write a detailed analysis paragraph covering: visual quality, composition, pacing, lighting (if 3D), clarity (if animation), technical issues, and how well it matches the brief.

Then return the following JSON (no markdown, no trailing commas):
{{
  "quality_score": <float 0.0-1.0>,
  "visual_quality": "<poor|fair|good|excellent>",
  "composition_feedback": "<feedback on framing, layout, camera work>",
  "pacing_feedback": "<feedback on timing, rhythm, duration>",
  "technical_issues": ["issue1", "issue2"],
  "suggested_improvements": ["improvement1", "improvement2"],
  "summary": "<2-3 sentence overall assessment>"
}}

Be critical and specific."""


async def review_video(
    video_url: str,
    brief: str,
) -> dict:
    """Review a rendered video against its creative brief.

    Pipeline:
      1. Gemini (File API) watches full video natively → analysis text + structured fields
      2. gemini-embedding-2 → embedding similarity → brief_match_score
      3. FFprobe → technical metadata
      4. Ollama/Gemma 4 (text-only) → final structured review
         If Ollama fails, return Gemini's output directly.
      5. Last resort: Claude direct URL review

    No frame extraction. Ollama sees only text, not video.
    """
    errors: list[str] = []
    gemini_key = _get_gemini_api_key()
    temp_path = None

    try:
        # --- Step 1: Download + FFprobe metadata ---
        tech_metadata: dict = {}
        try:
            async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
                resp = await client.get(video_url)
                resp.raise_for_status()
                video_bytes = resp.content

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(video_bytes)
                temp_path = tmp.name

            tech_metadata = _ffprobe_metadata(temp_path)
        except Exception as exc:
            msg = f"Download/FFprobe failed: {exc}"
            logger.warning("video_review: %s", msg)
            errors.append(msg)

        # --- Step 2: Gemini native video analysis ---
        gemini_result: dict | None = None
        if gemini_key and temp_path:
            try:
                gemini_result = await _analyze_with_gemini(temp_path, brief, gemini_key)
            except Exception as exc:
                msg = f"Gemini analysis failed: {exc}"
                logger.warning("video_review: %s", msg)
                errors.append(msg)

        # --- Step 3: Embedding similarity for brief_match_score ---
        brief_match_score = 0.0
        analysis_text = ""
        if gemini_result:
            analysis_text = gemini_result.get("_analysis_text", "")
            try:
                from tools.embedding_client import generate_embedding

                text_embed = generate_embedding(analysis_text or "", dimension=768)
                brief_text = brief or ""
                brief_embed = generate_embedding(brief_text, dimension=768)

                dot = sum(a * b for a, b in zip(text_embed, brief_embed))
                norm_a = math.sqrt(sum(x * x for x in text_embed))
                norm_b = math.sqrt(sum(x * x for x in brief_embed))
                if norm_a > 0 and norm_b > 0:
                    brief_match_score = max(0.0, min(1.0, dot / (norm_a * norm_b)))
            except Exception as exc:
                msg = f"Embedding similarity failed: {exc}"
                logger.warning("video_review: %s", msg)
                errors.append(msg)

        # --- Step 4: Ollama/Gemma 4 (text-only) final review ---
        if gemini_result and analysis_text:
            try:
                return await _review_with_ollama_text(
                    analysis_text=analysis_text,
                    brief=brief,
                    brief_match_score=brief_match_score,
                    tech_metadata=tech_metadata,
                )
            except Exception as exc:
                msg = f"Ollama text review failed: {exc}"
                logger.warning("video_review: %s", msg)
                errors.append(msg)

        # --- Step 4b: Use Gemini's structured output directly ---
        if gemini_result:
            result = dict(gemini_result)
            result.pop("_analysis_text", None)
            result["brief_match_score"] = brief_match_score
            result["_provider"] = f"gemini:{_GEMINI_MODEL}"
            result.setdefault("technical_issues", [])
            result.setdefault("suggested_improvements", [])
            _ensure_defaults(result, f"gemini:{_GEMINI_MODEL}")
            return result

        # --- Step 5: Last resort — Claude ---
        claude_key = os.getenv("ANTHROPIC_API_KEY")
        if claude_key and video_url:
            try:
                return await _review_with_claude(video_url, brief, claude_key)
            except Exception as exc:
                msg = f"Claude review failed: {exc}"
                logger.warning("video_review: %s", msg)
                errors.append(msg)

        # --- All failed ---
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
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# FFprobe metadata extraction
# ---------------------------------------------------------------------------


def _ffprobe_metadata(video_path: str) -> dict:
    """Extract technical metadata from a video file using FFprobe."""
    result: dict = {}
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration,bit_rate,size:stream=codec_name,width,height,r_frame_rate",
                "-of", "json",
                video_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0 and proc.stdout:
            data = json.loads(proc.stdout)
            fmt = data.get("format", {})
            if fmt.get("duration"):
                result["duration_sec"] = round(float(fmt["duration"]), 2)
            if fmt.get("bit_rate"):
                result["bitrate_kbps"] = round(int(fmt["bit_rate"]) / 1000)
            result["size_bytes"] = fmt.get("size")

            streams = data.get("streams", [])
            video_streams = [s for s in streams if s.get("codec_type") == "video"]
            if video_streams:
                vs = video_streams[0]
                result["codec"] = vs.get("codec_name", "")
                result["width"] = vs.get("width", 0)
                result["height"] = vs.get("height", 0)
                fps_str = vs.get("r_frame_rate", "")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    den = int(den) if den else 1
                    result["fps"] = round(int(num) / den, 2) if den else 0
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        logger.warning("FFprobe failed: %s", exc)
    return result


# ---------------------------------------------------------------------------
# Step 2: Gemini generative model — native video analysis
# ---------------------------------------------------------------------------


def _get_gemini_api_key() -> str | None:
    return (
        os.getenv("BLENDER_GEMINI_API_KEY")
        or os.getenv("VIDEO_GEMINI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )


async def _analyze_with_gemini(file_path: str, brief: str, api_key: str) -> dict:
    """Upload full video to Gemini File API and get structured analysis.

    Returns dict with _analysis_text (detailed paragraph) + structured fields.
    """
    file_uri = await _upload_to_gemini_file_api(api_key, file_path)

    prompt = _GEMINI_ANALYSIS_PROMPT.format(brief=brief)
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
        "generationConfig": {"responseMimeType": "text/plain", "temperature": 0.3},
    }

    url = f"{GEMINI_API_BASE}/models/{_GEMINI_MODEL}:generateContent?key={api_key}"
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(url, json=payload)

    if resp.status_code == 429:
        raise RuntimeError(f"Gemini 429 rate limit: {resp.text}")
    resp.raise_for_status()

    data = resp.json()
    try:
        full_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected Gemini response shape: {data}") from exc

    return _extract_analysis(full_text)


def _extract_analysis(full_text: str) -> dict:
    """Split Gemini response into analysis paragraph + structured JSON."""
    analysis_text = ""
    structured = {}

    json_start = full_text.find("{")
    json_end = full_text.rfind("}")

    if json_start != -1 and json_end > json_start:
        analysis_text = full_text[:json_start].strip()
        json_str = full_text[json_start:json_end + 1]
        try:
            structured = json.loads(json_str)
        except json.JSONDecodeError:
            analysis_text = full_text
    else:
        analysis_text = full_text

    result = structured if structured else {}
    result["_analysis_text"] = analysis_text or json.dumps(result)
    return result


# ---------------------------------------------------------------------------
# Step 4: Ollama/Gemma 4 — text-only final review
# ---------------------------------------------------------------------------


_OLLAMA_REVIEW_PROMPT = """You are a professional video quality reviewer. Based on the following analysis, generate a final quality review.

ANALYSIS:
{analysis_text}

BRIEF_MATCH_SCORE (cosine similarity): {brief_match_score}

TECHNICAL METADATA:
{tech_metadata}

Return ONLY valid JSON (no markdown):
{{
  "quality_score": <float 0.0-1.0>,
  "brief_match_score": <float 0.0-1.0>,
  "technical_issues": ["issue1"],
  "visual_quality": "<poor|fair|good|excellent>",
  "composition_feedback": "<feedback>",
  "pacing_feedback": "<feedback>",
  "suggested_improvements": ["improvement1"],
  "summary": "<2-3 sentence assessment>"
}}

Use the BRIEF_MATCH_SCORE as-is for the brief_match_score field.
Be critical and specific. quality_score < 0.6 means needs a re-render."""


async def _review_with_ollama_text(
    analysis_text: str,
    brief: str,
    brief_match_score: float,
    tech_metadata: dict,
) -> dict:
    """Send text-only review context to Ollama/Gemma 4 for final structured output."""
    meta_str = json.dumps(tech_metadata, indent=2) if tech_metadata else "None"

    prompt = _OLLAMA_REVIEW_PROMPT.format(
        analysis_text=analysis_text[:4000],
        brief_match_score=f"{brief_match_score:.4f}",
        tech_metadata=meta_str,
    )

    payload = {
        "model": _OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You output only valid JSON. No explanations."},
            {"role": "user", "content": prompt},
        ],
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

    cleaned = content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ollama returned non-JSON: {cleaned[:500]}") from exc

    parsed["brief_match_score"] = brief_match_score
    _ensure_defaults(parsed, f"ollama:{_OLLAMA_MODEL}")
    return parsed


# ---------------------------------------------------------------------------
# Claude fallback (direct URL, no upload needed)
# ---------------------------------------------------------------------------


async def _review_with_claude(video_url: str, brief: str, api_key: str) -> dict:
    """Last-resort Claude video review via direct URL."""
    import anthropic

    prompt = _GEMINI_ANALYSIS_PROMPT.format(brief=brief)
    client = anthropic.AsyncAnthropic(api_key=api_key)

    resp = await client.messages.create(
        model=_CLAUDE_MODEL,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {"type": "video", "source": {"type": "url", "url": video_url}},
                {"type": "text", "text": prompt},
            ],
        }],
    )

    text = resp.content[0].text if resp.content else ""
    return _extract_analysis(text) if text else {}


# ---------------------------------------------------------------------------
# Gemini File API helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ensure_defaults(parsed: dict, provider: str) -> None:
    parsed.setdefault("quality_score", 0.0)
    parsed.setdefault("brief_match_score", 0.0)
    parsed.setdefault("technical_issues", [])
    parsed.setdefault("visual_quality", "unknown")
    parsed.setdefault("composition_feedback", "")
    parsed.setdefault("pacing_feedback", "")
    parsed.setdefault("suggested_improvements", [])
    parsed.setdefault("summary", "")
    parsed["_provider"] = provider
