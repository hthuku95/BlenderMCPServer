from __future__ import annotations

import base64
import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx


def _service_url() -> str:
    return os.getenv("VIBEVOICE_SERVICE_URL", "").rstrip("/")


def vibevoice_available() -> bool:
    return bool(_service_url())


async def synthesize_speech_to_file(
    text: str,
    speaker: str = "Emma",
    audio_format: str = "mp3",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    service_url = _service_url()
    if not service_url:
        raise RuntimeError("VIBEVOICE_SERVICE_URL is not configured")

    payload = {
        "text": text,
        "speaker": speaker,
        "format": audio_format,
        "metadata": metadata or {},
    }

    timeout = httpx.Timeout(connect=30.0, read=900.0, write=30.0, pool=30.0)
    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for attempt in range(1, 4):
            try:
                response = await client.post(f"{service_url}/api/tts/base64", json=payload)
                response.raise_for_status()
                result = response.json()
                break
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text.strip()
                if exc.response.status_code >= 500 and attempt < 3:
                    last_error = RuntimeError(
                        f"VibeVoice TTS attempt {attempt} failed with {exc.response.status_code}: {detail}"
                    )
                    await asyncio.sleep(2 * attempt)
                    continue
                raise RuntimeError(
                    f"VibeVoice TTS request failed with {exc.response.status_code}: {detail}"
                ) from exc
            except httpx.RequestError as exc:
                if attempt < 3:
                    last_error = exc
                    await asyncio.sleep(2 * attempt)
                    continue
                raise RuntimeError(f"VibeVoice TTS request error: {exc}") from exc
        else:
            raise RuntimeError(f"VibeVoice TTS failed after retries: {last_error}")

    audio_base64 = result.get("audio_base64")
    if not audio_base64:
        raise RuntimeError("VibeVoice TTS response missing audio_base64")

    audio_bytes = base64.b64decode(audio_base64)
    suffix = "." + (result.get("format") or audio_format or "mp3").lstrip(".")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="vibevoice_audio_") as tmp_file:
        tmp_file.write(audio_bytes)
        local_path = tmp_file.name

    return {
        "local_path": local_path,
        "provider": result.get("provider", "vibevoice"),
        "format": result.get("format", audio_format),
        "duration_seconds": result.get("duration_seconds"),
    }


async def attach_narration_assets(
    *,
    video_path: str,
    narration_text: str,
    speaker: str = "Emma",
    prefix: str = "renders",
    metadata: dict[str, Any] | None = None,
    mux_narration: bool = True,
) -> dict[str, Any]:
    from tools.compositor import add_audio_to_video
    from tools.storage import upload_render

    if not narration_text.strip():
        return {}

    tts_result = await synthesize_speech_to_file(
        text=narration_text,
        speaker=speaker,
        audio_format="mp3",
        metadata=metadata,
    )

    audio_path = tts_result["local_path"]
    audio_url = upload_render(audio_path, prefix=f"{prefix}_audio")

    narrated_video_url = None
    narrated_local_path = ""
    try:
        if mux_narration:
            narrated_local_path = str(Path(video_path).with_name(f"{Path(video_path).stem}_narrated.mp4"))
            narrated_local_path = add_audio_to_video(video_path, audio_path, narrated_local_path)
            narrated_video_url = upload_render(narrated_local_path, prefix=f"{prefix}_narrated")
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass
        if narrated_local_path:
            try:
                os.unlink(narrated_local_path)
            except OSError:
                pass

    return {
        "narration_text": narration_text,
        "narration_speaker": speaker,
        "narration_audio_url": audio_url,
        "narration_provider": tts_result.get("provider", "vibevoice"),
        "narration_duration_seconds": tts_result.get("duration_seconds"),
        "narrated_video_url": narrated_video_url,
    }
