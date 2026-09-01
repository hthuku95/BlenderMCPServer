"""
Job handlers for bpy and manim renders — submitted via job_queue with
progress tracking written to Postgres at each stage.

Each handler wraps the existing generate_and_run_bpy / generate_and_run_manim
with structured stage reporting.  Called by job_queue._worker() as the
registered handler for "bpy_render" / "manim_render".
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any


async def bpy_render_handler(**kwargs: Any) -> dict[str, Any]:
    from tools.progress_store import record_job_progress
    from tools.bpy_codegen import generate_and_run_bpy
    from tools.storage import upload_render

    job_id          = kwargs.get("job_id", "")
    thread_id       = kwargs.get("workflow_thread_id", job_id)
    prompt          = kwargs.get("prompt", "")
    duration        = float(kwargs.get("duration", 10.0))
    style           = kwargs.get("style", "cinematic")
    ref_image_url   = kwargs.get("reference_image_url", "")
    narration_text  = kwargs.get("narration_text", "")
    narration_spk   = kwargs.get("narration_speaker", "Emma")
    include_narr    = kwargs.get("include_narration", False)

    await record_job_progress(
        job_id=job_id, workflow_thread_id=thread_id, tool="bpy_render",
        state="running", stage="code_generation",
        message="Generating Blender Python code via LLM",
        details={"prompt": prompt[:200], "duration": duration, "style": style},
        started_at=datetime.now(timezone.utc),
    )

    output_path = f"/tmp/bpy_job_{uuid.uuid4().hex}.mp4"
    result_path = await generate_and_run_bpy(
        prompt=prompt, duration=duration, style=style,
        output_path=output_path, reference_image_url=ref_image_url,
        thread_id=thread_id,
    )

    await record_job_progress(
        job_id=job_id, workflow_thread_id=thread_id, tool="bpy_render",
        state="running", stage="upload",
        message="Uploading render to Cloudflare R2",
        details={"output_path": result_path},
    )

    video_url = await asyncio.to_thread(upload_render, result_path, "scenes")
    response: dict[str, Any] = {
        "video_url": video_url,
        "duration": duration,
        "resolution": "1920x1080",
        "frames": int(duration * 60),
        "generation": "llm_dynamic_bpy",
    }

    if include_narr and narration_text.strip():
        try:
            from tools.vibevoice import attach_narration_assets
            response.update(
                await attach_narration_assets(
                    video_path=result_path,
                    narration_text=narration_text.strip(),
                    speaker=narration_spk, prefix="scenes",
                    metadata={"tool": "bpy_render", "style": style},
                )
            )
        except Exception as exc:
            response["narration_error"] = str(exc)

    await record_job_progress(
        job_id=job_id, workflow_thread_id=thread_id, tool="bpy_render",
        state="completed", stage="done",
        message="Blender render complete",
        details={"video_url": video_url, "duration": duration},
        result=response,
        finished_at=datetime.now(timezone.utc),
    )

    try:
        os.unlink(result_path)
        if result_path != output_path:
            os.unlink(output_path)
    except OSError:
        pass

    return response


async def manim_render_handler(**kwargs: Any) -> dict[str, Any]:
    from tools.progress_store import record_job_progress
    from tools.manim_codegen import generate_and_run_manim
    from tools.storage import upload_render

    job_id        = kwargs.get("job_id", "")
    thread_id     = kwargs.get("workflow_thread_id", job_id)
    description   = kwargs.get("description", "")
    duration      = float(kwargs.get("duration", 10.0))
    background    = kwargs.get("background", "dark")
    transparent   = bool(kwargs.get("transparent", False))
    quality       = kwargs.get("quality", "m")
    narration_text = kwargs.get("narration_text", "")
    narration_spk  = kwargs.get("narration_speaker", "Emma")
    include_narr  = kwargs.get("include_narration", False)

    await record_job_progress(
        job_id=job_id, workflow_thread_id=thread_id, tool="manim_render",
        state="running", stage="code_generation",
        message="Generating Manim Python code via LLM",
        details={"description": description[:200], "duration": duration, "quality": quality},
        started_at=datetime.now(timezone.utc),
    )

    ext = ".mov" if transparent else ".mp4"
    output_path = f"/tmp/manim_job_{uuid.uuid4().hex}{ext}"
    result_path = await generate_and_run_manim(
        description=description, duration=duration, background=background,
        output_path=output_path, transparent=transparent, quality=quality,
        thread_id=thread_id,
    )

    await record_job_progress(
        job_id=job_id, workflow_thread_id=thread_id, tool="manim_render",
        state="running", stage="upload",
        message="Uploading render to Cloudflare R2",
        details={"output_path": result_path},
    )

    video_url = await asyncio.to_thread(upload_render, result_path, "scenes")
    quality_map = {"l": "854x480", "m": "1280x720", "h": "1920x1080"}
    res = quality_map.get(quality, "1920x1080")
    response: dict[str, Any] = {
        "video_url": video_url,
        "duration": duration,
        "resolution": res,
        "frames": int(duration * 30),
        "generation": "llm_dynamic_manim",
    }

    if include_narr and narration_text.strip():
        try:
            from tools.vibevoice import attach_narration_assets
            response.update(
                await attach_narration_assets(
                    video_path=result_path,
                    narration_text=narration_text.strip(),
                    speaker=narration_spk, prefix="scenes",
                    metadata={"tool": "manim_render", "background": background},
                )
            )
        except Exception as exc:
            response["narration_error"] = str(exc)

    await record_job_progress(
        job_id=job_id, workflow_thread_id=thread_id, tool="manim_render",
        state="completed", stage="done",
        message="Manim render complete",
        details={"video_url": video_url, "duration": duration},
        result=response,
        finished_at=datetime.now(timezone.utc),
    )

    try:
        os.unlink(result_path)
        if result_path != output_path:
            os.unlink(output_path)
    except OSError:
        pass

    return response
