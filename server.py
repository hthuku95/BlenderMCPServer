"""
BlenderMCPServer — Phase 2

Exposes:
  - MCP SSE endpoint at /sse             (for Claude Desktop / other MCP clients)
  - REST endpoint at /api/call_tool      (for Rust BlenderMCPClient — single tool call)
  - REST endpoint at /api/director       (for Rust — run the LangGraph director agent)
  - Health check at /health

Run locally:
    source .venv/bin/activate
    python server.py

Deploy on Render:
    start command: python server.py   (or: xvfb-run -a python server.py)
"""

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from tools.job_queue import queue as _job_queue
from tools.progress_store import get_job_progress, get_job_progress_by_thread
from tools.rate_limiter import limiter as _limiter

load_dotenv()

MCP_API_KEY = os.getenv("MCP_API_KEY", "")
PORT = int(os.getenv("PORT", "8000"))
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP Server (for Claude Desktop / Cursor / other MCP clients)
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "BlenderMCPServer",
    instructions=(
        "AI-powered 3D animation service. Use these tools to generate Blender-rendered "
        "video clips, thumbnails, title cards, data visualisations, lower thirds, and "
        "LaTeX/Manim math animations from natural language descriptions."
    ),
)



@mcp.tool()
async def blender_execute_bpy_script(
    prompt: str,
    duration: float = 10.0,
    style: str = "cinematic",
    reference_image_url: str = "",
    include_narration: bool = False,
    narration_text: str = "",
    narration_speaker: str = "Emma",
) -> str:
    """
    Generate ANY 3D Blender scene from a natural language description.

    Unlike the old template-based tools (which only covered ~20% of Blender's
    capabilities), this tool uses LLM code generation to dynamically write and
    execute arbitrary bpy Python code — giving access to 100% of Blender's API:
    geometry nodes, physics simulations, character rigging, custom shaders,
    particle systems, Grease Pencil, camera motion, and more.

    The LLM writes the Blender Python script from scratch based on your prompt,
    validates it, runs it headless, and retries automatically on failure with
    web search debugging support (up to 5 attempts).

    This REPLACES all old template-based blender_generate_* tools in a single
    unified tool. Use this for ALL Blender 3D scene generation needs.

    Args:
        prompt: Natural language description of the scene. Be specific about:
                - Objects/shapes to include
                - Materials, colors, lighting
                - Camera angles and motion
                - Any animations or transformations
                - Scene atmosphere and mood
        duration: Target clip duration in seconds (default 10)
        style: Visual style — "cinematic", "minimal", "energetic", "calm",
               "neon", "dark", "bright", "toon"
        reference_image_url: Optional URL of a reference/inspiration image
        include_narration: If true, generate and attach VibeVoice narration
        narration_text: Custom narration text (auto-generated from prompt if empty)
        narration_speaker: VibeVoice speaker name (default "Emma")

    Returns JSON: {"video_url": str, "duration": float, "resolution": str, "frames": int}
    """
    from tools.bpy_codegen import generate_and_run_bpy
    from tools.storage import upload_render
    import uuid

    output_path = f"/tmp/bpy_scene_{uuid.uuid4().hex}.mp4"

    result_path = await generate_and_run_bpy(
        prompt=prompt,
        duration=duration,
        style=style,
        output_path=output_path,
        reference_image_url=reference_image_url,
    )

    video_url = upload_render(result_path, prefix="scenes")
    try:
        os.unlink(result_path)
    except OSError:
        pass

    response = {
        "video_url": video_url,
        "duration": duration,
        "resolution": "1920x1080",
        "frames": int(duration * 60),
        "generation": "llm_dynamic_bpy",
    }

    if include_narration:
        try:
            from tools.vibevoice import attach_narration_assets
            fallback_text = narration_text or prompt
            response.update(
                await attach_narration_assets(
                    video_path=result_path,
                    narration_text=fallback_text.strip(),
                    speaker=narration_speaker,
                    prefix="scenes",
                    metadata={"tool": "blender_execute_bpy_script", "style": style},
                )
            )
        except Exception as exc:
            response["narration_error"] = str(exc)

    return json.dumps(response)



@mcp.tool()
async def manim_execute_script(
    description: str,
    duration: float = 10.0,
    background: str = "dark",
    transparent: bool = False,
    quality: str = "m",
    include_narration: bool = False,
    narration_text: str = "",
    narration_speaker: str = "Emma",
) -> str:
    """
    Generate ANY Manim animation from a natural language description.

    Unlike the old template-based tools (which required pre-written scene files
    for each animation type), this tool uses LLM code generation to dynamically
    write and execute arbitrary Manim Python code — covering 100% of Manim's API:
    animations, 3D scenes, graphs, LaTeX, geometry, network graphs, timelines,
    code syntax highlighting, and more.

    The LLM writes the Manim scene class from scratch based on your description,
    validates it, renders it with ManimCE, and retries automatically on failure
    (up to 5 attempts).

    This REPLACES all old template-based blender_generate_* Manim tools in a single
    unified tool. Use this for ALL Manim-based animation needs.

    Args:
        description: Natural language description of the desired animation.
                     Be specific about: scene content, objects, colors, transforms,
                     camera angles, text content, mathematical expressions.
        duration: Target clip duration in seconds (default 10)
        background: Background style — "dark" | "light" | "transparent"
        transparent: If True, render with alpha channel (ProRes .mov)
        quality: Manim quality — "l" (480p) | "m" (720p) | "h" (1080p)
        include_narration: If true, generate and attach VibeVoice narration
        narration_text: Custom narration text (auto-generated from prompt if empty)
        narration_speaker: VibeVoice speaker name (default "Emma")

    Returns JSON: {"video_url": str, "duration": float, "resolution": str, "frames": int}
    """
    from tools.manim_codegen import generate_and_run_manim
    from tools.storage import upload_render
    import uuid

    ext = ".mov" if transparent else ".mp4"
    output_path = f"/tmp/manim_scene_{uuid.uuid4().hex}{ext}"

    result_path = await generate_and_run_manim(
        description=description,
        duration=duration,
        background=background,
        output_path=output_path,
        transparent=transparent,
        quality=quality,
    )

    video_url = upload_render(result_path, prefix="scenes")
    try:
        os.unlink(result_path)
    except OSError:
        pass

    quality_map = {"l": "854x480", "m": "1280x720", "h": "1920x1080"}
    res = quality_map.get(quality, "1920x1080")

    response = {
        "video_url": video_url,
        "duration": duration,
        "resolution": res,
        "frames": int(duration * 30),
        "generation": "llm_dynamic_manim",
    }

    if include_narration:
        try:
            from tools.vibevoice import attach_narration_assets
            fallback_text = narration_text or description
            response.update(
                await attach_narration_assets(
                    video_path=result_path,
                    narration_text=fallback_text.strip(),
                    speaker=narration_speaker,
                    prefix="scenes",
                    metadata={"tool": "manim_execute_script", "background": background},
                )
            )
        except Exception as exc:
            response["narration_error"] = str(exc)

    return json.dumps(response)


# ---------------------------------------------------------------------------
# Web Search tool for LLM debugging pipeline
# ---------------------------------------------------------------------------


@mcp.tool()
async def web_search(query: str) -> str:
    """Search the web for information. Used by the LLM code generation pipeline
    to look up bpy/Manim API documentation when generated code fails with
    unfamiliar errors. Also available for general use.

    Args:
        query: Natural language search query

    Returns: Search result snippets (up to 5 results)
    """
    from tools.bpy_codegen import web_search as _ws
    return await _ws(query)


@mcp.tool()
async def web_fetch(url: str) -> str:
    """Fetch a web page and return its content as clean markdown.
    Used by the LLM code generation pipeline to read API documentation,
    tutorials, or reference pages when generated code fails.

    Args:
        url: The full URL to fetch (including https://)

    Returns: Page content as clean markdown text (up to 8000 chars)
    """
    from tools.browserbase_client import browserbase_fetch
    return await browserbase_fetch(url)


# ---------------------------------------------------------------------------
# REST API (for Rust BlenderMCPClient)
# ── VIGA Investigator Tools ──────────────────────────────────────────

@mcp.tool()
async def blender_initialize_viewpoint(scene_name: str = "Scene") -> str:
    from tools.blender_investigator import initialize_viewpoint as _iv
    result = await _iv(scene_name)
    return json.dumps(result)


@mcp.tool()
async def blender_get_scene_info(scene_name: str = "Scene") -> str:
    from tools.blender_investigator import get_scene_info as _gs
    result = await _gs(scene_name)
    return json.dumps(result)


@mcp.tool()
async def blender_set_viewpoint(
    camera_name: str = "",
    location: str = "",
    rotation: str = "",
    target_object: str = "",
) -> str:
    from tools.blender_investigator import set_viewpoint as _sv
    loc = json.loads(location) if location else None
    rot = json.loads(rotation) if rotation else None
    cam = camera_name if camera_name else None
    tobj = target_object if target_object else None
    result = await _sv(cam, loc, rot, tobj)
    return json.dumps(result)


@mcp.tool()
async def blender_toggle_visibility(
    object_name: str,
    hide: bool = False,
    hide_render: bool = False,
) -> str:
    from tools.blender_investigator import toggle_visibility as _tv
    result = await _tv(object_name, hide, hide_render if hide_render else None)
    return json.dumps(result)


@mcp.tool()
async def blender_set_keyframe(
    object_name: str,
    frame: int,
    location: str = "",
    rotation: str = "",
    scale: str = "",
) -> str:
    from tools.blender_investigator import set_keyframe as _sk
    loc = json.loads(location) if location else None
    rot = json.loads(rotation) if rotation else None
    sca = json.loads(scale) if scale else None
    result = await _sk(object_name, frame, loc, rot, sca)
    return json.dumps(result)


@mcp.tool()
async def blender_investigate_object(object_name: str, detailed: bool = False) -> str:
    from tools.blender_investigator import investigate_object as _io
    result = await _io(object_name, detailed)
    return json.dumps(result)


@mcp.tool()
async def blender_investigate_render(scene_name: str = "Scene") -> str:
    from tools.blender_investigator import investigate_render as _ir
    result = await _ir(scene_name)
    return json.dumps(result)


@mcp.tool()
async def blender_save_blend_state(job_id: str) -> str:
    from tools.blender_runner import save_blend_state as _sbs
    return await _sbs(job_id)


@mcp.tool()
async def blender_undo_last_step(job_id: str) -> str:
    from tools.blender_runner import undo_last_step as _uls
    result = await _uls(job_id)
    return json.dumps(result)


@mcp.tool()
async def blender_cleanup_blend_states(job_id: str) -> str:
    from tools.blender_runner import cleanup_blend_states as _cbs
    result = await _cbs(job_id)
    return json.dumps({"success": result})


@mcp.tool()
async def blender_verifier_review(
    prompt: str,
    video_url: str,
    code: str = "",
    blender_file_path: str = "",
    iteration: int = 1,
) -> str:
    from agents.verifier import verify_and_suggest_fixes as _vsf
    result = await _vsf(
        prompt=prompt,
        video_url=video_url,
        code=code,
        blender_file_path=blender_file_path,
        previous_feedback=[{iteration: iteration}] if iteration > 1 else None,
    )
    return json.dumps(result)
# ---------------------------------------------------------------------------

async def _run_director_handler(**kwargs):
    """
    Adapt run_director for the call_tool interface.
    - When called via job queue: workflow_thread_id matches the submit() job_id,
      so progress events align with the polled job.
    - When called directly via call_tool: generates a fresh job_id internally.
    """
    from agents.director import run_director as _run_director
    brief = kwargs.get("brief", "")
    provider = kwargs.get("provider") or None
    job_id = kwargs.get("workflow_thread_id", "") or str(uuid.uuid4())
    result = await _run_director(brief, provider=provider, job_id=job_id)
    return {**result, "job_id": job_id}


TOOL_HANDLERS = {
    "blender_execute_bpy_script":     blender_execute_bpy_script,  # also registered for job queue
    "manim_execute_script":           manim_execute_script,  # also registered for job queue
    "web_search":                     web_search,  # also registered for job queue
    "web_fetch":                      web_fetch,
    "run_director":                   _run_director_handler,
    "blender_initialize_viewpoint":  blender_initialize_viewpoint,
    "blender_get_scene_info":       blender_get_scene_info,
    "blender_set_viewpoint":        blender_set_viewpoint,
    "blender_toggle_visibility":    blender_toggle_visibility,
    "blender_set_keyframe":         blender_set_keyframe,
    "blender_investigate_object":   blender_investigate_object,
    "blender_investigate_render":   blender_investigate_render,
    "blender_save_blend_state":     blender_save_blend_state,
    "blender_undo_last_step":       blender_undo_last_step,
    "blender_cleanup_blend_states": blender_cleanup_blend_states,
    "blender_verifier_review":      blender_verifier_review,
}


# Register all tools with the async job queue (skip tools with no impl_fn)
for _name, _fn in TOOL_HANDLERS.items():
    if _fn is not None:
        _job_queue.register(_name, _fn)


def _check_api_key(request: Request) -> bool:
    if not MCP_API_KEY:
        return True  # No key configured — open (dev mode)
    auth = request.headers.get("Authorization", "")
    return auth == f"Bearer {MCP_API_KEY}"


async def rest_health(request: Request) -> JSONResponse:
    from tools.llm_client import active_provider
    return JSONResponse({
        "status": "ok",
        "service": "BlenderMCPServer",
        "phase": 4,
        "tools": list(TOOL_HANDLERS),
        "llm_provider": active_provider(),
    })


async def rest_call_tool(request: Request) -> JSONResponse:
    if not _check_api_key(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    # Rate limiting — key by API key token (or IP as fallback)
    rl_key = request.headers.get("Authorization") or (request.client.host if request.client else "unknown")
    if not _limiter.allow(rl_key):
        return JSONResponse({"error": "Rate limit exceeded"}, status_code=429)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    tool_name = body.get("tool", "")
    args = body.get("args", {})
    logger.info(
        "server.call_tool_received tool=%s arg_keys=%s has_reference=%s client=%s",
        tool_name,
        sorted(args.keys()) if isinstance(args, dict) else [],
        bool(isinstance(args, dict) and args.get("reference_image_url")),
        request.client.host if request.client else "unknown",
    )

    if tool_name not in TOOL_HANDLERS:
        return JSONResponse(
            {"error": f"Unknown tool '{tool_name}'", "available": list(TOOL_HANDLERS)},
            status_code=400,
        )

    try:
        if tool_name == "blender_execute_bpy_script":
            # Route through the MCP tool handler directly
            result = await blender_execute_bpy_script(**args)
        elif tool_name == "manim_execute_script":
            result = await manim_execute_script(**args)
        elif tool_name == "web_search":
            from tools.bpy_codegen import web_search as _ws
            result = await _ws(args.get("query", ""))
        elif tool_name == "web_fetch":
            from tools.browserbase_client import browserbase_fetch
            result = await browserbase_fetch(args.get("url", ""))
        else:
            handler = TOOL_HANDLERS[tool_name]
            result = await handler(**args)
        logger.info("server.call_tool_completed tool=%s", tool_name)
        return JSONResponse({"result": result})
    except Exception as exc:
        logger.exception("server.call_tool_failed tool=%s error=%s", tool_name, exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


async def rest_director(request: Request) -> JSONResponse:
    """Run the LangGraph director agent with a high-level creative brief.

    Returns immediately with a job_id. Poll GET /api/jobs/{job_id} for progress.
    """
    if not _check_api_key(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    brief = body.get("brief", "")
    if not brief:
        return JSONResponse({"error": "'brief' field is required"}, status_code=400)

    provider = body.get("provider") or ""
    args = {"brief": brief, "provider": provider}

    try:
        job_id = await _job_queue.submit("run_director", args)
        logger.info("server.director_enqueued job_id=%s brief=%.60s", job_id, brief)
        return JSONResponse(
            {"job_id": job_id, "state": "running", "poll_url": f"/api/jobs/{job_id}"},
            status_code=202,
        )
    except Exception as exc:
        logger.exception("server.director_failed brief=%.60s", brief)
        return JSONResponse({"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Async job endpoints (Phase 5)
# ---------------------------------------------------------------------------

async def rest_submit_job(request: Request) -> JSONResponse:
    """POST /api/jobs — submit a tool call as a background job."""
    if not _check_api_key(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    rl_key = request.headers.get("Authorization") or (request.client.host if request.client else "unknown")
    if not _limiter.allow(rl_key):
        return JSONResponse({"error": "Rate limit exceeded"}, status_code=429)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    tool_name = body.get("tool", "")
    args = body.get("args", {})
    logger.info(
        "server.submit_job_received tool=%s arg_keys=%s has_reference=%s client=%s",
        tool_name,
        sorted(args.keys()) if isinstance(args, dict) else [],
        bool(isinstance(args, dict) and args.get("reference_image_url")),
        request.client.host if request.client else "unknown",
    )

    if tool_name not in TOOL_HANDLERS:
        return JSONResponse(
            {"error": f"Unknown tool '{tool_name}'", "available": list(TOOL_HANDLERS)},
            status_code=400,
        )

    try:
        job_id = await _job_queue.submit(tool_name, args)
        logger.info("server.submit_job_enqueued tool=%s job_id=%s", tool_name, job_id)
        return JSONResponse(
            {"job_id": job_id, "workflow_thread_id": job_id, "state": "pending"},
            status_code=202,
        )
    except Exception as exc:
        logger.exception("server.submit_job_failed tool=%s error=%s", tool_name, exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


async def rest_get_job(request: Request) -> JSONResponse:
    """GET /api/jobs/{job_id} — poll job status."""
    if not _check_api_key(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    job_id = request.path_params.get("job_id", "")
    status = _job_queue.get(job_id)
    progress = await get_job_progress(job_id)
    if progress is None and status is not None and status.workflow_thread_id:
        progress = await get_job_progress_by_thread(status.workflow_thread_id)

    if status is None and progress is None:
        return JSONResponse({"error": f"Job '{job_id}' not found"}, status_code=404)

    payload = status.to_dict() if status is not None else {
        "job_id": progress["job_id"],
        "tool": progress["tool"],
        "workflow_thread_id": progress["workflow_thread_id"],
        "state": progress["progress"]["state"],
        "result": progress.get("result"),
        "error": progress.get("error", ""),
        "created_at": progress.get("created_at", ""),
        "started_at": progress.get("started_at", ""),
        "finished_at": progress.get("finished_at", ""),
    }
    if progress is not None:
        payload["progress"] = progress["progress"]
        payload["progress_events"] = progress["progress_events"]
        payload["progress_persistence"] = "postgres"
    else:
        payload["progress_persistence"] = "memory"
    return JSONResponse(payload)


async def rest_list_jobs(request: Request) -> JSONResponse:
    """GET /api/jobs — list recent jobs."""
    if not _check_api_key(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    jobs = await _job_queue.list_jobs(limit=50)
    return JSONResponse({"jobs": jobs})


async def rest_cancel_job(request: Request) -> JSONResponse:
    """POST /api/jobs/{job_id}/cancel — cancel a running job."""
    if not _check_api_key(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    job_id = request.path_params.get("job_id", "")
    status = _job_queue.get(job_id)
    if status is None:
        return JSONResponse({"error": f"Job '{job_id}' not found"}, status_code=404)

    from agents.director import cancel_job as _cancel_director, is_job_cancelled

    _cancel_director(job_id)
    _job_queue.cancel(job_id)
    logger.info("server.cancel_job job_id=%s", job_id)

    return JSONResponse({"success": True, "job_id": job_id, "cancelled": True})


async def rest_jobs(request: Request) -> JSONResponse:
    """GET /api/jobs — list  |  POST /api/jobs — submit."""
    if request.method == "POST":
        return await rest_submit_job(request)
    return await rest_list_jobs(request)


async def rest_analyze_video(request: Request) -> JSONResponse:
    """
    POST /api/analyze-video — Analyze a video for viral clip moments.

    Uses BLENDER_GEMINI_API_KEY (dedicated quota, separate from the Rust app's keys).
    Called by the Rust BlenderMCPClient as a fallback when Gemini returns 429.

    Body JSON:
        video_url              — YouTube URL or R2 presigned URL
        clips_requested        — how many clips to find (default 3)
        min_duration           — minimum clip length in seconds (default 30)
        max_duration           — maximum clip length in seconds (default 90)
        high_performing_factors — optional list of viral factor hints

    Returns the VideoAnalysis JSON schema that the Rust side expects.
    """
    if not _check_api_key(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    video_url = body.get("video_url", "").strip()
    if not video_url:
        return JSONResponse({"error": "video_url is required"}, status_code=400)

    clips_requested = int(body.get("clips_requested", 3))
    min_duration = float(body.get("min_duration", 30.0))
    max_duration = float(body.get("max_duration", 90.0))
    factors = body.get("high_performing_factors", [])

    try:
        from tools.media_analyzer import analyze_video_for_clips
        result = await analyze_video_for_clips(
            video_url=video_url,
            clips_requested=clips_requested,
            min_duration=min_duration,
            max_duration=max_duration,
            high_performing_factors=factors,
        )
        return JSONResponse(result)
    except RuntimeError as exc:
        # 429 from Gemini — pass through so Rust caller can log it
        msg = str(exc)
        status = 429 if "429" in msg else 502
        return JSONResponse({"error": msg}, status_code=status)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Combined Starlette app
# ---------------------------------------------------------------------------

async def _start_queue_workers_on_startup() -> None:
    """Start the in-process job-queue workers (consumes SQS, dispatches to
    registered tool handlers). Without this, /api/jobs submissions are
    enqueued but never processed."""
    from tools.job_queue import start_job_workers as _start_workers
    count = int(os.getenv("JOB_QUEUE_WORKERS", "3"))
    await _start_workers(count)
    logger.info("Started %d job-queue worker(s)", count)


@asynccontextmanager
async def _lifespan(app):
    await _start_queue_workers_on_startup()
    yield


rest_routes = [
    Route("/health",                rest_health),
    Route("/api/call_tool",         rest_call_tool,   methods=["POST"]),
    Route("/api/director",          rest_director,    methods=["POST"]),
    Route("/api/analyze-video",     rest_analyze_video, methods=["POST"]),
    Route("/api/jobs",              rest_jobs,        methods=["GET", "POST"]),
    Route("/api/jobs/{job_id}",     rest_get_job,     methods=["GET"]),
    Route("/api/jobs/{job_id}/cancel", rest_cancel_job, methods=["POST"]),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
]

app = Starlette(
    routes=[
        *rest_routes,
        Mount("/", app=mcp.sse_app()),
    ],
    middleware=middleware,
    lifespan=_lifespan,
)


if __name__ == "__main__":
    print(f"BlenderMCPServer (Phase 5) starting on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
