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
import os

import uvicorn
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from tools.render_tools import (
    impl_generate_data_viz,
    impl_generate_latex,
    impl_generate_lower_third,
    impl_generate_scene,
    impl_generate_thumbnail,
    impl_generate_title_card,
    impl_generate_ui_mockup,
)
from tools.job_queue import queue as _job_queue
from tools.rate_limiter import limiter as _limiter

load_dotenv()

MCP_API_KEY = os.getenv("MCP_API_KEY", "")
PORT = int(os.getenv("PORT", "8000"))

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
async def blender_generate_scene(
    prompt: str,
    duration: float = 10.0,
    style: str = "cinematic",
    reference_image_url: str = "",
) -> str:
    """
    Generate a procedural 3D Blender scene as an MP4 clip.

    Args:
        prompt: Natural language description of the scene
        duration: Target clip duration in seconds (default 10)
        style: Visual style — "cinematic", "minimal", "energetic", or "calm"
        reference_image_url: Optional URL of a reference/inspiration image

    Returns JSON: {"video_url": str, "duration": float, "resolution": str, "frames": int}
    """
    return json.dumps(await impl_generate_scene(prompt, duration, style, reference_image_url))


@mcp.tool()
async def blender_generate_thumbnail(
    prompt: str,
    title_text: str = "",
    style: str = "youtube",
) -> str:
    """
    Generate a 3D rendered YouTube thumbnail image (1280×720 PNG).

    Args:
        prompt: Scene description
        title_text: Optional text to embed in the 3D scene
        style: "youtube" | "cinematic" | "minimal"

    Returns JSON: {"image_url": str, "width": int, "height": int}
    """
    return json.dumps(await impl_generate_thumbnail(prompt, title_text, style))


@mcp.tool()
async def blender_generate_title_card(
    title: str,
    subtitle: str = "",
    duration: float = 5.0,
    style: str = "cinematic",
) -> str:
    """
    Generate an animated 3D title card as an MP4 clip.

    Args:
        title: Main title text
        subtitle: Secondary text (optional)
        duration: Clip length in seconds (3–8 recommended)
        style: "cinematic" | "minimal" | "bold"

    Returns JSON: {"video_url": str, "duration": float}
    """
    return json.dumps(await impl_generate_title_card(title, subtitle, duration, style))


@mcp.tool()
async def blender_generate_data_viz(
    data_json: str,
    chart_type: str = "bar",
    title: str = "",
    duration: float = 10.0,
) -> str:
    """
    Generate an animated 3D data visualisation clip.

    Args:
        data_json: JSON array of data points e.g. '[{"label":"A","value":42},...]'
        chart_type: "bar" (line/pie reserved for future phases)
        title: Chart title overlay text
        duration: Animation length in seconds

    Returns JSON: {"video_url": str, "duration": float, "chart_type": str}
    """
    return json.dumps(await impl_generate_data_viz(data_json, chart_type, title, duration))


@mcp.tool()
async def blender_generate_lower_third(
    name_text: str,
    subtitle_text: str = "",
    style: str = "modern",
    duration: float = 5.0,
) -> str:
    """
    Generate an animated lower-third text overlay clip (green-screen background for keying).

    Args:
        name_text: Primary text (person name / topic)
        subtitle_text: Secondary text (job title / context)
        style: "modern" | "minimal" | "bold"
        duration: Display duration in seconds

    Returns JSON: {"video_url": str, "duration": float, "keying": "green_screen"}
    """
    return json.dumps(await impl_generate_lower_third(name_text, subtitle_text, style, duration))


@mcp.tool()
async def blender_generate_ui_mockup(
    device: str = "iphone",
    animation: str = "reveal",
    duration: float = 6.0,
    screenshot_url: str = "",
    screenshot_spec: str = "",
    background_color: str = "",
    accent_color: str = "",
) -> str:
    """
    Render a screenshot inside a 3D device frame (iPhone, MacBook, browser, iPad).

    Args:
        device: "iphone" | "macbook" | "browser" | "ipad"
        animation: "static" (PNG) | "reveal" (fade-in) | "scroll" | "tilt"
        duration: Clip length in seconds (ignored for static)
        screenshot_url: URL of screenshot to place on the device screen
        screenshot_spec: JSON design spec to auto-generate a screenshot
                         e.g. '{"type":"browser","url":"https://myapp.com","title":"My App",
                               "body":"...","bg_color":"#fff","accent_color":"#0070f3"}'
        background_color: JSON RGB float array e.g. "[0.05, 0.05, 0.08]"
        accent_color: JSON RGB float array e.g. "[0.3, 0.5, 1.0]"

    Returns JSON: {"video_url": str, "device": str, "animation": str, "duration": float}
               or {"image_url": str, "device": str, "animation": "static"}
    """
    import json as _json
    spec = _json.loads(screenshot_spec) if screenshot_spec else None
    bg   = _json.loads(background_color) if background_color else None
    acc  = _json.loads(accent_color)     if accent_color else None
    return json.dumps(await impl_generate_ui_mockup(
        device=device,
        animation=animation,
        duration=duration,
        screenshot_url=screenshot_url,
        screenshot_spec=spec,
        background_color=bg,
        accent_color=acc,
    ))


@mcp.tool()
async def blender_generate_latex(
    latex_expression: str,
    animation_type: str = "appear",
    duration: float = 8.0,
    background_style: str = "dark",
) -> str:
    r"""
    Generate a LaTeX/Manim math equation animation clip.

    Args:
        latex_expression: LaTeX string e.g. r"\frac{d}{dt}\int_a^b f(x,t)dx"
        animation_type: "appear" | "morph" | "step_by_step"
        duration: Clip length in seconds
        background_style: "dark" | "light" | "transparent"

    Returns JSON: {"video_url": str, "duration": float, "latex_expression": str}
    """
    return json.dumps(await impl_generate_latex(latex_expression, animation_type, duration, background_style))


# ---------------------------------------------------------------------------
# REST API (for Rust BlenderMCPClient)
# ---------------------------------------------------------------------------

TOOL_HANDLERS = {
    "blender_generate_scene":       impl_generate_scene,
    "blender_generate_thumbnail":   impl_generate_thumbnail,
    "blender_generate_title_card":  impl_generate_title_card,
    "blender_generate_data_viz":    impl_generate_data_viz,
    "blender_generate_lower_third": impl_generate_lower_third,
    "blender_generate_latex":       impl_generate_latex,
    "blender_generate_ui_mockup":   impl_generate_ui_mockup,
}


# Register all tools with the async job queue
for _name, _fn in TOOL_HANDLERS.items():
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

    if tool_name not in TOOL_HANDLERS:
        return JSONResponse(
            {"error": f"Unknown tool '{tool_name}'", "available": list(TOOL_HANDLERS)},
            status_code=400,
        )

    try:
        result = await TOOL_HANDLERS[tool_name](**args)
        return JSONResponse({"result": result})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


async def rest_director(request: Request) -> JSONResponse:
    """Run the LangGraph director agent with a high-level creative brief."""
    if not _check_api_key(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    brief = body.get("brief", "")
    if not brief:
        return JSONResponse({"error": "'brief' field is required"}, status_code=400)

    # Optional: caller can force a provider ("gemini" | "claude" | "auto")
    provider = body.get("provider") or None

    try:
        from agents.director import run_director
        result = await run_director(brief, provider=provider)
        return JSONResponse(result)
    except Exception as exc:
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

    if tool_name not in TOOL_HANDLERS:
        return JSONResponse(
            {"error": f"Unknown tool '{tool_name}'", "available": list(TOOL_HANDLERS)},
            status_code=400,
        )

    try:
        job_id = await _job_queue.submit(tool_name, args)
        return JSONResponse({"job_id": job_id, "state": "pending"}, status_code=202)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


async def rest_get_job(request: Request) -> JSONResponse:
    """GET /api/jobs/{job_id} — poll job status."""
    if not _check_api_key(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    job_id = request.path_params.get("job_id", "")
    status = _job_queue.get(job_id)
    if status is None:
        return JSONResponse({"error": f"Job '{job_id}' not found"}, status_code=404)
    return JSONResponse(status.to_dict())


async def rest_list_jobs(request: Request) -> JSONResponse:
    """GET /api/jobs — list recent jobs."""
    if not _check_api_key(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return JSONResponse({"jobs": _job_queue.list_jobs(limit=50)})


async def rest_jobs(request: Request) -> JSONResponse:
    """GET /api/jobs — list  |  POST /api/jobs — submit."""
    if request.method == "POST":
        return await rest_submit_job(request)
    return await rest_list_jobs(request)


# ---------------------------------------------------------------------------
# Combined Starlette app
# ---------------------------------------------------------------------------

rest_routes = [
    Route("/health",                rest_health),
    Route("/api/call_tool",         rest_call_tool,   methods=["POST"]),
    Route("/api/director",          rest_director,    methods=["POST"]),
    # Phase 5 — async job queue
    Route("/api/jobs",              rest_jobs,        methods=["GET", "POST"]),
    Route("/api/jobs/{job_id}",     rest_get_job,     methods=["GET"]),
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
)


if __name__ == "__main__":
    print(f"BlenderMCPServer (Phase 5) starting on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
