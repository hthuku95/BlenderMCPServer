"""
BlenderMCPServer — Phase 1

Exposes:
  - MCP SSE endpoint at /sse  (for Claude Desktop / other MCP clients)
  - REST endpoint at /api/call_tool  (for Rust video_editor BlenderMCPClient)
  - Health check at /health

Run locally:
    source .venv/bin/activate
    python server.py

Deploy on Render:
    start command: python server.py
"""

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

load_dotenv()

MCP_API_KEY = os.getenv("MCP_API_KEY", "")
PORT = int(os.getenv("PORT", "8000"))

# ---------------------------------------------------------------------------
# Core render implementation (shared by MCP tools and REST endpoint)
# ---------------------------------------------------------------------------

async def _impl_generate_scene(
    prompt: str,
    duration: float,
    style: str,
    reference_image_url: str = "",
) -> dict:
    from tools.blender_runner import run_blender_script_with_retry
    from tools.storage import upload_render

    script_path = Path(__file__).parent / "blender_scripts" / "base_scene.py"
    output_path = f"/tmp/blender_{uuid.uuid4().hex}.mp4"

    args = {
        "prompt": prompt,
        "duration": duration,
        "style": style,
        "output_path": output_path,
    }
    if reference_image_url:
        args["reference_image_url"] = reference_image_url

    result = await run_blender_script_with_retry(
        script_content=script_path.read_text(),
        args=args,
        max_attempts=3,
        timeout=600,
    )

    video_url = upload_render(output_path, prefix="scenes")

    # Clean up temp file
    try:
        os.unlink(output_path)
    except OSError:
        pass

    return {
        "video_url": video_url,
        "duration": result.get("duration", duration),
        "resolution": result.get("resolution", "1920x1080"),
        "frames": result.get("frames", int(duration * 24)),
    }


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
        prompt: Natural language description of the scene (e.g. "cinematic ocean at sunset, calm mood")
        duration: Target clip duration in seconds (default 10)
        style: Visual style — "cinematic", "minimal", "energetic", or "calm"
        reference_image_url: Optional URL of a reference/inspiration image

    Returns JSON: {"video_url": str, "duration": float, "resolution": str, "frames": int}
    """
    result = await _impl_generate_scene(prompt, duration, style, reference_image_url)
    return json.dumps(result)


@mcp.tool()
async def blender_generate_thumbnail(
    prompt: str,
    title_text: str = "",
    style: str = "youtube",
) -> str:
    """
    Generate a 3D rendered YouTube thumbnail image (1280x720 PNG).

    Args:
        prompt: Scene description (e.g. "tech startup success, dark background, neon blue accents")
        title_text: Optional text to overlay on the thumbnail
        style: "youtube" | "cinematic" | "minimal"

    Returns JSON: {"image_url": str, "width": int, "height": int}
    """
    # Phase 2 — stub for now; Blender thumbnail script will be added
    return json.dumps({
        "image_url": "",
        "width": 1280,
        "height": 720,
        "status": "Phase 2 — not yet implemented",
    })


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
        subtitle: Secondary/tagline text (optional)
        duration: Clip length in seconds (3–8 recommended)
        style: Visual style description

    Returns JSON: {"video_url": str}
    """
    return json.dumps({"video_url": "", "status": "Phase 2 — not yet implemented"})


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
        chart_type: "bar" | "line" | "pie" | "globe"
        title: Chart title overlay text
        duration: Animation length in seconds

    Returns JSON: {"video_url": str}
    """
    return json.dumps({"video_url": "", "status": "Phase 2 — not yet implemented"})


@mcp.tool()
async def blender_generate_lower_third(
    name_text: str,
    subtitle_text: str = "",
    style: str = "modern",
    duration: float = 5.0,
) -> str:
    """
    Generate an animated lower-third text overlay clip (transparent background).

    Args:
        name_text: Primary text (e.g. person name or topic)
        subtitle_text: Secondary text (e.g. job title or context)
        style: Animation/colour style
        duration: Display duration in seconds

    Returns JSON: {"video_url": str}
    """
    return json.dumps({"video_url": "", "status": "Phase 2 — not yet implemented"})


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
        latex_expression: LaTeX expression e.g. r"\frac{d}{dt}\int_a^b f(x,t)dx"
        animation_type: "appear" | "morph" | "step_by_step"
        duration: Clip length in seconds
        background_style: "dark" | "light" | "transparent"

    Returns JSON: {"video_url": str}
    """
    return json.dumps({"video_url": "", "status": "Phase 3 — Manim not yet implemented"})


# ---------------------------------------------------------------------------
# REST API (for Rust BlenderMCPClient)
# ---------------------------------------------------------------------------

TOOL_HANDLERS = {
    "blender_generate_scene": _impl_generate_scene,
}


def _check_api_key(request: Request) -> bool:
    if not MCP_API_KEY:
        return True  # No key configured — open (dev mode)
    auth = request.headers.get("Authorization", "")
    return auth == f"Bearer {MCP_API_KEY}"


async def rest_health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "BlenderMCPServer"})


async def rest_call_tool(request: Request) -> JSONResponse:
    if not _check_api_key(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    tool_name = body.get("tool", "")
    args = body.get("args", {})

    if tool_name not in TOOL_HANDLERS:
        return JSONResponse(
            {"error": f"Unknown tool '{tool_name}'. Available: {list(TOOL_HANDLERS.keys())}"},
            status_code=400,
        )

    try:
        result = await TOOL_HANDLERS[tool_name](**args)
        return JSONResponse({"result": result})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Combined Starlette app
# ---------------------------------------------------------------------------

rest_routes = [
    Route("/health", rest_health),
    Route("/api/call_tool", rest_call_tool, methods=["POST"]),
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
    print(f"BlenderMCPServer starting on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
