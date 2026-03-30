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
    impl_generate_abstract_bg,
    impl_generate_animation,
    impl_generate_chart,
    impl_generate_code_animation,
    impl_generate_countdown,
    impl_generate_data_viz,
    impl_generate_flowchart,
    impl_generate_latex,
    impl_generate_logo_reveal,
    impl_generate_lower_third,
    impl_generate_network_graph,
    impl_generate_scene,
    impl_generate_thumbnail,
    impl_generate_timeline,
    impl_generate_title_card,
    impl_generate_3d_math,
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
    prompt: str = "",
) -> str:
    r"""
    Generate a LaTeX/Manim math equation animation clip.

    The animation is LLM-generated from a description derived from latex_expression
    and animation_type, falling back to a static template if generation fails.

    Args:
        latex_expression: LaTeX string e.g. r"\frac{d}{dt}\int_a^b f(x,t)dx"
        animation_type: "appear" | "morph" | "step_by_step" | "custom"
        duration: Clip length in seconds
        background_style: "dark" | "light" | "transparent"
        prompt: Optional natural-language description to override animation_type
                e.g. "Show each step colour-coded, highlight the discriminant in red"

    Returns JSON: {"video_url": str, "duration": float, "latex_expression": str}
    """
    return json.dumps(await impl_generate_latex(
        latex_expression, animation_type, duration, background_style, prompt
    ))


@mcp.tool()
async def blender_generate_animation(
    description: str,
    duration: float = 10.0,
    background_style: str = "dark",
    composite_over_scene: bool = True,
) -> str:
    """
    Generate ANY Manim animation from a natural language description.

    Unlike blender_generate_latex (which is constrained to equation animations),
    this tool lets you describe any animation — diagrams, kinetic text, charts,
    physics simulations, geometry proofs, code step-throughs, etc.

    The LLM generates the Manim Python code, executes it in a sandbox, and retries
    automatically on failure (up to 5 attempts).

    Args:
        description:           Natural language description of the desired animation.
                               Be specific: colours, timing, animation style, content.
                               Examples:
                               - "Show a flowchart of a software deployment process with boxes and arrows"
                               - "Animate the word INNOVATION letter by letter with each letter a different colour"
                               - "Display a 3D rotating cube with coloured faces"
                               - "Show Pythagoras theorem proof with a right triangle and coloured squares"
        duration:              Clip length in seconds (3–60)
        background_style:      "dark" | "light" | "gradient"
        composite_over_scene:  If True, composite the animation over a 3D Blender background.
                               Set False for a plain Manim background.

    Returns JSON: {"video_url": str, "duration": float, "description": str}
    """
    return json.dumps(await impl_generate_animation(
        description=description,
        duration=duration,
        background_style=background_style,
        composite_over_scene=composite_over_scene,
    ))


@mcp.tool()
async def blender_generate_chart(
    chart_type: str = "bar_chart",
    title: str = "Data Visualisation",
    data: str = "[3, 7, 5, 9, 4, 6]",
    labels: str = '["A","B","C","D","E","F"]',
    duration: float = 10.0,
    y_range: str = "[0, 12, 2]",
    colors: str = "[]",
) -> str:
    """
    Generate an animated Manim data visualisation clip.

    Args:
        chart_type: "bar_chart" | "line_chart" | "pie_chart" | "counter" | "scatter"
        title:      Chart heading displayed at top
        data:       JSON array of numbers (y-values for bar/line/scatter; segment values for pie)
                    For scatter: JSON array of [x, y] pairs e.g. "[[1,2],[3,4],[5,3]]"
                    For counter: single-element array with the target value e.g. "[1000000]"
        labels:     JSON array of strings (bar names / axis labels / pie segment names)
        duration:   Clip length in seconds
        y_range:    JSON array [min, max, step] — used by bar/line/scatter
        colors:     JSON array of colour names e.g. '["BLUE","RED","GREEN"]'

    Returns JSON: {"video_url": str, "duration": float, "chart_type": str}
    """
    _json = __import__("json")
    return json.dumps(await impl_generate_chart(
        chart_type=chart_type,
        title=title,
        data=_json.loads(data),
        labels=_json.loads(labels),
        duration=duration,
        y_range=_json.loads(y_range),
        colors=_json.loads(colors),
    ))


@mcp.tool()
async def blender_generate_flowchart(
    nodes: str = "[]",
    edges: str = "[]",
    title: str = "Process Flowchart",
    duration: float = 12.0,
    style: str = "dark",
) -> str:
    """
    Generate an animated Manim flowchart with process boxes, decision diamonds, and arrows.

    Args:
        nodes:    JSON array of node objects: [{"id":"start","label":"Start","type":"start"},
                  {"id":"step1","label":"Process Data","type":"process"},
                  {"id":"decide","label":"Valid?","type":"decision"},...]
                  type: "start" | "process" | "decision" | "end"
        edges:    JSON array of connections: [{"from":"start","to":"step1"},
                  {"from":"decide","to":"step2","label":"Yes"},...]
        title:    Chart heading
        duration: Clip length in seconds
        style:    "dark" | "light" | "blue"

    Returns JSON: {"video_url": str, "duration": float, "title": str}
    """
    _j = __import__("json")
    return json.dumps(await impl_generate_flowchart(
        nodes=_j.loads(nodes), edges=_j.loads(edges),
        title=title, duration=duration, style=style,
    ))


@mcp.tool()
async def blender_generate_3d_math(
    scene_type: str = "surface",
    title: str = "3D Mathematics",
    function: str = "wave",
    duration: float = 12.0,
    color: str = "BLUE",
) -> str:
    """
    Generate a 3D mathematics animation using Manim's ThreeDScene.

    Args:
        scene_type: "surface" (3D function surface), "curve" (parametric helix),
                    "vector_field" (2D arrow vector field), "torus" (spinning torus)
        title:      Title text displayed on screen
        function:   For scene_type="surface": "wave" | "sin" | "cos" | "saddle" | "paraboloid" | "ripple"
        duration:   Clip length in seconds
        color:      Manim colour name: "BLUE" | "RED" | "GREEN" | "GOLD" | "PURPLE" | "TEAL"

    Returns JSON: {"video_url": str, "duration": float, "scene_type": str}
    """
    return json.dumps(await impl_generate_3d_math(
        scene_type=scene_type, title=title, function=function,
        duration=duration, color=color,
    ))


@mcp.tool()
async def blender_generate_code_animation(
    code: str = "",
    language: str = "python",
    title: str = "Code Walkthrough",
    highlight_lines: str = "[]",
    reveal_mode: str = "line_by_line",
    duration: float = 12.0,
    style: str = "monokai",
) -> str:
    """
    Generate an animated code syntax-highlighting clip — ideal for tech tutorials.

    Args:
        code:            The source code string to display and animate
        language:        Syntax highlighting language: "python" | "javascript" | "rust" |
                         "cpp" | "java" | "bash" | "sql" | "typescript" | "go"
        title:           Heading shown above the code block
        highlight_lines: JSON array of 1-indexed line numbers to highlight after reveal
                         e.g. "[3, 7, 11]"
        reveal_mode:     "line_by_line" (default) | "all_at_once" | "block"
        duration:        Clip length in seconds
        style:           Syntax theme: "monokai" | "dracula" | "solarized-dark"

    Returns JSON: {"video_url": str, "duration": float, "language": str}
    """
    _j = __import__("json")
    return json.dumps(await impl_generate_code_animation(
        code=code, language=language, title=title,
        highlight_lines=_j.loads(highlight_lines),
        reveal_mode=reveal_mode, duration=duration, style=style,
    ))


@mcp.tool()
async def blender_generate_timeline(
    events: str = "[]",
    title: str = "Project Timeline",
    duration: float = 12.0,
    style: str = "dark",
    orientation: str = "horizontal",
) -> str:
    """
    Generate an animated timeline / roadmap / Gantt-style clip.

    Args:
        events:      JSON array of event objects:
                     [{"date":"Jan","label":"Project Kickoff","color":"BLUE"},
                      {"date":"Mar","label":"MVP Launch","color":"GREEN"},...]
                     color: "BLUE"|"RED"|"GREEN"|"YELLOW"|"ORANGE"|"PURPLE"|"TEAL"|"GOLD"
        title:       Heading text
        duration:    Clip length in seconds
        style:       "dark" | "light" | "gradient"
        orientation: "horizontal" (default) | "vertical"

    Returns JSON: {"video_url": str, "duration": float, "title": str}
    """
    _j = __import__("json")
    return json.dumps(await impl_generate_timeline(
        events=_j.loads(events), title=title, duration=duration,
        style=style, orientation=orientation,
    ))


@mcp.tool()
async def blender_generate_network_graph(
    nodes: str = "[]",
    edges: str = "[]",
    title: str = "Network Graph",
    layout: str = "radial",
    duration: float = 12.0,
    style: str = "dark",
) -> str:
    """
    Generate an animated network / knowledge graph with nodes and edges.

    Args:
        nodes:   JSON array: [{"id":"A","label":"Machine Learning","color":"BLUE"},...]
                 color: "BLUE"|"RED"|"GREEN"|"YELLOW"|"ORANGE"|"PURPLE"|"TEAL"|"GOLD"|"PINK"
                 size:  float (optional, default 0.35) — node circle radius
        edges:   JSON array: [{"from":"A","to":"B"},{"from":"A","to":"C","label":"uses","directed":true},...]
        title:   Heading text
        layout:  "radial" (hub-and-spoke) | "circular" | "spring"
        duration: Clip length in seconds
        style:   "dark" | "neon"

    Returns JSON: {"video_url": str, "duration": float, "title": str}
    """
    _j = __import__("json")
    return json.dumps(await impl_generate_network_graph(
        nodes=_j.loads(nodes), edges=_j.loads(edges),
        title=title, layout=layout, duration=duration, style=style,
    ))


@mcp.tool()
async def blender_generate_logo_reveal(
    text: str = "BRAND",
    tagline: str = "",
    style: str = "extrude_reveal",
    color: str = "[0.1, 0.5, 1.0, 1.0]",
    bg_color: str = "[0.02, 0.02, 0.05, 1.0]",
    duration: float = 6.0,
) -> str:
    """
    Generate a 3D extruded text / logo reveal animation in Blender.

    Args:
        text:     Brand name or main text to extrude and reveal
        tagline:  Optional second line (smaller text below)
        style:    "extrude_reveal" (Z-scale grow-in, default) | "zoom_in" | "split" | "typewriter"
        color:    JSON RGBA float array for text material e.g. "[0.1, 0.5, 1.0, 1.0]"
        bg_color: JSON RGBA float array for background e.g. "[0.02, 0.02, 0.05, 1.0]"
        duration: Clip length in seconds

    Returns JSON: {"video_url": str, "duration": float, "text": str, "style": str}
    """
    _j = __import__("json")
    return json.dumps(await impl_generate_logo_reveal(
        text=text, tagline=tagline, style=style,
        color=_j.loads(color), bg_color=_j.loads(bg_color),
        duration=duration,
    ))


@mcp.tool()
async def blender_generate_abstract_bg(
    style: str = "geometric",
    primary_color: str = "[0.05, 0.2, 0.8, 1.0]",
    secondary_color: str = "[0.8, 0.1, 0.5, 1.0]",
    duration: float = 8.0,
) -> str:
    """
    Generate an animated abstract background loop for use as a video backdrop or overlay.

    Args:
        style:           "geometric" (orbiting shapes, default) | "waves" | "particles" |
                         "grid" (retro neon wireframe) | "gradient"
        primary_color:   JSON RGBA float array e.g. "[0.05, 0.2, 0.8, 1.0]"
        secondary_color: JSON RGBA float array e.g. "[0.8, 0.1, 0.5, 1.0]"
        duration:        Clip length in seconds

    Returns JSON: {"video_url": str, "duration": float, "style": str}
    """
    _j = __import__("json")
    return json.dumps(await impl_generate_abstract_bg(
        style=style,
        primary_color=_j.loads(primary_color),
        secondary_color=_j.loads(secondary_color),
        duration=duration,
    ))


@mcp.tool()
async def blender_generate_countdown(
    start_number: int = 5,
    end_number: int = 1,
    style: str = "bold",
    color: str = "[0.1, 0.6, 1.0, 1.0]",
    bg_color: str = "[0.02, 0.02, 0.05, 1.0]",
    show_ring: bool = True,
    duration: float = 0.0,
) -> str:
    """
    Generate a 3D animated countdown timer in Blender.

    Args:
        start_number: Count from this number (e.g. 10)
        end_number:   Count to this number (e.g. 1 or 0)
        style:        "bold" | "neon" | "minimal" | "cinematic"
        color:        JSON RGBA float array for number material
        bg_color:     JSON RGBA float array for background
        show_ring:    If true, adds an animated rotating ring around the number
        duration:     Total clip duration in seconds (0 = auto: 1 second per count)

    Returns JSON: {"video_url": str, "duration": float, "start_number": int, "end_number": int}
    """
    _j = __import__("json")
    dur = float(duration) if duration > 0 else None
    return json.dumps(await impl_generate_countdown(
        start_number=start_number, end_number=end_number,
        style=style, color=_j.loads(color), bg_color=_j.loads(bg_color),
        show_ring=show_ring, duration=dur,
    ))


# ---------------------------------------------------------------------------
# REST API (for Rust BlenderMCPClient)
# ---------------------------------------------------------------------------

TOOL_HANDLERS = {
    "blender_generate_scene":         impl_generate_scene,
    "blender_generate_thumbnail":     impl_generate_thumbnail,
    "blender_generate_title_card":    impl_generate_title_card,
    "blender_generate_data_viz":      impl_generate_data_viz,
    "blender_generate_lower_third":   impl_generate_lower_third,
    "blender_generate_latex":         impl_generate_latex,
    "blender_generate_ui_mockup":     impl_generate_ui_mockup,
    "blender_generate_animation":     impl_generate_animation,
    "blender_generate_chart":         impl_generate_chart,
    "blender_generate_flowchart":     impl_generate_flowchart,
    "blender_generate_3d_math":       impl_generate_3d_math,
    "blender_generate_code_animation": impl_generate_code_animation,
    "blender_generate_timeline":      impl_generate_timeline,
    "blender_generate_network_graph": impl_generate_network_graph,
    "blender_generate_logo_reveal":   impl_generate_logo_reveal,
    "blender_generate_abstract_bg":   impl_generate_abstract_bg,
    "blender_generate_countdown":     impl_generate_countdown,
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
