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
    impl_generate_geometry_proof,
    impl_generate_latex,
    impl_generate_logo_reveal,
    impl_generate_lower_third,
    impl_generate_matrix_transform,
    impl_generate_network_graph,
    impl_generate_polar_graph,
    impl_generate_scene,
    impl_generate_text_animation,
    impl_generate_thumbnail,
    impl_generate_timeline,
    impl_generate_title_card,
    impl_generate_3d_math,
    impl_generate_ui_mockup,
    impl_generate_vector_field,
    impl_generate_particle_confetti,
    impl_generate_rigid_body_drop,
    impl_generate_camera_path,
    impl_generate_toon_scene,
    impl_generate_grease_pencil_reveal,
    impl_generate_geometry_scatter,
)
from tools.job_queue import queue as _job_queue
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
async def blender_generate_scene(
    prompt: str,
    duration: float = 10.0,
    style: str = "cinematic",
    reference_image_url: str = "",
    include_narration: bool = False,
    narration_text: str = "",
    narration_speaker: str = "Emma",
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
    return json.dumps(await impl_generate_scene(
        prompt,
        duration,
        style,
        reference_image_url,
        include_narration,
        narration_text,
        narration_speaker,
    ))


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
    include_narration: bool = False,
    narration_text: str = "",
    narration_speaker: str = "Emma",
    workflow_thread_id: str = "",
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
        latex_expression,
        animation_type,
        duration,
        background_style,
        prompt,
        include_narration,
        narration_text,
        narration_speaker,
        workflow_thread_id,
    ))


@mcp.tool()
async def blender_generate_animation(
    description: str,
    duration: float = 10.0,
    background_style: str = "dark",
    composite_over_scene: bool = True,
    include_narration: bool = False,
    narration_text: str = "",
    narration_speaker: str = "Emma",
    workflow_thread_id: str = "",
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
        include_narration=include_narration,
        narration_text=narration_text,
        narration_speaker=narration_speaker,
        workflow_thread_id=workflow_thread_id,
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
    workflow_thread_id: str = "",
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
        workflow_thread_id=workflow_thread_id,
    ))


@mcp.tool()
async def blender_generate_flowchart(
    nodes: str = "[]",
    edges: str = "[]",
    title: str = "Process Flowchart",
    duration: float = 12.0,
    style: str = "dark",
    workflow_thread_id: str = "",
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
        workflow_thread_id=workflow_thread_id,
    ))


@mcp.tool()
async def blender_generate_3d_math(
    scene_type: str = "surface",
    title: str = "3D Mathematics",
    function: str = "wave",
    duration: float = 12.0,
    color: str = "BLUE",
    workflow_thread_id: str = "",
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
        workflow_thread_id=workflow_thread_id,
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
    workflow_thread_id: str = "",
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
        workflow_thread_id=workflow_thread_id,
    ))


@mcp.tool()
async def blender_generate_timeline(
    events: str = "[]",
    title: str = "Project Timeline",
    duration: float = 12.0,
    style: str = "dark",
    orientation: str = "horizontal",
    workflow_thread_id: str = "",
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
        workflow_thread_id=workflow_thread_id,
    ))


@mcp.tool()
async def blender_generate_network_graph(
    nodes: str = "[]",
    edges: str = "[]",
    title: str = "Network Graph",
    layout: str = "radial",
    duration: float = 12.0,
    style: str = "dark",
    workflow_thread_id: str = "",
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
        workflow_thread_id=workflow_thread_id,
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


@mcp.tool()
async def blender_generate_text_animation(
    text: str = "Make it Count",
    subtitle: str = "",
    mode: str = "letter_by_letter",
    color: str = "WHITE",
    bg_color: str = "dark",
    duration: float = 8.0,
    font_size: int = 72,
    words_to_highlight: str = "[]",
) -> str:
    """
    Generate kinetic typography / text animation using Manim.

    Args:
        text:               Main text to animate
        subtitle:           Optional smaller second line
        mode:               "letter_by_letter" | "word_by_word" | "typewriter" | "wave" |
                            "zoom_burst" | "spin_in" | "color_cycle" | "highlight_words"
        color:              Text colour: "WHITE" | "BLUE" | "RED" | "GREEN" | "GOLD" | "YELLOW"
        bg_color:           "dark" | "light"
        duration:           Clip length in seconds
        font_size:          Font size in points (default 72)
        words_to_highlight: JSON array of words to highlight (for highlight_words mode)

    Returns JSON: {"video_url": str, "duration": float, "mode": str}
    """
    _j = __import__("json")
    return json.dumps(await impl_generate_text_animation(
        text=text, subtitle=subtitle, mode=mode, color=color,
        bg_color=bg_color, duration=duration, font_size=font_size,
        words_to_highlight=_j.loads(words_to_highlight),
    ))


@mcp.tool()
async def blender_generate_vector_field(
    field_type: str = "rotation",
    title: str = "Vector Field",
    duration: float = 12.0,
    show_streams: bool = True,
    color: str = "BLUE",
    style: str = "dark",
) -> str:
    """
    Generate an animated vector field / flow visualization using Manim.

    Args:
        field_type:   "rotation" (curl around origin) | "radial" (outward) | "sink" (inward) |
                      "saddle" | "curl" | "gravity"
        title:        Heading text
        duration:     Clip length in seconds
        show_streams: If true, renders StreamLines on top of arrow field
        color:        Arrow/stream colour: "BLUE" | "RED" | "GREEN" | "TEAL" | "ORANGE"
        style:        "dark" | "grid" (with NumberPlane background)

    Returns JSON: {"video_url": str, "duration": float, "field_type": str}
    """
    return json.dumps(await impl_generate_vector_field(
        field_type=field_type, title=title, duration=duration,
        show_streams=show_streams, color=color, style=style,
    ))


@mcp.tool()
async def blender_generate_matrix_transform(
    matrix: str = "[[0,-1],[1,0]]",
    title: str = "Linear Transformation",
    duration: float = 12.0,
    show_vectors: bool = True,
    show_det: bool = True,
) -> str:
    """
    Generate a linear algebra matrix transformation animation using Manim's LinearTransformationScene.

    Args:
        matrix:       JSON 2×2 matrix e.g. "[[0,-1],[1,0]]" (90° rotation),
                      "[[2,0],[0,2]]" (scaling), "[[1,1],[0,1]]" (shear)
        title:        Heading text
        duration:     Clip length in seconds
        show_vectors: If true, shows sample vectors being transformed
        show_det:     If true, shows determinant annotation

    Returns JSON: {"video_url": str, "duration": float}
    """
    _j = __import__("json")
    return json.dumps(await impl_generate_matrix_transform(
        matrix=_j.loads(matrix), title=title, duration=duration,
        show_vectors=show_vectors, show_det=show_det,
    ))


@mcp.tool()
async def blender_generate_polar_graph(
    plane_type: str = "polar",
    title: str = "Polar Graph",
    function: str = "rose",
    k_value: int = 4,
    duration: float = 12.0,
    color: str = "BLUE",
    show_label: bool = True,
) -> str:
    """
    Generate a polar coordinate / complex plane / function graph animation using Manim.

    Args:
        plane_type: "polar" (polar coordinate system with r=f(θ)) |
                    "complex" (complex plane with multiplication) |
                    "number_plane" (standard 2D axes with function)
        title:      Heading text
        function:   For polar: "rose" | "lemniscate" | "spiral" | "cardioid" | "circle"
                    For number_plane: "sin" | "parabola"
        k_value:    Number of petals for rose function (default 4)
        duration:   Clip length in seconds
        color:      Curve colour: "BLUE" | "RED" | "GREEN" | "PURPLE" | "GOLD"
        show_label: Show formula label

    Returns JSON: {"video_url": str, "duration": float, "plane_type": str}
    """
    return json.dumps(await impl_generate_polar_graph(
        plane_type=plane_type, title=title, function=function,
        k_value=k_value, duration=duration, color=color, show_label=show_label,
    ))


@mcp.tool()
async def blender_generate_geometry_proof(
    proof_type: str = "pythagorean",
    title: str = "Geometry Proof",
    duration: float = 14.0,
    color_a: str = "BLUE",
    color_b: str = "RED",
    show_labels: bool = True,
) -> str:
    """
    Generate an animated geometry proof using Manim.

    Args:
        proof_type:  "pythagorean"    — visual proof of a²+b²=c²
                     "circle_area"   — π r² via inscribed polygon limit
                     "triangle_sum"  — angles of triangle sum to 180°
                     "boolean_ops"   — Union / Difference / Intersection of shapes
        title:       Heading text
        duration:    Clip length in seconds
        color_a:     Primary shape colour: "BLUE" | "RED" | "GREEN" | "GOLD"
        color_b:     Secondary shape colour
        show_labels: Show formula / angle labels

    Returns JSON: {"video_url": str, "duration": float, "proof_type": str}
    """
    return json.dumps(await impl_generate_geometry_proof(
        proof_type=proof_type, title=title, duration=duration,
        color_a=color_a, color_b=color_b, show_labels=show_labels,
    ))


@mcp.tool()
async def blender_generate_particle_confetti(
    style: str = "confetti",
    count: int = 400,
    duration: float = 6.0,
    primary_color: str = "",
    secondary_color: str = "",
    bg_color: str = "",
) -> str:
    """Generate an animated particle burst — confetti, snow, stars, rain, or bubbles.

    Args:
        style: "confetti" | "snow" | "stars" | "rain" | "bubbles"
        count: number of particles (default: 400)
        duration: clip length in seconds
        primary_color: JSON RGBA float array e.g. "[1,0.3,0.1,1]"
        secondary_color: JSON RGBA float array for second color variant
        bg_color: JSON RGBA float array for background

    Returns JSON: {"video_url": str, "duration": float, "style": str}
    """
    import json as _j
    kwargs = {"style": style, "count": count, "duration": duration}
    for k, v in [("primary_color", primary_color), ("secondary_color", secondary_color), ("bg_color", bg_color)]:
        if v:
            try: kwargs[k] = _j.loads(v)
            except Exception: pass
    return _j.dumps(await impl_generate_particle_confetti(**kwargs))


@mcp.tool()
async def blender_generate_rigid_body_drop(
    text: str = "DROP",
    object_type: str = "text",
    count: int = 12,
    duration: float = 5.0,
    color: str = "",
    bg_color: str = "",
    style: str = "dark",
) -> str:
    """Generate a physics rigid-body drop animation — 3D letters or objects fall and collide.

    Args:
        text: text to extrude as falling 3D letters (used when object_type="text")
        object_type: "text" | "spheres" | "cubes" | "mixed"
        count: number of objects if not text
        duration: clip length in seconds
        color: JSON RGBA float array
        bg_color: JSON RGBA float array
        style: "dark" | "bright" | "neon"

    Returns JSON: {"video_url": str, "duration": float, "text": str}
    """
    import json as _j
    kwargs = {"text": text, "object_type": object_type, "count": count, "duration": duration, "style": style}
    for k, v in [("color", color), ("bg_color", bg_color)]:
        if v:
            try: kwargs[k] = _j.loads(v)
            except Exception: pass
    return _j.dumps(await impl_generate_rigid_body_drop(**kwargs))


@mcp.tool()
async def blender_generate_camera_path(
    path_type: str = "orbit",
    subject: str = "abstract",
    title: str = "",
    duration: float = 8.0,
    color: str = "",
    bg_color: str = "",
    style: str = "cinematic",
) -> str:
    """Generate a smooth camera fly-through / orbit animation — orbit, helix, arc, dolly zoom, or flythrough.

    Args:
        path_type: "orbit" | "helix" | "arc" | "dolly_zoom" | "flythrough"
        subject: "spheres" | "cubes" | "text" | "abstract" | "landscape"
        title: optional 3D text placed in scene
        duration: clip length in seconds
        color: JSON RGBA float array for objects
        bg_color: JSON RGBA float array for background
        style: "cinematic" | "minimal" | "neon"

    Returns JSON: {"video_url": str, "duration": float, "path_type": str}
    """
    import json as _j
    kwargs = {"path_type": path_type, "subject": subject, "title": title, "duration": duration, "style": style}
    for k, v in [("color", color), ("bg_color", bg_color)]:
        if v:
            try: kwargs[k] = _j.loads(v)
            except Exception: pass
    return _j.dumps(await impl_generate_camera_path(**kwargs))


@mcp.tool()
async def blender_generate_toon_scene(
    subject: str = "abstract",
    title: str = "",
    duration: float = 6.0,
    outline_color: str = "",
    primary_color: str = "",
    bg_color: str = "",
    outline_width: float = 1.5,
    flat_shading: bool = True,
    animated: bool = True,
) -> str:
    """Generate an NPR cartoon / toon-shaded Blender scene with bold outlines and flat colours.

    Args:
        subject: "characters" | "robots" | "landscape" | "abstract" | "logo"
        title: optional text label
        duration: clip length in seconds
        outline_color: JSON RGBA float array for outlines
        primary_color: JSON RGBA float array for main objects
        bg_color: JSON RGBA float array
        outline_width: thickness of outlines (0.5–5.0, default 1.5)
        flat_shading: true for pure cartoon flat look
        animated: true for gentle floating animation

    Returns JSON: {"video_url": str, "duration": float, "subject": str}
    """
    import json as _j
    kwargs = {"subject": subject, "title": title, "duration": duration,
              "outline_width": outline_width, "flat_shading": flat_shading, "animated": animated}
    for k, v in [("outline_color", outline_color), ("primary_color", primary_color), ("bg_color", bg_color)]:
        if v:
            try: kwargs[k] = _j.loads(v)
            except Exception: pass
    return _j.dumps(await impl_generate_toon_scene(**kwargs))


@mcp.tool()
async def blender_generate_grease_pencil_reveal(
    text: str = "HELLO",
    style: str = "whiteboard",
    duration: float = 6.0,
    color: str = "",
    bg_color: str = "",
    stroke_width: int = 50,
) -> str:
    """Generate a whiteboard / sketch draw-on text reveal using Grease Pencil BUILD modifier.

    Args:
        text: text to draw (max 12 characters)
        style: "whiteboard" | "neon" | "sketch" | "chalkboard"
        duration: clip length in seconds
        color: JSON RGBA float array for strokes
        bg_color: JSON RGBA float array
        stroke_width: line thickness (10–200, default 50)

    Returns JSON: {"video_url": str, "duration": float, "text": str, "style": str}
    """
    import json as _j
    kwargs = {"text": text, "style": style, "duration": duration, "stroke_width": stroke_width}
    for k, v in [("color", color), ("bg_color", bg_color)]:
        if v:
            try: kwargs[k] = _j.loads(v)
            except Exception: pass
    return _j.dumps(await impl_generate_grease_pencil_reveal(**kwargs))


@mcp.tool()
async def blender_generate_geometry_scatter(
    instance_type: str = "spheres",
    surface: str = "plane",
    count: int = 200,
    duration: float = 8.0,
    primary_color: str = "",
    secondary_color: str = "",
    bg_color: str = "",
    animated: bool = True,
    scale: float = 1.0,
) -> str:
    """Generate a procedural instance-scatter animation — objects distributed across a surface with animated wave motion.

    Args:
        instance_type: "cubes" | "spheres" | "stars" | "arrows" | "crystals"
        surface: "plane" | "sphere" | "torus" | "grid"
        count: number of instances (default: 200)
        duration: clip length in seconds
        primary_color: JSON RGBA float array
        secondary_color: JSON RGBA float array for second color
        bg_color: JSON RGBA float array
        animated: true for wave displacement animation
        scale: instance object scale (default: 1.0)

    Returns JSON: {"video_url": str, "duration": float, "instance_type": str, "surface": str}
    """
    import json as _j
    kwargs = {"instance_type": instance_type, "surface": surface, "count": count,
              "duration": duration, "animated": animated, "scale": scale}
    for k, v in [("primary_color", primary_color), ("secondary_color", secondary_color), ("bg_color", bg_color)]:
        if v:
            try: kwargs[k] = _j.loads(v)
            except Exception: pass
    return _j.dumps(await impl_generate_geometry_scatter(**kwargs))


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
    "blender_generate_text_animation": impl_generate_text_animation,
    "blender_generate_vector_field":  impl_generate_vector_field,
    "blender_generate_matrix_transform": impl_generate_matrix_transform,
    "blender_generate_polar_graph":   impl_generate_polar_graph,
    "blender_generate_geometry_proof": impl_generate_geometry_proof,
    "blender_generate_particle_confetti": impl_generate_particle_confetti,
    "blender_generate_rigid_body_drop": impl_generate_rigid_body_drop,
    "blender_generate_camera_path":    impl_generate_camera_path,
    "blender_generate_toon_scene":     impl_generate_toon_scene,
    "blender_generate_grease_pencil_reveal": impl_generate_grease_pencil_reveal,
    "blender_generate_geometry_scatter": impl_generate_geometry_scatter,
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
        result = await TOOL_HANDLERS[tool_name](**args)
        logger.info("server.call_tool_completed tool=%s", tool_name)
        return JSONResponse({"result": result})
    except Exception as exc:
        logger.exception("server.call_tool_failed tool=%s error=%s", tool_name, exc)
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

rest_routes = [
    Route("/health",                rest_health),
    Route("/api/call_tool",         rest_call_tool,   methods=["POST"]),
    Route("/api/director",          rest_director,    methods=["POST"]),
    Route("/api/analyze-video",     rest_analyze_video, methods=["POST"]),
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
