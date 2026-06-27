"""
LLM-driven Blender Python (bpy) code generator with sandbox execution and retry loop.

Architecture
------------
1. Build a system prompt containing:
   - Headless Blender constraints (no UI, no modal operators)
   - bpy API patterns for common tasks (scene setup, cameras, materials, keyframes)
   - Three complete few-shot examples demonstrating correct headless patterns
2. Call the active LLM (Gemini/Claude via llm_client.generate_text)
3. Static pre-validation: syntax check + banned-API scan
4. Sandbox execution via blender_runner.run_blender_script()
5. On failure: inject (code + error) back into LLM with web search capability,
   up to MAX_RETRIES attempts
6. On final failure: raise RuntimeError

Usage
-----
    from tools.bpy_codegen import generate_and_run_bpy

    output_path = await generate_and_run_bpy(
        prompt="Create a 3D scene with a glowing cube rotating on a dark background",
        duration=10.0,
        style="cinematic",
        output_path="/tmp/my_scene.mp4",
    )

Web Search Integration
----------------------
When the generated bpy code fails with an unknown error, the LLM can request
a web search by including the marker:
    WEB_SEARCH: <natural language query>
in its response. The retry loop detects this, performs the search via
DuckDuckGo, and injects the results into the next fix attempt. This allows
the LLM to look up correct bpy API calls for unfamiliar operations.
"""
from __future__ import annotations

import ast
import json
import os
import re
import tempfile
import textwrap
from typing import Optional

import httpx


MAX_RETRIES = 5

# Deprecated / wrong-version identifiers for bpy code
_BANNED_PATTERNS = [
    "bpy.ops.wm",           # windowing operators (require UI context)
    "bpy.ops.screen",       # screen operators (require UI context)
    "bpy.ops.view3d",       # viewport operators (require UI context)
    "bpy.ops.ed",           # editor operators (require UI context)
    "bpy.ops.ui",           # UI operators (require UI context)
    "bpy.app.handlers",     # app handlers (runs indefinitely)
    "modal",                # modal operators (require UI)
    "invoke_default",       # UI-only
    "bpy.ops.object.mode_set",  # mode_set with no context override
    "CUSTOM_DRIVER",        # drivers won't evaluate in headless
    "bpy.app.timers",       # timers won't fire in headless
]

_WEB_SEARCH_RE = re.compile(r"WEB_SEARCH:\s*(.+?)(?:\n|$)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Web search for LLM debugging
# ---------------------------------------------------------------------------

async def web_search(query: str, num_results: int = 5) -> str:
    """Search the web via BrowserBase and return structured results."""
    from tools.browserbase_client import browserbase_search
    return await browserbase_search(query, num_results)


async def _execute_web_search(text: str) -> str:
    """Check if the LLM response contains WEB_SEARCH markers and execute them."""
    results = []
    for match in _WEB_SEARCH_RE.finditer(text):
        query = match.group(1).strip()
        result = await web_search(query)
        results.append(f"Search query: {query}\nResults:\n{result}")
    return "\n\n".join(results)


# ---------------------------------------------------------------------------
# System prompt with few-shot examples
# ---------------------------------------------------------------------------

_BPY_SYSTEM_INSTRUCTIONS = textwrap.dedent("""\
    Headless Blender constraints (run via `blender --background --python script.py`):
    1. ALWAYS start with `import bpy, math, random` (add other stdlib as needed)
    2. DO NOT use any bpy.ops that require a UI context (bpy.ops.wm.*,
       bpy.ops.screen.*, bpy.ops.view3d.*, modal operators, timers).
    3. Use bpy.ops.object.select_all(action='SELECT') for object operations.
    4. Use bpy.ops.object.delete() sparingly — prefer hiding or clearing.
    5. For materials: use bpy.data.materials.new() + node tree manipulation.
    6. For animation: set keyframes via obj.keyframe_insert().
    7. For camera: bpy.ops.object.camera_add() or bpy.data.cameras.new().
    8. For lighting: bpy.ops.object.light_add().
    9. Output final path in environment variable or direct file write.
    10. Scene class: NOT USED. Write script as top-level Python statements.
    11. All transforms use bpy.ops.transform.* OR direct obj.location = ...,
        obj.rotation_euler = ..., obj.scale = ...
    12. For rendering: configure scene.render settings, then call
        bpy.ops.render.render(animation=True, write_still=True).
""")


def _build_bpy_system_prompt(
    prompt: str,
    duration: float,
    style: str,
    reference_image_url: str = "",
) -> str:
    style_hints = {
        "cinematic": "Use dramatic lighting (area lights with warm/cool contrast), "
                     "depth of field, smooth camera motion, rich materials.",
        "minimal": "Clean geometry, neutral colors, soft lighting, "
                   "simple materials with low roughness.",
        "energetic": "Bright colors, fast camera motion, particle effects, "
                     "dynamic lighting, saturated materials.",
        "calm": "Soft pastel colors, slow camera motion, gentle lighting, "
                "simple geometry with smooth transitions.",
        "dark": "Dark background, rim lighting, neon accents, "
                "high contrast between light and shadow.",
        "neon": "Black background, neon emission materials, "
                "glowing edges, cyberpunk aesthetic.",
        "bright": "White or light background, colorful objects, "
                  "soft shadows, clean aesthetic.",
        "whiteboard": "White background, black strokes, Grease Pencil style, "
                      "no materials, wireframe-like appearance.",
        "youtube": "Bright appealing colors, clear focal point, "
                   "readable text, 3D depth without clutter.",
        "bold": "Strong saturated colors, thick geometry, dramatic lighting, "
                "large text with deep extrusion.",
        "modern": "Clean lines, flat materials, subtle gradients, "
                  "smooth animations, sans-serif text.",
        "sketch": "Grease Pencil strokes, hand-drawn look, rough edges, "
                  "paper-like background.",
    }
    style_guide = style_hints.get(style, style_hints["cinematic"])

    return f"""\
You are an expert Blender Python (bpy) programmer. You write scripts that run
headless (blender --background) and produce 3D rendered video files.

{_BPY_SYSTEM_INSTRUCTIONS}

═══ STYLE GUIDE ═══
Style: {style}
{style_guide}

═══ OUTPUT REQUIREMENTS ═══
• The output path is given as an argument. Write the final render to this path.
• Configure scene.render.filepath to the output path.
• Set resolution: bpy.context.scene.render.resolution_x = 1920,
  bpy.context.scene.render.resolution_y = 1080.
• Set fps: bpy.context.scene.render.fps = 60.
• Render engine: use 'CYCLES' for realism, 'BLENDER_EEVEE' for speed.
• Set frame_end based on duration: frame_end = int(duration * fps).
• At the very end, call:
    bpy.ops.render.render(animation=True, write_still=True)
• Print a RESULT line at the very end:
    print(f"RESULT:{{json.dumps({{{{'duration': {duration},
          'resolution': '1920x1080',
          'frames': int({duration} * 60),
          'output_path': output_path}})}}")
    where `output_path` is a Python variable with the target file path.

═══ USEFUL BPY PATTERNS ═══
• Add a camera:
    cam_data = bpy.data.cameras.new(name='Camera')
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = (0, -10, 5)
    cam_obj.rotation_euler = (1.1, 0, 0)

• Animate camera:
    cam_obj.location = (0, -10, 5)
    cam_obj.keyframe_insert(data_path='location', frame=1)
    cam_obj.location = (5, -8, 3)
    cam_obj.keyframe_insert(data_path='location', frame=120)

• Add material with emission:
    mat = bpy.data.materials.new(name='GlowMat')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Emission Color'].default_value = (0.1, 0.5, 1.0, 1.0)
    bsdf.inputs['Emission Strength'].default_value = 2.0
    obj.data.materials.append(mat)

• Add keyframes to object properties:
    obj.location.z = 0
    obj.keyframe_insert(data_path='location', index=2, frame=1)
    obj.location.z = 3
    obj.keyframe_insert(data_path='location', index=2, frame=60)

• Add Grease Pencil object:
    bpy.ops.object.gpencil_add(align='WORLD', location=(0, 0, 0))

═══ ERROR FIXING ═══
If you are not sure about the correct bpy API to use, you can search the web
by including the following marker in your response:
    WEB_SEARCH: <natural language query about bpy API>

Example:
    WEB_SEARCH: bpy how to add keyframe to material node value

The search results will be provided to you before the next attempt.

═══ YOUR TASK ═══
{prompt}

Write the complete Blender Python script. Begin with `import bpy, json`.
Include only the Python code — no markdown fences, no explanation.
"""


# ---------------------------------------------------------------------------
# Static pre-validator
# ---------------------------------------------------------------------------

def _extract_code(text: str) -> str:
    """Strip markdown fences if the LLM wrapped the code."""
    fence = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def _strip_web_search_markers(text: str) -> str:
    """Remove WEB_SEARCH markers from the response."""
    return _WEB_SEARCH_RE.sub("", text).strip()


def _static_validate(code: str) -> Optional[str]:
    """Check the generated code before running Blender.
    Returns an error description string, or None if the code looks OK."""
    if not code.strip():
        return "Generated code is empty."

    # 1. Python syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"Python SyntaxError on line {e.lineno}: {e.msg}"

    # 2. Banned / wrong-version API patterns
    for pattern in _BANNED_PATTERNS:
        if pattern in code:
            return f"Banned API pattern detected: '{pattern}'. This requires UI context and will fail in headless mode."

    # 3. Required imports
    if "import bpy" not in code:
        return "Missing `import bpy` at the top of the script."

    # 4. Render call present
    if "render" not in code or "animation" not in code:
        return "Missing render call. Add `bpy.ops.render.render(animation=True, write_still=True)` at the end."

    return None


# ---------------------------------------------------------------------------
# LLM code generation
# ---------------------------------------------------------------------------

async def _call_llm(prompt: str) -> str:
    """Generate text via the active LLM provider."""
    from tools.llm_client import generate_text
    text, _ = await generate_text(
        prompt,
        temperature=0.3,
        max_tokens=8192,
    )
    return text


async def _generate_code(
    prompt: str,
    duration: float,
    style: str,
    reference_image_url: str = "",
) -> str:
    """Ask the LLM to produce a Blender Python script."""
    system_prompt = _build_bpy_system_prompt(prompt, duration, style, reference_image_url)
    raw = await _call_llm(system_prompt)
    return _extract_code(raw)


async def _fix_code(
    code: str,
    error: str,
    original_prompt: str,
    duration: float,
    style: str,
    search_results: str = "",
) -> str:
    """Ask the LLM to fix failing code given the execution error."""
    search_section = ""
    if search_results:
        search_section = f"\n═══ WEB SEARCH RESULTS ═══\n{search_results}\n"

    fix_prompt = textwrap.dedent(f"""\
        The following Blender Python (bpy) code failed to execute.

        ═══ ORIGINAL TASK ═══
        {original_prompt}
        {search_section}
        ═══ FAILING CODE ═══
        ```python
        {code}
        ```

        ═══ ERROR ═══
        {error}

        ═══ INSTRUCTIONS ═══
        • Fix the error shown above.
        • Keep the overall scene intent the same.
        • If unsure about the correct bpy API, include:
            WEB_SEARCH: <query about the correct API>
          in your response. Search results will be provided on the next attempt.
        • Do NOT use UI-dependent operators (bpy.ops.wm.*, bpy.ops.screen.*, etc.).
        • If bpy.ops fails, try using direct data access (bpy.data.objects, etc.).
        • For errors with bpy.ops.object.*, add context_override or use direct attribute setting.
        • Simplify rather than guess — use a simpler approach that is guaranteed to work.
        • Target duration: {duration:.1f} seconds.
        • Output ONLY the corrected Python code. No explanation.
    """)
    raw = await _call_llm(fix_prompt)
    return _extract_code(raw)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def generate_and_run_bpy(
    prompt: str,
    duration: float = 10.0,
    style: str = "cinematic",
    output_path: Optional[str] = None,
    reference_image_url: str = "",
    **extra_args,
) -> str:
    """
    Generate a Blender scene from a natural language description and render it.

    The LLM generates raw bpy Python code, validates it statically, and runs
    it in headless Blender. On failure, it retries with LLM-driven fixes —
    optionally searching the web for the correct API calls.

    Args:
        prompt:             Natural language description of the 3D scene.
        duration:           Target clip duration in seconds.
        style:              Visual style ("cinematic"|"minimal"|"energetic"|"calm").
        output_path:        Destination file path (auto-generated if None).
        reference_image_url: Optional reference image URL for style guidance.

    Returns:
        Absolute path to the rendered MP4 file.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    from tools.blender_runner import run_blender_script

    if output_path is None:
        import uuid
        output_path = f"/tmp/bpy_gen_{uuid.uuid4().hex}.mp4"

    code: str = ""
    last_error: str = ""
    search_results: str = ""

    for attempt in range(1, MAX_RETRIES + 1):
        # ── generate / fix ────────────────────────────────────────────────
        if attempt == 1:
            code = await _generate_code(prompt, duration, style, reference_image_url)
        else:
            code = await _fix_code(code, last_error, prompt, duration, style, search_results)

        # ── check for web search markers ──────────────────────────────────
        search_results = await _execute_web_search(code)
        if search_results:
            # The LLM requested web search — strip markers, validate, retry
            code = _strip_web_search_markers(code)

        # ── static validate ───────────────────────────────────────────────
        static_err = _static_validate(code)
        if static_err:
            last_error = f"Static validation failed: {static_err}"
            continue

        # ── write to temp file ────────────────────────────────────────────
        with tempfile.NamedTemporaryFile(
            suffix=".py",
            prefix=f"bpy_gen_{attempt}_",
            delete=False,
            mode="w",
        ) as f:
            f.write(code)
            script_path = f.name

        # ── run blender ───────────────────────────────────────────────────
        try:
            args = {
                "prompt": prompt[:200],
                "duration": duration,
                "style": style,
                "output_path": output_path,
            }
            if reference_image_url:
                args["reference_image_url"] = reference_image_url

            result = await run_blender_script(
                script_path=script_path,
                args=args,
                timeout=600,
            )
            # success
            try:
                os.unlink(script_path)
            except OSError:
                pass
            return result.get("output_path", output_path)

        except RuntimeError as e:
            last_error = str(e)[-3000:]
            try:
                os.unlink(script_path)
            except OSError:
                pass

            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"LLM-generated bpy code failed after {MAX_RETRIES} attempts. "
                    f"Last error:\n{last_error}"
                ) from e
