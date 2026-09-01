"""
bpy_codegen.py — LLM-driven Blender Python code generator with VIGA features.

VIGA additions:
- Multiple candidates with VLM tournament selection (NUM_CANDIDATES env var)
- Parallel candidate generation and rendering
- Tournament bracket via Vision Language Model comparison
- Verifier quality gate (VIGA_ENABLE_VERIFIER env var)
- Docker sandbox pre-validation (VIGA_ENABLE_DOCKER_SANDBOX env var)
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
import textwrap
from typing import Optional

import httpx

from tools.rag_client import store_success, query_similar
from tools.verifier_loop import run_verifier_loop as _run_verifier_loop


MAX_RETRIES = 5
NUM_CANDIDATES = int(os.getenv("VIGA_NUM_CANDIDATES", "1"))
USE_VLM_TOURNAMENT = NUM_CANDIDATES > 1
_VIGA_ENABLE_DOCKER = os.getenv("VIGA_ENABLE_DOCKER_SANDBOX", "").lower() in ("true", "1", "yes")
VIGA_REACT_LOOP = os.getenv("VIGA_REACT_LOOP", "true").lower() in ("true", "1", "yes")

_BANNED_UI_OPS_NOTICE = (
    "Do NOT use: bpy.ops.wm.*, bpy.ops.screen.*, bpy.ops.view3d.*, "
    "bpy.ops.ed.*, bpy.app.handlers, bpy.app.timers, modal operators, "
    "or any operator that requires a UI context"
)

_WEB_SEARCH_RE = re.compile(r"WEB_SEARCH:\s*(.+?)(?:\n|$)", re.IGNORECASE)


async def web_search(query: str, num_results: int = 5) -> str:
    from tools.browserbase_client import browserbase_search
    return await browserbase_search(query, num_results)


async def _execute_web_search(text: str) -> str:
    results = []
    for match in _WEB_SEARCH_RE.finditer(text):
        query = match.group(1).strip()
        result = await web_search(query)
        results.append(f"Search query: {query}\nResults:\n{result}")
    return "\n\n".join(results)


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
        bpy.ops.render.render(animation=True).
""")


def _build_bpy_system_prompt(prompt: str, duration: float, style: str, reference_image_url: str = "") -> str:
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
You are an expert Blender Python (bpy) programmer. BLENDER VERSION: 4.0.2. Use only API features available in Blender 4.0. Do NOT use OPTIX denoiser, scene.cycles.denoiser, or other features added after 4.0. Prefer EEVEE over CYCLES for compatibility. You write scripts that run
headless (blender --background) and produce 3D rendered video files.

{_BPY_SYSTEM_INSTRUCTIONS}

═══ STYLE GUIDE ═══
Style: {style}
{style_guide}

═══ OUTPUT REQUIREMENTS ═══
• A JSON file path is passed as sys.argv[-1] (the last element after sys.argv[0]). Load it with: import json; args = json.load(open(sys.argv[-1])); output_path = args["output_path"]. Use this output_path for the render filepath.
• Configure scene.render.filepath to the output path.
• Set resolution: bpy.context.scene.render.resolution_x = 1920,
  bpy.context.scene.render.resolution_y = 1080.
• Set fps: bpy.context.scene.render.fps = 60.
• Set output format to FFMPEG: bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
• Set FFMPEG codec: bpy.context.scene.render.ffmpeg.format = 'MPEG4'
• Set H264 codec: bpy.context.scene.render.ffmpeg.codec = 'H264'
• Render engine: use 'CYCLES' for realism, 'BLENDER_EEVEE' for speed.
• Set frame_end based on duration: frame_end = int(duration * fps).
• At the very end, call:
    bpy.ops.render.render(animation=True)
• Print a RESULT line at the very end:
    print(f"RESULT:{{json.dumps({{{{'duration': {duration},
          'resolution': '1920x1080',
          'frames': int({duration} * 60),
          'output_path': output_path}})}}")
    where `output_path` is a Python variable with the target file path.

═══ USEFUL BPY PATTERNS ═══
• Create a mesh object — PREFER the primitive ops (they create data + object
  + link in one call and return the created object):
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    cube = bpy.context.active_object
    # also: primitive_uv_sphere_add(radius=1), primitive_cylinder_add(...),
    #       primitive_plane_add(size=10), primitive_torus_add(...)
    # text objects: bpy.ops.object.text_add(); txt = bpy.context.active_object
    #               txt.data.body = "Hello"

• Create an EMPTY (locator/parent-only object) — bpy.data.objects.new REQUIRES
  object_data as the 2nd positional arg; empties take None. NEVER pass
  type='EMPTY' (that raises TypeError: required parameter "object_data"):
    empty = bpy.data.objects.new("Container", None)
    bpy.context.collection.objects.link(empty)

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

═══ EXTERNAL 3D MODELS (Sketchfab assets stored in R2) ═══
Realistic products/devices/furniture should come from downloaded models, NOT
hand-modeled primitives. If the task benefits from real-world geometry, use
this helper VERBATIM at the top of your script (after `import bpy, json`):

    import urllib.request, zipfile, glob, os, hashlib

    def load_model_from_url(url):
        # Download a GLB/GLTF/ZIP model archive and import it.
        # Returns the root object. Prints MODEL_IMPORTED with its dimensions.
        dest = "/tmp/asset_" + hashlib.md5(url.encode()).hexdigest()[:10]
        os.makedirs(dest, exist_ok=True)
        if url.lower().endswith((".glb", ".gltf")):
            filepath = os.path.join(dest, "model" + os.path.splitext(url)[1])
            urllib.request.urlretrieve(url, filepath)
        else:
            archive = os.path.join(dest, "model.zip")
            urllib.request.urlretrieve(url, archive)
            zipfile.ZipFile(archive).extractall(dest)
            matches = (glob.glob(os.path.join(dest, "**", "*.gltf"), recursive=True)
                       or glob.glob(os.path.join(dest, "**", "*.glb"), recursive=True))
            if not matches:
                raise RuntimeError("No .gltf/.glb found in downloaded archive")
            filepath = sorted(matches)[0]
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=filepath)
        new_objs = [o for o in bpy.data.objects if o not in before]
        root = new_objs[0]
        bpy.context.view_layer.objects.active = root
        print(f"MODEL_IMPORTED name={{root.name}} dims={{[round(d,3) for d in root.dimensions]}}")
        return root

Full workflow (assets pipeline):
 1. Call sketchfab_search(query="coffee machine") -> pick a CC-licensed uid
 2. Call sketchfab_download(uid="<uid>") -> returns an R2 URL
 3. Inside your bpy script: root = load_model_from_url("<R2 URL>")
 4. The returned root behaves like ANY Blender object — animate it with code:
        root.location = (0, -2, 0); root.keyframe_insert(data_path='location', frame=1)
        root.rotation_euler.z += 3.14159; root.keyframe_insert(data_path='rotation_euler', frame=120)
    Parent other objects to it, add constraints, modifiers, physics — all standard bpy.
 5. ⚠️ Models ship at ARBITRARY scale/orientation. Read the printed dims and
    normalize, e.g. fit into 2m box:
        s = 2.0 / max(root.dimensions); root.scale = (s, s, s)
 6. ⚠️ ONLY use this helper when you actually downloaded a model in step 2.
    If no external model is needed for the scene, do NOT define or reference
    load_model_from_url or root anywhere — build geometry directly instead.
    Never reference a variable you have not assigned in YOUR script.

═══ ERROR FIXING ═══
If you are not sure about the correct bpy API to use, you can search the web
by including the following marker in your response:
    WEB_SEARCH: <natural language query about bpy API>

═══ YOUR TASK ═══
{prompt}

Write the complete Blender Python script. Begin with `import bpy, json`.
Do NOT write the code as plain response text — deliver it by calling the
`run_render` tool with the full script in the `code` argument (the AGENTIC
LOOP CONTRACT below is authoritative and overrides any wording elsewhere).
"""


def _extract_code(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        cleaned.append(line)
    result = "\n".join(cleaned).strip()
    if result:
        return _guard_orphan_root(result)
    fence = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if fence:
        return _guard_orphan_root(fence.group(1).strip())
    return _guard_orphan_root(text.strip())


def _guard_orphan_root(code: str) -> str:
    """Neutralize references to `root` when the Sketchfab helper was never
    called. The system prompt teaches a `root = load_model_from_url(...)`
    pattern; models sometimes copy the usage lines without the assignment,
    producing `NameError: name 'root' is not defined` at render time.
    Deterministic guard: comment out orphan root lines so the rest of the
    script renders instead of failing."""
    import re as _re
    calls_helper = bool(_re.search(r"\bload_model_from_url\s*\(", code))
    assigns_root = bool(_re.search(r"^\s*root\s*=", code, flags=_re.M)) or \
        bool(_re.search(r"\broot\s*=\s*load_model_from_url", code))
    if calls_helper and assigns_root:
        return code  # legitimate usage
    if "root" not in code:
        return code
    out = []
    for line in code.splitlines():
        if _re.search(r"\broot\b", line) and not line.lstrip().startswith("#"):
            out.append("# [orphan-root removed] " + line)
        else:
            out.append(line)
    return "\n".join(out)


def _strip_web_search_markers(text: str) -> str:
    return _WEB_SEARCH_RE.sub("", text).strip()


async def _call_llm(prompt: str) -> str:
    from tools.llm_client import generate_text
    text, _ = await generate_text(prompt, temperature=0.3, max_tokens=8192)
    return text


async def _generate_code(prompt: str, duration: float, style: str, reference_image_url: str = "") -> str:
    system_prompt = _build_bpy_system_prompt(prompt, duration, style, reference_image_url)
    raw = await _call_llm(system_prompt)
    return _extract_code(raw)


async def _fix_code(code: str, error: str, original_prompt: str, duration: float, style: str, search_results: str = "", rag_context: str = "") -> str:
    search_section = ""
    if search_results:
        search_section = f"\n═══ WEB SEARCH RESULTS ═══\n{search_results}\n"
    rag_section = rag_context if rag_context else ""

    fix_prompt = textwrap.dedent(f"""\
        The following Blender Python (bpy) code failed to execute.

        ═══ ORIGINAL TASK ═══
        {original_prompt}
        {search_section}
        {rag_section}
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
        • Do NOT use UI-dependent operators (bpy.ops.wm.*, bpy.ops.screen.*, etc.).
        • If bpy.ops fails, try using direct data access (bpy.data.objects, etc.).
        • Simplify rather than guess — use a simpler approach that is guaranteed to work.
        • Target duration: {duration:.1f} seconds.
        • Output ONLY the corrected Python code. No explanation.
    """)
    raw = await _call_llm(fix_prompt)
    return _extract_code(raw)


async def _docker_validate(code: str, script_type: str = "blender") -> str | None:
    """Run the script through Docker sandbox pre-validation. Returns error text
    if validation failed, None if passed."""
    from tools.docker_sandbox import validate_in_docker
    result = await validate_in_docker(code, script_type, timeout=15)
    if not result["passed"]:
        logs = result.get("logs", "")
        error = result.get("error", "Docker sandbox validation failed")
        # Keep logs under 3000 chars to avoid context overflow
        if len(logs) > 3000:
            logs = logs[-3000:]
        return f"[Docker sandbox pre-check] {error}\nLogs:\n{logs}"
    return None


async def _render_single(prompt: str, duration: float, style: str, output_path: str, reference_image_url: str, attempt: int) -> dict:
    """Generate code and render a single candidate. Returns dict with result or error."""
    from tools.blender_runner import run_blender_script

    code = await _generate_code(prompt, duration, style, reference_image_url)
    search_results = await _execute_web_search(code)
    if search_results:
        code = _strip_web_search_markers(code)
    code = _extract_code(code)

    if _VIGA_ENABLE_DOCKER:
        dock_err = await _docker_validate(code)
        if dock_err:
            return {"error": dock_err, "code": code}

    with tempfile.NamedTemporaryFile(suffix=".py", prefix=f"bpy_candidate_{attempt}_", delete=False, mode="w") as f:
        f.write(code)
        script_path = f.name

    try:
        args = {"prompt": prompt[:200], "duration": duration, "style": style, "output_path": output_path}
        if reference_image_url:
            args["reference_image_url"] = reference_image_url
        result = await run_blender_script(script_path=script_path, args=args, timeout=600)
        output = result.get("output_path", output_path)
        if os.path.exists(output):
            return {"path": output, "code": code, "score": None}
        return {"error": "no output file", "code": code}
    except RuntimeError as e:
        return {"error": str(e)[-2000:], "code": code}
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


async def _run_vlm_tournament(results: list[dict], prompt: str) -> int:
    from tools.vision_tools import compare_render_to_reference

    valid = [(i, r) for i, r in enumerate(results) if "path" in r and os.path.exists(r["path"])]
    if len(valid) <= 1:
        return valid[0][0] if valid else 0

    candidates = [(idx, r["path"]) for idx, r in valid]
    while len(candidates) > 1:
        next_round = []
        for i in range(0, len(candidates), 2):
            if i + 1 < len(candidates):
                idx_a, path_a = candidates[i]
                idx_b, path_b = candidates[i + 1]
                try:
                    comparison = compare_render_to_reference(
                        render_path=path_b, reference_path_or_url=path_a, prompt_context=prompt,
                    )
                    score_a = comparison.get("match_score", 0.5)
                    comparison_rev = compare_render_to_reference(
                        render_path=path_a, reference_path_or_url=path_b, prompt_context=prompt,
                    )
                    score_b = comparison_rev.get("match_score", 0.5)
                    winner = idx_a if score_a >= score_b else idx_b
                except Exception:
                    winner = idx_a
                next_round.append((winner, path_a if winner == idx_a else path_b))
            else:
                next_round.append(candidates[i])
        candidates = next_round
    return candidates[0][0] if candidates else 0


async def _react_render_single(prompt: str, duration: float, style: str, output_path: str, reference_image_url: str, thread_id: str = "") -> tuple[str, str]:
    """Agentic ReAct path: the LLM drives codegen + web search + docker validation
    + rendering turn-by-turn instead of a hard-coded retry loop."""
    from tools.react_codegen import run_agentic_codegen

    from tools.blender_runner import run_blender_script

    async def _render_bpy_code(code: str) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".py", prefix=f"bpy_react_", delete=False, mode="w") as f:
            f.write(code)
            script_path = f.name
        try:
            args = {"prompt": prompt[:200], "duration": duration, "style": style, "output_path": output_path}
            if reference_image_url:
                args["reference_image_url"] = reference_image_url
            result = await run_blender_script(script_path=script_path, args=args, timeout=600)
            out = result.get("output_path", output_path)
            if os.path.exists(out):
                return {"output_path": out}
            return {"error": "render produced no output file"}
        except RuntimeError as e:
            return {"error": str(e)[-3000:]}
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    return await run_agentic_codegen(
        engine="blender",
        system_prompt=_build_bpy_system_prompt(prompt, duration, style, reference_image_url),
        brief=prompt,
        render_func=_render_bpy_code,
        store_success=lambda code, b: store_success("bpy", code, b),
        rag_collection="bpy",
        docker_script_type="blender",
        thread_id=thread_id,
    )


async def _retry_render_single(prompt: str, duration: float, style: str, output_path: str, reference_image_url: str, thread_id: str = "") -> tuple[str, str]:
    """Standard single-candidate approach with retry loop. Returns (output_path, code)."""
    if VIGA_REACT_LOOP:
        return await _react_render_single(prompt, duration, style, output_path, reference_image_url, thread_id=thread_id)

    from tools.blender_runner import run_blender_script

    code = ""
    last_error = ""
    search_results = ""

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt == 1:
            code = await _generate_code(prompt, duration, style, reference_image_url)
        else:
            rag_context = await query_similar("bpy", last_error, prompt)
            code = await _fix_code(code, last_error, prompt, duration, style, search_results, rag_context)

        search_results = await _execute_web_search(code)
        if search_results:
            code = _strip_web_search_markers(code)

        # Docker sandbox pre-validation (fast, catches import/API errors)
        if _VIGA_ENABLE_DOCKER:
            dock_err = await _docker_validate(code)
            if dock_err:
                last_error = dock_err
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"bpy code failed Docker sandbox validation after {MAX_RETRIES} attempts.\n"
                        f"Last error:\n{last_error}"
                    )
                continue

        with tempfile.NamedTemporaryFile(suffix=".py", prefix=f"bpy_gen_{attempt}_", delete=False, mode="w") as f:
            f.write(code)
            script_path = f.name

        try:
            args = {"prompt": prompt[:200], "duration": duration, "style": style, "output_path": output_path}
            if reference_image_url:
                args["reference_image_url"] = reference_image_url
            result = await run_blender_script(script_path=script_path, args=args, timeout=600)
            await store_success("bpy", code, prompt)
            try:
                os.unlink(script_path)
            except OSError:
                pass
            return (result.get("output_path", output_path), code)
        except RuntimeError as e:
            last_error = str(e)[-3000:]
            try:
                os.unlink(script_path)
            except OSError:
                pass
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"LLM-generated bpy code failed after {MAX_RETRIES} attempts. Last error:\n{last_error}"
                ) from e

    raise RuntimeError("_retry_render_single: unexpected exit")


async def generate_and_run_bpy(
    prompt: str,
    duration: float = 10.0,
    style: str = "cinematic",
    output_path: Optional[str] = None,
    reference_image_url: str = "",
    thread_id: str = "",
    **extra_args,
) -> str:
    if output_path is None:
        import uuid
        output_path = f"/tmp/bpy_gen_{uuid.uuid4().hex}.mp4"

    async def _render_bpy(prompt: str, **kwargs) -> tuple[str, str]:
        if not USE_VLM_TOURNAMENT:
            return await _retry_render_single(prompt, duration, style, output_path, reference_image_url, thread_id=thread_id)

        candidate_dir = tempfile.mkdtemp(prefix="bpy_candidates_")
        try:
            candidate_paths = [
                os.path.join(candidate_dir, f"candidate_{i}.mp4")
                for i in range(NUM_CANDIDATES)
            ]
            tasks = [
                _render_single(prompt, duration, style, cp, reference_image_url, i + 1)
                for i, cp in enumerate(candidate_paths)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            processed = []
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    processed.append({"error": str(r), "code": ""})
                else:
                    processed.append(r)
            winner_idx = await _run_vlm_tournament(processed, prompt)
            winner = processed[winner_idx]
            if "path" in winner and os.path.exists(winner["path"]):
                shutil.copy2(winner["path"], output_path)
                return (output_path, winner.get("code", ""))
            return await _retry_render_single(prompt, duration, style, output_path, reference_image_url)
        finally:
            shutil.rmtree(candidate_dir, ignore_errors=True)

    final_path = await _run_verifier_loop(
        render_fn=_render_bpy,
        prompt=prompt,
        code="",
        render_kwargs={
            "duration": duration,
            "style": style,
            "reference_image_url": reference_image_url,
        },
    )
    return final_path
