"""
Shared render tool implementations.

Imported by both server.py (REST endpoint) and agents/director.py (LangGraph).
Each impl_* function drives a Blender or Manim script, uploads the result to R2,
and returns a plain dict with a video_url or image_url.
"""
import logging
import os
import uuid
from pathlib import Path

# Root of the BlenderMCPServer package
_ROOT = Path(__file__).parent.parent
logger = logging.getLogger(__name__)


async def impl_generate_scene(
    prompt: str,
    duration: float = 10.0,
    style: str = "cinematic",
    reference_image_url: str = "",
) -> dict:
    from tools.storage import upload_render

    # If a reference image is provided, use the Vision + QA pipeline (Phase 3)
    if reference_image_url:
        from agents.vision_agent import run_vision_agent
        from agents.qa_agent import run_qa_agent

        output_path = f"/tmp/blender_vision_{uuid.uuid4().hex}.mp4"
        logger.info(
            "render_tools.generate_scene_reference_start duration=%.2f style=%s has_reference=%s",
            duration,
            style,
            True,
        )
        vision_result = await run_vision_agent(
            prompt=prompt,
            reference_image_url=reference_image_url,
            duration=duration,
            style=style,
            output_path=output_path,
        )

        if vision_result.get("error"):
            raise RuntimeError(vision_result["error"])

        # QA refinement loop (max 3 iterations)
        scene_params = vision_result.get("scene_params", {})
        logger.info(
            "render_tools.scene_plan mode=%s camera=%s movement=%s material=%s layers=%s verify=%s",
            scene_params.get("blender_reference_mode", 2),
            scene_params.get("camera_angle", "front"),
            scene_params.get("movement_style", "slow_push"),
            scene_params.get("material_style", "mixed"),
            scene_params.get("scene_layers", []),
            scene_params.get("verification_focus", []),
        )
        if scene_params and os.path.exists(vision_result["output_path"]):
            # Download reference image for local QA comparison
            ref_local = vision_result.get("reference_image_path") or ""
            if not ref_local:
                # Try to get local path from vision agent (already downloaded)
                ref_local = output_path.replace(".mp4", "_ref.jpg")

            if os.path.exists(ref_local):
                qa_result = await run_qa_agent(
                    render_video_path=vision_result["output_path"],
                    reference_image_path=ref_local,
                    blender_script_path=str(_ROOT / "blender_scripts" / "reference_mode.py"),
                    blender_args={
                        "output_path": output_path,
                        "duration": duration,
                        "fps": 60,
                        "reference_image_path": ref_local,
                        "mode": scene_params.get("blender_reference_mode", 2),
                        "dominant_colors": scene_params.get("dominant_colors", []),
                        "lighting_type": scene_params.get("lighting_type", "studio"),
                        "camera_angle": scene_params.get("camera_angle", "front"),
                        "mood": scene_params.get("mood", "cinematic"),
                        "key_objects": scene_params.get("key_objects", []),
                        "composition_focus": scene_params.get("composition_focus", "centered"),
                        "movement_style": scene_params.get("movement_style", "slow_push"),
                        "material_style": scene_params.get("material_style", "mixed"),
                        "scene_layers": scene_params.get("scene_layers", []),
                        "verification_focus": scene_params.get("verification_focus", []),
                        "notes": scene_params.get("notes", ""),
                        "prompt": prompt,
                    },
                    prompt_context=(
                        f"{prompt}\n"
                        f"Verification focus: {', '.join(scene_params.get('verification_focus', [])) or 'preserve brand silhouette, hero subject, and palette'}.\n"
                        f"Scene layers: {', '.join(scene_params.get('scene_layers', [])) or 'foreground, subject, support, background'}.\n"
                        f"Planner notes: {scene_params.get('notes', '')}"
                    ).strip(),
                    max_iterations=3,
                )
                logger.info(
                    "render_tools.qa_result approved=%s best_score=%.3f iterations=%s error=%s",
                    qa_result.get("approved", False),
                    float(qa_result.get("best_score", 0.0)),
                    qa_result.get("iterations", 0),
                    qa_result.get("error", ""),
                )
                final_path = qa_result.get("best_video_path", vision_result["output_path"])
            else:
                logger.warning("render_tools.qa_skipped missing_local_reference output=%s", vision_result["output_path"])
                final_path = vision_result["output_path"]
        else:
            logger.warning("render_tools.qa_skipped missing_scene_params_or_output has_params=%s output_exists=%s",
                bool(scene_params),
                os.path.exists(vision_result.get("output_path", "")),
            )
            final_path = vision_result.get("output_path", output_path)

        video_url = upload_render(final_path, prefix="scenes")
        try:
            os.unlink(final_path)
        except OSError:
            pass

        return {
            "video_url": video_url,
            "duration": duration,
            "resolution": "1920x1080",
            "frames": int(duration * 60),
            "reference_mode": scene_params.get("blender_reference_mode", 2),
        }

    # No reference image — standard Blender scene
    from tools.blender_runner import run_blender_script_with_retry

    script_path = _ROOT / "blender_scripts" / "base_scene.py"
    output_path = f"/tmp/blender_{uuid.uuid4().hex}.mp4"
    logger.info(
        "render_tools.generate_scene_plain_start duration=%.2f style=%s",
        duration,
        style,
    )

    result = await run_blender_script_with_retry(
        script_content=script_path.read_text(),
        args={"prompt": prompt, "duration": duration, "style": style, "output_path": output_path},
        max_attempts=3,
        timeout=600,
    )

    video_url = upload_render(output_path, prefix="scenes")
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


async def impl_generate_thumbnail(
    prompt: str,
    title_text: str = "",
    style: str = "youtube",
) -> dict:
    from tools.blender_runner import run_blender_script_with_retry
    from tools.storage import upload_render

    script_path = _ROOT / "blender_scripts" / "thumbnail.py"
    output_path = f"/tmp/thumb_{uuid.uuid4().hex}.png"

    result = await run_blender_script_with_retry(
        script_content=script_path.read_text(),
        args={"prompt": prompt, "title_text": title_text, "style": style, "output_path": output_path},
        max_attempts=3,
        timeout=300,
    )

    image_url = upload_render(output_path, prefix="thumbnails")
    try:
        os.unlink(output_path)
    except OSError:
        pass

    return {
        "image_url": image_url,
        "width": result.get("width", 1280),
        "height": result.get("height", 720),
    }


async def impl_generate_title_card(
    title: str,
    subtitle: str = "",
    duration: float = 5.0,
    style: str = "cinematic",
) -> dict:
    from tools.blender_runner import run_blender_script_with_retry
    from tools.storage import upload_render

    script_path = _ROOT / "blender_scripts" / "title_card.py"
    output_path = f"/tmp/titlecard_{uuid.uuid4().hex}.mp4"

    result = await run_blender_script_with_retry(
        script_content=script_path.read_text(),
        args={"title": title, "subtitle": subtitle, "duration": duration, "style": style, "output_path": output_path},
        max_attempts=3,
        timeout=400,
    )

    video_url = upload_render(output_path, prefix="title_cards")
    try:
        os.unlink(output_path)
    except OSError:
        pass

    return {
        "video_url": video_url,
        "duration": result.get("duration", duration),
    }


async def impl_generate_data_viz(
    data_json: str,
    chart_type: str = "bar",
    title: str = "",
    duration: float = 10.0,
) -> dict:
    from tools.blender_runner import run_blender_script_with_retry
    from tools.storage import upload_render

    script_path = _ROOT / "blender_scripts" / "data_viz.py"
    output_path = f"/tmp/dataviz_{uuid.uuid4().hex}.mp4"

    result = await run_blender_script_with_retry(
        script_content=script_path.read_text(),
        args={
            "data_json": data_json,
            "chart_type": chart_type,
            "title": title,
            "duration": duration,
            "output_path": output_path,
        },
        max_attempts=3,
        timeout=600,
    )

    video_url = upload_render(output_path, prefix="data_viz")
    try:
        os.unlink(output_path)
    except OSError:
        pass

    return {
        "video_url": video_url,
        "duration": result.get("duration", duration),
        "chart_type": chart_type,
    }


async def impl_generate_lower_third(
    name_text: str,
    subtitle_text: str = "",
    style: str = "modern",
    duration: float = 5.0,
) -> dict:
    from tools.blender_runner import run_blender_script_with_retry
    from tools.storage import upload_render

    script_path = _ROOT / "blender_scripts" / "lower_third.py"
    output_path = f"/tmp/lowerthird_{uuid.uuid4().hex}.mp4"

    result = await run_blender_script_with_retry(
        script_content=script_path.read_text(),
        args={
            "name_text": name_text,
            "subtitle_text": subtitle_text,
            "style": style,
            "duration": duration,
            "output_path": output_path,
        },
        max_attempts=3,
        timeout=300,
    )

    video_url = upload_render(output_path, prefix="lower_thirds")
    try:
        os.unlink(output_path)
    except OSError:
        pass

    return {
        "video_url": video_url,
        "duration": result.get("duration", duration),
        "keying": result.get("keying", "green_screen"),
    }


async def impl_generate_ui_mockup(
    device: str = "iphone",
    animation: str = "reveal",
    duration: float = 6.0,
    screenshot_path: str = "",
    screenshot_url: str = "",
    screenshot_spec: dict | None = None,
    background_color: list | None = None,
    accent_color: list | None = None,
    fps: int = 60,
) -> dict:
    """
    Phase 4 — Device Mockup pipeline.

    Accepts a screenshot in one of three ways (in priority order):
      1. screenshot_path  — local file path
      2. screenshot_url   — remote URL (downloaded to /tmp)
      3. screenshot_spec  — design spec dict, rasterised via svg_export

    Renders via blender_scripts/device_mockup.py and uploads to R2.

    Returns:
        {"video_url": str, "device": str, "animation": str, "duration": float}
        or {"image_url": str, "device": str} for animation="static"
    """
    import tempfile
    from tools.blender_runner import run_blender_script_with_retry
    from tools.storage import upload_render

    # ---- resolve screenshot to a local path --------------------------------
    local_screenshot = screenshot_path

    if not local_screenshot and screenshot_url:
        import httpx
        ext = ".png" if screenshot_url.lower().endswith(".png") else ".jpg"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False, prefix="mockup_shot_") as f:
            dl_path = f.name
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(screenshot_url)
            resp.raise_for_status()
            with open(dl_path, "wb") as f:
                f.write(resp.content)
        local_screenshot = dl_path

    if not local_screenshot and screenshot_spec:
        from tools.svg_export import screenshot_from_spec
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="mockup_spec_") as f:
            spec_path = f.name
        local_screenshot = screenshot_from_spec(screenshot_spec, spec_path)

    # ---- build Blender args ------------------------------------------------
    uid = uuid.uuid4().hex
    if animation == "static":
        output_path = f"/tmp/mockup_{uid}.png"
    else:
        output_path = f"/tmp/mockup_{uid}.mp4"

    blender_args: dict = {
        "output_path": output_path,
        "device": device,
        "animation": animation,
        "duration": duration,
        "fps": fps,
    }
    if local_screenshot:
        blender_args["screenshot_path"] = local_screenshot
    if background_color:
        blender_args["background_color"] = background_color
    if accent_color:
        blender_args["accent_color"] = accent_color

    script_path = _ROOT / "blender_scripts" / "device_mockup.py"
    result = await run_blender_script_with_retry(
        script_content=script_path.read_text(),
        args=blender_args,
        max_attempts=2,
        timeout=600 if animation != "static" else 300,
    )

    # For static renders, Blender writes a .png even if output_path ends in .mp4
    final_path = result.get("output_path", output_path)
    if not os.path.exists(final_path):
        # Blender may have renamed .mp4 → .png for static
        candidate = output_path.replace(".mp4", ".png")
        if os.path.exists(candidate):
            final_path = candidate

    if animation == "static":
        image_url = upload_render(final_path, prefix="mockups")
        try:
            os.unlink(final_path)
        except OSError:
            pass
        return {"image_url": image_url, "device": device, "animation": "static"}
    else:
        video_url = upload_render(final_path, prefix="mockups")
        try:
            os.unlink(final_path)
        except OSError:
            pass
        return {
            "video_url": video_url,
            "device": device,
            "animation": animation,
            "duration": duration,
        }


async def impl_generate_animation(
    description: str,
    duration: float = 10.0,
    background_style: str = "dark",
    composite_over_scene: bool = True,
) -> dict:
    """
    Generate any Manim animation from a natural language description.

    Uses LLM code generation — the agent can request any animation Manim supports,
    not just the 3 hardcoded LaTeX animation types.

    If composite_over_scene=True, the Manim clip (transparent bg) is composited
    over a Blender 3D background scene for a polished final look.
    """
    from tools.manim_codegen import generate_and_run_manim
    from tools.storage import upload_render

    output_tmp = f"/tmp/anim_{uuid.uuid4().hex}"

    if composite_over_scene:
        # Transparent Manim clip → composite over Blender background
        manim_path = output_tmp + "_eq.mov"
        final_path  = output_tmp + ".mp4"

        manim_path = await generate_and_run_manim(
            description=description,
            duration=duration,
            background=background_style,
            output_path=manim_path,
            transparent=True,
            quality="m",
        )

        # Blender background scene
        blender_path = output_tmp + "_bg.mp4"
        from tools.blender_runner import run_blender_script
        blender_script = str(_ROOT / "blender_scripts" / "base_scene.py")
        try:
            result = await run_blender_script(
                script_path=blender_script,
                args={
                    "prompt": description[:200],
                    "duration": duration,
                    "style": "cinematic",
                    "output_path": blender_path,
                },
                timeout=600,
            )
            blender_path = result.get("output_path", blender_path)
        except RuntimeError:
            composite_over_scene = False   # fall through to plain upload

        if composite_over_scene:
            from tools.compositor import composite_manim_over_blender
            final_path = composite_manim_over_blender(
                blender_video_path=blender_path,
                equation_video_path=manim_path,
                output_path=final_path,
                eq_x_position=0.5,
                eq_y_position=0.5,
                eq_scale=1.0,
                fps=60,
            )
            for p in (manim_path, blender_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        else:
            final_path = manim_path
    else:
        final_path = output_tmp + ".mp4"
        await generate_and_run_manim(
            description=description,
            duration=duration,
            background=background_style,
            output_path=final_path,
            transparent=False,
            quality="m",
        )

    video_url = upload_render(final_path, prefix="animations")
    try:
        os.unlink(final_path)
    except OSError:
        pass

    return {
        "video_url": video_url,
        "duration": duration,
        "description": description[:200],
        "composited": composite_over_scene,
    }


async def impl_generate_chart(
    chart_type: str = "bar_chart",
    title: str = "Data",
    data: list | None = None,
    labels: list | None = None,
    duration: float = 10.0,
    y_range: list | None = None,
    colors: list | None = None,
    composite_over_scene: bool = False,
) -> dict:
    """
    Generate a Manim data-visualisation clip.

    chart_type: "bar_chart" | "line_chart" | "pie_chart" | "counter" | "scatter"
    data:       list of numbers (or [x,y] pairs for scatter)
    labels:     list of strings for axis labels or legend entries
    """
    import json as _json
    from tools.manim_runner import run_manim_scene
    from tools.storage import upload_render

    if data is None:
        data = [3, 7, 5, 9, 4, 6]
    if labels is None:
        labels = [str(i + 1) for i in range(len(data))]
    if y_range is None:
        max_val = max((v if isinstance(v, (int, float)) else max(v)) for v in data) if data else 10
        y_range = [0, max_val * 1.25, max(1, round(max_val / 5))]
    if colors is None:
        colors = []

    scene_file = str(_ROOT / "manim_scripts" / "data_chart_scene.py")
    output_path = f"/tmp/chart_{uuid.uuid4().hex}.mp4"

    await run_manim_scene(
        scene_file=scene_file,
        scene_class="DataChartScene",
        args={
            "chart_type": chart_type,
            "title": title,
            "data": data,
            "labels": labels,
            "duration": duration,
            "y_range": y_range,
            "colors": colors,
        },
        quality="m",
        output_path=output_path,
        transparent=False,
        timeout=300,
    )

    video_url = upload_render(output_path, prefix="charts")
    try:
        os.unlink(output_path)
    except OSError:
        pass

    return {
        "video_url": video_url,
        "duration": duration,
        "chart_type": chart_type,
        "title": title,
    }


async def impl_generate_latex(
    latex_expression: str,
    animation_type: str = "appear",
    duration: float = 8.0,
    background_style: str = "dark",
    prompt: str = "",
) -> dict:
    """
    Phase 2 LaTeX pipeline — routes between Option A (SVG→Blender 3D) and
    Option B (Manim transparent → Blender scene → MoviePy composite) via
    the LaTeX Agent.
    """
    from agents.latex_agent import run_latex_agent
    from tools.storage import upload_render

    output_path = f"/tmp/latex_{uuid.uuid4().hex}.mp4"

    result = await run_latex_agent(
        latex_expression=latex_expression,
        animation_type=animation_type,
        duration=duration,
        background_style=background_style,
        prompt=prompt,
        output_path=output_path,
        option="auto",
    )

    if result.get("error"):
        raise RuntimeError(result["error"])

    final_path = result.get("output_path", output_path)
    video_url = upload_render(final_path, prefix="latex")
    try:
        os.unlink(final_path)
    except OSError:
        pass

    return {
        "video_url": video_url,
        "duration": duration,
        "latex_expression": latex_expression,
        "animation_type": animation_type,
        "pipeline": result.get("chosen_option", "A"),  # "A" or "B"
    }


# ---------------------------------------------------------------------------
# Manim: Flowchart
# ---------------------------------------------------------------------------

async def impl_generate_flowchart(
    nodes: list | None = None,
    edges: list | None = None,
    title: str = "Process Flowchart",
    duration: float = 12.0,
    style: str = "dark",
) -> dict:
    from tools.manim_runner import run_manim_scene
    from tools.storage import upload_render

    scene_file  = str(_ROOT / "manim_scripts" / "flowchart_scene.py")
    output_path = f"/tmp/flowchart_{uuid.uuid4().hex}.mp4"

    await run_manim_scene(
        scene_file=scene_file,
        scene_class="FlowchartScene",
        args={"nodes": nodes or [], "edges": edges or [], "title": title, "duration": duration, "style": style},
        quality="m",
        output_path=output_path,
        timeout=300,
    )

    video_url = upload_render(output_path, prefix="flowcharts")
    try:
        import os as _os; _os.unlink(output_path)
    except OSError:
        pass

    return {"video_url": video_url, "duration": duration, "title": title}


# ---------------------------------------------------------------------------
# Manim: 3D Math
# ---------------------------------------------------------------------------

async def impl_generate_3d_math(
    scene_type: str = "surface",
    title: str = "3D Mathematics",
    function: str = "wave",
    duration: float = 12.0,
    color: str = "BLUE",
) -> dict:
    from tools.manim_runner import run_manim_scene
    from tools.storage import upload_render

    scene_file  = str(_ROOT / "manim_scripts" / "threed_math_scene.py")
    output_path = f"/tmp/3dmath_{uuid.uuid4().hex}.mp4"

    await run_manim_scene(
        scene_file=scene_file,
        scene_class="ThreeDMathScene",
        args={"scene_type": scene_type, "title": title, "function": function,
              "duration": duration, "color": color},
        quality="m",
        output_path=output_path,
        timeout=400,
    )

    video_url = upload_render(output_path, prefix="3d_math")
    try:
        import os as _os; _os.unlink(output_path)
    except OSError:
        pass

    return {"video_url": video_url, "duration": duration, "scene_type": scene_type}


# ---------------------------------------------------------------------------
# Manim: Code Animation
# ---------------------------------------------------------------------------

async def impl_generate_code_animation(
    code: str = "",
    language: str = "python",
    title: str = "Code Walkthrough",
    highlight_lines: list | None = None,
    reveal_mode: str = "line_by_line",
    duration: float = 12.0,
    style: str = "monokai",
) -> dict:
    from tools.manim_runner import run_manim_scene
    from tools.storage import upload_render

    scene_file  = str(_ROOT / "manim_scripts" / "code_animation_scene.py")
    output_path = f"/tmp/code_anim_{uuid.uuid4().hex}.mp4"

    await run_manim_scene(
        scene_file=scene_file,
        scene_class="CodeAnimationScene",
        args={"code": code, "language": language, "title": title,
              "highlight_lines": highlight_lines or [], "reveal_mode": reveal_mode,
              "duration": duration, "style": style},
        quality="m",
        output_path=output_path,
        timeout=300,
    )

    video_url = upload_render(output_path, prefix="code_animations")
    try:
        import os as _os; _os.unlink(output_path)
    except OSError:
        pass

    return {"video_url": video_url, "duration": duration, "language": language}


# ---------------------------------------------------------------------------
# Manim: Timeline
# ---------------------------------------------------------------------------

async def impl_generate_timeline(
    events: list | None = None,
    title: str = "Project Timeline",
    duration: float = 12.0,
    style: str = "dark",
    orientation: str = "horizontal",
) -> dict:
    from tools.manim_runner import run_manim_scene
    from tools.storage import upload_render

    scene_file  = str(_ROOT / "manim_scripts" / "timeline_scene.py")
    output_path = f"/tmp/timeline_{uuid.uuid4().hex}.mp4"

    await run_manim_scene(
        scene_file=scene_file,
        scene_class="TimelineScene",
        args={"events": events or [], "title": title, "duration": duration,
              "style": style, "orientation": orientation},
        quality="m",
        output_path=output_path,
        timeout=300,
    )

    video_url = upload_render(output_path, prefix="timelines")
    try:
        import os as _os; _os.unlink(output_path)
    except OSError:
        pass

    return {"video_url": video_url, "duration": duration, "title": title}


# ---------------------------------------------------------------------------
# Manim: Network Graph
# ---------------------------------------------------------------------------

async def impl_generate_network_graph(
    nodes: list | None = None,
    edges: list | None = None,
    title: str = "Network Graph",
    layout: str = "radial",
    duration: float = 12.0,
    style: str = "dark",
) -> dict:
    from tools.manim_runner import run_manim_scene
    from tools.storage import upload_render

    scene_file  = str(_ROOT / "manim_scripts" / "network_graph_scene.py")
    output_path = f"/tmp/netgraph_{uuid.uuid4().hex}.mp4"

    await run_manim_scene(
        scene_file=scene_file,
        scene_class="NetworkGraphScene",
        args={"nodes": nodes or [], "edges": edges or [], "title": title,
              "layout": layout, "duration": duration, "style": style},
        quality="m",
        output_path=output_path,
        timeout=300,
    )

    video_url = upload_render(output_path, prefix="network_graphs")
    try:
        import os as _os; _os.unlink(output_path)
    except OSError:
        pass

    return {"video_url": video_url, "duration": duration, "title": title}


# ---------------------------------------------------------------------------
# Blender: Logo / Text Reveal
# ---------------------------------------------------------------------------

async def impl_generate_logo_reveal(
    text: str = "BRAND",
    logo_text: str = "",
    tagline: str = "",
    style: str = "extrude_reveal",
    color: list | None = None,
    bg_color: list | None = None,
    duration: float = 6.0,
) -> dict:
    from tools.blender_runner import run_blender_script_with_retry
    from tools.storage import upload_render

    script_path = _ROOT / "blender_scripts" / "logo_reveal.py"
    output_path = f"/tmp/logo_{uuid.uuid4().hex}.mp4"
    resolved_text = (text or "").strip() or (logo_text or "").strip() or "BRAND"

    result = await run_blender_script_with_retry(
        script_content=script_path.read_text(),
        args={
            "text": resolved_text,
            "tagline": tagline,
            "style": style,
            "color": color or [0.1, 0.5, 1.0, 1.0],
            "bg_color": bg_color or [0.02, 0.02, 0.05, 1.0],
            "duration": duration,
            "output_path": output_path,
        },
        max_attempts=2,
        timeout=400,
    )

    video_url = upload_render(output_path, prefix="logo_reveals")
    try:
        import os as _os; _os.unlink(output_path)
    except OSError:
        pass

    return {"video_url": video_url, "duration": duration, "text": resolved_text, "style": style}


# ---------------------------------------------------------------------------
# Blender: Abstract Background
# ---------------------------------------------------------------------------

async def impl_generate_abstract_bg(
    style: str = "geometric",
    primary_color: list | None = None,
    secondary_color: list | None = None,
    duration: float = 8.0,
) -> dict:
    from tools.blender_runner import run_blender_script_with_retry
    from tools.storage import upload_render

    script_path = _ROOT / "blender_scripts" / "abstract_bg.py"
    output_path = f"/tmp/abstractbg_{uuid.uuid4().hex}.mp4"

    result = await run_blender_script_with_retry(
        script_content=script_path.read_text(),
        args={
            "style": style,
            "primary_color": primary_color or [0.05, 0.2, 0.8, 1.0],
            "secondary_color": secondary_color or [0.8, 0.1, 0.5, 1.0],
            "duration": duration,
            "output_path": output_path,
        },
        max_attempts=2,
        timeout=500,
    )

    video_url = upload_render(output_path, prefix="abstract_bgs")
    try:
        import os as _os; _os.unlink(output_path)
    except OSError:
        pass

    return {"video_url": video_url, "duration": duration, "style": style}


# ---------------------------------------------------------------------------
# Blender: Countdown Timer
# ---------------------------------------------------------------------------

async def impl_generate_countdown(
    start_number: int = 5,
    end_number: int = 1,
    style: str = "bold",
    color: list | None = None,
    bg_color: list | None = None,
    show_ring: bool = True,
    duration: float | None = None,
) -> dict:
    from tools.blender_runner import run_blender_script_with_retry
    from tools.storage import upload_render

    if duration is None:
        duration = float(abs(start_number - end_number) + 1)

    script_path = _ROOT / "blender_scripts" / "countdown.py"
    output_path = f"/tmp/countdown_{uuid.uuid4().hex}.mp4"

    result = await run_blender_script_with_retry(
        script_content=script_path.read_text(),
        args={
            "start_number": start_number,
            "end_number": end_number,
            "style": style,
            "color": color or [0.1, 0.6, 1.0, 1.0],
            "bg_color": bg_color or [0.02, 0.02, 0.05, 1.0],
            "show_ring": show_ring,
            "duration": duration,
            "output_path": output_path,
        },
        max_attempts=2,
        timeout=400,
    )

    video_url = upload_render(output_path, prefix="countdowns")
    try:
        import os as _os; _os.unlink(output_path)
    except OSError:
        pass

    return {"video_url": video_url, "duration": duration,
            "start_number": start_number, "end_number": end_number}


# ---------------------------------------------------------------------------
# Manim: Text Animation / Kinetic Typography
# ---------------------------------------------------------------------------

async def impl_generate_text_animation(
    text: str = "Make it Count",
    subtitle: str = "",
    mode: str = "letter_by_letter",
    color: str = "WHITE",
    bg_color: str = "dark",
    duration: float = 8.0,
    font_size: int = 72,
    words_to_highlight: list | None = None,
) -> dict:
    from tools.manim_runner import run_manim_scene
    from tools.storage import upload_render

    scene_file  = str(_ROOT / "manim_scripts" / "text_animation_scene.py")
    output_path = f"/tmp/textanim_{uuid.uuid4().hex}.mp4"

    await run_manim_scene(
        scene_file=scene_file,
        scene_class="TextAnimationScene",
        args={"text": text, "subtitle": subtitle, "mode": mode, "color": color,
              "bg_color": bg_color, "duration": duration, "font_size": font_size,
              "words_to_highlight": words_to_highlight or []},
        quality="m",
        output_path=output_path,
        timeout=300,
    )

    video_url = upload_render(output_path, prefix="text_animations")
    try:
        import os as _os; _os.unlink(output_path)
    except OSError:
        pass

    return {"video_url": video_url, "duration": duration, "mode": mode}


# ---------------------------------------------------------------------------
# Manim: Vector Field
# ---------------------------------------------------------------------------

async def impl_generate_vector_field(
    field_type: str = "rotation",
    title: str = "Vector Field",
    duration: float = 12.0,
    show_streams: bool = True,
    color: str = "BLUE",
    style: str = "dark",
) -> dict:
    from tools.manim_runner import run_manim_scene
    from tools.storage import upload_render

    scene_file  = str(_ROOT / "manim_scripts" / "vector_field_scene.py")
    output_path = f"/tmp/vecfield_{uuid.uuid4().hex}.mp4"

    await run_manim_scene(
        scene_file=scene_file,
        scene_class="VectorFieldScene",
        args={"field_type": field_type, "title": title, "duration": duration,
              "show_streams": show_streams, "color": color, "style": style},
        quality="m",
        output_path=output_path,
        timeout=300,
    )

    video_url = upload_render(output_path, prefix="vector_fields")
    try:
        import os as _os; _os.unlink(output_path)
    except OSError:
        pass

    return {"video_url": video_url, "duration": duration, "field_type": field_type}


# ---------------------------------------------------------------------------
# Manim: Matrix Transformation
# ---------------------------------------------------------------------------

async def impl_generate_matrix_transform(
    matrix: list | None = None,
    title: str = "Linear Transformation",
    duration: float = 12.0,
    show_vectors: bool = True,
    show_det: bool = True,
) -> dict:
    from tools.manim_runner import run_manim_scene
    from tools.storage import upload_render

    scene_file  = str(_ROOT / "manim_scripts" / "matrix_transform_scene.py")
    output_path = f"/tmp/matrix_{uuid.uuid4().hex}.mp4"

    await run_manim_scene(
        scene_file=scene_file,
        scene_class="MatrixTransformScene",
        args={"matrix": matrix or [[0, -1], [1, 0]], "title": title,
              "duration": duration, "show_vectors": show_vectors, "show_det": show_det},
        quality="m",
        output_path=output_path,
        timeout=300,
    )

    video_url = upload_render(output_path, prefix="matrix_transforms")
    try:
        import os as _os; _os.unlink(output_path)
    except OSError:
        pass

    return {"video_url": video_url, "duration": duration}


# ---------------------------------------------------------------------------
# Manim: Polar Graph / Complex Plane
# ---------------------------------------------------------------------------

async def impl_generate_polar_graph(
    plane_type: str = "polar",
    title: str = "Polar Graph",
    function: str = "rose",
    k_value: int = 4,
    duration: float = 12.0,
    color: str = "BLUE",
    show_label: bool = True,
) -> dict:
    from tools.manim_runner import run_manim_scene
    from tools.storage import upload_render

    scene_file  = str(_ROOT / "manim_scripts" / "polar_graph_scene.py")
    output_path = f"/tmp/polar_{uuid.uuid4().hex}.mp4"

    await run_manim_scene(
        scene_file=scene_file,
        scene_class="PolarGraphScene",
        args={"plane_type": plane_type, "title": title, "function": function,
              "k_value": k_value, "duration": duration, "color": color, "show_label": show_label},
        quality="m",
        output_path=output_path,
        timeout=300,
    )

    video_url = upload_render(output_path, prefix="polar_graphs")
    try:
        import os as _os; _os.unlink(output_path)
    except OSError:
        pass

    return {"video_url": video_url, "duration": duration, "plane_type": plane_type}


# ---------------------------------------------------------------------------
# Manim: Geometry Proof
# ---------------------------------------------------------------------------

async def impl_generate_geometry_proof(
    proof_type: str = "pythagorean",
    title: str = "Geometry Proof",
    duration: float = 14.0,
    color_a: str = "BLUE",
    color_b: str = "RED",
    show_labels: bool = True,
) -> dict:
    from tools.manim_runner import run_manim_scene
    from tools.storage import upload_render

    scene_file  = str(_ROOT / "manim_scripts" / "geometry_proof_scene.py")
    output_path = f"/tmp/geoproof_{uuid.uuid4().hex}.mp4"

    await run_manim_scene(
        scene_file=scene_file,
        scene_class="GeometryProofScene",
        args={"proof_type": proof_type, "title": title, "duration": duration,
              "color_a": color_a, "color_b": color_b, "show_labels": show_labels},
        quality="m",
        output_path=output_path,
        timeout=400,
    )

    video_url = upload_render(output_path, prefix="geometry_proofs")
    try:
        import os as _os; _os.unlink(output_path)
    except OSError:
        pass

    return {"video_url": video_url, "duration": duration, "proof_type": proof_type}


# ---------------------------------------------------------------------------
# Blender: Particle Confetti / Snow / Stars
# ---------------------------------------------------------------------------

async def impl_generate_particle_confetti(
    style: str = "confetti",
    count: int = 400,
    duration: float = 6.0,
    primary_color=None,
    secondary_color=None,
    bg_color=None,
) -> dict:
    from tools.blender_runner import run_blender_script
    from tools.storage import upload_render

    output_path = f"/tmp/confetti_{uuid.uuid4().hex}.mp4"
    args = {
        "style": style,
        "count": count,
        "duration": duration,
        "output_path": output_path,
    }
    if primary_color:   args["primary_color"]   = primary_color
    if secondary_color: args["secondary_color"]  = secondary_color
    if bg_color:        args["bg_color"]         = bg_color

    script = str(_ROOT / "blender_scripts" / "particle_confetti.py")
    await run_blender_script(script, args, timeout=600)
    video_url = upload_render(output_path, prefix="particle_confetti")
    try:
        import os as _os; _os.unlink(output_path)
    except OSError:
        pass
    return {"video_url": video_url, "duration": duration, "style": style}


# ---------------------------------------------------------------------------
# Blender: Rigid Body Drop
# ---------------------------------------------------------------------------

async def impl_generate_rigid_body_drop(
    text: str = "DROP",
    object_type: str = "text",
    count: int = 12,
    duration: float = 5.0,
    color=None,
    bg_color=None,
    style: str = "dark",
) -> dict:
    from tools.blender_runner import run_blender_script
    from tools.storage import upload_render

    output_path = f"/tmp/rigidbody_{uuid.uuid4().hex}.mp4"
    args = {
        "text": text,
        "object_type": object_type,
        "count": count,
        "duration": duration,
        "style": style,
        "output_path": output_path,
    }
    if color:    args["color"]    = color
    if bg_color: args["bg_color"] = bg_color

    script = str(_ROOT / "blender_scripts" / "rigid_body_drop.py")
    await run_blender_script(script, args, timeout=600)
    video_url = upload_render(output_path, prefix="rigid_body_drop")
    try:
        import os as _os; _os.unlink(output_path)
    except OSError:
        pass
    return {"video_url": video_url, "duration": duration, "text": text}


# ---------------------------------------------------------------------------
# Blender: Camera Path / Fly-through
# ---------------------------------------------------------------------------

async def impl_generate_camera_path(
    path_type: str = "orbit",
    subject: str = "abstract",
    title: str = "",
    duration: float = 8.0,
    color=None,
    bg_color=None,
    style: str = "cinematic",
) -> dict:
    from tools.blender_runner import run_blender_script
    from tools.storage import upload_render

    output_path = f"/tmp/campath_{uuid.uuid4().hex}.mp4"
    args = {
        "path_type": path_type,
        "subject": subject,
        "title": title,
        "duration": duration,
        "style": style,
        "output_path": output_path,
    }
    if color:    args["color"]    = color
    if bg_color: args["bg_color"] = bg_color

    script = str(_ROOT / "blender_scripts" / "camera_path.py")
    await run_blender_script(script, args, timeout=600)
    video_url = upload_render(output_path, prefix="camera_path")
    try:
        import os as _os; _os.unlink(output_path)
    except OSError:
        pass
    return {"video_url": video_url, "duration": duration, "path_type": path_type}


# ---------------------------------------------------------------------------
# Blender: Toon / NPR Cartoon Scene
# ---------------------------------------------------------------------------

async def impl_generate_toon_scene(
    subject: str = "abstract",
    title: str = "",
    duration: float = 6.0,
    outline_color=None,
    primary_color=None,
    bg_color=None,
    outline_width: float = 1.5,
    flat_shading: bool = True,
    animated: bool = True,
) -> dict:
    from tools.blender_runner import run_blender_script
    from tools.storage import upload_render

    output_path = f"/tmp/toon_{uuid.uuid4().hex}.mp4"
    args = {
        "subject": subject,
        "title": title,
        "duration": duration,
        "outline_width": outline_width,
        "flat_shading": flat_shading,
        "animated": animated,
        "output_path": output_path,
    }
    if outline_color:  args["outline_color"]  = outline_color
    if primary_color:  args["primary_color"]  = primary_color
    if bg_color:       args["bg_color"]       = bg_color

    script = str(_ROOT / "blender_scripts" / "toon_scene.py")
    await run_blender_script(script, args, timeout=600)
    video_url = upload_render(output_path, prefix="toon_scene")
    try:
        import os as _os; _os.unlink(output_path)
    except OSError:
        pass
    return {"video_url": video_url, "duration": duration, "subject": subject}


# ---------------------------------------------------------------------------
# Blender: Grease Pencil Whiteboard Reveal
# ---------------------------------------------------------------------------

async def impl_generate_grease_pencil_reveal(
    text: str = "HELLO",
    style: str = "whiteboard",
    duration: float = 6.0,
    color=None,
    bg_color=None,
    stroke_width: int = 50,
) -> dict:
    from tools.blender_runner import run_blender_script
    from tools.storage import upload_render

    output_path = f"/tmp/gp_reveal_{uuid.uuid4().hex}.mp4"
    args = {
        "text": text,
        "style": style,
        "duration": duration,
        "stroke_width": stroke_width,
        "output_path": output_path,
    }
    if color:    args["color"]    = color
    if bg_color: args["bg_color"] = bg_color

    script = str(_ROOT / "blender_scripts" / "grease_pencil_reveal.py")
    await run_blender_script(script, args, timeout=600)
    video_url = upload_render(output_path, prefix="gp_reveal")
    try:
        import os as _os; _os.unlink(output_path)
    except OSError:
        pass
    return {"video_url": video_url, "duration": duration, "text": text, "style": style}


# ---------------------------------------------------------------------------
# Blender: Geometry Scatter
# ---------------------------------------------------------------------------

async def impl_generate_geometry_scatter(
    instance_type: str = "spheres",
    surface: str = "plane",
    count: int = 200,
    duration: float = 8.0,
    primary_color=None,
    secondary_color=None,
    bg_color=None,
    animated: bool = True,
    scale: float = 1.0,
) -> dict:
    from tools.blender_runner import run_blender_script
    from tools.storage import upload_render

    output_path = f"/tmp/scatter_{uuid.uuid4().hex}.mp4"
    args = {
        "instance_type": instance_type,
        "surface": surface,
        "count": count,
        "duration": duration,
        "animated": animated,
        "scale": scale,
        "output_path": output_path,
    }
    if primary_color:   args["primary_color"]   = primary_color
    if secondary_color: args["secondary_color"]  = secondary_color
    if bg_color:        args["bg_color"]         = bg_color

    script = str(_ROOT / "blender_scripts" / "geometry_scatter.py")
    await run_blender_script(script, args, timeout=600)
    video_url = upload_render(output_path, prefix="geometry_scatter")
    try:
        import os as _os; _os.unlink(output_path)
    except OSError:
        pass
    return {"video_url": video_url, "duration": duration, "instance_type": instance_type, "surface": surface}
