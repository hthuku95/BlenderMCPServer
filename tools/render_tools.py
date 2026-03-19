"""
Shared render tool implementations.

Imported by both server.py (REST endpoint) and agents/director.py (LangGraph).
Each impl_* function drives a Blender or Manim script, uploads the result to R2,
and returns a plain dict with a video_url or image_url.
"""
import os
import uuid
from pathlib import Path

# Root of the BlenderMCPServer package
_ROOT = Path(__file__).parent.parent


async def impl_generate_scene(
    prompt: str,
    duration: float = 10.0,
    style: str = "cinematic",
    reference_image_url: str = "",
) -> dict:
    from tools.blender_runner import run_blender_script_with_retry
    from tools.storage import upload_render

    script_path = _ROOT / "blender_scripts" / "base_scene.py"
    output_path = f"/tmp/blender_{uuid.uuid4().hex}.mp4"

    args = {"prompt": prompt, "duration": duration, "style": style, "output_path": output_path}
    if reference_image_url:
        args["reference_image_url"] = reference_image_url

    result = await run_blender_script_with_retry(
        script_content=script_path.read_text(),
        args=args,
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


async def impl_generate_latex(
    latex_expression: str,
    animation_type: str = "appear",
    duration: float = 8.0,
    background_style: str = "dark",
) -> dict:
    from tools.manim_runner import run_manim_scene
    from tools.storage import upload_render

    scene_file = str(_ROOT / "manim_scripts" / "latex_scene.py")
    output_path = f"/tmp/latex_{uuid.uuid4().hex}.mp4"

    rendered = await run_manim_scene(
        scene_file=scene_file,
        scene_class="LatexScene",
        args={
            "latex_expression": latex_expression,
            "animation_type": animation_type,
            "duration": duration,
            "background_style": background_style,
        },
        quality="m",
        output_path=output_path,
        timeout=300,
    )

    video_url = upload_render(rendered, prefix="latex")
    try:
        os.unlink(rendered)
    except OSError:
        pass

    return {
        "video_url": video_url,
        "duration": duration,
        "latex_expression": latex_expression,
        "animation_type": animation_type,
    }
