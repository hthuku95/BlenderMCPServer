"""
Vision Agent — Plan Phase 3.

Analyses a reference image via Claude vision, then drives the Blender renderer
to reproduce the look using the appropriate reference mode (1/2/3).

Pipeline:
  1. analyse_reference_image()  → structured scene params
  2. run_blender_script(reference_mode.py, params)  → rendered MP4
  3. [QA loop handled by qa_agent]

This agent is used when blender_generate_scene() receives a `reference_image_url`.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph


logger = logging.getLogger(__name__)


_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class VisionState(TypedDict):
    # Inputs
    prompt: str
    reference_image_url: str
    duration: float
    style: str
    output_path: str
    # Filled by agent
    reference_image_path: str   # local download of reference URL
    scene_params: dict          # from analyse_reference_image
    render_result: dict
    error: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def _download_reference_node(state: VisionState) -> VisionState:
    """Download the reference image to a local temp file."""
    url = state["reference_image_url"]
    if os.path.exists(url):
        # already a local path
        return {**state, "reference_image_path": url, "error": ""}

    logger.info("vision_agent.reference_download_start url=%s", url)
    try:
        import httpx

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
            headers=_DOWNLOAD_HEADERS,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except Exception as e:
        logger.exception("vision_agent.reference_download_failed url=%s error=%s", url, e)
        return {**state, "error": f"Failed to download reference image: {e}"}

    suffix = ".jpg"
    if "png" in resp.headers.get("content-type", ""):
        suffix = ".png"
    elif "webp" in resp.headers.get("content-type", ""):
        suffix = ".webp"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="ref_img_") as f:
        f.write(resp.content)
        local_path = f.name

    logger.info(
        "vision_agent.reference_download_complete url=%s content_type=%s bytes=%s local_path=%s",
        url,
        resp.headers.get("content-type", ""),
        len(resp.content),
        local_path,
    )

    return {**state, "reference_image_path": local_path, "error": ""}


async def _analyse_reference_node(state: VisionState) -> VisionState:
    """Run Claude vision analysis on the reference image."""
    from tools.vision_tools import analyse_reference_image

    if state.get("error"):
        return state

    try:
        params = analyse_reference_image(state["reference_image_path"])
    except Exception as e:
        return {**state, "error": f"Vision analysis failed: {e}"}

    logger.info(
        "vision_agent.reference_analyzed provider=%s mode=%s camera=%s movement=%s material=%s layers=%s verify=%s",
        params.get("_provider", "unknown"),
        params.get("blender_reference_mode", 2),
        params.get("camera_angle", "front"),
        params.get("movement_style", "slow_push"),
        params.get("material_style", "mixed"),
        params.get("scene_layers", []),
        params.get("verification_focus", []),
    )

    return {**state, "scene_params": params, "error": ""}


async def _render_with_reference_node(state: VisionState) -> VisionState:
    """Render the Blender scene using the vision-derived parameters."""
    from tools.blender_runner import run_blender_script

    if state.get("error"):
        return state

    params = state.get("scene_params", {})
    output_mp4 = state.get("output_path") or f"/tmp/vision_render_{os.getpid()}.mp4"

    script = str(
        Path(__file__).parent.parent / "blender_scripts" / "reference_mode.py"
    )

    try:
        result = await run_blender_script(
            script_path=script,
            args={
                "output_path": output_mp4,
                "duration": state.get("duration", 10.0),
                "fps": 60,
                "reference_image_path": state.get("reference_image_path", ""),
                "mode": params.get("blender_reference_mode", 2),
                "dominant_colors": params.get("dominant_colors", ["#1a1a2e"]),
                "lighting_type": params.get("lighting_type", "studio"),
                "camera_angle": params.get("camera_angle", "front"),
                "mood": params.get("mood", "cinematic"),
                "key_objects": params.get("key_objects", []),
                "composition_focus": params.get("composition_focus", "centered"),
                "movement_style": params.get("movement_style", "slow_push"),
                "material_style": params.get("material_style", "mixed"),
                "scene_layers": params.get("scene_layers", []),
                "verification_focus": params.get("verification_focus", []),
                "notes": params.get("notes", ""),
                "corrections": {},
                "prompt": state.get("prompt", ""),
            },
            timeout=900,
        )
    except RuntimeError as e:
        return {**state, "error": f"Blender reference render failed: {e}"}

    logger.info(
        "vision_agent.reference_render_complete output=%s mode=%s camera=%s movement=%s frames=%s",
        result.get("output_path", output_mp4),
        params.get("blender_reference_mode", 2),
        params.get("camera_angle", "front"),
        params.get("movement_style", "slow_push"),
        result.get("frames"),
    )

    return {
        **state,
        "render_result": result,
        "output_path": result.get("output_path", output_mp4),
        "error": "",
    }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_vision_graph() -> StateGraph:
    g = StateGraph(VisionState)
    g.add_node("download_reference", _download_reference_node)
    g.add_node("analyse_reference", _analyse_reference_node)
    g.add_node("render_with_reference", _render_with_reference_node)

    g.set_entry_point("download_reference")
    g.add_edge("download_reference", "analyse_reference")
    g.add_edge("analyse_reference", "render_with_reference")
    g.add_edge("render_with_reference", END)

    return g.compile()


_GRAPH = None


async def run_vision_agent(
    prompt: str,
    reference_image_url: str,
    duration: float = 10.0,
    style: str = "cinematic",
    output_path: str | None = None,
) -> dict:
    """
    Run the vision-guided Blender render pipeline.

    Returns:
        {
          "output_path": str,
          "scene_params": dict,   # vision analysis results
          "error": str            # empty on success
        }
    """
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_vision_graph()

    initial: VisionState = {
        "prompt": prompt,
        "reference_image_url": reference_image_url,
        "duration": duration,
        "style": style,
        "output_path": output_path or "",
        "reference_image_path": "",
        "scene_params": {},
        "render_result": {},
        "error": "",
    }

    final = await _GRAPH.ainvoke(initial)
    return {
        "output_path": final.get("output_path", ""),
        "scene_params": final.get("scene_params", {}),
        "error": final.get("error", ""),
    }
