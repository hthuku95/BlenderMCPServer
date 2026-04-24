"""
Durable scene workflow for Blender scene generation.

This keeps the public tool contract unchanged while moving the long-running
reference-render pipeline behind a LangGraph workflow that can use checkpoints
when Postgres persistence is configured.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from tools.workflow_runtime import (
    child_thread_id,
    get_checkpointer,
    workflow_config,
    workflow_persistence_mode,
)

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).parent.parent
_GRAPHS: dict[int, Any] = {}


class SceneWorkflowState(TypedDict):
    prompt: str
    duration: float
    style: str
    reference_image_url: str
    workflow_thread_id: str
    output_path: str
    reference_image_path: str
    rendered_video_path: str
    final_video_path: str
    scene_params: dict
    result: dict
    error: str


async def _render_scene_node(state: SceneWorkflowState) -> SceneWorkflowState:
    if state.get("error"):
        return state

    prompt = state["prompt"]
    duration = state["duration"]
    style = state["style"]
    reference_image_url = state.get("reference_image_url", "")

    if reference_image_url:
        from agents.vision_agent import run_vision_agent

        output_path = state.get("output_path") or f"/tmp/blender_vision_{uuid.uuid4().hex}.mp4"
        vision_result = await run_vision_agent(
            prompt=prompt,
            reference_image_url=reference_image_url,
            duration=duration,
            style=style,
            output_path=output_path,
            thread_id=child_thread_id(state["workflow_thread_id"], "vision"),
        )
        if vision_result.get("error"):
            return {**state, "error": vision_result["error"]}

        scene_params = vision_result.get("scene_params", {})
        logger.info(
            "scene_workflow.reference_render_complete thread_id=%s mode=%s camera=%s movement=%s",
            state["workflow_thread_id"],
            scene_params.get("blender_reference_mode", 2),
            scene_params.get("camera_angle", "front"),
            scene_params.get("movement_style", "slow_push"),
        )
        return {
            **state,
            "output_path": vision_result.get("output_path", output_path),
            "rendered_video_path": vision_result.get("output_path", output_path),
            "final_video_path": vision_result.get("output_path", output_path),
            "reference_image_path": vision_result.get("reference_image_path", ""),
            "scene_params": scene_params,
            "error": "",
        }

    from tools.blender_runner import run_blender_script_with_retry

    script_path = _ROOT / "blender_scripts" / "base_scene.py"
    output_path = state.get("output_path") or f"/tmp/blender_{uuid.uuid4().hex}.mp4"
    result = await run_blender_script_with_retry(
        script_content=script_path.read_text(),
        args={
            "prompt": prompt,
            "duration": duration,
            "style": style,
            "output_path": output_path,
        },
        max_attempts=3,
        timeout=600,
    )
    logger.info(
        "scene_workflow.plain_render_complete thread_id=%s frames=%s resolution=%s",
        state["workflow_thread_id"],
        result.get("frames"),
        result.get("resolution", "1920x1080"),
    )
    return {
        **state,
        "output_path": output_path,
        "rendered_video_path": output_path,
        "final_video_path": output_path,
        "result": {
            "duration": result.get("duration", duration),
            "resolution": result.get("resolution", "1920x1080"),
            "frames": result.get("frames", int(duration * 24)),
        },
        "error": "",
    }


async def _qa_scene_node(state: SceneWorkflowState) -> SceneWorkflowState:
    if state.get("error"):
        return state

    if not state.get("reference_image_url"):
        return state

    reference_image_path = state.get("reference_image_path", "")
    rendered_video_path = state.get("rendered_video_path", "")
    scene_params = state.get("scene_params", {})

    if not reference_image_path or not os.path.exists(reference_image_path) or not os.path.exists(rendered_video_path):
        logger.warning(
            "scene_workflow.qa_skipped thread_id=%s has_reference=%s render_exists=%s",
            state["workflow_thread_id"],
            bool(reference_image_path and os.path.exists(reference_image_path)),
            bool(rendered_video_path and os.path.exists(rendered_video_path)),
        )
        return state

    from agents.qa_agent import run_qa_agent

    qa_result = await run_qa_agent(
        render_video_path=rendered_video_path,
        reference_image_path=reference_image_path,
        blender_script_path=str(_ROOT / "blender_scripts" / "reference_mode.py"),
        blender_args={
            "output_path": state.get("output_path") or f"/tmp/blender_vision_{uuid.uuid4().hex}.mp4",
            "duration": state["duration"],
            "fps": 60,
            "reference_image_path": reference_image_path,
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
            "prompt": state["prompt"],
        },
        prompt_context=(
            f"{state['prompt']}\n"
            f"Verification focus: {', '.join(scene_params.get('verification_focus', [])) or 'preserve brand silhouette, hero subject, and palette'}.\n"
            f"Scene layers: {', '.join(scene_params.get('scene_layers', [])) or 'foreground, subject, support, background'}.\n"
            f"Planner notes: {scene_params.get('notes', '')}"
        ).strip(),
        max_iterations=3,
        thread_id=child_thread_id(state["workflow_thread_id"], "qa"),
    )
    logger.info(
        "scene_workflow.qa_complete thread_id=%s approved=%s best_score=%.3f iterations=%s error=%s",
        state["workflow_thread_id"],
        qa_result.get("approved", False),
        float(qa_result.get("best_score", 0.0)),
        qa_result.get("iterations", 0),
        qa_result.get("error", ""),
    )
    return {
        **state,
        "final_video_path": qa_result.get("best_video_path", rendered_video_path),
        "error": qa_result.get("error", ""),
    }


async def _upload_result_node(state: SceneWorkflowState) -> SceneWorkflowState:
    if state.get("error"):
        return state

    from tools.storage import upload_render

    final_path = state.get("final_video_path") or state.get("rendered_video_path") or state.get("output_path", "")
    if not final_path or not os.path.exists(final_path):
        return {**state, "error": f"Rendered output missing: {final_path}"}

    video_url = upload_render(final_path, prefix="scenes")
    result = dict(state.get("result", {}))
    result.update(
        {
            "video_url": video_url,
            "duration": result.get("duration", state["duration"]),
            "resolution": result.get("resolution", "1920x1080"),
            "frames": result.get("frames", int(state["duration"] * 60)),
        }
    )
    if state.get("reference_image_url"):
        result["reference_mode"] = state.get("scene_params", {}).get("blender_reference_mode", 2)

    logger.info(
        "scene_workflow.upload_complete thread_id=%s persistence=%s video_url=%s",
        state["workflow_thread_id"],
        workflow_persistence_mode(),
        video_url,
    )
    return {**state, "result": result, "error": ""}


async def _cleanup_node(state: SceneWorkflowState) -> SceneWorkflowState:
    for path in {
        state.get("rendered_video_path", ""),
        state.get("final_video_path", ""),
    }:
        if not path:
            continue
        try:
            os.unlink(path)
        except OSError:
            pass

    reference_path = state.get("reference_image_path", "")
    if reference_path and os.path.exists(reference_path) and os.path.basename(reference_path).startswith("ref_img_"):
        try:
            os.unlink(reference_path)
        except OSError:
            pass

    return state


def build_scene_workflow_graph(checkpointer: Any) -> Any:
    graph = StateGraph(SceneWorkflowState)
    graph.add_node("render_scene", _render_scene_node)
    graph.add_node("qa_scene", _qa_scene_node)
    graph.add_node("upload_result", _upload_result_node)
    graph.add_node("cleanup", _cleanup_node)
    graph.set_entry_point("render_scene")
    graph.add_edge("render_scene", "qa_scene")
    graph.add_edge("qa_scene", "upload_result")
    graph.add_edge("upload_result", "cleanup")
    graph.add_edge("cleanup", END)
    return graph.compile(checkpointer=checkpointer)


async def run_scene_workflow(
    prompt: str,
    duration: float = 10.0,
    style: str = "cinematic",
    reference_image_url: str = "",
    workflow_thread_id: str = "",
) -> dict:
    checkpointer = await get_checkpointer()
    graph = _GRAPHS.get(id(checkpointer))
    if graph is None:
        graph = build_scene_workflow_graph(checkpointer)
        _GRAPHS[id(checkpointer)] = graph

    thread_id = workflow_thread_id.strip() or f"scene-{uuid.uuid4().hex}"
    initial: SceneWorkflowState = {
        "prompt": prompt,
        "duration": duration,
        "style": style,
        "reference_image_url": reference_image_url,
        "workflow_thread_id": thread_id,
        "output_path": "",
        "reference_image_path": "",
        "rendered_video_path": "",
        "final_video_path": "",
        "scene_params": {},
        "result": {},
        "error": "",
    }

    final = await graph.ainvoke(initial, config=workflow_config(thread_id, "scene_workflow"))
    if final.get("error"):
        raise RuntimeError(final["error"])
    return final.get("result", {})
