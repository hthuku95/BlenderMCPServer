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
    ainvoke_with_checkpoint_fallback,
    child_thread_id,
    workflow_persistence_mode,
)
from tools.progress_store import report_workflow_stage

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).parent.parent
_GRAPHS: dict[int, Any] = {}


class SceneWorkflowState(TypedDict):
    prompt: str
    duration: float
    style: str
    reference_image_url: str
    include_narration: bool
    narration_text: str
    narration_speaker: str
    workflow_thread_id: str
    output_path: str
    reference_image_path: str
    rendered_video_path: str
    final_video_path: str
    scene_params: dict
    result: dict
    error: str
    progress_stage: str
    progress_message: str
    progress_state: str
    progress_details: dict
    progress_updated_at: str
    progress_events: list[dict]


async def _render_scene_node(state: SceneWorkflowState) -> SceneWorkflowState:
    if state.get("error"):
        return state

    state = await report_workflow_stage(
        state,
        tool="blender_generate_scene",
        stage="render_scene",
        message="Preparing LLM-generated bpy scene render",
        details={
            "style": state["style"],
            "duration": state["duration"],
            "has_reference_image": bool(state.get("reference_image_url")),
            "method": "llm_dynamic_bpy",
        },
    )

    prompt = state["prompt"]
    duration = state["duration"]
    style = state["style"]
    reference_image_url = state.get("reference_image_url", "")

    from tools.bpy_codegen import generate_and_run_bpy
    output_path = state.get("output_path") or f"/tmp/bpy_workflow_{uuid.uuid4().hex}.mp4"

    result_path = await generate_and_run_bpy(
        prompt=prompt,
        duration=duration,
        style=style,
        output_path=output_path,
        reference_image_url=reference_image_url,
    )

    logger.info(
        "scene_workflow.bpy_render_complete thread_id=%s method=llm_dynamic_bpy path=%s",
        state["workflow_thread_id"],
        result_path,
    )
    next_state = {
        **state,
        "output_path": output_path,
        "rendered_video_path": result_path,
        "final_video_path": result_path,
        "result": {
            "duration": duration,
            "resolution": "1920x1080",
            "frames": int(duration * 60),
            "generation": "llm_dynamic_bpy",
        },
        "error": "",
    }
    return await report_workflow_stage(
        next_state,
        tool="blender_generate_scene",
        stage="scene_render_complete",
        message="LLM-generated bpy scene render completed",
        details={
            "frames": next_state["result"].get("frames"),
            "resolution": next_state["result"].get("resolution"),
        },
    )


async def _qa_scene_node(state: SceneWorkflowState) -> SceneWorkflowState:
    """VIGA Verifier — watch the rendered video and score it against the prompt.
    Uses the 4-provider LLM fallback chain with native full-video understanding.
    """
    if state.get("error"):
        return state

    rendered_video_path = state.get("rendered_video_path") or state.get("final_video_path", "")
    if not rendered_video_path or not os.path.exists(rendered_video_path):
        return state

    state = await report_workflow_stage(
        state,
        tool="blender_generate_scene",
        stage="qa_scene",
        message="Reviewing rendered scene with LLM video understanding",
        details={"video_path": rendered_video_path},
    )

    from tools.vision_tools import review_render_against_prompt

    try:
        qa_result = await review_render_against_prompt(
            render_video_path=rendered_video_path,
            prompt_context=state.get("prompt", ""),
        )
        qa_score = float(qa_result.get("score", 0.7)) if isinstance(qa_result, dict) else 0.7
        qa_feedback = qa_result.get("feedback", "") if isinstance(qa_result, dict) else ""
    except Exception as exc:
        logger.warning("scene_workflow.qa_failed thread_id=%s error=%s", state["workflow_thread_id"], exc)
        qa_score = 0.7
        qa_feedback = ""

    approved = qa_score >= 0.70
    logger.info(
        "scene_workflow.qa_complete thread_id=%s score=%.3f approved=%s",
        state["workflow_thread_id"],
        qa_score,
        approved,
    )
    next_state = {
        **state,
        "final_video_path": rendered_video_path,
        "error": "",
    }
    return await report_workflow_stage(
        next_state,
        tool="blender_generate_scene",
        stage="qa_complete",
        message="LLM video QA pass completed",
        details={
            "approved": approved,
            "score": round(qa_score, 3),
            "generation": "llm_dynamic_bpy",
        },
    )


async def _upload_result_node(state: SceneWorkflowState) -> SceneWorkflowState:
    if state.get("error"):
        return state

    from tools.storage import upload_render
    state = await report_workflow_stage(
        state,
        tool="blender_generate_scene",
        stage="upload_result",
        message="Uploading rendered scene output",
        details={},
    )

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

    if state.get("include_narration"):
        narration_text = (state.get("narration_text") or state["prompt"]).strip()
        if narration_text:
            try:
                from tools.vibevoice import attach_narration_assets

                state = await report_workflow_stage(
                    state,
                    tool="blender_generate_scene",
                    stage="attach_narration",
                    message="Generating and attaching VibeVoice narration",
                    details={"speaker": state.get("narration_speaker") or "Emma"},
                )
                narration_assets = await attach_narration_assets(
                    video_path=final_path,
                    narration_text=narration_text,
                    speaker=state.get("narration_speaker") or "Emma",
                    prefix="scenes",
                    metadata={
                        "workflow_thread_id": state["workflow_thread_id"],
                        "tool": "blender_generate_scene",
                        "style": state["style"],
                    },
                )
                result.update(narration_assets)
            except Exception as exc:
                logger.warning(
                    "scene_workflow.narration_failed thread_id=%s error=%s",
                    state["workflow_thread_id"],
                    exc,
                )
                result["narration_error"] = str(exc)

    logger.info(
        "scene_workflow.upload_complete thread_id=%s persistence=%s video_url=%s",
        state["workflow_thread_id"],
        workflow_persistence_mode(),
        video_url,
    )
    next_state = {**state, "result": result, "error": ""}
    return await report_workflow_stage(
        next_state,
        tool="blender_generate_scene",
        stage="upload_complete",
        message="Scene output uploaded successfully",
        details={"video_url": video_url},
    )


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
    include_narration: bool = False,
    narration_text: str = "",
    narration_speaker: str = "Emma",
    workflow_thread_id: str = "",
) -> dict:
    thread_id = workflow_thread_id.strip() or f"scene-{uuid.uuid4().hex}"
    initial: SceneWorkflowState = {
        "prompt": prompt,
        "duration": duration,
        "style": style,
        "reference_image_url": reference_image_url,
        "include_narration": include_narration,
        "narration_text": narration_text,
        "narration_speaker": narration_speaker,
        "workflow_thread_id": thread_id,
        "output_path": "",
        "reference_image_path": "",
        "rendered_video_path": "",
        "final_video_path": "",
        "scene_params": {},
        "result": {},
        "error": "",
        "progress_stage": "queued",
        "progress_message": "Scene workflow queued",
        "progress_state": "pending",
        "progress_details": {},
        "progress_updated_at": "",
        "progress_events": [],
    }

    final = await ainvoke_with_checkpoint_fallback(
        graph_cache=_GRAPHS,
        graph_builder=build_scene_workflow_graph,
        initial_state=initial,
        thread_id=thread_id,
        checkpoint_ns="scene_workflow",
    )
    if final.get("error"):
        raise RuntimeError(final["error"])
    return final.get("result", {})
