"""
Durable LangGraph workflow for freeform Manim animation generation.

This mirrors the scene workflow's persistence model so long-running Manim jobs
can survive process restarts when Postgres checkpointing is configured.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from tools.workflow_runtime import get_checkpointer, workflow_config, workflow_persistence_mode
from tools.progress_store import report_workflow_stage

logger = logging.getLogger(__name__)
_GRAPHS: dict[int, Any] = {}


class ManimWorkflowState(TypedDict):
    description: str
    duration: float
    background_style: str
    composite_over_scene: bool
    include_narration: bool
    narration_text: str
    narration_speaker: str
    workflow_thread_id: str
    output_tmp: str
    manim_path: str
    blender_path: str
    final_path: str
    result: dict
    error: str
    progress_stage: str
    progress_message: str
    progress_state: str
    progress_details: dict
    progress_updated_at: str
    progress_events: list[dict]


async def _render_manim_node(state: ManimWorkflowState) -> ManimWorkflowState:
    if state.get("error"):
        return state

    from tools.manim_codegen import generate_and_run_manim
    state = await report_workflow_stage(
        state,
        tool="blender_generate_animation",
        stage="render_manim",
        message="Generating Manim animation from the prompt",
        details={
            "duration": state["duration"],
            "background_style": state["background_style"],
            "composite_over_scene": state.get("composite_over_scene", True),
        },
    )

    output_tmp = state.get("output_tmp") or f"/tmp/anim_{uuid.uuid4().hex}"
    if state.get("composite_over_scene", True):
        manim_path = output_tmp + "_eq.mov"
        manim_path = await generate_and_run_manim(
            description=state["description"],
            duration=state["duration"],
            background=state["background_style"],
            output_path=manim_path,
            transparent=True,
            quality="m",
        )
    else:
        manim_path = output_tmp + ".mp4"
        await generate_and_run_manim(
            description=state["description"],
            duration=state["duration"],
            background=state["background_style"],
            output_path=manim_path,
            transparent=False,
            quality="m",
        )

    logger.info(
        "manim_workflow.render_complete thread_id=%s composite=%s persistence=%s",
        state["workflow_thread_id"],
        state.get("composite_over_scene", True),
        workflow_persistence_mode(),
    )
    next_state = {**state, "output_tmp": output_tmp, "manim_path": manim_path, "final_path": manim_path, "error": ""}
    return await report_workflow_stage(
        next_state,
        tool="blender_generate_animation",
        stage="manim_render_complete",
        message="Manim animation render completed",
        details={"manim_path": manim_path},
    )


async def _composite_node(state: ManimWorkflowState) -> ManimWorkflowState:
    if state.get("error") or not state.get("composite_over_scene", True):
        return state

    from tools.blender_runner import run_blender_script
    from tools.compositor import composite_manim_over_blender
    from pathlib import Path
    state = await report_workflow_stage(
        state,
        tool="blender_generate_animation",
        stage="render_background_scene",
        message="Rendering Blender background scene for the animation",
        details={},
    )

    blender_script = str(Path(__file__).parent.parent / "blender_scripts" / "base_scene.py")
    blender_path = state.get("output_tmp", f"/tmp/anim_{uuid.uuid4().hex}") + "_bg.mp4"

    try:
        result = await run_blender_script(
            script_path=blender_script,
            args={
                "prompt": state["description"][:200],
                "duration": state["duration"],
                "style": "cinematic",
                "output_path": blender_path,
            },
            timeout=600,
        )
        blender_path = result.get("output_path", blender_path)
    except RuntimeError as exc:
        logger.warning(
            "manim_workflow.background_failed thread_id=%s error=%s",
            state["workflow_thread_id"],
            exc,
        )
        return {**state, "composite_over_scene": False, "final_path": state.get("manim_path", ""), "error": ""}

    final_path = state.get("output_tmp", f"/tmp/anim_{uuid.uuid4().hex}") + ".mp4"
    state = await report_workflow_stage(
        state,
        tool="blender_generate_animation",
        stage="composite_animation",
        message="Compositing Manim animation over Blender background",
        details={"blender_path": blender_path},
    )
    loop = asyncio.get_running_loop()
    final_path = await loop.run_in_executor(
        None,
        lambda: composite_manim_over_blender(
            blender_video_path=blender_path,
            equation_video_path=state["manim_path"],
            output_path=final_path,
            eq_x_position=0.5,
            eq_y_position=0.5,
            eq_scale=1.0,
            fps=60,
        ),
    )

    logger.info(
        "manim_workflow.composite_complete thread_id=%s blender_path=%s",
        state["workflow_thread_id"],
        blender_path,
    )
    next_state = {
        **state,
        "blender_path": blender_path,
        "final_path": final_path,
        "error": "",
    }
    return await report_workflow_stage(
        next_state,
        tool="blender_generate_animation",
        stage="composite_complete",
        message="Animation compositing completed",
        details={"final_path": final_path},
    )


async def _upload_node(state: ManimWorkflowState) -> ManimWorkflowState:
    if state.get("error"):
        return state

    from tools.storage import upload_render
    state = await report_workflow_stage(
        state,
        tool="blender_generate_animation",
        stage="upload_output",
        message="Uploading animation output",
        details={},
    )

    final_path = state.get("final_path", "")
    if not final_path or not os.path.exists(final_path):
        return {**state, "error": f"Rendered output missing: {final_path}"}

    video_url = upload_render(final_path, prefix="animations")
    result = {
        "video_url": video_url,
        "duration": state["duration"],
        "description": state["description"][:200],
        "composited": state.get("composite_over_scene", True),
    }

    if state.get("include_narration"):
        try:
            from tools.vibevoice import attach_narration_assets

            state = await report_workflow_stage(
                state,
                tool="blender_generate_animation",
                stage="attach_narration",
                message="Generating and attaching VibeVoice narration",
                details={"speaker": state.get("narration_speaker") or "Emma"},
            )
            result.update(
                await attach_narration_assets(
                    video_path=final_path,
                    narration_text=(state.get("narration_text") or state["description"]).strip(),
                    speaker=state.get("narration_speaker") or "Emma",
                    prefix="animations",
                    metadata={
                        "workflow_thread_id": state["workflow_thread_id"],
                        "tool": "blender_generate_animation",
                        "background_style": state["background_style"],
                        "composited": state.get("composite_over_scene", True),
                    },
                )
            )
        except Exception as exc:
            logger.warning(
                "manim_workflow.narration_failed thread_id=%s error=%s",
                state["workflow_thread_id"],
                exc,
            )
            result["narration_error"] = str(exc)

    next_state = {**state, "result": result, "error": ""}
    return await report_workflow_stage(
        next_state,
        tool="blender_generate_animation",
        stage="upload_complete",
        message="Animation output uploaded successfully",
        details={"video_url": video_url},
    )


async def _cleanup_node(state: ManimWorkflowState) -> ManimWorkflowState:
    for path in {
        state.get("manim_path", ""),
        state.get("blender_path", ""),
        state.get("final_path", ""),
    }:
        if not path:
            continue
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass
    return state


def build_manim_workflow_graph(checkpointer: Any) -> Any:
    graph = StateGraph(ManimWorkflowState)
    graph.add_node("render_manim", _render_manim_node)
    graph.add_node("composite", _composite_node)
    graph.add_node("upload", _upload_node)
    graph.add_node("cleanup", _cleanup_node)
    graph.set_entry_point("render_manim")
    graph.add_edge("render_manim", "composite")
    graph.add_edge("composite", "upload")
    graph.add_edge("upload", "cleanup")
    graph.add_edge("cleanup", END)
    return graph.compile(checkpointer=checkpointer)


async def run_manim_workflow(
    description: str,
    duration: float = 10.0,
    background_style: str = "dark",
    composite_over_scene: bool = True,
    include_narration: bool = False,
    narration_text: str = "",
    narration_speaker: str = "Emma",
    workflow_thread_id: str = "",
) -> dict:
    checkpointer = await get_checkpointer()
    graph = _GRAPHS.get(id(checkpointer))
    if graph is None:
        graph = build_manim_workflow_graph(checkpointer)
        _GRAPHS[id(checkpointer)] = graph

    thread_id = workflow_thread_id.strip() or f"manim-{uuid.uuid4().hex}"
    initial: ManimWorkflowState = {
        "description": description,
        "duration": duration,
        "background_style": background_style,
        "composite_over_scene": composite_over_scene,
        "include_narration": include_narration,
        "narration_text": narration_text,
        "narration_speaker": narration_speaker,
        "workflow_thread_id": thread_id,
        "output_tmp": "",
        "manim_path": "",
        "blender_path": "",
        "final_path": "",
        "result": {},
        "error": "",
        "progress_stage": "queued",
        "progress_message": "Animation workflow queued",
        "progress_state": "pending",
        "progress_details": {},
        "progress_updated_at": "",
        "progress_events": [],
    }
    final = await graph.ainvoke(initial, config=workflow_config(thread_id, "manim_workflow"))
    if final.get("error"):
        raise RuntimeError(final["error"])
    return final.get("result", {})
