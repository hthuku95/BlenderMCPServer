"""
Durable LangGraph workflow for Manim scene renders with LLM fallback.

Use this for specialty Manim tools such as charts, flowcharts, timelines,
and 3D math scenes. If the scene template file doesn't exist (templates have
been replaced by LLM code generation), it falls through to generate_and_run_manim.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from tools.workflow_runtime import ainvoke_with_checkpoint_fallback

logger = logging.getLogger(__name__)
_GRAPHS: dict[int, Any] = {}


class ManimSceneWorkflowState(TypedDict):
    workflow_thread_id: str
    scene_file: str
    scene_class: str
    scene_args: dict
    prefix: str
    duration: float
    include_narration: bool
    narration_text: str
    narration_speaker: str
    output_path: str
    metadata: dict
    result: dict
    error: str


async def _render_scene_node(state: ManimSceneWorkflowState) -> ManimSceneWorkflowState:
    if state.get("error"):
        return state

    output_path = state.get("output_path") or f"/tmp/manim_scene_{uuid.uuid4().hex}.mp4"
    scene_file = state["scene_file"]

    # If the scene template file was deleted (replaced by LLM code generation),
    # fall through to generate_and_run_manim
    if not os.path.exists(scene_file):
        from tools.manim_codegen import generate_and_run_manim

        # Build a natural language description from the scene args
        scene_args = state.get("scene_args", {}) or {}
        chart_type = scene_args.get("chart_type", "")
        title = scene_args.get("title", "")
        args_desc = ", ".join(f"{k}={v}" for k, v in scene_args.items() if k not in ("title",))
        description = f"{title}: Create a {chart_type} " if chart_type else ""
        description += args_desc

        logger.info(
            "manim_scene_workflow: template %s not found, using LLM generation. description=%s",
            scene_file,
            description[:200],
        )
        await generate_and_run_manim(
            description=description,
            duration=state["duration"],
            background="dark",
            output_path=output_path,
            transparent=False,
            quality="m",
        )
        return {**state, "output_path": output_path, "error": ""}

    from tools.manim_runner import run_manim_scene

    await run_manim_scene(
        scene_file=scene_file,
        scene_class=state["scene_class"],
        args=scene_args,
        quality="m",
        output_path=output_path,
        timeout=300,
    )
    return {**state, "output_path": output_path, "error": ""}


async def _upload_node(state: ManimSceneWorkflowState) -> ManimSceneWorkflowState:
    if state.get("error"):
        return state

    from tools.storage import upload_render

    output_path = state.get("output_path", "")
    if not output_path or not os.path.exists(output_path):
        return {**state, "error": f"Rendered output missing: {output_path}"}

    result = {
        "video_url": upload_render(output_path, prefix=state["prefix"]),
        "duration": state["duration"],
    }
    result.update(state.get("metadata", {}))

    if state.get("include_narration"):
        try:
            from tools.vibevoice import attach_narration_assets

            fallback_text = (
                state.get("narration_text")
                or state.get("metadata", {}).get("title")
                or state.get("scene_class", "Manim scene")
            )
            result.update(
                await attach_narration_assets(
                    video_path=output_path,
                    narration_text=str(fallback_text).strip(),
                    speaker=state.get("narration_speaker") or "Emma",
                    prefix=state["prefix"],
                    metadata={
                        "workflow_thread_id": state["workflow_thread_id"],
                        "tool": state["scene_class"],
                        **(state.get("metadata", {}) or {}),
                    },
                )
            )
        except Exception as exc:
            result["narration_error"] = str(exc)

    return {**state, "result": result, "error": ""}


async def _cleanup_node(state: ManimSceneWorkflowState) -> ManimSceneWorkflowState:
    output_path = state.get("output_path", "")
    if output_path:
        try:
            if os.path.exists(output_path):
                os.unlink(output_path)
        except OSError:
            pass
    return state


def build_manim_scene_workflow_graph(checkpointer: Any) -> Any:
    graph = StateGraph(ManimSceneWorkflowState)
    graph.add_node("render_scene", _render_scene_node)
    graph.add_node("upload", _upload_node)
    graph.add_node("cleanup", _cleanup_node)
    graph.set_entry_point("render_scene")
    graph.add_edge("render_scene", "upload")
    graph.add_edge("upload", "cleanup")
    graph.add_edge("cleanup", END)
    return graph.compile(checkpointer=checkpointer)


async def run_manim_scene_workflow(
    *,
    scene_file: str,
    scene_class: str,
    scene_args: dict,
    prefix: str,
    duration: float,
    metadata: dict | None = None,
    include_narration: bool = False,
    narration_text: str = "",
    narration_speaker: str = "Emma",
    workflow_thread_id: str = "",
    output_path: str = "",
) -> dict:
    thread_id = workflow_thread_id.strip() or f"manim-scene-{uuid.uuid4().hex}"
    initial: ManimSceneWorkflowState = {
        "workflow_thread_id": thread_id,
        "scene_file": scene_file,
        "scene_class": scene_class,
        "scene_args": scene_args,
        "prefix": prefix,
        "duration": duration,
        "include_narration": include_narration,
        "narration_text": narration_text,
        "narration_speaker": narration_speaker,
        "output_path": output_path,
        "metadata": metadata or {},
        "result": {},
        "error": "",
    }
    final = await ainvoke_with_checkpoint_fallback(
        graph_cache=_GRAPHS,
        graph_builder=build_manim_scene_workflow_graph,
        initial_state=initial,
        thread_id=thread_id,
        checkpoint_ns="manim_scene_workflow",
    )
    if final.get("error"):
        raise RuntimeError(final["error"])
    return final.get("result", {})
