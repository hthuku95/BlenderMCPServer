"""
Durable LangGraph workflow for fixed-scene Manim renders.

Use this for specialty Manim tools that render a known scene file/class pair
such as charts, flowcharts, timelines, and 3D math scenes.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from tools.workflow_runtime import get_checkpointer, workflow_config

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

    from tools.manim_runner import run_manim_scene

    output_path = state.get("output_path") or f"/tmp/manim_scene_{uuid.uuid4().hex}.mp4"
    await run_manim_scene(
        scene_file=state["scene_file"],
        scene_class=state["scene_class"],
        args=state.get("scene_args", {}),
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
    checkpointer = await get_checkpointer()
    graph = _GRAPHS.get(id(checkpointer))
    if graph is None:
        graph = build_manim_scene_workflow_graph(checkpointer)
        _GRAPHS[id(checkpointer)] = graph

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
    final = await graph.ainvoke(initial, config=workflow_config(thread_id, "manim_scene_workflow"))
    if final.get("error"):
        raise RuntimeError(final["error"])
    return final.get("result", {})
