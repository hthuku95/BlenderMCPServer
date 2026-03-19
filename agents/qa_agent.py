"""
QA Agent — Plan Phase 3.

Iterative quality refinement loop:
  1. Extract a preview frame from the rendered MP4 (at 50% timestamp)
  2. Compare preview frame to the reference image via Claude vision
  3. If match_score >= 0.70 → approved, return result
  4. If match_score < 0.70 and attempts < max_iterations → apply corrections
     and re-render by calling the Blender runner again with corrections dict
  5. After max_iterations → return best result

This agent wraps any Blender render that also has a reference_image supplied.
It can be used standalone or called from the Director after blender_generate_scene.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class QAState(TypedDict):
    # Inputs
    render_video_path: str
    reference_image_path: str      # local path (already downloaded)
    prompt_context: str
    blender_script_path: str
    blender_args: dict             # the args used for the current render
    max_iterations: int
    # Runtime
    iteration: int
    current_video_path: str
    best_video_path: str
    best_score: float
    approved: bool
    corrections: dict
    error: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_frame(video_path: str, timestamp_frac: float = 0.5) -> str:
    """Extract a single frame from a video as a JPEG."""
    # Get video duration via ffprobe
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True, text=True, timeout=30,
    )
    try:
        duration = float(probe.stdout.strip())
    except ValueError:
        duration = 5.0

    t = duration * timestamp_frac

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, prefix="qa_frame_") as f:
        frame_path = f.name

    subprocess.run(
        ["ffmpeg", "-ss", str(t), "-i", video_path, "-vframes", "1", "-y", frame_path],
        capture_output=True, timeout=60,
    )

    if not os.path.exists(frame_path):
        raise RuntimeError(f"ffmpeg failed to extract frame from {video_path}")

    return frame_path


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def _evaluate_render_node(state: QAState) -> QAState:
    """Extract frame and compare to reference."""
    from tools.vision_tools import compare_render_to_reference

    video = state.get("current_video_path") or state["render_video_path"]

    try:
        frame_path = _extract_frame(video)
    except Exception as e:
        return {**state, "error": f"Frame extraction failed: {e}"}

    try:
        comparison = compare_render_to_reference(
            render_path=frame_path,
            reference_path_or_url=state["reference_image_path"],
            prompt_context=state.get("prompt_context", ""),
        )
    except Exception as e:
        return {**state, "error": f"Vision comparison failed: {e}"}
    finally:
        try:
            os.unlink(frame_path)
        except OSError:
            pass

    score = float(comparison.get("match_score", 0.5))
    approved = bool(comparison.get("approved", score >= 0.70))

    # Track best result
    best_score = state.get("best_score", 0.0)
    best_video = state.get("best_video_path", video)
    if score > best_score:
        best_score = score
        best_video = video

    return {
        **state,
        "approved": approved,
        "best_score": best_score,
        "best_video_path": best_video,
        "corrections": comparison.get("corrections", {}),
        "current_video_path": video,
        "error": "",
    }


async def _re_render_node(state: QAState) -> QAState:
    """Apply corrections and re-render."""
    from tools.blender_runner import run_blender_script

    iteration = state.get("iteration", 0) + 1
    output_mp4 = f"/tmp/qa_iter{iteration}_{os.getpid()}.mp4"

    # Merge corrections into blender_args
    updated_args = dict(state.get("blender_args", {}))
    updated_args["corrections"] = state.get("corrections", {})
    updated_args["output_path"] = output_mp4

    try:
        result = await run_blender_script(
            script_path=state["blender_script_path"],
            args=updated_args,
            timeout=900,
        )
    except RuntimeError as e:
        return {**state, "iteration": iteration, "error": f"Re-render failed: {e}"}

    return {
        **state,
        "iteration": iteration,
        "current_video_path": result.get("output_path", output_mp4),
        "blender_args": updated_args,
        "error": "",
    }


def _should_continue(state: QAState) -> str:
    """Route: continue iterating or exit."""
    if state.get("error"):
        return "done"
    if state.get("approved"):
        return "done"
    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        return "done"
    return "re_render"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_qa_graph() -> StateGraph:
    g = StateGraph(QAState)
    g.add_node("evaluate_render", _evaluate_render_node)
    g.add_node("re_render", _re_render_node)

    g.set_entry_point("evaluate_render")
    g.add_conditional_edges(
        "evaluate_render",
        _should_continue,
        {"done": END, "re_render": "re_render"},
    )
    g.add_edge("re_render", "evaluate_render")

    return g.compile()


_GRAPH = None


async def run_qa_agent(
    render_video_path: str,
    reference_image_path: str,
    blender_script_path: str,
    blender_args: dict,
    prompt_context: str = "",
    max_iterations: int = 3,
) -> dict:
    """
    Run the QA refinement loop on a rendered video.

    Returns:
        {
          "approved": bool,
          "best_video_path": str,
          "best_score": float,
          "iterations": int,
          "error": str
        }
    """
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_qa_graph()

    initial: QAState = {
        "render_video_path": render_video_path,
        "reference_image_path": reference_image_path,
        "prompt_context": prompt_context,
        "blender_script_path": blender_script_path,
        "blender_args": blender_args,
        "max_iterations": max_iterations,
        "iteration": 0,
        "current_video_path": render_video_path,
        "best_video_path": render_video_path,
        "best_score": 0.0,
        "approved": False,
        "corrections": {},
        "error": "",
    }

    final = await _GRAPH.ainvoke(initial)
    return {
        "approved": final.get("approved", False),
        "best_video_path": final.get("best_video_path", render_video_path),
        "best_score": final.get("best_score", 0.0),
        "iterations": final.get("iteration", 0),
        "error": final.get("error", ""),
    }
