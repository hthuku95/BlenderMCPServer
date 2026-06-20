"""
QA Agent — VIGA refinement loop with dense frame sampling + full-video temporal QA.

Iterative quality refinement:
  1. Extract frames at 0.5s intervals (short video) or 1fps (long video), up to 30 frames
  2. Compare each frame to the reference image via the 4-model chain (Claude/Gemini/Ollama/NVIDIA NIM)
  3. Run a full-video temporal quality review via the 4-model chain
  4. If match_score >= 0.70 + temporal_pass → approved
  5. If not and attempts < max_iterations → apply corrections and re-render
  6. After max_iterations → return best result
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import tempfile
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from tools.workflow_runtime import ainvoke_with_checkpoint_fallback


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class QAState(TypedDict):
    render_video_path: str
    reference_image_path: str
    prompt_context: str
    blender_script_path: str
    blender_args: dict
    max_iterations: int
    iteration: int
    current_video_path: str
    best_video_path: str
    best_score: float
    approved: bool
    corrections: dict
    error: str
    temporal_notes: str


# ---------------------------------------------------------------------------
# Frame extraction — dense sampling
# ---------------------------------------------------------------------------

def _get_video_duration(video_path: str) -> float:
    """Get video duration via ffprobe."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return float(probe.stdout.strip())
    except (ValueError, TypeError):
        return 5.0


def _extract_dense_frames(video_path: str) -> list[str]:
    """Extract frames at dense intervals: every 0.5s for short video, 1fps for long, max 30."""
    duration = _get_video_duration(video_path)
    if duration <= 0:
        raise RuntimeError(f"Cannot determine duration for {video_path}")

    # Dense interval: every 0.5s for <30s video, every 1s otherwise, max 30 frames
    if duration < 15:
        interval = 0.5
        max_frames = int(duration / interval)
    else:
        interval = max(1.0, duration / 30.0)
        max_frames = min(30, int(duration / interval))

    if max_frames < 3:
        max_frames = 3

    frame_paths: list[str] = []
    timestamps = [i * interval for i in range(max_frames)]
    # Ensure last frame is near the end
    if timestamps and timestamps[-1] < duration - 1:
        timestamps.append(duration - 0.5)

    for i, ts in enumerate(timestamps):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, prefix=f"qa_dense_{i}_") as f:
            frame_path = f.name
        subprocess.run(
            ["ffmpeg", "-ss", str(ts), "-i", video_path, "-vframes", "1", "-q:v", "2", "-y", frame_path],
            capture_output=True, timeout=60,
        )
        if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
            frame_paths.append(frame_path)
        else:
            try:
                os.unlink(frame_path)
            except OSError:
                pass

    if not frame_paths:
        raise RuntimeError(f"ffmpeg failed to extract any frames from {video_path}")

    logger.info(
        "qa_agent: extracted %d dense frames from %.1fs video (interval=%.1fs)",
        len(frame_paths), duration, interval,
    )
    return frame_paths


def _merge_corrections(comparisons: list[dict]) -> dict:
    """Merge non-empty correction hints across all frame comparisons."""
    merged: dict[str, str] = {}
    correction_keys = (
        "lighting_correction",
        "color_correction",
        "composition_correction",
        "object_correction",
    )
    for key in correction_keys:
        values: list[str] = []
        for comparison in comparisons:
            corrections = comparison.get("corrections", {}) or {}
            value = corrections.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
        if values:
            merged[key] = " | ".join(dict.fromkeys(values))
    return merged


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def _evaluate_render_node(state: QAState) -> QAState:
    """Extract dense frames, compare each to reference, and run full-video temporal QA."""
    from tools.vision_tools import compare_render_to_reference, review_render_against_prompt

    video = state.get("current_video_path") or state["render_video_path"]
    frame_paths: list[str] = []

    # Step 1: Extract dense frames
    try:
        frame_paths = _extract_dense_frames(video)
    except Exception as e:
        return {**state, "error": f"Dense frame extraction failed: {e}"}

    # Step 2: Compare each frame to the reference image (uses 4-model chain internally)
    comparisons: list[dict] = []
    try:
        for frame_path in frame_paths:
            comparisons.append(compare_render_to_reference(
                render_path=frame_path,
                reference_path_or_url=state["reference_image_path"],
                prompt_context=state.get("prompt_context", ""),
            ))
    except Exception as e:
        return {**state, "error": f"Vision comparison failed: {e}"}
    finally:
        for frame_path in frame_paths:
            try:
                os.unlink(frame_path)
            except OSError:
                pass

    if not comparisons:
        return {**state, "error": "Vision comparison returned no frame analyses"}

    # Step 3: Full-video temporal quality review (uses 4-model chain with native video)
    temporal_score = 0.0
    temporal_notes = ""
    try:
        temporal_review = await review_render_against_prompt(
            render_video_path=video,
            prompt_context=state.get("prompt_context", ""),
        )
        if isinstance(temporal_review, dict):
            temporal_score = float(temporal_review.get("score", temporal_score))
            temporal_notes = temporal_review.get("feedback", "")
    except Exception as e:
        logger.warning("qa_agent: temporal review failed (non-blocking): %s", e)
        temporal_score = 0.5

    # Step 4: Calculate combined score
    frame_scores = [float(c.get("match_score", 0.5)) for c in comparisons]
    avg_frame_score = sum(frame_scores) / len(frame_scores)
    min_frame_score = min(frame_scores)

    # Combined score: 70% frame-match + 30% temporal quality
    combined_score = avg_frame_score * 0.7 + temporal_score * 0.3
    approved = combined_score >= 0.70 and min_frame_score >= 0.55

    merged_corrections = _merge_corrections(comparisons)

    notes_parts = [
        c.get("notes", "").strip()
        for c in comparisons
        if isinstance(c.get("notes"), str) and c["notes"].strip()
    ]
    if temporal_notes:
        notes_parts.append(f"[Temporal] {temporal_notes}")
    notes = " | ".join(notes_parts)

    logger.info(
        "qa_agent.evaluate iter=%s avg_frame=%.3f min_frame=%.3f temporal=%.3f combined=%.3f approved=%s",
        state.get("iteration", 0),
        avg_frame_score, min_frame_score, temporal_score, combined_score, approved,
    )

    best_score = state.get("best_score", 0.0)
    best_video = state.get("best_video_path", video)
    if combined_score > best_score:
        best_score = combined_score
        best_video = video

    return {
        **state,
        "approved": approved,
        "best_score": best_score,
        "best_video_path": best_video,
        "corrections": merged_corrections,
        "current_video_path": video,
        "error": "",
        "temporal_notes": temporal_notes,
        "prompt_context": (
            f"{state.get('prompt_context', '')}\nQA: {notes}".strip()
            if notes else state.get("prompt_context", "")
        ),
    }


async def _re_render_node(state: QAState) -> QAState:
    """Apply corrections and re-render."""
    from tools.blender_runner import run_blender_script

    iteration = state.get("iteration", 0) + 1
    output_mp4 = f"/tmp/qa_iter{iteration}_{os.getpid()}.mp4"

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

    logger.info(
        "qa_agent.rerender_complete iteration=%s output=%s correction_keys=%s",
        iteration,
        result.get("output_path", output_mp4),
        sorted((state.get("corrections") or {}).keys()),
    )

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

def build_qa_graph(checkpointer: Any) -> StateGraph:
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

    return g.compile(checkpointer=checkpointer)


_GRAPHS: dict[int, Any] = {}


async def run_qa_agent(
    render_video_path: str,
    reference_image_path: str,
    blender_script_path: str,
    blender_args: dict,
    prompt_context: str = "",
    max_iterations: int = 3,
    thread_id: str = "",
) -> dict:
    """
    Run the QA refinement loop with dense frame sampling + full-video temporal QA.

    Returns:
        {"approved": bool, "best_video_path": str, "best_score": float,
         "iterations": int, "error": str, "temporal_notes": str}
    """
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
        "temporal_notes": "",
    }

    final = await ainvoke_with_checkpoint_fallback(
        graph_cache=_GRAPHS,
        graph_builder=build_qa_graph,
        initial_state=initial,
        thread_id=thread_id or f"qa-{os.getpid()}",
        checkpoint_ns="qa_agent",
    )
    return {
        "approved": final.get("approved", False),
        "best_video_path": final.get("best_video_path", render_video_path),
        "best_score": final.get("best_score", 0.0),
        "iterations": final.get("iteration", 0),
        "error": final.get("error", ""),
        "temporal_notes": final.get("temporal_notes", ""),
    }
