"""
LaTeX Agent — LangGraph sub-agent for Plan Phase 2.

Classifies a math render request into Option A or Option B and executes the
appropriate pipeline.

  Option A (SVG → Blender 3D):
    latex_to_svg() → blender_runner(latex_3d_object.py) → upload R2 → return url

  Option B (Manim transparent → Blender scene → MoviePy composite):
    manim_runner(latex_transparent.py) → blender_runner(base_scene.py) →
    compositor.composite_manim_over_blender() → upload R2 → return url

Decision rule:
  Use Option B if animation_type is "morph" or "step_by_step".
  Use Option A for "appear" (single equation, no animation complexity).
  The Director can force either option via an optional `option` parameter.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from tools.workflow_runtime import get_checkpointer, workflow_config
from tools.progress_store import report_workflow_stage

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class LatexState(TypedDict):
    latex_expression: str
    animation_type: str          # "appear" | "morph" | "step_by_step" | "custom"
    duration: float
    background_style: str        # "dark" | "light" | "transparent"
    prompt: str                  # Optional natural-language animation description
    option: str                  # "A" | "B" | "auto"
    # filled by pipeline
    chosen_option: str           # "A" or "B"
    svg_path: str
    eq_video_path: str           # Option B only
    scene_video_path: str        # Option B only
    output_path: str
    workflow_thread_id: str
    error: str
    progress_stage: str
    progress_message: str
    progress_state: str
    progress_details: dict
    progress_updated_at: str
    progress_events: list[dict]


# ---------------------------------------------------------------------------
# Helper imports
# ---------------------------------------------------------------------------

def _tools():
    """Lazy import to keep startup fast."""
    from tools.latex_compiler import latex_to_svg
    from tools.manim_runner import run_manim_scene
    from tools.blender_runner import run_blender_script
    from tools.compositor import composite_manim_over_blender
    return latex_to_svg, run_manim_scene, run_blender_script, composite_manim_over_blender


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def _classify_node(state: LatexState) -> LatexState:
    """Decide Option A vs B based on animation_type or forced option."""
    state = await report_workflow_stage(
        state,
        tool="blender_generate_latex",
        stage="classify",
        message="Selecting the best LaTeX animation pipeline",
        details={
            "animation_type": state.get("animation_type", "appear"),
            "forced_option": state.get("option", "auto"),
        },
    )
    if state.get("option", "auto") in ("A", "B"):
        chosen = state["option"]
    elif state.get("animation_type") in ("morph", "step_by_step"):
        chosen = "B"
    else:
        chosen = "A"
    next_state = {**state, "chosen_option": chosen, "error": ""}
    return await report_workflow_stage(
        next_state,
        tool="blender_generate_latex",
        stage="pipeline_selected",
        message=f"Selected LaTeX pipeline option {chosen}",
        details={"chosen_option": chosen},
    )


async def _option_a_node(state: LatexState) -> LatexState:
    """
    Option A: compile LaTeX → SVG → Blender 3D object → MP4.
    """
    latex_to_svg, _, run_blender_script, _ = _tools()
    state = await report_workflow_stage(
        state,
        tool="blender_generate_latex",
        stage="compile_svg",
        message="Compiling LaTeX expression to SVG geometry",
        details={"chosen_option": "A"},
    )

    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False, prefix="latex_a_") as f:
        svg_tmp = f.name

    output_mp4 = state.get("output_path") or f"/tmp/latex_a_{os.getpid()}.mp4"

    try:
        svg_path = await latex_to_svg(state["latex_expression"], svg_tmp)
    except RuntimeError as e:
        return {**state, "error": f"latex_to_svg failed: {e}"}

    script = str(
        Path(__file__).parent.parent / "blender_scripts" / "latex_3d_object.py"
    )
    state = await report_workflow_stage(
        state,
        tool="blender_generate_latex",
        stage="render_blender_latex",
        message="Rendering 3D Blender animation from SVG geometry",
        details={"background_style": state.get("background_style", "dark")},
    )
    try:
        result = await run_blender_script(
            script_path=script,
            args={
                "svg_path": svg_path,
                "output_path": output_mp4,
                "duration": state.get("duration", 8.0),
                "fps": 60,
                "background_style": state.get("background_style", "dark"),
                "extrude_depth": 0.05,
            },
            timeout=600,
        )
    except RuntimeError as e:
        return {**state, "error": f"Blender 3D object render failed: {e}"}

    next_state = {
        **state,
        "svg_path": svg_path,
        "output_path": result.get("output_path", output_mp4),
        "error": "",
    }
    return await report_workflow_stage(
        next_state,
        tool="blender_generate_latex",
        stage="latex_render_complete",
        message="LaTeX 3D render completed",
        details={"chosen_option": "A"},
    )


async def _option_b_node(state: LatexState) -> LatexState:
    """
    Option B: Manim transparent clip + Blender scene → MoviePy composite.

    Strategy (in order):
    1. If a custom `prompt` is provided, or `animation_type == "custom"`, use the
       LLM code-generation pipeline (manim_codegen) for full creative flexibility.
    2. Otherwise attempt LLM code generation with a description derived from the
       latex_expression and animation_type (richer than the 3-branch static script).
    3. If LLM generation fails after MAX_RETRIES, fall back to the hardcoded
       latex_transparent.py template — always a working safe result.
    """
    _, run_manim_scene, run_blender_script, composite_manim_over_blender = _tools()
    state = await report_workflow_stage(
        state,
        tool="blender_generate_latex",
        stage="generate_manim_latex",
        message="Generating transparent Manim equation animation",
        details={"chosen_option": "B", "animation_type": state.get("animation_type", "appear")},
    )

    duration = float(state.get("duration", 8.0))
    output_mp4 = state.get("output_path") or f"/tmp/latex_b_{os.getpid()}.mp4"

    # ── Build animation description for LLM ──────────────────────────────────
    custom_prompt = state.get("prompt", "").strip()
    anim_type = state.get("animation_type", "appear")
    expr = state["latex_expression"]

    if custom_prompt:
        description = custom_prompt
    elif anim_type == "morph":
        description = (
            f"Animate the LaTeX equation: {expr}\n"
            "Show the left-hand side first, then transform it into the full equation "
            "using TransformMatchingTex. Use colour to distinguish different terms. "
            f"Total duration: {duration:.1f}s."
        )
    elif anim_type == "step_by_step":
        description = (
            f"Animate the LaTeX equation: {expr}\n"
            "Reveal each term one at a time from left to right, using FadeIn with "
            "a slight upward shift. Colour each distinct term differently. "
            "After all terms are visible, regroup them to the centre and hold. "
            f"Total duration: {duration:.1f}s."
        )
    else:
        description = (
            f"Animate the LaTeX equation: {expr}\n"
            "Write the equation onto the screen using the Write animation. "
            "Then use Indicate to highlight the most important part. "
            f"Total duration: {duration:.1f}s. Background: {state.get('background_style','dark')}."
        )

    # ── Step 1: LLM-generated Manim (transparent background for compositing) ─
    eq_video = f"/tmp/latex_eq_{os.getpid()}.mov"
    try:
        from tools.manim_codegen import generate_and_run_manim
        eq_video = await generate_and_run_manim(
            description=description,
            duration=duration,
            background=state.get("background_style", "dark"),
            output_path=eq_video,
            transparent=True,
            quality="m",
        )
    except RuntimeError as llm_err:
        # ── Fallback: hardcoded latex_transparent.py ─────────────────────────
        import logging
        logging.getLogger(__name__).warning(
            "LLM Manim generation failed, falling back to latex_transparent.py: %s",
            str(llm_err)[:300],
        )
        manim_scene = str(
            Path(__file__).parent.parent / "manim_scripts" / "latex_transparent.py"
        )
        try:
            eq_video = await run_manim_scene(
                scene_file=manim_scene,
                scene_class="LatexTransparent",
                args={
                    "latex_expression": expr,
                    "animation_type": anim_type if anim_type in ("appear", "morph", "step_by_step") else "appear",
                    "duration": duration,
                },
                quality="m",
                output_path=eq_video,
                transparent=True,
            )
        except RuntimeError as e:
            return {**state, "error": f"Manim transparent render failed: {e}"}

    # Step 2 — Blender background scene
    state = await report_workflow_stage(
        state,
        tool="blender_generate_latex",
        stage="render_background_scene",
        message="Rendering Blender background scene for equation composite",
        details={},
    )
    scene_script = str(
        Path(__file__).parent.parent / "blender_scripts" / "base_scene.py"
    )
    scene_video = f"/tmp/latex_scene_{os.getpid()}.mp4"
    try:
        result = await run_blender_script(
            script_path=scene_script,
            args={
                "prompt": "clean mathematical background, dark gradient, subtle particles",
                "duration": duration,
                "style": "calm",
                "output_path": scene_video,
            },
            timeout=600,
        )
        scene_video = result.get("output_path", scene_video)
    except RuntimeError as e:
        return {**state, "error": f"Blender scene render failed: {e}"}

    # Step 3 — composite (run in executor to avoid blocking asyncio event loop)
    state = await report_workflow_stage(
        state,
        tool="blender_generate_latex",
        stage="composite_equation",
        message="Compositing Manim equation over Blender background",
        details={},
    )
    try:
        loop = asyncio.get_event_loop()
        final = await loop.run_in_executor(
            None,
            lambda: composite_manim_over_blender(
                blender_video_path=scene_video,
                equation_video_path=eq_video,
                output_path=output_mp4,
                eq_x_position=0.5,
                eq_y_position=0.5,
                eq_scale=1.0,
                fps=60,
            )
        )
    except Exception as e:
        return {**state, "error": f"Compositing failed: {e}"}

    next_state = {
        **state,
        "eq_video_path": eq_video,
        "scene_video_path": scene_video,
        "output_path": final,
        "error": "",
    }
    return await report_workflow_stage(
        next_state,
        tool="blender_generate_latex",
        stage="latex_render_complete",
        message="LaTeX composite render completed",
        details={"chosen_option": "B"},
    )


def _route(state: LatexState) -> Literal["option_a", "option_b"]:
    # Option A (SVG→Blender 3D) disabled — dvisvgm-generated paths produce no
    # Curve objects in Blender 3.4's SVG importer.  Always use Option B (Manim).
    return "option_b"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_latex_graph(checkpointer) -> StateGraph:
    g = StateGraph(LatexState)
    g.add_node("classify", _classify_node)
    g.add_node("option_a", _option_a_node)
    g.add_node("option_b", _option_b_node)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify", _route, {"option_a": "option_a", "option_b": "option_b"})
    g.add_edge("option_a", END)
    g.add_edge("option_b", END)

    return g.compile(checkpointer=checkpointer)


_GRAPHS = {}


async def run_latex_agent(
    latex_expression: str,
    animation_type: str = "appear",
    duration: float = 8.0,
    background_style: str = "dark",
    prompt: str = "",
    output_path: str | None = None,
    option: str = "auto",
    workflow_thread_id: str = "",
) -> dict:
    """
    Execute the LaTeX rendering pipeline.

    Returns:
        {
          "output_path": str,       # local MP4 path
          "chosen_option": "A"|"B",
          "error": str              # empty on success
        }
    """
    checkpointer = await get_checkpointer()
    graph = _GRAPHS.get(id(checkpointer))
    if graph is None:
        graph = build_latex_graph(checkpointer)
        _GRAPHS[id(checkpointer)] = graph

    thread_id = workflow_thread_id.strip() or f"latex-{uuid.uuid4().hex}"

    initial: LatexState = {
        "latex_expression": latex_expression,
        "animation_type": animation_type,
        "duration": duration,
        "background_style": background_style,
        "prompt": prompt,
        "option": option,
        "chosen_option": "",
        "svg_path": "",
        "eq_video_path": "",
        "scene_video_path": "",
        "output_path": output_path or "",
        "workflow_thread_id": thread_id,
        "error": "",
        "progress_stage": "queued",
        "progress_message": "LaTeX workflow queued",
        "progress_state": "pending",
        "progress_details": {},
        "progress_updated_at": "",
        "progress_events": [],
    }

    final = await graph.ainvoke(initial, config=workflow_config(thread_id, "latex_workflow"))
    return {
        "output_path": final.get("output_path", ""),
        "chosen_option": final.get("chosen_option", ""),
        "error": final.get("error", ""),
    }
