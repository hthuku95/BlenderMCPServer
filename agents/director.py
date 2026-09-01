"""
LangGraph Director Agent — Phase 2 (Consolidated Tools)

Takes a high-level creative brief and orchestrates 2 consolidated tools
(blender_execute_bpy_script + manim_execute_script) to produce a list of
video/image asset URLs ready for use in auto_generate_video.

Previously used 7 individual template-based tools. Now uses LLM-driven code
generation via the consolidated tools — covering 100% of Blender's bpy API
and 100% of Manim's API.

Provider: controlled by LLM_PROVIDER env var (see tools/llm_client.py).
  "auto"    — Gemini primary, Claude fallback  (default)
  "gemini"  — always Gemini gemini-2.0-flash
  "claude"  — always Claude claude-opus-4-6

Usage:
    import asyncio
    from agents.director import run_director

    result = asyncio.run(run_director(
        "Create an intro package for a tech finance YouTube channel: "
        "animated title card, a lower-third for the host, and a 3D scene."
    ))
    # result = {"assets": [{"tool": str, "url": str}, ...], "summary": str, "provider": str}
"""

import datetime
import json
import os
import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools.llm_client import get_chat_model, active_provider
from tools.progress_store import record_job_progress

# ---------------------------------------------------------------------------
# Global cancellation set — shared across graph nodes and server.py
# ---------------------------------------------------------------------------

_cancelled_jobs: set[str] = set()


def cancel_job(job_id: str) -> None:
    """Mark a job as cancelled. Graph nodes check this at each step."""
    _cancelled_jobs.add(job_id)


def is_job_cancelled(job_id: str) -> bool:
    """Check if a job has been cancelled."""
    return job_id in _cancelled_jobs

# ---------------------------------------------------------------------------
# LangChain tool wrappers (calls the consolidated codegen directly)
# ---------------------------------------------------------------------------

@tool
async def blender_execute_bpy_script(
    prompt: str,
    duration: float = 10.0,
    style: str = "cinematic",
    reference_image_url: str = "",
    include_narration: bool = False,
    narration_text: str = "",
    narration_speaker: str = "Emma",
) -> str:
    """
    Generate ANY 3D Blender scene, thumbnail, title card, lower-third,
    or UI device mockup from a natural language description.

    Uses LLM code generation to dynamically write and execute arbitrary
    bpy Python code — covers 100% of Blender's API: geometry nodes,
    physics, character rigging, custom shaders, particle systems,
    Grease Pencil, camera motion, and more.

    Tip: For thumbnails ask for a "1280x720 PNG thumbnail frame".
         For lower-thirds mention "green screen background".
         For device mockups mention the device type (iPhone, MacBook).

    Returns JSON: {"video_url": str, "duration": float, "resolution": str, "frames": int}
    """
    from tools.bpy_codegen import generate_and_run_bpy
    from tools.storage import upload_render

    output_path = f"/tmp/bpy_scene_{uuid.uuid4().hex}.mp4"
    result_path = await generate_and_run_bpy(
        prompt=prompt, duration=duration, style=style,
        output_path=output_path, reference_image_url=reference_image_url,
    )
    video_url = upload_render(result_path, prefix="scenes")
    try:
        os.unlink(result_path)
    except OSError:
        pass
    response = {
        "video_url": video_url, "duration": duration,
        "resolution": "1920x1080", "frames": int(duration * 60),
        "generation": "llm_dynamic_bpy",
    }
    if include_narration:
        try:
            from tools.vibevoice import attach_narration_assets
            fallback_text = narration_text or prompt
            response.update(
                await attach_narration_assets(
                    video_path=result_path, narration_text=fallback_text.strip(),
                    speaker=narration_speaker, prefix="scenes",
                    metadata={"tool": "blender_execute_bpy_script", "style": style},
                )
            )
        except Exception as exc:
            response["narration_error"] = str(exc)
    return json.dumps(response)


@tool
async def manim_execute_script(
    description: str,
    duration: float = 10.0,
    background: str = "dark",
    transparent: bool = False,
    quality: str = "m",
    include_narration: bool = False,
    narration_text: str = "",
    narration_speaker: str = "Emma",
) -> str:
    """
    Generate ANY Manim animation from a natural language description.

    Uses LLM code generation to dynamically write and execute arbitrary
    Manim Python code — covers 100% of Manim's API: animations, 3D scenes,
    graphs, LaTeX math, geometry proofs, network graphs, timelines, code
    syntax highlighting, vector fields, matrix transforms, and more.

    Tip: For math equations describe the expression in LaTeX-like terms.
         For charts mention the type (bar, line, pie, scatter).

    Returns JSON: {"video_url": str, "duration": float, "resolution": str, "frames": int}
    """
    from tools.manim_codegen import generate_and_run_manim
    from tools.storage import upload_render

    ext = ".mov" if transparent else ".mp4"
    output_path = f"/tmp/manim_scene_{uuid.uuid4().hex}{ext}"
    result_path = await generate_and_run_manim(
        description=description, duration=duration, background=background,
        output_path=output_path, transparent=transparent, quality=quality,
    )
    video_url = upload_render(result_path, prefix="scenes")
    try:
        os.unlink(result_path)
    except OSError:
        pass
    quality_map = {"l": "854x480", "m": "1280x720", "h": "1920x1080"}
    res = quality_map.get(quality, "1920x1080")
    response = {
        "video_url": video_url, "duration": duration,
        "resolution": res, "frames": int(duration * 30),
        "generation": "llm_dynamic_manim",
    }
    if include_narration:
        try:
            from tools.vibevoice import attach_narration_assets
            fallback_text = narration_text or description
            response.update(
                await attach_narration_assets(
                    video_path=result_path, narration_text=fallback_text.strip(),
                    speaker=narration_speaker, prefix="scenes",
                    metadata={"tool": "manim_execute_script", "background": background},
                )
            )
        except Exception as exc:
            response["narration_error"] = str(exc)
    return json.dumps(response)


# REMOVED (Phase 2, §40.3 gap #1): review_rendered_video imported
# tools/video_review.py which does not exist — it crashed at call time.
# The review loop nodes (review_node / review_decision) remain wired but
# dormant; a REAL verifier (native video understanding, §5) re-lights them
# in Phase 3.

TOOLS = [
    blender_execute_bpy_script,
    manim_execute_script,
]

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class DirectorState(TypedDict):
    messages: Annotated[list, add_messages]
    assets: list[dict]      # accumulated {"tool": str, "url": str} entries
    provider: str           # which LLM is driving this run
    job_id: str             # job_id for progress tracking
    retry_count: int        # re-edit attempts (hard loop, max max_retries)
    max_retries: int        # max re-render attempts when review score < 0.6
    cancelled: bool         # set to True to stop execution early
    needs_rerender: bool    # set by review_node when score < 0.6, consumed by review_decision


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------

async def _record(state: DirectorState, stage: str, message: str, details: dict | None = None):
    """Record a progress event if job_id is set."""
    jid = state.get("job_id", "")
    if not jid:
        return
    try:
        await record_job_progress(
            job_id=jid,
            workflow_thread_id=jid,
            tool="run_director",
            state="running" if stage != "completed" else "completed",
            stage=stage,
            message=message,
            details=details or {},
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a professional video production director AI with access to 2 tools "
    "that cover ALL 3D Blender rendering and ALL Manim animations:\n\n"
    "1. blender_execute_bpy_script — for EVERYTHING 3D/Blender: scenes, thumbnails, title cards, "
    "lower-thirds, UI device mockups, logo reveals, particle effects, abstract backgrounds, "
    "countdowns, camera fly-throughs, toon scenes, grease pencil, geometry scattering, etc.\n"
    "2. manim_execute_script — for EVERYTHING Manim: math equations, data charts (bar/line/pie/scatter), "
    "flowcharts, 3D math, code animations, timelines, network graphs, text animations, "
    "vector fields, matrix transforms, polar graphs, geometry proofs, etc.\n\n"
    "Given a creative brief, decide which tools to call and with what parameters to produce "
    "the best asset package. Call tools in a logical sequence — title cards before scenes, "
    "lower-thirds with the host's name when mentioned, thumbnails when the channel is mentioned, "
    "device mockups when showcasing an app or website. "
    "After all tools have finished, write a brief summary of what was produced."
)

# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

async def agent_node(state: DirectorState) -> dict:
    # Check cancellation before doing any work
    if state.get("cancelled", False) or is_job_cancelled(state.get("job_id", "")):
        await _record(state, "cancelled", "Director cancelled by user")
        return {"messages": [AIMessage(content="Director execution cancelled by user.")],
                "assets": state.get("assets", [])}

    provider = state.get("provider") or active_provider()
    await _record(state, "agent_planning", "Analyzing brief and planning scenes...")

    llm = get_chat_model(
        temperature=0.7,
        max_tokens=4096,
        provider=provider,
    ).bind_tools(TOOLS)

    messages = state["messages"]
    full_messages = [SystemMessage(content=_SYSTEM)] + list(messages)

    response = await llm.ainvoke(full_messages)
    tool_count = len(response.tool_calls) if hasattr(response, "tool_calls") and response.tool_calls else 0
    await _record(state, "agent_planned", f"Planned {tool_count} tool calls", {
        "tool_calls": [t.name for t in response.tool_calls] if hasattr(response, "tool_calls") and response.tool_calls else [],
    })
    return {"messages": [response], "assets": state.get("assets", []), "provider": provider}


async def tools_node(state: DirectorState) -> dict:
    """Run tool calls and harvest asset URLs from results."""
    # Check cancellation before executing tools
    if state.get("cancelled", False) or is_job_cancelled(state.get("job_id", "")):
        await _record(state, "cancelled", "Director cancelled by user")
        return {"messages": [AIMessage(content="Director execution cancelled by user.")],
                "assets": state.get("assets", [])}

    node = ToolNode(TOOLS)

    tool_names = []
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        for tc in last.tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else tc.name
            tool_names.append(name)
    tool_label = ", ".join(tool_names) if tool_names else "tools"
    await _record(state, "rendering", f"Rendering: {tool_label}...")

    result = await node.ainvoke(state)

    assets = list(state.get("assets", []))
    asset_urls = []
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                for url_key in ("video_url", "image_url"):
                    if url := data.get(url_key):
                        assets.append({"tool": msg.name, "url": url})
                        asset_urls.append(url)
            except (json.JSONDecodeError, AttributeError):
                pass

    await _record(state, "rendered", f"Completed {tool_label}", {
        "assets_created": len(asset_urls),
        "tool": tool_label,
    })

    return {
        "messages": result.get("messages", []),
        "assets": assets,
        "provider": state.get("provider", ""),
    }


async def review_node(state: DirectorState) -> dict:
    """
    Hard re-edit loop (DORMANT in Phase 2): checks results of a review tool
    and forces a re-render if quality_score or brief_match_score < 0.6.
    Up to max_retries (default 3) attempts, then give up and proceed.
    Inert until a real verifier tool is re-added in Phase 3 — the last
    message is never a review ToolMessage while the tool is removed.
    """
    result: dict = {"needs_rerender": False}

    # Only act if the last message is a review tool result
    last = state["messages"][-1]
    name = last.name if hasattr(last, "name") else ""
    if not (isinstance(last, ToolMessage) and name == "review_rendered_video"):
        return result

    try:
        data = json.loads(last.content)
    except (json.JSONDecodeError, AttributeError):
        return result

    quality = data.get("quality_score", 1.0)
    brief_match = data.get("brief_match_score", 1.0)
    score = min(quality, brief_match)

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if score >= 0.6:
        return result

    if retry_count >= max_retries:
        await _record(state, "max_retries_reached",
                      f"Max re-edit attempts ({max_retries}) reached. Score: {score:.2f}. Proceeding with current output.",
                      {"score": score, "retry_count": retry_count})
        return result

    improvements = data.get("suggested_improvements", [])
    improvement_text = "\n- ".join(improvements) if improvements else "General quality improvements"

    await _record(state, "re-edit",
                  f"Review score {score:.2f} below 0.6. Re-rendering (attempt {retry_count + 1}/{max_retries})...",
                  {"score": score, "retry_count": retry_count + 1, "improvements": improvements})

    feedback = HumanMessage(
        content=f"RE-EDIT REQUIRED: The previous render scored {score:.2f}/1.0 (below 0.6 threshold). "
                f"This is attempt {retry_count + 1} of {max_retries}. "
                f"Re-render with these specific improvements:\n- {improvement_text}\n\n"
                f"Focus on fixing the identified issues. Do NOT repeat the same mistakes."
    )

    result["messages"] = [feedback]
    result["retry_count"] = retry_count + 1
    result["needs_rerender"] = True
    return result


def should_continue(state: DirectorState) -> str:
    if state.get("cancelled", False) or is_job_cancelled(state.get("job_id", "")):
        return END
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def review_decision(state: DirectorState) -> str:
    """After review, go back to agent for re-render or finish."""
    if state.get("cancelled", False) or is_job_cancelled(state.get("job_id", "")):
        return END
    if state.get("needs_rerender", False):
        return "agent"
    return END


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

def build_director_graph():
    g = StateGraph(DirectorState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("review", review_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "review")
    g.add_conditional_edges("review", review_decision, {"agent": "agent", END: END})
    return g.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_director(brief: str, provider: str | None = None, job_id: str = "") -> dict:
    """
    Run the director agent with a creative brief.

    Args:
        brief:    Natural language description of what to produce.
        provider: Override LLM_PROVIDER for this run.
                  "gemini" | "claude" | "auto" | None (use env default).
        job_id:   Optional job_id for progress tracking. If empty, events
                  are not recorded.

    Returns:
        {
            "assets":   [{"tool": str, "url": str}, ...],
            "summary":  str,
            "provider": str,   # which LLM was used
        }
    """
    resolved = provider or active_provider()
    graph = build_director_graph()

    init: DirectorState = {
        "messages": [HumanMessage(content=brief)],
        "assets": [],
        "provider": resolved,
        "job_id": job_id,
        "retry_count": 0,
        "max_retries": 3,
        "cancelled": False,
        "needs_rerender": False,
    }

    if job_id:
        try:
            await record_job_progress(
                job_id=job_id,
                workflow_thread_id=job_id,
                tool="run_director",
                state="running",
                stage="starting",
                message="Director agent started",
                details={"brief": brief[:200], "provider": resolved},
            )
        except Exception:
            pass

    final = await graph.ainvoke(init)

    summary = ""
    for msg in reversed(final["messages"]):
        if isinstance(msg, AIMessage) and not (hasattr(msg, "tool_calls") and msg.tool_calls):
            summary = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    result = {
        "assets": final["assets"],
        "summary": summary,
        "provider": final.get("provider", resolved),
    }

    # Clean up cancellation flag
    _cancelled_jobs.discard(job_id)

    if job_id:
        try:
            await record_job_progress(
                job_id=job_id,
                workflow_thread_id=job_id,
                tool="run_director",
                state="completed",
                stage="completed",
                message="Director agent completed",
                details={"asset_count": len(final["assets"])},
                result=result,
                finished_at=datetime.datetime.now(datetime.timezone.utc),
            )
        except Exception:
            pass

    return result
