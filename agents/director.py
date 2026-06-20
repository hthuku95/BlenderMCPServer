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

import json
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools.llm_client import get_chat_model, active_provider

# ---------------------------------------------------------------------------
# LangChain tool wrappers (calls the consolidated MCP tools)
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
    from server import blender_execute_bpy_script as _mcp_impl
    return await _mcp_impl(
        prompt=prompt,
        duration=duration,
        style=style,
        reference_image_url=reference_image_url,
        include_narration=include_narration,
        narration_text=narration_text,
        narration_speaker=narration_speaker,
    )


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
    from server import manim_execute_script as _mcp_impl
    return await _mcp_impl(
        description=description,
        duration=duration,
        background=background,
        transparent=transparent,
        quality=quality,
        include_narration=include_narration,
        narration_text=narration_text,
        narration_speaker=narration_speaker,
    )


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


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a professional video production director AI with access to 2 consolidated tools "
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

def agent_node(state: DirectorState) -> dict:
    provider = state.get("provider") or active_provider()
    llm = get_chat_model(
        temperature=0.7,
        max_tokens=4096,
        provider=provider,
    ).bind_tools(TOOLS)

    # Use proper SystemMessage so both Claude and Gemini receive it correctly
    messages = state["messages"]
    full_messages = [SystemMessage(content=_SYSTEM)] + list(messages)

    response = llm.invoke(full_messages)
    return {"messages": [response], "assets": state.get("assets", []), "provider": provider}


def tools_node(state: DirectorState) -> dict:
    """Run tool calls and harvest asset URLs from results."""
    node = ToolNode(TOOLS)
    result = node.invoke(state)

    assets = list(state.get("assets", []))
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                for url_key in ("video_url", "image_url"):
                    if url := data.get(url_key):
                        assets.append({"tool": msg.name, "url": url})
            except (json.JSONDecodeError, AttributeError):
                pass

    return {
        "messages": result.get("messages", []),
        "assets": assets,
        "provider": state.get("provider", ""),
    }


def should_continue(state: DirectorState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

def build_director_graph():
    g = StateGraph(DirectorState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_director(brief: str, provider: str | None = None) -> dict:
    """
    Run the director agent with a creative brief.

    Args:
        brief:    Natural language description of what to produce.
        provider: Override LLM_PROVIDER for this run.
                  "gemini" | "claude" | "auto" | None (use env default).

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
    }

    final = await graph.ainvoke(init)

    summary = ""
    for msg in reversed(final["messages"]):
        if isinstance(msg, AIMessage) and not (hasattr(msg, "tool_calls") and msg.tool_calls):
            summary = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    return {
        "assets": final["assets"],
        "summary": summary,
        "provider": final.get("provider", resolved),
    }
