"""
LangGraph Director Agent — Phase 2

Takes a high-level creative brief and orchestrates the 6 BlenderMCPServer tools
to produce a list of video/image asset URLs ready for use in auto_generate_video.

Usage:
    import asyncio
    from agents.director import run_director

    result = asyncio.run(run_director(
        "Create an intro package for a tech finance YouTube channel: "
        "animated title card, a lower-third for the host, and a 3D scene."
    ))
    # result = {"assets": [{"tool": str, "url": str}, ...], "summary": str}
"""

import json
import os
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic  # type: ignore
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # type: ignore
from langchain_core.tools import tool  # type: ignore
from langgraph.graph import END, StateGraph  # type: ignore
from langgraph.graph.message import add_messages  # type: ignore
from langgraph.prebuilt import ToolNode  # type: ignore

from tools.render_tools import (
    impl_generate_data_viz,
    impl_generate_latex,
    impl_generate_lower_third,
    impl_generate_scene,
    impl_generate_thumbnail,
    impl_generate_title_card,
)

# ---------------------------------------------------------------------------
# LangChain tool wrappers (thin async wrappers over the shared impls)
# ---------------------------------------------------------------------------

@tool
async def blender_generate_scene(
    prompt: str,
    duration: float = 10.0,
    style: str = "cinematic",
) -> str:
    """Generate a procedural 3D Blender scene as an MP4 clip.
    Returns JSON: {"video_url": str, "duration": float, "resolution": str}"""
    return json.dumps(await impl_generate_scene(prompt, duration, style))


@tool
async def blender_generate_thumbnail(
    prompt: str,
    title_text: str = "",
    style: str = "youtube",
) -> str:
    """Generate a 3D rendered YouTube thumbnail image (1280×720 PNG).
    Returns JSON: {"image_url": str, "width": int, "height": int}"""
    return json.dumps(await impl_generate_thumbnail(prompt, title_text, style))


@tool
async def blender_generate_title_card(
    title: str,
    subtitle: str = "",
    duration: float = 5.0,
    style: str = "cinematic",
) -> str:
    """Generate an animated 3D title card as an MP4 clip.
    Returns JSON: {"video_url": str, "duration": float}"""
    return json.dumps(await impl_generate_title_card(title, subtitle, duration, style))


@tool
async def blender_generate_data_viz(
    data_json: str,
    chart_type: str = "bar",
    title: str = "",
    duration: float = 10.0,
) -> str:
    """Generate an animated 3D data visualisation clip from JSON data.
    data_json format: '[{"label":"A","value":42},...]'
    Returns JSON: {"video_url": str, "duration": float, "chart_type": str}"""
    return json.dumps(await impl_generate_data_viz(data_json, chart_type, title, duration))


@tool
async def blender_generate_lower_third(
    name_text: str,
    subtitle_text: str = "",
    style: str = "modern",
    duration: float = 5.0,
) -> str:
    """Generate an animated lower-third name plate clip (green-screen background for keying).
    Returns JSON: {"video_url": str, "duration": float, "keying": "green_screen"}"""
    return json.dumps(await impl_generate_lower_third(name_text, subtitle_text, style, duration))


@tool
async def blender_generate_latex(
    latex_expression: str,
    animation_type: str = "appear",
    duration: float = 8.0,
    background_style: str = "dark",
) -> str:
    r"""Generate a LaTeX/Manim math equation animation clip.
    animation_type: "appear" | "morph" | "step_by_step"
    Returns JSON: {"video_url": str, "duration": float, "latex_expression": str}"""
    return json.dumps(await impl_generate_latex(latex_expression, animation_type, duration, background_style))


TOOLS = [
    blender_generate_scene,
    blender_generate_thumbnail,
    blender_generate_title_card,
    blender_generate_data_viz,
    blender_generate_lower_third,
    blender_generate_latex,
]

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class DirectorState(TypedDict):
    messages: Annotated[list, add_messages]
    assets: list[dict]  # accumulated {"tool": str, "url": str} entries


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a professional video production director AI with access to 6 tools that generate "
    "3D Blender animations and Manim math animations. "
    "Given a creative brief, decide which tools to call and with what parameters to produce "
    "the best asset package. Call tools in a logical sequence — title cards before scenes, "
    "lower-thirds with the host's name when mentioned, thumbnails when the channel is mentioned. "
    "After all tools have finished, write a brief summary of what was produced."
)


def agent_node(state: DirectorState) -> dict:
    llm = ChatAnthropic(
        model="claude-opus-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=4096,
    ).bind_tools(TOOLS)

    messages = state["messages"]
    # Prepend system as a Human turn if first call
    if not any(
        isinstance(m, HumanMessage) and m.content.startswith("System:")
        for m in messages
    ):
        messages = [HumanMessage(content=f"System: {_SYSTEM}")] + messages

    response = llm.invoke(messages)
    return {"messages": [response], "assets": state.get("assets", [])}


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

    return {"messages": result.get("messages", []), "assets": assets}


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

async def run_director(brief: str) -> dict:
    """
    Run the director agent with a creative brief.

    Returns:
        {
            "assets": [{"tool": str, "url": str}, ...],
            "summary": str,
        }
    """
    graph = build_director_graph()
    init: DirectorState = {
        "messages": [HumanMessage(content=brief)],
        "assets": [],
    }

    final = await graph.ainvoke(init)

    # Last non-tool-call AI message is the summary
    summary = ""
    for msg in reversed(final["messages"]):
        if isinstance(msg, AIMessage) and not (hasattr(msg, "tool_calls") and msg.tool_calls):
            summary = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    return {"assets": final["assets"], "summary": summary}
