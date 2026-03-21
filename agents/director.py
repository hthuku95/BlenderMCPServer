"""
LangGraph Director Agent — Phase 2

Takes a high-level creative brief and orchestrates the 6 BlenderMCPServer tools
to produce a list of video/image asset URLs ready for use in auto_generate_video.

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
import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools.render_tools import (
    impl_generate_data_viz,
    impl_generate_latex,
    impl_generate_lower_third,
    impl_generate_scene,
    impl_generate_thumbnail,
    impl_generate_title_card,
    impl_generate_ui_mockup,
)
from tools.llm_client import get_chat_model, active_provider

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


@tool
async def blender_generate_ui_mockup(
    device: str = "iphone",
    animation: str = "reveal",
    duration: float = 6.0,
    screenshot_url: str = "",
    screenshot_spec: str = "",
    background_color: str = "",
    accent_color: str = "",
) -> str:
    """Render a screenshot/image inside a 3D device frame (iPhone, MacBook, browser, iPad).

    Args:
        device: "iphone" | "macbook" | "browser" | "ipad"
        animation: "static" (PNG) | "reveal" (fade-in) | "scroll" (vertical scroll) | "tilt" (product reveal)
        duration: Clip length in seconds (ignored for static)
        screenshot_url: URL of the screenshot image to place inside the device screen
        screenshot_spec: JSON string of a design spec to auto-generate a screenshot
                         (if screenshot_url is empty). Schema: {"type":"browser"|"app",
                         "url":str, "title":str, "body":str, "bg_color":str, "accent_color":str}
        background_color: Optional JSON RGB array e.g. "[0.05, 0.05, 0.08]"
        accent_color: Optional JSON RGB array e.g. "[0.3, 0.5, 1.0]"

    Returns JSON: {"video_url": str, "device": str, "animation": str, "duration": float}
               or {"image_url": str, "device": str, "animation": "static"}
    """
    import json as _json

    spec = _json.loads(screenshot_spec) if screenshot_spec else None
    bg   = _json.loads(background_color) if background_color else None
    acc  = _json.loads(accent_color)     if accent_color else None

    return _json.dumps(await impl_generate_ui_mockup(
        device=device,
        animation=animation,
        duration=duration,
        screenshot_url=screenshot_url,
        screenshot_spec=spec,
        background_color=bg,
        accent_color=acc,
    ))


TOOLS = [
    blender_generate_scene,
    blender_generate_thumbnail,
    blender_generate_title_card,
    blender_generate_data_viz,
    blender_generate_lower_third,
    blender_generate_latex,
    blender_generate_ui_mockup,
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
    "You are a professional video production director AI with access to 7 tools that generate "
    "3D Blender animations, Manim math animations, and device UI mockups. "
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
