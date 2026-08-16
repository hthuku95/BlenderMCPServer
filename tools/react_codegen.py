"""
react_codegen.py — Shared LangGraph ReAct loop that turns single-shot codegen
into a real agent.  Used by both bpy_codegen and manim_codegen.

Instead of the old procedural chain:
    generate (single LLM call) -> regex WEB_SEARCH hack -> docker check
    -> render -> hard-coded _fix_code retry loop (MAX_RETRIES=5)
the LLM now drives every decision turn-by-turn with bound tools:

    web_search(query)        real API research (replaces WEB_SEARCH marker hack)
    rag_retrieve(query)      pull past successful scripts similar to an error
    docker_validate(code)    fast sandbox pre-check for import/API errors
    run_render(code)         write code to a temp file and actually render it

The loop terminates when run_render returns ok; it otherwise iterates until
VIGA_REACT_MAX_TURNS (default 10) so a model that can't produce valid code
fails loudly instead of looping forever.

Gating: set VIGA_REACT_LOOP=false to fall back to the legacy procedural
retry inside bpy_codegen/manim_codegen. This module reads the same env knobs
as the old code (VIGA_ENABLE_DOCKER_SANDBOX) so behavior is preserved.
"""

from __future__ import annotations

import json
import os
import re
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools.llm_client import active_provider, get_chat_model

MAX_REACT_TURNS = int(os.getenv("VIGA_REACT_MAX_TURNS", "10"))

_VIGA_ENABLE_DOCKER = os.getenv("VIGA_ENABLE_DOCKER_SANDBOX", "").lower() in ("true", "1", "yes")


class CodegenState(TypedDict):
    messages: Annotated[list, add_messages]
    brief: str
    provider: str
    turn_count: int
    render_ok: bool
    final_path: str
    final_code: str


def _strip_fences(text: str) -> str:
    fence = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def _build_react_instructions(engine: str, system_prompt: str) -> str:
    """Append the agentic loop contract to the engine-specific system prompt."""
    return f"""{system_prompt}

═══ AGENTIC LOOP CONTRACT (override any conflicting instruction above) ═══
You are now operating as an agent in a tool loop. Follow this workflow:

1. Write the complete, runnable {engine} script.
2. You MAY call tools before rendering:
   - web_search: use when you are not sure about the correct {engine} API.
     Do NOT emit WEB_SEARCH: markers — use the web_search tool instead.
   - rag_retrieve: pull past successful scripts similar to a failing error.
   - docker_validate: fast pre-check; call with the code before rendering
     if you want a quick import/API sanity check.
3. Render by calling run_render with code as the `code` argument. Complete
   code only — no markdown fences needed (they are stripped automatically).
4. If run_render reports ok, you are done. The loop stops automatically.
   If it reports an error, fix the code and call run_render again.
5. Keep the scene/animation intent identical to the brief across fixes.
6. Simplify rather than guess — prefer APIs you are confident exist.
7. Each run_render failure gives you the full traceback; learn from it.

You have at most {MAX_REACT_TURNS} turns. Do not waste turns — prefer
rendering quickly, then fixing based on real errors.
"""


async def run_agentic_codegen(
    *,
    engine: str,
    system_prompt: str,
    brief: str,
    render_func: Callable,
    store_success: Callable | None = None,
    rag_collection: str,
    docker_script_type: str,
    provider: str | None = None,
    max_turns: int = MAX_REACT_TURNS,
) -> tuple[str, str]:
    """
    Run the ReAct codegen loop. Returns (output_path, code).

    Args:
        engine:              "blender" | "manim" (used in prompts/logs).
        system_prompt:       The full engine system prompt (already includes
                             style guide, output requirements, task brief).
        brief:               Original creative brief, used for RAG queries.
        render_func:         async (code: str) -> {"output_path": str} on
                             success or {"error": str} on failure.
        store_success:       async (code, brief) to persist winning code.
        rag_collection:      Qdrant collection ("bpy" | "manim").
        docker_script_type:  Sandbox script_type ("blender" | "manim").
        provider:            LLM provider override (see llm_client).
        max_turns:           Loop cap (env VIGA_REACT_MAX_TURNS).
    """
    system_content = _build_react_instructions(engine, system_prompt)

    @tool
    async def web_search(query: str, num_results: int = 5) -> str:
        """Search the web for API documentation or Blender/Manim usage examples."""
        from tools.browserbase_client import browserbase_search
        return await browserbase_search(query, num_results)

    @tool
    async def rag_retrieve(query: str) -> str:
        """Retrieve past successful scripts similar to the given error or context."""
        from tools.rag_client import query_similar
        return await query_similar(rag_collection, query, brief)

    @tool
    async def docker_validate(code: str) -> str:
        """Run a fast Docker sandbox pre-check on the code. Returns an error
        message if validation fails, or \"OK\" if it passes (or is disabled)."""
        if not _VIGA_ENABLE_DOCKER:
            return "OK (Docker sandbox validation disabled)"
        try:
            from tools.docker_sandbox import validate_in_docker
            result = await validate_in_docker(code, docker_script_type, timeout=15)
        except Exception as exc:
            return f"Docker validation unavailable: {exc}"
        if result.get("passed"):
            return "OK"
        logs = str(result.get("logs", ""))[-3000:]
        err = str(result.get("error", "Docker sandbox validation failed"))
        return f"[Docker sandbox pre-check] {err}\nLogs:\n{logs}"

    @tool
    async def run_render(code: str) -> str:
        """Write the code to a temp file, render it, and return the result.
        Returns JSON: {\"ok\": true, \"output_path\": \"...\"} on success or
        {\"ok\": false, \"error\": \"...\"} on failure."""
        cleaned = _strip_fences(code)
        result = await render_func(cleaned)
        if "error" in result:
            return json.dumps({"ok": False, "error": result["error"]})
        if store_success:
            try:
                await store_success(cleaned, brief)
            except Exception:
                pass
        return json.dumps({"ok": True, "output_path": result["output_path"]})

    tools = [web_search, rag_retrieve, run_render]
    if _VIGA_ENABLE_DOCKER:
        tools.insert(2, docker_validate)

    async def agent_node(state: CodegenState) -> dict:
        llm = get_chat_model(
            temperature=0.3,
            max_tokens=8192,
            provider=state.get("provider") or provider,
        ).bind_tools(tools)
        full_messages = [SystemMessage(content=system_content)] + list(state["messages"])
        response = llm.invoke(full_messages)
        return {
            "messages": [response],
            "turn_count": state.get("turn_count", 0) + 1,
            "render_ok": state.get("render_ok", False),
            "final_path": state.get("final_path", ""),
            "final_code": state.get("final_code", ""),
        }

    async def tools_node(state: CodegenState) -> dict:
        result = ToolNode(tools).invoke(state)
        last_tool = None
        for msg in result.get("messages", []):
            if getattr(msg, "name", "") == "run_render":
                last_tool = msg
        out = {
            "messages": result.get("messages", []),
            "turn_count": state.get("turn_count", 0),
        }
        if last_tool is not None:
            try:
                data = json.loads(last_tool.content)
            except (json.JSONDecodeError, AttributeError, TypeError):
                data = {}
            if data.get("ok"):
                out["render_ok"] = True
                out["final_path"] = data.get("output_path", "")
                last_msg = state["messages"][-1] if state["messages"] else None
                code = ""
                if last_msg is not None and hasattr(last_msg, "tool_calls"):
                    for tc in last_msg.tool_calls or []:
                        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                        if name == "run_render":
                            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                            code = args.get("code", "")
                            break
                out["final_code"] = _strip_fences(code)
        return out

    def should_continue(state: CodegenState) -> str:
        if state.get("render_ok"):
            return END
        if state.get("turn_count", 0) >= max_turns:
            return END
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(CodegenState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    compiled = graph.compile()

    resolved_provider = provider or active_provider()
    initial = {
        "messages": [HumanMessage(content=brief)],
        "brief": brief,
        "provider": resolved_provider,
        "turn_count": 0,
        "render_ok": False,
        "final_path": "",
        "final_code": "",
    }
    result = await compiled.ainvoke(initial)

    if result.get("render_ok") and result.get("final_path"):
        return (result["final_path"], result.get("final_code", ""))
    raise RuntimeError(
        f"agentic {engine} codegen failed after {result.get('turn_count', max_turns)} "
        f"turns with no successful render."
    )