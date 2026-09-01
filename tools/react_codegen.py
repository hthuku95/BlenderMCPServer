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

import asyncio
import json
import logging
import os
import re
import time
from typing import Annotated, Any, Callable, TypedDict

import httpx

logger = logging.getLogger("react_codegen")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools.llm_client import active_provider, get_chat_model

MAX_REACT_TURNS = int(os.getenv("VIGA_REACT_MAX_TURNS", "10"))

# Context-bloat guards. The ReAct loop feeds the full accumulated message
# history back to the model every turn; without caps, web_search/rag/docker
# logs and render tracebacks balloon the prompt to ~45K tokens, which the
# single-slot Ollama GPU prefill takes minutes to chew — wedging the whole
# job. Keep history small and truncate every tool result.
_MAX_HISTORY_MESSAGES = int(os.getenv("VIGA_REACT_MAX_HISTORY", "6"))
_EMPTY_RESPONSE_MAX_RETRIES = int(os.getenv("VIGA_REACT_EMPTY_RETRIES", "2"))
_MAX_TOOL_RESULT_CHARS = int(os.getenv("VIGA_REACT_MAX_TOOL_RESULT", "2000"))

# GPU admission control (Phase 2): Ollama serves ~1 concurrent request per
# GPU (NUM_PARALLEL=1); concurrent codegen loops queue behind each other and
# starve probes/turns (observed Aug 30-31: two interleaved jobs doubled each
# other's latency and blew the 1800s JOB_TIMEOUT). Bound concurrent ReAct
# loops per node; 0 disables the limit. Workers still receive jobs — they
# just wait here for a codegen slot.
_CODEGEN_CONCURRENCY = int(os.getenv("CODEGEN_MAX_CONCURRENT", "1"))
_codegen_sem: asyncio.Semaphore | None = None


def _codegen_semaphore() -> asyncio.Semaphore | None:
    global _codegen_sem
    if _CODEGEN_CONCURRENCY <= 0:
        return None
    if _codegen_sem is None:
        _codegen_sem = asyncio.Semaphore(_CODEGEN_CONCURRENCY)
    return _codegen_sem


# Render budget (Phase 2): one shared deadline across all ReAct turns.
# Enforced at turn boundaries so a job that cannot converge fails FAST with a
# clear error instead of being killed mid-render by the whole-job timeout.
# The whole-job JOB_TIMEOUT must remain the larger number (backstop only).
_RENDER_BUDGET_SECS = float(os.getenv("RENDER_BUDGET_SECS", "1200"))

# Durable execution (Phase 2): checkpoint the graph after every turn via
# LangGraph's Postgres saver so a crashed/reclaimed run resumes instead of
# restarting from turn 1. Kill switch: VIGA_CHECKPOINTER=off.
_CHECKPOINTER_ENABLED = os.getenv("VIGA_CHECKPOINTER", "on").lower() not in ("off", "false", "0", "no")
_CHECKPOINT_MAX_AGE_SECS = float(os.getenv("VIGA_CHECKPOINT_MAX_AGE_SECS", str(6 * 3600)))

# Cancel polling (Phase 2): cross-node cancel is honored by checking the
# job's DDB state between turns. 0 disables polling.
_CANCEL_POLL_SECS = float(os.getenv("VIGA_CANCEL_POLL_SECS", "30"))


class JobCancelled(RuntimeError):
    """Raised between turns when the job was cancelled in DDB (cross-node)."""


async def _job_cancelled(thread_id: str) -> bool:
    if not thread_id:
        return False
    try:
        from tools.job_queue import _async_get_status
        status = await _async_get_status(thread_id)
        return bool(status and status.state == "cancelled")
    except Exception:
        return False

# GPU health-probe guard. Before spending a full job's time on a ReAct loop,
# check that the Ollama backend actually returns tool_calls on a minimal probe.
# A T4 running gemma4:12b takes 30-60s+ for ANY tool-call generation, and with
# Ollama processing requests serially (NUM_PARALLEL=1) a probe can queue behind
# an in-flight generation — so the probe needs generous timeout and retries to
# avoid false "unhealthy" verdicts. We fail fast with a clear message instead
# of burning MAX_TURNS against a genuinely broken backend.
_GPU_PROBE_MAX_FAILS = int(os.getenv("VIGA_GPU_PROBE_MAX_FAILS", "3"))
_GPU_PROBE_TIMEOUT = float(os.getenv("VIGA_GPU_PROBE_TIMEOUT", "120"))

_VIGA_ENABLE_DOCKER = os.getenv("VIGA_ENABLE_DOCKER_SANDBOX", "").lower() in ("true", "1", "yes")


class CodegenState(TypedDict):
    messages: Annotated[list, add_messages]
    brief: str
    provider: str
    turn_count: int
    render_ok: bool
    final_path: str
    final_code: str
    last_error: str
    trace: list


def _strip_fences(text: str) -> str:
    fence = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def _extract_script_from_text(text: str, engine: str) -> str:
    """Best-effort extraction of a runnable script from a text-only model
    response (the failure mode where a model answers in prose instead of
    calling run_render). Returns "" when the text is not plausibly code.

    This is the text-salvage path: a model that emits a complete script as
    plain text still gets it rendered instead of the harness discarding it.
    """
    text = text or ""
    fenced = re.search(r"```(?:python|blender|manim)?\s*\n(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    candidate = candidate.strip()
    if not candidate:
        return ""
    # Require a definitive signal that this is code, not chat/explanations.
    if engine == "blender":
        strong = ("import bpy" in candidate, "bpy." in candidate)
    else:
        strong = ("import manim" in candidate, "from manim import" in candidate)
    weak = any(h in candidate for h in ("=", "class ", "def ", "("))
    if fenced:
        # A fenced block is unambiguous — trust it even on weak signals.
        return candidate
    if not (any(strong) and weak) and not candidate.startswith("import "):
        return ""
    # Drop any trailing prose the model appended after a code block.
    for marker in ("\nExplanation:", "\nHere is", "\n```"):
        idx = candidate.find(marker)
        if idx > 0:
            candidate = candidate[:idx].rstrip()
    return candidate


def _clip(text: str, limit: int = _MAX_TOOL_RESULT_CHARS) -> str:
    """Bound a tool result to stop the prompt ballooning to 45K tokens."""
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.7)]
    tail = text[-int(limit * 0.3):]
    return f"{head}\n…[truncated {len(text) - limit} chars]…\n{tail}"


async def _gpu_healthy(provider: str) -> tuple[bool, str]:
    """Probe the Ollama backend for real tool-calling capability.

    Returns (ok, detail). Only meaningful for the ollama provider — Gemini /
    Claude / DeepSeek / NVIDIA are remote and don't suffer the same single-slot
    refusal mode, so this is a no-op pass for them.

    The probe sends a tiny tool-scoped /api/chat request and requires a
    tool_calls array in the reply. An empty/refusal reply (no tool_calls, empty
    content) means the shared slot dropped the request — we treat that as an
    unhealthy backend so callers can fail fast instead of burning MAX_TURNS.
    """
    if provider != "ollama":
        return True, f"provider={provider} (probe skipped)"
    base = (os.getenv("OLLAMA_BASE_URL") or "http://172.31.43.45:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL") or "gemma4:12b"
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": "Call run_render with code that creates a blue cube."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "run_render",
                "description": "Render a Blender scene from code",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }],
    }
    try:
        async with httpx.AsyncClient(timeout=_GPU_PROBE_TIMEOUT) as client:
            resp = await client.post(f"{base}/api/chat", json=payload)
            resp.raise_for_status()
            j = resp.json()
        mc = j.get("message", {})
        types = (mc or {}).get("tool_calls") or []
        content = (mc or {}).get("content") or ""
        if types:
            return True, f"ollama probe ok (tool_calls={len(types)})"
        return False, f"ollama probe returned EMPTY/refusal (content={'empty' if not content else repr(content[:40])})"
    except Exception as exc:  # noqa: BLE001
        return False, f"ollama probe error: {type(exc).__name__}: {str(exc)[:120]}"


async def _probe_gpu_or_raise(provider: str) -> None:
    """Run the GPU probe up to _GPU_PROBE_MAX_FAILS times before raising.

    Fails fast with a clear, diagnosable error instead of letting the ReAct
    loop grind through MAX_TURNS against a backend that returns refusals.
    """
    if provider != "ollama":
        return
    last = "no probe attempted"
    for attempt in range(1, _GPU_PROBE_MAX_FAILS + 1):
        ok, detail = await _gpu_healthy(provider)
        if ok:
            return
        last = detail
        logger.warning("gpu probe failed attempt %s/%s: %s", attempt, _GPU_PROBE_MAX_FAILS, detail)
        await asyncio.sleep(1)
    raise RuntimeError(
        f"Ollama backend unhealthy ({_GPU_PROBE_MAX_FAILS} consecutive probes failed): {last}. "
        "Skipping codegen — check GPU pool health / context scheduling."
    )


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
    thread_id: str = "",
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

    # Phase 2: admission control — one codegen loop per node by default.
    sem = _codegen_semaphore()
    if sem is not None:
        await sem.acquire()
    deadline = (time.monotonic() + _RENDER_BUDGET_SECS) if _RENDER_BUDGET_SECS > 0 else None
    last_cancel_probe = 0.0

    def _budget_left() -> float:
        if deadline is None:
            return float("inf")
        return deadline - time.monotonic()

    async def _check_turn_gate() -> None:
        """Raised between turns: budget exhaustion (hard stop) or a
        cross-node cancel observed in DDB."""
        nonlocal last_cancel_probe
        left = _budget_left()
        if left <= 0:
            raise RuntimeError(
                f"RENDER_BUDGET_EXCEEDED after {_RENDER_BUDGET_SECS:.0f}s budget "
                f"({engine} codegen did not converge; raise RENDER_BUDGET_SECS or fix codegen feedback)"
            )
        now = time.monotonic()
        if _CANCEL_POLL_SECS > 0 and (now - last_cancel_probe) >= _CANCEL_POLL_SECS:
            last_cancel_probe = now
            if await _job_cancelled(thread_id):
                raise JobCancelled(f"job {thread_id} cancelled (observed in DDB between turns)")

    @tool
    async def web_search(query: str, num_results: int = 5) -> str:
        """Search the web for API documentation or Blender/Manim usage examples."""
        from tools.browserbase_client import browserbase_search
        return _clip(await browserbase_search(query, num_results))

    @tool
    async def rag_retrieve(query: str) -> str:
        """Retrieve past successful scripts similar to the given error or context."""
        from tools.rag_client import query_similar
        return _clip(await query_similar(rag_collection, query, brief))

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
        return _clip(f"[Docker sandbox pre-check] {err}\nLogs:\n{logs}")

    @tool
    async def run_render(code: str) -> str:
        """Write the code to a temp file, render it, and return the result.
        Returns JSON: {\"ok\": true, \"output_path\": \"...\"} on success or
        {\"ok\": false, \"error\": \"...\"} on failure."""
        cleaned = _strip_fences(code)
        try:
            from tools.code_guards import apply_all as _apply_script_guards
            cleaned = _apply_script_guards(cleaned)
        except Exception:
            pass
        result = await render_func(cleaned)
        if "error" in result:
            return json.dumps({"ok": False, "error": _clip(str(result["error"]))})
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
        await _check_turn_gate()
        llm = get_chat_model(
            temperature=0.3,
            max_tokens=8192,
            provider=state.get("provider") or provider,
        ).bind_tools(tools)
        history = list(state["messages"])
        if len(history) > _MAX_HISTORY_MESSAGES:
            history = history[-_MAX_HISTORY_MESSAGES:]
        full_messages = [SystemMessage(content=system_content)] + history
        response = await llm.ainvoke(full_messages)
        # EMPTY-RESPONSE RETRY: gemma4 can intermittently return a refusal with
        # empty content and no tool_calls (seen live: additional_kwargs removed
        # '../refusal'). A single nudge usually gets a real reply. Bounded so it
        # can never spin forever.
        _empty_retries = 0
        while (
            not (getattr(response, "content", "") or "")
            and not list(getattr(response, "tool_calls", None) or [])
            and _empty_retries < _EMPTY_RESPONSE_MAX_RETRIES
        ):
            _empty_retries += 1
            logger.warning(
                "codegen turn=%s empty response (refusal), retrying with nudge %s/%s",
                state.get("turn_count", 0) + 1, _empty_retries, _EMPTY_RESPONSE_MAX_RETRIES,
            )
            response = await llm.ainvoke(
                full_messages
                + [
                    HumanMessage(
                        content=(
                            "Your previous turn produced no output. Respond now: either "
                            "call the run_render tool with a complete Blender script that "
                            "renders the requested scene, or reply with the script text."
                        )
                    )
                ]
            )
        # TEXT-SALVAGE: if the model answered in prose (no tool call) but the
        # text contains a plausible script, render it via run_render instead of
        # letting the loop end at turn 1 and discard a complete answer.
        raw_text = getattr(response, "content", "") or ""
        raw_repr = str(response)[:600]
        tool_calls = list(getattr(response, "tool_calls", None) or [])
        logger.warning(
            "codegen turn=%s tool_calls=%s content_type=%s content_chars=%s repr=%s",
            state.get("turn_count", 0) + 1,
            [tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "") for tc in tool_calls],
            type(raw_text).__name__,
            len(raw_text),
            raw_repr,
        )
        extract_len = 0
        salvaged = False
        if not tool_calls:
            code = _extract_script_from_text(raw_text, engine)
            extract_len = len(code)
            if code:
                salvaged = True
                response = AIMessage(
                    content=raw_text,
                    tool_calls=[
                        {
                            "name": "run_render",
                            "args": {"code": code},
                            "id": f"salvage_{state.get('turn_count', 0)}",
                        }
                    ],
                )
        return {
            "messages": [response],
            "turn_count": state.get("turn_count", 0) + 1,
            "render_ok": state.get("render_ok", False),
            "final_path": state.get("final_path", ""),
            "final_code": state.get("final_code", ""),
            "last_error": state.get("last_error", ""),
            "trace": state.get("trace", []) + [
                {
                    "turn": state.get("turn_count", 0) + 1,
                    "tool_calls": [tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "") for tc in tool_calls],
                    "salvaged": salvaged,
                    "extract_len": extract_len,
                    "model_text": raw_text[-400:],
                    "model_repr": raw_repr,
                }
            ],
        }

    async def tools_node(state: CodegenState) -> dict:
        result = await ToolNode(tools).ainvoke(state)
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
            else:
                out["last_error"] = str(data.get("error", "") or "")[:2000]
        trace = list(state.get("trace", []))
        if trace:
            try:
                trace[-1]["render"] = "ok" if data.get("ok") else "error"
                trace[-1]["error_tail"] = str(data.get("error", "") or "")[-300:]
                out["trace"] = trace
            except Exception:
                out["trace"] = trace
        return out

    def should_continue(state: CodegenState) -> str:
        if state.get("render_ok"):
            return END
        if _budget_left() <= 0:
            return END
        if state.get("turn_count", 0) >= max_turns:
            return END
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    def _build_graph(checkpointer=None):
        g = StateGraph(CodegenState)
        g.add_node("agent", agent_node)
        g.add_node("tools", tools_node)
        g.set_entry_point("agent")
        g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        g.add_edge("tools", "agent")
        return g.compile(checkpointer=checkpointer) if checkpointer else g.compile()

    resolved_provider = provider or active_provider()
    await _probe_gpu_or_raise(resolved_provider)

    checkpointer = None
    cfg = None
    if _CHECKPOINTER_ENABLED and thread_id:
        try:
            from tools.workflow_runtime import get_checkpointer, workflow_config
            checkpointer = await get_checkpointer()
            cfg = workflow_config(thread_id)
        except Exception as exc:
            logger.warning("codegen checkpointer unavailable, running without: %s", exc)
            checkpointer = None

    initial = {
        "messages": [HumanMessage(content=brief)],
        "brief": brief,
        "provider": resolved_provider,
        "turn_count": 0,
        "render_ok": False,
        "final_path": "",
        "final_code": "",
        "last_error": "",
        "trace": [],
    }

    resume = False
    if checkpointer is not None and cfg is not None:
        try:
            tuple_ = await checkpointer.aget_tuple(cfg)
            if tuple_ and tuple_.checkpoint:
                saved = tuple_.checkpoint.get("channel_values", {}) or {}
                saved_turns = int(saved.get("turn_count", 0) or 0)
                saved_ok = bool(saved.get("render_ok", False))
                age_ok = True
                ts = getattr(tuple_, "checkpoint", {}).get("ts") if hasattr(tuple_, "checkpoint") else None
                if ts:
                    try:
                        from datetime import datetime as _dt
                        age = time.time() - _dt.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
                        age_ok = age <= _CHECKPOINT_MAX_AGE_SECS
                    except Exception:
                        pass
                if age_ok and 0 < saved_turns < max_turns and not saved_ok:
                    resume = True
                    logger.warning("codegen resume thread=%s from turn=%s", thread_id, saved_turns)
        except Exception as exc:
            logger.warning("codegen checkpoint probe failed, fresh start: %s", exc)
            resume = False

    try:
        _graph = _build_graph(checkpointer)
        if resume:
            result = await _graph.ainvoke(None, config=cfg)
        elif cfg:
            result = await _graph.ainvoke(initial, config=cfg)
        else:
            result = await _graph.ainvoke(initial)
    finally:
        if sem is not None:
            sem.release()

    if _budget_left() <= 0 and not result.get("render_ok"):
        raise RuntimeError(
            f"RENDER_BUDGET_EXCEEDED after {_RENDER_BUDGET_SECS:.0f}s budget "
            f"({engine} codegen did not converge; raise RENDER_BUDGET_SECS or fix codegen feedback)"
        )

    if result.get("render_ok") and result.get("final_path"):
        return (result["final_path"], result.get("final_code", ""))
    last_error = (result.get("last_error") or "").strip()
    if last_error:
        # Root-cause wins: if the last render produced a real error, surface it
        # so operators can see WHY codegen failed (not just "no successful render").
        raise RuntimeError(
            f"agentic {engine} codegen failed after {result.get('turn_count', max_turns)} "
            f"turns with no successful render. Last error: {last_error[-2000:]}"
        )
    raise RuntimeError(
        f"agentic {engine} codegen failed after {result.get('turn_count', max_turns)} "
        f"turns with no successful render (provider={resolved_provider}). "
        f"trace={json.dumps(result.get('trace', []))}"
    )