"""
Central LLM factory for BlenderMCPServer.

Mirrors the Rust video_editor pattern where both GeminiClient and ClaudeClient
are optional AppState fields, selected by env var at startup.

Provider selection (LLM_PROVIDER env var):
  "gemini"  — always use Gemini (requires GEMINI_API_KEY)
  "claude"  — always use Claude (requires ANTHROPIC_API_KEY)
  "auto"    — try Gemini first (faster/cheaper), fall back to Claude (default)

Models:
  Gemini  — gemini-2.0-flash  (overridable via GEMINI_MODEL)
  Claude  — claude-opus-4-6   (overridable via CLAUDE_MODEL)
  Claude opus-4-6 is the director default because it makes better multi-step
  tool-use decisions than Sonnet when orchestrating 7 render tools at once.

LangSmith tracing (Phase 5):
  Set LANGCHAIN_API_KEY to enable automatic LangSmith tracing of all
  LangChain/LangGraph calls.  Optional but highly recommended for debugging
  Director agent runs in production.
  LANGCHAIN_PROJECT   — project name (default "BlenderMCPServer")
  LANGCHAIN_TRACING_V2 — set to "true" automatically when LANGCHAIN_API_KEY is present

Usage:
    from tools.llm_client import get_chat_model, generate_text

    # LangChain chat model (supports .bind_tools())
    llm = get_chat_model()
    llm_with_tools = llm.bind_tools(my_tools)

    # Raw text generation (no tools, simple Q&A)
    text, provider = await generate_text("Describe this scene in 3 words.")
"""
from __future__ import annotations

import os
from typing import Any


# ---------------------------------------------------------------------------
# LangSmith tracing — enable automatically when API key is present (Phase 5)
# ---------------------------------------------------------------------------

def _configure_langsmith() -> None:
    """Set LangChain env vars for LangSmith tracing if LANGCHAIN_API_KEY is set."""
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "BlenderMCPServer")


_configure_langsmith()


# ---------------------------------------------------------------------------
# Model name constants (overridable via env)
# ---------------------------------------------------------------------------

_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
_CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")
_PROVIDER     = os.getenv("LLM_PROVIDER", "auto").lower()  # "gemini" | "claude" | "auto"


# ---------------------------------------------------------------------------
# Provider availability
# ---------------------------------------------------------------------------

def _has_gemini() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def _has_claude() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _resolved_provider() -> str:
    """Return the provider that will actually be used given current env."""
    if _PROVIDER == "gemini":
        if not _has_gemini():
            raise RuntimeError(
                "LLM_PROVIDER=gemini but GEMINI_API_KEY is not set"
            )
        return "gemini"

    if _PROVIDER == "claude":
        if not _has_claude():
            raise RuntimeError(
                "LLM_PROVIDER=claude but ANTHROPIC_API_KEY is not set"
            )
        return "claude"

    # auto — prefer Gemini, fall back to Claude
    if _has_gemini():
        return "gemini"
    if _has_claude():
        return "claude"
    raise RuntimeError(
        "No LLM API key found. Set GEMINI_API_KEY or ANTHROPIC_API_KEY."
    )


# ---------------------------------------------------------------------------
# LangChain chat model factory
# ---------------------------------------------------------------------------

def get_chat_model(
    temperature: float = 0.7,
    max_tokens: int = 4096,
    provider: str | None = None,
) -> Any:
    """
    Return a LangChain chat model instance for the active provider.

    Args:
        temperature:  Sampling temperature (0.0–1.0).
        max_tokens:   Max output tokens.
        provider:     Override the LLM_PROVIDER env var for this call.
                      "gemini" | "claude" | "auto" | None (use env).

    Returns:
        A LangChain BaseChatModel that supports .bind_tools() and .invoke().
    """
    resolved = _resolve(provider)

    if resolved == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        return ChatGoogleGenerativeAI(
            model=_GEMINI_MODEL,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    # claude
    from langchain_anthropic import ChatAnthropic  # type: ignore
    return ChatAnthropic(
        model=_CLAUDE_MODEL,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _resolve(override: str | None) -> str:
    """Resolve provider from override or env."""
    if override is None:
        return _resolved_provider()

    override = override.lower()
    if override == "gemini":
        if not _has_gemini():
            raise RuntimeError("provider='gemini' but GEMINI_API_KEY is not set")
        return "gemini"
    if override == "claude":
        if not _has_claude():
            raise RuntimeError("provider='claude' but ANTHROPIC_API_KEY is not set")
        return "claude"
    # "auto"
    return _resolved_provider()


# ---------------------------------------------------------------------------
# Simple raw text generation (no tools)
# ---------------------------------------------------------------------------

async def generate_text(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    provider: str | None = None,
) -> tuple[str, str]:
    """
    Generate a plain text response from the active LLM.

    Returns: (response_text: str, provider_used: str)
    """
    resolved = _resolve(provider)

    if resolved == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(
            _GEMINI_MODEL,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        resp = model.generate_content(prompt)
        return resp.text, "gemini"

    # claude path (only reached when LLM_PROVIDER=claude explicitly)
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=_CLAUDE_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text, "claude"


# ---------------------------------------------------------------------------
# Convenience: provider name for logging
# ---------------------------------------------------------------------------

def active_provider() -> str:
    """Return the provider that would be used right now (no side-effects)."""
    try:
        return _resolved_provider()
    except RuntimeError:
        return "none"
