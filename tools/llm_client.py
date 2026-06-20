"""
Central LLM factory for BlenderMCPServer.

Mirrors the Rust video_editor pattern where both GeminiClient and ClaudeClient
are optional AppState fields, selected by env var at startup.

Provider selection (LLM_PROVIDER env var):
  "ollama"   — always use self-hosted Ollama Gemma 4 (no API key needed, default)
  "gemini"   — always use Gemini (requires GEMINI_API_KEY)
  "nvidia"   — always use NVIDIA NIM Gemma (requires NVIDIA_API_KEY)
  "gemma"    — alias for NVIDIA NIM Gemma
  "deepseek" — always use DeepSeek (requires DEEPSEEK_API_KEY)
  "claude"   — always use Claude (requires ANTHROPIC_API_KEY)
  "auto"     — try Ollama first, then Gemini, then NVIDIA Gemma, then DeepSeek, then Claude (default)

Models:
  Ollama   — gemma4:12b           (overridable via OLLAMA_MODEL)
  Gemini   — gemini-2.5-flash     (overridable via GEMINI_MODEL)
  NVIDIA   — google/gemma-4-31b-it (overridable via NVIDIA_NIM_MODEL)
  DeepSeek — deepseek-v4-flash     (overridable via DEEPSEEK_MODEL)
  Claude   — claude-opus-4-6       (overridable via CLAUDE_MODEL)

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

import asyncio
import os
from typing import Any

import httpx


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

_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.0-flash")
_NVIDIA_NIM_ENDPOINT = os.getenv(
    "NVIDIA_NIM_ENDPOINT",
    "https://integrate.api.nvidia.com/v1/chat/completions",
)
_NVIDIA_NIM_MODEL = os.getenv("NVIDIA_NIM_MODEL", "google/gemma-4-31b-it")
_NVIDIA_NIM_TIMEOUT_SECONDS = float(os.getenv("NVIDIA_NIM_TIMEOUT_SECONDS", "75"))
_DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
_DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
_DEEPSEEK_TIMEOUT_SECONDS = float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "60"))
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://172.31.42.118:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:12b")
_OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
_CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")
_PROVIDER     = os.getenv("LLM_PROVIDER", "auto").lower()  # "ollama" | "gemini" | "nvidia" | "gemma" | "deepseek" | "claude" | "auto"


# ---------------------------------------------------------------------------
# Provider availability
# ---------------------------------------------------------------------------

def _has_ollama() -> bool:
    return True  # self-hosted, no API key needed


def _has_gemini() -> bool:
    return bool(os.getenv("VIDEO_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY"))


def _has_claude() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _has_nvidia() -> bool:
    return bool(os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY"))


def _has_deepseek() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY"))


def _resolved_provider() -> str:
    """Return the provider that will actually be used given current env."""
    if _PROVIDER == "ollama":
        return "ollama"

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

    if _PROVIDER in {"nvidia", "gemma"}:
        if not _has_nvidia():
            raise RuntimeError(
                "LLM_PROVIDER=nvidia/gemma but NVIDIA_API_KEY is not set"
            )
        return "nvidia"

    if _PROVIDER == "deepseek":
        if not _has_deepseek():
            raise RuntimeError(
                "LLM_PROVIDER=deepseek but DEEPSEEK_API_KEY is not set"
            )
        return "deepseek"

    # auto — prefer Ollama (self-hosted, free), then Gemini, then NVIDIA, then DeepSeek, then Claude
    if _has_ollama():
        return "ollama"
    if _has_gemini():
        return "gemini"
    if _has_nvidia():
        return "nvidia"
    if _has_deepseek():
        return "deepseek"
    if _has_claude():
        return "claude"
    raise RuntimeError(
        "No LLM provider available. Ollama should always be reachable."
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
                      "ollama" | "gemini" | "nvidia" | "gemma" | "deepseek" | "claude" | "auto" | None (use env).

    Returns:
        A LangChain BaseChatModel that supports .bind_tools() and .invoke().
    """
    resolved = _resolve(provider)

    if resolved == "ollama":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=_OLLAMA_MODEL,
            api_key="ollama",  # ignored by Ollama but required by ChatOpenAI
            base_url=_OLLAMA_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=_OLLAMA_TIMEOUT_SECONDS,
        )

    if resolved == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
            return ChatGoogleGenerativeAI(
                model=_GEMINI_MODEL,
                google_api_key=os.getenv("VIDEO_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY"),
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        except Exception:
            if _PROVIDER != "auto" or not _has_claude():
                raise
            # fall through to Claude (auto mode only)

    if resolved == "nvidia":
        try:
            from langchain_openai import ChatOpenAI  # type: ignore

            return ChatOpenAI(
                model=_NVIDIA_NIM_MODEL,
                api_key=os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY"),
                base_url=_NVIDIA_NIM_ENDPOINT.rsplit("/chat/completions", 1)[0],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=_NVIDIA_NIM_TIMEOUT_SECONDS,
            )
        except ImportError as exc:
            raise RuntimeError(
                "LLM_PROVIDER=nvidia requires langchain-openai for chat/tool models. "
                "Raw generate_text() already supports NVIDIA NIM without this package."
            ) from exc

    if resolved == "deepseek":
        try:
            from langchain_openai import ChatOpenAI  # type: ignore

            return ChatOpenAI(
                model=_DEEPSEEK_MODEL,
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=_DEEPSEEK_BASE_URL,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=_DEEPSEEK_TIMEOUT_SECONDS,
            )
        except ImportError as exc:
            raise RuntimeError(
                "LLM_PROVIDER=deepseek requires langchain-openai for chat/tool models. "
                "Raw generate_text() already supports DeepSeek without this package."
            ) from exc

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
    if override == "ollama":
        return "ollama"
    if override == "gemini":
        if not _has_gemini():
            raise RuntimeError("provider='gemini' but GEMINI_API_KEY is not set")
        return "gemini"
    if override == "claude":
        if not _has_claude():
            raise RuntimeError("provider='claude' but ANTHROPIC_API_KEY is not set")
        return "claude"
    if override in {"nvidia", "gemma"}:
        if not _has_nvidia():
            raise RuntimeError("provider='nvidia/gemma' but NVIDIA_API_KEY is not set")
        return "nvidia"
    if override == "deepseek":
        if not _has_deepseek():
            raise RuntimeError("provider='deepseek' but DEEPSEEK_API_KEY is not set")
        return "deepseek"
    # "auto"
    return _resolved_provider()


def _is_transient_gemini_error(exc: Exception) -> bool:
    message = str(exc)
    transient_markers = (
        "503",
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "429",
        "temporarily unavailable",
        "high demand",
        "try again later",
    )
    return any(marker in message for marker in transient_markers)


def _is_transient_nvidia_error(exc: Exception) -> bool:
    message = str(exc).lower()
    transient_markers = (
        "429",
        "408",
        "503",
        "504",
        "timeout",
        "temporarily unavailable",
        "rate limit",
        "connection",
    )
    return any(marker in message for marker in transient_markers)


def _is_transient_deepseek_error(exc: Exception) -> bool:
    message = str(exc).lower()
    transient_markers = (
        "429",
        "408",
        "503",
        "504",
        "timeout",
        "temporarily unavailable",
        "rate limit",
        "connection",
    )
    return any(marker in message for marker in transient_markers)


async def _generate_text_with_ollama(
    *,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": _OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=_OLLAMA_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{_OLLAMA_BASE_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Ollama error {response.status_code}: {response.text[:500]}"
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Ollama returned no content: {data}") from exc


async def _generate_text_with_nvidia(
    *,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set")

    payload = {
        "model": _NVIDIA_NIM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with httpx.AsyncClient(timeout=_NVIDIA_NIM_TIMEOUT_SECONDS) as client:
        response = await client.post(
            _NVIDIA_NIM_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"NVIDIA NIM error {response.status_code}: {response.text[:500]}"
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"NVIDIA NIM returned no content: {data}") from exc


async def _generate_text_with_deepseek(
    *,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    payload = {
        "model": _DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with httpx.AsyncClient(timeout=_DEEPSEEK_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{_DEEPSEEK_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"DeepSeek error {response.status_code}: {response.text[:500]}"
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"DeepSeek returned no content: {data}") from exc


async def _generate_text_with_gemini_model(
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    from google import genai as google_genai  # new google-genai SDK

    client = google_genai.Client(
        api_key=os.getenv("VIDEO_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=google_genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )

    try:
        return response.text
    except (ValueError, AttributeError):
        try:
            return response.candidates[0].content.parts[0].text
        except (IndexError, AttributeError) as inner_err:
            raise RuntimeError(
                f"Gemini returned empty/blocked response: {inner_err}"
            ) from inner_err


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

    if resolved == "ollama":
        ollama_errors: list[str] = []
        for attempt in range(1, 4):
            try:
                text = await _generate_text_with_ollama(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return text, "ollama"
            except Exception as ollama_err:
                ollama_errors.append(f"attempt {attempt}: {ollama_err}")
                if attempt < 3:
                    await asyncio.sleep(min(6, 2 * attempt))
                    continue
                break
        if _PROVIDER != "auto":
            raise RuntimeError("; ".join(ollama_errors))
        # fall through to Gemini in auto mode

    if resolved == "gemini":
        try:
            gemini_errors: list[str] = []
            gemini_models = [_GEMINI_MODEL]
            if _GEMINI_FALLBACK_MODEL and _GEMINI_FALLBACK_MODEL not in gemini_models:
                gemini_models.append(_GEMINI_FALLBACK_MODEL)

            for model in gemini_models:
                for attempt in range(1, 4):
                    try:
                        text = await _generate_text_with_gemini_model(
                            model=model,
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        return text, "gemini"
                    except Exception as gemini_err:
                        gemini_errors.append(f"{model} attempt {attempt}: {gemini_err}")
                        if attempt < 3 and _is_transient_gemini_error(gemini_err):
                            await asyncio.sleep(min(6, 2 * attempt))
                            continue
                        break
            raise RuntimeError("; ".join(gemini_errors))
        except Exception as gemini_err:
            # Only fall back in auto mode, never when explicitly pinned to Gemini.
            if _PROVIDER != "auto":
                raise RuntimeError(
                    f"Gemini generate_text failed: {gemini_err}"
                ) from gemini_err

            if _has_nvidia():
                try:
                    text = await _generate_text_with_nvidia(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return text, "nvidia"
                except Exception as nvidia_err:
                    if _has_deepseek():
                        try:
                            text = await _generate_text_with_deepseek(
                                prompt=prompt,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                            return text, "deepseek"
                        except Exception as deepseek_err:
                            if _has_claude():
                                pass
                            else:
                                raise RuntimeError(
                                    f"Gemini failed: {gemini_err}; NVIDIA NIM failed: {nvidia_err}; DeepSeek failed: {deepseek_err}"
                                ) from deepseek_err
                    elif _has_claude():
                        pass
                    else:
                        raise RuntimeError(
                            f"Gemini failed: {gemini_err}; NVIDIA NIM failed: {nvidia_err}"
                        ) from nvidia_err
            elif _has_deepseek():
                try:
                    text = await _generate_text_with_deepseek(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return text, "deepseek"
                except Exception as deepseek_err:
                    if _has_claude():
                        pass
                    else:
                        raise RuntimeError(
                            f"Gemini failed: {gemini_err}; DeepSeek failed: {deepseek_err}"
                        ) from deepseek_err
            elif not _has_claude():
                raise RuntimeError(
                    f"Gemini generate_text failed: {gemini_err}"
                ) from gemini_err

    if resolved == "nvidia":
        nvidia_errors: list[str] = []
        for attempt in range(1, 4):
            try:
                text = await _generate_text_with_nvidia(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return text, "nvidia"
            except Exception as nvidia_err:
                nvidia_errors.append(f"attempt {attempt}: {nvidia_err}")
                if attempt < 3 and _is_transient_nvidia_error(nvidia_err):
                    await asyncio.sleep(min(10, 2 * attempt))
                    continue
                break
        if _PROVIDER != "auto":
            raise RuntimeError("; ".join(nvidia_errors))
        if _has_deepseek():
            try:
                text = await _generate_text_with_deepseek(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return text, "deepseek"
            except Exception as deepseek_err:
                if _has_claude():
                    pass
                else:
                    raise RuntimeError(
                        f"NVIDIA NIM failed: {'; '.join(nvidia_errors)}; DeepSeek failed: {deepseek_err}"
                    ) from deepseek_err
        elif not _has_claude():
            raise RuntimeError("; ".join(nvidia_errors))

    if resolved == "deepseek":
        deepseek_errors: list[str] = []
        for attempt in range(1, 4):
            try:
                text = await _generate_text_with_deepseek(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return text, "deepseek"
            except Exception as deepseek_err:
                deepseek_errors.append(f"attempt {attempt}: {deepseek_err}")
                if attempt < 3 and _is_transient_deepseek_error(deepseek_err):
                    await asyncio.sleep(min(10, 2 * attempt))
                    continue
                break
        if _PROVIDER != "auto" or not _has_claude():
            raise RuntimeError("; ".join(deepseek_errors))

    # claude path
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
