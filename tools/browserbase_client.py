"""
BrowserBase search + fetch client for LLM-assisted code debugging.

Integrates two Browserbase cloud APIs:
  - Web Search (POST /v1/search)  — structured search results
  - Fetch     (POST /v1/fetch)    — full page content as markdown

Usage
-----
    from tools.browserbase_client import browserbase_search, browserbase_fetch

    results = await browserbase_search("blender bpy add keyframe material node")
    content = await browserbase_fetch("https://docs.manim.community/...")
"""
from __future__ import annotations

import os
from typing import Optional

import httpx


API_KEY_ENV = "BROWSERBASE_API_KEY"
BASE_URL = "https://api.browserbase.com/v1"
DEFAULT_TIMEOUT = 30  # seconds


def _headers() -> dict[str, str]:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"BrowserBase API key not set. Set {API_KEY_ENV} environment variable."
        )
    return {
        "X-BB-API-Key": key,
        "Content-Type": "application/json",
    }


async def browserbase_search(
    query: str,
    num_results: int = 5,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Search the web via BrowserBase and return formatted results.

    Args:
        query: Natural language search query (max 200 chars).
        num_results: Number of results to return (1-25, default 5).
        timeout: HTTP client timeout in seconds.

    Returns:
        Formatted string with result titles, URLs, and metadata.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{BASE_URL}/search",
                headers=_headers(),
                json={"query": query[:200], "numResults": min(num_results, 25)},
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        if not results:
            return f"No results found for: {query}"

        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            author = r.get("author", "")
            date = r.get("publishedDate", "")
            meta = ""
            if author:
                meta += f" by {author}"
            if date:
                meta += f" ({date[:10]})"
            lines.append(f"{i}. {title}{meta}")
            lines.append(f"   {url}")
        return "\n".join(lines)

    except httpx.HTTPStatusError as exc:
        return f"Web search failed (HTTP {exc.response.status_code}): {exc.response.text[:200]}"
    except Exception as exc:
        return f"Web search failed: {exc}"


async def browserbase_fetch(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
    max_chars: int = 8000,
) -> str:
    """Fetch a page via BrowserBase and return clean markdown content.

    Args:
        url: The URL to fetch.
        timeout: HTTP client timeout in seconds.
        max_chars: Maximum characters to return (content truncated to this).

    Returns:
        Markdown content or error message.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{BASE_URL}/fetch",
                headers=_headers(),
                json={"url": url, "format": "markdown"},
            )
            data = resp.json()

        if resp.status_code != 200:
            msg = data.get("message", data.get("error", "unknown error"))
            return f"Fetch failed (HTTP {resp.status_code}): {msg}"

        content = data.get("content", "")
        if not content or not content.strip():
            return f"Page returned empty content: {url}"

        if isinstance(content, str) and len(content) > max_chars:
            content = content[:max_chars] + "\n\n[...truncated]"

        return content

    except httpx.HTTPStatusError as exc:
        return f"Fetch failed (HTTP {exc.response.status_code}): {exc.response.text[:200]}"
    except Exception as exc:
        return f"Fetch failed: {exc}"
