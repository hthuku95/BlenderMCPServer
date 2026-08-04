"""
RAG client — stores & retrieves known-working code patterns in Qdrant.

Two collections:
  - manim_code_patterns  (ManimCE)
  - bpy_code_patterns    (Blender bpy)

On success:  stores (code, description) keyed by embedding of description.
On retry:    embeds (error + description), queries Qdrant, injects similar
             patterns into the fix prompt so the LLM can learn from past
             successes instead of guessing.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import httpx

from tools.embedding_client import generate_embedding

QDRANT_URL = os.getenv("QDRANT_URL", "").rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

_COLLECTIONS = {
    "manim": "manim_code_patterns",
    "bpy": "bpy_code_patterns",
}
_EMBEDDING_DIM = 768

_HEADERS = {"Content-Type": "application/json"}
if QDRANT_API_KEY:
    _HEADERS["api-key"] = QDRANT_API_KEY


async def _ensure_collection(collection: str) -> None:
    """Create the collection if it does not exist."""
    if not QDRANT_URL:
        return
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{QDRANT_URL}/collections/{collection}",
            headers=_HEADERS,
        )
        if resp.status_code == 404:
            await client.put(
                f"{QDRANT_URL}/collections/{collection}",
                headers=_HEADERS,
                json={
                    "vectors": {
                        "size": _EMBEDDING_DIM,
                        "distance": "Cosine",
                    },
                },
            )


async def store_success(
    framework: str,
    code: str,
    description: str,
) -> None:
    """Store a successfully rendered script for future RAG retrieval."""
    if not QDRANT_URL or not QDRANT_API_KEY or framework not in _COLLECTIONS:
        return
    collection = _COLLECTIONS[framework]

    try:
        vector = generate_embedding(description, dimension=_EMBEDDING_DIM)
    except RuntimeError:
        return

    await _ensure_collection(collection)

    point_id = int(hashlib.sha256(code.encode()).hexdigest()[:16], 16)

    async with httpx.AsyncClient() as client:
        await client.put(
            f"{QDRANT_URL}/collections/{collection}/points",
            headers=_HEADERS,
            json={
                "points": [{
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "code": code[:8000],
                        "description": description[:1000],
                        "framework": framework,
                    },
                }],
            },
        )


async def query_similar(
    framework: str,
    error: str,
    description: str,
    top_k: int = 2,
) -> str:
    """
    Query Qdrant for code patterns relevant to the current error + description.

    Returns a formatted string with the top-k matching code snippets,
    or empty string if no matches found / Qdrant unavailable.
    """
    if not QDRANT_URL or not QDRANT_API_KEY or framework not in _COLLECTIONS:
        return ""
    collection = _COLLECTIONS[framework]

    text_for_embed = f"{description}\n{error}"[:4000]

    try:
        vector = generate_embedding(text_for_embed, dimension=_EMBEDDING_DIM)
    except RuntimeError:
        return ""

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{QDRANT_URL}/collections/{collection}/points/search",
                headers=_HEADERS,
                json={
                    "vector": vector,
                    "limit": top_k,
                    "with_payload": ["code", "description"],
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return ""

    results = data.get("result", [])
    matches = []
    for r in results:
        payload = r.get("payload", {})
        code_snippet = payload.get("code", "")
        desc = payload.get("description", "")
        score = r.get("score", 0)
        if code_snippet and score > 0.55:
            matches.append(
                f"── Similar pattern (score={score:.3f}) ──\n"
                f"Original task: {desc}\n"
                f"```python\n{code_snippet}\n```"
            )

    if not matches:
        return ""

    return f"\n═══ SIMILAR WORKING CODE (from Qdrant) ═══\n" + "\n\n".join(matches) + "\n"
