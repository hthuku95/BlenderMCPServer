"""
Shared LangGraph workflow runtime helpers.

This module centralizes checkpointer selection so long-running Blender and
Manim workflows can use durable execution in production without making local
development painful. If a Postgres connection is available and the optional
LangGraph Postgres package is installed, we use AsyncPostgresSaver. Otherwise
we fall back to an in-memory saver.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)

_CHECKPOINTER: Any | None = None
_CHECKPOINTER_CTX: Any | None = None
_CHECKPOINTER_MODE = "memory"
_CHECKPOINTER_LOCK = asyncio.Lock()


def workflow_config(thread_id: str, checkpoint_ns: str = "") -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
    }


def child_thread_id(parent_thread_id: str, suffix: str) -> str:
    base = parent_thread_id.strip() or "workflow"
    return f"{base}:{suffix}"


def workflow_persistence_mode() -> str:
    return _CHECKPOINTER_MODE


async def get_checkpointer() -> Any:
    global _CHECKPOINTER, _CHECKPOINTER_CTX, _CHECKPOINTER_MODE

    if _CHECKPOINTER is not None:
        return _CHECKPOINTER

    async with _CHECKPOINTER_LOCK:
        if _CHECKPOINTER is not None:
            return _CHECKPOINTER

        db_uri = (
            os.getenv("LANGGRAPH_POSTGRES_URL")
            or os.getenv("NEON_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or ""
        ).strip()

        if db_uri:
            try:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                ctx = AsyncPostgresSaver.from_conn_string(db_uri)
                saver = await ctx.__aenter__()
                if os.getenv("LANGGRAPH_AUTO_SETUP", "1").lower() not in {"0", "false", "no"}:
                    await saver.setup()

                _CHECKPOINTER = saver
                _CHECKPOINTER_CTX = ctx
                _CHECKPOINTER_MODE = "postgres"
                logger.info("workflow_runtime.checkpointer_ready mode=postgres")
                return _CHECKPOINTER
            except Exception as exc:
                logger.warning(
                    "workflow_runtime.postgres_unavailable falling_back=memory error=%s",
                    exc,
                )

        _CHECKPOINTER = InMemorySaver()
        _CHECKPOINTER_MODE = "memory"
        logger.info("workflow_runtime.checkpointer_ready mode=memory")
        return _CHECKPOINTER
