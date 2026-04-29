"""
Postgres-backed job progress tracking for async Blender/Manim workflows.

This complements LangGraph checkpoints:
  - LangGraph keeps durable workflow state per thread.
  - This module keeps poll-friendly job progress summaries and event history.

Progress updates are written as structured events, not a fixed enum ladder, so
workflow nodes can report precise stage messages that reflect their actual work.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Json
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

_POOL: AsyncConnectionPool | None = None
_POOL_LOCK = asyncio.Lock()
_SETUP_DONE = False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _db_uri() -> str:
    return (
        os.getenv("LANGGRAPH_POSTGRES_URL")
        or os.getenv("NEON_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or ""
    ).strip()


async def _ensure_pool() -> AsyncConnectionPool | None:
    global _POOL, _SETUP_DONE

    if _POOL is not None:
        return _POOL

    db_uri = _db_uri()
    if not db_uri:
        return None

    async with _POOL_LOCK:
        if _POOL is None:
            _POOL = AsyncConnectionPool(
                conninfo=db_uri,
                min_size=1,
                max_size=int(os.getenv("JOB_PROGRESS_DB_POOL_SIZE", "4")),
                open=False,
                kwargs={
                    "autocommit": True,
                    "row_factory": dict_row,
                },
            )
            await _POOL.open()

        if not _SETUP_DONE:
            async with _POOL.connection() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workflow_job_status (
                        job_id TEXT PRIMARY KEY,
                        workflow_thread_id TEXT NOT NULL,
                        tool TEXT NOT NULL,
                        state TEXT NOT NULL,
                        stage TEXT NOT NULL DEFAULT '',
                        message TEXT NOT NULL DEFAULT '',
                        details JSONB NOT NULL DEFAULT '{}'::jsonb,
                        result JSONB,
                        error TEXT NOT NULL DEFAULT '',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        started_at TIMESTAMPTZ,
                        finished_at TIMESTAMPTZ,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workflow_job_events (
                        id BIGSERIAL PRIMARY KEY,
                        job_id TEXT NOT NULL,
                        workflow_thread_id TEXT NOT NULL,
                        tool TEXT NOT NULL,
                        state TEXT NOT NULL,
                        stage TEXT NOT NULL DEFAULT '',
                        message TEXT NOT NULL DEFAULT '',
                        details JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_workflow_job_status_updated_at
                    ON workflow_job_status (updated_at DESC)
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_workflow_job_status_thread
                    ON workflow_job_status (workflow_thread_id)
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_workflow_job_events_job_id
                    ON workflow_job_events (job_id, id DESC)
                    """
                )

            _SETUP_DONE = True
            logger.info("progress_store.ready backend=postgres")

    return _POOL


def progress_payload(
    *,
    stage: str,
    message: str,
    state: str,
    details: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    event = {
        "stage": stage,
        "message": message,
        "state": state,
        "details": details or {},
        "timestamp": _utc_now().isoformat(),
    }
    history = list(events or [])
    history.append(event)
    return {
        "progress_stage": stage,
        "progress_message": message,
        "progress_state": state,
        "progress_details": details or {},
        "progress_updated_at": event["timestamp"],
        "progress_events": history[-30:],
    }


async def report_workflow_stage(
    state: dict[str, Any],
    *,
    tool: str,
    stage: str,
    message: str,
    details: dict[str, Any] | None = None,
    job_state: str = "running",
) -> dict[str, Any]:
    job_id = str(state.get("job_id") or state.get("workflow_thread_id") or "").strip()
    workflow_thread_id = str(state.get("workflow_thread_id") or job_id).strip()
    payload = progress_payload(
        stage=stage,
        message=message,
        state=job_state,
        details=details,
        events=state.get("progress_events"),
    )
    if job_id:
        await record_job_progress(
            job_id=job_id,
            workflow_thread_id=workflow_thread_id or job_id,
            tool=tool,
            state=job_state,
            stage=stage,
            message=message,
            details=details,
        )
    return {**state, **payload}


async def record_job_progress(
    *,
    job_id: str,
    workflow_thread_id: str,
    tool: str,
    state: str,
    stage: str,
    message: str,
    details: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    error: str = "",
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> None:
    if not job_id.strip():
        return
    pool = await _ensure_pool()
    if pool is None:
        return

    details = details or {}
    now = _utc_now()
    try:
        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO workflow_job_status (
                    job_id,
                    workflow_thread_id,
                    tool,
                    state,
                    stage,
                    message,
                    details,
                    result,
                    error,
                    created_at,
                    started_at,
                    finished_at,
                    updated_at
                )
                VALUES (
                    %(job_id)s,
                    %(workflow_thread_id)s,
                    %(tool)s,
                    %(state)s,
                    %(stage)s,
                    %(message)s,
                    %(details)s,
                    %(result)s,
                    %(error)s,
                    %(created_at)s,
                    %(started_at)s,
                    %(finished_at)s,
                    %(updated_at)s
                )
                ON CONFLICT (job_id) DO UPDATE SET
                    workflow_thread_id = EXCLUDED.workflow_thread_id,
                    tool = EXCLUDED.tool,
                    state = EXCLUDED.state,
                    stage = EXCLUDED.stage,
                    message = EXCLUDED.message,
                    details = EXCLUDED.details,
                    result = COALESCE(EXCLUDED.result, workflow_job_status.result),
                    error = EXCLUDED.error,
                    started_at = COALESCE(workflow_job_status.started_at, EXCLUDED.started_at),
                    finished_at = COALESCE(EXCLUDED.finished_at, workflow_job_status.finished_at),
                    updated_at = EXCLUDED.updated_at
                """,
                {
                    "job_id": job_id,
                    "workflow_thread_id": workflow_thread_id,
                    "tool": tool,
                    "state": state,
                    "stage": stage,
                    "message": message,
                    "details": Json(details),
                    "result": Json(result) if result is not None else None,
                    "error": error,
                    "created_at": now,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "updated_at": now,
                },
            )
            await conn.execute(
                """
                INSERT INTO workflow_job_events (
                    job_id,
                    workflow_thread_id,
                    tool,
                    state,
                    stage,
                    message,
                    details,
                    created_at
                )
                VALUES (
                    %(job_id)s,
                    %(workflow_thread_id)s,
                    %(tool)s,
                    %(state)s,
                    %(stage)s,
                    %(message)s,
                    %(details)s,
                    %(created_at)s
                )
                """,
                {
                    "job_id": job_id,
                    "workflow_thread_id": workflow_thread_id,
                    "tool": tool,
                    "state": state,
                    "stage": stage,
                    "message": message,
                    "details": Json(details),
                    "created_at": now,
                },
            )
    except Exception as exc:
        logger.warning("progress_store.write_failed job_id=%s error=%s", job_id, exc)


async def get_job_progress(job_id: str, event_limit: int = 30) -> dict[str, Any] | None:
    pool = await _ensure_pool()
    if pool is None:
        return None

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT
                        job_id,
                        workflow_thread_id,
                        tool,
                        state,
                        stage,
                        message,
                        details,
                        result,
                        error,
                        created_at,
                        started_at,
                        finished_at,
                        updated_at
                    FROM workflow_job_status
                    WHERE job_id = %s
                    """,
                    (job_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    return None

                await cur.execute(
                    """
                    SELECT
                        state,
                        stage,
                        message,
                        details,
                        created_at
                    FROM workflow_job_events
                    WHERE job_id = %s
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (job_id, event_limit),
                )
                event_rows = await cur.fetchall()

        events = [
            {
                "state": item["state"],
                "stage": item["stage"],
                "message": item["message"],
                "details": item["details"] or {},
                "timestamp": item["created_at"].isoformat() if item["created_at"] else "",
            }
            for item in reversed(event_rows or [])
        ]
        return {
            "job_id": row["job_id"],
            "workflow_thread_id": row["workflow_thread_id"],
            "tool": row["tool"],
            "progress": {
                "state": row["state"],
                "stage": row["stage"],
                "message": row["message"],
                "details": row["details"] or {},
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else "",
            },
            "result": row["result"],
            "error": row["error"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else "",
            "started_at": row["started_at"].isoformat() if row["started_at"] else "",
            "finished_at": row["finished_at"].isoformat() if row["finished_at"] else "",
            "progress_events": events,
        }
    except Exception as exc:
        logger.warning("progress_store.read_failed job_id=%s error=%s", job_id, exc)
        return None


async def get_job_progress_by_thread(
    workflow_thread_id: str,
    event_limit: int = 30,
) -> dict[str, Any] | None:
    pool = await _ensure_pool()
    if pool is None:
        return None

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT job_id
                    FROM workflow_job_status
                    WHERE workflow_thread_id = %s
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (workflow_thread_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
        return await get_job_progress(str(row["job_id"]), event_limit=event_limit)
    except Exception as exc:
        logger.warning(
            "progress_store.read_by_thread_failed workflow_thread_id=%s error=%s",
            workflow_thread_id,
            exc,
        )
        return None
