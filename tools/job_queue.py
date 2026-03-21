"""
Async Job Queue — Phase 5

Allows callers to submit long-running Blender/Manim jobs asynchronously and
poll for results.  The queue runs entirely in-process using asyncio — no Redis
or external broker required.

Usage (from server.py):
    from tools.job_queue import queue, JobStatus

    job_id = await queue.submit("blender_generate_scene", {"prompt": "...", ...})
    # ... later ...
    status = queue.get(job_id)   # returns JobStatus

REST endpoints (wired in server.py):
    POST /api/jobs           — submit a job, returns {"job_id": str}
    GET  /api/jobs/{job_id}  — poll job status
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine


class State(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


@dataclass
class JobStatus:
    job_id:    str
    tool:      str
    state:     State          = State.PENDING
    result:    dict | None    = None
    error:     str            = ""
    created_at: str           = field(default_factory=lambda: _now())
    started_at: str           = ""
    finished_at: str          = ""

    def to_dict(self) -> dict:
        return {
            "job_id":      self.job_id,
            "tool":        self.tool,
            "state":       self.state.value,
            "result":      self.result,
            "error":       self.error,
            "created_at":  self.created_at,
            "started_at":  self.started_at,
            "finished_at": self.finished_at,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------

class JobQueue:
    """
    In-process async job queue.

    Workers run concurrently up to `max_workers` at a time (default 2 —
    Render Standard has 1 vCPU but Blender renders are I/O-heavy).
    """

    def __init__(self, max_workers: int = 2):
        self._jobs: dict[str, JobStatus] = {}
        self._pending: asyncio.Queue[str] = asyncio.Queue()
        self._max_workers = max_workers
        self._tool_registry: dict[str, Callable[..., Coroutine[Any, Any, dict]]] = {}
        self._started = False

    def register(self, tool_name: str, fn: Callable[..., Coroutine[Any, Any, dict]]) -> None:
        """Register a coroutine function as the handler for a tool name."""
        self._tool_registry[tool_name] = fn

    async def submit(self, tool_name: str, args: dict) -> str:
        """Submit a job and return its job_id.  Starts worker loop on first call."""
        if not self._started:
            await self._start_workers()

        job_id = str(uuid.uuid4())
        status = JobStatus(job_id=job_id, tool=tool_name)
        status.result = args  # store args temporarily (overwritten on completion)
        self._jobs[job_id] = status

        # Store args alongside job so worker can retrieve them
        self._jobs[job_id]._args = args  # type: ignore[attr-defined]
        await self._pending.put(job_id)
        return job_id

    def get(self, job_id: str) -> JobStatus | None:
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 100) -> list[dict]:
        """Return the most recent `limit` jobs, newest first."""
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs[:limit]]

    async def _start_workers(self) -> None:
        self._started = True
        for _ in range(self._max_workers):
            asyncio.create_task(self._worker())

    async def _worker(self) -> None:
        while True:
            job_id = await self._pending.get()
            status = self._jobs.get(job_id)
            if status is None:
                self._pending.task_done()
                continue

            args = getattr(status, "_args", {})
            handler = self._tool_registry.get(status.tool)

            status.state = State.RUNNING
            status.started_at = _now()
            status.result = None  # clear the temp args

            if handler is None:
                status.state = State.FAILED
                status.error = f"No handler registered for tool '{status.tool}'"
                status.finished_at = _now()
                self._pending.task_done()
                continue

            try:
                result = await handler(**args)
                status.state = State.COMPLETED
                status.result = result
            except Exception as exc:
                status.state = State.FAILED
                status.error = str(exc)
            finally:
                status.finished_at = _now()
                self._pending.task_done()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

queue = JobQueue(max_workers=int(__import__("os").getenv("JOB_QUEUE_WORKERS", "2")))
