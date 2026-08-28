import asyncio
import inspect
import json
import os
import traceback as _traceback
import uuid
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Any, Awaitable, Callable, Optional, Dict
from decimal import Decimal

import boto3
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# AWS clients (lazy — first call to any SQS/DDB function creates them)
# ---------------------------------------------------------------------------

def _aws_session():
    from boto3 import Session
    return Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION", "eu-north-1"),
    )


_sqs = None
_table = None


def _lazy_aws():
    global _sqs, _table
    if _sqs is not None:
        return
    session = _aws_session()
    _sqs = session.client("sqs")
    _table = session.resource("dynamodb").Table(os.environ["DYNAMODB_TABLE"])


SQS_QUEUE_URL = os.environ["SQS_QUEUE_URL"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_job_id() -> str:
    return uuid.uuid4().hex[:24]


def _format_failure_error(exc: Exception, limit_chars: int = 4000) -> str:
    """Return a diagnosable error string: the exception message plus the
    (trimmed) traceback, so a failure like 'agentic blender codegen failed
    after N turns' carries the underlying cause instead of just str(e)."""
    msg = str(exc) or exc.__class__.__name__
    tb = _traceback.format_exc().strip()
    if tb:
        # Keep the exception line + the most useful stack frames.
        lines = tb.splitlines()
        header = lines[0] if lines else ""
        tail = "\n".join(lines[-6:])
        detail = f"{header}\n[trimmed {max(len(lines) - 6, 0)} lines]\n{tail}"
        full = f"{msg}\n--- traceback ---\n{detail}"
    else:
        full = msg
    if len(full) > limit_chars:
        full = full[:limit_chars] + "\n...[truncated]"
    return full


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class State(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECOVERED = "recovered"


# ---------------------------------------------------------------------------
# Job status data model (stored in DynamoDB)
# ---------------------------------------------------------------------------

@dataclass
class JobStatus:
    job_id: str
    tool: str = ""
    state: str = State.PENDING.value
    args: dict = field(default_factory=dict)
    result: Any = None
    error: str = ""
    created_at: str = ""
    queued_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    recoveries: int = 0
    workflow_thread_id: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "JobStatus":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Async wrappers for SQS + DynamoDB (called from asyncio worker tasks)
# ---------------------------------------------------------------------------

def _put_status_sync(status: JobStatus) -> None:
    _lazy_aws()
    item = status.to_dict()
    if isinstance(item.get("result"), dict) and item["result"] is not None:
        item["result"] = json.dumps(item["result"])
    item = _convert_to_ddb_types(item)
    _table.put_item(Item=item)


async def _async_put_status(status: JobStatus) -> None:
    await asyncio.to_thread(_put_status_sync, status)


def _get_status_sync(job_id: str) -> Optional[JobStatus]:
    _lazy_aws()
    resp = _table.get_item(Key={"job_id": job_id})
    item = resp.get("Item")
    if item is None:
        return None
    # Convert DynamoDB Decimals to native types
    item = _convert_decimals(item)
    if isinstance(item.get("result"), str):
        try:
            item["result"] = json.loads(item["result"])
        except (json.JSONDecodeError, TypeError):
            pass
    return JobStatus.from_dict(item)


def _convert_decimals(obj):
    """Recursively convert DynamoDB Decimal types to int/float."""
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, dict):
        return {k: _convert_decimals(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_decimals(v) for v in obj]
    return obj


async def _async_get_status(job_id: str) -> Optional[JobStatus]:
    return await asyncio.to_thread(_get_status_sync, job_id)


def _convert_to_ddb_types(obj):
    """Convert floats to Decimal for DynamoDB compatibility."""
    from decimal import Decimal
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _convert_to_ddb_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_to_ddb_types(v) for v in obj]
    return obj

def _scan_orphans_sync() -> list[JobStatus]:
    _lazy_aws()
    resp = _table.scan(
        FilterExpression="#s IN (:p, :r)",
        ExpressionAttributeNames={"#s": "state"},
        ExpressionAttributeValues={
            ":p": State.PENDING.value,
            ":r": State.RUNNING.value,
        },
    )
    return [JobStatus.from_dict(i) for i in resp.get("Items", [])]


def _sqs_send_sync(status: JobStatus) -> str:
    _lazy_aws()
    resp = _sqs.send_message(
        QueueUrl=SQS_QUEUE_URL,
        MessageBody=json.dumps(status.to_dict(), default=str),

    )
    return resp["MessageId"]


async def _async_sqs_send(status: JobStatus) -> str:
    return await asyncio.to_thread(_sqs_send_sync, status)


def _sqs_receive_sync() -> list[dict]:
    _lazy_aws()
    resp = _sqs.receive_message(
        QueueUrl=SQS_QUEUE_URL,
        MaxNumberOfMessages=10,
        WaitTimeSeconds=5,
        VisibilityTimeout=int(os.getenv("SQS_VISIBILITY_TIMEOUT", "1800")),
    )
    return resp.get("Messages", [])


async def _async_sqs_receive() -> list[dict]:
    return await asyncio.to_thread(_sqs_receive_sync)


def _sqs_delete_sync(receipt_handle: str) -> None:
    _lazy_aws()
    _sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt_handle)


async def _async_sqs_delete_message(receipt_handle: str) -> None:
    await asyncio.to_thread(_sqs_delete_sync, receipt_handle)


# ---------------------------------------------------------------------------
# Progress recorder
# ---------------------------------------------------------------------------

async def record_job_progress(
    job_id: str,
    workflow_thread_id: str,
    tool: str,
    state: str,
    stage: str,
    message: str = "",
    details: dict = None,
    started_at: str = "",
) -> None:
    _lazy_aws()
    item = {
        "job_id": job_id,
        "workflow_thread_id": workflow_thread_id or job_id,
        "tool": tool,
        "state": state,
        "stage": stage,
        "message": message,
        "details": json.dumps(details or {}),
        "started_at": started_at or _now(),
        "timestamp": _now(),
    }
    await asyncio.to_thread(_table.put_item, Item=item)


async def _record_cancelled_progress_async(status: JobStatus) -> None:
    await record_job_progress(
        job_id=status.job_id,
        workflow_thread_id=status.workflow_thread_id or status.job_id,
        tool=status.tool,
        state=State.CANCELLED.value,
        stage="dispatch",
        message="Job was cancelled before execution",
        details={},
    )


async def _recover_orphans_async(queue: "JobQueue") -> None:
    try:
        orphans = await asyncio.to_thread(_scan_orphans_sync)
        for o in orphans:
            error_msg = f"orphan recovered on restart (was {o.state})"
            o.state = State.RECOVERED.value
            o.error = error_msg
            o.finished_at = _now()
            await _async_put_status(o)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# JobQueue — the core class
# ---------------------------------------------------------------------------

class JobQueue:
    def __init__(self) -> None:
        self._tool_registry: Dict[str, Callable] = {}
        self._workers: list[asyncio.Task] = []
        self._job_states: Dict[str, JobStatus] = {}
        self._cancelled: set[str] = set()

    def register(self, name: str, handler: Callable) -> None:
        self._tool_registry[name] = handler

    def _filter_handler_args(self, handler: Callable, args: dict) -> dict:
        sig = inspect.signature(handler)
        return {k: v for k, v in args.items() if k in sig.parameters}

    async def submit(
        self,
        tool: str,
        args: dict = None,
        workflow_thread_id: str = "",
    ) -> str:
        job_id = _make_job_id()
        now = _now()
        status = JobStatus(
            job_id=job_id,
            tool=tool,
            state=State.QUEUED.value,
            args=args or {},
            created_at=now,
            queued_at=now,
            workflow_thread_id=workflow_thread_id or job_id,
        )
        self._job_states[job_id] = status
        await _async_put_status(status)
        await _async_sqs_send(status)
        return job_id

    async def get_status(self, job_id: str) -> Optional[JobStatus]:
        if job_id in self._job_states:
            return self._job_states[job_id]
        return await _async_get_status(job_id)

    def get(self, job_id: str, force_refresh: bool = False) -> Optional[JobStatus]:
        if not force_refresh and job_id in self._job_states:
            cached = self._job_states[job_id]
            if cached.state in (State.PENDING.value, State.QUEUED.value):
                fresh = _get_status_sync(job_id)
                if fresh and fresh.state != cached.state:
                    self._job_states[job_id] = fresh
                    return fresh
            return cached
        fresh = _get_status_sync(job_id)
        if fresh:
            self._job_states[job_id] = fresh
        return fresh

    def is_cancelled(self, job_id: str) -> bool:
        return job_id in self._cancelled

    async def cancel(self, job_id: str) -> None:
        self._cancelled.add(job_id)
        status = await self.get_status(job_id)
        if status and status.state in (State.PENDING.value, State.QUEUED.value, State.RUNNING.value):
            status.state = State.CANCELLED.value
            status.finished_at = _now()
            await _async_put_status(status)

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    def start_workers(self, count: int = 3) -> None:
        if self._workers:
            return
        for _ in range(count):
            task = asyncio.create_task(self._worker())
            self._workers.append(task)

    async def _worker(self) -> None:
        while True:
            try:
                messages = await _async_sqs_receive()
                for msg in messages:
                    body = json.loads(msg["Body"])
                    status = JobStatus.from_dict(body)
                    receipt_handle = msg["ReceiptHandle"]
                    await self._handle_job(status, receipt_handle)
            except Exception:
                pass
            await asyncio.sleep(1)

    async def _handle_job(self, status: JobStatus, receipt_handle: str) -> None:
        job_id = status.job_id
        try:
            if self.is_cancelled(job_id):
                status.state = State.CANCELLED.value
                status.finished_at = _now()
                _put_status_sync(status)
                asyncio.create_task(_record_cancelled_progress_async(status))
                await _async_sqs_delete_message(receipt_handle)
                return

            args = status.args or {}
            handler = self._tool_registry.get(status.tool)
            open("/tmp/jq_debug.log","a").write(f"JQ_DEBUG job_id={job_id} tool={status.tool} handler_found={handler is not None} registry_keys={list(self._tool_registry.keys())}\n")

            status.state = State.RUNNING.value
            status.started_at = _now()
            status.result = None
            await _async_put_status(status)

            await record_job_progress(
                job_id=job_id,
                workflow_thread_id=status.workflow_thread_id or job_id,
                tool=status.tool,
                state=State.RUNNING.value,
                stage="dispatch",
                message=f"Dispatching {status.tool} handler",
                details={},
                started_at=_now(),
            )

            _JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT_SECS", "1500"))

            if handler is None:
                raise ValueError(f"No handler registered for tool {status.tool}")

            if self.is_cancelled(job_id):
                status.state = State.CANCELLED.value
                status.finished_at = _now()
                _put_status_sync(status)
                asyncio.create_task(_record_cancelled_progress_async(status))
                await _async_sqs_delete_message(receipt_handle)
                return

            handler_args = self._filter_handler_args(handler, args)
            result = await asyncio.wait_for(
                handler(**handler_args), timeout=_JOB_TIMEOUT
            )

            if self.is_cancelled(job_id):
                status.state = State.CANCELLED.value
                status.finished_at = _now()
                _put_status_sync(status)
                await _async_sqs_delete_message(receipt_handle)
                return

            if result is None:
                result = {}

            status.state = State.COMPLETED.value
            status.result = result
            status.finished_at = _now()
            if result is not None and isinstance(result, dict):
                status.result = result.get("data", result)
            await record_job_progress(
                job_id=job_id,
                workflow_thread_id=status.workflow_thread_id or job_id,
                tool=status.tool,
                state=State.COMPLETED.value,
                stage="done",
                message=f"Job {job_id} completed",
                details={"result": result} if isinstance(result, dict) else {},
            )
            _put_status_sync(status)

        except asyncio.CancelledError:
            status.state = State.CANCELLED.value
            status.finished_at = _now()
            _put_status_sync(status)
        except asyncio.TimeoutError:
            status.state = State.FAILED.value
            status.error = f"Job timed out after {_JOB_TIMEOUT}s"
            status.finished_at = _now()
            _put_status_sync(status)
        except Exception as e:
            status.state = State.FAILED.value
            status.error = _format_failure_error(e)
            status.finished_at = _now()
            _put_status_sync(status)

        try:
            await _async_sqs_delete_message(receipt_handle)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Integration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _json_ready(obj: Any) -> Any:
        """Recursively coerce boto3 Decimal values to int/float for JSON."""
        if isinstance(obj, Decimal):
            return int(obj) if obj == obj.to_integral_value() else float(obj)
        if isinstance(obj, dict):
            return {k: JobQueue._json_ready(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [JobQueue._json_ready(v) for v in obj]
        return obj

    async def list_jobs(
        self,
        limit: int = 50,
        status_filter: Optional[str] = None,
    ) -> list[dict]:
        _lazy_aws()
        kwargs: dict = {}
        if status_filter:
            kwargs["FilterExpression"] = "#s = :s"
            kwargs["ExpressionAttributeNames"] = {"#s": "state"}
            kwargs["ExpressionAttributeValues"] = {":s": status_filter}
        resp = await asyncio.to_thread(
            lambda: _table.scan(Limit=limit, **kwargs)
        )
        items = resp.get("Items", [])
        for item in items:
            if isinstance(item.get("result"), str):
                try:
                    item["result"] = json.loads(item["result"])
                except (json.JSONDecodeError, TypeError):
                    pass
        return [self._json_ready(i) for i in items]

    async def get_pending_run_counts(self) -> dict:
        _lazy_aws()
        resp = await asyncio.to_thread(
            lambda: _table.scan(
                FilterExpression="#s IN (:p, :r)",
                ExpressionAttributeNames={"#s": "state"},
                ExpressionAttributeValues={
                    ":p": State.PENDING.value,
                    ":r": State.RUNNING.value,
                },
            )
        )
        items = resp.get("Items", [])
        pending = sum(1 for i in items if i.get("state") == State.PENDING.value)
        running = sum(1 for i in items if i.get("state") == State.RUNNING.value)
        return {"pending": pending, "running": running}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

queue = JobQueue()


# ---------------------------------------------------------------------------
# Server lifecycle hooks (called by server.py)
# ---------------------------------------------------------------------------

async def start_job_workers(worker_count: int = 3) -> None:
    import sys
    sys.stderr.write("SJW_START\n"); sys.stderr.flush()
    await _recover_orphans_async(queue)
    sys.stderr.write("SJW_RECOVER_DONE\n"); sys.stderr.flush()
    queue.start_workers(worker_count)
    sys.stderr.write(f"SJW_WORKERS_STARTED count={len(queue._workers)}\n"); sys.stderr.flush()
