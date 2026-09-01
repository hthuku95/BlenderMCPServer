import asyncio
import inspect
import json
import logging
import os
import socket
import traceback as _traceback
import uuid
from datetime import datetime, timezone, timedelta
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
    claimed_by: str = ""
    lease_expires_at: str = ""

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


# ---------------------------------------------------------------------------
# Lease / heartbeat (Phase 2 durability)
# ---------------------------------------------------------------------------

_NODE_ID = (socket.gethostname() + "-" + str(os.getpid()))[:80]
_LEASE_MINUTES = float(os.getenv("JOB_LEASE_MINUTES", "15"))
_LEASE_RENEW_SECS = float(os.getenv("JOB_LEASE_RENEW_SECS", "60"))
_JOB_MAX_RECEIVES = int(os.getenv("JOB_MAX_RECEIVES", "3"))
# Failures whose message contains one of these are considered transient and
# eligible for SQS redelivery retry (up to JOB_MAX_RECEIVES receives).
_TRANSIENT_MARKERS = (
    "ReadTimeout", "ConnectError", "ConnectionError", "timed out",
    "unhealthy", "ServiceUnavailable", "503", "502", "429", "throttl",
    "connection reset", "connection closed",
)
# Checked BEFORE the transient scan: errors starting with these are
# permanent regardless of what the traceback happens to contain (e.g. an
# asyncio frame line could otherwise flip a budget-exhaustion into a retry).
_PERMANENT_PREFIXES = (
    "RENDER_BUDGET_EXCEEDED",
    "No handler registered",
    "RENDER_BUDGET",
)


def _iso_in_minutes(minutes: float) -> str:
    return (
        datetime.now(timezone.utc) + timedelta(minutes=minutes)
    ).isoformat()


def _lease_expired(iso_ts: str) -> bool:
    if not iso_ts:
        return True
    try:
        return datetime.now(timezone.utc) >= datetime.fromisoformat(iso_ts)
    except ValueError:
        return True


def _is_transient_failure(error_text: str) -> bool:
    t = (error_text or "").lstrip()
    for prefix in _PERMANENT_PREFIXES:
        if t.startswith(prefix):
            return False
    t = t.lower()
    return any(marker.lower() in t for marker in _TRANSIENT_MARKERS)


def _render_budget_effective() -> float:
    """Mirror react_codegen's RENDER_BUDGET_SECS so JOB_TIMEOUT can be raised
    above it (the budget must fire before the whole-job backstop)."""
    try:
        from tools.react_codegen import _RENDER_BUDGET_SECS as _rb
        v = float(_rb)
        return v if v > 0 else 1200.0
    except Exception:
        return 1200.0


async def _ddb_job_cancelled(job_id: str) -> bool:
    """Cross-node cancel check: cancel() writes CANCELLED to DDB; any node's
    worker sees it here."""
    try:
        st = await _async_get_status(job_id)
        return bool(st and st.state == State.CANCELLED.value)
    except Exception:
        return False


async def _heartbeat_loop(job_id: str, stop: asyncio.Event) -> None:
    """Renew this worker's lease every _LEASE_RENEW_SECS while the job runs.

    Uses a targeted conditional update so a late heartbeat can never clobber
    a terminal state written by the terminal path (or another owner)."""
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=_LEASE_RENEW_SECS)
            return  # stop set
        except asyncio.TimeoutError:
            pass
        try:
            await asyncio.to_thread(
                _renew_lease_sync, job_id, _iso_in_minutes(_LEASE_MINUTES)
            )
        except Exception:
            pass


def _renew_lease_sync(job_id: str, lease_iso: str) -> None:
    _lazy_aws()
    _table.update_item(
        Key={"job_id": job_id},
        UpdateExpression="SET lease_expires_at = :l",
        ExpressionAttributeValues={":l": lease_iso, ":r": State.RUNNING.value, ":n": _NODE_ID},
        ConditionExpression="attribute_exists(job_id) AND #s = :r AND claimed_by = :n",
        ExpressionAttributeNames={"#s": "state"},
    )


def _claim_job_sync(job_id: str, lease_iso: str, started_at: str) -> bool:
    """Atomically claim a job for this worker. Succeeds only if the job is
    NOT already running with a fresh lease (guards against duplicate
    execution when SQS redelivers a message whose visibility expired while
    the original worker is still alive). ISO-8601 UTC timestamps compare
    correctly as strings."""
    _lazy_aws()
    try:
        _table.update_item(
            Key={"job_id": job_id},
            UpdateExpression=(
                "SET #s = :running, claimed_by = :node, "
                "lease_expires_at = :lease, started_at = :started"
            ),
            ConditionExpression=(
                "#s <> :running OR attribute_not_exists(lease_expires_at) "
                "OR lease_expires_at < :now"
            ),
            ExpressionAttributeNames={"#s": "state"},
            ExpressionAttributeValues={
                ":running": State.RUNNING.value,
                ":node": _NODE_ID,
                ":lease": lease_iso,
                ":started": started_at,
                ":now": datetime.now(timezone.utc).isoformat(),
            },
        )
        return True
    except Exception:
        # ConditionalCheckFailedException -> another live worker owns it.
        return False


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
        # Phase 2: one message per worker iteration. Each worker task handles
        # its job end-to-end; JOB_QUEUE_WORKERS workers = that many parallel
        # jobs. Batch-receive + sequential processing caused head-of-line
        # blocking where messages 2..10 could outlive the visibility window.
        MaxNumberOfMessages=1,
        WaitTimeSeconds=5,
        VisibilityTimeout=int(os.getenv("SQS_VISIBILITY_TIMEOUT", "1800")),
        AttributeNames=["All"],  # ApproximateReceiveCount for retry policy
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
    # Phase 2 fix: this used to be a full put_item with a PARTIAL item, which
    # REPLACED the whole job record mid-flight — wiping claimed_by,
    # lease_expires_at, args, error and result (masked only because the
    # terminal path rewrote the full item). A targeted update preserves the
    # rest of the record. "state" and "timestamp" are DDB reserved words ->
    # aliased.
    await asyncio.to_thread(
        _update_progress_sync,
        job_id,
        state,
        stage,
        message,
        json.dumps(details or {}),
        workflow_thread_id or job_id,
        tool,
        started_at or _now(),
    )


def _update_progress_sync(
    job_id: str,
    state: str,
    stage: str,
    message: str,
    details_json: str,
    workflow_thread_id: str,
    tool: str,
    timestamp: str,
) -> None:
    _lazy_aws()
    _table.update_item(
        Key={"job_id": job_id},
        UpdateExpression=(
            "SET #s = :s, stage = :st, message = :m, details = :d, "
            "workflow_thread_id = :w, tool = :t, #ts = :ts"
        ),
        ExpressionAttributeNames={"#s": "state", "#ts": "timestamp"},
        ExpressionAttributeValues={
            ":s": state,
            ":st": stage,
            ":m": message,
            ":d": details_json,
            ":w": workflow_thread_id,
            ":t": tool,
            ":ts": timestamp,
        },
    )


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
    """Phase 2 recovery: a RUNNING job whose lease has expired belongs to a
    dead worker — requeue it (SQS redelivery) instead of the old behavior of
    marking it RECOVERED and losing the work. Fresh leases (another live
    worker, multi-node fleets) are left alone. Retry count is capped so a
    poison job cannot loop forever."""
    logger = logging.getLogger("job_queue")
    try:
        orphans = await asyncio.to_thread(_scan_orphans_sync)
    except Exception:
        logger.warning("orphan scan failed", exc_info=True)
        return
    for o in orphans:
        try:
            if o.state != State.RUNNING.value:
                # QUEUED/PENDING records are left alone: their SQS messages
                # are usually still in flight; blind re-sends would duplicate
                # execution (and resurrect days-old stale rows).
                continue
            if not _lease_expired(o.lease_expires_at):
                logger.info("orphan %s has a fresh lease (%s), leaving it", o.job_id, o.claimed_by)
                continue
            if int(o.recoveries or 0) >= _JOB_MAX_RECEIVES:
                o.state = State.FAILED.value
                o.error = f"permanently failed after {o.recoveries} recovery attempts"
                o.finished_at = _now()
                await _async_put_status(o)
                logger.warning("orphan %s exceeded recovery cap, marked failed", o.job_id)
                continue
            o.state = State.QUEUED.value
            o.error = f"requeued by recovery (previous owner {o.claimed_by or 'unknown'} died or lease expired)"
            o.recoveries = int(o.recoveries or 0) + 1
            o.claimed_by = ""
            o.lease_expires_at = ""
            o.started_at = ""
            await _async_put_status(o)
            await _async_sqs_send(o)
            logger.warning("orphan %s requeued for retry (recoveries=%s)", o.job_id, o.recoveries)
        except Exception:
            logger.warning("orphan recovery failed for %s", o.job_id, exc_info=True)


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
        logger = logging.getLogger("job_queue")
        while True:
            messages: list[dict] = []
            try:
                messages = await _async_sqs_receive()
            except Exception:
                logger.warning("sqs receive failed, retrying in 5s", exc_info=True)
                await asyncio.sleep(5)
                continue
            for msg in messages:
                try:
                    body = json.loads(msg["Body"])
                    status = JobStatus.from_dict(body)
                    receipt_handle = msg["ReceiptHandle"]
                    receive_count = int(
                        (msg.get("Attributes") or {}).get("ApproximateReceiveCount", "1") or "1"
                    )
                    await self._handle_job(status, receipt_handle, receive_count)
                except Exception:
                    # Never let one poison message kill the worker; log loudly
                    # instead of the old silent `except: pass`.
                    logger.warning("worker failed to process message", exc_info=True)
            await asyncio.sleep(1)

    async def _handle_job(self, status: JobStatus, receipt_handle: str, receive_count: int = 1) -> None:
        job_id = status.job_id
        logger = logging.getLogger("job_queue")
        heartbeat_task: Optional[asyncio.Task] = None
        stop_heartbeat = asyncio.Event()
        try:
            # IDEMPOTENCY ON REDELIVERY: if the job already reached a terminal
            # state (e.g. completed by a previous receive whose delete failed,
            # or marked cancelled cross-node), skip reprocessing and delete.
            fresh = await _async_get_status(job_id)
            if fresh and fresh.state in (
                State.COMPLETED.value, State.FAILED.value, State.CANCELLED.value,
            ):
                logger.info("job %s already terminal (%s), skipping redelivery", job_id, fresh.state)
                await _async_sqs_delete_message(receipt_handle)
                return

            args = fresh.args if fresh and fresh.args else (status.args or {})
            if fresh:
                status = fresh
            handler = self._tool_registry.get(status.tool)

            # ATOMIC CLAIM: only one live worker may own a job. If the claim
            # fails, another worker holds a fresh lease (our message was a
            # visibility-expiry redelivery) — skip and drop our copy.
            lease_iso = _iso_in_minutes(_LEASE_MINUTES) if _LEASE_MINUTES > 0 else ""
            if _LEASE_MINUTES > 0:
                if not await asyncio.to_thread(
                    _claim_job_sync, job_id, lease_iso, _now()
                ):
                    logger.warning(
                        "job %s claim rejected (fresh lease held by %s), skipping",
                        job_id, (fresh.claimed_by if fresh else "") or "unknown",
                    )
                    await _async_sqs_delete_message(receipt_handle)
                    return
                status.state = State.RUNNING.value
                status.started_at = _now()
                status.result = None
                status.claimed_by = _NODE_ID
                status.lease_expires_at = lease_iso
                await _async_put_status(status)
                heartbeat_task = asyncio.create_task(
                    _heartbeat_loop(job_id, stop_heartbeat)
                )
            else:
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
            # The whole-job timeout is a backstop ONLY; the render budget
            # (RENDER_BUDGET_SECS, enforced between codegen turns) must fire
            # first so jobs fail cleanly instead of being killed mid-render.
            if _JOB_TIMEOUT <= _render_budget_effective():
                _JOB_TIMEOUT = int(_render_budget_effective() + 300)

            if handler is None:
                raise ValueError(f"No handler registered for tool {status.tool}")

            if await _ddb_job_cancelled(job_id):
                status.state = State.CANCELLED.value
                status.finished_at = _now()
                _put_status_sync(status)
                asyncio.create_task(_record_cancelled_progress_async(status))
                await _async_sqs_delete_message(receipt_handle)
                return

            handler_args = self._filter_handler_args(handler, args)
            try:
                result = await asyncio.wait_for(
                    handler(**handler_args), timeout=_JOB_TIMEOUT
                )
            except asyncio.TimeoutError as te:
                # Give the message text so the transient classifier sees it.
                raise RuntimeError(f"Job timed out after {_JOB_TIMEOUT}s") from te

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
        except Exception as e:
            error_text = _format_failure_error(e)
            from tools.react_codegen import JobCancelled
            if isinstance(e, JobCancelled):
                status.state = State.CANCELLED.value
                status.error = error_text
                status.finished_at = _now()
                _put_status_sync(status)
                logger.warning("job %s cancelled between turns", job_id)
            elif (
                _is_transient_failure(error_text)
                and receive_count < _JOB_MAX_RECEIVES
                and status.recoveries < _JOB_MAX_RECEIVES
            ):
                # TRANSIENT: give the message back to SQS (don't delete) so it
                # redelivers after the visibility window. DDB state goes back
                # to queued; the redelivered receive reprocesses idempotently.
                status.state = State.QUEUED.value
                status.error = error_text
                status.recoveries = int(status.recoveries or 0) + 1
                status.claimed_by = ""
                status.lease_expires_at = ""
                status.finished_at = ""
                _put_status_sync(status)
                logger.warning(
                    "job %s transient failure (receive %s/%s), requeueing: %s",
                    job_id, receive_count, _JOB_MAX_RECEIVES, str(e)[:300],
                )
                return  # deliberately do NOT delete the SQS message
            else:
                status.state = State.FAILED.value
                status.error = error_text
                status.finished_at = _now()
                _put_status_sync(status)
        finally:
            stop_heartbeat.set()
            if heartbeat_task is not None:
                try:
                    await asyncio.wait_for(heartbeat_task, timeout=5)
                except Exception:
                    heartbeat_task.cancel()

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
