"""
Token-bucket Rate Limiter — Phase 5

Protects the BlenderMCPServer REST API from abuse.  Each authenticated caller
(identified by API key or IP address) gets a token bucket that refills at a
configurable rate.

Usage (in server.py):
    from tools.rate_limiter import RateLimiter

    limiter = RateLimiter(rate=10, capacity=20)   # 10 req/s, burst up to 20

    async def my_endpoint(request):
        key = request.headers.get("Authorization") or request.client.host
        if not limiter.allow(key):
            return JSONResponse({"error": "Rate limit exceeded"}, status_code=429)
        ...

Configuration via env vars:
    RATE_LIMIT_RPS       — tokens refilled per second (default 5)
    RATE_LIMIT_CAPACITY  — max burst size (default 10)
    RATE_LIMIT_ENABLED   — "true" | "false" (default "true")
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class _Bucket:
    tokens: float
    last_refill: float = field(default_factory=time.monotonic)


class RateLimiter:
    """
    Token-bucket limiter.

    Thread-safe (Lock), compatible with async handlers since `allow()` is
    non-blocking and returns instantly.
    """

    def __init__(
        self,
        rate: float | None = None,      # tokens per second
        capacity: float | None = None,  # max tokens (burst ceiling)
    ):
        self._rate     = rate     or float(os.getenv("RATE_LIMIT_RPS", "5"))
        self._capacity = capacity or float(os.getenv("RATE_LIMIT_CAPACITY", "10"))
        self._enabled  = os.getenv("RATE_LIMIT_ENABLED", "true").lower() != "false"
        self._buckets: dict[str, _Bucket] = {}
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        """
        Consume one token for `key`.  Returns True if allowed, False if rate-limited.

        Always returns True when the limiter is disabled.
        """
        if not self._enabled:
            return True

        now = time.monotonic()
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = _Bucket(tokens=self._capacity, last_refill=now)

            bucket = self._buckets[key]
            elapsed = now - bucket.last_refill
            bucket.last_refill = now

            # Refill
            bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._rate)

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True
            return False

    def reset(self, key: str) -> None:
        """Reset the bucket for a given key (useful in tests)."""
        with self._lock:
            self._buckets.pop(key, None)

    def stats(self) -> dict:
        """Return current bucket stats (for monitoring/health endpoint)."""
        with self._lock:
            return {
                "rate_rps":    self._rate,
                "capacity":    self._capacity,
                "enabled":     self._enabled,
                "active_keys": len(self._buckets),
            }


# ---------------------------------------------------------------------------
# Singleton (shared across all request handlers in server.py)
# ---------------------------------------------------------------------------

limiter = RateLimiter()
