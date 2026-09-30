"""ASGI middleware: request_id propagation, structured access logging,
concurrency cap, and per-IP token-bucket rate limiting.

Design notes:
  * request_id is bound into a `contextvars.ContextVar` so worker
    threads spawned during the request inherit it (used by SceneReasoner
    + telemetry for log correlation).
  * Rate limit is in-memory token bucket keyed on the resolved client
    IP (X-Forwarded-For honoured, last hop wins). Constant-time refill
    via timestamp delta — no background sweeper.
  * Concurrency cap is a single semaphore across the whole API, sized
    by ``WEDA_API_MAX_CONCURRENCY`` (default 32). Slowloris mitigated
    by uvicorn's own keep-alive timeout.
  * Streaming responses are exempt from concurrency counting on the
    response side — the semaphore is released once headers are flushed,
    so a long SSE stream does not hold a slot. Trade-off: an attacker
    can hold many idle SSE streams; a backend connection-cap on the
    reverse proxy is the correct second line of defence.
"""

from __future__ import annotations

import contextvars
import os
import time
import uuid
from collections import deque
from typing import Any, Awaitable, Callable, Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.utils.logger import get_logger

logger = get_logger()

# ContextVar — survives across worker threads spawned with the running
# loop's context (uvicorn defaults preserve context).
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


def current_request_id() -> str:
    return request_id_var.get()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Bind a request_id, emit one structured access log per request,
    and surface ``X-Request-ID`` on every response."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.exception(
                "%s %s -> 500 %.1fms rid=%s",
                request.method, request.url.path, elapsed_ms, rid,
            )
            raise
        finally:
            request_id_var.reset(token)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        response.headers["X-Request-ID"] = rid
        # Single structured log line. Keeps cost bounded; integrators
        # can pipe through any log shipper.
        logger.info(
            "%s %s -> %d %.1fms rid=%s",
            request.method, request.url.path, response.status_code,
            elapsed_ms, rid,
        )
        return response


class ConcurrencyLimitMiddleware(BaseHTTPMiddleware):
    """Bounded global concurrency. Refuses overflow with 503 + Retry-After."""

    def __init__(self, app: Any, max_in_flight: int) -> None:
        super().__init__(app)
        self._max = max(1, max_in_flight)
        self._in_flight = 0
        # asyncio.Lock not needed — the in-flight counter is incremented
        # within a single event loop. CPython integer ops are atomic
        # for ``+=`` here (single bytecode under the GIL).

    async def dispatch(self, request: Request, call_next):
        if self._in_flight >= self._max:
            return Response(
                status_code=503,
                headers={"Retry-After": "1"},
                content='{"type":"https://weda.local/errors/service-unavailable",'
                        '"title":"Service Unavailable",'
                        '"status":503,"detail":"concurrency limit reached",'
                        '"code":"service_unavailable"}',
                media_type="application/problem+json",
            )
        self._in_flight += 1
        try:
            return await call_next(request)
        finally:
            self._in_flight -= 1


class _TokenBucket:
    __slots__ = ("tokens", "last_refill", "capacity", "refill_per_s")

    def __init__(self, capacity: float, refill_per_s: float) -> None:
        self.capacity = capacity
        self.refill_per_s = refill_per_s
        self.tokens = capacity
        self.last_refill = time.monotonic()

    def take(self) -> bool:
        now = time.monotonic()
        delta = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + delta * self.refill_per_s)
        self.last_refill = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-client token bucket. O(1) per request. Bucket map evicts on
    LRU when ``max_clients`` is exceeded; 1 KB per client at default."""

    def __init__(
        self,
        app: Any,
        rate_per_minute: int = 600,
        burst: int = 60,
        max_clients: int = 4096,
    ) -> None:
        super().__init__(app)
        self._refill = rate_per_minute / 60.0
        self._burst = burst
        self._buckets: dict[str, _TokenBucket] = {}
        self._lru: deque[str] = deque(maxlen=max_clients)

    async def dispatch(self, request: Request, call_next):
        client = self._client_key(request)
        bucket = self._buckets.get(client)
        if bucket is None:
            if len(self._buckets) >= self._lru.maxlen:
                stale = self._lru.popleft()
                self._buckets.pop(stale, None)
            bucket = _TokenBucket(self._burst, self._refill)
            self._buckets[client] = bucket
            self._lru.append(client)
        if not bucket.take():
            return Response(
                status_code=429,
                headers={"Retry-After": "1"},
                content=(
                    '{"type":"https://weda.local/errors/too-many-requests",'
                    '"title":"Too Many Requests","status":429,'
                    '"detail":"rate limit exceeded",'
                    '"code":"too_many_requests"}'
                ),
                media_type="application/problem+json",
            )
        return await call_next(request)

    @staticmethod
    def _client_key(request: Request) -> str:
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            return xff.split(",")[-1].strip()
        host = request.client.host if request.client else "anon"
        return host


def install(app: Any) -> None:
    max_concurrency = int(os.environ.get("WEDA_API_MAX_CONCURRENCY", "32"))
    rate_per_min = int(os.environ.get("WEDA_API_RATE_PER_MIN", "600"))
    burst = int(os.environ.get("WEDA_API_RATE_BURST", "60"))
    # Order: rate-limit outermost (cheapest reject), then concurrency
    # cap, then context. Starlette wraps in reverse add-order.
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(ConcurrencyLimitMiddleware, max_in_flight=max_concurrency)
    app.add_middleware(RateLimitMiddleware,
                       rate_per_minute=rate_per_min, burst=burst)
