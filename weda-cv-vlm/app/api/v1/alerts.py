"""GET /v1/alerts/recent + SSE /v1/alerts/stream — live alert feed."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

try:
    from sse_starlette.sse import EventSourceResponse
    _HAS_SSE = True
except ImportError:
    _HAS_SSE = False

from app.api.v1._deps import PipelineHandle, require_handle
from app.modules.alert_bus import get_bus

router = APIRouter(prefix="/alerts", tags=["alerts"])


class AlertItem(BaseModel):
    id: str = Field(..., description="Stable id for this alert (12-char hex). Cross-reference with audit / telemetry.")
    ts: float = Field(..., description="Wall-clock epoch seconds when the alert fired.")
    monotonic_ts: float = Field(..., description="`time.monotonic()` at publish time. Use for ordering inside a process; do not compare across reboots.")
    severity: str = Field(..., description='One of `"info" | "warn" | "critical"`. Critical means imminent safety risk.')
    source: str = Field(..., description='Subsystem that emitted the alert: `"predictive-safety" | "fact-checker" | "alarm-engine" | "safety-distance" | "operator-override" | ...`.')
    message: str = Field(..., description="Human-readable summary suitable for UI display.")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Source-specific structured data (track ids, ETA, distances, etc.).")
    audit_id: Optional[str] = Field(None, description="NemoClaw audit-log id when the alert was routed through the agent; null for locally-published alerts.")
    replay_anchor: Optional[float] = Field(None, description="Optional epoch timestamp for `POST /v1/replay?anchor_ts=...` jump-to-context.")


class AlertsRecentResponse(BaseModel):
    alerts: List[AlertItem] = Field(..., description="Recent alerts, newest first. Capped at the requested `limit`.")


@router.get(
    "/recent",
    response_model=AlertsRecentResponse,
    summary="Recent alerts, newest first",
    description=(
        "Snapshot of the last N alerts from the in-process ring buffer "
        "(capped at 200). Optional `min_severity` filter: "
        "`info` | `warn` | `critical`."
    ),
)
async def alerts_recent(
    limit: int = 50,
    min_severity: Optional[str] = None,
    handle: PipelineHandle = Depends(require_handle),
) -> AlertsRecentResponse:
    bus = get_bus()
    return AlertsRecentResponse(
        alerts=[AlertItem(**a.to_dict()) for a in bus.recent(limit, min_severity)],
    )


@router.get(
    "/stream",
    summary="Live alert stream (SSE)",
    description=(
        "Event-driven SSE feed; one `alert` event per published Alert. "
        "Each event's `data` field is a JSON-encoded AlertItem (same "
        "shape as `/v1/alerts/recent[].alerts[]`). Subscriber queue is "
        "bounded at 200 — slow consumers see drops rather than blocking "
        "publishers."
    ),
    include_in_schema=False,
    responses={
        200: {
            "description": "SSE stream. Each event is `alert` with a JSON-encoded AlertItem in `data`.",
            "content": {"text/event-stream": {}},
        }
    },
)
async def alerts_stream():
    if not _HAS_SSE:
        return {"error": "sse-starlette not installed"}
    bus = get_bus()
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    def _enqueue(alert):
        try:
            queue.put_nowait(alert)
        except asyncio.QueueFull:
            pass

    unsub = bus.subscribe(_enqueue)

    async def _gen():
        try:
            while True:
                a = await queue.get()
                yield {"event": "alert", "data": json.dumps(a.to_dict())}
        except asyncio.CancelledError:
            return
        finally:
            unsub()

    return EventSourceResponse(_gen())
