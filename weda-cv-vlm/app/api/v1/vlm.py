"""POST /v1/vlm/* — streamed VLM inference (SSE).

The SceneReasoner publishes tokens to a per-query condvar-backed
buffer; this handler awaits on it in a thread-pool executor (no
polling). Concurrent requests cancel-and-replace via per-query
cancel events; the SceneReasoner serialises every ``model.generate()``
call through its generation lock.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

try:
    from sse_starlette.sse import EventSourceResponse
    _HAS_SSE = True
except ImportError:
    _HAS_SSE = False

from app.api.v1._deps import PipelineHandle, require_auth
from app.modules.fact_checker import missing_classes_in_query, verify as factcheck_verify
from app.modules.query_engine import QueryEngine, TIER1_TYPES, TIER2_TYPES, TIER25_TYPES
from app.modules.telemetry import QueryTelemetry
from app.utils.logger import get_logger

logger = get_logger()

router = APIRouter(prefix="/vlm", tags=["vlm"])

_TOKEN_WAIT_TIMEOUT_S = 5.0
_TOTAL_DEADLINE_S = 65.0


class VLMQueryRequest(BaseModel):
    query: str = Field(..., description="Natural-language question to ask of the live scene.")
    max_tokens: int = Field(200, description="Upper bound on tokens generated.")
    factcheck: bool = Field(True, description="When true, the final `done` event carries a fact-check report cross-referenced against the live FramePacket.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Is anyone in zone 3 without PPE?",
                "max_tokens": 200,
                "factcheck": True,
            }
        }
    }


class VLMDescribeRequest(BaseModel):
    track_ids: Optional[List[int]] = Field(None, description="Optional list of track ids to focus on. Track ids change as objects re-enter the frame — if any requested id isn't currently active the endpoint falls back to the live ones or to a general description. Omit entirely to describe the whole scene.")
    query: Optional[str] = Field(None, description="Optional user-supplied focus prompt. Defaults to a generic per-track measurement description.")
    max_tokens: int = Field(80, description="Upper bound on tokens generated.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Is the person wearing PPE?",
                "max_tokens": 80,
            }
        }
    }


class VLMDescribeResponse(BaseModel):
    track_ids: List[int] = Field(..., description="Echo of the track ids the description focused on.")
    query: str = Field(..., description="The resolved prompt that was sent to the reasoner.")
    response: str = Field(..., description="Reasoned answer. Empty when the reasoner is degraded or timed out.")
    factcheck: Optional[dict] = Field(None, description="Per-claim verification of the response against the live sensor data. Same shape as `POST /v1/factcheck`'s response. Null when fact-check could not run.")


@router.post(
    "/query",
    summary="Streamed VLM query (SSE)",
    description=(
        "Open-ended grounded VLM query. Streams `token` events as "
        "tokens arrive, terminates with `done` (carrying optional "
        "fact-check report) or `error`. Concurrent requests supersede "
        "each other via the cancel-event mutex. Body: "
        "`{query: str, max_tokens?: int, factcheck?: bool}`."
    ),
    responses={
        200: {
            "description": "Server-Sent Events stream. Each event is `token` (with `{text}` payload), `done` (with `{text_final, factcheck?}` payload), or `error` (with `{message}` payload).",
            "content": {"text/event-stream": {}},
        }
    },
)
async def vlm_query(
    body: VLMQueryRequest,
    handle: PipelineHandle = Depends(require_auth),
):
    if not _HAS_SSE:
        raise HTTPException(500, "sse-starlette not installed")
    sr = handle.scene_reasoner
    if sr is None:
        raise HTTPException(503, "scene reasoner not configured")
    packet = handle.latest_packet
    if packet is None:
        raise HTTPException(503, "no frame yet")
    query = body.query or ""
    if not query:
        raise HTTPException(400, "missing 'query'")
    do_factcheck = bool(body.factcheck)

    missing = missing_classes_in_query(query, packet.detections)
    if missing:
        present = sorted({d.class_name for d in (packet.detections or [])})
        msg = f"Cannot answer — object(s) not in scene: {', '.join(missing)}. Currently detected: {present or 'nothing'}."
        return EventSourceResponse(_emit_text_as_sse(msg, do_factcheck, packet, handle, tier="refused"), ping=15)

    qtype, _, _ = QueryEngine.classify(query)
    if qtype in TIER1_TYPES or qtype in TIER2_TYPES:
        cache = getattr(sr, "object_cache", {}) or {}
        answer, _ = QueryEngine.process(query, packet.detections, packet.spatial, cache)
        tier = "tier1" if qtype in TIER1_TYPES else "tier2"
        return EventSourceResponse(_emit_text_as_sse(answer, do_factcheck, packet, handle, tier=tier), ping=15)

    if qtype in TIER25_TYPES:
        pr = handle.physical_reasoner
        if pr is not None:
            return EventSourceResponse(
                _run_physical_as_sse(handle, pr, packet, query, do_factcheck),
                ping=15,
            )

    grounded_ctx = ""
    try:
        grounded_ctx = QueryEngine.build_grounded_json(
            packet.detections, packet.spatial,
            packet.rgb.shape if packet.rgb is not None else (0, 0, 0),
            getattr(sr, "object_cache", {}) or {},
        )
    except Exception as e:
        logger.debug("grounded JSON build failed: %s", e)

    qid = sr.submit_user_query(
        packet.rgb, packet.detections, grounded_ctx, query, packet.spatial,
    )
    if qid is None:
        # Either pre-flight rejected, or a pending payload was queued
        # behind a running worker. Surface the user-facing reason if
        # the reasoner already wrote one.
        msg = sr.user_error_message or "query queued behind in-flight worker"
        raise HTTPException(503, msg)

    cfg = handle.config
    tariff = float(getattr(cfg, "telemetry_tariff_usd_per_kwh", 0.12)) if cfg else 0.12
    return EventSourceResponse(
        _stream_with_telemetry(
            handle, query,
            _stream_tokens(sr, qid, packet, do_factcheck),
            tariff,
        ),
        ping=15,
    )


@router.post(
    "/describe",
    response_model=VLMDescribeResponse,
    summary="Per-track description (JSON)",
    description=(
        "Returns a synchronous description of the specified tracks. "
        "Non-streaming JSON — click Execute and the response lands "
        "without an SSE stream. Body: "
        "`{track_ids: [int], query?: str, max_tokens?: int}`."
    ),
)
async def vlm_describe(
    body: VLMDescribeRequest,
    handle: PipelineHandle = Depends(require_auth),
) -> VLMDescribeResponse:
    pr = handle.physical_reasoner
    if pr is None:
        raise HTTPException(503, "physical reasoner not configured")
    if handle.latest_packet is None:
        raise HTTPException(503, "no frame yet")
    packet = handle.latest_packet
    user_q = body.query or ""
    if user_q:
        missing_cls = missing_classes_in_query(user_q, packet.detections)
        if missing_cls:
            present = sorted({d.class_name for d in (packet.detections or [])})
            return VLMDescribeResponse(
                track_ids=body.track_ids or [],
                query=user_q,
                response=(
                    f"Cannot answer — object(s) not in scene: {', '.join(missing_cls)}. "
                    f"Currently detected: {present or 'nothing'}."
                ),
                factcheck=None,
            )

    active_ids = {int(d.track_id) for d in (packet.detections or [])}
    requested = [int(tid) for tid in (body.track_ids or [])]
    alive = [tid for tid in requested if tid in active_ids]
    dropped = [tid for tid in requested if tid not in active_ids]

    if requested and alive:
        effective_track_ids = alive
        prompt = f"Focus on track ids {alive}. " + (user_q or "Describe each in detail with measurements.")
    else:
        effective_track_ids = sorted(active_ids)
        prompt = user_q or "Describe the current scene with measurements."
    cfg = handle.config
    tariff = float(getattr(cfg, "telemetry_tariff_usd_per_kwh", 0.12)) if cfg else 0.12
    last = ""
    with QueryTelemetry(
        gpu_monitor=handle.gpu_monitor,
        query=prompt,
        source="http",
        tariff_usd_per_kwh=tariff,
    ) as rec:
        try:
            prev = pr.latest_query_response or ""
            pr.request_query(packet.rgb, packet.detections, packet.spatial, prompt)
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                await asyncio.sleep(0.1)
                cur = pr.latest_query_response or ""
                if cur and cur != prev:
                    last = cur
                    if not getattr(pr, "_query_running", False):
                        break
            rec.tokens_generated = len(last.split())
        except Exception as e:
            rec.success = False
            rec.error = repr(e)
            raise
    fc = None
    if last:
        try:
            cfg2 = handle.config
            fc = factcheck_verify(
                last, packet,
                distance_tolerance_pct=float(getattr(cfg2, "fact_check_distance_tolerance_pct", 15.0)),
                size_tolerance_pct=float(getattr(cfg2, "fact_check_size_tolerance_pct", 25.0)),
            ).to_dict()
        except Exception as e:
            logger.debug("describe factcheck failed: %s", e)
    final_response = last
    if dropped and not alive:
        final_response = (
            f"Requested track id(s) {dropped} are no longer on this frame "
            f"(track ids change as objects re-enter). Answering across all "
            f"current tracks {effective_track_ids}.\n\n"
        ) + last
    elif dropped:
        final_response = (
            f"Track id(s) {dropped} are no longer active; answered using "
            f"the alive subset {alive}.\n\n"
        ) + last
    return VLMDescribeResponse(
        track_ids=effective_track_ids,
        query=prompt,
        response=final_response,
        factcheck=fc,
    )


@router.post(
    "/cancel",
    summary="Cancel any in-flight VLM stream",
    description=(
        "Idempotent. Fires every active cancel event on the SceneReasoner; "
        "the running generate() exits within 1-2 tokens via the "
        "StoppingCriteria hook. Cancels both user-query and auto-caption "
        "in-flight runs — there is one shared model, so cancellation is "
        "global by design."
    ),
)
async def vlm_cancel(
    handle: PipelineHandle = Depends(require_auth),
) -> dict:
    sr = handle.scene_reasoner
    if sr is None:
        raise HTTPException(503, "scene reasoner not configured")
    cancel_events = getattr(sr, "_cancel_events", None)
    if isinstance(cancel_events, dict):
        for ev in cancel_events.values():
            try:
                ev.set()
            except Exception:
                pass
    auto_event = getattr(sr, "_auto_cancel_event", None)
    if auto_event is not None:
        try:
            auto_event.set()
        except Exception:
            pass
    return {"cancelled": True}


async def _emit_text_as_sse(
    text: str,
    do_factcheck: bool,
    packet,
    handle: PipelineHandle,
    tier: str = "tier1",
) -> AsyncIterator[dict]:
    cfg = handle.config
    tariff = float(getattr(cfg, "telemetry_tariff_usd_per_kwh", 0.12)) if cfg else 0.12
    with QueryTelemetry(
        gpu_monitor=handle.gpu_monitor,
        query=text[:200],
        source=f"http:{tier}",
        tariff_usd_per_kwh=tariff,
    ) as rec:
        rec.tokens_generated = len(text.split())
        chunk_size = 80
        for i in range(0, max(len(text), 1), chunk_size):
            yield {"event": "token", "data": json.dumps({"text": text[i:i + chunk_size]}, ensure_ascii=False)}
        payload: dict = {"text_final": text, "tier": tier}
        if do_factcheck and text and tier != "refused":
            try:
                report = factcheck_verify(
                    text, packet,
                    distance_tolerance_pct=float(getattr(cfg, "fact_check_distance_tolerance_pct", 15.0)),
                    size_tolerance_pct=float(getattr(cfg, "fact_check_size_tolerance_pct", 25.0)),
                )
                payload["factcheck"] = report.to_dict()
                rec.factcheck_verified = report.verified
                rec.factcheck_unverifiable = report.unverifiable
                rec.factcheck_contradicted = report.contradicted
            except Exception as e:
                logger.debug("inline factcheck failed: %s", e)
        yield {"event": "done", "data": json.dumps(payload, ensure_ascii=False)}


async def _run_physical_as_sse(
    handle: PipelineHandle,
    pr,
    packet,
    query: str,
    do_factcheck: bool,
) -> AsyncIterator[dict]:
    import time as _time
    cfg = handle.config
    tariff = float(getattr(cfg, "telemetry_tariff_usd_per_kwh", 0.12)) if cfg else 0.12
    last = ""
    with QueryTelemetry(
        gpu_monitor=handle.gpu_monitor,
        query=query,
        source="http:tier25",
        tariff_usd_per_kwh=tariff,
    ) as rec:
        try:
            prev = pr.latest_query_response or ""
            pr.request_query(packet.rgb, packet.detections, packet.spatial, query)
            deadline = _time.monotonic() + 30.0
            while _time.monotonic() < deadline:
                await asyncio.sleep(0.1)
                cur = pr.latest_query_response or ""
                if cur and cur != prev:
                    last = cur
                    if not getattr(pr, "_query_running", False):
                        break
            rec.tokens_generated = len(last.split())
        except Exception as e:
            rec.success = False
            rec.error = repr(e)
            last = last or f"physical reasoner error: {e}"
    chunk_size = 80
    for i in range(0, max(len(last), 1), chunk_size):
        yield {"event": "token", "data": json.dumps({"text": last[i:i + chunk_size]}, ensure_ascii=False)}
    payload: dict = {"text_final": last, "tier": "tier25"}
    if do_factcheck and last:
        try:
            report = factcheck_verify(
                last, packet,
                distance_tolerance_pct=float(getattr(cfg, "fact_check_distance_tolerance_pct", 15.0)),
                size_tolerance_pct=float(getattr(cfg, "fact_check_size_tolerance_pct", 25.0)),
            )
            payload["factcheck"] = report.to_dict()
        except Exception as e:
            logger.debug("tier25 factcheck failed: %s", e)
    yield {"event": "done", "data": json.dumps(payload, ensure_ascii=False)}


async def _stream_with_telemetry(
    handle: PipelineHandle,
    query: str,
    inner: AsyncIterator[dict],
    tariff_usd_per_kwh: float,
) -> AsyncIterator[dict]:
    with QueryTelemetry(
        gpu_monitor=handle.gpu_monitor,
        query=query,
        source="http",
        tariff_usd_per_kwh=tariff_usd_per_kwh,
    ) as rec:
        tokens = 0
        async for event in inner:
            kind = event.get("event") if isinstance(event, dict) else None
            if kind == "token":
                tokens += 1
            elif kind == "done":
                try:
                    payload = json.loads(event.get("data", "{}"))
                    summary = (payload.get("factcheck") or {}).get("summary") or {}
                    rec.factcheck_verified = int(summary.get("verified", 0))
                    rec.factcheck_unverifiable = int(summary.get("unverifiable", 0))
                    rec.factcheck_contradicted = int(summary.get("contradicted", 0))
                except Exception:
                    pass
            elif kind == "error":
                rec.success = False
                try:
                    payload = json.loads(event.get("data", "{}"))
                    rec.error = str(payload.get("message", ""))
                except Exception:
                    rec.error = "stream error"
            yield event
        rec.tokens_generated = tokens


async def _stream_tokens(
    sr,
    qid: int,
    packet,
    do_factcheck: bool,
) -> AsyncIterator[dict]:
    """Pump token events out by awaiting on the per-query condvar in
    the default executor. Yields `token` events as text arrives, then
    a single `done` event with the fact-check summary, or an `error`
    event on failure / timeout.
    """
    loop = asyncio.get_event_loop()
    pos = 0
    deadline = loop.time() + _TOTAL_DEADLINE_S
    accumulated = ""
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            yield {"event": "error", "data": json.dumps({"message": "timeout"})}
            return
        stream = sr.get_token_stream(qid)
        if stream is None:
            # Worker finished and evicted the stream before we caught
            # up. Final state lives in latest_user_response.
            break
        wait_for = min(remaining, _TOKEN_WAIT_TIMEOUT_S)
        result = await loop.run_in_executor(None, stream.wait_next, pos, wait_for)
        if result is None:
            # done / error / timeout — re-check stream.done explicitly,
            # since wait_next can return None on plain timeout too.
            if stream.done:
                if stream.error:
                    yield {"event": "error",
                           "data": json.dumps({"message": stream.error})}
                    return
                break
            continue
        pos, chunk = result
        if chunk:
            accumulated += chunk
            yield {"event": "token", "data": json.dumps({"text": chunk}, ensure_ascii=False)}

    # Streamer finished; fall back to the canonical final answer the
    # reasoner committed under its lock — handles markdown stripping
    # and supersession correctness.
    final = sr.latest_user_response or accumulated
    payload: dict = {"text_final": final}
    if do_factcheck and final:
        try:
            report = factcheck_verify(final, packet)
            payload["factcheck"] = report.to_dict()
        except Exception as e:
            payload["factcheck_error"] = repr(e)
    yield {"event": "done", "data": json.dumps(payload, ensure_ascii=False)}
