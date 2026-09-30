"""POST /v1/physical/query — Cosmos-Reason2 physical-reasoning query."""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.v1._deps import PipelineHandle, require_auth
from app.modules.fact_checker import missing_classes_in_query, verify as factcheck_verify
from app.modules.query_engine import QueryEngine, TIER1_TYPES, TIER2_TYPES
from app.modules.telemetry import QueryTelemetry
from app.utils.logger import get_logger

logger = get_logger()

router = APIRouter(prefix="/physical", tags=["physical"])


class PhysicalQueryRequest(BaseModel):
    query: str = Field(..., description="Natural-language spatio-temporal question to ask the physical reasoner, e.g. \"Is the forklift safely clear of the worker?\"")
    max_tokens: Optional[int] = Field(None, description="Optional override for the reasoner's max output tokens. Falls back to the server-side default (see `physical_reasoning.max_tokens` in pipeline.yaml).")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Is the forklift safely clear of the worker?",
                "max_tokens": 80,
            }
        }
    }


class PhysicalQueryResponse(BaseModel):
    query: str = Field(..., description="Echo of the input query.")
    response: str = Field(..., description="Reasoned answer. Empty when the query timed out (rare) or the reasoner is degraded.")
    tier: str = Field("model", description='Which path produced the answer: `"refused"` (pre-flight class miss), `"tier1"` (structured answer from detections + spatial, no model), `"tier2"` (visual cache lookup), or `"model"` (Cosmos-Reason2 invoked).')
    factcheck: Optional[dict] = Field(None, description="Per-claim verification of the response against the live sensor data. Same shape as `POST /v1/factcheck`'s response. Null when fact-check could not run.")


@router.post(
    "/query",
    response_model=PhysicalQueryResponse,
    summary="Cosmos-Reason2 spatial / safety reasoning",
    description=(
        "Tier-2.5 physical-reasoning query. Non-streaming JSON. "
        "Reads the live frame's detections + spatial info as grounding."
    ),
)
async def physical_query(
    body: PhysicalQueryRequest,
    handle: PipelineHandle = Depends(require_auth),
) -> PhysicalQueryResponse:
    pr = handle.physical_reasoner
    if pr is None:
        raise HTTPException(503, "physical reasoner not configured")
    if handle.latest_packet is None:
        raise HTTPException(503, "no frame yet")
    packet = handle.latest_packet
    cfg = handle.config
    tariff = float(getattr(cfg, "telemetry_tariff_usd_per_kwh", 0.12)) if cfg else 0.12

    missing = missing_classes_in_query(body.query, packet.detections)
    if missing:
        present = sorted({d.class_name for d in (packet.detections or [])})
        return PhysicalQueryResponse(
            query=body.query,
            response=f"Cannot answer — object(s) not in scene: {', '.join(missing)}. Currently detected: {present or 'nothing'}.",
            tier="refused",
            factcheck=None,
        )

    qtype, _, _ = QueryEngine.classify(body.query)
    if qtype in TIER1_TYPES or qtype in TIER2_TYPES:
        sr = handle.scene_reasoner
        cache = getattr(sr, "object_cache", {}) if sr else {}
        answer, _ = QueryEngine.process(
            body.query, packet.detections, packet.spatial, cache or {},
        )
        return PhysicalQueryResponse(
            query=body.query,
            response=answer,
            tier="tier1" if qtype in TIER1_TYPES else "tier2",
            factcheck=None,
        )
    if body.max_tokens is not None and cfg is not None:
        cfg.physical_max_tokens = int(body.max_tokens)
    with QueryTelemetry(
        gpu_monitor=handle.gpu_monitor,
        query=body.query,
        source="http",
        tariff_usd_per_kwh=tariff,
    ) as rec:
        try:
            prev = pr.latest_query_response or ""
            pr.request_query(packet.rgb, packet.detections, packet.spatial, body.query)
            deadline = time.monotonic() + 30.0
            last = ""
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
            fc = factcheck_verify(
                last, packet,
                distance_tolerance_pct=float(getattr(cfg, "fact_check_distance_tolerance_pct", 15.0)),
                size_tolerance_pct=float(getattr(cfg, "fact_check_size_tolerance_pct", 25.0)),
            ).to_dict()
        except Exception as e:
            logger.debug("physical factcheck failed: %s", e)
    return PhysicalQueryResponse(query=body.query, response=last, tier="model", factcheck=fc)
