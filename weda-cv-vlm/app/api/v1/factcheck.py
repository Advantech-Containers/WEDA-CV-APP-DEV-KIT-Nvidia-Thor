"""POST /v1/factcheck — verify VLM claims against live sensor data."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.v1._deps import PipelineHandle, require_handle
from app.modules.fact_checker import verify

router = APIRouter(prefix="/factcheck", tags=["factcheck"])


class FactCheckRequest(BaseModel):
    text: str = Field(..., description="Free-form text to verify. Numeric claims (e.g. \"the gap is 0.9 m\") and categorical claims (e.g. \"two workers\") are detected and checked.")
    frame_id: Optional[int] = Field(None, description="Optional frame id to verify against. Defaults to the current latest frame.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "The forklift is 2.5 metres from the worker.",
                "frame_id": None,
            }
        }
    }


class FactCheckClaim(BaseModel):
    raw: str = Field(..., description="The substring from the input that constituted the claim.")
    span: List[int] = Field(..., description="`[start, end]` character offsets in the input text. Length 2. Use these to splice badges inline.")
    kind: str = Field(..., description='Claim category: `"distance" | "count" | "size" | "presence"`.')
    value: float = Field(..., description="Extracted numeric value (metres for distance, integer count for count, etc.).")
    unit: str = Field(..., description='Canonical unit: `"m" | "count" | "presence" | "m²"`.')
    subject: Optional[str] = Field(None, description="Class name the claim refers to, when identifiable (e.g. `\"person\"`, `\"forklift\"`).")
    verdict: str = Field(..., description='One of `"verified"` / `"unverifiable"` / `"contradicted"`.')
    badge: str = Field(..., description="Emoji badge: 🟢 / 🟡 / 🔴.")
    sensor_value: Optional[float] = Field(None, description="Sensor-measured value used for comparison, or null when unverifiable.")
    tolerance: float = Field(0.0, description="Absolute tolerance applied to the comparison, in the claim's unit.")
    rationale: str = Field("", description="Human-readable explanation of the verdict.")


class FactCheckSummary(BaseModel):
    total: int = Field(..., description="Total claims parsed from the text.")
    verified: int = Field(..., description="Count of claims with verdict `verified`.")
    unverifiable: int = Field(..., description="Count of claims with verdict `unverifiable`.")
    contradicted: int = Field(..., description="Count of claims with verdict `contradicted`. Non-zero values should draw operator attention.")
    verified_pct: float = Field(..., description="Percentage of total claims verified (0.0 - 100.0).")


class FactCheckResponse(BaseModel):
    text: str = Field(..., description="Echo of the input text being verified.")
    claims: List[FactCheckClaim] = Field(..., description="One entry per detected claim. Empty when the text contained no verifiable claims.")
    summary: FactCheckSummary = Field(..., description="Aggregate counts for quick badge rendering.")


@router.post(
    "",
    response_model=FactCheckResponse,
    summary="Cross-reference free-form text against live sensor data",
    description=(
        "Parses the input for numeric / categorical claims (distances, "
        "counts, sizes, presence) and verifies each against the live "
        "FramePacket — depth, bbox geometry, class taxonomy. Per-claim "
        "verdicts include character spans so the caller can render "
        "badges inline. O(n + m) over claims and detections; no model "
        "inference."
    ),
)
async def factcheck(
    body: FactCheckRequest,
    handle: PipelineHandle = Depends(require_handle),
) -> FactCheckResponse:
    if not body.text:
        raise HTTPException(400, "missing 'text'")
    if handle.latest_packet is None:
        raise HTTPException(503, "no frame yet")
    cfg = handle.config
    report = verify(
        body.text,
        handle.latest_packet,
        distance_tolerance_pct=float(getattr(cfg, "fact_check_distance_tolerance_pct", 15.0)),
        size_tolerance_pct=float(getattr(cfg, "fact_check_size_tolerance_pct", 25.0)),
    )
    return FactCheckResponse(**report.to_dict())
