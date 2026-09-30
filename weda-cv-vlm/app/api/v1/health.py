"""GET /v1/health — service liveness + model load status."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.v1._deps import PipelineHandle, require_handle

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str = Field(..., description='"ok" when the service is up. Always "ok" on 200; on degraded state the call returns 503 Problem Details instead.')
    vlm_loaded: bool = Field(..., description="Scene-understanding VLM (Qwen3-VL or fallback) is loaded and ready to accept `/v1/vlm/*` calls.")
    physical_reasoner_loaded: bool = Field(..., description="Cosmos-Reason2-2B is loaded and ready to accept `/v1/physical/query`.")
    embedder_loaded: bool = Field(..., description="Cosmos-Embed1 is loaded and ready to accept `/v1/embeddings/*` and `/v1/scene/find`.")
    uma_pct: float = Field(..., description="Unified-memory utilisation as a percentage (0-100). > 90 indicates pressure.")
    temperature_c: float = Field(..., description="GPU temperature in Celsius from /sys/class/thermal.")
    frame_id: int = Field(..., description="Monotonic frame counter from the latest FramePacket. 0 before the first frame is rendered.")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness + per-model load status",
    description="Aggregate liveness signal — poll this before tool registration. All three `*_loaded` flags must be `true` before issuing inference calls.",
)
async def health(handle: PipelineHandle = Depends(require_handle)) -> HealthResponse:
    sr = handle.scene_reasoner
    pr = handle.physical_reasoner
    em = handle.embedder
    gm = handle.gpu_monitor

    uma_pct = 0.0
    temperature = 0.0
    if gm is not None:
        try:
            uma_pct = float(gm.get_memory_utilization() or 0.0) * 100.0
            temperature = float(gm.get_temperature() or 0.0)
        except Exception:
            pass

    return HealthResponse(
        status="ok",
        vlm_loaded=bool(getattr(sr, "_loaded", False)) if sr else False,
        physical_reasoner_loaded=bool(getattr(pr, "_loaded", False)) if pr else False,
        embedder_loaded=bool(getattr(em, "is_loaded", False)) if em else False,
        uma_pct=round(uma_pct, 1),
        temperature_c=round(temperature, 1),
        frame_id=getattr(handle.latest_packet, "frame_id", 0)
            if handle.latest_packet else 0,
    )
