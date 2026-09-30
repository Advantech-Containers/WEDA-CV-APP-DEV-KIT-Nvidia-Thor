"""GET/POST /v1/spatial/* — 3-D primitives over the live FramePacket."""

from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.v1._deps import PipelineHandle, require_handle
from app.modules import spatial_tools

router = APIRouter(prefix="/spatial", tags=["spatial"])


# ── Schemas ─────────────────────────────────────────────────────────

class DistanceResponse(BaseModel):
    track_a: int = Field(..., description="First track id (echo of request).")
    track_b: int = Field(..., description="Second track id (echo of request).")
    distance_m: Optional[float] = Field(None, description="3-D Euclidean distance in metres, or null when either track lacks reliable depth on this frame.")
    reliable: bool = Field(..., description="True when both tracks have a clean depth reading. False otherwise (the distance is still returned if any was computable, but is noisy).")


class Blocker(BaseModel):
    track_id: int = Field(..., description="Track id of the blocking object.")
    class_name: str = Field(..., description="Class name of the blocker.")
    distance_m: float = Field(..., description="Distance along the corridor from `from_pos` to the blocker, in metres.")
    lateral_offset_m: float = Field(..., description="Perpendicular distance from the corridor centre-line in metres. ≤ width_m/2 by definition.")


class PathClearResponse(BaseModel):
    clear: bool = Field(..., description="True when no tracked object intersects the corridor.")
    path_length_m: float = Field(..., description="Straight-line length from `from_pos` to `to_pos` in metres.")
    blockers: List[Blocker] = Field(..., description="Blockers ordered by `dist_along_m`, closest to `from_pos` first. Empty when `clear=true`.")


class ZoneOccupancy(BaseModel):
    zone_name: str = Field(..., description="Zone name (from config/zones.yaml).")
    track_ids: List[int] = Field(..., description="Track ids of objects currently inside the zone polygon.")
    classes: List[str] = Field(..., description="Class names of the occupying tracks, same order as `track_ids`.")


class ZonesNowResponse(BaseModel):
    zones: List[ZoneOccupancy] = Field(..., description="One entry per configured zone. Empty list when no zones are configured (not an error).")


class MeasureResponse(BaseModel):
    width_m: float = Field(..., description="Physical width of the region in metres.")
    height_m: float = Field(..., description="Physical height of the region in metres.")
    distance_m: float = Field(..., description="Distance to the region (camera → object) in metres.")
    reliable: bool = Field(..., description="True when the depth sample is clean. False when the depth came from an expanded fallback ROI (or no detection overlapped the bbox).")


class CollisionPrediction(BaseModel):
    track_a: int = Field(..., description="Track id of the first object in the pair.")
    track_b: int = Field(..., description="Track id of the second object in the pair.")
    class_a: str = Field(..., description="Class name of `track_a`.")
    class_b: str = Field(..., description="Class name of `track_b`.")
    eta_s: float = Field(..., description="Predicted seconds until closest approach. 0 if they're already at min distance.")
    closing_speed_mps: float = Field(..., description="Rate at which the pair is closing in m/s. Strictly positive — non-closing pairs are filtered out.")
    miss_distance_m: float = Field(..., description="3-D distance at the predicted closest-approach instant in metres.")


class PredictCollisionsRequest(BaseModel):
    horizon_s: float = Field(3.0, description="How far into the future to extrapolate trajectories, in seconds.")
    danger_distance_m: float = Field(0.8, description="Only predictions with miss distance below this are returned.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "horizon_s": 3.0,
                "danger_distance_m": 0.8,
            }
        }
    }


class PredictCollisionsResponse(BaseModel):
    predictions: List[CollisionPrediction] = Field(..., description="Pairs ordered by `eta_s`, most imminent first.")


# ── Helpers ─────────────────────────────────────────────────────────

def _packet(handle: PipelineHandle):
    if handle.latest_packet is None:
        raise HTTPException(503, "no frame yet")
    return handle.latest_packet


# ── Routes ──────────────────────────────────────────────────────────

@router.get(
    "/distance",
    response_model=DistanceResponse,
    summary="3-D distance between two tracks",
    description="Returns Euclidean distance in metres between two tracked objects' 3-D positions. `distance_m` is null when either track lacks a reliable depth measurement on the current frame.",
)
async def distance(
    track_a: int, track_b: int,
    handle: PipelineHandle = Depends(require_handle),
) -> DistanceResponse:
    return DistanceResponse(**asdict(spatial_tools.distance(_packet(handle), track_a, track_b)))


@router.get(
    "/path-clear",
    response_model=PathClearResponse,
    summary="Is a 3-D corridor clear of blockers?",
    description="Tests whether the straight-line corridor from `from_pos` to `to_pos` (3-D world coords, metres) of width `width_m` is free of any tracked object. Returns the blocker list ordered by along-path distance — closest first.",
    include_in_schema=False,
)
async def path_clear_xyz(
    from_x: float, from_y: float, from_z: float,
    to_x: float, to_y: float, to_z: float,
    width_m: float = 0.8,
    handle: PipelineHandle = Depends(require_handle),
) -> PathClearResponse:
    res = spatial_tools.path_clear(
        _packet(handle),
        from_pos=np.array([from_x, from_y, from_z], dtype=np.float32),
        to_pos=np.array([to_x, to_y, to_z], dtype=np.float32),
        width_m=width_m,
    )
    return PathClearResponse(
        clear=res.clear,
        path_length_m=res.path_length_m,
        blockers=[Blocker(**asdict(b)) for b in res.blockers],
    )


@router.get(
    "/zones-now",
    response_model=ZonesNowResponse,
    summary="Operator-defined zone occupancy",
    description="Which tracks are currently inside each zone defined in `config/zones.yaml`. Zones are pixel-space polygons. Returns an empty list when no zones are configured (not an error).",
    include_in_schema=False,
)
async def zones_now(handle: PipelineHandle = Depends(require_handle)) -> ZonesNowResponse:
    cfg = handle.config
    zones = []
    if cfg is not None:
        zones = list(getattr(cfg, "zones", []) or [])
    if not zones:
        return ZonesNowResponse(zones=[])
    occupancies = spatial_tools.zones_now(_packet(handle), zones)
    return ZonesNowResponse(zones=[ZoneOccupancy(**asdict(z)) for z in occupancies])


@router.get(
    "/measure",
    response_model=MeasureResponse,
    summary="Estimate physical W×H of a region",
    description="Physical width and height (metres) of the region defined by the pixel-space bbox `[x1, y1, x2, y2]`. Uses the spatial estimate of the most-overlapping detection.",
    include_in_schema=False,
)
async def measure(
    x1: float, y1: float, x2: float, y2: float,
    handle: PipelineHandle = Depends(require_handle),
) -> MeasureResponse:
    return MeasureResponse(**asdict(spatial_tools.measure(_packet(handle), (x1, y1, x2, y2))))


@router.post(
    "/predict-collisions",
    response_model=PredictCollisionsResponse,
    summary="Pairwise collision prediction",
    description="Pairwise closest approaches over the next `horizon_s` seconds, computed from each track's smoothed 3-D velocity. Returns only pairs with miss distance below `danger_distance_m` and positive closing speed.",
)
async def predict_collisions(
    body: Optional[PredictCollisionsRequest] = None,
    handle: PipelineHandle = Depends(require_handle),
) -> PredictCollisionsResponse:
    if body is None:
        body = PredictCollisionsRequest()
    preds = spatial_tools.predict_collisions(
        _packet(handle), horizon_s=body.horizon_s, danger_distance_m=body.danger_distance_m,
    )
    return PredictCollisionsResponse(predictions=[CollisionPrediction(**asdict(p)) for p in preds])
