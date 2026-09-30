"""GET /v1/scene/* — zero-inference reads of the live FramePacket."""

from __future__ import annotations

import time as _time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.v1._deps import PipelineHandle, require_handle

router = APIRouter(prefix="/scene", tags=["scene"])


# ── Schemas ─────────────────────────────────────────────────────────

class DetectionItem(BaseModel):
    track_id: int = Field(..., description="Stable tracker id (≥0). Use this as the agent's reference for the object across frames.")
    class_name: str = Field(..., description='COCO class label (e.g. "person", "tv", "forklift").')
    class_id: int = Field(..., description="COCO class index (0-79).")
    confidence: float = Field(..., description="Detector confidence, 0.0-1.0.")
    bbox: List[float] = Field(..., description="`[x1, y1, x2, y2]` in pixel space, camera-native resolution. Always length 4.")


class SpatialItem(BaseModel):
    distance_m: float = Field(..., description="3-D Euclidean distance from camera origin in metres.")
    position_3d: Optional[List[float]] = Field(None, description="`[x, y, z]` in metres, camera-centric (+X right, +Y down, +Z forward).")
    velocity_3d: Optional[List[float]] = Field(None, description="`[vx, vy, vz]` in m/s. Null when the track is brand-new (no Kalman estimate yet).")
    speed_ms: float = Field(..., description="`|velocity_3d|` in m/s — the L2 norm of the velocity vector.")
    size_estimate: List[float] = Field(..., description="`[width_m, height_m]` — physical dimensions of the object derived from depth + intrinsics.")
    nearest_neighbor_id: int = Field(..., description="Track id of the closest other object on this frame, or -1 if no neighbour.")
    nearest_neighbor_dist: Optional[float] = Field(None, description="3-D distance to that neighbour in metres, or null if no neighbour.")
    depth_reliable: bool = Field(..., description="True when the depth value came from a clean ROI sample. False when depth was inferred from an expanded fallback region — distance may be noisier.")


class SceneNowResponse(BaseModel):
    frame_id: int = Field(..., description="Monotonic frame counter for the captured frame.")
    ts: float = Field(..., description="Capture timestamp in epoch seconds (wall-clock). Feed this directly into `/v1/scene/at?ts=…` or `/v1/replay?anchor_ts=…`.")
    timestamp: float = Field(..., description="Same instant in `time.monotonic()` seconds. For in-process ordering only — not comparable across reboots.")
    detections: List[DetectionItem] = Field(..., description="All tracked objects in the current frame.")
    spatial: List[SpatialItem] = Field(..., description="Per-detection 3-D info. Same order as `detections`.")
    scene_description: str = Field(..., description="Auto-caption (multi-paragraph VLM output). Empty until the first auto-caption completes.")
    physical_assessment: str = Field(..., description="Cosmos-Reason2 latest physical-reasoner output. Empty until the first assessment fires.")
    scene_changed: bool = Field(..., description="True when the embedder detected a scene change on this frame (cosine sim < threshold).")
    safety_alert: bool = Field(..., description="True when at least one tracked object is inside the safety zone.")
    safety_alert_class: str = Field(..., description='Class name of the alert-triggering track, or "" when no alert.')
    safety_alert_distance: float = Field(..., description="Distance to the alert-triggering track in metres.")
    safety_alert_track_id: int = Field(..., description="Track id of the alert-triggering object, or -1 when no alert.")


class SceneAtResponse(BaseModel):
    ts: float = Field(..., description="Epoch timestamp of the buffered snapshot (wall-clock seconds).")
    frame_id: int = Field(..., description="Frame id of the captured snapshot.")
    scene_description: str = Field(..., description="Auto-caption text at that moment.")
    physical_assessment: str = Field(..., description="Physical-reasoner output at that moment.")
    detection_count: int = Field(..., description="How many tracked objects were present.")
    detection_classes: List[str] = Field(..., description="Class names of every tracked object at that moment.")
    track_ids: List[int] = Field(..., description="Track ids of every object at that moment.")


class SceneFindRequest(BaseModel):
    query: str = Field(..., description="Natural-language search phrase, e.g. `\"worker without PPE\"`.")
    since_s: int = Field(3600, description="How far back to search, in seconds. Capped to the scene-memory window (30 min by default).")
    top_k: int = Field(5, description="Maximum number of results.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "worker without PPE",
                "since_s": 600,
                "top_k": 5,
            }
        }
    }


class SceneFindHit(BaseModel):
    similarity: float = Field(..., description="Cosine similarity to the query embedding, 0.0-1.0.")
    ts: float = Field(..., description="Wall-clock timestamp of the matched snapshot.")
    frame_id: int = Field(..., description="Frame id of the matched snapshot.")
    scene_description: str = Field(..., description="Auto-caption at that moment.")
    detection_classes: List[str] = Field(..., description="Object classes present at that moment.")


class SceneFindResponse(BaseModel):
    query: str = Field(..., description="Echo of the query for client correlation.")
    hits: List[SceneFindHit] = Field(..., description="Top-k matches ordered by similarity, highest first.")


class ObjectDescriptionResponse(BaseModel):
    track_id: int = Field(..., description="Track id requested.")
    class_name: str = Field(..., description="Current class name for this track, or empty if the track is no longer active.")
    description: str = Field(..., description="Cached VLM description of the object.")


# ── Helpers ─────────────────────────────────────────────────────────

_EMPTY_PACKET: dict = {
    "frame_id": 0,
    "ts": 0.0,
    "timestamp": 0.0,
    "detections": [],
    "spatial": [],
    "scene_description": "",
    "physical_assessment": "",
    "scene_changed": False,
    "safety_alert": False,
    "safety_alert_class": "",
    "safety_alert_distance": 0.0,
    "safety_alert_track_id": -1,
}


def _packet_to_dict(p) -> dict:
    """Serialise a FramePacket. Strips the heavy RGB/depth arrays."""
    if p is None:
        return dict(_EMPTY_PACKET)
    dets = []
    for d in (p.detections or []):
        dets.append({
            "track_id": int(d.track_id),
            "class_name": d.class_name,
            "class_id": int(d.class_id),
            "confidence": float(d.confidence),
            "bbox": [float(x) for x in d.bbox.tolist()],
        })
    spatial = []
    for s in (p.spatial or []):
        spatial.append({
            "distance_m": float(s.distance_m),
            "position_3d": (
                [float(x) for x in s.position_3d.tolist()]
                if s.position_3d is not None else None
            ),
            "velocity_3d": (
                [float(x) for x in s.velocity_3d.tolist()]
                if getattr(s, "velocity_3d", None) is not None else None
            ),
            "speed_ms": float(getattr(s, "speed_ms", 0.0)),
            "size_estimate": [float(x) for x in s.size_estimate],
            "nearest_neighbor_id": int(s.nearest_neighbor_id),
            "nearest_neighbor_dist": (
                float(s.nearest_neighbor_dist)
                if s.nearest_neighbor_dist != float("inf") else None
            ),
            "depth_reliable": bool(s.depth_reliable),
        })
    # Convert the packet's monotonic timestamp to an epoch value so the
    # client can pass it back into /scene/at or /replay without doing
    # their own clock conversion. Monotonic and epoch don't share an
    # origin, so we reconstruct: epoch_now - (monotonic_now - p.timestamp).
    mono_now = _time.monotonic()
    epoch_now = _time.time()
    epoch_ts = epoch_now - (mono_now - float(p.timestamp))
    return {
        "frame_id": int(p.frame_id),
        "ts": float(epoch_ts),
        "timestamp": float(p.timestamp),
        "detections": dets,
        "spatial": spatial,
        "scene_description": p.scene_description or "",
        "physical_assessment": getattr(p, "physical_reasoning", "") or "",
        "scene_changed": bool(p.scene_changed),
        "safety_alert": bool(p.safety_alert),
        "safety_alert_class": "",
        "safety_alert_distance": float(p.safety_alert_distance),
        "safety_alert_track_id": int(p.safety_alert_track_id),
    }


# ── Routes ──────────────────────────────────────────────────────────

@router.get(
    "/now",
    response_model=SceneNowResponse,
    summary="Current frame snapshot",
    description=(
        "Returns the most recent FramePacket as JSON: detections, 3-D "
        "spatial info, the auto-caption scene description, the latest "
        "physical-reasoner assessment, and any active safety alert. "
        "Zero inference — cheap to poll at agent-tick rate."
    ),
)
async def scene_now(handle: PipelineHandle = Depends(require_handle)) -> SceneNowResponse:
    return SceneNowResponse(**_packet_to_dict(handle.latest_packet))


@router.get(
    "/at",
    response_model=SceneAtResponse,
    summary="Scene at a relative or absolute timestamp",
    description=(
        "Returns the buffered snapshot closest to a given moment. Three "
        "ways to ask:\n\n"
        "1. **`ago_s=N`** — N seconds ago (relative; easiest). e.g. `ago_s=10` for 10 s ago.\n"
        "2. **`ts=<epoch>`** — absolute Unix epoch seconds, e.g. from `/v1/scene/now.ts` or `alerts[*].ts`.\n"
        "3. **No params** — returns the most recent buffered entry.\n\n"
        "Returns 404 when no entry exists within the sample-period "
        "tolerance (~2 seconds at the default 1 Hz sample rate)."
    ),
)
async def scene_at(
    ts: Optional[float] = None,
    ago_s: Optional[float] = None,
    handle: PipelineHandle = Depends(require_handle),
) -> SceneAtResponse:
    if handle.scene_memory is None:
        raise HTTPException(503, "scene memory not configured")
    # Resolve which timestamp to look up. `ago_s` wins over `ts` if both
    # are provided (more user-friendly), and absent both means "now".
    if ago_s is not None:
        target_ts = _time.time() - float(ago_s)
    elif ts is not None:
        target_ts = float(ts)
    else:
        target_ts = _time.time()
    e = handle.scene_memory.at(target_ts)
    if e is None:
        raise HTTPException(
            404,
            f"no entry close to that timestamp (resolved to {target_ts:.1f} epoch)",
        )
    return SceneAtResponse(
        ts=e.ts,
        frame_id=e.frame_id,
        scene_description=e.scene_description,
        physical_assessment=e.physical_assessment,
        detection_count=e.detection_count,
        detection_classes=e.detection_classes,
        track_ids=e.track_ids,
    )


@router.post(
    "/find",
    response_model=SceneFindResponse,
    summary="Semantic search over scene memory",
    description=(
        "Top-k entries from the scene memory whose embeddings are "
        "closest to the query, computed via Cosmos-Embed1. Falls back "
        "to substring match in the description when the embedder is "
        "unavailable."
    ),
)
async def scene_find(
    body: SceneFindRequest,
    handle: PipelineHandle = Depends(require_handle),
) -> SceneFindResponse:
    if handle.scene_memory is None:
        raise HTTPException(503, "scene memory not configured")
    hits = handle.scene_memory.find(body.query, since_s=float(body.since_s), top_k=body.top_k)
    return SceneFindResponse(
        query=body.query,
        hits=[
            SceneFindHit(
                similarity=h.similarity,
                ts=h.entry.ts,
                frame_id=h.entry.frame_id,
                scene_description=h.entry.scene_description,
                detection_classes=h.entry.detection_classes,
            )
            for h in hits
        ],
    )


@router.get(
    "/objects/{track_id}",
    response_model=ObjectDescriptionResponse,
    summary="Cached visual description of a track",
    description=(
        "Cached VLM description for a tracked object. The auto-caption "
        "worker populates this in the background; reading it avoids "
        "re-running the VLM. Returns 404 when no description has been "
        "generated yet."
    ),
)
async def get_object(
    track_id: int,
    handle: PipelineHandle = Depends(require_handle),
) -> ObjectDescriptionResponse:
    sr = handle.scene_reasoner
    if sr is None:
        raise HTTPException(503, "scene reasoner not configured")
    cache = getattr(sr, "object_cache", {}) or {}
    desc = cache.get(track_id)
    if desc is None:
        raise HTTPException(404, "no cached description for that track")
    cls = ""
    if handle.latest_packet:
        for d in handle.latest_packet.detections or []:
            if d.track_id == track_id:
                cls = d.class_name
                break
    return ObjectDescriptionResponse(track_id=track_id, class_name=cls, description=desc)
