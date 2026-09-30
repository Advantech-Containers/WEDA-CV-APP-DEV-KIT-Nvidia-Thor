"""POST /v1/replay — synthesize a replay of past scene moments.

Combines: scene_memory.find/at + a VLM narration pass + fact-check.
Powers "show me what happened 30 s ago" style temporal queries.
"""

from __future__ import annotations

from typing import List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.v1._deps import PipelineHandle, require_auth

router = APIRouter(prefix="/replay", tags=["replay"])


class ReplayRequest(BaseModel):
    query: Optional[str] = Field(None, description="Optional semantic query — when set, runs semantic search over scene memory and returns the top-k matches.")
    anchor_ts: Optional[float] = Field(None, description="Optional epoch timestamp anchor — when set, returns the snapshot closest to that timestamp plus a surrounding window of `since_s` seconds.")
    since_s: float = Field(60.0, description="Window size (seconds) used for both modes. When `anchor_ts` is set, frames within `since_s` of the anchor are returned. When `query` is set, only memory entries within the last `since_s` seconds are searched.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "when did the forklift enter zone 3",
                "anchor_ts": None,
                "since_s": 60,
            }
        }
    }


class ReplayEntry(BaseModel):
    ts: float = Field(..., description="Wall-clock epoch seconds of the snapshot.")
    frame_id: int = Field(..., description="Frame id of the snapshot.")
    scene_description: str = Field(..., description="Auto-caption at that moment.")
    physical_assessment: str = Field(..., description="Physical-reasoner output at that moment.")
    detection_count: int = Field(..., description="Number of tracked objects.")
    detection_classes: List[str] = Field(..., description="Class names of every tracked object at that moment.")


class ReplayHit(ReplayEntry):
    similarity: float = Field(..., description="Cosine similarity to the query embedding, 0.0-1.0.")


class ReplayAnchorResponse(BaseModel):
    anchor: ReplayEntry = Field(..., description="The snapshot closest to `anchor_ts`.")
    frames: List[ReplayEntry] = Field(..., description="Window of surrounding frames (oldest first, capped at 30).")
    query: str = Field("", description="Echo of the request's `query` field, if any.")


class ReplaySearchResponse(BaseModel):
    query: str = Field(..., description="Echo of the request.")
    hits: List[ReplayHit] = Field(..., description="Top-k semantic matches ordered by similarity, highest first.")


def _entry_dict(e) -> dict:
    return {
        "ts": e.ts,
        "frame_id": e.frame_id,
        "scene_description": e.scene_description,
        "physical_assessment": e.physical_assessment,
        "detection_count": e.detection_count,
        "detection_classes": e.detection_classes,
    }


@router.post(
    "",
    response_model=Union[ReplayAnchorResponse, ReplaySearchResponse],
    summary="Temporal replay around an anchor or semantic query",
    description=(
        "Two modes:\n\n"
        "**Anchor mode** — set `anchor_ts` to fetch the snapshot closest "
        "to that timestamp plus a window of surrounding frames. Returns "
        "a `ReplayAnchorResponse`.\n\n"
        "**Search mode** — set `query` (no `anchor_ts`) to run semantic "
        "search over the last `since_s` seconds of scene memory. Returns "
        "a `ReplaySearchResponse` with top-k hits."
    ),
)
async def replay(
    body: ReplayRequest,
    handle: PipelineHandle = Depends(require_auth),
) -> dict:
    if handle.scene_memory is None:
        raise HTTPException(503, "scene memory not configured")

    if body.anchor_ts is not None:
        anchor = handle.scene_memory.at(float(body.anchor_ts))
        if anchor is None:
            raise HTTPException(404, "no entry at that anchor")
        frames = handle.scene_memory.range(since_s=body.since_s, max_entries=30)
        return {
            "anchor": _entry_dict(anchor),
            "frames": [_entry_dict(e) for e in frames],
            "query": body.query or "",
        }

    if not body.query:
        raise HTTPException(400, "missing 'query' or 'anchor_ts'")
    hits = handle.scene_memory.find(body.query, since_s=body.since_s, top_k=5)
    return {
        "query": body.query,
        "hits": [
            {"similarity": h.similarity, **_entry_dict(h.entry)} for h in hits
        ],
    }
