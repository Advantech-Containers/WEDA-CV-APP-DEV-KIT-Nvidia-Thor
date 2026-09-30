"""GET /v1/embeddings/text and POST /v1/embeddings/crops."""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.v1._deps import PipelineHandle, require_handle

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


class TextEmbedResponse(BaseModel):
    vec: Optional[List[float]] = Field(None, description="L2-normalised embedding vector for the query. Null when the embedder failed to produce one.")
    dim: int = Field(0, description="Embedding dimension (e.g. 1024 for Cosmos-Embed1).")


class CropsEmbedRequest(BaseModel):
    track_ids: List[int] = Field(..., description="Track ids on the current frame whose crops should be embedded.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "track_ids": [1, 2, 3],
            }
        }
    }


class CropsEmbedResponse(BaseModel):
    embeddings: Dict[str, Optional[List[float]]] = Field(..., description='Map from `str(track_id)` to embedding vector. Null when the crop was too small (< 8 px) or the track id was not present on the current frame.')


@router.get(
    "/text",
    response_model=TextEmbedResponse,
    summary="Cosmos-Embed1 text embedding",
    description=(
        "L2-normalised embedding vector for the supplied query. Cached "
        "server-side (LRU 200) so repeated identical queries are free. "
        "Returns 503 when the embedder failed to load."
    ),
)
async def embed_text(q: str, handle: PipelineHandle = Depends(require_handle)) -> TextEmbedResponse:
    em = handle.embedder
    if em is None or not getattr(em, "is_loaded", False):
        raise HTTPException(503, "embedder not loaded")
    try:
        vec = em.embed_text(q)
    except Exception as e:
        raise HTTPException(500, f"embed failed: {e!r}")
    if vec is None:
        return TextEmbedResponse(vec=None, dim=int(getattr(em, "embedding_dim", 0)))
    return TextEmbedResponse(
        vec=[float(x) for x in vec.ravel().tolist()],
        dim=int(getattr(em, "embedding_dim", 0)),
    )


@router.post(
    "/crops",
    response_model=CropsEmbedResponse,
    summary="Batched crop embeddings for tracks",
    description=(
        "Single batched GPU forward pass over crops of the requested "
        "tracks on the current frame. Crops smaller than 8 px or whose "
        "track id is not in the current frame return null."
    ),
)
async def embed_crops(
    body: CropsEmbedRequest,
    handle: PipelineHandle = Depends(require_handle),
) -> CropsEmbedResponse:
    em = handle.embedder
    if em is None or not getattr(em, "is_loaded", False):
        raise HTTPException(503, "embedder not loaded")
    if handle.latest_packet is None:
        raise HTTPException(503, "no frame yet")
    pkt = handle.latest_packet
    dets_for = [d for d in (pkt.detections or []) if d.track_id in set(body.track_ids)]
    if not dets_for:
        return CropsEmbedResponse(embeddings={})
    try:
        embs = em.embed_crops(pkt.rgb, dets_for)
    except Exception as e:
        raise HTTPException(500, f"embed_crops failed: {e!r}")
    out: Dict[str, Optional[List[float]]] = {}
    for d, v in zip(dets_for, embs):
        out[str(d.track_id)] = None if v is None else [float(x) for x in v.ravel().tolist()]
    return CropsEmbedResponse(embeddings=out)
