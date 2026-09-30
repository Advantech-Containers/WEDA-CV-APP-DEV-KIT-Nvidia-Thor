"""GET /v1/info — build/runtime metadata for integration teams.

Returns version (from ``WEDA_BUILD_VERSION`` env or ``"dev"``), git
SHA (from ``WEDA_GIT_SHA`` if injected), runtime fingerprint, feature
flags, and canonical doc URLs.
"""

from __future__ import annotations

import os
import platform
import time
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.v1._deps import PipelineHandle, require_handle

router = APIRouter(tags=["meta"])

_START_TS = time.time()


class FeatureFlags(BaseModel):
    """What's wired up and ready to serve."""
    vlm: bool = Field(..., description="Heavy VLM (Nemotron-VL or fallback) loaded and ready.")
    physical_reasoner: bool = Field(..., description="Cosmos-Reason2-2B loaded and ready.")
    embedder: bool = Field(..., description="Cosmos-Embed1 loaded and ready.")
    scene_memory: bool = Field(..., description="Temporal scene-memory ring buffer active.")
    fact_checker: bool = Field(..., description="Sensor-grounded claim verification available.")
    predictive_safety: bool = Field(..., description="Pairwise collision-prediction monitor active.")


class InfoResponse(BaseModel):
    name: str = Field(..., description="Service name.")
    version: str = Field(..., description="Build version string.")
    git_sha: Optional[str] = Field(default=None, description="Git commit SHA, when injected at build.")
    api_version: str = Field(..., description="REST API version (path prefix).")
    started_ts: float = Field(..., description="Process start, epoch seconds.")
    uptime_s: float = Field(..., description="Seconds since process start.")
    runtime: dict = Field(..., description="Python + platform fingerprint.")
    features: FeatureFlags
    docs: dict = Field(
        ...,
        description=(
            "URLs the integrator should bookmark — OpenAPI spec, Swagger UI, "
            "ReDoc, and the integrator-facing API.md."
        ),
    )


@router.get(
    "/info",
    response_model=InfoResponse,
    summary="Service metadata",
    description=(
        "Returns build version, git SHA, runtime fingerprint, feature "
        "flags, and the canonical doc URLs. Integration teams should "
        "call this once at startup to confirm they're hitting the "
        "expected build."
    ),
)
async def info(handle: PipelineHandle = Depends(require_handle)) -> InfoResponse:
    sr = handle.scene_reasoner
    pr = handle.physical_reasoner
    em = handle.embedder
    return InfoResponse(
        name="weda-cv-vlm",
        version=os.environ.get("WEDA_BUILD_VERSION", "dev"),
        git_sha=os.environ.get("WEDA_GIT_SHA"),
        api_version="v1",
        started_ts=_START_TS,
        uptime_s=time.time() - _START_TS,
        runtime={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        features=FeatureFlags(
            vlm=bool(getattr(sr, "_loaded", False)) if sr else False,
            physical_reasoner=bool(getattr(pr, "_loaded", False)) if pr else False,
            embedder=bool(getattr(em, "is_loaded", False)) if em else False,
            scene_memory=handle.scene_memory is not None,
            fact_checker=True,    # always available — pure regex + sensor reads
            predictive_safety=handle.predictive_safety is not None,
        ),
        docs={
            "openapi": "/v1/openapi.json",
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "integrator_guide": "/API.md",
        },
    )
