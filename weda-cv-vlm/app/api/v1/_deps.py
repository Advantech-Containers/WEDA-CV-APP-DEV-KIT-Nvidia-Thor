"""Shared FastAPI dependencies — handle access + bearer-token auth."""

from __future__ import annotations

from typing import Optional

try:
    from fastapi import Depends, Header, HTTPException, status
except ImportError:  # FastAPI not installed — let import fail at use-site
    Depends = Header = HTTPException = status = None   # type: ignore

from app.api.main import PipelineHandle, get_handle


def require_handle() -> PipelineHandle:
    """Return the singleton PipelineHandle or 503 if not configured."""
    h = get_handle()
    if h is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline handle not configured (orchestrator not started).",
        )
    return h


def require_auth(
    handle: PipelineHandle = Depends(require_handle),
    authorization: Optional[str] = Header(default=None),
) -> PipelineHandle:
    """Bearer-token gate for write endpoints. When `api_token` is None,
    auth is bypassed (read-only mode for sandboxed dev)."""
    if handle.api_token is None:
        return handle
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if authorization[7:] != handle.api_token:
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    return handle
