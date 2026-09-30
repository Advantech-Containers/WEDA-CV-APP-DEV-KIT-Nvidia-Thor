"""RFC 9457 Problem Details for HTTP APIs — error envelope.

This module declares the canonical error shape every endpoint emits.
Specification reference: https://datatracker.ietf.org/doc/html/rfc9457
(supersedes RFC 7807; the wire format is identical, the registry
guidance is new in 9457).

Wire format::

    HTTP/1.1 404 Not Found
    Content-Type: application/problem+json

    {
      "type":     "https://weda.local/errors/track-not-found",
      "title":    "Track not found",
      "status":   404,
      "detail":   "No track with id 42 in the current frame.",
      "instance": "/v1/objects/42"
    }

Custom extensions are permitted (RFC 9457 § 3.2). We add
``request_id`` for log correlation and ``code`` as a stable
machine-readable identifier independent of the human-readable
``title``.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

PROBLEM_BASE = "https://weda.local/errors/"
PROBLEM_JSON = "application/problem+json"


class Problem(BaseModel):
    """RFC 9457 Problem Details document.

    Every error response from the WEDA API conforms to this shape.
    Returned with ``Content-Type: application/problem+json`` and the
    HTTP ``status`` code mirroring the ``status`` field.
    """

    type: str = Field(
        default=f"{PROBLEM_BASE}about:blank",
        description=(
            "URI reference identifying the problem type. Defaults to "
            "``about:blank`` per RFC 9457 § 3.1."
        ),
        json_schema_extra={"example": f"{PROBLEM_BASE}track-not-found"},
    )
    title: str = Field(
        ...,
        description=(
            "Short, human-readable summary of the problem type. "
            "SHOULD NOT change between occurrences (RFC 9457 § 3.1.1)."
        ),
        json_schema_extra={"example": "Track not found"},
    )
    status: int = Field(
        ...,
        description="HTTP status code mirroring the response (RFC 9457 § 3.1.2).",
        json_schema_extra={"example": 404},
    )
    detail: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of THIS occurrence (RFC 9457 § 3.1.3).",
        json_schema_extra={"example": "No track with id 42 in the current frame."},
    )
    instance: Optional[str] = Field(
        default=None,
        description=(
            "URI reference identifying the specific occurrence — typically the "
            "request path that produced the error (RFC 9457 § 3.1.4)."
        ),
        json_schema_extra={"example": "/v1/objects/42"},
    )
    # Custom extensions (RFC 9457 § 3.2 — names ≥ 3 chars, alphanumeric + _).
    code: Optional[str] = Field(
        default=None,
        description=(
            "Stable machine-readable error code, independent of ``title``. "
            "Use this for client-side branching."
        ),
        json_schema_extra={"example": "track_not_found"},
    )
    request_id: Optional[str] = Field(
        default=None,
        description="UUID for log correlation across services.",
        json_schema_extra={"example": "8c7d49d2c2b94c8b"},
    )


def problem_response(
    status_code: int,
    title: str,
    detail: Optional[str] = None,
    code: Optional[str] = None,
    type_suffix: str = "about:blank",
    request: Optional[Request] = None,
) -> JSONResponse:
    """Build an RFC 9457 ``application/problem+json`` response."""
    instance = request.url.path if request is not None else None
    request_id = uuid.uuid4().hex[:16]
    body = Problem(
        type=PROBLEM_BASE + type_suffix,
        title=title,
        status=status_code,
        detail=detail,
        instance=instance,
        code=code,
        request_id=request_id,
    ).model_dump(exclude_none=True)
    headers = {"Content-Type": PROBLEM_JSON}
    return JSONResponse(status_code=status_code, content=body, headers=headers)


def install_handlers(app: FastAPI) -> None:
    """Replace FastAPI's default error formatting with RFC 9457 envelopes.

    Three handlers cover the common failure modes:

    1. ``HTTPException`` raised by route code (manual ``raise HTTPException``).
    2. ``RequestValidationError`` from Pydantic when input doesn't match.
    3. Generic ``Exception`` — last-resort 500 handler.

    The auth dependency in ``app/api/v1/_deps.py`` raises
    ``HTTPException(401)``; on that path we additionally emit the
    ``WWW-Authenticate: Bearer ...`` challenge as required by RFC 6750
    § 3 so clients see a proper bearer-auth challenge.
    """
    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(HTTPException)
    async def _http_exc_handler(request: Request, exc: HTTPException) -> JSONResponse:
        # RFC 6750 § 3 — emit WWW-Authenticate on 401s.
        # Format: Bearer realm="weda", error="invalid_token", ...
        extra_headers = {}
        if exc.status_code == status.HTTP_401_UNAUTHORIZED:
            error_code = "invalid_token"
            if exc.detail and "missing" in str(exc.detail).lower():
                error_code = "invalid_request"
            extra_headers["WWW-Authenticate"] = (
                f'Bearer realm="weda", error="{error_code}"'
            )
        title = _http_title(exc.status_code)
        resp = problem_response(
            status_code=exc.status_code,
            title=title,
            detail=str(exc.detail) if exc.detail else None,
            code=_default_code(exc.status_code),
            type_suffix=_default_code(exc.status_code),
            request=request,
        )
        for k, v in extra_headers.items():
            resp.headers[k] = v
        return resp

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(
        request: Request, exc: RequestValidationError,
    ) -> JSONResponse:
        return problem_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            title="Request validation failed",
            detail="; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            ),
            code="validation_error",
            type_suffix="validation-error",
            request=request,
        )

    @app.exception_handler(Exception)
    async def _generic_handler(request: Request, exc: Exception) -> JSONResponse:
        # Don't leak the raw exception message — log it server-side, return
        # a generic 500 to the client. Test/dev mode could be different.
        from app.utils.logger import get_logger
        get_logger().exception("Unhandled exception in %s", request.url.path)
        return problem_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            title="Internal Server Error",
            detail="An internal error occurred. Reference the request_id in the server logs.",
            code="internal_error",
            type_suffix="internal-error",
            request=request,
        )


def _http_title(code: int) -> str:
    """Map HTTP status codes to canonical RFC 7231 reason phrases."""
    titles = {
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        409: "Conflict",
        413: "Payload Too Large",
        415: "Unsupported Media Type",
        422: "Unprocessable Entity",
        429: "Too Many Requests",
        500: "Internal Server Error",
        501: "Not Implemented",
        503: "Service Unavailable",
        504: "Gateway Timeout",
    }
    return titles.get(code, f"HTTP {code}")


def _default_code(code: int) -> str:
    """Stable machine-readable identifier for the error code field."""
    return {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
        429: "too_many_requests",
        500: "internal_error",
        501: "not_implemented",
        503: "service_unavailable",
        504: "gateway_timeout",
    }.get(code, f"http_{code}")
