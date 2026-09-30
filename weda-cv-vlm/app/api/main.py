"""FastAPI launcher.

Runs in a daemon thread spawned by the orchestrator. The pipeline
populates a singleton ``PipelineHandle`` with live references to the
SceneReasoner / PhysicalReasoner / Embedder / GpuMonitor / etc.;
endpoints read from the handle.

Endpoints are read-only with respect to pipeline state — they never
mutate the FramePacket. Inference is triggered only via explicit
``/v1/vlm/*`` and ``/v1/physical/*`` calls, which go through the
async-submit + per-model generation lock.
"""

from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

from app.utils.logger import get_logger

logger = get_logger()


@dataclass
class PipelineHandle:
    """Live references the API endpoints read from. The orchestrator
    holds the same reference and keeps mutating the members; endpoints
    must read or call methods only — never replace references."""
    latest_packet: Any = None         # FramePacket
    latest_canvas: Any = None         # np.ndarray — annotated canvas from render thread
    scene_reasoner: Any = None        # SceneReasoner
    physical_reasoner: Any = None     # PhysicalReasoner
    embedder: Any = None              # Embedder
    gpu_monitor: Any = None           # GpuMonitor
    scene_memory: Any = None          # SceneMemory
    predictive_safety: Any = None     # PredictiveSafetyMonitor
    alarm_engine: Any = None          # AlarmEngine
    query_engine: Any = None          # QueryEngine
    config: Any = None                # PipelineConfig
    api_token: Optional[str] = None   # Bearer auth — None disables it
    bind_host: str = "127.0.0.1"
    bind_port: int = 9180


_HANDLE: Optional[PipelineHandle] = None
_SERVER_THREAD: Optional[threading.Thread] = None
_SERVER: Any = None    # uvicorn.Server, set when running
_SERVER_RUNNING: bool = False


def get_handle() -> Optional[PipelineHandle]:
    """Return the singleton handle, populated by ``configure()``."""
    return _HANDLE


def _wire_bearer_security_scheme(app: "FastAPI") -> None:
    """Inject the bearer-auth scheme + Problem Details response into the
    generated OpenAPI spec (OpenAPI 3.1 § 4.8.28). Server-side
    enforcement lives in ``v1/_deps.require_auth``."""
    from fastapi.openapi.utils import get_openapi

    def _custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            summary=getattr(app, "summary", None),
            description=app.description,
            routes=app.routes,
            contact=app.contact,
            license_info=app.license_info,
        )
        schema.setdefault("components", {})
        schema["components"].setdefault("securitySchemes", {})
        schema["components"]["securitySchemes"]["bearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "opaque",
            "description": (
                "RFC 6750 bearer token. Pass via "
                "`Authorization: Bearer <WEDA_API_TOKEN>`. Set the "
                "token server-side via the `WEDA_API_TOKEN` env var. "
                "When unset, write endpoints are open (dev mode)."
            ),
        }
        # Document the canonical Problem Details media type once at
        # the components level so route-level docs can reference it.
        schema["components"].setdefault("responses", {})
        schema["components"]["responses"]["Problem"] = {
            "description": "RFC 9457 Problem Details",
            "content": {
                "application/problem+json": {
                    "schema": {"$ref": "#/components/schemas/Problem"}
                }
            },
        }
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = _custom_openapi   # type: ignore[assignment]


def configure(handle: PipelineHandle) -> None:
    """Install the singleton handle. Call once during orchestrator boot."""
    global _HANDLE
    _HANDLE = handle


def build_app() -> "FastAPI":
    """Construct the FastAPI app with all v1 routes mounted.

    OpenAPI spec at ``/v1/openapi.json``, Swagger UI at ``/docs``,
    ReDoc at ``/redoc``. Errors follow RFC 9457 (Problem Details);
    auth follows RFC 6750 (HTTP Bearer)."""
    if not _HAS_FASTAPI:
        raise RuntimeError(
            "FastAPI not installed. Add fastapi + uvicorn[standard] + "
            "sse-starlette to entrypoint.sh"
        )
    app = FastAPI(
        title="WEDA Vision Tool API",
        summary="Industrial RGB+Depth+VLM platform on Jetson AGX Thor.",
        description=(
            "Stable REST + SSE + WebSocket surface exposing live scene "
            "perception, 3-D spatial primitives, grounded VLM and "
            "physical-reasoning inference, scene memory, fact-checking, "
            "and live telemetry.\n\n"
            "**Error responses** follow [RFC 9457 Problem Details]"
            "(https://datatracker.ietf.org/doc/html/rfc9457) with "
            "`Content-Type: application/problem+json`.\n\n"
            "**Authentication** is HTTP bearer per [RFC 6750]"
            "(https://datatracker.ietf.org/doc/html/rfc6750). Set "
            "`WEDA_API_TOKEN` server-side and pass "
            "`Authorization: Bearer <token>` on every write-side "
            "request. Read endpoints are unauthenticated by default.\n\n"
            "See [`API.md`](/API.md) for the integrator guide."
        ),
        version=os.environ.get("WEDA_BUILD_VERSION", "dev"),
        contact={"name": "WEDA Edge", "email": "weda-edge@example.com"},
        license_info={"name": "Proprietary"},
        openapi_url="/v1/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Permissive CORS — co-tenant agents (NemoClaw, etc.) hit the API
    # from the same host, but a partner team's tooling on the plant
    # LAN should also be able to connect. Bearer token guards
    # write-side actions.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["WWW-Authenticate"],
    )

    # RFC 9457 error envelope handlers.
    from app.api.errors import install_handlers
    install_handlers(app)

    # request_id propagation, structured access log, concurrency cap,
    # token-bucket rate limit. Order is enforced inside install().
    from app.api.middleware import install as install_middleware
    install_middleware(app)

    # Inject the bearer-auth security scheme into the OpenAPI spec so
    # Swagger UI shows the "Authorize" button and integrators see how
    # to authenticate (OpenAPI 3.1 § 4.8.28).
    _wire_bearer_security_scheme(app)

    _mounted: list[str] = []
    _failed: list[tuple[str, str]] = []
    for _name in (
        "info", "health", "scene", "spatial", "vlm", "physical",
        "embeddings", "factcheck", "replay", "alerts", "alarms",
        "stream", "telemetry",
    ):
        try:
            _mod = __import__(f"app.api.v1.{_name}", fromlist=["router"])
            app.include_router(_mod.router, prefix="/v1")
            _mounted.append(_name)
        except Exception as _e:
            _failed.append((_name, f"{type(_e).__name__}: {_e}"))
            logger.error("[API] router '%s' failed to mount: %s", _name, _e)

    logger.info("[API] %d/%d routers mounted: %s",
                len(_mounted), len(_mounted) + len(_failed),
                ", ".join(_mounted))
    if _failed:
        logger.warning("[API] %d routers failed: %s",
                       len(_failed),
                       "; ".join(f"{n}: {e}" for n, e in _failed))

    return app


def start_in_thread(handle: PipelineHandle) -> None:
    """Boot uvicorn.Server in a daemon thread; idempotent.

    Holds the Server instance in module state so ``shutdown()`` can
    flip ``should_exit`` for graceful termination.
    """
    global _SERVER_THREAD, _SERVER, _SERVER_RUNNING
    if _SERVER_RUNNING:
        return
    if not _HAS_FASTAPI:
        logger.warning("FastAPI not installed — HTTP API disabled")
        return
    configure(handle)
    _SERVER_RUNNING = True

    def _run() -> None:
        global _SERVER
        try:
            import uvicorn
        except ImportError:
            logger.error("uvicorn not installed — HTTP API disabled")
            return
        try:
            app = build_app()
        except Exception as e:
            logger.error("FastAPI app build failed: %s", e)
            return
        config = uvicorn.Config(
            app,
            host=handle.bind_host,
            port=handle.bind_port,
            log_level="warning",
            access_log=False,
            timeout_keep_alive=5,
            lifespan="on",
        )
        _SERVER = uvicorn.Server(config)
        try:
            _SERVER.run()
        except Exception as e:
            logger.error("uvicorn failed: %s", e)

    _SERVER_THREAD = threading.Thread(
        target=_run, daemon=True, name="weda-api-server",
    )
    _SERVER_THREAD.start()
    logger.info(
        "[API] Server starting on http://%s:%d/v1/* (REST/SSE/WS); docs at /docs",
        handle.bind_host, handle.bind_port,
    )


def shutdown(timeout_s: float = 5.0) -> None:
    """Graceful stop. Flips ``server.should_exit`` and joins the
    serving thread within the timeout. Safe to call when the server
    never started."""
    global _SERVER, _SERVER_THREAD, _SERVER_RUNNING
    server = _SERVER
    if server is not None:
        try:
            server.should_exit = True
        except Exception:
            pass
    th = _SERVER_THREAD
    if th is not None and th.is_alive():
        th.join(timeout=timeout_s)
    _SERVER_RUNNING = False
    _SERVER = None
    _SERVER_THREAD = None
