"""WEDA Vision Tool — v1 API routes.

Each module here exposes a related slice of the pipeline as HTTP/SSE/
WebSocket endpoints. All write-side endpoints require the bearer token
configured in `WEDA_API_TOKEN`. Read endpoints are auth-optional by
default so a fresh integration (e.g. NemoClaw) can probe the surface
without any configuration step.
"""
