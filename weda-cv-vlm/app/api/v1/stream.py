"""GET /v1/stream/rgb.mjpg, /v1/stream/depth.mjpg — MJPEG live feeds.

Serves the camera-native RGB frame with bbox annotations drawn
server-side; consumers add their own chrome. cv2 work runs in
``loop.run_in_executor`` so JPEG encoding doesn't block the asyncio
loop. The frame is copied before draw because ``packet.rgb`` is a
pool-managed buffer. Encode-once cache is keyed on ``frame_id`` so
multiple subscribers share the encoded bytes.
"""

from __future__ import annotations

import asyncio
import threading
from typing import AsyncIterator, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from app.api.v1._deps import PipelineHandle, require_handle

router = APIRouter(prefix="/stream", tags=["stream"])

_BOUNDARY = "weda-frame"
_DEFAULT_FPS_CAP = 10
_DEFAULT_JPEG_Q = 70

_CLASS_PALETTE = (
    (255, 200, 0), (50, 100, 255), (180, 180, 180),
    (0, 200, 83), (0, 165, 255), (255, 0, 200),
    (200, 200, 0), (0, 255, 0),
)


def _config_caps(handle) -> tuple[int, int]:
    cfg = getattr(handle, "config", None)
    fps = int(getattr(cfg, "stream_mjpeg_fps_cap", _DEFAULT_FPS_CAP))
    q = int(getattr(cfg, "stream_jpeg_quality", _DEFAULT_JPEG_Q))
    return max(1, fps), max(1, min(100, q))


class _FrameCache:
    """Thread-safe last-encoded JPEG cache keyed on frame_id."""
    __slots__ = ("_lock", "_frame_id", "_payload")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame_id: int = -1
        self._payload: Optional[bytes] = None

    def get(self, frame_id: int) -> Optional[bytes]:
        with self._lock:
            if frame_id == self._frame_id:
                return self._payload
        return None

    def put(self, frame_id: int, payload: bytes) -> None:
        with self._lock:
            self._frame_id = frame_id
            self._payload = payload


_RGB_CACHE = _FrameCache()
_RGB_RAW_CACHE = _FrameCache()
_DEPTH_CACHE = _FrameCache()


def _annotate_rgb(frame: np.ndarray, detections: list, q: int) -> Optional[bytes]:
    """Draw bboxes + class labels on a copy of the camera frame, encode JPEG."""
    if frame is None or not _HAS_CV2:
        return None
    try:
        annotated = np.ascontiguousarray(frame).copy()
    except Exception:
        return None

    for d in detections or []:
        try:
            x1, y1, x2, y2 = (int(v) for v in d.bbox.tolist())
        except Exception:
            continue
        cls_id = int(getattr(d, "class_id", 0))
        color = _CLASS_PALETTE[cls_id % len(_CLASS_PALETTE)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{getattr(d, 'class_name', '')} {getattr(d, 'confidence', 0.0):.2f}"
        if d.track_id >= 0:
            label = f"T{d.track_id} {label}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly = max(0, y1 - 4)
        cv2.rectangle(annotated, (x1, ly - th - 4), (x1 + tw + 6, ly), color, -1)
        cv2.putText(annotated, label, (x1 + 3, ly - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 12, 16), 1)

    ok, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return jpg.tobytes() if ok else None


def _encode_blocking(frame: np.ndarray, q: int) -> Optional[bytes]:
    if frame is None or not _HAS_CV2:
        return None
    try:
        snap = np.ascontiguousarray(frame).copy()
    except Exception:
        return None
    ok, jpg = cv2.imencode(".jpg", snap, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return jpg.tobytes() if ok else None


async def _mjpeg_loop(
    get_packet,
    cache: _FrameCache,
    extract,
    encode_fn,
    max_fps: int,
    jpeg_q: int,
) -> AsyncIterator[bytes]:
    if not _HAS_CV2:
        raise HTTPException(500, "opencv-python not installed")
    period = 1.0 / max_fps
    loop = asyncio.get_event_loop()
    while True:
        try:
            packet = get_packet()
            if packet is None:
                await asyncio.sleep(period)
                continue
            frame_id = int(getattr(packet, "frame_id", 0))
            payload = cache.get(frame_id)
            if payload is None:
                args = extract(packet)
                if args is None:
                    await asyncio.sleep(period)
                    continue
                payload = await loop.run_in_executor(
                    None, encode_fn, *args, jpeg_q,
                )
                if payload is None:
                    await asyncio.sleep(period)
                    continue
                cache.put(frame_id, payload)
            yield (
                f"--{_BOUNDARY}\r\n"
                f"Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(payload)}\r\n\r\n"
            ).encode("ascii") + payload + b"\r\n"
            await asyncio.sleep(period)
        except asyncio.CancelledError:
            return
        except Exception:
            await asyncio.sleep(period)


@router.get(
    "/rgb.mjpg",
    summary="Live RGB stream (MJPEG)",
    description=(
        "Camera-native RGB MJPEG. `annotate=true` draws detection boxes; "
        "`annotate=false` serves the bare feed. Default from "
        "`streaming.draw_detections`."
    ),
    include_in_schema=False,
    responses={
        200: {
            "description": "MJPEG multipart stream. Render in a browser via `<img src=...>` or open the URL directly in VLC.",
            "content": {f"multipart/x-mixed-replace; boundary={_BOUNDARY}": {}},
        }
    },
)
async def rgb_mjpeg(
    annotate: Optional[bool] = None,
    handle: PipelineHandle = Depends(require_handle),
):
    fps, q = _config_caps(handle)
    if annotate is None:
        annotate = bool(getattr(getattr(handle, "config", None),
                                "stream_draw_detections", True))

    if annotate:
        def _extract(packet):
            if packet.rgb is None:
                return None
            # snapshot detections in the event-loop tick; heavy work runs in the executor
            return (packet.rgb, list(packet.detections or []))

        return StreamingResponse(
            _mjpeg_loop(lambda: handle.latest_packet, _RGB_CACHE,
                        _extract, _annotate_rgb, fps, q),
            media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
        )

    def _extract_raw(packet):
        return (packet.rgb,) if packet.rgb is not None else None

    return StreamingResponse(
        _mjpeg_loop(lambda: handle.latest_packet, _RGB_RAW_CACHE,
                    _extract_raw, _encode_blocking, fps, q),
        media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
    )


@router.get(
    "/depth.mjpg",
    summary="Live colorized depth stream (MJPEG)",
    description="Same wire format as `/rgb.mjpg`; carries colorized depth without overlays.",
    include_in_schema=False,
    responses={
        200: {
            "description": "MJPEG multipart stream of the colorized depth map.",
            "content": {f"multipart/x-mixed-replace; boundary={_BOUNDARY}": {}},
        }
    },
)
async def depth_mjpeg(handle: PipelineHandle = Depends(require_handle)):
    fps, q = _config_caps(handle)

    def _extract(packet):
        return (packet.depth_colorized,) if packet.depth_colorized is not None else None

    return StreamingResponse(
        _mjpeg_loop(lambda: handle.latest_packet, _DEPTH_CACHE,
                    _extract, _encode_blocking, fps, q),
        media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
    )
