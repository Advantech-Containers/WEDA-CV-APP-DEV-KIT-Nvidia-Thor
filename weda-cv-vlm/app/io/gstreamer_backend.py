"""GStreamer NVDEC hardware-accelerated video decode backend.

Uses nvv4l2decoder for zero-CPU-overhead video decode on Jetson Thor.
Supports: camera (v4l2src), RTSP, file (filesrc), test pattern.
Falls back to V4L2Backend if GStreamer/NVDEC unavailable.

Decode: H.264/H.265/VP9/AV1 → NVDEC hardware → NV12 → BGR numpy
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.io.camera_backend import CameraBackend
from app.utils.data_structures import FramePacket, PipelineConfig
from app.utils.logger import get_logger

logger = get_logger()

# Check GStreamer availability
_HAS_GST = False
try:
    # Test if OpenCV was built with GStreamer
    cap = cv2.VideoCapture("videotestsrc num-buffers=1 ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
    _HAS_GST = cap.isOpened()
    cap.release()
except Exception:
    pass


def is_nvdec_available() -> bool:
    """Check if NVDEC hardware decode is available."""
    if not _HAS_GST:
        return False
    try:
        cap = cv2.VideoCapture(
            "videotestsrc num-buffers=1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink",
            cv2.CAP_GSTREAMER,
        )
        ok = cap.isOpened()
        cap.release()
        return ok
    except Exception:
        return False


class GStreamerBackend(CameraBackend):
    """Hardware-accelerated camera/video input via GStreamer NVDEC."""

    def __init__(self, config: PipelineConfig, pool=None) -> None:
        super().__init__(config, pool)
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        if not _HAS_GST:
            logger.error("GStreamer not available in OpenCV build")
            return False

        source = self._config.source
        w, h = self._config.width, self._config.height

        # Build GStreamer pipeline based on source type
        try:
            source_int = int(source)
            # USB camera: don't force resolution — let camera negotiate natively
            pipeline = (
                f"v4l2src device=/dev/video{source_int} ! "
                f"videoconvert ! video/x-raw,format=BGR ! "
                f"appsink drop=1 sync=0"
            )
        except (ValueError, TypeError):
            if source.lower().startswith("rtsp://"):
                # RTSP → NVDEC hardware decode
                pipeline = (
                    f"rtspsrc location={source} latency=100 ! "
                    f"rtph264depay ! h264parse ! nvv4l2decoder ! "
                    f"nvvidconv ! video/x-raw,format=BGRx,width={w},height={h} ! "
                    f"videoconvert ! video/x-raw,format=BGR ! "
                    f"appsink drop=1 sync=0"
                )
            else:
                # File → NVDEC hardware decode
                pipeline = (
                    f"filesrc location={source} ! decodebin ! "
                    f"nvvidconv ! video/x-raw,format=BGRx,width={w},height={h} ! "
                    f"videoconvert ! video/x-raw,format=BGR ! "
                    f"appsink drop=1 sync=0"
                )

        logger.info("GStreamer pipeline: %s", pipeline[:120])
        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self._cap.isOpened():
            logger.error("GStreamer pipeline failed to open")
            return False

        logger.info("GStreamer NVDEC active: %s", self._config.source)
        return True

    def read(self) -> Optional[FramePacket]:
        if self._cap is None:
            return None
        self._frame_tokens = []
        packet = FramePacket(frame_id=self._next_frame_id())

        ret, frame = self._cap.read()
        if not ret:
            return None

        buf = self._acquire_buffer("frame_rgb")
        buf_np = self._as_numpy(buf)
        if frame.shape[:2] == buf_np.shape[:2]:
            np.copyto(buf_np, frame)
        else:
            np.copyto(buf_np, cv2.resize(frame, (self._config.width, self._config.height)))
        packet.rgb = buf
        return packet

    def close(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    @property
    def has_depth(self) -> bool:
        return False  # GStreamer decode doesn't provide depth
