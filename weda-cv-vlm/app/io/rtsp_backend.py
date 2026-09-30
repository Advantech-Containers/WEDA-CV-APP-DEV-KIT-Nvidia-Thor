"""RTSP stream backend via PyAV with auto-reconnect."""

from __future__ import annotations

import re
import time
from typing import Optional

import cv2
import numpy as np

from app.io.camera_backend import CameraBackend
from app.utils.data_structures import FramePacket, PipelineConfig
from app.utils.logger import get_logger

logger = get_logger()

try:
    import av
    _HAS_AV = True
except ImportError:
    _HAS_AV = False


def _is_rtsp(source: str) -> bool:
    return source.lower().startswith("rtsp://")


def _sanitize_url(url: str) -> str:
    return re.sub(r"://[^@]+@", "://<credentials>@", url)


class RTSPBackend(CameraBackend):
    def __init__(self, config: PipelineConfig, pool=None) -> None:
        super().__init__(config, pool)
        self._container = None
        self._stream = None
        self._reconnect_count = 0

    def open(self) -> bool:
        if not _HAS_AV:
            logger.error("PyAV not installed — RTSP unavailable")
            return False
        return self._connect()

    def read(self) -> Optional[FramePacket]:
        if self._container is None:
            return None
        self._frame_tokens = []
        packet = FramePacket(frame_id=self._next_frame_id())
        try:
            for frame in self._container.decode(video=0):
                arr = frame.to_ndarray(format="rgb24")
                buf = self._acquire_buffer("frame_rgb")
                buf_np = self._as_numpy(buf)
                if arr.shape[:2] == buf_np.shape[:2]:
                    np.copyto(buf_np, arr)
                else:
                    np.copyto(buf_np, cv2.resize(arr, (self._config.width, self._config.height)))
                packet.rgb = buf
                return packet
            return None
        except Exception as e:
            logger.warning("RTSP read error: %s — reconnecting", e)
            if self._reconnect():
                return self.read()
            return None

    def close(self) -> None:
        if self._container:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None

    @property
    def has_depth(self) -> bool:
        return False

    def _connect(self) -> bool:
        safe = _sanitize_url(self._config.source)
        options = {
            "rtsp_transport": self._config.rtsp_protocol,
            "stimeout": str(int(self._config.rtsp_timeout * 1_000_000)),
        }
        try:
            self._container = av.open(self._config.source, options=options, timeout=self._config.rtsp_timeout)
            self._stream = self._container.streams.video[0]
            self._stream.thread_type = "AUTO"
            logger.info("RTSP connected: %s (%dx%d)", safe,
                        self._stream.codec_context.width, self._stream.codec_context.height)
            return True
        except Exception as e:
            logger.error("RTSP failed: %s — %s", safe, e)
            return False

    def _reconnect(self) -> bool:
        for attempt in range(1, self._config.rtsp_reconnect_attempts + 1):
            backoff = min(2 ** attempt, 32)
            logger.warning("RTSP reconnect %d/%d (%.0fs)", attempt, self._config.rtsp_reconnect_attempts, backoff)
            self.close()
            time.sleep(backoff)
            if self._connect():
                self._reconnect_count += 1
                return True
        return False
