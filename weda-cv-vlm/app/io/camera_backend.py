"""Abstract camera backend interface.

All camera sources (V4L2, Orbbec, RealSense, ZED, RTSP, GStreamer,
synthetic) implement this interface. Swap cameras via config — no
code changes needed.

To add a new camera:
  1. Create app/io/mycamera_backend.py
  2. Inherit CameraBackend, implement open/read/close/has_depth
  3. Register in camera_factory.py BACKENDS dict
  4. Set input.backend: "mycamera" in pipeline.yaml
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from app.utils.data_structures import FramePacket, PipelineConfig
from app.utils.logger import get_logger

logger = get_logger()


class CameraBackend(ABC):
    """Abstract base for all camera input sources."""

    def __init__(self, config: PipelineConfig, pool=None) -> None:
        self._config = config
        self._pool = pool
        self._frame_id = 0
        self._frame_tokens: list[tuple[str, int]] = []

    @abstractmethod
    def open(self) -> bool:
        """Initialize the camera/source. Return True if successful."""

    @abstractmethod
    def read(self) -> Optional[FramePacket]:
        """Acquire next frame. Return FramePacket or None."""

    @abstractmethod
    def close(self) -> None:
        """Cleanup resources."""

    @property
    @abstractmethod
    def has_depth(self) -> bool:
        """Whether depth frames are available."""

    @property
    def depth_index(self) -> Optional[int]:
        """V4L2 device index for depth (None for non-V4L2 backends)."""
        return None

    @property
    def camera_intrinsics(self) -> Optional[tuple]:
        """Return (fx, fy, cx, cy, depth_scale) from camera SDK, or None.

        depth_scale: raw depth unit → mm multiplier (usually 1.0 for Orbbec).
        When non-None, DepthProcessor uses these instead of its defaults.
        """
        return None

    @property
    def backend_name(self) -> str:
        """Human-readable backend name for display."""
        return self.__class__.__name__

    def pop_frame_tokens(self) -> list[tuple[str, int]]:
        """Return buffer pool tokens from last read() and clear."""
        tokens = self._frame_tokens
        self._frame_tokens = []
        return tokens

    def _acquire_buffer(self, pool_name: str):
        """Acquire from pool or allocate fresh numpy array."""
        if self._pool:
            try:
                idx, buf = self._pool.acquire(pool_name)
                self._frame_tokens.append((pool_name, idx))
                return buf
            except KeyError:
                pass
        h, w = self._config.height, self._config.width
        if pool_name == "frame_depth":
            return np.zeros((h, w), dtype=np.uint16)
        return np.zeros((h, w, 3), dtype=np.uint8)

    def _next_frame_id(self) -> int:
        self._frame_id += 1
        return self._frame_id

    @staticmethod
    def _as_numpy(buf) -> np.ndarray:
        if isinstance(buf, np.ndarray):
            return buf
        try:
            return buf.numpy()
        except Exception:
            return np.asarray(buf)
