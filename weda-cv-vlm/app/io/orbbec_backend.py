"""Orbbec SDK backend — unified RGB + depth from single pipeline.

Wraps OrbbecHandler to conform to CameraBackend interface.
Provides both RGB + hardware-aligned depth from one wait_for_frames() call.
"""

from __future__ import annotations

from typing import Optional

from app.io.camera_backend import CameraBackend
from app.io.orbbec_handler import OrbbecHandler, is_orbbec_available
from app.utils.data_structures import FramePacket, PipelineConfig
from app.utils.logger import get_logger

logger = get_logger()


class OrbbecBackend(CameraBackend):
    """Full Orbbec camera: RGB + depth from SDK pipeline."""

    def __init__(self, config: PipelineConfig, pool=None) -> None:
        super().__init__(config, pool)
        self._handler = OrbbecHandler(config, pool)

    def open(self) -> bool:
        if not is_orbbec_available():
            logger.error("pyorbbecsdk2 not installed")
            return False
        return self._handler.open()

    def read(self) -> Optional[FramePacket]:
        return self._handler.read()

    def close(self) -> None:
        self._handler.close()

    def pop_frame_tokens(self) -> list[tuple[str, int]]:
        return self._handler.pop_frame_tokens()

    @property
    def has_depth(self) -> bool:
        return True

    @property
    def camera_intrinsics(self) -> Optional[tuple]:
        """Returns (fx, fy, cx, cy, depth_scale) from Orbbec SDK after open()."""
        return self._handler.camera_intrinsics

    @property
    def backend_name(self) -> str:
        return "Orbbec SDK"
