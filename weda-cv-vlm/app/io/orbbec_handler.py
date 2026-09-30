"""Orbbec SDK camera backend — synchronized RGB + depth via AlignFilter.

Alignment strategy (confirmed from pyorbbecsdk2 examples and issues #131/#187):

  HW D2C (OBAlignMode.HW_MODE) requires OBFormat.RGB on the color stream.
  MJPG is a compressed format; the firmware D2C unit cannot process it.
  Most devices advertise MJPG as their default/only color profile, so HW_MODE
  raises "Current stream profile is not support hardware d2c process".

  The correct cross-format solution is AlignFilter — a software post-processing
  filter that warps the depth frame into color FOV.  It works with any color
  format (MJPG, RGB, YUYV …) and is the documented approach in:
    - examples/align_filter_viewer.py
    - examples/sync_align.py  (v2-main)
    - GitHub issues #131, #187 (Orbbec maintainer-confirmed)

  Pipeline config: no set_align_mode() call.
  Read path: AlignFilter.process(frameset) → depth is spatially aligned.

Requires: pip install pyorbbecsdk2
Falls back gracefully if SDK not installed.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from app.utils.data_structures import FramePacket, PipelineConfig
from app.utils.logger import get_logger

logger = get_logger()

try:
    from pyorbbecsdk import (
        Pipeline, Config, OBSensorType, OBFormat,
        OBAlignMode, VideoFrame,
    )
    _HAS_ORBBEC = True
except ImportError:
    _HAS_ORBBEC = False

# AlignFilter and OBStreamType are present in pyorbbecsdk2 >= 2.x
# Import separately so the rest of the backend works even on older builds.
try:
    from pyorbbecsdk import AlignFilter, OBStreamType
    _HAS_ALIGN_FILTER = True
except ImportError:
    _HAS_ALIGN_FILTER = False


def is_orbbec_available() -> bool:
    """Check if Orbbec SDK is installed."""
    return _HAS_ORBBEC


class OrbbecHandler:
    """Unified RGB + depth capture from Orbbec cameras via pyorbbecsdk2.

    Color is opened at the camera's native default (commonly 1280×720 MJPG).
    Depth is aligned to color FOV using AlignFilter (SW post-processing).
    Output frames are resized to config.width × config.height if needed.
    """

    def __init__(self, config: PipelineConfig, pool=None) -> None:
        self._config = config
        self._pool = pool
        self._pipeline: Optional[Pipeline] = None
        self._align_filter = None          # AlignFilter instance, set after open()
        self._frame_id = 0
        self._frame_tokens: list[tuple[str, int]] = []
        self._depth_scale: float = 1.0
        self._intrinsics: Optional[tuple[float, float, float, float]] = None

    # ------------------------------------------------------------------
    # Open / close
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Initialize Orbbec pipeline with color + depth, software-aligned."""
        if not _HAS_ORBBEC:
            logger.warning("pyorbbecsdk2 not installed — Orbbec backend unavailable")
            return False

        try:
            self._pipeline = Pipeline()
            cfg = Config()

            # ── Color stream ──────────────────────────────────────────
            # Use SDK default profile (1280×720 MJPG on most devices).
            # HW D2C is NOT set — AlignFilter handles alignment in SW
            # and works with any color format including MJPG.
            color_profile = self._open_best_color_profile(cfg)
            if color_profile is None:
                logger.error("Orbbec: no usable color profile found")
                return False

            # ── Depth stream ──────────────────────────────────────────
            self._open_depth_profile(cfg)

            # ── Start pipeline (no align mode set — AlignFilter used) ─
            self._pipeline.start(cfg)

            # ── AlignFilter: warp depth to color FOV in software ──────
            if _HAS_ALIGN_FILTER:
                try:
                    self._align_filter = AlignFilter(
                        align_to_stream=OBStreamType.COLOR_STREAM
                    )
                    logger.info("Orbbec AlignFilter (SW depth→color) active")
                except Exception as e:
                    logger.warning("AlignFilter init failed: %s — depth unaligned", e)
                    self._align_filter = None
            else:
                logger.warning(
                    "AlignFilter not available in this pyorbbecsdk2 build — "
                    "depth frames will not be spatially aligned to color"
                )

            # ── Depth scale ───────────────────────────────────────────
            try:
                device = self._pipeline.get_device()
                sensor = device.get_sensor(OBSensorType.DEPTH_SENSOR)
                self._depth_scale = (
                    sensor.get_depth_scale()
                    if hasattr(sensor, "get_depth_scale") else 1.0
                )
            except Exception:
                self._depth_scale = 1.0

            # ── Camera intrinsics (color, used after alignment) ───────
            try:
                cam_param = self._pipeline.get_camera_param()
                intr = cam_param.rgb_intrinsic
                self._intrinsics = (
                    float(intr.fx), float(intr.fy),
                    float(intr.cx), float(intr.cy),
                )
                logger.info(
                    "Orbbec intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                    *self._intrinsics,
                )
            except Exception as e:
                logger.warning(
                    "Could not read Orbbec intrinsics: %s — using defaults", e
                )
                self._intrinsics = None

            logger.info(
                "Orbbec pipeline started (depth_scale=%.4f, align_filter=%s)",
                self._depth_scale, self._align_filter is not None,
            )
            return True

        except Exception as e:
            logger.error("Orbbec pipeline open failed: %s", e)
            self._pipeline = None
            return False

    def _open_best_color_profile(self, cfg: "Config"):
        """Select the best available color profile and enable it in cfg.

        Strategy (in order):
          1. RGB profile at config resolution + highest FPS  (HW D2C capable)
          2. Any valid video profile at config resolution + highest FPS
          3. Any valid video profile at highest FPS overall  (resize in read)
          4. SDK default profile                             (last resort)

        NOTE: get_video_stream_profile(i) raises for most indices on
        pyorbbecsdk2 — iterate with per-entry try/except to collect
        only the profiles the SDK can actually return.
        """
        try:
            pl = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        except Exception as e:
            logger.error("Could not list color profiles: %s", e)
            return None

        valid = []
        for i in range(pl.get_count()):
            try:
                valid.append(pl.get_video_stream_profile(i))
            except Exception:
                continue

        profile = None
        if valid:
            rgb_exact = [
                p for p in valid
                if p.get_format() == OBFormat.RGB
                and p.get_width() == self._config.width
                and p.get_height() == self._config.height
            ]
            any_exact = [
                p for p in valid
                if p.get_width() == self._config.width
                and p.get_height() == self._config.height
            ]
            pool = rgb_exact or any_exact or valid
            profile = max(pool, key=lambda p: p.get_fps())

        if profile is None:
            try:
                profile = pl.get_default_video_stream_profile()
            except Exception as e:
                logger.error("get_default_video_stream_profile failed: %s", e)
                return None

        cfg.enable_stream(profile)
        logger.info(
            "Orbbec color: %dx%d @ %d fps fmt=%s",
            profile.get_width(), profile.get_height(),
            profile.get_fps(), profile.get_format(),
        )
        return profile

    def _open_depth_profile(self, cfg: "Config") -> None:
        """Enable the best available depth profile in cfg."""
        try:
            pl = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

            valid = []
            for i in range(pl.get_count()):
                try:
                    valid.append(pl.get_video_stream_profile(i))
                except Exception:
                    continue

            profile = (
                max(valid, key=lambda p: p.get_fps())
                if valid
                else pl.get_default_video_stream_profile()
            )
            cfg.enable_stream(profile)
            logger.info(
                "Orbbec depth: %dx%d @ %d fps fmt=%s",
                profile.get_width(), profile.get_height(),
                profile.get_fps(), profile.get_format(),
            )
        except Exception as e:
            logger.warning("Orbbec depth stream failed: %s — RGB only", e)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self) -> Optional[FramePacket]:
        """Read synchronized RGB + depth frame pair."""
        if self._pipeline is None:
            return None

        self._frame_id += 1
        self._frame_tokens = []
        packet = FramePacket(frame_id=self._frame_id)

        try:
            frames = self._pipeline.wait_for_frames(2000)
            if frames is None:
                # Timeout — camera not yet ready or momentary stall.
                # Return an empty packet (rgb=None) so the orchestrator keeps
                # running rather than treating this as end-of-stream.
                return FramePacket(frame_id=self._frame_id)

            # Apply software depth→color alignment if AlignFilter is available
            if self._align_filter is not None:
                try:
                    aligned = self._align_filter.process(frames)
                    if aligned is not None:
                        frames = aligned.as_frame_set() if hasattr(aligned, "as_frame_set") else aligned
                except Exception as e:
                    logger.debug("AlignFilter.process failed: %s", e)

            # ── Color frame ──
            color_frame = frames.get_color_frame()
            if color_frame is not None:
                rgb = self._color_to_numpy(color_frame)
                if rgb is not None:
                    buf = self._acquire_buffer("frame_rgb")
                    buf_np = _as_numpy(buf)
                    if rgb.shape[:2] != (self._config.height, self._config.width):
                        rgb = cv2.resize(rgb, (self._config.width, self._config.height))
                    np.copyto(buf_np, rgb)
                    packet.rgb = buf

            # ── Depth frame ──
            depth_frame = frames.get_depth_frame()
            if depth_frame is not None:
                depth = self._depth_to_numpy(depth_frame)
                if depth is not None:
                    buf = self._acquire_buffer("frame_depth")
                    buf_np = _as_numpy(buf)
                    if depth.shape[:2] != (self._config.height, self._config.width):
                        depth = cv2.resize(
                            depth, (self._config.width, self._config.height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(np.uint16)
                    np.copyto(buf_np, depth)
                    packet.depth = buf

            return packet

        except Exception as e:
            logger.debug("Orbbec read failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
            logger.info("Orbbec pipeline stopped")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def pop_frame_tokens(self) -> list[tuple[str, int]]:
        tokens = self._frame_tokens
        self._frame_tokens = []
        return tokens

    @property
    def has_depth(self) -> bool:
        return True

    @property
    def depth_index(self) -> Optional[int]:
        return None

    @property
    def camera_intrinsics(self) -> Optional[tuple]:
        if self._intrinsics is None:
            return None
        fx, fy, cx, cy = self._intrinsics
        return (fx, fy, cx, cy, self._depth_scale)

    # ------------------------------------------------------------------
    # Frame conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _color_to_numpy(frame) -> Optional[np.ndarray]:
        try:
            data = np.frombuffer(frame.get_data(), dtype=np.uint8)
            w, h = frame.get_width(), frame.get_height()
            fmt = frame.get_format()

            if fmt == OBFormat.RGB:
                return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
            elif fmt == OBFormat.BGR:
                return data.reshape(h, w, 3)
            elif fmt == OBFormat.BGRA:
                return data.reshape(h, w, 4)[:, :, :3]
            elif fmt == OBFormat.RGBA:
                return cv2.cvtColor(data.reshape(h, w, 4), cv2.COLOR_RGBA2BGR)
            elif fmt == OBFormat.MJPG:
                return cv2.imdecode(data, cv2.IMREAD_COLOR)
            elif fmt == OBFormat.YUYV:
                return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
            else:
                if data.size == h * w * 3:
                    return data.reshape(h, w, 3)
                return None
        except Exception as e:
            logger.debug("Color conversion failed: %s", e)
            return None

    @staticmethod
    def _depth_to_numpy(frame) -> Optional[np.ndarray]:
        try:
            w, h = frame.get_width(), frame.get_height()
            data = frame.get_data()
            fmt = frame.get_format()

            if fmt == OBFormat.Y16:
                return np.frombuffer(data, dtype=np.uint16).reshape(h, w)
            byte_data = np.frombuffer(data, dtype=np.uint8)
            if byte_data.size == h * w * 2:
                return byte_data.view(np.uint16).reshape(h, w)
            if byte_data.size == h * w:
                return byte_data.astype(np.uint16) * 32
            return None
        except Exception as e:
            logger.debug("Depth conversion failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Buffer pool
    # ------------------------------------------------------------------

    def _acquire_buffer(self, pool_name: str):
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


def _as_numpy(buf) -> np.ndarray:
    if isinstance(buf, np.ndarray):
        return buf
    try:
        return buf.numpy()
    except Exception:
        return np.asarray(buf)
