"""Depth processing: colorization, distance lookup, 3-D projection.

GPU path uses cv2.cuda (OpenCV CUDA) for colormap/normalization.
CPU fallback when GPU contention detected by the Scheduler.
Pinhole projection for 3-D coordinates from pixel + depth.
"""

from __future__ import annotations
from typing import Optional
import cv2
import numpy as np
import time
from weda_optim import BufferPool, Scheduler, GPU
from app.utils.data_structures import (
    Detection,
    PipelineConfig,
    SpatialInfo,
    StageID,
)
from app.utils.logger import get_logger

logger = get_logger()

try:
    _HAS_CUDA_CV = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
except Exception:
    _HAS_CUDA_CV = False

_L1_CONFIGURED = False
try:
    import torch
    _HAS_TORCH = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False

DEFAULT_FX = 615.0
DEFAULT_FY = 615.0
DEFAULT_CX = 320.0
DEFAULT_CY = 240.0
DEPTH_SCALE = 0.001


class DepthProcessor:
    def __init__(
        self,
        config: PipelineConfig,
        scheduler: Optional[Scheduler] = None,
        pool: Optional[BufferPool] = None,
    ) -> None:
        self._config = config
        self._scheduler = scheduler
        self._pool = pool
        self._fx = DEFAULT_FX
        self._fy = DEFAULT_FY
        self._cx = DEFAULT_CX
        self._cy = DEFAULT_CY
        self._depth_scale = DEPTH_SCALE
        self._prev_positions: dict[int, np.ndarray] = {}
        self._prev_time: float = 0.0
        self._zone_entry_times: dict[int, float] = {}

    def update_intrinsics(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depth_scale: Optional[float] = None,
    ) -> None:
        """Update pinhole intrinsics from camera SDK (called after camera open).
        For Orbbec Gemini 336L, these come from pipeline.get_camera_param().rgb_intrinsic.
        Overrides the RealSense D435 defaults so 3-D projection is accurate.
        depth_scale: raw unit → mm (Orbbec SDK value, typically 1.0).
        """
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        if depth_scale is not None and depth_scale > 0:
            self._depth_scale = depth_scale * 0.001
        logger.info(
            "DepthProcessor intrinsics updated from SDK: fx=%.1f fy=%.1f cx=%.1f cy=%.1f depth_scale=%.6f",
            fx, fy, cx, cy, self._depth_scale,
        )

    def colorize(
        self, depth: np.ndarray, placement: str = "auto"
    ) -> np.ndarray:
        depth_np = _as_numpy(depth)
        if depth_np.dtype == np.int16:
            depth_np = depth_np.view(np.uint16)

        effective = placement
        if effective == "auto" and self._scheduler:
            effective = self._scheduler.decide_placement(StageID.DEPTH.value)

        if effective == GPU and _HAS_CUDA_CV:
            self._configure_l1_preference()
            return self._colorize_gpu(depth_np)
        return self._colorize_cpu(depth_np)

    @staticmethod
    def _configure_l1_preference() -> None:
        """Set L1 cache preference for depth GPU kernels (256 KiB per SM pair).

        Depth colorization is memory-read-heavy with no shared memory usage.
        Maximizing L1 data cache improves throughput by ~5-10%.
        """
        global _L1_CONFIGURED
        if _L1_CONFIGURED or not _HAS_TORCH:
            return
        try:
            import ctypes
            libcudart = ctypes.CDLL("libcudart.so")
            # cudaFuncCachePreferL1 = 1
            libcudart.cudaDeviceSetCacheConfig(ctypes.c_int(1))
            _L1_CONFIGURED = True
            logger.debug("CUDA L1 cache preference set for depth processing")
        except Exception:
            _L1_CONFIGURED = True  # Don't retry

    def compute_spatial(
        self,
        detections: list[Detection],
        depth: np.ndarray,
    ) -> list[SpatialInfo]:
        depth_np = _as_numpy(depth)
        if depth_np.dtype == np.int16:
            depth_np = depth_np.view(np.uint16)

        now = time.monotonic()
        dt = now - self._prev_time if self._prev_time > 0 else 0.0
        active_ids: set[int] = {d.track_id for d in detections if d.track_id >= 0}

        spatial_list: list[SpatialInfo] = []
        positions: list[Optional[np.ndarray]] = []

        for det in detections:
            si = SpatialInfo()
            cx, cy = det.center
            _raw_d = self._depth_at(depth_np, det)
            si.distance_m = abs(_raw_d)
            si.depth_reliable = _raw_d > 0  # positive = bbox ROI; negative = expanded fallback

            if si.distance_m > 0:
                pos = self._project_3d(cx, cy, si.distance_m)
                si.position_3d = pos
                x1, y1, x2, y2 = det.bbox

                # ── Velocity tracking ──────────────────────────────────
                if dt > 0 and det.track_id >= 0 and det.track_id in self._prev_positions:
                    vel = (pos - self._prev_positions[det.track_id]) / dt
                    si.velocity_3d = vel
                    si.speed_ms = float(np.linalg.norm(vel))
                    si.approaching = bool(vel[2] < -0.15)  # closing on camera
                if det.track_id >= 0:
                    self._prev_positions[det.track_id] = pos.copy()

                # ── Mask-based accurate size estimate ──────────────────
                size_from_mask = False
                if det.mask is not None:
                    try:
                        mask_bool = det.mask > 127
                        ys, xs = np.where(mask_bool)
                        if len(xs) > 20:
                            depths_in_mask = depth_np[ys, xs].astype(np.float32) * self._depth_scale
                            valid_d = depths_in_mask[(depths_in_mask > 0.1) & (depths_in_mask < 20.0)]
                            if len(valid_d) > 10:
                                med_z = float(np.median(valid_d))
                                w_m = (int(xs.max()) - int(xs.min())) * med_z / self._fx
                                h_m = (int(ys.max()) - int(ys.min())) * med_z / self._fy
                                si.size_estimate = (round(w_m, 3), round(h_m, 3))
                                size_from_mask = True

                                # ── Lean / instability detection ───────
                                mask_cx = float(xs.mean())
                                bbox_cx = (det.bbox[0] + det.bbox[2]) / 2.0
                                offset_px = mask_cx - bbox_cx
                                det.lean_offset_m = round(offset_px * si.distance_m / self._fx, 3)
                                bbox_w = det.bbox[2] - det.bbox[0]
                                det.is_unstable = (
                                    abs(offset_px) > 0.15 * bbox_w and bbox_w > 20
                                )
                    except Exception:
                        pass

                if not size_from_mask:
                    si.size_estimate = (
                        abs(float(x2 - x1)) * si.distance_m / self._fx,
                        abs(float(y2 - y1)) * si.distance_m / self._fy,
                    )
            else:
                pos = None

            # ── Dwell time tracking ────────────────────────────────────
            if det.track_id >= 0:
                if det.track_id not in self._zone_entry_times:
                    self._zone_entry_times[det.track_id] = now
                si.dwell_seconds = now - self._zone_entry_times[det.track_id]

            positions.append(pos)
            spatial_list.append(si)

        # Prune state for tracks that are no longer active
        self._prev_positions = {k: v for k, v in self._prev_positions.items() if k in active_ids}
        self._zone_entry_times = {k: v for k, v in self._zone_entry_times.items() if k in active_ids}
        self._prev_time = now

        # Nearest-neighbor search
        for i, si in enumerate(spatial_list):
            if positions[i] is None:
                continue
            best_dist = float("inf")
            best_id = -1
            for j, sj in enumerate(spatial_list):
                if i == j or positions[j] is None:
                    continue
                d = float(np.linalg.norm(positions[i] - positions[j]))
                if d < best_dist:
                    best_dist = d
                    best_id = detections[j].track_id if detections[j].track_id >= 0 else j
            si.nearest_neighbor_id = best_id
            si.nearest_neighbor_dist = best_dist

        return spatial_list

    # ------------------------------------------------------------------
    # Colorization backends
    # ------------------------------------------------------------------

    def _colorize_gpu(self, depth: np.ndarray) -> np.ndarray:
        try:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(depth)
            # Normalize to 0-255
            norm = cv2.cuda.normalize(gpu_mat, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            result = norm.download()
            colored = cv2.applyColorMap(result, cv2.COLORMAP_TURBO)
            return colored
        except Exception as e:
            logger.debug("GPU colorize fallback: %s", e)
            return self._colorize_cpu(depth)

    @staticmethod
    def _colorize_cpu(depth: np.ndarray) -> np.ndarray:
        valid = depth[depth > 0]
        if valid.size == 0:
            return np.zeros((*depth.shape, 3), dtype=np.uint8)
        lo, hi = np.percentile(valid, [2, 98])
        normalized = np.clip((depth.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
        gray = (normalized * 255).astype(np.uint8)
        return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------

    def _depth_at(self, depth: np.ndarray, det: "Detection") -> float:
        """Robust depth estimate using full detection bbox.

        Returns positive float for reliable measurement (>=3 valid pixels in bbox).
        Returns negative float for fallback estimate (expanded 2x ROI).
        Returns 0.0 when no valid depth found anywhere near the detection.

        Uses 25th percentile (biased toward closer readings) rather than median
        for better safety-critical close-range accuracy.
        """
        h, w = depth.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        # Primary: full bbox ROI
        if x2 > x1 and y2 > y1:
            roi = depth[y1:y2, x1:x2].astype(np.float32) * self._depth_scale
            valid = roi[(roi > 0.01) & (roi < 20.0)]
            if valid.size >= 3:
                return float(np.percentile(valid, 25))  # positive = reliable

        # Fallback: 2x expanded ROI -- catches edge objects and depth holes
        pad = max(20, (x2 - x1) // 2, (y2 - y1) // 2)
        x1e, x2e = max(0, x1 - pad), min(w, x2 + pad)
        y1e, y2e = max(0, y1 - pad), min(h, y2 + pad)
        roi2 = depth[y1e:y2e, x1e:x2e].astype(np.float32) * self._depth_scale
        valid2 = roi2[(roi2 > 0.01) & (roi2 < 20.0)]
        if valid2.size >= 1:
            return -float(np.percentile(valid2, 25))  # negative = fallback

        return 0.0

    def _project_3d(self, px: float, py: float, depth_m: float) -> np.ndarray:
        x = (px - self._cx) * depth_m / self._fx
        y = (py - self._cy) * depth_m / self._fy
        return np.array([x, y, depth_m], dtype=np.float32)


def _as_numpy(buf) -> np.ndarray:
    if isinstance(buf, np.ndarray):
        return buf
    try:
        return buf.numpy()
    except Exception:
        return np.asarray(buf)
