"""Frame resize via GStreamer nvvidconv (Jetson VIC), with cv2 fallback.

Holds one persistent ``appsrc ! videoconvert ! nvvidconv ! videoconvert
! appsink`` pipeline; ``resize()`` pushes a buffer and pulls a sample.
Falls back to ``cv2.resize()`` when PyGObject / nvvidconv are unavailable
or on any pipeline error.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger("weda_optim")

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    Gst.init(None)
    _HAS_GST = True
except (ImportError, ValueError):
    _HAS_GST = False

_HAS_NVVIDCONV = False
if _HAS_GST:
    try:
        _HAS_NVVIDCONV = Gst.ElementFactory.find("nvvidconv") is not None
    except Exception:
        _HAS_NVVIDCONV = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

_PULL_TIMEOUT_NS = 1_000_000_000  # 1 s


class VicResizer:
    """VIC-backed frame resizer; falls back to cv2.resize. Thread-safe."""

    def __init__(self, target_width: int = 640, target_height: int = 640) -> None:
        self._tw = int(target_width)
        self._th = int(target_height)
        self._use_vic = bool(_HAS_GST and _HAS_NVVIDCONV)
        self._lock = threading.Lock()
        self._pipeline = None
        self._appsrc = None
        self._appsink = None
        self._src_size: Optional[tuple[int, int]] = None
        if not self._use_vic:
            reason = "PyGObject/GStreamer unavailable" if not _HAS_GST \
                else "nvvidconv not found"
            logger.info("VIC offload disabled (%s) — using cv2.resize", reason)

    def _build_pipeline(self, src_w: int, src_h: int) -> bool:
        """(Re)build the pipeline. Caller holds self._lock; disables VIC on failure."""
        self._teardown_pipeline()
        try:
            desc = (
                "appsrc name=src is-live=false block=true do-timestamp=true format=time "
                f"caps=video/x-raw,format=BGR,width={src_w},height={src_h},framerate=30/1 "
                "! videoconvert ! nvvidconv "
                f"! video/x-raw,format=BGRx,width={self._tw},height={self._th} "
                "! videoconvert ! video/x-raw,format=BGR "
                "! appsink name=sink emit-signals=false sync=false max-buffers=1 drop=false"
            )
            pipeline = Gst.parse_launch(desc)
            if pipeline is None:
                raise RuntimeError("parse_launch returned None")
            appsrc = pipeline.get_by_name("src")
            appsink = pipeline.get_by_name("sink")
            if appsrc is None or appsink is None:
                raise RuntimeError("appsrc/appsink not found")
            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("pipeline failed to reach PLAYING")
            ret, _state, _pending = pipeline.get_state(2 * Gst.SECOND)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("pipeline state change failed")
            self._pipeline = pipeline
            self._appsrc = appsrc
            self._appsink = appsink
            self._src_size = (src_w, src_h)
            return True
        except Exception as e:
            logger.warning("VIC pipeline build failed (%s) — using cv2.resize", e)
            self._teardown_pipeline()
            self._use_vic = False
            return False

    def _teardown_pipeline(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self._pipeline = None
        self._appsrc = None
        self._appsink = None
        self._src_size = None

    def resize(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return frame
        h, w = frame.shape[:2]
        if w == self._tw and h == self._th:
            return frame
        if self._use_vic:
            out = self._resize_vic(frame, w, h)
            if out is not None:
                return out
        return self._resize_cv2(frame)

    def _resize_vic(self, frame: np.ndarray, w: int, h: int) -> Optional[np.ndarray]:
        with self._lock:
            if not self._use_vic:
                return None
            if self._pipeline is None or self._src_size != (w, h):
                if not self._build_pipeline(w, h):
                    return None
            try:
                payload = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
                buf = Gst.Buffer.new_wrapped(payload)
                if self._appsrc.emit("push-buffer", buf) != Gst.FlowReturn.OK:
                    raise RuntimeError("push-buffer rejected")
                sample = self._appsink.emit("try-pull-sample", _PULL_TIMEOUT_NS)
                if sample is None:
                    raise RuntimeError("no sample within timeout")
                out_buf = sample.get_buffer()
                ok, minfo = out_buf.map(Gst.MapFlags.READ)
                if not ok:
                    raise RuntimeError("buffer map failed")
                try:
                    expected = self._th * self._tw * 3
                    # assumes tightly-packed BGR (true for 64-byte-aligned widths, e.g. 640)
                    if minfo.size < expected:
                        raise RuntimeError(f"short buffer {minfo.size} < {expected}")
                    arr = np.frombuffer(minfo.data, dtype=np.uint8, count=expected)
                    return arr.reshape(self._th, self._tw, 3).copy()
                finally:
                    out_buf.unmap(minfo)
            except Exception as e:
                logger.warning("VIC resize failed (%s) — using cv2.resize", e)
                self._teardown_pipeline()
                self._use_vic = False
                return None

    def _resize_cv2(self, frame: np.ndarray) -> np.ndarray:
        if _HAS_CV2:
            return cv2.resize(frame, (self._tw, self._th))
        h, w = frame.shape[:2]
        row_idx = (np.arange(self._th) * h // self._th).astype(int)
        col_idx = (np.arange(self._tw) * w // self._tw).astype(int)
        return frame[np.ix_(row_idx, col_idx)]

    def close(self) -> None:
        with self._lock:
            self._teardown_pipeline()

    def __del__(self):
        try:
            self._teardown_pipeline()
        except Exception:
            pass

    @property
    def uses_vic(self) -> bool:
        return self._use_vic

    @property
    def target_size(self) -> tuple[int, int]:
        return (self._tw, self._th)


def is_vic_available() -> bool:
    return bool(_HAS_GST and _HAS_NVVIDCONV)
