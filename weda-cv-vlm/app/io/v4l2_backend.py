"""V4L2 camera backend for USB/CSI cameras.

Handles RGB capture via cv2.VideoCapture with V4L2 backend.
Depth auto-detected via Orbbec SDK or V4L2 sibling probing.
Async depth reader thread prevents blocking RGB path.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Optional

import cv2
import numpy as np

from app.io.camera_backend import CameraBackend
from app.utils.data_structures import FramePacket, PipelineConfig
from app.utils.logger import get_logger

logger = get_logger()


class V4L2Backend(CameraBackend):
    """V4L2 USB/CSI camera with optional depth auto-detection."""

    def __init__(self, config: PipelineConfig, pool=None) -> None:
        super().__init__(config, pool)
        self._cap: Optional[cv2.VideoCapture] = None
        self._depth_cap: Optional[cv2.VideoCapture] = None
        self._depth_index: Optional[int] = None
        self._depth_lock = threading.Lock()
        self._latest_depth_frame: Optional[np.ndarray] = None
        self._depth_running = False
        self._depth_thread: Optional[threading.Thread] = None
        self._use_orbbec = False
        self._orbbec = None
        # RGB read-ahead producer-consumer (hides USB jitter from main loop)
        self._rgb_queue: collections.deque = collections.deque(maxlen=2)
        self._rgb_reader_running = False
        self._rgb_reader_thread: Optional[threading.Thread] = None

    def open(self) -> bool:
        source = self._config.source
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass

        if isinstance(source, int):
            self._cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        else:
            self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            logger.error("V4L2: failed to open source %s", self._config.source)
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        self._cap.set(cv2.CAP_PROP_FPS, 60)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info("V4L2 RGB: %s (%dx%d @ %.0f FPS)", self._config.source, actual_w, actual_h, actual_fps)

        # Start RGB read-ahead thread (producer fills deque; main loop reads non-blocking)
        self._rgb_reader_running = True
        self._rgb_reader_thread = threading.Thread(target=self._rgb_read_loop, daemon=True)
        self._rgb_reader_thread.start()

        # Auto-detect depth
        if isinstance(source, int):
            self._detect_depth(source)

        return True

    def read(self) -> Optional[FramePacket]:
        if self._cap is None:
            return None

        self._frame_tokens = []
        packet = FramePacket(frame_id=self._next_frame_id())

        # Non-blocking grab from producer queue; fall back to direct read on first frame
        if self._rgb_queue:
            frame = self._rgb_queue[-1]
        else:
            ret, frame = self._cap.read()
            if not ret:
                return None

        buf = self._acquire_buffer("frame_rgb")
        buf_np = self._as_numpy(buf)
        if frame.shape[:2] == buf_np.shape[:2]:
            np.copyto(buf_np, frame)
        else:
            resized = cv2.resize(frame, (self._config.width, self._config.height))
            np.copyto(buf_np, resized)
        packet.rgb = buf

        # Grab latest depth (non-blocking)
        with self._depth_lock:
            depth_frame = self._latest_depth_frame
        if depth_frame is not None:
            depth_buf = self._acquire_buffer("frame_depth")
            depth_np = self._as_numpy(depth_buf)
            if depth_frame.ndim == 3:
                gray = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = depth_frame
            if gray.shape[:2] == depth_np.shape[:2]:
                if gray.dtype == np.uint16:
                    np.copyto(depth_np, gray)
                else:
                    np.copyto(depth_np, gray.astype(np.uint16) * 32)
            else:
                resized = cv2.resize(gray, (self._config.width, self._config.height))
                np.copyto(depth_np, resized.astype(np.uint16) * 32)
            packet.depth = depth_buf

        return packet

    def _rgb_read_loop(self) -> None:
        """Producer: continuously reads frames into deque. Main loop never blocks.

        For video-file sources with `input.loop: true`, seeks back to frame 0
        on EOF so the demo runs continuously without manual restart.
        """
        src = self._config.source
        is_file_source = not isinstance(src, int) and not str(src).isdigit()
        loop_on_eof = is_file_source and bool(getattr(self._config, "input_loop", False))
        while self._rgb_reader_running and self._cap is not None:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                self._rgb_queue.append(frame)
            elif loop_on_eof:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.001)
            else:
                time.sleep(0.001)

    def close(self) -> None:
        self._rgb_reader_running = False
        if self._rgb_reader_thread:
            self._rgb_reader_thread.join(timeout=2.0)
        self._depth_running = False
        if self._depth_thread:
            self._depth_thread.join(timeout=2.0)
        if self._use_orbbec and self._orbbec:
            self._orbbec.close()
        if self._depth_cap:
            self._depth_cap.release()
            self._depth_cap = None
        if self._cap:
            self._cap.release()
            self._cap = None

    @property
    def has_depth(self) -> bool:
        return self._depth_cap is not None or self._use_orbbec

    @property
    def depth_index(self) -> Optional[int]:
        if self._use_orbbec:
            return None
        return self._depth_index

    # ------------------------------------------------------------------

    def _detect_depth(self, rgb_index: int) -> None:
        """V4L2 depth detection only. Orbbec SDK is NOT used here — it
        conflicts with V4L2 by grabbing the same USB device, causing
        the camera to disconnect. Use backend: 'orbbec' in config for
        Orbbec SDK depth instead."""

        # V4L2 smart depth finder only (no Orbbec — causes USB conflict)
        try:
            from app.io.depth_stream_finder import DepthStreamFinder
            idx = DepthStreamFinder.find(rgb_index)
            if idx is not None:
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if cap.isOpened():
                    self._depth_cap = cap
                    self._depth_index = idx
                    self._depth_running = True
                    self._depth_thread = threading.Thread(target=self._v4l2_depth_loop, daemon=True)
                    self._depth_thread.start()
                    logger.info("V4L2 depth: /dev/video%d (background)", idx)
                    return
        except Exception as e:
            logger.debug("V4L2 depth probe failed: %s", e)

        logger.info("No depth stream found — RGB only")

    def _v4l2_depth_loop(self) -> None:
        while self._depth_running and self._depth_cap:
            ret, frame = self._depth_cap.read()
            if ret and frame is not None:
                with self._depth_lock:
                    self._latest_depth_frame = frame
            else:
                time.sleep(0.001)

    def _orbbec_depth_loop(self) -> None:
        while self._depth_running and self._orbbec:
            try:
                packet = self._orbbec.read()
                if packet and packet.depth is not None:
                    depth = packet.depth
                    if hasattr(depth, 'numpy'):
                        depth = depth.numpy()
                    with self._depth_lock:
                        self._latest_depth_frame = np.asarray(depth)
                    self._orbbec.pop_frame_tokens()
            except Exception:
                time.sleep(0.001)
