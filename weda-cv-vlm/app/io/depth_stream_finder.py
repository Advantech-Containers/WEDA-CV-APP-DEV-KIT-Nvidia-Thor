"""Smart depth stream detection for multi-node USB cameras.

Scores sibling /dev/video* devices by V4L2 format keywords and
validates candidates with a frame quality test to reject IR streams.

Depth streams: Z16/Y16 format, uint16, high spatial variance (objects at different distances).
IR streams: Y8/GREY, uint8, uniform low-variance noise.
"""

from __future__ import annotations

import re
import subprocess
from typing import Optional

import cv2
import numpy as np

from app.utils.logger import get_logger

logger = get_logger()

# Format keywords and their scores
DEPTH_KEYWORDS = {"z16": 10, "grey16": 8, "y16": 6, "depth": 10, "rw16": 6, "inzi": 5}
IR_REJECT_KEYWORDS = {"y8": -5, "grey8": -5, "grey": -3, "y10": -2}


class DepthStreamFinder:
    """Find the best depth stream among sibling camera devices."""

    @staticmethod
    def find(rgb_index: int) -> Optional[int]:
        """Return the best depth device index, or None."""
        siblings = DepthStreamFinder._find_siblings(rgb_index)
        if not siblings:
            return None

        candidates = [s for s in siblings if s != rgb_index]
        if not candidates:
            return None

        logger.info("Depth probe: %d candidates from same camera: %s", len(candidates), candidates)

        # Pre-filter: only keep devices that can actually be opened for capture
        openable = []
        for idx in candidates:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.release()
                openable.append(idx)
            else:
                logger.debug("  /dev/video%d: can't open (metadata/control node), skipping", idx)

        if not openable:
            logger.info("No openable video devices among siblings")
            return None

        # Score each openable candidate by V4L2 format
        scored: list[tuple[int, int]] = []
        for idx in openable:
            score = DepthStreamFinder._score_device(idx)
            scored.append((idx, score))
            logger.debug("  /dev/video%d: score=%d", idx, score)

        # Sort by score descending
        scored.sort(key=lambda x: -x[1])

        # Validate top candidates (try up to 3)
        for idx, score in scored[:3]:
            if DepthStreamFinder._validate_depth_frame(idx):
                logger.info("Depth stream found: /dev/video%d (score=%d)", idx, score)
                return idx

        logger.info("No valid depth stream found among candidates")
        return None

    @staticmethod
    def _find_siblings(rgb_index: int) -> list[int]:
        """Find all /dev/video* indices belonging to the same physical camera."""
        try:
            result = subprocess.run(
                ["v4l2-ctl", "--list-devices"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return []

            lines = result.stdout.strip().split("\n")
            current_group: list[int] = []

            for line in lines:
                if line.startswith("\t") or line.startswith(" "):
                    match = re.search(r"/dev/video(\d+)", line.strip())
                    if match:
                        current_group.append(int(match.group(1)))
                else:
                    if rgb_index in current_group:
                        return current_group
                    current_group = []

            # Check last group
            if rgb_index in current_group:
                return current_group
            return []
        except Exception as e:
            logger.debug("v4l2-ctl failed: %s", e)
            return []

    @staticmethod
    def _score_device(index: int) -> int:
        """Score a device by its V4L2 format keywords."""
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", f"/dev/video{index}", "--list-formats-ext"],
                capture_output=True, text=True, timeout=3,
            )
            text = result.stdout.lower()

            score = 0
            for kw, pts in DEPTH_KEYWORDS.items():
                if kw in text:
                    score += pts
            for kw, pts in IR_REJECT_KEYWORDS.items():
                if kw in text and "z16" not in text:
                    score += pts  # pts is negative

            # Bonus for 16-bit formats
            if "16" in text:
                score += 2

            return score
        except Exception:
            return 0

    @staticmethod
    def _validate_depth_frame(index: int) -> bool:
        """Grab a frame and check if it looks like real depth data.

        Criteria:
          1. Can open and read a frame
          2. Values in uint16 range (not uint8 IR)
          3. Median in plausible depth range (200-20000 mm)
          4. Coefficient of variation > 0.05 (spatial variance — IR is uniform)
        """
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if not cap.isOpened():
                return False

            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return False

            # Convert to single-channel
            if frame.ndim == 3:
                if frame.shape[2] == 2:
                    # Y16 encoded as 2-channel uint8 → reinterpret as uint16
                    h, w = frame.shape[:2]
                    depth = frame[:, :, :2].reshape(h, w, 2)
                    depth = depth[:, :, 0].astype(np.uint16) | (depth[:, :, 1].astype(np.uint16) << 8)
                else:
                    depth = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                depth = frame

            # Check 1: should be uint16 or have values > 255
            max_val = np.max(depth)
            if max_val <= 255 and depth.dtype == np.uint8:
                logger.debug("  video%d: uint8 only (likely IR), rejecting", index)
                return False

            depth_f = depth.astype(np.float32)

            # Check 2: median in plausible depth range (200mm - 20000mm)
            valid = depth_f[depth_f > 0]
            if valid.size < 100:
                logger.debug("  video%d: too few valid pixels", index)
                return False

            median = np.median(valid)
            if not (100 < median < 30000):
                logger.debug("  video%d: median=%.0f outside depth range", index, median)
                return False

            # Check 3: coefficient of variation (depth has spatial variance, IR is uniform)
            mean = np.mean(valid)
            std = np.std(valid)
            cov = std / mean if mean > 0 else 0

            if cov < 0.05:
                logger.debug("  video%d: CoV=%.3f too low (uniform = IR noise)", index, cov)
                return False

            logger.info("  video%d: VALID depth (median=%.0fmm, CoV=%.3f, max=%d)",
                        index, median, cov, max_val)
            return True

        except Exception as e:
            logger.debug("  video%d: validation failed: %s", index, e)
            return False
