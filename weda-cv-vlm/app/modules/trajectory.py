"""Per-track 3-D velocity estimator for predictive collision checks.

The tracker provides 2-D pixel positions; the spatial stage lifts each
track into 3-D using depth + intrinsics. This module produces a
smoothed 3-D velocity vector per track via an EMA over per-frame
position deltas, which spatial_tools.predict_collisions consumes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.utils.data_structures import FramePacket, SpatialInfo
from app.utils.logger import get_logger

logger = get_logger()


@dataclass(slots=True)
class TrackState:
    track_id: int
    last_seen_ts: float
    position: np.ndarray              # (3,) float32 last position
    velocity: np.ndarray              # (3,) float32 EMA velocity (m/s)
    samples: int = 0                  # how many updates have we accumulated


class TrajectoryEstimator:
    """Per-track velocity EMA, updated each frame from spatial output.

    Side-effect: writes the smoothed velocity back into
    `SpatialInfo.velocity_3d` and `speed_ms` so downstream consumers
    (renderer, fact-checker, predictive safety) see a single canonical
    value.
    """

    def __init__(self, alpha: float = 0.3, stale_after_s: float = 2.0) -> None:
        # alpha is the EMA weight for the *new* sample. 0.3 → about
        # 3 samples to react, smooths jitter without lagging on real
        # motion.
        self._alpha = alpha
        self._stale_after_s = stale_after_s
        self._states: dict[int, TrackState] = {}
        self._lock = threading.Lock()

    def update(self, packet: FramePacket) -> None:
        """Read positions from `packet.spatial`, update EMA velocities,
        write back into the SpatialInfo entries (in-place)."""
        now = time.monotonic()
        seen: set[int] = set()
        with self._lock:
            for det, sp in zip(packet.detections, packet.spatial):
                if det.track_id < 0 or sp is None or sp.position_3d is None:
                    continue
                pos = np.asarray(sp.position_3d, dtype=np.float32)
                if pos.shape != (3,) or not np.all(np.isfinite(pos)):
                    continue
                tid = det.track_id
                seen.add(tid)
                state = self._states.get(tid)
                if state is None:
                    self._states[tid] = TrackState(
                        track_id=tid,
                        last_seen_ts=now,
                        position=pos.copy(),
                        velocity=np.zeros(3, dtype=np.float32),
                        samples=1,
                    )
                    continue
                dt = now - state.last_seen_ts
                if dt < 1e-3:
                    continue
                v_inst = (pos - state.position) / dt
                # EMA update
                if state.samples == 1:
                    # First real velocity sample — seed directly to
                    # avoid the first frame being half-zero.
                    state.velocity = v_inst
                else:
                    state.velocity = (
                        self._alpha * v_inst + (1.0 - self._alpha) * state.velocity
                    )
                state.position = pos.copy()
                state.last_seen_ts = now
                state.samples += 1
                # Write the smoothed velocity back into the SpatialInfo
                # so downstream consumers (predictive safety, renderer,
                # fact-checker) all see one canonical value.
                sp.velocity_3d = state.velocity.copy()
                sp.speed_ms = float(np.linalg.norm(state.velocity))

            # Garbage-collect tracks we haven't seen in a while.
            stale = [
                tid for tid, st in self._states.items()
                if now - st.last_seen_ts > self._stale_after_s
                and tid not in seen
            ]
            for tid in stale:
                del self._states[tid]

    def state_for(self, track_id: int) -> Optional[TrackState]:
        with self._lock:
            return self._states.get(track_id)

    def all_states(self) -> dict[int, TrackState]:
        with self._lock:
            return dict(self._states)
