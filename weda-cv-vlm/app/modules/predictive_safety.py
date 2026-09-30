"""Predictive safety monitor.

Per frame:
  1. TrajectoryEstimator.update(packet) populates per-track velocity.
  2. spatial_tools.predict_collisions(packet, horizon_s) returns
     pairwise predictions.
  3. For each prediction: dedupe against recent firings, then publish
     an Alert via the AlertBus with severity `warn` (eta > critical_eta_s)
     or `critical` (eta <= critical_eta_s). Payload carries track ids,
     ETA, closing speed and miss distance.

No model inference; the VLM is invoked separately by the agent layer.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from app.modules.alert_bus import (
    SEVERITY_CRITICAL,
    SEVERITY_WARN,
    Alert,
    get_bus,
)
from app.modules.spatial_tools import CollisionPrediction, predict_collisions
from app.utils.data_structures import FramePacket
from app.utils.logger import get_logger

logger = get_logger()


@dataclass(slots=True)
class _RecentKey:
    """Dedup key — same pair within `dedup_window_s` doesn't re-fire."""
    track_a: int
    track_b: int
    fired_at: float


class PredictiveSafetyMonitor:
    """One-call-per-frame collision predictor.

    Configurable thresholds:
      - horizon_s: how far in the future to look (default 3 s)
      - critical_eta_s: below this, severity escalates to critical
      - dedup_window_s: same pair can't re-fire within this window
      - cooldown_s: even different pairs are rate-limited globally
    """

    def __init__(
        self,
        horizon_s: float = 3.0,
        critical_eta_s: float = 1.5,
        dedup_window_s: float = 4.0,
        cooldown_s: float = 0.25,
    ) -> None:
        self._horizon_s = horizon_s
        self._critical_eta_s = critical_eta_s
        self._dedup_window_s = dedup_window_s
        self._cooldown_s = cooldown_s
        self._bus = get_bus()
        self._lock = threading.Lock()
        self._recent: dict[tuple[int, int], _RecentKey] = {}
        self._last_emit_ts: float = 0.0

    def step(self, packet: FramePacket) -> list[Alert]:
        """Run prediction once. Always GCs the dedup map before any
        early-out so the table can't accumulate stale entries during
        long quiet stretches."""
        now = time.monotonic()
        # Unconditional GC — runs even on cooldown / no-pred ticks so
        # the dedup table doesn't grow unbounded between hot windows.
        with self._lock:
            stale = [
                k for k, v in self._recent.items()
                if now - v.fired_at > self._dedup_window_s
            ]
            for k in stale:
                del self._recent[k]
        if now - self._last_emit_ts < self._cooldown_s:
            return []
        try:
            preds = predict_collisions(packet, horizon_s=self._horizon_s)
        except Exception as e:
            logger.debug("predict_collisions failed: %s", e)
            return []
        if not preds:
            return []

        emitted: list[Alert] = []
        with self._lock:

            for p in preds:
                key = _pair_key(p.track_a, p.track_b)
                if key in self._recent:
                    continue
                severity = (
                    SEVERITY_CRITICAL if p.eta_s <= self._critical_eta_s
                    else SEVERITY_WARN
                )
                msg = _format_message(p)
                alert = self._bus.publish(
                    severity=severity,
                    source="predictive-safety",
                    message=msg,
                    payload={
                        "track_a": p.track_a,
                        "track_b": p.track_b,
                        "class_a": p.class_a,
                        "class_b": p.class_b,
                        "eta_s": round(p.eta_s, 2),
                        "closing_speed_mps": round(p.closing_speed_mps, 2),
                        "miss_distance_m": round(p.miss_distance_m, 2),
                        "horizon_s": self._horizon_s,
                    },
                    replay_anchor=time.time(),
                )
                emitted.append(alert)
                self._recent[key] = _RecentKey(
                    track_a=p.track_a, track_b=p.track_b, fired_at=now,
                )
                self._last_emit_ts = now
                # One alert per tick is enough; if multiple imminent
                # collisions are happening simultaneously consumers
                # will see them across consecutive ticks.
                break
        return emitted

    def reset(self) -> None:
        """Drop the dedup state. Used when the camera disconnects or
        the operator clears alerts manually."""
        with self._lock:
            self._recent.clear()
            self._last_emit_ts = 0.0


def _pair_key(a: int, b: int) -> tuple[int, int]:
    """Order-independent pair key for dedup."""
    return (a, b) if a <= b else (b, a)


def _format_message(p: CollisionPrediction) -> str:
    """Human-readable alert text — what consumers actually surface."""
    return (
        f"{p.class_a.title()} T{p.track_a} → {p.class_b.title()} T{p.track_b} "
        f"collision predicted in {p.eta_s:.1f} s · "
        f"closing speed {p.closing_speed_mps:.2f} m/s · "
        f"miss {p.miss_distance_m:.2f} m"
    )
