"""Supervisor-defined semantic alarm engine.

Loads rules persisted by /v1/alarms/compose from JSONL, evaluates each
active rule against the live FramePacket, and publishes warn-severity
alerts via alert_bus when a rule fires.

Rule schema (matches app/api/v1/alarms.py:_heuristic_parse):

    {
      "id": "abc123",
      "active": true,
      "raw_text": "alert if anyone bends near press 3+ s",
      "rule": {
        "subject_class": "person" | None,
        "action": "presence" | "enter" | "exit" | "dwell" |
                  "bend" | "count_gt",
        "location_zone": "zone-3" | None,
        "duration_s": 3.0 | None,
        "count_threshold": 5 | None,
      },
      "fired_count": 0,
      "last_fired_ts": None,
    }

Per-frame cost O(R * D) where R is active-rule count, D is detection
count. No model inference. Dwell-state is per-(track, rule) and is
cleared on exit. Rule firings update in-memory counters; disk
write-through happens only on compose / activate / delete.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app.modules.alert_bus import SEVERITY_WARN, get_bus
from app.utils.data_structures import FramePacket
from app.utils.logger import get_logger

logger = get_logger()


@dataclass(slots=True)
class _RuleState:
    """Per-rule, per-track timer state for dwell / bend conditions."""
    rule_id: str
    track_id: int
    enter_ts: float


class AlarmEngine:
    """Loads rules, evaluates per frame, publishes alerts.

    Single-instance. Construct in the orchestrator, call ``step(packet)``
    inside the per-frame hook block.
    """

    def __init__(self, persist_path: str = "config/alarms.jsonl") -> None:
        self._persist = Path(persist_path)
        self._lock = threading.Lock()
        self._rules: dict[str, dict] = {}
        self._timers: dict[tuple[str, int], _RuleState] = {}
        self._zones: list[dict] = []
        self._cooldown_s: float = 5.0
        self._last_fire_per_rule: dict[str, float] = {}
        self._bus = get_bus()
        self._reload()

    # ── lifecycle ────────────────────────────────────────────────────
    def set_zones(self, zones: list[dict]) -> None:
        """Wire zone polygons in once at startup; cheap to refresh."""
        with self._lock:
            self._zones = list(zones or [])

    def reload(self) -> None:
        """Re-read the persisted JSONL — useful after compose/activate."""
        self._reload()

    def _reload(self) -> None:
        if not self._persist.exists():
            return
        with self._lock:
            self._rules.clear()
            try:
                with self._persist.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not rec.get("active"):
                            continue
                        rid = rec.get("id")
                        if not rid:
                            continue
                        self._rules[rid] = rec
            except OSError as e:
                logger.debug("alarm_engine reload failed: %s", e)

    # ── per-frame hot path ──────────────────────────────────────────
    def step(self, packet: FramePacket) -> int:
        """Evaluate all active rules. Returns the count of fires this tick."""
        if not self._rules:
            return 0
        now = time.monotonic()
        wall = time.time()
        fired = 0
        with self._lock:
            zones = self._zones
            rules = list(self._rules.values())
        occupancy = self._zones_now(packet, zones) if zones else {}

        for rec in rules:
            rid = rec["id"]
            rule = rec.get("rule") or {}
            if now - self._last_fire_per_rule.get(rid, 0.0) < self._cooldown_s:
                continue
            triggered = self._evaluate(rec, rule, packet, occupancy, now)
            if triggered:
                self._last_fire_per_rule[rid] = now
                fired += 1
                with self._lock:
                    rec["fired_count"] = int(rec.get("fired_count", 0)) + 1
                    rec["last_fired_ts"] = wall
                self._bus.publish(
                    severity=SEVERITY_WARN,
                    source="alarm-engine",
                    message=f"Rule fired: {rec.get('raw_text', rid)}",
                    payload={
                        "rule_id": rid,
                        "rule": rule,
                        "fired_count": rec["fired_count"],
                    },
                )
        return fired

    # ── rule dispatch ───────────────────────────────────────────────
    def _evaluate(
        self,
        rec: dict,
        rule: dict,
        packet: FramePacket,
        occupancy: dict[str, set[int]],
        now: float,
    ) -> bool:
        action = rule.get("action") or "presence"
        subj = rule.get("subject_class")
        zone = rule.get("location_zone")
        dur = rule.get("duration_s")
        cnt_thresh = rule.get("count_threshold")
        candidates = self._candidate_tracks(packet, subj, zone, occupancy)

        if action in ("presence", "enter"):
            return bool(candidates)

        if action == "exit":
            prev_state_key = ("__exit__", rec["id"])
            prev_nonempty = self._timers.pop(prev_state_key, None) is not None
            if candidates:
                self._timers[prev_state_key] = _RuleState(
                    rule_id=rec["id"], track_id=-1, enter_ts=now,
                )
                return False
            return prev_nonempty

        if action == "count_gt":
            n = int(cnt_thresh or 0)
            return len(candidates) > n

        if action in ("dwell", "bend"):
            if dur is None:
                return False
            duration_s = float(dur)
            current_tracks = {d.track_id for d in candidates}
            rid = rec["id"]
            keys_to_drop = [
                (rid_, tid) for (rid_, tid) in self._timers
                if rid_ == rid and tid not in current_tracks
            ]
            for k in keys_to_drop:
                self._timers.pop(k, None)
            for tid in current_tracks:
                key = (rid, tid)
                state = self._timers.get(key)
                if state is None:
                    self._timers[key] = _RuleState(
                        rule_id=rid, track_id=tid, enter_ts=now,
                    )
                else:
                    if now - state.enter_ts >= duration_s:
                        self._timers[key] = _RuleState(
                            rule_id=rid, track_id=tid, enter_ts=now,
                        )
                        return True
            return False

        return False

    def _candidate_tracks(
        self,
        packet: FramePacket,
        subject_class: Optional[str],
        zone_name: Optional[str],
        occupancy: dict[str, set[int]],
    ) -> list[Any]:
        if subject_class is None:
            cand = list(packet.detections or [])
        else:
            sc = subject_class.lower()
            cand = [d for d in (packet.detections or []) if d.class_name.lower() == sc]
        if zone_name and zone_name in occupancy:
            zone_tids = occupancy[zone_name]
            cand = [d for d in cand if d.track_id in zone_tids]
        return cand

    @staticmethod
    def _zones_now(packet: FramePacket, zones: list[dict]) -> dict[str, set[int]]:
        from app.modules.spatial_tools import zones_now
        out: dict[str, set[int]] = {}
        try:
            results = zones_now(packet, zones)
        except Exception:
            return out
        for r in results:
            out[r.zone_name] = set(r.track_ids)
        return out
