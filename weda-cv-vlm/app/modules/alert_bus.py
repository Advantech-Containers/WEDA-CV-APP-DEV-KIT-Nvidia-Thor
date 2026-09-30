"""Unified pub/sub alert bus.

Every subsystem that wants to surface a real-time alert (predictive
collision, fact-check contradiction, pose-driven semantic alarm,
safety-distance breach, operator override, etc.) publishes through
this module. Consumers are:

  - The HTTP API ``/v1/alerts/stream`` (SSE) — for live consumers
    (NemoClaw, partner tooling).
  - The HTTP API ``/v1/alerts/recent`` (snapshot) — for consumers
    joining mid-session.
  - The renderer's event timeline (existing) — receives a callback
    so alerts also appear on the engineer HUD.

Design notes:
  - Pure Python; no external deps.
  - Thread-safe via a ``threading.Lock`` around the ring buffer +
    subscriber list.
  - Bounded memory: ring buffer caps at ``maxlen`` (default 200 —
    enough for a long-running session, tight enough to bound RAM).
  - Subscribers are weak-ref-free callables; the publisher does
    NOT block on slow subscribers (each callback is fired in a
    short ``try/except`` so a buggy subscriber can't hang the
    main loop).
  - JSON-serialisable schema by construction so the SSE / NemoClaw
    audit / log file paths can all consume the same dict.
"""

from __future__ import annotations
import collections
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional
from app.utils.logger import get_logger
logger = get_logger()
# ─────────────────────────────────────────────────────────────────
# Severity vocabulary — keep this small so consumer colour mapping
# is unambiguous. Extending it requires a renderer change too, so
# don't add entries casually.
# ─────────────────────────────────────────────────────────────────
SEVERITY_INFO = "info"
SEVERITY_WARN = "warn"
SEVERITY_CRITICAL = "critical"
_SEVERITIES = frozenset({SEVERITY_INFO, SEVERITY_WARN, SEVERITY_CRITICAL})

@dataclass(slots=True)
class Alert:
    """One alert event. Produced by any subsystem; consumed by all."""
    id: str                         # UUID, monotonic per-process
    ts: float                       # time.time() at publish (epoch)
    monotonic_ts: float             # time.monotonic() — for ordering
    severity: str                   # "info" | "warn" | "critical"
    source: str                     # "predictive-safety" | "fact-checker" | …
    message: str                    # human-readable summary
    payload: dict = field(default_factory=dict)
    audit_id: Optional[str] = None
    replay_anchor: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

class AlertBus:
    """In-process pub/sub for Alert events.

    Use the module-level ``get_bus()`` rather than instantiating this
    directly — the bus is a process-wide singleton (one camera, one
    timeline, one alert stream).
    """

    def __init__(self, maxlen: int = 200) -> None:
        self._lock = threading.Lock()
        self._buf: collections.deque[Alert] = collections.deque(maxlen=maxlen)
        self._subscribers: list[Callable[[Alert], None]] = []

    def publish(
        self,
        severity: str,
        source: str,
        message: str,
        payload: Optional[dict] = None,
        audit_id: Optional[str] = None,
        replay_anchor: Optional[float] = None,
    ) -> Alert:
        """Publish a new alert. Returns the constructed Alert.
        Severity must be one of "info" / "warn" / "critical"; an unknown
        severity is normalised to "info" with a debug log so the bus
        never refuses a publish (alerts are by definition things you
        want to surface, not silence).
        """
        if severity not in _SEVERITIES:
            logger.debug("AlertBus: unknown severity %r → info", severity)
            severity = SEVERITY_INFO

        alert = Alert(
            id=uuid.uuid4().hex[:12],
            ts=time.time(),
            monotonic_ts=time.monotonic(),
            severity=severity,
            source=source,
            message=message,
            payload=dict(payload or {}),
            audit_id=audit_id,
            replay_anchor=replay_anchor,
        )

        with self._lock:
            self._buf.append(alert)
            subs = list(self._subscribers)

        for cb in subs:
            try:
                cb(alert)
            except Exception as e:
                logger.debug("AlertBus subscriber error: %s", e)

        return alert

    def subscribe(self, cb: Callable[[Alert], None]) -> Callable[[], None]:
        """Register a subscriber. Returns an unsubscribe callable."""
        with self._lock:
            self._subscribers.append(cb)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(cb)
                except ValueError:
                    pass

        return _unsubscribe

    def recent(self, limit: int = 200, min_severity: Optional[str] = None) -> list[Alert]:
        """Return up to `limit` most-recent alerts, newest first.
        Optionally filter by minimum severity (info ≤ warn ≤ critical).
        """
        order = {SEVERITY_INFO: 0, SEVERITY_WARN: 1, SEVERITY_CRITICAL: 2}
        threshold = order.get(min_severity, 0) if min_severity else 0
        with self._lock:
            snap = list(self._buf)
        snap.reverse()
        if min_severity:
            snap = [a for a in snap if order.get(a.severity, 0) >= threshold]
        return snap[:limit]

    def clear(self) -> None:
        """Drop the entire ring buffer. Use sparingly — typically only on
        explicit operator action ("clear alerts" button)."""
        with self._lock:
            self._buf.clear()

_BUS: Optional[AlertBus] = None
_BUS_LOCK = threading.Lock()

def get_bus() -> AlertBus:
    """Return the singleton AlertBus, creating it lazily."""
    global _BUS
    if _BUS is None:
        with _BUS_LOCK:
            if _BUS is None:
                _BUS = AlertBus()
    return _BUS
