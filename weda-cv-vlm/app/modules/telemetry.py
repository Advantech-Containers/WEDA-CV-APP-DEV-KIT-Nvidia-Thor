"""Per-query telemetry attribution.

One sampler thread for the entire process; queries register/unregister
themselves on a thread-safe set. The sampler ticks at 100 ms, walks
the registry, and pushes peak readings into each active record. This
removes the per-query thread cost from the previous implementation —
for a service handling N concurrent queries it cuts thread count from
N to 1, and avoids the join overhead on each query exit.

Hot-path cost per tick:
    O(active_queries) — no allocations beyond the existing record.

Memory:
    Each `QueryRecord` is ~256 B; the bounded ring (default 500)
    caps audit memory at ~128 KB. Active queries pin only their
    record, not a thread.
"""

from __future__ import annotations

import collections
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger()

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


@dataclass(slots=True)
class QueryRecord:
    id: str
    started_ts: float
    ended_ts: float = 0.0
    wall_ms: float = 0.0
    query: str = ""
    source: str = "unknown"
    request_id: str = ""
    gpu_util_peak_pct: float = 0.0
    uma_used_gb_start: float = 0.0
    uma_used_gb_peak: float = 0.0
    uma_used_gb_end: float = 0.0
    cpu_pct_peak: float = 0.0
    rss_mb_peak: float = 0.0
    temperature_c_peak: float = 0.0
    power_w_peak: float = 0.0
    power_w_mean: float = 0.0      # mean across the live samples
    energy_j: float = 0.0
    cost_usd: float = 0.0
    tokens_generated: int = 0
    tok_per_sec: float = 0.0
    tools_called: list[str] = field(default_factory=list)
    factcheck_verified: int = 0
    factcheck_unverifiable: int = 0
    factcheck_contradicted: int = 0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class TelemetryStore:
    """Bounded ring buffer + fan-out subscribers."""

    def __init__(self, maxlen: int = 500) -> None:
        self._lock = threading.Lock()
        self._buf: collections.deque[QueryRecord] = collections.deque(maxlen=maxlen)
        self._subscribers: list = []

    def append(self, rec: QueryRecord) -> None:
        with self._lock:
            self._buf.append(rec)
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(rec)
            except Exception as e:
                logger.debug("TelemetryStore subscriber error: %s", e)

    def subscribe(self, cb) -> callable:
        with self._lock:
            self._subscribers.append(cb)

        def _unsub() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(cb)
                except ValueError:
                    pass

        return _unsub

    def recent(self, limit: int = 50) -> list[QueryRecord]:
        with self._lock:
            snap = list(self._buf)
        snap.reverse()
        return snap[:limit]


_STORE: Optional[TelemetryStore] = None
_STORE_LOCK = threading.Lock()


def get_store() -> TelemetryStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = TelemetryStore()
    return _STORE


# ─────────────────────────────────────────────────────────────────
# Single registry-driven sampler
# ─────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class _ActiveSample:
    rec: QueryRecord
    proc_handle: Any                        # psutil.Process or None
    power_samples_sum: float = 0.0
    power_samples_n: int = 0


class _Registry:
    """Thread-safe set of active query records.

    Sampler thread iterates a copy of the registry per tick; mutations
    (register/unregister) take the lock briefly.
    """

    def __init__(self, gpu_monitor=None, interval_s: float = 0.1) -> None:
        self._gpu = gpu_monitor
        self._interval = interval_s
        self._active: dict[str, _ActiveSample] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def configure(self, gpu_monitor) -> None:
        self._gpu = gpu_monitor

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="weda-telemetry", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 5)
            self._thread = None

    def register(self, rec: QueryRecord) -> _ActiveSample:
        proc = psutil.Process(os.getpid()) if _HAS_PSUTIL else None
        if proc is not None:
            proc.cpu_percent(interval=None)   # prime
        sample = _ActiveSample(rec=rec, proc_handle=proc)
        with self._lock:
            self._active[rec.id] = sample
        # Lazy start so test environments without a sampler don't spin
        # a thread they never need.
        if self._thread is None:
            self.start()
        return sample

    def unregister(self, rec_id: str) -> Optional[_ActiveSample]:
        with self._lock:
            return self._active.pop(rec_id, None)

    def _loop(self) -> None:
        try:
            from weda_optim.gpu_monitor import read_jetson_power_w
        except Exception:
            read_jetson_power_w = None
        while not self._stop.is_set():
            with self._lock:
                samples = list(self._active.values())
            if samples:
                gpu_pct = uma_pct = temp_c = 0.0
                power_w = 0.0
                if self._gpu is not None:
                    try:
                        gpu_pct = float(self._gpu.get_gpu_utilization() or 0.0) * 100.0
                        uma_pct = float(self._gpu.get_memory_utilization() or 0.0) * 100.0
                        temp_c = float(self._gpu.get_temperature() or 0.0)
                    except Exception:
                        pass
                if read_jetson_power_w is not None:
                    try:
                        power_w = float(read_jetson_power_w() or 0.0)
                    except Exception:
                        power_w = 0.0
                for s in samples:
                    r = s.rec
                    if gpu_pct > r.gpu_util_peak_pct:
                        r.gpu_util_peak_pct = gpu_pct
                    if uma_pct > 0 and self._gpu is not None:
                        try:
                            total_gb = self._gpu.get_memory_total_gb()
                        except Exception:
                            total_gb = 0.0
                        used_gb = total_gb * (uma_pct / 100.0)
                        if used_gb > r.uma_used_gb_peak:
                            r.uma_used_gb_peak = used_gb
                    if temp_c > r.temperature_c_peak:
                        r.temperature_c_peak = temp_c
                    if power_w > 0:
                        if power_w > r.power_w_peak:
                            r.power_w_peak = power_w
                        s.power_samples_sum += power_w
                        s.power_samples_n += 1
                    if s.proc_handle is not None:
                        try:
                            cpu = float(s.proc_handle.cpu_percent(interval=None) or 0.0)
                            if cpu > r.cpu_pct_peak:
                                r.cpu_pct_peak = cpu
                            rss = s.proc_handle.memory_info().rss / (1024 * 1024)
                            if rss > r.rss_mb_peak:
                                r.rss_mb_peak = rss
                        except Exception:
                            pass
            self._stop.wait(self._interval)


_REGISTRY: Optional[_Registry] = None
_REG_LOCK = threading.Lock()


def get_registry(gpu_monitor=None) -> _Registry:
    global _REGISTRY
    if _REGISTRY is None:
        with _REG_LOCK:
            if _REGISTRY is None:
                _REGISTRY = _Registry(gpu_monitor=gpu_monitor)
    elif gpu_monitor is not None and _REGISTRY._gpu is None:
        _REGISTRY.configure(gpu_monitor)
    return _REGISTRY


@contextmanager
def QueryTelemetry(
    *,
    gpu_monitor=None,
    query: str = "",
    source: str = "unknown",
    request_id: str = "",
    tariff_usd_per_kwh: float = 0.12,
):
    """Wrap a query — the registry's single sampler thread captures
    peaks while the body runs; finalisation computes derived fields
    and pushes into the bounded store.
    """
    rec = QueryRecord(
        id=uuid.uuid4().hex[:12],
        started_ts=time.time(),
        query=query,
        source=source,
        request_id=request_id,
    )
    if gpu_monitor is not None:
        try:
            total_gb = gpu_monitor.get_memory_total_gb()
            mem_pct = gpu_monitor.get_memory_utilization() or 0.0
            rec.uma_used_gb_start = total_gb * mem_pct
            rec.uma_used_gb_peak = rec.uma_used_gb_start
        except Exception:
            pass
    sample = get_registry(gpu_monitor=gpu_monitor).register(rec)
    t0 = time.perf_counter()
    try:
        yield rec
    except Exception as e:
        rec.success = False
        rec.error = repr(e)
        raise
    finally:
        wall_s = time.perf_counter() - t0
        rec.wall_ms = wall_s * 1000.0
        rec.ended_ts = time.time()
        get_registry().unregister(rec.id)
        if sample is not None and sample.power_samples_n:
            rec.power_w_mean = sample.power_samples_sum / sample.power_samples_n
            rec.energy_j = rec.power_w_mean * wall_s
            rec.cost_usd = (rec.energy_j / 3.6e6) * tariff_usd_per_kwh
        if gpu_monitor is not None:
            try:
                total_gb = gpu_monitor.get_memory_total_gb()
                mem_pct_end = gpu_monitor.get_memory_utilization() or 0.0
                rec.uma_used_gb_end = total_gb * mem_pct_end
                if rec.uma_used_gb_end > rec.uma_used_gb_peak:
                    rec.uma_used_gb_peak = rec.uma_used_gb_end
            except Exception:
                pass
        if rec.tokens_generated > 0 and wall_s > 0:
            rec.tok_per_sec = rec.tokens_generated / wall_s
        get_store().append(rec)


# Need this for the dataclass type annotation above; placed at the end
# to avoid a forward-ref dance.
from typing import Any   # noqa: E402
