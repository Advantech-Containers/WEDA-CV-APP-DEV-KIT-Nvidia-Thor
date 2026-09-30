"""Per-stage latency and memory profiler.

Wall-clock timing in a rolling window (default 100 samples).
Percentile stats (p50, p95, p99) per stage and end-to-end.
Memory snapshots captured on demand.
Summary logged every N frames (configurable).
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import numpy as np

import logging

logger = logging.getLogger("weda_optim")

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_CUDA = False


@dataclass
class StageTimings:
    name: str
    samples: list[float] = field(default_factory=list)
    window: int = 100

    def record(self, elapsed: float) -> None:
        self.samples.append(elapsed)
        if len(self.samples) > self.window:
            self.samples = self.samples[-self.window:]

    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        return float(np.percentile(self.samples, p))

    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else 0.0

    def stats_dict(self) -> dict:
        return {
            "mean_ms": self.mean() * 1000,
            "p50_ms": self.percentile(50) * 1000,
            "p95_ms": self.percentile(95) * 1000,
            "p99_ms": self.percentile(99) * 1000,
            "samples": len(self.samples),
        }


class Profiler:
    def __init__(self, window: int = 100, log_interval: int = 100) -> None:
        self._window = window
        self._log_interval = log_interval
        self._stages: dict[str, StageTimings] = {}
        self._frame_times = StageTimings("end_to_end", window=window)
        self._frame_count = 0
        self._frame_start: float = 0.0
        self._memory_snapshots: list[dict] = []

    def get_or_create_stage(self, name: str) -> StageTimings:
        if name not in self._stages:
            self._stages[name] = StageTimings(name, window=self._window)
        return self._stages[name]

    @contextmanager
    def time_stage(self, name: str) -> Generator[None, None, None]:
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        self.get_or_create_stage(name).record(elapsed)

    def begin_frame(self) -> None:
        self._frame_start = time.perf_counter()

    def end_frame(self) -> None:
        if self._frame_start > 0:
            elapsed = time.perf_counter() - self._frame_start
            self._frame_times.record(elapsed)
        self._frame_count += 1
        if self._log_interval > 0 and self._frame_count % self._log_interval == 0:
            self.log_summary()

    def record_memory_snapshot(self, snapshot: dict) -> None:
        self._memory_snapshots.append(snapshot)
        if len(self._memory_snapshots) > 50:
            self._memory_snapshots = self._memory_snapshots[-25:]

    def summary(self) -> dict:
        result = {
            "frame_count": self._frame_count,
            "end_to_end": self._frame_times.stats_dict(),
            "stages": {},
        }
        for name, st in self._stages.items():
            result["stages"][name] = st.stats_dict()
        if self._frame_times.mean() > 0:
            result["throughput_fps"] = 1.0 / self._frame_times.mean()
        return result

    def log_summary(self) -> None:
        s = self.summary()
        fps = s.get("throughput_fps", 0)
        e2e = s["end_to_end"]
        lines = [
            f"Profiler @ frame {self._frame_count}: "
            f"FPS={fps:.1f}, e2e mean={e2e['mean_ms']:.1f}ms, "
            f"p95={e2e['p95_ms']:.1f}ms"
        ]
        for name, st in sorted(s["stages"].items()):
            lines.append(
                f"  {name}: mean={st['mean_ms']:.1f}ms, "
                f"p95={st['p95_ms']:.1f}ms, p99={st['p99_ms']:.1f}ms"
            )
        logger.info("\n".join(lines))

    def reset(self) -> None:
        self._stages.clear()
        self._frame_times = StageTimings("end_to_end", window=self._window)
        self._frame_count = 0
        self._memory_snapshots.clear()
