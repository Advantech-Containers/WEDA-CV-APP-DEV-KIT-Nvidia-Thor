"""CPU / GPU workload scheduler — parameterized, application-agnostic.

Three placement rules:
  1. Fixed stages: configured by the application (e.g. detection → GPU).
  2. Saturation guard: GPU util > threshold or max concurrent GPU → CPU.
  3. Latency-based: AUTO stages route to GPU only if speedup >= min_speedup.

Usage:
    scheduler = Scheduler(
        fixed_placements={"detection": "gpu", "tracking": "cpu"},
        auto_stages={"depth", "reasoning"},
    )
    with scheduler.scheduled_execution("detection") as placement:
        # placement is "cpu" or "gpu"
        run_stage(placement)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

logger = logging.getLogger("weda_optim")

MIN_SAMPLES_FOR_LATENCY_ROUTING = 5

# Canonical placement strings
CPU = "cpu"
GPU = "gpu"
AUTO = "auto"


@dataclass
class StageLatencyStats:
    cpu_times: list[float] = field(default_factory=list)
    gpu_times: list[float] = field(default_factory=list)

    def mean_cpu(self) -> float:
        return sum(self.cpu_times) / len(self.cpu_times) if self.cpu_times else float("inf")

    def mean_gpu(self) -> float:
        return sum(self.gpu_times) / len(self.gpu_times) if self.gpu_times else float("inf")


class Scheduler:
    """Application-agnostic CPU/GPU workload scheduler.

    Args:
        fixed_placements: dict mapping stage name → "cpu" or "gpu" (always).
        auto_stages: set of stage names eligible for dynamic routing.
        gpu_util_threshold: route AUTO stages to CPU above this GPU utilization.
        max_concurrent_gpu: max simultaneous GPU stages before CPU fallback.
        min_speedup: minimum GPU/CPU speedup for AUTO → GPU promotion.
        gpu_monitor: optional GpuMonitor instance for real-time utilization.
    """

    def __init__(
        self,
        fixed_placements: Optional[dict[str, str]] = None,
        auto_stages: Optional[set[str]] = None,
        gpu_util_threshold: float = 0.85,
        max_concurrent_gpu: int = 2,
        min_speedup: float = 2.0,
        gpu_monitor: Optional[Any] = None,
    ) -> None:
        self._fixed: dict[str, str] = {
            k: v.lower() for k, v in (fixed_placements or {}).items()
        }
        self._auto: set[str] = auto_stages or set()
        self._gpu_util_threshold = gpu_util_threshold
        self._max_concurrent_gpu = max_concurrent_gpu
        self._min_speedup = min_speedup
        self._gpu_monitor = gpu_monitor

        self._lock = threading.Lock()
        self._gpu_active: int = 0
        self._latency: dict[str, StageLatencyStats] = defaultdict(StageLatencyStats)
        self._decision_log: list[tuple[str, str]] = []

    def decide_placement(self, stage: str) -> str:
        """Return "cpu" or "gpu" for the given stage name."""
        stage_key = stage if isinstance(stage, str) else str(stage.value if hasattr(stage, "value") else stage)

        # Rule 1 — fixed stages
        fixed = self._fixed.get(stage_key)
        if fixed is not None:
            self._log_decision(stage_key, fixed)
            return fixed

        # Rule 2a — saturation guard (concurrent count)
        with self._lock:
            if self._gpu_active >= self._max_concurrent_gpu:
                self._log_decision(stage_key, CPU)
                return CPU

        # Rule 2b — saturation guard (real GPU utilization)
        if self._gpu_monitor is not None:
            util = self._gpu_monitor.get_gpu_utilization()
            if util > self._gpu_util_threshold:
                self._log_decision(stage_key, CPU)
                return CPU

        # Rule 3 — latency-based for AUTO stages
        if stage_key in self._auto:
            stats = self._latency[stage_key]
            if (
                len(stats.cpu_times) >= MIN_SAMPLES_FOR_LATENCY_ROUTING
                and len(stats.gpu_times) >= MIN_SAMPLES_FOR_LATENCY_ROUTING
            ):
                speedup = stats.mean_cpu() / stats.mean_gpu()
                if speedup >= self._min_speedup:
                    self._log_decision(stage_key, GPU)
                    return GPU
                self._log_decision(stage_key, CPU)
                return CPU
            # Not enough data — default GPU for AUTO
            self._log_decision(stage_key, GPU)
            return GPU

        self._log_decision(stage_key, CPU)
        return CPU

    def record_latency(self, stage: str, placement: str, elapsed: float) -> None:
        stage_key = stage if isinstance(stage, str) else str(stage.value if hasattr(stage, "value") else stage)
        stats = self._latency[stage_key]
        window = 50
        if placement == GPU:
            stats.gpu_times.append(elapsed)
            if len(stats.gpu_times) > window:
                stats.gpu_times = stats.gpu_times[-window:]
        else:
            stats.cpu_times.append(elapsed)
            if len(stats.cpu_times) > window:
                stats.cpu_times = stats.cpu_times[-window:]

    @contextmanager
    def gpu_context(self) -> Generator[None, None, None]:
        with self._lock:
            self._gpu_active += 1
        try:
            yield
        finally:
            with self._lock:
                self._gpu_active = max(0, self._gpu_active - 1)

    @contextmanager
    def scheduled_execution(self, stage: str) -> Generator[str, None, None]:
        """Context manager: yields "cpu" or "gpu", records latency on exit."""
        placement = self.decide_placement(stage)
        t0 = time.perf_counter()
        if placement == GPU:
            with self.gpu_context():
                yield placement
        else:
            yield placement
        elapsed = time.perf_counter() - t0
        self.record_latency(stage, placement, elapsed)

    def stats(self) -> dict:
        with self._lock:
            gpu_active = self._gpu_active
        result: dict = {"gpu_active": gpu_active, "stages": {}}
        for stage_key, lat in self._latency.items():
            result["stages"][stage_key] = {
                "cpu_samples": len(lat.cpu_times),
                "gpu_samples": len(lat.gpu_times),
                "mean_cpu_ms": lat.mean_cpu() * 1000,
                "mean_gpu_ms": lat.mean_gpu() * 1000,
            }
        if self._gpu_monitor is not None:
            result["gpu_utilization"] = self._gpu_monitor.get_gpu_utilization()
            result["gpu_memory"] = self._gpu_monitor.get_memory_utilization()
            result["gpu_temperature"] = self._gpu_monitor.get_temperature()
        return result

    def _log_decision(self, stage: str, placement: str) -> None:
        self._decision_log.append((stage, placement))
        if len(self._decision_log) > 500:
            self._decision_log = self._decision_log[-250:]
