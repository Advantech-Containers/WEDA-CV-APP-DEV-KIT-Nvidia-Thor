"""GPU utilization monitor for Jetson Thor and discrete GPUs.

Design constraints on Jetson Thor (UMA):
  - nvidia-smi and pynvml report 0 MiB for GPU memory ("Not Supported")
  - NVML does NOT expose GA10B compute utilisation through the standard
    sysfs node; gpu_utilization() typically reads 0
  - /proc/meminfo + /sys/class/thermal/thermal_zone* give honest UMA
    memory and GPU temperature without AGPL dependencies
  - torch.cuda.memory_allocated() works: per-process GPU memory tracking

Backends (auto-detected, in priority order):
  NvitopDeviceMonitor  — nvitop (NVML) + kernel sysfs fallbacks; Apache-2.0
  TegrastatsMonitor    — Parses tegrastats subprocess for GR3D_FREQ%
  NvmlMonitor          — pynvml for discrete GPUs
  TorchCudaMonitor     — Memory only (no GPU compute util)
  MockMonitor          — Fixed values for tests/dev without GPU

v2.1.2 note: the previous primary backend (JtopMonitor, via AGPL-3.0
jetson-stats) was removed to keep the default runtime licence-clean.
See docs/LICENSES.md for the full package audit.
"""

from __future__ import annotations

import abc
import glob
import os
import re
import subprocess
import threading
import time
from typing import Any, Optional

import logging

logger = logging.getLogger("weda_optim")

try:
    import torch
    _HAS_TORCH = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False

try:
    import pynvml
    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False

try:
    import nvitop  # type: ignore[import]
    _HAS_NVITOP = True
except ImportError:
    _HAS_NVITOP = False

try:
    import psutil  # type: ignore[import]
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ──────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────

class GpuMonitor(abc.ABC):
    """Background-polled GPU monitor with cached readings."""

    def __init__(self, interval: float = 1.0) -> None:
        self._interval = interval
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._gpu_util: float = 0.0
        self._mem_util: float = 0.0
        self._temperature: float = 0.0
        self._process_gpu_mem_mb: float = 0.0
        self._process_ram_mb: float = 0.0
        self._process_cpu_pct: float = 0.0

    def get_gpu_utilization(self) -> float:
        with self._lock:
            return self._gpu_util

    def get_memory_utilization(self) -> float:
        with self._lock:
            return self._mem_util

    def get_temperature(self) -> float:
        with self._lock:
            return self._temperature

    def get_memory_total_gb(self) -> float:
        """Total system memory in GB (UMA = shared CPU+GPU on Jetson)."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / 1024 / 1024
        except Exception:
            pass
        if _HAS_TORCH:
            try:
                return torch.cuda.get_device_properties(0).total_memory / 1024**3
            except Exception:
                pass
        return 0.0

    def get_memory_used_gb(self) -> float:
        """Used memory in GB."""
        total = self.get_memory_total_gb()
        if total > 0:
            return self.get_memory_utilization() * total
        return 0.0

    def get_process_gpu_mem_mb(self) -> float:
        with self._lock:
            return self._process_gpu_mem_mb

    def get_process_ram_mb(self) -> float:
        with self._lock:
            return self._process_ram_mb

    def get_process_cpu_pct(self) -> float:
        with self._lock:
            return self._process_cpu_pct

    def get_top_processes(self, n: int = 3) -> list[dict]:
        """Return up to ``n`` top application processes by resource usage.

        Default: empty list. Override in backend implementations that can
        enumerate per-PID GPU/CPU/memory statistics (see
        ``NvitopProcessMonitor``). The runtime HUD calls this every frame
        and silently hides the panel when the list is empty, so non-Jetson
        and non-nvitop environments get a clean degraded display.
        """
        return []

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._interval * 3)
        self._cleanup()

    def _run(self) -> None:
        while self._running:
            try:
                gpu, mem, temp = self._poll()
                with self._lock:
                    self._gpu_util = gpu
                    self._mem_util = mem
                    self._temperature = temp
            except Exception as e:
                logger.debug("GPU monitor poll error: %s", e)
            time.sleep(self._interval)

    def _update_process_stats(self, gpu_mem_mb: float, ram_mb: float, cpu_pct: float) -> None:
        with self._lock:
            self._process_gpu_mem_mb = gpu_mem_mb
            self._process_ram_mb = ram_mb
            self._process_cpu_pct = cpu_pct

    @abc.abstractmethod
    def _poll(self) -> tuple[float, float, float]:
        ...

    def _cleanup(self) -> None:
        pass

    @staticmethod
    def _get_torch_memory_util() -> float:
        if not _HAS_TORCH:
            return 0.0
        try:
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return allocated / total if total > 0 else 0.0
        except Exception:
            return 0.0


# ──────────────────────────────────────────────────────────
# NvitopDeviceMonitor — primary device-level monitor (Apache-2.0)
# ──────────────────────────────────────────────────────────
#
# Replaces the earlier JtopMonitor, which depended on AGPL-3.0 jetson-stats.
# This monitor uses nvitop (NVML) for anything NVML can surface, and falls
# back to pure kernel sysfs / /proc reads for fields NVML leaves at 'N/A' on
# Jetson UMA (notably temperature; GPU % is usually 0 on Thor because NVML
# does not expose GA10B compute utilisation through the standard sysfs node).
#
# All data sources here are Apache-2.0 / MIT / kernel ABI — no AGPL.


def read_jetson_gpu_temp_c() -> float:
    """Read GPU temperature directly from ``/sys/class/thermal``.

    Jetson exposes several ``thermal_zoneN`` directories. We try, in order:

      1. An explicit GPU zone — e.g. ``GPU-therm``, ``gpu-thermal``.
         This is the most accurate reading of the GA10B die.
      2. The junction temperature — ``tj-therm`` / ``tj``. On Jetson SoCs
         this is the max of the per-island sensors (CPU, GPU, CV, SOC)
         and is usually within a few degrees of the GPU reading —
         acceptable when a dedicated GPU zone is missing.
      3. Any zone whose name contains ``therm`` and whose temperature is
         in a plausible range (20–120 °C). Last-ditch heuristic for
         atypical L4T revisions.

    Returns 0.0 only if ``/sys/class/thermal`` is absent (non-Linux host)
    or every zone read fails.
    """
    try:
        zones = glob.glob("/sys/class/thermal/thermal_zone*")
    except Exception:
        return 0.0

    # Read (type, temp_c) for every zone once, defensively.
    snapshot: list[tuple[str, float]] = []
    for zone in zones:
        try:
            with open(os.path.join(zone, "type")) as f:
                zname = f.read().strip().lower()
            with open(os.path.join(zone, "temp")) as f:
                millideg = int(f.read().strip())
            snapshot.append((zname, millideg / 1000.0))
        except Exception:
            continue

    if not snapshot:
        return 0.0

    # Priority 1: explicit GPU-named zone.
    gpu_keys = ("gpu-therm", "gpu_therm", "gpu-thermal", "gpu thermal", "gpu")
    for zname, tc in snapshot:
        if any(k in zname for k in gpu_keys):
            return tc

    # Priority 2: junction temperature.
    for zname, tc in snapshot:
        if "tj" in zname:  # matches "tj-therm", "Tj", "tjunction", etc.
            return tc

    # Priority 3: any *therm* zone with a plausible GPU-ish reading.
    # This is a safety net for older L4T or custom boards that label
    # zones unpredictably. Excludes CPU since that's usually higher than
    # GPU under load and would be misleading.
    plausible: list[float] = []
    for zname, tc in snapshot:
        if "therm" in zname and "cpu" not in zname and 20.0 <= tc <= 120.0:
            plausible.append(tc)
    if plausible:
        # Median is more robust than max — avoids spiking on a single hot sensor.
        plausible.sort()
        return plausible[len(plausible) // 2]

    return 0.0


# Private alias preserved so other modules importing the original name keep
# working without churn in the same commit. New callers should use the
# un-prefixed name.
_read_jetson_gpu_temp_c = read_jetson_gpu_temp_c


# ──────────────────────────────────────────────────────────────────
# Jetson AGX Thor power-rail reader (added for Phase H telemetry)
# ──────────────────────────────────────────────────────────────────
#
# Thor exposes power monitoring via two I²C-attached sensors:
#   - INA3221 at 2-0040 → VDD_GPU, VDD_CPU_SOC_MSS, VIN_SYS_5V0
#   - INA238  at 2-0044 → VIN total board
#
# The hwmon* index is non-deterministic across boots, so we glob the
# directory and match by `name` file content. Cache the resolved paths
# for 60 s — re-resolve on read failure.
#
# Pure kernel ABI — no AGPL libraries, no NVML (which doesn't expose
# this on Thor), no jetson-stats (deliberately excluded for license).

_POWER_PATH_CACHE: dict[str, tuple[float, list[str]]] = {}
_POWER_PATH_TTL_S = 60.0


def _discover_power_input_paths() -> list[str]:
    """Find every `in_power*_input` file across Thor's two INA hwmon nodes.

    Returns the list of full paths (e.g.
    ``/sys/bus/i2c/devices/2-0040/hwmon/hwmon3/in_power1_input``). Empty
    list when none are present (e.g. running off-Thor).
    """
    paths: list[str] = []
    for i2c_addr in ("2-0040", "2-0044"):
        base = f"/sys/bus/i2c/devices/{i2c_addr}/hwmon"
        if not os.path.isdir(base):
            continue
        for hw in glob.glob(os.path.join(base, "hwmon*")):
            for f in glob.glob(os.path.join(hw, "in_power*_input")):
                paths.append(f)
    return paths


def _cached_power_paths() -> list[str]:
    """Return cached power-input paths, re-discovering when stale or empty."""
    now = time.monotonic()
    cached = _POWER_PATH_CACHE.get("paths")
    if cached is not None:
        ts, paths = cached
        if now - ts < _POWER_PATH_TTL_S and paths:
            return paths
    paths = _discover_power_input_paths()
    _POWER_PATH_CACHE["paths"] = (now, paths)
    return paths


def read_jetson_power_w() -> float:
    """Total instantaneous board power in watts.

    Sums every discovered ``in_power*_input`` file across the INA3221
    + INA238 hwmon nodes. Returns 0.0 on non-Thor systems or if the
    path discovery fails entirely.

    The kernel exposes these as microwatts; we divide by 1e6 for W.
    On Thor at idle this returns roughly 15-25 W; under inference
    load it climbs to 50-90 W. tegrastats reads the same sensors and
    its `VDD_*` lines should match within ±0.5 W.
    """
    paths = _cached_power_paths()
    if not paths:
        return 0.0
    total_uw = 0.0
    failed = False
    for p in paths:
        try:
            with open(p) as f:
                total_uw += float(f.read().strip())
        except (OSError, ValueError):
            failed = True
            continue
    if failed:
        # A read failed — invalidate the cache so next call re-discovers
        # (the hwmon index may have changed after a kernel reload).
        _POWER_PATH_CACHE.pop("paths", None)
    return total_uw / 1_000_000.0


def read_jetson_power_breakdown() -> dict[str, float]:
    """Per-rail power breakdown for the API's "raw power" view.

    Returns a dict keyed by the rail label from each hwmon node, with
    values in watts. Empty dict on non-Thor systems.

    Example return on a healthy Thor::

        {"VDD_GPU": 12.4, "VDD_CPU_SOC_MSS": 8.1, "VIN_SYS_5V0": 2.3,
         "VIN": 32.8}

    The labels come from the sibling `_label` files (e.g. for
    ``in_power1_input`` the label is at ``in_power1_label``).
    """
    out: dict[str, float] = {}
    for p in _cached_power_paths():
        try:
            label_path = p.replace("_input", "_label")
            try:
                with open(label_path) as f:
                    label = f.read().strip()
            except (OSError, FileNotFoundError):
                # No label file — derive from filename. Use the parent
                # directory's `name` file as a prefix (e.g. ina3221).
                stem = os.path.basename(p).replace("_input", "")
                parent_name = ""
                try:
                    with open(os.path.join(os.path.dirname(p), "name")) as f:
                        parent_name = f.read().strip()
                except OSError:
                    pass
                label = f"{parent_name}_{stem}" if parent_name else stem
            with open(p) as f:
                uw = float(f.read().strip())
            out[label] = out.get(label, 0.0) + uw / 1_000_000.0
        except (OSError, ValueError):
            continue
    return out


def _read_uma_mem_util() -> float:
    """Read Jetson UMA memory utilisation from /proc/meminfo.

    CPU and GPU share the same LPDDR pool, so "GPU memory utilisation" on
    Jetson is simply total system memory utilisation. Returns 0.0 on error.
    """
    try:
        total_kb = avail_kb = None
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
                if total_kb is not None and avail_kb is not None:
                    break
        if total_kb and avail_kb is not None and total_kb > 0:
            return max(0.0, min(1.0, 1.0 - (avail_kb / total_kb)))
    except Exception:
        pass
    return 0.0


class NvitopDeviceMonitor(GpuMonitor):
    """Device-level GPU telemetry via nvitop + kernel sysfs fallbacks.

    nvitop reads NVML. On Jetson UMA most device-level fields return 'N/A'
    (sysfs node not populated). This class supplements with direct /sys/
    and /proc/ reads — no AGPL libraries involved. Backend identifier in
    the HUD: NvitopDeviceMonitor.
    """

    def __init__(self, interval: float = 1.0) -> None:
        super().__init__(interval)
        self._device = None

    def start(self) -> None:
        if not _HAS_NVITOP:
            return
        try:
            devices = nvitop.Device.all()
            self._device = devices[0] if devices else None
        except Exception as e:
            logger.debug("NvitopDeviceMonitor: enumerate failed: %s", e)
            self._device = None
        super().start()

    @staticmethod
    def _numeric(v, default=0.0):
        """nvitop occasionally returns the string 'N/A' instead of a number."""
        if v is None or v == "N/A" or v == "":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _poll(self) -> tuple[float, float, float]:
        # GPU utilisation — NVML first; on Jetson UMA this is usually 0.
        gpu_util = 0.0
        if self._device is not None:
            try:
                raw = self._device.gpu_utilization()
                v = self._numeric(raw, 0.0)
                gpu_util = v / 100.0 if v > 1.0 else v
                gpu_util = max(0.0, min(1.0, gpu_util))
            except Exception:
                pass

        # Memory — always via /proc/meminfo (UMA is shared). NVML would
        # return 0 for GPU memory on Jetson and a wrong absolute value on
        # discrete GPUs relative to process-visible pool, so /proc is more
        # honest for our HUD's "how loaded is the box" question.
        mem_util = _read_uma_mem_util()
        if mem_util == 0.0:
            # Discrete-GPU fallback — use torch.cuda memory_allocated ratio.
            mem_util = self._get_torch_memory_util()

        # Temperature — NVML first; on Jetson fall back to thermal_zone.
        temp = 0.0
        if self._device is not None:
            try:
                temp = self._numeric(self._device.temperature(), 0.0)
            except Exception:
                temp = 0.0
        if temp == 0.0:
            temp = _read_jetson_gpu_temp_c()

        return gpu_util, mem_util, temp

    def _cleanup(self) -> None:
        self._device = None


# ──────────────────────────────────────────────────────────
# Tegrastats backend (Jetson fallback)
# ──────────────────────────────────────────────────────────

_GR3D_PATTERN = re.compile(r"GR3D_FREQ\s+(\d+)%")
_TJ_PATTERN = re.compile(r"\btj(?:_max)?@([\d.]+)\s*C", re.IGNORECASE)
_GPU_TEMP_PATTERN = re.compile(r"\b(?:gpu|GPU)[a-z0-9_]*@([\d.]+)\s*C")
_ANY_TEMP_PATTERN = re.compile(r"\b(?:cpu|soc|aux|thermal)[a-z0-9_]*@([\d.]+)\s*C")


class TegrastatsMonitor(GpuMonitor):
    """Parses tegrastats output for GPU utilization on Jetson."""

    def __init__(self, interval: float = 1.0) -> None:
        super().__init__(interval)
        self._proc: Optional[subprocess.Popen] = None
        self._line_lock = threading.Lock()
        self._latest_line: str = ""
        self._reader_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        interval_ms = max(100, int(self._interval * 1000))
        for cmd in ("/usr/bin/tegrastats", "tegrastats"):
            try:
                self._proc = subprocess.Popen(
                    [cmd, "--interval", str(interval_ms)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
                time.sleep(0.15)
                if self._proc.poll() is not None:
                    self._proc = None
                    continue
                break
            except FileNotFoundError:
                continue

        if self._proc is None:
            super().start()
            return

        self._reader_thread = threading.Thread(target=self._read_lines, daemon=True)
        self._reader_thread.start()
        super().start()

    def _read_lines(self) -> None:
        while self._running and self._proc and self._proc.stdout:
            line = self._proc.stdout.readline()
            if not line:
                break
            with self._line_lock:
                self._latest_line = line.strip()

    def _poll(self) -> tuple[float, float, float]:
        with self._line_lock:
            line = self._latest_line
        mem_util = _read_uma_mem_util()
        if mem_util == 0.0:
            mem_util = self._get_torch_memory_util()
        temperature = read_jetson_gpu_temp_c()
        if temperature == 0.0 and line:
            temperature = self._parse_temperature(line)
        if not line:
            return 0.0, mem_util, temperature
        gpu_util = self._parse_gpu_util(line)
        return gpu_util, mem_util, temperature

    def _cleanup(self) -> None:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    @staticmethod
    def _parse_gpu_util(line: str) -> float:
        m = _GR3D_PATTERN.search(line)
        return int(m.group(1)) / 100.0 if m else 0.0

    @staticmethod
    def _parse_temperature(line: str) -> float:
        for pat in (_TJ_PATTERN, _GPU_TEMP_PATTERN, _ANY_TEMP_PATTERN):
            m = pat.search(line)
            if m:
                try:
                    v = float(m.group(1))
                    if 10.0 <= v <= 130.0:
                        return v
                except (TypeError, ValueError):
                    continue
        return 0.0


# ──────────────────────────────────────────────────────────
# NVML backend (discrete GPUs)
# ──────────────────────────────────────────────────────────

class NvmlMonitor(GpuMonitor):
    def __init__(self, interval: float = 1.0) -> None:
        super().__init__(interval)
        self._handle = None

    def start(self) -> None:
        if not _HAS_PYNVML:
            return
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            super().start()
        except Exception:
            pass

    def _poll(self) -> tuple[float, float, float]:
        if self._handle is None:
            return 0.0, 0.0, 0.0
        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = 0.0
        return util.gpu / 100.0, mem.used / mem.total if mem.total > 0 else 0.0, float(temp)

    def _cleanup(self) -> None:
        if _HAS_PYNVML and self._handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────
# nvitop per-PID process monitor (application-scoped)
# ──────────────────────────────────────────────────────────
#
# Separate from the device-level GpuMonitor hierarchy: this monitor does NOT
# poll a background thread and does NOT contribute GPU %, memory, or
# temperature to the HUD. Its single job is to enumerate the processes in
# the CV-VLM application's own process tree (orchestrator + children) and
# return their per-PID resource usage.
#
# On Jetson UMA, NVML frequently fails to attribute GPU memory to individual
# processes. When that happens, ``get_top_processes`` falls back to a
# psutil-only view so the HUD still shows CPU % + RSS MB per PID.


def _as_num(v, default=0):
    """Coerce nvitop values (which may be the string 'N/A') to numeric."""
    try:
        if v is None or v == "N/A":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


class NvitopProcessMonitor(GpuMonitor):
    """Application-scoped per-PID monitor. Overrides ``get_top_processes``.

    The base-class device-level polling (``_poll``) is a no-op — device
    metrics are expected to come from the primary ``GpuMonitor`` backend
    (typically ``NvitopDeviceMonitor`` on Jetson). Instantiate this in
    addition to the device monitor, not instead of it.
    """

    def __init__(self, interval: float = 1.0) -> None:
        super().__init__(interval=interval)
        self._nvitop_device = None

    def start(self) -> None:
        # Resolve a device handle once so each call avoids the enumeration
        # cost. If nvitop enumerates nothing, stay silent and rely on the
        # psutil fallback inside get_top_processes.
        if _HAS_NVITOP:
            try:
                devs = nvitop.Device.all()
                self._nvitop_device = devs[0] if devs else None
            except Exception as e:
                logger.debug("NvitopProcessMonitor: enumerate failed: %s", e)
        # No background thread — we're pull-only.
        self._running = True

    def stop(self) -> None:
        self._running = False
        self._nvitop_device = None

    def _poll(self) -> tuple[float, float, float]:
        # Intentional no-op: device metrics come from another monitor.
        return 0.0, 0.0, 0.0

    # ── helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _role_for(cmdline: str) -> str:
        """Map a cmdline to a short, HUD-friendly role label."""
        if not cmdline:
            return ""
        cl = cmdline.lower()
        if "app.main" in cl or "orchestrator" in cl:
            return "orchestrator"
        if "wise-bench" in cl:
            return "wise-bench"
        if "gst-launch" in cl or "gstreamer" in cl:
            return "gst"
        if "reasoning" in cl or "qwen" in cl:
            return "vlm"
        if "physical_reasoner" in cl or "cosmos-reason" in cl.replace("_", "-"):
            return "reasoner"
        if "detector" in cl or "yolo" in cl or "ultralytics" in cl:
            return "detector"
        return ""

    def _own_pid_set(self) -> set[int]:
        """PID set for the current app process tree."""
        pids = {os.getpid()}
        if not _HAS_PSUTIL:
            return pids
        try:
            for c in psutil.Process().children(recursive=True):
                pids.add(c.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return pids

    def _format_nvitop(self, p) -> dict:
        try:
            pid = int(p.pid)
        except Exception:
            pid = -1
        try:
            name = (p.name() or "?")[:32]
        except Exception:
            name = "?"
        try:
            cmd = p.command() or ""
        except Exception:
            cmd = ""
        gpu_mem_b = _as_num(p.gpu_memory() if hasattr(p, "gpu_memory") else None, 0)
        gpu_mem_mb = gpu_mem_b / (1024 * 1024) if gpu_mem_b else 0.0
        try:
            cpu_pct = _as_num(p.cpu_percent(), 0.0)
        except Exception:
            cpu_pct = 0.0
        mem_mb = 0.0
        if _HAS_PSUTIL:
            try:
                mem_mb = psutil.Process(pid).memory_info().rss / (1024 * 1024)
            except Exception:
                pass
        return {
            "pid": pid,
            "name": name,
            "gpu_mem_mb": gpu_mem_mb if gpu_mem_mb else None,
            "cpu_pct": round(cpu_pct, 1),
            "mem_mb": round(mem_mb, 1),
            "role": self._role_for(cmd or name),
        }

    def _psutil_fallback(self, own_pids: set[int], n: int) -> list[dict]:
        """psutil-only view when NVML cannot attribute GPU memory per PID."""
        if not _HAS_PSUTIL:
            return []
        out: list[dict] = []
        for pid in own_pids:
            try:
                proc = psutil.Process(pid)
                with proc.oneshot():
                    info = {
                        "pid": pid,
                        "name": (proc.name() or "?")[:32],
                        "gpu_mem_mb": None,
                        "cpu_pct": round(proc.cpu_percent(interval=None), 1),
                        "mem_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
                        "role": self._role_for(" ".join(proc.cmdline())),
                    }
                out.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        out.sort(key=lambda r: (r["mem_mb"], r["cpu_pct"]), reverse=True)
        return out[:n]

    # ── public ──────────────────────────────────────────────────────────
    def get_top_processes(self, n: int = 3) -> list[dict]:
        own = self._own_pid_set()
        if not _HAS_NVITOP or self._nvitop_device is None:
            return self._psutil_fallback(own, n)

        try:
            procs = self._nvitop_device.processes() or {}
        except Exception as e:
            logger.debug("NvitopProcessMonitor: processes() failed: %s", e)
            return self._psutil_fallback(own, n)

        mine = [p for pid, p in procs.items() if pid in own]
        if not mine:
            # NVML did not attribute GPU usage to any of our PIDs (UMA case).
            return self._psutil_fallback(own, n)

        mine.sort(
            key=lambda p: (
                _as_num(p.gpu_memory() if hasattr(p, "gpu_memory") else 0, 0),
                _as_num(p.cpu_percent() if hasattr(p, "cpu_percent") else 0, 0),
            ),
            reverse=True,
        )
        return [self._format_nvitop(p) for p in mine[:n]]


class _NullProcessMonitor(GpuMonitor):
    """Returns an empty list — used when nvitop + psutil are both absent."""

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def _poll(self) -> tuple[float, float, float]:
        return 0.0, 0.0, 0.0


# ──────────────────────────────────────────────────────────
# torch.cuda fallback
# ──────────────────────────────────────────────────────────

class TorchCudaMonitor(GpuMonitor):
    def _poll(self) -> tuple[float, float, float]:
        return 0.0, self._get_torch_memory_util(), 0.0


# ──────────────────────────────────────────────────────────
# Mock backend
# ──────────────────────────────────────────────────────────

class MockMonitor(GpuMonitor):
    def __init__(self, gpu_util=0.5, mem_util=0.3, temperature=45.0) -> None:
        super().__init__(interval=0)
        self._gpu_util = gpu_util
        self._mem_util = mem_util
        self._temperature = temperature

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def _poll(self) -> tuple[float, float, float]:
        return self._gpu_util, self._mem_util, self._temperature


# ──────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────

def create_gpu_monitor(config: Any) -> GpuMonitor:
    """Auto-detect and create the best available device-level GPU monitor.

    Priority (all Apache-2.0 / MIT / kernel ABI — no AGPL):
        1. NvitopDeviceMonitor  — nvitop + kernel sysfs fallbacks (Jetson & discrete)
        2. TegrastatsMonitor    — Jetson without nvitop (tegrastats CLI exists)
        3. NvmlMonitor          — Discrete GPU with pynvml
        4. TorchCudaMonitor     — Any CUDA GPU, memory-only
        5. MockMonitor          — Fixed values for tests / dev without GPU
    """
    interval = config.gpu_monitor_interval

    # 1. nvitop + sysfs — primary device monitor everywhere nvitop imports.
    if _HAS_NVITOP:
        mon = NvitopDeviceMonitor(interval=interval)
        mon.start()
        return mon

    # 2. tegrastats (Jetson fallback when nvitop is unavailable)
    for path in ("/usr/bin/tegrastats",):
        if os.path.isfile(path):
            mon = TegrastatsMonitor(interval=interval)
            mon.start()
            return mon
    try:
        subprocess.run(["tegrastats", "--help"], capture_output=True, timeout=2)
        mon = TegrastatsMonitor(interval=interval)
        mon.start()
        return mon
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # 3. pynvml (discrete GPU)
    if _HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            mon = NvmlMonitor(interval=interval)
            mon.start()
            return mon
        except Exception:
            pass

    # 4. torch.cuda
    if _HAS_TORCH:
        mon = TorchCudaMonitor(interval=interval)
        mon.start()
        return mon

    # 5. Mock
    return MockMonitor()


def create_process_monitor() -> GpuMonitor:
    """Construct the best available per-PID process monitor.

    Returns an ``NvitopProcessMonitor`` when ``nvitop`` is importable
    (full per-PID GPU view, with psutil fallback for UMA), or a
    ``_NullProcessMonitor`` otherwise. Callers can always invoke
    ``get_top_processes()`` without exception handling.

    This is independent of ``create_gpu_monitor()``: the orchestrator
    typically holds one instance of each — a device-level monitor for the
    main HUD telemetry and a process monitor for the per-PID sub-panel.
    """
    if _HAS_NVITOP:
        mon = NvitopProcessMonitor()
        mon.start()
        return mon
    return _NullProcessMonitor()
