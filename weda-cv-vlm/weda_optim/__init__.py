"""WEDA Optimization SDK — reusable optimization components for Jetson Thor.

This package provides hardware-aware optimization primitives that any
application can use, independent of the CV-VLM pipeline.

Components:
  BufferPool      — Zero-allocation ring buffer management
  MemoryManager   — UMA/pinned/GPU buffer allocation with reference counting
  Scheduler       — Parameterized CPU/GPU workload placement
  Profiler        — Per-stage latency tracking with percentile stats
  GpuMonitor      — Real-time GPU utilization (nvitop+sysfs / NVML / torch.cuda / mock)
  NvitopDeviceMonitor  — Apache-2.0 device-level monitor (primary on Jetson & discrete)
  NvitopProcessMonitor — App-scoped per-PID GPU/CPU/memory monitor (nvitop+psutil)
  VicResizer      — VIC hardware-accelerated frame resize (GStreamer)
  quantization    — ModelOpt NVFP4/FP8 + bitsandbytes NF4 fallback
  model_loader    — Config-driven HuggingFace model loading
"""

from weda_optim.buffer_pool import BufferPool, PoolConfig, create_default_pools
from weda_optim.gpu_monitor import (
    GpuMonitor,
    NvitopDeviceMonitor,
    NvitopProcessMonitor,
    create_gpu_monitor,
    create_process_monitor,
)
from weda_optim.memory_manager import MemoryManager, MemoryType
from weda_optim.profiler import Profiler
from weda_optim.scheduler import Scheduler, CPU, GPU, AUTO
from weda_optim.vic_offload import VicResizer, is_vic_available

__all__ = [
    "BufferPool",
    "PoolConfig",
    "create_default_pools",
    "GpuMonitor",
    "NvitopDeviceMonitor",
    "NvitopProcessMonitor",
    "create_gpu_monitor",
    "create_process_monitor",
    "MemoryManager",
    "MemoryType",
    "Profiler",
    "Scheduler",
    "CPU",
    "GPU",
    "AUTO",
    "VicResizer",
    "is_vic_available",
]
