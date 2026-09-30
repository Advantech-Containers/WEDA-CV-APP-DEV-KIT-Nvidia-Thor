"""UMA-aware memory manager for Jetson Thor.

On Jetson (UMA), pinned memory maps to cudaHostAllocMapped — both CPU
and GPU access the same LPDDR5X pages without explicit memcpy.

Allocation modes:
  UNIFIED  — pin_memory=True on Jetson; both CPU/GPU can touch it.
  GPU      — torch.cuda device tensor; fastest for GPU-only stages.
  CPU      — plain numpy; no CUDA overhead.

All buffers are reference-counted and reused when a matching
(name, shape, dtype) request arrives.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

import logging

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

logger = logging.getLogger("weda_optim")


class MemoryType:
    UNIFIED = "unified"
    GPU = "gpu"
    CPU = "cpu"


@dataclass
class BufferRecord:
    name: str
    shape: tuple
    dtype: str
    mem_type: str
    data: Union[np.ndarray, "torch.Tensor"]
    ref_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def acquire(self) -> None:
        with self._lock:
            self.ref_count += 1

    def release(self) -> bool:
        with self._lock:
            self.ref_count = max(0, self.ref_count - 1)
            return self.ref_count == 0

    @property
    def nbytes(self) -> int:
        if isinstance(self.data, np.ndarray):
            return int(self.data.nbytes)
        return int(self.data.nelement() * self.data.element_size())


class MemoryManager:
    """Central buffer allocator. Thread-safe, reuse-first."""

    def __init__(self, device: str = "cuda:0") -> None:
        self._device = device
        self._buffers: dict[str, BufferRecord] = {}
        self._lock = threading.Lock()
        self._total_allocated = 0
        self._reuse_hits = 0
        self._alloc_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(
        self,
        name: str,
        shape: tuple,
        dtype: str = "float32",
        mem_type: str = MemoryType.UNIFIED,
    ) -> Union[np.ndarray, "torch.Tensor"]:
        with self._lock:
            existing = self._buffers.get(name)
            if existing and existing.shape == shape and existing.dtype == dtype:
                existing.acquire()
                self._reuse_hits += 1
                return existing.data

            buf = self._create_buffer(shape, dtype, mem_type)
            rec = BufferRecord(
                name=name, shape=shape, dtype=dtype,
                mem_type=mem_type, data=buf,
            )
            rec.acquire()
            self._buffers[name] = rec
            self._alloc_count += 1
            self._total_allocated += rec.nbytes
            return buf

    def release(self, name: str) -> None:
        with self._lock:
            rec = self._buffers.get(name)
            if rec:
                rec.release()

    def deallocate(self, name: str) -> None:
        with self._lock:
            rec = self._buffers.pop(name, None)
            if rec:
                self._total_allocated -= rec.nbytes
                del rec.data

    def get_buffer(self, name: str) -> Optional[Union[np.ndarray, "torch.Tensor"]]:
        with self._lock:
            rec = self._buffers.get(name)
            return rec.data if rec else None

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_allocated_bytes": self._total_allocated,
                "num_buffers": len(self._buffers),
                "alloc_count": self._alloc_count,
                "reuse_hits": self._reuse_hits,
                "buffers": {
                    n: {"shape": r.shape, "dtype": r.dtype, "refs": r.ref_count}
                    for n, r in self._buffers.items()
                },
            }

    def shutdown(self) -> None:
        with self._lock:
            for rec in self._buffers.values():
                del rec.data
            self._buffers.clear()
            self._total_allocated = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_buffer(
        self, shape: tuple, dtype: str, mem_type: str
    ) -> Union[np.ndarray, "torch.Tensor"]:
        np_dtype = getattr(np, dtype, np.float32)

        if mem_type == MemoryType.CPU or not _HAS_TORCH:
            return np.zeros(shape, dtype=np_dtype)

        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "int32": torch.int32,
            "uint8": torch.uint8,
        }.get(dtype, torch.float32)

        if mem_type == MemoryType.UNIFIED:
            # pin_memory=True → cudaHostAllocMapped on Jetson UMA
            buf = torch.zeros(shape, dtype=torch_dtype, device="cpu")
            try:
                buf = buf.pin_memory()
            except RuntimeError:
                pass  # no CUDA runtime — stays as regular CPU tensor
            return buf

        # GPU
        try:
            return torch.zeros(shape, dtype=torch_dtype, device=self._device)
        except RuntimeError:
            return np.zeros(shape, dtype=np_dtype)
