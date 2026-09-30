"""Pre-allocated buffer pool — zero runtime allocation.

Each pool is a fixed-size ring of identically-shaped buffers.
Acquire returns the next free slot; release marks it reusable.
Overflow allocation is supported but logged as a warning.
In steady state, overflow should never fire.
"""

from __future__ import annotations

import threading
from collections import deque
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


@dataclass
class PoolSlot:
    index: int
    data: Union[np.ndarray, "torch.Tensor"]
    in_use: bool = False


@dataclass
class PoolConfig:
    name: str
    shape: tuple
    dtype: str = "uint8"
    count: int = 4
    pinned: bool = True


class BufferPool:
    """Manages multiple named pools of pre-allocated buffers."""

    def __init__(self) -> None:
        self._pools: dict[str, list[PoolSlot]] = {}
        self._free_indices: dict[str, deque[int]] = {}
        self._configs: dict[str, PoolConfig] = {}
        self._lock = threading.Lock()
        self._overflow_count: dict[str, int] = {}
        self._total_bytes = 0

    def register_pool(self, config: PoolConfig) -> None:
        with self._lock:
            slots: list[PoolSlot] = []
            for i in range(config.count):
                buf = self._alloc_one(config.shape, config.dtype, config.pinned)
                slots.append(PoolSlot(index=i, data=buf))
                self._total_bytes += _nbytes(buf)
            self._pools[config.name] = slots
            self._free_indices[config.name] = deque(range(config.count))
            self._configs[config.name] = config
            self._overflow_count[config.name] = 0

    def acquire(self, pool_name: str) -> tuple[int, Union[np.ndarray, "torch.Tensor"]]:
        with self._lock:
            free = self._free_indices.get(pool_name)
            if free:
                idx = free.popleft()
                slot = self._pools[pool_name][idx]
                slot.in_use = True
                return idx, slot.data

            # Overflow — allocate a temporary buffer
            cfg = self._configs.get(pool_name)
            if cfg is None:
                raise KeyError(f"Pool '{pool_name}' not registered")
            self._overflow_count[pool_name] += 1
            logger.warning(
                "Pool '%s' exhausted, overflow alloc #%d",
                pool_name, self._overflow_count[pool_name],
            )
            buf = self._alloc_one(cfg.shape, cfg.dtype, cfg.pinned)
            idx = len(self._pools[pool_name])
            self._pools[pool_name].append(PoolSlot(index=idx, data=buf, in_use=True))
            self._total_bytes += _nbytes(buf)
            return idx, buf

    def release(self, pool_name: str, index: int) -> None:
        with self._lock:
            slots = self._pools.get(pool_name)
            if slots and index < len(slots):
                slots[index].in_use = False
                free = self._free_indices[pool_name]
                if index not in free:
                    free.append(index)

    def release_all(self, tokens: list[tuple[str, int]]) -> None:
        for pool_name, idx in tokens:
            self.release(pool_name, idx)

    def stats(self) -> dict:
        with self._lock:
            result = {"total_bytes": self._total_bytes, "pools": {}}
            for name, slots in self._pools.items():
                in_use = sum(1 for s in slots if s.in_use)
                result["pools"][name] = {
                    "capacity": len(slots),
                    "in_use": in_use,
                    "free": len(slots) - in_use,
                    "overflow": self._overflow_count.get(name, 0),
                }
            return result

    def shutdown(self) -> None:
        with self._lock:
            for slots in self._pools.values():
                for slot in slots:
                    del slot.data
            self._pools.clear()
            self._free_indices.clear()
            self._total_bytes = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _alloc_one(
        shape: tuple, dtype: str, pinned: bool
    ) -> Union[np.ndarray, "torch.Tensor"]:
        np_dtype = getattr(np, dtype, np.uint8)
        if not pinned or not _HAS_TORCH:
            return np.zeros(shape, dtype=np_dtype)

        torch_dtype = {
            "uint8": torch.uint8,
            "uint16": torch.int16,  # torch has no uint16
            "float32": torch.float32,
            "float16": torch.float16,
            "int32": torch.int32,
        }.get(dtype, torch.uint8)
        buf = torch.zeros(shape, dtype=torch_dtype, device="cpu")
        try:
            buf = buf.pin_memory()
        except RuntimeError:
            pass
        return buf


def _nbytes(buf: Union[np.ndarray, "torch.Tensor"]) -> int:
    if isinstance(buf, np.ndarray):
        return int(buf.nbytes)
    return int(buf.nelement() * buf.element_size())


def create_default_pools(
    width: int = 1920,
    height: int = 1080,
    pool_depth: int = 4,
) -> BufferPool:
    """Instantiate the standard set of pools for the pipeline."""
    pool = BufferPool()
    configs = [
        PoolConfig("frame_rgb", (height, width, 3), "uint8", pool_depth, True),
        PoolConfig("frame_depth", (height, width), "uint16", pool_depth, True),
        PoolConfig("frame_resized", (640, 640, 3), "uint8", 2, True),
        PoolConfig("det_output", (100, 7), "float32", 2, True),
        PoolConfig("depth_colorized", (height, width, 3), "uint8", 2, True),
    ]
    for cfg in configs:
        pool.register_pool(cfg)
    return pool
