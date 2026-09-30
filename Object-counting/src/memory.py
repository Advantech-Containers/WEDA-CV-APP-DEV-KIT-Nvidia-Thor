"""
WEDA-CV UMA Memory Manager
===========================
Zero-copy and low-copy memory optimization for Jetson UMA and dGPU.

On Jetson (UMA):
  CPU and GPU share the same physical DRAM. However, PyTorch's default
  allocator uses cudaMalloc (device-only virtual mapping), so .cpu()/.cuda()
  still triggers a virtual address remap + cache flush. This module eliminates
  that overhead via:

  1. CUDA pinned (page-locked) host memory — maps directly into GPU address
     space. torch.from_numpy() on a pinned buffer → GPU access without copy.
  2. Pre-allocated frame buffers — reuse the same GPU tensor across frames,
     eliminating per-frame cudaMalloc/cudaFree overhead.
  3. CUDA stream pipelining — overlap frame upload with previous frame's
     inference via non-default streams.

On dGPU (PCIe):
  Same API, but pinned memory enables DMA transfers that bypass CPU cache.
  Pre-allocated buffers eliminate PCIe round-trips for cudaMalloc.

Backend compatibility:
  - PyTorch (.pt): tensors stay on CUDA, plot() operates on GPU tensors
  - TensorRT (.engine): Ultralytics binds I/O to CUDA memory automatically;
    this module optimizes the frame input path and annotated output path
  - ONNX (.onnx): CUDAExecutionProvider receives GPU tensors when available;
    same input/output path optimization applies
  - All formats: the pre-allocated buffers and pinned memory work identically
    regardless of which backend Ultralytics uses internally

This file is part of the standard engine. Do not modify.
"""

import logging
import numpy as np
try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
log = logging.getLogger("WEDA-CV")
# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
def detect_platform() -> dict:
    """
    Detect GPU platform and UMA capability.
    Returns dict with:
      - is_cuda: bool
      - is_uma: bool (Jetson unified memory)
      - device_name: str
      - total_memory_mb: int
      - cuda_version: str
    """
    info = {
        "is_cuda": False,
        "is_uma": False,
        "device_name": "cpu",
        "total_memory_mb": 0,
        "cuda_version": "N/A",
    }

    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return info

    info["is_cuda"] = True
    info["device_name"] = torch.cuda.get_device_name(0)
    info["total_memory_mb"] = torch.cuda.get_device_properties(0).total_memory >> 20
    info["cuda_version"] = torch.version.cuda or "N/A"
    props = torch.cuda.get_device_properties(0)
    jetson_names = ["orin", "xavier", "tegra", "thor", "jetson"]
    name_lower = info["device_name"].lower()
    info["is_uma"] = (
        getattr(props, "is_integrated", False)
        or any(j in name_lower for j in jetson_names)
    )
    return info
# ---------------------------------------------------------------------------
# Pinned memory frame buffer
# ---------------------------------------------------------------------------
class PinnedFrameBuffer:
    """
    Pre-allocated CUDA pinned (page-locked) host buffer for frame I/O.

    On Jetson UMA:
      Pinned memory is directly accessible from GPU without copy.
      np.ndarray backed by pinned storage → torch.from_numpy().cuda()
      triggers zero-copy mapping, not a DMA transfer.

    On dGPU:
      Pinned memory enables async DMA via cudaMemcpyAsync, overlapping
      transfer with compute on a non-default stream.

    Usage:
      buf = PinnedFrameBuffer(height, width, channels=3)
      gpu_tensor = buf.upload(frame)           # frame → GPU (zero-copy on UMA)
      cpu_array = buf.download(gpu_tensor)     # GPU → frame (zero-copy on UMA)
    """

    def __init__(self, height: int, width: int, channels: int = 3, device: str = "cuda"):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for PinnedFrameBuffer")
        self._h = height
        self._w = width
        self._c = channels
        self._device = device
        self._pinned_tensor = torch.empty(
            (height, width, channels), dtype=torch.uint8, pin_memory=True
        )
        self._pinned_np = self._pinned_tensor.numpy()
        self._gpu_tensor = torch.empty(
            (height, width, channels), dtype=torch.uint8, device=device
        )
        self._stream = torch.cuda.Stream(device=device)
        log.debug(
            "PinnedFrameBuffer: %dx%dx%d, pinned=%s, device=%s",
            width, height, channels,
            self._pinned_tensor.is_pinned(), device,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self._h, self._w, self._c)
    def upload(self, frame: np.ndarray) -> torch.Tensor:
        """
        Upload BGR frame (H, W, 3) uint8 to GPU.

        On UMA: effectively zero-copy — copies into pinned buffer, then
        GPU accesses the same physical page.
        On dGPU: async DMA via non-default stream.
        Returns: (H, W, 3) uint8 CUDA tensor.
        """
        np.copyto(self._pinned_np, frame)
        with torch.cuda.stream(self._stream):
            self._gpu_tensor.copy_(self._pinned_tensor, non_blocking=True)
        self._stream.synchronize()
        return self._gpu_tensor
    def upload_async(self, frame: np.ndarray) -> torch.Tensor:
        """
        Same as upload() but does NOT synchronize the stream.
        Caller must call sync() before using the tensor.
        Enables overlap: upload frame N+1 while processing frame N.
        """
        np.copyto(self._pinned_np, frame)
        with torch.cuda.stream(self._stream):
            self._gpu_tensor.copy_(self._pinned_tensor, non_blocking=True)
        return self._gpu_tensor

    def sync(self):
        """Wait for async upload to complete."""
        self._stream.synchronize()
    def download(self, gpu_tensor: torch.Tensor) -> np.ndarray:
        """
        Download GPU tensor to CPU numpy array via pinned buffer.

        On UMA: near-zero-cost virtual remap.
        On dGPU: async DMA to pinned host memory.
        """
        with torch.cuda.stream(self._stream):
            self._pinned_tensor.copy_(gpu_tensor, non_blocking=True)
        self._stream.synchronize()
        return self._pinned_np.copy()

    def release(self):
        """Free GPU tensor and stream. Pinned memory freed by Python GC."""
        del self._gpu_tensor
        del self._stream
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
# ---------------------------------------------------------------------------
# Pre-allocated output buffer
# ---------------------------------------------------------------------------
class OutputFrameBuffer:
    """
    Pre-allocated numpy buffer for annotated output frames.
    Avoids per-frame allocation in the annotator → writer path.

    The annotator writes into this buffer, the writer reads from it.
    On UMA, if backed by pinned memory, the writer thread can DMA
    directly from this buffer.
    """

    def __init__(self, height: int, width: int, channels: int = 3, pinned: bool = True):
        if pinned and TORCH_AVAILABLE and torch.cuda.is_available():
            t = torch.empty((height, width, channels), dtype=torch.uint8, pin_memory=True)
            self._buf = t.numpy()
            self._pinned = True
        else:
            self._buf = np.empty((height, width, channels), dtype=np.uint8)
            self._pinned = False
        log.debug("OutputFrameBuffer: %dx%dx%d, pinned=%s", width, height, channels, self._pinned)

    @property
    def buffer(self) -> np.ndarray:
        return self._buf

    def store(self, frame: np.ndarray) -> np.ndarray:
        """Copy annotated frame into pre-allocated buffer. Returns the buffer."""
        np.copyto(self._buf, frame)
        return self._buf
# ---------------------------------------------------------------------------
# Memory manager — ties it all together
# ---------------------------------------------------------------------------
class MemoryManager:
    """
    Central memory manager for the WEDA-CV engine.

    Lifecycle:
      1. Engine calls MemoryManager(cfg) at startup
      2. After VideoCapture is opened, call initialize(height, width)
      3. Per frame: upload_frame(frame) → gpu_tensor
      4. Per frame: store_output(annotated) → buffer for writer
      5. At shutdown: release()

    Config keys (under performance.memory):
      enabled: true          # Enable UMA optimizations
      pinned_buffers: true   # Use CUDA pinned memory
      prealloc_output: true  # Pre-allocate output frame buffer
    """
    def __init__(self, cfg: dict):
        self._mem_cfg = cfg.get("performance", {}).get("memory", {})
        self._enabled = self._mem_cfg.get("enabled", True)
        self._use_pinned = self._mem_cfg.get("pinned_buffers", True)
        self._prealloc_output = self._mem_cfg.get("prealloc_output", True)
        self._platform = detect_platform()
        self._frame_buf = None
        self._output_buf = None
        self._initialized = False
        if not self._platform["is_cuda"]:
            self._enabled = False
    @property
    def platform(self) -> dict:
        return self._platform
    @property
    def enabled(self) -> bool:
        return self._enabled

    def initialize(self, height: int, width: int, device: str = "cuda"):
        """Allocate buffers after frame dimensions are known."""
        if not self._enabled:
            log.info("Memory manager: disabled (no CUDA or config off)")
            return
        log.info(
            "Memory manager: %s, UMA=%s, %dMB VRAM, pinned=%s",
            self._platform["device_name"],
            self._platform["is_uma"],
            self._platform["total_memory_mb"],
            self._use_pinned,
        )
        if self._use_pinned:
            self._frame_buf = PinnedFrameBuffer(height, width, 3, device)
            log.info("  ┣ Frame input buffer: %dx%d pinned → %s", width, height, device)
        if self._prealloc_output:
            self._output_buf = OutputFrameBuffer(height, width, 3, pinned=self._use_pinned)
            log.info("  ┣ Output frame buffer: %dx%d (pinned=%s)", width, height, self._use_pinned)
        self._initialized = True
    def upload_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Prepare frame for inference.
        When pinned buffers are enabled, copies frame into pinned memory
        for faster GPU access. Returns the frame (possibly pinned-backed).

        Note: Ultralytics model.track() handles the actual CPU→GPU transfer
        internally. This ensures the source buffer is in optimal memory
        for that transfer (pinned on UMA = zero-copy source).
        """
        if self._frame_buf and self._enabled:
            np.copyto(self._frame_buf._pinned_np, frame)
            return self._frame_buf._pinned_np
        return frame

    def store_output(self, annotated: np.ndarray) -> np.ndarray:
        """
        Store annotated frame in pre-allocated buffer.
        Returns the buffer reference (avoids allocation in writer path).
        """
        if self._output_buf and self._enabled:
            return self._output_buf.store(annotated)
        return annotated

    def release(self):
        """Release all GPU resources."""
        if self._frame_buf:
            self._frame_buf.release()
            self._frame_buf = None
        self._output_buf = None
        self._initialized = False
        log.debug("Memory manager: released")

    def log_status(self):
        """Log current memory state."""
        if not self._enabled or not TORCH_AVAILABLE:
            return
        allocated = torch.cuda.memory_allocated() >> 20
        reserved = torch.cuda.memory_reserved() >> 20
        log.info(
            "  ┣ GPU memory: %dMB allocated, %dMB reserved, %dMB total",
            allocated, reserved, self._platform["total_memory_mb"],
        )