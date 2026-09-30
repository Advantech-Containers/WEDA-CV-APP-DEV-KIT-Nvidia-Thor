# WEDA-CV Instance Segmentation + Tracking — Release Notes

**Release:** v3.0.0  
**Date:** 2026-03-24  
**Platform:** NVIDIA Jetson Thor · JetPack 7.1  
**Base Image:** `harbor.edgesync.cloud/weda-ai/weda-cv:thor-1.0.0`  
**Classification:** QA Release — Approved for Partner Distribution

---

## Release Summary

This release introduces a config-driven inference engine architecture, a UMA zero-copy memory manager, and pipeline concurrency optimizations (threaded camera reader, async video writer). The engine source files are now standardized and do not require modification for application customization. All tunable parameters have been externalized to YAML configuration files.

---

## What's New

### Config-Driven Engine (Breaking Change)

The inference pipeline has been restructured into a standard engine that reads all behavior from a YAML configuration file. Previous releases required code modifications to change model paths, class filters, visualization settings, or output parameters. In v3.0.0, the engine source files (`src/instance_segmentation.py`, `src/visualizer.py`, `src/memory.py`) are fixed and should not be edited. New applications are created by copying and modifying a configuration file.

This is a breaking change for users who modified v2.x source files directly. Migration requires moving all customizations into a `config.yaml` file and reverting to the standard engine files.

### UMA Memory Manager (`memory.py`)

A new module provides CUDA pinned (page-locked) frame buffers and pre-allocated output buffers optimized for Jetson's unified memory architecture. On UMA platforms, pinned host memory is directly accessible from the GPU without DMA transfer, eliminating per-frame `cudaMalloc`/`cudaFree` overhead. The module auto-detects the platform (Jetson vs. discrete GPU) and adapts its behavior accordingly. It is backend-agnostic and works with PyTorch, ONNX, and TensorRT model formats.

### Async Video Writer

Video encoding now runs in a background thread. The main inference loop enqueues frames via a non-blocking call, removing encoder latency (measured at 13.6 ms per frame for MJPG at 1280×720) from the critical path. Configurable queue depth prevents memory exhaustion. When the queue fills, frames are silently dropped rather than blocking inference.

### Threaded Camera Reader

Live video sources (USB cameras, RTSP streams) are captured in a background thread. The reader tracks frame uniqueness and returns `None` when no new frame has arrived since the last read, preventing duplicate frame processing. Combined with V4L2 direct backend selection, this resolves the GStreamer throttling issue that limited USB camera throughput to 20 FPS in previous releases.

### V4L2 Direct Backend

Camera sources now default to the V4L2 backend (`cv2.CAP_V4L2`) with MJPG capture format and a ring buffer depth of 1. This bypasses GStreamer's pipeline overhead. The backend, capture format, FPS, and buffer depth are configurable via the `camera:` section in the YAML config.

### Visualization Enhancements

The annotation pipeline now uses Ultralytics' C++-optimized `results.plot()` for mask and bounding box rendering, with custom "Track ID: X" labels drawn on top. This hybrid approach matches the Ultralytics reference documentation screenshots. The window name, label font scale, mask alpha, and all other visualization parameters are configurable.

---

## Performance Improvements

All measurements on Jetson Thor, JetPack 7.1, `yolo26n-seg.pt`, FP16, `imgsz=640`, ByteTrack.

| Metric | v2.1.0 | v3.0.0 | Improvement |
|--------|--------|--------|-------------|
| Video file FPS (1280×720) | 21.8 | 34.3 | +57% |
| USB camera FPS (640×480) | 20.4 | 30.0 | +47% |
| Write latency (main thread) | 13.6 ms | 0.0 ms | Async writer |
| Annotate latency | 11.7 ms | 11.3 ms | Hybrid plot |
| Track latency | 18.1 ms | 15.4 ms | Warmup + retina_masks optimization |

---

## Test Matrix

### Functional Tests

| Test Case | Source | Model Format | Status |
|-----------|--------|--------------|--------|
| Video file inference (MP4) | `assets/test.mp4` | PyTorch (`.pt`) | PASS |
| USB webcam live inference | Device index `0` | PyTorch (`.pt`) | PASS |
| RTSP stream (if available) | `rtsp://...` | PyTorch (`.pt`) | Not tested (environment-dependent) |
| Class filtering (person only) | `classes: [0]` | PyTorch (`.pt`) | PASS |
| Headless mode (`show: False`) | Video file | PyTorch (`.pt`) | PASS |
| Output disabled (`output.enabled: False`) | Video file | PyTorch (`.pt`) | PASS |
| ByteTrack tracker | Video file | PyTorch (`.pt`) | PASS |
| BoT-SORT tracker | Video file | PyTorch (`.pt`) | PASS |
| Custom config (`--config`) | Video file | PyTorch (`.pt`) | PASS |
| Signal handling (SIGINT/SIGTERM) | Video file | PyTorch (`.pt`) | PASS |
| Graceful end-of-video | Video file | PyTorch (`.pt`) | PASS |

### Platform Tests

| Test Case | Status |
|-----------|--------|
| Container build from Dockerfile | PASS |
| Container launch via `build.sh` | PASS |
| Container launch via `build.sh --run` | PASS |
| Container shell via `build.sh --shell` | PASS |
| Container stop via `build.sh --stop` | PASS |
| X11 display auto-detection | PASS |
| Xauthority auto-detection | PASS |
| EULA gate (interactive accept) | PASS |
| EULA gate (pre-accepted via `.env`) | PASS |
| Entrypoint hydration (fresh container) | PASS |
| Entrypoint idempotency (re-run) | PASS |
| Environment protection patch | PASS |
| Volume mount persistence (`./output`) | PASS |

### Memory Manager Tests

| Test Case | Status |
|-----------|--------|
| Platform detection (Jetson Thor UMA) | PASS |
| Pinned buffer allocation | PASS |
| Pre-allocated output buffer | PASS |
| Graceful fallback (memory disabled) | PASS |
| GPU memory status logging | PASS |
| Resource cleanup on shutdown | PASS |

### Edge Cases and Known Constraints

| Scenario | Behavior | Notes |
|----------|----------|-------|
| Model file not found locally | Ultralytics auto-download attempted | Requires outbound internet access |
| Camera device not present at `/dev/video0` | `Cannot open video source: 0` and exit | Connect camera before container start |
| X11 socket missing inside container | `cv2.imshow()` fails silently | Set `visualization.show: False` for headless |
| RTSP stream disconnects mid-inference | `Empty frame from source` and graceful exit | Reconnection logic is not implemented; restart required |
| Async writer queue overflow | Frames silently dropped | Increase `output.queue_size` or disable async writer |
| Multiple X11 sockets detected | First socket used by default | Override with `export DISPLAY=:N` before `build.sh` |
| Empty Xauthority file | `build.sh` exits with error | Verify display manager is active on host |

---

## Known Limitations

1. RTSP stream reconnection is not implemented. If the stream disconnects, the pipeline exits gracefully but does not attempt to reconnect. This is a planned feature for a future release.

2. TensorRT engine files (`.engine`) have not been tested in this release cycle. Ultralytics supports TensorRT export and inference; the engine and memory manager are designed to be backend-agnostic, but quantitative validation on TensorRT has not been performed.

3. ONNX model inference via the CUDA Execution Provider has not been validated in this release. The memory manager's pinned buffer optimization applies to the frame I/O path and is independent of the inference backend, but end-to-end benchmarks with ONNX models are pending.

4. The threaded camera reader does not implement timeout logic. If the camera hardware locks or the V4L2 driver hangs, the background thread will block indefinitely on `cap.read()`.

5. The `output.codec` setting of `mp4v` produces a `.avi` container by default. For proper `.mp4` output, the file extension and codec must both be changed. This is a configuration responsibility and is documented in the config comments.

---

## Compatibility

| Component | Version |
|-----------|---------|
| NVIDIA JetPack | 7.1 |
| L4T | R38.2.2 |
| CUDA | 13.0 |
| TensorRT | 10.x |
| PyTorch | 2.10.0 |
| OpenCV | 4.13.0 |
| Ultralytics | ≥ 8.4.0 |
| Python | 3.12 |

---

## Upgrade Instructions

From v2.x:

1. Back up any customized source files in `src/`.
2. Replace all files in `src/` with the v3.0.0 standard engine files.
3. Transfer all customizations from code into `config/config.yaml`.
4. Verify: `python3 src/instance_segmentation.py --config config/config.yaml`.

From v1.x:

A full replacement of `src/` and `config/` is required. The v1.x `solutions.InstanceSegmentation` API is no longer used. Refer to the User Guide for the v3.0.0 architecture.

---

## Approval

| Role | Name | Date |
|------|------|------|
| QA Engineer | | |
| Technical Lead | | |
| Release Manager | | |

---

© 2026 Advantech Corporation. All rights reserved.
