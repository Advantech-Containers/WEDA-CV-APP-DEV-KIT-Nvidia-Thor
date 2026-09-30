# WEDA-CV Object Counting Demo — Release Notes

**Release:** v1.0.1  
**Date:** 2026-08-05  
**Platform:** NVIDIA Jetson Thor · JetPack 7.1  
**Base Image:** `harbor.edgesync.cloud/weda-ai/weda-cv:1.0.0-thor`  
**Classification:** QA Release — Approved for Partner Distribution

---

## Release Summary

This is the initial release of the WEDA-CV Object Counting Demo. It provides a unified, config-driven object counting application supporting three counting modes (line crossing, rectangular region, arbitrary polygon) on NVIDIA Jetson Thor. The application is deployed as a self-hydrating Docker container on the WEDA-CV platform, with full GPU passthrough, X11 display support, and BYOL license management for Ultralytics components.

---

## Features

### Unified Counting Modes

The application supports three counting tasks selectable via the `task` key in `config/config.yaml`. Line counting (2-point ROI) tracks objects crossing a defined boundary. Region counting (4-point ROI) tracks objects entering or exiting a rectangular zone. Polygon counting (3+ point ROI) tracks objects within an arbitrary-shaped area. The same `object_counter.py` source file handles all three modes without modification.

### Config-Driven Application

All runtime behavior is controlled through `config/config.yaml`. This includes the counting task, model path, video source, ROI geometry, detection thresholds, class filter, tracker selection, FP16 inference toggle, input resolution, frame skipping, visualization options, and output path. A complete configuration schema is provided in `config/config.schema.md`.

### Self-Hydrating Container

The `entrypoint.sh` performs six steps at container startup: X11 display validation, system dependency detection and installation (X11 libs, GStreamer plugins, V4L2 utilities), Ultralytics EULA gate with interactive or pre-accepted modes, environment protection patch to prevent Ultralytics from overwriting system packages, solutions logging patch to suppress per-frame configuration dumps, and `.ready` sentinel file creation for `build.sh` synchronization.

### Automated Build and Launch

The `build.sh` script provides a single-command deployment pipeline with four modes: full build (`./build.sh`), run-only (`--run`), shell attach (`--shell`), and stop (`--stop`). The script auto-detects X11 display and Xauthority paths, validates all required environment variables from `.env`, and synchronizes progress with the container's hydration process via a progress bar.

### Integrated Benchmarking

The `wise-bench` tool validates GPU passthrough, framework health (PyTorch, TensorRT, OpenCV, ONNX Runtime), and inference throughput before application deployment.

---

## Test Matrix

### Functional Tests

| Test Case | Configuration | Status |
|:----------|:--------------|:-------|
| Line counting on video file | `task: line_count`, 2 ROI points | PASS |
| Region counting on video file | `task: region_count`, 4 ROI points | PASS |
| Polygon counting on video file | `task: polygon_count`, 5 ROI points | PASS |
| Class filtering (person only) | `classes: [0]` | PASS |
| Class filtering (person + car) | `classes: [0, 2]` | PASS |
| All classes (null) | `classes: null` | PASS |
| FP16 inference | `model.half: true` | PASS |
| FP32 inference | `model.half: false` | PASS |
| BoT-SORT tracker | `tracker: botsort.yaml` | PASS |
| ByteTrack tracker | `tracker: bytetrack.yaml` | PASS |
| Headless mode (`show: false`) | Video file | PASS |
| GUI mode (`show: true`) | Video file | PASS |
| Custom config (`--config`) | Alternate YAML path | PASS |
| Signal handling (SIGINT/SIGTERM) | Video file | PASS |
| Graceful end-of-video | Video file | PASS |
| Webcam source | `source: 0` | PASS |
| RTSP source | `source: rtsp://...` | Not tested (environment-dependent) |

### Platform Tests

| Test Case | Status |
|:----------|:-------|
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
| Solutions logging patch | PASS |
| Volume mount persistence (`./output`) | PASS |
| wise-bench execution | PASS |

### Configuration Validation Tests

| Test Case | Status |
|:----------|:-------|
| Invalid task name → error with valid options listed | PASS |
| Missing ROI points → error with minimum count | PASS |
| Insufficient ROI points for task → error with requirement | PASS |
| Empty config file → error | PASS |
| Missing config file → error with path | PASS |
| Model file not found (`.pt`) → auto-download attempted | PASS |
| Model file not found (non-`.pt`) → error | PASS |

### Edge Cases and Known Constraints

| Scenario | Behavior | Notes |
|:---------|:---------|:------|
| Model file not found locally (`.pt`) | Ultralytics auto-download attempted | Requires outbound internet access |
| Model file not found locally (non-`.pt`) | Application exits with error | ONNX, TensorRT, TorchScript must be provided as local files |
| Camera device not present | `Cannot open video source: 0` and exit | Connect camera before container start |
| X11 socket missing inside container | `cv2.imshow()` fails | Set `show: false` for headless operation |
| RTSP stream disconnects mid-inference | `Empty frame received from source` and graceful exit | Reconnection logic not implemented |
| Multiple X11 sockets detected | First socket used by default | Override with `export DISPLAY=:N` before `build.sh` |
| Video writer cannot open | Warning logged, output not saved | Verify output directory exists and is writable |

---

## Known Limitations

1. RTSP stream reconnection is not implemented. If the stream disconnects, the pipeline exits gracefully but does not attempt to reconnect.

2. TensorRT engine files (`.engine`) have not been tested in this release cycle. The application supports TensorRT via Ultralytics, but quantitative validation is pending.

3. ONNX model inference has not been validated end-to-end on this platform. The container includes ONNX Runtime with CUDA and TensorRT execution providers, but application-level testing with ONNX models is pending.

4. The video writer uses the synchronous `mp4v` codec. For high-FPS workloads, this may become a bottleneck. An async writer is not implemented in this release.

5. The `ObjectCounter` solution's annotation style is controlled by Ultralytics internally and cannot be fully customized via config. Label format, color palette, and count display positioning are determined by the `solutions.ObjectCounter` API.

6. Polygon counting with very high vertex counts (>20 points) has not been tested for performance impact on the annotation path.

---

## Compatibility

| Component | Version |
|:----------|:--------|
| NVIDIA JetPack | 7.1 |
| L4T | R38.2.2 |
| CUDA | 13.0 |
| TensorRT | 10.x |
| PyTorch | 2.10.0 |
| OpenCV | 4.13.0 |
| Ultralytics | ≥ 8.4.0 |
| Python | 3.12 |

---

## Approval

| Role | Name | Date |
|:-----|:-----|:-----|
| QA Engineer | | |
| Technical Lead | | |
| Release Manager | | |

---

© 2026 Advantech Corporation. All rights reserved.
