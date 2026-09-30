# WEDA-CV Instance Segmentation + Tracking Demo

**Platform:** NVIDIA Jetson Thor · JetPack 7.1 · WEDA-CV Container Platform  
**Engine Version:** 3.0.0  
**Base Image:** `harbor.edgesync.cloud/weda-ai/weda-cv:1.0.0-thor`  
**Maintainer:** Advantech Corporation — WISE-Edge Platform Division

---

## Overview

This project provides a production-grade, config-driven inference engine for real-time instance segmentation and multi-object tracking on NVIDIA Jetson edge devices. It is built on the Advantech WEDA-CV container platform and designed as a reference implementation that partners and system integrators can deploy, customize, and extend without modifying engine code.

The engine runs YOLO segmentation models (PyTorch, ONNX, TensorRT) with BoT-SORT or ByteTrack for persistent object identity across frames. A dedicated UMA memory manager exploits Jetson's unified memory architecture for zero-copy frame I/O, delivering measurably higher throughput than standard Ultralytics pipelines.

---

## Key Features

**Config-Driven Architecture.** All application behavior — model selection, video source, class filtering, visualization, output format, and performance tuning — is controlled exclusively through a YAML configuration file. The engine source files are standard and never require modification. New applications are created by copying and editing a config file.

**UMA Zero-Copy Memory Optimization.** CUDA pinned (page-locked) frame buffers eliminate per-frame `cudaMalloc`/`cudaFree` overhead. On Jetson UMA, pinned host memory is directly GPU-accessible without DMA transfer. This optimization is backend-agnostic and works identically across PyTorch (`.pt`), ONNX (`.onnx`), and TensorRT (`.engine`) model formats.

**Threaded Camera Reader.** Live video sources (USB cameras, RTSP streams) are captured in a dedicated background thread, decoupling V4L2/GStreamer frame acquisition from the inference pipeline. Prevents the capture-to-inference stall that typically limits USB camera pipelines to 20 FPS on Jetson.

**Async Video Writer.** Video encoding runs in a background thread with a configurable frame queue. Removes 10–15 ms per frame from the critical path on typical 1080p workloads.

**V4L2 Direct Backend.** Camera sources default to V4L2 with MJPG capture format, bypassing GStreamer's pipeline buffering and format negotiation overhead.

**Hybrid Annotation.** Mask and bounding box rendering uses Ultralytics' C++-optimized `results.plot()` for maximum throughput. Custom "Track ID: X" labels and an "Objects Tracked: N" count badge are drawn on top via lightweight OpenCV calls. Demo-quality visual output at less than 3 ms per frame annotation cost.

**Self-Hydrating Container.** The entrypoint detects and installs missing system dependencies, gates Ultralytics installation behind a BYOL/EULA acceptance workflow, and applies environment protection patches to prevent Ultralytics from overwriting system packages.

**Integrated Hardware Verification.** The bundled `wise-bench` tool validates the entire GPU passthrough chain, framework stack, and inference throughput before any application code runs.

---

## System Requirements

### Host System

| Component | Version | Description |
|:----------|:--------|:------------|
| OS | JetPack 7.1 | Ubuntu 24.04 (Noble Numbat) with NVIDIA L4T R38.2.2 |
| Docker | v24.0+ | Docker Engine installed and running |
| Docker Compose | v2.28+ | Installed as the v2 plugin |
| NVIDIA CTK | v1.14+ | NVIDIA Container Toolkit configured for Docker |

CUDA, cuDNN, and TensorRT do not need to be installed on the host. The container encapsulates the complete AI framework stack.

For GUI output (live visualization via `cv2.imshow`), `x11-xserver-utils` and `xauth` should be installed on the host and an X11 display session must be active.

### Container Environment

The base image (`weda-cv:1.0.0-thor`) provides the following pre-configured and tested components:

| Component | Version | Description |
|:----------|:--------|:------------|
| PyTorch | 2.10.0 | Deep learning framework with CUDA 13.0 support |
| OpenCV | 4.13.0 | Hardware-accelerated computer vision (GStreamer + CUDA) |
| TensorRT | 10.x | NVIDIA inference optimizer and runtime |
| ONNX Runtime | 1.24.0 | Cross-platform inference with TensorRT and CUDA execution providers |
| Ultralytics | == 8.4.24 | YOLO model framework (installed via BYOL entrypoint after EULA acceptance) |
| Python | 3.12 | Runtime environment |

### Hardware Acceleration Support

| Capability | Support | Notes |
|:-----------|:--------|:------|
| GPU Compute | Full | Extended access to compute, video, utility, graphics, and display capabilities |
| FP16 Tensor Cores | Full | Activated via `model.half: True` in config |
| X11 Display | Full | Bound to host X11 socket and Xauthority via `build.sh` auto-detection |
| Hardware Peripherals | Full | `/dev` mount provides access to USB cameras, CSI interfaces, and serial devices |

---

## Performance

Measured on NVIDIA Jetson Thor, JetPack 7.1, `yolo26n-seg.pt`, FP16, `imgsz=640`, ByteTrack.

| Source | Resolution | FPS |
|:-------|:-----------|:----|
| Video file (MP4) | 1280×720 | 34.3 |
| USB camera (V4L2) | 640×480 | 30.0 (camera-bound) |

---

## Quick Start

### 1. Configure

Review `.env` and set `ACCEPT_ULTRALYTICS_EULA=true` to enable non-interactive Ultralytics installation.

### 2. Build and Launch

```bash
chmod +x *.sh wise-bench
./build.sh
```

The script executes a full deployment pipeline: pre-flight validation (Docker, Compose, JQ), X11 display auto-detection, EULA acceptance gate, image synchronization, container launch, environment hydration, and interactive shell entry.

First launch hydrates the environment — system packages plus the BYOL Ultralytics install — which measures roughly 70 seconds on a Jetson Thor / JP7 device and longer on slow links; `build.sh` waits up to `HYDRATION_TIMEOUT` (300 s, set in `.env`) and only opens the shell once it prints `[  OK]  Environment hydrated.` Subsequent launches reuse the running container (~1 s), and the licensed install is cached in a named volume so even recreating the container does not re-download it.

### 3. Validate the Environment

Before running application code, verify that the GPU passthrough chain and framework stack are healthy:

```bash
./wise-bench
```

See the [Hardware Verification](#hardware-verification-wise-bench) section below for details on interpreting the output.

### 4. Run the Demo

```bash
python3 src/instance_segmentation.py
```
**Note: Make sure the camera node index points to the correct RGB node and is working before running the application.**
The engine loads the model, processes the configured source, and renders annotated output with instance masks, bounding boxes, and Track ID labels. Press `q` to exit. When `output.enabled: true` is set in the config, the annotated video is saved to `output/segmented_output.avi`.

### 5. Run with a Custom Config

```bash
python3 src/instance_segmentation.py --config config/person_tracking.yaml
```

### 6. Stop (from the host)

```bash
./build.sh --stop
```

### build.sh Modes

| Command | Description |
|:--------|:------------|
| `./build.sh` | Full pipeline: pull, launch, hydrate, enter shell |
| `./build.sh --run` | Launch only (skip pull, use existing image) |
| `./build.sh --shell` | Attach to an already-running container |
| `./build.sh --stop` | Stop and remove the container |
| `./build.sh --clean` | Stop, remove, and purge the cached BYOL install |
| `./build.sh --help` | Display usage information |

---

## Hardware Verification (wise-bench)

The `wise-bench` tool is an integrated hardware verification and benchmark suite bundled with every WEDA-CV container. It validates the complete stack — from host-level GPU passthrough through framework initialization to inference throughput — before any application code runs. Running `wise-bench` first ensures the container environment is healthy and establishes a performance baseline for the device.

### Running wise-bench

```bash
# Inside the container
./wise-bench
```

Validation results and performance metrics are written to `test_logs/` (persisted to the host via the volume mount at `./test_logs`).

### Example Output

A successful run on Jetson Thor produces a verification matrix similar to the following:

```
====================================================================================================
WEDA-CV Verification Summary
====================================================================================================
Test ID     | Component                    | Status | Details
----------------------------------------------------------------------------------------------------
SYS-001     | System Python                | PASS   | v3.12.3
SYS-002     | Platform                     | INFO   | advantech | Linux 6.8.12-tegra | aarch64
SYS-003     | CPU & RAM                    | INFO   | 14 Cores | 9.8GB / 122.8GB RAM
SYS-004     | L4T (JetPack)                | PASS   | R38 (release) | REVISION: 2.2
SYS-005     | System CUDA/cuDNN            | PASS   | CUDA 13.0 | cuDNN 9.2.0 | Path: /usr/local/cuda
SYS-006     | NVIDIA-SMI                   | PASS   | NVIDIA Thor | Driver: 580.00
SYS-007     | EULA Status                  | PASS   | ACCEPT_ULTRALYTICS_EULA=true
----------------------------------------------------------------------------------------------------
CV-001      | PyTorch Core                 | PASS   | v2.10.0a0+git449b176 | NVIDIA Thor | cc11.0
CV-002      | PyTorch Ops (FP32)           | PASS   | 1024×1024 FP32 GEMM on cuda:0 OK
CV-003      | PyTorch Ops (BF16)           | PASS   | BFloat16 GEMM OK (SM native)
CV-004      | TensorRT (Native)            | PASS   | v10.14.1.48 (CUDA 13.0)
CV-005      | TensorRT (Caps)              | INFO   | FP16: True | INT8: True
CV-006      | OpenCV (GStreamer)           | PASS   | v4.13.0 | Native video pipelines OK
CV-007      | OpenCV (CUDA)                | PASS   | v4.13.0 | CUDA devices: 1
CV-008      | ONNXRuntime (GPU)            | PASS   | v1.24.0 | TensorrtEP + CUDAExecutionProvider
CV-009      | ONNX Core                    | PASS   | v1.18.0
----------------------------------------------------------------------------------------------------
COM-001     | Ultralytics                  | PASS   | v8.4.24
INF-002     | YOLO GPU Inference           | PASS   | yolo26n detect | 640×640 synthetic | half=True
INF-003     | YOLO Latency (FP16)          | PASS   | avg=7.6ms | p99=10.1ms | ~131 FPS | cuda:0
INF-007     | YOLO CLI Predict             | PASS   | yolo26 | camera | ~138.9 FPS | saved
====================================================================================================
```

### Interpreting Results

The verification matrix is organized into three diagnostic areas. Each area validates a distinct layer of the deployment stack.

**Host-Level Passthrough (SYS-001 through SYS-007).** These tests confirm that the Docker container has correctly bound to the host's L4T system profile. SYS-004 validates the JetPack release (R38, revision 2.2). SYS-005 confirms the CUDA 13.0 and cuDNN 9.2.0 toolkit paths are visible inside the container. SYS-006 verifies that `nvidia-smi` can communicate with the GPU driver (580.00) and detect the NVIDIA Thor device. A failure at this level indicates that the NVIDIA Container Toolkit is not correctly configured or that the `runtime: nvidia` directive in `docker-compose.yml` is not being applied.

**Framework and Tensor Operations (CV-001 through CV-009).** These tests ensure that all ML framework primitives are executing on the GPU (`cuda:0`) without silent CPU fallback. CV-001 verifies PyTorch's CUDA compute capability mapping (`cc11.0` for Blackwell). CV-002 and CV-003 run FP32 and BF16 matrix multiplications on the GPU to confirm tensor core functionality. CV-004 validates that TensorRT 10.x has connected to the CUDA execution endpoint. CV-006 confirms OpenCV's GStreamer bindings for hardware-accelerated video decode. CV-008 verifies that ONNX Runtime has both the TensorRT and CUDA execution providers available. A failure in this layer means inference will either silently fall back to CPU (producing unexpectedly low FPS) or fail outright.

**Inference Throughput (COM-001, INF-002 through INF-007).** These tests push synthetic 640×640 tensor frames through a YOLO model in FP16 half-precision, measuring per-frame latency on the GPU memory bus. INF-003 reports the average latency (7.6 ms), the P99 tail latency (10.1 ms), and the derived FPS ceiling (~131 FPS) for raw inference without any I/O or annotation overhead. INF-007 validates end-to-end prediction through the YOLO CLI, including camera stream parsing via hardware-accelerated GStreamer (~138.9 FPS). These numbers represent the device's theoretical throughput ceiling. Application-level FPS will be lower due to annotation, video encoding, and display overhead — the engine's timing instrumentation (`debug.timing: True` in config) shows exactly where the per-frame budget is spent.

### When wise-bench Fails

If any test shows `FAIL` status, resolve the underlying hardware or framework issue before running application code. Common failure patterns and their resolutions are documented in the [Troubleshooting Guide](TROUBLESHOOTING.md). The most frequent causes are: NVIDIA Container Toolkit not installed on the host (SYS-006 fails), EULA not accepted so Ultralytics is missing (COM-001 fails), or a container started without `runtime: nvidia` (all CV tests fail with CUDA not available).

---

## Project Structure

```
instance-segmentation/
├── config/
│   ├── config.yaml                  # Default application configuration
│   ├── person_tracking.yaml         # Example: person-only on RTSP
│   └── vehicle_segmentation.yaml    # Example: vehicle detection on video
├── src/
│   ├── instance_segmentation.py     # Inference engine (standard — do not modify)
│   ├── visualizer.py                # Track ID annotator (standard — do not modify)
│   └── memory.py                    # UMA memory manager (standard — do not modify)
├── data/                            # Model files, sample videos
├── output/                          # Annotated video output (auto-created)
├── test_logs/                       # wise-bench logs (persisted to host)
├── Dockerfile                       # Container build definition
├── docker-compose.yml               # Service orchestration
├── build.sh                         # Build, launch, and manage script
├── entrypoint.sh                    # Self-hydrating container entrypoint
├── wise-bench                       # Hardware verification and benchmark tool
└── .env                             # Static environment variables
```

### Container Paths

```
/wise-edge/instance-segmentation/          ← WORKDIR
├── src/                                   ← mounted from host (ro)
├── config/                                ← mounted from host (ro)
├── data/                                  ← mounted from host (ro)
├── output/                                ← mounted from host (rw)
└── test_logs/                             ← mapped to /wise-edge/instance-segmentation/test_logs (rw)

/opt/weda-cv/byol/                         ← BYOL packages, named volume (rw), on PYTHONPATH
```

Source, config, and data directories are mounted read-only from the host. Changes made on the host are immediately reflected inside the container without rebuilding the image. The output and test_logs directories are mounted read-write for persisting annotated video and benchmark results to the host filesystem.

---

## Environment Variables (.env)

| Variable | Default | Description |
|:---------|:--------|:------------|
| `WEDA_CV_IMAGE` | `harbor.edgesync.cloud/weda-ai/weda-cv:1.0.0-thor` | Base container image |
| `CONTAINER_NAME` | `weda-cv-instance-segmentation` | Docker container name |
| `SERVICE_NAME` | `instance-segmentation` | Docker Compose service name |
| `ACCEPT_ULTRALYTICS_EULA` | `false` | Set to `true` to skip the interactive EULA prompt |
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU devices exposed to the container |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,video,utility,graphics,display` | NVIDIA driver capabilities |
| `QT_X11_NO_MITSHM` | `1` | Prevents Qt shared memory errors in containerized X11 |
| `LOG_MAX_SIZE` | `20m` | Docker JSON log file size limit |
| `LOG_MAX_FILE` | `3` | Docker JSON log file rotation count |

The `DISPLAY`, `XAUTHORITY`, and `X11_SOCKET` variables are auto-detected at runtime by `build.sh` and should not be hardcoded in `.env`. They are device-specific and session-specific.

---

## Input Sources

| Type | Config Example |
|:-----|:---------------|
| Video file | `source: data/test.mp4` |
| USB webcam | `source: 0` |
| RTSP stream | `source: rtsp://admin:pass@192.168.1.100:554/stream` |
| HTTP stream | `source: http://192.168.1.100:8080/video` |

---

## Output

| Item | Location |
|:-----|:---------|
| Annotated video | `output/segmented_output.avi` (configurable) |
| Console logs | stdout (with timestamps, FPS, component timing) |
| Benchmark results | `test_logs/` (from `wise-bench`) |

---

## Documentation

| Document | Description |
|:---------|:------------|
| [User Guide](USER_GUIDE.md) | Building custom applications, container deployment, performance tuning |
| [Troubleshooting Guide](TROUBLESHOOTING.md) | 32 production failure scenarios with diagnostics and resolutions |
| [Release Notes](RELEASE_NOTES.md) | QA release note — changelog, test matrix, known limitations |
| [config/config.yaml](config/config.yaml) | Complete configuration reference (inline comments on every key) |

---

## License

The WEDA-CV base container image is subject to Advantech commercial licensing terms. Ultralytics components are installed via BYOL (Bring Your Own License) and are subject to the Ultralytics EULA. NVIDIA JetPack and CUDA components are subject to the NVIDIA Software License Agreement.

---

## Support

For known issues and resolutions, consult the [Troubleshooting Guide](TROUBLESHOOTING.md). For build and deployment guidance, consult the [User Guide](USER_GUIDE.md). For issues outside the scope of this documentation, contact the WISE-Edge Platform Division.

---

© 2026 Advantech Corporation. All rights reserved.
