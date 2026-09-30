# WEDA-CV Object Counting Demo

**Platform:** NVIDIA Jetson Thor · JetPack 7.1 · WEDA-CV Container Platform  
**Version:** 1.0.0  
**Base Image:** `harbor.edgesync.cloud/weda-ai/weda-cv:1.0.0-thor`  
**Maintainer:** Advantech Corporation — WISE-Edge Platform Division

---

## Overview

This project provides a containerized, config-driven object counting application for real-time AI inference on NVIDIA Jetson edge devices. It is built on the Advantech WEDA-CV container platform and supports three counting modes — line crossing, rectangular region, and arbitrary polygon zone — all configured through a single YAML file.
The application uses Ultralytics YOLO models with BoT-SORT or ByteTrack for persistent object tracking, combined with line/region-crossing logic that determines counts. Objects are counted as they cross a defined line, enter a rectangular region, or appear within an arbitrary polygon zone. All counting logic, model selection, class filtering, and visualization parameters are controlled exclusively through `config/config.yaml`. No code modification is required to switch between counting tasks.

---

## Key Features

**Unified Counting Logic.** A single application supports line-crossing counting (2-point line), rectangular region counting (4-point zone), and polygon counting (3+ point arbitrary shape). The counting mode is selected by setting the `task` key in the configuration file.

**Pure-YAML Configuration.** All application behavior — model path, video source, counting task, ROI geometry, class filter, tracker selection, FP16 inference, resolution, frame skipping, and output — is driven entirely by `config/config.yaml`. No source code changes are required to build a new counting application.

**Full Hardware Acceleration.** The container exposes optimized access to NVIDIA Tensor Cores, video codec engines, and GPU compute capabilities. FP16 half-precision inference is enabled via a single config toggle (`model.half: true`).

**Self-Hydrating Container.** The `entrypoint.sh` detects and installs missing system dependencies (X11, GStreamer, V4L2 utilities), gates Ultralytics installation behind a BYOL/EULA acceptance workflow, and applies environment protection patches to prevent Ultralytics from overwriting the WEDA-CV base image's pre-tested packages.

**Integrated Hardware Verification.** The bundled `wise-bench` tool validates the GPU passthrough chain, framework stack, and inference throughput before any application code runs.

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

The base image (`weda-cv:1.0.0-thor`) provides the following pre-configured components:

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
| FP16 Tensor Cores | Full | Activated via `model.half: true` in config |
| X11 Display | Full | Bound to host X11 socket and Xauthority via `build.sh` auto-detection |
| Hardware Peripherals | Full | `/dev` mount provides access to USB cameras, CSI interfaces, and serial devices |

---

## Quick Start

### 1. Add a custom video file 

```bash
cp /path/to/your/video.mp4 data/test.mp4
```

### 2. Configure

Review `.env` and set `ACCEPT_ULTRALYTICS_EULA=true` to enable non-interactive Ultralytics installation.

### 3. Build and Launch

```bash
chmod +x *.sh wise-bench
./build.sh
```

The script executes a full deployment pipeline: pre-flight validation (Docker, Compose, JQ), X11 display auto-detection, EULA acceptance gate, image synchronization, container launch, environment hydration, and interactive shell entry.

First launch hydrates the environment — system packages plus the BYOL Ultralytics install — which measures roughly 70 seconds on a Jetson Thor / JP7 device and longer on slow links; `build.sh` waits up to `HYDRATION_TIMEOUT` (300 s, set in `.env`) and only opens the shell once it prints `[  OK]  Environment hydrated.` Subsequent launches reuse the running container (~1 s), and the licensed install is cached in a named volume so even recreating the container does not re-download it.

### 4. Validate the Environment

Before running application code, verify that the GPU passthrough chain and framework stack are healthy:

```bash
./wise-bench
```

See the [Hardware Verification](#hardware-verification-wise-bench) section below for details.

### 5. Run the Demo

```bash
python3 src/weda-object_counter.py
```
**Note: Make sure the camera node index points to the correct RGB node and is working before running the application.**
The application processes the configured source, renders an OpenCV GUI if `show: true`, and saves the annotated output to `output/counted_output.avi`.

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

## Counting Tasks

The application supports three counting modes, selected by the `task` key in `config/config.yaml`.

### Line Counting

Counts objects crossing a 2-point line. Typical use cases include doorway traffic, road lane counting, and conveyor belt monitoring.

```yaml
task: line_count
roi:
  points:
    - [20, 400]
    - [1080, 400]
```

### Region Counting

Counts objects entering or exiting a 4-point rectangular zone. Typical use cases include parking occupancy, loading dock monitoring, and retail area analysis.

```yaml
task: region_count
roi:
  points:
    - [20, 400]
    - [1080, 400]
    - [1080, 360]
    - [20, 360]
```

### Polygon Counting

Counts objects within an arbitrary N-point polygon zone. Typical use cases include irregularly shaped areas, intersection analysis, and custom geofenced regions.

```yaml
task: polygon_count
roi:
  points:
    - [100, 300]
    - [600, 280]
    - [700, 450]
    - [400, 500]
    - [50, 420]
```

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

The verification matrix is organized into three diagnostic areas.

**Host-Level Passthrough (SYS-001 through SYS-007).** These tests confirm that the Docker container has correctly bound to the host's L4T system profile. SYS-004 validates the JetPack release (R38, revision 2.2). SYS-005 confirms the CUDA 13.0 and cuDNN 9.2.0 toolkit paths are visible. SYS-006 verifies that `nvidia-smi` can communicate with the GPU driver (580.00) and detect the NVIDIA Thor device. A failure at this level indicates that the NVIDIA Container Toolkit is not correctly configured or that the `runtime: nvidia` directive in `docker-compose.yml` is not being applied.

**Framework and Tensor Operations (CV-001 through CV-009).** These tests ensure that all ML framework primitives are executing on the GPU (`cuda:0`) without silent CPU fallback. CV-002 and CV-003 run FP32 and BF16 matrix multiplications to confirm tensor core functionality. CV-004 validates TensorRT 10.x connectivity. CV-006 confirms OpenCV's GStreamer bindings for hardware-accelerated video decode. A failure in this layer means inference will either silently fall back to CPU or fail outright.

**Inference Throughput (COM-001, INF-002 through INF-007).** These tests push synthetic 640×640 tensor frames through a YOLO model in FP16, measuring per-frame latency on the GPU memory bus. INF-003 reports average latency (7.6 ms), P99 tail latency (10.1 ms), and the derived FPS ceiling (~131 FPS) for raw inference without I/O or annotation overhead. These numbers represent the device's theoretical throughput ceiling. Application-level FPS will be lower due to the ObjectCounter's annotation, video encoding, and display overhead.

### When wise-bench Fails

If any test shows `FAIL` status, resolve the underlying issue before running application code. The most frequent causes are: NVIDIA Container Toolkit not installed on the host (SYS-006 fails), EULA not accepted so Ultralytics is missing (COM-001 fails), or a container started without `runtime: nvidia` (all CV tests fail). See the [Troubleshooting Guide](TROUBLESHOOTING.md) for detailed resolutions.

---

## Project Structure

```
object-counting/
├── config/
│   ├── config.yaml                  # Runtime configuration (task, model, source, ROI)
│   └── config.schema.md             # Full configuration reference
├── src/
│   ├── memory.py                    # UMA memory manager (standard — do not modify)
│   └── weda-object_counter.py       # Unified object counting application
├── data/                            # Sample videos, model files
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
/wise-edge/object-counting/               ← WORKDIR
├── src/                                  ← mounted from host (ro)
├── config/                               ← mounted from host (ro)
├── data/                                 ← mounted from host (ro)
├── output/                               ← mounted from host (rw)
└── test_logs/                            ← mapped to /wise-edge/object-counting/test_logs (rw)

/opt/weda-cv/byol/                        ← BYOL packages, named volume (rw), on PYTHONPATH
```

Source, config, and asset directories are mounted read-only from the host. Changes made on the host are immediately reflected inside the container without rebuilding the image. The output directory is mounted read-write for persisting annotated video to the host filesystem.

---

## Environment Variables (.env)

| Variable | Default | Description |
|:---------|:--------|:------------|
| `WEDA_CV_IMAGE` | `harbor.edgesync.cloud/weda-ai/weda-cv:1.0.0-thor` | Base container image |
| `CONTAINER_NAME` | `weda-cv-object-counter` | Docker container name |
| `SERVICE_NAME` | `object-counter` | Docker Compose service name |
| `ACCEPT_ULTRALYTICS_EULA` | `false` | Set to `true` to skip the interactive EULA prompt |
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU devices exposed to the container |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,video,utility,graphics,display` | NVIDIA driver capabilities |
| `QT_X11_NO_MITSHM` | `1` | Prevents Qt shared memory errors in containerized X11 |
| `LOG_MAX_SIZE` | `20m` | Docker JSON log file size limit |
| `LOG_MAX_FILE` | `3` | Docker JSON log file rotation count |

The `DISPLAY`, `XAUTHORITY`, and `X11_SOCKET` variables are auto-detected at runtime by `build.sh` and should not be hardcoded in `.env`.

**Important:** `ACCEPT_ULTRALYTICS_EULA` defaults to `false`. The user must accept the license via the `build.sh` interactive prompt or explicitly set it to `true` in `.env`. Defaulting to `true` would violate the commercial license terms.

---

## Input Sources

| Type | Config Example |
|:-----|:---------------|
| Video file | `source: data/test.mp4` |
| USB webcam | `source: 0` |
| RTSP stream | `source: rtsp://admin:pass@192.168.1.100:554/stream` |
| Image directory | `source: data/frames/` |

---

## Output

| Item | Location |
|:-----|:---------|
| Annotated video | `output/counted_output.avi` (configurable) |
| Console logs | stdout (with timestamps, FPS, frame counts) |
| Benchmark results | `test_logs/` (from `wise-bench`) |

---

## Supported Model Formats

| Format | Extension | Auto-download | Notes |
|:-------|:----------|:--------------|:------|
| PyTorch | `.pt` | Yes | Preferred — default format |
| ONNX | `.onnx` | No | Must exist as a local file |
| TensorRT | `.engine` / `.trt` | No | Device-specific, must be pre-built on target |

For non-PyTorch formats, place the model file in `data/` or provide an absolute path in `model.path`.

---

## Documentation

| Document | Description |
|:---------|:------------|
| [User Guide](USER_GUIDE.md) | Building custom counting applications, deployment, and performance tuning |
| [Troubleshooting Guide](TROUBLESHOOTING.md) | Production failure scenarios with diagnostics and resolutions |
| [Release Notes](RELEASE_NOTES.md) | QA release note — changelog, test matrix, known limitations |
| [Configuration Schema](config/config.schema.md) | Full parameter reference for `config.yaml` |

---

## License

The WEDA-CV base container image is subject to Advantech commercial licensing terms. Ultralytics components are installed via BYOL (Bring Your Own License) and are subject to the Ultralytics EULA. NVIDIA JetPack and CUDA components are subject to the NVIDIA Software License Agreement.

---

## Support

For known issues and resolutions, consult the [Troubleshooting Guide](TROUBLESHOOTING.md). For deployment guidance, consult the [User Guide](USER_GUIDE.md). For issues outside the scope of this documentation, contact the WISE-Edge Platform Division.

---

© 2026 Advantech Corporation. All rights reserved.
