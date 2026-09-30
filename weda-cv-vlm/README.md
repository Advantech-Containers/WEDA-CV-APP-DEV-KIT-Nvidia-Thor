# WEDA-CV-VLM — Industrial Real-Time Alert and Display HUD

A hardware-accelerated Edge AI pipeline designed for low-latency, real-time spatial safety and vision-language (VLM) reasoning on **NVIDIA Jetson AGX Thor**. 
The main application runs a local, interactive, fullscreen **cv2 Engineer HUD (Heads-Up Display)** that merges multi-modal AI inference with physical RGB-D camera feeds, rendering live telemetry, trajectory collision alerts, and 3-D safety zone analysis at the edge.

## Features

| Capability | What it does |
|---|---|
| Real-time detection | YOLO26 instance segmentation/detection, TensorRT FP16, runs at camera-native frame rate |
| Multi-object tracking | ByteTrack — stable track IDs across frames |
| 3-D spatial reasoning | DPT depth + camera intrinsics → metric distance, position, velocity per track |
| Scene description | Qwen3-VL-30B-A3B grounded auto-caption on every scene change |
| Physical reasoning | Cosmos-Reason2-2B for spatio-temporal and safety queries |
| Semantic embeddings | Cosmos-Embed1-224p for scene-change detection and semantic search |
| Sensor-grounded fact-checking | Cross-references every VLM-emitted numeric claim against live depth and bbox data |
| Scene memory | 30-minute ring buffer with semantic search over past frames |
| Predictive collision | Pairwise 3-D closest-approach prediction with ETA and closing speed |
| Query tier routing | Cost-aware, context-aware dynamic routing of queries to rule engine or LLMs.|
| Engineer HUD | Local cv2 fullscreen with detections, depth, scene text, status LEDs, telemetry |
| Per-query telemetry | Wall time, GPU peak, UMA delta, energy and cost attribution per inference |
| Event Timeline | A scrolling historical log of safety violations, alarms, and zone intrusions.|
| Subsystem Status LEDs | Live status flags representing model loading, VLM query state, hardware camera connectivity, and overall system health.|
| Bird's-Eye View (BEV) Panel | Toggled spatial grid mapping that projects detected tracks onto a camera-centric 2D ground plane (depth metric grid).|
| Colorized Depth Panel (PiP) | A Picture-in-Picture color-mapped depth stream aligned pixel-for-pixel with the RGB stream (Depth-to-Color calibration).|
| Physical Safety Alerts | Real-time alerts based on spatio-temporal predictions, closest point of approach (CPA), and time-to-impact (TTC).|
| Event-Based Query | Based on the alert log query with timestamp and event | 


## 🖥️ Local Engineer HUD

The main visualization interface is a high-performance, fullscreen `OpenCV` HUD designed for industrial edge monitors. It consolidates raw RGB camera feeds, aligned depth maps, projected Bird's-Eye View coordinates, VLM captions, safety alarm logs, and hardware telemetry into a single unified display. 

All panels are synchronized with zero-copy memory buffers, allowing the operator to monitor full pipeline status and system loads in real time.

---

### ⌨️ Interactive Keyboard Controls
When the HUD window is active, operators can use the following hotkeys to control display parameters and query the VLM:

| Hotkey | Action | Description |
| :---: | --- | --- |
| `T` / `t` | **VLM Query Input** | Opens a text prompt overlay to type a question for the VLM in real-time. |
| `Enter` | **Submit Query** | Submits the query. Escapes query input mode. |
| `Esc` | **Cancel / Close** | Exits query input mode or dismisses the scrolling response modal. |
| `X` / `x` | **Toggle Full Modal** | Displays a scrollable modal overlay with the VLM's complete text response. |
| `J`/`K` / Arrows | **Scroll Response** | Scrolls up/down inside the full VLM response modal. |
| `M` / `m` | **Toggle Masks** | Toggle overlay rendering of YOLO26 segmentation masks. |
| `G` / `g` | **Toggle Safety Rings** | Toggle rendering of the safety distance boundary circles in the RGB/BEV views. |
| `B` / `b` | **Toggle BEV Panel** | Show/hide the spatial Bird's-Eye View projection panel. |
| `F` / `f` | **Toggle Fullscreen** | Toggle the cv2 display between windowed and fullscreen mode. |
| `R` / `r` | **Reset Cache** | Flush VLM scene memory cache and tracking histories. |
| `Q` / `q` | **Exit App** | Cleanly shuts down the pipeline, cameras, and display. |

---

## ⚙️ Target System & Host Prerequisites

The application is optimized for **NVIDIA Jetson AGX Thor** running **JetPack 7.1** or newer.

### Hardware & Driver Requirements

| Component | Why it is needed | Configuration |
|---|---|---|
| **[NVIDIA JetPack](https://developer.nvidia.com/embedded/jetpack) ≥ 7.1** | Provides CUDA 13, TensorRT 10, and hardware GStreamer codecs (`nvv4l2decoder`, `nvvidconv`, `nvv4l2h264enc`). | Must be installed on the Jetson Thor host. |
| **X11 Server & xhost** | Required to display the local fullscreen cv2 window on the host screen. | Run `xhost +local:root` on host before launching. |
| **[Orbbec SDK (pyorbbecsdk)](https://github.com/orbbec/pyorbbecsdk)** | Interfaces with Orbbec RGB-D cameras for hardware-synchronized spatial frames. | Install udev rules on host (`99-obsensor-libusb.rules`). |
| **Jetson MAXN Power Profile** | Maximizes clock rates for GPU/CPU UMA architectures. | `sudo nvpmodel -m 0 && sudo jetson_clocks` |

---

## 🚀 Use Cases & Industrial Applications

WEDA-CV-VLM is designed for edge deployment in high-churn safety-critical industrial settings:

1.  **Forklift & AGV Proximity Alerting**: Spatio-temporal tracking projects trajectories of vehicles and workers in real-time, raising critical flags if a collision is imminent.
2.  **3-D Safety Zone Intrusion**: Virtual exclusion zones (e.g., around high-voltage equipment, robotic arms, or conveyor lanes) are monitored via calibrated depth metrics, ignoring static background objects.
3.  **Autonomous Scene Narration**: High-level visual grounding allows the AI platform to narrate activities, detect safety violations (e.g., "Worker not wearing helmet near conveyor"), and report anomalies without manual rule programming.

---

## 🤖 Embedded AI Models

The edge intelligence pipeline runs four specialized models locally:

| Model | Role | Specs & Precision |
|---|---|---|
| **YOLO26n** | Real-time object detection & segmentation | TensorRT Engine (FP16, direct loading) |
| **Cosmos-Embed1-224p** | Scene change detection & re-identification | PyTorch BF16 (Jetson GPU optimized) |
| **Cosmos-Reason2-2B** | Spatio-temporal safety & physical common sense | PyTorch FP16 (Jetson GPU optimized) |
| **LLaVA-OneVision-Qwen2-7b** | Vision-language grounding & scene understanding | HuggingFace BF16 |

---

## 🏎️ End-to-End Jetson Optimizations

To achieve real-time (25–30 FPS) end-to-end execution on the Jetson Thor system, WEDA-CV-VLM implements a specialized edge optimization stack:

1.  **Direct TensorRT Inference Backend (`tensorrt_direct`)**
    *   *Metadata Header Stripping*: Automatically detects and strips Ultralytics engine headers from serialized `.engine` files, loading raw engine binaries into the TensorRT 10 runtime context without wrapper overhead.
    *   *Static Shape Optimization*: Bypasses context shape settings for static input layers, preventing runtime C++ exceptions in TensorRT 10.
2.  **Zero-Copy Unified Memory Architecture (UMA)**
    *   *Direct Memory Mapping*: Allocates GPU buffers in UMA memory mapped directly to PyTorch tensors, eliminating host-device CPU-GPU transfer bottlenecks.
3.  **Hardware Coprocessor Preprocessing**
    *   *VIC Offloading*: Offloads frame resizing and color space conversion to the Video Image Coprocessor (VIC) hardware block, keeping the main CPU cores free.
4.  **Host GPU Cache Tuning**
    *   *L1 Cache Preference*: Maximize L1 cache allocations for GPU depth kernels, improving spatial computation performance by 5–10%.
5.  **Memory Management & Scheduling**
    *   *Pre-allocated Buffer Pools*: Avoids runtime heap allocations by using cyclic pools for frame buffers.
    *   *Adaptive Placement Scheduler*: Continuously monitors GPU load; if GPU utilization exceeds 85%, less critical stages are offloaded dynamically to the CPU.

---

## 📁 Repository Layout

```
.
├── README.md                ← you are here
├── TROUBLESHOOTING.md       ← operator reference (errors, fixes, debug workflow)
├── docker-compose.yml       ← service orchestration
├── Dockerfile               ← container build
├── build.sh                 ← canonical entry point: X11 detect + compose up
├── entrypoint.sh            ← container init, deps, healthchecks
├── .env                     ← fill HF_TOKEN
│
├── app/
│   ├── main.py              ← pipeline launcher
│   ├── core/                ← Orchestrator + Pipeline
│   ├── modules/             ← detector, depth, tracker, embedder, reasoning, physical_reasoner
│   ├── display/             ← cv2 HUD renderer
│   ├── io/                  ← orbbec / v4l2 / RTSP / GStreamer backends
│   └── utils/               ← data structures, logger, alert log
│
├── weda_optim/              ← optimization utilities
│   ├── gpu_monitor.py       ← Thor-aware tegrastats / sysfs / nvitop
│   ├── model_loader.py      ← VLM quant dispatch (BF16 / AWQ / NVFP4)
│   └── scheduler.py         ← Placement scheduler
│
├── config/                  ← pipeline.yaml
├── scripts/                 ← prepare_models.py, wise-bench
└── assets/                  ← demo media
```

---

## 🚦 Local Startup & Verification

### 1. Environment Setup
The bundle ships a ready-to-edit `.env`. Update it to set your HuggingFace token:
```bash
# Edit .env to set:
#   HF_TOKEN=hf_...
#   ACCEPT_ULTRALYTICS_EULA=false
```

### 2. Grant X11 Display Permissions
Allow the Docker container to connect to the host's X11 server for the cv2 HUD:
```bash
xhost +local:root
```

### 3. Build & Launch Container (on host terminal)
Run the build script to compile and launch the containerized application:
```bash
chmod +x build.sh entrypoint.sh scripts/wise-bench
```
```bash
./build.sh
```
This builds or pulls the container, starts it, and opens a shell inside the container.

### 4. Initialize and Run the HUD (on container terminal)
Inside the container shell:
```bash
# Prepare and cache the AI models (first run only)
python3 scripts/prepare_models.py

# AI libraries and packages functionality checking
./scripts/wise-bench
./scripts/wise-bench --heavy # for model loading functionality test
# Launch the pipeline with the local cv2 HUD
python3 -m app.main --config config/pipeline.yaml
```

---
