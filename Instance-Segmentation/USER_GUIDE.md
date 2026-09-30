# WEDA-CV Inference Engine — User Guide

**Engine Version:** 3.0.0  
**Platform:** NVIDIA Jetson Thor · JetPack 7.1  
**Audience:** System integrators, application developers, and Advantech partner engineers  

---

## 1. Introduction

The WEDA-CV Inference Engine is a standardized, config-driven pipeline for real-time instance segmentation and multi-object tracking on NVIDIA Jetson edge devices. This guide explains how to use the engine to build custom computer vision applications, deploy them in containers, and optimize performance for production environments.

The engine follows a strict separation of concerns: all application behavior is defined in a YAML configuration file, and the engine source files are never modified. This design ensures that multiple applications (person tracking, vehicle counting, industrial defect detection) can share the same engine codebase, differing only in their configuration.

---

## 2. Core Concepts

### 2.1. The Engine Is Immutable

The following files in `src/` constitute the standard engine and must not be edited:

`instance_segmentation.py` reads the configuration, opens the video source, loads the model, runs the inference-and-annotation loop, and manages output. `visualizer.py` handles all frame annotation (masks, bounding boxes, Track ID labels, object count badges). `memory.py` provides UMA-optimized pinned memory buffers for zero-copy frame I/O on Jetson.

If you need behavior that the current engine does not support, contact the WISE-Edge Platform Division to request a feature rather than forking the engine.

### 2.2. Configuration Is the Application

Each YAML configuration file defines a complete application. The file specifies the model, video source, class filter, tracker, visualization options, output format, and performance settings. Running the engine with a different `--config` argument produces a different application without any code change.

### 2.3. Model Format Agnostic

The engine accepts any model format supported by Ultralytics: PyTorch (`.pt`), ONNX (`.onnx`), TensorRT (`.engine`), OpenVINO, CoreML, or TFLite. Set `model.path` to the file name or path. For TensorRT engines, export the model on the target device using `yolo export model=yolo26n-seg.pt format=engine half=True` inside the container.

---

## 3. Building a New Application

### 3.1. Copy the Default Configuration

Start from the default configuration and modify it:

```bash
cp config/config.yaml config/my_app.yaml
```

### 3.2. Define the Application Identity

Edit the `app:` section. This metadata appears in the startup banner and log output, making it easy to identify which application is running in a multi-container deployment.

```yaml
app:
  name: "Factory Floor Defect Detection"
  version: "1.0.0"
```

### 3.3. Select the Model

Set `model.path` to the model file. Place the file in the project root or in `assets/`. The engine searches in this order: (1) the exact path specified, (2) relative to the project root, (3) inside `assets/`. If the file is not found, Ultralytics attempts an auto-download from its model hub (requires internet access).

```yaml
model:
  path: "yolo26n-seg.pt"    # Nano segmentation model
  device: "cuda"
  half: True                 # FP16 — recommended for Jetson
  imgsz: 640                 # Reduce to 480 or 320 for higher FPS
  retina_masks: False        # False = faster; True = higher mask quality
```

For TensorRT deployment, export inside the container:

```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo26n-seg.pt')
model.export(format='engine', half=True, imgsz=640)
"
```

Then set `model.path: "yolo26n-seg.engine"` in the config.

### 3.4. Configure the Video Source

The `source` key accepts four formats:

```yaml
# Local video file
source: assets/factory_line.mp4

# USB camera (integer device index)
source: 0

# RTSP IP camera
source: rtsp://192.168.1.100:554/stream

# HTTP stream
source: http://192.168.1.100:8080/video
```

When using a USB camera, configure the `camera:` section to optimize capture performance:

```yaml
camera:
  backend: "v4l2"       # Bypass GStreamer overhead
  fourcc: "MJPG"        # Hardware-compressed capture (lower USB bandwidth)
  fps: 30               # Match camera capability
  buffer_size: 1        # Lowest latency
  threaded: True         # Decouple capture from inference
```

To determine what your camera supports, run on the host or inside the container:

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

### 3.5. Filter Object Classes

The `classes` key controls which COCO object categories the engine detects and tracks. Set `null` to detect all 80 COCO classes. Provide an integer list to filter to specific categories.

Common COCO class indices:

| Index | Class |
|-------|-------|
| 0 | person |
| 1 | bicycle |
| 2 | car |
| 3 | motorcycle |
| 5 | bus |
| 7 | truck |

```yaml
# Detect only people and cars
classes: [0, 2]
```

Filtering classes reduces computational load because the tracker processes fewer objects per frame.

### 3.6. Configure Visualization

Set `visualization.show: False` for headless deployments (no display attached). All annotation is still applied to the output video file.

For display-equipped demos, customize the appearance:

```yaml
visualization:
  show: True
  window_name: "Defect Detection Live"
  show_labels: True
  show_conf: True          # Show confidence for quality inspection
  show_boxes: True
  show_masks: True
  show_count: True
  line_width: 3            # Thicker lines for large displays
  mask_alpha: 0.40
  label_font_scale: 0.7
```

### 3.7. Configure Output

Disable output for live-only deployments or benchmarking:

```yaml
output:
  enabled: False
```

For recording, configure the codec and async writer:

```yaml
output:
  enabled: True
  dir: "output"
  filename: "defect_detection_output.avi"
  codec: "MJPG"              # Fastest encoder
  async_writer: True          # Encoding off critical path
  queue_size: 128             # Larger buffer for high-FPS pipelines
```

### 3.8. Run the Application

```bash
python3 src/instance_segmentation.py --config config/my_app.yaml
```

---

## 4. Deploying in a Container

### 4.1. Project Directory Layout

Ensure your project directory contains all required files:

```
my-application/
├── config/
│   └── config.yaml              # Your application config
├── src/
│   ├── instance_segmentation.py # Standard engine (unchanged)
│   ├── visualizer.py            # Standard annotator (unchanged)
│   └── memory.py                # Standard memory manager (unchanged)
├── assets/
│   └── (model files, test videos)
├── Dockerfile
├── docker-compose.yml
├── build.sh
├── entrypoint.sh
├── wise-bench
├── .env
└── output/
```

### 4.2. The `.env` File

Create a `.env` file with static configuration. These values do not change between runs:

```env
WEDA_CV_IMAGE=harbor.edgesync.cloud/weda-ai/weda-cv:thor-1.0.0
CONTAINER_NAME=my-defect-detection
SERVICE_NAME=instance-segmentation
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=all
LD_LIBRARY_PATH=/usr/local/lib
QT_X11_NO_MITSHM=1
NO_AT_SPI=1
DBUS_SESSION_BUS_ADDRESS=/dev/null
ACCEPT_ULTRALYTICS_EULA=true
LOG_MAX_SIZE=10m
LOG_MAX_FILE=3
```

Set `ACCEPT_ULTRALYTICS_EULA=true` to enable non-interactive Ultralytics installation inside the container. If set to `false`, the entrypoint will prompt interactively (requires `tty: true` in compose).

### 4.3. Build and Launch

```bash
./build.sh
```

The script performs: pre-flight validation (Docker, Compose, JQ), X11 display detection, EULA acceptance, image synchronization, container launch, environment hydration (dependency installation, Ultralytics setup, patches), and shell entry.

### 4.4. Volume Mounts

The `docker-compose.yml` mounts `./src`, `./config`, and `./assets` as read-only volumes. This means source changes on the host are immediately reflected inside the container without rebuilding the image. The `./output` directory is mounted read-write for persisting annotated video output to the host filesystem.

This mount strategy enables rapid iteration: edit `config/config.yaml` on the host, re-run `python3 src/instance_segmentation.py` inside the container, and results appear in `./output` on the host.

### 4.5. Rebuilding the Image

If you need to bake new model files or dependencies into the image (rather than mounting them), modify the `Dockerfile` and run `./build.sh` (without `--run`). The build step copies `src/`, `config/`, `assets/`, and `wise-bench` into the image.

---

## 5. Performance Tuning

### 5.1. Maximizing FPS

The engine provides several configuration knobs for trading accuracy against speed:

**Reduce input resolution.** Setting `model.imgsz: 480` or `model.imgsz: 320` reduces inference time proportionally. This is the highest-impact single change for FPS improvement.

**Skip frames.** Setting `model.vid_stride: 2` processes every other frame. The pipeline writes unprocessed frames directly to the output, maintaining video continuity. Effective FPS doubles at the cost of tracking granularity.

**Disable retina masks.** `model.retina_masks: False` produces masks at model resolution (160×160 for `imgsz=640`) and upscales via nearest-neighbor during annotation. This saves 8–10 ms per frame compared to full-resolution mask compositing.

**Disable recording.** `output.enabled: False` eliminates the async writer thread and all encoding overhead. Use this for live demo or benchmarking scenarios.

**Disable masks in visualization.** `visualization.show_masks: False` skips mask rendering entirely. Bounding boxes and Track ID labels remain. This reduces annotation cost to less than 1 ms per frame.

### 5.2. Interpreting Timing Output

Enable `debug.timing: True` to see per-component latency:

```
  Processed 100 frames (100/343) — 34.3 FPS avg
  ┣ Track: 15.4ms | Annotate: 11.3ms | Write: 0.0ms | Display: 1.5ms
```

**Track** is the YOLO inference + ByteTrack association time. This is GPU-bound and primarily determined by model size and `imgsz`. **Annotate** is `results.plot()` + custom label drawing. This is CPU-bound for mask compositing. **Write** should be 0.0 ms when the async writer is enabled (encoding runs on a background thread). If it shows a non-zero value, the writer queue is full and the main loop is blocking. **Display** is `cv2.imshow()` overhead, typically 1–2 ms.

### 5.3. UMA Memory Manager

The memory manager is enabled by default. To measure its impact, disable it and compare FPS:

```yaml
performance:
  memory:
    enabled: False    # Disable to measure baseline without pinned buffers
```

On Jetson Thor, the pinned buffer optimization primarily eliminates per-frame `cudaMalloc`/`cudaFree` jitter rather than improving peak throughput. The benefit is most visible in P99 latency and frame-time consistency, not average FPS.

---

## 6. Running Benchmarks

The `wise-bench` tool provides standardized hardware and software validation:

```bash
./wise-bench
```

This runs GPU passthrough, CUDA, TensorRT, GStreamer, and hardware accelerator tests. Results are written to `test_logs/`. Use `wise-bench` output to verify that the container environment is correctly configured before running application benchmarks.

For application-level benchmarking, run the engine with `debug.timing: True`, `visualization.show: False`, and `output.enabled: False`:

```yaml
visualization:
  show: False
output:
  enabled: False
debug:
  timing: True
  timing_interval: 50
```

This isolates inference and annotation throughput from display and I/O overhead.

---

## 7. Multiple Applications on One Device

The engine supports running multiple applications simultaneously by using separate container instances with different configuration files:

```bash
# Terminal 1: Person tracking on RTSP camera
docker-compose -f docker-compose.yml --env-file .env.person up -d

# Terminal 2: Vehicle counting on USB camera
docker-compose -f docker-compose.yml --env-file .env.vehicle up -d
```

Each instance requires its own `.env` file with a unique `CONTAINER_NAME`. Ensure sufficient GPU memory for concurrent models. Monitor with `tegrastats` on the host.

---

## 8. Extending the Engine

### 8.1. Adding Post-Processing Logic

If your application requires post-processing beyond what the engine provides (e.g., line-crossing counting, zone intrusion alerts, HTTP webhook notifications), implement it as a separate module that consumes the engine's output video or integrates via a callback mechanism. Do not modify the engine source files.

A recommended pattern is to write a wrapper script that imports the engine's `run()` function and adds pre/post hooks:

```python
# my_app.py — wraps the standard engine with custom logic
from src.instance_segmentation import load_config, run
cfg, _ = load_config("config/my_app.yaml")
# Add custom post-processing to cfg or wrap run()
run(cfg, log)
```

### 8.2. Custom Annotation

Annotation appearance is driven entirely by the `visualization` block of your config file — no code change is required:

| Key | Effect |
|:----|:-------|
| `show_boxes` | Draw the bounding box rectangles |
| `show_masks` | Overlay the segmentation masks |
| `show_labels` | Draw the `Track ID: N` labels |
| `show_conf` | Append the confidence score to each label |
| `show_count` | Draw the `Objects Tracked: N` badge |
| `line_width` | Bounding box thickness |
| `mask_alpha` | Mask transparency (0.0–1.0) |
| `label_font_scale` | Label text size |

If you need visual changes beyond these options — a different label layout, a custom colour palette, or additional overlays — edit the drawing helpers in `src/visualizer.py`. The engine calls `get_color`, `draw_track_labels`, and `draw_object_count` directly on every annotated frame, so changes there take effect immediately.

---

## 9. Frequently Asked Questions

**Q: Can I use a detection-only model (non-seg)?**

A: Yes. Set `model.path` to a detection model such as `yolo26n.pt`. The engine will still run tracking and draw bounding boxes and Track ID labels. Mask-related visualization options (`show_masks`, `mask_alpha`, `retina_masks`) will have no effect because the model does not produce masks.

**Q: How do I use a TensorRT engine file?**

A: Export the model to TensorRT inside the container (see Section 3.3), then set `model.path` to the `.engine` file. The memory manager's pinned buffer optimization applies to all backends, including TensorRT.

**Q: Why does the camera show 20 FPS even though it supports 30?**

A: OpenCV may default to the GStreamer backend, which adds buffering overhead. Set `camera.backend: "v4l2"` in your config. See the Troubleshooting section of the README for full details.

**Q: Can I run without a display (headless)?**

A: Yes. Set `visualization.show: False`. The engine will process frames and write output without attempting to open an X11 window. The X11 detection step in `build.sh` will still run but is non-blocking if no display is found.

**Q: How do I change the output video format to MP4?**

A: Set `output.filename: "output.mp4"` and `output.codec: "mp4v"`. Note that the `mp4v` codec may not be available in all OpenCV builds. `MJPG` with `.avi` is recommended for edge deployment due to its universal availability and low CPU overhead.

---

© 2026 Advantech Corporation. All rights reserved.
