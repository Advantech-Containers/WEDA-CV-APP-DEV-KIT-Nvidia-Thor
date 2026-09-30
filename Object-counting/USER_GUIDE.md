# WEDA-CV Object Counting Demo — User Guide

**Version:** 1.0.0  
**Platform:** NVIDIA Jetson Thor · JetPack 7.1  
**Audience:** System integrators, application developers, and Advantech partner engineers

---

## 1. Introduction

The WEDA-CV Object Counting Demo provides a unified object counting application that supports line crossing, rectangular region, and arbitrary polygon counting modes. All counting behavior is driven by a single YAML configuration file. This guide explains how to configure the application for different counting scenarios, deploy it in a container, tune performance for production workloads, and adapt it for custom use cases.

---

## 2. Core Concepts

### 2.1. One Application, Three Counting Modes

The `object_counter.py` file handles all three counting tasks. The `task` key in `config/config.yaml` selects the mode, and the `roi.points` key defines the counting geometry. Switching from line counting to polygon counting requires editing two keys in the YAML file — no code change.

### 2.2. Configuration Is the Application

Each YAML configuration file defines a complete counting application. The file specifies the counting task, ROI geometry, model, video source, class filter, tracker, visualization options, and output parameters. Running the application with a different `--config` argument produces a different counting scenario without any code modification.

### 2.3. Model Format Support

The application accepts any model format supported by Ultralytics: PyTorch (`.pt`), ONNX (`.onnx`), TensorRT (`.engine` / `.trt`), and TorchScript. PyTorch is the preferred and default format — `.pt` files are auto-downloaded by Ultralytics if not present locally. All other formats must exist as local files in the project root or `assets/` directory.

---

## 3. Configuration Walkthrough

### 3.1. Selecting the Counting Task

The `task` key determines the counting mode. Each mode requires a minimum number of ROI points.

```yaml
# Line crossing — 2 points define the crossing boundary
task: line_count
roi:
  points:
    - [20, 400]
    - [1080, 400]
```

```yaml
# Rectangular region — 4 points define the zone
task: region_count
roi:
  points:
    - [20, 400]
    - [1080, 400]
    - [1080, 360]
    - [20, 360]
```

```yaml
# Arbitrary polygon — 3+ points define the zone
task: polygon_count
roi:
  points:
    - [100, 300]
    - [600, 280]
    - [700, 450]
    - [400, 500]
    - [50, 420]
```

The application validates ROI point count at startup. If the number of points is insufficient for the selected task, the application exits with an error specifying the minimum requirement.

### 3.2. Defining the ROI Geometry

ROI points are pixel coordinates in the frame's coordinate space (`[x, y]`). To determine the correct coordinates for your video source, pause the video, note the pixel positions where you want the counting boundary, and enter them as `roi.points` in the config.

For line counting, the two points define a straight line. Objects are counted when their tracked centroid crosses this line.

For region and polygon counting, the points define a closed shape. Objects are counted when their tracked centroid enters or exits the enclosed area.

### 3.3. Filtering Object Classes

The `classes` key controls which COCO object categories are detected and counted. Set `null` to count all 80 COCO classes. Provide an integer list to filter to specific categories.

```yaml
# Count only people and cars
classes: [0, 2]
```

Common COCO class indices: 0 = person, 1 = bicycle, 2 = car, 3 = motorcycle, 5 = bus, 7 = truck. See `config/config.schema.md` for the full class ID table.

### 3.4. Configuring the Model

```yaml
model:
  path: yolo26n.pt       # Model file — auto-downloads .pt if not present
  format: pt              # pt, onnx, engine, trt, torchscript
  half: true              # FP16 inference (recommended for Jetson)
  imgsz: 640              # Input resolution
  vid_stride: 1           # Process every Nth frame (2 = skip alternate frames)
```

For TensorRT deployment, export the model inside the container:

```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo26n.pt')
model.export(format='engine', half=True, imgsz=640)
"
```

Then set `model.path: yolo26n.engine` and `model.format: engine` in the config.

### 3.5. Selecting the Video Source

```yaml
# Local video file
source: assets/test.mp4

# USB camera (device index)
source: 0

# RTSP IP camera
source: rtsp://admin:password@192.168.1.100:554/stream

# Directory of images
source: assets/frames/
```

### 3.6. Visualization and Output

```yaml
# Live display
show: true                # Requires X11 — set false for headless
show_labels: true         # Class labels on detections
show_conf: true           # Confidence scores on detections
line_width: 2             # Bounding box and ROI line thickness

# Output recording
output_dir: output
output_filename: counted_output.avi
```

Set `show: false` for headless deployments (no display attached). The annotated video is still written to the output file.

---

## 4. Building a Custom Counting Application

### 4.1. Copy the Default Configuration

```bash
cp config/config.yaml config/my_counting_app.yaml
```

### 4.2. Edit the Configuration

Change the task, ROI points, model, source, classes, and output filename to match your counting scenario. For example, to count vehicles crossing a highway lane boundary:

```yaml
task: line_count
model:
  path: yolo26n.pt
  format: pt
  half: true
  imgsz: 640
  vid_stride: 1

source: rtsp://192.168.1.50:554/highway

classes: [2, 5, 7]          # car, bus, truck

confidence: 0.35
iou_threshold: 0.5
tracker: botsort.yaml

roi:
  points:
    - [100, 500]
    - [1180, 500]

show: true
show_labels: true
show_conf: false
line_width: 3

output_dir: output
output_filename: highway_count.avi
```

### 4.3. Run the Custom Application

```bash
python3 src/object_counter.py --config config/my_counting_app.yaml
```

---

## 5. Deploying in a Container

### 5.1. Project Directory Layout

Ensure the project directory contains all required files:

```
my-counting-app/
├── config/
│   └── config.yaml
├── src/
│   └── object_counter.py
├── assets/
│   └── (model files, sample videos)
├── Dockerfile
├── docker-compose.yml
├── build.sh
├── entrypoint.sh
├── wise-bench
├── .env
└── output/
```

### 5.2. Deployment Modes

The demo supports two deployment strategies configurable in `docker-compose.yml`.

**Local build (development).** Code is mounted as volumes and dependencies are built into a local image. Use the `build:` section in `docker-compose.yml`:

```yaml
build:
  context: .
  dockerfile: Dockerfile
```

**Registry image (production).** Use the pre-built platform image directly. Missing demo-specific dependencies are installed at runtime by `entrypoint.sh`. Uncomment the `image:` line and comment out the `build:` section:

```yaml
image: ${WEDA_CV_IMAGE}
```

### 5.3. Volume Mounts

The `docker-compose.yml` mounts `./src`, `./config`, and `./assets` as read-only volumes. Source changes on the host are immediately reflected inside the container without rebuilding the image. The `./output` directory is mounted read-write for persisting annotated video.

---

## 6. Performance Tuning

### 6.1. Maximizing FPS

**Enable FP16 inference.** Set `model.half: true`. This uses the GPU's Tensor Cores for half-precision computation, roughly doubling inference throughput compared to FP32.

**Reduce input resolution.** Set `model.imgsz: 480` or `model.imgsz: 320`. Lower resolution reduces inference time proportionally at the cost of detection accuracy for small objects.

**Skip frames.** Set `model.vid_stride: 2` to process every other frame. The application writes unprocessed frames directly to the output to maintain video continuity. Effective FPS doubles at the cost of counting granularity.

**Disable GUI display.** Set `show: false`. The `cv2.imshow()` call adds 1–3 ms per frame. For headless deployments or benchmarking, disabling the display removes this overhead.

**Filter classes.** Set `classes: [0]` instead of `null`. Fewer detected objects means less tracker overhead per frame.

### 6.2. Monitoring Performance

The application logs average FPS every 100 frames:

```
  Processed 100 frames (100/343) — 28.5 FPS
  Processed 200 frames (200/343) — 29.1 FPS
```

For deeper analysis, use `wise-bench` to establish the device's raw inference throughput ceiling, then compare application FPS against that baseline.

---

## 7. Running Without Docker

If the WEDA-CV environment is installed natively on the host:

```bash
cd object-counting
pip install pyyaml ultralytics
python3 src/object_counter.py --config config/config.yaml
```

This bypasses the container entirely. Ensure that PyTorch, OpenCV, and CUDA are properly installed on the host.

---

## 8. Frequently Asked Questions

**Q: Can I use a segmentation model (-seg) for counting?**

A: Yes. Set `model.path` to a segmentation model such as `yolo26n-seg.pt`. The ObjectCounter will track and count objects using detection boxes. The segmentation masks are available in the results but the current ObjectCounter visualization does not render them.

**Q: How do I determine the correct ROI coordinates?**

A: Play the video in any media player, pause at a representative frame, and note the pixel coordinates of the boundary you want to define. Alternatively, run the application once with an approximate ROI, observe the rendered line/region on the GUI, and adjust the coordinates iteratively.

**Q: Can I count objects in multiple zones simultaneously?**

A: The current release supports a single ROI per application instance. For multi-zone counting, run multiple instances with separate config files and different ROI definitions. Each instance requires a unique `CONTAINER_NAME` in its `.env` file.

**Q: Why does the video writer produce a warning?**

A: If the output directory does not exist or is not writable, the `cv2.VideoWriter` cannot open. The application logs a warning and continues processing without recording. Ensure the `output/` directory exists on the host — `build.sh` creates it automatically.

**Q: How do I switch between BoT-SORT and ByteTrack?**

A: Set `tracker: botsort.yaml` for better re-identification in complex scenes with occlusions, or `tracker: bytetrack.yaml` for faster tracking with lower overhead. Both tracker configs ship with Ultralytics.

---

© 2026 Advantech Corporation. All rights reserved.
