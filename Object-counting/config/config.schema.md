# Configuration Schema — WEDA-CV ObjectCounting Demo

Complete reference for `config.yaml` parameters.

---

## Task

| Key | Type | Default | Description |
|---|---|---|---|
| `task` | `string` | `line_count` | Counting mode. One of: `line_count`, `region_count`, `polygon_count` |

### Task Descriptions

| Task | ROI Points | Use Case |
|---|---|---|
| `line_count` | 2 points (a line) | Count objects crossing a line (e.g., doorway, road lane) |
| `region_count` | 4 points (rectangle) | Count objects entering/exiting a rectangular zone |
| `polygon_count` | 3+ points (polygon) | Count objects in an arbitrary shaped area |

---

## Model

| Key | Type | Default | Description |
|---|---|---|---|
| `model.path` | `string` | `yolo26n.pt` | Path to the model file. `.pt` files auto-download if missing. |
| `model.format` | `string` | `pt` | Model format: `pt`, `onnx`, `engine`, `trt`, `torchscript` |
| `model.device` | `string` | `"0"` | Inference device: `"0"` (GPU 0), `"cpu"`, `"cuda:0"` |
| `model.half` | `bool` | `false` | Enable FP16 half-precision inference (highly recommended for edge GPUs). |
| `model.imgsz` | `int` | `640` | Native inference resolution size (e.g., 640). |
| `model.vid_stride` | `int` | `1` | Video frame stride. E.g., `2` processes every 2nd frame. |

> **Note:** `.pt` (PyTorch) is the preferred format. Other formats must exist as local files.

---

## Source

| Key | Type | Default | Description |
|---|---|---|---|
| `source` | `string \| int` | `assets/sample.mp4` | Video file path, RTSP URL, webcam index, or image directory |

**Examples:**
- `assets/sample.mp4` — local video file
- `rtsp://user:pass@192.168.1.100:554/stream` — RTSP stream
- `0` — default webcam
- `assets/frames/` — directory of images

---

## ROI (Region of Interest)

| Key | Type | Default | Description |
|---|---|---|---|
| `roi.points` | `list[list[int]]` | `[[20,400],[1080,400]]` | List of `[x, y]` coordinate pairs defining the region |

### ROI Examples by Task

**Line counting (2 points):**
```yaml
roi:
  points:
    - [20, 400]
    - [1080, 400]
```

**Region counting (4 points — rectangle):**
```yaml
roi:
  points:
    - [20, 400]
    - [1080, 400]
    - [1080, 360]
    - [20, 360]
```

**Polygon counting (5 points):**
```yaml
roi:
  points:
    - [100, 300]
    - [600, 280]
    - [700, 450]
    - [400, 500]
    - [50, 420]
```

---

## Detection

| Key | Type | Default | Description |
|---|---|---|---|
| `confidence` | `float` | `0.25` | Minimum confidence threshold (0.0–1.0) |
| `iou_threshold` | `float` | `0.7` | IoU threshold for NMS (0.0–1.0) |
| `classes` | `list[int] \| null` | `null` | COCO class IDs to filter. `null` = all classes. |
| `tracker` | `string` | `botsort.yaml` | Tracker algorithm: `botsort.yaml` or `bytetrack.yaml` |

### Common COCO Class IDs

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | person | 5 | bus |
| 1 | bicycle | 6 | train |
| 2 | car | 7 | truck |
| 3 | motorcycle | 8 | boat |
| 4 | airplane | 9 | traffic light |

---

## Visualization

| Key | Type | Default | Description |
|---|---|---|---|
| `show` | `bool` | `true` | Display live OpenCV window (requires X11) |
| `show_labels` | `bool` | `true` | Show class labels on detections |
| `show_conf` | `bool` | `true` | Show confidence scores |
| `line_width` | `int` | `2` | Line width for bounding boxes and regions |

---

## Output

| Key | Type | Default | Description |
|---|---|---|---|
| `output_dir` | `string` | `output` | Directory for saving output video |
| `output_filename` | `string` | `counted_output.avi` | Output video filename |
