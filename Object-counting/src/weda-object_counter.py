#!/usr/bin/env python3
"""
WEDA-CV ObjectCounting Demo
============================
Unified Ultralytics YOLO object counting application.

Supports three counting tasks configured via YAML:
  - line_count    : Count objects crossing a 2-point line
  - region_count  : Count objects in a 4-point rectangular zone
  - polygon_count : Count objects in an N-point polygon zone

Usage:
    python3 src/object_counter.py                          # uses default config
    python3 src/object_counter.py --config path/to/cfg.yaml
"""
import argparse
import logging
import os
os.environ["NO_AT_BRIDGE"] = "1"
import signal
import sys
import time
import queue
import threading
from pathlib import Path
import cv2
import yaml
import numpy as np
from memory import MemoryManager
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ObjectCounter")
_shutdown_requested = False

def _handle_signal(signum, _frame):
    global _shutdown_requested
    log.warning("Received signal %s — shutting down gracefully …", signal.Signals(signum).name)
    _shutdown_requested = True

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)
# ---------------------------------------------------------------------------
# Threaded Camera Reader
# ---------------------------------------------------------------------------
class ThreadedCamera:
    """
    Background capture thread using a Pinned Memory Ring Buffer and inline CUDA resizing.
    This completely decouples decoding, scaling, and host-to-device memory prep from inference.
    """
    def __init__(self, cap: cv2.VideoCapture, resize_target: tuple = None, maxsize: int = 32):
        self._cap = cap
        self._resize = resize_target
        self._q = queue.Queue(maxsize=maxsize)
        self._stopped = False
        self._pool = []
        try:
            import torch
            h = resize_target[1] if resize_target else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = resize_target[0] if resize_target else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            for _ in range(maxsize + 2):
                t = torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True)
                self._pool.append(t.numpy())
            log.info("ThreadedCamera: Allocated %d pinned memory buffers (%dx%d)", len(self._pool), w, h)
        except Exception as e:
            log.warning("ThreadedCamera: Pinned memory pool allocation failed: %s", e)
            self._pool = None
        self._pool_idx = 0
        self._dropped = 0

        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        try:
            if self._resize:
                gpu_in = cv2.cuda_GpuMat()
            while not self._stopped:
                ret, frame = self._cap.read()
                if not ret:
                    break
                if self._resize:
                    try:
                        gpu_in.upload(frame)
                        gpu_out = cv2.cuda.resize(gpu_in, self._resize)
                        frame = gpu_out.download()
                    except Exception:
                        frame = cv2.resize(frame, self._resize)
                if self._pool is not None:
                    _deadline = time.time() + 0.1
                    while self._q.full() and not self._stopped and time.time() < _deadline:
                        time.sleep(0.001)
                    if self._stopped:
                        break
                    if self._q.full():
                        self._note_drop()
                        continue
                    pinned_frame = self._pool[self._pool_idx]
                    np.copyto(pinned_frame, frame)
                    frame = pinned_frame
                    try:
                        self._q.put((True, frame), timeout=0.1)
                        self._pool_idx = (self._pool_idx + 1) % len(self._pool)
                    except queue.Full:
                        self._note_drop()
                    continue
                try:
                    self._q.put((True, frame), timeout=0.1)
                except queue.Full:
                    self._note_drop()
        except Exception as e:
            log.error(f"[ThreadedCamera] Fatal Error: {e}")
        finally:
            self._put_sentinel()

    def _note_drop(self):
        self._dropped += 1
        if self._dropped % 100 == 0:
            log.warning("ThreadedCamera: %d frames dropped (consumer slower than camera)",
                        self._dropped)

    def _put_sentinel(self):
        try:
            self._q.put_nowait((False, None))
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait((False, None))
            except queue.Full:
                pass

    def read(self):
        try:
            return self._q.get(timeout=0.01)
        except queue.Empty:
            return True, None

    def release(self):
        self._stopped = True
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._t.join(timeout=1.0)
        if not self._t.is_alive():
            self._cap.release()

class AsyncVideoWriter:
    def __init__(self, path: str, fourcc: int, fps: int, size: tuple, maxsize: int = 32):
        self._writer = cv2.VideoWriter(path, fourcc, fps, size)
        self._q = queue.Queue(maxsize=maxsize)
        self._stopped = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            self._writer.write(item)
    def write(self, frame):
        if not self._stopped:
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                pass
    def release(self):
        self._stopped = True
        self._q.put(None)
        self._t.join(timeout=10)
        self._writer.release()
    def isOpened(self) -> bool:
        return self._writer.isOpened()

class PostProcessWorker:
    def __init__(self, region_points, task, writer, show=False,
                 show_labels=True, show_conf=True, line_width=2):
        self._q = queue.Queue(maxsize=32)
        self._display_q = queue.Queue(maxsize=2)
        self._region = region_points
        self._task = task
        self._writer = writer
        self._show = show
        self._show_labels = show_labels
        self._show_conf = show_conf
        self._line_width = line_width
        self._counts = 0
        self._counted_ids = set()
        self._history = {}
        self._last_seen = {}
        self._frame_count = 0
        self._history_ttl_frames = 300
        self._last_gc_frame = 0
        self._stopped = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        
    def _run(self):
        while not self._stopped:
            item = self._q.get()
            if item is None: break
            frame, results = item
            self._frame_count += 1
            if self._task == "line_count" and len(self._region) >= 2:
                cv2.line(frame, self._region[0], self._region[1], (255, 0, 255), 3)
            elif len(self._region) >= 3:
                cv2.polylines(frame, [np.array(self._region, dtype=np.int32)], True, (255, 0, 255), 3)
            if results and len(results) > 0 and results[0].boxes and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                xyxy = results[0].boxes.xyxy.cpu().numpy()
                for i, tid in enumerate(track_ids):
                    self._last_seen[tid] = self._frame_count
                    cx = (xyxy[i][0] + xyxy[i][2]) / 2.0
                    cy = (xyxy[i][1] + xyxy[i][3]) / 2.0
                    if self._task == "line_count" and len(self._region) >= 2:
                        x1, y1 = self._region[0]
                        x2, y2 = self._region[1]
                        if tid in self._history:
                            px, py = self._history[tid]
                            d1 = (px - x1)*(y2 - y1) - (py - y1)*(x2 - x1)
                            d2 = (cx - x1)*(y2 - y1) - (cy - y1)*(x2 - x1)
                            if (d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0):
                                if tid not in self._counted_ids:
                                    self._counts += 1
                                    self._counted_ids.add(tid)
                        self._history[tid] = (cx, cy)
                    else:
                        dist = cv2.pointPolygonTest(np.array(self._region, dtype=np.int32), (cx, cy), False)
                        in_region = dist >= 0
                        if tid in self._history:
                            was_in = self._history[tid]
                            if not was_in and in_region:
                                if tid not in self._counted_ids:
                                    self._counts += 1
                                    self._counted_ids.add(tid)
                        self._history[tid] = in_region
            if self._frame_count - self._last_gc_frame >= self._history_ttl_frames:
                self._last_gc_frame = self._frame_count
                for tid, last in list(self._last_seen.items()):
                    age = self._frame_count - last
                    if age > self._history_ttl_frames:
                        self._history.pop(tid, None)
                    if age > 10 * self._history_ttl_frames:
                        self._last_seen.pop(tid, None)
                        self._counted_ids.discard(tid)
            if results:
                frame = results[0].plot(img=frame, labels=self._show_labels,
                                        conf=self._show_conf, line_width=self._line_width)
            cv2.putText(frame, f"Count: {self._counts}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            if self._writer and self._writer.isOpened():
                self._writer.write(frame)
            if self._show:
                try:
                    self._display_q.put_nowait(frame)
                except queue.Full:
                    pass
    def get_display_frame(self):
        try:
            return self._display_q.get_nowait()
        except queue.Empty:
            return None
    def process(self, frame, results):
        if not self._stopped:
            try:
                self._q.put_nowait((frame.copy(), results))
            except queue.Full:
                pass
    def release(self):
        self._stopped = True
        self._q.put(None)
        self._t.join(timeout=2.0)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = "config/config.yaml"
VALID_TASKS = {"line_count", "region_count", "polygon_count"}
SUPPORTED_MODEL_EXTENSIONS = {
    "pt": ".pt",
    "onnx": ".onnx",
    "engine": ".engine",
    "trt": ".trt",
    "torchscript": ".torchscript",
}

def load_config(path: str) -> dict:
    """Load and validate the YAML configuration file."""
    config_path = Path(path)
    if not config_path.is_file():
        log.error("Configuration file not found: %s", config_path)
        sys.exit(1)
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    if not cfg:
        log.error("Configuration file is empty: %s", config_path)
        sys.exit(1)
    task = cfg.get("task", "line_count")
    if task not in VALID_TASKS:
        log.error("Invalid task '%s'. Must be one of: %s", task, ", ".join(sorted(VALID_TASKS)))
        sys.exit(1)
    roi = cfg.get("roi", {})
    points = roi.get("points", [])
    if not points:
        log.error("ROI points not defined in config. At least 2 points are required.")
        sys.exit(1)
    min_points = {"line_count": 2, "region_count": 4, "polygon_count": 3}
    required = min_points.get(task, 2)
    if len(points) < required:
        log.error(
            "Task '%s' requires at least %d ROI points, but %d were provided.",
            task,
            required,
            len(points),
        )
        sys.exit(1)
    log.info("Configuration loaded from '%s'", config_path)
    log.info("  Task      : %s", task)
    log.info("  ROI points: %d", len(points))
    return cfg

def resolve_model(cfg: dict) -> str:
    """
    Resolve the model path.
    Priority:
      1. Exact path from config (if file exists)
      2. Within the mounted data/ directory
      3. Rely on Ultralytics auto-download for .pt files
    """
    model_cfg = cfg.get("model", {})
    model_path = model_cfg.get("path", "yolo26n.pt")
    model_format = model_cfg.get("format", "pt")
    if Path(model_path).is_file():
        log.info("Model resolved: %s (format: %s)", model_path, model_format)
        return model_path
    data_candidate = Path("data") / Path(model_path).name
    if data_candidate.is_file():
        log.info("Model found in data/: %s", data_candidate)
        return str(data_candidate)
    if model_format == "pt" or model_path.endswith(".pt"):
        log.info(
            "Model file '%s' not found locally — Ultralytics will attempt auto-download.",
            model_path,
        )
        return model_path
    log.error(
        "Model file '%s' not found. Non-PT formats must be provided as a local file.",
        model_path,
    )
    sys.exit(1)


def build_region_points(cfg: dict, frame_w: int, frame_h: int) -> list:
    """Convert ROI config points into the list-of-tuples format required by ObjectCounter."""
    raw_points = cfg.get("roi", {}).get("points", [])
    points = []
    for pt in raw_points:
        x, y = pt
        if isinstance(x, float) and x <= 1.0:
            x = x * frame_w
        if isinstance(y, float) and y <= 1.0:
            y = y * frame_h
        points.append((int(x), int(y)))
    return points
# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def run(cfg: dict):
    """Execute the object counting pipeline."""
    from ultralytics import solutions, YOLO
    task = cfg.get("task", "line_count")
    model_path = resolve_model(cfg)
    source = cfg.get("source", "assets/sample.mp4")
    confidence = cfg.get("confidence", 0.25)
    iou_threshold = cfg.get("iou_threshold", 0.7)
    classes = cfg.get("classes")
    tracker = cfg.get("tracker", "botsort.yaml")
    half_precision = cfg.get("model", {}).get("half", False)
    imgsz = cfg.get("model", {}).get("imgsz", 640)
    vid_stride = cfg.get("model", {}).get("vid_stride", 1)
    show = cfg.get("show", False)
    show_labels = cfg.get("show_labels", True)
    show_conf = cfg.get("show_conf", True)
    line_width = cfg.get("line_width", 2)
    output_dir = Path(cfg.get("output_dir", "output"))
    output_filename = cfg.get("output_filename", "counted_output.avi")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    # -----------------------------------------------------------------------
    # Determine source type
    # -----------------------------------------------------------------------
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        cap_source = int(source)
        log.info("Source: webcam device %d", cap_source)
    else:
        cap_source = str(source)
        log.info("Source: %s", cap_source)
    # -----------------------------------------------------------------------
    # Open video capture
    # -----------------------------------------------------------------------
    if isinstance(cap_source, int):
        cap = cv2.VideoCapture(cap_source, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(cap_source)
        
    if not cap.isOpened():
        log.error("Cannot open video source: %s", source)
        sys.exit(1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("Video properties: %dx%d @ %d FPS (%s frames)", w, h, fps,
             total_frames if total_frames > 0 else "live")
    resize_source = cfg.get("resize_source")
    if resize_source and isinstance(resize_source, list) and len(resize_source) == 2:
        w, h = int(resize_source[0]), int(resize_source[1])
        log.info("Scaling input frames to: %dx%d for processing", w, h)
    region_points = build_region_points(cfg, w, h)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = AsyncVideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        log.warning("Could not open video writer — output will not be saved.")
    # -----------------------------------------------------------------------
    # Initialize ObjectCounter
    # -----------------------------------------------------------------------
    log.info("Initializing ObjectCounter for task '%s' …", task)
    log.info("  Model      : %s", model_path)
    log.info("  Region     : %s", region_points)
    log.info("  Confidence : %.2f", confidence)
    log.info("  Tracker    : %s", tracker)
    log.info("  Half-Prec  : %s", half_precision)
    log.info("  Inference-S: %s", imgsz)
    log.info("  Vid-Stride : %d", vid_stride)
    if classes:
        log.info("  Classes    : %s", classes)
    model = YOLO(model_path)
    model.overrides["imgsz"] = imgsz         
    model.overrides["half"] = half_precision  
    model.overrides["verbose"] = False       
    if model_path.endswith('.pt'):
        model.to("cuda")                         
    post_processor = PostProcessWorker(region_points, task, writer, show=show,
                                       show_labels=show_labels, show_conf=show_conf,
                                       line_width=line_width)

    # -----------------------------------------------------------------------
    # Processing loop
    # -----------------------------------------------------------------------
    resize_target = (w, h) if resize_source else None
    cam_reader = ThreadedCamera(cap, resize_target=resize_target)
    mem = MemoryManager(cfg)
    mem.initialize(h, w, "cuda")
    frame_idx = 0
    t_start = time.time()
    disp_frame = None
    if show:
        cv2.namedWindow("WEDA-CV Object Counter", cv2.WINDOW_NORMAL)
    log.info("▶ Processing started … (press 'q' to stop)")

    try:
        while not _shutdown_requested:
            success, frame = cam_reader.read()
            if not success:
                if total_frames > 0:
                    log.info("End of video reached.")
                else:
                    log.warning("End of stream from source.")
                break
            if frame is None:
                time.sleep(0.001)
                continue
            frame_idx += 1
            if frame_idx % vid_stride != 0:
                if writer.isOpened():
                    writer.write(mem.store_output(frame)) 
                if show:
                    cv2.imshow("WEDA-CV Object Counter", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue
            results = model.track(frame, persist=True, verbose=False, tracker=tracker, classes=classes, conf=confidence, iou=iou_threshold)
            post_processor.process(frame, results)
            t2 = time.perf_counter()
            if show:
                latest = post_processor.get_display_frame()
                if latest is not None:
                    disp_frame = latest
                if disp_frame is not None:
                    disp_size = cfg.get("display_size", [1280, 720])
                    if disp_size and isinstance(disp_size, list) and len(disp_size) == 2:
                        show_frame = cv2.resize(disp_frame, tuple(disp_size))
                    else:
                        show_frame = disp_frame
                    cv2.imshow("WEDA-CV Object Counter", show_frame)
                    if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                        log.info("User exit.")
                        break
            if frame_idx % 100 == 0:
                elapsed = time.time() - t_start
                avg_fps = frame_idx / elapsed if elapsed > 0 else 0
                progress = f" ({frame_idx}/{total_frames})" if total_frames > 0 else ""
                log.info("  Processed %d frames%s — %.1f FPS", frame_idx, progress, avg_fps)
    except Exception as exc:
        log.error("Error during processing: %s", exc, exc_info=True)
    finally:
        elapsed = time.time() - t_start
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0
        if 'cam_reader' in locals():
            cam_reader.release()
        else:
            cap.release()
        if 'post_processor' in locals():
            post_processor.release()
        if 'writer' in locals() and writer.isOpened():
            writer.release()
        if 'mem' in locals():
            mem.release()
        if show:
            cv2.destroyAllWindows()
        log.info("═" * 60)
        log.info("  Processing complete")
        log.info("  Frames processed : %d", frame_idx)
        log.info("  Elapsed time     : %.1f s", elapsed)
        log.info("  Average FPS      : %.1f", avg_fps)
        if writer.isOpened() or output_path.exists():
            log.info("  Output saved to  : %s", output_path)
        log.info("═" * 60)
# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="WEDA-CV ObjectCounting Demo — Ultralytics YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Tasks:
  line_count     Count objects crossing a 2-point line
  region_count   Count objects entering a 4-point rectangular zone
  polygon_count  Count objects in an N-point polygon zone

Examples:
  python3 src/object_counter.py
  python3 src/object_counter.py --config config/my_config.yaml
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args()
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║          WEDA-CV ObjectCounting Demo  v1.0.0                 ║")
    log.info("║               Powered by WEDA-CV YOLO                        ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")
    cfg = load_config(args.config)
    run(cfg)
if __name__ == "__main__":
    main()