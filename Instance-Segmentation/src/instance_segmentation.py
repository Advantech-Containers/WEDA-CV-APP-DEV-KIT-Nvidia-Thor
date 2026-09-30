"""
WEDA-CV Inference Engine v3.0.0
===============================
Standard engine — DO NOT MODIFY this file.
All behavior is driven by config/config.yaml.

To build a new application:
  1. Copy config/config.yaml → config/my_app.yaml
  2. Edit model.path, source, classes, visualization, output
  3. Run: python src/instance_segmentation.py --config config/my_app.yaml

Architecture:
  config.yaml  →  instance_segmentation.py (engine)  →  visualizer.py (annotation)
                          ↓
                   Ultralytics YOLO .track()
                          ↓
              results.plot() + custom Track ID labels

UMA / Tensor sharing (Jetson):
  - model.track() returns Results with masks/boxes as CUDA tensors
  - results.plot() operates on GPU tensors internally, returns CPU numpy
  - On Jetson UMA, GPU↔CPU transfers are zero-copy (shared DRAM)
  - Async writer decouples MJPG encoding from the inference thread
  - Threaded camera reader decouples V4L2 capture from inference
"""
import argparse
import logging
import os
os.environ["NO_AT_BRIDGE"] = "1"
import queue
import signal
import sys
import threading
import time
import warnings
from pathlib import Path
import cv2
import numpy as np
import yaml
from memory import MemoryManager
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
# ---------------------------------------------------------------------------
# Path anchor — parent of src/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        print(f"[FATAL] Configuration file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    if not cfg:
        print(f"[FATAL] Configuration file is empty: {config_path}", file=sys.stderr)
        sys.exit(1)
    return cfg, config_path
# ---------------------------------------------------------------------------
# Path resolvers — all relative paths resolve against PROJECT_ROOT
# ---------------------------------------------------------------------------
def resolve_path(raw: str, subdirs: list[str] | None = None) -> str:
    """Resolve a path: check as-is, then PROJECT_ROOT, then PROJECT_ROOT/subdirs."""
    if Path(raw).is_file():
        return str(Path(raw).resolve())
    candidate = PROJECT_ROOT / raw
    if candidate.is_file():
        return str(candidate)
    for sub in (subdirs or []):
        candidate = PROJECT_ROOT / sub / Path(raw).name
        if candidate.is_file():
            return str(candidate)
    return raw
def resolve_source(source) -> str | int:
    if isinstance(source, int):
        return source
    if isinstance(source, str) and source.isdigit():
        return int(source)
    if isinstance(source, str) and source.startswith(("rtsp://", "http://", "https://")):
        return source
    return resolve_path(str(source), ["assets"])
def resolve_output(cfg: dict) -> Path:
    out_cfg = cfg.get("output", {})
    output_dir = Path(out_cfg.get("dir", "output"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / out_cfg.get("filename", "segmented_output.avi")
# ---------------------------------------------------------------------------
# Async video writer
# ---------------------------------------------------------------------------
class AsyncVideoWriter:
    def __init__(self, path: str, fourcc: int, fps: int, size: tuple, maxsize: int):
        self._writer = cv2.VideoWriter(path, fourcc, fps, size)
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stopped = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            self._writer.write(item)
    def write(self, frame: np.ndarray):
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
# ---------------------------------------------------------------------------
# Threaded camera reader
# ---------------------------------------------------------------------------
class ThreadedCamera:
    def __init__(self, cap: cv2.VideoCapture, resize_target: tuple = None, maxsize: int = 32, stride: int = 1):
        self._cap = cap
        self._resize = resize_target
        self._stride = stride
        self._frame_count = 0
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
        except Exception:
            self._pool = None
        self._pool_idx = 0
        self._dropped = 0
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        try:
            if self._resize:
                gpu_in = cv2.cuda_GpuMat()
                gpu_out = cv2.cuda_GpuMat()
                gpu_out.create(self._resize[1], self._resize[0], cv2.CV_8UC3)

            while True:
                self._frame_count += 1
                if self._stride > 1 and self._frame_count % self._stride != 0:
                    ret = self._cap.grab()
                    if not ret or self._stopped:
                        break
                    continue

                ret, frame = self._cap.read()
                if not ret or self._stopped:
                    break
                if frame is None or frame.size == 0:
                    continue

                if self._pool is not None:
                    _deadline = time.time() + 0.1
                    while self._q.full() and not self._stopped and time.time() < _deadline:
                        time.sleep(0.001)
                    if self._q.full():
                        self._note_drop()
                        continue
                    pinned_frame = self._pool[self._pool_idx]
                    if self._resize:
                        try:
                            gpu_in.upload(frame)
                            cv2.cuda.resize(gpu_in, self._resize, gpu_out)
                            gpu_out.download(pinned_frame)
                            frame = pinned_frame
                        except Exception:
                            frame = cv2.resize(frame, self._resize)
                            np.copyto(pinned_frame, frame)
                            frame = pinned_frame
                    else:
                        np.copyto(pinned_frame, frame)
                        frame = pinned_frame
                    try:
                        self._q.put((True, frame), timeout=0.1)
                        self._pool_idx = (self._pool_idx + 1) % len(self._pool)
                    except queue.Full:
                        self._note_drop()
                    continue
                else:
                    if self._resize:
                        try:
                            gpu_in.upload(frame)
                            cv2.cuda.resize(gpu_in, self._resize, gpu_out)
                            frame = gpu_out.download()
                        except Exception:
                            frame = cv2.resize(frame, self._resize)

                try:
                    self._q.put((True, frame), timeout=0.1)
                except queue.Full:
                    self._note_drop()
        except Exception as e:
            print(f"[ThreadedCamera] Fatal Error: {e}")
        finally:
            self._put_sentinel()

    def _note_drop(self):
        self._dropped += 1
        if self._dropped % 100 == 0:
            print(f"[ThreadedCamera] {self._dropped} frames dropped (consumer slower than camera)")

    def _put_sentinel(self):
        try:
            self._q.put_nowait((False, None))
        except queue.Full:
            try: self._q.get_nowait()
            except queue.Empty: pass
            try: self._q.put_nowait((False, None))
            except queue.Full: pass

    def read(self):
        try: return self._q.get(timeout=0.01)
        except queue.Empty: return True, None

    def release(self):
        self._stopped = True
        while not self._q.empty():
            try: self._q.get_nowait()
            except queue.Empty: break
        self._t.join(timeout=1.0)
        if not self._t.is_alive():
            self._cap.release()

class PostProcessWorker:
    def __init__(self, mem, writer, show, window_name, **kwargs):
        self._q = queue.Queue(maxsize=32)
        self._display_q = queue.Queue(maxsize=2)
        self._mem = mem
        self._writer = writer
        self._show = show
        self._window_name = window_name
        self._kwargs = kwargs
        self._stopped = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        
    def _run(self):
        from visualizer import get_color, draw_track_labels, draw_object_count
        import torch
        import torch.nn.functional as F
        
        while True:
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                if self._stopped: break
                continue
                
            if item is None or self._stopped: break
            frame, gpu_frame, results, orig_shape, imgsz = item
            
            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                annotated = np.ascontiguousarray(gpu_frame.cpu().numpy())
                if self._kwargs.get("show_count", True):
                    draw_object_count(annotated, 0)
            else:
                scale_x = orig_shape[0] / imgsz
                scale_y = orig_shape[1] / imgsz
                
                boxes_data = boxes.data.clone()
                boxes_data[:, [0, 2]] *= scale_x
                boxes_data[:, [1, 3]] *= scale_y
                
                if self._kwargs.get("show_masks", True) and getattr(results, 'masks', None) is not None:
                    masks = results.masks.data
                    if len(masks) > 0:
                        m = masks.unsqueeze(1)
                        m = F.interpolate(m, size=(orig_shape[1], orig_shape[0]), mode='nearest').squeeze(1)
                        m = m > 0.5
                        
                        track_ids = boxes.id.cpu().numpy().astype(int).flatten() if boxes.id is not None else np.arange(len(boxes))
                        if len(track_ids) != len(m): track_ids = np.arange(len(m))
                        
                        colors = [get_color(int(tid)) for tid in track_ids]
                        colors_t = torch.tensor(colors, device=gpu_frame.device, dtype=torch.float32)
                        
                        alpha = self._kwargs.get("mask_alpha", 0.35)
                        for i in range(len(m)):
                            mask_idx = m[i]
                            if mask_idx.any():
                                c_val = colors_t[i] * alpha
                                bg = gpu_frame[mask_idx].float() * (1.0 - alpha)
                                gpu_frame[mask_idx] = (bg + c_val).byte()
                                
                annotated = np.ascontiguousarray(gpu_frame.cpu().numpy())
                xyxy = boxes_data[:, :4].cpu().numpy()
                track_ids = boxes.id.cpu().numpy().astype(int).flatten() if boxes.id is not None else np.arange(len(boxes_data))
                
                if self._kwargs.get("show_boxes", True):
                    for i in range(len(xyxy)):
                        color = get_color(int(track_ids[i]))
                        cv2.rectangle(annotated, (int(xyxy[i][0]), int(xyxy[i][1])), (int(xyxy[i][2]), int(xyxy[i][3])), color, self._kwargs.get("line_width", 2))
                        
                if self._kwargs.get("show_labels", True):
                    confs = boxes_data[:, 4].cpu().numpy().flatten()
                    draw_track_labels(annotated, xyxy, track_ids, confs, self._kwargs.get("show_conf", False), self._kwargs.get("label_font_scale", 0.6))
                    
                if self._kwargs.get("show_count", True):
                    draw_object_count(annotated, len(boxes_data))
            
            if self._writer and self._writer.isOpened():
                if self._mem: self._writer.write(self._mem.store_output(annotated))
                else: self._writer.write(annotated)
                    
            if self._show:
                try: self._display_q.put_nowait(annotated)
                except queue.Full: pass

    def get_display_frame(self):
        try: return self._display_q.get_nowait()
        except queue.Empty: return None

    def process(self, frame, gpu_frame, results, orig_shape, imgsz):
        if not self._stopped:
            try: self._q.put_nowait((frame, gpu_frame, results, orig_shape, imgsz))
            except queue.Full: pass

    def release(self):
        self._stopped = True
        while not self._q.empty():
            try: self._q.get_nowait()
            except queue.Empty: break
        try: self._q.put_nowait(None)
        except queue.Full: pass
        self._t.join(timeout=2.0)
# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------
_shutdown_requested = False
def _handle_signal(signum, _frame):
    global _shutdown_requested
    _shutdown_requested = True
signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)
# ---------------------------------------------------------------------------
# Open video source — driven entirely by config
# ---------------------------------------------------------------------------
def open_source(source, cam_cfg: dict, log) -> cv2.VideoCapture:
    """Open VideoCapture with config-driven backend and settings."""
    is_camera = isinstance(source, int)
    if is_camera:
        backend_name = cam_cfg.get("backend", "v4l2").lower()
        backend_map = {
            "v4l2": cv2.CAP_V4L2,
            "gstreamer": cv2.CAP_GSTREAMER,
            "auto": cv2.CAP_ANY,
        }
        backend = backend_map.get(backend_name, cv2.CAP_ANY)
        log.info("Source: camera %d (backend: %s)", source, backend_name)
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened():
            fourcc = cam_cfg.get("fourcc", "MJPG")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FPS, cam_cfg.get("fps", 30))
            buf = cam_cfg.get("buffer_size", 1)
            if buf > 0:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, buf)
            log.info("Camera config: fourcc=%s, fps=%d, buffer=%d", fourcc, cam_cfg.get("fps", 30), buf)
    else:
        log.info("Source: %s", source)
        cap = cv2.VideoCapture(source)
    return cap
# ---------------------------------------------------------------------------
# Create video writer — driven by config
# ---------------------------------------------------------------------------
def create_writer(output_path: Path, fps: int, size: tuple, out_cfg: dict, log):
    """Create sync or async writer based on config."""
    if not out_cfg.get("enabled", True):
        log.info("Video recording: disabled")
        return None
    codec = out_cfg.get("codec", "MJPG")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    use_async = out_cfg.get("async_writer", True)
    queue_size = out_cfg.get("queue_size", 64)
    if use_async:
        writer = AsyncVideoWriter(str(output_path), fourcc, fps, size, maxsize=queue_size)
        log.info("Writer: async (codec=%s, queue=%d) → %s", codec, queue_size, output_path)
    else:
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
        log.info("Writer: sync (codec=%s) → %s", codec, output_path)
    if not writer.isOpened():
        log.error("Failed to open video writer: %s", output_path)
        sys.exit(1)
    return writer
# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def run(cfg: dict, log):
    from ultralytics import YOLO
    perf_cfg = cfg.get("performance", {})
    if perf_cfg.get("cuda_launch_blocking", False):
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    model_cfg = cfg.get("model", {})
    model_path = resolve_path(model_cfg.get("path", "yolo11n-seg.pt"), ["assets"])
    device = model_cfg.get("device", "cuda")
    half = model_cfg.get("half", False)
    imgsz = model_cfg.get("imgsz", 640)
    vid_stride = model_cfg.get("vid_stride", 1)
    retina_masks = model_cfg.get("retina_masks", False)
    confidence = cfg.get("confidence", 0.25)
    iou_threshold = cfg.get("iou_threshold", 0.7)
    classes = cfg.get("classes")
    tracker = cfg.get("tracker", "botsort.yaml")
    viz = cfg.get("visualization", {})
    show = viz.get("show", False)
    window_name = viz.get("window_name", "WEDA-CV")
    display_size = viz.get("display_size")
    
    if show:
        if display_size and isinstance(display_size, list) and len(display_size) == 2:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, display_size[0], display_size[1])
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    show_labels = viz.get("show_labels", True)
    show_conf = viz.get("show_conf", False)
    show_boxes = viz.get("show_boxes", True)
    show_masks = viz.get("show_masks", True)
    show_count = viz.get("show_count", True)
    line_width = viz.get("line_width", 2)
    mask_alpha = viz.get("mask_alpha", 0.35)
    label_font_scale = viz.get("label_font_scale", 0.6)
    dbg = cfg.get("debug", {})
    debug_timing = dbg.get("timing", False)
    timing_interval = dbg.get("timing_interval", 100)
    source = resolve_source(cfg.get("source", 0))
    cam_cfg = cfg.get("camera", {})
    cap = open_source(source, cam_cfg, log)
    if not cap.isOpened():
        log.error("Cannot open video source: %s", source)
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("Video: %dx%d @ %d FPS (%s frames)", w, h, fps, total_frames if total_frames > 0 else "live")
    resize_source = cfg.get("resize_source")
    if resize_source and isinstance(resize_source, list) and len(resize_source) == 2:
        w, h = int(resize_source[0]), int(resize_source[1])
        log.info("Scaling input frames to: %dx%d for processing", w, h)
    mem = MemoryManager(cfg)
    mem.initialize(h, w, device)
    mem.log_status()
    out_cfg = cfg.get("output", {})
    output_path = resolve_output(cfg)
    writer = create_writer(output_path, fps, (w, h), out_cfg, log)
    log.info("Loading model: %s → %s", model_path, device)
    model = YOLO(model_path)
    model.overrides["half"] = half
    model.overrides["imgsz"] = imgsz
    model.overrides["verbose"] = False
    if model_path.endswith(".pt"):
        model.to(device)
    log.info("Model ready — half: %s, imgsz: %d, retina_masks: %s", half, imgsz, retina_masks)
    if perf_cfg.get("warmup", True):
        log.info("Warmup …")
        _dummy = np.zeros((h, w, 3), dtype=np.uint8)
        model.track(
            _dummy, persist=True, tracker=tracker,
            conf=confidence, iou=iou_threshold,
            imgsz=imgsz, half=half,
            retina_masks=retina_masks, verbose=False,
            device=device,
        )
        del _dummy
    cam_reader = None
    vid_stride = model_cfg.get("vid_stride", 1)
    resize_target = (w, h) if resize_source else None
    if cam_cfg.get("threaded", True):
        cam_reader = ThreadedCamera(cap, resize_target=resize_target, stride=vid_stride)
        log.info("Threaded camera reader: enabled (GPU CUDA Resizer, stride=%d)", vid_stride)
        
    post_processor = PostProcessWorker(
        mem=mem, writer=writer, show=show, window_name=window_name,
        show_boxes=show_boxes, show_masks=show_masks,
        show_labels=show_labels, show_conf=show_conf,
        show_count=show_count, line_width=line_width,
        mask_alpha=mask_alpha, label_font_scale=label_font_scale
    )
    t_trk = t_ann = t_wrt = t_dsp = 0.0
    frame_idx = 0
    t_start = time.time()
    log.info("▶ Processing started … (press 'q' to stop)")

    try:
        while not _shutdown_requested:
            if cam_reader:
                ok, frame = cam_reader.read()
                if not ok:
                    if total_frames > 0:
                        log.info("End of video at frame %d.", frame_idx)
                    else:
                        log.warning("End of stream from source.")
                    break
                if frame is None:
                    time.sleep(0.001)
                    continue
            else:
                ok, frame = cap.read()
                if not ok:
                    if total_frames > 0:
                        log.info("End of video at frame %d.", frame_idx)
                    else:
                        log.warning("Empty frame from source.")
                    break

            if not cam_reader and resize_source and isinstance(resize_source, list) and len(resize_source) == 2:
                try:
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(frame)
                    gpu_frame = cv2.cuda.resize(gpu_frame, (w, h))
                    frame = gpu_frame.download()
                except Exception:
                    frame = cv2.resize(frame, (w, h))

            frame_idx += 1
            if frame_idx % vid_stride != 0:
                if writer:
                    writer.write(frame)
                if show:
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue                
            t0 = time.perf_counter()
            import torch
            import torch.nn.functional as F
            orig_h, orig_w = frame.shape[:2]
            gpu_frame = torch.from_numpy(frame).to(device)
            infer_tensor = gpu_frame[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).half() / 255.0
            infer_tensor = F.interpolate(infer_tensor, size=(imgsz, imgsz), mode='bilinear', align_corners=False)
            results = model.track(
                infer_tensor, persist=True, tracker=tracker,
                conf=confidence, iou=iou_threshold,
                classes=classes, imgsz=imgsz,
                half=half, retina_masks=retina_masks,
                verbose=False,
                device=device,
            )
            t1 = time.perf_counter()
            post_processor.process(frame, gpu_frame, results[0], (orig_w, orig_h), imgsz)
            t2 = time.perf_counter()
            if show:
                disp_frame = post_processor.get_display_frame()
                if disp_frame is not None:
                    cv2.imshow(window_name, disp_frame)
                    if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                        log.info("User exit.")
                        break
            
            t_trk += (t1 - t0)
            t_ann += (t2 - t1)
            t_wrt += 0
            t_dsp += 0

            if frame_idx % timing_interval == 0:
                elapsed = time.time() - t_start
                avg_fps = frame_idx / elapsed if elapsed > 0 else 0
                progress = f" ({frame_idx}/{total_frames})" if total_frames > 0 else ""
                log.info("  Processed %d frames%s — %.1f FPS avg", frame_idx, progress, avg_fps)
                if debug_timing:
                    n = min(frame_idx, timing_interval)
                    log.info(
                        "  ┣ Track: %.1fms | Annotate: %.1fms | Write: %.1fms | Display: %.1fms",
                        (t_trk / n) * 1000, (t_ann / n) * 1000,
                        (t_wrt / n) * 1000, (t_dsp / n) * 1000,
                    )
                    t_trk = t_ann = t_wrt = t_dsp = 0.0

    except Exception as exc:
        log.error("Fatal: %s", exc, exc_info=True)
    finally:
        elapsed = time.time() - t_start
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0
        if cam_reader:
            cam_reader.release()
        else:
            cap.release()   
        if 'post_processor' in locals():
            post_processor.release()
        if writer:
            writer.release()
        mem.release()
        if show:
            cv2.destroyAllWindows()
        log.info("═" * 60)
        log.info("  Processing complete")
        log.info("  Frames processed : %d", frame_idx)
        log.info("  Elapsed time     : %.1f s", elapsed)
        log.info("  Average FPS      : %.1f", avg_fps)
        if out_cfg.get("enabled", True) and output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            log.info("  Output saved to  : %s (%.1f MB)", output_path, size_mb)
        log.info("═" * 60)
# ---------------------------------------------------------------------------
# CLI — the only entry point
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = str(PROJECT_ROOT / "config" / "config.yaml")
def main():
    parser = argparse.ArgumentParser(description="WEDA-CV Inference Engine")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config")
    args = parser.parse_args()
    cfg, config_path = load_config(args.config)
    dbg = cfg.get("debug", {})
    log_level = getattr(logging, dbg.get("log_level", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    log = logging.getLogger("WEDA-CV")
    app = cfg.get("app", {})
    app_name = app.get("name", "WEDA-CV")
    app_version = app.get("version", "0.0.0")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║   %-54s ║", f"{app_name} v{app_version}")
    log.info("║   %-54s ║", "Powered by WEDA-CV YOLO")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info("Config: %s", config_path)
    run(cfg, log)
if __name__ == "__main__":
    main()