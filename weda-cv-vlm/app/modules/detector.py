"""YOLO26 detector with direct TensorRT inference (no Ultralytics wrapper).

Inference priority:
  1. Direct TRT engine via tensorrt Python API (fastest — 2.5ms)
  2. Ultralytics YOLO() wrapper (fallback — 8ms, handles export)
  3. ONNX Runtime GPU
  4. Mock mode (empty detections)

The direct TRT path bypasses Ultralytics' Python preprocessing,
result parsing, and tensor management — matching DeepStream nvinfer
performance in pure Python.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from weda_optim import BufferPool
from weda_optim.vic_offload import VicResizer, is_vic_available
from app.utils.data_structures import Detection, PipelineConfig
from app.utils.logger import get_logger

logger = get_logger()


@contextlib.contextmanager
def _suppress_stderr():
    """Redirect stderr to /dev/null to silence TRT C++ logs.

    Uses ctypes.fflush(None) to drain C stdio buffers before
    redirect and before restore — prevents buffered TRT output
    from leaking to the real stderr on fd restore.
    """
    import ctypes as _ct
    try:
        _ct.CDLL(None).fflush(None)
    except Exception:
        pass
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        try:
            _ct.CDLL(None).fflush(None)   # flush TRT output to /dev/null
        except Exception:
            pass
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)


try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False

try:
    import tensorrt as trt
    _HAS_TRT = True
    # Register global TRT logger at ERROR level BEFORE any operations.
    # TRT uses the first-registered logger — later loggers are ignored.
    _TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    try:
        trt.init_libnvinfer_plugins(_TRT_LOGGER, "")
    except Exception:
        pass
except ImportError:
    _HAS_TRT = False

try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class Detector:
    def __init__(self, config: PipelineConfig, pool: Optional[BufferPool] = None) -> None:
        self._config = config
        self._pool = pool
        self._model = None          # Ultralytics YOLO (fallback)
        self._trt_context = None     # Direct TRT execution context
        self._trt_runtime = None
        self._trt_engine = None
        self._trt_stream = None      # CUDA stream for async
        self._trt_bindings = None    # Pre-allocated GPU I/O buffers
        self._ort_session = None
        self._backend: str = "mock"
        sz = getattr(config, "det_input_size", 640)
        self._input_size = (sz, sz)
        self._class_names: dict[int, str] = {}
        self._is_yolo26 = "yolo26" in config.yolo_model.lower()
        # VIC hardware resize (if available and enabled)
        self._vic_resizer: Optional[VicResizer] = None
        if getattr(config, "enable_vic_offload", False) and is_vic_available():
            self._vic_resizer = VicResizer(sz, sz)
            logger.info("VIC hardware resize enabled for detection input")

    def initialize(self) -> None:
        model_path = Path(self._config.yolo_model)
        engine_path = model_path.with_suffix(".engine")

        # Pre-check: use pre-exported TRT engine from persistent cache if available
        _direct_trt_tried: set[str] = set()  # track paths we already tried direct TRT on
        if getattr(self._config, "use_tensorrt", False) and getattr(self._config, "quant_cache_dir", None):
            trt_cache = Path(self._config.quant_cache_dir) / "yolo" / model_path.with_suffix(".engine").name
            if trt_cache.exists():
                logger.info("[CACHE HIT] Loading pre-exported TRT engine: %s", trt_cache)
                if _HAS_TRT and _HAS_TORCH and self._init_direct_trt(trt_cache):
                    if not self._class_names:
                        self._class_names = _COCO_NAMES.copy()
                    self._backend = "tensorrt_direct"
                    logger.info("Direct TRT engine loaded from cache (%d classes)", len(self._class_names))
                    return
                _direct_trt_tried.add(str(trt_cache.resolve()))
                if _HAS_ULTRALYTICS:
                    try:
                        with _suppress_stderr():
                            self._model = YOLO(str(trt_cache), verbose=False)
                        self._class_names = self._model.names or {}
                        self._backend = "tensorrt"
                        logger.info("Ultralytics TRT wrapper loaded from cache: %s", trt_cache)
                        return
                    except Exception as e:
                        logger.warning("TRT cache load failed: %s — re-exporting", e)

        # Step 0: Auto-download model if not present (Ultralytics handles this)
        if _HAS_ULTRALYTICS and not model_path.exists() and not engine_path.exists():
            logger.info("Downloading YOLO model: %s (first run)...", self._config.yolo_model)
            try:
                tmp_model = YOLO(str(self._config.yolo_model))
                self._class_names = tmp_model.names or {}
                # If TRT requested, export engine now
                if self._config.use_tensorrt:
                    logger.info("Exporting TensorRT engine (first run, ~90s)...")
                    tmp_model.export(format="engine", half=True, device=self._config.device)
                self._model = tmp_model
                if engine_path.exists():
                    self._model = YOLO(str(engine_path))
                    self._class_names = self._model.names or {}
                    self._backend = "tensorrt"
                    logger.info("TensorRT engine ready: %s", engine_path)
                else:
                    self._backend = "ultralytics"
                    logger.info("Using Ultralytics PyTorch: %s", model_path)
                return
            except Exception as e:
                logger.warning("YOLO auto-download failed: %s", e)

        # Step 1: Ensure TRT engine exists (export via Ultralytics if needed)
        if self._config.use_tensorrt and not engine_path.exists():
            if _HAS_ULTRALYTICS and model_path.exists():
                logger.info("Exporting TensorRT engine from %s (first run only)...", model_path)
                try:
                    tmp_model = YOLO(str(model_path))
                    tmp_model.export(format="engine", half=True, device=self._config.device)
                    self._class_names = tmp_model.names or {}
                    del tmp_model
                except Exception as e:
                    logger.warning("TensorRT export failed: %s", e)

        # Step 2: Try direct TRT inference (fastest path — no Ultralytics wrapper)
        #         Skip if we already tried this exact engine file above
        if self._config.use_tensorrt and _HAS_TRT and _HAS_TORCH and engine_path.exists() \
                and str(engine_path.resolve()) not in _direct_trt_tried:
            if self._init_direct_trt(engine_path):
                # Get class names from Ultralytics if we don't have them yet
                if not self._class_names and _HAS_ULTRALYTICS and model_path.exists():
                    try:
                        tmp = YOLO(str(model_path))
                        self._class_names = tmp.names or {}
                        del tmp
                    except Exception:
                        self._class_names = {i: str(i) for i in range(80)}
                elif not self._class_names:
                    self._class_names = _COCO_NAMES.copy()
                self._backend = "tensorrt_direct"
                logger.info("Direct TRT engine loaded: %s (%d classes, %dx%d)",
                            engine_path, len(self._class_names), *self._input_size)
                return

        # Step 3: Fallback to Ultralytics TRT wrapper
        if self._config.use_tensorrt and _HAS_ULTRALYTICS and engine_path.exists():
            try:
                with _suppress_stderr():
                    self._model = YOLO(str(engine_path), verbose=False)
                self._class_names = self._model.names or {}
                self._backend = "tensorrt"
                logger.info("Ultralytics TRT wrapper loaded: %s", engine_path)
                return
            except Exception as e:
                logger.warning("Ultralytics TRT load failed: %s", e)

        # Step 4: Ultralytics PyTorch
        if _HAS_ULTRALYTICS and model_path.exists():
            self._model = YOLO(str(model_path))
            self._class_names = self._model.names or {}
            self._backend = "ultralytics"
            logger.info("Using Ultralytics PyTorch: %s", model_path)
            return

        # Step 5: ONNX Runtime
        if _HAS_ORT:
            onnx_path = model_path.with_suffix(".onnx")
            if onnx_path.exists():
                try:
                    self._ort_session = ort.InferenceSession(
                        str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                    )
                    self._backend = "onnxruntime"
                    logger.info("ONNX Runtime loaded: %s", onnx_path)
                    return
                except Exception as e:
                    logger.warning("ONNX Runtime failed: %s", e)

        # Last resort: try Ultralytics auto-download one more time
        if _HAS_ULTRALYTICS:
            logger.info("Attempting Ultralytics auto-download: %s", self._config.yolo_model)
            try:
                self._model = YOLO(str(self._config.yolo_model))
                self._class_names = self._model.names or {}
                self._backend = "ultralytics"
                logger.info("YOLO downloaded and loaded: %s", self._config.yolo_model)
                return
            except Exception as e:
                logger.error("YOLO download failed: %s", e)

        self._backend = "mock"
        logger.error(
            "DETECTION UNAVAILABLE — no model found or downloadable. "
            "Place %s in working directory or install ultralytics (pip install ultralytics).",
            self._config.yolo_model,
        )

    # ==================================================================
    # Direct TRT Engine — bypasses Ultralytics for 2.5ms inference
    # ==================================================================

    def _init_direct_trt(self, engine_path: Path) -> bool:
        """Load TRT engine and pre-allocate GPU I/O buffers."""
        try:
            trt_logger = trt.Logger(trt.Logger.ERROR)
            self._trt_runtime = trt.Runtime(trt_logger)

            with _suppress_stderr(), open(engine_path, "rb") as f:
                file_bytes = f.read()

            engine_bytes = file_bytes
            if len(file_bytes) >= 4:
                # Check for standard TRT magic tag (TRT\x00 or ftrt in TRT 10 little-endian)
                if not (file_bytes.startswith(b"TRT\x00") or file_bytes.startswith(b"ftrt")):
                    # Read length prefix for metadata
                    meta_len = int.from_bytes(file_bytes[:4], byteorder="little")
                    if len(file_bytes) >= meta_len + 8:
                        candidate_bytes = file_bytes[4 + meta_len :]
                        if candidate_bytes.startswith(b"TRT\x00") or candidate_bytes.startswith(b"ftrt"):
                            engine_bytes = candidate_bytes
                            logger.info("Detected Ultralytics metadata header (length %d), stripped successfully.", meta_len)

            with _suppress_stderr():
                self._trt_engine = self._trt_runtime.deserialize_cuda_engine(engine_bytes)

            if self._trt_engine is None:
                return False

            with _suppress_stderr():
                self._trt_context = self._trt_engine.create_execution_context()
            self._trt_stream = torch.cuda.Stream()

            # Pre-allocate I/O tensors
            self._trt_inputs = {}
            self._trt_outputs = {}

            for i in range(self._trt_engine.num_io_tensors):
                name = self._trt_engine.get_tensor_name(i)
                original_shape = self._trt_engine.get_tensor_shape(name)
                
                # Check if the shape has dynamic dimensions
                is_dynamic = any(s == -1 for s in original_shape)
                shape = [1 if s == -1 else s for s in original_shape]
                
                # Only set input shape for dynamic inputs to avoid TensorRT 10 static shape exception
                if is_dynamic and self._trt_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self._trt_context.set_input_shape(name, shape)

                # Map TensorRT DataType to PyTorch DataType
                trt_dtype = self._trt_engine.get_tensor_dtype(name)
                if trt_dtype == trt.DataType.FLOAT:
                    torch_dtype = torch.float32
                elif trt_dtype == trt.DataType.HALF:
                    torch_dtype = torch.float16
                elif trt_dtype == trt.DataType.INT32:
                    torch_dtype = torch.int32
                elif trt_dtype == trt.DataType.INT8:
                    torch_dtype = torch.int8
                elif trt_dtype == trt.DataType.UINT8:
                    torch_dtype = torch.uint8
                elif trt_dtype == trt.DataType.BOOL:
                    torch_dtype = torch.bool
                elif hasattr(trt.DataType, "BF16") and trt_dtype == trt.DataType.BF16:
                    torch_dtype = torch.bfloat16
                else:
                    try:
                        torch_dtype = torch.from_numpy(np.zeros((1,), dtype=trt.nptype(trt_dtype))).dtype
                    except Exception:
                        torch_dtype = torch.float32

                tensor = torch.zeros(shape, dtype=torch_dtype, device=self._config.device)

                if self._trt_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self._trt_inputs[name] = tensor
                else:
                    self._trt_outputs[name] = tensor

                self._trt_context.set_tensor_address(name, tensor.data_ptr())

            logger.debug("TRT I/O: inputs=%s, outputs=%s",
                        list(self._trt_inputs.keys()), list(self._trt_outputs.keys()))
            return True

        except Exception as e:
            logger.warning("Direct TRT init failed: %s — falling back to Ultralytics", e)
            self._trt_context = None
            self._trt_engine = None
            return False

    def _direct_trt_detect(self, rgb: np.ndarray, conf: float) -> list[Detection]:
        """Run inference directly on TRT engine — no Ultralytics overhead."""
        h, w = rgb.shape[:2]
        sz = self._input_size[0]

        # Preprocess: resize (VIC hardware or cv2) + normalize + HWC→CHW + batch
        if self._vic_resizer:
            resized = self._vic_resizer.resize(rgb)
        else:
            resized = cv2.resize(rgb, self._input_size)
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC → CHW
        blob = np.expand_dims(blob, 0)   # Add batch dim
        blob = np.ascontiguousarray(blob)

        # Copy input to pre-allocated GPU tensor
        input_name = list(self._trt_inputs.keys())[0]
        input_tensor = self._trt_inputs[input_name]
        input_tensor.copy_(torch.from_numpy(blob).to(device=self._config.device, dtype=input_tensor.dtype))

        # Execute
        self._trt_context.execute_async_v3(stream_handle=self._trt_stream.cuda_stream)
        self._trt_stream.synchronize()

        # Parse outputs
        detections: list[Detection] = []
        output_names = list(self._trt_outputs.keys())

        det_output = self._trt_outputs[output_names[0]].float().cpu().numpy()

        if det_output.ndim == 3:
            det_output = det_output[0]

        is_standard = det_output.shape[0] == 8400 or det_output.shape[1] == 8400

        if self._is_yolo26 and not is_standard and det_output.shape[-1] >= 6:
            return self._parse_yolo26_output(det_output, h, w, sz, conf)

        return self._parse_standard_output(det_output, h, w, sz, conf)

    def _parse_yolo26_output(
        self, output: np.ndarray, img_h: int, img_w: int, sz: int, conf: float
    ) -> list[Detection]:
        """Parse YOLO26 NMS-free output: (N, 38) = bbox(4) + score(1) + cls(1) + mask(32)."""
        detections: list[Detection] = []
        sx, sy = img_w / sz, img_h / sz

        for row in output:
            # YOLO26 end-to-end: first 4 = xyxy, 5th = score, 6th = class
            if row.shape[0] >= 6:
                score = float(row[4])
                if score < conf:
                    continue
                x1, y1, x2, y2 = row[0] * sx, row[1] * sy, row[2] * sx, row[3] * sy
                class_id = int(row[5])
            else:
                continue

            detections.append(Detection(
                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                confidence=score,
                class_id=class_id,
                class_name=self._class_names.get(class_id, str(class_id)),
            ))

        return detections

    def _parse_standard_output(
        self, output: np.ndarray, img_h: int, img_w: int, sz: int, conf: float
    ) -> list[Detection]:
        """Parse standard YOLO output: (84, 8400) or (8400, 84) and apply NMS."""
        if output.shape[0] < output.shape[1]:
            output = output.T  # Transpose to (N, 84)

        bboxes = []
        scores_list = []
        class_ids = []
        sx, sy = img_w / sz, img_h / sz

        for row in output:
            scores = row[4:]
            class_id = int(np.argmax(scores))
            score = float(scores[class_id])
            if score < conf:
                continue
            cx, cy, bw, bh = row[:4]
            x1 = (cx - bw / 2) * sx
            y1 = (cy - bh / 2) * sy
            x2 = (cx + bw / 2) * sx
            y2 = (cy + bh / 2) * sy
            bboxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            scores_list.append(score)
            class_ids.append(class_id)

        detections: list[Detection] = []
        if bboxes:
            indices = cv2.dnn.NMSBoxes(bboxes, scores_list, conf, 0.45)
            for i in indices:
                idx = i if isinstance(i, int) else i[0]
                x, y, w, h = bboxes[idx]
                class_id = class_ids[idx]
                detections.append(Detection(
                    bbox=np.array([x, y, x + w, y + h], dtype=np.float32),
                    confidence=scores_list[idx],
                    class_id=class_id,
                    class_name=self._class_names.get(class_id, str(class_id)),
                ))

        return detections

    # ==================================================================
    # Public API
    # ==================================================================

    def detect(self, rgb: np.ndarray, confidence: Optional[float] = None) -> list[Detection]:
        conf = confidence or self._config.det_conf
        rgb_np = _as_numpy(rgb)

        if self._backend == "mock":
            return []

        if self._backend == "tensorrt_direct":
            return self._direct_trt_detect(rgb_np, conf)

        if self._backend in ("ultralytics", "tensorrt") and self._model is not None:
            return self._ultralytics_detect(rgb_np, conf)

        if self._backend == "onnxruntime" and self._ort_session is not None:
            return self._ort_detect(rgb_np, conf)

        return []

    def shutdown(self) -> None:
        self._model = None
        self._ort_session = None
        if self._trt_context:
            del self._trt_context
            self._trt_context = None
        if self._trt_engine:
            del self._trt_engine
            self._trt_engine = None
        self._trt_inputs = None
        self._trt_outputs = None

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names

    @property
    def is_yolo26(self) -> bool:
        return self._is_yolo26

    # ------------------------------------------------------------------
    # Ultralytics fallback
    # ------------------------------------------------------------------

    def _ultralytics_detect(self, rgb: np.ndarray, conf: float) -> list[Detection]:
        results = self._model(rgb, conf=conf, verbose=False)
        detections: list[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(np.float32)
                det = Detection(
                    bbox=bbox,
                    confidence=float(boxes.conf[i]),
                    class_id=int(boxes.cls[i]),
                    class_name=self._class_names.get(int(boxes.cls[i]), "unknown"),
                )
                if hasattr(r, "masks") and r.masks is not None:
                    try:
                        mask_np = r.masks.data[i].cpu().numpy()  # float32 0-1, inference res
                        orig_h, orig_w = rgb.shape[:2]
                        mask_resized = cv2.resize(mask_np, (orig_w, orig_h),
                                                  interpolation=cv2.INTER_LINEAR)
                        det.mask = (mask_resized > 0.5).astype(np.uint8) * 255
                    except (IndexError, AttributeError):
                        pass
                detections.append(det)
        return detections

    def _ort_detect(self, rgb: np.ndarray, conf: float) -> list[Detection]:
        blob = cv2.dnn.blobFromImage(rgb, 1.0 / 255.0, self._input_size, swapRB=True)
        input_name = self._ort_session.get_inputs()[0].name
        outputs = self._ort_session.run(None, {input_name: blob})
        return self._parse_standard_output(outputs[0] if outputs[0].ndim == 2 else outputs[0][0],
                                           rgb.shape[0], rgb.shape[1],
                                           self._input_size[0], conf)


# ── COCO class names (fallback when Ultralytics not available) ──
_COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush",
}


def _as_numpy(buf) -> np.ndarray:
    if isinstance(buf, np.ndarray):
        return buf
    try:
        return buf.numpy()
    except Exception:
        return np.asarray(buf)
