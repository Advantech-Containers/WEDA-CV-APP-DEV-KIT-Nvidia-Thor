"""Cosmos-Reason2-2B: Physical reasoning for industrial safety/spatial queries.

Handles Tier 2.5 queries: safety assessment, physics understanding,
spatial relationships, PPE detection, path clearance.

Runs async on a dedicated thread. Grounded by YOLO detection facts
injected into the prompt. Chain-of-thought with <think>/<answer> tags.

Fails loudly if model unavailable — no synthetic fallback in production.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Optional

import numpy as np

from app.utils.data_structures import Detection, PipelineConfig, SpatialInfo
from app.utils.logger import get_logger


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting from VLM responses for clean display."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # **bold**
    text = re.sub(r'__(.+?)__', r'\1', text)         # __bold__
    text = re.sub(r'\*(.+?)\*', r'\1', text)         # *italic*
    text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'\1', text)  # _italic_
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # headings
    text = re.sub(r'`(.+?)`', r'\1', text)           # `code`
    text = text.replace('\u2014', '-').replace('\u2013', '-')  # em/en dash
    return text.strip()

logger = get_logger()

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class PhysicalReasoner:
    """Cosmos-Reason2-2B for physical/spatial/safety reasoning."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._enabled = config.enable_physical_reasoning
        self._cooldown = config.physical_cooldown
        self._model = None
        self._processor = None
        self._formatter = None
        self._device = config.device

        self._lock = threading.Lock()
        self._latest_response: str = ""
        self._latest_query_response: str = ""
        self._tok_per_sec: float = 0.0
        self._last_run: float = 0.0
        self._running = False
        self._query_running = False
        self._thread: Optional[threading.Thread] = None
        self._query_thread: Optional[threading.Thread] = None

    def initialize(self) -> None:
        if not self._enabled:
            logger.info("Physical reasoning (Cosmos-Reason2) disabled")
            return

        if not _HAS_TORCH:
            logger.error("Physical reasoning requires PyTorch — not installed. Cosmos-Reason2 unavailable.")
            return

        try:
            from app.modules.dep_resolver import auto_install
            from weda_optim.model_loader import load_physical
            self._model, self._processor, self._formatter = (
                auto_install(load_physical)(self._config)
            )
            if self._model is None:
                logger.error("Cosmos-Reason2 load returned None — unavailable. Check logs above.")
        except Exception as e:
            logger.error("Cosmos-Reason2 load FAILED: %s — unavailable", e)
            self._model = None

    def request_assessment(
        self,
        rgb: np.ndarray,
        detections: list[Detection],
        spatial: list[SpatialInfo],
        query: Optional[str] = None,
    ) -> None:
        """Non-blocking: submit frame for async physical reasoning."""
        now = time.monotonic()
        if now - self._last_run < self._cooldown:
            return
        if self._running:
            return

        self._running = True
        self._last_run = now
        context = self._build_grounding_context(detections, spatial)
        self._thread = threading.Thread(
            target=self._process,
            args=(_copy_frame(rgb), context, query),
            daemon=True,
        )
        self._thread.start()

    def query_sync(
        self,
        rgb: np.ndarray,
        detections: list[Detection],
        spatial: list[SpatialInfo],
        query: str,
    ) -> str:
        """Synchronous query for Tier 2.5 (called by query engine)."""
        if self._model is None or self._processor is None:
            return "[Cosmos-Reason2 not available - check initialization logs]"
        context = self._build_grounding_context(detections, spatial)
        return self._infer(rgb, context, query)

    def request_query(
        self,
        rgb: np.ndarray,
        detections: list[Detection],
        spatial: list[SpatialInfo],
        query: str,
    ) -> None:
        """Non-blocking Tier 2.5 query — result available via latest_query_response."""
        if self._query_running:
            return
        self._query_running = True
        # Use navigation context for path/navigation queries; standard context otherwise
        _nav_words = ("reach", "get to", "navigate", "path to", "route", "direction")
        if any(w in query.lower() for w in _nav_words):
            from app.modules.query_engine import QueryEngine as _QE
            _qtype, _target, _ = _QE.classify(query)
            _target_class = _target or ""
            context = self._build_navigation_context(_target_class, detections, spatial)
            # Append directional instruction to the query
            query = (query + "\n\nPlease provide: (1) which direction to turn "
                     "(left/right/straight), (2) how far to walk, "
                     "(3) whether path is clear or list any obstacles.")
        else:
            context = self._build_grounding_context(detections, spatial)
        self._query_thread = threading.Thread(
            target=self._process_query,
            args=(_copy_frame(rgb), context, query),
            daemon=True,
        )
        self._query_thread.start()

    def _process_query(self, rgb: np.ndarray, context: str, query: str) -> None:
        try:
            if self._model is None or self._processor is None:
                result = "[Cosmos-Reason2 not available]"
            else:
                result = self._infer(rgb, context, query)
            with self._lock:
                self._latest_query_response = result
        except AttributeError as e:
            # Specific guard for ModelOpt quantizer scaffolding being
            # absent on a cache-loaded model (e.g. `'Qwen3VLTextAttention'
            # object has no attribute 'q_bmm_quantizer'`). The cache-hit
            # path in weda_optim/model_loader.py re-applies the quant
            # config to bind these modules; if it still slips through,
            # the user query gets a graceful degraded response instead
            # of a hard crash.
            msg = str(e)
            if "quantizer" in msg.lower():
                logger.error(
                    "Physical query failed — ModelOpt quantizer modules not "
                    "attached. Wipe the physical model cache "
                    "(model_cache/quantized/physical_*) and restart so the "
                    "loader re-quantizes from scratch. (%s)", e,
                )
                with self._lock:
                    self._latest_query_response = (
                        "[Physical reasoner unavailable — quantization "
                        "scaffold mismatch; clear physical model cache]"
                    )
            else:
                logger.error("Physical query error: %s", e)
        except Exception as e:
            logger.error("Physical query error: %s", e)
        finally:
            self._query_running = False

    @property
    def _loaded(self) -> bool:
        return self._model is not None

    @property
    def latest_response(self) -> str:
        with self._lock:
            return self._latest_response

    @property
    def latest_query_response(self) -> str:
        with self._lock:
            return self._latest_query_response

    @property
    def query_running(self) -> bool:
        """True while an async query thread is still running."""
        return self._query_running

    @property
    def vlm_tok_per_sec(self) -> float:
        with self._lock:
            return self._tok_per_sec

    def _record_tok_per_sec(self, gen_tokens: int, elapsed: float) -> None:
        if elapsed > 0 and gen_tokens > 0:
            tps = gen_tokens / elapsed
            with self._lock:
                self._tok_per_sec = 0.3 * tps + 0.7 * self._tok_per_sec if self._tok_per_sec else tps

    def shutdown(self) -> None:
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process(
        self, rgb: np.ndarray, context: str, query: Optional[str]
    ) -> None:
        try:
            if self._model is None or self._processor is None:
                raise RuntimeError("Cosmos-Reason2 not loaded")
            prompt = query or "Assess the safety and spatial layout of this scene."
            result = self._infer(rgb, context, prompt)
            with self._lock:
                self._latest_response = result
        except Exception as e:
            logger.error("Physical reasoning error: %s", e)
        finally:
            self._running = False

    def _infer(self, rgb: np.ndarray, context: str, query: str) -> str:
        from PIL import Image
        image = Image.fromarray(rgb)
        max_tokens = getattr(self._config, "physical_max_tokens", 256)

        directive = (
            "Answer in 1-2 sentences directly addressing the question. "
            "Use ONLY the verified scene data above. If the asked object "
            "is not listed, reply exactly: 'Object not in current scene.' "
            "Do not preface with a scene description."
        )
        framed_query = f"{directive}\n\nQuestion: {query}"
        if self._formatter is not None:
            formatted = self._formatter(context, framed_query)
        else:
            formatted = f"Verified scene data:\n{context}\n\n{framed_query}"

        if isinstance(formatted, list):
            # Qwen3VLProcessor: images go inside message content, NOT as kwarg
            for msg in formatted:
                if isinstance(msg.get("content"), list):
                    for i, item in enumerate(msg["content"]):
                        if isinstance(item, dict) and item.get("type") == "image":
                            msg["content"][i] = {"type": "image", "image": image}
            # Use the canonical two-step recipe to avoid the AWQ Qwen-VL
            # token-count drift (see app.modules.reasoning._build_vlm_inputs).
            from app.modules.reasoning import _build_vlm_inputs
            inputs = _build_vlm_inputs(self._processor, formatted, image)
            if inputs is None:
                inputs = self._processor(text=formatted, images=image, return_tensors="pt")
        else:
            inputs = self._processor(text=formatted, images=image, return_tensors="pt")

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        t0 = time.perf_counter()
        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=max_tokens)
        self._record_tok_per_sec(output.shape[-1] - input_len, time.perf_counter() - t0)
        text = self._processor.batch_decode(output, skip_special_tokens=True)[0]

        for marker in ["<answer>", "ASSISTANT:", "assistant\n"]:
            if marker in text:
                text = text.split(marker)[-1]
                if "</answer>" in text:
                    text = text.split("</answer>")[0]
                text = text.strip()
                break
        return _strip_markdown(text)

    @staticmethod
    def _build_grounding_context(
        detections: list[Detection], spatial: list[SpatialInfo]
    ) -> str:
        if not detections:
            return "No objects detected in scene."
        lines = []
        for i, det in enumerate(detections):
            line = f"- {det.class_name} (id={det.track_id}, conf={det.confidence:.2f})"
            if i < len(spatial) and spatial[i].distance_m > 0:
                si = spatial[i]
                line += f" at {si.distance_m:.2f}m"
                if si.position_3d is not None:
                    line += f" pos=({si.position_3d[0]:.1f},{si.position_3d[1]:.1f},{si.position_3d[2]:.1f})"
                if si.nearest_neighbor_id >= 0:
                    line += f" nearest=#{si.nearest_neighbor_id}@{si.nearest_neighbor_dist:.2f}m"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _build_navigation_context(
        target_class: str,
        detections: list,
        spatial: list,
    ) -> str:
        """Build navigation-focused context: direction, distance, obstacles."""
        import math
        target_det = target_si = None
        for det, si in zip(detections, spatial):
            if target_class and target_class.lower() in det.class_name.lower():
                target_det, target_si = det, si
                break

        if target_det is None or target_si is None or target_si.position_3d is None:
            return PhysicalReasoner._build_grounding_context(detections, spatial)

        tx, ty, tz = (float(target_si.position_3d[0]),
                      float(target_si.position_3d[1]),
                      float(target_si.position_3d[2]))
        angle_deg = math.degrees(math.atan2(tx, max(tz, 0.01)))

        if tx > 0.3:
            direction = f"to your right - turn {abs(angle_deg):.0f} deg right"
        elif tx < -0.3:
            direction = f"to your left - turn {abs(angle_deg):.0f} deg left"
        else:
            direction = "straight ahead - no turn needed"

        lines = [
            "NAVIGATION ANALYSIS:",
            f"Target: {target_det.class_name} #{target_det.track_id}",
            f"  Distance:  {target_si.distance_m:.2f}m",
            f"  Direction: {direction}",
        ]

        obstacles = []
        for det, si in zip(detections, spatial):
            if det.track_id == target_det.track_id:
                continue
            if si.position_3d is None or si.distance_m <= 0:
                continue
            ox = float(si.position_3d[0])
            oz = float(si.position_3d[2])
            if oz < tz and abs(ox - tx) < 0.6:
                side = "left of path" if ox < tx else "right of path"
                obstacles.append(
                    f"  - {det.class_name} #{det.track_id} at {si.distance_m:.2f}m ({side})"
                )

        if obstacles:
            lines.append("Obstacles in path:")
            lines.extend(obstacles)
        else:
            lines.append("Path status: clear - no obstacles between camera and target")

        lines.append("\nAll objects:")
        lines.append(PhysicalReasoner._build_grounding_context(detections, spatial))
        return "\n".join(lines)


def _copy_frame(buf):
    """Copy frame buffer, handling both numpy arrays and torch tensors."""
    if isinstance(buf, np.ndarray):
        return buf.copy()
    try:
        return buf.clone().numpy()
    except Exception:
        return np.array(buf)
