"""Scene understanding via VLM with per-object visual caching.

Tier 3 engine: Qwen2.5-VL-72B-Instruct-AWQ.
Background cycle: crops each detected object, asks VLM for visual
description (color, material, logos), caches by track_id.

Scene change driven: only fires when Cosmos-Embed1 detects scene change,
or when a novel Tier 3 query arrives.

Grounding (Option D):
  - Every VLM call receives a sensor-verified JSON block with all
    available metrics: bbox_norm, confidence, distance_m, position_3d,
    physical_size, segmentation_coverage, nearest_object.
  - The image passed to Qwen is annotated: segmentation mask overlays,
    bounding boxes, and distance/class labels drawn on a copy of the
    frame.  This runs entirely on the background thread — zero FPS cost.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Optional

import cv2
import numpy as np

from app.utils.data_structures import Detection, PipelineConfig, SpatialInfo
from app.utils.logger import get_logger


# Image-size guards for Qwen3-VL chat templating. ``smart_resize``
# clamps incoming PIL/np images to a multiple of (patch_size *
# spatial_merge_size) = 28 px on each side, with a token budget bounded
# by [min_pixels, max_pixels] tokens-equivalent. Picking sane bounds
# here keeps tiny crops (e.g. a 30×40 px detection) from triggering
# inconsistent placeholder expansion in the chat template (the
# "tensor a (4304) must match tensor b (4352)" symptom on AWQ
# Qwen3-VL-30B-A3B), and keeps huge crops from blowing up KV cache.
#
# Defaults from the Qwen2.5-VL / Qwen3-VL model cards:
#   min_pixels = 256 * 28 * 28 = 200,704   (≈ 256 tokens)
#   max_pixels = 1280 * 28 * 28 = 1,003,520 (≈ 1280 tokens)
# We use a slightly tighter ceiling because we feed object crops, not
# full frames, and capping at ~768 tokens keeps the per-call cost
# bounded under UMA pressure.
_QWEN_MIN_PIXELS = 256 * 28 * 28
_QWEN_MAX_PIXELS = 768 * 28 * 28


def _build_vlm_inputs(processor, messages, image):
    """Tokenize a Qwen-VL chat-template message list using the canonical
    two-step recipe documented on every Qwen-VL model card:

        1. ``apply_chat_template(tokenize=False)`` to get the text prompt
           with ``<|image_pad|>`` placeholders.
        2. ``process_vision_info(messages)`` (from ``qwen_vl_utils``) to
           extract the image / video tensors with ``smart_resize``
           applied consistently with the encoder.
        3. Call the processor on text + images together — the processor
           expands placeholders to the *exact* count the visual encoder
           will produce, so the merged sequence length matches the
           hidden-state length.

    The single-step ``apply_chat_template(tokenize=True, return_dict=True,
    return_tensors="pt")`` shortcut introduced in transformers ≥ 4.49
    expands placeholders against an internal heuristic that drifts on
    AWQ-quantized Qwen3-VL checkpoints — that's the
    ``RuntimeError: The size of tensor a (4304) must match the size of
    tensor b (4352) at non-singleton dimension 0`` we hit in the
    background auto-caption path. Falling back to the canonical recipe
    eliminates the drift.

    Falls through to the LLaVA-style ``processor(text=..., images=...)``
    when ``apply_chat_template`` isn't available (non-Qwen processors).
    """
    # Fast path: LLaVA-style chat template doesn't return list-of-dicts;
    # detect by trying the chat template and falling through on TypeError.
    try:
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except (TypeError, AttributeError):
        return None  # Caller handles the LLaVA fallback.

    # Inject min/max pixels clamp on every image entry so smart_resize
    # is bounded.
    for _msg in messages:
        _content = _msg.get("content") if isinstance(_msg, dict) else None
        if not isinstance(_content, list):
            continue
        for _item in _content:
            if isinstance(_item, dict) and _item.get("type") == "image":
                _item.setdefault("min_pixels", _QWEN_MIN_PIXELS)
                _item.setdefault("max_pixels", _QWEN_MAX_PIXELS)

    # process_vision_info is shipped by qwen_vl_utils; it walks the
    # message list and produces the (image_inputs, video_inputs) lists
    # the processor expects. Optional dep — we install it via
    # dep_resolver, but if missing we fall back to passing the raw image
    # which works for single-image cases.
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore
        image_inputs, video_inputs = process_vision_info(messages)
    except ImportError:
        image_inputs, video_inputs = [image], None

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs


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

_HAS_VLM = _HAS_TORCH

# ──────────────────────────────────────────────────────────────────────
# Module-level helpers (run on background thread, no main-loop impact)
# ──────────────────────────────────────────────────────────────────────

_PALETTE = [
    (0, 230, 64),  (0, 112, 255), (255, 56, 56),  (255, 200, 0),
    (0, 230, 230), (230, 0, 230), (128, 255, 0),   (255, 128, 0),
]


def _annotate_frame(
    rgb: np.ndarray,
    detections: list[Detection],
    spatial: list[SpatialInfo],
) -> np.ndarray:
    """Return an annotated copy of *rgb* with bounding boxes and labels
    drawn for every detection — but NOT the segmentation mask overlay.

    Runs on the background VLM thread — never on the main pipeline loop.
    The annotation gives Qwen a visual anchor that matches the JSON
    grounding data, so it can correlate "object id=3 at 2.5 m" with
    what it sees.

    DESIGN NOTE — why no mask overlay?
    Earlier revisions blended the segmentation mask onto the object's
    pixels at 30 % opacity. That biased VLM colour-description output:
    the model dutifully reported the *blended* colour ("light-colored
    frame, white or very light gray") rather than the object's actual
    appearance — i.e. it was describing our highlight, not the TV.
    The bbox + label alone are enough of a visual anchor; the
    mask-coloured fill is what corrupts colour perception.

    If a future caller needs the object boundary as an extra cue
    without recolouring the pixels, draw the mask CONTOUR (outline,
    not fill) at full opacity. That preserves shape information
    without altering interior colour. A `draw_mask_contour=False`
    parameter could opt that in; not added here because the bbox
    has been sufficient in testing.
    """
    out = rgb.copy()
    for i, det in enumerate(detections):
        color = _PALETTE[det.track_id % len(_PALETTE)]
        x1, y1, x2, y2 = det.bbox.astype(int)

        # Segmentation mask overlay is intentionally not drawn — the
        # tinted fill biases VLM colour descriptions toward the highlight.

        # Bounding box (unchanged)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label: "#id class dist conf"
        dist_str = (
            f"{spatial[i].distance_m:.1f}m"
            if i < len(spatial) and spatial[i].distance_m > 0
            else "?"
        )
        label = f"#{det.track_id} {det.class_name} {dist_str} {det.confidence:.2f}"
        lx = x1
        ly = max(y1 - 6, 12)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (lx, ly - th - 2), (lx + tw + 2, ly + 2), color, -1)
        cv2.putText(out, label, (lx + 1, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _copy_frame(buf: np.ndarray) -> np.ndarray:
    """Copy frame buffer, handling both numpy arrays and torch tensors."""
    if isinstance(buf, np.ndarray):
        return buf.copy()
    try:
        return buf.clone().numpy()
    except Exception:
        return np.array(buf)


# ──────────────────────────────────────────────────────────────────────
# Cancellation + UMA pre-flight
# ──────────────────────────────────────────────────────────────────────
# ``model.generate()`` is an opaque blocking call. ``StoppingCriteria``
# is polled each token, so backing it with a ``threading.Event`` lets
# outside code preempt a running generation when a newer query arrives;
# the loop exits within 1–2 tokens.

_CancelCriteria = None  # type: ignore[assignment]
if _HAS_TORCH:
    try:
        from transformers import StoppingCriteria

        class _CancelCriteria(StoppingCriteria):  # type: ignore[no-redef]
            """Stop generate() the instant an external Event is set."""

            def __init__(self, event: threading.Event) -> None:
                super().__init__()
                self._event = event

            def __call__(self, input_ids, scores, **kwargs) -> bool:
                return self._event.is_set()
    except ImportError:
        _CancelCriteria = None  # type: ignore[assignment]


def _too_little_vram(min_free_gb: float = 8.0) -> bool:
    """True when CUDA reports < ``min_free_gb`` free — used as a soft
    guard before kicking off a 30B-MoE user query. Advisory on UMA
    (Jetson), not a hard ceiling. Never raises: on non-CUDA or missing
    API it reports healthy so we don't false-trigger the failover.
    """
    if not _HAS_TORCH:
        return False
    try:
        free, _total = torch.cuda.mem_get_info()
        return (free / (1024 ** 3)) < min_free_gb
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────
# SceneReasoner
# ──────────────────────────────────────────────────────────────────────

class _TokenStream:
    """Append-only buffer + condvar for one in-flight user query."""
    __slots__ = ("tokens", "done", "error", "lock", "cond", "qid")

    def __init__(self, qid: int) -> None:
        self.qid = qid
        self.tokens: list[str] = []
        self.done = False
        self.error: Optional[str] = None
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)

    def append(self, chunk: str) -> None:
        with self.cond:
            self.tokens.append(chunk)
            self.cond.notify_all()

    def finish(self, error: Optional[str] = None) -> None:
        with self.cond:
            self.done = True
            self.error = error
            self.cond.notify_all()

    def wait_next(self, pos: int, timeout: float) -> Optional[tuple[int, str]]:
        """Block until tokens past ``pos`` exist, ``done`` flips, or
        timeout. Returns ``(new_pos, joined_chunk)`` or ``None`` on
        completion / timeout.
        """
        with self.cond:
            deadline = time.monotonic() + timeout
            while True:
                if pos < len(self.tokens):
                    chunk = "".join(self.tokens[pos:])
                    return (len(self.tokens), chunk)
                if self.done:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.cond.wait(timeout=remaining)


class SceneReasoner:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._enabled = config.enable_vlm
        self._cooldown = config.vlm_cooldown
        self._model = None
        self._processor = None
        self._device = config.device

        self._lock = threading.Lock()
        self._load_lock = threading.Lock()   # guards lazy model load (double-checked)
        self._latest_description: str = ""
        # Separate slot for user-query responses so auto scene descriptions
        # don't overwrite the answer the user is waiting for (and vice versa).
        self._latest_user_response: str = ""
        self._tok_per_sec: float = 0.0
        self._last_run: float = 0.0
        self._running = False
        # Independent flag for user-query thread so a long-running auto scene
        # description can't silently drop a user question via the _running check.
        self._user_running = False
        self._thread: Optional[threading.Thread] = None
        self._user_thread: Optional[threading.Thread] = None
        # Latest-wins queue: when Q2 arrives while Q1 is still streaming, we
        # store Q2 here and dispatch it the moment Q1's worker exits. Only the
        # most recent query is retained — if Q3 arrives before Q2 starts, Q2
        # is overwritten (users rarely want a stale answer, they want the
        # freshest question answered).
        self._pending_user_payload: Optional[tuple] = None
        # Monotonic query id used to tag in-flight work. The streamer only
        # writes to _latest_user_response while the id it carries matches the
        # current id, so a superseded Q1 can't clobber Q2's partial output.
        self._user_query_counter: int = 0
        self._current_user_query_id: int = 0
        # Error message to surface when VLM load fails — displayed to the user
        # instead of a silent "Thinking…" loop.
        self._user_error_message: str = ""

        # Per-query cancellation events, keyed by query_id. Populated in
        # _start_user_worker, set by request_description() when a newer
        # query supersedes an older one, and consumed by _CancelCriteria
        # inside generate() so the in-flight run exits within a token or
        # two instead of burning GPU and UMA until completion.
        self._cancel_events: dict[int, threading.Event] = {}
        self._cancel_lock = threading.Lock()
        # Set True when the post-load warmup generate() raises OOM. When
        # set, request_description() short-circuits to the error path and
        # the pipeline failover routes future user queries to the
        # secondary VLM (Cosmos-Reason2) until the container restarts.
        self._warmup_failed: bool = False

        # Serializes every self._model.generate() call so the auto-caption
        # worker and the user-query worker can never hit the shared HF
        # model concurrently. `transformers.generate()` is not thread-safe
        # on a shared instance — overlapping runs on Jetson Thor UMA
        # corrupt CUDA context and take the display compositor down with
        # them (root cause of the open-ended-query screen freeze).
        #
        # Lock ordering: `_gen_lock` is innermost. Never acquire it while
        # holding `self._lock` or `self._cancel_lock` — the worker threads
        # take `_gen_lock`, run generate(), release it, then grab `_lock`
        # only to write results. Swapping that order would deadlock with
        # the main thread calling `request_description()`.
        self._gen_lock = threading.Lock()

        # Auto-caption cancellation signal. `_cache_objects` creates a
        # fresh Event per cycle and stashes it here; `request_description`
        # (user-query branch) fires it at the top so the auto-caption's
        # generate() exits within 1–2 tokens and releases `_gen_lock`,
        # letting the user query acquire it immediately instead of
        # waiting the full ~30 s cycle. `shutdown()` also fires this for
        # clean teardown.
        self._auto_cancel_event: Optional[threading.Event] = None

        self._formatter = None
        # Per-object visual description cache (track_id → description)
        self._object_cache: dict[int, str] = {}

        # Event-driven token publication for HTTP SSE consumers. Worker
        # appends each chunk to ``tokens`` and notifies the condvar;
        # readers wait on the condvar with a timeout. Avoids the
        # 100 ms polling loop in the prior /v1/vlm/query handler.
        # Lock ordering: per-stream lock is leaf — never held while
        # acquiring _gen_lock or _lock.
        self._streams: dict[int, "_TokenStream"] = {}
        self._streams_lock = threading.Lock()

    def initialize(self) -> None:
        """Kick off background pre-warm so model is resident before first query.

        Starts the disk→GPU transfer immediately (daemon thread) so Tier 1/2/2.5
        work is available right away while the 60 GB model loads in parallel.
        By the time the user types their first Tier 3 query the model is usually
        already in memory — zero visible wait.
        """
        if not self._enabled:
            logger.info("[VLM] Disabled — skipping load")
            return
        if not _HAS_VLM:
            logger.error("[VLM] PyTorch not available — Qwen-VL cannot load")
            return
        logger.info("[VLM] Starting background pre-warm (model loads while you work)...")
        threading.Thread(target=self._ensure_loaded, daemon=True, name="vlm-prewarm").start()

    def _ensure_loaded(self) -> bool:
        """Load model if not yet loaded.  Safe to call from background threads.

        Uses double-checked locking so concurrent callers don't trigger two
        simultaneous loads.  Returns True if model is ready, False otherwise.
        """
        if self._model is not None:
            return True
        if not self._enabled or not _HAS_VLM:
            return False
        with self._load_lock:
            if self._model is not None:    # another thread finished while we waited
                return True
            logger.info("[VLM] Lazy-loading model on first use (this may take several minutes)...")
            try:
                from app.modules.dep_resolver import auto_install
                from weda_optim.model_loader import load_vlm
                model, processor, formatter = auto_install(load_vlm)(self._config)
                if model is None:
                    logger.error("[VLM] Lazy load returned None — Qwen-VL unavailable")
                    return False
                self._model = model
                self._processor = processor
                self._formatter = formatter
                logger.info("[VLM] Model ready.")
                # Prime MoE expert routing + KV cache with a tiny dummy
                # generate() so the first user query doesn't cold-start
                # 128 experts and saturate UMA in one go (the main cause
                # of the screen-freeze on open-ended first queries).
                self._warmup()
                return True
            except Exception as e:
                logger.error("[VLM] Lazy load FAILED: %s", e)
                return False

    def _warmup(self) -> None:
        """Run a 4-token dummy generate() on a 64x64 black frame.

        Primes MoE expert routing and KV-cache allocation so the first
        real user query streams within seconds instead of cold-starting
        every expert simultaneously — which on the 30B-A3B model under
        BF16 was enough to stall the display compositor.

        Non-fatal: any exception (OOM most likely) flips
        ``self._warmup_failed`` and the pipeline's failover path takes
        over for subsequent user queries.
        """
        if self._model is None or self._processor is None:
            return
        try:
            from PIL import Image
            dummy = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
            messages = [{"role": "user", "content": [
                {"type": "image", "image": dummy},
                {"type": "text", "text": "ok"},
            ]}]
            inputs = _build_vlm_inputs(self._processor, messages, dummy)
            if inputs is None:
                # LLaVA-style fallback
                inputs = self._processor(
                    text="USER: <image>\nok\nASSISTANT:",
                    images=dummy, return_tensors="pt",
                )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            n_in = inputs["input_ids"].shape[-1]
            t0 = time.perf_counter()
            # _gen_lock: warmup is single-threaded (runs once during load
            # under _load_lock), but take the lock anyway so the invariant
            # "every model.generate() goes through _gen_lock" holds
            # uniformly — easier to audit than a special case.
            with torch.no_grad(), self._gen_lock:
                out = self._model.generate(**inputs, max_new_tokens=4)
            elapsed = time.perf_counter() - t0
            n_gen = max(0, int(out.shape[-1]) - int(n_in))
            tps = (n_gen / elapsed) if elapsed > 0 else 0.0
            logger.info(
                "[VLM] Warmup OK (%.1fs, %d tok, %.1f tok/s)",
                elapsed, n_gen, tps,
            )
        except Exception as e:
            self._warmup_failed = True
            logger.warning("[VLM] Warmup failed — failover enabled: %s", e)

    # ------------------------------------------------------------------
    # Background object caching (Path A)
    # ------------------------------------------------------------------

    def cache_object_descriptions(
        self,
        rgb: np.ndarray,
        detections: list[Detection],
        spatial: Optional[list[SpatialInfo]] = None,
    ) -> None:
        """Non-blocking: queue background VLM to describe uncached objects.

        Args:
            rgb:        Current RGB frame.
            detections: All detections for this frame.
            spatial:    Corresponding SpatialInfo list (same order as detections).
                        Passed through to _vlm_describe_crop so that each object
                        description is grounded with real distance/size/position data.
        """
        now = time.monotonic()
        if now - self._last_run < self._cooldown:
            return
        if self._running:
            return

        # Find objects not yet cached (limit to 3 per cycle)
        uncached = [
            d for d in detections
            if d.track_id >= 0 and d.track_id not in self._object_cache
        ][:3]

        if not uncached and not detections:
            return

        self._running = True
        self._last_run = now
        self._thread = threading.Thread(
            target=self._cache_objects,
            args=(_copy_frame(rgb), uncached, detections, spatial or []),
            daemon=True,
        )
        self._thread.start()

    def _cache_objects(
        self,
        rgb: np.ndarray,
        uncached: list[Detection],
        all_detections: list[Detection],
        spatial: list[SpatialInfo],
    ) -> None:
        """Background thread: describe uncached objects + update scene.

        Creates a fresh ``threading.Event`` and stashes it in
        ``self._auto_cancel_event`` so a user query arriving mid-cycle can
        abort the in-flight ``generate()`` via ``StoppingCriteria`` —
        otherwise the user would wait ~20 s per uncached object for us to
        finish before their query even acquires ``_gen_lock``.
        """
        if not self._ensure_loaded():
            self._running = False
            return

        # Install a fresh cancel event for this cycle so a user query
        # arriving after we start can signal us to abort. We replace any
        # prior event — the previous cycle's worker already exited (the
        # self._running guard in cache_object_descriptions enforces that)
        # so there's no risk of stealing its event.
        cancel_event = threading.Event()
        self._auto_cancel_event = cancel_event

        try:
            for det in uncached:
                # Check cancellation before spending time on the next
                # crop: a user query may have fired the event while we
                # were parsing the previous detection.
                if cancel_event.is_set():
                    break
                # Look up the spatial entry that corresponds to this detection
                det_idx = next(
                    (j for j, d in enumerate(all_detections) if d.track_id == det.track_id),
                    -1,
                )
                si = spatial[det_idx] if 0 <= det_idx < len(spatial) else None
                desc = self._describe_object(
                    rgb, det, spatial_info=si, cancel_event=cancel_event,
                )
                # Don't cache the "(description cancelled)" stub — next
                # cycle we want to retry this object with a real run.
                if cancel_event.is_set():
                    break
                with self._lock:
                    self._object_cache[det.track_id] = desc

            # Build scene summary from per-object descriptions
            with self._lock:
                cache_snapshot = dict(self._object_cache)
            if cache_snapshot and all_detections:
                parts = []
                for det in all_detections:
                    if det.track_id in cache_snapshot:
                        parts.append(f"#{det.track_id} {cache_snapshot[det.track_id]}")
                if parts:
                    with self._lock:
                        self._latest_description = "; ".join(parts)
        except Exception as e:
            logger.error(
                "VLM cache error: %s: %s",
                type(e).__name__, str(e) or repr(e),
                exc_info=True,
            )
        finally:
            # Clear the event reference so a stale Event can't be fired
            # by a future request_description() call against a worker
            # that's already gone.
            if self._auto_cancel_event is cancel_event:
                self._auto_cancel_event = None
            self._running = False

    def _describe_object(
        self,
        rgb: np.ndarray,
        det: Detection,
        spatial_info: Optional[SpatialInfo] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Describe a single object by cropping its bbox region.

        ``cancel_event`` is threaded to ``_vlm_describe_crop`` so the
        underlying ``generate()`` exits within 1–2 tokens when a user
        query arrives, releasing ``_gen_lock`` quickly.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("VLM not loaded")
        return self._vlm_describe_crop(
            rgb, det, spatial_info=spatial_info, cancel_event=cancel_event,
        )

    def _vlm_describe_crop(
        self,
        rgb: np.ndarray,
        det: Detection,
        spatial_info: Optional[SpatialInfo] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Crop object bbox from frame and ask VLM to describe it.

        The crop is annotated (mask overlay + bbox + label) and a single-object
        JSON block with all sensor metrics is prepended to the query, so Qwen
        bases size/distance statements on real sensor data rather than guessing.

        Args:
            cancel_event: Optional threading.Event — when set, the underlying
                generate() exits within a token or two via StoppingCriteria.
                Used by _cache_objects to abort mid-cycle when a user query
                arrives so the user doesn't wait 20–30 s for the auto-caption
                to finish before their answer starts streaming.
        """
        from PIL import Image
        from app.modules.query_engine import QueryEngine

        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = det.bbox.astype(int)
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        crop = rgb[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return f"{det.class_name} object (crop failed)"

        # Annotate the crop: re-express the bbox in crop-local coordinates
        crop_det = Detection(
            bbox=np.array([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], dtype=np.float32),
            confidence=det.confidence,
            class_id=det.class_id,
            class_name=det.class_name,
            mask=(det.mask[cy1:cy2, cx1:cx2] if det.mask is not None else None),
            track_id=det.track_id,
        )
        crop_spatial = [spatial_info] if spatial_info is not None else []
        annotated_crop = _annotate_frame(crop, [crop_det], crop_spatial)
        image = Image.fromarray(annotated_crop)

        # Build single-object JSON grounding block
        sensor_json = QueryEngine.build_grounded_json(
            [det], [spatial_info] if spatial_info is not None else [],
            rgb.shape, {},
        )

        query = (
            f"Sensor data for this {det.class_name}:\n{sensor_json}\n\n"
            f"Describe this {det.class_name}: color, material, any visible text "
            f"or logos, condition. Use the sensor data for any size or distance "
            f"statements — do not estimate what you can read from the JSON."
        )

        # Build messages using chat template (Qwen-VL style). Use the
        # canonical two-step recipe via ``_build_vlm_inputs`` to avoid
        # the placeholder/encoder token-count drift that crashes AWQ
        # Qwen3-VL with "tensor a (4304) vs tensor b (4352)".
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": query},
        ]}]
        inputs = _build_vlm_inputs(self._processor, messages, image)
        if inputs is None:
            # LLaVA-style fallback
            prompt = f"USER: <image>\n{query}\nASSISTANT:"
            inputs = self._processor(text=prompt, images=image, return_tensors="pt")

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        # Wire auto-caption cancellation: when request_description() fires
        # the event (top of the user-query branch), our StoppingCriteria
        # exits this generate() within 1–2 tokens so the lock releases
        # quickly and the user query doesn't wait ~20 s for our crop run.
        gen_kwargs: dict = {"max_new_tokens": 100}
        if cancel_event is not None and _CancelCriteria is not None:
            try:
                from transformers import StoppingCriteriaList
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [_CancelCriteria(cancel_event)]
                )
            except ImportError:
                pass  # older transformers — best effort, event just won't fire

        t0 = time.perf_counter()
        # _gen_lock: every model.generate() on the shared Qwen instance
        # serializes through this mutex. HF transformers generate() is
        # not thread-safe on a shared model — overlapping runs corrupt
        # CUDA context on Jetson UMA and freeze the display server.
        try:
            with torch.no_grad(), self._gen_lock:
                output = self._model.generate(**inputs, **gen_kwargs)
            self._record_tok_per_sec(output.shape[-1] - input_len, time.perf_counter() - t0)
        except RuntimeError as e:
            # Drop the inputs/intermediate tensors and force the
            # allocator to release whatever the failed forward held.
            # Without this, a stream of mismatched crops compounds into
            # UMA exhaustion (122 GB / 123 GB observed in the field
            # during a 16-h run with ~5 FPS).
            try:
                import torch as _t
                del inputs
                _t.cuda.empty_cache()
                _t.cuda.ipc_collect()
            except Exception:
                pass
            msg = str(e)
            if "size of tensor" in msg and "must match" in msg:
                logger.warning(
                    "[VLM] crop forward failed with token-count mismatch (%s); "
                    "dropping crop. The two-step chat-template recipe "
                    "(_build_vlm_inputs) should prevent this — if it "
                    "recurs, check qwen_vl_utils availability.", msg,
                )
                return f"{det.class_name} (vision-encoder mismatch)"
            raise

        # If we were cancelled mid-flight, don't try to parse the partial —
        # return a stub so _cache_objects can move on without polluting the
        # cache with half-sentences from a superseded run.
        if cancel_event is not None and cancel_event.is_set():
            return f"{det.class_name} (description cancelled)"

        text = self._processor.batch_decode(output, skip_special_tokens=True)[0]
        for marker in ["ASSISTANT:", "<answer>", "assistant\n"]:
            if marker in text:
                text = text.split(marker)[-1].strip()
                break
        return _strip_markdown(text)

    # ------------------------------------------------------------------
    # Tier 3: Live VLM with grounded context (Path B)
    # ------------------------------------------------------------------

    def request_description(
        self,
        rgb: np.ndarray,
        detections: list[Detection],
        grounded_context: Optional[str] = None,
        user_query: Optional[str] = None,
        spatial: Optional[list[SpatialInfo]] = None,
    ) -> None:
        """Non-blocking: submit frame for async VLM processing.

        Args:
            rgb:              Current RGB frame.
            detections:       All detections for grounding + annotation.
            grounded_context: JSON string from QueryEngine.build_grounded_json().
            user_query:       Optional user question for Tier 3 routing.
            spatial:          SpatialInfo list — used to annotate the frame.

        User-initiated queries bypass the auto scene-description cooldown and
        use a dedicated worker slot so they can't be silently dropped while a
        background caption is running. Their answer lands in
        ``_latest_user_response`` so the auto scene description can't clobber
        it on the next tick.
        """
        is_user_query = bool(user_query)

        if is_user_query:
            # ── Cancel any in-flight auto-caption FIRST ─────────────────
            # The auto-caption worker holds _gen_lock for ~20–30 s per
            # uncached object on the 30B MoE. Without this signal, the
            # user query would block that long waiting for the lock,
            # which is exactly the "typed a question, screen froze"
            # symptom. Firing the event triggers the StoppingCriteria
            # inside the auto-caption's generate(), so it exits within
            # 1–2 tokens, releases _gen_lock, and the user query
            # acquires it almost immediately.
            auto_event = self._auto_cancel_event
            if auto_event is not None:
                auto_event.set()

            # Cancel-and-replace semantics. Latest question wins: if a worker
            # is already streaming an older answer, we stash this newer
            # payload as "pending", bump the query id so the in-flight worker
            # knows it's been superseded, and dispatch the pending payload
            # the moment the current worker returns (see _process finally).
            payload = (
                _copy_frame(rgb),
                list(detections),
                grounded_context,
                user_query,
                list(spatial or []),
            )
            self._user_query_counter += 1
            new_id = self._user_query_counter

            if self._user_running:
                # Worker busy — cancel its in-flight generate() NOW so we
                # don't keep burning ~30 s of GPU (and saturating UMA)
                # finishing an answer the user already abandoned. The
                # StoppingCriteria hook reads this event every token and
                # exits within 1–2 tokens.
                with self._cancel_lock:
                    ev = self._cancel_events.get(self._current_user_query_id)
                if ev is not None:
                    ev.set()
                # Queue the new payload (overwrite any older pending one;
                # users want their *freshest* question answered).
                self._pending_user_payload = payload
                # Tell the UI the previous answer is stale: clear it so the
                # operator doesn't confuse Q1's text with Q2's pending work.
                with self._lock:
                    self._latest_user_response = ""
                    self._user_error_message = ""
                # self._current_user_query_id is intentionally not cleared:
                # the running worker keeps writing under the old id and its
                # output is discarded downstream when it finishes.
                return

            # ── UMA pre-flight guard ──
            # Qwen3-VL-30B-A3B under BF16 can spike 6–8 GB of fresh
            # activations on a long prompt. If we're already close to the
            # ceiling — or warmup itself already OOM'd — fail fast and let
            # the pipeline failover route the query to Cosmos-Reason2
            # rather than risk a display-server freeze.
            vlm_min_free_gb = float(getattr(self._config, "vlm_min_free_gb", 8.0))
            if self._warmup_failed or _too_little_vram(min_free_gb=vlm_min_free_gb):
                reason = (
                    "warmup previously failed"
                    if self._warmup_failed
                    else f"< {vlm_min_free_gb:.1f} GB free VRAM"
                )
                with self._lock:
                    self._user_error_message = (
                        f"VLM unavailable ({reason}) — failing over to "
                        "Cosmos-Reason2."
                    )
                logger.warning(
                    "[VLM] Pre-flight rejected user query: %s", reason
                )
                return

            # No worker running — dispatch immediately.
            self._current_user_query_id = new_id
            self._user_running = True
            with self._lock:
                self._latest_user_response = ""
                self._user_error_message = ""
            self._start_user_worker(payload)
            return

        # Auto scene description — respect cooldown so we don't thrash the VLM
        now = time.monotonic()
        if now - self._last_run < self._cooldown:
            return
        if self._running:
            return

        self._running = True
        self._last_run = now
        self._thread = threading.Thread(
            target=self._process,
            args=(
                _copy_frame(rgb), detections,
                grounded_context, user_query,
                spatial or [],
            ),
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def latest_description(self) -> str:
        with self._lock:
            return self._latest_description

    @property
    def latest_user_response(self) -> str:
        """Most recent Tier 3 response to a user-initiated query.

        Stored separately from ``latest_description`` so auto scene captions
        don't overwrite the answer the user is actively waiting for.
        """
        with self._lock:
            return self._latest_user_response

    @property
    def user_query_running(self) -> bool:
        """True while a user-initiated Tier 3 query is in-flight OR a newer
        one is queued behind it — from the UI's point of view, both states
        mean "the user's question is still being worked on"."""
        return self._user_running or self._pending_user_payload is not None

    @property
    def user_error_message(self) -> str:
        """Last user-facing error from the VLM worker (model load failure,
        OOM, etc). Empty string when the last query succeeded or none has
        been submitted. Displayed by pipeline.py when latest_user_response
        is empty and the worker has exited."""
        with self._lock:
            return self._user_error_message

    def last_failed(self) -> bool:
        """True when the most recent user query errored out AND no worker
        is currently running or queued — i.e. the pipeline should route
        the pending question through the Cosmos-Reason2 failover next
        frame. Combined gate keeps us from firing failover mid-stream.
        """
        with self._lock:
            return (
                bool(self._user_error_message)
                and not self._user_running
                and self._pending_user_payload is None
            )

    def clear_error(self) -> None:
        """Reset the error marker once the failover has committed an
        answer, so subsequent queries get a fresh attempt at the primary
        VLM (assuming warmup didn't permanently fail)."""
        with self._lock:
            self._user_error_message = ""

    def _start_user_worker(self, payload: tuple) -> None:
        """Spawn the background worker thread for a user query.

        Must be called with self._user_running already set to True and
        self._current_user_query_id already bumped to the payload's id.

        Creates a fresh ``threading.Event`` and stores it in
        ``self._cancel_events[qid]`` so a subsequent user query can
        interrupt this one via ``StoppingCriteria``.
        """
        rgb, detections, grounded_context, user_query, spatial = payload
        qid = self._current_user_query_id
        cancel_event = threading.Event()
        with self._cancel_lock:
            self._cancel_events[qid] = cancel_event
        with self._streams_lock:
            self._streams[qid] = _TokenStream(qid)
        self._user_thread = threading.Thread(
            target=self._process,
            args=(rgb, detections, grounded_context, user_query, spatial),
            kwargs={
                "is_user_query": True,
                "query_id": qid,
                "cancel_event": cancel_event,
            },
            daemon=True,
        )
        self._user_thread.start()

    def submit_user_query(
        self,
        rgb,
        detections: list[Detection],
        grounded_context: Optional[str],
        user_query: str,
        spatial: Optional[list[SpatialInfo]] = None,
    ) -> Optional[int]:
        """Public entry-point for HTTP callers that want event-driven
        token streaming. Returns the assigned ``query_id`` once the
        worker has been spawned, or ``None`` when the pre-flight guard
        rejected the request.

        Differs from ``request_description(user_query=...)`` only in
        that it returns the qid synchronously so the caller can
        ``open_stream(qid)`` and pump tokens out via
        ``_TokenStream.wait_next``.
        """
        # Cancel auto-caption so the query is not stuck waiting for
        # _gen_lock — same semantics as request_description.
        auto_event = self._auto_cancel_event
        if auto_event is not None:
            auto_event.set()

        payload = (
            _copy_frame(rgb), list(detections), grounded_context,
            user_query, list(spatial or []),
        )
        with self._lock:
            self._user_query_counter += 1
            new_id = self._user_query_counter
            if self._user_running:
                # Cancel-and-replace: same as keyboard path.
                with self._cancel_lock:
                    ev = self._cancel_events.get(self._current_user_query_id)
                if ev is not None:
                    ev.set()
                self._pending_user_payload = payload
                self._latest_user_response = ""
                self._user_error_message = ""
                # The pending payload will be dispatched by the
                # currently-running worker on its finally clause.
                # We don't have its qid yet to expose; return None and
                # let the caller fall back to a polling fetch.
                return None
            vlm_min_free_gb = float(getattr(self._config, "vlm_min_free_gb", 8.0))
            if self._warmup_failed or _too_little_vram(min_free_gb=vlm_min_free_gb):
                reason = (
                    "warmup previously failed" if self._warmup_failed
                    else f"< {vlm_min_free_gb:.1f} GB free VRAM"
                )
                self._user_error_message = (
                    f"VLM unavailable ({reason}) — failing over."
                )
                return None
            self._current_user_query_id = new_id
            self._user_running = True
            self._latest_user_response = ""
            self._user_error_message = ""
        self._start_user_worker(payload)
        return new_id

    def get_token_stream(self, qid: int) -> Optional["_TokenStream"]:
        """Return the live token stream for a query, or ``None`` once it
        has finished and been evicted."""
        with self._streams_lock:
            return self._streams.get(qid)

    @property
    def vlm_tok_per_sec(self) -> float:
        with self._lock:
            return self._tok_per_sec

    @property
    def _loaded(self) -> bool:
        return self._model is not None

    def _record_tok_per_sec(self, gen_tokens: int, elapsed: float) -> None:
        if elapsed > 0 and gen_tokens > 0:
            tps = gen_tokens / elapsed
            with self._lock:
                self._tok_per_sec = 0.3 * tps + 0.7 * self._tok_per_sec if self._tok_per_sec else tps

    @property
    def object_cache(self) -> dict[int, str]:
        with self._lock:
            return dict(self._object_cache)

    def prune_cache(self, active_track_ids: set[int]) -> None:
        """Remove cache entries for tracks no longer active."""
        with self._lock:
            stale = [tid for tid in self._object_cache if tid not in active_track_ids]
            for tid in stale:
                del self._object_cache[tid]

    def shutdown(self) -> None:
        # Fire every outstanding cancel event first so any in-flight
        # generate() exits within a token or two instead of blocking us
        # for the remainder of its run (30 s+ on the 30B MoE).
        with self._cancel_lock:
            for ev in self._cancel_events.values():
                ev.set()
            self._cancel_events.clear()
        # Same for the auto-caption side — without this, shutdown would
        # wait for a ~20 s per-object crop run to finish before joining
        # the worker thread, during which the rest of the pipeline is
        # already half-torn-down.
        auto_event = self._auto_cancel_event
        if auto_event is not None:
            auto_event.set()
        # Join the auto-caption worker (background scene description).
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        # Join the user-query worker too — it was left alive on the
        # daemon flag before, which mostly worked but could leave a
        # half-finished generate() blocking the GPU during restart.
        if self._user_thread and self._user_thread.is_alive():
            self._user_thread.join(timeout=5.0)
        self._model = None
        self._processor = None
        with self._lock:
            self._object_cache.clear()

    # ------------------------------------------------------------------
    # Internal processing
    # ------------------------------------------------------------------

    def _process(
        self,
        rgb: np.ndarray,
        detections: list[Detection],
        grounded_context: Optional[str] = None,
        user_query: Optional[str] = None,
        spatial: Optional[list[SpatialInfo]] = None,
        is_user_query: bool = False,
        query_id: int = 0,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        t_query_start = time.monotonic()
        load_failed = False
        infer_failed_reason = ""
        full_response = ""

        if not self._ensure_loaded():
            load_failed = True
            if is_user_query:
                # Surface the failure to the UI instead of letting the user
                # sit on an eternal "Thinking…" placeholder.
                with self._lock:
                    self._user_error_message = (
                        "VLM offline: model failed to load. "
                        "Check container logs for details."
                    )
        else:
            try:
                full_response = self._vlm_infer(
                    rgb, grounded_context, user_query,
                    detections=detections, spatial=spatial or [],
                    is_user_query=is_user_query,
                    query_id=query_id,
                    cancel_event=cancel_event,
                )
                # Only commit the final answer if this worker is still the
                # "current" one — a newer user query may have superseded us.
                with self._lock:
                    if is_user_query:
                        if query_id == self._current_user_query_id:
                            self._latest_user_response = full_response
                        # else: superseded — silently discard
                    else:
                        self._latest_description = full_response
            except TimeoutError as te:
                # Hard 60 s wall-clock hit inside the streaming loop.
                # Mark as a user-facing error so pipeline.py's failover
                # branch routes the pending question to Cosmos-Reason2
                # instead of leaving the user staring at a frozen
                # "Thinking…" placeholder.
                infer_failed_reason = f"timeout: {te}"
                logger.warning("[VLM] Hard timeout: %s", te)
                if is_user_query:
                    with self._lock:
                        self._user_error_message = (
                            "VLM timed out after 60s - falling back to "
                            "physical reasoner."
                        )
            except Exception as e:
                infer_failed_reason = str(e)
                logger.error("VLM inference error: %s", e)
                if is_user_query:
                    with self._lock:
                        self._user_error_message = (
                            "VLM inference error - see logs."
                        )

        # Structured per-query log for tuning + debugging. Lives in one place
        # (app/utils/query_log.py) so we don't scatter JSONL emitters.
        if is_user_query:
            try:
                from app.utils.query_log import log_query
                status = (
                    "load_failed" if load_failed
                    else "inference_error" if infer_failed_reason
                    else "ok"
                )
                log_query(
                    tier=3,
                    model="vlm",
                    query=user_query or "",
                    response=full_response,
                    elapsed_s=time.monotonic() - t_query_start,
                    status=status,
                    error=infer_failed_reason or ("load_failed" if load_failed else ""),
                )
            except Exception:
                pass  # logger never breaks the main path

        # Worker bookkeeping + dispatch-next-pending.
        if is_user_query:
            # Reap this worker's cancel event — nobody will flip it now
            # that the run is complete, and the dict shouldn't grow
            # unbounded across a long session.
            with self._cancel_lock:
                self._cancel_events.pop(query_id, None)
            # Signal end-of-stream to any HTTP SSE waiter, then evict.
            with self._streams_lock:
                stream = self._streams.pop(query_id, None)
            if stream is not None:
                stream.finish(error=infer_failed_reason or None)
            # Under lock so request_description() and _process() agree on
            # whether a queued payload should be launched here.
            payload_to_run = None
            with self._lock:
                self._user_running = False
                if self._pending_user_payload is not None:
                    payload_to_run = self._pending_user_payload
                    self._pending_user_payload = None
                    self._user_query_counter += 1
                    self._current_user_query_id = self._user_query_counter
                    self._user_running = True
                    # Clear error + stale text so the UI shows the new query
                    # starting from a clean slate.
                    self._latest_user_response = ""
                    self._user_error_message = ""
            if payload_to_run is not None:
                self._start_user_worker(payload_to_run)
        else:
            self._running = False

    def _vlm_infer(
        self,
        rgb: np.ndarray,
        grounded_context: Optional[str] = None,
        user_query: Optional[str] = None,
        detections: Optional[list[Detection]] = None,
        spatial: Optional[list[SpatialInfo]] = None,
        is_user_query: bool = False,
        query_id: int = 0,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Run one VLM forward pass.

        If *detections* are provided the frame is annotated before being
        passed to Qwen, giving it a visual anchor that matches the JSON
        grounding block.  All annotation work happens here on the background
        thread — the main loop is never touched.
        """
        from PIL import Image

        # Annotate frame if we have detection data
        if detections:
            try:
                frame_to_show = _annotate_frame(rgb, detections, spatial or [])
                logger.debug("[VLM] Frame annotated with %d objects", len(detections))
            except Exception as ann_err:
                logger.debug("[VLM] Frame annotation failed (%s) — using raw frame", ann_err)
                frame_to_show = rgb
        else:
            frame_to_show = rgb

        image = Image.fromarray(frame_to_show)

        query = user_query or "Describe this scene briefly for a robotic system."
        context = grounded_context or ""
        max_tokens = getattr(self._config, "vlm_max_tokens", 100)

        logger.debug("[VLM] Grounding context (%d chars):\n%s", len(context), context[:500])

        # Use config-driven prompt formatter
        if self._formatter is not None:
            formatted = self._formatter(context, query)
        else:
            formatted = f"USER: <image>\n{context}\n{query}\nASSISTANT:"

        # Handle chat-template format (list of messages) vs string format
        if isinstance(formatted, list):
            # Qwen-VL processor: replace {"type": "image"} placeholder with actual image
            for msg in formatted:
                if isinstance(msg.get("content"), list):
                    for i, item in enumerate(msg["content"]):
                        if isinstance(item, dict) and item.get("type") == "image":
                            msg["content"][i] = {"type": "image", "image": image}
            inputs = _build_vlm_inputs(self._processor, formatted, image)
            if inputs is None:
                inputs = self._processor(text=formatted, images=image, return_tensors="pt")
        else:
            inputs = self._processor(text=formatted, images=image, return_tensors="pt")

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Streaming: each token updates _latest_description in real time so the
        # display overlay shows the first words within ~1-2 sec instead of waiting
        # for the full response (10-50 sec for 100 tokens).
        try:
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(
                self._processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            gen_kwargs = {**inputs, "max_new_tokens": max_tokens, "streamer": streamer}
            # Wire the cancellation hook so a newer query (or the hard
            # 60 s timeout below) can break the generate() loop cleanly
            # within 1-2 tokens instead of burning the full run.
            if cancel_event is not None and _CancelCriteria is not None:
                try:
                    from transformers import StoppingCriteriaList
                    gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                        [_CancelCriteria(cancel_event)]
                    )
                except ImportError:
                    pass  # older transformers — fall back to id-mismatch drop
            input_len = inputs["input_ids"].shape[-1]
            t0_stream = time.perf_counter()

            # IMPORTANT: acquire _gen_lock *inside* the daemon's target so
            # the main pipeline thread (which owns request_description /
            # the streaming consumer loop below) never blocks on the lock
            # itself — only background workers ever wait on each other.
            # If another generate() is running (auto-caption crop worker),
            # request_description() will have already fired
            # _auto_cancel_event at the top of the user-query branch, so
            # that worker exits within 1–2 tokens and releases the lock
            # almost immediately. Without this lock, two concurrent
            # generate()s on the same HF model corrupt CUDA context on
            # Jetson UMA and freeze the display compositor.
            def _run_generate() -> None:
                with self._gen_lock:
                    self._model.generate(**gen_kwargs)

            gen_thread = threading.Thread(
                target=_run_generate,
                daemon=True,
            )
            gen_thread.start()

            # Hard wall-clock deadline. 60 s covers the worst-case legitimate
            # run on a 30B-MoE at low tok/s plus headroom; past that we'd
            # rather failover to Cosmos-Reason2 than keep the user frozen.
            hard_deadline = time.monotonic() + 60.0
            tokens: list[str] = []
            n_gen_tokens = 0
            for chunk in streamer:
                if cancel_event is not None and cancel_event.is_set():
                    # Superseded by a newer query — exit the streaming
                    # loop; the generate thread will terminate within a
                    # token via StoppingCriteria.
                    break
                if time.monotonic() > hard_deadline:
                    if cancel_event is not None:
                        cancel_event.set()
                    # Give the generate thread a moment to honor the
                    # stopping criteria before we abandon it.
                    gen_thread.join(timeout=2.0)
                    raise TimeoutError(
                        f"VLM generate exceeded 60s wall-clock limit "
                        f"(produced {n_gen_tokens} tokens)"
                    )
                tokens.append(chunk)
                n_gen_tokens += 1  # each chunk ≈ 1 token
                partial = _strip_markdown("".join(tokens))
                with self._lock:
                    if is_user_query:
                        # Drop partial tokens if a newer query has superseded
                        # us — keeps Q1's tail from flashing inside Q2's view.
                        if query_id == self._current_user_query_id:
                            self._latest_user_response = partial
                    else:
                        self._latest_description = partial
                if is_user_query:
                    stream = self._streams.get(query_id)
                    if stream is not None:
                        stream.append(chunk)
            gen_thread.join(timeout=5.0)
            self._record_tok_per_sec(n_gen_tokens, time.perf_counter() - t0_stream)
            text = "".join(tokens)
        except Exception as stream_err:
            # Fallback to blocking generate if streaming fails
            logger.debug("[VLM] Streaming unavailable (%s) — blocking generate", stream_err)
            input_len = inputs["input_ids"].shape[-1]
            t0 = time.perf_counter()
            # Same lock invariant as the streaming path: serialize every
            # generate() on the shared Qwen instance so CUDA context
            # can't be corrupted by an overlapping auto-caption run.
            with torch.no_grad(), self._gen_lock:
                output = self._model.generate(**inputs, max_new_tokens=max_tokens)
            self._record_tok_per_sec(output.shape[-1] - input_len, time.perf_counter() - t0)
            text = self._processor.batch_decode(output, skip_special_tokens=True)[0]

        # Extract response portion (non-streaming fallback may include prompt)
        for marker in ["ASSISTANT:", "<answer>", "assistant\n"]:
            if marker in text:
                text = text.split(marker)[-1].strip()
                break
        return _strip_markdown(text)
