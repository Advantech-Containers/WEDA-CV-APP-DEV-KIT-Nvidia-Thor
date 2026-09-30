"""RT-Embedding: Cosmos-Embed1-224p for video/image/text embeddings.

Provides:
  - Frame-level scene embeddings (scene change detection)
  - Object crop embeddings (tracker re-ID across occlusions)
  - Text query embeddings (semantic tier routing)

Fails loudly if Cosmos model unavailable — no synthetic fallback in production.
"""

from __future__ import annotations
import concurrent.futures as _cf
import threading
import time
from typing import Optional
import cv2
import numpy as np
from app.utils.data_structures import Detection, PipelineConfig
from app.utils.logger import get_logger

logger = get_logger()

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from transformers import AutoProcessor, AutoModel
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

try:
    import einops
    _HAS_EINOPS = True
except ImportError:
    _HAS_EINOPS = False


class Embedder:
    """Cosmos-Embed1-224p video/text embedder with scene change detection."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._enabled = config.enable_embedding
        self._dim = config.embedding_dim
        self._scene_threshold = config.scene_change_threshold
        self._model = None
        self._processor = None
        self._device = config.device
        self._last_frame_embedding: Optional[np.ndarray] = None
        self._text_cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._embed_executor = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="embedder")
        self._pending_frame_future: Optional[_cf.Future] = None
        self._last_frame_result: tuple[Optional[np.ndarray], bool] = (None, False)

    def initialize(self) -> None:
        if not self._enabled:
            logger.info("RT-Embedding disabled")
            return

        if not (_HAS_TORCH and _HAS_TRANSFORMERS and _HAS_EINOPS):
            missing = []
            if not _HAS_TORCH:
                missing.append("torch")
            if not _HAS_TRANSFORMERS:
                missing.append("transformers")
            if not _HAS_EINOPS:
                missing.append("einops")
            logger.warning(
                "Cosmos-Embed1 deps missing (%s) - embedder DISABLED. "
                "packet.frame_embedding will be None every frame. "
                "Check entrypoint install logs (e.g. /tmp/einops.log).",
                ", ".join(missing),
            )
            return

        try:
            self._patch_hf_strict()
            self._patch_transformers_compat()

            model_id = self._config.embedding_model
            logger.info("Loading embedding model: %s", model_id)
            revision = getattr(self._config, "embedding_revision", None)

            load_kwargs = dict(trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=False)
            if revision:
                load_kwargs["revision"] = revision

            import contextlib
            import importlib
            @contextlib.contextmanager
            def _noop_init_empty_weights(*args, **kwargs):
                yield

            _iew_locs = (
                "transformers.modeling_utils",
                "accelerate",
                "accelerate.utils",
                "accelerate.utils.modeling",
                "transformers.integrations.accelerate",
            )
            _iew_patches: list[tuple] = []
            for _loc in _iew_locs:
                try:
                    _mod = importlib.import_module(_loc)
                    if hasattr(_mod, "init_empty_weights"):
                        _iew_patches.append((_mod, "init_empty_weights",
                                             getattr(_mod, "init_empty_weights")))
                        setattr(_mod, "init_empty_weights", _noop_init_empty_weights)
                except (ImportError, AttributeError):
                    pass
            logger.debug("init_empty_weights patched in %d locations", len(_iew_patches))
            _old_linspace = torch.linspace

            def _patched_linspace(*args, **kwargs):
                kwargs["device"] = "cpu"
                return _old_linspace(*args, **kwargs)

            torch.linspace = _patched_linspace

            import transformers as _tf
            _orig_getattr = getattr(_tf.PreTrainedModel, "__getattr__", None)

            def _patched_getattr(self, name):
                if name == "all_tied_weights_keys":
                    res = getattr(self, "_tied_weights_keys", None)
                    return res if res is not None else {}
                if _orig_getattr is not None:
                    return _orig_getattr(self, name)
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                )

            _tf.PreTrainedModel.__getattr__ = _patched_getattr

            import inspect as _inspect
            _config_cls = getattr(_tf, "PreTrainedConfig", None) \
                or getattr(_tf, "PretrainedConfig", None)
            _orig_cfg_init = _config_cls.__init__ if _config_cls is not None else None
            _patched_cfg_init = None
            if _orig_cfg_init is not None:
                try:
                    _sig = _inspect.signature(_orig_cfg_init)
                    _params = _sig.parameters
                    _has_var_kw = any(
                        p.kind == _inspect.Parameter.VAR_KEYWORD
                        for p in _params.values()
                    )
                    _accepted = {n for n, p in _params.items()
                                 if p.kind != _inspect.Parameter.VAR_KEYWORD
                                 and n != "self"}
                except (TypeError, ValueError):
                    _has_var_kw = True
                    _accepted = set()

                def _patched_cfg_init(self, **kwargs):
                    rejected = {}
                    if not _has_var_kw:
                        accepted_kwargs = {}
                        for k, v in kwargs.items():
                            if k in _accepted:
                                accepted_kwargs[k] = v
                            else:
                                rejected[k] = v
                        kwargs = accepted_kwargs
                    _orig_cfg_init(self, **kwargs)
                    for k, v in rejected.items():
                        try:
                            setattr(self, k, v)
                        except Exception:
                            pass

                _config_cls.__init__ = _patched_cfg_init

            try:
                self._model = AutoModel.from_pretrained(model_id, **load_kwargs)
            finally:
                if _config_cls is not None and _orig_cfg_init is not None \
                        and _patched_cfg_init is not None:
                    _config_cls.__init__ = _orig_cfg_init
                _tf.PreTrainedModel.__getattr__ = _orig_getattr
                torch.linspace = _old_linspace
                for _mod, _attr, _orig in _iew_patches:
                    try:
                        setattr(_mod, _attr, _orig)
                    except Exception:
                        pass

            self._model = self._model.to(self._device, dtype=torch.bfloat16)
            self._model.eval()
            proc_kwargs = dict(trust_remote_code=True)
            if revision:
                proc_kwargs["revision"] = revision
            self._processor = AutoProcessor.from_pretrained(model_id, **proc_kwargs)
            self._dim = 256
            logger.info("Cosmos-Embed1 loaded (dim=%d)", self._dim)
        except Exception as e:
            import traceback
            logger.error("Cosmos-Embed1 load FAILED: %s\n%s", e, traceback.format_exc())
            self._model = None

    @staticmethod
    def _patch_hf_strict() -> None:
        """No-op the huggingface_hub @strict decorator that breaks non-dataclass configs.

        huggingface_hub >= 0.29 enforces @strict on config classes, but transformers
        config classes (BertConfig etc.) are not Python @dataclass — crashes on import.
        """
        try:
            import huggingface_hub.dataclasses as _hf_dc
            if hasattr(_hf_dc, "strict"):
                _hf_dc.strict = lambda cls=None, **kw: cls if cls is not None else lambda c: c
        except Exception:
            pass

    @staticmethod
    def _patch_transformers_compat() -> None:
        """Patch functions removed in transformers v5 back in for Cosmos-Embed1 compat.

        All patches here are PERMANENT (process lifetime) because they are needed
        during inference (every forward pass), not just during model loading.
        """
        try:
            import transformers.pytorch_utils as ptu
            if not hasattr(ptu, "find_pruneable_heads_and_indices"):
                def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
                    mask = torch.ones(n_heads, head_size)
                    heads = set(heads) - already_pruned_heads
                    for head in heads:
                        mask[head] = 0
                    mask = mask.view(-1).contiguous().eq(1)
                    index = torch.arange(len(mask))[mask].long()
                    return heads, index

                ptu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
                logger.debug("Patched find_pruneable_heads_and_indices into transformers.pytorch_utils")
        except Exception as e:
            logger.debug("transformers compat patch (find_pruneable_heads_and_indices) failed: %s", e)

        try:
            import transformers as _tf
            if not hasattr(_tf.PreTrainedModel, "get_head_mask"):
                def _get_head_mask(self, head_mask, num_hidden_layers,
                                   is_attention_chunked: bool = False):
                    if head_mask is not None:
                        head_mask = self._convert_head_mask_to_5d(
                            head_mask, num_hidden_layers
                        )
                        if is_attention_chunked:
                            head_mask = head_mask.unsqueeze(-1)
                    else:
                        head_mask = [None] * num_hidden_layers
                    return head_mask

                def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
                    if head_mask.dim() == 1:
                        head_mask = (
                            head_mask.unsqueeze(0).unsqueeze(0)
                            .unsqueeze(-1).unsqueeze(-1)
                        )
                        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                    elif head_mask.dim() == 2:
                        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                    assert head_mask.dim() == 5, (
                        f"head_mask.dim != 5, got {head_mask.dim()}"
                    )
                    head_mask = head_mask.to(dtype=self.dtype)
                    return head_mask

                _tf.PreTrainedModel.get_head_mask = _get_head_mask
                _tf.PreTrainedModel._convert_head_mask_to_5d = _convert_head_mask_to_5d
                logger.debug(
                    "Patched get_head_mask + _convert_head_mask_to_5d into PreTrainedModel"
                )
        except Exception as e:
            logger.debug("transformers compat patch (get_head_mask) failed: %s", e)

    def embed_frame_async(self, rgb: np.ndarray) -> tuple[Optional[np.ndarray], bool]:
        if self._pending_frame_future is not None:
            try:
                self._last_frame_result = self._pending_frame_future.result(timeout=0)
            except _cf.TimeoutError:
                return self._last_frame_result
            except Exception as e:
                logger.debug("Async embed_frame error: %s", e)
            self._pending_frame_future = None

        rgb_np = _as_numpy(rgb)
        self._pending_frame_future = self._embed_executor.submit(self.embed_frame, rgb_np.copy())
        return self._last_frame_result

    def embed_frame(self, rgb: np.ndarray) -> tuple[Optional[np.ndarray], bool]:
        """Embed a frame and detect scene change.

        Returns (embedding, scene_changed). Returns (None, False) if model unavailable.
        """
        if self._model is None or self._processor is None:
            return None, False
        emb = self._compute_image_embedding(rgb)
        changed = self._check_scene_change(emb)
        with self._lock:
            self._last_frame_embedding = emb
        return emb, changed

    def embed_crops(
        self, rgb: np.ndarray, detections: list[Detection]
    ) -> list[Optional[np.ndarray]]:
        """Embed cropped regions — single batched GPU forward pass.

        Stacks all valid crops into one [N,1,3,224,224] tensor and runs one
        get_video_embeddings call instead of N sequential calls (~60ms for any
        N vs N×50ms before).  Falls back to per-crop on batch error.
        """
        if not detections:
            return []
        if self._model is None or self._processor is None:
            return [None] * len(detections)

        rgb_np = _as_numpy(rgb)
        h, w = rgb_np.shape[:2]
        results: list[Optional[np.ndarray]] = [None] * len(detections)

        to_embed: list[tuple[int, "torch.Tensor"]] = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            to_embed.append((i, self._preprocess_crop(rgb_np[y1:y2, x1:x2])))

        if to_embed:
            try:
                indices, tensors = zip(*to_embed)
                batch = torch.cat(list(tensors), dim=0).unsqueeze(1)
                batch = batch.to(self._device, dtype=torch.bfloat16)
                with torch.no_grad():
                    out = self._model.get_video_embeddings(videos=batch)
                    embs = out.visual_proj.cpu().float()
                norms = embs.norm(dim=1, keepdim=True).clamp(min=1e-8)
                embs = (embs / norms).numpy()
                for idx, emb in zip(indices, embs):
                    results[idx] = emb
            except Exception as e:
                logger.error("Batch crop embedding failed: %s — per-crop fallback", e)
                for i, _ in to_embed:
                    try:
                        det = detections[i]
                        x1, y1, x2, y2 = det.bbox.astype(int)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        results[i] = self._compute_image_embedding(rgb_np[y1:y2, x1:x2])
                    except Exception:
                        pass

        return results

    def _preprocess_crop(self, crop: np.ndarray) -> "torch.Tensor":
        """Resize + normalise crop → [1, 3, 224, 224] float32 tensor (no time dim)."""
        img = cv2.resize(crop, (224, 224))
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text query for semantic similarity matching."""
        with self._lock:
            cached = self._text_cache.get(text)
            if cached is not None:
                return cached

        emb = self._compute_text_embedding(text)
        with self._lock:
            self._text_cache[text] = emb
            if len(self._text_cache) > 200:
                oldest = next(iter(self._text_cache))
                del self._text_cache[oldest]
        return emb

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def shutdown(self) -> None:
        try:
            self._embed_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._embed_executor.shutdown(wait=False)
        self._model = None
        self._processor = None
        self._text_cache.clear()

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def is_loaded(self) -> bool:
        """True once Cosmos-Embed1 finished loading to GPU.

        Used by the renderer to drive the three-state EMB LED:
          - disabled in config   → gray
          - enabled but not loaded (deps missing or load failed) → orange
          - enabled and loaded   → green

        Reads the same flag ``embed_frame`` uses to decide whether to
        short-circuit, so the LED state stays in lock-step with actual
        behaviour — no latch, no "ever seen" heuristic.
        """
        return self._model is not None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_image_embedding(self, image) -> np.ndarray:
        # Ensure numpy array (pool buffers may be pinned torch tensors)
        if hasattr(image, 'numpy'):
            image = image.numpy()
        image = np.asarray(image)

        if self._model is None or self._processor is None:
            raise RuntimeError("Cosmos-Embed1 not loaded")

        try:
            img_resized = cv2.resize(image, (224, 224))
            img_float = img_resized.astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.unsqueeze(1)
            tensor = tensor.to(self._device, dtype=torch.bfloat16)

            with torch.no_grad():
                out = self._model.get_video_embeddings(videos=tensor)
                emb = out.visual_proj[0].cpu().float().numpy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb
        except Exception as e:
            logger.error("Image embedding inference failed: %s", e)
            raise

    def _compute_text_embedding(self, text: str) -> np.ndarray:
        if self._model is None or self._processor is None:
            raise RuntimeError("Cosmos-Embed1 not loaded")

        try:
            inputs = self._processor(text=[text], return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model.get_text_embeddings(**inputs)
                emb = out.text_proj[0].cpu().float().numpy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb
        except Exception as e:
            logger.error("Text embedding inference failed: %s", e)
            raise

    def _check_scene_change(self, current: np.ndarray) -> bool:
        with self._lock:
            if self._last_frame_embedding is None:
                return True
            sim = self.cosine_similarity(current, self._last_frame_embedding)
            return sim < self._scene_threshold

def _as_numpy(buf) -> np.ndarray:
    if isinstance(buf, np.ndarray):
        return buf
    try:
        return buf.numpy()
    except Exception:
        return np.asarray(buf)
