from __future__ import annotations

import contextlib as _cl
import gc
import importlib as _il
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable


def _patch_hf_strict() -> None:
    try:
        import huggingface_hub.dataclasses as _hf_dc
        if hasattr(_hf_dc, "strict"):
            _hf_dc.strict = lambda cls=None, **kw: cls if cls is not None else lambda c: c
    except Exception:
        pass

_patch_hf_strict()

from weda_optim.quantization import (
    apply_modelopt_quantization,
    detect_backend,
    get_quantization_config,
)

logger = logging.getLogger("weda_optim")


def _log_mem(label: str) -> None:
    try:
        with open("/proc/meminfo", "r") as _f:
            _mi = {l.split(":")[0].strip(): l.split(":")[1].strip()
                   for l in _f if ":" in l}
        avail = int(_mi.get("MemAvailable", "0 kB").split()[0]) / 1_048_576
        total = int(_mi.get("MemTotal", "0 kB").split()[0]) / 1_048_576
        logger.info("[MEM %s] RAM %.1f / %.1f GB available", label, avail, total)
    except Exception:
        pass
    try:
        import torch as _t
        alloc = _t.cuda.memory_allocated() / 1_073_741_824
        resv  = _t.cuda.memory_reserved()  / 1_073_741_824
        logger.info("[MEM %s] CUDA %.1f GB allocated, %.1f GB reserved", label, alloc, resv)
    except Exception:
        pass


try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
    _HAS_TRANSFORMERS = True


    def _patch_generation_config(model):
        try:
            gc_cfg = getattr(model, "generation_config", None)
            if gc_cfg is not None and getattr(gc_cfg, "pad_token_id", None) is None:
                eos = getattr(gc_cfg, "eos_token_id", None)
                if eos is not None:
                    gc_cfg.pad_token_id = eos if isinstance(eos, int) else eos[0]
        except Exception:
            pass
        return model


    def _safe_from_pretrained(model_class, path_or_id, **kwargs):
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
        try:
            if "attn_implementation" in kwargs:
                return _patch_generation_config(
                    model_class.from_pretrained(path_or_id, **kwargs)
                )
            try:
                return _patch_generation_config(model_class.from_pretrained(
                    path_or_id, attn_implementation="sdpa", **kwargs,
                ))
            except (ValueError, TypeError) as e:
                msg = str(e).lower()
                if any(t in msg for t in (
                    "flash", "sdpa", "attn_implementation", "does not support",
                )):
                    return _patch_generation_config(model_class.from_pretrained(
                        path_or_id, attn_implementation="eager", **kwargs,
                    ))
                raise
        finally:
            if _orig_getattr is not None:
                _tf.PreTrainedModel.__getattr__ = _orig_getattr


    def _safe_load_processor(path_or_id, **kwargs):
        proc = AutoProcessor.from_pretrained(path_or_id, **kwargs)
        tok = getattr(proc, "tokenizer", None) or proc
        if (getattr(tok, "pad_token", None) is None
                and getattr(tok, "eos_token", None) is not None):
            try:
                tok.pad_token = tok.eos_token
            except Exception:
                pass
        return proc

except ImportError:
    _HAS_TRANSFORMERS = False


def _get_pretrained_quant(model_id: str) -> str | None:
    if not _HAS_TRANSFORMERS:
        return None
    model_lower = model_id.lower()
    for hint in ("awq", "gptq", "nvfp4", "fp4", "squeezellm"):
        if hint in model_lower:
            return hint.upper()
    try:
        hf_cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        qcfg = getattr(hf_cfg, "quantization_config", None)
        if qcfg is not None:
            quant_type = (getattr(qcfg, "quant_type", None)
                          or getattr(qcfg, "quant_method", None))
            return str(quant_type).upper() if quant_type else "QUANTIZED"
    except Exception:
        pass
    return None


def _format_llava(context: str, query: str, **kw) -> str:
    parts = []
    if context:
        parts.append(context)
    parts.append(query)
    return "USER: <image>\n" + "\n".join(parts) + "\nASSISTANT:"


def _format_qwen3(context: str, query: str, **kw) -> list[dict]:
    content = [{"type": "image"}]
    text = ""
    if context:
        text += (
            "The following JSON contains verified sensor measurements from "
            "calibrated RGB-D cameras and a neural object detector (YOLO). "
            "Trust these numbers for size, distance, and position — "
            "do not override them with visual estimates.\n\n"
            f"{context}\n\n"
        )
    text += query
    content.append({"type": "text", "text": text})
    return [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "You are a scene understanding system for a robotic platform. "
                    "When sensor data (JSON) is provided in the user message, "
                    "treat it as ground truth. Be concise and factual. "
                    "Reference object IDs and distances from the JSON directly."
                ),
            }],
        },
        {"role": "user", "content": content},
    ]


def _format_cosmos(context: str, query: str, **kw) -> list[dict]:
    text = ""
    if context:
        text += f"Verified scene data:\n{context}\n\n"
    text += f"{query}\n\n<think>\nYour reasoning.\n</think>"
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a physical AI reasoning system."}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": text},
        ]},
    ]


def _format_chatml(context: str, query: str, **kw) -> str:
    text = ""
    if context:
        text += f"{context}\n\n"
    text += query
    return f"<|im_start|>user\n<image>\n{text}<|im_end|>\n<|im_start|>assistant\n"


def _format_raw(context: str, query: str, **kw) -> str:
    if context:
        return f"{context}\n\n{query}"
    return query


PROMPT_FORMATS: dict[str, Callable] = {
    "llava": _format_llava,
    "qwen3": _format_qwen3,
    "cosmos": _format_cosmos,
    "chatml": _format_chatml,
    "raw": _format_raw,
}


def get_prompt_formatter(format_name: str) -> Callable:
    fmt = PROMPT_FORMATS.get(format_name)
    if fmt is None:
        logger.warning("Unknown prompt format '%s' — using 'raw'", format_name)
        return PROMPT_FORMATS["raw"]
    return fmt


_ARCH_MAP: dict[str, str] = {
    "Qwen3VLMoeForConditionalGeneration": "Qwen3VLMoeForConditionalGeneration",
    "Qwen3VLForConditionalGeneration": "Qwen3VLForConditionalGeneration",
    "LlavaForConditionalGeneration": "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration": "LlavaNextForConditionalGeneration",
    "LlavaOnevisionForConditionalGeneration": "LlavaOnevisionForConditionalGeneration",
    "MllamaForConditionalGeneration": "MllamaForConditionalGeneration",
}


def auto_detect_model_class(model_id: str) -> Any:
    if not _HAS_TRANSFORMERS:
        return None
    try:
        hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        architectures = getattr(hf_config, "architectures", []) or []
        for arch in architectures:
            if arch in _ARCH_MAP:
                import transformers
                cls = getattr(transformers, _ARCH_MAP[arch], None)
                if cls is not None:
                    logger.info("Auto-detected model class: %s", arch)
                    return cls
        logger.info("No specific class for %s — using AutoModelForCausalLM", model_id)
        return AutoModelForCausalLM
    except Exception as e:
        logger.warning("Auto-detect failed for %s: %s", model_id, e)
        return AutoModelForCausalLM


def load_vlm(config: Any) -> tuple[Any, Any, Callable]:
    if not (_HAS_TORCH and _HAS_TRANSFORMERS):
        logger.error("torch/transformers not available — VLM cannot load")
        return None, None, get_prompt_formatter(config.vlm_prompt_format)

    model_id = config.vlm_model
    formatter = get_prompt_formatter(config.vlm_prompt_format)
    backend = detect_backend(config.quant_backend)

    try:
        logger.info("Loading VLM: %s (backend=%s, strategy=%s)",
                     model_id, backend, config.quant_vlm_strategy)

        model_class = auto_detect_model_class(model_id)
        if model_class is None:
            return None, None, formatter

        device = getattr(config, "device", "cuda:0")

        pretrained_quant = _get_pretrained_quant(model_id)
        if pretrained_quant:
            logger.info("[VLM] Pre-quantized checkpoint detected (%s)", pretrained_quant)
            if pretrained_quant.upper() == "AWQ":
                _backends = []
                try:
                    import gptqmodel
                    _backends.append(f"gptqmodel v{getattr(gptqmodel, '__version__', '?')}")
                except ImportError:
                    pass
                try:
                    import awq
                    _backends.append(f"autoawq v{getattr(awq, '__version__', '?')}")
                except ImportError:
                    pass
                if _backends:
                    logger.info("[VLM] AWQ backends available: %s", ", ".join(_backends))
                else:
                    logger.warning("[VLM] No AWQ backend importable")
            gc.collect()
            if _HAS_TORCH:
                torch.cuda.empty_cache()
            _log_mem("before-vlm-load")
            _local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

            pq_kwargs_fast: dict[str, Any] = {
                "trust_remote_code": True,
                "dtype": torch.bfloat16,
                "local_files_only": _local_only,
                "device_map": {"": device},
                "low_cpu_mem_usage": True,
            }

            model = None
            t_load_start = time.perf_counter()
            try:
                logger.info("[VLM] PHASE 1 — fast path (device_map streaming)")
                model = _safe_from_pretrained(model_class, model_id, **pq_kwargs_fast)
                logger.info("[VLM] PHASE 1 succeeded in %.1fs",
                            time.perf_counter() - t_load_start)
            except (NotImplementedError, RuntimeError, ValueError) as fast_err:
                _msg = str(fast_err).lower()
                _is_meta_err = (
                    "meta tensor" in _msg
                    or "cannot be called on meta" in _msg
                    or "meta device" in _msg
                )
                if not _is_meta_err:
                    raise
                logger.warning("[VLM] PHASE 1 failed (%s) — falling back to PHASE 2", fast_err)
                model = None
                gc.collect()
                if _HAS_TORCH:
                    torch.cuda.empty_cache()

                pq_kwargs_slow: dict[str, Any] = {
                    "trust_remote_code": True,
                    "dtype": torch.bfloat16,
                    "local_files_only": _local_only,
                }

                @_cl.contextmanager
                def _noop_iew(*args, **kwargs):
                    yield

                _iew_locs = (
                    "transformers.modeling_utils",
                    "accelerate",
                    "accelerate.utils",
                    "accelerate.utils.modeling",
                    "transformers.integrations.accelerate",
                )
                _iew_patches: list = []
                for _loc in _iew_locs:
                    try:
                        _mod = _il.import_module(_loc)
                        if hasattr(_mod, "init_empty_weights"):
                            _iew_patches.append(
                                (_mod, "init_empty_weights",
                                 getattr(_mod, "init_empty_weights"))
                            )
                            setattr(_mod, "init_empty_weights", _noop_iew)
                    except (ImportError, AttributeError):
                        pass
                t_phase2 = time.perf_counter()
                try:
                    model = _safe_from_pretrained(model_class, model_id, **pq_kwargs_slow)
                finally:
                    for _mod, _attr, _orig in _iew_patches:
                        try:
                            setattr(_mod, _attr, _orig)
                        except Exception:
                            pass
                _log_mem("after-vlm-load-cpu")
                logger.info("[VLM] Moving %s to %s", model_id, device)
                model = model.to(device)
                logger.info("[VLM] PHASE 2 succeeded in %.1fs",
                            time.perf_counter() - t_phase2)
            _log_mem("after-vlm-load")
            processor = _safe_load_processor(
                model_id, trust_remote_code=True, local_files_only=_local_only,
            )
            try:
                _seen = []
                for _name, _mod in model.named_modules():
                    _cls = type(_mod)
                    _qualname = f"{_cls.__module__}.{_cls.__name__}"
                    _lower = _qualname.lower()
                    if ("wqlinear" in _lower or "awqlinear" in _lower
                            or "qlinear" in _lower or "gptq" in _lower):
                        _seen.append((_name, _qualname))
                        if len(_seen) >= 1:
                            break
                if _seen:
                    logger.info("[VLM] Active quantized layer: %s → %s",
                                _seen[0][0], _seen[0][1])
                else:
                    logger.warning("[VLM] No quantized linear layer found")
            except Exception:
                pass
            logger.info("[VLM] %s loaded (%s)", model_id, pretrained_quant)
            return model, processor, formatter

        if getattr(config, "quant_cache_dir", None):
            _safe_vlm = model_id.replace("/", "_").replace("\\", "_")
            _bf16_src = Path(config.quant_cache_dir) / f"vlm_{_safe_vlm}_bf16"
            if _bf16_src.is_dir() and (_bf16_src / "config.json").exists():
                logger.info("[CACHE HIT] BF16 source cache: %s", _bf16_src)
                model = _safe_from_pretrained(
                    model_class,
                    str(_bf16_src),
                    trust_remote_code=True,
                    device_map={"": device},
                    low_cpu_mem_usage=True,
                    dtype=torch.bfloat16,
                )
                processor = _safe_load_processor(str(_bf16_src), trust_remote_code=True)
                logger.info("[VLM] BF16 cache load complete")
                return model, processor, formatter

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": {"": device},
            "low_cpu_mem_usage": True,
        }

        moe_by_pattern = any(p in model_id for p in ["-A3B", "-A17B", "-A14B", "-A7B", "MoE", "moe"])
        moe_by_class = "moe" in str(model_class).lower()
        moe_by_arch = False
        try:
            hf_cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            archs = getattr(hf_cfg, "architectures", []) or []
            moe_by_arch = any("moe" in a.lower() for a in archs)
        except Exception:
            pass

        is_moe = moe_by_pattern or moe_by_class or moe_by_arch
        logger.info(
            "MoE detection: pattern=%s class=%s arch=%s → is_moe=%s",
            moe_by_pattern, moe_by_class, moe_by_arch, is_moe,
        )

        if is_moe:
            logger.warning("[VLM] MoE model — skipping ModelOpt/bitsandbytes, loading in BF16")
            backend = "none"

            if getattr(config, "quant_cache_dir", None):
                _safe = model_id.replace("/", "_").replace("\\", "_")
                _bf16 = Path(config.quant_cache_dir) / f"vlm_{_safe}_bf16"
                if _bf16.is_dir() and (_bf16 / "config.json").exists():
                    logger.info("[CACHE HIT] BF16 MoE VLM: %s", _bf16)
                    try:
                        model = _safe_from_pretrained(
                            model_class,
                            str(_bf16),
                            trust_remote_code=True,
                            device_map={"": device},
                            low_cpu_mem_usage=True,
                        )
                        processor = _safe_load_processor(str(_bf16), trust_remote_code=True)
                        logger.info("[CACHE HIT] BF16 MoE VLM loaded")
                        return model, processor, formatter
                    except Exception as _e:
                        logger.warning("[CACHE MISS] BF16 cache load failed: %s", _e)

        quant_cache_path = None
        if getattr(config, "quant_cache_dir", None):
            safe_id = model_id.replace("/", "_").replace("\\", "_")
            strat = (config.quant_vlm_strategy or "nf4").replace("/", "_")
            quant_cache_path = Path(config.quant_cache_dir) / f"vlm_{safe_id}_{strat}"

        if quant_cache_path and quant_cache_path.is_dir() and (quant_cache_path / "config.json").exists():
            logger.info("[CACHE HIT] Pre-quantized VLM: %s", quant_cache_path)
            try:
                cached_kwargs = {
                    "trust_remote_code": True,
                    "device_map": {"": device},
                    "low_cpu_mem_usage": True,
                }
                cached_kwargs["dtype"] = (
                    torch.float16 if backend == "modelopt" else torch.bfloat16
                )
                model = _safe_from_pretrained(model_class, str(quant_cache_path), **cached_kwargs)
                if backend == "modelopt":
                    logger.info("[CACHE HIT] Re-attaching ModelOpt %s quantizers",
                                config.quant_vlm_strategy)
                    model = apply_modelopt_quantization(
                        model,
                        strategy=config.quant_vlm_strategy,
                        cache_dir=config.quant_cache_dir,
                        model_id=model_id,
                    )
                processor = _safe_load_processor(model_id, trust_remote_code=True)
                logger.info("[CACHE HIT] VLM ready")
                return model, processor, formatter
            except Exception as cache_err:
                logger.warning("[CACHE MISS] %s — re-quantizing", cache_err)

        gc.collect()
        if _HAS_TORCH:
            torch.cuda.empty_cache()
        _log_mem("before-vlm-load")

        if backend == "modelopt":
            load_kwargs["dtype"] = torch.float16
            logger.info("Loading VLM in FP16 for ModelOpt post-quantization")
            model = _safe_from_pretrained(model_class, model_id, **load_kwargs)
            model = apply_modelopt_quantization(
                model,
                strategy=config.quant_vlm_strategy,
                cache_dir=config.quant_cache_dir,
                model_id=model_id,
            )
        elif backend == "bitsandbytes":
            bnb_config = get_quantization_config(backend, config.quant_vlm_strategy)
            if bnb_config is not None:
                load_kwargs["quantization_config"] = bnb_config
            load_kwargs["dtype"] = torch.bfloat16
            logger.info("[VLM] Loading NF4 to %s", device)
            model = _safe_from_pretrained(model_class, model_id, **load_kwargs)
            _log_mem("after-vlm-bnb-load")
        else:
            load_kwargs["dtype"] = torch.bfloat16
            model = _safe_from_pretrained(model_class, model_id, **load_kwargs)

        if quant_cache_path is not None and backend != "bitsandbytes":
            try:
                quant_cache_path.mkdir(parents=True, exist_ok=True)
                logger.info("[CACHE SAVE] Saving quantized VLM to %s", quant_cache_path)
                model.save_pretrained(str(quant_cache_path))
                logger.info("[CACHE SAVE] Done")
            except Exception as save_err:
                logger.warning("[CACHE SAVE] Failed: %s", save_err)

        processor = _safe_load_processor(model_id, trust_remote_code=True)

        if is_moe and getattr(config, "quant_cache_dir", None):
            _safe = model_id.replace("/", "_").replace("\\", "_")
            _bf16 = Path(config.quant_cache_dir) / f"vlm_{_safe}_bf16"
            if not (_bf16 / "config.json").exists():
                try:
                    _bf16.mkdir(parents=True, exist_ok=True)
                    logger.info("[CACHE SAVE] Saving BF16 MoE VLM to %s", _bf16)
                    model.save_pretrained(str(_bf16))
                    processor.save_pretrained(str(_bf16))
                except Exception as _save_e:
                    logger.warning("[CACHE SAVE] BF16 save failed: %s", _save_e)

        logger.info("VLM loaded: %s", model_id)
        return model, processor, formatter

    except Exception as e:
        logger.error("VLM load FAILED: %s", e)
        return None, None, formatter


def load_physical(config: Any) -> tuple[Any, Any, Callable]:
    if not (_HAS_TORCH and _HAS_TRANSFORMERS):
        logger.error("torch/transformers not available — physical model cannot load")
        return None, None, get_prompt_formatter(config.physical_prompt_format)

    model_id = config.physical_model
    formatter = get_prompt_formatter(config.physical_prompt_format)
    backend = detect_backend(config.quant_backend)

    _vlm_id = (getattr(config, "vlm_model", "") or "").lower()
    _phys_id = model_id.lower()
    _vlm_is_qwen3_vl = ("qwen3-vl" in _vlm_id or "qwen3vl" in _vlm_id)
    _phys_is_qwen3_vl = ("qwen3-vl" in _phys_id or "qwen3vl" in _phys_id
                         or "cosmos-reason2" in _phys_id)
    _strategy = (getattr(config, "quant_physical_strategy", "") or "").lower()
    _modelopt_strategies = ("fp8", "nvfp4", "fp4")
    if (backend == "modelopt" and _strategy in _modelopt_strategies
            and _vlm_is_qwen3_vl and _phys_is_qwen3_vl):
        logger.warning(
            "[PHYS] Qwen3-VL class collision risk with physical_strategy=%s. "
            "Set physical_strategy=fp16 in pipeline.yaml.",
            _strategy,
        )

    try:
        logger.info("Loading physical model: %s (backend=%s, strategy=%s)",
                     model_id, backend, config.quant_physical_strategy)

        model_class = auto_detect_model_class(model_id)
        if model_class is None:
            return None, None, formatter

        device = getattr(config, "device", "cuda:0")
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": {"": device},
            "low_cpu_mem_usage": True,
        }

        phys_cache_path = None
        if getattr(config, "quant_cache_dir", None):
            safe_id = model_id.replace("/", "_").replace("\\", "_")
            strat = (config.quant_physical_strategy or "fp8").replace("/", "_")
            phys_cache_path = Path(config.quant_cache_dir) / f"physical_{safe_id}_{strat}"

        if phys_cache_path and phys_cache_path.is_dir() and (phys_cache_path / "config.json").exists():
            logger.info("[CACHE HIT] Physical: %s", phys_cache_path)
            try:
                cached_kwargs = {
                    "trust_remote_code": True,
                    "device_map": {"": device},
                    "low_cpu_mem_usage": True,
                }
                cached_kwargs["dtype"] = (
                    torch.float16 if backend == "modelopt" else torch.bfloat16
                )
                model = _safe_from_pretrained(model_class, str(phys_cache_path), **cached_kwargs)
                if backend == "modelopt":
                    logger.info("[CACHE HIT] Re-attaching ModelOpt %s quantizers",
                                config.quant_physical_strategy)
                    model = apply_modelopt_quantization(
                        model,
                        strategy=config.quant_physical_strategy,
                        cache_dir=config.quant_cache_dir,
                        model_id=model_id,
                    )
                processor = _safe_load_processor(model_id, trust_remote_code=True)
                logger.info("[CACHE HIT] Physical model ready")
                return model, processor, formatter
            except Exception as cache_err:
                logger.warning("[CACHE MISS] %s — re-quantizing", cache_err)

        gc.collect()
        if _HAS_TORCH:
            torch.cuda.empty_cache()
        _log_mem("before-phys-load")

        if backend == "modelopt":
            load_kwargs["dtype"] = torch.float16
            model = _safe_from_pretrained(model_class, model_id, **load_kwargs)
            model = apply_modelopt_quantization(
                model,
                strategy=config.quant_physical_strategy,
                cache_dir=config.quant_cache_dir,
                model_id=model_id,
            )
        elif backend == "bitsandbytes":
            bnb_config = get_quantization_config(backend, config.quant_physical_strategy)
            if bnb_config is not None:
                load_kwargs["quantization_config"] = bnb_config
            load_kwargs["dtype"] = torch.bfloat16
            logger.info("[PHYS] Loading NF4 to %s", device)
            model = _safe_from_pretrained(model_class, model_id, **load_kwargs)
            _log_mem("after-phys-bnb-load")
        else:
            load_kwargs["dtype"] = torch.bfloat16
            model = _safe_from_pretrained(model_class, model_id, **load_kwargs)

        if phys_cache_path is not None and backend != "bitsandbytes":
            try:
                phys_cache_path.mkdir(parents=True, exist_ok=True)
                logger.info("[CACHE SAVE] Saving quantized physical model to %s", phys_cache_path)
                model.save_pretrained(str(phys_cache_path))
                logger.info("[CACHE SAVE] Done")
            except Exception as save_err:
                logger.warning("[CACHE SAVE] Failed: %s", save_err)

        processor = _safe_load_processor(model_id, trust_remote_code=True)
        logger.info("Physical model loaded: %s", model_id)
        return model, processor, formatter

    except Exception as e:
        logger.error("Physical model load FAILED: %s", e)
        return None, None, formatter
