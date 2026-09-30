"""Unified quantization API with NVIDIA ModelOpt + bitsandbytes fallback.

Priority chain:
  1. NVIDIA ModelOpt NVFP4/FP8 (Jetson Thor production)
  2. bitsandbytes NF4/INT8 (local dev with GPU)
  3. No quantization — FP16/BF16 (CPU dev)

Cached calibration: quantized checkpoints saved to disk after first
calibration run, loaded directly on subsequent starts.
"""

from __future__ import annotations

from typing import Any, Optional

import logging

logger = logging.getLogger("weda_optim")

# Probe available backends
try:
    import modelopt.torch.quantization as mtq
    _HAS_MODELOPT = True
except ImportError:
    _HAS_MODELOPT = False

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ── Strategy configs ──────────────────────────────────────────────────

MODELOPT_STRATEGIES = {
    "nvfp4": lambda: _modelopt_cfg(mtq.NVFP4_DEFAULT_CFG, mtq.NVFP4_KV_CFG),
    "fp8": lambda: _modelopt_cfg(mtq.FP8_DEFAULT_CFG, mtq.FP8_KV_CFG),
    "fp8_no_kv": lambda: mtq.FP8_DEFAULT_CFG.copy(),
}

BNB_STRATEGIES = {
    "nf4": lambda: BitsAndBytesConfig(
        load_in_4bit=True,
        # bfloat16 matches the native dtype of modern LLMs (e.g. Qwen3-VL config
        # has "dtype": "bfloat16").  float16 would cause implicit dtype casts
        # during quantization that add memory pressure.
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        # Double quantization: additionally quantize the NF4 quantization
        # constants themselves, saving ~0.4 bits/param (~1.5 GB on a 30B model).
        bnb_4bit_use_double_quant=True,
        # uint8 storage: pack two 4-bit values per byte immediately on load.
        # Without this, newer bitsandbytes versions may defer quantization and
        # store weights in bfloat16 (lazy mode) until first forward pass —
        # doubling peak memory during loading (60 GB instead of 15 GB for 30B).
        bnb_4bit_quant_storage=torch.uint8,
    ),
    "int8": lambda: BitsAndBytesConfig(load_in_8bit=True),
}


def _modelopt_cfg(weight_cfg, kv_cfg):
    cfg = weight_cfg.copy()
    cfg["quant_cfg"].update(kv_cfg["quant_cfg"])
    return cfg


# ── Public API ────────────────────────────────────────────────────────

def detect_backend(preferred: str = "auto") -> str:
    """Detect best available quantization backend.

    Returns: "modelopt" | "bitsandbytes" | "none"
    """
    if preferred == "modelopt":
        return "modelopt" if _HAS_MODELOPT else "none"
    if preferred == "bitsandbytes":
        return "bitsandbytes" if _HAS_BNB else "none"
    if preferred == "none":
        return "none"

    # auto
    if _HAS_MODELOPT:
        return "modelopt"
    if _HAS_BNB:
        return "bitsandbytes"
    return "none"


def get_quantization_config(
    backend: str,
    strategy: str,
) -> Optional[Any]:
    """Get a quantization config object for model loading.

    For bitsandbytes: returns BitsAndBytesConfig (pass to from_pretrained).
    For modelopt: returns None (quantization applied post-load).
    For none: returns None.
    """
    if backend == "bitsandbytes" and _HAS_BNB:
        bnb_strategy = _map_to_bnb(strategy)
        factory = BNB_STRATEGIES.get(bnb_strategy)
        if factory:
            return factory()
    return None


def apply_modelopt_quantization(
    model: Any,
    strategy: str,
    calibration_fn: Optional[Any] = None,
    cache_dir: Optional[str] = None,
    model_id: str = "",
) -> Any:
    """Apply NVIDIA ModelOpt quantization post-load.

    Args:
        model: loaded PyTorch model
        strategy: "nvfp4" | "fp8"
        calibration_fn: callable(model) for calibration forward pass
        cache_dir: unused (caching handled in model_loader via save_pretrained)
        model_id: unused (caching handled in model_loader via save_pretrained)
    """
    if not _HAS_MODELOPT:
        logger.warning("ModelOpt not available — skipping quantization")
        return model

    cfg_factory = MODELOPT_STRATEGIES.get(strategy)
    if cfg_factory is None:
        logger.warning("Unknown ModelOpt strategy '%s' — skipping", strategy)
        return model

    quant_cfg = cfg_factory()

    if calibration_fn is None:
        def calibration_fn(m):
            logger.info("No calibration data — using identity calibration")

    logger.info("Applying ModelOpt %s quantization...", strategy)
    model = mtq.quantize(model, quant_cfg, calibration_fn)
    return model


# ── Helpers ───────────────────────────────────────────────────────────

def _map_to_bnb(strategy: str) -> str:
    """Map generic strategy names to bitsandbytes equivalents."""
    return {
        "nvfp4": "nf4",
        "nf4": "nf4",
        "fp8": "int8",
        "int8": "int8",
    }.get(strategy, "nf4")
