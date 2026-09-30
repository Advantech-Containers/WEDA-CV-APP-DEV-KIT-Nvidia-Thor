"""Structured logging with optional file output."""

from __future__ import annotations

import logging
import sys
from typing import Optional


_CONFIGURED = False


def setup_logger(
    name: str = "cv_pipeline",
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    global _CONFIGURED
    if _CONFIGURED:
        return logging.getLogger(name)

    level_int = getattr(logging, level.upper(), logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)s.%(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    # Configure both the app logger and weda_optim so GPU monitor messages appear
    for logger_name in (name, "weda_optim"):
        lg = logging.getLogger(logger_name)
        lg.setLevel(level_int)
        if not lg.handlers:
            lg.addHandler(sh)
            if log_file:
                fh = logging.FileHandler(log_file, mode="a")
                fh.setFormatter(fmt)
                lg.addHandler(fh)

    _CONFIGURED = True
    return logging.getLogger(name)


def get_logger(name: str = "cv_pipeline") -> logging.Logger:
    return logging.getLogger(name)


def silence_runtime_logs() -> None:
    """Raise all app/library loggers to ERROR for clean runtime output."""
    for name in ("cv_pipeline", "weda_optim"):
        logging.getLogger(name).setLevel(logging.ERROR)
    # Also suppress any straggler library loggers
    for name in ("transformers", "transformers.modeling_utils",
                 "transformers.configuration_utils", "transformers.tokenization_utils_base",
                 "accelerate", "safetensors", "huggingface_hub",
                 "torch", "tensorrt", "ultralytics"):
        logging.getLogger(name).setLevel(logging.ERROR)
    # Suppress transformers' own logging system (separate from stdlib logging)
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass
