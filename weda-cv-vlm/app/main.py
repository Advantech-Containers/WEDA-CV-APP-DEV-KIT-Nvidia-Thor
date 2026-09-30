"""CLI entry point for the Jetson Thor CV Pipeline.

Configuration is loaded from YAML (config/pipeline.yaml by default).
CLI arguments serve as quick overrides for the most common settings.
"""

from __future__ import annotations

# Suppress known-harmless library warnings before imports
import os
import logging
import warnings
os.environ["ORT_LOG_LEVEL"] = "ERROR"
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["NO_AT_BRIDGE"] = "1"         # suppress GTK accessibility bus warning
os.environ["ULTRALYTICS_VERBOSE"] = "0"  # suppress Ultralytics YOLO logging
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*subnormal.*")
warnings.filterwarnings("ignore", message=".*not tested with nvidia-modelopt.*")
warnings.filterwarnings("ignore", message=".*triton kernel.*")
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
warnings.filterwarnings("ignore", message=".*tied weights.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*engine plan file across different.*")
warnings.filterwarnings("ignore", message=".*Skipping import of cpp extensions.*")
# Suppress verbose library loggers
for _lib in ("transformers", "accelerate", "safetensors", "huggingface_hub",
             "torch", "tensorrt", "ultralytics"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

# ── Redirect stdout + stderr to suppress C++/extension noise during imports ──
import sys as _sys, ctypes as _ct
_sys.stdout.flush()
_sys.stderr.flush()
try:
    _ct.CDLL(None).fflush(None)      # flush C stdio buffers too
except Exception:
    pass
_saved_stdout_fd = os.dup(1)
_saved_stderr_fd = os.dup(2)
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull_fd, 1)              # suppress stdout ("load extensions...")
os.dup2(_devnull_fd, 2)              # suppress stderr (TRT C++ logs)
os.close(_devnull_fd)

import argparse
from pathlib import Path
from typing import Optional

from app.config import load_config
from app.core.orchestrator import Orchestrator

# ── Restore stdout + stderr now that noisy imports are done ──
try:
    _ct.CDLL(None).fflush(None)      # flush any buffered C output to /dev/null
except Exception:
    pass
os.dup2(_saved_stdout_fd, 1)
os.close(_saved_stdout_fd)
os.dup2(_saved_stderr_fd, 2)
os.close(_saved_stderr_fd)


def parse_args(argv: Optional[list[str]] = None) -> tuple:
    """Parse CLI args → (yaml_path, overrides dict)."""
    p = argparse.ArgumentParser(
        description="Jetson Thor Optimized CV Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Config file
    p.add_argument("--config", default="config/pipeline.yaml",
                   help="Path to YAML config file")

    # Quick overrides (most common settings)
    p.add_argument("--source", default=None,
                   help="Camera index, video path, rtsp:// URL, or 'synthetic'")
    p.add_argument("--headless", action="store_true", default=None,
                   help="Disable display window")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after N frames (0 = forever)")
    p.add_argument("--device", default=None,
                   help="CUDA device (e.g. cuda:0)")
    p.add_argument("--enable-gpu-monitor", action="store_true", default=None,
                   help="Enable real-time GPU monitoring")
    p.add_argument("--log-level", default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--profile-interval", type=int, default=None,
                   help="Log profiling every N frames")

    args = p.parse_args(argv)

    # Build overrides dict (only non-None values)
    overrides = {}
    if args.source is not None:
        overrides["source"] = args.source
    if args.headless is not None and args.headless:
        overrides["headless"] = True
    if args.max_frames is not None:
        overrides["max_frames"] = args.max_frames
    if args.device is not None:
        overrides["device"] = args.device
    if args.enable_gpu_monitor is not None and args.enable_gpu_monitor:
        overrides["enable_gpu_monitor"] = True
    if args.log_level is not None:
        overrides["log_level"] = args.log_level
    if args.profile_interval is not None:
        overrides["profile_interval"] = args.profile_interval

    return args.config, overrides


def main(argv: Optional[list[str]] = None) -> None:
    yaml_path, overrides = parse_args(argv)
    config = load_config(yaml_path, cli_overrides=overrides)
    orchestrator = Orchestrator(config)
    orchestrator.run()


if __name__ == "__main__":
    main()
