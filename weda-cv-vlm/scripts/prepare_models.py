#!/usr/bin/env python3
"""One-time model preparation: download, quantize, and cache all pipeline models.

Run once before first application start (or inside the container on first boot):
    python scripts/prepare_models.py

What it does
------------
Step 1  VLM                         Download from HF → save to persistent cache.
        The cache directory and disk-size estimate are derived from the
        repo's own `config.json` `quantization_config`. Pre-quantized
        checkpoints (AWQ / GPTQ / FP4 / FP8 / NF4) are stored in their
        native format; unquantized repos are stored as BF16.
        Cache path: {cache_dir}/vlm_{safe_id}_{quant_kind}/

Step 2  Cosmos-Reason2-2B (FP8)      Download + quantize → save to persistent cache
        Reuses load_physical() which already saves the quantized checkpoint
        Cache path: {cache_dir}/physical_{safe_id}_fp8/

Step 3  Cosmos-Embed1-224p (BF16)    Download to HF hub cache + verify load
        No quantization (1.5 GB, fast to load as-is)

Step 4  YOLO26n-seg TRT engine       Export .pt → TRT .engine → save to cache dir
        Cache path: {cache_dir}/yolo/{stem}.engine

On all subsequent starts each model loads from the persistent path.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# Allow downloading — clear offline flags set by entrypoint.sh
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_id(model_id: str) -> str:
    return model_id.replace("/", "_").replace("\\", "_")


def _dir_size_gb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1e9


def _header(text: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}")


def _probe_vlm_quant(model_id: str) -> tuple[str, float, str]:
    """Inspect the HF repo to decide cache-dir suffix, expected size, and a
    human-readable description.

    Returns (suffix, expected_gb, description). On any failure falls back to
    ('bf16', 0.0, 'BF16 (unquantized — fallback after probe failure)').
    """
    try:
        import json
        from huggingface_hub import HfApi, hf_hub_download

        token = os.environ.get("HF_TOKEN")

        # 1. Pull config.json — small (a few KB) — to read quantization_config.
        cfg_path = hf_hub_download(model_id, "config.json", token=token)
        with open(cfg_path) as f:
            cfg = json.load(f)
        qc = cfg.get("quantization_config") or {}
        method = (qc.get("quant_method") or "").lower()
        bits = qc.get("bits")

        # 2. Sum the size of weight files via the HF API so we report the
        # actual repo size, not a guess.
        try:
            info = HfApi().repo_info(model_id, files_metadata=True, token=token)
            weight_bytes = sum(
                (s.size or 0) for s in info.siblings
                if s.rfilename.endswith((".safetensors", ".bin", ".pth"))
            )
            expected_gb = weight_bytes / 1e9
        except Exception:
            expected_gb = 0.0

        if method == "awq":
            return "awq", expected_gb, f"AWQ int{bits or 4} (BF16 activations)"
        if method == "gptq":
            return f"gptq{bits or 4}", expected_gb, f"GPTQ int{bits or 4} (BF16 activations)"
        if method in ("nvfp4", "fp4"):
            return "nvfp4", expected_gb, "NVFP4 (BF16 activations)"
        if method == "fp8":
            return "fp8", expected_gb, "FP8 weights + activations"
        if method in ("bnb_4bit", "nf4"):
            return "nf4", expected_gb, "bitsandbytes NF4 (BF16 activations)"
        if method:
            return method, expected_gb, f"{method} (BF16 activations)"
        return "bf16", expected_gb, "BF16 (unquantized)"
    except Exception as e:
        return "bf16", 0.0, f"BF16 (probe failed: {e})"


# ---------------------------------------------------------------------------
# Step 1 — Qwen3-VL-30B-A3B BF16
# ---------------------------------------------------------------------------

def prepare_vlm(config, force: bool = False) -> bool:
    model_id = config.vlm_model
    quant_kind, expected_gb, quant_desc = _probe_vlm_quant(model_id)
    cache_path = Path(config.quant_cache_dir) / f"vlm_{_safe_id(model_id)}_{quant_kind}"
    marker = cache_path / "config.json"

    _header(f"Step 1 — VLM: {model_id}")
    print(f"  Format:    {quant_desc}")
    if expected_gb > 0:
        print(f"  Size:      ~{expected_gb:.1f} GB")
    print(f"  Cache:     {cache_path}")

    if marker.exists() and not force:
        print(f"  [SKIP] Already cached. Pass --force to re-download.")
        return True

    t0 = time.time()
    try:
        import torch
        from transformers import AutoProcessor
        from weda_optim.model_loader import auto_detect_model_class

        model_class = auto_detect_model_class(model_id)
        if model_class is None:
            print("  [ERROR] Could not detect model class — check model ID in pipeline.yaml")
            return False

        # Use the shared transformers-5.x compat helper so any model class
        # (Nemotron-VL, Cosmos, InternVL, Phi-VL, …) loads with the same
        # FA2-eager fallback + `all_tied_weights_keys` shim applied at
        # runtime. `dtype=bfloat16` is the activation dtype for everything
        # (quantized layers stay packed; the kernel dequantizes per-layer
        # at compute time).
        from weda_optim.model_loader import _safe_from_pretrained
        print(f"  Loading {model_class.__name__} (CPU staging)...")
        model = _safe_from_pretrained(
            model_class,
            model_id,
            dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        cache_path.mkdir(parents=True, exist_ok=True)
        print(f"  Saving to {cache_path}")
        model.save_pretrained(str(cache_path))
        processor.save_pretrained(str(cache_path))

        del model
        elapsed = time.time() - t0
        gb = _dir_size_gb(cache_path)
        print(f"  [DONE] {gb:.1f} GB saved in {elapsed / 60:.1f} min")
        return True

    except Exception as e:
        import traceback
        print(f"  [ERROR] VLM preparation failed: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Step 2 — Cosmos-Reason2-2B FP8
# ---------------------------------------------------------------------------

def prepare_physical(config, force: bool = False) -> bool:
    model_id = config.physical_model
    strat = (config.quant_physical_strategy or "fp8").replace("/", "_")
    is_passthrough = strat.lower() in ("fp16", "bf16", "f16", "bfloat16", "float16")
    cache_path = Path(config.quant_cache_dir) / f"physical_{_safe_id(model_id)}_{strat}"
    marker = cache_path / "config.json"

    _header(f"Step 2 — Cosmos-Reason2 {strat.upper()}: {model_id}")

    if is_passthrough:
        # fp16/bf16 strategies are passthrough — the runtime loader pulls
        # the model from the HF cache as-is, no quantized checkpoint exists.
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(model_id, "config.json")
        except Exception:
            cached = None
        if cached and not force:
            print(f"  [SKIP] {strat.upper()} passthrough — already in HF cache: {model_id}")
            return True
        print(f"  Downloading {model_id} to HF cache ({strat.upper()} passthrough — no quantization)...")
        t0 = time.time()
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model_id,
                local_files_only=False,
                ignore_patterns=["*.msgpack", "flax_model*", "rust_model*"],
            )
            print(f"  Downloaded in {time.time() - t0:.0f} sec")
            # Verify load via the runtime loader so any config/quantizer
            # init problems surface here rather than at app start.
            from weda_optim.model_loader import load_physical
            model, _, _ = load_physical(config)
            if model is None:
                print("  [ERROR] load_physical returned None — check logs above")
                return False
            del model
            print(f"  [DONE] {strat.upper()} verified — runtime loads from HF cache")
            return True
        except Exception as e:
            import traceback
            print(f"  [ERROR] Physical model preparation failed: {e}")
            traceback.print_exc()
            return False

    # Actual quantization path (fp8 / nvfp4 / nf4) — load_physical saves
    # a quantized checkpoint to cache_path on first run.
    if marker.exists() and not force:
        print(f"  [SKIP] Already quantized + cached at {cache_path}")
        return True

    print(f"  Downloading + quantizing to {strat.upper()} → {cache_path}")
    print("  First run: ~5-15 min. Subsequent starts load pre-quantized checkpoint in ~3-5 sec.")

    t0 = time.time()
    try:
        from weda_optim.model_loader import load_physical
        model, _, _ = load_physical(config)
        if model is None:
            print("  [ERROR] load_physical returned None — check initialization logs above")
            return False
        del model
        elapsed = time.time() - t0
        if marker.exists():
            gb = _dir_size_gb(cache_path)
            print(f"  [DONE] {gb:.1f} GB saved in {elapsed / 60:.1f} min → {cache_path}")
            return True
        print(f"  [WARN] Model loaded but quantized checkpoint not saved at {cache_path}")
        print("         The runtime will re-quantize on next start.")
        return True
    except Exception as e:
        import traceback
        print(f"  [ERROR] Physical model preparation failed: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Step 3 — Cosmos-Embed1-224p
# ---------------------------------------------------------------------------

def prepare_embedding(config, force: bool = False) -> bool:
    model_id = config.embedding_model
    revision = getattr(config, "embedding_revision", None) or None

    _header(f"Step 3 — Cosmos-Embed1: {model_id}")

    # Check HF cache presence
    try:
        from huggingface_hub import snapshot_download, try_to_load_from_cache
        cached = try_to_load_from_cache(model_id, "config.json", revision=revision)
    except Exception:
        cached = None

    if cached and not force:
        print(f"  [SKIP] Already in HF cache: {model_id}")
    else:
        print(f"  Downloading {model_id} (~1.5 GB)...")
        t0 = time.time()
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_files_only=False,
                ignore_patterns=["*.msgpack", "flax_model*", "rust_model*"],
            )
            print(f"  Downloaded in {time.time() - t0:.0f} sec")
        except Exception as e:
            print(f"  [ERROR] Download failed: {e}")
            return False

    # Verify load
    print("  Verifying load...")
    try:
        from app.modules.embedder import Embedder
        e = Embedder(config)
        e.initialize()
        if e._model is None:
            print("  [WARN] Embedder loaded but model is None — check logs")
        else:
            print("  [DONE] Cosmos-Embed1 verified")
        e.shutdown()
        return True
    except Exception as e:
        print(f"  [ERROR] Verify failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Step 4 — YOLO TRT engine
# ---------------------------------------------------------------------------

def prepare_yolo(config, force: bool = False) -> bool:
    try:
        from ultralytics import YOLO as _YOLO
        _HAS_UL = True
    except ImportError:
        _HAS_UL = False

    model_path = Path(config.yolo_model)
    yolo_cache = Path(config.quant_cache_dir) / "yolo"
    engine_dest = yolo_cache / model_path.with_suffix(".engine").name

    _header(f"Step 4 — YOLO TRT engine: {config.yolo_model}")

    if engine_dest.exists() and not force:
        print(f"  [SKIP] Pre-exported engine found: {engine_dest}")
        return True

    if not _HAS_UL:
        print("  [SKIP] ultralytics not installed — skipping TRT export")
        return True

    if not getattr(config, "use_tensorrt", False):
        print("  [SKIP] use_tensorrt=false in config — no engine needed")
        return True

    # Ensure we load from a .pt file because ultralytics cannot download an .engine file
    load_path = model_path
    if load_path.suffix == ".engine":
        load_path = load_path.with_suffix(".pt")

    print(f"  Exporting TRT engine from {load_path} (~90 sec first run)...")
    t0 = time.time()
    try:
        model = _YOLO(str(load_path))
        exported = model.export(
            format="engine",
            half=True,
            device=getattr(config, "device", "cuda:0"),
            imgsz=getattr(config, "det_input_size", 640),
        )
        yolo_cache.mkdir(parents=True, exist_ok=True)
        src = Path(exported) if isinstance(exported, str) else model_path.with_suffix(".engine")
        if src.exists():
            shutil.copy2(str(src), str(engine_dest))
            # Write path marker so detector.py can find it
            (yolo_cache / "engine_path.txt").write_text(str(engine_dest))
            print(f"  [DONE] Engine saved in {time.time() - t0:.0f} sec → {engine_dest}")
            return True
        else:
            print(f"  [WARN] Engine export completed but source not found at {src}")
            return False
    except Exception as e:
        import traceback
        print(f"  [ERROR] TRT export failed: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def dry_run(config) -> None:
    print("\n  Model cache status:\n")

    # VLM
    vlm_path = Path(config.quant_cache_dir) / f"vlm_{_safe_id(config.vlm_model)}_bf16"
    status = "READY" if (vlm_path / "config.json").exists() else "MISSING"
    print(f"  [{status:7s}] Qwen3-VL BF16      {vlm_path}")

    # Physical
    strat = (config.quant_physical_strategy or "fp8").replace("/", "_")
    phys_path = Path(config.quant_cache_dir) / f"physical_{_safe_id(config.physical_model)}_{strat}"
    status = "READY" if (phys_path / "config.json").exists() else "MISSING"
    print(f"  [{status:7s}] Cosmos-Reason2 {strat.upper():4s} {phys_path}")

    # Embedding (HF cache)
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(config.embedding_model, "config.json")
        status = "READY" if cached else "MISSING"
    except Exception:
        status = "UNKNOWN"
    print(f"  [{status:7s}] Cosmos-Embed1      HF hub cache: {config.embedding_model}")

    # YOLO
    yolo_engine = Path(config.quant_cache_dir) / "yolo" / Path(config.yolo_model).with_suffix(".engine").name
    status = "READY" if yolo_engine.exists() else "MISSING"
    print(f"  [{status:7s}] YOLO TRT engine    {yolo_engine}")

    print()


# ---------------------------------------------------------------------------
# Pose assets — small files, fetched directly. Folded into this script
# so model-prep is a single config-driven entry point.
# ---------------------------------------------------------------------------

_YOLO_POSE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt"


def _http_fetch(url: str, dest: Path, force: bool = False) -> bool:
    if dest.exists() and not force:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f, length=1024 * 1024)
        return dest.exists() and dest.stat().st_size > 0
    except Exception as e:
        print(f"  [WARN] {url}\n         → {dest}\n         {e}")
        # Best-effort; partial downloads are removed so retry isn't blocked.
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        return False


def prepare_assets(config, repo_root: Path, force: bool = False) -> bool:
    print(f"\n{'═' * 60}\n  STEP 5: Pose assets\n{'═' * 60}")

    yolo_pose_dst = Path(config.quant_cache_dir) / "yolo" / "yolo26n-pose.pt"
    print(f"  YOLOv26-pose→ {yolo_pose_dst}")
    yolo_pose_ok = _http_fetch(_YOLO_POSE_URL, yolo_pose_dst, force=force)
    print(f"  [{'OK' if yolo_pose_ok else 'WARN'}] yolo26n-pose")

    # Returning True even on partial misses — pose is an optional asset;
    # the pipeline degrades gracefully when it's absent so it shouldn't
    # gate the whole prep run.
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_config = str(repo_root / "config" / "pipeline.yaml")
    ap = argparse.ArgumentParser(
        description="Prepare (download/quantize/cache) all pipeline models + pose assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config",          default=default_config, help="Path to pipeline.yaml (default: <repo>/config/pipeline.yaml)")
    ap.add_argument("--cache-dir",       default=None,  help="Override quantization.cache_dir from config")
    ap.add_argument("--skip-vlm",        action="store_true", help="Skip VLM download")
    ap.add_argument("--skip-physical",   action="store_true", help="Skip physical-reasoner download")
    ap.add_argument("--skip-embedding",  action="store_true", help="Skip embedder download")
    ap.add_argument("--skip-yolo",       action="store_true", help="Skip YOLO TRT export")
    ap.add_argument("--skip-assets",     action="store_true", help="Skip pose assets (yolo26n-pose)")
    ap.add_argument("--force",           action="store_true", help="Re-run steps even if cache exists")
    ap.add_argument("--dry-run",         action="store_true", help="Print cache status and exit")
    args = ap.parse_args()

    from app.config import load_config
    config = load_config(args.config)
    if args.cache_dir:
        config.quant_cache_dir = args.cache_dir

    # Resolve cache to an absolute path against the repo root when the
    # config value is relative. Avoids the read-only-fs trap when the
    # script is run from a working directory mounted ro:.
    cache = Path(config.quant_cache_dir)
    if not cache.is_absolute():
        cache = (repo_root / cache).resolve()
        config.quant_cache_dir = str(cache)
    cache.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        dry_run(config)
        return

    results: dict[str, bool] = {}

    if not args.skip_vlm:
        results["vlm"] = prepare_vlm(config, args.force)
    if not args.skip_physical:
        results["physical"] = prepare_physical(config, args.force)
    if not args.skip_embedding:
        results["embedding"] = prepare_embedding(config, args.force)
    if not args.skip_yolo:
        results["yolo"] = prepare_yolo(config, args.force)
    if not args.skip_assets:
        results["assets"] = prepare_assets(config, repo_root, args.force)

    # Write global marker only when all requested steps succeeded
    if results and all(results.values()):
        marker = Path(config.quant_cache_dir) / ".prep_complete"
        import json
        marker.write_text(json.dumps({
            "models": list(results.keys()),
            "vlm_model": config.vlm_model,
            "physical_model": config.physical_model,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2))
        print(f"\n{'═' * 60}")
        print("  All steps complete. Marker written:")
        print(f"  {marker}")
        print("  Start the application — all models load from persistent cache.")
        print(f"{'═' * 60}\n")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n  [WARN] Some steps failed: {failed}")
        print("  Re-run or check logs above. Application may still work but will re-download on next start.\n")


if __name__ == "__main__":
    main()
