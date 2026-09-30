"""YAML configuration loader.

Merges three layers (lowest → highest priority):
  1. Hardcoded defaults (PipelineConfig dataclass)
  2. YAML config file
  3. CLI overrides (dict of non-None values)

Usage:
    config = load_config("config/pipeline.yaml", cli_overrides={"source": "0"})
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from app.utils.data_structures import PipelineConfig
from app.utils.logger import get_logger

logger = get_logger()

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# Mapping from flat YAML paths to PipelineConfig field names
_YAML_MAP: dict[tuple[str, ...], str] = {
    ("input", "backend"): "input_backend",
    ("input", "source"): "source",
    ("input", "loop"): "input_loop",
    ("input", "width"): "width",
    ("input", "height"): "height",
    ("input", "rtsp", "protocol"): "rtsp_protocol",
    ("input", "rtsp", "reconnect_attempts"): "rtsp_reconnect_attempts",
    ("input", "rtsp", "timeout_sec"): "rtsp_timeout",
    ("detection", "model"): "yolo_model",
    ("detection", "confidence"): "det_conf",
    ("detection", "use_tensorrt"): "use_tensorrt",
    ("detection", "device"): "device",
    ("embedding", "enabled"): "enable_embedding",
    ("embedding", "model"): "embedding_model",
    ("embedding", "revision"): "embedding_revision",
    ("embedding", "dim"): "embedding_dim",
    ("embedding", "scene_change_threshold"): "scene_change_threshold",
    ("physical_reasoning", "enabled"): "enable_physical_reasoning",
    ("physical_reasoning", "model"): "physical_model",
    ("physical_reasoning", "cooldown_sec"): "physical_cooldown",
    ("detection", "input_size"): "det_input_size",
    ("vlm", "enabled"): "enable_vlm",
    ("vlm", "model"): "vlm_model",
    ("vlm", "cooldown_sec"): "vlm_cooldown",
    ("vlm", "prompt_format"): "vlm_prompt_format",
    ("vlm", "max_tokens"): "vlm_max_tokens",
    ("physical_reasoning", "prompt_format"): "physical_prompt_format",
    ("physical_reasoning", "max_tokens"): "physical_max_tokens",
    ("quantization", "backend"): "quant_backend",
    ("quantization", "vlm_strategy"): "quant_vlm_strategy",
    ("quantization", "vlm_kv_cache"): "quant_vlm_kv_cache",
    ("quantization", "physical_strategy"): "quant_physical_strategy",
    ("quantization", "physical_kv_cache"): "quant_physical_kv_cache",
    ("quantization", "cache_dir"): "quant_cache_dir",
    ("quantization", "calibration_samples"): "quant_calibration_samples",
    ("input", "orbbec", "align_depth"): "orbbec_align_depth",
    ("input", "orbbec", "depth_min_mm"): "orbbec_depth_min_mm",
    ("input", "orbbec", "depth_max_mm"): "orbbec_depth_max_mm",
    ("optimization", "vic_offload", "enabled"): "enable_vic_offload",
    ("optimization", "cuda_cache_config", "depth_prefer_l1"): "depth_prefer_l1",
    ("optimization", "buffer_pool", "depth"): "pool_depth",
    ("optimization", "scheduler", "gpu_util_threshold"): "gpu_util_threshold",
    ("optimization", "scheduler", "max_concurrent_gpu"): "max_concurrent_gpu",
    ("optimization", "scheduler", "speedup_min"): "scheduler_speedup_min",
    ("optimization", "gpu_monitor", "enabled"): "enable_gpu_monitor",
    ("optimization", "gpu_monitor", "interval_sec"): "gpu_monitor_interval",
    ("profiling", "interval_frames"): "profile_interval",
    ("profiling", "window_size"): "profiler_window",
    ("display", "headless"): "headless",
    ("display", "max_frames"): "max_frames",
    ("logging", "level"): "log_level",
    ("logging", "file"): "log_file",
    # Tracking parameters
    ("tracking", "min_display_age"):      "tracker_min_display_age",
    ("tracking", "iou_threshold"):        "tracker_iou_threshold",
    ("tracking", "max_age_sec"):          "tracker_max_age_sec",
    ("tracking", "high_thresh"):          "tracker_high_thresh",
    ("tracking", "embedding_threshold"): "tracker_embedding_threshold",
    ("tracking", "dormancy_sec"):         "tracker_dormancy_sec",
    # Navigation (Potential Field Path Planner)
    ("navigation", "enabled"):          "enable_navigation",
    ("navigation", "attractive_gain"):  "nav_attractive_gain",
    ("navigation", "repulsive_gain"):   "nav_repulsive_gain",
    ("navigation", "obstacle_radius"):  "nav_obstacle_radius",
    # Safety zone (proximity alert)
    ("safety", "zone_radius_m"):        "safety_zone_radius",
    ("safety", "dwell_alert_seconds"):  "dwell_alert_seconds",
    ("safety", "alert_classes"):        "safety_alert_classes",
    ("safety", "enter_frames"):         "safety_alert_enter_frames",
    ("safety", "exit_frames"):          "safety_alert_exit_frames",
    # ── HTTP API
    ("api", "enable"):                  "api_enable",
    ("api", "bind"):                    "api_bind",
    ("api", "port"):                    "api_port",
    ("api", "token_env"):               "api_token_env",
    # ── Innovation pillars
    ("innovation", "fact_check", "enable"):                 "fact_check_enable",
    ("innovation", "fact_check", "distance_tolerance_pct"): "fact_check_distance_tolerance_pct",
    ("innovation", "fact_check", "size_tolerance_pct"):     "fact_check_size_tolerance_pct",
    ("innovation", "scene_memory", "enable"):               "scene_memory_enable",
    ("innovation", "scene_memory", "window_min"):           "scene_memory_window_min",
    ("innovation", "scene_memory", "sample_hz"):            "scene_memory_sample_hz",
    ("innovation", "predictive_safety", "enable"):          "predictive_safety_enable",
    ("innovation", "predictive_safety", "horizon_s"):       "predictive_safety_horizon_s",
    ("innovation", "predictive_safety", "critical_eta_s"):  "predictive_safety_critical_eta_s",
    ("innovation", "predictive_safety", "danger_distance_m"): "predictive_safety_danger_distance_m",
    ("innovation", "alarm_engine", "enable"):               "alarm_engine_enable",
    ("innovation", "alarm_engine", "persist_path"):         "alarm_engine_persist_path",
    # ── MJPEG streaming (REST /v1/stream/*.mjpg)
    ("streaming", "mjpeg_fps_cap"):      "stream_mjpeg_fps_cap",
    ("streaming", "jpeg_quality"):       "stream_jpeg_quality",
    ("streaming", "draw_detections"):    "stream_draw_detections",
    # ── Telemetry
    ("telemetry", "enable"):             "telemetry_enable",
    ("telemetry", "sample_interval_ms"): "telemetry_sample_interval_ms",
    ("telemetry", "tariff_usd_per_kwh"): "telemetry_tariff_usd_per_kwh",
    # ── Sidecar paths
    ("zones_path",):                     "zones_path",
    ("applications_path",):              "applications_path",
}


def load_config(
    yaml_path: Optional[str] = None,
    cli_overrides: Optional[dict[str, Any]] = None,
) -> PipelineConfig:
    """Load configuration: defaults → YAML → CLI overrides."""
    config = PipelineConfig()

    # Layer 2: YAML file
    if yaml_path:
        yaml_values = _load_yaml(yaml_path)
        if yaml_values:
            _apply_yaml(config, yaml_values)

    # Layer 3: CLI overrides (highest priority)
    if cli_overrides:
        _apply_overrides(config, cli_overrides)

    _validate(config)
    return config


def _load_yaml(path: str) -> Optional[dict]:
    if not _HAS_YAML:
        logger.warning("PyYAML not installed — ignoring config file %s", path)
        return None

    p = Path(path)
    if not p.exists():
        logger.debug("Config file not found: %s — using defaults", path)
        return None

    with open(p, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        logger.warning("Config file %s is not a YAML mapping — using defaults", path)
        return None

    logger.info("Loaded config from %s", path)
    return data


def _apply_yaml(config: PipelineConfig, data: dict) -> None:
    for yaml_keys, field_name in _YAML_MAP.items():
        value = _nested_get(data, yaml_keys)
        if value is not None:
            _set_field(config, field_name, value)


def _apply_overrides(config: PipelineConfig, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if value is not None and hasattr(config, key):
            _set_field(config, key, value)


def _set_field(config: PipelineConfig, name: str, value: Any) -> None:
    try:
        # Coerce to the field's existing type
        current = getattr(config, name)
        if current is not None and not isinstance(value, type(current)):
            target_type = type(current)
            if target_type == bool and isinstance(value, str):
                value = value.lower() in ("true", "1", "yes")
            else:
                value = target_type(value)
        object.__setattr__(config, name, value)
    except (TypeError, ValueError) as e:
        logger.warning("Cannot set config.%s = %r: %s", name, value, e)


def _nested_get(data: dict, keys: tuple[str, ...]) -> Any:
    current = data
    for k in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(k)
        if current is None:
            return None
    return current


def _validate(config: PipelineConfig) -> None:
    if not 0.0 <= config.gpu_util_threshold <= 1.0:
        logger.warning("gpu_util_threshold=%.2f out of [0,1], clamping", config.gpu_util_threshold)
        object.__setattr__(config, "gpu_util_threshold", max(0.0, min(1.0, config.gpu_util_threshold)))
    if config.max_concurrent_gpu < 1:
        object.__setattr__(config, "max_concurrent_gpu", 1)
    if config.pool_depth < 1:
        object.__setattr__(config, "pool_depth", 1)
    if config.det_conf < 0.0 or config.det_conf > 1.0:
        object.__setattr__(config, "det_conf", max(0.0, min(1.0, config.det_conf)))
