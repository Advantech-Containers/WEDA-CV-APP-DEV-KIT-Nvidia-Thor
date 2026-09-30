"""Lightweight value types for the pipeline. No external dependencies."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


class DevicePlacement(Enum):
    CPU = auto()
    GPU = auto()
    AUTO = auto()


class StageID(Enum):
    DETECTION = "detection"
    DEPTH = "depth"
    SPATIAL = "spatial"
    TRACKING = "tracking"
    EMBEDDING = "embedding"
    PHYSICAL_REASONING = "physical_reasoning"
    REASONING = "reasoning"
    QUERY = "query"


# Fixed placement rules (scheduler honours these unconditionally)
FIXED_PLACEMENTS: dict[StageID, DevicePlacement] = {
    StageID.DETECTION: DevicePlacement.GPU,
    StageID.TRACKING: DevicePlacement.CPU,
    StageID.SPATIAL: DevicePlacement.CPU,
    StageID.QUERY: DevicePlacement.CPU,
}

AUTO_STAGES: set[StageID] = {StageID.DEPTH, StageID.EMBEDDING, StageID.REASONING}


@dataclass(slots=True)
class Detection:
    """Single detected object in pixel coordinates."""
    bbox: np.ndarray          # (4,) float32: x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    mask: Optional[np.ndarray] = None   # (H, W) uint8 segmentation mask
    track_id: int = -1
    embedding: Optional[np.ndarray] = None  # visual embedding vector
    lean_offset_m: float = 0.0
    is_unstable: bool = False

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )

    @property
    def area(self) -> float:
        return float(
            (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
        )


@dataclass(slots=True)
class SpatialInfo:
    """3-D spatial data for a detected object."""
    distance_m: float = 0.0
    position_3d: Optional[np.ndarray] = None   # (3,) float32: X, Y, Z
    size_estimate: tuple[float, float] = (0.0, 0.0)  # width_m, height_m
    nearest_neighbor_id: int = -1
    nearest_neighbor_dist: float = float("inf")
    depth_reliable: bool = True
    velocity_3d: Optional[np.ndarray] = None   # (3,) float32: dX, dY, dZ per sec
    speed_ms: float = 0.0
    approaching: bool = False
    dwell_seconds: float = 0.0


@dataclass(slots=True)
class FramePacket:
    """All data associated with a single captured frame.

    Created by the input handler, enriched by each pipeline stage,
    and consumed by the orchestrator for display / output.
    """
    frame_id: int
    timestamp: float = field(default_factory=time.monotonic)
    rgb: Optional[np.ndarray] = None          # (H, W, 3) uint8 — may be a pool buffer
    depth: Optional[np.ndarray] = None        # (H, W) uint16
    detections: list[Detection] = field(default_factory=list)
    spatial: list[SpatialInfo] = field(default_factory=list)
    depth_colorized: Optional[np.ndarray] = None
    scene_description: str = ""
    physical_reasoning: str = ""
    frame_embedding: Optional[np.ndarray] = None  # scene-level embedding
    scene_changed: bool = False
    # Query engine
    pending_query: Optional[str] = None
    query_response: Optional[str] = None
    # Safety zone alerts
    safety_alert: bool = False
    safety_alert_distance: float = 0.0
    safety_alert_track_id: int = -1
    # Navigation
    navigation_plan: Optional[object] = None
    # Active tracker lock (set by pipeline from tracker.locked_track_id)
    # -1 = no lock active; any >=0 value is the pinned track_id the renderer
    # should highlight with a [LOCK] prefix + status line.
    locked_track_id: int = -1
    # Fact-check verdicts for the most recent auto-caption; populated by
    # the orchestrator after the SceneReasoner emits a scene description.
    # The renderer reads this to overlay per-claim verdict badges.
    fact_check_report: Optional[dict] = None
    # Pool buffer tokens for release after processing
    _pool_tokens: list[tuple[str, int]] = field(default_factory=list)

    def record_pool_token(self, pool_name: str, idx: int) -> None:
        self._pool_tokens.append((pool_name, idx))


@dataclass(slots=True)
class PipelineConfig:
    """Runtime configuration — loaded from YAML, overridable via CLI."""
    # Input
    input_backend: str = "auto"  # auto | v4l2 | orbbec | gstreamer | rtsp | synthetic
    source: str = "synthetic"
    input_loop: bool = False     # if source is a video file, restart from frame 0 on EOF
    width: int = 1920
    height: int = 1080
    # RTSP
    rtsp_protocol: str = "tcp"
    rtsp_reconnect_attempts: int = 5
    rtsp_timeout: float = 10.0
    # Detection (any Ultralytics model — YOLO26/11/v8, RT-DETR)
    yolo_model: str = "yolo26n-seg.pt"
    det_conf: float = 0.5
    det_input_size: int = 640
    use_tensorrt: bool = True
    device: str = "cuda:0"
    # Display
    headless: bool = False
    max_frames: int = 0  # 0 = unlimited
    # RT-Embedding (Cosmos-Embed1)
    enable_embedding: bool = False
    embedding_model: str = "nvidia/Cosmos-Embed1-224p"
    embedding_dim: int = 256
    scene_change_threshold: float = 0.95
    # Physical Reasoning (Cosmos-Reason2-2B)
    enable_physical_reasoning: bool = False
    physical_model: str = "nvidia/Cosmos-Reason2-2B"
    physical_cooldown: float = 2.0
    # VLM Scene Reasoning (config-driven — swap any HF model)
    enable_vlm: bool = False
    vlm_model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    vlm_cooldown: float = 5.0
    vlm_prompt_format: str = "qwen3"
    vlm_max_tokens: int = 300
    # Physical Reasoning prompt
    physical_prompt_format: str = "cosmos"
    physical_max_tokens: int = 256
    # Quantization
    quant_backend: str = "auto"
    quant_vlm_strategy: str = "nvfp4"
    quant_vlm_kv_cache: str = "nvfp4"
    quant_physical_strategy: str = "fp8"
    quant_physical_kv_cache: str = "fp8"
    quant_cache_dir: str = "model_cache"
    quant_calibration_samples: int = 8
    # Orbbec camera
    orbbec_align_depth: bool = True
    orbbec_depth_min_mm: int = 20
    orbbec_depth_max_mm: int = 10000
    # VIC Hardware Offload
    enable_vic_offload: bool = True
    # L1 Cache Preference
    depth_prefer_l1: bool = True
    # Optimization
    pool_depth: int = 4
    gpu_util_threshold: float = 0.85
    max_concurrent_gpu: int = 2
    scheduler_speedup_min: float = 2.0
    # Tracker
    tracker_high_thresh: float = 0.5
    tracker_iou_threshold: float = 0.3
    tracker_max_age_sec: float = 60.0
    tracker_embedding_threshold: float = 0.7
    tracker_min_display_age: int = 3
    tracker_dormancy_sec: float = 1.0
    # GPU Monitor
    enable_gpu_monitor: bool = False
    gpu_monitor_interval: float = 1.0
    # Profiling
    profile_interval: int = 100
    profiler_window: int = 100
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    alert_log_file: str = "logs"
    # Safety
    safety_zone_radius: float = 1.0
    dwell_alert_seconds: float = 30.0
    # Classes eligible to trigger the safety proximity alert. Static
    # furniture / appliances (tv, couch, refrigerator, …) are excluded by
    # default. Override via YAML key `safety.alert_classes`; empty list
    # allows all classes.
    safety_alert_classes: tuple = (
        "person", "worker", "operator", "pedestrian",
        "forklift", "agv", "robot", "cart", "trolley",
        "box", "pallet", "crate", "bin",
        "dog", "cat",
    )
    # Hysteresis: require this many consecutive in-radius frames before firing,
    # and this many consecutive out-of-radius frames before clearing. Prevents
    # the alert from flickering on one-frame depth noise or a detector blink.
    safety_alert_enter_frames: int = 3
    safety_alert_exit_frames: int = 5
    # Navigation (potential field planner)
    enable_navigation: bool = False
    nav_attractive_gain: float = 1.0
    nav_repulsive_gain: float = 100.0
    nav_obstacle_radius: float = 0.5

    # ── HTTP API ──
    api_enable: bool = True
    api_bind: str = "127.0.0.1"
    api_port: int = 9180
    api_token_env: str = "WEDA_API_TOKEN"
    # ── Innovation pillars ──
    fact_check_enable: bool = True
    fact_check_distance_tolerance_pct: float = 15.0
    fact_check_size_tolerance_pct: float = 25.0
    scene_memory_enable: bool = True
    scene_memory_window_min: int = 30
    scene_memory_sample_hz: float = 1.0
    predictive_safety_enable: bool = True
    predictive_safety_horizon_s: float = 3.0
    predictive_safety_critical_eta_s: float = 1.5
    predictive_safety_danger_distance_m: float = 0.8
    alarm_engine_enable: bool = True
    alarm_engine_persist_path: str = "config/alarms.jsonl"
    # ── MJPEG streaming (consumed by /v1/stream/*.mjpg) ──
    stream_mjpeg_fps_cap: int = 10
    stream_jpeg_quality: int = 70
    stream_draw_detections: bool = True
    # ── Telemetry ──
    telemetry_enable: bool = True
    telemetry_sample_interval_ms: int = 100
    telemetry_tariff_usd_per_kwh: float = 0.12
    # ── Sidecar paths ──
    zones_path: str = "config/zones.yaml"
    applications_path: str = "config/applications.yaml"
