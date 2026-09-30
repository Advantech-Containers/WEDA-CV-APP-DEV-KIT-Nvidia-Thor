"""Main loop, lifecycle management, display, signal handling.
The Orchestrator owns all components and runs the deterministic
acquisition -> process -> display -> release cycle until shutdown.
Display rendering delegated to app.display.renderer.DisplayRenderer.
"""

from __future__ import annotations

import collections
import concurrent.futures as _cf
import os
import signal
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.core.pipeline import Pipeline
from app.display.renderer import DisplayRenderer
from app.io.camera_factory import create_camera_backend
from app.modules.depth import DepthProcessor
from app.modules.detector import Detector
from app.modules.embedder import Embedder
from app.modules.physical_reasoner import PhysicalReasoner
from app.modules.reasoning import SceneReasoner
from app.modules.bytetrack import ByteTracker as Tracker
from weda_optim import BufferPool, create_default_pools
from weda_optim import GpuMonitor, create_gpu_monitor, create_process_monitor
from weda_optim import MemoryManager
from weda_optim import Profiler
from weda_optim import Scheduler
from app.utils.alert_logger import AlertLogger
from app.utils.data_structures import FramePacket, PipelineConfig
from app.utils.logger import get_logger, setup_logger, silence_runtime_logs

logger = get_logger()

_VERSION = "1.0.0"
WIN_NAME = "WEDA Edge CV-VLM Pipeline"

def _read_mem_total_gb() -> float:
    """Read total system memory from /proc/meminfo (UMA = shared on Jetson)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return 0.0

_TRACK_PREFIXES = (
    "keep tracking ", "track ", "follow ", "start tracking ", "begin tracking ",
)
_STOP_COMMANDS = frozenset((
    "stop tracking", "clear target", "stop track", "untrack", "stop following",
))


class Orchestrator:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._running = False
        self._last_physical_query_response: str = ""
        self._last_vlm_description: str = ""
        self._tier3_query_pending: bool = False

        setup_logger(level=config.log_level, log_file=config.log_file)
        silence_runtime_logs()

        self._start_time: float = 0.0
        self._frame_count: int = 0
        self._det_counts: dict[str, int] = {}
        self._alert_logger = AlertLogger()
        self._gpu_monitor: Optional[GpuMonitor] = None
        try:
            self._gpu_monitor = create_gpu_monitor(config)
        except Exception:
            pass

        try:
            self._process_monitor = create_process_monitor()
        except Exception:
            self._process_monitor = None

        self._mem_mgr = MemoryManager(device=config.device)
        self._pool = create_default_pools(config.width, config.height, config.pool_depth)
        self._scheduler = Scheduler(
            fixed_placements={
                "detection": "gpu", "tracking": "cpu",
                "spatial": "cpu", "query": "cpu",
            },
            auto_stages={"depth", "reasoning"},
            gpu_util_threshold=config.gpu_util_threshold,
            max_concurrent_gpu=config.max_concurrent_gpu,
            min_speedup=config.scheduler_speedup_min,
            gpu_monitor=self._gpu_monitor,
        )
        self._profiler = Profiler(
            window=config.profiler_window,
            log_interval=0,
        )

        self._input = create_camera_backend(config, pool=self._pool)
        if self._input is None:
            raise RuntimeError(f"Failed to create camera backend: {getattr(config, 'input_backend', 'auto')}")
        self._detector = Detector(config, pool=self._pool)
        self._depth = DepthProcessor(config, scheduler=self._scheduler, pool=self._pool)
        self._tracker = Tracker(
            high_thresh=config.tracker_high_thresh,
            iou_threshold=config.tracker_iou_threshold,
            max_age_sec=config.tracker_max_age_sec,
            embedding_threshold=config.tracker_embedding_threshold,
            min_display_age=config.tracker_min_display_age,
            dormancy_sec=config.tracker_dormancy_sec,
        )
        self._embedder = Embedder(config)
        self._physical = PhysicalReasoner(config)
        self._reasoner = SceneReasoner(config)

        self._pipeline = Pipeline(
            config=config, detector=self._detector, depth_proc=self._depth,
            tracker=self._tracker, embedder=self._embedder,
            physical_reasoner=self._physical, reasoner=self._reasoner,
            scheduler=self._scheduler, profiler=self._profiler, pool=self._pool,
        )

        self._renderer = DisplayRenderer()
        self._fullscreen = False
        self._window_created = False
        self._render_queue: collections.deque = collections.deque(maxlen=1)
        self._display_lock = threading.Lock()
        self._display_buffer: Optional[np.ndarray] = None
        self._display_running = False
        self._display_thread: Optional[threading.Thread] = None

        from app.modules.scene_memory import SceneMemory
        self._scene_memory = SceneMemory(
            embedder=self._embedder,
            window_min=int(getattr(config, "scene_memory_window_min", 30)),
            sample_hz=float(getattr(config, "scene_memory_sample_hz", 1.0)),
        )
        from app.modules.predictive_safety import PredictiveSafetyMonitor
        self._predictive_safety = PredictiveSafetyMonitor(
            horizon_s=float(getattr(config, "predictive_safety_horizon_s", 3.0)),
            critical_eta_s=float(getattr(config, "predictive_safety_critical_eta_s", 1.5)),
        )
        from app.modules.trajectory import TrajectoryEstimator
        self._trajectory = TrajectoryEstimator()

        from app.modules.alarm_engine import AlarmEngine
        self._alarm_engine = AlarmEngine(
            persist_path=getattr(config, "alarm_engine_persist_path",
                                 "config/alarms.jsonl"),
        )

    def run(self) -> None:
        self._setup_signals()
        self._running = True

        self._print_banner()
        self._load_all_models()

        if not self._input.open():
            print("  [ERROR] Failed to open input source, aborting")
            return

        intrinsics = self._input.camera_intrinsics
        if intrinsics is not None:
            fx, fy, cx, cy, depth_scale = intrinsics
            self._depth.update_intrinsics(fx, fy, cx, cy, depth_scale)

        print(f"\n  Pipeline ready: {self._config.width}x{self._config.height}")
        print()
        self._display_running = True
        self._display_thread = threading.Thread(
            target=self._display_loop, daemon=True, name="render"
        )
        self._display_thread.start()

        self._api_handle = None
        if getattr(self._config, "api_enable", True):
            try:
                import os
                from app.api.main import PipelineHandle, start_in_thread
                cfg = self._config
                token_env = getattr(cfg, "api_token_env", "WEDA_API_TOKEN")
                api_handle = PipelineHandle(
                    latest_packet=None,
                    scene_reasoner=self._reasoner,
                    physical_reasoner=self._physical,
                    embedder=self._embedder,
                    gpu_monitor=self._gpu_monitor,
                    scene_memory=self._scene_memory,
                    predictive_safety=self._predictive_safety,
                    alarm_engine=self._alarm_engine,
                    config=cfg,
                    api_token=os.environ.get(token_env) or None,
                    bind_host=os.environ.get("WEDA_API_BIND") or getattr(cfg, "api_bind", "127.0.0.1"),
                    bind_port=int(os.environ.get("WEDA_API_PORT") or getattr(cfg, "api_port", 9180)),
                )
                self._api_handle = api_handle
                start_in_thread(api_handle)
            except Exception as e:
                logger.warning("HTTP API server failed to start: %s", e)
                self._api_handle = None
        else:
            logger.info("HTTP API server is disabled by configuration")

        self._start_time = time.monotonic()
        frame_count = 0
        try:
            while self._running:
                packet = self._input.read()
                if packet is None:
                    logger.info("Input exhausted")
                    break

                for token in self._input.pop_frame_tokens():
                    packet.record_pool_token(*token)

                query = self._renderer.consume_query()
                if query:
                    query_lower = query.lower().strip()
                    if any(query_lower.startswith(p) for p in _TRACK_PREFIXES):
                        for p in sorted(_TRACK_PREFIXES, key=len, reverse=True):
                            if query_lower.startswith(p):
                                raw = query_lower[len(p):].strip()
                                break
                        target_class = raw.lstrip("the ").lstrip("a ").strip()
                        import re as _re
                        _id_m = _re.search(r'\s+(\d+)$', target_class)
                        if _id_m:
                            _numeric_id = int(_id_m.group(1))
                            _base_class = target_class[:_id_m.start()].strip()
                            tid = self._tracker.set_target(track_id=_numeric_id)
                            if tid is None:
                                tid = self._tracker.set_target(class_name=_base_class or target_class)
                        else:
                            tid = self._tracker.set_target(class_name=target_class)
                        if tid is not None:
                            self._tracker.set_locked(track_id=tid)
                            self._renderer.set_response(
                                f"Tracking {target_class} (track #{tid}) \u2014 LOCKED"
                            )
                        else:
                            self._renderer.set_response(f"No {target_class} found to track")
                    elif query_lower in _STOP_COMMANDS:
                        self._tracker.clear_target()
                        self._tracker.clear_locked()
                        self._renderer.set_response("Target tracking cleared")
                    else:
                        packet.pending_query = query

                packet = self._pipeline.process(packet)

                try:
                    self._trajectory.update(packet)
                except Exception as _e:
                    logger.debug("trajectory update failed: %s", _e)
                try:
                    self._predictive_safety.step(packet)
                except Exception as _e:
                    logger.debug("predictive_safety step failed: %s", _e)
                try:
                    self._scene_memory.maybe_push(packet)
                except Exception as _e:
                    logger.debug("scene_memory push failed: %s", _e)

                if getattr(self._config, "alarm_engine_enable", True):
                    try:
                        self._alarm_engine.step(packet)
                    except Exception as _e:
                        logger.debug("alarm_engine step failed: %s", _e)

                if getattr(self._config, "fact_check_enable", True):
                    try:
                        desc = (packet.scene_description or
                                self._reasoner.latest_description)
                        if desc and desc != getattr(self, "_last_factchecked_desc", ""):
                            from app.modules.fact_checker import verify as _fc_verify
                            report = _fc_verify(
                                desc, packet,
                                distance_tolerance_pct=float(getattr(
                                    self._config, "fact_check_distance_tolerance_pct", 15.0)),
                                size_tolerance_pct=float(getattr(
                                    self._config, "fact_check_size_tolerance_pct", 25.0)),
                            )
                            packet.fact_check_report = report.to_dict()
                            self._last_factchecked_desc = desc
                    except Exception as _e:
                        logger.debug("fact_check failed: %s", _e)

                if getattr(self, "_api_handle", None) is not None:
                    self._api_handle.latest_packet = packet
                for det in packet.detections:
                    self._det_counts[det.class_name] = self._det_counts.get(det.class_name, 0) + 1
                self._frame_count += 1

                if packet.query_response:
                    _resp = packet.query_response
                    self._renderer.set_response(_resp)
                    _is_deferred = (
                        _resp.startswith("Processing")
                        or _resp.startswith("Analyzing")
                        or _resp.startswith("Thinking")
                    )
                    self._tier3_query_pending = _is_deferred

                _phys_q = self._physical.latest_query_response
                if _phys_q and _phys_q != self._last_physical_query_response:
                    self._renderer.set_physical_response(_phys_q)
                    self._last_physical_query_response = _phys_q
                    logger.debug("Physical query response received: %.80s", _phys_q)

                if self._tier3_query_pending:
                    _reasoner = self._pipeline._reasoner
                    _vlm_resp = _reasoner.latest_user_response
                    if _vlm_resp and _vlm_resp != self._last_vlm_description:
                        self._renderer.set_response(_vlm_resp)
                        self._last_vlm_description = _vlm_resp
                        if not _reasoner.user_query_running:
                            self._tier3_query_pending = False
                        logger.debug("Tier 3 VLM response received: %.80s", _vlm_resp)
                    elif not _reasoner.user_query_running:
                        _err = _reasoner.user_error_message
                        if _err:
                            self._renderer.set_response(_err)
                            self._last_vlm_description = _err
                        self._tier3_query_pending = False

                if not self._config.headless and hasattr(self._renderer, 'safety_zone_radius'):
                    self._config.safety_zone_radius = self._renderer.safety_zone_radius
                self._enqueue_display(packet)

                self._pool.release_all(packet._pool_tokens)

                frame_count += 1
                if self._config.max_frames > 0 and frame_count >= self._config.max_frames:
                    logger.info("Reached max frames (%d)", self._config.max_frames)
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._shutdown()

    def _enqueue_display(self, packet: FramePacket) -> None:
        """Snapshot current frame data and enqueue for display thread. Non-blocking."""
        rgb_copy = None
        if packet.rgb is not None:
            arr = packet.rgb if isinstance(packet.rgb, np.ndarray) else np.asarray(packet.rgb)
            rgb_copy = arr.copy()

        depth_colorized_copy = None
        if packet.depth_colorized is not None:
            arr = packet.depth_colorized if isinstance(packet.depth_colorized, np.ndarray) else np.asarray(packet.depth_colorized)
            depth_colorized_copy = arr.copy()

        gpu_stats = None
        if self._gpu_monitor:
            _mem_frac = self._gpu_monitor.get_memory_utilization()
            _mem_total = _read_mem_total_gb()
            _top_procs: list[dict] = []
            if self._process_monitor is not None:
                try:
                    _top_procs = self._process_monitor.get_top_processes(n=3)
                except Exception:
                    _top_procs = []
            gpu_stats = {
                "utilization": self._gpu_monitor.get_gpu_utilization(),
                "memory": _mem_frac,
                "memory_used_gb": round(_mem_frac * _mem_total, 1) if _mem_total > 0 else 0.0,
                "memory_total_gb": round(_mem_total, 1),
                "temperature": self._gpu_monitor.get_temperature(),
                "process_gpu_mem_mb": self._gpu_monitor.get_process_gpu_mem_mb(),
                "process_ram_mb": self._gpu_monitor.get_process_ram_mb(),
                "process_cpu_pct": self._gpu_monitor.get_process_cpu_pct(),
                "top_processes": _top_procs,
            }

        render_kwargs = dict(
            rgb=rgb_copy,
            detections=list(packet.detections or []),
            depth_colorized=depth_colorized_copy,
            spatial=list(packet.spatial or []),
            scene_changed=packet.scene_changed,
            frame_id=packet.frame_id,
            profiler_summary=self._profiler.summary(),
            gpu_stats=gpu_stats,
            scheduler_stats=self._scheduler.stats(),
            detector_backend=self._detector.backend,
            vlm_status="active" if self._reasoner.latest_description else "waiting",
            vlm_model=getattr(self._config, "vlm_model", ""),
            scene_desc=self._reasoner.latest_description,
            physical_desc=self._physical.latest_response,
            active_tracks=self._tracker.active_tracks,
            has_depth_stream=self._input.has_depth,
            depth_index=self._input.depth_index,
            target_track_id=self._tracker.target_track_id,
            display_ready_fn=self._tracker.is_display_ready,
            safety_alert=packet.safety_alert,
            safety_alert_distance=packet.safety_alert_distance,
            safety_alert_class=next(
                (d.class_name for d in (packet.detections or [])
                 if d.track_id == packet.safety_alert_track_id), ""
            ),
            fx=getattr(self._depth, '_fx', 615.0),
            dwell_alert_seconds=self._config.dwell_alert_seconds,
            gpu_backend=type(self._gpu_monitor).__name__ if self._gpu_monitor else "none",
            navigation_plan=packet.navigation_plan,
            vlm_tok_per_sec=self._reasoner.vlm_tok_per_sec,
            physical_tok_per_sec=self._physical.vlm_tok_per_sec,
            embedder_status=(
                "loaded"
                if self._embedder.is_loaded
                else ("failed" if self._config.enable_embedding else "disabled")
            ),
        )
        self._render_queue.append(render_kwargs)

    def _display_loop(self) -> None:
        """Render thread.

        Always produces the annotated canvas and stashes it in
        ``self._display_buffer`` so any REST consumer of
        ``/v1/stream/*.mjpg`` can broadcast it. ``cv2.imshow`` +
        ``cv2.waitKey`` are only invoked when ``config.headless ==
        False``, otherwise we skip the cv2 GUI calls entirely. This
        avoids GIL contention between cv2's GUI event loop and
        uvicorn's asyncio loop in headless deployments.
        """
        gui_on = not self._config.headless
        if gui_on and not self._window_created:
            try:
                cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WIN_NAME, 1920, 1080)
                self._window_created = True
            except cv2.error as e:
                logger.warning("cv2 GUI unavailable (%s) — falling to headless render", e)
                gui_on = False
                self._config.headless = True

        while self._display_running:
            if not self._render_queue:
                time.sleep(0.002)
                continue

            kwargs = self._render_queue.pop()
            try:
                from app.utils.data_structures import FramePacket as _FP
                pkt = _FP(frame_id=kwargs.get("frame_id", 0))
                pkt.rgb = kwargs.get("rgb")
                pkt.detections = kwargs.get("detections", [])
                pkt.depth_colorized = kwargs.get("depth_colorized")
                pkt.spatial = kwargs.get("spatial", [])
                pkt.scene_changed = kwargs.get("scene_changed", False)
                pkt.navigation_plan = kwargs.get("navigation_plan")

                canvas = self._renderer.render(
                    packet=pkt,
                    profiler_summary=kwargs.get("profiler_summary"),
                    gpu_stats=kwargs.get("gpu_stats"),
                    scheduler_stats=kwargs.get("scheduler_stats"),
                    detector_backend=kwargs.get("detector_backend", "unknown"),
                    vlm_status=kwargs.get("vlm_status", "waiting"),
                    vlm_model=kwargs.get("vlm_model", ""),
                    scene_desc=kwargs.get("scene_desc", ""),
                    physical_desc=kwargs.get("physical_desc", ""),
                    active_tracks=kwargs.get("active_tracks", {}),
                    has_depth_stream=kwargs.get("has_depth_stream", False),
                    depth_index=kwargs.get("depth_index"),
                    target_track_id=kwargs.get("target_track_id"),
                    display_ready_fn=kwargs.get("display_ready_fn"),
                    safety_alert=kwargs.get("safety_alert", False),
                    safety_alert_distance=kwargs.get("safety_alert_distance", 0.0),
                    safety_alert_class=kwargs.get("safety_alert_class", ""),
                    fx=kwargs.get("fx", 615.0),
                    dwell_alert_seconds=kwargs.get("dwell_alert_seconds", 30.0),
                    gpu_backend=kwargs.get("gpu_backend", ""),
                    vlm_tok_per_sec=kwargs.get("vlm_tok_per_sec", 0.0),
                    physical_tok_per_sec=kwargs.get("physical_tok_per_sec", 0.0),
                    camera_id=getattr(self._config, "source", "") or "",
                    embedder_status=kwargs.get("embedder_status", "disabled"),
                )
                with self._display_lock:
                    self._display_buffer = canvas
                if getattr(self, "_api_handle", None) is not None:
                    self._api_handle.latest_canvas = canvas

                if gui_on:
                    cv2.imshow(WIN_NAME, canvas)
            except Exception as e:
                if not getattr(self, "_render_warned", False):
                    logger.warning("Render thread error: %s", e, exc_info=True)
                    self._render_warned = True
                else:
                    logger.debug("Render thread error (suppressed repeat): %s", e)

            if gui_on:
                key = cv2.waitKey(1) & 0xFF
                action = self._renderer.handle_key(key)
                if action == "quit":
                    self._running = False
                    self._display_running = False
                elif action == "fullscreen":
                    self._fullscreen = not self._fullscreen
                    if self._fullscreen:
                        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(WIN_NAME, 1920, 1080)
                elif action == "reset_cache":
                    self._reasoner.prune_cache(set())
                    self._renderer.set_response("")
            else:
                time.sleep(0.005)
                logger.info("VLM cache reset")

    @staticmethod
    def _suppress_stderr():
        """Redirect fd 2 to /dev/null to suppress C++ library output."""
        import os as _os, sys as _sys, ctypes as _ct
        _sys.stdout.flush()
        _sys.stderr.flush()
        try:
            _ct.CDLL(None).fflush(None)
        except Exception:
            pass
        devnull = _os.open(_os.devnull, _os.O_WRONLY)
        old = _os.dup(2)
        _os.dup2(devnull, 2)
        return old, devnull

    @staticmethod
    def _restore_stderr(old, devnull):
        import os as _os, sys as _sys, ctypes as _ct
        try:
            _ct.CDLL(None).fflush(None)
        except Exception:
            pass
        _os.dup2(old, 2)
        _os.close(devnull)
        _os.close(old)

    def _load_all_models(self) -> None:
        """Block-load all models before display starts."""
        t_total = time.monotonic()

        # 1. Detector (suppress TRT C++ logs during load + warmup)
        print("  [1/4] Loading detector...")
        import sys as _sys, ctypes as _ct
        _sys.stdout.flush(); _sys.stderr.flush()
        try:
            _ct.CDLL(None).fflush(None)
        except Exception:
            pass
        _saved_1 = os.dup(1)
        _saved_2 = os.dup(2)
        _dnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_dnull, 1)
        os.dup2(_dnull, 2)
        os.close(_dnull)
        try:
            self._detector.initialize()
            dummy = np.zeros((self._config.height, self._config.width, 3), dtype=np.uint8)
            self._detector.detect(dummy, confidence=0.5)
        except Exception as e:
            try:
                _ct.CDLL(None).fflush(None)
            except Exception:
                pass
            os.dup2(_saved_1, 1); os.dup2(_saved_2, 2)
            os.close(_saved_1); os.close(_saved_2)
            print(f"  [1/4] Detector FAILED: {e}")
        else:
            try:
                _ct.CDLL(None).fflush(None)
            except Exception:
                pass
            os.dup2(_saved_1, 1); os.dup2(_saved_2, 2)
            os.close(_saved_1); os.close(_saved_2)
            print(f"  [1/4] Detector ready ({self._detector.backend})")

        print("  [2/4] Loading embedder...")
        print("  [3/4] Loading physical reasoning...")
        _sys.stdout.flush(); _sys.stderr.flush()
        try:
            _ct.CDLL(None).fflush(None)
        except Exception:
            pass
        _saved_1 = os.dup(1)
        _saved_2 = os.dup(2)
        _dnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_dnull, 1)
        os.dup2(_dnull, 2)
        os.close(_dnull)
        with _cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="init") as pool:
            fut_embed = pool.submit(self._embedder.initialize)
            fut_phys = pool.submit(self._physical.initialize)
        try:
            _ct.CDLL(None).fflush(None)
        except Exception:
            pass
        os.dup2(_saved_1, 1); os.dup2(_saved_2, 2)
        os.close(_saved_1); os.close(_saved_2)
        try:
            fut_embed.result()
            print("  [2/4] Embedder ready")
        except Exception as e:
            print(f"  [2/4] Embedder FAILED: {e}")
        try:
            fut_phys.result()
            print("  [3/4] Physical reasoner ready")
        except Exception as e:
            print(f"  [3/4] Physical reasoner FAILED: {e}")

        print("  [4/4] Loading VLM...")
        _sys.stdout.flush(); _sys.stderr.flush()
        try:
            _ct.CDLL(None).fflush(None)
        except Exception:
            pass
        _saved_1 = os.dup(1)
        _saved_2 = os.dup(2)
        _dnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_dnull, 1)
        os.dup2(_dnull, 2)
        os.close(_dnull)
        self._reasoner.initialize()
        try:
            self._reasoner._ensure_loaded()
        except Exception:
            pass
        try:
            _ct.CDLL(None).fflush(None)
        except Exception:
            pass
        os.dup2(_saved_1, 1); os.dup2(_saved_2, 2)
        os.close(_saved_1); os.close(_saved_2)
        try:
            print("  [4/4] VLM ready")
        except Exception as e:
            print(f"  [4/4] VLM FAILED: {e}")
        elapsed = time.monotonic() - t_total
        print(f"\n  All models loaded in {elapsed:.1f}s")
    def _print_banner(self) -> None:
        _det_model = getattr(self._config, "yolo_model", "unknown")
        _vlm_model = getattr(self._config, "vlm_model", "unknown").split("/")[-1]
        _phys_model = getattr(self._config, "physical_model", "unknown").split("/")[-1]
        _device = getattr(self._config, "device", "unknown")
        print()
        print("=" * 58)
        print(f"   Advantech WEDA-CV VLM SDK v{_VERSION}")
        print("   Powered by Advantech Co., Ltd.")
        print("-" * 58)
        print(f"   Detector : {_det_model}")
        print(f"   VLM      : {_vlm_model}")
        print(f"   Physical : {_phys_model}")
        print(f"   Device   : {_device}")
        print("=" * 58)
        print()

    def _shutdown(self) -> None:
        self._running = False
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        if getattr(self, "_api_handle", None) is not None:
            try:
                from app.api.main import shutdown as api_shutdown
                api_shutdown(timeout_s=5.0)
            except Exception as e:
                logger.debug("api shutdown skipped: %s", e)
        if self._gpu_monitor:
            self._gpu_monitor.stop()
        if getattr(self, "_process_monitor", None) is not None:
            try:
                self._process_monitor.stop()
            except Exception:
                pass
        self._input.close()
        self._detector.shutdown()
        self._embedder.shutdown()
        self._physical.shutdown()
        self._reasoner.shutdown()
        try:
            from app.modules.telemetry import get_registry
            get_registry().stop()
        except Exception:
            pass
        self._alert_logger.close()
        print()
        print("=" * 58)
        print("   SHUTDOWN SUMMARY")
        print("=" * 58)
        avg_fps = self._frame_count / elapsed if elapsed > 0 else 0
        print(f"  Runtime:     {elapsed:.1f}s")
        print(f"  Frames:      {self._frame_count}")
        print(f"  Avg FPS:     {avg_fps:.1f}")

        if self._gpu_monitor:
            _mem_frac = self._gpu_monitor.get_memory_utilization()
            _total_gb = _read_mem_total_gb()
            _used_gb = round(_mem_frac * _total_gb, 1) if _total_gb > 0 else 0.0
            print(f"  GPU util:    {self._gpu_monitor.get_gpu_utilization() * 100:.0f}%")
            print(f"  GPU memory:  {_used_gb:.1f}GB / {_total_gb:.0f}GB")
            print(f"  GPU temp:    {self._gpu_monitor.get_temperature():.1f} C")

        _vlm_model = getattr(self._config, "vlm_model", "unknown").split("/")[-1]
        _phys_model = getattr(self._config, "physical_model", "unknown").split("/")[-1]
        vlm_tps = self._reasoner.vlm_tok_per_sec
        phys_tps = self._physical.vlm_tok_per_sec
        if vlm_tps > 0:
            print(f"  VLM tok/s:   {vlm_tps:.1f} ({_vlm_model})")
        if phys_tps > 0:
            print(f"  Phys tok/s:  {phys_tps:.1f} ({_phys_model})")

        if self._det_counts:
            det_parts = [f"{v} {k}" for k, v in sorted(self._det_counts.items(), key=lambda x: -x[1])]
            print(f"  Detections:  {', '.join(det_parts)}")

        alert_count = self._alert_logger.alert_count
        print(f"  Alerts:      {alert_count}")
        if alert_count > 0:
            for atype, cnt in self._alert_logger.alerts_by_type.items():
                print(f"    - {atype}: {cnt}")
        try:
            _alert_path = self._alert_logger.current_log_path()
            print(f"  Alert log:   {_alert_path}")
        except Exception:
            _legacy = self._alert_logger.log_path
            if _legacy:
                print(f"  Alert log:   {_legacy}")

        print("=" * 58)

        logger.debug("Scheduler: %s", self._scheduler.stats())
        logger.debug("Pool: %s", self._pool.stats())
        self._pool.shutdown()
        self._mem_mgr.shutdown()
        self._display_running = False
        if self._display_thread and self._display_thread.is_alive():
            self._display_thread.join(timeout=2.0)
        if not self._config.headless:
            try:
                cv2.waitKey(1)
                time.sleep(0.05)
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception:
                pass
    def _setup_signals(self) -> None:
        def _handler(sig, frame):
            logger.info("Signal %s received, stopping...", sig)
            self._running = False
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
