"""Single-frame processing pipeline.

Executes the deterministic stage sequence:
  Frame -> Detection -> Embedding -> Tracking -> Depth/Spatial
        -> Query (4-tier) -> Physical Reasoning -> VLM

RT-CV:        YOLO26 detection (every frame)
RT-Embedding: Cosmos-Embed1 scene/crop embeddings (every frame)
RT-VLM:       Cosmos-Reason2 physical reasoning (on scene change)
              Qwen3-VL-30B-A3B scene understanding (on novel query)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from app.modules.depth import DepthProcessor
from app.modules.detector import Detector
from app.modules.embedder import Embedder
from app.modules.physical_reasoner import PhysicalReasoner
from app.modules.query_engine import (
    QueryEngine, TIER1_TYPES, TIER2_TYPES, TIER25_TYPES, QueryType,
)
from app.modules.reasoning import SceneReasoner
from app.modules.path_planner import PotentialFieldPlanner
from app.modules.bytetrack import ByteTracker as Tracker
from weda_optim import BufferPool
from weda_optim import Profiler
from weda_optim import Scheduler
from app.utils.alert_logger import AlertLogger
from app.utils.data_structures import (
    DevicePlacement,
    FramePacket,
    PipelineConfig,
    StageID,
)
from app.utils.logger import get_logger

logger = get_logger()


def _log_user_query(**kwargs) -> None:
    """Thin wrapper around app.utils.query_log.log_query that never raises.

    Kept local to the pipeline so a broken import (e.g. missing query_log
    during a partial upgrade) can't take down frame processing.
    """
    try:
        from app.utils.query_log import log_query
        log_query(**kwargs)
    except Exception:
        pass


class Pipeline:
    """Stateful, deterministic frame processor."""

    def __init__(
        self,
        config: PipelineConfig,
        detector: Detector,
        depth_proc: DepthProcessor,
        tracker: Tracker,
        embedder: Embedder,
        physical_reasoner: PhysicalReasoner,
        reasoner: SceneReasoner,
        scheduler: Scheduler,
        profiler: Profiler,
        pool: Optional[BufferPool] = None,
    ) -> None:
        self._config = config
        self._detector = detector
        self._depth = depth_proc
        self._tracker = tracker
        self._embedder = embedder
        self._physical = physical_reasoner
        self._reasoner = reasoner
        self._scheduler = scheduler
        self._profiler = profiler
        self._pool = pool
        # Embedding cache: track_id → {"bbox": np.ndarray, "emb": np.ndarray}
        self._embed_cache: dict[int, dict] = {}
        # Alert logger — JSONL file for safety events
        self._alert_logger = AlertLogger(config.alert_log_file)
        # Track previous alert state so we log only on transitions (no-alert → alert)
        self._prev_alert_state: bool = False
        # Safety-alert hysteresis per track — keyed by track_id so a different
        # object entering the zone restarts the enter counter rather than
        # inheriting confidence from a previously-tracked one. Map value:
        # [in_radius_consec, out_of_radius_consec, currently_alerting].
        self._safety_counters: dict[int, list] = {}
        # Frozen allowlist in lowercase for O(1) lookup. Empty means "all".
        _allow = getattr(config, "safety_alert_classes", ()) or ()
        self._safety_allow: set = {c.lower() for c in _allow}
        # Potential field path planner (zero cost when no target)
        self._planner = PotentialFieldPlanner(
            attractive_gain=config.nav_attractive_gain,
            repulsive_gain=config.nav_repulsive_gain,
            obstacle_radius=config.nav_obstacle_radius,
        )

    def process(self, packet: FramePacket) -> FramePacket:
        self._profiler.begin_frame()

        if packet.rgb is None:
            self._profiler.end_frame()
            return packet

        # Stage 1: RT-CV — Detection (GPU fixed, YOLO26 NMS-free)
        with self._scheduler.scheduled_execution(StageID.DETECTION.value) as placement:
            with self._profiler.time_stage("detection"):
                packet.detections = self._detector.detect(packet.rgb)

        # Stage 2: RT-Embedding — scene + crop embeddings
        if self._config.enable_embedding:
            with self._profiler.time_stage("embedding"):
                try:
                    # Frame-level: scene change detection (async — returns prev result)
                    emb, changed = self._embedder.embed_frame_async(packet.rgb)
                    packet.frame_embedding = emb
                    packet.scene_changed = changed

                    # Object-level: inject cached embeddings for stable detections,
                    # only re-embed new/moved objects (saves N×50ms per frame)
                    if packet.detections and emb is not None:
                        need_embed = []
                        for det in packet.detections:
                            cached = _lookup_embed_cache(det.bbox, self._embed_cache)
                            if cached is not None:
                                det.embedding = cached
                            else:
                                need_embed.append(det)
                        if need_embed:
                            new_embs = self._embedder.embed_crops(
                                packet.rgb, need_embed
                            )
                            for det, crop_emb in zip(need_embed, new_embs):
                                det.embedding = crop_emb
                except Exception as _emb_err:
                    logger.error("Embedding stage error: %s", _emb_err)

        # Stage 3: Tracking (CPU fixed) — IoU + appearance re-ID
        if packet.detections:
            with self._profiler.time_stage("tracking"):
                packet.detections = self._tracker.update(packet.detections)
            # Update embed cache now that detections have track IDs
            _update_embed_cache(packet.detections, self._embed_cache)

        # Propagate the active tracker lock onto the frame so downstream
        # consumers (renderer, alert logger) can highlight the pinned ID
        # without reaching into tracker internals.
        _locked = getattr(self._tracker, "locked_track_id", None)
        packet.locked_track_id = _locked if _locked is not None else -1

        # Stage 4: Depth colorization (AUTO — scheduler decides GPU/CPU)
        if packet.depth is not None:
            with self._scheduler.scheduled_execution(StageID.DEPTH.value) as placement:
                with self._profiler.time_stage("depth_colorize"):
                    packet.depth_colorized = self._depth.colorize(
                        packet.depth, placement
                    )

            # Stage 5: Spatial computation (CPU fixed)
            if packet.detections:
                with self._profiler.time_stage("spatial"):
                    packet.spatial = self._depth.compute_spatial(
                        packet.detections, packet.depth
                    )

                # ── Safety zone check with class allowlist + hysteresis ──
                # Pipeline:
                #   1. Estimate per-track distance (real depth first, bbox-area
                #      fallback only when depth is missing).
                #   2. Drop tracks whose class isn't in the allowlist (static
                #      furniture like tv/couch should never pulse the border).
                #   3. Apply per-track hysteresis: N consecutive in-radius
                #      frames to enter, M consecutive out frames to clear.
                #      Stops single-frame depth noise from flickering the alert.
                #   4. Among tracks currently in the "alerting" hysteresis
                #      state, pick the closest one as THE alert for this frame.
                import math as _math
                _frame_area = (packet.rgb.shape[0] * packet.rgb.shape[1]
                               if packet.rgb is not None else 0)
                _radius = self._config.safety_zone_radius
                _enter_n = getattr(self._config, "safety_alert_enter_frames", 3)
                _exit_n = getattr(self._config, "safety_alert_exit_frames", 5)

                def _effective_distance(_det, _si) -> float:
                    """Return best-effort metric distance, or +inf if unknown."""
                    if _si is not None and _si.distance_m > 0:
                        return float(_si.distance_m)
                    if _frame_area <= 0:
                        return float("inf")
                    _ba = (_det.bbox[2]-_det.bbox[0]) * (_det.bbox[3]-_det.bbox[1])
                    _ar = _ba / _frame_area
                    if _ar <= 0.15:
                        return float("inf")   # too small to judge
                    return round(0.4 / _math.sqrt(max(_ar, 0.01)), 2)

                _alert_candidate_tid = -1
                _alert_candidate_dist = float("inf")
                _seen_tids: set = set()

                for _idx, _det in enumerate(packet.detections):
                    _tid = _det.track_id
                    if _tid < 0:
                        continue  # unstable — no track state to key on
                    _seen_tids.add(_tid)
                    # Allowlist gate — class must be eligible to *ever* alert.
                    _cls_ok = (
                        not self._safety_allow
                        or (_det.class_name or "").lower() in self._safety_allow
                    )
                    _si = packet.spatial[_idx] if _idx < len(packet.spatial) else None
                    _dist = _effective_distance(_det, _si) if _cls_ok else float("inf")
                    _in_radius = _cls_ok and _dist < _radius

                    # Per-track counters: [in_consec, out_consec, alerting]
                    _st = self._safety_counters.get(_tid)
                    if _st is None:
                        _st = [0, 0, False]
                        self._safety_counters[_tid] = _st
                    if _in_radius:
                        _st[0] += 1
                        _st[1] = 0
                        if not _st[2] and _st[0] >= _enter_n:
                            _st[2] = True
                    else:
                        _st[1] += 1
                        _st[0] = 0
                        if _st[2] and _st[1] >= _exit_n:
                            _st[2] = False

                    # Candidate for "this frame's alert track" only if fully
                    # confirmed (past enter hysteresis) AND has a real distance.
                    if _st[2] and _dist < _alert_candidate_dist:
                        _alert_candidate_dist = _dist
                        _alert_candidate_tid = _tid

                # Prune counters for tracks that have fully disappeared so the
                # dict doesn't grow unbounded across long sessions.
                for _stale_tid in list(self._safety_counters.keys()):
                    if _stale_tid not in _seen_tids:
                        del self._safety_counters[_stale_tid]

                if _alert_candidate_tid >= 0:
                    packet.safety_alert = True
                    packet.safety_alert_distance = _alert_candidate_dist
                    packet.safety_alert_track_id = _alert_candidate_tid

                # Log to alert file on state transition (avoid flooding)
                if packet.safety_alert and not self._prev_alert_state:
                    _alert_cls = next(
                        (d.class_name for d in packet.detections
                         if d.track_id == _alert_candidate_tid),
                        "",
                    )
                    self._alert_logger.log_safety_alert(
                        track_id=_alert_candidate_tid,
                        distance_m=_alert_candidate_dist,
                        class_name=_alert_cls,
                        frame_id=packet.frame_id,
                    )
                self._prev_alert_state = packet.safety_alert

        # Stage 5.5: Navigation planning (potential field — only when target set)
        if (self._config.enable_navigation
                and self._tracker.target_track_id is not None
                and packet.spatial):
            _target_tid = self._tracker.target_track_id
            _target_xz = None
            _obstacle_xzs: list[tuple[float, float]] = []
            for _i, _det in enumerate(packet.detections):
                _si = packet.spatial[_i] if _i < len(packet.spatial) else None
                if _si and _si.position_3d is not None and _si.distance_m > 0:
                    _xz = (float(_si.position_3d[0]), float(_si.position_3d[2]))
                    if _det.track_id == _target_tid:
                        _target_xz = _xz
                    else:
                        _obstacle_xzs.append(_xz)
            if _target_xz is not None:
                packet.navigation_plan = self._planner.plan(_target_xz, _obstacle_xzs)
                packet.navigation_plan.target_track_id = _target_tid

        # Stage 6: Background VLM scene description + object caching
        if self._config.enable_vlm and packet.detections:
            # Build JSON grounding block for VLM — all sensor-verified metrics.
            # build_grounded_json includes bbox_norm, confidence, distance_m,
            # position_3d, physical_size, segmentation_coverage, nearest_object.
            grounded_json = ""
            if packet.spatial and packet.rgb is not None:
                grounded_json = QueryEngine.build_grounded_json(
                    packet.detections, packet.spatial, packet.rgb.shape,
                    self._reasoner.object_cache,
                )

            # Pass spatial so crop descriptions are also grounded
            self._reasoner.cache_object_descriptions(
                packet.rgb, packet.detections, packet.spatial
            )
            # Pass JSON context + spatial for annotated full-frame description
            if grounded_json:
                self._reasoner.request_description(
                    packet.rgb, packet.detections,
                    grounded_context=grounded_json,
                    spatial=packet.spatial,
                )
            active_ids = self._tracker.active_track_ids
            self._reasoner.prune_cache(active_ids)

        # Stage 7: Physical reasoning on scene change (async)
        if self._config.enable_physical_reasoning and packet.scene_changed:
            self._physical.request_assessment(
                packet.rgb, packet.detections, packet.spatial
            )
        packet.physical_reasoning = self._physical.latest_response

        # Stage 8: Query processing (4-tier routing)
        if packet.pending_query:
            import time as _time
            _q_t0 = _time.monotonic()
            with self._profiler.time_stage("query"):
                response, qtype = QueryEngine.process(
                    packet.pending_query,
                    packet.detections,
                    packet.spatial,
                    self._reasoner.object_cache,
                    alert_logger=self._alert_logger,
                    embedder=self._embedder,
                )

                # Per-tier user-query logging. Tier 3 is logged separately
                # from inside reasoning._process once the async worker
                # completes (it has the real elapsed + status). Tier 1/2/2.5
                # we log here synchronously because the rule-based / Tier 2.5
                # responses are effectively instantaneous.
                if qtype in TIER1_TYPES:
                    packet.query_response = response
                    _log_user_query(
                        tier=1, model="rule",
                        query=packet.pending_query, response=response,
                        elapsed_s=_time.monotonic() - _q_t0,
                    )

                elif qtype in TIER2_TYPES:
                    packet.query_response = response
                    _log_user_query(
                        tier=2, model="rule+cache",
                        query=packet.pending_query, response=response,
                        elapsed_s=_time.monotonic() - _q_t0,
                    )

                elif qtype in TIER25_TYPES:
                    # Tier 2.5: async Cosmos-Reason2 — no frame freeze
                    # Only start a new query if the previous one has finished
                    if not self._physical.query_running:
                        self._physical.request_query(
                            packet.rgb, packet.detections,
                            packet.spatial, packet.pending_query,
                        )
                        # Log the dispatch event — final response logged when
                        # latest_query_response updates (polled in
                        # orchestrator). Elapsed here is dispatch latency only.
                        _log_user_query(
                            tier=2.5, model="cosmos-reason2",
                            query=packet.pending_query,
                            response="[dispatched — streaming]",
                            elapsed_s=_time.monotonic() - _q_t0,
                            status="dispatched",
                        )
                    latest = self._physical.latest_query_response
                    packet.query_response = latest if latest else "Analyzing..."

                else:
                    # Tier 3: Qwen3-VL with grounded context. User queries use
                    # a dedicated worker + response slot so the auto scene
                    # description running in parallel can't silently drop the
                    # request or overwrite its answer.
                    grounded_ctx = response
                    self._reasoner.request_description(
                        packet.rgb, packet.detections,
                        grounded_context=grounded_ctx,
                        user_query=packet.pending_query,
                    )
                    latest = self._reasoner.latest_user_response
                    err = self._reasoner.user_error_message
                    if latest:
                        packet.query_response = latest
                    elif self._reasoner.user_query_running:
                        # Keep ASCII only — OpenCV's Hershey font can't render
                        # em-dashes or other Unicode punctuation (shows as `?`).
                        packet.query_response = (
                            "Thinking... VLM response streaming in"
                        )
                    elif self._reasoner.last_failed():
                        # Primary VLM (Qwen3-VL) timed out / OOM'd / warmup
                        # failed / pre-flight rejected. Re-dispatch the same
                        # question to Cosmos-Reason2 — already loaded as the
                        # Tier 2.5 physical reasoner — so the application
                        # keeps responding instead of leaving the user on a
                        # stale error message.
                        if not self._physical.query_running:
                            self._physical.request_query(
                                packet.rgb, packet.detections,
                                packet.spatial, packet.pending_query,
                            )
                            # Swallow the error once we've committed to the
                            # failover dispatch, so subsequent frames don't
                            # keep re-dispatching.
                            self._reasoner.clear_error()
                            _log_user_query(
                                tier=3, model="cosmos-reason2[fallback]",
                                query=packet.pending_query,
                                response="[fallback dispatched]",
                                elapsed_s=_time.monotonic() - _q_t0,
                                status="fallback",
                            )
                        latest_fb = self._physical.latest_query_response
                        packet.query_response = (
                            f"[fallback: Cosmos-Reason2] {latest_fb}"
                            if latest_fb
                            else "Falling back to Cosmos-Reason2..."
                        )
                    elif err:
                        # Worker exited without producing text and failover
                        # didn't apply (e.g. physical reasoner already busy
                        # on an unrelated Tier 2.5 query). Surface the reason
                        # so the operator knows to act.
                        packet.query_response = err
                    else:
                        packet.query_response = (
                            "Processing - response available shortly."
                        )

        # Always read latest descriptions
        packet.scene_description = self._reasoner.latest_description

        self._profiler.end_frame()
        return packet


# ---------------------------------------------------------------------------
# Embed cache helpers
# ---------------------------------------------------------------------------

def _bbox_iou_single(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _lookup_embed_cache(bbox: np.ndarray, cache: dict) -> Optional[np.ndarray]:
    """Return cached embedding if any cached bbox has IoU > 0.7 with bbox."""
    best_iou = 0.0
    best_emb = None
    for entry in cache.values():
        iou = _bbox_iou_single(bbox, entry["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_emb = entry["emb"]
    return best_emb if best_iou > 0.7 else None


def _update_embed_cache(detections: list, cache: dict) -> None:
    """Refresh cache with track IDs and bboxes from just-tracked detections."""
    active = set()
    for det in detections:
        if det.track_id >= 0 and det.embedding is not None:
            cache[det.track_id] = {"bbox": det.bbox.copy(), "emb": det.embedding}
            active.add(det.track_id)
    stale = [tid for tid in cache if tid not in active]
    for tid in stale:
        del cache[tid]
