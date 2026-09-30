"""Professional dark-themed display renderer for CV-VLM pipeline.

Layout (1920x1080 canvas, WINDOW_NORMAL scales to any display):

    +--------------------------------------------------------------+
    |  HEADER (1920x36) — pipeline name + status badges            |
    +-------------------------------------------+------------------+
    |                                           |                  |
    |  RGB HERO (1440x684)                      | INFO PANEL (480) |
    |  - Detection boxes with pill labels       | - FPS (large)    |
    |  - Depth PIP (288x216) top-right corner   | - GPU stats      |
    |  - Scene description overlay at bottom    | - Detections     |
    |                                           | - Tracker        |
    +-------------------------------------------+------------------+
    |  VLM RESPONSE (1200x360)   | QUERY INPUT (720x360)          |
    |  - Word-wrapped text       | - Large, prominent input       |
    +----------------------------+---------------------------------+

All rendering uses pre-allocated numpy buffers. Target: <2.5ms overhead.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from app.utils.data_structures import Detection, FramePacket

# ── Canvas dimensions ──
CANVAS_W, CANVAS_H = 1920, 1080
HEADER_H = 36
RGB_W, RGB_H = 1440, 684
INFO_X, INFO_W = 1440, 480
BOTTOM_Y, BOTTOM_H = 720, 360
VLM_W = 1200
QUERY_X, QUERY_W = 1200, 720
PIP_W, PIP_H = 288, 216
PIP_X, PIP_Y = RGB_W - PIP_W - 12, HEADER_H + 12

# ── Color palette (BGR) ──
BG_DARK = (18, 18, 18)
BG_PANEL = (30, 30, 30)
BG_HEADER = (35, 35, 35)
BG_INPUT = (40, 40, 40)
ACCENT_GREEN = (0, 200, 83)
ACCENT_CYAN = (220, 200, 0)
ACCENT_ORANGE = (0, 140, 255)
ACCENT_RED = (60, 60, 230)
TEXT_PRIMARY = (230, 230, 230)
TEXT_SECONDARY = (150, 150, 150)
TEXT_MUTED = (90, 90, 90)
SEPARATOR = (55, 55, 55)
PIP_BORDER = ACCENT_GREEN
QUERY_BORDER = (220, 160, 0)

# ── Font shortcuts ──
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Detection colors (vivid, high contrast on dark) ──
DET_COLORS = [
    (0, 200, 83), (255, 140, 0), (0, 180, 255), (200, 50, 255),
    (0, 255, 255), (255, 80, 80), (180, 255, 0), (255, 200, 0),
]


class DisplayRenderer:
    """Pre-allocated 1920x1080 dark-themed display renderer."""

    def __init__(self) -> None:
        self._canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        self._query_mode = False
        self._query_text = ""
        self._pending_query: Optional[str] = None
        self._last_response = ""
        self._fps = 0.0
        self._fps_count = 0
        self._fps_timer = time.monotonic()
        # Feature toggles
        self._show_masks = True
        self._show_rings = True
        self._show_birdseye = True
        # Press [X] to open a modal that shows the FULL, untruncated
        # text of all three response sections (QUERY RESPONSE, SCENE
        # UNDERSTANDING, PHYSICAL REASONING). Useful when the
        # bottom-panel per-section caps clip a long VLM answer. Esc
        # also dismisses. j/k or arrow keys scroll.
        self._show_response_modal: bool = False
        self._response_modal_scroll: int = 0
        # ── Per-section update timestamps ──
        # When a response stops being updated for RESPONSE_STALENESS_S
        # seconds, the bottom panel hides that row entirely so the
        # remaining (active) rows can claim its vertical budget. The
        # operator can still pull up the full text via [X] which
        # ignores staleness. Each section can be independently fresh
        # or stale — e.g. an old QUERY RESPONSE ages out while SCENE
        # is still being updated by the auto-caption worker.
        self._last_response_ts: float = 0.0
        self._last_physical_response_ts: float = 0.0
        self._last_scene_desc_text: str = ""
        self._last_scene_desc_ts: float = 0.0
        # Trail history: track_id → deque of (cx, cy) pixel positions
        import collections as _col
        self._track_history: dict[int, _col.deque] = {}
        self._trail_length = 20
        # Safety zone runtime control — cycle with [ / ] keys
        self._safety_presets = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        self._safety_preset_idx: int = 2  # default 1.5m
        # Last alert tracking for timestamp display
        self._last_alert_time: float = 0.0
        self._last_alert_class: str = ""
        # Physical reasoning dedicated response field
        self._last_physical_response: str = ""
        # ── Pillarbox widths stashed by _render_rgb_hero each frame so
        # the EVENT TIMELINE + ACTIVE TRACKS strips can fit themselves to
        # whatever vacant margin the letterboxed hero left behind.
        self._pillar_w_left: int = 0
        self._pillar_w_right: int = 0
        # ── Event timeline ring buffer (newest first when rendered) ──
        # Each entry: (monotonic_ts, category, message). 48 events ≈ the most
        # a 684-px strip can display at 14 px/line, and keeps memory trivial.
        self._events: _col.deque = _col.deque(maxlen=48)
        # Track-state deltas for auto-harvested timeline events
        self._prev_track_ids: dict[int, str] = {}   # tid -> class_name (last seen)
        self._last_scene_desc: str = ""
        self._last_query_text: str = ""
        self._last_response_logged: str = ""
        self._last_alert_logged_ts: float = 0.0
        self._session_start: float = time.monotonic()
        self._wall_start: float = time.time()
        # EMB LED state comes from the `embedder_status` render kwarg
        # (orchestrator reads `Embedder.is_loaded`), not a per-frame
        # packet field, so off-by-config, loading, ready and failed are
        # all distinguishable colours.

    def render(
        self,
        packet: FramePacket,
        profiler_summary: dict,
        gpu_stats: Optional[dict],
        scheduler_stats: dict,
        detector_backend: str,
        vlm_status: str,
        vlm_model: str,
        scene_desc: str,
        physical_desc: str,
        active_tracks: int,
        has_depth_stream: bool,
        depth_index: Optional[int],
        target_track_id: Optional[int] = None,
        display_ready_fn=None,  # callable(track_id) -> bool
        safety_alert: bool = False,
        safety_alert_distance: float = 0.0,
        safety_alert_class: str = "",
        fx: float = 615.0,
        dwell_alert_seconds: float = 30.0,
        gpu_backend: str = "",
        vlm_tok_per_sec: float = 0.0,
        physical_tok_per_sec: float = 0.0,
        camera_id: str = "",
        embedder_status: str = "disabled",
    ) -> np.ndarray:
        """Render full display frame. Returns the canvas (do NOT modify after)."""
        self._fx = fx
        canvas = self._canvas
        canvas[:] = BG_DARK

        # FPS counter
        self._fps_count += 1
        now = time.monotonic()
        if now - self._fps_timer >= 1.0:
            self._fps = self._fps_count / (now - self._fps_timer)
            self._fps_count = 0
            self._fps_timer = now

        # ── Header ──
        # active_tracks from the orchestrator can be a dict (per-class
        # counts) or an int total — normalise to int for the TRK LED.
        _tcount = active_tracks
        if isinstance(_tcount, dict):
            _tcount = sum(_tcount.values()) if _tcount else 0
        # EMB LED state now comes from the orchestrator (embedder_status
        # kwarg), derived directly from Embedder.is_loaded. No per-frame
        # latch — load state is already sticky by nature, and this way a
        # failed load shows orange instead of silent gray.
        self._render_header(canvas, packet, detector_backend, vlm_status,
                            camera_id=camera_id, active_tracks=_tcount or 0,
                            has_depth_stream=has_depth_stream,
                            physical_desc=physical_desc,
                            embedder_status=embedder_status,
                            physical_tok_per_sec=physical_tok_per_sec)

        # ── RGB Hero ──
        self._render_rgb_hero(canvas, packet, scene_desc, target_track_id,
                              display_ready_fn, dwell_alert_seconds)

        # ── Harvest + render EVENT TIMELINE into left pillarbox ──
        # (must come BEFORE the safety alert so the pulsing red border,
        # drawn afterwards, still frames the entire canvas visibly.)
        self._harvest_events(packet, scene_desc, safety_alert,
                             safety_alert_class, safety_alert_distance)
        self._render_event_timeline(canvas)

        # ── Safety Alert Overlay ──
        if safety_alert:
            self._last_alert_time = time.monotonic()
            self._last_alert_class = safety_alert_class
            self._render_safety_alert(canvas, safety_alert_distance, safety_alert_class)

        # ── Depth PIP ──
        self._render_depth_pip(canvas, packet)

        # ── Active Tracks strip (right pillarbox, below the PIP) ──
        self._render_active_tracks(canvas, packet)

        # ── Info Panel ──
        self._render_info_panel(
            canvas, packet, profiler_summary, gpu_stats,
            scheduler_stats, detector_backend, vlm_model,
            vlm_status, active_tracks, has_depth_stream, depth_index,
            gpu_backend=gpu_backend,
            vlm_tok_per_sec=vlm_tok_per_sec,
            physical_tok_per_sec=physical_tok_per_sec,
        )

        # ── Bird's Eye View ──
        if self._show_birdseye and packet.detections and packet.spatial:
            _alert_tid = getattr(packet, "safety_alert_track_id", -1) if safety_alert else -1
            _nav_plan = getattr(packet, "navigation_plan", None)
            self._render_birdseye(canvas, packet.detections, packet.spatial,
                                  _alert_tid, _nav_plan)

        # ── Bottom Bar ──
        self._render_bottom_bar(canvas, scene_desc, physical_desc)

        # ── Full-response modal overlay (toggled by [X]) ──
        # Drawn LAST so it sits on top of every other element when
        # active. Operator dismisses with [X] again or [Esc].
        if self._show_response_modal:
            self._render_response_modal(canvas, scene_desc, physical_desc)

        return canvas

    def handle_key(self, key: int) -> Optional[str]:
        """Process keypress. Returns action string or None."""
        if key == 255:
            return None

        if self._query_mode:
            if key == 27:  # Esc
                self._query_mode = False
                self._query_text = ""
            elif key == 13:  # Enter
                if self._query_text.strip():
                    self._pending_query = self._query_text.strip()
                    self._last_response = ""
                self._query_mode = False
                self._query_text = ""
            elif key == 8:  # Backspace
                self._query_text = self._query_text[:-1]
            elif 32 <= key < 127:
                self._query_text += chr(key)
            return None

        if key == ord("q"):
            return "quit"
        if key in (ord("t"), ord("T")):
            self._query_mode = True
            self._query_text = ""
        if key in (ord("f"), ord("F")):
            return "fullscreen"
        if key in (ord("r"), ord("R")):
            return "reset_cache"
        if key in (ord("m"), ord("M")):
            self._show_masks = not self._show_masks
        if key in (ord("g"), ord("G")):
            self._show_rings = not self._show_rings
        if key in (ord("b"), ord("B")):
            self._show_birdseye = not self._show_birdseye
        if key in (ord("x"), ord("X")):
            # Toggle the full-response modal. Reset scroll on each
            # open so the operator sees the top of the response.
            self._show_response_modal = not self._show_response_modal
            if self._show_response_modal:
                self._response_modal_scroll = 0
        if key == 27 and self._show_response_modal:  # Esc dismisses modal
            self._show_response_modal = False
            self._response_modal_scroll = 0
        # PgUp / PgDn / arrow / vim-style scroll keys for the modal.
        # cv2 keycodes for arrows differ across builds — accept the
        # common variants. Each press scrolls one line; PgUp/PgDn 8.
        if self._show_response_modal:
            if key in (82, 0xFF52, ord("k")):  # up
                self._response_modal_scroll = max(0, self._response_modal_scroll - 1)
            if key in (84, 0xFF54, ord("j")):  # down
                self._response_modal_scroll += 1   # clamped at render time
            if key == 33:  # PgUp
                self._response_modal_scroll = max(0, self._response_modal_scroll - 8)
            if key == 34:  # PgDn
                self._response_modal_scroll += 8
        if key == ord("["):
            self._safety_preset_idx = max(0, self._safety_preset_idx - 1)
        if key == ord("]"):
            self._safety_preset_idx = min(len(self._safety_presets) - 1, self._safety_preset_idx + 1)
        return None

    @property
    def safety_zone_radius(self) -> float:
        """Current safety zone radius (metres). Adjustable via [ / ] keys at runtime."""
        return self._safety_presets[self._safety_preset_idx]

    def consume_query(self) -> Optional[str]:
        q = self._pending_query
        self._pending_query = None
        return q

    def set_physical_response(self, text: str) -> None:
        """Store a physical-reasoner query response for the PHYSICAL REASONING section."""
        # Bump the freshness timestamp on every set, including
        # incremental token-streaming updates (caller passes the
        # full accumulated text each time). That keeps the section
        # visible while tokens are flowing; once streaming stops the
        # 60 s staleness timer takes over.
        if text:
            self._last_physical_response_ts = time.monotonic()
        self._last_physical_response = text

    def set_response(self, text: str) -> None:
        if text:
            self._last_response_ts = time.monotonic()
        self._last_response = text

    # ══════════════════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════════════════

    def _render_header(self, c: np.ndarray, pkt: FramePacket,
                       backend: str, vlm_status: str,
                       camera_id: str = "",
                       active_tracks: int = 0,
                       has_depth_stream: bool = False,
                       physical_desc: str = "",
                       physical_tok_per_sec: float = 0.0,
                       embedder_status: str = "disabled") -> None:
        """Header bar: title on the left, subsystem-status LEDs in the
        middle, wall-clock / uptime / FPS / frame counter on the right.

        LED source of truth — all session-level signals now, so a single
        transient per-frame packet field never drops a LED that's actually
        alive (fixes DPT / EMB / PHY flicker seen prior to Apr-2026):
          DET — green when the backend is anything other than Mock / empty.
          DPT — green when the depth stream is attached (has_depth_stream)
                OR the current frame already carries a colorized depth map.
          TRK — green when any detection carries track_id >= 0 (active_tracks).
          EMB — three-state: green when loaded, orange when enabled-but-
                failed (so operator sees missing deps instead of thinking
                the feature is "just off"), gray when disabled in config.
                Driven by the `embedder_status` kwarg from orchestrator —
                reads Embedder.is_loaded directly, no latching needed since
                load state is already sticky.
          VLM — from the vlm_status string ("active" / "loading" / else).
          PHY — green when physical_desc is non-empty OR Cosmos-Reason2 is
                actively producing tokens (physical_tok_per_sec > 0).
        """
        cv2.rectangle(c, (0, 0), (CANVAS_W, HEADER_H), BG_HEADER, -1)
        cv2.line(c, (0, HEADER_H - 1), (CANVAS_W, HEADER_H - 1), SEPARATOR, 1)

        # ── Title (left) ──
        cv2.putText(c, "WEDA Edge CV-VLM", (16, 24),
                    FONT, 0.55, TEXT_PRIMARY, 1)
        (title_w, _), _ = cv2.getTextSize("WEDA Edge CV-VLM", FONT, 0.55, 1)
        # Camera id / source (muted, immediately after title)
        if camera_id:
            cam_lbl = f"cam:{camera_id[:20]}"
            cv2.putText(c, cam_lbl, (16 + title_w + 14, 24),
                        FONT, 0.38, TEXT_MUTED, 1)

        # ── LED cluster (centre-left) ──
        # Session-level signals ensure subsystems don't flicker off during
        # legitimate empty frames (e.g. depth pool release, embedder cooldown,
        # physical reasoner between assessments). See docstring above.
        _dpt_alive = bool(has_depth_stream) or (
            getattr(pkt, "depth_colorized", None) is not None
        ) or (pkt.depth is not None)
        _phy_alive = bool(physical_desc) or physical_tok_per_sec > 0 or bool(
            getattr(pkt, "physical_reasoning", "")
        )
        # EMB LED colour map — keep this table in lock-step with the
        # `embedder_status` values set in orchestrator._render_kwargs:
        #   "loaded"   → green   (Cosmos-Embed1 alive, embeddings flowing)
        #   "failed"   → orange  (enable_embedding: true but load failed —
        #                         usually missing einops, OOM, or HF 403;
        #                         actionable: check /tmp/einops.log)
        #   "disabled" → gray    (enable_embedding: false by operator choice)
        #   (anything else) → gray, safe fallback
        _emb_col = (
            ACCENT_GREEN if embedder_status == "loaded"
            else (ACCENT_ORANGE if embedder_status == "failed" else TEXT_MUTED)
        )
        leds: list[tuple] = [
            ("DET", ACCENT_GREEN if (backend and backend.lower() not in ("mock", "unknown", "")) else TEXT_MUTED),
            ("DPT", ACCENT_GREEN if _dpt_alive else TEXT_MUTED),
            ("TRK", ACCENT_GREEN if active_tracks and active_tracks > 0 else TEXT_MUTED),
            ("EMB", _emb_col),
            ("VLM",
                ACCENT_GREEN if vlm_status == "active"
                else (ACCENT_ORANGE if vlm_status == "loading" else TEXT_MUTED)),
            ("PHY", ACCENT_GREEN if _phy_alive else TEXT_MUTED),
        ]
        led_x = max(360, 16 + title_w + 180)
        for code, col in leds:
            # Filled dot + code pill
            cv2.circle(c, (led_x + 6, 18), 5, col, -1)
            cv2.putText(c, code, (led_x + 16, 24), FONT, 0.42,
                        (TEXT_PRIMARY if col != TEXT_MUTED else TEXT_SECONDARY), 1)
            led_x += 64

        # ── Right cluster: wall clock / uptime / FPS / frame ──
        import time as _t
        uptime = max(0.0, time.monotonic() - self._session_start)
        mins, secs = divmod(int(uptime), 60)
        hrs, mins = divmod(mins, 60)
        uptime_s = f"{hrs:d}:{mins:02d}:{secs:02d}"
        wall_s = _t.strftime("%H:%M:%S", _t.localtime())

        badges: list[tuple[str, tuple]] = [
            (f"FRAME {pkt.frame_id}", TEXT_SECONDARY),
            (f"UP {uptime_s}", TEXT_SECONDARY),
            (wall_s, TEXT_PRIMARY),
            (f"FPS {self._fps:.0f}",
                ACCENT_GREEN if self._fps > 20 else ACCENT_ORANGE),
        ]
        x = CANVAS_W - 16
        for label, color in reversed(badges):
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.4, 1)
            x -= tw + 18
            cv2.rectangle(c, (x, 8), (x + tw + 12, 28), (50, 50, 50), -1)
            cv2.rectangle(c, (x, 8), (x + tw + 12, 28), color, 1)
            cv2.putText(c, label, (x + 6, 23), FONT, 0.4, color, 1)

    # ══════════════════════════════════════════════════════════════════
    # RGB HERO
    # ══════════════════════════════════════════════════════════════════

    def _render_rgb_hero(self, c: np.ndarray, pkt: FramePacket,
                         scene_desc: str,
                         target_track_id: Optional[int] = None,
                         display_ready_fn=None,
                         dwell_alert_seconds: float = 30.0) -> None:
        if pkt.rgb is None:
            return
        rgb = _as_numpy(pkt.rgb)
        fh, fw = rgb.shape[:2]

        frame = rgb.copy()

        # ── Prune track history for gone tracks ──
        current_ids = {d.track_id for d in pkt.detections}
        self._track_history = {k: v for k, v in self._track_history.items()
                                if k in current_ids}

        for i, det in enumerate(pkt.detections):
            if display_ready_fn and not display_ready_fn(det.track_id):
                continue

            is_target = (target_track_id is not None and det.track_id == target_track_id)
            is_dimmed = (target_track_id is not None and det.track_id != target_track_id)

            if is_dimmed:
                color = (80, 80, 80)
                thickness = 1
            elif is_target:
                color = (0, 255, 255)
                thickness = 3
            else:
                color = DET_COLORS[det.track_id % len(DET_COLORS)] if det.track_id >= 0 else ACCENT_GREEN
                thickness = 2

            # ── Trail history ──
            import collections as _col
            cx_i, cy_i = int(det.center[0]), int(det.center[1])
            if det.track_id not in self._track_history:
                self._track_history[det.track_id] = _col.deque(maxlen=self._trail_length)
            self._track_history[det.track_id].append((cx_i, cy_i))
            trail_pts = list(self._track_history[det.track_id])
            n = len(trail_pts)
            for k in range(1, n):
                alpha = k / n
                tc = tuple(int(ch * alpha * 0.7) for ch in color)
                cv2.line(frame, trail_pts[k - 1], trail_pts[k], tc, 1)

            x1, y1 = int(det.bbox[0]), int(det.bbox[1])
            x2, y2 = int(det.bbox[2]), int(det.bbox[3])

            # ── Segmentation mask overlay ──
            if self._show_masks and det.mask is not None:
                try:
                    mask_bool = det.mask > 127
                    if mask_bool.shape[:2] == frame.shape[:2] and mask_bool.any():
                        overlay = frame.copy()
                        overlay[mask_bool, 0] = color[0]
                        overlay[mask_bool, 1] = color[1]
                        overlay[mask_bool, 2] = color[2]
                        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                except Exception:
                    pass

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            if not is_dimmed:
                si = pkt.spatial[i] if i < len(pkt.spatial) else None
                # [LOCK] prefix when this detection IS the pinned track.
                # locked_track_id is propagated by pipeline.py; -1 means no lock.
                _lock_prefix = (
                    "[LOCK] "
                    if getattr(pkt, "locked_track_id", -1) == det.track_id
                    and det.track_id >= 0
                    else ""
                )
                label = f"{_lock_prefix}#{det.track_id} {det.class_name} {det.confidence:.0%}"
                if si and si.distance_m > 0:
                    # Tilde prefix when depth came from expanded fallback ROI
                    _prefix = "" if getattr(si, "depth_reliable", True) else "~"
                    label += f" {_prefix}{si.distance_m:.1f}m"
                if si and getattr(si, "speed_ms", 0) > 0.1:
                    arrow = "v" if getattr(si, "approaching", False) else ">"
                    label += f" {arrow}{si.speed_ms:.1f}m/s"
                # Side indicator: L/R/C based on 3D X position
                if si and si.distance_m > 0 and si.position_3d is not None:
                    _x = si.position_3d[0]
                    _side = "L" if _x < -0.25 else ("R" if _x > 0.25 else "C")
                    label += f" [{_side}]"
                if is_target:
                    label = f"[TARGET] {label}"
                # Approaching: flash label red
                lbl_color = (0, 0, 0) if not getattr(si, "approaching", False) else (0, 0, 180)
                (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
                pill_y = max(0, y1 - th - 10)
                bg_c = color if not getattr(si, "approaching", False) else (0, 0, 200)
                cv2.rectangle(frame, (x1, pill_y), (x1 + tw + 10, pill_y + th + 8), bg_c, -1)
                cv2.putText(frame, label, (x1 + 5, pill_y + th + 3), FONT, 0.45, lbl_color, 1)

                # ── Dwell badge ──
                if si and getattr(si, "dwell_seconds", 0) >= dwell_alert_seconds:
                    dwell_lbl = f"T:{si.dwell_seconds:.0f}s"
                    cv2.putText(frame, dwell_lbl, (x1, y2 + 14), FONT, 0.38,
                                ACCENT_ORANGE, 1)

                # ── Lean / instability badge ──
                if getattr(det, "is_unstable", False):
                    cv2.putText(frame, "LEAN", (x1, y2 + 28), FONT, 0.38,
                                ACCENT_ORANGE, 1)

        # ── Distance rings overlay ──
        if self._show_rings:
            fx = getattr(self, "_fx", 615.0)
            arc_cx, arc_cy = fw // 2, fh
            for dist_m, lbl in ((1, "1m"), (2, "2m"), (3, "3m"), (5, "5m")):
                r = int(fx * dist_m / 2)
                if 10 < r < fw * 3:
                    cv2.ellipse(frame, (arc_cx, arc_cy), (r, r // 2),
                                0, 180, 360, (55, 55, 55), 1)
                    lx = min(arc_cx + r + 3, fw - 24)
                    cv2.putText(frame, lbl, (lx, arc_cy - 3), FONT, 0.32, (65, 65, 65), 1)

        # Upscale / letterbox
        scale = min(RGB_W / fw, RGB_H / fh)
        new_w, new_h = int(fw * scale), int(fh * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        hero = np.zeros((RGB_H, RGB_W, 3), dtype=np.uint8)
        x_off = (RGB_W - new_w) // 2
        y_off = (RGB_H - new_h) // 2
        hero[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        # Stash pillarbox widths for the EVENT TIMELINE / ACTIVE TRACKS
        # strips — they fit themselves to whatever margin is available.
        self._pillar_w_left = x_off
        self._pillar_w_right = RGB_W - (x_off + new_w)

        # ── Toggle status HUD ──
        hud = (f"[M]masks:{'ON' if self._show_masks else 'OFF'}  "
               f"[G]rings:{'ON' if self._show_rings else 'OFF'}  "
               f"[B]map:{'ON' if self._show_birdseye else 'OFF'}")
        cv2.putText(hero, hud, (8, RGB_H - 48), FONT, 0.35, TEXT_MUTED, 1)

        if scene_desc:
            oy = RGB_H - 34
            overlay = hero[oy:RGB_H, :].copy()
            cv2.rectangle(overlay, (0, 0), (RGB_W, 34), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, hero[oy:RGB_H, :], 0.4, 0, hero[oy:RGB_H, :])
            cv2.putText(hero, scene_desc[:120], (12, RGB_H - 10), FONT, 0.45, TEXT_PRIMARY, 1)

        c[HEADER_H:HEADER_H + RGB_H, 0:RGB_W] = hero

    # ══════════════════════════════════════════════════════════════════
    # EVENT TIMELINE (left pillarbox)
    # ══════════════════════════════════════════════════════════════════
    #
    # Categories (single-char codes chosen to render cleanly in Hershey
    # Simplex — no Unicode, no em-dashes):
    #   Q  user query dispatched
    #   A  VLM answer arrived
    #   !  safety alert fired
    #   +  new track appeared
    #   -  track vanished
    #   ~  scene change detected
    # ------------------------------------------------------------------

    _EVT_COLORS: dict = {
        "Q": (220, 160, 0),     # amber — query bus colour
        "A": (0, 200, 83),      # green — answer OK
        "!": (60, 60, 230),     # red   — alert
        "+": (180, 200, 200),   # light grey — track +
        "-": (90, 90, 90),      # muted grey — track -
        "~": (0, 180, 255),     # cyan  — scene delta
    }

    def _push_event(self, category: str, message: str) -> None:
        """Append an event to the ring buffer. Never raises."""
        try:
            self._events.append((time.time(), str(category)[:1] or " ",
                                 str(message)[:48]))
        except Exception:
            pass

    def _harvest_events(self, pkt: FramePacket, scene_desc: str,
                       safety_alert: bool,
                       safety_alert_class: str,
                       safety_alert_distance: float) -> None:
        """Diff the current packet against prior state and log any new
        events. Called once per frame from render(). All comparisons are
        defensive — missing fields degrade silently rather than crash."""
        # ── Track appearance / disappearance ──
        current: dict[int, str] = {}
        for det in pkt.detections or []:
            tid = getattr(det, "track_id", -1)
            if tid is None or tid < 0:
                continue
            current[tid] = getattr(det, "class_name", "?") or "?"
        prev = self._prev_track_ids
        for tid, cname in current.items():
            if tid not in prev:
                self._push_event("+", f"track#{tid} {cname[:10]}")
        for tid, cname in prev.items():
            if tid not in current:
                self._push_event("-", f"track#{tid} {cname[:10]}")
        self._prev_track_ids = current

        # ── Scene change signal from the pipeline ──
        if getattr(pkt, "scene_changed", False) and scene_desc and \
                scene_desc != self._last_scene_desc:
            self._push_event("~", "scene change")
            self._last_scene_desc = scene_desc

        # ── User queries & responses ──
        q = getattr(pkt, "pending_query", None)
        if q and q != self._last_query_text:
            self._push_event("Q", q.strip())
            self._last_query_text = q
        resp = getattr(pkt, "query_response", None)
        if resp and resp != self._last_response_logged:
            # Skip the "Thinking..." / "Processing..." placeholders — they
            # aren't answers, they're status. Only log final responses.
            _first = resp.split(None, 1)[0].lower() if resp else ""
            if _first not in ("thinking...", "processing...", "analyzing..."):
                short = resp[:46].replace("\n", " ")
                self._push_event("A", short)
                self._last_response_logged = resp

        # ── Safety alert edge (once per alert latch, not every frame) ──
        if safety_alert:
            now = time.monotonic()
            if now - self._last_alert_logged_ts > 2.0:
                _cls = (safety_alert_class or "object")[:10]
                self._push_event("!", f"{_cls} @{safety_alert_distance:.1f}m")
                self._last_alert_logged_ts = now

    def _render_event_timeline(self, c: np.ndarray) -> None:
        """Render the scrolling EVENT TIMELINE into the left pillarbox.

        Silently no-op when there is less than 120 px of pillar margin
        (e.g. wide-aspect source like 16:9 upscaled). A narrower strip
        would just overlap the camera feed."""
        pw = self._pillar_w_left
        if pw < 120:
            return
        # Leave 10 px breathing room between the strip and the hero image.
        panel_w = pw - 10
        x0, x1 = 0, panel_w
        y0, y1 = HEADER_H, HEADER_H + RGB_H
        # Panel background + subtle right-edge separator
        c[y0:y1, x0:x1] = BG_PANEL
        cv2.line(c, (x1, y0), (x1, y1), SEPARATOR, 1)

        # Header
        cv2.putText(c, "EVENT TIMELINE", (8, y0 + 18),
                    FONT, 0.42, TEXT_PRIMARY, 1)
        cv2.line(c, (6, y0 + 26), (panel_w - 6, y0 + 26), SEPARATOR, 1)

        # ── Event rows (newest first) ──
        row_h = 14
        top = y0 + 36
        max_rows = max(1, (y1 - top - 6) // row_h)
        events = list(self._events)[-max_rows:][::-1]
        if not events:
            cv2.putText(c, "(no events yet)", (8, top + 8),
                        FONT, 0.32, TEXT_MUTED, 1)
            return

        # Time formatting: local clock HH:MM:SS — operators recognise wall
        # time faster than "12.8s ago" in an incident report.
        import time as _t
        for i, (ts, cat, msg) in enumerate(events):
            y = top + i * row_h + 10
            if y > y1 - 6:
                break
            col = self._EVT_COLORS.get(cat, TEXT_SECONDARY)
            hhmmss = _t.strftime("%H:%M:%S", _t.localtime(ts))
            # Category badge (coloured single-char pill)
            cv2.rectangle(c, (6, y - 10), (20, y + 2), col, -1)
            cv2.putText(c, cat, (10, y), FONT, 0.38, (0, 0, 0), 1)
            # Timestamp
            cv2.putText(c, hhmmss, (24, y), FONT, 0.34, TEXT_SECONDARY, 1)
            # Message (truncated to fit)
            max_chars = max(6, (panel_w - 82) // 6)
            cv2.putText(c, msg[:max_chars], (82, y),
                        FONT, 0.34, TEXT_PRIMARY, 1)

    # ══════════════════════════════════════════════════════════════════
    # DEPTH PIP
    # ══════════════════════════════════════════════════════════════════

    def _render_depth_pip(self, c: np.ndarray, pkt: FramePacket) -> None:
        if pkt.depth_colorized is None:
            return
        depth = _as_numpy(pkt.depth_colorized)
        pip = cv2.resize(depth, (PIP_W, PIP_H), interpolation=cv2.INTER_NEAREST)

        # Green border
        py, px = PIP_Y, PIP_X
        cv2.rectangle(c, (px - 2, py - 2), (px + PIP_W + 2, py + PIP_H + 2), PIP_BORDER, 2)
        c[py:py + PIP_H, px:px + PIP_W] = pip

        # Label
        cv2.putText(c, "DEPTH", (px + 8, py + 18), FONT, 0.45, ACCENT_GREEN, 1)

    # ══════════════════════════════════════════════════════════════════
    # ACTIVE TRACKS (right pillarbox, below the depth PIP)
    # ══════════════════════════════════════════════════════════════════

    def _render_active_tracks(self, c: np.ndarray, pkt: FramePacket) -> None:
        """Render one compact card per active track in the right pillarbox
        below the depth PIP. Silently no-op when the pillarbox is narrower
        than 120 px (wide-aspect source) or when there are no tracks."""
        pw = self._pillar_w_right
        if pw < 120:
            return

        # Panel bounds — left edge starts at the beginning of the right
        # pillarbox (with 4 px gap after the hero image), right edge stops
        # 2 px shy of the canvas column RGB_W so the seam to the info
        # panel stays clean.
        panel_l = RGB_W - pw + 4
        panel_r = RGB_W - 2
        panel_t = PIP_Y + PIP_H + 12        # below depth PIP
        panel_b = HEADER_H + RGB_H - 4       # just above the bottom bar
        if panel_r - panel_l < 120 or panel_b - panel_t < 60:
            return

        # Background + left-edge separator
        c[panel_t:panel_b, panel_l:panel_r] = BG_PANEL
        cv2.line(c, (panel_l, panel_t), (panel_l, panel_b), SEPARATOR, 1)

        # Header
        cv2.putText(c, "ACTIVE TRACKS", (panel_l + 8, panel_t + 18),
                    FONT, 0.42, TEXT_PRIMARY, 1)
        cv2.line(c, (panel_l + 6, panel_t + 26),
                 (panel_r - 6, panel_t + 26), SEPARATOR, 1)

        # ── One card per tracked detection ──
        # Filter to detections that have a real track_id (>=0). Pair each
        # with its spatial entry by index so distance / speed / side can
        # be pulled cheaply without another lookup pass.
        rows: list[tuple] = []
        for i, det in enumerate(pkt.detections or []):
            tid = getattr(det, "track_id", -1)
            if tid is None or tid < 0:
                continue
            si = pkt.spatial[i] if i < len(pkt.spatial or []) else None
            rows.append((tid, det, si))
        if not rows:
            cv2.putText(c, "(no active tracks)",
                        (panel_l + 8, panel_t + 46),
                        FONT, 0.34, TEXT_MUTED, 1)
            return

        # Sort: target track first, then approaching, then by nearest
        # distance so the operator sees the most actionable track on top.
        def _sort_key(r):
            tid, det, si = r
            d = (si.distance_m if (si and si.distance_m > 0) else 999.0)
            is_approaching = bool(si and getattr(si, "approaching", False))
            return (0 if is_approaching else 1, d)
        rows.sort(key=_sort_key)

        # Card geometry
        card_h = 44
        gap = 4
        y = panel_t + 34
        max_cards = max(1, (panel_b - y) // (card_h + gap))
        locked_tid = getattr(pkt, "locked_track_id", -1)

        for tid, det, si in rows[:max_cards]:
            if y + card_h > panel_b:
                break
            col = DET_COLORS[tid % len(DET_COLORS)]
            # Card background
            cv2.rectangle(c, (panel_l + 6, y),
                          (panel_r - 6, y + card_h),
                          (26, 26, 26), -1)
            # Left-edge class colour bar (visually matches the bbox colour)
            cv2.rectangle(c, (panel_l + 6, y),
                          (panel_l + 9, y + card_h), col, -1)
            # Row 1: [LOCK] #id classname
            lock = "[LOCK] " if tid == locked_tid else ""
            cname = (det.class_name or "?")[:10]
            title = f"{lock}#{tid} {cname}"
            cv2.putText(c, title, (panel_l + 14, y + 15),
                        FONT, 0.40, TEXT_PRIMARY, 1)
            # Row 2: distance / speed / side
            dist_s = f"{si.distance_m:.1f}m" if (si and si.distance_m > 0) else "--"
            spd = getattr(si, "speed_ms", 0.0) if si else 0.0
            approaching = bool(si and getattr(si, "approaching", False))
            arrow = "v" if approaching else ">"
            spd_s = f"{arrow}{spd:.1f}m/s" if spd > 0.1 else ""
            side_s = ""
            if si and si.position_3d is not None and si.distance_m > 0:
                _x = float(si.position_3d[0])
                side_s = "L" if _x < -0.25 else ("R" if _x > 0.25 else "C")
            line2 = f"{dist_s}  {spd_s}  {side_s}".strip()
            line_col = ACCENT_RED if approaching else TEXT_SECONDARY
            cv2.putText(c, line2, (panel_l + 14, y + 34),
                        FONT, 0.36, line_col, 1)
            # Dwell badge if extended presence
            if si and getattr(si, "dwell_seconds", 0) > 0:
                dwell_s = f"{si.dwell_seconds:.0f}s"
                (tw, _), _ = cv2.getTextSize(dwell_s, FONT, 0.34, 1)
                cv2.putText(c, dwell_s,
                            (panel_r - tw - 10, y + 15),
                            FONT, 0.34, ACCENT_ORANGE, 1)
            y += card_h + gap

        # "+N more" spill line when we had to truncate
        extra = len(rows) - max_cards
        if extra > 0 and y + 14 <= panel_b:
            cv2.putText(c, f"+{extra} more",
                        (panel_l + 14, y + 10),
                        FONT, 0.34, TEXT_MUTED, 1)

    # ══════════════════════════════════════════════════════════════════
    # SAFETY ALERT
    # ══════════════════════════════════════════════════════════════════

    def _render_safety_alert(
        self, c: np.ndarray, distance_m: float, class_name: str = ""
    ) -> None:
        """Draw pulsing red border + alert banner when an object is too close."""
        import math as _math
        # Pulsing border thickness (2→6px at ~2Hz)
        _pulse = int(2 + 4 * abs(_math.sin(time.monotonic() * 3.14)))
        cv2.rectangle(c, (0, HEADER_H), (RGB_W, HEADER_H + RGB_H), ACCENT_RED, _pulse)
        # Alert banner text -- always show real distance when available
        dist_str = f"{distance_m:.2f}m" if distance_m > 0 else "CLOSE"
        cls_str = f" [{class_name}]" if class_name else ""
        msg = f"  [!] SAFETY ALERT{cls_str}  {dist_str}  "
        (tw, th), _ = cv2.getTextSize(msg, FONT, 0.7, 2)
        bx = (RGB_W - tw) // 2
        by = HEADER_H + 18
        cv2.rectangle(c, (bx - 8, by - th - 6), (bx + tw + 8, by + 8), (0, 0, 0), -1)
        cv2.rectangle(c, (bx - 8, by - th - 6), (bx + tw + 8, by + 8), ACCENT_RED, 2)
        cv2.putText(c, msg, (bx, by), FONT, 0.7, ACCENT_RED, 2)

    # ══════════════════════════════════════════════════════════════════
    # BIRD'S EYE VIEW
    # ══════════════════════════════════════════════════════════════════

    def _render_birdseye(self, c: np.ndarray, detections, spatial,
                         alert_track_id: int = -1, nav_plan=None) -> None:
        """Top-down map in lower info panel: X=lateral, Z=depth."""
        MAP_X, MAP_Y = INFO_X + 8, BOTTOM_Y - 236
        MAP_W, MAP_H = INFO_W - 16, 228
        X_RANGE = 5.0   # ±5m lateral
        Z_RANGE = 10.0  # 0–10m depth

        panel = np.full((MAP_H, MAP_W, 3), 22, dtype=np.uint8)

        # Grid lines at 2m depth intervals
        for z_m in range(0, int(Z_RANGE) + 1, 2):
            py = MAP_H - int(z_m / Z_RANGE * MAP_H)
            cv2.line(panel, (0, py), (MAP_W, py), (42, 42, 42), 1)
            cv2.putText(panel, f"{z_m}m", (2, max(py - 2, 10)), FONT, 0.28, (58, 58, 58), 1)

        # Camera marker at bottom center
        cam_x, cam_y = MAP_W // 2, MAP_H - 4
        cv2.drawMarker(panel, (cam_x, cam_y), ACCENT_GREEN,
                       cv2.MARKER_TRIANGLE_UP, 8, 1)

        for det, si in zip(detections, spatial):
            if si.position_3d is None or si.distance_m <= 0:
                continue
            wx = float(si.position_3d[0])
            wz = float(si.position_3d[2])
            px = int(MAP_W / 2 + wx / X_RANGE * (MAP_W / 2))
            py = MAP_H - int(wz / Z_RANGE * MAP_H)
            px = max(4, min(MAP_W - 4, px))
            py = max(4, min(MAP_H - 4, py))
            col = DET_COLORS[det.track_id % len(DET_COLORS)] if det.track_id >= 0 else ACCENT_GREEN
            cv2.circle(panel, (px, py), 5, col, -1)
            # Alert track: outer ring in red
            if det.track_id == alert_track_id:
                cv2.circle(panel, (px, py), 9, ACCENT_RED, 1)
            # Label: class abbreviation + track id (e.g. "PER3", "BOT7")
            abbrev = det.class_name[:3].upper() if det.class_name else "OBJ"
            tid_str = str(det.track_id) if det.track_id >= 0 else "?"
            cv2.putText(panel, f"{abbrev}{tid_str}", (px + 4, py + 4), FONT, 0.28, col, 1)
            # Velocity arrow
            vel = getattr(si, "velocity_3d", None)
            spd = getattr(si, "speed_ms", 0)
            if vel is not None and spd > 0.1:
                ex = px + int(float(vel[0]) * 12)
                ey = py - int(float(vel[2]) * 12)
                ex = max(0, min(MAP_W - 1, ex))
                ey = max(0, min(MAP_H - 1, ey))
                cv2.arrowedLine(panel, (px, py), (ex, ey), col, 1, tipLength=0.35)

        # Navigation path (potential field planner)
        if nav_plan and hasattr(nav_plan, 'waypoints') and len(nav_plan.waypoints) > 1:
            _path_pts = []
            for _wx, _wz in nav_plan.waypoints:
                _ppx = int(MAP_W / 2 + _wx / X_RANGE * (MAP_W / 2))
                _ppy = MAP_H - int(_wz / Z_RANGE * MAP_H)
                _ppx = max(0, min(MAP_W - 1, _ppx))
                _ppy = max(0, min(MAP_H - 1, _ppy))
                _path_pts.append((_ppx, _ppy))
            _path_col = ACCENT_GREEN if nav_plan.path_clear else (0, 165, 255)
            for _pi in range(len(_path_pts) - 1):
                cv2.line(panel, _path_pts[_pi], _path_pts[_pi + 1], _path_col, 1, cv2.LINE_AA)
            # Target marker (diamond) at end of path
            if _path_pts:
                cv2.drawMarker(panel, _path_pts[-1], ACCENT_GREEN,
                               cv2.MARKER_DIAMOND, 8, 1)
                cv2.putText(panel, f"{nav_plan.path_length_m:.1f}m",
                            (_path_pts[-1][0] + 6, _path_pts[-1][1] - 4),
                            FONT, 0.26, ACCENT_GREEN, 1)

        cv2.putText(panel, "BIRD'S EYE", (4, 12), FONT, 0.32, TEXT_SECONDARY, 1)
        # Guard: don't write outside canvas
        y1c = max(0, MAP_Y)
        y2c = min(c.shape[0], MAP_Y + MAP_H)
        x1c = max(0, MAP_X)
        x2c = min(c.shape[1], MAP_X + MAP_W)
        ph = y2c - y1c
        pw = x2c - x1c
        if ph > 0 and pw > 0:
            c[y1c:y2c, x1c:x2c] = panel[:ph, :pw]

    # ══════════════════════════════════════════════════════════════════
    # INFO PANEL
    # ══════════════════════════════════════════════════════════════════

    def _render_info_panel(
        self, c: np.ndarray, pkt: FramePacket,
        prof: dict, gpu: Optional[dict], sched: dict,
        backend: str, vlm_model: str, vlm_status: str,
        active_tracks: int, has_depth: bool, depth_idx: Optional[int],
        gpu_backend: str = "",
        vlm_tok_per_sec: float = 0.0, physical_tok_per_sec: float = 0.0,
    ) -> None:
        # Panel background
        cv2.rectangle(c, (INFO_X, HEADER_H), (CANVAS_W, BOTTOM_Y), BG_PANEL, -1)
        # Left border accent
        cv2.line(c, (INFO_X, HEADER_H), (INFO_X, BOTTOM_Y), SEPARATOR, 1)

        x0 = INFO_X + 16
        y = HEADER_H + 8
        lh = 22  # line height

        def put(text, color=TEXT_PRIMARY, scale=0.45, bold=False):
            nonlocal y
            y += lh
            cv2.putText(c, text, (x0, y), FONT, scale, color, 2 if bold else 1)

        def section(title):
            nonlocal y
            y += 8
            cv2.line(c, (x0, y), (CANVAS_W - 16, y), SEPARATOR, 1)
            y += 4
            put(title, ACCENT_CYAN, 0.5, bold=True)

        # ── PERFORMANCE ──
        section("PERFORMANCE")
        fps_color = ACCENT_GREEN if self._fps > 25 else (ACCENT_ORANGE if self._fps > 10 else ACCENT_RED)
        cv2.putText(c, f"{self._fps:.0f}", (x0, y + 38), FONT, 1.2, fps_color, 2)
        cv2.putText(c, "FPS", (x0 + 70, y + 38), FONT, 0.45, TEXT_SECONDARY, 1)
        # End-to-end latency, discreetly rendered to the right of the big FPS
        # figure — gives a single signal about pipeline health without the
        # noisy per-stage breakdown.
        e2e = prof.get("end_to_end", {})
        if e2e.get("mean_ms", 0) > 0:
            e2e_text = f"E2E {e2e['mean_ms']:.0f}ms  p95 {e2e.get('p95_ms', 0):.0f}ms"
            cv2.putText(c, e2e_text, (x0 + 110, y + 38), FONT, 0.40,
                        TEXT_SECONDARY, 1)
        y += 42

        # ── INFERENCE (tok/s per model) ──
        # Replaces the per-stage latency dump — operators care about throughput
        # of each reasoning head, not how many ms each bookkeeping stage took.
        section("INFERENCE")
        stages = prof.get("stages", {})
        _det_ms = stages.get("detection", {}).get("mean_ms", 0) or 0
        _det_fps = (1000.0 / _det_ms) if _det_ms > 0 else 0.0
        if _det_fps > 0:
            put(f"Detection   {_det_fps:>5.1f} fps   ({_det_ms:.1f} ms/inf)",
                ACCENT_GREEN, 0.42)
        else:
            put("Detection   --  fps", TEXT_MUTED, 0.42)
        if vlm_tok_per_sec > 0:
            put(f"Scene VLM   {vlm_tok_per_sec:>5.1f} tok/s", ACCENT_GREEN, 0.42)
        else:
            _scene_color = ACCENT_ORANGE if vlm_status == "loading" else TEXT_MUTED
            _scene_text = "loading" if vlm_status == "loading" else "idle"
            put(f"Scene VLM   --    tok/s ({_scene_text})", _scene_color, 0.42)
        if physical_tok_per_sec > 0:
            put(f"Physics     {physical_tok_per_sec:>5.1f} tok/s",
                ACCENT_GREEN, 0.42)
        else:
            put("Physics     --    tok/s (idle)", TEXT_MUTED, 0.42)

        # ── GPU ──
        _backend_short = gpu_backend.replace("Monitor", "").replace("Torch", "torch") if gpu_backend else ""
        _section_label = f"GPU  [{_backend_short}]" if _backend_short else "GPU"
        section(_section_label)
        if gpu:
            util = gpu.get("utilization", 0)
            mem = gpu.get("memory", 0)
            temp = gpu.get("temperature", 0)
            # Util bar
            bar_w = 200
            bar_x = x0
            y += lh
            cv2.rectangle(c, (bar_x, y - 8), (bar_x + bar_w, y + 4), (50, 50, 50), -1)
            fill_w = int(bar_w * min(util, 1.0))
            bar_color = ACCENT_GREEN if util < 0.7 else (ACCENT_ORANGE if util < 0.9 else ACCENT_RED)
            cv2.rectangle(c, (bar_x, y - 8), (bar_x + fill_w, y + 4), bar_color, -1)
            cv2.putText(c, f"GPU {util*100:.0f}%", (bar_x + bar_w + 10, y + 2), FONT, 0.4, bar_color, 1)
            # Memory bar
            y += lh
            cv2.rectangle(c, (bar_x, y - 8), (bar_x + bar_w, y + 4), (50, 50, 50), -1)
            mem_w = int(bar_w * min(mem, 1.0))
            cv2.rectangle(c, (bar_x, y - 8), (bar_x + mem_w, y + 4), ACCENT_CYAN, -1)
            _mem_used_gb = gpu.get("memory_used_gb", 0)
            _mem_total_gb = gpu.get("memory_total_gb", 0)
            if _mem_total_gb > 0:
                cv2.putText(c, f"MEM {_mem_used_gb:.1f}GB/{_mem_total_gb:.0f}GB", (bar_x + bar_w + 10, y + 2), FONT, 0.4, TEXT_SECONDARY, 1)
            else:
                cv2.putText(c, f"MEM {mem*100:.0f}%", (bar_x + bar_w + 10, y + 2), FONT, 0.4, TEXT_SECONDARY, 1)
            put(f"Temp {temp:.0f}C  Stages {sched.get('gpu_active', 0)}", TEXT_SECONDARY, 0.38)
            # Per-process stats (set by the current-process snapshot in
            # orchestrator; separate from the top-processes panel below).
            _pgpu = gpu.get("process_gpu_mem_mb", 0)
            _pram = gpu.get("process_ram_mb", 0)
            _pcpu = gpu.get("process_cpu_pct", 0)
            if _pgpu > 0 or _pram > 0:
                _ram_str = f"{_pram/1024:.1f}GB" if _pram >= 1024 else f"{_pram:.0f}MB"
                put(f"App GPU:{_pgpu:.0f}MB RAM:{_ram_str} CPU:{_pcpu:.0f}%", ACCENT_CYAN, 0.36)

            # Top GPU processes mini-panel (from NvitopProcessMonitor,
            # scoped to the orchestrator's own process tree). Hidden when
            # nvitop is unavailable or the list is empty — zero layout shift.
            _top = gpu.get("top_processes") or []
            if _top:
                put("Top GPU:", TEXT_SECONDARY, 0.36)
                for _p in _top[:3]:
                    # show role/name; PID is omitted (changes every restart).
                    _nm = (_p.get("role") or _p.get("name") or "?")[:16]
                    _gm = _p.get("gpu_mem_mb")
                    _cpu = _p.get("cpu_pct", 0)
                    _mem = _p.get("mem_mb", 0)
                    if _gm is not None and _gm > 0:
                        _mem_str = f"GPU:{_gm:.0f}MB"
                    elif _mem >= 1024:
                        _mem_str = f"MEM:{_mem/1024:.1f}GB"
                    else:
                        _mem_str = f"MEM:{_mem:.0f}MB"
                    put(f"  {_nm:<16} {_mem_str}  {_cpu:>3.0f}% CPU",
                        ACCENT_CYAN, 0.34)
            if util == 0.0:
                if _backend_short == "NvitopDevice":
                    # On Jetson Thor UMA, NVML does not surface GA10B compute
                    # utilisation — this reading is expected, not an error.
                    put("nvitop: NVML reports 0 util (Jetson UMA limit)", TEXT_MUTED, 0.35)
                elif _backend_short == "Tegrastats":
                    put("tegrastats active - waiting for GR3D_FREQ", TEXT_MUTED, 0.35)
                elif _backend_short == "TorchCuda":
                    put("torch.cuda only - no compute util", TEXT_MUTED, 0.35)
                else:
                    put("waiting for GPU data...", TEXT_MUTED, 0.35)
        else:
            put("Monitor: off", TEXT_MUTED)

        # ── SAFETY ZONE ──
        section("SAFETY ZONE")
        _sz = self._safety_presets[self._safety_preset_idx]
        _sz_color = ACCENT_GREEN if _sz >= 2.0 else (ACCENT_ORANGE if _sz >= 1.0 else ACCENT_RED)
        put(f"Alert radius: {_sz:.1f}m", _sz_color, 0.42)
        put("[ decrease   ] increase", TEXT_MUTED, 0.35)
        # Last alert timestamp
        if self._last_alert_time > 0:
            _elapsed = time.monotonic() - self._last_alert_time
            _ts_str = f"Last: {int(_elapsed)}s ago"
            if self._last_alert_class:
                _ts_str += f" ({self._last_alert_class})"
            _ts_color = ACCENT_RED if _elapsed < 10 else (ACCENT_ORANGE if _elapsed < 60 else TEXT_MUTED)
            put(_ts_str, _ts_color, 0.38)

        # ── DETECTIONS ──
        det_count = len(pkt.detections)
        section(f"DETECTIONS ({det_count})")
        classes: dict[str, int] = {}
        for d in pkt.detections:
            classes[d.class_name] = classes.get(d.class_name, 0) + 1
        for cls, cnt in list(classes.items())[:8]:
            y += lh
            color = DET_COLORS[hash(cls) % len(DET_COLORS)]
            cv2.circle(c, (x0 + 6, y - 4), 5, color, -1)
            cv2.putText(c, f"{cls}: {cnt}", (x0 + 18, y), FONT, 0.42, TEXT_PRIMARY, 1)

        # ── TRACKER ──
        section("TRACKER")
        put(f"Active: {active_tracks}  Total det: {det_count}", TEXT_PRIMARY, 0.42)
        # Persistent "Tracking: …" status line when a track is locked. Shown
        # only when packet.locked_track_id >= 0; hidden otherwise. Looks up
        # the class name from the current detection list if available so the
        # label says "person" rather than just "id=42".
        _locked_id = getattr(pkt, "locked_track_id", -1) if pkt is not None else -1
        if _locked_id is not None and _locked_id >= 0:
            _locked_name = ""
            if pkt is not None:
                for _d in pkt.detections or []:
                    if getattr(_d, "track_id", -1) == _locked_id:
                        _locked_name = _d.class_name
                        break
            _label = (
                f"Tracking: {_locked_name} (id={_locked_id})"
                if _locked_name else f"Tracking: id={_locked_id}"
            )
            put(_label, ACCENT_ORANGE, 0.40)

        # ── DEPTH ──
        section("DEPTH")
        if has_depth:
            put(f"/dev/video{depth_idx}", ACCENT_GREEN, 0.42)
        else:
            put("No depth stream", TEXT_MUTED, 0.42)

        # ── VLM ──
        # Model identity + status only; throughput lives in the INFERENCE
        # section at the top of the panel so it's visible at a glance.
        section("VLM")
        short_model = vlm_model.split("/")[-1][:24] if vlm_model else "disabled"
        put(short_model, TEXT_SECONDARY, 0.40)
        if vlm_status == "loading":
            put("Status: loading...", ACCENT_ORANGE, 0.38)
        else:
            _status_color = ACCENT_GREEN if vlm_status == "active" else TEXT_MUTED
            put(f"Status: {vlm_status}", _status_color, 0.38)

    # ══════════════════════════════════════════════════════════════════
    # BOTTOM BAR (VLM Response + Query Input)
    # ══════════════════════════════════════════════════════════════════

    def _render_bottom_bar(self, c: np.ndarray, scene: str, phys: str) -> None:
        # Background
        cv2.rectangle(c, (0, BOTTOM_Y), (CANVAS_W, CANVAS_H), BG_PANEL, -1)
        # Top border
        cv2.line(c, (0, BOTTOM_Y), (CANVAS_W, BOTTOM_Y), SEPARATOR, 1)
        # Divider between VLM and query
        cv2.line(c, (QUERY_X, BOTTOM_Y), (QUERY_X, CANVAS_H), SEPARATOR, 1)

        # ── VLM Response Area (left) — stacked 3-row layout ──
        # Each row: section label (cyan) + per-section line cap.
        # QUERY RESPONSE gets the largest budget because it's what the
        # operator just asked about; SCENE / PHYSICAL get smaller. The
        # `if vy + 18 > CANVAS_H - 4: break` guard at the inner loop
        # silently caps total height so we can't overflow the canvas
        # — sections later in the stack just get fewer lines if
        # earlier sections take more space.
        # Width was bumped 75 → 110 chars to fit more text per line
        # (the bottom-bar region is ~1180 px wide; ~110 chars at scale
        # 0.42 fits comfortably with margin).
        # Press [X] at runtime to open a modal with the FULL,
        # untruncated text of all three responses.
        vx = 20
        vy = BOTTOM_Y + 12

        # Track the latest scene_desc text → bump its freshness
        # timestamp whenever it changes. The orchestrator passes the
        # current scene description as a render kwarg every frame
        # (not via a setter), so we maintain this here instead of in
        # set_response.
        if scene and scene != self._last_scene_desc_text:
            self._last_scene_desc_text = scene
            self._last_scene_desc_ts = time.monotonic()

        # ── Staleness window ──
        # A response that hasn't been touched in this many seconds is
        # considered "done and no longer updating" — we drop it from
        # the bottom panel so its vertical budget can be claimed by
        # the actively-streaming section. The operator can still pull
        # up the full untruncated history via [X] (modal ignores
        # staleness). 60 s strikes a reasonable balance: long enough
        # to keep a just-finished answer on screen while the operator
        # reads it, short enough that it gets out of the way before
        # the next auto-caption update.
        STALENESS_S = 60.0
        now_mono = time.monotonic()
        query_fresh   = (now_mono - self._last_response_ts) <= STALENESS_S \
                        if self._last_response_ts > 0 else False
        scene_fresh   = (now_mono - self._last_scene_desc_ts) <= STALENESS_S \
                        if self._last_scene_desc_ts > 0 else False
        physical_fresh = (now_mono - self._last_physical_response_ts) <= STALENESS_S \
                        if self._last_physical_response_ts > 0 else False

        phys_to_show = self._last_physical_response or phys
        rows: list[tuple[str, str, tuple[int, int, int]]] = []

        # QUERY RESPONSE — only when fresh (operator's recent question)
        if self._last_response and query_fresh:
            rows.append(("QUERY RESPONSE", self._last_response, ACCENT_GREEN))

        # SCENE UNDERSTANDING — fresh keeps it; cold-start placeholder
        # only when nothing else is on screen.
        if scene and scene_fresh:
            rows.append(("SCENE UNDERSTANDING", scene, TEXT_PRIMARY))
        elif not rows and not phys_to_show:
            rows.append(("SCENE UNDERSTANDING", "Waiting for detections...", TEXT_MUTED))

        # PHYSICAL REASONING — only when fresh
        if phys_to_show and physical_fresh:
            rows.append(("PHYSICAL REASONING", phys_to_show, ACCENT_GREEN))

        # ── Dynamic per-section caps ──
        # When the only fresh row is QUERY RESPONSE, give it a much
        # larger cap so the operator's answer fills the panel. As
        # other sections come back, each row gets its standard cap.
        # This is what "free space goes to the active section" means
        # in practice: a single-row layout shows ~16 lines instead
        # of 8.
        if len(rows) == 1:
            SECTION_CAPS = {
                "QUERY RESPONSE":      16,
                "SCENE UNDERSTANDING": 14,
                "PHYSICAL REASONING":  14,
            }
        elif len(rows) == 2:
            SECTION_CAPS = {
                "QUERY RESPONSE":      10,
                "SCENE UNDERSTANDING": 6,
                "PHYSICAL REASONING":  6,
            }
        else:
            SECTION_CAPS = {
                "QUERY RESPONSE":      8,
                "SCENE UNDERSTANDING": 3,
                "PHYSICAL REASONING":  4,
            }
        DEFAULT_CAP = 2
        WRAP_WIDTH = 110

        for label, body, body_color in rows:
            if vy + 20 > CANVAS_H - 4:
                break
            cv2.putText(c, label, (vx, vy + 14), FONT, 0.42, ACCENT_CYAN, 1)
            vy += 22
            cap = SECTION_CAPS.get(label, DEFAULT_CAP)
            wrapped = _wrap_text(body, WRAP_WIDTH)
            shown = wrapped[:cap]
            for line in shown:
                if vy + 18 > CANVAS_H - 4:
                    break
                cv2.putText(c, line, (vx, vy + 14), FONT, 0.42, body_color, 1)
                vy += 20
            # If we truncated, surface a small "+N more — press [X]"
            # hint so the operator knows there's more text available.
            if len(wrapped) > len(shown) and vy + 18 <= CANVAS_H - 4:
                hint = f"... +{len(wrapped) - len(shown)} more lines — press [X] for full text"
                cv2.putText(c, hint, (vx, vy + 14), FONT, 0.36, TEXT_MUTED, 1)
                vy += 20
            vy += 8

        # ── Query Input Area (right) ──
        qx = QUERY_X + 20
        qy = BOTTOM_Y + 12
        cv2.putText(c, "QUERY", (qx, qy + 16), FONT, 0.5, ACCENT_CYAN, 1)

        if self._query_mode:
            # Active input field
            field_y = qy + 40
            field_h = 44
            cv2.rectangle(c, (qx, field_y), (CANVAS_W - 20, field_y + field_h), BG_INPUT, -1)
            cv2.rectangle(c, (qx, field_y), (CANVAS_W - 20, field_y + field_h), QUERY_BORDER, 2)
            display_text = self._query_text + "|"
            cv2.putText(c, display_text, (qx + 12, field_y + 30), FONT, 0.55, TEXT_PRIMARY, 1)

            # Hints
            cv2.putText(c, "[Enter] Submit    [Esc] Cancel", (qx, field_y + field_h + 30),
                        FONT, 0.42, TEXT_SECONDARY, 1)
        else:
            # Idle — show shortcuts
            shortcuts = [
                ("[T]  Ask a question", TEXT_PRIMARY),
                ("[X]  Show full response (VLM + Physical)", TEXT_PRIMARY),
                ("[M]  Toggle masks", TEXT_SECONDARY),
                ("[G]  Toggle distance rings", TEXT_SECONDARY),
                ("[B]  Toggle bird's eye map", TEXT_SECONDARY),
                ("[ ]  Decrease / increase alert dist", TEXT_SECONDARY),
                ("[F]  Toggle fullscreen", TEXT_SECONDARY),
                ("[R]  Reset VLM cache", TEXT_SECONDARY),
                ("[Q]  Quit", TEXT_SECONDARY),
            ]
            sy = qy + 50
            for text, color in shortcuts:
                cv2.putText(c, text, (qx, sy), FONT, 0.5, color, 1)
                sy += 32

            # Example queries
            sy += 16
            cv2.putText(c, "Example queries:", (qx, sy), FONT, 0.4, TEXT_MUTED, 1)
            sy += 22
            for example in [
                '"How many people are there?"',
                '"What is the distance of the bottle?"',
                '"Describe the scene"',
            ]:
                cv2.putText(c, example, (qx + 8, sy), FONT, 0.38, TEXT_MUTED, 1)
                sy += 20

    # ══════════════════════════════════════════════════════════════════
    # FULL-RESPONSE MODAL (toggled by [X])
    # ══════════════════════════════════════════════════════════════════

    def _render_response_modal(self, c: np.ndarray,
                               scene_desc: str, physical_desc: str) -> None:
        """Overlay a centred modal on the canvas showing the FULL,
        untruncated text of QUERY RESPONSE / SCENE UNDERSTANDING /
        PHYSICAL REASONING. Stacked top-down with cyan section labels
        and accent-coloured body text.

        Geometry: occupies the camera-hero region (1440×684) with a
        16 px inset; backdrop is dimmed to 35 % so the underlying
        scene is still faintly visible. Text is word-wrapped at
        ~140 chars (the modal width gives more room than the bottom
        panel). Vertical scroll via j/k or arrow keys.
        """
        # Modal rectangle inside the camera-hero region.
        m_x0 = 16
        m_y0 = HEADER_H + 16
        m_x1 = RGB_W - 16
        m_y1 = HEADER_H + RGB_H - 16
        m_w = m_x1 - m_x0
        m_h = m_y1 - m_y0

        # Dim the underlying region (camera frame) by alpha-blending
        # with black so the modal pops visually but the scene is still
        # perceptible underneath.
        roi = c[m_y0:m_y1, m_x0:m_x1]
        dimmed = (roi.astype(np.int32) * 35 // 100).astype(np.uint8)
        c[m_y0:m_y1, m_x0:m_x1] = dimmed

        # Modal frame + thin amber border so it reads as a foreground UI.
        cv2.rectangle(c, (m_x0, m_y0), (m_x1, m_y1), BG_PANEL, -1)
        cv2.rectangle(c, (m_x0, m_y0), (m_x1, m_y1), QUERY_BORDER, 2)

        # Title bar
        title = "FULL RESPONSE   [X] / [Esc] close   [j/k] scroll"
        cv2.putText(c, title, (m_x0 + 16, m_y0 + 28),
                    FONT, 0.55, ACCENT_CYAN, 1)
        cv2.line(c, (m_x0 + 8, m_y0 + 40), (m_x1 - 8, m_y0 + 40),
                 SEPARATOR, 1)

        # Build the line list. We word-wrap each section to 140 chars
        # (modal is ~1408 px wide; ~140 chars at scale 0.45 fits with
        # margin), then concatenate with a blank-line separator.
        WRAP = 140
        sections: list[tuple[str, str, tuple[int, int, int]]] = []
        if self._last_response:
            sections.append(("QUERY RESPONSE", self._last_response, ACCENT_GREEN))
        scene_text = scene_desc or ""
        if scene_text:
            sections.append(("SCENE UNDERSTANDING", scene_text, TEXT_PRIMARY))
        phys_text = self._last_physical_response or physical_desc or ""
        if phys_text:
            sections.append(("PHYSICAL REASONING", phys_text, ACCENT_GREEN))
        if not sections:
            cv2.putText(c, "(no responses yet — issue a query with [T])",
                        (m_x0 + 16, m_y0 + 80),
                        FONT, 0.5, TEXT_MUTED, 1)
            return

        # Flatten to a list of (text, color) tuples for easy scrolling.
        # Each section emits: heading row + body lines + blank-line gap.
        all_lines: list[tuple[str, tuple[int, int, int], float]] = []
        for label, body, body_col in sections:
            all_lines.append((label, ACCENT_CYAN, 0.5))   # section heading
            for ln in _wrap_text(body, WRAP):
                all_lines.append((ln, body_col, 0.45))
            all_lines.append(("", TEXT_MUTED, 0.45))      # spacer

        # Page geometry — start drawing under the title bar.
        line_h = 22
        text_top = m_y0 + 60
        text_bot = m_y1 - 36
        max_visible = max(1, (text_bot - text_top) // line_h)

        # Clamp scroll so we don't blow past the end of the document.
        max_scroll = max(0, len(all_lines) - max_visible)
        if self._response_modal_scroll > max_scroll:
            self._response_modal_scroll = max_scroll
        scroll = self._response_modal_scroll

        visible = all_lines[scroll : scroll + max_visible]
        y = text_top + 14
        for text, color, scale in visible:
            cv2.putText(c, text, (m_x0 + 16, y), FONT, scale, color, 1)
            y += line_h

        # Footer status: scroll position + total line count.
        total = len(all_lines)
        end = min(scroll + max_visible, total)
        footer = f"Lines {scroll + 1}-{end} of {total}"
        if max_scroll > 0:
            footer += "    (more — press [j] / Down to scroll)"
        cv2.putText(c, footer, (m_x0 + 16, m_y1 - 14),
                    FONT, 0.40, TEXT_SECONDARY, 1)


# ── Utilities ──

def _as_numpy(buf) -> np.ndarray:
    if isinstance(buf, np.ndarray):
        return buf
    try:
        return buf.numpy()
    except Exception:
        return np.asarray(buf)


def _wrap_text(text: str, max_chars: int) -> list[str]:
    """Simple word-wrap."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > max_chars:
            if current:
                lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return lines
