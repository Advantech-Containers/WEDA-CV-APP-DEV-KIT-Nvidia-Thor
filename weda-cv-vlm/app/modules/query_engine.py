"""4-Tier Query Engine for industrial CV pipeline.

Tier 1   (< 1ms):    Structured — count, distance, position, nearest, size
                      Direct lookup from detection + spatial data.

Tier 2   (< 5ms):    Visual — color, describe object, visual attributes
                      Reads from VLM object cache (background pre-populated).

Tier 2.5 (50-100ms): Physical reasoning — safety, path, PPE, physics
                      Cosmos-Reason2-2B with grounded context.

Tier 3   (1-2.5s):   Novel — anything not matching tiers above
                      Qwen3-VL-30B-A3B live VLM (grounded, rare).

All methods are @staticmethod — zero instance state, zero allocation
when no query is pending.
"""

from __future__ import annotations

import json as _json
import re
from enum import Enum, auto
from typing import Optional

from app.utils.data_structures import Detection, SpatialInfo


# ──────────────────────────────────────────────────────────────────────
# Tracking-intent parsing (pipeline-level, independent of QueryType tiers)
# ──────────────────────────────────────────────────────────────────────
#
# The user can issue natural-language tracking commands alongside ordinary
# queries ("track the person", "stop tracking"). These are parsed OUT of
# the query stream before tier classification and dispatched to the
# tracker's set_locked() / clear_locked() API — they do not reach the VLM.

_TRACK_INTENT = re.compile(
    r"\b(?:track|follow|keep\s+watching|watch|lock\s+on(?:to)?)\s+"
    r"(?:the\s+|that\s+|this\s+)?(?P<subject>[a-zA-Z][a-zA-Z_]*)",
    re.I,
)
_STOP_TRACK_INTENT = re.compile(
    r"\b(?:stop|untrack|release|unlock|cancel)\b.*?\b(?:track(?:ing)?|follow(?:ing)?|lock)\b",
    re.I,
)


def classify_tracking_intent(query: str) -> Optional[tuple[str, Optional[str]]]:
    """Return ``('track', class_name)`` | ``('untrack', None)`` | ``None``.

    Call this FIRST on every user query. If it returns non-None, dispatch
    to ``tracker.set_locked()`` / ``tracker.clear_locked()`` and skip the
    normal query-engine tier routing (no VLM involvement needed).

    Examples:
        "track the person"      -> ('track', 'person')
        "follow the forklift"   -> ('track', 'forklift')
        "lock onto that box"    -> ('track', 'box')
        "stop tracking"         -> ('untrack', None)
        "release lock"          -> ('untrack', None)
        "how many people?"      -> None  (falls through to tier routing)
    """
    if not query:
        return None
    if _STOP_TRACK_INTENT.search(query):
        return ("untrack", None)
    m = _TRACK_INTENT.search(query)
    if m:
        subject = m.group("subject").lower().strip()
        # Filter out common non-class words that follow "track"/"follow"
        if subject in {"it", "them", "up", "down", "back", "away", "along"}:
            return None
        return ("track", subject)
    return None


# ──────────────────────────────────────────────────────────────────────
# Query classification
# ──────────────────────────────────────────────────────────────────────

class QueryType(Enum):
    # Tier 1 — structured, instant
    COUNT = auto()
    DISTANCE = auto()
    POSITION = auto()
    NEAREST = auto()
    LIST_OBJECTS = auto()
    SIZE = auto()
    # Tier 2 — visual, from cache
    COLOR = auto()
    DESCRIBE_OBJECT = auto()
    # Tier 2.5 — physical reasoning (Cosmos-Reason2)
    SAFETY = auto()
    PATH_CLEAR = auto()
    PHYSICS = auto()
    PPE_CHECK = auto()
    # Tier 1 extended — object-to-object spatial
    RELATIVE_DISTANCE = auto()   # "how far is X from Y"
    # Tier 1 extended — alert log history
    ALERT_HISTORY = auto()       # "what alerts happened", "show alert log"
    # Tier 3 — novel, live VLM
    OPEN_ENDED = auto()


TIER1_TYPES = {
    QueryType.COUNT, QueryType.DISTANCE, QueryType.POSITION,
    QueryType.NEAREST, QueryType.LIST_OBJECTS, QueryType.SIZE,
    QueryType.RELATIVE_DISTANCE, QueryType.ALERT_HISTORY,
}
TIER2_TYPES = {QueryType.COLOR, QueryType.DESCRIBE_OBJECT}
TIER25_TYPES = {QueryType.SAFETY, QueryType.PATH_CLEAR, QueryType.PHYSICS, QueryType.PPE_CHECK}

# Compiled regex patterns — evaluated once at module load
_PATTERNS: list[tuple[re.Pattern, QueryType]] = [
    # Tier 1
    (re.compile(r"how\s+many\s+(\w+)", re.I), QueryType.COUNT),
    (re.compile(r"count\s+(?:the\s+)?(\w+)", re.I), QueryType.COUNT),
    (re.compile(r"number\s+of\s+(\w+)", re.I), QueryType.COUNT),
    (re.compile(r"distance\s+(?:of|to|from)\s+(?:the\s+)?(\w+)", re.I), QueryType.DISTANCE),
    (re.compile(r"how\s+far\s+is\s+(?:the\s+)?(\w+)\s+from\s+(?:the\s+)?(\w+)", re.I), QueryType.RELATIVE_DISTANCE),
    (re.compile(r"distance\s+between\s+(?:the\s+)?(\w+)\s+and\s+(?:the\s+)?(\w+)", re.I), QueryType.RELATIVE_DISTANCE),
    (re.compile(r"angle\s+between\s+(?:the\s+)?(\w+)\s+and\s+(?:the\s+)?(\w+)", re.I), QueryType.RELATIVE_DISTANCE),
    (re.compile(r"how\s+far\s+(?:is\s+)?(?:the\s+)?(\w+)", re.I), QueryType.DISTANCE),
    (re.compile(r"where\s+is\s+(?:the\s+)?(\w+)", re.I), QueryType.POSITION),
    (re.compile(r"position\s+of\s+(?:the\s+)?(\w+)", re.I), QueryType.POSITION),
    (re.compile(r"(?:what(?:'s)?|which)\s+(?:is\s+)?(?:the\s+)?closest\s+(?:to\s+)?(?:the\s+)?(\w+)", re.I), QueryType.NEAREST),
    (re.compile(r"nearest\s+(?:object\s+)?(?:to\s+)?(?:the\s+)?(\w+)", re.I), QueryType.NEAREST),
    (re.compile(r"what\s+(?:objects?|things?)\s+(?:are|do|can)", re.I), QueryType.LIST_OBJECTS),
    (re.compile(r"list\s+(?:all\s+)?(?:the\s+)?objects?", re.I), QueryType.LIST_OBJECTS),
    (re.compile(r"what\s+(?:do\s+you|can\s+you)\s+see", re.I), QueryType.LIST_OBJECTS),
    (re.compile(r"how\s+big\s+(?:is\s+)?(?:the\s+)?(\w+)", re.I), QueryType.SIZE),
    (re.compile(r"size\s+of\s+(?:the\s+)?(\w+)", re.I), QueryType.SIZE),
    # Tier 2
    (re.compile(r"what\s+colou?r\s+(?:is\s+)?(?:the\s+)?(\w+)", re.I), QueryType.COLOR),
    (re.compile(r"colou?r\s+of\s+(?:the\s+)?(\w+)", re.I), QueryType.COLOR),
    (re.compile(r"describe\s+(?:the\s+)?scene", re.I), QueryType.LIST_OBJECTS),
    (re.compile(r"what(?:'s)?\s+(?:in\s+)?(?:the\s+)?scene", re.I), QueryType.LIST_OBJECTS),
    (re.compile(r"what\s+(?:is|are)\s+(?:the\s+)?(\w+)\s+doing", re.I), QueryType.OPEN_ENDED),
    (re.compile(r"describe\s+(?:the\s+)?(\w+)", re.I), QueryType.DESCRIBE_OBJECT),
    (re.compile(r"tell\s+me\s+about\s+(?:the\s+)?(\w+)", re.I), QueryType.DESCRIBE_OBJECT),
    (re.compile(r"what\s+does\s+(?:the\s+)?(\w+)\s+look\s+like", re.I), QueryType.DESCRIBE_OBJECT),
    # Tier 2.5 — physical reasoning (Cosmos-Reason2). Keep patterns liberal so
    # natural safety/physics phrasings route here (no cooldown on
    # physical_reasoner.request_query) instead of falling through to the
    # cooldown-gated Tier 3 path.
    (re.compile(r"is\s+(?:it|this|that|the\s+\w+(?:\s+\w+)?|he|she|they|anyone|anything|everyone|everything)\s+safe", re.I), QueryType.SAFETY),
    (re.compile(r"safe\s+to\s+(?:move|proceed|go|turn|walk|stand|operate|continue|enter|pass)", re.I), QueryType.SAFETY),
    (re.compile(r"(?:any|is\s+there\s+(?:a|any))\s+(?:danger|hazard|risk|threat)", re.I), QueryType.SAFETY),
    (re.compile(r"(?:dangerous|hazardous|unsafe|risky)", re.I), QueryType.SAFETY),
    (re.compile(r"(?:is|are)\s+(?:the\s+)?(?:path|way|route|floor|area|zone|space)\s+(?:clear|safe|blocked|obstructed)", re.I), QueryType.PATH_CLEAR),
    (re.compile(r"how\s+(?:do\s+I|to|can\s+I)\s+(?:reach|get\s+to|navigate\s+to|go\s+to)\s+(?:the\s+)?(\w+)", re.I), QueryType.PATH_CLEAR),
    (re.compile(r"(?:path|route|way)\s+to\s+(?:the\s+)?(\w+)", re.I), QueryType.PATH_CLEAR),
    (re.compile(r"navigate\s+to\s+(?:the\s+)?(\w+)", re.I), QueryType.PATH_CLEAR),
    (re.compile(r"directions?\s+to\s+(?:the\s+)?(\w+)", re.I), QueryType.PATH_CLEAR),
    (re.compile(r"(?:will|can|could|would|might|may)\s+(?:it|that|this|the\s+\w+)\s+(?:fall|tip|collapse|slide|topple|break|drop|slip)", re.I), QueryType.PHYSICS),
    (re.compile(r"(?:is|are)\s+(?:the\s+)?(?:object|box|load|item)s?\s+(?:stable|balanced|secure|supported)", re.I), QueryType.PHYSICS),
    (re.compile(r"(?:is|are)\s+(?:the\s+)?(?:worker|person|operator|he|she|they)\s+wearing\s+(?:ppe|helmet|vest|gloves|mask|goggles|hardhat)", re.I), QueryType.PPE_CHECK),
    (re.compile(r"ppe\s+(?:check|status|compliance|violation)", re.I), QueryType.PPE_CHECK),
    (re.compile(r"(?:missing|without|no)\s+(?:ppe|helmet|vest|gloves|mask|goggles|hardhat)", re.I), QueryType.PPE_CHECK),
    # Tier 1 — alert history
    (re.compile(r"(?:what|show|list|get)\s+(?:the\s+)?(?:alerts?|safety\s+events?|warnings?)", re.I), QueryType.ALERT_HISTORY),
    (re.compile(r"(?:alert|warning|safety)\s+(?:log|history|records?)", re.I), QueryType.ALERT_HISTORY),
    (re.compile(r"when\s+(?:did|was|were)\s+(?:the\s+)?(?:alert|alarm|safety)", re.I), QueryType.ALERT_HISTORY),
    (re.compile(r"recent\s+(?:alerts?|safety\s+events?|incidents?)", re.I), QueryType.ALERT_HISTORY),
    # Tier 1 — fallback class/spatial phrasings that earlier blocks missed.
    # Evaluated last so a tier-2.5 safety/path/physics pattern wins first.
    (re.compile(r"what(?:'s|\s+is|\s+are)\s+(?:in|around|here|there|visible|present|on|near|behind|ahead)", re.I), QueryType.LIST_OBJECTS),
    (re.compile(r"what(?:'s|\s+is)\s+(?:happening|going\s+on)", re.I), QueryType.LIST_OBJECTS),
    (re.compile(r"what\s+(?:do|can)\s+(?:i|you|we)\s+see", re.I), QueryType.LIST_OBJECTS),
    (re.compile(r"is\s+there\s+(?:a|any|an)\s+(\w+)", re.I), QueryType.COUNT),
    (re.compile(r"are\s+there\s+(?:any\s+)?(\w+)", re.I), QueryType.COUNT),
    (re.compile(r"do\s+(?:you|i|we)\s+see\s+(?:any\s+|a\s+|an\s+)?(\w+)", re.I), QueryType.COUNT),
    (re.compile(r"any\s+(\w+)\s+(?:in\s+the\s+)?(?:scene|frame|view|around|here|there)", re.I), QueryType.COUNT),
]


# ──────────────────────────────────────────────────────────────────────
# Query Engine
# ──────────────────────────────────────────────────────────────────────

class QueryEngine:
    """Stateless 4-tier query router. All methods are @staticmethod."""

    @staticmethod
    def classify(query: str) -> tuple[QueryType, Optional[str], Optional[str]]:
        """Pattern-match query → (QueryType, target1_or_None, target2_or_None)."""
        for pattern, qtype in _PATTERNS:
            m = pattern.search(query)
            if m:
                t1 = m.group(1) if m.lastindex and m.lastindex >= 1 else None
                t2 = m.group(2) if m.lastindex and m.lastindex >= 2 else None
                return qtype, t1, t2
        return QueryType.OPEN_ENDED, None, None

    @staticmethod
    def process(
        query: str,
        detections: list[Detection],
        spatial: list[SpatialInfo],
        object_cache: Optional[dict[int, str]] = None,
        alert_logger=None,   # Optional[AlertLogger] — avoid circular import
        embedder=None,       # Optional[Embedder] — for alert search ranking
    ) -> tuple[str, QueryType]:
        """Route query through tiers. Returns (response_string, query_type)."""
        cache = object_cache or {}
        qtype, target, target2 = QueryEngine.classify(query)

        if qtype == QueryType.COUNT:
            return QueryEngine._handle_count(target, detections, spatial), qtype
        elif qtype == QueryType.DISTANCE:
            return QueryEngine._handle_distance(target, detections, spatial), qtype
        elif qtype == QueryType.RELATIVE_DISTANCE:
            return QueryEngine._handle_relative_distance(target, target2, detections, spatial), qtype
        elif qtype == QueryType.POSITION:
            return QueryEngine._handle_position(target, detections, spatial), qtype
        elif qtype == QueryType.NEAREST:
            return QueryEngine._handle_nearest(target, detections, spatial), qtype
        elif qtype == QueryType.LIST_OBJECTS:
            return QueryEngine._handle_list(detections, spatial), qtype
        elif qtype == QueryType.SIZE:
            return QueryEngine._handle_size(target, detections, spatial), qtype
        elif qtype == QueryType.ALERT_HISTORY:
            return QueryEngine._handle_alert_history(query, alert_logger, embedder), qtype
        elif qtype in TIER2_TYPES:
            return QueryEngine._handle_visual(target, detections, spatial, cache), qtype
        else:
            # Tier 3 — build grounded context for live VLM
            ctx = QueryEngine.build_grounded_context(query, detections, spatial, cache)
            return ctx, qtype

    # ──────────────────────────────────────────────────────────────────
    # Tier 1 handlers — pure functions, < 1ms
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _handle_count(
        target: Optional[str],
        detections: list[Detection],
        spatial: list[SpatialInfo],
    ) -> str:
        if not detections:
            return "No objects detected in the current scene."
        if target:
            matches = _find_by_class(target, detections)
            if not matches:
                return f"No {target} detected in the current scene."
            count = len(matches)
            parts = []
            for idx, det in matches:
                dist = f" at {spatial[idx].distance_m:.2f}m" if idx < len(spatial) and spatial[idx].distance_m > 0 else ""
                parts.append(f"{det.class_name} (track #{det.track_id}){dist}")
            listing = ", ".join(parts)
            plural = "s" if count > 1 else ""
            return f"There {'are' if count > 1 else 'is'} {count} {_singular(target)}{plural} in the scene: {listing}."
        else:
            count = len(detections)
            return f"There are {count} objects detected in the current scene."

    @staticmethod
    def _handle_distance(
        target: Optional[str],
        detections: list[Detection],
        spatial: list[SpatialInfo],
    ) -> str:
        if not detections:
            return "No objects detected in the current scene."
        matches = _find_by_class(target, detections) if target else list(enumerate(detections))
        if not matches:
            return f"No {target} detected in the current scene."
        parts = []
        for idx, det in matches:
            if idx < len(spatial) and spatial[idx].distance_m > 0:
                si = spatial[idx]
                side = ""
                if si.position_3d is not None:
                    x = si.position_3d[0]
                    side = ", to your left" if x < -0.2 else ", to your right" if x > 0.2 else ", straight ahead"
                approach = " (approaching)" if getattr(si, "approaching", False) else ""
                speed = f" [{si.speed_ms:.1f} m/s]" if getattr(si, "speed_ms", 0) > 0.1 else ""
                parts.append(f"{det.class_name} (track #{det.track_id}) is {si.distance_m:.2f}m from the camera{side}{approach}{speed}.")
            else:
                parts.append(f"{det.class_name} (track #{det.track_id}): distance unavailable (no depth data).")
        return " ".join(parts)

    @staticmethod
    def _handle_position(
        target: Optional[str],
        detections: list[Detection],
        spatial: list[SpatialInfo],
    ) -> str:
        if not detections:
            return "No objects detected in the current scene."
        matches = _find_by_class(target, detections) if target else [(0, detections[0])]
        if not matches:
            return f"No {target} detected in the current scene."
        parts = []
        for idx, det in matches:
            cx, cy = det.center
            pos_str = f"at pixel ({cx:.0f}, {cy:.0f})"
            if idx < len(spatial) and spatial[idx].position_3d is not None:
                p = spatial[idx].position_3d
                x, y, z = p[0], p[1], p[2]
                side = "left" if x < -0.2 else "right" if x > 0.2 else "center"
                zone = "close" if z < 2.0 else "mid-range" if z < 5.0 else "far"
                pos_str += f", 3D position X={x:.2f}m, Y={y:.2f}m, Z={z:.2f}m ({side} side, {zone})"
            if idx < len(spatial) and spatial[idx].distance_m > 0:
                pos_str += f", distance {spatial[idx].distance_m:.2f}m"
            parts.append(f"{det.class_name} (track #{det.track_id}) is {pos_str}.")
        return " ".join(parts)

    @staticmethod
    def _handle_nearest(
        target: Optional[str],
        detections: list[Detection],
        spatial: list[SpatialInfo],
    ) -> str:
        if not detections or not spatial:
            return "No objects detected to compute nearest neighbor."
        matches = _find_by_class(target, detections) if target else [(0, detections[0])]
        if not matches:
            return f"No {target} detected in the current scene."
        parts = []
        for idx, det in matches:
            if idx >= len(spatial):
                continue
            si = spatial[idx]
            if si.nearest_neighbor_id < 0 or si.nearest_neighbor_dist == float("inf"):
                parts.append(f"{det.class_name} (track #{det.track_id}) has no nearby objects.")
                continue
            # Resolve neighbor
            neighbor_name = "unknown"
            for d in detections:
                if d.track_id == si.nearest_neighbor_id:
                    neighbor_name = d.class_name
                    break
            parts.append(
                f"The nearest object to {det.class_name} (track #{det.track_id}) "
                f"is {neighbor_name} (track #{si.nearest_neighbor_id}) "
                f"at {si.nearest_neighbor_dist:.2f}m distance."
            )
        return " ".join(parts)

    @staticmethod
    def _handle_relative_distance(
        target_a: Optional[str],
        target_b: Optional[str],
        detections: list[Detection],
        spatial: list[SpatialInfo],
    ) -> str:
        import math
        if not detections:
            return "No objects detected in the current scene."
        if not target_a or not target_b:
            return "Please specify two objects (e.g. 'how far is the person from the box')."
        matches_a = _find_by_class(target_a, detections)
        matches_b = _find_by_class(target_b, detections)
        if not matches_a:
            return f"No {target_a} detected in the current scene."
        if not matches_b:
            return f"No {target_b} detected in the current scene."

        idx_a, det_a = matches_a[0]
        idx_b, det_b = matches_b[0]
        si_a = spatial[idx_a] if idx_a < len(spatial) else None
        si_b = spatial[idx_b] if idx_b < len(spatial) else None

        if (si_a and si_b and
                si_a.position_3d is not None and si_b.position_3d is not None):
            pa = si_a.position_3d
            pb = si_b.position_3d
            dist_3d = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(pa, pb)))
            horiz_sep = abs(float(pa[0]) - float(pb[0]))
            depth_sep = abs(float(pa[2]) - float(pb[2]))
            angle_deg = math.degrees(math.atan2(horiz_sep, max(depth_sep, 0.01)))
            side_a = "left" if float(pa[0]) < -0.2 else "right" if float(pa[0]) > 0.2 else "center"
            side_b = "left" if float(pb[0]) < -0.2 else "right" if float(pb[0]) > 0.2 else "center"
            depth_rel = "closer to camera" if float(pa[2]) < float(pb[2]) else "farther from camera"
            return (
                f"{det_a.class_name} #{det_a.track_id} is {dist_3d:.2f}m from "
                f"{det_b.class_name} #{det_b.track_id} "
                f"(angle {angle_deg:.0f} deg apart). "
                f"{det_a.class_name} is on the {side_a} side at {si_a.distance_m:.2f}m; "
                f"{det_b.class_name} is on the {side_b} side at {si_b.distance_m:.2f}m. "
                f"{det_a.class_name} is {depth_rel} than {det_b.class_name}."
            )
        else:
            # Fallback — pixel distance
            cx_a, cy_a = det_a.center
            cx_b, cy_b = det_b.center
            px_dist = math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
            d_a = f"{si_a.distance_m:.2f}m" if si_a and si_a.distance_m > 0 else "unknown depth"
            d_b = f"{si_b.distance_m:.2f}m" if si_b and si_b.distance_m > 0 else "unknown depth"
            return (
                f"3D distance unavailable. "
                f"{det_a.class_name} (track #{det_a.track_id}) is at {d_a}; "
                f"{det_b.class_name} (track #{det_b.track_id}) is at {d_b}. "
                f"Pixel separation: {px_dist:.0f}px."
            )

    @staticmethod
    def _handle_list(
        detections: list[Detection],
        spatial: list[SpatialInfo],
    ) -> str:
        if not detections:
            return "No objects detected in the current scene."
        groups: dict[str, list[tuple[int, Detection]]] = {}
        for i, det in enumerate(detections):
            groups.setdefault(det.class_name, []).append((i, det))
        parts = []
        for cls_name, items in groups.items():
            count = len(items)
            dists = []
            for idx, _ in items:
                if idx < len(spatial) and spatial[idx].distance_m > 0:
                    dists.append(spatial[idx].distance_m)
            plural = "s" if count > 1 else ""
            dist_str = ""
            if dists:
                dist_str = f" at distances [{', '.join(f'{d:.2f}m' for d in dists)}]"
            parts.append(f"{count} {cls_name}{plural}{dist_str}")
        return f"Objects in scene: {'; '.join(parts)}. Total: {len(detections)} objects."

    @staticmethod
    def _handle_size(
        target: Optional[str],
        detections: list[Detection],
        spatial: list[SpatialInfo],
    ) -> str:
        if not detections:
            return "No objects detected in the current scene."
        matches = _find_by_class(target, detections) if target else [(0, detections[0])]
        if not matches:
            return f"No {target} detected in the current scene."
        parts = []
        for idx, det in matches:
            if idx < len(spatial) and spatial[idx].size_estimate != (0.0, 0.0):
                w, h = spatial[idx].size_estimate
                parts.append(
                    f"{det.class_name} (track #{det.track_id}) is approximately "
                    f"{w:.2f}m wide and {h:.2f}m tall."
                )
            else:
                # Fallback: pixel dimensions from bbox
                bw = det.bbox[2] - det.bbox[0]
                bh = det.bbox[3] - det.bbox[1]
                parts.append(
                    f"{det.class_name} (track #{det.track_id}): "
                    f"{bw:.0f}×{bh:.0f} pixels (no depth for metric size)."
                )
        return " ".join(parts)

    # ──────────────────────────────────────────────────────────────────
    # Tier 1 alert history handler — reads JSONL log + optional embedding rank
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _handle_alert_history(
        query: str,
        alert_logger=None,
        embedder=None,
    ) -> str:
        """Return recent safety alerts, ranked by embedding similarity if available."""
        if alert_logger is None:
            return "Alert logging is not enabled in this session."
        return alert_logger.query_alerts(query, embedder=embedder, top_k=5, limit=100)

    # ──────────────────────────────────────────────────────────────────
    # Tier 2 handler — reads from VLM object cache
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _handle_visual(
        target: Optional[str],
        detections: list[Detection],
        spatial: list[SpatialInfo],
        object_cache: dict[int, str],
    ) -> str:
        if not detections:
            return "No objects detected in the current scene."
        matches = _find_by_class(target, detections) if target else [(0, detections[0])]
        if not matches:
            return f"No {target} detected in the current scene."
        parts = []
        for idx, det in matches:
            # Structured data
            dist_str = ""
            if idx < len(spatial) and spatial[idx].distance_m > 0:
                dist_str = f" at {spatial[idx].distance_m:.2f}m"
            # Visual cache
            visual = object_cache.get(det.track_id, "")
            if visual:
                parts.append(
                    f"{det.class_name} (track #{det.track_id}){dist_str}: {visual}"
                )
            else:
                parts.append(
                    f"{det.class_name} (track #{det.track_id}){dist_str}: "
                    f"visual details not yet cached (will be available shortly)."
                )
        return " ".join(parts)

    # ──────────────────────────────────────────────────────────────────
    # Tier 3 context builder — grounded context for live VLM
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def build_grounded_context(
        query: str,
        detections: list[Detection],
        spatial: list[SpatialInfo],
        object_cache: Optional[dict[int, str]] = None,
    ) -> str:
        cache = object_cache or {}
        lines = ["VERIFIED SCENE DATA:"]
        lines.append(f"Total objects detected: {len(detections)}")

        if not detections:
            lines.append("No objects in scene.")
            lines.append(f"\nUSER QUESTION: {query}")
            return "\n".join(lines)

        # Group by class
        groups: dict[str, list[tuple[int, Detection]]] = {}
        for i, det in enumerate(detections):
            groups.setdefault(det.class_name, []).append((i, det))

        lines.append("\nObject inventory:")
        for cls_name, items in groups.items():
            count = len(items)
            plural = "s" if count > 1 else ""
            lines.append(f"  {count} {cls_name}{plural}:")
            for idx, det in items:
                dist = f"{spatial[idx].distance_m:.2f}m" if idx < len(spatial) and spatial[idx].distance_m > 0 else "unknown"
                size_str = ""
                if idx < len(spatial) and spatial[idx].size_estimate != (0.0, 0.0):
                    w, h = spatial[idx].size_estimate
                    size_str = f", size {w:.2f}m x {h:.2f}m"
                visual = cache.get(det.track_id, "")
                vis_str = f", visual: {visual}" if visual else ""
                lines.append(
                    f"    - track #{det.track_id}: distance {dist}{size_str}{vis_str}"
                )

        # Nearest pairs
        pairs = []
        for i, det in enumerate(detections):
            if i < len(spatial) and spatial[i].nearest_neighbor_id >= 0:
                si = spatial[i]
                neighbor_name = "unknown"
                for d in detections:
                    if d.track_id == si.nearest_neighbor_id:
                        neighbor_name = d.class_name
                        break
                pairs.append(
                    f"  {det.class_name}#{det.track_id} <-> "
                    f"{neighbor_name}#{si.nearest_neighbor_id}: "
                    f"{si.nearest_neighbor_dist:.2f}m"
                )
        if pairs:
            lines.append("\nNearest pairs:")
            lines.extend(pairs)

        lines.append(f"\nUSER QUESTION: {query}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────
    # Tier 3 JSON context builder — full sensor-verified grounding
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def build_grounded_json(
        detections: list[Detection],
        spatial: list[SpatialInfo],
        frame_shape: tuple,
        object_cache: Optional[dict[int, str]] = None,
    ) -> str:
        """Return a JSON string with every available sensor metric for each
        detected object.  This replaces the plain-text context string when
        Qwen is being asked to reason about the scene.  Qwen is told to treat
        these numbers as ground-truth sensor readings (not visual estimates).

        Fields per object:
          id, class, confidence, bbox_normalized, frame_region,
          distance_m, position_3d_m, physical_size_m, nearest_object,
          segmentation_coverage_pct, cached_visual
        """
        h, w = frame_shape[:2]
        cache = object_cache or {}
        objects = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            region_h = "top" if cy < 0.33 else "bottom" if cy > 0.66 else "middle"
            region_w = "left" if cx < 0.33 else "right" if cx > 0.66 else "center"
            region = f"{region_h}-{region_w}" if region_h != "middle" else region_w

            obj: dict = {
                "id": det.track_id,
                "class": det.class_name,
                "confidence": round(float(det.confidence), 2),
                "bbox_normalized": {
                    "left":   round(float(x1) / w, 3),
                    "top":    round(float(y1) / h, 3),
                    "right":  round(float(x2) / w, 3),
                    "bottom": round(float(y2) / h, 3),
                },
                "frame_region": region,
            }

            if i < len(spatial):
                si = spatial[i]
                if si.distance_m > 0:
                    obj["distance_m"] = round(si.distance_m, 2)
                if si.position_3d is not None:
                    obj["position_3d_m"] = {
                        "x": round(float(si.position_3d[0]), 2),
                        "y": round(float(si.position_3d[1]), 2),
                        "z": round(float(si.position_3d[2]), 2),
                    }
                if si.size_estimate != (0.0, 0.0):
                    obj["physical_size_m"] = {
                        "width":  round(si.size_estimate[0], 2),
                        "height": round(si.size_estimate[1], 2),
                    }
                if si.nearest_neighbor_id >= 0:
                    obj["nearest_object"] = {
                        "id":         si.nearest_neighbor_id,
                        "distance_m": round(si.nearest_neighbor_dist, 2),
                    }

            if det.mask is not None:
                coverage = float(det.mask.sum()) / (h * w)
                obj["segmentation_coverage_pct"] = round(coverage * 100, 1)

            if det.track_id in cache:
                obj["cached_visual"] = cache[det.track_id]

            objects.append(obj)

        return _json.dumps(
            {"sensor_verified_detections": objects, "total_objects": len(objects)},
            indent=2,
        )


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _singular(word: str) -> str:
    """Strip trailing 's' for simple plural → singular."""
    if word and word.lower().endswith("s") and len(word) > 2:
        return word[:-1]
    return word


def _find_by_class(
    target: str, detections: list[Detection]
) -> list[tuple[int, Detection]]:
    """Case-insensitive class matching with plural handling."""
    t = target.lower()
    t_singular = _singular(t)
    results = []
    for i, det in enumerate(detections):
        name = det.class_name.lower()
        if name == t or name == t_singular or _singular(name) == t_singular:
            results.append((i, det))
    return results
