"""3-D spatial primitives over the live FramePacket.

Cheap geometry on detections + spatial; no model inference. Exposed
through /v1/spatial/*:

  - distance(track_a, track_b)         metres
  - path_clear(from, to, width_m)      bool + list of blockers
  - zones_now(zones_cfg)               which tracks occupy which zones
  - measure(bbox_xyxy)                 physical width/height in metres
  - predict_collisions(horizon_s)      list[{a, b, eta_s, closing_mps}]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.utils.data_structures import Detection, FramePacket, SpatialInfo
from app.utils.logger import get_logger

logger = get_logger()


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _track_to_index(packet: FramePacket, track_id: int) -> Optional[int]:
    """Return index in `packet.detections` for a given track_id, or None."""
    for i, d in enumerate(packet.detections):
        if d.track_id == track_id:
            return i
    return None


def _spatial_for_track(packet: FramePacket, track_id: int) -> Optional[SpatialInfo]:
    idx = _track_to_index(packet, track_id)
    if idx is None or idx >= len(packet.spatial):
        return None
    return packet.spatial[idx]


def _xyz(spatial: SpatialInfo) -> Optional[np.ndarray]:
    """Return (3,) world-coords for a SpatialInfo, or None if unavailable."""
    if spatial is None or spatial.position_3d is None:
        return None
    p = np.asarray(spatial.position_3d, dtype=np.float32)
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        return None
    return p


# ─────────────────────────────────────────────────────────────────
# Public results — JSON-serialisable dataclasses
# ─────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class DistanceResult:
    track_a: int
    track_b: int
    distance_m: Optional[float]   # None if either track lacks 3-D pose
    reliable: bool                # both tracks had depth_reliable=True


@dataclass(slots=True)
class PathBlocker:
    track_id: int
    class_name: str
    distance_m: float             # distance from `from_pos` to blocker centre
    lateral_offset_m: float       # perpendicular distance from the path line


@dataclass(slots=True)
class PathClearResult:
    clear: bool
    blockers: list[PathBlocker]
    path_length_m: float


@dataclass(slots=True)
class ZoneOccupancyResult:
    zone_name: str
    track_ids: list[int]
    classes: list[str]


@dataclass(slots=True)
class MeasureResult:
    width_m: float
    height_m: float
    distance_m: float
    reliable: bool


@dataclass(slots=True)
class CollisionPrediction:
    track_a: int
    track_b: int
    class_a: str
    class_b: str
    eta_s: float                   # seconds until closest approach
    closing_speed_mps: float       # positive = approaching
    miss_distance_m: float         # how close they get at the closest point


# ─────────────────────────────────────────────────────────────────
# 1. Distance between two tracks
# ─────────────────────────────────────────────────────────────────

def distance(packet: FramePacket, track_a: int, track_b: int) -> DistanceResult:
    """Euclidean 3-D distance between two tracked objects.

    Returns DistanceResult with `distance_m=None` if either track is
    missing a reliable 3-D position. Caller should treat that as
    unverifiable rather than zero.
    """
    sa = _spatial_for_track(packet, track_a)
    sb = _spatial_for_track(packet, track_b)
    pa = _xyz(sa) if sa else None
    pb = _xyz(sb) if sb else None
    if pa is None or pb is None:
        return DistanceResult(track_a, track_b, None, False)
    d = float(np.linalg.norm(pa - pb))
    reliable = bool(getattr(sa, "depth_reliable", True) and
                    getattr(sb, "depth_reliable", True))
    return DistanceResult(track_a, track_b, d, reliable)


# ─────────────────────────────────────────────────────────────────
# 2. Path-clear test
# ─────────────────────────────────────────────────────────────────

def path_clear(
    packet: FramePacket,
    from_pos: np.ndarray,
    to_pos: np.ndarray,
    width_m: float = 0.8,
    ignore_track_ids: Optional[set[int]] = None,
) -> PathClearResult:
    """Is the straight-line corridor from `from_pos` to `to_pos` clear?

    Treats every other track as a cylinder of radius `width_m / 2`
    around its 3-D centre. A blocker is any track whose perpendicular
    distance from the path line is below that radius.

    Args:
        from_pos / to_pos: (3,) float arrays in world coords (metres)
        width_m: path width — cylinder diameter to test against
        ignore_track_ids: tracks to skip (typically the operator's
            own track + the destination object)
    """
    ignore = ignore_track_ids or set()
    fp = np.asarray(from_pos, dtype=np.float32)
    tp = np.asarray(to_pos, dtype=np.float32)
    seg = tp - fp
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1e-3:
        return PathClearResult(clear=True, blockers=[], path_length_m=0.0)
    seg_unit = seg / seg_len
    radius = width_m / 2.0

    blockers: list[PathBlocker] = []
    for det, sp in zip(packet.detections, packet.spatial):
        if det.track_id in ignore:
            continue
        p = _xyz(sp)
        if p is None:
            continue
        # Project (p - fp) onto seg_unit — that's the along-path distance.
        along = float(np.dot(p - fp, seg_unit))
        if along < 0 or along > seg_len:
            continue   # behind us or past the destination
        # Perpendicular distance from the line.
        perp = float(np.linalg.norm((p - fp) - along * seg_unit))
        if perp <= radius:
            blockers.append(PathBlocker(
                track_id=det.track_id,
                class_name=det.class_name,
                distance_m=along,
                lateral_offset_m=perp,
            ))
    blockers.sort(key=lambda b: b.distance_m)
    return PathClearResult(
        clear=not blockers,
        blockers=blockers,
        path_length_m=seg_len,
    )


# ─────────────────────────────────────────────────────────────────
# 3. Zone occupancy
# ─────────────────────────────────────────────────────────────────

def zones_now(
    packet: FramePacket,
    zones: list[dict],
) -> list[ZoneOccupancyResult]:
    """Test every zone (defined as a 2-D pixel-space polygon) against
    every detection's bbox centre.

    Args:
        zones: list of `{"name": str, "polygon": [[x,y], [x,y], ...]}`
            polygons in PIXEL coordinates (matching `det.bbox`).

    Why pixel-space (not 3-D world)? Zones are configured in
    ``config/zones.yaml`` (or written there by an operator-facing
    drawing tool) using pixel coordinates aligned with ``det.bbox``.
    3-D zones are a future extension.
    """
    out: list[ZoneOccupancyResult] = []
    for zone in zones:
        name = zone.get("name", "unnamed")
        poly = zone.get("polygon") or []
        if len(poly) < 3:
            continue   # not a polygon
        poly_arr = np.asarray(poly, dtype=np.float32)
        ids: list[int] = []
        classes: list[str] = []
        for det in packet.detections:
            cx, cy = det.center
            if _point_in_poly(cx, cy, poly_arr):
                ids.append(det.track_id)
                classes.append(det.class_name)
        out.append(ZoneOccupancyResult(zone_name=name, track_ids=ids, classes=classes))
    return out


def _point_in_poly(x: float, y: float, poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test. poly is (N, 2)."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


# ─────────────────────────────────────────────────────────────────
# 4. Measure a region
# ─────────────────────────────────────────────────────────────────

def measure(packet: FramePacket, bbox_xyxy: tuple[float, float, float, float]) -> MeasureResult:
    """Estimate physical dimensions of an arbitrary bbox region.

    Strategy: find the detection whose bbox most overlaps the requested
    region; use its existing SpatialInfo.size_estimate (already populated
    by the depth/spatial stage). Falls back to (0, 0, 0, False) if no
    detection overlaps — agents should treat this as unmeasurable.
    """
    x1, y1, x2, y2 = bbox_xyxy
    best_iou = 0.0
    best_idx = -1
    for i, det in enumerate(packet.detections):
        iou = _iou(bbox_xyxy, tuple(det.bbox.tolist()))
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    if best_idx < 0 or best_iou < 0.1:
        return MeasureResult(width_m=0.0, height_m=0.0, distance_m=0.0, reliable=False)
    sp = packet.spatial[best_idx] if best_idx < len(packet.spatial) else None
    if sp is None:
        return MeasureResult(width_m=0.0, height_m=0.0, distance_m=0.0, reliable=False)
    w_m, h_m = sp.size_estimate
    return MeasureResult(
        width_m=float(w_m),
        height_m=float(h_m),
        distance_m=float(sp.distance_m),
        reliable=bool(sp.depth_reliable),
    )


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-Union of two xyxy bboxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return float(inter / union) if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────
# Predictive collisions
# ─────────────────────────────────────────────────────────────────

# Spatial-bin pruning: hash tracks into a uniform 3-D grid keyed on
# `max_pair_distance_m`. For sparse industrial scenes this turns the
# pairwise loop into O(N · k) where k is the average neighbour count
# (typically ≤ 4). For dense scenes the bin cell size keeps pairs
# bounded — no quadratic blow-up regardless of N.

_BIN_PRUNE_THRESHOLD = 16   # below this, full O(n²) is faster than binning


def _bin_neighbours(
    positions: list[Optional[np.ndarray]],
    cell_m: float,
) -> dict[tuple[int, int, int], list[int]]:
    bins: dict[tuple[int, int, int], list[int]] = {}
    inv = 1.0 / cell_m
    for i, p in enumerate(positions):
        if p is None:
            continue
        key = (int(p[0] * inv), int(p[1] * inv), int(p[2] * inv))
        bins.setdefault(key, []).append(i)
    return bins


def _candidate_pairs(
    positions: list[Optional[np.ndarray]],
    cell_m: float,
) -> list[tuple[int, int]]:
    bins = _bin_neighbours(positions, cell_m)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for key, ids in bins.items():
        # 27 neighbouring cells (incl. self); each pair counted once.
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nb = bins.get((key[0] + dx, key[1] + dy, key[2] + dz))
                    if not nb:
                        continue
                    for i in ids:
                        for j in nb:
                            if i >= j:
                                continue
                            pk = (i, j)
                            if pk in seen:
                                continue
                            seen.add(pk)
                            pairs.append(pk)
    return pairs


def predict_collisions(
    packet: FramePacket,
    horizon_s: float = 3.0,
    danger_distance_m: float = 0.8,
    min_closing_mps: float = 0.3,
    max_pair_distance_m: float = 5.0,
) -> list[CollisionPrediction]:
    """Closest-approach prediction within ``horizon_s``.

    Algorithm: per pair, solve d|p_i+v_i·t - p_j-v_j·t|² / dt = 0 for
    t* in closed form (linear-relative-motion model). Filter by
    horizon, miss distance, and closing speed.

    Complexity: O(N) for small N (≤ _BIN_PRUNE_THRESHOLD) via direct
    iteration; O(N · neighbours) for larger N via uniform-grid
    binning at cell size ``max_pair_distance_m``. Avoids accidental
    quadratic blow-up on dense scenes.
    """
    n = len(packet.detections)
    if n < 2:
        return []
    positions: list[Optional[np.ndarray]] = []
    velocities: list[Optional[np.ndarray]] = []
    for sp in packet.spatial:
        positions.append(_xyz(sp))
        v = getattr(sp, "velocity_3d", None) if sp is not None else None
        velocities.append(np.asarray(v, dtype=np.float32) if v is not None else None)

    if n >= _BIN_PRUNE_THRESHOLD:
        candidate_pairs = _candidate_pairs(positions, max_pair_distance_m)
    else:
        candidate_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    out: list[CollisionPrediction] = []
    for i, j in candidate_pairs:
        pi, pj = positions[i], positions[j]
        vi, vj = velocities[i], velocities[j]
        if pi is None or pj is None or vi is None or vj is None:
            continue
        d_now = float(np.linalg.norm(pi - pj))
        if d_now > max_pair_distance_m:
            continue
        dp = pi - pj
        dv = vi - vj
        denom = float(np.dot(dv, dv))
        if denom < 1e-6:
            continue
        t_star = -float(np.dot(dp, dv)) / denom
        if t_star < 0 or t_star > horizon_s:
            continue
        closest = dp + dv * t_star
        miss = float(np.linalg.norm(closest))
        if miss > danger_distance_m:
            continue
        dir_ji = -dp
        dir_norm = float(np.linalg.norm(dir_ji))
        if dir_norm < 1e-6:
            continue
        closing = float(np.dot(vi - vj, dir_ji / dir_norm))
        if closing < min_closing_mps:
            continue
        out.append(CollisionPrediction(
            track_a=packet.detections[i].track_id,
            track_b=packet.detections[j].track_id,
            class_a=packet.detections[i].class_name,
            class_b=packet.detections[j].class_name,
            eta_s=t_star,
            closing_speed_mps=closing,
            miss_distance_m=miss,
        ))
    out.sort(key=lambda c: c.eta_s)
    return out
