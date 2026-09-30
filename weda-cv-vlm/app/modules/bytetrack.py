"""ByteTrack multi-object tracker — handles low-confidence detections.

ByteTrack's key innovation: tracks objects even when detection confidence
drops (partial occlusion, motion blur) by using a two-stage matching:
  1. High-confidence detections matched to existing tracks (IoU)
  2. Low-confidence detections matched to unmatched tracks (IoU)

This prevents track ID loss during brief confidence dips — common in
industrial scenarios with variable lighting and occlusion.

Appearance embedding re-ID for 60s dormant track recovery.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from app.utils.data_structures import Detection
from app.utils.logger import get_logger

logger = get_logger()

@dataclass
class BTrack:
    track_id: int
    bbox: np.ndarray
    class_id: int
    class_name: str
    last_seen: float
    age: int = 1
    hits: int = 1
    score: float = 0.0
    embedding: Optional[np.ndarray] = None
    is_activated: bool = True

class ByteTracker:
    """ByteTrack-style tracker with appearance re-ID.

    Args:
        high_thresh: confidence threshold for first-stage matching
        low_thresh: confidence threshold for second-stage (low-conf recovery)
        iou_threshold: IoU threshold for matching
        max_age_sec: seconds before track is permanently lost
        embedding_threshold: cosine similarity threshold for re-ID
        min_display_age: minimum consecutive hits before showing track
    """

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        iou_threshold: float = 0.3,
        max_age_sec: float = 60.0,
        class_match: bool = True,
        embedding_threshold: float = 0.7,
        min_display_age: int = 3,
        dormancy_sec: float = 1.0,
    ) -> None:
        self._high_thresh = high_thresh
        self._low_thresh = low_thresh
        self._iou_threshold = iou_threshold
        self._max_age = max_age_sec
        self._class_match = class_match
        self._embedding_threshold = embedding_threshold
        self._min_display_age = min_display_age
        self._dormancy_sec = dormancy_sec
        self._tracks: dict[int, BTrack] = {}
        self._dormant: dict[int, BTrack] = {}
        self._next_id = 1
        self._target_track_id: Optional[int] = None
        self._locked_track_id: Optional[int] = None

    def update(self, detections: list[Detection]) -> list[Detection]:
        now = time.monotonic()
        self._prune_stale(now)
        if not detections:
            return detections

        high_dets = [(i, d) for i, d in enumerate(detections) if d.confidence >= self._high_thresh]
        low_dets = [(i, d) for i, d in enumerate(detections) if self._low_thresh <= d.confidence < self._high_thresh]
        matched_track_ids: set[int] = set()
        matched_dets: set[int] = set()

        if self._tracks and high_dets:
            track_ids = list(self._tracks.keys())
            track_bboxes = np.array([self._tracks[t].bbox for t in track_ids])
            det_bboxes = np.array([d.bbox for _, d in high_dets])
            iou_matrix = _batch_iou(track_bboxes, det_bboxes)
            for ti, tid in enumerate(track_ids):
                track = self._tracks[tid]
                if track.embedding is None:
                    continue
                for di, (_, det) in enumerate(high_dets):
                    if det.embedding is not None:
                        sim = _cosine_sim(track.embedding, det.embedding)
                        iou_matrix[ti, di] += sim * 0.3  # Weighted combo

            self._greedy_match(
                iou_matrix, track_ids, high_dets,
                detections, matched_track_ids, matched_dets, now,
            )

        unmatched_track_ids = [
            tid for tid in self._tracks if tid not in matched_track_ids
        ]
        if unmatched_track_ids and low_dets:
            track_bboxes = np.array([self._tracks[t].bbox for t in unmatched_track_ids])
            det_bboxes = np.array([d.bbox for _, d in low_dets])
            iou_matrix = _batch_iou(track_bboxes, det_bboxes)

            for ti, tid in enumerate(unmatched_track_ids):
                for di, (orig_idx, _) in enumerate(low_dets):
                    if orig_idx in matched_dets:
                        iou_matrix[ti, di] = 0

            low_matched_tracks: set[int] = set()
            low_matched_dets: set[int] = set()
            self._greedy_match(
                iou_matrix, unmatched_track_ids, low_dets,
                detections, low_matched_tracks, low_matched_dets, now,
            )
            matched_dets.update(low_matched_dets)

        unmatched_det_indices = [i for i in range(len(detections)) if i not in matched_dets]
        if unmatched_det_indices and self._dormant:
            self._try_reidentify(detections, unmatched_det_indices, matched_dets, now)
        for i in range(len(detections)):
            if i not in matched_dets and detections[i].confidence >= self._high_thresh:
                det = detections[i]
                tid = self._assign_new_id()
                det.track_id = tid
                self._tracks[tid] = BTrack(
                    track_id=tid, bbox=det.bbox.copy(),
                    class_id=det.class_id, class_name=det.class_name,
                    last_seen=now, embedding=getattr(det, 'embedding', None),
                    score=det.confidence,
                )

        active_ids = list(self._tracks.keys())
        for tid in active_ids:
            if tid == self._locked_track_id:
                continue
            track = self._tracks[tid]
            if now - track.last_seen > self._dormancy_sec:
                if track.embedding is not None:
                    self._dormant[tid] = self._tracks.pop(tid)

        return detections

    def _greedy_match(
        self, iou_matrix, track_ids, det_pairs,
        detections, matched_tracks, matched_dets, now,
    ):
        if self._locked_track_id is not None and iou_matrix.size > 0:
            for ti, tid in enumerate(track_ids):
                if tid == self._locked_track_id:
                    iou_matrix[ti, :] = iou_matrix[ti, :] + 0.5

        while True:
            if iou_matrix.size == 0:
                break
            best = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            best_score = iou_matrix[best]
            if best_score < self._iou_threshold:
                break
            ti, di = int(best[0]), int(best[1])
            tid = track_ids[ti]
            orig_idx, det = det_pairs[di]

            if self._class_match and det.class_id != self._tracks[tid].class_id:
                iou_matrix[ti, di] = 0
                continue

            detections[orig_idx].track_id = tid
            track = self._tracks[tid]
            _A = 0.55
            track.bbox = tuple(_A * nb + (1.0 - _A) * ob
                               for nb, ob in zip(det.bbox, track.bbox))
            track.last_seen = now
            track.age += 1
            track.hits += 1
            track.score = det.confidence
            if getattr(det, 'embedding', None) is not None:
                track.embedding = det.embedding

            matched_tracks.add(tid)
            matched_dets.add(orig_idx)
            iou_matrix[ti, :] = 0
            iou_matrix[:, di] = 0

    def _try_reidentify(self, detections, unmatched, matched_dets, now):
        dormant_ids = list(self._dormant.keys())
        for di in unmatched:
            det = detections[di]
            if getattr(det, 'embedding', None) is None:
                continue
            if di in matched_dets:
                continue
            best_sim = -1.0
            best_tid = -1
            for tid in dormant_ids:
                track = self._dormant.get(tid)
                if track is None or track.embedding is None:
                    continue
                if self._class_match and det.class_id != track.class_id:
                    continue
                sim = _cosine_sim(det.embedding, track.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_tid = tid
            if best_sim >= self._embedding_threshold and best_tid >= 0:
                track = self._dormant.pop(best_tid)
                track.bbox = det.bbox.copy()
                track.last_seen = now
                track.hits += 1
                track.embedding = det.embedding
                self._tracks[best_tid] = track
                det.track_id = best_tid
                matched_dets.add(di)
                logger.debug("Re-ID: track #%d recovered (sim=%.2f)", best_tid, best_sim)

    # ── Target tracking ──

    def set_target(self, class_name=None, track_id=None) -> Optional[int]:
        if track_id is not None and track_id in self._tracks:
            self._target_track_id = track_id
            return track_id
        if class_name:
            best_tid, best_hits = None, -1
            for tid, track in self._tracks.items():
                if track.class_name.lower() == class_name.lower():
                    if track.hits > best_hits:
                        best_hits = track.hits
                        best_tid = tid
            if best_tid is not None:
                self._target_track_id = best_tid
                return best_tid
        return None

    def clear_target(self) -> None:
        self._target_track_id = None

    @property
    def target_track_id(self) -> Optional[int]:
        return self._target_track_id

    # ── Locked-track (persistent) ──

    def set_locked(
        self,
        track_id: Optional[int] = None,
        class_name: Optional[str] = None,
    ) -> Optional[int]:
        """Pin a track so it survives dormancy / max_age / marginal matches.

        Single-lock semantics: a new call replaces the previous lock.
        Passing both args as ``None`` is equivalent to ``clear_locked()``.

        Resolution mirrors ``set_target``:
          - ``track_id`` directly locks that active track (if present)
          - ``class_name`` picks the most established track of that class
            (highest hit count)

        Returns the resolved track_id, or ``None`` if no matching track
        exists. When locked, the track receives a +0.5 match-score boost
        in Stage-1 / Stage-2 matching, is exempt from migration to the
        dormant pool, and is never pruned by ``max_age_sec``.
        """
        if track_id is None and class_name is None:
            self._locked_track_id = None
            return None
        resolved = self.set_target(class_name=class_name, track_id=track_id)
        self._locked_track_id = resolved
        if resolved is not None:
            logger.info(
                "[LOCK] track_id=%d class=%s",
                resolved,
                self._tracks[resolved].class_name if resolved in self._tracks else "?",
            )
        else:
            logger.info(
                "[LOCK] no matching track for class_name=%r track_id=%r",
                class_name, track_id,
            )
        return resolved

    def clear_locked(self) -> None:
        if self._locked_track_id is not None:
            logger.info("[LOCK] released (was track_id=%d)", self._locked_track_id)
        self._locked_track_id = None

    @property
    def locked_track_id(self) -> Optional[int]:
        return self._locked_track_id

    def is_display_ready(self, track_id: int) -> bool:
        track = self._tracks.get(track_id)
        return track is not None and track.hits >= self._min_display_age

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    @property
    def active_track_ids(self) -> set[int]:
        return set(self._tracks.keys())

    def reset(self) -> None:
        self._tracks.clear()
        self._dormant.clear()
        self._next_id = 1
        self._target_track_id = None
        self._locked_track_id = None

    def _assign_new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def _prune_stale(self, now: float) -> None:
        locked = self._locked_track_id
        for pool in (self._tracks, self._dormant):
            stale = [
                tid for tid, t in pool.items()
                if tid != locked and now - t.last_seen > self._max_age
            ]
            for tid in stale:
                del pool[tid]

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _batch_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)
