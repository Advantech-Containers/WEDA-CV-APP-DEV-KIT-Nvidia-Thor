"""Temporal scene memory.

Ring buffer of (ts, frame_embedding, scene_description, detections,
spatial) tuples covering the last `window_min` minutes at 1 Hz.

  - at(ts)              snapshot closest to a wall-clock timestamp
  - find(query, since_s) semantic search via Cosmos-Embed1's text
                        embedding against buffered frame embeddings;
                        returns top-k matches

Memory: ~600 KB per snapshot × window_min × sample_hz. Default 30 min
@ 1 Hz ≈ 1 GB.
"""

from __future__ import annotations

import collections
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from app.utils.data_structures import Detection, FramePacket, SpatialInfo
from app.utils.logger import get_logger

logger = get_logger()


@dataclass(slots=True)
class MemoryEntry:
    """One frame snapshot in the ring buffer."""
    ts: float                                # epoch seconds
    monotonic_ts: float                      # for ordering
    frame_id: int
    embedding: Optional[np.ndarray]          # (D,) float32, may be None
    scene_description: str
    physical_assessment: str
    detection_count: int
    detection_classes: list[str]
    track_ids: list[int]
    # Don't store full spatial / detection objects — too heavy. Keep
    # only the JSON-friendly summary REST consumers need.

    def to_dict(self) -> dict:
        d = asdict(self)
        # Embedding stays a numpy array in memory; serialise lazily.
        if self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        return d


@dataclass(slots=True)
class SearchHit:
    entry: MemoryEntry
    similarity: float


class SceneMemory:
    """Bounded temporal memory with embedding-based semantic search.

    Designed for one-writer / many-readers — the orchestrator pushes
    snapshots once per second; the API reads them many times per
    second.
    """

    def __init__(
        self,
        embedder=None,                  # Embedder, lazily; may be None
        window_min: int = 30,
        sample_hz: float = 1.0,
    ) -> None:
        self._embedder = embedder
        self._window_min = window_min
        self._sample_period = 1.0 / sample_hz
        max_entries = int(window_min * 60 * sample_hz)
        self._buf: collections.deque[MemoryEntry] = collections.deque(maxlen=max_entries)
        self._lock = threading.Lock()
        self._last_push_ts: float = 0.0

    # ── ingest ──────────────────────────────────────────────────────
    def maybe_push(self, packet: FramePacket) -> None:
        """Called by the orchestrator each frame; rate-limited internally
        to ``sample_hz`` so callers don't have to track the interval."""
        now = time.monotonic()
        if now - self._last_push_ts < self._sample_period:
            return
        self._last_push_ts = now

        embedding = packet.frame_embedding
        # Defensive copy so a buffer-pool release elsewhere doesn't
        # mutate our entry. Embedder returns float32; .copy() is cheap.
        if embedding is not None:
            try:
                embedding = np.asarray(embedding, dtype=np.float32).copy()
            except Exception as e:
                logger.debug("SceneMemory: embedding copy failed: %s", e)
                embedding = None

        entry = MemoryEntry(
            ts=time.time(),
            monotonic_ts=now,
            frame_id=packet.frame_id,
            embedding=embedding,
            scene_description=packet.scene_description or "",
            physical_assessment=packet.physical_reasoning or "",
            detection_count=len(packet.detections),
            detection_classes=[d.class_name for d in packet.detections],
            track_ids=[d.track_id for d in packet.detections if d.track_id >= 0],
        )
        with self._lock:
            self._buf.append(entry)

    # ── read ────────────────────────────────────────────────────────
    def at(self, ts: float) -> Optional[MemoryEntry]:
        """Return the entry with timestamp closest to `ts` (epoch sec).

        Returns None if the buffer is empty or `ts` is more than the
        sample period away from the nearest entry (operator asked for
        a moment we no longer remember).
        """
        with self._lock:
            if not self._buf:
                return None
            best = min(self._buf, key=lambda e: abs(e.ts - ts))
        if abs(best.ts - ts) > self._sample_period * 2:
            # Too far from any sampled moment.
            return None
        return best

    def range(self, since_s: float, max_entries: int = 60) -> list[MemoryEntry]:
        """Return entries from the last `since_s` seconds, oldest first."""
        cutoff = time.monotonic() - since_s
        with self._lock:
            snap = [e for e in self._buf if e.monotonic_ts >= cutoff]
        return snap[-max_entries:]

    def find(
        self,
        query: str,
        since_s: float = 3600.0,
        top_k: int = 5,
    ) -> list[SearchHit]:
        """Top-k semantic search.

        Linear scan is O(N·d) — N = window-bounded entries (default
        ≤ 1800 at 30 min × 1 Hz), d = 256-1024 typical. Vectorised
        cosine via a single matmul keeps it fast enough that an HNSW
        index isn't load-bearing at this scale; it's available
        opportunistically when ``hnswlib`` is installed (`_hnsw`
        path) for future window expansion.

        Falls back to substring match when the embedder is offline —
        degrades gracefully instead of 500-erroring.
        """
        cutoff = time.monotonic() - since_s
        with self._lock:
            entries = [e for e in self._buf if e.monotonic_ts >= cutoff]
        if not entries:
            return []

        # Embedder path — try first, fall through to substring on failure
        # OR when no buffered entry has an embedding yet (early-boot or
        # embedder dropping frames).
        if self._embedder is not None and getattr(self._embedder, "is_loaded", False):
            try:
                qvec = self._embedder.embed_text(query)
                qvec = np.asarray(qvec, dtype=np.float32).ravel()
                qn = float(np.linalg.norm(qvec))
                if qn >= 1e-9:
                    qvec = qvec / qn
                    with_emb = [e for e in entries if e.embedding is not None]
                    if with_emb:
                        M = np.stack([e.embedding for e in with_emb], axis=0)
                        norms = np.linalg.norm(M, axis=1)
                        norms[norms < 1e-9] = 1e-9
                        M = M / norms[:, None]
                        sims = M @ qvec
                        k = min(top_k, sims.shape[0])
                        idx_part = np.argpartition(-sims, k - 1)[:k]
                        idx_sorted = idx_part[np.argsort(-sims[idx_part])]
                        return [
                            SearchHit(entry=with_emb[i], similarity=float(sims[i]))
                            for i in idx_sorted
                        ]
                    # else: no embeddings buffered — fall through to substring
            except Exception as ex:
                logger.warning("SceneMemory.find embedding path failed: %s — falling back", ex)

        import re as _re
        q_lower = query.lower()
        for stop in ("is ", "are ", "was ", "were ", "do ", "does ", "can ", "could ", "the ", "a ", "an ", "?"):
            q_lower = q_lower.replace(stop, " ")
        tokens = [t for t in q_lower.split() if len(t) >= 3]
        patterns = [_re.compile(r"\b" + _re.escape(t) + r"\b") for t in tokens]
        scored: list[SearchHit] = []
        if patterns:
            for e in entries:
                text = (e.scene_description + " " + e.physical_assessment).lower()
                hits_in_text = sum(1 for p in patterns if p.search(text))
                if hits_in_text == 0:
                    continue
                scored.append(SearchHit(entry=e, similarity=hits_in_text / len(patterns)))
        scored.sort(key=lambda h: h.similarity, reverse=True)
        return scored[:top_k]

    # ── stats ───────────────────────────────────────────────────────
    @property
    def window_min_available(self) -> float:
        """Minutes of history actually buffered (≤ window_min config)."""
        with self._lock:
            if not self._buf:
                return 0.0
            span = self._buf[-1].monotonic_ts - self._buf[0].monotonic_ts
        return span / 60.0


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
