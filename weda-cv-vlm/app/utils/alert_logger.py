"""JSON Lines alert logger for proximity and safety events.

Writes one JSON object per line to ``${ALERT_LOG_DIR}/alerts_YYYY-MM-DD.jsonl``.
Thread-safe, low-overhead (file opened once, flushed per write).

Default log directory (in-container): ``/wise-edge/weda-cv-vlm/logs``, which is
volume-mounted to ``./logs`` on the host by ``docker-compose.yml`` so alerts
survive container restarts. Override via the ``ALERT_LOG_DIR`` environment
variable.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional


# Default directory is absolute and matches the compose volume mount at
# /wise-edge/weda-cv-vlm/logs (bind-mounted to ./logs on the host). Falls back
# to a relative path only when that absolute path is unwritable (e.g. running
# the logger on a dev machine outside the container).
_DEFAULT_LOG_DIR = os.environ.get(
    "ALERT_LOG_DIR",
    "/wise-edge/weda-cv-vlm/logs"
    if os.path.isdir("/wise-edge/weda-cv-vlm")
    else "logs",
)


class AlertLogger:
    """Append-only JSON Lines logger for alerts."""

    def __init__(self, log_dir: Optional[str] = None) -> None:
        self._log_dir = log_dir if log_dir is not None else _DEFAULT_LOG_DIR
        self._lock = threading.Lock()
        self._file = None
        self._current_date: Optional[str] = None
        self._alert_count = 0
        self._alerts_by_type: dict[str, int] = {}
        self._cooldowns: dict[int, float] = {}
        self._cooldown_sec = 5.0

    @property
    def alert_count(self) -> int:
        return self._alert_count

    @property
    def alerts_by_type(self) -> dict[str, int]:
        with self._lock:
            return dict(self._alerts_by_type)

    @property
    def log_path(self) -> Optional[str]:
        if self._file is not None:
            return self._file.name
        return None

    def current_log_path(self) -> str:
        """Return the full path of the active `alerts_YYYY-MM-DD.jsonl` file.

        If no alert has been logged yet this session, returns the path that
        *would* be used for the first write. Suitable for printing in shutdown
        banners so operators know where to look for audit trail.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self._log_dir, f"alerts_{today}.jsonl")

    def log_proximity(
        self,
        object_class: str,
        distance_m: float,
        track_id: int = -1,
        nearest_object: str = "",
        nearest_distance: float = 0.0,
    ) -> None:
        """Log a proximity alert if cooldown has elapsed for this track."""
        now = time.monotonic()
        with self._lock:
            if track_id >= 0 and track_id in self._cooldowns:
                if now - self._cooldowns[track_id] < self._cooldown_sec:
                    return
            self._cooldowns[track_id] = now

        msg = f"{object_class} at {distance_m:.2f}m"
        if nearest_object:
            msg += f", {nearest_distance:.2f}m from {nearest_object}"

        self._write({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "proximity",
            "object": object_class,
            "distance_m": round(distance_m, 2),
            "nearest_object": nearest_object or None,
            "nearest_distance_m": round(nearest_distance, 2) if nearest_object else None,
            "track_id": track_id,
            "message": msg,
        })

    def log_safety(self, message: str, details: Optional[dict] = None) -> None:
        """Log a safety/physical reasoning alert."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "safety",
            "message": message,
        }
        if details:
            entry["details"] = details
        self._write(entry)

    def log_safety_alert(
        self,
        track_id: int = -1,
        distance_m: float = 0.0,
        class_name: str = "",
        frame_id: int = -1,
    ) -> None:
        """Log a safety zone alert from pipeline."""
        msg = f"{class_name} at {distance_m:.2f}m" if distance_m > 0 else f"{class_name} (estimated proximity)"
        self._write({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "safety",
            "object": class_name,
            "distance_m": round(distance_m, 2) if distance_m > 0 else None,
            "track_id": track_id,
            "frame_id": frame_id,
            "message": msg,
        })

    def _write(self, entry: dict) -> None:
        with self._lock:
            self._alert_count += 1
            alert_type = entry.get("type", "unknown")
            self._alerts_by_type[alert_type] = self._alerts_by_type.get(alert_type, 0) + 1
            try:
                self._ensure_file()
                self._file.write(json.dumps(entry) + "\n")
                self._file.flush()
            except OSError:
                pass

    def _ensure_file(self) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        if self._current_date == today and self._file is not None:
            return
        if self._file is not None:
            self._file.close()
        os.makedirs(self._log_dir, exist_ok=True)
        path = os.path.join(self._log_dir, f"alerts_{today}.jsonl")
        self._file = open(path, "a", encoding="utf-8")
        self._current_date = today

    def query_alerts(
        self,
        query: str,
        embedder=None,
        top_k: int = 5,
        limit: int = 100,
    ) -> str:
        """Query recent alerts, ranking them by semantic similarity if an embedder is provided."""
        if not os.path.isdir(self._log_dir):
            return "No safety alerts have been logged yet."

        log_files = []
        for f in os.listdir(self._log_dir):
            if f.startswith("alerts_") and f.endswith(".jsonl"):
                log_files.append(os.path.join(self._log_dir, f))

        if not log_files:
            return "No safety alerts have been logged yet."

        log_files.sort(reverse=True)

        entries = []
        for file_path in log_files:
            if len(entries) >= limit:
                break
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in reversed(lines):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                        if len(entries) >= limit:
                            break
                    except json.JSONDecodeError:
                        continue
            except OSError:
                continue

        if not entries:
            return "No safety alerts found in the logs."

        ranked_entries = []
        if embedder is not None and hasattr(embedder, "embed_text") and hasattr(embedder, "cosine_similarity"):
            try:
                query_emb = embedder.embed_text(query)
                for entry in entries:
                    msg = entry.get("message", "")
                    if msg:
                        msg_emb = embedder.embed_text(msg)
                        sim = embedder.cosine_similarity(query_emb, msg_emb)
                        ranked_entries.append((sim, entry))
                    else:
                        ranked_entries.append((0.0, entry))
                ranked_entries.sort(key=lambda x: (x[0], x[1].get("timestamp", "")), reverse=True)
                ranked_entries = [item[1] for item in ranked_entries[:top_k]]
            except Exception:
                ranked_entries = entries[:top_k]
        else:
            ranked_entries = entries[:top_k]

        lines_out = [f"Found {len(ranked_entries)} safety alerts matching your query:"]
        for idx, entry in enumerate(ranked_entries, 1):
            ts = entry.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    ts_str = ts
            else:
                ts_str = "unknown time"

            alert_type = entry.get("type", "alert").upper()
            msg = entry.get("message", "Safety event detected")
            lines_out.append(f"  {idx}. [{ts_str}] {alert_type}: {msg}")

        return "\n".join(lines_out)

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None
