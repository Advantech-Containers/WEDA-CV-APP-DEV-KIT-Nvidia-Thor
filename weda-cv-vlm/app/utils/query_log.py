"""JSON Lines query logger for tuning Tier 1/2/2.5/3 routing.

Writes one JSON object per processed user query to
``${QUERY_LOG_DIR}/queries_YYYY-MM-DD.jsonl``. Intended for:

* tuning regex coverage in ``app/modules/query_engine.py`` — which queries
  fell through to Tier 3 that should have matched a Tier 1/2/2.5 handler?
* measuring VLM/Cosmos-Reason2 inference time distribution per tier.
* spot-checking response quality against the original query text.

Schema (one line per query)::

    {
      "ts":       "2026-04-21T14:03:22.123Z",
      "tier":     3,                      // 1 / 2 / 2.5 / 3
      "model":    "vlm",                  // vlm | cosmos | rule | null
      "query":    "is the person safe?",
      "response": "Yes, the person...",
      "elapsed_s": 1.84,
      "status":   "ok",                   // ok | load_failed | inference_error | empty
      "error":    ""                      // populated when status != ok
    }

Directory defaults mirror ``alert_logger.py`` — same volume mount so both
artefacts travel together when operators zip up ``./logs``.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Optional


_DEFAULT_LOG_DIR = os.environ.get(
    "QUERY_LOG_DIR",
    "/wise-edge/weda-cv-vlm/logs"
    if os.path.isdir("/wise-edge/weda-cv-vlm")
    else "logs",
)

_LOCK = threading.Lock()
_FILE = None
_CURRENT_DATE: Optional[str] = None
# Cap a single response at ~4 KB in the log so a runaway VLM generation
# doesn't blow up an otherwise line-delimited file.
_MAX_RESPONSE_CHARS = 4000


def _ensure_file():
    """Open (or rotate) the daily JSONL file. Caller must hold _LOCK."""
    global _FILE, _CURRENT_DATE
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _FILE is not None and _CURRENT_DATE == today:
        return _FILE
    # Close the previous day's file if we rolled over
    if _FILE is not None:
        try:
            _FILE.close()
        except Exception:
            pass
        _FILE = None
    try:
        os.makedirs(_DEFAULT_LOG_DIR, exist_ok=True)
        path = os.path.join(_DEFAULT_LOG_DIR, f"queries_{today}.jsonl")
        _FILE = open(path, "a", encoding="utf-8", buffering=1)
        _CURRENT_DATE = today
    except Exception:
        # Fallback to in-container relative path if the absolute mount is
        # read-only (dev laptop, CI). Still fail silently if that also fails.
        try:
            os.makedirs("logs", exist_ok=True)
            path = os.path.join("logs", f"queries_{today}.jsonl")
            _FILE = open(path, "a", encoding="utf-8", buffering=1)
            _CURRENT_DATE = today
        except Exception:
            _FILE = None
    return _FILE


def log_query(
    *,
    tier,
    model: str,
    query: str,
    response: str,
    elapsed_s: float,
    status: str = "ok",
    error: str = "",
) -> None:
    """Append one query record. Never raises — logging must not break the
    main pipeline. Use kwargs-only to avoid positional-argument drift.

    Args:
        tier:      1 | 2 | 2.5 | 3 (accepts int or float — recorded verbatim).
        model:     Free-form label: "vlm", "cosmos", "rule", "alert_search", …
        query:     The user's original question text.
        response:  The answer string shown to the user. Truncated to 4 KB.
        elapsed_s: Wall-clock seconds from query arrival to response ready.
        status:    "ok" / "load_failed" / "inference_error" / "empty".
        error:     Short error message when status != ok (empty otherwise).
    """
    try:
        record = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") +
                  f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z",
            "tier": tier,
            "model": model or "",
            "query": (query or "")[:1000],
            "response": (response or "")[:_MAX_RESPONSE_CHARS],
            "elapsed_s": round(float(elapsed_s), 3),
            "status": status or "ok",
            "error": (error or "")[:500],
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _LOCK:
            f = _ensure_file()
            if f is not None:
                f.write(line)
    except Exception:
        # Logger must never crash the pipeline. Silently drop on failure.
        return


def close() -> None:
    """Close the current log file. Safe to call multiple times."""
    global _FILE, _CURRENT_DATE
    with _LOCK:
        if _FILE is not None:
            try:
                _FILE.close()
            except Exception:
                pass
            _FILE = None
            _CURRENT_DATE = None
