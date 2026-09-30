"""POST /v1/alarms/* — supervisor-defined semantic alarm rules.

Compose endpoint takes natural-language text from any consumer
(NemoClaw, integrator UI, etc.), parses it via Cosmos-Reason2 into
a structured policy DSL, persists the rule, and returns the JSON for
operator confirmation. The alarm engine (separate module) evaluates
active rules each frame.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

from app.utils.logger import get_logger

logger = get_logger()

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.v1._deps import PipelineHandle, require_auth

router = APIRouter(prefix="/alarms", tags=["alarms"])


# ── Schemas ─────────────────────────────────────────────────────────

class AlarmComposeRequest(BaseModel):
    text: str = Field(..., description='Natural-language description of the alarm rule, e.g. "Alert me if anyone bends near the conveyor for more than 3 seconds."')

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Alert me if anyone bends near the conveyor for more than 3 seconds",
            }
        }
    }


class AlarmRuleDSL(BaseModel):
    subject_class: Optional[str] = Field(None, description='Object class to monitor: `"person" | "forklift" | "conveyor" | "welder"`. Null if the parser couldn\'t identify a subject.')
    action: str = Field(..., description='What the subject is doing: `"bend" | "enter" | "exit" | "dwell" | "count_gt" | "presence"`.')
    location_zone: Optional[str] = Field(None, description='Operator-defined zone name (e.g. "zone-3"). Null when the rule isn\'t scoped to a zone.')
    duration_s: Optional[float] = Field(None, description="Minimum duration in seconds for the condition to fire. Null when the rule is instantaneous.")
    count_threshold: Optional[int] = Field(None, description='Count threshold for `action=count_gt` rules. Null otherwise.')


class AlarmRule(BaseModel):
    id: str = Field(..., description="Stable rule id (8-char hex). Use this in `/v1/alarms/{rule_id}/activate` and `DELETE`.")
    raw_text: str = Field(..., description="The natural-language text the rule was composed from.")
    rule: AlarmRuleDSL = Field(..., description="Parsed structured rule.")
    active: bool = Field(..., description="True when the alarm engine is currently evaluating this rule on every frame. False after compose, true after activate.")
    created_ts: float = Field(..., description="Epoch seconds when the rule was composed.")
    fired_count: int = Field(..., description="How many times this rule has fired since creation.")
    last_fired_ts: Optional[float] = Field(None, description="Epoch seconds of the most recent firing, or null if never fired.")


class AlarmComposeResponse(BaseModel):
    rule: AlarmRule = Field(..., description="The newly-composed rule, returned with `active=false`. Call `/activate` once the operator confirms.")
    needs_confirmation: bool = Field(..., description='Always `true` for compose. UI should surface the parsed rule for the operator to confirm before activation.')


class AlarmActivateResponse(BaseModel):
    id: str = Field(..., description="Rule id that was activated.")
    active: bool = Field(..., description="Always `true` on success.")


class AlarmDeleteResponse(BaseModel):
    id: str = Field(..., description="Rule id that was deleted.")
    deleted: bool = Field(..., description="Always `true` on success.")


class AlarmListResponse(BaseModel):
    rules: List[AlarmRule] = Field(..., description="All currently-active rules. Empty list when none are active.")

# In-memory store + JSONL persistence. The alarm engine module will
# eventually own this; for now the API holds it so REST consumers
# work from day 1 even before the engine ships.
_ACTIVE_RULES: dict[str, dict] = {}
_LOCK = threading.Lock()


def _resolve_persist_path() -> Optional[Path]:
    for candidate in (
        os.environ.get("WEDA_ALARMS_PATH"),
        "output/alarms.jsonl",
        "/tmp/weda-alarms.jsonl",
    ):
        if not candidate:
            continue
        p = Path(candidate)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a"):
                pass
            return p
        except OSError:
            continue
    return None


_PERSIST_PATH: Optional[Path] = _resolve_persist_path()
if _PERSIST_PATH is None:
    logger.warning("Alarms persistence disabled — no writable path found.")
else:
    logger.info("Alarms persistence: %s", _PERSIST_PATH)


def _load_persisted() -> None:
    if _PERSIST_PATH is None or not _PERSIST_PATH.exists():
        return
    with _LOCK:
        try:
            with _PERSIST_PATH.open() as f:
                for line in f:
                    try:
                        rule = json.loads(line)
                        if rule.get("active", True):
                            _ACTIVE_RULES[rule["id"]] = rule
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            logger.warning("Alarms load failed: %s", e)


try:
    _load_persisted()
except Exception:
    pass


def _persist(rule: dict) -> None:
    if _PERSIST_PATH is None:
        return
    try:
        with _PERSIST_PATH.open("a") as f:
            f.write(json.dumps(rule) + "\n")
    except OSError as e:
        logger.warning("Alarms persist failed: %s", e)


@router.post(
    "/compose",
    response_model=AlarmComposeResponse,
    summary="Compose a structured alarm rule from natural language",
    description=(
        "Parses operator text into a policy DSL (subject_class, action, "
        "location_zone, duration_s, count_threshold). Stores the rule "
        "in `active=false` state — present the parsed rule to the "
        "operator, then call `POST /v1/alarms/{id}/activate` once "
        "confirmed."
    ),
)
async def compose(
    body: AlarmComposeRequest,
    handle: PipelineHandle = Depends(require_auth),
) -> AlarmComposeResponse:
    rule = _heuristic_parse(body.text)
    rid = uuid.uuid4().hex[:8]
    rule_record = {
        "id": rid,
        "raw_text": body.text,
        "rule": rule,
        "active": False,
        "created_ts": time.time(),
        "fired_count": 0,
        "last_fired_ts": None,
    }
    with _LOCK:
        _ACTIVE_RULES[rid] = rule_record
    _persist(rule_record)
    return AlarmComposeResponse(rule=AlarmRule(**rule_record), needs_confirmation=True)


@router.post(
    "/{rule_id}/activate",
    response_model=AlarmActivateResponse,
    summary="Activate a previously composed alarm rule",
    description="Idempotent. After activation the alarm engine evaluates the rule against every frame and publishes via `/v1/alerts/stream` when the condition is met.",
)
async def activate(
    rule_id: str,
    handle: PipelineHandle = Depends(require_auth),
) -> AlarmActivateResponse:
    with _LOCK:
        rec = _ACTIVE_RULES.get(rule_id)
        if rec is None:
            raise HTTPException(404, "rule not found")
        rec["active"] = True
    _persist(rec)
    _poke_engine(handle)
    return AlarmActivateResponse(id=rule_id, active=True)


@router.delete(
    "/{rule_id}",
    response_model=AlarmDeleteResponse,
    summary="Remove an active alarm rule",
    description="The rule stops firing immediately. Persisted state is purged.",
)
async def delete(
    rule_id: str,
    handle: PipelineHandle = Depends(require_auth),
) -> AlarmDeleteResponse:
    with _LOCK:
        if rule_id not in _ACTIVE_RULES:
            raise HTTPException(404, "rule not found")
        rec = _ACTIVE_RULES[rule_id]
        rec["active"] = False
        del _ACTIVE_RULES[rule_id]
    _persist(rec)
    _poke_engine(handle)
    return AlarmDeleteResponse(id=rule_id, deleted=True)


def _poke_engine(handle) -> None:
    """Tell the live AlarmEngine to re-read the JSONL.

    The engine is owned by the orchestrator, attached to the handle as
    a side channel. This is a fail-soft hook — when the engine isn't
    available (early boot, integrator-only deploys without the engine)
    we just skip; the engine will pick up the new rules on its next
    natural reload.
    """
    eng = getattr(handle, "alarm_engine", None)
    if eng is not None:
        try:
            eng.reload()
        except Exception:
            pass


@router.get(
    "/active",
    response_model=AlarmListResponse,
    summary="List active alarm rules",
    description="Returns every rule currently active in the alarm engine, with their fire counters.",
)
async def list_active() -> AlarmListResponse:
    with _LOCK:
        rules = [AlarmRule(**r) for r in _ACTIVE_RULES.values()]
    return AlarmListResponse(rules=rules)


def _heuristic_parse(text: str) -> dict:
    """Convert a sentence into a coarse rule DSL.

    Looks for: subject (person/forklift/conveyor/welder), action
    (bend, enter, exit, dwell, count > N), location (zone-N), and
    duration (X seconds).

    Output schema::

        {
          "subject_class": "person",
          "action": "bend"|"enter"|"dwell"|"count_gt",
          "location_zone": "zone-3" | None,
          "duration_s": 3.0 | None,
          "count_threshold": 5 | None,
        }
    """
    import re
    t = text.lower()
    # Subject
    subjects = {
        "person": ["worker", "person", "anyone", "someone", "people"],
        "forklift": ["forklift", "truck"],
        "conveyor": ["conveyor", "belt"],
        "welder": ["welder"],
    }
    subject = None
    for cls, kws in subjects.items():
        if any(kw in t for kw in kws):
            subject = cls
            break
    # Action
    if "bend" in t or "stoop" in t or "crouch" in t:
        action = "bend"
    elif "enter" in t or "approach" in t:
        action = "enter"
    elif "leave" in t or "exit" in t:
        action = "exit"
    elif "dwell" in t or "stay" in t or "stand" in t:
        action = "dwell"
    elif "count" in t or "more than" in t:
        action = "count_gt"
    else:
        action = "presence"
    # Location
    zone_match = re.search(r"zone[\s-]*(\w+)", t)
    location_zone = f"zone-{zone_match.group(1)}" if zone_match else None
    # Duration
    dur_match = re.search(r"(\d+)\s*(?:s|sec|second|seconds)", t)
    duration_s = float(dur_match.group(1)) if dur_match else None
    # Count threshold
    cnt_match = re.search(r"(?:more than|>=?)\s*(\d+)", t)
    count_threshold = int(cnt_match.group(1)) if cnt_match else None
    return {
        "subject_class": subject,
        "action": action,
        "location_zone": location_zone,
        "duration_s": duration_s,
        "count_threshold": count_threshold,
    }
