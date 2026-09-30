"""Sensor-grounded fact-checker.

Parses VLM-emitted text for numeric / categorical claims (distances,
sizes, counts, presence) and cross-references each against the live
sensor data on the current FramePacket. Each claim is tagged with one
of three verdicts: verified, unverifiable, contradicted.

Pure-CPU. Reads FramePacket.detections and FramePacket.spatial; no
model inference. Target latency ≤ 50 ms per 200-token VLM answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.modules.spatial_tools import distance, measure
from app.utils.data_structures import Detection, FramePacket
from app.utils.logger import get_logger

logger = get_logger()


class Verdict(str, Enum):
    VERIFIED = "verified"
    UNVERIFIABLE = "unverifiable"
    CONTRADICTED = "contradicted"

    @property
    def emoji(self) -> str:
        return {"verified": "🟢", "unverifiable": "🟡", "contradicted": "🔴"}[self.value]


@dataclass(slots=True)
class Claim:
    """A single parsed claim from the VLM output."""
    raw: str                       # the substring the parser matched
    span: tuple[int, int]          # (start, end) char offsets in the source
    kind: str                      # "distance" | "count" | "size" | "presence"
    value: float                   # extracted number (m / count / m²)
    unit: str                      # canonical unit ("m" / "count" / "m²")
    subject: Optional[str] = None  # class name the claim refers to, when known


@dataclass(slots=True)
class CheckedClaim:
    """A Claim plus its verification result."""
    claim: Claim
    verdict: Verdict
    sensor_value: Optional[float] = None
    tolerance: float = 0.0
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "raw": self.claim.raw,
            "span": list(self.claim.span),
            "kind": self.claim.kind,
            "value": self.claim.value,
            "unit": self.claim.unit,
            "subject": self.claim.subject,
            "verdict": self.verdict.value,
            "badge": self.verdict.emoji,
            "sensor_value": self.sensor_value,
            "tolerance": self.tolerance,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class FactCheckReport:
    """Output of `verify` — the full set of claims + summary counts."""
    text: str
    claims: list[CheckedClaim] = field(default_factory=list)
    verified: int = 0
    unverifiable: int = 0
    contradicted: int = 0

    @property
    def total(self) -> int:
        return self.verified + self.unverifiable + self.contradicted

    @property
    def verified_pct(self) -> float:
        return (self.verified / self.total * 100.0) if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "claims": [c.to_dict() for c in self.claims],
            "summary": {
                "total": self.total,
                "verified": self.verified,
                "unverifiable": self.unverifiable,
                "contradicted": self.contradicted,
                "verified_pct": self.verified_pct,
            },
        }

    def annotate_inline(self) -> str:
        """Return `text` with badges inserted right after each matched claim."""
        if not self.claims:
            return self.text
        # Process from right to left so character offsets stay valid as
        # we splice badges in.
        out = list(self.text)
        for c in sorted(self.claims, key=lambda c: c.claim.span[0], reverse=True):
            end = c.claim.span[1]
            badge = " " + c.verdict.emoji
            out[end:end] = badge
        return "".join(out)


# ─────────────────────────────────────────────────────────────────
# Claim parsing — single master alternation, single linear scan.
#
# Each named group corresponds to a claim kind; the dispatch table
# below maps the group name to a Claim builder. Single-pass over the
# input means O(n) on text length with no allocations beyond the
# Claim list itself. Patterns are ordered: longer / more-specific
# first, so a "0.5 m wide" never matches both `size` and `distance`.

_NUMBER_WORDS = {
    "zero": 0, "no": 0,
    "one": 1, "a": 1, "an": 1, "single": 1,
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_COUNT_SUBJECT_STOPWORDS = frozenset({
    "and", "or", "but", "the", "of", "in", "on", "at", "to", "from",
    "with", "without", "for", "is", "are", "was", "were", "be", "been",
    "by", "as", "if", "than", "then", "that", "this", "these", "those",
    "more", "less", "other", "another", "such", "same",
})

_COCO80_CLASSES = frozenset({
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "kite",
    "skateboard", "surfboard", "bottle", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "pizza", "donut", "cake", "chair", "couch", "bed", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "toothbrush",
})

_ACTION_VERBS = frozenset({
    "sleeping", "sitting", "standing", "walking", "running", "lying",
    "crouching", "bending", "kneeling", "jumping", "falling", "climbing",
    "reaching", "carrying", "holding", "pushing", "pulling", "lifting",
    "throwing", "catching", "eating", "drinking", "talking", "looking",
    "reading", "writing", "typing", "working", "resting", "stretching",
})

_DETECTOR_KNOWN = _COCO80_CLASSES | {"worker", "forklift", "conveyor", "welder"}

_WORD_RX = re.compile(r"\b([a-z]+)\b")

_CLASS_ALIASES = {
    "worker": "person", "operator": "person", "pedestrian": "person",
    "vehicle": "car", "automobile": "car",
}


def missing_classes_in_query(query: str, detections) -> list[str]:
    detected = {_normalise_class(d.class_name) for d in (detections or [])}
    q_lower = query.lower()
    candidates = _DETECTOR_KNOWN | set(_CLASS_ALIASES.keys())
    missing: list[str] = []
    for cls in candidates:
        if cls in q_lower:
            resolved = _CLASS_ALIASES.get(cls, cls)
            if cls not in detected and resolved not in detected:
                missing.append(cls)
    return sorted(set(missing))

_RX_MASTER = re.compile(
    r"\b(?:"
    r"(?P<size>(?P<size_v>\d+(?:\.\d+)?)\s*(?P<size_u>m|metres?|cm)[\s\-]+"
    r"(?:wide|tall|long|high))"
    r"|"
    r"(?P<distance>(?P<dist_v>\d+(?:\.\d+)?)\s*(?P<dist_u>m|metres?|meters?|cm|mm))"
    r"|"
    r"(?P<count>(?P<cnt_v>\d+|" + "|".join(_NUMBER_WORDS) + r")"
    r"\s+(?P<cnt_subj>[a-z]+s?))"
    r"|"
    r"(?P<presence>(?:there\s+is\s+(?:a|an|no)|the)\s+(?P<pres_subj>[a-z]+))"
    r")\b",
    re.IGNORECASE,
)


def _to_metres(value: float, unit: str) -> float:
    u = unit.lower()
    if u in ("m", "metre", "metres", "meter", "meters"):
        return value
    if u == "cm":
        return value / 100.0
    if u == "mm":
        return value / 1000.0
    return value


def _normalise_class(word: str) -> str:
    w = word.lower()
    return w[:-1] if w.endswith("s") and len(w) > 3 else w


def parse_claims(text: str) -> list[Claim]:
    """Single-pass extraction of every claim. O(n) on len(text)."""
    out: list[Claim] = []
    seen: set[int] = set()
    for m in _RX_MASTER.finditer(text):
        span = m.span()
        raw = m.group(0)
        if m.group("size") is not None:
            v = _to_metres(float(m.group("size_v")), m.group("size_u"))
            out.append(Claim(raw=raw, span=span, kind="size", value=v, unit="m"))
        elif m.group("distance") is not None:
            v = _to_metres(float(m.group("dist_v")), m.group("dist_u"))
            out.append(Claim(raw=raw, span=span, kind="distance", value=v, unit="m"))
        elif m.group("count") is not None:
            n_raw = m.group("cnt_v").lower()
            n = int(n_raw) if n_raw.isdigit() else _NUMBER_WORDS.get(n_raw)
            if n is None:
                continue
            cls = _normalise_class(m.group("cnt_subj"))
            if cls in _COUNT_SUBJECT_STOPWORDS:
                continue
            out.append(Claim(
                raw=raw, span=span,
                kind="count", value=float(n), unit="count", subject=cls,
            ))
        elif m.group("presence") is not None:
            cls = _normalise_class(m.group("pres_subj"))
            if cls in _COUNT_SUBJECT_STOPWORDS:
                continue
            out.append(Claim(
                raw=raw, span=span,
                kind="presence", value=1.0, unit="presence", subject=cls,
            ))
        seen.update(range(span[0], span[1]))

    for m in _WORD_RX.finditer(text.lower()):
        s, e = m.span()
        if any(i in seen for i in range(s, e)):
            continue
        word = m.group(1)
        cls = _normalise_class(word)
        if cls in _COCO80_CLASSES or word in _COCO80_CLASSES:
            out.append(Claim(
                raw=text[s:e], span=(s, e),
                kind="presence", value=1.0, unit="presence", subject=cls,
            ))
            seen.update(range(s, e))
        elif word in _ACTION_VERBS:
            out.append(Claim(
                raw=text[s:e], span=(s, e),
                kind="action", value=0.0, unit="action", subject=word,
            ))
            seen.update(range(s, e))
    return out


# ─────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────

def verify(
    text: str,
    packet: FramePacket,
    distance_tolerance_pct: float = 15.0,
    size_tolerance_pct: float = 25.0,
) -> FactCheckReport:
    """Verify every claim parsed from `text` against `packet` ground truth.

    Tolerances are percentages — distance claims within 15 % of sensor,
    sizes within 25 % (more variation expected because object pose
    affects bbox-derived size).
    """
    claims = parse_claims(text)
    checked: list[CheckedClaim] = []
    counts = {"verified": 0, "unverifiable": 0, "contradicted": 0}

    # Pre-compute a quick class→count map for count claims.
    class_counts: dict[str, int] = {}
    for det in packet.detections:
        class_counts[_normalise_class(det.class_name)] = class_counts.get(
            _normalise_class(det.class_name), 0
        ) + 1

    for c in claims:
        cc = _verify_one(c, packet, class_counts,
                         distance_tolerance_pct, size_tolerance_pct)
        checked.append(cc)
        counts[cc.verdict.value] = counts[cc.verdict.value] + 1

    return FactCheckReport(
        text=text,
        claims=checked,
        verified=counts["verified"],
        unverifiable=counts["unverifiable"],
        contradicted=counts["contradicted"],
    )


def _verify_one(
    c: Claim,
    packet: FramePacket,
    class_counts: dict[str, int],
    dist_tol_pct: float,
    size_tol_pct: float,
) -> CheckedClaim:
    """Cross-reference a single claim against the FramePacket."""
    if c.kind == "count":
        actual = class_counts.get(c.subject or "", 0)
        if actual == 0 and c.value > 0:
            # Either the claim is wrong, or the class isn't in our
            # taxonomy (YOLO doesn't track wrenches). Without YOLO
            # support we can't disambiguate → unverifiable.
            return CheckedClaim(
                claim=c, verdict=Verdict.UNVERIFIABLE,
                rationale=f"class '{c.subject}' not in detection taxonomy",
            )
        verdict = Verdict.VERIFIED if abs(actual - c.value) < 0.5 else Verdict.CONTRADICTED
        return CheckedClaim(
            claim=c, verdict=verdict,
            sensor_value=float(actual), tolerance=0.5,
            rationale=f"claimed {int(c.value)} {c.subject}, sensor counts {actual}",
        )

    if c.kind == "distance":
        # Need a paired-track distance OR a "from camera" distance.
        # The claim parser is not yet smart enough to attach a
        # subject pair, so we use the median single-track distance to
        # the camera as the comparison — coarse but useful.
        if not packet.spatial:
            return CheckedClaim(
                claim=c, verdict=Verdict.UNVERIFIABLE,
                rationale="no spatial data on this frame",
            )
        depths = sorted([s.distance_m for s in packet.spatial if s.distance_m > 0])
        if not depths:
            return CheckedClaim(
                claim=c, verdict=Verdict.UNVERIFIABLE,
                rationale="no reliable depth measurements",
            )
        # Use the closest object as the reference — VLMs typically
        # describe what's most prominent.
        sensor = depths[0]
        tol = abs(sensor) * dist_tol_pct / 100.0
        verdict = (
            Verdict.VERIFIED if abs(sensor - c.value) <= tol else Verdict.CONTRADICTED
        )
        return CheckedClaim(
            claim=c, verdict=verdict,
            sensor_value=sensor, tolerance=tol,
            rationale=f"VLM said {c.value:.2f} m; sensor closest object at {sensor:.2f} m (±{tol:.2f})",
        )

    if c.kind == "size":
        if not packet.spatial:
            return CheckedClaim(
                claim=c, verdict=Verdict.UNVERIFIABLE,
                rationale="no spatial data on this frame",
            )
        # Use the largest bbox-derived size as the comparison.
        sizes = [max(s.size_estimate) for s in packet.spatial if any(s.size_estimate)]
        if not sizes:
            return CheckedClaim(
                claim=c, verdict=Verdict.UNVERIFIABLE,
                rationale="no size estimates on this frame",
            )
        sensor = max(sizes)
        tol = abs(sensor) * size_tol_pct / 100.0
        verdict = (
            Verdict.VERIFIED if abs(sensor - c.value) <= tol else Verdict.CONTRADICTED
        )
        return CheckedClaim(
            claim=c, verdict=verdict,
            sensor_value=sensor, tolerance=tol,
            rationale=f"VLM said {c.value:.2f} m; sensor largest object {sensor:.2f} m (±{tol:.2f})",
        )

    if c.kind == "action":
        return CheckedClaim(
            claim=c, verdict=Verdict.UNVERIFIABLE,
            rationale=f"action '{c.subject}' requires pose/activity recognition",
        )

    if c.kind == "presence":
        actual = class_counts.get(c.subject or "", 0)
        if actual > 0:
            return CheckedClaim(
                claim=c, verdict=Verdict.VERIFIED,
                sensor_value=float(actual),
                rationale=f"sensor confirms {actual} {c.subject}(s) in scene",
            )
        if c.subject in _DETECTOR_KNOWN:
            return CheckedClaim(
                claim=c, verdict=Verdict.CONTRADICTED,
                sensor_value=0.0,
                rationale=f"claimed presence of '{c.subject}', detector sees none",
            )
        return CheckedClaim(
            claim=c, verdict=Verdict.UNVERIFIABLE,
            rationale=f"class '{c.subject}' outside detection taxonomy",
        )

    return CheckedClaim(claim=c, verdict=Verdict.UNVERIFIABLE,
                       rationale=f"unsupported claim kind: {c.kind}")
