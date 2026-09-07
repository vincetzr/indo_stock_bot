"""The measured cost of each shipped level, READ from the study that made it.

WHY THIS MODULE EXISTS. The standing instruction's fourth column is *"the
measured cost of each level — what that SL and that TP did to CAGR, drawdown
and win rate in the backtest that produced them"*, and it is the column it
calls non-negotiable. Until now those figures were TYPED into the surfaces that
print them, which means they are true only for as long as somebody remembers to
retype them.

Somebody did not. H62 moved `KEEP_HI` from 0.80 to 0.70; `scripts/stoptest.py`
held a second copy of that constant which did not move; and every level cost the
rule card printed went on describing a rule nobody ships. Nothing failed,
because a number copied out of a study has no link back to the study.

THE LINK IS THE FIX, AND IT HAS TWO HALVES.
  * `stoptest.py` STAMPS the constants it ran with into `reports/stoptest.json`.
  * this module refuses the file when that stamp disagrees with what
    `scripts/rules.py` currently ships, and says which constant moved.

So a stale figure becomes a visible refusal rather than a silent lie. The
fallbacks below are the last known-good values and are labelled as such
wherever they are used; they are never presented as current.

WHAT IS DELIBERATELY NOT HERE. No interpretation, no verdict, no rounding to a
"headline". This module answers "what did the study measure for this arm" and
nothing else — the reading of those numbers belongs where it can be qualified.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

#: Where `scripts/stoptest.py` writes its arms.
DEFAULT_PATH = os.path.join("reports", "stoptest.json")

#: The arms the shipped surfaces quote, by the label `stoptest.ARMS` uses.
BASE_ARM = "BASE: quarterly keep-band only"
SHIPPED_ARM = "SHIPPED: stop 20% + sell HALF at +100%"
DAILY_ARM = "keep band checked DAILY (S2)"
HALF_ARM = "+ sell HALF at +100%"

#: The two FAMILIES the card quotes a curve from, keyed by the level so the
#: caller never has to build a label. The families are what make the shipped
#: constants defensible: H56 chose the stop as the MIDDLE of its family rather
#: than its argmax, and S3's predicted null is that a target's cost is monotone
#: in how tight it is. Neither claim can be printed from a single arm.
STOP_ARMS = {0.10: "+ hard stop 10%", 0.15: "+ hard stop 15%",
             0.20: "+ hard stop 20%", 0.25: "+ hard stop 25%",
             0.30: "+ hard stop 30%"}
TP_ARMS = {0.20: "+ take-profit 20% (S3)", 0.30: "+ take-profit 30% (S3)",
           0.50: "+ take-profit 50% (S3)", 0.75: "+ take-profit 75%",
           1.00: "+ take-profit 100%"}

#: Field names as the WRITER emits them. A first reader assumed `cagr`/`maxdd`
#: and raised; the writer uses `cagr_med`/`dd`. Named here so the assumption is
#: in one place and testable rather than scattered through format strings.
F_CAGR, F_DD, F_EARLY, F_LATE, F_WORST = ("cagr_med", "dd", "early", "late",
                                          "worst_name")

#: Last known-good, from the H63 run on the shipped 0.70/0.60 buffer. Used ONLY
#: when the file is missing or disagrees, and always labelled as a fallback.
#:
#: THIS IS A COPY AND THAT IS THE POINT OF A FALLBACK. What separates it from
#: the typed figures it replaces is that it can never be printed unlabelled:
#: `load()` returns it only alongside a provenance line saying why, and the
#: surfaces print that line. A stale number the reader is told about is a
#: different object from a stale number presented as current.
FALLBACK: Dict[str, Dict[str, float]] = {
    BASE_ARM: {F_CAGR: 0.1346, F_DD: -0.4017, F_EARLY: 0.1271,
               F_LATE: 0.1160, F_WORST: -0.8444},
    SHIPPED_ARM: {F_CAGR: 0.1297, F_DD: -0.3190, F_EARLY: 0.1104,
                  F_LATE: 0.1242, F_WORST: -0.4106},
    DAILY_ARM: {F_CAGR: 0.0429, F_DD: -0.3650, F_EARLY: 0.0386,
                F_LATE: 0.0407, F_WORST: -0.4239},
    HALF_ARM: {F_CAGR: 0.1318, F_DD: -0.3842, F_WORST: -0.8444},
    STOP_ARMS[0.10]: {F_CAGR: 0.1100, F_DD: -0.2921, F_WORST: -0.3000},
    STOP_ARMS[0.15]: {F_CAGR: 0.1201, F_DD: -0.3182, F_WORST: -0.3000},
    STOP_ARMS[0.20]: {F_CAGR: 0.1276, F_DD: -0.3384, F_WORST: -0.4106},
    STOP_ARMS[0.25]: {F_CAGR: 0.1279, F_DD: -0.3598, F_WORST: -0.4201},
    STOP_ARMS[0.30]: {F_CAGR: 0.1272, F_DD: -0.3629, F_WORST: -0.4386},
    TP_ARMS[0.20]: {F_CAGR: 0.0662, F_DD: -0.3915},
    TP_ARMS[0.30]: {F_CAGR: 0.0829, F_DD: -0.3597},
    TP_ARMS[0.50]: {F_CAGR: 0.0968, F_DD: -0.3533},
    TP_ARMS[0.75]: {F_CAGR: 0.1073, F_DD: -0.3699},
    TP_ARMS[1.00]: {F_CAGR: 0.1197, F_DD: -0.3624},
}

#: The IHSG over the arms' own span, from the H63 run — last known-good, like
#: everything above it, and reaching the caller only alongside a provenance
#: line saying the file could not be read.
#:
#: A FRESH FILE THAT LACKS THE FIELD IS A DIFFERENT CASE AND GETS `None`. The
#: writer only started persisting `index_cagr` after H63, so a current file can
#: be perfectly good and still not carry it — and quoting the stored figure
#: beside numbers from a newer run would pair a benchmark with a window it was
#: not measured over, which is A19's error class. Declining to quote it is the
#: honest outcome, so the caller is handed `None` and says so.
FALLBACK_INDEX = 0.0681


def _live_constants() -> Dict[str, float]:
    import sys                                               # noqa: PLC0415
    root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        os.pardir, os.pardir))
    scripts = os.path.join(root, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import rules                                             # noqa: PLC0415
    return {"ENTRY_HI": rules.ENTRY_HI, "ENTRY_VOL": rules.ENTRY_VOL,
            "KEEP_HI": rules.KEEP_HI, "KEEP_VOL": rules.KEEP_VOL}


@dataclass(frozen=True)
class Measured:
    """What the study measured, and whether it measured the rule that ships."""

    arms: Dict[str, Dict[str, float]]
    provenance: str
    index_cagr: Optional[float]
    fresh: bool

    def get(self, arm: str, field: str = F_CAGR) -> float:
        """One field of one arm, or NaN. Never raises — a missing arm is a
        thing the caller has to be able to print around."""
        try:
            return float(self.arms[arm][field])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    def cost(self, arm: str, field: str = F_CAGR) -> float:
        return cost_of(self.arms, arm, field)


def _fallback(why: str) -> Measured:
    return Measured(FALLBACK, why, FALLBACK_INDEX, False)


def load(path: Optional[str] = None) -> Measured:
    """The arms, plus a one-line provenance the caller MUST print.

    Falls back with an explanation whenever the file is absent, unstamped,
    stamped for a different rule, or missing the fields it should carry. Every
    one of those is a case where quoting the file would attach a number to the
    wrong rule, and every one has actually happened.
    """
    p = path or DEFAULT_PATH
    try:
        blob = json.load(open(p))
    except Exception as exc:                                 # noqa: BLE001
        return _fallback(f"stoptest.json unavailable ({exc}) — figures below "
                         f"are the last known-good values, not a fresh "
                         f"measurement")
    if not isinstance(blob, dict):
        return _fallback("stoptest.json carries no rule stamp (old format) so "
                         "it cannot be checked against the live rule — figures "
                         "below are the last known-good values")
    ran = blob.get("rule") or {}
    if not ran:
        return _fallback("stoptest.json has an empty rule stamp — figures "
                         "below are the last known-good values")
    live = _live_constants()
    drift = {k: (ran.get(k), v) for k, v in live.items()
             if ran.get(k) is not None and abs(float(ran[k]) - v) > 1e-9}
    if drift:
        return _fallback(
            "stoptest.json measured a DIFFERENT rule than ships — "
            + ", ".join(f"{k}: ran {a}, ships {b}"
                        for k, (a, b) in drift.items())
            + ". Re-run scripts/stoptest.py; figures below are the last "
              "known-good values")
    arms = {a.get("arm"): a for a in (blob.get("arms") or []) if a.get("arm")}
    need = (BASE_ARM, SHIPPED_ARM)
    missing = [a for a in need if a not in arms]
    if missing:
        return _fallback(f"stoptest.json has no {missing[0]!r} arm — figures "
                         f"below are the last known-good values")
    for a in need:
        for f in (F_CAGR, F_DD):
            if f not in arms[a]:
                return _fallback(f"stoptest.json arms lack {f!r} — figures "
                                 f"below are the last known-good values")
    #  The index is optional because the field post-dates some written files.
    #  A missing benchmark is declined, never invented: `None` reaches the
    #  caller and the caller says so.
    idx = blob.get("index_cagr")
    try:
        idx = float(idx) if idx is not None else None
    except (TypeError, ValueError):
        idx = None
    return Measured(arms, "measured on the live rule "
                          "(reports/stoptest.json)", idx, True)


def cost_of(arms: Dict[str, Dict[str, float]], arm: str,
            field: str = F_CAGR) -> float:
    """`arm` minus BASE on `field`. NEGATIVE means the level COSTS.

    The sign convention is stated because the direction has already flipped
    once: on the pre-H62 buffer the shipped levels showed a +0.76-point CAGR
    gain, and on the shipped buffer they cost 0.49. Prose that says "edge"
    where the number says "cost" is the drift this module exists to end.
    """
    if arm not in arms or BASE_ARM not in arms:
        return float("nan")
    try:
        return float(arms[arm][field]) - float(arms[BASE_ARM][field])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def noun(gap: float) -> str:
    """"edge" or "cost", from the sign. Never typed next to the number."""
    if gap != gap:
        return "effect"
    return "edge" if gap > 0 else "cost"


def family(m: Measured, arms: Dict[float, str],
           field: str = F_CAGR) -> Dict[float, float]:
    """`{level: value}` for one family, skipping arms the file does not hold."""
    out = {}
    for lvl, label in sorted(arms.items()):
        v = m.get(label, field)
        if v == v:
            out[lvl] = v
    return out


def family_costs(m: Measured, arms: Dict[float, str]) -> Dict[float, float]:
    """`{level: points of CAGR the level COSTS}`, positive meaning it costs.

    S3's predicted null is that a target's cost is monotone in how tight it is,
    and the card prints that curve. Printing it from a dict typed into a format
    string is how the curve went on describing the pre-H62 buffer.
    """
    return {lvl: -m.cost(label) for lvl, label in sorted(arms.items())
            if m.get(label) == m.get(label)}


def is_monotone(values: Iterable[float]) -> bool:
    """Non-increasing in the iteration order. The card claims a monotone curve;
    a claim about a shape is checkable, so it is checked rather than typed."""
    v = list(values)
    return all(b <= a + 1e-12 for a, b in zip(v, v[1:]))


def middle(levels: Iterable[float]) -> float:
    """The middle of a family by its LEVELS, never by its results.

    H56 chose `STOP = 0.20` this way and H62 chose the buffer the same way;
    `buffer.middle_cell()` is the same idea on a two-dimensional grid. Defining
    it on the outcomes would be an argmax wearing a different word.
    """
    s = sorted(levels)
    return float(s[(len(s) - 1) // 2]) if s else float("nan")
