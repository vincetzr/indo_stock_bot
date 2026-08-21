"""Corporate actions checked by hand against announcements — Gate 0 check 2.

CLAUDE.md §5's Gate 0 asks for two reconciliations. This is the second:
"Reconcile 5 known corporate-action events by hand."

WHY IT IS A SEPARATE FILE FROM THE DETECTOR
-------------------------------------------
:mod:`idxbot.spine.quality` finds level shifts in the PRICE SERIES. That tells
you where the data jumps, and says nothing about whether a corporate action
happened. This file records what the announcements actually say, so the two can
be compared - and the comparison is the check.

THE FIRST EVENT CHECKED FAILED, AND IT MATTERS
----------------------------------------------
SCCO's 1:4 stock split is real and well documented:

    2024-01-15  announced; the stock hit ARA on the news, trading near Rp 10,000
    2024-02-20  approved at the RUPSLB (extraordinary shareholders' meeting)
    2024-03-07  LAST day of trading at the old nominal, regular market
    2024-03-08  FIRST day at the new nominal

The cached price series steps 4x down on **2024-02-01** - nineteen days before
shareholders approved the split and thirty-six days before it took effect. The
ratio ``adj_close / close`` is a constant 0.8825 straight through, so this is
not a half-applied adjustment: the source has simply placed the split on the
wrong date.

**Consequence: for roughly 25 trading days, 2024-02-01 to 2024-03-07, SCCO's
cached close is a quarter of the price at which the stock actually traded.**
Any study touching SCCO in that window is wrong, and the error is invisible -
the series is smooth, internally consistent, and passes every structural check
in this repo.

WHAT THIS IMPLIES FOR THE OTHER TEN DETECTED SHIFTS
---------------------------------------------------
They are UNVERIFIED. One was checked and one failed. A 1-in-1 failure rate on a
sample of one is not an estimate of anything, but it is emphatically not
evidence that the rest are fine, and the honest position is that no detected
level shift in this repo may be treated as correctly dated until it has been
checked against an announcement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class VerifiedAction:
    """A corporate action whose terms and dates came from an announcement."""

    ticker: str
    kind: str
    #: First day trading WITHOUT the entitlement, per the announcement.
    ex_date: pd.Timestamp
    ratio: float
    announced: Optional[pd.Timestamp] = None
    approved: Optional[pd.Timestamp] = None
    source: str = ""
    #: Where the cached price series actually steps, if it does. None = not
    #: yet compared.
    observed_break: Optional[pd.Timestamp] = None
    note: str = ""

    @property
    def date_error_days(self) -> Optional[int]:
        if self.observed_break is None:
            return None
        return int((pd.Timestamp(self.observed_break)
                    - pd.Timestamp(self.ex_date)).days)

    @property
    def reconciles(self) -> Optional[bool]:
        """True when the data's break matches the announced ex-date."""
        e = self.date_error_days
        return None if e is None else abs(e) <= 3


#: Hand-checked events. Small on purpose: every row here cost a search and a
#: read, and a row added without that is not a verification.
VERIFIED: List[VerifiedAction] = [
    VerifiedAction(
        ticker="SCCO", kind="split", ratio=4.0,
        ex_date=pd.Timestamp("2024-03-08"),
        announced=pd.Timestamp("2024-01-15"),
        approved=pd.Timestamp("2024-02-20"),
        observed_break=pd.Timestamp("2024-02-01"),
        source="RUPSLB 2024-02-20; last day old nominal 2024-03-07, first day "
               "new nominal 2024-03-08 (multiple Indonesian market sources)",
        note="FAILS. Cached series steps 4x on 2024-02-01, 19 days before "
             "shareholder approval. Close is 1/4 of the traded price for "
             "~25 sessions. adj_close/close is constant at 0.8825 throughout, "
             "so the adjustment is internally consistent and simply misdated."),
]


def reconciliation() -> pd.DataFrame:
    """The Gate 0 check-2 table: announced date vs where the data jumps."""
    return pd.DataFrame([{
        "ticker": v.ticker, "kind": v.kind, "ratio": v.ratio,
        "announced_ex": v.ex_date, "observed_break": v.observed_break,
        "error_days": v.date_error_days, "reconciles": v.reconciles,
        "source": v.source, "note": v.note} for v in VERIFIED])


def summary() -> Dict[str, object]:
    """Pass/fail for Gate 0 check 2, including whether enough were checked.

    §5 asks for FIVE. Checking one and passing it would not satisfy the gate;
    checking one and failing it does not satisfy it either, but it does settle
    the outcome - a gate with a known failure in it has failed regardless of
    how many further cases would have passed.
    """
    R = reconciliation()
    checked = int(R["reconciles"].notna().sum())
    failed = int((R["reconciles"] == False).sum())        # noqa: E712
    return {
        "required": 5,
        "checked": checked,
        "passed": int((R["reconciles"] == True).sum()),   # noqa: E712
        "failed": failed,
        "enough_checked": checked >= 5,
        # A single confirmed failure fails the gate. More checks would refine
        # the rate, not rescue the verdict.
        "gate_passes": bool(checked >= 5 and failed == 0),
        "verdict": ("FAIL — a verified event is misdated in the data"
                    if failed else
                    "INCOMPLETE — fewer than 5 events checked"
                    if checked < 5 else "PASS"),
    }
