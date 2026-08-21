"""Corporate actions checked by hand against announcements — Gate 0 check 2.

CLAUDE.md §5's Gate 0 asks for two reconciliations. This is the second:
"Reconcile 5 known corporate-action events by hand."

WHAT "RECONCILES" ACTUALLY MEANS, WHICH TOOK TWO TRIES TO GET RIGHT
-------------------------------------------------------------------
The first version of this check asked "is there a price step at the announced
ex-date?" and treated any step as a failure. That is wrong, because a price
series can be in either of two states and BOTH are correct:

    BACK_ADJUSTED         prior prices have already been scaled, so there is
                          NO step at the ex-date. BBCA's 1:5, BMRI's 1:2,
                          ISAT's 1:4 and DSSA's 1:10 are all like this.
    UNADJUSTED_CONSISTENT the series shows the real ex-date drop, and that
                          drop EQUALS the theoretical factor. WIKA's 2024
                          rights issue is like this: 240 -> 203.91, against a
                          published theoretical ex-rights price of Rp 204.

Both are internally consistent. A price series is entitled to show a rights
issue as a real fall - the share genuinely is worth less afterwards - as long
as anything computing RETURNS adjusts for it, which is what
:mod:`idxbot.spine.corporate_actions` exists to do.

There is only one real failure mode, and it is the one that hides:

    MISDATED              an action-sized step at a date that is not the
                          ex-date. SCCO's 1:4 step lands on 2024-02-01, five
                          weeks before the split took effect and nineteen days
                          before shareholders even approved it.

THE SECOND THING THE FIRST VERSION GOT WRONG
--------------------------------------------
It compared the last TRADED bar before the ex-date with the first traded bar
after. WIKA was suspended for three weeks straight through its rights issue, so
that comparison spanned the suspension and reported a 32% "break" that is
simply the stock reopening lower. The comparison has to run over quoted bars,
stale ones included, because the adjustment applies to the quote whether or not
anyone traded.

And it used a flat 15% threshold, which flagged DSSA's ordinary +16.4% day.
The threshold is the auto-rejection band for that date - a move the exchange
permitted is by definition not evidence of a corporate action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .reference import OutsideCoverage, auto_rejection

#: How close a step must sit to the theoretical factor to count as that action.
FACTOR_TOLERANCE = 0.02

#: A step within this many days of the announced ex-date is on time. IDX's own
#: cum/ex boundary and a vendor's can differ by a day.
DATE_TOLERANCE_DAYS = 3

#: How far from the ex-date an action-sized step can fall and still be a
#: MISDATING of that action rather than a different event. SCCO's error is 36
#: days; a step years away belongs to some other corporate action and blaming
#: it on this one is how the first version of this scan went wrong.
MISDATE_WINDOW_DAYS = 90

STATES = ("back_adjusted", "unadjusted_consistent", "misdated", "unclear")


@dataclass(frozen=True)
class VerifiedAction:
    """A corporate action whose terms and dates came from an announcement."""

    ticker: str
    kind: str
    #: First day trading WITHOUT the entitlement, per the announcement.
    ex_date: pd.Timestamp
    #: Shares after per share before. For a rights issue use ``factor``.
    ratio: Optional[float] = None
    #: For a rights issue: theoretical-ex-price / cum-price, from the
    #: announcement. WIKA published a theoretical price of Rp 204 against a
    #: Rp 240 cum price.
    factor: Optional[float] = None
    announced: Optional[pd.Timestamp] = None
    approved: Optional[pd.Timestamp] = None
    source: str = ""
    note: str = ""

    @property
    def expected_factor(self) -> Optional[float]:
        """What prior prices get multiplied by, if the series is adjusted."""
        if self.factor is not None:
            return float(self.factor)
        if self.ratio:
            return 1.0 / float(self.ratio)
        return None


#: Hand-checked events. Every row cost a search and a read against Indonesian
#: market sources; a row added without that is not a verification.
VERIFIED: List[VerifiedAction] = [
    VerifiedAction("BBCA", "split", pd.Timestamp("2021-10-13"), ratio=5.0,
                   source="new nominal trading from 2021-10-13; recording "
                          "2021-10-14"),
    VerifiedAction("BMRI", "split", pd.Timestamp("2023-04-04"), ratio=2.0,
                   source="new nominal trading from 2023-04-04; recording "
                          "2023-04-05"),
    VerifiedAction("SCCO", "split", pd.Timestamp("2024-03-08"), ratio=4.0,
                   announced=pd.Timestamp("2024-01-15"),
                   approved=pd.Timestamp("2024-02-20"),
                   source="RUPSLB 2024-02-20; last day old nominal "
                          "2024-03-07, first day new nominal 2024-03-08",
                   note="the failing case"),
    VerifiedAction("WIKA", "rights", pd.Timestamp("2024-04-17"),
                   factor=204.0 / 240.0,
                   source="cum-right 2024-04-16, ex-right 2024-04-17 regular "
                          "market; published theoretical price Rp 204",
                   note="suspended either side of the ex-date"),
    VerifiedAction("PYFA", "rights", pd.Timestamp("2024-04-22"),
                   factor=257.0 / 1197.0,
                   announced=pd.Timestamp("2024-04-04"),
                   source="PMHMETD I: 1 old share : 20 HMETD at Rp 100, cum "
                          "2024-04-19, ex 2024-04-22 regular market; 10.70bn "
                          "new shares for Rp 1.07tn",
                   note="extreme dilution - TERP is 21.5% of the cum price, "
                        "the kind of case CLAUDE.md 5 asks for as a fixture. "
                        "The vendor adjusted only the last four cum sessions; "
                        "see repairs.PYFA"),
    VerifiedAction("SINI", "rights", pd.Timestamp("2026-07-09"),
                   factor=246.0 / 365.0,
                   source="PMHMETD: 2 old shares : 3 HMETD at Rp 5,000, DPS "
                          "2026-07-10 so cum 2026-07-08 under T+2; 721.5m new "
                          "shares for ~Rp 3.61tn, PTRO standby buyer",
                   note="the vendor adjusted only the last eight cum "
                        "sessions, four of them a halt; see repairs.SINI"),
    VerifiedAction("DSSA", "split", pd.Timestamp("2024-07-18"), ratio=10.0,
                   source="last day old nominal 2024-07-17, first day new "
                          "nominal 2024-07-18"),
    VerifiedAction("ISAT", "split", pd.Timestamp("2024-10-14"), ratio=4.0,
                   approved=pd.Timestamp("2024-09-24"),
                   source="RUPSLB 2024-09-24; last day old nominal "
                          "2024-10-11, effective 2024-10-14"),
    VerifiedAction("DSSA", "split", pd.Timestamp("2026-04-09"), ratio=25.0,
                   source="last day old nominal 2026-04-08, first day new "
                          "nominal 2026-04-09"),
]


def _band(price: float, day) -> float:
    """The largest single-day move the exchange allowed. Used as the threshold."""
    try:
        up, dn = auto_rejection(price, day)
    except (OutsideCoverage, ValueError):
        return 0.35
    up = abs(up) / price if up < 0 else up
    dn = abs(dn) / price if dn < 0 else dn
    return float(max(up, dn))


def classify(prices: pd.DataFrame, action: VerifiedAction) -> Dict[str, object]:
    """Which of the three consistent-or-not states is this series in?

    Runs over QUOTED bars, stale ones included: a suspension across the
    ex-date does not exempt the quote from being adjusted, and filtering stale
    bars is what made the first version report WIKA as a 32% break.
    """
    d = prices.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["close"] > 0].sort_values("date").reset_index(drop=True)
    ex = pd.Timestamp(action.ex_date).normalize()
    idx = d.index[d["date"] >= ex]
    if not len(idx) or idx[0] == 0:
        return {"state": "unclear", "reason": "no quoted bar at/after ex-date"}
    i = int(idx[0])
    prev, now = float(d["close"].iloc[i - 1]), float(d["close"].iloc[i])
    step = now / prev if prev > 0 else np.nan
    band = _band(prev, d["date"].iloc[i])
    fac = action.expected_factor

    # 1. no step at all, beyond what the exchange allowed -> already adjusted
    if abs(step - 1.0) <= band:
        state, reason = "back_adjusted", (
            f"no step at ex ({step - 1:+.1%} inside the {band:.0%} band)")
    # 2. a step that equals the theoretical factor -> unadjusted but correct
    elif fac and abs(step / fac - 1.0) <= FACTOR_TOLERANCE:
        state, reason = "unadjusted_consistent", (
            f"step {step:.4f} matches the theoretical factor {fac:.4f}")
    else:
        state, reason = "unclear", (
            f"step {step:.4f} matches neither 1.0 nor the factor "
            f"{fac if fac else float('nan'):.4f}")

    # 3. A RIGHTS issue does not have to land on its factor, and demanding
    #    that it does is a category error this check made at first.
    #
    #    A split is mechanical: four shares for one, the price is a quarter,
    #    to the tick. A theoretical ex-rights price is a VALUATION, and the
    #    market is free to disagree with it the moment trading opens. PYFA
    #    opened its 1:20 ex-day 14% above TERP and closed 34% above it, which
    #    is not a data defect - it is the market repricing a share that had
    #    just been diluted twentyfold.
    #
    #    The exchange's own rule is the right test, and this repo already
    #    encodes it: on an ex-date IDX resets the reference price to the
    #    theoretical one and the auto-rejection band applies AROUND THAT. So
    #    the question is whether the first ex price is inside the band around
    #    TERP. PYFA closed at Rp 164 against a TERP of Rp 122 with a 35% ARA
    #    at that price level - inside, and legal. Widening the tolerance
    #    instead would have been the wrong fix twice over: it would have let
    #    a genuinely broken split through as well.
    rights_ok = False
    if fac and action.kind == "rights" and state == "unclear":
        terp = prev * fac
        rband = _band(terp, d["date"].iloc[i])
        if abs(now / terp - 1.0) <= rband:
            rights_ok = True
            state, reason = "unadjusted_consistent", (
                f"first ex price {now:,.0f} is inside the {rband:.0%} band "
                f"around the theoretical ex-rights price {terp:,.0f} "
                f"(cum {prev:,.0f} x {fac:.4f})")
        else:
            reason = (
                f"first ex price {now:,.0f} is outside the {rband:.0%} band "
                f"around the theoretical ex-rights price {terp:,.0f}")

    # 4. Where does the action-sized step actually fall?
    #
    # Scoped to a window around the ex-date on purpose. An action-sized step
    # years away is a DIFFERENT corporate action, not a misdating of this one -
    # the first version of this scan searched the whole history and blamed
    # SCCO's 2024 split for a step in 2019, while missing the real error five
    # weeks away.
    mis = None
    on_time = False
    if fac:
        c = d["close"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            steps = c[1:] / c[:-1]
        near = np.abs(steps / fac - 1.0) <= FACTOR_TOLERANCE
        for j in np.where(near)[0]:
            when = d["date"].iloc[j + 1]
            gap = abs((when - ex).days)
            if gap <= DATE_TOLERANCE_DAYS:
                on_time = True
                break
            if gap <= MISDATE_WINDOW_DAYS and mis is None:
                mis = when
    if on_time:
        # The step is present and correctly dated: the series simply is not
        # back-adjusted for it, which is a legitimate state for a price series.
        state = "unadjusted_consistent"
        reason = (f"action-sized step {fac:.4f} falls on the announced "
                  f"ex-date (within {DATE_TOLERANCE_DAYS} days)")
    elif mis is not None and not rights_ok:
        # A rights issue already confirmed against the band around its own
        # theoretical price is not overturned by a factor-sized step weeks
        # away. It IS still overturned for a split, which is how SCCO's
        # misdating is caught - there the ex-date looks clean and the step
        # sits five weeks earlier.
        state = "misdated"
        reason = (f"an action-sized step ({fac:.4f}) falls on "
                  f"{mis:%Y-%m-%d}, {abs((mis - ex).days)} days from the "
                  f"announced ex-date of {ex:%Y-%m-%d}")
    return {"state": state, "reason": reason, "step": float(step),
            "expected_factor": fac, "band": band,
            "misdated_on": mis, "last_cum": prev, "first_ex": now}


def reconcile(load) -> pd.DataFrame:
    """Run every verified action against the spine. ``load(ticker) -> frame``."""
    rows = []
    for a in VERIFIED:
        try:
            px = load(a.ticker)
        except Exception as exc:                            # noqa: BLE001
            rows.append({"ticker": a.ticker, "kind": a.kind,
                         "ex_date": a.ex_date, "state": "unclear",
                         "reason": f"could not load: {exc}"})
            continue
        r = classify(px, a)
        rows.append({"ticker": a.ticker, "kind": a.kind, "ex_date": a.ex_date,
                     "ratio": a.ratio, **r, "source": a.source})
    return pd.DataFrame(rows)


def summary(R: pd.DataFrame) -> Dict[str, object]:
    """Pass/fail for Gate 0 check 2.

    Consistent means back_adjusted OR unadjusted_consistent - both are correct
    states for a price series. Only a MISDATED action is a failure, and one is
    enough to fail the gate: further checks would refine the rate, not rescue
    the verdict.
    """
    if R.empty:
        return {"required": 5, "checked": 0, "gate_passes": False,
                "verdict": "INCOMPLETE — nothing checked"}
    ok = R["state"].isin(("back_adjusted", "unadjusted_consistent")).sum()
    bad = int((R["state"] == "misdated").sum())
    unclear = int((R["state"] == "unclear").sum())
    checked = len(R)
    passes = bool(checked >= 5 and bad == 0 and unclear == 0)
    return {
        "required": 5, "checked": checked, "consistent": int(ok),
        "misdated": bad, "unclear": unclear, "gate_passes": passes,
        "verdict": ("FAIL — an action is misdated in the data" if bad else
                    "FAIL — an action could not be classified" if unclear else
                    "INCOMPLETE — fewer than 5 events checked"
                    if checked < 5 else "PASS"),
    }
