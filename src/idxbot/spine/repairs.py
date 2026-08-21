"""Repairs to the price spine, each justified by an announcement.

WHY REPAIRS ARE A SEPARATE, NAMED, TINY REGISTRY
------------------------------------------------
Editing market data is the most dangerous thing this repo does. A repair that
is wrong is worse than the defect it replaces, because the defect is at least
discoverable - a repaired series looks clean by construction.

So every entry here carries the announcement that justifies it, the arithmetic
that determines the factor, and the check that confirms the result. There is no
"clean the data" function and there never should be: repairs are enumerated one
at a time, by hand, or not at all.

THE SCCO CASE, WHICH IS THE WHOLE REASON THIS EXISTS
-----------------------------------------------------
SCCO's 1:4 stock split took effect on **2024-03-08** (last day at the old
nominal 2024-03-07; RUPSLB approval 2024-02-20; announced 2024-01-15 with the
stock trading near Rp 10,000).

The cached series switches basis on **2024-02-01** - five weeks early, and
nineteen days before shareholders approved it.

The direction of the error is settled by the rest of the history. SCCO's median
close runs 3,800 / 4,825 / 8,700 / 9,700 / 9,100 / 9,288 / 10,800 / 9,750 /
8,675 from 2015 to 2023, and 2,190 from 2024. So the series is NOT
back-adjusted: everything before the split is on the old basis and everything
after is on the new one. The window 2024-02-01 to 2024-03-07 is on the NEW
basis when it should still be on the OLD one.

    repair: multiply prices in that window by 4

The confirmation is exact. The last cum bar, 2024-03-07, reads 2,543.75;
repaired it is 10,175, and the first ex bar reads 2,550. 2,550 / 10,175 =
0.2506, which is the 1:4 split to within a tick. Before the repair that same
boundary reads +0.2%, and the split has silently vanished from the series.

**Volume is deliberately NOT repaired.** Share count did not change until
2024-03-08, so February volume is correctly on the old basis - it is only the
price that moved early. That asymmetry is the actual harm: for ~25 sessions
price x volume understates traded value four-fold while both columns look
entirely reasonable on their own.

IT IS NOT ONE CASE. IT IS A SHAPE, AND THE TICK GRID FINDS IT
--------------------------------------------------------------
SCCO looked like a one-off until the same shape turned up twice more. In all
three the vendor back-adjusted only the last few sessions before an ex-date
instead of the whole history, leaving an ISLAND on a basis nothing was ever
traded on, with a fake crash going in and a fake rally coming out.

    SCCO  2024-02-01..2024-03-07  23 sessions   1:4 split
    PYFA  2024-04-16..2024-04-19   4 sessions   1:20 rights at Rp 100
    SINI  2026-06-29..2026-07-08   8 sessions   2:3 rights at Rp 5,000

:func:`idxbot.spine.quality.suspect_islands` now looks for the shape directly -
an off-tick-grid stretch that :func:`~idxbot.spine.quality.level_shifts` also
calls a break - and across 937 tickers and 2.85 million bars it returns exactly
these three and nothing else. SCCO appears only in the unrepaired cache, which
is the repair proving itself: applying it puts every bar back on the tick grid.

WHERE THE FACTORS CAME FROM, AND WHY NOT FROM THE CHART
--------------------------------------------------------
Both rights factors are the announced terms, not a ratio read off the price
move - reading it off the move would be circular, since the move is the thing
being explained. The announcement fixes the factor, and the TICK GRID then gets
an independent vote:

    PYFA  1 old share : 20 new at Rp 100, cum 2024-04-19, ex 2024-04-22.
          TERP = (P + 20x100)/21, so the vendor's factor is TERP/P and the last
          cum bar's adjusted close IS TERP. That reads 122.380951, which pins
          P = Rp 570 - an exact Rp 5 tick. Dividing the window by the resulting
          factor puts all 16 open/high/low/close values on the tick grid
          (1,040 / 1,170 / 940 / 950 / 980 / 750 / 775 / 735 / 815 / 675 / 685
          / 515 / 570) and all 4 volumes on a whole 100-share lot. A factor 2%
          away scores zero on both.

    SINI  2 old shares : 3 new at Rp 5,000, DPS 2026-07-10, so cum 2026-07-08
          on T+2. TERP = (2P + 3x5,000)/5. Solving for the one tick-valued P
          consistent with the block gives P = Rp 10,950 and TERP = Rp 7,380 -
          and Rp 7,380.00 is exactly what sits on 2026-07-08, a bar that was
          not used to fit it. 32 prices land on the grid, 4 volumes on the lot.

Both windows also carry the vendor's volume adjustment, unlike SCCO: their
volumes are not multiples of 100, which no IDX print ever is. So these two
repair BOTH columns and SCCO repairs only price. That difference is not a
style choice - it is what the lot grid says happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Repair:
    """One correction to one ticker over one closed date interval."""

    ticker: str
    start: pd.Timestamp
    end: pd.Timestamp
    #: Multiply open/high/low/close by this over ``[start, end]``.
    price_factor: float = 1.0
    #: Multiply volume by this. Usually 1.0 - see the SCCO note above.
    volume_factor: float = 1.0
    reason: str = ""
    source: str = ""

    def covers(self, dates: pd.Series) -> pd.Series:
        d = pd.to_datetime(dates)
        return (d >= self.start) & (d <= self.end)


REPAIRS: List[Repair] = [
    Repair(
        ticker="SCCO",
        start=pd.Timestamp("2024-02-01"),
        end=pd.Timestamp("2024-03-07"),
        price_factor=4.0,
        volume_factor=1.0,
        reason=("the 1:4 split basis was applied 36 days early. The series is "
                "not back-adjusted, so this window belongs on the OLD basis; "
                "prices here are a quarter of what SCCO actually traded at. "
                "Volume is untouched: share count did not change until "
                "2024-03-08."),
        source=("split announced 2024-01-15, approved at RUPSLB 2024-02-20, "
                "last day old nominal 2024-03-07, first day new nominal "
                "2024-03-08"),
    ),
    Repair(
        ticker="PYFA",
        start=pd.Timestamp("2024-04-16"),
        end=pd.Timestamp("2024-04-19"),
        # vendor factor = TERP/P_cum = ((570 + 20*100)/21) / 570 = 257/1197
        price_factor=1197 / 257,
        volume_factor=257 / 1197,
        reason=("the last four CUM sessions were back-adjusted for the 1:20 "
                "rights issue while the rest of the series was left raw. The "
                "window shows Rp 204-223 where PYFA traded at Rp 570-1,040. "
                "Undoing it restores all 16 prices to the Rp 5 tick grid and "
                "all 4 volumes to whole lots. Volume IS repaired here, unlike "
                "SCCO: the vendor divided it too, which is why those four "
                "figures are not multiples of 100."),
        source=("PMHMETD I: 1 old share : 20 HMETD at Rp 100, cum date "
                "2024-04-19, ex date 2024-04-22 in the regular market; "
                "10.70bn new shares raising Rp 1.07tn"),
    ),
    Repair(
        ticker="SINI",
        start=pd.Timestamp("2026-06-29"),
        end=pd.Timestamp("2026-07-08"),
        # vendor factor = TERP/P_cum = 7,380 / 10,950 = 246/365
        price_factor=365 / 246,
        volume_factor=246 / 365,
        reason=("the last eight CUM sessions were back-adjusted for the 2:3 "
                "rights issue while the rest of the series was left raw. Four "
                "of the eight are a trading halt carried at the adjusted "
                "quote. Undoing it restores all 32 prices to the Rp 25 tick "
                "grid - 9,125 / 9,250 / 9,000 / 9,075 / 9,975 / 10,950 - and "
                "the volumes to whole lots."),
        source=("PMHMETD: every 2 old shares carry 3 HMETD at Rp 5,000, "
                "721.5m new shares for ~Rp 3.61tn, DPS 2026-07-10 so cum "
                "2026-07-08 and ex 2026-07-09 under T+2; PTRO standby buyer"),
    ),
]


@dataclass(frozen=True)
class Suspect:
    """A detected level shift that has NOT been checked against an announcement.

    SCCO proved that a detected shift can be confidently wrong about its own
    date, so an unverified one may not be treated as a corporate action and may
    not be treated as a real price move either. It is quarantined: research can
    exclude the window, and nothing silently assumes it is fine.
    """

    ticker: str
    date: pd.Timestamp
    ratio: float
    note: str = ""


#: Level shifts found by :func:`idxbot.spine.quality.level_shifts` that no
#: announcement has confirmed. Every one is a candidate for the SCCO defect.
#: Moving a row from here to REPAIRS or to verified_actions.VERIFIED requires
#: reading an announcement, not looking at the chart harder.
SUSPECT: List[Suspect] = [
    # PYFA (2024-04-16) and SINI (2026-06-29) were quarantined here because
    # their cause was known and their factor was not. Both are now REPAIRS:
    # the announced ratios were found, and dividing each window by the factor
    # those ratios imply puts every price back on the tick grid and every
    # volume on a whole lot. The grid was the missing evidence - it is what
    # let the factor be confirmed without reading it off the move it explains.
    Suspect("ELTY", pd.Timestamp("2018-06-07"), 10.00,
            "NOT a corporate action. Bakrieland's 10:1 reverse split was "
            "proposed in June 2018 and REJECTED by shareholders - there were "
            "petitions against it, and a 2019 follow-up confirms it was never "
            "completed. The series sits at exactly Rp 500 through 2015-2017 on "
            "near-zero volume and at exactly Rp 50 from 2018-06-07 with real "
            "volume the next day, so this is a dormant quote being re-marked "
            "on resumption. Auto-rejection does not bind on resumption after a "
            "long suspension, so the -90% is legal and real. Stays quarantined "
            "because the window is uninformative, not because the cause is "
            "unknown."),
]

#: How wide a quarantine to place around a suspect shift. SCCO's error spanned
#: 36 days, so a window narrower than that would have missed it.
SUSPECT_WINDOW_DAYS = 45


def suspect_mask(df: pd.DataFrame, ticker: str) -> pd.Series:
    """True on bars near an unverified level shift for this ticker."""
    d = pd.to_datetime(df["date"]) if "date" in df else pd.Series(dtype="datetime64[ns]")
    out = pd.Series(False, index=df.index)
    t = str(ticker).upper().replace(".JK", "")
    w = pd.Timedelta(days=SUSPECT_WINDOW_DAYS)
    for s in SUSPECT:
        if s.ticker == t:
            out |= (d >= s.date - w) & (d <= s.date + w)
    return out.rename("suspect")


def suspects_for(ticker: str) -> List[Suspect]:
    t = str(ticker).upper().replace(".JK", "")
    return [s for s in SUSPECT if s.ticker == t]


def repairs_for(ticker: str) -> List[Repair]:
    t = str(ticker).upper().replace(".JK", "")
    return [r for r in REPAIRS if r.ticker == t]


def apply_repairs(df: pd.DataFrame, ticker: str,
                  price_columns=("open", "high", "low", "close", "adj_close"),
                  volume_column: str = "volume") -> pd.DataFrame:
    """Apply every registered repair for ``ticker``. A no-op when none apply.

    Returns a copy with a ``repaired`` boolean column, so a repaired bar is
    never silently indistinguishable from an original one. Anything reporting
    on this data can and should say how much of it was touched.
    """
    d = df.copy()
    if d.empty:
        return d
    d["date"] = pd.to_datetime(d["date"])
    if "repaired" not in d:
        d["repaired"] = False
    for r in repairs_for(ticker):
        # IDEMPOTENT BY CONSTRUCTION. Applying a x4 repair twice gives x16 and
        # the result still looks like a price series - it is the single most
        # dangerous thing this module could do, and it happened the first time
        # Gate 0 wired repairs into its loader and then re-repaired inside
        # verify(). The flag is the guard, not a label.
        m = r.covers(d["date"]) & ~d["repaired"].astype(bool)
        if not m.any():
            continue
        for c in price_columns:
            if c in d:
                d.loc[m, c] = pd.to_numeric(d.loc[m, c],
                                            errors="coerce") * r.price_factor
        if volume_column in d and r.volume_factor != 1.0:
            d.loc[m, volume_column] = pd.to_numeric(
                d.loc[m, volume_column], errors="coerce") * r.volume_factor
        d.loc[m, "repaired"] = True
    return d


def summary() -> pd.DataFrame:
    """Every repair, for a report or an audit."""
    return pd.DataFrame([{
        "ticker": r.ticker, "from": r.start, "to": r.end,
        "price_factor": r.price_factor, "volume_factor": r.volume_factor,
        "reason": r.reason, "source": r.source} for r in REPAIRS])


def verify(load) -> pd.DataFrame:
    """Confirm each repair produces the split step the announcement implies.

    ``load(ticker) -> frame``. A repair that does not restore the expected
    boundary is a repair that is wrong, and this is what says so.
    """
    rows = []
    for r in REPAIRS:
        try:
            raw = load(r.ticker)
        except Exception as exc:                            # noqa: BLE001
            rows.append({"ticker": r.ticker, "ok": False,
                         "detail": f"could not load: {exc}"})
            continue
        fixed = apply_repairs(raw, r.ticker)
        fixed["date"] = pd.to_datetime(fixed["date"])
        fixed = fixed[fixed["close"] > 0].sort_values("date")
        after = fixed[fixed["date"] > r.end]
        before = fixed[fixed["date"] <= r.end]
        if after.empty or before.empty:
            rows.append({"ticker": r.ticker, "ok": False,
                         "detail": "window sits at the edge of the series"})
            continue
        last_cum = float(before["close"].iloc[-1])
        first_ex = float(after["close"].iloc[0])
        step = first_ex / last_cum if last_cum > 0 else np.nan
        rows.append({
            "ticker": r.ticker, "last_cum": last_cum, "first_ex": first_ex,
            "step": step, "expected": 1.0 / r.price_factor,
            "ok": bool(np.isfinite(step)
                       and abs(step * r.price_factor - 1.0) <= 0.02),
            "detail": (f"repaired boundary {last_cum:,.0f} -> {first_ex:,.0f} "
                       f"= x{step:.4f}, expected x{1 / r.price_factor:.4f}")})
    return pd.DataFrame(rows)
