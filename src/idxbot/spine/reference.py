"""Point-in-time IDX trading rules: auto-rejection, tick size, lot, halts.

WHY THIS MODULE EXISTS AND WHY IT REFUSES TO GUESS
--------------------------------------------------
IDX's *rules* changed over the sample, not just its prices, and applying
today's rules to older data is a lookahead error that never announces itself:

  A day a stock sat locked at auto-rejection is a day nobody could buy it. A
  backtest that fills at that close is fiction, and it will be a *profitable*
  fiction, because the days you could not buy are exactly the days the price
  ran away from you.

  Tick size sets the half-spread floor. The 2016 schedule is finer than the
  2014 one, so applying today's to 2015 understates the cost of every small-cap
  trade in the sample - again in the flattering direction.

So every lookup here takes a DATE, and every lookup **raises** for a date
before its schedule's coverage begins rather than falling back on the nearest
entry. A silent fallback is precisely the bug this module exists to prevent:
it would return today's answer for 2010 and nothing downstream would notice.

THE HISTORY, WITH SOURCES
-------------------------
Auto rejection has been changed six times in the period covered here, and
twice it was changed *back*, so "the ARB is 7%" is only ever true of a
particular window:

    from 2016-01-04   symmetric, 35 / 25 / 20 by price band
    2020-03-10        ARB -> 10% (COVID), ARA unchanged: the asymmetry begins
    2020-03-13        ARB -> 7%
    2023-06-05        ARB -> 15%   "tahap I" of the normalisation
    2023-09-04        ARB = ARA, symmetric again "tahap II"
    2025-04-08        ARB -> 15%   asymmetric AGAIN (Kep-00003/BEI/04-2025)

That last one matters more than it looks: it lands in the middle of the
2025-2026 broker panel this repo collected, so a study of that panel which
assumes the September 2023 symmetric regime is wrong for everything after
8 April 2025.

COVERAGE IS PER SCHEDULE, BECAUSE THE EVIDENCE IS
--------------------------------------------------
    tick size, lot, max step   from 2005-01-03
    auto rejection, halts      from 2010-01-04

They differ because they were established differently. The pre-2014 five-group
tick ladder could not be fetched from any reachable source, so it was read out
of the prices themselves: on a Rp 10 grid essentially every close divides by
10, and the observed granularity is stable in every year from 2005 to 2013
(200-500: Rp 5, 500-2,000: Rp 10, 2,000-5,000: Rp 25, >= 5,000: Rp 50). 2004 is
too thin to read, so coverage starts where the evidence does.

Auto rejection was read the same way - it truncates the return distribution, so
the band is where the tail stops - but the reading only holds back to 2010.
Calibrated on the documented 2014-2016 regime the truncation lands exactly on
35/25/20, and 2010-2013 reproduces it; 2005-2009 does not, with a materially
fatter upper tail. So the bands stop at 2010 while the ladder goes back to
2005, and the 2010-2013 portion is marked as inferred rather than read.

WHAT IS DELIBERATELY NOT CLAIMED
--------------------------------
Nothing before 2005, and no auto-rejection band before 2010: lookups raise
:class:`OutsideCoverage` rather than falling back. IPO first-day bands, and the
temporary per-stock relaxations IDX occasionally grants, are not modelled -
:func:`known_gaps` lists them so a caller can see what is missing rather than
discovering it in a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

#: The day IDX moved to a 100-share lot and the three-group tick ladder. Kept
#: as the headline date because it is where every schedule is documented, but
#: coverage is now PER SCHEDULE - see :data:`EARLY_START` - because the tick
#: ladder is known further back than the auto-rejection bands are.
COVERAGE_START = pd.Timestamp("2014-01-06")

#: Coverage of the older tick ladder and lot size. Not a regulation date: it is
#: where the EVIDENCE begins. The five-group ladder is observed stable in the
#: price data every year from 2005 to 2013 (200-500: Rp 5, 500-2,000: Rp 10,
#: 2,000-5,000: Rp 25, >= 5,000: Rp 50), and 2004 is too thin to read. The
#: regulation that introduced it was not identified, so nothing is claimed
#: before the data supports it.
EARLY_START = pd.Timestamp("2005-01-03")

#: Coverage of the older auto-rejection ladder, and it is LATER than the tick
#: ladder's on purpose. Auto rejection truncates the return distribution, so it
#: can be read off the data: calibrating on the documented 2014-2016 regime,
#: the truncation lands exactly where the encoded 35/25/20 says it should.
#: Applying the same reading to 2010-2013 gives the same structure. Applying it
#: to 2005-2009 does NOT - the upper tail is materially fatter (0.19% of
#: sub-Rp 200 days above 35%, against 0.02% in 2010-2013) - so that period is
#: left uncovered rather than assumed to match.
ARB_EARLY_START = pd.Timestamp("2010-01-04")

#: Boards whose auto-rejection bands differ from the main ladder.
THIN_BOARDS = ("acceleration", "watchlist")
MAIN_BOARDS = ("main", "development", "new_economy")


class OutsideCoverage(ValueError):
    """Raised for a date this module does not have a verified rule for.

    Deliberately an exception rather than a None or a fallback. A caller that
    silently received today's schedule for 2011 would produce a backtest that
    looks fine and is wrong.
    """


@dataclass(frozen=True)
class Regime:
    """One rule set, valid over a half-open date interval ``[start, end)``."""

    start: pd.Timestamp
    #: None means "still in force as far as this module knows".
    end: Optional[pd.Timestamp]
    #: (upper_price_exclusive, value). The last band uses None for "no ceiling".
    bands: Sequence[Tuple[Optional[float], float]]
    source: str
    note: str = ""

    def value_for(self, price: float) -> float:
        for ceiling, value in self.bands:
            if ceiling is None or float(price) < ceiling:
                return value
        return self.bands[-1][1]


def _pick(regimes: Sequence[Regime], day) -> Regime:
    """The regime covering ``day``, or a refusal.

    Coverage is per SCHEDULE rather than global: the tick ladder is readable in
    the data back to 2005 while the auto-rejection bands are only readable back
    to 2010, and pretending both start on the same day would either throw away
    five years of usable tick history or assert bands nothing supports.
    """
    d = pd.Timestamp(day).normalize()
    for r in regimes:
        if r.start <= d and (r.end is None or d < r.end):
            return r
    earliest = min(r.start for r in regimes)
    raise OutsideCoverage(
        f"{d:%Y-%m-%d} is before this schedule's coverage "
        f"({earliest:%Y-%m-%d}). The rules that applied then are not encoded, "
        f"and returning a later schedule would be a lookahead error. Extend "
        f"the tables or exclude the period.")


# --------------------------------------------------------------------------
# auto rejection
# --------------------------------------------------------------------------
#: Upper limit. Unchanged across the whole covered period - every change since
#: 2020 has been to the LOWER limit, which is why the asymmetry appears and
#: disappears without the ceiling ever moving.
_ARA = [
    Regime(ARB_EARLY_START, None,
           ((200.0, 0.35), (5000.0, 0.25), (None, 0.20)),
           "IDX Peraturan II-A; unchanged through Kep-00003/BEI/04-2025",
           "the ceiling has not moved in the covered period"),
]

_ARB = [
    Regime(ARB_EARLY_START, pd.Timestamp("2020-03-10"),
           ((200.0, 0.35), (5000.0, 0.25), (None, 0.20)),
           "IDX Peraturan II-A; 2010-2013 portion INFERRED from the return "
           "distribution, not read from a regulation",
           "symmetric with ARA. The inference is calibrated on the documented "
           "2014-2016 regime, where the truncation lands exactly on 35/25/20, "
           "and 2010-2013 reproduces it"),
    Regime(pd.Timestamp("2020-03-10"), pd.Timestamp("2020-03-13"),
           ((None, 0.10),), "IDX announcement 2020-03-09",
           "COVID emergency; three trading days only"),
    Regime(pd.Timestamp("2020-03-13"), pd.Timestamp("2023-06-05"),
           ((None, 0.07),), "IDX announcement 2020-03-13",
           "COVID emergency; the long asymmetric period"),
    Regime(pd.Timestamp("2023-06-05"), pd.Timestamp("2023-09-04"),
           ((None, 0.15),), "Kep-00055/BEI/03-2023", "normalisation tahap I"),
    Regime(pd.Timestamp("2023-09-04"), pd.Timestamp("2025-04-08"),
           ((200.0, 0.35), (5000.0, 0.25), (None, 0.20)),
           "Kep-00055/BEI/03-2023", "normalisation tahap II; symmetric again"),
    Regime(pd.Timestamp("2025-04-08"), None,
           ((None, 0.15),), "Kep-00003/BEI/04-2025",
           "asymmetric again; effective the reopening after Nyepi/Eid 2025"),
]

#: Acceleration board and watchlist names trade on a different ladder: a flat
#: one-rupiah band at the very bottom, ten percent above it. Treating them with
#: the main ladder would allow a backtest 35% moves that could not happen.
_THIN = [
    Regime(ARB_EARLY_START, None, ((10.0, -1.0), (None, 0.10)),
           "Kep-00003/BEI/04-2025 and predecessors",
           "a NEGATIVE value means an absolute rupiah band, not a percentage"),
]


def auto_rejection(price: float, day, board: str = "main") -> Tuple[float, float]:
    """``(upper, lower)`` auto-rejection limits as FRACTIONS of the reference.

    A negative value means the band is an absolute rupiah amount rather than a
    percentage - the acceleration board uses a flat Rp1 band below Rp10, where
    a percentage would be meaningless at that granularity.
    """
    b = str(board).lower()
    if b in THIN_BOARDS:
        v = _pick(_THIN, day).value_for(price)
        return (v, v)
    if b not in MAIN_BOARDS:
        raise ValueError(f"unknown board {board!r}; expected one of "
                         f"{MAIN_BOARDS + THIN_BOARDS}")
    return (_pick(_ARA, day).value_for(price),
            _pick(_ARB, day).value_for(price))


def rejection_prices(prev_close: float, day, board: str = "main"
                     ) -> Tuple[float, float]:
    """The actual rupiah prices at which trading auto-rejects.

    The band is taken on the PREVIOUS close, which is the reference IDX uses,
    and the result is rounded to the prevailing tick - an auto-rejection price
    that is not on the tick ladder cannot be quoted.
    """
    up, dn = auto_rejection(prev_close, day, board)
    hi = prev_close + (abs(up) if up < 0 else prev_close * up)
    lo = prev_close - (abs(dn) if dn < 0 else prev_close * dn)
    t_hi, t_lo = tick_size(hi, day), tick_size(max(lo, 1.0), day)
    return (float(int(hi / t_hi) * t_hi),
            float(int(lo / t_lo + 0.999999) * t_lo))


def was_locked(open_, high, low, close, prev_close, day, board: str = "main",
               tol: float = 1e-9) -> Optional[str]:
    """Did this bar sit at an auto-rejection limit? ``'ARA'``, ``'ARB'`` or None.

    A bar that both opened and closed at the ceiling, having never traded
    below it, is a day the stock was bid limit-up all session: an order to buy
    at the close would not have filled. That is the case a backtest must skip,
    and it is why this asks about the OPEN as well as the close.
    """
    hi, lo = rejection_prices(prev_close, day, board)
    if close >= hi - tol and low >= hi - tol:
        return "ARA"
    if close <= lo + tol and high <= lo + tol:
        return "ARB"
    return None


# --------------------------------------------------------------------------
# tick size and the maximum single price step
# --------------------------------------------------------------------------
_TICK = [
    Regime(EARLY_START, pd.Timestamp("2014-01-06"),
           ((200.0, 1.0), (500.0, 5.0), (2000.0, 10.0), (5000.0, 25.0),
            (None, 50.0)),
           "pre-Kep-00071/BEI/11-2013 five-group ladder",
           "the 500-share-lot era; coarser than anything since"),
    Regime(pd.Timestamp("2014-01-06"), pd.Timestamp("2016-05-02"),
           ((500.0, 1.0), (5000.0, 5.0), (None, 25.0)),
           "Kep-00071/BEI/11-2013 eff. 2014-01-06",
           "three groups; two published sources disagree on the Rp 500-5,000 "
           "band (Rp 5 vs Rp 10) and the data settles it at Rp 5 - 97.9% of "
           "closes there divide by 5"),
    Regime(pd.Timestamp("2016-05-02"), None,
           ((200.0, 1.0), (500.0, 2.0), (2000.0, 5.0), (5000.0, 10.0),
            (None, 25.0)),
           "Kep-00023/BEI/04-2016, reaffirmed by Kep-00003/BEI/04-2025",
           "five groups"),
]

#: "Jenjang maksimum perubahan harga" - the largest single step an order may
#: move the price by. A distinct constraint from the tick and from the
#: auto-rejection band, and it binds first on illiquid names.
_MAX_STEP = [
    Regime(EARLY_START, pd.Timestamp("2014-01-06"),
           ((200.0, 10.0), (500.0, 50.0), (2000.0, 100.0), (5000.0, 250.0),
            (None, 500.0)),
           "pre-2014 five-group ladder"),
    Regime(pd.Timestamp("2014-01-06"), pd.Timestamp("2016-05-02"),
           ((500.0, 20.0), (5000.0, 100.0), (None, 500.0)),
           "Kep-00071/BEI/11-2013 eff. 2014-01-06"),
    Regime(pd.Timestamp("2016-05-02"), None,
           ((200.0, 10.0), (500.0, 20.0), (2000.0, 50.0), (5000.0, 100.0),
            (None, 250.0)),
           "Kep-00003/BEI/04-2025"),
]


def tick_size(price: float, day) -> float:
    """Minimum price increment for ``price`` on ``day``."""
    return _pick(_TICK, day).value_for(price)


def max_price_step(price: float, day) -> float:
    """Largest single price step an order may make."""
    return _pick(_MAX_STEP, day).value_for(price)


def half_spread(price: float, day) -> float:
    """Cost floor of crossing the spread once, as a FRACTION of price.

    Half a tick is the optimistic case - it assumes a one-tick-wide book, which
    on a liquid large cap is roughly right and on a small cap is generous. It
    is a floor, not an estimate.
    """
    p = float(price)
    if p <= 0:
        return float("nan")
    return 0.5 * tick_size(p, day) / p


def on_tick(price: float, day) -> bool:
    """Is this a price that could actually have been quoted?"""
    t = tick_size(price, day)
    return abs(round(float(price) / t) * t - float(price)) < 1e-9


def round_to_tick(price: float, day, direction: str = "nearest") -> float:
    """Snap a price onto the prevailing ladder.

    ``direction`` of ``down`` is the honest choice for a sell limit and ``up``
    for a buy limit, because rounding a fill in the favourable direction is a
    small, systematic, entirely invisible way to make a backtest look better.
    """
    t = tick_size(price, day)
    q = float(price) / t
    if direction == "down":
        return float(int(q) * t)
    if direction == "up":
        return float(-int(-q // 1) * t)
    return float(round(q) * t)


# --------------------------------------------------------------------------
# lot size and index-level halts
# --------------------------------------------------------------------------
_LOT = [
    Regime(EARLY_START, pd.Timestamp("2014-01-06"), ((None, 500.0),),
           "Kep-00071/BEI/11-2013 records the change FROM 500",
           "the 500-share-lot era"),
    Regime(pd.Timestamp("2014-01-06"), None, ((None, 100.0),),
           "Kep-00071/BEI/11-2013 eff. 2014-01-06",
           "reduced from 500 shares on this date"),
]


def lot_size(day) -> int:
    """Shares per round lot."""
    return int(_pick(_LOT, day).value_for(0.0))


@dataclass(frozen=True)
class HaltRule:
    start: pd.Timestamp
    end: Optional[pd.Timestamp]
    #: (index fall as a positive fraction, action)
    steps: Sequence[Tuple[float, str]]
    source: str


_HALTS = [
    HaltRule(pd.Timestamp("2020-03-11"), pd.Timestamp("2025-04-08"),
             ((0.05, "halt 30 minutes"),), "IDX 2020 emergency measures"),
    HaltRule(pd.Timestamp("2025-04-08"), None,
             ((0.08, "halt 30 minutes"), (0.15, "halt a further 30 minutes"),
              (0.20, "suspend, possibly for the session")),
             "Kep-00003/BEI/04-2025"),
]


def trading_halt(index_change: float, day) -> Optional[str]:
    """What an IHSG move of ``index_change`` triggers, if anything.

    Returns None before 2020-03-11: index-level halts were introduced then, so
    "no halt" is the correct answer for earlier dates rather than a gap.
    """
    d = pd.Timestamp(day).normalize()
    if d < ARB_EARLY_START:
        raise OutsideCoverage(f"{d:%Y-%m-%d} is before coverage")
    for r in _HALTS:
        if r.start <= d and (r.end is None or d < r.end):
            hit = [a for f, a in r.steps if -float(index_change) >= f]
            return hit[-1] if hit else None
    return None


# --------------------------------------------------------------------------
# provenance and self-audit
# --------------------------------------------------------------------------
def schedule(name: str) -> pd.DataFrame:
    """Every regime for one rule, as a frame - so it can be eyeballed."""
    table = {"ara": _ARA, "arb": _ARB, "thin": _THIN, "tick": _TICK,
             "max_step": _MAX_STEP, "lot": _LOT}.get(str(name).lower())
    if table is None:
        raise ValueError(f"unknown schedule {name!r}")
    return pd.DataFrame([{
        "from": r.start, "to": r.end, "bands": r.bands,
        "source": r.source, "note": r.note} for r in table])


def known_gaps() -> List[str]:
    """What this module does NOT model. Printed by the Gate 0 report.

    Kept as a function rather than a comment because an unmodelled rule that
    nobody can enumerate is indistinguishable from one nobody thought of.
    """
    return [
        f"nothing before {EARLY_START:%Y-%m-%d}; auto-rejection bands only "
        f"from {ARB_EARLY_START:%Y-%m-%d}. Lookups raise OutsideCoverage "
        f"rather than falling back",
        "the 2010-2013 auto-rejection bands are INFERRED from where the return "
        "distribution truncates, calibrated on the documented 2014-2016 "
        "regime, not read from a regulation",
        "IPO first-day auto-rejection bands, which are wider than the "
        "steady-state ladder",
        "per-stock temporary relaxations and suspensions granted by IDX",
        "pre-closing and random-closing microstructure; only the daily bar is "
        "modelled",
        "the full-call-auction and pre-opening sessions",
        "whether a name was on the watchlist on a given day - the thin-board "
        "ladder is available but nothing currently supplies board membership "
        "per ticker-day",
    ]


def audit() -> pd.DataFrame:
    """Check the tables are internally coherent. Run by the tests and Gate 0.

    Catches the two ways a hand-entered schedule goes wrong: a gap between
    regimes, where a date has no rule at all, and an overlap, where it has two.
    """
    rows = []
    for name in ("ara", "arb", "thin", "tick", "max_step", "lot"):
        table = {"ara": _ARA, "arb": _ARB, "thin": _THIN, "tick": _TICK,
                 "max_step": _MAX_STEP, "lot": _LOT}[name]
        ordered = sorted(table, key=lambda r: r.start)
        problems = []
        expected = {"ara": ARB_EARLY_START, "arb": ARB_EARLY_START,
                    "thin": ARB_EARLY_START}.get(name, EARLY_START)
        if ordered[0].start != expected:
            problems.append(f"starts {ordered[0].start:%Y-%m-%d}, "
                            f"not {expected:%Y-%m-%d}")
        for a, b in zip(ordered, ordered[1:]):
            if a.end is None:
                problems.append(f"open-ended regime at {a.start:%Y-%m-%d} "
                                f"followed by another")
            elif a.end != b.start:
                problems.append(f"{'gap' if a.end < b.start else 'overlap'} "
                                f"at {a.end:%Y-%m-%d}")
        if ordered[-1].end is not None:
            problems.append("final regime is closed; nothing covers today")
        rows.append({"schedule": name, "regimes": len(ordered),
                     "ok": not problems, "problems": "; ".join(problems)})
    return pd.DataFrame(rows)
