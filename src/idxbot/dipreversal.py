"""Intraday reversal rules — buying capitulation, out by the close.

The validated rule is ``capitulation_*``. Read that one; the other two are kept
only because their failure is instructive.

**Capitulation gap reversal.** A stock that opens ≥10% below yesterday's close,
*after already falling for a month*, on a morning *the whole index gaps down*.
That is exhausted forced selling — margin calls and redemptions completing — and
it is what bounces.

**The 10% threshold is the exchange's auto-rejection floor, not a fitted
parameter.** That distinction decides how much the rule can be trusted, and the
data is unambiguous about it. Sliced finely, the hit rate does not slope, it
steps:

    gap in (-10.0%, -9.5%]     n=188    56.9%
    gap in (-10.5%, -10.0%]    n= 51    86.3%

A 29-point jump across half a percent. And the opens are not scattered inside
those buckets: **81% of the -10% cohort opens within 0.4% of exactly -10%**, and
92% of the -15% cohort within 0.4% of exactly -15%. These stocks opened *on the
limit price*.

The mechanism follows directly. A stock that opens at or through auto-rejection
had its sell queue clear at the floor — the sellers who wanted out are out. One
that gaps -9% never reached the limit, so the supply is still arriving. Against
the -8% to -10% cohort the difference is **+19.1 points, z = 7.67**.

This is why the threshold is a step rather than a slope, and why it is a
microstructure fact rather than a number found by searching.

    when    open <= 91% of yesterday's close
      and   the stock's prior 20-day return is negative
      and   IHSG also gapped down this morning
    buy     at the open
    target  +5%          stop -20%          exit the close, unconditionally

    n = 544 over 25 years   84.6% made >=5%   95% CI [81.5%, 87.6%]
    walk-forward, each year scored on prior years only:
                            86.2% made >=5%   95% CI [83.0%, 89.5%]
    +3.39%/trade, stopped out 0.9%, ~21 trades/year across the exchange

**The direction was originally backwards, and that is worth knowing.** The first
version of this module required the index to be *up* and the stock's trend to be
*up* — "idiosyncratic panic that reverts". That came from 25 hourly trades. On
761,458 daily sessions every one of those filters inverts:

                         made >=5%      n
      prior 20d UP           68.2%    594
      prior 20d DOWN         82.2%    837
      index gapped UP        72.0%    529
      index gapped DOWN      79.8%    901

A lone name collapsing into a rising market is more often the *start* of
something. It is the market-wide flush that pays.

**Why this one is measurable over 25 years and the others are not.** It enters
at the open, so there is no fill to locate in time and no ordering question
about it, and the outcome — did the session high reach open x1.05 — is exact on
a daily bar. Only 0.9% of qualifying sessions touch both barriers. The dip rules
below enter on a *limit*, which has to be located in the tape, and that is why
they needed hourly data and never got past a few hundred trades.

**Caveats that must travel with the number.** Roughly 250 configurations were
compared across this investigation. 2025 measured 65% on 68 trades — the one
weak year, and the most recent complete one. The rule only fires in market-wide
selloffs, so its trades cluster exactly when the rest of a portfolio is also
falling; it is long-only capitulation buying and should be sized for that.

---

The two older variants, retained for reference and *not* recommended:

  * ``gap_*`` — deep gap down while the index is UP. 84% on n=25; the same
    setup measures far worse once the sample grows. Superseded.
  * ``simulate_session`` / ``DIP_*`` — a -10% intraday limit while the index is
    up, on hourly bars: 138 trades, 63.0% made >=5%, +1.25%/trade. Positive but
    thin, and built on the direction the large sample later reversed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Measured parameters. Changing any of these invalidates the numbers above.
DIP_PCT = 0.10           # limit depth below the session open
TARGET_PCT = 0.05        # take profit from the fill
STOP_PCT = 0.20          # stop from the fill; wide, because the entry is a knife
MAX_ENTRY_BAR = 2        # hourly bars: only the first three hours qualify
MIN_INDEX_RETURN = 0.005  # index must be up this much at the moment of fill
MIN_TREND = 0.0          # prior 20-session mean open-to-close must be positive

# --- gap variant -----------------------------------------------------------
# Measured on 168,504 liquid sessions, full IDX, 2023-07 to 2026-08 (hourly).
#
#   gap >=15%, index up >0.5%   n= 25   84.0% made +5%   +3.80%/trade
#   gap >=15%, index up >0.3%   n= 32   81.2% made +5%   +3.66%/trade
#   gap >=10%, index up >0.5%   n= 96   71.9% made +5%   +3.19%/trade
#
# READ THE SAMPLE SIZE BEFORE THE HIT RATE. At n=25 the 95% confidence interval
# on 84% is [70%, 98%] - the data cannot distinguish 84% from 71%. Roughly 230
# configurations were compared across this investigation, and the point estimate
# fell from 84% to 78.6% once the universe was widened, then returned to 84%
# only after an illiquidity filter cut the sample back to 25. No configuration
# tested has a confidence-interval LOWER bound clearing 80%.
#
# The honest reading: the deep-gap reversal is a strong effect with a point
# estimate at the 80% mark and error bars far too wide to bank on. The 10% gap
# threshold, at n=96 and 71.9%, is the version with enough observations to
# deserve confidence, and it does not reach 80%.
GAP_MIN = 0.15           # how far below yesterday's close the open must be
GAP_INDEX_MIN = 0.005    # index return at the open
GAP_TARGET = 0.05
GAP_STOP = 0.20          # measured irrelevant - almost nothing stops out

# --- capitulation variant: the validated rule -------------------------------
# The hourly figures above came from 25 trades and pointed the wrong way. On
# 761,458 liquid daily sessions across 25 years the filters INVERT:
#
#                          made >=5%      n
#   prior 20d UP              68.2%     594
#   prior 20d DOWN            82.2%     837
#   index gapped UP           72.0%     529
#   index gapped DOWN         79.8%     901
#
# The story is capitulation, not idiosyncratic panic. A stock already falling
# for a month, gapping down >=10% on a morning the whole market gaps down, is
# exhausted forced selling - margin calls and redemptions completing - and that
# is what bounces. A lone name dropping into a rising market is more often the
# start of something.
#
# Measured, entry at the open, exit at the close:
#
#   n = 544    84.6% made >=5%    95% CI [81.5%, 87.6%]    +3.39%/trade
#   walk-forward, each year scored on prior years only:
#              86.2% made >=5%    95% CI [83.0%, 89.5%]    n = 429
#
# Every input is knowable at 09:00 - the two gaps are open-vs-prior-close and
# the 20-day return is lagged - and the outcome is high/open >= 1.05, which is
# exact on a daily bar. Only 0.9% of trades break the -20% stop, so the one
# thing daily bars cannot resolve (which barrier came first) is near-moot here.
# That is why this rule can be measured over 25 years while the intraday dip
# rules could not.
#
# Caveats that travel with it: ~250 configurations were compared across this
# investigation; 2025 was a weak year (65% on 68 trades) against 8 of 9 years
# at or above 75%; and it fires about 21 times a year across the whole exchange.
CAP_GAP_MIN = 0.10       # open at least this far below yesterday's close
CAP_TARGET = 0.05
CAP_STOP = 0.20


@dataclass
class DipSignal:
    ticker: str
    session_open: float
    limit_price: float
    target_price: float
    stop_price: float
    index_return: float = np.nan
    prior_trend: float = np.nan
    filled: bool = False
    reasons: List[str] = None

    def as_row(self) -> Dict[str, object]:
        return {
            "ticker": self.ticker,
            "session_open": self.session_open,
            "limit": self.limit_price,
            "target": self.target_price,
            "stop": self.stop_price,
            "index_return": self.index_return,
            "prior_trend": self.prior_trend,
            "filled": self.filled,
        }


def levels(session_open: float) -> Dict[str, float]:
    """The three prices this rule needs, from the session open alone."""
    limit = session_open * (1 - DIP_PCT)
    return {
        "limit": limit,
        "target": limit * (1 + TARGET_PCT),
        "stop": limit * (1 - STOP_PCT),
    }


def qualifies(index_return: float, prior_trend: float,
              entry_bar: Optional[int] = None) -> tuple:
    """Do the non-price conditions hold? Returns ``(ok, reasons)``.

    Separated from the price logic so the same test governs both the backtest
    and the live scanner - a rule that is checked differently in the two places
    is not the rule that was measured.
    """
    reasons: List[str] = []
    if not np.isfinite(index_return) or index_return <= MIN_INDEX_RETURN:
        reasons.append(f"index not up >{MIN_INDEX_RETURN:.1%} at fill "
                       f"({index_return:+.2%})" if np.isfinite(index_return)
                       else "index return unknown")
    if not np.isfinite(prior_trend) or prior_trend <= MIN_TREND:
        reasons.append("prior 20-session trend is not positive")
    if entry_bar is not None and entry_bar > MAX_ENTRY_BAR:
        reasons.append(f"dip arrived too late in the session (bar {entry_bar})")
    return (not reasons), reasons


def simulate_session(highs, lows, closes, opens_first: float,
                     index_returns=None, prior_trend: float = np.nan) -> Optional[dict]:
    """Resolve one session. Returns None when the rule does not trade it.

    Ordering is enforced the same way as everywhere else in this repo: the fill
    bar is located first, and only bars *strictly after* it can resolve the
    outcome. Within a bar the stop is checked before the target, because the
    intraday order is unknowable and the pessimistic reading is the honest one.
    """
    if opens_first <= 0 or not np.isfinite(opens_first) or len(highs) < 2:
        return None
    lv = levels(opens_first)
    fills = np.where(np.asarray(lows) <= lv["limit"])[0]
    if not len(fills):
        return None
    i = int(fills[0])
    if i >= len(closes) - 1 or i > MAX_ENTRY_BAR:
        return None

    index_return = np.nan
    if index_returns is not None and i < len(index_returns):
        index_return = float(index_returns[i])
    ok, reasons = qualifies(index_return, prior_trend, entry_bar=i)
    if not ok:
        return None

    H = np.asarray(highs)[i + 1:]
    L = np.asarray(lows)[i + 1:]
    C = np.asarray(closes)[i + 1:]
    for k in range(len(H)):
        if L[k] <= lv["stop"]:
            return {"outcome": "stop", "ret": -STOP_PCT, "bars": k + 1, "entry": lv["limit"]}
        if H[k] >= lv["target"]:
            return {"outcome": "target", "ret": TARGET_PCT, "bars": k + 1, "entry": lv["limit"]}
    return {"outcome": "close", "ret": float(C[-1] / lv["limit"] - 1),
            "bars": len(C), "entry": lv["limit"]}


def summarise(returns, cost_pct: float = 0.004) -> Dict[str, float]:
    r = np.asarray([x for x in returns if x is not None and np.isfinite(x)], dtype=float)
    if not len(r):
        return {}
    net = r - cost_pct
    won, lost = net[net > 0], net[net <= 0]
    return {
        "trades": int(len(net)),
        "win_rate": float((net > 0).mean()),
        "expectancy": float(net.mean()),
        "avg_win": float(won.mean()) if len(won) else np.nan,
        "avg_loss": float(lost.mean()) if len(lost) else np.nan,
        "profit_factor": (float(won.sum() / abs(lost.sum()))
                          if len(lost) and lost.sum() != 0 else np.inf),
    }


def render_plan(ticker: str, session_open: float, index_return: float = np.nan,
                prior_trend: float = np.nan, width: int = 78) -> str:
    lv = levels(session_open)
    ok, reasons = qualifies(index_return, prior_trend)
    line = "=" * width
    out = [line, f" {ticker}  -  INTRADAY DIP REVERSAL", line,
           f" session open   {session_open:,.0f}", ""]
    out.append(" ORDERS")
    out.append(f"   buy limit    {lv['limit']:,.0f}   (-{DIP_PCT:.0%} from the open)")
    out.append(f"   target       {lv['target']:,.0f}   (+{TARGET_PCT:.0%} from fill)")
    out.append(f"   stop         {lv['stop']:,.0f}   (-{STOP_PCT:.0%} from fill)")
    out.append(f"   valid until  3 hours after the open, then cancel")
    out.append(f"   hard exit    the close - never hold this overnight")
    out.append("")
    out.append(" CONDITIONS AT FILL")
    if ok:
        out.append("   all conditions met")
    else:
        for r in reasons:
            out.append(f"   ! {r}")
        out.append("   -> do NOT take the fill")
    out.append("")
    out.append(" MEASURED: 138 trades, 63.0% reach +5%, +1.25%/trade, PF 1.71")
    out.append(" ~140 configurations were compared to find this, on 138 trades.")
    out.append(" It does NOT reach an 80% win rate and no variant does. Small size.")
    return "\n".join(out + [line])


# ---------------------------------------------------------------------------
# Gap variant
# ---------------------------------------------------------------------------

def gap_levels(session_open: float) -> Dict[str, float]:
    """Entry is the open itself, so the levels hang off it directly."""
    return {
        "entry": session_open,
        "target": session_open * (1 + GAP_TARGET),
        "stop": session_open * (1 - GAP_STOP),
    }


def gap_qualifies(gap: float, index_return: float, prior_trend: float) -> tuple:
    """``gap`` is today's open over yesterday's close, so it is negative here.

    Everything needed is known at 09:00, before the first trade - which is what
    makes this rule executable at the open rather than a post-hoc observation.
    """
    reasons: List[str] = []
    if not np.isfinite(gap) or gap > -GAP_MIN:
        reasons.append(f"gap {gap:+.1%} is not below -{GAP_MIN:.0%}"
                       if np.isfinite(gap) else "no prior close to measure a gap")
    if not np.isfinite(index_return) or index_return <= GAP_INDEX_MIN:
        reasons.append(f"index not up >{GAP_INDEX_MIN:.1%} at the open"
                       if np.isfinite(index_return) else "index return unknown")
    if not np.isfinite(prior_trend) or prior_trend <= MIN_TREND:
        reasons.append("prior 20-session trend is not positive")
    return (not reasons), reasons


def simulate_gap_session(highs, lows, closes, session_open: float, gap: float,
                         index_return: float, prior_trend: float) -> Optional[dict]:
    """Resolve one gap-down session. Entry is the open; bar 0 is the entry bar."""
    ok, _ = gap_qualifies(gap, index_return, prior_trend)
    if not ok or session_open <= 0 or len(highs) < 2:
        return None
    lv = gap_levels(session_open)
    H, L, C = np.asarray(highs)[1:], np.asarray(lows)[1:], np.asarray(closes)[1:]
    for k in range(len(H)):
        if L[k] <= lv["stop"]:
            return {"outcome": "stop", "ret": -GAP_STOP, "bars": k + 1}
        if H[k] >= lv["target"]:
            return {"outcome": "target", "ret": GAP_TARGET, "bars": k + 1}
    return {"outcome": "close", "ret": float(C[-1] / session_open - 1),
            "bars": len(C)}


def render_gap_plan(ticker: str, session_open: float, gap: float,
                    index_return: float = np.nan, prior_trend: float = np.nan,
                    width: int = 78) -> str:
    lv = gap_levels(session_open)
    ok, reasons = gap_qualifies(gap, index_return, prior_trend)
    line = "=" * width
    out = [line, f" {ticker}  -  GAP-DOWN REVERSAL", line,
           f" opens {session_open:,.0f}   gap {gap:+.1%} vs yesterday's close", ""]
    out.append(" ORDERS")
    out.append(f"   buy          {lv['entry']:,.0f}   at the open")
    out.append(f"   target       {lv['target']:,.0f}   (+{GAP_TARGET:.0%})")
    out.append(f"   stop         {lv['stop']:,.0f}   (-{GAP_STOP:.0%}); rarely reached")
    out.append(f"   hard exit    the close - never hold this overnight")
    out.append("")
    if ok:
        out.append(" all conditions met at the open")
    else:
        for r in reasons:
            out.append(f"   ! {r}")
        out.append("   -> no trade")
    out.append("")
    out.append(" MEASURED: n=25, 84.0% made +5%, +3.80%/trade")
    out.append(" 95% CI on that hit rate is [70%, 98%]. Twenty-five trades cannot")
    out.append(" tell 84% from 71%, and ~230 configurations were compared to find")
    out.append(" it. The n=96 version (gap >10%) measures 71.9%. Size accordingly.")
    return "\n".join(out + [line])


# ---------------------------------------------------------------------------
# Capitulation gap reversal - the validated rule
# ---------------------------------------------------------------------------

def capitulation_qualifies(gap: float, index_gap: float,
                           prior_20d_return: float) -> tuple:
    """All three are knowable at 09:00, before the opening auction clears.

    ``gap`` and ``index_gap`` are today's open over yesterday's close, so both
    are negative for a qualifying setup. ``prior_20d_return`` is lagged by a
    day and must also be negative: the rule wants a stock that was already
    falling before this morning's gap, because that is what makes the gap
    capitulation rather than the beginning of a decline.
    """
    reasons: List[str] = []
    if not np.isfinite(gap) or gap > -CAP_GAP_MIN:
        reasons.append(f"gap {gap:+.1%} is not below -{CAP_GAP_MIN:.0%}"
                       if np.isfinite(gap) else "no prior close to measure a gap")
    if not np.isfinite(prior_20d_return) or prior_20d_return >= 0:
        reasons.append("stock was not already falling over the prior 20 sessions")
    if not np.isfinite(index_gap) or index_gap >= 0:
        reasons.append("index did not gap down - this is not market-wide capitulation")
    return (not reasons), reasons


def capitulation_levels(session_open: float) -> Dict[str, float]:
    return {"entry": session_open,
            "target": session_open * (1 + CAP_TARGET),
            "stop": session_open * (1 - CAP_STOP)}


def simulate_capitulation(session_high: float, session_low: float,
                          session_close: float, session_open: float,
                          gap: float, index_gap: float,
                          prior_20d_return: float) -> Optional[dict]:
    """Resolve one session from daily OHLC alone.

    Legitimate on daily bars precisely because the entry is the open - there is
    no fill to locate and therefore no ordering question about it. The residual
    ambiguity is whether a session that touched both barriers hit the stop
    first; that is resolved pessimistically, and it applies to under 1% of
    qualifying sessions.
    """
    ok, _ = capitulation_qualifies(gap, index_gap, prior_20d_return)
    if not ok or session_open <= 0 or not np.isfinite(session_open):
        return None
    lv = capitulation_levels(session_open)
    if session_low <= lv["stop"]:
        return {"outcome": "stop", "ret": -CAP_STOP, "made_target": False}
    if session_high >= lv["target"]:
        return {"outcome": "target", "ret": CAP_TARGET, "made_target": True}
    return {"outcome": "close", "ret": float(session_close / session_open - 1.0),
            "made_target": False}


def render_capitulation_plan(ticker: str, session_open: float, gap: float,
                             index_gap: float, prior_20d_return: float,
                             width: int = 78) -> str:
    lv = capitulation_levels(session_open)
    ok, reasons = capitulation_qualifies(gap, index_gap, prior_20d_return)
    line = "=" * width
    out = [line, f" {ticker}  -  CAPITULATION GAP REVERSAL", line,
           f" opens {session_open:,.0f}   gap {gap:+.1%}   index gap {index_gap:+.1%}"
           f"   prior 20d {prior_20d_return:+.1%}", ""]
    out.append(" ORDERS")
    out.append(f"   buy          {lv['entry']:,.0f}   at the open")
    out.append(f"   target       {lv['target']:,.0f}   (+{CAP_TARGET:.0%})")
    out.append(f"   stop         {lv['stop']:,.0f}   (-{CAP_STOP:.0%}); hit 0.9% of the time")
    out.append(f"   hard exit    the close - never hold this overnight")
    out.append("")
    if ok:
        out.append(" all three conditions met at the open")
    else:
        for r in reasons:
            out.append(f"   ! {r}")
        out.append("   -> no trade")
    out.append("")
    out.append(" MEASURED n=544 over 25 years: 84.6% made >=5%, CI [81.5%, 87.6%]")
    out.append(" Walk-forward (each year scored on prior years): 86.2%, CI [83.0%, 89.5%]")
    out.append(" +3.39%/trade. Fires ~21x a year across the whole exchange.")
    out.append(" Weakest year 2025 at 65% (n=68); 8 of 9 years >=75%.")
    return "\n".join(out + [line])
