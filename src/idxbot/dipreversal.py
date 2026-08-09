"""Intraday reversal rules — the only positive-expectancy intraday setups found.

Two variants, both buying capitulation and both exiting the same session:

  * ``GAP`` — the stock **opens** far below yesterday's close while the index is
    up. Buy at the open. This is the stronger of the two and the only rule in
    this repo whose point estimate reaches an 80% rate of making +5%.
  * ``DIP`` — the stock **falls** 10% from its own open while the index is up.
    Buy on a limit. Weaker but far more frequent.

Both rest on the same idea: a collapse that the market is not sharing is forced
or panicked selling in one name, and that reverts. A collapse the whole market
is sharing does not. The index filter is the rule, not a refinement — removing
it costs roughly eight points of hit rate.

Everything else tried intraday lost money. Opening-range breakouts, VWAP entries,
volatility filters, momentum bursts: all negative after the 0.40% round trip,
because the average IDX session drifts -0.42% from open to close and selection
could not overcome a starting position that bad.

This one works, and the reason it works is the reason it is narrow. It buys a
**deep idiosyncratic dip** — a stock down 10% intraday *while the index is up* —
which is the signature of forced or panicked selling in one name rather than a
market event. Those revert. Market-wide selloffs do not, which is why the index
filter is not decoration: without it the win rate drops from 69% to 61%.

    entry     limit at -10% from the session open, first three hours only
    filter    index up >0.5% at the moment of fill, prior 20-session trend positive
    target    +5% from the fill
    stop      -20% from the fill
    exit      the close, unconditionally - this never holds overnight

Measured on 168,586 sessions, 251 names, 2023-07 to 2026-08 (hourly bars):

    138 trades    63.0% made >=5%    68.8% net positive    +1.25%/trade    PF 1.71

Those first two columns answer different questions and the difference matters:
68.8% of trades finish above the fill, but only **63.0% actually reach the +5%
target**. Quote the second when the question is "how often does this make 5%".

Split chronologically the second half is the stronger one (73.9%, +1.46%), which
is reassuring but not conclusive: **roughly 140 configurations were compared to
arrive here**, and 138 trades is a thin sample for that much searching. Treat it
as a live experiment worth funding in small size, not as an established edge.

**What it is not.** It does not reach an 80% rate and no variant does. Dropping
the target to +1% reaches 74.6% net-positive, by which point expectancy is
-1.01%. Widening or removing the stop changes nothing (-20%, -30% and no stop
all measure 63.0%, because the stopped trades are the ones that would have
missed the target anyway). Entry depth peaks at -10% and falls away either side.
See docs/DAYTRADE.md §9 for the full record.

**Fill risk is the main practical caveat.** A limit 10% below the open fills
about 2% of sessions, and it fills preferentially on days that keep falling.
The index filter is what separates "panic that reverts" from "the start of a
real decline", and it is doing heavy lifting on a small sample.
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
