"""Path-dependent exits: what a target-and-stop trade actually does.

Every return in ``evaluate`` and ``walkforward`` is close-to-close at a fixed
horizon: buy today, look at the price 60 bars later, done. That is the right
way to measure whether a *signal* carries information, and the wrong way to
measure a *trade*, because nobody holds a position blind for three months. A
real position exits when it hits a target or a stop, whichever comes first.

The two measurements can differ enormously, and the difference is not noise:

    entry 1000, target +5% (1050), stop -8% (920), 60 bars
    path: 1000 -> 1060 (day 4) -> 940 -> 890 (day 55)
    fixed-horizon return : -11%   "a loss"
    barrier return       : +5%    "a win on day 4"

Same data, opposite verdicts, and the barrier one is what the account
experiences. So the probability of *touching* +5% is far higher than the
probability of *closing* above +5% at a fixed horizon, and a hit rate quoted
against the wrong one is meaningless.

Three rules keep this honest:

  * **Entry is the next bar's open, never the signal bar's close.** The close
    that produced the signal is not a price you could have traded on; by the
    time you have it, the session is over.
  * **Same-bar ambiguity resolves as a loss.** When a daily bar's high clears
    the target *and* its low breaks the stop, the tape order is unknowable from
    daily data. Assuming the good one is the single easiest way to manufacture
    an 80% win rate that does not exist.
  * **A high hit rate is not an edge.** Widening the stop raises the hit rate
    monotonically toward 100% and drives expectancy toward ruin. Every result
    here reports expectancy beside the hit rate, because the pair is the only
    honest unit. 90% winners at +3% against 10% losers at -30% is a losing
    strategy that looks like a winning one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

TARGET, STOP, TIMEOUT, NO_EXIT = "target", "stop", "timeout", "none"


@dataclass
class BarrierConfig:
    """A single all-or-nothing target, or a scale-out with a trailing remainder.

    The plain ``target_pct``/``stop_pct`` form exits the whole position at the
    first barrier. That is the rule most people describe when they say "I just
    want 5%", and on IDX it is close to the worst thing you can do: it caps the
    right tail, and the right tail is where this market's entire edge lives.

    ``scale_out`` fixes that without giving up the high hit rate. Sell a slice
    at the target - which banks a win on most trades - and let the rest run
    behind a trailing stop, so the occasional +80% name is still allowed to
    become one.
    """
    target_pct: float = 0.05
    stop_pct: float = 0.08
    max_days: int = 60
    scale_out: float = 1.0     # share sold at the target; 1.0 = exit everything
    trail_pct: float = 0.0     # trailing stop on the remainder; 0 = keep the fixed stop
    breakeven: bool = False    # after the target, never let the trade turn negative

    @property
    def label(self) -> str:
        base = f"+{self.target_pct:.0%}/-{self.stop_pct:.0%}/{self.max_days}d"
        if self.scale_out < 1.0:
            base += f" x{self.scale_out:.0%}"
        if self.breakeven:
            base += " BE"
        if self.trail_pct > 0 and self.scale_out < 1.0:
            base += f" trail{self.trail_pct:.0%}"
        return base


def simulate_one(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    entry_price: float,
    cfg: BarrierConfig,
) -> Dict[str, object]:
    """Walk one position forward bar by bar until a barrier is touched.

    ``highs``/``lows``/``closes`` start at the *entry* bar and run forward.
    """
    if not np.isfinite(entry_price) or entry_price <= 0 or len(highs) == 0:
        return {"outcome": NO_EXIT, "ret": np.nan, "days": 0}

    target = entry_price * (1.0 + cfg.target_pct)
    stop = entry_price * (1.0 - cfg.stop_pct)
    n = min(cfg.max_days, len(highs))

    remaining = 1.0
    realised = 0.0            # profit already banked, as a share of the position
    took_target = False
    peak = entry_price

    for i in range(n):
        hit_stop = lows[i] <= stop
        if hit_stop:
            # Checked before the target on purpose. If one bar spans both, the
            # order within the session is unknowable from daily data, so the
            # bad outcome is assumed. The optimistic reading is how backtests
            # grow win rates that evaporate live.
            leg = float(stop / entry_price - 1.0)
            return {"outcome": TARGET if took_target else STOP,
                    "ret": realised + remaining * leg, "days": i + 1}

        if not took_target and highs[i] >= target:
            realised += cfg.scale_out * cfg.target_pct
            remaining -= cfg.scale_out
            took_target = True
            if remaining <= 1e-9:
                return {"outcome": TARGET, "ret": realised, "days": i + 1}
            # The remainder now rides a trailing stop measured from the peak.
            # Raising it to at least breakeven is the point of scaling out: the
            # banked slice can no longer be given back.
            peak = max(peak, target)
            if cfg.breakeven:
                # Once the target prints, the entry price becomes the floor, so
                # the trade can no longer end negative. This is what actually
                # lifts the win rate - scaling out alone does not, because a
                # 25% slice banked at +3% cannot rescue the other 75% falling
                # to the stop. The cost is being shaken out of names that dip
                # under entry and then recover, which is a real cost to the
                # right tail and is measured, not assumed.
                stop = max(stop, entry_price)
            if cfg.trail_pct > 0:
                stop = max(stop, peak * (1.0 - cfg.trail_pct))

        if took_target and cfg.trail_pct > 0:
            peak = max(peak, highs[i])
            stop = max(stop, peak * (1.0 - cfg.trail_pct))

    leg = float(closes[n - 1] / entry_price - 1.0)
    return {"outcome": TARGET if took_target else TIMEOUT,
            "ret": realised + remaining * leg, "days": n}


def simulate_ticker(
    bars: pd.DataFrame,
    entry_dates: Sequence[pd.Timestamp],
    cfg: BarrierConfig,
) -> pd.DataFrame:
    """Barrier outcomes for every signal date on one ticker.

    A signal on bar ``i`` is entered at the open of bar ``i+1``. If there is no
    bar ``i+1`` the signal is dropped rather than filled at a price that never
    existed.
    """
    if bars is None or bars.empty or not len(entry_dates):
        return pd.DataFrame()

    frame = bars.reset_index(drop=True)
    pos = {d: i for i, d in enumerate(frame["date"])}
    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)

    rows = []
    for date in pd.to_datetime(list(entry_dates)):
        i = pos.get(date)
        if i is None or i + 1 >= len(frame):
            continue
        entry_price = opens[i + 1]
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue
        result = simulate_one(highs[i + 1:], lows[i + 1:], closes[i + 1:],
                              entry_price, cfg)
        rows.append({"date": date, "entry": entry_price, **result})

    return pd.DataFrame(rows)


def summarise(outcomes: pd.DataFrame, cost_pct: float = 0.004) -> Dict[str, float]:
    """Hit rate *and* expectancy. Neither means anything without the other.

    ``cost_pct`` is the IDX round trip - roughly 0.15% buy plus 0.25% sell
    including the 0.1% sales tax. It is charged on every trade, and at a 3%
    target it eats an eighth of the gross win, which is exactly why small
    targets flatter the hit rate and starve the return.
    """
    if outcomes is None or outcomes.empty:
        return {}

    df = outcomes.dropna(subset=["ret"])
    if df.empty:
        return {}

    net = df["ret"] - cost_pct
    wins = net > 0
    won, lost = net[wins], net[~wins]

    return {
        "trades": int(len(df)),
        "hit_rate": float((df["outcome"] == TARGET).mean()),
        "win_rate": float(wins.mean()),
        "target_rate": float((df["outcome"] == TARGET).mean()),
        "stop_rate": float((df["outcome"] == STOP).mean()),
        "timeout_rate": float((df["outcome"] == TIMEOUT).mean()),
        "mean_net": float(net.mean()),
        "median_net": float(net.median()),
        "avg_win": float(won.mean()) if len(won) else np.nan,
        "avg_loss": float(lost.mean()) if len(lost) else np.nan,
        "expectancy": float(net.mean()),
        "profit_factor": (float(won.sum() / abs(lost.sum()))
                          if len(lost) and lost.sum() != 0 else np.inf),
        "avg_days": float(df["days"].mean()),
        # Return per unit of time held. A rule that wins 80% of the time but
        # ties capital up for three months can lose to one that wins 55% in a
        # fortnight, and comparing raw expectancy hides that entirely.
        "ann_return": float(net.mean() * (252.0 / df["days"].mean()))
        if df["days"].mean() > 0 else np.nan,
    }


def render_grid(rows: List[Dict[str, object]], width: int = 100,
                target_hit_rate: float = 0.80) -> str:
    """Compare barrier settings, sorted by hit rate, with expectancy alongside."""
    line = "=" * width
    out = [line, " BARRIER GRID - hit rate is meaningless without expectancy", line]
    if not rows:
        return "\n".join(out + [" (no results)", line])

    # Two different questions, both called "success rate" in conversation:
    #   hit = did price ever touch the target?
    #   win = did the trade finish positive after costs?
    # They diverge as soon as you scale out, because a position can bank its
    # target slice and still hand the remainder back. Showing only one invites
    # the reader to assume it was the other.
    out.append(f" {'setup':<26}{'trades':>7}{'hit':>6}{'win':>6}{'stop':>6}"
               f"{'expect':>9}{'avg win':>9}{'avg loss':>9}{'PF':>8}{'days':>6}{'ann':>9}")
    for r in sorted(rows, key=lambda x: -float(x.get("win_rate", 0))):
        pf = r.get("profit_factor", np.nan)
        pf_text = "     inf" if pf == np.inf else f"{pf:>8.2f}"
        out.append(
            f" {str(r['setup']):<26}{int(r['trades']):>7}{r['hit_rate']:>6.0%}"
            f"{r['win_rate']:>6.0%}{r['stop_rate']:>6.0%}"
            f"{r['expectancy']:>9.2%}{r['avg_win']:>9.2%}{r['avg_loss']:>9.2%}"
            f"{pf_text}{r['avg_days']:>6.0f}{r['ann_return']:>9.1%}")

    qualifying = [r for r in rows if float(r.get("win_rate", 0)) >= target_hit_rate]
    profitable = [r for r in qualifying if float(r.get("expectancy", 0)) > 0]
    out.append("")
    out.append(f" setups reaching a {target_hit_rate:.0%} win rate : {len(qualifying)}")
    out.append(f" ...of which are profitable after costs   : {len(profitable)}")
    if qualifying and not profitable:
        out.append("")
        out.append(" Every setup that clears the win-rate bar loses money. That is the")
        out.append(" expected shape: the win rate was bought by widening the stop, so")
        out.append(" the rare loss is large enough to consume many small wins.")
    elif profitable:
        best = max(profitable, key=lambda r: float(r["ann_return"]))
        out.append(f" best of those, annualised            : {best['setup']} "
                   f"at {best['ann_return']:.1%}")
    return "\n".join(out + [line])
