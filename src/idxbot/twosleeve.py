"""A 50/50 book: blue-chip compounding on one side, multibagger lottery on the other.

The two sleeves are not a diversification gesture. They are driven by *opposite*
factors, which the data says plainly:

=========================  ====================  =====================
feature                    blue-chip sleeve      multibagger sleeve
=========================  ====================  =====================
12-month momentum          **high** wins         **low** wins (1.3x)
price level                irrelevant            **low** wins (4.0x)
turnover / size            large, for fills      **small** wins (2.9x)
distance below high        near the high wins    **far below** wins (1.8x)
volatility                 irrelevant            raises 3x odds, cuts return
holding period             20 days               3 years
=========================  ====================  =====================

Ranking a single universe on a single score cannot express that, because the two
objectives want opposite ends of four of the same columns. Two sleeves can.

What each sleeve is
-------------------
**Blue chip (50%).** The cross-sectional momentum book validated in Part II and
walk-forwarded in Part VIII: rank liquid large caps within each date, hold the
top few for 20 days, no take-profit, always invested. Its job is to compound.

**Multibagger (50%).** Small, cheap, and far below its old high. Held for three
years, because that is the horizon on which the pattern was measured and a
20-day rebalance would sell every one of them before it worked. Its job is to
own a lottery-ticket book where the mean is carried by a handful of names.

Note what is *not* in it: volatility and volume-surge, both of which looked
strong on 3x probability and both of which cost return when traded. See
``REJECTED_FACTORS``.

The honest warning, which is not boilerplate
--------------------------------------------
**Survivorship falls almost entirely on the multibagger sleeve.** The panel is
built from today's listing, so companies that went to zero and delisted are
absent. "Small, cheap, beaten-down, volatile" is not only the profile of a stock
about to 5x - it is also precisely the profile of a stock about to be delisted.
The winners are all in the data and an unknown share of the losers are not.

So the multibagger hit rates here are **upper bounds**, and the true left tail is
worse than anything this module can measure. The blue-chip sleeve is far less
exposed: large liquid names rarely vanish. Treat the two halves as carrying very
different evidential weight, because they do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .maxprofit import COST, _cagr

#: Multibagger sleeve: what each feature should be ranked toward. +1 means high
#: values score well, -1 means low values do. Derived from the quintile study in
#: ``scripts/multibagger_study.py``, not chosen by taste.
BAGGER_FACTORS: Dict[str, float] = {
    "price": -1.0,        # cheapest quintile 3x'd at 11.2% vs 2.8% dearest
    "turnover": -1.0,     # smallest quintile 9.0% vs 3.1% largest
    "hi_750": -1.0,       # furthest below the 3-year high 10.1% vs 5.5%
}

#: Volatility and volume-surge are deliberately EXCLUDED, and the reason is the
#: most useful lesson in this module.
#:
#: Both looked like strong factors in the quintile study - high realised
#: volatility lifted the 3x rate 1.66x, a volume surge lifted it 1.69x. Both
#: *reduce* the sleeve's return when actually traded: dropping ``vol_trend``
#: improved CAGR by +5.6%, dropping ``vol_60`` by +3.7%, and either used alone
#: underperformed owning the whole universe.
#:
#: They are not contradictory findings. Volatility raises the chance of a 3x and
#: raises the chance of a wipeout by more. **Selecting on hit-rate lift is not
#: the same as selecting on expected return**, and a screen built from a
#: probability table without checking the return table will quietly buy lottery
#: tickets at above fair value.
REJECTED_FACTORS: Dict[str, float] = {"vol_trend": +1.0, "vol_60": +1.0}

#: Floor for the multibagger sleeve. Below this a fill is imaginary, and the
#: "small is better" gradient runs straight into untradeable microcaps.
BAGGER_MIN_TURNOVER = 1e9


def _rank_within_date(df: pd.DataFrame, column: str, sign: float) -> pd.Series:
    """Percentile rank inside each date, oriented so that higher is better.

    Ranking within the date rather than against an absolute threshold is what
    makes a screen usable across regimes: "cheap" meant something different in
    2008 and in 2021, but "cheapest fifth of what is liquid today" does not.
    """
    return df.groupby("date")[column].rank(pct=True) * sign


def bagger_score(panel: pd.DataFrame,
                 factors: Optional[Dict[str, float]] = None) -> pd.Series:
    """Composite multibagger score: mean of oriented within-date ranks."""
    factors = factors or BAGGER_FACTORS
    parts = [_rank_within_date(panel, col, sign)
             for col, sign in factors.items() if col in panel.columns]
    if not parts:
        return pd.Series(np.nan, index=panel.index)
    return sum(parts) / len(parts)


def _thin(dates: Sequence[pd.Timestamp], hold_days: int) -> List[pd.Timestamp]:
    """Space rebalances so no two holding periods overlap."""
    ordered = sorted(pd.to_datetime(list(dates)))
    if not ordered:
        return []
    gap = pd.Timedelta(days=int(round(hold_days * 7 / 5)))
    kept = [ordered[0]]
    for d in ordered[1:]:
        if d - kept[-1] >= gap:
            kept.append(d)
    return kept


@dataclass
class SleeveResult:
    label: str
    periods: pd.Series          # return per rebalance
    equity: pd.Series
    stats: Dict[str, float]
    picks: pd.DataFrame


def _summarise(label: str, stamps: List[pd.Timestamp], rets: List[float],
               picks: pd.DataFrame) -> SleeveResult:
    arr = np.asarray(rets, dtype=float)
    growth = np.cumprod(1.0 + arr)
    equity = pd.Series(growth, index=pd.DatetimeIndex(stamps))
    years = max((stamps[-1] - stamps[0]).days / 365.25, 1e-9)
    peak = np.maximum.accumulate(growth)
    return SleeveResult(
        label=label,
        periods=pd.Series(arr, index=pd.DatetimeIndex(stamps)),
        equity=equity,
        picks=picks,
        stats={
            "rebalances": float(len(arr)), "years": years,
            "total_growth": float(growth[-1]),
            "cagr": _cagr(float(growth[-1]), years),
            "mean_period": float(arr.mean()),
            "median_period": float(np.median(arr)),
            "hit_rate": float((arr > 0).mean()),
            "best_period": float(arr.max()), "worst_period": float(arr.min()),
            "max_drawdown": float(np.min(growth / peak - 1.0)),
        })


def run_bagger_sleeve(panel: pd.DataFrame, top_n: int = 10,
                      hold_days: int = 750, cost: float = COST,
                      min_turnover: float = BAGGER_MIN_TURNOVER,
                      min_names: int = 20,
                      factors: Optional[Dict[str, float]] = None
                      ) -> Optional[SleeveResult]:
    """Buy the top-N multibagger candidates and hold them for three years.

    ``panel`` must carry the trailing features and a ``fwd_3y`` label built by
    ``scripts/multibagger_study.py``. The long hold is not a preference: the
    pattern was measured over three years, and rebalancing monthly would sell
    every candidate long before the thesis had a chance to pay.
    """
    df = panel[(panel["turnover"] >= min_turnover) & (panel["price"] >= 50)].copy()
    df = df.dropna(subset=["fwd_3y"])
    if df.empty:
        return None
    df["_score"] = bagger_score(df, factors)
    df = df.dropna(subset=["_score"])

    keep = _thin(df["date"].unique(), hold_days)
    df = df[df["date"].isin(keep)]

    stamps, rets, rows = [], [], []
    for date, group in df.groupby("date"):
        if len(group) < min_names:
            continue
        picks = group.nlargest(top_n, "_score")
        stamps.append(date)
        rets.append(float(picks["fwd_3y"].mean()) - cost)
        for _, r in picks.iterrows():
            rows.append({"date": date, "ticker": r["ticker"],
                         "price": r["price"], "fwd_3y": r["fwd_3y"]})
    if len(stamps) < 3:
        return None
    return _summarise(f"multibagger top{top_n} / {hold_days}d",
                      stamps, rets, pd.DataFrame(rows))


def bagger_benchmark(panel: pd.DataFrame, hold_days: int = 750,
                     min_turnover: float = BAGGER_MIN_TURNOVER,
                     min_names: int = 20) -> Optional[SleeveResult]:
    """Own every eligible name over the same windows: the 'no screen' book."""
    df = panel[(panel["turnover"] >= min_turnover)
               & (panel["price"] >= 50)].dropna(subset=["fwd_3y"])
    keep = _thin(df["date"].unique(), hold_days)
    df = df[df["date"].isin(keep)]
    stamps, rets = [], []
    for date, group in df.groupby("date"):
        if len(group) < min_names:
            continue
        stamps.append(date)
        rets.append(float(group["fwd_3y"].mean()))
    if len(stamps) < 3:
        return None
    return _summarise("equal-weight, no screen", stamps, rets, pd.DataFrame())


def blend_sleeves(blue: pd.Series, bagger: pd.Series, weight_blue: float = 0.5,
                  rebalance: bool = True) -> Tuple[pd.Series, Dict[str, float]]:
    """Combine two equity curves into one book.

    ``rebalance=True`` resets the split back to the target at every common
    date - the classic constant-mix. That is not a detail: with two sleeves
    whose returns are driven by opposite factors, rebalancing systematically
    sells whichever has run and buys whichever has not, and on volatile,
    weakly-correlated sleeves that transfer is worth real return on its own.

    ``rebalance=False`` lets the winner take over the book, which is what
    actually happens if you never touch it.
    """
    idx = blue.index.union(bagger.index).sort_values()
    b = blue.reindex(idx).ffill().bfill()
    g = bagger.reindex(idx).ffill().bfill()
    b_ret = b.pct_change().fillna(0.0)
    g_ret = g.pct_change().fillna(0.0)

    if rebalance:
        combined = (1.0 + weight_blue * b_ret
                    + (1.0 - weight_blue) * g_ret).cumprod()
    else:
        combined = weight_blue * (b / b.iloc[0]) + (1.0 - weight_blue) * (g / g.iloc[0])

    years = max((idx[-1] - idx[0]).days / 365.25, 1e-9)
    curve = combined.to_numpy(float)
    peak = np.maximum.accumulate(curve)
    stats = {
        "years": years,
        "total_growth": float(curve[-1]),
        "cagr": _cagr(float(curve[-1]), years),
        "max_drawdown": float(np.min(curve / peak - 1.0)),
        "weight_blue": weight_blue,
        "rebalanced": float(bool(rebalance)),
    }
    return combined, stats


def render_sleeve(result: SleeveResult, benchmark: Optional[SleeveResult] = None,
                  width: int = 78) -> str:
    s = result.stats
    lines = ["=" * width, f" {result.label}", "=" * width,
             f" {'rebalances':<24}{s['rebalances']:>12,.0f}  over {s['years']:.1f} years",
             f" {'total growth':<24}{s['total_growth']:>12,.2f}x",
             f" {'CAGR':<24}{s['cagr']:>12.2%}"]
    if benchmark is not None:
        lines.append(f" {'benchmark CAGR':<24}{benchmark.stats['cagr']:>12.2%}")
        lines.append(f" {'excess':<24}"
                     f"{s['cagr'] - benchmark.stats['cagr']:>12.2%}")
    lines += [f" {'mean per period':<24}{s['mean_period']:>12.1%}",
              f" {'median per period':<24}{s['median_period']:>12.1%}",
              f" {'hit rate':<24}{s['hit_rate']:>12.0%}",
              f" {'best / worst period':<24}"
              f"{s['best_period']:>7.0%} / {s['worst_period']:.0%}",
              f" {'max drawdown':<24}{s['max_drawdown']:>12.1%}",
              "=" * width]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# combining sleeves that run on completely different clocks
# ---------------------------------------------------------------------------
def annualise_blue(periods: pd.Series) -> pd.Series:
    """Compound 20-day book returns into calendar-year returns."""
    if periods.empty:
        return pd.Series(dtype=float)
    df = pd.DataFrame({"r": periods.to_numpy()}, index=pd.DatetimeIndex(periods.index))
    return df.groupby(df.index.year)["r"].apply(lambda s: float(np.prod(1.0 + s) - 1.0))


def ladder_bagger(periods: pd.Series, hold_years: int = 3) -> pd.Series:
    """Turn lumpy three-year bets into an annual return series by laddering.

    A single three-year holding is unusable as half a portfolio: capital is
    committed in one lump, on one date, with one outcome. The standard fix is a
    ladder - commit a third of the sleeve each year and hold each tranche three
    years, so at steady state three tranches are always live and one matures
    annually.

    That is a genuine improvement in *implementation*: it removes start-date
    risk and gives the sleeve an annual cadence. It does **not** manufacture
    evidence. The underlying three-year outcomes are still the same handful of
    overlapping windows, and averaging them more smoothly does not make them
    more independent.

    Each period return is spread geometrically across the years it covers, which
    assumes a tranche compounds evenly over its life. It does not, but nothing
    in the data says which years within a hold did the work.
    """
    if periods.empty:
        return pd.Series(dtype=float)
    contributions: Dict[int, List[float]] = {}
    for start, total in periods.items():
        per_year = (1.0 + max(total, -0.999)) ** (1.0 / hold_years) - 1.0
        for k in range(hold_years):
            contributions.setdefault(pd.Timestamp(start).year + k, []).append(per_year)
    return pd.Series({y: float(np.mean(v)) for y, v in sorted(contributions.items())})


def combine_annual(blue_annual: pd.Series, bagger_annual: pd.Series,
                   weight_blue: float = 0.5, rebalance: bool = True
                   ) -> Tuple[pd.Series, Dict[str, float]]:
    """Blend two annual return series into one book.

    ``rebalance=True`` is constant-mix: reset to the target weights each year,
    which mechanically trims whichever sleeve ran and adds to whichever lagged.
    With two sleeves driven by opposite factors that transfer is worth real
    return, so the comparison against ``rebalance=False`` is reported rather
    than assumed.
    """
    years = sorted(set(blue_annual.index) & set(bagger_annual.index))
    if not years:
        return pd.Series(dtype=float), {}
    b = blue_annual.reindex(years).astype(float)
    g = bagger_annual.reindex(years).astype(float)

    if rebalance:
        yearly = weight_blue * b + (1.0 - weight_blue) * g
        curve = np.cumprod(1.0 + yearly.to_numpy())
    else:
        wb = np.cumprod(1.0 + b.to_numpy()) * weight_blue
        wg = np.cumprod(1.0 + g.to_numpy()) * (1.0 - weight_blue)
        curve = wb + wg
        yearly = pd.Series(np.diff(np.concatenate([[1.0], curve])) /
                           np.concatenate([[1.0], curve[:-1]]), index=years)

    equity = pd.Series(curve, index=years)
    peak = np.maximum.accumulate(curve)
    n = len(years)
    return equity, {
        "years": float(n),
        "total_growth": float(curve[-1]),
        "cagr": _cagr(float(curve[-1]), float(n)),
        "worst_year": float(yearly.min()),
        "best_year": float(yearly.max()),
        "positive_years": float((yearly > 0).mean()),
        "max_drawdown": float(np.min(curve / peak - 1.0)),
        "weight_blue": weight_blue,
        "rebalanced": float(bool(rebalance)),
    }
