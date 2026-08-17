"""Maximum-terminal-wealth portfolio construction for IDX.

This module exists because of Result 28 in ``docs/FINDINGS.md``. Hull+UT lost to
buy-and-hold not because its signal was inverted but because it sat in cash 82%
of the time in a market that drifts up ~12% a year: **time out of the market is
charged before the signal is even consulted.** Every design choice here follows
from that one finding.

The design, and the evidence behind each part
---------------------------------------------
1. **Always invested. No market timing.** Part IV found macro times the index
   but does not improve selection, and Result 28 priced what being absent costs.
   So the only decision this makes is *which* stocks, never *whether*.
2. **Cross-sectional selection, ranked within each date.** Market direction
   cancels out of a within-date ranking, which is why the IC results in Part II
   survived and the timing results did not.
3. **No take-profit.** Part III measured this directly: holding 60 days flat
   returned +9.27% while capping the same trades at +5% returned +0.02%, because
   the winners average +28.6%. A target is a machine for selling the only trades
   that pay for the rest.
4. **Concentration is a parameter, and it is swept.** For maximum terminal
   wealth the question is empirical, not a matter of taste: if the ranking has
   signal, fewer names capture more of it and carry more variance. This measures
   the trade-off instead of assuming a "prudent" answer.
5. **Rebalances do not overlap.** Sampling a 60-day forward return every 5 days
   reuses the same future twelve times; spacing rebalances by the holding period
   is both the statistically honest choice and what trading it looks like.

What is deliberately *not* in here
----------------------------------
* **Fundamentals as a ranking factor.** Only a present-day snapshot of ratios
  exists, so joining it to history is look-ahead by construction (Part I). They
  are available as a *present-tense exclusion filter* and nothing more, and this
  module does not pretend otherwise.
* **Broker flow.** Real rekap broker is finally connected (Part VI) but has
  never been tested against forward returns, and Part V is the standing warning
  about trusting a flow statistic before its deciles are examined.

Survivorship, stated once and loudly
------------------------------------
The observation panel is built from today's IDX listing, so names that delisted
are missing. Part II measured this at roughly three quarters of the headline
CAGR for a long-only equal-weight book. **Every absolute return in this module
is inflated by it.** The equal-weight benchmark is inflated by the same bias, so
*differences* against that benchmark are far more trustworthy than levels, and
that is why nothing here is reported without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .walkforward import blend, nonoverlapping

#: Round-trip cost per rebalance: IDX retail fees plus slippage on both legs.
COST = 0.006

#: Below this median daily turnover a fill is a fantasy at portfolio size.
MIN_TURNOVER = 5e9

TRADING_DAYS = 252.0


@dataclass
class BookResult:
    """One configuration's compounded record."""

    label: str
    equity: pd.Series
    picks: pd.DataFrame
    stats: Dict[str, float] = field(default_factory=dict)


def _cagr(total_growth: float, years: float) -> float:
    if years <= 0 or total_growth <= 0:
        return float("nan")
    return float(total_growth ** (1.0 / years) - 1.0)


def eligible(observations: pd.DataFrame, min_turnover: float = MIN_TURNOVER,
             min_close: float = 50.0) -> pd.DataFrame:
    """Point-in-time liquidity and price screen.

    ``vt`` is the trailing 20-day median turnover computed as of each row's own
    date, so this filter never consults the future. The Rp50 floor is the IDX
    regular-market minimum price, below which a "stock" is a lottery ticket
    whose tick size is 2% of its value.
    """
    df = observations
    out = df[(df["vt"] >= min_turnover) & (df["close"] >= min_close)]
    return out


def run_book(observations: pd.DataFrame, weights: Dict[str, float],
             horizon: int = 60, top_n: int = 5, cost: float = COST,
             min_names: int = 20, min_turnover: float = MIN_TURNOVER,
             label: str = "") -> Optional[BookResult]:
    """Compound an equal-weight top-N book through non-overlapping rebalances.

    Returns ``None`` when the panel cannot support the configuration, rather
    than a misleading near-empty equity curve.
    """
    fwd = f"fwd_{horizon}"
    if fwd not in observations.columns:
        return None
    df = eligible(observations, min_turnover).copy()
    df["_score"] = blend(df, weights)
    df = df.dropna(subset=["_score", fwd])
    if df.empty:
        return None

    dates = nonoverlapping(df["date"].unique(), horizon)
    df = df[df["date"].isin(pd.to_datetime(list(dates)))]

    rows: List[Dict[str, object]] = []
    periods: List[Tuple[pd.Timestamp, float]] = []
    for date, group in df.groupby("date"):
        if len(group) < min_names:
            continue                     # cross-section too thin to be a choice
        picks = group.nlargest(top_n, "_score")
        # Equal weight, held to the horizon, one round trip charged per period.
        period_return = float(picks[fwd].mean()) - cost
        periods.append((date, period_return))
        for _, r in picks.iterrows():
            rows.append({"date": date, "ticker": r["ticker"],
                         "score": r["_score"], "fwd": r[fwd]})
    if len(periods) < 8:
        return None

    stamps = [d for d, _ in periods]
    growth = np.cumprod([1.0 + r for _, r in periods])
    equity = pd.Series(growth, index=pd.DatetimeIndex(stamps))
    years = max((stamps[-1] - stamps[0]).days / 365.25, 1e-9)

    returns = np.array([r for _, r in periods])
    peak = np.maximum.accumulate(equity.to_numpy())
    drawdown = float(np.min(equity.to_numpy() / peak - 1.0))
    per_year = len(periods) / years

    stats = {
        "rebalances": float(len(periods)),
        "years": years,
        "total_growth": float(growth[-1]),
        "cagr": _cagr(float(growth[-1]), years),
        "mean_period": float(returns.mean()),
        "median_period": float(np.median(returns)),
        "hit_rate": float((returns > 0).mean()),
        "worst_period": float(returns.min()),
        "best_period": float(returns.max()),
        "max_drawdown": drawdown,
        "sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(per_year))
        if returns.std(ddof=1) > 0 else float("nan"),
    }
    return BookResult(label=label or f"top{top_n}/{horizon}d",
                      equity=equity, picks=pd.DataFrame(rows), stats=stats)


def equal_weight_benchmark(observations: pd.DataFrame, horizon: int = 60,
                           cost: float = 0.0, min_names: int = 20,
                           min_turnover: float = MIN_TURNOVER) -> Optional[BookResult]:
    """Own every eligible name. The 'no selection at all' book.

    Charged zero cost by default: a buy-and-hold comparison should not be
    handicapped by trading it does not do. That makes it a *harder* benchmark
    than the strategy, which is the direction of conservatism worth having.
    """
    fwd = f"fwd_{horizon}"
    df = eligible(observations, min_turnover).dropna(subset=[fwd])
    dates = nonoverlapping(df["date"].unique(), horizon)
    df = df[df["date"].isin(pd.to_datetime(list(dates)))]
    periods = [(d, float(g[fwd].mean()) - cost)
               for d, g in df.groupby("date") if len(g) >= min_names]
    if len(periods) < 8:
        return None
    stamps = [d for d, _ in periods]
    growth = np.cumprod([1.0 + r for _, r in periods])
    years = max((stamps[-1] - stamps[0]).days / 365.25, 1e-9)
    returns = np.array([r for _, r in periods])
    equity = pd.Series(growth, index=pd.DatetimeIndex(stamps))
    peak = np.maximum.accumulate(equity.to_numpy())
    return BookResult(
        label="equal-weight (no selection)", equity=equity,
        picks=pd.DataFrame(),
        stats={"rebalances": float(len(periods)), "years": years,
               "total_growth": float(growth[-1]),
               "cagr": _cagr(float(growth[-1]), years),
               "mean_period": float(returns.mean()),
               "median_period": float(np.median(returns)),
               "hit_rate": float((returns > 0).mean()),
               "worst_period": float(returns.min()),
               "best_period": float(returns.max()),
               "max_drawdown": float(np.min(equity.to_numpy() / peak - 1.0)),
               "sharpe": float("nan")})


def sweep(observations: pd.DataFrame,
          weight_sets: "Dict[str, Dict[str, float]] | Sequence[Dict[str, float]]",
          horizons: Iterable[int] = (5, 10, 20, 60),
          top_ns: Iterable[int] = (1, 2, 3, 5, 8, 10, 15, 20, 30),
          cost: float = COST, min_turnover: float = MIN_TURNOVER,
          verbose: bool = False) -> pd.DataFrame:
    """Every (weight set, horizon, concentration) combination, ranked by CAGR.

    Optimising CAGR rather than a risk-adjusted ratio is a deliberate choice
    here: the objective asked for is terminal wealth. Drawdown is still reported
    on every row, because a configuration that compounds fastest through a 70%
    drawdown is one most people cannot actually hold.
    """
    if isinstance(weight_sets, dict):
        named = list(weight_sets.items())
    else:
        named = [(str(w.get("profile", "custom")), w) for w in weight_sets]

    rows = []
    for name, weights in named:
        for horizon in horizons:
            for top_n in top_ns:
                book = run_book(observations, weights, horizon=horizon,
                                top_n=top_n, cost=cost,
                                min_turnover=min_turnover)
                if book is None:
                    continue
                rows.append({"weights": name, "horizon": horizon,
                             "top_n": top_n, **book.stats})
        if verbose:
            print(f"    swept {name}")
    out = pd.DataFrame(rows)
    return out.sort_values("cagr", ascending=False).reset_index(drop=True) \
        if not out.empty else out


def split(observations: pd.DataFrame, fraction: float = 0.6
          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/holdout. Never random - rows share dates."""
    cut = observations["date"].quantile(fraction)
    return (observations[observations["date"] <= cut],
            observations[observations["date"] > cut])


def render(book: BookResult, benchmark: Optional[BookResult] = None,
           width: int = 78) -> str:
    lines = ["=" * width, f" {book.label}", "=" * width]
    s = book.stats
    lines.append(f" {'rebalances':<22}{s['rebalances']:>12,.0f}"
                 f"   over {s['years']:.1f} years")
    lines.append(f" {'total growth':<22}{s['total_growth']:>12,.1f}x")
    lines.append(f" {'CAGR':<22}{s['cagr']:>12.2%}")
    if benchmark is not None:
        b = benchmark.stats
        lines.append(f" {'equal-weight CAGR':<22}{b['cagr']:>12.2%}")
        lines.append(f" {'excess':<22}{s['cagr'] - b['cagr']:>12.2%}")
    lines.append(f" {'per-rebalance mean':<22}{s['mean_period']:>12.2%}")
    lines.append(f" {'hit rate':<22}{s['hit_rate']:>12.0%}")
    lines.append(f" {'best / worst period':<22}"
                 f"{s['best_period']:>7.1%} / {s['worst_period']:.1%}")
    lines.append(f" {'max drawdown':<22}{s['max_drawdown']:>12.1%}")
    lines.append("=" * width)
    return "\n".join(lines)


def walk_forward(observations: pd.DataFrame,
                 weight_sets: "Dict[str, Dict[str, float]]",
                 n_folds: int = 5, min_train_years: float = 8.0,
                 horizons: Iterable[int] = (5, 10, 20, 60),
                 top_ns: Iterable[int] = (3, 5, 8, 10, 15),
                 cost: float = COST) -> pd.DataFrame:
    """Choose the configuration in-sample, then score it on the slice that follows.

    A single train/holdout split gives one out-of-sample number, and one number
    is indistinguishable from luck. This re-selects from scratch in every fold
    and reports what the *selection procedure* earned, which is the thing you
    would actually be running - not what the best configuration earned with
    hindsight.

    The equal-weight book over the identical window is scored alongside, because
    a strategy that returns +20% in a window where owning everything returned
    +25% has not selected anything.
    """
    dates = pd.DatetimeIndex(sorted(observations["date"].unique()))
    first, last = dates[0], dates[-1]
    train_end0 = first + pd.Timedelta(days=int(min_train_years * 365.25))
    if train_end0 >= last:
        return pd.DataFrame()
    edges = pd.date_range(train_end0, last, periods=n_folds + 1)

    rows = []
    for k in range(n_folds):
        train_end, test_end = edges[k], edges[k + 1]
        tr = observations[observations["date"] <= train_end]
        te = observations[(observations["date"] > train_end)
                          & (observations["date"] <= test_end)]
        if tr.empty or te.empty:
            continue
        scored = sweep(tr, weight_sets, horizons=horizons, top_ns=top_ns, cost=cost)
        if scored.empty:
            continue
        best = scored.iloc[0]
        book = run_book(te, weight_sets[best["weights"]],
                        horizon=int(best["horizon"]), top_n=int(best["top_n"]),
                        cost=cost)
        bench = equal_weight_benchmark(te, horizon=int(best["horizon"]))
        if book is None or bench is None:
            continue
        rows.append({
            "fold": k + 1, "train_end": train_end, "test_end": test_end,
            "chosen": f"{best['weights']} {int(best['horizon'])}d "
                      f"top{int(best['top_n'])}",
            "in_sample_cagr": float(best["cagr"]),
            "oos_cagr": book.stats["cagr"],
            "oos_equal_weight": bench.stats["cagr"],
            "oos_excess": book.stats["cagr"] - bench.stats["cagr"],
            "oos_max_dd": book.stats["max_drawdown"],
            "rebalances": book.stats["rebalances"],
        })
    return pd.DataFrame(rows)
