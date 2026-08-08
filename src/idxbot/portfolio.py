"""Long-only portfolio simulation from scored observations.

Why this exists on top of ``evaluate.py``: a rank IC of +0.041 is a statement
about correlation, not about money. A quintile *spread* is closer, but it
assumes you can short the bottom leg - and shorting IDX single names is
restricted enough in practice that the number is not achievable.

This simulates what a retail account can actually do: hold the top N names by
score, equal weighted, rebalanced every H trading days, paying costs on the
turnover, and compare it against holding the whole universe equal weighted.

Method, and its one important simplification: the holding period is set equal to
the forward-return horizon, so each period's return is simply the mean of the
selected names' ``fwd_H``. That keeps the simulation free of any look-ahead
(``fwd_H`` at date *t* uses only prices after *t*), at the cost of assuming
positions are held for exactly H days with no intra-period stop. Real execution
with stops would differ - usually with a lower average return and a shallower
drawdown.

What is still NOT corrected for:
  * survivorship - the universe is today's constituent list
  * liquidity - no check that your size fits the name's daily value traded
  * dividends - price returns only, so total return is understated
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class PortfolioResult:
    periods: pd.DataFrame = field(default_factory=pd.DataFrame)
    stats: Dict[str, float] = field(default_factory=dict)
    benchmark_stats: Dict[str, float] = field(default_factory=dict)
    index_stats: Dict[str, float] = field(default_factory=dict)
    holdings: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def empty(self) -> bool:
        return self.periods.empty


def _annualise(total_return: float, periods: int, periods_per_year: float) -> float:
    if periods <= 0 or total_return <= -1.0:
        return float("nan")
    years = periods / periods_per_year
    if years <= 0:
        return float("nan")
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def _stats(returns: pd.Series, periods_per_year: float) -> Dict[str, float]:
    """Summary statistics for a series of per-period returns."""
    if returns is None or len(returns) < 2:
        return {}
    equity = (1.0 + returns).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    drawdown = equity / equity.cummax() - 1.0

    mean, std = float(returns.mean()), float(returns.std(ddof=1))
    sharpe = (mean / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")

    downside = returns[returns < 0]
    sortino = (
        mean / downside.std(ddof=1) * np.sqrt(periods_per_year)
        if len(downside) > 1 and downside.std(ddof=1) > 0 else float("nan")
    )

    return {
        "periods": float(len(returns)),
        "total_return": total,
        "cagr": _annualise(total, len(returns), periods_per_year),
        "mean_period_return": mean,
        "volatility_annual": float(std * np.sqrt(periods_per_year)),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((returns > 0).mean()),
        "best_period": float(returns.max()),
        "worst_period": float(returns.min()),
    }


def simulate(
    observations: pd.DataFrame,
    top_n: int = 10,
    horizon: int = 60,
    cost_per_side: float = 0.002,
    min_names: int = 15,
    score_col: str = "score",
    min_score: Optional[float] = None,
    index_prices: Optional[pd.Series] = None,
) -> PortfolioResult:
    """Simulate a long-only, equal-weight, top-N portfolio.

    ``cost_per_side`` is charged on the fraction of the book that actually
    turns over, so a portfolio that keeps most of its names between rebalances
    pays proportionally less.

    ``index_prices`` (a date-indexed close series for the IHSG) adds a third,
    **survivorship-free** comparison. This matters more than it sounds: the
    equal-weight universe here is built from *today's* constituent lists, so it
    silently contains only names that survived and were promoted into an index.
    A momentum strategy benefits from that bias more than equal weight does,
    because the survivors are precisely the high-momentum names. Comparing
    against the real index is what exposes the size of the effect.
    """
    col = f"fwd_{horizon}"
    if observations is None or observations.empty or col not in observations.columns:
        return PortfolioResult()

    df = observations[["date", "ticker", score_col, col]].dropna().copy()
    if df.empty:
        return PortfolioResult()
    df = df.sort_values("date")

    # Rebalance dates spaced one holding period apart, so periods do not
    # overlap. Overlapping periods would double-count the same return and
    # flatter the equity curve.
    dates = np.array(sorted(df["date"].unique()))
    if len(dates) < 4:
        return PortfolioResult()

    spacing = max(1, len(dates) // max(1, _estimate_periods(dates, horizon)))
    rebalance_dates = dates[::spacing]

    # Index closes on the exchange calendar, used for the survivorship-free leg.
    index_series = None
    if index_prices is not None and len(index_prices):
        index_series = index_prices.copy()
        index_series.index = pd.DatetimeIndex(index_series.index).normalize()
        index_series = index_series.sort_index()

    rows: List[dict] = []
    holdings: List[dict] = []
    previous: set = set()

    for date in rebalance_dates:
        day = df[df["date"] == date]
        if len(day) < min_names:
            continue
        if min_score is not None:
            day = day[day[score_col] >= min_score]
            if day.empty:
                continue

        selected = day.nlargest(min(top_n, len(day)), score_col)
        names = set(selected["ticker"])

        # Turnover: fraction of the book replaced since the last rebalance.
        turnover = 1.0 if not previous else len(names - previous) / max(len(names), 1)
        cost = turnover * cost_per_side * 2.0

        gross = float(selected[col].mean())
        benchmark = float(day[col].mean())
        index_return = _index_return(index_series, date, horizon)

        rows.append({
            "date": pd.Timestamp(date),
            "names": len(names),
            "gross_return": gross,
            "turnover": turnover,
            "cost": cost,
            "net_return": gross - cost,
            "benchmark_return": benchmark,
            "index_return": index_return,
            "excess": gross - cost - benchmark,
            "excess_vs_index": (gross - cost - index_return
                                if np.isfinite(index_return) else np.nan),
            "mean_score": float(selected[score_col].mean()),
        })
        for _, r in selected.iterrows():
            holdings.append({"date": pd.Timestamp(date), "ticker": r["ticker"],
                             "score": r[score_col], "fwd_return": r[col]})
        previous = names

    if len(rows) < 3:
        return PortfolioResult()

    periods = pd.DataFrame(rows)
    periods_per_year = 252.0 / horizon

    index_stats = {}
    if periods["index_return"].notna().sum() >= 3:
        index_stats = _stats(periods["index_return"].dropna(), periods_per_year)

    return PortfolioResult(
        periods=periods,
        stats=_stats(periods["net_return"], periods_per_year),
        benchmark_stats=_stats(periods["benchmark_return"], periods_per_year),
        index_stats=index_stats,
        holdings=pd.DataFrame(holdings),
    )


def _index_return(index_series: Optional[pd.Series], date, horizon: int) -> float:
    """Index return over the same holding period, on the exchange calendar."""
    if index_series is None:
        return float("nan")
    positions = index_series.index.searchsorted(pd.Timestamp(date))
    if positions >= len(index_series):
        return float("nan")
    end = positions + horizon
    if end >= len(index_series):
        return float("nan")
    start_price = float(index_series.iloc[positions])
    if start_price <= 0:
        return float("nan")
    return float(index_series.iloc[end] / start_price - 1.0)


def _estimate_periods(dates: np.ndarray, horizon: int) -> int:
    """How many non-overlapping holding periods the sample supports."""
    span_days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
    trading_days = span_days * (252.0 / 365.25)
    return max(3, int(trading_days / max(horizon, 1)))


def render(result: PortfolioResult, label: str = "", width: int = 78) -> str:
    out: List[str] = []
    line = "=" * width
    out.append(line)
    out.append(f" LONG-ONLY PORTFOLIO SIMULATION{('  -  ' + label) if label else ''}")
    out.append(line)

    if result.empty:
        out.append(" Not enough data to simulate.")
        return "\n".join(out)

    periods = result.periods
    out.append(f" rebalances   : {len(periods)}")
    out.append(f" period       : {periods['date'].min():%Y-%m-%d} -> "
               f"{periods['date'].max():%Y-%m-%d}")
    out.append(f" avg turnover : {periods['turnover'].mean():.0%} per rebalance")
    out.append(f" avg cost     : {periods['cost'].mean():.2%} per rebalance")
    out.append("")

    strategy, benchmark = result.stats, result.benchmark_stats
    index = result.index_stats
    out.append(f" {'metric':<20}{'strategy':>13}{'universe EW':>13}{'IHSG':>13}")
    out.append(" " + "-" * (width - 2))

    def row(name: str, key: str, fmt: str = "pct") -> None:
        values = [strategy.get(key, np.nan), benchmark.get(key, np.nan),
                  index.get(key, np.nan)]
        cells = ""
        for v in values:
            if not np.isfinite(v):
                cells += f"{'-':>13}"
            elif fmt == "pct":
                cells += f"{v:>12.2%} "
            else:
                cells += f"{v:>12.2f} "
        out.append(f" {name:<20}{cells}")

    row("total return", "total_return")
    row("CAGR", "cagr")
    row("annual volatility", "volatility_annual")
    row("Sharpe", "sharpe", "num")
    row("Sortino", "sortino", "num")
    row("max drawdown", "max_drawdown")
    row("hit rate", "hit_rate")
    row("worst period", "worst_period")

    out.append("")
    excess = periods["excess"]
    t_stat = (excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess)))) \
        if excess.std(ddof=1) else np.nan
    out.append(f" excess vs universe EW : {excess.mean():+.2%} per period, "
               f"positive {(excess > 0).mean():.0%} of periods, t={t_stat:.2f}")

    if "excess_vs_index" in periods and periods["excess_vs_index"].notna().sum() > 2:
        vi = periods["excess_vs_index"].dropna()
        ti = (vi.mean() / (vi.std(ddof=1) / np.sqrt(len(vi)))) if vi.std(ddof=1) else np.nan
        out.append(f" excess vs IHSG        : {vi.mean():+.2%} per period, "
                   f"positive {(vi > 0).mean():.0%} of periods, t={ti:.2f}")
        gap = (benchmark.get("cagr", np.nan) - index.get("cagr", np.nan))
        if np.isfinite(gap):
            out.append("")
            out.append(f" SURVIVORSHIP CHECK: the equal-weight universe beat IHSG by "
                       f"{gap:+.2%} CAGR.")
            out.append(" That gap is not skill - it is this universe being built from")
            out.append(" today's index constituents. Momentum benefits from that bias more")
            out.append(" than equal weight does, so treat the strategy-vs-IHSG column as an")
            out.append(" upper bound, not an expectation.")
    out.append("")
    out.append(" Net of costs. No survivorship adjustment, no liquidity check, price")
    out.append(" returns only (dividends excluded). Positions are assumed held for the")
    out.append(" full period with no stop, so real execution will differ.")
    out.append(line)
    return "\n".join(out)


def equity_curve(result: PortfolioResult) -> pd.DataFrame:
    """Cumulative equity for the strategy and the equal-weight universe."""
    if result.empty:
        return pd.DataFrame()
    periods = result.periods
    return pd.DataFrame({
        "date": periods["date"],
        "strategy": (1.0 + periods["net_return"]).cumprod(),
        "universe": (1.0 + periods["benchmark_return"]).cumprod(),
    })
