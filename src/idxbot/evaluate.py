"""Cross-sectional and component-level evaluation of a signal.

Why this exists separately from ``backtest.py``: that module pools every
(ticker, date) observation into one bucket comparison, which mixes two very
different questions together.

  * **Time-series question** - "is now a good time to be long IDX?" Pooled
    bucket means answer mostly this, and they are dominated by market beta:
    every score moves together in a market-wide base, so the comparison is
    contaminated by whatever the index did next.
  * **Cross-sectional question** - "of the 45 names in front of me *today*,
    which should I buy?" This is what a screener is actually used for, and it
    is answered by ranking within each date and comparing the top of the
    ranking against the bottom *on that same date*. Market direction cancels
    out because both legs live in the same session.

The two can disagree completely: a signal can be useless in the time series and
genuinely informative cross-sectionally, or the reverse. The pooled test in
``backtest.py`` is the former; everything here is the latter.

Metrics produced:
  * **Rank IC** - Spearman correlation between score and forward return, computed
    within each date and then averaged. The *t*-statistic on the series of daily
    ICs is the honest significance test, because each date contributes one
    largely independent observation instead of thousands of overlapping ones.
  * **Quantile spread** - mean forward return of the top quantile minus the
    bottom, again computed within each date.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation, NaN-safe, without a scipy dependency."""
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < 5:
        return np.nan
    ranked = joined.rank()
    x, y = ranked.iloc[:, 0], ranked.iloc[:, 1]
    sx, sy = x.std(ddof=0), y.std(ddof=0)
    if sx == 0 or sy == 0:
        return np.nan
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def rank_ic(
    observations: pd.DataFrame,
    signal_col: str = "score",
    horizons: Sequence[int] = (5, 10, 20, 60),
    min_names: int = 8,
) -> pd.DataFrame:
    """Per-horizon cross-sectional rank IC, with a t-stat over dates.

    ``min_names`` guards against dates where only a handful of tickers have a
    score - a rank correlation over three names is noise.
    """
    if observations is None or observations.empty or signal_col not in observations:
        return pd.DataFrame()

    rows = []
    for h in horizons:
        col = f"fwd_{h}"
        if col not in observations.columns:
            continue

        daily: List[float] = []
        for _date, g in observations.groupby("date"):
            usable = g[[signal_col, col]].dropna()
            if len(usable) < min_names:
                continue
            ic = _spearman(usable[signal_col], usable[col])
            if np.isfinite(ic):
                daily.append(ic)

        if len(daily) < 20:
            continue
        series = pd.Series(daily)
        mean_ic = float(series.mean())
        # t over dates: each date is one observation, so this is far more
        # honest than a t-stat over overlapping pooled returns.
        t_stat = float(mean_ic / (series.std(ddof=1) / np.sqrt(len(series)))) if series.std(ddof=1) else np.nan
        rows.append({
            "signal": signal_col,
            "horizon_days": h,
            "dates": len(series),
            "mean_ic": mean_ic,
            "median_ic": float(series.median()),
            "ic_std": float(series.std(ddof=1)),
            "t_stat": t_stat,
            "pct_positive": float((series > 0).mean()),
            # Information ratio of the IC series - the usual "IC IR".
            "ic_ir": float(mean_ic / series.std(ddof=1)) if series.std(ddof=1) else np.nan,
        })
    return pd.DataFrame(rows)


def quantile_spread(
    observations: pd.DataFrame,
    signal_col: str = "score",
    horizons: Sequence[int] = (5, 10, 20, 60),
    quantiles: int = 5,
    min_names: int = 10,
) -> pd.DataFrame:
    """Top-minus-bottom quantile forward return, computed within each date."""
    if observations is None or observations.empty or signal_col not in observations:
        return pd.DataFrame()

    rows = []
    for h in horizons:
        col = f"fwd_{h}"
        if col not in observations.columns:
            continue

        spreads: List[float] = []
        top_returns: List[float] = []
        bottom_returns: List[float] = []

        for _date, g in observations.groupby("date"):
            usable = g[[signal_col, col]].dropna()
            if len(usable) < min_names:
                continue
            try:
                buckets = pd.qcut(usable[signal_col], quantiles, labels=False,
                                  duplicates="drop")
            except ValueError:
                continue
            if buckets.nunique() < 2:
                continue
            top = usable.loc[buckets == buckets.max(), col].mean()
            bottom = usable.loc[buckets == buckets.min(), col].mean()
            if np.isfinite(top) and np.isfinite(bottom):
                spreads.append(top - bottom)
                top_returns.append(top)
                bottom_returns.append(bottom)

        if len(spreads) < 20:
            continue
        series = pd.Series(spreads)
        t_stat = float(series.mean() / (series.std(ddof=1) / np.sqrt(len(series)))) \
            if series.std(ddof=1) else np.nan
        rows.append({
            "signal": signal_col,
            "horizon_days": h,
            "dates": len(series),
            "top_mean": float(np.mean(top_returns)),
            "bottom_mean": float(np.mean(bottom_returns)),
            "spread": float(series.mean()),
            "t_stat": t_stat,
            "pct_positive": float((series > 0).mean()),
        })
    return pd.DataFrame(rows)


def component_scan(
    observations: pd.DataFrame,
    horizons: Sequence[int] = (20,),
    prefix: str = "c_",
    min_names: int = 8,
) -> pd.DataFrame:
    """Rank IC for every component column, so each is judged on its own.

    A composite can be flat while its parts are individually informative in
    opposite directions - that cancellation is invisible until you look here.
    """
    if observations is None or observations.empty:
        return pd.DataFrame()

    components = [c for c in observations.columns if c.startswith(prefix)]
    frames = []
    for component in components:
        ic = rank_ic(observations, signal_col=component, horizons=horizons,
                     min_names=min_names)
        if not ic.empty:
            frames.append(ic)
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["signal"] = out["signal"].str.replace(f"^{prefix}", "", regex=True)
    return out.sort_values(["horizon_days", "mean_ic"], ascending=[True, False]).reset_index(
        drop=True
    )


def split_sample(observations: pd.DataFrame, fraction: float = 0.5) -> tuple:
    """Chronological train/test split.

    Splitting by *date* rather than randomly is essential: a random split leaks
    the future into the training set through overlapping forward windows and
    through the simple fact that neighbouring observations of the same ticker
    are nearly identical.
    """
    if observations is None or observations.empty:
        return observations, observations
    cutoff = observations["date"].quantile(fraction)
    train = observations[observations["date"] < cutoff].copy()
    test = observations[observations["date"] >= cutoff].copy()
    return train, test


def evaluate_signal(
    observations: pd.DataFrame,
    signal_col: str = "score",
    horizons: Sequence[int] = (5, 10, 20, 60),
    label: str = "",
) -> Dict[str, pd.DataFrame]:
    return {
        "label": label,
        "rank_ic": rank_ic(observations, signal_col, horizons),
        "quantile_spread": quantile_spread(observations, signal_col, horizons),
    }


def render_ic(df: pd.DataFrame, title: str, width: int = 78) -> str:
    out = [title, "-" * width]
    if df is None or df.empty:
        out.append("  (insufficient data)")
        return "\n".join(out)
    out.append(f"  {'signal':<24}{'horizon':>8}{'mean IC':>10}{'t':>8}"
               f"{'IC IR':>8}{'% > 0':>8}{'dates':>8}")
    for _, r in df.iterrows():
        out.append(
            f"  {str(r['signal']):<24}{int(r['horizon_days']):>7}d"
            f"{r['mean_ic']:>10.4f}{r['t_stat']:>8.2f}{r['ic_ir']:>8.3f}"
            f"{r['pct_positive']:>7.1%}{int(r['dates']):>8,}"
        )
    return "\n".join(out)


def render_spread(df: pd.DataFrame, title: str, width: int = 78) -> str:
    out = [title, "-" * width]
    if df is None or df.empty:
        out.append("  (insufficient data)")
        return "\n".join(out)
    out.append(f"  {'horizon':>8}{'top':>10}{'bottom':>10}{'spread':>10}"
               f"{'t':>8}{'% > 0':>8}{'dates':>8}")
    for _, r in df.iterrows():
        out.append(
            f"  {int(r['horizon_days']):>7}d{r['top_mean']:>9.2%}{r['bottom_mean']:>10.2%}"
            f"{r['spread']:>+10.2%}{r['t_stat']:>8.2f}{r['pct_positive']:>7.1%}"
            f"{int(r['dates']):>8,}"
        )
    return "\n".join(out)
