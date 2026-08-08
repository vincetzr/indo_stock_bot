"""Walk-forward optimisation — the only honest way to "train until optimum".

Tuning parameters against a fixed dataset until the result looks good is not
training. It is fitting the noise in that specific history, and it is the single
most reliable way to manufacture a backtest that dies on contact with live
money. The more parameter combinations tried, the better the winner looks and
the less it means: with 50 candidates, the best one clears a 2-sigma bar by
chance alone.

Walk-forward is the disciplined alternative:

    ┌──── train ────┐┌ test ┐
    2001........2011  2012          pick best params on train, score on test
      ┌──── train ────┐┌ test ┐
      2001........2012  2013        refit, score on the next unseen year
        ┌──── train ────┐┌ test ┐
        2001........2013  2014      ... and so on

Every test slice is scored with parameters chosen *before* that slice existed,
so the concatenated test results are genuinely out-of-sample. The engine's
scorer is already look-ahead-free (unit-tested), and this adds the second half:
no look-ahead in the *parameter choice* either.

Two diagnostics matter as much as the returns:

  * **Parameter stability.** If the winning parameter jumps between refits,
    there is no stable optimum — the search is tracking noise, and whatever it
    picks today tells you nothing about tomorrow.
  * **Optimisation value-add.** Compare walk-forward against simply fixing one
    sensible parameter for the whole period. If they match, the optimisation
    contributed nothing and only added a way to fool yourself.

That second comparison is the one people skip, and it is usually the one that
kills the idea.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Candidate weight sets.
#
# A curated list, not a dense grid. Every extra candidate raises the chance the
# winner is lucky rather than good, so the search is kept deliberately narrow
# and each entry encodes a *thesis* worth testing rather than a grid point:
# each of the four trend components alone, the profile currently shipped, and a
# few blends that lean on whichever component the component scan liked.
# ---------------------------------------------------------------------------
WEIGHT_CANDIDATES: Dict[str, Dict[str, float]] = {
    "near_high only":     {"near_high": 1.0},
    "momentum only":      {"momentum": 1.0},
    "trend only":         {"trend_persistence": 1.0},
    "rel_strength only":  {"relative_strength": 1.0},
    "shipped momentum":   {"momentum": 0.30, "relative_strength": 0.25,
                           "trend_persistence": 0.25, "near_high": 0.20},
    "equal trend-4":      {"momentum": 0.25, "relative_strength": 0.25,
                           "trend_persistence": 0.25, "near_high": 0.25},
    "near_high heavy":    {"near_high": 0.50, "momentum": 0.20,
                           "trend_persistence": 0.20, "relative_strength": 0.10},
    "near_high + trend":  {"near_high": 0.50, "trend_persistence": 0.50},
    "contrarian":         {"volume_dryup": 0.40, "range_compression": 0.30,
                           "obv_divergence": 0.30},
    "equal all-7":        {"momentum": 1 / 7, "relative_strength": 1 / 7,
                           "trend_persistence": 1 / 7, "near_high": 1 / 7,
                           "volume_dryup": 1 / 7, "range_compression": 1 / 7,
                           "obv_divergence": 1 / 7},
}


def blend(observations: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Rebuild the composite score from stored components.

    ``score`` in the engine is a normalised weighted sum of the components, so
    re-weighting is exact arithmetic on columns that were already computed
    point-in-time. Nothing is refetched and no future data can leak in through
    the back door: every ``c_*`` value was produced from bars up to its own date.
    """
    usable: Dict[str, float] = {}
    for name, raw in weights.items():
        if f"c_{name}" not in observations.columns:
            continue  # skips label keys like "profile" and broker-only components
        try:
            w = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if w > 0:
            usable[name] = w
    if not usable:
        return pd.Series(np.nan, index=observations.index)
    total = sum(usable.values())
    out = None
    for name, w in usable.items():
        term = observations[f"c_{name}"].astype(float) * (w / total)
        out = term if out is None else out + term
    return out * 100.0


def nonoverlapping(dates: Sequence[pd.Timestamp], horizon_days: int) -> List[pd.Timestamp]:
    """Thin a date list so no two forward windows overlap.

    Sampling a 60-day forward return every 5 days reuses the same future twelve
    times over. The returns look like twelve observations but carry roughly one
    observation's worth of information, which inflates every t-statistic built
    on them by about sqrt(12). Spacing the rebalances by the holding period is
    the fix, and it also happens to be what actually trading the rule looks like.
    """
    if not len(dates):
        return []
    ordered = sorted(pd.to_datetime(list(dates)))
    gap = pd.Timedelta(days=int(round(horizon_days * 7 / 5)))  # trading -> calendar
    kept = [ordered[0]]
    for d in ordered[1:]:
        if d - kept[-1] >= gap:
            kept.append(d)
    return kept


def portfolio_evaluator(
    horizon: int = 60,
    top_n: int = 5,
    min_names: int = 20,
    overlap: bool = False,
) -> Callable[[pd.DataFrame, Dict[str, object]], pd.Series]:
    """Build the ``evaluate`` callable: 'what did the top N names return?'

    Rank IC would also work as an objective, but it measures whether the whole
    ranking is ordered correctly, and nobody trades the whole ranking. The top-N
    forward return is the thing the money actually experiences, so that is what
    gets optimised and what gets reported.
    """
    fwd = f"fwd_{horizon}"

    def evaluate(slice_df: pd.DataFrame, params: Dict[str, object]) -> pd.Series:
        if slice_df is None or slice_df.empty or fwd not in slice_df.columns:
            return pd.Series(dtype=float)
        df = slice_df.copy()
        df["_s"] = blend(df, params)  # type: ignore[arg-type]
        df = df.dropna(subset=["_s", fwd])
        if df.empty:
            return pd.Series(dtype=float)

        dates = df["date"].unique()
        if not overlap:
            dates = nonoverlapping(dates, horizon)
        df = df[df["date"].isin(pd.to_datetime(list(dates)))]

        out: Dict[pd.Timestamp, float] = {}
        for date, g in df.groupby("date"):
            if len(g) < min_names:
                continue  # too thin a cross-section to call it a selection
            picks = g.nlargest(top_n, "_s")
            out[date] = float(picks[fwd].mean())
        return pd.Series(out).sort_index()

    return evaluate


def benchmark(
    observations: pd.DataFrame,
    horizon: int = 60,
    min_names: int = 20,
    overlap: bool = False,
) -> pd.Series:
    """Equal-weight every eligible name — the 'no selection at all' baseline.

    Any strategy that cannot beat this is not picking stocks, it is picking
    exposure, and exposure is available for free.
    """
    fwd = f"fwd_{horizon}"
    df = observations.dropna(subset=[fwd])
    dates = df["date"].unique()
    if not overlap:
        dates = nonoverlapping(dates, horizon)
    df = df[df["date"].isin(pd.to_datetime(list(dates)))]
    out = {d: float(g[fwd].mean()) for d, g in df.groupby("date") if len(g) >= min_names}
    return pd.Series(out).sort_index()


def _label(params: Dict[str, object]) -> str:
    """Prefer a human label over the raw weight vector when one was supplied."""
    if "profile" in params:
        return str(params["profile"])
    return ", ".join(f"{k}={v}" for k, v in params.items())


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    chosen: Dict[str, object] = field(default_factory=dict)
    train_score: float = np.nan
    test_return: float = np.nan
    test_periods: int = 0


@dataclass
class WalkForwardResult:
    folds: List[Fold] = field(default_factory=list)
    oos_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    fixed_returns: Dict[str, pd.Series] = field(default_factory=dict)
    benchmark_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    n_candidates: int = 0

    @property
    def empty(self) -> bool:
        return not self.folds


def make_folds(
    dates: Sequence[pd.Timestamp],
    train_years: float = 8.0,
    test_years: float = 1.0,
    min_train_periods: int = 20,
) -> List[Fold]:
    """Expanding-window folds: train grows, test rolls forward one slice."""
    if len(dates) < 2:
        return []
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(dates))))
    start, end = dates[0], dates[-1]

    folds: List[Fold] = []
    test_start = start + pd.DateOffset(years=int(train_years))
    while test_start < end:
        test_end = min(test_start + pd.DateOffset(years=int(test_years)), end)
        n_train = int(((dates >= start) & (dates < test_start)).sum())
        n_test = int(((dates >= test_start) & (dates < test_end)).sum())
        if n_train >= min_train_periods and n_test > 0:
            folds.append(Fold(train_start=start, train_end=test_start,
                              test_start=test_start, test_end=test_end))
        test_start = test_end
    return folds


def run(
    observations: pd.DataFrame,
    param_grid: List[Dict[str, object]],
    evaluate: Callable[[pd.DataFrame, Dict[str, object]], pd.Series],
    train_years: float = 8.0,
    test_years: float = 1.0,
    objective: str = "mean",
    fixed_params: Optional[Dict[str, Dict[str, object]]] = None,
    verbose: bool = True,
) -> WalkForwardResult:
    """Walk-forward parameter selection.

    ``evaluate(slice, params) -> per-period returns`` is supplied by the caller,
    so this module stays agnostic about what a "parameter" is.

    ``objective`` picks the winner on the training slice: ``mean`` maximises
    average return, ``sharpe`` reward per unit of variability, ``median``
    resists a single lucky period dominating the choice.
    """
    if observations is None or observations.empty or not param_grid:
        return WalkForwardResult()

    df = observations.copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())
    folds = make_folds(dates, train_years, test_years)
    if not folds:
        return WalkForwardResult()

    def score_of(returns: pd.Series) -> float:
        r = returns.dropna()
        if len(r) < 3:
            return -np.inf
        if objective == "sharpe":
            sd = r.std(ddof=1)
            return float(r.mean() / sd) if sd > 0 else -np.inf
        if objective == "median":
            return float(r.median())
        return float(r.mean())

    oos_chunks: List[pd.Series] = []
    for i, fold in enumerate(folds, 1):
        train = df[(df["date"] >= fold.train_start) & (df["date"] < fold.train_end)]
        test = df[(df["date"] >= fold.test_start) & (df["date"] < fold.test_end)]
        if train.empty or test.empty:
            continue

        best_params, best_score = None, -np.inf
        for params in param_grid:
            s = score_of(evaluate(train, params))
            if s > best_score:
                best_params, best_score = params, s

        if best_params is None:
            continue

        # The chosen parameters never saw this slice.
        test_returns = evaluate(test, best_params).dropna()
        fold.chosen = dict(best_params)
        fold.train_score = float(best_score)
        fold.test_return = float(test_returns.mean()) if len(test_returns) else np.nan
        fold.test_periods = int(len(test_returns))
        if len(test_returns):
            oos_chunks.append(test_returns)

        if verbose:
            label = _label(best_params)
            print(f"  fold {i:>2}  train->{fold.train_end:%Y-%m}  "
                  f"test {fold.test_start:%Y-%m}..{fold.test_end:%Y-%m}  "
                  f"chose [{label}]  oos {fold.test_return:+.2%}  n={fold.test_periods}")

    result = WalkForwardResult(folds=folds, n_candidates=len(param_grid))
    if oos_chunks:
        result.oos_returns = pd.concat(oos_chunks).reset_index(drop=True)

    # The comparison that decides whether the optimisation earned its keep:
    # each fixed parameter set, applied over the same out-of-sample span.
    if fixed_params:
        span = df[df["date"] >= folds[0].test_start]
        for name, params in fixed_params.items():
            result.fixed_returns[name] = evaluate(span, params).dropna().reset_index(drop=True)
    return result


def attach_benchmark(result: WalkForwardResult, observations: pd.DataFrame,
                     horizon: int, min_names: int, overlap: bool = False) -> None:
    """Score the equal-weight baseline over the same out-of-sample span."""
    if result.empty:
        return
    df = observations.copy()
    df["date"] = pd.to_datetime(df["date"])
    span = df[df["date"] >= result.folds[0].test_start]
    result.benchmark_returns = benchmark(span, horizon, min_names, overlap).reset_index(drop=True)


def stability(result: WalkForwardResult) -> pd.DataFrame:
    """How often each parameter set won, and whether the choice is stable."""
    if result.empty:
        return pd.DataFrame()
    rows = []
    for fold in result.folds:
        if not fold.chosen:
            continue
        rows.append({"test_start": fold.test_start,
                     "params": _label(fold.chosen),
                     "oos": fold.test_return})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    counts = df.groupby("params").agg(
        wins=("params", "size"),
        mean_oos=("oos", "mean"),
    ).sort_values("wins", ascending=False).reset_index()
    counts["share"] = counts["wins"] / counts["wins"].sum()
    return counts


def render(result: WalkForwardResult, width: int = 78) -> str:
    line = "=" * width
    out = [line, " WALK-FORWARD OPTIMISATION", line]

    if result.empty:
        out.append(" Not enough history to build a single train/test fold.")
        return "\n".join(out + [line])

    oos = result.oos_returns
    out.append(f" folds          : {len(result.folds)}")
    out.append(f" candidates/fold: {result.n_candidates}")
    out.append(f" OOS periods    : {len(oos)}")
    if len(oos):
        eq = (1 + oos).cumprod()
        dd = (eq / eq.cummax() - 1).min()
        out.append(f" OOS mean/period: {oos.mean():+.2%}")
        out.append(f" OOS median     : {oos.median():+.2%}")
        out.append(f" OOS win rate   : {(oos > 0).mean():.0%}")
        out.append(f" OOS total      : {eq.iloc[-1] - 1:+.1%}")
        out.append(f" OOS maxDD      : {dd:.1%}")
        sd = oos.std(ddof=1)
        if sd > 0:
            t = oos.mean() / (sd / np.sqrt(len(oos)))
            out.append(f" t-stat         : {t:.2f}")

    out.append("")
    out.append(" PARAMETER STABILITY")
    st = stability(result)
    if st.empty:
        out.append("   (no selections recorded)")
    else:
        for _, r in st.iterrows():
            out.append(f"   {r['params']:<40} won {int(r['wins'])}/{len(result.folds)} "
                       f"folds ({r['share']:.0%})   oos {r['mean_oos']:+.2%}")
        if st.iloc[0]["share"] < 0.5:
            out.append("")
            out.append("   The winner changes between refits. That is the signature of")
            out.append("   fitting noise: there is no stable optimum to find, and the")
            out.append("   parameter chosen today says little about tomorrow.")

    if result.fixed_returns or len(result.benchmark_returns):
        out.append("")
        out.append(" DID THE OPTIMISATION ADD ANYTHING?")
        out.append(f"   {'strategy':<34}{'mean':>10}{'median':>10}{'win':>8}")
        if len(oos):
            out.append(f"   {'walk-forward (optimised)':<34}{oos.mean():>9.2%}"
                       f"{oos.median():>10.2%}{(oos > 0).mean():>8.0%}")
        best_fixed = None
        for name, series in result.fixed_returns.items():
            if not len(series):
                continue
            out.append(f"   {('fixed: ' + name):<34}{series.mean():>9.2%}"
                       f"{series.median():>10.2%}{(series > 0).mean():>8.0%}")
            if best_fixed is None or series.mean() > best_fixed[1]:
                best_fixed = (name, float(series.mean()))

        bench = result.benchmark_returns
        if len(bench):
            out.append(f"   {'--- equal-weight universe':<34}{bench.mean():>9.2%}"
                       f"{bench.median():>10.2%}{(bench > 0).mean():>8.0%}")

        if best_fixed and len(oos) and best_fixed[1] >= oos.mean():
            out.append("")
            out.append(f"   Fixing '{best_fixed[0]}' for the whole period matched or beat")
            out.append("   the optimisation. The search added no value - it only added")
            out.append("   a way to fool yourself.")
        if len(bench) and len(oos) and oos.mean() <= bench.mean():
            out.append("")
            out.append("   The selection did not beat holding every eligible name. That is")
            out.append("   not stock picking, it is buying exposure - and exposure is free.")

    out.append(line)
    return "\n".join(out)
