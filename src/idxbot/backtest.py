"""Walk-forward evaluation of the accumulation score.

The question this answers: **when the score is high, does the stock actually go
up more than usual?** Without that check, a screener is just an opinion
generator with a number attached.

Method: score every Nth bar across the full available history for each ticker,
join each score to the forward return over several horizons, bucket by score,
and compare each bucket's mean forward return against the unconditional mean.

Everything the scorer sees at bar *i* comes from bars <= *i*, so there is no
look-ahead. What this design does NOT correct for:

  * **Survivorship.** The universe is today's constituent list. Names that
    delisted or fell out of the index are absent, which flatters the result.
  * **Overlapping windows.** A 60-day forward return sampled every 5 days
    produces heavily autocorrelated observations, so t-statistics overstate
    significance. Use them to rank, not to declare a discovery.
  * **Costs.** Bucket returns are gross. The configured round-trip cost is
    reported alongside so you can see which edges survive it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .analytics import accumulation
from .config import Config
from .engine import Engine
from .market import Costs


def run(
    engine: Engine,
    tickers: Sequence[str],
    horizons: Optional[Sequence[int]] = None,
    step: int = 5,
    start_index: int = 300,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Score history across ``tickers`` and evaluate forward returns.

    Returns a dict with ``observations``, ``buckets``, ``by_phase`` and
    ``summary`` frames.
    """
    cfg: Config = engine.cfg
    horizons = list(horizons or cfg.get("backtest.horizons", [5, 10, 20, 60]))

    frames: List[pd.DataFrame] = []
    for i, ticker in enumerate(tickers, 1):
        if verbose:
            print(f"  [{i:>3}/{len(tickers)}] {ticker:<6}", end=" ", flush=True)
        try:
            analysis = engine.analyze(ticker, with_campaigns=False)
        except Exception as exc:
            if verbose:
                print(f"error: {exc}")
            continue
        if analysis is None or len(analysis.bars) <= start_index + max(horizons):
            if verbose:
                print("insufficient history")
            continue

        scored = accumulation.score_series(
            analysis.bars, cfg,
            flow=analysis.flow if not analysis.flow.empty else None,
            ticker=ticker, start_index=start_index, step=step,
            profile=engine.profile,
        )
        if scored.empty:
            if verbose:
                print("no scores")
            continue

        # Attach forward returns from the price series.
        bars = analysis.bars[["date", "close"]].copy()
        for h in horizons:
            bars[f"fwd_{h}"] = bars["close"].shift(-h) / bars["close"] - 1.0
        merged = scored.merge(bars.drop(columns=["close"]), on="date", how="left")
        merged["ticker"] = ticker
        frames.append(merged)
        if verbose:
            print(f"{len(merged):>4} observations")

    if not frames:
        return {"observations": pd.DataFrame(), "buckets": pd.DataFrame(),
                "by_phase": pd.DataFrame(), "summary": pd.DataFrame()}

    observations = pd.concat(frames, ignore_index=True)
    buckets = bucket_returns(observations, horizons, cfg)
    by_phase = phase_returns(observations, horizons, cfg)
    summary = overall_summary(observations, horizons, cfg)
    return {
        "observations": observations,
        "buckets": buckets,
        "by_phase": by_phase,
        "summary": summary,
    }


def bucket_returns(observations: pd.DataFrame, horizons: Sequence[int],
                   cfg: Config) -> pd.DataFrame:
    """Mean forward return by score bucket, versus the unconditional mean."""
    if observations.empty:
        return pd.DataFrame()

    edges = list(cfg.get("backtest.score_buckets", [0, 40, 55, 65, 78, 101]))
    min_samples = int(cfg.get("backtest.min_samples", 30))
    labels = [f"{edges[i]}-{edges[i + 1] - 1}" for i in range(len(edges) - 1)]

    df = observations.copy()
    df["bucket"] = pd.cut(df["score"], bins=edges, labels=labels, right=False,
                          include_lowest=True)

    rows = []
    for bucket, g in df.groupby("bucket", observed=True):
        if len(g) < min_samples:
            continue
        record = {"bucket": str(bucket), "observations": int(len(g)),
                  "tickers": int(g["ticker"].nunique())}
        for h in horizons:
            col = f"fwd_{h}"
            if col not in g:
                continue
            values = g[col].dropna()
            baseline = df[col].dropna()
            if len(values) < min_samples:
                record[f"mean_{h}d"] = np.nan
                continue
            record[f"mean_{h}d"] = float(values.mean())
            record[f"median_{h}d"] = float(values.median())
            record[f"hit_{h}d"] = float((values > 0).mean())
            record[f"excess_{h}d"] = float(values.mean() - baseline.mean())
            record[f"t_{h}d"] = _welch_t(values, baseline)
        rows.append(record)

    return pd.DataFrame(rows)


def phase_returns(observations: pd.DataFrame, horizons: Sequence[int],
                  cfg: Config) -> pd.DataFrame:
    """Forward returns grouped by Wyckoff phase."""
    if observations.empty or "wyckoff_phase" not in observations.columns:
        return pd.DataFrame()
    min_samples = int(cfg.get("backtest.min_samples", 30))

    rows = []
    for phase, g in observations.groupby("wyckoff_phase"):
        if len(g) < min_samples:
            continue
        record = {"phase": phase, "observations": int(len(g))}
        for h in horizons:
            col = f"fwd_{h}"
            values = g[col].dropna() if col in g else pd.Series(dtype=float)
            baseline = observations[col].dropna() if col in observations else pd.Series(dtype=float)
            if len(values) < min_samples:
                record[f"mean_{h}d"] = np.nan
                continue
            record[f"mean_{h}d"] = float(values.mean())
            record[f"hit_{h}d"] = float((values > 0).mean())
            record[f"excess_{h}d"] = float(values.mean() - baseline.mean())
        rows.append(record)
    return pd.DataFrame(rows).sort_values("phase").reset_index(drop=True)


def overall_summary(observations: pd.DataFrame, horizons: Sequence[int],
                    cfg: Config) -> pd.DataFrame:
    """Unconditional baseline plus the signal-threshold cohort, net of costs."""
    if observations.empty:
        return pd.DataFrame()

    threshold = float(cfg.get("accumulation.signal_threshold", 65))
    costs = Costs.from_config(cfg)
    signal = observations[observations["score"] >= threshold]

    rows = []
    for h in horizons:
        col = f"fwd_{h}"
        if col not in observations:
            continue
        base = observations[col].dropna()
        sig = signal[col].dropna()
        rows.append({
            "horizon_days": h,
            "baseline_mean": float(base.mean()) if len(base) else np.nan,
            "baseline_hit": float((base > 0).mean()) if len(base) else np.nan,
            "signal_observations": int(len(sig)),
            "signal_mean": float(sig.mean()) if len(sig) else np.nan,
            "signal_hit": float((sig > 0).mean()) if len(sig) else np.nan,
            "excess": float(sig.mean() - base.mean()) if len(sig) and len(base) else np.nan,
            "signal_mean_net_costs": (
                float(sig.mean() - costs.round_trip_pct) if len(sig) else np.nan
            ),
            "t_stat": _welch_t(sig, base) if len(sig) > 2 and len(base) > 2 else np.nan,
        })
    return pd.DataFrame(rows)


def _welch_t(a: pd.Series, b: pd.Series) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    denominator = np.sqrt(va / na + vb / nb)
    if denominator == 0 or not np.isfinite(denominator):
        return float("nan")
    return float((a.mean() - b.mean()) / denominator)


def render(results: Dict[str, pd.DataFrame], cfg: Config, width: int = 78) -> str:
    """Format backtest results for the terminal."""
    out: List[str] = []
    line = "=" * width
    observations = results.get("observations", pd.DataFrame())

    out.append(line)
    out.append(" ACCUMULATION SCORE - WALK-FORWARD EVALUATION")
    out.append(line)
    if observations.empty:
        out.append(" No observations produced.")
        return "\n".join(out)

    modes = observations["data_mode"].value_counts() if "data_mode" in observations else {}
    out.append(f" observations : {len(observations):,}")
    out.append(f" tickers      : {observations['ticker'].nunique()}")
    out.append(f" period       : {observations['date'].min():%Y-%m-%d} -> "
               f"{observations['date'].max():%Y-%m-%d}")
    for mode, count in dict(modes).items():
        out.append(f" mode         : {mode} ({count:,} observations)")

    costs = Costs.from_config(cfg)
    out.append(f" round-trip cost assumption: {costs.round_trip_pct:.2%}")
    out.append("")

    summary = results.get("summary", pd.DataFrame())
    if not summary.empty:
        out.append(" SIGNAL COHORT vs BASELINE")
        out.append(f" {'horizon':>8} {'baseline':>10} {'signal':>10} {'excess':>9} "
                   f"{'net cost':>10} {'hit':>7} {'t':>7} {'n':>8}")
        for _, r in summary.iterrows():
            out.append(
                f" {int(r['horizon_days']):>7}d {r['baseline_mean']:>9.2%} "
                f"{r['signal_mean']:>9.2%} {r['excess']:>+8.2%} "
                f"{r['signal_mean_net_costs']:>9.2%} {r['signal_hit']:>6.1%} "
                f"{r['t_stat']:>7.2f} {int(r['signal_observations']):>8,}"
            )
        out.append("")

    buckets = results.get("buckets", pd.DataFrame())
    if not buckets.empty:
        horizons = list(cfg.get("backtest.horizons", [5, 10, 20, 60]))
        out.append(" MEAN FORWARD RETURN BY SCORE BUCKET")
        header = f" {'bucket':>10} {'n':>7}" + "".join(f"{str(h) + 'd':>10}" for h in horizons)
        out.append(header)
        for _, r in buckets.iterrows():
            cells = "".join(
                f"{r.get(f'mean_{h}d', np.nan):>9.2%}" + " " if np.isfinite(
                    r.get(f"mean_{h}d", np.nan)) else f"{'-':>10}"
                for h in horizons
            )
            out.append(f" {r['bucket']:>10} {int(r['observations']):>7,}{cells}")
        out.append("")

    by_phase = results.get("by_phase", pd.DataFrame())
    if not by_phase.empty:
        horizons = list(cfg.get("backtest.horizons", [5, 10, 20, 60]))
        out.append(" MEAN FORWARD RETURN BY WYCKOFF PHASE")
        out.append(f" {'phase':>10} {'n':>7}" + "".join(f"{str(h) + 'd':>10}" for h in horizons))
        for _, r in by_phase.iterrows():
            cells = "".join(
                f"{r.get(f'mean_{h}d', np.nan):>9.2%} " if np.isfinite(
                    r.get(f"mean_{h}d", np.nan)) else f"{'-':>10}"
                for h in horizons
            )
            out.append(f" {r['phase']:>10} {int(r['observations']):>7,}{cells}")
        out.append("")

    out.append(" Caveats: no survivorship adjustment; overlapping forward windows inflate")
    out.append(" t-statistics; returns are gross unless the net-cost column is shown.")
    out.append(line)
    return "\n".join(out)
