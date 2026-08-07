"""Per-broker behavioural profiles - the "playbook" each desk appears to run.

Two independent views are produced, and they are deliberately different in
kind because each has a weakness the other does not share:

**Campaign profile** (:func:`build_playbook`) aggregates the discrete round
trips found by ``campaigns.py``. Rich and interpretable - median holding
period, where in the range they buy, how much of the move they keep - but it
depends on the zigzag segmentation being sensible, and a desk with few
campaigns yields a small sample.

**Forward-return edge** (:func:`broker_forward_edge`) ignores segmentation
entirely and asks a blunter question: after this broker buys unusually hard,
what does the stock do over the next N days, versus its unconditional
behaviour? No campaign model, no assumptions about entry or exit - just
conditional means with a t-statistic. This is the honest test of whether a
broker's footprint carries information.

When the two disagree, believe the forward-return test.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import BrokerRegistry, Config
from ..data.broker_summary import LOT_SIZE


def _safe_median(series: pd.Series) -> float:
    values = series.dropna()
    return float(values.median()) if len(values) else float("nan")


def build_playbook(campaigns: pd.DataFrame, registry: BrokerRegistry,
                   min_campaigns: int = 3) -> pd.DataFrame:
    """Aggregate campaigns into one behavioural profile per broker."""
    if campaigns is None or campaigns.empty:
        return pd.DataFrame()

    rows: List[dict] = []
    for broker, g in campaigns.groupby("broker"):
        complete = g[g["complete"]]
        if len(g) < min_campaigns:
            continue

        meta = registry.get(broker)
        realized = complete["realized_return"].dropna()
        wins = realized[realized > 0]
        losses = realized[realized <= 0]

        win_rate = float(len(wins) / len(realized)) if len(realized) else float("nan")
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        expectancy = (
            win_rate * avg_win + (1 - win_rate) * avg_loss
            if np.isfinite(win_rate) else float("nan")
        )

        rows.append({
            "broker": broker,
            "name": meta.name,
            "tier": meta.tier,
            "foreign": meta.foreign,
            "campaigns": int(len(g)),
            "completed": int(len(complete)),
            "tickers": int(g["ticker"].nunique()),

            # -- entry behaviour
            "median_acc_days": _safe_median(g["acc_days"]),
            "median_entry_percentile": _safe_median(g["entry_percentile"]),
            "median_stealth_ratio": _safe_median(g["stealth_ratio"]),
            "median_participation": _safe_median(g["participation_pct"]),
            "median_value_idr": _safe_median(g["value_accumulated"]),

            # -- exit behaviour
            "median_holding_days": _safe_median(complete["holding_days"]),
            "median_markup_pct": _safe_median(g["markup_pct"]),
            "median_realized_return": _safe_median(realized),
            "median_exit_capture": _safe_median(complete["exit_capture"]),
            "median_exit_vs_peak_days": _safe_median(complete["exit_vs_peak_days"]),
            "pct_exit_before_peak": (
                float((complete["exit_vs_peak_days"] <= 0).mean())
                if len(complete) else float("nan")
            ),

            # -- outcome
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "median_max_adverse": _safe_median(g["max_adverse_pct"]),

            # -- target distribution feeding the trading plan
            "markup_p25": float(g["markup_pct"].quantile(0.25)),
            "markup_p50": float(g["markup_pct"].quantile(0.50)),
            "markup_p75": float(g["markup_pct"].quantile(0.75)),
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["style"] = out.apply(_classify_style, axis=1)
    return out.sort_values(["tier", "expectancy"], ascending=[True, False]).reset_index(drop=True)


def _classify_style(row: pd.Series) -> str:
    """A short label for how a desk appears to operate."""
    entry = row.get("median_entry_percentile", np.nan)
    acc_days = row.get("median_acc_days", np.nan)
    stealth = row.get("median_stealth_ratio", np.nan)
    capture = row.get("median_exit_capture", np.nan)

    if not np.isfinite(entry) or not np.isfinite(acc_days):
        return "insufficient data"

    if entry < 0.35 and acc_days >= 15:
        base = "patient value accumulator"
    elif entry < 0.35:
        base = "dip buyer"
    elif entry > 0.65 and acc_days < 12:
        base = "momentum chaser"
    elif entry > 0.65:
        base = "breakout accumulator"
    else:
        base = "range trader"

    tags = []
    if np.isfinite(stealth) and stealth < 0.3:
        tags.append("stealth")
    if np.isfinite(capture):
        if capture > 0.7:
            tags.append("sells near the high")
        elif capture < 0.3:
            tags.append("scales out early")
    return f"{base} ({', '.join(tags)})" if tags else base


def broker_forward_edge(
    summary: pd.DataFrame,
    prices: pd.DataFrame,
    registry: BrokerRegistry,
    horizons: Sequence[int] = (5, 10, 20, 60),
    zscore_window: int = 60,
    zscore_threshold: float = 1.5,
    min_events: int = 20,
) -> pd.DataFrame:
    """Does a broker's heavy buying predict returns?

    For each broker, find days where their net buy value was unusually large
    (rolling z-score above ``zscore_threshold``), then compare the stock's
    forward return after those days against the unconditional forward return
    over the same sample. The reported ``t_stat`` is Welch's t on that
    difference.

    Caveats that matter when reading the output:
      * Overlapping forward windows make observations serially correlated, so
        the t-statistic overstates significance. Treat it as a ranking device,
        not a p-value.
      * No survivorship or delisting adjustment is applied.
      * On simulated data this measures the simulator, nothing more.
    """
    if summary is None or summary.empty or prices is None or prices.empty:
        return pd.DataFrame()

    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px = px.sort_values(["ticker", "date"]) if "ticker" in px.columns else px.sort_values("date")

    forward: Dict[int, pd.Series] = {}
    if "ticker" in px.columns:
        for h in horizons:
            forward[h] = px.groupby("ticker")["close"].transform(
                lambda s, hh=h: s.shift(-hh) / s - 1.0
            )
        key_cols = ["ticker", "date"]
    else:
        for h in horizons:
            forward[h] = px["close"].shift(-h) / px["close"] - 1.0
        px["ticker"] = summary["ticker"].iloc[0]
        key_cols = ["ticker", "date"]

    for h in horizons:
        px[f"fwd_{h}"] = forward[h]
    lookup = px.set_index(key_cols)[[f"fwd_{h}" for h in horizons]]

    df = summary.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["net_val"] = df["buy_val"] - df["sell_val"]

    rows: List[dict] = []
    for broker, g in df.groupby("broker"):
        g = g.sort_values(["ticker", "date"])
        # Z-score net value within each ticker so a broker's activity is judged
        # against its own normal size in that name.
        grouped = g.groupby("ticker")["net_val"]
        mean = grouped.transform(lambda s: s.rolling(zscore_window, min_periods=20).mean())
        std = grouped.transform(lambda s: s.rolling(zscore_window, min_periods=20).std(ddof=0))
        g = g.assign(z=(g["net_val"] - mean) / std.replace(0, np.nan))

        joined = g.join(lookup, on=key_cols)
        events = joined[joined["z"] >= zscore_threshold]
        if len(events) < min_events:
            continue

        meta = registry.get(broker)
        record = {
            "broker": broker,
            "name": meta.name,
            "tier": meta.tier,
            "foreign": meta.foreign,
            "events": int(len(events)),
            "tickers": int(events["ticker"].nunique()),
        }
        for h in horizons:
            col = f"fwd_{h}"
            signal = events[col].dropna()
            baseline = joined[col].dropna()
            if len(signal) < min_events or len(baseline) < min_events:
                record[f"edge_{h}d"] = np.nan
                record[f"t_{h}d"] = np.nan
                record[f"hit_{h}d"] = np.nan
                continue
            edge = float(signal.mean() - baseline.mean())
            record[f"edge_{h}d"] = edge
            record[f"hit_{h}d"] = float((signal > 0).mean())
            record[f"t_{h}d"] = _welch_t(signal, baseline)
        rows.append(record)

    if not rows:
        return pd.DataFrame()
    sort_col = f"edge_{horizons[len(horizons) // 2]}d"
    out = pd.DataFrame(rows)
    if sort_col in out.columns:
        out = out.sort_values(sort_col, ascending=False)
    return out.reset_index(drop=True)


def _welch_t(a: pd.Series, b: pd.Series) -> float:
    """Welch's t-statistic for a difference in means, unequal variances."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    denominator = np.sqrt(va / na + vb / nb)
    if denominator == 0 or not np.isfinite(denominator):
        return float("nan")
    return float((a.mean() - b.mean()) / denominator)


def playbook_targets(playbook: pd.DataFrame, broker: str,
                     cfg: Optional[Config] = None) -> List[float]:
    """Profit targets (as fractional gains) learned from a broker's own history.

    Falls back to the configured defaults when that broker has no usable
    campaign record.
    """
    fallback = list(cfg.get("plan.fallback_targets", [0.08, 0.15, 0.25])) if cfg else [0.08, 0.15, 0.25]
    if playbook is None or playbook.empty:
        return fallback
    row = playbook[playbook["broker"] == str(broker).upper()]
    if row.empty:
        return fallback

    row = row.iloc[0]
    p25, p50, p75 = row.get("markup_p25"), row.get("markup_p50"), row.get("markup_p75")
    targets = [t for t in (p25, p50, p75) if t is not None and np.isfinite(t) and t > 0.01]
    if len(targets) < 3:
        return fallback

    # A desk that habitually scales out before the high is telling you to take
    # profit earlier than its raw markup distribution suggests.
    capture = row.get("median_exit_capture", np.nan)
    if np.isfinite(capture) and 0 < capture < 1:
        targets = [t * max(0.55, min(1.0, capture + 0.25)) for t in targets]
    return sorted(round(float(t), 4) for t in targets)


def describe_playbook_row(row: pd.Series) -> str:
    """One-paragraph plain-language reading of a broker's profile."""
    def pct(value, digits=1):
        return "n/a" if not np.isfinite(value) else f"{value * 100:.{digits}f}%"

    def num(value, digits=0):
        return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"

    entry = row.get("median_entry_percentile", np.nan)
    where = (
        "near the lows of" if entry < 0.35 else
        "in the upper part of" if entry > 0.65 else
        "mid-range in"
    ) if np.isfinite(entry) else "somewhere in"

    return (
        f"{row['broker']} ({row['name']}) - {row['style']}. "
        f"Across {int(row['campaigns'])} campaigns in {int(row['tickers'])} names, it builds over "
        f"~{num(row.get('median_acc_days'))} trading days, buying {where} the prior "
        f"{120}-day range (percentile {num(entry * 100 if np.isfinite(entry) else np.nan)}). "
        f"Price typically runs {pct(row.get('median_markup_pct'))} above its entry VWAP before the "
        f"unwind; it realises {pct(row.get('median_realized_return'))}, capturing "
        f"{pct(row.get('median_exit_capture'))} of the available move, and exits "
        f"{num(abs(row.get('median_exit_vs_peak_days', 0)))} days "
        f"{'before' if row.get('median_exit_vs_peak_days', 0) <= 0 else 'after'} the high. "
        f"Win rate {pct(row.get('win_rate'))}, expectancy {pct(row.get('expectancy'), 2)} per campaign."
    )
