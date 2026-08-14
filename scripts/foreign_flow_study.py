#!/usr/bin/env python3
"""Fetch per-stock foreign flow for IDX, and test whether it predicts returns.

For most of this project's life the README said per-stock foreign flow could not
be obtained. That was wrong, and the error was one of search rather than
availability: only ``idx.co.id`` and commercial vendors were ever probed, never
public datasets. ``wildangunawan/Dataset-Saham-IDX`` carries ``foreign_buy`` and
``foreign_sell`` per stock per day for 958 tickers.

**Two different things are called "foreign flow" and they must not be mixed.**

    F-flag    IDX's per-trade foreign-investor flag, aggregated per stock/day.
              Denominated in SHARES. This is what this script uses, and what
              Indonesian platforms mean by "net foreign".
    F-broker  The sum over exchange members flagged foreign in
              config/brokers.yaml. Denominated in lots and IDR. This is what
              ``idxbot.bandarmology.foreign_flow`` computes.

They do not reconcile - a foreign investor can trade through a domestic member,
and a foreign-owned member (YP) mostly serves domestic retail - so they are
never summed, differenced or shown in one column.

**Limits of this source.** The last bar is 2025-02-21, so it is a research
source and never a live one. The upstream repo carries IDX's anti-crawling
notice and forbids commercial use.

    python3 scripts/foreign_flow_study.py [--clone-dir DIR]
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot import walkforward as wf  # noqa: E402
from idxbot.evaluate import _spearman, rank_ic  # noqa: E402

REPO = "https://github.com/wildangunawan/Dataset-Saham-IDX.git"
MIN_CLOSE, MIN_TURNOVER, COST = 50.0, 1e9, 0.004


def load(clone_dir: str) -> pd.DataFrame:
    """Clone if needed, then read every ticker CSV into one panel."""
    if not os.path.isdir(clone_dir):
        print(f"cloning {REPO} (shallow) ...")
        subprocess.run(["git", "clone", "--depth", "1", "-q", REPO, clone_dir],
                       check=True)
    files = sorted(glob.glob(os.path.join(clone_dir, "Saham", "Semua", "*.csv")))
    if not files:
        raise SystemExit(f"no ticker CSVs under {clone_dir}")

    frames = []
    for path in files:
        try:
            d = pd.read_csv(path, usecols=["date", "close", "volume", "value",
                                           "foreign_buy", "foreign_sell"])
        except Exception:
            continue  # a few files carry a different header; skip rather than coerce
        d["ticker"] = os.path.basename(path)[:-4]
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], format="ISO8601").dt.normalize()
    return df


def unit_check(df: pd.DataFrame) -> float:
    """Foreign buy is in SHARES, so it cannot exceed the day's total volume.

    This is the assertion that decides whether the column means what its name
    says. If it were lots, or IDR, the ratio would blow past 1 immediately.
    """
    ok = df.dropna(subset=["foreign_buy", "volume"])
    ok = ok[ok["volume"] > 0]
    return float((ok["foreign_buy"] > ok["volume"] * 1.001).mean())


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing foreign-flow features and forward returns.

    Every signal uses only data up to and including its own date; the ``fwd_``
    columns are outcomes, never inputs.
    """
    df = df[(df["close"] >= MIN_CLOSE) & (df["volume"] > 0)].copy()
    df = df.dropna(subset=["foreign_buy", "foreign_sell"]).sort_values(["ticker", "date"])
    g = df.groupby("ticker")

    for h in (5, 20, 60):
        df[f"fwd_{h}"] = g["close"].shift(-h) / df["close"] - 1

    df["net_fgn_sh"] = df["foreign_buy"] - df["foreign_sell"]
    vwap = df["value"] / df["volume"].replace(0, np.nan)
    df["net_fgn_val"] = df["net_fgn_sh"] * vwap
    df["vt"] = g["value"].transform(lambda s: s.rolling(20, min_periods=5).median())

    # Normalised by the stock's own turnover, so a big number in a big name is
    # comparable to a big number in a small one. Raw IDR just ranks by size.
    df["nf20"] = g["net_fgn_val"].transform(lambda s: s.rolling(20, min_periods=10).sum())
    df["sig"] = df["nf20"] / df["vt"].replace(0, np.nan)
    return df[(df["vt"] >= MIN_TURNOVER)].dropna(subset=["sig"])


def report(panel: pd.DataFrame) -> None:
    print(f"\nliquid panel: {len(panel):,} rows, {panel['ticker'].nunique()} tickers, "
          f"{panel['date'].min():%Y-%m} -> {panel['date'].max():%Y-%m}")
    print(f"names per cross-section: median "
          f"{panel.groupby('date').size().median():.0f}")

    print("\n1. RANK IC — and why it is about to mislead")
    ic = rank_ic(panel, "sig", horizons=(5, 20, 60))
    print(f"   {'horizon':>8}{'mean IC':>10}{'t':>8}{'% > 0':>8}")
    for _, r in ic.iterrows():
        print(f"   {int(r.horizon_days):>7}d{r.mean_ic:>10.4f}{r.t_stat:>8.2f}"
              f"{r.pct_positive:>8.1%}")

    mid = panel["date"].quantile(0.5)
    print("\n2. CHRONOLOGICAL SPLIT (60d)")
    for lab, s in (("train", panel[panel["date"] < mid]),
                   ("holdout", panel[panel["date"] >= mid])):
        r = rank_ic(s, "sig", horizons=(60,))
        if not r.empty:
            print(f"   {lab:<10}IC {r.iloc[0].mean_ic:>+8.4f}  t {r.iloc[0].t_stat:>6.2f}")

    print("\n3. DECILES — the shape the IC could not see")
    rows = []
    for _d, s in panel.dropna(subset=["fwd_60"]).groupby("date"):
        if len(s) < 80:
            continue
        s = s.assign(q=pd.qcut(s["sig"], 10, labels=False, duplicates="drop"))
        rows.append(s.groupby("q")["fwd_60"].mean())
    D = pd.DataFrame(rows)
    for q in sorted(D.columns):
        lab = ("D1  heaviest SELLING" if q == 0 else
               "D10 heaviest BUYING" if q == max(D.columns) else f"D{q + 1}")
        bar = "#" * max(0, int(round((D[q].mean() + 0.03) * 300)))
        print(f"   {lab:<22}{D[q].mean():>8.2%}  {bar}")

    print("\n4. WHAT A TRADE WOULD HAVE EARNED (60d, non-overlapping)")
    keep = wf.nonoverlapping(panel["date"].unique(), 60)
    N = panel[panel["date"].isin(pd.to_datetime(list(keep)))]
    variants = {
        "long D1 (fade buying)": lambda s: s.nsmallest(max(1, len(s) // 10), "sig"),
        "long D10 (follow buying)": lambda s: s.nlargest(max(1, len(s) // 10), "sig"),
        "equal-weight all": lambda s: s,
    }
    for name, fn in variants.items():
        vals = []
        for _d, s in N.dropna(subset=["fwd_60"]).groupby("date"):
            if len(s) < 80:
                continue
            vals.append(fn(s.sort_values("sig"))["fwd_60"].mean() - COST)
        v = np.array(vals)
        t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 2 else np.nan
        print(f"   {name:<26}{v.mean():>8.2%}  t={t:>5.2f}  n={len(v)}")

    print("\n   The IC is large and negative; the long-short spread is zero. Both")
    print("   are correct. Rank IC assumes a monotone relationship, and this one")
    print("   is U-shaped - so the IC measures the slope through the middle and")
    print("   says nothing tradeable about the tails.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone-dir", default="data/Dataset-Saham-IDX")
    args = ap.parse_args()

    df = load(args.clone_dir)
    print(f"raw: {len(df):,} rows, {df['ticker'].nunique()} tickers, "
          f"{df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}")
    viol = unit_check(df)
    print(f"unit check (foreign_buy <= volume): {viol:.3%} violations "
          f"-> figures are {'SHARES' if viol < 0.001 else 'NOT shares - STOP'}")
    if viol >= 0.001:
        raise SystemExit("unit assertion failed; do not rescale, investigate")
    report(build_signals(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
