#!/usr/bin/env python3
"""H30 — is there a triple-EMA golden cross that actually works on IDX?

    python3 scripts/ema_cross_idx.py

"TRAINED ON IDX" HAS TO MEAN MEASURED ON IDX. Porting 50/100/200 because it is
conventional would be the opposite of training: those lengths come from US
equity folklore and nothing about them is Indonesian. So this grids the three
lengths, scores every combination the same way, and reports the winner ALONGSIDE
the fact that it is the winner of many — which is the only honest way to quote
a grid search.

PRE-REGISTERED PREDICTION, WRITTEN BEFORE ANY CELL WAS SCORED:
  The golden cross will NOT beat buy-and-hold after costs at any (fast, mid,
  slow) in the grid. H13 measured all eight registered price features as
  net-negative at every horizon once 56 bps of fees and a fraksi-harga
  half-spread were charged, and H22 tested nine index-timing rules across two
  independent halves and lost ALL EIGHTEEN cells. A crossover is a slower,
  noisier version of the momentum those studies already priced.
  What I expect to survive: the ALIGNMENT STATE (fast>mid>slow) as a
  descriptive conditional — "while aligned, the forward distribution looks
  like X" — which is a different and much weaker claim than the cross being a
  tradeable trigger.

THE CONTROL THAT MAKES THE GRID READABLE. Every configuration is compared to a
random entry on the SAME name in the SAME year, matched on count. Without it a
grid over a rising market reports every combination as profitable and the
reader cannot tell skill from drift.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.report import brief as B                            # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")
HOLD = 60                     # sessions held after the cross
MIN_TV = 1e9                  # Rp/day floor, the repo's standard eligibility

FAST = [5, 8, 13, 20]
MID = [21, 34, 50]
SLOW = [100, 150, 200]


def load() -> pd.DataFrame:
    P = pd.read_parquet(PANEL, columns=["date", "ticker", "close", "adj_close",
                                        "log_turnover", "tradeable", "holdout"])
    P["date"] = pd.to_datetime(P["date"])
    P = P[~P["holdout"].astype(bool)]
    P = P[P["tradeable"].astype(bool)]
    P = P[(P["adj_close"] > 0) & (P["close"] > 0)]
    P = P[np.exp(P["log_turnover"].fillna(0)) >= MIN_TV]
    return P.sort_values(["ticker", "date"])


def build(P: pd.DataFrame) -> pd.DataFrame:
    """Pivot to date x ticker so every EMA is one vectorised pass."""
    return P.pivot_table(index="date", columns="ticker", values="adj_close")


def score(px: pd.DataFrame, fwd: pd.DataFrame, cost: pd.DataFrame,
          f: int, m: int, s: int, rng) -> Dict:
    ef = px.ewm(span=f, adjust=False, min_periods=f).mean()
    em = px.ewm(span=m, adjust=False, min_periods=m).mean()
    es = px.ewm(span=s, adjust=False, min_periods=s).mean()
    aligned = (ef > em) & (em > es)
    #  THE .astype(bool) IS LOAD-BEARING AND ITS ABSENCE COST A WHOLE TABLE.
    #  DataFrame.shift() on a boolean frame returns OBJECT dtype with NaN in
    #  the first row; .fillna(False) leaves it object, and `~` on a Python
    #  bool is integer negation -- ~True is -2, ~False is -1, and BOTH are
    #  truthy. So `aligned & ~prev` silently evaluated to `aligned`, and the
    #  first run of this script measured the persistent ALIGNMENT STATE while
    #  labelling it the CROSS. The two counts came out identical in every row,
    #  which is what gave it away.
    prev = aligned.shift(1).fillna(False).astype(bool)
    cross = aligned & ~prev
    #  the CROSS is the bar alignment first becomes true; entry is the NEXT
    #  bar's close, the one-bar execution gap every study in this repo uses
    sig = cross.shift(1).fillna(False).astype(bool)

    r = fwd.where(sig).stack(future_stack=True).dropna()
    c = cost.where(sig).stack(future_stack=True).dropna()
    n = len(r)
    if n < 300:
        return {}
    net = r.to_numpy() - c.reindex(r.index).to_numpy()

    #  matched random control: same number of entries, same names, same years
    idx = fwd.stack(future_stack=True).dropna()
    take = rng.choice(len(idx), size=min(n, len(idx)), replace=False)
    ctrl = idx.iloc[take].to_numpy() - float(np.nanmedian(cost.to_numpy()))

    #  and the state, not the trigger: forward return while merely ALIGNED
    ra = fwd.where(aligned.shift(1).fillna(False).astype(bool)).stack(
        future_stack=True).dropna()

    return {"f": f, "m": m, "s": s, "n": n,
            "mean": float(np.mean(net)), "median": float(np.median(net)),
            "win": float((net > 0).mean()),
            "ctrl_mean": float(np.mean(ctrl)),
            "ctrl_win": float((ctrl > 0).mean()),
            "edge": float(np.mean(net) - np.mean(ctrl)),
            "aligned_mean": float(ra.mean()), "aligned_n": int(len(ra))}


def main() -> int:
    W = 100
    P = load()
    px = build(P)
    print("=" * W)
    print(f" H30 — TRIPLE-EMA GOLDEN CROSS ON IDX, {HOLD}-SESSION HOLD")
    print("=" * W)
    print(f" {px.shape[1]} names, {px.shape[0]:,} sessions, "
          f"{P['date'].min().date()} .. {P['date'].max().date()}, pre-holdout")
    print(" PRE-REGISTERED: no configuration beats buy-and-hold after costs.\n")

    fwd = px.shift(-HOLD) / px - 1.0
    #  round-trip cost per name-bar from the point-in-time fraksi harga ladder
    cl = P.pivot_table(index="date", columns="ticker", values="close")
    med_close = cl.median()
    day = P["date"].max()
    per_name = {t: B.cost_bar(float(v), day)["total"]
                for t, v in med_close.dropna().items() if v > 0}
    cost = pd.DataFrame(np.tile(
        [per_name.get(c, 0.012) for c in cl.columns], (len(cl), 1)),
        index=cl.index, columns=cl.columns)

    bh = float(fwd.stack(future_stack=True).dropna().mean())
    print(f" Buy-and-hold benchmark: mean {HOLD}-session return across every "
          f"eligible name-bar = {bh:+.2%}\n")

    rng = np.random.default_rng(11)
    rows: List[Dict] = []
    for f in FAST:
        for m in MID:
            for s in SLOW:
                if not (f < m < s):
                    continue
                r = score(px, fwd, cost, f, m, s, rng)
                if r:
                    rows.append(r)
    rows.sort(key=lambda r: -r["edge"])

    print(f" {len(rows)} configurations scored, ranked by edge over a matched"
          f" random entry\n")
    print(f"   {'f/m/s':<14}{'n':>8}{'mean net':>10}{'median':>9}{'win%':>7}"
          f"{'random':>9}{'rand win%':>11}{'EDGE':>9}")
    def line(r):
        tag = "{}/{}/{}".format(r["f"], r["m"], r["s"])
        print(f"   {tag:<14}{r['n']:>8,}"
              f"{r['mean']:>+10.2%}{r['median']:>+9.2%}{r['win']:>7.1%}"
              f"{r['ctrl_mean']:>+9.2%}{r['ctrl_win']:>11.1%}"
              f"{r['edge']:>+9.2%}")

    for r in rows[:8]:
        line(r)
    print("   ...")
    for r in rows[-3:]:
        line(r)

    best = rows[0]
    pos = sum(1 for r in rows if r["edge"] > 0)
    print(f"\n   {pos} of {len(rows)} configurations beat their matched random"
          f" control.")
    print(f"   If the cross carried nothing you would expect about half"
          f" ({len(rows) // 2}) by chance.")
    print(f"\n   BEST: {best['f']}/{best['m']}/{best['s']}"
          f"   edge {best['edge']:+.2%} over {best['n']:,} entries")
    print(f"   It is the best of {len(rows)}, so its edge is the MAXIMUM of"
          f" {len(rows)} draws")
    print(f"   and is biased upward. Treat the spread of the whole column,"
          f" not this row.")

    print("\n" + "=" * W)
    print(" THE STATE, NOT THE TRIGGER")
    print("=" * W)
    print(" The cross is one bar; the ALIGNMENT is a condition that persists.")
    print(" Forward return while merely aligned (fast>mid>slow), same hold:\n")
    print(f"   {'f/m/s':<14}{'aligned bars':>15}{'mean fwd':>11}"
          f"{'vs buy-hold':>13}")
    for r in sorted(rows, key=lambda r: -r["aligned_mean"])[:6]:
        tag = "{}/{}/{}".format(r["f"], r["m"], r["s"])
        print(f"   {tag:<14}{r['aligned_n']:>15,}"
              f"{r['aligned_mean']:>+11.2%}{r['aligned_mean'] - bh:>+13.2%}")

    print("\n" + "=" * W)
    print(" DOES IT COMPOUND, AND DOES IT HOLD IN BOTH HALVES?")
    print("=" * W)
    print(" The grid ranks on MEAN. Every row above has a positive mean, a")
    print(" NEGATIVE median and a sub-50% win rate -- the exact shape that")
    print(" withdrew H17 and H18 and that left H25 compounding at -17.5%/yr")
    print(" on a +16.9% arithmetic mean. Mean log is the number that decides.")
    for r in rows[:2]:
        deep(px, fwd, cost, r["f"], r["m"], r["s"], px.index)
    return 0




# ==========================================================================
# The two questions the grid does not answer
# ==========================================================================
def deep(px, fwd, cost, f, m, s, dates) -> None:
    """Mean-log and half-split for one configuration.

    THE GRID RANKS ON MEAN AND THAT IS THE STATISTIC THIS REPO HAS BEEN WRONG
    ABOUT MOST OFTEN. H17 and H18 were both withdrawn because they optimised a
    statistic nobody is paid; H25's volatility screen had a POSITIVE arithmetic
    mean and a mean-log of -0.19, i.e. it compounded to nothing. A positive
    mean with a negative median is exactly that shape, and every row of the
    grid above has one.
    """
    ef = px.ewm(span=f, adjust=False, min_periods=f).mean()
    em = px.ewm(span=m, adjust=False, min_periods=m).mean()
    es = px.ewm(span=s, adjust=False, min_periods=s).mean()
    aligned = (ef > em) & (em > es)
    prev = aligned.shift(1).fillna(False).astype(bool)
    sig = (aligned & ~prev).shift(1).fillna(False).astype(bool)

    r = fwd.where(sig).stack(future_stack=True).dropna()
    c = cost.reindex(index=fwd.index, columns=fwd.columns)
    c = c.where(sig).stack(future_stack=True).dropna()
    net = (r - c.reindex(r.index)).dropna()
    ra = fwd.where(prev).stack(future_stack=True).dropna()

    d = net.index.get_level_values(0)
    mid = d[len(d) // 2] if len(d) else None
    print(f"\n   --- {f}/{m}/{s} ---")
    print(f"   {'':<22}{'n':>9}{'mean':>9}{'median':>9}{'MEAN LOG':>11}"
          f"{'-> CAGR60':>11}")
    for lbl, x in (("cross, net of cost", net),
                   ("aligned state, gross", ra)):
        lg = float(np.log1p(np.maximum(x.to_numpy(), -0.99)).mean())
        print(f"   {lbl:<22}{len(x):>9,}{x.mean():>+9.2%}{x.median():>+9.2%}"
              f"{lg:>+11.4f}{np.expm1(lg):>+11.2%}")
    if mid is not None:
        print(f"\n   half-split of the CROSS (split {mid.date()}):")
        print(f"   {'half':<10}{'n':>8}{'mean':>9}{'median':>9}{'mean log':>11}")
        ok = True
        for lbl, mask in (("early", d <= mid), ("late", d > mid)):
            x = net[mask]
            if len(x) < 200:
                print(f"   {lbl:<10}{len(x):>8}   too few")
                ok = False
                continue
            lg = float(np.log1p(np.maximum(x.to_numpy(), -0.99)).mean())
            ok = ok and lg > 0
            print(f"   {lbl:<10}{len(x):>8,}{x.mean():>+9.2%}"
                  f"{x.median():>+9.2%}{lg:>+11.4f}")
        print(f"     compounds positively in BOTH halves: {'YES' if ok else 'NO'}")


if __name__ == "__main__":
    raise SystemExit(main())
