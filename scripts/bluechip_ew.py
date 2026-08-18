#!/usr/bin/env python3
"""Own the large caps - but not the ones that are falling.

The grid in ``bluechip.py`` searches selections: rank the universe, hold the best
few. On the whole exchange that is the right shape, because the ranking is where
the edge is. On large caps the quick run said the ranking is worth less than the
universe itself, which points at a different strategy shape entirely:

    hold EVERY large cap whose gate is on, equally, and hold cash for the rest

No ranking, no top-N, no lookback. The only decision is whether each name is
currently in an uptrend by the same bounded-lag or moving-average test used
everywhere else. It is the portfolio form of the annotated chart - out of the
ones that are heading down, in the ones that are heading up - applied across
thirty-odd names instead of one.

Costs are charged on the weight actually traded each day, which for a gated
equal-weight book is small but not zero: the gate flips a few names a week and
each flip is a full round trip on that name's slice.

    python3 scripts/bluechip_ew.py [--universe-size 40]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from bluechip import equal_weight, fixed_universe, pit_universe   # noqa: E402
from optimize_consistent import CAP, load_wide, score_curve       # noqa: E402
from turn_book import ma_gate, reversal_gate                      # noqa: E402

ONE_WAY = 0.002          # 0.15% buy / 0.25% sell, averaged


def gated_equal_weight(W: Dict, mask: np.ndarray, gate: Optional[np.ndarray],
                       lo: int = 0, hi: Optional[int] = None,
                       cost: float = ONE_WAY,
                       rebalance: int = 0) -> Tuple[pd.Series, float]:
    """Hold every member whose gate is on; positions drift between events.

    The naive version of this - recompute exact equal weights every day - charges
    a fee on the entire book every session, because adding one name changes all
    the other weights by 1/k. That is an artefact of the implementation, not a
    property of the strategy, and it is large enough to turn a working book into
    a -100% drawdown.

    So this trades only what actually changes: a name that turns its gate on is
    bought at the current per-name target, a name that turns it off is sold in
    full, and everything else is left to drift. ``rebalance`` optionally
    re-levels the survivors every N sessions; 0 means never.

    Positions are set from the previous close and earn the next day's return.
    """
    sl = slice(lo, hi)
    mark = W["mark"].to_numpy()[sl]
    fac = W["fac"].to_numpy()[sl]
    m = mask[sl] if gate is None else (mask[sl] & gate[sl])
    ret = np.full(mark.shape, np.nan)
    ret[1:] = (mark[1:] / mark[:-1]) * fac[1:] - 1.0
    ret = np.clip(ret, -CAP, CAP)
    tradable = np.isfinite(ret)

    n, k = mark.shape
    pos = np.zeros(k)                    # value held per name
    cash = 1.0
    eq = np.ones(n)
    traded = 0.0
    for i in range(1, n):
        pos = pos * (1.0 + np.nan_to_num(ret[i]))
        want = (m[i - 1] == 1) & tradable[i]
        equity = pos.sum() + cash

        out = (pos > 0) & ~want
        if out.any():
            v = pos[out].sum()
            cash += v * (1.0 - cost)
            traded += v / max(equity, 1e-12)
            pos[out] = 0.0

        n_want = int(want.sum())
        if n_want:
            target = equity / n_want
            enter = want & (pos <= 0)
            if rebalance and i % rebalance == 0:
                enter = want            # re-level everyone on the schedule
            for j in np.flatnonzero(enter):
                delta = target - pos[j]
                if delta > 0 and cash > 0:
                    buy = min(delta, cash)
                    cash -= buy * (1.0 + cost)
                    pos[j] += buy
                    traded += buy / max(equity, 1e-12)
                elif delta < 0:
                    cash += (-delta) * (1.0 - cost)
                    pos[j] += delta
                    traded += (-delta) / max(equity, 1e-12)
        eq[i] = pos.sum() + cash
    return pd.Series(eq, index=W["mark"].index[sl]), traded


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-size", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    W = load_wide()
    close = W["close"]
    pit = pit_universe(W, args.universe_size)
    ma20 = ma_gate(close, 20)
    ma50 = ma_gate(close, 50)
    ma200 = ma_gate(close, 200)
    rev15 = reversal_gate(close, 0.15, 0.15)
    rev25 = reversal_gate(close, 0.25, 0.25)
    gates = {"none (own them all)": None, "ma20": ma20, "ma50": ma50,
             "ma200": ma200, "rev 15%": rev15, "rev 25%": rev25,
             "ma20 AND rev15": (ma20 & rev15).astype(np.int8),
             "ma20 OR rev15": (ma20 | rev15).astype(np.int8)}

    n = len(W["mark"])
    edges = np.linspace(int(n * 0.35), n, args.folds + 1).astype(int)
    windows = [(edges[k], edges[k + 1]) for k in range(args.folds)]
    labels = [f"{W['mark'].index[a]:%Y-%m}" for a, _ in windows]

    print(f"\n{'=' * 112}\n GATED EQUAL WEIGHT ON THE LARGE CAPS — no ranking, "
          f"only whether each name is heading up\n{'=' * 112}")
    print(f" {'gate':<22}" + "".join(f"{f'from {l}':>12}" for l in labels)
          + f"{'mean':>9}{'worst':>9}{'full':>9}{'maxDD':>8}{'ulcer':>7}")
    rows = []
    for name, g in gates.items():
        cs, dds = [], []
        for a, b in windows:
            eq, _ = gated_equal_weight(W, pit, g, a, b)
            s = score_curve(eq, 0)
            cs.append(s["cagr"])
            dds.append(s["max_dd"])
        eq_full, turn = gated_equal_weight(W, pit, g)
        f = score_curve(eq_full, 0)
        rows.append({"gate": name, **{f"fold{i+1}": cs[i] for i in range(args.folds)},
                     "mean": float(np.mean(cs)), "worst": float(np.min(cs)),
                     "full_cagr": f["cagr"], "full_growth": f["growth"],
                     "max_dd": f["max_dd"], "ulcer": f["ulcer"],
                     "median_year": f["median_year"], "pct_positive": f["pct_positive"],
                     "worst_year": f["worst_year"], "turnover": turn})
        print(f" {name:<22}" + "".join(f"{v:>+12.1%}" for v in cs)
              + f"{np.mean(cs):>+9.1%}{np.min(cs):>+9.1%}"
              f"{f['cagr']:>+9.1%}{f['max_dd']:>8.0%}{f['ulcer']:>7.2f}")

    R = pd.DataFrame(rows)
    R.to_csv("reports/bluechip_ew.csv", index=False)

    print(f"\n{'=' * 112}\n THE FULL RECORD — what each would have been like to own\n{'=' * 112}")
    print(f" {'gate':<22}{'growth':>10}{'CAGR':>9}{'median yr':>12}"
          f"{'worst yr':>11}{'+years':>9}{'maxDD':>8}{'ulcer':>7}")
    for _, r in R.iterrows():
        print(f" {r['gate']:<22}{r['full_growth']:>9,.1f}x{r['full_cagr']:>+9.1%}"
              f"{r['median_year']:>+12.1%}{r['worst_year']:>+11.1%}"
              f"{r['pct_positive']:>9.0%}{r['max_dd']:>8.0%}{r['ulcer']:>7.2f}")

    best = R.sort_values("mean", ascending=False).iloc[0]
    plain = R[R["gate"] == "none (own them all)"].iloc[0]
    print(f"\n best gate out of sample: {best['gate']} at {best['mean']:+.1%} mean, "
          f"worst fold {best['worst']:+.1%}")
    print(f" owning them all:          {plain['mean']:+.1%} mean, "
          f"worst fold {plain['worst']:+.1%}")
    print(f" the gate is worth {best['mean'] - plain['mean']:+.1%} a year out of "
          f"sample and {best['max_dd'] - plain['max_dd']:+.0%} of drawdown.")
    print("\n -> reports/bluechip_ew.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
