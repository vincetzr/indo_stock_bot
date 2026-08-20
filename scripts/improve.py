#!/usr/bin/env python3
"""Act on what the review found, and check whether it actually helped.

Result 114 pointed in one direction, and it pointed with numbers rather than
intuition:

    weekly 12%, 18 round trips   edge vs null  +0.034   clears it on 52%
    daily  8%,  54 round trips   edge vs null  -0.222   clears it on 43%
    liquid names                 edge vs null  +0.050 (large), +0.013 (mid)
    illiquid names (590 of 749)  edge vs null  -0.277

Two readings of that, and they agree: trade LESS, and trade only where the
spread is not the strategy. So the candidate improvement is a wider band on a
slower clock, restricted to names that actually trade.

WHY THIS IS THE DANGEROUS PART
------------------------------
Every one of those numbers came from the same data this sweep is about to search
for a better setting. Finding a band that scores well in-sample is guaranteed -
there is always a best cell in a grid - and means nothing. So:

  * the band is chosen on a TRAINING half and scored on a HOLDOUT half it never
    touched, per name;
  * the score is the edge over the same-exposure null, never over buy-and-hold;
  * the number of settings tried is reported, and the best result is compared
    against a Bonferroni-corrected threshold, because a 12-cell grid finds a
    2-sigma cell about half the time by luck alone;
  * a random-band control is run alongside: pick the band by coin flip on the
    training half. If the fitted choice cannot beat the coin flip out of sample,
    the fitting added nothing and the honest answer is that there is no
    improvement to be had.

    python3 scripts/improve.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config           # noqa: E402
from idxbot.data.cache import Cache             # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV        # noqa: E402
from account_sim import load_ohlc               # noqa: E402
from method_review import TURNOVER_EDGES, measure, resample   # noqa: E402


def score(s: pd.Series, band: float, timeframe: str) -> Optional[float]:
    m = measure(resample(s, timeframe), band)
    return None if m is None else m["edge"]


def split_test(s: pd.Series, grid: List[float], timeframe: str,
               rng: np.random.Generator) -> Optional[Dict[str, float]]:
    """Choose on the first half, score on the second. Never the reverse."""
    cut = len(s) // 2
    train, test = s.iloc[:cut], s.iloc[cut:]
    if len(train) < 300 or len(test) < 300:
        return None
    tr = [(b, score(train, b, timeframe)) for b in grid]
    tr = [(b, v) for b, v in tr if v is not None]
    if not tr:
        return None
    best = max(tr, key=lambda x: x[1])[0]
    coin = float(rng.choice(grid))
    out = {"fitted_band": best, "coin_band": coin}
    for key, b in (("fitted", best), ("coin", coin), ("base", 0.12)):
        v = score(test, b, timeframe)
        if v is None:
            return None
        out[f"{key}_oos"] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", default="weekly", choices=["weekly", "daily"])
    ap.add_argument("--min-turnover", type=float, default=TURNOVER_EDGES[0],
                    help="Rp/day; the review found illiquid names are far worse")
    ap.add_argument("--min-price", type=float, default=50.0)
    ap.add_argument("--grid", type=float, nargs="+",
                    default=[0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30, 0.35])
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)
    rng = np.random.default_rng(0)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    names = sorted({os.path.basename(f).split(".")[0].upper()
                    for f in glob.glob(os.path.join(
                        cfg.path("data.cache_dir", "data/cache"),
                        "ohlcv", "*.JK.csv.gz"))})
    names = [n for n in names if n.isalpha() and len(n) == 4]

    series: Dict[str, pd.Series] = {}
    for t in names:
        df = load_ohlc(loader, t)
        if df is None or len(df) < 900:
            continue
        s = df["close"]
        s = s[s.index >= pd.Timestamp("2015-01-01")]
        if len(s) < 900 or float(s.median()) < args.min_price:
            continue
        tv = float((df["close"] * df["volume"]).tail(250).median())
        if not np.isfinite(tv) or tv < args.min_turnover:
            continue
        series[t] = s
    if not series:
        raise SystemExit("no names clear the liquidity screen")

    print(f"{'=' * 96}\n THE CANDIDATE IMPROVEMENT — {args.timeframe}, "
          f"liquid names only\n{'=' * 96}")
    print(f" {len(series)} names clear Rp{args.min_turnover/1e9:,.0f}bn/day "
          f"turnover and Rp{args.min_price:.0f} price")
    print(f" grid of {len(args.grid)} bands: "
          + ", ".join(f"{b:.0%}" for b in args.grid))

    # ---- in-sample sweep, shown so the temptation is visible ---------------
    print(f"\n{'=' * 96}\n IN SAMPLE — the number that means nothing\n{'=' * 96}")
    print(f" {'band':>6}{'names':>8}{'median edge vs null':>22}{'beats null':>12}")
    ins = []
    for b in args.grid:
        vals = [v for v in (score(s, b, args.timeframe) for s in series.values())
                if v is not None]
        if not vals:
            continue
        ins.append({"band": b, "edge": float(np.median(vals)),
                    "share": float(np.mean(np.array(vals) > 0))})
        print(f" {b:>6.0%}{len(vals):>8}{ins[-1]['edge']:>+22.4f}"
              f"{ins[-1]['share']:>12.0%}")
    best_ins = max(ins, key=lambda r: r["edge"])
    print(f"\n best in sample: {best_ins['band']:.0%} at {best_ins['edge']:+.4f}. "
          f"This is the number\n to distrust — a grid always has a best cell.")

    # ---- out of sample, which is the only one that counts ------------------
    rows = [r for r in (split_test(s, args.grid, args.timeframe, rng)
                        for s in series.values()) if r]
    if not rows:
        raise SystemExit("no name had enough history to split")
    R = pd.DataFrame(rows)
    R.to_csv("reports/improve.csv", index=False)

    print(f"\n{'=' * 96}\n OUT OF SAMPLE — band chosen on the first half, scored "
          f"on the second\n{'=' * 96}")
    print(f" {len(R)} names with enough history to split\n")
    print(f" {'selector':<26}{'median edge vs null':>22}{'beats null':>13}"
          f"{'median band':>13}")
    for key, label, bcol in (("fitted", "fitted on the train half", "fitted_band"),
                             ("coin", "band chosen at random", "coin_band"),
                             ("base", "the incumbent 12%", None)):
        col = R[f"{key}_oos"]
        b = f"{R[bcol].median():.0%}" if bcol else "12%"
        print(f" {label:<26}{col.median():>+22.4f}"
              f"{float((col > 0).mean()):>13.0%}{b:>13}")

    # ---- did the fitting add anything at all? ------------------------------
    d = R["fitted_oos"] - R["coin_oos"]
    t_, p_ = stats.ttest_rel(R["fitted_oos"], R["coin_oos"])
    thresh = 0.05 / len(args.grid)
    print(f"\n{'=' * 96}\n DID THE FITTING BEAT A COIN FLIP?\n{'=' * 96}")
    print(f" fitted minus random, per name: median {d.median():+.4f}, "
          f"ahead on {float((d > 0).mean()):.0%}")
    print(f" paired t = {t_:.2f}, p = {p_:.4f}")
    print(f" Bonferroni threshold for {len(args.grid)} bands tried: "
          f"p < {thresh:.4f}")

    print(f"\n{'=' * 96}\n VERDICT\n{'=' * 96}")
    fit_edge = float(R["fitted_oos"].median())
    fit_share = float((R["fitted_oos"] > 0).mean())
    beat_coin = p_ < thresh and d.median() > 0
    if fit_edge > 0 and fit_share >= 0.55 and beat_coin:
        print(f" IMPROVED. The fitted band clears the null out of sample "
              f"({fit_edge:+.4f} on\n {fit_share:.0%} of names) and beats a coin "
              f"flip at the corrected threshold.")
    elif fit_edge > 0 and fit_share >= 0.55:
        print(f" Out of sample the fitted band clears the null "
              f"({fit_edge:+.4f}, {fit_share:.0%} of names),\n but it does NOT "
              f"beat picking a band at random (p = {p_:.3f} against a "
              f"{thresh:.4f}\n threshold). So the LIQUIDITY SCREEN and the "
              f"slower clock are doing the work,\n not the fitted band. Keep the "
              f"screen; do not sell the fit.")
    else:
        print(f" NOT IMPROVED. Out of sample the fitted band scores "
              f"{fit_edge:+.4f} and clears the\n null on {fit_share:.0%} of "
              f"names — short of the 55% bar. Searching the grid did\n not "
              f"produce a rule that survives contact with data it was not "
              f"chosen on.")
    print("\n -> reports/improve.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
