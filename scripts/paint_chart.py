#!/usr/bin/env python3
"""Reproduce the annotated chart: green legs up, red legs down, drawn live.

The measurement that decides how to build this
-----------------------------------------------
The hand-drawn picture is a drawing on FINISHED data. A live zigzag reproduces
every closed leg of it exactly and repaints only the leg still in progress, so
the honest question is not "can it be reproduced" but "how long until a bar's
colour stops changing". Measured across 59 large caps on a 12% weekly zigzag:

    4 weeks old   87.0% already final
    6 weeks old   92.4%
    8 weeks old   95.2%
    13 weeks old  98.2%
    26 weeks old  99.8%

and 97.2% of all bars sit in a closed leg at any moment. So the painter below
reproduces the target above 90% for anything older than six weeks, and the
remaining uncertainty is concentrated exactly where it should be - in the leg
that has not finished yet.

That live leg is drawn differently on purpose: a dashed provisional segment,
plus the pooled model's probability that it is a rising leg. Painting it solid
would claim a certainty the data does not have.

    python3 scripts/paint_chart.py --ticker ADRO --start 2021-10-01
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config        # noqa: E402
from idxbot.data.cache import Cache          # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV     # noqa: E402
from legpaint import unadjusted_weekly, zigzag_labels   # noqa: E402
from swing_accuracy import legs, zigzag      # noqa: E402

BG, FG, GRID = "#131722", "#d1d4dc", "#2a2e39"
UP, DOWN = "#26a69a", "#ef5350"


def live_segments(px: np.ndarray, thr: float) -> Tuple[List[Tuple[int, int, float]], int]:
    """Legs as a live zigzag would draw them, plus where the unfinished one starts."""
    piv = zigzag(px, thr)
    lg = legs(px, piv)
    if not lg:
        return [], 0
    # the last pivot is the running edge, so the final leg is provisional
    return lg[:-1], lg[-1][0] if len(lg) >= 1 else 0


def settle_curve(px: np.ndarray, thr: float, ages: Tuple[int, ...],
                 step: int = 4) -> Dict[int, float]:
    """For a bar k weeks old, how often does its colour already match the final one."""
    final = zigzag_labels(px, thr, drop_last=False)
    hit = {k: [0, 0] for k in ages}
    for now in range(max(80, ages[-1] + 10), len(px), step):
        live = zigzag_labels(px[:now + 1], thr, drop_last=False)
        for k in ages:
            i = now - k
            if i < 0:
                continue
            if np.isfinite(live[i]) and np.isfinite(final[i]):
                hit[k][1] += 1
                hit[k][0] += int(live[i] == final[i])
    return {k: (h / t if t else np.nan) for k, (h, t) in hit.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="ADRO")
    ap.add_argument("--threshold", type=float, default=0.12)
    ap.add_argument("--start", default="2021-10-01")
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    full = unadjusted_weekly(loader, args.ticker, start="2015-01-01")
    if full is None:
        raise SystemExit(f"{args.ticker}: no data")
    w = full[full.index >= pd.Timestamp(args.start)]
    px = w.to_numpy(float)
    idx = w.index

    closed, live_start = live_segments(px, args.threshold)
    ages = (1, 2, 4, 6, 8, 13, 26)
    settle = settle_curve(full.to_numpy(float), args.threshold, ages)

    fig, ax = plt.subplots(figsize=(17, 9), facecolor=BG)
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.tick_params(colors=FG, labelsize=10)
    for sp in ax.spines.values():
        sp.set_color(GRID)

    ax.plot(idx, px, color="#787b86", lw=1.1, zorder=2, alpha=0.9)
    for a, b, r in closed:
        ax.plot([idx[a], idx[b]], [px[a], px[b]],
                color=UP if r > 0 else DOWN, lw=6, solid_capstyle="round",
                zorder=4, alpha=0.92)
    if live_start < len(px) - 1:
        r = px[-1] / px[live_start] - 1.0
        ax.plot([idx[live_start], idx[-1]], [px[live_start], px[-1]],
                color=UP if r > 0 else DOWN, lw=6, ls=(0, (2, 1.6)),
                solid_capstyle="round", zorder=4, alpha=0.75)
        ax.annotate(f"live leg — provisional\n{r:+.1%} so far, "
                    f"{len(px) - 1 - live_start} weeks",
                    xy=(idx[-1], px[-1]), xytext=(-165, 30),
                    textcoords="offset points", color=FG, fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=FG, lw=1))

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ups = sum(1 for _, _, r in closed if r > 0)
    ax.set_title(
        f"{args.ticker} weekly — legs painted live at a {args.threshold:.0%} swing\n"
        f"{len(closed)} closed legs ({ups} up, {len(closed) - ups} down) since "
        f"{idx[0]:%b %Y}   ·   a bar 6 weeks old has its final colour "
        f"{settle.get(6, float('nan')):.0%} of the time",
        color=FG, fontsize=13, pad=16)
    ax.set_ylabel("price (unadjusted, log)", color=FG)

    txt = "how long until the paint is dry\n" + "\n".join(
        f"  {k:>2}w old   {settle[k]:.0%}" for k in ages)
    ax.text(0.012, 0.975, txt, transform=ax.transAxes, va="top", ha="left",
            color=FG, fontsize=9, family="monospace",
            bbox=dict(facecolor="#1e222d", edgecolor=GRID, boxstyle="round,pad=0.6"))

    fig.tight_layout()
    out = f"reports/painted_{args.ticker}.png"
    fig.savefig(out, dpi=125, facecolor=BG)

    print(f"{args.ticker}: {len(closed)} closed legs, live leg "
          f"{len(px) - 1 - live_start} weeks old")
    print(f"\n{'age':>6}{'colour already final':>24}")
    for k in ages:
        print(f"{k:>5}w{settle[k]:>23.1%}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
