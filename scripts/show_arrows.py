#!/usr/bin/env python3
"""Render exactly what `leg_arrows.pine` draws, so it can be seen before it is used.

Mirrors the Pine logic bar for bar: a fast band and a slow band, an arrow on the
bar the fast band confirms a leg, filled when the slow band agrees and hollow
when it does not, plus the live trigger line and the confirmed-leg shading.

Anything visible here is what TradingView will show. Nothing in it uses a bar
that had not printed, which is why the arrows in the rendered history sit where
they really would have appeared at the time.

    python3 scripts/show_arrows.py --tickers CUAN ADRO
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
from legpaint import unadjusted_weekly       # noqa: E402
from paint_daily import unadjusted_daily     # noqa: E402
from paint_live import band_state            # noqa: E402

BG, FG, GRID = "#131722", "#d1d4dc", "#2a2e39"
UP, DOWN = "#26a69a", "#ef5350"


def panel(ax, w: pd.Series, fast: float, slow: float, title: str) -> Dict[str, float]:
    px = w.to_numpy(float)
    idx = w.index
    f_st, f_tr = band_state(px, fast)
    s_st, _ = band_state(px, slow)
    agree = f_st == s_st

    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
    ax.tick_params(colors=FG, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(GRID)

    ax.plot(idx, px, color="#9aa0aa", lw=1.2, zorder=3)
    ax.fill_between(idx, px.min() * 0.85, px.max() * 1.1, where=f_st.astype(bool),
                    color=UP, alpha=0.07, zorder=1)
    ax.fill_between(idx, px.min() * 0.85, px.max() * 1.1, where=~f_st.astype(bool),
                    color=DOWN, alpha=0.05, zorder=1)
    ax.plot(idx, f_tr, color="#5b8def", lw=1.0, ls="-", drawstyle="steps-post",
            alpha=0.75, zorder=2, label=f"flips at ({fast:.0%} band)")

    flips = np.flatnonzero(np.diff(f_st.astype(int)) != 0) + 1
    n_fill = n_hollow = 0
    for i in flips:
        up = f_st[i] == 1
        col = UP if up else DOWN
        y = px[i] * (0.93 if up else 1.07)
        if agree[i]:
            ax.scatter(idx[i], y, marker="^" if up else "v", s=190, color=col,
                       zorder=6, edgecolors="none")
            n_fill += 1
        else:
            ax.scatter(idx[i], y, marker="^" if up else "v", s=110,
                       facecolors="none", edgecolors=col, lw=1.8, zorder=6)
            n_hollow += 1

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    state = "UP LEG" if f_st[-1] else "DOWN LEG"
    conv = "both bands agree" if agree[-1] else "fast band only"
    ax.set_title(
        f"{title}   ·   now: {state} ({conv})   ·   flips at {f_tr[-1]:,.0f} "
        f"({f_tr[-1] / px[-1] - 1:+.1%} from {px[-1]:,.0f})",
        color=FG, fontsize=11.5, pad=10, loc="left")
    ax.legend(facecolor="#1e222d", edgecolor=GRID, labelcolor=FG, fontsize=8,
              loc="upper left")
    return {"filled": n_fill, "hollow": n_hollow, "state": state,
            "trigger": float(f_tr[-1]), "price": float(px[-1])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["CUAN", "ADRO"])
    ap.add_argument("--fast", type=float, default=0.12)
    ap.add_argument("--slow", type=float, default=0.25)
    ap.add_argument("--start", default="2021-06-01")
    ap.add_argument("--daily", action="store_true",
                    help="daily bars; the band should be narrower than the weekly one")
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    series = []
    for t in args.tickers:
        w = (unadjusted_daily(loader, t, start=args.start) if args.daily
             else unadjusted_weekly(loader, t, start=args.start))
        if w is not None and len(w) > 60:
            series.append((t, w))
    if not series:
        raise SystemExit("no data")

    fig, axes = plt.subplots(len(series), 1, figsize=(16, 5.2 * len(series)),
                             facecolor=BG)
    if len(series) == 1:
        axes = [axes]
    out = {}
    for ax, (t, w) in zip(axes, series):
        out[t] = panel(ax, w, args.fast, args.slow,
                       f"{t} {'daily' if args.daily else 'weekly'}")

    fig.suptitle(
        f"IDX Leg Arrows — filled = both bands agree ({args.fast:.0%} and "
        f"{args.slow:.0%}), hollow = fast band only\n"
        f"arrows print on confirmation and never move; the blue line is the "
        f"price that flips the next one",
        color=FG, fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = "reports/leg_arrows_demo.png"
    fig.savefig(path, dpi=125, facecolor=BG)

    for t, r in out.items():
        print(f"{t:<6} {r['filled']:>3} filled + {r['hollow']:>3} hollow arrows   "
              f"now {r['state']:<9} at {r['price']:,.0f}, flips at {r['trigger']:,.0f} "
              f"({r['trigger'] / r['price'] - 1:+.1%})")
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
