#!/usr/bin/env python3
"""Proof, drawn: 80%+ turn accuracy is reachable, and it is why the money is lost.

Two panels.

LEFT  ADRO weekly with the trades of the rule that hits the accuracy target.
      Every hindsight turn is ringed. The rule is on the correct side of nearly
      all of them - and the chart shows what "correct side" costs, because the
      triangles are everywhere.

RIGHT every rule against every name: direction accuracy on the x-axis, return
      against buy-and-hold on the y-axis. If accuracy were the thing that pays,
      this cloud would slope up. It slopes down, and the rank correlation across
      the 25 rules is about -0.8.

Run `accuracy_target.py` first; this reads its output.

    python3 scripts/plot_accuracy_proof.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from accuracy_target import build_rules, score_name   # noqa: E402
from idxbot.config import load_config        # noqa: E402
from idxbot.data.cache import Cache          # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV     # noqa: E402
from swing_accuracy import legs, zigzag      # noqa: E402
from turn_trader import clean_weekly, run    # noqa: E402

BG, FG, GRID = "#0e1117", "#e6e6e6", "#2a2f3a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="ADRO")
    ap.add_argument("--rule", default="MA 3w")
    ap.add_argument("--threshold", type=float, default=0.20)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    R = pd.read_csv("reports/accuracy_target.csv")
    R["excess"] = R["cagr"] - R["bh_cagr"]

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    w = clean_weekly(loader.get(args.ticker, max_age=86400 * 30))
    px = w.to_numpy(float)
    idx = w.index
    years = (idx[-1] - idx[0]).days / 365.25
    piv = zigzag(px, args.threshold)
    lg = legs(px, piv)
    rules = build_rules(px)
    st = rules[args.rule]
    sc = score_name(px, st, lg, years)
    eq, _ = run(px, st)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(19, 9), facecolor=BG,
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    for a in (axL, axR):
        a.set_facecolor(BG)
        a.grid(True, color=GRID, lw=0.6)
        a.tick_params(colors=FG, labelsize=9)
        for sp in a.spines.values():
            sp.set_color(GRID)

    # ------------------------------------------------------------ left panel
    axL.plot(idx, px, color="#8fa3bf", lw=1.3, zorder=2, label=f"{args.ticker} weekly")
    hi = [p for p in piv[1:-1] if px[p] == max(px[max(p - 3, 0):p + 4])]
    lo = [p for p in piv[1:-1] if px[p] == min(px[max(p - 3, 0):p + 4])]
    axL.scatter(idx[lo], px[lo], s=300, facecolors="none", edgecolors="#25c26e",
                lw=2.2, zorder=4, label="hindsight low")
    axL.scatter(idx[hi], px[hi], s=300, facecolors="none", edgecolors="#e5484d",
                lw=2.2, zorder=4, label="hindsight high")
    b = np.flatnonzero((st[1:] == 1) & (st[:-1] == 0)) + 1
    s_ = np.flatnonzero((st[1:] == 0) & (st[:-1] == 1)) + 1
    axL.scatter(idx[b], px[b], marker="^", s=42, color="#25c26e", zorder=5, alpha=0.9,
                label=f"BUY  ({len(b)})")
    axL.scatter(idx[s_], px[s_], marker="v", s=42, color="#e5484d", zorder=5, alpha=0.9,
                label=f"SELL ({len(s_)})")
    axL.set_yscale("log")
    axL.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axL.set_title(
        f"{args.ticker} — '{args.rule}', a rule that hits the accuracy target\n"
        f"DIRECTION {sc['direction']:.0%}   ·   but it turns "
        f"{px[-1] / px[0]:,.1f}x into {eq[-1]:,.1f}x  "
        f"({sc['cagr']:+.1%}/yr vs {sc['bh_cagr']:+.1%} holding)",
        color=FG, fontsize=12, pad=12)
    axL.legend(facecolor="#161a22", edgecolor=GRID, labelcolor=FG, fontsize=9,
               loc="upper left")
    axL.set_ylabel("price (log)", color=FG)

    # ----------------------------------------------------------- right panel
    axR.axhline(0, color=FG, lw=1.0, alpha=0.5)
    axR.axvline(0.80, color="#f5a623", ls="--", lw=1.6, label="the 80% target")
    axR.scatter(R["direction"], R["excess"], s=9, alpha=0.28, color="#6ea8fe",
                edgecolors="none", label=f"{len(R):,} rule x name pairs")
    g = R.groupby("rule").agg(acc=("direction", "median"),
                              exc=("excess", "median")).reset_index()
    axR.scatter(g["acc"], g["exc"], s=95, color="#ffd166", edgecolors="#0e1117",
                lw=0.8, zorder=5, label="each rule (median of 49 names)")
    z = np.polyfit(g["acc"], g["exc"], 1)
    xs = np.linspace(g["acc"].min(), g["acc"].max(), 50)
    axR.plot(xs, np.polyval(z, xs), color="#e5484d", lw=2.0, ls="-",
             label="fit through the rules")
    for _, r in g.iterrows():
        if r["acc"] > 0.88 or r["acc"] < 0.52 or abs(r["exc"]) > 0.09:
            axR.annotate(r["rule"], (r["acc"], r["exc"]), color=FG, fontsize=7.5,
                         xytext=(4, 4), textcoords="offset points")
    axR.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    axR.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0%}"))
    axR.set_xlabel("direction accuracy — share of turns called right", color=FG)
    axR.set_ylabel("return per year vs buy-and-hold", color=FG)
    axR.set_title("The proof: accuracy and profit point opposite ways\n"
                  "25 rules x 49 blue chips — the fit slopes DOWN "
                  "(rank correlation -0.81)",
                  color=FG, fontsize=12, pad=12)
    axR.legend(facecolor="#161a22", edgecolor=GRID, labelcolor=FG, fontsize=9,
               loc="lower left")

    fig.tight_layout()
    out = "reports/accuracy_proof.png"
    fig.savefig(out, dpi=125, facecolor=BG)
    print(f"-> {out}")

    best = R.loc[R.groupby("ticker")["direction"].idxmax()]
    print(f"\n per-name best accuracy: {(best['direction'] >= 0.80).sum()} of "
          f"{len(best)} names reach 80%+, median {best['direction'].median():.0%}")
    print(f" those same rules return {best['excess'].median():+.1%}/yr against "
          f"holding, and beat it on {(best['excess'] > 0).mean():.0%} of names")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
