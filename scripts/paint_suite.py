#!/usr/bin/env python3
"""Render IDX Suite for one ticker — the same arithmetic the Pine file runs.

    python3 scripts/paint_suite.py BBCA
    python3 scripts/paint_suite.py BBCA --bars 500 --target 15

WHY THIS EXISTS AND WHAT IT IS NOT.

It is a REPLICA, not a screenshot. Pine cannot be executed from here, so this
reimplements every line of `pine/IDX_Suite.pine` in Python from the same panel
the research was measured on, prints the panel the chart would print, and draws
the chart the chart would draw. Two independent implementations agreeing is
worth something; it is not the same as the editor compiling.

THE SERIES. The chart is painted on `adj_close` RESCALED so the final bar equals
the real closing price. That keeps the right edge at a price you could actually
deal at while removing the split cliffs that would otherwise put a fake 80% gap
in BBCA's history and hand the ZigZag a swing that never happened. Every
level, every EMA and the volatility are computed on that same series, so they
are mutually consistent; the IDX tick and auto-rejection ladders are computed
from the real close, because those are rules about the real price.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.cone import (SIGMA_MAX, SIGMA_MIN, p_target_first,    # noqa: E402
                         p_touch, sessions_to, vol_decile)
from levels import ZZ, pivots_confirmed                           # noqa: E402
from time_price import hma                                        # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")
OUT = "reports"
FEE = 0.28 + 0.18 + 0.10          # the user's Mandiri schedule, percent
DAYS_PER_SESSION = 365.25 / 252.0

#  H37 turn-detection accuracy, measured on 47,002 confirmed swing highs.
DET = {"close over EMA34": (0.692, 0.554, 0.109),
       "hull55 rising": (0.683, 0.472, 0.118),
       "close over EMA50": (0.666, 0.531, 0.115),
       "hma21 over hma55": (0.640, 0.479, 0.105),
       "price>50>100>200": (0.445, 0.396, 0.125)}


# ------------------------------------------------------------ IDX mechanics --
def tick_of(p: float) -> float:
    return 1.0 if p < 200 else 2.0 if p < 500 else 5.0 if p < 2000 else \
        10.0 if p < 5000 else 25.0


def max_step_of(p: float) -> float:
    return 10.0 if p < 200 else 20.0 if p < 500 else 50.0 if p < 2000 else \
        100.0 if p < 5000 else 250.0


def ara_of(p: float, thin: bool) -> float:
    if thin:
        return 0.10
    return 0.35 if p < 200 else 0.25 if p < 5000 else 0.20


# ------------------------------------------------------------------ compute --
def build(tk: str) -> pd.DataFrame:
    P = pd.read_parquet(PANEL, columns=["date", "ticker", "close", "adj_close",
                                        "volume", "tradeable"])
    g = P[P["ticker"] == tk].sort_values("date").reset_index(drop=True)
    if g.empty:
        raise SystemExit(f"{tk} is not in the panel")
    g = g[(g["adj_close"] > 0) & (g["close"] > 0)].reset_index(drop=True)
    #  Rescale the total-return series to end at the real close.
    g["px"] = g["adj_close"] * (g["close"].iloc[-1] / g["adj_close"].iloc[-1])
    s = g["px"]
    for n in (34, 50, 100, 200):
        g[f"ema{n}"] = s.ewm(span=n, adjust=False, min_periods=n).mean()
    g["hull"] = hma(s, 55)
    g["hull2"] = g["hull"].shift(2)
    g["hfast"] = hma(s, 18)
    g["ret1"] = s.pct_change()
    g["vol60"] = g["ret1"].rolling(60, min_periods=60).std(ddof=1)
    g["hi252"] = s.rolling(252, min_periods=252).max()
    g["rp"] = (g["px"] * g["volume"]).rolling(20, min_periods=5).median()
    g["stack"] = (s > g["ema50"]) & (g["ema50"] > g["ema100"]) \
        & (g["ema100"] > g["ema200"])

    #  Confirmed swing levels, stepped forward exactly as the Pine does.
    p = s.to_numpy(float)
    piv, conf = pivots_confirmed(p, ZZ)
    #  RESISTANCE IS THE NEAREST CONFIRMED HIGH **ABOVE PRICE**, NOT THE MOST
    #  RECENT ONE. A first version carried only the latest confirmed high, so
    #  after a deep fall and a lower high the panel printed "none above - at new
    #  highs" for a name 24.7% below its 52-week peak. The contradiction with
    #  its own drawdown row is what exposed it; a chart that only ever looks at
    #  the last pivot cannot see the level everyone else is watching.
    res = np.full(len(p), np.nan)
    sup = np.full(len(p), np.nan)
    highs: List[float] = []
    lows: List[float] = []
    ptr = 0
    for i in range(len(p)):
        while ptr < len(piv) and conf[ptr] <= i:
            (highs if (ptr > 0 and p[piv[ptr]] > p[piv[ptr - 1]]) else lows) \
                .append(float(p[piv[ptr]]))
            ptr += 1
        above = [h for h in highs if h > p[i]]
        below = [l for l in lows if l < p[i]]
        if above:
            res[i] = min(above)
        if below:
            sup[i] = max(below)
    g["res"], g["sup"] = res, sup
    #  The flip the panel labels: close crossing the EMA34, the most accurate
    #  turn detector of the five H37 scored.
    on = (s > g["ema34"]).fillna(False).to_numpy()
    g["buy"] = on & ~np.concatenate([[False], on[:-1]])
    g["sell"] = ~on & np.concatenate([[False], on[:-1]])
    return g


def panel(g: pd.DataFrame, tk: str, target: float) -> List[str]:
    r = g.iloc[-1]
    px, close = float(r["px"]), float(r["close"])
    sig = float(r["vol60"])
    stk = bool(r["stack"])
    thin = bool(g["px"].tail(126).mean() < 51)
    tick, step = tick_of(close), max_step_of(close)
    ara, arb = ara_of(close, thin), 0.10 if thin else 0.15
    round_trip = FEE / 100.0 + 2.0 * (tick / 2.0) / close

    up, dn = 1.0 + target / 100.0, 1.0 - target / 100.0
    ok = np.isfinite(sig) and sig > 0
    pu = p_touch(up, sig, stk) if ok else np.nan
    pd_ = p_touch(dn, sig, stk) if ok else np.nan
    q1, med, q3 = ((sessions_to(up, sig, q) for q in ("q1", "med", "q3"))
                   if ok else (np.nan,) * 3)
    last = pd.Timestamp(r["date"])

    def when(n: float) -> str:
        if not np.isfinite(n):
            return "n/a"
        return (last + pd.Timedelta(days=float(n) * DAYS_PER_SESSION)) \
            .strftime("%d %b %y")

    res, sup = r["res"], r["sup"]
    p_res = p_touch(res / px, sig, stk) if np.isfinite(res) and ok else np.nan
    p_sup = p_touch(sup / px, sig, stk) if np.isfinite(sup) and ok else np.nan
    p_first = (p_target_first(res / px - 1.0, 1.0 - sup / px, sig)
               if np.isfinite(res) and np.isfinite(sup) and ok else np.nan)
    f1, null, give = DET["close over EMA34"]

    def row(k: str, v: str) -> str:
        return f"  {k:<30}{v:>30}"

    L = [f"IDX SUITE — {tk}   {last:%Y-%m-%d}   close Rp {close:,.0f}", ""]
    L += ["STATE"]
    L += [row("annualised vol (60b)", f"{sig * np.sqrt(252) * 100:.1f}%"),
          row("IDX volatility decile", f"{vol_decile(sig)} of 10"),
          row("% of 52-week high", f"{100 * px / r['hi252']:.1f}%"),
          row("drawdown from peak", f"{100 * (px / r['hi252'] - 1):.1f}%"),
          row("turnover (20b median)", f"Rp {r['rp'] / 1e9:.2f}bn")]
    L += ["", "TREND  [H30, corrected]"]
    lbl = ("price>F>M>S (best)" if stk else
           "ordered, price below" if r["ema50"] > r["ema100"] > r["ema200"]
           else "not aligned")
    sc = int(px > r["ema50"]) + int(r["ema50"] > r["ema100"]) \
        + int(r["ema100"] > r["ema200"])
    L += [row("EMA stack", f"{lbl}  {sc}/3"),
          row("60b fwd mean log, this state",
              f"{0.0127 if stk else -0.0279:+.4f}  (base -0.0140)"),
          row("Hull 55 slope",
              "rising" if r["hull"] > r["hull2"] else "falling")]
    L += ["", f"PROJECTION  [H32, in-sample, 1y]"]
    L += [row(f"target  +{target:.0f}%", f"Rp {px * up:,.0f}"),
          row("P(touch it within a year)", f"{pu * 100:.0f}%"),
          row("  earliest quartile", f"{when(q1)}  ({q1:.0f}b)"),
          row("  median", f"{when(med)}  ({med:.0f}b)"),
          row("  latest quartile", f"{when(q3)}  ({q3:.0f}b)"),
          row(f"mirror  -{target:.0f}%",
              f"Rp {px * dn:,.0f}   {pd_ * 100:.0f}%"),
          row("up / down odds", f"{pu / pd_:.2f}x"),
          row("caveat", "in-sample; band, not a date")]
    L += ["", "TARGET / STOP  [H34, swing levels]"]
    L += [row("resistance (target)",
              "none above - at new highs" if not np.isfinite(res)
              else f"Rp {res:,.0f}   +{100 * (res / px - 1):.1f}%"),
          row("  P(touch it within a year)",
              "n/a" if not np.isfinite(p_res) else f"{p_res * 100:.0f}%"),
          row("support (stop)",
              "none below - at new lows" if not np.isfinite(sup)
              else f"Rp {sup:,.0f}   -{100 * (1 - sup / px):.1f}%"),
          row("  P(touch it within a year)",
              "n/a" if not np.isfinite(p_sup) else f"{p_sup * 100:.0f}%"),
          row("P(target first | one is hit)",
              "n/a" if not np.isfinite(p_first) else f"{p_first * 100:.0f}%"),
          row("bracket verdict", "0 of 30 beat hold in both halves"),
          row("fibonacci", "off - measured nothing")]
    L += ["", "HOW ACCURATE  [H36/H37, measured]"]
    L += [row("flip catches the top (F1)", f"{f1:.3f}  vs null {null:.3f}"),
          row("  peak already given back", f"{give * 100:.1f}%"),
          row("P(touch) skill vs base rate", "+0.013 out-of-sample"),
          row("date band coverage", "0.500 measured / 0.50 claimed")]
    L += ["", "IDX MECHANICS  [exact]"]
    L += [row("board", "THIN +/-10%" if thin else "main"),
          row("tick / max step", f"Rp {tick:.0f} / Rp {step:.0f}"),
          row("ARA tomorrow",
              f"Rp {np.floor(close * (1 + ara) / tick) * tick:,.0f}  (+{ara:.0%})"),
          row("ARB tomorrow",
              f"Rp {np.ceil(close * (1 - arb) / tick) * tick:,.0f}  (-{arb:.0%})"),
          row("round trip (floor)", f"{round_trip * 100:.2f}%"),
          row("flip labels", "EMA34 break - none compound")]
    #  H44'S LIQUIDITY GATE, mirrored from the .pine so the two agree. Every
    #  apparent edge in the simple-method sweep lived BELOW this line: the rank
    #  IC of a stochastic against the forward return is +0.072 in the bottom
    #  turnover tercile and -0.0021 (t -0.15) in the top, rising with price
    #  STALENESS (+0.099 for names flat >30% of the month). That is
    #  non-synchronous pricing, not forecasting.
    rp60 = float(np.nanmedian((g["close"] * g["volume"]).tail(60)))
    ok = bool(close >= 500 and rp60 >= 1e10)
    L += ["", "H44 LIQUIDITY GATE  [the row that matters most]"]
    L += [row("tradeable universe",
              "YES - edges tested here are ~ZERO" if ok
              else "NO - edges here are non-synchronous pricing"),
          row("60-bar median turnover", f"Rp {rp60 / 1e9:,.1f}bn")]
    return L


def paint(g: pd.DataFrame, tk: str, bars: int, target: float, path: str,
          lines: List[str]) -> None:
    d = g.tail(bars).reset_index(drop=True)
    x = np.arange(len(d))
    #  THE PANEL GETS ITS OWN COLUMN. Overlaid on the price axes it sat on top
    #  of the projection boxes — the one part of the chart the panel is about.
    fig = plt.figure(figsize=(17.5, 9), dpi=130)
    fig.patch.set_facecolor("#131722")
    gs = fig.add_gridspec(1, 2, width_ratios=[3.05, 1.0], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    axp = fig.add_subplot(gs[0, 1])
    axp.axis("off")
    ax.set_facecolor("#131722")

    up = (d["hull"] > d["hull2"]).to_numpy()
    for i in range(1, len(d)):
        ax.fill_between(x[i - 1:i + 1], d["hull"].iloc[i - 1:i + 1],
                        d["hull2"].iloc[i - 1:i + 1],
                        color="#26a69a" if up[i] else "#ef5350", alpha=0.55,
                        linewidth=0)
    ax.plot(x, d["px"], color="#d1d4dc", lw=1.2, label="close (adjusted)")
    ax.plot(x, d["ema34"], color="#ffeb3b", lw=1.5, label="EMA 34")
    ax.plot(x, d["ema50"], color="#26c6da", lw=0.8, alpha=0.7, label="EMA 50")
    ax.plot(x, d["ema100"], color="#ffa726", lw=0.8, alpha=0.7, label="EMA 100")
    ax.plot(x, d["ema200"], color="#ab47bc", lw=1.1, alpha=0.8, label="EMA 200")
    #  THE STEPPED LEVELS ARE DELIBERATELY FAINT. They are what the indicator
    #  knew at each past bar, which is what makes the chart auditable, and they
    #  jump whenever price crosses one pivot and the next one up becomes the
    #  nearest. Drawn boldly they read as a barcode and hide the price.
    ax.step(x, d["res"], where="post", color="#ef5350", lw=0.7, ls=":",
            alpha=0.45, label="resistance, as known then")
    ax.step(x, d["sup"], where="post", color="#26a69a", lw=0.7, ls=":",
            alpha=0.45, label="support, as known then")

    r = d.iloc[-1]
    n = len(d) - 1
    for lvl, col, tag in ((r["res"], "#ef5350", "resistance"),
                          (r["sup"], "#26a69a", "support")):
        if np.isfinite(lvl):
            ax.hlines(lvl, n - min(90, n), n + 4, color=col, lw=1.6)
            ax.text(n - min(90, n) - 2, lvl, f"{tag} {lvl:,.0f}", color=col,
                    fontsize=8, ha="right", va="center")

    #  Triangles, not text boxes. An 0.85-recall detector fires often enough
    #  that labelled boxes overlap into an unreadable stripe — which is itself
    #  worth seeing, and is why the flip is a context read and not an entry.
    lo, hi = float(d["px"].min()), float(d["px"].max())
    pad = (hi - lo) * 0.03
    bi = np.flatnonzero(d["buy"].to_numpy())
    si = np.flatnonzero(d["sell"].to_numpy())
    ax.scatter(x[bi], d["px"].iloc[bi] - pad, marker="^", s=26, c="#26a69a",
               zorder=5, label=f"EMA34 flip up ({len(bi)})")
    ax.scatter(x[si], d["px"].iloc[si] + pad, marker="v", s=26, c="#ef5350",
               zorder=5, label=f"EMA34 flip down ({len(si)})")

    sig = float(r["vol60"])
    right = n + 40
    if np.isfinite(sig) and sig > 0:
        m = 1.0 + target / 100.0
        q1 = sessions_to(m, sig, "q1")
        q3 = sessions_to(m, sig, "q3")
        med = sessions_to(m, sig, "med")
        tgt = float(r["px"]) * m
        stop = float(r["px"]) * (1.0 - target / 100.0)
        pu = p_touch(m, sig, bool(r["stack"]))
        pdn = p_touch(1.0 - target / 100.0, sig, bool(r["stack"]))
        h = (hi - lo) * 0.006
        for y, col, txt in ((tgt, "#26a69a", f"+{target:.0f}%  {pu:.0%}"),
                            (stop, "#ef5350", f"-{target:.0f}%  {pdn:.0%}")):
            ax.plot([n, n + q1], [float(r["px"]), y], color=col, lw=0.8,
                    ls="--", alpha=0.6)
            ax.add_patch(plt.Rectangle((n + q1, y - h), q3 - q1, 2 * h,
                                       color=col, alpha=0.35))
            ax.plot([n + med], [y], marker="|", color=col, ms=12)
            ax.text(n + q3 + 3, y, txt, color=col, fontsize=8.5, va="center")
        ax.set_xlim(-2, n + q3 + 34)

    step = max(1, len(d) // 11)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([t.strftime("%b %y") for t in d["date"][::step]],
                       color="#787b86", fontsize=8)
    ax.tick_params(colors="#787b86", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2a2e39")
    ax.grid(color="#2a2e39", lw=0.4)
    fig.suptitle(f"IDX Suite — {tk}   {d['date'].iloc[-1]:%Y-%m-%d}   "
                 f"close Rp {r['close']:,.0f}", color="#d1d4dc", fontsize=13)
    ax.legend(loc="lower left", fontsize=7, facecolor="#1e222d",
              edgecolor="#2a2e39", labelcolor="#d1d4dc", ncol=3)
    fig.subplots_adjust(left=0.045, right=0.995, top=0.94, bottom=0.07)
    #  The panel, on the image, because a chart without its numbers is a
    #  picture of a trend and this indicator's whole point is the numbers.
    body = "\n".join(lines[2:])
    axp.text(0.0, 1.0, body, transform=axp.transAxes, ha="left", va="top",
             family="DejaVu Sans Mono", fontsize=7.0, color="#d1d4dc",
             bbox=dict(boxstyle="round,pad=0.6", fc="#1e222d", ec="#2a2e39"))
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="BBCA")
    ap.add_argument("--bars", type=int, default=420)
    ap.add_argument("--target", type=float, default=20.0)
    a = ap.parse_args()
    g = build(a.ticker.upper())
    lines = panel(g, a.ticker.upper(), a.target)
    for line in lines:
        print(line)
    png = os.path.join(OUT, f"suite_{a.ticker.upper()}.png")
    paint(g, a.ticker.upper(), a.bars, a.target, png, lines)
    print(f"\nchart -> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
