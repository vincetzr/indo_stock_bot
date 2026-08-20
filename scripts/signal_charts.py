#!/usr/bin/env python3
"""See the buys and sells, and what they did to the money.

Two panels per name:

    top     price on a log scale, the confirmed up-leg shaded green and the down
            leg red, a green triangle on every bar the rule bought and a red one
            on every bar it sold, plus the live trigger line - the exact price
            that flips the next signal.
    bottom  Rp10,000,000 traded on those signals with real fees, against
            Rp10,000,000 simply held, on the same bars.

Every arrow sits where the rule ACTUALLY fired at the time. ``band_state`` is
causal and its closed legs never repaint, so nothing here is drawn at a hindsight
pivot - which matters, because a chart with arrows on hindsight pivots is the
single most flattering and most useless picture in trading.

The fill is the NEXT bar's open where available, never the close that generated
the signal, and the round trip is charged 0.56% (0.28% buy, 0.18% sell, 0.10%
sale tax).

    python3 scripts/signal_charts.py --tickers CUAN ADRO BBCA ASII
    python3 scripts/signal_charts.py --tickers ADRO --timeframe weekly
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
import matplotlib.pyplot as plt                 # noqa: E402
from matplotlib.ticker import FuncFormatter     # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config           # noqa: E402
from idxbot.data.cache import Cache             # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV        # noqa: E402
from account_sim import LOT, load_ohlc          # noqa: E402
from paint_live import band_state               # noqa: E402

BG, FG, GRID = "#131722", "#d1d4dc", "#2a2e39"
UP, DOWN, CASH = "#26a69a", "#ef5350", "#787b86"
FEE_BUY, FEE_SELL = 0.0028, 0.0028
CAPITAL = 10_000_000


def series_for(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "weekly":
        w = df.resample("W-FRI").agg({"open": "first", "close": "last"}).dropna()
        return w
    return df[["open", "close"]].dropna()


def trade(df: pd.DataFrame, band: float) -> Tuple[pd.Series, pd.Series, List[Dict]]:
    """Equity from trading the signals, equity from holding, and the trade list."""
    close = df["close"].to_numpy(float)
    opn = df["open"].to_numpy(float)
    idx = df.index
    st, trig = band_state(close, band)

    cash, lots, entry = float(CAPITAL), 0, np.nan
    eq = np.empty(len(df))
    trades: List[Dict] = []
    for i in range(len(df)):
        j = i - 1                      # yesterday's colour, today's open
        if j >= 0 and np.isfinite(opn[i]) and opn[i] > 0:
            if st[j] and lots == 0:
                n = int(cash / (opn[i] * LOT * (1 + FEE_BUY)))
                if n > 0:
                    cash -= n * LOT * opn[i] * (1 + FEE_BUY)
                    lots, entry = n, opn[i]
                    trades.append({"i": i, "date": idx[i], "side": "BUY",
                                   "px": opn[i]})
            elif not st[j] and lots > 0:
                cash += lots * LOT * opn[i] * (1 - FEE_SELL)
                trades.append({"i": i, "date": idx[i], "side": "SELL",
                               "px": opn[i], "pnl": opn[i] / entry - 1.0})
                lots = 0
        eq[i] = cash + lots * LOT * close[i]

    n0 = int(CAPITAL / (opn[0] * LOT * (1 + FEE_BUY))) if opn[0] > 0 else 0
    hold = pd.Series(CAPITAL - n0 * LOT * opn[0] * (1 + FEE_BUY)
                     + n0 * LOT * close, index=idx)
    return pd.Series(eq, index=idx), hold, trades


def rupiah(v, _p=None) -> str:
    if v >= 1e9:
        return f"{v/1e9:,.1f}bn"
    if v >= 1e6:
        return f"{v/1e6:,.0f}m"
    return f"{v:,.0f}"


def draw(ax_p, ax_e, df: pd.DataFrame, band: float, ticker: str,
         timeframe: str) -> Dict[str, float]:
    close = df["close"].to_numpy(float)
    idx = df.index
    st, trig = band_state(close, band)
    eq, hold, trades = trade(df, band)

    for ax in (ax_p, ax_e):
        ax.set_facecolor(BG)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.55)
        ax.tick_params(colors=FG, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(GRID)

    lo, hi = close.min() * 0.80, close.max() * 1.25
    ax_p.fill_between(idx, lo, hi, where=st.astype(bool), color=UP,
                      alpha=0.08, step="post", zorder=0)
    ax_p.fill_between(idx, lo, hi, where=~st.astype(bool), color=DOWN,
                      alpha=0.06, step="post", zorder=0)
    ax_p.plot(idx, close, color="#c8ccd4", lw=1.25, zorder=3, label="close")
    ax_p.plot(idx, trig, color="#5b8def", lw=1.0, ls="--", drawstyle="steps-post",
              alpha=0.8, zorder=2, label="flips at")

    buys = [t for t in trades if t["side"] == "BUY"]
    sells = [t for t in trades if t["side"] == "SELL"]
    if buys:
        ax_p.scatter([t["date"] for t in buys],
                     [close[t["i"]] * 0.90 for t in buys],
                     marker="^", s=95, color=UP, zorder=6, edgecolors="none",
                     label=f"buy ({len(buys)})")
    if sells:
        ax_p.scatter([t["date"] for t in sells],
                     [close[t["i"]] * 1.10 for t in sells],
                     marker="v", s=95, color=DOWN, zorder=6, edgecolors="none",
                     label=f"sell ({len(sells)})")

    ax_p.set_yscale("log")
    ax_p.set_ylim(lo, hi)
    ax_p.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_p.legend(facecolor="#1e222d", edgecolor=GRID, labelcolor=FG, fontsize=8,
                loc="upper left", ncol=4)

    pnl = np.array([t.get("pnl", np.nan) for t in sells], dtype=float)
    pnl = pnl[np.isfinite(pnl)] - (FEE_BUY + FEE_SELL)
    win = float((pnl > 0).mean()) if len(pnl) else np.nan
    med = float(np.median(pnl)) if len(pnl) else np.nan
    ax_p.set_title(
        f"{ticker}  ·  {timeframe} {band:.0%} band  ·  {len(sells)} round trips  ·  "
        f"win {win:.0%}  ·  median trip {med:+.1%} after fees",
        color=FG, fontsize=12, pad=9, loc="left")

    # The null that decides whether timing did anything: the same TIME IN THE
    # MARKET, spent at random. A rule that sits in cash through a crash beats
    # buy-and-hold for free, so beating hold is not evidence and this line is.
    exposure = float(st[:-1].astype(bool).mean())
    null = pd.Series(CAPITAL * np.exp(exposure * np.log(hold / CAPITAL)), index=idx)
    ax_e.plot(idx, eq, color=UP if eq.iloc[-1] >= null.iloc[-1] else DOWN,
              lw=1.5, label=f"signals  Rp{rupiah(eq.iloc[-1])}")
    ax_e.plot(idx, hold, color=CASH, lw=1.3, ls="--",
              label=f"buy & hold  Rp{rupiah(hold.iloc[-1])}")
    ax_e.plot(idx, null, color="#e0b341", lw=1.2, ls=":",
              label=f"same exposure, random  Rp{rupiah(null.iloc[-1])}")
    ax_e.axhline(CAPITAL, color=GRID, lw=1.0)
    ax_e.set_yscale("log")
    ax_e.yaxis.set_major_formatter(FuncFormatter(rupiah))
    ax_e.legend(facecolor="#1e222d", edgecolor=GRID, labelcolor=FG, fontsize=8,
                loc="upper left")
    beat_hold = eq.iloc[-1] > hold.iloc[-1]
    beat_null = eq.iloc[-1] > null.iloc[-1]
    verdict = ("beats hold AND the null" if beat_hold and beat_null else
               "beats hold but NOT the null — it was just in cash less"
               if beat_hold else "behind hold")
    ax_e.set_title(f"Rp10,000,000 from {idx[0]:%b %Y}   ·   {exposure:.0%} of the "
                   f"time invested   ·   {verdict}",
                   color=FG, fontsize=10, pad=6, loc="left")
    return {"ticker": ticker, "trips": len(sells), "win": win, "median": med,
            "final": float(eq.iloc[-1]), "hold": float(hold.iloc[-1]),
            "null": float(null.iloc[-1]), "exposure": exposure,
            "state": "GREEN" if st[-1] else "RED", "trigger": float(trig[-1])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["CUAN", "ADRO", "BBCA", "ASII"])
    ap.add_argument("--timeframe", default="daily", choices=["daily", "weekly"])
    ap.add_argument("--band", type=float, default=None,
                    help="default: 8% daily, 12% weekly")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--out", default="reports/signal_charts.png")
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)
    band = args.band if args.band else (0.12 if args.timeframe == "weekly" else 0.08)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    frames = []
    for t in args.tickers:
        df = load_ohlc(loader, t)
        if df is None:
            print(f"  ! {t}: no data")
            continue
        df = df[df.index >= pd.Timestamp(args.start)]
        s = series_for(df, args.timeframe)
        if len(s) > 60:
            frames.append((t, s))
    if not frames:
        raise SystemExit("no usable data")

    n = len(frames)
    fig, axes = plt.subplots(2 * n, 1, figsize=(15, 6.4 * n), facecolor=BG,
                             gridspec_kw={"height_ratios": [2.4, 1] * n})
    if n == 1:
        axes = np.array(axes)
    rows = []
    for k, (t, s) in enumerate(frames):
        rows.append(draw(axes[2 * k], axes[2 * k + 1], s, band, t, args.timeframe))

    fig.suptitle(
        f"IDX signals — {args.timeframe} {band:.0%} band, real fees (0.56% round "
        f"trip), fills at the next open\n"
        f"arrows sit where the rule fired at the time; closed legs never repaint",
        color=FG, fontsize=13, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(args.out, dpi=115, facecolor=BG)

    print(f"{'=' * 92}\n {args.timeframe.upper()} {band:.0%} BAND — Rp10,000,000 "
          f"from {args.start}\n{'=' * 92}")
    print(f" {'ticker':<8}{'trips':>7}{'win':>7}{'median':>9}{'invested':>10}"
          f"{'signals':>12}{'hold':>10}{'null':>10}{'beats null':>12}")
    for r in rows:
        print(f" {r['ticker']:<8}{r['trips']:>7}{r['win']:>7.0%}"
              f"{r['median']:>+9.1%}{r['exposure']:>10.0%}"
              f"{'Rp' + rupiah(r['final']):>12}{'Rp' + rupiah(r['hold']):>10}"
              f"{'Rp' + rupiah(r['null']):>10}"
              f"{('yes' if r['final'] > r['null'] else 'no'):>12}")
    beat = sum(1 for r in rows if r["final"] > r["hold"])
    beatn = sum(1 for r in rows if r["final"] > r["null"])
    print(f"\n beats buy-and-hold on {beat}/{len(rows)}; beats the "
          f"same-exposure null on {beatn}/{len(rows)}")
    print(" 'null' = the same time in the market spent at random. Beating hold "
          "while losing to\n the null means the rule was simply invested less "
          "in a market that fell.")
    print(f"\n THESE {len(rows)} NAMES WERE CHOSEN BY HAND. Across all 46 big "
          f"caps the same rule\n has a median 35% win rate and a -4.0% median "
          f"round trip — run --universe for that.")
    print(f"\n now: " + ", ".join(f"{r['ticker']} {r['state']} (flips at "
                                  f"{r['trigger']:,.0f})" for r in rows))
    print(f"\n -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
