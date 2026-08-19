#!/usr/bin/env python3
"""Trade the daily arrows in a real account: whole lots, real fees, no leverage.

Answers one question exactly: start with Rp10,000,000, buy every green arrow and
sell every red one on the daily band, long only, and what is in the account at
the end.

Everything that makes a paper backtest differ from a real one is included:

    lots          IDX trades in lots of 100. Cash that cannot buy a whole lot
                  stays in cash, so a Rp10m account in a Rp20,000 stock owns
                  five lots and Rp0 of the sixth.
    fees          0.15% to buy, 0.25% to sell (the extra 0.1% is the sale tax).
    fills         the signal comes from a close, the fill is the NEXT session's
                  open. Nothing is bought at the price that generated its signal.
    no shorting   a red arrow means sit in cash. There is no short leg.
    no leverage   the account never spends money it does not have.
    capacity      a position is capped at 10% of the name's 20-day median
                  turnover. Without this a Rp10m account compounding into
                  Rp380m buys a position the stock cannot fill, and the
                  backtest quietly becomes fiction (Result 52).

The start date is swept, because on a stock that went 43 -> 2,690 -> 815 the
answer depends far more on when you started than on the rule.

    python3 scripts/account_sim.py --ticker CUAN --capital 10000000
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV       # noqa: E402
from paint_live import band_state              # noqa: E402

LOT = 100
FEE_BUY, FEE_SELL = 0.0015, 0.0025


def load_ohlc(loader: YahooOHLCV, ticker: str) -> Optional[pd.DataFrame]:
    d = loader.get(ticker, max_age=86400 * 30)
    if d is None or len(d) < 300:
        return None
    d = d.set_index("date").sort_index()
    out = d[["open", "close", "volume"]].astype(float).dropna()
    # cap impossible prints on the close, and carry the same factor to the open
    r = out["close"].pct_change().clip(-0.35, 0.35).fillna(0.0)
    clean = out["close"].iloc[0] * (1.0 + r).cumprod()
    scale = (clean / out["close"]).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    out["close"] = clean
    out["open"] = out["open"] * scale
    out["turnover"] = (out["close"] * out["volume"]).rolling(
        20, min_periods=5).median()
    return out


def run_account(df: pd.DataFrame, band: float, capital: float,
                max_participation: float = 0.10
                ) -> Tuple[pd.Series, List[Dict]]:
    """Buy the green arrows, sell the red ones, in whole lots at the next open."""
    close = df["close"].to_numpy(float)
    opn = df["open"].to_numpy(float)
    tv = df["turnover"].to_numpy(float)
    idx = df.index
    state, _ = band_state(close, band)

    cash, lots = float(capital), 0
    equity = np.empty(len(df))
    trades: List[Dict] = []
    entry_px = np.nan
    for i in range(len(df)):
        # signal from bar i-1's close is acted on at bar i's open
        if i > 0 and state[i - 1] != state[i - 2 if i > 1 else 0] or (i == 1 and state[0] == 1):
            pass
        if i > 0:
            want = state[i - 1]
            have = 1 if lots > 0 else 0
            if want == 1 and have == 0 and np.isfinite(opn[i]) and opn[i] > 0:
                cap = (max_participation * tv[i - 1]
                       if np.isfinite(tv[i - 1]) else cash)
                n = int(min(cash, cap) / (opn[i] * LOT * (1 + FEE_BUY)))
                if n > 0:
                    cost = n * LOT * opn[i] * (1 + FEE_BUY)
                    cash -= cost
                    lots = n
                    entry_px = opn[i]
                    trades.append({"date": idx[i], "side": "BUY", "px": opn[i],
                                   "lots": n, "value": cost})
            elif want == 0 and have == 1 and np.isfinite(opn[i]) and opn[i] > 0:
                proceeds = lots * LOT * opn[i] * (1 - FEE_SELL)
                cash += proceeds
                trades.append({"date": idx[i], "side": "SELL", "px": opn[i],
                               "lots": lots, "value": proceeds,
                               "pnl_pct": opn[i] / entry_px - 1.0})
                lots = 0
        equity[i] = cash + lots * LOT * close[i]
    return pd.Series(equity, index=idx), trades


def summarise(eq: pd.Series, trades: List[Dict], capital: float,
              hold: pd.Series) -> Dict[str, float]:
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    peak = eq.cummax()
    sells = [t for t in trades if t["side"] == "SELL"]
    wins = [t for t in sells if t.get("pnl_pct", 0) > 0]
    return {
        "final": float(eq.iloc[-1]),
        "growth": float(eq.iloc[-1] / capital),
        "cagr": float((eq.iloc[-1] / capital) ** (1 / yrs) - 1) if yrs > 0 else np.nan,
        "hold_final": float(hold.iloc[-1]),
        "hold_cagr": float((hold.iloc[-1] / capital) ** (1 / yrs) - 1) if yrs > 0 else np.nan,
        "max_dd": float((eq / peak - 1).min()),
        "hold_dd": float((hold / hold.cummax() - 1).min()),
        "round_trips": len(sells),
        "win_rate": float(len(wins) / len(sells)) if sells else np.nan,
        "years": yrs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="CUAN")
    ap.add_argument("--capital", type=float, default=10_000_000)
    ap.add_argument("--band", type=float, default=0.08)
    ap.add_argument("--participation", type=float, default=0.10,
                    help="max position as a share of 20-day median turnover")
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    df = load_ohlc(loader, args.ticker)
    if df is None:
        raise SystemExit(f"no data for {args.ticker}")
    last = df.index[-1]
    print(f"{args.ticker}: {len(df):,} sessions to {last:%Y-%m-%d}, "
          f"last close {df['close'].iloc[-1]:,.0f}, "
          f"20d turnover Rp{df['turnover'].iloc[-1]/1e9:,.1f}bn/day "
          f"(cap: Rp{args.participation*df['turnover'].iloc[-1]/1e6:,.0f}m per position)")
    st, tr = band_state(df["close"].to_numpy(float), args.band)
    print(f"daily {args.band:.0%} band today: "
          f"{'GREEN (holding)' if st[-1] else 'RED (in cash)'}, "
          f"flips at {tr[-1]:,.0f} ({tr[-1]/df['close'].iloc[-1]-1:+.1%})\n")

    rows = []
    for years in (1, 2, 3):
        start = last - pd.DateOffset(years=years)
        sub = df[df.index >= start]
        if len(sub) < 100:
            continue
        eq, trades = run_account(sub, args.band, args.capital, args.participation)
        n0 = int(args.capital / (sub["open"].iloc[0] * LOT * (1 + FEE_BUY)))
        hold = pd.Series(
            args.capital - n0 * LOT * sub["open"].iloc[0] * (1 + FEE_BUY)
            + n0 * LOT * sub["close"].to_numpy(float), index=sub.index)
        s = summarise(eq, trades, args.capital, hold)
        rows.append({"window": f"{years}y", "start": sub.index[0], **s})

    print(f"{'=' * 96}\n Rp{args.capital:,.0f} TRADING THE DAILY {args.band:.0%} "
          f"ARROWS — long only, whole lots, fees paid\n{'=' * 96}")
    print(f" {'window':<8}{'from':<13}{'final':>16}{'CAGR':>9}{'buy&hold':>16}"
          f"{'B&H CAGR':>10}{'trades':>8}{'win%':>7}{'maxDD':>8}")
    for r in rows:
        print(f" {r['window']:<8}{r['start']:%Y-%m-%d}  Rp{r['final']:>13,.0f}"
              f"{r['cagr']:>+9.1%}Rp{r['hold_final']:>13,.0f}{r['hold_cagr']:>+10.1%}"
              f"{r['round_trips']:>8}{r['win_rate']:>7.0%}{r['max_dd']:>8.0%}")

    # full history, for context
    eq, trades = run_account(df, args.band, args.capital, args.participation)
    n0 = int(args.capital / (df["open"].iloc[0] * LOT * (1 + FEE_BUY)))
    hold = pd.Series(args.capital - n0 * LOT * df["open"].iloc[0] * (1 + FEE_BUY)
                     + n0 * LOT * df["close"].to_numpy(float), index=df.index)
    s = summarise(eq, trades, args.capital, hold)
    print(f"\n full history ({s['years']:.1f}y from {df.index[0]:%Y-%m-%d}): "
          f"Rp{s['final']:,.0f} ({s['cagr']:+.1%}/yr) vs buy&hold "
          f"Rp{s['hold_final']:,.0f} ({s['hold_cagr']:+.1%}/yr)")
    print(f" {s['round_trips']} round trips, {s['win_rate']:.0%} winners, "
          f"drawdown {s['max_dd']:.0%} vs {s['hold_dd']:.0%} holding")

    T = pd.DataFrame(trades)
    T.to_csv(f"reports/account_{args.ticker}.csv", index=False)
    if len(T):
        print(f"\n last 8 trades")
        for _, t in T.tail(8).iterrows():
            pnl = f"{t['pnl_pct']:+.1%}" if t["side"] == "SELL" and pd.notna(
                t.get("pnl_pct")) else ""
            print(f"   {t['date']:%Y-%m-%d}  {t['side']:<5}{t['lots']:>5} lots "
                  f"@ {t['px']:>8,.0f}   Rp{t['value']:>13,.0f}  {pnl}")
    print(f"\n -> reports/account_{args.ticker}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
