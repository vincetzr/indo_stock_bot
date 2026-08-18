#!/usr/bin/env python3
"""Volatility targeting on the book - the one exposure rule never tested here.

Result 85 found that scaling exposure beats switching it: holding 25% when the
market filter says out beat going fully to cash on BOTH return and drawdown.
That result was about *when* to scale. This is about *how much*, driven by the
book's own realised volatility rather than by a regime flag.

Volatility targeting is the most reliable risk-adjusted-return improvement in the
literature and it has never been applied here at the book level. Part X dismissed
it, but that was vol targeting a SINGLE NAME (ADRO), where the volatility of one
stock is mostly idiosyncratic noise. A portfolio's volatility is far more
persistent - calm months follow calm months - which is exactly the property the
rule needs.

    exposure = target_vol / realised_vol(book, lookback), capped at `max_lev`

With ``max_lev = 1.0`` this can only de-risk, so it should cut drawdown and cost
return. Above 1.0 it borrows in calm periods, which is where the return could
actually increase - so margin interest is charged, because a leveraged backtest
that ignores financing is a fiction.

Combined with the regime overlay from Result 85, four rules are compared:

    flat            always 100%
    regime          100% above the 30-week average, 25% below
    voltarget       target_vol / realised_vol
    both            the product of the two, capped

Everything is walk-forward and the parameters are held fixed across folds,
because Results 75 and 79 both showed that selecting them per fold loses.

    python3 scripts/exposure.py [--folds 5]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from bluechip import pit_universe                        # noqa: E402
from bluechip_final import market_state, OVERLAY_COST    # noqa: E402
from optimize_consistent import load_wide, score_curve   # noqa: E402
from turn_book import simulate                           # noqa: E402

MARGIN_RATE = 0.09          # IDR margin, annual, charged on borrowed exposure
CASH_RATE = 0.03            # what idle cash earns


def vol_exposure(eq: pd.Series, target: float, lookback: int,
                 max_lev: float) -> pd.Series:
    """target / realised vol, computed only from returns already observed.

    The volatility estimate uses ``lookback`` days ending YESTERDAY and is
    shifted one more day before it is acted on, so no day's exposure is set with
    that day's move.
    """
    r = eq.pct_change()
    rv = r.rolling(lookback, min_periods=max(20, lookback // 3)).std() * np.sqrt(252)
    e = (target / rv).clip(upper=max_lev).shift(1)
    return e.fillna(0.0).clip(lower=0.0)


def apply_exposure(eq: pd.Series, e: pd.Series, cost: float = OVERLAY_COST,
                   margin: float = MARGIN_RATE, cash: float = CASH_RATE
                   ) -> Tuple[pd.Series, float]:
    """Run the book at a time-varying exposure, charging turnover and financing.

    Borrowed exposure pays margin interest; unused cash earns the deposit rate.
    Ignoring both is how a leveraged backtest turns into a fiction.
    """
    r = eq.pct_change().fillna(0.0).to_numpy()
    x = e.reindex(eq.index).fillna(0.0).to_numpy()
    out = np.ones(len(r))
    turnover = 0.0
    for i in range(1, len(r)):
        w = x[i]
        carry = -(max(w - 1.0, 0.0) * margin) + (max(1.0 - w, 0.0) * cash)
        out[i] = out[i - 1] * (1.0 + r[i] * w + carry / 252.0)
        d = abs(w - x[i - 1])
        if d > 1e-9:
            out[i] *= (1.0 - cost * d)
            turnover += d
    return pd.Series(out, index=eq.index), turnover


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--universe-size", type=int, default=30)
    ap.add_argument("--lookback", type=int, default=60)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    W = load_wide()
    pit = pit_universe(W, args.universe_size)
    print("building the blue-chip book (250d momentum, top 12, quarterly) ...")
    eq, trades = simulate(W, pit, lookback=250, top_n=12, rebalance=60)
    state = market_state(eq.index, 30)
    base_vol = eq.pct_change().std() * np.sqrt(252)
    print(f"  {trades:,} trades; the book's own realised volatility is "
          f"{base_vol:.0%} a year")

    n = len(eq)
    edges = np.linspace(int(n * 0.35), n, args.folds + 1).astype(int)
    windows = [(edges[k], edges[k + 1]) for k in range(args.folds)]
    labels = [f"{eq.index[a]:%Y-%m}" for a, _ in windows]

    regime_e = pd.Series(np.where(state.to_numpy() > 0.5, 1.0, 0.25), index=eq.index)
    rules: Dict[str, pd.Series] = {
        "flat 100%": pd.Series(1.0, index=eq.index),
        "regime (Result 85)": regime_e,
    }
    for tgt in (0.15, 0.20, 0.25):
        for lev in (1.0, 1.5, 2.0):
            ve = vol_exposure(eq, tgt, args.lookback, lev)
            rules[f"voltarget {tgt:.0%} cap {lev:.1f}"] = ve
            if lev == 1.5:
                rules[f"both {tgt:.0%} cap {lev:.1f}"] = (ve * regime_e).clip(upper=lev)

    print(f"\n{'=' * 112}\n EXPOSURE RULES ON THE BLUE-CHIP BOOK\n{'=' * 112}")
    print(f" {'rule':<24}" + "".join(f"{f'from {l}':>11}" for l in labels)
          + f"{'mean':>8}{'worst':>8}{'CAGR':>8}{'maxDD':>7}{'ulcer':>7}{'turn':>7}")
    rows = []
    for name, e in rules.items():
        cur, turn = apply_exposure(eq, e)
        cs = []
        for a, b in windows:
            seg = cur.iloc[a:b] / cur.iloc[a]
            cs.append(score_curve(seg, 0)["cagr"])
        full = score_curve(cur, trades)
        rows.append({"rule": name, **{f"fold{k+1}": cs[k] for k in range(args.folds)},
                     "mean": float(np.mean(cs)), "worst": float(np.min(cs)),
                     "cagr": full["cagr"], "growth": full["growth"],
                     "max_dd": full["max_dd"], "ulcer": full["ulcer"],
                     "worst_year": full["worst_year"],
                     "pct_positive": full["pct_positive"],
                     "mean_exposure": float(e.mean()), "turnover": turn})
        print(f" {name:<24}" + "".join(f"{c:>+11.1%}" for c in cs)
              + f"{np.mean(cs):>+8.1%}{np.min(cs):>+8.1%}{full['cagr']:>+8.1%}"
              f"{full['max_dd']:>7.0%}{full['ulcer']:>7.2f}{turn:>7.0f}")
    R = pd.DataFrame(rows)
    R.to_csv("reports/exposure.csv", index=False)

    base = R[R["rule"] == "flat 100%"].iloc[0]
    print(f"\n{'=' * 112}\n AGAINST ALWAYS-ON\n{'=' * 112}")
    print(f" {'rule':<24}{'CAGR':>9}{'vs flat':>10}{'maxDD':>8}{'DD saved':>10}"
          f"{'worst yr':>10}{'+yrs':>7}{'avg exposure':>14}")
    for _, r in R.sort_values("cagr", ascending=False).iterrows():
        print(f" {r['rule']:<24}{r['cagr']:>+9.1%}{r['cagr'] - base['cagr']:>+10.2%}"
              f"{r['max_dd']:>8.0%}{(r['max_dd'] - base['max_dd']) * 100:>+9.0f}pt"
              f"{r['worst_year']:>+10.1%}{r['pct_positive']:>7.0%}"
              f"{r['mean_exposure']:>14.0%}")

    winners = R[(R["cagr"] > base["cagr"]) & (R["max_dd"] > base["max_dd"])]
    print(f"\n rules beating always-on on BOTH return and drawdown: {len(winners)}")
    for _, r in winners.iterrows():
        print(f"   {r['rule']}: {r['cagr']:+.1%} vs {base['cagr']:+.1%}, "
              f"{r['max_dd']:.0%} vs {base['max_dd']:.0%}, "
              f"worst fold {r['worst']:+.1%} vs {base['worst']:+.1%}")
    if winners.empty:
        print("   none — every rule that cuts drawdown also costs return")
    print("\n -> reports/exposure.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
