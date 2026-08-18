#!/usr/bin/env python3
"""Does the bounded-lag turn rule pay where the book actually trades?

``turn_trader.py`` showed the causal reversal filter fixes the capture problem
Result 69 identified - 68-87% of each up leg banked versus 31% for a moving
average - and still fails to beat holding a single name out of sample. That
leaves one question worth asking: the deployed book does not hold single names,
it holds the eight strongest, and it currently gates them on a 20-day average.
Does swapping that gate for the bounded-lag rule improve it?

The comparison, on identical fills, identical costs, identical selection:

    none        hold the top eight until the next rebalance, whatever they do
    ma20        hold only while above the 20-day average  (Part XV, deployed)
    ma200       hold only while above the 200-day average (Part XIII)
    reversal    hold only while the causal reversal state is long: sell when the
                close falls ``exit`` below its high since the gate turned on, buy
                back when it rises ``entry`` above its low since the gate turned
                off

Only the gate changes. The momentum ranking, the liquidity floor, the lot
rounding, the turnover cap and the 0.15/0.25% fees are the same engine used for
every other result in this repository.

Every gate is evaluated on closed bars and acted on at the next session's open,
and the reversal thresholds are chosen inside each walk-forward fold's training
window only, so the out-of-sample column never sees its own tuning.

    python3 scripts/turn_book.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from optimize_consistent import (LOT, FEE_BUY, FEE_SELL, MIN_TURNOVER,      # noqa: E402
                                 MAX_PARTICIPATION, CAPITAL, load_wide,
                                 score_curve)

LOOKBACK, TOP_N, REBALANCE = 120, 8, 10      # the configuration all five folds chose


def reversal_gate(close: pd.DataFrame, entry: float, exit_: float) -> np.ndarray:
    """Per-name causal reversal state on daily closes: 1 long, 0 flat.

    Same state machine as ``turn_trader.reversal_state`` but vectorised across
    the panel one column at a time. NaN days (the name did not trade) hold the
    state rather than resetting it, so a suspension does not manufacture a turn.
    """
    v = close.to_numpy(float)
    n, m = v.shape
    out = np.zeros((n, m), dtype=np.int8)
    for j in range(m):
        col = v[:, j]
        long = False
        ext = np.nan
        for i in range(n):
            p = col[i]
            if not np.isfinite(p):
                out[i, j] = 1 if long else 0
                continue
            if not np.isfinite(ext):
                ext = p
            if long:
                if p > ext:
                    ext = p
                elif 1.0 - p / ext >= exit_ - 1e-12:
                    long = False
                    ext = p
            else:
                if p < ext:
                    ext = p
                elif p / ext - 1.0 >= entry - 1e-12:
                    long = True
                    ext = p
            out[i, j] = 1 if long else 0
    return out


def ma_gate(close: pd.DataFrame, window: int) -> np.ndarray:
    ma = close.rolling(window, min_periods=window).mean()
    return (close > ma).fillna(False).to_numpy().astype(np.int8)


def simulate(W: Dict, gate: Optional[np.ndarray], lookback: int = LOOKBACK,
             top_n: int = TOP_N, rebalance: int = REBALANCE,
             lo: int = 0, hi: Optional[int] = None) -> Tuple[pd.Series, int]:
    """The Part XIV engine with a pluggable daily gate.

    ``gate[n, j]`` is decided at bar n's close and governs bar n+1's open, both
    for buying (a name off its gate is not eligible at a rebalance) and for
    selling (a holding whose gate turns off is sold the next session).
    """
    dates = W["mark"].index[lo:hi]
    sl = slice(lo, hi)
    op_v = W["open"].to_numpy()[sl]
    mark_v = W["mark"].to_numpy()[sl]
    tv_v = W["tv"].to_numpy()[sl]
    fac_v = W["fac"].to_numpy()[sl]
    mom_v = W["mom"][lookback].to_numpy()[sl]
    g = None if gate is None else gate[sl]
    cols = list(W["mark"].columns)
    idx = {t: j for j, t in enumerate(cols)}
    tradable = ~np.isnan(op_v)

    cash = CAPITAL
    lots: Dict[str, int] = {}
    equity = np.empty(len(dates))
    trades = 0

    for n in range(len(dates)):
        for t, l in list(lots.items()):
            j = idx[t]
            f = fac_v[n, j]
            m = mark_v[n, j]
            if np.isfinite(f) and f > 1.0 and np.isfinite(m):
                cash += l * LOT * m * (f - 1.0)

        # --- gate exit, checked daily, filled next open ---
        if g is not None and n + 1 < len(dates):
            for t, l in list(lots.items()):
                j = idx[t]
                if g[n, j] == 0 and tradable[n + 1, j]:
                    cash += lots.pop(t) * LOT * op_v[n + 1, j] * (1 - FEE_SELL)
                    trades += 1

        if n % rebalance == 0 and n + 1 < len(dates):
            ok = (tv_v[n] >= MIN_TURNOVER) & (mark_v[n] >= 50) & tradable[n + 1]
            if g is not None:
                ok &= g[n] == 1
            score = np.where(ok, mom_v[n], np.nan)
            order = np.argsort(np.where(np.isnan(score), -np.inf, score))[::-1]
            want = [cols[j] for j in order[:top_n] if np.isfinite(score[j])]

            for t in list(lots):
                if t in want or not tradable[n + 1, idx[t]]:
                    continue
                cash += lots.pop(t) * LOT * op_v[n + 1, idx[t]] * (1 - FEE_SELL)
                trades += 1

            if want:
                held = sum(l * LOT * mark_v[n + 1, idx[t]] for t, l in lots.items()
                           if np.isfinite(mark_v[n + 1, idx[t]]))
                budget = (cash + held) / len(want)
                for t in want:
                    j = idx[t]
                    p = op_v[n + 1, j]
                    if not np.isfinite(p) or p <= 0:
                        continue
                    capv = MAX_PARTICIPATION * tv_v[n, j]
                    allowed = min(budget, capv) if np.isfinite(capv) else budget
                    target = int(allowed / (p * LOT * (1 + FEE_BUY)))
                    have = lots.get(t, 0)
                    if target > have:
                        buy = target - have
                        cost = buy * LOT * p * (1 + FEE_BUY)
                        while buy > 0 and cost > cash:
                            buy -= 1
                            cost = buy * LOT * p * (1 + FEE_BUY)
                        if buy > 0:
                            cash -= cost
                            lots[t] = have + buy
                            trades += 1
                    elif target < have:
                        sell = have - target
                        cash += sell * LOT * p * (1 - FEE_SELL)
                        lots[t] = have - sell
                        if not lots[t]:
                            lots.pop(t)
                        trades += 1

        mv = 0.0
        for t, l in lots.items():
            m = mark_v[n, idx[t]]
            if np.isfinite(m):
                mv += l * LOT * m
        equity[n] = cash + mv

    return pd.Series(equity, index=dates), trades


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    W = load_wide()
    close = W["close"]
    pairs = ([(0.12, 0.12), (0.20, 0.20)] if args.quick else
             [(0.08, 0.08), (0.12, 0.12), (0.15, 0.15), (0.20, 0.20),
              (0.25, 0.25), (0.30, 0.30), (0.12, 0.20), (0.20, 0.12),
              (0.08, 0.20), (0.20, 0.30)])

    print("building gates ...")
    gates: Dict[str, Optional[np.ndarray]] = {"none": None,
                                              "ma20": ma_gate(close, 20),
                                              "ma200": ma_gate(close, 200)}
    for e, x in pairs:
        gates[f"rev {e:.0%}/{x:.0%}"] = reversal_gate(close, e, x)

    print(f"\n{'=' * 104}\n FULL HISTORY — same book, only the gate changes\n{'=' * 104}")
    print(f" {'gate':<16}{'growth':>11}{'CAGR':>8}{'medYr':>8}{'worstYr':>9}"
          f"{'+yrs':>6}{'worst3y':>9}{'maxDD':>8}{'ulcer':>7}{'trades':>8}")
    rows = []
    for name, gg in gates.items():
        eq, nt = simulate(W, gg)
        s = score_curve(eq, nt)
        rows.append({"gate": name, **s})
        print(f" {name:<16}{s['growth']:>10,.1f}x{s['cagr']:>+8.1%}"
              f"{s['median_year']:>+8.1%}{s['worst_year']:>+9.1%}"
              f"{s['pct_positive']:>6.0%}{s['worst_3y']:>+9.1%}"
              f"{s['max_dd']:>8.0%}{s['ulcer']:>7.2f}{s['trades']:>8.0f}")
    pd.DataFrame(rows).to_csv("reports/turn_book_full.csv", index=False)

    # ------------------------------ walk forward ------------------------------ #
    # Every gate is run on every out-of-sample window, and the whole table is
    # printed. Choosing one in-fold and reporting only that hides the thing worth
    # knowing - whether a gate is CONSISTENTLY better - behind a selector that
    # Part XIII already showed to be unstable. The selector is still run, on the
    # side, so its instability is visible rather than assumed.
    n = len(W["mark"])
    edges = np.linspace(int(n * 0.35), n, args.folds + 1).astype(int)
    windows = [(edges[k], edges[k + 1]) for k in range(args.folds)]
    labels = [f"{W['mark'].index[a]:%Y-%m}" for a, _ in windows]

    oos: Dict[str, List[Dict[str, float]]] = {}
    for name, gg in gates.items():
        oos[name] = [score_curve(*simulate(W, gg, lo=a, hi=b)) for a, b in windows]

    print(f"\n{'=' * 104}\n WALK FORWARD — every gate on every out-of-sample window "
          f"({args.folds} folds)\n{'=' * 104}")
    print(f" {'gate':<16}" + "".join(f"{f'from {l}':>14}" for l in labels)
          + f"{'mean':>9}{'worst':>9}{'beats none':>12}")
    for name in gates:
        c = [s["cagr"] for s in oos[name]]
        wins = sum(1 for a, b in zip(c, [s["cagr"] for s in oos["none"]]) if a > b)
        print(f" {name:<16}" + "".join(f"{v:>+14.1%}" for v in c)
              + f"{np.mean(c):>+9.1%}{np.min(c):>+9.1%}"
              + f"{f'{wins}/{args.folds}':>12}")
    print(f"\n {'gate':<16}" + "".join(f"{'maxDD':>14}" for _ in labels)
          + f"{'mean':>9}")
    for name in gates:
        d = [s["max_dd"] for s in oos[name]]
        print(f" {name:<16}" + "".join(f"{v:>14.0%}" for v in d)
              + f"{np.mean(d):>9.0%}")

    sel = []
    for k, (tr_hi, te_hi) in enumerate(windows):
        best, best_score = None, -np.inf
        for name, gg in gates.items():
            s = score_curve(*simulate(W, gg, hi=tr_hi))
            v = s["cagr"] / s["ulcer"] if s["ulcer"] > 0 else -np.inf
            if v > best_score:
                best, best_score = name, v
        sel.append({"fold": k + 1, "start": W["mark"].index[tr_hi],
                    "chosen": best, "oos_cagr": oos[best][k]["cagr"],
                    "oos_dd": oos[best][k]["max_dd"]})
    S = pd.DataFrame(sel)
    print(f"\n what an in-fold selector would have picked "
          f"(training-window CAGR per unit ulcer):")
    for _, r in S.iterrows():
        print(f"   fold {r['fold']} from {r['start']:%Y-%m}: chose {r['chosen']:<14}"
              f"-> {r['oos_cagr']:+.1%} OOS")
    print(f"   mean {S['oos_cagr'].mean():+.1%} — compare the table above to see "
          f"whether that selection was worth making.")

    rec = []
    for name in gates:
        rec.append({"gate": name,
                    **{f"fold{k+1}_cagr": oos[name][k]["cagr"] for k in range(args.folds)},
                    **{f"fold{k+1}_dd": oos[name][k]["max_dd"] for k in range(args.folds)},
                    "mean_cagr": float(np.mean([s["cagr"] for s in oos[name]])),
                    "mean_dd": float(np.mean([s["max_dd"] for s in oos[name]]))})
    pd.DataFrame(rec).to_csv("reports/turn_book_walkforward.csv", index=False)
    print("\n -> reports/turn_book_full.csv, reports/turn_book_walkforward.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
