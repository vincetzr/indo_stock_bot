#!/usr/bin/env python3
"""The PURE Hull-55 ribbon colour rule, across every IDX name, full history.

    python3 scripts/hull_colour.py

WHY THIS SCRIPT EXISTS. I told the user the Hull suite loses to buy-and-hold,
and when challenged it turned out I had measured EMA34 breaks and labelled them
as the ribbon. On BBCA the actual hull55 colour returns -17.9% against a hold of
-29.0% — it BEAT holding on the one chart being eyeballed. A claim about the
whole board cannot rest on one name in either direction, so this is the board.

THE RULE, AS PLAINLY AS IT CAN BE STATED
  in   when HMA(55) is rising (h[t] > h[t-2], which is how the Pine plots the
       colour, and it is the colour the user sees)
  out  when it stops rising
  cost 56 bp of fees plus that name's own fraksi-harga half-spread, per round
       trip, charged at the exit

THE BENCHMARK IS OWNING THE NAME OVER THE SAME SPAN — not a duration-matched
hold, which is degenerate here (H39 printed "beat buy-and-hold on 0.0% of
trades" from exactly that mistake, because a trade and a hold over the trade's
own bars differ only by the toll). Both legs are compounded per ticker over the
ticker's FULL span, so the rule is charged for the time it sits in cash.

AND IT IS SPLIT BY LIQUIDITY, because H44's headline was withdrawn when the
by-decile check (CLAUDE.md §7, which had never been run) showed the whole effect
was non-synchronous pricing in thin, stale names. Any board-wide number that is
not split this way is one artefact away from being wrong.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from paint_suite import tick_of                                  # noqa: E402
from selloff import ema, hma                                     # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")
OUT = "reports"
FEE = 0.0056
MID = pd.Timestamp("2013-06-30")


def colour_campaign(p: np.ndarray, green: np.ndarray, cost: float):
    """Compound the rule's log return over the name's whole span.

    Returns (rule log total, hold log total, n round trips, bars in market).
    """
    lg = 0.0
    inb = 0
    n_tr = 0
    i = 1
    n = len(p)
    while i < n:
        if not (green[i] and not green[i - 1]):
            i += 1
            continue
        j = i + 1
        while j < n and green[j]:
            j += 1
        j = min(j, n - 1)
        lg += np.log(max(p[j] / p[i] - cost, 0.01) if p[i] > 0 else 0.01)
        inb += j - i
        n_tr += 1
        i = j + 1
    return lg, np.log(max(p[-1] / p[0], 0.01)), n_tr, inb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-bars", type=int, default=400)
    a = ap.parse_args()

    P = pd.read_parquet(PANEL)
    P = P[P["adj_close"] > 0].sort_values(["ticker", "date"])
    rows = []
    for tk, g in P.groupby("ticker", sort=False):
        if len(g) < a.min_bars:
            continue
        p = g["adj_close"].to_numpy(float)
        raw = g["close"].to_numpy(float)
        med = float(np.nanmedian(raw))
        if not np.isfinite(med) or med <= 0:
            continue
        tv = np.exp(g["log_turnover"].to_numpy(float))
        mtv = float(np.nanmedian(tv))
        cost = FEE + tick_of(med) / med
        h = hma(p, 55)
        green = np.zeros(len(p), bool)
        green[2:] = h[2:] > h[:-2]
        lg, hl, ntr, inb = colour_campaign(p, green, cost)
        span = len(p) / 252.0
        rec = {"ticker": tk, "bars": len(p), "span": span, "tv": mtv,
               "cost": cost, "trades": ntr,
               "in_mkt": inb / max(len(p), 1),
               "rule": float(np.exp(lg / span) - 1.0),
               "hold": float(np.exp(hl / span) - 1.0),
               "start": g["date"].iloc[0], "end": g["date"].iloc[-1]}
        #  HALF-SPLIT IN CALENDAR TIME, per A18: it is the only replication
        #  test this repo trusts, and it is cheap. Each name's own bars are cut
        #  at the panel midpoint, so a name listed late contributes to one half
        #  only rather than being dropped.
        for lab, m in (("early", g["date"] < MID), ("late", g["date"] >= MID)):
            mm = m.to_numpy()
            if mm.sum() < 120:
                rec[f"rule_{lab}"] = rec[f"hold_{lab}"] = np.nan
                continue
            q, gq = p[mm], green[mm]
            l2, h2, _, _ = colour_campaign(q, gq, cost)
            s2 = mm.sum() / 252.0
            rec[f"rule_{lab}"] = float(np.exp(l2 / s2) - 1.0)
            rec[f"hold_{lab}"] = float(np.exp(h2 / s2) - 1.0)
        rows.append(rec)
    R = pd.DataFrame(rows)
    R["edge"] = R["rule"] - R["hold"]
    R.to_csv(os.path.join(OUT, "hull_colour.csv"), index=False)

    def block(D, label):
        if not len(D):
            return
        print(f"{label:<28}{len(D):>6}{D['trades'].mean():>8.1f}"
              f"{D['in_mkt'].mean():>8.0%}{D['cost'].mean():>8.2%}"
              f"{D['rule'].mean():>+10.2%}{D['hold'].mean():>+10.2%}"
              f"{D['edge'].mean():>+10.2%}{(D['edge'] > 0).mean():>9.1%}")

    print(f"PURE hull55 colour rule — in when the ribbon is green, out when it "
          f"is not.\n{len(R)} names with >= {a.min_bars} bars, "
          f"{R['bars'].sum():,} name-bars, "
          f"{R['start'].min():%Y-%m} → {R['end'].max():%Y-%m}\n")
    print(f"{'universe':<28}{'names':>6}{'trades':>8}{'in mkt':>8}{'cost':>8}"
          f"{'RULE':>10}{'HOLD':>10}{'edge':>10}{'win%':>9}")
    block(R, "every name")
    for lo, hi, lab in ((1e9, 1e99, "liquid  >= Rp1bn/day"),
                        (1e8, 1e9, "middle  Rp0.1-1bn/day"),
                        (0, 1e8, "thin    < Rp0.1bn/day")):
        block(R[(R["tv"] >= lo) & (R["tv"] < hi)], lab)
    #  The by-decile split is the check that withdrew H44. If the edge lives
    #  only in the thin deciles it is non-synchronous pricing, not a strategy.
    R["dec"] = pd.qcut(R["tv"].rank(method="first"), 10, labels=False)
    print()
    for d, g in R.groupby("dec"):
        block(g, f"turnover decile {int(d) + 1}")

    print(f"\nHALF-SPLIT at {MID:%Y-%m-%d} — the only replication test this "
          f"repo trusts (A18).")
    print(f"{'universe':<28}{'names':>7}{'RULE early':>12}{'HOLD early':>12}"
          f"{'edge':>9}{'names':>7}{'RULE late':>12}{'HOLD late':>12}"
          f"{'edge':>9}{'both':>7}")
    for lo, hi, lab in ((0, 1e99, "every name"),
                        (1e9, 1e99, "liquid  >= Rp1bn/day"),
                        (1e8, 1e9, "middle  Rp0.1-1bn/day"),
                        (0, 1e8, "thin    < Rp0.1bn/day")):
        D = R[(R["tv"] >= lo) & (R["tv"] < hi)]
        E = D.dropna(subset=["rule_early"])
        L = D.dropna(subset=["rule_late"])
        if not len(E) or not len(L):
            continue
        de = E["rule_early"].mean() - E["hold_early"].mean()
        dl = L["rule_late"].mean() - L["hold_late"].mean()
        print(f"{lab:<28}{len(E):>7}{E['rule_early'].mean():>+12.2%}"
              f"{E['hold_early'].mean():>+12.2%}{de:>+9.2%}"
              f"{len(L):>7}{L['rule_late'].mean():>+12.2%}"
              f"{L['hold_late'].mean():>+12.2%}{dl:>+9.2%}"
              f"{('YES' if de > 0 and dl > 0 else 'no'):>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
