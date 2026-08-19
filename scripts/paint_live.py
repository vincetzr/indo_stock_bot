#!/usr/bin/env python3
"""Two questions the painter has to answer to be worth anything.

1. HALF-CHART TEST. Hide everything after a cut, paint forward bar by bar with
   only what a trader would have had, then compare the result to the painting
   made with the whole series. If the algorithm cheats, the two disagree on the
   hidden half; if it does not, they agree everywhere except the leg that was
   still open at each moment. This is the honest version of "give it half the
   chart and see if it draws the same thing".

2. WHERE THE LIGHT COMES ON. A reversal band has an exact trigger price at every
   moment, so "when will it flip" is not a forecast - it is arithmetic:

       inside a green leg -> it turns red at   running high x (1 - band)
       inside a red leg   -> it turns green at running low  x (1 + band)

   The level is known now. What is unknown is whether price gets there, and the
   script says which of those two things it is reporting.

    python3 scripts/paint_live.py --min-mcap 1e13
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
from leg_signals import market_caps            # noqa: E402
from legpaint import unadjusted_weekly, zigzag_labels   # noqa: E402


def band_state(px: np.ndarray, band: float) -> Tuple[np.ndarray, np.ndarray]:
    """Causal leg colour, and the trigger price that would flip it, bar by bar."""
    n = len(px)
    state = np.zeros(n, dtype=np.int8)
    trig = np.full(n, np.nan)
    long = False
    ext = px[0]
    for i in range(n):
        p = px[i]
        if long:
            if p > ext:
                ext = p
            elif 1.0 - p / ext >= band - 1e-12:
                long = False
                ext = p
        else:
            if p < ext:
                ext = p
            elif p / ext - 1.0 >= band - 1e-12:
                long = True
                ext = p
        state[i] = 1 if long else 0
        trig[i] = ext * (1.0 - band) if long else ext * (1.0 + band)
    return state, trig


def half_chart_test(px: np.ndarray, band: float, cut_frac: float = 0.5,
                    ages: Tuple[int, ...] = (0, 4, 6, 13)) -> Dict[str, float]:
    """Hide the second half, paint it forward, and compare two different ways.

    ``prefix_identical`` - painting the FIRST half using only the first half must
    give exactly the painting the full series gives that half. This is the
    no-cheating proof: if the rule peeked, the two would differ.

    ``agree@k`` - for a hidden bar, the colour it carries once it is k weeks old,
    against the colour the full series finally assigns it. Asking this at k=0 is
    meaningless - the newest bar is always inside the leg still in progress, so
    it is never settled. That is a property of the question, not of the rule.
    """
    n = len(px)
    cut = int(n * cut_frac)
    if cut < 60 or n - cut < 40:
        return {}

    s_full, _ = band_state(px, band)
    s_part, _ = band_state(px[:cut], band)
    out: Dict[str, float] = {
        "hidden_bars": int(n - cut),
        "prefix_identical": bool(np.array_equal(s_full[:cut], s_part)),
    }

    final = zigzag_labels(px, band, drop_last=False)
    for k in ages:
        hit = tot = 0
        for now in range(cut, n):
            i = now - k
            if i < cut:
                continue
            lab = zigzag_labels(px[:now + 1], band, drop_last=True)
            if np.isfinite(lab[i]) and np.isfinite(final[i]):
                tot += 1
                hit += int(lab[i] == final[i])
        out[f"agree_{k}w"] = hit / tot if tot else np.nan
        out[f"settled_{k}w"] = tot / max(n - cut - k, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mcap", type=float, default=1e13)
    ap.add_argument("--band", type=float, default=0.12)
    ap.add_argument("--ticker", default="ADRO")
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    mc = market_caps()
    big = sorted(mc[mc >= args.min_mcap].index)
    series: Dict[str, pd.Series] = {}
    for t in big:
        w = unadjusted_weekly(loader, t, start="2012-01-01")
        if w is not None and len(w) >= 200:
            series[t] = w
    print(f"{len(series)} big caps at Rp{args.min_mcap/1e12:,.0f}T+ with 200+ weekly bars")

    # ------------------------------- half-chart ------------------------------ #
    rows = []
    for t, w in series.items():
        r = half_chart_test(w.to_numpy(float), args.band)
        if r:
            rows.append({"ticker": t, **r})
    H = pd.DataFrame(rows)
    H.to_csv("reports/half_chart_test.csv", index=False)

    print(f"\n{'=' * 92}\n HALF-CHART TEST — hide the second half, paint it forward"
          f"\n{'=' * 92}")
    print(f" {len(H)} names, {H['hidden_bars'].sum():,} hidden weeks re-painted "
          f"one bar at a time")
    print(f"\n NO-CHEATING PROOF")
    print(f"   painting the visible half with ONLY the visible half reproduces")
    print(f"   the full-chart painting of it exactly on "
          f"{int(H['prefix_identical'].sum())} of {len(H)} names")
    print(f"\n HIDDEN HALF — colour a bar carries once it is k weeks old,")
    print(f" against the colour the finished chart gives it")
    print(f"   {'age':>5}{'agrees':>10}{'settled':>10}{'names at 90%+':>16}")
    for k in (0, 4, 6, 13):
        c, s_ = f"agree_{k}w", f"settled_{k}w"
        if c in H.columns:
            print(f"   {k:>4}w{H[c].median():>10.1%}{H[s_].median():>10.1%}"
                  f"{(H[c] >= 0.90).sum():>10} of {len(H)}")

    # ------------------------------- triggers -------------------------------- #
    print(f"\n{'=' * 92}\n WHERE THE LIGHT COMES ON — {args.band:.0%} band, "
          f"as of the last weekly close\n{'=' * 92}")
    tr = []
    for t, w in series.items():
        px = w.to_numpy(float)
        st, trig = band_state(px, args.band)
        gap = trig[-1] / px[-1] - 1.0
        tr.append({"ticker": t, "date": w.index[-1], "price": px[-1],
                   "leg": "GREEN" if st[-1] else "RED",
                   "trigger": trig[-1], "move_needed": gap,
                   "mcap": float(mc.get(t, np.nan))})
    T = pd.DataFrame(tr).sort_values(["leg", "move_needed"],
                                     key=lambda s: s.abs() if s.name == "move_needed" else s)
    T.to_csv("reports/live_triggers.csv", index=False)
    green = T[T["leg"] == "GREEN"]
    red = T[T["leg"] == "RED"]
    print(f" {len(green)} names in a GREEN leg, {len(red)} in a RED leg\n")
    print(f" {'ticker':<8}{'leg':<7}{'price':>9}{'flips at':>11}{'move needed':>13}"
          f"{'action if hit':>16}")
    for _, r in pd.concat([green.head(10), red.head(10)]).iterrows():
        act = "SELL" if r["leg"] == "GREEN" else "BUY"
        print(f" {r['ticker']:<8}{r['leg']:<7}{r['price']:>9,.0f}"
              f"{r['trigger']:>11,.0f}{r['move_needed']:>+13.1%}{act:>16}")

    if args.ticker.upper() in series:
        r = T[T["ticker"] == args.ticker.upper()].iloc[0]
        print(f"\n {args.ticker}: in a {r['leg']} leg at {r['price']:,.0f}. "
              f"It flips at {r['trigger']:,.0f} "
              f"({r['move_needed']:+.1%} from here) and that would be a "
              f"{'SELL' if r['leg'] == 'GREEN' else 'BUY'}.")
    print("\n The trigger PRICE is arithmetic and known now. Whether price reaches")
    print(" it is not, and nothing here claims to know that.")
    print("\n -> reports/half_chart_test.csv, reports/live_triggers.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
