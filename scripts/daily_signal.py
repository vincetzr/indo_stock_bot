#!/usr/bin/env python3
"""The end-of-day list: IDX names whose Hull ribbon has just turned green.

    python3 scripts/refresh.py            # pull today's prices first
    python3 scripts/daily_signal.py       # then the list

Run it after the close. IDX shuts at 15:50 WIB, so 19:00 Shanghai (11:00 UTC,
18:00 WIB) is two hours after the bell and the daily bar is final.

WHAT "BEST POTENTIAL" MEANS HERE, AND WHY IT IS NOT A RANKING BY UPSIDE.

The screen orders candidates by the MEASURED ASYMMETRY of their current state —
P(touch the target within a year) divided by P(touch the mirror loss) from the
H32 first-passage laws. That choice is not cosmetic. H25 ranked on the upside
alone, found a screen that cleared this project's Bonferroni bar, and H26 then
showed the same screen was indistinguishable from a random cell on the ratio
and compounded at -16%/yr. **A rate is not an objective: check what the
denominator is doing before ranking on the numerator.**

WHAT EVERY ROW CARRIES, because a ticker on its own is an invitation to invent
a reason for it:
  entry      today's close - the level the setup is measured from
  stop       the Hull's own flip price (H40), i.e. the close at which the
             ribbon turns red tomorrow. EXACT arithmetic, not a forecast.
  target     the nearest CONFIRMED swing high above price (H34b: a real level
             beats a displaced one by 6.95pp on the false-break rate)
  odds       P(target) / P(stop-distance loss), both from H32
  cost       the round trip at your broker's schedule plus half a fraksi-harga
             tick each way - the floor, never the estimate

AND THE MEASUREMENT THAT SAYS NOT TO OVERTRADE IT (H39/H40, 891 names):
the Hull round trip wins 32.5% of the time, averages +5.54% a trade, and
compounds at about +1.1%/yr against +9.9% for simply owning the same names over
the same spans. Zero of forty configurations beat buy-and-hold. This list is a
watchlist of names in a measured state. It is not an instruction to buy them.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.cone import (SIGMA_MAX, SIGMA_MIN, p_target_first,   # noqa: E402
                         p_touch, sessions_to, vol_decile)
from hull_stop import flip_price, swing_levels                   # noqa: E402
from hull_trade import COST, states                              # noqa: E402
from paint_suite import tick_of                                  # noqa: E402
from time_price import MIN_BARS, load                            # noqa: E402

OUT = "reports"
MIN_TV = 1e9                  # Rp/day: below this the spread eats the trade
FRESH = 5                     # "just turned green" = within this many sessions
DAYS_PER_SESSION = 365.25 / 252.0


def scan(P: pd.DataFrame, asof: pd.Timestamp,
         target_pct: float = 0.0) -> pd.DataFrame:
    #  COUNTED SEPARATELY FROM THE ROWS THAT SURVIVE, because they are not the
    #  same number and the first version printed one as the other. With the
    #  swing-high target it said "33 names have a green ribbon"; with a fixed
    #  +30% target, same board and same day, it said 126. Two runs of one
    #  quantity disagreeing is a self-contradicting panel, which is the
    #  cheapest bug detector there is. The gap is real and worth seeing: most
    #  green names either have NO confirmed resistance above them (they are at
    #  new highs) or one closer than the 5% floor.
    rows: List[Dict] = []
    n_green = 0
    for tk, g in P.groupby("ticker", sort=False):
        if len(g) < MIN_BARS or g["date"].iloc[-1] != asof:
            continue
        px = g["px"].reset_index(drop=True)
        p = px.to_numpy(float)
        up, on = states(px, 55, "EMA stack")
        if not up[-1]:
            continue                              # ribbon is not green
        n_green += 1
        #  How many sessions since the ribbon turned? A name green for months
        #  is a different setup from one that turned this week, and the request
        #  was for the ones that just turned.
        age = 0
        while age < len(up) - 1 and up[-1 - age - 1]:
            age += 1
        ret1 = px.pct_change()
        sig = float(ret1.rolling(60, min_periods=60).std(ddof=1).iloc[-1])
        #  The panel already carries log turnover per bar; recomputing it from
        #  close x volume would use a different definition from the one every
        #  eligibility filter in this repo is built on.
        rp = float(np.exp(g["log_turnover"]).rolling(20, min_periods=5)
                   .median().iloc[-1])
        if not np.isfinite(sig) or sig <= 0 or rp < MIN_TV:
            continue
        res, sup = swing_levels(p)
        fp = flip_price(px, 55)[-1]
        close = float(p[-1])
        real_close = float(g["close"].iloc[-1])
        #  Either the nearest confirmed swing high (the measured level, H34b)
        #  or a fixed distance. The swing high is the honest default; a fixed
        #  target is offered because H38 measured that a FURTHER target beats a
        #  nearer one (mean net -0.52% five percent short of the level against
        #  +0.35% five percent beyond it), and the nearest high is often close.
        target = (close * (1.0 + target_pct) if target_pct > 0
                  else float(res[-1]) if np.isfinite(res[-1]) else np.nan)
        #  A NEGATIVE FLIP PRICE IS NOT A STOP. On a violently trending name
        #  the solved level can come out below zero, which means no achievable
        #  close would turn the ribbon tomorrow. It is a real answer and it is
        #  not a level anyone can place an order at, so the name is skipped
        #  rather than quoted with an impossible stop.
        stop = float(fp) if np.isfinite(fp) and 0.0 < fp < close else np.nan
        if not np.isfinite(target) or not np.isfinite(stop):
            continue
        d_up, d_dn = target / close - 1.0, 1.0 - stop / close
        #  A TARGET INSIDE 5% IS NOT A TRADE. After ~1% of round-trip cost a
        #  3% target is noise, and H36 measured that the probability laws have
        #  NO discrimination that close in (AUC 0.502 at +5%, 0.524 at +10%) --
        #  almost everything touches a near level inside a year, so ranking on
        #  P(touch) alone promotes exactly the setups worth least.
        if d_up < 0.05 or not 0.02 < d_dn < 0.95:
            continue
        stk = bool(on[-1])
        pu = p_touch(1.0 + d_up, sig, stk)
        pdn = p_touch(1.0 - d_dn, sig, stk)
        tick = tick_of(real_close)
        cost = COST + 2.0 * (tick / 2.0) / real_close
        rows.append({
            "ticker": tk, "close": real_close, "age": age,
            "vol_decile": vol_decile(sig), "ann_vol": sig * np.sqrt(252),
            "turnover_bn": rp / 1e9,
            "stop": stop * real_close / close, "stop_pct": -d_dn,
            "target": target * real_close / close, "target_pct": d_up,
            "p_target": pu, "p_stop": pdn, "odds": pu / pdn,
            "p_first": p_target_first(d_up, d_dn, sig),
            "rr": d_up / d_dn,
            #  EXPECTANCY, which is what a bracket is actually worth: the
            #  target times the chance of reaching it first, minus the stop
            #  times the chance of the other one arriving first, minus the
            #  toll. Ranking on the odds RATIO instead promoted setups with a
            #  3% target and a 24% stop -- a ratio above 1.0 and an expectancy
            #  well below zero.
            "ev": (p_target_first(d_up, d_dn, sig) * d_up
                   - (1.0 - p_target_first(d_up, d_dn, sig)) * d_dn
                   - (COST + 2.0 * (tick / 2.0) / real_close)),
            "med_days": sessions_to(1.0 + d_up, sig, "med"),
            "cost": cost,
            "in_domain": SIGMA_MIN <= sig <= SIGMA_MAX})
    out = pd.DataFrame(rows)
    out.attrs["n_green"] = n_green
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--fresh", type=int, default=FRESH,
                    help="max sessions since the ribbon turned green")
    ap.add_argument("--target-pct", type=float, default=0.0,
                    help="use a fixed %% target instead of the swing high")
    a = ap.parse_args()

    P = load(holdout=True)
    P["px"] = P["adj_close"]
    P = P[P["tradeable"].astype(bool)].sort_values(["ticker", "date"])
    #  A11's defect: a partial refresh leaves a handful of names on the last
    #  date, and scanning those is scanning a watchlist, not the market.
    counts = P.groupby("date")["ticker"].size()
    good = counts[counts >= 0.5 * counts.max()]
    asof = pd.Timestamp(good.index.max())
    S = scan(P, asof, a.target_pct)
    if S.empty:
        print(f"no candidates on {asof:%Y-%m-%d}")
        return 0
    fresh = S[S["age"] <= a.fresh].copy()
    #  Rank on EXPECTANCY, never on the upside alone (H25 -> H26) and never on
    #  the odds ratio alone (which promotes a near target with a far stop).
    fresh = fresh.sort_values("ev", ascending=False).head(a.top)

    print(f"IDX SUITE — end-of-day scan, {asof:%A %d %B %Y}")
    print(f"{S.attrs['n_green']} names have a green hull55 ribbon; "
          f"{len(S)} of them carry a quotable setup (liquid enough, a "
          f"confirmed level at least 5% above, a reachable stop); "
          f"{int((S['age'] <= a.fresh).sum())} of THOSE turned green within "
          f"{a.fresh} sessions.")
    print(f"Showing the top {len(fresh)} by EXPECTANCY, target = "
          f"{'fixed +' + format(a.target_pct, '.0%') if a.target_pct > 0 else 'nearest confirmed swing high'}.\n")
    print(f"{'ticker':<8}{'close':>9}{'age':>5}{'stop':>9}{'':>7}"
          f"{'target':>9}{'':>7}{'P(1st)':>8}{'R:R':>6}{'EV':>8}"
          f"{'days':>6}{'vol':>5}{'cost':>7}")
    for _, r in fresh.iterrows():
        flag = "" if r["in_domain"] else " !"
        print(f"{r['ticker']:<8}{r['close']:>9,.0f}{int(r['age']):>5}"
              f"{r['stop']:>9,.0f}{r['stop_pct']:>7.1%}"
              f"{r['target']:>9,.0f}{r['target_pct']:>7.1%}"
              f"{r['p_first']:>8.0%}{r['rr']:>6.2f}{r['ev']:>8.2%}"
              f"{r['med_days']:>6.0f}"
              f"{int(r['vol_decile']):>5}{r['cost']:>7.2%}{flag}")
    print("\n  age   = sessions since the ribbon turned green")
    print("  stop  = the close at which the hull55 ribbon turns RED tomorrow")
    print("          (exact arithmetic; it says WHERE, never WHEN)")
    print("  target= nearest CONFIRMED swing high above price")
    print("  P(1st) = chance the target arrives BEFORE the stop, measured")
    print("  R:R    = target distance divided by stop distance")
    print("  EV     = P(1st)*target - (1-P(1st))*stop - cost. THE COLUMN THAT")
    print("           MATTERS: a positive number is a setup worth taking on")
    print("           its own arithmetic, a negative one is not.")
    print("  days  = median sessions to the target IF it gets there")
    print("  !     = volatility outside the fitted range; the odds are")
    print("          extrapolation and should not be leaned on")
    print("\n  H39/H40, 891 names: this round trip wins 32.5% of trades,")
    print("  averages +5.54%, and compounds at ~+1.1%/yr against ~+9.9% for")
    print("  simply owning the same names. It is a watchlist, not a buy list.")
    n_pos = int((fresh["ev"] > 0).sum())
    print(f"\n  {n_pos} of the {len(fresh)} shown have a POSITIVE expectancy.")
    S.to_csv(os.path.join(OUT, "daily_signal.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
