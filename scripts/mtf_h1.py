#!/usr/bin/env python3
"""H55b — the hourly half. What is intraday timing actually worth on IDX?

3,114,456 hourly bars over 766 names, 2023-07-25 to 2026-08-07, sitting in
`data/cache/intraday/` since an early task and never used by any of H1-H54.
Every hypothesis in this repo has been on daily bars.

TWO QUESTIONS, AND THEY HAVE DIFFERENT ANSWERS.

  (1) Can you TRADE the hourly bar? No, and it is subtraction rather than a
      hypothesis. Measured here: the median absolute hourly return is 0.556%
      and one round trip is 0.560% in fees alone, 0.902% with half a
      fraksi-harga tick. The toll is 1.62x the move it is trying to capture and
      only 31.9% of hourly bars move further than one round trip.

  (2) Can the hourly bar improve the FILL on a swing trade you were going to
      take anyway? This is the honest multi-timeframe question, because a fill
      choice adds no round trips at all — it changes WHICH HOUR you buy, not
      how often you trade. It is what this file measures.

THE ORACLE ARM IS THE POINT. Alongside the real entry rules there is an arm
that buys at the LOWEST hourly close of the entry day, chosen with perfect
hindsight. It is a look-ahead by construction and is labelled as one. It exists
because it BOUNDS the question: whatever intraday timing could ever be worth, it
cannot be worth more than the oracle, and if the oracle's advantage is small
relative to the horizon's noise then no real rule can matter and the route is
closed without testing a hundred of them. A19 records that the comparison a
study omits is what manufactures its result; this is the comparison that stops
an hourly-timing programme before it starts.

REGISTERED, before any arm was scored.
  H1a  PREDICTED NULL. An hourly trend confirmation (buy the first hour whose
       close is above the hourly EMA) does NOT beat simply buying the daily
       close, on mean log net of cost, at any horizon. Prediction: null.
  H1b  The ORACLE's advantage over the close, expressed as a multiple of the
       standard error of the horizon's return, is small. Prediction: the oracle
       is worth roughly one intraday range — around 1 to 2% — which is real but
       is a fraction of the 20-day return's dispersion, so even perfect
       hindsight buys little.
  H1c  The bars are unadjusted and must be corrected. Verified before use: the
       last hourly close matches the daily panel's RAW close on 99.54% of
       65,415 name-days and its adj_close on 51.75%, so the panel's own
       adj_close/close factor is applied to every hourly price. Skipping this
       turns a split inside the window into a fake crash.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from paint_suite import tick_of                                  # noqa: E402

H1 = os.environ.get("IDX_H1_PANEL", "")
PANEL = os.path.join("data", "spine", "price_panel.parquet")
FEE = 0.0056
SPREAD_MULT = 0.5
HORIZONS = (5, 20, 60)
SEED = 20260903


def cost_wall(H: pd.DataFrame, P: pd.DataFrame) -> Dict:
    """Question (1): the toll against the move. This is the whole intraday answer."""
    H = H.sort_values(["ticker", "ts"])
    r1 = H.groupby("ticker")["a_close"].pct_change().abs()
    px = H["close"].to_numpy(float)
    spread = np.array([tick_of(p) for p in px]) / px
    Pd = P.sort_values(["ticker", "date"])
    rd = Pd.groupby("ticker")["adj_close"].pct_change().abs()
    rd = rd[Pd["ticker"].isin(H["ticker"].unique()).to_numpy()]
    med_h, med_d = float(r1.median()), float(rd.median())
    out = {"median_abs_1h_return": med_h, "median_abs_daily_return": med_d,
           "median_tick_pct": float(np.median(spread)),
           "zero_return_1h_share": float((r1 < 1e-12).mean()),
           "flat_bar_share": float(((H["open"] == H["high"])
                                    & (H["high"] == H["low"])
                                    & (H["low"] == H["close"])).mean())}
    for nm, c in (("fee_only", FEE),
                  ("fee_half_tick", FEE + 0.5 * float(np.median(spread))),
                  ("fee_full_tick", FEE + 1.0 * float(np.median(spread)))):
        out[nm] = c
        out[nm + "_x_1h_move"] = c / med_h
        out[nm + "_x_daily_move"] = c / med_d
    rt = FEE + SPREAD_MULT * spread
    out["share_1h_bars_clearing_a_round_trip"] = float(
        (r1.to_numpy() > rt).mean())
    return out


def entries(H: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker, day): the fills each arm would have got, plus the outcome.

    NO LOOK-AHEAD EXCEPT IN THE ARM LABELLED ORACLE. The setup is decided on
    day t from daily data; every fill is drawn from day t+1's own hourly bars;
    the outcome runs from the fill to a later daily close.
    """
    H = H.sort_values(["ticker", "ts"]).copy()
    H["hema"] = H.groupby("ticker")["a_close"].transform(
        lambda s: s.ewm(span=10, adjust=False).mean())
    H["above"] = H["a_close"] > H["hema"]
    H["n"] = H.groupby(["ticker", "date"]).cumcount()

    g = H.groupby(["ticker", "date"], sort=False)
    d = g.agg(open_=("a_open", "first"), close=("a_close", "last"),
              lo=("a_low", "min"), hi=("a_high", "max"),
              lo_close=("a_close", "min"), hi_close=("a_close", "max"),
              bars=("a_close", "size"), raw_close=("close", "last"),
              elig=("elig", "last")).reset_index()

    #  First hour of the day whose close is above the hourly EMA — the
    #  "confirmation" entry a multi-timeframe trader would actually use.
    conf = H[H["above"]].groupby(["ticker", "date"], sort=False).agg(
        conf=("a_close", "first"), conf_n=("n", "first")).reset_index()
    d = d.merge(conf, on=["ticker", "date"], how="left")
    #  Mid-session fill: a fixed clock rule with no discretion in it.
    mid = H[H["n"] == 3].groupby(["ticker", "date"], sort=False).agg(
        mid=("a_close", "first")).reset_index()
    d = d.merge(mid, on=["ticker", "date"], how="left")
    return d


def run(H: pd.DataFrame, P: pd.DataFrame, out_txt: str) -> Dict:
    L: List[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        L.append(s)

    say("H55b — THE HOURLY HALF")
    say(f"  {len(H):,} hourly bars, {H['ticker'].nunique()} names, "
        f"{H['date'].min().date()} -> {H['date'].max().date()}")
    say()
    say("(1) THE COST WALL — can you trade the hourly bar at all?")
    cw = cost_wall(H, P)
    say(f"  median |1h return|                    {cw['median_abs_1h_return']:.4%}")
    say(f"  median |daily return|                 {cw['median_abs_daily_return']:.4%}")
    say(f"  median fraksi tick, % of price        {cw['median_tick_pct']:.4%}")
    for nm in ("fee_only", "fee_half_tick", "fee_full_tick"):
        say(f"  round trip, {nm:<16}      {cw[nm]:.4%}"
            f"   = {cw[nm + '_x_1h_move']:.2f}x a median 1h move"
            f"   = {cw[nm + '_x_daily_move']:.2f}x a median daily move")
    say(f"  1h bars clearing one round trip       "
        f"{cw['share_1h_bars_clearing_a_round_trip']:.2%}")
    say(f"  1h bars with zero return              {cw['zero_return_1h_share']:.2%}")
    say(f"  1h bars with no range at all (o=h=l=c) {cw['flat_bar_share']:.2%}")
    say()

    say("(2) WHAT IS THE FILL WORTH? Same trade, same cost, different hour.")
    D = entries(H)
    P2 = P[["ticker", "date", "adj_close"]].sort_values(["ticker", "date"])
    P2 = P2.rename(columns={"adj_close": "px"})
    for k in HORIZONS:
        P2[f"fwd{k}"] = P2.groupby("ticker")["px"].shift(-k)
    D = D.merge(P2, on=["ticker", "date"], how="left")
    D = D[D["elig"].astype(bool) & (D["bars"] >= 5)]
    D["cost"] = FEE + SPREAD_MULT * np.array(
        [tick_of(p) for p in D["raw_close"].to_numpy(float)]) / D["raw_close"]
    say(f"  {len(D):,} eligible entry-days, {D['ticker'].nunique()} names")
    say()

    ARMS = [("daily close (the baseline)", "close"),
            ("day's open (first hour)", "open_"),
            ("mid-session, 4th hour (a clock rule)", "mid"),
            ("first hour above the hourly EMA", "conf"),
            ("ORACLE: lowest hourly close [LOOK-AHEAD]", "lo_close")]

    rows = []
    for k in HORIZONS:
        say(f"---------------------------------------- horizon {k} sessions")
        say(f"{'entry arm':<44}{'n':>9}{'mean log':>10}{'mean':>9}"
            f"{'median':>9}{'win':>7}{'vs close':>10}")
        say("-" * 98)
        base = None
        for name, col in ARMS:
            d = D.dropna(subset=[col, f"fwd{k}"])
            fill = d[col].to_numpy(float)
            r = d[f"fwd{k}"].to_numpy(float) / fill - 1.0 - d["cost"].to_numpy(float)
            r = r[np.isfinite(r) & (r > -1 + 1e-9)]
            if len(r) < 500:
                continue
            ml = float(np.mean(np.log1p(r)))
            if base is None:
                base = ml
            say(f"{name:<44}{len(r):>9,}{ml:>+10.4f}{np.mean(r):>+9.2%}"
                f"{np.median(r):>+9.2%}{np.mean(r > 0):>7.1%}"
                f"{ml - base:>+10.4f}")
            rows.append({"horizon": k, "arm": name, "n": int(len(r)),
                         "mlog": ml, "mean": float(np.mean(r)),
                         "median": float(np.median(r)),
                         "win": float(np.mean(r > 0)), "vs_close": ml - base})
        #  THE MATCHED CHECK, AND IT REVERSES THE NAIVE READING.
        #  The confirmation arm fires on ~24% FEWER days -- it needs an hour
        #  whose close is above the hourly EMA, and on some days there is none.
        #  Scored at its natural size it therefore mixes "which HOUR did I buy"
        #  with "which DAY did I trade", and the second is not a fill effect at
        #  all. Restricting the baseline to exactly the days the confirmation
        #  fired separates them, and the two terms have OPPOSITE SIGNS:
        #  waiting for confirmation makes the FILL worse by ~0.0010 of log at
        #  every horizon, while the days on which it fires are better by
        #  +0.0012 to +0.0036. A35: ask of every headline what the thing in
        #  between is, and whether it was priced.
        d = D.dropna(subset=["close", "conf", f"fwd{k}"])
        if len(d) > 500:
            def _ml(fill, sub):
                r = (sub[f"fwd{k}"].to_numpy(float) / fill - 1.0
                     - sub["cost"].to_numpy(float))
                r = r[np.isfinite(r) & (r > -1 + 1e-9)]
                return float(np.mean(np.log1p(r))) if len(r) else np.nan
            alld = D.dropna(subset=["close", f"fwd{k}"])
            a = _ml(alld["close"].to_numpy(float), alld)
            b = _ml(d["close"].to_numpy(float), d)
            c = _ml(d["conf"].to_numpy(float), d)
            say(f"  MATCHED: close|all {a:+.4f}  close|conf-days {b:+.4f}  "
                f"conf-fill {c:+.4f}")
            say(f"           FILL edge {c - b:+.4f}   "
                f"DAY-SELECTION edge {b - a:+.4f}"
                f"   <- opposite signs; the naive arm reads the second")
            rows.append({"horizon": k, "arm": "confirmation [MATCHED]",
                         "fill_edge": c - b, "day_edge": b - a})

        #  H1b's scale check: the oracle's edge against the horizon's own noise.
        d = D.dropna(subset=["close", f"fwd{k}"])
        sd = float(np.std(d[f"fwd{k}"].to_numpy(float)
                          / d["close"].to_numpy(float) - 1.0, ddof=1))
        orc = [r for r in rows if r["horizon"] == k and r["arm"].startswith("ORACLE")]
        if orc:
            say(f"  H1b: the ORACLE's edge is {orc[0]['vs_close']:+.4f} of log "
                f"against a {k}-session return sd of {sd:.4f}")
            say(f"       => perfect intraday hindsight is worth "
                f"{abs(orc[0]['vs_close']) / sd:.3f} of one standard deviation "
                f"of the outcome it is trying to improve.")
        say()

    os.makedirs("reports", exist_ok=True)
    with open(out_txt, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(out_txt.replace(".txt", ".json"), "w") as f:
        json.dump({"cost_wall": cw, "arms": rows}, f, indent=1, default=str)
    say(f"wrote {out_txt}")
    return {"cost_wall": cw, "arms": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1", default=H1, help="path to the built 1h panel")
    ap.add_argument("--out", default=os.path.join("reports", "mtf_h1.txt"))
    args = ap.parse_args()
    if not args.h1 or not os.path.exists(args.h1):
        raise SystemExit("need --h1 (build it with scripts/build_h1_panel.py)")
    H = pd.read_parquet(args.h1)
    P = pd.read_parquet(PANEL)
    P = P[P["adj_close"] > 0]
    run(H, P, args.out)


if __name__ == "__main__":
    main()
