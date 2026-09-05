#!/usr/bin/env python3
"""H56 — put a stop and a take-profit on THIS basket and see what happens.

THE CHALLENGE, in the user's words: "how can you trade without sl and tp?"

It is a fair challenge and one framing in my answer was wrong. The H54 rule DOES
have a stop: it sells when `hi52` falls under the keep threshold, which for a
name at its 52-week high is about −12%. What it has no fixed-percentage stop
and no take-profit, and — the real hole — it is only EVALUATED QUARTERLY, so a
name can fall 40% in month one and still be held in month three.

Every prior exit study in this repo (H17, H18, H35, H38, H40, H47 — 169
configurations) was run on a DIFFERENT entry rule. Citing them at someone asking
about THIS basket is an argument, not a measurement. So this file bolts stops,
take-profits and faster monitoring onto the H54 sticky rule directly, with
portfolio accounting, and reports what they do.

WHAT MAKES THIS DIFFERENT FROM THE PRIOR STUDIES
  * The exit is checked DAILY between quarterly selections, so a stop can
    actually fire when it would fire in real life, rather than at the next mark.
  * MAX DRAWDOWN of the PORTFOLIO is reported alongside CAGR, because that is
    what the question is really about and it is the statistic H20 found stops
    make WORSE while making every individual position look safer.
  * Both halves, and the same six rebalance calendars H54 uses, because a
    single calendar is a start date and not a result (A39).

REGISTERED, before any cell was scored.
  S1  A hard stop cuts the worst SINGLE-NAME loss (this is arithmetic, not a
      prediction) but does NOT cut the PORTFOLIO's maximum drawdown.
      PREDICTION: confirmed. H20 measured `stop 25%` producing the worst
      portfolio drawdown in its table (−68.7%) while capping every name at −25%,
      because it realises losses and redeploys into the same falling regime.
      If this reverses on the H54 rule, H20 does not generalise and the stop
      goes into the shipped rule.
  S2  Checking the keep band DAILY instead of quarterly improves CAGR.
      PREDICTION: it does not — it raises turnover, and H43 measured the
      frequency curve as humped with monthly, quarterly and six-month mutually
      indistinguishable. This is the honest form of the user's objection and it
      is the one I most expect to be wrong about.
  S3  PREDICTED NULL. A take-profit does not improve CAGR at any level.
      A21 measured selling everything at 2x costing 374 points of mean return
      for 9.5 points of hit rate; H40 measured a take-profit lifting the win
      rate to 46.6% while taking CAGR to −1.98%. If a take-profit helps HERE,
      that is a real and surprising finding.
  S4  Nothing beats the plain quarterly rule on CAGR in BOTH HALVES.

THRESHOLD, fixed now: a variant is adopted only if it beats the base rule on
CAGR in both halves at a majority of the six calendars AND does not worsen
portfolio max drawdown. Anything else is reported and not shipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from bhbench import IDX_YIELD, INDEX, MIN_BASKET, MIN_UNIV, load   # noqa: E402
from paint_suite import tick_of                                    # noqa: E402

ENTRY_HI, ENTRY_VOL = 0.90, 0.50
KEEP_HI, KEEP_VOL = 0.80, 0.60
K = 10
FEE = 0.0056
SPREAD_MULT = 0.5
FREQ = 63


def prep(P: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional thresholds, so the keep band can be checked DAILY.

    The thresholds are percentiles of that day's eligible board — the same
    definition the quarterly rule uses, evaluated every session instead of four
    times a year. Computed with one groupby, never on a pivot (A11).
    """
    P = P.dropna(subset=["hi52", "vol60"]).copy()
    e = P[P["elig"]]
    q = e.groupby("date").agg(
        hi_entry=("hi52", lambda s: s.quantile(ENTRY_HI)),
        hi_keep=("hi52", lambda s: s.quantile(KEEP_HI)),
        vol_entry=("vol60", lambda s: s.quantile(ENTRY_VOL)),
        vol_keep=("vol60", lambda s: s.quantile(KEEP_VOL)),
        n_elig=("hi52", "size")).reset_index()
    P = P.merge(q, on="date", how="left")
    P["in_entry"] = (P["elig"] & (P["hi52"] >= P["hi_entry"])
                     & (P["vol60"] <= P["vol_entry"]))
    P["in_keep"] = (P["elig"] & (P["hi52"] >= P["hi_keep"])
                    & (P["vol60"] <= P["vol_keep"]))
    P["cost"] = FEE + SPREAD_MULT * np.array(
        [tick_of(p) for p in P["close"].to_numpy(float)]) / P["close"]
    return P.sort_values(["date", "ticker"])


def _dd(curve: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of the PORTFOLIO.

    The statistic the question is actually about, and the one H20 found stops
    make worse while making every individual position look safer.
    """
    peak = np.maximum.accumulate(curve)
    return float(np.min(curve / peak) - 1.0)


def walk(P: pd.DataFrame, offset: int = 0, stop: float = 0.0,
         tp: float = 0.0, tp_frac: float = 1.0, trail: float = 0.0,
         daily_keep: bool = False, redeploy: bool = True) -> Dict:
    """One path. Selection quarterly; exits checked EVERY SESSION.

    `stop` / `tp` are fractions from the ENTRY price of that position, checked
    on the close (A27: filling at the nominal level flatters tight stops, and on
    IDX a breach can gap to auto-rejection where nothing trades at all).
    `daily_keep` checks the keep band every session instead of at marks only.
    `tp_frac` < 1 SCALES OUT: sell that fraction at the target and let the rest
    run, which is the only form of take-profit that does not truncate the right
    tail outright. `trail` is a trailing stop from the position's own peak,
    which unlike a fixed target rises with a winner instead of capping it.
    """
    dates = np.sort(P["date"].unique())
    marks = set(dates[offset::FREQ].tolist())
    by = {d: g for d, g in P.groupby("date", sort=False)}

    #  EXPLICIT RUPIAH ACCOUNTING. `cash` plus a rupiah value per position, and
    #  equity is their sum every day. A first draft carried "weights" that were
    #  also growing with price and then reconstructed equity from them, which
    #  double-counted cash and would have printed a plausible curve — the
    #  failure mode A37 records in `book()` and A39 in the cash drag. Two
    #  invariants below make the accounting checkable rather than trusted.
    cash = 1.0
    held: Dict[str, Dict] = {}          # ticker -> {val, entry, px}
    curve, t0, n_stop, n_tp, n_band, n_sel = [], None, 0, 0, 0, 0
    worst_name = 0.0

    for d in dates:
        g = by.get(d)
        if g is None:
            continue
        px = dict(zip(g["ticker"], g["adj_close"]))
        keep = dict(zip(g["ticker"], g["in_keep"]))
        cost = dict(zip(g["ticker"], g["cost"]))

        # ---- mark the book to today ------------------------------------
        for t, h in held.items():
            p = px.get(t, h["px"])
            if p > 0:
                h["val"] *= p / h["px"]
                h["px"] = p

        # ---- EXITS, checked EVERY SESSION ------------------------------
        drop = []
        scale = []
        for t, h in held.items():
            r = h["px"] / h["entry"] - 1.0
            h["peak"] = max(h.get("peak", h["entry"]), h["px"])
            worst_name = min(worst_name, r)
            if stop and r <= -stop:
                drop.append((t, "stop"))
            elif trail and h["px"] <= h["peak"] * (1.0 - trail):
                drop.append((t, "stop"))
            elif tp and r >= tp and not h.get("tp_done"):
                if tp_frac >= 1.0:
                    drop.append((t, "tp"))
                else:
                    scale.append(t)
            elif daily_keep and not keep.get(t, False):
                drop.append((t, "band"))
        for t in scale:
            #  Sell part, bank it, and mark so it fires ONCE. A target that can
            #  re-arm on the same position is a different rule and a much
            #  churnier one.
            take = held[t]["val"] * tp_frac
            held[t]["val"] -= take
            cash += take * (1.0 - cost.get(t, FEE))
            held[t]["tp_done"] = True
            n_tp += 1
        for t, why in drop:
            #  Fill at the CLOSE, not at the nominal level (A27): a bar that
            #  breaches −20% often closes at −25%, and on IDX it can gap to
            #  auto-rejection where nothing trades at all. Filling at the level
            #  flatters exactly the tight stops this test is about.
            cash += held[t]["val"] * (1.0 - cost.get(t, FEE))
            n_stop += why == "stop"
            n_tp += why == "tp"
            n_band += why == "band"
            del held[t]

        # ---- SELECTION, quarterly --------------------------------------
        if d in marks:
            e = g[g["elig"]]
            if len(e) >= MIN_UNIV:
                if not daily_keep:
                    for t in [t for t in held if not keep.get(t, False)]:
                        cash += held[t]["val"] * (1.0 - cost.get(t, FEE))
                        n_band += 1
                        del held[t]
                #  A NAME THAT STOPS PRINTING IS A DELISTING, realised at its
                #  last price, never carried forward at cost (H41's bug).
                for t in [t for t in held if t not in px]:
                    cash += held[t]["val"] * (1.0 - FEE)
                    del held[t]
                need = K - len(held)
                if need > 0:
                    cand = e[e["in_entry"] & ~e["ticker"].isin(list(held))]
                    for t in cand.nlargest(need, "hi52")["ticker"]:
                        held[t] = {"val": 0.0, "entry": px[t], "px": px[t],
                                   "peak": px[t]}
                        n_sel += 1
                if len(held) >= MIN_BASKET:
                    total = cash + sum(h["val"] for h in held.values())
                    target = total / len(held)
                    #  Cost is charged on TURNOVER, one side each way, using
                    #  each name's own fraksi-harga tick (A38).
                    turn = sum(abs(target - h["val"]) for h in held.values())
                    fee = float(np.mean([cost.get(t, FEE) for t in held]))
                    total -= 0.5 * turn * fee
                    target = total / len(held)
                    for h in held.values():
                        h["val"] = target
                    cash = 0.0
                    if t0 is None:
                        t0 = d
                else:
                    #  Not enough names to hold a basket: stay in what we have,
                    #  do NOT liquidate into cash. A39 measured that mistake
                    #  (the cash drag) as worth more than any effect in the
                    #  study it was inside.
                    pass

        eq = cash + sum(h["val"] for h in held.values())
        if t0 is not None:
            curve.append((d, eq))

    if t0 is None or len(curve) < 200:
        return {}
    dts = np.array([c[0] for c in curve])
    eqs = np.array([c[1] for c in curve], float)
    eqs = eqs / eqs[0]
    yrs = (dts[-1] - dts[0]).astype("timedelta64[D]").astype(float) / 365.25
    mid = len(eqs) // 2
    ye = (dts[mid] - dts[0]).astype("timedelta64[D]").astype(float) / 365.25
    yl = (dts[-1] - dts[mid]).astype("timedelta64[D]").astype(float) / 365.25

    def _c(m, y):
        return float(max(m, 1e-9) ** (1.0 / max(y, 1e-9)) - 1.0)

    return {"cagr": _c(eqs[-1], yrs), "dd": _dd(eqs), "years": yrs,
            "early": _c(eqs[mid], ye), "late": _c(eqs[-1] / eqs[mid], yl),
            "start": dts[0], "end": dts[-1], "worst_name": worst_name,
            "n_stop": n_stop, "n_tp": n_tp, "n_band": n_band, "n_sel": n_sel,
            "curve": (dts, eqs)}


def index_cagr(a, b) -> float:
    J = pd.read_csv(INDEX, parse_dates=["date"]).sort_values("date")
    s = J.set_index("date")["close"]
    s = s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]
    y = (s.index[-1] - s.index[0]).days / 365.25
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / y) - 1.0) + IDX_YIELD


ARMS: List[Tuple[str, Dict]] = [
    ("BASE: quarterly keep-band only", {}),
    ("keep band checked DAILY (S2)", {"daily_keep": True}),
    ("+ hard stop 10%", {"stop": 0.10}),
    ("+ hard stop 15%", {"stop": 0.15}),
    ("+ hard stop 20%", {"stop": 0.20}),
    ("+ hard stop 25%", {"stop": 0.25}),
    ("+ hard stop 30%", {"stop": 0.30}),
    #  S3's original three, plus the WIDER targets the first sweep never
    #  reached. If a take-profit is going to be shipped it should be the one
    #  that costs least, and that cannot be known from a grid stopping at +50%.
    ("+ take-profit 20% (S3)", {"tp": 0.20}),
    ("+ take-profit 30% (S3)", {"tp": 0.30}),
    ("+ take-profit 50% (S3)", {"tp": 0.50}),
    ("+ take-profit 75%", {"tp": 0.75}),
    ("+ take-profit 100%", {"tp": 1.00}),
    ("+ take-profit 150%", {"tp": 1.50}),
    #  SCALE-OUT: bank part, let the rest run. The only take-profit that does
    #  not truncate the right tail outright.
    ("+ sell HALF at +50%", {"tp": 0.50, "tp_frac": 0.5}),
    ("+ sell HALF at +100%", {"tp": 1.00, "tp_frac": 0.5}),
    ("+ sell THIRD at +50%", {"tp": 0.50, "tp_frac": 1 / 3}),
    ("+ sell THIRD at +100%", {"tp": 1.00, "tp_frac": 1 / 3}),
    #  A TRAILING stop rises with a winner instead of capping it -- the honest
    #  way to "take profit" without naming a level.
    ("+ trail 25% from peak", {"trail": 0.25}),
    ("+ trail 35% from peak", {"trail": 0.35}),
    ("SHIPPED: stop 20% + sell HALF at +100%",
     {"stop": 0.20, "tp": 1.00, "tp_frac": 0.5}),
    ("+ stop 20% AND take-profit 30%", {"stop": 0.20, "tp": 0.30}),
    ("DAILY band + stop 20%", {"daily_keep": True, "stop": 0.20}),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", type=int, default=6)
    ap.add_argument("--out", default=os.path.join("reports", "stoptest.txt"))
    args = ap.parse_args()

    P = prep(load())
    offs = [int(round(i * FREQ / args.phases)) for i in range(args.phases)]
    L: List[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        L.append(s)

    say(f"H56 — stops and take-profits on the H54 basket. "
        f"{args.phases} rebalance calendars, portfolio accounting.")
    say(f"{P['date'].min().date()} -> {P['date'].max().date()}, "
        f"{P['ticker'].nunique()} names\n")
    say(f"{'arm':<34}{'CAGR':>9}{'lo..hi':>17}{'maxDD':>9}{'early':>9}"
        f"{'late':>9}{'exits/yr':>10}{'worst 1':>9}")
    say("-" * 106)

    base_med = None
    rows = []
    for label, kw in ARMS:
        runs = [walk(P, offset=o, **kw) for o in offs]
        ok = [r for r in runs if r]
        if not ok:
            say(f"{label:<34}  no path")
            continue
        cg = np.array([r["cagr"] for r in ok])
        med = float(np.median(cg))
        if base_med is None:
            base_med = med
        dd = float(np.median([r["dd"] for r in ok]))
        ex = float(np.median([(r["n_stop"] + r["n_tp"] + r["n_band"])
                              / r["years"] for r in ok]))
        wn = float(np.median([r["worst_name"] for r in ok]))
        eb = sum(1 for r in ok
                 if r["early"] > 0 and r["late"] > 0)
        say(f"{label:<34}{med:>9.2%}{cg.min():>8.2%}..{cg.max():<8.2%}"
            f"{dd:>9.1%}{np.median([r['early'] for r in ok]):>9.2%}"
            f"{np.median([r['late'] for r in ok]):>9.2%}{ex:>10.1f}"
            f"{wn:>9.0%}")
        rows.append({"arm": label, "cagr_med": med, "dd": dd,
                     "lo": float(cg.min()), "hi": float(cg.max()),
                     "early": float(np.median([r["early"] for r in ok])),
                     "late": float(np.median([r["late"] for r in ok])),
                     "exits_per_yr": ex, "worst_name": wn,
                     "beats_base": med > base_med,
                     "phases_both_halves_pos": eb, "phases": len(ok)})

    a, b = ok[0]["start"], ok[0]["end"]
    say()
    say(f"IHSG, total-return, same span: {index_cagr(a, b):+.2%}")
    say()
    say("S1: 'worst 1' is the worst single-name loss on the path. Compare it")
    say("    with 'maxDD', the PORTFOLIO's drawdown. If a stop cuts the first")
    say("    and not the second, position-level risk control is not")
    say("    portfolio-level risk control.")

    os.makedirs("reports", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(args.out.replace(".txt", ".json"), "w") as f:
        json.dump(rows, f, indent=1, default=str)
    say(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
