#!/usr/bin/env python3
"""Is the method still working? Re-measured, per timeframe, per period, per size.

This is the self-check, and it is built to be able to say NO. Three benchmarks,
because only the third one is hard to beat by accident:

    vs buy-and-hold      the obvious comparison, and a weak one - in a falling
                         market anything that sits in cash wins for free
    vs the same-exposure the honest one. Take the rule's own time-in-market and
       null              spend it at random: exposure x the hold log return. A
                         rule earns the word "timing" only by beating this.
    drift                the same numbers split by period. A method that worked
                         until 2023 and stopped is not a working method, and an
                         all-history average hides exactly that.

Colour accuracy is deliberately NOT the headline. Result 91 measured ~94%
directional accuracy alongside losing money, so an accuracy number that is not
sitting next to a P/L number is worse than no number at all. Both are reported.

Size buckets are included because the user asked to expand past big caps, and
because the toll law predicts small caps should be WORSE: the tick is a larger
share of a cheap price, so the realised entry overshoot - and therefore the
break-even leg - grows. That prediction is tested rather than assumed.

Writes reports/method_review.json with a pass/fail per timeframe so a scheduled
run can alert on degradation instead of a person reading a table every morning.

    python3 scripts/method_review.py
    python3 scripts/method_review.py --universe all
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config           # noqa: E402
from idxbot.data.cache import Cache             # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV        # noqa: E402
from account_sim import load_ohlc               # noqa: E402
from capture_toll import realised_tolls         # noqa: E402
from leg_signals import market_caps             # noqa: E402
from paint_daily import unadjusted_daily        # noqa: E402
from paint_live import band_state               # noqa: E402

FEE = 0.0056
BANDS = {"weekly": 0.12, "daily": 0.08}
# A timeframe passes only if it beats the null. Beating buy-and-hold is not a
# threshold because a cash-heavy rule does that for free in a falling market.
PASS_EDGE = 0.0
PASS_SHARE = 0.55


def resample(s: pd.Series, timeframe: str) -> pd.Series:
    return s.resample("W-FRI").last().dropna() if timeframe == "weekly" else s


def measure(s: pd.Series, band: float) -> Optional[Dict[str, float]]:
    """Everything the review needs from one name on one timeframe, causally."""
    px = s.to_numpy(float)
    if len(px) < 120:
        return None
    st, _ = band_state(px, band)
    r = np.diff(np.log(px))
    held = st[:-1].astype(bool)
    trips = int((np.diff(st.astype(int)) == 1).sum())
    gross = float(r[held].sum())
    net = gross + trips * np.log(1.0 - FEE)
    hold = float(np.log(px[-1] / px[0]))
    exposure = float(held.mean())
    rows = realised_tolls(px, band)
    pl = np.array([x[2] for x in rows]) - FEE if rows else np.array([])
    toll = np.array([(1 + x[0]) / (1 - x[1]) for x in rows]) if rows else np.array([])
    return {
        "log": net, "hold": hold, "null": exposure * hold,
        "edge": net - exposure * hold, "exposure": exposure, "trips": trips,
        "win": float((pl > 0).mean()) if len(pl) else np.nan,
        "median_pl": float(np.median(pl)) if len(pl) else np.nan,
        "toll": float(np.median(toll)) if len(toll) else np.nan,
        "bars": len(px),
    }


def roll_up(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {}
    D = pd.DataFrame(rows)
    return {
        "names": len(D),
        "log": float(D["log"].median()),
        "hold": float(D["hold"].median()),
        "null": float(D["null"].median()),
        "edge": float(D["edge"].median()),
        "beats_hold": float((D["log"] > D["hold"]).mean()),
        "beats_null": float((D["log"] > D["null"]).mean()),
        "exposure": float(D["exposure"].median()),
        "trips": float(D["trips"].median()),
        "win": float(D["win"].median()),
        "median_pl": float(D["median_pl"].median()),
        "toll": float(D["toll"].median()),
    }


def bucket_of(cap: float) -> str:
    if cap >= 1e13:
        return "big"
    if cap >= 1e12:
        return "mid"
    return "small"


# Size proxy that exists for EVERY cached name, not just the 59 with market caps.
# Cap data left the small bucket at n=3, which no conclusion can rest on. Median
# daily turnover is available for every name, is causal, and separates a
# Rp1bn/day stock from a Rp1tn/day one - the distinction the question is about.
TURNOVER_EDGES = (5e9, 1e11)     # < Rp5bn/day = small; >= Rp100bn/day = large


def turnover_bucket(turnover: float) -> str:
    if not np.isfinite(turnover):
        return "unknown"
    if turnover >= TURNOVER_EDGES[1]:
        return "large"
    return "mid" if turnover >= TURNOVER_EDGES[0] else "small"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="all", choices=["big", "all"])
    ap.add_argument("--min-price", type=float, default=50.0,
                    help="drop names below this: one tick is 2-100% of price and "
                         "the return clip used everywhere else is wrong for them")
    ap.add_argument("--periods", type=int, default=4, help="how many recent years")
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    mc = market_caps()
    if args.universe == "big":
        names = sorted(mc[mc >= 1e13].index)
    else:
        # every IDX equity in the cache, not just the ones with fundamentals
        # cache files are named TICKER.JK.csv.gz; the .JK must come off, and
        # non-equities (^JKSE, DX-Y.NYB, futures) must not enter an IDX universe
        names = sorted({os.path.basename(f).split(".")[0].upper()
                        for f in glob.glob(os.path.join(
                            cfg.path("data.cache_dir", "data/cache"),
                            "ohlcv", "*.JK.csv.gz"))})
        names = [n for n in names if n.isalpha() and len(n) == 4]

    series: Dict[str, pd.Series] = {}
    tsizes: Dict[str, str] = {}
    dropped_cheap = dropped_short = 0
    for t in names:
        df = load_ohlc(loader, t)
        if df is None or len(df) < 400:
            dropped_short += 1
            continue
        s = df["close"]
        s = s[s.index >= pd.Timestamp("2015-01-01")]
        if len(s) < 400:
            dropped_short += 1
            continue
        if float(s.median()) < args.min_price:
            dropped_cheap += 1
            continue
        series[t] = s
        tv = float((df["close"] * df["volume"]).tail(250).median())
        tsizes[t] = turnover_bucket(tv)
    if not series:
        raise SystemExit("no usable names")

    counts = pd.Series(tsizes).value_counts().to_dict()
    print(f"{'=' * 100}\n METHOD REVIEW — {len(series)} names by turnover: "
          f"{counts}\n{'=' * 100}")
    print(f" dropped {dropped_cheap} under Rp{args.min_price:.0f} (one tick is "
          f"2-100% of price there, so the +/-35%\n return clip used elsewhere "
          f"mangles them) and {dropped_short} with too little history.\n")

    out: Dict[str, object] = {"generated": pd.Timestamp.now("UTC").isoformat(),
                              "names": len(series)}

    # ---------------------------------------------------------------- by timeframe
    print(f"{'=' * 100}\n BY TIMEFRAME — whole history\n{'=' * 100}")
    print(f" {'timeframe':<11}{'names':>7}{'trips':>7}{'win':>7}{'med trip':>10}"
          f"{'toll':>8}{'invested':>10}{'vs hold':>10}{'vs NULL':>10}"
          f"{'beats null':>12}")
    tf_summary = {}
    for tf, band in BANDS.items():
        rows = [m for m in (measure(resample(s, tf), band)
                            for s in series.values()) if m]
        agg = roll_up(rows)
        tf_summary[tf] = agg
        print(f" {tf + ' ' + format(band, '.0%'):<11}{agg['names']:>7}"
              f"{agg['trips']:>7.0f}{agg['win']:>7.0%}{agg['median_pl']:>+10.1%}"
              f"{agg['toll'] - 1:>8.1%}{agg['exposure']:>10.0%}"
              f"{agg['log'] - agg['hold']:>+10.3f}{agg['edge']:>+10.3f}"
              f"{agg['beats_null']:>12.0%}")
    out["timeframes"] = tf_summary

    # ---------------------------------------------------------------- by size
    print(f"\n{'=' * 100}\n BY SIZE — does it work better on small caps?\n{'=' * 100}")
    print(f" buckets by median daily turnover, which exists for every cached "
          f"name; market cap is\n only cached for {int((mc > 0).sum())} tickers "
          f"and left the small bucket at n=3.\n")
    print(f" {'bucket':<8}{'names':>7}{'trips':>7}{'win':>7}{'med trip':>10}"
          f"{'toll':>8}{'vs NULL':>10}{'beats null':>12}")
    size_summary = {}
    for b in ("large", "mid", "small"):
        sel = [t for t in series if tsizes[t] == b]
        rows = [m for m in (measure(series[t], BANDS["daily"]) for t in sel) if m]
        if not rows:
            continue
        agg = roll_up(rows)
        size_summary[b] = agg
        print(f" {b:<8}{agg['names']:>7}{agg['trips']:>7.0f}{agg['win']:>7.0%}"
              f"{agg['median_pl']:>+10.1%}{agg['toll'] - 1:>8.1%}"
              f"{agg['edge']:>+10.3f}{agg['beats_null']:>12.0%}")
    out["sizes"] = size_summary
    if {"large", "small"} <= set(size_summary):
        dt = size_summary["small"]["toll"] - size_summary["large"]["toll"]
        print(f"\n realised toll, small minus big: {dt:+.1%} of price — the toll "
              f"law predicts this is\n POSITIVE, because a fixed tick is a bigger "
              f"share of a cheaper price.")
        print(f" {'CONFIRMED' if dt > 0 else 'NOT CONFIRMED'}: small caps carry a "
              f"{'higher' if dt > 0 else 'lower'} break-even than large ones.")
        if dt <= 0:
            print(f" Note the screen works against this test: names under "
                  f"Rp{args.min_price:.0f} were dropped, and those\n are exactly "
                  f"the ones where a fixed tick is the largest share of price.")
        for b, agg in size_summary.items():
            if agg["names"] < 8:
                print(f" ! the '{b}' bucket has only {agg['names']:.0f} names — "
                      f"too few to conclude anything from.")

    # ---------------------------------------------------------------- drift
    print(f"\n{'=' * 100}\n DRIFT — the same daily rule, split by year\n{'=' * 100}")
    last_year = max(s.index[-1].year for s in series.values())
    years = list(range(last_year - args.periods + 1, last_year + 1))
    print(f" {'year':<7}{'names':>7}{'trips':>7}{'win':>7}{'med trip':>10}"
          f"{'hold':>9}{'signals':>10}{'vs NULL':>10}{'beats null':>12}")
    drift = {}
    for y in years:
        rows = []
        for s in series.values():
            sub = s[s.index.year == y]
            if len(sub) < 150:
                continue
            m = measure(sub, BANDS["daily"])
            if m:
                rows.append(m)
        if not rows:
            continue
        agg = roll_up(rows)
        drift[str(y)] = agg
        print(f" {y:<7}{agg['names']:>7}{agg['trips']:>7.0f}{agg['win']:>7.0%}"
              f"{agg['median_pl']:>+10.1%}{agg['hold']:>+9.3f}{agg['log']:>+10.3f}"
              f"{agg['edge']:>+10.3f}{agg['beats_null']:>12.0%}")
    # a year whose name count collapses is a STALE CACHE, not a market event
    if drift:
        base = max(a["names"] for a in drift.values())
        for y, a in drift.items():
            if a["names"] < 0.5 * base:
                print(f" ! {y} covers only {a['names']:.0f} of {base:.0f} names — "
                      f"the daily cache is refreshed for a\n   subset, so that "
                      f"row is a sample artefact, not a change in the market.")
    out["drift"] = drift

    # ---------------------------------------------------------------- verdict
    print(f"\n{'=' * 100}\n VERDICT\n{'=' * 100}")
    verdicts = {}
    for tf, agg in tf_summary.items():
        ok = agg["edge"] > PASS_EDGE and agg["beats_null"] >= PASS_SHARE
        verdicts[tf] = "PASS" if ok else "FAIL"
        print(f" {tf:<8} {verdicts[tf]}   edge vs null {agg['edge']:+.3f}, "
              f"clears it on {agg['beats_null']:.0%} of names "
              f"(needs > {PASS_EDGE:+.2f} and >= {PASS_SHARE:.0%})")
    out["verdicts"] = verdicts

    if drift:
        e = [drift[k]["edge"] for k in sorted(drift)]
        trend = e[-1] - e[0]
        print(f"\n drift in the edge over {len(e)} years: {e[0]:+.3f} -> "
              f"{e[-1]:+.3f} ({trend:+.3f})")
        print(f" {'still negative throughout' if max(e) <= 0 else 'not uniformly negative — check the positive years'}")
        out["drift_trend"] = float(trend)

    if all(v == "FAIL" for v in verdicts.values()):
        print("\n Every timeframe fails the null. The rule remains what Results "
              "110-113 measured:\n exact about where price is, and not "
              "predictive about where it goes. Nothing in\n this review changes "
              "that, and the review is built to notice if it ever does.")
    with open("reports/method_review.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("\n -> reports/method_review.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
