#!/usr/bin/env python3
"""The signal feed: what changed, what is about to, and how often that has paid.

THE THREE LAYERS, IN THE ORDER THEY WERE ASKED FOR
--------------------------------------------------
    layer 1  news and fundamentals   regime context, dated and sourced
    layer 2  broker-summary flow     who is accumulating, when it is available
    layer 3  technical               the band state and its exact trigger price

Each layer is reported separately and labelled with what it is actually worth,
because they are not worth the same and pretending otherwise is how a signal
feed becomes a horoscope:

    layer 3 is EXACT but not predictive. A flip cannot be missed (Result 109),
            never repaints, and its trigger price is arithmetic known in advance.
            It also loses to buy-and-hold (Result 100) and to random timing at
            the same exposure (Results 111, 113).
    layer 2 is MEASURED AT ZERO on the one hypothesis it has been asked, twice,
            pre-registered (d = 0.002, d = -0.005). It is reported as an
            observation of who traded, never as a forecast.
    layer 1 is UNTESTED as a predictor here. It is context a person should know,
            with a date and a source attached, not a scored input.

"ABOUT TO HAPPEN" — WHAT THAT CAN HONESTLY MEAN
------------------------------------------------
A rally that has not started yet cannot be detected; everything measured in this
repo says so. What CAN be stated exactly is the price at which the answer
changes, and how far away it is:

    ARMED_BUY    in a down leg, and within `--near` of the price that confirms
                 an up leg. Not a prediction - an arithmetic distance.
    CONFIRMED    the flip has happened on the last completed bar.

ARMED is the earliest honest warning available. It says "a 2.1% move from here
flips this", which is a fact, rather than "this is about to rally", which is not.

EVERY SIGNAL CARRIES ITS OWN HIT RATE
--------------------------------------
For each name and timeframe the feed reports, from that name's own causal
history, how often a confirmed buy was followed by a profitable round trip and
what the median outcome was. A signal without its base rate is advertising.

    python3 scripts/signals.py --universe big
    python3 scripts/signals.py --universe all --near 0.03
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

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV       # noqa: E402
from capture_toll import realised_tolls        # noqa: E402
from leg_signals import market_caps            # noqa: E402
from paint_daily import unadjusted_daily       # noqa: E402
from paint_live import band_state              # noqa: E402

TIMEFRAMES = {"weekly": 0.12, "daily": 0.08}
ROUND_TRIP_FEE = 0.0056
IMPORTED = os.path.join("data", "cache", "imported")

BUY, SELL = "CONFIRMED_BUY", "CONFIRMED_SELL"
ARM_B, ARM_S = "ARMED_BUY", "ARMED_SELL"
HOLD_L, HOLD_C = "HOLD_LONG", "HOLD_CASH"


# --------------------------------------------------------------------------- #
# layer 3 - the only exact one
# --------------------------------------------------------------------------- #
def classify(px: np.ndarray, band: float, near: float) -> Dict[str, object]:
    """State, trigger, and whether this bar is a flip or merely close to one."""
    st, trig = band_state(px, band)
    last, prev = int(st[-1]), int(st[-2]) if len(st) > 1 else int(st[-1])
    price, level = float(px[-1]), float(trig[-1])
    gap = level / price - 1.0
    if last and not prev:
        sig = BUY
    elif prev and not last:
        sig = SELL
    elif last:
        sig = ARM_S if abs(gap) <= near else HOLD_L
    else:
        sig = ARM_B if abs(gap) <= near else HOLD_C
    # np.diff marks the bar BEFORE the change, so the leg starts at flips[-1]+1
    # and bars_in_leg counts the bars IN the current leg, the flip bar included.
    flips = np.flatnonzero(np.diff(st.astype(int)) != 0)
    return {"signal": sig, "state": "GREEN" if last else "RED",
            "trigger": level, "gap": gap,
            "bars_in_leg": int(len(st) - 1 - flips[-1]) if len(flips) else len(st)}


def hit_rate(px: np.ndarray, band: float) -> Dict[str, float]:
    """How this exact rule has actually done on this exact name, causally.

    Round trips only - a signal's base rate is the outcome of acting on it, not
    the accuracy of the colour it painted.
    """
    rows = realised_tolls(px, band)
    if len(rows) < 5:
        return {"trips": len(rows), "win_rate": np.nan, "median_pl": np.nan,
                "mean_pl": np.nan}
    pl = np.array([r[2] for r in rows]) - ROUND_TRIP_FEE
    return {"trips": len(rows), "win_rate": float((pl > 0).mean()),
            "median_pl": float(np.median(pl)), "mean_pl": float(np.mean(pl))}


# --------------------------------------------------------------------------- #
# layer 1 - context, with provenance, never scored
# --------------------------------------------------------------------------- #
def regime(s: pd.Series) -> Dict[str, float]:
    """Price-derived regime context. Result 113 measured these as NOT predictive."""
    out: Dict[str, float] = {}
    if len(s) > 200:
        out["vs_200d"] = float(s.iloc[-1] / s.rolling(200).mean().iloc[-1] - 1.0)
    if len(s) > 21:
        out["ret_1m"] = float(s.iloc[-1] / s.iloc[-21] - 1.0)
    if len(s) > 63:
        out["ret_3m"] = float(s.iloc[-1] / s.iloc[-63] - 1.0)
    yr = s[s.index.year == s.index[-1].year]
    if len(yr) > 1 and float(yr.iloc[0]) > 0:
        out["ret_ytd"] = float(s.iloc[-1] / yr.iloc[0] - 1.0)
    if len(s) > 60:
        out["vol_ann"] = float(s.pct_change().tail(60).std() * np.sqrt(252))
    return out


def load_notes() -> Dict[str, List[Dict]]:
    """Dated, sourced layer-1 facts, if any have been recorded."""
    path = os.path.join("data", "notes", "layer1.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            raw = json.load(fh)
    except Exception:
        return {}
    out: Dict[str, List[Dict]] = {}
    for item in raw if isinstance(raw, list) else raw.get("findings", []):
        out.setdefault(str(item.get("ticker", "")).upper(), []).append(item)
    return out


# --------------------------------------------------------------------------- #
# layer 2 - observation only
# --------------------------------------------------------------------------- #
def load_flow(ticker: str) -> Optional[pd.DataFrame]:
    """Imported broker rows for one name, newest first, if any exist."""
    files = sorted(glob.glob(os.path.join(IMPORTED, f"{ticker}_*.csv.gz")))
    if not files:
        return None
    try:
        d = pd.concat([pd.read_csv(f) for f in files[-10:]], ignore_index=True)
    except Exception:
        return None
    return d if len(d) else None


def flow_summary(d: Optional[pd.DataFrame]) -> Optional[Dict[str, object]]:
    if d is None or d.empty:
        return None
    d = d.copy()
    d["net"] = d["buy_lot"].fillna(0) - d["sell_lot"].fillna(0)
    last = d[d["date"] == d["date"].max()]
    top = last.nlargest(3, "net")[["broker", "net"]]
    bal = abs(last["buy_lot"].sum() - last["sell_lot"].sum()) / max(
        last["buy_lot"].sum(), last["sell_lot"].sum(), 1.0)
    return {"date": str(last["date"].max()), "brokers": int(last["broker"].nunique()),
            "complete": bool(bal < 1e-6),
            "top_buyers": ", ".join(f"{r.broker}+{r.net:,.0f}"
                                    for r in top.itertuples() if r.net > 0)}


# --------------------------------------------------------------------------- #
def build(loader: YahooOHLCV, tickers: List[str], near: float) -> pd.DataFrame:
    notes = load_notes()
    rows = []
    for t in tickers:
        s = unadjusted_daily(loader, t, start="2015-01-01")
        if s is None or len(s) < 260:
            continue
        row: Dict[str, object] = {"ticker": t, "close": float(s.iloc[-1]),
                                  "asof": s.index[-1].date().isoformat()}
        for name, band in TIMEFRAMES.items():
            px = (s.resample("W-FRI").last().dropna() if name == "weekly"
                  else s).to_numpy(float)
            if len(px) < 60:
                continue
            c = classify(px, band, near)
            h = hit_rate(px, band)
            row[f"{name}_signal"] = c["signal"]
            row[f"{name}_state"] = c["state"]
            row[f"{name}_trigger"] = c["trigger"]
            row[f"{name}_gap"] = c["gap"]
            row[f"{name}_bars"] = c["bars_in_leg"]
            row[f"{name}_trips"] = h["trips"]
            row[f"{name}_win"] = h["win_rate"]
            row[f"{name}_med_pl"] = h["median_pl"]
        row.update(regime(s))
        fl = flow_summary(load_flow(t))
        row["flow"] = fl["top_buyers"] if fl else ""
        row["flow_complete"] = bool(fl["complete"]) if fl else False
        row["notes"] = len(notes.get(t, []))
        rows.append(row)
    return pd.DataFrame(rows)


def actionable(R: pd.DataFrame) -> pd.DataFrame:
    """Anything that flipped or is close to flipping, on either timeframe."""
    live = {BUY, SELL, ARM_B, ARM_S}
    m = (R.get("daily_signal", pd.Series(dtype=str)).isin(live)
         | R.get("weekly_signal", pd.Series(dtype=str)).isin(live))
    return R[m].copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="big", choices=["big", "mid", "all"])
    ap.add_argument("--min-mcap", type=float, default=1e13)
    ap.add_argument("--near", type=float, default=0.03,
                    help="how close to the trigger counts as ARMED")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    mc = market_caps()
    floor = {"big": args.min_mcap, "mid": 1e12, "all": 0.0}[args.universe]
    names = sorted(mc[mc >= floor].index)
    if args.limit:
        names = names[:args.limit]

    R = build(loader, names, args.near)
    if R.empty:
        raise SystemExit("no signals produced")
    R.to_csv("reports/signals_latest.csv", index=False)

    asof = pd.to_datetime(R["asof"]).max()
    print(f"{'=' * 104}\n IDX SIGNAL FEED — {len(R)} names, data to {asof:%Y-%m-%d}"
          f"\n{'=' * 104}")
    print(" layer 3 (bands) is exact but not predictive; layer 2 (flow) is an "
          "observation;\n layer 1 (news) is context. Each signal below carries "
          "the hit rate that rule has\n actually had on that name. Read the hit "
          "rate before the signal.\n")

    A = actionable(R)
    if A.empty:
        print(" nothing flipped and nothing is within "
              f"{args.near:.0%} of flipping.")
    else:
        print(f" {'ticker':<8}{'close':>9}{'weekly':>16}{'daily':>16}"
              f"{'flips at':>10}{'move':>8}{'win':>7}{'med P/L':>9}{'vs200d':>9}")
        A = A.assign(_k=A["daily_gap"].abs()).sort_values("_k")
        for _, r in A.iterrows():
            print(f" {r['ticker']:<8}{r['close']:>9,.0f}"
                  f"{str(r.get('weekly_signal', '-')):>16}"
                  f"{str(r.get('daily_signal', '-')):>16}"
                  f"{r.get('daily_trigger', np.nan):>10,.0f}"
                  f"{r.get('daily_gap', np.nan):>+8.1%}"
                  f"{r.get('daily_win', np.nan):>7.0%}"
                  f"{r.get('daily_med_pl', np.nan):>+9.1%}"
                  f"{r.get('vs_200d', np.nan):>+9.1%}")

    notes = load_notes()
    for label, sig in (("CONFIRMED BUY today", BUY), ("CONFIRMED SELL today", SELL)):
        hits = R[R.get("daily_signal") == sig]
        print(f"\n {label}: {len(hits)}"
              + (f" — {', '.join(hits['ticker'])}" if len(hits) else ""))

    # layer 1 and layer 2 for anything that actually moved - the whole reason
    # they are collected is to be read at the moment a signal fires
    moved = R[R.get("daily_signal").isin([BUY, SELL, ARM_B, ARM_S])]
    if len(moved):
        print(f"\n{'=' * 104}\n LAYER 1 AND 2 FOR WHAT MOVED\n{'=' * 104}")
        for _, r in moved.iterrows():
            t = r["ticker"]
            mine = notes.get(t, []) + notes.get("MACRO", []) if t in notes else notes.get(t, [])
            flow = r.get("flow", "")
            if not mine and not flow:
                continue
            print(f"\n {t}  [{r.get('daily_signal')}]")
            for n in mine[:3]:
                print(f"   L1 {n.get('date', '?')} ({n.get('confidence', '?')}) "
                      f"{str(n.get('claim', ''))[:150]}")
            if flow:
                print(f"   L2 top net buyers: {flow}"
                      + ("" if r.get("flow_complete") else
                         "   (top-N only — truncated, not a full rekap)"))
        missing = [t for t in moved["ticker"] if t not in notes]
        if missing:
            print(f"\n no layer-1 note on file for: {', '.join(missing)}")
        if not moved["flow"].astype(bool).any():
            print(" no layer-2 broker data imported for any of these — see "
                  "data/inbox/README.md")

    med_win = R["daily_win"].median()
    med_pl = R["daily_med_pl"].median()
    print(f"\n{'=' * 104}\n WHAT THESE SIGNALS ARE WORTH — measured, not claimed"
          f"\n{'=' * 104}")
    print(f" across {len(R)} names, the daily 8% rule's own history:")
    print(f"   round trips per name (median): {R['daily_trips'].median():.0f}")
    print(f"   win rate (median):             {med_win:.0%}")
    print(f"   median round trip:             {med_pl:+.1%} after fees")
    print(f" and it still loses to buy-and-hold (Result 100) and to random "
          f"timing at the\n same exposure (Results 111, 113). Use it to know "
          f"WHERE you are, not what is next.")
    print(f"\n -> reports/signals_latest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
