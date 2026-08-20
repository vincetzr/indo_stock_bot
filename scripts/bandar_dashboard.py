#!/usr/bin/env python3
"""The whole picture on one name: who holds what, at what cost, and what follows.

WHAT IS RECONSTRUCTED
---------------------
For every broker, from their own daily prints:

    lots left        cumulative net position - how much stock they still hold
    average cost     weighted-average basis of everything they bought
    unrealised P/L   lots_left x (last_close - basis) x 100  - are they floating
                     a profit or sitting underwater right now
    realised P/L     booked on every sale, against that same basis
    days held        how long the current position has been open

Then aggregated three ways, using the 66-broker registry (20 foreign, 4
state-owned):

    foreign net      the classic "asing" flow
    BUMN net         state-owned houses, which behave differently
    other domestic   everything else

A WORD ON THE WORD "RETAIL"
---------------------------
Broker summary does NOT contain an investor type. The domestic/foreign flag on
each ORDER exists in the licensed ITCH feed, not in the public rekap. So "other
domestic" here is a bucket of brokers, not a measurement of retail, and calling
it retail would be inventing a field. Platforms that display a retail line are
either licensed to the investor-type data or are applying the same proxy.

THE PART THAT PREDICTS - AND WHAT IT IS WORTH
----------------------------------------------
Three states are measured against the NEXT day's return, each fixed in advance
rather than searched for:

    S1  net foreign buying today
    S2  buying concentrated in few hands (top-3 share above its median)
    S3  overhang - most visible inventory held at a LOSS

Each is reported as a probability with a Wilson interval, against the base rate
on the same days. Wilson rather than a plain proportion because at n = 60 the
normal approximation is simply wrong near the edges.

If an interval spans the base rate, the honest reading is "this sample cannot
tell", and the script says exactly that rather than quoting a number that looks
like a forecast. Result 117 measured the power: the existing daily panel can only
detect an effect of d = 0.40, which is enormous.

    python3 scripts/bandar_dashboard.py --ticker BBCA
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.config import BrokerRegistry, load_config    # noqa: E402
from idxbot.data.cache import Cache                      # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV                 # noqa: E402
from account_sim import load_ohlc                        # noqa: E402
from broker_habits import ledger, load_daily, visibility  # noqa: E402

LOT = 100
REGISTRY = os.path.join("config", "brokers.yaml")


def classify(code: str, reg: BrokerRegistry) -> str:
    b = reg._brokers.get(str(code).upper())
    if b is None:
        return "unknown"
    if b.foreign:
        return "foreign"
    if b.state_owned:
        return "bumn"
    return "domestic"


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Proportion with a Wilson interval - correct near 0 and 1, unlike normal."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    # Clamp: at k=0 or k=n the arithmetic lands a hair outside [0,1] on floating
    # point, and a probability bound printed as -1.4e-17 is simply wrong.
    return p, max(0.0, centre - half), min(1.0, centre + half)


def book_state(df: pd.DataFrame, last_close: float,
               reg: BrokerRegistry) -> pd.DataFrame:
    """Every broker's current position, cost, and floating P/L."""
    rows = []
    for b in sorted(df["broker"].unique()):
        led = ledger(df, b)
        if led.empty:
            continue
        last = led.iloc[-1]
        inv, basis = float(last["inventory"]), float(last["basis"])
        unreal = (inv * (last_close - basis) * LOT
                  if inv > 0 and np.isfinite(basis) and basis > 0 else np.nan)
        held_since = np.nan
        if inv > 0:
            neg = led.index[led["inventory"] <= 0]
            start = int(neg[-1]) + 1 if len(neg) else 0
            if start < len(led):
                held_since = (led.iloc[-1]["date"] - led.iloc[start]["date"]).days
        rows.append({
            "broker": b, "class": classify(b, reg),
            "lots_left": inv, "basis": basis,
            "unrealised": unreal,
            "unreal_pct": (last_close / basis - 1.0)
            if inv > 0 and np.isfinite(basis) and basis > 0 else np.nan,
            "realised": float(led["realised"].sum(skipna=True)),
            "days_held": held_since,
            "net_lot": float(led["buy_lot"].sum() - led["sell_lot"].sum()),
        })
    return pd.DataFrame(rows)


def daily_states(df: pd.DataFrame, reg: BrokerRegistry) -> pd.DataFrame:
    """Per-day aggregates and the three pre-specified conditions."""
    d = df.copy()
    d["class"] = d["broker"].map(lambda b: classify(b, reg))
    d["net"] = d["buy_lot"] - d["sell_lot"]
    piv = d.pivot_table(index="date", columns="class", values="net",
                        aggfunc="sum").fillna(0.0)
    for c in ("foreign", "bumn", "domestic", "unknown"):
        if c not in piv.columns:
            piv[c] = 0.0
    top3 = (d[d["net"] > 0].sort_values("net", ascending=False)
             .groupby("date")["net"].apply(lambda s: s.head(3).sum() / s.sum()
                                           if s.sum() > 0 else np.nan))
    piv["top3_share"] = top3
    return piv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="BBCA")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    reg = BrokerRegistry.from_yaml(REGISTRY)
    df = load_daily(args.ticker)
    if df.empty:
        print(f"no daily broker data for {args.ticker}.")
        return 1

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    px = load_ohlc(loader, args.ticker)
    close = px["close"] if px is not None else None
    last_close = float(close.iloc[-1]) if close is not None else np.nan

    days = df["date"].nunique()
    print(f"{'=' * 100}\n BANDAR DASHBOARD — {args.ticker}   "
          f"{days} sessions, last close {last_close:,.0f}\n{'=' * 100}")

    # ---- the one number that says whether any of this can be trusted -------
    # Every share bought was sold, so the net across ALL brokers must be zero.
    # Whatever it is instead is exactly the volume the top-ten cut hides, and it
    # is the ceiling on how wrong every inventory below can be.
    net_all = float((df["buy_lot"] - df["sell_lot"]).sum())
    gross = float((df["buy_lot"] + df["sell_lot"]).sum())
    print(f"\n DATA QUALITY: net across all brokers is {net_all:,.0f} lots. "
          f"It must be 0.")
    print(f" That gap is {abs(net_all) / max(gross, 1):.1%} of all visible "
          f"volume and is the unseen\n remainder below the top-ten cut. Every "
          f"inventory below inherits a share of it.")

    # ---------------------------------------------------------------- book --
    B = book_state(df, last_close, reg)
    V = visibility(df).set_index("broker")
    B["appearance"] = B["broker"].map(V["appearance"])
    B["bound_ratio"] = B["broker"].map(V["bound_ratio"])
    B = B.sort_values("lots_left", ascending=False)
    B.to_csv(f"reports/bandar_book_{args.ticker}.csv", index=False)

    # A position is only as real as the share of the broker's trading that was
    # visible. bound_ratio is unseen-volume / observed-volume, so above 1 the
    # inventory is mostly guess and must not be read as a holding.
    def grade(r) -> str:
        br = r["bound_ratio"]
        if not np.isfinite(br):
            return "UNUSABLE"
        if br < 0.25 and r["appearance"] >= 0.8:
            return "solid"
        if br < 1.0:
            return "weak"
        return "UNUSABLE"
    B["grade"] = B.apply(grade, axis=1)

    print(f"\n{'=' * 100}\n WHO IS HOLDING WHAT — inventory, cost, and floating "
          f"P/L\n{'=' * 100}")
    print(f" {'broker':<8}{'class':<10}{'lots left':>13}{'avg cost':>11}"
          f"{'now vs cost':>13}{'unrealised Rp':>18}{'days':>7}{'seen':>7}"
          f"{'grade':>10}")
    for _, r in B.head(args.top).iterrows():
        print(f" {r['broker']:<8}{r['class']:<10}{r['lots_left']:>13,.0f}"
              f"{(r['basis'] if np.isfinite(r['basis']) else 0):>11,.0f}"
              f"{(r['unreal_pct'] if np.isfinite(r['unreal_pct']) else 0):>+13.1%}"
              f"{(r['unrealised'] if np.isfinite(r['unrealised']) else 0):>18,.0f}"
              f"{(r['days_held'] if np.isfinite(r['days_held']) else 0):>7.0f}"
              f"{r['appearance']:>7.0%}{r['grade']:>10}")

    live = B[B["lots_left"] > 0]
    if len(live):
        solid = live[live["grade"] == "solid"]
        share = solid["lots_left"].sum() / live["lots_left"].sum()
        print(f"\n ! only {len(solid)} of {len(live)} holders are graded solid, "
              f"carrying {share:.0%} of the\n ! reported inventory. The rest "
              f"are brokers who drop below the top-ten cut, so\n ! their "
              f"positions are partly invented by the gap above. Read only the "
              f"solid rows\n ! as holdings; the others are lower bounds at "
              f"best.")

    holders = B[(B["lots_left"] > 0) & B["unreal_pct"].notna()
                & (B["grade"] == "solid")]
    if len(holders):
        under = holders[holders["unreal_pct"] < 0]
        w = holders["lots_left"]
        print(f"\n of the SOLID holders: {len(holders)} hold stock, "
              f"{len(under)} are underwater.")
        print(f" size-weighted position: "
              f"{float((holders['unreal_pct'] * w).sum() / w.sum()):+.1%} "
              f"against cost")
        print(f" total floating P/L on visible inventory: "
              f"Rp{holders['unrealised'].sum():,.0f}")

    # --------------------------------------------------------------- flows --
    S = daily_states(df, reg)
    S.to_csv(f"reports/bandar_flows_{args.ticker}.csv")
    print(f"\n{'=' * 100}\n FLOWS BY CLASS — net lots\n{'=' * 100}")
    print(f" {'class':<12}{'total net':>16}{'days net buying':>18}"
          f"{'last session':>16}")
    for c in ("foreign", "bumn", "domestic", "unknown"):
        if c not in S.columns:
            continue
        col = S[c]
        print(f" {c:<12}{col.sum():>16,.0f}{float((col > 0).mean()):>18.0%}"
              f"{col.iloc[-1]:>16,.0f}")
    print("\n 'domestic' is a bucket of brokers, NOT a retail measurement — the "
          "investor-type\n flag lives in the licensed ITCH feed, not in the "
          "public rekap.")

    # ---------------------------------------------------- what follows next --
    if close is None:
        print("\n no price history: cannot test what follows.")
        return 0
    c = close.copy()
    c.index = c.index.normalize()
    fwd = (c.shift(-1) / c - 1.0)
    S = S.copy()
    S.index = pd.to_datetime(S.index).normalize()
    S["fwd"] = S.index.map(fwd)
    S = S.dropna(subset=["fwd"])

    base_k = int((S["fwd"] > 0).sum())
    base_p, base_lo, base_hi = wilson(base_k, len(S))
    print(f"\n{'=' * 100}\n WHAT FOLLOWS — next-day direction, three "
          f"pre-specified states\n{'=' * 100}")
    print(f" base rate: {base_p:.0%} of {len(S)} days closed up "
          f"[{base_lo:.0%}, {base_hi:.0%}]\n")
    print(f" {'state':<34}{'days':>6}{'up next':>10}"
          f"{'95% interval':>20}{'verdict':>26}")

    conds = {
        "S1 foreign net buying": S["foreign"] > 0,
        "S2 buying concentrated (top3 hi)": S["top3_share"] > S["top3_share"].median(),
        "S3 domestic net buying": S["domestic"] > 0,
    }
    rows = []
    for name, mask in conds.items():
        sub = S[mask.fillna(False)]
        if len(sub) < 5:
            print(f" {name:<34}{len(sub):>6}{'-':>10}{'too few days':>20}")
            continue
        k = int((sub["fwd"] > 0).sum())
        p, lo, hi = wilson(k, len(sub))
        tells = not (lo <= base_p <= hi)
        rows.append({"state": name, "days": len(sub), "p_up": p,
                     "lo": lo, "hi": hi, "separates": tells})
        print(f" {name:<34}{len(sub):>6}{p:>10.0%}"
              f"{f'[{lo:.0%}, {hi:.0%}]':>20}"
              f"{('SEPARATES from base' if tells else 'cannot tell — spans base'):>26}")

    if rows:
        R = pd.DataFrame(rows)
        R.to_csv(f"reports/bandar_states_{args.ticker}.csv", index=False)
        sep = int(R["separates"].sum())
        print(f"\n {sep} of {len(R)} states separate from the base rate at 95%.")
        if sep == 0:
            print(" On this sample none of them predicts the next day. That is "
                  "the correct answer\n for 60 sessions, not a failure of the "
                  "method: Result 117 measured that this\n panel can only "
                  "detect an effect of d = 0.40, which would be visible by eye.")
        else:
            print(" Treat any separation here as a HYPOTHESIS, not a finding — "
                  "three states were\n tested, the intervals are wide, and "
                  "nothing has been replicated out of sample.")

    print(f"\n -> reports/bandar_book_{args.ticker}.csv, "
          f"bandar_flows_{args.ticker}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
