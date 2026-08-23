#!/usr/bin/env python3
"""§9.3 cohort P&L, on both stores, reported SEPARATELY and never blended.

WHY TWO TRACKS AND WHY THEY MUST NOT BE COMBINED
--------------------------------------------------
§9.3 specifies a DAILY walk-forward and makes round-trip episodes the primary
estimate. This repo has two broker stores and **neither one satisfies that
specification**, for opposite reasons:

    TRACK A — daily, 9 names, ~360 sessions each.
        Right resolution, wrong length. §9.3 says discard the first 250 trading
        days as burn-in; on a 361-session series that leaves 111 usable days.
        The burn-in prescription and the sample are in direct conflict.

    TRACK B — fortnightly, 176 names (64 delisted), 329 windows.
        Right length, wrong resolution. A fortnight gives NET flow over ten
        sessions. Everything that happened inside the window — the path — is
        gone, and the path is exactly what a round trip is made of. Buying
        50,000 lots on Monday and selling them Friday reads as zero.

So Track A can compute round trips but has little sample; Track B has sample
but cannot compute a round trip at all. Averaging a 9-name exact number with a
176-name approximate one would produce a figure that is neither, and would read
as more authoritative than either. They stay in separate tables.

WHAT TRACK B CAN HONESTLY SAY
------------------------------
Net inventory drift and its sign, an execution-price comparison against the
window VWAP, and the crossing ratio — all of which survive aggregation because
they are computed from window totals rather than from a path. What it cannot
say is anything containing the words "round trip", and it does not try.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.spine.cohort_pnl import (BURN_IN_DAYS, LOT,        # noqa: E402
                                     bootstrap_ci, crossing_ratio,
                                     margin_bps,
                                     negative_inventory_share, round_trips,
                                     shuffle_broker_labels, walk_forward)
from idxbot.data.ohlcv import YahooOHLCV                       # noqa: E402
from idxbot.config import Config                               # noqa: E402

DAILY = os.path.join("data", "cache", "broker_daily")
PANEL = os.path.join("data", "spine", "flow_panel.csv.gz")

#: Fewest sessions a broker-ticker series needs before it is worth walking.
MIN_SESSIONS = 60


def load_daily() -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(os.path.join(DAILY, "*_ipot-all.csv.gz"))):
        try:
            d = pd.read_csv(p, usecols=["date", "ticker", "broker", "buy_lot",
                                        "buy_avg", "sell_lot", "sell_avg"])
        except Exception:                                       # noqa: BLE001
            continue
        if not d.empty:
            rows.append(d)
    if not rows:
        return pd.DataFrame()
    D = pd.concat(rows, ignore_index=True)
    D["date"] = pd.to_datetime(D["date"])
    return D.sort_values(["ticker", "broker", "date"]).reset_index(drop=True)


def closes(tickers) -> Dict[str, pd.Series]:
    o = YahooOHLCV(Config())
    out = {}
    for t in tickers:
        try:
            d = o.get(t, max_age=10 ** 9)
        except Exception:                                       # noqa: BLE001
            continue
        if d is not None and not d.empty:
            s = d.set_index(pd.to_datetime(d["date"]))["close"]
            out[t] = pd.to_numeric(s, errors="coerce")
    return out


def track_a(D: pd.DataFrame, burn: int, label: str = "") -> pd.DataFrame:
    """Full §9.3 walk per (broker, ticker), with round-trip episodes."""
    px = closes(sorted(D["ticker"].unique()))
    rows: List[Dict] = []
    for (t, b), g in D.groupby(["ticker", "broker"]):
        if len(g) < MIN_SESSIONS:
            continue
        g = g.sort_values("date")
        c = px.get(t)
        cs = (c.reindex(pd.to_datetime(g["date"])).ffill().to_numpy()
              if c is not None else None)
        w = walk_forward(g, pd.Series(cs) if cs is not None else None)
        if burn:
            w = w.iloc[burn:].reset_index(drop=True)
            if w.empty:
                continue
        rt = round_trips(w)
        gross = float(w["gross_value"].sum())
        # The unrealised leg is only computable when the series ends LONG. On a
        # negative final inventory the cohort is holding a position this data
        # never saw it acquire, priced against a WAC that means nothing, and
        # multiplying the two produced full-path margins of -13,000 bps - a
        # 130% loss on gross traded value, which is not a thing that can happen.
        # So it is NaN there rather than a number, and the share is reported.
        fin_inv = float(w["inventory"].iloc[-1])
        unreal = float(w["unrealized"].iloc[-1])
        computable = fin_inv >= 0 and np.isfinite(unreal)
        fp = (float(w["realized"].sum()) + unreal) if computable else np.nan
        rows.append({
            "full_path_computable": bool(computable),
            "ticker": t, "broker": b, "days": len(w),
            "gross_value": gross,
            "full_path_pnl": fp,
            "full_path_bps": margin_bps(fp, gross) if computable else np.nan,
            "n_round_trips": len(rt),
            "round_trip_pnl": float(rt["pnl"].sum()) if len(rt) else np.nan,
            "round_trip_bps": (margin_bps(float(rt["pnl"].sum()),
                                          float(rt["gross_value"].sum()))
                               if len(rt) else np.nan),
            "neg_inventory": negative_inventory_share(w),
            "crossing": crossing_ratio(g),
        })
    R = pd.DataFrame(rows)
    if not R.empty:
        R["label"] = label
    return R


def report_a(R: pd.DataFrame, name: str) -> None:
    if R.empty:
        print(f"  {name}: nothing with >= {MIN_SESSIONS} sessions")
        return
    rt = R[R["n_round_trips"] > 0]
    print(f"  {name:<26} broker-tickers {len(R):>4}   "
          f"with round trips {len(rt):>4}   episodes {int(R.n_round_trips.sum()):>5}")
    for col, lab in (("round_trip_bps", "round-trip margin_bps (PRIMARY)"),
                     ("full_path_bps", "full-path margin_bps (noisy)")):
        v = R[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(v) < 3:
            print(f"    {lab:<34} too few to summarise")
            continue
        lo, hi = bootstrap_ci(v)
        print(f"    {lab:<34} median {v.median():>9.1f}  mean {v.mean():>9.1f}"
              f"  95% CI [{lo:>8.1f}, {hi:>8.1f}]  n={len(v)}")
    print(f"    full-path computable on             "
          f"{R.full_path_computable.mean():.0%} of series "
          f"(the rest end SHORT of shares they were never seen to buy)")
    print(f"    negative-inventory share           median "
          f"{R.neg_inventory.median():.1%}  "
          f"(§9.2 starting-inventory problem, direct measure)")
    print(f"    crossing ratio                     median {R.crossing.median():.2f}")


def track_b(P: pd.DataFrame) -> None:
    """What the fortnightly panel can say WITHOUT claiming a round trip."""
    print(f"  names {P.ticker.nunique()}  "
          f"({P[P.src=='delisted'].ticker.nunique()} delisted)  "
          f"windows {P.window_end.nunique()}")
    print(f"  NO round-trip estimate is computed here, and none can be: a")
    print(f"  fortnight gives net flow over ten sessions, and the path inside")
    print(f"  the window — which is what an episode is made of — is gone.")
    print()
    imb = P["imbalance"].dropna()
    print(f"    net imbalance      median {imb.median():+.4f}  "
          f"mean {imb.mean():+.4f}  sd {imb.std():.4f}")
    fn = P["foreign_net"].dropna()
    print(f"    foreign net share  median {fn.median():+.4f}  "
          f"mean {fn.mean():+.4f}")
    dn = P["domestic_net"].dropna()
    print(f"    domestic net share median {dn.median():+.4f}  "
          f"mean {dn.mean():+.4f}")
    print(f"    two-sided codes    median {P.two_sided_codes.median():.0f} of "
          f"{P.n_brokers.median():.0f} — only these support a per-broker NET")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", action="store_true",
                    help="also run Track A on shuffled broker labels")
    a = ap.parse_args()

    print("=" * 76)
    print(" TRACK A — DAILY STORE: full §9.3 walk-forward, round trips possible")
    print("=" * 76)
    D = load_daily()
    if D.empty:
        print("  no daily store")
    else:
        n = D.groupby("ticker")["date"].nunique()
        print(f"  {D.ticker.nunique()} tickers, {D.broker.nunique()} codes, "
              f"{len(D):,} broker-days, {n.max()} sessions max")
        print(f"  §9.3 asks for {BURN_IN_DAYS}-day burn-in. On {n.max()} "
              f"sessions that leaves {n.max() - BURN_IN_DAYS} usable days.")
        print(f"  Both are reported because that trade-off must be visible.\n")
        report_a(track_a(D, 0, "no burn-in"), "WITHOUT burn-in")
        print()
        report_a(track_a(D, BURN_IN_DAYS, "burn-in"), f"WITH {BURN_IN_DAYS}-day burn-in")
        if a.null:
            print()
            report_a(track_a(shuffle_broker_labels(D), 0, "null"),
                     "NULL (labels shuffled)")

    print()
    print("=" * 76)
    print(" TRACK B — FORTNIGHTLY PANEL: net flow only, NO round trips")
    print("=" * 76)
    if os.path.exists(PANEL):
        track_b(pd.read_csv(PANEL, parse_dates=["window_end"]))
    else:
        print("  no panel")

    print()
    print("These two tables are NOT combined and must not be averaged. Track A")
    print("is exact on 9 names; Track B is approximate on 176. A blend would be")
    print("neither, and would read as more authoritative than either.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
