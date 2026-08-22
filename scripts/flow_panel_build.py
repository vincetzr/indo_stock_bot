#!/usr/bin/env python3
"""Fold the collected range files into the panel §7 consumes.

One row per (ticker, fortnight): the flow measured over that fortnight, the
controls known at its close, and the forward return that follows it. Nothing in
a row may be stamped later than the fortnight's last bar except the label.

THE LOOK-AHEAD RULES, WHICH ARE THE WHOLE POINT OF THIS FILE
-------------------------------------------------------------
IndoPremier's rekap for a window ending on day T is only complete after T's
close, so T is the DECISION bar and the first bar that can be traded is T+1.
The label therefore runs from T+1's close, not T's:

    fwd_k = adj_close[T+1+k] / adj_close[T+1] - 1

Entering at T's close would be free money that did not exist - it uses the
window's own last print to buy before the window's own summary is published.
That single choice is worth more than any feature in the file.

Controls are computed on bars up to and including T and never past it.

WHAT THE FLOW FEATURES CAN AND CANNOT BE
-----------------------------------------
The source gives the TOP TEN buyers and TOP TEN sellers, ranked
independently, aggregated over the window. So:

    RATIOS ARE SAFE, LEVELS ARE NOT. §6 warns the raw net-buy column is public
    and simultaneous; it is also truncated here, and truncation biases a level
    much harder than a ratio. Everything below is a ratio or a share.

    THE TWO SIDES ARE DIFFERENT SETS. Row 3's buyer and row 3's seller are
    unrelated, so a per-broker NET is only available for codes appearing on
    both sides. ``net_by_broker`` handles the union and treats an absent side
    as an unknown lower bound rather than as zero - a broker ranked eleventh
    bought something, not nothing.

    COVERAGE IS MEASURABLE AND MUST TRAVEL. Total traded lots come from the
    spine, so ``coverage`` is the share of the window's real volume the top ten
    actually account for. On a liquid name that can be 40%; on a quiet one it
    is often 100%. An imbalance computed at 40% coverage is a different
    quantity from one at 100%, and any result that does not condition on it is
    mixing them.

Figures at or above one million are abbreviated by the source to 2-3
significant figures. That is why value is never used where lots will do, and
why nothing here is a rupiah-exact quantity.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.spine.repairs import apply_repairs           # noqa: E402
from idxbot.spine.quality import stale_bars              # noqa: E402
from flow_panel_collect import (LIVE, DEAD, load_panel,  # noqa: E402
                                windows)

STORE = os.path.join("data", "cache", "ipot_broker")
OUT = os.path.join("data", "spine", "flow_panel.csv.gz")


def foreign_codes() -> set:
    """Broker codes classified foreign, HIGH CONFIDENCE ONLY.

    The low-confidence entries are left out rather than guessed. §9.2 is blunt
    that nominee codes are omnibus and that a wrong classification does not
    average out - it puts one cohort's flow in the other cohort's column.
    """
    d = yaml.safe_load(open(os.path.join("config", "brokers.yaml")))["brokers"]
    return {k for k, v in d.items()
            if v.get("foreign") and v.get("confidence") == "high"}


def load_prices(ticker: str, src: str) -> Optional[pd.DataFrame]:
    fp = os.path.join(DEAD if src == "delisted" else LIVE, f"{ticker}.JK.csv.gz")
    if not os.path.exists(fp):
        return None
    try:
        d = pd.read_csv(fp)
    except Exception:                                           # noqa: BLE001
        return None
    if d.empty or "close" not in d:
        return None
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["close"] > 0].sort_values("date").reset_index(drop=True)
    d = apply_repairs(d, ticker)
    d["stale"] = stale_bars(d).to_numpy()
    if "adj_close" not in d:
        d["adj_close"] = d["close"]
    return d.set_index("date")


def flow_features(g: pd.DataFrame, foreign: set) -> Dict[str, float]:
    """Ratios only. See the module docstring for why no level appears here."""
    b = pd.to_numeric(g["buy_lot"], errors="coerce").fillna(0.0)
    s = pd.to_numeric(g["sell_lot"], errors="coerce").fillna(0.0)
    tot = float(b.sum() + s.sum())
    if tot <= 0:
        return {}
    out = {
        "imbalance": float((b.sum() - s.sum()) / tot),
        "n_brokers": float(len(g)),
        "top1_buy_share": float(b.max() / b.sum()) if b.sum() > 0 else np.nan,
        "hhi_buy": float(((b / b.sum()) ** 2).sum()) if b.sum() > 0 else np.nan,
        "hhi_sell": float(((s / s.sum()) ** 2).sum()) if s.sum() > 0 else np.nan,
        "buy_lot_top10": float(b.sum()),
        "sell_lot_top10": float(s.sum()),
    }
    isf = g["broker"].astype(str).str.upper().isin(foreign)
    out["foreign_net"] = float((b[isf].sum() - s[isf].sum()) / tot)
    out["foreign_share"] = float((b[isf].sum() + s[isf].sum()) / tot)
    out["domestic_net"] = float((b[~isf].sum() - s[~isf].sum()) / tot)
    # The two sides are independent rankings, so a broker on one side only has
    # an UNKNOWN other side. Counting how many codes appear on both is the
    # honest measure of how much per-broker netting the row can support.
    both = set(g.loc[b > 0, "broker"]) & set(g.loc[s > 0, "broker"])
    out["two_sided_codes"] = float(len(both))
    return out


def build(step: int = 10, start: str = "2014-01-01",
          end: Optional[str] = None) -> pd.DataFrame:
    P = load_panel()
    end = end or (pd.Timestamp.today().normalize()
                  - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    wins = windows(start, end, step)
    foreign = foreign_codes()
    rows: List[Dict] = []

    for _, meta in P.iterrows():
        t, src = meta["ticker"], meta["src"]
        px = load_prices(t, src)
        if px is None:
            continue
        idx = px.index
        adj = px["adj_close"].to_numpy(dtype=float)
        vol = pd.to_numeric(px["volume"], errors="coerce").fillna(0.0).to_numpy()
        raw = pd.to_numeric(px["close"], errors="coerce").to_numpy(dtype=float)
        turnover = raw * vol

        for (a, b) in wins:
            key = f"{t}_{a:%Y%m%d}_{b:%Y%m%d}_RG_range"
            fp = os.path.join(STORE, key + ".csv.gz")
            if not os.path.exists(fp):
                continue
            try:
                g = pd.read_csv(fp)
            except Exception:                                   # noqa: BLE001
                continue
            if g.empty:
                continue
            f = flow_features(g, foreign)
            if not f:
                continue

            # T = the last SPINE bar inside the window. Everything known.
            inwin = (idx >= a) & (idx <= b)
            if not inwin.any():
                continue
            iT = int(np.flatnonzero(inwin)[-1])
            if iT + 2 >= len(idx):
                continue                       # no tradeable bar after T yet

            traded = float(vol[inwin].sum())
            f["coverage"] = ((f["buy_lot_top10"] * 100.0 / traded)
                             if traded > 0 else np.nan)

            # controls, all stamped at or before T
            def past(n):
                lo = max(0, iT - n)
                return adj[iT] / adj[lo] - 1.0 if adj[lo] > 0 else np.nan

            f.update({
                "ticker": t, "src": src, "decile": int(meta["decile"]),
                "window_start": a, "window_end": b, "T": idx[iT],
                "mom12_1": (adj[iT - 21] / adj[iT - 250] - 1.0
                            if iT >= 250 and adj[iT - 250] > 0 else np.nan),
                "rev1": past(21),
                "log_turnover": (np.log(np.nanmedian(turnover[max(0, iT-60):iT+1]))
                                 if iT > 0 else np.nan),
                "vol60": float(np.nanstd(np.diff(np.log(
                    np.maximum(adj[max(0, iT-60):iT+1], 1e-9))))),
                "stale_share": float(px["stale"].to_numpy()[inwin].mean()),
            })

            # LABEL: entry at T+1's close, because the rekap for a window
            # ending at T is not public until after T closes.
            e0 = iT + 1
            f["entry_date"] = idx[e0]
            for k, lab in ((step, "fwd_1w"), (2 * step, "fwd_2w")):
                e1 = e0 + k
                f[lab] = (adj[e1] / adj[e0] - 1.0
                          if e1 < len(adj) and adj[e0] > 0 else np.nan)
            f["entry_tradeable"] = bool(vol[e0] > 0)
            rows.append(f)

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    D = build(a.step, a.start)
    if D.empty:
        print("nothing built — no collected windows overlap the spine yet")
        return 1
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    D.to_csv(a.out, index=False, compression="gzip")

    print(f"rows        {len(D):,}")
    print(f"names       {D['ticker'].nunique()} "
          f"({D[D.src=='delisted']['ticker'].nunique()} delisted)")
    print(f"windows     {D['window_end'].nunique()}  "
          f"{D['window_end'].min():%Y-%m-%d} .. {D['window_end'].max():%Y-%m-%d}")
    print(f"names/window median {D.groupby('window_end')['ticker'].nunique().median():.0f}")
    print(f"coverage    median {D['coverage'].median():.1%}, "
          f"p10 {D['coverage'].quantile(0.1):.1%}, "
          f"p90 {D['coverage'].quantile(0.9):.1%}")
    print(f"labelled    fwd_1w {D['fwd_1w'].notna().sum():,}, "
          f"fwd_2w {D['fwd_2w'].notna().sum():,}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
