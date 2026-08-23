#!/usr/bin/env python3
"""§12 at investor-class resolution — is foreign or domestic persistently right?

WHAT THIS ANSWERS THAT H11 COULD NOT
--------------------------------------
H11 found that a broker code's margin rank does not persist, and closed by
saying the finding was about the INSTRUMENT rather than the phenomenon: a
broker code aggregates thousands of accounts of mixed type (§6.1), while the
Taiwan and Finland results §12 cites identify account TYPES. IDX does publish
a type split — the per-trade foreign/domestic investor flag — and IndoPremier
serves it through the same endpoint with ``fd=F`` / ``fd=D``.

So this is §12's question asked with an instrument that matches it:

    which flow is persistently dumb, is it identifiable in real time, and is
    it large enough to trade against after costs

with "flow" now meaning an investor class rather than a member firm.

THREE TESTS, ALL OF WHICH MUST PASS
-------------------------------------
    1. LEVEL        margin distinguishable from zero AND from a null that
                    destroys the flow-to-return pairing within each window
    2. PERSISTENCE  the sign holds across years rather than being carried by
                    a couple of them
    3. SIZE         |margin| beats A5's 56 bps round trip

Failing any one of them means §12's strategy has no instrument here either,
and that is a result to report rather than a threshold to loosen (§2).

NO LOOK-AHEAD
--------------
Flow is measured over [window_start, window_end]; the payoff is the return
from window_end's close to the NEXT window_end's close. Windows more than one
fortnight apart are dropped rather than stretched, so a missing window never
turns "the next fortnight" into "the next quarter".
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

from idxbot.spine.investor_split import (ROUND_TRIP_BPS,          # noqa: E402
                                         class_margin, margin_bps,
                                         mirror_residual,
                                         permutation_margin,
                                         sign_persistence)
from flow_panel_build import load_prices                          # noqa: E402
from flow_panel_collect import load_panel                         # noqa: E402

STORE = os.path.join("data", "cache", "ipot_broker")
CACHE = os.path.join("data", "spine", "investor_split.csv.gz")
MAX_WINDOW_GAP_DAYS = 21

#: A ticker-window needs this much of the class's flow visible before its
#: margin is trusted. Rp 100m is small; the guard exists to drop windows where
#: the class barely participated and the ratio is one trade over another.
MIN_GROSS = 1e8


def load_views(store: str = STORE) -> pd.DataFrame:
    """Per (ticker, window, view) net and gross value from the range files."""
    rows = []
    for p in sorted(glob.glob(os.path.join(store, "*_RG_F_range.csv.gz"))
                    + glob.glob(os.path.join(store, "*_RG_D_range.csv.gz"))):
        m = re.match(r"([A-Z]{4})_(\d{8})_(\d{8})_RG_([FD])_range",
                     os.path.basename(p))
        if not m:
            continue
        try:
            d = pd.read_csv(p, usecols=["broker", "buy_val", "sell_val"])
        except Exception:                                       # noqa: BLE001
            continue
        if d.empty:
            continue
        b = pd.to_numeric(d["buy_val"], errors="coerce").fillna(0.0).sum()
        s = pd.to_numeric(d["sell_val"], errors="coerce").fillna(0.0).sum()
        rows.append({"ticker": m.group(1),
                     "window_end": pd.Timestamp(m.group(3)),
                     "view": m.group(4),
                     "net_value": float(b - s),
                     "gross_value": float(b + s),
                     "n_brokers": int(len(d))})
    return pd.DataFrame(rows).drop_duplicates(
        subset=["ticker", "window_end", "view"])


def forward_returns(tickers, wins_by_ticker: Dict[str, List[pd.Timestamp]],
                    src: Dict[str, str]) -> Dict[tuple, float]:
    """close(next window end) / close(this window end) - 1."""
    fwd: Dict[tuple, float] = {}
    for t in tickers:
        px = load_prices(t, src.get(t, "live"))
        if px is None or px.empty:
            continue
        c = pd.to_numeric(px["close"], errors="coerce")
        idx = px.index
        wins = sorted(wins_by_ticker.get(t, []))
        for a, b in zip(wins, wins[1:]):
            if (b - a).days > MAX_WINDOW_GAP_DAYS:
                continue
            ia, ib = idx[idx <= a], idx[idx <= b]
            if not len(ia) or not len(ib) or ia[-1] == ib[-1]:
                continue
            pa, pb = float(c.loc[ia[-1]]), float(c.loc[ib[-1]])
            if np.isfinite(pa) and np.isfinite(pb) and pa > 0:
                fwd[(t, a)] = pb / pa - 1.0
    return fwd


def build(rebuild: bool = False) -> pd.DataFrame:
    if os.path.exists(CACHE) and not rebuild:
        return pd.read_csv(CACHE, parse_dates=["window_end"])
    V = load_views()
    if V.empty:
        return V
    P = load_panel()
    src = dict(zip(P["ticker"], P["src"]))
    wins = {t: sorted(g["window_end"].unique())
            for t, g in V.groupby("ticker")}
    fwd = forward_returns(sorted(V["ticker"].unique()), wins, src)
    V["fwd_ret"] = [fwd.get((t, w), np.nan)
                    for t, w in zip(V["ticker"], V["window_end"])]
    V = V.dropna(subset=["fwd_ret"])
    V = V[V["gross_value"] >= MIN_GROSS]
    V["timing_pnl"] = V["net_value"] * V["fwd_ret"]
    V["year"] = V["window_end"].dt.year
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    # compression MUST be explicit. pandas infers it from the extension, and
    # the temp file ends ".gz.tmp", so the inferred codec is "none" — which
    # writes plain CSV and then renames it to a .gz name. The write succeeds,
    # the file looks right, and the next read dies with "Not a gzipped file".
    tmp = CACHE + ".tmp"
    V.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, CACHE)
    return V


def report(V: pd.DataFrame, draws: int, seed: int) -> None:
    print("=" * 78)
    print(" §12 AT INVESTOR-CLASS RESOLUTION — foreign vs domestic")
    print("=" * 78)
    both = V.pivot_table(index=["ticker", "window_end"], columns="view",
                         values="net_value", aggfunc="sum").dropna()
    print(f"   {V.ticker.nunique()} names, {V.window_end.nunique()} windows, "
          f"{len(V):,} class-windows, "
          f"{V.window_end.min().date()} … {V.window_end.max().date()}")
    print(f"   {len(both):,} ticker-windows have BOTH views")
    res = mirror_residual(V)
    if len(res):
        print(f"   censoring bound: |F_net + D_net| / gross — median "
              f"{res.median():.2%}, p90 {res.quantile(0.9):.2%}")
        print("   (structurally zero; what is left is the top-10 cut, and it")
        print("    is the honest error bar on every net below)")
    print()

    for view, name in (("F", "FOREIGN"), ("D", "DOMESTIC")):
        d = V[V["view"] == view]
        if d.empty:
            print(f" {name}: no rows\n")
            continue
        print(f" {name}  ({len(d):,} class-windows, "
              f"{100*d.gross_value.sum()/V.gross_value.sum():.0f}% of gross)")
        obs = np.nan
        for kind, what in (("block_window", "DIRECTION — was it long at the "
                                            "right TIMES"),
                           ("within_window", "SELECTION — did it pick the "
                                             "right NAMES")):
            obs, nulls, p = permutation_margin(V, view, kind=kind,
                                               draws=draws, seed=seed)
            v = nulls[np.isfinite(nulls)]
            lo, hi = (np.percentile(v, [2.5, 97.5]) if len(v) > 20
                      else (np.nan, np.nan))
            if kind == "block_window":
                print(f"   1. LEVEL        margin {obs:+8.2f} bps per fortnight")
            print(f"      {what}")
            print(f"        null {np.nanmean(v):+8.2f} "
                  f"[{lo:+.2f}, {hi:+.2f}] over {len(v)} draws   p {p:.3f}")
        ann = class_margin(V, view, by="year")
        sp = sign_persistence(ann)
        if sp.get("n_years", 0) >= 3:
            print(f"   2. PERSISTENCE  {sp['share_same_sign']:.0%} of "
                  f"{int(sp['n_years'])} years share the pooled sign; "
                  f"annual sd {sp['sd']:.1f} bps")
            if "lag1_autocorr" in sp:
                print(f"                   lag-1 autocorrelation of the "
                      f"annual margin {sp['lag1_autocorr']:+.3f}")
            print(f"                   annual mean {sp['mean']:+.2f}; "
                  f"dropping {int(sp['largest_year'])} "
                  f"(the largest) {sp['mean_drop_largest']:+.2f}")
        else:
            print(f"   2. PERSISTENCE  only {int(sp.get('n_years', 0))} years "
                  f"— not enough to speak to persistence")
        v_abs = abs(obs) if np.isfinite(obs) else np.nan
        print(f"   3. SIZE         |margin| {v_abs:.2f} bps vs "
              f"{ROUND_TRIP_BPS:.0f} bps round trip — "
              f"{'CLEARS' if v_abs > ROUND_TRIP_BPS else 'does NOT clear'}")
        if sp.get("n_years", 0) >= 3:
            print("      annual: " + "  ".join(
                f"{int(y)}:{m:+.0f}" for y, m in ann.items()))
        print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    V = build(a.rebuild)
    if V.empty:
        print("no investor-split data yet; run investor_split_collect.py")
        return 1
    report(V, a.draws, a.seed)
    print(" All three tests must pass before §12's strategy has an instrument")
    print(" here. Failing any one is a result, not a threshold to loosen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
