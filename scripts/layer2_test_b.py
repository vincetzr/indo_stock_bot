#!/usr/bin/env python3
"""Run Protocol B on the names that did NOT generate it.

The exclusion is the whole point. BBCA produced these hypotheses by being looked
at; testing them on BBCA would measure how well a hypothesis fits the data that
suggested it, which is guaranteed to look good and means nothing.

Two gates a result must pass here that Protocol A did not have:

    the exclusion   BBCA is dropped before anything is computed
    the control     flow correlates +0.22 with the day's own return, and up days
                    mean-revert, so every hypothesis is re-run inside same-day
                    UP days and inside same-day DOWN days separately. A signal
                    that only works on one side is short-term reversal wearing a
                    flow costume.

    python3 scripts/layer2_test_b.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.config import load_config                     # noqa: E402
from idxbot.data.cache import Cache                       # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV                  # noqa: E402
from layer2_protocol import ALPHA, POWER, detectable_effect   # noqa: E402
from layer2_protocol_b import (CONTROLS, EXCLUDED,        # noqa: E402
                               HYPOTHESES_B, protocol_b_hash)
from layer2_test import (attach_returns, build_panel,      # noqa: E402
                         cluster_by_day, one_sided)
from factor_study import total_return_series               # noqa: E402


def add_same_day_return(P: pd.DataFrame, loader: YahooOHLCV) -> pd.DataFrame:
    """The day's own move, which is the alternative explanation to rule out."""
    px = {}
    for tk in sorted(P["ticker"].unique()):
        d = total_return_series(loader, tk, total=True)
        if d is not None:
            px[tk] = d["px"]
    vals = np.full(len(P), np.nan)
    for tk, g in P.groupby("ticker"):
        if tk not in px:
            continue
        s = px[tk]
        pos = {d: i for i, d in enumerate(s.index)}
        for j, d in zip(g.index, g["date"]):
            i = pos.get(pd.Timestamp(d))
            if i is not None and i > 0 and float(s.iloc[i - 1]) > 0:
                vals[P.index.get_loc(j)] = float(s.iloc[i] / s.iloc[i - 1] - 1.0)
    P["ret_t"] = vals
    return P


def run_b(P: pd.DataFrame, h: Dict, subset: Optional[str] = None,
          top_q: float = 0.8) -> Dict:
    col, ex = h["signal"], f"ex{h['horizon']}"
    if col not in P or ex not in P:
        return {"n": 0}
    d = P.dropna(subset=[ex])
    if subset == "up":
        d = d[d["ret_t"] > 0]
    elif subset == "down":
        d = d[d["ret_t"] <= 0]
    if d.empty:
        return {"n": 0}
    events = d[d[col]] if col == "streak3" else d[d[col] >= d[col].quantile(top_q)]
    # The horizon MUST reach one_sided. Without it the overlapping forward
    # windows are treated as independent and the t-statistic is inflated -
    # which is exactly how H6 first read p = 0.0041 instead of p = 0.106.
    out = one_sided(cluster_by_day(events, ex), h["direction"], h["horizon"])
    out["events"] = len(events)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="lo,mid,hi")
    args = ap.parse_args()
    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    a = ALPHA / len(HYPOTHESES_B)

    print(f"{'=' * 96}\n LAYER-2 PROTOCOL B — {protocol_b_hash()}\n{'=' * 96}")
    panels = {}
    for lvl in args.levels.split(","):
        P = build_panel("ipot-all", lvl)
        if P.empty:
            continue
        P = P[~P["ticker"].isin(EXCLUDED)].reset_index(drop=True)
        if P.empty:
            continue
        P = add_same_day_return(attach_returns(P, loader), loader)
        panels[lvl] = P
    if not panels:
        print(f" no data outside the excluded names {EXCLUDED}. Nothing to test.")
        return 1

    # `panels.get("mid") or ...` puts a DataFrame in a boolean context, which
    # pandas refuses. The fallback has to be an explicit None check.
    P = panels["mid"] if "mid" in panels else next(iter(panels.values()))
    names, days = P["ticker"].nunique(), P["date"].nunique()
    deff = 1.0 + (names - 1) * 0.30
    eff = days * names / deff
    mde = detectable_effect(int(eff), a, POWER)
    print(f" confirmatory sample EXCLUDES {', '.join(EXCLUDED)}")
    print(f" {names} untouched names x {days} sessions = {len(P):,} ticker-days,"
          f" {P['date'].min():%Y-%m-%d} to {P['date'].max():%Y-%m-%d}")
    print(f" design effect {deff:.1f}, effective n about {eff:,.0f}, "
          f"smallest detectable effect d = {mde:.3f}")
    ready = mde <= 0.20
    if not ready:
        print(f"\n ! UNDERPOWERED — the protocol's stopping rule says do not "
              f"report a verdict.\n ! Numbers below are printed as a progress "
              f"check, not as a result.")

    print(f"\n{'=' * 96}\n RESULTS — one-sided NEGATIVE, Bonferroni alpha "
          f"{a:.4f}\n{'=' * 96}")
    print(f" {'':<4}{'signal':<15}{'h':>3}{'lvl':>5}{'days':>6}{'events':>7}"
          f"{'mean ex':>10}{'d':>8}{'t':>7}{'p':>9}"
          f"{'up-days':>10}{'down-days':>11}")
    survives: Dict[str, List[bool]] = {}
    #: A signal only responds to the censoring level if it is DERIVED from the
    #: bracketed net figure. `concentration` and `foreign_net` are computed
    #: from observed buy lots and the published foreign value, so re-running
    #: them at three levels re-runs the identical arithmetic three times.
    #: Reporting that as "survives at every censoring level" would claim three
    #: independent robustness checks where there is only one.
    invariant: Dict[str, bool] = {}
    for h in HYPOTHESES_B:
        seen: List[float] = []
        for lvl, Pl in panels.items():
            r = run_b(Pl, h)
            if not r.get("n"):
                continue
            if np.isfinite(r.get("mean", np.nan)):
                seen.append(round(float(r["mean"]), 12))
            invariant[h["id"]] = len(panels) > 1 and len(set(seen)) == 1
            up = run_b(Pl, h, "up")
            dn = run_b(Pl, h, "down")
            sig = np.isfinite(r["p"]) and r["p"] < a
            both = (np.isfinite(up.get("d", np.nan)) and up["d"] < 0
                    and np.isfinite(dn.get("d", np.nan)) and dn["d"] < 0)
            survives.setdefault(h["id"], []).append(bool(sig and both))
            print(f" {h['id']:<4}{h['signal']:<15}{h['horizon']:>3}{lvl:>5}"
                  f"{r['n']:>6}{r['events']:>7}{r['mean']:>10.3%}{r['d']:>8.3f}"
                  f"{r['t']:>7.2f}{r['p']:>9.4f}"
                  f"{up.get('d', float('nan')):>10.2f}"
                  f"{dn.get('d', float('nan')):>11.2f}")

    print(f"\n{'=' * 96}\n READING\n{'=' * 96}")
    print(f" The last two columns are the required control ({CONTROLS[0]}). A "
          f"real flow effect\n is negative in BOTH; one negative and one "
          f"positive is short-term reversal.")
    print(f" 't' and 'p' are Newey-West corrected for the overlap between "
          f"consecutive\n forward windows. At h = 20 that correction is worth "
          f"about a factor of two.\n")
    for hid, v in survives.items():
        note = ("  (censoring level does NOT move this signal — the three rows "
                "above are one\n      check printed three times, not three)"
                if invariant.get(hid) else "")
        if not ready:
            print(f" {hid}: no verdict — underpowered.{note}")
        elif all(v) and v:
            lvls = ("both control subsets" if invariant.get(hid)
                    else "every censoring level and both control subsets")
            print(f" {hid}: SURVIVES at {lvls}.{note}")
        elif any(v):
            print(f" {hid}: partial — fails at some censoring level or one "
                  f"control subset. Not a result.{note}")
        else:
            print(f" {hid}: does not survive.{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
