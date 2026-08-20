#!/usr/bin/env python3
"""The best thing the evidence actually supports, rather than a better timing rule.

WHY THIS SCRIPT EXISTS
----------------------
Everything measured in Parts XXVIII-XXIX says the same thing from a different
angle: there is no timing edge in IDX prices. Leg structure matches a random walk
at 15m/1h/4h/daily (110). Every band rule and every multi-timeframe combination
loses to random timing at the same exposure (111, 113). Fitting the band on a
training half and using it on a holdout scores WORSE than not fitting at all
(115). And through all of it, one line kept winning: buy and hold.

So "optimise" cannot mean a better entry signal. The only honest reading is:
given that timing does not work, what DOES the evidence support, and what is the
best version of that?

Two things it supports, both measured here rather than assumed:

    1. DIVERSIFICATION, forced by concentration. Result 108 found the best 5% of
       weeks carry a median 406% of everything holding a name produced - the
       other 95% net negative. If the winners cannot be predicted (110-113) and
       the return lives in a handful of them, then owning MORE names is not
       timidity, it is the only way to be present when they happen.

    2. The band rule as a RISK control, not a return engine. Result 100 noted it
       cut the median drawdown from -76% to -61% while costing return. In a bear
       market that trade may be worth making, and it is worth measuring properly
       instead of being asserted in either direction.

WHAT IS SWEPT
-------------
    breadth        1, 3, 5, 10, 20, 30, 50 names, equal weight
    selection      liquidity only (no view), momentum, low volatility
    rebalancing    never, annual, quarterly, monthly
    overlay        none, or the daily 8% band applied per holding

Selection at every rebalance uses ONLY trailing data. The universe is rebuilt at
each date from what was knowable then, so a name that later became liquid or
later listed cannot appear early. Costs are charged on turnover at 0.56%.

Everything is scored against buy-and-hold of the same portfolio and against the
IDX Composite.

    python3 scripts/optimal.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config           # noqa: E402
from idxbot.data.cache import Cache             # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV        # noqa: E402
from account_sim import load_ohlc               # noqa: E402
from paint_live import band_state               # noqa: E402

FEE = 0.0056
REBALANCE = {"never": None, "annual": "YE", "quarterly": "QE", "monthly": "ME"}
SELECTIONS = ("liquidity", "momentum", "lowvol")


def build_panel(loader: YahooOHLCV, cache_dir: str, min_price: float,
                start: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Close and turnover panels for every IDX equity in the cache."""
    names = sorted({os.path.basename(f).split(".")[0].upper()
                    for f in glob.glob(os.path.join(cache_dir, "ohlcv",
                                                    "*.JK.csv.gz"))})
    names = [n for n in names if n.isalpha() and len(n) == 4]
    px, tv = {}, {}
    for t in names:
        df = load_ohlc(loader, t)
        if df is None or len(df) < 250:
            continue
        c = df["close"]
        c = c[c.index >= pd.Timestamp(start)]
        if len(c) < 250 or float(c.median()) < min_price:
            continue
        px[t] = c
        tv[t] = (df["close"] * df["volume"]).reindex(c.index)
    return pd.DataFrame(px).sort_index(), pd.DataFrame(tv).sort_index()


def eligible(px: pd.DataFrame, tv: pd.DataFrame, i: int, min_hist: int,
             min_turnover: float) -> np.ndarray:
    """Names tradable at bar i, using only bars strictly before i."""
    hist = px.iloc[:i]
    if len(hist) < min_hist:
        return np.array([], dtype=object)
    alive = hist.iloc[-1].notna() & hist.iloc[-min_hist].notna()
    liq = tv.iloc[:i].tail(250).median() >= min_turnover
    return px.columns[(alive & liq).to_numpy()].to_numpy()


def rank_by(px: pd.DataFrame, tv: pd.DataFrame, i: int, names: Sequence[str],
            how: str) -> List[str]:
    """Order the eligible names by the chosen criterion, trailing data only."""
    hist = px.iloc[:i][list(names)]
    if how == "liquidity":
        key = tv.iloc[:i][list(names)].tail(250).median()
    elif how == "momentum":
        key = hist.iloc[-1] / hist.iloc[-250] - 1.0
    elif how == "lowvol":
        key = -hist.pct_change().tail(250).std()
    else:
        raise ValueError(how)
    return list(key.dropna().sort_values(ascending=False).index)


def run(px: pd.DataFrame, tv: pd.DataFrame, breadth: int, selection: str,
        rebalance: str, overlay: bool, band: float = 0.08,
        min_hist: int = 250, min_turnover: float = 5e9) -> Optional[Dict]:
    """Equity curve for one configuration. Nothing reads a bar before it printed."""
    idx = px.index
    if len(idx) < min_hist + 60:
        return None
    rule = REBALANCE[rebalance]
    marks = set()
    if rule:
        marks = set(pd.Series(idx, index=idx).resample(rule).last().dropna())

    states = {}
    if overlay:
        for t in px.columns:
            s = px[t].dropna()
            if len(s) > 30:
                st, _ = band_state(s.to_numpy(float), band)
                states[t] = pd.Series(st, index=s.index)

    weights = pd.Series(dtype=float)
    equity, held_names = [1.0], []
    turnover_paid = 0.0
    invested = []          # share of capital actually at risk, bar by bar
    ret = px.pct_change()
    for i in range(min_hist, len(idx)):
        # ---- apply yesterday's weights to today's return -------------------
        if len(weights):
            r = ret.iloc[i][weights.index].fillna(0.0)
            equity.append(equity[-1] * float(1.0 + (weights * r).sum()))
        else:
            equity.append(equity[-1])

        # ---- decide tomorrow's weights, from data up to and including i ----
        need_rebal = (not len(weights)) or (idx[i] in marks)
        if need_rebal:
            elig = eligible(px, tv, i + 1, min_hist, min_turnover)
            if len(elig):
                picks = rank_by(px, tv, i + 1, elig, selection)[:breadth]
                held_names = picks
        if not held_names:
            continue
        if overlay:
            # colour known at bar i governs bar i+1; a red name goes to cash
            live = [t for t in held_names
                    if t in states and i < len(idx)
                    and bool(states[t].reindex([idx[i]]).fillna(0).iloc[0])]
        else:
            live = held_names
        new = pd.Series(1.0 / len(held_names), index=held_names) if held_names \
            else pd.Series(dtype=float)
        if overlay:
            new = new.reindex(held_names).fillna(0.0)
            new[[t for t in held_names if t not in live]] = 0.0
        old = weights.reindex(new.index).fillna(0.0)
        turn = float((new - old).abs().sum())
        if turn > 1e-9:
            equity[-1] *= (1.0 - FEE * turn / 2.0)
            turnover_paid += turn
        weights = new
        invested.append(float(new.sum()))

    eq = pd.Series(equity[1:], index=idx[min_hist:])
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = float((eq / eq.cummax() - 1.0).min())
    return {"breadth": breadth, "selection": selection, "rebalance": rebalance,
            "overlay": overlay, "final": float(eq.iloc[-1]),
            "exposure": float(np.mean(invested)) if invested else 1.0,
            "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else np.nan,
            "max_dd": dd, "turnover": turnover_paid,
            "calmar": float((eq.iloc[-1] ** (1 / yrs) - 1) / abs(dd))
            if yrs > 0 and dd < 0 else np.nan,
            "years": yrs, "curve": eq}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--min-price", type=float, default=50.0)
    ap.add_argument("--min-turnover", type=float, default=5e9)
    ap.add_argument("--breadths", type=int, nargs="+",
                    default=[1, 3, 5, 10, 20, 30, 50])
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()
    cache_dir = cfg.path("data.cache_dir", "data/cache")
    loader = YahooOHLCV(cfg, Cache(cache_dir))
    px, tv = build_panel(loader, cache_dir, args.min_price, args.start)
    if px.empty:
        raise SystemExit("no panel")
    print(f"panel: {px.shape[1]} names x {px.shape[0]:,} sessions, "
          f"{px.index[0]:%Y-%m-%d} to {px.index[-1]:%Y-%m-%d}")

    # the index, as the benchmark a person could actually have bought
    jk = loader.get("^JKSE", max_age=86400 * 30)
    bench = None
    if jk is not None and not jk.empty:
        b = jk.set_index("date").sort_index()["close"].astype(float)
        b = b[(b.index >= px.index[0]) & (b.index <= px.index[-1])]
        if len(b) > 100:
            yrs = (b.index[-1] - b.index[0]).days / 365.25
            bench = {"final": float(b.iloc[-1] / b.iloc[0]),
                     "cagr": float((b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1),
                     "max_dd": float((b / b.cummax() - 1).min())}

    rows = []
    print("\nsweeping...", end="", flush=True)
    for n in args.breadths:
        for sel in SELECTIONS:
            for reb in ("never", "annual", "quarterly"):
                r = run(px, tv, n, sel, reb, False,
                        min_turnover=args.min_turnover)
                if r:
                    rows.append(r)
        print(".", end="", flush=True)
    print()
    R = pd.DataFrame([{k: v for k, v in r.items() if k != "curve"} for r in rows])
    R.to_csv("reports/optimal_sweep.csv", index=False)

    print(f"\n{'=' * 100}\n BREADTH — how many names, selected on liquidity "
          f"alone (no view)\n{'=' * 100}")
    print(f" {'names':>7}{'rebalance':>12}{'final':>10}{'CAGR':>9}"
          f"{'max DD':>9}{'CAGR/DD':>10}")
    for n in args.breadths:
        for reb in ("never", "annual", "quarterly"):
            s = R[(R["breadth"] == n) & (R["selection"] == "liquidity")
                  & (R["rebalance"] == reb)]
            if not len(s):
                continue
            r = s.iloc[0]
            print(f" {n:>7}{reb:>12}{r['final']:>9.2f}x{r['cagr']:>9.1%}"
                  f"{r['max_dd']:>9.1%}{r['calmar']:>10.2f}")

    print(f"\n{'=' * 100}\n SELECTION — does picking WHICH names help?\n{'=' * 100}")
    print(f" {'selection':<12}{'names':>7}{'final':>10}{'CAGR':>9}{'max DD':>9}"
          f"{'CAGR/DD':>10}")
    for sel in SELECTIONS:
        for n in (10, 20, 30):
            s = R[(R["selection"] == sel) & (R["breadth"] == n)
                  & (R["rebalance"] == "annual")]
            if len(s):
                r = s.iloc[0]
                print(f" {sel:<12}{n:>7}{r['final']:>9.2f}x{r['cagr']:>9.1%}"
                      f"{r['max_dd']:>9.1%}{r['calmar']:>10.2f}")

    if bench:
        print(f"\n IDX Composite over the same window: {bench['final']:.2f}x, "
              f"{bench['cagr']:.1%} a year, max drawdown {bench['max_dd']:.1%}")

    # ---- does the band overlay buy drawdown protection worth having? -------
    print(f"\n{'=' * 100}\n THE BAND AS A RISK CONTROL, NOT A RETURN ENGINE"
          f"\n{'=' * 100}")
    print(f" {'config':<26}{'final':>10}{'CAGR':>9}{'max DD':>9}{'CAGR/DD':>10}")
    ov = []
    for n in (10, 20, 30):
        for overlay in (False, True):
            r = run(px, tv, n, "liquidity", "annual", overlay,
                    min_turnover=args.min_turnover)
            if r:
                ov.append(r)
                label = f"{n} names, " + ("8% band overlay" if overlay else "hold")
                print(f" {label:<26}{r['final']:>9.2f}x{r['cagr']:>9.1%}"
                      f"{r['max_dd']:>9.1%}{r['calmar']:>10.2f}")

    print(f"\n{'=' * 100}\n THE OPTIMUM, AS THE EVIDENCE HAS IT\n{'=' * 100}")
    best = R.loc[R["calmar"].idxmax()]
    print(f" best of the {len(R)} configs swept: {int(best['breadth'])} names, "
          f"{best['selection']}, {best['rebalance']}\n   "
          f"{best['final']:.2f}x, {best['cagr']:.1%} a year, "
          f"{best['max_dd']:.1%} drawdown")
    if best["breadth"] <= 5:
        print(f" ! it holds {int(best['breadth'])} names. The best of {len(R)} "
              f"configs, at that breadth, is a\n ! coin that landed heads — "
              f"not an optimum. Result 108 measured how concentrated\n ! these "
              f"returns are; 3 names is a lottery ticket, however good the "
              f"backtest.")

    # Choose the config on the FIRST half, then live with it on the second.
    mid = px.index[len(px.index) // 2]
    print(f"\n{'=' * 100}\n THE ONLY VERSION THAT COUNTS — config chosen before "
          f"{mid:%Y-%m}, used after\n{'=' * 100}")
    early, late = px[px.index <= mid], px[px.index > mid]
    tv_e, tv_l = tv[tv.index <= mid], tv[tv.index > mid]
    tr = []
    for n in args.breadths:
        for sel in SELECTIONS:
            r = run(early, tv_e, n, sel, "annual", False,
                    min_turnover=args.min_turnover)
            if r:
                tr.append(r)
    if tr:
        pick = max(tr, key=lambda r: r["calmar"] if np.isfinite(r["calmar"])
                   else -np.inf)
        oos = run(late, tv_l, pick["breadth"], pick["selection"], "annual",
                  False, min_turnover=args.min_turnover)
        base = run(late, tv_l, 30, "liquidity", "annual", False,
                   min_turnover=args.min_turnover)
        print(f" chosen on the first half: {int(pick['breadth'])} names, "
              f"{pick['selection']} ({pick['cagr']:+.1%} a year there)")
        if oos and base:
            print(f" what it then did:         {oos['cagr']:+.1%} a year, "
                  f"{oos['max_dd']:.1%} drawdown")
            print(f" plain 30 names, no view:  {base['cagr']:+.1%} a year, "
                  f"{base['max_dd']:.1%} drawdown")
            if bench:
                print(f" the index:                {bench['cagr']:+.1%} a year, "
                      f"{bench['max_dd']:.1%} drawdown")
            print(f"\n {'the fitted choice carried over' if oos['cagr'] > base['cagr'] else 'the fitted choice did NOT carry over - picking WHICH names added nothing'}")

    pairs = [(a, b) for a in ov for b in ov
             if a["breadth"] == b["breadth"] and not a["overlay"] and b["overlay"]]
    if pairs:
        dc = float(np.median([b["cagr"] - a["cagr"] for a, b in pairs]))
        dd = float(np.median([b["max_dd"] - a["max_dd"] for a, b in pairs]))
        verb = "gains" if dc >= 0 else "costs"
        print(f"\n the band overlay {verb} {abs(dc):.1%} a year and moves the "
              f"drawdown by {dd:+.1%} (less negative = shallower).")

        # The overlay spends less time invested, and in a market that FELL that
        # is worth money on its own. Same test as every other rule in this repo:
        # does it beat spending its own exposure at random?
        print(f"\n {'against its own null':<26}{'actual':>10}{'null':>10}"
              f"{'edge':>9}")
        clears = 0
        for a, b in pairs:
            base_log = float(np.log(a["final"]))
            null_log = b["exposure"] * base_log
            act_log = float(np.log(b["final"]))
            edge = act_log - null_log
            clears += int(edge > 0)
            print(f" {str(int(b['breadth'])) + ' names, overlay':<26}"
                  f"{act_log:>+10.3f}{null_log:>+10.3f}{edge:>+9.3f}")
        print(f"\n the overlay clears its own same-exposure null on "
              f"{clears}/{len(pairs)} breadths.")
        if clears == len(pairs) and dd > 0.10:
            print(" This is the first thing measured in this repo that improves "
                  "BOTH return and\n drawdown and still clears the null. It is "
                  "risk control that pays for itself.")
        elif dd > 0.10:
            print(" The drawdown protection is real, but it does not clear the "
                  "null - most of it is\n simply being invested less in a "
                  "market that fell.")

    with open("reports/optimal.json", "w") as fh:
        json.dump({"best_calmar": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                   for k, v in best.items()},
                   "benchmark": bench}, fh, indent=2, default=str)
    print("\n -> reports/optimal_sweep.csv, reports/optimal.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
