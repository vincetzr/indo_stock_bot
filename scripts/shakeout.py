#!/usr/bin/env python3
"""Hold a narrow band without being shaken out of it.

`gap_anatomy.py` located the addressable part of the gap precisely. A causal
band of ``b`` costs ``2b`` per leg, so the ceiling collapses as the band widens:

    band  5%  ->  6,722x       band 20%  ->  50.8x
    band 10%  ->  1,344x       band 25%  ->   9.4x
    band 15%  ->    264x       band 30%  ->    1.7x

The best rule built so far uses a **25%** band and returns 27.3x. But the median
deepest pullback inside a leg is only **11%**. The band is more than twice as
wide as the noise it exists to survive - and it is that wide only because the
narrow versions whipsaw in practice, not because the legs demand it.

So the question is sharp: at a 15% band, what fraction of exits are FALSE - the
leg continues afterwards - and can they be vetoed by something other than price?

Four vetoes are tested. An exit signal fires only if the veto agrees:

    none        exit on the band alone (the baseline)
    trend       do not exit while price is still above its 30-week average
    volume      do not exit on a quiet pullback; real turns should not be quiet
    age         do not exit in the first N weeks of a position, since early
                pullbacks in a young leg are usually continuation
    breadth     do not exit while the index itself is still above its own trend

Each is scored by false-exit rate AND by return, because a veto that removes
false exits by never exiting is not a filter, it is buy-and-hold.

    python3 scripts/shakeout.py [--band 0.15]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV       # noqa: E402
from swing_accuracy import legs, zigzag        # noqa: E402
from turn_trader import (MIN_TURNOVER, ROUND_TRIP,      # noqa: E402
                         clean_weekly, reversal_state, run)

MIN_WEEKS = 200


def banded_state(px: np.ndarray, band: float, veto: Optional[np.ndarray] = None,
                 min_hold: int = 0) -> Tuple[np.ndarray, List[int]]:
    """Reversal band whose EXIT can be vetoed. Returns state and exit bars.

    ``veto[i]`` true means "do not exit on this bar even if the band says so".
    The entry side is untouched: vetoing entries would just be a slower band.
    """
    n = len(prices := px)
    state = np.zeros(n, dtype=np.int8)
    exits: List[int] = []
    long = False
    ext = prices[0]
    held = 0
    for i in range(n):
        p = prices[i]
        if long:
            held += 1
            if p > ext:
                ext = p
            elif 1.0 - p / ext >= band - 1e-12:
                blocked = (veto is not None and bool(veto[i])) or held < min_hold
                if not blocked:
                    long = False
                    ext = p
                    exits.append(i)
        else:
            if p < ext:
                ext = p
            elif p / ext - 1.0 >= band - 1e-12:
                long = True
                ext = p
                held = 0
        state[i] = 1 if long else 0
    return state, exits


def false_exit_rate(px: np.ndarray, exits: List[int], horizon: int = 13) -> float:
    """Share of exits after which the price was HIGHER within ``horizon`` weeks.

    That is the operational definition of a shakeout: you sold and the move you
    sold out of carried on without you.
    """
    if not exits:
        return np.nan
    bad = 0
    for e in exits:
        end = min(e + horizon, len(px) - 1)
        if end > e and px[e + 1:end + 1].max() > px[e] * 1.05:
            bad += 1
    return bad / len(exits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=float, default=0.15)
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--universe", default="bluechip")
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    names = sorted(set(cfg.universe(args.universe)) | set(cfg.universe("lq45")))
    raw = loader.get_many(names, max_age=86400 * 30, verbose=False)

    idxw = clean_weekly(loader.get("^JKSE", max_age=86400 * 30))
    idx_up = (idxw > idxw.rolling(30, min_periods=30).mean()).fillna(False)

    rows: List[Dict] = []
    for t, d in raw.items():
        if len(d) < 500:
            continue
        c = d["close"].astype(float)
        if float((c * d["volume"]).median()) < MIN_TURNOVER:
            continue
        w = clean_weekly(d)
        if w is None or len(w) < MIN_WEEKS:
            continue
        px = w.to_numpy(float)
        years = (w.index[-1] - w.index[0]).days / 365.25
        lg = legs(px, zigzag(px, args.threshold))
        if len(lg) < 4:
            continue

        s = pd.Series(px)
        above30 = (s > s.rolling(30, min_periods=30).mean()).fillna(False).to_numpy()
        x = d.set_index("date").sort_index()
        volw = x["volume"].resample("W-FRI").sum().reindex(w.index)
        vz = ((volw - volw.rolling(52, min_periods=26).mean())
              / volw.rolling(52, min_periods=26).std()).fillna(0.0).to_numpy()
        mkt = idx_up.reindex(w.index, method="ffill").fillna(False).to_numpy()

        vetoes: Dict[str, Tuple[Optional[np.ndarray], int]] = {
            "none": (None, 0),
            "trend (above 30w)": (above30, 0),
            "quiet pullback": (vz < 0.5, 0),
            "age (hold 13w)": (None, 13),
            "market still up": (mkt, 0),
            "trend AND market": (above30 & mkt, 0),
        }
        bh = px[-1] / px[0]
        for name, (v, mh) in vetoes.items():
            st, ex = banded_state(px, args.band, v, mh)
            eq, trades = run(px, st)
            rows.append({
                "ticker": t, "veto": name, "band": args.band,
                "growth": float(eq[-1]),
                "cagr": float(eq[-1]) ** (1 / years) - 1,
                "bh_cagr": bh ** (1 / years) - 1,
                "exits": len(ex), "trades": trades,
                "false_exit": false_exit_rate(px, ex),
                "time_in": float(st.mean())})

    R = pd.DataFrame(rows)
    R["excess"] = R["cagr"] - R["bh_cagr"]
    R.to_csv("reports/shakeout.csv", index=False)

    g = R.groupby("veto").agg(
        false_exit=("false_exit", "median"), exits=("exits", "median"),
        median_excess=("excess", "median"), mean_excess=("excess", "mean"),
        pct_beat=("excess", lambda s: float((s > 0).mean())),
        time_in=("time_in", "median")).reset_index()
    g = g.sort_values("median_excess", ascending=False)

    print(f"\n{'=' * 100}")
    print(f" A {args.band:.0%} BAND ON {R['ticker'].nunique()} BLUE CHIPS — "
          f"can the false exits be vetoed?")
    print("=" * 100)
    print(f" {'veto':<22}{'false exits':>13}{'exits':>8}{'median excess':>16}"
          f"{'beats hold':>12}{'in market':>11}")
    for _, r in g.iterrows():
        print(f" {r['veto']:<22}{r['false_exit']:>13.0%}{r['exits']:>8.0f}"
              f"{r['median_excess']:>+16.2%}{r['pct_beat']:>12.0%}"
              f"{r['time_in']:>11.0%}")

    # the 25% band with no veto, as the incumbent
    inc = []
    for t, sub in R.groupby("ticker"):
        pass
    print(f"\n{'=' * 100}\n AGAINST THE INCUMBENT (25% band, no veto)\n{'=' * 100}")
    base_rows = []
    for t, d in raw.items():
        w = clean_weekly(d) if len(d) >= 500 else None
        if w is None or len(w) < MIN_WEEKS:
            continue
        c = d["close"].astype(float)
        if float((c * d["volume"]).median()) < MIN_TURNOVER:
            continue
        px = w.to_numpy(float)
        years = (w.index[-1] - w.index[0]).days / 365.25
        st = reversal_state(px, 0.25, 0.25)
        eq, _ = run(px, st)
        bh = px[-1] / px[0]
        base_rows.append(float(eq[-1]) ** (1 / years) - 1 - (bh ** (1 / years) - 1))
    print(f" 25% band, no veto:  median excess {np.median(base_rows):+.2%}/yr, "
          f"beats hold on {np.mean(np.array(base_rows) > 0):.0%} of names")
    best = g.iloc[0]
    print(f" {args.band:.0%} band, best veto ({best['veto']}): median excess "
          f"{best['median_excess']:+.2%}/yr, beats hold on {best['pct_beat']:.0%}")
    delta = best["median_excess"] - float(np.median(base_rows))
    print(f"\n narrowing the band from 25% to {args.band:.0%} with that veto is worth "
          f"{delta:+.2%}/yr")
    print("\n -> reports/shakeout.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
