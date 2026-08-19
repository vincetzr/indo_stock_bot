#!/usr/bin/env python3
"""Choosing the band, with a reason instead of a grid search.

Result 109 gives an exact break-even: at band ``b`` a leg must exceed

    M* = 2b / (1 - b)

to be profitable at all. That turns band selection from "try values and keep the
best backtest" - which is how a curve gets fitted - into a question about the
leg-size DISTRIBUTION of the series being traded, which is measurable on past
data alone and does not need the future to answer.

Note the fixed point: raising b makes the surviving legs bigger, but it raises
M* just as fast. So there is a genuine optimum rather than a monotone
improvement, and it sits wherever the leg distribution outruns the toll fastest.

Three selectors are compared, all fitted on the training window only:

    fixed       b = 0.08, the value used up to now, chosen by nothing
    grid        the b with the best training equity - the fitted answer, and
                the benchmark for how much of that fit survives out of sample
    breakeven   the b that maximises the training expected log return implied by
                the toll law: for each leg, what the band could keep, summed and
                divided by elapsed bars, minus fees per round trip

If ``breakeven`` matches ``grid`` in sample but beats it out of sample, the law
is doing real work. If it does not, the law is a nice identity that does not
help pick anything, and this script should say so.

Works on any bar size: pass a loader for daily, 4h, 1h or 15m bars.

    python3 scripts/band_optimizer.py --timeframe daily
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV       # noqa: E402
from capture_toll import breakeven_move, capture_ceiling   # noqa: E402
from leg_signals import market_caps            # noqa: E402
from paint_daily import unadjusted_daily       # noqa: E402
from paint_live import band_state              # noqa: E402
from swing_accuracy import legs, zigzag        # noqa: E402

ROUND_TRIP_FEE = 0.0056        # 0.28% buy + 0.18% sell + 0.10% sale tax


def equity_log(px: np.ndarray, band: float, fee: float = ROUND_TRIP_FEE) -> float:
    """Total log return of trading the band, long only, fees on every round trip.

    Colour known at bar t-1 governs bar t. Nothing here reads a future bar.
    """
    st, _ = band_state(px, band)
    if len(px) < 3:
        return 0.0
    r = np.diff(np.log(px))
    held = st[:-1].astype(bool)
    gross = float(r[held].sum())
    trips = int((np.diff(st.astype(int)) == 1).sum())
    return gross + trips * np.log(1.0 - fee)


def toll_expectancy(px: np.ndarray, band: float,
                    fee: float = ROUND_TRIP_FEE) -> float:
    """Expected log return per bar implied by the toll law on this series.

    Uses only the leg-size distribution and the arithmetic ceiling - never the
    realised path of any particular trade - so it is a statement about the shape
    of the series rather than about one sequence of trades. Legs below break-even
    contribute their real (negative) ceiling, because the rule is obliged to
    trade them.
    """
    piv = zigzag(px, band)
    lg = legs(px, piv)
    if len(lg) < 2:
        return -np.inf
    tot = 0.0
    for _a, _b, r in lg:
        if r <= 0:
            continue                        # long only: down legs sit in cash
        cap = capture_ceiling(r, band)
        if cap <= 0:
            continue
        tot += np.log(cap) + np.log(1.0 - fee)
    return tot / max(len(px), 1)


def pick_band(px: np.ndarray, grid: np.ndarray, how: str) -> float:
    if how == "grid":
        return float(grid[int(np.argmax([equity_log(px, b) for b in grid]))])
    if how == "breakeven":
        return float(grid[int(np.argmax([toll_expectancy(px, b) for b in grid]))])
    raise ValueError(how)


def walk_forward(px: np.ndarray, grid: np.ndarray, folds: int = 4,
                 min_train: int = 250) -> Optional[Dict[str, float]]:
    """Choose on train, score on test, never the other way round."""
    n = len(px)
    if n < min_train + 120:
        return None
    edges = np.linspace(min_train, n, folds + 1).astype(int)
    out = {k: 0.0 for k in ("fixed", "grid", "breakeven", "hold")}
    bands = {"grid": [], "breakeven": []}
    tested = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi - lo < 40:
            continue
        train, test = px[:lo], px[lo - 1:hi]
        for how in ("grid", "breakeven"):
            b = pick_band(train, grid, how)
            bands[how].append(b)
            out[how] += equity_log(test, b)
        out["fixed"] += equity_log(test, 0.08)
        out["hold"] += float(np.log(test[-1] / test[0]))
        tested += 1
    if not tested:
        return None
    res = {f"{k}_log": v for k, v in out.items()}
    res["folds"] = tested
    res["band_grid"] = float(np.median(bands["grid"]))
    res["band_breakeven"] = float(np.median(bands["breakeven"]))
    return res


def profile(px: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    """The leg distribution against the toll, band by band - the whole argument."""
    rows = []
    for b in grid:
        piv = zigzag(px, b)
        ups = [r for _a, _b, r in legs(px, piv) if r > 0]
        if not ups:
            continue
        m = breakeven_move(b)
        ups = np.array(ups)
        rows.append({
            "band": b, "legs": len(ups),
            "median_move": float(np.median(ups)),
            "breakeven": m,
            "above_breakeven": float((ups > m).mean()),
            "ratio": float(np.median(ups) / m),
            "expectancy": toll_expectancy(px, b),
        })
    return pd.DataFrame(rows)


def random_walk_profile(px: np.ndarray, grid: np.ndarray, draws: int = 20,
                        seed: int = 0) -> pd.DataFrame:
    """The same leg/M* profile for a driftless walk with the SAME volatility.

    This is the null that decides whether band tuning can work at all. A zigzag
    on a random walk still produces legs, and those legs still have a median
    size - so "the legs are bigger than the band" is not evidence of anything on
    its own. If the real series matches the walk, the leg distribution carries no
    information a band could exploit, and no amount of tuning creates any.
    """
    rng = np.random.default_rng(seed)
    r = np.diff(np.log(px))
    sd = float(np.std(r))
    out = []
    for d in range(draws):
        sim = float(px[0]) * np.exp(np.cumsum(rng.normal(0.0, sd, len(px) - 1)))
        sim = np.concatenate([[px[0]], sim])
        p = profile(sim, grid)
        if len(p):
            out.append(p.assign(draw=d))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", default="daily",
                    choices=["daily", "4h", "1h", "15m"])
    ap.add_argument("--min-mcap", type=float, default=1e13)
    ap.add_argument("--names", type=int, default=60)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--grid", type=float, nargs="+",
                    default=[0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12,
                             0.15, 0.20, 0.25, 0.30])
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)
    grid = np.array(args.grid, dtype=float)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    mc = market_caps()
    big = sorted(mc[mc >= args.min_mcap].index)

    if args.timeframe == "daily":
        get = lambda t: unadjusted_daily(loader, t)      # noqa: E731
        min_len = 500
    else:
        from tf_data import load_4h, load_1h, load_15m   # noqa: E402
        fn = {"4h": load_4h, "1h": load_1h, "15m": load_15m}[args.timeframe]

        def get(t):
            d = fn(t)
            return None if d is None or not len(d) else d["close"].astype(float)
        min_len = {"4h": 300, "1h": 800, "15m": 400}[args.timeframe]

    series: Dict[str, np.ndarray] = {}
    for t in big:
        s = get(t)
        if s is not None and len(s) >= min_len:
            series[t] = s.to_numpy(float)
        if len(series) >= args.names:
            break
    if not series:
        raise SystemExit(f"no usable {args.timeframe} series")
    print(f"{args.timeframe}: {len(series)} names, median "
          f"{int(np.median([len(v) for v in series.values()])):,} bars each")

    # --- the shape of the problem, before any optimisation ------------------
    prof = pd.concat([profile(v, grid).assign(ticker=k)
                      for k, v in series.items()], ignore_index=True)
    P = prof.groupby("band").agg(
        legs=("legs", "median"), median_move=("median_move", "median"),
        breakeven=("breakeven", "first"), above=("above_breakeven", "median"),
        ratio=("ratio", "median"), exp=("expectancy", "median")).reset_index()
    P.to_csv(f"reports/band_profile_{args.timeframe}.csv", index=False)

    print(f"\n{'=' * 86}\n LEG SIZES vs THE TOLL — {args.timeframe} "
          f"(median across {len(series)} names)\n{'=' * 86}")
    print(f" {'band':>6}{'break-even':>12}{'legs':>7}{'median leg':>12}"
          f"{'leg/M*':>9}{'above M*':>10}{'exp/bar':>11}")
    for _, r in P.iterrows():
        print(f" {r['band']:>6.0%}{r['breakeven']:>12.1%}{r['legs']:>7.0f}"
              f"{r['median_move']:>12.1%}{r['ratio']:>9.2f}"
              f"{r['above']:>10.0%}{r['exp']:>11.5f}")
    best_ratio = P.loc[P["ratio"].idxmax(), "band"]
    best_exp = P.loc[P["exp"].idxmax(), "band"]
    print(f"\n leg/M* peaks at band {best_ratio:.0%}; toll expectancy peaks at "
          f"{best_exp:.0%}")

    # --- the null that decides whether any of this can be tuned -------------
    sub = list(series.items())[:min(12, len(series))]
    rw = pd.concat([random_walk_profile(v, grid, draws=8, seed=i)
                    for i, (_k, v) in enumerate(sub)], ignore_index=True)
    if len(rw):
        R = rw.groupby("band").agg(ratio=("ratio", "median"),
                                   above=("above_breakeven", "median")).reset_index()
        M = P[["band", "ratio", "above"]].merge(R, on="band",
                                                suffixes=("_real", "_walk"))
        M.to_csv(f"reports/band_null_{args.timeframe}.csv", index=False)
        print(f"\n{'=' * 86}\n THE NULL — same volatility, no drift, no "
              f"structure ({len(sub)} names x 8 draws)\n{'=' * 86}")
        print(f" {'band':>6}{'leg/M* real':>14}{'leg/M* walk':>14}"
              f"{'above M* real':>16}{'above M* walk':>16}")
        for _, r in M.iterrows():
            print(f" {r['band']:>6.0%}{r['ratio_real']:>14.2f}"
                  f"{r['ratio_walk']:>14.2f}{r['above_real']:>16.0%}"
                  f"{r['above_walk']:>16.0%}")
        gap = float((M["ratio_real"] - M["ratio_walk"]).median())
        print(f"\n median gap, real minus random walk: {gap:+.3f}")
        if abs(gap) < 0.05:
            print(" The real leg distribution is indistinguishable from a "
                  "random walk's.\n Legs being 'bigger than the band' is "
                  "therefore not evidence of anything:\n a walk does it too. "
                  "No choice of band can extract what is not there,\n which is "
                  "why the grid search below tops out below buy-and-hold.")
        else:
            print(" The real leg distribution differs from the walk, so there "
                  "is something\n for a band to exploit - see whether the "
                  "walk-forward below actually does.")

    # --- does choosing on that basis survive out of sample? -----------------
    rows = []
    for t, px in series.items():
        r = walk_forward(px, grid, args.folds)
        if r:
            rows.append({"ticker": t, **r})
    W = pd.DataFrame(rows)
    W.to_csv(f"reports/band_walkforward_{args.timeframe}.csv", index=False)

    print(f"\n{'=' * 86}\n OUT OF SAMPLE — band chosen on the training window "
          f"only, {len(W)} names\n{'=' * 86}")
    print(f" {'selector':<14}{'median OOS log':>16}{'beats hold':>13}"
          f"{'median band':>14}")
    hold = W["hold_log"].median()
    for key, label in (("fixed", "fixed 8%"), ("grid", "grid (fitted)"),
                       ("breakeven", "break-even law"), ("hold", "buy & hold")):
        col = W[f"{key}_log"]
        wins = float((col > W["hold_log"]).mean()) if key != "hold" else np.nan
        bcol = {"grid": "band_grid", "breakeven": "band_breakeven"}.get(key)
        bstr = f"{W[bcol].median():.0%}" if bcol else "-"
        wstr = f"{wins:.0%}" if np.isfinite(wins) else "-"
        print(f" {label:<14}{col.median():>16.3f}{wstr:>13}{bstr:>14}")

    print(f"\n{'=' * 86}\n VERDICT\n{'=' * 86}")
    g, bkv = W["grid_log"].median(), W["breakeven_log"].median()
    if bkv > g:
        print(f" The law picks better than the grid out of sample "
              f"({bkv:.3f} vs {g:.3f}).\n Choosing the band from the leg "
              f"distribution generalises further than choosing\n it from the "
              f"best training equity, which is what over-fitting looks like\n "
              f"when you measure it.")
    else:
        print(f" The law does NOT pick better than the grid out of sample "
              f"({bkv:.3f} vs {g:.3f}).\n It explains WHY a band wins - the "
              f"break-even is real arithmetic - but on this\n evidence it is "
              f"not a better selector, and should not be sold as one.")
    if max(g, bkv) < hold:
        print(f"\n Neither beats simply holding ({hold:.3f}), which is the "
              f"result that matters\n and is unchanged from Result 100.")
    print(f"\n -> reports/band_profile_{args.timeframe}.csv, "
          f"reports/band_walkforward_{args.timeframe}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
