#!/usr/bin/env python3
"""Does Hull Suite + UT Bot beat owning the stock? Tested on IDX large caps.

Run in stages so the cheap, unfittable questions get answered before any
parameter is touched:

    baseline   the published defaults, no fitting at all, against buy-and-hold
    costs      how much of the result is edge and how much is friction
    eras       whether it ever worked, and whether it still does
    grid       the parameter surface: a plateau or a spike
    walk       fit in-sample, score out-of-sample, against fixed defaults

    python3 scripts/hullut_study.py baseline costs eras
    python3 scripts/hullut_study.py grid walk

The order is the argument. Anyone can find a profitable configuration by
searching hard enough; the question is whether the *published* rule works, and
then whether searching adds anything a fixed default did not already have.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config                       # noqa: E402
from idxbot.data.cache import Cache                          # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV                     # noqa: E402
from idxbot.hullut import (                                  # noqa: E402
    Params, aggregate, expanding_folds, grid, prepare, run_universe,
    score_grid, walk_forward,
)

MIN_BARS = 2500          # ~10 years; anything shorter cannot span a full cycle
W = 86


def banner(title: str) -> None:
    print("\n" + "=" * W)
    print(f" {title}")
    print("=" * W)


def load_panel(verbose: bool = True) -> Dict[str, pd.DataFrame]:
    cfg = load_config()
    tickers = sorted(set(cfg.universe("bluechip"))
                     | set(cfg.universe("conglomerate"))
                     | set(cfg.universe("lq45")))
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    raw = loader.get_many(tickers, max_age=86400, verbose=False)
    panel = {t: prepare(d) for t, d in raw.items() if len(d) >= MIN_BARS}
    if verbose:
        span = pd.Series([len(d) for d in panel.values()])
        print(f"panel: {len(panel)} names with >= {MIN_BARS} bars "
              f"(median {span.median():.0f}, max {span.max()})")
    return panel


def cohorts(panel: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
    cfg = load_config()
    blue = set(cfg.universe("bluechip")) | set(cfg.universe("lq45"))
    cong = set(cfg.universe("conglomerate"))
    return {
        "all": panel,
        "blue chips": {t: d for t, d in panel.items() if t in blue},
        "conglomerate": {t: d for t, d in panel.items() if t in cong},
    }


VARIANTS = {
    "UT Bot alone": Params(entry="ut", exit="ut"),
    "Hull alone": Params(entry="hull", exit="hull"),
    "confluence (the method)": Params(entry="confluence", exit="either"),
    "confluence, UT exit only": Params(entry="confluence", exit="ut"),
    "confluence, Hull exit only": Params(entry="confluence", exit="hull"),
}


def _row(name: str, stats: Dict[str, float]) -> str:
    return (f" {name:<28}{stats['median_cagr']:>9.2%}"
            f"{stats['median_buy_hold_cagr']:>11.2%}"
            f"{stats['median_excess_cagr']:>10.2%}"
            f"{stats['beat_buy_hold']:>9.0%}"
            f"{stats['median_win_rate']:>8.0%}"
            f"{stats['median_max_dd']:>9.0%}"
            f"{stats['median_time_in_market']:>8.0%}")


def _header() -> str:
    return (f" {'variant':<28}{'CAGR':>9}{'buy&hold':>11}{'excess':>10}"
            f"{'beat':>9}{'win':>8}{'maxDD':>9}{'in mkt':>8}")


def stage_baseline(panel) -> None:
    banner("BASELINE — published defaults, nothing fitted")
    print(" Hull 55/HMA, UT key 1.0 / ATR 10. Costs: 0.15% buy, 0.25% sell,")
    print(" 0.10% slippage each way. Dividends accrue only while in position.")
    print(" 'buy&hold' is the same stock over the same window, total return.\n")
    for label, sub in cohorts(panel).items():
        print(f"--- {label} ({len(sub)} names) ---")
        print(_header())
        for name, params in VARIANTS.items():
            stats = aggregate(run_universe(sub, params))
            if stats:
                print(_row(name, stats))
        print()


def stage_costs(panel) -> None:
    banner("COSTS — how much is edge, how much is friction")
    print(" Same rule, same trades. Only the cost assumption moves.\n")
    params = Params()
    print(f" {'costs':<28}{'CAGR':>9}{'buy&hold':>11}{'excess':>10}{'beat':>9}"
          f"{'win':>8}")
    settings = [
        ("frictionless (unreal)", dict(fee_buy=0.0, fee_sell=0.0, slippage=0.0)),
        ("fees only, no slippage", dict(fee_buy=0.0015, fee_sell=0.0025, slippage=0.0)),
        ("realistic (default)", {}),
        ("wide spread (0.25% slip)", dict(slippage=0.0025)),
    ]
    for name, kwargs in settings:
        stats = aggregate(run_universe(panel, params, **kwargs))
        if stats:
            print(f" {name:<28}{stats['median_cagr']:>9.2%}"
                  f"{stats['median_buy_hold_cagr']:>11.2%}"
                  f"{stats['median_excess_cagr']:>10.2%}"
                  f"{stats['beat_buy_hold']:>9.0%}"
                  f"{stats['median_win_rate']:>8.0%}")
    print("\n If the rule only works frictionless, it is not a strategy.")


def stage_eras(panel) -> None:
    banner("ERAS — did it ever work, and does it still")
    eras = [("2001-2008", "2001-01-01", "2008-12-31"),
            ("2009-2014", "2009-01-01", "2014-12-31"),
            ("2015-2020", "2015-01-01", "2020-12-31"),
            ("2021-2026", "2021-01-01", "2026-12-31")]
    print(f"\n {'era':<12}{'variant':<26}{'CAGR':>9}{'buy&hold':>11}"
          f"{'excess':>10}{'beat':>8}{'names':>7}")
    for era, start, end in eras:
        for name in ("UT Bot alone", "Hull alone", "confluence (the method)"):
            per = run_universe(panel, VARIANTS[name], start=start, end=end)
            stats = aggregate(per)
            if stats:
                print(f" {era:<12}{name:<26}{stats['median_cagr']:>9.2%}"
                      f"{stats['median_buy_hold_cagr']:>11.2%}"
                      f"{stats['median_excess_cagr']:>10.2%}"
                      f"{stats['beat_buy_hold']:>8.0%}{stats['names']:>7.0f}")
        print()


def stage_grid(panel) -> None:
    banner("GRID — is there a plateau, or only a spike")
    candidates = grid()
    print(f" evaluating {len(candidates)} configurations over the full sample")
    print(" (this is IN-SAMPLE everywhere and proves nothing on its own -")
    print("  it exists to show the SHAPE of the surface)\n")
    table = score_grid(panel, candidates, verbose=True)
    if table.empty:
        print(" no results")
        return
    show = ["label", "median_cagr", "median_buy_hold_cagr", "median_excess_cagr",
            "beat_buy_hold", "median_win_rate"]
    print("\n best 10:")
    print(table[show].head(10).to_string(index=False,
          formatters={c: "{:.2%}".format for c in show[1:]}))
    print("\n worst 5:")
    print(table[show].tail(5).to_string(index=False,
          formatters={c: "{:.2%}".format for c in show[1:]}))
    best = table.iloc[0]["median_excess_cagr"]
    positive = (table["median_excess_cagr"] > 0).mean()
    print(f"\n configurations beating buy-and-hold at the median: {positive:.0%}")
    print(f" best in-sample excess CAGR: {best:+.2%}")
    print(" A single spike surrounded by losses is a fitted artefact.")
    print(" A broad plateau is the only shape worth believing.")
    table.drop(columns=["params"]).to_csv("reports/hullut_grid.csv", index=False)
    print(" -> reports/hullut_grid.csv")


def stage_walk(panel) -> None:
    banner("WALK-FORWARD — fit in-sample, score out-of-sample")
    dates = pd.DatetimeIndex(sorted({d for df in panel.values()
                                     for d in df["date"]}))
    folds = expanding_folds(dates, n_folds=5, min_train_years=8.0)
    if not folds:
        print(" not enough history")
        return
    candidates = grid()
    print(f" {len(candidates)} candidates, {len(folds)} folds\n")
    result = walk_forward(panel, candidates, folds, baseline=Params())
    if result.empty:
        print(" no folds produced results")
        return
    print()
    cols = ["fold", "chosen", "is_objective", "oos_objective",
            "baseline_objective", "optimisation_value_add"]
    print(result[cols].to_string(index=False, formatters={
        c: "{:+.2%}".format for c in cols[2:]}))
    print(f"\n mean out-of-sample excess CAGR, optimised : "
          f"{result['oos_objective'].mean():+.2%}")
    print(f" mean out-of-sample excess CAGR, fixed default: "
          f"{result['baseline_objective'].mean():+.2%}")
    print(f" value added by optimising                   : "
          f"{result['optimisation_value_add'].mean():+.2%}")
    # Signed and named explicitly: a NEGATIVE value here means out-of-sample
    # scored BETTER than in-sample, which is not "negative decay" but a regime
    # difference - the expanding training windows all contain 2009-2014, the
    # worst stretch for this rule, while the test slices sit after it.
    gap = result["oos_objective"].mean() - result["is_objective"].mean()
    print(f" out-of-sample minus in-sample               : {gap:+.2%}")
    print("   (positive = test windows were kinder than training, not an edge)")
    result.to_csv("reports/hullut_walkforward.csv", index=False)
    print(" -> reports/hullut_walkforward.csv")


def stage_broad(panel) -> None:
    """The survivorship question, and the one argument left for trend following.

    Every cohort above is *today's* index membership, so buy-and-hold is being
    measured only on the names that made it. That flatters the benchmark, and
    it flatters it in exactly the place a trend system claims to earn its
    keep: the stocks that fell 90% and never came back, which a stop would have
    exited and a buy-and-holder would have ridden down.

    So the fair test is a universe that is not filtered by survival. This runs
    the whole liquid exchange instead of the winners' list. If the method has a
    defensible purpose, the gap to buy-and-hold must narrow here.
    """
    banner("SURVIVORSHIP — the same rule on the whole exchange, not the winners")
    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    everything = cfg.universe("idx_all")
    raw = loader.get_many(everything, max_age=86400 * 30, verbose=False)
    broad = {t: prepare(d) for t, d in raw.items() if len(d) >= MIN_BARS}
    print(f" broad panel: {len(broad)} names with >= {MIN_BARS} bars"
          f"  (vs {len(panel)} in the curated cohorts)\n")

    print(_header())
    for name in ("UT Bot alone", "Hull alone", "confluence (the method)"):
        stats = aggregate(run_universe(broad, VARIANTS[name]))
        if stats:
            print(_row(name, stats))

    per = run_universe(broad, VARIANTS["confluence (the method)"])
    if per.empty:
        return
    print("\n Split by what buy-and-hold did, because that is the whole claim:")
    print(f" {'buy&hold outcome':<28}{'names':>7}{'strategy':>11}{'buy&hold':>11}"
          f"{'excess':>10}{'beat':>8}")
    # These are ANNUALISED, not total. Labelling a -4.8%/yr cohort "lost 0-50%"
    # reads as a total drawdown and understates it: over 15 years that is a
    # name that lost more than half its value.
    buckets = [("fell > 10%/yr", -1.0, -0.10), ("fell 0-10%/yr", -0.10, 0.0),
               ("gained 0-15%/yr", 0.0, 0.15), ("gained > 15%/yr", 0.15, 9.9)]
    for label, lo, hi in buckets:
        sub = per[(per["buy_hold_cagr"] > lo) & (per["buy_hold_cagr"] <= hi)]
        if sub.empty:
            continue
        print(f" {label:<28}{len(sub):>7}{sub['cagr'].median():>11.2%}"
              f"{sub['buy_hold_cagr'].median():>11.2%}"
              f"{sub['excess_cagr'].median():>10.2%}"
              f"{(sub['excess_cagr'] > 0).mean():>8.0%}")
    print("\n If trend following has a purpose it is the falling rows: cutting")
    print(" the losers. That is where to look, not at the headline average.")


STAGES = {"baseline": stage_baseline, "costs": stage_costs, "eras": stage_eras,
          "grid": stage_grid, "walk": stage_walk, "broad": stage_broad}


def main() -> int:
    wanted = sys.argv[1:] or ["baseline", "costs", "eras"]
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")
    os.makedirs("reports", exist_ok=True)
    panel = load_panel()
    for stage in wanted:
        STAGES[stage](panel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
