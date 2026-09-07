#!/usr/bin/env python3
"""H59 — sector rotation (§8), on a map that has been on disk since A14 unused.

WHAT IS BEING TESTED AND WHAT IS NOT. §8 asks for a sector-rotation engine.
Before any engine there is a prior question this repo has never asked: does the
IDX-IC sector map carry anything, and can it be used at all without a
look-ahead? Two properties of the file decide that, and both are measured here
rather than assumed.

  * IT IS FROZEN AT 2024-07-10, so it names only companies listed on that day.
    Every delisted name in this panel therefore has NO sector, and a study run
    on the covered names alone is survivorship-biased by construction. A14
    recorded the freeze; nobody measured what it costs. §2 of the coverage
    report below is the measurement, and if delisted coverage is near zero the
    universe restriction has to be stated with every number.

  * IDX-IC ITSELF LAUNCHED 2021-01-25, replacing JASICA. A sector label applied
    to a 2010 bar is a backward projection of a taxonomy that did not exist.
    That is NOT look-ahead in H57's sense -- a company's industry does not
    respond to its own returns -- but it is a real limit and it is stated with
    the result rather than in a footnote.

`shares` is deliberately NOT used, per A25: applying a 2024 share count to a
2010 bar is look-ahead, and `scripts/shares_pit.py` is where a real one lives.

--------------------------------------------------------------------- REGISTERED
Written before any cell was scored.

  R1  Sector momentum persists cross-sectionally: a sector's trailing 12-1
      return ranks its next-quarter return. PREDICTION: a positive rank IC that
      clears its clustered null. This is the weakest form of §8's claim and the
      one with the most prior support in the literature.

  R2  PREDICTED NULL. Adding a sector tilt to H26's strength+calm screen does
      NOT improve it against `bhbench`'s benchmarks. H27 measured sector as
      worth +0.15 of skew and COSTING 0.0098 of mean log -- marginal and
      two-signed. PREDICTION: confirmed; no arm beats the plain screen in both
      halves.

  R3  The binding constraint is the width of the cross-section, not the size of
      the effect. Eleven sectors is not a population. PREDICTION: even a
      respectable IC fails this repo's Bonferroni bar, and the power statement
      says so before the IC does.
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

from bhbench import Bench, load                                  # noqa: E402
from beathold import ew, strcalm                                 # noqa: E402
from idxbot.metrics import periods_to_detect                     # noqa: E402

MAP = os.path.join("data", "reference", "idx_classification.parquet")
IDXIC_LAUNCH = pd.Timestamp("2021-01-25")
MIN_NAMES = 5          # a "sector" below this is a name wearing a label
TRIALS = 354           # hypotheses.md after H58
BAR = 0.05 / TRIALS


def sector_map(path: str = MAP) -> pd.DataFrame:
    S = pd.read_parquet(path)
    #  `shares` is dropped HERE rather than merely unused, so no downstream
    #  edit can reach for it by accident. A25: a 2024 count on a 2010 bar is
    #  look-ahead and Indonesian rights issues are what make it wrong.
    return S[["ticker", "sector", "board", "listing_date"]].dropna(
        subset=["sector"])


def coverage(P: pd.DataFrame, S: pd.DataFrame) -> Dict[str, float]:
    """What the 2024 freeze costs, measured rather than assumed."""
    last = P.groupby("ticker")["date"].max()
    alive = last >= P["date"].max() - pd.Timedelta(days=30)
    have = set(S["ticker"])
    names = pd.DataFrame({"alive": alive})
    names["mapped"] = [t in have for t in names.index]
    liveN = int(names["alive"].sum())
    deadN = int((~names["alive"]).sum())
    return {
        "names": float(len(names)),
        "mapped": float(names["mapped"].sum()),
        "live": float(liveN),
        "live_mapped": float(names.loc[names["alive"], "mapped"].sum()),
        "dead": float(deadN),
        "dead_mapped": float(names.loc[~names["alive"], "mapped"].sum()),
        "bars": float(len(P)),
        "bars_mapped": float(P["ticker"].isin(have).sum()),
    }


def sector_panel(P: pd.DataFrame, S: pd.DataFrame, freq: int = 63
                 ) -> pd.DataFrame:
    """Equal-weighted sector index returns, per rebalance mark.

    EQUAL WEIGHTED AND NOT CAP WEIGHTED, and the reason is not a preference.
    A19 established that an equal-weighted IDX basket structurally trails the
    cap-weighted index, so a cap-weighted sector index would be a different
    animal from the baskets this repo actually builds -- and a point-in-time
    cap only exists for 2019-2025 (H57). Equal weighting is what the strategy
    arms use, so it is what the signal must be measured on.
    """
    F = P[P["elig"]].merge(S[["ticker", "sector"]], on="ticker", how="inner")
    F = F.sort_values(["ticker", "date"])
    #  Per-name returns computed WITHIN ticker. A11: never roll on a pivot.
    F["r1"] = F.groupby("ticker")["adj_close"].pct_change()
    dates = np.sort(F["date"].unique())
    marks = dates[::freq]
    F = F[F["date"].isin(marks)].copy()
    #  Trailing 12-1 momentum and forward return, both per name, both stepped
    #  in mark units so a sector's history is on the rebalance clock.
    k12 = max(int(round(252 / freq)), 2)
    k1 = 1
    g = F.groupby("ticker")["adj_close"]
    F["mom"] = g.shift(k1) / g.shift(k12) - 1.0
    F["fwd"] = g.shift(-1) / F["adj_close"] - 1.0
    A = (F.dropna(subset=["sector"])
         .groupby(["date", "sector"])
         .agg(mom=("mom", "mean"), fwd=("fwd", "mean"), n=("ticker", "size"))
         .reset_index())
    return A[A["n"] >= MIN_NAMES].dropna(subset=["mom", "fwd"])


def rank_ic(A: pd.DataFrame) -> float:
    """Mean per-date Spearman between sector momentum and forward return."""
    out = []
    for _d, g in A.groupby("date"):
        if len(g) < 4:
            continue
        c = g[["mom", "fwd"]].corr(method="spearman").iloc[0, 1]
        if np.isfinite(c):
            out.append(c)
    return float(np.mean(out)) if out else float("nan")


def ic_series(A: pd.DataFrame) -> np.ndarray:
    out = []
    for _d, g in A.groupby("date"):
        if len(g) < 4:
            continue
        c = g[["mom", "fwd"]].corr(method="spearman").iloc[0, 1]
        if np.isfinite(c):
            out.append(c)
    return np.asarray(out, dtype=float)


def date_block_null(A: pd.DataFrame, draws: int = 500, seed: int = 0) -> tuple:
    """Reassign whole DATES' forward returns to other dates.

    THE UNIT IS THE DATE, NOT THE ROW, AND WITH ELEVEN SECTORS IT HAS TO BE.
    A row shuffle inside a date would still leave every sector's forward return
    drawn from the same day, which is most of the dependence -- eleven sectors
    on one day share the market's move almost entirely. Moving a whole date's
    vector of forward returns onto another date's momentum destroys the mapping
    while preserving both the cross-sectional and the time-series structure.
    A17 and A25 record the two ways to get this wrong; this is the same fix one
    level up.
    """
    rng = np.random.default_rng(seed)
    dates = np.sort(A["date"].unique())
    byd = {d: g.sort_values("sector") for d, g in A.groupby("date")}
    out = []
    for _ in range(draws):
        perm = rng.permutation(len(dates))
        vals = []
        for i, d in enumerate(dates):
            src = byd[dates[perm[i]]]
            dst = byd[d]
            if len(dst) < 4:
                continue
            f = src["fwd"].to_numpy()
            if len(f) < len(dst):
                f = np.tile(f, int(np.ceil(len(dst) / max(len(f), 1))))
            m = dst["mom"].to_numpy()
            y = f[:len(dst)]
            #  A constant vector has no rank correlation; scipy warns and
            #  returns NaN. Skipping it is right, but it must be SKIPPED
            #  rather than silently coerced to zero -- a zero would pull the
            #  null toward the origin and make every z too large.
            if len(np.unique(m)) < 2 or len(np.unique(y)) < 2:
                continue
            c = pd.Series(m).corr(pd.Series(y), method="spearman")
            if np.isfinite(c):
                vals.append(c)
        if vals:
            out.append(float(np.mean(vals)))
    a = np.asarray(out)
    return (float(a.mean()), float(a.std(ddof=1)), len(a)) if len(a) > 1 \
        else (float("nan"), float("nan"), len(a))


def sector_tilt(S: pd.DataFrame, top: int, k: int = 10,
                invert: bool = False, random_sectors: bool = False):
    """H26's screen restricted to the `top` sectors by trailing momentum.

    The random-sector arm is the control that makes the result readable: it
    restricts to the SAME NUMBER of sectors chosen at random, so the comparison
    is selection rather than concentration. A34 records a control denied the
    treatment's own affordances as a handicap, not a null.
    """
    smap = dict(zip(S["ticker"], S["sector"]))
    base = strcalm(k)
    rng = np.random.default_rng(0)

    def f(day: pd.DataFrame) -> List:
        d = day.copy()
        d["sector"] = d["ticker"].map(smap)
        d = d.dropna(subset=["sector", "mom12_1", "hi52", "vol60"])
        if len(d) < 3 * k:
            return []
        agg = d.groupby("sector")["mom12_1"].mean()
        agg = agg[d.groupby("sector").size() >= MIN_NAMES]
        if len(agg) < top + 1:
            return []
        if random_sectors:
            keep = list(rng.choice(agg.index.to_numpy(), top, replace=False))
        else:
            keep = list(agg.nsmallest(top).index if invert
                        else agg.nlargest(top).index)
        sub = d[d["sector"].isin(keep)]
        if len(sub) < 2 * k:
            return []
        return base(sub)
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--freq", type=int, default=63)
    ap.add_argument("--phases", type=int, default=6)
    ap.add_argument("--skip-arms", action="store_true")
    a = ap.parse_args()

    P = load()
    S = sector_map()
    print("H59 — sector rotation (§8)")
    print(f"map: {len(S)} names, {S['sector'].nunique()} IDX-IC sectors, "
          f"frozen 2024-07-10")

    # ------------------------------------------------------------ coverage
    c = coverage(P, S)
    print("\n1. WHAT THE 2024 FREEZE COSTS")
    print(f"   panel names          {c['names']:,.0f}")
    print(f"   mapped               {c['mapped']:,.0f} "
          f"({c['mapped'] / c['names']:.1%})")
    print(f"   still trading        {c['live']:,.0f}, of which mapped "
          f"{c['live_mapped']:,.0f} ({c['live_mapped'] / max(c['live'], 1):.1%})")
    print(f"   no longer trading    {c['dead']:,.0f}, of which mapped "
          f"{c['dead_mapped']:,.0f} ({c['dead_mapped'] / max(c['dead'], 1):.1%})")
    print(f"   bars covered         {c['bars_mapped']:,.0f} of {c['bars']:,.0f} "
          f"({c['bars_mapped'] / c['bars']:.1%})")
    dead_cov = c["dead_mapped"] / max(c["dead"], 1)
    if dead_cov < 0.5:
        print(f"\n   *** SURVIVORSHIP: only {dead_cov:.1%} of dead names carry a")
        print("   sector, so every number below is computed on a universe that")
        print("   is tilted toward survivors. This is a property of the FILE,")
        print("   not of the method, and no permutation null can repair it.")

    # ------------------------------------------------------------------ R1
    A = sector_panel(P, S, a.freq)
    ics = ic_series(A)
    ic = float(np.mean(ics)) if len(ics) else float("nan")
    print(f"\n2. R1 — does sector momentum rank next-quarter sector return?")
    print(f"   {A['date'].nunique()} marks, {A['sector'].nunique()} sectors, "
          f"{len(A):,} sector-marks, "
          f"{A['date'].min().date()} -> {A['date'].max().date()}")
    print(f"   pre-IDX-IC marks (a backward projection of a 2021 taxonomy): "
          f"{int((A['date'] < IDXIC_LAUNCH).sum())} of {len(A)}")
    mu, sd, nd = date_block_null(A, draws=a.draws)
    z = (ic - mu) / sd if sd and sd > 0 else float("nan")
    print(f"   rank IC {ic:+.4f}   null {mu:+.4f} +- {sd:.4f} ({nd} draws)   "
          f"z {z:+.2f}")
    #  R3's power statement comes BEFORE the verdict, per A31/A19.
    need = periods_to_detect(ics)
    print(f"   POWER: {need:,.0f} marks to tell this IC from zero at |t|=2; "
          f"{len(ics)} observed")
    mid = A["date"].quantile(0.5)
    e = rank_ic(A[A["date"] <= mid])
    l = rank_ic(A[A["date"] > mid])
    print(f"   half-split: early {e:+.4f}   late {l:+.4f}   "
          f"{'same sign' if np.sign(e) == np.sign(l) else 'SIGN FLIPS'}")
    p_emp = float(np.nan)
    print(f"   Bonferroni bar at {TRIALS} trials: {BAR:.5f}; "
          f"z {z:+.2f} {'clears' if abs(z) > 3.63 else 'does NOT clear'} it")

    # ------------------------------------------------------------------ R2
    if a.skip_arms:
        print("\n3. R2 — skipped (--skip-arms)")
        return
    print("\n3. R2 — does a sector tilt improve H26's screen?")
    B = Bench(P)
    arms = [("strength+calm, no tilt", strcalm(10)),
            ("+ top 3 sectors by momentum", sector_tilt(S, 3)),
            ("+ top 5 sectors by momentum", sector_tilt(S, 5)),
            ("+ BOTTOM 3 sectors (inverted)", sector_tilt(S, 3, invert=True)),
            ("+ 3 RANDOM sectors (control)",
             sector_tilt(S, 3, random_sectors=True))]
    #  THE ARMS MUST SHARE A WINDOW OR THEIR CAGRs ARE NOT COMPARABLE.
    #  A tilt arm cannot trade until enough sectors carry MIN_NAMES, so it
    #  starts later, and a first run printed the no-tilt arm at +12.01% against
    #  a top-3 tilt at +6.65% -- with their INDEX benchmarks at +8.71% and
    #  +4.63%. A benchmark that moves four points between arms is proof the
    #  windows differ; the difference was the calendar, not the tilt. A19
    #  records this error class three times and it is committed here a fourth.
    #  Two guards: a scout pass fixes one window for every arm, and the excess
    #  over each arm's OWN index is printed beside the raw CAGR.
    scout = []
    for label, sel in arms:
        if hasattr(sel, "reset"):
            sel.reset()
        r = B.walk(sel, freq=a.freq, offset=0)
        if r:
            scout.append((r["start"], r["end"]))
    if not scout:
        print("   no arm produced a valid path")
        return
    lo = max(s for s, _e in scout)
    hi = min(e for _s, e in scout)
    print(f"   common window {pd.Timestamp(lo).date()} -> "
          f"{pd.Timestamp(hi).date()} "
          f"(scouted across all {len(scout)} arms)")

    rows = []
    for label, sel in arms:
        v = B.evaluate(sel, label=label, freq=a.freq, phases=a.phases,
                       lo=lo, hi=hi)
        if not v.get("ok"):
            print(f"   {label:<32} FAILED TO RUN — {v.get('why')}")
            continue
        v["excess"] = v["cagr"] - v["bh_index"]
        rows.append(v)
        print(f"   {label:<32} CAGR {v['cagr']:+.2%}  index "
              f"{v['bh_index']:+.2%}  excess {v['excess']:+.2%}  "
              f"beats index {v.get('beats_index_n', '?')}/{a.phases}")
    #  THE MATCHED COMPARISON IS THE ONE THAT ANSWERS §8, AND IT IS NOT
    #  TILT-VERSUS-NO-TILT. Restricting to three sectors concentrates the book
    #  whatever chooses them, so a top-3 arm losing to the untilted screen
    #  measures CONCENTRATION. Top-3 against RANDOM-3 holds concentration fixed
    #  and varies only the choosing, which is what §8 actually claims.
    by = {r["label"]: r for r in rows}
    t3 = by.get("+ top 3 sectors by momentum")
    r3 = by.get("+ 3 RANDOM sectors (control)")
    b3 = by.get("+ BOTTOM 3 sectors (inverted)")
    if t3 and r3:
        print(f"\n   MATCHED AT 3 SECTORS (concentration held fixed):")
        print(f"     momentum-chosen {t3['excess']:+.2%} excess   "
              f"random-chosen {r3['excess']:+.2%}   "
              f"difference {t3['excess'] - r3['excess']:+.2%}")
        if b3:
            print(f"     bottom-chosen   {b3['excess']:+.2%}   "
                  f"top minus bottom {t3['excess'] - b3['excess']:+.2%}")
        print("     This is the only cell in the table that varies the sector")
        print("     CHOICE while holding the number of sectors constant.")

    if len(rows) >= 2:
        base = rows[0]
        print("\n   R2 verdict: ", end="")
        #  Compared on EXCESS, not raw CAGR: even inside a common window the
        #  arms differ in how much of it they are actually invested for.
        better = [r for r in rows[1:] if r["excess"] > base["excess"]]
        ctl = next((r for r in rows if "RANDOM" in r["label"]), None)
        if not better:
            print("no tilt arm beats the plain screen on excess over its own")
            print("   index. PREDICTED NULL CONFIRMED.")
        else:
            names = ", ".join(r["label"] for r in better)
            print(f"{len(better)} arm(s) beat the plain screen: {names}.")
            if ctl is not None and ctl["excess"] > base["excess"]:
                print("   AND SO DOES THE RANDOM-SECTOR CONTROL, so what is "
                      "being measured is")
                print("   CONCENTRATION, not sector selection. A38 records "
                      "exactly this tell:")
                print("   a selection effect cannot lift the control, so a "
                      "shift in both arms")
                print("   is an accounting artefact by construction.")
            else:
                print("   The random-sector control does NOT beat it, so the "
                      "lift is not")
                print("   merely concentration. Still one calendar and one "
                      "in-sample panel.")


if __name__ == "__main__":
    main()
