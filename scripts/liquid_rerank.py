#!/usr/bin/env python3
"""Re-rank the multiplier entry INSIDE the liquid tercile and price it there.

    python3 scripts/liquid_rerank.py            # build (~10 min), then report
    python3 scripts/liquid_rerank.py --report   # reuse the cached table

WHY THIS EXISTS. H21's C3d asked whether the picks' 2.5%/yr shortfall against
the IHSG is a small-cap handicap rather than a bad rule. Splitting the existing
per-name table three ways could not answer it: a twelve-name basket cut into
terciles leaves the liquid cell scoring 54 of 212 cohorts at a median of four
names, and a first draft of that section printed −13.9% for the cell — the
smallest cell producing the largest effect, which is the degenerate-cell trap
this repo has recorded twice already.

The memo priced the proper answer at "about twenty minutes of rebuild, not a
limit of the data". Naming a cost and then not paying it is the premature
closure `docs/STANDING_ORDERS.md` exists to stop, so this pays it: the rule is
re-run with its universe restricted to the liquid tercile BEFORE ranking, so it
selects a full twelve-name basket from that segment on every cohort.

WHAT WOULD CHANGE THE CONCLUSION. If the rule beats the index here, the
shortfall was the segment and the rule is fine applied upmarket. If it does not,
the size explanation is dead and the shortfall is the selection itself. The
random-draw table in C3d already says the liquid tercile of this pool is the
WORST of the three against the index (−9.5% against the thin end's −3.5%), so
the prediction registered here, before scoring, is that **the rule will NOT beat
the index in the liquid tercile** — this is a confirmatory run of a rescue
expected to fail, not a search for a cell where it works.

NO NEW TIE-BREAK RISK. `MU.select(tie="all")` is used exactly as elsewhere, so
a tied cut takes the whole tied group rather than whatever `sort_values` left on
top (H17b).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.spine import exits as X                              # noqa: E402
from idxbot.spine import multiplier as MU                        # noqa: E402
from portfolio_critique import (agg, bench_slots,                # noqa: E402
                                index_series, yield_by_liquidity)
from portfolio_sim import baskets, slots                         # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")
CACHE = os.path.join("data", "spine", "liquid_rerank.parquet")
SLOTS = 12
DRAWS = 20

#: Which liquidity tercile to re-rank inside. 2 is the most liquid third of the
#: eligible pool on each cohort date — the segment closest to where the index
#: lives, which is the whole point of the test.
TERCILE = 2
#: Names to price per cohort beyond the basket, forming the random control.
POOL = 120


def liquid_universe(P: pd.DataFrame, day: pd.Timestamp,
                    M: pd.DataFrame) -> List[str]:
    """The names in `M` that sit in the top liquidity tercile ON `day`.

    Liquidity is read at the cohort date and nowhere later, so this cannot see
    past the decision bar (A5). The tercile is taken WITHIN the cohort's own
    cross-section, so it ranks names against their contemporaries rather than
    against a twenty-year trend in market-wide turnover.
    """
    row = P[(P["date"] == day) & P["ticker"].isin(set(M["ticker"]))]
    row = row.dropna(subset=["log_turnover"])
    if len(row) < 30:
        return []
    cut = row["log_turnover"].quantile(TERCILE / 3.0)
    return list(row.loc[row["log_turnover"] >= cut, "ticker"])


def build(P: pd.DataFrame, start="2002-01-01") -> pd.DataFrame:
    """Per-name buy-and-hold return, liquid tercile only, re-ranked inside it."""
    cells, tab = MU.build_cells(P)
    pre_end = P.loc[~P["holdout"].astype(bool), "date"].max()
    last = pre_end - pd.Timedelta(days=int(X.HORIZON * 1.5))
    hold = X.catalogue()["hold 252"]
    rng = np.random.default_rng(20260825)
    rows: List[Dict] = []

    for d in pd.date_range(start, last, freq="MS"):
        day, M = MU.rank_live(P, d, cells, tab)
        if day is None or len(M) < MU.TOP_N:
            continue
        keep = liquid_universe(P, day, M)
        if len(keep) < MU.TOP_N * 2:
            continue
        #  RE-RANK: restrict first, then select, so the basket is a full one
        #  drawn from this segment rather than the segment's share of a basket
        #  chosen elsewhere.
        Ml = M[M["ticker"].isin(keep)]
        picked = set(MU.select(Ml, MU.TOP_N, "all")["ticker"])
        pool = list(Ml["ticker"])
        if len(pool) > POOL:
            rest = [t for t in pool if t not in picked]
            pool = list(picked) + list(rng.choice(
                rest, size=max(POOL - len(picked), 0), replace=False))
        for t, (path, cost) in MU.path_map(P, day, pool).items():
            r, held = hold(path)
            rows.append({"as_of": day, "ticker": t, "picked": t in picked,
                         "cost": cost, "hold 252": r - cost,
                         "hold 252|held": held})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    if a.report and os.path.exists(CACHE):
        D = pd.read_parquet(CACHE)
        D["as_of"] = pd.to_datetime(D["as_of"])
        print(f" [reusing {CACHE}]")
    else:
        P = pd.read_parquet(PANEL)
        P["date"] = pd.to_datetime(P["date"])
        P = P.sort_values(["ticker", "date"])
        D = build(P)
        D.to_parquet(CACHE, index=False)
        print(f" [cached to {CACHE}]")

    W = 96
    print("=" * W)
    print(" H21 C3d(b) — THE MULTIPLIER RULE RE-RANKED INSIDE THE LIQUID"
          " TERCILE")
    print("=" * W)
    print(f" {D['as_of'].nunique()} cohorts, {len(D):,} priced name-cohorts, "
          f"{D['ticker'].nunique()} names, "
          f"{int(D['picked'].sum()):,} picks")

    B = baskets(D, "hold 252", "picked")
    if B.empty:
        print(" no scoreable cohort — nothing to report")
        return 1
    print(f" basket size: median {B['n'].median():.0f}, "
          f"min {B['n'].min():.0f}, max {B['n'].max():.0f}"
          f"   ({len(B)}/{D['as_of'].nunique()} cohorts scored)")
    if B["n"].median() < 8:
        print(" !! baskets are still thin — this does NOT answer the question")

    S = index_series()
    Y = yield_by_liquidity(D["as_of"].min(),
                           D["as_of"].max() + pd.Timedelta(days=400))
    top = float(Y.loc[Y.index >= 8, "yld"].mean())

    Sp = slots(B, SLOTS)
    rr = [agg(slots(baskets(D, "hold 252", "random", size=12, seed=1000 + k),
                    SLOTS)) for k in range(DRAWS)]
    rn = float(np.mean([r[0] for r in rr]))
    Bs = bench_slots(S, Sp, top).dropna(subset=["d"])
    x = Bs["d"].to_numpy()
    m, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    se = sd / np.sqrt(len(x))

    print(f"\n   re-ranked picks CAGR          {agg(Sp)[0]:>+8.1%}")
    print(f"   random draw, same universe    {rn:>+8.1%}")
    print(f"   -> edge over its own segment  {agg(Sp)[0] - rn:>+8.1%}")
    print(f"\n   index total return            {Bs['index'].mean():>+8.1%}")
    print(f"   picks minus index, per slot   {m:>+8.2%}"
          f"   [{m - 1.96 * se:+.2%}, {m + 1.96 * se:+.2%}]")
    print(f"   slots beating the index       {int((x > 0).sum())}/{len(x)}")

    print("\n" + "=" * W)
    print(" VERDICT ON THE SIZE RESCUE")
    print("=" * W)
    if m > 0 and (m - 1.96 * se) > 0:
        print(" The rule DOES beat the index in the liquid tercile. The 2.5%")
        print(" shortfall was the segment, and H21's conclusion needs revising")
        print(" to say so.")
    elif m > 0:
        print(" Positive but inside its own interval — not established.")
    else:
        print(" The rule does NOT beat the index in the liquid tercile either.")
        print(" The size explanation is dead: the shortfall is the selection,")
        print(" and applying the same rule upmarket does not repair it. This")
        print(" was the registered prediction, so it is a confirmation and not")
        print(" a discovery.")
    print("\n Twelve overlapping slots over one history are not twelve")
    print(" independent trials; the interval is the FAVOURABLE reading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
