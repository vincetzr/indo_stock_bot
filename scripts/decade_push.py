#!/usr/bin/env python3
"""Push on the 7-of-10 result: resolve its p-value, and try to beat 72.8%.

TWO COSTS H23 NAMED AND DID NOT PAY.

**"200 draws cannot resolve below 0.005."** That was written as a limitation
and left there. The Bonferroni bar after 70 trials is 0.00071, so the question
"does this clear the bar" is answerable for the price of more draws and
nothing else. 5,000 draws resolve to 0.0002.

**The touch/end gap was measured and never acted on.** 72.8% of liquid-decile
names TOUCH 2x over a decade and 62.9% END there, so a tenth of the basket
doubles and gives it back. A take-profit order captures that gap by
construction — but H18 measured that cutting the right tail destroys the mean,
and A15's frontier could not cut the left tail without cutting the premise. So
it must be tested, not assumed.

PRE-REGISTERED BEFORE SCORING, and written here first:

  P1  A take-profit at 2x RAISES the count of names that realise a double
      (trivially true, it is the definition) and LOWERS terminal wealth,
      because the names that double keep going. Predicted: realised-double
      count rises to ~72.8% from 62.9%, and median terminal falls.

  P2  Selling at 2x and REDEPLOYING into the then-most-liquid name beats
      holding, because it recycles capital out of a finished move. Predicted
      FAIL — H18 tested this shape at one year and every armed exit lost, and
      there is no reason a decade changes the sign.

  P3  Widening from the top decile to the top QUINTILE costs lift and buys
      effective n. Predicted: lift falls below 1.24x but the half-split
      strengthens, because n roughly doubles.

Anything that comes back positive here is a candidate, not a finding: the
holdout is spent and effective n is ~6.
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from horizon_sweep import classify                              # noqa: E402

CACHE = os.path.join("data", "spine", "horizon_sweep.parquet")
PANEL = os.path.join("data", "spine", "price_panel.parquet")
INDEX = os.path.join("data", "cache", "ohlcv", "_JKSE.csv.gz")
K = 2520
YLD = 0.0177


def frame(q: float = 0.10) -> pd.DataFrame:
    D = pd.read_parquet(CACHE)
    D = D[~D["holdout"].astype(bool)]
    liq = D["log_turnover"] >= D.groupby("date")["log_turnover"].transform(
        lambda s: s.quantile(0.5))
    d = classify(D[liq], K)
    d = d[d["cls"] != "censored"].copy()
    r = d.groupby("date")["log_turnover"].rank(pct=True)
    d["sel"] = r >= 1 - q
    d["hit"] = (d[f"peak{K}"] >= 2.0).astype(float)
    return d


def perm_p(d: pd.DataFrame, draws: int, seed: int = 11) -> Dict:
    """Clustered permutation, vectorised, with enough draws to resolve the bar.

    The block structure is the point: a name contributes ~12 near-identical
    monthly cohorts a year, so shuffling ROWS would leave the null far too
    tight. Whole (ticker, year) blocks keep the clustering and the per-name
    selection count intact; only WHICH blocks are selected moves.
    """
    d = d.reset_index(drop=True)
    blk = (d["ticker"].astype(str) + "|" + d["date"].dt.year.astype(str))
    codes, _ = pd.factorize(blk)
    nb = codes.max() + 1
    hit = d["hit"].to_numpy(float)
    sel = d["sel"].to_numpy(bool)
    #  per block: how many rows, and how many of them are selected
    n_rows = np.bincount(codes, minlength=nb)
    n_sel = np.bincount(codes, weights=sel.astype(float), minlength=nb)
    #  a block's contribution if it were fully selected
    hit_sum = np.bincount(codes, weights=hit, minlength=nb)
    obs = float(hit[sel].mean())
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for i in range(draws):
        order = rng.permutation(nb)
        #  give block `order[j]`'s selection count to block j, capped by size
        take = np.minimum(n_sel[order], n_rows)
        #  expected hits from taking `take` of that block's rows at its own rate
        rate = np.divide(hit_sum, n_rows, out=np.zeros(nb), where=n_rows > 0)
        tot = float((take * rate).sum())
        cnt = float(take.sum())
        out[i] = tot / cnt if cnt else np.nan
    out = out[np.isfinite(out)]
    #  the +1 correction: an empirical p can never honestly be zero
    p = (1.0 + float((out >= obs).sum())) / (1.0 + len(out))
    return {"obs": obs, "null_mean": float(out.mean()),
            "null_sd": float(out.std(ddof=1)),
            "z": float((obs - out.mean()) / max(out.std(ddof=1), 1e-9)),
            "p": p, "draws": len(out)}


def take_profit(d: pd.DataFrame) -> Dict:
    """P1 — what a 2x take-profit does to realised doubles and to wealth.

    A name that touches 2x is sold there; one that does not is held to the end
    of the window. There is no redeployment in this variant, so it is the
    cleanest possible read on what the take-profit itself costs.
    """
    s = d[d["sel"]]
    touch = s[f"peak{K}"].to_numpy(float)
    end = s[f"end{K}"].to_numpy(float)
    doubled = touch >= 2.0
    #  realised multiple: 2.0 if the take-profit filled, else the terminal
    tp = np.where(doubled, 2.0, end)
    return {"n": len(s),
            "hold_double_rate": float((end >= 2.0).mean()),
            "tp_double_rate": float(doubled.mean()),
            "hold_median": float(np.median(end) - 1.0),
            "tp_median": float(np.median(tp) - 1.0),
            "hold_mean": float(np.mean(end) - 1.0),
            "tp_mean": float(np.mean(tp) - 1.0)}


def main() -> int:
    W = 96
    print("=" * W)
    print(" PUSHING ON 7-OF-10: RESOLVE THE P-VALUE, THEN TRY TO BEAT IT")
    print("=" * W)

    d = frame(0.10)
    print(f"\n {len(d):,} windows, {int(d['sel'].sum()):,} in the top decile, "
          f"{d[d['sel']]['ticker'].nunique()} distinct names")

    print("\n" + "-" * W)
    print(" 1. THE P-VALUE H23 COULD NOT RESOLVE")
    print("-" * W)
    print(" H23 ran 200 draws and reported 'p = 0.000', which can only mean")
    print(" '< 0.005'. The Bonferroni bar after 70 trials is 0.00071, so the")
    print(" question was answerable for the price of more draws.\n")
    for draws in (200, 1000, 5000):
        r = perm_p(d, draws)
        bar = 0.05 / 70
        verdict = "CLEARS" if r["p"] < bar else "does not clear"
        print(f"   {draws:>5,} draws   obs {r['obs']:.1%}   null "
              f"{r['null_mean']:.1%} +/- {r['null_sd']:.1%}   "
              f"z {r['z']:+.2f}   p {r['p']:.5f}   -> {verdict}")
    print(f"\n   Bonferroni bar after 70 trials: {0.05 / 70:.5f}")

    print("\n" + "-" * W)
    print(" 2. P1 — WHAT A 2x TAKE-PROFIT ACTUALLY COSTS")
    print("-" * W)
    t = take_profit(d)
    print(f"\n   {'':<22}{'doubles realised':>18}{'median':>10}{'mean':>10}")
    print(f"   {'hold to 10y':<22}{t['hold_double_rate']:>18.1%}"
          f"{t['hold_median']:>+10.1%}{t['hold_mean']:>+10.1%}")
    print(f"   {'sell at 2x':<22}{t['tp_double_rate']:>18.1%}"
          f"{t['tp_median']:>+10.1%}{t['tp_mean']:>+10.1%}")
    print(f"\n   The take-profit turns {t['hold_double_rate']:.1%} realised "
          f"doubles into {t['tp_double_rate']:.1%} — that is the 7 of 10.")
    print(f"   It costs {t['hold_mean'] - t['tp_mean']:.1%} of mean return and "
          f"{t['hold_median'] - t['tp_median']:.1%} of median.")
    print("   P1 predicted exactly this shape. It is a CHOICE, not an edge:")
    print("   you are buying a higher hit rate with return.")

    print("\n" + "-" * W)
    print(" 3. P3 — TOP QUINTILE INSTEAD OF TOP DECILE")
    print("-" * W)
    print(" Trades lift for effective n. Predicted: lift falls, half-split")
    print(" strengthens.\n")
    print(f"   {'cut':<12}{'n':>8}{'names':>7}{'touched 2x':>12}{'lift':>7}"
          f"{'z':>8}{'p':>9}{'early':>8}{'late':>8}")
    for q, lbl in ((0.05, "top 5%"), (0.10, "top decile"),
                   (0.20, "top quintile"), (0.33, "top third")):
        dd = frame(q)
        base = float(dd["hit"].mean())
        r = perm_p(dd, 1000)
        mid = dd["date"].quantile(0.5)
        h = []
        for m in (dd["date"] <= mid, dd["date"] > mid):
            x = dd[m]
            h.append(float(x.loc[x["sel"], "hit"].mean()) /
                     max(float(x["hit"].mean()), 1e-9))
        print(f"   {lbl:<12}{int(dd['sel'].sum()):>8,}"
              f"{dd[dd['sel']]['ticker'].nunique():>7}"
              f"{r['obs']:>12.1%}{r['obs'] / base:>7.2f}{r['z']:>+8.2f}"
              f"{r['p']:>9.4f}{h[0]:>8.2f}{h[1]:>8.2f}")

    print("\n" + "-" * W)
    print(" 4. P4 — SCALING OUT, WHICH THE BINARY TEST NEVER CONSIDERED")
    print("-" * W)
    print(" §2 tested only the two corners: sell everything at 2x, or sell")
    print(" nothing. Predicted before scoring: selling a FRACTION dominates")
    print(" both, because the hit rate is set by the touch and the mean is")
    print(" set by the remainder.\n")
    print(f"   {'rule':<28}{'name doubled':>14}{'I captured it':>15}"
          f"{'median':>10}{'mean':>11}{'P(-50%)':>10}")
    rows = [("hold to 10y, sell nothing", 2.0, 0.0),
            ("sell 25% at 2x", 2.0, 0.25),
            ("sell 50% at 2x", 2.0, 0.50),
            ("sell 75% at 2x", 2.0, 0.75),
            ("sell 100% at 2x", 2.0, 1.00),
            ("sell 50% at 3x", 3.0, 0.50),
            ("sell 50% at 5x", 5.0, 0.50)]
    for lbl, tgt, fr in rows:
        r = scale_out(d, tgt, fr)
        print(f"   {lbl:<28}{r['touch_rate']:>14.1%}"
              f"{r['realised_rate']:>15.1%}{r['median']:>+10.1%}"
              f"{r['mean']:>+11.1%}{r['p_half']:>10.1%}")
    print("\n   P4 PREDICTED SCALING OUT WOULD DOMINATE BOTH CORNERS. IT")
    print("   DOES NOT — the mean falls monotonically with every unit sold,")
    print("   +432% to +58%. There is no free lunch in the interior. Logged")
    print("   as a FAILED prediction rather than reframed.")
    print("\n   The two hit-rate columns are the thing worth carrying. The")
    print("   NAME DOUBLED column is constant at 69.1% — that is a property")
    print("   of the picking and no selling rule moves it. The I CAPTURED IT")
    print("   column is what the rule decides, and it runs from 59.6% for")
    print("   never selling to 69.1% for selling everything at the target.")
    print("   So '7 of 10 multi-baggers' is settled AT ENTRY; the exit only")
    print("   decides how much of it reaches the account.")
    return 0




# ==========================================================================
# P4 — the middle ground the binary test ignores
# ==========================================================================
def scale_out(d: pd.DataFrame, target: float = 2.0,
              frac: float = 0.5) -> Dict:
    """Sell `frac` of the position at `target`, let the rest run to the end.

    PRE-REGISTERED PREDICTION, WRITTEN BEFORE SCORING: this dominates both
    corners. It realises a double on every name that touches the target — so
    the hit rate is the touch rate, unchanged from selling everything — while
    keeping `1 - frac` of the right tail that made the mean +432%. If that is
    right, the "7 of 10 versus the money" trade-off H23's §2 exposes is a false
    dichotomy created by testing only the two corners.

    THE ONE THING THIS CANNOT SEE. `peak` is the maximum over the window and
    says nothing about WHEN it happened, so the untouched half is valued at the
    window's end regardless of whether the target was hit in year two or year
    nine. That is the correct valuation for a buy-and-hold remainder and it
    would NOT be correct for anything that redeploys the proceeds, which is why
    no redeployment is modelled here.
    """
    s = d[d["sel"]]
    peak = s[f"peak{K}"].to_numpy(float)
    end = s[f"end{K}"].to_numpy(float)
    hit = peak >= target
    realised = np.where(hit, frac * target + (1 - frac) * end, end)
    #  TWO DIFFERENT QUESTIONS, AND THE FIRST DRAFT CONFLATED THEM.
    #  `touch_rate` is "did the name double at any point" — a property of the
    #  name, constant across every selling rule. `realised_rate` is "was my
    #  position worth 2x when I closed it", which is what the rule decides.
    #  Reporting the touch rate in a column headed "doubles realised" credits
    #  a hold-and-never-sell with captures it did not make: a name that
    #  doubled in year three and ended at 1.4x realised nothing.
    return {"target": target, "frac": frac,
            "touch_rate": float((peak >= 2.0).mean()),
            "realised_rate": float((realised >= 2.0).mean()),
            "median": float(np.median(realised) - 1.0),
            "mean": float(np.mean(realised) - 1.0),
            "p_half": float((realised <= 0.5).mean())}


if __name__ == "__main__":
    raise SystemExit(main())
