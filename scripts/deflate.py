#!/usr/bin/env python3
"""H58 — the deflated Sharpe ratio §11 has demanded since day one.

WHAT WAS NEVER DONE. CLAUDE.md §11 names the deflated Sharpe as
non-negotiable. `hypotheses.md` opens by saying the trial count exists SO THAT
it can be computed. Fifty-seven hypotheses and 350 trials later it had never
been computed once. A running count that never enters a statistic is
bookkeeping, not a correction.

WHY IT IS A DIFFERENT CORRECTION FROM EVERY NULL IN THIS REPO. Seven times a
clustered permutation null has decided a result here, and it asks one question:
does the LABEL carry information? It cannot ask whether the strategy is the best
of 350 attempts, because it is only ever handed one attempt. The deflated Sharpe
is the correction for the SEARCH, and the two are not substitutes.

THE ONE INPUT THAT COULD HAVE BEEN FAKED, AND HOW IT IS MEASURED INSTEAD.
DSR needs the VARIANCE OF THE SHARPE RATIOS ACROSS TRIALS. A plausible constant
is easy to type and would make the deflation look measured when it was assumed.
It is estimated here instead, from the random-selection control arms that
`bhbench` already runs through the identical machinery: `draws` random baskets,
same universe, same calendar, same costs, each producing an equity path and
therefore a Sharpe.

    THIS PARAGRAPH ORIGINALLY ENDED "their spread IS the dispersion a search
    over this data generates from nothing, which is exactly the quantity DSR
    wants." THAT IS WRONG AND THE SENTENCE IS KEPT, MARKED, RATHER THAN
    DELETED -- A19 records fixing the code while leaving the refuted claim in
    the docstring above it. A control draw varies only WHICH NAMES are bought.
    This repo's 350 trials varied the whole hypothesis: broker flow, investor
    class, price features, exit rules, horizons, timing rules. The dispersion
    across THAT family is necessarily wider, so the control estimate is a
    LOWER BOUND on sr_variance and therefore an UPPER BOUND on the DSR. D4
    reports the DSR as a function of the assumption instead of resting on it.

-------------------------------------------------------------------- REGISTERED
Written before the numbers existed.

  D1  The surviving arms of H54 -- strength+calm and its sticky variants --
      have annualised Sharpes that clear zero comfortably. PREDICTION: they do
      NOT clear the deflated benchmark at 350 trials. If they did, this repo
      would be holding a result that survives both a permutation null and a
      search correction, which nothing here has ever done.

  D2  PREDICTED NULL. The random-selection control arm, run through the same
      deflation, lands at DSR ~0.5 or below. If a random basket clears the
      deflated bar, the deflation is not working and D1's answer means nothing.

  D3  The trial count needed to kill a result is far more informative than the
      DSR itself, because a probability near one is the number most likely to
      be over-read. PREDICTION: for the best arm, `trials_to_kill` is within an
      order of magnitude of 350 -- i.e. this repo's own search is roughly the
      size that decides the answer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from bhbench import Bench, load                                  # noqa: E402
from beathold import Sticky, s_all, strcalm, top                 # noqa: E402
from idxbot.metrics import (deflated_sharpe, describe,           # noqa: E402
                            expected_max_sharpe, psr, sharpe, summary)

#  hypotheses.md's own running count after H57. NOT a guess: the log states it.
TRIALS = 350
OUT = os.path.join("reports", "deflated.json")


def curve_returns(curve) -> np.ndarray:
    """Per-rebalance returns from a bhbench equity path."""
    if not curve or len(curve) < 3:
        return np.array([])
    eq = np.array([e for _d, e in curve], dtype=float)
    eq = np.concatenate([[1.0], eq])
    return eq[1:] / eq[:-1] - 1.0


def arms():
    """The families H54 measured, one bar each, plus the two benchmarks."""
    Q, A = 63, 252
    return [
        ("own everything, quarterly", s_all, Q),
        ("momentum top 10, quarterly", top("mom12_1", 10), Q),
        ("low vol top 10, quarterly", top("lowvol", 10), Q),
        ("strength+calm 10, quarterly", strcalm(10), Q),
        ("strength+calm sticky, quarterly", Sticky(10), Q),
        ("sticky tight buffer, quarterly",
         Sticky(10, keep_hi_q=0.80, keep_vol_q=0.60), Q),
        ("strength+calm 10, annual", strcalm(10), A),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=24,
                    help="random control baskets; these ESTIMATE sr_variance")
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()

    P = load()
    B = Bench(P)
    ppy = 252.0

    print(f"H58 — deflated Sharpe at {a.trials} trials")
    print(f"panel {P['date'].min().date()} -> {P['date'].max().date()}, "
          f"{P['ticker'].nunique()} names")

    rows = []
    arm_returns = {}
    for label, sel, freq in arms():
        if hasattr(sel, "reset"):
            sel.reset()
        r = B.walk(sel, freq=freq, offset=a.offset)
        if not r:
            print(f"\n{label}: no valid path")
            continue
        ret = curve_returns(r["curve"])
        if len(ret) < 8:
            print(f"\n{label}: too few rebalances ({len(ret)})")
            continue

        #  THE CONTROL ARMS ARE THE MEASUREMENT, NOT A SIDESHOW. Each is a
        #  random basket from the SAME universe on the SAME calendar with the
        #  SAME costs, so the spread of their Sharpes is the dispersion this
        #  data produces from no information at all.
        ctl_sr, ctl_ret = [], []
        for s in range(a.draws):
            if hasattr(sel, "reset"):
                sel.reset()
            c = B.walk(sel, freq=freq, offset=a.offset,
                       rng=np.random.default_rng(s))
            if not c:
                continue
            cr = curve_returns(c["curve"])
            if len(cr) < 8:
                continue
            v = sharpe(cr, ppy / freq)
            if np.isfinite(v):
                ctl_sr.append(v)
                ctl_ret.append(cr)
        if len(ctl_sr) < 4:
            print(f"\n{label}: only {len(ctl_sr)} usable control draws — "
                  f"sr_variance is not estimable, so NO DSR is reported")
            continue
        srv = float(np.var(ctl_sr, ddof=1))

        m = summary(ret, ppy / freq, trials=a.trials, sr_variance=srv)
        print(f"\n{label}   (rebalance every {freq} sessions)")
        print(describe(m))
        print(f"           control Sharpes: mean {np.mean(ctl_sr):+.2f}, "
              f"sd {np.std(ctl_sr, ddof=1):.2f}, n {len(ctl_sr)}")

        #  D2: the same deflation applied to a control arm. If a random basket
        #  clears the bar the deflation is broken and D1 means nothing.
        cm = deflated_sharpe(ctl_ret[0], a.trials, srv, ppy / freq)
        print(f"           D2 control arm  Sharpe {cm['sharpe']:+.2f}  "
              f"DSR {cm['dsr']:.3f}")

        arm_returns[label] = ret
        rows.append({
            "label": label, "freq": freq, "n": int(len(ret)),
            "cagr": m["cagr"], "sharpe": m["sharpe"], "psr": m["psr"],
            "sr_variance": srv, "sr_star": m["sr_star"], "dsr": m["dsr"],
            "trials_to_kill": m["trials_to_kill"],
            "control_sharpe_mean": float(np.mean(ctl_sr)),
            "control_dsr": cm["dsr"],
        })

    if not rows:
        print("\nnothing ran")
        return

    R = pd.DataFrame(rows)
    print("\n" + "=" * 88)
    print(f"{'arm':<34}{'Sharpe':>8}{'PSR':>8}{'SR*':>8}{'DSR':>8}"
          f"{'kill@':>10}{'ctlDSR':>8}")
    print("-" * 88)
    for _, x in R.iterrows():
        kill = (f"{x['trials_to_kill']:,.0f}"
                if np.isfinite(x["trials_to_kill"]) else "already")
        print(f"{x['label']:<34}{x['sharpe']:>+8.2f}{x['psr']:>8.3f}"
              f"{x['sr_star']:>+8.2f}{x['dsr']:>8.3f}{kill:>10}"
              f"{x['control_dsr']:>8.3f}")

    #  ---------------------------------------------------------------- D4
    #  THE ESTIMATE ABOVE IS A LOWER BOUND AND THAT CHANGES HOW IT READS.
    #  A control draw varies only WHICH NAMES are bought: same universe, same
    #  calendar, same rebalance frequency, same cost model. This repo's 350
    #  trials varied the whole hypothesis -- broker flow, investor class, price
    #  features, exit rules, horizons, timing rules. The dispersion of Sharpe
    #  across THAT family is necessarily wider than the dispersion across
    #  baskets of one family, and no control draw can see it, because this repo
    #  never stored a Sharpe per trial.
    #
    #  A lower bound on sr_variance is an UPPER bound on the DSR. So every
    #  number above is the most flattering reading available, and the sweep
    #  below is the honest deliverable: DSR as a FUNCTION of the assumption,
    #  rather than a single number resting on it.
    ctl_srv = float(R["sr_variance"].max())
    arm_srv = float(np.var(R["sharpe"], ddof=1)) if len(R) > 2 else float("nan")
    best_i = R["dsr"].idxmax()
    best = R.loc[best_i]
    best_ret = arm_returns[best["label"]]
    freq_best = int(best["freq"])
    print("\nD4  DSR of the best arm as a function of the assumed trial "
          "dispersion")
    print(f"    control draws (one family, basket luck only) give "
          f"sr_variance {ctl_srv:.4f}  <- a LOWER BOUND")
    print(f"    the 7 arms tested here (different families) give "
          f"{arm_srv:.4f}")
    print(f"\n    {'sr_variance':>12}{'sd of trial SR':>16}{'SR*':>8}"
          f"{'DSR':>8}")
    print("    " + "-" * 44)
    sweep = []
    for srv in (ctl_srv, arm_srv, 0.05, 0.10, 0.25, 0.50, 1.00):
        if not np.isfinite(srv) or srv <= 0:
            continue
        s_star = expected_max_sharpe(a.trials, srv, ppy / freq_best)
        d = psr(best_ret, s_star, ppy / freq_best)
        print(f"    {srv:>12.4f}{math.sqrt(srv):>16.2f}{s_star:>+8.2f}"
              f"{d:>8.3f}")
        sweep.append({"sr_variance": srv, "sr_star": s_star, "dsr": d})

    #  The verdict is printed by the script so it cannot drift from the numbers.
    n_clear = int((R["dsr"] >= 0.95).sum())
    ctl_max = float(R["control_dsr"].max())
    survives = [s for s in sweep if s["dsr"] >= 0.95]
    arm_dsr = next((s["dsr"] for s in sweep
                    if np.isfinite(arm_srv) and s["sr_variance"] == arm_srv),
                   float("nan"))
    print("\nVERDICT")
    print(f"  D1  {n_clear} of {len(R)} arms clear DSR 0.95 at "
          f"{a.trials} trials ON THE LOWER-BOUND dispersion.")
    print(f"      Best: {best['label']} at {best['dsr']:.3f}.")
    if np.isfinite(arm_dsr):
        print(f"      On the NEXT-WIDEST estimate available — the spread of "
              f"the {len(R)} families")
        print(f"      actually run here — the same arm reads {arm_dsr:.3f} and "
              f"{'still clears' if arm_dsr >= 0.95 else 'FAILS'}.")
        if arm_dsr < 0.95:
            print("      D1 is therefore CONFIRMED in substance: the best arm "
                  "survives only")
            print("      the single most flattering dispersion assumption "
                  "available.")
    print(f"  D2  the control arm's largest DSR is {ctl_max:.3f} "
          f"({'PASSES — the deflation bites' if ctl_max < 0.95 else 'FAILS — a random basket clears the bar, so nothing above is readable'}).")
    if np.isfinite(best["trials_to_kill"]):
        print(f"  D3  the best arm needs {best['trials_to_kill']:,.0f} trials "
              f"to fall below 0.95, against this repo's {a.trials}.")
    else:
        print(f"  D3  the best arm is already below 0.95 at {a.trials} "
              f"trials, so `trials_to_kill` does not apply.")
    print(f"  D4  it survives {len(survives)} of {len(sweep)} dispersion "
          f"assumptions. It dies once the trial Sharpes spread by about "
          f"{math.sqrt(min([s['sr_variance'] for s in sweep if s['dsr'] < 0.95], default=float('nan'))):.2f},")
    print("      which is a narrower spread than 350 hypotheses across six")
    print("      instruments would plausibly produce.")
    print("\n  WHAT THIS DOES NOT SAY. `sr_variance` is NOT recoverable from "
          "this repo's")
    print("  log, because no Sharpe was stored per trial. The control estimate "
          "is a")
    print("  lower bound by construction and therefore the DSR built on it is "
          "an")
    print("  UPPER bound on believability. A DSR is also a probability about "
          "the TRUE")
    print("  Sharpe of THIS series, not a claim it would repeat. The holdout "
          "was")
    print("  spent at H16 and every number here is in-sample.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"trials": a.trials, "draws": a.draws,
                   "offset": a.offset, "arms": rows,
                   "dispersion_sweep": sweep}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
