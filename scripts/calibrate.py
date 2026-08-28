#!/usr/bin/env python3
"""H36 — how accurate is the shipped indicator? Purged walk-forward calibration.

    python3 scripts/calibrate.py

WHAT "ACCURATE" HAS TO MEAN HERE, AND WHY A WIN RATE WOULD BE A LIE.

`pine/IDX_Suite.pine` does not emit calls. It emits PROBABILITIES — "58% chance
of touching +20% inside a year", "half the arrivals land between these two
dates". A hit rate is undefined for that output until someone picks a threshold,
and whoever picks the threshold decides the answer. The measures that are
defined are:

  CALIBRATION   when it says 60%, does it happen 60% of the time? A reliability
                table and a Brier score answer this and nothing else does.
  SKILL         is the Brier score better than just quoting the base rate? A
                perfectly calibrated constant is calibrated and useless, so the
                BRIER SKILL SCORE against the base rate is the number that
                separates a model from a lookup of the average.
  DISCRIMINATION does it rank? AUC, which is invariant to any miscalibration and
                so answers a different question from the first two.
  COVERAGE      the date band claims to contain half the arrivals. Does it?

EVERY NUMBER IN THE SHIPPED PANEL IS CURRENTLY IN-SAMPLE, which this script
exists to fix as far as it can be fixed. The reserved holdout was spent at H16
and cannot be un-spent, but a PURGED WALK-FORWARD is still genuinely
out-of-sample: for test year Y the laws are refitted using only bars whose
252-session forward window CLOSED before Y began. Without the purge a bar from
December Y-1 is still resolving inside Y and the "training" set contains the
test year's own outcomes.

TWO ARMS, AND THEY ANSWER DIFFERENT QUESTIONS.
  walk-forward  what the indicator would have told you at the time. Honest.
  shipped       the constants actually in the .pine file, scored year by year.
                In-sample by construction, so its skill is an upper bound — but
                its CALIBRATION DRIFT over the years is real information and is
                the thing a user running it today needs to know.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.cone import p_touch, sessions_to                    # noqa: E402
from levels import HORIZON                                       # noqa: E402
from time_price import TARGETS, design                           # noqa: E402

CONE = os.path.join("data", "spine", "cone.parquet")
OUT = "reports"
SEED = 20260828
FIRST_TEST_YEAR = 2008
MIN_TRAIN_CELLS = 40
#  Purge: a bar's outcome is not known until 252 sessions later, so a training
#  bar must be at least a year older than the test year's first bar.
PURGE_YEARS = 2


def wmean(v: pd.Series, w: pd.Series) -> float:
    """Weighted mean that SKIPS missing cells instead of returning nan.

    AUC is undefined in a cell where every outcome is the same class, which
    happens at the extreme targets in a quiet year. A plain np.average then
    propagates one nan through four million observations and prints `nan` for
    the whole study — which is exactly what the first run of this script did.
    """
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.average(v[m], weights=w[m])) if m.any() else np.nan


def load(liquid_only: bool = True) -> pd.DataFrame:
    C = pd.read_parquet(CONE)
    C = C[C["vol60"].notna() & (C["vol60"] > 0)].copy()
    if liquid_only:
        C = C[C["elig"]]
    C["year"] = pd.DatetimeIndex(C["date"]).year
    C["stk"] = C["stack"].fillna(False).astype(bool)
    return C


# ============================================================ fitting =========
def cells(C: pd.DataFrame, deciles: int = 10) -> pd.DataFrame:
    """Aggregate to (volatility decile x target x trend state), as H32b did."""
    C = C.copy()
    C["vd"] = pd.qcut(C["vol60"], deciles, labels=False, duplicates="drop")
    rows: List[Dict] = []
    for (vd, st), g in C.groupby(["vd", "stk"]):
        s = float(g["vol60"].median())
        for t in TARGETS:
            a = g[f"t{int(t * 100)}"].to_numpy()
            live = a != -2
            if live.sum() < 300:
                continue
            hit = a[live] > 0
            w = a[live][hit]
            r = {"sig": s, "stack": int(st), "up": int(t > 1),
                 "d": abs(np.log(t)), "n": int(live.sum()),
                 "p": float(np.clip(hit.mean(), 1e-4, 1 - 1e-4))}
            if len(w) >= 150:
                r["q1"], r["med"], r["q3"] = (float(np.percentile(w, q))
                                              for q in (25, 50, 75))
            rows.append(r)
    return pd.DataFrame(rows)


def fit(R: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Both laws, in the same design H32b shipped."""
    out: Dict[str, np.ndarray] = {}
    for up in (1, 0):
        S = R[R["up"] == up]
        if len(S) < MIN_TRAIN_CELLS // 2:
            return {}
        X = np.column_stack([design(S), S["stack"].to_numpy(float)])
        y = np.log(S["p"] / (1 - S["p"])).to_numpy()
        w = np.sqrt(S["n"].to_numpy(float))
        out["up" if up else "down"] = np.linalg.lstsq(X * w[:, None], y * w,
                                                      rcond=None)[0]
    T = R.dropna(subset=["med"])
    if len(T) < MIN_TRAIN_CELLS:
        return {}
    X = design(T)
    for c in ("q1", "med", "q3"):
        out[c] = np.linalg.lstsq(X, np.log(T[c].to_numpy(float)),
                                 rcond=None)[0]
    return out


def predict_p(beta: np.ndarray, d: np.ndarray, sig: np.ndarray,
              stack: np.ndarray) -> np.ndarray:
    ld, ls = np.log(d), np.log(sig)
    z = (beta[0] + beta[1] * ld + beta[2] * ld ** 2 + beta[3] * ls
         + beta[4] * ld * ls + beta[5] * stack)
    return 1.0 / (1.0 + np.exp(-z))


def predict_t(beta: np.ndarray, d: np.ndarray, sig: np.ndarray) -> np.ndarray:
    ld, ls = np.log(d), np.log(sig)
    return np.exp(beta[0] + beta[1] * ld + beta[2] * ld ** 2 + beta[3] * ls
                  + beta[4] * ld * ls)


# ============================================================ scoring =========
def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank AUC, computed from the Mann-Whitney statistic so ties are handled
    at half weight rather than silently favouring the model."""
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(p).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def score(y: np.ndarray, p: np.ndarray, base: float) -> Dict[str, float]:
    """Brier, skill against the base rate, AUC, and the calibration slope.

    THE BASE RATE IS THE THING TO BEAT, not zero. A model that always answers
    "12%" on a market that doubles 12% of the time is perfectly calibrated and
    carries no information, and only the skill score exposes that.
    """
    b = float(np.mean((p - y) ** 2))
    b0 = float(np.mean((base - y) ** 2))
    return {"n": len(y), "obs": float(y.mean()), "pred": float(p.mean()),
            "brier": b, "brier_base": b0,
            "bss": 1.0 - b / b0 if b0 > 0 else np.nan,
            "auc": auc(y, p)}


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(pd.Series(p), bins, labels=False, duplicates="drop")
    D = pd.DataFrame({"y": y, "p": p, "b": q})
    return D.groupby("b").agg(n=("y", "size"), predicted=("p", "mean"),
                              observed=("y", "mean")).reset_index()


def evaluate(C: pd.DataFrame, arm: str) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                 pd.DataFrame]:
    """One row per (test year, target); plus pooled reliability and coverage."""
    years = sorted(y for y in C["year"].unique() if y >= FIRST_TEST_YEAR)
    rows: List[Dict] = []
    keep_y: List[np.ndarray] = []
    keep_p: List[np.ndarray] = []
    cov: List[Dict] = []
    for Y in years:
        test = C[C["year"] == Y]
        if len(test) < 500:
            continue
        if arm == "walk":
            train = C[C["year"] <= Y - PURGE_YEARS]
            if train["year"].nunique() < 4:
                continue
            beta = fit(cells(train))
            if not beta:
                continue
        for t in TARGETS:
            col = f"t{int(t * 100)}"
            a = test[col].to_numpy()
            live = a != -2
            if live.sum() < 200:
                continue
            y = (a[live] > 0).astype(float)
            d = np.full(live.sum(), abs(np.log(t)))
            sig = np.clip(test["vol60"].to_numpy()[live], 0.0117, 0.0623)
            stk = test["stk"].to_numpy()[live].astype(float)
            if arm == "walk":
                p = predict_p(beta["up" if t > 1 else "down"], d, sig, stk)
            else:
                p = np.array([p_touch(t, s, bool(k)) for s, k in zip(sig, stk)])
            #  The base rate a naive user would quote: the pooled frequency for
            #  this target over the TRAINING years, never the test year's own.
            hist = C[C["year"] < Y][col].to_numpy()
            hl = hist != -2
            base = float((hist[hl] > 0).mean()) if hl.sum() > 500 else np.nan
            if not np.isfinite(base):
                continue
            r = score(y, p, base)
            r.update({"year": Y, "target": t, "arm": arm})
            rows.append(r)
            keep_y.append(y)
            keep_p.append(p)
            #  Date-band coverage, on the bars that actually arrived.
            w = a[live][y == 1]
            if len(w) >= 200:
                if arm == "walk":
                    q1 = predict_t(beta["q1"], d[y == 1], sig[y == 1])
                    q3 = predict_t(beta["q3"], d[y == 1], sig[y == 1])
                else:
                    q1 = np.array([sessions_to(t, s, "q1")
                                   for s in sig[y == 1]])
                    q3 = np.array([sessions_to(t, s, "q3")
                                   for s in sig[y == 1]])
                cov.append({"year": Y, "target": t, "arm": arm, "n": len(w),
                            "inside": float(np.mean((w >= q1) & (w <= q3))),
                            "below": float(np.mean(w < q1))})
    R = pd.DataFrame(rows)
    REL = reliability(np.concatenate(keep_y), np.concatenate(keep_p)) \
        if keep_y else pd.DataFrame()
    return R, REL, pd.DataFrame(cov)


# ============================================================= the race =======
def race(C: pd.DataFrame) -> pd.DataFrame:
    """Which barrier arrived first, against the shipped race law."""
    from idxbot.cone import p_target_first
    rows: List[Dict] = []
    for tp in (1.10, 1.20, 1.50):
        for sl in (0.90, 0.80, 0.67):
            a = C[f"t{int(tp * 100)}"].to_numpy()
            b = C[f"t{int(sl * 100)}"].to_numpy()
            ta = np.where(a > 0, a, np.inf)
            tb = np.where(b > 0, b, np.inf)
            sel = np.isfinite(np.minimum(ta, tb)) & (a != -2) & (b != -2)
            if sel.sum() < 1000:
                continue
            y = (ta[sel] < tb[sel]).astype(float)
            sig = np.clip(C["vol60"].to_numpy()[sel], 0.0117, 0.0623)
            p = np.array([p_target_first(tp - 1.0, 1.0 - sl, s) for s in sig])
            r = score(y, p, float(y.mean()))
            r.update({"tp": tp - 1.0, "sl": 1.0 - sl})
            rows.append(r)
    return pd.DataFrame(rows)


# ================================================================== main ======
def main() -> int:
    C = load()
    print(f"{len(C):,} eligible bars, {C['ticker'].nunique()} names, "
          f"{C['year'].min()}-{C['year'].max()}\n")

    tables = {}
    for arm in ("shipped", "walk"):
        R, REL, COV = evaluate(C, arm)
        tables[arm] = (R, REL, COV)
        R.to_csv(os.path.join(OUT, f"calibrate_{arm}.csv"), index=False)
        w = R["n"]
        print(f"=== {arm.upper()}  {R['year'].min()}-{R['year'].max()}, "
              f"{int(R['n'].sum()):,} scored bar-targets")
        print(f"  Brier {wmean(R['brier'], w):.4f}   "
              f"base-rate Brier {wmean(R['brier_base'], w):.4f}   "
              f"skill {wmean(R['bss'], w):+.4f}   "
              f"AUC {wmean(R['auc'], w):.4f}")
        print(f"  mean predicted {wmean(R['pred'], w):.4f} vs "
              f"observed {wmean(R['obs'], w):.4f}")
        print("  reliability — when it says X, how often does it happen?")
        print("   " + REL.to_string(index=False,
                                    float_format=lambda v: f"{v:,.4f}")
              .replace("\n", "\n   "))
        if not COV.empty:
            print(f"  date band covers {wmean(COV['inside'], COV['n']):.3f}"
                  f" of arrivals (claims 0.50), "
                  f"{wmean(COV['below'], COV['n']):.3f} arrive early")
        print()

    print("=== by target, walk-forward")
    R = tables["walk"][0]
    G = R.groupby("target").apply(
        lambda g: pd.Series({
            "n": g["n"].sum(),
            "observed": wmean(g["obs"], g["n"]),
            "predicted": wmean(g["pred"], g["n"]),
            "bss": wmean(g["bss"], g["n"]),
            "auc": wmean(g["auc"], g["n"])}),
        include_groups=False).reset_index()
    print(G.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print("\n=== by year, walk-forward (is it drifting?)")
    Y = R.groupby("year").apply(
        lambda g: pd.Series({
            "n": g["n"].sum(),
            "observed": wmean(g["obs"], g["n"]),
            "predicted": wmean(g["pred"], g["n"]),
            "bss": wmean(g["bss"], g["n"]),
            "auc": wmean(g["auc"], g["n"])}),
        include_groups=False).reset_index()
    print(Y.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    #  "Run it on ALL the stocks" is a different question from the one the laws
    #  were fitted on: they were fitted on names clearing Rp1bn/day, and the
    #  rest of the board is four times as many bars and much thinner. Whether
    #  the calibration survives out there is the user's actual question.
    A = load(liquid_only=False)
    print(f"\n=== THE WHOLE BOARD: {len(A):,} bars, {A['ticker'].nunique()} "
          f"names, against {len(C):,} bars / {C['ticker'].nunique()} names "
          f"above the Rp1bn/day floor the laws were fitted on")
    RB, RELB, COVB = evaluate(A, "shipped")
    RB.to_csv(os.path.join(OUT, "calibrate_allboard.csv"), index=False)
    print(f"  Brier {wmean(RB['brier'], RB['n']):.4f}   "
          f"base-rate Brier {wmean(RB['brier_base'], RB['n']):.4f}   "
          f"skill {wmean(RB['bss'], RB['n']):+.4f}   "
          f"AUC {wmean(RB['auc'], RB['n']):.4f}")
    print(f"  mean predicted {wmean(RB['pred'], RB['n']):.4f} vs "
          f"observed {wmean(RB['obs'], RB['n']):.4f}")
    print("   " + RELB.to_string(index=False,
                                 float_format=lambda v: f"{v:,.4f}")
          .replace("\n", "\n   "))
    if not COVB.empty:
        print(f"  date band covers {wmean(COVB['inside'], COVB['n']):.3f} "
              f"of arrivals (claims 0.50)")

    RA = race(C)
    RA.to_csv(os.path.join(OUT, "calibrate_race.csv"), index=False)
    print("\n=== the race law: P(target first | one of them arrives)")
    print(RA[["tp", "sl", "n", "obs", "pred", "brier", "bss", "auc"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
