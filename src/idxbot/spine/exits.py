"""Exit rules for the multiplier-cell entry — the half of the design that was
missing.

WHY THIS EXISTS
----------------
H16 ran the published entry rule on 2025-08-25 and held for a fixed year with
no stop. The attribution was unambiguous: the ten names reached a mean PEAK of
**+102.2%** and realised **+15.1%**, giving back 43.4 points. Four peaked above
+100% and two ended there. INET was a triple at session 77 and finished +27.9%.

The entry was not the failure. The absence of an exit was.

And it is structural rather than bad luck. The entry rule selects for
P(2x), which selects for path volatility, which produces large peaks AND large
give-backs by construction. **A selection rule that maximises tail width cannot
be held passively.** Entry and exit are one design and only half of it existed.

THE METHODOLOGICAL POINT THAT DECIDES EVERYTHING HERE
------------------------------------------------------
Exit parameters are trivially overfittable — on the H16 cohort a six-month hold
"returned" +41.4% and a nine-month hold −14.6%, from the same ten names. That
surface is noise, and any parameter read off it is worth nothing.

So every rule in this module is fitted and selected on **pre-holdout data
only** (through 2024-08-23), by walk-forward across cohorts: choose on the
years up to Y, score on Y+1, never the reverse. The 2025 cohort is used once at
the end as an illustration and is explicitly NOT the basis of any choice — the
holdout is already spent (H16) and cannot certify anything now.

WHAT A RULE IS SCORED ON
--------------------------
The COHORT is the unit, not the name. Ten names picked on one date share a
market, a regime and often a sector, so treating them as ten independent
observations overstates the sample by roughly the number of names. Cohort-level
statistics with a cohort-level bootstrap is the honest accounting, and it is
why the numbers here have wider intervals than a naive per-name view.

Costs are charged once per name per cohort — one round trip whether the exit
comes from a stop or from the horizon — at A5's 0.56% plus a fraksi-harga
half-spread at the entry price.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Longest a position is held under any rule, in sessions. One year.
HORIZON = 252

#: A5's round trip, before the spread term the caller adds per name.
FEE = 0.0056


# ==========================================================================
# the rules
# ==========================================================================
def hold(path: np.ndarray, n: int = HORIZON) -> Tuple[float, int]:
    """Buy and hold for ``n`` sessions. The rule H16 actually ran."""
    if not len(path):
        return (np.nan, 0)
    i = min(n, len(path)) - 1
    return (float(path[i]) - 1.0, i + 1)


def trailing(path: np.ndarray, drop: float, arm: float = 0.0,
             n: int = HORIZON) -> Tuple[float, int]:
    """Exit when price falls ``drop`` from its running peak.

    ``arm`` delays the trail until the position is up that much, which stops a
    tight trail from being knocked out by ordinary noise on day three. With
    ``arm=0`` it trails from entry.
    """
    peak = 1.0
    for i, p in enumerate(path[:n]):
        peak = max(peak, p)
        if peak >= 1.0 + arm and p <= peak * (1.0 - drop):
            return (float(p) - 1.0, i + 1)
    return hold(path, n)


def hard_stop(path: np.ndarray, drop: float, n: int = HORIZON
              ) -> Tuple[float, int]:
    """Exit when price falls ``drop`` below the ENTRY, never re-entering."""
    for i, p in enumerate(path[:n]):
        if p <= 1.0 - drop:
            return (float(p) - 1.0, i + 1)
    return hold(path, n)


def time_stop(path: np.ndarray, by: int, need: float, n: int = HORIZON
              ) -> Tuple[float, int]:
    """Exit at session ``by`` unless the position is up ``need`` by then.

    MOTIVATED BY A CLEAN PATTERN, NOT BY A GRID. In the H16 cohort the three
    names that never worked — BLOG, KRYA, MERI — peaked at −3.3%, −1.6% and
    +2.2% within sessions 10 to 14 and then collapsed to −34%, −75% and −55%.
    A name that has not moved in three weeks is a different animal from one
    that has, and this is the cheapest test of that.
    """
    if not len(path):
        return (np.nan, 0)
    k = min(by, len(path)) - 1
    if float(np.max(path[:k + 1])) < 1.0 + need:
        return (float(path[k]) - 1.0, k + 1)
    return hold(path, n)


def combined(path: np.ndarray, stop: float, drop: float, arm: float,
             by: int = 0, need: float = 0.0, n: int = HORIZON
             ) -> Tuple[float, int]:
    """Hard stop, then a trail once armed, with an optional time stop.

    Evaluated in the order a real book would: the hard stop is checked first
    because it is the one that caps the loss.
    """
    peak = 1.0
    for i, p in enumerate(path[:n]):
        if p <= 1.0 - stop:
            return (float(p) - 1.0, i + 1)
        peak = max(peak, p)
        if peak >= 1.0 + arm and p <= peak * (1.0 - drop):
            return (float(p) - 1.0, i + 1)
        if by and i + 1 == min(by, len(path)) and peak < 1.0 + need:
            return (float(p) - 1.0, i + 1)
    return hold(path, n)


def catalogue() -> Dict[str, Callable[[np.ndarray], Tuple[float, int]]]:
    """Every candidate, named. The search space, fixed before any scoring."""
    R: Dict[str, Callable] = {}
    for n in (63, 126, 189, 252):
        R[f"hold {n}"] = (lambda p, n=n: hold(p, n))
    for d in (0.15, 0.20, 0.25, 0.30, 0.40):
        R[f"trail {d:.0%}"] = (lambda p, d=d: trailing(p, d))
        R[f"trail {d:.0%} armed +50%"] = (lambda p, d=d: trailing(p, d, 0.50))
    for s in (0.15, 0.20, 0.25, 0.30):
        R[f"stop {s:.0%}"] = (lambda p, s=s: hard_stop(p, s))
    for by, need in ((21, 0.0), (21, 0.10), (42, 0.10)):
        R[f"time {by}d need +{need:.0%}"] = (
            lambda p, by=by, need=need: time_stop(p, by, need))
    for s in (0.20, 0.25, 0.30):
        for d in (0.25, 0.30, 0.40):
            R[f"stop {s:.0%} + trail {d:.0%} armed +50%"] = (
                lambda p, s=s, d=d: combined(p, s, d, 0.50))
    for s in (0.25,):
        for d in (0.25, 0.30):
            R[f"stop {s:.0%} + trail {d:.0%} + time 21d"] = (
                lambda p, s=s, d=d: combined(p, s, d, 0.50, by=21, need=0.10))
    return R


# ==========================================================================
# scoring
# ==========================================================================
def apply_rule(paths: Sequence[np.ndarray], costs: Sequence[float],
               rule: Callable) -> pd.DataFrame:
    """Net return and holding period per position under one rule."""
    out = []
    for p, c in zip(paths, costs):
        r, held = rule(np.asarray(p, dtype=float))
        out.append({"gross": r, "cost": c, "net": r - c, "held": held})
    return pd.DataFrame(out)


def score_cohorts(cohorts: Dict[pd.Timestamp, Dict[str, object]],
                  rule: Callable) -> pd.DataFrame:
    """One row per cohort: what the rule did to that date's ten names."""
    rows = []
    for day, c in cohorts.items():
        D = apply_rule(c["paths"], c["costs"], rule)
        D = D[np.isfinite(D["net"])]
        if D.empty:
            continue
        rows.append({
            "as_of": day, "n": len(D),
            "mean": float(D["net"].mean()),
            "median": float(D["net"].median()),
            "p2": float((D["gross"] >= 1.0).mean()),
            "pdn": float((D["gross"] <= -0.5).mean()),
            "held": float(D["held"].mean()),
            "uni_med": c.get("uni_med", np.nan)})
    return pd.DataFrame(rows)


def cohort_block(S: pd.DataFrame, horizon: int = HORIZON) -> int:
    """How many consecutive cohorts share a forward window.

    Monthly cohorts each holding a year overlap in eleven of their twelve
    months, so twelve consecutive cohorts are very nearly one observation.
    Inferred from the actual spacing of ``as_of`` rather than assumed, because
    the same function is used on weekly and quarterly cohort sets.
    """
    if "as_of" not in S.columns or len(S) < 3:
        return 1
    d = pd.DatetimeIndex(S["as_of"]).sort_values()
    # NOT ``.asi8``: this pandas builds date ranges as datetime64[us], so the
    # integer view is microseconds here and nanoseconds elsewhere, and a
    # hardcoded divisor was silently off by a thousand — it returned a block
    # of 11,783 for monthly cohorts, which then clipped to a degenerate value.
    step = float(np.median(np.diff(
        d.to_numpy().astype("datetime64[D]").astype("int64"))))
    if not np.isfinite(step) or step <= 0:
        return 1
    return int(max(1, np.ceil(horizon * 365.25 / 252 / step)))


def bootstrap_cohorts(S: pd.DataFrame, col: str = "mean", draws: int = 2000,
                      seed: int = 20260825, block: Optional[int] = None
                      ) -> Tuple[float, float]:
    """CI on the average cohort statistic, by MOVING-BLOCK resample.

    Two levels of dependence have to be respected and the first version of
    this function respected only one.

    *Within a cohort* — ten names chosen on one date share a market, a regime
    and often a sector, so the unit is the cohort, never the name.

    *Between cohorts* — and this is the one that was missed. Monthly cohorts
    holding for a year overlap in eleven months out of twelve, so 188 of them
    span about sixteen independent year-windows. An iid resample over them
    treats near-duplicates as fresh evidence and returns an interval roughly
    ``sqrt(188/16) ≈ 3.4x`` too narrow. H16 made exactly this point about its
    own twelve cohorts ("effective n is ~1, not 12") and then the exit study
    reintroduced it one layer up.

    Blocks are drawn WITH replacement and duplicates are kept — A11 records
    that filtering them with ``np.isin`` silently shrank every resample and
    made every interval too narrow, which is the failure mode a bootstrap is
    supposed to protect against.
    """
    T = S.dropna(subset=[col])
    if "as_of" in T.columns:
        T = T.sort_values("as_of")
    x = T[col].to_numpy(dtype=float)
    n = len(x)
    if n < 5:
        return (np.nan, np.nan)
    # A block that is a large fraction of the sample DEGENERATES: with only
    # three or four blocks covering the whole series each block mean is already
    # close to the sample mean, and the interval collapses back to something
    # narrower than the iid one. Measured here: widths 0.049 (b=1), 0.105
    # (b=13), 0.047 (b=63) on n=189. Cap at a fifth so that can't happen
    # quietly.
    b = int(block) if block is not None else cohort_block(T)
    b = int(np.clip(b, 1, max(1, n // 5)))
    rng = np.random.default_rng(seed)
    if b == 1:
        m = rng.choice(x, size=(draws, n), replace=True).mean(axis=1)
    else:
        k = int(np.ceil(n / b))
        start = rng.integers(0, n - b + 1, size=(draws, k))
        take = (start[:, :, None] + np.arange(b)[None, None, :]
                ).reshape(draws, k * b)[:, :n]
        m = x[take].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def walk_forward_select(cohorts: Dict[pd.Timestamp, Dict[str, object]],
                        rules: Dict[str, Callable],
                        min_train: int = 24,
                        purge: bool = True) -> pd.DataFrame:
    """Choose the rule on SETTLED past cohorts only, score it on the next one.

    THIS IS THE WHOLE POINT OF THE MODULE. Picking the best rule over the full
    sample and quoting its full-sample return is how a six-month hold comes to
    look like +41%. Here the choice at cohort *t* sees only cohorts before *t*,
    so the reported series is what the procedure would actually have earned.

    THE PURGE IS NOT OPTIONAL AND THE FIRST VERSION OF THIS FUNCTION OMITTED
    IT. Cohorts are monthly and the horizon is a full year, so last month's
    cohort has eleven months still to run when this month's decision is made —
    its outcome is not known and training on it is a look-ahead of exactly the
    kind §11 forbids. With the purge on, the training set at cohort *t* holds
    only cohorts whose forward window CLOSED on or before *t*, which for a
    252-session horizon discards the most recent ~12 cohorts at every step.

    Each cohort may carry ``settles`` — the date its horizon ends. Without one
    the horizon is approximated on the calendar (252 sessions ≈ 366 days),
    which is conservative in the right direction because it over-purges.

    Returns one row per scored cohort: which rule was chosen, what it earned,
    and what plain buy-and-hold earned on the same names.
    """
    days = sorted(cohorts)
    approx = pd.Timedelta(days=int(HORIZON * 365.25 / 252) + 1)
    settles = {d: (cohorts[d].get("settles") or d + approx) for d in days}
    per_rule = {name: score_cohorts(cohorts, fn).set_index("as_of")
                for name, fn in rules.items()}
    base = score_cohorts(cohorts, lambda p: hold(p, HORIZON)).set_index("as_of")
    rows = []
    for i, d in enumerate(days):
        past = [p for p in days[:i] if not purge or settles[p] <= d]
        if len(past) < min_train:
            continue
        best, best_v = None, -np.inf
        for name, S in per_rule.items():
            v = S.reindex(past)["median"].mean()
            if np.isfinite(v) and v > best_v:
                best, best_v = name, v
        if best is None or d not in per_rule[best].index:
            continue
        r = per_rule[best].loc[d]
        rows.append({"as_of": d, "rule": best,
                     "median": r["median"], "mean": r["mean"],
                     "p2": r["p2"], "pdn": r["pdn"], "held": r["held"],
                     "bh_median": base.loc[d, "median"] if d in base.index
                                  else np.nan,
                     "bh_mean": base.loc[d, "mean"] if d in base.index
                                else np.nan,
                     "uni_med": r.get("uni_med", np.nan)})
    return pd.DataFrame(rows)
