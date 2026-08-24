#!/usr/bin/env python3
"""H14 — does a broker's STYLE persist even though its EDGE does not?

THREE QUESTIONS, ALL PRE-REGISTERED IN hypotheses.md
------------------------------------------------------
    Q1  PERSISTENCE OF STYLE. Year-over-year rank correlation of each §9.4
        fingerprint metric, against the identical within-ticker-window label
        shuffle H11 used for margin. H11's margin persisted at +0.078 — inside
        its null. The registered prediction is that STYLE persists far more
        strongly, because a fingerprint describes a firm's business model
        rather than its skill.

    Q2  DEGRADATION. §6.3 says large players split orders across brokers
        precisely because the summary is public, and §9.5 says to plot
        distinctiveness by year and report any decay prominently rather than
        averaging it away. Measured as the mean pairwise distance between
        standardised fingerprints.

    Q3  ARCHETYPE STABILITY. §9.5's mandatory check: fit on the early era,
        assign on the late era, and see whether assignments persist. HDBSCAN
        and GMM, compared, as §9.5 asks.

WHY THIS IS NOT A REPEAT OF H11
---------------------------------
H11 asked whether a broker's PROFITABILITY rank carries over and found it does
not. Nothing in that answers whether the broker's behaviour is stable, and the
two come apart cleanly: a market maker that crosses 90% of its flow every year
of its life has a completely stable style and, on H11's evidence, no stable
edge at all. Reporting the first as if it were the second is the specific error
§9.6 exists to prevent, so this script computes no P&L anywhere.

§9.6 DOSSIERS ARE CONDITIONAL ON Q3
-------------------------------------
If archetypes do not prove stable, the honest output is "no stable archetype"
and no dossier is written. That is §9.6's own rule for below-threshold cases,
and fabricating a behavioural read is exactly what it forbids.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.spine.fingerprint import (METRICS, MIN_GROSS,          # noqa: E402
                                      MIN_WINDOWS, STYLE,
                                      distinctiveness, execution_edges,
                                      fingerprints, standardise)
from idxbot.spine.persistence import spearman                      # noqa: E402
from persistence import build_track_b                              # noqa: E402

CACHE = os.path.join("data", "spine", "fingerprints.csv.gz")
EARLY = (2014, 2019)
LATE = (2020, 2026)


def build(rebuild: bool = False) -> pd.DataFrame:
    if os.path.exists(CACHE) and not rebuild:
        return pd.read_csv(CACHE)
    T = build_track_b()
    if T.empty:
        return T
    T["window_end"] = pd.to_datetime(T["window_end"])
    T = T[T["window_end"].dt.year >= EARLY[0]]
    print(f"   computing self-excluded visible VWAP over {len(T):,} rows …",
          flush=True)
    E = execution_edges(T)
    F = fingerprints(T, E)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    tmp = CACHE + ".tmp"
    F.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, CACHE)
    return F


def yoy(F: pd.DataFrame, metric: str) -> tuple:
    """Mean year-over-year rank correlation of one metric, and its pairs."""
    w = F.pivot_table(index="broker", columns="year", values=metric)
    years = sorted(w.columns)
    out = []
    for a, b in zip(years, years[1:]):
        if b - a != 1:
            continue
        s = w[[a, b]].dropna()
        if len(s) < 6:
            continue
        r = spearman(s[a].to_numpy(), s[b].to_numpy())
        if np.isfinite(r):
            out.append(r)
    return (float(np.mean(out)) if out else np.nan, np.asarray(out))


def null_yoy(T: pd.DataFrame, E: pd.DataFrame, draws: int,
             seed: int) -> Dict[str, np.ndarray]:
    """Every metric's year-over-year persistence under shuffled broker labels.

    The identical null H11 used, for the identical reason: it preserves each
    window's flow and each broker's size distribution and destroys only which
    code owned which row.

    Two efficiencies that do not weaken it. All seven metrics come from ONE
    recomputed fingerprint per draw rather than one draw per metric. And the
    execution edges are reused rather than recomputed: shuffling a LABEL leaves
    every row's own prices and volumes untouched, so a row's edge is unchanged
    and only its attribution moves — the edge frame is row-aligned with T, so
    the same permuted label column applies to both.

    What is NOT skipped is recomputing the fingerprint itself inside the loop.
    Shuffling finished fingerprints would test something far easier.
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, List[float]] = {m: [] for m in METRICS}
    for i in range(draws):
        N = T.copy()
        N["broker"] = N.groupby(["ticker", "window_end"])["broker"].transform(
            lambda s: rng.permutation(s.to_numpy()))
        Ne = E.copy()
        Ne["broker"] = N["broker"].to_numpy()
        F = fingerprints(N, Ne)
        for m in METRICS:
            out[m].append(yoy(F, m)[0] if m in F else np.nan)
        print(f"       null draw {i+1}/{draws}", end="\r",
              file=sys.stderr, flush=True)
    print(" " * 40, end="\r", file=sys.stderr)
    return {m: np.asarray(v, dtype=float) for m, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    print("=" * 78)
    print(" H14 — does a broker's STYLE persist even though its EDGE does not?")
    print("=" * 78)
    F = build(a.rebuild)
    if F.empty:
        print(" no fingerprints; run the collection first")
        return 1
    print(f"   {F.broker.nunique()} codes, {F.year.nunique()} years, "
          f"{len(F):,} broker-years  "
          f"(guards: >={MIN_WINDOWS} windows, >=Rp {MIN_GROSS:,.0f})\n")

    # ------------------------------------------------------------ Q1
    print(" Q1  DOES STYLE PERSIST?   (H11's margin managed +0.078, inside null)")
    nulls: Dict[str, np.ndarray] = {}
    if a.draws:
        T = build_track_b()
        T["window_end"] = pd.to_datetime(T["window_end"])
        T = T[T["window_end"].dt.year >= EARLY[0]]
        nulls = null_yoy(T, execution_edges(T), a.draws, a.seed)
    for m in METRICS:
        if m not in F:
            continue
        r, pairs = yoy(F, m)
        if not np.isfinite(r):
            print(f"     {m:<10} not enough adjacent years")
            continue
        tag = "STYLE" if m in STYLE else "artefact"
        nb = ""
        v = nulls.get(m, np.array([]))
        v = v[np.isfinite(v)]
        if len(v):
            sd = v.std()
            # The permutation p FLOORS at 1/(draws+1), so with a modest number
            # of draws it cannot separate "just outside" from "nowhere near".
            # The distance in null standard deviations is not floored and is
            # what actually discriminates here — and it is the only thing that
            # shows `share` sitting BELOW its own null.
            z = (r - v.mean()) / sd if sd > 0 else np.nan
            nb = (f"   null {v.mean():+.3f}+/-{sd:.3f}"
                  f"   {z:+5.1f} sd")
        print(f"     {m:<10} {r:+.3f}  pairs {len(pairs)}"
              f"  range [{pairs.min():+.2f}, {pairs.max():+.2f}]{nb}   {tag}",
              flush=True)
    print()

    # ------------------------------------------------------------ Q2
    print(" Q2  DOES DISTINCTIVENESS DEGRADE?  (§6.3 order-splitting, §9.5)")
    Dst = distinctiveness(F)
    if len(Dst) >= 4:
        x = Dst["year"].to_numpy(dtype=float)
        y = Dst["mean_distance"].to_numpy(dtype=float)
        slope = np.polyfit(x, y, 1)[0]
        rho = spearman(x, y)
        print("     " + "  ".join(f"{int(r.year)}:{r.mean_distance:.2f}"
                                  for r in Dst.itertuples()))
        print(f"     trend {slope:+.4f} per year   rank corr with year "
              f"{rho:+.3f}   n = {len(Dst)} years")
        # A slope of -0.003 on a series that sits at 3.4 is not a decline, and
        # calling it one because the sign is negative is how a null result
        # becomes a finding. The verdict needs the slope, the rank correlation
        # with time AND the endpoints to agree before it says anything.
        span = abs(y.max() - y.min())
        falls = (slope < 0) and (rho < 0) and (y[-1] < y[0])
        rises = (slope > 0) and (rho > 0) and (y[-1] > y[0])
        verdict = ("DECLINES" if falls else "RISES" if rises
                   else "NO DETECTABLE TREND")
        print(f"     {verdict} — first {y[0]:.2f}, last {y[-1]:.2f}, "
              f"full range {span:.2f} ({100*span/y.mean():.1f}% of the level)")
    else:
        print("     too few years to speak to a trend")
    print()

    # ------------------------------------------------------------ Q3
    print(" Q3  ARE ARCHETYPES STABLE?  (fit "
          f"{EARLY[0]}-{EARLY[1]}, assign {LATE[0]}-{LATE[1]})")
    archetypes(F, a.seed)

    print("\n §9.6 dossiers are written only if Q3 shows stable archetypes.")
    print(" Where it does not, the honest output is 'no stable archetype' —")
    print(" §9.6's own rule for below-threshold cases.")
    return 0


def archetypes(F: pd.DataFrame, seed: int) -> None:
    """§9.5: cluster, then check the assignment survives an era change."""
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.mixture import GaussianMixture

    Z = standardise(F)
    zc = [c + "_z" for c in STYLE if c + "_z" in Z]
    # one fingerprint per broker per era, averaged over that era's years
    Z["era"] = np.where(Z["year"] <= EARLY[1], "early", "late")
    P = Z.groupby(["broker", "era"])[zc].mean().reset_index()
    w = P.pivot_table(index="broker", columns="era", values=zc)
    both = w.dropna()
    if len(both) < 10:
        print("     too few brokers present in both eras")
        return
    Xe = both[[(c, "early") for c in zc]].to_numpy(dtype=float)
    Xl = both[[(c, "late") for c in zc]].to_numpy(dtype=float)
    print(f"     {len(both)} codes present in both eras, "
          f"{len(zc)} style dimensions")

    # Q1's answer at the VECTOR level: does the whole fingerprint carry over?
    per_dim = [spearman(Xe[:, i], Xl[:, i]) for i in range(len(zc))]
    print("     per-dimension early->late rank corr  "
          + "  ".join(f"{c[:-2]}:{r:+.2f}" for c, r in zip(zc, per_dim)))

    for name, fit in (("GMM", lambda k: GaussianMixture(
                            n_components=k, random_state=seed, n_init=3)),
                      ("KMeans", lambda k: KMeans(
                            n_clusters=k, random_state=seed, n_init=10))):
        row = []
        for k in (2, 3, 4, 5):
            m = fit(k).fit(Xe)
            a_e, a_l = m.predict(Xe), m.predict(Xl)
            agree = float((a_e == a_l).mean())
            # chance agreement given the early label distribution
            p = np.bincount(a_e, minlength=k) / len(a_e)
            chance = float((p ** 2).sum())
            row.append(f"k={k}: {agree:.0%} (chance {chance:.0%})")
        print(f"     {name:<7} same cluster in both eras   " + "   ".join(row))

    h = HDBSCAN(min_cluster_size=max(5, len(both) // 10)).fit(Xe)
    lab = h.labels_
    n_cl = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"     HDBSCAN finds {n_cl} cluster(s) on the early era, "
          f"{(lab == -1).mean():.0%} of codes unassigned (noise)")


if __name__ == "__main__":
    raise SystemExit(main())
