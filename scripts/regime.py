#!/usr/bin/env python3
"""H60 — the regime engine (§7 / CLAUDE.md §10), paired with a method that can
say "there are none".

CLAUDE.md §10 IS EXPLICIT ABOUT WHAT MACRO IS FOR HERE. Not prediction —
segmentation: *"in which regimes does the signal work, and in which does it
invert?"* And it is equally explicit about the danger: with ~26 years and a
handful of switches there are very few independent observations, so "any regime
split producing beautiful results on three episodes is almost certainly
overfit."

A10 ADDS THE METHOD RULE AND IT IS THE ONE THAT MATTERS. Clustering broker
fingerprints, HDBSCAN found ZERO clusters and labelled 100% of codes noise,
while GMM and k-means — forced to return k — produced partitions that decayed
to chance by k=5. *A method that must return k clusters cannot tell you there
are none.* So both are run here, and HDBSCAN's answer is reported first
whatever it is.

--------------------------------------------------------------------- REGISTERED
Written before any cell was scored.

  G1  PREDICTED NULL. HDBSCAN on standardised market-state features finds no
      stable cluster structure — fewer than two clusters, or a noise share
      above 50%. PREDICTION: confirmed. A10 found none on a far wider
      cross-section (89 brokers); here the cross-section is ONE market observed
      repeatedly, which is a harder case, not an easier one.

  G2  PREDICTED NULL. H26's strength+calm edge does not differ across regimes
      beyond its own permutation null. PREDICTION: confirmed. If it FAILS —
      i.e. the edge really does invert somewhere — that is CLAUDE.md §10's
      question answered in the affirmative and the most valuable result this
      script could produce, so the failure mode is the interesting one.

  G3  The binding constraint is the number of independent REGIME EPISODES, not
      the number of days. PREDICTION: fewer than 25 episodes over 24 years, so
      any per-regime statistic rests on single digits of independent
      observations and the power statement says so before the effect does.

WHAT IS DELIBERATELY EXCLUDED. No macro variable is used to predict a return
anywhere in this file. §10's job for macro is conditioning, and A13 already
measured the predictive version: the strongest overnight link to the IHSG
explains about 4% of variance against a 56 bp round trip.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from bhbench import load                                          # noqa: E402
from idxbot.config import Config                                  # noqa: E402
from idxbot.data.cache import Cache                               # noqa: E402
from idxbot.metrics import periods_to_detect                      # noqa: E402

TRIALS = 357          # hypotheses.md after H59
BAR = 0.05 / TRIALS

#: Market-state features. Every one is BACKWARD-LOOKING at the bar it is
#: stamped on: a regime label that used tomorrow's volatility would segment the
#: sample by the outcome it is meant to condition.
MACRO = {
    "dxy": "DX-Y.NYB",
    "usdidr": "IDR=X",
    "ust10": "^TNX",
    "spx": "^GSPC",
    "brent": "BZ=F",
    "copper": "HG=F",
}
#: `^TNX` is a RATE, not a price (A13): it fell 0.93 -> 0.50 in March 2020, a
#: real 43 bp move and a spurious -46% "return". Rates are DIFFERENCED.
RATES = {"ust10"}


def _series(name: str, cache: Cache) -> Optional[pd.Series]:
    key = name if name.startswith("^") is False else name
    d = cache.read("ohlcv", key.replace("^", "_"))
    if d is None or d.empty:
        d = cache.read("ohlcv", key)
    if d is None or d.empty:
        return None
    d = d.dropna(subset=["close"])
    return pd.Series(d["close"].to_numpy(float),
                     index=pd.to_datetime(d["date"])).sort_index()


def features(P: pd.DataFrame, cache: Cache) -> pd.DataFrame:
    """One row per session: the market's own state, plus what it woke up to."""
    idx = _series("^JKSE", cache)
    if idx is None:
        raise SystemExit("^JKSE not cached — run scripts/refresh.py first")
    r = np.log(idx).diff()
    F = pd.DataFrame(index=idx.index)
    F["ihsg_vol"] = r.rolling(60, min_periods=40).std() * np.sqrt(252)
    F["ihsg_trend"] = np.log(idx / idx.rolling(200, min_periods=150).mean())
    F["ihsg_dd"] = np.log(idx / idx.cummax())

    #  Breadth from the panel, grouped BY TICKER. A11: a rolling window on a
    #  date x ticker pivot is indexed by the UNION of trading days, so one
    #  suspended name inserts NaN rows it never had and `min_periods` fails for
    #  every column at once.
    Q = P[["date", "ticker", "adj_close"]].sort_values(["ticker", "date"])
    Q["ma200"] = Q.groupby("ticker")["adj_close"].transform(
        lambda s: s.rolling(200, min_periods=150).mean())
    Q["above"] = (Q["adj_close"] > Q["ma200"]).where(Q["ma200"].notna())
    br = Q.groupby("date")["above"].mean()
    F["breadth"] = br.reindex(F.index)

    for name, sym in MACRO.items():
        s = _series(sym, cache)
        if s is None:
            continue
        s = s.reindex(F.index).ffill(limit=5)
        if name in RATES:
            F[f"{name}_chg"] = s.diff(60)
        else:
            F[f"{name}_mom"] = np.log(s / s.shift(60))
        if name in ("usdidr", "spx"):
            F[f"{name}_vol"] = (np.log(s).diff()
                                .rolling(60, min_periods=40).std()
                                * np.sqrt(252))
    return F.dropna()


def standardise(F: pd.DataFrame) -> np.ndarray:
    X = F.to_numpy(float)
    mu, sd = X.mean(0), X.std(0, ddof=1)
    sd[sd <= 0] = 1.0
    return (X - mu) / sd


def episodes(labels: np.ndarray) -> int:
    """Contiguous runs of one label. THE UNIT OF INDEPENDENCE, not the day.

    6,000 days of two regimes is not 6,000 observations of a regime effect; it
    is however many times the market actually switched. G3 turns on this and it
    is why it is a function rather than a remark.
    """
    if len(labels) == 0:
        return 0
    return int(1 + np.sum(labels[1:] != labels[:-1]))


def hdbscan_answer(X: np.ndarray, min_size: int) -> Dict[str, object]:
    from sklearn.cluster import HDBSCAN                      # noqa: PLC0415
    m = HDBSCAN(min_cluster_size=min_size, min_samples=max(min_size // 4, 5),
                copy=True)
    lab = m.fit_predict(X)
    k = len(set(lab[lab >= 0]))
    return {"k": k, "noise": float(np.mean(lab < 0)), "labels": lab}


def gmm_answer(X: np.ndarray, k: int, seed: int = 0) -> Dict[str, object]:
    from sklearn.mixture import GaussianMixture              # noqa: PLC0415
    g = GaussianMixture(n_components=k, covariance_type="full",
                        random_state=seed, n_init=3).fit(X)
    lab = g.predict(X)
    return {"k": k, "labels": lab, "bic": float(g.bic(X)),
            "episodes": episodes(lab)}


def screen_frame(P: pd.DataFrame, horizon: int = 63) -> pd.DataFrame:
    """H26's screen and the forward mean-log return, computed ONCE.

    The statistic is the mean LOG forward return of screen minus rest, because
    A36 measured the arithmetic mean and the mean log disagreeing in SIGN on
    this repo's data and H57 found them disagreeing about whether an effect
    exists at all.

    Built once and reused by every null draw. A first version rebuilt it inside
    the draw loop -- two groupby-transforms over 2.8m rows, two hundred times.
    Correct and unusable, which is its own kind of wrong: a null nobody can
    afford to run at enough draws is a null that gets quietly dropped.
    """
    F = P[P["elig"]].copy()
    F["fwd"] = (F.groupby("ticker")["adj_close"].shift(-horizon)
                / F["adj_close"] - 1.0)
    F["lf"] = np.log1p(F["fwd"].clip(lower=-0.999))
    hi = F.groupby("date")["hi52"].transform(lambda s: s.quantile(0.90))
    vo = F.groupby("date")["vol60"].transform(lambda s: s.quantile(0.50))
    F["screen"] = (F["hi52"] >= hi) & (F["vol60"] <= vo)
    return F.dropna(subset=["lf"])[["date", "ticker", "lf", "screen"]]


def edge_by_regime(S: pd.DataFrame, lab: pd.Series) -> pd.DataFrame:
    """Screen-minus-rest mean log, separately inside each regime."""
    reg = S["date"].map(lab)
    ok = reg.notna().to_numpy()
    r = reg.to_numpy()
    lf = S["lf"].to_numpy(float)
    sc = S["screen"].to_numpy(bool)
    out = []
    for g in np.unique(r[ok]):
        m = ok & (r == g)
        s, o = lf[m & sc], lf[m & ~sc]
        if len(s) < 300 or len(o) < 300:
            out.append({"regime": g, "n": len(s), "edge": np.nan,
                        "why": "insufficient"})
            continue
        out.append({"regime": g, "n": int(len(s)),
                    "days": int(S.loc[m, "date"].nunique()),
                    "screen": float(s.mean()), "rest": float(o.mean()),
                    "edge": float(s.mean() - o.mean())})
    return pd.DataFrame(out)


def regime_null(S: pd.DataFrame, lab: pd.Series, draws: int, seed: int = 0
                ) -> Tuple[float, float, int]:
    """Null for "the edge differs by regime": rotate the regime LABEL SERIES.

    A CIRCULAR SHIFT, NOT A SHUFFLE, AND THE REASON IS G3. Regimes are long
    contiguous runs; a day-level shuffle destroys that structure entirely and
    produces a null of tiny, evenly-mixed pseudo-regimes whose edge spread is
    far too small. Rotating the whole series keeps every run length and every
    transition exactly, and only breaks the alignment with the market -- which
    is the thing being tested. A17 and A34 record the same class of error from
    two directions.
    """
    rng = np.random.default_rng(seed)
    v = lab.to_numpy()
    n = len(v)
    out = []
    for _ in range(draws):
        k = int(rng.integers(1, n))
        rot = pd.Series(np.roll(v, k), index=lab.index)
        e = edge_by_regime(S, rot)["edge"].dropna()
        if len(e) >= 2:
            out.append(float(e.max() - e.min()))
    a = np.asarray(out)
    return (float(a.mean()), float(a.std(ddof=1)), len(a)) if len(a) > 1 \
        else (float("nan"), float("nan"), len(a))


def _episode_table(lab: pd.Series) -> pd.DataFrame:
    """One row per contiguous run: regime, first date, last date, length."""
    v = lab.to_numpy()
    idx = lab.index
    cuts = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])
    ends = np.r_[cuts[1:], len(v)]
    return pd.DataFrame({
        "regime": v[cuts],
        "start": idx[cuts],
        "end": idx[ends - 1],
        "days": ends - cuts,
    })


def _drop_largest_episode(S: pd.DataFrame, lab: pd.Series,
                          runs: pd.DataFrame, t: pd.DataFrame):
    """Recompute the spread with the longest episode of the EXTREME regime cut.

    The extreme regime is the one holding the max or min edge, i.e. the one
    the spread is made of. If one episode carries it, the spread is one
    historical event with a label on it.
    """
    e = t.dropna(subset=["edge"])
    if len(e) < 2:
        return None
    hot = int(e.loc[e["edge"].idxmin(), "regime"])
    own = runs[runs["regime"] == hot]
    if own.empty:
        return None
    big = own.loc[own["days"].idxmax()]
    keep = ~((lab.index >= big["start"]) & (lab.index <= big["end"]))
    lab2 = lab[keep]
    e2 = edge_by_regime(S, lab2)["edge"].dropna()
    if len(e2) < 2:
        return {"what": f"regime {hot} {pd.Timestamp(big['start']).date()}"
                        f"->{pd.Timestamp(big['end']).date()}",
                "spread": float("nan")}
    return {"what": f"regime {hot} {pd.Timestamp(big['start']).date()}"
                    f"->{pd.Timestamp(big['end']).date()}, "
                    f"{int(big['days'])}d",
            "spread": float(e2.max() - e2.min())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=63)
    a = ap.parse_args()

    cfg = Config()
    cache = Cache(cfg.path("data.cache_dir", "data/cache"))
    P = load()
    F = features(P, cache)
    X = standardise(F)
    print("H60 — regime engine (§7 / CLAUDE.md §10)")
    print(f"{len(F):,} sessions {F.index.min().date()} -> "
          f"{F.index.max().date()}, {F.shape[1]} state features")
    print(f"features: {', '.join(F.columns)}")

    # ------------------------------------------------------------------ G1
    print("\nG1  HDBSCAN — the method that CAN return 'there are none'")
    for frac in (0.02, 0.05, 0.10):
        ms = max(int(len(X) * frac), 20)
        h = hdbscan_answer(X, ms)
        print(f"    min_cluster_size {ms:>5} ({frac:.0%})  clusters {h['k']}  "
              f"noise {h['noise']:.1%}  episodes {episodes(h['labels'])}")
    h = hdbscan_answer(X, max(int(len(X) * 0.05), 20))
    g1_null = (h["k"] < 2) or (h["noise"] > 0.5)
    print(f"    G1 {'CONFIRMED — no stable structure' if g1_null else 'FAILED — HDBSCAN found structure'}")

    print("\n    GMM forced to k, for the contrast A10 insists on")
    print(f"    {'k':>3}{'BIC':>14}{'episodes':>10}{'days/episode':>14}")
    fits = {}
    for k in range(2, a.kmax + 1):
        g = gmm_answer(X, k)
        fits[k] = g
        print(f"    {k:>3}{g['bic']:>14,.0f}{g['episodes']:>10}"
              f"{len(X) / max(g['episodes'], 1):>14,.0f}")
    kbest = min(fits, key=lambda k: fits[k]["bic"])
    print(f"    lowest BIC at k={kbest} — and BIC on 6,000 AUTOCORRELATED days")
    print("    is not a model-selection statistic, because it counts each day")
    print("    as an independent observation. It is printed, not obeyed.")

    # ------------------------------------------------------------------ G3
    print("\nG3  the unit of independence")
    for k in (2, 3, kbest):
        ep = fits[k]["episodes"] if k in fits else gmm_answer(X, k)["episodes"]
        print(f"    k={k}: {ep} contiguous episodes over "
              f"{(F.index.max() - F.index.min()).days / 365.25:.1f} years "
              f"= {ep / max((F.index.max() - F.index.min()).days / 365.25, 1):.1f}/yr")
    ep2 = fits[2]["episodes"]
    print(f"    G3 {'CONFIRMED' if ep2 < 25 else 'FAILED'}: k=2 gives {ep2} "
          f"episodes, and an episode is the observation.")

    # ------------------------------------------------------------------ G2
    print(f"\nG2  does H26's edge differ by regime? "
          f"(mean log, {a.horizon}-session fwd)")
    print("    EVERY k is reported, not the one that fired. Trying two values")
    print("    and quoting the louder is a search of size two.")
    S = screen_frame(P, a.horizon)
    fired = []
    _drops_all = []
    for k in range(2, a.kmax + 1):
        lab = pd.Series(fits[k]["labels"], index=F.index)
        t = edge_by_regime(S, lab)
        print(f"\n    k={k}")
        runs = _episode_table(lab)
        for _, x in t.iterrows():
            g = int(x["regime"])
            ne = int((runs["regime"] == g).sum())
            if not np.isfinite(x.get("edge", np.nan)):
                print(f"      regime {g}  n={x['n']:,}  {x.get('why')}")
                continue
            print(f"      regime {g}  {x['days']:>5,} days in {ne:>3} episodes"
                  f"  n={x['n']:>7,}  screen {x['screen']:+.4f}  "
                  f"rest {x['rest']:+.4f}  edge {x['edge']:+.4f}")
        e = t["edge"].dropna()
        if len(e) < 2:
            print("      too few readable regimes to compare")
            continue
        spread = float(e.max() - e.min())
        mu, sd, nd = regime_null(S, lab, a.draws)
        z = (spread - mu) / sd if sd and sd > 0 else float("nan")
        inv = bool((e > 0).any() and (e < 0).any())
        print(f"      SPREAD {spread:+.4f}   rotation null {mu:+.4f} +- {sd:.4f}"
              f" ({nd})   z {z:+.2f}   sign inverts: {inv}")

        #  DROP THE LARGEST CONTRIBUTING EPISODE. A8 introduced this check
        #  because H11's headline was carried by one thin year, and records it
        #  earning its place twice. A regime cell built from three episodes is
        #  precisely the case CLAUDE.md §10 warns about by name.
        drop = _drop_largest_episode(S, lab, runs, t)
        _drops_all.append((drop, inv))
        if drop is not None:
            print(f"      drop the single largest episode "
                  f"({drop['what']}): spread {drop['spread']:+.4f} "
                  f"({drop['spread'] - spread:+.4f})")
        if abs(z) > 3.63:
            fired.append((k, spread, z, inv, drop))
            print("      *** clears the Bonferroni bar")
        else:
            print(f"      does NOT clear the {BAR:.5f} bar (needs |z| > 3.63)")

    #  IS IT THE SAME EPISODE EVERY TIME? If the inverting regime resolves to
    #  one historical window regardless of k, then "regime" is a label on that
    #  window and the k sweep is not five pieces of evidence but one.
    #  ONLY the k values where the sign actually inverts. At k=2 no regime is
    #  negative, so the "extreme" regime is just the larger one and its longest
    #  episode is a decade of ordinary market -- including it would mix a
    #  meaningful window with a degenerate one and hide the answer.
    windows = [d["what"] for d, inv in _drops_all if d and inv]
    if windows:
        yrs = sorted({w.split()[2][:4] for w in windows if len(w.split()) > 2})
        print(f"\n    THE EXTREME REGIME'S LARGEST EPISODE, at every k tried: "
              f"{', '.join(yrs)}")
        if len(yrs) == 1:
            print("    One window, five clusterings. The 'regime' effect is a")
            print(f"    label on {yrs[0]}, and dropping that one episode is")
            print("    what moves the spread — not a property of regimes in")
            print("    general.")

    print("\n    G2 VERDICT")
    if not fired:
        print("    No k clears the bar. PREDICTED NULL CONFIRMED: H26's edge")
        print("    does not measurably differ by regime.")
    else:
        ks = ", ".join(f"k={k}" for k, *_ in fired)
        print(f"    {len(fired)} of {a.kmax - 1} cluster counts clear the bar "
              f"({ks}).")
        print("    G2 FAILED — which is the outcome registered as the valuable")
        print("    one, and it is also the one that most needs qualifying:")
        worst = min(fired, key=lambda r: r[1])
        for k, spread, z, inv, drop in fired:
            note = ""
            if drop is not None:
                note = (f"; dropping its largest episode moves the spread "
                        f"{drop['spread'] - spread:+.4f}")
            print(f"      k={k}: spread {spread:+.4f}, z {z:+.2f}, "
                  f"sign inverts {inv}{note}")
        print("    An episode is the observation (G3), so a cell built from a")
        print("    handful of them is exactly what CLAUDE.md §10 warns about")
        print("    by name. This is a REGISTERED FAILURE, not a tradeable")
        print("    finding, and no arm is built on it here.")


if __name__ == "__main__":
    main()
