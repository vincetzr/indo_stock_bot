#!/usr/bin/env python3
"""H23 — what horizon makes a high multi-bagger hit rate reachable, if any.

THE UNTESTED LEVER. Every P(2x) in this repo — H13, H16, H18, H21 — is
measured at 252 sessions, because that is the horizon H16 happened to pick.
A doubling rate is mostly a function of how long you wait, and CLAUDE.md A9
named "a horizon long enough that turnover stops mattering" as untested and
then never tested it. So the question "can 8 of 10 double?" has only ever been
asked at one year.

TWO DEFINITIONS OF A HIT, AND THE GAP BETWEEN THEM IS THE WHOLE POINT.

  touched 2x   the path reached 2x at some point in the window
  ended 2x     the terminal price is 2x

H16 measured a mean PEAK of +102.2% against a realised +15.1%, so these two
are very far apart, and the gap is exactly what a take-profit order captures.
A buy-and-hold study can only ever see the second one.

THREE BENCHMARKS, because a hit rate without them is meaningless:
  * the same statistic on the whole liquid universe (is selection doing
    anything?)
  * the same statistic on the IHSG (if the index doubles over the window,
    "8 of 10 doubled" is not an achievement, it is a rising tide)
  * the median and P(-50%), because a rule can raise P(2x) and still lose
    money — H21 measured exactly that.

EFFECTIVE SAMPLE SIZE IS THE BINDING PROBLEM AT LONG HORIZONS and it is
reported, not buried: a 10-year window over a 22-year panel gives about two
independent observations per name, and overlapping cohorts do not add
information no matter how many rows they produce.
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

from idxbot.spine import multiplier as MU                        # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")
INDEX = os.path.join("data", "cache", "ohlcv", "_JKSE.csv.gz")
CACHE = os.path.join("data", "spine", "horizon_sweep.parquet")

#: Sessions. ~252 to the year, so: 1, 2, 3, 5, 7.5 and 10 years.
HORIZONS = [252, 504, 756, 1260, 1890, 2520]


def label(k: int) -> str:
    return f"{k / 252:.1f}y".replace(".0y", "y")


def forward(g: pd.DataFrame, k: int) -> pd.DataFrame:
    """Forward peak and forward terminal over the next `k` bars, per ticker.

    A name that stops trading inside the window keeps whatever bars it has and
    is marked at its LAST TRADED PRICE, which is optimistic — a delisting is
    usually worth far less than its last print. The truncation rate is counted
    and reported so the size of that optimism is visible rather than assumed.
    """
    p = g["adj_close"].to_numpy(float)
    n = len(p)
    #  reversed rolling max gives the forward peak in one pass
    peak = pd.Series(p[::-1]).rolling(k + 1, min_periods=1).max()[::-1]
    peak = peak.to_numpy()
    idx = np.minimum(np.arange(n) + k, n - 1)
    return pd.DataFrame({
        "date": g["date"].to_numpy(),
        "ticker": g["ticker"].to_numpy(),
        "peak": peak / p,
        "end": p[idx] / p,
        #  bars actually available: short means the name died or the panel did
        "bars": idx - np.arange(n),
    })


def build(P: pd.DataFrame) -> pd.DataFrame:
    """Forward peak/terminal at every horizon, monthly cohorts.

    TWO BUGS THE FIRST VERSION HAD, both of which invalidated the table.

    `MU.PX` is a list of CUT EDGES for the price bucket, not a [min, max]
    pair, so `close >= PX[0] & close <= PX[1]` restricted the universe to
    names under Rp50 — the penny board — and the whole sweep was measured on
    336 sub-Rp50 tickers. Eligibility in `rank_live` is `MIN_VALUE` on traded
    value and there is no price restriction at all.

    And the filter was applied to EVERY bar, so a name dropping out of the
    universe mid-hold had its forward path cut there. Eligibility is a
    condition for BUYING; once held, the path is whatever the name does. It
    is now applied at the cohort date only.
    """
    P = P.sort_values(["ticker", "date"])
    P = P[P["adj_close"] > 0]
    out = None
    for k in HORIZONS:
        F = pd.concat([forward(g, k) for _, g in P.groupby("ticker",
                                                           sort=False)],
                      ignore_index=True)
        F = F.rename(columns={"peak": f"peak{k}", "end": f"end{k}",
                              "bars": f"bars{k}"})
        out = F if out is None else out.merge(F, on=["date", "ticker"],
                                              how="outer")
    want = ["date", "ticker", "close", "holdout", "tradeable",
            "log_turnover"] + FEATURES
    keep = P[list(dict.fromkeys(c for c in want if c in P.columns))]
    out = out.merge(keep, on=["date", "ticker"],
                    how="inner").reset_index(drop=True)
    #  ENTRY eligibility only: tradeable and above the traded-value floor
    out = out[out["tradeable"].astype(bool)]
    out = out[out["log_turnover"] >= np.log(MU.MIN_VALUE)]
    #  when did this ticker last trade at all, and when did the panel end
    last = P.groupby("ticker")["date"].max().rename("last_bar")
    out = out.merge(last, on="ticker", how="left").reset_index(drop=True)
    out["panel_end"] = P["date"].max()
    #  monthly cohorts: first session of each month, which is what every other
    #  study here uses, so the sample is comparable
    out["ym"] = out["date"].dt.to_period("M")
    first = out.groupby("ym")["date"].transform("min")
    return out[out["date"] == first].drop(columns="ym")


def index_stats(k: int) -> Dict:
    ix = pd.read_csv(INDEX)
    ix["date"] = pd.to_datetime(ix["date"], utc=True,
                                errors="coerce").dt.tz_localize(None)
    s = ix.set_index("date")["close"].astype(float).sort_index().dropna()
    s = s[(s > 0) & (s.index >= "2004-01-01")]
    p = s.to_numpy()
    n = len(p)
    peak = pd.Series(p[::-1]).rolling(k + 1, min_periods=1).max()[::-1]
    idx = np.minimum(np.arange(n) + k, n - 1)
    ok = (np.arange(n) + k) < n
    return {"touch": float((peak.to_numpy()[ok] / p[ok] >= 2.0).mean()),
            "end": float((p[idx][ok] / p[ok] >= 2.0).mean())}


def classify(D: pd.DataFrame, k: int) -> pd.DataFrame:
    """Split windows into full, died-mid-window, and censored-by-the-panel.

    THE DISTINCTION IS THE WHOLE STUDY AT LONG HORIZONS. Requiring `bars >= k`
    keeps only names that traded for the full window, which at 7.5 years threw
    away 91% of cohorts and measured the doubling rate of the SURVIVORS. That
    is not a bias worth a footnote; it is the answer.

      full      the name traded every bar of the window
      died      the name stopped trading before the window closed, and long
                enough before the panel ends that the stop is real
      censored  the window simply runs past the end of the data — no outcome
                exists yet, and these must be dropped, not filled
    """
    d = D.copy()
    full = d[f"bars{k}"] >= k
    #  a name whose last bar is within ~1.5 calendar years of the panel end is
    #  probably still alive; only an earlier stop counts as a death
    alive_at_end = d["last_bar"] >= (d["panel_end"] - pd.Timedelta(days=400))
    died = (~full) & (~alive_at_end) & (d[f"bars{k}"] > 20)
    censored = (~full) & (~died)
    d["cls"] = np.where(full, "full", np.where(died, "died", "censored"))
    return d


def summarise(D: pd.DataFrame, k: int, mask=None,
              deaths: str = "last", min_n: int = 60) -> Dict:
    """Doubling and loss rates at horizon `k`.

    `deaths` decides how a name that stopped trading mid-window is valued:
    "last" marks it at its final traded price, which is optimistic because a
    delisting is usually worth far less than its last print; "zero" writes it
    to a total loss, which is pessimistic. The two bracket the truth and both
    are printed, because at long horizons the gap between them is larger than
    any effect being measured. "drop" reproduces the survivors-only figure
    that the first version of this script reported by accident.
    """
    d = D if mask is None else D[mask]
    d = classify(d, k)
    d = d[d["cls"] != "censored"]
    if deaths == "drop":
        d = d[d["cls"] == "full"]
    if len(d) < min_n:
        return {}
    e = d[f"end{k}"].to_numpy(float).copy()
    pk = d[f"peak{k}"].to_numpy(float).copy()
    if deaths == "zero":
        dead = (d["cls"] == "died").to_numpy()
        e[dead] = 0.0                       # terminal is a write-off...
        #  ...but the PEAK still happened; a name that trebled and then died
        #  did touch 2x, and a take-profit order would have filled.
    #  effective n: overlapping monthly cohorts over a k-bar window share
    #  almost all of their path, so divide by the overlap factor
    eff = len(e) / max(k / 21.0, 1.0)
    return {"n": len(e), "eff_n": eff,
            "died": int((d["cls"] == "died").sum()),
            "touch2x": float((pk >= 2.0).mean()),
            "end2x": float((e >= 2.0).mean()),
            "profit": float((e > 1.0).mean()),
            "half": float((e <= 0.5).mean()),
            "median": float(np.median(e) - 1.0),
            "mean": float(np.mean(e) - 1.0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    if a.report and os.path.exists(CACHE):
        D = pd.read_parquet(CACHE)
        print(f" [reusing {CACHE}]")
    else:
        cols = ["date", "ticker", "close", "adj_close", "tradeable",
                "holdout", "log_turnover"] + FEATURES
        P = pd.read_parquet(PANEL, columns=sorted(set(cols)))
        P["date"] = pd.to_datetime(P["date"])
        D = build(P)
        D.to_parquet(CACHE, index=False)
        print(f" [cached to {CACHE}]")

    D = D[~D["holdout"].astype(bool)]
    liq = D["log_turnover"] >= D.groupby("date")["log_turnover"].transform(
        lambda s: s.quantile(0.5))

    W = 104
    print("=" * W)
    print(" H23 — DOES A LONGER HORIZON MAKE A HIGH DOUBLING RATE REACHABLE?")
    print("=" * W)
    print(f" {len(D):,} monthly name-cohorts, {D['ticker'].nunique()} names, "
          f"{D['date'].min().date()} .. {D['date'].max().date()}")
    print(" 'touched 2x' = the path reached 2x at any point (what a")
    print(" take-profit order captures). 'ended 2x' = buy-and-hold.\n")

    print(f"   {'horizon':<8}{'deaths':<8}{'n':>7}{'eff n':>7}{'died':>7}"
          f"{'touched 2x':>12}{'ended 2x':>10}{'profit':>8}{'P(-50%)':>9}"
          f"{'median':>9}{'IHSG 2x':>9}")
    for k in HORIZONS:
        ix = index_stats(k)
        for how, lbl in (("drop", "survivors"), ("last", "at last"),
                         ("zero", "at zero")):
            s = summarise(D, k, liq, deaths=how)
            if not s:
                continue
            print(f"   {label(k) if how == 'drop' else '':<8}{lbl:<8}"
                  f"{s['n']:>7,}{s['eff_n']:>7,.0f}{s['died']:>7,}"
                  f"{s['touch2x']:>12.1%}{s['end2x']:>10.1%}"
                  f"{s['profit']:>8.1%}{s['half']:>9.1%}{s['median']:>+9.1%}"
                  f"{ix['end']:>9.1%}")
        print()

    print("   'survivors' is what the first version of this script reported")
    print("   by accident: it requires the name to trade every bar of the")
    print("   window, which at 7.5 years keeps 9% of cohorts and measures the")
    print("   doubling rate of the ones that lived. 'at last' and 'at zero'")
    print("   bracket the truth by valuing a dead name at its final print or")
    print("   at nothing. A name that trebled and then died still TOUCHED 2x")
    print("   under both, because a take-profit order would have filled.")
    print("\n   THE GAP between 'touched' and 'ended' is what a take-profit")
    print("   order could capture, and it is the largest number in the table.")
    print("   IHSG 2x is the tide: a hit rate near it is the index in a hat.")

    print("\n" + "=" * W)
    print(" HOW CLOSE DOES ANYTHING GET TO 8 OF 10 (80%)?")
    print("=" * W)
    print(f"\n   {'horizon':<9}{'best touched 2x':>18}{'shortfall to 80%':>20}"
          f"{'what it would need':>22}")
    for k in HORIZONS:
        s = summarise(D, k, liq)
        if not s:
            continue
        #  the multiplier rule's measured lift on P(2x), 4.89 / 1.51 = 3.24x,
        #  applied as an UPPER BOUND to the unconditional touch rate
        lift = 3.24
        best = min(s["touch2x"] * lift, 1.0)
        need = 0.80 / max(s["touch2x"], 1e-9)
        print(f"   {label(k):<9}{best:>18.1%}{0.80 - best:>+20.1%}"
              f"{need:>18.1f}x lift")
    print("\n   'best' applies the LARGEST selection lift this project has")
    print("   ever measured (3.24x, H21's liquid re-rank) to the base rate.")
    print("   It is an upper bound and not an achieved result — that rule")
    print("   LOST money while delivering that lift.")

    print("\n" + "=" * W)
    print(" SO: DOES ANY REGISTERED SIGNAL DELIVER THAT LIFT AT THAT HORIZON?")
    print("=" * W)
    print(" The 1-year lift is not transferable — it has to be measured where")
    print(" it would be used. Top and bottom decile of every registered")
    print(" feature, `squeeze` kept as H13's predicted-null control.\n")
    for k in (756, 2520):
        rows = feature_lift(D, k, liq)
        if not rows:
            continue
        d = classify(D[liq], k)
        base = float((d[d["cls"] != "censored"][f"peak{k}"] >= 2.0).mean())
        need = 0.80 / max(base, 1e-9)
        print(f"   --- {label(k)}: base touch {base:.1%}, "
              f"80% needs {need:.2f}x ---")
        print(f"   {'feature':<14}{'side':<5}{'n':>7}{'touched 2x':>12}"
              f"{'lift':>7}{'P(-50%)':>9}{'median':>9}{'enough?':>9}")
        for r in rows[:6]:
            print(f"   {r['feature']:<14}{r['side']:<5}{r['n']:>7,}"
                  f"{r['touch']:>12.1%}{r['lift']:>7.2f}{r['half']:>9.1%}"
                  f"{r['median']:>+9.1%}"
                  f"{('YES' if r['lift'] >= need else 'no'):>9}")
        print()

    print("=" * W)
    print(" THE ONE CANDIDATE, TESTED PROPERLY")
    print("=" * W)
    print(" `log_turnover` top decile at 10y is the max of ~22 cells, so it")
    print(" gets the permutation null and the half-split rather than the")
    print(" headline. Null permutes whole (ticker, year) BLOCKS: a name")
    print(" contributes ~12 near-identical monthly cohorts a year and a row")
    print(" shuffle would leave the null far too tight (A17).\n")
    v = validate_liquid(D[liq], 2520)
    if v:
        print(f"   base touch rate         {v['base']:>8.1%}")
        print(f"   liquid decile           {v['obs']:>8.1%}"
              f"   lift {v['lift']:.2f}x   (n = {v['n_sel']:,})")
        print(f"   permutation null        {v['null_mean']:>8.1%}"
              f"   sd {v['null_sd']:.1%}   p95 {v['null_p95']:.1%}")
        print(f"   z = {v['z']:+.2f}   empirical p = {v['p_emp']:.3f}")
        print(f"   effective n (whole sample, not the decile): "
              f"{v['eff_n']:.0f}")
        print(f"\n   {'half':<8}{'base':>9}{'liquid decile':>16}{'lift':>8}"
              f"{'n':>8}")
        for nm, h in v["halves"].items():
            lf = h["sel"] / max(h["base"], 1e-9)
            print(f"   {nm:<8}{h['base']:>9.1%}{h['sel']:>16.1%}"
                  f"{lf:>8.2f}{h['n']:>8,}")
        both = all(h["sel"] > h["base"] for h in v["halves"].values())
        print(f"\n   positive in BOTH halves: {'YES' if both else 'no'}")
    print("\n   A 10-year window over a 24-year panel is about two")
    print("   independent observations per name. Overlapping cohorts produce")
    print("   rows, not information, and no null can manufacture the")
    print("   independent samples the panel does not contain.")
    return 0




# ==========================================================================
# Does ANY selection lift the touch rate at a long horizon?
# ==========================================================================
#: Every registered price feature, plus the two raw risk axes. `squeeze` is
#: the predicted-null from H13 and stays in for the same reason it did there:
#: a sweep with no negative control cannot tell signal from pipeline.
FEATURES = ["mom12_1", "rev1", "rev5", "lowvol", "amihud60", "volz20",
            "hi52", "atr_mom20", "squeeze", "vol60", "log_turnover"]


def feature_lift(D: pd.DataFrame, k: int, mask, q: float = 0.10) -> List[Dict]:
    """Top- and bottom-decile touch rate for every feature, as a lift.

    The question this answers is narrow and exact: 80% of a basket touching 2x
    needs the base rate multiplied by 80/base. This measures what multiple any
    registered signal actually delivers AT THAT HORIZON, rather than assuming
    the 1-year lift carries.
    """
    d = classify(D[mask], k)
    d = d[d["cls"] != "censored"]
    if len(d) < 500:
        return []
    base = float((d[f"peak{k}"] >= 2.0).mean())
    out = []
    for f in FEATURES:
        if f not in d.columns or d[f].notna().sum() < 500:
            continue
        r = d.groupby("date")[f].rank(pct=True)
        for side, m in (("top", r >= 1 - q), ("bot", r <= q)):
            sub = d[m.reindex(d.index).fillna(False)]
            if len(sub) < 300:
                continue
            t = float((sub[f"peak{k}"] >= 2.0).mean())
            out.append({"feature": f, "side": side, "n": len(sub),
                        "touch": t, "lift": t / max(base, 1e-9),
                        "half": float((sub[f"end{k}"] <= 0.5).mean()),
                        "median": float(sub[f"end{k}"].median() - 1.0)})
    return sorted(out, key=lambda r: -r["lift"])




# ==========================================================================
# The one candidate, tested properly
# ==========================================================================
def validate_liquid(D: pd.DataFrame, k: int, feature: str = "log_turnover",
                    side: str = "top", q: float = 0.10,
                    draws: int = 200, seed: int = 11) -> Dict:
    """Half-split, clustered permutation null, and the index over the SAME
    windows, for the single best long-horizon cell.

    WHY THIS CELL AND NOT A BETTER ONE. It is the maximum of ~22 (feature x
    side) cells, which is exactly the post-hoc trap this repo's trial count
    exists to catch, so it gets the treatment rather than the headline. It is
    kept in preference to the marginally higher `amihud60 bot` because the two
    are the same axis — both say "the most liquid names" — and `log_turnover`
    is the one with a prior mechanism already established here: A19 found the
    cap-weighted index beat equal-weighted baskets because a handful of
    mega-caps carried it. This is that finding from the other side, not an
    independent discovery, and it should not be reported as one.

    THE NULL PERMUTES WHOLE (ticker, year) BLOCKS, not rows. A name
    contributes ~12 near-identical monthly cohorts a year over a 10-year
    window; a row shuffle destroys the label while leaving the null far too
    tight, which A17 records inflating a z to -8.7 before it became a
    headline.
    """
    d = classify(D, k)
    d = d[d["cls"] != "censored"].copy()
    if len(d) < 500 or feature not in d.columns:
        return {}
    d["hit"] = (d[f"peak{k}"] >= 2.0).astype(float)
    r = d.groupby("date")[feature].rank(pct=True)
    d["sel"] = (r >= 1 - q) if side == "top" else (r <= q)
    base = float(d["hit"].mean())
    obs = float(d.loc[d["sel"], "hit"].mean())

    #  clustered permutation: shuffle the selection flag within (ticker, year)
    #  blocks, preserving how often each name is picked and when
    rng = np.random.default_rng(seed)
    d["blk"] = (d["ticker"].astype(str) + "|" +
                d["date"].dt.year.astype(str))
    blocks = d.groupby("blk").indices
    keys = list(blocks)
    null = []
    sel = d["sel"].to_numpy()
    hit = d["hit"].to_numpy()
    for _ in range(draws):
        perm = np.zeros(len(d), bool)
        order = rng.permutation(len(keys))
        #  reassign each block's selection COUNT to a randomly chosen block,
        #  so the number picked and the clustering are both preserved
        for a, b in zip(keys, [keys[i] for i in order]):
            ia, ib = blocks[a], blocks[b]
            n = int(sel[ia].sum())
            if n:
                perm[ib[:min(n, len(ib))]] = True
        if perm.sum():
            null.append(float(hit[perm].mean()))
    null = np.asarray(null)
    z = (obs - null.mean()) / max(null.std(ddof=1), 1e-9)

    #  half-split on cohort date
    mid = d["date"].quantile(0.5)
    halves = {}
    for nm, m in (("early", d["date"] <= mid), ("late", d["date"] > mid)):
        h = d[m]
        halves[nm] = {
            "base": float(h["hit"].mean()),
            "sel": float(h.loc[h["sel"], "hit"].mean())
            if h["sel"].any() else np.nan,
            "n": int(h["sel"].sum())}
    return {"base": base, "obs": obs, "lift": obs / max(base, 1e-9),
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_p95": float(np.percentile(null, 95)), "z": float(z),
            "p_emp": float((null >= obs).mean()),
            "n_sel": int(d["sel"].sum()),
            "eff_n": len(d) / max(k / 21.0, 1.0), "halves": halves}


if __name__ == "__main__":
    raise SystemExit(main())
