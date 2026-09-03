#!/usr/bin/env python3
"""H54 family — DEFENSIVE / QUALITY PROXIES BUILT FROM PRICE AND VOLUME.

THE QUESTION. This repo has no fundamentals at panel scale (A25: 59 names of
725), so "quality" has to be inferred from the price path. The academic
low-volatility / defensive-equity literature says exactly that is possible:
names that fall less compound more, because the geometric mean is the
arithmetic mean minus half the variance, and a basket that loses less does not
have to gain as much. So:

  does a LOW-DRAWDOWN basket compound past a CAP-WEIGHTED index because it
  loses less, even if it gains less?

THE PROXIES, all trailing, all computed per ticker, never on a pivot (A11).

  dd_min    worst trailing-252-session drawdown. Shallower is better. This is
            the family's headline variable — "how bad does it get".
  dd_cur    how far below its own trailing-252 high the name sits today.
  vol60     realised 60-session volatility (already in the panel). Lower better.
  semidev   downside semi-deviation over 252: std of the negative daily
            returns only. Volatility punishes upside; this does not.
  cons      consistency — share of trailing 252 sessions whose forward-looking-
            free 21-session TRAILING return was positive. Higher better.
  amihud60  Amihud illiquidity (already in the panel). Lower better.
  tvstd     stability of turnover: rolling std of log turnover. Lower better.
  beta      252-session beta to ^JKSE, computed from rolling moments. Lower
            better. This is the one proxy that is defensive in the CAPM sense
            rather than the standalone-risk sense.

HOW THEY COMBINE. `select` sees one rebalance bar. Each feature is converted to
a PERCENTILE RANK among the eligible names ON THAT BAR ONLY and the ranks are
averaged. No full-sample statistic ever defines the universe or the score, so
the cross-sectional standardisation is causal by construction.

WHAT THIS FAMILY IS RUNNING INTO BEFORE IT STARTS, stated in advance because
the harness will otherwise look like it is punishing the idea rather than the
implementation:

  1. TURNOVER IS THE ENEMY (the repo's own key fact). Owning everything and
     rebalancing quarterly returns ~+1.15%/yr; owning the same names and never
     touching them returns ~+10.05%/yr. Two of the three benchmarks are
     DRIFTING baskets. Any equal-weight rebalanced strategy therefore starts
     roughly nine points behind BH_UNIVERSE for reasons that have nothing to do
     with whether its selection is any good. Variant I below removes that
     handicap by returning DRIFTED weights.
  2. AN EQUAL-WEIGHTED IDX BASKET STRUCTURALLY TRAILS THE CAP-WEIGHTED INDEX
     (A19). A defensive screen is a mid-cap screen more often than not, so it
     inherits that handicap too.
  3. `squeeze`, a registered predicted-null, fires at |t|>3 on this panel, so
     no t-statistic here is evidence. Only effect size against the three
     benchmarks and the random control counts.

Everything below is IN SAMPLE. The 24-month holdout was spent before this
script existed and cannot be un-spent.

WHAT THE RUN OF 2026-08-30 FOUND. 81 harness verdicts, 76 distinct
configurations, 2 of them PASS — and both are the same feature on the same
rebalance calendar.

  * THE FAMILY'S PREMISE IS HALF TRUE AND THE HALF THAT IS TRUE DOES NOT PAY.
    The defensive tilt genuinely loses less. Max drawdown on the annual mark
    grid: low-beta -30.6%, low-vol -35.6%, all-8 composite -36.0%, against
    BH_UNIVERSE at -59.8% and the IHSG at -51.4%. That is a 25-to-29 point
    reduction in the worst peak-to-trough and it is not marginal.

  * IT DOES NOT CONVERT INTO COMPOUNDING, AND PROBE 6 SHOWS EXACTLY WHY.
    EVERY defensive variant loses the EARLY half and wins the LATE half:
    early idx +13.53% against low-vol +5.69%, low-beta +8.05%, low-drawdown
    +9.59%; late idx +2.71% against +7.53%, +4.62%, +9.76%. That is the
    textbook defensive-equity profile — lag the bull, win the flat — and it is
    precisely what the harness's both-halves requirement is built to catch.
    Full-sample CAGR can still come out ahead because the late half dominates
    the arithmetic; the both-halves test refuses that, correctly.
    Genuinely defensive variants: 0 of 5 pass.

  * THE ONE PASSING FEATURE IS THE LEAST DEFENSIVE MEMBER OF THE FAMILY.
    `cons` has the DEEPEST drawdown of the six (-53.1%, worse than the index)
    and a cross-sectional Spearman of +0.753 with `mom12_1`, sharing 53% of
    its basket with a plain 12-month momentum screen. It is smoothed momentum
    wearing a quality label, so its pass is not evidence for this family.

  * AND THE PASS DOES NOT SURVIVE BEING MOVED. Same rule, same everything, on
    12 different rebalance calendars: 1 of 12 PASS, and it is offset 0 — the
    one that got run first. Median CAGR across calendars +10.03% against a
    median BH_picks of +11.53%, beating its own first basket in 4 of 12. On
    the (return window, lookback) grid the feature was born with: 1 of 15,
    and again it is the (21, 252) cell chosen because a month is 21 sessions.
    On basket size and clock: 2 of 12, both at freq=252, offset 0.

    WHY OFFSET 0 IS THE ONE THAT WINS is visible in the table and is not about
    the strategy: its first eligible mark falls on 2007-01-24 while every
    other calendar starts in 2005, so it is the only window that EXCLUDES the
    2005-2007 index surge. Its BH_INDEX bar is +8.29% against +10.50% to
    +12.46% everywhere else. The strategy's own CAGR at offset 0 is ordinary;
    the benchmark it cleared was low.

  * THE DRIFT VARIANTS UNDERPERFORM THE REBALANCED ONES, which is the
    opposite of this repo's key fact and is worth stating. Same window,
    quarterly marks: equal-weight rebalanced +6.03%, drifted +5.36%. Letting
    weights run only pays if something in the basket runs, and a screen that
    selects for calm selects against exactly that.

CONCLUSION: the family fails. Price-based defensive proxies buy a real and
large reduction in drawdown and pay for it with the bull half of the sample.

Run:  python scripts/strat_quality.py            # A-R, the 18 headline variants
      python scripts/strat_quality.py --probe    # attack the one that passed
      python scripts/strat_quality.py --downside # does it actually lose less
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))

from bhbench import Bench, load, report, INDEX          # noqa: E402

CACHE = os.path.join("data", "spine", "_quality_feats.parquet")

#  Direction: +1 means "bigger is better/more defensive", -1 the reverse.
QCOLS = {
    "dd_min": +1,      # shallower worst drawdown
    "dd_cur": +1,      # closer to its own high
    "vol60": -1,       # calmer
    "semidev": -1,     # smaller downside deviation
    "cons": +1,        # more positive months
    "amihud60": -1,    # more liquid per unit of value
    "tvstd": -1,       # steadier turnover
    "beta": -1,        # less index-sensitive
}


# --------------------------------------------------------------------- build
def build(P: pd.DataFrame) -> pd.DataFrame:
    """Add the eight defensive proxies. Trailing only, grouped by ticker."""
    P = P.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = P.groupby("ticker", sort=False)

    P["ret1"] = g["adj_close"].pct_change()
    rmax = g["adj_close"].transform(lambda s: s.rolling(252, min_periods=120).max())
    P["dd_cur"] = P["adj_close"] / rmax - 1.0
    P["dd_min"] = P.groupby("ticker", sort=False)["dd_cur"].transform(
        lambda s: s.rolling(252, min_periods=120).min())

    neg = P["ret1"].where(P["ret1"] < 0, 0.0)
    P["_neg"] = neg
    P["semidev"] = P.groupby("ticker", sort=False)["_neg"].transform(
        lambda s: s.rolling(252, min_periods=120).std())

    #  Consistency: share of trailing 21-session returns that were positive,
    #  measured over the trailing year. TRAILING return, so bar t uses bars
    #  t-21..t and nothing after.
    r21 = g["adj_close"].transform(lambda s: s.pct_change(21))
    P["_up21"] = (r21 > 0).astype(float).where(r21.notna())
    P["cons"] = P.groupby("ticker", sort=False)["_up21"].transform(
        lambda s: s.rolling(252, min_periods=120).mean())

    P["tvstd"] = g["log_turnover"].transform(
        lambda s: s.rolling(120, min_periods=60).std())

    #  Beta from rolling moments rather than a groupby-apply on two columns:
    #  cov = E[xy] - E[x]E[y], var = E[y^2] - E[y]^2. Cheap and exact.
    J = pd.read_csv(INDEX, parse_dates=["date"]).sort_values("date")
    J["mret"] = J["close"].pct_change()
    P = P.merge(J[["date", "mret"]], on="date", how="left")
    P["_xy"] = P["ret1"] * P["mret"]
    P["_yy"] = P["mret"] ** 2
    W, MP = 252, 120
    gg = P.groupby("ticker", sort=False)
    ex = gg["ret1"].transform(lambda s: s.rolling(W, min_periods=MP).mean())
    ey = gg["mret"].transform(lambda s: s.rolling(W, min_periods=MP).mean())
    exy = gg["_xy"].transform(lambda s: s.rolling(W, min_periods=MP).mean())
    eyy = gg["_yy"].transform(lambda s: s.rolling(W, min_periods=MP).mean())
    var = eyy - ey ** 2
    P["beta"] = ((exy - ex * ey) / var.where(var > 1e-12)).clip(-5, 5)

    return P.drop(columns=["_neg", "_up21", "_xy", "_yy"])


def get_panel(force: bool = False) -> pd.DataFrame:
    if os.path.exists(CACHE) and not force:
        return pd.read_parquet(CACHE)
    t = time.time()
    #  Built at the harness's OWN default floor so `elig` matches `bhbench`
    #  exactly; `regate` moves the floor later from the stored tv60.
    P = build(load())
    P.to_parquet(CACHE, index=False)
    print(f"[built quality features in {time.time() - t:.0f}s]", flush=True)
    return P


def regate(P: pd.DataFrame, min_tv: float) -> pd.DataFrame:
    """Re-apply the harness's eligibility gate at a different liquidity floor."""
    P = P.copy()
    P["elig"] = (P["tradeable"].astype(bool) & (P["tv60"] >= min_tv)
                 & (P["close"] >= 100))
    return P


# ------------------------------------------------------------------ scoring
def score(day: pd.DataFrame, cols) -> pd.Series:
    """Average percentile rank of the named proxies, within THIS bar only."""
    s = np.zeros(len(day))
    n = np.zeros(len(day))
    for c in cols:
        v = day[c].to_numpy(float) * QCOLS[c]
        r = pd.Series(v).rank(pct=True).to_numpy()
        ok = np.isfinite(v)
        s = np.where(ok, s + np.nan_to_num(r), s)
        n = n + ok
    out = np.where(n >= max(1, len(cols) - 1), s / np.maximum(n, 1), np.nan)
    return pd.Series(out, index=day.index)


def topn(cols, n=20):
    def sel(day):
        d = day.copy()
        d["q"] = score(d, cols)
        d = d.dropna(subset=["q"])
        if len(d) < 5:
            return []
        s = d.nlargest(n, "q")
        return [(t, 1.0 / len(s)) for t in s["ticker"]]
    return sel


class Drift:
    """Re-select on a slow clock; between re-selections hand back the DRIFTED
    weights so the harness's rebalance is a no-op and winners compound.

    This is `strat_holdforever`'s device. It matters here because two of the
    three benchmarks are drifting baskets, so an equal-weight strategy is
    handicapped by construction rather than by its selection.

    The book is carried in SHARES, so its weights drift exactly as a real held
    book's do. The harness still charges `0.5*sum|w_t - w_{t-1}|` at every
    mark, so a purely held book is billed for dispersion it never traded. That
    runs AGAINST the strategy and is left in rather than netted out.
    """

    def __init__(self, cols, n=20, every=4, px=None):
        self.cols, self.n, self.every, self.px = cols, n, every, px
        self.k = 0
        self.shares = {}

    def __call__(self, day):
        d0 = day["date"].iloc[0]
        vals = {}
        for t, sh in self.shares.items():
            p = self.px.at(t, d0)
            if np.isfinite(p) and p > 0:
                vals[t] = sh * p
        if self.k % self.every == 0 or not vals:
            d = day.copy()
            d["q"] = score(d, self.cols)
            d = d.dropna(subset=["q"])
            if len(d) < 5:
                self.k += 1
                return []
            s = d.nlargest(self.n, "q")
            tot = sum(vals.values()) or 1.0
            per = tot / len(s)
            self.shares, vals = {}, {}
            for t in s["ticker"]:
                p = self.px.at(t, d0)
                if np.isfinite(p) and p > 0:
                    self.shares[t] = per / p
                    vals[t] = per
        else:
            #  A name that stopped printing is gone; the harness realises it.
            self.shares = {t: sh for t, sh in self.shares.items() if t in vals}
        self.k += 1
        return list(vals.items())


# --------------------------------------------------------------------- main
CORE = ["dd_min", "vol60", "cons", "amihud60", "tvstd"]
FULL = list(QCOLS)


def main() -> None:
    P = get_panel()
    print(f"panel {P.shape[0]:,} rows, {P['ticker'].nunique()} names, "
          f"{P['date'].min().date()} → {P['date'].max().date()}\n")

    B = Bench(P)
    rows = []

    def run(sel, label, freq=63, bench=None):
        v = (bench or B).evaluate(sel, label=label, freq=freq)
        print(report(v), "\n", flush=True)
        rows.append(v)
        return v

    print("=" * 74)
    print("A–H: equal-weight, rebalanced. The literal defensive tilt.")
    print("=" * 74)
    run(topn(CORE, 20), "A core composite, top20, quarterly", freq=63)
    run(topn(CORE, 20), "B core composite, top20, semiannual", freq=126)
    run(topn(CORE, 20), "C core composite, top20, annual", freq=252)
    run(topn(FULL, 20), "D all-8 composite, top20, annual", freq=252)
    run(topn(["dd_min"], 20), "E low-drawdown only, top20, annual", freq=252)
    run(topn(["vol60"], 20), "F low-vol only, top20, annual", freq=252)
    run(topn(["cons"], 20), "G consistency only, top20, annual", freq=252)
    run(topn(["beta"], 20), "H low-beta only, top20, annual", freq=252)

    print("=" * 74)
    print("I–K: basket size. Does diversifying the defensive tilt help?")
    print("=" * 74)
    run(topn(CORE, 10), "I core composite, top10, annual", freq=252)
    run(topn(CORE, 40), "J core composite, top40, annual", freq=252)
    run(topn(CORE, 80), "K core composite, top80, annual", freq=252)

    print("=" * 74)
    print("L–N: DRIFTED weights — remove the equal-weight handicap.")
    print("=" * 74)
    for lab, every, n in (("L", 4, 20), ("M", 8, 20), ("N", 4, 40)):
        dr = Drift(CORE, n=n, every=every, px=B.PX)
        run(lambda d, _dr=dr: _dr(d),
            f"{lab} core composite drift, top{n}, re-select every "
            f"{every} quarters", freq=63)

    print("=" * 74)
    print("O–P: liquidity floor. Defensive screens drift small; does moving")
    print("     upmarket close the gap to a cap-weighted index? (A38 says the")
    print("     edge usually dies faster than the handicap does.)")
    print("=" * 74)
    for lab, tv in (("O", 5e9), ("P", 2e10)):
        Bx = Bench(regate(P, tv))
        run(topn(CORE, 20), f"{lab} core composite, top20, annual, "
            f"tv>={tv:.0e}", freq=252, bench=Bx)

    print("=" * 74)
    print("Q–R: cost sensitivity. Is the toll what is killing it? Costs are")
    print("     NOT reduced to manufacture a win — this is a diagnostic and")
    print("     the headline stays at the default fee and half-tick spread.")
    print("=" * 74)
    run(topn(CORE, 20), "Q core composite, annual, ZERO cost (diagnostic)",
        freq=252, bench=Bench(P, fee=0.0, spread_mult=0.0))
    run(topn(CORE, 20), "R core composite, annual, full tick (pessimistic)",
        freq=252, bench=Bench(P, fee=0.0056, spread_mult=1.0))

    print("=" * 74)
    print("SUMMARY — every variant, honestly, pass or fail")
    print("=" * 74)
    hd = (f"{'variant':<52}{'CAGR':>8}{'idx':>8}{'univ':>8}{'picks':>8}"
          f"{'rand':>8}  halves  PASS")
    print(hd)
    for v in rows:
        if not v.get("ok"):
            print(f"{v['label']:<52}   did not run")
            continue
        h = (("I" if v["both_halves_index"] else ".")
             + ("U" if v["both_halves_universe"] else ".")
             + ("P" if v["both_halves_picks"] else "."))
        print(f"{v['label']:<52}{v['cagr']:>8.2%}{v['bh_index']:>8.2%}"
              f"{v['bh_universe']:>8.2%}{v['bh_picks']:>8.2%}"
              f"{v['random']:>8.2%}  {h:^6}  "
              f"{'PASS' if v['PASS'] else 'fail'}")
    n_pass = sum(bool(v.get("PASS")) for v in rows)
    print(f"\n{n_pass} of {len(rows)} variants PASS.")


#  ------------------------------------------------------------------ probe
#  `evaluate` fixes offset=0, so every headline in `main` is ONE draw of the
#  rebalance calendar. At freq=252 that is 20 rebalance dates chosen by where
#  the panel happens to start. A19/H52 record the smallest-cell trap three
#  times; a single-calendar result quoted without its dispersion is the same
#  shape. This mirrors `Bench.evaluate` exactly but exposes `offset`.
def evaluate_at(B, select, label, freq, offset, draws=4):
    r = B.walk(select, freq=freq, offset=offset)
    if not r:
        return None
    a0, b1 = r["start"], r["end"]
    uni0 = B.P[(B.P["date"] == a0) & B.P["elig"]]["ticker"].tolist()
    bh = {"index": B.index_cagr(a0, b1),
          "universe": B.hold_basket(uni0, a0, b1),
          "picks": B.hold_basket(r["first_basket"], a0, b1)}
    mid = r["curve"][len(r["curve"]) // 2][0]
    be = {"index": B.index_cagr(a0, mid),
          "universe": B.hold_basket(uni0, a0, mid),
          "picks": B.hold_basket(r["first_basket"], a0, mid)}
    bl = {"index": B.index_cagr(mid, b1),
          "universe": B.hold_basket(uni0, mid, b1),
          "picks": B.hold_basket(r["first_basket"], mid, b1)}
    ctl = [B.walk(select, freq=freq, offset=offset,
                  rng=np.random.default_rng(s)) for s in range(draws)]
    ctl = [c for c in ctl if c]
    rand = float(np.mean([c["cagr"] for c in ctl])) if ctl else np.nan
    ok = all(r["cagr"] > bh[k] for k in bh) and \
        all(r["early"] > be[k] and r["late"] > bl[k] for k in be) and \
        r["cagr"] > rand
    return {"label": label, "offset": offset, "cagr": r["cagr"],
            "early": r["early"], "late": r["late"], "years": r["years"],
            "bh_index": bh["index"], "bh_universe": bh["universe"],
            "bh_picks": bh["picks"], "random": rand, "PASS": bool(ok),
            "start": str(pd.Timestamp(a0).date())}


def cons_feature(P, win, look):
    """Rebuild `cons` at a different (return window, lookback). Trailing only."""
    r = P.groupby("ticker", sort=False)["adj_close"].transform(
        lambda s: s.pct_change(win))
    up = (r > 0).astype(float).where(r.notna())
    return up.groupby(P["ticker"]).transform(
        lambda s: s.rolling(look, min_periods=look // 2).mean())


def probe() -> None:
    """Attack variant G. One pass out of eighteen is what a sweep produces by
    accident, so the only question worth asking is whether it survives being
    moved."""
    P = get_panel()
    B = Bench(P)
    print("=" * 74)
    print("PROBE 1 — is `cons` just momentum wearing a defensive label?")
    print("=" * 74)
    marks = B.dates[0::252]
    cs = []
    for d in marks:
        g = P[(P["date"] == d) & P["elig"]]
        g = g.dropna(subset=["cons", "mom12_1"])
        if len(g) > 40:
            cs.append(g["cons"].corr(g["mom12_1"], method="spearman"))
    print(f"  cross-sectional Spearman(cons, mom12_1) over {len(cs)} annual "
          f"bars: median {np.median(cs):+.3f}, "
          f"range [{min(cs):+.3f}, {max(cs):+.3f}]")
    ov = []
    for d in marks:
        g = P[(P["date"] == d) & P["elig"]].dropna(subset=["cons", "mom12_1"])
        if len(g) > 40:
            a = set(g.nlargest(20, "cons")["ticker"])
            b = set(g.nlargest(20, "mom12_1")["ticker"])
            ov.append(len(a & b) / 20)
    print(f"  top-20 basket overlap with top-20 mom12_1: "
          f"mean {np.mean(ov):.0%}, max {max(ov):.0%}")

    print()
    print("=" * 74)
    print("PROBE 2 — 12 rebalance calendars. Same rule, different start bar.")
    print("          Run for the THREE best variants, not only the winner:")
    print("          stress-testing only what passed is how a sweep launders")
    print("          its own maximum.")
    print("=" * 74)
    for nm, cols in (("G consistency", ["cons"]), ("D all-8", FULL),
                     ("E low-drawdown", ["dd_min"])):
        sel = topn(cols, 20)
        res = [evaluate_at(B, sel, nm, 252, off) for off in range(0, 252, 21)]
        res = [r for r in res if r]
        print(f"  --- {nm} ---")
        for r in res:
            print(f"  offset {r['offset']:>3}  start {r['start']}  "
                  f"CAGR {r['cagr']:+7.2%}  idx {r['bh_index']:+7.2%}  "
                  f"univ {r['bh_universe']:+7.2%}  picks {r['bh_picks']:+7.2%}"
                  f"  rand {r['random']:+7.2%}  "
                  f"{'PASS' if r['PASS'] else 'fail'}")
        npass = sum(r["PASS"] for r in res)
        won = sum(r["cagr"] > r["bh_picks"] for r in res)
        print(f"  ==> {npass} of {len(res)} calendars PASS. "
              f"median CAGR {np.median([r['cagr'] for r in res]):+.2%} vs "
              f"median BH_picks "
              f"{np.median([r['bh_picks'] for r in res]):+.2%}; "
              f"beat own picks in {won}/{len(res)}\n")

    print()
    print("=" * 74)
    print("PROBE 3 — the two parameters `cons` was born with (21, 252) were")
    print("          chosen because a month is 21 sessions and a year is 252.")
    print("          If the result is real it should not need them.")
    print("=" * 74)
    grid = []
    for win in (5, 10, 21, 42, 63):
        for look in (126, 252, 504):
            col = f"cons_{win}_{look}"
            P[col] = cons_feature(P, win, look)
            QCOLS[col] = +1
            Bx = Bench(P)
            v = Bx.evaluate(topn([col], 20), label=col, freq=252)
            if v.get("ok"):
                grid.append((win, look, v))
                print(f"  win {win:>2}d look {look:>3}d  "
                      f"CAGR {v['cagr']:+7.2%}  picks {v['bh_picks']:+7.2%}  "
                      f"halves "
                      f"{'I' if v['both_halves_index'] else '.'}"
                      f"{'U' if v['both_halves_universe'] else '.'}"
                      f"{'P' if v['both_halves_picks'] else '.'}  "
                      f"{'PASS' if v['PASS'] else 'fail'}")
            P.drop(columns=[col], inplace=True)
            QCOLS.pop(col)
    print(f"  ==> {sum(v['PASS'] for _, _, v in grid)} of {len(grid)} "
          f"parameter cells PASS")

    print()
    print("=" * 74)
    print("PROBE 4 — basket size and rebalance clock around the winner.")
    print("=" * 74)
    out = []
    for n in (10, 20, 30, 50):
        for freq in (126, 252, 504):
            v = B.evaluate(topn(["cons"], n), label=f"cons top{n} f{freq}",
                           freq=freq)
            if v.get("ok"):
                out.append(v)
                print(f"  top{n:<3} freq {freq:>3}  CAGR {v['cagr']:+7.2%}  "
                      f"idx {v['bh_index']:+7.2%} univ {v['bh_universe']:+7.2%}"
                      f" picks {v['bh_picks']:+7.2%}  "
                      f"halves "
                      f"{'I' if v['both_halves_index'] else '.'}"
                      f"{'U' if v['both_halves_universe'] else '.'}"
                      f"{'P' if v['both_halves_picks'] else '.'}  "
                      f"{'PASS' if v['PASS'] else 'fail'}")
    print(f"  ==> {sum(v['PASS'] for v in out)} of {len(out)} cells PASS")


def hold_curve(B, tks, marks):
    """Value path of an equal-rupiah basket bought at marks[0] and never
    touched. Dead names are carried at their last print, which is exactly what
    `Bench.hold_basket` does, so the curve and the benchmark CAGR agree."""
    sh, out = {}, []
    for t in tks:
        p = B.PX.at(t, marks[0])
        if np.isfinite(p) and p > 0:
            sh[t] = 1.0 / p
    if not sh:
        return []
    for d in marks:
        v = [s * B.PX.exit_price(t, d) for t, s in sh.items()
             if np.isfinite(B.PX.exit_price(t, d))]
        if v:
            out.append((d, float(np.sum(v)) / len(sh)))
    return out


def maxdd(curve):
    e = np.array([v for _, v in curve], float)
    if len(e) < 3:
        return np.nan
    return float((e / np.maximum.accumulate(e) - 1.0).min())


def downside() -> None:
    """THE FAMILY'S ACTUAL QUESTION, asked directly: does a defensive basket
    lose less? Compounding past the index is one claim; falling less is the
    mechanism the claim rests on, and it is separately measurable."""
    P = get_panel()
    B = Bench(P)
    print("=" * 74)
    print("PROBE 5 — does the defensive tilt actually LOSE LESS?")
    print("          Max drawdown measured on the ANNUAL mark grid, so all")
    print("          three arms are sampled at identical dates.")
    print("=" * 74)
    specs = [("core composite", CORE, 20), ("all-8 composite", FULL, 20),
             ("low-drawdown only", ["dd_min"], 20),
             ("low-vol only", ["vol60"], 20),
             ("low-beta only", ["beta"], 20),
             ("consistency only", ["cons"], 20)]
    ref = None
    for name, cols, n in specs:
        r = B.walk(topn(cols, n), freq=252)
        if not r:
            continue
        marks = [d for d, _ in r["curve"]]
        if ref is None:
            a0 = r["start"]
            uni = P[(P["date"] == a0) & P["elig"]]["ticker"].tolist()
            uc = hold_curve(B, uni, [a0] + marks)
            J = B.J[(B.J.index >= pd.Timestamp(a0))
                    & (B.J.index <= pd.Timestamp(marks[-1]))]
            jc = [(d, float(J.asof(pd.Timestamp(d))))
                  for d in [a0] + marks if pd.Timestamp(d) >= J.index[0]]
            ref = (uc, jc)
            print(f"  BH universe (drifting, {len(uni)} names)   "
                  f"maxDD {maxdd(uc):+.1%}")
            print(f"  IHSG price index                    "
                  f"maxDD {maxdd(jc):+.1%}")
        print(f"  {name:<34}maxDD {maxdd([(a0, 1.0)] + r['curve']):+.1%}   "
              f"CAGR {r['cagr']:+.2%}")
    print("\n  (annual sampling understates every drawdown equally; the")
    print("   comparison between arms is what is being read, not the level)")

    print()
    print("=" * 74)
    print("PROBE 6 — WHERE the both-halves test is failed. `report` prints")
    print("          only YES/no, and 'beats on the full sample but not in")
    print("          both halves' is the exact shape a regime effect takes.")
    print("=" * 74)
    for name, cols, n in specs:
        r = B.walk(topn(cols, n), freq=252)
        if not r:
            continue
        a0, b1 = r["start"], r["end"]
        mid = r["curve"][len(r["curve"]) // 2][0]
        uni = P[(P["date"] == a0) & P["elig"]]["ticker"].tolist()
        e = (B.index_cagr(a0, mid), B.hold_basket(uni, a0, mid),
             B.hold_basket(r["first_basket"], a0, mid))
        l = (B.index_cagr(mid, b1), B.hold_basket(uni, mid, b1),
             B.hold_basket(r["first_basket"], mid, b1))
        print(f"  {name}")
        print(f"    early  strat {r['early']:+7.2%} | idx {e[0]:+7.2%} "
              f"univ {e[1]:+7.2%} picks {e[2]:+7.2%}")
        print(f"    late   strat {r['late']:+7.2%} | idx {l[0]:+7.2%} "
              f"univ {l[1]:+7.2%} picks {l[2]:+7.2%}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--downside":
        downside()
    elif len(sys.argv) > 1 and sys.argv[1] == "--probe":
        probe()
    else:
        main()
