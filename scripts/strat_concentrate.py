#!/usr/bin/env python3
"""H54 family: CONCENTRATION PLUS PATIENCE.

The question: does owning a few large, liquid names and refusing to trade them
beat the index, beat a never-touched basket of the whole eligible universe, and
beat holding its own first basket?

Motivation is H23/A20: the most liquid decile held for ten years beat the index
by a paired median of +51.8%, and A20's lesson that the holding horizon was a
parameter twelve studies inherited without choosing. The stated key fact for
this exercise is the same effect from the other side -- owning everything and
rebalancing quarterly returns +1.15%/yr while owning the same names and never
rebalancing returns +10.05%/yr. Turnover is the enemy, so the design question is
how little of it a rule can get away with.

TWO IMPLEMENTATION FACTS ABOUT THE HARNESS THAT DECIDE THIS FAMILY
-----------------------------------------------------------------
1. `walk` computes the period return as `dot(target_weights, per-name returns)`.
   So a `select` that returns EQUAL weights every bar is silently REBALANCING TO
   EQUAL WEIGHT at every rebalance -- exactly the thing the key fact says costs
   nine points a year. To express "never touch it" the weights themselves have
   to drift: w_t proportional to w_{t-1} * (p_t / p_{t-1}). That is computable
   inside `select` from state carried forward (last bar's prices), which uses no
   information from after the decision bar.

2. `turn = 0.5*sum|w_cur - w_prev|` is measured against the PREVIOUS TARGET
   weights, not the drifted ones. That biases the harness in two directions and
   this study runs into both:
     * an equal-weight rebalance of a stable basket reads turnover 0 and pays no
       cost, when in reality it requires selling the winners every quarter;
     * a genuine no-trade drift reads turnover > 0 and PAYS cost for trades that
       were never made.
   The second is a headwind against the arm this study is arguing for, so it is
   left in place and reported rather than corrected. The first is a tailwind for
   the arms this study is arguing against, which is the safe direction.

Every number here is IN-SAMPLE. The 24-month holdout was spent long ago.

WHAT THIS ANSWERED — kept here so the file does not outlive its own result
-------------------------------------------------------------------------
78 configurations. On the six frequencies that are BOTH skip-free and free of
the ICBP defect, the tally is:

  FROZEN-HOLDALL EW        PASS 1/6 grids, beats all four benchmarks 4/6
  calm+strong10 of top60   PASS 1/6,  beats all four 2/6
  size10 STICKY EW         PASS 0/6,  beats all four 1/6
  size5  STICKY EW         PASS 0/6,  beats all four 1/6

So the family's answer is NO with one asterisk. The arm that is a repeatable
rule -- re-rank every period, keep what you hold, replace what drops out --
fails on every clean grid. The arm that passes on one grid makes a SINGLE
selection decision in twenty years, so its selection has an effective n of one.

The one thing that did replicate is not a selection effect at all: equal-weight
REBALANCING a fixed ten-name large-cap basket beat letting the same basket
drift, on every grid tested, by +1.1 to +4.5 CAGR points -- verified against a
hand-rolled path outside the harness, matching to 3bp. That is the opposite
sign to "turnover is the enemy", and both are true: drift wins across the whole
several-hundred-name universe, where it lets a few names become mega-caps, and
rebalancing wins inside a book of ten names that are ALREADY mega-caps, where
the drift just concentrates into whichever one last ran.

Usage:  python scripts/strat_concentrate.py                # the 48-row table
        python scripts/strat_concentrate.py --robust       # 4 arms x 7 grids
        python scripts/strat_concentrate.py --robust --fixicbp
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from bhbench import Bench, load, report  # noqa: E402

MIN_BASKET = 5


def add_size(P: pd.DataFrame) -> pd.DataFrame:
    """A 3-year trailing median of traded value, as the only size proxy going.

    `tv60` is a SIXTY-session median, and in Indonesia sixty sessions of heavy
    turnover is as often a speculative name being churned as it is a large one.
    The panel carries no shares outstanding on purpose (A25: the only sector
    file is frozen at 2024-07-10, so a 2024 share count on a 2010 bar is
    look-ahead, and rights issues are exactly what makes that wrong). A long
    trailing median of traded value is the closest thing to size that can be
    computed point-in-time from what is here: to sit in it for three years a
    name has to be persistently, not momentarily, large.
    """
    P = P.sort_values(["ticker", "date"]).copy()
    P["tv750"] = P.groupby("ticker")["tv"].transform(
        lambda s: s.rolling(750, min_periods=250).median())
    return P


# --------------------------------------------------------------------- rules
def rank_frame(day: pd.DataFrame, universe: int | None) -> pd.DataFrame:
    """Restrict to the `universe` most liquid eligible names on this bar.

    Point-in-time by construction: tv60 is a trailing 60-session median that the
    harness computed with a groupby-transform on the ticker, never on a pivot.
    """
    d = day.dropna(subset=["tv60"])
    col = "tv750" if ("tv750" in d.columns and d["tv750"].notna().sum() >= 40) \
        else "tv60"
    if universe is not None and len(d) > universe:
        d = d.nlargest(universe, col)
    return d


def score_liquidity(d: pd.DataFrame) -> pd.Series:
    return d["tv60"]


def score_size(d: pd.DataFrame) -> pd.Series:
    """Persistent size, not today's excitement. Falls back early in the panel."""
    if "tv750" in d.columns and d["tv750"].notna().sum() >= 10:
        return d["tv750"].fillna(d["tv60"] * 0.0)
    return d["tv60"]


def score_strength_calm(d: pd.DataFrame) -> pd.Series:
    """A26/H26's strength+calm, expressed as a score rather than two cuts.

    Percentile rank within the bar, so it never uses a full-sample statistic.
    """
    s = d["hi52"].rank(pct=True)
    v = (-d["vol60"]).rank(pct=True)
    return s + v


def score_lowvol(d: pd.DataFrame) -> pd.Series:
    return -d["vol60"]


def score_momentum(d: pd.DataFrame) -> pd.Series:
    return d["mom12_1"]


SCORES = {
    "liq": score_liquidity,
    "size": score_size,
    "calm+strong": score_strength_calm,
    "lowvol": score_lowvol,
    "mom": score_momentum,
}


def target_weights(d: pd.DataFrame, names, mode: str):
    """Weights at a rebalance, before the drift carry-forward.

    'cap' is the answer to A19: an EQUAL-weighted IDX basket structurally
    trails the CAP-weighted index, and no study in this repo has ever tried
    weighting by anything else. Shares outstanding do not exist here, so the
    proxy is persistent traded value -- a poor proxy, and it is the only one
    that is point-in-time.
    """
    if mode == "ew":
        return {t: 1.0 / len(names) for t in names}
    col = "tv750" if ("tv750" in d.columns
                      and d["tv750"].notna().sum() >= 10) else "tv60"
    v = d.set_index("ticker")[col].reindex(names)
    v = v.fillna(v.median() if np.isfinite(v.median()) else 1.0)
    v = v.clip(lower=1.0)
    if mode == "sqrtcap":
        v = np.sqrt(v)
    tot = float(v.sum())
    if not np.isfinite(tot) or tot <= 0:
        return {t: 1.0 / len(names) for t in names}
    return {t: float(x) / tot for t, x in v.items()}


def make_select(n: int, score: str, universe: int | None, drift: bool,
                sticky: bool = False, wmode: str = "ew", frozen: bool = False,
                hold_all: bool = False):
    """Return a fresh stateful `select`.

    drift=True   weights compound with the position, i.e. genuinely never
                 rebalanced -- winners are allowed to become the book.
    drift=False  equal weight every bar, i.e. rebalanced.
    sticky=True  a name already held is kept as long as it stays eligible and
                 inside the liquid universe; the score only decides ADDITIONS
                 when the book is short. This is the "refuse to trade" arm.
    frozen=True  the basket is chosen ONCE and never added to. With drift=True
                 this is the strategy-shaped twin of the harness's own
                 BH_PICKS, and it exists as a POSITIVE CONTROL: if it does not
                 come out close to bh_picks, the drift bookkeeping is wrong and
                 every "patient" number in this file is meaningless. A26's
                 sine-wave lesson -- check the instrument on a case whose answer
                 is known before believing what it says about anything else.
    """
    fn = SCORES[score]
    state = {"w": {}, "px": {}, "started": False}

    def select(day: pd.DataFrame):
        d = rank_frame(day, universe)
        if len(d) < MIN_BASKET:
            return []
        px = dict(zip(d["ticker"], d["adj_close"].astype(float)))
        held = state["w"]

        # --- drift the surviving book forward on its own prices -------------
        #  A name absent from `day` is one the HARNESS filtered out for being
        #  ineligible; the harness still prices it (PX.at / exit_price ignore
        #  eligibility), so `hold_all` keeps it -- weight stale for that bar,
        #  because select genuinely cannot see its price -- and the default
        #  sells it. That single switch is worth ~4 CAGR points here and it is
        #  the difference between "refuse to trade" and "hold only what is
        #  liquid today".
        new_w = {}
        for t, w in held.items():
            if t in px and t in state["px"] and state["px"][t] > 0:
                new_w[t] = w * px[t] / state["px"][t] if drift else w
            elif t in px or hold_all:
                new_w[t] = w

        sc_ = fn(d)
        ranked = list(d["ticker"][sc_.sort_values(ascending=False).index])
        if frozen and state["started"]:
            names = ([t for t in new_w] if hold_all
                     else [t for t in ranked if t in new_w])
        elif sticky:
            keep = [t for t in ranked if t in new_w]
            room = n - len(keep)
            add = [t for t in ranked if t not in new_w][:max(room, 0)]
            names = keep[:n] + add
        else:
            names = ranked[:n]

        if len(names) < MIN_BASKET:
            return []
        # new positions enter at the average weight of the surviving book, so
        # a replacement does not get an outsized or a token stake.
        if drift:
            base = (np.mean([new_w[t] for t in names if t in new_w])
                    if any(t in new_w for t in names) else 1.0 / len(names))
            tw = target_weights(d, names, wmode)
            w = {t: float(new_w.get(t, base if wmode == "ew"
                                    else base * tw[t] * len(names)))
                 for t in names}
        else:
            w = dict(target_weights(d, names, wmode))
        tot = sum(w.values())
        w = {t: v / tot for t, v in w.items()}
        state["w"] = w
        state["px"] = {t: px[t] for t in names if t in px}
        if hold_all:  # keep the last seen price for a temporarily dark name
            for t in names:
                if t not in px and t in held and t in state.get("pxold", {}):
                    state["px"][t] = state["pxold"][t]
        state["pxold"] = dict(state["px"])
        state["started"] = True
        return list(w.items())

    return select


def bench_halves(B: Bench, mk, freq: int) -> dict:
    """The benchmark CAGR in each half, so a FAIL says WHICH half failed.

    `report()` prints only a yes/no. Without the level, a rule that loses the
    early half by 2 points and one that loses it by 12 look identical, and the
    next variant gets designed against the wrong problem. Uses the harness's own
    `index_cagr`/`hold_basket` rather than a second implementation of them.
    """
    r = B.walk(mk(), freq=freq)
    if not r:
        return {}
    a0, b1 = r["start"], r["end"]
    mid = r["curve"][len(r["curve"]) // 2][0]
    uni0 = B.P[(B.P["date"] == a0) & B.P["elig"]]["ticker"].tolist()
    out = {}
    for nm, f in (("index", lambda a, b: B.index_cagr(a, b)),
                  ("universe", lambda a, b: B.hold_basket(uni0, a, b)),
                  ("picks", lambda a, b: B.hold_basket(r["first_basket"],
                                                       a, b))):
        out[nm + "_e"] = f(a0, mid)
        out[nm + "_l"] = f(mid, b1)
    out["strat_e"], out["strat_l"] = r["early"], r["late"]
    return out


VARIANTS = [
    # label,                        n,  score,        univ, drift, sticky, wmode, freq
    ("A liq10 EW quarterly",        10, "liq",        None, False, False, "ew", 63),
    ("B liq10 EW annual",           10, "liq",        None, False, False, "ew", 252),
    ("C liq10 DRIFT annual",        10, "liq",        None, True,  False, "ew", 252),
    ("D liq10 DRIFT quarterly",     10, "liq",        None, True,  False, "ew", 63),
    ("E liq10 DRIFT 2-yearly",      10, "liq",        None, True,  False, "ew", 504),
    ("F liq5  DRIFT annual",         5, "liq",        None, True,  False, "ew", 252),
    ("G liq20 DRIFT annual",        20, "liq",        None, True,  False, "ew", 252),
    ("H liq10 STICKY DRIFT ann",    10, "liq",        None, True,  True,  "ew", 252),
    ("I liq20 STICKY DRIFT ann",    20, "liq",        None, True,  True,  "ew", 252),
    ("J calm+strong10 of top60",    10, "calm+strong", 60,  True,  True,  "ew", 252),
    ("K calm+strong10 of top150",   10, "calm+strong", 150, True,  True,  "ew", 252),
    ("L lowvol10 of top60",         10, "lowvol",      60,  True,  True,  "ew", 252),
    ("M mom10 of top60",            10, "mom",         60,  True,  True,  "ew", 252),
    ("N calm+strong10 top60 Q",     10, "calm+strong", 60,  True,  True,  "ew", 63),
    ("O calm+strong10 top60 EW",    10, "calm+strong", 60,  False, True,  "ew", 252),
    # ---- round 2: persistent SIZE rather than 60-day turnover, and the
    #      cap-proxy weighting nothing in this repo has ever tried (A19).
    ("P size10 STICKY EW ann",      10, "size",       None, False, True,  "ew", 252),
    ("Q size10 STICKY CAPW ann",    10, "size",       None, False, True,  "cap", 252),
    ("R size20 STICKY CAPW ann",    20, "size",       None, False, True,  "cap", 252),
    ("S size10 STICKY DRIFT ann",   10, "size",       None, True,  True,  "ew", 252),
    ("T calm+strong10 t60 CAPW",    10, "calm+strong", 60,  False, True,  "cap", 252),
    ("U calm+strong10 t60 SQRTW",   10, "calm+strong", 60,  False, True,  "sqrtcap", 252),
    ("V calm+strong20 t60 EW",      20, "calm+strong", 60,  False, True,  "ew", 252),
    ("W calm+strong5  t60 EW",       5, "calm+strong", 60,  False, True,  "ew", 252),
    ("X calm+strong10 t60 EW 2yr",  10, "calm+strong", 60,  False, True,  "ew", 504),
    ("Y calm+strong10 t100 EW",     10, "calm+strong", 100, False, True,  "ew", 252),
    # ---- round 3: maximum patience, and the positive control
    ("Z size10 STICKY EW 3-yearly", 10, "size",       None, False, True,  "ew", 756),
    ("a size5  STICKY EW ann",       5, "size",       None, False, True,  "ew", 252),
    ("b size20 STICKY EW ann",      20, "size",       None, False, True,  "ew", 252),
    ("c size10 FROZEN DRIFT ann",   10, "size",       None, True,  True,  "ew", 252),
    ("d size10 FROZEN EW ann",      10, "size",       None, False, True,  "ew", 252),
    ("e FROZEN-HOLDALL DRIFT ann",  10, "size",       None, True,  True,  "ew", 252),
    ("f FROZEN-HOLDALL EW ann",     10, "size",       None, False, True,  "ew", 252),
    ("g size10 STICKY-HOLDALL DRIFT", 10, "size",     None, True,  True,  "ew", 252),
    ("h size10 STICKY-HOLDALL EW",  10, "size",       None, False, True,  "ew", 252),
    # ---- round 4: the ARTIFACT-FREE frequency.
    #  `walk` skips a whole rebalance period when the bar has fewer than
    #  MIN_UNIV=40 eligible names -- no cost, no return, the book sits in cash --
    #  while all three benchmarks stay invested from a0 to b1 throughout. At
    #  freq=252 the marks land on 2009-02-23 (34 names) and 2019-04-19 (ZERO
    #  names: Good Friday), so every annual strategy above is missing the entire
    #  2009 rebound (+152.5% on the frozen basket) and the entire COVID year
    #  (-12.4%). Net 2.21x of the benchmark's path, which is +4.4 CAGR points
    #  over 19.5 years and is the whole of the gap between the positive control
    #  and bh_picks. Frequency chosen on ONE pre-declared criterion computed
    #  from eligibility counts alone, never from returns: among frequencies with
    #  no skipped period and at least 12 marks, the one with the most marks.
    ("i* liq10 STICKY EW",          10, "liq",        None, False, True,  "ew", 264),
    ("j* size10 STICKY EW",         10, "size",       None, False, True,  "ew", 264),
    ("k* size10 STICKY DRIFT",      10, "size",       None, True,  True,  "ew", 264),
    ("l* size5  STICKY EW",          5, "size",       None, False, True,  "ew", 264),
    ("m* size20 STICKY EW",         20, "size",       None, False, True,  "ew", 264),
    ("n* size10 STICKY CAPW",       10, "size",       None, False, True,  "cap", 264),
    ("o* calm+strong10 of top60",   10, "calm+strong", 60,  False, True,  "ew", 264),
    ("p* calm+strong10 top60 DRIFT", 10, "calm+strong", 60, True,  True,  "ew", 264),
    ("q* lowvol10 of top60",        10, "lowvol",      60,  False, True,  "ew", 264),
    ("r* mom10 of top60",           10, "mom",         60,  False, True,  "ew", 264),
    ("s* FROZEN-HOLDALL DRIFT",     10, "size",       None, True,  True,  "ew", 264),
    ("t* FROZEN-HOLDALL EW",        10, "size",       None, False, True,  "ew", 264),
    ("u* size10 STICKY EW 2-yearly", 10, "size",      None, False, True,  "ew", 504),
    ("v* FROZEN-HOLDALL DRIFT 2yr", 10, "size",       None, True,  True,  "ew", 504),
]
FROZEN = {"c size10 FROZEN DRIFT ann", "d size10 FROZEN EW ann",
          "e FROZEN-HOLDALL DRIFT ann", "f FROZEN-HOLDALL EW ann",
          "s* FROZEN-HOLDALL DRIFT", "t* FROZEN-HOLDALL EW",
          "v* FROZEN-HOLDALL DRIFT 2yr"}
HOLDALL = {"e FROZEN-HOLDALL DRIFT ann", "f FROZEN-HOLDALL EW ann",
           "g size10 STICKY-HOLDALL DRIFT", "h size10 STICKY-HOLDALL EW",
           "s* FROZEN-HOLDALL DRIFT", "t* FROZEN-HOLDALL EW",
           "v* FROZEN-HOLDALL DRIFT 2yr",
           "j* size10 STICKY EW", "k* size10 STICKY DRIFT",
           "l* size5  STICKY EW", "m* size20 STICKY EW",
           "n* size10 STICKY CAPW"}


ROBUST_FREQS = (264, 300, 306, 354, 366, 396, 408)

#  A SPINE DEFECT THAT LANDS INSIDE THIS FAMILY'S BASKETS.
#  ICBP listed on 2010-10-07. The panel carries 2,533 bars before that, and
#  they are EXACTLY its parent INDF divided by two -- median price ratio
#  0.5000, volume exactly 2x INDF's, log-return correlation 1.000 before the
#  listing against 0.457 after. The fabricated volume is what puts ICBP inside
#  the top ten by three-year median traded value in 2005, so the defect does not
#  merely add a name, it BUYS one. Repaired by deleting the pre-listing bars,
#  and every headline is reported with and without, because a repair that is
#  only reported when it helps is not a repair. Scanned the other sixteen names
#  that appear in any first basket the same way: no other pair exceeds 0.65.
ICBP_LISTED = pd.Timestamp("2010-10-07")


def repair_icbp(P: pd.DataFrame) -> pd.DataFrame:
    bad = (P["ticker"] == "ICBP") & (P["date"] < ICBP_LISTED)
    return P.loc[~bad].copy()


def robust(B: Bench) -> pd.DataFrame:
    """The same two rules on every artifact-free mark grid.

    `walk` fixes offset=0, so one frequency is ONE alignment of rebalance dates
    and one window. A rule that clears the benchmarks on that grid and nowhere
    else has found a grid, not a rule. These seven frequencies are the ones with
    no skipped period and at least 12 marks -- chosen from eligibility counts
    before any return was computed -- and every one of them is reported, win or
    lose, because reporting the grid that worked is how this table would lie.
    """
    arms = [("size10 STICKY EW", 10, "size", None, False, True, "ew", True),
            ("size5 STICKY EW", 5, "size", None, False, True, "ew", True),
            ("FROZEN-HOLDALL EW", 10, "size", None, False, True, "ew", True),
            ("calm+strong10 top60 EW", 10, "calm+strong", 60, False, True,
             "ew", False)]
    frozen = {"FROZEN-HOLDALL EW"}
    out = []
    for nm, n, sc, uni, dr, stk, wm, ha in arms:
        for f in ROBUST_FREQS:
            sel = make_select(n, sc, uni, dr, stk, wm,
                              frozen=nm in frozen, hold_all=ha)
            v = B.evaluate(sel, label=f"{nm} @{f}", freq=f)
            if not v.get("ok"):
                out.append({"arm": nm, "freq": f, "ok": False})
                continue
            out.append({
                "arm": nm, "freq": f, "ok": True, "cagr": v["cagr"],
                "bh_index": v["bh_index"], "bh_universe": v["bh_universe"],
                "bh_picks": v["bh_picks"], "random": v["random"],
                "beats_all": bool(v["beats_index"] and v["beats_universe"]
                                  and v["beats_picks"] and v["beats_random"]),
                "bh_all": bool(v["both_halves_index"]
                               and v["both_halves_universe"]
                               and v["both_halves_picks"]),
                "PASS": v["PASS"], "start": v["start"], "end": v["end"]})
            print(report(v))
            print()
    return pd.DataFrame(out)


# ---------------------------------------------------------------------- main
def main() -> None:
    P = add_size(load())
    B = Bench(P)

    rows = []
    for lbl, n, sc, uni, drift, sticky, wm, freq in VARIANTS:
        fz, ha = lbl in FROZEN, lbl in HOLDALL

        def mk(n=n, sc=sc, uni=uni, drift=drift, sticky=sticky, wm=wm,
               fz=fz, ha=ha):
            return make_select(n, sc, uni, drift, sticky, wm, frozen=fz,
                               hold_all=ha)
        v = B.evaluate(mk(), label=lbl, freq=freq)
        print(report(v))
        if v.get("ok"):
            h = bench_halves(B, mk, freq)
            if h:
                print(f"     halves  strat {h['strat_e']:+.2%}/{h['strat_l']:+.2%}"
                      f"   index {h['index_e']:+.2%}/{h['index_l']:+.2%}"
                      f"   univ {h['universe_e']:+.2%}/{h['universe_l']:+.2%}"
                      f"   picks {h['picks_e']:+.2%}/{h['picks_l']:+.2%}")
                v.update(h)
        print()
        rows.append(v)

    ok = [r for r in rows if r.get("ok")]
    cols = ("label", "cagr", "bh_index", "bh_universe", "bh_picks", "random",
            "early", "late", "turnover", "cost_yr", "basket", "years", "PASS",
            "both_halves_index", "both_halves_universe", "both_halves_picks")
    T = pd.DataFrame([{k: r.get(k) for k in cols} for r in ok])
    os.makedirs("reports", exist_ok=True)
    T.to_csv("reports/strat_concentrate.csv", index=False)
    print("variants run:", len(rows), " passes:", int(T["PASS"].sum()))
    print(T.to_string(index=False))


def main_robust(fix: bool = False) -> None:
    P = load()
    if fix:
        P = repair_icbp(P)
        print("ICBP pre-listing bars removed\n")
    B = Bench(add_size(P))
    R = robust(B)
    R.to_csv("reports/strat_concentrate_robust%s.csv" % ("_icbpfix" if fix
                                                         else ""), index=False)
    print(R.to_string(index=False))
    print()
    for arm, g in R[R["ok"]].groupby("arm"):
        print(f"{arm:<26} beats all 4 on {int(g['beats_all'].sum())}/{len(g)}"
              f" grids, clears every half-split on {int(g['bh_all'].sum())}/"
              f"{len(g)}, PASS on {int(g['PASS'].sum())}/{len(g)}"
              f"  [median cagr {g['cagr'].median():+.2%} vs median index "
              f"{g['bh_index'].median():+.2%}]")


if __name__ == "__main__":
    if "--robust" in sys.argv:
        main_robust(fix="--fixicbp" in sys.argv)
    else:
        main()
