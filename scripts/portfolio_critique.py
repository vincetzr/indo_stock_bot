"""Three critiques of H20 that H20 did not run on itself.

H20 concluded that no exit rule is established, that the entry works before
2017 and is a coin flip after, and that the defensible position is
buy-and-hold on the picks. Each of those three sentences has a hole.

**C1. "The entry is a coin flip after 2017" may be a POWER statement dressed
as an EFFECT statement.** The late half is six years, and A8 is explicit that
"no effect large enough to trade" and "no effect detected" are different
claims and only a power calculation licenses the first. H20 made the second
and wrote it as the first. This computes the late half's detectable effect and
asks whether it could have seen the early half's +5.86% at all.

**C2. The break at 2017 is where the sample happens to halve.** A single cut
chosen for convenience will find a break somewhere. Rolling windows say
whether 2017 is a date or an artefact.

**C3. THERE IS NO MARKET BENCHMARK ANYWHERE IN H20.** Every number is measured
against buy-and-hold on the same picks or against random draws from the same
pool. The IHSG is not in the study. `_JKSE.csv.gz` has been sitting in
`data/cache/ohlcv/` the whole time. A portfolio returning +10.5% a year is a
finding or a triviality depending entirely on what the index did, and H20
never asked.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from portfolio_sim import baskets, slots                            # noqa: E402

NAMES = "data/spine/portfolio_names.parquet"
PANEL = "data/spine/price_panel.parquet"
INDEX = "data/cache/ohlcv/_JKSE.csv.gz"
DRAWS = 20
SLOTS = 12


def agg(S: pd.DataFrame, col: str = "cagr") -> tuple:
    """Median and mean across slots.

    `portfolio_sim.summarise` reports the MEDIAN across slots and this script
    originally reported the MEAN, which is the whole of the +10.5% vs +9.8%
    discrepancy between the memo and the first run of this file. Both are
    printed so the choice is visible rather than load-bearing.
    """
    if S is None or S.empty:
        return (np.nan, np.nan)
    return (float(S[col].median()), float(S[col].mean()))


# ---------------------------------------------------------------- the index --
def index_series() -> pd.Series:
    """IHSG close, indexed by date. The benchmark H20 never printed."""
    ix = pd.read_csv(INDEX)
    dc = "date" if "date" in ix.columns else ix.columns[0]
    cc = [c for c in ix.columns if c.lower() in ("close", "adj_close",
                                                 "adjclose")][0]
    ix[dc] = pd.to_datetime(ix[dc], utc=True, errors="coerce").dt.tz_localize(
        None)
    s = ix.set_index(dc)[cc].astype(float).sort_index()
    return s[s > 0].dropna()


def dividend_drag(tickers: set, lo: pd.Timestamp, hi: pd.Timestamp) -> Dict:
    """How much of the picks' return is dividends the index never receives.

    THE COMPARISON IS NOT LIKE FOR LIKE AND THE DIRECTION FAVOURS THE PICKS.
    The name returns run on `adj_close`, which is Yahoo's back-adjusted close
    and therefore a TOTAL return; `^JKSE` is a price index with no free
    total-return version. So the picks are credited a dividend stream the
    benchmark is not.

    Measured rather than assumed, and the measurement identifies itself:
    log(adj_close/close) is a step function that moves only at corporate
    actions, and back-adjustment makes every dividend step POSITIVE going
    forward. Across 1.75m steps in this universe there are 3,675 small
    positive steps and ZERO small negative ones, which is what a dividend
    series looks like and not what noise looks like. Steps above 0.10 in log
    are splits and bonuses and are excluded by magnitude.
    """
    P = pd.read_parquet(PANEL, columns=["date", "ticker", "close",
                                        "adj_close"])
    P = P[P["ticker"].isin(tickers)]
    P = P[(P["date"] >= lo) & (P["date"] <= hi)]
    P = P[(P["close"] > 0) & (P["adj_close"] > 0)].sort_values(
        ["ticker", "date"])
    P["step"] = P.groupby("ticker")["adj_close"].transform(
        lambda s: np.log(s)) - np.log(P["close"])
    P["step"] = P.groupby("ticker")["step"].diff()
    P["yr"] = P["date"].dt.year
    small = P[(P["step"] > 0.0005) & (P["step"] < 0.10)]
    ann = small.groupby(["ticker", "yr"])["step"].sum()
    present = P.groupby(["ticker", "yr"]).size()
    full = present[present >= 200].index
    ann = ann.reindex(full).fillna(0.0)
    return {"n_ticker_years": int(len(full)), "mean": float(ann.mean()),
            "median": float(ann.median()),
            "share_paying": float((ann > 0.005).mean()),
            "n_pos_steps": int(((P["step"] > 0.0005) &
                                (P["step"] < 0.10)).sum()),
            "n_neg_steps": int(((P["step"] < -0.0005) &
                                (P["step"] > -0.10)).sum())}


def yield_by_liquidity(lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    """Dividend yield by liquidity decile, across the WHOLE spine.

    Correcting the picks down to a price basis was one framing. The other is
    to correct the index UP, and those give different answers unless the two
    have the same yield. The IHSG is cap-weighted, so its dividend is the
    large-cap dividend, and large caps in Indonesia are banks and telcos that
    pay considerably more than the mid-caps this rule selects. Measuring the
    decile curve says which way that cuts instead of assuming.
    """
    P = pd.read_parquet(PANEL, columns=["date", "ticker", "close",
                                        "adj_close", "log_turnover"])
    P = P[(P["date"] >= lo) & (P["date"] <= hi)]
    P = P[(P["close"] > 0) & (P["adj_close"] > 0)].sort_values(
        ["ticker", "date"])
    P["step"] = np.log(P["adj_close"]) - np.log(P["close"])
    P["step"] = P.groupby("ticker")["step"].diff()
    P["yr"] = P["date"].dt.year
    small = P[(P["step"] > 0.0005) & (P["step"] < 0.10)]
    ann = small.groupby(["ticker", "yr"])["step"].sum()
    g = P.groupby(["ticker", "yr"]).agg(n=("close", "size"),
                                        liq=("log_turnover", "median"))
    g = g[g["n"] >= 200].copy()
    g["div"] = ann.reindex(g.index).fillna(0.0)
    #  decile WITHIN year, so the ranking is cross-sectional not a time trend
    g = g.dropna(subset=["liq"])
    g["dec"] = g.groupby("yr")["liq"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False)
        if len(s) >= 10 else np.nan)
    return g.dropna(subset=["dec"]).groupby("dec").agg(
        n=("div", "size"), yld=("div", "mean"), paying=("div",
                                                        lambda s: (s > 0.005).mean()))


def max_dd(s: pd.Series, a: pd.Timestamp, b: pd.Timestamp) -> float:
    """The index's own worst drawdown, which H20 never put beside the picks'.

    The guard is at two points, which is where a peak-to-trough is genuinely
    undefined, and not at some round number of sessions. A threshold like
    ``len(w) < 10`` silently returns NaN for a window that has a perfectly
    well-defined answer, and a NaN that means "I declined" is indistinguishable
    downstream from one that means "no data".
    """
    w = s[(s.index >= a) & (s.index <= b)]
    if len(w) < 2:
        return np.nan
    return float((w / w.cummax() - 1.0).min())


#: Published IHSG year-end closes. A13 found DECIMAL-SHIFT ERRORS in `IDR=X`
#: from this same unauthenticated endpoint — 888.11 against a true ~8,881 —
#: so a benchmark taken from it on trust is exactly the mistake this repo has
#: already made once. An endpoint-to-endpoint CAGR reads only two bars, which
#: makes a defect at either end maximally damaging rather than averaged away.
LANDMARKS = {"2004-12-31": 1000.2, "2007-12-28": 2745.8, "2008-12-30": 1355.4,
             "2013-12-30": 4274.2, "2019-12-30": 6299.5, "2023-08-01": 6886.4}


def validate_index(s: pd.Series) -> Dict:
    """Check the cached IHSG before any number is computed from it.

    The landmark comparison is the WEAKER half — those values come from
    knowledge, not from a fetch of IDX's own publication, so it can confirm
    the series is the IHSG but cannot independently certify a level. The
    internal checks are the stronger half and need no external reference: a
    decimal shift announces itself as a huge move immediately reversed, and a
    calendar gap that is not an Indonesian holiday announces itself as a gap.
    """
    out = {"n": int(len(s)), "start": s.index.min(), "end": s.index.max()}
    worst = 0.0
    for d, exp in LANDMARKS.items():
        i = s.index.searchsorted(pd.Timestamp(d), "right") - 1
        if i >= 0:
            worst = max(worst, abs(float(s.iloc[i]) / exp - 1.0))
    r = np.log(s).diff().dropna()
    g = pd.Series(s.index).diff().dt.days
    out |= {"worst_landmark_error": worst,
            "n_moves_over_20pct": int((r.abs() > 0.20).sum()),
            "n_moves_over_10pct": int((r.abs() > 0.10).sum()),
            "kurtosis": float(r.kurt()),
            "max_gap_days": float(g.max()),
            "n_gaps_over_12d": int((g > 12).sum())}
    return out


def bench_slots(S: pd.Series, P: pd.DataFrame, yld: float = 0.0
                ) -> pd.DataFrame:
    """Pair each slot against the index over THAT SLOT'S OWN span.

    TWO DEFECTS IN THE FIRST VERSION OF THIS FILE, BOTH IN THE SAME DIRECTION
    AS EVERY OTHER ERROR HERE — they compared things measured over different
    windows.

    First, slot *s* starts at cohort *s*, so the twelve slots begin in twelve
    different months and end in twelve different months. Benchmarking all of
    them against one global index number compares each slot to a window it did
    not occupy. Second, a slot's last position is still open for its holding
    period after the final entry, so its span ends a year AFTER its last
    cohort date; an index measured to the last cohort date is short by that
    year. `slots()` now returns `start` and `end` so both go away.

    `yld` is added to the index return to put it on a total-return basis.
    """
    rows: List[Dict] = []
    for _, r in P.iterrows():
        ix = index_cagr(S, r["start"], r["end"])
        rows.append({"slot": int(r["slot"]), "picks": float(r["cagr"]),
                     "index": ix + yld, "d": float(r["cagr"]) - (ix + yld),
                     "start": r["start"], "end": r["end"],
                     "years": float(r["years"])})
    return pd.DataFrame(rows)


def index_cagr(s: pd.Series, a: pd.Timestamp, b: pd.Timestamp) -> float:
    """Index CAGR between two dates, using the nearest prior close to each."""
    ia = s.index.searchsorted(a, "right") - 1
    ib = s.index.searchsorted(b, "right") - 1
    if ia < 0 or ib <= ia:
        return np.nan
    yrs = (s.index[ib] - s.index[ia]).days / 365.25
    if yrs <= 0:
        return np.nan
    return float((s.iloc[ib] / s.iloc[ia]) ** (1.0 / yrs) - 1.0)


# ------------------------------------------------------ paired entry per slot --
def entry_pairs(D: pd.DataFrame, rule: str = "hold 252",
                draws: int = DRAWS) -> pd.DataFrame:
    """ΔCAGR per (slot, draw): the picks against a random draw, same slots.

    Returns the long frame so the SLOT can be used as the unit of analysis.
    H20 pooled 240 pairs and then disclaimed the t; the disclaimer is right
    and the fix is to average within slot first, which is done by the callers.
    """
    P = slots(baskets(D, rule, "picked"), SLOTS).set_index("slot")["cagr"]
    rows: List[Dict] = []
    for k in range(draws):
        R = slots(baskets(D, rule, "random", size=12, seed=1000 + k), SLOTS)
        if R.empty:
            continue
        R = R.set_index("slot")["cagr"]
        for s in P.index.intersection(R.index):
            rows.append({"slot": int(s), "draw": k,
                         "d": float(P[s] - R[s])})
    return pd.DataFrame(rows)


def by_slot(pairs: pd.DataFrame) -> np.ndarray:
    """Collapse the redraws so each SLOT contributes one number, not twenty."""
    if pairs.empty:
        return np.array([])
    return pairs.groupby("slot")["d"].mean().to_numpy()


def power(x: np.ndarray, target: float) -> Dict:
    """What this sample could have detected, and whether it could see `target`.

    The twelve slots are overlapping and therefore not twelve independent
    trials; treating them as independent OVERSTATES the power, which is the
    conservative direction for a claim of the form "we could have seen it and
    did not".
    """
    n = len(x)
    if n < 3:
        return {}
    m, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    se = sd / np.sqrt(n)
    lo, hi = m - 1.96 * se, m + 1.96 * se
    return {"n": n, "mean": m, "sd": sd, "se": se, "lo": lo, "hi": hi,
            # smallest effect this sample would call significant at 95%
            "mde": 1.96 * se,
            # is the other half's effect inside this half's interval?
            "covers_target": bool(lo <= target <= hi)}


def main() -> int:
    D = pd.read_parquet(NAMES)
    S = index_series()
    W = 100

    def head(t):
        print("\n" + "=" * W + f"\n {t}\n" + "=" * W)

    print(f" {len(D):,} name-cohorts, {D['as_of'].nunique()} cohorts, "
          f"{D['ticker'].nunique()} names")
    v = validate_index(S)
    print(f" IHSG cached {v['start'].date()} .. {v['end'].date()}, "
          f"{v['n']:,} sessions")
    print(f"   validated: worst landmark error "
          f"{v['worst_landmark_error']:.4%}, "
          f"{v['n_moves_over_20pct']} moves >20% "
          f"({v['n_moves_over_10pct']} >10%), kurtosis {v['kurtosis']:.1f}, "
          f"max gap {v['max_gap_days']:.0f}d, {v['n_gaps_over_12d']} gaps >12d")
    if v["worst_landmark_error"] > 0.01 or v["n_moves_over_20pct"] > 0:
        print("   !! the benchmark series FAILS validation — see A13's"
              " decimal-shift defect in IDR=X from the same endpoint")

    cut = D["as_of"].quantile(0.5)
    a, b = D["as_of"].min(), D["as_of"].max()
    halves = [("early", a, cut), ("late", cut, b), ("full", a, b)]

    # ---------------------------------------------------------------- C3 --
    head("C3 — THE BENCHMARK H20 NEVER PRINTED")
    print(" Every H20 number is picks-vs-picks or picks-vs-pool. The IHSG is")
    print(" the thing a retail account can buy instead, for one round trip.\n")
    div = dividend_drag(set(D["ticker"]), D["as_of"].min(),
                        D["as_of"].max() + pd.Timedelta(days=400))
    print(f"   dividend steps in this universe: {div['n_pos_steps']:,} "
          f"positive, {div['n_neg_steps']:,} negative")
    print(f"   -> mean annual dividend on a held name: {div['mean']:.2%} "
          f"({div['n_ticker_years']:,} ticker-years, "
          f"{div['share_paying']:.0%} pay)")
    print("   The names are on adj_close and so receive it; ^JKSE is a price")
    print("   index and does not. Subtracted below to make the two")
    print("   comparable, which can only WIDEN any shortfall.\n")
    dv = div["mean"]

    print(f"   {'window':<8}{'picks':>16}{'pool':>16}{'IHSG':>9}"
          f"{'picks-IHSG':>12}{'ex-div':>9}")
    print(f"   {'':8}{'med / mean':>16}{'med / mean':>16}")
    for nm, lo, hi in halves:
        d = D[(D["as_of"] >= lo) & (D["as_of"] <= hi)]
        pk = agg(slots(baskets(d, "hold 252", "picked"), SLOTS))
        rr = [agg(slots(baskets(d, "hold 252", "random", size=12,
                                seed=1000 + k), SLOTS)) for k in range(DRAWS)]
        rn = (float(np.mean([r[0] for r in rr])),
              float(np.mean([r[1] for r in rr])))
        ix = index_cagr(S, lo, hi)
        print(f"   {nm:<8}{pk[0]:>+8.1%}/{pk[1]:>+7.1%}"
              f"{rn[0]:>+8.1%}/{rn[1]:>+7.1%}{ix:>+9.1%}"
              f"{pk[0] - ix:>+12.1%}{pk[0] - ix - dv:>+9.1%}")
    print("\n   The index is cap-weighted and the baskets are equal-weighted,")
    print("   so pool-minus-IHSG is a size statement, not a skill statement.")
    print("   picks-minus-IHSG is the one that matters to a buyer, and the")
    print("   ex-div column is the honest version of it.")
    print("   Note the picks are already NET of the 0.56% round trip; buying")
    print("   the index once over nineteen years pays it about once.")

    Y = yield_by_liquidity(D["as_of"].min(),
                           D["as_of"].max() + pd.Timedelta(days=400))
    print("\n   And the index does not yield 1.27% — it is cap-weighted, so")
    print("   it yields what LARGE caps yield. Measured by liquidity decile:")
    print(f"\n   {'decile':<9}{'ticker-yrs':>12}{'div yield':>11}{'% paying':>10}")
    for dec, r in Y.iterrows():
        mark = "  <- index-like" if dec >= 8 else ""
        print(f"   {int(dec) + 1:<9}{int(r['n']):>12,}{r['yld']:>11.2%}"
              f"{r['paying']:>10.0%}{mark}")
    top = float(Y.loc[Y.index >= 8, "yld"].mean())
    print(f"\n   Top two deciles yield {top:.2%}; the picks yield "
          f"{div['mean']:.2%}.")
    print("   So correcting the index UP to a total-return basis is the")
    print("   larger correction, and it cuts the same way:")
    for nm, lo, hi in halves:
        d = D[(D["as_of"] >= lo) & (D["as_of"] <= hi)]
        pk = agg(slots(baskets(d, "hold 252", "picked"), SLOTS))[0]
        ix = index_cagr(S, lo, hi)
        print(f"     {nm:<8} picks TR {pk:>+7.1%}   index TR "
              f"{ix + top:>+7.1%}   gap {pk - ix - top:>+7.1%}")

    print("\n   DRAWDOWN, which H20 also never put beside the picks':")
    print(f"   {'window':<8}{'picks maxDD':>13}{'IHSG maxDD':>13}")
    for nm, lo, hi in halves:
        d = D[(D["as_of"] >= lo) & (D["as_of"] <= hi)]
        Sl = slots(baskets(d, "hold 252", "picked"), SLOTS)
        pd_ = agg(Sl, "maxdd")[0]
        e = Sl["end"].max() if not Sl.empty else hi
        print(f"   {nm:<8}{pd_:>+13.1%}{max_dd(S, lo, e):>+13.1%}")

    # ------------------------------------------------------- paired version --
    head("C3b — THE SAME COMPARISON, PAIRED PER SLOT")
    print(" The table above benchmarks twelve slots that begin in twelve")
    print(" different months against ONE index window, and stops the index at")
    print(" the last cohort date while the picks hold a year past it. Both")
    print(" are window mismatches. Each slot is now compared to the index over")
    print(" its own span, which is the same pairing §3 and §4 already use.\n")
    print(f"   {'window':<8}{'slots':>6}{'picks':>9}{'index TR':>10}"
          f"{'mean d':>9}{'sd':>8}{'wins':>7}{'95% CI':>20}")
    for nm, lo, hi in halves:
        d = D[(D["as_of"] >= lo) & (D["as_of"] <= hi)]
        Sl = slots(baskets(d, "hold 252", "picked"), SLOTS)
        if Sl.empty:
            continue
        Bs = bench_slots(S, Sl, top).dropna(subset=["d"])
        x = Bs["d"].to_numpy()
        p = power(x, 0.0)
        print(f"   {nm:<8}{len(Bs):>6}{Bs['picks'].mean():>+9.1%}"
              f"{Bs['index'].mean():>+10.1%}{p['mean']:>+9.2%}"
              f"{p['sd']:>8.2%}{int((x > 0).sum()):>4}/{len(x)}"
              f"   [{p['lo']:>+7.2%},{p['hi']:>+7.2%}]")
    print("\n   Twelve overlapping slots over one history are NOT twelve")
    print("   independent trials, so the interval understates uncertainty.")
    print("   It is quoted because it is the FAVOURABLE reading and the")
    print("   picks still do not win it.")

    # ---------------------------------------------------------------- C1 --
    head("C1 — IS THE LATE HALF A NULL, OR IS IT UNPOWERED?")
    print(" A8: 'no effect large enough to trade' and 'no effect detected'")
    print(" are different claims and only the power calculation licenses the")
    print(" first. H20 made the second and wrote it as the first.\n")
    res = {}
    for nm, lo, hi in halves:
        d = D[(D["as_of"] >= lo) & (D["as_of"] <= hi)]
        res[nm] = by_slot(entry_pairs(d))
    early_eff = float(np.mean(res["early"])) if len(res["early"]) else np.nan

    print(f"   {'window':<8}{'slots':>7}{'mean d':>10}{'sd':>9}{'se':>9}"
          f"{'95% CI':>20}{'MDE':>9}")
    for nm in ("early", "late", "full"):
        p = power(res[nm], early_eff)
        if not p:
            continue
        print(f"   {nm:<8}{p['n']:>7}{p['mean']:>+10.2%}{p['sd']:>+9.2%}"
              f"{p['se']:>9.2%}"
              f"   [{p['lo']:>+7.2%},{p['hi']:>+7.2%}]{p['mde']:>9.2%}")
    pl = power(res["late"], early_eff)
    if pl:
        print(f"\n   The early half measured {early_eff:+.2%}.")
        print(f"   The late half's 95% interval is "
              f"[{pl['lo']:+.2%}, {pl['hi']:+.2%}] and it "
              f"{'COVERS' if pl['covers_target'] else 'EXCLUDES'} that value.")
        if pl["covers_target"]:
            print("   -> The late half CANNOT distinguish 'the edge died'")
            print("      from 'the edge continued at its early size'. H20's")
            print("      sentence 'in the last six years the entry is a coin")
            print("      flip' is not supported. It is UNRESOLVED, not null.")
        else:
            print("   -> The late half genuinely excludes the early effect.")
            print("      H20's claim of a break stands.")
        print(f"\n   Smallest effect the late half could have called "
              f"significant: {pl['mde']:.2%} a year.")
        print(f"   The round-trip cost bar alone is 0.56% per trade.")

    # ---------------------------------------------------------------- C2 --
    head("C2 — IS 2017 A DATE, OR IS IT WHERE THE SAMPLE HALVES?")
    print(" A single cut chosen for convenience finds a break somewhere.")
    print(" Rolling six-year windows, stepped a year, same paired statistic.\n")
    print(f"   {'window':<26}{'cohorts':>9}{'mean d':>10}{'sd':>9}"
          f"{'slots>0':>9}")
    yrs = sorted({d.year for d in D["as_of"]})
    roll = []
    for y0 in yrs:
        lo = pd.Timestamp(f"{y0}-01-01")
        hi = pd.Timestamp(f"{y0 + 6}-01-01")
        d = D[(D["as_of"] >= lo) & (D["as_of"] < hi)]
        if d["as_of"].nunique() < 48:
            continue
        x = by_slot(entry_pairs(d))
        if len(x) < 6:
            continue
        roll.append((y0, float(np.mean(x))))
        print(f"   {str(lo.date()) + ' .. ' + str(hi.date()):<26}"
              f"{d['as_of'].nunique():>9}{np.mean(x):>+10.2%}"
              f"{np.std(x, ddof=1):>+9.2%}{int((x > 0).sum()):>6}/{len(x)}")
    if len(roll) >= 4:
        v = np.array([r[1] for r in roll])
        yy = np.array([r[0] for r in roll], float)
        sl = np.polyfit(yy, v, 1)[0]
        print(f"\n   {int((v > 0).sum())}/{len(v)} rolling windows positive; "
              f"trend {sl:+.2%} per year of start date.")
        print("   Overlapping windows share five years of six, so this is a")
        print("   shape, not a set of independent measurements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
