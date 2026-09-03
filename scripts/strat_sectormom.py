#!/usr/bin/env python3
"""H54 family: SECTOR / INDUSTRY MOMENTUM on IDX, against the bhbench harness.

THE HYPOTHESIS
  Indonesia is a commodity economy. Coal, palm oil, banks and property move as
  BLOCS, so the sector's own trailing return should carry more signal per unit
  of noise than any single name's, because averaging ~10-30 members kills the
  idiosyncratic term. Rank the eleven IDX-IC sectors by their own trailing
  return, hold the names inside the strongest ones.

  This is one of the most replicated anomalies in the literature (Moskowitz &
  Grinblatt 1999, Asness/Porter/Stevens) and it has NEVER been tested in this
  repo. Sector data was only found in A14 and only used once, marginally, in
  H27's cross-sectional model.

WHAT THE SECTOR FILE IS, AND WHAT THAT COSTS US  (read before any number)
  data/reference/idx_classification.parquet, 934 tickers, 11 IDX-IC sectors.
  It is a SNAPSHOT, not a point-in-time history. Measured, not assumed:

  * max(listing_date) = 2024-07-10, so it is frozen there and cannot know the
    names listed since.  Those names get NO sector and are simply unpickable.
  * IDX-IC replaced JASICA in JANUARY 2021.  Every label before that date is
    therefore applied RETROACTIVELY.  A company that changed business over the
    sample carries its 2024 label for its whole history.  This is a genuine,
    unfixable, mild look-ahead and it is stated in the memo rather than
    papered over.  Its size is bounded by how often IDX reclassifies a name,
    which is rarely, but "rarely" is not "never".
  * SURVIVORSHIP, the one that would actually manufacture a result, is
    MEASURED: of panel names whose last print is >90d before the panel end,
    94.4% carry a sector; of names still printing, 96.4% do.  A two-point gap
    is not a survivorship filter.  (If dead names had been systematically
    absent, restricting the universe to sector-covered names would have been
    look-ahead by construction.)
  * `shares` and `listing_board_id` are DELIBERATELY NOT USED.  A25 records
    why: a share count frozen at 2024 applied to a 2010 bar is look-ahead, and
    Indonesian rights issues are exactly what makes that wrong.

  The universe / `elig` flag is left EXACTLY as the harness defines it.  The
  strategy simply cannot pick a name with no sector.  Narrowing `elig` to
  sector-covered names would have moved BH_UNIVERSE, and A19 records comparing
  quantities over different windows as this repo's most productive error class.

WHAT IS COMPUTED, AND WHY IT CANNOT PEEK
  Trailing returns over 21/63/126/252 sessions and 12-1, per TICKER, on the
  name's own bars via groupby -- never rolled on a date x ticker pivot, which
  A11 records as the defect that once printed "0 of 830 names above the
  200-day".  `select` is then handed one bar and computes the sector aggregate
  from THAT BAR'S ROWS ONLY.  The future is not in the dataframe.

ALL VARIANTS ARE REPORTED, INCLUDING THE FAILURES.  The count is printed.
The 24-month holdout is already spent; every number here is IN-SAMPLE.

================================ THE RESULT ================================
NEGATIVE.  36 variants, 7 of them "PASS" the harness at its default offset=0,
and NOT ONE of those passes survives the two controls this script exists to
run.  Do not read the PASS column without BLOCK 9 and BLOCK 10 beside it.

1. SCRAMBLED SECTOR LABELS PASS TOO.  `FAKE TILT ret252 top15 annual` --
   identical arithmetic, sector labels replaced by an arbitrary fixed
   partition -- returns +18.27%/yr and clears all three benchmarks in both
   halves.  It scores HIGHER than the real-label version (+15.04%).  Matched
   at 12 phases, the real labels average -0.32% excess over the index and
   three arbitrary scrambles average +1.61%, -2.34% and -1.43%: the real
   value sits INSIDE the spread of the nulls.

2. THE SECTOR TERM CONTRIBUTES NOTHING OVER THE NAME TERM IT IS BLENDED
   WITH.  Sector-only (w_name=0) returns +8.28% against an index at +8.29%.
   Name-only, no sector anywhere, returns +16.70% -- BETTER than the sector
   tilt's +13.54%.  Every point of the tilt's apparent edge is name momentum;
   the sector half is a decoration that costs return.

3. AND THE PASSES THEMSELVES ARE A PHASE ARTEFACT, WHICH IS THE MORE GENERAL
   FINDING.  The harness only ever walks at offset=0.  An annual rebalance
   over 19.5 years is ~19 decisions, and `offset` slides WHICH 19 days those
   are without changing the rule or the data.  Re-run on 12 phases, every
   passer collapses: sector tilt top20 mean -0.37% (5/12 phases positive),
   name-mom top20 mean -1.67% (3/12).  offset=0 is at or near the MAXIMUM
   phase for every rule tested, because the benchmark's own CAGR swings from
   +6.02% to +12.46% purely on which day of the year the walk lands, and
   offset=0 draws the second-weakest index window of the twelve.  A19's error
   class -- comparing quantities measured over different windows -- surviving
   inside a harness built to prevent it.

WHAT DID REPLICATE, and it is the repo's own prior, not a discovery: cutting
turnover is worth more than any selection here.  The SAME sector rule returns
+3.57% net quarterly and +8.59% annual; cost falls 1.39% -> 0.59%/yr and gross
rises 5.02% -> 9.23%, so most of the gain is not the toll but not selling
winners.  Drifting weights (BLOCK 4) adds a further +0.94% and is charged for
drift it never traded, so it is measured conservatively.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from bhbench import Bench, load, report                        # noqa: E402

SECTORS = os.path.join("data", "reference", "idx_classification.parquet")
LOOKBACKS = (21, 63, 126, 252)
MIN_MEMBERS = 5          # a sector ranked on fewer names is ranked on noise


# --------------------------------------------------------------- panel build
def sector_map() -> pd.Series:
    d = pd.read_parquet(SECTORS)
    return d.set_index("ticker")["sector"]


def build(min_tv: float = 1e9) -> pd.DataFrame:
    """Panel + sector label + strictly-trailing returns, computed per ticker."""
    P = load(min_tv)
    P["sector"] = P["ticker"].map(sector_map())
    g = P.groupby("ticker", sort=False)["adj_close"]
    for n in LOOKBACKS:
        P[f"ret{n}"] = g.transform(lambda s, n=n: s / s.shift(n) - 1.0)
    #  12-1: the classic momentum window, last month skipped for reversal.
    P["ret252_21"] = g.transform(lambda s: s.shift(21) / s.shift(252) - 1.0)
    return P


def coverage(P: pd.DataFrame) -> str:
    e = P[P["elig"]]
    have = e["sector"].notna().mean()
    n_lost = e.loc[e["sector"].isna(), "ticker"].nunique()
    return (f"sector coverage on ELIGIBLE bars: {have:.1%} "
            f"({n_lost} distinct eligible names have no label)")


# ------------------------------------------------------------- the selectors
def sector_scores(day: pd.DataFrame, col: str, agg: str) -> pd.Series:
    """Sector return on THIS BAR ONLY, from its own eligible members."""
    d = day.dropna(subset=["sector", col])
    grp = d.groupby("sector")[col]
    s = grp.median() if agg == "median" else grp.mean()
    return s[grp.size() >= MIN_MEMBERS]


def make_select(col: str, n_sec: int = 3, agg: str = "median",
                within: str = "all", n_name: int = 0) -> Callable:
    """Hold names inside the top `n_sec` sectors ranked on `col`.

    within = 'all'      every eligible member, equal weight
             'mom'      the strongest `n_name` members by the same column
             'calm'     H26's strength+calm inside the winning sectors
    """
    def select(day: pd.DataFrame) -> List[Tuple[str, float]]:
        sc = sector_scores(day, col, agg)
        if len(sc) < n_sec + 1:
            return []
        top = set(sc.nlargest(n_sec).index)
        d = day[day["sector"].isin(top)].dropna(subset=[col])
        if within == "mom":
            d = d.nlargest(min(n_name, len(d)), col)
        elif within == "calm":
            if len(d) >= 10:
                d = d[(d["hi52"] >= d["hi52"].quantile(0.5))
                      & (d["vol60"] <= d["vol60"].quantile(0.5))]
        if len(d) < 5:
            return []
        return [(t, 1.0 / len(d)) for t in d["ticker"]]
    return select


class Drift:
    """Membership from `base`, but WEIGHTS ARE LET DRIFT with realised return.

    THE ONE STRUCTURAL IDEA IN THIS SCRIPT.  BH_UNIVERSE returns +10.05%/yr and
    an equal-weight quarterly rebalance of the SAME names returns +1.15%: the
    whole ~9-point gap is that a drifting basket lets a winner compound into a
    mega-cap while a rebalance sells it every quarter.  So a sector rule that
    rebalances to 1/N is fighting a 9-point handicap before it selects
    anything.  This holds incumbents at whatever they have grown to and trades
    ONLY the names entering and leaving the winning sectors.

    It uses no information the bar does not contain: the drift factor is the
    name's own realised return between two marks, which is the past.
    """

    def __init__(self, base: Callable):
        self.base, self.w, self.px, self.last = base, {}, {}, None

    def __call__(self, day: pd.DataFrame) -> List[Tuple[str, float]]:
        d0 = day["date"].iloc[0]
        if self.last is not None and d0 <= self.last:
            self.w, self.px = {}, {}          # a re-run starts flat
        self.last = d0
        px = dict(zip(day["ticker"], day["adj_close"].astype(float)))
        held = {}
        for t, w in self.w.items():
            p1, p0 = px.get(t), self.px.get(t)
            if p1 and p0 and p0 > 0 and np.isfinite(p1):
                held[t] = w * p1 / p0          # grown to whatever it is worth
        tgt = [t for t, _ in (self.base(day) or [])]
        if len(tgt) < 5:
            self.w, self.px = {}, {}
            return []
        keep = {t: held[t] for t in tgt if t in held}
        #  A NEW NAME IS BOUGHT AT THE AVERAGE SIZE OF WHAT IS ALREADY THERE,
        #  not at 1/N of the whole book -- 1/N would force a rebalance of the
        #  incumbents through the back door and undo the entire point.
        base_w = float(np.mean(list(keep.values()))) if keep else 1.0
        new = {t: base_w for t in tgt if t not in keep}
        self.w = {**keep, **new}
        tot = sum(self.w.values())
        self.w = {t: w / tot for t, w in self.w.items()}
        self.px = {t: px[t] for t in self.w if t in px}
        return list(self.w.items())


def fake_sector(P: pd.DataFrame, seed: int = 0) -> pd.Series:
    """A FIXED, ARBITRARY partition of the same names into 11 groups.

    THE CONTROL THAT DECIDES THIS FAMILY.  Beating the harness's random basket
    only says the score carries information; it does not say the SECTOR carries
    it, because a sector's trailing return is a weighted average of its
    members' trailing returns and could be nothing but name momentum wearing a
    hat.  Scrambling the labels once, permanently, keeps the group SIZES and
    the aggregation ARITHMETIC identical and destroys only the economics.  If
    the fake sectors do as well, the family is dead however good the numbers.
    """
    tk = np.sort(P["ticker"].unique())
    lab = np.sort(P["sector"].dropna().unique())
    r = np.random.default_rng(seed)
    return pd.Series(r.choice(lab, len(tk)), index=tk)


def make_namemom(col: str, n_name: int = 20) -> Callable:
    """NO SECTOR ANYWHERE.  Top `n_name` names by their own trailing return.

    THE CONTROL THAT DECIDES WHETHER THIS FAMILY EXISTS.  The fake-sector arm
    tests whether the LABELS carry economics; this tests whether the sector
    term contributes anything AT ALL over the name score it is blended with.
    A19 records the missing comparison as the error class that manufactures
    results, and a sector strategy whose name-only twin does just as well is
    a momentum strategy with a sector-shaped decoration on it.
    """
    def select(day: pd.DataFrame) -> List[Tuple[str, float]]:
        d = day.dropna(subset=[col])
        if len(d) < 20:
            return []
        d = d.nlargest(min(n_name, len(d)), col)
        if len(d) < 5:
            return []
        return [(t, 1.0 / len(d)) for t in d["ticker"]]
    return select


def make_tilt(col: str, agg: str = "median", n_name: int = 20,
              w_sec: float = 1.0, w_name: float = 1.0) -> Callable:
    """No sector CUT -- a sector TILT: pick names on (sector score + own score).

    Registered because the cut is a coarse instrument.  Eleven sectors over a
    ~120-name universe means the top-3 cut moves ~30 names at a time, so a
    single sector flip rotates a quarter of the book.  A continuous blend of
    the two z-scores should carry the same information at lower turnover, and
    TURNOVER IS THE ENEMY is the one thing this repo is certain of.
    """
    def select(day: pd.DataFrame) -> List[Tuple[str, float]]:
        sc = sector_scores(day, col, agg)
        d = day.dropna(subset=["sector", col])
        d = d[d["sector"].isin(sc.index)]
        if len(d) < 20:
            return []
        z = lambda s: (s - s.mean()) / (s.std() + 1e-12)          # noqa: E731
        blend = w_sec * z(d["sector"].map(sc)) + w_name * z(d[col])
        d = d.assign(_b=blend).nlargest(min(n_name, len(d)), "_b")
        if len(d) < 5:
            return []
        return [(t, 1.0 / len(d)) for t in d["ticker"]]
    return select


# --------------------------------------------------------------------- drive
def run(P: pd.DataFrame, variants: List[Tuple], draws: int = 6) -> List[Dict]:
    B = Bench(P)
    out = []
    for label, sel, freq in variants:
        v = B.evaluate(sel, label=label, freq=freq, draws=draws)
        print(report(v)); print()
        out.append(v)
    return out


def summary(rows: List[Dict]) -> str:
    L = [f"{'variant':<44}{'CAGR':>8}{'idx':>8}{'univ':>8}{'picks':>8}"
         f"{'rand':>8}  both-halves      PASS"]
    for v in rows:
        if not v.get("ok"):
            L.append(f"{v['label']:<44}  did not run"); continue
        bh = (("I" if v["both_halves_index"] else ".")
              + ("U" if v["both_halves_universe"] else ".")
              + ("P" if v["both_halves_picks"] else "."))
        L.append(f"{v['label']:<44}{v['cagr']:>8.2%}{v['bh_index']:>8.2%}"
                 f"{v['bh_universe']:>8.2%}{v['bh_picks']:>8.2%}"
                 f"{v['random']:>8.2%}  {bh:<15}"
                 f"{'PASS' if v['PASS'] else 'fail'}")
    return "\n".join(L)


def phase_check(B: Bench, factory: Callable, freq: int, name: str,
                n_phase: int = 6) -> List[float]:
    """THE SAME RULE STARTED ON A DIFFERENT DAY.

    An annual rebalance over 21 years is ~20 decisions.  `offset` slides which
    20 days those are.  A33 measured a within-frequency PHASE spread of 2.93%
    on this panel and warned that differences smaller than that are not
    readable.  A rule whose edge survives only at offset 0 has not been
    measured, it has been sampled once.
    """
    print(f"-- phase robustness: {name}, freq={freq}")
    print(f"   {'offset':>7}{'CAGR':>9}{'index':>9}{'univ':>9}{'picks':>9}"
          f"{'  vs idx':>10}")
    d = []
    for off in range(0, freq, max(freq // n_phase, 1)):
        r = B.walk(factory(), freq=freq, offset=off)
        if not r:
            continue
        a0, b1 = r["start"], r["end"]
        uni0 = B.P[(B.P["date"] == a0) & B.P["elig"]]["ticker"].tolist()
        i, u = B.index_cagr(a0, b1), B.hold_basket(uni0, a0, b1)
        p = B.hold_basket(r["first_basket"], a0, b1)
        d.append(r["cagr"] - i)
        print(f"   {off:>7}{r['cagr']:>9.2%}{i:>9.2%}{u:>9.2%}{p:>9.2%}"
              f"{r['cagr'] - i:>10.2%}")
    if d:
        print(f"   spread vs index: mean {np.mean(d):+.2%}, "
              f"sd {np.std(d, ddof=1):.2%}, "
              f"min {min(d):+.2%}, max {max(d):+.2%}, "
              f"positive in {sum(x > 0 for x in d)}/{len(d)} phases")
    print()
    return d


def block(name: str, V: List[Tuple], P: pd.DataFrame,
          acc: List[Dict]) -> None:
    print("=" * 78); print(name); print("=" * 78)
    acc.extend(run(P, V))


def main() -> None:
    P = build()
    print(coverage(P)); print()
    acc: List[Dict] = []

    block("BLOCK 1 -- lookback sweep, quarterly, equal-weight rebalance", [
        (f"top3 ret{n} allmem  q", make_select(f"ret{n}", 3), 63)
        for n in (63, 126, 252)
    ] + [("top3 ret252_21 allmem  q", make_select("ret252_21", 3), 63)], P, acc)

    block("BLOCK 2 -- holding period: the same rule traded less often", [
        ("top3 ret252 allmem  semiannual", make_select("ret252", 3), 126),
        ("top3 ret252 allmem  annual", make_select("ret252", 3), 252),
        ("top3 ret252 allmem  biennial", make_select("ret252", 3), 504),
    ], P, acc)

    block("BLOCK 3 -- breadth: how many sectors, and what inside them", [
        ("top2 ret252 allmem  annual", make_select("ret252", 2), 252),
        ("top5 ret252 allmem  annual", make_select("ret252", 5), 252),
        ("top3 ret252 mean-agg annual",
         make_select("ret252", 3, agg="mean"), 252),
        ("top3 ret252 x strength+calm annual",
         make_select("ret252", 3, within="calm"), 252),
        ("top3 ret252 x top15 name-mom annual",
         make_select("ret252", 3, within="mom", n_name=15), 252),
        ("sector TILT ret252 top20 annual", make_tilt("ret252"), 252),
    ], P, acc)

    block("BLOCK 4 -- DRIFTING WEIGHTS: trade membership, never the winners", [
        ("DRIFT top3 ret252 allmem  annual",
         Drift(make_select("ret252", 3)), 252),
        ("DRIFT top3 ret252 allmem  quarterly",
         Drift(make_select("ret252", 3)), 63),
        ("DRIFT top5 ret252 allmem  annual",
         Drift(make_select("ret252", 5)), 252),
        ("DRIFT top3 ret126 allmem  annual",
         Drift(make_select("ret126", 3)), 252),
        ("DRIFT top3 ret252 allmem  biennial",
         Drift(make_select("ret252", 3)), 504),
    ], P, acc)

    #  ---- the control that decides the family ----------------------------
    F = P.copy()
    F["sector"] = F["ticker"].map(fake_sector(P, seed=0))
    block("BLOCK 5 -- FAKE SECTORS: identical arithmetic, scrambled economics",
          [("FAKE top3 ret252 allmem  annual", make_select("ret252", 3), 252),
           ("FAKE DRIFT top3 ret252 annual",
            Drift(make_select("ret252", 3)), 252),
           ("FAKE top3 ret63 allmem  quarterly",
            make_select("ret63", 3), 63)], F, acc)

    block("BLOCK 6 -- DOES THE SECTOR TERM DO ANYTHING?  the deciding controls",
          [("CTRL name-mom top20 NO SECTOR annual",
            make_namemom("ret252", 20), 252),
           ("CTRL name-mom top15 NO SECTOR annual",
            make_namemom("ret252", 15), 252),
           ("CTRL name-mom top30 NO SECTOR annual",
            make_namemom("ret252", 30), 252),
           ("TILT sector-only (w_name=0) top20 annual",
            make_tilt("ret252", n_name=20, w_name=0.0), 252),
           ("TILT w_sec=0.5 top20 annual",
            make_tilt("ret252", n_name=20, w_sec=0.5), 252),
           ("TILT top10 annual", make_tilt("ret252", n_name=10), 252),
           ("TILT top15 annual", make_tilt("ret252", n_name=15), 252),
           ("TILT top30 annual", make_tilt("ret252", n_name=30), 252),
           ("TILT top40 annual", make_tilt("ret252", n_name=40), 252),
           ("TILT ret126 top20 annual", make_tilt("ret126", n_name=20), 252),
           ("TILT ret252_21 top20 annual",
            make_tilt("ret252_21", n_name=20), 252),
           ("TILT ret252 top20 biennial",
            make_tilt("ret252", n_name=20), 504),
           ("TILT ret252 top20 semiannual",
            make_tilt("ret252", n_name=20), 126)], P, acc)

    block("BLOCK 7 -- FAKE SECTORS run through the TILT itself",
          [("FAKE TILT ret252 top20 annual",
            make_tilt("ret252", n_name=20), 252),
           ("FAKE TILT ret252 top15 annual",
            make_tilt("ret252", n_name=15), 252)], F, acc)

    B = Bench(P)
    print("=" * 78); print("BLOCK 8 -- phase robustness"); print("=" * 78)
    phase_check(B, lambda: make_tilt("ret252", n_name=20), 252,
                "TILT ret252 top20")
    phase_check(B, lambda: make_namemom("ret252", 20), 252,
                "CTRL name-mom top20 (no sector)")
    phase_check(B, lambda: make_select("ret252", 3), 252,
                "top3 ret252 allmem")

    print("=" * 78)
    print("BLOCK 9 -- EVERY offset-0 PASS, re-run on 12 rebalance PHASES")
    print("THE HARNESS ONLY EVER TESTS offset=0.  An annual walk over 19.5yr")
    print("is ~19 decisions; `offset` slides WHICH 19 days those are, using")
    print("the same rule and the same data.  A rule that only wins on one")
    print("phase has been sampled once, not measured.")
    print("=" * 78)
    PASSERS = [
        ("sector TILT ret252 top20", lambda: make_tilt("ret252", n_name=20)),
        ("TILT w_sec=0.5 top20",
         lambda: make_tilt("ret252", n_name=20, w_sec=0.5)),
        ("TILT top10", lambda: make_tilt("ret252", n_name=10)),
        ("TILT top15", lambda: make_tilt("ret252", n_name=15)),
        ("CTRL name-mom top20 (NO SECTOR)",
         lambda: make_namemom("ret252", 20)),
        ("CTRL name-mom top15 (NO SECTOR)",
         lambda: make_namemom("ret252", 15)),
    ]
    tab = []
    for nm, fac in PASSERS:
        tab.append((nm, phase_check(B, fac, 252, nm, n_phase=12)))
    FB = Bench(F)
    tab.append(("FAKE TILT top15 (scrambled labels)",
                phase_check(FB, lambda: make_tilt("ret252", n_name=15), 252,
                            "FAKE TILT top15 (scrambled labels)",
                            n_phase=12)))
    print("PHASE VERDICT — excess CAGR over BH index, by rebalance phase")
    print(f"  {'rule':<36}{'mean':>8}{'sd':>8}{'min':>8}{'max':>8}"
          f"{'  phases>0':>11}")
    for nm, d in tab:
        if not d:
            continue
        print(f"  {nm:<36}{np.mean(d):>8.2%}{np.std(d, ddof=1):>8.2%}"
              f"{min(d):>8.2%}{max(d):>8.2%}"
              f"{f'{sum(x > 0 for x in d)}/{len(d)}':>11}")

    print("=" * 78)
    print("BLOCK 10 -- THE FAMILY VERDICT: real labels vs scrambled vs none")
    print("Matched configuration, matched turnover, 12 phases each.  The only")
    print("thing that varies is whether the sector labels mean anything.")
    print("=" * 78)
    print(f"  {'arm':<38}{'mean':>8}{'sd':>8}{'phases>0':>11}")
    for n_name in (15, 20):
        rows = [(f"REAL sectors, tilt top{n_name}", P,
                 lambda n=n_name: make_tilt("ret252", n_name=n))]
        for s in (0, 1, 2):
            Fs = P.copy()
            Fs["sector"] = Fs["ticker"].map(fake_sector(P, seed=s))
            rows.append((f"SCRAMBLED sectors seed {s}, tilt top{n_name}", Fs,
                         lambda n=n_name: make_tilt("ret252", n_name=n)))
        rows.append((f"NO sector at all, name-mom top{n_name}", P,
                     lambda n=n_name: make_namemom("ret252", n)))
        for nm, panel, fac in rows:
            Bx = Bench(panel)
            d = []
            for off in range(0, 252, 21):
                r = Bx.walk(fac(), freq=252, offset=off)
                if r:
                    d.append(r["cagr"] - Bx.index_cagr(r["start"], r["end"]))
            if d:
                print(f"  {nm:<38}{np.mean(d):>8.2%}"
                      f"{np.std(d, ddof=1):>8.2%}"
                      f"{f'{sum(x > 0 for x in d)}/{len(d)}':>11}")
        print()

    print("=" * 78)
    print(f"SUMMARY  ({len(acc)} variants)")
    print("=" * 78)
    print(summary(acc))
    gp = [v for v in acc if v.get("ok")]
    print()
    print("gross (pre-cost) CAGR, same order:")
    for v in gp:
        print(f"  {v['label']:<40}{v['gross']:+7.2%} gross  "
              f"{v['cagr']:+7.2%} net  cost {v['cost_yr']:.2%}/yr  "
              f"turnover {v['turnover']:.0%}")
    print()
    print(f"PASSING VARIANTS: {sum(v.get('PASS', False) for v in acc)} "
          f"of {len(acc)}")


if __name__ == "__main__":
    main()
