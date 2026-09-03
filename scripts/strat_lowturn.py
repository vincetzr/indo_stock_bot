#!/usr/bin/env python3
"""H54 family — THE STRENGTH+CALM SCREEN AT MINIMUM TURNOVER.

THE QUESTION. `hi52` top decile AND `vol60` below median is the best selection
signal this repo has found (A24/H26, A38/H52: ~+12.9%/yr gross over a random
basket from the same universe). It has only ever been tested with a HARD
rebalance — every quarter or month, sell everything that left the screen, buy
everything that entered, reset to equal weight. A33/H43 then measured the same
screen's edge DECAYING with holding period, and the H54 brief records the
larger fact: owning every eligible name and rebalancing quarterly returns
+1.15%/yr while owning the same names and never touching them returns
+10.05%/yr. Turnover is the enemy.

So: does the screen's edge survive when you STOP PAYING FOR IT?

THREE SEPARABLE CHANGES, AND THEY ARE TESTED SEPARATELY BECAUSE THEY ARE
DIFFERENT CLAIMS.

  BUFFER      buy in the top decile of `hi52`, but only SELL when the name
              falls out of a much wider band (below the MEDIAN, say). A name
              that is merely no longer the strongest is kept. This is the
              standard index-construction device for cutting turnover without
              changing the tilt, and it is what the brief asks for.

  DRIFT       do not reset to equal weight. A name that trebles becomes three
              units of the book. This is the mechanism behind the +1.15% vs
              +10.05% gap: rebalancing systematically sells the compounder.

  SLOW CLOCK  look less often. freq=252 instead of 63.

Each is tested alone and in combination, against the hard-rebalance version of
the identical screen, so the answer is not "the low-turnover version wins"
without saying WHICH of the three did the work.

HOW DRIFT IS EXPRESSED INSIDE A HARNESS THAT REBALANCES TO WHATEVER YOU RETURN.
`Bench.walk` rebalances to the weights `select` returns. So a drifting book is
produced by RETURNING THE DRIFTED WEIGHTS: this module carries the book in
value space, marks it to market at each mark with its own point-in-time price
accessor (never a price stamped after the mark), and hands the drifted vector
back. The harness normalises it, so the period return it computes is exactly
the return of the un-rebalanced book.

TWO PLACES THAT IS NOT EXACT, BOTH MEASURED BELOW RATHER THAN ASSERTED.

  1. SPURIOUS TURNOVER — AGAINST the strategy, and it is left in. `walk`
     charges 0.5*sum|w_t - w_{t-1}| * toll at every mark. Drift moves weights
     with no trade, so a book that traded nothing is still billed. Every
     variant therefore prints `traded` (the turnover this module actually
     transacted) beside the harness's `turnover`, and the gap is the bill for
     nothing. It is NOT netted out of any reported CAGR.

  2. COSTLESS REDEPLOYMENT OF DELISTING PROCEEDS — FOR the strategy. When a
     held name stops printing, its value is realised at its last print and
     spread over the survivors free. The buy-and-hold benchmarks do not get
     that: `hold_basket` keeps the dead name's terminal return in the mean
     forever. The count of such events is printed per variant so the size of
     the favour is visible.

A STRUCTURAL WARNING ABOUT BH_PICKS, WHICH IS THE BAR THAT MATTERS HERE.
`bh_picks` buys the strategy's OWN first basket and holds it to the end. The
lower the turnover, the closer the strategy IS to that benchmark, so this
family is being pushed toward the one bar it cannot clear by construction. That
is the correct tension and it is not worked around: a rule that cannot beat
owning its own first ten names has not earned its trading.

EVERYTHING IS IN SAMPLE. The 24-month holdout was spent long ago (A21).

Usage:  python3 scripts/strat_lowturn.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from bhbench import Bench, load, report          # noqa: E402

OUT = os.path.join("reports", "strat_lowturn.txt")


# --------------------------------------------------------------------- prices
class PIT:
    """Point-in-time adjusted close. Never reads a bar stamped after `day`."""

    def __init__(self, P: pd.DataFrame):
        self.d: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for tk, g in P.groupby("ticker", sort=False):
            self.d[tk] = (g["date"].to_numpy(), g["adj_close"].to_numpy(float))

    def last_on_or_before(self, tk: str, day) -> float:
        dt, px = self.d.get(tk, (None, None))
        if dt is None:
            return float("nan")
        j = np.searchsorted(dt, day, side="right") - 1
        return float(px[j]) if j >= 0 else float("nan")

    def prints_on(self, tk: str, day) -> bool:
        dt, _ = self.d.get(tk, (None, None))
        if dt is None:
            return False
        i = np.searchsorted(dt, day)
        return bool(i < len(dt) and dt[i] == day)


# ------------------------------------------------------------------ selection
class Screen:
    """Strength+calm with an optional hold buffer and optional weight drift.

    Stateful across marks, which is legitimate because `walk` calls it in
    chronological order and this object only ever looks at the bar it is handed
    and at prices dated on or before that bar.

    entry_hi / entry_vol   percentile gates to BUY   (0.90 / 0.50 = the screen)
    keep_hi  / keep_vol    percentile gates to KEEP  (wider = lower turnover)
    n                      target number of holdings
    drift                  True = let weights run; False = reset to equal
    """

    def __init__(self, pit: PIT, n: int = 15,
                 entry_hi: float = 0.90, entry_vol: float = 0.50,
                 keep_hi: float = 0.50, keep_vol: float = 0.80,
                 drift: bool = True, keep_abs_hi: float | None = None):
        self.pit = pit
        self.n = n
        self.entry_hi, self.entry_vol = entry_hi, entry_vol
        self.keep_hi, self.keep_vol = keep_hi, keep_vol
        self.keep_abs_hi = keep_abs_hi     # ABSOLUTE give-back band, not a rank
        self.drift = drift
        self.reset()

    def reset(self) -> None:
        self.book: Dict[str, float] = {}     # ticker -> rupiah value
        self.last_mark = None
        self.traded: List[float] = []        # real one-way-ish turnover
        self.dead = 0                        # names realised at last print
        self.marks = 0

    # -- the callable the harness sees
    def __call__(self, day: pd.DataFrame) -> List[Tuple[str, float]]:
        dt = day["date"].iloc[0]
        self.marks += 1

        # 1. mark the existing book to market, and realise anything that has
        #    stopped printing (a delisting is not a hold forever).
        if self.book and self.last_mark is not None:
            newbook = {}
            for tk, val in self.book.items():
                p0 = self.pit.last_on_or_before(tk, self.last_mark)
                p1 = self.pit.last_on_or_before(tk, dt)
                if not (np.isfinite(p0) and p0 > 0 and np.isfinite(p1)
                        and p1 > 0):
                    continue
                v = val * (p1 / p0)
                if self.pit.prints_on(tk, dt):
                    newbook[tk] = v
                else:
                    self.dead += 1           # realised, proceeds redeployed
            self.book = newbook
        self.last_mark = dt

        # 2. cross-sectional percentile ranks WITHIN this bar only.
        g = day.dropna(subset=["hi52", "vol60"])
        g = g[g["vol60"] > 0]
        if len(g) < 20:
            return []
        hi = g["hi52"].rank(pct=True)
        vo = g["vol60"].rank(pct=True)
        tick = g["ticker"].to_numpy()
        hip = dict(zip(tick, hi.to_numpy()))
        vop = dict(zip(tick, vo.to_numpy()))
        hab = dict(zip(tick, g["hi52"].to_numpy()))   # raw close/52w-high

        before = dict(self.book)
        tot_before = sum(before.values())

        # 3. SELL only on the wide band. A name merely no longer strongest is
        #    kept; that is the whole point of the buffer.
        for tk in list(self.book):
            if tk not in hip:                       # left the eligible set
                self.book.pop(tk)
                continue
            if hip[tk] < self.keep_hi or vop[tk] > self.keep_vol:
                self.book.pop(tk)
                continue
            #  An ABSOLUTE give-back band is a different rule from a rank band:
            #  in a bear market every name falls, so a name can hold its
            #  cross-sectional rank while sitting 40% below its own high.
            #  A17/H19 measured the recovery median crossing at -10 to -15%.
            if self.keep_abs_hi is not None and hab[tk] < self.keep_abs_hi:
                self.book.pop(tk)

        # 4. BUY from the strict screen, best strength first, up to n.
        cand = [t for t in tick
                if hip[t] >= self.entry_hi and vop[t] <= self.entry_vol
                and t not in self.book]
        cand.sort(key=lambda t: -hip[t])
        room = self.n - len(self.book)
        if room > 0 and cand:
            base = sum(self.book.values())
            if base <= 0:
                base = 1.0 if tot_before <= 0 else tot_before
            unit = base / max(len(self.book), 1) if self.book \
                else base / min(self.n, len(cand))
            for t in cand[:room]:
                self.book[t] = unit

        if len(self.book) < 5:
            #  Nothing valid to hold. Give the book back as-is (the harness
            #  will skip the mark) rather than forcing a basket.
            return [(t, v) for t, v in self.book.items()]

        # 5. record the turnover actually TRANSACTED, for the honesty column.
        tot_after = sum(self.book.values())
        if tot_before > 0 and tot_after > 0:
            wb = {t: v / tot_before for t, v in before.items()}
            #  what the book would have been with no trade, drifted:
            wa = {t: v / tot_after for t, v in self.book.items()}
            keys = set(wb) | set(wa)
            self.traded.append(
                0.5 * sum(abs(wa.get(k, 0.0) - wb.get(k, 0.0)) for k in keys))

        if not self.drift:
            k = len(self.book)
            eq = tot_after / k if tot_after > 0 else 1.0 / k
            self.book = {t: eq for t in self.book}

        return [(t, v) for t, v in self.book.items()]


# ---------------------------------------------------------------------- run
def bench_halves(B: Bench, r: Dict) -> Dict[str, float]:
    """The three bars measured over each HALF of the strategy's own window.

    `Bench.evaluate` computes these internally to set `both_halves_*` but only
    returns the boolean. A verdict flag with no number behind it cannot be read,
    so they are recomputed here from a second deterministic walk.
    """
    a0, b1 = r["start"], r["end"]
    mid = r["curve"][len(r["curve"]) // 2][0]
    uni0 = B.P[(B.P["date"] == a0) & B.P["elig"]]["ticker"].tolist()
    fb = r["first_basket"]
    return {"e_index": B.index_cagr(a0, mid), "l_index": B.index_cagr(mid, b1),
            "e_universe": B.hold_basket(uni0, a0, mid),
            "l_universe": B.hold_basket(uni0, mid, b1),
            "e_picks": B.hold_basket(fb, a0, mid),
            "l_picks": B.hold_basket(fb, mid, b1)}


def run(B: Bench, pit: PIT, label: str, freq: int, **kw) -> Tuple[Dict, Screen]:
    s = Screen(pit, **kw)
    v = B.evaluate(s, label=label, freq=freq)
    if v.get("ok"):
        r = B.walk(Screen(pit, **kw), freq=freq)     # identical, deterministic
        v.update(bench_halves(B, r))
    return v, s


def offsets(B: Bench, pit: PIT, label: str, freq: int, step: int,
            emit, **kw) -> None:
    """Re-run one variant from every start phase inside the rebalance cycle.

    A single walk is ONE draw: it picks one basket on one date and every later
    holding descends from it. This repo has recorded the smallest-cell /
    single-draw trap three times, so the phase spread is printed rather than
    the phase-0 number alone.
    """
    emit("")
    emit(f"START-PHASE SENSITIVITY — {label}, freq={freq}")
    emit(f"  {'phase':>5}{'start':>12}{'cagr':>9}{'index':>9}{'univ':>9}"
         f"{'picks':>9}   beats")
    got = []
    for off in range(0, freq, step):
        r = B.walk(Screen(pit, **kw), freq=freq, offset=off)
        if not r:
            continue
        a0, b1 = r["start"], r["end"]
        uni0 = B.P[(B.P["date"] == a0) & B.P["elig"]]["ticker"].tolist()
        bi = B.index_cagr(a0, b1)
        bu = B.hold_basket(uni0, a0, b1)
        bp = B.hold_basket(r["first_basket"], a0, b1)
        c = r["cagr"]
        flags = ("I" if c > bi else "-") + ("U" if c > bu else "-") \
            + ("P" if c > bp else "-")
        emit(f"  {off:>5}{str(pd.Timestamp(a0).date()):>12}{c:>9.2%}"
             f"{bi:>9.2%}{bu:>9.2%}{bp:>9.2%}   {flags}")
        got.append((c, bp))
    if len(got) > 1:
        cs = np.array([g[0] for g in got])
        ex = np.array([g[0] - g[1] for g in got])
        emit(f"  cagr across phases: mean {cs.mean():+.2%}, "
             f"sd {cs.std(ddof=1):.2%}, min {cs.min():+.2%}, "
             f"max {cs.max():+.2%}")
        emit(f"  excess over BH_PICKS: mean {ex.mean():+.2%}, "
             f"sd {ex.std(ddof=1):.2%}, "
             f"positive in {int((ex > 0).sum())} of {len(ex)} phases")


def main() -> None:
    P = load()
    B = Bench(P)
    pit = PIT(P)

    lines: List[str] = []

    def emit(txt: str) -> None:
        print(txt)
        lines.append(txt)

    emit(__doc__.split("Usage:")[0].rstrip())
    emit("=" * 78)

    #  Every variant registered here BEFORE any was run. The strict screen is
    #  entry_hi=0.90 / entry_vol=0.50 throughout; only the exit band, the
    #  weighting and the clock move.
    VARIANTS = [
        # label,                       freq, kwargs
        ("A hard quarterly, eq-wt (control)",   63,
         dict(keep_hi=0.90, keep_vol=0.50, drift=False)),
        ("B buffer 0.50, eq-wt",                63,
         dict(keep_hi=0.50, keep_vol=0.80, drift=False)),
        ("C buffer 0.50, drift",                63,
         dict(keep_hi=0.50, keep_vol=0.80, drift=True)),
        ("D no buffer, drift",                  63,
         dict(keep_hi=0.90, keep_vol=0.50, drift=True)),
        ("E buffer 0.30, drift",                63,
         dict(keep_hi=0.30, keep_vol=0.90, drift=True)),
        ("F buffer 0.70, drift",                63,
         dict(keep_hi=0.70, keep_vol=0.65, drift=True)),
        ("G buffer 0.50, drift, annual",       252,
         dict(keep_hi=0.50, keep_vol=0.80, drift=True)),
        ("H buffer 0.50, drift, semiannual",   126,
         dict(keep_hi=0.50, keep_vol=0.80, drift=True)),
        ("I buffer 0.50, drift, monthly",       21,
         dict(keep_hi=0.50, keep_vol=0.80, drift=True)),
        ("J buffer 0.30, drift, annual, n=25", 252,
         dict(keep_hi=0.30, keep_vol=0.90, drift=True, n=25)),
        ("K buffer 0.30, drift, annual, n=10", 252,
         dict(keep_hi=0.30, keep_vol=0.90, drift=True, n=10)),
        ("L sell-never (elig only), drift",    252,
         dict(keep_hi=0.0, keep_vol=1.0, drift=True)),

        #  BATCH 2, registered AFTER seeing batch 1 and marked as such. Batch 1
        #  says the buffer costs more gross return than it saves in cost, and
        #  the obvious mechanism is that a rank buffer holds names all the way
        #  down: `hi52` rank can stay high while the name itself is 40% off its
        #  own high, because in a bear market everything falls together. These
        #  test that mechanism with an ABSOLUTE give-back band instead.
        ("M buffer 0.50 + abs 0.85, drift",     63,
         dict(keep_hi=0.50, keep_vol=0.80, drift=True, keep_abs_hi=0.85)),
        ("N buffer 0.50 + abs 0.75, drift",     63,
         dict(keep_hi=0.50, keep_vol=0.80, drift=True, keep_abs_hi=0.75)),
        ("O buffer 0.30 + abs 0.75, drift",     63,
         dict(keep_hi=0.30, keep_vol=0.90, drift=True, keep_abs_hi=0.75)),
        ("P buffer 0.50 + abs 0.75, annual",   252,
         dict(keep_hi=0.50, keep_vol=0.80, drift=True, keep_abs_hi=0.75)),
        ("Q buffer 0.50, drift, n=25",          63,
         dict(keep_hi=0.50, keep_vol=0.80, drift=True, n=25)),
        ("R buffer 0.50, drift, n=8",           63,
         dict(keep_hi=0.50, keep_vol=0.80, drift=True, n=8)),
    ]

    results = []
    for label, freq, kw in VARIANTS:
        v, s = run(B, pit, label, freq, **kw)
        emit("")
        emit(report(v))
        if v.get("ok"):
            tr = float(np.mean(s.traded)) if s.traded else float("nan")
            emit(f"  traded turnover     {tr:8.1%}   "
                 f"(harness charges {v['turnover']:.1%} — the gap is drift "
                 f"billed as a trade)")
            emit(f"  delistings realised {s.dead:8d}")
            v["traded"] = tr
            v["dead"] = s.dead
            results.append(v)

    emit("")
    emit("=" * 78)
    emit("SUMMARY — CAGR net of cost, and which bars were cleared")
    emit(f"{'variant':<38}{'gross':>8}{'cagr':>8}{'trd':>6}{'cost':>7}"
         f"{'idx':>8}{'univ':>8}{'picks':>8}{'rand':>8}  halves  PASS")
    for v in results:
        bh = ("I" if v["both_halves_index"] else "-") \
            + ("U" if v["both_halves_universe"] else "-") \
            + ("P" if v["both_halves_picks"] else "-")
        emit(f"{v['label']:<38}{v['gross']:>8.2%}{v['cagr']:>8.2%}"
             f"{v['traded']:>6.0%}{v['cost_yr']:>7.2%}"
             f"{v['bh_index']:>8.2%}{v['bh_universe']:>8.2%}"
             f"{v['bh_picks']:>8.2%}{v['random']:>8.2%}"
             f"  {bh:<6}{'PASS' if v['PASS'] else 'fail'}")

    #  THE FAMILY'S QUESTION, ANSWERED LIKE FOR LIKE. Variants A/B/C/D share
    #  one window, so their gross returns are directly comparable and the
    #  decomposition needs no adjustment. Variants on other clocks start on
    #  other dates and their benchmarks differ; they are NOT compared here.
    by = {v["label"][0]: v for v in results}

    def dec(name: str, x: str, y: str) -> None:
        emit(f"  {name:<42}"
             f"gross {100 * (by[y]['gross'] - by[x]['gross']):+5.2f} pts, "
             f"cost {100 * (by[x]['cost_yr'] - by[y]['cost_yr']):+5.2f} saved, "
             f"net {100 * (by[y]['cagr'] - by[x]['cagr']):+5.2f} pts")

    if all(k in by for k in "ABCDMN") and \
            len({by[k]["start"] for k in "ABCDMN"}) == 1:
        emit("")
        emit("DECOMPOSITION — identical window "
             f"{by['A']['start']} → {by['A']['end']}, so gross IS comparable "
             "and no window adjustment is needed (A19)")
        dec("add rank buffer, equal weight (A->B)", "A", "B")
        dec("add drift, buffer wide (B->C)", "B", "C")
        dec("add drift, no buffer (A->D)", "A", "D")
        dec("rank buffer + drift (A->C)", "A", "C")
        dec("rank buffer + drift + abs 0.85 band (A->M)", "A", "M")
        dec("rank buffer + drift + abs 0.75 band (A->N)", "A", "N")

    #  THE BENCHMARK HALVES, so "both halves = no" says WHICH half and by how
    #  much. A verdict flag without the number behind it is not a result.
    emit("")
    emit("HALF-SPLIT DETAIL — strategy vs each bar, early / late")
    emit(f"{'variant':<38}{'strat':>16}{'index':>16}{'universe':>16}"
         f"{'picks':>16}")
    for v in results:
        emit(f"{v['label']:<38}"
             f"{v['early']:>7.1%}/{v['late']:>7.1%} "
             f"{v['e_index']:>7.1%}/{v['l_index']:>7.1%} "
             f"{v['e_universe']:>7.1%}/{v['l_universe']:>7.1%} "
             f"{v['e_picks']:>7.1%}/{v['l_picks']:>7.1%}")

    #  Phase sweep on the best-by-CAGR variant AND on the hard-rebalance
    #  control, so the comparison is like for like rather than a sweep run only
    #  on the winner.
    best = max(results, key=lambda v: v["cagr"])
    #  ...and on the only variant that cleared BH_PICKS in BOTH halves, which
    #  is the structurally hardest of the three bars for a low-turnover rule.
    hard = [v["label"] for v in results if v["both_halves_picks"]]
    want = {best["label"], "A hard quarterly, eq-wt (control)", *hard}
    for lab, freq, kw in VARIANTS:
        if lab in want:
            offsets(B, pit, lab, freq, max(freq // 7, 1), emit, **kw)

    emit("")
    emit(f"VARIANTS TRIED: {len(results)}   PASSES: "
         f"{sum(1 for v in results if v['PASS'])}")

    os.makedirs("reports", exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
