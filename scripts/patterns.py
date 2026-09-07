#!/usr/bin/env python3
"""H61 — §14 chart patterns, and the base-rate harness a vision layer would need.

WHAT §14 ASKS FOR AND WHY THIS IS NOT IT. The brief wants a vision model that
looks at a chart and names what it sees. That is not buildable from a script
here — there is no vision endpoint callable from this process — and it would be
the wrong thing to build first anyway. A vision layer that returns "double
bottom, high confidence" is only usable if someone has already measured what a
double bottom is worth. **Every pattern needs a base rate before it is used**,
and this file is that base rate.

Each pattern below is defined PROGRAMMATICALLY and CAUSALLY: it fires on the
bar at which it would have been visible, never at the pivot it is built from.
A32 records why that is the whole game — one non-causal helper anywhere in the
chain (a ZigZag drawn at the pivot instead of the confirmation bar, a centred
window) turns the study into a look-ahead and nothing in the output looks
wrong. `tests/test_patterns.py` truncates the panel and asserts each detector
returns the same firings on what was knowable then.

THE CONTROL IS THE POINT. A28 scored every trend detector against a random
detector spending the SAME NUMBER of flips, and A34 records that a control
denied the treatment's own affordances is a handicap rather than a null. So
each pattern is scored against random bars drawn from the SAME NAMES in the
SAME YEARS at the SAME rate — matched on everything except the shape.

--------------------------------------------------------------------- REGISTERED
Written before any cell was scored.

  P1  At least one classic pattern shows a forward edge over its matched
      control that clears its clustered null. PREDICTION: yes for the
      trend-continuation family (breakout, golden cross, higher-highs), because
      H13 already found momentum-shaped features significant at t above 10 —
      the question this asks is whether the DISCRETE pattern adds anything to
      the continuous feature, and §8 of CLAUDE.md is explicit that discrete
      rules with hand-tuned parameters are the wrong instrument.

  P2  PREDICTED NULL. `digit` — the close ending in 0 or 5 — shows nothing.
      It is read off the price like every other pattern here and means nothing,
      so it is the cheapest possible check that the pipeline is not
      manufacturing its own signal. A9 registered `squeeze` the same way and it
      FIRED at t = +3.55 on two million rows, which is why this is not
      optional.

  P3  No pattern survives costs. PREDICTION: confirmed. H13 measured every one
      of eight registered price features net-negative at every horizon against
      a 1.7-1.9% rebalance cost, and a discrete pattern is a coarser version of
      the same information.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from bhbench import load                                          # noqa: E402
from idxbot.metrics import periods_to_detect                      # noqa: E402

TRIALS = 360          # hypotheses.md after H60
BAR = 0.05 / TRIALS
MIN_FIRES = 300       # a pattern below this is a name wearing a shape
FEE = 0.0056          # A5, quoted separately from the spread (A38)


# --------------------------------------------------------------- detectors
#
# Every detector takes one ticker's frame sorted by date and returns a boolean
# array marking the CONFIRMATION bar. Reading a value at index i uses only
# indices <= i. Any `shift(-k)` in here would be the whole study's undoing.
#
# AND EVERY ONE GOES THROUGH `_b()`, WHICH IS NOT TIDINESS.
# `Series.shift(1)` on a bool column returns OBJECT dtype (it has to hold the
# NaN it introduces), `.fillna(False)` leaves it object, and `~` on an object
# Series applies bitwise NOT to the underlying Python bools: `~False == -1` and
# `~True == -2`, BOTH TRUTHY. So `up & ~up.shift(1).fillna(False)` is `up`
# with the negation silently doing nothing. Measured: the golden cross fired
# 357,925 times on 690,591 eligible bars — 52% of the panel, for a signal that
# should fire a dozen times per name in twenty-six years. The number was
# impossible, which is the only reason it was caught (A30: a statistic that
# cannot occur is the cheapest bug detector available).


def _b(s) -> np.ndarray:
    """A pandas boolean-ish Series -> a real numpy bool array."""
    return s.fillna(False).astype(bool).to_numpy()

def p_breakout(d: pd.DataFrame, look: int = 60, squeeze: float = 0.6
               ) -> np.ndarray:
    """Close above the prior `look`-bar high, out of a compressed range."""
    c = d["adj_close"]
    hi = c.rolling(look, min_periods=look).max().shift(1)
    rng = (c.rolling(20, min_periods=20).max()
           / c.rolling(20, min_periods=20).min() - 1.0).shift(1)
    wide = rng.rolling(250, min_periods=100).median()
    return _b((c > hi) & (rng < squeeze * wide))


def p_golden(d: pd.DataFrame) -> np.ndarray:
    """MA50 crosses above MA200, marked on the crossing bar."""
    c = d["adj_close"]
    a = c.rolling(50, min_periods=50).mean()
    b = c.rolling(200, min_periods=200).mean()
    up = pd.Series(_b(a > b), index=c.index)
    prev = pd.Series(_b(up.shift(1)), index=c.index)
    return _b(up & ~prev)


def p_hh_hl(d: pd.DataFrame, w: int = 20) -> np.ndarray:
    """Three consecutive `w`-bar windows each with a higher high AND low."""
    hi = d["adj_close"].rolling(w, min_periods=w).max()
    lo = d["adj_close"].rolling(w, min_periods=w).min()
    ok = pd.Series(_b((hi > hi.shift(w)) & (lo > lo.shift(w))), index=hi.index)
    return _b(ok & pd.Series(_b(ok.shift(w)), index=ok.index)
              & pd.Series(_b(ok.shift(2 * w)), index=ok.index))


def p_double_bottom(d: pd.DataFrame, w: int = 40, tol: float = 0.05
                    ) -> np.ndarray:
    """Two lows within `tol`, `w` apart, CONFIRMED by clearing the middle high.

    The confirmation bar is what makes this causal. A double bottom drawn at
    the second low is drawn with knowledge that no lower low followed, which is
    exactly A27's Pine ZigZag defect one level up.
    """
    c = d["adj_close"].to_numpy(float)
    n = len(c)
    out = np.zeros(n, dtype=bool)
    if n < 2 * w + 2:
        return out
    #  The first testable bar is 2*w: both windows are then full and c[i-1]
    #  exists. A `2*w + 1` bound discards one bar for no reason, which a
    #  planted-shape test caught by not firing on a textbook example.
    for i in range(2 * w, n):
        left = c[i - 2 * w:i - w]
        right = c[i - w:i]
        if len(left) < w or len(right) < w:
            continue
        l1, l2 = left.min(), right.min()
        if l1 <= 0 or l2 <= 0:
            continue
        if abs(l2 / l1 - 1.0) > tol:
            continue
        mid = max(left.max(), right.max())
        #  fires only when today CLEARS the intervening high for the first time
        if c[i] > mid >= c[i - 1]:
            out[i] = True
    return out


def p_gap_volume(d: pd.DataFrame, gap: float = 0.04, z: float = 2.0
                 ) -> np.ndarray:
    """A jump of `gap` on volume `z` standard deviations above its own norm."""
    r = d["adj_close"].pct_change()
    v = np.log1p(d.get("volume", pd.Series(np.nan, index=d.index)))
    mu = v.rolling(60, min_periods=40).mean()
    sd = v.rolling(60, min_periods=40).std()
    zz = (v - mu) / sd.replace(0.0, np.nan)
    return _b((r > gap) & (zz > z))


def p_digit(d: pd.DataFrame) -> np.ndarray:
    """PREDICTED NULL. The close ends in 0 or 5.

    Read off the price like every pattern above and meaning nothing. Its rate
    is high by construction because IDX prices sit on a fraksi-harga grid, so
    it also checks that the harness is not simply rewarding frequency.
    """
    c = d["close"] if "close" in d else d["adj_close"]
    v = c.round().astype("int64")
    return ((v % 5) == 0).to_numpy()


PATTERNS: Tuple[Tuple[str, Callable], ...] = (
    ("breakout from squeeze", p_breakout),
    ("golden cross", p_golden),
    ("higher highs and lows", p_hh_hl),
    ("double bottom confirmed", p_double_bottom),
    ("gap up on volume", p_gap_volume),
    ("digit (PREDICTED NULL)", p_digit),
)


# ------------------------------------------------------------------ harness

def fire_table(P: pd.DataFrame, fn: Callable, horizon: int) -> pd.DataFrame:
    """Every firing, with the forward return from the confirmation bar."""
    rows = []
    for tk, g in P.groupby("ticker", sort=False):
        g = g.sort_values("date")
        if len(g) < 300:
            continue
        try:
            hit = fn(g)
        except Exception:                                   # noqa: BLE001
            continue
        if hit is None or not hit.any():
            continue
        c = g["adj_close"].to_numpy(float)
        fwd = np.full(len(c), np.nan)
        fwd[:-horizon] = c[horizon:] / c[:-horizon] - 1.0
        elig = g["elig"].to_numpy(bool)
        m = hit & elig & np.isfinite(fwd)
        if not m.any():
            continue
        rows.append(pd.DataFrame({
            "ticker": tk, "date": g["date"].to_numpy()[m],
            "fwd": fwd[m], "lf": np.log1p(np.clip(fwd[m], -0.999, None))}))
    return (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=["ticker", "date", "fwd", "lf"]))


class Pool:
    """Eligible bars and their forward returns, indexed by (ticker, year).

    BUILT ONCE. A first version filtered the whole 2.8m-row panel inside the
    loop over (ticker, year) cells, which is O(cells x panel) and never
    finished. Correct and unusable is its own kind of wrong: a control nobody
    can afford to redraw is a control that quietly gets dropped, which is the
    one thing this study cannot lose.
    """

    def __init__(self, P: pd.DataFrame, horizon: int):
        self.cells: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
        for tk, g in P.groupby("ticker", sort=False):
            g = g.sort_values("date")
            c = g["adj_close"].to_numpy(float)
            if len(c) <= horizon:
                continue
            fwd = np.full(len(c), np.nan)
            fwd[:-horizon] = c[horizon:] / c[:-horizon] - 1.0
            ok = g["elig"].to_numpy(bool) & np.isfinite(fwd)
            dates = g["date"].to_numpy()
            years = g["date"].dt.year.to_numpy()
            for y in np.unique(years[ok]):
                m = ok & (years == y)
                if m.any():
                    self.cells[(tk, int(y))] = (dates[m], fwd[m])

    def draw(self, want: pd.Series, rng) -> pd.DataFrame:
        rows = []
        for (tk, y), k in want.items():
            got = self.cells.get((tk, int(y)))
            if got is None:
                continue
            dates, fwd = got
            idx = rng.choice(len(fwd), min(int(k), len(fwd)), replace=False)
            rows.append(pd.DataFrame({
                "ticker": tk, "date": dates[idx], "fwd": fwd[idx],
                "lf": np.log1p(np.clip(fwd[idx], -0.999, None))}))
        return (pd.concat(rows, ignore_index=True) if rows
                else pd.DataFrame(columns=["ticker", "date", "fwd", "lf"]))


def matched_control(pool: "Pool", fires: pd.DataFrame,
                    seed: int = 0) -> pd.DataFrame:
    """Random bars from the SAME (ticker, year) cells, at the same counts.

    A34: a control denied the treatment's own affordances is a handicap, not a
    null. Drawing from the whole panel would let a pattern that happens to fire
    on liquid names in good years beat a control drawn from everything.
    """
    rng = np.random.default_rng(seed)
    want = (fires.assign(y=fires["date"].dt.year)
            .groupby(["ticker", "y"]).size())
    return pool.draw(want, rng)


def block_null(pool: "Pool", fires: pd.DataFrame,
               draws: int, seed: int = 0) -> Tuple[float, float, int]:
    """The control, redrawn `draws` times. Its spread IS the null.

    Each draw is a fresh matched control, so the null answers "how far from the
    control's own mean does a matched random selection of this size land" —
    which is the question, and it is clustered by construction because the
    draws respect the same (ticker, year) blocks.
    """
    out = []
    for s in range(draws):
        c = matched_control(pool, fires, seed=seed + s)
        if len(c) >= MIN_FIRES:
            out.append(float(c["lf"].mean()))
    a = np.asarray(out)
    return (float(a.mean()), float(a.std(ddof=1)), len(a)) if len(a) > 1 \
        else (float("nan"), float("nan"), len(a))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=63)
    ap.add_argument("--draws", type=int, default=60)
    a = ap.parse_args()

    P = load()
    P = P[P["adj_close"] > 0].copy()
    pool = Pool(P, a.horizon)
    print("H61 — §14 chart patterns, base rates before any vision layer")
    print(f"panel {P['date'].min().date()} -> {P['date'].max().date()}, "
          f"{P['ticker'].nunique()} names, horizon {a.horizon} sessions")
    print(f"cost of one round trip: fee {FEE:.2%} + a fraksi-harga half-spread "
          f"(A38: quoted separately, and the spread is an EXECUTION assumption)")

    #  THE `net` COLUMN IS THE EDGE MINUS COST, NOT THE RAW RETURN MINUS COST.
    #  A first version printed the pattern's own mean forward return net of the
    #  fee, which read "+6.30%" for a pattern whose matched control returned
    #  MORE. That is A19's error class exactly: a number with no benchmark in
    #  it, and the one a reader would quote. The raw column stays for
    #  reference and the gated one is the difference.
    print(f"\n{'pattern':<26}{'fires':>8}{'mean log':>10}{'control':>10}"
          f"{'edge':>9}{'null sd':>9}{'z':>7}{'raw':>9}{'net edge':>10}")
    print("-" * 98)
    rows = []
    for label, fn in PATTERNS:
        fires = fire_table(P, fn, a.horizon)
        if len(fires) < MIN_FIRES:
            print(f"{label:<26}{len(fires):>8,}   insufficient")
            continue
        ctl = matched_control(pool, fires, seed=0)
        if len(ctl) < MIN_FIRES:
            print(f"{label:<26}{len(fires):>8,}   control too thin")
            continue
        mu, sd, nd = block_null(pool, fires, a.draws)
        edge = float(fires["lf"].mean() - ctl["lf"].mean())
        z = ((fires["lf"].mean() - mu) / sd) if sd and sd > 0 else float("nan")
        raw = float(fires["fwd"].mean())
        net = raw - float(ctl["fwd"].mean()) - FEE
        print(f"{label:<26}{len(fires):>8,}{fires['lf'].mean():>+10.4f}"
              f"{ctl['lf'].mean():>+10.4f}{edge:>+9.4f}{sd:>9.4f}{z:>+7.2f}"
              f"{raw:>+9.2%}{net:>+10.2%}")
        rows.append({"label": label, "fires": len(fires), "edge": edge,
                     "z": z, "net": net, "raw": raw,
                     "lf": float(fires["lf"].mean()),
                     "fwd": float(fires["fwd"].mean()), "df": fires})

    if not rows:
        print("\nnothing fired often enough to read")
        return
    R = pd.DataFrame([{k: v for k, v in r.items() if k != "df"} for r in rows])

    #  THE HALF-SPLIT, because A18 records it as the only replication test this
    #  repo trusts and a table of six patterns over 26 years reads far more
    #  solid than it is.
    print("\nhalf-split of the edge (mean log, vs a control drawn in the same "
          "half)")
    mid = P["date"].quantile(0.5)
    print(f"{'pattern':<26}{'early':>10}{'late':>10}{'both signs':>12}")
    for r in rows:
        halves = []
        for lo, hi in ((P["date"].min(), mid), (mid, P["date"].max())):
            #  A HALF IS A DATE WINDOW, NOT A ROW MEMBERSHIP TEST. A first
            #  version used `isin(Q["date"])`, which matches any firing whose
            #  DATE appears anywhere in the half for ANY name -- so a firing
            #  outside the window survives the filter whenever some other
            #  ticker traded that day, which is nearly always.
            f = r["df"][(r["df"]["date"] > lo) & (r["df"]["date"] <= hi)]
            if len(f) < MIN_FIRES:
                halves.append(float("nan"))
                continue
            pl = Pool(P[(P["date"] > lo) & (P["date"] <= hi)], a.horizon)
            c = matched_control(pl, f, seed=1)
            halves.append(float(f["lf"].mean() - c["lf"].mean())
                          if len(c) >= MIN_FIRES else float("nan"))
        same = (np.isfinite(halves[0]) and np.isfinite(halves[1])
                and np.sign(halves[0]) == np.sign(halves[1]))
        print(f"{r['label']:<26}{halves[0]:>+10.4f}{halves[1]:>+10.4f}"
              f"{str(same):>12}")
        r["both"] = bool(same)

    print("\nVERDICT")
    clears = R[np.abs(R["z"]) > 3.63]
    null_row = R[R["label"].str.contains("PREDICTED NULL")]
    print(f"  P1  {len(clears)} of {len(R)} patterns clear the "
          f"{BAR:.5f} bar (|z| > 3.63)"
          + (f": {', '.join(clears['label'])}" if len(clears) else ""))
    if len(null_row):
        nz = float(null_row["z"].iloc[0])
        fired = abs(nz) > 3.63
        print(f"  P2  the predicted null reads z {nz:+.2f} — "
              f"{'IT FIRED, so significance here is not evidence and the whole table must be read on EFFECT SIZE against cost' if fired else 'it does not fire, so the pipeline is not manufacturing its own signal'}")
    pos = R[R["net"] > 0]
    both = [r["label"] for r in rows if r.get("both")]
    print(f"  P3  {len(pos)} of {len(R)} patterns beat their matched control "
          f"after the {FEE:.2%} fee alone, before any spread"
          + (f": {', '.join(pos['label'])}" if len(pos)
             else " — PREDICTED NULL CONFIRMED"))
    print(f"      (raw returns look positive for {int((R['raw'] > FEE).sum())} "
          f"of {len(R)}, which is the market's drift and not the pattern)")
    print(f"      the edge is the SAME SIGN in both halves for "
          f"{len(both)} of {len(R)}: {', '.join(both) if both else 'none'}")
    print("\n  WHAT THIS DOES NOT SAY. Every number is in-sample (holdout "
          "spent at H16),")
    print("  the cost is A23's small-order model with no impact, suspension or")
    print("  auto-rejection term, and a discrete pattern is a COARSER version "
          "of the")
    print("  continuous features H13 already measured — CLAUDE.md §8 asks for "
          "the")
    print("  continuous form for exactly that reason.")


if __name__ == "__main__":
    main()
