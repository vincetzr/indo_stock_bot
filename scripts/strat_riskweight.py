#!/usr/bin/env python3
"""H54 / WEIGHTING FAMILY — same names every bar, only the WEIGHTS change.

THE QUESTION THIS ANSWERS
    Owning every eligible name and rebalancing quarterly returns ~+1.15%/yr.
    Owning the same names and never touching them returns ~+10.05%/yr. That
    ~9-point gap is the price of turnover plus the price of systematically
    selling whatever is compounding. This family holds the WHOLE eligible
    universe on every bar -- NO stock selection whatsoever -- and varies only
    the weighting policy, so any difference between variants is attributable
    to the weighting policy alone.

THE SPECTRUM
    target  : where a full rebalance would put the money
              equal      1/N
              invvol     1/vol60          (naive risk parity)
              invvar     1/vol60^2        (variance parity, more extreme)
              mom        rank of mom12_1  (momentum-weighted)
    policy  : how far back toward that target you drag the drifted book
              blend=1.0            full rebalance every bar
              blend=lam            w <- (1-lam)*drifted + lam*target
              band=B               only names outside [tgt/B, tgt*B] are reset
              cap=C                only TRIM names above C*tgt; never top up
              blend=0 / band=inf   pure drift, and no new entrants at all

MECHANICS, AND THE TWO PLACES THIS COULD HAVE CHEATED
    * State is carried across bars in a closure (previous weights, previous
      prices). That is PAST information only -- `select` still sees one bar --
      so no look-ahead is introduced. A fresh selector is built for every
      evaluate() call so one variant cannot inherit another's book.
    * `select` NEVER returns a ticker that is absent from the bar it was
      handed. It is tempting to keep holding a name that has fallen out of the
      eligible set -- that is what a true never-touch basket does -- but the
      harness prices an off-frame ticker with PX.at(), which returns NaN for a
      name that did not print, and the harness then silently REDISTRIBUTES that
      weight across the survivors. That would credit a vanished name with the
      basket's average return. So a name leaving the eligible set is SOLD at
      that bar's price and the proceeds are spread over the rest, and the
      turnover column charges for it. This is the one respect in which even the
      "pure drift" arm is not literally never-touch, and it is measured below
      (`churn` = share of book forced out per bar).

EVERYTHING HERE IS IN-SAMPLE. The 24-month holdout was spent long ago.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from bhbench import Bench, load, report  # noqa: E402


# --------------------------------------------------------------- targets
def target_weights(day: pd.DataFrame, scheme: str) -> np.ndarray:
    n = len(day)
    if scheme == "equal":
        w = np.ones(n)
    elif scheme in ("invvol", "invvar"):
        v = day["vol60"].to_numpy(float).copy()
        med = np.nanmedian(v[np.isfinite(v) & (v > 0)])
        v[~np.isfinite(v) | (v <= 0)] = med
        # floor the denominator: a near-zero vol60 is a suspended name, not a
        # riskless one, and 1/eps would hand it the whole book.
        v = np.maximum(v, 0.25 * med)
        w = 1.0 / (v ** (1.0 if scheme == "invvol" else 2.0))
    elif scheme == "mom":
        m = day["mom12_1"].to_numpy(float)
        r = pd.Series(m).rank(pct=True, na_option="bottom").to_numpy()
        w = r + 0.05                      # strictly positive, long-only
    else:
        raise ValueError(scheme)
    s = w.sum()
    return w / s if s > 0 else np.ones(n) / n


class Weighter:
    """One weighting policy. Stateful across bars; state is past-only.

    `sticky` decides what happens to a name the strategy already owns that has
    fallen OUT of the eligible set. False = sold at this bar (the only thing
    possible from the handed frame). True = kept, priced from `pmap`, which is
    this bar's own closing prices for the WHOLE board -- contemporaneous, not
    future, so it is not look-ahead, but it does reach outside the frame the
    harness hands `select`, and both versions are therefore reported. A name
    that did not PRINT on this bar is dropped either way: the harness prices an
    absent print as NaN and silently redistributes the weight, which would
    credit a vanished name with the basket's average return.
    """

    def __init__(self, scheme: str = "equal", blend: float = 1.0,
                 band: float = 0.0, cap: float = 0.0, entrants: bool = True,
                 sticky: bool = False, pmap: Dict | None = None):
        self.scheme, self.blend = scheme, blend
        self.band, self.cap, self.entrants = band, cap, entrants
        self.sticky, self.pmap = sticky, (pmap or {})
        self.w: Dict[str, float] = {}
        self.px: Dict[str, float] = {}
        self.churn: List[float] = []
        self.hhi: List[float] = []
        self.nheld: List[int] = []

    def __call__(self, day: pd.DataFrame):
        tks = day["ticker"].to_numpy()
        px = dict(zip(tks, day["adj_close"].to_numpy(float)))
        tgt = dict(zip(tks, target_weights(day, self.scheme)))
        board = self.pmap.get(day["date"].iloc[0], {}) if self.sticky else {}

        # ---- drift the previous book forward to this bar's prices
        drift: Dict[str, float] = {}
        lost = 0.0
        for t, w in self.w.items():
            p0 = self.px.get(t)
            p1 = px.get(t, board.get(t))
            if p0 and p1 and np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                drift[t] = w * (p1 / p0)
                px.setdefault(t, p1)       # carry the price for the next bar
            else:
                lost += w                  # forced out: sold at this bar
        tot = sum(drift.values())
        self.churn.append(lost / (lost + tot) if (lost + tot) > 0 else 0.0)
        if tot <= 0:                       # first bar, or the book emptied
            new = dict(tgt)
        else:
            drift = {t: v / tot for t, v in drift.items()}
            new = {}
            #  A held name that is no longer eligible has no target, so it can
            #  only ever drift -- there is nothing to rebalance it to.
            for t, d in drift.items():
                if t not in tgt:
                    new[t] = d
            for t, g in tgt.items():
                d = drift.get(t, 0.0)
                if d == 0.0:               # a NEW name, not yet owned
                    if not self.entrants:
                        continue           # closed book: no new names ever
                    v = g                  # bought at its target weight
                elif self.blend > 0:
                    v = (1.0 - self.blend) * d + self.blend * g
                elif self.band > 0:
                    v = g if (d > self.band * g or d < g / self.band) else d
                elif self.cap > 0:
                    v = min(d, self.cap * g)
                else:
                    v = d                  # pure drift
                if v > 0:
                    new[t] = v
            if not new:
                new = dict(tgt)
        s = sum(new.values())
        new = {t: v / s for t, v in new.items()}
        self.w, self.px = new, px
        self.nheld.append(len(new))
        self.hhi.append(float(np.sum(np.square(list(new.values())))))
        return list(new.items())


VARIANTS: Sequence[Tuple[str, dict]] = [
    # ---- the spectrum, equal target: full rebalance -> pure drift
    ("equal, FULL rebalance",        dict(scheme="equal", blend=1.0)),
    ("equal, blend 0.50",            dict(scheme="equal", blend=0.50)),
    ("equal, blend 0.25",            dict(scheme="equal", blend=0.25)),
    ("equal, band 2x",               dict(scheme="equal", blend=0.0, band=2.0)),
    ("equal, band 3x",               dict(scheme="equal", blend=0.0, band=3.0)),
    ("equal, cap 3x (trim only)",    dict(scheme="equal", blend=0.0, cap=3.0)),
    ("equal, cap 5x (trim only)",    dict(scheme="equal", blend=0.0, cap=5.0)),
    ("equal, PURE DRIFT (open)",     dict(scheme="equal", blend=0.0)),
    ("equal, PURE DRIFT (closed)",   dict(scheme="equal", blend=0.0,
                                          entrants=False)),
    # ---- risk-based targets, full rebalance
    ("inv-vol, FULL rebalance",      dict(scheme="invvol", blend=1.0)),
    ("inv-var, FULL rebalance",      dict(scheme="invvar", blend=1.0)),
    ("momentum-wt, FULL rebalance",  dict(scheme="mom", blend=1.0)),
    # ---- risk-based targets, low-touch
    ("inv-vol, band 2x",             dict(scheme="invvol", blend=0.0,
                                          band=2.0)),
    ("inv-vol, cap 3x",              dict(scheme="invvol", blend=0.0,
                                          cap=3.0)),
    ("inv-vol, PURE DRIFT (open)",   dict(scheme="invvol", blend=0.0)),
    ("momentum-wt, cap 3x",          dict(scheme="mom", blend=0.0, cap=3.0)),
    # ---- STICKY: an owned name that leaves the eligible set is KEPT, which is
    #      what the never-touch benchmark does. Reaches outside the handed
    #      frame for a contemporaneous price; flagged everywhere.
    ("STICKY equal, FULL rebal",     dict(scheme="equal", blend=1.0,
                                          sticky=True)),
    ("STICKY equal, cap 3x",         dict(scheme="equal", blend=0.0, cap=3.0,
                                          sticky=True)),
    ("STICKY equal, PURE DRIFT",     dict(scheme="equal", blend=0.0,
                                          sticky=True)),
    ("STICKY equal, DRIFT closed",   dict(scheme="equal", blend=0.0,
                                          entrants=False, sticky=True)),
    ("STICKY inv-vol, PURE DRIFT",   dict(scheme="invvol", blend=0.0,
                                          sticky=True)),
    ("STICKY inv-vol, cap 3x",       dict(scheme="invvol", blend=0.0, cap=3.0,
                                          sticky=True)),
    ("STICKY mom-wt, PURE DRIFT",    dict(scheme="mom", blend=0.0,
                                          sticky=True)),
]


def price_by_date(P: pd.DataFrame, freq: int) -> Dict:
    """Closing price of the whole board on each rebalance bar. Same marks the
    harness walks, so no date outside a decision bar is ever consulted."""
    marks = set(pd.Timestamp(d) for d in np.sort(P["date"].unique())[::freq])
    out: Dict = {}
    for d, g in P[P["date"].isin(marks)].groupby("date"):
        out[pd.Timestamp(d)] = dict(zip(g["ticker"].to_numpy(),
                                        g["adj_close"].to_numpy(float)))
    return out


def flat_bars(P: pd.DataFrame, freq: int) -> Tuple[int, int, List[str]]:
    """Marks the harness will NOT trade because the eligible universe is below
    MIN_UNIV=40. `select` is never even called on those bars, so the strategy
    sits in cash while every buy-and-hold benchmark stays fully invested. No
    weighting policy -- no strategy of any kind -- can avoid this, so it has to
    be measured rather than absorbed into a verdict.
    """
    marks = pd.DatetimeIndex(np.sort(P["date"].unique())[::freq])
    c = (P[P["date"].isin(marks) & P["elig"]].groupby("date").size()
         .reindex(marks).fillna(0))
    ok = c[c >= 40]
    if not len(ok):
        return 0, 0, []
    after = c[c.index >= ok.index[0]]
    flat = after[after < 40]
    return len(after), len(flat), [str(d.date()) for d in flat.index]


def main() -> None:
    freq = int(sys.argv[1]) if len(sys.argv) > 1 else 63
    min_tv = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
    P = load(min_tv=min_tv)
    n_bar, n_flat, when = flat_bars(P, freq)
    print(f"# freq={freq} min_tv={min_tv:.0e}   the harness FORCES CASH on "
          f"{n_flat} of {n_bar} bars (universe < 40): {when}\n")
    B = Bench(P)
    PBD = price_by_date(P, freq)
    rows = []
    for label, kw in VARIANTS:
        W = Weighter(pmap=PBD, **kw)
        v = B.evaluate(W, label=label, freq=freq)
        print(report(v))
        if v.get("ok"):
            print(f"  [held {np.mean(W.nheld):.0f} names, forced churn "
                  f"{np.mean(W.churn):.2%}/bar, HHI {np.mean(W.hhi):.4f}, "
                  f"gross {v['gross']:+.2%}]")
            rows.append({"label": label, "freq": freq, **{
                k: v[k] for k in ("cagr", "gross", "years", "basket",
                                  "turnover", "cost_yr", "early", "late",
                                  "bh_index", "bh_universe", "bh_picks",
                                  "random", "random_sd", "beats_index",
                                  "beats_universe", "beats_picks",
                                  "beats_random", "both_halves_index",
                                  "both_halves_universe", "both_halves_picks",
                                  "PASS")},
                "held": float(np.mean(W.nheld)),
                "churn": float(np.mean(W.churn)),
                "hhi": float(np.mean(W.hhi))})
        print()
    if rows:
        df = pd.DataFrame(rows)
        out = f"reports/strat_riskweight_f{freq}_tv{min_tv:.0e}.csv"
        df.to_csv(out, index=False)
        print(df[["label", "cagr", "gross", "turnover", "cost_yr", "hhi",
                  "early", "late", "PASS"]].to_string(index=False))
        print(f"\nwrote {out}")
        print(f"PASS count: {int(df['PASS'].sum())} of {len(df)}")


if __name__ == "__main__":
    main()
