#!/usr/bin/env python3
"""H49 — why does the ribbon rule only beat holding on 31.3% of names?

    python3 scripts/winrate.py --draws 20

THE CHALLENGE, WHICH IS A GOOD ONE. "I refuse to believe it only wins 30-something
percent of the time. By the central limit theorem it should be 50%, since the
price can only go up or down." That intuition is right about a coin flip and the
right response is a measurement, not an argument — so this script builds the
ladder from a fair coin to the observed number and prices each rung.

TWO DIFFERENT NUMBERS WERE BOTH CALLED "WIN" AND THAT IS PARTLY MY FAULT.
  per-TRADE win rate    the share of individual round trips that ended positive
  per-NAME win rate     the share of tickers where the rule's CAGR beat simply
                        owning the ticker over the same span
Nothing makes the second 50% even in principle: it compares a rule that is in
the market ~44% of the time and pays ~71 tolls against one that is in 100% of
the time and pays one. On an asset with positive drift that is a handicap match.
The first genuinely should be ~50% for a rule with no information, and the
control says it is.

THE ANCHOR IS A CASE WITH A KNOWN ANSWER. A26 introduced the rule that a
detector which cannot find a cycle in a pure sine wave proves nothing by finding
none. The same discipline applies to a win-rate harness: before believing 31.3%
about IDX, the machinery has to return 50% on a driftless random walk with no
cost, where 50% is the arithmetically correct answer. If it does not, the number
is a bug and the challenge is right.

THE CONTROL THAT WAS MISSING. H47 ran a matched-speed random EXIT against every
sell-off detector. H48's per-name comparison had no control at all — it was
rule-versus-hold with nothing in between. The null here shuffles the ORDER of
the ribbon's own green and red runs, which preserves exposure exactly, trade
count exactly and run-length distribution exactly, and destroys only the
alignment with price. That isolates "is the ribbon's TIMING worth anything"
from "does being out of the market half the time cost you".

PRE-REGISTERED, WRITTEN BEFORE ANY CELL WAS SCORED
--------------------------------------------------
W1  On a driftless log random walk with zero cost, both win rates come out at
    50% +/- sampling error. If not, the harness is broken.
W2  Adding realistic DRIFT alone drops the per-name win rate well below 50%,
    because time out of a rising asset is forgone return. This is the single
    largest term.
W3  Adding the toll drops it further, and the two together account for most of
    the gap between 50% and 31.3%.
W4  PREDICTED NULL — the run-shuffled ribbon wins on about the same share of
    names as the real ribbon. If the real one is materially better, its timing
    carries information after all and H48's conclusion needs qualifying; if it
    is materially worse, the ribbon is actively mistimed.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from hull_colour import FEE, colour_campaign, colour_trades      # noqa: E402
from paint_suite import tick_of                                  # noqa: E402
from selloff import hma                                          # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")
OUT = "reports"


def runs_of(mask: np.ndarray) -> Tuple[List[int], List[int], bool]:
    """Split a boolean mask into its alternating run lengths."""
    if mask.all() or not mask.any():
        return [], [], bool(mask[0])
    cut = np.flatnonzero(np.diff(mask.astype(np.int8)) != 0) + 1
    parts = np.split(mask, cut)
    first = bool(parts[0][0])
    on = [len(p) for p in parts if p[0]]
    off = [len(p) for p in parts if not p[0]]
    return on, off, first


def shuffle_runs(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reorder the green and red runs, keeping every one of them.

    Exposure, trade count and the run-length distribution are all preserved
    EXACTLY; only the alignment between the ribbon and the price path is
    destroyed. A null that changed how often or how long the rule was invested
    would confound timing with exposure, which is the whole thing being
    separated here.
    """
    on, off, first = runs_of(mask)
    if not on or not off:
        return mask.copy()
    rng.shuffle(on)
    rng.shuffle(off)
    out = np.empty(len(mask), bool)
    k = 0
    a, b = (on, off) if first else (off, on)
    va, vb = (True, False) if first else (False, True)
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out[k:k + a[i]] = va
            k += a[i]
        if i < len(b):
            out[k:k + b[i]] = vb
            k += b[i]
    return out[:len(mask)]


def green_of(p: np.ndarray, n: int = 55) -> np.ndarray:
    h = hma(p, n)
    g = np.zeros(len(p), bool)
    g[2:] = h[2:] > h[:-2]
    return g


# ============================================ RUNG 0 — the known-answer case ==
def synthetic(n_names: int, n_bars: int, mu: float, sigma: float, cost: float,
              expo: float, mean_run: float, seed: int) -> dict:
    """A market with a KNOWN answer, so the harness can be checked against it.

    `mu` is the mean LOG return per bar. At mu=0 the median terminal price is
    the starting price, so a coin flip on "did the trade make money" is exactly
    50% and a rule with random timing must beat buy-and-hold exactly 50% of the
    time. Anything else is a defect in this file, not a fact about Indonesia.
    """
    rng = np.random.default_rng(seed)
    per, tr = [], []
    for _ in range(n_names):
        p = np.exp(np.cumsum(rng.normal(mu, sigma, n_bars))) * 1000.0
        #  A random mask with the requested exposure and realistic run lengths,
        #  built from geometric runs so it is not one contiguous block.
        g = np.zeros(n_bars, bool)
        i = 0
        while i < n_bars:
            on = max(1, int(rng.geometric(1.0 / mean_run)))
            off = max(1, int(rng.geometric(expo / (mean_run * (1 - expo)))))
            g[i:i + on] = True
            i += on + off
        lg, hl, ntr, inb = colour_campaign(p, g, cost)
        if ntr:
            per.append(lg > hl)
            tr.extend(r for _, _, r in colour_trades(p, g, cost))
    return {"per_name": float(np.mean(per)) if per else float("nan"),
            "per_trade": float(np.mean(np.array(tr) > 0)) if tr else float("nan"),
            "mean_trade": float(np.mean(tr)) if tr else float("nan"),
            "med_trade": float(np.median(tr)) if tr else float("nan"),
            "names": len(per), "trades": len(tr)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--min-bars", type=int, default=400)
    a = ap.parse_args()

    #  ---------------------------------------------------------------- rung 0
    print("W1 — THE HARNESS ON A CASE WITH A KNOWN ANSWER.")
    print("A driftless log random walk, zero cost, random timing at 44% "
          "exposure.\nBoth win rates must come out at 50%. They are the "
          "arithmetically correct\nanswer, and a harness that cannot reproduce "
          "them cannot be believed about IDX.\n")
    print(f"{'market':<46}{'per-NAME win':>14}{'per-TRADE win':>15}"
          f"{'mean trade':>12}{'median':>10}")
    rungs = [
        ("driftless, no cost  <- the coin flip", 0.0, 0.0),
        ("driftless, 1.44% toll", 0.0, 0.0144),
    ]
    #  The board's own drift and volatility, so the ladder ends where IDX is.
    for lab, mu, cost in rungs:
        s = synthetic(300, 3000, mu, 0.030, cost, 0.44, 25.0, seed=11)
        print(f"{lab:<46}{s['per_name']:>14.1%}{s['per_trade']:>15.1%}"
              f"{s['mean_trade']:>+12.2%}{s['med_trade']:>+10.2%}")

    #  ---------------------------------------------------------------- panel
    P = pd.read_parquet(PANEL)
    P = P[P["adj_close"] > 0].sort_values(["ticker", "date"])
    names = []
    for tk, g in P.groupby("ticker", sort=False):
        if len(g) < a.min_bars:
            continue
        p = g["adj_close"].to_numpy(float)
        med = float(np.nanmedian(g["close"].to_numpy(float)))
        if not np.isfinite(med) or med <= 0:
            continue
        tv = float(np.nanmedian(np.exp(g["log_turnover"].to_numpy(float))))
        names.append((tk, p, FEE + tick_of(med) / med, tv, green_of(p)))

    #  Board drift, measured rather than assumed, so rung 2 is IDX's own.
    dr = np.array([np.log(p[-1] / p[0]) / len(p) for _, p, _, _, _ in names])
    sg = np.array([np.std(np.diff(np.log(p))) for _, p, _, _, _ in names])
    mu, sig = float(np.median(dr)), float(np.median(sg))
    print(f"\n  board's own median log drift {mu * 252:+.2%}/yr, "
          f"daily sd {sig:.2%} — feeding those in:")
    for lab, cost in (("IDX drift and vol, no cost", 0.0),
                      ("IDX drift and vol, 1.44% toll  <- full ladder", 0.0144)):
        s = synthetic(300, 3000, mu, sig, cost, 0.44, 25.0, seed=12)
        print(f"{lab:<46}{s['per_name']:>14.1%}{s['per_trade']:>15.1%}"
              f"{s['mean_trade']:>+12.2%}{s['med_trade']:>+10.2%}")

    #  ------------------------------------------------- the real board + null
    print(f"\nW4 — THE REAL RIBBON AGAINST ITS OWN RUN-SHUFFLED NULL, "
          f"{a.draws} draws.")
    print("The null keeps every green run and every red run and only reorders "
          "them, so\nexposure, trade count and run lengths are identical and "
          "ONLY the alignment\nwith price is destroyed.\n")
    rng = np.random.default_rng(4949)
    rows = []
    for tk, p, cost, tv, g in names:
        lg, hl, ntr, inb = colour_campaign(p, g, cost)
        if not ntr:
            continue
        rets = [r for _, _, r in colour_trades(p, g, cost)]
        nul, ntw = [], []
        for _ in range(a.draws):
            gs = shuffle_runs(g, rng)
            l2, _, n2, _ = colour_campaign(p, gs, cost)
            if n2:
                nul.append(l2 > hl)
                ntw.extend(r for _, _, r in colour_trades(p, gs, cost))
        rows.append({"ticker": tk, "tv": tv, "bars": len(p), "cost": cost,
                     "beat": bool(lg > hl), "trades": ntr,
                     "expo": inb / len(p),
                     "win": float(np.mean(np.array(rets) > 0)),
                     "mean_tr": float(np.mean(rets)),
                     "med_tr": float(np.median(rets)),
                     "null_beat": float(np.mean(nul)) if nul else np.nan,
                     "null_win": (float(np.mean(np.array(ntw) > 0))
                                  if ntw else np.nan),
                     "null_mean": float(np.mean(ntw)) if ntw else np.nan})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "winrate.csv"), index=False)

    def blk(D, lab):
        if not len(D):
            return
        print(f"{lab:<30}{len(D):>6}{D['expo'].mean():>8.0%}"
              f"{D['beat'].mean():>14.1%}{D['null_beat'].mean():>13.1%}"
              f"{D['win'].mean():>13.1%}{D['null_win'].mean():>12.1%}"
              f"{D['mean_tr'].mean():>+11.2%}{D['med_tr'].mean():>+10.2%}")

    print(f"{'universe':<30}{'names':>6}{'expo':>8}{'BEATS HOLD':>14}"
          f"{'null':>13}{'trade win':>13}{'null':>12}{'mean tr':>11}"
          f"{'med tr':>10}")
    blk(R, "every name")
    for lo, hi, lab in ((1e9, 1e99, "liquid  >= Rp1bn/day"),
                        (1e8, 1e9, "middle  Rp0.1-1bn/day"),
                        (0, 1e8, "thin    < Rp0.1bn/day")):
        blk(R[(R["tv"] >= lo) & (R["tv"] < hi)], lab)

    d = R["beat"].mean() - R["null_beat"].mean()
    sd = R["null_beat"].std(ddof=1) / np.sqrt(len(R))
    print(f"\n  real minus run-shuffled null, per-name beat rate: "
          f"{d:+.1%}  (se {sd:.1%})")
    print(f"  trades per name {R['trades'].mean():.0f}, "
          f"mean toll {R['cost'].mean():.2%}, "
          f"so the rule pays {R['trades'].mean() * R['cost'].mean():.0%} of "
          f"its capital in tolls over the span")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
