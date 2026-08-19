#!/usr/bin/env python3
"""You cannot miss a big move with a band rule. You can only pay a toll on it.

The objection answered in Result 108 - "remove the best 5 and it loses" - deletes
those trades outright. That is the wrong model of failure for THIS rule, and the
reason is structural rather than statistical.

THE LAW
-------
The rule flips green when price rises ``b`` off the running low and flips red
when it falls ``b`` off the running high. So for a leg running from a low L to a
high H, with M = H/L - 1:

    you cannot buy below   L * (1 + b)      the flip is what buys you in
    you cannot sell above  H * (1 - b)      the flip is what sells you out

    best possible capture  =  (1 + M) * (1 - b) / (1 + b)

One consequence is exact. The other two, as first written here, were wrong, and
an adversarial re-derivation caught them. Both corrections are kept in place
rather than quietly patched, because the difference between them matters:

    1. EXACT. A move larger than the band CANNOT be missed. While the state is
       red the running minimum is non-increasing, so the trigger price is
       non-increasing, so the first close that reaches it fires the flip. Proven
       in one line, verified with zero violations on 1.3m red bars across 719
       names, and it holds through gaps - a one-bar 100 -> 400 jump flips the
       state on the gap bar itself.

    2. WRONG AS FIRST STATED. This said "the cost is a FIXED TOLL of (1+b)/(1-b)
       on price". It is not fixed. It is a floor on the SIGNAL price only: the
       flip fires on the first close at or ABOVE min*(1+b), overshooting the
       trigger by however far that bar travelled, and the fill is the bar AFTER
       that - which can come back under it. So realised tolls straddle the
       figure rather than respecting it, with the typical one well above.
       Measured on 2,571 real round trips across 46 IDX large caps:

           entry overshoot   median  9.4%   (70% of entries exceed 8%)
           exit shortfall    median  8.8%   (63% exceed 8%)
           realised toll     median 20.6% of price, p90 31.7%

       The same passage also said the toll "eats 8% of the log move" on a +292%
       leg. The correct figure from this file's own ceiling_fraction is 11.7%.

    3. WRONG AS FIRST STATED. This said M* = 2b/(1-b) = 17.4% is the break-even
       and that every smaller leg is a "guaranteed loss". 17.4% is the
       ALGEBRAIC FLOOR on the break-even, reachable only with perfect fills. The
       realised break-even, taken from where round-trip P/L actually crosses
       zero, is ~21% for these names:

           move spanned    trips   median P/L   win rate
           10-17%            813        -6.5%          6%
           17-20%            171        -2.5%         26%
           20-25%            291        +0.0%         50%    <- crosses here
           25-30%            195        +3.3%         68%
           >40%              329       +25.4%         98%

       And "guaranteed" is wrong regardless: fills gap, so a sub-break-even leg
       can still pay. The direction of the claim survives; the certainty does not.

AND THE ONE THAT MATTERS MOST FOR READING ANY OF THIS
-----------------------------------------------------
A "leg" here is defined by a zigzag at the SAME band as the rule, which makes it
by construction a move containing no b-sized retracement - i.e. exactly one round
trip. That is near-circular, and it is NOT what a person means by a big move.

Measured against real bull moves (a 30% zigzag, >= +50%), 203 of them:

    a real bull move contains a median of 5 separate 8% legs, and pays 5 tolls
    captured: 55% of the move's log return, where the one-leg formula implies 77%

So the honest headline is not "the rule keeps 78-92% of a big move". It is that
the rule cannot MISS one, and keeps a bit over half of it.

And the honest robustness test is still not "delete the best five trades" - it is
"arrive late at them", which is what a real trader actually does. Both are
measured here, along with every corrected figure quoted above.

WHAT IS MEASURED
----------------
For every hindsight leg on every big cap, the fraction of that leg's log return
the causal rule actually collected. Capture is computed from the rule's realised
long-only exposure over the leg's own window, so whipsaw inside a leg is charged
against it rather than hidden.

    python3 scripts/capture_toll.py --band 0.08
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV       # noqa: E402
from leg_signals import market_caps            # noqa: E402
from paint_daily import unadjusted_daily       # noqa: E402
from paint_live import band_state              # noqa: E402
from swing_accuracy import legs, zigzag        # noqa: E402


# --------------------------------------------------------------------------- #
# the algebra
# --------------------------------------------------------------------------- #
def capture_ceiling(move: float, band: float) -> float:
    """Best possible multiple on a leg of size ``move``, as a growth factor."""
    return (1.0 + move) * (1.0 - band) / (1.0 + band)


def breakeven_move(band: float) -> float:
    """Leg size at which the toll exactly eats the move: M* = 2b / (1 - b)."""
    return 2.0 * band / (1.0 - band)


def ceiling_fraction(move: float, band: float) -> float:
    """Round-trip ceiling as a share of the leg's LOG return.

    Both tolls: you enter no lower than L(1+b) and exit no higher than H(1-b).
    Compare this only against a return measured to the ACTUAL EXIT.
    """
    if move <= 0:
        return np.nan
    cap = capture_ceiling(move, band)
    if cap <= 0:
        return -np.inf
    return float(np.log(cap) / np.log(1.0 + move))


def ceiling_entry_only(move: float, band: float) -> float:
    """Ceiling for a return measured only as far as the PEAK.

    The exit flip happens by construction after the peak, so a window that stops
    at the peak has not yet paid the exit toll. Measuring to the peak and then
    comparing against the round-trip ceiling understates the ceiling and makes
    realised capture look impossibly good; this is the bound that window earns.
    """
    if move <= 0:
        return np.nan
    return float(np.log((1.0 + move) / (1.0 + band)) / np.log(1.0 + move))


def round_trip_from(px: np.ndarray, state: np.ndarray, a: int
                    ) -> Optional[Tuple[int, int]]:
    """First (entry, exit) pair the rule takes at or after bar ``a``.

    Entry is the bar after the flip to green - you cannot buy on the close that
    generated the signal. Exit is the bar after the flip back to red. The exit
    routinely lands beyond the leg, which is the point: that is where the second
    toll is actually paid.
    """
    n = len(px)
    i = a
    while i < n and not state[i]:
        i += 1
    if i >= n - 1:
        return None
    entry = i + 1
    j = entry
    while j < n and state[j]:
        j += 1
    if j >= n - 1:
        return None
    return entry, j + 1


def realised_tolls(px: np.ndarray, band: float) -> List[Tuple[float, ...]]:
    """The toll the rule ACTUALLY paid on each round trip, versus the algebra.

    The entry is the first close at or above min*(1+b) - which overshoots the
    trigger by however far that bar moved - and the fill is the bar after. So the
    realised toll is (1 + overshoot) / (1 - shortfall), and (1+b)/(1-b) is only
    its floor. Returns one row per completed round trip:

        (entry overshoot, exit shortfall, round-trip P/L, move the trip spanned)
    """
    st, _ = band_state(px, band)
    n = len(px)
    out: List[Tuple[float, ...]] = []
    i = 0
    while i < n - 1:
        while i < n and not st[i]:
            i += 1
        if i >= n - 1:
            break
        j = i
        while j > 0 and st[j - 1] == 0:
            j -= 1
        lo = float(px[j:i + 1].min())
        entry = float(px[i + 1])
        k = i + 1
        while k < n and st[k]:
            k += 1
        if k >= n - 1:
            break
        hi = float(px[i + 1:k + 1].max())
        exit_ = float(px[k + 1])
        out.append((entry / lo - 1.0, 1.0 - exit_ / hi,
                    exit_ / entry - 1.0, hi / lo - 1.0))
        i = k + 1
    return out


def bull_moves(px: np.ndarray, band: float, big_band: float = 0.30,
               floor: float = 0.50) -> List[Tuple[float, ...]]:
    """What a PERSON calls a big move, and how many band-legs hide inside it.

    A leg measured at the rule's own band contains no b-sized retracement by
    construction, so it is exactly one round trip - which makes any capture
    figure computed on it close to circular. A real bull run is found with a
    much wider zigzag and normally contains several band-legs, each charging its
    own toll. Returns (move, legs inside, captured share, formula's share).
    """
    st, _ = band_state(px, band)
    out: List[Tuple[float, ...]] = []
    for a, c, r in legs(px, zigzag(px, big_band)):
        if r < floor or c <= a:
            continue
        sub = px[a:c + 1]
        inside = sum(1 for _x, _y, rr in legs(sub, zigzag(sub, band)) if rr > 0)
        lr = np.diff(np.log(sub))
        held = st[a:c].astype(bool)
        tot = float(np.log(px[c] / px[a]))
        got = float(lr[held].sum())
        out.append((r, float(inside), got / tot if tot > 0 else np.nan,
                    ceiling_fraction(r, band)))
    return out


# --------------------------------------------------------------------------- #
# what the rule actually collected
# --------------------------------------------------------------------------- #
def exposure_log_return(px: np.ndarray, state: np.ndarray,
                        i: int, j: int, delay: int = 0) -> float:
    """Log return the rule earned between bars i and j, long only.

    The colour known at the close of bar t-1 governs bar t, so nothing here uses
    a price before it printed. ``delay`` pushes the decision further back, which
    is what arriving late means.
    """
    lag = 1 + delay
    tot = 0.0
    for t in range(i + 1, j + 1):
        s = state[t - lag] if t - lag >= 0 else 0
        if s:
            tot += np.log(px[t] / px[t - 1])
    return float(tot)


def leg_table(px: np.ndarray, band: float, delay: int = 0) -> pd.DataFrame:
    """Every hindsight up-leg, and how much of it the causal rule collected."""
    piv = zigzag(px, band)
    state, _ = band_state(px, band)
    rows = []
    for a, b_, r in legs(px, piv):
        if r <= 0:                       # up legs only; the rule is long-only
            continue
        leg_lg = float(np.log(px[b_] / px[a]))
        if leg_lg <= 0:
            continue
        got = exposure_log_return(px, state, a, b_, delay)
        # Two different questions, and conflating them overstates the case:
        #   by_peak   did the rule turn green at the latest ON the peak bar?
        #             This is the arithmetic guarantee - the move itself trips it.
        #   before    did it turn green with any of the leg left to collect?
        #             This is what participation means, and it can fail on a leg
        #             that happens entirely in its final bar.
        by_peak = bool(state[b_])
        before = bool(state[max(a - 1, 0):b_].max()) if b_ > a else False
        rt = round_trip_from(px, state, a)
        rt_log = float(np.log(px[rt[1]] / px[rt[0]])) if rt else np.nan
        rows.append({
            "start": a, "end": b_, "bars": b_ - a,
            "move": r,
            "leg_log": leg_lg,
            "got_log": got,
            "capture": got / leg_lg,
            "ceiling_entry": ceiling_entry_only(r, band),
            # the round trip is NOT confined to this leg - its exit routinely
            # lands past the peak, sometimes inside a later move - so it is
            # reported as an absolute return, never as a fraction of this leg
            "rt_return": float(np.exp(rt_log) - 1.0) if rt else np.nan,
            "ceiling": ceiling_fraction(r, band),
            "green_by_peak": by_peak,
            "green_before_peak": before,
            # already long when the leg began, so no entry toll was paid on it -
            # this is why realised capture can exceed the fresh-entry ceiling
            "held_in": bool(state[a]) if a < len(state) else False,
            "collected_nothing": got <= 0.0,
        })
    return pd.DataFrame(rows)


def bucket(df: pd.DataFrame, band: float) -> pd.DataFrame:
    """Capture by leg size, with the break-even line marked."""
    m = breakeven_move(band)
    edges = [0, m, 0.25, 0.50, 1.00, 2.00, np.inf]
    names = [f"< {m:.0%} (break-even)", f"{m:.0%}-25%", "25-50%",
             "50-100%", "100-200%", "> 200%"]
    out = []
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        sel = df[(df["move"] >= lo) & (df["move"] < hi)]
        if not len(sel):
            continue
        out.append({
            "bucket": nm, "legs": len(sel),
            "median_move": sel["move"].median(),
            "median_capture": sel["capture"].median(),
            "median_ceiling": sel["ceiling_entry"].median(),
            "median_rt_return": sel["rt_return"].median(),
            "rt_win_rate": float((sel["rt_return"] > 0).mean()),
            "late_at_peak": int((~sel["green_before_peak"]).sum()),
            "missed_entirely": int((~sel["green_by_peak"]).sum()),
            "held_in": int(sel["held_in"].sum()),
            "collected_nothing": int(sel["collected_nothing"].sum()),
        })
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=float, default=0.08)
    ap.add_argument("--min-mcap", type=float, default=1e13)
    ap.add_argument("--delays", type=int, nargs="+", default=[0, 1, 2, 3, 5, 10])
    ap.add_argument("--top", type=int, default=5,
                    help="how many of each name's biggest legs to report separately")
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)
    b = args.band

    print(f"{'=' * 78}\n THE ARITHMETIC, BEFORE ANY DATA — band {b:.0%}\n{'=' * 78}")
    print(f" toll on price, round trip:  (1+b)/(1-b) = {(1 + b) / (1 - b):.4f}"
          f"  ->  {(1 + b) / (1 - b) - 1:.2%} of price")
    print(f" break-even leg size:        M* = 2b/(1-b) = {breakeven_move(b):.2%}")
    print(f"\n {'leg size':>10}{'best case':>12}{'kept':>9}   what the toll costs")
    for M in (0.10, 0.174, 0.25, 0.50, 1.0, 2.0, 2.92, 5.0):
        cap = capture_ceiling(M, b)
        frac = ceiling_fraction(M, b)
        print(f" {M:>10.0%}{cap - 1:>+12.1%}{frac:>9.0%}   "
              f"{'below the algebraic floor' if cap <= 1 else 'floor only: real fills overshoot it'}")

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    mc = market_caps()
    big = sorted(mc[mc >= args.min_mcap].index)

    allrows, tops, per_name = [], [], []
    keep: Dict[str, np.ndarray] = {}
    for t in big:
        s = unadjusted_daily(loader, t)
        if s is None or len(s) < 500:
            continue
        px = s.to_numpy(float)
        keep[t] = px
        df = leg_table(px, b)
        if not len(df):
            continue
        df["ticker"] = t
        allrows.append(df)
        big_legs = df.nlargest(min(args.top, len(df)), "move")
        big_legs = big_legs.assign(ticker=t)
        tops.append(big_legs)
        per_name.append({"ticker": t, "legs": len(df),
                         "median_capture": df["capture"].median()})
    A = pd.concat(allrows, ignore_index=True)
    T = pd.concat(tops, ignore_index=True)
    A.to_csv("reports/capture_toll_legs.csv", index=False)

    print(f"\n{'=' * 78}\n WHAT ACTUALLY HAPPENED — {len(per_name)} big caps, "
          f"{len(A):,} up legs\n{'=' * 78}")
    B = bucket(A, b)
    print(f" {'leg size':>22}{'legs':>7}{'move':>7}{'captured':>10}{'ceiling':>9}"
          f"{'trade P/L':>11}{'wins':>7}{'late':>6}{'missed':>7}")
    for _, r in B.iterrows():
        print(f" {r['bucket']:>22}{r['legs']:>7,}{r['median_move']:>7.0%}"
              f"{r['median_capture']:>10.0%}{r['median_ceiling']:>9.0%}"
              f"{r['median_rt_return']:>+11.1%}{r['rt_win_rate']:>7.0%}"
              f"{r['late_at_peak']:>6}{r['missed_entirely']:>7}")
    print("\n  'late'   = turned green only ON the peak bar, so there was "
          "nothing left to collect")
    print("  'missed' = never turned green at all. This column is the "
          "arithmetic claim.")
    print("\n  'captured' runs to the leg's peak and so charges the ENTRY toll "
          "only; 'ceiling' is\n  the matching entry-only bound. The exit flip "
          "lands after the peak by construction,\n  which is where the second "
          "toll gets paid and why the break-even is 2b/(1-b).")
    print("  'trade P/L' is the real round trip that began inside the leg, in "
          "money. It is not\n  shown as a fraction of the leg: its exit is "
          "often past the leg's end, so the leg\n  does not bound it.")
    tradable = A[A["move"] > b]
    over_c = int((tradable["capture"] > tradable["ceiling_entry"] + 1e-9).sum())
    print(f"\n  legs above the band that beat their own SIGNAL-level ceiling: "
          f"{over_c} of {len(tradable):,}.")
    print("  This bounds signal prices, not fills. It is near-zero rather than "
          "identically zero\n  because band_state tests `p/ext-1 >= band-1e-12` "
          "while zigzag tests `p >= lo*(1+band)`\n  with no tolerance, and on "
          "tick-quantised IDX prices an exactly-b move is common.\n  The "
          "realised-fill ceiling below is breached far more often, and that is "
          "the real point.")

    # --- the question as asked: can a big move be missed? -------------------
    over = A[A["move"] > b]
    missed = int((~over["green_by_peak"]).sum())
    late = int((~over["green_before_peak"]).sum())
    print(f"\n{'=' * 78}\n CAN A MOVE BIGGER THAN THE BAND BE MISSED?\n{'=' * 78}")
    print(f" up legs larger than the {b:.0%} band: {len(over):,}")
    print(f" never turned green, even by the peak:  {missed}"
          f"   <- the move itself trips the flip, so this is 0 by construction")
    print(f" turned green only ON the peak bar:     {late}"
          f"   ({late / max(len(over), 1):.1%}) - real, and it is why fast "
          f"legs pay")
    print(f" collected nothing net:                 "
          f"{int(over['collected_nothing'].sum())}")
    med_bars = over.loc[~over["green_before_peak"], "bars"].median()
    print(f" those late ones last a median {med_bars:.0f} bars against "
          f"{over['bars'].median():.0f} for the rest —\n the failure mode is "
          f"SPEED, not size.")

    print(f"\n{'=' * 78}\n THE BIGGEST {args.top} LEGS OF EACH NAME — the ones the "
          f"objection deletes\n{'=' * 78}")
    print(f" legs: {len(T):,}    median size {T['move'].median():.0%}    "
          f"largest {T['move'].max():.0%}")
    print(f" to the peak: captured {T['capture'].median():.0%} of the leg's log "
          f"return (ceiling {T['ceiling_entry'].median():.0%})")
    print(f" the trade that started in them returned a median "
          f"{T['rt_return'].median():+.1%}, "
          f"{(T['rt_return'] > 0).mean():.0%} of them winners")
    print(f" never green at all:         {int((~T['green_by_peak']).sum())} of {len(T):,}")
    print(f" green only at the peak:     {int((~T['green_before_peak']).sum())} of {len(T):,}")
    print(f" collected nothing:          {int(T['collected_nothing'].sum())} of {len(T):,}")
    worst = T.nsmallest(5, "capture")[["ticker", "move", "capture", "bars"]]
    print("\n the five worst-captured big legs in the whole sample:")
    for _, r in worst.iterrows():
        print(f"   {r['ticker']:<6} {r['move']:>8.0%} move over {int(r['bars']):>4} bars"
              f"  ->  captured {r['capture']:>7.0%}")

    # --- arriving late, which is the realistic failure ----------------------
    print(f"\n{'=' * 78}\n ARRIVING LATE INSTEAD OF NOT ARRIVING\n{'=' * 78}")
    print(f" {'delay':>12}{'all legs':>12}{'big legs':>12}{'change':>10}")
    base_all = base_top = None
    rows = []
    states = {t: band_state(px, b)[0] for t, px in keep.items()}
    for d in args.delays:
        caps_all, caps_top = [], []
        for t, g in A.groupby("ticker"):
            px, st = keep[t], states[t]
            for _, r in g.iterrows():
                lg = r["leg_log"]
                got = exposure_log_return(px, st, int(r["start"]), int(r["end"]), d)
                caps_all.append(got / lg)
            gt = g.nlargest(min(args.top, len(g)), "move")
            for _, r in gt.iterrows():
                lg = r["leg_log"]
                got = exposure_log_return(px, st, int(r["start"]), int(r["end"]), d)
                caps_top.append(got / lg)
        ma, mt = float(np.median(caps_all)), float(np.median(caps_top))
        if base_all is None:
            base_all, base_top = ma, mt
        rows.append({"delay": d, "all": ma, "top": mt})
        print(f" {str(d) + ' bars':>12}{ma:>12.0%}{mt:>12.0%}{mt - base_top:>+10.1%}")

    pd.DataFrame(rows).to_csv("reports/capture_toll_delay.csv", index=False)

    # --- what the toll REALLY was, versus what the algebra claimed -----------
    RT = np.array([r for px in keep.values() for r in realised_tolls(px, b)])
    print(f"\n{'=' * 78}\n THE TOLL THE RULE ACTUALLY PAID — {len(RT):,} complete "
          f"round trips\n{'=' * 78}")
    print(f" {'':20}{'p25':>8}{'median':>9}{'p75':>8}{'p90':>8}{'over 8%':>10}")
    for lab, col in (("entry overshoot", 0), ("exit shortfall", 1)):
        v = RT[:, col]
        print(f" {lab:20}" + "".join(f"{np.quantile(v, q):>8.1%}"
                                     for q in (.25, .5, .75, .9))
              + f"{float((v > b + 1e-9).mean()):>10.0%}")
    toll = (1 + RT[:, 0]) / (1 - RT[:, 1])
    print(f"\n realised round-trip toll: median {np.median(toll) - 1:.1%} of "
          f"price, p90 {np.quantile(toll, .9) - 1:.1%}")
    print(f" the algebra's floor:      {(1 + b) / (1 - b) - 1:.1%} — reachable "
          f"only with perfect fills")

    print(f"\n where round-trip P/L actually crosses zero:")
    print(f" {'move spanned':>16}{'trips':>8}{'median P/L':>13}{'win rate':>11}")
    for lo_, hi_ in [(0, .10), (.10, breakeven_move(b)), (breakeven_move(b), .20),
                     (.20, .25), (.25, .30), (.30, .40), (.40, 1e9)]:
        m = (RT[:, 3] >= lo_) & (RT[:, 3] < hi_)
        if m.sum() < 10:
            continue
        lab = f"{lo_:.0%}-{hi_:.0%}" if hi_ < 1e9 else f">{lo_:.0%}"
        print(f" {lab:>16}{int(m.sum()):>8,}{np.median(RT[m, 2]):>+13.1%}"
              f"{float((RT[m, 2] > 0).mean()):>11.0%}")

    # --- and what a big move looks like when it is not defined by the band ---
    Bm = np.array([r for px in keep.values() for r in bull_moves(px, b)])
    if len(Bm):
        print(f"\n{'=' * 78}\n A REAL BULL MOVE IS NOT ONE LEG — {len(Bm):,} "
              f"moves of +50% or more (30% zigzag)\n{'=' * 78}")
        print(f" {'move size':>14}{'moves':>8}{'8% legs inside':>17}"
              f"{'captured':>11}{'formula says':>14}")
        for lo_, hi_ in [(.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)]:
            m = (Bm[:, 0] >= lo_) & (Bm[:, 0] < hi_)
            if m.sum() < 3:
                continue
            lab = f"{lo_:.0%}-{hi_:.0%}" if hi_ < 1e9 else f">{lo_:.0%}"
            print(f" {lab:>14}{int(m.sum()):>8}{np.median(Bm[m, 1]):>17.0f}"
                  f"{np.nanmedian(Bm[m, 2]):>11.0%}"
                  f"{np.nanmedian(Bm[m, 3]):>14.0%}")
        print(f"\n a real bull move holds a median {np.median(Bm[:, 1]):.0f} "
              f"separate {b:.0%} legs and pays that\n many tolls: captured "
              f"{np.nanmedian(Bm[:, 2]):.0%} of its log return where the "
              f"one-leg formula implies {np.nanmedian(Bm[:, 3]):.0%}.")

    print(f"\n{'=' * 78}\n THE ANSWER\n{'=' * 78}")
    tm = T["capture"].median()
    real_be = float(np.median((1 + RT[:, 0]) / (1 - RT[:, 1])) - 1)
    print(f" What is EXACT: the rule never once failed to turn green on a big "
          f"leg —\n {int((~T['green_by_peak']).sum())} of {len(T):,}. A move "
          f"bigger than the band trips the flip by construction,\n so it cannot "
          f"be missed. That is the part that is arithmetic.")
    print(f"\n What it KEEPS is smaller than this file first claimed. Measured on "
          f"a same-band\n leg it is {tm:.0%}, but such a leg is one round trip by "
          f"construction. Measured on a\n real bull move — a median of "
          f"{np.median(Bm[:, 1]):.0f} band-legs, each paying its own toll — it is "
          f"{np.nanmedian(Bm[:, 2]):.0%}.")
    print(f"\n And the toll is a FLOOR, not a fixed cost: entries overshoot the "
          f"trigger, so the\n realised round trip costs a median {real_be:.1%} of "
          f"price against the algebra's "
          f"{(1 + b) / (1 - b) - 1:.1%},\n which puts the true break-even near "
          f"{real_be:.0%} rather than {breakeven_move(b):.1%}.")
    print(f"\n The losses remain where they were: the {int(B.iloc[0]['legs']):,} "
          f"legs below break-even,\n which no amount of accuracy can rescue.")
    print("\n -> reports/capture_toll_legs.csv, reports/capture_toll_delay.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
