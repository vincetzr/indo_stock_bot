"""Data-quality gates for the price spine: what is real, and what is tradeable.

WHY THIS EXISTS
---------------
Running the encoded auto-rejection bands against 843 IDX tickers and 2.6 million
bars was meant to validate the RULES. It validated them - 99.99% of days sit
inside the bands - but it also surfaced three defects in the price data itself,
each of which corrupts returns in a way that runs clean and never announces
itself:

    STALE BARS      421,942 bars, 16.2% of the whole spine, have zero volume.
                    They repeat the previous close with open = high = low =
                    close. They are days the stock DID NOT TRADE. A backtest
                    that fills on one has bought something nobody was selling,
                    and a return series that includes them reports a real zero
                    where there was no observation at all. Some names are over
                    70% stale.

    DECIMAL SPIKES  a handful of bars where the entire row - open, high, low,
                    close AND adjusted close - is exactly ten times, or one
                    tenth of, both neighbours, then reverts. MAPI in May 2018
                    has three; ELTY has four. These are source errors, not
                    prices.

    LEVEL SHIFTS    a persistent jump by a clean ratio that never reverts, i.e.
                    a split or reverse split the cached series was not adjusted
                    for. SCCO fell 75% on 2024-02-01 on a 1:4 split. Nothing
                    happened to the company that day.

THE ONE THAT MATTERS MOST IS THE BORING ONE. Decimal spikes are rare and
dramatic; stale bars are one bar in six and invisible. A momentum signal
measured across stale bars is measuring the exchange's habit of repeating a
price, not the market's.

WHY THE NAIVE SPLIT DETECTOR IS WRONG ON THIS EXCHANGE
------------------------------------------------------
A first pass flagged 79 "unadjusted corporate actions", and most were nonsense:
BTEK going from Rp 3 to Rp 2 is a ratio of 1.5, and on a stock priced at Rp 3
that is ONE TICK. IDX has hundreds of names trading in the single rupiah, where
an ordinary tick is a 33-50% move. So a ratio alone cannot distinguish a split
from a normal day, and :func:`level_shifts` requires the move to be large in
TICKS as well as in ratio. That cut the 79 down to the three that are real.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .reference import (COVERAGE_START, OutsideCoverage,
                        auto_rejection, infer_board, tick_size,
                        was_locked)

#: Ratios a genuine split or reverse split takes. Anything else is a price move.
SPLIT_RATIOS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0, 25.0,
                40.0, 50.0, 100.0)

#: How close to one of those a ratio must sit to be called a split.
RATIO_TOLERANCE = 0.02

#: A move must ALSO be at least this many ticks to be a split candidate. On a
#: stock priced in single rupiah a 2x ratio is one tick, so ratio alone is
#: meaningless there - this is what separates SCCO's real 1:4 from BTEK's
#: ordinary Rp 3 to Rp 2.
MIN_SPLIT_TICKS = 20

#: IDX's main board price floor. Superseded for board inference by
#: :func:`idxbot.spine.reference.infer_board`, which uses IDX's published
#: watchlist criterion instead of a bare price cut, but kept because the floor
#: is still the reason a sub-Rp 50 quote pre-2023 is unexplained.
MAIN_BOARD_FLOOR = 50.0


def stale_bars(df: pd.DataFrame) -> pd.Series:
    """True where the bar records no trading.

    Zero volume is the definition. The flat open = high = low = close shape
    usually accompanies it but is not required: a bar can print a real range on
    a single lot, and a bar can be flat on genuine volume.
    """
    v = pd.to_numeric(df.get("volume"), errors="coerce")
    return (v.isna() | (v <= 0)).rename("stale")


def decimal_spikes(df: pd.DataFrame, tol: float = 0.02,
                   ratios: Optional[Sequence[float]] = None) -> pd.Series:
    """True where a bar is a clean-ratio source error that reverts next day.

    Two conditions together, because either alone has honest explanations: the
    move must be a clean corporate-action-shaped ratio, AND it must undo itself
    on the following bar. A real 10x move does not reverse exactly.

    Originally this looked only for powers of ten, which is the MAPI/ELTY shape.
    Tracing SCCO's misdated split turned up a different one: 2019-06-19 reads
    9,250 -> 2,325 -> 9,350, an isolated FOUR-fold dip that recovers the next
    day. Same defect, different ratio, and the decade-only test walked straight
    past it. The ratio set is now :data:`SPLIT_RATIOS`, which is where a
    mis-applied adjustment factor would land.
    """
    c = pd.to_numeric(df.get("close"), errors="coerce").to_numpy(dtype=float)
    out = np.zeros(len(c), dtype=bool)
    if len(c) < 3:
        return pd.Series(out, index=df.index, name="spike")
    cand = np.asarray(SPLIT_RATIOS if ratios is None else ratios, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_in = c[1:-1] / c[:-2]
        r_out = c[2:] / c[1:-1]
        reverts = np.abs(r_in * r_out - 1.0) < tol
        size = np.where(r_in > 1.0, r_in, 1.0 / np.where(r_in > 0, r_in, np.nan))
        clean = np.min(np.abs(cand[None, :] - size[:, None]) / cand[None, :],
                       axis=1) < tol
    # A clean ratio is not enough on this exchange, for the same reason it was
    # not enough in `level_shifts`: on a stock priced in single rupiah a 1.5x
    # "spike" is one tick down and one tick back. The move must be one the
    # exchange COULD NOT HAVE PERMITTED - beyond that day's auto-rejection band
    # - which is impossible without a corporate action and therefore an error.
    # Without this, 121 spikes are found across the universe and most of them
    # are ordinary days on penny stocks.
    impossible = np.zeros(len(r_in), dtype=bool)
    dates = pd.to_datetime(df["date"]).to_numpy() if "date" in df else None
    for k in range(len(r_in)):
        if not (reverts[k] and clean[k]):
            continue
        ref = c[k]
        day = dates[k + 1] if dates is not None else None
        # Board membership is DERIVED from IDX's published criterion rather
        # than guessed: from 2023-06-12 a six-month average regular-market
        # price below Rp 51 puts a stock on the Papan Pemantauan Khusus, whose
        # ladder is far looser. Before that the criterion did not exist, so a
        # sub-Rp 50 quote returns "unknown" - and unknown is treated as the
        # LOOSER ladder, because assuming the tight one would manufacture
        # impossible-move flags on exactly the names least able to bear them.
        if day is None:
            continue
        lo = max(0, k - 120)
        avg6 = float(np.nanmean(c[lo:k + 1])) if k > lo else ref
        board = infer_board(day, avg_price_6m=avg6, price=ref)
        if board == "unknown":
            board = "acceleration"
        try:
            up, dn = auto_rejection(ref, day, board)
        except (OutsideCoverage, ValueError):
            # Before 2014 the bands are not encoded, so the impossible-move
            # test cannot be applied and nothing may be asserted. Falling back
            # on a guessed 35% flagged every pre-2014 penny-stock tick as a
            # source error - APIC in 2005, BNBR in 2002, SMRA in 2002 - none of
            # which is evidence of anything.
            continue
        up = abs(up) / ref if up < 0 else up
        dn = abs(dn) / ref if dn < 0 else dn
        move = r_in[k] - 1.0
        impossible[k] = move > up + 1e-9 if move > 0 else -move > dn + 1e-9
    out[1:-1] = (reverts & clean & impossible
                 & np.isfinite(r_in) & np.isfinite(r_out))
    return pd.Series(out, index=df.index, name="spike")


def level_shifts(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Persistent clean-ratio jumps: splits the series was never adjusted for.

    Requires the shift to persist (the median of the next ``window`` bars stays
    at the new level) AND to be worth at least :data:`MIN_SPLIT_TICKS` ticks.
    The tick requirement is what stops every penny stock in the market being
    reported as splitting weekly.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    t = d[~stale_bars(d).to_numpy()].reset_index(drop=True)
    if len(t) < 2 * window + 1:
        return pd.DataFrame(columns=["date", "ratio", "before", "after",
                                     "ticks"])
    c = pd.to_numeric(t["close"], errors="coerce").to_numpy(dtype=float)
    ratios = np.asarray(SPLIT_RATIOS)
    hits: List[Dict] = []
    for i in range(window, len(c) - window):
        if not (c[i] > 0 and c[i - 1] > 0):
            continue
        r = c[i - 1] / c[i]
        rr = r if r > 1 else (1.0 / r if r > 0 else 0.0)
        if rr < 1.4:
            continue
        if np.min(np.abs(ratios - rr)) / rr > RATIO_TOLERANCE:
            continue
        before = float(np.median(c[i - window:i]))
        after = float(np.median(c[i:i + window]))
        if after <= 0 or abs((before / after) / rr - 1.0) > 0.05:
            continue                            # the ratio does not hold
        # The MEDIAN persisting is not enough: a three-bar dip inside a
        # five-bar window has the same median as a genuine shift, and a split
        # does not undo itself the same week. Require EVERY bar in the window
        # to sit closer to the new level than to the old one, on a log scale
        # so the test is symmetric in direction.
        after_win = c[i:i + window]
        if not np.all(np.abs(np.log(after_win / after))
                      < np.abs(np.log(after_win / before))):
            continue                            # the shift reverted
        day = t["date"].iloc[i]
        try:
            tick = tick_size(min(before, after), day)
        except OutsideCoverage:
            # Before 2014 the tick ladder is not encoded, so the test that
            # separates a real split from one tick on a penny stock cannot be
            # applied. Skipping is the honest outcome: asserting a split here
            # would be a guess, and ELTY alone produced seven of them in 2003.
            continue
        ticks = abs(before - after) / tick if tick > 0 else np.inf
        if ticks < MIN_SPLIT_TICKS:
            continue                            # one tick on a penny stock
        # AND the move must be one the exchange could not have permitted, the
        # same requirement :func:`decimal_spikes` makes. A clean ratio near 1.5
        # is a 33% fall, and on a sub-Rp 200 stock under a 35% band that is an
        # ordinary bad day. Without this, RODA (99 -> 65, -34.3%), MMLP and
        # BAPI were all quarantined as suspected corporate actions when nothing
        # happened to any of them beyond a legal fall.
        step = c[i] / c[i - 1]
        lo_w = max(0, i - 120)
        avg6 = float(np.nanmean(c[lo_w:i])) if i > lo_w else float(c[i - 1])
        board = infer_board(day, avg_price_6m=avg6, price=float(c[i - 1]))
        if board == "unknown":
            board = "acceleration"
        try:
            up, dn = auto_rejection(float(c[i - 1]), day, board)
        except (OutsideCoverage, ValueError):
            continue
        ref = float(c[i - 1])
        up = abs(up) / ref if up < 0 else up
        dn = abs(dn) / ref if dn < 0 else dn
        move = step - 1.0
        if not (move > up + 1e-9 if move > 0 else -move > dn + 1e-9):
            continue                            # the exchange allowed this
        hits.append({"date": day, "ratio": round(rr, 3), "before": before,
                     "after": after, "ticks": float(ticks)})
    return pd.DataFrame(hits)


def locked_bars(df: pd.DataFrame, board: str = "main") -> pd.Series:
    """True where the bar sat at an auto-rejection limit all session.

    A day you could not have traded, for the same practical reason as a stale
    bar: there was no counterparty at any price you could reach.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    prev = pd.to_numeric(d["close"], errors="coerce").shift(1)
    out = np.zeros(len(d), dtype=bool)
    for j, (i, row) in enumerate(d.iterrows()):
        p = prev.iloc[j]
        if not np.isfinite(p) or p <= 0:
            continue
        try:
            out[j] = was_locked(row["open"], row["high"], row["low"],
                                row["close"], float(p), row["date"],
                                board) is not None
        except OutsideCoverage:
            out[j] = False
    return pd.Series(out, index=df.index, name="locked")


def tradeable(df: pd.DataFrame, board: str = "main") -> pd.Series:
    """The mask every backtest must apply before assuming it could have filled.

    A bar is tradeable when it is not stale, not a source error, and not pinned
    at a rejection limit. This is deliberately conservative in the direction
    that COSTS return: excluding a day you could actually have traded loses a
    little edge, while including one you could not invents edge that never
    existed.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    ok = ~stale_bars(d) & ~decimal_spikes(d) & ~locked_bars(d, board)
    return ok.rename("tradeable")


def clean(df: pd.DataFrame, board: str = "main",
          drop_stale: bool = True) -> pd.DataFrame:
    """Annotate a price frame with every quality flag, optionally dropping stale.

    Dropping is the default because a stale bar is not a cheap observation, it
    is the ABSENCE of one, and carrying it forward into a return series
    manufactures zeros. Returns must be recomputed after the drop, not before -
    a return measured across a gap is a real return over a longer interval,
    which is honest; a zero on a day nothing traded is not.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    d["stale"] = stale_bars(d).to_numpy()
    d["spike"] = decimal_spikes(d).to_numpy()
    d["locked"] = locked_bars(d, board).to_numpy()
    d["tradeable"] = ~d["stale"] & ~d["spike"] & ~d["locked"]
    if drop_stale:
        d = d[~d["stale"] & ~d["spike"]].reset_index(drop=True)
    return d


def report(df: pd.DataFrame, ticker: str = "") -> Dict[str, object]:
    """One ticker's quality summary, for the Gate 0 table."""
    d = df.copy()
    if d.empty:
        return {"ticker": ticker, "bars": 0}
    d["date"] = pd.to_datetime(d["date"])
    covered = d[d["date"] >= COVERAGE_START]
    st = stale_bars(d)
    sp = decimal_spikes(d)
    sh = level_shifts(d)
    lk = locked_bars(covered) if len(covered) else pd.Series(dtype=bool)
    return {
        "ticker": ticker, "bars": len(d),
        "first": d["date"].min(), "last": d["date"].max(),
        "stale": int(st.sum()), "stale_pct": float(st.mean()),
        "spikes": int(sp.sum()),
        "shifts": len(sh),
        "shift_dates": ", ".join(f"{r['date']:%Y-%m-%d}(1:{r['ratio']:g})"
                                 for _, r in sh.iterrows()),
        "locked": int(lk.sum()) if len(lk) else 0,
        "tradeable_pct": float((~st & ~sp).mean()),
    }
