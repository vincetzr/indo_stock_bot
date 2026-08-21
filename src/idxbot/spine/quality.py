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

THE TICK GRID IS A FINGERPRINT, AND IT SETTLES WHAT THE OTHER TESTS CANNOT
--------------------------------------------------------------------------
Every price IDX ever printed is an exact multiple of that day's fraksi harga.
A price that is not - Rp 6,234.246582, Rp 122.380951 - was never traded. It is
arithmetic a data vendor did, and the only thing that produces it is a
back-adjustment by a factor that is not a whole number of ticks.

That gives two things the ratio tests cannot give.

    IT PROVES A FACTOR.  :func:`factor_fits` divides a window by a CANDIDATE
    factor - taken from an announcement, never from the tape - and asks whether
    every open, high, low and close lands exactly on the grid, and every volume
    on a whole lot. PYFA's window has 16 prices and 4 volumes; SINI's has 32 and
    3. Chance of that under a wrong factor is the tick spacing to the twentieth
    power. This is what promoted both from "cause known, factor guessed" to a
    repair, and it is why neither was adjusted on a factor read off the chart -
    which would have been circular, the move being the thing to explain.

    IT BOUNDS HOW MUCH OF THE SPINE IS ADJUSTED.  :func:`off_grid_rate` counts
    the bars that provably were not traded at the price shown. That matters
    beyond corporate actions: the cost model in
    :func:`idxbot.spine.reference.half_spread` looks the tick up BY PRICE, so a
    series back-adjusted to a fifth of its traded level is charged out of the
    wrong band and understates the spread it would really have paid.

THE TEST IS ONE-SIDED AND TWO VERSIONS OF THIS GOT IT WRONG
-----------------------------------------------------------
Off-grid proves adjustment. On-grid proves NOTHING, and every mistake here came
from forgetting the second half.

    First attempt read each bar alone. BMRI's whole 2005-2023 history is
    divided by 4 and 2.3% of those bars land on the grid regardless - 2,850
    divides by 10, 2,837.50 does not - so one uniform region came back as
    18,300 fictitious ones.

    Second attempt fixed that with gap-closing and then claimed the result
    SEGMENTED the series into raw and adjusted. It does not. PTBA's pre-2017
    history is divided by 5, and every session where the real price was a
    multiple of Rp 50 lands on the grid - 122 consecutive sessions of it across
    the 2008 crash. Those read as a clean raw stretch and split one adjusted
    region into thirteen, of which twelve are fiction. Across the spine that
    produced 2,760 "defects".

So there is no segmentation function here, deliberately. There is a rate, which
is a LOWER BOUND, and there is :func:`suspect_islands`, which reports only what
:func:`level_shifts` independently agrees is a break. A whole-number factor
(MAPI's 10, ELTY's proposed 10) leaves every price on the grid and is invisible
to this test entirely. It complements the ratio tests; it does not replace them.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import reference as _ref
from .reference import (COVERAGE_START, EARLY_START, OutsideCoverage,
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


# --------------------------------------------------------------------------
# the tick grid
# --------------------------------------------------------------------------
#: Relative slack when asking whether a price sits on its tick. Prices arrive
#: as float32 from the cache, so an exact multiple can be off by ~1e-7 of its
#: own size. Anything looser starts accepting genuinely off-grid prices: the
#: smallest real offence in the spine is half a tick.
GRID_TOL = 1e-4

#: The longest run of on-grid bars that may sit INSIDE an adjusted stretch
#: without ending it, when gathering off-grid bars into islands.
BASIS_GAP = 5

#: How long the clean stretch on each side of an island should be before the
#: island is worth reporting at all - and the shortest flank still accepted
#: when the island sits against the start or the end of the series. Insisting
#: on the full 60 at the end of the series makes any defect in the last three
#: months invisible, which is the opposite of the priority: SINI's island is
#: eight weeks old and it is the one that would be traded on.
ISLAND_CLEAN_BARS = 60
ISLAND_MIN_FLANK = 5

#: Longest island still shaped like the defect. SCCO's is 23 sessions, SINI's
#: 8, PYFA's 4.
ISLAND_MAX_BARS = 120


def tick_grid(prices, dates) -> np.ndarray:
    """The tick size for each (price, date) pair. Vectorised over regimes.

    NaN where the date is outside the tick schedule's coverage, so a caller
    gets an absence rather than a wrong grid. :func:`off_tick` treats NaN as
    "cannot say", never as "on grid".
    """
    p = np.asarray(prices, dtype=float)
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    out = np.full(p.shape, np.nan)
    for r in _ref._TICK:
        end = np.datetime64(r.end) if r.end is not None else None
        m = d >= np.datetime64(r.start)
        if end is not None:
            m &= d < end
        if not m.any():
            continue
        # bands are (upper_exclusive, tick) ascending, so walking them
        # BACKWARDS lets each cheaper band overwrite the dearer one and the
        # lowest matching ceiling wins - the same rule as Regime.value_for.
        px = p[m]
        sub = np.full(px.shape, np.nan)
        for ceiling, value in reversed(r.bands):
            if ceiling is None:
                sub[:] = value
            else:
                sub = np.where(px < ceiling, value, sub)
        out[m] = sub
    out[~np.isfinite(p) | (p <= 0)] = np.nan
    return out


def off_tick(df: pd.DataFrame, tol: float = GRID_TOL) -> pd.Series:
    """True where any of open/high/low/close is not a multiple of its tick.

    A True bar was not traded at that price - it is a vendor back-adjustment.
    Bars outside the tick schedule's coverage come back False, because "no rule
    encoded" is not evidence of a defect.
    """
    if df.empty or "date" not in df:
        return pd.Series(dtype=bool, name="off_tick")
    d = pd.to_datetime(df["date"])
    bad = np.zeros(len(df), dtype=bool)
    for c in ("open", "high", "low", "close"):
        if c not in df:
            continue
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        g = tick_grid(v, d)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.abs(v / g - np.round(v / g))
        bad |= np.isfinite(r) & (r > tol)
    return pd.Series(bad, index=df.index, name="off_tick")


def _runs(mask: np.ndarray):
    """(start, stop) index pairs of each maximal True run, stop exclusive."""
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(e[::2], e[1::2]))


def off_grid_rate(df: pd.DataFrame) -> float:
    """Share of bars proven to be on a vendor-adjusted basis.

    A LOWER BOUND on how much of the series is adjusted, never an estimate of
    it, and the asymmetry is the whole point. Off-grid proves adjustment;
    on-grid proves nothing. PTBA's entire pre-2017 history is divided by five,
    and about a quarter of it lands on the grid anyway - every session where
    the real price was a multiple of Rp 50. Read this as "at least this much",
    and read a zero as "nothing proven", not as "clean".
    """
    if df.empty:
        return float("nan")
    b = off_tick(df)
    return float(b.mean()) if len(b) else float("nan")


def adjustment_islands(df: pd.DataFrame, gap: int = BASIS_GAP,
                       clean: int = ISLAND_CLEAN_BARS,
                       max_bars: int = ISLAND_MAX_BARS,
                       min_flank: int = ISLAND_MIN_FLANK) -> pd.DataFrame:
    """Short off-grid stretches with long clean history on BOTH sides.

    This is the shape of the SCCO defect: a vendor that back-adjusts a whole
    history leaves one long adjusted region, which is consistent and corrupts
    no single return, while a vendor that adjusts only the last few sessions
    before an ex-date leaves an ISLAND that corrupts the return at both of its
    edges - a fake crash going in, a fake rally coming out.

    CANDIDATES, NOT VERDICTS. Because on-grid proves nothing, a stretch of a
    divided history that happens to land on the grid looks exactly like clean
    flanking data, and PTBA alone produces a dozen such islands that are not
    defects. Confirmation comes from :func:`level_shifts` agreeing there is a
    break at an edge and from :func:`factor_fits` accepting an announced
    factor - see :func:`suspect_islands`.
    """
    cols = ["start", "end", "bars", "off_grid_rate"]
    if df.empty or "date" not in df:
        return pd.DataFrame(columns=cols)
    d = pd.to_datetime(df["date"]).reset_index(drop=True)
    # A zero-volume bar is the previous quote repeated, so its being off-grid
    # is inherited and says nothing new. Counting them made WIKA - suspended
    # from November 2023 to April 2024, its stale quote re-marked once for a
    # rights issue - look like a ten-session vendor adjustment. It is one
    # forward-filled number, and stale_bars already reports it.
    bad = (off_tick(df) & ~stale_bars(df)).to_numpy()
    if not len(bad) or not bad.any():
        return pd.DataFrame(columns=cols)
    state = bad.copy()
    for s, e in _runs(~bad):
        if 0 < s and e < len(bad) and (e - s) <= gap:
            state[s:e] = True                       # close the small gaps
    rows = []
    for s, e in _runs(state):
        if (e - s) > max_bars:
            continue
        left, right = s, len(state) - e
        if min(left, right) < min_flank:
            continue
        if (state[max(0, s - clean):s].any()
                or state[e:e + clean].any()):
            continue                                # flanks must be clean
        rows.append({"start": d[s], "end": d[e - 1], "bars": int(e - s),
                     "off_grid_rate": float(bad[s:e].mean())})
    return pd.DataFrame(rows, columns=cols)


def suspect_islands(df: pd.DataFrame, **kw) -> pd.DataFrame:
    """Islands that :func:`level_shifts` also calls a break. The real defects.

    The conjunction is what makes this usable. An island on its own is a
    candidate and there are thousands of them; a level shift on its own says a
    break happened but not that a vendor caused it. Both together say the
    series changed basis and the new basis was never traded, which is the SCCO
    defect and nothing else.

    Adds ``shift_date`` and ``shift_ratio`` from the matching break.
    """
    isl = adjustment_islands(df, **kw)
    cols = list(isl.columns) + ["shift_date", "shift_ratio"]
    if isl.empty:
        return pd.DataFrame(columns=cols)
    sh = level_shifts(df)
    if sh.empty:
        return pd.DataFrame(columns=cols)
    sd = pd.to_datetime(sh["date"])
    rows = []
    for _, r in isl.iterrows():
        # a break at either edge, allowing the gap-closing's slack in days
        near = sh[(sd >= r["start"] - pd.Timedelta(days=7))
                  & (sd <= r["end"] + pd.Timedelta(days=7))]
        if near.empty:
            continue
        rows.append({**r, "shift_date": near.iloc[0]["date"],
                     "shift_ratio": float(near.iloc[0]["ratio"])})
    return pd.DataFrame(rows, columns=cols)


def factor_fits(df: pd.DataFrame, start, end, factor: float,
                tol: float = GRID_TOL, lot: int = 100) -> Dict[str, object]:
    """Test a CANDIDATE adjustment factor against the tick and lot grids.

    ``factor`` is what the vendor multiplied prices by, so dividing by it
    should recover what the exchange printed. The candidate must come from an
    announcement. Deriving it from the price move and then testing it here
    would be circular - the move is the thing being explained - so the value of
    this test is precisely that the announcement fixes the factor and the grid
    then gets a free vote.

    Returns the worst grid error over every price, the share of prices that
    land exactly, and the same for volume against the 100-share lot. A real
    factor scores 1.0 on both; a wrong one scores at chance, which is the tick
    spacing itself.
    """
    d = pd.to_datetime(df["date"])
    w = df[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]
    if w.empty or not factor:
        return {"prices": 0, "price_exact": float("nan"),
                "max_grid_error": float("nan"), "volumes": 0,
                "volume_exact": float("nan")}
    wd = pd.to_datetime(w["date"])
    errs, n_ok, n = [], 0, 0
    for c in ("open", "high", "low", "close"):
        if c not in w:
            continue
        v = pd.to_numeric(w[c], errors="coerce").to_numpy(dtype=float) / factor
        g = tick_grid(v, wd)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.abs(v / g - np.round(v / g))
        r = r[np.isfinite(r)]
        errs.extend(r.tolist())
        n_ok += int((r <= tol).sum())
        n += len(r)
    vol = pd.to_numeric(w.get("volume"), errors="coerce").to_numpy(dtype=float)
    vol = vol[np.isfinite(vol) & (vol > 0)] * factor
    # Volume is checked in SHARES, not in units of the lot, and to about one
    # share. The vendor rounds the adjusted volume to a whole share before
    # storing it, so dividing by a factor of 0.67 recovers the true figure only
    # to +-0.7 shares. Demanding exactness here failed SINI on a discrepancy of
    # 0.22 shares in 3.4 million - which is the rounding, not the factor.
    slack = np.maximum(1.0, 2e-6 * np.abs(vol))
    v_ok = int((np.abs(vol - np.round(vol / lot) * lot) <= slack).sum())
    return {"prices": n,
            "price_exact": (n_ok / n) if n else float("nan"),
            "max_grid_error": max(errs) if errs else float("nan"),
            "volumes": len(vol),
            "volume_exact": (v_ok / len(vol)) if len(vol) else float("nan")}


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
