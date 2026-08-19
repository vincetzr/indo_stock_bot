#!/usr/bin/env python3
"""Multi-timeframe OHLCV for IDX, built ONLY from the on-disk cache.

Nothing here touches the network. Every frame is assembled from
``data/cache/intraday/*.csv.gz`` (fetched previously) and ``data/cache/ohlcv/``
for the daily reference. If a ticker is not already cached this module returns
an empty frame; it never goes and gets it.

    load_1h(t)     hourly, straight from the 1h cache      ~3 years
    load_4h(t)     half-day session bars, resampled from 1h ~3 years
    load_15m(t)    15-minute, resampled from 5m             ~12 weeks
    load_daily(t)  daily reference, from the daily cache    since listing


THE SESSION, AS THE FEED ACTUALLY DELIVERS IT
---------------------------------------------
IDX trades 09:00-12:00 and 13:30-15:50 WIB, with a short Friday (09:00-11:30 and
14:00-15:50). Timestamps in the cache are naive Asia/Jakarta wall clock - the
fetcher converted from epoch UTC and dropped the tz - so hour-of-day is directly
comparable across the whole file with no DST to worry about (WIB has no DST).

Measured over the whole cache, the 5-minute grid that actually appears is:

    09:00 .. 11:55   Mon-Thu morning        (36 bars)
    09:00 .. 11:25   Fri morning            (30 bars)
    13:30 .. 15:45   Mon-Thu afternoon      (28 bars)
    14:00 .. 15:45   Fri afternoon          (24 bars)
    15:50            almost never present   (94 bars in 44 of 221 files)
    16:00            pre-closing auction     - always present, large volume
    16:05, 16:10     post-trading            - fixed at the closing price

The 16:00 bar is the closing cross, so it is kept. 16:05/16:10 are the
post-trading window, where IDX only allows trades at the already-fixed closing
price; they are kept but they are zero-range by construction, and on resampling
they fold into the 16:00 bar, which is where they belong.

The last intraday close does NOT always equal the daily close, and the mismatch
is not spread evenly. Over 39,878 session-name pairs the two agree 96.4% of the
time, but by quarter:

    2023Q3  84.5%    2024Q1-2026Q3  98.5% - 99.9%
    2023Q4  73.6%

The first five months of the hourly history are the bad part. From 2024-01-01 on
the layers agree on 99%+ of sessions. When they do disagree it is usually small
(median 0.85%, 76% of cases under 1.5%, roughly one tick), but 0.6% of mismatches
exceed 10%. **Start hourly backtests at 2024-01-01.** That costs five months of a
three-year window and removes the stretch where one session in four has an
intraday close the daily bar does not recognise. On BBCA agreement happens to be
100% throughout, which is exactly why this needs measuring on more than one name.

Hourly buckets land on 09,10,11,13,14,15,16 Mon-Thu and 09,10,11,14,15,16 on
Friday. A stray bar at 12:00 appears in 209 of the 778 files (3,003 bars total)
and is dropped: an hourly bucket starting at 12:00 lies entirely inside the
lunch break. Nothing in the cache falls before 09:00 or after 16:15.


RESAMPLING CONVENTION
---------------------
Two rules, applied to every interval:

  1. A bar may never span the lunch break or an overnight boundary. No bar is
     ever built from trades on two different days, or from trades on both sides
     of lunch.
  2. Bars are labelled by their LEFT edge - the timestamp of the first source
     bar inside them. A bar labelled 09:00 is the 09:00 bar; it is complete only
     after its window has elapsed. Nothing in this module ever reads forward.


AS-OF RECONSTRUCTION
--------------------
``end=X`` is not a slice of the finished frame. It truncates the raw source at X
and re-runs everything - guard, resample, the lot - so what comes back is what
you would have had standing at X and nothing else. ``--selftest`` proves this by
rebuilding from scratch at two cut points per name and comparing bar for bar.

It also withholds the bar that was still open at X. A 15m bar labelled 09:00 is
not finished at 09:00, and handing it back looking like a finished bar is how a
walk-forward loop ends up trading on a five-minute bar it believes is fifteen.
Exactly one trailing bar is ever affected; the count lands in
``df.attrs['report'].partial_tail``, and ``complete_only=False`` keeps it.

``start=`` is applied last, after the guard, on purpose. The guard recompounds
from the first bar, so trimming the head first would shift every price level
after it and two overlapping windows would disagree about the same bar.

**15m** is a plain 15-minute floor grid. This works without any special casing
because every IDX session boundary is already a multiple of 15 minutes past the
hour: 09:00, 12:00, 13:30, 14:00, 16:00. The bin [11:45,12:00) ends exactly at
the lunch bell and the bin [13:30,13:45) starts exactly at the afternoon bell,
so no 15-minute bin can straddle lunch, and no bin can straddle midnight.

**4h** is NOT a 4-hour clock grid, and the name is a convenient lie that has to
be stated plainly. The IDX trading day is 5h20m long (3h + 2h20m). There is no
way to lay a 4-hour grid on it that does not either span the lunch break or
chop the day into a ragged 4h + 1h20m pair. So ``load_4h`` returns the two
natural blocks the exchange itself defines:

    AM = session I  (09:00-12:00, 3h;    Fri 09:00-11:30, 2h30m)
    PM = session II (13:30-16:00, 2h30m; Fri 14:00-16:00, 2h)

Two bars a day, averaging 2h40m of trading each. Closer to a 2.5h bar than a 4h
one. It is used because it is the only session-respecting way to get a
coarse-but-intraday timeframe out of hourly data, not because it is four hours.

Bars per trading day, measured (not assumed):

    interval   Mon-Thu   Fri     bars/year (~240 sessions)
    1h              7      6         ~1,650
    4h              2      2           ~480
    15m            23     19         ~5,300
    5m (source)    67     55        ~15,400


DROPPED BARS
------------
NaN close, non-positive close, duplicate timestamps and out-of-session bars are
all dropped unconditionally. Zero-volume bars are dropped too, with one
exception that matters a great deal:

**the first bar of each session is kept even at zero volume.** In the 5-minute
feed the 09:00 bar has volume 0 in a median 100% of sessions (never below 89%
for any name); in the hourly feed the 09:00 bar reads zero in a median 74% of
sessions, and on BBCA that rate climbs from ~0% in late 2023 to 100% from
mid-2025 on. That is a feed change, not a market change.

The volume is *missing*, not zero, and the daily cache proves it. Over a
119-name sample: on sessions where the 09:00 hourly bar reads zero, the intraday
bars sum to a median 75.1% of the day's reported volume; on sessions where it
reads non-zero they sum to 98.1%. The missing ~23% is the first hour. The bar's
price range is also real - its close chains continuously into the next bar's
open - so dropping it would throw away the session open and the opening range
for three quarters of sessions while keeping bars that are no more real. Pass
``keep_session_open=False`` to drop them anyway; the count is reported either
way, and it is large: 391,301 hourly session opens across the 778 names.

The corollary is a warning, not a fix: **intraday volume in this cache does not
reconcile with daily volume, and the shortfall is time-varying.** Any signal of
the form "volume so far vs normal" is measuring a fraction that was ~98% in 2023
and is ~75% now. Calibrate such a signal on hourly data at your own risk.


THE IMPOSSIBLE-PRINT GUARD, AND WHY 35% IS THE WRONG NUMBER AT 15m
------------------------------------------------------------------
``scripts/paint_daily.py`` caps daily ``pct_change`` at +/-35% and recompounds.
That number is chosen as an upper bound on IDX auto-rejection: no stock can
close more than one ARA/ARB band away from the previous close, and 35% is the
widest band. Applied to a daily close series it is well calibrated - across
53,080 sampled overnight steps the 99.9th percentile is 25.0% and the 99.99th is
35.0%, so 35% sits exactly on the shoulder of the real distribution and clips
only what cannot be real. Cache-wide the largest raw overnight step is CUAN's
+905.6% on 2025-07-10, a reverse split; 83 names carry a step above 35% and 7
carry one above 100%.

Applied to a 15-minute bar it is **inert**. Measured over 91,491 in-session
15-minute steps: the 99.99th percentile is 13.1%, the largest is 24.6%, and the
number exceeding 35% is zero. A 35% cap at 15m does not clip a single bar in the
entire cache - it is not a loose guard, it is no guard at all. A fat-finger
print of +30% inside one 15-minute bar sails straight through it.

So the cap is split in two, because the two steps are different animals:

  * the OVERNIGHT step (last bar of one day -> first bar of the next) keeps
    35%, identical to ``paint_daily``. It is the same quantity the daily loader
    caps, it must be capped the same way or the timeframes disagree at every day
    boundary, and it is the step where corporate actions actually live.

  * the IN-SESSION step is capped at the resolution it is measured at:

        source   cap    in-session steps clipped, cache-wide sample
        1h       35%    4 in 270,756   (0.0015%)
        5m       20%    2 in 265,828   (0.0008%)

    20% at 5m is ~2,000x the median 5-minute step (0.24%) and ~2x the 99.99th
    percentile (9.6%), so it cannot touch a real move, while still catching a
    print that 35% would wave through. 1h keeps 35% because at hourly resolution
    the real distribution genuinely reaches into the twenties (99.99th pct
    26.0%, max 44.1%) and a tighter cap would start deleting real limit-up
    rushes.

The guard runs at the SOURCE resolution, before resampling, so a corrupt print
is corrected before it can contaminate a resampled bar's high or low. The
correction factor computed on the close is applied to open/high/low as well, so
each bar keeps its own geometry and OHLC stays internally consistent.

WHAT THE GUARD DOES NOT DO
--------------------------
It does not adjust for corporate actions. The intraday cache carries no
adj_close and no split factors. A reverse split shows up as a genuine
discontinuity; the guard bounds it to 35% and moves on, which turns an
impossible +905% into a merely wrong +35%. Every overnight step that hit the cap
before clipping is reported in ``df.attrs['suspect_jumps']`` - treat those dates
as unbacktestable for that name rather than trusting the clipped value. 730
steps larger than 30% were found across the 778 hourly files, and the guard
fires on 85 overnight steps spread over 58 names.

It is also wrong for sub-Rp50 names. 66 of the 778 hourly tickers have a median
close under Rp50; they trade on the special-monitoring full-call-auction board
where the tick is Rp1, so one tick on a Rp1 stock is +100% and the guard clips a
real move. BTEK alone takes 37 in-session clips. Filter these names out rather
than trusting the guard on them.

And it does nothing about the biggest hazard in this cache, which is not a price
at all: **five sessions are simply absent from the intraday feed.** 2025-09-24,
25, 26, 29 and 30 have no intraday bars for any of the 778 names, while the
daily cache has all five with normal volume - the exchange was open. An hourly
backtest steps straight from 2025-09-23's close to 2025-10-01's open and never
sees the week in between (BBCA fell 4.8% across it). ``--selftest`` will not
catch this; the coverage run cross-checks the intraday calendar against the
daily cache and prints it.

    python3 scripts/tf_data.py                 # coverage table + warnings
    python3 scripts/tf_data.py --names 50      # quick pass
    python3 scripts/tf_data.py --selftest      # structure + prefix identity
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))

from idxbot.data.ohlcv import to_yahoo_symbol  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
CACHE_ROOT = os.path.join(REPO_ROOT, "data", "cache")
INTRADAY_DIR = os.path.join(CACHE_ROOT, "intraday")
DAILY_DIR = os.path.join(CACHE_ROOT, "ohlcv")

# Session landmarks, minutes past midnight WIB.
SESSION_OPEN = 9 * 60           # 09:00
LUNCH_START = 12 * 60           # 12:00
LUNCH_END = 13 * 60 + 30        # 13:30
CLOSE_AUCTION = 16 * 60         # 16:00 pre-closing cross
POST_END = 16 * 60 + 15         # 16:15 end of post-trading

# Impossible-print caps. See the module docstring for the measurements behind
# these; OVERNIGHT_CAP is deliberately identical to paint_daily.DAILY_CAP.
OVERNIGHT_CAP = 0.35
INTRA_CAP = {"1h": 0.35, "4h": 0.35, "5m": 0.20, "15m": 0.20, "1d": 0.35}

# Width in minutes of one source bar, used to decide whether a bucket falls
# entirely inside the lunch break.
BAR_MINUTES = {"1h": 60, "5m": 5}

CACHE_KEY = {"1h": "1h_730d", "5m": "5m_60d"}

# Which cached interval each public timeframe is built from.
SOURCE = {"1h": "1h", "4h": "1h", "15m": "5m", "5m": "5m"}

OHLCV = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

@dataclass
class LoadReport:
    """Everything thrown away on the way to a clean frame."""

    ticker: str = ""
    interval: str = ""
    raw_bars: int = 0
    dup_ts: int = 0
    bad_close: int = 0
    out_of_session: int = 0
    zero_volume_dropped: int = 0
    zero_volume_kept_open: int = 0
    volume_unavailable: bool = False
    clipped_intra: int = 0
    clipped_overnight: int = 0
    max_abs_step_raw: float = float("nan")
    suspect_jumps: List[Tuple[str, float]] = field(default_factory=list)
    partial_tail: int = 0
    final_bars: int = 0

    def as_dict(self) -> Dict[str, object]:
        d = dict(self.__dict__)
        d["suspect_jumps"] = list(self.suspect_jumps)
        return d


def _empty(interval: str, ticker: str) -> pd.DataFrame:
    df = pd.DataFrame(columns=["ts"] + OHLCV + ["date", "session", "n_src"])
    df.attrs["report"] = LoadReport(ticker=ticker, interval=interval)
    return df


# ---------------------------------------------------------------------------
# cache access (read-only, never fetches)
# ---------------------------------------------------------------------------

def _safe(name: str) -> str:
    """The fetcher's filename sanitiser (see data/cache.py)."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _symbols(ticker: str) -> List[str]:
    """Cache basenames a ticker could be stored under, most likely first.

    ``to_yahoo_symbol`` alone is not enough to get back to a filename. The cache
    sanitises non-alphanumerics, so ``BZ=F`` is on disk as ``BZ_F`` - and
    ``to_yahoo_symbol("BZ_F")`` sees four characters with no dot and helpfully
    returns ``BZ_F.JK``, a symbol that cannot exist. The same trap catches
    ``^VIX`` -> ``_VIX`` -> ``_VIX.JK``. Try the sanitised name as given too.
    """
    raw = ticker.strip().upper()
    out, seen = [], set()
    for cand in (_safe(to_yahoo_symbol(raw)), _safe(raw)):
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def _pick(directory: str, ticker: str, suffix: str) -> str:
    for sym in _symbols(ticker):
        path = os.path.join(directory, f"{sym}{suffix}")
        if os.path.exists(path):
            return path
    return os.path.join(directory, f"{_symbols(ticker)[0]}{suffix}")


def _cache_key(ticker: str, interval: str) -> str:
    """Reproduce the fetcher's cache key, including its filename sanitising."""
    return _safe(f"{to_yahoo_symbol(ticker)}_{CACHE_KEY[interval]}")


def intraday_path(ticker: str, interval: str) -> str:
    return _pick(INTRADAY_DIR, ticker, f"_{CACHE_KEY[interval]}.csv.gz")


def daily_path(ticker: str) -> str:
    return _pick(DAILY_DIR, ticker, ".csv.gz")


def is_idx_equity(ticker: str) -> bool:
    """Four letters. Everything else in the cache is an index, FX or a future."""
    return len(ticker) == 4 and ticker.isalpha()


def available(interval: str, equities_only: bool = False) -> List[str]:
    """Tickers with a cache file for this interval, as ticker codes.

    The daily namespace is not an IDX universe: it also holds ``^JKSE``,
    ``^GSPC``, ``DX-Y.NYB`` and a handful of futures, cached by the macro code.
    Globbing it without ``equities_only`` gets you all of them.
    """
    if interval == "1d":
        paths = sorted(glob.glob(os.path.join(DAILY_DIR, "*.csv.gz")))
        names = [os.path.basename(p)[: -len(".csv.gz")].replace(".JK", "") for p in paths]
    else:
        suffix = f"_{CACHE_KEY[SOURCE[interval]]}.csv.gz"
        paths = sorted(glob.glob(os.path.join(INTRADAY_DIR, f"*{suffix}")))
        names = [os.path.basename(p)[: -len(suffix)].replace(".JK", "") for p in paths]
    return [n for n in names if is_idx_equity(n)] if equities_only else names


def _read_raw(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["ts"])
    except Exception:
        # A truncated cache entry must never break a run. Same policy as Cache.
        return None
    if df.empty or "close" not in df.columns:
        return None
    return df


# ---------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------

def _in_session(minute: pd.Series, width: int) -> pd.Series:
    """A bucket [minute, minute+width) that contains tradeable time.

    Dropped: anything before the 09:00 bell, anything after the 16:15 end of
    post-trading, and any bucket lying wholly inside the 12:00-13:30 lunch
    break. The hourly bucket labelled 13:00 survives because it reaches past
    13:30 and therefore holds real trades; the hourly bucket labelled 12:00
    does not.
    """
    before = minute < SESSION_OPEN
    after = minute > POST_END
    in_lunch = (minute >= LUNCH_START) & (minute + width <= LUNCH_END)
    return ~(before | after | in_lunch)


def _clean(df: pd.DataFrame, interval: str, ticker: str, *,
           drop_zero_volume: bool = True,
           keep_session_open: bool = True) -> Tuple[pd.DataFrame, LoadReport]:
    """Sort, de-duplicate, drop unusable and out-of-session bars."""
    rep = LoadReport(ticker=ticker, interval=interval, raw_bars=len(df))
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"])
    out = out.sort_values("ts", kind="mergesort")

    dup = out["ts"].duplicated(keep="last")
    rep.dup_ts = int(dup.sum())
    out = out[~dup]

    close = pd.to_numeric(out["close"], errors="coerce")
    bad = close.isna() | (close <= 0)
    rep.bad_close = int(bad.sum())
    out = out[~bad]
    if out.empty:
        return out, rep

    minute = out["ts"].dt.hour * 60 + out["ts"].dt.minute
    keep = _in_session(minute, BAR_MINUTES[interval])
    rep.out_of_session = int((~keep).sum())
    out = out[keep]
    minute = minute[keep]
    if out.empty:
        return out, rep

    for col in OHLCV:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["volume"] = out["volume"].fillna(0.0)
    out["date"] = out["ts"].dt.normalize()
    out["session"] = np.where(minute < LUNCH_START, "AM", "PM")

    # An index (^JKSE) has no volume at all. Dropping zero-volume bars would
    # delete it entirely, so detect the case and leave volume alone.
    zero = out["volume"] <= 0
    if zero.mean() > 0.9:
        rep.volume_unavailable = True
    elif drop_zero_volume:
        first_of_session = ~out.duplicated(subset=["date", "session"], keep="first")
        if keep_session_open:
            rep.zero_volume_kept_open = int((zero & first_of_session).sum())
            zero = zero & ~first_of_session
        rep.zero_volume_dropped = int(zero.sum())
        out = out[~zero]

    return out.reset_index(drop=True), rep


# ---------------------------------------------------------------------------
# impossible-print guard
# ---------------------------------------------------------------------------

def _guard(df: pd.DataFrame, rep: LoadReport, intra_cap: float,
           overnight_cap: float = OVERNIGHT_CAP) -> pd.DataFrame:
    """paint_daily's clip-and-recompound, split by step type and applied to OHLC.

    The close path is rebuilt from clipped returns exactly as
    ``paint_daily.unadjusted_daily`` does. Each bar's open/high/low is then
    multiplied by that bar's close correction factor, so a bar keeps its own
    shape and the frame stays OHLC-coherent instead of having a corrected close
    sitting inside an uncorrected range.
    """
    if len(df) < 2:
        return df
    out = df.copy()
    close = out["close"].to_numpy(float)
    same_day = out["date"].to_numpy()[1:] == out["date"].to_numpy()[:-1]

    step = close[1:] / close[:-1] - 1.0
    rep.max_abs_step_raw = float(np.nanmax(np.abs(step))) if len(step) else float("nan")

    cap = np.where(same_day, intra_cap, overnight_cap)
    clipped = np.clip(step, -cap, cap)
    touched = clipped != step
    rep.clipped_intra = int((touched & same_day).sum())
    rep.clipped_overnight = int((touched & ~same_day).sum())

    # Overnight jumps beyond the daily band are corporate actions or feed
    # errors. Bounding them to 35% does not make them right, so name them.
    ts = out["ts"].to_numpy()
    for i in np.where(touched & ~same_day)[0]:
        rep.suspect_jumps.append((str(pd.Timestamp(ts[i + 1])), float(step[i])))

    rebuilt = close[0] * np.concatenate([[1.0], np.cumprod(1.0 + clipped)])
    factor = rebuilt / close
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].to_numpy(float) * factor
    return out


# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------

def _agg(grouped) -> pd.DataFrame:
    return grouped.agg(
        ts=("ts", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        n_src=("close", "size"),
    )


def resample_15m(df: pd.DataFrame) -> pd.DataFrame:
    """5m -> 15m on a plain 15-minute floor grid.

    Safe without special casing: 12:00 and 13:30 are both on the grid, so no bin
    straddles lunch, and a floor grid cannot straddle midnight.
    """
    if df.empty:
        return df
    key = df["ts"].dt.floor("15min")
    out = _agg(df.groupby(key, sort=True)).reset_index(drop=True)
    out["date"] = out["ts"].dt.normalize()
    minute = out["ts"].dt.hour * 60 + out["ts"].dt.minute
    out["session"] = np.where(minute < LUNCH_START, "AM", "PM")
    return out


#: Nominal width of one output bar, in minutes. ``None`` = the whole session.
BAR_WIDTH = {"1h": 60, "15m": 15, "4h": None, "5m": 5}


def _bar_end(df: pd.DataFrame, target: str) -> pd.Series:
    """When each bar's window closes, capped at the end of its own session.

    A 15m bar labelled 15:45 closes at 16:00, not at 16:00 plus whatever the
    grid says, because the session ends first. Without the cap the last bar of
    every day would look permanently unfinished.
    """
    label = df["ts"]
    day = label.dt.normalize()
    session_end = day + pd.to_timedelta(
        np.where(df["session"].to_numpy() == "AM", LUNCH_START, POST_END), unit="m")
    width = BAR_WIDTH[target]
    if width is None:
        return session_end
    return pd.concat([label + pd.Timedelta(minutes=width), session_end], axis=1).min(axis=1)


def drop_incomplete_tail(df: pd.DataFrame, target: str, end) -> Tuple[pd.DataFrame, int]:
    """Remove trailing bars whose window had not closed at ``end``.

    Only the tail can be affected: truncating the source at ``end`` leaves every
    earlier bar exactly as it was. Handing back a half-formed bar that looks like
    a finished one is how an as-of reconstruction turns into a lookahead bug in
    the opposite direction - the rule sees a 09:00 bar covering five minutes and
    treats it as fifteen.
    """
    if df.empty or end is None:
        return df, 0
    complete = _bar_end(df, target) <= pd.Timestamp(end)
    keep = len(df)
    while keep > 0 and not bool(complete.iloc[keep - 1]):
        keep -= 1
    return df.iloc[:keep].reset_index(drop=True), len(df) - keep


def resample_session(df: pd.DataFrame) -> pd.DataFrame:
    """1h -> one bar per exchange session (the '4h' timeframe). Two per day."""
    if df.empty:
        return df
    out = _agg(df.groupby(["date", "session"], sort=True)).reset_index()
    return out.sort_values("ts").reset_index(drop=True)[
        ["ts"] + OHLCV + ["date", "session", "n_src"]
    ]


# ---------------------------------------------------------------------------
# public loaders
# ---------------------------------------------------------------------------

def _load_intraday(ticker: str, source: str, target: str, *,
                   drop_zero_volume: bool, keep_session_open: bool,
                   guard: bool, start=None, end=None,
                   complete_only: bool = True) -> pd.DataFrame:
    raw = _read_raw(intraday_path(ticker, source))
    if raw is None:
        return _empty(target, ticker)

    df, rep = _clean(raw, source, ticker, drop_zero_volume=drop_zero_volume,
                     keep_session_open=keep_session_open)
    rep.interval = target
    if df.empty:
        df.attrs["report"] = rep
        return df

    # ``end`` is applied HERE, before anything derived is computed, so that
    # load(end=X) returns exactly the frame you would have had standing at X.
    # Truncating afterwards would let the guard see prices from beyond X.
    # ``start`` is applied last on purpose: the guard recompounds from the first
    # bar, so trimming the head before it would move every subsequent price
    # level and two overlapping windows would disagree.
    if end is not None:
        df = df[df["ts"] <= pd.Timestamp(end)]
        if df.empty:
            df.attrs["report"] = rep
            return df

    # Guard at the SOURCE resolution: a corrupt print has to be fixed before it
    # can leak into a resampled bar's high or low.
    if guard:
        df = _guard(df, rep, INTRA_CAP[source])

    if target == "15m":
        df = resample_15m(df)
    elif target == "4h":
        df = resample_session(df)
    else:
        df = df.reset_index(drop=True)
        df["n_src"] = 1

    if complete_only:
        df, rep.partial_tail = drop_incomplete_tail(df, target, end)

    if start is not None:
        df = df[df["ts"] >= pd.Timestamp(start)]

    df = df[["ts"] + OHLCV + ["date", "session", "n_src"]].reset_index(drop=True)
    rep.final_bars = len(df)
    df.attrs["report"] = rep
    return df


def load_1h(ticker: str, *, drop_zero_volume: bool = True,
            keep_session_open: bool = True, guard: bool = True,
            start=None, end=None, complete_only: bool = True) -> pd.DataFrame:
    """Hourly bars straight from the 1h cache. 7 bars Mon-Thu, 6 on Friday."""
    return _load_intraday(ticker, "1h", "1h", drop_zero_volume=drop_zero_volume,
                          keep_session_open=keep_session_open, guard=guard,
                          start=start, end=end, complete_only=complete_only)


def load_4h(ticker: str, *, drop_zero_volume: bool = True,
            keep_session_open: bool = True, guard: bool = True,
            start=None, end=None, complete_only: bool = True) -> pd.DataFrame:
    """Session bars resampled from 1h. Two per day (morning, afternoon).

    Not a four-hour clock grid - see the module docstring. The IDX day is 5h20m,
    so these average 2h40m.
    """
    return _load_intraday(ticker, "1h", "4h", drop_zero_volume=drop_zero_volume,
                          keep_session_open=keep_session_open, guard=guard,
                          start=start, end=end, complete_only=complete_only)


def load_15m(ticker: str, *, drop_zero_volume: bool = True,
             keep_session_open: bool = True, guard: bool = True,
             start=None, end=None, complete_only: bool = True) -> pd.DataFrame:
    """15-minute bars resampled from 5m. 23 bars Mon-Thu, 19 on Friday.

    Only ~12 weeks of history exists. This is a live-monitoring resolution, not
    a backtesting one.
    """
    return _load_intraday(ticker, "5m", "15m", drop_zero_volume=drop_zero_volume,
                          keep_session_open=keep_session_open, guard=guard,
                          start=start, end=end, complete_only=complete_only)


def load_daily(ticker: str, *, drop_zero_volume: bool = True, guard: bool = True,
               start=None, end=None) -> pd.DataFrame:
    """Daily reference bars, read from the daily cache. Never fetches.

    Raw (not dividend-adjusted) close, capped and recompounded at 35% exactly as
    ``paint_daily.unadjusted_daily`` does, so the daily and intraday layers agree
    on what an impossible print is.
    """
    raw = _read_raw_daily(ticker)
    if raw is None:
        return _empty("1d", ticker)

    rep = LoadReport(ticker=ticker, interval="1d", raw_bars=len(raw))
    out = raw.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values("date", kind="mergesort")

    dup = out["date"].duplicated(keep="last")
    rep.dup_ts = int(dup.sum())
    out = out[~dup]

    close = pd.to_numeric(out["close"], errors="coerce")
    bad = close.isna() | (close <= 0)
    rep.bad_close = int(bad.sum())
    out = out[~bad]
    if out.empty:
        out.attrs["report"] = rep
        return out

    for col in OHLCV:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["volume"] = out["volume"].fillna(0.0)

    zero = out["volume"] <= 0
    if zero.mean() > 0.9:
        rep.volume_unavailable = True
    elif drop_zero_volume:
        rep.zero_volume_dropped = int(zero.sum())
        out = out[~zero]

    out = out.reset_index(drop=True)
    out["ts"] = out["date"]
    out["session"] = "D"
    out["n_src"] = 1

    # Same ordering rule as the intraday loaders: ``end`` before the guard so
    # the frame is a faithful as-of view, ``start`` after so the recompounded
    # level does not depend on the window requested.
    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end)]
    if guard and not out.empty:
        # Every daily step is an overnight step, so both caps are 35%.
        out = _guard(out.reset_index(drop=True), rep, INTRA_CAP["1d"])
    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]

    out = out[["ts"] + OHLCV + ["date", "session", "n_src"]].reset_index(drop=True)
    rep.final_bars = len(out)
    out.attrs["report"] = rep
    return out


def _read_raw_daily(ticker: str) -> Optional[pd.DataFrame]:
    path = daily_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception:
        return None
    if df.empty or "close" not in df.columns:
        return None
    return df


LOADERS = {"1h": load_1h, "4h": load_4h, "15m": load_15m, "1d": load_daily}


def load(ticker: str, interval: str, **kw) -> pd.DataFrame:
    return LOADERS[interval](ticker, **kw)


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def _cache_mtime(interval: str) -> pd.Timestamp:
    """When the newest cache file for this interval was written.

    Not the same question as "what is the last bar". A cache written today can
    still end on a bar from last week, and knowing which of the two is stale
    tells you whether to refetch or to stop trusting the feed.
    """
    if interval == "1d":
        paths = glob.glob(os.path.join(DAILY_DIR, "*.csv.gz"))
    else:
        paths = glob.glob(os.path.join(
            INTRADAY_DIR, f"*_{CACHE_KEY[SOURCE[interval]]}.csv.gz"))
    if not paths:
        return pd.NaT
    # Median, not max: one file refetched yesterday says nothing about the
    # other 777, and the max would quietly claim the whole cache is fresh.
    return pd.Timestamp(float(np.median([os.path.getmtime(p) for p in paths])), unit="s")


def _constant_run(close: np.ndarray) -> int:
    """Longest run of identical closes - a suspended or non-trading stretch."""
    if len(close) < 2:
        return len(close)
    change = np.flatnonzero(np.diff(close) != 0)
    edges = np.concatenate([[-1], change, [len(close) - 1]])
    return int(np.diff(edges).max())


def _round_trips(close: np.ndarray, lookback: int) -> int:
    """Round trips from a strictly causal Donchian channel rule.

    Long when the close exceeds the highest of the PRIOR ``lookback`` closes,
    flat when it falls below the lowest. Every comparison uses a window that
    ends on the previous bar, so no bar can see itself.
    """
    n = len(close)
    if n <= lookback + 2:
        return 0
    s = pd.Series(close)
    hi = s.rolling(lookback).max().shift(1).to_numpy()
    lo = s.rolling(lookback).min().shift(1).to_numpy()
    trips = 0
    in_pos = False
    for i in range(lookback + 1, n):
        if not in_pos and np.isfinite(hi[i]) and close[i] > hi[i]:
            in_pos = True
        elif in_pos and np.isfinite(lo[i]) and close[i] < lo[i]:
            in_pos = False
            trips += 1
    return trips


def calendar_gaps(dates: List[pd.Timestamp], min_days: int = 5) -> List[Tuple[str, str, int]]:
    """Holes in the union session calendar, in calendar days.

    A rule that measures "n bars ago" as "n periods of time ago" is wrong across
    these. Some are IDX holidays (Lebaran closes the exchange for over a week);
    at least one is not.
    """
    s = pd.Series(sorted(set(dates)))
    if len(s) < 2:
        return []
    d = s.diff().dt.days
    return [(f"{s.iloc[i - 1]:%Y-%m-%d}", f"{s.iloc[i]:%Y-%m-%d}", int(d.iloc[i]))
            for i in np.flatnonzero(d.to_numpy() >= min_days)]


def sessions_missing_from_intraday(intraday_dates, reference: List[str],
                                   start: pd.Timestamp) -> List[pd.Timestamp]:
    """Sessions the DAILY cache has that the intraday cache does not.

    This is the check that catches a feed hole masquerading as a holiday. A gap
    in the intraday calendar is only benign if the exchange was shut; the daily
    cache is the independent witness for that.
    """
    have = set(pd.Timestamp(d) for d in intraday_dates)
    if not have:
        return []
    # Only the interior matters. Sessions after the last intraday bar are
    # staleness, which is a different problem reported separately.
    last = max(have)
    ref: set = set()
    for t in reference:
        d = load_daily(t, guard=False)
        if not d.empty:
            ref.update(x for x in d["date"] if start <= x <= last)
    return sorted(ref - have)


def close_agreement(interval: str, tickers: List[str]) -> Tuple[float, int, int]:
    """Share of sessions where the last intraday close equals the daily close.

    Two layers that disagree about the closing price will disagree about every
    trade that exits on the close, and nothing in either layer complains.
    """
    shares, sessions = [], 0
    for t in tickers:
        d = load(t, interval, guard=False)
        day = load_daily(t, guard=False, drop_zero_volume=False)
        if d.empty or day.empty:
            continue
        last = d.groupby("date")["close"].last()
        j = pd.concat([last.rename("i"), day.set_index("date")["close"].rename("d")],
                      axis=1, sort=True).dropna()
        if len(j) < 50:
            continue
        shares.append(float(np.isclose(j["i"], j["d"], rtol=1e-6).mean()))
        sessions += len(j)
    if not shares:
        return float("nan"), 0, 0
    return float(np.median(shares)), len(shares), sessions


def audit(intervals: List[str], limit: Optional[int], asof: pd.Timestamp,
          probe: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, list]]:
    rows, per_name = [], []
    gaps: Dict[str, list] = {}
    holes: Dict[str, list] = {}
    agree: Dict[str, tuple] = {}
    # Liquid names that trade every session, used as the reference calendar.
    REF = ["BBCA", "BBRI", "BMRI", "ASII", "TLKM"]
    # Donchian lookback per interval, chosen so the rule holds a swing of
    # roughly comparable calendar length at each resolution.
    donchian = {"1h": 20, "4h": 10, "15m": 20, "1d": 20}

    for interval in intervals:
        names = available(interval)
        if limit:
            names = names[:limit]
        recs = []
        seen_dates: set = set()
        for i, t in enumerate(names):
            df = load(t, interval)
            rep = df.attrs.get("report", LoadReport())
            if df.empty:
                recs.append(dict(ticker=t, bars=0))
                continue
            seen_dates.update(df["date"].unique())
            close = df["close"].to_numpy(float)
            days = df["date"].nunique()
            recs.append(dict(
                ticker=t, interval=interval, bars=len(df), sessions=days,
                bars_per_day=len(df) / max(days, 1),
                first=df["ts"].iloc[0], last=df["ts"].iloc[-1],
                dup_ts=rep.dup_ts, bad_close=rep.bad_close,
                zero_vol=rep.zero_volume_dropped,
                zero_vol_open_kept=rep.zero_volume_kept_open,
                out_of_session=rep.out_of_session,
                clipped_intra=rep.clipped_intra,
                clipped_overnight=rep.clipped_overnight,
                max_step_raw=rep.max_abs_step_raw,
                n_suspect=len(rep.suspect_jumps),
                vol_unavailable=rep.volume_unavailable,
                const_run=_constant_run(close),
                med_close=float(np.median(close)),
                trips=_round_trips(close, donchian[interval]) if i < probe else np.nan,
            ))
        R = pd.DataFrame(recs)
        R["interval"] = interval
        per_name.append(R)
        dates = [pd.Timestamp(d) for d in seen_dates]
        gaps[interval] = calendar_gaps(dates)
        holes[interval] = (sessions_missing_from_intraday(dates, REF, min(dates))
                           if dates and interval != "1d" else [])
        agree[interval] = (close_agreement(interval, [n for n in names[:probe]
                                                      if is_idx_equity(n)])
                           if interval != "1d" else (1.0, 0, 0))

        live = R[R["bars"] > 0]
        usable = live[live["bars"] >= _MIN_BARS[interval]]
        rows.append(dict(
            interval=interval,
            files=len(R),
            with_data=len(live),
            usable=len(usable),
            median_bars=int(live["bars"].median()) if len(live) else 0,
            p10_bars=int(live["bars"].quantile(0.10)) if len(live) else 0,
            p90_bars=int(live["bars"].quantile(0.90)) if len(live) else 0,
            median_sessions=int(live["sessions"].median()) if len(live) else 0,
            bars_per_day=round(float(live["bars_per_day"].median()), 2) if len(live) else 0,
            first=live["first"].min() if len(live) else pd.NaT,
            last=live["last"].median() if len(live) else pd.NaT,
            freshest=live["last"].max() if len(live) else pd.NaT,
            stale_days=int((asof - live["last"].median()).days) if len(live) else -1,
            median_trips=float(live["trips"].median()) if live["trips"].notna().any() else float("nan"),
        ))

    P = pd.concat(per_name, ignore_index=True)
    return pd.DataFrame(rows), P, {"gaps": gaps, "holes": holes, "agree": agree}


# Minimum clean bars for a name to count as usable at each interval. Set so a
# 20-bar Donchian rule has room to produce a double-digit number of round trips:
# roughly 25 bars of history per round trip at the observed trade rates.
_MIN_BARS = {"1h": 500, "4h": 150, "15m": 500, "1d": 250}


def selftest(tickers: List[str]) -> int:
    """Structural and no-cheating checks. Every one of these must be exact.

    The prefix check is the important one. Painting the first half of a series
    with only the first half must reproduce, bar for bar, what the full series
    gives for that half. If it does not, something in the pipeline is reading
    forward, and every backtest built on it is worthless.
    """
    expect_bpd = {"1h": {6, 7}, "4h": {2}, "15m": {19, 23}}
    LIQUID = {"BBCA", "ADRO", "_JKSE", "BBRI", "BMRI", "TLKM", "ASII"}
    fails = 0

    def check(ok: bool, label: str) -> None:
        nonlocal fails
        if not ok:
            fails += 1
        print(f"   {'ok  ' if ok else 'FAIL'}  {label}")

    for t in tickers:
        print(f"\n [{t}]")
        for interval in ("1h", "4h", "15m"):
            df = load(t, interval)
            if df.empty:
                print(f"   skip  {interval}: not cached")
                continue

            check(df["ts"].is_monotonic_increasing, f"{interval}: timestamps sorted")
            check(not df["ts"].duplicated().any(), f"{interval}: no duplicate timestamps")
            check(bool((df["high"] >= df[["open", "close"]].max(axis=1) - 1e-6).all()
                       and (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-6).all()),
                  f"{interval}: OHLC coherent")

            # No bar may be built across lunch or across midnight.
            minute = df["ts"].dt.hour * 60 + df["ts"].dt.minute
            check(bool(((minute >= SESSION_OPEN) & (minute <= POST_END)).all()),
                  f"{interval}: every bar inside 09:00-16:15")
            check(bool((~((minute >= LUNCH_START) & (minute < LUNCH_END - 30))).all()
                       if interval != "1h" else True),
                  f"{interval}: no bar labelled inside the lunch break")
            spans = df.groupby(["date", "session"])["ts"].count()
            check(len(spans) > 0, f"{interval}: bars group cleanly by (date, session)")

            # A liquid name must hit the documented count exactly. An illiquid
            # one may have fewer - whole hours pass with no trade - but never
            # more, because more would mean the layer invented a bar.
            per_day = df.groupby("date").size()
            bpd = set(per_day.mode().tolist())
            cap = max(expect_bpd[interval])
            check(int(per_day.max()) <= cap,
                  f"{interval}: max bars/day {int(per_day.max())} <= {cap} (none invented)")
            if t in LIQUID:
                check(bool(bpd & expect_bpd[interval]),
                      f"{interval}: modal bars/day {sorted(bpd)} in "
                      f"{sorted(expect_bpd[interval])}")
            else:
                print(f"   note  {interval}: modal bars/day {sorted(bpd)} "
                      f"(illiquid name, sparse by nature)")

            # Prefix identity. ``end=cut`` re-runs the whole pipeline on data
            # truncated at cut, so this really is "what would I have had then",
            # not a slice of the finished frame. The bar labelled cut is by
            # definition still open at cut, so exactly one bar may be withheld.
            for cut in (df["ts"].iloc[len(df) // 2], df["ts"].iloc[max(1, len(df) - 40)]):
                part = load(t, interval, end=cut)
                full = df[df["ts"] <= cut].reset_index(drop=True)
                dropped = len(full) - len(part)
                same = (0 <= dropped <= 1 and len(part) > 0 and all(
                    np.allclose(part[c].to_numpy(float),
                                full[c].to_numpy(float)[:len(part)], rtol=0, atol=1e-9)
                    for c in OHLCV))
                check(same, f"{interval}: prefix identical rebuilding from scratch at "
                            f"{cut:%Y-%m-%d %H:%M} ({len(part)} bars, "
                            f"{dropped} open bar withheld)")

    print(f"\n {'ALL CHECKS PASSED' if not fails else str(fails) + ' CHECKS FAILED'}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true",
                    help="run the structural and no-cheating checks and exit")
    ap.add_argument("--names", type=int, default=0, help="cap names per interval (0 = all)")
    ap.add_argument("--intervals", default="1h,4h,15m,1d")
    ap.add_argument("--asof", default="2026-08-19")
    ap.add_argument("--probe", type=int, default=60,
                    help="how many names to run the Donchian round-trip probe on")
    ap.add_argument("--csv", default="reports/tf_data_coverage.csv")
    args = ap.parse_args()

    if args.selftest:
        print("=" * 108 + "\n STRUCTURE AND NO-CHEATING CHECKS\n" + "=" * 108)
        return selftest(["BBCA", "ADRO", "CUAN", "BTEK", "_JKSE"])

    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]
    asof = pd.Timestamp(args.asof)
    C, P, CAL = audit(intervals, args.names or None, asof, probe=args.probe)
    GAPS, HOLES, AGREE = CAL["gaps"], CAL["holes"], CAL["agree"]

    os.makedirs(os.path.join(REPO_ROOT, "reports"), exist_ok=True)
    out_csv = args.csv if os.path.isabs(args.csv) else os.path.join(REPO_ROOT, args.csv)
    C.to_csv(out_csv, index=False)
    P.to_csv(out_csv.replace(".csv", "_per_name.csv"), index=False)

    line = "=" * 108
    print(f"\n{line}\n MULTI-TIMEFRAME COVERAGE  (cache only, as of {asof:%Y-%m-%d})\n{line}")
    print(f" {'tf':>4}{'files':>7}{'usable':>8}{'med bars':>10}{'p10':>8}{'p90':>8}"
          f"{'sess':>7}{'b/day':>7}  {'first':<12}{'last':<12}{'stale':>7}{'trips':>7}")
    for _, r in C.iterrows():
        print(f" {r['interval']:>4}{r['files']:>7}{r['usable']:>8}{r['median_bars']:>10,}"
              f"{r['p10_bars']:>8,}{r['p90_bars']:>8,}{r['median_sessions']:>7,}"
              f"{r['bars_per_day']:>7.2f}  {pd.Timestamp(r['first']):%Y-%m-%d}  "
              f"{pd.Timestamp(r['last']):%Y-%m-%d}{r['stale_days']:>7}"
              f"{r['median_trips']:>7.0f}")
    print("\n first = earliest bar of any name; last = MEDIAN name's last bar; "
          "stale = days from that to as-of")
    print("\n usable = has a cache file and at least "
          + ", ".join(f"{_MIN_BARS[i]} {i} bars" for i in intervals if i in _MIN_BARS))
    print(" trips  = median round trips from a causal 20/10-bar Donchian rule on the "
          f"first {args.probe} names")

    print(f"\n{line}\n WHAT WOULD SILENTLY CORRUPT A BACKTEST\n{line}")
    for interval in intervals:
        S = P[(P["interval"] == interval) & (P["bars"] > 0)]
        if S.empty:
            continue
        short = S[S["bars"] < _MIN_BARS[interval]]
        const = S[S["const_run"] >= 30]
        cheap = S[S["med_close"] < 50]
        cohort = S["last"].median()
        stale = S[S["last"] < cohort - pd.Timedelta(days=3)]
        print(f"\n [{interval}] {len(S)} names with data")
        print(f"   duplicated timestamps ............ {int(S['dup_ts'].sum()):,} bars "
              f"in {int((S['dup_ts'] > 0).sum())} names")
        print(f"   NaN / non-positive close ......... {int(S['bad_close'].sum()):,} bars")
        print(f"   out-of-session bars dropped ...... {int(S['out_of_session'].sum()):,} bars")
        print(f"   zero-volume bars dropped ......... {int(S['zero_vol'].sum()):,} bars")
        print(f"   zero-volume session opens KEPT ... {int(S['zero_vol_open_kept'].sum()):,} bars "
              f"(would be lost by a naive volume>0 filter)")
        print(f"   clipped in-session steps ......... {int(S['clipped_intra'].sum()):,}")
        print(f"   clipped overnight steps .......... {int(S['clipped_overnight'].sum()):,} "
              f"in {int((S['clipped_overnight'] > 0).sum())} names "
              f"(corporate actions - NOT fixed, only bounded)")
        print(f"   history far short of the window .. {len(short)} names below "
              f"{_MIN_BARS[interval]} bars")
        print(f"   constant-price runs >= 30 bars ... {len(const)} names "
              f"(max run {int(S['const_run'].max()):,})")
        print(f"   median close under Rp50 .......... {len(cheap)} names "
              f"(one tick = 2%-100%; the clip is wrong for these)")
        print(f"   last bar >3d behind the cohort ... {len(stale)} names "
              f"(cohort last bar {pd.Timestamp(cohort):%Y-%m-%d})")
        print(f"   cache files last written ......... "
              f"{_cache_mtime(interval):%Y-%m-%d %H:%M} (median file mtime)")
        print(f"   staleness ........................ cohort last bar "
              f"{pd.Timestamp(cohort):%Y-%m-%d}, freshest "
              f"{pd.Timestamp(S['last'].max()):%Y-%m-%d} -> "
              f"{int((asof - cohort).days)} days stale at {asof:%Y-%m-%d}")
        share, nn, ns = AGREE.get(interval, (float("nan"), 0, 0))
        if nn:
            print(f"   last close == daily close ........ {share:.1%} of sessions "
                  f"(median over {nn} names, {ns:,} sessions)")
        non_idx = S[~S["ticker"].map(is_idx_equity)]
        if len(non_idx):
            print(f"   NOT IDX equities ................. {len(non_idx)} "
                  f"({', '.join(sorted(non_idx['ticker'])[:8])}"
                  f"{' ...' if len(non_idx) > 8 else ''})")
        g = GAPS.get(interval, [])
        print(f"   gaps in the session calendar ..... {len(g)} of >=5 calendar days")
        for a, b, n in g:
            print(f"       {a} -> {b}   ({n} days)")
        h = HOLES.get(interval, [])
        if h:
            print(f"   ** {len(h)} sessions MISSING from inside this interval's range "
                  f"that the daily cache has **")
            print("       " + ", ".join(f"{pd.Timestamp(x):%Y-%m-%d}" for x in h[:12])
                  + (" ..." if len(h) > 12 else ""))
            print("       the exchange traded on these days. A feed hole, not a "
                  "holiday - and invisible unless you check.")
        else:
            print("   interior sessions vs daily cache .. none missing")

    print(f"\n -> {out_csv}")
    print(f" -> {out_csv.replace('.csv', '_per_name.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
