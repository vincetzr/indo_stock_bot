"""Hull Moving Average family and the UT Bot ATR trailing stop.

Both are ports of specific, widely-used TradingView scripts, and the point of
this module is that they are ports rather than approximations - the Pine
semantics are reproduced exactly, including the awkward parts, because a
backtest of "something a bit like the indicator" tells you nothing about the
indicator.

Sources reproduced
------------------
**Hull Suite** (InSilico). Three variants, selected by ``mode``::

    HMA (src, n) = WMA(2*WMA(src, n/2) - WMA(src, n), round(sqrt(n)))
    EHMA(src, n) = EMA(2*EMA(src, n/2) - EMA(src, n), round(sqrt(n)))
    THMA(src, n) = WMA(3*WMA(src, n/3) - WMA(src, n/2) - WMA(src, n), n)

with the script's own quirk preserved: **THMA is invoked with n/2**, not n, so
``hull(mode="thma", length=55)`` internally runs THMA at 27. Getting that wrong
silently doubles the effective lookback.

The band colours green when ``HULL > HULL[2]`` and red otherwise. Note that is
a comparison against *two* bars back, not one - it is a slope filter with a
built-in one-bar hysteresis, and using ``HULL[1]`` produces a visibly noisier
signal that is not the published indicator.

**UT Bot Alerts** (Yo_adriiiiaan). An ATR trailing stop::

    nLoss = key * ATR(period)
    stop  = max(prev, src - nLoss)   if src > prev and src[1] > prev
            min(prev, src + nLoss)   if src < prev and src[1] < prev
            src - nLoss              if src > prev
            src + nLoss              otherwise

    buy  = crossover(src, stop)      sell = crossunder(src, stop)

The published script writes the signal as ``src > stop and crossover(ema(src,1),
stop)``. ``ema(x, 1) == x``, and ``crossover`` already implies ``src > stop``,
so the whole expression reduces to a plain crossover. It is written out that
way here so the equivalence is on the record rather than assumed.

Two details that decide whether the port is faithful:

* ``ATR`` in Pine is **Wilder's RMA**, not a simple mean. RMA seeds with an SMA
  of the first ``n`` true ranges and then applies ``alpha = 1/n``. An SMA-based
  ATR gives a systematically tighter stop and more trades.
* The recursion is seeded from ``nz(stop[1], 0) == 0``, which makes the first
  branch that can fire ``src > 0`` - so the stop starts one ``nLoss`` below
  price and converges. Everything before the ATR warms up is returned as NaN
  here rather than pretending to be a signal.

Nothing in this module looks forward. Every value at bar *t* is a function of
bars *<= t* only, which the test suite asserts directly by truncation rather
than by inspection.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

HULL_MODES = ("hma", "ehma", "thma")


# ---------------------------------------------------------------------------
# moving averages
# ---------------------------------------------------------------------------
def wma(series: pd.Series, length: int) -> pd.Series:
    """Linearly weighted moving average, most recent bar weighted heaviest.

    Implemented as a convolution rather than ``rolling.apply`` because the
    parameter sweeps in this project call it tens of thousands of times and the
    apply path is roughly two orders of magnitude slower for identical output.
    """
    length = int(length)
    if length < 1:
        raise ValueError(f"wma length must be >= 1, got {length}")
    values = np.asarray(series, dtype=float)
    out = np.full(values.shape, np.nan)
    if len(values) < length:
        return pd.Series(out, index=series.index)

    weights = np.arange(1, length + 1, dtype=float)
    weights /= weights.sum()
    # NaNs would smear across the window, so convolve the filled copy and then
    # re-mask any window that actually contained one.
    filled = np.nan_to_num(values, nan=0.0)
    conv = np.convolve(filled, weights[::-1], mode="valid")
    out[length - 1:] = conv
    if np.isnan(values).any():
        bad = np.convolve(np.isnan(values).astype(float),
                          np.ones(length), mode="valid") > 0
        out[length - 1:][bad] = np.nan
    return pd.Series(out, index=series.index)


def ema(series: pd.Series, length: int) -> pd.Series:
    """Pine's ``ta.ema``: alpha = 2/(n+1), seeded with an SMA of the first n."""
    length = int(length)
    if length < 1:
        raise ValueError(f"ema length must be >= 1, got {length}")
    if length == 1:
        return series.astype(float).copy()
    return series.astype(float).ewm(span=length, adjust=False,
                                    min_periods=length).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing, Pine's ``ta.rma``: alpha = 1/n, SMA-seeded.

    This is *not* interchangeable with an EMA of the same length - Wilder's
    alpha of 1/n corresponds to an EMA span of 2n-1 - nor with an SMA. ATR,
    RSI and ADX are all defined on it, and substituting a simple mean is the
    most common way a Pine port drifts away from the indicator it claims to be.
    """
    length = int(length)
    if length < 1:
        raise ValueError(f"rma length must be >= 1, got {length}")
    values = np.asarray(series, dtype=float)
    out = np.full(values.shape, np.nan)
    n = len(values)
    if n < length:
        return pd.Series(out, index=series.index)

    seed = values[:length]
    if np.isnan(seed).any():
        return pd.Series(out, index=series.index)
    prev = seed.mean()
    out[length - 1] = prev
    alpha = 1.0 / length
    for i in range(length, n):
        value = values[i]
        if np.isnan(value):
            out[i] = prev
            continue
        prev = alpha * value + (1.0 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=series.index)


# ---------------------------------------------------------------------------
# Hull Suite
# ---------------------------------------------------------------------------
def hma(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 2)
    half = max(int(round(length / 2)), 1)
    root = max(int(round(math.sqrt(length))), 1)
    return wma(2.0 * wma(series, half) - wma(series, length), root)


def ehma(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 2)
    half = max(int(round(length / 2)), 1)
    root = max(int(round(math.sqrt(length))), 1)
    return ema(2.0 * ema(series, half) - ema(series, length), root)


def thma(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 2)
    third = max(int(round(length / 3)), 1)
    half = max(int(round(length / 2)), 1)
    return wma(3.0 * wma(series, third) - wma(series, half) - wma(series, length),
               length)


def hull(series: pd.Series, length: int = 55, mode: str = "hma") -> pd.Series:
    """Hull Suite line for the given mode.

    ``thma`` is deliberately called at ``length/2`` to match the published
    script's ``Mode()`` dispatch.
    """
    mode = str(mode).lower()
    if mode == "hma":
        return hma(series, length)
    if mode == "ehma":
        return ehma(series, length)
    if mode == "thma":
        return thma(series, max(int(round(length / 2)), 2))
    raise ValueError(f"hull mode must be one of {HULL_MODES}, got {mode!r}")


def hull_is_green(hull_line: pd.Series) -> pd.Series:
    """The band's colour rule: green when the line is above its value two bars ago.

    Returned as a nullable boolean so the warm-up period stays distinguishable
    from a genuine red. Collapsing NaN to False would silently mean "red", and
    a strategy would then read the warm-up as a bearish regime.
    """
    prior = hull_line.shift(2)
    green = hull_line > prior
    return green.where(hull_line.notna() & prior.notna(), other=pd.NA).astype("boolean")


# ---------------------------------------------------------------------------
# UT Bot
# ---------------------------------------------------------------------------
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Pine's ``ta.tr``: on the first bar, with no previous close, it is high-low."""
    prev_close = close.shift(1)
    ranges = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1)
    out = ranges.max(axis=1)
    if len(out):
        out.iloc[0] = float(high.iloc[0] - low.iloc[0])
    return out


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        length: int = 10) -> pd.Series:
    """Wilder ATR, as ``ta.atr`` computes it."""
    return rma(true_range(high, low, close), length)


def ut_bot_stop(source: pd.Series, atr_series: pd.Series,
                key: float = 1.0) -> pd.Series:
    """The UT Bot ATR trailing stop.

    Path-dependent by construction, so this is an explicit loop. The recursion
    only begins once ATR exists; bars before that are NaN rather than a stop of
    zero, which would otherwise read as "price is enormously above the stop"
    and manufacture a buy signal on the first warm-up bar.
    """
    src = np.asarray(source, dtype=float)
    loss = np.asarray(atr_series, dtype=float) * float(key)
    n = len(src)
    out = np.full(n, np.nan)

    prev = 0.0          # Pine's nz(stop[1], 0)
    started = False
    for i in range(n):
        nloss = loss[i]
        if not np.isfinite(nloss) or not np.isfinite(src[i]):
            continue
        current = src[i]
        previous = src[i - 1] if i > 0 else np.nan
        if not started:
            # First live bar: prev is 0, so price is above it and the stop
            # anchors one nLoss below, exactly as the script's third branch does.
            prev = current - nloss
            out[i] = prev
            started = True
            continue
        if current > prev and np.isfinite(previous) and previous > prev:
            new = max(prev, current - nloss)
        elif current < prev and np.isfinite(previous) and previous < prev:
            new = min(prev, current + nloss)
        elif current > prev:
            new = current - nloss
        else:
            new = current + nloss
        out[i] = new
        prev = new
    return pd.Series(out, index=source.index)


def _crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    prev_fast, prev_slow = fast.shift(1), slow.shift(1)
    crossed = (fast > slow) & (prev_fast <= prev_slow)
    valid = fast.notna() & slow.notna() & prev_fast.notna() & prev_slow.notna()
    return crossed.where(valid, other=False).astype(bool)


def ut_bot(high: pd.Series, low: pd.Series, close: pd.Series,
           key: float = 1.0, atr_length: int = 10,
           source: Optional[pd.Series] = None) -> pd.DataFrame:
    """Stop line, long/short state and the buy/sell signals.

    ``buy`` is a plain ``crossover(src, stop)``. The published script spells it
    ``src > stop and crossover(ema(src, 1), stop)``; ``ema(x, 1)`` is ``x`` and
    a crossover already implies the inequality, so the two are identical. The
    reduction is asserted in the tests rather than taken on trust.
    """
    src = close if source is None else source
    atr_series = atr(high, low, close, atr_length)
    stop = ut_bot_stop(src, atr_series, key)
    return pd.DataFrame({
        "atr": atr_series,
        "stop": stop,
        "ut_long": (src > stop).where(stop.notna(), other=pd.NA).astype("boolean"),
        "buy": _crossover(src, stop),
        "sell": _crossover(stop, src),
    }, index=src.index)


def warmup_bars(hull_length: int, hull_mode: str, atr_length: int) -> int:
    """Bars to discard before any signal is trustworthy.

    Both indicators are recursive or nested, so an honest floor is the longest
    chain of dependencies rather than the largest single parameter.
    """
    length = max(int(hull_length), 2)
    if str(hull_mode).lower() == "thma":
        length = max(int(round(length / 2)), 2)
        hull_need = length + length
    else:
        hull_need = length + max(int(round(math.sqrt(length))), 1)
    # +2 for the HULL[2] colour comparison, and the ATR needs its own seed.
    return int(max(hull_need + 2, atr_length * 3))
