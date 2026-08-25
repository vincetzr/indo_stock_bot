"""Causal technical indicators — EMA, ATR, stochastic, RSI, volume z.

WHAT THIS IS FOR
-----------------
H17 built exit rules that see only the normalised price path: trail this far
from the peak, stop that far below entry. Those are the right first rules
because they have no parameters to fit beyond a distance. But they are also
blind in a specific way: **a 15% give-back means something completely
different on a name whose daily move is 2% than on one whose daily move is
8%**, and the multiplier-cell entry selects for exactly the second kind. An
indicator layer is the cheapest way to condition the exit on how the name is
actually behaving rather than on a fixed percentage.

EVERY FUNCTION HERE IS STRICTLY CAUSAL. The value at bar *i* uses bars ``<= i``
and nothing after. This is asserted in `tests/test_signals.py` by recomputing
each indicator on truncated prefixes and requiring the answer at *i* to be
identical — the only test that actually proves the property, as opposed to
inspecting the code and believing it.

WHERE THE HIGH AND LOW COME FROM, AND WHY IT NEEDED CHECKING
-------------------------------------------------------------
`data/spine/price_panel.parquet` carries close, adj_close and volume — no high
or low — so a first reading says ATR and stochastic are not computable and only
close-only variants are available. That reading is wrong. `data/cache/ohlcv/`
holds the full OHLCV the Yahoo provider always fetched, for **all 919** panel
names, and on a 25-name sample the cached close matches the panel's close on
**100%** of overlapping bars while high and low bracket it on **99.999%**.

The bars are RAW, though, and the panel's ``adj_close`` is adjusted and in
three cases (SCCO, PYFA, SINI) repaired. So high and low are rebased with the
panel's own factor,

    factor = adj_close / close   ->   adj_high = high * factor

which makes the adjusted high/low inherit the panel's repaired basis rather
than the vendor's. Volume sidesteps the question entirely by being used only as
rupiah turnover ``volume * close``, which is value traded and needs no
adjustment at all.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def ema(x, n: int) -> np.ndarray:
    """Exponential moving average, ``alpha = 2/(n+1)``, seeded on the first bar.

    ``adjust=False`` so the value at bar *i* is the recursive one a live
    implementation would hold, not the infinite-window expansion pandas
    defaults to — those differ early in the series and the difference is
    exactly the kind that looks like alpha.
    """
    s = pd.Series(np.asarray(x, dtype=float))
    return s.ewm(span=n, adjust=False, min_periods=1).mean().to_numpy()


def wilder(x, n: int) -> np.ndarray:
    """Wilder's smoothing, ``alpha = 1/n``. What ATR and RSI actually use."""
    s = pd.Series(np.asarray(x, dtype=float))
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=1).mean().to_numpy()


def true_range(high, low, close) -> np.ndarray:
    """``max(h-l, |h-c_prev|, |l-c_prev|)``; the first bar has no prior close."""
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    prev = np.concatenate([[np.nan], c[:-1]])
    a = h - l
    b = np.abs(h - prev)
    d = np.abs(l - prev)
    with np.errstate(invalid="ignore"):
        # an all-NaN column is a bar with no usable high/low — the repaired
        # names carry a few by design — and NaN is the correct answer for it
        stack = np.vstack([a, b, d])
        tr = np.where(np.isnan(stack).all(axis=0), np.nan,
                      np.nanmax(np.where(np.isnan(stack), -np.inf, stack),
                                axis=0))
    tr[0] = a[0]
    return tr


def atr(high, low, close, n: int = 22) -> np.ndarray:
    """Average true range. The unit the chandelier exit measures give-back in."""
    return wilder(true_range(high, low, close), n)


def stochastic(high, low, close, k: int = 14, d: int = 3, smooth: int = 3
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Slow stochastic: returns ``(%K, %D)``, both 0-100.

    ``%K_raw = 100 * (close - min(low, k)) / (max(high, k) - min(low, k))``,
    then %K is that smoothed over ``smooth`` bars and %D is %K over ``d``.
    A flat window (high == low for k bars, which happens on suspended or
    limit-locked names) has no defined position in its range and returns NaN
    rather than an arbitrary 50.
    """
    h = pd.Series(np.asarray(high, dtype=float))
    l = pd.Series(np.asarray(low, dtype=float))
    c = pd.Series(np.asarray(close, dtype=float))
    hh = h.rolling(k, min_periods=k).max()
    ll = l.rolling(k, min_periods=k).min()
    rng = (hh - ll).replace(0.0, np.nan)
    raw = 100.0 * (c - ll) / rng
    kk = raw.rolling(smooth, min_periods=smooth).mean()
    dd = kk.rolling(d, min_periods=d).mean()
    return (kk.to_numpy(), dd.to_numpy())


def rsi(close, n: int = 14) -> np.ndarray:
    """Wilder's RSI, 0-100."""
    c = np.asarray(close, dtype=float)
    ch = np.diff(c, prepend=c[0])
    up = wilder(np.clip(ch, 0, None), n)
    dn = wilder(np.clip(-ch, 0, None), n)
    out = np.full(len(c), np.nan)
    tot = up + dn
    ok = tot > 0
    out[ok] = 100.0 * up[ok] / tot[ok]
    out[(~ok) & np.isfinite(tot)] = 50.0        # no movement either way
    out[:n] = np.nan                            # not yet estimated
    return out


def turnover_z(volume, close, n: int = 20) -> np.ndarray:
    """Z-score of log rupiah turnover over a trailing ``n`` bars, inclusive.

    Rupiah turnover rather than share volume because value traded is invariant
    to splits, so a 20-bar window spanning a corporate action does not need an
    adjustment factor at all. Inclusive of bar *i*: you know today's volume at
    today's close, and the whole point is to detect that today is unusual.
    """
    v = np.asarray(volume, dtype=float) * np.asarray(close, dtype=float)
    s = pd.Series(np.where(v > 0, np.log(np.maximum(v, 1e-12)), np.nan))
    m = s.rolling(n, min_periods=max(5, n // 2)).mean()
    sd = s.rolling(n, min_periods=max(5, n // 2)).std(ddof=0).replace(0.0, np.nan)
    return ((s - m) / sd).to_numpy()


#: Every column ``build`` produces. The exit rules index this by name.
COLUMNS = ["close", "adj_high", "adj_low", "ema10", "ema20", "ema30",
           "ema50", "atr22", "stoch_k", "stoch_d", "rsi14", "tvz20"]


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Indicator columns for ONE ticker, in date order.

    ``df`` needs ``high``, ``low``, ``close`` (raw, from the OHLCV cache),
    ``adj_close`` and ``volume`` (from the panel). Everything price-shaped is
    returned on the ADJUSTED basis so it is directly comparable with the
    normalised path the exit rules walk.
    """
    d = df.sort_values("date")
    close = d["close"].to_numpy(dtype=float)
    adj = d["adj_close"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(close > 0, adj / close, np.nan)
    hi = d["high"].to_numpy(dtype=float) * f
    lo = d["low"].to_numpy(dtype=float) * f
    k, dd = stochastic(hi, lo, adj)
    out = pd.DataFrame({
        "date": d["date"].to_numpy(),
        "ticker": d["ticker"].to_numpy() if "ticker" in d else None,
        "close": adj,                       # ADJUSTED close, the rules' scale
        "adj_high": hi, "adj_low": lo,
        "ema10": ema(adj, 10), "ema20": ema(adj, 20),
        "ema30": ema(adj, 30), "ema50": ema(adj, 50),
        "atr22": atr(hi, lo, adj, 22),
        "stoch_k": k, "stoch_d": dd,
        "rsi14": rsi(adj, 14),
        "tvz20": turnover_z(d["volume"].to_numpy(dtype=float), close, 20),
    })
    if "ticker" not in d:
        out = out.drop(columns=["ticker"])
    return out
