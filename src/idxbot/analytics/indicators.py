"""Price/volume indicators used by the accumulation engine.

All functions take and return pandas Series aligned to the OHLCV frame's index,
and none of them look ahead: value at row *i* uses only rows <= *i*. That
property is what makes the backtester's results meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR."""
    return true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return atr(df, period) / df["close"].replace(0, np.nan)


def obv(df: pd.DataFrame) -> pd.Series:
    """On-balance volume: cumulative signed volume."""
    direction = np.sign(df["close"].diff().fillna(0.0))
    return (direction * df["volume"]).cumsum()


def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
    """Chaikin A/D line: volume weighted by where the close sits in the bar."""
    span = (df["high"] - df["low"]).replace(0, np.nan)
    multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / span
    return (multiplier.fillna(0.0) * df["volume"]).cumsum()


def chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.Series:
    span = (df["high"] - df["low"]).replace(0, np.nan)
    multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / span
    mfv = (multiplier.fillna(0.0) * df["volume"]).rolling(period).sum()
    return mfv / df["volume"].rolling(period).sum().replace(0, np.nan)


def volume_price_trend(df: pd.DataFrame) -> pd.Series:
    return (df["close"].pct_change().fillna(0.0) * df["volume"]).cumsum()


def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    raw_flow = typical * df["volume"]
    up = raw_flow.where(typical.diff() > 0, 0.0).rolling(period).sum()
    down = raw_flow.where(typical.diff() < 0, 0.0).rolling(period).sum()
    ratio = up / down.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + ratio))


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 3)).mean()
    std = series.rolling(window, min_periods=max(5, window // 3)).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Least-squares slope over a rolling window, normalised by the mean level.

    Expressed per-bar as a fraction so it is comparable across price scales.
    """
    x = np.arange(window, dtype=float)
    x_centred = x - x.mean()
    denominator = (x_centred ** 2).sum()

    def _slope(values: np.ndarray) -> float:
        y = np.asarray(values, dtype=float)
        if np.isnan(y).any():
            return np.nan
        level = y.mean()
        if level == 0:
            return np.nan
        return float((x_centred * (y - level)).sum() / denominator / level)

    return series.rolling(window).apply(_slope, raw=True)


def range_position(df: pd.DataFrame, window: int) -> pd.Series:
    """Where the close sits inside its trailing range: 0 at the low, 1 at the high."""
    low = df["low"].rolling(window, min_periods=max(5, window // 4)).min()
    high = df["high"].rolling(window, min_periods=max(5, window // 4)).max()
    span = (high - low).replace(0, np.nan)
    return ((df["close"] - low) / span).clip(0.0, 1.0)


def range_compression(df: pd.DataFrame, window: int = 20, baseline: int = 120) -> pd.Series:
    """Ratio of recent average true range to its longer-run norm.

    Values well below 1 mean the bar range has contracted - the coiling that
    precedes a markup in Wyckoff terms.
    """
    tr = true_range(df)
    recent = tr.rolling(window, min_periods=window // 2).mean()
    norm = tr.rolling(baseline, min_periods=baseline // 3).mean()
    return recent / norm.replace(0, np.nan)


def volume_ratio(df: pd.DataFrame, window: int = 20, baseline: int = 120) -> pd.Series:
    recent = df["volume"].rolling(window, min_periods=window // 2).mean()
    norm = df["volume"].rolling(baseline, min_periods=baseline // 3).mean()
    return recent / norm.replace(0, np.nan)


def relative_strength(close: pd.Series, benchmark: pd.Series, window: int = 60) -> pd.Series:
    """Return of the stock minus return of the benchmark over ``window`` bars."""
    stock = close.pct_change(window)
    index = benchmark.pct_change(window)
    return stock - index


def divergence(price: pd.Series, flow: pd.Series, window: int = 60) -> pd.Series:
    """Flow trend minus price trend, both normalised.

    Positive means volume flow is rising faster than price - the classic
    footprint of absorption: someone is taking stock without paying up.
    """
    price_slope = rolling_slope(price, window)
    flow_norm = flow / flow.abs().rolling(window * 2, min_periods=window).mean().replace(0, np.nan)
    flow_slope = rolling_slope(flow_norm.ffill(), window)
    return flow_slope - price_slope


def drawdown(close: pd.Series) -> pd.Series:
    return close / close.cummax() - 1.0


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Return realised over the NEXT ``horizon`` bars (NaN at the tail)."""
    return close.shift(-horizon) / close - 1.0


def enrich(df: pd.DataFrame, cfg=None, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Attach the standard indicator set to an OHLCV frame."""
    atr_period = int(cfg.get("accumulation.atr_period", 14)) if cfg else 14
    vol_ma = int(cfg.get("accumulation.volume_ma", 20)) if cfg else 20
    lookback = int(cfg.get("accumulation.lookback", 60)) if cfg else 60
    short = int(cfg.get("accumulation.short_lookback", 20)) if cfg else 20

    out = df.copy()
    out["atr"] = atr(out, atr_period)
    out["atr_pct"] = out["atr"] / out["close"].replace(0, np.nan)
    out["obv"] = obv(out)
    out["ad"] = accumulation_distribution(out)
    out["cmf"] = chaikin_money_flow(out, vol_ma)
    out["vpt"] = volume_price_trend(out)
    out["mfi"] = money_flow_index(out, atr_period)
    out["vol_ma"] = out["volume"].rolling(vol_ma, min_periods=vol_ma // 2).mean()
    out["vol_ratio"] = volume_ratio(out, short, lookback * 2)
    out["range_compression"] = range_compression(out, short, lookback * 2)
    out["range_pos_60"] = range_position(out, lookback)
    out["range_pos_120"] = range_position(out, lookback * 2)
    out["price_slope"] = rolling_slope(out["close"], lookback)
    out["obv_divergence"] = divergence(out["close"], out["obv"], lookback)
    out["ad_divergence"] = divergence(out["close"], out["ad"], lookback)
    out["drawdown"] = drawdown(out["close"])
    out["ret_20"] = out["close"].pct_change(short)
    out["ret_60"] = out["close"].pct_change(lookback)

    if benchmark is not None and not benchmark.empty:
        aligned = benchmark.reindex(out.index).ffill()
        out["rel_strength"] = relative_strength(out["close"], aligned, lookback)
    else:
        out["rel_strength"] = np.nan
    return out
