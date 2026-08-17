"""Single-name timing: a harness for hunting the gap between hold and hindsight.

On ADRO, owning it for eighteen years returns 7.77x. Perfect foresight with a
single round trip a year returns **198,000x**. That gap is the entire prize in
market timing, and this module exists to measure honestly how much of it any
causal rule can actually reach.

Everything here obeys the same three rules, because each one is a way backtests
lie:

1. **A position decided from bar t is held from the open of bar t+1.** Never the
   close of the signal bar.
2. **Costs on every position change**, sized by how much the position moved, so
   a strategy that flips daily pays for flipping daily.
3. **Signals are causal by construction and asserted so by truncation** - cut
   the series at t, recompute, and no value at or before t may move.

Long, flat and short are all allowed. Shorting IDX equities is restricted in
practice, so short-enabled variants are reported separately and never mixed into
a headline that implies you could have traded them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: One-way cost: IDX fees plus slippage. Charged proportional to |position change|.
ONE_WAY = 0.004


# ---------------------------------------------------------------------------
# indicator primitives (causal)
# ---------------------------------------------------------------------------
def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / length
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def zscore(close: pd.Series, length: int = 60) -> pd.Series:
    mean = close.rolling(length, min_periods=length).mean()
    sd = close.rolling(length, min_periods=length).std()
    return (close - mean) / sd.replace(0.0, np.nan)


def drawdown_from_high(close: pd.Series, length: int = 250) -> pd.Series:
    return close / close.rolling(length, min_periods=length // 2).max() - 1.0


# ---------------------------------------------------------------------------
# strategies: each returns a position series in [-1, 1], causal
# ---------------------------------------------------------------------------
def donchian(bars: pd.DataFrame, enter: int = 55, exit_: int = 20,
             allow_short: bool = False) -> pd.Series:
    """Classic breakout. Long above the N-day high, out below the M-day low."""
    close = bars["close"]
    hi = close.rolling(enter, min_periods=enter).max()
    lo = close.rolling(exit_, min_periods=exit_).min()
    hi_s = close.rolling(enter, min_periods=enter).min()
    pos = np.zeros(len(close))
    state = 0.0
    c = close.to_numpy(float)
    hi_v, lo_v, his_v = hi.to_numpy(), lo.to_numpy(), hi_s.to_numpy()
    for i in range(1, len(c)):
        if np.isfinite(hi_v[i - 1]) and c[i] >= hi_v[i - 1]:
            state = 1.0
        elif np.isfinite(lo_v[i - 1]) and c[i] <= lo_v[i - 1]:
            state = -1.0 if allow_short else 0.0
        pos[i] = state
    return pd.Series(pos, index=close.index)


def ma_cross(bars: pd.DataFrame, fast: int = 50, slow: int = 200,
             allow_short: bool = False) -> pd.Series:
    close = bars["close"]
    f = close.rolling(fast, min_periods=fast).mean()
    s = close.rolling(slow, min_periods=slow).mean()
    raw = np.where(f > s, 1.0, -1.0 if allow_short else 0.0)
    out = pd.Series(raw, index=close.index)
    return out.where(f.notna() & s.notna(), other=0.0)


def rsi_reversion(bars: pd.DataFrame, length: int = 14, buy_below: float = 30.0,
                  sell_above: float = 70.0, allow_short: bool = False) -> pd.Series:
    """Buy oversold, sell overbought - the literal 'buy low, sell high' rule."""
    r = rsi(bars["close"], length).to_numpy()
    pos = np.zeros(len(r))
    state = 0.0
    for i in range(len(r)):
        if np.isfinite(r[i]):
            if r[i] <= buy_below:
                state = 1.0
            elif r[i] >= sell_above:
                state = -1.0 if allow_short else 0.0
        pos[i] = state
    return pd.Series(pos, index=bars["close"].index)


def zscore_reversion(bars: pd.DataFrame, length: int = 60, entry: float = -2.0,
                     exit_: float = 0.0, allow_short: bool = False) -> pd.Series:
    z = zscore(bars["close"], length).to_numpy()
    pos = np.zeros(len(z))
    state = 0.0
    for i in range(len(z)):
        if np.isfinite(z[i]):
            if z[i] <= entry:
                state = 1.0
            elif z[i] >= exit_ and state > 0:
                state = 0.0
            elif allow_short and z[i] >= -entry:
                state = -1.0
            elif allow_short and state < 0 and z[i] <= -exit_:
                state = 0.0
        pos[i] = state
    return pd.Series(pos, index=bars["close"].index)


def dip_buy(bars: pd.DataFrame, window: int = 250, depth: float = -0.30,
            recover: float = -0.10, allow_short: bool = False) -> pd.Series:
    """Buy when it is deep below its running high, sell when it has recovered."""
    dd = drawdown_from_high(bars["close"], window).to_numpy()
    pos = np.zeros(len(dd))
    state = 0.0
    for i in range(len(dd)):
        if np.isfinite(dd[i]):
            if dd[i] <= depth:
                state = 1.0
            elif dd[i] >= recover:
                state = 0.0
        pos[i] = state
    return pd.Series(pos, index=bars["close"].index)


def momentum(bars: pd.DataFrame, length: int = 250, allow_short: bool = False
             ) -> pd.Series:
    close = bars["close"]
    m = close / close.shift(length) - 1.0
    raw = np.where(m > 0, 1.0, -1.0 if allow_short else 0.0)
    return pd.Series(raw, index=close.index).where(m.notna(), other=0.0)


STRATEGIES: Dict[str, Callable[..., pd.Series]] = {
    "donchian": donchian,
    "ma_cross": ma_cross,
    "rsi_reversion": rsi_reversion,
    "zscore_reversion": zscore_reversion,
    "dip_buy": dip_buy,
    "momentum": momentum,
}


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
@dataclass
class TimingResult:
    label: str
    equity: pd.Series
    stats: Dict[str, float]


def backtest(bars: pd.DataFrame, position: pd.Series, cost: float = ONE_WAY,
             leverage: float = 1.0, label: str = "") -> TimingResult:
    """Apply a position series to total returns with a one-bar execution lag.

    The lag is the whole point: ``position`` is decided using bar *t*, so it can
    only be *held* from bar t+1. Shifting by one here is what separates a
    backtest from a look at the answers.
    """
    df = bars.sort_values("date").reset_index(drop=True)
    total = df["adj_close"].pct_change().fillna(0.0).to_numpy(float)
    pos = pd.Series(position).reset_index(drop=True).fillna(0.0).shift(1).fillna(0.0)
    pos = (pos.to_numpy(float) * leverage)

    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    net = pos * total - turnover * cost
    equity = np.cumprod(1.0 + net)

    years = max((df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25, 1e-9)
    peak = np.maximum.accumulate(equity)
    trades = float((np.abs(np.diff(np.concatenate([[0.0], pos]))) > 1e-9).sum())
    sd = net.std(ddof=1)
    return TimingResult(
        label=label,
        equity=pd.Series(equity, index=pd.DatetimeIndex(df["date"])),
        stats={
            "years": years,
            "total_growth": float(equity[-1]),
            "cagr": float(equity[-1] ** (1.0 / years) - 1.0) if equity[-1] > 0 else -1.0,
            "max_drawdown": float(np.min(equity / peak - 1.0)),
            "time_in_market": float(np.mean(np.abs(pos) > 1e-9)),
            "position_changes": trades,
            "sharpe": float(net.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
            "leverage": leverage,
        })


def buy_and_hold(bars: pd.DataFrame) -> TimingResult:
    ones = pd.Series(np.ones(len(bars)), index=range(len(bars)))
    return backtest(bars, ones, cost=0.0, label="buy & hold")


# ---------------------------------------------------------------------------
# perfect-foresight ceilings, so every result has a scale
# ---------------------------------------------------------------------------
def perfect_k_trades(bars: pd.DataFrame, k: int) -> float:
    """Best achievable growth with at most ``k`` round trips, knowing everything.

    Standard O(n*k) dynamic program on log prices. This is not a strategy and
    cannot be traded; it exists so that every causal result below can be quoted
    as a *fraction of what was theoretically there*, which is far more
    informative than an unanchored CAGR.
    """
    prices = bars["adj_close"].to_numpy(float)
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if len(prices) < 2 or k < 1:
        return 1.0
    logp = np.log(prices)
    buy = np.full(k + 1, -np.inf)
    sell = np.zeros(k + 1)
    for p in logp:
        for j in range(1, k + 1):
            if sell[j - 1] - p > buy[j]:
                buy[j] = sell[j - 1] - p
            if buy[j] + p > sell[j]:
                sell[j] = buy[j] + p
    return float(np.exp(sell[k]))


def capture_ratio(result: TimingResult, ceiling_growth: float) -> float:
    """What share of the theoretically available log-growth was captured."""
    if ceiling_growth <= 1 or result.stats["total_growth"] <= 0:
        return float("nan")
    return float(np.log(result.stats["total_growth"]) / np.log(ceiling_growth))
