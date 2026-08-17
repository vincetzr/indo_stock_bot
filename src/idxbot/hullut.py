"""Hull Suite + UT Bot: signals, honest execution, and walk-forward selection.

The method, as usually described: *buy when UT Bot prints a buy and the Hull
band is green, sell when UT Bot prints a sell and the band turns red.* This
module implements exactly that, plus each half on its own, because the only way
to know whether the combination earns its complexity is to price the parts
separately.

Execution rules, and why each one is there
------------------------------------------
Every one of these exists because leaving it out flatters the result:

* **Signals are read at the close of bar t and filled at the open of bar t+1.**
  Filling at the close of the signal bar is the single most common way a
  trend-following backtest invents returns it could never have captured. This
  repo has already been bitten once by same-bar execution (``docs/FINDINGS.md``,
  the 96%-win-rate bug), so the fill bar is never the signal bar.
* **Auto-rejection is respected.** A bar locked limit-up (``open == high ==
  low`` on a large gap) cannot be bought, and one locked limit-down cannot be
  sold. Orders wait for the next bar that actually trades. Ignoring ARA/ARB
  lets a backtest buy exactly the moves that were unbuyable.
* **Costs are charged on both legs**: an IDX retail schedule of 0.15% to buy and
  0.25% to sell (the extra 0.1% being the sale tax), plus slippage each way.
* **Dividends accrue only while the position is open.** A timing strategy is out
  of the market part of the time and forgoes the dividends paid during those
  spells. Comparing a price-return strategy against a price-return buy-and-hold
  hides that, and on IDX blue chips, at 3-6% yields, it is most of the argument.
  Signals are computed on the price series - what the chart actually shows -
  while P&L runs on total return.

What the benchmark is
---------------------
Buy-and-hold **total return in the same name over the same window**, not the
index. A strategy that trades ASII must beat owning ASII; beating IHSG while
losing to ASII is a stock-picking result wearing a timing costume.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .indicators import HULL_MODES, hull, hull_is_green, ut_bot, warmup_bars

# IDX retail cost schedule. Sell carries an extra 0.1% final income tax, which
# is why the two legs differ.
FEE_BUY = 0.0015
FEE_SELL = 0.0025
SLIPPAGE = 0.0010          # one way, on top of fees

#: A bar that never moved off its open, on a gap of at least this size, is
#: taken to be locked at auto-rejection and therefore unfillable.
LOCK_GAP = 0.045

ENTRY_MODES = ("confluence", "ut", "hull")
EXIT_MODES = ("either", "ut", "hull")


@dataclass(frozen=True)
class Params:
    """One configuration of the pair. Defaults are the published defaults."""

    hull_length: int = 55
    hull_mode: str = "hma"
    ut_key: float = 1.0
    ut_atr: int = 10
    entry: str = "confluence"
    exit: str = "either"

    def label(self) -> str:
        return (f"{self.entry}/{self.exit} hull={self.hull_mode}{self.hull_length} "
                f"ut={self.ut_key:g}x{self.ut_atr}")

    def validate(self) -> "Params":
        if self.hull_mode not in HULL_MODES:
            raise ValueError(f"hull_mode must be one of {HULL_MODES}")
        if self.entry not in ENTRY_MODES:
            raise ValueError(f"entry must be one of {ENTRY_MODES}")
        if self.exit not in EXIT_MODES:
            raise ValueError(f"exit must be one of {EXIT_MODES}")
        return self


@dataclass
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp]
    exit_price: float
    bars_held: int
    gross_return: float
    net_return: float
    exit_reason: str


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------
def prepare(bars: pd.DataFrame) -> pd.DataFrame:
    """Attach the per-bar dividend factor and the tradeability mask.

    ``div_factor`` isolates the dividend from the price move: total return
    divided by price return is 1.0 on an ordinary day and greater than 1.0 on
    an ex-dividend day. Yahoo's OHLC is already split-adjusted (verified: no
    IDX blue chip shows a raw daily move beyond +/-30% at any split date), so
    the only thing separating ``close`` from ``adj_close`` is distributions.
    """
    df = bars.copy().sort_values("date").reset_index(drop=True)
    if "adj_close" not in df:
        df["adj_close"] = df["close"]

    price_ret = df["close"].pct_change()
    total_ret = df["adj_close"].pct_change()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["div_factor"] = (1.0 + total_ret) / (1.0 + price_ret)
    df["div_factor"] = df["div_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    # Guard against adjustment noise being read as a 40% dividend.
    df.loc[(df["div_factor"] < 0.5) | (df["div_factor"] > 1.5), "div_factor"] = 1.0

    prev_close = df["close"].shift(1)
    gap = (df["open"] / prev_close - 1.0).abs()
    flat = (df["high"] <= df["low"] * 1.0001)
    locked = flat & (gap >= LOCK_GAP)
    df["can_buy"] = ~(locked & (df["open"] > prev_close))
    df["can_sell"] = ~(locked & (df["open"] < prev_close))
    df.loc[df.index[0], ["can_buy", "can_sell"]] = True
    return df


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------
def signals(bars: pd.DataFrame, params: Params,
            hull_line: Optional[pd.Series] = None,
            ut: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Entry and exit flags, each a function of bars up to and including t.

    ``hull_line`` and ``ut`` let a caller supply values it has already computed
    for this exact series and parameter set (see :class:`SignalBank`). They are
    an optimisation only - passing values computed from a different series or
    different parameters produces nonsense, which is why nothing outside the
    bank uses them.
    """
    params.validate()
    df = bars if "div_factor" in bars else prepare(bars)
    close = df["close"].astype(float)

    line = hull(close, params.hull_length, params.hull_mode) if hull_line is None \
        else hull_line
    green = hull_is_green(line)
    if ut is None:
        ut = ut_bot(df["high"].astype(float), df["low"].astype(float), close,
                    key=params.ut_key, atr_length=params.ut_atr)

    green_true = green.fillna(False).astype(bool)
    red_true = (~green.fillna(True)).astype(bool)   # unknown never counts as red
    turned_green = green_true & ~green_true.shift(1, fill_value=False)
    turned_red = red_true & ~red_true.shift(1, fill_value=False)

    if params.entry == "confluence":
        # UT prints a buy while the band is already green. This is the rule as
        # people describe it, and it is strict: both must hold on the same bar.
        entry = ut["buy"] & green_true
    elif params.entry == "ut":
        entry = ut["buy"]
    else:
        entry = turned_green

    if params.exit == "either":
        exit_ = ut["sell"] | turned_red
    elif params.exit == "ut":
        exit_ = ut["sell"]
    else:
        exit_ = turned_red

    warm = warmup_bars(params.hull_length, params.hull_mode, params.ut_atr)
    ready = pd.Series(np.arange(len(df)) >= warm, index=df.index)

    out = df.copy()
    out["hull"] = line
    out["hull_green"] = green
    out["ut_stop"] = ut["stop"]
    out["ut_buy"] = ut["buy"]
    out["ut_sell"] = ut["sell"]
    out["entry_signal"] = (entry.fillna(False).astype(bool) & ready)
    out["exit_signal"] = (exit_.fillna(False).astype(bool) & ready)
    return out



class SignalBank:
    """Caches the expensive per-ticker indicator pieces across a sweep.

    A grid of L hull settings x U stop settings does not need L*U passes: the
    Hull line depends only on (length, mode) and the UT stop only on (key, ATR
    period). Computing each once and combining brings a 240-configuration sweep
    down to 31 indicator evaluations per name.

    Everything is computed on the **full** series and sliced afterwards. That is
    not a shortcut - both indicators are causal, which the test suite proves by
    truncation, so the value at bar t is identical whether the series was cut at
    t or not. Slicing after the fact therefore gives a window the warm-up it is
    entitled to instead of making it start cold.
    """

    def __init__(self, panel: Dict[str, pd.DataFrame]):
        self.panel = panel
        self._hull: Dict[Tuple[str, int, str], pd.Series] = {}
        self._ut: Dict[Tuple[str, float, int], pd.DataFrame] = {}
        self._signals: Dict[Tuple[str, Params], pd.DataFrame] = {}

    def hull_line(self, ticker: str, length: int, mode: str) -> pd.Series:
        key = (ticker, int(length), str(mode))
        if key not in self._hull:
            self._hull[key] = hull(self.panel[ticker]["close"].astype(float),
                                   length, mode)
        return self._hull[key]

    def ut_frame(self, ticker: str, ut_key: float, atr_length: int) -> pd.DataFrame:
        key = (ticker, float(ut_key), int(atr_length))
        if key not in self._ut:
            bars = self.panel[ticker]
            self._ut[key] = ut_bot(bars["high"].astype(float),
                                   bars["low"].astype(float),
                                   bars["close"].astype(float),
                                   key=ut_key, atr_length=atr_length)
        return self._ut[key]

    def signals(self, ticker: str, params: Params) -> pd.DataFrame:
        key = (ticker, params)
        if key not in self._signals:
            self._signals[key] = signals(
                self.panel[ticker], params,
                hull_line=self.hull_line(ticker, params.hull_length, params.hull_mode),
                ut=self.ut_frame(ticker, params.ut_key, params.ut_atr))
        return self._signals[key]


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------
def simulate(bars: pd.DataFrame, params: Params, ticker: str = "",
             fee_buy: float = FEE_BUY, fee_sell: float = FEE_SELL,
             slippage: float = SLIPPAGE,
             trade_from: Optional[pd.Timestamp] = None
             ) -> Tuple[List[Trade], pd.DataFrame]:
    """Run the rules over one series and return its trades and daily equity.

    The loop is deliberately explicit. A vectorised version of this is where
    off-by-one look-ahead hides, and the whole value of the exercise rests on
    the fill bar being strictly later than the signal bar.

    ``trade_from`` separates *when the indicator may look* from *when the
    strategy may trade*. Without it, every out-of-sample window would begin
    with the indicator cold and spend its first 60-160 bars unable to signal at
    all - up to 17% of a walk-forward test slice silently sitting in cash and
    scoring as though that were a decision. Bars before ``trade_from`` warm the
    indicator and nothing else.
    """
    df = signals(bars, params)
    return simulate_signals(df, params, ticker=ticker, fee_buy=fee_buy,
                            fee_sell=fee_sell, slippage=slippage,
                            trade_from=trade_from)


def simulate_signals(df: pd.DataFrame, params: Params, ticker: str = "",
                     fee_buy: float = FEE_BUY, fee_sell: float = FEE_SELL,
                     slippage: float = SLIPPAGE,
                     trade_from: Optional[pd.Timestamp] = None
                     ) -> Tuple[List[Trade], pd.DataFrame]:
    """The execution loop, over an already-computed signal frame.

    Split out from :func:`simulate` so a parameter sweep can reuse one set of
    indicator values across many windows instead of recomputing them, without
    the sweep and the single-name path drifting into two different execution
    models.
    """
    n = len(df)
    open_px = df["open"].to_numpy(float)
    close_px = df["close"].to_numpy(float)
    div = df["div_factor"].to_numpy(float)
    can_buy = df["can_buy"].to_numpy(bool)
    can_sell = df["can_sell"].to_numpy(bool)
    entry_sig = df["entry_signal"].to_numpy(bool)
    exit_sig = df["exit_signal"].to_numpy(bool)
    # Kept as numpy datetime64 and converted per trade, not per bar. Building a
    # 5,500-element list of Timestamps for every name on every configuration was
    # the single largest cost in a parameter sweep.
    date_values = df["date"].to_numpy()

    def _date(i: int) -> pd.Timestamp:
        return pd.Timestamp(date_values[i])

    if trade_from is not None:
        # Masked on the array rather than by rebuilding the frame: a sweep calls
        # this thousands of times and a DataFrame copy per call dominated
        # everything else.
        entry_sig = entry_sig & (df["date"].to_numpy() >= np.datetime64(
            pd.Timestamp(trade_from)))

    trades: List[Trade] = []
    position = 0.0          # units held, 1.0 when long
    entry_price = np.nan
    entry_index = -1
    div_mult = 1.0
    pending_entry = pending_exit = False
    equity = np.ones(n)
    held = np.zeros(n, dtype=bool)
    cash = 1.0
    units = 0.0

    for i in range(n):
        # --- fills happen at THIS bar's open, from a signal seen yesterday ---
        if position == 0.0 and pending_entry and can_buy[i]:
            fill = open_px[i] * (1.0 + slippage)
            if np.isfinite(fill) and fill > 0:
                units = cash / (fill * (1.0 + fee_buy))
                cash = 0.0
                position, entry_price, entry_index, div_mult = 1.0, fill, i, 1.0
                pending_entry = False
        elif position > 0.0 and pending_exit and can_sell[i]:
            fill = open_px[i] * (1.0 - slippage)
            if np.isfinite(fill) and fill > 0:
                cash = units * fill * (1.0 - fee_sell) * div_mult
                gross = (fill / entry_price) * div_mult - 1.0
                net = cash / (units * entry_price * (1.0 + fee_buy)) - 1.0
                trades.append(Trade(
                    ticker=ticker, entry_date=_date(entry_index),
                    entry_price=entry_price, exit_date=_date(i), exit_price=fill,
                    bars_held=i - entry_index, gross_return=gross,
                    net_return=net, exit_reason="signal"))
                units, position, pending_exit = 0.0, 0.0, False

        # --- dividends accrue only for bars held ---
        if position > 0.0 and i > entry_index:
            div_mult *= div[i]

        equity[i] = cash if position == 0.0 else units * close_px[i] * div_mult
        held[i] = position > 0.0

        # --- signals are READ at this bar's close, acted on next bar ---
        if position == 0.0:
            if entry_sig[i]:
                pending_entry = True
        else:
            if exit_sig[i]:
                pending_exit = True
                pending_entry = False

    if position > 0.0:
        fill = close_px[n - 1]
        cash = units * fill * (1.0 - fee_sell) * div_mult
        trades.append(Trade(
            ticker=ticker, entry_date=_date(entry_index), entry_price=entry_price,
            exit_date=None, exit_price=fill, bars_held=n - 1 - entry_index,
            gross_return=(fill / entry_price) * div_mult - 1.0,
            net_return=cash / (units * entry_price * (1.0 + fee_buy)) - 1.0,
            exit_reason="open at end"))

    curve = pd.DataFrame({"date": df["date"].to_numpy(), "equity": equity,
                          "in_market": held})
    return trades, curve


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def buy_and_hold(bars: pd.DataFrame, start_index: int = 0) -> float:
    """Total return of simply owning it, dividends included."""
    df = bars if "adj_close" in bars else prepare(bars)
    series = df["adj_close"].to_numpy(float)[start_index:]
    series = series[np.isfinite(series)]
    if len(series) < 2 or series[0] <= 0:
        return float("nan")
    return float(series[-1] / series[0] - 1.0)


def _cagr(total_return: float, years: float) -> float:
    if not np.isfinite(total_return) or years <= 0 or total_return <= -1:
        return float("nan")
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def summarise(trades: Sequence[Trade], curve: pd.DataFrame,
              bars: pd.DataFrame) -> Dict[str, float]:
    """Per-name statistics, always alongside the buy-and-hold it must beat."""
    if curve.empty:
        return {}
    equity = curve["equity"].to_numpy(float)
    span = curve["date"].to_numpy()
    years = max((pd.Timestamp(span[-1]) - pd.Timestamp(span[0])).days / 365.25, 1e-9)
    total = float(equity[-1] / equity[0] - 1.0)

    nets = np.array([t.net_return for t in trades], dtype=float)
    wins = nets[nets > 0]
    losses = nets[nets <= 0]
    peak = np.maximum.accumulate(equity)
    drawdown = float(np.min(equity / np.where(peak > 0, peak, np.nan) - 1.0))

    # Warm-up bars are unusable by construction, so the benchmark starts where
    # the strategy could first have acted rather than at the first bar.
    prepared = bars if "div_factor" in bars else prepare(bars)
    bh = buy_and_hold(prepared)

    with np.errstate(divide="ignore", invalid="ignore"):
        daily = np.diff(equity) / np.where(equity[:-1] != 0, equity[:-1], np.nan)
    daily = daily[np.isfinite(daily)]
    sd = daily.std(ddof=1) if len(daily) > 2 else 0.0
    sharpe = float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan

    return {
        "trades": float(len(trades)),
        "total_return": total,
        "cagr": _cagr(total, years),
        "buy_hold": bh,
        "buy_hold_cagr": _cagr(bh, years),
        "excess_cagr": _cagr(total, years) - _cagr(bh, years),
        "win_rate": float(len(wins) / len(nets)) if len(nets) else np.nan,
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() else np.inf,
        "expectancy": float(nets.mean()) if len(nets) else np.nan,
        "max_drawdown": drawdown,
        "sharpe": sharpe,
        "time_in_market": float(curve["in_market"].mean()),
        "avg_bars_held": float(np.mean([t.bars_held for t in trades])) if trades else np.nan,
        "years": years,
    }


def run_universe(panel: Dict[str, pd.DataFrame], params: Params,
                 start: Optional[pd.Timestamp] = None,
                 end: Optional[pd.Timestamp] = None,
                 bank: Optional["SignalBank"] = None,
                 **costs) -> pd.DataFrame:
    """Apply one configuration across many names, one row per name.

    ``start`` is the first date the strategy may **trade**, not the first date
    it may **see**. Bars before it are handed to the indicator as warm-up and
    are excluded from every number reported, so a window that begins mid-series
    is scored on live decisions from its first day rather than spending its
    opening months structurally flat.

    ``start`` is also exclusive of the previous window's last bar, so
    consecutive walk-forward slices share no data at all.
    """
    warm = warmup_bars(params.hull_length, params.hull_mode, params.ut_atr)
    rows = []
    for ticker, bars in panel.items():
        # Indicators are computed over the whole series and only then sliced.
        # Both are causal, so a value is unchanged by the presence of later
        # bars - which means the window inherits a warm indicator rather than
        # spending its first months unable to signal.
        sig = bank.signals(ticker, params) if bank is not None \
            else signals(bars, params)
        df = sig
        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]
        trade_from = None
        if start is not None:
            live = df.index[df["date"] > pd.Timestamp(start)]
            if not len(live):
                continue
            trade_from = df.loc[live[0], "date"]
        if len(df) < warm + 30:
            continue
        trades, curve = simulate_signals(df, params, ticker=ticker,
                                         trade_from=trade_from, **costs)
        scored = df
        if trade_from is not None:
            curve = curve[curve["date"] >= trade_from]
            scored = df[df["date"] >= trade_from]
            if curve.empty or len(scored) < 30:
                continue
        stats = summarise(trades, curve, scored)
        if stats:
            rows.append({"ticker": ticker, **stats})
    return pd.DataFrame(rows)


def aggregate(per_name: pd.DataFrame) -> Dict[str, float]:
    """Equal-weight portfolio view of a per-name table.

    ``median_excess_cagr`` is the number that matters. A mean is dominated by
    one or two names that happened to trend for a decade, and the question is
    whether the method helps the *typical* stock it is pointed at.
    """
    if per_name.empty:
        return {}
    return {
        "names": float(len(per_name)),
        "mean_cagr": float(per_name["cagr"].mean()),
        "median_cagr": float(per_name["cagr"].median()),
        "mean_buy_hold_cagr": float(per_name["buy_hold_cagr"].mean()),
        "median_buy_hold_cagr": float(per_name["buy_hold_cagr"].median()),
        "median_excess_cagr": float(per_name["excess_cagr"].median()),
        "beat_buy_hold": float((per_name["excess_cagr"] > 0).mean()),
        "median_win_rate": float(per_name["win_rate"].median()),
        "median_max_dd": float(per_name["max_drawdown"].median()),
        "median_trades": float(per_name["trades"].median()),
        "median_sharpe": float(per_name["sharpe"].median()),
        "median_time_in_market": float(per_name["time_in_market"].median()),
    }


# ---------------------------------------------------------------------------
# parameter search
# ---------------------------------------------------------------------------
def grid(hull_lengths: Iterable[int] = (21, 34, 55, 89, 144),
         hull_modes: Iterable[str] = ("hma", "ehma", "thma"),
         ut_keys: Iterable[float] = (0.5, 1.0, 2.0, 3.0),
         ut_atrs: Iterable[int] = (5, 10, 14, 21),
         entries: Iterable[str] = ("confluence",),
         exits: Iterable[str] = ("either",)) -> List[Params]:
    return [Params(hull_length=hl, hull_mode=hm, ut_key=k, ut_atr=a,
                   entry=e, exit=x)
            for hl, hm, k, a, e, x in itertools.product(
                hull_lengths, hull_modes, ut_keys, ut_atrs, entries, exits)]


def score_grid(panel: Dict[str, pd.DataFrame], candidates: Sequence[Params],
               start=None, end=None, objective: str = "median_excess_cagr",
               verbose: bool = False,
               bank: Optional["SignalBank"] = None) -> pd.DataFrame:
    """Evaluate every candidate over one window. One row per candidate."""
    bank = bank if bank is not None else SignalBank(panel)
    rows = []
    for i, params in enumerate(candidates, 1):
        per_name = run_universe(panel, params, start=start, end=end, bank=bank)
        stats = aggregate(per_name)
        if not stats:
            continue
        rows.append({"label": params.label(), "params": params, **stats})
        if verbose and i % 20 == 0:
            print(f"    ... {i}/{len(candidates)}")
    out = pd.DataFrame(rows)
    return out.sort_values(objective, ascending=False).reset_index(drop=True) if not out.empty else out


def walk_forward(panel: Dict[str, pd.DataFrame], candidates: Sequence[Params],
                 folds: Sequence[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
                 objective: str = "median_excess_cagr",
                 baseline: Optional[Params] = None,
                 verbose: bool = True) -> pd.DataFrame:
    """Choose parameters in-sample, then score them on the untouched next slice.

    ``folds`` are ``(train_start, train_end, test_end)`` triples. The chosen
    configuration is fitted only on ``[train_start, train_end]`` and applied to
    ``(train_end, test_end]``, and the fixed ``baseline`` is scored on the same
    out-of-sample window so the comparison answers the question that matters:
    **did the optimisation add anything over leaving the defaults alone?**
    """
    baseline = baseline or Params()
    bank = SignalBank(panel)
    rows = []
    for k, (train_start, train_end, test_end) in enumerate(folds, 1):
        if verbose:
            print(f"  fold {k}: train {train_start:%Y-%m} -> {train_end:%Y-%m}, "
                  f"test -> {test_end:%Y-%m}")
        in_sample = score_grid(panel, candidates, start=train_start,
                               end=train_end, objective=objective, bank=bank)
        if in_sample.empty:
            continue
        chosen: Params = in_sample.iloc[0]["params"]

        oos = aggregate(run_universe(panel, chosen, start=train_end,
                                     end=test_end, bank=bank))
        base = aggregate(run_universe(panel, baseline, start=train_end,
                                      end=test_end, bank=bank))
        if not oos or not base:
            continue
        rows.append({
            "fold": k, "train_start": train_start, "train_end": train_end,
            "test_end": test_end, "chosen": chosen.label(),
            "is_objective": float(in_sample.iloc[0][objective]),
            "oos_objective": float(oos[objective]),
            "oos_cagr": oos["median_cagr"],
            "oos_buy_hold": oos["median_buy_hold_cagr"],
            "baseline_objective": float(base[objective]),
            "baseline_cagr": base["median_cagr"],
            "optimisation_value_add": float(oos[objective] - base[objective]),
        })
    return pd.DataFrame(rows)


def expanding_folds(dates: pd.DatetimeIndex, n_folds: int = 5,
                    min_train_years: float = 6.0) -> List[Tuple]:
    """Expanding in-sample windows with contiguous, non-overlapping test slices."""
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates)))
    if len(dates) < 2:
        return []
    first, last = dates[0], dates[-1]
    train_end0 = first + pd.Timedelta(days=int(min_train_years * 365.25))
    if train_end0 >= last:
        return []
    edges = pd.date_range(train_end0, last, periods=n_folds + 1)
    return [(first, edges[i], edges[i + 1]) for i in range(n_folds)]
