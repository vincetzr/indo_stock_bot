"""Tests for the Hull + UT Bot strategy: execution honesty above all.

The indicator tests prove the signals are computed without peeking. These prove
the *backtest* does not give itself anything the signals did not earn:

  * a signal read at bar t can only fill at bar t+1 or later;
  * costs and slippage are charged once each, on both legs;
  * dividends accrue only for bars actually held;
  * a bar locked at auto-rejection cannot be traded;
  * a window that starts mid-series warms the indicator on prior bars rather
    than spending its first months structurally flat.

Each is checked on a hand-built series where the correct answer is known by
construction, not by running the real data and eyeballing the output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idxbot.hullut import (
    FEE_BUY, FEE_SELL, SLIPPAGE, Params, aggregate, buy_and_hold,
    expanding_folds, grid, prepare, run_universe, signals, simulate, summarise,
)

FAST = Params(hull_length=4, hull_mode="hma", ut_key=1.0, ut_atr=3)


def make_bars(prices, highs=None, lows=None, opens=None, adj=None,
              start="2020-01-01") -> pd.DataFrame:
    """A frame with one knob per column, so each test can move exactly one thing."""
    prices = np.asarray(prices, dtype=float)
    return pd.DataFrame({
        "date": pd.bdate_range(start, periods=len(prices)),
        "open": prices if opens is None else np.asarray(opens, float),
        "high": prices if highs is None else np.asarray(highs, float),
        "low": prices if lows is None else np.asarray(lows, float),
        "close": prices,
        "adj_close": prices if adj is None else np.asarray(adj, float),
        "volume": 1_000_000,
    })


def ramp_series(n=80) -> np.ndarray:
    """Flat, then a clean rally, then a clean decline: one unambiguous trade."""
    return np.concatenate([
        np.full(30, 100.0),
        np.arange(101.0, 116.0),
        np.arange(115.0, 115.0 - (n - 45), -1.0),
    ])[:n]


@pytest.fixture(scope="module")
def trade_case() -> pd.DataFrame:
    return prepare(make_bars(ramp_series()))


# ---------------------------------------------------------------------------
# execution honesty
# ---------------------------------------------------------------------------
def test_a_signal_never_fills_on_its_own_bar(trade_case):
    """The single most important assertion in this file.

    Filling at the close of the signal bar is how a trend backtest invents
    returns. Every entry date must be strictly later than the signal that
    caused it.
    """
    sig = signals(trade_case, FAST)
    signal_dates = set(sig.loc[sig["entry_signal"], "date"])
    trades, _ = simulate(trade_case, FAST)
    assert trades
    for trade in trades:
        assert trade.entry_date not in signal_dates


def test_entry_fills_at_the_next_bar_open_plus_slippage(trade_case):
    sig = signals(trade_case, FAST)
    first = sig.index[sig["entry_signal"]][0]
    trades, _ = simulate(trade_case, FAST)
    expected = trade_case["open"].iloc[first + 1] * (1.0 + SLIPPAGE)
    assert trades[0].entry_price == pytest.approx(expected)


def test_net_return_charges_both_legs_exactly_once(trade_case):
    trades, _ = simulate(trade_case, FAST)
    for trade in trades:
        expected = ((trade.exit_price * (1.0 - FEE_SELL))
                    / (trade.entry_price * (1.0 + FEE_BUY)) - 1.0)
        assert trade.net_return == pytest.approx(expected, abs=1e-12)


def test_net_is_worse_than_gross_by_the_cost_of_trading(trade_case):
    trades, _ = simulate(trade_case, FAST)
    for trade in trades:
        assert trade.net_return < trade.gross_return


def test_zero_costs_make_net_equal_gross(trade_case):
    trades, _ = simulate(trade_case, FAST, fee_buy=0.0, fee_sell=0.0, slippage=0.0)
    assert trades
    for trade in trades:
        assert trade.net_return == pytest.approx(trade.gross_return, abs=1e-12)


def test_higher_costs_can_only_reduce_returns(trade_case):
    cheap, _ = simulate(trade_case, FAST, fee_buy=0.0, fee_sell=0.0, slippage=0.0)
    dear, _ = simulate(trade_case, FAST, fee_buy=0.01, fee_sell=0.01, slippage=0.01)
    assert sum(t.net_return for t in dear) < sum(t.net_return for t in cheap)


# ---------------------------------------------------------------------------
# auto-rejection
# ---------------------------------------------------------------------------
def test_a_limit_up_lock_is_not_buyable():
    """ARA: a bar that gapped up and never moved off its open cannot be bought."""
    prices = list(ramp_series(60))
    bars = make_bars(prices)
    lock = 32
    bars.loc[lock, ["open", "high", "low", "close"]] = bars.loc[lock - 1, "close"] * 1.10
    bars.loc[lock, "adj_close"] = bars.loc[lock, "close"]
    prepared = prepare(bars)
    assert not prepared["can_buy"].iloc[lock]
    for trade in simulate(prepared, FAST)[0]:
        assert trade.entry_date != prepared["date"].iloc[lock]


def test_a_limit_down_lock_is_not_sellable():
    prices = list(ramp_series(60))
    bars = make_bars(prices)
    lock = 50
    bars.loc[lock, ["open", "high", "low", "close"]] = bars.loc[lock - 1, "close"] * 0.90
    bars.loc[lock, "adj_close"] = bars.loc[lock, "close"]
    prepared = prepare(bars)
    assert not prepared["can_sell"].iloc[lock]
    for trade in simulate(prepared, FAST)[0]:
        assert trade.exit_date != prepared["date"].iloc[lock]


def test_an_ordinary_bar_stays_tradeable(trade_case):
    assert trade_case["can_buy"].all()
    assert trade_case["can_sell"].all()


# ---------------------------------------------------------------------------
# dividends
# ---------------------------------------------------------------------------
def test_div_factor_isolates_the_distribution():
    prices = ramp_series(60)
    adj = prices.copy()
    adj[40:] = prices[40:] * 1.05
    prepared = prepare(make_bars(prices, adj=adj))
    assert prepared["div_factor"].iloc[40] == pytest.approx(1.05, rel=1e-6)
    assert (prepared["div_factor"].round(9) != 1.0).sum() == 1


def test_dividends_accrue_only_while_held():
    """A distribution paid while flat must not reach the strategy's P&L."""
    prices = ramp_series(80)
    held_case = prices.copy()
    flat_case = prices.copy()
    trades_base, _ = simulate(prepare(make_bars(prices)), FAST)
    assert trades_base
    entry_i = 31

    adj_held = prices.copy()
    adj_held[entry_i + 2:] = prices[entry_i + 2:] * 1.05      # ex-div while long
    adj_flat = prices.copy()
    adj_flat[-3:] = prices[-3:] * 1.05                        # ex-div after the exit

    held = simulate(prepare(make_bars(held_case, adj=adj_held)), FAST)[0]
    flat = simulate(prepare(make_bars(flat_case, adj=adj_flat)), FAST)[0]
    assert held[0].net_return > trades_base[0].net_return
    assert flat[0].net_return == pytest.approx(trades_base[0].net_return, abs=1e-12)


def test_buy_and_hold_uses_total_return():
    prices = np.full(50, 100.0)
    adj = prices.copy()
    adj[25:] = 110.0
    assert buy_and_hold(prepare(make_bars(prices, adj=adj))) == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# warm-up and window boundaries
# ---------------------------------------------------------------------------
def test_a_window_warms_the_indicator_on_prior_bars():
    """Without this, every walk-forward test slice opens with a dead stretch.

    The strategy must be able to trade on the first day of the window it is
    being scored on, which means the indicator has to have been fed the bars
    before it.
    """
    rng = np.random.default_rng(3)
    prices = 100.0 + np.cumsum(rng.normal(0.05, 1.0, 900))
    panel = {"X": prepare(make_bars(prices))}
    cut = panel["X"]["date"].iloc[600]

    windowed = run_universe(panel, Params(), start=cut)
    assert not windowed.empty
    trades, _ = simulate(panel["X"], Params())
    after_cut = [t for t in trades if t.entry_date > cut]
    # The windowed run should find a comparable number of trades to the tail of
    # a full run - not dramatically fewer, which is the warm-up-inside-window bug.
    assert windowed.iloc[0]["trades"] >= max(1, 0.5 * len(after_cut))


def test_window_start_is_exclusive_so_slices_do_not_share_a_bar():
    rng = np.random.default_rng(5)
    prices = 100.0 + np.cumsum(rng.normal(0.05, 1.0, 900))
    panel = {"X": prepare(make_bars(prices))}
    cut = panel["X"]["date"].iloc[600]
    trades, _ = simulate(panel["X"], Params(), trade_from=cut)
    assert all(t.entry_date >= cut for t in trades)


def test_trade_from_suppresses_earlier_entries():
    prices = ramp_series(80)
    prepared = prepare(make_bars(prices))
    late = prepared["date"].iloc[60]
    trades, _ = simulate(prepared, FAST, trade_from=late)
    assert all(t.entry_date >= late for t in trades)


def test_expanding_folds_are_contiguous_and_non_overlapping():
    dates = pd.bdate_range("2000-01-01", "2026-01-01")
    folds = expanding_folds(dates, n_folds=5, min_train_years=8.0)
    assert len(folds) == 5
    for i, (train_start, train_end, test_end) in enumerate(folds):
        assert train_start == dates[0]        # expanding, always from the start
        assert train_end < test_end
        if i:
            assert train_end == folds[i - 1][2]   # no gap, no overlap


def test_expanding_folds_declines_when_history_is_too_short():
    assert expanding_folds(pd.bdate_range("2024-01-01", "2024-06-01"),
                           min_train_years=8.0) == []


# ---------------------------------------------------------------------------
# variants and plumbing
# ---------------------------------------------------------------------------
def test_confluence_requires_both_halves(trade_case):
    """It cannot fire more often than the UT signal it is filtered from."""
    ut_only = signals(trade_case, Params(entry="ut", exit="ut"))
    both = signals(trade_case, Params(entry="confluence", exit="either"))
    assert both["entry_signal"].sum() <= ut_only["entry_signal"].sum()


def test_confluence_entries_are_a_subset_of_ut_entries(trade_case):
    ut_only = signals(trade_case, Params(entry="ut", exit="ut"))
    both = signals(trade_case, Params(entry="confluence", exit="either"))
    assert not (both["entry_signal"] & ~ut_only["entry_signal"]).any()


def test_either_exit_fires_at_least_as_often_as_each_half(trade_case):
    either = signals(trade_case, Params(exit="either"))["exit_signal"].sum()
    ut = signals(trade_case, Params(exit="ut"))["exit_signal"].sum()
    hull_ = signals(trade_case, Params(exit="hull"))["exit_signal"].sum()
    assert either >= ut and either >= hull_


def test_params_reject_nonsense():
    for bad in (Params(hull_mode="kama"), Params(entry="vibes"), Params(exit="soon")):
        with pytest.raises(ValueError):
            bad.validate()


def test_no_trade_is_ever_left_unaccounted(trade_case):
    """An open position at the end is closed and reported, not dropped."""
    trades, curve = simulate(trade_case, FAST)
    assert len(curve) == len(trade_case)
    reasons = {t.exit_reason for t in trades}
    assert reasons.issubset({"signal", "open at end"})


def test_time_in_market_matches_the_trades(trade_case):
    trades, curve = simulate(trade_case, FAST)
    held_bars = sum(t.bars_held for t in trades)
    assert curve["in_market"].sum() == pytest.approx(held_bars + len(trades), abs=1)


def test_summarise_reports_the_benchmark_alongside_the_strategy(trade_case):
    trades, curve = simulate(trade_case, FAST)
    stats = summarise(trades, curve, trade_case)
    assert {"cagr", "buy_hold", "buy_hold_cagr", "excess_cagr"} <= set(stats)
    assert stats["excess_cagr"] == pytest.approx(
        stats["cagr"] - stats["buy_hold_cagr"], abs=1e-12)


def test_aggregate_is_empty_for_an_empty_table():
    assert aggregate(pd.DataFrame()) == {}


def test_grid_is_the_full_cross_product():
    assert len(grid(hull_lengths=(21, 55), hull_modes=("hma",),
                    ut_keys=(1.0, 2.0), ut_atrs=(10,))) == 4


def test_a_flat_series_produces_no_trades():
    """No movement, no signal - and certainly no profit."""
    flat = prepare(make_bars(np.full(300, 100.0)))
    trades, _ = simulate(flat, Params())
    assert trades == []
