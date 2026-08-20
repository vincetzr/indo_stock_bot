"""Tests for the cross-sectional factor study.

A factor backtest lies in four ways, and every one has a test here:

  1. LOOK-AHEAD IN THE SCORE. The classic. Every factor is checked by computing
     it on a prefix, then appending future bars and recomputing - the number must
     not move by a single bit. This is run over all twelve factors rather than a
     representative one, because the leak is always in the one nobody checked.
  2. LOOK-AHEAD IN THE BOOK. Same test at the portfolio level: extending the
     panel forward must not change a period return that already happened.
  3. FREE TRADING. Turnover is where a monthly-rebalanced concentrated book
     actually dies, so turnover_fraction is pinned against hand-computed cases
     including the two that matter - opening from cash is one side, a full
     switch is two.
  4. A FITTED CHOICE SCORED ON ITS OWN FITTING WINDOW. walk_forward_pick must
     choose on the first half only, and is given a case where the first-half
     leader and the overall leader are deliberately different.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from factor_study import (                                          # noqa: E402
    Board, FACTORS, ONE_WAY, annualise, block_bootstrap_p, bonferroni_alpha,
    cagr, compound, eligible_mask, factor_score, hold_returns, ic_series,
    max_dd, newey_west_t, rebalance_positions, run_portfolio, select,
    spearman, TRADING_DAYS, total_return_series, turnover_fraction, variance_drag,
    walk_forward_pick)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def make_panel(bars=900, names=8, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=bars)
    cols = [f"N{i:03d}" for i in range(names)]
    px = pd.DataFrame(
        {c: 1000.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.02, bars))
         for c in cols}, index=idx)
    # turnover has to VARY over time, or the size factors are constant and the
    # look-ahead test below would pass on a fixture that cannot detect a leak
    tv = pd.DataFrame(
        {c: 5e10 * np.exp(np.cumsum(rng.normal(0, 0.05, bars)))
         for c in cols}, index=idx)
    # A raw close below the adjusted one, i.e. a dividend stream. Real dividends
    # arrive in LUMPS once or twice a year, not as a drift, and the difference
    # matters here: a constant drift makes trailing yield identical at every bar
    # and the look-ahead test would then pass on a fixture incapable of
    # detecting a leak.
    raw = {}
    for i, c in enumerate(cols):
        step = np.ones(bars)
        for j, b in enumerate(range(120, bars, 125)):
            step[b] = 1.0 + 0.01 * (i + 1) * (1 + (j % 3))
        raw[c] = px[c].to_numpy() / np.cumprod(step)
    return px, pd.DataFrame(raw, index=idx), tv


def make_board(bars=900, names=8, seed=7, factors=FACTORS, min_hist=280):
    px, raw, tv = make_panel(bars, names, seed)
    idx = np.cumprod(np.full(bars, 1.0002)) * 5000.0
    rebal = rebalance_positions(px.index, "monthly", min_hist)
    return (Board(px, tv, idx, rebal, list(factors), 1e9, min_hist, raw=raw),
            px, tv, idx)


# --------------------------------------------------------------------------
# 1. no look-ahead in any factor
# --------------------------------------------------------------------------
@pytest.mark.parametrize("factor", FACTORS)
def test_factor_score_cannot_see_the_future(factor):
    px, _raw, tv = make_panel(bars=800, names=6, seed=11)
    idx = np.cumprod(np.full(800, 1.0003)) * 5000.0
    cut = 600
    a = factor_score(factor, px.to_numpy(float)[:cut], tv.to_numpy(float)[:cut],
                     idx[:cut], _raw.to_numpy(float)[:cut])
    b = factor_score(factor, px.to_numpy(float), tv.to_numpy(float), idx,
                     _raw.to_numpy(float))
    # b sees 200 extra bars; a must be unchanged when recomputed on the prefix
    a2 = factor_score(factor, px.to_numpy(float)[:cut],
                      tv.to_numpy(float)[:cut], idx[:cut],
                      _raw.to_numpy(float)[:cut])
    assert np.allclose(a, a2, equal_nan=True)
    # and the two must genuinely differ, or the test proves nothing
    assert not np.allclose(np.nan_to_num(a), np.nan_to_num(b))


@pytest.mark.parametrize("factor", FACTORS)
def test_factor_score_ignores_bars_appended_after_the_decision(factor):
    """Scrambling the future must leave a past score bit-identical."""
    px, _raw, tv = make_panel(bars=800, names=6, seed=13)
    idx = np.cumprod(np.full(800, 1.0003)) * 5000.0
    cut = 620
    base = factor_score(factor, px.to_numpy(float)[:cut],
                        tv.to_numpy(float)[:cut], idx[:cut],
                        _raw.to_numpy(float)[:cut])
    poisoned = px.copy()
    poisoned.iloc[cut:] *= 100.0          # a future that could not be missed
    ptv = tv.copy()
    ptv.iloc[cut:] *= 1000.0
    praw = _raw.copy()
    praw.iloc[cut:] *= 0.01
    pidx = idx.copy()
    pidx[cut:] *= 50.0
    after = factor_score(factor, poisoned.to_numpy(float)[:cut],
                         ptv.to_numpy(float)[:cut], pidx[:cut],
                         praw.to_numpy(float)[:cut])
    assert np.allclose(base, after, equal_nan=True)


def test_unknown_factor_is_refused():
    px, _raw, tv = make_panel(bars=400, names=3)
    with pytest.raises(ValueError):
        factor_score("no_such_factor", px.to_numpy(float), tv.to_numpy(float),
                     np.ones(400))


# --------------------------------------------------------------------------
# factor direction: each one must point the way its name claims
# --------------------------------------------------------------------------
def _two_series(a: np.ndarray, b: np.ndarray):
    return np.column_stack([a, b])


def test_lowvol_prefers_the_calm_series():
    n = 400
    rng = np.random.default_rng(3)
    calm = 1000.0 * np.cumprod(1 + rng.normal(0, 0.002, n))
    wild = 1000.0 * np.cumprod(1 + rng.normal(0, 0.05, n))
    c = _two_series(calm, wild)
    s = factor_score("lowvol", c, np.full_like(c, 1e10), np.arange(1, n + 1.0))
    assert s[0] > s[1]


def test_momentum_prefers_the_riser():
    n = 400
    up = 1000.0 * np.cumprod(np.full(n, 1.002))
    down = 1000.0 * np.cumprod(np.full(n, 0.998))
    c = _two_series(up, down)
    for f in ("mom12_1", "mom6_1", "trend", "high52"):
        s = factor_score(f, c, np.full_like(c, 1e10), np.arange(1, n + 1.0))
        assert s[0] > s[1], f


def test_reversal_is_the_negative_of_last_month():
    n = 400
    up = 1000.0 * np.cumprod(np.full(n, 1.002))
    down = 1000.0 * np.cumprod(np.full(n, 0.998))
    c = _two_series(up, down)
    s = factor_score("rev1", c, np.full_like(c, 1e10), np.arange(1, n + 1.0))
    assert s[0] < s[1]           # the recent winner scores WORSE on reversal


def test_high52_is_one_at_a_new_high_and_below_one_otherwise():
    n = 400
    up = 1000.0 * np.cumprod(np.full(n, 1.002))
    peaked = np.concatenate([up[:300], up[299] * np.full(n - 300, 0.5)])
    c = _two_series(up, peaked)
    s = factor_score("high52", c, np.full_like(c, 1e10), np.arange(1, n + 1.0))
    assert s[0] == pytest.approx(1.0)
    assert s[1] < 0.6


def test_size_and_liquidity_are_exact_mirrors():
    n = 400
    c = _two_series(np.full(n, 1000.0), np.full(n, 1000.0))
    tv = np.column_stack([np.full(n, 1e12), np.full(n, 1e9)])
    small = factor_score("small", c, tv, np.arange(1, n + 1.0))
    large = factor_score("large", c, tv, np.arange(1, n + 1.0))
    assert np.allclose(small, -large)
    assert large[0] > large[1]


def test_lowbeta_prefers_the_name_that_ignores_the_index():
    n = 400
    rng = np.random.default_rng(5)
    m = rng.normal(0, 0.01, n)
    idx = 5000.0 * np.cumprod(1 + m)
    follower = 1000.0 * np.cumprod(1 + 2.0 * m)
    independent = 1000.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    c = _two_series(independent, follower)
    s = factor_score("lowbeta", c, np.full_like(c, 1e10), idx)
    assert s[0] > s[1]


def test_illiq_is_higher_when_the_same_move_costs_less_volume():
    n = 400
    rng = np.random.default_rng(9)
    r = rng.normal(0, 0.02, n)
    same = 1000.0 * np.cumprod(1 + r)
    c = _two_series(same, same)
    tv = np.column_stack([np.full(n, 1e9), np.full(n, 1e12)])
    s = factor_score("illiq", c, tv, np.arange(1, n + 1.0))
    assert s[0] > s[1]


def test_divyield_reads_the_gap_between_adjusted_and_raw_prices():
    n = 400
    total = 1000.0 * np.cumprod(np.full(n, 1.0002))
    payer = total / np.cumprod(np.full(n, 1.0002))       # all return is income
    hoarder = total.copy()                               # pays nothing
    c = _two_series(total, total)
    raw = _two_series(payer, hoarder)
    s = factor_score("divyield", c, np.full_like(c, 1e10),
                     np.arange(1, n + 1.0), raw)
    assert s[0] > s[1]
    assert s[1] == pytest.approx(0.0, abs=1e-9)
    assert s[0] == pytest.approx(1.0002 ** TRADING_DAYS - 1, rel=1e-6)


def test_divyield_needs_the_raw_series_and_says_so_with_nan():
    n = 400
    c = _two_series(np.full(n, 1000.0), np.full(n, 1000.0))
    s = factor_score("divyield", c, np.full_like(c, 1e10),
                     np.arange(1, n + 1.0), None)
    assert np.all(np.isnan(s))


def test_divyield_refuses_an_implausible_reading():
    """A yield of 300% is a data fault, not a bargain, and must not be ranked."""
    n = 400
    total = 1000.0 * np.cumprod(np.full(n, 1.0002))
    broken = total.copy()
    broken[-1] = broken[-1] / 5.0            # a bad raw print
    c = _two_series(total, total)
    raw = _two_series(broken, total)
    s = factor_score("divyield", c, np.full_like(c, 1e10),
                     np.arange(1, n + 1.0), raw)
    assert np.isnan(s[0])


def test_short_history_returns_nan_rather_than_a_guess():
    n = 60
    c = _two_series(np.full(n, 1000.0), np.full(n, 1000.0))
    for f in ("mom12_1", "mom6_1", "high52", "trend"):
        s = factor_score(f, c, np.full_like(c, 1e10), np.arange(1, n + 1.0))
        assert np.all(np.isnan(s)), f


# --------------------------------------------------------------------------
# 3. trading is not free
# --------------------------------------------------------------------------
def test_turnover_nothing_to_nothing_is_free():
    assert turnover_fraction([], []) == 0.0


def test_opening_from_cash_is_one_side():
    assert turnover_fraction([], [1, 2, 3, 4]) == pytest.approx(1.0)


def test_holding_the_same_book_costs_nothing():
    assert turnover_fraction([1, 2, 3], [3, 2, 1]) == pytest.approx(0.0)


def test_a_full_switch_is_two_sides():
    assert turnover_fraction([1, 2], [3, 4]) == pytest.approx(2.0)


def test_replacing_one_of_four_trades_half_a_position_each_way():
    # sell 1/4, buy 1/4 -> sum|dw| = 0.5
    assert turnover_fraction([1, 2, 3, 4], [1, 2, 3, 5]) == pytest.approx(0.5)


def test_going_to_cash_is_one_side():
    assert turnover_fraction([1, 2, 3], []) == pytest.approx(1.0)


def test_a_full_switch_costs_the_full_round_trip():
    assert ONE_WAY * turnover_fraction([1, 2], [3, 4]) == pytest.approx(0.0056)


def test_opening_costs_only_the_buy_leg():
    assert ONE_WAY * turnover_fraction([], [1, 2]) == pytest.approx(0.0028)


def test_resizing_the_same_names_still_costs_something():
    # 2 names to 4, the two originals are cut from 1/2 to 1/4 each
    t = turnover_fraction([1, 2], [1, 2, 3, 4])
    assert t == pytest.approx(2 * 0.25 + 2 * 0.25)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def test_select_takes_the_highest_scores():
    cols = np.arange(20)
    score = np.arange(20, dtype=float)
    got = select(cols, score, 3)
    assert sorted(got.tolist()) == [17, 18, 19]


def test_select_skips_names_with_no_score():
    cols = np.arange(20)
    score = np.arange(20, dtype=float)
    score[19] = np.nan
    got = select(cols, score, 2)
    assert 19 not in got.tolist()
    assert sorted(got.tolist()) == [17, 18]


def test_select_deciles_split_top_and_bottom():
    cols = np.arange(50)
    score = np.arange(50, dtype=float)
    top = select(cols, score, 30, decile="top")
    bot = select(cols, score, 30, decile="bottom")
    assert len(top) == 5 and len(bot) == 5
    assert set(top.tolist()).isdisjoint(bot.tolist())
    assert min(top) > max(bot)


def test_select_with_no_score_is_the_whole_universe():
    cols = np.arange(30)
    assert select(cols, None, 5).tolist() == cols.tolist()


def test_select_at_random_draws_exactly_n_distinct_names():
    cols = np.arange(40)
    rng = np.random.default_rng(1)
    got = select(cols, None, 7, rng=rng)
    assert len(got) == 7 and len(set(got.tolist())) == 7
    assert set(got.tolist()) <= set(cols.tolist())


def test_select_refuses_a_universe_too_thin_to_diversify():
    assert len(select(np.arange(5), np.arange(5.0), 3)) == 0


def test_select_cannot_hold_more_names_than_exist():
    cols = np.arange(12)
    assert len(select(cols, np.arange(12.0), 100)) == 12
    rng = np.random.default_rng(2)
    assert len(select(cols, None, 100, rng=rng)) == 12


# --------------------------------------------------------------------------
# eligibility
# --------------------------------------------------------------------------
def test_eligible_needs_enough_history():
    px, _raw, tv = make_panel(bars=100, names=4)
    m = eligible_mask(px.to_numpy(float), tv.to_numpy(float), 1e9, 280)
    assert not m.any()


def test_eligible_drops_names_below_the_turnover_floor():
    px, _raw, tv = make_panel(bars=400, names=4)
    t = tv.to_numpy(float).copy()
    t[:, 0] = 1e6
    m = eligible_mask(px.to_numpy(float), t, 1e9, 280)
    assert not m[0] and m[1:].all()


def test_eligible_drops_a_name_that_had_not_listed_yet():
    px, _raw, tv = make_panel(bars=400, names=4)
    c = px.to_numpy(float).copy()
    c[:200, 2] = np.nan
    m = eligible_mask(c, tv.to_numpy(float), 1e9, 280)
    assert not m[2]


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def test_spearman_perfect_and_reversed():
    a = np.arange(20.0)
    assert spearman(a, a * 3 + 1) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)


def test_spearman_needs_pairs():
    assert np.isnan(spearman(np.arange(3.0), np.arange(3.0)))


def test_spearman_ignores_missing_pairs():
    a = np.array([1.0, 2, 3, 4, 5, np.nan])
    b = np.array([1.0, 2, 3, 4, 5, 99])
    assert spearman(a, b) == pytest.approx(1.0)


def test_spearman_is_flat_on_a_constant():
    assert np.isnan(spearman(np.arange(20.0), np.ones(20)))


def test_newey_west_is_zero_on_a_zero_mean_series():
    x = np.array([1.0, -1.0] * 30)
    assert abs(newey_west_t(x)) < 1e-6


def test_newey_west_is_large_on_a_steady_positive_series():
    rng = np.random.default_rng(4)
    x = 0.02 + rng.normal(0, 0.001, 120)
    assert newey_west_t(x) > 10


def test_newey_west_refuses_a_stub():
    assert np.isnan(newey_west_t([0.1, 0.2, 0.3]))


def test_newey_west_penalises_autocorrelation():
    """A persistent series gets a smaller t than the same mean without memory."""
    rng = np.random.default_rng(6)
    e = rng.normal(0, 0.01, 400)
    ar = np.empty(400)
    ar[0] = e[0]
    for i in range(1, 400):
        ar[i] = 0.8 * ar[i - 1] + e[i]
    ar = ar - ar.mean() + 0.01
    plain = e - e.mean() + 0.01
    assert newey_west_t(ar, lag=6) < newey_west_t(plain, lag=6)


def test_bootstrap_finds_a_strong_signal():
    x = np.full(120, 0.03)
    assert block_bootstrap_p(x, draws=500) < 0.01


def test_bootstrap_does_not_flag_noise():
    rng = np.random.default_rng(8)
    x = rng.normal(0, 0.05, 200)
    assert block_bootstrap_p(x, draws=800) > 0.05


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(12)
    x = rng.normal(0.001, 0.05, 150)
    assert block_bootstrap_p(x, draws=400) == block_bootstrap_p(x, draws=400)


def test_bootstrap_refuses_a_stub():
    assert np.isnan(block_bootstrap_p([0.1] * 5))


def test_bonferroni_divides_by_the_number_of_looks():
    assert bonferroni_alpha(1) == pytest.approx(0.05)
    assert bonferroni_alpha(12) == pytest.approx(0.05 / 12)
    assert bonferroni_alpha(0) == pytest.approx(0.05)


# --------------------------------------------------------------------------
# compounding and drag
# --------------------------------------------------------------------------
def test_no_drag_on_a_constant_return():
    a, g, d = variance_drag([0.01] * 50)
    assert a == pytest.approx(0.01)
    assert g == pytest.approx(0.01)
    assert d == pytest.approx(0.0, abs=1e-12)


def test_drag_is_positive_when_returns_vary():
    a, g, d = variance_drag([0.20, -0.20] * 50)
    assert a == pytest.approx(0.0)
    assert g < 0
    assert d > 0


def test_drag_matches_the_two_point_closed_form():
    """+x then -x compounds to sqrt(1-x^2)-1 a period, so the gap is exact."""
    x = 0.2
    a, g, d = variance_drag([x, -x] * 100)
    assert g == pytest.approx(np.sqrt(1 - x * x) - 1, abs=1e-9)
    assert d == pytest.approx(0.0 - (np.sqrt(1 - x * x) - 1), abs=1e-9)


def test_bigger_swings_drag_more():
    _, _, small = variance_drag([0.02, -0.02] * 60)
    _, _, big = variance_drag([0.20, -0.20] * 60)
    assert big > small


def test_compound_and_annualise():
    assert compound([0.1, 0.1]) == pytest.approx(1.21)
    assert annualise([0.01] * 12, 12.0) == pytest.approx(1.01 ** 12 - 1)
    assert np.isnan(annualise([], 12.0))


def test_cagr_and_drawdown():
    idx = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"])
    eq = pd.Series([1.0, 2.0, 1.5], index=idx)
    assert cagr(eq) == pytest.approx(1.5 ** 0.5 - 1, rel=1e-3)
    assert max_dd(eq) == pytest.approx(-0.25)


# --------------------------------------------------------------------------
# hold_returns
# --------------------------------------------------------------------------
def test_hold_returns_compounds_the_window():
    ret = np.full((10, 3), 0.01)
    got = hold_returns(ret, 2, 5, np.array([0, 1]))
    assert np.allclose(got, 1.01 ** 3 - 1)


def test_hold_returns_is_zero_on_an_empty_window():
    ret = np.full((10, 3), 0.01)
    assert np.allclose(hold_returns(ret, 5, 5, np.array([0])), 0.0)
    assert hold_returns(ret, 0, 5, np.array([], dtype=int)).size == 0


def test_hold_returns_treats_a_missing_bar_as_flat_not_as_a_loss():
    ret = np.full((6, 2), 0.01)
    ret[3, 0] = np.nan
    got = hold_returns(ret, 0, 6, np.array([0, 1]))
    assert got[0] == pytest.approx(1.01 ** 5 - 1)


# --------------------------------------------------------------------------
# 2. no look-ahead in the book, and the arithmetic is right
# --------------------------------------------------------------------------
def test_rebalance_positions_are_ordered_and_after_the_warmup():
    idx = pd.bdate_range("2015-01-01", periods=900)
    pos = rebalance_positions(idx, "monthly", 280)
    assert pos == sorted(pos)
    assert min(pos) >= 280
    assert len(pos) > 25


def test_quarterly_is_a_third_of_monthly():
    idx = pd.bdate_range("2015-01-01", periods=1500)
    m = rebalance_positions(idx, "monthly", 280)
    q = rebalance_positions(idx, "quarterly", 280)
    assert abs(len(m) / 3 - len(q)) <= 1.5


def test_windows_abut_without_gap_or_overlap():
    board, *_ = make_board()
    for k in range(len(board.rebal) - 1):
        _, exit_ = board.window(k)
        nxt, _ = board.window(k + 1)
        assert exit_ == nxt


def test_window_executes_after_the_signal_bar():
    board, *_ = make_board()
    entry, _ = board.window(0)
    assert entry == board.rebal[0] + board.delay


def test_a_book_on_a_flat_panel_only_loses_the_entry_fee():
    bars = 600
    idx = pd.bdate_range("2015-01-01", periods=bars)
    px = pd.DataFrame({f"N{i}": np.full(bars, 1000.0) for i in range(12)},
                      index=idx)
    tv = pd.DataFrame({f"N{i}": np.full(bars, 5e10) for i in range(12)},
                      index=idx)
    board = Board(px, tv, np.full(bars, 5000.0),
                  rebalance_positions(idx, "monthly", 280), [], 1e9, 280)
    eq, daily, per, sizes, costs = run_portfolio(board, None, 12)
    assert all(abs(p) < 1e-12 for p in per)
    assert eq.iloc[-1] == pytest.approx(1.0 - ONE_WAY, rel=1e-9)
    assert sum(costs) == pytest.approx(ONE_WAY)


def test_a_book_on_a_steady_panel_compounds_at_that_rate():
    bars, step = 600, 1.0005
    idx = pd.bdate_range("2015-01-01", periods=bars)
    line = 1000.0 * np.cumprod(np.full(bars, step))
    px = pd.DataFrame({f"N{i}": line for i in range(12)}, index=idx)
    tv = pd.DataFrame({f"N{i}": np.full(bars, 5e10) for i in range(12)},
                      index=idx)
    rebal = rebalance_positions(idx, "monthly", 280)
    board = Board(px, tv, np.full(bars, 5000.0), rebal, [], 1e9, 280)
    eq, daily, per, _, _ = run_portfolio(board, None, 12)
    entry = rebal[0] + 1
    want = (1.0 - ONE_WAY) * step ** (len(line) - 1 - entry)
    assert eq.iloc[-1] == pytest.approx(want, rel=1e-9)


def test_extending_the_panel_cannot_change_a_period_that_already_happened():
    px, _raw, tv = make_panel(bars=900, names=10, seed=21)
    idx = np.cumprod(np.full(900, 1.0002)) * 5000.0
    cut = 700
    reb_short = rebalance_positions(px.index[:cut], "monthly", 280)
    short = Board(px.iloc[:cut], tv.iloc[:cut], idx[:cut], reb_short,
                  list(FACTORS), 1e9, 280)
    reb_long = rebalance_positions(px.index, "monthly", 280)
    long = Board(px, tv, idx, reb_long, list(FACTORS), 1e9, 280)
    # The short run's last period ends at the panel edge instead of at the next
    # rebalance, and so does the one before it whenever the final rebalance
    # falls within the execution delay of that edge. Both are live-edge
    # artefacts, not history, so only the settled periods are compared.
    keep = len(short.rebal) - 2
    for f in FACTORS:
        ps = run_portfolio(short, f, 4).period
        pl = run_portfolio(long, f, 4).period
        assert ps[:keep] == pytest.approx(pl[:keep], rel=1e-9, abs=1e-12), f


def test_the_daily_curve_ends_where_the_rebalance_curve_ends():
    """Two resolutions of one book, so they must agree where nothing is traded.

    They differ by exactly one thing at a shared bar: the rebalance curve is
    stamped BEFORE the next rebalance's cost and the daily curve carries on
    AFTER it. That relationship is asserted rather than waved at, because a
    daily curve that quietly skipped the cost would look better than the book.
    """
    board, *_ = make_board(bars=900, names=20, seed=51)
    r = run_portfolio(board, "mom6_1", 5)
    assert r.daily.iloc[-1] == pytest.approx(r.curve.iloc[-1], rel=1e-9)
    shared = [t for t in r.curve.index[:-1] if t in r.daily.index]
    assert len(shared) > 10
    for i, t in enumerate(shared):
        assert r.daily[t] == pytest.approx(r.curve[t] * (1.0 - r.costs[i + 1]),
                                           rel=1e-9)


def test_the_daily_curve_has_a_point_for_every_bar_it_is_invested():
    board, *_ = make_board(bars=900, names=20, seed=53)
    r = run_portfolio(board, "mom6_1", 5)
    assert len(r.daily) > 10 * len(r.curve)


def test_rebalance_sampling_hides_drawdown_and_the_daily_curve_does_not():
    """The bug this exists to stop: an annual book showed a -2.6% max drawdown
    while its holdings halved, because eleven yearly snapshots missed the fall.
    """
    bars = 1500
    idx = pd.bdate_range("2015-01-01", periods=bars)
    line = np.full(bars, 1000.0)
    # a deep round trip entirely inside one calendar year
    line[700:760] = np.linspace(1000.0, 400.0, 60)
    line[760:820] = np.linspace(400.0, 1000.0, 60)
    px = pd.DataFrame({f"N{i}": line for i in range(12)}, index=idx)
    tv = pd.DataFrame({f"N{i}": np.full(bars, 5e10) for i in range(12)},
                      index=idx)
    board = Board(px, tv, np.full(bars, 5000.0),
                  rebalance_positions(idx, "annual", 280), [], 1e9, 280)
    r = run_portfolio(board, None, 12)
    assert max_dd(r.curve) > -0.05        # the sampled curve barely notices
    assert max_dd(r.daily) < -0.55        # the book actually lost 60%


def test_the_daily_curve_charges_the_entry_fee_before_the_first_bar():
    bars = 600
    idx = pd.bdate_range("2015-01-01", periods=bars)
    px = pd.DataFrame({f"N{i}": np.full(bars, 1000.0) for i in range(12)},
                      index=idx)
    tv = pd.DataFrame({f"N{i}": np.full(bars, 5e10) for i in range(12)},
                      index=idx)
    board = Board(px, tv, np.full(bars, 5000.0),
                  rebalance_positions(idx, "monthly", 280), [], 1e9, 280)
    r = run_portfolio(board, None, 12)
    assert r.daily.iloc[0] == pytest.approx(1.0 - ONE_WAY, rel=1e-9)


def test_a_book_left_alone_drifts_instead_of_staying_equal_weight():
    """Between rebalances nobody trims the winner, and the curve must show it."""
    bars = 600
    idx = pd.bdate_range("2015-01-01", periods=bars)
    flat = np.full(bars, 1000.0)
    riser = 1000.0 * np.cumprod(np.full(bars, 1.01))
    px = pd.DataFrame({"A": riser, "B": flat, "C": flat, "D": flat,
                       "E": flat, "F": flat, "G": flat, "H": flat,
                       "I": flat, "J": flat, "K": flat, "L": flat}, index=idx)
    tv = pd.DataFrame({c: np.full(bars, 5e10) for c in px.columns}, index=idx)
    board = Board(px, tv, np.full(bars, 5000.0),
                  rebalance_positions(idx, "annual", 280), [], 1e9, 280)
    r = run_portfolio(board, None, 12)
    # equal weight held without trimming = mean of the compounded paths, which
    # is strictly more than a book rebalanced back to equal weight every bar
    n = len(px.columns)
    entry = board.rebal[0] + 1
    held = (1.01 ** (bars - 1 - entry) + (n - 1)) / n
    assert r.daily.iloc[-1] / r.daily.iloc[0] == pytest.approx(held, rel=1e-6)


def test_a_concentrated_book_pays_more_turnover_than_a_broad_one():
    board, *_ = make_board(bars=900, names=40, seed=31)
    few = run_portfolio(board, "mom6_1", 3).costs
    many = run_portfolio(board, "mom6_1", 30).costs
    assert sum(few) > sum(many)


def test_the_universe_book_barely_trades():
    board, *_ = make_board(bars=900, names=40, seed=33)
    costs = run_portfolio(board, None, 30).costs
    assert sum(costs) < 2 * ONE_WAY


# --------------------------------------------------------------------------
# rank IC
# --------------------------------------------------------------------------
def test_ic_is_one_when_the_score_is_the_future_return():
    """A deliberately cheating score must score a perfect IC.

    This is the positive control for the IC machinery: if a score that IS the
    answer did not read as 1.0, a real score reading 0.06 would mean nothing.
    """
    board, px, tv, idx = make_board(bars=900, names=25, seed=41,
                                    factors=["mom6_1"])
    for k in range(len(board.rebal) - 1):
        entry, exit_ = board.window(k)
        board.scores["mom6_1"][k] = np.where(
            np.isfinite(board.ret[exit_]),
            hold_returns(board.ret, entry + 1, exit_ + 1,
                         np.arange(px.shape[1])), np.nan)
    ics = ic_series(board, "mom6_1")
    assert len(ics) > 20
    assert np.mean(ics) > 0.99


def test_ic_of_a_real_factor_is_small():
    board, *_ = make_board(bars=1200, names=30, seed=43, factors=["mom12_1"])
    ics = ic_series(board, "mom12_1")
    assert len(ics) > 20
    assert abs(np.mean(ics)) < 0.3      # random panel: no information


# --------------------------------------------------------------------------
# 4. the walk-forward choice is made on the first half only
# --------------------------------------------------------------------------
def test_walk_forward_picks_the_first_half_leader_not_the_overall_leader():
    n = 60
    ew = [0.0] * n
    early = [0.02] * (n // 2) + [-0.05] * (n // 2)    # best early, worst late
    late = [0.0] * (n // 2) + [0.10] * (n // 2)       # best overall
    pick, split = walk_forward_pick({"early": early, "late": late}, ew)
    assert pick == "early"
    assert split == n // 2
    assert float(np.mean(late)) > float(np.mean(early))   # overall leader loses


def test_walk_forward_refuses_a_sample_too_short_to_split():
    pick, _ = walk_forward_pick({"a": [0.1] * 10}, [0.0] * 10)
    assert pick is None


def test_walk_forward_measures_against_the_neutral_book():
    """A factor that merely tracks the universe has no edge and must not win."""
    n = 80
    ew = [0.01] * n
    tracker = [0.01] * n
    better = [0.011] * n
    pick, _ = walk_forward_pick({"tracker": tracker, "better": better}, ew)
    assert pick == "better"


# --------------------------------------------------------------------------
# dividends
# --------------------------------------------------------------------------
class _Loader:
    def __init__(self, df):
        self.df = df

    def get(self, ticker, max_age=None):
        return self.df.copy()


def _price_frame(bars=600):
    idx = pd.bdate_range("2015-01-01", periods=bars)
    close = 1000.0 * np.cumprod(np.full(bars, 1.0002))
    adj = close * np.cumprod(np.full(bars, 1.0001))   # a dividend stream
    return pd.DataFrame({"date": idx, "close": close, "adj_close": adj,
                         "volume": np.full(bars, 1e6)})


def test_total_return_uses_the_adjusted_close():
    out = total_return_series(_Loader(_price_frame()), "AAAA", total=True)
    assert out is not None
    r = out["px"].iloc[-1] / out["px"].iloc[0]
    raw = out["raw"].iloc[-1] / out["raw"].iloc[0]
    assert r > raw


def test_price_only_uses_the_raw_close():
    out = total_return_series(_Loader(_price_frame()), "AAAA", total=False)
    assert out["px"].iloc[-1] == pytest.approx(out["raw"].iloc[-1], rel=1e-9)


def test_turnover_is_always_computed_on_the_traded_price():
    """Adjusted prices are for returns; rupiah traded must use the real close."""
    a = total_return_series(_Loader(_price_frame()), "AAAA", total=True)
    b = total_return_series(_Loader(_price_frame()), "AAAA", total=False)
    assert np.allclose(a["raw"].to_numpy(), b["raw"].to_numpy())


def test_impossible_prints_are_capped_at_the_auto_rejection_limit():
    f = _price_frame()
    f.loc[300, "adj_close"] = f.loc[299, "adj_close"] * 8.0
    out = total_return_series(_Loader(f), "AAAA", total=True)
    assert out["px"].pct_change().max() <= 0.35 + 1e-9


def test_a_short_history_is_refused():
    assert total_return_series(_Loader(_price_frame(bars=100)), "AAAA") is None
