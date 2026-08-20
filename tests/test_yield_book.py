"""Tests for the yield-book robustness pass.

The whole case for this book rests on one measurement - that the adj_close /
close ratio is the accumulated dividend and that its trailing growth is
therefore a point-in-time yield. If that is wrong the book is built on nothing,
so it is pinned here against hand-built dividend streams with a known answer,
and checked once more for look-ahead at every lookback.

The other three tests are for the claims the report makes ABOUT the book:
that the characteristic persists (score_persistence), that a boom cannot hide
inside a CAGR (yearly), and that the gross number is not quietly presented as
what a taxpayer keeps (net_of_tax).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from factor_study import Board, rebalance_positions       # noqa: E402
from yield_book import (DIV_TAX, net_of_tax, score_persistence,   # noqa: E402
                        yearly, yield_scores)


def payer_board(bars=1500, names=6, rate_per_name=(0.0, 0.02, 0.04, 0.06,
                                                   0.08, 0.10), min_hist=280):
    """A panel where name i pays a known dividend yield, and nothing else moves.

    Price is flat and the whole return is income, so the trailing yield the
    scorer recovers has an exact right answer to be checked against.
    """
    idx = pd.bdate_range("2015-01-01", periods=bars)
    cols = [f"N{i}" for i in range(names)]
    raw = pd.DataFrame({c: np.full(bars, 1000.0) for c in cols}, index=idx)
    total = pd.DataFrame(
        {c: 1000.0 * np.cumprod(np.full(bars, (1.0 + rate_per_name[i])
                                        ** (1 / 250.0)))
         for i, c in enumerate(cols)}, index=idx)
    tv = pd.DataFrame({c: np.full(bars, 5e10) for c in cols}, index=idx)
    rebal = rebalance_positions(idx, "monthly", min_hist)
    return Board(total, tv, np.full(bars, 5000.0), rebal, [], 1e9, min_hist,
                 raw=raw), total, raw


# --------------------------------------------------------------------------
# the measurement the whole book rests on
# --------------------------------------------------------------------------
def test_the_recovered_yield_is_the_dividend_that_was_paid():
    rates = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)
    board, _, _ = payer_board(rate_per_name=rates)
    s = yield_scores(board, 250)[-1]
    assert np.allclose(s, rates, atol=1e-6)


def test_a_name_that_pays_nothing_scores_zero_not_missing():
    board, _, _ = payer_board(rate_per_name=(0.0,) * 6)
    s = yield_scores(board, 250)[-1]
    assert np.allclose(s, 0.0, atol=1e-9)


def test_the_ranking_is_the_yield_ranking():
    rates = (0.09, 0.01, 0.05, 0.00, 0.03, 0.07)
    board, _, _ = payer_board(rate_per_name=rates)
    s = yield_scores(board, 250)[-1]
    assert list(np.argsort(-s)) == list(np.argsort(-np.array(rates)))


def test_every_lookback_recovers_the_same_annual_rate():
    """Annualising is what puts 6-month and 2-year books on one scale."""
    rates = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)
    board, _, _ = payer_board(bars=1800, rate_per_name=rates)
    for lb in (125, 250, 500):
        s = yield_scores(board, lb)[-1]
        assert np.allclose(s, rates, atol=2e-3), lb


def test_yield_scores_cannot_see_the_future():
    rates = (0.0, 0.03, 0.06, 0.09, 0.12, 0.15)
    board, total, raw = payer_board(bars=1500, rate_per_name=rates)
    cut = 1100
    reb = [b for b in board.rebal if b < cut]
    short = Board(total.iloc[:cut], pd.DataFrame(
        {c: np.full(cut, 5e10) for c in total.columns},
        index=total.index[:cut]), np.full(cut, 5000.0), reb, [], 1e9, 280,
        raw=raw.iloc[:cut])
    a = yield_scores(short, 250)
    b = yield_scores(board, 250)
    for k in range(len(a)):
        assert np.allclose(a[k], b[k], equal_nan=True), k


def test_a_lookback_longer_than_the_history_returns_missing_not_zero():
    board, _, _ = payer_board(bars=800)
    s = yield_scores(board, 5000)
    assert all(np.all(np.isnan(x)) for x in s)


def test_an_impossible_yield_is_refused_rather_than_ranked_first():
    """A 300% reading is a corporate action or a bad print, not a bargain."""
    bars = 1500
    idx = pd.bdate_range("2015-01-01", periods=bars)
    raw = pd.DataFrame({"A": np.full(bars, 1000.0),
                        "B": np.full(bars, 1000.0)}, index=idx)
    total = pd.DataFrame({"A": np.full(bars, 1000.0),
                          "B": np.full(bars, 1000.0)}, index=idx)
    total.loc[total.index[1400:], "A"] = 5000.0      # a 400% "yield"
    tv = pd.DataFrame({c: np.full(bars, 5e10) for c in raw.columns}, index=idx)
    board = Board(total, tv, np.full(bars, 5000.0),
                  rebalance_positions(idx, "monthly", 280), [], 1e9, 280,
                  raw=raw)
    s = yield_scores(board, 250)[-1]
    assert np.isnan(s[0])
    assert s[1] == pytest.approx(0.0)


def test_a_negative_reading_is_refused():
    """The adjustment factor cannot shrink; if it did, the data is wrong."""
    bars = 1500
    idx = pd.bdate_range("2015-01-01", periods=bars)
    raw = pd.DataFrame({"A": np.full(bars, 1000.0)}, index=idx)
    total = pd.DataFrame({"A": np.full(bars, 1000.0)}, index=idx)
    total.loc[total.index[1400:], "A"] = 700.0
    tv = pd.DataFrame({"A": np.full(bars, 5e10)}, index=idx)
    board = Board(total, tv, np.full(bars, 5000.0),
                  rebalance_positions(idx, "monthly", 280), [], 1e9, 280,
                  raw=raw)
    assert np.isnan(yield_scores(board, 250)[-1][0])


# --------------------------------------------------------------------------
# does the characteristic stick around
# --------------------------------------------------------------------------
def test_a_stable_characteristic_scores_full_persistence():
    scores = [np.arange(50.0) for _ in range(30)]
    cols = [np.arange(50) for _ in range(30)]
    assert score_persistence(scores, cols, 12) == pytest.approx(1.0)


def test_a_reshuffled_characteristic_scores_none():
    rng = np.random.default_rng(3)
    scores = [rng.permutation(50).astype(float) for _ in range(40)]
    cols = [np.arange(50) for _ in range(40)]
    assert abs(score_persistence(scores, cols, 12)) < 0.3


def test_persistence_only_compares_names_present_in_both_periods():
    scores = [np.arange(50.0) for _ in range(20)]
    cols = [np.arange(50)] * 10 + [np.arange(25, 50)] * 10
    v = score_persistence(scores, cols, 5)
    assert np.isfinite(v)
    assert v == pytest.approx(1.0)


def test_persistence_of_too_short_a_run_is_undefined():
    assert np.isnan(score_persistence([np.arange(50.0)] * 3,
                                      [np.arange(50)] * 3, 12))


# --------------------------------------------------------------------------
# calendar years
# --------------------------------------------------------------------------
def test_yearly_splits_a_doubling_into_its_years():
    idx = pd.to_datetime(["2020-01-31", "2020-12-31", "2021-12-31"])
    eq = pd.Series([1.0, 2.0, 3.0], index=idx)
    y = yearly(eq)
    assert y.loc["2020-12-31"] == pytest.approx(1.0)
    assert y.loc["2021-12-31"] == pytest.approx(0.5)


def test_yearly_counts_the_first_partial_year_from_where_the_book_started():
    idx = pd.to_datetime(["2020-06-30", "2020-12-31"])
    eq = pd.Series([1.0, 1.2], index=idx)
    assert yearly(eq).iloc[0] == pytest.approx(0.2)


def test_yearly_of_a_stub_is_empty():
    idx = pd.to_datetime(["2020-06-30"])
    assert yearly(pd.Series([1.0], index=idx)).empty


# --------------------------------------------------------------------------
# tax
# --------------------------------------------------------------------------
def test_only_the_income_part_is_taxed():
    # 10% total return of which 6% is dividend: the haircut is 0.6%, not 1%
    assert net_of_tax(0.10, 0.06, 0.10) == pytest.approx(0.094)


def test_a_book_with_no_income_pays_no_dividend_tax():
    assert net_of_tax(0.10, 0.0) == pytest.approx(0.10)


def test_a_foreign_holder_keeps_less():
    assert net_of_tax(0.10, 0.06, 0.20) < net_of_tax(0.10, 0.06, DIV_TAX)


def test_the_default_rate_is_the_domestic_final_rate():
    assert net_of_tax(0.10, 0.05) == pytest.approx(0.10 - 0.05 * DIV_TAX)
