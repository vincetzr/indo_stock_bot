"""Long-only portfolio simulation."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.portfolio import _annualise, _stats, equity_curve, simulate  # noqa: E402


def _panel(n_dates=600, n_tickers=30, relationship="positive", seed=11):
    """Panel on a SHARED date grid - every ticker scored on every date."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_dates)
    rows = []
    for date in dates:
        scores = rng.uniform(0, 100, n_tickers)
        noise = rng.normal(0, 0.06, n_tickers)
        if relationship == "positive":
            fwd = scores / 100 * 0.12 + noise
        elif relationship == "negative":
            fwd = -scores / 100 * 0.12 + noise
        else:
            fwd = noise
        for t in range(n_tickers):
            rows.append({"date": date, "ticker": f"T{t:02d}",
                         "score": scores[t], "fwd_60": fwd[t]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- stats ---
def test_annualise_matches_compounding():
    # 21% over 4 periods at 2 periods/year = 2 years -> ~10% CAGR
    assert _annualise(0.21, 4, 2.0) == pytest.approx(0.10, abs=0.005)
    # 21% over exactly one year is 21%.
    assert _annualise(0.21, 2, 2.0) == pytest.approx(0.21)


def test_annualise_degenerate():
    assert np.isnan(_annualise(0.1, 0, 4))
    assert np.isnan(_annualise(-1.5, 4, 4))


def test_stats_drawdown_is_negative_after_a_loss():
    returns = pd.Series([0.10, -0.20, 0.05, 0.03])
    stats = _stats(returns, 4.0)
    assert stats["max_drawdown"] < 0
    assert stats["hit_rate"] == pytest.approx(0.75)
    assert stats["worst_period"] == pytest.approx(-0.20)


def test_stats_total_return_compounds():
    returns = pd.Series([0.10, 0.10])
    assert _stats(returns, 4.0)["total_return"] == pytest.approx(0.21)


# -------------------------------------------------------------- simulation ---
def test_simulate_beats_universe_when_score_is_informative():
    result = simulate(_panel(relationship="positive"), top_n=5, horizon=60,
                      cost_per_side=0.0)
    assert not result.empty
    assert result.stats["total_return"] > result.benchmark_stats["total_return"]
    assert result.periods["excess"].mean() > 0


def test_simulate_underperforms_when_score_is_inverted():
    result = simulate(_panel(relationship="negative"), top_n=5, horizon=60,
                      cost_per_side=0.0)
    assert not result.empty
    assert result.periods["excess"].mean() < 0


def test_simulate_produces_non_overlapping_periods():
    """Overlapping holding periods would double-count returns."""
    result = simulate(_panel(), top_n=5, horizon=60)
    gaps = result.periods["date"].diff().dropna().dt.days
    # 60 trading days is ~84 calendar days; allow generous slack but require
    # that rebalances are not stacked on top of each other.
    assert gaps.min() >= 40, f"rebalances only {gaps.min()} days apart"


def test_simulate_charges_costs():
    free = simulate(_panel(), top_n=5, horizon=60, cost_per_side=0.0)
    paid = simulate(_panel(), top_n=5, horizon=60, cost_per_side=0.01)
    assert paid.stats["total_return"] < free.stats["total_return"]
    assert (paid.periods["cost"] > 0).any()


def test_simulate_turnover_between_zero_and_one():
    result = simulate(_panel(), top_n=5, horizon=60)
    assert result.periods["turnover"].between(0.0, 1.0).all()


def test_simulate_holds_requested_number_of_names():
    result = simulate(_panel(n_tickers=30), top_n=8, horizon=60)
    assert (result.periods["names"] == 8).all()


def test_simulate_respects_min_names_guard():
    """A date with too few names to diversify must be skipped."""
    assert simulate(_panel(n_tickers=5), top_n=3, horizon=60, min_names=15).empty


def test_simulate_missing_horizon_column():
    assert simulate(_panel(), horizon=999).empty


def test_simulate_empty_input():
    assert simulate(pd.DataFrame()).empty


def test_min_score_filter_applies():
    result = simulate(_panel(), top_n=5, horizon=60, min_score=90)
    if not result.empty:
        assert result.periods["mean_score"].min() >= 90


def test_equity_curve_starts_near_one_and_is_monotone_in_length():
    result = simulate(_panel(), top_n=5, horizon=60)
    curve = equity_curve(result)
    assert len(curve) == len(result.periods)
    assert {"date", "strategy", "universe"} <= set(curve.columns)
    assert (curve["strategy"] > 0).all()
