"""Cross-sectional evaluation metrics.

These tests use constructed signals with a known relationship to forward
returns, so the metrics can be checked against an answer that is known in
advance rather than merely "looking plausible".
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.evaluate import (  # noqa: E402
    _spearman,
    component_scan,
    quantile_spread,
    rank_ic,
    split_sample,
)


def _panel(n_dates=200, n_tickers=20, relationship="positive", seed=3):
    """A synthetic panel where the score's true relationship is known."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    rows = []
    for date in dates:
        scores = rng.uniform(0, 100, n_tickers)
        noise = rng.normal(0, 0.05, n_tickers)
        if relationship == "positive":
            fwd = scores / 100 * 0.10 + noise
        elif relationship == "negative":
            fwd = -scores / 100 * 0.10 + noise
        else:
            fwd = noise
        for t in range(n_tickers):
            rows.append({"date": date, "ticker": f"T{t:02d}",
                         "score": scores[t], "fwd_20": fwd[t]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- spearman ---
def test_spearman_perfect_and_inverse():
    a = pd.Series([1.0, 2, 3, 4, 5])
    assert _spearman(a, a) == pytest.approx(1.0)
    assert _spearman(a, -a) == pytest.approx(-1.0)


def test_spearman_is_rank_based_not_linear():
    a = pd.Series([1.0, 2, 3, 4, 5])
    assert _spearman(a, a ** 3) == pytest.approx(1.0)   # monotone but nonlinear


def test_spearman_degenerate_inputs():
    assert np.isnan(_spearman(pd.Series([1.0, 2]), pd.Series([1.0, 2])))       # too few
    assert np.isnan(_spearman(pd.Series([1.0] * 10), pd.Series(range(10))))    # zero variance


# ----------------------------------------------------------------- rank IC ---
def test_rank_ic_detects_a_positive_relationship():
    ic = rank_ic(_panel(relationship="positive"), horizons=(20,))
    assert len(ic) == 1
    row = ic.iloc[0]
    assert row["mean_ic"] > 0.3
    assert row["t_stat"] > 5
    assert row["pct_positive"] > 0.8


def test_rank_ic_detects_a_negative_relationship():
    row = rank_ic(_panel(relationship="negative"), horizons=(20,)).iloc[0]
    assert row["mean_ic"] < -0.3
    assert row["t_stat"] < -5


def test_rank_ic_reports_nothing_for_pure_noise():
    """A signal with no relationship must not produce a significant t."""
    row = rank_ic(_panel(relationship="none"), horizons=(20,)).iloc[0]
    assert abs(row["mean_ic"]) < 0.1
    assert abs(row["t_stat"]) < 3


def test_rank_ic_skips_dates_with_too_few_names():
    """A rank correlation over three names is noise, not information."""
    panel = _panel(n_tickers=3)
    assert rank_ic(panel, horizons=(20,), min_names=8).empty


def test_rank_ic_missing_column():
    assert rank_ic(_panel(), signal_col="does_not_exist", horizons=(20,)).empty
    assert rank_ic(pd.DataFrame(), horizons=(20,)).empty


# --------------------------------------------------------- quantile spread ---
def test_quantile_spread_positive_when_top_outperforms():
    row = quantile_spread(_panel(relationship="positive"), horizons=(20,)).iloc[0]
    assert row["spread"] > 0
    assert row["top_mean"] > row["bottom_mean"]
    assert row["t_stat"] > 5


def test_quantile_spread_negative_when_inverted():
    row = quantile_spread(_panel(relationship="negative"), horizons=(20,)).iloc[0]
    assert row["spread"] < 0
    assert row["t_stat"] < -5


# --------------------------------------------------------------- splitting ---
def test_split_sample_is_chronological_and_disjoint():
    panel = _panel()
    train, test = split_sample(panel, 0.5)
    assert len(train) + len(test) == len(panel)
    # No future data may appear in the training set.
    assert train["date"].max() < test["date"].min()


def test_split_sample_fraction_respected():
    train, test = split_sample(_panel(), 0.8)
    assert len(train) > len(test)


def test_split_sample_empty():
    empty = pd.DataFrame()
    a, b = split_sample(empty)
    assert a.empty and b.empty


# ----------------------------------------------------------- component scan --
def test_component_scan_ranks_components():
    panel = _panel(relationship="positive")
    panel["c_good"] = panel["score"]
    panel["c_bad"] = -panel["score"]
    scan = component_scan(panel, horizons=(20,))
    assert set(scan["signal"]) == {"good", "bad"}
    good = scan[scan["signal"] == "good"].iloc[0]
    bad = scan[scan["signal"] == "bad"].iloc[0]
    assert good["mean_ic"] > 0 > bad["mean_ic"]
    # Output must be sorted best-first.
    assert scan.iloc[0]["signal"] == "good"


def test_component_scan_without_components():
    assert component_scan(_panel(), horizons=(20,)).empty
