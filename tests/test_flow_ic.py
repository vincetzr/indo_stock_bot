"""Tests for the §7 information-coefficient machinery.

Three things here can be wrong in ways that produce a beautiful, publishable,
false result, and each has a test that would catch it:

  NEUTRALISATION THAT DOES NOT NEUTRALISE. If the residual still carries the
  control, a momentum signal wearing a flow costume passes Gate 1. The test
  plants a score that IS momentum and requires the neutralised IC to collapse.

  A FACTOR MODEL FITTED ON THE FUTURE. Statistical factors stand in for the
  sector control, and the obvious implementation - fit on the whole panel -
  leaks every period's own return into its own residual. The test requires the
  loadings for a period to be unchanged when later data is deleted.

  AN IID t-STAT ON OVERLAPPING LABELS. Already burned this repo once: H6 read
  p = 0.0041 iid and p = 0.106 Newey-West. The test drives strongly
  autocorrelated series through both and requires the HAC standard error to be
  the larger one.

The null baseline is tested too, because a null that is not actually null is
worse than no null - it certifies whatever it is compared against.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))

from flow_ic import (ic_series, neutralise, newey_west_t,   # noqa: E402
                     spearman, statistical_factors)


# --------------------------------------------------------------------------
# spearman
# --------------------------------------------------------------------------
def test_a_perfect_monotone_relation_is_ic_one():
    x = np.arange(20.0)
    assert spearman(x, x ** 3) == pytest.approx(1.0)


def test_a_reversed_relation_is_minus_one():
    x = np.arange(20.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_too_few_names_is_nan_not_a_lucky_number():
    """With four names a Spearman can read 1.0 by coincidence. A period that
    thin must not contribute an IC at all."""
    assert np.isnan(spearman(np.arange(4.0), np.arange(4.0)))


def test_a_constant_column_is_nan_not_zero():
    """Zero would say 'no relation measured here'. There is no relation
    MEASURABLE here, which is a different claim and must not be averaged in."""
    assert np.isnan(spearman(np.ones(20), np.arange(20.0)))


# --------------------------------------------------------------------------
# neutralisation
# --------------------------------------------------------------------------
def test_a_signal_that_is_just_momentum_does_not_survive_neutralisation():
    """The regression this file exists for. §7: flow that works only because
    it proxies momentum is not a discovery."""
    rng = np.random.default_rng(0)
    n = 200
    mom = rng.normal(size=n)
    fwd = 0.5 * mom + rng.normal(scale=0.5, size=n)
    # The score is momentum plus a whisper of noise. Making it EXACTLY momentum
    # would neutralise to a constant column, and spearman returns NaN there by
    # design - correct, but it would let this test pass for the wrong reason.
    score = mom + rng.normal(scale=0.02, size=n)
    D = pd.DataFrame({"window_end": pd.Timestamp("2024-01-31"),
                      "ticker": [f"T{i:03d}" for i in range(n)],
                      "score": score, "fwd_1w": fwd,
                      "mom12_1": mom, "rev1": rng.normal(size=n),
                      "log_turnover": rng.normal(size=n),
                      "vol60": rng.normal(size=n)})
    S = ic_series(D, "score", "fwd_1w", use_pc=False)
    assert abs(S["ic_raw"].iloc[0]) > 0.4, "raw IC should be strong"
    assert abs(S["ic"].iloc[0]) < 0.12, "neutralised IC must collapse"


def test_a_signal_orthogonal_to_the_controls_survives():
    rng = np.random.default_rng(1)
    n = 200
    edge = rng.normal(size=n)
    D = pd.DataFrame({"window_end": pd.Timestamp("2024-01-31"),
                      "ticker": [f"T{i:03d}" for i in range(n)],
                      "score": edge,
                      "fwd_1w": 0.6 * edge + rng.normal(scale=0.5, size=n),
                      "mom12_1": rng.normal(size=n), "rev1": rng.normal(size=n),
                      "log_turnover": rng.normal(size=n),
                      "vol60": rng.normal(size=n)})
    S = ic_series(D, "score", "fwd_1w", use_pc=False)
    assert abs(S["ic"].iloc[0]) > 0.4


def test_a_name_missing_a_control_is_absent_not_averaged():
    """Filling a missing control with the mean gives that name the average
    exposure, which is a fabricated observation. It must drop out instead."""
    y = np.array([1.0, 2.0, 3.0, np.nan, 5.0] * 8)
    X = np.column_stack([np.arange(40.0), np.arange(40.0) ** 2])
    r = neutralise(y, X)
    assert np.isnan(r[3])
    assert np.isfinite(r[0])


def test_neutralisation_refuses_when_there_are_fewer_names_than_controls():
    y = np.arange(4.0)
    X = np.column_stack([np.arange(4.0)] * 6)
    assert np.isnan(neutralise(y, X)).all()


# --------------------------------------------------------------------------
# the statistical factors must not see the future
# --------------------------------------------------------------------------
def _factor_panel(n_names=25, n_periods=80, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-31", periods=n_periods, freq="10B")
    common = rng.normal(size=n_periods)
    rows = []
    for j in range(n_names):
        load = rng.normal()
        for i, d in enumerate(dates):
            rows.append({"window_end": d, "ticker": f"T{j:02d}",
                         "period_ret": load * common[i] + rng.normal(scale=0.3)})
    return pd.DataFrame(rows)


def test_factor_loadings_for_a_period_ignore_everything_after_it():
    """Delete the tail of the panel and the loadings for an earlier period must
    not move. If they do, every residual carries its own future."""
    D = _factor_panel()
    p = sorted(D["window_end"].unique())[40]
    full = statistical_factors(D)
    trimmed = statistical_factors(D[D["window_end"] <= p])
    a = full[full["window_end"] == p].sort_values("ticker").reset_index(drop=True)
    b = trimmed[trimmed["window_end"] == p].sort_values("ticker").reset_index(
        drop=True)
    assert len(a) and len(a) == len(b)
    for c in [c for c in a.columns if c.startswith("pc")]:
        assert np.allclose(a[c].to_numpy(), b[c].to_numpy())


def test_no_factors_are_produced_before_there_is_history_to_fit_on():
    D = _factor_panel(n_periods=20)
    assert statistical_factors(D).empty


# --------------------------------------------------------------------------
# HAC standard errors
# --------------------------------------------------------------------------
def test_overlapping_labels_get_a_bigger_standard_error_than_iid_would_give():
    """H6 read p = 0.0041 iid and p = 0.106 Newey-West. Never again."""
    rng = np.random.default_rng(5)
    n = 300
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.8 * x[i - 1] + rng.normal(scale=0.1)
    x += 0.02
    _, se_hac, _ = newey_west_t(x, lags=6)
    se_iid = x.std(ddof=1) / np.sqrt(n)
    assert se_hac > se_iid * 1.5


def test_with_no_autocorrelation_hac_and_iid_broadly_agree():
    rng = np.random.default_rng(6)
    x = rng.normal(scale=0.1, size=500) + 0.01
    _, se_hac, _ = newey_west_t(x, lags=2)
    se_iid = x.std(ddof=1) / np.sqrt(500)
    assert 0.7 < se_hac / se_iid < 1.4


def test_too_few_periods_returns_nan_rather_than_a_confident_number():
    assert np.isnan(newey_west_t(np.array([0.1, 0.2]), 2)[2])


# --------------------------------------------------------------------------
# the null must actually be null
# --------------------------------------------------------------------------
def test_shuffling_within_a_period_destroys_the_pairing_and_keeps_the_marginal():
    """Averaged over periods, as the real pipeline does it.

    A single period's null IC has a standard error of about 1/sqrt(n) - 0.08 on
    150 names - so one period reading 0.21 is ordinary luck, not a broken null.
    Judging the null on one draw is the same mistake as judging the signal on
    one draw.
    """
    rng = np.random.default_rng(7)
    n, P = 150, 40
    frames = []
    for p in range(P):
        edge = rng.normal(size=n)
        g = pd.DataFrame({"window_end": pd.Timestamp("2024-01-31")
                          + pd.Timedelta(days=14 * p),
                          "ticker": [f"T{i:03d}" for i in range(n)],
                          "score": edge,
                          "fwd_1w": 0.7 * edge + rng.normal(scale=0.4, size=n)})
        g["null"] = rng.permutation(g["score"].to_numpy())
        assert np.allclose(np.sort(g["null"].to_numpy()),
                           np.sort(g["score"].to_numpy())), "marginal kept"
        frames.append(g)
    D = pd.concat(frames, ignore_index=True)
    real = ic_series(D, "score", "fwd_1w", controls=(), use_pc=False)
    null = ic_series(D, "null", "fwd_1w", controls=(), use_pc=False)
    assert real["ic"].mean() > 0.4
    assert abs(null["ic"].mean()) < 0.05
    # and the null must not be significant under the same t-stat the study uses
    assert abs(newey_west_t(null["ic"].to_numpy(), 2)[2]) < 2.0
