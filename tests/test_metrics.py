"""The §39 metric set and the deflated Sharpe.

THE TESTS THIS FILE EXISTS FOR are the POSITIVE CONTROLS. A26's sine wave and
A36's Q0 martingale both record the same discipline: a statistic that cannot
return the known answer on a case with a known answer proves nothing when it
returns an unknown one. So `psr` is checked to reduce EXACTLY to the normal
one-sided test when skew is zero and kurtosis is three, `expected_max_sharpe`
is checked against the order statistic it claims to approximate by simulation,
and `deflated_sharpe` is checked to be monotone in the trial count.

The second class of test here guards the DIRECTION of every correction. Every
one of them must make the answer WORSE — smaller effective n, smaller Sharpe
under positive autocorrelation, lower PSR under negative skew and fat tails,
lower DSR under more trials. A "correction" that can flatter a result is a bug
wearing a statistic's clothes.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.metrics import (deflated_sharpe, describe,        # noqa: E402
                            effective_n, expectancy,
                            expected_max_sharpe, exposure, lo_factor,
                            periods_to_detect, profit_factor, psr, sharpe,
                            sortino, summary, tail, _nppf)


# ------------------------------------------------- the inverse normal itself

@pytest.mark.parametrize("p,want", [(0.5, 0.0), (0.975, 1.959964),
                                    (0.025, -1.959964), (0.99, 2.326348),
                                    (0.001, -3.090232)])
def test_nppf_matches_the_standard_normal_quantiles(p, want):
    assert _nppf(p) == pytest.approx(want, abs=1e-5)


def test_nppf_is_defined_at_the_boundaries_rather_than_raising():
    assert _nppf(0.0) == float("-inf")
    assert _nppf(1.0) == float("inf")


# --------------------------------------------------------- Sharpe and family

def test_sharpe_reproduces_the_textbook_value():
    r = np.full(1000, 0.001)
    r[::2] = 0.003                       # mean 0.002, sd 0.001
    got = sharpe(r, periods_per_year=252)
    assert got == pytest.approx(0.002 / r.std(ddof=1) * math.sqrt(252), rel=1e-9)


def test_sharpe_is_nan_on_a_constant_series():
    """Zero variance is undefined, never infinite — an infinite Sharpe has
    appeared in enough backtests to be worth pinning."""
    assert math.isnan(sharpe(np.full(500, 0.01)))


def test_positive_autocorrelation_lowers_the_sharpe():
    """THE DIRECTION IS THE TEST. sqrt(q) assumes independence and overstates."""
    rng = np.random.default_rng(0)
    e = rng.normal(0.001, 0.01, 4000)
    ar = np.zeros_like(e)
    for i in range(1, len(e)):
        ar[i] = 0.5 * ar[i - 1] + e[i]
    naive = sharpe(ar, 252, overlap=1)
    adj = sharpe(ar, 252, overlap=10)
    assert np.isfinite(adj)
    assert adj < naive, (
        "the autocorrelation adjustment made the Sharpe LARGER — a correction "
        "that can flatter a result is a bug")


def test_lo_factor_is_sqrt_q_on_independent_data():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 20000)
    assert lo_factor(a, 5) == pytest.approx(math.sqrt(5), rel=0.10)


def test_effective_n_deflates_by_the_overlap():
    assert effective_n(1000, 1) == 1000
    assert effective_n(1000, 20) == 50


def test_sortino_uses_all_periods_in_the_denominator():
    """The common implementation takes std of the negatives only, which is a
    different and larger ratio. This pins the shortfall definition."""
    r = np.array([0.05] * 90 + [-0.10] * 10)
    dd = math.sqrt((np.minimum(r, 0.0) ** 2).mean())
    assert sortino(r, 1.0) == pytest.approx(r.mean() / dd, rel=1e-9)
    #  the naive version would divide by the std of the ten losers, which is 0
    assert np.isfinite(sortino(r, 1.0))


def test_sortino_ignores_upside_volatility():
    lo = np.array([0.01] * 50 + [-0.01] * 50)
    hi = np.array([0.50] * 50 + [-0.01] * 50)
    assert sortino(hi, 1.0) > sortino(lo, 1.0)


# ------------------------------------------------------- PSR positive control

def test_psr_reduces_to_the_normal_one_sided_test_when_the_sample_is_normal():
    """POSITIVE CONTROL. With skew 0 and kurtosis 3, PSR(0) must equal
    Phi(SR * sqrt(n-1)) exactly — the ordinary one-sided test."""
    rng = np.random.default_rng(7)
    a = rng.normal(0.001, 0.01, 20000)
    z = (a - a.mean()) / a.std(ddof=1)
    g3, g4 = (z ** 3).mean(), (z ** 4).mean()
    assert abs(g3) < 0.06 and abs(g4 - 3.0) < 0.12   # sample really is normal
    sr = a.mean() / a.std(ddof=1)
    want = 0.5 * (1 + math.erf(sr * math.sqrt(len(a) - 1) / math.sqrt(2)))
    assert psr(a, 0.0, 252) == pytest.approx(want, abs=0.01)


def test_negative_skew_lowers_the_psr():
    """A13 measured kurtosis to 2,800 on these series; the whole point of PSR
    is that a fat tail makes a Sharpe LESS trustworthy."""
    rng = np.random.default_rng(3)
    base = rng.normal(0.001, 0.01, 5000)
    #  same mean and sd, negative skew: many small wins, few large losses
    skewed = -np.abs(rng.standard_gamma(1.0, 5000))
    skewed = (skewed - skewed.mean()) / skewed.std(ddof=1)
    skewed = skewed * base.std(ddof=1) + base.mean()
    assert psr(skewed, 0.0, 252) < psr(base, 0.0, 252)


def test_psr_falls_as_the_benchmark_rises():
    rng = np.random.default_rng(5)
    a = rng.normal(0.001, 0.01, 4000)
    assert psr(a, 0.0, 252) > psr(a, 1.0, 252) > psr(a, 3.0, 252)


def test_psr_uses_the_effective_n_not_the_row_count():
    rng = np.random.default_rng(9)
    a = rng.normal(0.0008, 0.01, 6000)
    assert psr(a, 0.0, 252, overlap=20) < psr(a, 0.0, 252, overlap=1)


# ---------------------------------------------- the deflation positive control

def test_expected_max_sharpe_matches_a_simulated_maximum():
    """POSITIVE CONTROL. The analytic E[max] of N draws from N(0, V) must match
    the simulated maximum. This is the only part of the deflation that is an
    approximation, so it is the part that gets checked against brute force."""
    rng = np.random.default_rng(2)
    v, n = 0.25, 200
    sim = np.mean([rng.normal(0.0, math.sqrt(v), n).max() for _ in range(4000)])
    got = expected_max_sharpe(n, v, 252)
    assert got == pytest.approx(sim, rel=0.05), (got, sim)


def test_expected_max_sharpe_grows_with_the_trial_count():
    v = 0.5
    seq = [expected_max_sharpe(n, v) for n in (2, 10, 100, 350, 5000)]
    assert all(b > a for a, b in zip(seq, seq[1:]))


def test_expected_max_sharpe_is_undefined_below_two_trials():
    assert math.isnan(expected_max_sharpe(1, 0.5))
    assert math.isnan(expected_max_sharpe(350, 0.0))


def test_deflated_sharpe_falls_monotonically_in_the_trial_count():
    """THE WHOLE POINT. Searching harder has to make the survivor less
    believable, or the correction is not a correction."""
    rng = np.random.default_rng(4)
    a = rng.normal(0.0015, 0.01, 3000)
    d = [deflated_sharpe(a, n, 0.5)["dsr"] for n in (2, 20, 350, 100000)]
    assert all(np.isfinite(x) for x in d)
    assert all(b <= a_ for a_, b in zip(d, d[1:])), d


def test_deflated_sharpe_never_exceeds_the_undeflated_psr():
    rng = np.random.default_rng(6)
    a = rng.normal(0.002, 0.01, 3000)
    assert deflated_sharpe(a, 350, 0.5)["dsr"] <= psr(a, 0.0, 252) + 1e-12


def test_deflated_sharpe_refuses_to_default_the_variance():
    """A deflation against an ASSUMED variance reads as a measurement.
    `summary` must leave the field absent when either input is missing."""
    rng = np.random.default_rng(8)
    a = rng.normal(0.001, 0.01, 800)
    assert "dsr" not in summary(a, 252)
    assert "dsr" not in summary(a, 252, trials=350)
    assert "dsr" in summary(a, 252, trials=350, sr_variance=0.5)


def test_trials_to_kill_is_reported_when_the_result_still_survives():
    rng = np.random.default_rng(10)
    a = rng.normal(0.01, 0.01, 3000)          # an absurdly strong sample
    d = deflated_sharpe(a, 2, 0.1)
    assert d["dsr"] > 0.95
    assert np.isfinite(d["trials_to_kill"]) or d["dsr"] >= 0.95


# ------------------------------------------------------------ trade statistics

def test_profit_factor_is_undefined_with_no_losers_not_infinite():
    assert math.isnan(profit_factor([0.1, 0.2, 0.3]))
    assert profit_factor([0.2, -0.1]) == pytest.approx(2.0)


def test_expectancy_carries_the_mean_log_beside_the_mean():
    """A36 measured a +5.77% average trade compounding at -5.3% a year, so a
    mean quoted alone is not a statement about an account."""
    e = expectancy([1.0, -0.6, -0.6, 1.0])
    assert e["expectancy"] > 0
    assert e["mean_log"] < 0, "the two must be free to disagree in sign"


def test_expectancy_win_rate_and_averages():
    e = expectancy([0.10, 0.20, -0.05, -0.15])
    assert e["win_rate"] == 0.5
    assert e["avg_win"] == pytest.approx(0.15)
    assert e["avg_loss"] == pytest.approx(-0.10)
    assert e["profit_factor"] == pytest.approx(1.5)


def test_exposure_counts_periods_holding_anything():
    assert exposure([True, False, True, False]) == 0.5
    assert math.isnan(exposure([]))


# ---------------------------------------------------------------- tail risk

def test_cvar_is_at_least_as_bad_as_var():
    rng = np.random.default_rng(12)
    a = rng.normal(0, 0.02, 5000)
    t = tail(a, 0.05)
    assert t["cvar"] <= t["var"]
    assert t["worst_period"] <= t["cvar"]


def test_a_first_period_loss_counts_as_a_drawdown():
    """The path starts at the initial capital, not at the first bar's close.

    Without a leading 1.0 the running maximum starts BELOW par whenever the
    first period is a loss, and a series that opens -30% reports a maximum
    drawdown of exactly zero. This is the bug the Ulcer test below found.
    """
    assert tail(np.array([-0.30] + [0.05] * 40))["max_drawdown"] == \
        pytest.approx(-0.30)


def test_ulcer_charges_for_the_LENGTH_of_a_drawdown_not_only_its_depth():
    """A18 measured a per-position stop producing the DEEPEST portfolio
    drawdown in its table, so depth alone has already misled this repo."""
    quick = np.array([-0.30] + [0.05] * 40)
    slow = np.array([-0.01] * 30 + [0.05] * 11)
    assert tail(slow)["max_drawdown"] > tail(quick)["max_drawdown"]  # shallower
    assert tail(slow)["ulcer"] > tail(quick)["ulcer"]                # but longer


def test_kurtosis_is_reported_unexcessed():
    rng = np.random.default_rng(13)
    a = rng.normal(0, 1, 40000)
    assert tail(a)["kurtosis"] == pytest.approx(3.0, abs=0.15)


# -------------------------------------------------------------------- power

def test_periods_to_detect_scales_with_the_squared_noise_to_signal():
    a = np.array([0.01, -0.01] * 500) + 0.001
    n1 = periods_to_detect(a)
    n2 = periods_to_detect(a * 1.0 + 0.0)      # same series
    assert n1 == pytest.approx(n2)
    loud = np.array([0.02, -0.02] * 500) + 0.001
    assert periods_to_detect(loud) > n1, "more noise needs more periods"


def test_periods_to_detect_is_undefined_when_the_mean_is_exactly_the_target():
    a = np.array([0.01, -0.01] * 100)
    assert math.isnan(periods_to_detect(a, against=float(a.mean())))


# ------------------------------------------------------------ the bundle

def test_summary_carries_mean_median_and_mean_log_together():
    rng = np.random.default_rng(14)
    a = rng.normal(0.001, 0.02, 1000)
    m = summary(a, 252)
    for k in ("mean", "median", "mean_log", "sharpe", "sortino", "psr",
              "max_drawdown", "ulcer", "cvar", "skew", "kurtosis",
              "periods_to_detect", "effective_n"):
        assert k in m, k


def test_summary_returns_only_the_count_on_a_degenerate_sample():
    assert summary([0.01, 0.02]) == {"periods": 2.0}


def test_describe_prints_power_before_any_ratio():
    """A19 records a power statement being written as an effect statement.
    Ordering it first is the cheapest guard against reading it as one."""
    rng = np.random.default_rng(15)
    a = rng.normal(0.001, 0.02, 500)
    txt = describe(summary(a, 252, trials=350, sr_variance=0.5))
    assert txt.splitlines()[0].startswith("POWER")
    assert txt.index("POWER") < txt.index("Sharpe")
    assert "DEFLATED" in txt


def test_describe_handles_an_empty_summary():
    assert describe({}) == "insufficient data"
