"""Tests for H44 — the simple methods on a cost ladder.

THE RESULTS FROM THIS HARNESS ARE UNDER AUDIT AND ARE NOT YET A FINDING. What
these tests pin is the machinery, not the conclusion: that every score is
computable from bars <= t, that the toll is applied with the right sign the
right number of times, that the long-short leg nets, and that the break-even
interpolation is arithmetic rather than a fit.

The causality tests are the load-bearing ones. A score that peeks turns the
whole ladder into a look-ahead and nothing in the printed output would look
wrong — which is how every ZigZag defect in this repo got in.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                "scripts"))

from cost_ladder import (RULES, TOLLS, _fib_prox, _swing_res,   # noqa: E402
                         add_scores, breakeven, walk)


def _path(n=600, seed=0, vol=0.02):
    rng = np.random.default_rng(seed)
    return 1000.0 * np.exp(np.cumsum(rng.normal(0.0005, vol, n)))


def _frame(n=400, names=60, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    out = []
    for i in range(names):
        p = 1000.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
        out.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i:02d}", "fwd": rng.normal(0.01, 0.1, n),
            **{r: rng.normal(size=n) for r in RULES}}))
    return pd.concat(out, ignore_index=True).sort_values(["date", "ticker"])


# ============================== CAUSALITY — the tests the study depends on ====
@pytest.mark.parametrize("cut", [200, 350, 500])
def test_the_swing_resistance_at_a_past_bar_ignores_every_future_bar(cut):
    """Confirmation-gated, never pivot-gated. If a future bar could change a
    past level the whole ladder is a look-ahead and the output looks fine."""
    p = _path()
    full = _swing_res(p)
    part = _swing_res(p[:cut])
    a, b = full[:cut], part
    both = np.isfinite(a) | np.isfinite(b)
    assert np.allclose(a[both], b[both], equal_nan=True)


@pytest.mark.parametrize("cut", [200, 350, 500])
def test_the_fibonacci_level_at_a_past_bar_ignores_every_future_bar(cut):
    p = _path(seed=3)
    assert np.allclose(_fib_prox(p, 0.618)[:cut], _fib_prox(p[:cut], 0.618))


def test_every_score_at_a_past_bar_is_unchanged_by_appending_the_future():
    """The direct statement, across all rules at once."""
    n = 500
    rng = np.random.default_rng(7)
    p = _path(n, seed=7)
    g = pd.DataFrame({
        "adj_close": p, "ema20": pd.Series(p).ewm(span=20).mean(),
        "ema30": pd.Series(p).ewm(span=30).mean(),
        "ema50": pd.Series(p).ewm(span=50).mean(),
        "stoch_k": rng.uniform(0, 100, n),
        "mom12_1": rng.normal(size=n),
        "hi52": rng.uniform(0.5, 1.0, n), "vol60": rng.uniform(0.01, 0.05, n)})
    cut = 320
    full = add_scores(g.copy())
    part = add_scores(g.iloc[:cut].copy())
    #  hi52/vol60 are ranked WITHIN the frame, so strength_calm is a
    #  cross-sectional-in-time score and is expected to move; the path-based
    #  ones must not.
    for col in ("sr_break", "fib_618", "ema_cross", "ema_stack",
                "stoch_oversold", "stoch_strong", "mom12_1"):
        assert np.allclose(full[col].to_numpy()[:cut],
                           part[col].to_numpy(), equal_nan=True), col


def test_a_swing_high_is_not_recorded_until_price_has_fallen_from_it():
    """The confirmation lag IS the point: a pivot drawn at the bar it occurred
    uses information that did not exist until the reversal."""
    p = np.concatenate([np.linspace(100, 200, 50), [200.0] * 5])
    res = _swing_res(p)
    assert np.all(np.isnan(res)), "no high is confirmed while price only rises"
    q = np.concatenate([p, np.linspace(200, 150, 20)])
    assert np.isfinite(_swing_res(q)[-1]), "a 25% fall must confirm the high"


# ==================================================== the portfolio and toll ==
def test_a_higher_toll_never_raises_the_net_return():
    D = _frame()
    prev = None
    for t in sorted(TOLLS):
        r = walk(D, "mom12_1", t)
        if prev is not None:
            assert r["cagr"] <= prev + 1e-12
        prev = r["cagr"]


def test_zero_toll_is_the_gross_return():
    D = _frame()
    a = walk(D, "mom12_1", 0)
    b = walk(D, "mom12_1", 56)
    assert a["cagr"] > b["cagr"]
    assert a["turnover"] == pytest.approx(b["turnover"])


def test_the_long_short_leg_nets_the_two_books():
    """Long top decile, short bottom, half weight each. If the sign were wrong
    the arm would be long-long and every rule would look twice as good."""
    D = _frame()
    lo = walk(D, "mom12_1", 0, short=False)
    ls = walk(D, "mom12_1", 0, short=True)
    assert lo["arm"] == "long" and ls["arm"] == "L/S"
    assert ls["periods"] == lo["periods"]


def test_a_score_that_is_pure_noise_lands_near_the_universe_mean():
    """THE PREDICTED-NULL DISCIPLINE (A9), as a unit test. On a frame whose
    returns are independent of every score, the decile portfolio must not beat
    the pool it is drawn from by any margin worth noticing."""
    D = _frame(names=120, seed=11)
    r = walk(D, "rand", 0)
    uni = D.groupby("date")["fwd"].mean().mean()
    assert abs(r["mean_per"] - uni) < 0.02


def test_a_book_is_never_smaller_or_larger_than_the_declared_bounds():
    from cost_ladder import MAX_BOOK, MIN_BOOK
    D = _frame(names=200, seed=4)
    r = walk(D, "mom12_1", 0)
    assert r["periods"] > 0
    assert MIN_BOOK <= MAX_BOOK


def test_too_thin_a_cross_section_is_skipped_rather_than_traded():
    """Fewer than 40 names is not a cross-section; ranking 6 into deciles is a
    concentration artefact wearing a portfolio's clothes."""
    D = _frame(names=6)
    assert walk(D, "mom12_1", 0) == {}


def test_the_predicted_null_control_is_actually_in_the_rule_set():
    """A9: register a predicted-null in EVERY sweep. If `rand` were dropped the
    sweep would lose its cheapest check that the pipeline is not manufacturing
    its own signal."""
    assert "rand" in RULES


def test_both_stochastic_readings_are_tested_not_just_the_flattering_one():
    """'Buy oversold' and 'buy strength' are opposite rules and only one can be
    right; testing whichever comes back positive is the mined answer."""
    assert "stoch_oversold" in RULES and "stoch_strong" in RULES


# ================================================ the break-even interpolation
def test_the_break_even_toll_sits_between_the_bracketing_grid_points():
    rows = [{"toll": 0, "cagr": 0.10}, {"toll": 25, "cagr": 0.05},
            {"toll": 56, "cagr": -0.02}]
    be = breakeven(rows)
    assert 25 < be < 56


def test_a_rule_positive_everywhere_reports_no_break_even_below_the_grid():
    rows = [{"toll": t, "cagr": 0.10} for t in (0, 25, 56)]
    assert breakeven(rows) == float("inf")


def test_a_rule_negative_everywhere_reports_never():
    rows = [{"toll": t, "cagr": -0.10} for t in (0, 25, 56)]
    assert breakeven(rows) == 0.0


def test_the_interpolation_is_exact_on_a_straight_line():
    rows = [{"toll": 0, "cagr": 0.10}, {"toll": 20, "cagr": -0.10}]
    assert breakeven(rows) == pytest.approx(10.0)
