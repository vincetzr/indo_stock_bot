"""Tests for H13's cross-sectional IC machinery.

Two things here are worth more than the rest.

THE COST FUNCTION. `reference.half_spread` already returns a FRACTION of price,
and the first version of `price_ic` divided by price a second time. That makes
every spread cost roughly a thousand times too small — a 25 bps half-spread
becomes 0.0025 bps — and it turns a losing quintile spread into a winning one
without any other symptom. The test below pins the magnitude against a tick
band read straight off the published fraksi harga schedule.

THE FAST PATH. The IC loop was rewritten onto presorted numpy arrays so that a
200-draw null is affordable; an unaffordable null is how a null ends up run
once, and H9 records where that leads. Rewritten hot loops are exactly where a
silent indexing error hides, so the fast path is checked against the obvious
pandas implementation rather than trusted.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))

from price_ic import (MIN_NAMES, analyse, half_spread_frac,      # noqa: E402
                      ic_by_day, ic_from_arrays, permutation_null,
                      prepare, shuffle_within_days)
from flow_ic import neutralise, spearman                          # noqa: E402


def panel(n_days=60, n_names=80, seed=0, signal=0.0):
    """A cross-sectional panel where ``signal`` couples the feature to the
    NEXT period's return and nothing else does."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2018-01-01", periods=n_days)
    rows = []
    for d in days:
        f = rng.normal(size=n_names)
        r = signal * f + rng.normal(0, 1.0, n_names)
        for i in range(n_names):
            rows.append({"date": d, "ticker": f"T{i:03d}",
                         "feat": float(f[i]), "fwd5": float(r[i]),
                         "mom12_1": float(rng.normal()),
                         "rev1": float(rng.normal()),
                         "log_turnover": float(rng.normal()),
                         "vol60": float(abs(rng.normal())),
                         "close": float(rng.uniform(100, 5000))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# THE COST FUNCTION — the bug that would have flipped a verdict
# --------------------------------------------------------------------------
def test_half_spread_is_a_fraction_of_price_not_a_fraction_of_a_fraction():
    """From 2016-05-02 the band 500-2000 has a Rp 5 tick. At a price of
    Rp 1,000 half a tick is Rp 2.50, which is 25 bps of price.

    The earlier version divided by price twice and returned 2.5e-6 instead of
    2.5e-3 — a cost a thousand times too small, with no other symptom."""
    hs = half_spread_frac([1000.0], [pd.Timestamp("2020-06-01")])[0]
    assert hs == pytest.approx(0.0025, rel=1e-9)
    assert hs > 1e-4, "a half-spread below a basis point is the double-divide bug"


def test_half_spread_is_larger_for_a_cheaper_share():
    """The whole reason §7 insists on the point-in-time fraksi harga: the tick
    is a bigger fraction of a small price, so small caps pay far more.

    From 2016: Rp 120 sits in the sub-200 band (Rp 1 tick) so half a tick is
    0.5/120 = 417 bps; Rp 4,000 sits in the 2,000-5,000 band (Rp 10 tick), so
    5/4000 = 125 bps. That is 3.3x, not the 5x an earlier version of this test
    asserted — the assertion was wrong, the schedule is not."""
    day = [pd.Timestamp("2020-06-01")] * 2
    cheap, dear = half_spread_frac([120.0, 4000.0], day)
    assert cheap == pytest.approx(0.5 / 120.0)
    assert dear == pytest.approx(5.0 / 4000.0)
    assert cheap > dear * 3


def test_half_spread_respects_the_point_in_time_schedule():
    """2014-2016 ran a three-group ladder, 2016 onward a five-group one.

    The price has to be chosen where they actually differ. At Rp 1,000 both
    schedules give a Rp 5 tick, so an earlier version of this test asserted a
    difference that does not exist. At Rp 3,000 they part: the old ladder's
    500-5,000 band gives Rp 5, the new one's 2,000-5,000 band gives Rp 10."""
    a = half_spread_frac([3000.0], [pd.Timestamp("2015-06-01")])[0]
    b = half_spread_frac([3000.0], [pd.Timestamp("2020-06-01")])[0]
    assert a == pytest.approx(2.5 / 3000.0)
    assert b == pytest.approx(5.0 / 3000.0)


def test_the_same_price_can_share_a_tick_across_regimes():
    """The converse, pinned so the schedule is not 'fixed' into disagreeing
    everywhere: at Rp 1,000 both ladders really do give a Rp 5 tick."""
    a = half_spread_frac([1000.0], [pd.Timestamp("2015-06-01")])[0]
    b = half_spread_frac([1000.0], [pd.Timestamp("2020-06-01")])[0]
    assert a == pytest.approx(b) == pytest.approx(2.5 / 1000.0)


def test_half_spread_is_nan_for_a_nonsense_price():
    assert np.isnan(half_spread_frac([0.0], [pd.Timestamp("2020-06-01")])[0])


# --------------------------------------------------------------------------
# THE FAST PATH must equal the obvious one
# --------------------------------------------------------------------------
def reference_ic(D, feature, label, controls):
    out = []
    for _, g in D.groupby("date", sort=True):
        if len(g) < MIN_NAMES:
            continue
        y = g[feature].to_numpy(float)
        r = g[label].to_numpy(float)
        if controls:
            y = neutralise(y, g[list(controls)].to_numpy(float))
        out.append(spearman(y, r))
    return np.array(out, dtype=float)


def test_the_fast_ic_matches_the_pandas_reference():
    D = panel(40, 60, seed=1, signal=0.3)
    ctrl = ("mom12_1", "rev1", "log_turnover", "vol60")
    fast = ic_by_day(D, "feat", "fwd5", ctrl)["ic"].to_numpy()
    ref = reference_ic(D, "feat", "fwd5", ctrl)
    ref = ref[np.isfinite(ref)]
    assert len(fast) == len(ref)
    assert np.allclose(fast, ref, atol=1e-12)


def test_days_below_the_minimum_cross_section_are_dropped():
    D = panel(10, MIN_NAMES - 5, seed=2)
    assert ic_by_day(D, "feat", "fwd5", ()).empty


# --------------------------------------------------------------------------
# the shuffle
# --------------------------------------------------------------------------
def test_the_shuffle_preserves_every_days_values():
    D = panel(15, 50, seed=3)
    P = prepare(D, "feat", "fwd5", ())
    y = shuffle_within_days(P["y"], P["starts"], P["ends"],
                            np.random.default_rng(4))
    for a, b in zip(P["starts"], P["ends"]):
        assert sorted(np.round(y[a:b], 12)) == sorted(np.round(P["y"][a:b], 12))


def test_the_shuffle_never_moves_a_value_between_days():
    D = panel(12, 40, seed=5)
    P = prepare(D, "feat", "fwd5", ())
    tag = np.repeat(np.arange(len(P["starts"])),
                    P["ends"] - P["starts"]).astype(float)
    y = shuffle_within_days(tag, P["starts"], P["ends"],
                            np.random.default_rng(6))
    assert np.array_equal(y, tag), "a value crossed a day boundary"


# --------------------------------------------------------------------------
# known answers
# --------------------------------------------------------------------------
def test_an_injected_cross_sectional_signal_is_found():
    D = panel(80, 80, seed=7, signal=0.5)
    ic = ic_by_day(D, "feat", "fwd5", ())["ic"]
    assert ic.mean() > 0.2, ic.mean()


def test_pure_noise_is_not_found_and_its_null_is_centred_at_zero():
    D = panel(80, 80, seed=8, signal=0.0)
    obs = ic_by_day(D, "feat", "fwd5", ())["ic"].mean()
    nulls = permutation_null(prepare(D, "feat", "fwd5", ()), 40, 9, 5)
    v = nulls[np.isfinite(nulls)]
    assert abs(obs) < 0.05, obs
    assert abs(v.mean()) < 0.02, v.mean()
    assert v.std() > 0, "a null with no spread cannot certify anything"


def test_an_injected_signal_sits_outside_its_null():
    D = panel(80, 80, seed=10, signal=0.5)
    obs = ic_by_day(D, "feat", "fwd5", ())["ic"].mean()
    nulls = permutation_null(prepare(D, "feat", "fwd5", ()), 40, 11, 5)
    assert obs > np.nanmax(nulls)


def test_the_null_recomputes_the_neutralisation_inside_the_loop():
    """Applying controls once, outside the shuffle, answers an easier question.
    With controls present the null must still be centred near zero."""
    D = panel(60, 70, seed=12, signal=0.0)
    ctrl = ("mom12_1", "rev1", "log_turnover", "vol60")
    nulls = permutation_null(prepare(D, "feat", "fwd5", ctrl), 40, 13, 5)
    v = nulls[np.isfinite(nulls)]
    assert abs(v.mean()) < 0.02, v.mean()


# --------------------------------------------------------------------------
# the one-pass core
# --------------------------------------------------------------------------
def test_analyse_reproduces_the_ic_from_the_readable_path():
    """The one-pass core must give the same IC as the obvious per-day loop."""
    D = panel(50, 70, seed=20, signal=0.4)
    D["hs_frac"] = 0.002
    ctrl = ("mom12_1", "rev1", "log_turnover", "vol60")
    A = analyse(prepare(D, "feat", "fwd5", ctrl), 5)
    ref = reference_ic(D, "feat", "fwd5", ctrl)
    assert A["ic_mean"] == pytest.approx(np.nanmean(ref), abs=1e-12)


def test_the_quintile_spread_is_cut_on_the_NEUTRALISED_score():
    """H9 ranked its spread on the RAW score while its IC used the neutralised
    one, so the two statistics described different signals. Here a control that
    fully explains the feature must collapse the SPREAD too, not just the IC.

    Compared against the same panel with no control rather than against zero:
    once the residual is numerically zero the sort order is arbitrary, so the
    spread is small-but-not-exactly-zero on any given day. An earlier version
    of this test asserted < 1e-6 and failed on that, which was the assertion
    being wrong rather than the code."""
    rng = np.random.default_rng(21)
    D = panel(60, 80, seed=21, signal=0.0)
    # The feature carries the control plus genuine independent noise, and the
    # return is driven ONLY by the control. So the raw feature predicts, and
    # everything it predicts with is exactly what the control removes.
    # (Making the feature EXACTLY collinear instead leaves a residual that is
    # pure floating-point noise, and the test then measures rounding.)
    D["feat"] = D["mom12_1"] + 0.3 * rng.normal(size=len(D))
    D["fwd5"] = D["mom12_1"]
    D["hs_frac"] = 0.0
    with_ctrl = analyse(prepare(D, "feat", "fwd5", ("mom12_1",)), 5)
    without = analyse(prepare(D, "feat", "fwd5", ()), 5)
    assert abs(without["gross_per"]) > 0.5, "the raw feature must show a spread"
    assert abs(with_ctrl["gross_per"]) < 0.05 * abs(without["gross_per"]), (
        f"neutralising must collapse the spread: {with_ctrl['gross_per']:.4f} "
        f"against a raw {without['gross_per']:.4f}")


def test_cost_is_charged_per_rebalance_and_scales_with_horizon():
    """A signal held 20 days pays the round trip a quarter as often as one held
    5 days. If the annualiser ignored that, short horizons would look far worse
    than they are and the decay curve would be meaningless."""
    D = panel(80, 80, seed=22, signal=0.3)
    D["fwd20"] = D["fwd5"]
    D["hs_frac"] = 0.001
    a5 = analyse(prepare(D, "feat", "fwd5", ()), 5)
    a20 = analyse(prepare(D, "feat", "fwd20", ()), 20)
    assert a5["cost_per"] == pytest.approx(a20["cost_per"])
    # same per-rebalance cost, but annualised it is 4x cheaper at k=20
    assert (a5["gross_annual"] - a5["net_annual"]) == pytest.approx(
        4.0 * (a20["gross_annual"] - a20["net_annual"]), rel=1e-9)


def test_by_liquidity_returns_one_ic_per_quintile():
    D = panel(60, 100, seed=23, signal=0.3)
    D["hs_frac"] = 0.001
    A = analyse(prepare(D, "feat", "fwd5", ()), 5)
    assert len(A["by_liquidity"]) == 5
    assert np.isfinite(A["by_liquidity"]).sum() >= 4


def test_fast_spearman_equals_the_pandas_one():
    """The rank correlation was rewritten in numpy so 100 permutation draws
    are affordable. A rewritten statistic is exactly where a silent difference
    hides, so it is asserted equal to the original rather than trusted —
    including on ties, which is where naive ranking diverges."""
    from price_ic import fast_spearman
    rng = np.random.default_rng(40)
    for _ in range(20):
        a = rng.normal(size=200)
        b = 0.4 * a + rng.normal(size=200)
        assert fast_spearman(a, b) == pytest.approx(spearman(a, b), abs=1e-12)
    tied_a = np.repeat(rng.normal(size=20), 10)
    tied_b = np.repeat(rng.normal(size=20), 10)
    assert fast_spearman(tied_a, tied_b) == pytest.approx(
        spearman(tied_a, tied_b), abs=1e-12)


def test_fast_spearman_refuses_a_constant_series():
    from price_ic import fast_spearman
    assert np.isnan(fast_spearman(np.ones(50), np.arange(50, dtype=float)))


def test_light_mode_gives_the_same_ic_as_the_full_pass():
    """The null runs `light` to skip the quintile and liquidity work it never
    reads. If light drifted from full, every p-value would be measured against
    a different statistic than the one reported."""
    D = panel(50, 70, seed=41, signal=0.35)
    D["hs_frac"] = 0.001
    P = prepare(D, "feat", "fwd5", ("mom12_1", "rev1"))
    full = analyse(P, 5)
    lite = analyse(P, 5, light=True)
    assert lite["ic_mean"] == pytest.approx(full["ic_mean"], abs=1e-12)
    assert "gross_per" not in lite
