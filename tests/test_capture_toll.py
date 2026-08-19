"""Tests for the capture-toll law.

The central claim is arithmetic rather than statistical - a move bigger than the
band cannot fail to flip the state - so most of these are property tests over
generated paths rather than fixtures. If a single adversarial path can hide a
big leg from the rule, the claim is dead and one of these should catch it.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from capture_toll import (breakeven_move, capture_ceiling,      # noqa: E402
                          ceiling_entry_only, ceiling_fraction,
                          exposure_log_return, leg_table, round_trip_from)
from paint_live import band_state                                # noqa: E402


# --------------------------------------------------------------------------- #
# the algebra
# --------------------------------------------------------------------------- #
def test_breakeven_is_exactly_where_the_ceiling_is_one():
    for b in (0.02, 0.05, 0.08, 0.12, 0.25):
        assert np.isclose(capture_ceiling(breakeven_move(b), b), 1.0)


def test_below_breakeven_is_a_guaranteed_loss():
    b = 0.08
    m = breakeven_move(b)
    assert capture_ceiling(m * 0.99, b) < 1.0
    assert capture_ceiling(m * 1.01, b) > 1.0


def test_the_toll_is_a_fixed_fraction_of_price_not_of_the_move():
    b = 0.08
    toll = (1 + b) / (1 - b)
    for m in (0.2, 0.5, 1.0, 3.0):
        assert np.isclose(capture_ceiling(m, b), (1 + m) / toll)


def test_bigger_moves_keep_a_larger_share():
    b = 0.08
    shares = [ceiling_fraction(m, b) for m in (0.25, 0.5, 1.0, 2.0, 5.0)]
    assert all(shares[i] < shares[i + 1] for i in range(len(shares) - 1))


def test_entry_only_ceiling_is_above_the_round_trip_one():
    b = 0.08
    for m in (0.2, 0.5, 2.0):
        assert ceiling_entry_only(m, b) > ceiling_fraction(m, b)


def test_a_zero_band_keeps_everything():
    assert np.isclose(capture_ceiling(1.0, 0.0), 2.0)
    assert np.isclose(breakeven_move(0.0), 0.0)


# --------------------------------------------------------------------------- #
# the claim that cannot be probabilistic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("band", [0.05, 0.08, 0.12, 0.20])
def test_a_move_larger_than_the_band_always_flips_the_state(band):
    rng = np.random.default_rng(11)
    for _ in range(40):
        n = int(rng.integers(60, 400))
        px = 100 * np.cumprod(1 + rng.normal(0.0005, 0.03, n))
        df = leg_table(px, band)
        big = df[df["move"] > band]
        if len(big):
            assert bool(big["green_by_peak"].all()), "a big leg was never flagged"


def test_a_single_gap_through_the_move_still_flips():
    # the whole move arrives in one bar; the flip lands on that bar
    px = np.array([100.0] * 10 + [400.0] * 10)
    st, _ = band_state(px, 0.08)
    assert st[10] == 1


def test_the_state_flips_exactly_at_the_trigger_and_not_before():
    b = 0.10
    px = np.array([100.0, 95.0, 104.4, 104.6])   # trigger is 95 * 1.10 = 104.5
    st, trig = band_state(px, b)
    assert st[2] == 0
    assert st[3] == 1
    assert np.isclose(trig[1], 95.0 * 1.10)


def test_a_leg_smaller_than_the_band_need_not_flip():
    px = np.array([100.0, 95.0, 100.0, 96.0, 100.0])
    st, _ = band_state(px, 0.20)
    assert st.max() == 0


# --------------------------------------------------------------------------- #
# capture accounting
# --------------------------------------------------------------------------- #
def test_exposure_uses_only_prior_colour():
    # colour is deliberately all-green from a point onward; the bar that pays is
    # the one AFTER the colour was known, never the one that set it
    px = np.array([100.0, 110.0, 121.0, 133.1])
    st = np.array([0, 1, 1, 1], dtype=np.int8)
    got = exposure_log_return(px, st, 0, 3)
    assert np.isclose(got, np.log(121 / 110) + np.log(133.1 / 121))


def test_delay_can_only_remove_exposure_in_a_pure_uptrend():
    px = 100 * 1.05 ** np.arange(40)
    st = np.ones(40, dtype=np.int8)
    base = exposure_log_return(px, st, 0, 39, delay=0)
    for d in (1, 2, 5):
        assert exposure_log_return(px, st, 0, 39, delay=d) <= base + 1e-12


def test_capture_never_beats_its_ceiling_on_tradable_legs():
    rng = np.random.default_rng(5)
    band = 0.08
    for _ in range(30):
        px = 100 * np.cumprod(1 + rng.normal(0.001, 0.025, 300))
        df = leg_table(px, band)
        ok = df[df["move"] > band]
        if len(ok):
            assert bool((ok["capture"] <= ok["ceiling_entry"] + 1e-9).all())


def test_round_trip_enters_after_the_signal_not_on_it():
    px = np.array([100.0, 90.0, 100.0, 120.0, 130.0, 100.0, 95.0, 96.0])
    st, _ = band_state(px, 0.08)
    rt = round_trip_from(px, st, 1)
    assert rt is not None
    entry, exit_ = rt
    assert st[entry - 1] == 1          # the flip is on the bar BEFORE the fill
    assert entry < exit_


def test_round_trip_is_none_when_the_position_never_closes():
    px = np.concatenate([np.full(5, 100.0), 100 * 1.05 ** np.arange(40)])
    st, _ = band_state(px, 0.08)
    assert round_trip_from(px, st, 0) is None


def test_leg_table_only_reports_up_legs():
    rng = np.random.default_rng(3)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.03, 400))
    df = leg_table(px, 0.10)
    assert (df["move"] > 0).all()


# --------------------------------------------------------------------------- #
# the corrections: the toll is a floor, and a bull move is not one leg
# --------------------------------------------------------------------------- #
from capture_toll import bull_moves, realised_tolls                # noqa: E402


def test_entry_overshoot_is_recorded_not_assumed():
    # the trigger is 100*1.08 = 108 but the first close at/above it is 130
    px = np.array([100.0, 100.0, 130.0, 140.0, 150.0, 120.0, 110.0, 111.0])
    rows = realised_tolls(px, 0.08)
    assert rows, "no round trip found"
    overshoot = rows[0][0]
    assert overshoot > 0.08, "overshoot was assumed to equal the band"


def test_the_floor_binds_at_signal_prices_but_not_at_fills():
    """(1+b)/(1-b) bounds the SIGNAL price, not the fill.

    The flip fires on a close at or above min(1+b), but the fill is the NEXT
    bar, which can come back under the trigger. So realised tolls straddle the
    'floor' - most land above it, some below - and a test asserting it always
    binds is asserting something false. This one pins both directions.
    """
    rng = np.random.default_rng(31)
    b = 0.08
    floor = (1 + b) / (1 - b)
    tolls = []
    for _ in range(20):
        px = 100 * np.cumprod(1 + rng.normal(0.0005, 0.025, 500))
        tolls += [(1 + e) / (1 - x) for e, x, _p, _m in realised_tolls(px, b)]
    tolls = np.array(tolls)
    assert len(tolls) > 50
    assert (tolls < floor).any(), "no fill ever beat the floor - check the lag"
    assert (tolls > floor).any()
    assert np.median(tolls) > floor, "the typical fill should be worse, not better"


def test_a_move_bigger_than_the_band_can_still_lose_on_real_fills():
    """'Guaranteed loss below break-even' was wrong in the other direction too:
    gaps mean the realised toll is normally worse than the algebra's floor."""
    rng = np.random.default_rng(33)
    px = 100 * np.cumprod(1 + rng.normal(0.0005, 0.03, 900))
    rows = realised_tolls(px, 0.08)
    tolls = np.array([(1 + e) / (1 - x) for e, x, _p, _m in rows])
    assert np.median(tolls) > (1.08 / 0.92), "realised toll never exceeded the floor"


def test_bull_moves_contain_more_than_one_band_leg():
    # a staircase: up, pull back more than the band, up again - one 30% move
    up1 = np.linspace(100, 140, 25)
    dip = np.linspace(140, 120, 10)
    up2 = np.linspace(120, 190, 25)
    px = np.concatenate([np.full(20, 100.0), up1, dip, up2, np.full(20, 190.0)])
    out = bull_moves(px, 0.08, big_band=0.30, floor=0.50)
    assert out, "no bull move detected"
    assert out[0][1] >= 2, "a multi-leg move was counted as a single leg"


def test_bull_move_capture_never_exceeds_one():
    rng = np.random.default_rng(35)
    for _ in range(10):
        px = 100 * np.cumprod(1 + rng.normal(0.001, 0.025, 900))
        for _m, _n, cap, _f in bull_moves(px, 0.08):
            if np.isfinite(cap):
                assert cap <= 1.0 + 1e-9


def test_realised_tolls_returns_nothing_on_a_flat_series():
    assert realised_tolls(np.full(200, 100.0), 0.08) == []
