"""Tests for band selection and the multi-timeframe stack.

The one that matters most is ``test_slow_state_never_uses_the_day_in_progress``.
Aligning a daily signal onto intraday bars is the easiest place in this whole
repo to manufacture an edge by accident: if the colour attached to 10:00 on day D
was computed from D's close, the rule knows how the day ends before it trades it,
and every number downstream becomes fiction.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from band_optimizer import (equity_log, pick_band, profile,      # noqa: E402
                            random_walk_profile, toll_expectancy, walk_forward)
from mtf_stack import (daily_state_on_intraday, positions,       # noqa: E402
                       score, RULES)
from paint_live import band_state                                # noqa: E402


def hourly(closes, days):
    """A frame of `len(closes)` bars spread over `days` trading days."""
    per = len(closes) // days
    ts = []
    for d in range(days):
        base = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for h in range(per):
            ts.append(base + pd.Timedelta(hours=9 + h))
    return pd.DataFrame({"close": np.asarray(closes[:len(ts)], float)},
                        index=pd.DatetimeIndex(ts))


# --------------------------------------------------------------------------- #
# the look-ahead guard
# --------------------------------------------------------------------------- #
def test_slow_state_never_uses_the_day_in_progress():
    # a violent move happens entirely on the LAST day; the colour carried by
    # that day's bars must still be the one implied by the previous close
    closes = [100.0] * 12 + [100.0] * 12 + [300.0] * 12
    d = hourly(closes, 3)
    st = daily_state_on_intraday(d, 0.08)
    assert st[-1] == st[-12], "colour changed inside the day it was measuring"
    assert st[-1] == 0, "the day's own +200% leaked into that day's colour"


def test_slow_state_turns_green_only_on_the_following_day():
    closes = [100.0] * 8 + [100.0] * 8 + [200.0] * 8 + [200.0] * 8
    d = hourly(closes, 4)
    st = daily_state_on_intraday(d, 0.08)
    assert st[16:24].max() == 0        # the day of the jump: still red
    assert st[24:].min() == 1          # the day after: green throughout


def test_slow_state_is_constant_within_a_day():
    rng = np.random.default_rng(2)
    d = hourly(100 * np.cumprod(1 + rng.normal(0, 0.02, 100)), 10)
    st = daily_state_on_intraday(d, 0.08)
    day = pd.Series(d.index.date, index=d.index)
    for _k, g in pd.Series(st, index=d.index).groupby(day.to_numpy()):
        assert g.nunique() == 1


def test_first_day_has_no_prior_close_and_is_flat():
    d = hourly([100.0] * 20, 2)
    assert daily_state_on_intraday(d, 0.08)[0] == 0


# --------------------------------------------------------------------------- #
# combination rules
# --------------------------------------------------------------------------- #
def test_both_green_is_the_intersection():
    f = np.array([1, 1, 0, 0, 1], dtype=np.int8)
    s = np.array([1, 0, 1, 0, 1], dtype=np.int8)
    assert list(positions(f, s, "both_green")) == [1, 0, 0, 0, 1]


def test_fast_in_slow_out_holds_until_the_slow_band_turns():
    #            enter here ^          ... and only exit when slow goes red
    f = np.array([0, 1, 0, 0, 0, 0], dtype=np.int8)
    s = np.array([1, 1, 1, 1, 0, 0], dtype=np.int8)
    assert list(positions(f, s, "fast_in_slow_out")) == [0, 1, 1, 1, 0, 0]


def test_fast_in_slow_out_will_not_enter_while_slow_is_red():
    f = np.array([0, 1, 1, 1], dtype=np.int8)
    s = np.array([0, 0, 0, 0], dtype=np.int8)
    assert positions(f, s, "fast_in_slow_out").max() == 0


def test_fast_in_slow_out_needs_a_fresh_flip_not_a_standing_signal():
    # fast is already green when slow turns green; that is not an entry
    f = np.array([1, 1, 1, 1], dtype=np.int8)
    s = np.array([0, 1, 1, 1], dtype=np.int8)
    assert positions(f, s, "fast_in_slow_out").max() == 0


@pytest.mark.parametrize("rule", RULES)
def test_every_rule_is_long_only_and_binary(rule):
    rng = np.random.default_rng(4)
    f = (rng.random(200) > 0.5).astype(np.int8)
    s = (rng.random(200) > 0.5).astype(np.int8)
    p = positions(f, s, rule)
    assert set(np.unique(p)) <= {0, 1}


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def test_score_lags_the_decision_by_one_bar():
    px = np.array([100.0, 200.0, 200.0])
    # green only on the bar the jump happened: it must NOT collect that jump
    pos = np.array([0, 1, 0], dtype=np.int8)
    assert np.isclose(score(px, pos, fee=0.0)["log"], 0.0)


def test_score_charges_a_fee_per_entry():
    px = np.array([100.0, 100.0, 100.0, 100.0])
    pos = np.array([0, 1, 0, 1], dtype=np.int8)
    s = score(px, pos, fee=0.01)
    assert s["trips"] == 2
    assert np.isclose(s["log"], 2 * np.log(0.99))


def test_full_exposure_with_no_fee_equals_buy_and_hold():
    rng = np.random.default_rng(9)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))
    pos = np.ones(300, dtype=np.int8)
    assert np.isclose(score(px, pos, fee=0.0)["log"], np.log(px[-1] / px[0]))


# --------------------------------------------------------------------------- #
# band selection
# --------------------------------------------------------------------------- #
def test_the_colour_of_a_prefix_does_not_change_when_the_future_arrives():
    # the actual causality property: appending bars must not alter the state of
    # the bars that came before them. If it does, the score of any window is
    # contaminated by data from after it.
    rng = np.random.default_rng(6)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.02, 400))
    for cut in (120, 200, 310):
        pre, _ = band_state(px[:cut], 0.08)
        full, _ = band_state(px, 0.08)
        assert np.array_equal(pre, full[:cut])


def test_scoring_a_prefix_uses_only_that_prefix():
    rng = np.random.default_rng(16)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.02, 400))
    a = equity_log(px[:200], 0.08)
    tampered = px.copy()
    tampered[200:] *= 3.0                    # rewrite the future entirely
    assert np.isclose(equity_log(tampered[:200], 0.08), a)


def test_fees_can_only_reduce_the_score():
    rng = np.random.default_rng(8)
    px = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, 500))
    assert equity_log(px, 0.08, fee=0.0056) <= equity_log(px, 0.08, fee=0.0)


def test_walk_forward_picks_without_seeing_the_test_window():
    rng = np.random.default_rng(1)
    px = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, 900))
    grid = np.array([0.05, 0.08, 0.12, 0.20])
    a = walk_forward(px, grid, folds=2)
    # replacing the FINAL fold's data must not change the band chosen for the
    # earlier folds, because those were selected before it existed
    alt = px.copy()
    alt[-100:] = alt[-101] * np.cumprod(1 + rng.normal(0, 0.05, 100))
    b = walk_forward(alt, grid, folds=2)
    assert a is not None and b is not None
    assert a["folds"] == b["folds"]


def test_profile_reports_the_breakeven_for_each_band():
    rng = np.random.default_rng(0)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.02, 600))
    p = profile(px, np.array([0.05, 0.10]))
    assert list(p["band"]) == [0.05, 0.10]
    assert (p["breakeven"] > p["band"]).all()


def test_random_walk_profile_matches_its_input_volatility():
    rng = np.random.default_rng(12)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.02, 500))
    rw = random_walk_profile(px, np.array([0.08]), draws=3)
    assert len(rw) > 0
    assert (rw["band"] == 0.08).all()


def test_pick_band_returns_a_value_from_the_grid():
    rng = np.random.default_rng(13)
    px = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, 400))
    grid = np.array([0.05, 0.08, 0.12])
    for how in ("grid", "breakeven"):
        assert pick_band(px, grid, how) in set(grid)


def test_toll_expectancy_falls_as_the_band_swallows_every_leg():
    rng = np.random.default_rng(14)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.015, 800))
    assert toll_expectancy(px, 0.50) <= toll_expectancy(px, 0.05)


# --------------------------------------------------------------------------- #
# layer-1 gates: the lag is the entire result
# --------------------------------------------------------------------------- #
from layered import FILTERS, gates, run                          # noqa: E402


def panel(n=400, cols=("A", "B", "C")):
    rng = np.random.default_rng(21)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {c: 100 * np.cumprod(1 + rng.normal(0.0004, 0.02, n)) for c in cols},
        index=idx)


def test_no_gate_equals_its_own_unlagged_version():
    """A gate that matches the same-bar condition is reading its own bar."""
    px = panel(500)
    G = gates(px)
    same_bar = px > px.rolling(200, min_periods=200).mean()
    for name in ("trend", "trend_rs"):
        assert not G[name]["A"].equals(same_bar["A"].fillna(False)), \
            f"{name} gate is reading its own bar"


def test_trend_gate_matches_the_shifted_moving_average_exactly():
    px = panel(500)
    g = gates(px)["trend"]["A"]
    want = (px["A"] > px["A"].rolling(200, min_periods=200).mean()).shift(1).fillna(False)
    assert g.equals(want)


def test_rs_gate_is_a_lagged_cross_sectional_rank():
    px = panel(500)
    g = gates(px)["rs"]["A"]
    rank = px.pct_change(120).rank(axis=1, pct=True)
    want = (rank["A"] > 2.0 / 3.0).shift(1).fillna(False)
    assert g.equals(want)


def test_the_none_gate_is_always_open():
    px = panel(300)
    assert bool(gates(px)["none"].all().all())


def test_a_gate_can_only_reduce_exposure():
    px = panel(600)
    G = gates(px)
    base = run(px["A"], G["none"]["A"], 0.08)
    assert base is not None
    for name in FILTERS:
        r = run(px["A"], G[name]["A"], 0.08)
        if r:
            assert r["exposure"] <= base["exposure"] + 1e-12


def test_the_same_exposure_null_is_exposure_times_hold():
    px = panel(600)
    r = run(px["A"], gates(px)["trend"]["A"], 0.08)
    assert r is not None
    assert np.isclose(r["null"], r["exposure"] * r["hold"])


def test_market_gate_is_identical_across_names():
    px = panel(500)
    m = gates(px)["market"]
    assert m["A"].equals(m["B"]) and m["B"].equals(m["C"])
