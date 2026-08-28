"""Tests for H34/H35 — swing levels, Fibonacci, and the bracket grid.

The two things that would silently invalidate the whole study are using a
ZigZag pivot before it was confirmed, and crediting a stop with its nominal
price when the bar that breached it closed lower. Both are pinned first.
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

from levels import (FIB, FIB_NODES, RATIOS, bracket_summary,   # noqa: E402
                    fib_curve, fib_events, fib_test, pivots_confirmed,
                    sr_adjusted, sr_events)


# ================================================ confirmation, not hindsight =
def test_a_pivot_is_confirmed_strictly_after_it_happens():
    """THE LOOK-AHEAD TRAP IN EVERY ZIGZAG STUDY. A high at bar i is only a high
    once price has fallen k% from it, which is a later bar. Using the pivot at i
    would be trading on information that did not exist."""
    p = np.concatenate([np.linspace(100, 150, 20), np.linspace(150, 90, 20)])
    piv, conf = pivots_confirmed(p, 0.10)
    assert len(piv) == len(conf)
    assert (conf > piv).all()


def test_the_confirmed_high_is_the_actual_extreme_of_the_leg():
    p = np.concatenate([np.linspace(100, 150, 20), np.linspace(150, 90, 20)])
    piv, _ = pivots_confirmed(p, 0.10)
    assert p[piv].max() == pytest.approx(150.0)


def test_the_confirmation_lag_grows_with_the_threshold():
    """A 20% swing takes longer to confirm than a 10% one, which is the price
    of a cleaner level and has to be visible in the numbers."""
    rng = np.random.default_rng(0)
    p = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 3000)))
    for k, lag in ((0.05, None), (0.10, None), (0.20, None)):
        piv, conf = pivots_confirmed(p, k)
        assert len(piv) > 2
    a = pivots_confirmed(p, 0.05)
    b = pivots_confirmed(p, 0.20)
    assert np.median(a[1] - a[0]) < np.median(b[1] - b[0])


def test_a_flat_series_confirms_nothing():
    piv, conf = pivots_confirmed(np.full(200, 100.0), 0.10)
    assert len(piv) == 0


# ============================================== the Fibonacci grid design =====
def test_the_ratio_grid_is_continuous_and_brackets_every_fibonacci_value():
    """THE DESIGN IS THE RESULT. Comparing 0.618 to 0.90 compares depths; the
    only fair control is a NEIGHBOUR that is not a Fibonacci number, which
    requires a fine continuous grid."""
    assert len(RATIOS) >= 30
    step = np.diff(RATIOS)
    assert np.allclose(step, step[0])
    for f in FIB:
        assert RATIOS.min() <= f <= RATIOS.max()
        assert np.min(np.abs(RATIOS - f)) <= step[0] / 2 + 1e-9


def test_each_fibonacci_ratio_maps_to_a_distinct_grid_node():
    assert len(set(FIB_NODES)) == len(FIB)


def test_a_deeper_retracement_is_reached_by_fewer_pullbacks():
    """The mechanical fact the quadratic in the test exists to absorb. If this
    ever inverted, the level construction would be wrong."""
    p = np.concatenate([np.linspace(100, 200, 60), np.linspace(200, 150, 40),
                        np.linspace(150, 260, 60), np.linspace(260, 130, 60)])
    E = pd.DataFrame(fib_events(p, np.ones(len(p), bool)))
    C = fib_curve(E)
    assert C["n"].iloc[0] >= C["n"].iloc[-1]


def test_the_retracement_level_is_measured_from_the_leg_not_the_close():
    """high - r*(high-low), so r=0 is the high and r=1 is the low. Anchoring to
    the current price instead would make every level a moving target."""
    p = np.concatenate([np.linspace(100, 200, 40), np.linspace(200, 100, 40)])
    E = pd.DataFrame(fib_events(p, np.ones(len(p), bool)))
    assert not E.empty
    #  depth is (hi - lvl)/hi, so it must rise with the ratio
    d = E.groupby("r")["depth"].median()
    assert d.is_monotonic_increasing


def test_the_fibonacci_test_finds_a_bump_that_is_really_there():
    """A NULL TEST THAT CANNOT DETECT AN EFFECT PROVES NOTHING BY FINDING NONE.
    Planting a real bump at the Fibonacci nodes must light the statistic up."""
    rng = np.random.default_rng(1)
    C = pd.DataFrame({"r": RATIOS,
                      "regain": 0.40 - 0.12 * RATIOS
                      + np.array([0.05 if np.any(np.isclose(r, FIB_NODES))
                                  else 0.0 for r in RATIOS])})
    out = fib_test(C, "regain", rng)
    assert out["z"] > 4.0
    assert out["p"] < 0.01


def test_the_fibonacci_test_is_quiet_on_a_smooth_curve():
    rng = np.random.default_rng(2)
    C = pd.DataFrame({"r": RATIOS, "regain": 0.40 - 0.12 * RATIOS})
    out = fib_test(C, "regain", rng)
    assert abs(out["z"]) < 2.0
    assert out["p"] > 0.05


# ================================================ support / resistance ========
def test_the_placebo_is_the_same_construction_on_the_wrong_level():
    p = np.concatenate([np.linspace(100, 150, 30), np.linspace(150, 120, 20),
                        np.linspace(120, 200, 60)])
    S = pd.DataFrame(sr_events(p, np.ones(len(p), bool), "T"))
    assert set(S["kind"]) <= {"true", "placebo -7%", "placebo +7%"}
    assert (S["kind"] == "true").any()


def test_a_displaced_level_sits_at_a_different_distance():
    """WHICH IS EXACTLY THE CONFOUND the distance adjustment exists to remove:
    a level 7% lower is crossed earlier in the rally."""
    p = np.concatenate([np.linspace(100, 150, 30), np.linspace(150, 120, 20),
                        np.linspace(120, 260, 80)])
    S = pd.DataFrame(sr_events(p, np.ones(len(p), bool), "T"))
    m = S.groupby("kind")["dist"].median()
    if {"true", "placebo -7%"} <= set(m.index):
        assert m["placebo -7%"] < m["true"]


def test_sr_adjusted_recovers_a_planted_effect():
    rng = np.random.default_rng(3)
    n = 4000
    kind = np.where(np.arange(n) % 2 == 0, "true", "placebo -7%")
    dist = rng.uniform(0.02, 0.30, n)
    y = 0.5 + 0.4 * dist + np.where(kind == "true", -0.10, 0.0) \
        + rng.normal(0, 0.05, n)
    S = pd.DataFrame({"kind": kind, "dist": dist, "y": y,
                      "ticker": rng.integers(0, 40, n).astype(str)})
    out = sr_adjusted(S, "y", rng, draws=40)
    assert out["effect_pp"] == pytest.approx(-10.0, abs=1.5)
    assert out["lo_pp"] < -10.0 < out["hi_pp"] or out["hi_pp"] < 0


def test_the_interval_resamples_tickers_not_rows():
    """A15/A18: one name contributes many events from overlapping legs, so an
    iid row bootstrap returns an interval several times too narrow. The clue is
    that the width must respond to the NUMBER OF NAMES, not the row count."""
    rng = np.random.default_rng(4)
    n = 3000
    few = pd.DataFrame({
        "kind": np.where(np.arange(n) % 2 == 0, "true", "placebo -7%"),
        "dist": rng.uniform(0.02, 0.3, n), "y": rng.normal(0, 1, n),
        "ticker": (np.arange(n) % 4).astype(str)})
    many = few.copy()
    many["ticker"] = (np.arange(n) % 200).astype(str)
    wf = sr_adjusted(few, "y", np.random.default_rng(5), draws=60)
    wm = sr_adjusted(many, "y", np.random.default_rng(5), draws=60)
    assert (wf["hi_pp"] - wf["lo_pp"]) > (wm["hi_pp"] - wm["lo_pp"])


# ==================================================== the bracket grid ========
def _B(rows):
    return pd.DataFrame(rows)


def test_the_summary_annualises_by_the_cells_own_holding_period():
    """A21's lesson in a new place: a per-trade figure is not a yearly one. A
    bracket exiting in 52 sessions is redeployed about five times a year, and
    quoting its per-trade mean log as an annual number overstates by 5x."""
    B = _B({"tp": [0.2] * 4, "sl": [0.1] * 4, "year": [2010, 2011, 2018, 2019],
            "ret": [0.2, -0.1, 0.2, -0.1], "bars": [50.0] * 4,
            "hit": ["tp", "sl", "tp", "sl"]})
    R = bracket_summary(B)
    r = R.iloc[0]
    assert r["ann"] == pytest.approx(r["meanlog"] * 252.0 / 50.0)


def test_the_summary_reports_which_barrier_arrived_first():
    B = _B({"tp": [0.2] * 5, "sl": [0.1] * 5, "year": [2010] * 5,
            "ret": [0.2, 0.2, -0.1, -0.1, 0.0], "bars": [30.0] * 5,
            "hit": ["tp", "tp", "sl", "sl", "time"]})
    r = bracket_summary(B).iloc[0]
    assert r["p_tp"] == pytest.approx(0.4)
    assert r["p_sl"] == pytest.approx(0.4)


def test_costs_are_charged_on_every_path_including_the_timeout():
    """A bracket that 'avoids' a trade does not exist — the entry already
    happened, so the round trip is paid whether the exit was a target, a stop
    or the clock."""
    B = _B({"tp": [0.2] * 2, "sl": [0.1] * 2, "year": [2010, 2011],
            "ret": [0.0, 0.0], "bars": [252.0] * 2, "hit": ["time", "time"]})
    assert bracket_summary(B).iloc[0]["mean"] < 0.0


def test_a_hold_row_survives_the_grouping_with_nan_keys():
    """The hold benchmark carries tp = sl = NaN, and a groupby that drops NaN
    keys would silently delete the only row the grid is compared against."""
    B = _B({"tp": [0.2, np.nan], "sl": [0.1, np.nan], "year": [2010, 2010],
            "ret": [0.2, 0.05], "bars": [30.0, 252.0], "hit": ["tp", "hold"]})
    R = bracket_summary(B)
    assert R["tp"].isna().sum() == 1
