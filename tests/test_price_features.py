"""Tests for §8's price/TA features.

The single property that matters most is NO LOOK-AHEAD. A feature that peeks at
the future produces a beautiful IC and a worthless signal, and the failure is
silent — the numbers all look plausible. So the first block here does not test
values at all; it tests that changing a FUTURE bar cannot change a PAST
feature. That is the property, stated directly, rather than inspected by eye.

The second block pins the mechanisms: each feature must respond in the
direction its H13 registration claims, on a series constructed to have exactly
that property and nothing else.

The third block is arithmetic that is easy to get subtly wrong — true range
against the previous close, the one-bar execution gap in the label, and the
refusal to fill a lookback that is not yet satisfied.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.features.price import (CONTROLS, FEATURES,           # noqa: E402
                                   MIN_HISTORY, PREDICTED,
                                   SELF_CONTROL, atr, compute,
                                   forward_return, true_range)


def bars(n=600, seed=0, drift=0.0, vol=0.02):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    close = 1000.0 * np.exp(np.cumsum(r))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame({
        "date": pd.bdate_range("2015-01-01", periods=n),
        "open": close, "high": high, "low": low, "close": close,
        "adj_close": close,
        "volume": rng.uniform(1e5, 1e7, n)})


# --------------------------------------------------------------------------
# NO LOOK-AHEAD — the property, not a spot check
# --------------------------------------------------------------------------
@pytest.mark.parametrize("feat", sorted(set(FEATURES) | set(CONTROLS)))
def test_a_future_bar_cannot_change_a_past_feature(feat):
    """THE test. Perturb the last 50 bars beyond recognition; every feature
    value before that point must be bit-identical. Any feature that moves is
    reading the future, however plausible its numbers look."""
    a = bars(400, seed=3)
    b = a.copy()
    cut = len(b) - 50
    b.loc[cut:, ["close", "adj_close", "high", "low"]] *= 3.0
    b.loc[cut:, "volume"] *= 17.0

    fa = compute(a)[feat].to_numpy()[:cut]
    fb = compute(b)[feat].to_numpy()[:cut]
    ok = np.isfinite(fa) & np.isfinite(fb)
    assert ok.sum() > 50, "not enough finite values to make this meaningful"
    assert np.allclose(fa[ok], fb[ok], rtol=1e-12, atol=1e-12), (
        f"{feat} changed in the PAST when only future bars moved")


def test_the_label_skips_a_bar_for_execution():
    """The feature is known at the close of t; the position is taken at the
    close of t+1. If the label started at t the test would be filling at a
    price the decision was made from."""
    px = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41])
    f = forward_return(px, k=1, gap=1)
    # at t=0: enter at px[1]=110, exit at px[2]=121 -> +10%
    assert f.iloc[0] == pytest.approx(0.10)
    assert np.isnan(f.iloc[-1]) and np.isnan(f.iloc[-2])


def test_the_label_without_a_gap_would_use_the_decision_bar():
    """Pinning the alternative so the gap cannot be silently removed."""
    px = pd.Series([100.0, 110.0, 121.0])
    assert forward_return(px, k=1, gap=0).iloc[0] == pytest.approx(0.10)
    assert forward_return(px, k=1, gap=1).iloc[0] == pytest.approx(0.10)
    # they differ once the series is not geometric
    px = pd.Series([100.0, 150.0, 160.0, 170.0])
    assert forward_return(px, 1, 0).iloc[0] != pytest.approx(
        forward_return(px, 1, 1).iloc[0])


# --------------------------------------------------------------------------
# the mechanisms, each on a series built to have exactly that property
# --------------------------------------------------------------------------
def test_rev5_is_negative_of_the_recent_return():
    """Signed so that POSITIVE means 'predicted to outperform'. A name that
    just fell must score positively."""
    d = bars(300, seed=1)
    d.loc[len(d) - 1, ["close", "adj_close"]] = d["close"].iloc[-6] * 0.8
    out = compute(d)
    assert out["rev5"].iloc[-1] > 0, "a 20% fall over 5 bars must score positive"


def test_lowvol_is_negative_of_volatility():
    calm, wild = bars(400, seed=2, vol=0.002), bars(400, seed=2, vol=0.05)
    assert compute(calm)["lowvol"].iloc[-1] > compute(wild)["lowvol"].iloc[-1]


def test_hi52_is_one_at_a_new_high_and_below_one_otherwise():
    d = bars(400, seed=4)
    d["adj_close"] = np.linspace(100, 500, len(d))     # monotone rise
    assert compute(d)["hi52"].iloc[-1] == pytest.approx(1.0)
    d2 = bars(400, seed=4)
    d2["adj_close"] = np.linspace(500, 100, len(d2))   # monotone fall
    assert compute(d2)["hi52"].iloc[-1] < 0.5


def test_amihud_is_higher_for_the_thinner_name():
    thin, thick = bars(300, seed=5), bars(300, seed=5)
    thick["volume"] = thin["volume"] * 1000.0
    assert (compute(thin)["amihud60"].iloc[-1]
            > compute(thick)["amihud60"].iloc[-1])


def test_volz_spikes_when_recent_volume_jumps():
    d = bars(400, seed=6)
    d.loc[len(d) - 20:, "volume"] = d["volume"].mean() * 50
    assert compute(d)["volz20"].iloc[-1] > 3.0


def test_squeeze_is_below_one_when_the_range_has_compressed():
    d = bars(500, seed=7)
    tail = slice(len(d) - 30, None)
    mid = d["close"].iloc[tail]
    d.loc[tail, "high"] = mid * 1.0005
    d.loc[tail, "low"] = mid * 0.9995
    assert compute(d)["squeeze"].iloc[-1] < 1.0


def test_atr_mom_is_positive_after_a_rise():
    d = bars(300, seed=8, drift=0.004)
    assert compute(d)["atr_mom20"].iloc[-1] > 0


def test_mom12_1_skips_the_most_recent_month():
    """The skip is the whole point: 12-1 momentum excludes the last month
    precisely because short-horizon reversal lives there."""
    d = bars(400, seed=9)
    d["adj_close"] = 100.0
    d.loc[len(d) - 10:, "adj_close"] = 1e6      # a huge move in the skip window
    assert compute(d)["mom12_1"].iloc[-1] == pytest.approx(0.0), \
        "a move inside the skipped month must not enter 12-1 momentum"


# --------------------------------------------------------------------------
# arithmetic that is easy to get subtly wrong
# --------------------------------------------------------------------------
def test_true_range_uses_the_previous_close_not_just_the_days_range():
    h = [10.0, 12.0]
    l = [9.0, 11.5]
    c = [9.5, 12.0]
    tr = true_range(h, l, c)
    assert tr.iloc[1] == pytest.approx(2.5)      # 12 - 9.5, the gap, not 0.5


def test_atr_needs_its_window_before_it_reports():
    d = bars(60, seed=10)
    a = atr(d["high"], d["low"], d["close"], 250)
    assert a.isna().all(), "an ATR window longer than the series must be NaN"


def test_features_are_nan_before_their_lookback_is_satisfied():
    """Never filled. A feature computed on half its window is a different
    feature and would quietly enter the cross-section as if it were the same."""
    d = compute(bars(300, seed=11))
    assert d["hi52"].iloc[:100].isna().any()
    assert np.isnan(d["mom12_1"].iloc[0])


def test_no_feature_carries_an_infinity():
    """Zero volume and zero ATR both divide, and an inf would dominate every
    cross-sectional rank it appears in."""
    d = bars(400, seed=12)
    d.loc[100:150, "volume"] = 0.0
    d.loc[200:260, ["high", "low", "close", "adj_close"]] = 500.0
    out = compute(d)
    for c in set(FEATURES) | set(CONTROLS):
        assert not np.isinf(out[c].to_numpy(dtype=float)).any(), c


# --------------------------------------------------------------------------
# the registration itself
# --------------------------------------------------------------------------
def test_every_tested_feature_has_a_registered_prediction():
    assert set(FEATURES) == set(PREDICTED)
    assert all(v in (-1, 0, 1) for v in PREDICTED.values())


def test_exactly_one_feature_is_a_negative_control():
    """H13 registers `squeeze` as predicted-null on purpose: a pipeline that
    finds signal everywhere is finding artefacts."""
    nulls = [k for k, v in PREDICTED.items() if v == 0]
    assert nulls == ["squeeze"]


def test_the_self_control_map_covers_every_feature_that_is_a_control():
    """A feature regressed on itself residuals to zero. Any tested feature that
    is also a control must be listed so it can be dropped from its own set."""
    for f in FEATURES:
        if f in CONTROLS:
            assert f in SELF_CONTROL, f"{f} is a control and must be mapped"
    assert SELF_CONTROL["lowvol"] == "vol60"
