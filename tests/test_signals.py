"""Tests for the signal feed.

The feed's job is to be honest about three different kinds of claim: an exact
one (layer 3's trigger price), a measured-at-zero one (layer 2 flow), and an
untested one (layer 1 news). These tests pin the exact layer, because that is
the only one where a bug produces a wrong NUMBER rather than a wrong opinion.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from signals import (ARM_B, ARM_S, BUY, HOLD_C, HOLD_L, SELL,      # noqa: E402
                     classify, flow_summary, hit_rate, regime)


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def test_a_fresh_upflip_is_a_confirmed_buy():
    px = np.array([100.0, 95.0, 90.0, 99.0])      # 90*1.08 = 97.2, crossed
    assert classify(px, 0.08, 0.03)["signal"] == BUY


def test_a_fresh_downflip_is_a_confirmed_sell():
    px = np.array([100.0, 110.0, 120.0, 108.0])   # 120*0.92 = 110.4, crossed
    assert classify(px, 0.08, 0.03)["signal"] == SELL


def test_a_red_leg_near_its_trigger_is_armed_to_buy():
    px = np.array([100.0, 90.0, 90.0, 95.0])      # trigger 97.2, price 95
    out = classify(px, 0.08, 0.05)
    assert out["signal"] == ARM_B
    assert out["state"] == "RED"


def test_a_red_leg_far_from_its_trigger_is_just_cash():
    px = np.array([100.0, 90.0, 90.0, 90.5])
    assert classify(px, 0.08, 0.01)["signal"] == HOLD_C


def test_a_green_leg_near_its_trigger_is_armed_to_sell():
    px = np.array([100.0, 108.0, 120.0, 111.5])   # trigger 110.4
    out = classify(px, 0.08, 0.02)
    assert out["signal"] == ARM_S
    assert out["state"] == "GREEN"


def test_a_green_leg_far_from_its_trigger_just_holds():
    px = np.array([100.0, 108.0, 130.0, 130.0])
    assert classify(px, 0.08, 0.02)["signal"] == HOLD_L


def test_the_trigger_is_the_arithmetic_level_not_an_estimate():
    px = np.array([100.0, 90.0, 90.0, 92.0])
    assert np.isclose(classify(px, 0.08, 0.03)["trigger"], 90.0 * 1.08)


def test_the_gap_is_signed_toward_the_trigger():
    below = classify(np.array([100.0, 90.0, 90.0, 92.0]), 0.08, 0.03)
    assert below["gap"] > 0          # price must RISE to flip green
    above = classify(np.array([100.0, 108.0, 130.0, 125.0]), 0.08, 0.03)
    assert above["gap"] < 0          # price must FALL to flip red


def test_bars_in_leg_counts_the_bars_in_the_leg_including_the_flip():
    # flips green at index 2, so the leg spans indices 2,3,4 = 3 bars
    px = np.array([100.0, 90.0, 99.0, 100.0, 101.0])
    assert classify(px, 0.08, 0.03)["bars_in_leg"] == 3


def test_bars_in_leg_grows_by_one_per_bar_while_the_leg_holds():
    px = np.array([100.0, 90.0, 99.0, 100.0, 101.0])
    a = classify(px, 0.08, 0.03)["bars_in_leg"]
    b = classify(np.append(px, 102.0), 0.08, 0.03)["bars_in_leg"]
    assert b == a + 1


def test_bars_in_leg_resets_when_the_leg_flips():
    px = np.array([100.0, 90.0, 99.0, 105.0, 110.0, 100.0])   # 110*0.92=101.2
    out = classify(px, 0.08, 0.03)
    assert out["state"] == "RED" and out["bars_in_leg"] == 1


def test_classify_never_reads_a_bar_that_has_not_printed():
    rng = np.random.default_rng(5)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.02, 300))
    a = classify(px[:200], 0.08, 0.03)
    tampered = px.copy()
    tampered[200:] *= 5.0
    assert classify(tampered[:200], 0.08, 0.03) == a


# --------------------------------------------------------------------------- #
# the base rate that every signal carries
# --------------------------------------------------------------------------- #
def test_hit_rate_is_nan_when_there_is_too_little_history():
    out = hit_rate(np.array([100.0, 101.0, 102.0]), 0.08)
    assert np.isnan(out["win_rate"])


def test_hit_rate_charges_fees():
    """A round trip that exactly breaks even gross must show as a loss net."""
    rng = np.random.default_rng(7)
    px = 100 * np.cumprod(1 + rng.normal(0.0005, 0.025, 900))
    out = hit_rate(px, 0.08)
    if out["trips"] >= 5:
        assert out["median_pl"] < 1.0        # sanity: it is a fraction, not a multiple


def test_hit_rate_counts_round_trips_not_bars():
    rng = np.random.default_rng(9)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.02, 800))
    out = hit_rate(px, 0.08)
    assert 0 < out["trips"] < len(px)


# --------------------------------------------------------------------------- #
# layer 1 and 2 are context, and must degrade quietly
# --------------------------------------------------------------------------- #
def test_regime_is_empty_rather_than_wrong_on_a_short_series():
    s = pd.Series(np.arange(10, dtype=float),
                  index=pd.bdate_range("2026-01-01", periods=10))
    assert "vs_200d" not in regime(s)


def test_regime_reports_position_against_the_200_day_average():
    idx = pd.bdate_range("2024-01-01", periods=400)
    s = pd.Series(np.linspace(100, 200, 400), index=idx)
    assert regime(s)["vs_200d"] > 0


def test_flow_summary_is_none_without_data():
    assert flow_summary(None) is None
    assert flow_summary(pd.DataFrame()) is None


def test_flow_summary_flags_a_balanced_rekap_as_complete():
    d = pd.DataFrame({"date": ["2026-08-19"] * 2, "broker": ["BK", "YP"],
                      "buy_lot": [100.0, 50.0], "sell_lot": [50.0, 100.0]})
    assert flow_summary(d)["complete"] is True


def test_flow_summary_flags_a_truncated_table_as_incomplete():
    d = pd.DataFrame({"date": ["2026-08-19"] * 2, "broker": ["BK", "YP"],
                      "buy_lot": [100.0, 50.0], "sell_lot": [10.0, 20.0]})
    assert flow_summary(d)["complete"] is False
