"""Tests for the bandar dashboard.

The dashboard's job is to present a book reconstruction WITHOUT letting the
reader mistake a guess for a holding. So the tests concentrate on the two things
that decide that: the Wilson interval (which must be honest at small n) and the
zero-sum identity that measures how much of the tape is invisible.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from bandar_dashboard import book_state, classify, daily_states, wilson  # noqa: E402
from idxbot.config import BrokerRegistry                                 # noqa: E402

REG = BrokerRegistry.from_yaml(os.path.join(
    os.path.dirname(__file__), os.pardir, "config", "brokers.yaml"))


def frame(rows):
    return pd.DataFrame(rows, columns=["date", "broker", "buy_lot", "sell_lot",
                                       "buy_avg", "sell_avg"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


# --------------------------------------------------------------------------- #
# Wilson intervals
# --------------------------------------------------------------------------- #
def test_wilson_brackets_the_point_estimate():
    p, lo, hi = wilson(30, 60)
    assert lo < p < hi
    assert p == pytest.approx(0.5)


def test_wilson_stays_inside_zero_and_one_at_the_edges():
    """The normal approximation goes below 0 here; Wilson must not."""
    _p, lo, hi = wilson(0, 20)
    assert lo >= 0.0
    _p, lo, hi = wilson(20, 20)
    assert hi <= 1.0


def test_a_smaller_sample_gives_a_wider_interval():
    _p, lo1, hi1 = wilson(5, 10)
    _p, lo2, hi2 = wilson(50, 100)
    assert (hi1 - lo1) > (hi2 - lo2)


def test_wilson_is_nan_without_observations():
    p, _lo, _hi = wilson(0, 0)
    assert np.isnan(p)


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def test_an_unknown_code_is_labelled_unknown_not_domestic():
    """Silently bucketing unknown codes as domestic would inflate that flow."""
    assert classify("ZZZ", REG) == "unknown"


def test_known_codes_land_in_a_real_bucket():
    got = {classify(c, REG) for c in ("YP", "CC", "BK", "AK", "DX")}
    assert got <= {"foreign", "domestic", "bumn", "unknown"}
    assert got != {"unknown"}


# --------------------------------------------------------------------------- #
# the zero-sum identity
# --------------------------------------------------------------------------- #
def test_a_complete_rekap_nets_to_zero():
    df = frame([("2026-01-05", "BK", 100, 40, 1000, 1000),
                ("2026-01-05", "CC", 40, 100, 1000, 1000)])
    assert float((df["buy_lot"] - df["sell_lot"]).sum()) == pytest.approx(0.0)


def test_a_truncated_rekap_does_not_net_to_zero():
    df = frame([("2026-01-05", "BK", 100, 10, 1000, 1000),
                ("2026-01-05", "CC", 40, 20, 1000, 1000)])
    assert float((df["buy_lot"] - df["sell_lot"]).sum()) != pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# book state
# --------------------------------------------------------------------------- #
def test_unrealised_uses_the_last_close_against_the_basis():
    df = frame([("2026-01-05", "BK", 100, 0, 1000, 0)])
    B = book_state(df, 1200.0, REG).set_index("broker")
    assert B.loc["BK", "lots_left"] == pytest.approx(100.0)
    assert B.loc["BK", "unreal_pct"] == pytest.approx(0.20)
    assert B.loc["BK", "unrealised"] == pytest.approx(100 * (1200 - 1000) * 100)


def test_a_broker_holding_nothing_reports_no_floating_pl():
    df = frame([("2026-01-05", "BK", 100, 0, 1000, 0),
                ("2026-01-06", "BK", 0, 100, 0, 1100)])
    B = book_state(df, 1200.0, REG).set_index("broker")
    assert B.loc["BK", "lots_left"] == pytest.approx(0.0)
    assert np.isnan(B.loc["BK", "unreal_pct"])


def test_being_underwater_shows_as_a_negative_percentage():
    df = frame([("2026-01-05", "BK", 100, 0, 1000, 0)])
    B = book_state(df, 900.0, REG).set_index("broker")
    assert B.loc["BK", "unreal_pct"] < 0


# --------------------------------------------------------------------------- #
# daily aggregation
# --------------------------------------------------------------------------- #
def test_flows_split_by_class_and_sum_back():
    df = frame([("2026-01-05", "BK", 100, 0, 1000, 0),
                ("2026-01-05", "CC", 0, 60, 0, 1000)])
    S = daily_states(df, REG)
    total = S[["foreign", "bumn", "domestic", "unknown"]].sum(axis=1).iloc[0]
    assert total == pytest.approx(40.0)


def test_top3_share_is_one_when_a_single_broker_buys():
    df = frame([("2026-01-05", "BK", 100, 0, 1000, 0),
                ("2026-01-05", "CC", 0, 100, 0, 1000)])
    S = daily_states(df, REG)
    assert S["top3_share"].iloc[0] == pytest.approx(1.0)


def test_top3_share_falls_when_buying_is_spread():
    rows = [("2026-01-05", c, 100, 0, 1000, 0)
            for c in ("BK", "CC", "AK", "YP", "ZP", "LG")]
    S = daily_states(frame(rows), REG)
    assert S["top3_share"].iloc[0] == pytest.approx(0.5)
