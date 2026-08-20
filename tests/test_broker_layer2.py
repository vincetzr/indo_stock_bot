"""Tests for broker collection and the pre-registered layer-2 protocol.

Two properties carry the weight here:

  1. `is_complete` - a full rekap BALANCES because every share bought was sold.
     That single arithmetic check is what separates data that can answer the
     question from data that cannot, so it must never be fudged.
  2. `protocol_hash` - the hypotheses are frozen before the data exists. If the
     hash can drift without the hypotheses changing, or stay put when they DO
     change, the pre-registration is decorative.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import layer2_protocol as L2                                       # noqa: E402
from broker_collect import coverage, is_complete, store_path       # noqa: E402


def frame(buy, sell, ticker="ADRO", date="2026-08-19"):
    return pd.DataFrame({"date": [date] * len(buy), "ticker": [ticker] * len(buy),
                         "broker": [f"B{i}" for i in range(len(buy))],
                         "buy_lot": buy, "sell_lot": sell})


# --------------------------------------------------------------------------- #
# completeness
# --------------------------------------------------------------------------- #
def test_a_balanced_rekap_is_complete():
    assert is_complete(frame([100.0, 50.0], [50.0, 100.0])) is True


def test_a_truncated_table_is_not_complete():
    assert is_complete(frame([100.0, 50.0], [10.0, 20.0])) is False


def test_an_empty_table_is_not_complete():
    assert is_complete(frame([0.0], [0.0])) is False


def test_completeness_tolerates_only_floating_point_noise():
    assert is_complete(frame([1e9], [1e9 * (1 + 1e-9)])) is True
    assert is_complete(frame([1e9], [1e9 * 1.001])) is False


def test_store_path_is_one_file_per_ticker_day():
    a = store_path("ADRO", pd.Timestamp("2026-08-19"))
    b = store_path("ADRO", pd.Timestamp("2026-08-20"))
    assert a != b and a.endswith("ADRO_20260819.csv.gz")


def test_coverage_counts_complete_days_separately_from_days():
    df = pd.concat([
        frame([100.0], [100.0], "ADRO", "2026-08-18").assign(complete=True),
        frame([100.0], [10.0], "ADRO", "2026-08-19").assign(complete=False)])
    C = coverage(df)
    assert int(C.loc["ADRO", "days"]) == 2
    assert int(C.loc["ADRO", "complete_days"]) == 1


def test_coverage_is_empty_without_data():
    assert coverage(pd.DataFrame()).empty


# --------------------------------------------------------------------------- #
# the protocol is frozen
# --------------------------------------------------------------------------- #
def test_the_hash_is_stable_across_calls():
    assert L2.protocol_hash() == L2.protocol_hash()


def test_changing_a_hypothesis_changes_the_hash(monkeypatch):
    before = L2.protocol_hash()
    monkeypatch.setattr(L2, "HYPOTHESES",
                        L2.HYPOTHESES + [{"id": "H5", "claim": "x",
                                          "horizon": 1, "direction": "positive"}])
    assert L2.protocol_hash() != before


def test_changing_alpha_changes_the_hash(monkeypatch):
    before = L2.protocol_hash()
    monkeypatch.setattr(L2, "ALPHA", 0.10)
    assert L2.protocol_hash() != before


def test_every_hypothesis_is_directional_and_has_a_horizon():
    for h in L2.HYPOTHESES:
        assert h["direction"] in ("positive", "negative")
        assert h["horizon"] > 0
        assert h["claim"]


# --------------------------------------------------------------------------- #
# power
# --------------------------------------------------------------------------- #
def test_smaller_effects_need_more_data():
    a = L2.required_n(0.20, 0.0125, 0.8)
    b = L2.required_n(0.10, 0.0125, 0.8)
    assert b > a


def test_required_n_scales_as_one_over_effect_squared():
    a = L2.required_n(0.20, 0.0125, 0.8)
    b = L2.required_n(0.10, 0.0125, 0.8)
    assert b / a == pytest.approx(4.0, rel=0.02)


def test_a_zero_effect_is_unreachable():
    assert L2.required_n(0.0, 0.0125, 0.8) > 10 ** 8


def test_detectable_effect_shrinks_as_data_grows():
    assert L2.detectable_effect(1000, 0.0125, 0.8) < \
        L2.detectable_effect(100, 0.0125, 0.8)


def test_detectable_effect_and_required_n_are_inverses():
    n = L2.required_n(0.15, 0.0125, 0.8)
    assert L2.detectable_effect(n, 0.0125, 0.8) == pytest.approx(0.15, rel=0.02)


def test_one_bar_of_data_can_detect_nothing():
    assert not np.isfinite(L2.detectable_effect(1, 0.0125, 0.8))


def test_clustering_reduces_the_effective_sample():
    """40 names x 250 days is not 10,000 independent draws."""
    raw = 40 * 250
    assert L2.effective_n(raw, 40) < raw


def test_a_single_name_is_not_penalised_for_clustering():
    assert L2.effective_n(60, 1) == 60


def test_effective_n_is_zero_without_data():
    assert L2.effective_n(0, 40) == 0
