"""Tests for the broker-habit reconstruction.

Two things carry the weight:

  1. Weighted-average cost accounting. Buying moves the basis, selling does not.
     Get that backwards and every exit percentage is wrong in a way that still
     looks plausible.
  2. The visibility bound. Top-10 truncation means an inventory can be mostly
     invented; the bound is what stops an invented one being reported.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from broker_habits import exit_profile, ledger, visibility          # noqa: E402
from broker_collect import price_sane                                # noqa: E402


def frame(rows):
    return pd.DataFrame(rows, columns=["date", "broker", "buy_lot", "sell_lot",
                                       "buy_avg", "sell_avg"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


# --------------------------------------------------------------------------- #
# cost-basis accounting
# --------------------------------------------------------------------------- #
def test_a_single_buy_sets_the_basis():
    led = ledger(frame([("2026-01-05", "BK", 100, 0, 1000, 0)]), "BK")
    assert led.iloc[0]["basis"] == pytest.approx(1000.0)
    assert led.iloc[0]["inventory"] == pytest.approx(100.0)


def test_a_second_buy_weights_the_basis_by_size():
    led = ledger(frame([("2026-01-05", "BK", 100, 0, 1000, 0),
                        ("2026-01-06", "BK", 300, 0, 2000, 0)]), "BK")
    # (1000*100 + 2000*300) / 400 = 1750
    assert led.iloc[-1]["basis"] == pytest.approx(1750.0)


def test_selling_does_not_move_the_basis():
    led = ledger(frame([("2026-01-05", "BK", 100, 0, 1000, 0),
                        ("2026-01-06", "BK", 0, 50, 0, 1500)]), "BK")
    assert led.iloc[-1]["basis"] == pytest.approx(1000.0)


def test_a_sale_realises_against_the_basis_not_the_last_price():
    led = ledger(frame([("2026-01-05", "BK", 100, 0, 1000, 0),
                        ("2026-01-06", "BK", 0, 50, 0, 1200)]), "BK")
    r = led.iloc[-1]
    assert r["pl_pct"] == pytest.approx(0.20)
    assert r["realised"] == pytest.approx(50 * (1200 - 1000) * 100)


def test_a_losing_sale_is_negative():
    led = ledger(frame([("2026-01-05", "BK", 100, 0, 1000, 0),
                        ("2026-01-06", "BK", 0, 50, 0, 900)]), "BK")
    assert led.iloc[-1]["pl_pct"] == pytest.approx(-0.10)


def test_a_sale_before_any_buy_realises_nothing():
    """No basis exists yet, so no P/L may be claimed."""
    led = ledger(frame([("2026-01-05", "BK", 0, 50, 0, 1200)]), "BK")
    assert np.isnan(led.iloc[0]["pl_pct"])


def test_inventory_tracks_net_lots():
    led = ledger(frame([("2026-01-05", "BK", 100, 0, 1000, 0),
                        ("2026-01-06", "BK", 20, 50, 1100, 1200)]), "BK")
    assert led.iloc[-1]["inventory"] == pytest.approx(70.0)


# --------------------------------------------------------------------------- #
# exit profile
# --------------------------------------------------------------------------- #
def test_take_profit_and_cut_loss_are_separated_by_sign():
    led = ledger(frame([("2026-01-05", "BK", 1000, 0, 1000, 0),
                        ("2026-01-06", "BK", 0, 100, 0, 1100),
                        ("2026-01-07", "BK", 0, 100, 0, 1200),
                        ("2026-01-08", "BK", 0, 100, 0, 950)]), "BK")
    p = exit_profile(led)
    assert p["take_profit"] == pytest.approx(0.15)     # median of +10%, +20%
    assert p["cut_loss"] == pytest.approx(-0.05)
    assert p["win_rate"] == pytest.approx(2 / 3)


def test_too_few_sales_gives_no_profile():
    led = ledger(frame([("2026-01-05", "BK", 100, 0, 1000, 0),
                        ("2026-01-06", "BK", 0, 10, 0, 1100)]), "BK")
    assert exit_profile(led) == {}


# --------------------------------------------------------------------------- #
# visibility: what truncation hides
# --------------------------------------------------------------------------- #
def test_a_broker_present_every_day_has_no_unseen_bound():
    df = frame([("2026-01-05", "BK", 100, 0, 1000, 0),
                ("2026-01-06", "BK", 100, 0, 1000, 0),
                ("2026-01-05", "CC", 50, 0, 1000, 0),
                ("2026-01-06", "CC", 50, 0, 1000, 0)])
    V = visibility(df).set_index("broker")
    assert V.loc["BK", "appearance"] == pytest.approx(1.0)
    assert V.loc["BK", "unseen_bound"] == pytest.approx(0.0)


def test_a_missing_day_is_bounded_by_the_smallest_visible_broker():
    df = frame([("2026-01-05", "BK", 100, 0, 1000, 0),
                ("2026-01-05", "CC", 30, 0, 1000, 0),
                ("2026-01-06", "CC", 40, 0, 1000, 0)])
    V = visibility(df).set_index("broker")
    # BK missed 06, when the smallest visible total was CC's 40
    assert V.loc["BK", "appearance"] == pytest.approx(0.5)
    assert V.loc["BK", "unseen_bound"] == pytest.approx(40.0)


def test_the_bound_ratio_flags_an_unmeasurable_broker():
    df = frame([("2026-01-05", "BK", 1, 0, 1000, 0),
                ("2026-01-05", "CC", 500, 0, 1000, 0),
                ("2026-01-06", "CC", 500, 0, 1000, 0),
                ("2026-01-07", "CC", 500, 0, 1000, 0)])
    V = visibility(df).set_index("broker")
    assert V.loc["BK", "bound_ratio"] > 1.0      # unseen dwarfs observed


# --------------------------------------------------------------------------- #
# the guard that keeps bad prices out of the store
# --------------------------------------------------------------------------- #
def test_price_sanity_passes_when_there_are_no_prices():
    g = pd.DataFrame({"buy_avg": [0.0], "sell_avg": [0.0]})
    ok, _ = price_sane(g, "BBCA", pd.Timestamp("2026-08-19"))
    assert ok


def test_price_sanity_passes_when_the_date_has_no_bar():
    g = pd.DataFrame({"buy_avg": [6300.0], "sell_avg": [6300.0]})
    ok, why = price_sane(g, "BBCA", pd.Timestamp("2035-01-01"))
    assert ok and "date" in why.lower() or ok
