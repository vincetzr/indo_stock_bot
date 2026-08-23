"""Tests for §9.3 cohort P&L.

Two defects lived in this module and both produced confident nonsense from
correct-looking code. Each has a test that reproduces the original and requires
the fix.

  WAC STARTING AT ZERO. §9.3's formula is
  realized = sell_vol x (sell_avg - WAC_{t-1}), and the walk begins with WAC at
  zero. A cohort already long when the series starts books its opening sell's
  ENTIRE PROCEEDS as profit. On a flat buy-and-sell at the same price it
  returned +Rp 10,000,000 against a true P&L of zero. The shuffled-label null
  read +6.3 bps with a CI excluding zero, which is how it was caught.

  UNREALISED ON NEGATIVE INVENTORY. inventory x (close - WAC) where inventory
  is negative is a position the data never saw acquired, priced against a
  meaningless basis. It produced full-path margins of -13,000 bps: a 130% loss
  on gross traded value, which cannot happen.

The naming is also tested. §9.1 requires cohort_pnl everywhere and never
broker_profit, because the code aggregates thousands of client accounts and the
conceptual error leaks into the dossiers the moment the variable is misnamed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.spine.cohort_pnl import (crossing_ratio,           # noqa: E402
                                     margin_bps,
                                     negative_inventory_share, round_trips,
                                     shuffle_broker_labels, walk_forward)


def series(buys, sells, buy_px, sell_px, start="2025-01-01"):
    n = len(buys)
    return pd.DataFrame({
        "date": pd.bdate_range(start, periods=n),
        "broker": "XX", "ticker": "TEST",
        "buy_lot": buys, "sell_lot": sells,
        "buy_avg": buy_px, "sell_avg": sell_px,
    })


# --------------------------------------------------------------------------
# the WAC-at-zero bug
# --------------------------------------------------------------------------
def test_an_opening_sell_is_not_booked_as_pure_profit():
    """THE regression. A cohort already long sells first; its cost is unknown
    and must not be taken as zero."""
    g = series([0, 0, 100, 0], [100, 0, 0, 100],
               [np.nan, np.nan, 1000, np.nan], [1000, np.nan, np.nan, 1000])
    w = walk_forward(g)
    assert w["realized"].sum() == pytest.approx(0.0), \
        "buying and selling 10,000 shares at the same price is zero P&L"
    assert w["unattributable_sh"].sum() > 0, \
        "the opening sell must be recorded as unattributable, not as profit"


def test_shares_sold_beyond_holdings_are_counted_not_priced():
    g = series([0, 0], [50, 50], [np.nan, np.nan], [900, 1100])
    w = walk_forward(g)
    assert w["realized"].sum() == 0.0
    assert w["unattributable_sh"].sum() == pytest.approx(10000.0)


def test_a_genuine_gain_is_still_measured():
    """The fix must not simply zero everything out."""
    g = series([100, 0], [0, 100], [1000, np.nan], [np.nan, 1100])
    w = walk_forward(g)
    assert w["realized"].sum() == pytest.approx(10000 * 100.0)


# --------------------------------------------------------------------------
# round trips are the clean estimate, and clean means no WAC
# --------------------------------------------------------------------------
def test_a_round_trip_is_sell_value_minus_buy_value():
    """§9.3: inside an episode that opens and closes near flat, everything
    bought is sold, so P&L needs no cost basis and no starting inventory."""
    g = series([0, 100, 0, 0, 0], [0, 0, 0, 100, 0],
               [np.nan, 1000, np.nan, np.nan, np.nan],
               [np.nan, np.nan, np.nan, 1100, np.nan])
    rt = round_trips(walk_forward(g))
    assert len(rt) == 1
    r = rt.iloc[0]
    assert r["pnl"] == pytest.approx(10000 * 1100.0 - 10000 * 1000.0)
    assert margin_bps(r["pnl"], r["gross_value"]) == pytest.approx(476.19, abs=0.1)


def test_a_losing_round_trip_is_negative():
    g = series([0, 100, 0, 0, 0], [0, 0, 0, 100, 0],
               [np.nan, 1100, np.nan, np.nan, np.nan],
               [np.nan, np.nan, np.nan, 1000, np.nan])
    rt = round_trips(walk_forward(g))
    assert len(rt) == 1 and rt.iloc[0]["pnl"] < 0


def test_a_one_day_in_and_out_is_not_an_episode():
    """A same-window spike is a crossing artefact, not an accumulation."""
    g = series([0, 100, 0], [0, 100, 0],
               [np.nan, 1000, np.nan], [np.nan, 1000, np.nan])
    assert round_trips(walk_forward(g)).empty


def test_a_series_that_never_returns_to_flat_yields_no_episode():
    g = series([0] + [100] * 6, [0] * 7,
               [np.nan] + [1000] * 6, [np.nan] * 7)
    assert round_trips(walk_forward(g)).empty


# --------------------------------------------------------------------------
# §9.2's limits are measured, not asserted
# --------------------------------------------------------------------------
def test_negative_inventory_is_measured_not_hidden():
    """It is the direct measure of the starting-inventory problem, so the walk
    must LET inventory go negative rather than clamping it."""
    g = series([0, 0, 0], [100, 100, 0], [np.nan] * 3, [1000, 1000, np.nan])
    w = walk_forward(g)
    assert (w["inventory"] < 0).any()
    assert negative_inventory_share(w) > 0.5


def test_crossing_ratio_is_one_when_both_sides_match():
    g = series([100, 100], [100, 100], [1000, 1000], [1000, 1000])
    assert crossing_ratio(g) == pytest.approx(1.0)


def test_crossing_ratio_is_zero_for_one_sided_flow():
    g = series([100, 100], [0, 0], [1000, 1000], [np.nan, np.nan])
    assert crossing_ratio(g) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# the null
# --------------------------------------------------------------------------
def test_the_null_shuffles_labels_within_a_ticker_day_only():
    """It must preserve the day's total flow and the size distribution, and
    destroy only WHICH code did what — that is the claim being tested."""
    d = pd.DataFrame({
        "date": ["2025-01-01"] * 3 + ["2025-01-02"] * 3,
        "ticker": "T", "broker": list("abcabc"),
        "buy_lot": [1, 2, 3, 10, 20, 30], "sell_lot": 0,
        "buy_avg": 100.0, "sell_avg": np.nan})
    s = shuffle_broker_labels(d, seed=1)
    for day, g in s.groupby("date"):
        o = d[d.date == day]
        assert sorted(g["broker"]) == sorted(o["broker"])
        assert g["buy_lot"].sum() == o["buy_lot"].sum()


def test_margin_bps_is_per_rupiah_traded_so_size_cancels():
    """§9.3 wants margin_bps not absolute rupiah, precisely so a large and a
    small broker with the same edge read the same."""
    assert margin_bps(100.0, 10_000.0) == pytest.approx(
        margin_bps(1_000_000.0, 100_000_000.0))


# --------------------------------------------------------------------------
# naming discipline — §9.1
# --------------------------------------------------------------------------
def test_the_module_never_calls_it_broker_profit():
    """§9.1: "Name it cohort_pnl everywhere in code and output. Never
    broker_profit. The naming discipline stops the conceptual error leaking
    into the dossiers." A broker code is not a client."""
    import idxbot.spine.cohort_pnl as m
    src = open(m.__file__).read()
    assert "broker_profit" not in src.replace(
        "Never ``broker_profit``", "").replace("never ``broker_profit``", "")
    assert "cohort_pnl" in src
