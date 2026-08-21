"""Tests for the execution-edge and flow-memory analysis.

The claim these support - that some members' clients consistently buy cheaper
than the day's average participant - is the only positive finding to come out of
broker data in this whole project. So the tests are built around the ways it
could be an artefact rather than a fact:

  SIGN. Buying below VWAP must score POSITIVE and selling below must score
  negative. A flipped sign would invert the entire ranking and read as a
  discovery about the wrong people.
  IMPUTATION. A broker absent from a side has no average price. Filling that
  with the VWAP would manufacture a zero edge out of nothing and drag every
  broker toward the middle.
  WEIGHTING. One lucky fill on a hundred lots must not outrank a year of
  millions.
  NOISE. A table of twenty brokers each a tenth of a percent from zero is
  exactly the shape noise takes, so the split-half test has to detect a planted
  edge AND fail to detect one that is not there.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from broker_economics import (edge_persistence, edge_vs_size,       # noqa: E402
                              execution_edge, persistence, zero_sum_check)


def day(brokers, buy_lot, buy_avg, sell_lot, sell_avg, vwap=1000.0,
        date="2026-01-05", ticker="TEST"):
    return pd.DataFrame({"ticker": ticker, "date": pd.Timestamp(date),
                         "broker": brokers, "buy_lot": buy_lot,
                         "buy_avg": buy_avg, "sell_lot": sell_lot,
                         "sell_avg": sell_avg, "vwap": vwap})


# --------------------------------------------------------------------------
# sign and arithmetic
# --------------------------------------------------------------------------
def test_buying_below_the_average_scores_positive():
    d = day(["AA"], [100.0], [990.0], [0.0], [np.nan])
    e = execution_edge(d)
    assert e.loc["AA", "edge_buy"] == pytest.approx(0.01)
    assert e.loc["AA", "edge_all"] == pytest.approx(0.01)


def test_buying_above_the_average_scores_negative():
    e = execution_edge(day(["AA"], [100.0], [1010.0], [0.0], [np.nan]))
    assert e.loc["AA", "edge_buy"] == pytest.approx(-0.01)


def test_selling_above_the_average_scores_positive():
    e = execution_edge(day(["AA"], [0.0], [np.nan], [100.0], [1010.0]))
    assert e.loc["AA", "edge_sell"] == pytest.approx(0.01)


def test_selling_below_the_average_scores_negative():
    e = execution_edge(day(["AA"], [0.0], [np.nan], [100.0], [990.0]))
    assert e.loc["AA", "edge_sell"] == pytest.approx(-0.01)


def test_trading_exactly_at_the_average_scores_zero():
    e = execution_edge(day(["AA"], [100.0], [1000.0], [100.0], [1000.0]))
    assert e.loc["AA", "edge_all"] == pytest.approx(0.0)


def test_an_unlisted_side_is_omitted_not_imputed_as_average():
    """Filling a missing side with the VWAP would invent a zero edge."""
    e = execution_edge(day(["AA"], [100.0], [980.0], [0.0], [np.nan]))
    assert np.isnan(e.loc["AA", "edge_sell"])
    # the blended edge must reflect the buy side alone, undiluted
    assert e.loc["AA", "edge_all"] == pytest.approx(0.02)


def test_the_blend_is_weighted_by_lots_not_by_side():
    d = day(["AA"], [900.0], [990.0], [100.0], [990.0])   # sells BELOW vwap
    e = execution_edge(d)
    # buy edge +1% on 900 lots, sell edge -1% on 100 lots -> +0.8%
    assert e.loc["AA", "edge_all"] == pytest.approx(0.008)


def test_a_big_record_outranks_one_lucky_fill():
    d = pd.concat([
        day(["BIG"], [1_000_000.0], [999.0], [0.0], [np.nan], date="2026-01-05"),
        day(["LUCKY"], [100.0], [900.0], [0.0], [np.nan], date="2026-01-05"),
    ], ignore_index=True)
    e = execution_edge(d)
    # LUCKY has the bigger edge but BIG has the volume; both are reported and
    # the caller filters on lots — the point is the lots column is honest
    assert e.loc["BIG", "lots_total"] > e.loc["LUCKY", "lots_total"] * 100


def test_a_day_with_no_vwap_is_dropped_rather_than_divided_by_zero():
    d = day(["AA"], [100.0], [990.0], [0.0], [np.nan], vwap=0.0)
    assert execution_edge(d).empty


def test_days_seen_counts_both_sides_and_never_goes_missing():
    d = pd.concat([day(["AA"], [10.0], [990.0], [0.0], [np.nan]),
                   day(["AA"], [0.0], [np.nan], [10.0], [1010.0],
                       date="2026-01-06")], ignore_index=True)
    e = execution_edge(d)
    assert e.loc["AA", "days_seen"] == 2
    assert np.isfinite(e["days_buy"]).all() and np.isfinite(e["days_sell"]).all()


# --------------------------------------------------------------------------
# the zero-sum constraint
# --------------------------------------------------------------------------
def test_a_complete_market_nets_to_zero_edge():
    """One broker's gain against the average is another's loss, exactly."""
    # AA buys 100 at 990, BB buys 100 at 1010; VWAP of the two is 1000
    d = day(["AA", "BB"], [100.0, 100.0], [990.0, 1010.0],
            [0.0, 0.0], [np.nan, np.nan], vwap=1000.0)
    z = zero_sum_check(execution_edge(d))
    assert z["weighted_edge"] == pytest.approx(0.0, abs=1e-9)
    assert z["brokers"] == 2


def test_the_zero_sum_check_of_nothing_is_empty():
    assert zero_sum_check(pd.DataFrame()) == {}


# --------------------------------------------------------------------------
# is the edge real, or a list of coin flips
# --------------------------------------------------------------------------
def planted(n_days=200, edges=(0.02, 0.015, 0.01, 0.005, 0.002,
                               -0.002, -0.005, -0.01, -0.015, -0.02),
            seed=0, noise=0.005):
    """Brokers with a FIXED true edge, observed through noise.

    Ten of them, not five, because ``edge_persistence`` refuses to correlate
    fewer than eight pairs - a Spearman on five points cannot reach any useful
    p-value, so returning one would be worse than returning nothing.
    """
    rng = np.random.default_rng(seed)
    codes = [f"B{i}" for i in range(len(edges))]
    rows = []
    for d in pd.bdate_range("2025-01-01", periods=n_days):
        for c, e in zip(codes, edges):
            rows.append({"ticker": "T", "date": d, "broker": c,
                         "buy_lot": 1000.0,
                         "buy_avg": 1000.0 * (1 - e - rng.normal(0, noise)),
                         "sell_lot": 0.0, "sell_avg": np.nan, "vwap": 1000.0})
    return pd.DataFrame(rows)


def test_split_half_recovers_a_planted_edge():
    sp = edge_persistence(planted(), min_lots=0, min_days=5)
    assert sp["spearman"] > 0.8
    assert sp["p"] < 0.05
    assert sp["same_sign"] >= sp["n"] - 1


def test_split_half_does_not_find_an_edge_that_is_not_there():
    """Every broker identical: the test must fire at its stated rate, no more.

    Asserting one seed comes back insignificant is not a test of anything - an
    honest test at alpha = 0.05 is SUPPOSED to fire on one null sample in
    twenty, so a single-seed assertion is a coin that lands wrong 5% of the
    time and would be silenced by whoever it inconveniences. What has to hold
    is the RATE. Forty null samples, no broker with any edge at all: if the
    split-half correlation is calibrated the false-positive rate sits near 5%,
    and if it is picking up structure that is not in the data the rate runs
    far above it.
    """
    p = [edge_persistence(planted(edges=(0.0,) * 10, noise=0.01, seed=s),
                          min_lots=0, min_days=5)["p"] for s in range(40)]
    rate = float(np.mean(np.array(p) < 0.05))
    # 40 draws puts the standard error at ~3.4pp, so the band is generous on
    # the high side and still catches a test that fires two or three times too
    # often. The median is the stronger check: uniform p-values sit near 0.5.
    assert rate <= 0.20, f"false-positive rate {rate:.0%}, expected about 5%"
    assert 0.25 < float(np.median(p)) < 0.75


def test_split_half_needs_both_halves_to_have_a_record():
    tiny = planted(n_days=6)
    assert edge_persistence(tiny, min_lots=0, min_days=5).get("spearman") is None


def test_size_test_finds_a_planted_size_effect():
    rows = []
    for i, (lots, edge) in enumerate([(1e3, 0.03), (1e4, 0.02), (1e5, 0.01),
                                      (1e6, 0.0), (1e7, -0.01), (1e8, -0.02),
                                      (1e9, -0.03), (1e10, -0.04)]):
        rows.append({"ticker": "T", "date": pd.Timestamp("2026-01-05"),
                     "broker": f"B{i}", "buy_lot": lots,
                     "buy_avg": 1000.0 * (1 - edge), "sell_lot": 0.0,
                     "sell_avg": np.nan, "vwap": 1000.0})
    sz = edge_vs_size(execution_edge(pd.DataFrame(rows)))
    assert sz["pearson"] < -0.9 and sz["p"] < 0.01


def test_size_test_reports_nothing_when_size_does_not_matter():
    sz = edge_vs_size(execution_edge(planted(n_days=5)))
    assert sz == {} or sz["p"] > 0.05


# --------------------------------------------------------------------------
# does flow have memory
# --------------------------------------------------------------------------
def test_a_campaign_shows_positive_autocorrelation():
    """One broker buying every day, another alternating."""
    rows = []
    for i, d in enumerate(pd.bdate_range("2025-01-01", periods=120)):
        rows.append({"ticker": "T", "date": d, "broker": "STEADY",
                     "buy_lot": 1000.0 + i, "sell_lot": 0.0})
        rows.append({"ticker": "T", "date": d, "broker": "FLIPPY",
                     "buy_lot": 1000.0 if i % 2 else 0.0,
                     "sell_lot": 0.0 if i % 2 else 1000.0})
    P = persistence(pd.DataFrame(rows))
    assert P.loc["STEADY", "ac1"] > 0.5
    assert P.loc["FLIPPY", "ac1"] < -0.5


def test_persistence_ignores_a_broker_with_too_short_a_record():
    rows = [{"ticker": "T", "date": d, "broker": "SHORT", "buy_lot": 100.0,
             "sell_lot": 0.0}
            for d in pd.bdate_range("2025-01-01", periods=10)]
    assert persistence(pd.DataFrame(rows)).empty


def test_persistence_of_a_constant_flow_is_undefined_not_one():
    """Zero variance carries no information about memory."""
    rows = [{"ticker": "T", "date": d, "broker": "FLAT", "buy_lot": 100.0,
             "sell_lot": 0.0}
            for d in pd.bdate_range("2025-01-01", periods=60)]
    assert persistence(pd.DataFrame(rows)).empty
