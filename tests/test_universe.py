"""Tests for the survivorship-bias measurement.

The temptation this file guards against is not a bug, it is a fudge. The
delisted history is not obtainable, so there is a standing pull toward
inventing a correction factor and quietly applying it - which would turn a
KNOWN gap into a hidden assumption and make every downstream number look
defensible when it is not.

So the tests here check that the module bounds rather than corrects: that the
bias always points in the direction of overstatement, that a cap-weighted book
carries far less of it than an equal-weighted one (the finding that actually
matters, because it is what separates the repo's large-cap work from its
small-cap work), and that the audit reports zero coverage honestly rather than
degrading gracefully into silence.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.spine.universe import (DELIST_RATE_HIGH,             # noqa: E402
                                   DELIST_RATE_LOW, FORCED_RETURN,
                                   KNOWN_DELISTED, audit_universe,
                                   bias_estimate, caveat,
                                   expected_delisted_return, liquidity_shield,
                                   sensitivity)


# --------------------------------------------------------------------------
# the audit
# --------------------------------------------------------------------------
def test_a_universe_of_only_live_names_is_flagged_biased():
    a = audit_universe(["BBCA", "TLKM", "ASII"])
    assert a["survivorship_biased"] is True
    assert a["coverage"] == 0.0
    assert len(a["missing"]) == len(KNOWN_DELISTED)


def test_a_universe_containing_dead_names_is_not_flagged():
    a = audit_universe(["BBCA"] + list(KNOWN_DELISTED))
    assert a["survivorship_biased"] is False
    assert a["coverage"] == pytest.approx(1.0)


def test_the_audit_accepts_tickers_with_or_without_the_suffix():
    a = audit_universe([f"{t}.JK" for t in KNOWN_DELISTED])
    assert a["coverage"] == pytest.approx(1.0)


def test_partial_coverage_is_reported_as_partial():
    half = list(KNOWN_DELISTED)[:len(KNOWN_DELISTED) // 2]
    a = audit_universe(["BBCA"] + half)
    assert 0.0 < a["coverage"] < 1.0
    assert a["survivorship_biased"] is False     # some dead names ARE present


# --------------------------------------------------------------------------
# the bias always points one way
# --------------------------------------------------------------------------
def test_survivorship_bias_always_overstates_never_understates():
    for r_s in (-0.20, 0.0, 0.10, 0.50):
        for f in (0.01, 0.05, 0.10):
            b = bias_estimate(r_s, f)
            assert b["bias"] >= 0, "a survivorship correction cannot flatter"
            assert b["true"] <= b["observed"]


def test_no_delistings_means_no_bias():
    b = bias_estimate(0.10, 0.0)
    assert b["bias"] == pytest.approx(0.0)
    assert b["true"] == pytest.approx(0.10)


def test_the_bias_grows_with_the_delisting_rate():
    prev = -1.0
    for f in (0.01, 0.02, 0.04, 0.08):
        b = bias_estimate(0.10, f)["bias"]
        assert b > prev
        prev = b


def test_a_total_loss_costs_the_full_delisted_weight():
    """f = 8%, r_s = +10%, r_d = -100% -> bias = 0.08 * 1.10 = 8.8pp."""
    b = bias_estimate(0.10, 0.08, delisted_return=-1.0)
    assert b["bias"] == pytest.approx(0.088, abs=1e-9)


# --------------------------------------------------------------------------
# the finding that matters: weighting decides how much this hurts
# --------------------------------------------------------------------------
def test_a_cap_weighted_book_carries_far_less_bias():
    """A name about to delist is a micro cap. This is why the repo's large-cap
    work is close to safe and its equal-weight small-cap work is not."""
    eq = bias_estimate(0.10, DELIST_RATE_HIGH, weighting="equal")["bias"]
    cap = bias_estimate(0.10, DELIST_RATE_HIGH, weighting="cap")["bias"]
    assert cap < eq / 10


def test_an_unknown_weighting_is_an_error_not_a_default():
    with pytest.raises(ValueError):
        bias_estimate(0.10, 0.05, weighting="risk_parity")


def test_the_sensitivity_table_spans_both_weightings_and_the_rate_range():
    s = sensitivity(0.10)
    assert set(s["weighting"]) == {"equal", "cap"}
    assert s["delist_rate"].min() <= DELIST_RATE_LOW
    assert s["delist_rate"].max() >= DELIST_RATE_HIGH
    assert (s["bias_pp"] >= 0).all()


# --------------------------------------------------------------------------
# what a delisted holding returns
# --------------------------------------------------------------------------
def test_the_blended_delisted_return_is_a_heavy_loss_but_not_total():
    r = expected_delisted_return()
    assert FORCED_RETURN < r < 0.0


def test_all_forced_delistings_are_a_total_loss():
    assert expected_delisted_return(1.0) == pytest.approx(FORCED_RETURN)


def test_all_voluntary_delistings_can_pay_a_premium():
    assert expected_delisted_return(0.0) > 0


# --------------------------------------------------------------------------
# the liquidity shield
# --------------------------------------------------------------------------
def test_a_tighter_liquidity_filter_keeps_fewer_names():
    s = pd.Series(np.linspace(0.0, 0.5, 100))
    assert (liquidity_shield(s, 0.05)["kept"]
            < liquidity_shield(s, 0.20)["kept"])


def test_the_shield_reports_nothing_when_there_is_nothing_to_report():
    assert liquidity_shield(pd.Series(dtype=float)) == {}


def test_the_shield_does_not_return_a_corrected_return():
    """It must not invent a correction - the correction is unidentifiable."""
    r = liquidity_shield(pd.Series([0.01, 0.5, 0.02]))
    assert not any("return" in k or "corrected" in k for k in r)


# --------------------------------------------------------------------------
# the caveat that has to travel with every backtest
# --------------------------------------------------------------------------
def test_the_caveat_names_the_bound_and_refuses_to_call_it_a_correction():
    c = caveat("equal")
    assert "bound, not a correction" in c
    assert "survivorship" in c.lower()


def test_the_caveat_reads_as_english_for_both_weightings():
    assert "On an equal-weighted" in caveat("equal")
    assert "On a cap-weighted" in caveat("cap")


def test_the_equal_weighted_caveat_quotes_a_bigger_number_than_the_cap_one():
    import re
    eq = re.findall(r"roughly ([\d.]+) to ([\d.]+)", caveat("equal"))[0]
    cap = re.findall(r"roughly ([\d.]+) to ([\d.]+)", caveat("cap"))[0]
    assert float(eq[1]) > float(cap[1])
