"""§22-25 — the multibagger scenario layer that refuses to predict.

THE TESTS THIS FILE EXISTS FOR are the three refusals, because everything else
here is arithmetic that would be obvious if it broke:

  * `test_a_prediction_rung_value_cannot_be_constructed` — the module's whole
    claim is that it does not produce name-level probabilities. A `Value` with
    rung PREDICTION raises, so the refusal is structural rather than a
    convention someone can drift away from.
  * `test_the_scenario_and_the_base_rate_are_never_multiplied` — multiplying
    them yields something with the FORM of a probability and no referent, and
    it is exactly the number a reader would quote.
  * `test_a_horizon_outside_the_measured_range_raises` — A21 records a ten-year
    doubling rate being read as a one-year one, and the tilt INVERTS inside the
    range, so a fitted curve says nothing about either end.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot import multibagger                                    # noqa: E402
from idxbot.multibagger import (BASE_RATES, EFFECTIVE_N,          # noqa: E402
                                SCREEN_ONE_YEAR, Scenario, Value,
                                base_rate, compare, what_would_falsify)


def _s(**kw) -> Scenario:
    d = dict(ticker="XXXX", horizon_years=5.0, revenue_cagr=0.15,
             margin_now=0.10, margin_then=0.15, multiple_now=8.0,
             multiple_then=12.0, share_growth=0.02)
    d.update(kw)
    return Scenario(**d)


# ----------------------------------------------------------- THE REFUSALS

def test_a_prediction_rung_value_cannot_be_constructed():
    with pytest.raises(ValueError, match="does not produce PREDICTION"):
        Value(0.5, "PREDICTION", "chance this doubles")


def test_an_unknown_rung_is_refused():
    with pytest.raises(ValueError, match="unknown rung"):
        Value(1.0, "GUESS", "x")


def test_the_scenario_and_the_base_rate_are_never_multiplied():
    """Their product has the form of a probability and no referent."""
    src = open(os.path.join(os.path.dirname(__file__), os.pardir, "src",
                            "idxbot", "multibagger.py")).read()
    out = "\n".join(compare(_s()))
    assert "NOT COMBINED" in out
    assert "ARE NOT MULTIPLIED" in src
    #  the joint number must not appear anywhere in the output
    imp = _s().implied()["implied_multiple"].value
    br = base_rate(5.0).value
    assert f"{imp * br:.2f}" not in out


def test_a_horizon_outside_the_measured_range_raises():
    with pytest.raises(ValueError, match="outside the measured range"):
        base_rate(0.5)
    with pytest.raises(ValueError, match="outside the measured range"):
        base_rate(15.0)


def test_the_inversion_is_the_stated_reason_for_refusing_to_extrapolate():
    with pytest.raises(ValueError, match="INVERTS"):
        base_rate(20.0)


# ------------------------------------------------------- THE HORIZON TRAVELS

def test_every_base_rate_carries_its_horizon_in_the_note():
    """A21: a ten-year doubling rate written without 'over ten years' beside it
    was read as a one-year number, and the fix belongs in the code that prints
    it, not in the discipline of whoever quotes it."""
    for yrs in (1.0, 3.0, 10.0):
        v = base_rate(yrs)
        assert f"over {yrs:g} years" in v.note
        assert "IN-SAMPLE" in v.note
        assert "effective n" in v.note


def test_the_tilt_really_does_invert_in_the_stored_table():
    """The claim that licenses refusing to extrapolate must be true of the
    data, not merely asserted in the docstring."""
    one = dict(zip(("y", "uni", "dec", "core"), BASE_RATES[0]))
    ten = dict(zip(("y", "uni", "dec", "core"), BASE_RATES[-1]))
    assert one["dec"] < one["uni"], "no inversion at one year"
    assert ten["dec"] > ten["uni"], "no inversion at ten years"
    assert one["core"] < one["dec"] and ten["core"] > ten["dec"]


def test_the_base_rates_match_the_pinned_source_table():
    """`scripts/decade.py` BY_HORIZON is the source and is itself pinned by
    four tests. Two copies of a constant that stop matching is the failure
    A26's Pine drift guard exists for."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                    "scripts"))
    from decade import BY_HORIZON                              # noqa: PLC0415
    assert tuple(tuple(r) for r in BY_HORIZON) == BASE_RATES


def test_effective_n_is_carried_with_the_rate_not_left_in_a_memo():
    assert EFFECTIVE_N["whole_panel_10y"] == 56
    assert EFFECTIVE_N["per_decile_10y"] == 6
    assert EFFECTIVE_N["distinct_names_ever_in_decile"] == 30
    assert "56" in base_rate(10.0).note


def test_the_screen_rate_carries_its_own_one_year_horizon():
    """H26's 10.5% is a ONE-YEAR number and is stored with that attached."""
    assert SCREEN_ONE_YEAR["horizon_years"] == 1.0
    assert SCREEN_ONE_YEAR["p_double"] > SCREEN_ONE_YEAR["p_halve"]
    assert SCREEN_ONE_YEAR["skew"] == pytest.approx(
        SCREEN_ONE_YEAR["p_double"] / SCREEN_ONE_YEAR["p_halve"], rel=0.02)


# ------------------------------------------------------------- THE ARITHMETIC

def test_the_multiple_is_the_product_of_the_four_legs():
    s = _s(horizon_years=1.0, revenue_cagr=0.20, margin_now=0.10,
           margin_then=0.20, multiple_now=10.0, multiple_then=15.0,
           share_growth=0.0)
    imp = s.implied()
    assert imp["implied_multiple"].value == pytest.approx(1.2 * 2.0 * 1.5)


def test_dilution_divides_rather_than_multiplies():
    a = _s(share_growth=0.0).implied()["implied_multiple"].value
    b = _s(share_growth=0.10).implied()["implied_multiple"].value
    assert b < a


def test_an_absent_leg_is_neutral_not_zero():
    """A missing assumption must contribute a factor of 1, or a partly filled
    scenario silently collapses to zero and looks like a bearish view."""
    s = _s(margin_now=0.0, margin_then=0.0, multiple_now=0.0,
           multiple_then=0.0, share_growth=0.0, revenue_cagr=0.10,
           horizon_years=3.0)
    assert s.implied()["implied_multiple"].value == pytest.approx(1.1 ** 3)


def test_implied_cagr_and_multiple_agree():
    s = _s(horizon_years=4.0)
    imp = s.implied()
    m, c = imp["implied_multiple"].value, imp["implied_cagr"].value
    assert (1.0 + c) ** 4.0 == pytest.approx(m)


def test_a_non_positive_horizon_is_refused():
    with pytest.raises(ValueError, match="horizon_years must be positive"):
        _s(horizon_years=0.0).implied()


def test_every_assumption_is_labelled_thesis_and_every_derived_value_interpretation():
    imp = _s().implied()
    assert imp["revenue_factor"].rung == "THESIS"
    assert imp["margin_factor"].rung == "THESIS"
    assert imp["implied_multiple"].rung == "INTERPRETATION"
    assert imp["implied_cagr"].rung == "INTERPRETATION"
    assert base_rate(5.0).rung == "DATA"


# ------------------------------------------------------------ THE SENSITIVITY

def test_sensitivity_restores_the_original_assumption():
    s = _s(revenue_cagr=0.15)
    s.sensitivity("revenue_cagr", [0.0, 0.1, 0.3])
    assert s.revenue_cagr == 0.15


def test_sensitivity_restores_the_assumption_even_if_it_raises():
    s = _s(horizon_years=3.0)
    with pytest.raises(ValueError):
        s.sensitivity("horizon_years", [1.0, -1.0])
    assert s.horizon_years == 3.0


def test_sensitivity_is_monotone_in_growth():
    got = _s().sensitivity("revenue_cagr", [0.0, 0.10, 0.20, 0.30])
    vals = [v for _x, v in got]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_sensitivity_rejects_an_unknown_assumption():
    with pytest.raises(ValueError, match="no such assumption"):
        _s().sensitivity("vibes", [1.0])


# ------------------------------------------------------------ THE FALSIFIER

def test_every_supplied_leg_gets_a_disconfirming_observation():
    got = what_would_falsify(_s())
    assert len(got) == 4
    assert any("revenue" in g for g in got)
    assert any("margin" in g for g in got)
    assert any("multiple" in g for g in got)
    assert any("share count" in g for g in got)


def test_an_empty_scenario_says_so_rather_than_inventing_a_falsifier():
    s = Scenario(ticker="X", horizon_years=5.0)
    got = what_would_falsify(s)
    assert len(got) == 1 and "nothing can falsify it" in got[0]


def test_a_margin_that_is_not_assumed_to_expand_gets_no_margin_falsifier():
    got = what_would_falsify(_s(margin_now=0.20, margin_then=0.10))
    assert not any("margin fails to exceed" in g for g in got)
