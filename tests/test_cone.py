"""Tests for the price×time cone laws and the Pine file that carries them.

Two things can break here and both are silent. The laws can be extrapolated
past their fitted domain and return a confident number off the end of a line;
and the coefficients — which necessarily exist twice, because Pine cannot
import anything — can drift apart. Both are pinned.
"""

from __future__ import annotations

import math
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.cone import (DAYS_PER_SESSION, HORIZON, PROB_LAW,   # noqa: E402
                         RACE_LAW, SIGMA_MAX, SIGMA_MIN, TIME_LAW, VOL_DECILES,
                         cone, in_domain, p_target_first, p_touch, sessions_to,
                         vol_decile)

PINE = os.path.join(os.path.dirname(__file__), os.pardir, "pine",
                    "IDX_Suite.pine")


# ================================================= the shape of the answer ===
def test_a_further_target_takes_longer():
    s = 0.025
    assert (sessions_to(1.05, s) < sessions_to(1.20, s)
            < sessions_to(1.50, s) < sessions_to(2.00, s))


def test_a_further_target_is_less_likely():
    s = 0.025
    assert p_touch(1.05, s) > p_touch(1.20, s) > p_touch(2.00, s)


def test_a_more_volatile_name_gets_there_sooner_and_more_often():
    """H32's clearest result: volatility is the clock. Median sessions to +20%
    ran 89 in the calmest decile against 30 in the wildest."""
    assert sessions_to(1.20, 0.055) < sessions_to(1.20, 0.013)
    assert p_touch(1.20, 0.055) > p_touch(1.20, 0.013)


def test_volatility_raises_the_downside_too_which_is_the_whole_point():
    """A22: a volatile name is likelier to touch ANY level. A screen that only
    reported the upside of that would be selling variance as edge."""
    assert p_touch(0.80, 0.055) > p_touch(0.80, 0.013)


def test_the_quartiles_are_ordered():
    for s in (0.013, 0.025, 0.055):
        assert (sessions_to(1.20, s, "q1") < sessions_to(1.20, s, "med")
                < sessions_to(1.20, s, "q3"))


def test_probabilities_stay_in_the_unit_interval_across_the_domain():
    for s in (SIGMA_MIN, 0.02, 0.04, SIGMA_MAX):
        for m in (1.05, 1.2, 1.5, 2.0, 0.9, 0.8, 0.67, 0.5):
            p = p_touch(m, s)
            assert 0.0 < p < 1.0, (m, s, p)


# ======================================================= the trend state =====
def test_the_stack_helps_the_downside_nearly_twice_as_much_as_the_upside():
    """THE FINDING THE PANEL EXISTS TO PRINT. In odds terms the stack is worth
    x1.19 on the upside and x0.71 on the downside — a trend filter is mostly a
    RISK filter, and quoting only its upside effect halves the story."""
    up_odds = math.exp(PROB_LAW["up"][-1])
    dn_odds = math.exp(PROB_LAW["down"][-1])
    assert up_odds == pytest.approx(1.19, abs=0.02)
    assert dn_odds == pytest.approx(0.71, abs=0.02)
    assert abs(math.log(dn_odds)) > abs(math.log(up_odds))


def test_the_stack_moves_the_odds_and_not_the_clock():
    """P4, and it is why this ships a band. Across the trend states the median
    time to +20% was 53, 54, 54 and 54 sessions."""
    s = 0.025
    assert p_touch(1.20, s, True) > p_touch(1.20, s, False)
    assert p_touch(0.80, s, True) < p_touch(0.80, s, False)
    #  sessions_to takes no stack argument at all, by construction
    assert "stack" not in sessions_to.__code__.co_varnames


def test_a_stacked_name_beats_even_odds_and_an_unstacked_one_does_not():
    c_on = cone(20.0, 0.025, stack=True)
    c_off = cone(20.0, 0.025, stack=False)
    assert c_on["odds"] > 1.0
    assert c_off["odds"] < 1.0


# ============================================== the domain, and saying so ====
def test_sigma_is_clamped_to_the_fitted_range():
    """Outside [0.0117, 0.0623] the laws are extrapolation. Clamping keeps the
    number finite; in_domain is what tells the caller to distrust it."""
    assert sessions_to(1.20, 0.200) == pytest.approx(sessions_to(1.20, SIGMA_MAX))
    assert p_touch(1.20, 0.0001) == pytest.approx(p_touch(1.20, SIGMA_MIN))
    assert not in_domain(0.200)
    assert not in_domain(0.0001)
    assert in_domain(0.025)


def test_a_degenerate_target_or_volatility_returns_nan_not_a_number():
    assert math.isnan(sessions_to(1.0, 0.025))
    assert math.isnan(p_touch(1.0, 0.025))
    assert math.isnan(p_touch(1.2, 0.0))
    assert math.isnan(p_touch(1.2, float("nan")))


def test_the_horizon_is_a_year_and_the_band_can_exceed_it():
    """A target far enough away is not reached inside the sample's horizon, so
    the law will happily return more than 252. That is extrapolation and the
    report says so; the constant is pinned here so the claim stays checkable."""
    assert HORIZON == 252
    assert sessions_to(2.00, SIGMA_MIN, "q3") > 0


def test_a_session_is_not_a_calendar_day():
    """The panel turns a session count into a DATE, which is the only place
    calendar arithmetic enters. 252 sessions a year is ~1.45 days each."""
    assert DAYS_PER_SESSION == pytest.approx(1.4494, abs=0.001)
    assert 252 * DAYS_PER_SESSION == pytest.approx(365.25, abs=0.01)


# ================================================ the decile, for context ====
def test_the_volatility_deciles_are_ordered_and_span_the_domain():
    assert list(VOL_DECILES) == sorted(VOL_DECILES)
    assert SIGMA_MIN < VOL_DECILES[0] and VOL_DECILES[-1] < SIGMA_MAX


def test_vol_decile_reads_one_for_calm_and_ten_for_wild():
    assert vol_decile(0.010) == 1
    assert vol_decile(0.0256) == 5
    assert vol_decile(0.100) == 10


# ================================================== the race, H35 ============
def test_a_nearer_target_is_likelier_to_arrive_first():
    assert p_target_first(0.10, 0.20, 0.025) > p_target_first(0.20, 0.20, 0.025)
    assert p_target_first(0.20, 0.10, 0.025) < p_target_first(0.20, 0.20, 0.025)


def test_a_symmetric_bracket_is_close_to_a_coin_flip_and_slightly_against_you():
    """L4. The median short-horizon IDX return is negative, so at equal
    distances the stop arrives first slightly more often. Symmetry is not
    neutrality on a series whose median drifts down."""
    p = p_target_first(0.10, 0.10, 0.026)
    assert 0.45 < p < 0.55
    assert p < 0.52


def test_which_barrier_arrives_first_barely_depends_on_volatility():
    """THE FINDING IN THE RACE LAW. Volatility speeds both barriers up equally,
    so it moves the race almost not at all — the fitted coefficient is -0.0225
    against distance coefficients near 0.8."""
    calm = p_target_first(0.20, 0.10, SIGMA_MIN)
    wild = p_target_first(0.20, 0.10, SIGMA_MAX)
    assert abs(calm - wild) < 0.02
    assert abs(RACE_LAW[3]) < 0.1 * min(abs(RACE_LAW[1]), abs(RACE_LAW[2]))


def test_the_race_is_nearly_a_function_of_the_distance_ratio_alone():
    """The two distance coefficients are near mirror images, so doubling both
    barriers should barely move the answer."""
    a = p_target_first(0.10, 0.10, 0.025)
    b = p_target_first(0.20, 0.20, 0.025)
    assert abs(a - b) < 0.05


def test_a_degenerate_bracket_returns_nan():
    assert math.isnan(p_target_first(0.0, 0.10, 0.025))
    assert math.isnan(p_target_first(0.10, 0.0, 0.025))
    assert math.isnan(p_target_first(0.10, 0.10, 0.0))


# ====================================== the two copies must not drift apart ==
def _pine() -> str:
    with open(PINE, encoding="utf-8") as fh:
        return fh.read()


def test_every_coefficient_appears_verbatim_in_the_pine_file():
    """THE ONLY THING WORSE THAN AN UNVALIDATED CONSTANT IS TWO COPIES OF IT
    THAT STOP MATCHING. Pine cannot import Python, so the numbers live twice
    and nothing but this test connects them."""
    src = _pine()
    missing = [f"{v:.4f}" for law in (TIME_LAW, PROB_LAW)
               for row in law.values() for v in row
               if f"{abs(v):.4f}" not in src]
    missing += [f"{v:.4f}" for v in RACE_LAW if f"{abs(v):.4f}" not in src]
    assert not missing, f"absent from IDX_Suite.pine: {missing}"


def test_the_pine_file_declares_the_same_clamp():
    src = _pine()
    assert f"{SIGMA_MIN:.4f}" in src and f"{SIGMA_MAX:.4f}" in src


def test_the_pine_file_states_the_flip_rules_do_not_work():
    """The labels were asked for and are drawn. The measurement that says not
    to trade them travels with them or the file is misleading by omission."""
    src = _pine().lower()
    assert "-0.0191" in src, "the dual Hull cross's measured mean log"
    assert "none compound" in src or "compounds" in src


def _pine_code() -> str:
    """The file with its comments removed.

    The withdrawn H30 constants SHOULD still appear in the prose — A19's rule
    is that a retracted claim is kept in place and marked, never deleted, so a
    reader does not meet it again somewhere else. What must not survive is the
    withdrawn number still being COMPUTED with.
    """
    out = []
    for ln in _pine().splitlines():
        i = ln.find("//")
        out.append(ln if i < 0 else ln[:i])
    return "\n".join(out)


def test_the_pine_file_carries_the_corrected_H30_constants_not_the_withdrawn():
    """H30 published +0.0218 against a base of +0.0021 and both were artefacts
    of a pivot-indexed forward return. The withdrawn pair must not reappear in
    executable code — only in the paragraph that retracts it."""
    code, prose = _pine_code(), _pine()
    assert "0.0127" in code and "-0.0140" in code
    assert "0.0218" not in code and "0.0021" not in code
    assert "0.0218" in prose, "the retraction itself must stay in the file"


def test_the_pine_file_has_no_comma_chained_statements():
    """Pine has no comma operator. This file has made that mistake before, and
    it is invisible until the editor rejects it."""
    #  The pattern is a STATEMENT call followed by another on the same line.
    #  `line.new(x, color.new(c, 55), line.style_dotted)` is an argument list
    #  and is fine, so the match has to be anchored at the start of the line.
    call = r"(box|line|label|table|plot|alert)"
    bad = [ln for ln in _pine().splitlines()
           if re.match(rf"\s*{call}\.\w+\(.*\)\s*,\s*{call}\.", ln)]
    assert not bad, bad


def test_no_continuation_line_is_indented_on_a_multiple_of_four():
    """PINE RESERVES 4-SPACE INDENTS FOR LOCAL BLOCKS, so a wrapped expression
    indented 20 spaces is parsed as a new statement and the file will not
    compile. Ten lines in this file were wrong, all of them inside the
    projection block that was added last. Nothing in Python catches it and the
    author cannot compile Pine from here, so it is caught structurally."""
    src = _pine().splitlines()
    depth, prev, bad = 0, "", []
    for i, ln in enumerate(src, 1):
        code = ln.split("//")[0].rstrip()
        if not code.strip():
            continue
        if depth > 0 or prev.endswith((",", "+", "-", "*", "/", "?", ":", "(",
                                       "=")):
            ind = len(code) - len(code.lstrip())
            if ind and ind % 4 == 0:
                bad.append(f"L{i}: {ln.strip()[:60]}")
        depth = max(0, depth + code.count("(") - code.count(")")
                    + code.count("[") - code.count("]"))
        prev = code
    assert not bad, bad


def test_the_pine_file_states_that_no_bracket_survived():
    """H35 scored 30 (tp, sl) pairs and none was positive in both halves. A
    chart that prints target and stop levels without that sentence is selling
    a bracket the measurement rejects."""
    src = _pine()
    assert "0 of 30 beat hold in both halves" in src


def test_the_pine_file_says_fibonacci_measured_nothing():
    """The ratios were asked for and are drawn. Drawing them silently would be
    the misleading option; the z-scores travel with them."""
    src = _pine()
    assert "measured nothing" in src
    assert "+0.77" in src and "0.3611" in src


def test_the_pine_file_uses_the_sample_stdev_form():
    """ta.stdev defaults to the POPULATION form; the laws are fitted in the
    sample form, and the difference silently shifts the whole date band."""
    assert "ta.stdev(ret1, 60, false)" in _pine()
