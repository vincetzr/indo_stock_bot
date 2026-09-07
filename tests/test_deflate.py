"""H58 — the deflated Sharpe applied to this repo's own arms.

THE TEST THIS FILE EXISTS FOR is `test_the_lower_bound_claim_is_in_the_module`.
The whole reading of H58 turns on one sentence: the control-draw estimate of
`sr_variance` is a LOWER bound, so every DSR built on it is an UPPER bound on
believability. A future edit that quietly promotes it to "the" estimate would
turn a hedged result into a claimed one with no test failing and no number
changing. A19 records exactly that shape — a refuted sentence surviving above
the corrected code — so both the retraction and the correction are pinned here.

The second concern is the harness itself: `curve_returns` turns a bhbench
equity path into per-period returns, and getting that off by one bar would
change every Sharpe in the study while looking entirely plausible.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import deflate                                                    # noqa: E402
from deflate import arms, curve_returns                           # noqa: E402
from idxbot.metrics import expected_max_sharpe, psr, sharpe       # noqa: E402

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, os.pardir, "scripts", "deflate.py")


# --------------------------------------------------------------- the harness

def test_curve_returns_starts_from_par_not_from_the_first_mark():
    """The first rebalance's return is measured from the initial capital.

    Dropping it loses the first period entirely; measuring it from the first
    mark's equity makes it identically zero. Both are plausible-looking and
    both are wrong.
    """
    curve = [("d1", 1.10), ("d2", 1.21), ("d3", 1.331)]
    r = curve_returns(curve)
    assert len(r) == 3
    assert r == pytest.approx([0.10, 0.10, 0.10])


def test_curve_returns_is_empty_on_a_degenerate_path():
    assert len(curve_returns([])) == 0
    assert len(curve_returns([("d", 1.0)])) == 0


def test_curve_returns_round_trips_to_the_final_equity():
    """A compounding identity: the product of the returns must be the path."""
    rng = np.random.default_rng(0)
    eq, curve = 1.0, []
    for i in range(30):
        eq *= 1.0 + rng.normal(0.02, 0.08)
        curve.append((i, eq))
    assert float(np.prod(1.0 + curve_returns(curve))) == pytest.approx(eq)


def test_arms_are_distinct_and_carry_a_frequency():
    got = arms()
    assert len(got) >= 5
    labels = [x[0] for x in got]
    assert len(set(labels)) == len(labels), "duplicate arm labels"
    for _label, sel, freq in got:
        assert callable(sel)
        assert freq in (63, 126, 252)


def test_stateful_selectors_are_reset_between_walks():
    """`Sticky` holds a basket across calls. A walk that inherits the previous
    walk's holdings measures a different rule from the one it names — and the
    control draws would inherit the TREATMENT's book, which is worse."""
    src = open(SRC).read()
    assert src.count('hasattr(sel, "reset")') >= 2, (
        "the control loop must reset a stateful selector too, or every random "
        "draw starts from the strategy's own holdings")


# ------------------------------------------------- the load-bearing hedge

def test_the_lower_bound_claim_is_in_the_module():
    """The reading of H58 rests on this and nothing else guards it."""
    doc = deflate.__doc__ or ""
    assert "LOWER BOUND" in doc
    assert "UPPER BOUND" in doc.upper()


def test_the_refuted_sentence_is_kept_and_marked_not_deleted():
    """A19: fixing the code is not fixing the claim. The original wording is
    retained under a retraction so a reader meets the correction, not the
    refuted version, and not a silent absence."""
    doc = deflate.__doc__ or ""
    assert "THAT IS WRONG" in doc
    assert "exactly the quantity DSR" in doc, (
        "the retracted sentence was deleted rather than marked — a reader of "
        "the git history now has no way to see what was corrected")


def test_the_dispersion_sweep_is_reported_rather_than_one_number():
    src = open(SRC).read()
    assert "dispersion_sweep" in src
    assert "D4" in src


def test_the_registration_survives_the_answer():
    doc = deflate.__doc__ or ""
    for k in ("D1", "D2", "D3"):
        assert k in doc
    assert "PREDICTED NULL" in doc


# --------------------------------------------------- the arithmetic it relies on

def test_a_wider_assumed_dispersion_always_lowers_the_dsr():
    """D4's whole logic: the sweep must be monotone, or reading the lower
    bound as the most flattering row is unjustified."""
    rng = np.random.default_rng(1)
    r = rng.normal(0.04, 0.11, 73)          # ~a quarterly path of this repo
    prev = 1.1
    for srv in (0.005, 0.0127, 0.05, 0.10, 0.25, 1.0):
        d = psr(r, expected_max_sharpe(350, srv, 4.0), 4.0)
        assert d <= prev + 1e-12, srv
        prev = d


def test_the_trial_count_matches_the_hypothesis_log():
    """`TRIALS` is not a guess: hypotheses.md states the running count, and a
    deflation against the wrong count is worse than none."""
    log = open(os.path.join(HERE, os.pardir, "hypotheses.md")).read()
    assert f"Trials after H57: {deflate.TRIALS}" in log or \
        f"Trials after H58: {deflate.TRIALS + 4}" in log, (
        "scripts/deflate.py's TRIALS no longer matches the running count in "
        "hypotheses.md — update both or the deflation is against a stale bar")


def test_a_random_arm_should_not_clear_the_bar_on_synthetic_data():
    """D2 as a unit test: zero-edge returns must not survive the deflation."""
    rng = np.random.default_rng(2)
    r = rng.normal(0.0, 0.11, 73)
    d = psr(r, expected_max_sharpe(350, 0.0072, 4.0), 4.0)
    assert d < 0.95
