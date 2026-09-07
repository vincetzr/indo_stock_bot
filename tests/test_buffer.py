"""H62 — the buffer sweep, and the decision rule fixed before it ran.

THE TEST THIS FILE EXISTS FOR is `test_the_decision_rule_is_registered_before_
the_numbers`. A sweep that chooses its own winner after seeing the grid is an
argmax whatever it is called, and this repo has been burned by argmaxes three
times (A11's O1, A21's 8-of-10 cell, and H54's own tight buffer). So the rule —
*if B1 and B3 both hold, the constant moves to the MIDDLE of the family, and it
does not move to the argmax under any outcome* — is written in the module
docstring, and this asserts it is still there.

The second is `test_the_middle_is_defined_on_the_grid_not_on_the_results`.
Picking the middle by outcome is an argmax wearing a different word; the middle
is the median of each grid AXIS.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import buffer as bf                                               # noqa: E402
import rules                                                      # noqa: E402

SRC = os.path.join(os.path.dirname(__file__), os.pardir, "scripts", "buffer.py")


# ------------------------------------------------------- THE DECISION RULE

def test_the_decision_rule_is_registered_before_the_numbers():
    doc = bf.__doc__ or ""
    assert "REGISTERED" in doc
    assert "does not move to a new argmax under any outcome" in doc
    for k in ("B1", "B2", "B3"):
        assert k in doc


def test_the_middle_is_defined_on_the_grid_not_on_the_results():
    """Picking the middle by outcome is an argmax wearing a different word."""
    src = open(SRC).read()
    assert "np.median(HI_GRID)" in src and "np.median(VOL_GRID)" in src
    assert "not the median of the RESULTS" in src


def test_the_shipped_constant_is_the_grid_middle():
    """An even-length axis has no median that is also a grid point, so the
    middle is the nearest grid point to the axis median. `middle_cell()` is the
    ONE definition, used by the script and by this test, so the shipped
    constant and the rule that chose it cannot drift apart."""
    assert (rules.KEEP_HI, rules.KEEP_VOL) == bf.middle_cell()


def test_the_middle_is_a_real_grid_point_and_ties_break_deterministically():
    hi, vol = bf.middle_cell()
    assert hi in bf.HI_GRID and vol in bf.VOL_GRID
    assert bf.middle_cell() == bf.middle_cell()


def test_the_grid_spans_from_no_buffer_to_a_wide_one():
    """`keep == entry` IS no buffer, and it has to be in the grid or the sweep
    cannot say whether a buffer helps at all."""
    assert bf.ENTRY_HI in bf.HI_GRID
    assert bf.ENTRY_VOL in bf.VOL_GRID
    assert min(bf.HI_GRID) < bf.ENTRY_HI
    assert max(bf.VOL_GRID) > bf.ENTRY_VOL


def test_the_entry_line_is_held_fixed():
    """Moving the entry line too would make this a sweep of the screen, and the
    buffer effect would not be separable from it."""
    assert (bf.ENTRY_HI, bf.ENTRY_VOL) == (rules.ENTRY_HI, rules.ENTRY_VOL)
    src = open(SRC).read()
    assert "only the KEEP band moves" in src


# ------------------------------------------------------------- THE HARNESS

def test_the_sweep_varies_the_rebalance_phase():
    """A39 records three strategies passing this harness at offset 0 and at no
    other phase. A buffer chosen at one phase is chosen on one draw."""
    src = open(SRC).read()
    assert "offset=off" in src
    assert bf.PHASES >= 4


def test_each_walk_gets_a_fresh_selector():
    """`Sticky` carries a basket across calls; a walk inheriting the previous
    one's holdings measures a different rule from the one it names."""
    src = open(SRC).read()
    i = src.index("for off in offsets:")
    assert "sel.reset()" in src[i:i + 200]


def test_the_excess_is_over_each_walks_own_index_window():
    """A19's error class: comparing quantities measured over different windows.
    Each cell's window differs, so the index must be priced over that window."""
    src = open(SRC).read()
    assert 'B.index_cagr(r["start"], r["end"])' in src


def test_a_walk_that_fails_is_skipped_not_scored_as_zero():
    src = open(SRC).read()
    assert "if not r:" in src and "continue" in src
    assert "np.isfinite(idx)" in src


# ---------------------------------------------------- THE RECORDED OUTCOME

def test_the_superseded_constant_survives_as_a_retraction():
    """A19: deleting a refuted claim leaves a reader unable to see what was
    corrected."""
    src = open(rules.__file__).read()
    assert "0.80, 0.60" in src
    assert "WAS `KEEP_HI" in src


def test_the_improvement_is_explicitly_not_claimed():
    """B1 says the phase spread swamps it, so the +1.92pp that came with the
    move is a side effect and not the reason for it."""
    src = open(rules.__file__).read()
    assert "SIDE" in src and "not claimed" in src


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), os.pardir,
                                    "reports", "buffer_sweep.csv")),
    reason="sweep not run in this checkout")
def test_the_recorded_sweep_supports_the_move():
    """The claim in `rules.py` must match the file the run wrote."""
    p = os.path.join(os.path.dirname(__file__), os.pardir, "reports",
                     "buffer_sweep.csv")
    S = pd.read_csv(p)
    G = S.groupby(["hi", "vol"])["excess"].median()
    across = float(G.max() - G.min())
    within = float(S.groupby(["hi", "vol"])["excess"].apply(
        lambda x: x.max() - x.min()).median())
    assert within > across, (within, across)
    win = S.loc[S.groupby("offset")["excess"].idxmax()]
    assert win.groupby(["hi", "vol"]).ngroups > 1
