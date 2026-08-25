"""Tests for H20's portfolio accounting.

The study exists because H17 and H18 optimised the cohort MEDIAN, which an
equal-weighted holder never receives, and because twelve month-offset slots
were about to be read as twelve trials. Both mistakes are pinned here:
`test_basket_return_is_the_mean_not_the_median` and
`test_paired_pairs_by_slot_and_cancels_the_common_component`.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))

from portfolio_sim import baskets, paired, slots                 # noqa: E402


def _named(n_cohorts=150, n_names=20, seed=3):
    """A per-name table shaped like the real one."""
    rng = np.random.default_rng(seed)
    d = pd.date_range("2010-01-01", periods=n_cohorts, freq="MS")
    rows = []
    for day in d:
        for j in range(n_names):
            rows.append({
                "as_of": day, "ticker": f"T{j:02d}", "picked": j < 12,
                "cost": 0.01,
                "A": rng.normal(0.05, 0.3), "A|held": 252.0,
                "B": rng.normal(0.05, 0.3), "B|held": 126.0})
    return pd.DataFrame(rows)


# ==========================================================================
# what you actually earn
# ==========================================================================
def test_basket_return_is_the_mean_not_the_median():
    """THE DEFECT THIS FILE EXISTS FOR. Hold twelve names equal-weighted and
    you receive their mean; H17 and H18 both optimised the median."""
    D = _named(n_cohorts=1, n_names=12)
    D.loc[:, "A"] = [0.0] * 11 + [11.0]        # one huge winner
    B = baskets(D, "A", "picked")
    assert B["ret"].iloc[0] == pytest.approx(11.0 / 12)
    assert B["med"].iloc[0] == pytest.approx(0.0)
    assert B["ret"].iloc[0] != pytest.approx(B["med"].iloc[0])


def test_picked_mode_selects_exactly_the_flagged_names():
    D = _named()
    B = baskets(D, "A", "picked")
    assert (B["n"] == 12).all()


def test_random_mode_draws_the_requested_size_and_is_reproducible():
    D = _named()
    a = baskets(D, "A", "random", size=12, seed=7)
    b = baskets(D, "A", "random", size=12, seed=7)
    c = baskets(D, "A", "random", size=12, seed=8)
    assert (a["n"] == 12).all()
    assert np.allclose(a["ret"], b["ret"])
    assert not np.allclose(a["ret"], c["ret"])


def test_random_mode_can_draw_names_the_entry_did_not_pick():
    """The control is worthless if it only ever redraws the same basket."""
    D = _named()
    picked_only = baskets(D[D["picked"]], "A", "random", seed=1)
    whole_pool = baskets(D, "A", "random", seed=1)
    assert not np.allclose(picked_only["ret"], whole_pool["ret"])


def test_a_cohort_with_too_few_priced_names_is_dropped():
    D = _named(n_cohorts=2, n_names=12)
    D.loc[D["as_of"] == D["as_of"].min(), "A"] = np.nan
    assert len(baskets(D, "A", "picked")) == 1


# ==========================================================================
# the slot simulator
# ==========================================================================
def _B(rets, held, start="2010-01-01"):
    d = pd.date_range(start, periods=len(rets), freq="MS")
    return pd.DataFrame({"as_of": d, "n": 12, "ret": rets, "med": rets,
                         "logret": np.log1p(rets), "p2": 0.0, "pdn": 0.0,
                         "held": held})


def test_a_full_horizon_hold_trades_about_once_a_year():
    B = _B(np.full(120, 0.0), np.full(120, 252.0))
    S = slots(B, n_slots=1)
    assert S["trades"].iloc[0] == pytest.approx(10, abs=1)


def test_an_early_exit_redeploys_and_therefore_trades_more():
    """The whole point of simulating slots: freed capital comes back."""
    long_ = slots(_B(np.zeros(120), np.full(120, 252.0)), n_slots=1)
    short = slots(_B(np.zeros(120), np.full(120, 63.0)), n_slots=1)
    assert short["trades"].iloc[0] > long_["trades"].iloc[0]


def test_a_slot_never_re_enters_before_its_position_is_free():
    B = _B(np.zeros(120), np.full(120, 252.0))
    S = slots(B, n_slots=1)
    #  a 252-session hold is ~366 days, so entries are at least a year apart
    assert S["trades"].iloc[0] <= 11        # 120 months / ~12


def test_terminal_wealth_compounds_the_realised_returns():
    B = _B(np.full(120, 0.10), np.full(120, 252.0))
    S = slots(B, n_slots=1)
    assert S["terminal"].iloc[0] == pytest.approx(
        1.10 ** S["trades"].iloc[0])


def test_max_drawdown_is_measured_on_the_equity_path_not_per_trade():
    r = np.concatenate([[0.5, -0.4, -0.4], np.full(20, 0.05)])
    B = _B(r, np.full(len(r), 21.0))
    S = slots(B, n_slots=1)
    #  two consecutive -40% trades compound to -64%, worse than either alone
    assert S["maxdd"].iloc[0] < -0.5


def test_a_slot_with_too_few_trades_is_dropped_rather_than_reported():
    assert slots(_B(np.zeros(3), np.full(3, 252.0)), n_slots=1).empty


# ==========================================================================
# the paired comparison
# ==========================================================================
def test_paired_pairs_by_slot_and_cancels_the_common_component():
    """THE SECOND DEFECT. Slot 3 under two rules trades the same dates, so
    most of the cross-slot spread is shared and must cancel in the difference.
    Comparing the two 10-90% bands instead discards exactly that."""
    D = _named(n_cohorts=150, n_names=20, seed=11)
    D["B"] = D["A"] + 0.02                     # a constant per-name edge
    D["B|held"] = D["A|held"]
    q = paired(D, "B", "A")
    assert q["mean"] > 0
    assert q["wins"] == q["n"]                 # every slot, not just on average
    #  the paired sd must be far below the raw spread of either rule's CAGRs
    raw = slots(baskets(D, "A", "picked"))["cagr"].std(ddof=1)
    assert q["sd"] < raw


def test_paired_reports_no_difference_when_the_rules_are_identical():
    D = _named()
    D["B"] = D["A"]
    D["B|held"] = D["A|held"]
    q = paired(D, "B", "A")
    assert q["mean"] == pytest.approx(0.0, abs=1e-12)
    assert q["wins"] == 0


def test_paired_returns_empty_when_a_rule_cannot_be_scored():
    D = _named()
    D["B"] = np.nan
    assert paired(D, "B", "A") == {}


def test_paired_sign_is_the_direction_of_the_first_rule():
    D = _named(n_cohorts=150, seed=5)
    D["B"] = D["A"] - 0.05
    D["B|held"] = D["A|held"]
    assert paired(D, "B", "A")["mean"] < 0
    assert paired(D, "A", "B")["mean"] > 0
