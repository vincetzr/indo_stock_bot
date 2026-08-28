"""Tests for H43 — rebalance frequency for the strength+calm screen.

The study compares a screened portfolio to a random one at six holding periods,
so the tests are aimed at the ways that comparison can be quietly unfair: a
control that churns more than the treatment, a cost charged on frequency rather
than turnover, a delisted name carried forever, a half-split that never ran,
and a price lookup that forward-fills a suspension into a real bar.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                "scripts"))

from rebalance import Prices, half_cagr, run, screen        # noqa: E402


def _panel(n=800, names=40, seed=2):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=n)
    out = []
    for i in range(names):
        p = 1000.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
        hi = pd.Series(p) / pd.Series(p).rolling(252, min_periods=60).max()
        vol = pd.Series(p).pct_change().rolling(60, min_periods=60).std(ddof=1)
        out.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i:02d}", "adj_close": p,
            "hi52": hi.to_numpy(), "vol60": vol.to_numpy(),
            "elig": True}))
    return pd.concat(out, ignore_index=True).sort_values(["ticker", "date"])


# ============================================== the screen is what it claims ==
def test_the_screen_selects_strong_and_calm_names_only():
    P = _panel()
    d = P[P["date"] == P["date"].max()]
    got = screen(d, 10)
    if got:
        s = d[d["ticker"].isin(got)]
        assert (s["hi52"] >= d["hi52"].quantile(0.90)).all()
        assert (s["vol60"] <= d["vol60"].quantile(0.50)).all()


def test_the_screen_holds_the_whole_cell_not_a_top_n_slice():
    """A first version sorted by hi52 and took the top ten, which is a SECOND
    selection stacked on the screen — and H26's own frontier says a stronger
    strength cut scores WORSE (skew 1.95 against 2.15). A15's lesson: an
    arbitrary cut imposed on a cell is a choice nobody registered."""
    P = _panel(names=200, seed=5)
    d = P[P["date"] == P["date"].max()]
    got = screen(d, 10)
    cell = d[(d["hi52"] >= d["hi52"].quantile(0.90))
             & (d["vol60"] <= d["vol60"].quantile(0.50))]
    assert len(got) == len(cell)


def test_the_random_control_is_size_matched_to_the_cell():
    """A random basket of a different width is a different portfolio: breadth
    alone changes variance and therefore compounding."""
    P = _panel(names=200, seed=6)
    d = P[P["date"] == P["date"].max()]
    real = screen(d, 10)
    rnd = screen(d, 10, rng=np.random.default_rng(0))
    assert len(rnd) == len(real)


def test_the_random_control_ignores_the_screen():
    """If the control still filtered on hi52 and vol60 the comparison would be
    the rule against itself."""
    P = _panel(names=200, seed=7)
    d = P[P["date"] == P["date"].max()]
    a = set(screen(d, 10, rng=np.random.default_rng(1)))
    b = set(screen(d, 10, rng=np.random.default_rng(2)))
    assert a != b, "two draws must differ or it is not random"


def test_the_screen_respects_eligibility():
    P = _panel(names=60, seed=8)
    P.loc[P["ticker"] > "T05", "elig"] = False
    d = P[P["date"] == P["date"].max()]
    assert screen(d, 10) == []          # too few eligible names to rank


# ==================================================== prices and delistings ===
def test_a_price_lookup_never_invents_a_bar_the_name_did_not_trade():
    """A11's pivot defect with the sign reversed: forward-filling a suspension
    turns a halt into a real print at yesterday's price."""
    P = _panel(names=2)
    #  the halted date has to be captured BEFORE the row is dropped; taking
    #  iloc[50] afterwards points at a different bar and the test passes for
    #  the wrong reason
    halt = P["date"].iloc[50]
    P = P[~((P["ticker"] == "T00") & (P["date"] == halt))]
    PX = Prices(P)
    assert np.isnan(PX.at("T00", halt))
    assert np.isfinite(PX.at("T01", halt))    # the other name still trades


def test_a_delisted_name_is_realised_at_its_last_print_and_flagged():
    """H41's bug: a position whose ticker vanishes must not be carried at its
    last price for the rest of the backtest."""
    P = _panel(names=2)
    cut = P["date"].iloc[400]
    P = P[~((P["ticker"] == "T00") & (P["date"] > cut))]
    PX = Prices(P)
    px, dead = PX.exit_price("T00", P["date"].iloc[0], P["date"].max())
    assert dead
    assert np.isfinite(px)


def test_a_name_alive_at_the_sell_date_is_not_flagged_dead():
    P = _panel(names=2)
    PX = Prices(P)
    _, dead = PX.exit_price("T00", P["date"].iloc[0], P["date"].max())
    assert not dead


# ============================================================ the cost model ==
def test_cost_is_charged_on_turnover_not_on_frequency():
    """Two names changing out of ten costs a fifth of a round trip, not a whole
    one. Every earlier study here priced the whole book each rebalance."""
    P = _panel(names=60, seed=3)
    dates = np.sort(P["date"].unique())
    PX = Prices(P)
    r = run(P, PX, dates, 21, 0, 10)
    assert r
    assert 0.0 <= r["turnover"] <= 1.0
    #  a sticky screen must cost strictly less than a full rotation would
    assert r["cost_yr"] < 1.0


def test_a_more_frequent_schedule_costs_more_per_year():
    P = _panel(names=60, seed=3)
    dates = np.sort(P["date"].unique())
    PX = Prices(P)
    fast = run(P, PX, dates, 21, 0, 10)
    slow = run(P, PX, dates, 252, 0, 10)
    if fast and slow:
        assert fast["cost_yr"] > slow["cost_yr"]


def test_gross_is_never_worse_than_net():
    """GROSS EXISTS BECAUSE THE CONTROL CHURNS MORE THAN THE TREATMENT. A
    random basket redrawn monthly rotates ~95% of its book against the screen's
    58%, so a net-of-cost difference confounds selection with turnover."""
    P = _panel(names=60, seed=4)
    dates = np.sort(P["date"].unique())
    r = run(P, Prices(P), dates, 21, 0, 10)
    assert r
    assert r["gross_cagr"] >= r["cagr"] - 1e-12


# ============================================================= the half-split =
def test_the_half_split_is_on_the_return_period_not_the_start_date():
    """Every arm starts within a few years of the panel's beginning, so
    splitting on `start` would put all of them in the early bucket and silently
    report a replication test that never ran."""
    d = pd.bdate_range("2010-01-01", periods=400).to_numpy()
    eq = np.exp(np.concatenate([np.linspace(0, 0.6, 200),
                                0.6 + np.linspace(0, 0.05, 200)]))
    e, l = half_cagr(list(zip(d, eq)))
    assert e > l


def test_a_flat_path_has_no_growth_in_either_half():
    d = pd.bdate_range("2010-01-01", periods=200).to_numpy()
    e, l = half_cagr(list(zip(d, np.ones(200))))
    assert abs(e) < 1e-6 and abs(l) < 1e-6


def test_too_short_a_path_returns_nan_rather_than_a_number():
    assert all(np.isnan(v) for v in half_cagr([]))


# ============================================================ the whole path ==
def test_a_run_is_reproducible_from_its_seed():
    P = _panel(names=60, seed=9)
    dates = np.sort(P["date"].unique())
    PX = Prices(P)
    a = run(P, PX, dates, 63, 0, 10, seed=11)
    b = run(P, PX, dates, 63, 0, 10, seed=11)
    assert a["cagr"] == pytest.approx(b["cagr"])


def test_two_control_seeds_give_different_paths():
    P = _panel(names=60, seed=9)
    dates = np.sort(P["date"].unique())
    PX = Prices(P)
    a = run(P, PX, dates, 63, 0, 10, seed=11)
    b = run(P, PX, dates, 63, 0, 10, seed=12)
    assert a["cagr"] != b["cagr"]


def test_a_schedule_with_too_few_marks_returns_nothing_rather_than_raising():
    P = _panel(n=120, names=20)
    dates = np.sort(P["date"].unique())
    assert run(P, Prices(P), dates, 756, 0, 10) == {}


def test_the_reported_years_match_the_span_actually_walked():
    P = _panel(names=40, seed=10)
    dates = np.sort(P["date"].unique())
    r = run(P, Prices(P), dates, 63, 0, 10)
    assert r
    span = (r["end"] - r["start"]).astype("timedelta64[D]").astype(float) / 365.25
    assert r["years"] == pytest.approx(span, rel=1e-9)
