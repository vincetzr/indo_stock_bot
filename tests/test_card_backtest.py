"""scripts/card_backtest.py and the exit-cost switch it added to stoptest.walk.

The backtest exists to show the card year by year beside the index. Three
things would make it quietly wrong, and each has a test here:

  * a window that differs between the card and the index (A19's error class);
  * a cost correction that does not charge what it says it charges;
  * a tick ladder that does not match the point-in-time schedule it claims.
"""

from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import card_backtest as cb                                         # noqa: E402
import rules                                                       # noqa: E402
import stoptest as st                                              # noqa: E402

from idxbot.spine import reference as ref                          # noqa: E402


# ----------------------------------------------------------- the levels --
def test_the_card_arm_is_the_shipped_card_not_a_copy():
    assert cb.ARMS["card"] == {"stop": rules.STOP, "tp": rules.TP,
                               "tp_frac": rules.TP_FRAC}
    assert cb.ARMS["no_levels"] == {}


def test_walk_still_defaults_to_the_accounting_the_card_quotes():
    """The correction is opt-in until adopted; the shipped file must not move."""
    sig = inspect.signature(st.walk)
    assert sig.parameters["exit_cost_mult"].default == 1.0
    assert cb.VARIANTS["as_printed"] == {"pit_ticks": False,
                                         "exit_cost_mult": 1.0}
    assert cb.VARIANTS["corrected"] == {"pit_ticks": True,
                                        "exit_cost_mult": 0.5}


# ---------------------------------------------------- the cost accounting --
def _panel(n_dates: int = 300, c: float = 0.01) -> pd.DataFrame:
    """Flat prices. A0..A9 bought at the first review and failed out of the
    keep band at the second, B0..B9 bought there and kept to the end."""
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    names = [f"A{i}" for i in range(10)] + [f"B{i}" for i in range(10)] + \
        [f"Z{i:02d}" for i in range(40)]
    rows = []
    for j, d in enumerate(dates):
        early = j < st.FREQ
        for t in names:
            a, b = t.startswith("A"), t.startswith("B")
            rows.append({
                "date": d, "ticker": t, "adj_close": 100.0, "close": 100.0,
                "elig": True, "cost": c,
                "in_entry": (a and early) or (b and not early),
                "in_keep": (a and early) or (b and not early),
                "hi52": 1.0 if (a or b) else 0.5,
            })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("m", [1.0, 0.5])
def test_exit_cost_mult_charges_exactly_what_it_says(m):
    """One full turnover on flat prices.

    The first basket is the curve's base (its entry cost is already paid when
    the curve starts), so what remains is the SALE of ten names and the PURCHASE
    of ten more: (1 - m*c) * (1 - c/2). With m = 0.5 a round trip is c, which
    is what `cost` means; with m = 1.0 it is 1.5c.
    """
    c = 0.01
    r = st.walk(_panel(c=c), exit_cost_mult=m)
    eqs = r["curve"][1]
    assert eqs[-1] == pytest.approx((1 - m * c) * (1 - 0.5 * c), abs=1e-12)
    assert r["n_band"] == 10


def test_no_sales_means_no_difference():
    P = _panel()
    P["in_keep"] = P["in_keep"] | P["ticker"].str.startswith("A")
    a = st.walk(P, exit_cost_mult=1.0)["curve"][1]
    b = st.walk(P, exit_cost_mult=0.5)["curve"][1]
    np.testing.assert_allclose(a, b)


# -------------------------------------------------------- the tick ladder --
def test_pit_tick_is_reference_tick_size_on_every_regime():
    prices = [50, 199, 200, 201, 499, 500, 501, 1999, 2000, 2001, 4999,
              5000, 5001, 12000]
    days = ["2005-03-01", "2010-06-01", "2014-01-03", "2014-01-06",
            "2015-06-01", "2016-04-29", "2016-05-02", "2020-01-02",
            "2026-11-13"]
    for d in days:
        got = cb.pit_tick(np.array(prices, float), [d] * len(prices))
        want = [ref.tick_size(p, d) for p in prices]
        assert list(got) == want, d


def test_pit_tick_is_coarser_than_todays_ladder_before_2014():
    """The direction of the lookahead the correction removes."""
    p = np.array([300.0, 1000.0, 3000.0, 6000.0])
    old = cb.pit_tick(p, ["2010-06-01"] * 4)
    now = np.array([st.tick_of(x) for x in p])
    assert (old > now).all()


def test_with_costs_leaves_the_printed_panel_alone():
    P = pd.DataFrame({"date": pd.to_datetime(["2010-06-01", "2020-06-01"]),
                      "close": [3000.0, 3000.0], "cost": [0.1, 0.1]})
    assert cb.with_costs(P, False) is P
    Q = cb.with_costs(P, True)
    assert Q["cost"].tolist() == pytest.approx(
        [st.FEE + st.SPREAD_MULT * 25 / 3000, st.FEE + st.SPREAD_MULT * 10 /
         3000])
    assert P["cost"].tolist() == [0.1, 0.1]


# ------------------------------------------------------------ the windows --
def test_common_window_is_latest_start_and_earliest_end():
    d1 = np.array(pd.bdate_range("2006-01-02", "2026-11-13"))
    d2 = np.array(pd.bdate_range("2007-03-01", "2026-11-12"))
    a, b = cb.common_window([(d1, np.ones(len(d1))),
                             (d2, np.ones(len(d2)))])
    assert a == pd.Timestamp("2007-03-01")
    assert b == pd.Timestamp("2026-11-12")


def test_year_returns_chain_to_the_whole_window():
    idx = pd.bdate_range("2008-03-25", "2011-07-15")
    rng = np.random.default_rng(0)
    s = pd.Series(np.cumprod(1 + rng.normal(0, 0.01, len(idx))), index=idx)
    y = cb.year_returns(s)
    assert list(y.index) == [2008, 2009, 2010, 2011]
    assert np.prod(1 + y.to_numpy()) == pytest.approx(s.iloc[-1] / s.iloc[0])


def test_index_is_read_on_the_cards_dates(tmp_path):
    """A session the index lacks takes the last close before it, so the two
    series can never drift onto different windows."""
    p = tmp_path / "jkse.csv"
    pd.DataFrame({"date": ["2020-01-02", "2020-01-03", "2020-01-07"],
                  "close": [100.0, 110.0, 121.0]}).to_csv(p, index=False)
    on = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06",
                           "2020-01-07"])
    s = cb.index_tr(on[0], on[-1], on, path=str(p), yld=0.0)
    assert s.index.equals(on)
    assert s.tolist() == pytest.approx([1.0, 1.1, 1.1, 1.21])


def test_index_total_return_accrues_the_yield_in_calendar_time(tmp_path):
    p = tmp_path / "jkse.csv"
    pd.DataFrame({"date": ["2020-01-01", "2021-01-01"],
                  "close": [100.0, 100.0]}).to_csv(p, index=False)
    on = pd.DatetimeIndex(["2020-01-01", "2021-01-01"])
    s = cb.index_tr(on[0], on[-1], on, path=str(p), yld=0.0177)
    assert s.iloc[-1] == pytest.approx(1.0177 ** (366 / 365.25))


def test_halves_split_at_the_date_midpoint():
    idx = pd.date_range("2010-01-01", "2020-01-01", freq="D")
    mid = idx[0] + (idx[-1] - idx[0]) / 2
    v = np.where(idx <= mid, 1.0,
                 1.0 + (idx - mid).days.to_numpy(float) / 1826.0)
    s = pd.Series(v, index=idx)
    early, late = cb.halves(s)
    assert early == pytest.approx(0.0)
    assert (1 + late) ** 5 == pytest.approx(2.0, rel=1e-3)
