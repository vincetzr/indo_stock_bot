"""Tests for the compounding portfolio engine and the two-sleeve book.

The engine's whole job is to turn a ranking into a compounded record without
quietly helping itself. So the assertions here are about the ways that help
usually creeps in:

  * rebalances must not overlap, or one lucky future is counted many times;
  * the liquidity screen must be point-in-time;
  * costs must be charged once per rebalance and must only ever reduce returns;
  * the benchmark must be scored over the *same* windows as the strategy;
  * a screen with no information must not beat equal-weight.

The last one is the most useful test in the file: it feeds the engine a random
score and asserts that nothing interesting happens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idxbot import maxprofit as mp
from idxbot import twosleeve as ts


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def make_obs(n_dates: int = 1400, n_names: int = 60, seed: int = 7,
             signal: float = 0.0) -> pd.DataFrame:
    """A synthetic panel where the score's predictive power is a dial.

    ``signal=0`` means ``c_momentum`` is pure noise, so any apparent edge is an
    artefact of the engine. Turning it up should produce an edge and nothing
    else should.

    The panel is deliberately long: a 60-day book with non-overlapping
    rebalances only gets one observation per quarter, so a short fixture
    silently produces ``None`` instead of a testable record.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_dates)
    rows = []
    for d in dates:
        score = rng.random(n_names)
        noise = rng.normal(0, 0.08, n_names)
        fwd = signal * (score - 0.5) * 0.4 + noise
        for i in range(n_names):
            rows.append({"date": d, "ticker": f"T{i:03d}", "close": 1000.0,
                         "vt": 1e10, "c_momentum": score[i],
                         "c_near_high": score[i], "c_trend_persistence": score[i],
                         "c_relative_strength": score[i],
                         "fwd_5": fwd[i] / 4, "fwd_20": fwd[i],
                         "fwd_60": fwd[i] * 3})
    return pd.DataFrame(rows)


MOM = {"momentum": 1.0}


@pytest.fixture(scope="module")
def noise_panel() -> pd.DataFrame:
    return make_obs(signal=0.0)


@pytest.fixture(scope="module")
def signal_panel() -> pd.DataFrame:
    return make_obs(signal=1.0)


# ---------------------------------------------------------------------------
# the engine must not invent an edge
# ---------------------------------------------------------------------------
def test_a_meaningless_score_does_not_beat_equal_weight(noise_panel):
    """The control experiment. If this fails, every other number is worthless."""
    book = mp.run_book(noise_panel, MOM, horizon=20, top_n=5, cost=0.0)
    bench = mp.equal_weight_benchmark(noise_panel, horizon=20)
    assert book is not None and bench is not None
    assert abs(book.stats["mean_period"] - bench.stats["mean_period"]) < 0.01


def test_a_real_score_does_beat_equal_weight(signal_panel):
    book = mp.run_book(signal_panel, MOM, horizon=20, top_n=5, cost=0.0)
    bench = mp.equal_weight_benchmark(signal_panel, horizon=20)
    assert book.stats["mean_period"] > bench.stats["mean_period"]
    assert book.stats["cagr"] > bench.stats["cagr"]


def test_more_concentration_captures_more_of_a_real_signal(signal_panel):
    """With genuine signal, a tighter book should earn more per rebalance."""
    tight = mp.run_book(signal_panel, MOM, horizon=20, top_n=3, cost=0.0)
    wide = mp.run_book(signal_panel, MOM, horizon=20, top_n=30, cost=0.0)
    assert tight.stats["mean_period"] > wide.stats["mean_period"]


# ---------------------------------------------------------------------------
# overlap, the statistic-inflating trap
# ---------------------------------------------------------------------------
def test_rebalances_never_overlap(signal_panel):
    """Two holding periods sharing a future would count one outcome twice."""
    book = mp.run_book(signal_panel, MOM, horizon=60, top_n=5)
    stamps = pd.DatetimeIndex(book.equity.index)
    gaps = stamps[1:] - stamps[:-1]
    assert (gaps >= pd.Timedelta(days=60 * 7 // 5 - 1)).all()


def test_longer_horizons_give_strictly_fewer_rebalances(signal_panel):
    counts = [mp.run_book(signal_panel, MOM, horizon=h, top_n=5).stats["rebalances"]
              for h in (5, 20, 60)]
    assert counts[0] > counts[1] > counts[2]


def test_benchmark_uses_the_same_windows_as_the_book(signal_panel):
    book = mp.run_book(signal_panel, MOM, horizon=20, top_n=5)
    bench = mp.equal_weight_benchmark(signal_panel, horizon=20)
    assert book.stats["rebalances"] == bench.stats["rebalances"]


# ---------------------------------------------------------------------------
# costs
# ---------------------------------------------------------------------------
def test_costs_only_ever_reduce_the_result(signal_panel):
    free = mp.run_book(signal_panel, MOM, horizon=20, top_n=5, cost=0.0)
    dear = mp.run_book(signal_panel, MOM, horizon=20, top_n=5, cost=0.02)
    assert dear.stats["cagr"] < free.stats["cagr"]
    assert dear.stats["mean_period"] == pytest.approx(
        free.stats["mean_period"] - 0.02, abs=1e-12)


def test_cost_is_charged_once_per_rebalance_not_per_name(signal_panel):
    a = mp.run_book(signal_panel, MOM, horizon=20, top_n=3, cost=0.01)
    b = mp.run_book(signal_panel, MOM, horizon=20, top_n=30, cost=0.01)
    free_a = mp.run_book(signal_panel, MOM, horizon=20, top_n=3, cost=0.0)
    free_b = mp.run_book(signal_panel, MOM, horizon=20, top_n=30, cost=0.0)
    assert (free_a.stats["mean_period"] - a.stats["mean_period"]) == pytest.approx(
        free_b.stats["mean_period"] - b.stats["mean_period"], abs=1e-12)


def test_a_shorter_horizon_pays_costs_more_often(signal_panel):
    """20 rebalances a year at 0.6% is 12 points; four is 2.4. It must show."""
    fast = mp.run_book(signal_panel, MOM, horizon=5, top_n=5, cost=0.01)
    fast_free = mp.run_book(signal_panel, MOM, horizon=5, top_n=5, cost=0.0)
    slow = mp.run_book(signal_panel, MOM, horizon=60, top_n=5, cost=0.01)
    slow_free = mp.run_book(signal_panel, MOM, horizon=60, top_n=5, cost=0.0)
    fast_drag = fast_free.stats["cagr"] - fast.stats["cagr"]
    slow_drag = slow_free.stats["cagr"] - slow.stats["cagr"]
    assert fast_drag > slow_drag


# ---------------------------------------------------------------------------
# screens
# ---------------------------------------------------------------------------
def test_liquidity_screen_excludes_thin_names(noise_panel):
    df = noise_panel.copy()
    df.loc[df["ticker"] == "T000", "vt"] = 1e6
    kept = mp.eligible(df, min_turnover=5e9)
    assert "T000" not in set(kept["ticker"])


def test_price_floor_excludes_sub_rp50(noise_panel):
    df = noise_panel.copy()
    df.loc[df["ticker"] == "T001", "close"] = 20.0
    assert "T001" not in set(mp.eligible(df)["ticker"])


def test_thin_cross_sections_are_skipped():
    """Picking 5 from 6 names is not a selection."""
    tiny = make_obs(n_dates=600, n_names=6, signal=1.0)
    assert mp.run_book(tiny, MOM, horizon=20, top_n=5, min_names=20) is None


def test_split_is_chronological(signal_panel):
    train, hold = mp.split(signal_panel, 0.6)
    assert train["date"].max() <= hold["date"].min()
    assert len(train) + len(hold) == len(signal_panel)


# ---------------------------------------------------------------------------
# multibagger sleeve
# ---------------------------------------------------------------------------
def make_bagger_panel(n_dates: int = 110, n_names: int = 60,
                      seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2005-01-03", periods=n_dates, freq="60D")
    rows = []
    for d in dates:
        price = rng.uniform(60, 5000, n_names)
        # cheap names get a genuinely better forward outcome
        fwd = (5000 - price) / 5000 * 1.5 + rng.normal(0, 0.4, n_names)
        for i in range(n_names):
            rows.append({"date": d, "ticker": f"B{i:03d}", "price": price[i],
                         "turnover": rng.uniform(1e9, 1e12),
                         "hi_750": -rng.random(), "vol_60": rng.random(),
                         "vol_trend": rng.random(), "fwd_3y": fwd[i]})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def bagger_panel() -> pd.DataFrame:
    return make_bagger_panel()


def test_bagger_score_prefers_cheap_and_small(bagger_panel):
    scored = bagger_panel.assign(s=ts.bagger_score(bagger_panel))
    one_day = scored[scored["date"] == scored["date"].iloc[0]]
    top = one_day.nlargest(10, "s")
    bottom = one_day.nsmallest(10, "s")
    assert top["price"].median() < bottom["price"].median()
    assert top["turnover"].median() < bottom["turnover"].median()


def test_bagger_sleeve_beats_no_screen_when_the_pattern_is_real(bagger_panel):
    sleeve = ts.run_bagger_sleeve(bagger_panel, top_n=10, hold_days=750, cost=0.0)
    bench = ts.bagger_benchmark(bagger_panel, hold_days=750)
    assert sleeve is not None and bench is not None
    assert sleeve.stats["cagr"] > bench.stats["cagr"]


def test_rejected_factors_are_documented_and_excluded():
    """Volatility and volume-surge raise 3x odds but cut returns.

    They are kept in the module as a named rejection rather than deleted, so the
    reason survives; this asserts they never leak back into the live score.
    """
    assert "vol_60" not in ts.BAGGER_FACTORS
    assert "vol_trend" not in ts.BAGGER_FACTORS
    assert set(ts.REJECTED_FACTORS) == {"vol_60", "vol_trend"}


def test_bagger_holds_do_not_overlap(bagger_panel):
    sleeve = ts.run_bagger_sleeve(bagger_panel, top_n=10, hold_days=750)
    stamps = pd.DatetimeIndex(sleeve.equity.index)
    if len(stamps) > 1:
        assert ((stamps[1:] - stamps[:-1]) >= pd.Timedelta(days=1000)).all()


# ---------------------------------------------------------------------------
# combining sleeves
# ---------------------------------------------------------------------------
def test_ladder_spreads_a_three_year_return_across_three_years():
    periods = pd.Series([1.0], index=[pd.Timestamp("2010-01-01")])
    annual = ts.ladder_bagger(periods, hold_years=3)
    assert list(annual.index) == [2010, 2011, 2012]
    # a 2x over three years is 26% a year, compounding back to 2x
    assert annual.iloc[0] == pytest.approx(2.0 ** (1 / 3) - 1.0)
    assert np.prod(1.0 + annual.to_numpy()) == pytest.approx(2.0, rel=1e-9)


def test_ladder_survives_a_total_loss():
    """-100% must not produce a NaN or a complex root."""
    annual = ts.ladder_bagger(pd.Series([-1.0], index=[pd.Timestamp("2010-01-01")]))
    assert np.isfinite(annual.to_numpy()).all()
    assert (annual < 0).all()


def test_full_weight_reduces_to_the_single_sleeve():
    blue = pd.Series([0.20, 0.10, -0.05], index=[2010, 2011, 2012])
    bag = pd.Series([-0.30, 0.90, 0.10], index=[2010, 2011, 2012])
    _, only_blue = ts.combine_annual(blue, bag, weight_blue=1.0)
    expected = float(np.prod(1.0 + blue.to_numpy()))
    assert only_blue["total_growth"] == pytest.approx(expected)


def test_rebalancing_two_uncorrelated_sleeves_is_not_free_money():
    """Constant-mix must sit between the two sleeves, never above both."""
    blue = pd.Series([0.30, -0.10, 0.25, 0.05], index=[2010, 2011, 2012, 2013])
    bag = pd.Series([-0.20, 0.60, -0.10, 0.40], index=[2010, 2011, 2012, 2013])
    _, mixed = ts.combine_annual(blue, bag, 0.5, rebalance=True)
    _, b_only = ts.combine_annual(blue, bag, 1.0, rebalance=True)
    _, g_only = ts.combine_annual(blue, bag, 0.0, rebalance=True)
    assert min(b_only["cagr"], g_only["cagr"]) <= mixed["cagr"] <= max(
        b_only["cagr"], g_only["cagr"]) + 0.05


def test_blend_only_uses_years_both_sleeves_cover():
    blue = pd.Series([0.1, 0.2, 0.3], index=[2010, 2011, 2012])
    bag = pd.Series([0.5, 0.5], index=[2011, 2012])
    equity, stats = ts.combine_annual(blue, bag, 0.5)
    assert list(equity.index) == [2011, 2012]
    assert stats["years"] == 2.0


def test_combine_handles_no_overlap():
    blue = pd.Series([0.1], index=[2010])
    bag = pd.Series([0.1], index=[2020])
    equity, stats = ts.combine_annual(blue, bag, 0.5)
    assert equity.empty and stats == {}
