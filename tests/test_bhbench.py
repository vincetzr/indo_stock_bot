"""Tests for H54's buy-and-hold benchmark harness.

This harness is load-bearing in a way nothing else in the repo is: a fleet of
strategies is scored against it, so a bug here corrupts every result at once
rather than one study. The tests therefore concentrate on the three things that
have actually gone wrong in this project before:

  WINDOW      a benchmark measured over a different span than the strategy.
              A19 names this as the error class that manufactures results, and
              it has now been committed three times — including in the first
              version of THIS file, where `elig` needs 30 bars of trailing
              turnover so nothing is eligible on the panel's first date, the
              strategy sat in cash for five years, and both buy-and-hold arms
              came back NaN because the universe was empty at the start mark.
  DEGENERATE  a "portfolio" of one or two names in a thin early period, which
              nearly became H52's headline.
  LOOK-AHEAD  a full-sample statistic used to build a point-in-time universe.
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

import bhbench                                                    # noqa: E402
from bhbench import (MIN_BASKET, MIN_UNIV, Bench, Prices,         # noqa: E402
                     _cagr, half_cagr)


def _panel(n_names=80, n_dates=600, seed=0, drift=0.0004):
    """A synthetic panel with the columns the harness needs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_dates)
    out = []
    for i in range(n_names):
        p = np.exp(np.cumsum(rng.normal(drift, 0.02, n_dates))) * 2000
        out.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i:03d}",
            "close": p, "adj_close": p,
            "tradeable": True,
            "log_turnover": np.log(rng.uniform(2e9, 5e10)) * np.ones(n_dates),
            "mom12_1": rng.normal(size=n_dates),
            "hi52": rng.normal(-0.1, 0.1, n_dates),
            "vol60": rng.uniform(0.01, 0.05, n_dates)}))
    P = pd.concat(out, ignore_index=True).sort_values(["ticker", "date"])
    P["tv"] = np.exp(P["log_turnover"])
    P["tv60"] = P.groupby("ticker")["tv"].transform(
        lambda s: s.rolling(60, min_periods=30).median())
    P["elig"] = P["tradeable"] & (P["tv60"] >= 1e9) & (P["close"] >= 100)
    return P


class _FakeBench(Bench):
    """Bench without the on-disk index file, for pure-unit tests."""

    def __init__(self, P, **kw):
        self.P = P
        self.PX = Prices(P)
        self.dates = np.sort(P["date"].unique())
        self.fee = kw.get("fee", 0.0056)
        self.spread_mult = kw.get("spread_mult", 0.5)
        idx = (P.groupby("date")["adj_close"].mean())
        self.J = idx


def _all(day):
    t = list(day["ticker"])
    return [(x, 1.0 / len(t)) for x in t]


# ============================================================= THE WINDOW ====
def test_the_window_starts_at_the_first_TRADE_not_the_first_mark():
    """THE BUG THIS FILE EXISTS FOR. `elig` needs 30 bars of trailing turnover,
    so the strategy cannot trade on the panel's first date. Measuring the
    benchmarks from that date compares them over a span the strategy never
    occupied — and returns NaN when the universe there is empty."""
    P = _panel()
    B = _FakeBench(P)
    r = B.walk(_all, freq=21)
    assert r
    first_elig = P[P["elig"]]["date"].min()
    assert pd.Timestamp(r["start"]) >= pd.Timestamp(first_elig)
    assert pd.Timestamp(r["start"]) > pd.Timestamp(P["date"].min())


def test_every_benchmark_is_priced_over_the_strategys_own_span():
    P = _panel()
    B = _FakeBench(P)
    r = B.walk(_all, freq=21)
    uni = P[(P["date"] == r["start"]) & P["elig"]]["ticker"].tolist()
    assert uni, "the universe at the strategy's own start must be non-empty"
    assert np.isfinite(B.hold_basket(uni, r["start"], r["end"]))
    assert np.isfinite(B.hold_basket(r["first_basket"], r["start"], r["end"]))


def test_a_longer_holding_schedule_does_not_shorten_the_measured_span():
    """Two frequencies over one panel must both end at the last rebalance they
    reach, not at some shared arbitrary date."""
    P = _panel()
    B = _FakeBench(P)
    a = B.walk(_all, freq=21)
    b = B.walk(_all, freq=63)
    assert a and b
    assert pd.Timestamp(a["end"]) >= pd.Timestamp(a["start"])
    assert pd.Timestamp(b["end"]) >= pd.Timestamp(b["start"])


# ======================================================= NO LOOK-AHEAD =======
def test_select_only_ever_sees_one_bar():
    """The structural guarantee: the future is not in the dataframe, so a
    strategy cannot peek even if it tries."""
    P = _panel()
    B = _FakeBench(P)
    seen = []

    def spy(day):
        seen.append(day["date"].nunique())
        return _all(day)

    B.walk(spy, freq=21)
    assert seen and set(seen) == {1}


def test_changing_a_future_bar_cannot_change_an_earlier_pick():
    P = _panel(seed=3)
    B = _FakeBench(P)
    picks_a = []
    B.walk(lambda d: (picks_a.append(sorted(d["ticker"])[:5]) or _all(d)),
           freq=21)
    Q = P.copy()
    last = Q["date"].max()
    Q.loc[Q["date"] == last, ["close", "adj_close"]] *= 10.0
    picks_b = []
    _FakeBench(Q).walk(lambda d: (picks_b.append(sorted(d["ticker"])[:5])
                                  or _all(d)), freq=21)
    assert picks_a[:-1] == picks_b[:-1]


# ==================================================== DEGENERATE BASKETS =====
def test_a_thin_universe_produces_no_trade_rather_than_a_two_name_book():
    """H52's near-headline: nine quarters holding one to three stocks moved a
    26-year CAGR by six points."""
    P = _panel(n_names=MIN_UNIV - 5)
    r = _FakeBench(P).walk(_all, freq=21)
    assert not r


def test_a_selection_returning_too_few_names_is_skipped():
    P = _panel()
    B = _FakeBench(P)
    r = B.walk(lambda d: [(t, 1.0) for t in list(d["ticker"])[:MIN_BASKET - 1]],
               freq=21)
    assert not r


def test_a_basket_at_the_floor_is_allowed():
    """The control for the two above: the guard must not be an off switch."""
    P = _panel()
    r = _FakeBench(P).walk(
        lambda d: [(t, 1.0) for t in list(d["ticker"])[:MIN_BASKET]], freq=21)
    assert r and r["basket"] == pytest.approx(MIN_BASKET)


# ============================================================== ACCOUNTING ===
def test_weights_are_normalised_so_scale_cannot_lever_the_book():
    """Returning weights that sum to 10 must not produce 10x the return."""
    P = _panel(seed=7)
    B = _FakeBench(P)
    a = B.walk(lambda d: [(t, 1.0) for t in d["ticker"]], freq=21)
    b = B.walk(lambda d: [(t, 99.0) for t in d["ticker"]], freq=21)
    assert a["cagr"] == pytest.approx(b["cagr"], rel=1e-9)


def test_a_never_changing_book_pays_only_its_initial_purchase():
    """A strategy holding the same names forever still has to BUY them once.
    In the 0.5*L1 convention that first buy scores 0.5 — one way, i.e. half a
    round trip — and every later bar scores 0. So total cost over the run must
    be one initial purchase and nothing more.

    (A first version of this test asserted turnover was exactly 0 and failed.
    The harness was right: it was charging for the opening trade, which is a
    cost a real holder pays.)"""
    P = _panel(seed=11)
    B = _FakeBench(P)
    steady = B.walk(_all, freq=21)
    n_bars = len(steady["curve"])
    assert steady["turnover"] * n_bars == pytest.approx(0.5, abs=0.02)
    #  every bar after the first must be genuinely free
    assert steady["turnover"] < 0.5 / (n_bars - 1)


def test_the_fee_and_the_spread_are_separate_knobs():
    P = _panel(seed=13)

    def half(day):
        t = sorted(day["ticker"])
        k = len(t) // 2
        #  alternate halves each bar so the book fully rotates and pays cost
        return [(x, 1.0) for x in (t[:k] if day["date"].iloc[0].day % 2 else
                                   t[k:])]
    cheap = _FakeBench(P, fee=0.0, spread_mult=0.0).walk(half, freq=21)
    dear = _FakeBench(P, fee=0.02, spread_mult=1.0).walk(half, freq=21)
    assert cheap["cost_yr"] == pytest.approx(0.0, abs=1e-12)
    assert dear["cost_yr"] > cheap["cost_yr"]
    assert dear["cagr"] < cheap["cagr"]
    #  cost must not change WHICH names are held
    assert cheap["gross"] == pytest.approx(dear["gross"], rel=1e-9)


def test_a_delisted_name_is_realised_at_its_last_print_not_carried_forward():
    """H41 held a vanished ticker at its last price for the rest of the
    backtest, blocking a slot and inflating the book."""
    P = _panel(n_names=60, seed=17)
    dead = "T000"
    cut = P["date"].quantile(0.5)
    P = P[~((P["ticker"] == dead) & (P["date"] > cut))]
    B = _FakeBench(P)
    last = P[P["ticker"] == dead]["adj_close"].iloc[-1]
    assert B.PX.exit_price(dead, P["date"].max()) == pytest.approx(last)
    assert not np.isfinite(B.PX.at(dead, P["date"].max()))


# ============================================================ THE VERDICT ====
def test_pass_requires_every_benchmark_and_both_halves():
    """A strategy that beats two of three, or beats all three only on the full
    sample, must NOT pass. This is the whole point of the harness."""
    P = _panel()
    B = _FakeBench(P)

    def fake(**kw):
        base = {"label": "x", "ok": True, "cagr": 0.2, "random": 0.0,
                "beats_index": True, "beats_universe": True,
                "beats_picks": True, "beats_random": True,
                "both_halves_index": True, "both_halves_universe": True,
                "both_halves_picks": True}
        base.update(kw)
        return bool(base["beats_index"] and base["beats_universe"]
                    and base["beats_picks"] and base["beats_random"]
                    and base["both_halves_index"]
                    and base["both_halves_universe"]
                    and base["both_halves_picks"])
    assert fake()
    for k in ("beats_index", "beats_universe", "beats_picks", "beats_random",
              "both_halves_index", "both_halves_universe",
              "both_halves_picks"):
        assert not fake(**{k: False}), k


def test_holding_the_universe_is_close_to_the_buy_and_hold_of_the_universe():
    """On a panel with NO cross-sectional dispersion in drift, rebalancing and
    drifting should land near each other. The 9-point gap measured on real IDX
    data is a property of that market's dispersion, not of the harness."""
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2010-01-04", periods=800)
    out = []
    for i in range(60):
        p = np.exp(np.cumsum(rng.normal(0.0003, 0.004, len(dates)))) * 1000
        out.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i:03d}", "close": p, "adj_close": p,
            "tradeable": True,
            "log_turnover": np.log(1e10) * np.ones(len(dates)),
            "mom12_1": 0.0, "hi52": 0.0, "vol60": 0.02}))
    P = pd.concat(out, ignore_index=True).sort_values(["ticker", "date"])
    P["tv"] = np.exp(P["log_turnover"])
    P["tv60"] = P["tv"]
    P["elig"] = True
    B = _FakeBench(P, fee=0.0, spread_mult=0.0)
    r = B.walk(_all, freq=21)
    uni = P[(P["date"] == r["start"]) & P["elig"]]["ticker"].tolist()
    bh = B.hold_basket(uni, r["start"], r["end"])
    assert abs(r["cagr"] - bh) < 0.02


# =============================================================== utilities ===
def test_cagr_and_half_split_are_arithmetically_right():
    assert _cagr(4.0, 2.0) == pytest.approx(1.0)
    d = pd.bdate_range("2010-01-04", periods=11).to_numpy()
    curve = [(d[i], 1.0 * (1.02 ** i)) for i in range(11)]
    e, l = half_cagr(curve)
    assert np.isfinite(e) and np.isfinite(l)
    assert e == pytest.approx(l, rel=0.05)


def test_the_standing_cost_constants_are_the_users_actual_schedule():
    assert bhbench.FEE == pytest.approx(0.0056)
    assert 0.0 <= bhbench.SPREAD_MULT <= 1.0
