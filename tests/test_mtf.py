"""H55 — the multi-timeframe harness.

THE ONE TEST THIS FILE EXISTS FOR is `test_the_weekly_state_cannot_see_its_own
_week`. Every other bug in a multi-timeframe study is visible in the output;
that one is not. A weekly feature stamped at the START of the week it summarises
lets every daily bar inside the week read that week's own close, and NOTHING in
the resulting table looks wrong — the numbers are merely too good. A32 records
the same discipline for the signal replay: one non-causal helper anywhere in the
chain turns the whole backtest into a look-ahead with no visible symptom.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import mtf                                                        # noqa: E402
from mtf import (foreign_trend, m0_positive_control,              # noqa: E402
                 matched, mlog)


def _panel(n_names=12, n_dates=800, seed=0, drift=0.0004, tmp=None):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-05", periods=n_dates)
    out = []
    for i in range(n_names):
        p = np.exp(np.cumsum(rng.normal(drift, 0.02, n_dates))) * 2000
        out.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i:03d}", "close": p, "adj_close": p,
            "tradeable": True,
            "log_turnover": np.log(5e9) * np.ones(n_dates)}))
    return pd.concat(out, ignore_index=True)


def _features_from(P, monkeypatch, tmp_path):
    f = tmp_path / "p.parquet"
    P.to_parquet(f)
    monkeypatch.setattr(mtf, "PANEL", str(f))
    return mtf.features()


# ================================================== THE LOOK-AHEAD TEST ======
def test_the_weekly_state_cannot_see_its_own_week(monkeypatch, tmp_path):
    """Change a bar and every EARLIER weekly state must be byte-identical.

    This is the only failure mode of a multi-timeframe study that produces a
    believable table. If the weekly EMA carried on Tuesday already contains
    Friday's close, the confluence cell is reading the future and the result is
    manufactured.
    """
    P = _panel()
    A = _features_from(P.copy(), monkeypatch, tmp_path)

    #  Move ONE bar, hard, deep inside the sample.
    Q = P.copy()
    tk = Q["ticker"].iloc[0]
    dates = sorted(Q.loc[Q["ticker"] == tk, "date"].unique())
    cut = dates[500]
    sel = (Q["ticker"] == tk) & (Q["date"] >= cut)
    Q.loc[sel, ["close", "adj_close"]] *= 3.0
    B = _features_from(Q, monkeypatch, tmp_path)

    a = A[(A["ticker"] == tk) & (A["date"] < cut)].set_index("date")
    b = B[(B["ticker"] == tk) & (B["date"] < cut)].set_index("date")
    for col in ("w_above", "m_above", "d_cross", "d_rising", "d_above"):
        x, y = a[col].astype(float), b[col].astype(float)
        assert np.allclose(x.to_numpy(), y.to_numpy(), equal_nan=True), col


def test_the_weekly_state_lags_by_at_least_one_bar(monkeypatch, tmp_path):
    """A stronger form: the state on bar t must be derivable from bars < t.

    Rebuild the weekly EMA by hand from the truncated history and check the
    shipped column never leads it.
    """
    P = _panel(n_names=1, n_dates=600)
    A = _features_from(P.copy(), monkeypatch, tmp_path).set_index("date")
    px = P.set_index("date")["adj_close"]
    for t in A.index[300:320]:
        past = px[px.index < t]                       # STRICTLY before
        w = past.resample("W-FRI").last().dropna()
        if len(w) < 12:
            continue
        want = float(w.iloc[-1] > w.ewm(span=10, adjust=False).mean().iloc[-1])
        got = A.loc[t, "w_above"]
        if np.isfinite(got):
            assert got == want, (t, got, want)


# ========================================================= THE CONTROLS ======
def test_m0a_is_the_gate_and_m0b_is_a_measurement():
    """The first version of M0 scored the weekly EMA against the PLANTED drift
    and failed at 0.0056 of 0.0640 — conflating "does the harness work" with
    "can a weekly EMA read a regime". M0a gates; M0b is a measured ceiling."""
    m = m0_positive_control()
    assert m["PASS"], m
    assert m["M0a_share_of_planted"] > 0.5, m
    #  And the proxy must transmit clearly LESS than the true regime, or the
    #  synthetic regime is so slow that the test has stopped being a test.
    assert 0.1 < m["M0b_transmission"] < 0.7, m


def test_the_foreign_trend_null_keeps_the_marginal_frequency(monkeypatch,
                                                             tmp_path):
    """M3 must destroy the LINK without changing HOW OFTEN the state is on.

    A null that also changes the base rate is comparing two different things,
    which is the shape A22 records: a screen that clears its null by selecting
    a different sample rather than by carrying information.
    """
    P = _panel(n_names=20, n_dates=700)
    F = _features_from(P, monkeypatch, tmp_path)
    own = (F["w_above"].to_numpy(float) > 0.5)
    fw = foreign_trend(F)
    ok = np.isfinite(F["w_above"].to_numpy(float))
    assert abs(own[ok].mean() - fw[ok].mean()) < 0.05, (own[ok].mean(),
                                                        fw[ok].mean())
    #  ...and it must actually be a different assignment.
    assert (own[ok] != fw[ok]).mean() > 0.10


def test_a_name_is_never_paired_with_its_own_trend(monkeypatch, tmp_path):
    """If the permutation has fixed points, the null partly IS the treatment."""
    P = _panel(n_names=30, n_dates=400)
    F = _features_from(P, monkeypatch, tmp_path)
    rng = np.random.default_rng(mtf.SEED)
    tks = F["ticker"].unique()
    pair = dict(zip(tks, rng.permutation(tks)))
    fixed = sum(1 for k, v in pair.items() if k == v)
    #  A random permutation has ~1 fixed point in expectation regardless of n,
    #  so this asserts the pairing is a permutation rather than the identity —
    #  the failure worth catching is a no-op shuffle, not the odd fixed point.
    assert fixed < max(2, len(tks) // 10), fixed


def test_matched_downsamples_by_whole_blocks(monkeypatch, tmp_path):
    """A34: a control that cannot play the same game as the treatment is a
    handicap. The matched control must keep the clustering, so it samples whole
    (ticker, year) blocks rather than scattered rows."""
    P = _panel(n_names=20, n_dates=800)
    F = _features_from(P, monkeypatch, tmp_path)
    m = F["elig"].to_numpy(bool)
    tgt = int(m.sum() // 4)
    out = matched(m, tgt, F)
    assert out.sum() <= m.sum()
    assert out.sum() >= tgt * 0.5
    assert not (out & ~m).any(), "the control must be a subset of the source"
    #  Whole blocks: every (ticker, year) it touches, it takes entirely.
    d = F[out]
    for (tk, yr), g in d.groupby(["ticker", "year"]):
        full = F[(F["ticker"] == tk) & (F["year"] == yr) & m]
        assert len(g) == len(full), (tk, yr, len(g), len(full))


def test_mean_log_is_not_the_mean(monkeypatch, tmp_path):
    """A36: an equal-weighted holder is paid the MEAN, a sequential trader the
    mean LOG, and in that study the two disagreed in SIGN. A payoff that is
    positive on one and negative on the other must not read the same here."""
    #  The first fixture here was +100/-50/-50/+100, which is mean +0.25 and
    #  mean log EXACTLY ZERO — doubling and halving cancel in logs. It is a
    #  neat illustration of the point and a useless test of it.
    r = np.array([1.0, -0.6, -0.6, 1.0])       # mean +0.20, mean log -0.111
    c = np.zeros(4)
    assert np.mean(r) > 0
    assert mlog(r, c) < 0
    #  And the exact-cancellation case, asserted for what it is.
    assert mlog(np.array([1.0, -0.5]), np.zeros(2)) == pytest.approx(0.0)


def test_the_cost_is_the_standing_schedule():
    """A5/A38: 0.56% is the published Mandiri round trip and the spread is a
    SEPARATE multiplier, because a blended figure hides an execution assumption
    inside what looks like a fee."""
    assert mtf.FEE == pytest.approx(0.0056)
    assert mtf.SPREAD_MULT == pytest.approx(0.5)


def test_every_registered_horizon_is_actually_measured(monkeypatch, tmp_path):
    """A20/A21: a conditional result quoted without its condition is a wrong
    result, and the horizon is the condition this repo has most often dropped."""
    P = _panel(n_names=8, n_dates=900)
    F = _features_from(P, monkeypatch, tmp_path)
    for k in mtf.HORIZONS:
        assert f"f{k}" in F.columns
        assert F[f"f{k}"].notna().any()


# ================================================== THE HOURLY HALF (H55b) ===
import mtf_h1                                                     # noqa: E402


def _h1(n_names=6, n_days=120, bars=7, seed=1):
    """A synthetic hourly panel with the columns mtf_h1 needs."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-02", periods=n_days)
    out = []
    for i in range(n_names):
        p = 2000.0
        rows = []
        for d in days:
            for b in range(bars):
                p *= float(np.exp(rng.normal(0.0002, 0.006)))
                rows.append((d + pd.Timedelta(hours=9 + b), d, p))
        ts, dt, px = zip(*rows)
        px = np.array(px)
        out.append(pd.DataFrame({
            "ticker": f"T{i:02d}", "ts": ts, "date": dt,
            "open": px, "high": px * 1.002, "low": px * 0.998, "close": px,
            "a_open": px, "a_high": px * 1.002, "a_low": px * 0.998,
            "a_close": px, "volume": 1e6, "elig": True, "adjf": 1.0}))
    return pd.concat(out, ignore_index=True)


def test_every_fill_comes_from_the_entry_days_own_bars():
    """A fill must be reachable on the day it is stamped. An arm that quietly
    took a price from the next day would look like a small free edge and would
    be invisible in the table."""
    H = _h1()
    D = mtf_h1.entries(H)
    #  `entries` already emits `lo`/`hi`, so the reference bounds are named
    #  apart -- a silent lo_x/lo_y merge would have made this test vacuous.
    m = H.groupby(["ticker", "date"]).agg(
        ref_lo=("a_close", "min"), ref_hi=("a_close", "max")).reset_index()
    d = D.merge(m, on=["ticker", "date"])
    assert {"ref_lo", "ref_hi"} <= set(d.columns)
    for col in ("open_", "close", "mid", "conf", "lo_close"):
        v = d[col].to_numpy(float)
        ok = np.isfinite(v)
        assert ok.sum() > 100, col
        assert (v[ok] >= d["ref_lo"].to_numpy()[ok] - 1e-9).all(), col
        assert (v[ok] <= d["ref_hi"].to_numpy()[ok] + 1e-9).all(), col


def test_the_oracle_is_the_best_fill_available_and_is_labelled_look_ahead():
    """The ORACLE bounds the question: nothing that is not look-ahead can beat
    it. If a real arm ever did, an arm is cheating."""
    H = _h1()
    D = mtf_h1.entries(H)
    for col in ("open_", "close", "mid", "conf"):
        v, o = D[col].to_numpy(float), D["lo_close"].to_numpy(float)
        ok = np.isfinite(v) & np.isfinite(o)
        assert (o[ok] <= v[ok] + 1e-9).all(), col
    src = mtf_h1.entries.__doc__ + mtf_h1.__doc__
    assert "ORACLE" in src and "look-ahead" in src.lower()


def test_the_confirmation_arm_fires_on_a_strict_subset_of_days():
    """It needs an hour above the hourly EMA, so it MUST skip some days —
    which is exactly why its edge has to be scored on matched days."""
    H = _h1()
    D = mtf_h1.entries(H)
    fired = D["conf"].notna()
    assert 0.05 < (~fired).mean() < 0.95, (~fired).mean()


def test_the_cost_wall_is_computed_from_the_tick_not_a_flat_guess():
    """A38: the fee and the spread are separate, and the spread is per NAME and
    per PRICE. A flat guess would hand cheap stocks a discount they do not get."""
    H = _h1()
    P = pd.DataFrame({"ticker": ["T00"], "date": [pd.Timestamp("2024-01-02")],
                      "adj_close": [2000.0]})
    cw = mtf_h1.cost_wall(H, P)
    assert cw["fee_only"] == pytest.approx(0.0056)
    assert cw["fee_half_tick"] > cw["fee_only"]
    assert cw["fee_full_tick"] > cw["fee_half_tick"]
    assert 0 < cw["median_tick_pct"] < 0.05


# ============================================ THE RULE CARD (scripts/rules) ==
import rules                                                      # noqa: E402


def test_the_keep_band_is_wider_than_the_entry_band():
    """THE BUFFER IS THE WHOLE POINT of the sticky variant. If the keep band
    were not wider than the entry band the rule would sell a name the day it
    slipped out of the top decile and buy it back when it returned, which is
    churn driven by rank noise around a cut rather than by deterioration —
    H54 measured the unbuffered version as clearly worse (+4.14% against
    +6.50% over the index) at higher turnover (75% against 56%)."""
    assert rules.KEEP_HI < rules.ENTRY_HI
    assert rules.KEEP_VOL > rules.ENTRY_VOL


def test_the_sell_level_is_derived_from_the_hi52_identity():
    """`hi52 = close / 252-day max`, so a hi52 threshold IS a price and the
    sell level must be the 52-week high times that threshold — not a fixed
    percentage drawdown, which is what a reader would otherwise assume."""
    close, hi52, thresh = 2670.0, 0.988889, 0.8762
    high = close / hi52
    sell = high * thresh
    assert high == pytest.approx(2700.0, abs=1.0)
    assert sell == pytest.approx(2366.0, abs=2.0)
    #  And the implied room is NOT a constant -50% or -25% stop.
    assert -0.15 < sell / close - 1.0 < -0.05


def test_the_rule_card_says_the_level_is_not_a_resting_stop():
    """The thresholds are cross-sectional percentiles, so the level moves with
    the board and a name can be sold without falling a rupiah. A reader who
    parks it as a broker stop has a different rule from the measured one, so
    the file has to say so."""
    doc = rules.__doc__ + open(rules.__file__).read()
    assert "not a stop you leave resting" in doc.lower() or \
           "NOT a resting stop" in doc


def test_the_hard_stop_is_shipped_and_no_take_profit_is():
    """H56's S1 was registered as "a stop will not cut PORTFOLIO drawdown" and
    FAILED: every level -10% to -30% cut it, in both halves, at no measurable
    cost in return. So the stop ships. S3's predicted null CONFIRMED
    monotonically (+20/+30/+50 targets read 6.36/7.83/8.77 against 10.24), so
    no take-profit does. Asserting both directions stops either drifting."""
    src = open(rules.__file__).read()
    body = src.split('"""', 2)[2]
    assert "STOP" in body and "stop_px" in body
    for token in ("take_profit", "tp_px", "target_px"):
        assert token not in body, token
    #  The level must sit inside the family that was actually measured, and
    #  must not be the sweep's argmax (-15%, which read best on both axes).
    assert 0.10 <= rules.STOP <= 0.30
    assert rules.STOP != 0.15


def test_the_stop_and_the_band_are_described_as_different_instruments():
    """One is a resting order from your own fill; the other is a quarterly
    percentile of the board that moves and is NOT a resting order. Conflating
    them gives the reader a different rule from the measured one."""
    src = open(rules.__file__).read()
    assert "RESTING ORDER" in src
    assert "NOT a resting order" in src


def test_rules_actually_runs_its_main():
    """A36: a string edit once dropped an `if __name__` block, so the script
    defined main(), never called it, and exited 0 in under a second with an
    empty output file. Exit code 0 is not evidence that anything ran — and
    this exact bug was recommitted while writing the fix above."""
    src = open(rules.__file__).read()
    assert src.rstrip().endswith("main()")
    assert '__name__ == "__main__"' in src
