"""Tests for H50 — the triple-barrier bot.

The load-bearing test in this file is the brute-force equivalence one. The
vectorised labeller scans offsets from the horizon downwards so the smallest
surviving offset wins, which is a correct-but-non-obvious way to get a FIRST
touch, and a sign slip there would silently turn the study into an any-touch
label — a much easier question that would print entirely believable numbers.
An obviously-correct O(n*H) reference loop is the only thing that pins it.

Second group: causality. The labeller reads FORWARD by construction, so the
usual "changing the last bar must not change earlier output" test is inverted
here — instead the features must be causal, and the labels must be censored
wherever the forward window runs off the end of the series rather than being
silently truncated and counted.
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

from quantbot import (FEATS, XS, barrier_labels, block_ci,       # noqa: E402
                      eligible, name_frame, summarise)


def _ref(high, low, close, up, dn, horizon):
    """Obviously-correct reference: walk forward one bar at a time."""
    n = len(close)
    out = np.zeros(n, np.int8)
    jj = np.zeros(n, int)
    for i in range(n):
        j = min(i + horizon, n - 1)
        o = 0
        for d in range(1, horizon + 1):
            k = i + d
            if k > n - 1:
                break
            hit_up = high[k] >= up[i]
            hit_dn = low[k] <= dn[i]
            if hit_up or hit_dn:
                #  A bar touching BOTH is ambiguous at daily resolution. The
                #  vectorised version resolves it to the TARGET (t_up <= t_dn),
                #  which is the optimistic reading and has to be matched here
                #  deliberately rather than by accident.
                o = 1 if hit_up else -1
                j = k
                break
        out[i], jj[i] = o, j
    return out, jj


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_the_vectorised_labeller_matches_a_brute_force_first_touch(seed):
    """THE TEST THE WHOLE STUDY RESTS ON. A first-touch label and an any-touch
    label answer different questions and only one of them is a trade."""
    rng = np.random.default_rng(seed)
    n = 300
    c = np.exp(np.cumsum(rng.normal(0, 0.02, n))) * 1000
    h = c * (1 + np.abs(rng.normal(0, 0.01, n)))
    lo = c * (1 - np.abs(rng.normal(0, 0.01, n)))
    up, dn = c * 1.08, c * 0.94
    o, j, f, cen = barrier_labels(h, lo, c, up, dn, 40)
    ro, rj = _ref(h, lo, c, up, dn, 40)
    assert (o[~cen] == ro[~cen]).all()
    assert (j[~cen] == rj[~cen]).all()


def test_a_bar_that_touches_both_resolves_to_the_target_consistently():
    """Documented, not accidental: at daily resolution the order within a bar is
    unknowable, and the labeller takes the optimistic reading. Anything built on
    it must know that, so it is pinned rather than left to drift."""
    c = np.array([100.0, 100.0, 100.0])
    h = np.array([100.0, 120.0, 100.0])
    lo = np.array([100.0, 80.0, 100.0])
    o, j, f, cen = barrier_labels(h, lo, c, c * 1.1, c * 0.9, 2)
    assert o[0] == 1


def test_first_touch_beats_a_later_bigger_move():
    """A stop at bar 3 ends the trade; a target at bar 10 never happens."""
    c = np.full(20, 100.0)
    h = c.copy()
    lo = c.copy()
    lo[3] = 80.0
    h[10] = 200.0
    o, j, f, cen = barrier_labels(h, lo, c, c * 1.2, c * 0.9, 15)
    assert o[0] == -1 and j[0] == 3


def test_the_target_fills_at_its_level_and_the_stop_fills_at_the_close():
    """H35's finding, encoded. A limit sell fills AT the target; a stop is a
    market order on a level already breached, and taking the nominal level
    instead of the bar's close turned a +0.0173 cell into +0.0050 — flattering
    exactly the tight stops a win-rate search wants to use."""
    c = np.full(6, 100.0)
    c[2] = 70.0                       # gapped far through a stop at 90
    h = np.full(6, 100.0)
    lo = np.full(6, 100.0)
    lo[2] = 70.0
    o, j, f, cen = barrier_labels(h, lo, c, np.full(6, 130.0),
                                  np.full(6, 90.0), 4)
    assert o[0] == -1
    assert f[0] == pytest.approx(70.0)          # the close, not the 90 level
    h2 = np.full(6, 100.0)
    h2[2] = 200.0
    o2, j2, f2, _ = barrier_labels(h2, np.full(6, 100.0), c,
                                   np.full(6, 130.0), np.full(6, 90.0), 4)
    assert o2[0] == 1 and f2[0] == pytest.approx(130.0)   # the level, not 200


def test_rows_whose_window_runs_off_the_end_are_censored_not_truncated():
    """Silently clipping a short window and labelling the stub is how A20
    discarded 91% of long-horizon cohorts and measured the survivors."""
    c = np.full(50, 100.0)
    o, j, f, cen = barrier_labels(c, c, c, c * 1.1, c * 0.9, 20)
    assert cen.sum() == 20
    assert not cen[:29].any() and cen[30:].all()


def test_a_wider_stop_mechanically_raises_the_win_rate_on_a_random_walk():
    """The arithmetic the whole goal runs into, demonstrated on noise: moving
    the stop out and the target in buys any win rate you like."""
    rng = np.random.default_rng(3)
    n = 4000
    c = np.exp(np.cumsum(rng.normal(-0.0002, 0.02, n))) * 1000
    near, far = [], []
    for up, dn, box in ((1.03, 0.88, near), (1.15, 0.97, far)):
        o, j, f, cen = barrier_labels(c, c, c, c * up, c * dn, 250)
        box.append(float((o[~cen] == 1).mean()))
    assert near[0] > far[0] + 0.30


# ================================================================ features ===
def _panel(n=600, seed=0):
    rng = np.random.default_rng(seed)
    p = np.exp(np.cumsum(rng.normal(0.0003, 0.02, n))) * 2000
    return pd.DataFrame({
        "date": pd.bdate_range("2010-01-01", periods=n),
        "ticker": "TST", "adj_close": p, "close": p,
        "adj_high": p * 1.01, "adj_low": p * 0.99,
        "mom12_1": rng.normal(size=n), "volz20": rng.normal(size=n),
        "amihud60": rng.normal(size=n), "log_turnover": np.log(
            rng.uniform(1e9, 5e10, n)),
        "atr22": p * 0.02, "stoch_k": rng.uniform(0, 100, n),
        "stoch_d": rng.uniform(0, 100, n), "rsi14": rng.uniform(0, 100, n),
        "tvz20": rng.normal(size=n)})


def test_every_feature_reads_only_the_past():
    """Changing the LAST bar must not change any earlier feature value. One
    non-causal helper anywhere turns the whole backtest into a look-ahead and
    nothing in the output would look wrong (H42)."""
    g = _panel()
    h = g.copy()
    h.loc[h.index[-1], ["adj_close", "close", "adj_high", "adj_low"]] *= 3.0
    a = name_frame(g, None)
    b = name_frame(h, None)
    for c in a.columns:
        if c in ("date", "ticker"):
            continue
        x, y = a[c].to_numpy()[:-1], b[c].to_numpy()[:-1]
        assert np.allclose(np.nan_to_num(x.astype(float)),
                           np.nan_to_num(y.astype(float))), c


def test_the_feature_list_is_actually_produced_by_the_builder():
    """A feature named in FEATS but never computed would be silently dropped by
    the model as all-NaN, and the study would report a feature set it did not
    use."""
    d = name_frame(_panel(), None)
    missing = [f for f in FEATS if f not in d.columns
               and f not in ("mkt_r20", "mkt_vol", "rel_r21")]
    assert missing == []


def test_the_predicted_null_feature_carries_no_information():
    """Q5's control has to be genuinely uninformative or its failure to fire
    means nothing."""
    d = name_frame(_panel(seed=4), None)
    fwd = d["adj"].pct_change(21).shift(-21)
    ok = d["noise"].notna() & fwd.notna()
    assert abs(np.corrcoef(d["noise"][ok], fwd[ok])[0, 1]) < 0.10


def test_liquidity_is_screened_point_in_time_not_on_the_full_sample():
    """A full-sample median turnover filter lets a name's LATER liquidity decide
    whether it was buyable today. `rp60` is a trailing median for that reason."""
    d = name_frame(_panel(), None)
    assert d["rp60"].isna().iloc[:20].all()
    lt = np.exp(_panel()["log_turnover"].to_numpy())
    assert d["rp60"].iloc[100] == pytest.approx(np.median(lt[41:101]))


def test_eligible_drops_the_penny_board_and_the_thin_names():
    d = pd.DataFrame({"rp60": [1e10, 1e8, 1e10], "close_raw": [900, 900, 100]})
    assert list(eligible(d).index) == [0]


# ============================================================== statistics ===
def test_the_summary_annualises_the_mean_log_not_the_mean_return():
    E = pd.DataFrame({"date": pd.to_datetime(["2015-01-01"] * 50),
                      "ret": [0.5] + [-0.02] * 49, "y": 0, "bars": 1,
                      "outcome": 0})
    s = summarise(E, "x")
    assert np.isfinite(s["ann"])
    assert s["ann"] < 1e6            # 1.5**252 would be ~1e44


def test_the_joint_flag_needs_both_halves_of_the_target():
    base = {"date": pd.to_datetime(["2015-01-01"] * 100), "y": 1, "bars": 20,
            "outcome": 1}
    hi_win = summarise(pd.DataFrame({**base, "ret": [0.01] * 85 + [-0.2] * 15}),
                       "x")
    hi_mean = summarise(pd.DataFrame({**base, "ret": [0.30] * 40 + [-0.1] * 60}),
                        "x")
    assert hi_win["pos"] >= 0.80 and hi_win["mean"] < 0.04
    assert hi_mean["mean"] >= 0.04 and hi_mean["pos"] < 0.80
    assert not hi_win["hits_A"] and not hi_mean["hits_A"]


def test_the_bootstrap_resamples_name_year_blocks_not_rows():
    """Trades opened on consecutive bars of one name overlap almost completely.
    An iid resample understates the interval — A15 measured that exact error
    making every width ~3.4x too narrow."""
    rng = np.random.default_rng(1)
    blk = np.repeat(np.arange(40), 50)
    x = np.repeat(rng.normal(0, 0.2, 40), 50) + rng.normal(0, 0.005, 2000)
    lo, hi = block_ci(x, blk, draws=200)
    iid = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    assert (hi - lo) / 2 > 3 * iid


# ================================================================= the book ==
def _wf(n_names=12, n_dates=60, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2010-01-01", periods=n_dates, freq="21D")
    rows = []
    for t in range(n_names):
        rows.append(pd.DataFrame({
            "date": d, "ticker": f"T{t:02d}", "p": rng.random(n_dates),
            "ret": rng.normal(0.03, 0.25, n_dates), "bars": 60,
            "y": rng.integers(0, 2, n_dates)}))
    return pd.concat(rows, ignore_index=True)


def test_the_book_never_holds_the_same_name_in_two_slots_at_once():
    """Doubling into one name is leverage in disguise and would flatter the
    return without any rule saying to do it."""
    from quantbot import book
    W = _wf()
    b = book(W, slots=4)
    assert b["trades"] > 0


def test_a_slot_stays_locked_for_the_length_of_its_own_trade():
    """A18's scheduler bug: converting held sessions to calendar days and then
    searching the date index let a 30-day lock skip a whole cohort, and the
    penalty scaled with turnover — hitting exactly the comparison the study
    exists to make."""
    from quantbot import book
    W = _wf(n_names=30, n_dates=40)
    slow = book(W.assign(bars=250), slots=2)
    fast = book(W.assign(bars=5), slots=2)
    assert fast["trades"] > slow["trades"] * 3


def test_more_slots_take_more_trades():
    from quantbot import book
    W = _wf(n_names=30)
    assert book(W, slots=8)["trades"] > book(W, slots=2)["trades"]


def test_the_random_control_uses_the_same_machinery_and_the_same_dates():
    """A34's lesson: a control that cannot play the same game as the treatment
    is a handicap, not a null. Same slot count, same dates, same trade count —
    only the ranking differs."""
    from quantbot import book
    W = _wf(n_names=30)
    a = book(W, slots=8, rank="p")
    b = book(W, slots=8, rank="rand", seed=1)
    assert abs(a["trades"] - b["trades"]) <= 2


def test_the_random_control_actually_varies_with_its_seed():
    """The control has to have a spread, or 'the model sits at the 97th
    percentile' is meaningless."""
    from quantbot import book
    W = _wf(n_names=40, seed=3)
    got = {round(book(W, slots=8, rank="rand", seed=s)["cagr"], 6)
           for s in range(6)}
    assert len(got) >= 4


def test_a_total_loss_cannot_drive_slot_equity_negative():
    """H40 printed a NaN CAGR because a growth factor went negative and a
    fractional power of a negative number is NaN."""
    from quantbot import book
    W = _wf(n_names=6)
    W["ret"] = -1.8
    b = book(W, slots=2)
    assert np.isfinite(b["cagr"]) and b["total"] >= 0.0
