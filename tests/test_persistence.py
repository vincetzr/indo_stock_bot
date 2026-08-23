"""Tests for §12's persistence statistic.

The statistic is vectorised so that 200 permutation draws are affordable, and
vectorised code is exactly where a silent indexing error hides. H9 in this repo
shipped a null that assigned period-ordered permutations against a
ticker-sorted frame and read t = −2.96 against a signal's −2.86; a null that
certifies anything is worse than no null. So this module is tested three ways:

  AGAINST A READABLE REFERENCE. The pandas implementation anybody would write
  first is here, and the fast path must agree with it exactly.

  ON KNOWN ANSWERS. Injected persistence must be found; independent noise must
  not be; and a pure TICKER effect with no broker effect must be reproduced by
  the null, because that is the specific false positive the design exists to
  catch.

  ON THE SHUFFLE ITSELF. It must preserve each group's multiset of labels and
  actually move them.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.spine.persistence import (MIN_BROKERS,                # noqa: E402
                                      adjacent_corr, margin_matrix,
                                      permutation_test, shuffle_within,
                                      spearman)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def reference_matrix(df, min_windows, min_gross):
    """The readable version. Slow, obvious, and the thing the fast path must
    reproduce."""
    a = df.groupby(["broker", "period"]).agg(
        pnl=("pnl", "sum"), gross=("gross", "sum"),
        n=("window", "nunique")).reset_index()
    a = a[(a["n"] >= min_windows) & (a["gross"] >= min_gross)]
    a = a[a["gross"] > 0].copy()
    a["bps"] = 10000.0 * a["pnl"] / a["gross"]
    brokers = sorted(df["broker"].unique())
    periods = sorted(df["period"].unique())
    M = np.full((len(brokers), len(periods)), np.nan)
    bi = {b: i for i, b in enumerate(brokers)}
    pi = {p: i for i, p in enumerate(periods)}
    for _, r in a.iterrows():
        M[bi[r["broker"]], pi[r["period"]]] = r["bps"]
    return M


def fast_matrix(df, min_windows, min_gross):
    b, bu = pd.factorize(df["broker"], sort=True)
    p, pu = pd.factorize(df["period"], sort=True)
    w, wu = pd.factorize(df["window"], sort=True)
    return margin_matrix(b.astype(np.int64), p.astype(np.int64),
                         w.astype(np.int64),
                         df["pnl"].to_numpy(float),
                         df["gross"].to_numpy(float),
                         len(bu), len(pu), len(wu), min_windows, min_gross)


def frame(n_brokers, n_periods, n_win, rng, broker_skill=None,
          ticker_drift=None, n_tickers=4):
    """A synthetic broker panel with optional injected structure."""
    rows = []
    for w in range(n_win):
        period = w * n_periods // n_win
        for t in range(n_tickers):
            drift = 0.0 if ticker_drift is None else ticker_drift[t]
            for b in range(n_brokers):
                skill = 0.0 if broker_skill is None else broker_skill[b]
                gross = float(rng.uniform(2e9, 1e10))
                rows.append({
                    "broker": f"B{b:02d}", "period": period, "window": w,
                    "ticker": f"T{t}",
                    "pnl": (skill + drift) * gross + rng.normal(0, 0.02) * gross,
                    "gross": gross})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# the fast path must equal the readable one
# --------------------------------------------------------------------------
def test_fast_matrix_matches_the_pandas_reference():
    rng = np.random.default_rng(7)
    df = frame(9, 3, 30, rng)
    a = reference_matrix(df, 4, 1e9)
    b = fast_matrix(df, 4, 1e9)
    assert np.allclose(a, b, equal_nan=True)


def test_the_window_guard_counts_distinct_windows_not_rows():
    """A broker trading four tickers in one window has four rows and one
    window. Counting rows would let it through a guard it should fail."""
    df = pd.DataFrame({
        "broker": ["A"] * 4, "period": [0] * 4, "window": [0] * 4,
        "ticker": list("wxyz"), "pnl": [1e8] * 4, "gross": [1e10] * 4})
    M = fast_matrix(df, min_windows=2, min_gross=0.0)
    assert np.isnan(M[0, 0]), "one window is one window, whatever the row count"
    assert np.isfinite(fast_matrix(df, 1, 0.0)[0, 0])


def test_the_gross_guard_drops_a_broker_that_barely_traded():
    df = pd.DataFrame({
        "broker": ["A", "B"], "period": [0, 0], "window": [0, 1],
        "ticker": ["t", "t"], "pnl": [1.0, 1e8], "gross": [10.0, 1e10]})
    M = fast_matrix(df, 1, 1e9)
    assert np.isnan(M[0, 0]) and np.isfinite(M[1, 0])


def test_margin_is_value_weighted_not_an_average_of_margins():
    """§9.3's margin_bps is pnl over gross traded value. A big losing window
    and a small winning one must not average out to neutral."""
    df = pd.DataFrame({
        "broker": ["A", "A"], "period": [0, 0], "window": [0, 1],
        "ticker": ["t", "t"], "pnl": [-1e9, +1e8], "gross": [1e11, 1e9]})
    M = fast_matrix(df, 1, 0.0)
    assert M[0, 0] == pytest.approx(10000 * (-9e8) / 1.01e11)


# --------------------------------------------------------------------------
# known answers
# --------------------------------------------------------------------------
def test_injected_persistence_is_detected():
    """Brokers with fixed, distinct skill must rank the same way in every
    period. If this fails the statistic cannot find persistence that is there."""
    rng = np.random.default_rng(1)
    skill = np.linspace(-0.02, 0.02, 12)
    df = frame(12, 4, 60, rng, broker_skill=skill)
    r, pairs = adjacent_corr(fast_matrix(df, 4, 1e9))
    assert r > 0.7, f"persistent skill should show a high rank corr, got {r}"
    assert len(pairs) == 3


def test_independent_noise_is_not_detected():
    rng = np.random.default_rng(2)
    df = frame(12, 4, 60, rng)
    r, _ = adjacent_corr(fast_matrix(df, 4, 1e9))
    assert abs(r) < 0.5


def test_a_pure_ticker_effect_is_reproduced_by_the_null():
    """THE false positive this design exists to catch. Here every broker is
    identical and only the TICKERS drift; any apparent persistence is the
    name's drift, and the label shuffle must reproduce it rather than collapse
    to zero."""
    rng = np.random.default_rng(3)
    drift = np.array([0.03, 0.01, -0.01, -0.03])
    df = frame(10, 4, 60, rng, ticker_drift=drift)
    grp, _ = pd.factorize(df["ticker"].astype(str) + "|"
                          + df["window"].astype(str), sort=True)
    b, bu = pd.factorize(df["broker"], sort=True)
    p, pu = pd.factorize(df["period"], sort=True)
    w, wu = pd.factorize(df["window"], sort=True)
    obs, _, nulls, pval = permutation_test(
        grp.astype(np.int64), b.astype(np.int64), p.astype(np.int64),
        w.astype(np.int64), df["pnl"].to_numpy(float),
        df["gross"].to_numpy(float), len(bu), len(pu), len(wu),
        4, 1e9, draws=60, seed=11)
    assert pval > 0.05, (
        "a ticker effect with no broker effect must not read as significant "
        f"persistence (p={pval})")


def test_injected_persistence_beats_its_own_null():
    rng = np.random.default_rng(4)
    skill = np.linspace(-0.02, 0.02, 12)
    df = frame(12, 4, 60, rng, broker_skill=skill)
    grp, _ = pd.factorize(df["ticker"].astype(str) + "|"
                          + df["window"].astype(str), sort=True)
    b, bu = pd.factorize(df["broker"], sort=True)
    p, pu = pd.factorize(df["period"], sort=True)
    w, wu = pd.factorize(df["window"], sort=True)
    obs, _, nulls, pval = permutation_test(
        grp.astype(np.int64), b.astype(np.int64), p.astype(np.int64),
        w.astype(np.int64), df["pnl"].to_numpy(float),
        df["gross"].to_numpy(float), len(bu), len(pu), len(wu),
        4, 1e9, draws=60, seed=12)
    assert pval < 0.05 and obs > np.nanmax(nulls)


def test_the_null_is_centred_near_zero_on_noise():
    """It is the null's CENTRE that makes a p-value meaningful. A null that
    sits away from zero is measuring something the shuffle failed to destroy."""
    rng = np.random.default_rng(5)
    df = frame(12, 4, 60, rng)
    grp, _ = pd.factorize(df["ticker"].astype(str) + "|"
                          + df["window"].astype(str), sort=True)
    b, bu = pd.factorize(df["broker"], sort=True)
    p, pu = pd.factorize(df["period"], sort=True)
    w, wu = pd.factorize(df["window"], sort=True)
    _, _, nulls, _ = permutation_test(
        grp.astype(np.int64), b.astype(np.int64), p.astype(np.int64),
        w.astype(np.int64), df["pnl"].to_numpy(float),
        df["gross"].to_numpy(float), len(bu), len(pu), len(wu),
        4, 1e9, draws=120, seed=13)
    assert abs(np.nanmean(nulls)) < 0.12, np.nanmean(nulls)


# --------------------------------------------------------------------------
# the shuffle
# --------------------------------------------------------------------------
def test_the_shuffle_preserves_every_group_exactly():
    rng = np.random.default_rng(6)
    g = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    v = np.arange(len(g))
    s = shuffle_within(g, v, rng)
    for k in np.unique(g):
        assert sorted(v[g == k]) == sorted(s[g == k])


def test_the_shuffle_actually_moves_labels():
    rng = np.random.default_rng(6)
    g = np.zeros(200, dtype=int)
    v = np.arange(200)
    assert (shuffle_within(g, v, rng) != v).mean() > 0.5


def test_the_shuffle_never_crosses_a_group_boundary():
    """The point of shuffling within ticker-window is that the window's total
    flow survives. If a label crossed groups it would not."""
    rng = np.random.default_rng(8)
    g = np.repeat(np.arange(20), 5)
    v = np.repeat(np.arange(20), 5) * 100 + np.tile(np.arange(5), 20)
    s = shuffle_within(g, v, rng)
    assert np.array_equal(s // 100, g), "a value landed in another group"


# --------------------------------------------------------------------------
# the correlation itself
# --------------------------------------------------------------------------
def test_spearman_is_rank_based_not_level_based():
    a = np.array([1.0, 2, 3, 4, 5, 6])
    assert spearman(a, np.exp(a)) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)


def test_spearman_refuses_too_few_points():
    a = np.arange(MIN_BROKERS - 1, dtype=float)
    assert np.isnan(spearman(a, a))


def test_ties_do_not_manufacture_correlation():
    a = np.ones(10)
    assert np.isnan(spearman(a, np.arange(10, dtype=float)))


def test_adjacent_corr_skips_pairs_that_cannot_be_measured():
    M = np.full((10, 3), np.nan)
    M[:, 0] = np.arange(10)
    M[:, 1] = np.arange(10)
    r, pairs = adjacent_corr(M)
    assert len(pairs) == 1 and r == pytest.approx(1.0)


# --------------------------------------------------------------------------
# the Track B measure itself
# --------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))


def test_timing_pnl_is_positive_when_net_long_and_price_rises():
    import persistence as P
    D = pd.DataFrame({
        "broker": ["A"], "ticker": ["T"],
        "window_end": [pd.Timestamp("2025-01-14")],
        "buy_lot": [100.0], "sell_lot": [0.0],
        "buy_val": [1e9], "sell_val": [0.0]})
    out = P.timing_pnl(D, {("T", pd.Timestamp("2025-01-14")): 50.0})
    assert out["timing_pnl"].iloc[0] == pytest.approx(100 * 100 * 50.0)
    assert not out["two_sided"].iloc[0], "buy-only is censored, not two-sided"


def test_timing_pnl_is_negative_when_net_long_and_price_falls():
    import persistence as P
    D = pd.DataFrame({
        "broker": ["A"], "ticker": ["T"],
        "window_end": [pd.Timestamp("2025-01-14")],
        "buy_lot": [100.0], "sell_lot": [40.0],
        "buy_val": [1e9], "sell_val": [4e8]})
    out = P.timing_pnl(D, {("T", pd.Timestamp("2025-01-14")): -20.0})
    assert out["timing_pnl"].iloc[0] == pytest.approx(60 * 100 * -20.0)
    assert out["two_sided"].iloc[0]
    assert out["gross_value"].iloc[0] == pytest.approx(1.4e9)


def test_a_window_with_no_next_window_is_dropped_not_zeroed():
    """A zero would enter the margin as a real observation of no skill."""
    import persistence as P
    D = pd.DataFrame({
        "broker": ["A"], "ticker": ["T"],
        "window_end": [pd.Timestamp("2025-01-14")],
        "buy_lot": [100.0], "sell_lot": [0.0],
        "buy_val": [1e9], "sell_val": [0.0]})
    assert P.timing_pnl(D, {}).empty


def test_forward_moves_refuses_to_stretch_across_a_missing_window(monkeypatch):
    """The store carries sparse older probes where consecutive windows sit
    months apart. Pairing those would silently turn 'the next fortnight' into
    'the next quarter' and inflate every move."""
    import persistence as P
    idx = pd.date_range("2025-01-01", "2025-12-31", freq="B")
    px = pd.DataFrame({"close": np.linspace(100, 200, len(idx))}, index=idx)
    monkeypatch.setattr(P, "load_prices", lambda t, s: px)
    D = pd.DataFrame({
        "ticker": "T",
        "window_end": pd.to_datetime(
            ["2025-01-14", "2025-01-28", "2025-06-30"])})
    fwd = P.forward_moves(D, {})
    assert (("T", pd.Timestamp("2025-01-14"))) in fwd, "14 days apart pairs"
    assert (("T", pd.Timestamp("2025-01-28"))) not in fwd, \
        "a 153-day gap must not be treated as the next fortnight"


def test_forward_moves_never_looks_past_the_next_window(monkeypatch):
    """No look-ahead: the payoff ends at the NEXT window's close, not later."""
    import persistence as P
    idx = pd.date_range("2025-01-01", "2025-03-31", freq="B")
    c = pd.Series(100.0, index=idx)
    c.loc[idx[idx <= pd.Timestamp("2025-01-28")][-1]] = 110.0
    c.loc[idx[-1]] = 999.0                       # far future, must be ignored
    monkeypatch.setattr(P, "load_prices", lambda t, s: pd.DataFrame({"close": c}))
    D = pd.DataFrame({"ticker": "T", "window_end": pd.to_datetime(
        ["2025-01-14", "2025-01-28"])})
    fwd = P.forward_moves(D, {})
    assert fwd[("T", pd.Timestamp("2025-01-14"))] == pytest.approx(10.0)
