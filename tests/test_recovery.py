"""Tests for H19's recovery-curve state panel.

The whole study is a conditional expectation, so the only things that can
invalidate it are a look-ahead in how the state is built and a null that is
too tight. Both are pinned here.

THE NULL IS THE ONE THAT ALREADY BIT. The first version permuted ROWS. A name
contributes roughly twenty near-identical bars a month — same indicator state,
overlapping forward windows — so a row shuffle destroys the label while
leaving the null far too narrow, and z-scores came back inflated (one read
-8.7). Permuting whole (ticker, month) blocks is the fix, and
`test_the_block_null_is_wider_than_a_row_null` is what makes the difference
visible rather than asserted.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))

from recovery import ARM, EDGES, LOOK, block_ci, build, conditional, curve  # noqa: E402


def _panels(n=900, seed=2, ticker="AAAA"):
    """A name that runs hard, peaks, and gives it back — so ARM is satisfied."""
    rng = np.random.default_rng(seed)
    up = np.cumprod(1.0 + rng.normal(0.004, 0.02, n // 2))
    dn = np.cumprod(1.0 + rng.normal(-0.001, 0.02, n - n // 2))
    a = 100.0 * np.concatenate([up, up[-1] * dn])
    d = pd.bdate_range("2005-01-03", periods=n)
    P = pd.DataFrame({"date": d, "ticker": ticker, "close": a, "adj_close": a,
                      "log_turnover": np.full(n, np.log(5e9)),
                      "tradeable": True, "holdout": False})
    I = pd.DataFrame({"date": d, "ticker": ticker, "close": a,
                      "ema20": pd.Series(a).ewm(span=20, adjust=False).mean(),
                      "ema50": pd.Series(a).ewm(span=50, adjust=False).mean(),
                      "atr22": np.full(n, 2.0),
                      "stoch_k": np.full(n, 60.0),
                      "stoch_d": np.full(n, 55.0),
                      "tvz20": np.zeros(n)})
    return P, I


# ==========================================================================
# no look-ahead in the state, and none in the label
# ==========================================================================
def test_the_state_at_bar_i_does_not_change_when_later_bars_are_removed():
    """dd and runup describe the past; truncating the future must not move
    them. This is the same prefix check the indicator layer uses."""
    P, I = _panels()
    k = 20
    full = build(P, I, k).set_index("date")
    cut = 700
    pre = build(P.iloc[:cut], I.iloc[:cut], k).set_index("date")
    common = full.index.intersection(pre.index)
    assert len(common) > 100
    for col in ("dd", "runup"):
        a = full.loc[common, col].to_numpy()
        b = pre.loc[common, col].to_numpy()
        ok = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[ok], b[ok]), col


def test_forward_return_is_measured_forward_not_backward():
    P, I = _panels()
    k = 20
    D = build(P, I, k).set_index("date")
    a = P.set_index("date")["adj_close"]
    for dt in D.index[:40]:
        j = a.index.get_loc(dt)
        if j + k >= len(a):
            continue
        assert D.loc[dt, "fwd"] == pytest.approx(
            a.iloc[j + k] / a.iloc[j] - 1.0)


def test_new_high_looks_only_at_bars_after_the_current_one():
    """A bar sitting AT its own running peak must not count itself."""
    n = LOOK + 60
    a = np.concatenate([np.linspace(100, 300, LOOK), np.full(60, 200.0)])
    d = pd.bdate_range("2005-01-03", periods=n)
    P = pd.DataFrame({"date": d, "ticker": "T", "close": a, "adj_close": a,
                      "log_turnover": np.full(n, np.log(5e9)),
                      "tradeable": True, "holdout": False})
    I = pd.DataFrame({"date": d, "ticker": "T", "close": a,
                      "ema20": a, "ema50": a, "atr22": np.full(n, 1.0),
                      "stoch_k": np.full(n, 50.0), "stoch_d": np.full(n, 50.0),
                      "tvz20": np.zeros(n)})
    D = build(P, I, 20)
    #   price is flat at 200 under a peak of 300, so nothing ever exceeds it
    assert not D["newhigh"].any()


def test_only_armed_bars_survive_the_filter():
    """A name that never ran cannot be in an armed state by construction."""
    n = LOOK + 100
    a = np.full(n, 100.0) + np.sin(np.arange(n) / 20.0)     # goes nowhere
    d = pd.bdate_range("2005-01-03", periods=n)
    P = pd.DataFrame({"date": d, "ticker": "T", "close": a, "adj_close": a,
                      "log_turnover": np.full(n, np.log(5e9)),
                      "tradeable": True, "holdout": False})
    I = pd.DataFrame({"date": d, "ticker": "T", "close": a,
                      "ema20": a, "ema50": a, "atr22": np.full(n, 1.0),
                      "stoch_k": np.full(n, 50.0), "stoch_d": np.full(n, 50.0),
                      "tvz20": np.zeros(n)})
    assert build(P, I, 20).empty


def test_holdout_and_illiquid_bars_are_excluded():
    P, I = _panels()
    P2 = P.copy()
    P2["holdout"] = True
    assert build(P2, I, 20).empty
    P3 = P.copy()
    P3["log_turnover"] = np.log(1e6)                        # far below the floor
    assert build(P3, I, 20).empty


# ==========================================================================
# the buckets
# ==========================================================================
def test_buckets_are_ordered_by_depth_not_lexicographically():
    """A string bucket sorted as text puts -100% between -10% and -15%, which
    silently scrambles the whole curve."""
    P, I = _panels()
    D = build(P, I, 20)
    assert D["bin"].dtype.kind in "iu"
    g = D.groupby("bin")["dd"].mean()
    assert g.is_monotonic_increasing


def test_bucket_label_matches_the_bin_it_came_from():
    P, I = _panels()
    D = build(P, I, 20)
    for b, g in D.groupby("bin"):
        lo, hi = EDGES[b], EDGES[b + 1]
        assert g["dd"].min() > lo - 1e-9
        assert g["dd"].max() <= hi + 1e-9


def test_the_state_panel_survives_a_parquet_round_trip(tmp_path):
    """An interval categorical cannot be written, and that killed a 25-minute
    run after it had finished computing."""
    P, I = _panels()
    D = build(P, I, 20)
    p = tmp_path / "state.parquet"
    D.drop(columns=["month"]).to_parquet(p, index=False)
    assert len(pd.read_parquet(p)) == len(D)


# ==========================================================================
# the intervals and the null
# ==========================================================================
def test_block_ci_widens_on_a_serially_correlated_series():
    m = pd.period_range("2005-01", periods=200, freq="M")
    rng = np.random.default_rng(4)
    x = pd.Series(rng.normal(size=200)).rolling(12).mean().fillna(0.0)
    D = pd.DataFrame({"month": m, "v": x.to_numpy()})
    wide = block_ci(D, "v", block=12)
    tight = block_ci(D, "v", block=1)
    assert (wide[1] - wide[0]) > (tight[1] - tight[0])


def test_block_ci_declines_to_answer_on_too_few_months():
    D = pd.DataFrame({"month": pd.period_range("2005-01", periods=10,
                                               freq="M"),
                      "v": np.arange(10.0)})
    assert all(np.isnan(v) for v in block_ci(D, "v", block=12))


def _cells(n_names=60, n_months=40, effect=0.0, seed=1):
    """Synthetic (ticker, month) blocks with a controllable true effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_names):
        flag = t % 2 == 0
        for mi in range(n_months):
            base = rng.normal(0, 0.05)                  # month-level common
            for _ in range(20):                         # 20 bars, near-identical
                rows.append({
                    "ticker": f"T{t:03d}",
                    "month": pd.Period("2005-01", freq="M") + mi,
                    "dd": -0.12, "bin": 2, "bucket": "-15% to -10%",
                    "fwd": base + (effect if flag else 0.0)
                    + rng.normal(0, 0.001),
                    "newhigh": bool(flag), "adj": 100.0,
                    "ema20": 90.0 if flag else 110.0,
                    "ema50": 90.0 if flag else 110.0,
                    "atr22": 5.0, "stoch_k": 60.0, "stoch_d": 55.0,
                    "tvz20": 0.0, "dd_atr": 2.4})
    return pd.DataFrame(rows)


def test_the_block_null_is_wider_than_a_row_null_on_clustered_data():
    """THE DEFECT THIS FILE EXISTS FOR. Twenty near-identical bars per name-
    month is one observation, not twenty; a row shuffle prices it as twenty."""
    D = _cells(effect=0.0)
    C = conditional(D, draws=100, min_side=100, min_share=0.05)
    assert not C.empty
    block_sd = float(C["null_sd_fwd"].iloc[0])

    # the row-level null the first version used, on the same data
    rng = np.random.default_rng(0)
    lab = (D["ema20"] < D["adj"]).to_numpy()
    fw = D["fwd"].to_numpy()
    row = []
    for _ in range(100):
        sh = rng.permutation(lab)
        row.append(fw[sh].mean() - fw[~sh].mean())
    assert block_sd > 3.0 * float(np.std(row, ddof=1))


def test_the_null_does_not_flag_a_pure_month_effect_as_an_indicator_effect():
    """With no true effect the permutation p must not be small."""
    C = conditional(_cells(effect=0.0), draws=200, min_side=100)
    assert not C.empty
    assert C["p_fwd"].min() > 0.01


def test_the_null_does_detect_a_real_effect_when_one_is_planted():
    C = conditional(_cells(effect=0.08), draws=200, min_side=100)
    assert not C.empty
    r = C[C["test"] == "above EMA20"].iloc[0]
    assert r["d_fwd"] > 0.05
    assert r["p_fwd"] < 0.05


def test_a_one_sided_cell_is_dropped_rather_than_reported():
    """share 0.2% against everything else produced the largest effect in the
    first table (-46.1%) off a handful of rows."""
    D = _cells()
    D.loc[D["ticker"] != "T000", "ema20"] = 110.0        # only one name above
    C = conditional(D, draws=50, min_side=300, min_share=0.05)
    assert "above EMA20" not in set(C["test"])


def test_curve_returns_one_row_per_populated_bucket_shallowest_first():
    P, I = _panels()
    D = build(P, I, 20)
    R = curve(D, 20)
    assert len(R) == D["bin"].nunique()
    assert R["give_back"].iloc[0].startswith("-5%")
