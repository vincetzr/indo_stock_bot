"""H60 — the regime engine (§7 / CLAUDE.md §10).

THE TEST THIS FILE EXISTS FOR is `test_the_null_rotates_and_does_not_shuffle`.
Regimes are long contiguous runs. A day-level shuffle of the regime label
destroys that structure and manufactures a null of tiny, evenly-mixed
pseudo-regimes whose edge spread is far too small — so every z comes back
inflated and the study finds a regime effect that is not there. A circular
rotation preserves every run length and every transition and breaks only the
alignment with the market, which is the thing being tested. A17 and A34 record
the same class of error from two directions.

The second is `test_episodes_counts_runs_not_days`. G3 turns entirely on it:
6,000 days of two regimes is not 6,000 observations, it is however many times
the market actually switched, and getting that wrong makes every per-regime
statistic look thirty times better supported than it is.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import regime                                                     # noqa: E402
from regime import (_drop_largest_episode, _episode_table,        # noqa: E402
                    edge_by_regime, episodes, gmm_answer,
                    hdbscan_answer, regime_null, screen_frame,
                    standardise)

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, os.pardir, "scripts", "regime.py")


def _screen(n_dates=800, n_names=30, seed=0, effect=0.0, hot=None,
            persist=0.0):
    """A synthetic screen frame.

    `effect` lifts the NON-screen names inside the `hot` date window, which is
    what a momentum crash looks like. `persist` adds a slow common component to
    the SCREEN-minus-rest gap, which is what makes the rotation-versus-shuffle
    distinction visible: with iid noise the two nulls are identical, because
    contiguity only matters when the thing being blocked is autocorrelated.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_dates)
    #  A slow AR(1) wedge between screen and rest — the real panel has one,
    #  because 63-session forward returns on consecutive bars overlap by 62.
    wedge = np.zeros(n_dates)
    for t in range(1, n_dates):
        wedge[t] = 0.995 * wedge[t - 1] + rng.normal(0, persist)
    rows = []
    for i in range(n_names):
        sc = (i % 5 == 0)
        lf = rng.normal(0.0, 0.10, n_dates) + (wedge if sc else 0.0)
        if effect and hot is not None and not sc:
            m = (dates >= hot[0]) & (dates <= hot[1])
            lf = lf + m * effect
        rows.append(pd.DataFrame({"date": dates, "ticker": f"T{i:03d}",
                                  "lf": lf, "screen": sc}))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------- the unit of counting

def test_episodes_counts_runs_not_days():
    assert episodes(np.array([0, 0, 0, 1, 1, 0])) == 3
    assert episodes(np.array([0] * 5000)) == 1
    assert episodes(np.array([])) == 0


def test_episode_table_gives_the_dates_of_each_run():
    idx = pd.bdate_range("2020-01-01", periods=10)
    lab = pd.Series([0, 0, 0, 1, 1, 1, 1, 0, 0, 0], index=idx)
    t = _episode_table(lab)
    assert list(t["regime"]) == [0, 1, 0]
    assert list(t["days"]) == [3, 4, 3]
    assert t["start"].iloc[1] == idx[3] and t["end"].iloc[1] == idx[6]


# ------------------------------------------------------------------ the null

def test_the_null_rotates_and_does_not_shuffle():
    """A rotation must preserve run structure; a shuffle destroys it.

    THE SYNTHETIC HAS TO BE AUTOCORRELATED OR THE TEST PROVES NOTHING. A first
    version used iid noise, and the two nulls came out the same size — as they
    must, because contiguity only matters when the thing being blocked is
    persistent. The real panel is: 63-session forward returns on consecutive
    bars overlap by 62. `persist` puts that back, and only then does a
    day-level shuffle visibly collapse the spread.
    """
    idx = pd.bdate_range("2010-01-04", periods=800)
    lab = pd.Series((np.arange(800) // 100) % 2, index=idx)
    S = _screen(n_dates=800, persist=0.01)
    mu_rot, sd_rot, n_rot = regime_null(S, lab, draws=40, seed=1)
    assert n_rot > 20 and np.isfinite(sd_rot)

    rng = np.random.default_rng(0)
    shuffled = []
    for _ in range(40):
        sh = pd.Series(rng.permutation(lab.to_numpy()), index=idx)
        e = edge_by_regime(S, sh)["edge"].dropna()
        if len(e) >= 2:
            shuffled.append(float(e.max() - e.min()))
    assert shuffled
    assert mu_rot > 1.5 * float(np.mean(shuffled)), (
        f"the rotation null ({mu_rot:.4f}) is not materially wider than a "
        f"day-level shuffle ({np.mean(shuffled):.4f}) — a shuffle produces "
        "evenly-mixed pseudo-regimes and inflates every z")


def test_the_null_preserves_every_run_length():
    idx = pd.bdate_range("2010-01-04", periods=300)
    v = (np.arange(300) // 37) % 3
    lab = pd.Series(v, index=idx)
    rot = pd.Series(np.roll(v, 91), index=idx)
    #  A rotation changes at most one run (the one straddling the wrap point).
    assert abs(episodes(rot.to_numpy()) - episodes(v)) <= 1


def test_the_null_finds_a_planted_regime_effect():
    """POSITIVE CONTROL. A null that cannot detect an effect that IS there
    proves nothing by failing to detect one — A26's sine wave, A27's planted
    Fibonacci bump, A36's Q0 martingale."""
    idx = pd.bdate_range("2010-01-04", periods=800)
    hot = (idx[300], idx[420])
    lab = pd.Series(((idx >= hot[0]) & (idx <= hot[1])).astype(int), index=idx)
    S = _screen(n_dates=800, effect=0.40, hot=hot, seed=7)
    t = edge_by_regime(S, lab)
    e = t["edge"].dropna()
    spread = float(e.max() - e.min())
    mu, sd, _ = regime_null(S, lab, draws=60, seed=2)
    z = (spread - mu) / sd
    assert z > 3.0, (spread, mu, sd, z)


def test_edge_by_regime_refuses_a_degenerate_cell():
    idx = pd.bdate_range("2010-01-04", periods=800)
    lab = pd.Series(np.r_[np.zeros(797), np.ones(3)].astype(int), index=idx)
    t = edge_by_regime(_screen(n_dates=800), lab)
    tiny = t[t["regime"] == 1]
    assert len(tiny) == 1 and not np.isfinite(tiny["edge"].iloc[0])


# ------------------------------------------------------- the drop-one check

def test_dropping_the_largest_episode_removes_a_one_event_effect():
    """A8 introduced the drop-largest check because H11's headline was carried
    by one thin year. A regime effect made of one crisis must not survive it."""
    idx = pd.bdate_range("2010-01-04", periods=800)
    hot = (idx[300], idx[420])
    lab = pd.Series(((idx >= hot[0]) & (idx <= hot[1])).astype(int), index=idx)
    S = _screen(n_dates=800, effect=0.40, hot=hot, seed=8)
    t = edge_by_regime(S, lab)
    runs = _episode_table(lab)
    before = float(t["edge"].dropna().max() - t["edge"].dropna().min())
    got = _drop_largest_episode(S, lab, runs, t)
    assert got is not None
    assert not np.isfinite(got["spread"]) or got["spread"] < before


# --------------------------------------------------------------- the methods

def test_hdbscan_can_return_no_clusters():
    """A10's rule: a method that MUST return k cannot tell you there are none.
    On uniform noise with a large minimum size, HDBSCAN must be able to say so."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6))
    h = hdbscan_answer(X, 200)
    assert h["k"] == 0 or h["noise"] > 0.5


def test_hdbscan_finds_clusters_that_are_really_there():
    """POSITIVE CONTROL for the clustering arm."""
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(-6, 0.3, (300, 4)),
                   rng.normal(+6, 0.3, (300, 4))])
    h = hdbscan_answer(X, 50)
    assert h["k"] >= 2 and h["noise"] < 0.2


def test_gmm_always_returns_k_which_is_the_point_of_the_contrast():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 5))
    for k in (2, 3, 4):
        assert gmm_answer(X, k)["k"] == k
        assert len(set(gmm_answer(X, k)["labels"])) <= k


def test_standardise_leaves_a_constant_column_alone_rather_than_dividing_by_zero():
    X = np.c_[np.ones(50), np.arange(50, dtype=float)]
    Z = standardise(pd.DataFrame(X))
    assert np.all(np.isfinite(Z))


# ------------------------------------------------------------- the discipline

def test_rates_are_differenced_and_prices_are_not():
    """A13: ^TNX fell 0.93 -> 0.50 in March 2020, a real 43 bp move and a
    spurious -46% 'return'."""
    assert "ust10" in regime.RATES
    assert not (regime.RATES & {"dxy", "spx", "brent", "copper", "usdidr"})


def test_every_k_is_reported_not_only_the_one_that_fired():
    src = open(SRC).read()
    assert "EVERY k is reported" in src
    assert "range(2, a.kmax + 1)" in src, (
        "the sweep must cover every k, or quoting the loudest is a search "
        "whose size is not disclosed")


def test_no_macro_variable_is_used_to_predict_a_return():
    doc = regime.__doc__ or ""
    assert "DELIBERATELY EXCLUDED" in doc
    assert "conditioning" in doc


def test_the_registration_survives_the_answer():
    doc = regime.__doc__ or ""
    for k in ("G1", "G2", "G3"):
        assert k in doc
    assert "PREDICTED NULL" in doc
