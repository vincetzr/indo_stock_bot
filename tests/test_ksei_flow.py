"""H64's machinery must find a planted effect and must not invent one.

The real run is only readable if the instrument has been shown to return the
known answer on a known case (A26's sine wave, A27's planted Fibonacci bump,
A36's Q0 martingale). These are those checks for the KSEI flow test.
"""

from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, os.pardir, "src"))
sys.path.insert(0, os.path.join(HERE, os.pardir, "scripts"))

import ksei_flow as kf                                          # noqa: E402


def _month(n=200, beta=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = rng.normal(size=n) + X @ np.array([0.5, -0.2, 0.1])
    r = beta * (y - X @ np.array([0.5, -0.2, 0.1])) + rng.normal(size=n)
    return kf.Month(y, r, X, rng.normal(size=n), np.full(n, 0.002))


def test_month_ic_is_the_spearman_of_the_residual():
    M = _month(beta=0.3)
    e = M.resid()
    assert M.ic(e) == pytest.approx(spearmanr(e, M.r)[0], abs=1e-12)
    # the residual is orthogonal to the controls and the intercept
    assert np.abs(M.Q.T @ e).max() < 1e-9


def test_the_vectorised_null_matches_a_slow_null():
    M = _month(beta=0.0, seed=3)
    rng = np.random.default_rng(11)
    fast = M.null_ics(50, rng)
    rng = np.random.default_rng(11)
    perm = np.argsort(rng.random((50, len(M.y))), axis=1)
    slow = np.array([spearmanr(M.resid(M.y[p]), M.r)[0] for p in perm])
    assert np.allclose(fast, slow, atol=1e-12)


def test_a_planted_effect_is_found_and_a_null_is_not():
    """The positive and the negative control, through null_test itself."""
    planted = [(None, None, _month(beta=-0.25, seed=s)) for s in range(24)]
    A = kf.analyse([(pd.Timestamp("2020-01-31") + pd.offsets.MonthEnd(i),
                     pd.Timestamp("2020-02-05") + pd.DateOffset(months=i), m)
                    for i, (_, _, m) in enumerate(planted)], 21, -1)
    null = kf.null_test(planted, 400, 7)
    assert A["ic_mean"] < -0.15
    assert A["ic_mean"] < np.percentile(null, 0.5)
    quiet = [(None, None, _month(beta=0.0, seed=100 + s)) for s in range(24)]
    Aq = kf.analyse([(pd.Timestamp("2020-01-31") + pd.offsets.MonthEnd(i),
                      pd.Timestamp("2020-02-05") + pd.DateOffset(months=i), m)
                     for i, (_, _, m) in enumerate(quiet)], 21, -1)
    nq = kf.null_test(quiet, 400, 7)
    lo, hi = np.percentile(nq, [0.5, 99.5])
    assert lo < Aq["ic_mean"] < hi
    assert abs(nq.mean()) < 0.01


def test_the_null_is_centred_on_zero_not_on_the_signal():
    """A3's failure: a null that certifies anything is worse than none."""
    planted = [(None, None, _month(beta=-0.5, seed=s)) for s in range(12)]
    null = kf.null_test(planted, 300, 1)
    assert abs(null.mean()) < 0.01


def _sessions(start="2024-01-01", n=200):
    return pd.bdate_range(start, periods=n)


def test_attach_prices_is_within_ticker_and_refuses_a_missing_bar():
    s = _sessions()
    rows = []
    for t, drop in (("AAAA", None), ("BBBB", s[10])):
        for i, d in enumerate(s):
            if d == drop:
                continue
            rows.append({"date": d, "ticker": t, "close": 100.0,
                         "adj_close": 100.0 * (1.01 if t == "AAAA" else 1.0) ** i,
                         "tradeable": True, "mom12_1": 0.0, "rev1": 0.0,
                         "log_turnover": 25.0, "vol60": 0.02})
    D = pd.DataFrame(rows)
    K = pd.DataFrame({"ticker": ["AAAA", "BBBB"],
                      "decision": [s[10], s[10]]})
    X = kf.attach_prices(K, D)
    assert list(X["ticker"]) == ["AAAA"]           # BBBB had no bar that day
    assert X["fwd21"].iloc[0] == pytest.approx(1.01 ** 21 - 1)
    assert pd.Timestamp(X["exit21"].iloc[0]) == s[31]


def test_a_backfilled_previous_file_makes_the_signal_stale():
    """The 2020-12 file was created in October 2021: its change is not usable."""
    sess = pd.bdate_range("2020-10-01", "2021-12-31")
    P = []
    for me, created in ((pd.Timestamp("2020-11-30"), dt.datetime(2020, 12, 1, 6)),
                        (pd.Timestamp("2020-12-30"), dt.datetime(2021, 10, 19, 17)),
                        (pd.Timestamp("2021-01-29"), dt.datetime(2021, 2, 1, 6)),
                        (pd.Timestamp("2021-02-26"), dt.datetime(2021, 3, 1, 6))):
        row = {"month_end": me, "ticker": "AAAA", "sec_num": 1000, "price": 100,
               "local_total": 800, "foreign_total": 200,
               "created": pd.Timestamp(created)}
        row.update({c: 0 for c in kf.ksei.LOCAL})
        row.update({"local_ID": 300, "local_CP": 500})
        P.append(row)
    K = kf.ksei_frame(pd.DataFrame(P), sess, lag=1).set_index("month_end")
    assert K.loc["2020-12-30", "stale_days"] > kf.STALE_DAYS
    assert K.loc["2021-01-29", "stale_days"] > kf.STALE_DAYS
    assert K.loc["2021-02-26", "stale_days"] <= kf.STALE_DAYS
    assert K.loc["2021-02-26", "decision"] == pd.Timestamp("2021-03-02")


def test_the_long_leg_follows_the_prediction():
    """pred < 0 holds the bottom quintile (what retail SOLD), pred > 0 the top."""
    n = 100
    y = np.arange(n, dtype=float)
    r = -y / 1000.0
    M = kf.Month(y, r, np.zeros((n, 0)), np.arange(n, dtype=float),
                 np.zeros(n))
    neg = kf.analyse([(pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-03"),
                       M)] * 6, 21, -1)
    pos = kf.analyse([(pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-03"),
                       M)] * 6, 21, +1)
    assert neg["gross_ex"] > 0 > pos["gross_ex"]
    assert neg["net_ex"] == pytest.approx(neg["gross_ex"] - kf.FEE)


# --------------------------------------------------------------------------- #
# the 2026-11-16 review's fixes, each with the case that exposed it
# --------------------------------------------------------------------------- #

def test_a_degenerate_column_placed_first_does_not_break_the_projection():
    """Unpivoted QR left max |A'e| = 0.049 on exactly this case."""
    rng = np.random.default_rng(5)
    n = 50
    c1, c3 = rng.normal(size=n), rng.normal(size=n)
    X = np.column_stack([c1, np.zeros(n), c3])
    y = 2 * c3 + rng.normal(size=n)
    M = kf.Month(y, rng.normal(size=n), X, np.zeros(n), np.zeros(n))
    A = np.column_stack([np.ones(n), X])
    assert np.abs(A.T @ M.resid()).max() < 1e-9


def _panel_months(n_names=120, n_months=24, persistent=True, seed=0):
    """Signal and label share a persistent NAME component under the null.

    With overlapping labels and persistent flows, consecutive monthly ICs are
    correlated; a null that reshuffles every month independently cannot see
    that, and its spread is too narrow.
    """
    rng = np.random.default_rng(seed)
    y0, a0 = rng.normal(size=n_names), rng.normal(size=n_names)
    out = []
    for m in range(n_months):
        y = y0 + 0.3 * rng.normal(size=n_names) if persistent else rng.normal(size=n_names)
        r = a0 + 0.3 * rng.normal(size=n_names) if persistent else rng.normal(size=n_names)
        out.append((m, m, kf.Month(y, r, np.zeros((n_names, 0)), np.zeros(n_names),
                                   np.zeros(n_names), ids=np.arange(n_names))))
    return out


def test_the_relabelling_null_sees_dependence_the_within_month_null_cannot():
    months = _panel_months(persistent=True)
    within = kf.null_test(months, 400, 1)
    relabel = kf.relabel_test(months, 400, 1)
    assert np.std(relabel) > 3 * np.std(within)
    # and with independent months the two nulls agree on scale
    ind = _panel_months(persistent=False, seed=3)
    w2, r2 = kf.null_test(ind, 400, 2), kf.relabel_test(ind, 400, 2)
    assert 0.7 < np.std(r2) / np.std(w2) < 1.4


def test_the_relabelling_permutation_has_no_fixed_points_and_is_shared():
    """A name never keeps its own signal, and keeps the SAME donor every month."""
    n = 30
    y = np.arange(n, dtype=float)
    M1 = kf.Month(y, np.zeros(n) + np.arange(n), np.zeros((n, 0)), np.zeros(n),
                  np.zeros(n), ids=np.arange(n))
    keys = np.random.default_rng(9).random((5, n))
    order = np.argsort(keys[:, M1.ids], axis=1)
    src = np.roll(order, -1, axis=1)
    Y = np.empty((5, n))
    np.put_along_axis(Y, order, y[src], axis=1)
    assert not (Y == y).any()
    # a second month with the same names in a different row order gets the
    # same donor for every name
    perm = np.random.default_rng(1).permutation(n)
    M2 = kf.Month(y[perm], np.arange(n, dtype=float), np.zeros((n, 0)),
                  np.zeros(n), np.zeros(n), ids=perm)
    order2 = np.argsort(keys[:, M2.ids], axis=1)
    src2 = np.roll(order2, -1, axis=1)
    Y2 = np.empty((5, n))
    np.put_along_axis(Y2, order2, M2.y[src2], axis=1)
    donor1 = {i: Y[0, i] for i in range(n)}
    donor2 = {perm[j]: Y2[0, j] for j in range(n)}
    assert donor1 == donor2


def test_scoring_out_rows_keeps_them_in_the_fit():
    rng = np.random.default_rng(4)
    n = 80
    X = rng.normal(size=(n, 2))
    y = rng.standard_t(2, size=n)
    r = rng.normal(size=n)
    mask = np.zeros(n, dtype=bool)
    mask[:6] = True                                   # rows to score out
    full = kf.Month(y, r, X, np.zeros(n), np.zeros(n))
    so = kf.Month(y, r, X, np.zeros(n), np.zeros(n), score=~mask)
    e = full.resid()
    assert so.ic() == pytest.approx(spearmanr(e[~mask], r[~mask])[0], abs=1e-12)


def test_the_mechanism_pairs_only_adjacent_archive_months():
    """A filtered-out month must not turn a two-month gap into a lag of one."""
    grid = pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-28",
                             "2024-04-30"])
    X = pd.DataFrame({
        "ticker": ["AAAA", "AAAA", "AAAA"],
        "month_end": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-04-30"]),
        "consecutive": [True, True, True],      # archive-consecutive; March
        "f_retail": [0.1, 0.2, 0.4],            # was FILTERED from the frame
        "f_foreign": [0.0, 0.0, 0.0]})
    Z = kf.with_predecessor(X, grid)
    assert list(Z["prev_me"]) == [pd.NaT, grid[0], grid[2]]
    ar = kf.adjacent_pairs(Z)
    assert len(ar) == 1 and ar.iloc[0]["f_prev"] == 0.1   # Feb<-Jan only
