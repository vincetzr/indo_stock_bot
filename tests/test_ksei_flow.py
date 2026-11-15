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
               "local_total": 800, "foreign_total": 200, "local_ID": 300,
               "created": pd.Timestamp(created)}
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
