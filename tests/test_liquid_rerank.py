"""Tests for the liquid-tercile re-rank (H21 C3d(b)).

The whole point of this script is that C3d's tercile split produced baskets too
thin to read, so the guards that keep THIS version's baskets full are what the
result rests on, and they are pinned here.
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

from liquid_rerank import TERCILE, liquid_universe        # noqa: E402


def _panel(n=90, day="2015-06-01"):
    """A cohort cross-section with a known liquidity ordering."""
    d = pd.Timestamp(day)
    return pd.DataFrame({
        "date": [d] * n + [d + pd.Timedelta(days=1)] * n,
        "ticker": [f"T{i:03d}" for i in range(n)] * 2,
        "log_turnover": list(np.linspace(10.0, 30.0, n)) * 2})


def _M(n=90):
    return pd.DataFrame({"ticker": [f"T{i:03d}" for i in range(n)],
                         "p2": np.linspace(0.9, 0.1, n)})


def test_it_returns_about_the_top_third_by_turnover():
    got = liquid_universe(_panel(), pd.Timestamp("2015-06-01"), _M())
    assert 25 <= len(got) <= 35
    #  and it is the HIGH end: T089 has the largest turnover
    assert "T089" in got and "T000" not in got


def test_it_never_returns_a_name_absent_from_the_ranking():
    M = _M(90).iloc[:40]                       # only the least liquid 40
    got = liquid_universe(_panel(), pd.Timestamp("2015-06-01"), M)
    assert set(got) <= set(M["ticker"])


def test_it_declines_on_too_thin_a_cross_section():
    """THE GUARD THAT MATTERS. C3d's liquid cell scored 54 of 212 cohorts at a
    median of four names and produced the largest effect in the table. A cohort
    that cannot fill a basket must be skipped, not scored."""
    assert liquid_universe(_panel(n=20), pd.Timestamp("2015-06-01"),
                           _M(20)) == []


def test_it_reads_liquidity_only_on_the_cohort_date():
    """A5: never use data stamped after the decision bar. Rewriting the NEXT
    day's turnover must not move the answer."""
    P = _panel()
    a = liquid_universe(P, pd.Timestamp("2015-06-01"), _M())
    P.loc[P["date"] > pd.Timestamp("2015-06-01"), "log_turnover"] = \
        np.linspace(30.0, 10.0, 90)            # exactly reversed
    b = liquid_universe(P, pd.Timestamp("2015-06-01"), _M())
    assert a == b


def test_it_skips_names_with_no_liquidity_reading():
    P = _panel()
    P.loc[P["ticker"].isin([f"T{i:03d}" for i in range(80, 90)]),
          "log_turnover"] = np.nan
    got = liquid_universe(P, pd.Timestamp("2015-06-01"), _M())
    assert not any(t in got for t in [f"T{i:03d}" for i in range(80, 90)])


def test_the_tercile_constant_selects_the_liquid_end():
    """A silent flip of TERCILE to 0 would test the opposite claim and still
    print a plausible table."""
    assert TERCILE == 2
