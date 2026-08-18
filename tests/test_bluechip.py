"""Tests for the point-in-time large-cap universe.

The whole blue-chip result rests on one claim: membership on any date was
knowable that morning. If it is not, the numbers are the survivorship bias they
were built to avoid, only better hidden. So the tests are:

  * truncation - removing the future must not change membership on any past
    date (the same test the reversal filter gets, for the same reason);
  * a name enters when it becomes liquid and leaves when it stops, rather than
    being present because of what it became;
  * the equal-weight benchmark holds only members, values what it holds, and
    caps returns at the auto-rejection band;
  * the deliberately-biased fixed list is exactly the config's list, since the
    measured size of the bias depends on it being what it claims to be.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from bluechip import equal_weight, pit_universe            # noqa: E402
from optimize_consistent import CAP                        # noqa: E402


def panel(turnovers: dict, n: int = 400, vols: dict | None = None) -> dict:
    """A minimal W-shaped dict: per-name turnover paths and optional volatility.

    Prices wiggle deterministically so the volatility screen has something to
    rank; ``vols`` sets the amplitude per name.
    """
    idx = pd.bdate_range("2005-01-01", periods=n)
    cols = list(turnovers)
    tv = pd.DataFrame({c: np.asarray(v, dtype=float) for c, v in turnovers.items()},
                      index=idx)
    wob = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    px = pd.DataFrame({c: 1000.0 * (1.0 + (vols or {}).get(c, 0.01) * wob)
                       for c in cols}, index=idx)
    return {"tv": tv, "mark": px, "close": px,
            "fac": pd.DataFrame(1.0, index=idx, columns=cols)}


# every fixture below is short, so the three-year listing screen is relaxed to
# something the fixture can satisfy; it is tested separately and explicitly.
SHORT = dict(min_history=50, max_vol_pct=1.0)


def test_universe_respects_size():
    n = 400
    W = panel({f"A{i}": np.full(n, (10 - i) * 1e10) for i in range(6)}, n)
    m = pit_universe(W, size=3, **SHORT)
    # after the 100-bar warmup exactly three names are members, and they are the
    # three with the most turnover
    assert m[200].sum() == 3
    assert m[200][:3].tolist() == [1, 1, 1]


def test_universe_applies_the_liquidity_floor():
    n = 400
    W = panel({"BIG": np.full(n, 1e11), "TINY": np.full(n, 1e6)}, n)
    m = pit_universe(W, size=10, min_turnover=5e9, **SHORT)
    assert m[200].tolist() == [1, 0]        # size would admit both; the floor does not


def test_universe_is_empty_during_warmup():
    n = 400
    W = panel({"A": np.full(n, 1e11)}, n)
    m = pit_universe(W, size=5, **SHORT)
    assert m[:99].sum() == 0                # 100-bar minimum before any membership


def test_a_name_enters_when_it_becomes_liquid():
    """Membership must follow the data, not the destination."""
    n = 600
    late = np.concatenate([np.full(300, 1e6), np.full(300, 1e12)])
    W = panel({"EARLY": np.full(n, 1e11), "LATE": late}, n)
    m = pit_universe(W, size=5, **SHORT)
    assert m[250][1] == 0                   # not yet liquid
    assert m[550][1] == 1                   # liquid now, and only now


@pytest.mark.parametrize("cut", [150, 300, 500])
def test_universe_has_no_look_ahead(cut):
    rng = np.random.default_rng(17)
    n = 600
    W = panel({f"N{i}": np.abs(rng.lognormal(23, 1.5, n)) for i in range(8)}, n)
    full = pit_universe(W, size=3, **SHORT)
    part = pit_universe({k: v.iloc[:cut] for k, v in W.items()}, size=3, **SHORT)
    assert np.array_equal(full[:cut], part)


def test_equal_weight_tracks_a_single_member():
    idx = pd.bdate_range("2015-01-01", periods=50)
    px = pd.DataFrame({"A": 1000.0 * np.cumprod(np.full(50, 1.01))}, index=idx)
    W = {"mark": px, "close": px, "tv": px,
         "fac": pd.DataFrame(1.0, index=idx, columns=["A"])}
    eq = equal_weight(W, np.ones((50, 1), dtype=np.int8))
    # membership at bar i-1 governs the return into bar i, so all 49 returns
    # are collected and none is earned on a day it was not yet a member
    assert eq.iloc[-1] == pytest.approx(1.01 ** 49)


def test_equal_weight_holds_nothing_when_nobody_qualifies():
    idx = pd.bdate_range("2015-01-01", periods=30)
    px = pd.DataFrame({"A": 1000.0 * np.cumprod(np.full(30, 1.05))}, index=idx)
    W = {"mark": px, "close": px, "tv": px,
         "fac": pd.DataFrame(1.0, index=idx, columns=["A"])}
    eq = equal_weight(W, np.zeros((30, 1), dtype=np.int8))
    assert eq.iloc[-1] == pytest.approx(1.0)      # flat: it owned nothing


def test_equal_weight_caps_at_the_rejection_band():
    """One impossible print must not compound into the record."""
    idx = pd.bdate_range("2015-01-01", periods=10)
    v = np.full(10, 1000.0)
    v[5] = 1000000.0                              # a 1000x print
    px = pd.DataFrame({"A": v}, index=idx)
    W = {"mark": px, "close": px, "tv": px,
         "fac": pd.DataFrame(1.0, index=idx, columns=["A"])}
    eq = equal_weight(W, np.ones((10, 1), dtype=np.int8))
    step = eq.pct_change().dropna()
    assert step.max() <= CAP + 1e-9


def test_equal_weight_collects_dividends():
    idx = pd.bdate_range("2015-01-01", periods=10)
    px = pd.DataFrame({"A": np.full(10, 1000.0)}, index=idx)
    fac = pd.DataFrame(1.0, index=idx, columns=["A"])
    fac.iloc[5, 0] = 1.05                         # a 5% distribution
    W = {"mark": px, "close": px, "tv": px, "fac": fac}
    eq = equal_weight(W, np.ones((10, 1), dtype=np.int8))
    assert eq.iloc[-1] == pytest.approx(1.05)     # flat price, dividend kept


def test_fixed_universe_is_the_config_list():
    from bluechip import fixed_universe
    from idxbot.config import load_config
    idx = pd.bdate_range("2015-01-01", periods=5)
    cfg = load_config()
    names = set(cfg.universe("bluechip")) | set(cfg.universe("lq45"))
    cols = sorted(names | {"NOTABLUECHIP_XYZ"})
    px = pd.DataFrame(1.0, index=idx, columns=cols)
    m = fixed_universe({"mark": px})
    assert m.shape == (5, len(cols))
    assert m[0].sum() == len(names)
    assert m[0][cols.index("NOTABLUECHIP_XYZ")] == 0
    # and it is constant through time, which is exactly the bias being measured
    assert np.array_equal(m[0], m[-1])


def test_universe_excludes_the_freshly_listed():
    """Three years of trading behind it, or it is not a blue chip yet."""
    n = 900
    fresh = np.concatenate([np.full(300, np.nan), np.full(600, 1e12)])
    W = panel({"OLD": np.full(n, 1e11), "NEW": np.full(n, 1e12)}, n)
    W["tv"]["NEW"] = fresh
    W["close"] = W["close"].copy()
    W["close"].iloc[:300, W["close"].columns.get_loc("NEW")] = np.nan
    m = pit_universe(W, size=5, min_history=750, max_vol_pct=1.0)
    # NEW out-turns OLD by 10x but has only 600 sessions behind it
    assert m[850][W["close"].columns.get_loc("NEW")] == 0
    assert m[850][W["close"].columns.get_loc("OLD")] == 1


def test_universe_excludes_the_wild_half():
    """The line that separates a large cap from a penny stock having a year."""
    n = 900
    W = panel({"CALM1": np.full(n, 9e11), "CALM2": np.full(n, 8e11),
               "WILD1": np.full(n, 1e12), "WILD2": np.full(n, 1e12)},
              n, vols={"CALM1": 0.001, "CALM2": 0.001,
                       "WILD1": 0.25, "WILD2": 0.25})
    m = pit_universe(W, size=4, min_history=50, max_vol_pct=0.5)
    cols = list(W["close"].columns)
    # the wild pair turn over MORE and are still excluded
    assert m[800][cols.index("WILD1")] == 0
    assert m[800][cols.index("WILD2")] == 0
    assert m[800][cols.index("CALM1")] == 1
    assert m[800][cols.index("CALM2")] == 1


# --------------------------------------------------------------------------- #
# sizing the live book
# --------------------------------------------------------------------------- #
def _book(prices, turnover=1e12):
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(len(prices))],
        "price": [float(p) for p in prices],
        "turnover": [turnover] * len(prices),
        "mom250": list(np.linspace(0.9, 0.1, len(prices))),
        "dd250": list(np.linspace(-0.05, -0.5, len(prices))),
        "lowvol": list(np.linspace(-0.01, -0.05, len(prices))),
        "vol_250": list(np.linspace(0.01, 0.05, len(prices))),
    })


def test_sizing_drops_what_one_lot_cannot_buy():
    """A name is dropped, not shown with zero lots, and its share is spread."""
    from bluechip_picks import select_and_size
    picks, dropped = select_and_size(_book([100, 200, 50_000]), 1_000_000,
                                     "mom250", 3)
    assert dropped == ["T2"]
    assert set(picks["ticker"]) == {"T0", "T1"}
    assert (picks["lots"] > 0).all()


def test_sizing_never_exceeds_the_capital():
    from bluechip_picks import select_and_size
    picks, _ = select_and_size(_book([100, 250, 1_000, 3_300]), 5_000_000,
                               "mom250", 4)
    assert picks["cost"].sum() <= 5_000_000 + 1e-6


def test_sizing_respects_the_turnover_cap():
    """A thin name is capped at a tenth of its daily turnover, and flagged."""
    from bluechip_picks import select_and_size
    df = _book([100, 100])
    df.loc[1, "turnover"] = 1e6          # Rp1m/day: 10% of it is Rp100,000
    picks, _ = select_and_size(df, 10_000_000, "mom250", 2)
    thin = picks[picks["ticker"] == "T1"].iloc[0]
    assert thin["capped"]
    assert thin["cost"] <= 0.10 * 1e6


def test_sizing_ranks_by_the_requested_signal():
    from bluechip_picks import select_and_size
    df = _book([100] * 5)
    top, _ = select_and_size(df, 10_000_000, "mom250", 2)
    assert top["ticker"].tolist() == ["T0", "T1"]          # highest momentum
    beaten, _ = select_and_size(df, 10_000_000, "dd250", 2)
    assert beaten["ticker"].tolist() == ["T4", "T3"]       # furthest below the high
