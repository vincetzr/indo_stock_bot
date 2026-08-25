"""Tests for the live position monitor.

The monitor's job is to turn the validated exit rules into prices. Two
properties matter more than the formatting:

  * a level must be the SAME number the backtest would have acted on, because
    two implementations of one rule is how H17b's tie-break failure happened;
  * the news column must never become load-bearing. It is printed and nothing
    is computed from it, and the degradation test below pins that a total
    news-layer failure leaves the levels untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idxbot.report import monitor as M
from idxbot.spine import exits as X


def _panels(n=120, seed=4, start="2024-01-02", ticker="AAAA"):
    rng = np.random.default_rng(seed)
    c = 1000.0 * np.cumprod(1.0 + rng.normal(0.004, 0.03, n))
    d = pd.bdate_range(start, periods=n)
    P = pd.DataFrame({"date": d, "ticker": ticker, "close": c,
                      "adj_close": c, "volume": np.full(n, 1e7),
                      "tradeable": True, "holdout": False})
    spread = 0.01 * c
    I = pd.DataFrame({"date": d, "ticker": ticker, "close": c,
                      "adj_high": c + spread, "adj_low": c - spread,
                      "ema10": pd.Series(c).ewm(span=10, adjust=False).mean(),
                      "ema20": pd.Series(c).ewm(span=20, adjust=False).mean(),
                      "ema30": pd.Series(c).ewm(span=30, adjust=False).mean(),
                      "ema50": pd.Series(c).ewm(span=50, adjust=False).mean(),
                      "atr22": np.full(n, 20.0),
                      "stoch_k": np.full(n, 85.0),
                      "stoch_d": np.full(n, 80.0),
                      "tvz20": np.zeros(n)})
    return P, I


def test_position_frame_reports_gain_peak_and_give_back():
    P, I = _panels()
    day = P["date"].iloc[-1]
    F = M.position_frame(P, I, [{"ticker": "AAAA",
                                 "entry_date": P["date"].iloc[0]}], day)
    r = F.iloc[0]
    assert r["status"] == "held"
    assert r["sessions"] == len(P)
    assert r["peak_gain"] >= r["gain"]
    assert r["give_back"] <= 0


def test_position_frame_flags_a_name_it_has_no_data_for():
    P, I = _panels()
    F = M.position_frame(P, I, [{"ticker": "ZZZZ", "entry_date": "2024-01-02"}],
                         P["date"].iloc[-1])
    assert F.iloc[0]["status"] == "no data"


def test_position_frame_flags_an_entry_in_the_future():
    P, I = _panels()
    F = M.position_frame(P, I, [{"ticker": "AAAA", "entry_date": "2099-01-01"}],
                         P["date"].iloc[-1])
    assert F.iloc[0]["status"] == "not yet held"


def test_the_trail_level_is_the_peak_less_the_drop():
    P, I = _panels()
    day = P["date"].iloc[-1]
    F = M.position_frame(P, I, [{"ticker": "AAAA",
                                 "entry_date": P["date"].iloc[0]}], day)
    r = F.iloc[0]
    L = M.levels(r, arm=0.0, trail=0.15)
    row = L[L["rule"].str.startswith("trail")].iloc[0]
    assert row["level"] == pytest.approx(r["peak_adj"] * 0.85 * r["q"])


def test_the_chandelier_level_is_the_peak_less_k_atrs():
    P, I = _panels()
    day = P["date"].iloc[-1]
    r = M.position_frame(P, I, [{"ticker": "AAAA",
                                 "entry_date": P["date"].iloc[0]}], day).iloc[0]
    L = M.levels(r, arm=0.0, chand_k=3.0)
    row = L[L["rule"].str.startswith("chandelier 3x ATR armed")].iloc[0]
    assert row["level"] == pytest.approx(
        (r["peak_adj"] - 3.0 * r["atr22"]) * r["q"])


def test_an_unarmed_rule_reports_no_level_at_all():
    """Printing a level for a rule that cannot fire is worse than printing
    nothing — it is the exact confusion H17's P(-50%) result turns on."""
    P, I = _panels(seed=1)
    P["adj_close"] = P["close"] = np.linspace(1000, 900, len(P))   # never up
    I["close"] = P["adj_close"].to_numpy()
    I["adj_high"] = P["adj_close"].to_numpy() * 1.001
    day = P["date"].iloc[-1]
    r = M.position_frame(P, I, [{"ticker": "AAAA",
                                 "entry_date": P["date"].iloc[0]}], day).iloc[0]
    L = M.levels(r, arm=0.50)
    armed = L[L["rule"].str.contains("armed +", regex=False)]
    assert not armed["active"].any()
    assert armed["level"].isna().all()
    assert all("not armed" in n for n in armed["note"])


def test_levels_are_quoted_rupiah_not_the_adjusted_basis():
    """A level you cannot type into a broker screen is not a level."""
    P, I = _panels()
    P["close"] = P["adj_close"] * 2.0                 # factor 0.5
    day = P["date"].iloc[-1]
    r = M.position_frame(P, I, [{"ticker": "AAAA",
                                 "entry_date": P["date"].iloc[0]}], day).iloc[0]
    assert r["q"] == pytest.approx(2.0)
    L = M.levels(r, arm=0.0, trail=0.15)
    row = L[L["rule"].str.startswith("trail")].iloc[0]
    assert row["level"] == pytest.approx(r["peak_adj"] * 0.85 * 2.0)


def test_replay_agrees_with_the_backtest_on_the_same_path():
    """One rule, one number. Two implementations is how H17b happened."""
    P, I = _panels()
    day = P["date"].iloc[-1]
    r = M.position_frame(P, I, [{"ticker": "AAAA",
                                 "entry_date": P["date"].iloc[0]}], day).iloc[0]
    rules = {"trail 15% armed +50%": X.catalogue()["trail 15% armed +50%"],
             "hold 252": X.catalogue()["hold 252"]}
    R = M.replay(P, I, r, rules)
    adj = P["adj_close"].to_numpy(float)
    path = adj[1:X.HORIZON + 1] / adj[0]
    for name, fn in rules.items():
        want, _ = fn(path, None)
        got = R[R["rule"] == name]["gross"].iloc[0]
        assert got == pytest.approx(want)


def test_replay_marks_a_rule_that_never_fired_as_still_holding():
    P, I = _panels()
    day = P["date"].iloc[-1]
    r = M.position_frame(P, I, [{"ticker": "AAAA",
                                 "entry_date": P["date"].iloc[0]}], day).iloc[0]
    R = M.replay(P, I, r, {"hold 252": X.catalogue()["hold 252"]})
    assert not R.iloc[0]["fired"]


def test_nearest_trigger_prefers_a_breached_rule_over_a_live_one():
    L = pd.DataFrame({"rule": ["a", "b"], "level": [10.0, 5.0],
                      "distance": [0.10, -0.30], "active": [True, True],
                      "note": ["", ""]})
    assert M.nearest_trigger(L)["rule"] == "a"


def test_nearest_trigger_picks_the_closest_live_stop_when_none_are_breached():
    L = pd.DataFrame({"rule": ["a", "b"], "level": [9.0, 5.0],
                      "distance": [-0.10, -0.50], "active": [True, True],
                      "note": ["", ""]})
    assert M.nearest_trigger(L)["rule"] == "a"


def test_nearest_trigger_ignores_rules_that_cannot_fire():
    L = pd.DataFrame({"rule": ["a", "b"], "level": [np.nan, 5.0],
                      "distance": [np.nan, -0.50], "active": [False, True],
                      "note": ["not armed", ""]})
    assert M.nearest_trigger(L)["rule"] == "b"


def test_oscillator_state_reads_the_stochastic_and_says_it_is_not_a_rule():
    P, I = _panels()
    r = M.position_frame(P, I, [{"ticker": "AAAA",
                                 "entry_date": P["date"].iloc[0]}],
                         P["date"].iloc[-1]).iloc[0]
    s = M.oscillator_state(r)
    assert "%K" in s and "overbought" in s


def test_oscillator_state_says_undefined_rather_than_inventing_a_number():
    P, I = _panels()
    I["stoch_k"] = np.nan
    I["stoch_d"] = np.nan
    r = M.position_frame(P, I, [{"ticker": "AAAA",
                                 "entry_date": P["date"].iloc[0]}],
                         P["date"].iloc[-1]).iloc[0]
    assert "undefined" in M.oscillator_state(r)


def test_a_dead_news_layer_degrades_to_no_tags_and_breaks_nothing(monkeypatch):
    """The news column is decoration by design. If the feed is down the levels
    must still print — nothing in this repo is computed from a headline."""
    import idxbot.data.news as N
    monkeypatch.setattr(N, "ticker_news",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    assert M.event_tags(["AAAA"]) == {}


def test_event_tags_keeps_only_standing_corporate_events(monkeypatch):
    import idxbot.data.news as N
    monkeypatch.setattr(N, "ticker_news", lambda *a, **k: pd.DataFrame(
        {"ticker": ["AAAA", "AAAA"],
         "tags": [["SUSPEND", "EARNINGS"], ["RIGHTS"]]}))
    got = M.event_tags(["AAAA"])
    assert got == {"AAAA": ["RIGHTS", "SUSPEND"]}      # EARNINGS is not standing
