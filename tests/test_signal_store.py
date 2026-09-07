"""§38 — the signal log. Tests are about IMMUTABILITY and CAUSALITY.

With the holdout spent at H16, this store is the only mechanism in the repo that
can produce out-of-sample evidence. That makes two properties load-bearing: a
record must never be rewritten after the fact, and an outcome must never be
computed from a bar the decision could not have seen. Everything else is detail.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot import signal_store as ss                              # noqa: E402


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(ss, "EMITTED", str(tmp_path / "emitted.csv.gz"))
    monkeypatch.setattr(ss, "OUTCOMES", str(tmp_path / "outcomes.csv.gz"))


def _panel(tk="AAAA", start="2026-01-05", n=60, path="up"):
    d = pd.bdate_range(start, periods=n)
    if path == "up":
        px = np.linspace(100, 200, n)
    elif path == "down":
        px = np.linspace(100, 40, n)
    else:
        px = np.full(n, 100.0)
    return pd.DataFrame({"ticker": tk, "date": d, "adj_close": px,
                         "close": px})


# ================================================================ IMMUTABILITY
def test_the_same_signal_emitted_twice_is_recorded_once():
    """The daily job re-running is normal and must be a no-op, not a duplicate
    and not a crash."""
    rows = [{"ticker": "AAAA", "entry": 100.0, "sl": 80.0, "tp": 200.0}]
    a = ss.emit(rows, "test", "v1", "2026-01-02", 20)
    b = ss.emit(rows, "test", "v1", "2026-01-02", 20)
    assert a["written"] == 1
    assert b["written"] == 0 and b["skipped"] == 1
    assert len(ss.load_emitted()) == 1


def test_an_existing_record_is_never_overwritten():
    """A record a later run can rewrite is a draft, not evidence. Re-emitting
    the SAME id with different levels must leave the original standing."""
    ss.emit([{"ticker": "AAAA", "entry": 100.0, "sl": 80.0, "tp": 200.0}],
            "test", "v1", "2026-01-02", 20)
    ss.emit([{"ticker": "AAAA", "entry": 999.0, "sl": 1.0, "tp": 2.0}],
            "test", "v1", "2026-01-02", 20)
    e = ss.load_emitted()
    assert len(e) == 1
    assert e["entry"].iloc[0] == 100.0, "the original record was overwritten"


def test_changing_the_rule_version_makes_it_a_DIFFERENT_prediction():
    """H56 changed STOP and H56b added TP. A signal under new parameters is not
    the old signal corrected — it is a new claim and needs its own row."""
    rows = [{"ticker": "AAAA", "entry": 100.0, "sl": 80.0, "tp": 200.0}]
    ss.emit(rows, "test", "v1", "2026-01-02", 20)
    ss.emit(rows, "test", "v2", "2026-01-02", 20)
    assert len(ss.load_emitted()) == 2


def test_every_row_carries_the_code_version_and_the_asof_bar():
    """§51: a signal generated today must be explainable months later. Without
    the git SHA it cannot be re-derived; without `asof` a scan run at 19:00 is
    indistinguishable from one acting on the next day's open."""
    ss.emit([{"ticker": "AAAA", "entry": 100.0, "sl": 80.0, "tp": 200.0,
              "hi52": 0.99}], "test", "v1", "2026-01-02", 20)
    e = ss.load_emitted().iloc[0]
    assert e["code_version"] and str(e["code_version"]) != "nan"
    assert pd.Timestamp(e["asof"]) == pd.Timestamp("2026-01-02")
    assert "hi52" in e["features"], "the knowable state was not recorded"


# =================================================================== CAUSALITY
def test_the_outcome_never_uses_the_decision_bar_or_anything_before_it():
    """THE TEST THIS FILE EXISTS FOR. A signal decided on the 15:50 close cannot
    be filled before the next session. Scoring from its own bar is a
    one-session look-ahead that would make a live record look better than the
    thing it tracks, and nothing in the output would look wrong."""
    P = _panel(n=60, start="2026-01-05")
    #  Decide on a bar in the middle; everything before must be invisible.
    asof = P["date"].iloc[30]
    ss.emit([{"ticker": "AAAA", "entry": float(P["adj_close"].iloc[30]),
              "sl": None, "tp": None}], "test", "v1", asof, 5)
    o = ss.score(P).iloc[0]
    assert pd.Timestamp(o["exit_date"]) > asof
    #  And the bars it saw are exactly the 5 after asof.
    assert o["bars_seen"] == 5
    expected = float(P["adj_close"].iloc[35])
    assert o["exit_px"] == pytest.approx(expected)


def test_an_unsettled_signal_is_reported_and_never_counted():
    """Some of the sample has not finished happening. Dropping it biases the
    record toward whatever resolves fastest; counting it as a win or a loss is
    worse."""
    P = _panel(n=10)
    ss.emit([{"ticker": "AAAA", "entry": 100.0, "sl": None, "tp": None}],
            "test", "v1", P["date"].iloc[0], 250)
    o = ss.score(P).iloc[0]
    assert not o["settled"]
    assert o["bars_seen"] == 9
    txt = ss.summary()
    assert "NOTHING HAS SETTLED" in txt or "open" in txt


def test_a_stop_and_a_target_are_taken_at_whichever_comes_FIRST():
    P = _panel(path="up", n=60)
    ss.emit([{"ticker": "AAAA", "entry": 100.0, "sl": 50.0, "tp": 150.0}],
            "test", "v1", P["date"].iloc[0], 250)
    o = ss.score(P).iloc[0]
    assert o["exit_reason"] == "tp" and o["hit_tp"]
    assert not o["hit_sl"]


def test_a_missing_ticker_is_recorded_as_no_data_not_as_a_zero_return():
    P = _panel(tk="BBBB")
    ss.emit([{"ticker": "AAAA", "entry": 100.0, "sl": None, "tp": None}],
            "test", "v1", "2026-01-05", 20)
    o = ss.score(P).iloc[0]
    assert o["exit_reason"] == "no data"
    assert not o["settled"]
    assert pd.isna(o["ret"]), "a delisted/absent name became a 0% return"


# ==================================================================== HONESTY
def test_the_summary_prints_power_before_it_concludes_anything():
    """A31 recorded conflating a POWER statement with an EFFECT statement as
    its own error, and measured that this kind of edge needs 46,856 months to
    distinguish from random. The store must say how much evidence exists before
    any number derived from it."""
    P = _panel(path="up", n=60)
    for i in range(6):
        ss.emit([{"ticker": "AAAA", "entry": float(P["adj_close"].iloc[i]),
                  "sl": None, "tp": None}], "test", "v1", P["date"].iloc[i], 5)
    ss.score(P)
    txt = ss.summary()
    assert "POWER" in txt
    assert "mean log" in txt, "the mean was printed without the mean log (A36)"


def test_an_empty_store_says_so_rather_than_printing_a_rate():
    assert "EMPTY" in ss.summary()
