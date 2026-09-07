"""GDELT — the backtestable narrative route, and its quarantine.

THE TEST THIS FILE LEADS WITH is `test_no_statistical_module_imports_gdelt`,
for the same reason `tests/test_news.py` leads with its twin: a narrative
feature whose point-in-time property is unverified is look-ahead with NO
VISIBLE SYMPTOM. The series looks perfectly historical either way.

The second is `test_a_bare_ticker_is_refused_as_a_query`. Measured over 2023,
the bare token GULA returns 238 non-zero days — a dense, plausible series about
sugar, since that is what the word means. No downstream statistic can detect
that, so the refusal has to happen at the query.
"""

from __future__ import annotations

import ast
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.data import gdelt                                     # noqa: E402
from idxbot.data.cache import Cache                               # noqa: E402
from idxbot.data.gdelt import (GDELT, parse_timeline,             # noqa: E402
                               quarantine_note, query_for, stability)

ROOT = os.path.join(os.path.dirname(__file__), os.pardir, "src", "idxbot")


# --------------------------------------------------------------- QUARANTINE

def test_no_statistical_module_imports_gdelt():
    """Whether GDELT's index is stable for past dates is UNVERIFIED, so any
    statistic reading it may be look-ahead by construction."""
    offenders = []
    for sub in ("spine", "features"):
        d = os.path.join(ROOT, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".py"):
                continue
            tree = ast.parse(open(os.path.join(d, fn)).read())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [f"{node.module or ''}.{a.name}"
                             for a in node.names]
                if any(n.split(".")[-1] == "gdelt" for n in names):
                    offenders.append(f"{sub}/{fn}")
    assert not offenders, f"gdelt must not reach a statistic: {offenders}"


def test_the_quarantine_note_says_what_would_lift_it():
    """A19: a negative answer is unfinished until it says what would change it.
    'Unverified' with no test attached is a permanent excuse."""
    n = quarantine_note()
    assert "MAY NOT ENTER ANY STATISTIC" in n
    assert "stability()" in n
    assert "fetched_at" in n


def test_every_row_carries_when_it_was_fetched():
    """The ONLY thing that makes the stability question answerable later."""
    df = parse_timeline({"timeline": [{"data": [
        {"date": "20200101T000000Z", "value": 1.5},
        {"date": "20200102T000000Z", "value": 2.0}]}]}, '"X"')
    assert list(df.columns) == ["date", "value", "query", "fetched_at"]
    assert df["fetched_at"].notna().all()


# ------------------------------------------------------------ THE RELEVANCE

def test_a_bare_ticker_is_refused_as_a_query():
    with pytest.raises(ValueError, match="company NAME"):
        query_for("", ticker="GULA")


def test_the_company_name_is_quoted_so_it_is_a_phrase():
    assert query_for("Adaro Energy") == '"Adaro Energy"'
    assert query_for("  Bank Central Asia  ") == '"Bank Central Asia"'


def test_raw_is_available_but_must_be_asked_for_explicitly():
    assert query_for("", ticker="adro", raw=True) == "ADRO"
    with pytest.raises(ValueError):
        query_for("Adaro", raw=True)


def test_the_measured_relevance_failure_is_recorded_not_merely_warned_about():
    doc = gdelt.__doc__ or ""
    assert "238" in doc and "GULA" in doc, (
        "the docstring must carry the MEASUREMENT, not just the warning — a "
        "warning with no number behind it gets edited away")


# ----------------------------------------------------------------- THE RATE

def test_the_rate_is_measured_and_slower_than_the_operator_states():
    """The 429 body asks for one request every 5 seconds. Measured from this
    host, 6s gets 429 and 15s succeeds 2 of 4."""
    assert gdelt.MIN_GAP >= 20.0
    doc = gdelt.__doc__ or ""
    assert "MEASURED RATE" in doc
    assert "15s" in doc
    assert "NOT ESTABLISHED" in doc, (
        "the rate must be reported as unestablished — 2 of 4 at 15s and a "
        "failure at 20s is not a schedule, and probing harder to pin it down "
        "means hammering a rate-limited public API to see how hard you can "
        "hammer it")


def test_the_client_waits_between_calls(monkeypatch):
    slept = []
    monkeypatch.setattr(gdelt.time, "sleep", lambda s: slept.append(s))
    g = GDELT(cache=Cache("/tmp/gdelt-test-cache"), min_gap=7.0)
    g._last = gdelt.time.time()
    g._wait()
    assert slept and 0 < slept[0] <= 7.0


def test_a_429_is_backed_off_and_never_worked_around(monkeypatch):
    """Defeating a rate limit is the one response that is not available."""
    calls = {"n": 0}

    class R:
        status_code = 429
        text = "Please limit requests"

        def json(self):
            raise ValueError

    class S:
        headers = {}

        def get(self, *a, **k):
            calls["n"] += 1
            return R()

    monkeypatch.setattr(gdelt.time, "sleep", lambda s: None)
    g = GDELT(cache=Cache("/tmp/gdelt-test-cache"), session=S(), min_gap=0.0)
    assert g._get({"query": "x"}) is None
    assert calls["n"] == gdelt.RETRIES
    src = open(os.path.join(ROOT, "data", "gdelt.py")).read()
    assert "curl_cffi" not in src and "impersonate" not in src


# ---------------------------------------------------------------- THE PARSE

def test_an_empty_payload_gives_an_empty_frame_not_a_raise():
    for p in (None, {}, {"timeline": []}, {"timeline": [{}]},
              {"timeline": [{"data": []}]}):
        assert parse_timeline(p, "q").empty


def test_an_unparseable_date_is_dropped_rather_than_becoming_today():
    df = parse_timeline({"timeline": [{"data": [
        {"date": "not-a-date", "value": 1.0},
        {"date": "20200102T000000Z", "value": 2.0}]}]}, "q")
    assert len(df) == 1
    assert df["date"].iloc[0] == pd.Timestamp("2020-01-02")


def test_a_missing_value_becomes_zero_not_nan():
    df = parse_timeline({"timeline": [{"data": [
        {"date": "20200101T000000Z"}]}]}, "q")
    assert df["value"].iloc[0] == 0.0


# ------------------------------------------------------------ THE STABILITY

def test_stability_is_undefined_on_a_single_fetch():
    df = parse_timeline({"timeline": [{"data": [
        {"date": "20200101T000000Z", "value": 1.0}]}]}, "q")
    assert stability(df, None) == {}
    assert stability(df, df.iloc[0:0]) == {}


def test_stability_detects_a_retro_indexed_revision():
    a = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                      "value": [1.0, 2.0],
                      "fetched_at": pd.Timestamp("2026-01-01")})
    b = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                      "value": [1.4, 2.0],
                      "fetched_at": pd.Timestamp("2026-06-01")})
    s = stability(a, b)
    assert s["disagree"] == 0.5
    assert s["mean_change"] == pytest.approx(0.2)
    assert s["gap_days"] > 100


def test_stability_reports_zero_disagreement_on_an_identical_refetch():
    a = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "value": [3.0],
                      "fetched_at": pd.Timestamp("2026-01-01")})
    b = a.copy()
    b["fetched_at"] = pd.Timestamp("2026-06-01")
    s = stability(a, b)
    assert s["disagree"] == 0.0 and s["mean_change"] == 0.0


def test_the_immutable_alternative_is_recorded_as_available_and_not_taken():
    """A19: naming a cost and not paying it is premature closure, but so is
    pretending the expensive route does not exist."""
    doc = gdelt.__doc__ or ""
    assert "masterfilelist" in doc
    assert "terabytes" in doc
    assert gdelt.MASTER_LIST.endswith("masterfilelist.txt")
