"""DuckDB views over the existing spine files.

THE TEST THIS FILE EXISTS FOR is `test_duckdb_and_pandas_agree_on_every_check`.
A query engine that returns DIFFERENT numbers from the path every existing
study used is worse than no query engine, and the difference would be silent —
a null handled differently, a float read as decimal, a date normalised in
another timezone. Nothing in the output would look wrong.

The second is `test_there_is_no_write_path`. The spine is built by
`scripts/refresh.py` and repaired by an auditable, reversible registry. A SQL
UPDATE would be neither, and a `read_only=False` that quietly worked would make
the repair registry a lie.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot import store                                          # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
HAS_PANEL = os.path.exists(os.path.join(ROOT, "data/spine/price_panel.parquet"))


def _tiny(tmp_path):
    """A two-file miniature spine, so the structural tests need no 258 MB."""
    (tmp_path / "data" / "spine").mkdir(parents=True)
    p = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-02"]),
        "ticker": ["AAAA", "AAAA", "BBBB"],
        "adj_close": [100.0, 110.0, 50.0],
        "tradeable": [True, True, False],
    })
    p.to_parquet(tmp_path / "data/spine/price_panel.parquet", index=False)
    s = pd.DataFrame({"ticker": ["AAAA", "AAAA", "BBBB"],
                      "date": pd.to_datetime(["2020-01-02", "2020-01-03",
                                              "2020-01-02"]),
                      "listed_shares": [1e6, 2e6, 5e6]})
    s.to_parquet(tmp_path / "data/spine/shares_pit.parquet", index=False)
    return str(tmp_path)


# ------------------------------------------------------------- the catalogue

def test_only_files_that_exist_become_views(tmp_path):
    """A partial checkout or a fresh container must give a smaller catalogue,
    never an import error — a container rebuild has already emptied `data/`
    once in this project."""
    root = _tiny(tmp_path)
    assert set(store.available(root)) == {"prices", "shares_pit"}


def test_a_query_runs_against_the_miniature_spine(tmp_path):
    root = _tiny(tmp_path)
    got = store.sql("SELECT ticker, COUNT(*) n FROM prices GROUP BY ticker "
                    "ORDER BY ticker", root)
    assert list(got["ticker"]) == ["AAAA", "BBBB"]
    assert list(got["n"]) == [2, 1]
    store.close()


def test_the_connection_is_cached_between_calls(tmp_path):
    """The whole performance case rests on this. A fresh connection per call
    re-registers every view and re-opens a 258 MB file, which measured 2.5-35x
    SLOWER than pandas while returning perfectly correct answers."""
    root = _tiny(tmp_path)
    store.close()
    store.sql("SELECT 1", root)
    first = store._CON[os.path.abspath(root)]
    store.sql("SELECT 2", root)
    assert store._CON[os.path.abspath(root)] is first
    store.close()
    assert not store._CON


def test_there_is_no_write_path():
    with pytest.raises(ValueError, match="no write path"):
        store.connect(read_only=False)


def test_the_speedup_claim_is_reproducible_not_asserted():
    """`docs/PLATFORM_ASSESSMENT.md` claimed 13-28x with nothing behind it.
    A performance claim that cannot be re-run is the kind this repo deletes."""
    src = open(os.path.join(ROOT, "src", "idxbot", "store.py")).read()
    assert "def benchmark(" in src
    assert "SLOWER" in src, (
        "the module must record that the naive version was slower, or the "
        "next author re-derives the wrong conclusion")


# --------------------------------------------------- the agreement, on real data

@pytest.mark.skipif(not HAS_PANEL, reason="spine panel not present")
def test_duckdb_and_pandas_agree_on_every_check():
    rows = store.verify(ROOT)
    bad = [r for r in rows if r.get("ok") is False]
    assert not bad, store.describe(rows)
    assert any(r.get("ok") for r in rows), "no check actually ran"


@pytest.mark.skipif(not HAS_PANEL, reason="spine panel not present")
def test_verify_reports_a_missing_source_rather_than_failing(tmp_path):
    rows = store.verify(_tiny(tmp_path))
    skipped = [r for r in rows if r.get("ok") is None]
    ran = [r for r in rows if r.get("ok") is not None]
    assert ran, "the miniature spine should still exercise the price checks"
    assert all(r.get("why") for r in skipped)
    store.close()


def test_describe_names_the_failing_check():
    txt = store.describe([{"check": "x", "view": "prices", "ok": False,
                           "duckdb": 1.0, "pandas": 2.0}])
    assert "FAIL" in txt and "1 of 1" in txt
    assert "Do not use the SQL path" in txt
