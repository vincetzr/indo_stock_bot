"""Gate 0 §5 check 1 — traded value against an independent source.

THE TEST THIS FILE EXISTS FOR is `test_a_tiny_store_cannot_pass_the_gate`.
A container rebuild left `data/cache/broker_daily` holding exactly two
ticker-days of one name. The check happily reconciled them, found a 0.006%
median error, and printed PASS — a gate satisfied by two rows. A19 records the
smallest cell producing the largest effect three separate times; this is the
same trap with a PASS printed on it.

The second is `test_the_headline_reconciliation_figure_is_computed`. The
verdict block asserted "traded value agrees with an independent source to
0.017%" long after the source that produced 0.017% had been destroyed — the
sentence outlived the number. A26's Pine drift guard is the same lesson.

The third is `test_the_cross_source_comparison_is_split_invariant`. Yahoo
back-adjusts old volume up and old price down by the same factor, so comparing
volumes or prices directly reports a stock split as a data error. A2 already
established that 21% of the spine provably sits on a vendor-adjusted basis, so
this is not a corner case.
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import gate0                                                      # noqa: E402

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, os.pardir, "scripts", "gate0.py")
HAS_IDX = os.path.isdir(gate0.IDX_DAILY)


def _pair(tmp_path, n=400, split=1.0, seed=0):
    """One name in both pipelines. `split` back-adjusts the Yahoo side."""
    rng = np.random.default_rng(seed)
    d = pd.date_range("2020-01-01", periods=n, freq="B")
    px = np.round(np.exp(np.cumsum(rng.normal(0, 0.02, n))) * 1000)
    vol = rng.integers(1e5, 1e7, n).astype(float)
    tmp_path = pathlib.Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    idx = tmp_path / "idx"
    idx.mkdir(exist_ok=True)
    pd.DataFrame({"date": d, "close": px, "high": px * 1.02, "low": px * 0.98,
                  "volume": vol, "value": vol * px * 1.001}
                 ).to_csv(idx / "AAAA.csv", index=False)
    oh = tmp_path / "ohlcv"
    oh.mkdir(exist_ok=True)
    pd.DataFrame({"date": d, "open": px / split, "high": px * 1.02 / split,
                  "low": px * 0.98 / split, "close": px / split,
                  "volume": vol * split}
                 ).to_csv(oh / "AAAA.JK.csv.gz", index=False,
                          compression="gzip")
    return str(idx), str(oh)


# ------------------------------------------------------------- THE FLOORS

def test_a_tiny_store_cannot_pass_the_gate():
    """Two ticker-days is not a reconciliation."""
    assert gate0.MIN_RECON_FILES >= 40
    assert gate0.MIN_RECON_ROWS >= 200
    src = open(SRC).read()
    assert "len(blobs) < MIN_RECON_FILES" in src
    assert "len(M) < MIN_RECON_ROWS" in src, (
        "the floor must be checked on ROWS after the merge as well as on "
        "FILES before it — forty files of one overlapping day each would "
        "otherwise pass")


def test_the_floor_falls_back_rather_than_failing():
    """A thin store is a missing source, not a failed reconciliation. Reporting
    it as FAIL would be as wrong as reporting it as PASS."""
    src = open(SRC).read()
    i = src.index("len(M) < MIN_RECON_ROWS")
    tail = src[i:i + 500]
    assert "_reconcile_against_idx_dataset()" in tail
    assert "Falling back" in tail


# ------------------------------------------------- THE FIGURE IN THE VERDICT

def test_the_headline_reconciliation_figure_is_computed():
    """The retracted 0.017% survives ONLY as a marked retraction in a comment.

    A19 records deleting a refuted claim as its own failure — a reader of the
    history then has no way to see what was corrected. So the number is kept
    where it explains itself and banned from anywhere it could be printed.
    """
    src = open(SRC).read()
    live = [ln for ln in src.splitlines()
            if "0.017%" in ln and not ln.lstrip().startswith("#")]
    assert not live, (
        f"the verdict's reconciliation figure is hardcoded again — it "
        f"outlived its source once already: {live}")
    assert any("0.017%" in ln for ln in src.splitlines()), (
        "the retraction was deleted rather than marked")
    assert "LAST_RECON" in src
    assert "r['median']" in src


def test_both_paths_record_which_source_they_used():
    src = open(SRC).read()
    assert src.count("LAST_RECON.update(") == 2
    assert "IndoPremier session footer" in src
    assert "IDX daily summary" in src


# ---------------------------------------------------------- THE ARITHMETIC

def test_the_cross_source_comparison_is_split_invariant(tmp_path, monkeypatch):
    """A 5:1 split moves Yahoo's volume up and its price down by 5. The gated
    comparison is the PRODUCT, so it must not move."""
    idx_a, oh_a = _pair(tmp_path / "a", split=1.0)
    idx_b, oh_b = _pair(tmp_path / "b", split=5.0)
    monkeypatch.setattr(gate0, "OHLCV", oh_a)
    ok_a, txt_a = gate0._reconcile_against_idx_dataset(idx_a)
    monkeypatch.setattr(gate0, "OHLCV", oh_b)
    ok_b, txt_b = gate0._reconcile_against_idx_dataset(idx_b)
    assert ok_a and ok_b, (txt_a, txt_b)
    #  the split arm must report the same cross-source error as the unsplit one
    def med(t):
        seg = t.split("SPLIT-INVARIANT")[1]
        return float(seg.split("median ")[1].split("%")[0])
    assert med(txt_a) == pytest.approx(med(txt_b), abs=1e-6)


def test_a_split_would_break_a_raw_volume_comparison(tmp_path, monkeypatch):
    """The control for the test above: raw volume DOES move, which is why the
    check reports that number and does not gate on it."""
    idx_b, oh_b = _pair(tmp_path / "b", split=5.0)
    monkeypatch.setattr(gate0, "OHLCV", oh_b)
    _ok, txt = gate0._reconcile_against_idx_dataset(idx_b)
    assert "raw volume agrees within 1% on 0.0%" in txt


def test_a_genuinely_disagreeing_source_fails(tmp_path, monkeypatch):
    """The check must be able to FAIL, or its PASS says nothing."""
    idx, oh = _pair(tmp_path, split=1.0)
    y = pd.read_csv(os.path.join(oh, "AAAA.JK.csv.gz"))
    y["volume"] = y["volume"] * 1.35          # a 35% pipeline disagreement
    y.to_csv(os.path.join(oh, "AAAA.JK.csv.gz"), index=False,
             compression="gzip")
    monkeypatch.setattr(gate0, "OHLCV", oh)
    ok, txt = gate0._reconcile_against_idx_dataset(idx)
    assert not ok, txt


def test_an_absent_dataset_is_reported_rather_than_passed(tmp_path,
                                                          monkeypatch):
    monkeypatch.setattr(gate0, "OHLCV", str(tmp_path))
    ok, txt = gate0._reconcile_against_idx_dataset(str(tmp_path / "nope"))
    assert not ok
    assert "nothing independent to reconcile against" in txt


def test_the_published_value_gap_is_reported_but_not_gated(tmp_path,
                                                           monkeypatch):
    """close is not VWAP, so a ~0.5% gap against the published value is
    expected — the original IndoPremier check documented the same at 0.55%.
    Gating on it would fail a correct spine."""
    idx, oh = _pair(tmp_path)
    monkeypatch.setattr(gate0, "OHLCV", oh)
    ok, txt = gate0._reconcile_against_idx_dataset(idx)
    assert ok
    assert "close is not VWAP, so a gap is expected" in txt


# ------------------------------------------------------- THE REAL SPINE

@pytest.mark.skipif(not HAS_IDX, reason="IDX daily summary not on disk")
def test_the_real_reconciliation_is_wide_and_agrees():
    ok, txt = gate0._reconcile_against_idx_dataset(limit=40)
    assert ok, txt
    assert "overlapping ticker-days" in txt


def test_the_substitution_is_documented_as_a_source_change_not_a_bar_change():
    """CLAUDE.md §2 forbids loosening a criterion to keep a gate alive. What
    changed here is the second source; the thresholds are identical, and the
    file has to say so where a reader will meet it."""
    src = open(SRC).read()
    assert "SAME THRESHOLDS AS THE BROKER-STORE PATH" in src
    assert "forbids" in src or "§2" in src
    assert "med < 0.01 and inside > 0.95" in src
    assert src.count("med < 0.01 and inside > 0.95") == 2, (
        "the two paths must share the same literal threshold expression, or "
        "one can drift from the other")
