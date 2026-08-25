"""Tests for the IDX-IC classification layer.

The first block is policy, like `test_news.py`'s: this is a CURRENT-STATE
snapshot with no history and no delisted names, so a statistic that read it
would be assigning 2026 sectors to 2015 bars and silently dropping every name
that died. It must never reach `spine/` or `features/`.

The rest pin the things the source actually gets wrong, or that a caller could
easily get wrong about it: a frozen snapshot that misses recent listings, a
sector label asserted where the loadings do not support one, and a listing
board that is NOT the same fact as the auto-rejection board.
"""

from __future__ import annotations

import ast
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.data import classification as C                    # noqa: E402


def cls(rows) -> pd.DataFrame:
    """rows: (ticker, sector) or (ticker, sector, board)."""
    return pd.DataFrame([
        {"ticker": r[0], "sector": r[1], "company": r[0],
         "board": r[2] if len(r) > 2 else "main",
         "listing_date": pd.Timestamp("2020-01-01"), "shares": 1e9}
        for r in rows])


# --------------------------------------------------------------------------
# THE QUARANTINE
# --------------------------------------------------------------------------
def test_no_statistical_module_imports_the_classification():
    """A name's sector today is not its sector in 2015, and this file holds no
    delisted names at all — so a backtest reading it would be both look-ahead
    and survivorship-biased at once."""
    root = os.path.join(os.path.dirname(__file__), os.pardir, "src", "idxbot")
    offenders = []
    for sub in ("spine", "features"):
        d = os.path.join(root, sub)
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
                if any(n.split(".")[-1] == "classification" for n in names):
                    offenders.append(f"{sub}/{fn}")
    assert not offenders, f"classification must not reach a statistic: {offenders}"


def test_the_attribution_is_carried_and_names_the_licence():
    """CC BY-NC 4.0 requires it, and the underlying data is IDX's."""
    a = C.ATTRIBUTION
    assert "CC BY-NC" in a
    assert "Bursa Efek Indonesia" in a
    assert "2024-07-10" in a, "the freeze date must travel with the data"


def test_only_the_one_allowed_host_is_fetched():
    with pytest.raises(C.HostNotAllowed):
        C._fetch("https://evil.example.com/Energy.csv")


def test_idx_co_id_is_not_the_source():
    """Its own list is 403 behind Cloudflare; docs/FULL_REKAP.md §3 explains
    why this repo does not go through that."""
    assert "idx.co.id" not in C.BASE
    assert C.HOST == "raw.githubusercontent.com"


def test_the_eleven_official_sectors_are_the_taxonomy():
    assert len(C.SECTORS) == 11
    for s in ("Energy", "Financials", "Healthcare", "Technology",
              "Properties & Real Estate"):
        assert s in C.SECTORS


# --------------------------------------------------------------------------
# COVERAGE — a frozen snapshot must SAY it is frozen
# --------------------------------------------------------------------------
def test_coverage_reports_the_names_it_cannot_classify():
    D = cls([("BBCA", "Financials"), ("ADRO", "Energy")])
    cov = C.coverage(D, ["BBCA", "ADRO", "COIN", "RATU"])
    assert cov["n_classified"] == 2
    assert cov["share"] == pytest.approx(0.5)
    assert cov["missing"] == ["COIN", "RATU"]


def test_an_unclassified_name_is_labelled_not_dropped():
    """The gap is systematically the newest listings — exactly the names a
    daily brief is most likely to be asked about. Dropping them would make the
    sector table silently unrepresentative."""
    S = pd.DataFrame({"adj_close": [10.0, 20.0, 30.0], "ma20": [9.0, 21.0, 29.0],
                      "ret1": [0.01, -0.01, 0.02]},
                     index=pd.Index(["BBCA", "ADRO", "COIN"], name="ticker"))
    D = cls([("BBCA", "Financials"), ("ADRO", "Energy")])
    SB = C.sector_breadth(S, D, min_names=1)
    assert "unclassified" in set(SB["sector"])
    assert int(SB["n"].sum()) == 3, "every traded name must appear somewhere"


def test_a_sector_with_too_few_names_is_dropped_rather_than_reported():
    S = pd.DataFrame({"adj_close": [10.0] * 6, "ma20": [9.0] * 6,
                      "ret1": [0.01] * 6},
                     index=pd.Index([f"T{i}" for i in range(6)], name="ticker"))
    D = cls([(f"T{i}", "Energy" if i < 5 else "Healthcare") for i in range(6)])
    SB = C.sector_breadth(S, D, min_names=4)
    assert set(SB["sector"]) == {"Energy"}, "one name is not a sector reading"


def test_sector_breadth_counts_above_the_average_correctly():
    S = pd.DataFrame({"adj_close": [10.0, 10.0, 10.0, 10.0],
                      "ma20": [9.0, 9.0, 11.0, 11.0],
                      "ret1": [0.01, 0.01, -0.01, -0.01]},
                     index=pd.Index(["A", "B", "C", "D"], name="ticker"))
    D = cls([(t, "Energy") for t in "ABCD"])
    SB = C.sector_breadth(S, D, min_names=1)
    assert SB["above_ma"].iloc[0] == pytest.approx(0.5)
    assert SB["advancing"].iloc[0] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# ANNOTATING THE PCA — a label must be earned
# --------------------------------------------------------------------------
def comp(names) -> pd.DataFrame:
    return pd.DataFrame([{"pc": 1, "var_share": 0.1, "today_share": 0.1,
                          "score_z": 1.0, "abs_pct": 0.5,
                          "with": list(names), "against": []}])


def test_a_concentrated_component_gets_a_counted_label():
    """The label is a COUNT, not a title. §9.6's rule: the component is the
    observation and the sector name is an interpretation."""
    D = cls([(t, "Financials") for t in ("BBCA", "BBRI", "BMRI", "BBNI",
                                         "BBTN")] + [("ADRO", "Energy")])
    A = C.annotate_components(comp(["BBCA", "BBRI", "BMRI", "BBNI", "BBTN",
                                    "ADRO"]), D)
    assert A["sector_label"].iloc[0] == "5/6 Financials"
    assert A["sector_purity"].iloc[0] == pytest.approx(5 / 6)


def test_a_component_spanning_sectors_gets_NO_label():
    """And that is the more interesting output: something is moving together
    that the taxonomy does not explain. On the live panel PC4 is coal plus palm
    oil, which IDX-IC splits across Energy and Consumer Non-Cyclicals."""
    D = cls([("A", "Energy"), ("B", "Financials"), ("C", "Healthcare"),
             ("D", "Technology"), ("E", "Industrials"), ("F", "Energy")])
    A = C.annotate_components(comp(list("ABCDEF")), D)
    assert A["sector_label"].iloc[0] == ""
    assert A["sector_purity"].iloc[0] < 0.5


def test_annotation_survives_names_the_map_has_never_heard_of():
    D = cls([("A", "Energy"), ("B", "Energy")])
    A = C.annotate_components(comp(["A", "B", "NEWCO", "ALSONEW"]), D)
    assert "sector_label" in A
    assert np.isfinite(A["sector_purity"].iloc[0])


def test_annotation_is_a_no_op_when_there_is_no_map():
    cm = comp(["A", "B"])
    assert C.annotate_components(cm, pd.DataFrame()).equals(cm)


# --------------------------------------------------------------------------
# THE BOARD, which is NOT the same fact as the auto-rejection board
# --------------------------------------------------------------------------
def test_the_listing_boards_map_onto_the_reference_ladder_names():
    from idxbot.spine import reference
    known = set(reference.MAIN_BOARDS) | set(reference.THIN_BOARDS)
    for v in C.BOARD_MAP.values():
        assert v in known, f"{v} is not a board reference.py can band"


def test_the_board_map_documents_why_it_is_not_used_for_bands():
    """This snapshot puts 216 names on Pemantauan Khusus; the price rule finds
    41 today. Both are right about different things — IDX has eleven criteria
    and only the price one is derivable — so the price rule is a lower bound
    and this file is a two-year-stale upper one. The brief keeps the price
    rule because it is point-in-time."""
    # normalise the comment wrapping before matching: the phrase is split
    # across lines by "#: " continuations, which is a formatting detail and
    # not something a test should be sensitive to.
    import re
    src = re.sub(r"\s*#:\s*|\s+", " ", open(C.__file__).read())
    assert "LOWER BOUND" in src
    assert "eleven criteria" in src
    assert "point-in-time" in src
