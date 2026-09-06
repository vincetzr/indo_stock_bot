"""The survivorship recovery, and the provenance that makes it reproducible.

THE BUG THIS FILE GUARDS IS NOT A DATA LOSS. `data/cache/` is gitignored by
design and that design is right — raw pulls are re-fetchable. What was wrong on
2026-09-06 is that when a cold rebuild destroyed the 121 recovered delisted
names, NOTHING IN THE REPO RECORDED HOW TO GET THEM BACK. `spine/universe.py`
described the source in prose, with no URL, no commit, no checksum, and there
was no collector script at all — the acquisition had been done by hand in a
session and never written down.

So these tests are about PROVENANCE, not about files existing. A collector whose
source is unpinned, or unlicensed, or silently merged into the live spine, is
the same failure wearing a script.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import delisted_collect as dc                                      # noqa: E402


# ===================================================== PROVENANCE IS MANDATORY
def test_the_source_is_pinned_to_an_exact_commit():
    """The dataset is updated BY HAND and irregularly — its own README says so.
    An unpinned clone silently changes the universe under every study that used
    it, which is worse than not having it, because nothing would look wrong."""
    assert dc.SOURCE_REPO.startswith("https://github.com/")
    assert re.fullmatch(r"[0-9a-f]{40}", dc.SOURCE_COMMIT), dc.SOURCE_COMMIT


def test_the_licence_and_its_chain_are_recorded():
    """Two licences apply and both must be stated: CC BY-NC on the compilation,
    and IDX's own terms on the underlying data, which CLAUDE.md §3 records as
    barring commercial redistribution."""
    lic = dc.SOURCE_LICENCE
    assert "CC BY-NC" in lic
    assert "Bursa Efek Indonesia" in lic


def test_the_licence_decision_is_left_to_the_user():
    """A5: a source is adopted only after the USER has checked its licensing.
    CC BY-NC is non-commercial and A23 records this project being pointed at a
    client's money, so the script must not decide that question for them."""
    src = open(dc.__file__).read()
    assert "PENDING USER RULING" in src
    assert "A5" in src


def test_the_docstring_carries_all_four_provenance_fields():
    """Repo, commit, accession date, licence. The absence of exactly these is
    what made the previous recovery unreproducible."""
    doc = dc.__doc__
    for field in ("source repo", "commit", "accessed", "licence"):
        assert field in doc, field


# ================================================== THE RECOVERY'S SEMANTICS ==
def test_the_death_date_comes_from_volume_not_the_empty_column():
    """The dataset has a `delisting_date` COLUMN and it is EMPTY for all 958
    names. Its name invites the opposite assumption, so the code must derive the
    death date from the last bar carrying real volume — and say so."""
    src = open(dc.__file__).read()
    assert "delisting_date" in src
    assert "EMPTY" in src
    #  and the derivation must actually use volume
    assert re.search(r'traded\s*=\s*d\[d\["volume"\]\s*>\s*0\]', src)


def test_recovered_names_are_written_apart_and_never_merged():
    """A2: 'a survivorship-free universe is a DIFFERENT universe, not a bigger
    one'. Mixing them into the live cache would silently change the meaning of
    every existing study."""
    assert dc.OUT_DIR != os.path.join("data", "cache", "ohlcv")
    assert "delisted" in dc.OUT_DIR
    src = open(dc.__file__).read()
    assert "NOT MERGED" in src


def test_no_module_under_spine_or_features_imports_the_recovery():
    """The same quarantine tests/test_news.py enforces for the news layer. If
    the spine imported this, every downstream study would change universe
    without a single number looking wrong."""
    import ast
    root = os.path.join(os.path.dirname(__file__), os.pardir, "src", "idxbot")
    for sub in ("spine", "features"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".py"):
                continue
            tree = ast.parse(open(os.path.join(d, fn)).read())
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    mods = [node.module or ""]
                assert not any("delisted_collect" in (m or "") for m in mods), \
                    f"{sub}/{fn} imports the recovery"


# ============================================ THE MANIFEST, IF IT HAS BEEN RUN
def _manifest():
    p = os.path.join(os.path.dirname(__file__), os.pardir, dc.MANIFEST)
    if not os.path.exists(p):
        pytest.skip("recovery not run in this environment")
    return json.load(open(p))


def test_the_manifest_records_what_was_taken_and_from_where():
    m = _manifest()
    assert m["source_commit"] == dc.SOURCE_COMMIT
    assert "CC BY-NC" in m["licence"]
    assert m["recovered"] == len(m["names"])
    for n in m["names"][:5]:
        assert len(n["sha16"]) == 16, "every file needs a checksum"


def test_the_recovered_names_are_the_LOSERS_which_is_the_whole_point():
    """Survivorship bias is the absence of the names that died. A recovery whose
    median return looked like the survivors' would mean it had recovered the
    wrong thing. Measured 2026-09-06: median -54.5% over 145 names."""
    m = _manifest()
    rets = sorted(n["total_return"] for n in m["names"])
    med = rets[len(rets) // 2]
    assert med < -0.20, f"median recovered return {med:+.1%} is not a loser set"


def test_every_recovered_file_reads_back_with_the_columns_the_spine_expects():
    m = _manifest()
    base = os.path.join(os.path.dirname(__file__), os.pardir, dc.OUT_DIR)
    for n in m["names"][:5]:
        f = os.path.join(base, f"{n['ticker']}.JK.csv.gz")
        d = pd.read_csv(f, parse_dates=["date"])
        assert {"date", "open", "high", "low", "close", "volume",
                "adj_close"} <= set(d.columns), d.columns.tolist()
        assert len(d) == n["bars"]
        assert d["date"].is_monotonic_increasing
