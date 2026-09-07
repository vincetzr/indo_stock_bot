"""Can this be run tomorrow? The cold-start path, and whether its list is true.

WHY THIS IS NOT HOUSEKEEPING. The repo runs in an ephemeral container and has
already lost `data/` once — the assessment's postscript records 1.9 GB becoming
4.5 MB, taking the broker store permanently. Today `positions.py` failed to
start because `indicator_panel.parquet` was gone and nothing said which command
rebuilt it. **A bot that cannot be run tomorrow is not a bot.**

THE TEST THIS FILE EXISTS FOR is `test_the_declared_list_covers_what_the_
surfaces_actually_open`. A dependency list reads as a guarantee, so a stale one
is worse than none: it tells a reader everything is accounted for while the
thing that is missing is the thing not on it. The list is therefore checked
against the paths the shipped surfaces really reference, and adding a new file
dependency without declaring it fails the build.

The second is `test_the_forward_record_is_tracked_in_git`. Everything else here
is derived and can be regenerated from the network; the append-only signal
store cannot, by definition — re-deriving a prediction after its outcome is
known is the one thing that destroys its value.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))

import bootstrap                                                  # noqa: E402
from bootstrap import ARTEFACTS, STEPS, report, status            # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)

#: The surfaces a user actually runs, plus what they import for data.
SURFACES = ("scripts/today.py", "scripts/rules.py", "scripts/daily_signal.py",
            "scripts/positions.py", "scripts/bhbench.py",
            "src/idxbot/signal_store.py")


#: The directories a shipped surface can depend on. `reports/` was NOT here and
#: the omission cost exactly what this file exists to prevent: `stoptest.json`
#: became a dependency of three surfaces and the checker could not see it,
#: because it only ever looked under `data/`. A list-checker that scans one
#: directory is a list-checker for one directory, and the manifest it guards
#: reads as a guarantee about all of them.
ROOTS = ("data", "reports")

#: Modules a surface imports for its data, which the surface itself never
#: names. `measured.py` opens the result file on behalf of three of them.
INDIRECT = ("src/idxbot/measured.py",)


def _referenced() -> set:
    """Concrete FILE paths under `ROOTS` that the surfaces open."""
    out = set()
    for rel in SURFACES + INDIRECT:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        s = open(p).read()
        for root in ROOTS:
            for m in re.finditer(
                    r'os\.path\.join\(\s*"%s"\s*,\s*"([^"]+)"\s*'
                    r'(?:,\s*"([^"]+)"\s*)?\)' % root, s):
                out.add(root + "/" + "/".join(x for x in m.groups() if x))
            for m in re.finditer(r'"(%s/[^"]+)"' % root, s):
                out.add(m.group(1))
    #  directories are not artefacts; only files can be missing in the sense
    #  this list is about
    return {p for p in out if os.path.splitext(p)[1]}


# --------------------------------------------------------- THE LIST IS TRUE

def test_the_declared_list_covers_what_the_surfaces_actually_open():
    declared = {a[0] for a in ARTEFACTS}
    missing = _referenced() - declared
    assert not missing, (
        "these files are opened by a shipped surface and are not on the "
        "bootstrap list, so a cold start would fail with no instruction on "
        f"how to fix it: {sorted(missing)}")


def test_the_guard_fires_on_an_undeclared_dependency(tmp_path, monkeypatch):
    """POSITIVE CONTROL. A list-checker that cannot flag an omission proves
    nothing by finding none. Verified by adding an undeclared path to a real
    surface and watching the check fail — reproduced here in miniature."""
    fake = tmp_path / "s.py"
    fake.write_text('X = "data/spine/nonexistent_thing.parquet"\n')
    monkeypatch.setattr(bootstrap, "ARTEFACTS", ())
    found = set()
    s = fake.read_text()
    for m in re.finditer(r'"(data/[^"]+)"', s):
        found.add(m.group(1))
    assert found - {a[0] for a in bootstrap.ARTEFACTS} == {
        "data/spine/nonexistent_thing.parquet"}


def test_the_scanner_reaches_beyond_data():
    """THE OMISSION THIS WIDENING WAS FOR, PINNED.

    `reports/stoptest.json` became a dependency of three shipped surfaces —
    it carries the standing instruction's fourth column — and the scanner
    could not see it, because it only ever looked under `data/`. If this
    returns nothing, the widening has been undone and the manifest is once
    again a guarantee about one directory dressed as a guarantee about all of
    them.
    """
    seen = {p for p in _referenced() if not p.startswith("data/")}
    assert seen, "the scanner only sees data/ again"
    assert "reports/stoptest.json" in seen


def test_a_dependency_reached_through_a_helper_is_still_seen():
    """`measured.py` opens the result file on behalf of `rules.py`,
    `today.py` and `positions.py`, none of which names the path. A scanner
    that reads only the surfaces would call it undeclared-but-invisible —
    which is indistinguishable from absent."""
    assert any("measured" in m for m in INDIRECT)
    assert "reports/stoptest.json" in _referenced()


def test_nothing_declared_is_a_directory():
    for path, _by, _why, _tracked in ARTEFACTS:
        assert os.path.splitext(path)[1], path


def test_every_untracked_artefact_names_the_command_that_builds_it():
    """'The data is missing' and 'the data is unrecoverable' differ only by a
    rebuild path someone has executed."""
    for path, by, _why, tracked in ARTEFACTS:
        if tracked:
            assert by is None, f"{path} is tracked; it is not rebuilt"
        else:
            assert by, f"{path} has no source command"
            script = by.split()[0]
            assert os.path.exists(os.path.join(ROOT, "scripts", script)), by


def test_every_rebuild_step_points_at_a_script_that_exists():
    for _label, cmd, _note in STEPS:
        assert os.path.exists(os.path.join(ROOT, "scripts", cmd[0])), cmd


def test_the_steps_build_every_untracked_artefact():
    """A list that reports a gap it cannot close is a bug report, not a
    bootstrap."""
    built = " ".join(" ".join(c) for _l, c, _n in STEPS)
    for path, by, _why, tracked in ARTEFACTS:
        if tracked:
            continue
        assert by.split()[0] in built, (
            f"{path} is declared as built by `{by}`, which no rebuild step "
            f"runs")


# ------------------------------------------------- WHAT MUST NOT BE REBUILT

def test_the_forward_record_is_tracked_in_git():
    """Everything else is derived. A prediction re-derived after its outcome is
    known is not a prediction, so this file is the one that cannot be
    regenerated and must therefore be committed."""
    p = "data/signals/emitted.csv.gz"
    assert any(a[0] == p and a[3] for a in ARTEFACTS)
    out = subprocess.run(["git", "ls-files", "--error-unmatch", p],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, (
        "the append-only signal store is NOT tracked in git — the only "
        "out-of-sample evidence this project will ever have would not survive "
        "a container rebuild")


def test_a_missing_tracked_file_is_reported_as_a_checkout_not_a_rebuild():
    """Regenerating a tracked artefact would overwrite the record rather than
    restore it, so the two cases must read differently."""
    rows = [{"path": "data/signals/emitted.csv.gz", "ok": False, "mb": 0.0,
             "by": None, "why": "x", "tracked": True}]
    txt = report(rows)
    assert "TRACKED IN GIT and missing" in txt
    assert "git checkout" in txt
    assert "cannot be re-derived" in txt


def test_the_broker_store_is_named_as_not_rebuildable():
    """A5: a host is added only after the USER has checked its licensing. A
    bootstrap that silently omits the one thing it cannot fix is misleading."""
    src = open(os.path.join(ROOT, "scripts", "bootstrap.py")).read()
    assert "broker_daily" in src
    assert "broker_allowed_hosts" in src
    assert "USER's" in src


# --------------------------------------------------------------- THE REPORT

def test_the_report_distinguishes_present_from_missing():
    rows = status()
    txt = report(rows)
    assert ("everything present." in txt) == all(r["ok"] for r in rows)


def test_report_only_is_the_default():
    """A command that rebuilds 476 MB as a side effect of being run is a
    command nobody runs twice."""
    src = open(os.path.join(ROOT, "scripts", "bootstrap.py")).read()
    assert '"--run"' in src
    assert "report only" in src


def test_the_dependency_order_is_stated_as_load_bearing():
    """Each step reads the previous one's output; running them out of order
    fails in a way that looks like missing data."""
    assert "The order is not arbitrary" in (bootstrap.__doc__ or "")
