"""A guard against a module silently losing every test it had.

WHY THIS EXISTS. `tests/test_portfolio.py` held fifteen tests for
`src/idxbot/portfolio.py`. H20's portfolio study was written in
`scripts/portfolio.py`, its tests were written to the same obvious filename,
and the write replaced the file. Both files held exactly fifteen tests, so the
suite total did not move and nothing in the run said anything had gone. `git
status` reported ` M` rather than `??` and that single character was the only
warning issued.

The general failure is: a module can go from covered to uncovered without any
test failing, because the evidence of coverage is the tests themselves. So this
walks the package and asserts each module is named somewhere under `tests/`.
It is a weak check — a mention is not a test — but it is the check that fires
on total loss, which is the case that is otherwise invisible.
"""

from __future__ import annotations

import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "idxbot"
TESTS = pathlib.Path(__file__).resolve().parent

#  Modules with no test coverage today. Listed rather than hidden: shrinking
#  this set is progress, growing it needs a reason in the commit message.
UNCOVERED = {"analytics.playbook", "tradingview.links"}


def _modules() -> list[str]:
    out = []
    for f in sorted(SRC.rglob("*.py")):
        if f.name == "__init__.py":
            continue
        out.append(".".join(f.relative_to(SRC).with_suffix("").parts))
    return out


def _test_source() -> str:
    return "\n".join(p.read_text() for p in sorted(TESTS.glob("*.py")))


def test_every_module_is_named_by_some_test():
    blob = _test_source()
    missing = [
        m for m in _modules()
        if m not in UNCOVERED
        and m not in blob
        and m.split(".")[-1] not in blob
    ]
    assert not missing, (
        "these modules are named by no test — if a test file was overwritten, "
        f"the coverage is gone and the suite total will not say so: {missing}")


def test_the_uncovered_list_does_not_name_modules_that_are_gone():
    """A stale exemption silently re-permits a real gap under the same name."""
    mods = set(_modules())
    stale = sorted(UNCOVERED - mods)
    assert not stale, f"exempted modules that no longer exist: {stale}"


def test_portfolio_module_specifically_is_still_covered():
    """The exact regression: `idxbot.portfolio` reachable from the CLI, its
    tests replaced wholesale by an unrelated study of the same name."""
    assert "idxbot.portfolio" in _test_source()
