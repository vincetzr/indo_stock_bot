"""The STANDING INSTRUCTION's prohibitions, made mechanical.

CLAUDE.md's standing instruction lists four things this repo may never say, and
until now all four were a matter of whoever was writing remembering them. They
were not remembered: an audit found **twelve** live claims of the banned kind,
including a printed verdict reading `"validated - the one to use"` and a
position monitor describing its exit rules as "the ones H17/H18 validated" —
a claim A18 had explicitly WITHDRAWN, sitting in a file edited the same day.

That is the A19 shape for the seventh time: the code gets corrected and the
sentence above it does not. A prohibition nobody can fail is not a prohibition,
so these are tests.

WHAT IS AND IS NOT FLAGGED. `validate_tick_schedule`, `validation_fraction`,
`params.validate()`, `invalidation` and "cross-validation" are IDENTIFIERS and
ordinary technical vocabulary; banning them would make the check noise and the
check would then be turned off. What is banned is the word used as a CLAIM
about a result. The rule below is therefore: the claim-form of the word may
appear only where the same sentence takes it back.
"""

from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.insert(0, os.path.join(ROOT, "src"))
TREES = ("scripts", os.path.join("src", "idxbot"))

#: Identifiers and technical vocabulary. Not claims.
#: CASE-INSENSITIVE, because it fired on the author's own `INVALIDATED` in an
#: upper-case comment. `invalidated` is the OPPOSITE of the banned claim, so
#: matching it only in lower case made the guard reject a retraction.
IDENTIFIER = re.compile(
    r"validate_|_validate|\.validate\(|validation_|invalidat|"
    r"cross-validation|cross_validation|sectors_validate|"
    r'"validated":|validate\(|def validate|Validated categorical|'
    r"validation slice|validation fraction|fails validation",
    re.IGNORECASE)

#: A claim is taken back if the surrounding sentence says so.
NEGATION = (
    "not ", "n't", "never", "cannot", "no ", "nothing", "NOT", "NEVER",
    "superseded", "SUPERSEDED", "withdrew", "withdrawn", "retracted",
    "unvalidated", "is what this said", "is what this printed",
    "in-sample", "IN-SAMPLE", "would be", "bans it", "not available here",
)


def _claims(word: str = "validated"):
    """Every line using `word` as a claim, with a two-line context window."""
    out = []
    for tree in TREES:
        for f in sorted(glob.glob(os.path.join(ROOT, tree, "**", "*.py"),
                                  recursive=True)):
            if "__pycache__" in f:
                continue
            lines = open(f, errors="replace").read().splitlines()
            for i, ln in enumerate(lines):
                if word not in ln.lower():
                    continue
                if IDENTIFIER.search(ln):
                    continue
                window = " ".join(lines[max(0, i - 1):i + 2])
                if any(n in window for n in NEGATION):
                    continue
                out.append((os.path.relpath(f, ROOT), i + 1, ln.strip()[:110]))
    return out


# ------------------------------------------------------- "nothing validated"

def test_nothing_is_described_as_validated():
    """CLAUDE.md: 'Nothing described as validated. The holdout was spent at
    H16. Every number in this repo is in-sample.'"""
    bad = _claims("validated")
    assert not bad, (
        "these describe a result as validated, which the standing instruction "
        "forbids because the holdout was spent at H16:\n"
        + "\n".join(f"  {f}:{i}  {ln}" for f, i, ln in bad))


def test_the_checker_can_actually_fire():
    """POSITIVE CONTROL. A guard that cannot flag anything proves nothing by
    flagging nothing — A26's sine wave, A27's planted Fibonacci bump, A36's Q0
    martingale, and now this."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "scripts"))
        p = os.path.join(d, "scripts", "x.py")
        open(p, "w").write('"""The validated rule is the one to use."""\n')
        lines = open(p).read().splitlines()
        ln = lines[0]
        assert "validated" in ln.lower()
        assert not IDENTIFIER.search(ln)
        assert not any(n in ln for n in NEGATION)


def test_the_checker_does_not_flag_identifiers():
    for ln in ("def validate_tick_schedule(files) -> tuple:",
               "    ticks_ok, t = validate_tick_schedule(files)",
               "    params.validate()",
               "        early_stopping=True, validation_fraction=0.15,",
               "    invalidation: str = \"\"",
               "* **No shuffled cross-validation.** Rows are ordered in time",
               '        **conc, "validated": False}'):
        assert IDENTIFIER.search(ln), ln


# --------------------------------------------- the other three prohibitions

def test_no_per_trade_mean_without_the_mean_log_beside_it():
    """A36 measured a +5.77% average trade compounding at −5.3% a year — the
    two disagreeing in SIGN. The metrics module is where a per-trade mean is
    computed, so it is where the pairing has to be structural."""
    from idxbot.metrics import expectancy
    e = expectancy([1.0, -0.6, -0.6, 1.0])
    assert "expectancy" in e and "mean_log" in e, (
        "a per-trade mean must not be computable without its mean log")
    assert e["expectancy"] > 0 > e["mean_log"], (
        "the fixture must keep the two free to disagree in sign, or the "
        "pairing is untested")


def test_the_power_statement_precedes_any_ratio():
    """A31 found it the most useful line in that study; A19 records writing it
    AS an effect statement as its own error. Ordering it first is the cheapest
    guard against reading it as one."""
    import numpy as np
    from idxbot.metrics import describe, summary
    rng = np.random.default_rng(0)
    txt = describe(summary(rng.normal(0.001, 0.02, 500), 252))
    assert txt.splitlines()[0].startswith("POWER")
    assert txt.index("POWER") < txt.index("Sharpe")


def test_every_shipped_surface_says_the_holdout_is_spent():
    """'No level quoted without the benchmark' and 'nothing validated' both
    reduce to the same disclosure, and a surface that omits it is quotable
    without it."""
    for name in ("rules.py", "today.py", "daily_signal.py"):
        src = open(os.path.join(ROOT, "scripts", name)).read()
        assert "H16" in src, f"{name} never says the holdout is spent"


def test_the_standing_contract_is_still_in_claude_md():
    """The prohibitions these tests enforce must remain stated where a reader
    meets them, not survive only as assertions in a test file."""
    md = open(os.path.join(ROOT, "CLAUDE.md")).read()
    assert "Nothing described as validated" in md
    assert "No per-trade mean without the mean LOG beside it" in md
    assert "No win rate without its horizon and its threshold" in md
    assert "No level quoted without the benchmark" in md
