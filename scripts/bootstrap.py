#!/usr/bin/env python3
"""Everything `scripts/today.py` needs, rebuilt from an empty `data/`.

WHY THIS EXISTS, AND IT IS NOT HOUSEKEEPING. This repo runs in an ephemeral
container and has already lost `data/` once — the postscript to
`docs/PLATFORM_ASSESSMENT.md` records 1.9 GB going to 4.5 MB, taking the
broker store with it permanently. Today `positions.py` failed to start because
`indicator_panel.parquet` was gone and nothing said which command rebuilds it.

**A bot that cannot be run tomorrow is not a bot**, and the difference between
"the data is missing" and "the data is unrecoverable" is a rebuild path that
someone has actually executed. So the dependency list below is DECLARED, each
entry names the command that produces it, and a test asserts the list matches
what the shipped surfaces actually open — a stale list is worse than none,
because it reads as a guarantee.

WHAT SURVIVES A REBUILD AND WHAT DOES NOT
  survives   `data/signals/*` — the append-only record is TRACKED IN GIT, and
             it has to be: it is the only out-of-sample evidence this project
             will ever have, and re-deriving it is definitionally impossible.
  survives   `data/reference/idx_classification.parquet`, tracked, 42 KB.
  rebuilds   the OHLCV cache (~860 files), the spine panel (258 MB) and the
             indicator panel (218 MB) — derived, large, gitignored, and each
             reproducible from the network in the order below.
  GONE       `data/cache/broker_daily`. Re-accumulating it needs a host in
             `data.broker_allowed_hosts`, which ships EMPTY and is the USER's
             call (A5). Gate 0 no longer depends on it, and this script says so
             rather than leaving the reader to discover it.

The order is not arbitrary: each step reads the previous one's output.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: (artefact, produced by, why it is needed, tracked in git)
ARTEFACTS: Tuple[Tuple[str, str, str, bool], ...] = (
    ("data/reference/idx_classification.parquet", None,
     "sector-mix disclosure on the rule card (H59)", True),
    ("data/signals/emitted.csv.gz", None,
     "the append-only forward record — the ONLY out-of-sample evidence", True),
    ("data/cache/ohlcv/_JKSE.csv.gz", "refresh.py",
     "the index benchmark every result is measured against (A19)", False),
    ("data/spine/price_panel.parquet", "refresh.py --panel",
     "the spine every surface reads", False),
    ("data/spine/indicator_panel.parquet", "build_indicators.py",
     "EMA/ATR/stochastic levels for the position monitor", False),
)

#: The rebuild, in dependency order. Each reads the previous one's output.
STEPS: Tuple[Tuple[str, List[str], str], ...] = (
    ("OHLCV cache + spine panel",
     ["refresh.py", "--panel"],
     "~5 min for ~860 names at the polite fetch delay"),
    ("indicator panel",
     ["build_indicators.py"],
     "~2 min, reads the spine panel"),
    ("brief tables",
     ["refresh.py", "--tables"],
     "conditional tables the daily brief reads"),
    ("today's signals, both rules, into the append-only store",
     ["refresh.py", "--signals"],
     "quarterly card + daily bracket scan"),
)


def status() -> List[Dict]:
    out = []
    for path, by, why, tracked in ARTEFACTS:
        full = os.path.join(ROOT, path)
        ok = os.path.exists(full)
        out.append({"path": path, "ok": ok,
                    "mb": (os.path.getsize(full) / 1e6) if ok else 0.0,
                    "by": by, "why": why, "tracked": tracked})
    return out


def report(rows: List[Dict]) -> str:
    L = [f"  {'artefact':<45}{'MB':>9}  {'source':<26}why", "  " + "-" * 110]
    for r in rows:
        mark = "OK  " if r["ok"] else "MISS"
        src = "tracked in git" if r["tracked"] else (r["by"] or "?")
        L.append(f"  {mark} {r['path']:<40}{r['mb']:>9.1f}  {src:<26}"
                 f"{r['why']}")
    miss = [r for r in rows if not r["ok"]]
    L.append("")
    if miss:
        L.append(f"  {len(miss)} missing. Rebuild with: "
                 f"python3 scripts/bootstrap.py --run")
        untracked = [r for r in miss if r["tracked"]]
        if untracked:
            L.append("  *** " + ", ".join(r["path"] for r in untracked)
                     + " is TRACKED IN GIT and missing — that is a working-copy")
            L.append("      problem, not a rebuild one. `git checkout` it "
                     "rather than regenerating,")
            L.append("      because the forward record cannot be re-derived.")
    else:
        L.append("  everything present.")
    return "\n".join(L)


def run(cmd: List[str]) -> int:
    print(f"\n$ python3 scripts/{' '.join(cmd)}", flush=True)
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, cmd[0])] + cmd[1:],
                       cwd=ROOT)
    print(f"  -> exit {p.returncode} in {time.time() - t0:.0f}s", flush=True)
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true",
                    help="actually rebuild; without it this only reports")
    a = ap.parse_args()

    print("=" * 112)
    print(" BOOTSTRAP — what scripts/today.py needs, and whether it is here")
    print("=" * 112)
    rows = status()
    print(report(rows))

    print("\n  NOT REBUILDABLE FROM HERE, and it is not a bug:")
    print("    data/cache/broker_daily — re-accumulating needs a host in")
    print("    data.broker_allowed_hosts, which ships EMPTY and is the USER's")
    print("    call after checking its licensing (A5). Gate 0 check 1 no")
    print("    longer depends on it: it reconciles against the IDX daily")
    print("    summary instead, on 838,227 ticker-days against 814 names.")

    if not a.run:
        print("\n  (report only; pass --run to rebuild)")
        return 0 if all(r["ok"] for r in rows) else 1

    rc = 0
    for label, cmd, note in STEPS:
        print("\n" + "-" * 112)
        print(f" {label}   ({note})")
        print("-" * 112)
        rc |= run(cmd)

    print("\n" + "=" * 112)
    rows = status()
    print(report(rows))
    still = [r["path"] for r in rows if not r["ok"]]
    if still:
        print(f"\n  STILL MISSING AFTER A FULL REBUILD: {', '.join(still)}")
        print("  That is a real failure, not a slow step — read the exit codes "
              "above.")
        return 1
    print("\n  ready:  python3 scripts/today.py")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
