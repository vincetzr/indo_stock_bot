"""DuckDB over the files that already exist. No migration, no second copy.

WHY A DATABASE AT ALL, AND WHY NOT A MIGRATION.
`price_panel.parquet` is 258 MB and every script that touches it starts with
`pd.read_parquet(PANEL)` — the whole panel into memory to answer a question
about eight columns and one year. DuckDB reads Parquet in place with predicate
and projection pushdown, so the same question touches a fraction of the file.

The tempting next step is to COPY everything into a `.duckdb` file, and that is
the step this module refuses. A second copy of the spine is a second thing that
can drift from the first, and this repo has already lost a store once to a
container rebuild (A-postscript) and a module's entire test coverage to a
filename collision (A19). Views over the source files cannot drift, because
there is nothing to drift from: the Parquet file IS the table.

THE ONE THING THAT MAKES THIS SAFE IS `verify()`.
A query engine that returns *different* numbers from the pandas path is worse
than no query engine, and the difference would be silent — a null handled
differently, a float64 read as decimal, a date normalised in another timezone.
`verify()` computes the same aggregates both ways and reports any cell that
disagrees. It is a test, and it is also a function callers can run against
their own query before believing it.

WHAT IS DELIBERATELY NOT HERE. No write path. Nothing in this repo should
learn to mutate the spine through SQL: the spine is built by
`scripts/refresh.py` and repaired by a registry that is auditable and
reversible (`spine/repairs.py`). A SQL UPDATE would be neither.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

#: (view name, path relative to the repo root, reader)
#: A view is registered only if its file exists, so a partial checkout or a
#: fresh container gives a smaller catalogue rather than an import error.
SOURCES: Tuple[Tuple[str, str, str], ...] = (
    ("prices", "data/spine/price_panel.parquet", "parquet"),
    ("shares_pit", "data/spine/shares_pit.parquet", "parquet"),
    ("flow_panel", "data/spine/flow_panel.csv.gz", "csv"),
    ("investor_split", "data/spine/investor_split.csv.gz", "csv"),
    ("fingerprints", "data/spine/fingerprints.csv.gz", "csv"),
    ("sectors", "data/reference/idx_classification.parquet", "parquet"),
    ("signals", "data/signals/emitted.csv.gz", "csv"),
    ("signal_outcomes", "data/signals/outcomes.csv.gz", "csv"),
)


def _root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        os.pardir, os.pardir))


def available(root: Optional[str] = None) -> List[str]:
    """View names whose backing file is actually on disk."""
    r = root or _root()
    return [n for n, p, _k in SOURCES if os.path.exists(os.path.join(r, p))]


def connect(root: Optional[str] = None, read_only: bool = True):
    """An in-memory DuckDB with a view per source file.

    `read_only` is a promise about intent, not a DuckDB flag: the connection is
    in-memory, so nothing here can write to the source files whatever SQL is
    run. It is kept in the signature so a future caller asking for write access
    has to change something visible.
    """
    import duckdb                                            # noqa: PLC0415

    if not read_only:
        raise ValueError(
            "this module has no write path by design — the spine is built by "
            "scripts/refresh.py and repaired by an auditable registry, and a "
            "SQL UPDATE would be neither")
    r = root or _root()
    con = duckdb.connect(":memory:")
    for name, rel, kind in SOURCES:
        path = os.path.join(r, rel)
        if not os.path.exists(path):
            continue
        fn = "read_parquet" if kind == "parquet" else "read_csv_auto"
        con.execute(
            f"CREATE OR REPLACE VIEW {name} AS "
            f"SELECT * FROM {fn}('{path}')")
    return con


_CON: Dict[str, object] = {}


def sql(query: str, root: Optional[str] = None) -> pd.DataFrame:
    """Run one query against the views and return a DataFrame.

    THE CONNECTION IS CACHED AND THAT IS THE WHOLE PERFORMANCE STORY.
    A first version opened a fresh connection per call, which re-registers
    eight views and re-opens a 258 MB Parquet file every time. Measured on this
    panel, that made DuckDB **2.5x to 35x SLOWER than pandas** on every query
    tried -- the exact opposite of the reason to adopt it, and it printed
    correct answers the whole time. With the connection reused the same queries
    run **3.3x to 29x FASTER**:

        one ticker, two columns   0.008s vs pandas 0.083s   10.4x
        one date, cross-section   0.019s vs        0.049s    2.6x
        group-by, all names       0.020s vs        0.114s    5.6x
        count only                0.002s vs        0.026s   15.3x

    AND THE HEADLINE NUMBER IS NOT UNIFORM, WHICH MATTERS MORE THAN ITS SIZE.
    `SELECT *` over the whole panel is roughly 35x SLOWER through DuckDB than
    `pd.read_parquet`, because materialising 2.8m x 40 back into a DataFrame is
    pure overhead when nothing was filtered away. So this is a win for
    SELECTIVE queries and a loss for "give me everything", which is exactly
    what every existing script does. `docs/PLATFORM_ASSESSMENT.md` claimed
    "13-28x faster, no migration" with no measurement behind it; the range is
    real for the queries above and does not license rewriting the panel loads.

    `benchmark()` reproduces that table, because a performance claim nobody can
    re-run is the same kind of claim this repo spends its time deleting.
    """
    r = os.path.abspath(root or _root())
    con = _CON.get(r)
    if con is None:
        con = connect(r)
        _CON[r] = con
    return con.execute(query).fetchdf()


def close() -> None:
    """Drop cached connections. Only needed when the source files change."""
    for c in list(_CON.values()):
        try:
            c.close()
        except Exception:                                    # noqa: BLE001
            pass
    _CON.clear()


#: Aggregates computed identically in SQL and in pandas. Each entry is
#: (label, sql, callable over the pandas frame). Adding a row here is how a new
#: view earns the right to be trusted.
def _checks() -> Tuple[Tuple[str, str, str, callable], ...]:
    return (
        ("prices: row count", "prices",
         "SELECT COUNT(*) AS v FROM prices",
         lambda d: float(len(d))),
        ("prices: distinct tickers", "prices",
         "SELECT COUNT(DISTINCT ticker) AS v FROM prices",
         lambda d: float(d["ticker"].nunique())),
        ("prices: sum adj_close", "prices",
         "SELECT SUM(adj_close) AS v FROM prices",
         lambda d: float(d["adj_close"].sum())),
        ("prices: max date", "prices",
         "SELECT CAST(MAX(date) AS VARCHAR) AS v FROM prices",
         lambda d: str(pd.Timestamp(d["date"].max()).date())),
        ("prices: tradeable share", "prices",
         "SELECT AVG(CAST(tradeable AS DOUBLE)) AS v FROM prices",
         lambda d: float(d["tradeable"].astype(float).mean())),
        ("shares_pit: row count", "shares_pit",
         "SELECT COUNT(*) AS v FROM shares_pit",
         lambda d: float(len(d))),
        ("shares_pit: names that ever change", "shares_pit",
         "SELECT COUNT(*) AS v FROM (SELECT ticker FROM shares_pit "
         "GROUP BY ticker HAVING COUNT(DISTINCT listed_shares) > 1)",
         lambda d: float((d.groupby("ticker")["listed_shares"].nunique() > 1)
                         .sum())),
    )


def verify(root: Optional[str] = None, tol: float = 1e-6
           ) -> List[Dict[str, object]]:
    """Compare each aggregate computed in SQL against the pandas path.

    Returns one row per check with both values and whether they agree. A query
    engine that quietly disagrees with the path every existing study used is
    worse than no query engine, and the disagreement would not announce itself.
    """
    r = root or _root()
    have = set(available(r))
    paths = {n: os.path.join(r, p) for n, p, _k in SOURCES}
    kinds = {n: k for n, _p, k in SOURCES}
    out: List[Dict[str, object]] = []
    frames: Dict[str, pd.DataFrame] = {}
    con = connect(r)
    try:
        for label, view, q, fn in _checks():
            if view not in have:
                out.append({"check": label, "view": view, "ok": None,
                            "why": "source file absent"})
                continue
            if view not in frames:
                frames[view] = (pd.read_parquet(paths[view])
                                if kinds[view] == "parquet"
                                else pd.read_csv(paths[view]))
            got = con.execute(q).fetchone()[0]
            want = fn(frames[view])
            if isinstance(want, str) or isinstance(got, str):
                ok = str(got)[:10] == str(want)[:10]
            else:
                got = float(got)
                ok = abs(got - float(want)) <= tol * max(1.0, abs(float(want)))
            out.append({"check": label, "view": view, "duckdb": got,
                        "pandas": want, "ok": bool(ok)})
    finally:
        con.close()
    return out


#: (label, SQL, pandas columns, pandas callable). The pandas arm reads only the
#: columns it needs, so the comparison is fair: it is not being handicapped by
#: loading a panel the SQL arm projected away.
BENCH: Tuple[Tuple[str, str, Tuple[str, ...], callable], ...] = (
    ("one ticker, 2 cols",
     "SELECT date, adj_close FROM prices WHERE ticker='BBCA' ORDER BY date",
     ("date", "ticker", "adj_close"),
     lambda d: d[d["ticker"] == "BBCA"]),
    ("one date cross-section",
     "SELECT ticker, adj_close FROM prices WHERE date='2020-01-02'",
     ("date", "ticker", "adj_close"),
     lambda d: d[d["date"] == "2020-01-02"]),
    ("group-by, all names",
     "SELECT ticker, COUNT(*) n, AVG(adj_close) m FROM prices GROUP BY ticker",
     ("ticker", "adj_close"),
     lambda d: d.groupby("ticker")["adj_close"].agg(["size", "mean"])),
    ("count only", "SELECT COUNT(*) FROM prices", ("ticker",),
     lambda d: len(d)),
)


def benchmark(root: Optional[str] = None, repeats: int = 3
              ) -> List[Dict[str, object]]:
    """Time each query both ways. A performance claim nobody can re-run is not
    a measurement, and the first version of `sql()` was 2.5-35x SLOWER while
    returning perfectly correct answers."""
    import time                                              # noqa: PLC0415

    r = root or _root()
    path = os.path.join(r, "data/spine/price_panel.parquet")
    if not os.path.exists(path):
        return []
    con = connect(r)
    con.execute("SELECT COUNT(*) FROM prices").fetchone()     # warm
    out = []
    try:
        for label, q, cols, fn in BENCH:
            a = min(_time(lambda: con.execute(q).fetchdf(), time)
                    for _ in range(repeats))
            b = min(_time(lambda: fn(pd.read_parquet(path, columns=list(cols))),
                          time) for _ in range(repeats))
            out.append({"query": label, "duckdb_s": a, "pandas_s": b,
                        "speedup": (b / a) if a > 0 else float("nan")})
    finally:
        con.close()
    return out


def _time(fn, time_mod) -> float:
    t0 = time_mod.time()
    fn()
    return time_mod.time() - t0


def describe(rows: List[Dict[str, object]]) -> str:
    L = []
    for r in rows:
        if r.get("ok") is None:
            L.append(f"  SKIP  {r['check']:<40} {r.get('why')}")
            continue
        mark = "ok  " if r["ok"] else "FAIL"
        L.append(f"  {mark}  {r['check']:<40} duckdb={r['duckdb']!r:>22}  "
                 f"pandas={r['pandas']!r}")
    bad = [r for r in rows if r.get("ok") is False]
    L.append(f"\n  {len(bad)} of {len(rows)} checks disagree")
    if bad:
        L.append("  A disagreement here means SQL and pandas are reading the "
                 "same file differently.")
        L.append("  Do not use the SQL path until it is explained.")
    return "\n".join(L)
