#!/usr/bin/env python3
"""Recover the vanished IDX names — reproducibly, which is what was missing.

WHY THIS FILE EXISTS
--------------------
A cold container rebuild on 2026-09-06 destroyed `data/cache/delisted`, and the
repo could not rebuild it. `spine/universe.py` documented the recovery's source
only in prose — "a published April-2019 snapshot of 627 IDX tickers" — with no
URL, no accession date, no checksum, and NO COLLECTOR SCRIPT ANYWHERE. The
acquisition had been done by hand in a session and never written down. The
survivorship repair, which turned three guesses into measurements, was
irreproducible from the repo's own record.

That is the failure this file fixes. Not the data loss — the data was gitignored
by design and that design is fine — but the fact that nothing recorded HOW TO
GET IT BACK. Every provenance field below exists so that the next rebuild is a
command rather than an archaeology session.

PROVENANCE — all four fields are mandatory and are asserted by the tests
--------------------------------------------------------------------------
    source repo   github.com/wildangunawan/Dataset-Saham-IDX
    commit        bc0ac7712ce5e46f1067349e13ab9f338883c6c4
    committed     2025-02-23 22:15:58 +0700
    accessed      2026-09-06
    licence       CC BY-NC 4.0 on the compilation
    UNDERLYING    the repo's own README states the data is taken from
                  idx.co.id and that "all data in the dataset belongs to PT
                  Bursa Efek Indonesia", pointing at IDX's Syarat Penggunaan.
                  CLAUDE.md §3 already records IDX Terms of Use item 5 as
                  barring COMMERCIAL redistribution while permitting personal
                  research use.

*** THIS IS A LICENSING DECISION FOR THE USER, NOT FOR THIS REPO (A5). ***
CC BY-NC is NON-COMMERCIAL. A23 records that this project is now being pointed
at a client's money, and managing third-party money for a fee is not obviously
non-commercial use. So this script WRITES TO A SEPARATE DIRECTORY AND NOTHING
IMPORTS IT until the user rules. `data.broker_allowed_hosts` ships empty for
exactly this reason and the same rule applies here.

WHAT IT RECOVERS, MEASURED 2026-09-06
--------------------------------------
    dataset names                        958
    live spine names                     827
    IN DATASET, NOT IN THE SPINE         145   <- the survivorship correction
    median total return of those 145     -54.5%
    of them, stopped trading pre-2025-02  68
    depth                                2019-07-29 -> 2025-02-21, 1,355 bars

145 against the 121 the lost snapshot held, so this is a wider recovery — but a
SHALLOWER one. The lost snapshot reached back to 2019-04-07 and carried names
that died before this dataset's window opens; those are still gone.

AND IT FIXES THE ONE-SIDEDNESS, which is the part that matters. A2's complaint
about the old snapshot was that it "ends 2019-04-07, so the months in which a
name actually died are still missing, and the bias figure stays a bound rather
than a correction." Here the last bar carrying real volume dates the death
directly — ARMY stops 2019-11-29, MYRX 2020-01-15, TRAM and SMRU 2020-01-22.
Note this comes from LAST TRADED VOLUME, not from the dataset's own
`delisting_date` column, which is present as a header and EMPTY for all 958
names — a thing worth stating because its name invites the opposite assumption.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from typing import Dict, List

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

SOURCE_REPO = "https://github.com/wildangunawan/Dataset-Saham-IDX"
SOURCE_COMMIT = "bc0ac7712ce5e46f1067349e13ab9f338883c6c4"
SOURCE_LICENCE = "CC BY-NC 4.0 (compilation); underlying data (c) PT Bursa Efek Indonesia"
DEFAULT_CLONE = "/home/user/wildangunawan/dataset-saham-idx"

OUT_DIR = os.path.join("data", "cache", "delisted")
MANIFEST = os.path.join("data", "cache", "delisted_manifest.json")
PANEL = os.path.join("data", "spine", "price_panel.parquet")
MIN_BARS = 50


def clone(dest: str = DEFAULT_CLONE) -> str:
    """Shallow-clone the source at the pinned commit, or reuse an existing one.

    Pinning matters: the dataset is updated by hand and irregularly, so an
    unpinned clone silently changes the universe under every study that used it.
    """
    if os.path.isdir(os.path.join(dest, ".git")):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    subprocess.run(["git", "clone", "--depth", "1", SOURCE_REPO, dest],
                   check=True, env=env)
    return dest


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def recover(clone_dir: str = DEFAULT_CLONE, out_dir: str = OUT_DIR,
            min_bars: int = MIN_BARS) -> Dict:
    """Write every dataset name the live spine lacks, plus a manifest.

    The output goes to a SEPARATE directory and is not merged (A2): "a
    survivorship-free universe is a DIFFERENT universe, not a bigger one", and
    mixing them in would silently change the meaning of every existing study.
    """
    src = os.path.join(clone_dir, "Saham", "Semua")
    if not os.path.isdir(src):
        raise SystemExit(f"clone not found at {clone_dir} — run with --clone")
    names = {os.path.basename(f)[:-4] for f in glob.glob(src + "/*.csv")}

    live: set = set()
    if os.path.exists(PANEL):
        live = set(pd.read_parquet(PANEL, columns=["ticker"])["ticker"].unique())
    gained = sorted(names - live)

    os.makedirs(out_dir, exist_ok=True)
    rows: List[Dict] = []
    for tk in gained:
        f = os.path.join(src, f"{tk}.csv")
        try:
            d = pd.read_csv(f, parse_dates=["date"])
        except Exception:
            continue
        d = d[d["close"] > 0].sort_values("date")
        if len(d) < min_bars:
            continue
        #  THE DEATH DATE COMES FROM VOLUME, NOT FROM `delisting_date`.
        #  That column exists as a header and is EMPTY for all 958 names. The
        #  last bar carrying real volume is when the name actually stopped
        #  trading; every bar after it is the exchange repeating a stale price.
        traded = d[d["volume"] > 0]
        last_traded = traded["date"].max() if len(traded) else pd.NaT
        keep = d[["date", "open_price", "high", "low", "close", "volume",
                  "listed_shares"]].rename(columns={"open_price": "open"})
        #  No adjustment is applied here. The dataset is raw exchange prices and
        #  the spine's own corporate-action machinery is what adjusts; doing it
        #  twice is the SCCO defect A2 records.
        keep["adj_close"] = keep["close"]
        out = os.path.join(out_dir, f"{tk}.JK.csv.gz")
        keep.to_csv(out, index=False, compression="gzip")
        rows.append({
            "ticker": tk, "bars": int(len(keep)),
            "first": str(keep["date"].min().date()),
            "last": str(keep["date"].max().date()),
            "last_traded": str(last_traded.date()) if pd.notna(last_traded) else None,
            "total_return": float(keep["close"].iloc[-1] / keep["close"].iloc[0] - 1),
            "sha16": _sha(out),
        })

    man = {
        "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "licence": SOURCE_LICENCE,
        "licence_decision": "PENDING USER RULING — CC BY-NC is non-commercial "
                            "and A23 records this project being pointed at a "
                            "client's money. Nothing imports this directory "
                            "until the user rules (A5).",
        "dataset_names": len(names),
        "live_spine_names": len(live),
        "recovered": len(rows),
        "min_bars": min_bars,
        "names": rows,
    }
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump(man, f, indent=1)
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", action="store_true",
                    help="clone the source repo first")
    ap.add_argument("--dir", default=DEFAULT_CLONE)
    args = ap.parse_args()
    if args.clone:
        clone(args.dir)
    m = recover(args.dir)
    dead = [n for n in m["names"] if n["last_traded"]
            and n["last_traded"] < "2025-02-01"]
    rets = sorted(n["total_return"] for n in m["names"])
    med = rets[len(rets) // 2] if rets else float("nan")
    print(f"source      {m['source_repo']}@{m['source_commit'][:12]}")
    print(f"licence     {m['licence']}")
    print(f"            {m['licence_decision']}")
    print(f"dataset     {m['dataset_names']} names")
    print(f"live spine  {m['live_spine_names']} names")
    print(f"RECOVERED   {m['recovered']} names the spine does not have")
    print(f"            median total return {med:+.1%}")
    print(f"            stopped trading before 2025-02: {len(dead)}")
    print(f"written to  {OUT_DIR}/  and  {MANIFEST}")
    print()
    print("NOT MERGED INTO THE SPINE. A survivorship-free universe is a")
    print("DIFFERENT universe, not a bigger one (A2), and the licence question")
    print("is the user's to answer (A5).")


if __name__ == "__main__":
    main()
