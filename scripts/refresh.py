#!/usr/bin/env python3
"""Refresh the WHOLE universe, then rebuild what depends on it.

    python3 scripts/refresh.py                 # bars only
    python3 scripts/refresh.py --panel           # bars, then rebuild the panel
    python3 scripts/refresh.py --panel --tables  # ... and the brief's tables

    # the one command to run each morning and each evening:
    python3 scripts/refresh.py --panel --brief pre
    python3 scripts/refresh.py --panel --brief post

Tables only need rebuilding when the panel changes materially — weekly is
plenty — so the daily runs above leave --tables off and take about six
minutes rather than ten.

WHY THIS EXISTS
----------------
`scripts/daily_update.py` refreshes a large-cap watchlist — 40-60 names — and
that is all it was ever meant to do. Nothing else refreshed the rest, so the
panel accumulated a ragged edge: 830 names through 2026-08-14, then 46 names
for four more sessions. The brief detects that and falls back (see
`brief.coverage_warning`), which is the right behaviour but leaves it
permanently ten days stale. This is the other half of the fix.

RATE
-----
Measured at ~0.34 s per ticker against Yahoo's chart endpoint, so the full
843-name universe takes about five minutes. Sequential on purpose: the loader
already backs off on 429 and parallel requests are the fastest way to earn
more of them.

WHAT "FAILED" MEANS HERE
-------------------------
A name can come back empty for three quite different reasons and the summary
separates them, because they call for different responses:

    suspended   the name exists and did not trade. ADHI on 2026-08-24 is the
                worked example — BEI halted it over a missed bond coupon, and
                the missing bar is correct rather than broken.
    delisted    gone from Yahoo entirely. Belongs in the delisted store, and
                §5 is blunt that dropping it instead is survivorship bias.
    error       the request failed. Worth retrying; nothing else here is.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import subprocess
import sys
import time
from typing import Dict, List

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config                          # noqa: E402
from idxbot.data.cache import Cache                            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV                       # noqa: E402

WIB = dt.timezone(dt.timedelta(hours=7))
LIVE = os.path.join("data", "cache", "ohlcv")
TICKER_FILE = os.path.join("data", "idx_all_tickers.txt")


def universe() -> List[str]:
    """Every live name: what is already cached, plus the published list.

    The union rather than either alone. The cache alone would never add a new
    listing; the ticker file alone would silently drop a name that was cached
    from some other route.
    """
    have = {os.path.basename(p).replace(".JK.csv.gz", "")
            for p in glob.glob(os.path.join(LIVE, "*.JK.csv.gz"))}
    listed = set()
    if os.path.exists(TICKER_FILE):
        with open(TICKER_FILE) as fh:
            for line in fh:
                t = line.strip().split(",")[0].strip().upper()
                if t and t.isalpha() and len(t) == 4:
                    listed.add(t)
    return sorted(have | listed)


def refresh(tickers: List[str], max_age: float,
            report_every: int = 100) -> Dict[str, object]:
    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    stamps: Dict[str, pd.Timestamp] = {}
    empty: List[str] = []
    errors: List[str] = []
    t0 = time.time()
    for i, t in enumerate(tickers, 1):
        try:
            d = loader.get(t, max_age=max_age)
        except Exception as exc:                                # noqa: BLE001
            errors.append(f"{t}: {type(exc).__name__}")
            continue
        if d is None or d.empty or "date" not in d:
            empty.append(t)
            continue
        stamps[t] = pd.to_datetime(d["date"]).max()
        if i % report_every == 0:
            rate = i / max(time.time() - t0, 1e-9)
            left = (len(tickers) - i) / max(rate, 1e-9)
            print(f"   {i}/{len(tickers)}  {rate:.1f}/s  "
                  f"~{left/60:.1f} min left", flush=True)
    return {"stamps": stamps, "empty": empty, "errors": errors,
            "seconds": time.time() - t0}


def coverage(stamps: Dict[str, pd.Timestamp]) -> pd.DataFrame:
    """Names per last-bar date, most recent first."""
    if not stamps:
        return pd.DataFrame(columns=["last_bar", "n_names"])
    s = pd.Series(stamps)
    c = s.value_counts().sort_index(ascending=False)
    return pd.DataFrame({"last_bar": c.index.date, "n_names": c.to_numpy()})


def run(cmd: List[str]) -> int:
    print(f"\n $ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-age", type=float, default=3600.0,
                    help="seconds; a cache entry younger than this is kept")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N names (for a smoke test)")
    ap.add_argument("--panel", action="store_true",
                    help="rebuild the price panel afterwards")
    ap.add_argument("--tables", action="store_true",
                    help="rebuild the brief's conditional tables afterwards")
    ap.add_argument("--brief", choices=["pre", "post"], default=None,
                    help="run the brief afterwards and save md + html")
    a = ap.parse_args()

    now = dt.datetime.now(tz=WIB)
    U = universe()
    if a.limit:
        U = U[:a.limit]
    print("=" * 78)
    print(f" UNIVERSE REFRESH — {now:%Y-%m-%d %H:%M} WIB, {len(U)} names")
    print("=" * 78)
    r = refresh(U, a.max_age)
    stamps = r["stamps"]
    print(f"\n {len(stamps)}/{len(U)} returned bars in "
          f"{r['seconds']/60:.1f} min")

    C = coverage(stamps)
    print("\n names by last bar:")
    for x in C.head(6).itertuples():
        print(f"   {x.last_bar}   {x.n_names}")
    if len(C) > 6:
        print(f"   ... and {len(C)-6} older dates")

    if len(C):
        newest, n_new = C.iloc[0]["last_bar"], int(C.iloc[0]["n_names"])
        share = n_new / max(len(stamps), 1)
        print(f"\n {n_new} names ({share:.0%}) have a bar on {newest}")
        if share < 0.80:
            print(" ! under 80% — the brief will fall back to an earlier "
                  "session rather than\n !   report this one as a "
                  "cross-section. That is usually a half-finished\n "
                  "!   refresh or a market holiday, not a fault.")

    if r["empty"]:
        print(f"\n {len(r['empty'])} returned nothing: "
              f"{' '.join(r['empty'][:25])}"
              + (" ..." if len(r["empty"]) > 25 else ""))
        print("   suspended, delisted or never listed — see the module "
              "docstring. A\n   delisted name belongs in data/cache/delisted, "
              "not dropped (§5).")
    if r["errors"]:
        print(f"\n {len(r['errors'])} errored: {' '.join(r['errors'][:15])}")
        print("   these are worth retrying; nothing else above is.")

    rc = 0
    if a.panel:
        rc |= run([sys.executable, "scripts/price_panel_build.py"])
    if a.tables:
        rc |= run([sys.executable, "scripts/brief.py", "--build-tables",
                   "--no-news"])
    if a.brief:
        os.makedirs("reports", exist_ok=True)
        rc |= run([sys.executable, "scripts/brief.py",
                   "--session", a.brief, "--save",
                   "--html", os.path.join("reports", "brief_latest.html")])
    print("\n done." if rc == 0 else "\n done, with a failing rebuild step.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
