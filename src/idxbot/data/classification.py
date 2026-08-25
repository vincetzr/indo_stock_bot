"""IDX-IC sector, listing board and share count — the classification the brief
had to work without.

WHY THIS IS `classification.py` AND NOT `sectors.py`
------------------------------------------------------
`idxbot.data.sectors` is already the client for the licensed Sectors.app API
and needs a paid key. Reusing the name for a free, unrelated source would be a
genuinely confusing collision in a repo where the licensing distinction is
load-bearing.

WHAT THIS FIXES
----------------
The brief printed "67% of names above the 20-day" and could not say which
names. Sector breadth — "Energy 91%, Financials 42%" — is the single most-asked
situational question about a market and it was unanswerable, because §7 records
that this repo has no sector data and substitutes trailing principal
components.

That substitution is good and stays: `comovement()` measures which names
actually trade together rather than asserting which ones ought to. What a
sector map adds is the ability to CHECK it — "PC2's positive tail is 7 of 8
Financials" validates the component, and a component that maps to no sector at
all is the more interesting reading.

THE SOURCE, AND WHY THIS ONE
------------------------------
Eleven CSVs under `wildangunawan/Dataset-Saham-IDX`, one per official IDX-IC
sector, header ``code,name,listingDate,shares,listingBoard``. 934 tickers,
95.1% of this repo's universe. **CC BY-NC 4.0** — an explicit, published
licence permitting exactly this use with attribution, which is why it is built
and TradingView's screener is not: that one is an undocumented internal
endpoint whose `/indonesia/scan` path its own robots.txt disallows, and A5 is
explicit that a host is added only after the user has checked its licensing.

idx.co.id's own list returns 403 behind Cloudflare, exactly as
`docs/FULL_REKAP.md` records, so the authoritative taxonomy is only available
second-hand.

THREE LIMITS, ALL LOAD-BEARING
--------------------------------
**It is frozen at 2024-07-10.** The 41 names listed since then get no sector.
They are marked ``unclassified`` and counted in the output rather than
back-filled from a different taxonomy — TradingView's scheme is FactSet/RBICS,
the crosswalk to IDX-IC is many-to-many, and it returns ASII, a top-ten name,
as "Technology Services".

**`shares` is stale by the same margin**, so a market cap computed from it is
wrong after any rights issue or split — and this repo has a whole memo on
rights issues being the adjustment trap. It is exposed, and it is not used to
build a cap-weighted index.

**It is current-state only, with no history**, so it can never enter a
backtest. A name's sector today is not its sector in 2015, and the file has no
delisted names at all. `tests/test_classification.py` walks the AST of
`spine/` and `features/` and fails if either imports this, the same guard
`news.py` carries.
"""

from __future__ import annotations

import io
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Sequence

import pandas as pd

#: The eleven official IDX-IC sectors, exactly as the source names its files.
SECTORS: Sequence[str] = (
    "Energy", "Basic Materials", "Industrials", "Consumer Non-Cyclicals",
    "Consumer Cyclicals", "Healthcare", "Financials",
    "Properties & Real Estate", "Technology", "Infrastructures",
    "Transportation & Logistic",
)

BASE = ("https://raw.githubusercontent.com/wildangunawan/Dataset-Saham-IDX/"
        "master/List%20Emiten/Sectors")

HOST = "raw.githubusercontent.com"

#: Printed with any output derived from this file. CC BY-NC 4.0 requires
#: attribution, and the underlying data is IDX's.
ATTRIBUTION = (
    "IDX-IC sector and listing board from wildangunawan/Dataset-Saham-IDX "
    "(CC BY-NC 4.0); underlying data © PT Bursa Efek Indonesia. "
    "Non-commercial research use. Snapshot frozen 2024-07-10.")

CACHE = os.path.join("data", "reference", "idx_classification.parquet")
UA = "idxbot/1.0 (personal research; contact via repository)"
DELAY = 0.4

#: Listing boards as the source spells them, mapped to the names
#: `spine.reference` uses for its auto-rejection ladders.
#:
#: A DISCREPANCY WORTH KNOWING BEFORE USING THIS FOR BANDS. This snapshot puts
#: 216 names on Pemantauan Khusus; `reference.infer_board`, applied to today's
#: prices, finds 41. Both are right about different things. IDX has eleven
#: criteria for that board — going-concern opinions, prolonged suspension, no
#: revenue — and only ONE, the six-month average price below Rp 51, is
#: derivable from a price series. So the price rule is a strict LOWER BOUND on
#: membership, and this file is a two-year-stale upper one.
#:
#: The brief keeps using the price rule for auto-rejection bands, because it is
#: point-in-time and this file is not, and a stale board assignment would band
#: a name on a status it may have left. The gap is the honest uncertainty.
BOARD_MAP = {
    "Utama": "main",
    "Pengembangan": "development",
    "Ekonomi Baru": "new_economy",
    "Akselerasi": "acceleration",
    "Pemantauan Khusus": "watchlist",
}


class HostNotAllowed(RuntimeError):
    pass


def _fetch(url: str, timeout: float = 25.0) -> Optional[bytes]:
    if urllib.parse.urlparse(url).netloc.lower() != HOST:
        raise HostNotAllowed(f"{url} is not {HOST}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            TimeoutError):
        return None


def download(sectors: Sequence[str] = SECTORS) -> pd.DataFrame:
    """One frame from the eleven per-sector CSVs.

    A sector that fails to fetch is omitted and reported by its absence from
    the result, rather than silently shrinking the map — a half-downloaded
    classification would quietly relabel a third of the market as
    ``unclassified``.
    """
    frames: List[pd.DataFrame] = []
    for s in sectors:
        body = _fetch(f"{BASE}/{urllib.parse.quote(s)}.csv")
        if not body:
            continue
        try:
            d = pd.read_csv(io.BytesIO(body))
        except Exception:                                       # noqa: BLE001
            continue
        if "code" not in d:
            continue
        d = d.rename(columns={"code": "ticker", "name": "company",
                              "listingDate": "listing_date",
                              "listingBoard": "listing_board_id"})
        d["sector"] = s
        frames.append(d)
        time.sleep(DELAY)
    if not frames:
        return pd.DataFrame(columns=["ticker", "company", "listing_date",
                                     "shares", "listing_board_id", "sector"])
    D = pd.concat(frames, ignore_index=True)
    D["ticker"] = D["ticker"].astype(str).str.strip().str.upper()
    D["listing_date"] = pd.to_datetime(D["listing_date"], errors="coerce")
    D["shares"] = pd.to_numeric(D.get("shares"), errors="coerce")
    D["board"] = D["listing_board_id"].map(BOARD_MAP).fillna("unknown")
    return D.drop_duplicates("ticker").reset_index(drop=True)


def load(rebuild: bool = False) -> pd.DataFrame:
    """The cached classification, downloading once if absent."""
    if os.path.exists(CACHE) and not rebuild:
        try:
            return pd.read_parquet(CACHE)
        except Exception:                                       # noqa: BLE001
            pass
    D = download()
    if D.empty:
        return D
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    tmp = CACHE + ".tmp"
    D.to_parquet(tmp, index=False)
    os.replace(tmp, CACHE)
    return D


def sector_of(D: pd.DataFrame) -> Dict[str, str]:
    if D.empty:
        return {}
    return dict(zip(D["ticker"], D["sector"]))


def coverage(D: pd.DataFrame, universe: Sequence[str]) -> Dict[str, object]:
    """How much of a universe this map actually classifies, and what it misses.

    Reported rather than assumed. The snapshot is frozen at 2024-07-10, so the
    gap is systematically the most recent listings — exactly the names a daily
    brief is most likely to be asked about.
    """
    have = set(D["ticker"]) if not D.empty else set()
    uni = {str(t).upper() for t in universe}
    missing = sorted(uni - have)
    return {"n_universe": len(uni), "n_classified": len(uni & have),
            "share": (len(uni & have) / len(uni)) if uni else 0.0,
            "missing": missing, "n_missing": len(missing),
            "n_in_map": len(have)}


def sector_breadth(S: pd.DataFrame, D: pd.DataFrame,
                   window: int = 20, min_names: int = 4) -> pd.DataFrame:
    """Share of each sector above its own N-day average, plus today's move.

    THE QUESTION THE BRIEF COULD NOT ANSWER. "67% of names above the 20-day" is
    a market statistic; "Energy 91%, Financials 42%" is the one a reader
    actually acts on, and it is the difference between a broad advance and a
    single sector carrying the tape.

    Takes the brief's own snapshot so the cross-section is the same one every
    other section used. Sectors with fewer than ``min_names`` present are
    dropped rather than reported on three tickers.
    """
    if D.empty or S.empty:
        return pd.DataFrame()
    col = f"ma{window}"
    if col not in S:
        return pd.DataFrame()
    s = S.copy()
    s["sector"] = s.index.map(sector_of(D))
    s["sector"] = s["sector"].fillna("unclassified")
    rows = []
    for sec, g in s.groupby("sector"):
        ok = g[col].notna() & g["adj_close"].notna()
        if int(ok.sum()) < min_names:
            continue
        r = g["ret1"]
        rows.append({
            "sector": sec, "n": int(ok.sum()),
            "above_ma": float((g["adj_close"][ok] > g[col][ok]).mean()),
            "median_move": float(r.median()) if r.notna().any() else float("nan"),
            "advancing": float((r > 0).mean()) if r.notna().any()
                         else float("nan")})
    return (pd.DataFrame(rows).sort_values("above_ma", ascending=False)
            .reset_index(drop=True))


def annotate_components(cm: pd.DataFrame, D: pd.DataFrame,
                        top: int = 8) -> pd.DataFrame:
    """Label each co-movement component by the sector its loadings concentrate in.

    THIS CHECKS THE PCA, IT DOES NOT REPLACE IT. §9.6's rule holds: the
    component is the observation and the sector name is an interpretation, so
    the label is reported as a COUNT — "6/8 Financials" — rather than as a
    title. A component whose heaviest names span five sectors gets no label,
    and that is the more interesting output: it means something is moving
    together that the taxonomy does not explain.
    """
    if cm.empty or D.empty:
        return cm
    smap = sector_of(D)
    out = cm.copy()
    labels, purity = [], []
    for _, r in out.iterrows():
        names = list(r["with"])[:top]
        secs = [smap.get(n) for n in names if smap.get(n)]
        if not secs:
            labels.append("")
            purity.append(float("nan"))
            continue
        top_sec = max(set(secs), key=secs.count)
        k = secs.count(top_sec)
        purity.append(k / len(names))
        labels.append(f"{k}/{len(names)} {top_sec}" if k >= 4 else "")
    out["sector_label"] = labels
    out["sector_purity"] = purity
    return out
