"""GDELT — the only free route to a BACKTESTABLE narrative series, and it is
quarantined until one specific property is verified.

WHY THIS IS DIFFERENT FROM `news.py`. The RSS layer is live-only: a headline
visible today cannot be reconstructed as it stood on a past bar, so anything
under `spine/` or `features/` importing it would make every downstream backtest
look-ahead by construction, and `tests/test_news.py` fails the build if either
does. GDELT is different in principle — its `timelinevol` mode returns a DAILY
ARTICLE-VOLUME SERIES for an arbitrary query going back to 2015, indexed by the
article's own publication date. That is a historical series, not a live feed.

AND THAT DIFFERENCE IS NOT YET ESTABLISHED, SO THE QUARANTINE STAYS.
`timelinevol` is a query over GDELT's CURRENT index, and GDELT has grown its
source list and reprocessed its corpus repeatedly. So the volume it reports for
2020-03-15 today is not necessarily the volume it would have reported on
2020-03-16. If it is not, a "point-in-time" narrative feature is look-ahead
with no visible symptom — the series looks perfectly historical.

THAT IS TESTABLE, AND THE ONLY THING THAT MAKES IT TESTABLE IS RECORDING WHEN
EACH OBSERVATION WAS FETCHED. Every cached row here carries `fetched_at`, and
`stability()` compares two fetches of the same window. Nothing can lift the
quarantine except that comparison coming back clean, and the comparison needs
two fetches months apart, so it cannot be run today. Writing the timestamp down
now is what makes it runnable later; not writing it down is what would make the
question permanently unanswerable.

The immutable alternative is the raw 15-minute file archive at
`data.gdeltproject.org/gdeltv2/` — those files are published once and never
rewritten, so they ARE point-in-time by construction. `masterfilelist.txt`
answers 200 and is 127 MB listing ~150,000 files; the corpus behind it is
measured in terabytes. Recorded as available and not taken.

------------------------------------------------------------------ MEASURED RATE
The operator's own 429 body says "Please limit requests to one every 5
seconds". Measured on 2026-09-07 from this host, that is not sufficient:

    6s spacing    2 of 2, then 429, then a connection reset
    15s spacing   2 of 4 succeeded (ADRO and PANI gave up after 3 tries each)
    20s spacing   ADRO still gave up

**SO THE SUSTAINABLE RATE IS NOT ESTABLISHED, and that is the honest state.**
It is slower than 20s from this host, and probing further to pin it down means
hammering a rate-limited public API to find out how hard you can hammer it,
which is not a thing to do. `MIN_GAP` is 20s with exponential backoff on top,
and a full-universe collection is NOT schedulable from this environment with
any confidence. `docs/PLATFORM_ASSESSMENT.md` stage 7's "sustainable rate not
established" stands, now with numbers behind it instead of a single 429.

What IS established is the unit price, and it is far better than the assessment
assumed: `startdatetime`/`enddatetime` span arbitrary ranges, so a full history
for one name is ONE request, not one per name-day. About 840 requests for the
whole universe. A1's lesson holds — check the unit price before writing down
the arithmetic — the blocker here is the rate, not the count.

------------------------------------------------------- THE RELEVANCE PROBLEM
A12 recorded that many IDX tickers are ordinary Indonesian words — GULA
(sugar), KOTA (city), RAJA (king), CASH, BABY, COAL — so an unfiltered query
returns the commodity and the municipality. GDELT is worse than RSS here, not
better: it indexes global news in every language, so a bare four-letter token
matches far more. Measured over 2023: `GULA` returns **238 of 364 days
non-zero**, which is a series about sugar. `BBCA` returns 354 of 364, which may
be Bank Central Asia or may be an acronym collision — a volume series cannot
tell you which.

So the ticker is NOT the query. `query_for()` builds a quoted company-name
query and requires the name, and the ticker alone is available only through an
explicit `raw=True` that the docstring warns about.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import pandas as pd
import requests

from .cache import Cache

DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
MASTER_LIST = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"

#: Measured, not quoted from the operator. See the module docstring.
MIN_GAP = 20.0
RETRIES = 4

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}

COLUMNS = ["date", "value", "query", "fetched_at"]


def query_for(company: str, ticker: Optional[str] = None,
              raw: bool = False) -> str:
    """Build a GDELT query for one IDX name.

    THE TICKER IS NOT THE QUERY AND `raw=True` IS NOT A CONVENIENCE.
    Measured over 2023, the bare token `GULA` returns 238 non-zero days — a
    series about sugar, since gula is the Indonesian word for it. A ticker
    query produces a plausible, dense, entirely wrong series, and no downstream
    statistic can detect that. Pass the company name.
    """
    company = (company or "").strip()
    if raw:
        if not ticker:
            raise ValueError("raw=True needs a ticker")
        return ticker.strip().upper()
    if not company:
        raise ValueError(
            "GDELT needs a company NAME — a bare ticker matches ordinary "
            "words (GULA=sugar, KOTA=city, RAJA=king) and returns a dense, "
            "plausible series about the wrong subject")
    return f'"{company}"'


class GDELT:
    """Polite client for the DOC 2.0 API. Read-only, permanently cached."""

    def __init__(self, cache: Optional[Cache] = None,
                 session: Optional[requests.Session] = None,
                 min_gap: float = MIN_GAP):
        self.cache = cache or Cache("data/cache")
        self.session = session or requests.Session()
        self.session.headers.update(HEADERS)
        self.min_gap = float(min_gap)
        self._last = 0.0

    def _wait(self) -> None:
        gap = time.time() - self._last
        if gap < self.min_gap:
            time.sleep(self.min_gap - gap)
        self._last = time.time()

    def _get(self, params: Dict[str, str]) -> Optional[dict]:
        delay = self.min_gap
        for attempt in range(RETRIES):
            self._wait()
            try:
                r = self.session.get(DOC_API, params=params, timeout=90)
            except requests.RequestException:
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    return None
            #  429 is the operator asking us to slow down. Backing off is the
            #  whole of the correct response; there is no other route and
            #  there must not be one.
            if attempt < RETRIES - 1:
                time.sleep(delay)
                delay *= 2
        return None

    def timeline(self, query: str, start: str, end: str,
                 max_age: float = 30 * 86400.0) -> pd.DataFrame:
        """Daily article-volume series for `query` between two YYYYMMDD dates.

        Cached permanently under a key derived from the query and window, with
        `fetched_at` stamped on every row — see the module docstring for why
        that column is the whole point.
        """
        key = f"{query}_{start}_{end}".replace(" ", "_").replace('"', "")
        cached = self.cache.read("gdelt", key, max_age=max_age)
        if cached is not None and not cached.empty:
            return cached
        payload = self._get({
            "query": query, "mode": "timelinevol", "format": "json",
            "startdatetime": f"{start}000000", "enddatetime": f"{end}000000",
        })
        df = parse_timeline(payload, query)
        if not df.empty:
            self.cache.write("gdelt", key, df, merge_on="date")
        return df


def parse_timeline(payload: Optional[dict], query: str) -> pd.DataFrame:
    """DOC 2.0 `timelinevol` JSON -> (date, value, query, fetched_at)."""
    series = ((payload or {}).get("timeline") or [{}])
    data = series[0].get("data") if series else None
    if not data:
        return pd.DataFrame(columns=COLUMNS)
    rows = []
    now = pd.Timestamp.now("UTC").tz_localize(None)
    for d in data:
        ts = d.get("date")
        if ts is None:
            continue
        rows.append({"date": pd.to_datetime(str(ts)[:8], format="%Y%m%d",
                                            errors="coerce"),
                     "value": float(d.get("value") or 0.0),
                     "query": query, "fetched_at": now})
    out = pd.DataFrame(rows, columns=COLUMNS).dropna(subset=["date"])
    return out.sort_values("date").reset_index(drop=True)


def stability(a: pd.DataFrame, b: pd.DataFrame) -> Dict[str, float]:
    """Compare two fetches of the SAME window. The quarantine turns on this.

    If GDELT's index is append-only for past dates, `a` and `b` agree and the
    series is genuinely point-in-time. If it retro-indexes, they do not, and a
    narrative feature built on it is look-ahead with no visible symptom.

    Returns the share of dates that disagree and the mean signed change, so a
    systematic upward revision (more articles found later) is distinguishable
    from noise. It CANNOT be run on one fetch, and that is the honest state
    today: the two fetches have to be months apart.
    """
    if a is None or b is None or a.empty or b.empty:
        return {}
    m = a.merge(b, on="date", suffixes=("_a", "_b"))
    if m.empty:
        return {}
    d = m["value_b"] - m["value_a"]
    return {
        "dates": float(len(m)),
        "disagree": float((d.abs() > 1e-9).mean()),
        "mean_change": float(d.mean()),
        "max_abs_change": float(d.abs().max()),
        "gap_days": float((pd.to_datetime(m["fetched_at_b"]).max()
                           - pd.to_datetime(m["fetched_at_a"]).max()).days),
    }


def quarantine_note() -> str:
    """The sentence that must accompany any GDELT number, anywhere."""
    return (
        "GDELT MAY NOT ENTER ANY STATISTIC. Its timeline is a query over the "
        "CURRENT index, and whether that index is stable for past dates is "
        "UNVERIFIED — it needs two fetches months apart, compared with "
        "gdelt.stability(). Until that comes back clean a narrative feature "
        "built on it is look-ahead with no visible symptom. Every cached row "
        "carries fetched_at so the comparison stays possible."
    )
