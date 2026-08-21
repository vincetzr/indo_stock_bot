"""The full broker summary, from a licensed redistributor, with a key.

WHAT THIS SOLVES
----------------
Every other route in this package sees a **top ten**. IndoPremier's public
module publishes the ten largest buyers and the ten largest sellers and nothing
else, which leaves 10-15% of a session's volume unattributed and forces
:mod:`idxbot.broker_bounds` to report positions as intervals rather than
numbers. That censoring is the single largest source of imprecision in the
whole project.

Sectors (``sectors.app``, operated by Supertype) redistributes IDX market data
under licence and exposes it as a REST API. Its broker endpoint returns, in the
vendor's own words, *"every broker active on that day"* - not a top ten. The
worked example in their specification is a broker that traded **55 lots**, which
no top-ten table on earth would ever show. That is the full rekap.

WHY THIS ROUTE AND NOT THE OTHER ONE
------------------------------------
IDX publishes the same table free at
``idx.co.id/en/market-data/trading-summary/broker-summary``, and it is
unreachable from a script because Cloudflare blocks non-browser TLS
fingerprints. Defeating that check is the thing IDX's terms forbid, so this
package does not do it (see ``docs/FULL_REKAP.md``). Paying a licensed
redistributor for the same bytes is the same data obtained the way it is meant
to be obtained. The difference is not technical, it is whether anyone has the
right to hand it over.

THE CREDIT ECONOMY, WHICH DICTATES THE DESIGN
---------------------------------------------
One call costs **one credit** and returns **up to fourteen days**. A one-day
call costs exactly the same as a fourteen-day call.

So a per-day fetcher would burn fourteen times the money for the same data, and
this module never does one. :meth:`SectorsBrokerSummary.fetch_day` resolves the
day to the fixed fortnight that contains it, fetches and caches that whole
fortnight, and returns the requested slice. Ask for 400 sessions of one ticker
and it costs about 29 credits, not 400. The fortnights are aligned to a fixed
epoch rather than to the requested date, so two callers asking for two
different days in the same fortnight hit one cache entry instead of paying
twice for overlapping windows.

The cache is permanent. A settled broker summary never changes, so a day that
has been paid for is never paid for again.

WHAT YOU GET THAT THE FREE ROUTE CANNOT GIVE
--------------------------------------------
    every broker      no censoring, so positions are exact rather than bracketed
    exact integers    ``bval`` is rupiah to the rupiah. The free route prints
                      four significant figures in billions and two in
                      trillions, which is a 4.55% uncertainty on a busy day
    trade frequency   ``bfreq``/``sfreq``, the number of trades behind the
                      volume. Not available from the free route at all, and it
                      separates one institutional block from a thousand retail
                      tickets - which is the distinction most of the folklore
                      in this market is really reaching for

Credentials
-----------
The key is read from the ``SECTORS_API_KEY`` environment variable, or from
``data.sectors_api_key`` in the config. It is never written to the cache, never
logged, and never included in an error message: a failure prints the status
code and the endpoint, not the header.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np
import pandas as pd

from .broker_summary import LOT_SIZE, BrokerSummaryProvider, empty_frame

BASE_URL = "https://api.sectors.app/v2"

#: The vendor's hard cap on a single request. Not a tuning knob - asking for 15
#: is a 400, and asking for 1 costs the same credit as asking for 14.
MAX_WINDOW_DAYS = 14

#: Fortnights are counted from this Monday rather than from whatever day the
#: caller happened to ask for. Two callers wanting two days in the same
#: fortnight then compute the same window and share one cache entry; without a
#: fixed epoch they would compute two overlapping windows and pay twice.
EPOCH = pd.Timestamp("2000-01-03")

#: Columns this route adds beyond the canonical schema. Trade counts are not
#: available from any free route, so nothing downstream may require them.
EXTRA_COLUMNS = ("buy_freq", "sell_freq")

#: Largest tolerated ``|value - lots x 100 x average| / value``. The vendor
#: publishes exact integers, so unlike the abbreviated free route this should
#: reconcile to floating-point noise. Anything above this is a field-mapping
#: fault - reading ``navg`` where ``bavg`` was meant, say - not rounding.
CONSISTENCY_BOUND = 1e-6


def window_for(day: pd.Timestamp) -> tuple:
    """The fixed fortnight containing ``day``, as ``(start, end)`` inclusive."""
    day = pd.Timestamp(day).normalize()
    n = (day - EPOCH).days // MAX_WINDOW_DAYS
    start = EPOCH + pd.Timedelta(days=int(n) * MAX_WINDOW_DAYS)
    return start, start + pd.Timedelta(days=MAX_WINDOW_DAYS - 1)


def windows_covering(start: pd.Timestamp, end: pd.Timestamp) -> List[tuple]:
    """Every fortnight needed to cover ``[start, end]``, in order, no overlap."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if end < start:
        return []
    out, cur = [], window_for(start)[0]
    while cur <= end:
        out.append((cur, cur + pd.Timedelta(days=MAX_WINDOW_DAYS - 1)))
        cur += pd.Timedelta(days=MAX_WINDOW_DAYS)
    return out


def api_key(cfg=None) -> Optional[str]:
    """Environment first, config second. Absent is a normal state, not an error."""
    key = os.environ.get("SECTORS_API_KEY", "").strip()
    if key:
        return key
    if cfg is not None:
        val = cfg.get("data.sectors_api_key", "") or ""
        val = str(val).strip()
        # A config shipped with a placeholder must not be sent as a credential.
        if val and not val.lower().startswith(("your", "<", "changeme", "todo")):
            return val
    return None


def parse_payload(payload: dict, ticker: str) -> pd.DataFrame:
    """Vendor JSON -> the canonical schema, plus trade counts.

    ``bavg_per_share`` is per SHARE while ``blot`` is in lots, and the identity
    that ties them together is ``value = lots x 100 x average``. Their own
    example satisfies it exactly (55 x 100 x 8,900 = 48,950,000), which is what
    :func:`consistency` re-checks on live data - a parser that read the wrong
    average column would fail it by orders of magnitude.
    """
    rows: List[dict] = []
    for block in (payload or {}).get("data", []) or []:
        try:
            day = pd.Timestamp(block.get("date")).normalize()
        except Exception:                                   # noqa: BLE001
            continue
        for r in block.get("summary", []) or []:
            code = str(r.get("broker_code", "")).strip().upper()
            if not code:
                continue
            rows.append({
                "date": day, "ticker": str(ticker).upper(), "broker": code,
                "buy_lot": float(r.get("blot") or 0.0),
                "buy_val": float(r.get("bval") or 0.0),
                "buy_avg": float(r.get("bavg_per_share") or 0.0),
                "sell_lot": float(r.get("slot") or 0.0),
                "sell_val": float(r.get("sval") or 0.0),
                "sell_avg": float(r.get("savg_per_share") or 0.0),
                "buy_freq": float(r.get("bfreq") or 0.0),
                "sell_freq": float(r.get("sfreq") or 0.0),
                "source": "sectors",
            })
    if not rows:
        return empty_frame()
    return pd.DataFrame(rows).sort_values(["date", "broker"]).reset_index(drop=True)


def consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Re-derive value from lots x 100 x average and report the disagreement.

    The redundancy is the acceptance test. It costs nothing to run and it is
    the only thing standing between a silently mis-mapped column and a year of
    conclusions drawn from it.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["side", "rows", "worst", "median"])
    out = []
    for side in ("buy", "sell"):
        lot = pd.to_numeric(df.get(f"{side}_lot"), errors="coerce")
        val = pd.to_numeric(df.get(f"{side}_val"), errors="coerce")
        avg = pd.to_numeric(df.get(f"{side}_avg"), errors="coerce")
        implied = lot * LOT_SIZE * avg
        ok = np.isfinite(implied) & np.isfinite(val) & (val > 0)
        if not ok.any():
            out.append({"side": side, "rows": 0, "worst": np.nan,
                        "median": np.nan})
            continue
        err = (implied[ok] - val[ok]).abs() / val[ok]
        out.append({"side": side, "rows": int(ok.sum()),
                    "worst": float(err.max()), "median": float(err.median())})
    return pd.DataFrame(out)


class SectorsBrokerSummary(BrokerSummaryProvider):
    """Full-depth rekap from a licensed redistributor.

    ``delay`` is a courtesy floor between calls, not a rate limit to tune down.
    The vendor's own limiter returns 429 and does not bill for it, but a client
    that discovers a rate limit by hitting it is a client that will one day hit
    something worse.
    """

    name = "sectors"
    is_real = True
    #: This route is NOT censored, so a day from it is a complete rekap and
    #: downstream bounds collapse from intervals to points. Nothing else in the
    #: package may claim this.
    complete = True

    def __init__(self, cache=None, key: Optional[str] = None, cfg=None,
                 delay: float = 0.4, timeout: int = 30, verbose: bool = False,
                 retries: int = 3):
        self.cache = cache
        self._key = key or api_key(cfg)
        self.delay = float(delay)
        self.timeout = int(timeout)
        self.verbose = bool(verbose)
        self.retries = max(1, int(retries))
        self._last_call = 0.0
        #: Credits actually spent this session, so a backfill can be costed
        #: before it is run rather than discovered on an invoice.
        self.credits_spent = 0

    def available(self) -> bool:
        try:
            import requests                                 # noqa: F401
        except ImportError:
            return False
        return bool(self._key)

    # -- network ----------------------------------------------------------
    def _get_window(self, ticker: str, start: pd.Timestamp,
                    end: pd.Timestamp) -> Optional[dict]:
        import requests

        gap = self.delay - (time.time() - self._last_call)
        if gap > 0:
            time.sleep(gap)
        url = f"{BASE_URL}/broker-summary/{str(ticker).upper()}/"
        params = {"start": f"{start:%Y-%m-%d}", "end": f"{end:%Y-%m-%d}"}
        last: Optional[Exception] = None
        for attempt in range(self.retries):
            if attempt:
                time.sleep(self.delay * (2 ** attempt))
            try:
                resp = requests.get(url, params=params, timeout=self.timeout,
                                    headers={"Authorization": self._key,
                                             "Accept": "application/json"})
                self._last_call = time.time()
                if resp.status_code == 429:
                    # Not billed, and the fix is to wait rather than to retry
                    # harder. Fall through to the backoff.
                    last = RuntimeError("429 rate limited")
                    continue
                if resp.status_code in (401, 403):
                    # Never echo the header. A bad key is a configuration
                    # problem and printing the credential does not help fix it.
                    raise RuntimeError(f"{resp.status_code} - the API key was "
                                       f"rejected by {url}")
                if resp.status_code == 404:
                    self.credits_spent += 1      # billed even though empty
                    return None
                resp.raise_for_status()
                self.credits_spent += 1
                return resp.json()
            except Exception as exc:                        # noqa: BLE001
                last = exc
                self._last_call = time.time()
        if self.verbose and last is not None:
            print(f"  ! sectors {ticker} {start:%Y-%m-%d}..{end:%Y-%m-%d}: {last}")
        return None

    def fetch_window(self, ticker: str, day: pd.Timestamp) -> pd.DataFrame:
        """The whole fortnight containing ``day``. One credit, cached forever."""
        start, end = window_for(day)
        key = f"{str(ticker).upper()}_{start:%Y%m%d}"
        if self.cache is not None:
            hit = self.cache.read("sectors_broker", key)
            if hit is not None:
                return hit
        if not self.available():
            return empty_frame()
        payload = self._get_window(ticker, start, end)
        df = parse_payload(payload or {}, ticker)
        # A fortnight of holidays and a fortnight the key could not reach look
        # identical from here, and caching a blank would make the second one
        # permanent. Only real rows are written.
        if not df.empty and self.cache is not None:
            self.cache.write("sectors_broker", key, df)
        return df

    def fetch_day(self, ticker: str, day: pd.Timestamp) -> pd.DataFrame:
        """One session - but it pays for, and keeps, the fortnight around it."""
        day = pd.Timestamp(day).normalize()
        df = self.fetch_window(ticker, day)
        if df.empty:
            return df
        d = df[pd.to_datetime(df["date"]).dt.normalize() == day]
        return d.reset_index(drop=True) if len(d) else empty_frame()

    def fetch_range(self, ticker: str, start, end) -> pd.DataFrame:
        """Every session in ``[start, end]``, at the fewest credits possible."""
        frames = [self.fetch_window(ticker, s)
                  for s, _ in windows_covering(start, end)]
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            return empty_frame()
        out = pd.concat(frames, ignore_index=True)
        d = pd.to_datetime(out["date"]).dt.normalize()
        out = out[(d >= pd.Timestamp(start).normalize())
                  & (d <= pd.Timestamp(end).normalize())]
        return out.sort_values(["date", "broker"]).reset_index(drop=True)

    def fetch(self, ticker: str, start=None, end=None) -> pd.DataFrame:
        """The provider interface. Defaults match the free route's 90 days."""
        if end is None:
            end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
        if start is None:
            start = pd.Timestamp(end) - pd.Timedelta(days=90)
        return self.fetch_range(ticker, start, end)

    @staticmethod
    def credits_for(tickers: int, sessions: int) -> int:
        """What a backfill will cost, before it is run.

        Calendar days, not trading days: a fortnight window spans 14 calendar
        days and holds about 10 sessions, so quoting it in sessions would
        understate the bill by roughly 40%.
        """
        cal = int(np.ceil(max(0, sessions) * 7.0 / 5.0))
        return int(max(0, tickers) * int(np.ceil(cal / MAX_WINDOW_DAYS)))
