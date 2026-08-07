"""Full-history OHLCV for IDX tickers.

Uses Yahoo Finance's chart endpoint directly with ``period1=0`` so every request
returns the instrument's complete daily history from its first trade date.

A caveat worth knowing: Yahoo silently downgrades ``range=max`` to *monthly*
bars. Passing an explicit ``period1``/``period2`` pair is what actually returns
full-depth daily data, which is why this module never uses ``range``.

Verified depths (Aug 2026): ^JKSE from 1990-04-06 (9,166 bars), ASII from
2000-10-17, BBRI from 2003-11-10, BBCA from 2004-06-08. Yahoo does not carry
IDX data before 1990, so "since the start of the exchange" in practice means
"since 1990 for the index, and since each stock's listing or 2000, whichever is
later".
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import pandas as pd
import requests

from ..config import Config
from .cache import Cache

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# Yahoo rejects requests without a browser-like User-Agent.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}

OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


def to_yahoo_symbol(ticker: str) -> str:
    """``BBCA`` -> ``BBCA.JK``; index symbols such as ``^JKSE`` pass through."""
    ticker = ticker.strip().upper()
    if ticker.startswith("^") or "." in ticker:
        return ticker
    return f"{ticker}.JK"


class YahooOHLCV:
    def __init__(self, cfg: Config, cache: Optional[Cache] = None,
                 session: Optional[requests.Session] = None):
        self.cfg = cfg
        self.cache = cache or Cache(cfg.path("data.cache_dir", "data/cache"))
        self.timeout = int(cfg.get("data.request_timeout", 30))
        self.retries = int(cfg.get("data.request_retries", 4))
        self.session = session or requests.Session()
        self.session.headers.update(HEADERS)

    # -- network ------------------------------------------------------------
    def _fetch_json(self, symbol: str) -> Optional[dict]:
        params = {
            "period1": 0,
            "period2": int(time.time()) + 86400,
            "interval": "1d",
            "events": "div,split",
            "includeAdjustedClose": "true",
        }
        delay = 2.0
        last_error: Optional[str] = None
        for attempt in range(self.retries):
            try:
                resp = self.session.get(
                    CHART_URL.format(symbol=symbol), params=params, timeout=self.timeout
                )
                if resp.status_code == 200:
                    return resp.json()
                # 429 is Yahoo rate limiting; backing off usually clears it.
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {resp.status_code}"
                else:
                    return None
            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
            if attempt < self.retries - 1:
                time.sleep(delay)
                delay *= 2
        if last_error:
            print(f"  ! {symbol}: giving up after {self.retries} attempts ({last_error})")
        return None

    @staticmethod
    def _parse(payload: dict) -> pd.DataFrame:
        chart = (payload or {}).get("chart") or {}
        results = chart.get("result") or []
        if not results:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        result = results[0]
        timestamps = result.get("timestamp") or []
        if not timestamps:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        adj = (result.get("indicators", {}).get("adjclose") or [{}])
        adj_close = adj[0].get("adjclose") if adj else None

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(
                    "Asia/Jakarta"
                ).tz_localize(None).normalize(),
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "close": quote.get("close"),
                "volume": quote.get("volume"),
            }
        )
        df["adj_close"] = adj_close if adj_close is not None else df["close"]

        df = df.dropna(subset=["close"])
        # Suspended sessions come back as zero-volume flat bars; they distort
        # volume statistics and Wyckoff volume tests, so drop them.
        df = df[df["close"] > 0]
        df = df.drop_duplicates(subset=["date"], keep="last")
        df = df.sort_values("date").reset_index(drop=True)
        return df[OHLCV_COLUMNS]

    # -- public API ---------------------------------------------------------
    def get(self, ticker: str, max_age: float = 3600.0,
            force_refresh: bool = False) -> pd.DataFrame:
        """Return the full daily history for one ticker."""
        symbol = to_yahoo_symbol(ticker)
        if not force_refresh:
            cached = self.cache.read("ohlcv", symbol, max_age=max_age)
            if cached is not None and not cached.empty:
                return cached

        payload = self._fetch_json(symbol)
        df = self._parse(payload) if payload else pd.DataFrame(columns=OHLCV_COLUMNS)

        if df.empty:
            # Fall back to a stale cache entry rather than returning nothing:
            # week-old prices beat no prices when the network is flaky.
            stale = self.cache.read("ohlcv", symbol)
            if stale is not None and not stale.empty:
                print(f"  ! {symbol}: fetch failed, using cached data")
                return stale
            return df

        self.cache.write("ohlcv", symbol, df)
        return df

    def get_many(self, tickers: List[str], max_age: float = 3600.0,
                 force_refresh: bool = False, pause: float = 0.35,
                 verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """Fetch several tickers, pausing between network calls to avoid 429s."""
        out: Dict[str, pd.DataFrame] = {}
        for i, ticker in enumerate(tickers, 1):
            symbol = to_yahoo_symbol(ticker)
            had_cache = self.cache.read("ohlcv", symbol, max_age=max_age) is not None
            df = self.get(ticker, max_age=max_age, force_refresh=force_refresh)
            if not df.empty:
                out[ticker.upper()] = df
            if verbose:
                status = f"{len(df):>5} bars" if not df.empty else "   no data"
                span = ""
                if not df.empty:
                    span = f"  {df['date'].iloc[0]:%Y-%m-%d} -> {df['date'].iloc[-1]:%Y-%m-%d}"
                print(f"  [{i:>3}/{len(tickers)}] {ticker:<6} {status}{span}")
            if not had_cache and not force_refresh:
                time.sleep(pause)
            elif force_refresh:
                time.sleep(pause)
        return out
