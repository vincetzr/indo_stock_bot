"""Intraday bars for IDX tickers.

Day trading cannot be evaluated on daily bars. The reason is specific and
decisive: when a trade's target and its stop are both touched during the same
session, a daily bar records only that both happened, not which came first. On
the momentum-burst setups this engine hunts, that ambiguity covers ~13% of
trades and is large enough to flip the strategy's expectancy from -0.56% to
+0.52% per trade. Intraday bars are the only way to settle it.

Availability, measured against Yahoo for ``.JK`` symbols:

    interval   history      notes
    1m         ~7 days      too short for anything but live monitoring
    5m         ~60 days     the workable resolution
    15m        ~60 days
    1h         ~730 days    useful for context, too coarse for entries

Sixty days of 5-minute bars is enough to run live and to sanity-check path
resolution on recent setups. It is *not* enough to backtest a day-trading
strategy properly, and this module does not pretend otherwise.

IDX session (WIB, UTC+7): 09:00-12:00 and 13:30-15:50, Monday to Friday, with a
shorter Friday morning. Timestamps are converted to Asia/Jakarta and tagged with
minutes-since-open so "how far into the session are we" is directly available.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from ..config import Config
from .cache import Cache
from .ohlcv import HEADERS, to_yahoo_symbol

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

INTRADAY_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]

# Longest history Yahoo will serve per interval.
MAX_RANGE = {"1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d", "1h": "730d"}

SESSION_OPEN_MINUTE = 9 * 60          # 09:00 WIB
SESSION_CLOSE_MINUTE = 15 * 60 + 50   # 15:50 WIB
LUNCH_START_MINUTE = 12 * 60
LUNCH_END_MINUTE = 13 * 60 + 30


class YahooIntraday:
    def __init__(self, cfg: Config, cache: Optional[Cache] = None,
                 session: Optional[requests.Session] = None):
        self.cfg = cfg
        self.cache = cache or Cache(cfg.path("data.cache_dir", "data/cache"))
        self.timeout = int(cfg.get("data.request_timeout", 30))
        self.retries = int(cfg.get("data.request_retries", 4))
        self.session = session or requests.Session()
        self.session.headers.update(HEADERS)

    def _fetch_json(self, symbol: str, interval: str, range_: str) -> Optional[dict]:
        params = {"interval": interval, "range": range_, "includePrePost": "false"}
        delay = 2.0
        for attempt in range(self.retries):
            try:
                resp = self.session.get(CHART_URL.format(symbol=symbol),
                                        params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code not in (429, 500, 502, 503, 504):
                    return None
            except (requests.RequestException, ValueError):
                pass
            if attempt < self.retries - 1:
                time.sleep(delay)
                delay *= 2
        return None

    @staticmethod
    def _parse(payload: dict) -> pd.DataFrame:
        chart = (payload or {}).get("chart") or {}
        results = chart.get("result") or []
        if not results:
            return pd.DataFrame(columns=INTRADAY_COLUMNS)
        result = results[0]
        timestamps = result.get("timestamp") or []
        if not timestamps:
            return pd.DataFrame(columns=INTRADAY_COLUMNS)

        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        df = pd.DataFrame({
            "ts": pd.to_datetime(timestamps, unit="s", utc=True)
                    .tz_convert("Asia/Jakarta").tz_localize(None),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        })
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]
        df["volume"] = df["volume"].fillna(0.0)
        return df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    def get(self, ticker: str, interval: str = "5m", range_: Optional[str] = None,
            max_age: float = 300.0, force_refresh: bool = False) -> pd.DataFrame:
        """Intraday bars, cached briefly because they go stale within minutes."""
        symbol = to_yahoo_symbol(ticker)
        range_ = range_ or MAX_RANGE.get(interval, "60d")
        key = f"{symbol}_{interval}_{range_}"

        if not force_refresh:
            cached = self.cache.read("intraday", key, max_age=max_age,
                                     parse_dates=["ts"])
            if cached is not None and not cached.empty:
                return enrich_session(cached)

        payload = self._fetch_json(symbol, interval, range_)
        df = self._parse(payload) if payload else pd.DataFrame(columns=INTRADAY_COLUMNS)
        if df.empty:
            stale = self.cache.read("intraday", key, parse_dates=["ts"])
            return enrich_session(stale) if stale is not None and not stale.empty else df

        self.cache.write("intraday", key, df)
        return enrich_session(df)

    def get_many(self, tickers: List[str], interval: str = "5m",
                 pause: float = 0.3, verbose: bool = False) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for i, ticker in enumerate(tickers, 1):
            df = self.get(ticker, interval=interval)
            if not df.empty:
                out[ticker.upper()] = df
            if verbose:
                print(f"  [{i:>3}/{len(tickers)}] {ticker:<6} {len(df):>5} bars")
            time.sleep(pause)
        return out


def enrich_session(df: pd.DataFrame) -> pd.DataFrame:
    """Attach session date, minutes-since-open, and a session-anchored VWAP.

    VWAP is reset at each session open. An intraday VWAP that carried across
    days would be meaningless as an entry reference.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"])
    out["date"] = out["ts"].dt.normalize()
    minutes = out["ts"].dt.hour * 60 + out["ts"].dt.minute
    out["minute_of_day"] = minutes
    out["minutes_since_open"] = (minutes - SESSION_OPEN_MINUTE).clip(lower=0)
    # Exclude the lunch break so "session elapsed" reflects tradeable time.
    lunch = (minutes > LUNCH_START_MINUTE) & (minutes <= LUNCH_END_MINUTE)
    out["in_lunch"] = lunch
    tradeable = np.where(
        minutes <= LUNCH_START_MINUTE,
        minutes - SESSION_OPEN_MINUTE,
        minutes - SESSION_OPEN_MINUTE - (LUNCH_END_MINUTE - LUNCH_START_MINUTE),
    )
    total = (SESSION_CLOSE_MINUTE - SESSION_OPEN_MINUTE) - (LUNCH_END_MINUTE - LUNCH_START_MINUTE)
    out["session_pct"] = np.clip(tradeable / max(total, 1), 0.0, 1.0)

    typical = (out["high"] + out["low"] + out["close"]) / 3.0
    grouped = out.groupby("date")
    cum_pv = (typical * out["volume"]).groupby(out["date"]).cumsum()
    cum_v = grouped["volume"].cumsum()
    out["vwap"] = np.where(cum_v > 0, cum_pv / cum_v, out["close"])
    out["cum_volume"] = cum_v
    return out


def session_frame(df: pd.DataFrame, date=None) -> pd.DataFrame:
    """Bars for one session (default: the most recent)."""
    if df is None or df.empty:
        return pd.DataFrame()
    if "date" not in df.columns:
        df = enrich_session(df)
    target = pd.Timestamp(date).normalize() if date is not None else df["date"].max()
    return df[df["date"] == target].reset_index(drop=True)


def opening_range(session: pd.DataFrame, minutes: int = 30) -> Dict[str, float]:
    """High/low of the first ``minutes`` of trading.

    The opening range is the reference every breakout entry is measured against;
    a burst that cannot clear its own first 30 minutes is not a burst.
    """
    if session is None or session.empty:
        return {}
    window = session[session["minutes_since_open"] <= minutes]
    if window.empty:
        window = session.head(max(1, minutes // 5))
    return {
        "or_high": float(window["high"].max()),
        "or_low": float(window["low"].min()),
        "or_volume": float(window["volume"].sum()),
        "or_bars": int(len(window)),
        "open": float(session["open"].iloc[0]),
    }


def resolve_path(session: pd.DataFrame, entry: float, target: float,
                 stop: float) -> Dict[str, object]:
    """Walk a session bar by bar to see which level was reached first.

    This is the function that answers what daily bars cannot. Within a single
    5-minute bar that spans both levels the order is still unknowable, so those
    are reported as ``ambiguous`` rather than guessed.
    """
    if session is None or session.empty or entry <= 0:
        return {"outcome": "no_data"}

    for _, bar in session.iterrows():
        hit_target = bar["high"] >= target
        hit_stop = bar["low"] <= stop
        if hit_target and hit_stop:
            return {"outcome": "ambiguous", "ts": bar["ts"],
                    "bar_high": float(bar["high"]), "bar_low": float(bar["low"])}
        if hit_target:
            return {"outcome": "target", "ts": bar["ts"],
                    "return": target / entry - 1.0}
        if hit_stop:
            return {"outcome": "stop", "ts": bar["ts"],
                    "return": stop / entry - 1.0}

    close = float(session["close"].iloc[-1])
    return {"outcome": "close", "ts": session["ts"].iloc[-1],
            "return": close / entry - 1.0}


def volume_pace(session: pd.DataFrame, reference_daily_volume: float) -> float:
    """Today's volume so far, divided by what is normal by this point.

    Raw intraday volume always looks small early in the day. Comparing against
    the same point in an average session is what makes it informative.
    """
    if session is None or session.empty or reference_daily_volume <= 0:
        return float("nan")
    elapsed = float(session["session_pct"].iloc[-1])
    if elapsed <= 0.02:
        return float("nan")
    so_far = float(session["volume"].sum())
    expected = reference_daily_volume * elapsed
    return so_far / expected if expected > 0 else float("nan")
