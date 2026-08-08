"""Broker summary ("rekap broker") ingestion.

Broker summary is the per-day, per-stock breakdown of how much each exchange
member bought and sold, and at what average price. It is the raw material for
every "bandarmology" inference in this package: without it you can only guess at
institutional intent from price and volume.

There is no free public API for it. IDX's own endpoint sits behind a WAF, and
the retail platforms that display it require an authenticated session. The
options that actually work are documented in ``docs/LIVE_DATA.md``; this module
provides the plumbing for all of them behind one interface.

Canonical schema produced by every provider
-------------------------------------------
    date      datetime64[ns]  trading day (Asia/Jakarta, normalised to midnight)
    ticker    str             bare IDX code, e.g. "BBCA"
    broker    str             exchange member code, e.g. "BK"
    buy_lot   float           lots bought  (1 lot = 100 shares)
    buy_val   float           rupiah bought
    buy_avg   float           volume-weighted average buy price
    sell_lot  float           lots sold
    sell_val  float           rupiah sold
    sell_avg  float           volume-weighted average sell price
    source    str             provenance tag, carried into every downstream output
"""

from __future__ import annotations

import glob
import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import Config

SCHEMA = [
    "date", "ticker", "broker",
    "buy_lot", "buy_val", "buy_avg",
    "sell_lot", "sell_val", "sell_avg",
    "source",
]

# Header aliases seen across Stockbit, RTI, IPOT, Mirae HOTS and IDX exports,
# in both English and Indonesian. Matching is done on a normalised key
# (lowercased, non-alphanumerics stripped).
COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "date": ("date", "tanggal", "tgl", "tradedate", "trade_date", "waktu"),
    "ticker": ("ticker", "symbol", "stock", "kode", "kodesaham", "emiten", "code"),
    "broker": ("broker", "brokercode", "kodebroker", "brokerid", "ab", "anggotabursa",
               "member", "brk"),
    "buy_lot": ("blot", "buylot", "buyvolume", "buyvol", "volumebeli", "volbeli",
                "beli", "bvolume", "buyqty", "lotbeli"),
    "buy_val": ("bval", "buyvalue", "nilaibeli", "buyval", "valuebeli", "bvalue"),
    "buy_avg": ("bavg", "buyaverage", "buyavg", "rataratabeli", "avgbeli",
                "averagebuy", "bavgprice", "hargabeli"),
    "sell_lot": ("slot", "selllot", "sellvolume", "sellvol", "volumejual", "voljual",
                 "jual", "svolume", "sellqty", "lotjual"),
    "sell_val": ("sval", "sellvalue", "nilaijual", "sellval", "valuejual", "svalue"),
    "sell_avg": ("savg", "sellaverage", "sellavg", "rataratajual", "avgjual",
                 "averagesell", "savgprice", "hargajual"),
}

LOT_SIZE = 100


def _norm_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _resolve_columns(columns: Sequence[str]) -> Dict[str, str]:
    """Map canonical field -> actual column name in a source file."""
    normalised = {_norm_key(c): c for c in columns}
    resolved: Dict[str, str] = {}
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalised:
                resolved[field] = normalised[alias]
                break
    return resolved


def _to_number(series: pd.Series) -> pd.Series:
    """Parse numbers that may carry thousand separators or Indonesian decimals.

    Indonesian exports commonly use ``1.234.567,89``. Detect that form by
    checking whether a comma appears after the last dot.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    s = series.astype(str).str.strip()
    s = s.str.replace(r"[^\d,.\-]", "", regex=True)

    def _one(value: str) -> float:
        if not value or value in {"-", "."}:
            return 0.0
        last_dot, last_comma = value.rfind("."), value.rfind(",")

        if last_dot >= 0 and last_comma >= 0:
            # Both present: whichever comes last is the decimal separator.
            if last_comma > last_dot:      # 1.234.567,89 -> Indonesian
                value = value.replace(".", "").replace(",", ".")
            else:                           # 1,234,567.89 -> English
                value = value.replace(",", "")
        else:
            # Only one separator kind. "1.234" is genuinely ambiguous: 1234 with
            # an Indonesian thousands dot, or 1.234 as an English decimal. IDX
            # quotes whole rupiah and reports volume in whole lots, so a
            # separator followed by exactly three digits is a thousands
            # separator. Anything else is a decimal point.
            separator = "." if last_dot >= 0 else ("," if last_comma >= 0 else "")
            if separator:
                head, _, tail = value.rpartition(separator)
                grouped = len(tail) == 3 and tail.isdigit() and head not in ("", "-")
                if grouped or value.count(separator) > 1:
                    value = value.replace(separator, "")
                elif separator == ",":
                    value = value.replace(",", ".")
        try:
            return float(value)
        except ValueError:
            return 0.0

    return s.map(_one).astype(float)


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in SCHEMA}).astype(
        {"date": "datetime64[ns]", "ticker": "object", "broker": "object", "source": "object"}
    )


def normalise(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    source: str = "unknown",
    volume_unit: str = "auto",
    reference_prices: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Coerce an arbitrary broker-summary export into the canonical schema.

    ``volume_unit`` handles the single most common ingestion bug: some platforms
    report volume in lots and others in shares. With ``auto``, the implied
    average price (value / volume) is compared against the day's traded range;
    whichever unit lands inside the range wins.
    """
    if df is None or df.empty:
        return empty_frame()

    cols = _resolve_columns(df.columns)
    if "broker" not in cols:
        raise ValueError(
            f"[{source}] no broker-code column found. Columns seen: {list(df.columns)}. "
            f"Add the header to COLUMN_ALIASES['broker'] if your export uses a new name."
        )

    out = pd.DataFrame()
    out["broker"] = df[cols["broker"]].astype(str).str.strip().str.upper()

    if "ticker" in cols:
        out["ticker"] = df[cols["ticker"]].astype(str).str.strip().str.upper()
        out["ticker"] = out["ticker"].str.replace(r"\.JK$", "", regex=True)
    elif ticker:
        out["ticker"] = str(ticker).upper()
    else:
        raise ValueError(f"[{source}] no ticker column and no ticker argument given")

    if "date" in cols:
        out["date"] = pd.to_datetime(df[cols["date"]], errors="coerce", dayfirst=True)
    else:
        raise ValueError(f"[{source}] no date column found")

    for field in ("buy_lot", "buy_val", "sell_lot", "sell_val"):
        out[field] = _to_number(df[cols[field]]) if field in cols else 0.0
    for field in ("buy_avg", "sell_avg"):
        out[field] = _to_number(df[cols[field]]) if field in cols else np.nan

    out = out.dropna(subset=["date"])
    out = out[out["broker"].str.len().between(1, 4)]
    if out.empty:
        return empty_frame()
    out["date"] = out["date"].dt.normalize()

    # -- unit reconciliation ------------------------------------------------
    divisor = _resolve_volume_divisor(out, volume_unit, reference_prices)
    if divisor != 1.0:
        out["buy_lot"] = out["buy_lot"] / divisor
        out["sell_lot"] = out["sell_lot"] / divisor

    # Fill missing averages from value/volume, and missing values from avg*volume,
    # so downstream code can rely on all six fields being populated.
    buy_shares = out["buy_lot"] * LOT_SIZE
    sell_shares = out["sell_lot"] * LOT_SIZE
    out["buy_avg"] = out["buy_avg"].where(
        out["buy_avg"].notna() & (out["buy_avg"] > 0),
        np.divide(out["buy_val"], buy_shares, out=np.zeros(len(out)), where=buy_shares > 0),
    )
    out["sell_avg"] = out["sell_avg"].where(
        out["sell_avg"].notna() & (out["sell_avg"] > 0),
        np.divide(out["sell_val"], sell_shares, out=np.zeros(len(out)), where=sell_shares > 0),
    )
    no_buy_val = out["buy_val"] <= 0
    out.loc[no_buy_val, "buy_val"] = (buy_shares * out["buy_avg"].fillna(0.0))[no_buy_val]
    no_sell_val = out["sell_val"] <= 0
    out.loc[no_sell_val, "sell_val"] = (sell_shares * out["sell_avg"].fillna(0.0))[no_sell_val]

    out[["buy_avg", "sell_avg"]] = out[["buy_avg", "sell_avg"]].fillna(0.0)
    out["source"] = source

    # One row per (date, ticker, broker): some exports split by board.
    out = (
        out.groupby(["date", "ticker", "broker", "source"], as_index=False)
        .agg({"buy_lot": "sum", "buy_val": "sum", "sell_lot": "sum", "sell_val": "sum"})
    )
    bs = out["buy_lot"] * LOT_SIZE
    ss = out["sell_lot"] * LOT_SIZE
    out["buy_avg"] = np.divide(out["buy_val"], bs, out=np.zeros(len(out)), where=bs > 0)
    out["sell_avg"] = np.divide(out["sell_val"], ss, out=np.zeros(len(out)), where=ss > 0)

    return out[SCHEMA].sort_values(["date", "ticker", "broker"]).reset_index(drop=True)


def _resolve_volume_divisor(
    df: pd.DataFrame, volume_unit: str, reference_prices: Optional[pd.DataFrame]
) -> float:
    """Return the factor converting the file's volume unit into lots."""
    if volume_unit == "lot":
        return 1.0
    if volume_unit == "share":
        return float(LOT_SIZE)
    if volume_unit != "auto":
        raise ValueError(f"volume_unit must be auto/lot/share, got {volume_unit!r}")

    mask = (df["buy_lot"] > 0) & (df["buy_val"] > 0)
    if not mask.any():
        return 1.0

    implied_if_lots = (df.loc[mask, "buy_val"] / (df.loc[mask, "buy_lot"] * LOT_SIZE)).median()
    implied_if_shares = (df.loc[mask, "buy_val"] / df.loc[mask, "buy_lot"]).median()

    # Prefer a real price reference when we have one.
    if reference_prices is not None and not reference_prices.empty:
        ref = reference_prices.copy()
        ref["date"] = pd.to_datetime(ref["date"]).dt.normalize()
        days = df.loc[mask, "date"].unique()
        window = ref[ref["date"].isin(days)]
        if not window.empty:
            low, high = window["low"].min(), window["high"].max()
            if low > 0:
                lots_ok = low * 0.5 <= implied_if_lots <= high * 2.0
                shares_ok = low * 0.5 <= implied_if_shares <= high * 2.0
                if lots_ok and not shares_ok:
                    return 1.0
                if shares_ok and not lots_ok:
                    return float(LOT_SIZE)

    # No reference: fall back on the fact that IDX quotes whole rupiah and the
    # cheapest tradeable price is Rp50. An implied price below that means the
    # volume column is already in shares.
    if implied_if_lots < 50 <= implied_if_shares:
        return float(LOT_SIZE)
    return 1.0


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class BrokerSummaryProvider(ABC):
    """Interface every broker-summary source implements."""

    name: str = "base"
    #: True when the data describes real exchange activity. Simulated sources
    #: set this False and every downstream report marks the result accordingly.
    is_real: bool = True

    @abstractmethod
    def fetch(self, ticker: str, start: Optional[pd.Timestamp] = None,
              end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        ...

    def available(self) -> bool:
        return True

    def describe(self) -> str:
        return self.name


class CsvBrokerSummary(BrokerSummaryProvider):
    """Reads broker summary you exported from your own trading platform.

    Layout (either works)::

        data/broker_summary/BBCA.csv          # one file per ticker
        data/broker_summary/2026-08-07.csv    # one file per day, many tickers

    Any column naming in ``COLUMN_ALIASES`` is understood, in English or
    Indonesian, with lot- or share-denominated volume.
    """

    name = "csv"
    is_real = True

    def __init__(self, directory: str, volume_unit: str = "auto"):
        self.directory = directory
        self.volume_unit = volume_unit
        self._cache: Optional[pd.DataFrame] = None

    def available(self) -> bool:
        return bool(self._files())

    def _files(self) -> List[str]:
        if not os.path.isdir(self.directory):
            return []
        files: List[str] = []
        for pattern in ("*.csv", "*.CSV", "*.txt", "*.xlsx", "*.xls"):
            files.extend(glob.glob(os.path.join(self.directory, pattern)))
            files.extend(glob.glob(os.path.join(self.directory, "**", pattern), recursive=True))
        return sorted(set(files))

    def _load_all(self) -> pd.DataFrame:
        if self._cache is not None:
            return self._cache
        frames = []
        for path in self._files():
            stem = os.path.splitext(os.path.basename(path))[0].upper()
            # A filename that looks like a ticker supplies the ticker for files
            # whose contents omit that column.
            implied = stem if re.fullmatch(r"[A-Z]{4}", stem) else None
            try:
                if path.lower().endswith((".xlsx", ".xls")):
                    raw = pd.read_excel(path)
                else:
                    raw = pd.read_csv(path, sep=None, engine="python")
                frames.append(
                    normalise(raw, ticker=implied, source=f"csv:{os.path.basename(path)}",
                              volume_unit=self.volume_unit)
                )
            except Exception as exc:
                print(f"  ! skipping {os.path.basename(path)}: {exc}")
        self._cache = pd.concat(frames, ignore_index=True) if frames else empty_frame()
        return self._cache

    def fetch(self, ticker: str, start=None, end=None) -> pd.DataFrame:
        df = self._load_all()
        if df.empty:
            return empty_frame()
        out = df[df["ticker"] == str(ticker).upper()]
        return _slice_dates(out, start, end)

    def describe(self) -> str:
        n = len(self._files())
        return f"csv ({n} file{'s' if n != 1 else ''} in {self.directory})"


class GoApiBrokerSummary(BrokerSummaryProvider):
    """GoAPI.id broker-summary endpoint (commercial, requires an API key).

    Probing from this environment, ``/stock/idx/{symbol}/broker_summary``
    answered ``401 invalid API key`` rather than 404 - the route is live and
    gated on a key. Register at https://goapi.id/ and set::

        export IDXBOT_GOAPI_KEY=your_key

    Response shapes differ by plan, so the parser is deliberately forgiving and
    hands whatever it finds to :func:`normalise`.
    """

    name = "goapi"
    is_real = True
    BASE = "https://api.goapi.io/stock/idx"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get("IDXBOT_GOAPI_KEY", "")
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def fetch(self, ticker: str, start=None, end=None) -> pd.DataFrame:
        if not self.available():
            return empty_frame()
        import requests

        params = {"api_key": self.api_key}
        if start is not None:
            params["from"] = pd.Timestamp(start).strftime("%Y-%m-%d")
        if end is not None:
            params["to"] = pd.Timestamp(end).strftime("%Y-%m-%d")
        url = f"{self.BASE}/{str(ticker).upper()}/broker_summary"
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            if resp.status_code != 200:
                print(f"  ! goapi {ticker}: HTTP {resp.status_code} {resp.text[:120]}")
                return empty_frame()
            payload = resp.json()
        except Exception as exc:
            print(f"  ! goapi {ticker}: {exc}")
            return empty_frame()

        records = _find_records(payload)
        if not records:
            return empty_frame()
        return normalise(pd.DataFrame(records), ticker=ticker, source="goapi")

    def describe(self) -> str:
        return "goapi" + ("" if self.available() else " (no IDXBOT_GOAPI_KEY set)")


def _find_records(payload) -> List[dict]:
    """Pull the first list-of-dicts out of an arbitrary JSON envelope."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("data", "results", "result", "broker_summary", "brokers", "items"):
            if key in payload:
                found = _find_records(payload[key])
                if found:
                    return found
        for value in payload.values():
            if isinstance(value, (list, dict)):
                found = _find_records(value)
                if found:
                    return found
    return []


def _slice_dates(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if df.empty:
        return df
    out = df
    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end)]
    return out.reset_index(drop=True)


class RestBrokerSummary(BrokerSummaryProvider):
    """Generic REST adapter for any commercial broker-summary vendor.

    Several vendors sell exactly this data (see ``docs/LIVE_DATA.md``), but each
    uses its own paths, auth style and field names, and their docs sit behind
    signup. Rather than hardcode a guessed endpoint that breaks on contact with
    reality, this is driven entirely from config::

        data:
          rest_broker_summary:
            url: "https://api.vendor.com/v1/broker-summary/{ticker}"
            api_key_env: "IDXBOT_VENDOR_KEY"
            auth: "bearer"            # bearer | header | query
            auth_param: "api_key"     # for header/query styles
            date_params: {from: "start", to: "end"}
            date_format: "%Y-%m-%d"

    ``{ticker}`` and ``{date}`` are substituted into the URL. The response is
    handed to :func:`_find_records` and :func:`normalise`, which already cope
    with arbitrary envelopes and both English and Indonesian field names, so a
    new vendor is usually zero code.
    """

    name = "rest"
    is_real = True

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.settings = dict(cfg.get("data.rest_broker_summary", {}) or {})
        env = self.settings.get("api_key_env", "IDXBOT_BROKER_API_KEY")
        self.api_key = os.environ.get(env, "")
        self.timeout = int(cfg.get("data.request_timeout", 30))

    def available(self) -> bool:
        return bool(self.settings.get("url"))

    def fetch(self, ticker: str, start=None, end=None) -> pd.DataFrame:
        if not self.available():
            return empty_frame()
        import requests

        url_template = str(self.settings["url"])
        date_format = self.settings.get("date_format", "%Y-%m-%d")
        url = url_template.replace("{ticker}", str(ticker).upper())
        if "{date}" in url:
            stamp = pd.Timestamp(end or pd.Timestamp.now()).strftime(date_format)
            url = url.replace("{date}", stamp)

        headers, params = {}, {}
        auth = str(self.settings.get("auth", "bearer")).lower()
        if self.api_key:
            if auth == "bearer":
                headers["Authorization"] = f"Bearer {self.api_key}"
            elif auth == "header":
                headers[self.settings.get("auth_param", "X-API-Key")] = self.api_key
            else:
                params[self.settings.get("auth_param", "api_key")] = self.api_key

        date_params = self.settings.get("date_params") or {}
        if start is not None and date_params.get("from"):
            params[date_params["from"]] = pd.Timestamp(start).strftime(date_format)
        if end is not None and date_params.get("to"):
            params[date_params["to"]] = pd.Timestamp(end).strftime(date_format)

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=self.timeout)
            if resp.status_code != 200:
                print(f"  ! rest {ticker}: HTTP {resp.status_code} {resp.text[:140]}")
                return empty_frame()
            payload = resp.json()
        except Exception as exc:
            print(f"  ! rest {ticker}: {exc}")
            return empty_frame()

        records = _find_records(payload)
        if not records:
            print(f"  ! rest {ticker}: no records found in response envelope")
            return empty_frame()
        return normalise(pd.DataFrame(records), ticker=ticker,
                         source=f"rest:{self.settings.get('name', 'vendor')}")

    def describe(self) -> str:
        if not self.available():
            return "rest (no data.rest_broker_summary.url configured)"
        label = self.settings.get("name", self.settings.get("url", "vendor"))
        return f"rest:{label}" + ("" if self.api_key else " (no API key in env)")


class NullBrokerSummary(BrokerSummaryProvider):
    """Returns nothing, forcing the engine into price-only mode.

    Use this to evaluate the price/volume half of the signal on its own, with
    no simulated broker flow anywhere in the result.
    """

    name = "none"
    is_real = True   # an honest absence of data, not fabricated data

    def fetch(self, ticker: str, start=None, end=None) -> pd.DataFrame:
        return empty_frame()

    def describe(self) -> str:
        return "none (price-only mode: no broker data)"


class ChainedBrokerSummary(BrokerSummaryProvider):
    """Try providers in order; the first with data for a ticker wins."""

    name = "chain"

    def __init__(self, providers: Sequence[BrokerSummaryProvider]):
        self.providers = [p for p in providers if p is not None]

    @property
    def is_real(self) -> bool:  # type: ignore[override]
        return all(p.is_real for p in self.providers if p.available())

    def fetch(self, ticker: str, start=None, end=None) -> pd.DataFrame:
        for provider in self.providers:
            if not provider.available():
                continue
            df = provider.fetch(ticker, start, end)
            if df is not None and not df.empty:
                return df
        return empty_frame()

    def describe(self) -> str:
        return " -> ".join(p.describe() for p in self.providers)


def build_provider(cfg: Config, names: Optional[Sequence[str]] = None,
                   ohlcv: Optional[Dict[str, pd.DataFrame]] = None) -> ChainedBrokerSummary:
    """Construct the provider chain named in config (or overridden by ``names``)."""
    from .synthetic import SyntheticBrokerSummary  # imported late: optional path

    names = names or cfg.get("data.broker_summary_providers", ["csv", "synthetic"])
    providers: List[BrokerSummaryProvider] = []
    for name in names:
        name = str(name).lower()
        if name == "csv":
            providers.append(CsvBrokerSummary(cfg.path("data.csv_dir", "data/broker_summary")))
        elif name == "goapi":
            providers.append(GoApiBrokerSummary())
        elif name == "rest":
            providers.append(RestBrokerSummary(cfg))
        elif name == "synthetic":
            providers.append(SyntheticBrokerSummary(cfg, ohlcv=ohlcv))
        elif name == "none":
            # Deliberately empty: forces price-only mode, so results rest
            # entirely on genuine exchange data with nothing simulated.
            providers.append(NullBrokerSummary())
        else:
            raise ValueError(f"Unknown broker summary provider: {name!r}")
    return ChainedBrokerSummary(providers)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Attach net flow columns used throughout the analytics layer."""
    if df.empty:
        out = df.copy()
        for col in ("net_lot", "net_val", "total_lot", "total_val"):
            out[col] = pd.Series(dtype="float64")
        return out
    out = df.copy()
    out["net_lot"] = out["buy_lot"] - out["sell_lot"]
    out["net_val"] = out["buy_val"] - out["sell_val"]
    out["total_lot"] = out["buy_lot"] + out["sell_lot"]
    out["total_val"] = out["buy_val"] + out["sell_val"]
    return out
