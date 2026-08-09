"""Macro and foreign-flow context for IDX.

Indonesia is a commodity-exporting emerging market with an open capital account,
which makes its equity index unusually exposed to things that have nothing to do
with any Indonesian company: the price of coal and copper, the dollar, US real
yields, and whether global funds are adding or cutting emerging-market risk. A
technical score computed from IDX prices alone is, in effect, guessing at those
forces from their shadow.

**On "foreign accumulation".** No broker-summary source was ever obtainable, so
foreign net buy/sell per stock is still unmeasured and nothing here should be
read as measuring it. What *is* measurable is the constraint every foreign buyer
faces: to own an Indonesian share you must first own rupiah. Sustained foreign
accumulation therefore shows up as rupiah strength alongside EM risk appetite,
and sustained foreign distribution as the reverse. USDIDR combined with EM
equity performance and the dollar index is a genuine proxy for foreign
*appetite* — it is not, and cannot substitute for, per-broker flow.

Every feature here is **trailing**. A macro series is only allowed to contribute
what was published on or before the decision date, and index levels are aligned
by forward-fill from the last known observation, never interpolated backwards.
Interpolating a macro series across a gap is one of the quietest ways to leak
the future into a backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .data.ohlcv import YahooOHLCV

# Symbol -> (label, why it matters for IDX specifically).
SERIES: Dict[str, tuple] = {
    "USDIDR=X":  ("rupiah", "foreign buyers must buy IDR first; weakness = outflow"),
    "DX-Y.NYB":  ("dollar index", "a strong dollar drains EM equity allocations"),
    "^TNX":      ("US 10y yield", "the discount rate EM risk assets compete with"),
    "CL=F":      ("WTI crude", "energy is a top-weight IDX sector"),
    "HG=F":      ("copper", "global industrial demand, and IDX mining revenue"),
    "GC=F":      ("gold", "risk-off gauge and a real IDX sector"),
    "EEM":       ("EM equities", "the allocation bucket IDX sits inside"),
    "^VIX":      ("VIX", "global risk appetite"),
    "^JKSE":     ("IHSG", "the market's own trend"),
    "FCX":       ("Freeport", "operates Grasberg; a listed read on IDN mining"),
    "^KLSE":     ("Malaysia", "nearest comparable commodity exporter"),
    "^STI":      ("Singapore", "regional risk proxy with a long history"),
}

# How far a stale macro observation may be carried forward before it is dropped.
# A week covers holidays and exchange-calendar mismatches; beyond that the
# series has genuinely stopped reporting and pretending otherwise is fiction.
MAX_STALE_DAYS = 7


@dataclass
class MacroPanel:
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    available: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return self.frame is None or self.frame.empty


def fetch(cfg: Config, symbols: Optional[List[str]] = None,
          verbose: bool = False) -> MacroPanel:
    """Pull every macro series and align them onto one daily calendar."""
    provider = YahooOHLCV(cfg)
    wanted = symbols or list(SERIES)

    frames, available, missing = [], [], []
    for symbol in wanted:
        try:
            df = provider.get(symbol)
        except Exception:
            df = None
        if df is None or df.empty or "close" not in df:
            missing.append(symbol)
            continue
        label = SERIES.get(symbol, (symbol, ""))[0]
        s = df[["date", "close"]].rename(columns={"close": label})
        frames.append(s.set_index("date")[label])
        available.append(label)
        if verbose:
            print(f"  {label:<16}{len(df):>7} bars  {df['date'].min():%Y-%m}")

    if not frames:
        return MacroPanel(missing=missing)

    panel = pd.concat(frames, axis=1).sort_index()
    # Forward-fill only. Every value on a given date was published on or before
    # that date; a back-fill would import a number from the future.
    panel = panel.ffill(limit=MAX_STALE_DAYS)
    return MacroPanel(frame=panel, available=available, missing=missing)


def features(panel: MacroPanel) -> pd.DataFrame:
    """Trailing macro state: trends, levels and the foreign-appetite proxy."""
    if panel.empty:
        return pd.DataFrame()
    f = panel.frame
    out = pd.DataFrame(index=f.index)

    def chg(col: str, days: int) -> Optional[pd.Series]:
        return f[col].pct_change(days) if col in f else None

    def put(name: str, series: Optional[pd.Series]) -> None:
        if series is not None:
            out[name] = series

    put("idr_20d", chg("rupiah", 20))       # positive = rupiah WEAKENING
    put("idr_60d", chg("rupiah", 60))
    put("dxy_60d", chg("dollar index", 60))
    put("oil_60d", chg("WTI crude", 60))
    put("copper_60d", chg("copper", 60))
    put("gold_60d", chg("gold", 60))
    put("em_60d", chg("EM equities", 60))
    put("fcx_60d", chg("Freeport", 60))
    put("ihsg_20d", chg("IHSG", 20))
    put("ihsg_60d", chg("IHSG", 60))

    if "IHSG" in f:
        ma200 = f["IHSG"].rolling(200, min_periods=100).mean()
        out["ihsg_above_200d"] = (f["IHSG"] > ma200).astype(float)
        out["ihsg_vs_200d"] = f["IHSG"] / ma200 - 1.0
    if "VIX" in f:
        out["vix"] = f["VIX"]
        out["vix_pctile"] = f["VIX"].rolling(500, min_periods=200).rank(pct=True)
    if "US 10y yield" in f:
        out["us10y"] = f["US 10y yield"]
        out["us10y_60d"] = f["US 10y yield"].diff(60)

    # The foreign-appetite proxy: rupiah strength AND emerging-market strength
    # pointing the same way. Either alone is ambiguous - the rupiah moves on
    # trade and policy as well as portfolio flow, and EM moves on China. Both
    # together is the closest thing to "foreign money is arriving" that is
    # measurable without broker summary.
    if {"idr_60d", "em_60d"} <= set(out.columns):
        # Ranked against the PAST only. A plain .rank(pct=True) scores each date
        # against the whole series including dates that had not happened yet, so
        # "the rupiah is in its strongest quartile" would be a judgement made
        # with knowledge of the next twenty years. A trailing window is the
        # percentile an observer could actually have computed that morning.
        out["foreign_appetite"] = (
            _trailing_rank(-out["idr_60d"]) * 0.5 + _trailing_rank(out["em_60d"]) * 0.5
        )
    return out


def _trailing_rank(series: pd.Series, window: int = 500,
                   min_periods: int = 200) -> pd.Series:
    """Percentile of each value within the preceding ``window`` observations."""
    return series.rolling(window, min_periods=min_periods).rank(pct=True)


def align_to(features_df: pd.DataFrame, dates) -> pd.DataFrame:
    """Attach macro state to decision dates, using only prior publications.

    ``reindex(method="ffill")`` is the whole safety property: a decision on a
    date the macro series did not publish inherits the *previous* value, never
    the next one.
    """
    if features_df is None or features_df.empty:
        return pd.DataFrame()
    idx = pd.DatetimeIndex(sorted(pd.to_datetime(list(dates))))
    return features_df.reindex(idx, method="ffill")


def render(panel: MacroPanel, feats: pd.DataFrame, width: int = 78) -> str:
    line = "=" * width
    out = [line, " MACRO CONTEXT", line]
    if panel.empty:
        return "\n".join(out + [" No macro series retrieved.", line])

    out.append(f" series    : {len(panel.available)} retrieved"
               + (f", {len(panel.missing)} unavailable" if panel.missing else ""))
    out.append(f" span      : {panel.frame.index.min():%Y-%m} -> "
               f"{panel.frame.index.max():%Y-%m}")

    if feats is None or feats.empty:
        return "\n".join(out + [line])

    latest = feats.dropna(how="all").iloc[-1]
    out.append(f" as of     : {feats.dropna(how='all').index[-1]:%Y-%m-%d}")
    out.append("")
    out.append(" CURRENT STATE")

    def show(key: str, label: str, pct: bool = True, invert: bool = False) -> None:
        if key not in latest or not np.isfinite(latest[key]):
            return
        v = float(latest[key])
        arrow = "up  " if v > 0 else ("down" if v < 0 else "flat")
        reading = ""
        if invert:
            reading = "  (headwind)" if v > 0 else "  (tailwind)"
        out.append(f"   {label:<22}{v:>8.1%} {arrow}{reading}"
                   if pct else f"   {label:<22}{v:>8.2f} {arrow}{reading}")

    show("idr_60d", "rupiah 60d", invert=True)
    show("dxy_60d", "dollar index 60d", invert=True)
    show("oil_60d", "WTI crude 60d")
    show("copper_60d", "copper 60d")
    show("em_60d", "EM equities 60d")
    show("ihsg_60d", "IHSG 60d")
    show("ihsg_vs_200d", "IHSG vs 200d MA")
    show("vix", "VIX", pct=False)

    if "foreign_appetite" in latest and np.isfinite(latest["foreign_appetite"]):
        fa = float(latest["foreign_appetite"])
        mood = "risk-on" if fa > 0.6 else ("risk-off" if fa < 0.4 else "neutral")
        out.append("")
        out.append(f"   foreign appetite proxy {fa:>7.2f}  ({mood})")
        out.append("   Rupiah and EM equity direction combined. A PROXY for foreign")
        out.append("   appetite, not a measurement of foreign flow - per-broker data")
        out.append("   remains unavailable.")

    out.append(line)
    return "\n".join(out)
