"""Composite accumulation scoring.

Blends three independent evidence streams into one 0-100 score:

  * **Broker flow**  - institutional inventory building, buying concentration,
    institutions absorbing while retail supplies. Requires broker summary.
  * **Price/volume** - the classic footprints of absorption: volume drying up
    in a base, range compression, volume flow rising faster than price,
    relative strength versus the IHSG. Needs only OHLCV.
  * **Structure**    - the Wyckoff phase.

If no broker summary is available the engine runs in **price-only mode**: the
broker-dependent components are dropped and the remaining weights are
renormalised, so the score stays on the same 0-100 scale and stays comparable
across names. The mode is recorded on every result and printed on every report,
because a price-only 70 and a broker-confirmed 70 are not the same claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import BrokerRegistry, Config
from . import indicators, wyckoff
from .wyckoff import WyckoffState

# Components that cannot be computed without broker summary data.
BROKER_COMPONENTS = {
    "inventory_zscore", "stealth", "concentration", "smart_dumb_divergence",
}


def sigmoid(x: float, scale: float = 1.0) -> float:
    """Squash to (0, 1); NaN-safe and overflow-safe.

    Raw divergence inputs can be enormous when a denominator is near zero, and
    an unclipped exponent overflows to a RuntimeWarning. Clipping at +/-40
    saturates the result to 0 or 1 anyway, which is the intended behaviour.
    """
    if x is None or not np.isfinite(x):
        return 0.5
    z = float(np.clip(x / max(scale, 1e-9), -40.0, 40.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def _clamp01(x: float) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))


@dataclass
class AccumulationSignal:
    ticker: str
    date: pd.Timestamp
    close: float
    score: float
    components: Dict[str, float] = field(default_factory=dict)
    weights_used: Dict[str, float] = field(default_factory=dict)
    wyckoff_state: Optional[WyckoffState] = None
    flags: List[str] = field(default_factory=list)
    data_mode: str = "price-only"
    data_source: str = "none"
    data_is_real: bool = False
    top_buyer: str = ""
    top_buyer_share: float = 0.0
    bulge_net_60d: float = 0.0
    atr_pct: float = np.nan
    range_pos: float = np.nan
    profile: str = "accumulation"

    @property
    def level(self) -> str:
        if self.score >= 78:
            return "STRONG"
        if self.score >= 65:
            return "SIGNAL"
        if self.score >= 50:
            return "WATCH"
        return "NONE"

    def to_row(self) -> dict:
        row = {
            "ticker": self.ticker,
            "date": self.date,
            "close": self.close,
            "score": round(self.score, 1),
            "level": self.level,
            "data_mode": self.data_mode,
            "profile": self.profile,
            "data_source": self.data_source,
            "data_is_real": self.data_is_real,
            "top_buyer": self.top_buyer,
            "top_buyer_share": round(self.top_buyer_share, 3),
            "bulge_net_60d": self.bulge_net_60d,
            "atr_pct": self.atr_pct,
            "range_pos": self.range_pos,
            "flags": " | ".join(self.flags),
        }
        row.update({f"c_{k}": round(v, 3) for k, v in self.components.items()})
        if self.wyckoff_state is not None:
            row.update(self.wyckoff_state.to_dict())
        return row


def _normalise_weights(weights: Dict[str, float], available: set) -> Dict[str, float]:
    usable = {k: float(v) for k, v in weights.items() if k in available and v > 0}
    total = sum(usable.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in usable.items()}


def compute_components(
    bars: pd.DataFrame,
    index: int,
    flow: Optional[pd.DataFrame],
    cfg: Config,
) -> tuple:
    """Return ``(components, flags, extras)`` evaluated at ``bars.iloc[index]``."""
    lookback = int(cfg.get("accumulation.lookback", 60))
    row = bars.iloc[index]
    components: Dict[str, float] = {}
    flags: List[tuple] = []
    extras: Dict[str, float] = {}

    # ---- price / volume components (always available) ---------------------
    vol_ratio = row.get("vol_ratio", np.nan)
    components["volume_dryup"] = _clamp01((1.15 - vol_ratio) / 0.5) if np.isfinite(vol_ratio) else 0.0
    if np.isfinite(vol_ratio) and vol_ratio < 0.75:
        flags.append(("volume_dryup", f"volume dried up to {vol_ratio:.0%} of its norm"))

    compression = row.get("range_compression", np.nan)
    components["range_compression"] = (
        _clamp01((1.10 - compression) / 0.5) if np.isfinite(compression) else 0.0
    )
    if np.isfinite(compression) and compression < 0.7:
        flags.append(("range_compression", f"range compressed to {compression:.0%} of normal"))

    obv_div = row.get("obv_divergence", np.nan)
    components["obv_divergence"] = sigmoid(obv_div, scale=0.004)
    if np.isfinite(obv_div) and obv_div > 0.004:
        flags.append(("obv_divergence", "volume flow rising faster than price (absorption)"))

    rel = row.get("rel_strength", np.nan)
    components["relative_strength"] = sigmoid(rel, scale=0.08)
    if np.isfinite(rel) and rel > 0.05:
        flags.append(("relative_strength", f"outperforming IHSG by {rel:.0%} over {lookback}d"))

    # ---- momentum family --------------------------------------------------
    # These are trend-following by construction and pull in the opposite
    # direction to the accumulation components above. Which family the score
    # actually uses is decided by the weight profile, not by this function.
    mom = row.get("mom_12_1", np.nan)
    components["momentum"] = sigmoid(mom, scale=0.25)
    if np.isfinite(mom) and mom > 0.20:
        flags.append(("momentum", f"12-1 momentum {mom:+.0%}"))

    persistence = row.get("trend_persistence", np.nan)
    components["trend_persistence"] = _clamp01(persistence)
    if np.isfinite(persistence) and persistence > 0.75:
        flags.append(("trend_persistence", f"held above its 50d average {persistence:.0%} of the "
                      f"last {lookback * 2} sessions"))

    from_high = row.get("dist_from_high", np.nan)
    # 0 at the 52-week high, falling away below it.
    components["near_high"] = _clamp01(1.0 + from_high / 0.30) if np.isfinite(from_high) else 0.0
    if np.isfinite(from_high) and from_high > -0.05:
        flags.append(("near_high", "within 5% of its 52-week high"))

    extras["atr_pct"] = float(row.get("atr_pct", np.nan))
    extras["range_pos"] = float(row.get("range_pos_120", np.nan))

    # ---- broker-flow components (require broker summary) ------------------
    if flow is not None and not flow.empty:
        window = flow[flow["date"] <= row["date"]].tail(lookback)
        if len(window) >= max(10, lookback // 4):
            gross = float(window["gross_val"].sum())

            bulge_cum = float(window["bulge_net_val"].sum())
            extras["bulge_net_60d"] = bulge_cum
            # Compare this window's build against the same statistic measured
            # over the broker's own history in this name.
            history = flow[flow["date"] <= row["date"]]
            rolling = history["bulge_net_val"].rolling(lookback).sum().dropna()
            if len(rolling) >= 30:
                mean, std = rolling.mean(), rolling.std(ddof=0)
                z = (bulge_cum - mean) / std if std > 0 else 0.0
            else:
                z = 0.0
            components["inventory_zscore"] = sigmoid(z, scale=1.2)
            if z > 1.5:
                flags.append(("inventory_zscore", f"bulge-desk inventory build {z:+.1f} sigma vs its own history"))

            inst_share = (
                float(window["inst_net_val"].sum()) / gross if gross > 0 else 0.0
            )
            price_change = abs(float(bars["close"].iloc[index] / bars["close"].iloc[
                max(0, index - lookback)] - 1.0))
            # Buying hard while the price barely moves = absorption.
            stealth_penalty = 1.0 - _clamp01(price_change / 0.15)
            components["stealth"] = _clamp01(inst_share * 4.0) * stealth_penalty
            if inst_share > 0.05 and price_change < 0.08:
                flags.append((
                    "stealth",
                    f"institutions took {inst_share:.1%} of gross value while price moved "
                    f"only {price_change:.1%}",
                ))

            top5 = float(window["top5_buyer_share"].mean())
            components["concentration"] = _clamp01((top5 - 0.40) / 0.40)
            if top5 > 0.65:
                flags.append(("concentration", f"top-5 buyers absorbed {top5:.0%} of net buying"))

            recent = window.tail(max(5, lookback // 6))
            leader = recent["top_buyer"].mode()
            extras["top_buyer"] = str(leader.iloc[0]) if len(leader) else ""
            extras["top_buyer_share"] = float(recent["top_buyer_share"].mean())

            smart_dumb = float(window["smart_dumb_pct"].mean())
            components["smart_dumb_divergence"] = sigmoid(smart_dumb, scale=0.05)
            if smart_dumb > 0.04:
                flags.append(("smart_dumb_divergence", "institutions absorbing while retail distributes"))

    return components, flags, extras


def resolve_weights(cfg: Config, profile: Optional[str] = None) -> Dict[str, float]:
    """Return the weight set for ``profile``.

    Profiles let the same engine express different hypotheses. ``accumulation``
    is the original contrarian thesis; ``momentum`` is the trend-following one
    that cross-sectional testing selected. Falling back to the flat
    ``accumulation.weights`` block keeps older configs working.
    """
    profile = profile or str(cfg.get("accumulation.default_profile", "accumulation"))
    profiles = cfg.get("accumulation.profiles", {}) or {}
    if profile in profiles:
        return dict(profiles[profile] or {})

    flat = dict(cfg.get("accumulation.weights", {}) or {})
    if flat:
        flat.setdefault("wyckoff", 0.15)
        return flat
    raise KeyError(
        f"Unknown weight profile {profile!r}. Available: {sorted(profiles)}"
    )


def score(
    bars: pd.DataFrame,
    cfg: Config,
    flow: Optional[pd.DataFrame] = None,
    index: Optional[int] = None,
    ticker: str = "",
    data_source: str = "none",
    data_is_real: bool = False,
    profile: Optional[str] = None,
) -> AccumulationSignal:
    """Score one ticker at one point in time.

    ``bars`` must already carry indicator columns (see ``indicators.enrich``).
    Only rows up to ``index`` are consulted, so this is backtest-safe.

    ``profile`` selects a weight set from ``accumulation.profiles`` in config.
    """
    if bars is None or bars.empty:
        return AccumulationSignal(ticker=ticker, date=pd.NaT, close=np.nan, score=0.0)

    i = len(bars) - 1 if index is None else int(index)
    i = max(0, min(i, len(bars) - 1))
    row = bars.iloc[i]

    components, flags, extras = compute_components(bars, i, flow, cfg)

    state = wyckoff.classify(bars.iloc[:i + 1], lookback=int(cfg.get("accumulation.lookback", 60)) + 30)
    components["wyckoff"] = wyckoff.phase_score(state)
    if state.phase in ("C", "D"):
        flags.append(("wyckoff", f"Wyckoff phase {state.phase}: {state.meaning.split(' - ')[0]}"))

    configured = resolve_weights(cfg, profile)
    available = set(components)
    weights = _normalise_weights(configured, available)

    # Only surface evidence for components the active profile actually uses.
    # A flag from a zero-weight component reads as support for the score while
    # contributing nothing to it - actively misleading under a profile that
    # deliberately excludes that family.
    visible_flags = [text for component, text in flags if weights.get(component, 0.0) > 0]

    total = sum(weights.get(k, 0.0) * components.get(k, 0.0) for k in weights)
    broker_mode = bool(available & BROKER_COMPONENTS)

    return AccumulationSignal(
        ticker=ticker or str(row.get("ticker", "")),
        date=pd.Timestamp(row["date"]),
        close=float(row["close"]),
        score=float(np.clip(total * 100.0, 0.0, 100.0)),
        components=components,
        weights_used=weights,
        wyckoff_state=state,
        flags=visible_flags,
        data_mode="broker+price" if broker_mode else "price-only",
        profile=profile or str(cfg.get("accumulation.default_profile", "accumulation")),
        data_source=data_source,
        data_is_real=data_is_real,
        top_buyer=str(extras.get("top_buyer", "")),
        top_buyer_share=float(extras.get("top_buyer_share", 0.0)),
        bulge_net_60d=float(extras.get("bulge_net_60d", 0.0)),
        atr_pct=float(extras.get("atr_pct", np.nan)),
        range_pos=float(extras.get("range_pos", np.nan)),
    )


def score_series(
    bars: pd.DataFrame,
    cfg: Config,
    flow: Optional[pd.DataFrame] = None,
    ticker: str = "",
    start_index: int = 250,
    step: int = 1,
    include_components: bool = True,
    profile: Optional[str] = None,
) -> pd.DataFrame:
    """Score every ``step``-th bar - the input to the backtester.

    Deliberately loops rather than vectorising: the Wyckoff classifier is
    path-dependent and must see a growing window, and correctness matters more
    here than speed.

    ``include_components`` emits each raw component as a ``c_*`` column, which
    is what lets the analysis tools measure whether an individual component
    carries any predictive power on its own rather than only in aggregate.
    """
    if bars is None or len(bars) <= start_index:
        return pd.DataFrame()

    rows = []
    for i in range(start_index, len(bars), max(1, step)):
        signal = score(bars, cfg, flow=flow, index=i, ticker=ticker, profile=profile)
        row = {
            "date": signal.date,
            "ticker": signal.ticker or ticker,
            "close": signal.close,
            "score": signal.score,
            "level": signal.level,
            "data_mode": signal.data_mode,
            "profile": signal.profile,
            "wyckoff_phase": signal.wyckoff_state.phase if signal.wyckoff_state else "none",
        }
        if include_components:
            row.update({f"c_{k}": v for k, v in signal.components.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_bars(bars: pd.DataFrame, cfg: Config,
                 benchmark: Optional[pd.Series] = None) -> pd.DataFrame:
    """Convenience wrapper attaching the indicator set."""
    return indicators.enrich(bars, cfg=cfg, benchmark=benchmark)
