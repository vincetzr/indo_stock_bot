"""Wyckoff accumulation-phase classification from daily bars.

Wyckoff's framework describes how a large operator absorbs supply before a
markup. The sequence, and what each event looks like in OHLCV:

  Phase A  stopping the decline
           SC  selling climax   - wide down bar, huge volume, close off the low
           AR  automatic rally  - sharp bounce that sets the range top
           ST  secondary test   - retest of the low on lighter volume
  Phase B  building the cause - sideways, volume drying up, repeated tests
  Phase C  the spring         - a probe below support that fails to follow
                                through; supply is exhausted
  Phase D  markup begins
           SOS last point of support / sign of strength - close above the range
                on expanding volume
  Phase E  markup proper

The classifier is heuristic, not doctrine: it looks for the mechanical
signature of each event and reports the furthest one it can justify, with a
confidence. It reads only bars up to the evaluation index, so it is safe to use
inside a backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PHASES = ["none", "A", "B", "C", "D", "E"]

# Human-readable meaning of each phase, surfaced in reports.
PHASE_MEANING: Dict[str, str] = {
    "none": "No identifiable accumulation structure",
    "A": "Decline being stopped - climax and automatic rally, too early to act",
    "B": "Cause being built - sideways absorption, the accumulation window",
    "C": "Spring / shakeout - supply tested one last time, the lowest-risk entry",
    "D": "Markup starting - strength off the range, entry on pullbacks",
    "E": "Markup underway - trend following, no longer an accumulation entry",
}


@dataclass
class WyckoffState:
    phase: str = "none"
    confidence: float = 0.0
    support: float = np.nan
    resistance: float = np.nan
    range_days: int = 0
    events: List[str] = field(default_factory=list)
    spring_price: float = np.nan
    spring_date: Optional[pd.Timestamp] = None
    sos_date: Optional[pd.Timestamp] = None
    notes: str = ""

    @property
    def is_actionable(self) -> bool:
        """Phases C and D are where the risk/reward of an entry is best."""
        return self.phase in ("C", "D")

    @property
    def meaning(self) -> str:
        return PHASE_MEANING.get(self.phase, "")

    def to_dict(self) -> dict:
        return {
            "wyckoff_phase": self.phase,
            "wyckoff_confidence": round(self.confidence, 3),
            "wyckoff_support": self.support,
            "wyckoff_resistance": self.resistance,
            "wyckoff_range_days": self.range_days,
            "wyckoff_events": ",".join(self.events),
            "wyckoff_spring_price": self.spring_price,
        }


def classify(df: pd.DataFrame, index: Optional[int] = None,
             lookback: int = 90, atr_col: str = "atr") -> WyckoffState:
    """Classify the accumulation phase as of ``index`` (default: the last bar)."""
    if df is None or len(df) < 40:
        return WyckoffState(notes="insufficient history")

    i = len(df) - 1 if index is None else int(index)
    if i < 40:
        return WyckoffState(notes="insufficient history")

    start = max(0, i - lookback + 1)
    window = df.iloc[start:i + 1]
    if len(window) < 30:
        return WyckoffState(notes="insufficient history")

    high = window["high"].to_numpy(float)
    low = window["low"].to_numpy(float)
    close = window["close"].to_numpy(float)
    volume = window["volume"].to_numpy(float)
    dates = window["date"].to_numpy()

    atr_series = window[atr_col] if atr_col in window.columns else (window["high"] - window["low"])
    atr = float(atr_series.tail(20).mean())
    avg_volume = float(np.nanmean(volume))
    if not np.isfinite(atr) or atr <= 0 or avg_volume <= 0:
        return WyckoffState(notes="degenerate bars")

    support = float(np.nanmin(low))
    resistance = float(np.nanmax(high))
    if resistance <= support:
        return WyckoffState(notes="degenerate range")

    state = WyckoffState(support=support, resistance=resistance, range_days=len(window))
    events: List[str] = []
    confidence = 0.0

    # --- prior decline: accumulation only makes sense after one ---------------
    prior_start = max(0, start - lookback)
    prior = df.iloc[prior_start:start]
    had_decline = False
    if len(prior) >= 20:
        peak = float(prior["close"].max())
        if peak > 0 and close[0] / peak - 1.0 < -0.12:
            had_decline = True
            events.append("prior decline")
            confidence += 0.10

    # --- Phase A: selling climax -------------------------------------------
    bar_range = high - low
    climax = (
        (volume > 2.2 * avg_volume)
        & (bar_range > 1.8 * atr)
        & ((close - low) / np.where(bar_range > 0, bar_range, np.nan) > 0.4)
    )
    climax_idx = np.where(climax)[0]
    # A climax must sit in the earlier part of the window; a wide high-volume
    # bar in the last few sessions is more likely a breakout than a bottom.
    climax_idx = climax_idx[climax_idx < len(window) * 0.7]
    if len(climax_idx):
        events.append("selling climax")
        confidence += 0.15
        state.phase = "A"

    # --- Phase B: sideways absorption on drying volume ----------------------
    range_pct = (resistance - support) / close[-1]
    recent_half = volume[len(volume) // 2:]
    early_half = volume[: len(volume) // 2]
    volume_drying = float(np.nanmean(recent_half)) < 0.85 * float(np.nanmean(early_half))

    slope = np.polyfit(np.arange(len(close)), close, 1)[0] / max(float(np.mean(close)), 1e-9)
    sideways = abs(slope) < 0.0015  # < ~0.15% drift per bar

    if sideways and range_pct < 0.45:
        events.append("trading range")
        confidence += 0.15
        if state.phase in ("none", "A"):
            state.phase = "B"
        if volume_drying:
            events.append("volume drying up")
            confidence += 0.12

    # --- Phase C: spring / shakeout ----------------------------------------
    # Look for a probe below the range support that closes back inside within a
    # few bars. The range support is measured excluding the probe itself.
    lookback_c = min(30, len(window) - 5)
    for k in range(len(window) - lookback_c, len(window) - 1):
        if k < 5:
            continue
        prior_support = float(np.nanmin(low[:k]))
        if prior_support <= 0:
            continue
        pierce = (prior_support - low[k]) / prior_support
        if 0 < pierce < 0.06:
            recovery = close[k + 1:min(k + 4, len(window))]
            if len(recovery) and float(np.nanmax(recovery)) > prior_support:
                events.append("spring")
                confidence += 0.20
                state.phase = "C"
                state.spring_price = float(low[k])
                state.spring_date = pd.Timestamp(dates[k])
                break

    # --- Phase D: sign of strength ------------------------------------------
    # A close above the range top (measured before the breakout bar) with
    # expanding volume.
    for k in range(max(len(window) - 20, 5), len(window)):
        prior_resistance = float(np.nanmax(high[:k]))
        if close[k] > prior_resistance and volume[k] > 1.6 * avg_volume:
            events.append("sign of strength")
            confidence += 0.20
            state.phase = "D"
            state.sos_date = pd.Timestamp(dates[k])
            break

    # --- Phase E: already marked up -----------------------------------------
    if state.phase == "D":
        gain_from_support = close[-1] / support - 1.0
        if gain_from_support > 0.35:
            state.phase = "E"
            events.append("markup extended")

    if had_decline and state.phase in ("B", "C", "D"):
        confidence += 0.08

    state.events = events
    state.confidence = float(np.clip(confidence, 0.0, 1.0))
    state.notes = PHASE_MEANING.get(state.phase, "")
    return state


def phase_score(state: WyckoffState) -> float:
    """Map a phase onto a 0-1 desirability score for the composite.

    C (spring) and D (breakout) score highest because they are the phases where
    the accumulation is confirmed but the markup has not yet run.
    """
    base = {"none": 0.0, "A": 0.25, "B": 0.55, "C": 0.95, "D": 0.85, "E": 0.30}
    return float(base.get(state.phase, 0.0) * (0.55 + 0.45 * state.confidence))
