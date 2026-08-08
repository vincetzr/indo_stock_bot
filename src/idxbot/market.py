"""IDX market microstructure: tick sizes, lots, auto-rejection limits, costs.

Getting these details right is the difference between a plan that can actually
be executed and one that gets rejected by the exchange. A limit order at a price
that is not on the tick grid is rejected outright, and a target that sits beyond
the auto-rejection band cannot print that day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .config import Config

# Fallback bands, used only if config is unavailable. [max_price_exclusive, tick]
DEFAULT_TICK_BANDS: List[Tuple[Optional[float], int]] = [
    (200, 1),
    (500, 2),
    (2000, 5),
    (5000, 10),
    (None, 25),
]

DEFAULT_ARA: List[Tuple[Optional[float], float]] = [
    (200, 0.35),
    (5000, 0.25),
    (None, 0.20),
]


def _bands(cfg: Optional[Config], dotted: str, fallback):
    if cfg is None:
        return fallback
    raw = cfg.get(dotted)
    if not raw:
        return fallback
    out = []
    for entry in raw:
        threshold, value = entry[0], entry[1]
        out.append((None if threshold is None else float(threshold), value))
    return out


def tick_size(price: float, cfg: Optional[Config] = None) -> int:
    """Return the IDX tick (fraksi harga) applicable at ``price``."""
    if price is None or not math.isfinite(price) or price <= 0:
        return 1
    for threshold, tick in _bands(cfg, "market.tick_bands", DEFAULT_TICK_BANDS):
        if threshold is None or price < threshold:
            return int(tick)
    return 25


def round_to_tick(price: float, cfg: Optional[Config] = None, mode: str = "nearest") -> float:
    """Snap a price onto the exchange tick grid.

    ``mode`` is ``nearest``, ``down`` (safer for stops on longs / buy limits) or
    ``up`` (safer for sell targets).

    The tick applicable to the *rounded* price can differ from the tick at the
    raw price when the raw price sits just above a band boundary, so the result
    is re-checked once against its own band.
    """
    if price is None or not math.isfinite(price) or price <= 0:
        return 0.0

    def _snap(p: float) -> float:
        t = tick_size(p, cfg)
        if mode == "down":
            return math.floor(p / t) * t
        if mode == "up":
            return math.ceil(p / t) * t
        return round(p / t) * t

    snapped = _snap(price)
    # Crossing a band boundary (e.g. 5000 -> tick 25 vs 4990 -> tick 10) can put
    # the result off-grid for its own band; one re-snap converges.
    if tick_size(snapped, cfg) != tick_size(price, cfg):
        snapped = _snap(snapped)
    return float(max(snapped, tick_size(price, cfg)))


def lots_to_shares(lots: float, cfg: Optional[Config] = None) -> float:
    lot_size = 100 if cfg is None else int(cfg.get("market.lot_size", 100))
    return float(lots) * lot_size


def shares_to_lots(shares: float, cfg: Optional[Config] = None) -> int:
    lot_size = 100 if cfg is None else int(cfg.get("market.lot_size", 100))
    return int(math.floor(float(shares) / lot_size))


def ara_pct(price: float, cfg: Optional[Config] = None) -> float:
    """Auto-rejection-atas (upper limit) percentage for a reference price."""
    for threshold, pct in _bands(cfg, "market.auto_rejection.ara", DEFAULT_ARA):
        if threshold is None or price < threshold:
            return float(pct)
    return 0.20


def arb_pct(cfg: Optional[Config] = None) -> float:
    if cfg is None:
        return 0.15
    return float(cfg.get("market.auto_rejection.arb_symmetric_pct", 0.15))


def auto_rejection_bounds(prev_close: float, cfg: Optional[Config] = None) -> Tuple[float, float]:
    """Return ``(arb_price, ara_price)`` - the day's tradeable price envelope."""
    up = prev_close * (1.0 + ara_pct(prev_close, cfg))
    down = prev_close * (1.0 - arb_pct(cfg))
    return round_to_tick(down, cfg, "up"), round_to_tick(up, cfg, "down")


def days_to_reach(prev_close: float, target: float, cfg: Optional[Config] = None) -> int:
    """Minimum trading days for a price to reach ``target`` at the ARA limit.

    A target that needs many consecutive limit-up days is not a realistic
    same-week objective; the trading plan surfaces this so a target of "+80% in
    3 days" is visibly flagged as mechanically impossible.
    """
    if target <= prev_close or prev_close <= 0:
        return 0
    price = prev_close
    days = 0
    while price < target and days < 100:
        price *= 1.0 + ara_pct(price, cfg)
        days += 1
    return days


@dataclass
class Costs:
    buy_fee_pct: float = 0.0015
    sell_fee_pct: float = 0.0025
    slippage_ticks: int = 1

    @classmethod
    def from_config(cls, cfg: Config) -> "Costs":
        return cls(
            buy_fee_pct=float(cfg.get("market.costs.buy_fee_pct", 0.0015)),
            sell_fee_pct=float(cfg.get("market.costs.sell_fee_pct", 0.0025)),
            slippage_ticks=int(cfg.get("market.costs.slippage_ticks", 1)),
        )

    @property
    def round_trip_pct(self) -> float:
        return self.buy_fee_pct + self.sell_fee_pct

    def breakeven_price(self, entry: float, cfg: Optional[Config] = None) -> float:
        """Price at which a long position clears fees on both sides."""
        gross = entry * (1.0 + self.buy_fee_pct) / (1.0 - self.sell_fee_pct)
        return round_to_tick(gross, cfg, "up")

    def net_return(self, entry: float, exit_price: float) -> float:
        """Fee-adjusted fractional return of a long round trip."""
        if entry <= 0:
            return 0.0
        paid = entry * (1.0 + self.buy_fee_pct)
        received = exit_price * (1.0 - self.sell_fee_pct)
        return received / paid - 1.0


def tradeability(bars, window: int = 20, min_value_traded: float = 1e9,
                 max_flat_share: float = 0.3, max_zero_volume: int = 2) -> dict:
    """Can this name actually be traded right now?

    A suspended stock is the trap this exists to catch. Its price is frozen and
    its volume is zero, which makes momentum indicators score it *highly*: a
    flat line sits above a declining moving average, its close equals its high
    forever, and its trend never breaks. WIKA in August 2026 is the worked
    example - 20 straight zero-volume days at 204, and it ranked into a
    long-horizon basket until this check was added.

    Returns a dict with ``tradeable`` plus the reasons it failed.
    """
    reasons = []
    if bars is None or len(bars) < 5:
        return {"tradeable": False, "reasons": ["insufficient history"]}

    recent = bars.tail(window)
    zero_volume = int((recent["volume"] <= 0).sum())
    if zero_volume > max_zero_volume:
        reasons.append(f"{zero_volume} of the last {len(recent)} sessions had no volume "
                       f"- likely suspended")

    flat = int((recent["high"] <= recent["low"]).sum())
    if flat / max(len(recent), 1) > max_flat_share:
        reasons.append(f"{flat} of the last {len(recent)} bars had zero range "
                       f"- not trading normally")

    value_traded = float((recent["close"] * recent["volume"]).median())
    if value_traded < min_value_traded:
        reasons.append(f"median turnover {format_idr(value_traded)}/day is below "
                       f"{format_idr(min_value_traded)}")

    return {
        "tradeable": not reasons,
        "reasons": reasons,
        "zero_volume_days": zero_volume,
        "flat_bars": flat,
        "median_value_traded": value_traded,
    }


def format_idr(value: float, short: bool = True) -> str:
    """Human-readable rupiah, using Indonesian scale words when ``short``."""
    if value is None or not math.isfinite(value):
        return "-"
    sign = "-" if value < 0 else ""
    v = abs(float(value))
    if not short:
        return f"{sign}Rp{v:,.0f}"
    for scale, suffix in ((1e12, " T"), (1e9, " M"), (1e6, " jt"), (1e3, " rb")):
        if v >= scale:
            return f"{sign}Rp{v / scale:,.2f}{suffix}"
    return f"{sign}Rp{v:,.0f}"


def position_size(
    equity: float,
    entry: float,
    stop: float,
    risk_pct: float,
    max_position_pct: float,
    cfg: Optional[Config] = None,
) -> Tuple[int, float, float]:
    """Size a long position by risk, rounded down to whole lots.

    Returns ``(lots, notional_idr, actual_risk_idr)``. Rounding down to a lot
    means the realised risk is always at or below the requested budget, never
    above it.
    """
    risk_per_share = entry - stop
    if risk_per_share <= 0 or entry <= 0:
        return 0, 0.0, 0.0

    risk_budget = equity * risk_pct
    shares_by_risk = risk_budget / risk_per_share
    shares_by_cap = (equity * max_position_pct) / entry
    shares = min(shares_by_risk, shares_by_cap)

    lots = shares_to_lots(shares, cfg)
    if lots <= 0:
        return 0, 0.0, 0.0

    actual_shares = lots_to_shares(lots, cfg)
    return lots, actual_shares * entry, actual_shares * risk_per_share


def price_percentile(price: float, window_low: float, window_high: float) -> float:
    """Where ``price`` sits inside a range, 0.0 (at low) to 1.0 (at high)."""
    if window_high <= window_low:
        return 0.5
    return float(min(1.0, max(0.0, (price - window_low) / (window_high - window_low))))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(min(high, max(low, value)))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator is None or denominator == 0 or not math.isfinite(denominator):
        return default
    result = numerator / denominator
    return result if math.isfinite(result) else default


def rolling_percentile_rank(series: Sequence[float], value: float) -> float:
    """Fraction of ``series`` at or below ``value``."""
    values = [v for v in series if v is not None and math.isfinite(v)]
    if not values:
        return 0.5
    below = sum(1 for v in values if v <= value)
    return below / len(values)
