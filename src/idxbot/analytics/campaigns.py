"""Campaign segmentation: how a broker enters, holds and takes profit.

A "campaign" is one complete round trip in a broker's inventory: a trough, a
build to a peak, then an unwind back toward a trough. Segmenting a ledger this
way converts a noisy daily flow series into a handful of discrete, measurable
episodes - and it is those episodes that reveal a desk's operating pattern.

The metrics computed per campaign are chosen to answer four questions:

  1. **Where do they buy?**   ``entry_percentile`` - where their own buy VWAP
     sat inside the trailing range when they started.
  2. **Do they push the price while buying?**  ``stealth_ratio`` - price change
     during accumulation relative to the campaign's full move. Low means they
     absorbed supply without paying up.
  3. **When do they sell?**  ``markup_pct`` (how far it ran before they
     started), and ``exit_vs_peak_days`` (did they leave before or after the
     high).
  4. **How much of the move do they keep?**  ``exit_capture`` - realised gain
     divided by the maximum available gain. This is the single most useful
     number in the file: it separates desks that top-tick from desks that
     scale out early into strength.

Everything here is descriptive. It measures what happened in the data you
supply. Fed simulated data it will faithfully describe the simulation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..data.broker_summary import LOT_SIZE


@dataclass
class Campaign:
    ticker: str
    broker: str
    acc_start: pd.Timestamp
    acc_end: pd.Timestamp
    dist_end: Optional[pd.Timestamp]
    complete: bool                 # False while the campaign is still running

    acc_days: int
    dist_days: int
    holding_days: int

    lots_accumulated: float
    peak_inventory: float
    value_accumulated: float

    entry_vwap: float
    exit_vwap: float
    price_at_start: float
    price_at_acc_end: float
    price_peak: float
    price_at_exit: float

    entry_percentile: float        # 0 = bought at the range low, 1 = at the high
    acc_price_change: float        # price move during the accumulation leg
    stealth_ratio: float           # acc move / total campaign move
    markup_pct: float              # peak vs entry VWAP
    realized_return: float         # exit VWAP vs entry VWAP
    exit_capture: float            # realised / maximum available
    exit_vs_peak_days: int         # negative = sold before the high
    max_adverse_pct: float         # worst drawdown vs entry VWAP while holding
    participation_pct: float       # share of traded volume during accumulation

    def to_dict(self) -> dict:
        return asdict(self)


def _detrend(inventory: np.ndarray, span: int) -> np.ndarray:
    """Remove a broker's structural drift so campaigns stand out.

    Broker summary measures flow, so a member that is a standing conduit for
    index or custody business accumulates a persistent trend that is not a
    trading campaign. Subtracting a slow EWMA leaves the deviations from that
    baseline - which is what a campaign actually is.
    """
    if span <= 1 or len(inventory) < 5:
        return np.asarray(inventory, dtype=float)
    baseline = pd.Series(inventory).ewm(span=span, adjust=False, min_periods=1).mean()
    return (pd.Series(inventory) - baseline).to_numpy(float)


def find_pivots(values: np.ndarray, threshold: float) -> List[tuple]:
    """Zigzag pivot detection.

    Returns ``[(index, value, kind)]`` with ``kind`` in ``{"low", "high"}``,
    strictly alternating. A pivot is confirmed only once the series retraces
    ``threshold`` (an absolute amount) away from the running extreme, which is
    what stops day-to-day noise from creating spurious campaigns.
    """
    n = len(values)
    if n < 3 or threshold <= 0:
        return []

    pivots: List[tuple] = []
    extreme_idx = 0
    extreme_val = values[0]
    direction = 0  # 0 unknown, +1 rising, -1 falling

    for i in range(1, n):
        value = values[i]
        if direction >= 0 and value > extreme_val:
            extreme_idx, extreme_val = i, value
            direction = 1 if direction == 0 else direction
        elif direction <= 0 and value < extreme_val:
            extreme_idx, extreme_val = i, value
            direction = -1 if direction == 0 else direction

        if direction > 0 and extreme_val - value >= threshold:
            pivots.append((extreme_idx, extreme_val, "high"))
            direction, extreme_idx, extreme_val = -1, i, value
        elif direction < 0 and value - extreme_val >= threshold:
            pivots.append((extreme_idx, extreme_val, "low"))
            direction, extreme_idx, extreme_val = 1, i, value

    # Close out the series with the running extreme so an in-progress campaign
    # is still visible.
    if pivots and pivots[-1][0] != extreme_idx:
        pivots.append((extreme_idx, extreme_val, "high" if direction > 0 else "low"))
    elif not pivots:
        kind = "high" if values[extreme_idx] >= values[0] else "low"
        pivots.append((extreme_idx, extreme_val, kind))

    # Seed the opening extreme. Without this, a series that rises from its very
    # first bar produces a "high" as its first pivot and the leg that built it
    # is invisible - which silently drops the opening campaign of every ledger.
    if pivots and pivots[0][0] > 0:
        head = values[: pivots[0][0] + 1]
        if pivots[0][2] == "high":
            seed_idx = int(np.argmin(head))
            if seed_idx < pivots[0][0]:
                pivots.insert(0, (seed_idx, values[seed_idx], "low"))
        else:
            seed_idx = int(np.argmax(head))
            if seed_idx < pivots[0][0]:
                pivots.insert(0, (seed_idx, values[seed_idx], "high"))

    # Enforce strict alternation, keeping the more extreme of any duplicate pair.
    cleaned: List[tuple] = []
    for pivot in pivots:
        if cleaned and cleaned[-1][2] == pivot[2]:
            better = pivot[1] > cleaned[-1][1] if pivot[2] == "high" else pivot[1] < cleaned[-1][1]
            if better:
                cleaned[-1] = pivot
        else:
            cleaned.append(pivot)
    return cleaned


def extract_campaigns(
    ledger: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: Config,
    ticker: Optional[str] = None,
    broker: Optional[str] = None,
) -> pd.DataFrame:
    """Segment every (ticker, broker) ledger into campaigns."""
    if ledger is None or ledger.empty:
        return pd.DataFrame()

    df = ledger
    if ticker:
        df = df[df["ticker"] == str(ticker).upper()]
    if broker:
        df = df[df["broker"] == str(broker).upper()]
    if df.empty:
        return pd.DataFrame()

    retrace = float(cfg.get("campaigns.zigzag_retrace", 0.25))
    min_peak_share = float(cfg.get("campaigns.min_peak_share", 0.15))
    min_acc_days = int(cfg.get("campaigns.min_accumulation_days", 5))
    max_acc_days = int(cfg.get("campaigns.max_accumulation_days", 180))
    pct_window = int(cfg.get("campaigns.entry_percentile_window", 120))
    detrend_span = int(cfg.get("campaigns.detrend_span", 250))

    price_index = (
        prices.assign(date=pd.to_datetime(prices["date"]).dt.normalize())
        .drop_duplicates(subset=["date"])
        .set_index("date")
    )

    records: List[dict] = []
    for (tkr, brk), group in df.groupby(["ticker", "broker"], sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        if len(group) < min_acc_days * 3:
            continue

        inventory = group["inventory_lot"].to_numpy(float)
        signal = _detrend(inventory, detrend_span)

        # Threshold from a robust spread (p90-p10) rather than min-max: a single
        # outlier day must not set the scale for the whole series.
        spread = float(np.percentile(signal, 90) - np.percentile(signal, 10))
        if spread <= 0:
            continue

        pivots = find_pivots(signal, threshold=spread * retrace)
        if len(pivots) < 2:
            continue

        for k in range(len(pivots) - 1):
            start_idx, _start_val, start_kind = pivots[k]
            end_idx, _end_val, end_kind = pivots[k + 1]
            if start_kind != "low" or end_kind != "high":
                continue

            # Size the leg on the RAW inventory: that is the number of lots the
            # broker actually took on, independent of the detrending.
            lots = float(inventory[end_idx] - inventory[start_idx])
            if lots <= 0 or lots < spread * min_peak_share:
                continue
            if not (min_acc_days <= end_idx - start_idx <= max_acc_days):
                continue

            # The unwind runs to the next trough, or to the end of data if the
            # campaign has not closed yet.
            if k + 2 < len(pivots):
                exit_idx = pivots[k + 2][0]
                complete = True
            else:
                exit_idx = len(group) - 1
                complete = False

            record = _measure(
                group, price_index, tkr, brk,
                start_idx, end_idx, exit_idx, complete, lots, pct_window,
            )
            if record is not None:
                records.append(record)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(["ticker", "broker", "acc_start"]).reset_index(
        drop=True
    )


def _measure(
    group: pd.DataFrame,
    price_index: pd.DataFrame,
    ticker: str,
    broker: str,
    start_idx: int,
    end_idx: int,
    exit_idx: int,
    complete: bool,
    lots: float,
    pct_window: int,
) -> Optional[dict]:
    acc = group.iloc[start_idx:end_idx + 1]
    dist = group.iloc[end_idx:exit_idx + 1]

    acc_buy_shares = float(acc["buy_lot"].sum()) * LOT_SIZE
    entry_vwap = float(acc["buy_val"].sum()) / acc_buy_shares if acc_buy_shares > 0 else 0.0
    if entry_vwap <= 0:
        # Fall back to the ledger's own cost basis at the inventory peak.
        entry_vwap = float(group["avg_cost"].iloc[end_idx])
    if entry_vwap <= 0:
        return None

    dist_sell_shares = float(dist["sell_lot"].sum()) * LOT_SIZE
    exit_vwap = float(dist["sell_val"].sum()) / dist_sell_shares if dist_sell_shares > 0 else 0.0

    acc_start = pd.Timestamp(group["date"].iloc[start_idx])
    acc_end = pd.Timestamp(group["date"].iloc[end_idx])
    dist_end = pd.Timestamp(group["date"].iloc[exit_idx])

    price_at_start = float(group["close"].iloc[start_idx])
    price_at_acc_end = float(group["close"].iloc[end_idx])
    price_at_exit = float(group["close"].iloc[exit_idx])

    # Highest close between the inventory peak and the unwind - the best the
    # broker could plausibly have sold into.
    hold_window = group.iloc[end_idx:exit_idx + 1]
    price_peak = float(hold_window["close"].max()) if len(hold_window) else price_at_acc_end
    peak_offset = int(hold_window["close"].to_numpy().argmax()) if len(hold_window) else 0
    peak_date = pd.Timestamp(group["date"].iloc[end_idx + peak_offset])

    # Where their entry VWAP sat inside the range they were actually buying in:
    # the accumulation window itself plus the prior context window. Measuring
    # against only the pre-start range clips to 0 or 1 whenever the build runs
    # long enough for price to leave that earlier range.
    context = price_index.loc[:acc_start].tail(pct_window)
    buying = price_index.loc[acc_start:acc_end]
    window = pd.concat([context, buying]) if len(context) else buying
    if len(window) >= 10:
        low, high = float(window["low"].min()), float(window["high"].max())
        entry_percentile = (entry_vwap - low) / (high - low) if high > low else 0.5
    else:
        entry_percentile = 0.5
    entry_percentile = float(np.clip(entry_percentile, 0.0, 1.0))

    acc_price_change = price_at_acc_end / price_at_start - 1.0 if price_at_start > 0 else 0.0
    total_move = price_peak / price_at_start - 1.0 if price_at_start > 0 else 0.0
    # Low stealth_ratio = most of the move happened AFTER they finished buying.
    stealth_ratio = float(acc_price_change / total_move) if abs(total_move) > 1e-9 else 1.0

    markup_pct = price_peak / entry_vwap - 1.0
    realized_return = (exit_vwap / entry_vwap - 1.0) if exit_vwap > 0 else np.nan

    # exit_capture is only defined when there was a gain available to capture.
    # If price never traded above their entry VWAP the campaign simply failed,
    # and forcing a ratio there would pollute the median.
    max_gain = price_peak - entry_vwap
    if exit_vwap > 0 and max_gain > 0:
        exit_capture = float(np.clip((exit_vwap - entry_vwap) / max_gain, -2.0, 2.0))
    else:
        exit_capture = np.nan

    hold_lows = group["close"].iloc[start_idx:exit_idx + 1]
    max_adverse = float(hold_lows.min() / entry_vwap - 1.0) if len(hold_lows) else 0.0

    # Participation: the broker's gross activity as a share of the market's.
    market = price_index.loc[acc_start:acc_end, "volume"]
    market_lots = float(market.sum()) / LOT_SIZE if len(market) else 0.0
    broker_lots = float(acc["buy_lot"].sum() + acc["sell_lot"].sum()) / 2.0
    participation = broker_lots / market_lots if market_lots > 0 else 0.0

    return Campaign(
        ticker=ticker,
        broker=broker,
        acc_start=acc_start,
        acc_end=acc_end,
        dist_end=dist_end if complete else None,
        complete=complete,
        acc_days=int(end_idx - start_idx),
        dist_days=int(exit_idx - end_idx),
        holding_days=int(exit_idx - start_idx),
        lots_accumulated=float(lots),
        peak_inventory=float(group["inventory_lot"].iloc[end_idx]),
        value_accumulated=float(lots * LOT_SIZE * entry_vwap),
        entry_vwap=float(entry_vwap),
        exit_vwap=float(exit_vwap),
        price_at_start=price_at_start,
        price_at_acc_end=price_at_acc_end,
        price_peak=price_peak,
        price_at_exit=price_at_exit,
        entry_percentile=entry_percentile,
        acc_price_change=float(acc_price_change),
        stealth_ratio=float(np.clip(stealth_ratio, -3.0, 3.0)),
        markup_pct=float(markup_pct),
        realized_return=float(realized_return) if np.isfinite(realized_return) else np.nan,
        exit_capture=float(np.clip(exit_capture, -3.0, 3.0)) if np.isfinite(exit_capture) else np.nan,
        exit_vs_peak_days=int((dist_end - peak_date).days),
        max_adverse_pct=max_adverse,
        participation_pct=float(participation),
    ).to_dict()


def active_campaign(campaigns: pd.DataFrame, ticker: str, broker: str) -> Optional[pd.Series]:
    """The in-progress campaign for a (ticker, broker), if one is running."""
    if campaigns is None or campaigns.empty:
        return None
    mask = (
        (campaigns["ticker"] == str(ticker).upper())
        & (campaigns["broker"] == str(broker).upper())
        & (~campaigns["complete"])
    )
    rows = campaigns[mask]
    if rows.empty:
        return None
    return rows.sort_values("acc_start").iloc[-1]
