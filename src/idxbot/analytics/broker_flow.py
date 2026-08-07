"""Per-broker position ledgers reconstructed from broker summary.

This is where "what is J.P. Morgan actually doing in this name" becomes a
number. From the daily buy/sell tape per exchange member we rebuild, for each
(ticker, broker) pair:

  * **inventory** - cumulative net position, in lots
  * **average cost** - the broker's own volume-weighted cost basis
  * **realised P/L** - locked in when they sell
  * **unrealised P/L** - marked against the current close

IMPORTANT CAVEAT ON INVENTORY
-----------------------------
Broker summary shows *flow*, never *holdings*. A ledger built from it measures
position change since the first day of data, not absolute ownership. If UBS
already held 200m shares before your window opens, this ledger starts them at
zero and can show a negative inventory while they are still net long overall.

Two consequences worth internalising:
  * Treat inventory as "position relative to window start". The *changes* and
    the *cost basis of those changes* are the signal; the absolute level is not.
  * The longer the history you feed in, the less this distortion matters. Start
    the ledger at a genuine quiet base, not mid-campaign.

A second caveat: a broker code aggregates every client trading through that
member, plus the firm's own book. "J.P. Morgan bought" means "flow crossed
J.P. Morgan's membership", which may be a foreign pension fund, a hedge fund,
an index tracker rebalancing, or a hedge against a derivative. It is a strong
signal, not a confession of intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import BrokerRegistry, Config
from ..data.broker_summary import LOT_SIZE

LEDGER_COLUMNS = [
    "date", "ticker", "broker", "buy_lot", "sell_lot", "net_lot",
    "buy_val", "sell_val",
    "inventory_lot", "avg_cost", "realized_pnl", "unrealized_pnl", "total_pnl",
    "inventory_value", "close",
]


@dataclass
class _Position:
    """Signed moving-average-cost position tracker."""

    shares: float = 0.0
    avg_cost: float = 0.0
    realized: float = 0.0

    def trade(self, qty: float, price: float) -> None:
        """Apply a signed trade: ``qty`` > 0 buys, < 0 sells."""
        if qty == 0 or price <= 0:
            return

        same_direction = self.shares == 0 or (self.shares > 0) == (qty > 0)
        if same_direction:
            new_shares = self.shares + qty
            if new_shares == 0:
                self.shares, self.avg_cost = 0.0, 0.0
                return
            self.avg_cost = (
                self.avg_cost * abs(self.shares) + price * abs(qty)
            ) / abs(new_shares)
            self.shares = new_shares
            return

        # Opposing trade: realise P/L on the overlap first.
        closing = min(abs(qty), abs(self.shares))
        direction = 1.0 if self.shares > 0 else -1.0
        self.realized += (price - self.avg_cost) * closing * direction

        new_shares = self.shares + qty
        if new_shares == 0:
            self.shares, self.avg_cost = 0.0, 0.0
        elif (new_shares > 0) != (self.shares > 0):
            # Flipped through zero: the residual is a fresh position at price.
            self.shares, self.avg_cost = new_shares, price
        else:
            self.shares = new_shares


def build_ledger(
    summary: pd.DataFrame,
    prices: pd.DataFrame,
    ticker: Optional[str] = None,
    brokers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Reconstruct daily position ledgers for every broker in ``summary``.

    ``prices`` must carry ``date`` and ``close``; it supplies the mark for
    unrealised P/L and fills days a broker did not trade so inventory is a
    continuous daily series.
    """
    if summary is None or summary.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    df = summary.copy()
    if ticker:
        df = df[df["ticker"] == str(ticker).upper()]
    if brokers:
        wanted = {b.upper() for b in brokers}
        df = df[df["broker"].isin(wanted)]
    if df.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    close_by_date = (
        prices.assign(date=pd.to_datetime(prices["date"]).dt.normalize())
        .set_index("date")["close"]
    )

    frames: List[pd.DataFrame] = []
    for (tkr, broker), group in df.groupby(["ticker", "broker"], sort=False):
        group = group.sort_values("date")
        # Reindex onto the trading calendar so inventory persists on quiet days.
        calendar = close_by_date.loc[
            (close_by_date.index >= group["date"].iloc[0])
            & (close_by_date.index <= group["date"].iloc[-1])
        ].index
        if len(calendar) == 0:
            calendar = pd.DatetimeIndex(group["date"].unique())

        g = group.set_index("date").reindex(calendar)
        buy_lot = g["buy_lot"].fillna(0.0).to_numpy(float)
        sell_lot = g["sell_lot"].fillna(0.0).to_numpy(float)
        buy_avg = g["buy_avg"].fillna(0.0).to_numpy(float)
        sell_avg = g["sell_avg"].fillna(0.0).to_numpy(float)
        buy_val = g["buy_val"].fillna(0.0).to_numpy(float)
        sell_val = g["sell_val"].fillna(0.0).to_numpy(float)
        closes = close_by_date.reindex(calendar).ffill().to_numpy(float)

        n = len(calendar)
        inventory = np.zeros(n)
        avg_cost = np.zeros(n)
        realized = np.zeros(n)
        unrealized = np.zeros(n)

        pos = _Position()
        for i in range(n):
            mark = closes[i] if np.isfinite(closes[i]) and closes[i] > 0 else pos.avg_cost
            # Fall back to the mark when a feed omits an average price.
            bp = buy_avg[i] if buy_avg[i] > 0 else mark
            sp = sell_avg[i] if sell_avg[i] > 0 else mark
            if buy_lot[i] > 0:
                pos.trade(buy_lot[i] * LOT_SIZE, bp)
            if sell_lot[i] > 0:
                pos.trade(-sell_lot[i] * LOT_SIZE, sp)

            inventory[i] = pos.shares / LOT_SIZE
            avg_cost[i] = pos.avg_cost
            realized[i] = pos.realized
            unrealized[i] = (mark - pos.avg_cost) * pos.shares if pos.shares else 0.0

        frames.append(pd.DataFrame({
            "date": calendar,
            "ticker": tkr,
            "broker": broker,
            "buy_lot": buy_lot,
            "sell_lot": sell_lot,
            "net_lot": buy_lot - sell_lot,
            "buy_val": buy_val,
            "sell_val": sell_val,
            "inventory_lot": inventory,
            "avg_cost": avg_cost,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "total_pnl": realized + unrealized,
            "inventory_value": inventory * LOT_SIZE * closes,
            "close": closes,
        }))

    if not frames:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["ticker", "broker", "date"]
    ).reset_index(drop=True)


def daily_aggregates(summary: pd.DataFrame, registry: BrokerRegistry) -> pd.DataFrame:
    """Collapse broker summary into one row per (ticker, day) of flow metrics.

    These are the cross-sectional features the accumulation score consumes:
    who is buying, how concentrated the buying is, and whether the
    institutional and retail sides are on opposite sides of the tape.
    """
    if summary is None or summary.empty:
        return pd.DataFrame()

    df = summary.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["net_val"] = df["buy_val"] - df["sell_val"]
    df["net_lot"] = df["buy_lot"] - df["sell_lot"]

    tiers = df["broker"].map(lambda c: registry.get(c).tier)
    df["is_bulge"] = tiers == "bulge"
    df["is_retail"] = tiers == "retail"
    df["is_foreign"] = df["broker"].map(lambda c: registry.get(c).foreign)
    df["is_institutional"] = tiers.isin(["bulge", "foreign", "local_inst"])

    rows: List[dict] = []
    for (tkr, date), g in df.groupby(["ticker", "date"], sort=True):
        buyers = g[g["net_val"] > 0]
        total_buy_val = float(buyers["net_val"].sum())

        # Herfindahl index over net buyers: 1.0 = a single broker took all the
        # supply, near 0 = the buying was spread across the market.
        if total_buy_val > 0:
            shares = buyers["net_val"] / total_buy_val
            hhi = float((shares ** 2).sum())
            top5 = float(shares.nlargest(5).sum())
            top_buyer = str(buyers.loc[buyers["net_val"].idxmax(), "broker"])
            top_buyer_share = float(shares.max())
        else:
            hhi, top5, top_buyer, top_buyer_share = 0.0, 0.0, "", 0.0

        gross_val = float(g["buy_val"].sum())
        rows.append({
            "date": date,
            "ticker": tkr,
            "gross_val": gross_val,
            "broker_count": int(g["broker"].nunique()),
            "bulge_net_val": float(g.loc[g["is_bulge"], "net_val"].sum()),
            "bulge_net_lot": float(g.loc[g["is_bulge"], "net_lot"].sum()),
            "foreign_net_val": float(g.loc[g["is_foreign"], "net_val"].sum()),
            "inst_net_val": float(g.loc[g["is_institutional"], "net_val"].sum()),
            "retail_net_val": float(g.loc[g["is_retail"], "net_val"].sum()),
            "buyer_hhi": hhi,
            "top5_buyer_share": top5,
            "top_buyer": top_buyer,
            "top_buyer_share": top_buyer_share,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    gross = out["gross_val"].replace(0, np.nan)
    out["bulge_net_pct"] = (out["bulge_net_val"] / gross).fillna(0.0)
    out["foreign_net_pct"] = (out["foreign_net_val"] / gross).fillna(0.0)
    out["retail_net_pct"] = (out["retail_net_val"] / gross).fillna(0.0)
    # Positive when institutions absorb while retail supplies - the classic
    # distribution-of-risk pattern at a base.
    out["smart_dumb_spread"] = out["inst_net_val"] - out["retail_net_val"]
    out["smart_dumb_pct"] = (out["smart_dumb_spread"] / gross).fillna(0.0)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def broker_positions(ledger: pd.DataFrame, as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Latest ledger state per broker: inventory, cost basis, open P/L."""
    if ledger is None or ledger.empty:
        return pd.DataFrame()
    df = ledger if as_of is None else ledger[ledger["date"] <= pd.Timestamp(as_of)]
    if df.empty:
        return pd.DataFrame()

    latest = df.sort_values("date").groupby(["ticker", "broker"], as_index=False).last()
    latest["pnl_pct"] = np.where(
        (latest["avg_cost"] > 0) & (latest["inventory_lot"] != 0),
        (latest["close"] - latest["avg_cost"]) / latest["avg_cost"],
        0.0,
    )
    return latest.sort_values("inventory_value", ascending=False).reset_index(drop=True)


def inventory_matrix(ledger: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Wide date x broker inventory frame, handy for charting and correlation."""
    if ledger is None or ledger.empty:
        return pd.DataFrame()
    df = ledger[ledger["ticker"] == str(ticker).upper()]
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(index="date", columns="broker", values="inventory_lot",
                          aggfunc="last").ffill().fillna(0.0)


def tier_flow(summary: pd.DataFrame, registry: BrokerRegistry,
              ticker: Optional[str] = None) -> pd.DataFrame:
    """Daily net value by broker tier - the 'who is on which side' view."""
    if summary is None or summary.empty:
        return pd.DataFrame()
    df = summary.copy()
    if ticker:
        df = df[df["ticker"] == str(ticker).upper()]
    if df.empty:
        return pd.DataFrame()
    df["tier"] = df["broker"].map(lambda c: registry.get(c).tier)
    df["net_val"] = df["buy_val"] - df["sell_val"]
    pivot = df.pivot_table(index="date", columns="tier", values="net_val", aggfunc="sum")
    return pivot.fillna(0.0)


def summarise_broker(ledger: pd.DataFrame, broker: str,
                     cfg: Optional[Config] = None) -> Dict[str, float]:
    """Headline stats for one broker across every ticker in the ledger."""
    df = ledger[ledger["broker"] == str(broker).upper()]
    if df.empty:
        return {}
    latest = df.sort_values("date").groupby("ticker").last()
    return {
        "broker": broker.upper(),
        "tickers": int(len(latest)),
        "net_long_tickers": int((latest["inventory_lot"] > 0).sum()),
        "total_inventory_value": float(latest["inventory_value"].sum()),
        "total_realized_pnl": float(latest["realized_pnl"].sum()),
        "total_unrealized_pnl": float(latest["unrealized_pnl"].sum()),
        "total_pnl": float(latest["total_pnl"].sum()),
    }
