"""Simulated broker summary, for development and pipeline testing.

READ THIS BEFORE DRAWING ANY CONCLUSION FROM SYNTHETIC OUTPUT
-------------------------------------------------------------
This module invents broker flow. It does not observe it. The generative model
below *assumes* that institutional desks accumulate into weakness and distribute
into strength, and that retail flow chases momentum. Any analysis run on
synthetic data will therefore "discover" exactly that, because it was put there
by hand. That is circular and proves nothing about the real market.

What synthetic data is legitimately good for:
  * exercising the full pipeline end to end without a paid data feed
  * unit-testing the ledger, campaign segmentation and scoring maths
  * seeing the shape of the reports before you commit to a data subscription

What it must never be used for: deciding a real trade, or claiming anything
about how J.P. Morgan or UBS actually behave. Every frame produced here is
stamped ``source="synthetic"``, and every report checks that stamp and labels
itself accordingly.

The one thing that *is* real here is the price and volume series it is
conditioned on: the daily bars come from the exchange, so the aggregate volume
each simulated day distributes is genuine.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from .broker_summary import LOT_SIZE, SCHEMA, BrokerSummaryProvider, empty_frame

# Persona parameters by broker tier.
#   contrarian   : +1 buys weakness / sells strength, -1 chases momentum
#   patience     : smoothing of the desire signal; higher = slower, multi-week
#   intensity    : how much of gross volume is directional rather than churn
#   share        : baseline share of daily volume
#   price_skew   : where in the day's range they transact (0 = at the low)
PERSONAS: Dict[str, Dict[str, float]] = {
    "bulge":      {"contrarian":  0.85, "patience": 25, "intensity": 0.55, "share": 0.055, "price_skew": 0.42},
    "foreign":    {"contrarian":  0.50, "patience": 15, "intensity": 0.40, "share": 0.035, "price_skew": 0.47},
    "local_inst": {"contrarian":  0.20, "patience": 10, "intensity": 0.30, "share": 0.030, "price_skew": 0.50},
    "retail":     {"contrarian": -0.70, "patience":  4, "intensity": 0.45, "share": 0.060, "price_skew": 0.58},
    "unknown":    {"contrarian":  0.00, "patience":  8, "intensity": 0.20, "share": 0.020, "price_skew": 0.50},
}


def _stable_seed(*parts: str) -> int:
    """Deterministic seed so a ticker's simulated history never changes."""
    text = "|".join(parts)
    h = 2166136261
    for ch in text:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h


class SyntheticBrokerSummary(BrokerSummaryProvider):
    name = "synthetic"
    is_real = False

    def __init__(self, cfg: Config, ohlcv: Optional[Dict[str, pd.DataFrame]] = None,
                 max_days: int = 1500):
        self.cfg = cfg
        self.registry = cfg.brokers
        self._ohlcv = ohlcv or {}
        self.max_days = max_days
        self._loader = None

    def set_ohlcv(self, ohlcv: Dict[str, pd.DataFrame]) -> None:
        self._ohlcv.update(ohlcv)

    def available(self) -> bool:
        return True

    def _bars(self, ticker: str) -> pd.DataFrame:
        ticker = ticker.upper()
        if ticker in self._ohlcv:
            return self._ohlcv[ticker]
        if self._loader is None:
            from .ohlcv import YahooOHLCV
            self._loader = YahooOHLCV(self.cfg)
        df = self._loader.get(ticker)
        self._ohlcv[ticker] = df
        return df

    def fetch(self, ticker: str, start=None, end=None) -> pd.DataFrame:
        bars = self._bars(ticker)
        if bars is None or bars.empty:
            return empty_frame()

        bars = bars.copy()
        if start is not None:
            bars = bars[bars["date"] >= pd.Timestamp(start)]
        if end is not None:
            bars = bars[bars["date"] <= pd.Timestamp(end)]
        # Simulating 25 years x 40 brokers is pointless cost; the campaign
        # analytics only need a few years of depth.
        if len(bars) > self.max_days:
            bars = bars.iloc[-self.max_days:]
        if len(bars) < 30:
            return empty_frame()

        return self._simulate(ticker.upper(), bars.reset_index(drop=True))

    # -- generative model ---------------------------------------------------
    def _simulate(self, ticker: str, bars: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(_stable_seed(ticker, "v1"))

        close = bars["close"].to_numpy(float)
        high = bars["high"].to_numpy(float)
        low = bars["low"].to_numpy(float)
        volume_lots = np.maximum(bars["volume"].to_numpy(float) / LOT_SIZE, 1.0)
        n = len(bars)

        # Latent state the personas react to.
        window = 120
        pct_rank = np.empty(n)
        for i in range(n):
            lo = max(0, i - window + 1)
            hist = close[lo:i + 1]
            span = hist.max() - hist.min()
            pct_rank[i] = 0.5 if span <= 0 else (close[i] - hist.min()) / span

        momentum = pd.Series(close).pct_change(20).fillna(0.0).to_numpy()
        momentum = np.clip(momentum, -0.5, 0.5)

        codes = self._broker_codes()
        rows: List[pd.DataFrame] = []
        buy_matrix = np.zeros((n, len(codes)))
        sell_matrix = np.zeros((n, len(codes)))
        buy_price = np.zeros((n, len(codes)))
        sell_price = np.zeros((n, len(codes)))

        for j, code in enumerate(codes):
            persona = PERSONAS.get(self.registry.get(code).tier, PERSONAS["unknown"])
            broker_rng = np.random.default_rng(_stable_seed(ticker, code, "v1"))

            # Directional desire: contrarian on range position, momentum on 20d.
            raw_desire = (
                persona["contrarian"] * (0.5 - pct_rank) * 2.0
                + (1.0 - abs(persona["contrarian"])) * momentum * 2.0
            )
            if persona["contrarian"] < 0:
                raw_desire = persona["contrarian"] * (0.5 - pct_rank) * 2.0 + momentum * 3.0

            # A slow episodic regime: desks are not always engaged in a name.
            engagement = self._episodic_engagement(n, broker_rng, persona["patience"])
            desire = _ewma(raw_desire, span=persona["patience"]) * engagement
            desire = np.tanh(desire * 1.5)

            # Gross participation, log-normal around the persona's baseline share.
            noise = broker_rng.lognormal(mean=0.0, sigma=0.55, size=n)
            share = persona["share"] * noise * engagement.clip(0.15, 1.0)
            gross = volume_lots * share

            net = gross * desire * persona["intensity"]
            buy_matrix[:, j] = np.maximum((gross + net) / 2.0, 0.0)
            sell_matrix[:, j] = np.maximum((gross - net) / 2.0, 0.0)

            # Where in the day's range they transacted. This is an ASSUMPTION,
            # not an observation - see the module docstring.
            skew = persona["price_skew"]
            bp = np.clip(skew + broker_rng.normal(0, 0.12, n) - 0.06 * desire, 0.02, 0.98)
            sp = np.clip(skew + broker_rng.normal(0, 0.12, n) + 0.06 * desire, 0.02, 0.98)
            buy_price[:, j] = low + bp * (high - low)
            sell_price[:, j] = low + sp * (high - low)

        # Reconcile to the real traded volume: every lot bought was sold.
        buy_totals = buy_matrix.sum(axis=1, keepdims=True)
        sell_totals = sell_matrix.sum(axis=1, keepdims=True)
        buy_matrix *= np.divide(volume_lots[:, None], buy_totals,
                                out=np.zeros_like(buy_matrix), where=buy_totals > 0)
        sell_matrix *= np.divide(volume_lots[:, None], sell_totals,
                                 out=np.zeros_like(sell_matrix), where=sell_totals > 0)

        dates = bars["date"].to_numpy()
        for j, code in enumerate(codes):
            active = (buy_matrix[:, j] + sell_matrix[:, j]) > 0.5
            if not active.any():
                continue
            frame = pd.DataFrame({
                "date": dates[active],
                "ticker": ticker,
                "broker": code,
                "buy_lot": np.round(buy_matrix[active, j], 1),
                "sell_lot": np.round(sell_matrix[active, j], 1),
                "buy_avg": np.round(buy_price[active, j], 0),
                "sell_avg": np.round(sell_price[active, j], 0),
                "source": "synthetic",
            })
            frame["buy_val"] = frame["buy_lot"] * LOT_SIZE * frame["buy_avg"]
            frame["sell_val"] = frame["sell_lot"] * LOT_SIZE * frame["sell_avg"]
            rows.append(frame)

        if not rows:
            return empty_frame()
        out = pd.concat(rows, ignore_index=True)
        return out[SCHEMA].sort_values(["date", "broker"]).reset_index(drop=True)

    def _broker_codes(self) -> List[str]:
        codes = self.registry.codes()
        return codes if codes else ["BK", "AK", "KZ", "YP", "PD", "CC"]

    @staticmethod
    def _episodic_engagement(n: int, rng: np.random.Generator, patience: float) -> np.ndarray:
        """A slow on/off process: desks work a name in campaigns, then move on."""
        engagement = np.zeros(n)
        i = 0
        while i < n:
            active = rng.random() < 0.45
            length = int(max(5, rng.gamma(shape=2.0, scale=patience * 1.5)))
            level = rng.uniform(0.6, 1.4) if active else rng.uniform(0.05, 0.25)
            engagement[i:i + length] = level
            i += length
        return _ewma(engagement, span=5)


def _ewma(values: np.ndarray, span: float) -> np.ndarray:
    if span <= 1:
        return np.asarray(values, dtype=float)
    return pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy()
