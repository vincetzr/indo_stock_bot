"""Orchestration: ties data providers and analytics into one reusable object.

Every CLI command runs through :class:`Engine`, so the screener, the plan
generator, the backtester and the dashboard all see identical data and identical
scores. Per-ticker analysis is memoised for the lifetime of the object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .analytics import accumulation, broker_flow, campaigns as campaign_mod, indicators, playbook
from .analytics.accumulation import AccumulationSignal
from .config import Config, load_config
from .data import YahooOHLCV, build_provider
from .data.broker_summary import ChainedBrokerSummary


@dataclass
class TickerAnalysis:
    """Everything the engine knows about one ticker."""

    ticker: str
    bars: pd.DataFrame                                  # OHLCV + indicators
    signal: AccumulationSignal
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)      # broker summary
    flow: pd.DataFrame = field(default_factory=pd.DataFrame)         # daily aggregates
    ledger: pd.DataFrame = field(default_factory=pd.DataFrame)       # per-broker positions
    campaigns: pd.DataFrame = field(default_factory=pd.DataFrame)
    playbook: pd.DataFrame = field(default_factory=pd.DataFrame)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def has_broker_data(self) -> bool:
        return not self.summary.empty

    @property
    def data_is_real(self) -> bool:
        return self.signal.data_is_real

    @property
    def last_close(self) -> float:
        return float(self.bars["close"].iloc[-1]) if len(self.bars) else float("nan")

    @property
    def last_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.bars["date"].iloc[-1]) if len(self.bars) else pd.NaT

    def top_brokers(self, n: int = 8, side: str = "net") -> pd.DataFrame:
        """Largest current inventory holders reconstructed from flow."""
        if self.positions.empty:
            return pd.DataFrame()
        df = self.positions.copy()
        if side == "net":
            df = df.reindex(df["inventory_lot"].abs().sort_values(ascending=False).index)
        return df.head(n)

    def largest_holder(self) -> Optional[str]:
        """Broker with the largest positive inventory build, ignoring tier.

        For the tier-aware version used by the trading plan, see
        :meth:`Engine.lead_institutional_broker`, which needs the registry.
        """
        if self.positions.empty:
            return None
        longs = self.positions[self.positions["inventory_lot"] > 0]
        if longs.empty:
            return None
        return str(longs.sort_values("inventory_value", ascending=False)["broker"].iloc[0])


class Engine:
    def __init__(
        self,
        cfg: Optional[Config] = None,
        broker_provider: Optional[ChainedBrokerSummary] = None,
        provider_names: Optional[List[str]] = None,
        verbose: bool = True,
        profile: Optional[str] = None,
    ):
        self.cfg = cfg or load_config()
        self.profile = profile
        self.ohlcv = YahooOHLCV(self.cfg)
        self.verbose = verbose
        self._provider_names = provider_names
        self._broker_provider = broker_provider
        self._benchmark: Optional[pd.Series] = None
        self._analysis_cache: Dict[str, TickerAnalysis] = {}
        self._price_cache: Dict[str, pd.DataFrame] = {}

    # -- providers ----------------------------------------------------------
    @property
    def broker_provider(self) -> ChainedBrokerSummary:
        if self._broker_provider is None:
            self._broker_provider = build_provider(
                self.cfg, names=self._provider_names, ohlcv=self._price_cache
            )
        return self._broker_provider

    def data_provenance(self) -> str:
        return self.broker_provider.describe()

    # -- prices -------------------------------------------------------------
    def prices(self, ticker: str, force_refresh: bool = False) -> pd.DataFrame:
        key = ticker.upper()
        if key not in self._price_cache or force_refresh:
            self._price_cache[key] = self.ohlcv.get(key, force_refresh=force_refresh)
        return self._price_cache[key]

    def benchmark(self) -> pd.Series:
        """IHSG closes, used for relative strength."""
        if self._benchmark is None:
            symbol = self.cfg.indices.get("composite", "^JKSE")
            df = self.ohlcv.get(symbol)
            self._benchmark = (
                df.set_index("date")["close"] if not df.empty else pd.Series(dtype=float)
            )
        return self._benchmark

    def _aligned_benchmark(self, bars: pd.DataFrame) -> Optional[pd.Series]:
        bench = self.benchmark()
        if bench is None or bench.empty or bars.empty:
            return None
        return bench.reindex(bars["date"]).ffill().reset_index(drop=True)

    # -- analysis -----------------------------------------------------------
    def analyze(self, ticker: str, with_campaigns: bool = True,
                force_refresh: bool = False) -> Optional[TickerAnalysis]:
        key = ticker.upper()
        if key in self._analysis_cache and not force_refresh:
            return self._analysis_cache[key]

        bars = self.prices(key, force_refresh=force_refresh)
        if bars is None or len(bars) < 120:
            if self.verbose:
                print(f"  ! {key}: only {0 if bars is None else len(bars)} bars, skipping")
            return None

        enriched = indicators.enrich(bars, cfg=self.cfg,
                                     benchmark=self._aligned_benchmark(bars))

        summary = self.broker_provider.fetch(key)
        flow = pd.DataFrame()
        ledger = pd.DataFrame()
        campaigns = pd.DataFrame()
        book = pd.DataFrame()
        positions = pd.DataFrame()
        source = "none"
        is_real = False

        if summary is not None and not summary.empty:
            source = str(summary["source"].iloc[0])
            is_real = not source.startswith("synthetic")
            flow = broker_flow.daily_aggregates(summary, self.cfg.brokers)
            ledger = broker_flow.build_ledger(summary, bars, ticker=key)
            positions = broker_flow.broker_positions(ledger)
            if with_campaigns and not ledger.empty:
                campaigns = campaign_mod.extract_campaigns(ledger, bars, self.cfg)
                if not campaigns.empty:
                    book = playbook.build_playbook(campaigns, self.cfg.brokers)

        signal = accumulation.score(
            enriched, self.cfg, flow=flow if not flow.empty else None,
            ticker=key, data_source=source, data_is_real=is_real,
            profile=self.profile,
        )

        analysis = TickerAnalysis(
            ticker=key, bars=enriched, signal=signal, summary=summary,
            flow=flow, ledger=ledger, campaigns=campaigns, playbook=book,
            positions=positions,
        )
        self._analysis_cache[key] = analysis
        return analysis

    def lead_institutional_broker(self, analysis: TickerAnalysis) -> Optional[str]:
        """Institutional desk with the largest inventory build in this name."""
        if analysis.positions.empty:
            return None
        df = analysis.positions.copy()
        df["tier"] = df["broker"].map(lambda c: self.cfg.brokers.get(c).tier)
        institutional = df[
            df["tier"].isin(["bulge", "foreign", "local_inst"]) & (df["inventory_lot"] > 0)
        ]
        if institutional.empty:
            return None
        # Prefer bulge desks when one holds a meaningful position.
        bulge = institutional[institutional["tier"] == "bulge"]
        pool = bulge if not bulge.empty else institutional
        return str(pool.sort_values("inventory_value", ascending=False)["broker"].iloc[0])

    # -- screening ----------------------------------------------------------
    def screen(self, tickers: List[str], with_campaigns: bool = False,
               force_refresh: bool = False) -> pd.DataFrame:
        rows = []
        total = len(tickers)
        for i, ticker in enumerate(tickers, 1):
            if self.verbose:
                print(f"  [{i:>3}/{total}] {ticker:<6}", end=" ", flush=True)
            try:
                analysis = self.analyze(ticker, with_campaigns=with_campaigns,
                                        force_refresh=force_refresh)
            except Exception as exc:                       # keep the sweep alive
                if self.verbose:
                    print(f"error: {exc}")
                continue
            if analysis is None:
                continue

            row = analysis.signal.to_row()
            row["lead_broker"] = self.lead_institutional_broker(analysis) or ""
            rows.append(row)
            if self.verbose:
                print(f"score {row['score']:>5.1f}  {row['level']:<7} {row.get('wyckoff_phase','')}")

        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        return out

    def portfolio_playbook(self, tickers: List[str],
                           min_campaigns: int = 3) -> tuple:
        """Aggregate campaigns across many tickers into one broker playbook.

        Per-ticker samples are far too small to characterise a desk. Pooling
        campaigns across the universe is what makes the profile meaningful.
        """
        all_campaigns = []
        for ticker in tickers:
            analysis = self.analyze(ticker, with_campaigns=True)
            if analysis is not None and not analysis.campaigns.empty:
                all_campaigns.append(analysis.campaigns)
        if not all_campaigns:
            return pd.DataFrame(), pd.DataFrame()

        pooled = pd.concat(all_campaigns, ignore_index=True)
        book = playbook.build_playbook(pooled, self.cfg.brokers, min_campaigns=min_campaigns)
        return pooled, book

    def pooled_forward_edge(self, tickers: List[str]) -> pd.DataFrame:
        """Broker forward-return edge measured across the whole universe."""
        summaries, prices = [], []
        for ticker in tickers:
            analysis = self.analyze(ticker, with_campaigns=False)
            if analysis is None or analysis.summary.empty:
                continue
            summaries.append(analysis.summary)
            prices.append(analysis.bars[["date", "close"]].assign(ticker=ticker.upper()))
        if not summaries:
            return pd.DataFrame()
        return playbook.broker_forward_edge(
            pd.concat(summaries, ignore_index=True),
            pd.concat(prices, ignore_index=True),
            self.cfg.brokers,
            horizons=tuple(self.cfg.get("backtest.horizons", [5, 10, 20, 60])),
        )
