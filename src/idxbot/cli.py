"""idxbot command line interface.

    idxbot screen      --universe lq45          rank a universe by accumulation
    idxbot analyze     BBCA                     deep dive on one ticker
    idxbot plan        BBCA                     executable trading plan
    idxbot playbook    --universe lq45          per-desk behavioural profile
    idxbot reverse     --universe lq45          full institutional plan: who leads,
                                                do they coordinate, when to join
    idxbot backtest    --universe lq45          does the score predict returns?
    idxbot evaluate    --split --components     cross-sectional IC, train/holdout
    idxbot portfolio   --split                  long-only top-N equity curve
    idxbot daytrade    --universe all           intraday momentum scan + plans
    idxbot invest      --universe lq45          long-horizon basket (60d momentum)
    idxbot paste       BBCA                    ingest broker summary you copied
    idxbot verify      --tickers BBCA           is that data good enough to trust?
    idxbot dashboard   --universe lq45          offline HTML report
    idxbot live        --file ticks.jsonl       live broker summary from running trade
    idxbot pine        [BBCA]                   Pine Script / plan inputs
    idxbot watchlist   --universe lq45          TradingView-importable watchlist
    idxbot brokers                              exchange member registry
    idxbot data        --universe deep_history  what history is actually available
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

from . import backtest as backtest_mod
from . import plan as plan_mod
from .analytics import playbook as playbook_mod
from .config import load_config
from .engine import Engine
from .market import format_idr
from .report import dashboard
from .tradingview import links as tv


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _resolve_tickers(cfg, args) -> List[str]:
    if getattr(args, "tickers", None):
        return [t.strip().upper() for t in args.tickers if t.strip()]
    universe = getattr(args, "universe", None) or "lq45"
    tickers = cfg.universe(universe)
    limit = getattr(args, "limit", None)
    return tickers[:limit] if limit else tickers


def _provenance_notice(engine: Engine, results: Optional[pd.DataFrame] = None) -> None:
    """Report where the broker data came from.

    Three distinct states, and conflating them is misleading:
      * real broker summary      -> no warning
      * simulated broker flow    -> loud warning, the numbers are fabricated
      * no broker data at all    -> price-only, an honest absence, not a lie
    """
    provenance = engine.data_provenance()
    print()
    print("  data provenance : " + provenance)

    price_only = False
    if results is not None and not results.empty and "data_mode" in results:
        price_only = bool((results["data_mode"] == "price-only").all())

    simulated = "synthetic" in provenance.lower() and not price_only
    if results is not None and not results.empty and "data_source" in results:
        simulated = bool(results["data_source"].astype(str).str.startswith("synthetic").any())

    if price_only and not simulated:
        print("  note: price-only mode - no broker data, so no institutional")
        print("        confirmation. Nothing here is simulated.")
        print()
        return

    if simulated:
        print("  " + "!" * 68)
        print("  ! BROKER FLOW IS SIMULATED. Prices and volumes are real exchange data,")
        print("  ! but every broker-level number below came from data/synthetic.py, which")
        print("  ! ASSUMES institutions buy weakness and retail chases momentum. Conclusions")
        print("  ! about broker behaviour drawn from it are circular. See docs/LIVE_DATA.md")
        print("  ! for how to connect a real broker-summary source.")
        print("  " + "!" * 68)
    print()


def _write_csv(df: pd.DataFrame, path: Optional[str], label: str) -> None:
    if not path or df is None or df.empty:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {label} -> {path}")


def _engine(args) -> Engine:
    cfg = load_config()
    providers = None
    if getattr(args, "providers", None):
        providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    as_of = getattr(args, "as_of", None)
    return Engine(cfg, provider_names=providers,
                  verbose=not getattr(args, "quiet", False),
                  profile=getattr(args, "profile", None),
                  as_of=pd.Timestamp(as_of) if as_of else None)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_screen(args) -> int:
    engine = _engine(args)
    tickers = _resolve_tickers(engine.cfg, args)
    print(f"Screening {len(tickers)} tickers for accumulation...\n")

    results = engine.screen(tickers, with_campaigns=args.campaigns)
    if results.empty:
        print("\nNo results produced.")
        return 1

    _provenance_notice(engine, results)

    shown = results[results["score"] >= args.min_score] if args.min_score else results
    if args.top:
        shown = shown.head(args.top)

    columns = ["rank", "ticker", "score", "level", "wyckoff_phase", "close",
               "lead_broker", "data_mode"]
    columns = [c for c in columns if c in shown.columns]
    print(shown[columns].to_string(index=False))

    print()
    for _, r in shown.head(args.explain).iterrows():
        if r.get("flags"):
            print(f"  {r['ticker']}  ({r['score']:.1f})")
            for flag in str(r["flags"]).split(" | "):
                if flag:
                    print(f"     - {flag}")
    _write_csv(results, args.out, "screener")
    return 0


def cmd_analyze(args) -> int:
    engine = _engine(args)
    ticker = args.ticker.upper()
    analysis = engine.analyze(ticker, with_campaigns=True)
    if analysis is None:
        print(f"No usable data for {ticker}.")
        return 1

    signal = analysis.signal
    bars = analysis.bars
    print("=" * 78)
    print(f" {ticker}   score {signal.score:.1f} ({signal.level})   "
          f"Wyckoff {signal.wyckoff_state.phase if signal.wyckoff_state else '-'}")
    print("=" * 78)
    print(f" history      : {bars['date'].iloc[0]:%Y-%m-%d} -> {bars['date'].iloc[-1]:%Y-%m-%d} "
          f"({len(bars):,} sessions)")
    print(f" last close   : {analysis.last_close:,.0f}")
    print(f" scoring mode : {signal.data_mode}")
    _provenance_notice(engine, pd.DataFrame([{"data_is_real": signal.data_is_real}]))

    print(" SCORE COMPONENTS")
    for name, value in sorted(signal.components.items(), key=lambda kv: -kv[1]):
        weight = signal.weights_used.get(name, 0.0)
        bar = "#" * int(round(value * 28))
        print(f"   {name:<24} {value:>5.2f}  w={weight:>4.2f}  {bar}")

    if signal.flags:
        print("\n EVIDENCE")
        for flag in signal.flags:
            print(f"   - {flag}")

    if signal.wyckoff_state:
        state = signal.wyckoff_state
        print(f"\n WYCKOFF: phase {state.phase} (confidence {state.confidence:.2f})")
        print(f"   {state.meaning}")
        if state.events:
            print(f"   events: {', '.join(state.events)}")
        if np.isfinite(state.support):
            print(f"   range: {state.support:,.0f} - {state.resistance:,.0f} "
                  f"over {state.range_days} sessions")

    if not analysis.positions.empty:
        print("\n RECONSTRUCTED BROKER POSITIONS (relative to the start of the data window)")
        top = analysis.positions.copy()
        top["tier"] = top["broker"].map(lambda c: engine.cfg.brokers.get(c).tier)
        top = top.reindex(top["inventory_lot"].abs().sort_values(ascending=False).index).head(12)
        print(f"   {'broker':<7}{'tier':<12}{'inventory (lot)':>18}{'avg cost':>12}"
              f"{'open P/L':>10}{'value':>14}")
        for _, r in top.iterrows():
            pnl = (r["close"] / r["avg_cost"] - 1.0) if r["avg_cost"] > 0 else 0.0
            print(f"   {r['broker']:<7}{r['tier']:<12}{r['inventory_lot']:>18,.0f}"
                  f"{r['avg_cost']:>12,.0f}{pnl:>9.1%}{format_idr(r['inventory_value']):>14}")

    if not analysis.campaigns.empty:
        print(f"\n CAMPAIGNS DETECTED: {len(analysis.campaigns)}")
        cols = ["broker", "acc_start", "acc_days", "entry_vwap", "entry_percentile",
                "markup_pct", "realized_return", "exit_capture", "complete"]
        recent = analysis.campaigns.sort_values("acc_start").tail(args.campaign_limit)
        print(recent[cols].to_string(index=False))

    if not analysis.playbook.empty:
        print("\n PER-BROKER PLAYBOOK (this ticker only - pool across a universe for"
              " meaningful samples)")
        for _, r in analysis.playbook.iterrows():
            print("   " + playbook_mod.describe_playbook_row(r))

    _write_csv(analysis.campaigns, args.out, "campaigns")
    return 0


def cmd_plan(args) -> int:
    engine = _engine(args)
    tickers = _resolve_tickers(engine.cfg, args)

    pooled = None
    if args.pool:
        print(f"Pooling campaign history across {args.pool} for the broker playbook...")
        _, pooled = engine.portfolio_playbook(engine.cfg.universe(args.pool))
        print(f"  profiled {len(pooled)} brokers\n")

    plans = []
    for ticker in tickers:
        analysis = engine.analyze(ticker, with_campaigns=True)
        if analysis is None:
            continue
        plan = plan_mod.build_plan(analysis, engine, equity=args.equity,
                                   risk_pct=args.risk, pooled_playbook=pooled)
        plans.append(plan)
        if args.all or plan.verdict != "SKIP":
            print(plan.render())
            print()

    if not plans:
        print("No plans produced.")
        return 1

    equity = args.equity or float(engine.cfg.get("plan.account_equity_idr", 1e8))
    heat = plan_mod.portfolio_heat(plans, equity)
    print("PORTFOLIO HEAT IF EVERY 'TAKE' WERE FILLED")
    print(f"  positions        : {int(heat['positions'])}")
    print(f"  total risk       : {format_idr(heat['total_risk_idr'])} "
          f"({heat['total_risk_pct']:.2%} of equity)")
    print(f"  gross exposure   : {format_idr(heat['total_notional_idr'])} "
          f"({heat['gross_exposure_pct']:.1%} of equity)")

    if args.out:
        _write_csv(pd.DataFrame([p.to_dict() for p in plans]), args.out, "plans")
    return 0


def cmd_playbook(args) -> int:
    engine = _engine(args)
    tickers = _resolve_tickers(engine.cfg, args)
    print(f"Reverse-engineering broker behaviour across {len(tickers)} tickers...")
    print("(pooling campaigns across many names - per-ticker samples are too small)\n")

    campaigns, book = engine.portfolio_playbook(tickers, min_campaigns=args.min_campaigns)
    if campaigns.empty:
        print("No campaigns detected. Check that broker summary data is available.")
        return 1

    _provenance_notice(engine)
    print(f"  campaigns analysed : {len(campaigns):,}")
    print(f"  period             : {campaigns['acc_start'].min():%Y-%m-%d} -> "
          f"{campaigns['acc_start'].max():%Y-%m-%d}")
    print(f"  brokers profiled   : {len(book)}\n")

    if book.empty:
        print(f"No broker reached the {args.min_campaigns}-campaign minimum.")
        return 1

    focus = book
    if args.tier:
        focus = book[book["tier"] == args.tier]

    print("=" * 78)
    print(" BROKER PLAYBOOKS")
    print("=" * 78)
    for _, r in focus.iterrows():
        print()
        print(playbook_mod.describe_playbook_row(r))

    print("\n" + "=" * 78)
    print(" SUMMARY TABLE")
    print("=" * 78)
    cols = ["broker", "tier", "campaigns", "median_acc_days", "median_entry_percentile",
            "median_stealth_ratio", "median_markup_pct", "median_realized_return",
            "median_exit_capture", "win_rate", "median_holding_days"]
    cols = [c for c in cols if c in focus.columns]
    print(focus[cols].to_string(index=False))

    if args.edge:
        print("\n" + "=" * 78)
        print(" FORWARD-RETURN EDGE (does heavy buying by this broker predict returns?)")
        print("=" * 78)
        edge = engine.pooled_forward_edge(tickers)
        if edge.empty:
            print(" Not enough events.")
        else:
            edge_cols = ["broker", "tier", "events"] + [
                c for c in edge.columns if c.startswith(("edge_", "t_", "hit_"))
            ]
            print(edge[edge_cols].to_string(index=False))
            print("\n Overlapping forward windows inflate these t-statistics.")
            print(" Treat them as a ranking, not as evidence of significance.")

    _write_csv(campaigns, args.out, "campaigns")
    _write_csv(book, args.playbook_out, "playbook")
    return 0


def cmd_verify(args) -> int:
    """Acceptance test: can this broker data actually support the analysis?"""
    from . import verify as vf

    engine = _engine(args)
    tickers = _resolve_tickers(engine.cfg, args)

    frames, prices = [], {}
    for ticker in tickers:
        summary = engine.broker_provider.fetch(ticker)
        if summary is not None and not summary.empty:
            frames.append(summary)
            bars = engine.prices(ticker)
            if bars is not None and not bars.empty:
                prices[ticker.upper()] = bars

    if not frames:
        print("No broker summary found for the requested tickers.")
        print(f"  provider chain: {engine.data_provenance()}")
        print("  Drop files in data/broker_summary/, or run: idxbot paste <TICKER>")
        return 1

    summary = pd.concat(frames, ignore_index=True)
    report = vf.verify(summary, engine.cfg.brokers, prices=prices)
    print(vf.render(report))
    return 0 if report.usable else 1


def cmd_plan_reverse(args) -> int:
    """Reverse-engineer the institutional operating plan from broker flow."""
    from .analytics import coordination as coord

    engine = _engine(args)
    tickers = _resolve_tickers(engine.cfg, args)
    print(f"Reverse-engineering institutional behaviour across {len(tickers)} names...")
    print("(pooling campaigns - per-ticker samples are far too small)\n")

    summaries, prices, campaign_frames = [], {}, []
    for ticker in tickers:
        analysis = engine.analyze(ticker, with_campaigns=True)
        if analysis is None:
            continue
        prices[ticker.upper()] = analysis.bars
        if not analysis.summary.empty:
            summaries.append(analysis.summary)
        if not analysis.campaigns.empty:
            campaign_frames.append(analysis.campaigns)

    if not summaries:
        print("No broker summary available - this command needs it.")
        print("See docs/LIVE_DATA.md, or run: idxbot paste <TICKER>")
        return 1

    summary = pd.concat(summaries, ignore_index=True)
    data_is_real = not str(summary["source"].iloc[0]).startswith("synthetic")

    pooled = pd.concat(campaign_frames, ignore_index=True) if campaign_frames \
        else pd.DataFrame()
    book = playbook_mod.build_playbook(pooled, engine.cfg.brokers,
                                       min_campaigns=args.min_campaigns) \
        if not pooled.empty else pd.DataFrame()

    tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip())
    matrix = coord.coordination_matrix(summary, engine.cfg.brokers, tiers=tiers)
    leaders = coord.lead_lag(summary, engine.cfg.brokers, tiers=tiers,
                             max_lag=args.max_lag)
    stages = coord.campaign_stage_returns(pooled, prices) if not pooled.empty \
        else pd.DataFrame()
    stage_summary = coord.summarise_stage_returns(stages)

    print(coord.render_plan(book, matrix, leaders, stage_summary,
                            data_is_real=data_is_real,
                            provenance=engine.data_provenance()))

    if args.out:
        _write_csv(stages, args.out, "stage returns")
    return 0


def cmd_backtest(args) -> int:
    engine = _engine(args)
    tickers = _resolve_tickers(engine.cfg, args)
    print(f"Walk-forward evaluation across {len(tickers)} tickers "
          f"(every {args.step} bars)...\n")

    results = backtest_mod.run(engine, tickers, step=args.step,
                               start_index=args.start_index,
                               verbose=not args.quiet)
    print()
    print(backtest_mod.render(results, engine.cfg))
    _write_csv(results.get("observations", pd.DataFrame()), args.out, "observations")
    return 0


def cmd_evaluate(args) -> int:
    """Cross-sectional evaluation of saved backtest observations."""
    from . import evaluate as ev

    if not os.path.exists(args.observations):
        print(f"No observations file at {args.observations}.")
        print("Produce one first:  idxbot backtest --universe all --providers none "
              "--out reports/obs.csv")
        return 2

    df = pd.read_csv(args.observations, parse_dates=["date"])
    horizons = tuple(args.horizons) if args.horizons else (5, 10, 20, 60)

    print("=" * 78)
    print(" CROSS-SECTIONAL EVALUATION")
    print("=" * 78)
    print(f" observations : {len(df):,}")
    print(f" dates        : {df['date'].nunique():,}")
    print(f" tickers      : {df['ticker'].nunique()}")
    print(f" period       : {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}")
    print()
    print(" A screener is used cross-sectionally: 'of the names in front of me today,")
    print(" which do I buy?' These metrics rank within each date, so market direction")
    print(" cancels. The t-statistic is computed over dates - each date is one")
    print(" largely independent observation, unlike the pooled test in `backtest`.")
    print()

    print(ev.render_ic(ev.rank_ic(df, args.signal, horizons),
                       f"RANK IC - {args.signal} (full sample)"))
    print()
    print(ev.render_spread(ev.quantile_spread(df, args.signal, horizons, args.quantiles),
                           f"TOP-MINUS-BOTTOM QUINTILE - {args.signal}"))

    if args.split:
        train, test = ev.split_sample(df, args.split_fraction)
        print()
        print(ev.render_ic(ev.rank_ic(train, args.signal, horizons),
                           f"RANK IC - TRAIN ({train['date'].min():%Y-%m} to "
                           f"{train['date'].max():%Y-%m})"))
        print()
        print(ev.render_ic(ev.rank_ic(test, args.signal, horizons),
                           f"RANK IC - HOLDOUT ({test['date'].min():%Y-%m} to "
                           f"{test['date'].max():%Y-%m})"))

    if args.components:
        print()
        scan = ev.component_scan(df, horizons=(args.component_horizon,))
        print(ev.render_ic(scan, f"COMPONENT RANK IC at {args.component_horizon}d "
                                 f"(each component judged on its own)"))
        print()
        print(" A component with a positive IC helps; a negative one is actively")
        print(" hurting the composite it is weighted into.")
        _write_csv(scan, args.out, "component scan")

    return 0


def cmd_daytrade(args) -> int:
    """Intraday momentum: end-of-day scan, live monitor, or path study."""
    from . import daytrade as dt
    from .data.intraday import YahooIntraday

    engine = _engine(args)
    cfg = engine.cfg
    tickers = _resolve_tickers(cfg, args)

    print("=" * 74)
    print(" INTRADAY MOMENTUM DAY TRADING")
    print("=" * 74)
    print(" Measured on 296,344 real IDX stock-days (2001-2026):")
    print("   a random stock-day touches +5% from the open  6.9% of the time")
    print("   the 'burst' setup below raises that to       38.7%  (5.6x)")
    print("   but it fired only 416 times in 25 years across 66 names")
    print(" Net expectancy on daily bars spans -0.56% to +0.52% per trade: 13.5%")
    print(" of trades touch BOTH target and stop, and daily bars cannot say which")
    print(" came first. Use --study with intraday data to resolve it.")
    print()

    print(f"Scanning {len(tickers)} tickers...")
    bars = {}
    for ticker in tickers:
        df = engine.prices(ticker)
        if df is not None and not df.empty:
            bars[ticker] = df

    candidates = dt.scan(bars, cfg, as_of=args.as_of,
                         min_value_traded=float(cfg.get(
                             "daytrade.min_value_traded_idr", 5e9)))
    if args.setup:
        candidates = [c for c in candidates if c.setup == args.setup]

    if not candidates:
        print("\n  No qualifying setups. That is the normal outcome - this setup")
        print("  fires roughly 17 times a year across a universe this size.")
        print("  Widening the universe is the lever; loosening the filter is not.")
        return 0

    print(f"\n{len(candidates)} candidate(s):\n")
    rows = pd.DataFrame([c.to_row() for c in candidates])
    cols = ["ticker", "date", "setup", "score", "close", "rvol", "day_return",
            "atr_pct", "p_touch_5"]
    print(rows[[c for c in cols if c in rows.columns]].to_string(index=False))

    if args.study:
        print("\n" + "=" * 74)
        print(" PATH RESOLUTION on intraday bars (which came first: target or stop?)")
        print("=" * 74)
        intraday_loader = YahooIntraday(cfg)
        intraday = {}
        for c in candidates[: args.limit_study]:
            df = intraday_loader.get(c.ticker, interval="5m")
            if df is not None and not df.empty:
                intraday[c.ticker] = df
        paths = dt.study_paths(candidates, intraday, cfg,
                               target_pct=args.target, stop_pct=args.stop)
        if paths.empty:
            print(" No overlap between candidates and available intraday history.")
            print(" Yahoo serves only ~60 days of 5-minute bars, so only very recent")
            print(" signals can be resolved this way.")
        else:
            print(paths.to_string(index=False))
            counts = paths["outcome"].value_counts()
            print("\n outcomes:", dict(counts))
        _write_csv(paths, args.out, "path study")
        return 0

    for candidate in candidates[: args.plans]:
        plan = dt.build_day_plan(candidate, cfg, equity=args.equity)
        print()
        print(plan.render())

    _write_csv(rows, args.out, "day-trade candidates")
    return 0


def cmd_paste(args) -> int:
    """Ingest a broker-summary table copied from your trading platform."""
    from . import ingest

    cfg = load_config()
    print("=" * 74)
    print(" PASTE BROKER SUMMARY")
    print("=" * 74)
    print(" This is the one input the engine cannot fetch for itself. Select the")
    print(" broker-summary table in your platform, copy it, and paste it here.")
    print(" Tabs, pipes, multi-space or comma separated all work, as does the")
    print(" usual side-by-side buyers|sellers layout and Indonesian numbers.")
    print()

    if args.file:
        with open(args.file, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        print(f" Paste the table for {args.ticker.upper()}, then press Ctrl-D:")
        print()
        text = sys.stdin.read()

    date = args.date or pd.Timestamp.now().normalize()
    frame, report = ingest.parse_pasted(text, args.ticker, date,
                                        volume_unit=args.volume_unit)
    print()
    print(f" layout detected : {report['layout']}")
    print(f" lines read      : {report['rows_seen']}")
    if frame.empty:
        print("\n No broker rows found. The parser anchors on 2-3 letter broker")
        print(" codes (BK, AK, YP...). Check that the codes are in the pasted text.")
        if report["skipped"]:
            print(" First few unparsed lines:")
            for line in report["skipped"][:5]:
                print(f"   {line}")
        return 1

    print(ingest.describe(frame, cfg))
    if args.dry_run:
        print("\n --dry-run: nothing written.")
        print(frame.to_string(index=False))
        return 0

    path = ingest.save(frame, cfg, args.ticker)
    print(f"\n wrote -> {path}")
    print(" The CSV provider picks this up automatically. Now run:")
    print(f"   idxbot analyze {args.ticker.upper()}")
    print(f"   idxbot screen --providers csv")
    return 0


def cmd_invest(args) -> int:
    """Long-horizon portfolio: the validated 60-day momentum basket."""
    from . import invest as inv

    engine = _engine(args)
    tickers = _resolve_tickers(engine.cfg, args)

    if args.horizons:
        print("HORIZON COMPARISON - what the evidence actually supports\n")
        print(inv.compare_horizons(engine.cfg).to_string(index=False))
        print("\n The slowest horizon has the best evidence. That is the opposite")
        print(" of most people's instinct, and it is what the data says.")
        return 0

    print(f"Screening {len(tickers)} names for the long-horizon basket...\n")
    plan = inv.build(
        engine, tickers, equity=args.equity, top_n=args.top_n,
        horizon_days=args.hold_days, max_weight=args.max_weight,
        min_score=args.min_score, max_atr_pct=args.max_atr,
    )
    print()
    print(plan.render())
    if args.out and plan.holdings:
        _write_csv(pd.DataFrame([h.to_row() for h in plan.holdings]), args.out,
                   "holdings")
    return 0


def cmd_portfolio(args) -> int:
    """Long-only portfolio simulation from saved observations."""
    from . import portfolio as pf
    from . import evaluate as ev

    if not os.path.exists(args.observations):
        print(f"No observations file at {args.observations}.")
        print("Produce one first:  idxbot backtest --universe all --providers none "
              "--out reports/obs.csv")
        return 2

    df = pd.read_csv(args.observations, parse_dates=["date"])

    # Survivorship-free benchmark leg.
    index_prices = None
    try:
        index_prices = _engine(args).benchmark()
    except Exception as exc:
        print(f"  (IHSG benchmark unavailable: {exc})")

    def run(frame: pd.DataFrame, label: str) -> None:
        result = pf.simulate(
            frame, top_n=args.top_n, horizon=args.horizon,
            cost_per_side=args.cost, min_score=args.min_score,
            index_prices=index_prices,
        )
        print(pf.render(result, label))
        print()
        if args.out and not result.empty:
            _write_csv(pf.equity_curve(result), args.out, f"equity curve ({label})")

    if args.split:
        train, test = ev.split_sample(df, args.split_fraction)
        run(train, f"TRAIN {train['date'].min():%Y-%m} to {train['date'].max():%Y-%m}")
        run(test, f"HOLDOUT {test['date'].min():%Y-%m} to {test['date'].max():%Y-%m}")
    else:
        run(df, f"{df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}")
    return 0


def cmd_dashboard(args) -> int:
    engine = _engine(args)
    tickers = _resolve_tickers(engine.cfg, args)
    print(f"Building dashboard for {len(tickers)} tickers...\n")

    results = engine.screen(tickers, with_campaigns=False)
    if results.empty:
        print("No results.")
        return 1

    top = results.head(args.detail)["ticker"].tolist()
    analyses, plans = [], {}
    _, pooled = engine.portfolio_playbook(tickers, min_campaigns=args.min_campaigns)

    for ticker in top:
        analysis = engine.analyze(ticker, with_campaigns=True)
        if analysis is None:
            continue
        analyses.append(analysis)
        plans[ticker] = plan_mod.build_plan(analysis, engine, equity=args.equity,
                                            pooled_playbook=pooled)

    html_text = dashboard.render(
        results, analyses, plans, pooled, engine.cfg,
        provenance=engine.data_provenance(), detail_limit=args.detail,
    )
    path = dashboard.write(args.out, html_text)
    print(f"\n  wrote dashboard -> {path}")
    print(f"  open it with: file://{os.path.abspath(path)}")
    return 0


def cmd_live(args) -> int:
    """Reconstruct broker summary live from a running-trade stream."""
    from .data.running_trade import RunningTradeAggregator, intraday_pace

    cfg = load_config()
    aggregator = RunningTradeAggregator(
        session_date=pd.Timestamp(args.date) if args.date else pd.Timestamp.now().normalize()
    )

    print("=" * 78)
    print(" LIVE BROKER SUMMARY FROM RUNNING TRADE")
    print("=" * 78)
    print(" Broker summary is an aggregation of running trade: every print carries a")
    print(" buyer and a seller member code. Running trade is live on every Indonesian")
    print(" platform, so aggregating it yourself removes the end-of-day delay.")
    print(" See docs/LIVE_DATA.md for how to obtain the stream.")
    print()

    def report(agg: RunningTradeAggregator) -> None:
        snapshot = agg.snapshot()
        if snapshot.empty:
            return
        os.system("")  # no-op; keeps output flushing predictable on some terminals
        for ticker in sorted(snapshot["ticker"].unique()):
            rows = snapshot[snapshot["ticker"] == ticker].copy()
            rows["net_lot"] = rows["buy_lot"] - rows["sell_lot"]
            rows["tier"] = rows["broker"].map(lambda c: cfg.brokers.get(c).tier)
            top = rows.reindex(rows["net_lot"].abs().sort_values(ascending=False).index).head(10)
            pace = intraday_pace(agg, ticker)
            print(f"\n--- {ticker}  ({agg.tick_count:,} ticks"
                  + (f", {pace['session_pct']:.0%} of session" if "session_pct" in pace else "")
                  + ") ---")
            print(f"   {'broker':<8}{'tier':<12}{'net lot':>12}{'buy avg':>10}{'sell avg':>10}")
            for _, r in top.iterrows():
                print(f"   {r['broker']:<8}{r['tier']:<12}{r['net_lot']:>12,.0f}"
                      f"{r['buy_avg']:>10,.0f}{r['sell_avg']:>10,.0f}")

    if args.stdin:
        print(" Reading newline-delimited JSON ticks from stdin...")
        aggregator.ingest_stream(sys.stdin)
        report(aggregator)
    elif args.file:
        if args.follow:
            print(f" Following {args.file} (Ctrl-C to stop)...")
            try:
                aggregator.tail_file(args.file, poll_seconds=args.poll,
                                     on_update=report, max_seconds=args.max_seconds)
            except KeyboardInterrupt:
                print("\n stopped")
        else:
            from .data.running_trade import from_ticks_file
            snapshot = from_ticks_file(args.file, ticker=args.ticker)
            if snapshot.empty:
                print(" No usable ticks found. Expected fields: ts, ticker, price, lot,"
                      " buyer, seller.")
                return 1
            aggregator.ingest([])
            print(snapshot.to_string(index=False))
            _write_csv(snapshot, args.out, "broker summary")
            return 0
    else:
        print(" Nothing to read. Pass --file PATH or --stdin.")
        return 1

    _write_csv(aggregator.snapshot(), args.out, "broker summary")
    return 0


def cmd_pine(args) -> int:
    if args.list:
        print("Bundled Pine scripts:")
        for name in tv.list_pine():
            print(f"  {name}")
        return 0

    if args.ticker:
        engine = _engine(args)
        analysis = engine.analyze(args.ticker.upper(), with_campaigns=True)
        if analysis is None:
            print(f"No data for {args.ticker}.")
            return 1
        pooled = None
        if args.pool:
            _, pooled = engine.portfolio_playbook(engine.cfg.universe(args.pool))
        plan = plan_mod.build_plan(analysis, engine, equity=args.equity,
                                   pooled_playbook=pooled)
        print("// Paste this over the inputs block of broker_campaign.pine")
        print(tv.pine_inputs(plan))
        return 0

    print(tv.read_pine(args.script))
    return 0


def cmd_watchlist(args) -> int:
    engine = _engine(args)
    tickers = _resolve_tickers(engine.cfg, args)

    if args.screen:
        results = engine.screen(tickers, with_campaigns=False)
        path = tv.watchlist_from_screener(results, args.out)
        print(f"\n  wrote grouped watchlist -> {path}")
    else:
        path = tv.export_watchlist(tickers, args.out)
        print(f"  wrote watchlist ({len(tickers)} symbols) -> {path}")
    print("  Import in TradingView: watchlist menu -> Import list...")
    return 0


def cmd_brokers(args) -> int:
    cfg = load_config()
    registry = cfg.brokers
    print(f"{len(registry)} exchange members configured "
          f"(edit config/brokers.yaml; verify against IDX's member directory)\n")
    print(f" {'code':<6}{'tier':<12}{'foreign':<9}{'conf':<10}name")
    print(" " + "-" * 72)
    for code in registry.codes():
        b = registry.get(code)
        if args.tier and b.tier != args.tier:
            continue
        print(f" {b.code:<6}{b.tier:<12}{str(b.foreign):<9}{b.confidence:<10}{b.name}")
    return 0


def cmd_data(args) -> int:
    """Report what history is actually retrievable, per ticker."""
    engine = _engine(args)
    tickers = _resolve_tickers(engine.cfg, args)

    print("Checking available history (Yahoo daily bars, from each first trade date)\n")
    rows = []
    for ticker in tickers:
        bars = engine.prices(ticker, force_refresh=args.refresh)
        if bars is None or bars.empty:
            rows.append({"ticker": ticker, "bars": 0, "first": None, "last": None,
                         "years": 0.0})
            continue
        first, last = bars["date"].iloc[0], bars["date"].iloc[-1]
        rows.append({
            "ticker": ticker,
            "bars": len(bars),
            "first": first.date(),
            "last": last.date(),
            "years": round((last - first).days / 365.25, 1),
        })

    index = engine.cfg.indices.get("composite", "^JKSE")
    jk = engine.prices(index)
    if not jk.empty:
        print(f" IHSG ({index}): {jk['date'].iloc[0]:%Y-%m-%d} -> "
              f"{jk['date'].iloc[-1]:%Y-%m-%d}  ({len(jk):,} sessions)\n")

    df = pd.DataFrame(rows).sort_values("years", ascending=False)
    print(df.to_string(index=False))
    print(f"\n total bars: {df['bars'].sum():,} across {len(df)} tickers")
    print(" Note: Yahoo carries no IDX data before 1990, and most individual stocks")
    print(" start between 2000 and 2004. There is no free source for the 1977-1990")
    print(" era of the exchange.")
    _write_csv(df, args.out, "data availability")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idxbot",
        description="IDX broker-flow accumulation engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, universe_default="lq45"):
        p.add_argument("--universe", default=universe_default,
                       help="named universe from config/universe.yaml, or 'all'")
        p.add_argument("--tickers", nargs="+", help="explicit tickers, overrides --universe")
        p.add_argument("--limit", type=int, help="cap the number of tickers")
        p.add_argument("--providers", help="comma-separated broker summary providers "
                                           "(csv, goapi, synthetic)")
        p.add_argument("--quiet", action="store_true")
        p.add_argument("--profile", help="score weight profile (accumulation, "
                                         "momentum, momentum_plus_flow)")
        p.add_argument("--as-of", help="decide as of this date; all later bars "
                                       "are hidden from the engine")
        p.add_argument("--out", help="write results to this CSV path")

    # screen
    p = sub.add_parser("screen", help="rank a universe by accumulation score")
    common(p)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--top", type=int, help="show only the top N")
    p.add_argument("--explain", type=int, default=10, help="explain the top N results")
    p.add_argument("--campaigns", action="store_true",
                   help="also segment campaigns (slower)")
    p.set_defaults(func=cmd_screen)

    # analyze
    p = sub.add_parser("analyze", help="deep dive on one ticker")
    p.add_argument("ticker")
    p.add_argument("--providers")
    p.add_argument("--profile")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out")
    p.add_argument("--campaign-limit", type=int, default=12)
    p.set_defaults(func=cmd_analyze)

    # plan
    p = sub.add_parser("plan", help="generate an executable trading plan")
    common(p)
    p.add_argument("--equity", type=float, help="account equity in IDR")
    p.add_argument("--risk", type=float, help="fraction of equity risked per trade")
    p.add_argument("--pool", help="universe to pool broker campaign history from")
    p.add_argument("--all", action="store_true", help="print SKIP plans too")
    p.set_defaults(func=cmd_plan)

    # playbook
    p = sub.add_parser("playbook", help="reverse-engineer how each broker operates")
    common(p)
    p.add_argument("--min-campaigns", type=int, default=5)
    p.add_argument("--tier", help="restrict output to one tier (bulge, foreign, ...)")
    p.add_argument("--edge", action="store_true",
                   help="also run the forward-return edge test")
    p.add_argument("--playbook-out", help="write the playbook table to this CSV")
    p.set_defaults(func=cmd_playbook)

    # verify
    p = sub.add_parser("verify", help="acceptance test for broker-summary data")
    common(p)
    p.set_defaults(func=cmd_verify)

    # reverse
    p = sub.add_parser("reverse", help="reverse-engineer the institutional plan")
    common(p)
    p.add_argument("--min-campaigns", type=int, default=5)
    p.add_argument("--tiers", default="bulge",
                   help="comma-separated tiers to analyse (bulge, foreign, ...)")
    p.add_argument("--max-lag", type=int, default=5)
    p.set_defaults(func=cmd_plan_reverse)

    # backtest
    p = sub.add_parser("backtest", help="walk-forward evaluation of the score")
    common(p)
    p.add_argument("--step", type=int, default=5, help="score every Nth bar")
    p.add_argument("--start-index", type=int, default=300,
                   help="skip this many warm-up bars")
    p.set_defaults(func=cmd_backtest)

    # evaluate
    p = sub.add_parser("evaluate", help="cross-sectional IC / quantile evaluation")
    p.add_argument("--observations", default="reports/obs_components.csv",
                   help="CSV produced by `idxbot backtest --out ...`")
    p.add_argument("--signal", default="score", help="column to evaluate")
    p.add_argument("--horizons", type=int, nargs="+")
    p.add_argument("--quantiles", type=int, default=5)
    p.add_argument("--split", action="store_true", help="also show train/holdout")
    p.add_argument("--split-fraction", type=float, default=0.5)
    p.add_argument("--components", action="store_true",
                   help="rank IC for each component individually")
    p.add_argument("--component-horizon", type=int, default=20)
    p.add_argument("--out", help="write the component scan to this CSV")
    p.set_defaults(func=cmd_evaluate)

    # daytrade
    p = sub.add_parser("daytrade", help="intraday momentum scan and plans")
    common(p, universe_default="all")
    p.add_argument("--setup", choices=["burst", "surge"], help="filter by setup")
    p.add_argument("--plans", type=int, default=3, help="print plans for the top N")
    p.add_argument("--equity", type=float)
    p.add_argument("--study", action="store_true",
                   help="resolve target-vs-stop ordering on intraday bars")
    p.add_argument("--limit-study", type=int, default=20)
    p.add_argument("--target", type=float, default=0.05)
    p.add_argument("--stop", type=float, default=0.03)
    p.set_defaults(func=cmd_daytrade)

    # paste
    p = sub.add_parser("paste", help="ingest broker summary copied from your platform")
    p.add_argument("ticker")
    p.add_argument("--file", help="read from a file instead of stdin")
    p.add_argument("--date", help="trading date (default: today)")
    p.add_argument("--volume-unit", default="auto", choices=["auto", "lot", "share"])
    p.add_argument("--dry-run", action="store_true", help="parse but do not save")
    p.set_defaults(func=cmd_paste)

    # invest
    p = sub.add_parser("invest", help="long-horizon portfolio (validated 60d momentum)")
    common(p, universe_default="lq45")
    p.add_argument("--equity", type=float)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--hold-days", type=int, default=60)
    p.add_argument("--max-weight", type=float, default=0.20)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--max-atr", type=float, default=0.10,
                   help="exclude names whose average daily range exceeds this")
    p.add_argument("--horizons", action="store_true",
                   help="show what evidence each horizon actually has")
    p.set_defaults(func=cmd_invest)

    # portfolio
    p = sub.add_parser("portfolio", help="long-only top-N portfolio simulation")
    p.add_argument("--observations", default="reports/obs_momentum.csv")
    p.add_argument("--top-n", type=int, default=10, help="names held")
    p.add_argument("--horizon", type=int, default=60, help="holding period in bars")
    p.add_argument("--cost", type=float, default=0.002, help="cost per side")
    p.add_argument("--min-score", type=float, help="only hold names above this score")
    p.add_argument("--providers")
    p.add_argument("--profile")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--split", action="store_true", help="show train and holdout")
    p.add_argument("--split-fraction", type=float, default=0.5)
    p.add_argument("--out", help="write the equity curve to this CSV")
    p.set_defaults(func=cmd_portfolio)

    # dashboard
    p = sub.add_parser("dashboard", help="render the offline HTML dashboard")
    common(p)
    p.add_argument("--detail", type=int, default=8, help="detailed cards for the top N")
    p.add_argument("--min-campaigns", type=int, default=5)
    p.add_argument("--equity", type=float)
    p.set_defaults(func=cmd_dashboard, out="reports/dashboard.html")
    p.set_defaults(out="reports/dashboard.html")

    # live
    p = sub.add_parser("live", help="live broker summary from a running-trade stream")
    p.add_argument("--file", help="running-trade file (JSONL or CSV)")
    p.add_argument("--stdin", action="store_true", help="read JSONL ticks from stdin")
    p.add_argument("--follow", action="store_true", help="tail the file continuously")
    p.add_argument("--poll", type=float, default=2.0, help="seconds between polls")
    p.add_argument("--max-seconds", type=float, help="stop following after N seconds")
    p.add_argument("--date", help="session date for clock-only timestamps (YYYY-MM-DD)")
    p.add_argument("--ticker", help="filter to one ticker")
    p.add_argument("--out", help="write the reconstructed summary to CSV")
    p.set_defaults(func=cmd_live)

    # pine
    p = sub.add_parser("pine", help="Pine Script source, or plan inputs for a ticker")
    p.add_argument("ticker", nargs="?", help="emit plan inputs for this ticker")
    p.add_argument("--script", default="accumulation_score",
                   help="bundled script to print")
    p.add_argument("--list", action="store_true", help="list bundled scripts")
    p.add_argument("--pool", help="universe to pool broker playbook from")
    p.add_argument("--equity", type=float)
    p.add_argument("--providers")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_pine)

    # watchlist
    p = sub.add_parser("watchlist", help="export a TradingView watchlist")
    common(p)
    p.add_argument("--screen", action="store_true",
                   help="screen first and group by signal level")
    p.set_defaults(func=cmd_watchlist, out="reports/watchlist.txt")
    p.set_defaults(out="reports/watchlist.txt")

    # brokers
    p = sub.add_parser("brokers", help="list configured exchange members")
    p.add_argument("--tier")
    p.set_defaults(func=cmd_brokers)

    # data
    p = sub.add_parser("data", help="report retrievable history per ticker")
    common(p, universe_default="deep_history")
    p.add_argument("--refresh", action="store_true", help="bypass the cache")
    p.set_defaults(func=cmd_data)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 60)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
