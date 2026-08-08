"""idxbot command line interface.

    idxbot screen      --universe lq45          rank a universe by accumulation
    idxbot analyze     BBCA                     deep dive on one ticker
    idxbot plan        BBCA                     executable trading plan
    idxbot playbook    --universe lq45          reverse-engineer broker behaviour
    idxbot backtest    --universe lq45          does the score predict returns?
    idxbot evaluate    --split --components     cross-sectional IC, train/holdout
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
    return Engine(cfg, provider_names=providers,
                  verbose=not getattr(args, "quiet", False),
                  profile=getattr(args, "profile", None))


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
