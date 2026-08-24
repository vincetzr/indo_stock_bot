#!/usr/bin/env python3
"""The twice-daily brief.

    python3 scripts/refresh.py --panel --tables    # once a day, ~6 min
    python3 scripts/brief.py --session pre         # before the 09:00 WIB open
    python3 scripts/brief.py --session post        # after the 15:50 WIB close
    python3 scripts/brief.py --ticker BBCA         # one name, in detail
    python3 scripts/brief.py --save                # also write md + html

WHAT IT ANSWERS, AND HOW EACH ANSWER IS GROUNDED
--------------------------------------------------
    what the market is doing   breadth, dispersion, regime, limit-ups, movers.
                               Arithmetic on 800-odd names. Exact.
    what moved overnight       the markets that trade AFTER Jakarta closes,
                               with a measured record of which of them IDX has
                               historically tracked — not folklore about
                               tailwinds.
    the narratives             co-movement groups derived from returns, plus
                               headlines from four public RSS feeds with UMA,
                               suspension and corporate-action tagging.
    potential candidates       one fused watchlist: run state, cell history,
                               event tags, round-trip cost, feature hits.
    is the run over            where the move sits in its own history and what
                               followed bars in the same cell — base rate,
                               effective n, bootstrap interval, permutation
                               null.

WHAT IT IS NOT
---------------
A forecast, and not a buy list. Four instruments were run to their end in this
repository — aggregate broker flow (H9), broker identity (H10/H11), investor
class (H12) and price/TA (H13) — and none produced an edge that survived
costs. H13 is the sharpest: all eight registered price features are
statistically real and every one is net-negative after 56 bps of fees plus a
fraksi-harga half-spread. Sections state their own status individually, and
nothing here is evidence that trading it pays.

THE HOLDOUT
------------
§11 reserves the most recent 24 months to be spent once. Every reference
distribution — the conditional cells, the overnight sensitivities — is
estimated on `holdout == False` rows, which also makes them out-of-sample with
respect to the bar being described.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import textwrap

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.report import brief as B                          # noqa: E402

WIB = dt.timezone(dt.timedelta(hours=7))
PANEL = os.path.join("data", "spine", "price_panel.parquet")
W = 96
_LINES: list = []


def out(s: str = "") -> None:
    print(s)
    _LINES.append(s)


def rule(title: str = "") -> None:
    out("=" * W)
    if title:
        out(f" {title}")
        out("=" * W)


def wrap(text: str, indent: str = " ") -> None:
    for line in textwrap.wrap(text, width=W - 2):
        out(indent + line)


def pct(x, d: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.{d}%}"


# ==========================================================================
def section_overnight(loader, day, session: str, blob) -> pd.DataFrame:
    from idxbot.data import overnight as O
    rule("1. OVERNIGHT — what moved while Jakarta was shut")
    bars = O.load(loader, [s for s, _ in O.SYMBOLS])
    if not bars:
        out(" no global series available\n")
        return pd.DataFrame()
    Bd = O.board(bars, day)
    out(" Jakarta closes 15:50 WIB = 08:50 UTC. New York closes 20:00 UTC and")
    out(" London 15:30 UTC ON THE SAME DATE, so their bar dated like today's")
    out(" IDX session landed hours after it — that is the overnight column.")
    out(" Tokyo, Hong Kong and Shanghai close BEFORE Jakarta, so their session")
    out(" was already visible while IDX traded; they show no overnight number")
    out(" because for IDX there is none, only context.")
    out("")
    out(f" {'':<16}{'last':>12}{'overnight':>11}{'1d':>9}{'5d':>9}"
        f"{'1m':>9}   {'as of':<11}")
    for group in Bd["group"].unique():
        g = Bd[Bd["group"] == group]
        out(f" {group}")
        for _, r in g.iterrows():
            ovn = pct(r["overnight"], 2) if np.isfinite(r["overnight"]) \
                else ("  —" if not r["after_jakarta"] else "  ·")
            flag = "  (feed behind)" if r["behind"] and r["after_jakarta"] \
                else ("  " + r["proxy"] if r["proxy"] else "")
            out(f"   {r['name']:<14}{r['last']:>12,.2f}{ovn:>11}"
                f"{pct(r['d1']):>9}{pct(r['d5']):>9}{pct(r['d21']):>9}   "
                f"{str(r['asof'])[:10]:<11}{flag}")
    out("")
    out(" '—' = closes before Jakarta, so no overnight number exists.")
    out(" '·' = closes after Jakarta but has not printed since.")

    if blob is not None and not blob["rows"].empty:
        S = blob["rows"]
        out("")
        out(f" WHICH OF THESE IDX HAS ACTUALLY TRACKED  "
            f"({blob['n_sessions']:,} pre-holdout sessions, "
            f"{blob['date_min']} to {blob['date_max']})")
        out(" Rank correlation of IDX's equal-weighted session return with each")
        out(" market's PREVIOUS completed move — the one Jakarta could act on.")
        out("")
        out(f"   {'':<14}{'rank corr':>11}{'95% CI':>18}{'vs null':>9}"
            f"{'stale':>8}")
        for _, r in S.head(10).iterrows():
            out(f"   {r['name']:<14}{r['r']:>+11.3f}"
                f"{f'[{r.lo:+.2f},{r.hi:+.2f}]':>18}{r['z']:>+9.1f}"
                f"{r['stale']:>8.0%}")
        out("")
        wrap("RANK correlation, not Pearson: these series carry kurtosis from "
             "10 to 2,800, and a Pearson estimate on that is a statistic "
             "about its four largest days. Computing it that way first said "
             "the S&P was uncorrelated with IDX (-0.001) when the rank "
             "figure is +0.207. Yahoo's IDR=X also carries decimal-shift "
             "defects — 888.11 where the true rate was ~8,881 — which are "
             "dropped, and 'stale' is the share of days a series merely "
             "repeated its last print.")
        out("")
        wrap("EVEN THE STRONGEST OF THESE IS CONTEXT, NOT A SIGNAL. A rank "
             "correlation of 0.21 explains about 4% of variance; against a "
             "typical daily cross-sectional spread that is a fraction of the "
             "56 bps round trip before any spread is added. Asia reads near "
             "zero partly because the alignment deliberately uses the "
             "previous session for every market rather than risk a leak.")
    out("")
    return Bd


def section_state(P, S, day) -> tuple:
    rule("2. MARKET STATE — arithmetic, no claim attached")
    b = B.breadth(S, day)
    r = B.regime(P, day)
    L = B.limit_moves(S, day)
    out(f" {b['n_names']} names traded.  {b['advancing']:.0%} advanced, "
        f"{b['unchanged']:.0%} did not move at all (the illiquid tail), "
        f"median {b['median_move']:+.2%}")
    out(f" cross-sectional spread of today's returns: {b['dispersion']:.2%}")
    out("")
    out(f" {'':<12}{'1d':>9}{'1w':>9}{'1m':>9}{'3m':>9}{'ytd':>9}"
        f"{'20d vol':>10}{'vs 5y':>8}")
    for name, lab in (("equal", "equal-wt"), ("turnover", "turnover-wt")):
        if f"{name}_1d" not in r:
            continue
        out(f" {lab:<12}" + "".join(
            f"{pct(r[f'{name}_{h}']):>9}"
            for h in ("1d", "1w", "1m", "3m", "ytd"))
            + f"{r[f'{name}_vol']:>10.1%}{r[f'{name}_vol_pct']:>8.0%}")
    out(" turnover-weighted is a PROXY for cap-weighted — no shares-outstanding")
    out(" series exists here. A gap between the rows is the big names carrying")
    out(" the tape while the median name does not, or the reverse. Not IHSG.")
    out("")
    out(f" above the 20-day  {b['above_20d']:>6.0%}   "
        f"50-day {b['above_50d']:>6.0%}   200-day {b['above_200d']:>6.0%}")
    out(f" at a 250-day high {b['new_highs']:>6}   "
        f"at a 250-day low {b['new_lows']:>5}   of {b['n_250d']}")
    out(f" closed at the auto-rejection band: {L['ara']} ARA, {L['arb']} ARB "
        f"of {L['n']}   (close test only — an upper bound)")
    out("")
    m = B.movers(S)
    out(f" biggest movers, >= Rp {B.MIN_VALUE/1e9:.0f}bn traded and above the "
        f"{B.LIQUID_PCT:.0%} turnover percentile")
    for lab, D in (("up  ", m["up"]), ("down", m["down"])):
        if not D.empty:
            out(f"   {lab}  " + "  ".join(f"{x.ticker} {x.ret:+.1%}"
                                          for x in D.itertuples()))
    out("")
    return b, r, L, m


def section_comovement(P, day) -> pd.DataFrame:
    rule("3. WHAT MOVED TOGETHER — narrative derived from returns")
    cm = B.comovement(P, day, n_pc=5)
    if cm.empty:
        out(" not enough liquid names with a year of history\n")
        return cm
    out(" Components fitted on 250 sessions ending the day BEFORE this one,")
    out(" then today's cross-section projected onto them. A group is named by")
    out(" its members and nothing else — whether eight coal names moving")
    out(" together is 'the coal trade' is your reading, not a finding.")
    out("")
    out(f" {'':<5}{'of hist var':>12}{'of today':>10}{'today':>9}"
        f"{'|size| pct':>12}")
    for _, x in cm.iterrows():
        out(f" PC{x['pc']:<3}{x['var_share']:>12.1%}{x['today_share']:>10.1%}"
            f"{x['score_z']:>+9.2f}z{x['abs_pct']:>11.0%}")
        out(f"        with:    {' '.join(x['with'])}")
        out(f"        against: {' '.join(x['against'])}")
    out("")
    return cm


def section_news(names, no_news: bool, per: int, limit: int):
    rule("4. WHAT IS BEING SAID — public news feeds")
    if no_news:
        out(" skipped (--no-news)\n")
        return None
    from idxbot.data import news
    src = news.source_report()
    out(" sources: " + "  ".join(f"{r.feed}({r.items})"
                                 for r in src.itertuples()))
    if not src["ok"].any():
        out(" no feed answered — this is an outage, not a quiet day.\n")
        return None
    out("")
    M = news.market_news(limit=limit)
    out(" MARKET")
    if M.empty:
        out("   nothing in the window")
    for _, r in M.iterrows():
        tg = ("[" + ",".join(r["tags"]) + "] ") if r["tags"] else ""
        out(f"   {str(r['published'])[:16]}  {tg}{r['title'][:74]}")
        out(f"      {r['source']}")
    T = None
    if names:
        out("")
        out(f" BY NAME  ({len(names)} names on the watchlist)")
        T = news.ticker_news(names, per=per)
        if T is None or T.empty:
            out("   nothing found for any of them")
        else:
            for tk, g in T.groupby("ticker", sort=False):
                out(f"   {tk}")
                for _, r in g.iterrows():
                    tg = ("[" + ",".join(r["tags"]) + "] ") if r["tags"] else ""
                    out(f"     {'new' if r['recent'] else 'old'} "
                        f"{str(r['published'])[:10]}  {tg}{r['title'][:68]}")
            out("")
            out("   'old' items are corporate actions kept beyond the 14-day")
            out("   window because they change what the price series MEANS — a")
            out("   rights issue is §5's named adjustment trap.")
    out("")
    wrap(B.news_caveat())
    out("")
    return T


def section_watchlist(S, states, C, T, blob, n: int) -> None:
    rule("5. WATCHLIST — every name the brief noticed, with its own context")
    Wl = B.watchlist(S, states, C, T, n=n)
    if Wl.empty:
        out(" no liquid name has a usable run state today\n")
        return
    k = blob["k"]
    out(f" Sorted by absolute move today — NOT by attractiveness. There is no")
    out(f" measured attractiveness here and sorting by one would invent it.")
    out("")
    out(f" {'ticker':<7}{'close':>9}{'today':>8}{'leg':>6}{'since':>7}"
        f"{'run':>9}{'run_z':>7}{'off ext':>9}{'excess':>9}{'cost':>7}"
        f"{'net':>7}{'f':>3}  events")
    for _, x in Wl.iterrows():
        ex = f"{x['diff']:+.2%}" if np.isfinite(x.get("diff", np.nan)) else "  n/a"
        nt = f"{x['net']:+.2%}" if np.isfinite(x.get("net", np.nan)) else "  n/a"
        out(f" {x['ticker']:<7}{x['close']:>9,.0f}"
            f"{x.get('ret1', np.nan):>+8.1%}{x['leg']:>6}"
            f"{int(x['run_days']):>7}{x['run_pct']:>+9.1%}"
            f"{x['run_z']:>+7.2f}{x['give_pct']:>+9.1%}{ex:>9}"
            f"{x['cost']:>7.2%}{nt:>7}{x['n_feat']:>3}  {x['events'][:30]}")
    out("")
    out(f" 'excess'  historical mean return of the cell this name currently")
    out(f"           occupies, over {k} sessions, minus the equal-weighted")
    out(f"           return of all liquid names on the same dates.")
    out(" 'cost'    0.56% round trip plus half a tick each way at this price —")
    out("           a FLOOR, since it assumes a one-tick-wide book.")
    out(" 'net'     excess minus cost. A HISTORICAL AVERAGE, not an expectation")
    out("           for this name, and uncorrected for the 54 cells computed.")
    out(" 'f'       how many of the eight registered features rank this name in")
    out("           their top decile. A COUNT, not a blend: a composite of eight")
    out("           separately-tested features is a new signal wearing their")
    out("           credibility, and H13 measured every one net-negative.")
    out("")


def section_conditional(day, blob, R, P) -> pd.DataFrame:
    k = blob["k"]
    rule(f"6. IS THE RUN OVER — what followed states like these, {k} sessions on")
    T, E, N = blob["table"], blob["edges"], blob["null"]
    out(f" Reference: {blob['n_rows']:,} liquid pre-holdout bars, "
        f"{blob['date_min']} to {blob['date_max']}.")
    out(f" A run is measured from the last {B.RUN_WINDOW}-session extreme of the")
    out(" opposite sign; run_z is the move in standard deviations FOR A MOVE OF")
    out(" ITS LENGTH, so a quiet name up 30% in ten days and a volatile one up")
    out(" 30% in two hundred are not confused.")
    out("")
    out(" THE PERMUTATION NULL FIRST — reading a statistic against zero rather")
    out(" than against its own shuffled null has produced a confident wrong")
    out(" answer four separate times in this repo.")
    out(f"   {N['n_cells']} cells.  largest |excess over base rate| observed "
        f"{N['obs_max_abs']:.2%}")
    out(f"   under shuffled state labels: {N['null_max_abs_mean']:.2%} mean, "
        f"{N['null_max_abs_p95']:.2%} at the 95th percentile")
    out(f"   spread across cells: observed {N['obs_spread']:.2%} against a null "
        f"{N['null_spread_mean']:.2%}")
    out(f"   p(null >= observed) = {N['p_max']:.3f}")
    out("")
    if N["p_max"] < 0.05:
        wrap("=> the state conditioning carries information beyond chance. IT "
             "IS STILL IN-SAMPLE AND POST-HOC. Fifty-four cells were computed "
             "and the intervals are uncorrected, so roughly "
             f"{N['expected_false_cells']:.0f} clear zero by luck; the largest "
             "cell is the largest OF FIFTY-FOUR and is biased upward by "
             "exactly the selection that found it. H13 measured very nearly "
             "this and found it net-negative after costs. Treat it as a lead "
             "for a pre-registered test, not a result.", "   ")
    else:
        out("   => indistinguishable from shuffled labels. Read nothing from "
            "the cells.")
    out("")
    D = B.current_states(P, R, day, T, E)
    if not D.empty:
        best = D.head(3)
        worst = D.tail(3)
        out(" the cells today's liquid names actually occupy, best and worst:")
        for lab, g in (("best ", best), ("worst", worst)):
            for _, x in g.iterrows():
                out(f"   {lab} {x['ticker']:<7}{x['what']:<46}"
                    f"{x['diff']:>+8.2%} excess   cost {x['cost']:>6.2%}")
    out("")
    return D


def section_flow(names) -> None:
    rule("7. FLOW — what the repo has, and why it is small")
    f = os.path.join("data", "spine", "investor_split.csv.gz")
    if not os.path.exists(f):
        out(" no flow data collected\n")
        return
    D = pd.read_csv(f)
    D["window_end"] = pd.to_datetime(D["window_end"])
    last = D["window_end"].max()
    out(f" Foreign/domestic split: {D['ticker'].nunique()} names, "
        f"{len(D):,} class-windows, latest window ending {last.date()}.")
    L = D[D["window_end"] == last]
    if not L.empty:
        piv = L.pivot_table(index="ticker", columns="view",
                            values="net_value", aggfunc="sum")
        if "F" in piv:
            piv = piv.reindex(piv["F"].abs().sort_values(
                ascending=False).index)
            out("")
            out(f"   {'ticker':<8}{'foreign net':>16}{'domestic net':>16}")
            for t, r in piv.head(8).iterrows():
                out(f"   {t:<8}{r.get('F', np.nan):>16,.0f}"
                    f"{r.get('D', np.nan):>16,.0f}")
    out("")
    wrap("THIS IS FORTNIGHTLY AND NARROW, AND IT HAS NO PREDICTIVE CONTENT "
         "HERE. H12 measured foreign margin at -1.70 bps a fortnight "
         "(p 0.69) and domestic at +1.02 (p 0.82), against a 56 bps cost bar "
         "— a powered null, 8.8 to 11.3 null-sds from anything tradeable. "
         "H9 found aggregate broker flow no better and H11 found broker "
         "identity does not persist. It is printed as description because the "
         "repo collected it, not because it decides anything.")
    out("")


def section_ticker(P, R, day, blob, t: str) -> None:
    rule(f"DETAIL — {t}")
    T, E = blob["table"], blob["edges"]
    D = B.current_states(P, R, day, T, E, min_value=0.0, liquid_pct=0.0)
    row = D[D["ticker"] == t]
    if row.empty:
        out(f" {t} has no usable run state on {day.date()} — it did not trade,")
        out(f" or it lacks {B.RUN_WINDOW} sessions of history.\n")
        return
    x = row.iloc[0]
    out(f" close {x['close']:,.0f}   {x['leg']} leg, anchored "
        f"{int(x['run_days'])} sessions ago at its 250-day "
        f"{'low' if x['leg'] == 'up' else 'high'}")
    out(f" from the anchor: {x['run_pct']:+.1%}  ({x['run_z']:+.2f} sd for a "
        f"move of that length)")
    out(f" come back from the leg's far end: {x['give_pct']:+.1%}")
    out(f" cell: {x['what']}")
    if np.isfinite(x.get("diff", np.nan)):
        out(f" historically from that cell, over {blob['k']} sessions:")
        out(f"   mean {x['fwd_mean']:+.2%} against a base rate of "
            f"{x['base_mean']:+.2%}  ->  excess {x['diff']:+.2%}")
        out(f"   95% CI [{x['diff_lo']:+.2%}, {x['diff_hi']:+.2%}]   "
            f"n = {int(x['n']):,} bars, n_eff = {int(x['n_eff'])} "
            f"non-overlapping windows")
        out(f"   up {x['p_up']:.0%} of the time")
        out(f"   round trip here costs {x['cost']:.2%}  ->  net {x['net']:+.2%}")
    out("")
    out(" A historical frequency from a cell holding thousands of other bars.")
    out(" Not a forecast for this name, and uncorrected for 54 cells.")
    out("")


def save(session: str, day) -> None:
    os.makedirs("reports", exist_ok=True)
    stamp = f"{day.date()}_{session}"
    md = os.path.join("reports", f"brief_{stamp}.md")
    with open(md, "w") as fh:
        fh.write(f"# IDX brief — {day.date()} {session}\n\n```\n"
                 + "\n".join(_LINES) + "\n```\n")
    latest = os.path.join("reports", "brief_latest.md")
    with open(latest, "w") as fh:
        fh.write(f"# IDX brief — {day.date()} {session}\n\n```\n"
                 + "\n".join(_LINES) + "\n```\n")
    print(f"\n saved {md} and {latest}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", choices=["pre", "post"], default="post")
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--names", type=int, default=8)
    ap.add_argument("--watch", type=int, default=20,
                    help="rows in the fused watchlist")
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--no-overnight", action="store_true")
    ap.add_argument("--news-names", type=int, default=12)
    ap.add_argument("--news-per", type=int, default=3)
    ap.add_argument("--save", action="store_true",
                    help="write reports/brief_<date>_<session>.md")
    ap.add_argument("--build-tables", action="store_true")
    ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()

    if not os.path.exists(a.panel):
        print(f"no panel at {a.panel} — run scripts/price_panel_build.py")
        return 1
    now = dt.datetime.now(tz=WIB)
    P = pd.read_parquet(a.panel)

    from idxbot.config import load_config
    from idxbot.data.cache import Cache
    from idxbot.data.ohlcv import YahooOHLCV
    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))

    if a.build_tables:
        print(f" building reference tables (draws={a.draws}) …", flush=True)
        B.build_tables(P, ks=(5, 20), draws=a.draws, null_draws=a.draws)
        B.build_sensitivity(P, loader, draws=a.draws)
        print(" done")

    blob = B.load_table(a.horizon)
    if blob is None:
        print(f" no reference table for k={a.horizon}. Run --build-tables once.")
        return 1
    sens = B.load_sensitivity()

    day = B.resolve_asof(P)
    warn = B.coverage_warning(P, day)

    rule()
    out(f" IDX BRIEF — {a.session.upper()} session, {now:%Y-%m-%d %H:%M} WIB")
    out(f" IDX bars through {day.date()}")
    rule()
    if warn:
        out(" ! " + warn)
        out("")

    S = B.snapshot(P, day)
    R = B.run_state(P)
    b = B.breadth(S, day)
    reg = B.regime(P, day)
    L = B.limit_moves(S, day)
    cm_pre = B.comovement(P, day, n_pc=5)
    Bd = pd.DataFrame()
    if not a.no_overnight:
        from idxbot.data import overnight as O
        Bd = O.board(O.load(loader, [s for s, _ in O.SYMBOLS]), day)

    out(" THE READ")
    for line in B.headline(b, reg, L, Bd if not Bd.empty else None, cm_pre):
        wrap(line, "   ")
    out("")
    wrap("Every clause above is a printed number from the sections below. "
         "Nothing here says what happens next: four instruments were run to "
         "their end in this repo and none produced an edge that survived "
         "costs.")
    if a.session == "pre":
        out("")
        wrap("PRE-OPEN: IDX data is unchanged since the last close. What IS "
             "new is section 1 — the markets that traded after Jakarta shut.")
    out("")

    if not a.no_overnight:
        section_overnight(loader, day, a.session, sens)
    section_state(P, S, day)
    section_comovement(P, day)

    mv = B.movers(S, n=max(4, a.news_names // 2))
    watch = list(dict.fromkeys(
        list(mv["up"]["ticker"]) + list(mv["down"]["ticker"])
        + ([a.ticker.upper()] if a.ticker else [])))[:a.news_names]
    T = section_news(watch, a.no_news, a.news_per, a.names + 4)

    states = B.current_states(P, R, day, blob["table"], blob["edges"])
    C = B.candidates(P, day, n=max(10, a.watch))
    section_watchlist(S, states, C, T, blob, a.watch)
    section_conditional(day, blob, R, P)
    section_flow(watch)
    if a.ticker:
        section_ticker(P, R, day, blob, a.ticker.upper())

    rule("WHAT WOULD CHANGE THE PICTURE")
    out(" The conditional cells beat their permutation null, so the state")
    out(" conditioning is real. That does NOT make it tradeable: the cells are")
    out(" post-hoc, in-sample and uncorrected for 54 tests, and H13 measured")
    out(" nearly the same thing net-negative. The test that would settle it is")
    out(" pre-registered — fix the cells, the rule, the horizon and the cost")
    out(" model in writing, then spend the 24-month holdout once. That has not")
    out(" been done. The holdout is untouched and every reference table here")
    out(" is built on pre-holdout rows only.")
    out("")
    if a.save:
        save(a.session, day)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
