#!/usr/bin/env python3
"""What IDX actually pays, before anybody tries to be clever about it.

THE QUESTION EVERY OTHER SCRIPT SKIPPED
---------------------------------------
Thirty scripts in this repo ask "can I beat the market". Not one asked what the
market pays, and the answer turns out to change how the first question should be
read. Three things were being got wrong at once:

    1. EVERY BACKTEST RAN ON THE PRICE LINE. load_ohlc reads ``close``, not
       ``adj_close``, so every result in Parts XXVIII-XXXI excluded dividends.
       On the liquid book that omission is worth about two points a year -
       larger than any edge this repo has ever claimed to find.
    2. THE SAMPLE ENDS INSIDE A CRASH. The IHSG went 5,243 (Jan 2015) to 8,748
       (Jan 2026) and then to 6,337 by August. Measuring "the market returned
       2.6%/yr" from a peak-to-trough sample is not a fact about IDX, it is a
       fact about the last eight months, and any strategy scored against it
       inherits that accident.
    3. NOTHING WAS EVER COMPARED TO CASH. An Indonesian saver can hold a bank
       deposit. A strategy that returns less than the deposit rate has negative
       value however impressive its Sharpe looks against zero.

WHAT IS COMPUTED HERE
---------------------
    dividends     the adj_close/close gap per name and at portfolio level
    windows       every 1/3/5/10-year holding period in the sample, so the
                  answer is a DISTRIBUTION rather than one path-dependent number
    breadth       what share of names lost money, and what share beat the index
    currency      the same in USD, because the rupiah did not stand still
    hurdle        all of it against a deposit rate and against inflation

THE BIAS THAT IS DECLARED RATHER THAN FIXED
-------------------------------------------
Every cached name still trades today. Not one delisting is present. So the
single-name distribution below is the distribution among SURVIVORS and the true
one is worse - which makes the diversification conclusion stronger, not weaker,
since the missing names are the failures.

    python3 scripts/base_rates.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.config import load_config           # noqa: E402
from idxbot.data.cache import Cache             # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV        # noqa: E402
from factor_study import (Board, build_panel, cagr, load_index,   # noqa: E402
                          max_dd, rebalance_positions, run_portfolio,
                          total_return_series)

# Stated as assumptions, not measured here, because no IDR rate series is
# cached. Both are roughly the 2015-2026 averages and both are round numbers on
# purpose - the conclusions below do not turn on the second decimal.
DEPOSIT_RATE = 0.055        # 1-year rupiah time deposit, indicative
INFLATION = 0.030           # Indonesian CPI, indicative long-run average


def window_returns(eq: pd.Series, years: float) -> np.ndarray:
    """Annualised return of every holding period of the given length.

    A single start-to-finish number answers "what would have happened to
    somebody who bought on one particular morning". Nobody is that person. The
    distribution over all start dates is what an investor is actually choosing
    between.
    """
    if len(eq) < 3:
        return np.array([])
    out = []
    for i in range(len(eq)):
        target = eq.index[i] + pd.Timedelta(days=int(round(365.25 * years)))
        j = eq.index.searchsorted(target)
        if j >= len(eq):
            break
        span = (eq.index[j] - eq.index[i]).days / 365.25
        if span <= 0 or eq.iloc[i] <= 0:
            continue
        out.append(float((eq.iloc[j] / eq.iloc[i]) ** (1 / span) - 1))
    return np.asarray(out)


def independent_windows(n_windows: int, years: float, span_years: float
                        ) -> float:
    """How many genuinely separate observations a set of overlapping windows is.

    Five ten-year windows inside an eleven-year sample are not five facts, they
    are one fact counted five times, and a percentage computed over them ("100%
    of ten-year holds lost to cash") reads as overwhelming when it rests on a
    single overlapping stretch. This returns the non-overlapping count so the
    claim can be sized honestly.
    """
    if years <= 0 or n_windows <= 0:
        return 0.0
    return max(1.0, span_years / years)


def end_date_sensitivity(eq: pd.Series, years_back: Sequence[int] = (),
                         ) -> pd.DataFrame:
    """The same series' CAGR measured to each year-end it passes through.

    A sample that stops inside a drawdown does not report the market, it reports
    the drawdown. Running the finish line backwards through the sample shows how
    much of any headline number is the asset and how much is the stopping point.
    """
    if len(eq) < 3:
        return pd.DataFrame()
    rows = []
    for y in sorted({d.year for d in eq.index}):
        cut = eq[eq.index <= pd.Timestamp(f"{y}-12-31")]
        span = (cut.index[-1] - cut.index[0]).days / 365.25 if len(cut) else 0
        if span < 3:
            continue
        rows.append({"through": cut.index[-1].date(), "years": span,
                     "cagr": float(cut.iloc[-1] ** (1 / span) - 1),
                     "final": float(cut.iloc[-1])})
    return pd.DataFrame(rows)


def describe(v: np.ndarray) -> Dict[str, float]:
    if not len(v):
        return {"n": 0}
    return {"n": len(v), "worst": float(np.min(v)), "p25": float(np.percentile(v, 25)),
            "median": float(np.median(v)), "p75": float(np.percentile(v, 75)),
            "best": float(np.max(v)), "neg": float((v < 0).mean()),
            "below_cash": float((v < DEPOSIT_RATE).mean())}


def dividend_gap(loader: YahooOHLCV, cache_dir: str, start: str,
                 min_bars: int = 1200) -> pd.DataFrame:
    """Per name: price CAGR, total-return CAGR, and the difference."""
    names = sorted({os.path.basename(f).split(".")[0].upper()
                    for f in glob.glob(os.path.join(cache_dir, "ohlcv",
                                                    "*.JK.csv.gz"))})
    names = [n for n in names if n.isalpha() and len(n) == 4]
    rows = []
    for t in names:
        d = total_return_series(loader, t, total=True)
        if d is None:
            continue
        d = d[d.index >= pd.Timestamp(start)]
        if len(d) < min_bars or float(d["raw"].iloc[0]) <= 0:
            continue
        yrs = (d.index[-1] - d.index[0]).days / 365.25
        if yrs <= 1:
            continue
        price = float(d["raw"].iloc[-1] / d["raw"].iloc[0]) ** (1 / yrs) - 1
        total = float(d["px"].iloc[-1] / d["px"].iloc[0]) ** (1 / yrs) - 1
        rows.append({"ticker": t, "years": yrs, "price": price,
                     "total": total, "dividend": total - price,
                     "turnover": float((d["raw"] * d["volume"]).median())})
    return pd.DataFrame(rows)


def wealth_concentration(total: Sequence[float], years: Sequence[float]
                         ) -> Dict[str, float]:
    """How much of the aggregate gain came from how few names.

    This is the number that decides whether stock picking is a game worth
    playing at all. If the whole market's gain sits in a handful of names, then
    missing them is not bad luck, it is the default outcome - and the only
    defence that does not require predicting them is owning enough of them.
    """
    mult = np.array([(1.0 + t) ** y for t, y in zip(total, years)])
    mult = mult[np.isfinite(mult)]
    if not len(mult):
        return {}
    gain = mult - 1.0
    winners = gain[gain > 0]
    order = np.sort(gain)[::-1]
    total_gain = float(gain.sum())
    out = {"names": len(gain), "losers": float((gain < 0).mean()),
           "halved": float((mult < 0.5).mean()),
           "winner_share": float(winners.sum() / total_gain)
           if total_gain > 0 else np.nan}
    for k in (5, 10, 20):
        n = max(1, int(round(len(order) * k / 100)))
        out[f"top{k}pct"] = float(order[:n].sum() / total_gain) \
            if total_gain > 0 else np.nan
    return out


def usd_curve(eq: pd.Series, fx: pd.Series) -> pd.Series:
    """The same equity curve seen by somebody who counts in dollars."""
    f = fx.reindex(eq.index).ffill().bfill()
    if f.isna().all() or float(f.iloc[0]) <= 0:
        return pd.Series(dtype=float)
    return eq / (f / float(f.iloc[0]))


def real_return(nominal: float, inflation: float = INFLATION) -> float:
    """Purchasing power, not rupiah count. (1+r)/(1+i) - 1, not r - i."""
    return (1.0 + nominal) / (1.0 + inflation) - 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--min-turnover", type=float, default=5e9)
    ap.add_argument("--min-hist", type=int, default=280)
    args = ap.parse_args()

    cfg = load_config()
    cache_dir = cfg.path("data.cache_dir", "data/cache")
    loader = YahooOHLCV(cfg, Cache(cache_dir))

    print(f"{'=' * 96}\n WHAT IDX PAYS — the base rate everything else is "
          f"measured against\n{'=' * 96}")
    close, raw, turn = build_panel(loader, cache_dir, args.start, total=True)
    rebal = rebalance_positions(close.index, "monthly", args.min_hist)
    idx_s = load_index(loader, "^JKSE", close.index)
    idx = idx_s.to_numpy(float) if len(idx_s) else np.full(len(close), np.nan)
    board = Board(close, turn, idx, rebal, [], args.min_turnover,
                  args.min_hist, raw=raw)
    ew, ew_p, ew_sizes, ew_c = run_portfolio(board, None, 0)
    bench = pd.Series(dtype=float)
    if len(idx_s):
        b = idx_s.reindex(ew.index).dropna()
        bench = b / float(b.iloc[0]) if len(b) else bench
    print(f" {close.shape[1]} names, {close.index[0]:%Y-%m-%d} to "
          f"{close.index[-1]:%Y-%m-%d}, "
          f"{int(np.median(ew_sizes))} names pass the liquidity floor")

    # ---- 1. dividends ----------------------------------------------------
    print(f"\n{'=' * 96}\n 1. THE DIVIDEND — the largest reliably positive "
          f"number in this repo\n{'=' * 96}")
    gaps = dividend_gap(loader, cache_dir, args.start)
    liquid = gaps[gaps["turnover"] >= args.min_turnover]
    print(f" {len(gaps)} names with a full history, {len(liquid)} of them "
          f"liquid enough to hold")
    for label, g in (("all names", gaps), ("liquid names", liquid)):
        if g.empty:
            continue
        print(f" {label:<14} price {g['price'].median():>+7.2%}/yr   "
              f"total {g['total'].median():>+7.2%}/yr   "
              f"dividend {g['dividend'].median():>+6.2%}/yr median, "
              f"{g['dividend'].mean():>+6.2%}/yr mean, "
              f"{(g['dividend'] > 0).mean():.0%} pay anything")
    if not liquid.empty:
        top = liquid.nlargest(8, "dividend")
        print("\n biggest dividend contributors among liquid names:")
        print("  " + "  ".join(f"{r.ticker} {r.dividend:+.1%}"
                               for r in top.itertuples()))

    # ---- 2. the market ---------------------------------------------------
    print(f"\n{'=' * 96}\n 2. THE MARKET, MEASURED THREE WAYS\n{'=' * 96}")
    print(f" {'series':<34}{'final':>8}{'CAGR':>9}{'real':>9}{'max DD':>9}")
    lines = [("equal-weight liquid, total return", ew)]
    if len(bench):
        lines.append(("IHSG price index (no dividends)", bench))
    for name, s in lines:
        print(f" {name:<34}{s.iloc[-1]:>8.2f}{cagr(s):>9.2%}"
              f"{real_return(cagr(s)):>9.2%}{max_dd(s):>9.1%}")
    print(f" {'rupiah time deposit (assumed)':<34}{'':>8}{DEPOSIT_RATE:>9.2%}"
          f"{real_return(DEPOSIT_RATE):>9.2%}{0.0:>9.1%}")
    fx = pd.Series(dtype=float)
    d = loader.get("USDIDR=X", max_age=86400 * 30)
    if d is not None and not d.empty:
        fx = d.set_index("date").sort_index()["close"].astype(float)
        u = usd_curve(ew, fx)
        if len(u):
            print(f" {'equal-weight liquid, in USD':<34}{u.iloc[-1]:>8.2f}"
                  f"{cagr(u):>9.2%}{'':>9}{max_dd(u):>9.1%}")
        f0, f1 = float(fx.reindex(ew.index).ffill().iloc[0]), \
            float(fx.reindex(ew.index).ffill().iloc[-1])
        yrs = (ew.index[-1] - ew.index[0]).days / 365.25
        print(f" the rupiah went {f0:,.0f} to {f1:,.0f} per USD, "
              f"{((f1 / f0) ** (1 / yrs) - 1):+.2%}/yr")

    # ---- 2b. where the finish line was drawn -----------------------------
    print(f"\n{'=' * 96}\n 2b. HOW MUCH OF THAT IS JUST WHERE THE SAMPLE "
          f"STOPS\n{'=' * 96}")
    es_ew = end_date_sensitivity(ew)
    es_ix = end_date_sensitivity(bench) if len(bench) else pd.DataFrame()
    print(f" {'measured through':<20}{'years':>7}{'EW book':>10}"
          f"{'IHSG':>10}{'beats cash?':>13}")
    for _, r in es_ew.iterrows():
        ix = es_ix[es_ix["through"] == r["through"]]["cagr"]
        ixv = float(ix.iloc[0]) if len(ix) else np.nan
        print(f" {str(r['through']):<20}{r['years']:>7.1f}{r['cagr']:>10.2%}"
              f"{ixv:>10.2%}"
              f"{'yes' if r['cagr'] > DEPOSIT_RATE else 'no':>13}")
    if len(es_ew) >= 2:
        spread = es_ew["cagr"].max() - es_ew["cagr"].min()
        print(f"\n the same book, the same holdings, the same rules: "
              f"{spread:.2%} of annual return\n depends only on which year you "
              f"stopped counting. Any strategy comparison that\n does not "
              f"survive this table is a comparison of end dates.")

    # ---- 3. when you started --------------------------------------------
    print(f"\n{'=' * 96}\n 3. WHEN YOU STARTED DECIDED ALMOST EVERYTHING\n"
          f"{'=' * 96}")
    print(f" annualised return of the equal-weight liquid book over every "
          f"holding period in the sample")
    print(f" {'hold':>6}{'windows':>9}{'worst':>9}{'p25':>9}{'median':>9}"
          f"{'p75':>9}{'best':>9}{'% < 0':>8}{'% < cash':>10}")
    span = (ew.index[-1] - ew.index[0]).days / 365.25
    for y in (1, 3, 5, 10):
        s = describe(window_returns(ew, y))
        if not s.get("n"):
            continue
        print(f" {y:>5}y{s['n']:>9}{s['worst']:>9.2%}{s['p25']:>9.2%}"
              f"{s['median']:>9.2%}{s['p75']:>9.2%}{s['best']:>9.2%}"
              f"{s['neg']:>8.0%}{s['below_cash']:>10.0%}")
    print(f" the windows overlap heavily, so the columns above are not "
          f"independent draws:")
    print("  " + "  ".join(
        f"{y}y = {independent_windows(len(window_returns(ew, y)), y, span):.1f}"
        f" separate stretches" for y in (1, 3, 5, 10)))
    print(f" a 10-year percentage here rests on roughly one stretch of history "
          f"and should be\n read as an illustration, not as a frequency.")
    if len(bench):
        print(f"\n the same for the IHSG price index:")
        for y in (1, 3, 5, 10):
            s = describe(window_returns(bench, y))
            if not s.get("n"):
                continue
            print(f" {y:>5}y{s['n']:>9}{s['worst']:>9.2%}{s['p25']:>9.2%}"
                  f"{s['median']:>9.2%}{s['p75']:>9.2%}{s['best']:>9.2%}"
                  f"{s['neg']:>8.0%}{s['below_cash']:>10.0%}")

    # ---- 4. the shape of single-name outcomes ---------------------------
    print(f"\n{'=' * 96}\n 4. WHY BREADTH IS NOT TIMIDITY\n{'=' * 96}")
    for label, g in (("all names", gaps), ("liquid names", liquid)):
        if g.empty:
            continue
        w = wealth_concentration(g["total"], g["years"])
        if not w:
            continue
        print(f" {label} ({w['names']}):")
        print(f"   {w['losers']:.0%} lost money over the whole sample, "
              f"{w['halved']:.0%} lost at least half — dividends included")
        print(f"   the top 5% of names carry {w['top5pct']:.0%} of the "
              f"aggregate gain, the top 10% {w['top10pct']:.0%}, "
              f"the top 20% {w['top20pct']:.0%}")
        if len(bench):
            beat = float((g["total"] > cagr(bench)).mean())
            print(f"   {beat:.0%} of names beat the IHSG; a dart thrown at "
                  f"this list misses more often than it hits")

    print(f"\n{'=' * 96}\n WHAT THIS FIXES\n{'=' * 96}")
    print(" Every earlier result in this repo was measured on the price line "
          "and against a\n sample that ends inside a drawdown. Both push the "
          "same way: they understate\n what holding pays and they make any "
          "timing rule look better by comparison,\n because a rule that sits "
          "in cash is not charged for the dividends it misses.")
    print(" Nothing here is a strategy. It is the hurdle a strategy has to "
          "clear, and it is\n higher than the one that has been used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
