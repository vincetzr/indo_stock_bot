#!/usr/bin/env python3
"""Act AT the turn, not a month after it: the causal reversal filter.

Result 69 found that "above the 20-day MA" calls the direction of a weekly swing
right 94% of the time and still loses money, because it captures only 31% of each
up leg and flips 569 times inside them. The diagnosis was timing, not direction.

This is the fix, and it is the only rule shape whose lag is *bounded by
construction*. A moving average's lag depends on the window and on how the price
got there; a reversal filter's lag is exactly its threshold:

    hold long   -> sell when price closes ``exit_thr`` below the highest close
                   seen since you bought
    hold cash   -> buy  when price closes ``entry_thr`` above the lowest close
                   seen since you sold

That is the causal twin of the zigzag in ``swing_accuracy.py``. The zigzag marks
the turn with hindsight; this marks it once the market has moved far enough to
prove it, and never later than that. You give up ``entry_thr`` at the bottom and
``exit_thr`` at the top of every leg - that is the whole cost, and it is knowable
in advance, which the moving-average lag is not.

The tension the sweep resolves
------------------------------
Small thresholds capture more of each leg but fire on noise inside it. Large
thresholds ignore noise but hand back so much at both ends that avoiding a -37%
down leg is worth almost nothing. There is no reason to assume the optimum sits
anywhere in particular, so it is searched - and searched separately for the exit
and the entry, because the user's question ("can you not cut the moment there is
an indication of going down?") is exactly the question of whether the two should
differ.

Everything is measured on WEEKLY bars, because that is the chart the turns were
drawn on, and because weekly bars are what kill the 569 flips.

What is reported
----------------
Not just the return. **Capture fraction** - what share of each hindsight up leg
the rule actually banked, and what share of each down leg it still absorbed -
because Result 69 showed that is the number that compounds, and a rule can look
accurate while capturing nothing.

Honesty rails
-------------
* Signals are computed on closed weekly bars; the fill is the NEXT week's close.
  Nothing is decided with a bar the trader had not seen.
* Daily returns are capped at the +/-35% auto-rejection band before the weekly
  series is built (0.044% of IDX prints are physically impossible; uncapped they
  compounded into a fake 195x in Result 51).
* The threshold is chosen on an early slice of history and applied, untouched, to
  a later one. The in-sample table is printed too, so the shrinkage is visible.
* Survivorship is not fixed and cannot be: delisted names are absent from the
  data, so every absolute number here is optimistic.

    python3 scripts/turn_trader.py --ticker ADRO
    python3 scripts/turn_trader.py --universe
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV       # noqa: E402
from swing_accuracy import zigzag, legs        # noqa: E402

ROUND_TRIP = 0.006          # 0.15% buy + 0.25% sell + slippage
DAILY_CAP = 0.35            # auto-rejection band
MIN_TURNOVER = 5e9          # Rp5bn/day median: a name you could actually trade
MIN_WEEKS = 200             # ~4 years; fewer and the leg count is meaningless


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def clean_weekly(df: pd.DataFrame) -> Optional[pd.Series]:
    """Weekly total-return series with impossible daily prints removed.

    The cap is applied to daily returns *before* resampling, because a single
    bad print inside a week survives a weekly close otherwise.
    """
    x = df.set_index("date").sort_index()
    a = x["adj_close"].astype(float).dropna()
    if len(a) < 100:
        return None
    r = a.pct_change().clip(-DAILY_CAP, DAILY_CAP).fillna(0.0)
    clean = a.iloc[0] * (1.0 + r).cumprod()
    w = clean.resample("W-FRI").last().dropna()
    return w if len(w) >= MIN_WEEKS else None


def load_weekly(universe_only: bool = True, verbose: bool = True
                ) -> Dict[str, pd.Series]:
    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    names = sorted(set(cfg.universe("idx_all")) | set(cfg.universe("bluechip"))
                   | set(cfg.universe("lq45")) | set(cfg.universe("conglomerate")))
    raw = loader.get_many(names, max_age=86400 * 30, verbose=False)
    out: Dict[str, pd.Series] = {}
    for t, d in raw.items():
        if len(d) < 500:
            continue
        c = d["close"].astype(float)
        if float((c * d["volume"]).median()) < MIN_TURNOVER:
            continue
        w = clean_weekly(d)
        if w is not None:
            out[t] = w
    if verbose:
        span = pd.DatetimeIndex(sorted({i for s in out.values() for i in s.index}))
        print(f"universe: {len(out)} names with >={MIN_WEEKS} weekly bars and "
              f">=Rp{MIN_TURNOVER/1e9:.0f}bn/day, {span[0]:%Y-%m-%d} -> {span[-1]:%Y-%m-%d}")
    return out


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
def reversal_state(prices: np.ndarray, entry_thr: float, exit_thr: float,
                   start_long: bool = False) -> np.ndarray:
    """Causal position: 1 while long, 0 while flat. Decided on the bar's close.

    Purely a running-extreme state machine. It looks at nothing but the prices up
    to and including bar ``i``, so ``state[i]`` is knowable at bar ``i``'s close
    and can only be acted on at bar ``i+1``.
    """
    n = len(prices)
    state = np.zeros(n, dtype=np.int8)
    long = bool(start_long)
    extreme = prices[0]
    # Compare moves, not levels. ``p >= extreme * 1.1`` misses a price that is
    # exactly 10% up, because 100 * 1.1 is 110.00000000000001 in binary floating
    # point; comparing ``p / extreme - 1`` against the threshold with a relative
    # tolerance makes the boundary behave the way it is documented to.
    tol = 1e-12
    for i in range(n):
        p = prices[i]
        if long:
            if p > extreme:
                extreme = p
            elif 1.0 - p / extreme >= exit_thr - tol:
                long = False
                extreme = p
        else:
            if p < extreme:
                extreme = p
            elif p / extreme - 1.0 >= entry_thr - tol:
                long = True
                extreme = p
        state[i] = 1 if long else 0
    return state


def vol_reversal_state(prices: np.ndarray, k_entry: float, k_exit: float,
                       window: int = 52) -> np.ndarray:
    """Reversal filter whose thresholds are multiples of the name's own volatility.

    The fixed-percentage filter failed out of sample, and the obvious suspect is
    the parameter itself: 25% is an enormous move for BBCA and a quiet week for a
    coal name, and ADRO's own weekly volatility in 2008 is nothing like its 2016.
    A threshold expressed in sigmas has no scale to mis-transfer, which is the one
    structural reason a percentage grid could have failed for reasons that are
    fixable rather than fundamental.

    Sigma is the trailing standard deviation of weekly log returns over
    ``window`` bars, known at the bar it is used on. Before there are enough bars
    to measure it the filter stays flat rather than guessing.
    """
    n = len(prices)
    lr = np.zeros(n)
    lr[1:] = np.log(prices[1:] / prices[:-1])
    sig = pd.Series(lr).rolling(window, min_periods=window).std().shift(1).to_numpy()

    state = np.zeros(n, dtype=np.int8)
    long = False
    extreme = prices[0]
    for i in range(n):
        p = prices[i]
        s = sig[i]
        if not np.isfinite(s) or s <= 0:
            extreme = min(extreme, p) if not long else max(extreme, p)
            state[i] = 1 if long else 0
            continue
        # k weekly sigmas, clamped so a dead or a berserk name cannot produce a
        # threshold that is either never crossed or crossed every bar
        entry_thr = float(np.clip(k_entry * s, 0.03, 0.60))
        exit_thr = float(np.clip(k_exit * s, 0.03, 0.60))
        if long:
            if p > extreme:
                extreme = p
            elif 1.0 - p / extreme >= exit_thr:
                long = False
                extreme = p
        else:
            if p < extreme:
                extreme = p
            elif p / extreme - 1.0 >= entry_thr:
                long = True
                extreme = p
        state[i] = 1 if long else 0
    return state


def run(prices: np.ndarray, state: np.ndarray, cost: float = ROUND_TRIP
        ) -> Tuple[np.ndarray, int]:
    """Equity from holding ``state`` with a one-bar delay. Returns curve + trades.

    ``state[i]`` is decided at bar i's close, so it governs the return from bar
    i+1 to i+2. Charging the fee on the bar the position changes means the fee is
    paid at the fill, not at the signal.
    """
    n = len(prices)
    ret = np.zeros(n)
    ret[1:] = prices[1:] / prices[:-1] - 1.0
    held = np.zeros(n, dtype=np.int8)
    held[2:] = state[:-2]                       # signal at i-2 -> return i-1..i
    eq = np.ones(n)
    trades = 0
    for i in range(1, n):
        eq[i] = eq[i - 1] * (1.0 + ret[i] * held[i])
        if held[i] != held[i - 1]:
            eq[i] *= (1.0 - cost / 2.0)         # half the round trip per side
            trades += 1
    return eq, trades


# --------------------------------------------------------------------------- #
# diagnostics: capture fraction against the hindsight legs
# --------------------------------------------------------------------------- #
def capture(prices: np.ndarray, state: np.ndarray, leg_list, cost: float = ROUND_TRIP
            ) -> Dict[str, float]:
    """What share of each hindsight leg did the rule actually bank?

    This is the number Result 69 showed matters. Direction accuracy asks "were
    you on the right side"; this asks "how much of the move did you get", which
    is what compounds.
    """
    eq, _ = run(prices, state, cost)
    up_move, up_got, dn_move, dn_got, flips = [], [], [], [], 0
    right = 0
    for a, b, r in leg_list:
        got = eq[b] / eq[a] - 1.0
        flips += int(np.abs(np.diff(state[a:b + 1])).sum()) if b > a else 0
        if r > 0:
            up_move.append(r)
            up_got.append(got)
            right += int(state[a:b].mean() > 0.5)
        else:
            dn_move.append(r)
            dn_got.append(got)
            right += int(state[a:b].mean() <= 0.5)
    return {
        "legs": len(leg_list),
        "direction_acc": right / max(len(leg_list), 1),
        "up_mean": float(np.mean(up_move)) if up_move else np.nan,
        "up_captured": float(np.mean(up_got)) if up_got else np.nan,
        "up_fraction": (float(np.mean(up_got) / np.mean(up_move))
                        if up_move and np.mean(up_move) else np.nan),
        "dn_mean": float(np.mean(dn_move)) if dn_move else np.nan,
        "dn_absorbed": float(np.mean(dn_got)) if dn_got else np.nan,
        "flips": int(flips),
    }


def score(prices: np.ndarray, index: pd.DatetimeIndex, entry_thr: float,
          exit_thr: float) -> Dict[str, float]:
    state = reversal_state(prices, entry_thr, exit_thr)
    eq, trades = run(prices, state)
    years = (index[-1] - index[0]).days / 365.25
    bh = prices[-1] / prices[0]
    growth = float(eq[-1])
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return {
        "entry": entry_thr, "exit": exit_thr,
        "growth": growth,
        "cagr": growth ** (1 / years) - 1 if years > 0 else np.nan,
        "bh_growth": float(bh),
        "bh_cagr": bh ** (1 / years) - 1 if years > 0 else np.nan,
        "excess": (growth ** (1 / years) - bh ** (1 / years)) if years > 0 else np.nan,
        "trades": int(trades),
        "time_in": float(state.mean()),
        "max_dd": float(dd.min()),
        "years": years,
    }


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #
GRID = (0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)
VOL_GRID = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)       # thresholds in weekly sigmas


def single(ticker: str, w: pd.Series, threshold: float) -> pd.DataFrame:
    px = w.to_numpy(float)
    years = (w.index[-1] - w.index[0]).days / 365.25
    piv = zigzag(px, threshold)
    lg = legs(px, piv)

    print("=" * 96)
    print(f" {ticker} WEEKLY — acting AT the turn")
    print("=" * 96)
    print(f" {len(w)} weekly bars, {w.index[0]:%Y-%m-%d} -> {w.index[-1]:%Y-%m-%d} "
          f"({years:.1f} years)")
    print(f" buy & hold {px[-1]/px[0]:.1f}x ({(px[-1]/px[0])**(1/years)-1:+.1%}/yr); "
          f"{len(lg)} hindsight legs at a {threshold:.0%} threshold")

    rows = []
    for e in GRID:
        for x in GRID:
            s = score(px, w.index, e, x)
            st = reversal_state(px, e, x)
            s.update(capture(px, st, lg))
            rows.append(s)
    R = pd.DataFrame(rows)

    print(f"\n{'-' * 96}\n SYMMETRIC THRESHOLDS — what the filter earns, and how much of each leg it gets"
          f"\n{'-' * 96}")
    print(f" {'thr':>6}{'growth':>12}{'CAGR':>9}{'vs B&H':>9}{'trades':>8}"
          f"{'in mkt':>8}{'maxDD':>8}{'up captured':>14}{'dn absorbed':>13}{'flips':>7}")
    for t in GRID:
        r = R[(R["entry"] == t) & (R["exit"] == t)].iloc[0]
        print(f" {t:>6.0%}{r['growth']:>11,.1f}x{r['cagr']:>9.1%}{r['excess']:>+9.1%}"
              f"{r['trades']:>8.0f}{r['time_in']:>8.0%}{r['max_dd']:>8.0%}"
              f"{r['up_fraction']:>13.0%}  {r['dn_absorbed']:>11.1%}{r['flips']:>7.0f}")

    best = R.sort_values("cagr", ascending=False).iloc[0]
    print(f"\n{'-' * 96}\n ASYMMETRIC — is it right to be slower to cut than to buy?"
          f"\n{'-' * 96}")
    print(f" best of {len(R)}: buy {best['entry']:.0%} above the low, "
          f"sell {best['exit']:.0%} below the high "
          f"-> {best['growth']:,.1f}x ({best['cagr']:+.1%}/yr)")
    piv_tab = R.pivot(index="exit", columns="entry", values="cagr")
    print("\n CAGR by (rows: sell threshold, cols: buy threshold)")
    print("  " + "".join(f"{c:>9.0%}" for c in piv_tab.columns))
    for idx, row in piv_tab.iterrows():
        print(f" {idx:>4.0%}" + "".join(f"{v:>9.1%}" for v in row))

    R.to_csv(f"reports/turn_trader_{ticker}.csv", index=False)
    print(f"\n -> reports/turn_trader_{ticker}.csv")
    return R


def universe(weeks: Dict[str, pd.Series], split: str, threshold: float) -> pd.DataFrame:
    """Choose the threshold before ``split``, then apply it after, untouched."""
    cut = pd.Timestamp(split)
    tr = {t: s[s.index < cut] for t, s in weeks.items()}
    te = {t: s[s.index >= cut] for t, s in weeks.items()}
    tr = {t: s for t, s in tr.items() if len(s) >= MIN_WEEKS}
    te = {t: s for t, s in te.items() if len(s) >= MIN_WEEKS}
    print(f"\n train: {len(tr)} names before {split};  test: {len(te)} names after")

    def sweep_vol(book: Dict[str, pd.Series], label: str) -> pd.DataFrame:
        """Same walk, thresholds in sigmas instead of percent."""
        rows = []
        for ke in VOL_GRID:
            for kx in VOL_GRID:
                ex, beat = [], []
                for s in book.values():
                    px = s.to_numpy(float)
                    st = vol_reversal_state(px, ke, kx)
                    eq, _ = run(px, st)
                    yrs = (s.index[-1] - s.index[0]).days / 365.25
                    if yrs <= 0:
                        continue
                    c = float(eq[-1]) ** (1 / yrs) - 1
                    b = (px[-1] / px[0]) ** (1 / yrs) - 1
                    ex.append(c - b)
                    beat.append(c > b)
                rows.append({"slice": label, "k_entry": ke, "k_exit": kx,
                             "median_excess": float(np.median(ex)),
                             "pct_beat": float(np.mean(beat)), "n": len(ex)})
        return pd.DataFrame(rows)

    def sweep(book: Dict[str, pd.Series], label: str) -> pd.DataFrame:
        # The hindsight legs depend only on the price series, so they are built
        # once per name rather than once per threshold pair.
        prepared = [(s.to_numpy(float), s.index,
                     legs(s.to_numpy(float), zigzag(s.to_numpy(float), threshold)))
                    for s in book.values()]
        rows = []
        for e in GRID:
            for x in GRID:
                ex, beat, cap_up, cap_dn = [], [], [], []
                for px, idx, lg in prepared:
                    sc = score(px, idx, e, x)
                    ex.append(sc["excess"])
                    beat.append(sc["cagr"] > sc["bh_cagr"])
                    if lg:
                        c = capture(px, reversal_state(px, e, x), lg)
                        if not np.isnan(c["up_fraction"]):
                            cap_up.append(c["up_fraction"])
                        if not np.isnan(c["dn_absorbed"]):
                            cap_dn.append(c["dn_absorbed"])
                rows.append({"slice": label, "entry": e, "exit": x,
                             "median_excess": float(np.median(ex)),
                             "mean_excess": float(np.mean(ex)),
                             "pct_beat": float(np.mean(beat)),
                             "up_fraction": float(np.median(cap_up)) if cap_up else np.nan,
                             "dn_absorbed": float(np.median(cap_dn)) if cap_dn else np.nan,
                             "n": len(book)})
        return pd.DataFrame(rows)

    T = sweep(tr, "train")
    best = T.sort_values("median_excess", ascending=False).iloc[0]
    print(f"\n{'=' * 96}\n IN SAMPLE (before {split}) — top 8 of {len(T)} threshold pairs"
          f"\n{'=' * 96}")
    print(f" {'buy':>6}{'sell':>7}{'median excess':>16}{'mean excess':>14}"
          f"{'% beating B&H':>16}{'up captured':>13}{'dn absorbed':>13}")
    for _, r in T.sort_values("median_excess", ascending=False).head(8).iterrows():
        print(f" {r['entry']:>6.0%}{r['exit']:>7.0%}{r['median_excess']:>+16.2%}"
              f"{r['mean_excess']:>+14.2%}{r['pct_beat']:>16.0%}"
              f"{r['up_fraction']:>13.0%}{r['dn_absorbed']:>13.1%}")

    S = sweep(te, "test")
    chosen = S[(S["entry"] == best["entry"]) & (S["exit"] == best["exit"])].iloc[0]
    print(f"\n{'=' * 96}\n OUT OF SAMPLE (from {split}) — the pair chosen above, applied untouched"
          f"\n{'=' * 96}")
    print(f" buy {chosen['entry']:.0%} off the low / sell {chosen['exit']:.0%} off the high:")
    print(f"   median excess over buy & hold {chosen['median_excess']:+.2%}/yr,"
          f" mean {chosen['mean_excess']:+.2%}/yr,"
          f" beats B&H on {chosen['pct_beat']:.0%} of {int(chosen['n'])} names")
    print(f"   captures {chosen['up_fraction']:.0%} of the median up leg, "
          f"still absorbs {chosen['dn_absorbed']:.1%} of the median down leg")
    best_oos = S.sort_values("median_excess", ascending=False).iloc[0]
    print(f"\n for reference the BEST pair out of sample was "
          f"buy {best_oos['entry']:.0%} / sell {best_oos['exit']:.0%} at "
          f"{best_oos['median_excess']:+.2%}/yr — the gap to it is the tuning premium "
          f"you do not get to keep.")

    # Where does the loss come from? Split the out-of-sample names by whether the
    # stock itself went up, because a timing rule is supposed to earn its keep in
    # the names that fell.
    e, x = float(best["entry"]), float(best["exit"])
    det = []
    for t, s in te.items():
        px = s.to_numpy(float)
        sc = score(px, s.index, e, x)
        det.append({"ticker": t, "excess": sc["excess"], "cagr": sc["cagr"],
                    "bh_cagr": sc["bh_cagr"], "time_in": sc["time_in"],
                    "trades": sc["trades"], "max_dd": sc["max_dd"],
                    "bh_max_dd": float((lambda c: (c / np.maximum.accumulate(c) - 1).min())(px))})
    D = pd.DataFrame(det)
    winners, losers = D[D["bh_cagr"] > 0], D[D["bh_cagr"] <= 0]
    print(f"\n{'-' * 96}\n WHERE THE OUT-OF-SAMPLE RESULT COMES FROM\n{'-' * 96}")
    print(f" {'group':<28}{'n':>5}{'median B&H':>13}{'median filtered':>18}"
          f"{'median excess':>16}{'% beat':>9}")
    for label, g in (("stock rose (B&H > 0)", winners), ("stock fell (B&H <= 0)", losers)):
        if len(g):
            print(f" {label:<28}{len(g):>5}{g['bh_cagr'].median():>+13.1%}"
                  f"{g['cagr'].median():>+18.1%}{g['excess'].median():>+16.2%}"
                  f"{(g['excess'] > 0).mean():>9.0%}")
    print(f"\n median drawdown: buy & hold {D['bh_max_dd'].median():.0%}, "
          f"filtered {D['max_dd'].median():.0%};  median time in market "
          f"{D['time_in'].median():.0%};  median {D['trades'].median():.0f} trades")
    print("\n best and worst names out of sample")
    for _, r in D.sort_values("excess", ascending=False).head(5).iterrows():
        print(f"   {r['ticker']:<8}{r['excess']:>+8.1%}/yr   "
              f"(B&H {r['bh_cagr']:+.1%} -> {r['cagr']:+.1%})")
    for _, r in D.sort_values("excess").head(5).iterrows():
        print(f"   {r['ticker']:<8}{r['excess']:>+8.1%}/yr   "
              f"(B&H {r['bh_cagr']:+.1%} -> {r['cagr']:+.1%})")
    D.to_csv("reports/turn_trader_oos_names.csv", index=False)

    # --- does expressing the threshold in sigmas rescue it? --- #
    VT, VS = sweep_vol(tr, "train"), sweep_vol(te, "test")
    vbest = VT.sort_values("median_excess", ascending=False).iloc[0]
    vchosen = VS[(VS["k_entry"] == vbest["k_entry"])
                 & (VS["k_exit"] == vbest["k_exit"])].iloc[0]
    vtop = VS.sort_values("median_excess", ascending=False).iloc[0]
    print(f"\n{'-' * 96}\n VOLATILITY-SCALED THRESHOLDS — the same walk, in sigmas"
          f"\n{'-' * 96}")
    print(f" {'':30}{'buy':>6}{'sell':>7}{'median excess':>16}{'% beating B&H':>16}")
    print(f" {'best in sample':<30}{vbest['k_entry']:>5.1f}s{vbest['k_exit']:>6.1f}s"
          f"{vbest['median_excess']:>+16.2%}{vbest['pct_beat']:>16.0%}")
    print(f" {'  applied out of sample':<30}{vchosen['k_entry']:>5.1f}s"
          f"{vchosen['k_exit']:>6.1f}s{vchosen['median_excess']:>+16.2%}"
          f"{vchosen['pct_beat']:>16.0%}")
    print(f" {'best out of sample (ceiling)':<30}{vtop['k_entry']:>5.1f}s"
          f"{vtop['k_exit']:>6.1f}s{vtop['median_excess']:>+16.2%}"
          f"{vtop['pct_beat']:>16.0%}")
    pd.concat([VT, VS], ignore_index=True).to_csv(
        "reports/turn_trader_vol.csv", index=False)

    out = pd.concat([T, S], ignore_index=True)
    out.to_csv("reports/turn_trader_universe.csv", index=False)
    print("\n -> reports/turn_trader_universe.csv")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="ADRO")
    ap.add_argument("--threshold", type=float, default=0.20,
                    help="zigzag size used to define the hindsight legs")
    ap.add_argument("--universe", action="store_true")
    ap.add_argument("--split", default="2013-01-01")
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    if args.universe:
        weeks = load_weekly()
        universe(weeks, args.split, args.threshold)
        return 0

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    d = loader.get(args.ticker, max_age=86400 * 30)
    if d is None or d.empty:
        raise SystemExit(f"no data for {args.ticker}")
    w = clean_weekly(d)
    if w is None:
        raise SystemExit(f"{args.ticker}: not enough weekly history")
    single(args.ticker, w, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
