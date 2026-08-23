#!/usr/bin/env python3
"""§12 — is a losing cohort the SAME cohort next period?

THE QUESTION, AND WHY IT IS THE ONE THAT MATTERS
--------------------------------------------------
Phase 2b established that the median cohort round trip loses ~25-32 bps and
that a shuffled label loses about the same. §12 argues that is not yet
interesting:

    "That retail cohorts lose persistently while institutions gain is among the
    most robust results in the market-microstructure literature ... and it is
    durable precisely because the losing cohort continuously regenerates."

The operative word is PERSISTENTLY. A cohort that lost last year is only
tradeable against if it is the same cohort losing this year. One period's
ranking is a snapshot; persistence is the claim, and it is the claim Phase 2b
left untested.

TWO ESTIMATORS, BECAUSE THE OBVIOUS ONE IS UNDER-POWERED
----------------------------------------------------------
    SPLIT-HALF. Rank brokers by margin in the first half of the sample, rank
    them again in the second, correlate. Simple, but it gives each broker
    exactly TWO observations, so a near-zero result from it alone would be
    weak evidence rather than a measurement.

    PERIOD-OVER-PERIOD. Rank brokers within each calendar year (Track B) or
    quarter (Track A) and correlate adjacent periods. Many pairs instead of
    one, so the spread across pairs shows whether any persistence is stable or
    episodic.

Both are reported and both go through the same permutation null.

TWO MEASURES, BECAUSE THE TWO STORES SUPPORT DIFFERENT THINGS
--------------------------------------------------------------
    TRACK A, daily, 9 names, 18 months. Round-trip margin_bps — the clean §9.3
    estimate. Short, so sample size is the binding constraint, not the
    statistic.

    TRACK B, fortnightly, 221 names, 12.6 years. Round trips are NOT computable
    at this resolution, so the measure is different and is named differently:

        timing_pnl = net_shares_bought x (close at end of NEXT window
                                          - close at end of THIS window)

    This asks whether a broker's net DIRECTION anticipated the move that
    followed. It is not §9.3's realised cohort P&L and must never be reported
    as such — it needs no inventory, no cost basis and no path, which is
    exactly why a fortnightly panel can carry it, and also why it answers a
    narrower question.

WHAT WOULD MAKE THIS A FALSE POSITIVE, AND WHAT CONTROLS IT
-------------------------------------------------------------
    A TICKER EFFECT. A broker that trades one drifting name heavily in both
    halves shows "persistence" that is really the name's drift. The null
    shuffles broker labels within each (ticker, window), preserving every
    ticker's flow and every broker's size distribution while destroying which
    code owned which row. A ticker effect survives that shuffle; a genuine
    broker effect does not.

    THE TOP-10 CENSOR. The source ranks buyers and sellers independently and
    publishes ten of each, so a code appearing only among buyers has its sell
    side recorded as zero when it is really an unknown lower bound, and its net
    is biased long by construction. The TWO-SIDED subsample keeps only rows
    where the code printed on both sides. It is smaller and it is the honest
    one.

    ONE NULL DRAW. H9 in this repo was nearly reported with a broken null that
    a single draw could not have exposed, and a later single draw was briefly
    over-read as systematic bias before a second seed contradicted it. So the
    null here is a DISTRIBUTION of 200 draws through the identical pipeline —
    guards included — and the observed statistic is quoted as a position within
    it, never against one shuffle.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.spine.cohort_pnl import LOT, round_trips, walk_forward  # noqa: E402
from idxbot.spine.persistence import (MIN_BROKERS,                  # noqa: E402
                                      permutation_test)
from flow_panel_build import load_prices                            # noqa: E402
from flow_panel_collect import load_panel                           # noqa: E402
from cohort_pnl_run import MIN_SESSIONS, closes, load_daily         # noqa: E402

STORE = os.path.join("data", "cache", "ipot_broker")

#: Rebuilding the per-broker frame means reading 36,000 gzip files, so the
#: assembled version is cached. It is a pure function of the range store and is
#: rebuilt with --rebuild whenever that store grows.
CACHE = os.path.join("data", "spine", "broker_windows.csv.gz")

#: A broker needs this many windows in EACH half before its margin is ranked.
#: Below it the half-sample estimate is one or two windows of noise.
MIN_WINDOWS_PER_HALF = 20

#: Same idea inside a single calendar year, which holds ~26 windows.
MIN_WINDOWS_PER_YEAR = 8

#: Rp 1bn, about USD 60k, so a code that printed once is not ranked alongside
#: one that traded throughout.
MIN_GROSS = 1e9

#: Consecutive fortnights are 14 days apart. A wider gap means a window is
#: missing, and "the next window's move" would silently span a month or more.
MAX_WINDOW_GAP_DAYS = 21

#: The panel collection starts in 2014. Before it the store holds only
#: exploratory probes — 454 rows total across 2010-2013, against 378,689 from
#: 2014 on — and 2013 puts just 13 brokers through the year guard against 82 in
#: 2014. A rank correlation on 13 brokers from 15 windows is noise with a
#: decimal point, and on the first run it produced the sample's largest year
#: pair (+0.63) and carried the whole headline. Both the floored and unfloored
#: statistics are reported so the effect of this line is visible rather than
#: quietly applied.
PANEL_START_YEAR = 2014

DRAWS = 200


# ---------------------------------------------------------------------------
# Track B — the long history
# ---------------------------------------------------------------------------
def load_broker_windows(store: str = STORE) -> pd.DataFrame:
    """Per (broker, ticker, window) rows from the collected range files.

    The flow panel collapses these to ticker level, which is right for §7 and
    useless here: §12's question is about brokers, so it has to be rebuilt from
    the per-broker rows the range files actually contain.
    """
    rows = []
    cols = ["broker", "buy_lot", "buy_val", "sell_lot", "sell_val"]
    for p in sorted(glob.glob(os.path.join(store, "*_RG_range.csv.gz"))):
        m = re.match(r"([A-Z]{4})_(\d{8})_(\d{8})_RG_range",
                     os.path.basename(p))
        if not m:
            continue
        try:
            d = pd.read_csv(p, usecols=cols)
        except Exception:                                       # noqa: BLE001
            continue
        if d.empty:
            continue
        d["ticker"] = m.group(1)
        d["window_end"] = pd.Timestamp(m.group(3))
        rows.append(d)
    if not rows:
        return pd.DataFrame(columns=cols + ["ticker", "window_end"])
    D = pd.concat(rows, ignore_index=True)
    for c in cols[1:]:
        D[c] = pd.to_numeric(D[c], errors="coerce").fillna(0.0)
    return D


def forward_moves(D: pd.DataFrame, src: Dict[str, str]) -> Dict[tuple, float]:
    """close(next window end) − close(this window end) per (ticker, window).

    No look-ahead: the flow is known at this window's close and the payoff is
    measured strictly after it. Pairs separated by more than one fortnight are
    dropped rather than stretched, because a missing window would otherwise
    turn "the next fortnight" into "the next quarter" without saying so — the
    store carries older sparse probes for 45 large caps where consecutive
    windows sit months apart.
    """
    fwd: Dict[tuple, float] = {}
    for t in sorted(D["ticker"].unique()):
        px = load_prices(t, src.get(t, "live"))
        if px is None or px.empty:
            continue
        c = pd.to_numeric(px["close"], errors="coerce")
        idx = px.index
        wins = sorted(pd.to_datetime(
            D.loc[D.ticker == t, "window_end"].unique()))
        for a, b in zip(wins, wins[1:]):
            if (b - a).days > MAX_WINDOW_GAP_DAYS:
                continue
            ia, ib = idx[idx <= a], idx[idx <= b]
            if not len(ia) or not len(ib) or ia[-1] == ib[-1]:
                continue
            pa, pb = float(c.loc[ia[-1]]), float(c.loc[ib[-1]])
            if np.isfinite(pa) and np.isfinite(pb) and pa > 0:
                fwd[(t, a)] = pb - pa
    return fwd


def timing_pnl(D: pd.DataFrame, fwd: Dict[tuple, float]) -> pd.DataFrame:
    """Per (broker, ticker, window): did the net direction anticipate the move?

    Deliberately NOT called cohort P&L. It needs no inventory, no cost basis
    and no intra-window path, which is what lets a fortnightly panel carry it.
    """
    D = D.copy()
    D["px_move"] = [fwd.get((t, w), np.nan)
                    for t, w in zip(D["ticker"], D["window_end"])]
    D["net_sh"] = (D["buy_lot"] - D["sell_lot"]) * LOT
    D["timing_pnl"] = D["net_sh"] * D["px_move"]
    D["gross_value"] = D["buy_val"] + D["sell_val"]
    D["two_sided"] = (D["buy_lot"] > 0) & (D["sell_lot"] > 0)
    return D.dropna(subset=["timing_pnl"])


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------
def codes(D: pd.DataFrame, when: str, period: pd.Series
          ) -> Tuple[np.ndarray, ...]:
    """Integer codes so the permutation loop never touches pandas."""
    b, _ = pd.factorize(D["broker"], sort=True)
    p, pu = pd.factorize(period, sort=True)
    w, wu = pd.factorize(D[when], sort=True)
    grp, _ = pd.factorize(
        D["ticker"].astype(str) + "|" + D[when].astype(str), sort=True)
    return (grp, b.astype(np.int64), p.astype(np.int64), w.astype(np.int64),
            len(np.unique(b)), len(pu), len(wu))


def run(D: pd.DataFrame, value: str, gross: str, when: str,
        period: pd.Series, min_windows: int, label: str, seed: int,
        draws: int = DRAWS) -> None:
    grp, b, p, w, nb, npd, nw = codes(D, when, period)
    obs, pairs, nulls, pval = permutation_test(
        grp, b, p, w,
        D[value].to_numpy(dtype=float), D[gross].to_numpy(dtype=float),
        nb, npd, nw, min_windows, MIN_GROSS, draws=draws, seed=seed)

    if not np.isfinite(obs):
        print(f"    {label:<24} no period pair clears the guards "
              f"(needs {MIN_BROKERS} brokers in both)")
        return
    v = nulls[np.isfinite(nulls)]
    lo, hi = (np.percentile(v, [2.5, 97.5]) if len(v) > 20 else (np.nan,) * 2)
    extra = ""
    if len(pairs) > 1:
        extra = (f"   pairs {len(pairs)}  range "
                 f"[{pairs.min():+.2f}, {pairs.max():+.2f}]  "
                 f"{int((pairs > 0).sum())}/{len(pairs)} positive")
    print(f"    {label:<24} rank corr {obs:+.3f}{extra}")
    print(f"      {'':22} null {np.nanmean(v):+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}] over {len(v)} draws   "
          f"one-sided p {pval:.3f}")


def build_track_b(rebuild: bool = False) -> pd.DataFrame:
    """The (broker, ticker, window) timing frame, cached."""
    if os.path.exists(CACHE) and not rebuild:
        return pd.read_csv(CACHE, parse_dates=["window_end"])
    D = load_broker_windows()
    if D.empty:
        return D
    P = load_panel()
    T = timing_pnl(D, forward_moves(D, dict(zip(P["ticker"], P["src"]))))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    # Written via a temp file and renamed: a reader that arrives mid-write
    # otherwise gets "Compressed file ended before the end-of-stream marker",
    # which looks like a corrupt store rather than a race.
    tmp = CACHE + ".tmp"
    T.to_csv(tmp, index=False)
    os.replace(tmp, CACHE)
    return T


def track_b(seed: int, draws: int, rebuild: bool = False) -> None:
    print("=" * 78)
    print(" TRACK B — fortnightly, timing_pnl, 2014-2026")
    print(" (net direction vs the NEXT window's move; NOT §9.3 realised P&L)")
    print("=" * 78)
    T = build_track_b(rebuild)
    if T.empty:
        print("   no window pairs with prices\n")
        return
    print(f"   {T.broker.nunique()} codes, {T.ticker.nunique()} tickers, "
          f"{T.window_end.nunique()} windows, {len(T):,} broker-windows, "
          f"{T.window_end.min().date()} … {T.window_end.max().date()}")
    print(f"   two-sided rows {T.two_sided.mean():.1%} — the rest have one "
          f"side censored by the top-10 cut\n")

    F = T[T["window_end"].dt.year >= PANEL_START_YEAR]
    print(f"   the {len(T) - len(F)} rows before {PANEL_START_YEAR} are "
          f"exploratory probes, not the panel — reported both ways\n")
    for era, base in ((f"{PANEL_START_YEAR}+", F), ("incl. pre-panel", T)):
        for name, S in (("ALL ROWS", base),
                        ("TWO-SIDED ONLY", base[base.two_sided])):
            print(f"   {era:<16} {name}  ({len(S):,} rows)")
            half = pd.Series(
                np.where(S["window_end"] <= S["window_end"].median(), 0, 1),
                index=S.index)
            run(S, "timing_pnl", "gross_value", "window_end", half,
                MIN_WINDOWS_PER_HALF, "split-half", seed, draws)
            run(S, "timing_pnl", "gross_value", "window_end",
                S["window_end"].dt.year, MIN_WINDOWS_PER_YEAR,
                "year-over-year", seed + 1, draws)
            print()


def track_a(seed: int, draws: int) -> None:
    print("=" * 78)
    print(" TRACK A — daily, round-trip margin_bps, 2025-02 … 2026-08")
    print("=" * 78)
    Dd = load_daily()
    if Dd.empty:
        print("   no daily store")
        return
    px = closes(sorted(Dd["ticker"].unique()))
    rows: List[Dict] = []
    for (t, b), g in Dd.groupby(["ticker", "broker"]):
        if len(g) < MIN_SESSIONS:
            continue
        g = g.sort_values("date")
        c = px.get(t)
        cs = (c.reindex(pd.to_datetime(g["date"])).ffill().to_numpy()
              if c is not None else None)
        rt = round_trips(walk_forward(
            g, pd.Series(cs) if cs is not None else None))
        for _, e in rt.iterrows():
            rows.append({"broker": b, "ticker": t,
                         "window_end": pd.Timestamp(e["end"]),
                         "pnl": float(e["pnl"]),
                         "gross_value": float(e["gross_value"])})
    E = pd.DataFrame(rows)
    if E.empty:
        print("   no episodes")
        return
    span = (E.window_end.max() - E.window_end.min()).days / 365.25
    print(f"   {E.broker.nunique()} codes, {E.ticker.nunique()} tickers, "
          f"{len(E):,} round-trip episodes over {span:.1f} years")
    print("   Episodes, not windows — so the guards count episodes and are")
    print("   set far lower than Track B's. This track is sample-limited and")
    print("   the numbers below should be read as such.\n")
    half = pd.Series(np.where(E["window_end"] <= E["window_end"].median(),
                              0, 1), index=E.index)
    run(E, "pnl", "gross_value", "window_end", half, 3, "split-half",
        seed, draws)
    q = E["window_end"].dt.year * 4 + E["window_end"].dt.quarter
    run(E, "pnl", "gross_value", "window_end", q, 2, "quarter-over-quarter",
        seed + 1, draws)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--rebuild", action="store_true",
                    help="re-read the 36,000 range files instead of the cache")
    a = ap.parse_args()

    print("=" * 78)
    print(" §12 PERSISTENCE — is the losing cohort the SAME cohort next period?")
    print("=" * 78)
    print(" Rank brokers by margin in one period, rank them again in the next,")
    print(" correlate. §12 predicts POSITIVE persistence, so the p-value is")
    print(" one-sided upward. Near zero means last period's ranking does not")
    print(" predict the next, and there is nothing stable to take the other")
    print(" side of.\n")

    track_b(a.seed, a.draws, a.rebuild)
    track_a(a.seed, a.draws)

    print("\n A rank correlation inside its own null is a RESULT, not a failure")
    print(" to find one: it says the identity of last period's losers does not")
    print(" carry over at this resolution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
