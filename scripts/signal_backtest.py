#!/usr/bin/env python3
"""H42 — replay the daily scan through history: does the target actually get hit?

    python3 scripts/signal_backtest.py

THE QUESTION, AND WHY IT HAS NEVER BEEN ASKED.

`scripts/daily_signal.py` prints a target, a stop, and `P(target first)` for
every green-ribbon name. Every one of those probabilities comes from a FITTED
LAW (H32's first-passage laws and H36's race law), not from any measurement of
the list itself. H36 calibrated the laws on all bars; nobody has ever asked
whether the rows the SCANNER SELECTS behave the way it says they will, or
whether its `EV` column — the one it ranks on and the one a reader would act on
— predicts anything at all.

This replays the identical scan at every historical bar and walks each signal
forward 252 sessions.

WHY THIS IS A LEGITIMATE REPLAY AND NOT A LOOK-AHEAD.
Every input the scanner uses is causal, which is what makes a full-series
computation identical to an as-of-t one:
  * `states` — EMA and HMA, causal by construction
  * `flip_price` — solved from WMAs of bars <= t
  * `swing_levels` — pivots gated on their CONFIRMATION bar, never the pivot
    bar. That lag is the whole point; a ZigZag drawn at the pivot uses
    information that did not exist until the reversal.
  * vol60, turnover — trailing rolling windows
Verified by `tests/test_signal_backtest.py`, which re-runs the live scanner at
a past date against a truncated panel and demands the same row.

WHAT IS STILL IN-SAMPLE, STATED BEFORE THE NUMBERS.
The probability laws were fitted on 2000-2024, which is this sample. So the
CALIBRATION arm is in-sample by construction and can only ever be a consistency
check. What is NOT contaminated is the realised hit rate and the realised
return: those are facts about the price path, and they are what the question
actually asks.

EXIT ACCOUNTING — A27's THREE CONTROLS, ALL OF THEM.
  * Fill at the ACTUAL CLOSE of the exit bar, never the nominal level. A bar
    that breaches a stop often closes well below it, and on IDX can gap to ARB
    untradeable. This alone took H35's best bracket from +0.0173 to +0.0050 and
    it flatters exactly the tight stops that win a naive grid.
  * ANNUALISE, because a bracket that closes in 40 sessions is out of the
    market for 212 and hands the capital back to be redeployed. Note that a
    SAME-DURATION hold is NOT a usable benchmark here: the bracket exits at the
    close of its exit bar and so does a hold of that many bars, so the two are
    identically equal by construction. That is H39's "beat buy-and-hold on 0.0%
    of trades" waiting to happen, and the benchmarks used instead are a
    full-horizon hold and the annualised bracket.
  * Cost every round trip at the broker schedule plus a fraksi-harga half
    spread each way.

PRE-REGISTERED, WRITTEN BEFORE ANY CELL WAS SCORED
--------------------------------------------------
B1  The race law is CALIBRATED but has NO DISCRIMINATION. H36 measured it
    calibrated to within 0.3 points on average with AUC 0.51, so aggregate
    predicted P(target first) should land near the realised rate while the
    calibration curve is nearly flat across predicted deciles. If instead the
    aggregate is badly off, the scanner is quoting a broken number.
B2  "Reliably hits the target" is FALSE. For the reward-to-risk ratios the
    list actually carries, the realised rate at which the target arrives first
    will be well under half.
B3  THE `EV` COLUMN WILL NOT PREDICT REALISED RETURN. H35 found 0 of 30
    brackets positive in both halves and H38 found 0 of 25 placements; EV is
    built from an undiscriminating race law, so sorting on it should sort
    mostly by the two distances, which are already measured as not tradeable.
    This is the closest thing here to a PREDICTED-NULL control (A9): if EV
    comes back strongly predictive, suspect the harness before believing it.
B4  The top-N-by-EV basket will not beat the same bracket applied to RANDOM
    eligible names on the same dates. This is H41's control, which is the only
    one that has ever changed the answer in this repo.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.cone import p_target_first, p_touch, vol_decile      # noqa: E402
from daily_signal import MIN_TV                                  # noqa: E402
from hull_stop import flip_price, swing_levels                   # noqa: E402
from hull_trade import COST, states                              # noqa: E402
from paint_suite import tick_of                                  # noqa: E402
from time_price import MIN_BARS, load                            # noqa: E402

OUT = "reports"
HORIZON = 252
MIN_UP = 0.05                 # the scanner's target floor
STOP_LO, STOP_HI = 0.02, 0.95


def signal_rows(P: pd.DataFrame, min_rr: float = 0.0) -> pd.DataFrame:
    """Every (ticker, bar) the live scanner would have emitted, with its
    quoted target, stop, probabilities and expectancy — and the realised
    outcome over the next 252 sessions.

    `min_rr` is the live scanner's reward-to-risk gate and DEFAULTS TO ZERO
    here, because the study has to measure the cells the gate rejects in order
    to justify rejecting them. A backtest that inherits the filter it is
    supposed to be evaluating can only ever confirm it. The equivalence test
    against the live scanner passes `MIN_RR` explicitly.
    """
    frames: List[pd.DataFrame] = []
    for tk, g in P.groupby("ticker", sort=False):
        if len(g) < MIN_BARS + 60:
            continue
        px = g["adj_close"].reset_index(drop=True)
        p = px.to_numpy(float)
        raw = g["close"].to_numpy(float)
        up, on = states(px, 55, "EMA stack")
        sig = px.pct_change().rolling(60, min_periods=60).std(ddof=1).to_numpy()
        rp = (np.exp(g["log_turnover"].to_numpy(float))
              if "log_turnover" in g else np.full(len(g), np.nan))
        rp = pd.Series(rp).rolling(20, min_periods=5).median().to_numpy()
        res, _ = swing_levels(p)
        fp = flip_price(px, 55)
        #  AGE — sessions since the ribbon turned green. Cheap running counter;
        #  the live scanner walks backwards from the last bar for the same
        #  quantity, and the test asserts the two agree.
        age = np.zeros(len(p), dtype=np.int64)
        for i in range(1, len(p)):
            age[i] = age[i - 1] + 1 if (up[i] and up[i - 1]) else 0

        ok = (up & np.isfinite(sig) & (sig > 0) & (rp >= MIN_TV)
              & np.isfinite(res) & np.isfinite(fp) & (fp > 0) & (fp < p)
              & np.isfinite(raw) & (raw > 0))
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        d_up = res[idx] / p[idx] - 1.0
        d_dn = 1.0 - fp[idx] / p[idx]
        keep = ((d_up >= MIN_UP) & (d_dn > STOP_LO) & (d_dn < STOP_HI)
                & (d_up / np.maximum(d_dn, 1e-12) >= min_rr))
        idx, d_up, d_dn = idx[keep], d_up[keep], d_dn[keep]
        if not len(idx):
            continue

        tick = np.array([tick_of(v) for v in raw[idx]])
        cost = COST + tick / raw[idx]
        s = sig[idx]
        pf = np.array([p_target_first(u, d, v)
                       for u, d, v in zip(d_up, d_dn, s)])
        frames.append(pd.DataFrame({
            "ticker": tk, "date": g["date"].to_numpy()[idx], "i": idx,
            "close": raw[idx], "px": p[idx], "age": age[idx], "vol": s,
            "vol_decile": [vol_decile(v) for v in s],
            "d_up": d_up, "d_dn": d_dn, "rr": d_up / d_dn,
            "p_target": [p_touch(1.0 + u, v, bool(k))
                         for u, v, k in zip(d_up, s, on[idx])],
            "p_stop": [p_touch(1.0 - d, v, bool(k))
                       for d, v, k in zip(d_dn, s, on[idx])],
            "p_first": pf, "cost": cost,
            "ev": pf * d_up - (1.0 - pf) * d_dn - cost,
            **_walk(p, idx, d_up, d_dn, cost)}))
    #  A board on which nothing qualifies is a legitimate answer, not an error.
    #  `pd.concat([])` raises, which would turn "no signals today" into a
    #  traceback in whatever calls this.
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "d_up", "d_dn",
                                     "p_first", "ev", "first", "ret", "hold",
                                     "censored"])
    return pd.concat(frames, ignore_index=True)


def _walk(p: np.ndarray, idx: np.ndarray, d_up: np.ndarray,
          d_dn: np.ndarray, cost: np.ndarray) -> Dict[str, np.ndarray]:
    """Forward outcome per signal: which barrier arrived first, when, and what
    the position actually returned FILLED AT THE CLOSE."""
    n = len(p)
    m = min(HORIZON, n - 1)
    pad = np.concatenate([p[1:], np.full(m, np.nan)])
    W = np.lib.stride_tricks.sliding_window_view(pad, m)[idx]
    tgt = (p[idx] * (1.0 + d_up))[:, None]
    stp = (p[idx] * (1.0 - d_dn))[:, None]
    hit_t = np.where(W >= tgt, np.arange(m), m + 1).min(axis=1)
    hit_s = np.where(W <= stp, np.arange(m), m + 1).min(axis=1)
    #  A window running off the end of the name's life has no answer. Marking
    #  it censored rather than letting a short window read as a miss is the
    #  same discipline `first_passage` uses, and it matters here because a
    #  delisted name would otherwise count as "target never reached".
    live = np.minimum(HORIZON, n - 1 - idx)
    cens = (hit_t > m) & (hit_s > m) & (live < HORIZON)
    #  exit bar: the first barrier, or the end of the window
    ex = np.clip(np.minimum(np.minimum(hit_t, hit_s), live - 1), 0, m - 1)
    #  FILL AT THE ACTUAL CLOSE OF THE EXIT BAR, never the nominal level.
    exit_px = W[np.arange(len(idx)), ex]
    ret = exit_px / p[idx] - 1.0 - cost
    #  A SAME-DURATION HOLD IS NOT A BENCHMARK HERE, IT IS THE SAME NUMBER. The
    #  bracket exits at the CLOSE of bar `ex`, and so does a hold of `ex+1`
    #  bars, so the difference is identically zero by construction -- the shape
    #  of H39's "beat buy-and-hold on 0.0% of trades". The two comparisons that
    #  are not degenerate: hold the FULL horizon, and ANNUALISE the bracket for
    #  the capital it hands back early (A27: a 52-session bracket is redeployed
    #  five times a year, and being out of the market is most of what a stop
    #  does).
    full = W[np.arange(len(idx)), np.clip(live - 1, 0, m - 1)]
    hold = full / p[idx] - 1.0 - cost
    bars = ex + 1
    ann = np.clip(1.0 + ret, 0.01, None) ** (HORIZON / np.maximum(bars, 1)) - 1.0
    first = np.where(hit_t > m, np.where(hit_s > m, "none", "stop"),
                     np.where(hit_s > m, "target",
                              np.where(hit_t < hit_s, "target", "stop")))
    return {"first": first, "bars": bars,
            "t_target": np.where(hit_t > m, -1, hit_t + 1),
            "t_stop": np.where(hit_s > m, -1, hit_s + 1),
            "ret": ret, "ann": ann, "hold": hold, "censored": cens,
            "hit_target": hit_t <= m, "hit_stop": hit_s <= m}


def control(S: pd.DataFrame, P: pd.DataFrame, seed: int = 20260828
            ) -> pd.DataFrame:
    """THE CONTROL THAT DECIDES THE STUDY (H41). Same date, same bracket
    distances, a RANDOM eligible name. It isolates "does the Hull ribbon plus a
    confirmed swing high pick better names" from "is a bracket with these
    distances just arithmetic that any name satisfies at this rate"."""
    rng = np.random.default_rng(seed)
    px_of: Dict[str, np.ndarray] = {}
    dt_of: Dict[str, np.ndarray] = {}
    for tk, g in P.groupby("ticker", sort=False):
        px_of[tk] = g["adj_close"].to_numpy(float)
        dt_of[tk] = g["date"].to_numpy()
    pool = {d: g["ticker"].to_numpy()
            for d, g in P[P["elig"]].groupby("date")}
    #  Draw first, then walk the paths in TICKER BATCHES. A row-by-row loop
    #  over half a million signals rebuilding a sliding window each time is the
    #  difference between ten seconds and an hour, and the control is exactly
    #  the arm one is tempted to skip for being slow.
    drawn = np.empty(len(S), dtype=object)
    for k, d in enumerate(S["date"].to_numpy()):
        c = pool.get(d)
        drawn[k] = c[rng.integers(0, len(c))] if c is not None and len(c) else None
    T = S.assign(rnd=drawn).dropna(subset=["rnd"])
    out: List[pd.DataFrame] = []
    for tk, g in T.groupby("rnd", sort=False):
        p, dts = px_of[tk], dt_of[tk]
        want = g["date"].to_numpy()
        pos = np.clip(np.searchsorted(dts, want), 0, len(dts) - 1)
        ok = ((dts[pos] == want) & np.isfinite(p[pos]) & (p[pos] > 0)
              & (pos + 2 < len(p)))
        if not ok.any():
            continue
        gg, ii = g[ok], pos[ok]
        o = _walk(p, ii, gg["d_up"].to_numpy(), gg["d_dn"].to_numpy(),
                  gg["cost"].to_numpy())
        out.append(pd.DataFrame({
            "date": gg["date"].to_numpy(), "ticker": tk,
            #  the SIGNAL's ticker is kept so every control row can be paired
            #  with the row it is controlling for. An unpaired difference of
            #  two means over overlapping panels has no usable error bar.
            "src": gg["ticker"].to_numpy(),
            "d_up": gg["d_up"].to_numpy(), "d_dn": gg["d_dn"].to_numpy(),
            "ev": gg["ev"].to_numpy(), "p_first": gg["p_first"].to_numpy(),
            **o}))
    return pd.concat(out, ignore_index=True)


def block_boot(v: np.ndarray, blocks: np.ndarray, draws: int = 2000,
               seed: int = 7) -> Tuple[float, float, float]:
    """Mean of `v` with a 95% interval, resampling whole (ticker, year) BLOCKS.

    A17's lesson, and it is not optional here: one name in a green ribbon
    contributes a near-identical signal every session for months, so a row-level
    resample destroys nothing and returns an interval several times too narrow.
    The unit of resampling has to be the unit of independence.
    """
    rng = np.random.default_rng(seed)
    keys, inv = np.unique(blocks, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=len(keys))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    means = np.empty(draws)
    for d in range(draws):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([order[starts[k]:starts[k] + counts[k]]
                              for k in pick])
        means[d] = v[idx].mean() if len(idx) else np.nan
    return float(v.mean()), float(np.nanpercentile(means, 2.5)), \
        float(np.nanpercentile(means, 97.5))


def gates(M: pd.DataFrame, by: str, bins, labels) -> pd.DataFrame:
    """For each candidate gate: what it returns, what the SAME BRACKET on a
    random name returns, what HOLDING returns, and whether the difference
    survives both halves.

    This is the table that decides whether a filter is a fix or a relabelling.
    A cell that pays because of the geometry pays identically for a randomly
    chosen name, and gating on it is then not a stock screen at all — it is an
    instruction to quote a further target, which the reader can do without a
    scanner.
    """
    M = M.copy()
    M["cell"] = pd.cut(M[by], bins, labels=labels)
    rows: List[Dict] = []
    for cell, d in M.groupby("cell", observed=True):
        if len(d) < 200:
            rows.append({"cell": cell, "n": len(d),
                         "note": "insufficient data"})
            continue
        m, lo, hi = block_boot(d["diff"].to_numpy(), d["blk"].to_numpy(),
                               draws=800)
        e, l = halves(d)
        rows.append({
            "cell": cell, "n": len(d), "share": len(d) / len(M),
            "picks": d["ret"].mean(), "random": d["ret_ctl"].mean(),
            "diff": m, "lo": lo, "hi": hi,
            "early": e["diff"].mean(), "late": l["diff"].mean(),
            "both": (e["diff"].mean() > 0) and (l["diff"].mean() > 0),
            "hold": d["hold"].mean(),
            "vs hold": d["ret"].mean() - d["hold"].mean()})
    return pd.DataFrame(rows)


def calib(S: pd.DataFrame, col: str = "p_first", q: int = 10) -> pd.DataFrame:
    """Predicted against realised, in bins of the prediction."""
    R = S[S["first"] != "none"].copy()
    R["bin"] = pd.qcut(R[col], q, labels=False, duplicates="drop")
    g = R.groupby("bin")
    return pd.DataFrame({
        "n": g.size(), "predicted": g[col].mean(),
        "realised": g.apply(lambda d: float((d["first"] == "target").mean()),
                            include_groups=False)}).reset_index()


def halves(S: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mid = S["date"].quantile(0.5)
    return S[S["date"] <= mid], S[S["date"] > mid]


def summarise(S: pd.DataFrame, name: str) -> Dict:
    R = S[~S["censored"]]
    dec = R[R["first"] != "none"]
    #  BOTH MEANS, BECAUSE THEY ANSWER DIFFERENT QUESTIONS AND THIS REPO HAS
    #  WATCHED THEM DISAGREE FIVE TIMES. An equal-weighted holder is paid the
    #  ARITHMETIC mean (A18); a sequence of trades COMPOUNDS at the log mean.
    #  Annualising each trade's arithmetic return separately and averaging is
    #  the wrong third thing: an 18-bar -30% loss annualises to -99%, and a
    #  handful of those swamp the average of everything else.
    lg = np.log(np.clip(1.0 + R["ret"].to_numpy(), 0.01, None))
    bars = float(R["bars"].mean())
    return {"arm": name, "n": len(R),
            "P(tgt ever)": float(R["hit_target"].mean()),
            "P(tgt first)": float((dec["first"] == "target").mean())
            if len(dec) else np.nan,
            "predicted": float(R["p_first"].mean()),
            "bars": bars,
            "mean ret": float(R["ret"].mean()),
            "median": float(R["ret"].median()),
            "mean log": float(lg.mean()),
            "ann log": float(np.exp(lg.mean() * HORIZON / max(bars, 1)) - 1.0),
            "hold 252": float(R["hold"].mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--step", type=int, default=21)
    ap.add_argument("--reuse", action="store_true",
                    help="load the cached signal frame instead of rebuilding "
                         "it; the swing-level scan is the only slow part")
    a = ap.parse_args()

    P = load(holdout=True)
    P["elig"] = (P["tradeable"].astype(bool)
                 & (np.exp(P["log_turnover"].fillna(-np.inf)) >= MIN_TV))
    P = P[P["tradeable"].astype(bool)].sort_values(["ticker", "date"])
    cache = os.path.join(OUT, "signal_backtest.csv.gz")
    if a.reuse and os.path.exists(cache):
        S = pd.read_csv(cache, parse_dates=["date"])
        print(f"(reusing {cache})")
    else:
        S = signal_rows(P)
    S = S.sort_values(["date", "ticker"]).reset_index(drop=True)
    print(f"{len(S):,} signal rows, {S['ticker'].nunique()} names, "
          f"{pd.Timestamp(S['date'].min()):%Y-%m-%d} -> "
          f"{pd.Timestamp(S['date'].max()):%Y-%m-%d}")
    print(f"{int(S['censored'].sum()):,} censored (window runs off the name's "
          f"life) and excluded from every rate below\n")

    C = control(S, P)
    print("=== B1/B2 — the whole signal population, 252-session horizon")
    rows = [summarise(S, "scanner signals"), summarise(C, "random name, "
                                                       "same bracket")]
    D = pd.DataFrame(rows)
    print(D.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print("\n=== B1 — is P(target first) calibrated?")
    K = calib(S)
    print(f"{'bin':>4}{'n':>9}{'predicted':>11}{'realised':>11}{'gap':>9}")
    for _, r in K.iterrows():
        print(f"{int(r['bin']):>4}{int(r['n']):>9,}{r['predicted']:>11.3f}"
              f"{r['realised']:>11.3f}{r['realised'] - r['predicted']:>+9.3f}")
    dec = S[(S["first"] != "none") & ~S["censored"]]
    print(f"  pooled predicted {dec['p_first'].mean():.3f} against realised "
          f"{(dec['first'] == 'target').mean():.3f}")

    print("\n=== B2 — by reward-to-risk, which is what the list varies most")
    S2 = S[~S["censored"]].copy()
    S2["rr_bin"] = pd.cut(S2["rr"], [0, 0.75, 1.5, 2.5, 4.0, 99],
                          labels=["<0.75", "0.75-1.5", "1.5-2.5", "2.5-4",
                                  ">4"])
    g = S2.groupby("rr_bin", observed=True)
    T = pd.DataFrame({
        "n": g.size(),
        "P(target first)": g.apply(
            lambda d: float((d[d["first"] != "none"]["first"] == "target")
                            .mean()), include_groups=False),
        "predicted": g["p_first"].mean(),
        "mean ret": g["ret"].mean(), "mean hold": g["hold"].mean()})
    print(T.to_string(float_format=lambda v: f"{v:,.4f}"))

    print("\n=== B3 — does the EV column predict the realised return?")
    S2["ev_bin"] = pd.qcut(S2["ev"], 10, labels=False, duplicates="drop")
    g = S2.groupby("ev_bin")
    E = pd.DataFrame({"n": g.size(), "EV": g["ev"].mean(),
                      "realised": g["ret"].mean(),
                      "hold": g["hold"].mean(),
                      "edge": g["ret"].mean() - g["hold"].mean()})
    print(E.to_string(float_format=lambda v: f"{v:,.4f}"))
    pos, neg = S2[S2["ev"] > 0], S2[S2["ev"] <= 0]
    print(f"  EV>0: n {len(pos):,}, realised {pos['ret'].mean():+.4f}, "
          f"predicted {pos['ev'].mean():+.4f}")
    print(f"  EV<=0: n {len(neg):,}, realised {neg['ret'].mean():+.4f}, "
          f"predicted {neg['ev'].mean():+.4f}")
    e, l = halves(S2)
    for nm, h in (("early", e), ("late", l)):
        hp, hn = h[h["ev"] > 0], h[h["ev"] <= 0]
        print(f"  {nm}: EV>0 {hp['ret'].mean():+.4f} vs EV<=0 "
              f"{hn['ret'].mean():+.4f}   "
              f"{'SEPARATES' if hp['ret'].mean() > hn['ret'].mean() else 'INVERTS'}")

    print(f"\n=== B4 — the tradeable version: top {a.top} by EV, "
          f"rebalanced every {a.step} sessions")
    dates = np.sort(S["date"].unique())[::a.step]
    B = S[S["date"].isin(dates) & ~S["censored"]]
    picks = (B.sort_values("ev", ascending=False).groupby("date")
             .head(a.top))
    CB = C[C["date"].isin(dates) & ~C["censored"]]
    print(f"{'arm':>28}{'n':>8}{'P(tgt 1st)':>12}{'mean':>9}{'median':>9}"
          f"{'bars':>7}{'ann log':>9}{'hold252':>9}")
    for nm, d in (("top-EV picks", picks), ("all signals", B),
                  ("random name, same bracket", CB)):
        dd = d[d["first"] != "none"]
        #  ANNUALISE THE MEAN LOG, never the mean of the per-trade annualised
        #  return. A first version did the latter and printed 1.7e28: a +50%
        #  trade held one bar compounds to 1.5^252, and one such row swamps a
        #  hundred thousand others. A statistic that cannot occur is the
        #  cheapest bug detector available, and this is the fourth catch of
        #  that kind in four studies.
        lg = np.log(np.clip(1.0 + d["ret"].to_numpy(), 0.01, None)).mean()
        bars = max(d["bars"].mean(), 1.0)
        print(f"{nm:>28}{len(d):>8,}"
              f"{(dd['first'] == 'target').mean():>12.3f}"
              f"{d['ret'].mean():>+9.4f}{d['ret'].median():>+9.4f}"
              f"{bars:>7.0f}"
              f"{np.exp(lg * HORIZON / bars) - 1.0:>+9.4f}"
              f"{d['hold'].mean():>+9.4f}")
    #  PAIRED, CLUSTERED, AND HALF-SPLIT — the three things that have decided
    #  every result in this repo. Each signal is matched to the control row
    #  drawn for it, so the difference is within-date and within-bracket.
    M = S[~S["censored"]].merge(
        C[~C["censored"]][["date", "src", "ret", "hold", "first"]],
        left_on=["date", "ticker"], right_on=["date", "src"],
        suffixes=("", "_ctl"))
    M["diff"] = M["ret"] - M["ret_ctl"]
    M["blk"] = (M["ticker"].astype(str) + "|"
                + pd.to_datetime(M["date"]).dt.year.astype(str))
    print("\n=== the paired difference against the control, "
          "(ticker, year) block bootstrap")
    for nm, d in (("all signals", M),
                  ("EV > 0 only", M[M["ev"] > 0]),
                  (f"top {a.top} by EV", M.merge(
                      picks[["date", "ticker"]], on=["date", "ticker"]))):
        if len(d) < 50:
            print(f"{nm:>18}  insufficient data (n={len(d)})")
            continue
        m, lo, hi = block_boot(d["diff"].to_numpy(), d["blk"].to_numpy())
        e, l = halves(d)
        print(f"{nm:>18}  n {len(d):>7,}  {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
              f"   early {e['diff'].mean():+.4f}  late {l['diff'].mean():+.4f}"
              f"   {'BOTH' if e['diff'].mean() > 0 and l['diff'].mean() > 0 else 'NOT BOTH'}")

    #  And against the alternative the reader would actually take: own the name
    #  for the year instead of bracketing it.
    m, lo, hi = block_boot((M["ret"] - M["hold"]).to_numpy(),
                           M["blk"].to_numpy())
    print(f"{'bracket vs hold':>18}  n {len(M):>7,}  {m:+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}]")

    #  ===== WHICH GATE, IF ANY, IS WORTH PUTTING IN THE LIVE SCANNER =========
    def _show(title, G):
        print(f"\n=== {title}")
        print(f"{'cell':>10}{'n':>9}{'share':>7}{'picks':>9}{'random':>9}"
              f"{'diff':>9}{'95% CI':>20}{'early':>9}{'late':>9}{'both':>6}"
              f"{'hold':>9}{'vs hold':>9}")
        for _, r in G.iterrows():
            if r.get("note"):
                print(f"{str(r['cell']):>10}{int(r['n']):>9,}   "
                      f"{r['note']}")
                continue
            print(f"{str(r['cell']):>10}{int(r['n']):>9,}{r['share']:>7.0%}"
                  f"{r['picks']:>+9.4f}{r['random']:>+9.4f}{r['diff']:>+9.4f}"
                  f"  [{r['lo']:+.4f}, {r['hi']:+.4f}]"
                  f"{r['early']:>+9.4f}{r['late']:>+9.4f}"
                  f"{'YES' if r['both'] else 'no':>6}"
                  f"{r['hold']:>+9.4f}{r['vs hold']:>+9.4f}")

    _show("gate on reward:risk — is the paying geometry a SELECTION effect?",
          gates(M, "rr", [0, 0.75, 1.5, 2.5, 4.0, 99],
                ["<0.75", "0.75-1.5", "1.5-2.5", "2.5-4", ">4"]))
    _show("gate on expectancy",
          gates(M, "ev", [-9, -0.02, 0.0, 0.01, 0.02, 9],
                ["<-2%", "-2-0%", "0-1%", "1-2%", ">2%"]))
    _show("gate on how long the ribbon has been green",
          gates(M, "age", [-1, 5, 20, 60, 9999],
                ["0-5", "6-20", "21-60", ">60"]))
    S.to_csv(os.path.join(OUT, "signal_backtest.csv.gz"), index=False)
    C.to_csv(os.path.join(OUT, "signal_control.csv.gz"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
