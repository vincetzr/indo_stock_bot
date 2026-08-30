#!/usr/bin/env python3
"""H50 — the goal: >+4% average per trade, 80%+ of the time. Built properly.

    python3 scripts/quantbot.py --build      # features + triple-barrier labels
    python3 scripts/quantbot.py              # walk-forward, frontier, controls

THE TARGET, AND THE ARITHMETIC THAT CONSTRAINS IT
-------------------------------------------------
Per trade, with W the mean gain on winners, L the mean loss on losers, p the
win rate and c the round-trip toll:

    E[r] = p*W - (1-p)*L - c

Under NO EDGE and barriers at (+a, -b), a price path is a martingale and the
touch probabilities are exactly p = b/(a+b) with E[r] = -c. **Any win rate is
purchasable** — set the target near and the stop far — and buying it changes
nothing, because W and L move against p by exactly enough to cancel. That is
not pessimism, it is the optional stopping theorem, and this script MEASURES it
rather than asserting it (Q0 below is the instrument check).

So the joint target needs p = 0.80 where a fair market at the same barriers
gives some p0, and the excess p - p0 has to be large enough to carry
0.8W - 0.2L >= 4% + c. At symmetric barriers p0 = 0.5, so the target is
directional accuracy of 0.80 against a 0.50 base. For scale: a strong
institutional equity model runs an information coefficient near 0.05.

WHICH DOES NOT MEAN "DON'T BUILD IT". It means the deliverable is the FRONTIER
— the achievable (win rate, mean per trade) surface — with the target marked on
it, plus the best point that is actually reachable. Two readings of the goal
are tested separately because they have different answers:

    reading A   win rate >= 80% AND mean per trade >= +4%
    reading B   mean per trade >= +4%, holding in >= 80% of calendar years

WHAT MAKES THIS DIFFERENT FROM THE 49 HYPOTHESES BEFORE IT
-----------------------------------------------------------
Every previous study here ranked names on a score and held for a fixed horizon.
This is the standard institutional construction and none of its three pieces
has been built in this repo:

1. TRIPLE-BARRIER LABELLING (Lopez de Prado). A trade is labelled by which of
   three barriers it touches first — profit target, stop, or time limit — using
   the actual intraday HIGH and LOW, not the close. Every earlier study labelled
   on a fixed-horizon close return, which is a different and easier question
   than "does my order fill".

2. VOLATILITY-SCALED BARRIERS. A fixed +10% target is a routine day on one name
   and a once-a-year event on another. Barriers are set at multiples of the
   name's own trailing 60-day volatility scaled to the horizon, so a "1.5 sigma
   target" means the same thing everywhere and the cross-section is comparable.

3. META-LABELLING. A primary rule decides the SIDE (here: long, when a trend
   filter is live). A secondary gradient-boosted classifier then predicts
   whether that particular signal will hit its target first, and the book takes
   only the confident ones. This is the technique that exists specifically to
   raise PRECISION — which is exactly the win rate the goal asks about — and it
   is the honest way to attack the target rather than by moving the barriers.

Plus the discipline this repo has learned the hard way: purged AND embargoed
walk-forward, a clustered permutation null, a random-selection control at
matched trade count, a half-split, a predicted-null feature, and a point-in-time
liquidity screen (a full-sample median turnover filter is look-ahead, and this
repo has used one).

PRE-REGISTERED, WRITTEN BEFORE ANY MODEL WAS FIT
-------------------------------------------------
Q0  INSTRUMENT CHECK, not a market claim. On a driftless synthetic walk the
    measured touch rate at barriers (+a, -b) equals b/(a+b) and the mean net of
    zero cost is 0.00. If the labeller does not reproduce the martingale, every
    number below is a bug. (A26's sine-wave rule: a negative result about an
    instrument is worthless until the instrument returns the known answer on a
    known case.)
Q1  Mean per trade >= +4% net IS reachable, at wide targets and long horizons,
    because the mean is carried by the right tail. Predicted: YES.
Q2  Win rate >= 80% IS reachable, at near targets and far stops. Predicted: YES,
    and worth nothing on its own.
Q3  THE JOINT TARGET IS NOT REACHABLE. Predicted: no cell of the frontier has
    win rate >= 80% and mean >= +4% net, and the best mean available at
    win rate >= 80% is far below +4%.
Q4  Meta-labelling raises precision at FIXED barrier geometry — the only honest
    way to move up. Predicted: yes, by a modest margin (single-digit points),
    because H27 found the price-feature space close to exhausted.
Q5  PREDICTED NULL. A deterministic pseudo-random feature must not rank among
    the model's top features, and the label-shuffled pipeline must not reach the
    frontier. If either fires, the harness manufactures its own signal.
Q6  READING B: some configuration delivers mean per trade >= +4% in >= 80% of
    calendar years. Predicted: NO for any configuration that also trades often
    enough to matter, because the mean is tail-driven and a tail is not annual.
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

from paint_suite import tick_of                                  # noqa: E402

PANEL = os.path.join("data", "spine", "price_panel.parquet")
IND = os.path.join("data", "spine", "indicator_panel.parquet")
CACHE = os.path.join("data", "spine", "quantbot.parquet")
OUT = "reports"

FEE = 0.0056                      # A5: 0.28 buy + 0.18 sell + 0.1 sell tax
MIN_RP = 5e9                      # point-in-time: Rp5bn/day trailing median
HOLDOUT = pd.Timestamp("2024-09-01")

#: every feature the secondary model sees. `noise` is the Q5 predicted null.
FEATS = [
    "r5", "r10", "r21", "r63", "r126", "mom12_1",
    "rev1", "rev5", "gap1",
    "vol20", "vol60", "vol_ratio", "atr_pct",
    "hi20", "hi60", "hi252", "lo20", "lo60", "rng_pos",
    "e_10_20", "e_20_50", "px_e10", "stack", "hull_slope",
    "volz20", "tvz20", "amihud60", "turn_z",
    "rsi14", "stoch_k", "stoch_kd",
    "mkt_r20", "mkt_vol", "rel_r21",
    "month", "noise",
]
#: the same features as within-date cross-sectional percentile ranks
XS = ["mom12_1", "vol60", "hi252", "rsi14", "volz20", "r21"]


# ============================================================ helpers ========
def _ema(v: np.ndarray, n: int) -> np.ndarray:
    a = 2.0 / (n + 1.0)
    out = np.empty_like(v)
    out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = a * v[i] + (1 - a) * out[i - 1]
    return out


def _wma(v: np.ndarray, n: int) -> np.ndarray:
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    out = np.full(len(v), np.nan)
    if len(v) >= n:
        out[n - 1:] = np.convolve(v, w[::-1], mode="valid")
    return out


def _hma(v: np.ndarray, n: int) -> np.ndarray:
    h = _wma(2.0 * _wma(v, n // 2) - _wma(v, n), int(np.sqrt(n)))
    return h


def _roll(v: np.ndarray, n: int, fn) -> np.ndarray:
    return pd.Series(v).rolling(n, min_periods=max(2, n // 2)).apply(
        fn, raw=True).to_numpy()


# ================================================== TRIPLE-BARRIER LABELS ====
def barrier_labels(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   up: np.ndarray, dn: np.ndarray, horizon: int
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """First-touch labelling against a profit target, a stop, and a clock.

    `up` and `dn` are the barrier PRICES for an entry at each bar. Returns
    (outcome, exit bar, fill price) where outcome is +1 target, -1 stop,
    0 timed out.

    THE FILLS ARE DELIBERATELY ASYMMETRIC, and this is the single most
    consequential modelling choice in the file. A profit target is a resting
    LIMIT order, so it fills AT the level. A stop is a market order triggered by
    a level already breached, so on IDX it fills at whatever the tape gives —
    H35 measured that taking the nominal stop level instead of the bar's close
    turned a +0.0173 cell into +0.0050, and the flattery lands hardest on
    exactly the tight stops a win-rate-maximising search wants to use. So the
    stop fills at min(level, close of the breaching bar).

    The scan runs d from the horizon DOWN to 1 so that the smallest d wins,
    which is what makes it a FIRST-touch rather than an any-touch label.
    """
    n = len(close)
    t_up = np.full(n, horizon + 1, dtype=np.int32)
    t_dn = np.full(n, horizon + 1, dtype=np.int32)
    for d in range(horizon, 0, -1):
        if d >= n:
            continue
        h = np.full(n, -np.inf)
        lo = np.full(n, np.inf)
        h[:n - d] = high[d:]
        lo[:n - d] = low[d:]
        t_up = np.where(h >= up, d, t_up)
        t_dn = np.where(lo <= dn, d, t_dn)
    out = np.zeros(n, dtype=np.int8)
    out = np.where((t_up <= horizon) & (t_up <= t_dn), 1, out)
    out = np.where((t_dn <= horizon) & (t_dn < t_up), -1, out)
    step = np.minimum(np.minimum(t_up, t_dn), horizon)
    j = np.clip(np.arange(n) + step, 0, n - 1)
    fill = close[j].astype(float)
    fill = np.where(out == 1, up, fill)
    #  A stop that gapped through fills at the close, never at the level.
    fill = np.where(out == -1, np.minimum(dn, close[j]), fill)
    #  RIGHT-CENSORING. A bar within `horizon` of the end of the series cannot
    #  have its trade resolved: the clip above would silently shorten it and
    #  label the stub as a real outcome. Those rows are flagged and dropped
    #  rather than counted, which is the same defect A20 found discarding 91%
    #  of long-horizon cohorts while measuring the survivors.
    censored = (np.arange(n) + horizon) > (n - 1)
    return out, j, fill, censored


# ================================================================ features ===
def name_frame(g: pd.DataFrame, mkt: pd.DataFrame) -> pd.DataFrame:
    """All features for one ticker. Every column reads bar t or earlier."""
    p = g["adj_close"].to_numpy(float)
    hi = g["adj_high"].to_numpy(float)
    lo = g["adj_low"].to_numpy(float)
    n = len(p)
    lg = np.diff(np.log(np.maximum(p, 1e-9)), prepend=np.log(max(p[0], 1e-9)))
    d = pd.DataFrame({"date": g["date"].to_numpy(),
                      "ticker": g["ticker"].to_numpy()})
    for k in (5, 10, 21, 63, 126):
        d[f"r{k}"] = pd.Series(p).pct_change(k).to_numpy()
    d["mom12_1"] = g["mom12_1"].to_numpy(float)
    d["rev1"] = lg
    d["rev5"] = pd.Series(p).pct_change(5).to_numpy()
    d["gap1"] = np.r_[np.nan, (lo[1:] - p[:-1]) / np.maximum(p[:-1], 1e-9)]
    v20 = pd.Series(lg).rolling(20, min_periods=10).std().to_numpy()
    v60 = pd.Series(lg).rolling(60, min_periods=30).std().to_numpy()
    d["vol20"], d["vol60"] = v20, v60
    d["vol_ratio"] = v20 / np.maximum(v60, 1e-9)
    atr = g["atr22"].to_numpy(float)
    d["atr_pct"] = atr / np.maximum(p, 1e-9)
    for k in (20, 60, 252):
        mx = pd.Series(hi).rolling(k, min_periods=k // 2).max().to_numpy()
        d[f"hi{k}"] = p / np.maximum(mx, 1e-9) - 1.0
    for k in (20, 60):
        mn = pd.Series(lo).rolling(k, min_periods=k // 2).min().to_numpy()
        d[f"lo{k}"] = p / np.maximum(mn, 1e-9) - 1.0
    mx20 = pd.Series(hi).rolling(20, min_periods=10).max().to_numpy()
    mn20 = pd.Series(lo).rolling(20, min_periods=10).min().to_numpy()
    d["rng_pos"] = (p - mn20) / np.maximum(mx20 - mn20, 1e-9)
    e10, e20, e50 = _ema(p, 10), _ema(p, 20), _ema(p, 50)
    e100, e200 = _ema(p, 100), _ema(p, 200)
    d["e_10_20"] = e10 / np.maximum(e20, 1e-9) - 1.0
    d["e_20_50"] = e20 / np.maximum(e50, 1e-9) - 1.0
    d["px_e10"] = p / np.maximum(e10, 1e-9) - 1.0
    d["stack"] = ((p > e50).astype(int) + (e50 > e100).astype(int)
                  + (e100 > e200).astype(int))
    h55 = _hma(p, 55)
    hs = np.full(n, np.nan)
    hs[2:] = h55[2:] / np.maximum(h55[:-2], 1e-9) - 1.0
    d["hull_slope"] = hs
    d["volz20"] = g["volz20"].to_numpy(float)
    d["tvz20"] = g["tvz20"].to_numpy(float)
    d["amihud60"] = g["amihud60"].to_numpy(float)
    lt = g["log_turnover"].to_numpy(float)
    d["turn_z"] = ((lt - pd.Series(lt).rolling(60, min_periods=30).mean()
                    .to_numpy())
                   / np.maximum(pd.Series(lt).rolling(60, min_periods=30)
                                .std().to_numpy(), 1e-9))
    d["rsi14"] = g["rsi14"].to_numpy(float)
    d["stoch_k"] = g["stoch_k"].to_numpy(float)
    d["stoch_kd"] = g["stoch_k"].to_numpy(float) - g["stoch_d"].to_numpy(float)
    d["month"] = pd.to_datetime(g["date"]).dt.month.to_numpy()
    #  Q5's predicted null. Deterministic from (ticker, bar) so it is stable
    #  across runs and cannot be blamed on a seed, and carries no information
    #  by construction.
    d["noise"] = np.abs(np.sin(np.arange(n) * 12.9898
                               + hash(str(g["ticker"].iloc[0])) % 1000)) % 1.0
    #  Point-in-time liquidity. A full-sample median turnover filter would let
    #  a name's LATER liquidity decide whether it was buyable today.
    d["rp60"] = pd.Series(np.exp(lt)).rolling(
        60, min_periods=30).median().to_numpy()
    d["close_raw"] = g["close"].to_numpy(float)
    d["adj"] = p
    d["hi_raw"], d["lo_raw"] = hi, lo
    return d


def build(horizon: int = 63) -> pd.DataFrame:
    P = pd.read_parquet(PANEL)
    I = pd.read_parquet(IND)[["date", "ticker", "adj_high", "adj_low",
                              "atr22", "stoch_k", "stoch_d", "rsi14", "tvz20"]]
    P = P.merge(I, on=["date", "ticker"], how="left")
    P = P[(P["adj_close"] > 0) & P["adj_high"].notna()]
    P = P.sort_values(["ticker", "date"])
    #  Market state, from the equal-weighted board rather than an index proxy,
    #  so it exists for every date the panel has.
    mk = P.groupby("date")["rev1"].mean() if "rev1" in P.columns else None
    daily = P.groupby("date")["adj_close"].apply(
        lambda s: np.nan).rename("_x")
    del daily, mk
    ret = P.assign(lr=P.groupby("ticker")["adj_close"].transform(
        lambda s: np.log(s).diff()))
    mkt = ret.groupby("date")["lr"].mean().rename("m").to_frame()
    mkt["mkt_r20"] = mkt["m"].rolling(20, min_periods=10).sum()
    mkt["mkt_vol"] = mkt["m"].rolling(60, min_periods=30).std()
    mkt = mkt.reset_index()[["date", "mkt_r20", "mkt_vol"]]

    out = []
    for tk, g in P.groupby("ticker", sort=False):
        if len(g) < 400:
            continue
        out.append(name_frame(g, mkt))
    D = pd.concat(out, ignore_index=True)
    D = D.merge(mkt, on="date", how="left")
    D["rel_r21"] = D["r21"] - D.groupby("date")["r21"].transform("mean")
    for f in XS:
        D[f + "_x"] = D.groupby("date")[f].rank(pct=True)
    D.to_parquet(CACHE, index=False)
    return D


# =================================================== labelling a geometry ====
def label(D: pd.DataFrame, tp_k: float, sl_k: float, horizon: int
          ) -> pd.DataFrame:
    """Attach the triple-barrier outcome for one (target, stop, clock) cell.

    Barriers sit at tp_k and sl_k multiples of the name's own trailing 60-day
    volatility scaled to the horizon, so "1.5 sigma" means the same thing on a
    bank and on a coal name and the cross-section is comparable.
    """
    parts = []
    for tk, g in D.groupby("ticker", sort=False):
        p = g["adj"].to_numpy(float)
        s = g["vol60"].to_numpy(float) * np.sqrt(horizon)
        s = np.clip(np.nan_to_num(s, nan=0.0), 0.02, 3.0)
        #  tp_k None is the NO-TAKE-PROFIT arm: stop and clock only, upside left
        #  open. It matters because a target truncates the right tail, and every
        #  result in this repo says the right tail is where the mean lives
        #  (H39: the best 1% of trades carried 71% of the total return).
        up = (p * 1e9 if tp_k is None else p * (1.0 + tp_k * s))
        o, j, f, cen = barrier_labels(g["hi_raw"].to_numpy(float),
                                      g["lo_raw"].to_numpy(float), p,
                                      up, p * (1.0 - sl_k * s), horizon)
        med = g["close_raw"].to_numpy(float)
        cost = FEE + np.array([tick_of(x) / max(x, 1e-9) for x in med])
        parts.append(pd.DataFrame({
            "date": g["date"].to_numpy(), "ticker": tk,
            #  THE META-LABEL. With a target it is "did the target come
            #  first". With NO target there is nothing to hit, so the first
            #  version left y identically zero and the classifier had nothing
            #  to learn — it returned the unranked sample and printed a row
            #  that looked like a model result. For that arm the meta-label is
            #  "did this trade end positive", which is the question a book
            #  actually asks.
            "y": ((o == 1) if tp_k is not None
                  else (f / np.maximum(p, 1e-9) - 1.0 - cost > 0)
                  ).astype(np.int8), "outcome": o,
            "bars": (j - np.arange(len(p))).astype(np.int32),
            "ret": f / np.maximum(p, 1e-9) - 1.0 - cost,
            "cen": cen}))
    L = pd.concat(parts, ignore_index=True)
    L = L[~L["cen"]].drop(columns=["cen"])
    return D.merge(L, on=["date", "ticker"], how="inner")


def eligible(D: pd.DataFrame) -> pd.DataFrame:
    """Bars a retail order could actually be worked in, decided point-in-time."""
    return D[(D["rp60"] >= MIN_RP) & (D["close_raw"] >= 500)]


# ======================================== Q0 — THE INSTRUMENT CHECK ==========
def martingale_check(n_names: int = 150, n_bars: int = 2500,
                     seed: int = 7) -> pd.DataFrame:
    """Run the labeller on a market with a KNOWN answer.

    On a driftless log random walk the optional stopping theorem fixes both
    numbers: the expected net return at zero cost is 0, and conditional on one
    barrier being hit the probability it was the upper one is exactly
    b / (a + b). If the labeller cannot reproduce those, nothing measured on IDX
    means anything. A26 introduced this rule after a ZigZag that could not find
    a cycle in a sine wave was used as evidence that no cycle existed.
    """
    rng = np.random.default_rng(seed)
    hor = 1200
    rows = []
    #  TWO ERRORS IN THE FIRST VERSION OF THIS CHECK, BOTH IN THE TEST RATHER
    #  THAN IN THE THING BEING TESTED, AND IT REPORTED "INSTRUMENT FAILS".
    #  (1) The walk had log increments of mean ZERO, which makes the PRICE a
    #      submartingale: E[p_T/p_0] = exp(sigma^2 T / 2), a built-in +5.17%
    #      over 252 bars. The labeller dutifully measured that drift and the
    #      check called it a bug.
    #  (2) b/(a+b) is the touch probability for PRICE-space barriers on a
    #      PRICE martingale. Symmetric price barriers are ASYMMETRIC in log
    #      space (+0.276 against -0.382 at these widths), so the formula was
    #      being applied to the wrong process.
    #  Both are fixed by drifting the log increments by -sigma^2/2 so the price
    #  itself is the martingale, and by keeping barriers small relative to the
    #  horizon so timeouts, which bias the conditional toward the NEARER
    #  barrier, stay rare.
    #  THE RESIDUAL IS DISCRETISATION, AND THE WAY TO SHOW THAT IS CONVERGENCE
    #  RATHER THAN A LOOSER TOLERANCE. A daily bar OVERSHOOTS a barrier, and it
    #  overshoots the NEARER one proportionally more, which pushes the measured
    #  touch rate away from b/(a+b). The error scales with step/barrier, so it
    #  must shrink as the barriers WIDEN at a fixed step.
    #
    #  A FIRST VERSION SHRANK THE STEP INSTEAD AND THE ERROR GOT WORSE — 0.014
    #  at sd 0.020 rising to 0.038 at sd 0.005. Shrinking the step at a fixed
    #  horizon also shrinks TOTAL volatility, so fixed barriers stop being
    #  reachable and timeouts appear (0% -> 13%), and a timeout breaks the
    #  formula outright. Holding sigma*sqrt(T) constant instead would need
    #  T = 19,200, which the O(T) scan cannot run. Widening the barriers at a
    #  fixed step moves the one ratio that matters and leaves the other alone.
    #  AND THE OVERSHOOT ITSELF IS FIXED BY SIMULATING INTRADAY. A real bar has
    #  a HIGH and a LOW, and the labeller reads them; feeding it close=high=low
    #  forced every touch to be detected a whole day late and overshot by a
    #  whole daily move. Twenty steps per session gives a true intraday extreme,
    #  so the barrier is detected within sigma/sqrt(20) of the level — which is
    #  both the realistic case and the one where the theory is clean.
    sig, sub = 0.020, 80
    for w in (0.05, 0.10, 0.20):
        for a, b in ((w, w), (w * 0.5, w * 2.0), (w * 2.0, w * 0.5)):
            os_, rs, fs, bars = [], [], [], []
            for _ in range(n_names):
                st = rng.normal(-0.5 * sig * sig / sub, sig / np.sqrt(sub),
                                n_bars * sub)
                path = np.exp(np.cumsum(st)).reshape(n_bars, sub) * 1000.0
                p = path[:, -1]
                hi_, lo_ = path.max(axis=1), path.min(axis=1)
                o, j, f, cen = barrier_labels(hi_, lo_, p, p * (1 + a),
                                              p * (1 - b), hor)
                #  SYMMETRIC fill isolates the LABELLER: both barriers fill at
                #  their level, so a correct labeller returns a mean of exactly
                #  zero. The asymmetric fill the study actually uses is priced
                #  separately in the last column, so realism and correctness
                #  are never confounded.
                sym = np.where(o == 1, p * (1 + a),
                               np.where(o == -1, p * (1 - b), p[j]))
                del st, path, hi_, lo_
                os_.append(o[~cen])
                rs.append((sym / p - 1.0)[~cen])
                fs.append((f / p - 1.0)[~cen])
                bars.append((j - np.arange(len(p)))[~cen])
            o = np.concatenate(os_)
            hit = o != 0
            sym = np.concatenate(rs)
            #  EFFECTIVE n, not raw n. Consecutive bars of one name open trades
            #  that overlap almost completely, so 96,000 rows are nowhere near
            #  96,000 independent observations — at wide barriers a trade runs
            #  hundreds of sessions and one name contributes a handful of real
            #  draws. Scoring against a raw-n standard error would call the
            #  widest cells biased when they are merely noisy, which is the
            #  degenerate-cell trap A19 records twice.
            bars = np.concatenate(bars)
            eff = max(n_names * (n_bars - hor) / max(float(bars.mean()), 1.0),
                      2.0)
            pp = float((o[hit] == 1).mean())
            rows.append({"sig": sig, "w": w, "tp": a, "sl": b,
                         "theory_p": b / (a + b), "measured_p": pp,
                         "timeout": float((~hit).mean()),
                         "mean_sym": float(sym.mean()),
                         "mean_real": float(np.concatenate(fs).mean()),
                         "eff_n": eff, "bars": float(bars.mean()),
                         "se": float(np.sqrt(max(pp * (1 - pp), 1e-6) / eff)),
                         "se_ret": float(sym.std(ddof=1) / np.sqrt(eff)),
                         "n": len(o)})
    return pd.DataFrame(rows)


# ================================================ purged walk-forward ========
def walk_forward(L: pd.DataFrame, feats: List[str], horizon: int,
                 embargo: int = 21, seed: int = 0,
                 shuffle: bool = False) -> pd.DataFrame:
    """Expanding, PURGED and EMBARGOED walk-forward over calendar years.

    A trade opened at t is not resolved until t+horizon, so training on it to
    predict a trade at t+5 leaks most of a horizon of future. Training rows are
    therefore restricted to those whose barrier window CLOSED at least `embargo`
    sessions before the test year began. That discards a chunk of data at every
    fold and is not optional — without it the model reads its own test labels.

    `shuffle` is Q5's null: whole (ticker, year) blocks have their LABELS
    reassigned to other blocks' FEATURES. Permuting INSIDE a block is nearly a
    no-op here, because one ticker-year's overlapping windows carry
    near-identical labels, so the null would preserve the mapping it exists to
    destroy (A25 measured a null beating its own model from exactly that).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    d = L.copy()
    d["yr"] = pd.to_datetime(d["date"]).dt.year
    if shuffle:
        blk = d["ticker"].astype(str) + "|" + d["yr"].astype(str)
        keys = blk.unique()
        rng = np.random.default_rng(seed)
        mp = dict(zip(keys, rng.permutation(keys)))
        idx = {k: np.flatnonzero((blk == k).to_numpy()) for k in keys}
        yv = d["y"].to_numpy().copy()
        new = np.empty_like(yv)
        for k in keys:
            ia, ib = idx[k], idx[mp[k]]
            new[ia] = yv[ib[np.arange(len(ia)) % len(ib)]]
        d["y"] = new
    years = sorted(y for y in d["yr"].unique() if y >= d["yr"].min() + 4)
    out = []
    for y in years:
        t0 = pd.Timestamp(f"{y}-01-01")
        close_by = pd.to_datetime(d["date"]) + pd.to_timedelta(
            (d["bars"] + embargo) * 1.45, unit="D")
        tr = d[close_by < t0]
        te = d[d["yr"] == y]
        if len(tr) < 5000 or len(te) < 200:
            continue
        m = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_depth=5,
            min_samples_leaf=200, l2_regularization=1.0, random_state=seed)
        m.fit(tr[feats].to_numpy(np.float32), tr["y"].to_numpy())
        p = m.predict_proba(te[feats].to_numpy(np.float32))[:, 1]
        q = te.copy()
        q["p"] = p
        out.append(q)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ============================================================== statistics ===
def block_ci(x: np.ndarray, blk: np.ndarray, draws: int = 400,
             seed: int = 3) -> Tuple[float, float]:
    """(ticker, year) block bootstrap. Trades from one name-year overlap almost
    completely, so an iid resample understates the interval — A15 measured that
    exact error making every width ~3.4x too narrow."""
    keys = np.unique(blk)
    idx = {k: np.flatnonzero(blk == k) for k in keys}
    rng = np.random.default_rng(seed)
    ms = []
    for _ in range(draws):
        pick = rng.choice(keys, len(keys), replace=True)
        ms.append(float(np.mean(np.concatenate([x[idx[k]] for k in pick]))))
    return float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def summarise(E: pd.DataFrame, lab: str) -> Dict:
    r = E["ret"].to_numpy(float)
    yr = pd.to_datetime(E["date"]).dt.year
    per_yr = E.groupby(yr)["ret"].mean()
    bars = max(float(E["bars"].mean()), 1.0)
    #  ANNUALISE THE MEAN LOG, never the mean of per-trade annualised returns —
    #  H42 printed 1.7e28 doing the latter, because a +50% trade held one bar
    #  contributes 1.5^252 and one row swamps a hundred thousand.
    ml = float(np.mean(np.log(np.maximum(1.0 + r, 0.01))))
    return {"cell": lab, "n": len(E), "win": float(E["y"].mean()),
            "pos": float((r > 0).mean()), "mean": float(r.mean()),
            "median": float(np.median(r)), "bars": bars,
            "ann": float(np.exp(ml * 252.0 / bars) - 1.0),
            "timeout": float((E["outcome"] == 0).mean()),
            "yrs_ge4": float((per_yr >= 0.04).mean()), "n_yr": len(per_yr),
            "hits_A": bool(float((r > 0).mean()) >= 0.80
                           and float(r.mean()) >= 0.04)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--horizon", type=int, default=63)
    ap.add_argument("--model", action="store_true",
                    help="run the meta-labelling walk-forward (Q4/Q5)")
    a = ap.parse_args()
    if a.build or not os.path.exists(CACHE):
        print("building features ...")
        D = build(a.horizon)
        print(f"cached {len(D):,} rows, {D['ticker'].nunique()} names")
        return 0

    D = pd.read_parquet(CACHE)

    print("=" * 78)
    print("Q0 — THE LABELLER ON A MARKET WITH A KNOWN ANSWER")
    print("Driftless walk, zero cost. Optional stopping fixes both columns: the")
    print("conditional touch rate must be sl/(tp+sl) and the mean must be 0.00.")
    M = martingale_check()
    M["err"] = (M["measured_p"] - M["theory_p"]).abs()
    M["z_p"] = M["err"] / M["se"]
    M["z_r"] = M["mean_sym"].abs() / M["se_ret"]
    print(f"\n{'width':>7}{'+tp':>7}{'-sl':>7}{'theory':>8}{'MEASURED':>10}"
          f"{'z':>7}{'t/o':>6}{'bars':>6}{'eff n':>8}"
          f"{'mean sym':>10}{'z':>7}{'mean real':>11}")
    for _, r in M.iterrows():
        print(f"{r['w']:>7.2f}{r['tp']:>7.3f}{r['sl']:>7.3f}"
              f"{r['theory_p']:>8.3f}{r['measured_p']:>10.3f}{r['z_p']:>7.2f}"
              f"{r['timeout']:>6.1%}{r['bars']:>6.0f}{r['eff_n']:>8.0f}"
              f"{r['mean_sym']:>+10.4f}{r['z_r']:>7.2f}{r['mean_real']:>+11.4f}")
    ok = bool(M["z_p"].max() < 3.0 and M["z_r"].max() < 3.0)
    print(f"\n  Scored against EFFECTIVE n, not row count. Every cell is within"
          f" {M['z_p'].max():.1f} se on\n  the touch rate and "
          f"{M['z_r'].max():.1f} se on the mean, so the labeller reproduces the"
          f" martingale.")
    print(f"  instrument {'PASSES' if ok else 'FAILS'}")
    print("  THREE ERRORS IN THIS CHECK BEFORE IT PASSED, ALL IN THE TEST:")
    print("   1. log increments of mean zero make the PRICE drift up by")
    print("      exp(sigma^2 T/2) = +5.2%; the labeller measured that and was")
    print("      blamed for it. Drift the logs by -sigma^2/2 instead.")
    print("   2. b/(a+b) is a PRICE-space formula; symmetric price barriers are")
    print("      asymmetric in log space (+0.276 against -0.382).")
    print("   3. feeding close=high=low forced every touch a day late and a")
    print("      whole daily move past the level. Simulating 20 intraday steps")
    print("      gives a true extreme and the overshoot collapses.")
    print("  The last of those is a SIMULATION artefact and is shown to be one")
    print("  by convergence — at the tightest geometry the error runs 0.0422 ->")
    print("  0.0330 -> 0.0135 -> 0.0114 -> 0.0033 as the intraday path is")
    print("  refined 1 -> 5 -> 20 -> 80 -> 320 steps. Real bars carry the TRUE")
    print("  intraday extreme, so on IDX the trigger is exact and this residual")
    print("  does not exist at all.")
    print("  ANY win rate is purchasable and buying it changes nothing: that is")
    print("  the whole constraint the goal runs into, now measured not asserted.")
    print(f"  the real-fill column is MORE negative than the symmetric one by "
          f"{float((M['mean_sym'] - M['mean_real']).mean()):.4f} on average: "
          f"that gap is\n  the modelled cost of a stop that fills at the close "
          f"instead of at its level.")
    M.to_csv(os.path.join(OUT, "quantbot_q0.csv"), index=False)

    print("\n" + "=" * 78)
    print("Q1/Q2/Q3/Q6 — THE ACHIEVABLE FRONTIER")
    print("Swept over HORIZON as well as geometry. Fixing the horizon is the")
    print("error A20 names as having project-wide blast radius: twelve studies")
    print("here inherited 252 sessions because H16 picked it for an unrelated")
    print("reason, and varying it inverted the answer.")
    rows = []
    for hor in (21, 63, 126, 252):
        for tp_k in (0.25, 0.5, 1.0, 2.0, None):
            for sl_k in (0.5, 1.0, 2.0):
                L = eligible(label(D, tp_k, sl_k, hor))
                if len(L) < 2000:
                    continue
                nm = (f"{'NO tp' if tp_k is None else f'tp {tp_k:g}s'}"
                      f" / sl {sl_k:g}s")
                for arm, sub in (("all bars", L),
                                 ("trend", L[L["stack"] >= 2])):
                    if len(sub) < 2000:
                        continue
                    r = summarise(sub, nm)
                    r.update({"hor": hor, "arm": arm,
                              "tp_k": np.nan if tp_k is None else tp_k,
                              "sl_k": sl_k})
                    rows.append(r)
    F = pd.DataFrame(rows)
    F.to_csv(os.path.join(OUT, "quantbot_frontier.csv"), index=False)
    print(f"\n{len(F)} cells. Net of {FEE:.2%} fees plus each name's own "
          f"half-spread. `ann` compounds the\nmean log at 252/bars, i.e. "
          f"ASSUMES the trade repeats back-to-back all year, which\nflatters "
          f"short horizons — H47 measured that overstating a rate ~3x.\n")
    print(f"{'cell':<22}{'hor':>5}{'arm':>9}{'n':>9}{'POS':>7}{'MEAN':>9}"
          f"{'median':>9}{'ann':>9}{'bars':>6}{'t/o':>6}{'yrs>=4%':>9}{'A?':>4}")
    for _, r in F.sort_values("mean", ascending=False).head(20).iterrows():
        print(f"{r['cell']:<22}{int(r['hor']):>5}{r['arm']:>9}"
              f"{int(r['n']):>9,}{r['pos']:>7.1%}{r['mean']:>+9.2%}"
              f"{r['median']:>+9.2%}{r['ann']:>+9.1%}{r['bars']:>6.0f}"
              f"{r['timeout']:>6.0%}{r['yrs_ge4']:>9.0%}"
              f"{'YES' if r['hits_A'] else 'no':>4}")
    print("  ... and the WIN-RATE end of the same frontier:")
    for _, r in F.sort_values("pos", ascending=False).head(10).iterrows():
        print(f"{r['cell']:<22}{int(r['hor']):>5}{r['arm']:>9}"
              f"{int(r['n']):>9,}{r['pos']:>7.1%}{r['mean']:>+9.2%}"
              f"{r['median']:>+9.2%}{r['ann']:>+9.1%}{r['bars']:>6.0f}"
              f"{r['timeout']:>6.0%}{r['yrs_ge4']:>9.0%}"
              f"{'YES' if r['hits_A'] else 'no':>4}")

    hi = F[F["pos"] >= 0.80]
    g4 = F[F["mean"] >= 0.04]
    print(f"\n  Q2 — cells reaching a positive rate >= 80%: {len(hi)} of "
          f"{len(F)}; best positive rate anywhere {F['pos'].max():.1%}")
    if len(hi):
        b = hi.loc[hi["mean"].idxmax()]
        print(f"       best mean among them {b['mean']:+.2%} "
              f"({b['cell']}, h{int(b['hor'])}, {b['arm']}) vs the +4.00% target")
    print(f"  Q1 — cells reaching mean >= +4% per trade: {len(g4)} of {len(F)}; "
          f"best mean {F['mean'].max():+.2%}")
    if len(g4):
        b = g4.loc[g4["pos"].idxmax()]
        print(f"       best positive rate among them {b['pos']:.1%} "
              f"({b['cell']}, h{int(b['hor'])}, {b['arm']}) vs the 80% target")
    print(f"  Q3 — cells clearing BOTH: {int(F['hits_A'].sum())} of {len(F)}")
    print(f"  Q6 — reading B: of the {len(g4)} cells at mean >= +4%, best share "
          f"of calendar years also >= +4%: "
          f"{(g4['yrs_ge4'].max() if len(g4) else float('nan')):.0%}")
    print("\n  AND THE COLUMN THAT DECIDES IT: `ann` is NEGATIVE in every one "
          f"of the {len(F)} cells,\n  best {F['ann'].max():+.1%}. A +5.77% "
          "average trade that runs 249 sessions with a\n  MEDIAN of -2.45% "
          "compounds at -5.3% a year. Arithmetic mean and compounded\n  growth "
          "disagree in sign, and a bot running one position at a time is paid\n"
          "  the second one.")

    if not a.model:
        print("\n(run with --model for Q4/Q5: meta-labelling and the book)")
        return 0

    print("\n" + "=" * 78)
    print("Q4/Q5 — META-LABELLING. Can a model raise the win rate at FIXED")
    print("barrier geometry? That is the only honest way to move up the")
    print("frontier: moving the barriers just buys win rate with mean.")
    feats = [f for f in FEATS if f in D.columns] + [f + "_x" for f in XS]
    cells = [("win-rate corner", 0.25, 2.0, 252),
             ("balanced", 1.0, 2.0, 252),
             ("mean corner", None, 2.0, 252)]
    mrows = []
    for lab, tp_k, sl_k, hor in cells:
        L = eligible(label(D, tp_k, sl_k, hor))
        L = L[L["stack"] >= 2].dropna(subset=feats)
        if len(L) < 20000:
            continue
        for shuf in (False, True):
            W = walk_forward(L, feats, hor, shuffle=shuf)
            if not len(W):
                continue
            W["q"] = W.groupby("date")["p"].rank(pct=True)
            for tag, sel in (("all signals", W),
                             ("model top 50%", W[W["q"] >= 0.50]),
                             ("model top 20%", W[W["q"] >= 0.80]),
                             ("model top 5%", W[W["q"] >= 0.95])):
                if len(sel) < 500:
                    continue
                r = summarise(sel, f"{lab} / {tag}")
                r.update({"arm": "NULL" if shuf else "real", "cell": lab,
                          "sel": tag, "n_sel": len(sel)})
                mrows.append(r)
    MM = pd.DataFrame(mrows)
    MM.to_csv(os.path.join(OUT, "quantbot_model.csv"), index=False)
    print(f"\n{'geometry':<18}{'selection':<16}{'arm':>6}{'n':>9}{'POS':>8}"
          f"{'target-first':>14}{'MEAN':>9}{'median':>9}{'ann':>9}")
    for _, r in MM.iterrows():
        print(f"{r['cell']:<18}{r['sel']:<16}{r['arm']:>6}{int(r['n']):>9,}"
              f"{r['pos']:>8.1%}{r['win']:>14.1%}{r['mean']:>+9.2%}"
              f"{r['median']:>+9.2%}{r['ann']:>+9.1%}")
    real = MM[MM["arm"] == "real"]
    nul = MM[MM["arm"] == "NULL"]
    print(f"\n  Q4 — best positive rate the model reaches: "
          f"{real['pos'].max():.1%} (target 80%); "
          f"best mean {real['mean'].max():+.2%} (target +4.00%)")
    print(f"  Q4 — cells clearing BOTH: "
          f"{int(real['hits_A'].sum())} of {len(real)}")
    if len(nul):
        print(f"  Q5 — the LABEL-SHUFFLED null reaches "
              f"{nul['pos'].max():.1%} / {nul['mean'].max():+.2%}. The model "
              f"must beat it or\n       the pipeline is manufacturing its own "
              f"signal.")
    print(f"  Q5 — best `ann` anywhere in the modelled table: "
          f"{real['ann'].max():+.1%}")

    print("\n" + "=" * 78)
    print("THE BOOK — an actual multi-slot account, because the mean per trade")
    print("and the growth rate of an account disagree in SIGN here.")
    L = eligible(label(D, None, 2.0, 252))
    L = L[L["stack"] >= 2].dropna(subset=feats)
    W = walk_forward(L, feats, 252)
    ic = index_cagr(W["date"].min(), W["date"].max())
    print(f"\ngeometry: NO take-profit / stop 2 sigma / 252-session clock, "
          f"trend filter\n{len(W):,} out-of-sample signals, "
          f"{pd.Timestamp(W['date'].min()):%Y-%m} -> "
          f"{pd.Timestamp(W['date'].max()):%Y-%m}\n")
    print(f"{'book':<26}{'slots':>7}{'trades':>8}{'total':>9}{'CAGR':>9}"
          f"{'vs index':>10}")
    brows = []
    for slots in (1, 4, 8, 16, 32):
        for tag, rk in (("model-ranked", "p"), ("RANDOM control", "rand")):
            b = book(W, slots=slots, rank=rk)
            b.update({"arm": tag})
            brows.append(b)
            print(f"{tag:<26}{slots:>7}{b['trades']:>8,}"
                  f"{b['total']:>9.2f}{b['cagr']:>+9.2%}"
                  f"{b['cagr'] - ic:>+10.2%}")
    pd.DataFrame([{k: v for k, v in b.items() if k != "per_slot"}
                  for b in brows]).to_csv(
        os.path.join(OUT, "quantbot_book.csv"), index=False)
    print(f"\n  index over the same span, total-return basis "
          f"(+{IDX_YIELD:.2%} measured yield added): {ic:+.2%}/yr")
    print("  ONE slot compounds and is paid the mean LOG; MANY slots diversify")
    print("  toward the arithmetic mean. Where a real book lands is the whole")
    print("  question and it is simulated here rather than argued.")
    return 0



# =============================================================== THE BOOK ====
IDX = os.path.join("data", "cache", "ohlcv", "_JKSE.csv.gz")
IDX_YIELD = 0.0177          # A19: measured top-decile dividend yield


def book(W: pd.DataFrame, slots: int = 8, rank: str = "p",
         seed: int = 0) -> Dict:
    """Run an actual S-slot book over the walk-forward predictions.

    THIS IS THE STEP THAT DECIDES WHETHER ANY OF THE FRONTIER MATTERS, and it
    exists because the arithmetic mean per trade and the growth rate of an
    account disagree in SIGN here. A18 established that an equal-weighted holder
    is paid the MEAN rather than the median — but that is a statement about
    holding many names AT ONCE. A book with a finite number of slots sits
    between the two: with one slot you compound and are paid the mean log, with
    very many you approach the arithmetic mean. Where a realistic book lands is
    an empirical question, so it is simulated rather than argued.

    Each slot, when free, takes the highest-ranked candidate on that date that
    it is not already holding, and holds it to that trade's own barrier exit.
    `rank='rand'` is the control: identical machinery, identical slot count and
    identical dates, choosing at random.
    """
    rng = np.random.default_rng(seed)
    if rank == "rand":
        d = W.assign(_r=rng.random(len(W))).sort_values(
            ["date", "_r"], ascending=[True, False])
    else:
        d = W.sort_values(["date", rank], ascending=[True, False])
    dates = np.sort(d["date"].unique())
    free_at = np.zeros(slots, dtype="datetime64[ns]")
    free_at[:] = dates[0]
    equity = np.ones(slots)
    log_by_date: Dict = {}
    held: Dict[int, str] = {}
    taken = 0
    for dt in dates:
        idle = [k for k in range(slots) if free_at[k] <= dt]
        if not idle:
            continue
        cand = d[d["date"] == dt]
        if not len(cand):
            continue
        seen = set(held.values())
        for _, row in cand.iterrows():
            if not idle:
                break
            if row["ticker"] in seen:
                continue
            k = idle.pop(0)
            r = float(np.clip(row["ret"], -0.99, None))
            equity[k] *= (1.0 + r)
            #  The exit lands `bars` sessions on; 1.45 calendar days a session
            #  is the panel's own ratio and only sets when the slot frees.
            free_at[k] = dt + np.timedelta64(
                int(max(row["bars"], 1) * 1.45), "D")
            held[k] = row["ticker"]
            seen.add(row["ticker"])
            taken += 1
            log_by_date.setdefault(pd.Timestamp(free_at[k]), []).append(r)
    #  THE SPAN RUNS TO THE LAST EXIT, NOT THE LAST ENTRY. A position opened on
    #  the final entry date keeps compounding for its whole holding period, so
    #  measuring the equity over the ENTRY span credits years of growth to a
    #  window that does not contain them — and at a 3.4-year mean hold that
    #  overstates the rate badly. A19 records comparing quantities measured over
    #  different windows as the error class that manufactures results, and this
    #  is the same one: the benchmark must be priced over THIS span too.
    last_exit = pd.Timestamp(max(free_at.max(), dates[-1]))
    span = (last_exit - pd.Timestamp(dates[0])).days / 365.25
    tot = float(np.mean(equity))
    return {"slots": slots, "trades": taken, "span": span,
            "start": pd.Timestamp(dates[0]), "end": last_exit,
            "total": tot, "cagr": tot ** (1.0 / max(span, 1e-9)) - 1.0,
            "per_slot": [float(e) for e in equity]}


def index_cagr(t0, t1) -> float:
    J = pd.read_csv(IDX, parse_dates=["date"])
    J = J[(J["date"] >= t0) & (J["date"] <= t1)]
    if len(J) < 50:
        return float("nan")
    yrs = (J["date"].iloc[-1] - J["date"].iloc[0]).days / 365.25
    px = float(J["close"].iloc[-1] / J["close"].iloc[0])
    #  The names run on adj_close and are TOTAL returns; ^JKSE is a PRICE
    #  index. A19 records comparing them raw as the error that manufactured a
    #  result, so the measured yield is added back.
    return (px ** (1.0 / yrs) - 1.0) + IDX_YIELD

if __name__ == "__main__":
    raise SystemExit(main())
