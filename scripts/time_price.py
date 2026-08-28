#!/usr/bin/env python3
"""H31/H32/H33 — the time-cycle premise, the price×time cone, and the flip rule.

    python3 scripts/time_price.py cycles     # H31  do turning points recur on a schedule?
    python3 scripts/time_price.py cone       # H32  joint (how far, how long) from a state
    python3 scripts/time_price.py flips      # H33  is a Hull/EMA flip a tradeable trigger?

WHY THESE THREE, AND WHY IN THIS ORDER.

The request is for an Astronacci-style read: technical analysis gives a
horizontal price target, a time method gives the date, and the two intersect at
"this price on this date". That decomposes into exactly two claims and they are
not equally likely to be true:

  T1  TURNING POINTS RECUR ON A SCHEDULE, so a future date is forecastable from
      past dates alone. This is the strong claim and it is what makes the method
      distinctive. `cycles` tests it.
  T2  FROM A DEFINED STATE, the joint distribution of (how far price runs, how
      long it takes) is tight enough to quote as a range with odds. This is the
      weak claim, it needs no cycle to be true, and it is what an honest
      "price target by date" actually is. `cone` measures it.

If T1 fails, T2 still stands on its own and the deliverable becomes a measured
CONE — a price band crossed with a date band, carrying its own hit rate —
rather than a point. If T1 holds, the cone narrows around the cycle date. Either
way the output has the shape the request asked for; only the width changes.

PRE-REGISTERED PREDICTIONS, WRITTEN BEFORE ANY CELL WAS SCORED
--------------------------------------------------------------
P1  (cycles) T1 FAILS. Inter-pivot intervals will show no useful memory: the
    R² of predicting the next interval from the previous ones will be under
    0.05, and the observed interval dispersion will sit inside the null built
    from block-bootstrapped returns of the same names. A price series with
    volatility clustering and no cycle at all still produces alternating swings
    whose spacing LOOKS rhythmic, which is the whole reason this needs a null
    rather than an eyeball.
P2  (cycles) The month-of-year pivot distribution will be indistinguishable
    from its null. Ramadan/Idul Fitri liquidity effects are the one real
    candidate for a genuine calendar effect and even that moves ~11 days a year
    against the Gregorian calendar, so it cannot produce a fixed-date cycle.
P3  (cone) T2 HOLDS, weakly and usefully. P(touch +10% within a year) will be
    well above half and P(touch 2x) near the 12% base H26 measured, and the
    time-to-touch distribution will be WIDE — an interquartile range spanning
    a factor of three or more. So a date can be quoted as a band and never as
    a day.
P4  (cone) Conditioning on the EMA stack state will move the TIME distribution
    less than it moves the PRICE distribution. Trend states are about direction;
    nothing in them encodes a clock.
P5  (flips) NO flip rule compounds positively in both halves after cost. H30
    already found this for the triple-EMA cross, and reports/hullut_*.csv found
    the published Hull Suite + UT Bot rule losing to buy-and-hold on 84 IDX
    names in 240 of 240 grid cells and in all five walk-forward folds. A Hull
    slope flip is the same family.

THE NULL IS THE FIRST STATISTIC, NOT THE LAST. This repo has now recorded seven
occasions where reading a number against zero rather than against its own
permutation null produced a confident wrong answer, so every table below ships
with the matched null beside it.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

PANEL = os.path.join("data", "spine", "price_panel.parquet")
OUT = "reports"

MIN_TV = 1e9              # Rp/day, the repo's standard eligibility floor
MIN_BARS = 400            # a name needs enough history for a 252-bar forward
HORIZON = 252             # one year, the horizon every other study here uses
UP = (1.05, 1.10, 1.20, 1.50, 2.00)
DOWN = (0.90, 0.80, 0.67, 0.50)
TARGETS = UP + DOWN
DRAWS = 200
SEED = 20260828


# ============================================================== loading ======
def load(holdout: bool = False) -> pd.DataFrame:
    P = pd.read_parquet(PANEL, columns=["date", "ticker", "close", "adj_close",
                                        "log_turnover", "tradeable", "holdout"])
    P["date"] = pd.to_datetime(P["date"])
    if not holdout:
        P = P[~P["holdout"].astype(bool)]
    P = P[(P["adj_close"] > 0) & (P["close"] > 0)]
    return P.sort_values(["ticker", "date"]).reset_index(drop=True)


def eligible(P: pd.DataFrame) -> pd.Series:
    """A bar you could actually have BOUGHT on. A19's lesson: eligibility is a
    condition for entering, never a filter applied along the forward path."""
    return (P["tradeable"].astype(bool)
            & (np.exp(P["log_turnover"].fillna(-np.inf)) >= MIN_TV))


# =========================================================== H31 cycles ======
def pivots(p: np.ndarray, k: float) -> np.ndarray:
    """ZigZag turning points: indices where the series reverses by >= k.

    Alternating by construction — a high is only confirmed by a k% fall from it
    — so the output is a strict high/low/high sequence and the gaps between
    consecutive entries are the half-cycles a time method claims to forecast.
    """
    n = len(p)
    if n < 3:
        return np.empty(0, dtype=np.int64)
    out: List[int] = []
    hi_i = lo_i = 0
    hi = lo = p[0]
    up: object = None                          # direction of the leg in progress
    for i in range(1, n):
        v = p[i]
        if up is None:
            #  BEFORE THE FIRST LEG IS CONFIRMED BOTH EXTREMES HAVE TO BE
            #  CARRIED. A first version tracked only the previous bar here, so
            #  the detector could not start until a SINGLE bar moved k -- which
            #  a smooth series never does. It silently found no pivots at all
            #  in any name without a one-day limit move, and read zero turns in
            #  a pure sine wave. The test that caught it is the one asserting a
            #  periodic series reads as a cycle: a detector that cannot find a
            #  cycle it is shown proves nothing by failing to find one in IDX.
            if v > hi:
                hi_i, hi = i, v
            if v < lo:
                lo_i, lo = i, v
            if v <= hi * (1.0 - k):
                out.append(hi_i)
                up, lo_i, lo = False, i, v
            elif v >= lo * (1.0 + k):
                out.append(lo_i)
                up, hi_i, hi = True, i, v
            continue
        if up:
            if v > hi:
                hi_i, hi = i, v
            elif v <= hi * (1.0 - k):
                out.append(hi_i)
                up, lo_i, lo = False, i, v
        else:
            if v < lo:
                lo_i, lo = i, v
            elif v >= lo * (1.0 + k):
                out.append(lo_i)
                up, hi_i, hi = True, i, v
    return np.asarray(out, dtype=np.int64)


def gaps_of(iv: np.ndarray) -> np.ndarray:
    """Sessions between confirmed turns.

    The first entry `pivots` returns is the ANCHOR — the running extreme at the
    start of the series, which is where the record happens to begin rather than
    a turn the market made. It is dropped, so every gap measured is a genuine
    half-cycle between two reversals.
    """
    return np.diff(iv[1:]) if len(iv) > 2 else np.empty(0, dtype=np.int64)


def interval_stats(gaps: np.ndarray) -> Dict[str, float]:
    """Everything a cycle claim would have to move. CV is regularity; the AR
    and R2 terms are whether knowing the last gaps narrows the next one."""
    g = gaps[gaps > 0].astype(float)
    if len(g) < 30:
        return {}
    lg = np.log(g)
    out = {"n": float(len(g)), "median": float(np.median(g)),
           "cv": float(np.std(g, ddof=1) / np.mean(g))}
    #  AR(1) on log gaps, and the R2 of a two-lag linear forecast. If turning
    #  points arrive on a schedule these are the numbers that show it.
    a, b = lg[:-1], lg[1:]
    out["ar1"] = float(np.corrcoef(a, b)[0, 1])
    if len(lg) >= 60:
        X = np.column_stack([np.ones(len(lg) - 2), lg[:-2], lg[1:-1]])
        y = lg[2:]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss = float(np.sum((y - y.mean()) ** 2))
        out["r2"] = float(1.0 - np.sum(resid ** 2) / ss) if ss > 0 else np.nan
        #  The number a practitioner would feel: how many sessions wide is the
        #  50% band for the next turn, with and without the forecast.
        out["sd_uncond"] = float(np.std(y, ddof=1))
        out["sd_cond"] = float(np.std(resid, ddof=1))
    return out


def block_bootstrap(r: np.ndarray, rng, block: int = 21) -> np.ndarray:
    """Circular moving-block resample of returns. Preserves the marginal
    distribution and short-range volatility clustering — which is what makes
    a random walk LOOK cyclic — and destroys any deterministic period."""
    n = len(r)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=nb)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
    return r[idx[:n]]


def cycles(P: pd.DataFrame, k: float = 0.10, draws: int = 100,
           null_names: int = 150, block: int = 21) -> Dict:
    """Observed pivot spacing against spacing produced by a series with the
    same returns in a scrambled order.

    The null runs on a random SUBSET of names rather than all of them: the cost
    is names x draws pivot scans and the statistic is pooled over hundreds of
    thousands of gaps either way, so the null's own sampling error is set by
    the gap count, not by the name count. 150 x 100 already gives a null sd
    small enough to resolve the effect sizes a cycle claim would need.
    """
    rng = np.random.default_rng(SEED)
    keep = P.groupby("ticker")["adj_close"].transform("size") >= MIN_BARS
    P = P[keep]
    pool = np.sort(P["ticker"].unique())
    sub = set(rng.choice(pool, size=min(null_names, len(pool)), replace=False))
    obs_gaps: List[np.ndarray] = []          # matched to the null's name set
    all_gaps: List[np.ndarray] = []          # descriptive, whole panel
    obs_dates: List[pd.Timestamp] = []
    null_gaps: List[List[np.ndarray]] = [[] for _ in range(draws)]
    names = 0
    for tk, g in P.groupby("ticker", sort=False):
        p = g["adj_close"].to_numpy(float)
        iv = pivots(p, k)
        if len(iv) < 8:
            continue
        names += 1
        all_gaps.append(gaps_of(iv))
        obs_dates.extend(g["date"].to_numpy()[iv[1:]])
        if tk not in sub:
            continue
        #  THE OBSERVED STATISTIC IS COMPUTED ON THE SAME NAMES THE NULL USES.
        #  Pooling the observation over 891 names and the null over 150 would
        #  compare a tighter sampling distribution to a looser one and the z
        #  would measure the name count, not the effect.
        obs_gaps.append(gaps_of(iv))
        r = np.diff(np.log(p))
        for d in range(draws):
            sim = np.concatenate([[p[0]], p[0] * np.exp(
                np.cumsum(block_bootstrap(r, rng, block)))])
            jv = pivots(sim, k)
            if len(jv) >= 8:
                null_gaps[d].append(gaps_of(jv))

    obs = interval_stats(np.concatenate(obs_gaps))
    obs["panel_median"] = float(np.median(np.concatenate(all_gaps)))
    nulls: Dict[str, List[float]] = {}
    for d in range(draws):
        if not null_gaps[d]:
            continue
        s = interval_stats(np.concatenate(null_gaps[d]))
        for kk, vv in s.items():
            nulls.setdefault(kk, []).append(vv)

    #  P2: the calendar. Pivot months against the months the panel offers, so
    #  an unbalanced panel cannot masquerade as seasonality.
    dt = pd.DatetimeIndex(obs_dates)
    piv_m = pd.Series(dt.month).value_counts().sort_index()
    all_m = P["date"].dt.month.value_counts().sort_index()
    share = (piv_m / piv_m.sum()) / (all_m / all_m.sum())
    return {"obs": obs, "nulls": nulls, "names": names, "k": k, "block": block,
            "n_pivots": len(dt), "month_ratio": share}


def block_sweep(P: pd.DataFrame, k: float = 0.10) -> pd.DataFrame:
    """THE CHECK THAT DISCRIMINATES A CYCLE FROM VOLATILITY CLUSTERING.

    Real pivot gaps carry more memory than a 21-day block bootstrap produces.
    Two things could do that: a genuine period, or the fact that a quiet regime
    yields several long gaps in a row and a violent one several short gaps.
    Only the second is undone by lengthening the block, so if the observed
    excess shrinks as the block grows toward a year, the memory was volatility
    clustering wearing a cycle's clothes.
    """
    rows = []
    for b in (5, 21, 63, 252):
        R = cycles(P, k=k, draws=60, null_names=120, block=b)
        m, sd, z = _z(R["obs"]["r2"], R["nulls"].get("r2", []))
        rows.append({"block": b, "obs_r2": R["obs"]["r2"], "null_r2": m,
                     "excess": R["obs"]["r2"] - m, "z": z})
    return pd.DataFrame(rows)


def spectrum(path: str = os.path.join("data", "cache", "ohlcv", "_JKSE.csv.gz"),
             draws: int = 500, block: int = 252) -> Dict:
    """THE OTHER HALF OF THE CLAIM: a FIXED period, not a predictable spacing.

    A method that projects "the next turn lands N sessions after the last" is
    tested by `cycles`. A method that says "this market turns every N sessions,
    full stop" is a claim about a fixed frequency, and the test for that is a
    periodogram: if a dominant period exists, its power stands above what the
    same returns produce in a scrambled order.

    Detrending is not optional. Log price rises over 25 years, and a trend is
    all low-frequency power — undetrended, the periodogram's maximum is always
    at the longest period and says nothing about cycles.
    """
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d["adj_close"] > 0)].sort_values("date").reset_index(drop=True)
    y = np.log(d["adj_close"].to_numpy(float))
    n = len(y)
    t = np.arange(n, dtype=float)
    A = np.column_stack([np.ones(n), t])

    def power(series: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(A, series, rcond=None)
        z = series - A @ beta
        z = z - z.mean()
        f = np.fft.rfft(z * np.hanning(n))
        return (np.abs(f) ** 2)[1:]

    per = n / np.arange(1, n // 2 + 1, dtype=float)
    band = (per >= 10) & (per <= 1260)              # ~2 weeks to 5 years
    obs = power(y)[band]
    rng = np.random.default_rng(SEED)
    r = np.diff(y)
    peaks = np.empty(draws)
    for i in range(draws):
        sim = np.concatenate([[y[0]], y[0] + np.cumsum(
            block_bootstrap(r, rng, block))])
        peaks[i] = power(sim)[band].max() / power(sim)[band].mean()
    stat = obs.max() / obs.mean()
    return {"n": n, "start": d["date"].iloc[0], "end": d["date"].iloc[-1],
            "period": float(per[band][int(np.argmax(obs))]),
            "stat": float(stat), "null_mean": float(peaks.mean()),
            "null_sd": float(peaks.std(ddof=1)),
            "p": float((np.sum(peaks >= stat) + 1) / (draws + 1))}


# ============================================================= H32 cone ======
def first_passage(p: np.ndarray, mult: float, horizon: int) -> np.ndarray:
    """Sessions until the path first reaches entry*mult, or -1 inside horizon.

    Strictly forward: the window for bar i is i+1 .. i+horizon, so the entry
    bar itself can never satisfy its own target. A sliding view costs no copy,
    which is what makes a 2.5-million-row panel tractable here. mult < 1 asks
    the downside question, and the downside is not optional — a target price
    without the matching stop distance is half a trade.
    """
    n = len(p)
    out = np.full(n, -1, dtype=np.int64)
    if n < 2:
        return out
    m = min(horizon, n - 1)
    fill = -np.inf if mult >= 1.0 else np.inf
    pad = np.concatenate([p[1:], np.full(m, fill)])
    W = np.lib.stride_tricks.sliding_window_view(pad, m)[:n]
    lvl = (p * mult)[:, None]
    hit = W >= lvl if mult >= 1.0 else W <= lvl
    any_hit = hit.any(axis=1)
    out[any_hit] = hit.argmax(axis=1)[any_hit] + 1
    #  A window that runs off the end of the name's life has no answer; mark it
    #  censored rather than letting a short window read as a miss.
    tail = np.arange(n) + m >= n
    out[tail & ~any_hit] = -2
    return out


def cone(P: pd.DataFrame) -> pd.DataFrame:
    """The joint (how far, how long) table, by state."""
    P = P.copy()
    P["elig"] = eligible(P)
    frames: List[pd.DataFrame] = []
    for tk, g in P.groupby("ticker", sort=False):
        if len(g) < MIN_BARS:
            continue
        p = g["adj_close"].to_numpy(float)
        d = pd.DataFrame({"date": g["date"].to_numpy(),
                          "ticker": tk, "elig": g["elig"].to_numpy()})
        for t in TARGETS:
            d[f"t{int(t * 100)}"] = first_passage(p, t, HORIZON)
        #  The states the chart can actually show, computed the same way the
        #  Pine script computes them so the numbers transfer.
        s = pd.Series(p)
        e50 = s.ewm(span=50, adjust=False, min_periods=50).mean().to_numpy()
        e100 = s.ewm(span=100, adjust=False, min_periods=100).mean().to_numpy()
        e200 = s.ewm(span=200, adjust=False, min_periods=200).mean().to_numpy()
        d["stack"] = (p > e50) & (e50 > e100) & (e100 > e200)
        h = hma(s, 55).to_numpy()
        d["hull_up"] = np.concatenate([[False], h[1:] > h[:-1]])
        hi252 = s.rolling(252, min_periods=252).max().to_numpy()
        d["hi52"] = p / hi252
        d["vol60"] = s.pct_change().rolling(60, min_periods=60).std().to_numpy()
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def cone_row(C: pd.DataFrame, label: str) -> Dict:
    out: Dict[str, object] = {"state": label, "n": len(C)}
    for t in TARGETS:
        col = f"t{int(t * 100)}"
        v = C[col].to_numpy()
        live = v != -2                       # drop censored windows entirely
        hit = v[live] > 0
        out[f"p{int(t * 100)}"] = float(hit.mean()) if live.sum() else np.nan
        w = v[live][hit]
        for q, nm in ((25, "q1"), (50, "med"), (75, "q3")):
            out[f"{col}_{nm}"] = float(np.percentile(w, q)) if len(w) else np.nan
    return out


# =========================================== H32b the closed-form cone =======
#  A ten-by-nine grid of measured cells is not shippable to a chart: it only
#  answers the nine targets it was tabulated for, and encoding 360 constants in
#  Pine invites a transcription error nobody would ever catch. So the grid is
#  reduced to two laws in (distance, volatility, trend state), fitted to the
#  cells and reported WITH their residuals.
#
#  NOTE WHAT THE EXPONENTS ARE NOT. A driftless random walk reaches a log
#  barrier d with per-bar sd s in a time scaling as (d/s)^2. These fit near
#  d^0.9 / s^0.6, which is not that, for two reasons worth stating rather than
#  smoothing over: the series trends and its volatility clusters, and the
#  sample conditions on touching WITHIN 252 sessions. The censoring is why the
#  q3 exponent (0.59) is so much flatter than the q1 exponent (1.07) -- the
#  upper edge of the band is partly the horizon and not the market.
def law_cells(C: pd.DataFrame, deciles: int = 10) -> pd.DataFrame:
    E = C[C["elig"] & C["vol60"].notna()].copy()
    E["vd"] = pd.qcut(E["vol60"], deciles, labels=False, duplicates="drop")
    E["stk"] = E["stack"].fillna(False).astype(bool)
    rows: List[Dict] = []
    for (vd, st), g in E.groupby(["vd", "stk"]):
        s = float(g["vol60"].median())
        for t in TARGETS:
            col = f"t{int(t * 100)}"
            a = g[col].to_numpy()
            live = a != -2
            if live.sum() < 500:
                continue
            hit = a[live] > 0
            w = a[live][hit]
            r = {"vd": int(vd), "stack": int(st), "sig": s, "up": int(t > 1),
                 "d": abs(np.log(t)), "n": int(live.sum()),
                 "p": float(np.clip(hit.mean(), 1e-4, 1 - 1e-4))}
            if len(w) >= 200:
                r["q1"], r["med"], r["q3"] = (float(np.percentile(w, q))
                                              for q in (25, 50, 75))
            rows.append(r)
    return pd.DataFrame(rows)


def design(R: pd.DataFrame) -> np.ndarray:
    """1, log d, (log d)^2, log sigma, log d * log sigma.

    THE FIRST VERSION OF THIS WAS LINEAR IN log d AND IT MISSED BY 11 POINTS AT
    THE DEFAULT TARGET. Over a distance range from +5% to 2x the logit is
    visibly curved, and a straight line through it under-predicted the middle —
    which is exactly where a user sets the dial. The quadratic and the
    distance x volatility interaction take the median error on the upside from
    4.2 probability points to 1.7 and the median time error from 14% to 6%.
    A fit is only as good as the cell the reader actually asks for.
    """
    ld, ls = np.log(R["d"].to_numpy(float)), np.log(R["sig"].to_numpy(float))
    return np.column_stack([np.ones(len(R)), ld, ld ** 2, ls, ld * ls])


def fit_time(R: pd.DataFrame) -> pd.DataFrame:
    R = R.dropna(subset=["med"])
    X = design(R)
    out: List[Dict] = []
    for col in ("q1", "med", "q3"):
        y = np.log(R[col].to_numpy(float))
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = np.exp(np.abs(y - X @ b)) - 1.0
        out.append({"quantile": col, "a": b[0], "b_d": b[1], "c_d2": b[2],
                    "e_s": b[3], "f_ds": b[4],
                    "r2": 1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2),
                    "med_err_pct": 100 * float(np.median(e)),
                    "p90_err_pct": 100 * float(np.quantile(e, 0.9))})
    return pd.DataFrame(out)


def fit_prob(R: pd.DataFrame) -> pd.DataFrame:
    out: List[Dict] = []
    for up in (1, 0):
        S = R[R["up"] == up]
        X = np.column_stack([design(S), S["stack"].to_numpy(float)])
        y = np.log(S["p"] / (1 - S["p"])).to_numpy()
        #  Cells carry wildly different sample sizes; weighting by sqrt(n)
        #  stops a 500-row cell from pulling the fit as hard as a 60,000-row one.
        w = np.sqrt(S["n"].to_numpy(float))
        b, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
        e = np.abs(1.0 / (1.0 + np.exp(-(X @ b))) - S["p"].to_numpy())
        out.append({"side": "up" if up else "down", "a": b[0], "b_d": b[1],
                    "c_d2": b[2], "e_s": b[3], "f_ds": b[4], "g_stack": b[5],
                    "med_err_pp": 100 * float(np.median(e)),
                    "p90_err_pp": 100 * float(np.quantile(e, 0.9))})
    return pd.DataFrame(out)


# ============================================================ H33 flips ======
def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)


def hma(s: pd.Series, n: int) -> pd.Series:
    """Hull moving average, the line the requested chart draws.

    Computed on a SINGLE name's series. Rolling a weighted mean over a pivot
    frame would index by the union of trading days and insert NaN rows a
    suspended name never had — A11's defect, and it silently emptied a table.
    """
    half, sq = max(1, n // 2), max(1, int(round(np.sqrt(n))))
    return wma(2.0 * wma(s, half) - wma(s, n), sq)


def flips(P: pd.DataFrame, hold: int = 60, cost: float = 0.0056) -> pd.DataFrame:
    """Every flip rule the requested chart would draw a label on, scored the
    same way H30 scored the EMA cross: mean, median and MEAN LOG, net of a
    round trip, split into halves."""
    P = P.copy()
    P["elig"] = eligible(P)
    rows: List[Dict] = []
    for tk, g in P.groupby("ticker", sort=False):
        if len(g) < MIN_BARS:
            continue
        s = g["adj_close"].reset_index(drop=True)
        p = s.to_numpy(float)
        n = len(p)
        fwd = np.full(n, np.nan)
        j = np.arange(n) + hold
        ok = j < n
        fwd[ok] = p[j[ok]] / p[ok] - 1.0
        h55, h21 = hma(s, 55).to_numpy(), hma(s, 21).to_numpy()
        e50 = s.ewm(span=50, adjust=False, min_periods=50).mean().to_numpy()
        e100 = s.ewm(span=100, adjust=False, min_periods=100).mean().to_numpy()
        e200 = s.ewm(span=200, adjust=False, min_periods=200).mean().to_numpy()
        states = {
            "hull55 slope up": np.concatenate([[False], h55[1:] > h55[:-1]]),
            "hma21 over hma55": h21 > h55,
            "price>50>100>200": (p > e50) & (e50 > e100) & (e100 > e200),
        }
        el = g["elig"].to_numpy()
        yr = pd.DatetimeIndex(g["date"]).year.to_numpy()
        for nm, st in states.items():
            st = np.nan_to_num(st, nan=False).astype(bool)
            prev = np.concatenate([[False], st[:-1]])
            entry = st & ~prev
            sel = entry & el & np.isfinite(fwd)
            if not sel.any():
                continue
            rows.append(pd.DataFrame({"rule": nm, "ticker": tk,
                                      "year": yr[sel], "fwd": fwd[sel]}))
        #  The benchmark every rule has to beat: holding the same name over the
        #  same window from an eligible bar, with no flip and no toll.
        sel = el & np.isfinite(fwd)
        if sel.any():
            rows.append(pd.DataFrame({"rule": "base (any eligible bar)",
                                      "ticker": tk, "year": yr[sel],
                                      "fwd": fwd[sel]}))
    F = pd.concat(rows, ignore_index=True)
    F["net"] = F["fwd"] - np.where(F["rule"].eq("base (any eligible bar)"),
                                   0.0, cost)
    cut = int(F["year"].median())
    out: List[Dict] = []
    for nm, g in F.groupby("rule"):
        lg = np.log1p(np.clip(g["net"], -0.99, None))
        e = g[g["year"] < cut]
        l = g[g["year"] >= cut]
        out.append({
            "rule": nm, "n": len(g), "mean": g["net"].mean(),
            "median": g["net"].median(), "win": float((g["net"] > 0).mean()),
            "meanlog": float(lg.mean()),
            "early": float(np.log1p(np.clip(e["net"], -0.99, None)).mean()),
            "late": float(np.log1p(np.clip(l["net"], -0.99, None)).mean())})
    return pd.DataFrame(out).sort_values("meanlog", ascending=False)


# ================================================================== main =====
def _z(obs: float, null: List[float]) -> Tuple[float, float, float]:
    a = np.asarray(null, float)
    a = a[np.isfinite(a)]
    if len(a) < 10:
        return np.nan, np.nan, np.nan
    sd = a.std(ddof=1)
    return float(a.mean()), float(sd), float((obs - a.mean()) / sd) if sd else np.nan


def main(argv: List[str]) -> int:
    modes = argv or ["cycles", "cone", "flips"]
    P = load()
    print(f"panel {len(P):,} rows  {P['ticker'].nunique()} names  "
          f"{P['date'].min():%Y-%m-%d} -> {P['date'].max():%Y-%m-%d}\n")

    if "cycles" in modes:
        for k in (0.05, 0.10, 0.20):
            R = cycles(P, k=k)
            o, nl = R["obs"], R["nulls"]
            print(f"=== H31 cycles, zigzag {k:.0%}  "
                  f"{R['names']} names, {R['n_pivots']:,} pivots")
            print(f"{'statistic':<12}{'observed':>10}{'null mean':>11}"
                  f"{'null sd':>9}{'z':>8}")
            for st in ("median", "cv", "ar1", "r2", "sd_cond", "sd_uncond"):
                if st not in o:
                    continue
                m, sd, z = _z(o[st], nl.get(st, []))
                print(f"{st:<12}{o[st]:>10.4f}{m:>11.4f}{sd:>9.4f}{z:>8.2f}")
            if "sd_cond" in o:
                print(f"  50% band for the NEXT turn: "
                      f"+/-{100 * (np.exp(0.674 * o['sd_uncond']) - 1):.0f}% of the "
                      f"median gap unconditionally, "
                      f"+/-{100 * (np.exp(0.674 * o['sd_cond']) - 1):.0f}% knowing "
                      f"the last two gaps")
            print()
        print("month-of-year pivot share / panel share (1.00 = no seasonality)")
        print(R["month_ratio"].round(3).to_string())
        print(f"  range {R['month_ratio'].min():.3f} - "
              f"{R['month_ratio'].max():.3f}\n")
        S = block_sweep(P)
        S.to_csv(os.path.join(OUT, "time_price_blocks.csv"), index=False)
        print("=== H31b is the interval memory a CYCLE or volatility clustering?")
        print(S.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
        sp = spectrum()
        print(f"\n=== H31c a FIXED period in the IHSG? {sp['n']:,} sessions "
              f"{sp['start']:%Y-%m-%d} -> {sp['end']:%Y-%m-%d}")
        print(f"  strongest period {sp['period']:.0f} sessions "
              f"({sp['period'] / 252:.2f} years), peak/mean power {sp['stat']:.2f}"
              f"  null {sp['null_mean']:.2f} +/- {sp['null_sd']:.2f}"
              f"  p = {sp['p']:.3f}")
        print()

    if "cone" in modes:
        C = cone(P)
        C.to_parquet(os.path.join("data", "spine", "cone.parquet"), index=False)
        E = C[C["elig"]]
        rows = [cone_row(E, "base (any eligible bar)"),
                cone_row(E[E["stack"]], "price>50>100>200"),
                cone_row(E[~E["stack"]], "not stacked"),
                cone_row(E[E["hull_up"]], "hull55 rising"),
                cone_row(E[(E["hi52"] >= 0.9625) & (E["vol60"] <= 0.0257)],
                         "strength+calm (H26)")]
        T = pd.DataFrame(rows)
        T.to_csv(os.path.join(OUT, "time_price_cone.csv"), index=False)
        print("=== H32 the cone: P(touch) within 252 sessions")
        for nm, grp in (("UP", UP), ("DOWN", DOWN)):
            cols = ["state", "n"] + [f"p{int(t * 100)}" for t in grp]
            print(f"-- {nm}")
            print(T[cols].to_string(index=False,
                                    float_format=lambda v: f"{v:,.3f}"))
        print("\n=== H32 sessions to touch, GIVEN it touches (q1 / median / q3)")
        for _, r in T.iterrows():
            parts = [f"{int(t * 100)}%: "
                     f"{r[f't{int(t * 100)}_q1']:.0f}/"
                     f"{r[f't{int(t * 100)}_med']:.0f}/"
                     f"{r[f't{int(t * 100)}_q3']:.0f}" for t in TARGETS]
            print(f"  {r['state']:<26}" + "  ".join(parts))
        print("\n=== H32 the ratio the request actually asked for")
        print(f"{'state':<26}{'P(+20%)':>9}{'P(-20%)':>9}{'up/down':>9}"
              f"{'P(2x)':>8}{'P(half)':>9}{'skew':>7}")
        for _, r in T.iterrows():
            print(f"{r['state']:<26}{r['p120']:>9.3f}{r['p80']:>9.3f}"
                  f"{r['p120'] / r['p80']:>9.2f}{r['p200']:>8.3f}"
                  f"{r['p50']:>9.3f}{r['p200'] / r['p50']:>7.2f}")
        print()

    if "law" in modes:
        cp = os.path.join("data", "spine", "cone.parquet")
        C = pd.read_parquet(cp) if os.path.exists(cp) else cone(P)
        R = law_cells(C)
        R.to_csv(os.path.join(OUT, "time_price_cells.csv"), index=False)
        FT, FP = fit_time(R), fit_prob(R)
        FT.to_csv(os.path.join(OUT, "time_price_law_time.csv"), index=False)
        FP.to_csv(os.path.join(OUT, "time_price_law_prob.csv"), index=False)
        print(f"=== H32b closed-form cone, fitted to {len(R)} measured cells")
        print("terms: 1, log d, (log d)^2, log sigma, log d * log sigma [, stack]")
        print("\nlog(sessions to target)")
        print(FT.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
        print("\nlogit P(touch within 252)")
        print(FP.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
        print("\n  stack odds multiplier   up x"
              f"{np.exp(FP.loc[0, 'g_stack']):.2f}   "
              f"down x{np.exp(FP.loc[1, 'g_stack']):.2f}")
        print("  volatility by decile, and the clock it sets")
        E = C[C["elig"] & C["vol60"].notna()]
        print(E.groupby(pd.qcut(E["vol60"], 10, labels=False,
                                duplicates="drop"))["vol60"]
              .median().round(4).to_string())
        print()

    if "flips" in modes:
        F = flips(P)
        F.to_csv(os.path.join(OUT, "time_price_flips.csv"), index=False)
        print("=== H33 flip rules, 60-session hold, net of 56 bps")
        print(F.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
