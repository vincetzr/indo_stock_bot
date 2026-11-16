#!/usr/bin/env python3
"""H64: does the depository's investor-type split identify persistently dumb flow?

REGISTERED FIRST. The specification below is copied from hypotheses.md, H64,
committed in f5b1ac5 BEFORE this file existed and before any KSEI number had
been joined to a price. A change to anything in the CONSTANTS block is a new
hypothesis, not a fix.

    signal    f_retail = Δ local_ID units over month m / held units at m−1
              f_foreign = the same for foreign_total           (H64b)
              parity    = sec_num mod 2                         (P0, null)
    decision  the later of the two files' available_on(stamp, lag=1); stale
              (> 31 days after the state date) is dropped
    label     total return from the decision close to k own-bars later,
              k ∈ {21, 63, 126}, confirmatory k = 63
    filters   consecutive months; |Δlog sec_num|, |Δlog held| ≤ 0.005;
              [coverage ≥ 0.90 at both ends — REMOVED by amendment A, kept
              behind --registered]; mom12_1 defined; tradeable bar on the
              decision date; 60-session median value ≥ Rp 1bn; ≥ 30 names
    controls  mom12_1, rev1, log_turnover, vol60, log mcap, the class's level
              at m−1, IDX-IC sector dummies
    stats     per-month Spearman IC on the neutralised signal; HAC t with
              ⌈k/21⌉ lags; 10,000 within-month permutation draws; liquidity
              terciles; halves; drop-largest-year; long-only quintile excess
              net of 0.56% + 2 × half-spread per rebalance
    gate      predicted sign, p < 0.000136 AND |t| > 3.81, both halves, no
              tercile flip, drop-largest-year keeps sign with |t| > 2, net
              long-only excess > 0 in both halves
    holdout   decisions after the price panel's own holdout cut; straddling
              label windows purged; spent once and only if the gate passes

Usage
-----
    python3 scripts/ksei_flow.py                 # in-sample, holdout untouched
    python3 scripts/ksei_flow.py --draws 2000    # faster, cannot resolve the bar
    python3 scripts/ksei_flow.py --spend-holdout # refused unless a gate passed

WHAT THE LONG-ONLY LEG MEANS. A5 forbids shorting, so "taking the other side of
retail" can only mean HOLDING what retail sold and not holding what it bought.
The tradeable quantity is therefore the favoured quintile minus the
equal-weighted universe, with the full round trip charged to the quintile every
rebalance and nothing charged to the universe. That is generous to the
benchmark, which is the direction a cost assumption should err.

WHAT A DELISTED NAME DOES HERE. A name whose own bars end inside the label
window has no label and drops out of that month. That is the repo's standing
convention (``features.price.forward_return``) and it biases toward survivors;
it is stated, not fixed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.linalg import qr as _pivoted_qr
from scipy.stats import rankdata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.data import ksei                                    # noqa: E402
from idxbot.spine.quality import tick_grid                      # noqa: E402
from flow_ic import newey_west_t                                # noqa: E402

# ------------------------------------------------------------ CONSTANTS ---- #
PANEL = os.path.join("data", "spine", "price_panel.parquet")
#: Written every run; scripts/ownership.py reads its caveat from here rather
#: than from numbers typed into a format string (A47).
RESULT = os.path.join("reports", "ksei_flow.json")
SECTORS = os.path.join("data", "reference", "idx_classification.parquet")
HORIZONS = (21, 63, 126)
CONFIRM_K = 63
LAG = 1
LAG_SENSITIVITY = 5
STALE_DAYS = 31
DLOG_MAX = 0.005
COVER_MIN = 0.90
LIQ_MIN = 1e9
MIN_NAMES = 30
FEE = 0.0056
DRAWS = 10_000
TRIALS_AFTER = 368
BONFERRONI = 0.05 / TRIALS_AFTER
T_BAR = 3.81
N_Q = 5
#: signal -> predicted sign, and the share level it is neutralised on
SIGNALS = {"f_retail": -1, "f_foreign": +1, "parity": 0}
LEVEL = {"f_retail": "retail_lvl", "f_foreign": "foreign_lvl", "parity": None}
PRICE_CONTROLS = ["mom12_1", "rev1", "log_turnover", "vol60"]
# --------------------------------------------------------------------------- #


def ksei_frame(P: pd.DataFrame, sessions: pd.DatetimeIndex,
               lag: int) -> pd.DataFrame:
    """One row per (ticker, state month) with flows, levels and a decision date."""
    P = P.sort_values(["ticker", "month_end"]).reset_index(drop=True)
    grid = pd.DatetimeIndex(sorted(P["month_end"].unique()))
    pos = pd.Series(np.arange(len(grid)), index=grid)
    created = P.groupby("month_end")["created"].first()
    avail = pd.Series({m: ksei.available_on(created[m].to_pydatetime(),
                                            sessions, lag) for m in grid})

    P["held"] = (P["local_total"] + P["foreign_total"]).astype(float)
    P["_k"] = P["month_end"].map(pos).astype(int)
    g = P.groupby("ticker")
    prev = {c: g[c].shift(1) for c in ("_k", "local_ID", "foreign_total",
                                       "held", "sec_num", "month_end")}
    out = P[["ticker", "month_end", "sec_num", "price"]].copy()
    out["consecutive"] = (P["_k"] - prev["_k"]) == 1
    with np.errstate(divide="ignore", invalid="ignore"):
        out["f_retail"] = (P["local_ID"] - prev["local_ID"]) / prev["held"]
        out["f_foreign"] = ((P["foreign_total"] - prev["foreign_total"])
                            / prev["held"])
        out["retail_lvl"] = prev["local_ID"] / prev["held"]
        out["foreign_lvl"] = prev["foreign_total"] / prev["held"]
        out["dlog_sec"] = np.log(P["sec_num"] / prev["sec_num"])
        out["dlog_held"] = np.log(P["held"] / prev["held"])
        out["cover"] = P["held"] / P["sec_num"]
        out["cover_prev"] = prev["held"] / prev["sec_num"]
        out["log_mcap"] = np.log(P["sec_num"].astype(float)
                                 * P["price"].astype(float))
    out["parity"] = (P["sec_num"].astype("int64") % 2).astype(float)
    # Not a registered filter: printed as a post-result robustness line only.
    out["xfer"] = ksei.transfer_flags(P)
    a_now = P["month_end"].map(avail)
    a_prev = prev["month_end"].map(avail)
    out["decision"] = pd.to_datetime(
        np.where(a_prev > a_now, a_prev, a_now))
    out["stale_days"] = (out["decision"] - out["month_end"]).dt.days
    return out


def attach_prices(K: pd.DataFrame, D: pd.DataFrame) -> pd.DataFrame:
    """Controls at the decision bar and forward returns, WITHIN ticker."""
    D = D.sort_values(["ticker", "date"])
    rows = []
    want = K.dropna(subset=["decision"])
    by = {t: g for t, g in want.groupby("ticker")}
    for t, g in D.groupby("ticker", sort=False):
        if t not in by:
            continue
        dates = g["date"].to_numpy()
        adj = g["adj_close"].to_numpy(dtype=float)
        idx = pd.Index(dates)
        k = by[t]
        loc = idx.get_indexer(k["decision"].to_numpy())
        ok = loc >= 0
        if not ok.any():
            continue
        kk = k[ok].copy()
        li = loc[ok]
        for c in ("tradeable", "close") + tuple(PRICE_CONTROLS):
            kk[c] = g[c].to_numpy()[li]
        for h in HORIZONS:
            j = li + h
            inside = j < len(adj)
            f = np.full(len(li), np.nan)
            ex = np.full(len(li), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
            f[inside] = adj[j[inside]] / adj[li[inside]] - 1.0
            ex[inside] = dates[j[inside]]
            kk[f"fwd{h}"] = f
            kk[f"exit{h}"] = ex
        rows.append(kk)
    return pd.concat(rows, ignore_index=True) if rows else K.iloc[0:0]


def apply_filters(X: pd.DataFrame, registered: bool = False
                  ) -> (pd.DataFrame, List[tuple]):
    """The registered exclusions, in order, with a count after each.

    ``registered=True`` is the specification exactly as first committed,
    including the coverage filter amendment A removed.
    """
    log = [("joined to a decision bar", len(X))]
    steps = [
        ("consecutive months", X["consecutive"].fillna(False)),
        ("not stale", X["stale_days"] <= STALE_DAYS),
        ("|dlog sec_num| <= 0.005", X["dlog_sec"].abs() <= DLOG_MAX),
        ("|dlog held| <= 0.005", X["dlog_held"].abs() <= DLOG_MAX),
        # AMENDMENT A (hypotheses.md, before any return was joined): the
        # coverage filter is OFF in the confirmatory run. Its premise was that
        # held < sec_num means scrip; it means the controlling block, so it
        # would have dropped BBCA, TLKM, BBRI, ASII and UNVR. --registered
        # reinstates it as the sensitivity the amendment promises.
        ("coverage >= 0.90 both ends" if registered else
         "coverage filter OFF (amendment A)",
         ((X["cover"] >= COVER_MIN) & (X["cover_prev"] >= COVER_MIN))
         if registered else pd.Series(True, index=X.index)),
        ("mom12_1 defined", np.isfinite(X["mom12_1"])),
        ("tradeable on the decision bar", X["tradeable"].astype(bool)),
        ("value >= Rp 1bn/day", X["log_turnover"] >= np.log1p(LIQ_MIN)),
    ]
    m = pd.Series(True, index=X.index)
    for name, cond in steps:
        m &= cond.fillna(False).astype(bool)
        log.append((name, int(m.sum())))
    return X[m].copy(), log


def sector_dummies(X: pd.DataFrame) -> np.ndarray:
    s = (pd.read_parquet(SECTORS).set_index("ticker")["sector"]
         if os.path.exists(SECTORS) else pd.Series(dtype=object))
    sec = X["ticker"].map(s).fillna("unknown")
    return pd.get_dummies(sec, drop_first=True).to_numpy(dtype=float)


def _rank(v: np.ndarray) -> np.ndarray:
    r = rankdata(v)
    r = r - r.mean()
    n = np.sqrt((r * r).sum())
    return r / n if n > 0 else r


class Month:
    """One month's cross-section, factorised once for every draw.

    PIVOTED QR (review, 2026-11-16). The first version used an unpivoted
    ``np.linalg.qr`` and dropped columns with a tiny |diag R|. Without pivoting
    a degenerate column placed BEFORE others leaves later columns with
    components along the dropped direction, so the kept basis no longer spans
    the controls and the residual is silently correlated with them (a synthetic
    [c1, zeros, c3] case left max |A'e| = 0.049). It never fired on H64's data
    (smallest relative |diag R| 0.081), so no reported number moves; with
    pivoting the first ``rank`` columns of Q span the column space by
    construction.

    ``ids`` are integer ticker codes, used only by the relabelling null.
    ``score`` masks which rows enter the IC (all rows still enter the fit).
    """

    def __init__(self, y, r, X, liq, hs, ids=None, score=None):
        A = np.column_stack([np.ones(len(y)), X])
        Q, R, _ = _pivoted_qr(A, mode="economic", pivoting=True)
        d = np.abs(np.diag(R))
        rank = int((d > 1e-10 * max(1.0, d.max())).sum())
        self.Q = Q[:, :rank]
        self.y, self.r, self.liq, self.hs = y, r, liq, hs
        self.ids = (np.asarray(ids) if ids is not None
                    else np.arange(len(y)))
        self.score = (np.asarray(score, dtype=bool) if score is not None
                      else np.ones(len(y), dtype=bool))
        self.rr = _rank(r[self.score])

    def resid(self, y=None) -> np.ndarray:
        y = self.y if y is None else y
        return y - self.Q @ (self.Q.T @ y)

    def ic(self, e=None) -> float:
        e = self.resid() if e is None else e
        return float(_rank(e[self.score]) @ self.rr)

    def _ics_of(self, Y: np.ndarray) -> np.ndarray:
        E = Y - (Y @ self.Q) @ self.Q.T
        E = E[:, self.score]
        R = np.argsort(np.argsort(E, axis=1), axis=1).astype(float)
        R -= R.mean(axis=1, keepdims=True)
        R /= np.sqrt((R * R).sum(axis=1, keepdims=True))
        return R @ self.rr

    def null_ics(self, draws: int, rng) -> np.ndarray:
        """The REGISTERED null: an independent shuffle inside this month."""
        n = len(self.y)
        perm = np.argsort(rng.random((draws, n)), axis=1)
        return self._ics_of(self.y[perm])

    def relabel_ics(self, keys: np.ndarray) -> np.ndarray:
        """The DEPENDENCE-PRESERVING null (review, 2026-11-16).

        ``keys`` is (draws, n_tickers): one random key per ticker per draw,
        SHARED BY EVERY MONTH. Present tickers are ordered by key and each
        takes the signal of its successor in that order — a fixed-point-free
        permutation that is the same in every month a name is present, so a
        name's whole signal history moves to another name together. That keeps
        what the within-month shuffle destroys: the dependence across months
        that overlapping 63-session labels and persistent flows create.
        """
        order = np.argsort(keys[:, self.ids], axis=1)
        src = np.roll(order, -1, axis=1)
        Y = np.empty((keys.shape[0], len(self.y)))
        np.put_along_axis(Y, order, self.y[src], axis=1)
        return self._ics_of(Y)


#: Neutralisation variants. "ols" is the REGISTERED one. The other two were
#: added by the 2026-11-16 review as labelled sensitivities, because f_retail is
#: heavy-tailed (per-month kurtosis median 25) and a few rows were setting the
#: OLS coefficients for everyone else.
MODES = ("ols", "rank", "winsor")


def build_months(X: pd.DataFrame, sig: str, k: int, mode: str = "ols",
                 score_col: Optional[str] = None,
                 ticker_codes: Optional[Dict[str, int]] = None) -> List[tuple]:
    """One Month per state month. ``score_col``: rows where it is True are
    scored OUT of the IC (still fitted), to separate a row's own contribution
    from its leverage on the fit."""
    lvl = LEVEL[sig]
    ctrl = PRICE_CONTROLS + ["log_mcap"] + ([lvl] if lvl else [])
    need = [sig, f"fwd{k}"] + ctrl
    Z = X[np.isfinite(X[need].to_numpy(dtype=float)).all(axis=1)]
    codes = ticker_codes or {t: i for i, t in
                             enumerate(sorted(Z["ticker"].unique()))}
    out = []
    for me, g in Z.groupby("month_end"):
        if len(g) < MIN_NAMES:
            continue
        S = sector_dummies(g)
        y = g[sig].to_numpy(dtype=float)
        C = g[ctrl].to_numpy(dtype=float)
        if mode == "rank":
            y = rankdata(y)
            C = np.column_stack([rankdata(c) for c in C.T])
        elif mode == "winsor":
            lo, hi = np.percentile(y, [1, 99])
            y = np.clip(y, lo, hi)
        elif mode != "ols":
            raise ValueError(mode)
        C = (C - C.mean(axis=0)) / np.where(C.std(axis=0) > 0, C.std(axis=0), 1)
        Xm = np.column_stack([C, S]) if S.size else C
        score = (~g[score_col].astype(bool).to_numpy() if score_col else None)
        out.append((me, g["decision"].iloc[0],
                    Month(y, g[f"fwd{k}"].to_numpy(dtype=float), Xm,
                          g["log_turnover"].to_numpy(dtype=float),
                          g["hs_frac"].to_numpy(dtype=float),
                          ids=g["ticker"].map(codes).to_numpy(), score=score)))
    return out


def analyse(months: List[tuple], k: int, pred: int) -> Dict[str, object]:
    lags = math.ceil(k / 21)
    ics, ex, net, spr, dates, terc = [], [], [], [], [], []
    for me, dec, M in months:
        e = M.resid()
        ics.append(M.ic(e))
        dates.append(dec)
        n = len(e)
        o = np.argsort(e, kind="mergesort")
        q = n // N_Q
        lo, hi = o[:q], o[-q:]
        fav = lo if pred < 0 else hi
        g = float(M.r[fav].mean() - M.r.mean())
        c = FEE + 2.0 * float(np.nanmean(M.hs[fav]))
        ex.append(g)
        net.append(g - c)
        spr.append(float(M.r[hi].mean() - M.r[lo].mean()))
        t3 = np.full(3, np.nan)
        cuts = np.quantile(M.liq, [1 / 3, 2 / 3])
        b = np.searchsorted(cuts, M.liq)
        for j in range(3):
            m = b == j
            if m.sum() >= 10:
                rr = rankdata(M.r[m])
                re = rankdata(e[m])
                t3[j] = float(np.corrcoef(re, rr)[0, 1])
        terc.append(t3)
    ics = np.array(ics)
    mu, se, t = newey_west_t(ics, lags)
    nm, _, nt = newey_west_t(np.array(net), lags)
    gm, _, gt = newey_west_t(np.array(ex), lags)
    sm, _, st = newey_west_t(np.array(spr), lags)
    return {"n": len(ics), "ic": ics, "dates": pd.DatetimeIndex(dates),
            "ic_mean": mu, "ic_t": t, "gross_ex": gm, "gross_t": gt,
            "net_ex": nm, "net_t": nt, "net": np.array(net),
            "spread": sm, "spread_t": st, "annual": 252.0 / k,
            "terciles": np.nanmean(np.array(terc), axis=0)}


def null_test(months: List[tuple], draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tot = np.zeros(draws)
    for _, _, M in months:
        tot += M.null_ics(draws, rng)
    return tot / len(months)


def relabel_test(months: List[tuple], draws: int, seed: int) -> np.ndarray:
    """Mean IC under the dependence-preserving null. See Month.relabel_ics."""
    rng = np.random.default_rng(seed)
    n_t = 1 + max(int(M.ids.max()) for _, _, M in months)
    keys = rng.random((draws, n_t))
    tot = np.zeros(draws)
    for _, _, M in months:
        tot += M.relabel_ics(keys)
    return tot / len(months)


def two_sided_p(null: np.ndarray, obs: float) -> float:
    v = null[np.isfinite(null)]
    return float((np.abs(v - v.mean()) >= abs(obs - v.mean())).sum() + 1) / (
        len(v) + 1)


def halves(A: Dict[str, object], k: int):
    n = A["n"]
    h = n // 2
    lags = math.ceil(k / 21)
    out = []
    for sl in (slice(0, h), slice(h, n)):
        mu, _, t = newey_west_t(A["ic"][sl], lags)
        nm, _, _ = newey_west_t(A["net"][sl], lags)
        out.append((A["dates"][sl][0].date(), A["dates"][sl][-1].date(),
                    mu, t, nm))
    return out


def drop_largest_year(A: Dict[str, object], k: int):
    yrs = A["dates"].year
    s = pd.Series(A["ic"]).groupby(yrs).sum()
    worst = int(s.abs().idxmax())
    keep = yrs != worst
    mu, _, t = newey_west_t(A["ic"][keep], math.ceil(k / 21))
    return worst, mu, t


def mechanism(X: pd.DataFrame, D: pd.DataFrame,
              grid: pd.DatetimeIndex) -> Dict[str, float]:
    """Same-month return link, AR(1), retail-vs-foreign. Descriptive only."""
    D = D.sort_values(["ticker", "date"])
    px = {}
    for t, g in D.groupby("ticker", sort=False):
        px[t] = pd.Series(g["adj_close"].to_numpy(dtype=float),
                          index=pd.DatetimeIndex(g["date"]))
    def at(t, d):
        s = px.get(t)
        if s is None:
            return np.nan
        i = s.index.searchsorted(d, side="right") - 1
        return float(s.iloc[i]) if i >= 0 else np.nan
    Z = with_predecessor(X, grid)
    Z["ret_m"] = [at(t, a) / at(t, b) - 1.0 if pd.notna(b) else np.nan
                  for t, a, b in zip(Z["ticker"], Z["month_end"], Z["prev_me"])]
    same = Z.groupby("month_end").apply(
        lambda g: pd.Series(g["f_retail"]).rank().corr(
            pd.Series(g["ret_m"]).rank()) if len(g) >= MIN_NAMES else np.nan)
    rf = Z.groupby("month_end").apply(
        lambda g: g["f_retail"].rank().corr(g["f_foreign"].rank())
        if len(g) >= MIN_NAMES else np.nan)
    ar = adjacent_pairs(Z)
    return {"same_month": float(np.nanmean(same)),
            "same_month_pos": float(np.nanmean(same > 0)),
            "same_month_months": int(np.isfinite(same).sum()),
            "retail_vs_foreign": float(np.nanmean(rf)),
            "ar1": float(ar["f_retail"].rank().corr(ar["f_prev"].rank())),
            "n_ar": len(ar)}


def with_predecessor(X: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Each row's previous state date from the ARCHIVE grid, not the frame.

    REVIEW, 2026-11-16. The first version took ``shift(1)`` over the FILTERED
    rows, so when a name's previous month had failed a filter the "same-month"
    return spanned several months against a one-month flow (806 of 10,655 rows,
    2 to 50 months apart) — A19's window mismatch. The flow is defined only for
    archive-consecutive months, so its window is exactly (grid[k-1], grid[k]].
    """
    Z = X[X["consecutive"].fillna(False).astype(bool)].copy()
    pos = pd.Series(np.arange(len(grid)), index=grid)
    k = Z["month_end"].map(pos)
    Z["prev_me"] = [grid[int(i) - 1] if pd.notna(i) and int(i) > 0 else pd.NaT
                    for i in k]
    return Z


def adjacent_pairs(Z: pd.DataFrame) -> pd.DataFrame:
    """(f_retail, f_prev) only where the previous row is the PREVIOUS MONTH.

    The first version paired each row with the previous FILTERED row, which
    could be up to 50 months back.
    """
    prev = Z[["ticker", "month_end", "f_retail"]].rename(
        columns={"month_end": "prev_me", "f_retail": "f_prev"})
    return Z.merge(prev, on=["ticker", "prev_me"], how="inner")[
        ["f_retail", "f_prev"]].dropna()


def report(sig: str, A, null, k: int, H, Y) -> tuple:
    pred = SIGNALS[sig]
    v = null
    p = float((np.abs(v - v.mean()) >= abs(A["ic_mean"] - v.mean())).sum()
              + 1) / (len(v) + 1)
    lo, hi = np.percentile(v, [2.5, 97.5])
    terc = A["terciles"]
    fin = terc[np.isfinite(terc)]
    flip = bool(len(fin) > 1 and not np.all(np.sign(fin) == np.sign(fin[0])))
    print(f"   null mean {v.mean():+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  over "
          f"{len(v):,} draws   permutation p {p:.5f}")
    print("   IC by liquidity tercile  " + "  ".join(
        f"T{i+1} {x:+.4f}" for i, x in enumerate(terc))
        + ("   SIGN FLIPS" if flip else "   sign stable"))
    for (a, b, mu, t, nm) in H:
        print(f"   half {a} … {b}:  IC {mu:+.4f} (t {t:+.2f})   "
              f"long-only net {nm*100:+.2f}%/period")
    wy, wm, wt = Y
    print(f"   dropping {wy} (largest |IC sum|): IC {wm:+.4f} (t {wt:+.2f})")
    if pred == 0:
        fired = abs(A["ic_t"]) > T_BAR or p < BONFERRONI
        print("   PREDICTED-NULL CONTROL: "
              + ("FIRED — the pipeline manufactures signal; H64/H64b are VOID"
                 if fired else "silent, as predicted — the pipeline is clean"))
        return not fired, p
    checks = {
        "predicted sign": np.sign(A["ic_mean"]) == pred,
        "p < bar and |t| > 3.81": (p < BONFERRONI) and abs(A["ic_t"]) > T_BAR,
        "sign in both halves": all(np.sign(h[2]) == pred for h in H),
        "no tercile flip": not flip,
        "drop-largest-year keeps it": (np.sign(wm) == pred) and abs(wt) > 2,
        "net long-only > 0 both halves": all(h[4] > 0 for h in H),
    }
    for c, ok in checks.items():
        print(f"     [{'PASS' if ok else 'FAIL'}] {c}")
    passed = all(checks.values())
    print(f"   GATE: {'PASSES' if passed else 'FAILS'}")
    return passed, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--spend-holdout", action="store_true")
    ap.add_argument("--registered", action="store_true",
                    help="the pre-amendment specification (coverage >= 0.90)")
    a = ap.parse_args()

    P = pd.read_parquet(ksei.PANEL_PATH)
    D = pd.read_parquet(PANEL, columns=["date", "ticker", "close", "adj_close",
                                        "tradeable", "holdout"]
                        + PRICE_CONTROLS)
    cut = D.loc[~D["holdout"], "date"].max()
    sessions = pd.DatetimeIndex(sorted(D["date"].unique()))

    print("=" * 78)
    print(" H64 — KSEI investor-type flow and the next quarter's return"
          + ("   [REGISTERED SPEC, pre-amendment]" if a.registered else ""))
    print("=" * 78)
    print(f"   KSEI files {P.month_end.nunique()} "
          f"({P.month_end.min().date()} … {P.month_end.max().date()}), "
          f"{len(P):,} equity rows")
    print(f"   price panel cut (in-sample label windows close by) {cut.date()}")

    K = ksei_frame(P, sessions, LAG)
    X = attach_prices(K, D)
    X["hs_frac"] = np.where(X["close"] > 0,
                            0.5 * tick_grid(X["close"].to_numpy(dtype=float),
                                            X["decision"]) / X["close"], np.nan)
    X, log = apply_filters(X, a.registered)
    for name, n in log:
        print(f"     {name:<34} {n:>8,}")
    ins = X[X[f"exit{CONFIRM_K}"] <= cut]
    hold = X[X["decision"] > cut]
    print(f"   in-sample {len(ins):,} rows over {ins.month_end.nunique()} "
          f"months ({ins.decision.min().date()} … {ins.decision.max().date()});"
          f" purged {len(X) - len(ins) - len(hold):,}; holdout "
          f"{len(hold):,} rows, {hold.month_end.nunique()} months — "
          f"{'UNTOUCHED' if not a.spend_holdout else 'requested'}")
    print(f"   names per month: median "
          f"{ins.groupby('month_end').size().median():.0f}\n")

    me = mechanism(ins, D, pd.DatetimeIndex(sorted(P["month_end"].unique())))
    print(" MECHANISM (descriptive, in-sample, not trials)")
    print(f"   same-month rank corr(f_retail, month return)  "
          f"{me['same_month']:+.3f}  (positive in "
          f"{round(me['same_month_pos'] * me['same_month_months'])} of "
          f"{me['same_month_months']} months)")
    print(f"   rank corr(f_retail, f_foreign) within month    "
          f"{me['retail_vs_foreign']:+.3f}")
    print(f"   AR(1) of f_retail within name, rank           "
          f"{me['ar1']:+.3f}  (n {me['n_ar']:,})\n")

    passed = {}
    results: Dict[str, Dict[str, object]] = {}
    for sig, pred in SIGNALS.items():
        print("-" * 78)
        label = {"f_retail": "H64 retail flow", "f_foreign": "H64b foreign flow",
                 "parity": "P0 sec_num parity (predicted null)"}[sig]
        print(f" {label}   predicted {'+' if pred > 0 else '−' if pred < 0 else '0'}")
        for k in HORIZONS:
            Ins = ins[ins[f"exit{k}"] <= cut] if k != CONFIRM_K else ins
            months = build_months(Ins, sig, k)
            A = analyse(months, k, pred)
            print(f"   k={k:>3}  IC {A['ic_mean']:+.4f} (HAC t {A['ic_t']:+5.2f}) "
                  f" months {A['n']:>3}   Q5−Q1 {A['spread']*100:+.2f}% "
                  f"(t {A['spread_t']:+.2f})   long-only excess gross "
                  f"{A['gross_ex']*100:+.2f}% net {A['net_ex']*100:+.2f}% "
                  f"= {A['net_ex']*A['annual']*100:+.1f}%/yr")
            if k == CONFIRM_K:
                keep = (months, A)
        months, A = keep
        null = null_test(months, a.draws, a.seed)
        passed[sig], pval = report(sig, A, null, CONFIRM_K,
                                   halves(A, CONFIRM_K),
                                   drop_largest_year(A, CONFIRM_K))
        results[sig] = {"predicted": pred, "ic": A["ic_mean"], "t": A["ic_t"],
                        "p": pval, "months": A["n"],
                        "net_annual_predicted_leg": A["net_ex"] * A["annual"],
                        "gate": bool(passed[sig])}
        # REVIEW, 2026-11-16 — NOT TRIALS, AND THE GATE ABOVE IS UNCHANGED.
        # (1) The registered null shuffles each month independently, so its
        # draws are independent across months while the real monthly ICs are
        # not (63-session labels overlap three monthly decisions and flows
        # persist). The relabelling null keeps that dependence. Its p is the
        # one to quote; the registered p is anti-conservative here.
        rel = relabel_test(months, a.draws, a.seed + 1)
        results[sig]["p_dependence_preserving"] = two_sided_p(rel, A["ic_mean"])
        results[sig]["null_sd_within_month"] = float(np.std(null))
        results[sig]["null_sd_dependence_preserving"] = float(np.std(rel))
        print(f"   REVIEW: dependence-preserving null (names relabelled "
              f"consistently across months): sd {np.std(rel):.4f} against the "
              f"within-month {np.std(null):.4f}; p "
              f"{results[sig]['p_dependence_preserving']:.4f}")
        # (2) Neutralisation: OLS on a heavy-tailed raw flow lets a few rows
        # set every coefficient. Rank-space and winsorised variants bracket it.
        for mode in ("rank", "winsor"):
            Am = analyse(build_months(ins, sig, CONFIRM_K, mode=mode),
                         CONFIRM_K, pred)
            results[sig][f"ic_{mode}"] = Am["ic_mean"]
            results[sig][f"t_{mode}"] = Am["ic_t"]
            print(f"   REVIEW: {mode}-space neutralisation: IC "
                  f"{Am['ic_mean']:+.4f} (t {Am['ic_t']:+.2f})")
        if sig != "parity":
            # ROBUSTNESS, ADDED AFTER THE RESULT WAS SEEN, NOT A TRIAL. A
            # type-to-type re-registration (BYAN 2026-10: 10.75bn shares from
            # individual to corporate) passes every registered filter and reads
            # as flow. Two numbers separate its OWN contribution from its
            # LEVERAGE on the per-month OLS (the review found the first version
            # of this comment wrong: ranks bound a row's own Spearman term, not
            # its pull on the regression that runs before the ranks).
            Xr = ins[~ins["xfer"].astype(bool)]
            Ar = analyse(build_months(Xr, sig, CONFIRM_K), CONFIRM_K, pred)
            As_ = analyse(build_months(ins, sig, CONFIRM_K, score_col="xfer"),
                          CONFIRM_K, pred)
            results[sig]["robust_no_transfers"] = {
                "ic": Ar["ic_mean"], "t": Ar["ic_t"],
                "ic_scored_out_only": As_["ic_mean"],
                "t_scored_out_only": As_["ic_t"],
                "rows_dropped": int(ins["xfer"].astype(bool).sum())}
            print(f"   ROBUSTNESS (post-result, not a trial): "
                  f"{int(ins['xfer'].astype(bool).sum())} type-to-type transfer "
                  f"rows scored out but still fitted: IC "
                  f"{As_['ic_mean']:+.4f} (t {As_['ic_t']:+.2f}); dropped from "
                  f"the fit too: IC {Ar['ic_mean']:+.4f} (t {Ar['ic_t']:+.2f}). "
                  f"The gap is leverage on the fit, not the rows' own tilt.")
        if sig == "f_retail":
            # EXPLORATORY, ADDED AFTER THE CONFIRMATORY RESULT WAS SEEN. The
            # registered leg holds what retail SOLD; the sign came back the
            # other way, so the opposite leg is priced here for the record. It
            # is not a trial, enters no gate, and is labelled wherever printed.
            for kk in (CONFIRM_K, 126):
                Ik = ins[ins[f"exit{kk}"] <= cut]
                Ao = analyse(build_months(Ik, sig, kk), kk, +1)
                results[sig][f"exploratory_hold_retail_bought_net_annual_k{kk}"] = (
                    Ao["net_ex"] * Ao["annual"])
                results[sig][f"exploratory_hold_retail_bought_net_t_k{kk}"] = (
                    Ao["net_t"])
                print(f"   EXPLORATORY (post-result, not a trial): hold what "
                      f"retail BOUGHT, k={kk}: net {Ao['net_ex']*100:+.2f}%/"
                      f"period = {Ao['net_ex']*Ao['annual']*100:+.1f}%/yr "
                      f"(t {Ao['net_t']:+.2f})")
        if sig != "parity":
            Ks = ksei_frame(P, sessions, LAG_SENSITIVITY)
            Xs = attach_prices(Ks, D)
            Xs["hs_frac"] = np.where(
                Xs["close"] > 0, 0.5 * tick_grid(Xs["close"].to_numpy(dtype=float),
                                                 Xs["decision"]) / Xs["close"],
                np.nan)
            Xs, _ = apply_filters(Xs, a.registered)
            Xs = Xs[Xs[f"exit{CONFIRM_K}"] <= cut]
            As = analyse(build_months(Xs, sig, CONFIRM_K), CONFIRM_K, pred)
            print(f"   sensitivity lag={LAG_SENSITIVITY}: IC {As['ic_mean']:+.4f} "
                  f"(t {As['ic_t']:+.2f})")
        print()

    print("=" * 78)
    clean = passed["parity"]
    for sig in ("f_retail", "f_foreign"):
        ok = passed[sig] and clean
        print(f" {sig:<10} gate {'PASSES' if ok else 'FAILS'}"
              + ("" if clean else "  (void: the null control fired)"))
    if a.spend_holdout:
        if not any(passed[s] and clean for s in ("f_retail", "f_foreign")):
            print("\n --spend-holdout REFUSED: no signal passed its in-sample "
                  "gate. The holdout stays unspent for a later registered test.")
        else:
            for sig in ("f_retail", "f_foreign"):
                if not (passed[sig] and clean):
                    continue
                Hm = build_months(hold, sig, CONFIRM_K)
                Ah = analyse(Hm, CONFIRM_K, SIGNALS[sig])
                print(f"\n HOLDOUT {sig}: IC {Ah['ic_mean']:+.4f} "
                      f"(t {Ah['ic_t']:+.2f}) over {Ah['n']} months; "
                      f"net long-only {Ah['net_ex']*100:+.2f}%/period")
    else:
        print(f"\n holdout UNTOUCHED ({len(hold):,} rows reserved).")
    out = {"run": pd.Timestamp.now().isoformat(timespec="seconds"),
           "spec": "registered" if a.registered else "amendment A",
           "confirm_k": CONFIRM_K, "trials_after": TRIALS_AFTER,
           "bonferroni": BONFERRONI, "draws": a.draws,
           "in_sample": [str(ins.decision.min().date()),
                         str(ins.decision.max().date())],
           "holdout_spent": bool(a.spend_holdout),
           "mechanism": me, "null_control_clean": bool(clean),
           "signals": results}
    path = RESULT if not a.registered else RESULT.replace(".json",
                                                          "_registered.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f" wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
