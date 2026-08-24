"""The twice-daily situational brief: state, co-movement, run age, candidates.

WHY EVERY SECTION HERE IS DESCRIPTIVE, AND SAYS SO
----------------------------------------------------
The research programme in this repo reached one answer (CLAUDE.md A9): there is
real structure in IDX prices and none of it survives the cost of acting on it.
Four independent instruments returned nothing tradeable, and H13 in particular
found **all eight** registered price features net-negative at every horizon
once 56 bps of fees and a point-in-time fraksi-harga half-spread were charged.

A brief that ranked names and called them buys would therefore be contradicting
the evidence that sits in the same repository. What it can do honestly is:

  * **State.** Breadth, dispersion, regime, who moved, how much of the board is
    locked at its auto-rejection limit. This is arithmetic and it is exact.
  * **Co-movement.** Which names moved *together* today, derived from a
    factorisation fit on data strictly before the bar. That is the closest
    thing to a "narrative" this data supports: a group is described by its
    constituents and its loading, never by a story about why.
  * **Run age.** For any move: where it sits in its own history, and the
    historical distribution of what followed states like it — with the
    unconditional base rate and the *effective* sample size printed beside it,
    so the reader can see how little the conditioning buys.
  * **Candidates.** A ranking on the eight registered features, each printed
    next to H13's measured post-cost result for that same feature.

WHAT IS NOT HERE
-----------------
**News narrative.** There is no news source anywhere in this repo and §3's
data table lists none. Inventing one from a price move is precisely the failure
§9.6 was written to prevent — a story is very easy to write and very hard to
falsify. :func:`narrative_gap` states the absence in the output rather than
letting it be filled in silently.

THE HOLDOUT STAYS UNTOUCHED
-----------------------------
§11 reserves the most recent 24 months to be spent once. Every reference
distribution here — the run-length quantiles, the conditional forward tables —
is estimated on ``holdout == False`` rows only. That is not merely compliance:
it makes the conditionals genuinely out-of-sample with respect to the bar the
brief is describing, which is the stronger position anyway.

NO LOOKAHEAD, ENFORCED STRUCTURALLY
-------------------------------------
Every function that takes an as-of day slices to ``date <= day`` before it
computes anything, and the factorisation is fit on ``date < day``. A5: never
use data stamped after the decision bar.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..features.price import PREDICTED
from ..spine import reference

#: Trading days in the window that defines a "run". One parameter, not tuned:
#: 250 sessions is the same one-year lookback ``hi52`` already uses, so the run
#: anchor and the 52-week position are talking about the same window.
RUN_WINDOW = 250

#: A name needs this many bars before its run state means anything.
MIN_BARS = RUN_WINDOW + 20

#: A5's actual schedule: 0.28% buy + 0.18% sell + 0.1% sell tax. §7 quotes a
#: lighter retail range; the user's Mandiri schedule wins.
ROUND_TRIP_FEE = 0.0056

#: Below this cross-sectional turnover percentile a name is not in the
#: conditional reference sample. H13 found the price effects concentrate where
#: the spread eats them; conditioning on illiquid names measures a market that
#: cannot be traded at the prices shown.
LIQUID_PCT = 0.50

#: Terciles, so each of the 54 conditional buckets keeps a usable sample.
N_BUCKET = 3

#: Block length for the date-block bootstrap, in sessions. Forward windows of
#: 20 days overlap, so resampling individual dates would understate the
#: uncertainty by roughly the square root of the overlap.
BLOCK = 21


# ==========================================================================
# as-of handling — every entry point goes through these two
# ==========================================================================
#: A day needs this fraction of the recent typical name count before a
#: cross-sectional statistic computed on it means anything.
MIN_COVERAGE = 0.80


def coverage(P: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Names present per date, and that count against the recent norm.

    THE RAGGED EDGE IS THE FAILURE MODE THIS EXISTS TO CATCH. Refreshing a
    watchlist rather than the universe leaves the last few dates populated by
    a handful of large caps while the panel still looks current. Breadth on
    that cross-section is not breadth — it is forty blue chips — and nothing in
    the output would say so. So coverage is checked before any statistic, not
    after one looks strange.
    """
    d = pd.to_datetime(P["date"])
    n = P.groupby(d).size().sort_index().rename("n_names")
    norm = n.rolling(lookback, min_periods=5).median().shift(1)
    return pd.DataFrame({"n_names": n, "typical": norm,
                         "coverage": n / norm})


def resolve_asof(P: pd.DataFrame, day=None,
                 min_coverage: float = MIN_COVERAGE) -> pd.Timestamp:
    """The last bar at or before ``day`` with a representative cross-section.

    Defaults to the panel's last such bar rather than its last bar. When those
    differ the caller is looking at a partial refresh, and
    :func:`coverage_warning` turns that into a line in the output.
    """
    d = pd.to_datetime(P["date"])
    cov = coverage(P)
    ok = cov.index[(cov["coverage"] >= min_coverage) | cov["typical"].isna()]
    if day is not None:
        ok = ok[ok <= pd.Timestamp(day)]
    if not len(ok):
        raise ValueError("no date has a representative cross-section")
    return pd.Timestamp(ok.max())


def coverage_warning(P: pd.DataFrame, day: pd.Timestamp) -> Optional[str]:
    """The line to print when the panel holds bars newer than ``day``."""
    d = pd.to_datetime(P["date"])
    newest = pd.Timestamp(d.max())
    if newest <= day:
        return None
    cov = coverage(P)
    tail = cov[cov.index > day]
    return (f"PARTIAL REFRESH: the panel holds bars through "
            f"{newest.date()} but only {int(tail['n_names'].iloc[-1])} names "
            f"trade in them against a typical "
            f"{int(cov.loc[day, 'n_names'])}. Those dates are a watchlist, not "
            f"a cross-section, so the brief is cut back to {day.date()}. "
            f"Refresh the whole universe to move it forward.")


def upto(P: pd.DataFrame, day: pd.Timestamp,
         strict: bool = False) -> pd.DataFrame:
    """Everything the decision bar is allowed to see.

    ``strict`` excludes the bar itself, which the factorisation needs: a
    component fitted on today's returns would explain today's returns by
    construction.
    """
    d = pd.to_datetime(P["date"])
    return P[d < day] if strict else P[d <= day]


# ==========================================================================
# 1. MARKET STATE — arithmetic, and exact
# ==========================================================================
def daily_returns(P: pd.DataFrame) -> pd.DataFrame:
    """Wide date x ticker frame of simple returns on the adjusted close."""
    w = P.pivot_table(index="date", columns="ticker", values="adj_close")
    return w.sort_index().pct_change()


def index_series(P: pd.DataFrame, weight: str = "equal") -> pd.Series:
    """A cumulative index from the panel itself.

    ``equal`` is the equal-weighted cross-sectional mean return — the honest
    default, since it is what every cross-sectional statistic in this repo is
    implicitly about.

    ``turnover`` weights by each name's own trailing liquidity, **lagged one
    bar** so the weight is knowable before the return it multiplies. It is a
    proxy for a cap-weighted index, not a reconstruction of IHSG: this repo has
    no shares-outstanding series, and turnover and market cap are correlated
    but not the same thing. Both are printed so a divergence between them —
    big names carrying the tape while the median name does not — is visible
    rather than hidden inside one number.
    """
    R = daily_returns(P)
    if weight == "equal":
        r = R.mean(axis=1, skipna=True)
    elif weight == "turnover":
        W = P.pivot_table(index="date", columns="ticker",
                          values="log_turnover").reindex_like(R)
        W = np.exp(W.shift(1))                       # lagged: knowable ex ante
        W = W.where(R.notna())
        r = (R * W).sum(axis=1, min_count=1) / W.sum(axis=1, min_count=1)
    else:
        raise ValueError(f"unknown weight {weight!r}")
    return (1.0 + r.fillna(0.0)).cumprod()


def _pct_rank(hist: np.ndarray, x: float) -> float:
    """Where ``x`` sits in ``hist``, as a fraction in [0, 1]."""
    h = np.asarray(hist, dtype=float)
    h = h[np.isfinite(h)]
    if not len(h) or not np.isfinite(x):
        return np.nan
    return float((h <= x).mean())


MA_WINDOWS = (20, 50, 200)


def snapshot(P: pd.DataFrame, day: pd.Timestamp,
             wins: Sequence[int] = MA_WINDOWS) -> pd.DataFrame:
    """One row per name that traded on ``day``: where it stands, on its own bars.

    COMPUTED ON THE LONG FRAME, NOT A PIVOT, and that is not a stylistic
    preference. A wide date x ticker frame is indexed by the UNION of every
    name's trading days, so a name that was suspended for a week acquires NaN
    rows it never had, and ``rolling(200, min_periods=200)`` then returns
    nothing for it. The first version of this function did exactly that and
    reported "0 names above the 200-day average" on a panel of 830 names —
    a statistic that was not wrong so much as vacant. Grouping by ticker keeps
    each rolling window on that ticker's own consecutive bars.
    """
    H = upto(P, day)
    H = H.sort_values(["ticker", "date"], kind="mergesort")
    # everything here looks back at most RUN_WINDOW bars, so carrying the full
    # history through the rolling is waste
    H = H.groupby("ticker", sort=False).tail(RUN_WINDOW + max(wins) + 5)
    g = H.groupby("ticker", sort=False)["adj_close"]
    out = pd.DataFrame({"ticker": H["ticker"].to_numpy(),
                        "date": pd.to_datetime(H["date"]).to_numpy(),
                        "adj_close": H["adj_close"].to_numpy(),
                        "close": H["close"].to_numpy()})
    out["ret1"] = g.pct_change().to_numpy()
    # the RAW previous close, not one backed out of the adjusted return: the
    # auto-rejection band is taken on the unadjusted reference price, and on an
    # ex-dividend day the two differ by exactly the amount that matters.
    out["prev_close"] = H.groupby("ticker", sort=False)["close"].shift(1) \
                         .to_numpy()
    for n in wins:
        out[f"ma{n}"] = g.rolling(n, min_periods=n).mean().to_numpy()
    r = g.rolling(RUN_WINDOW, min_periods=RUN_WINDOW)
    out["hi250"] = r.max().to_numpy()
    out["lo250"] = r.min().to_numpy()
    S = out[out["date"] == day].set_index("ticker")
    keep = [c for c in ("log_turnover", "vol60", "tradeable") if c in P]
    C = P[pd.to_datetime(P["date"]) == day].set_index("ticker")[keep]
    return S.join(C, how="left")


def breadth(S: pd.DataFrame, day: pd.Timestamp,
            wins: Sequence[int] = MA_WINDOWS) -> Dict[str, object]:
    """How much of the board is above its own moving averages, and moving.

    Breadth is the one market-state statistic a handful of index heavyweights
    cannot fake, which is why it comes first. Takes the :func:`snapshot` rather
    than the panel so the whole brief agrees on one cross-section.
    """
    out: Dict[str, object] = {"asof": day, "n_names": int(len(S))}
    for n in wins:
        ma = S[f"ma{n}"]
        ok = ma.notna() & S["adj_close"].notna()
        out[f"above_{n}d"] = float((S["adj_close"][ok] > ma[ok]).mean()) \
            if ok.any() else np.nan
        out[f"n_{n}d"] = int(ok.sum())

    r = S["ret1"]
    r = r[np.isfinite(r)]
    out["advancing"] = float((r > 0).mean()) if len(r) else np.nan
    out["unchanged"] = float((r == 0).mean()) if len(r) else np.nan
    out["median_move"] = float(r.median()) if len(r) else np.nan
    out["dispersion"] = float(r.std()) if len(r) else np.nan

    ok = S["hi250"].notna() & S["adj_close"].notna()
    out["new_highs"] = int((S["adj_close"][ok] >= S["hi250"][ok]).sum())
    out["new_lows"] = int((S["adj_close"][ok] <= S["lo250"][ok]).sum())
    out["n_250d"] = int(ok.sum())
    return out


def limit_moves(S: pd.DataFrame, day: pd.Timestamp,
                tol: float = 1e-6) -> Dict[str, int]:
    """Names that closed at the point-in-time auto-rejection band.

    §5 is blunt that a day locked at ARA is a day you could not buy, and this
    repo carries the historical schedule precisely so today's band is not
    applied to an older bar. Counting the locked names is the sharpest single
    read on whether a move is orderly or a stampede — and it is IDX-specific,
    with no analogue in the US-market literature most technical work borrows
    from.

    This is the CLOSE test only, not :func:`reference.was_locked`'s full test,
    because the panel carries no intraday high or low. A name that closed at
    the ceiling having traded below it during the session counts here and would
    not count there, so this is an upper bound on genuine lock-ups and is
    labelled as one everywhere it is printed.
    """
    ok = S["close"].notna() & S["prev_close"].notna() & (S["prev_close"] > 0)
    prev = S["prev_close"][ok]
    ara = arb = 0
    for p, c in zip(prev.to_numpy(dtype=float),
                    S["close"][ok].to_numpy(dtype=float)):
        if not np.isfinite(p) or p <= 0:
            continue
        try:
            up, dn = reference.auto_rejection(p, day)
        except reference.OutsideCoverage:
            continue
        hi = p + (abs(up) if up < 0 else p * up)
        lo = p - (abs(dn) if dn < 0 else p * dn)
        if c >= hi - tol:
            ara += 1
        elif c <= lo + tol:
            arb += 1
    return {"ara": ara, "arb": arb, "n": int(ok.sum())}


def regime(P: pd.DataFrame, day: pd.Timestamp,
           lookback: int = 20, hist: int = 1250) -> Dict[str, object]:
    """Index return over several horizons, and where today's vol sits.

    The percentile is against the trailing ``hist`` sessions (five years), not
    against the whole sample, so "high vol" means high relative to the recent
    past rather than relative to 2008.
    """
    H = upto(P, day)
    out: Dict[str, object] = {}
    for name in ("equal", "turnover"):
        idx = index_series(H, name)
        if len(idx) < lookback + 5:
            continue
        for k, lab in ((1, "1d"), (5, "1w"), (21, "1m"), (63, "3m")):
            out[f"{name}_{lab}"] = (float(idx.iloc[-1] / idx.iloc[-1 - k] - 1.0)
                                    if len(idx) > k else np.nan)
        ytd = idx[idx.index.year == day.year]
        out[f"{name}_ytd"] = (float(idx.iloc[-1] / ytd.iloc[0] - 1.0)
                              if len(ytd) > 1 else np.nan)
        r = idx.pct_change()
        rv = r.rolling(lookback, min_periods=lookback).std() * np.sqrt(252)
        out[f"{name}_vol"] = float(rv.iloc[-1]) if len(rv.dropna()) else np.nan
        out[f"{name}_vol_pct"] = _pct_rank(rv.iloc[-hist:].to_numpy(),
                                           out[f"{name}_vol"])
    return out


#: Absolute daily-value floor for anything the brief names. The median IDX name
#: turns over well under Rp 1bn a day, so a purely relative filter still admits
#: names where a single retail order is the day's tape.
MIN_VALUE = 5e9


def movers(S: pd.DataFrame, n: int = 10,
           min_turnover_pct: float = LIQUID_PCT,
           min_value: float = MIN_VALUE) -> Dict[str, pd.DataFrame]:
    """Biggest movers among names liquid enough for the print to mean anything.

    BOTH filters, and the absolute one does the real work. A percentile alone
    still admits a name doing 25% on Rp 780m of turnover, because the median
    IDX name trades less than that — such a list says nothing about the market
    and everything about the tick grid at Rp 100. The floor is stated in the
    output so the reader knows what was excluded.
    """
    lt = S["log_turnover"]
    ok = np.isfinite(S["ret1"]) & np.isfinite(lt) & \
        (lt >= lt.quantile(min_turnover_pct)) & (np.exp(lt) >= min_value)
    D = pd.DataFrame({"ticker": S.index[ok], "ret": S["ret1"][ok].to_numpy(),
                      "close": S["close"][ok].to_numpy(),
                      "value": np.exp(lt[ok].to_numpy())})
    D = D.sort_values("ret", ascending=False)
    return {"up": D.head(n).reset_index(drop=True),
            "down": D.tail(n).iloc[::-1].reset_index(drop=True)}


# ==========================================================================
# 2. CO-MOVEMENT — the only "narrative" this data can support
# ==========================================================================
def comovement(P: pd.DataFrame, day: pd.Timestamp, n_pc: int = 4,
               lookback: int = 250, min_names: int = 60, top: int = 8,
               min_value: float = MIN_VALUE) -> pd.DataFrame:
    """Which groups of names moved together today, from a factorisation.

    §7 uses trailing principal components as a substitute for a sector map,
    because this repo has none — ``data/sectors.py`` is the licensed API and
    needs a key. The substitute is arguably better for this purpose: a sector
    label is a fixed opinion about which names belong together, whereas a
    component is measured from how they actually traded over the last year.

    NO LOOKAHEAD: the components are fitted on returns strictly BEFORE ``day``,
    then today's cross-section is projected onto them. A factorisation that saw
    today would explain today by construction.

    LIQUID NAMES ONLY. Six hundred names that trade under Rp 1bn a day
    contribute mostly tick noise, and a component fitted through that noise is
    a component about the tick grid. The filter is stated in the output.

    Each row is one component with the standardised size of today's move along
    it and its heaviest-loading names in both directions. **It is named by its
    constituents and nothing else.** Whether eight coal names moving together
    is "the coal trade" is an interpretation, and §9.6's rule — state the
    regularity, mark the interpretation separately — governs here too.
    """
    C = P[pd.to_datetime(P["date"]) == day]
    liquid = set(C.loc[np.exp(C["log_turnover"]) >= min_value, "ticker"])
    Hs = upto(P, day, strict=True)
    Hs = Hs[Hs["ticker"].isin(liquid)]
    R = daily_returns(Hs).tail(lookback)
    R = R.dropna(axis=1, thresh=int(0.9 * len(R)))
    if R.shape[1] < min_names or len(R) < 60:
        return pd.DataFrame()
    R = R.fillna(0.0)
    mu, sd = R.mean(), R.std().replace(0.0, np.nan)
    Z = ((R - mu) / sd).dropna(axis=1)
    if Z.shape[1] < min_names:
        return pd.DataFrame()

    U, S, Vt = np.linalg.svd(Z.to_numpy(dtype=float), full_matrices=False)
    k = min(n_pc, Vt.shape[0])
    load = Vt[:k]                                    # (k, n_names)
    names = np.asarray(Z.columns)
    var = (S[:k] ** 2) / float((S ** 2).sum())

    Ht = upto(P, day)
    Ht = Ht[Ht["ticker"].isin(liquid)]
    today = daily_returns(Ht).iloc[-1].reindex(Z.columns)
    zt = ((today - mu.reindex(Z.columns)) / sd.reindex(Z.columns))
    zt = zt.to_numpy(dtype=float)
    ok = np.isfinite(zt)
    if ok.sum() < min_names:
        return pd.DataFrame()
    total_today = float(np.square(zt[ok]).sum())

    # historical scores, for the standardisation that makes "how big is today"
    # a comparable number across components
    hist = Z.to_numpy(dtype=float) @ load.T          # (T, k)
    rows = []
    for j in range(k):
        w = load[j][ok]
        score = float(zt[ok] @ w)
        hs = hist[:, j]
        s = float(hs.std())
        order = np.argsort(load[j])
        # sign the component so a positive score always means "these names up"
        flip = -1.0 if score < 0 else 1.0
        pos = names[order[::-1]][:top] if flip > 0 else names[order][:top]
        neg = names[order][:top] if flip > 0 else names[order[::-1]][:top]
        rows.append({
            "pc": j + 1,
            "var_share": float(var[j]),
            "today_share": (float(score ** 2 / total_today)
                            if total_today > 0 else np.nan),
            "score_z": float(flip * score / s) if s > 0 else np.nan,
            "abs_pct": _pct_rank(np.abs(hs), abs(score)),
            "with": list(pos),
            "against": list(neg)})
    return pd.DataFrame(rows)


def news_caveat() -> str:
    """What the news section is, and the one thing it must never be used for.

    THIS REPLACES A WRONG CLAIM. The first version of this function said a news
    narrative was "not available… there is no news source anywhere in this repo
    and §3's data table lists none". Both facts were true; the conclusion was
    not. §3 listed none because nobody had looked. Eight endpoints were tested
    in about a minute and five answered — see `idxbot.data.news`. The failure
    is the one CLAUDE.md A1 already records, and its lesson stands: check the
    cheapest route before writing down the constraint.
    """
    return (
        "These are public RSS syndication feeds — headline, link, source and "
        "timestamp only, no article bodies, nothing republished. THEY MAY NOT "
        "ENTER ANY STATISTIC. There is no point-in-time news archive, so a "
        "headline visible today cannot be reconstructed as it stood on a past "
        "bar, and any backtest built on it is look-ahead by construction "
        "(A5). This section is for reading. A quiet section is not the same "
        "as a quiet day — the source line above says which feeds answered.")


# ==========================================================================
# 3. RUN AGE — "is it over or just started", answered descriptively
# ==========================================================================
#: A window needs this fraction of usable bars before its extremes mean
#: anything. A name suspended for four of the last twelve months has a
#: 250-session high that is really a 130-session high.
MIN_VALID_FRAC = 0.80


def _last_argext(x: np.ndarray, W: int,
                 min_frac: float = MIN_VALID_FRAC
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Index of the LAST max and LAST min in each trailing window of length W.

    Last rather than first: after a long flat stretch at the high, the run is
    measured from the most recent time price was there, not the first.

    NAN AND NON-POSITIVE PRICES ARE MASKED, and this is not defensive padding.
    ``np.argmax`` returns the index of a NaN if one is present — NaN compares
    False against everything, so the scan never displaces it — which silently
    anchors the run to a hole in the data. The spine carries 2,327 bars with a
    non-positive adjusted close (BATA in 2002, chiefly), and before this mask
    they produced 915 "advances" with a NEGATIVE return from their own anchor:
    an impossibility by the definition, and exactly the sort of thing that
    reaches a conditional table looking like a state rather than a defect.

    Masking to -inf for the max and +inf for the min makes an invalid bar
    unable to win either scan, and the validity count then discards windows
    too holed to describe.

    Uses a sliding view per ticker. Materialising one for the whole panel at
    once would be ~5.6 GB; per ticker it is a few megabytes.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(x)
    hi = np.full(n, -1, dtype=np.int64)
    lo = np.full(n, -1, dtype=np.int64)
    if n < W:
        return hi, lo
    bad = ~np.isfinite(x) | (x <= 0)
    for arr, out, fill in ((np.where(bad, -np.inf, x), hi, -np.inf),
                           (np.where(bad, np.inf, x), lo, np.inf)):
        win = sliding_window_view(arr, W)[:, ::-1]
        idx = win.argmax(1) if fill == -np.inf else win.argmin(1)
        out[W - 1:] = np.arange(n - W + 1) + (W - 1 - idx)

    ok = np.cumsum(~bad)
    valid = ok[W - 1:] - np.concatenate(([0], ok[:n - W]))
    thin = valid < int(min_frac * W)
    hi[W - 1:][thin] = -1
    lo[W - 1:][thin] = -1
    hi[bad] = -1
    lo[bad] = -1
    return hi, lo


def run_state(P: pd.DataFrame, window: int = RUN_WINDOW) -> pd.DataFrame:
    """Per bar: which leg the name is in, how old it is, how far it has come.

    THE DEFINITION, which is deliberately parameter-light. Inside the trailing
    ``window``, find the last high and the last low. If the high came after the
    low the name is in an ADVANCE that began at that low; otherwise it is in a
    DECLINE that began at that high. ``run_days`` is the age of that leg and
    ``run_ret`` its log return from the anchor.

    ``run_z = run_ret / (vol60 * sqrt(run_days))`` is the extension in standard
    deviations *for a move of this length*. This is the number that answers the
    question as asked: a 30% advance is enormous for a quiet name over ten days
    and unremarkable for a volatile one over two hundred, and run_z says which.

    READ THE ANCHOR CAREFULLY. The leg runs from the OLDER extreme to the
    NEWER one, and ``run_days`` is the distance from that older extreme — the
    start of the leg — not the age of the move price is making right now.
    BBCA is a worked example: its 250-day high came before its 250-day low, so
    it reads "down leg, 244 sessions" (the decline that began at the high)
    while ``give_back`` reads +30.9% (it has since recovered that far off the
    low). Both numbers are right and together they say what one alone cannot.

    ``give_back`` is how far price has retraced from the leg's extreme — below
    the high for an advance, above the low for a decline. It is the difference
    between a run that is extended and one that has already turned, and it is
    deliberately NOT part of the bucket definition: the four conditioning
    variables were fixed before any cell was looked at, and adding a fifth
    after seeing which cells came out large is the multiple-testing trap this
    repo keeps a trial count for.

    None of these forecast anything. They locate the bar. What FOLLOWED bars
    like it is :func:`conditional_table`'s question, and it is asked separately
    on purpose.
    """
    d = P[["date", "ticker", "adj_close", "vol60"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["ticker", "date"], kind="mergesort")
    out = []
    for t, g in d.groupby("ticker", sort=False):
        x = g["adj_close"].to_numpy(dtype=float)
        n = len(x)
        if n < window:
            continue
        ihi, ilo = _last_argext(x, window)
        pos = np.arange(n)
        up = ihi > ilo
        anchor = np.where(up, ilo, ihi)
        valid = (ihi >= 0) & (ilo >= 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ax = np.where(valid, x[np.clip(anchor, 0, n - 1)], np.nan)
            ext = np.where(valid, x[np.clip(np.where(up, ihi, ilo),
                                            0, n - 1)], np.nan)
            run_ret = np.log(x / ax)
            give = np.log(x / ext)
        days = np.where(valid, pos - anchor, -1)
        # A flat series has zero realised vol, and dividing by it turns a
        # motionless name into an infinite extension. 164,627 panel bars carry
        # vol60 <= 0; they get no run_z rather than an enormous one.
        vol = g["vol60"].to_numpy(dtype=float)
        vol = np.where(np.isfinite(vol) & (vol > 0), vol, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            rz = run_ret / (vol * np.sqrt(np.maximum(days, 1)))
        out.append(pd.DataFrame({
            "date": g["date"].to_numpy(), "ticker": t,
            "leg": np.where(valid, np.where(up, "up", "down"), None),
            "run_days": days, "run_ret": run_ret, "run_z": rz,
            "give_back": give,
            # LOG returns are what run_z needs — they are additive, so a move
            # of n days scales against sd*sqrt(n) correctly. They are NOT what
            # a reader should see: a log -2.61 printed as a percentage reads
            # "-261%", which no share price can do. The simple equivalents
            # travel alongside so display never has to convert.
            "run_pct": np.expm1(run_ret),
            "give_pct": np.expm1(give)})[valid])
    if not out:
        return pd.DataFrame(columns=["date", "ticker", "leg", "run_days",
                                     "run_ret", "run_z", "give_back",
                                     "run_pct", "give_pct"])
    return pd.concat(out, ignore_index=True)


# ==========================================================================
# the conditional: what followed bars in this state
# ==========================================================================
def _terciles(x: np.ndarray) -> np.ndarray:
    q = np.nanquantile(np.asarray(x, dtype=float), [1 / 3, 2 / 3])
    return q


def bucket_of(run_days, run_z, ivol_pct, edges: Dict[str, Sequence[float]],
              leg) -> pd.Series:
    """Assign the (leg, age, extension, market-vol) bucket.

    THE EXTENSION CUTS ARE PER LEG, and they have to be. An advance's ``run_z``
    is non-negative by construction — the anchor is the window's low, so price
    cannot be below it — and a decline's is non-positive. Pooled terciles
    therefore put every advance in the top two and every decline in the bottom
    two, leaving structurally impossible buckets that fill only with data
    defects. The first version did exactly that and produced an "advance" cell
    holding 54 rows with a median extension of -8.2 standard deviations.

    The edges come from the reference sample and are stored with the table, so
    a live bar is bucketed by exactly the cuts the historical distribution was
    built with. Recomputing them on the live cross-section would silently move
    the bucket definitions between runs.
    """
    def cut(v, e):
        v = np.asarray(v, dtype=float)
        return np.where(np.isfinite(v), np.searchsorted(
            np.asarray(e, dtype=float), np.nan_to_num(v)), -1)
    a = cut(run_days, edges["run_days"])
    m = cut(ivol_pct, edges["ivol"])
    leg = np.asarray(pd.Series(leg).to_numpy())
    z = np.full(len(leg), -1)
    for side in ("up", "down"):
        sel = leg == side
        if sel.any():
            z[sel] = cut(np.asarray(run_z, dtype=float)[sel],
                         edges[f"run_z_{side}"])
    return pd.Series([f"{l}|{int(ai)}|{int(zi)}|{int(mi)}"
                      if (l in ("up", "down") and ai >= 0 and zi >= 0
                          and mi >= 0) else None
                      for l, ai, zi, mi in zip(leg, a, z, m)])


def _block_bootstrap(dates: np.ndarray, values: np.ndarray, base: np.ndarray,
                     draws: int, rng: np.random.Generator,
                     block: int = BLOCK) -> Tuple[float, float]:
    """CI on (bucket mean - base rate), resampling CONTIGUOUS blocks of dates.

    Forward windows of 20 sessions overlap, so two rows five days apart share
    three quarters of their return. Resampling rows, or even individual dates,
    would treat those as independent and understate the interval by roughly the
    square root of the overlap. Blocks of ``block`` sessions keep the overlap
    inside the resampled unit where it belongs.

    THE RESAMPLE MUST KEEP DUPLICATES. An earlier version selected the drawn
    blocks with ``np.isin``, which is a set membership test: a block drawn
    twice contributed once, so every resample was smaller and less variable
    than the sample it came from and every interval came out too narrow. A
    bootstrap that understates uncertainty is worse than no bootstrap, because
    it looks like rigour. The ragged gather below repeats a block as often as
    it is drawn.
    """
    ud, inv = np.unique(dates, return_inverse=True)
    n_slots = len(ud)
    if n_slots < 3 * block:
        return (np.nan, np.nan)
    order = np.argsort(inv, kind="mergesort")
    counts = np.bincount(inv, minlength=n_slots)
    bounds = np.concatenate(([0], np.cumsum(counts)))
    n_blocks = max(1, n_slots // block)
    out = np.empty(draws)
    for i in range(draws):
        s = rng.integers(0, n_slots - block + 1, size=n_blocks)
        slots = (s[:, None] + np.arange(block)).ravel()
        c = counts[slots]
        tot = int(c.sum())
        if tot == 0:
            out[i] = np.nan
            continue
        # ragged gather: every chosen slot's rows, repeats included
        off = (np.repeat(bounds[slots], c)
               + np.arange(tot) - np.repeat(np.cumsum(c) - c, c))
        idx = order[off]
        out[i] = np.nanmean(values[idx]) - np.nanmean(base[idx])
    return (float(np.nanpercentile(out, 2.5)),
            float(np.nanpercentile(out, 97.5)))


def conditional_frame(P: pd.DataFrame, R: pd.DataFrame, k: int = 20,
                      liquid_pct: float = LIQUID_PCT
                      ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """The bucketed reference sample, PRE-HOLDOUT only, and its bucket edges.

    Split out from :func:`conditional_table` so the permutation null runs on
    exactly the same rows, filters and cuts as the observed table. A null built
    from a separately-prepared frame answers a slightly different question,
    and the difference is invisible in the output.
    """
    fwd = f"fwd{k}"
    base = P[~P["holdout"].astype(bool)][
        ["date", "ticker", fwd, "log_turnover"]].copy()
    base["date"] = pd.to_datetime(base["date"])
    D = base.merge(R, on=["date", "ticker"], how="inner")
    D = D[np.isfinite(D[fwd])]
    if D.empty:
        return pd.DataFrame(), {}

    # liquidity filter, cross-sectionally within each day
    thr = D.groupby("date")["log_turnover"].transform(
        lambda s: s.quantile(liquid_pct))
    D = D[D["log_turnover"] >= thr]

    # the market-vol conditioner, from the same index the brief prints
    idx = index_series(P[~P["holdout"].astype(bool)], "equal")
    rv = (idx.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252))
    D["ivol"] = D["date"].map(rv.rank(pct=True))

    D["base"] = D.groupby("date")[fwd].transform("mean")
    D["exc"] = D[fwd] - D["base"]

    edges = {"run_days": _terciles(D["run_days"]),
             "run_z_up": _terciles(D.loc[D["leg"] == "up", "run_z"]),
             "run_z_down": _terciles(D.loc[D["leg"] == "down", "run_z"]),
             "ivol": np.array([1 / 3, 2 / 3])}
    D["bucket"] = bucket_of(D["run_days"], D["run_z"], D["ivol"],
                            edges, D["leg"]).to_numpy()
    return D[D["bucket"].notna()].reset_index(drop=True), edges


def conditional_table(P: pd.DataFrame, R: pd.DataFrame, k: int = 20,
                      draws: int = 200, seed: int = 20260824,
                      liquid_pct: float = LIQUID_PCT
                      ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """What followed bars in each state, on PRE-HOLDOUT data only.

    ``P`` is the price panel, ``R`` the output of :func:`run_state`. Returns
    the table and the bucket edges that produced it.

    TWO OUTCOME COLUMNS, and the second is the honest one. ``fwd`` is the raw
    forward return; ``exc`` is the same return minus that day's cross-sectional
    mean. A bucket can look excellent purely because the states in it cluster
    in bull markets, and ``exc`` removes exactly that. Both are reported.

    THE BASE RATE IS NOT OPTIONAL. Every row carries the mean over all liquid
    reference rows on the SAME dates, so the comparison is against what holding
    anything at all would have returned in the same weather. §11: every result
    table includes a baseline run through the identical pipeline.

    ``n_eff`` divides the distinct-date count by the overlap, because 30,000
    overlapping 20-day windows are not 30,000 observations.

    FIFTY-FOUR CELLS ARE COMPUTED AND THE INTERVALS ARE UNCORRECTED. At 95%,
    between two and three of them clear zero by chance alone. That is what
    :func:`conditional_null` is for, and no cell from this table should be read
    without it.
    """
    rng = np.random.default_rng(seed)
    fwd = f"fwd{k}"
    D, edges = conditional_frame(P, R, k, liquid_pct)
    if D.empty:
        return pd.DataFrame(), {}
    rows = []
    for b, g in D.groupby("bucket"):
        n = len(g)
        n_dates = g["date"].nunique()
        lo, hi = _block_bootstrap(g["date"].to_numpy(), g[fwd].to_numpy(),
                                  g["base"].to_numpy(), draws, rng)
        leg, a, z, m = b.split("|")
        rows.append({
            "bucket": b, "leg": leg, "age": int(a), "ext": int(z),
            "mkt_vol": int(m), "n": n, "n_dates": n_dates,
            "n_eff": int(n_dates / k) if n_dates >= k else 0,
            "run_days_med": float(g["run_days"].median()),
            "run_z_med": float(g["run_z"].median()),
            "fwd_mean": float(g[fwd].mean()),
            "fwd_med": float(g[fwd].median()),
            "p_up": float((g[fwd] > 0).mean()),
            "base_mean": float(g["base"].mean()),
            "exc_mean": float(g["exc"].mean()),
            "diff": float(g[fwd].mean() - g["base"].mean()),
            "diff_lo": lo, "diff_hi": hi})
    T = pd.DataFrame(rows).sort_values(["leg", "age", "ext", "mkt_vol"])
    T.attrs["k"] = k
    return T.reset_index(drop=True), edges


def conditional_null(D: pd.DataFrame, k: int = 20, draws: int = 200,
                     seed: int = 20260824) -> Dict[str, object]:
    """How big the biggest cell would look if the state carried nothing.

    Shuffles the BUCKET LABEL within each date and recomputes every cell's
    excess over the base rate. That preserves each day's cross-section of
    returns, each day's market weather and the overall size of every bucket,
    and destroys only which state met which name — the same construction H11
    and H14 used, for the same reason.

    This is the statistic that decides whether the table means anything. A9
    records the negative control firing at t = +3.55 on two million rows, and
    A10 records a persistence statistic of +0.912 sitting BELOW its own null of
    +0.919. Reading a cell against zero rather than against this distribution
    has produced a confident wrong answer four times in this repo. So the
    return here carries the observed maximum alongside the null's, and the
    count of cells clearing an uncorrected 95% interval alongside the ~2.7
    expected by chance from 54 cells.
    """
    fwd = f"fwd{k}"
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(D["bucket"])
    dcodes = pd.factorize(D["date"])[0]
    y = (D[fwd] - D["base"]).to_numpy(dtype=float)
    nb = len(uniq)

    def cell_diffs(c: np.ndarray) -> np.ndarray:
        n = np.bincount(c, minlength=nb).astype(float)
        s = np.bincount(c, weights=y, minlength=nb)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(n > 0, s / n, np.nan)

    obs = cell_diffs(codes)
    # permute the bucket label within each date: sort by (date, random) and
    # scatter back, the lexsort trick persistence.py uses
    nulls = np.empty(draws)
    spread = np.empty(draws)
    for i in range(draws):
        order = np.lexsort((rng.random(len(codes)), dcodes))
        perm = np.empty_like(codes)
        perm[np.argsort(dcodes, kind="mergesort")] = codes[order]
        d = cell_diffs(perm)
        nulls[i] = np.nanmax(np.abs(d))
        spread[i] = np.nanstd(d)
    return {"n_cells": nb,
            "obs_max_abs": float(np.nanmax(np.abs(obs))),
            "obs_spread": float(np.nanstd(obs)),
            "null_max_abs_mean": float(nulls.mean()),
            "null_max_abs_p95": float(np.percentile(nulls, 95)),
            "null_spread_mean": float(spread.mean()),
            "p_max": float((nulls >= np.nanmax(np.abs(obs))).mean()),
            "expected_false_cells": 0.05 * nb}


AGE_LABEL = {0: "young", 1: "middling", 2: "old"}
EXT_LABEL = {0: "shallow", 1: "middling", 2: "stretched"}
VOL_LABEL = {0: "calm", 1: "normal", 2: "turbulent"}

#: Fewer rows than this and the bucket is reported as insufficient rather than
#: given a mean. §9.6's rule for below-threshold cases, applied here.
MIN_BUCKET_N = 500


def describe_bucket(b: str) -> str:
    """Plain words for a bucket key, so the output is readable without a map.

    The extension label is REVERSED for declines. ``run_z`` is signed, so a
    decline's most stretched cell is its most negative one — tercile 0 — while
    an advance's is tercile 2. Printing "shallow" over the deepest cell would
    invert the reader's whole understanding of the table.
    """
    leg, a, z, m = b.split("|")
    z = int(z) if leg == "up" else 2 - int(z)
    return (f"{'advance' if leg == 'up' else 'decline'}, "
            f"{AGE_LABEL[int(a)]} and {EXT_LABEL[z]}, "
            f"{VOL_LABEL[int(m)]} market")


# ==========================================================================
# 4. CANDIDATES — a ranking, with its measured post-cost result attached
# ==========================================================================
#: H13's measured result per feature, from reports/phase3_price_features.md.
#: Carried here so a candidate list can never be printed without it. If these
#: are re-measured the source of truth is that memo, not this dict.
H13_NET = {
    "rev5": "net-negative at every horizon",
    "mom12_1": "net-negative at every horizon",
    "lowvol": "net-negative at every horizon",
    "amihud60": "net-negative at every horizon",
    "volz20": "net-negative at every horizon; sign was AGAINST prediction",
    "hi52": "net-negative at every horizon",
    "atr_mom20": "net-negative at every horizon",
    "squeeze": "registered as PREDICTED-NULL and it fired anyway (t +3.55) — "
               "the negative control, and the reason a t-statistic on this "
               "panel is not evidence",
}


def candidates(P: pd.DataFrame, day: pd.Timestamp, n: int = 10,
               liquid_pct: float = LIQUID_PCT,
               features: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Top names on each registered feature, in the feature's predicted sign.

    ONE FEATURE AT A TIME, deliberately. A composite of the eight would be a
    new signal that has never been tested, and dressing it as a summary of
    eight tested ones is how an untested thing acquires borrowed credibility.
    The overlap count in :func:`candidate_overlap` is the honest version of
    "which names show up more than once".

    ``squeeze`` is excluded from the ranking because it was registered as
    predicted-null; it has no directional sign to rank by.
    """
    C = P[pd.to_datetime(P["date"]) == day].copy()
    if C.empty:
        return pd.DataFrame()
    C = C[C["tradeable"].astype(bool)]
    thr = C["log_turnover"].quantile(liquid_pct)
    C = C[C["log_turnover"] >= thr]
    feats = [f for f in (features or PREDICTED) if PREDICTED.get(f, 0) != 0]
    rows = []
    for f in feats:
        if f not in C:
            continue
        sign = PREDICTED[f]
        g = C[np.isfinite(C[f])].copy()
        if g.empty:
            continue
        # PREDICTED[f] = +1 means a HIGH value was predicted to precede a high
        # return, so the list wants the high end. pandas ranks the largest
        # value at pct 1.0 under ascending=True, which reads backwards and was
        # got backwards first time: the mom12_1 list came back holding the
        # year's biggest LOSERS under a +1 momentum sign.
        g["_rank"] = g[f].rank(pct=True, ascending=(sign > 0))
        g = g.sort_values("_rank", ascending=False).head(n)
        for _, r in g.iterrows():
            rows.append({"feature": f, "sign": sign, "ticker": r["ticker"],
                         "close": float(r["close"]), "value": float(r[f]),
                         "pct": float(r["_rank"]),
                         "turnover": float(np.exp(r["log_turnover"])),
                         "h13": H13_NET.get(f, "")})
    return pd.DataFrame(rows)


def candidate_overlap(C: pd.DataFrame) -> pd.DataFrame:
    """Names appearing on more than one feature's list, and on which."""
    if C.empty:
        return pd.DataFrame()
    g = C.groupby("ticker").agg(n_lists=("feature", "nunique"),
                                lists=("feature", lambda s: sorted(set(s))))
    return g[g["n_lists"] > 1].sort_values("n_lists", ascending=False
                                           ).reset_index()


def current_states(P: pd.DataFrame, R: pd.DataFrame, day: pd.Timestamp,
                   T: pd.DataFrame, edges: Dict[str, Sequence[float]],
                   min_value: float = MIN_VALUE,
                   liquid_pct: float = LIQUID_PCT) -> pd.DataFrame:
    """Every liquid name today, its run state, and what followed states like it.

    THIS IS THE SECTION THAT ANSWERS "over or just started", and it answers it
    the only way the data can: by saying where the move sits and what the
    historical distribution from there looked like. It does not say what will
    happen. The join is to the SAME buckets and the SAME edges the pre-holdout
    table was built with, so the number attached to a name is a historical
    frequency and not a fit to that name.

    ``diff`` and its interval come from the table; ``cost`` is what one round
    trip would cost at this name's price today. The two are printed side by
    side because a historical excess smaller than the cost of capturing it is
    the entire finding of this repo, and it should be visible per row rather
    than argued in a footnote.
    """
    C = P[pd.to_datetime(P["date"]) == day].copy()
    C = C[C["tradeable"].astype(bool)]
    lt = C["log_turnover"]
    C = C[(lt >= lt.quantile(liquid_pct)) & (np.exp(lt) >= min_value)]
    S = R[R["date"] == day]
    D = C[["ticker", "close", "log_turnover"]].merge(
        S[["ticker", "leg", "run_days", "run_ret", "run_z", "give_back",
           "run_pct", "give_pct"]], on="ticker", how="inner")
    if D.empty:
        return D
    # the market-vol conditioner must be the SAME quantity the table used: a
    # percentile of the pre-holdout index-vol distribution, not of this year's
    ref = P[~P["holdout"].astype(bool)]
    rv = (index_series(ref, "equal").pct_change()
          .rolling(20, min_periods=20).std() * np.sqrt(252))
    live = (index_series(upto(P, day), "equal").pct_change()
            .rolling(20, min_periods=20).std() * np.sqrt(252)).iloc[-1]
    D["ivol"] = _pct_rank(rv.to_numpy(), float(live))
    D["bucket"] = bucket_of(D["run_days"], D["run_z"], D["ivol"],
                            edges, D["leg"]).to_numpy()
    cols = ["bucket", "n", "n_eff", "fwd_mean", "base_mean", "diff",
            "diff_lo", "diff_hi", "p_up"]
    D = D.merge(T[[c for c in cols if c in T]], on="bucket", how="left")
    D["cost"] = [cost_bar(p, day)["total"] for p in D["close"]]
    D["net"] = D["diff"] - D["cost"]
    D["what"] = D["bucket"].map(
        lambda b: describe_bucket(b) if isinstance(b, str) else "")
    return D.sort_values("diff", ascending=False).reset_index(drop=True)


def cost_bar(price: float, day, fee: float = ROUND_TRIP_FEE) -> Dict[str, float]:
    """What one round trip costs at this price on this date, as a fraction.

    Fee is A5's schedule. The spread term is half a tick each way from the
    point-in-time fraksi harga, which A2 records is a FLOOR rather than an
    estimate — it assumes a one-tick-wide book, generous on anything but a
    large cap. So every net number computed against this bar is optimistic.
    """
    try:
        tick = reference.tick_size(float(price), day)
    except reference.OutsideCoverage:
        return {"fee": fee, "spread": np.nan, "total": np.nan}
    spread = float(tick / price) if price > 0 else np.nan   # half a tick x 2
    return {"fee": fee, "spread": spread, "total": fee + spread}


# ==========================================================================
# the cached reference tables
# ==========================================================================
TABLE_DIR = os.path.join("data", "spine")


def table_path(k: int) -> str:
    return os.path.join(TABLE_DIR, f"conditional_fwd{k}.json")


def build_tables(P: pd.DataFrame, ks: Sequence[int] = (5, 20),
                 draws: int = 200, null_draws: int = 200,
                 seed: int = 20260824) -> Dict[int, Dict[str, object]]:
    """Compute and cache the conditional tables, their edges and their nulls.

    Expensive — a minute or so per horizon — and it only changes when the
    panel does, so the brief reads it rather than recomputing it twice a day.
    Written atomically: a half-written reference table that still parses is
    worse than none, because the brief would carry on quoting it.
    """
    R = run_state(P)
    out = {}
    os.makedirs(TABLE_DIR, exist_ok=True)
    for k in ks:
        D, edges = conditional_frame(P, R, k)
        T, _ = conditional_table(P, R, k, draws=draws, seed=seed)
        N = conditional_null(D, k, draws=null_draws, seed=seed)
        blob = {"k": k, "built": dt.date.today().isoformat(),
                "n_rows": int(len(D)),
                "date_min": str(pd.to_datetime(D["date"]).min().date()),
                "date_max": str(pd.to_datetime(D["date"]).max().date()),
                "edges": {a: np.asarray(b, dtype=float).tolist()
                          for a, b in edges.items()},
                "null": N, "table": T.to_dict(orient="records")}
        tmp = table_path(k) + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(blob, fh)
        os.replace(tmp, table_path(k))
        out[k] = blob
    return out


def load_table(k: int) -> Optional[Dict[str, object]]:
    p = table_path(k)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        blob = json.load(fh)
    blob["table"] = pd.DataFrame(blob["table"])
    blob["edges"] = {a: np.asarray(b, dtype=float)
                     for a, b in blob["edges"].items()}
    return blob
