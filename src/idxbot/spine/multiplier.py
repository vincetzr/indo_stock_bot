"""The multiplier-cell entry rule, in one place.

WHY THIS MODULE EXISTS AT ALL — a reproducibility failure
-----------------------------------------------------------
The rule was implemented twice: once in the H16 holdout script and once in
``scripts/exit_study.py``. Run on the same date they returned different names —
H16 drew MERI, exit_study drew IMPC — and therefore different headline returns
(+15.1% against +26.3% for the identical buy-and-hold rule).

Neither was wrong. **The rule was under-determined.** It scores each name by the
historical P(2x) of the (price band, liquidity quintile, 60-day-vol quintile)
cell it occupies. There are at most 125 such cells and roughly 800 live names,
so scores are massively tied: on 2025-08-25 only four distinct (p2, p5) pairs
existed in the top thirty and **seventeen names shared the tenth-place value.**
"Take the top ten" then selects ten of seventeen equals by whatever order the
frame happened to be in — a decision made by ``sort_values`` stability, not by
the research.

Two consequences, both structural rather than fixable by tidier code:

* Any number quoted from a ten-name draw carries tie-break variance that is not
  in its stated interval. ``scripts/tie_sensitivity.py`` measures it.
* The only version of the rule two implementations must agree on is the one
  with no tie-break in it: **take every name in the tied group.** That is the
  default here (``tie="all"``). ``tie="first"`` reproduces the old behaviour
  for continuity, and an integer seed draws a random tie-break for the
  sensitivity study.

Everything else — the cells, the liquidity floor, the one-bar entry gap — is
carried over unchanged so earlier results remain comparable.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..report import brief as B
from . import exits as X

#: Close-price bands, in rupiah. Cheap names behave differently and the
#: fraksi-harga spread is a function of this band, so it cannot be dropped.
PX = [0, 50, 200, 1000, 5000, 1e9]
PXL = ["<Rp50", "Rp50-200", "Rp200-1k", "Rp1k-5k", ">Rp5k"]

#: Minimum 20-day median traded value to be considered, in rupiah.
MIN_VALUE = 1e9

#: Cells below this many historical observations are dropped as unestimated.
MIN_CELL = 300

#: The nominal basket size. With ``tie="all"`` the realised basket is larger.
TOP_N = 10

#: Forward horizon used to build the cell statistics, in sessions.
FWD = 251


def edges(x) -> np.ndarray:
    """Quintile edges, strictly increasing so ``pd.cut`` cannot fail on ties."""
    q = np.nanquantile(x, [0, .2, .4, .6, .8, 1.0]).astype(float)
    q[0] -= 1.0
    q[-1] += 1.0
    for i in range(1, len(q)):
        if q[i] <= q[i - 1]:
            q[i] = q[i - 1] + 1e-9
    return q


def build_cells(P: pd.DataFrame, min_cell: int = MIN_CELL
                ) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame]:
    """Historical P(2x) and P(5x) per cell, from PRE-HOLDOUT rows only.

    Returns the labelling function and the cell table, so a live cross-section
    is bucketed with exactly the edges the table was built on.
    """
    ref = P[~P["holdout"].astype(bool)].copy()
    ref["fwd"] = ref.groupby("ticker")["adj_close"].transform(
        lambda s: s.shift(-FWD) / s.shift(-1) - 1.0)
    D = ref[np.isfinite(ref["fwd"]) & ref["tradeable"].astype(bool)
            & (ref["adj_close"] > 0)]
    lq = edges(D["log_turnover"])
    vq = edges(D["vol60"].replace(0, np.nan))

    def cells(df: pd.DataFrame) -> pd.DataFrame:
        o = df.copy()
        o["px"] = pd.cut(o["close"], PX, labels=PXL)
        o["liq"] = pd.cut(o["log_turnover"], lq, labels=list("LMNOP"))
        o["vq"] = pd.cut(o["vol60"], vq, labels=list("ABCDE"))
        return o

    tab = (cells(D).groupby(["px", "liq", "vq"], observed=True)["fwd"]
           .agg(n="size", p2=lambda s: (s >= 1).mean(),
                p5=lambda s: (s >= 4).mean()).reset_index())
    return cells, tab[tab["n"] >= min_cell]


def rank_live(P: pd.DataFrame, as_of: pd.Timestamp, cells, tab,
              min_value: float = MIN_VALUE) -> Tuple[Optional[pd.Timestamp],
                                                     pd.DataFrame]:
    """Every eligible name on ``as_of``, scored and sorted, ties intact.

    No cut is applied here. The caller decides what to do about the ties,
    which is the whole point of splitting this out.
    """
    hist = P[P["date"] <= as_of]
    try:
        day = B.resolve_asof(hist)
    except Exception:                                              # noqa: BLE001
        return (None, pd.DataFrame())
    S = B.snapshot(hist, day)
    C = hist[hist["date"] == day].set_index("ticker")
    cur = S.join(C[["vol60"]].rename(columns={"vol60": "v60"}), how="left")
    cur = cur[cur["log_turnover"].notna()
              & (np.exp(cur["log_turnover"]) >= min_value)]
    if len(cur) < 40:
        return (day, pd.DataFrame())
    live = cells(cur.assign(vol60=cur["v60"]).reset_index())
    M = (live.merge(tab, on=["px", "liq", "vq"], how="inner")
         .drop_duplicates("ticker"))
    if M.empty:
        return (day, M)
    # sort by score, then by ticker so the ORDER IS DEFINED. It is still an
    # arbitrary order — that is the point — but at least it is the same
    # arbitrary order in every process, which frame order was not.
    return (day, M.sort_values(["p2", "p5", "ticker"],
                               ascending=[False, False, True])
            .reset_index(drop=True))


def tie_report(M: pd.DataFrame, top_n: int = TOP_N) -> Dict[str, object]:
    """How arbitrary is this cut? Computed, not asserted."""
    if len(M) < top_n:
        return {}
    key = ["p2", "p5"]
    cut = tuple(M.iloc[top_n - 1][key])
    tied = M[[tuple(r) == cut for r in M[key].to_numpy()]]
    inside = int((M.index[:top_n].isin(tied.index)).sum())
    return {"cut_value": cut,
            "n_tied_at_cut": int(len(tied)),
            "taken_from_tied": inside,
            "distinct_scores_top30": int(
                len(M.head(30)[key].drop_duplicates())),
            "arbitrary": bool(len(tied) > inside)}


def select(M: pd.DataFrame, top_n: int = TOP_N, tie: object = "all"
           ) -> pd.DataFrame:
    """Turn the ranked frame into a basket.

    ``tie="all"``    every name in the group straddling the cut is held, so
                     nothing arbitrary decides membership. The basket is larger
                     than ``top_n`` whenever the cut falls inside a tie.
    ``tie="first"``  the old behaviour: the first ``top_n`` rows in sort order.
                     Kept only so prior results can be reproduced.
    ``tie=<int>``    a random tie-break with that seed, for the sensitivity
                     study. Names strictly above the cut are always held;
                     the remainder are drawn from the tied group.

    The returned frame carries an equal ``weight`` column summing to one, so a
    twenty-name tied basket is not silently twice the size of a ten-name one.
    """
    if len(M) < top_n:
        return M.assign(weight=np.nan).head(0)
    key = ["p2", "p5"]
    cut = tuple(M.iloc[top_n - 1][key])
    # M is sorted, so the tied group is contiguous and starts at the first
    # row carrying the cut value.
    same = np.array([tuple(r) == cut for r in M[key].to_numpy()])
    start = int(np.argmax(same))
    above, tied = M.iloc[:start], M[same]

    if tie == "all":
        out = pd.concat([above, tied])
    elif tie == "first":
        out = M.head(top_n)
    else:
        need = top_n - len(above)
        rng = np.random.default_rng(int(tie))
        take = rng.choice(len(tied), size=min(need, len(tied)), replace=False)
        out = pd.concat([above, tied.iloc[np.sort(take)]])
    out = out.copy()
    out["weight"] = 1.0 / len(out)
    return out


def settle_date(dates: Sequence[pd.Timestamp], day: pd.Timestamp,
                horizon: int = X.HORIZON) -> pd.Timestamp:
    """When a cohort opened on ``day`` is fully resolved.

    The walk-forward selector needs this: a cohort whose year is still running
    has no outcome yet, and training a rule choice on it is a look-ahead. Uses
    the panel's own trading calendar rather than a calendar-day guess, and the
    one-bar entry gap is included so the count matches ``path_map``.
    """
    d = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(dates))))
    i = int(d.searchsorted(pd.Timestamp(day), side="right"))
    return d[min(i + horizon, len(d) - 1)]


#: Indicator columns handed to an exit rule, and the key each gets in ``F``.
FEATS = {"close": "close", "adj_high": "high",
         "ema10": "ema10", "ema20": "ema20",
         "ema30": "ema30", "ema50": "ema50", "atr22": "atr22",
         "stoch_k": "stoch_k", "stoch_d": "stoch_d", "tvz20": "tvz20"}


def feature_map(IND: pd.DataFrame, day: pd.Timestamp, tickers: Sequence[str],
                horizon: int = X.HORIZON) -> Dict[str, dict]:
    """``{ticker: {name: array}}`` over the same forward bars as ``path_map``.

    NO WARM-UP PROBLEM, and that is the point of precomputing the indicator
    panel over full history rather than per cohort: an EMA(50) sliced out of a
    continuous series is already seeded at the first forward bar, whereas one
    started at entry would be undefined for fifty sessions and the rule would
    silently be "hold 50 then trade", which is a different rule.
    """
    want = list(dict.fromkeys(tickers))
    fut = (IND[(IND["date"] > day) & IND["ticker"].isin(want)]
           .sort_values(["ticker", "date"]))
    out: Dict[str, dict] = {}
    for t, g in fut.groupby("ticker", sort=False):
        if len(g) < 2:
            continue
        F = {v: g[k].to_numpy(dtype=float)[1:horizon + 1]
             for k, v in FEATS.items() if k in g}
        out[str(t)] = F
    return out


def path_map(P: pd.DataFrame, day: pd.Timestamp, tickers: Sequence[str],
             horizon: int = X.HORIZON
             ) -> Dict[str, Tuple[np.ndarray, float]]:
    """``{ticker: (normalised forward path, round-trip cost)}`` in ONE scan.

    ENTRY IS THE NEXT BAR'S CLOSE. The score is computed from the close of
    ``day``, so filling at that same close is an execution nobody achieves;
    every result in this repo uses the same one-bar gap.

    Built as a map rather than two parallel lists because the tie-break study
    scores the same name under many different baskets — recomputing its path
    per basket turned a two-minute job into an hour-long one.
    """
    want = list(dict.fromkeys(tickers))
    fut = (P[(P["date"] > day) & P["ticker"].isin(want)]
           .sort_values(["ticker", "date"]))
    out: Dict[str, Tuple[np.ndarray, float]] = {}
    for t, g in fut.groupby("ticker", sort=False):
        if len(g) < 2:
            continue
        px = g["adj_close"].astype(float).to_numpy()
        entry = px[0]
        if not np.isfinite(entry) or entry <= 0:
            continue
        cb = B.cost_bar(float(g["close"].iloc[0]), day)
        out[str(t)] = (px[1:horizon + 1] / entry,
                       cb["total"] if np.isfinite(cb["total"]) else X.FEE)
    return out


def paths(P: pd.DataFrame, day: pd.Timestamp, tickers: Sequence[str],
          horizon: int = X.HORIZON) -> Tuple[List[np.ndarray], List[float]]:
    """``path_map`` as two parallel lists, in the caller's ticker order."""
    m = path_map(P, day, tickers, horizon)
    got = [t for t in tickers if t in m]
    return ([m[t][0] for t in got], [m[t][1] for t in got])
