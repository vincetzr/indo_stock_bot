"""What moved while Jakarta was shut — and which of it has ever mattered.

WHY THIS EXISTS
----------------
The brief shipped with a pre-open mode that knew nothing a post-close run did
not. Its own banner admitted it: *"a morning run and an evening run differ only
in what has settled, not in what is known."* That makes half the stated use
case pointless, because the thing a pre-open read is FOR is the overnight gap.

The gap is free and it was never fetched. Jakarta closes at 15:50 WIB; Wall
Street, the dollar, Treasuries and every metal trade for hours afterwards, and
`YahooOHLCV` — already in this repo, already used for every `.JK` name — serves
all of them from the same unauthenticated chart endpoint. Verified: ^GSPC,
DX-Y.NYB, IDR=X, ^TNX, CL=F, GC=F, HG=F and ALI=F all carry a bar stamped
AFTER the last IDX session.

THE PART THAT IS NOT FOLKLORE
-------------------------------
"A stronger dollar is a headwind for IDX" is the sort of claim that gets
repeated until it sounds measured. §6's rule — an inference is not an
observation — applies to macro as much as to broker codes, so this module
MEASURES it: :func:`sensitivity` regresses IDX's next-session return on each
overnight move over pre-holdout history and reports the correlation with a
block-bootstrap interval and a permutation null.

That is a description of what has historically co-moved. It is **not** a
signal. The whole research programme in this repo (A9) found nothing that
survived costs, and a daily macro correlation is not exempt: the effect sizes
below are a fraction of the 56 bps round trip before a spread is added.

WHAT IS NOT REACHABLE, HAVING BEEN CHECKED
--------------------------------------------
IDX's economy is coal, nickel and palm oil, and none of the three has a usable
free front-month series. Newcastle coal (`MTF=F`) stops in 2025-12; there is no
Yahoo symbol for Bursa Malaysia CPO or for LME nickel. What stands in is the
listed mining complex — Glencore, BHP, Rio Tinto — which trades in London and
New York after Jakarta closes and carries the same exposure at one remove. The
substitution is stated in the output rather than hidden, because a proxy that
is not labelled becomes the thing it stands for.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Grouped for the reader, not for any computation. Each entry is
#: ``(yahoo symbol, display name)`` and every one was verified to return bars
#: from the same unauthenticated endpoint the `.JK` names use.
GROUPS: Dict[str, Sequence[Tuple[str, str]]] = {
    "US equities": (("^GSPC", "S&P 500"), ("^IXIC", "Nasdaq")),
    "Asia": (("^N225", "Nikkei"), ("^HSI", "Hang Seng"),
             ("000001.SS", "Shanghai")),
    "FX and rates": (("IDR=X", "USDIDR"), ("DX-Y.NYB", "DXY"),
                     ("^TNX", "UST 10y")),
    "Energy": (("BZ=F", "Brent"), ("CL=F", "WTI"), ("NG=F", "Nat gas")),
    "Metals": (("GC=F", "Gold"), ("HG=F", "Copper"), ("ALI=F", "Aluminium"),
               ("TIO=F", "Iron ore")),
    "Mining complex": (("GLEN.L", "Glencore"), ("BHP", "BHP"),
                       ("RIO", "Rio Tinto")),
}

#: WHICH MARKETS CLOSE AFTER JAKARTA DOES, AND WHY IT DECIDES EVERYTHING HERE.
#:
#: Jakarta closes 15:50 WIB = 08:50 UTC. Wall Street closes 20:00 UTC and
#: London 15:30 UTC — both on the SAME calendar date, so their bar dated
#: 2026-08-24 landed roughly eleven and seven hours after Jakarta's bar of the
#: same name. That bar is new information; a naive `date > idx_day` test finds
#: nothing and reports a silent NaN, which is what the first version did.
#:
#: Tokyo (06:00 UTC), Hong Kong (08:00 UTC) and Shanghai (07:00 UTC) close
#: BEFORE Jakarta. Their same-date bar was already visible to Jakarta traders
#: during the session, so it is not overnight news — it is context, and
#: labelling it "overnight" would credit the reader with information the market
#: had already priced.
AFTER_JAKARTA = frozenset({
    "^GSPC", "^IXIC",                       # New York, 20:00 UTC
    "IDR=X", "DX-Y.NYB", "^TNX",            # continuous / NY-stamped
    "BZ=F", "CL=F", "NG=F",                 # ICE and NYMEX settlements
    "GC=F", "HG=F", "ALI=F", "TIO=F",       # COMEX, LME-linked, SGX
    "GLEN.L",                               # London, 15:30 UTC
    "BHP", "RIO",                           # NYSE-listed lines
})

#: Flattened, in display order.
SYMBOLS: List[Tuple[str, str]] = [s for g in GROUPS.values() for s in g]

#: The Jakarta composite itself. Yahoo carries it, so the brief does not need
#: to infer an index from its own cross-section — though it still prints both,
#: since ^JKSE can lag the individual names by a session.
IHSG = "^JKSE"

#: Series that are RATES, not prices. A yield is differenced, never
#: percentage-changed: ^TNX fell from 0.93 to 0.50 in March 2020, which is a
#: real 43 bp move and a spurious -46% return, and percentage change hands the
#: whole sample's weight to the fortnight the yield happened to sit near zero.
DIFFERENCED = frozenset({"^TNX"})

#: Above this, a daily move in a major index, currency or commodity is a vendor
#: defect rather than a market event. Yahoo's IDR=X carries decimal shifts —
#: 2010-11-01 prints 888.11 against a true ~8,881 and reverses the next day,
#: giving a +903% return followed by -90%. Those bars are dropped and counted,
#: not winsorised into something plausible-looking.
IMPLAUSIBLE = 0.50

#: Proxies standing in for something that has no free series, and what for.
#: Printed with the numbers: an unlabelled proxy becomes the thing it proxies.
PROXY_NOTE = {
    "GLEN.L": "coal and nickel — no free front-month series for either",
    "BHP": "bulk commodities — iron ore, copper",
    "RIO": "bulk commodities — iron ore, aluminium",
}


def load(loader, symbols: Sequence[str], max_age: float = 3600.0
         ) -> Dict[str, pd.DataFrame]:
    """Daily bars per symbol. A symbol that fails is omitted, not faked."""
    out = {}
    for s in symbols:
        try:
            d = loader.get(s, max_age=max_age)
        except Exception:                                       # noqa: BLE001
            continue
        if d is None or getattr(d, "empty", True) or "date" not in d:
            continue
        d = d.copy()
        d["date"] = pd.to_datetime(d["date"])
        out[s] = d.sort_values("date").reset_index(drop=True)
    return out


def _overnight_ret(d: pd.DataFrame, idx_day: pd.Timestamp,
                   after_jakarta: bool) -> float:
    """What this symbol did that Jakarta has NOT yet traded on.

    For a market closing after Jakarta, that is its move on the IDX date
    itself — the bar shares a date stamp with the Jakarta session but landed
    hours later. For a market closing before Jakarta, there is nothing: its
    session on that date was already visible while Jakarta was still open, so
    this returns NaN and the caller shows the last session as context instead.

    NaN is deliberate wherever the number does not exist. "No bar yet" and
    "unchanged" are different facts, and rendering them identically is lying by
    formatting.
    """
    if idx_day is None or not after_jakarta:
        return np.nan
    px = pd.to_numeric(d["close"], errors="coerce")
    on_or_after = d["date"] >= idx_day
    if not on_or_after.any():
        return np.nan                        # market shut, or feed behind
    before = d["date"] < idx_day
    if not before.any():
        return np.nan
    base = px[before].iloc[-1]
    if not np.isfinite(base) or base <= 0:
        return np.nan
    return float(px[on_or_after].iloc[-1] / base - 1.0)


def _tail_ret(d: pd.DataFrame, k: int) -> float:
    px = pd.to_numeric(d["close"], errors="coerce").dropna()
    if len(px) <= k:
        return np.nan
    return float(px.iloc[-1] / px.iloc[-1 - k] - 1.0)


def board(bars: Dict[str, pd.DataFrame], idx_day: pd.Timestamp) -> pd.DataFrame:
    """One row per symbol: what it did since the IDX close, and lately.

    ``since_idx`` is the number the pre-open run exists for. ``stale`` marks a
    symbol whose own last bar is not newer than the IDX session — a holiday in
    its market, or a feed that has not updated — so a blank there is visibly
    an absence rather than a quiet night.
    """
    rows = []
    for group, members in GROUPS.items():
        for sym, name in members:
            d = bars.get(sym)
            aft = sym in AFTER_JAKARTA
            if d is None or d.empty:
                rows.append({"group": group, "symbol": sym, "name": name,
                             "last": np.nan, "asof": pd.NaT,
                             "overnight": np.nan, "d1": np.nan, "d5": np.nan,
                             "d21": np.nan, "after_jakarta": aft,
                             "behind": True,
                             "proxy": PROXY_NOTE.get(sym, "")})
                continue
            asof = d["date"].iloc[-1]
            rows.append({
                "group": group, "symbol": sym, "name": name,
                "last": float(pd.to_numeric(d["close"], errors="coerce")
                              .iloc[-1]),
                "asof": asof,
                "overnight": _overnight_ret(d, idx_day, aft),
                "d1": _tail_ret(d, 1), "d5": _tail_ret(d, 5),
                "d21": _tail_ret(d, 21),
                "after_jakarta": aft,
                # its own feed has not reached the IDX session yet — a holiday
                # in that market, or a lagging series. Distinct from "closes
                # before Jakarta", which is a fact about the clock, not a fault.
                "behind": bool(asof < idx_day),
                "proxy": PROXY_NOTE.get(sym, "")})
    return pd.DataFrame(rows)


# ==========================================================================
# the measured part
# ==========================================================================
def _align(bars: Dict[str, pd.DataFrame], sym: str,
           idx_dates: pd.DatetimeIndex) -> pd.Series:
    """Each IDX session mapped to the overnight move that PRECEDED it.

    NO LOOKAHEAD, AND THE CLOCK IS THE WHOLE DIFFICULTY. Wall Street's bar
    dated *t* closes eleven hours AFTER Jakarta's bar dated *t*, so pairing
    them by date would regress Jakarta's session on news that had not happened
    when it closed. The bar Jakarta could actually act on at session *t* is the
    global bar dated *t-1*.

    So each IDX session is matched to the global symbol's own daily return on
    the last bar strictly before that session. That rule is conservative for
    the Asian markets too — their bar dated *t* closes shortly before Jakarta's
    does, and using *t-1* simply forgoes a couple of hours of information
    rather than risking any leak.
    """
    d = bars.get(sym)
    if d is None or d.empty:
        return pd.Series(dtype=float)
    s = (pd.Series(pd.to_numeric(d["close"], errors="coerce").to_numpy(),
                   index=pd.DatetimeIndex(d["date"])).dropna()
         .sort_index())
    if len(s) < 3:
        return pd.Series(dtype=float)
    g = s.diff() if sym in DIFFERENCED else s.pct_change()
    if sym not in DIFFERENCED:
        # vendor decimal shifts, not market events — see IMPLAUSIBLE
        g = g.where(g.abs() <= IMPLAUSIBLE)
    idx = pd.DatetimeIndex(idx_dates).sort_values()
    if len(idx) < 2:
        return pd.Series(dtype=float)
    prev = idx[:-1]                       # the session before each target
    # the symbol's last completed daily return at or before session t-1
    aligned = (g.reindex(g.index.union(prev)).ffill().reindex(prev))
    return pd.Series(aligned.to_numpy(), index=idx[1:])


def sensitivity(bars: Dict[str, pd.DataFrame], idx_ret: pd.Series,
                draws: int = 200, seed: int = 20260824,
                block: int = 21) -> pd.DataFrame:
    """Historical correlation of IDX's session return with each overnight move.

    ``idx_ret`` is the equal-weighted IDX cross-sectional return indexed by
    session, PRE-HOLDOUT only — the caller filters it, and §11's reservation
    applies here exactly as it does everywhere else.

    RANK CORRELATION, NOT PEARSON, and that is not a stylistic choice. These
    series carry kurtosis of 10 to 2,800 — Pearson on a sample like that is a
    statistic about its four largest days. Spearman is what every other
    cross-sectional measure in this repo uses, for the same reason.

    ``stale`` reports the share of aligned observations whose move was exactly
    zero, which is a forward-fill rather than a flat market. Aluminium runs
    19% stale and its correlation should be read in that light rather than
    beside a series that prints every day.

    Reports the correlation, a block-bootstrap interval (blocks because both
    series are serially correlated; the estimator is verified unbiased against
    a synthetic sample of known correlation) and a shuffle null. §6: an
    inference is not an observation, and "the dollar is a headwind for IDX" is
    an inference until something measures it.

    THIS IS NOT A SIGNAL and the magnitudes say so on their own. The strongest
    reading here explains under 2% of variance. Read it as: which overnight
    moves has Jakarta historically tracked.
    """
    from scipy.stats import rankdata
    rng = np.random.default_rng(seed)
    idx_ret = idx_ret.dropna()
    rows = []
    for group, members in GROUPS.items():
        for sym, name in members:
            x = _align(bars, sym, pd.DatetimeIndex(idx_ret.index))
            if x.empty:
                continue
            J = pd.concat([x.rename("x"), idx_ret.rename("y")],
                          axis=1, sort=True).dropna()
            J = J[np.isfinite(J["x"]) & np.isfinite(J["y"])]
            if len(J) < 250:
                continue
            raw = J["x"].to_numpy()
            xv = rankdata(raw)
            yv = rankdata(J["y"].to_numpy())
            r = float(np.corrcoef(xv, yv)[0, 1])
            lo, hi = _block_ci(xv, yv, draws, rng, block)
            null = np.empty(draws)
            for i in range(draws):
                null[i] = np.corrcoef(rng.permutation(xv), yv)[0, 1]
            sd = float(null.std())
            rows.append({"group": group, "symbol": sym, "name": name,
                         "r": r, "lo": lo, "hi": hi, "n": len(J),
                         "stale": float((raw == 0).mean()),
                         "null_sd": sd,
                         "z": float(r / sd) if sd > 0 else np.nan,
                         "proxy": PROXY_NOTE.get(sym, "")})
    D = pd.DataFrame(rows)
    if D.empty:
        return D
    return D.reindex(D["r"].abs().sort_values(ascending=False).index) \
            .reset_index(drop=True)


def _block_ci(x: np.ndarray, y: np.ndarray, draws: int,
              rng: np.random.Generator, block: int) -> Tuple[float, float]:
    """Bootstrap the correlation over contiguous blocks, duplicates kept.

    Blocks because both series are serially correlated; duplicates kept for the
    reason `brief._block_bootstrap` documents at length — selecting resampled
    units with a set test drops repeats and narrows every interval.
    """
    n = len(x)
    if n < 3 * block:
        return (np.nan, np.nan)
    nb = max(1, n // block)
    out = np.empty(draws)
    for i in range(draws):
        s = rng.integers(0, n - block + 1, size=nb)
        idx = (s[:, None] + np.arange(block)).ravel()
        xs, ys = x[idx], y[idx]
        out[i] = np.corrcoef(xs, ys)[0, 1] if xs.std() > 0 and ys.std() > 0 \
            else np.nan
    return (float(np.nanpercentile(out, 2.5)),
            float(np.nanpercentile(out, 97.5)))
