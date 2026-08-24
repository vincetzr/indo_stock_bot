"""§9.4 behavioural fingerprints, and §9.5's two checks on them.

WHAT THIS MEASURES, AND WHAT IT DELIBERATELY DOES NOT
-------------------------------------------------------
H11 found that a broker code's *margin rank* does not persist. This module asks
a different question: is the broker's **behaviour** stable? A firm can have a
completely stable business model — always crossing, always concentrated in a
handful of names — while having no stable edge whatsoever. §9.4's fingerprint
is the description of the business; §9.3's margin was the description of the
result. Conflating them is how a stable *style* gets reported as a stable
*edge*, and this module never computes the second.

THE VWAP HERE IS NOT IDX'S VWAP
---------------------------------
Execution edge needs the day's average traded price. The combined range files
carry no footer — `pullback_flow.fetch_window` calls `parse_table` and never
`attach_totals` (A7) — so there is no published VWAP for this history. What is
available is the value-weighted average price across the brokers the top-10 cut
*shows*, and that is what is computed. It is called ``visible_vwap`` in every
name and docstring so it is never mistaken for the real thing, and ``censor``
travels beside it as the direct bound on how much is missing.

§9.4'S MANDATORY BIAS CORRECTION STILL APPLIES
------------------------------------------------
"A broker who is a large share of the day's volume pulls VWAP toward their own
price, shrinking the measured edge toward zero." That is true of a visible VWAP
too — more so, since the visible pool is smaller and each broker is a larger
share of it. So the comparison always excludes the broker's own trades. A4
records that skipping this made every large broker look identically neutral.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

#: Fewest windows a broker needs in a year before it is fingerprinted. Below
#: this the metrics are one or two prints of noise.
MIN_WINDOWS = 8

#: And this much gross value, so a code that printed once is not profiled
#: alongside one that traded all year.
MIN_GROSS = 1e9

#: The fingerprint, in the order every output uses.
METRICS = ("cross", "hhi", "edge_buy", "edge_sell", "ar1", "share", "censor")

#: Which metrics describe the FIRM (its business model) rather than a data
#: artefact. Q1's prediction is about these.
STYLE = ("cross", "hhi", "edge_buy", "edge_sell", "ar1", "share")


def visible_vwap(g: pd.DataFrame, exclude: Optional[str] = None) -> float:
    """Value-weighted average price across the brokers the top-10 cut shows.

    NOT IDX's published VWAP — see the module docstring. ``exclude`` drops one
    broker's own trades, which §9.4 requires: a broker that is a large share of
    the visible pool otherwise pulls the benchmark toward its own price and
    measures itself as neutral by construction.
    """
    d = g if exclude is None else g[g["broker"] != exclude]
    lots = float(d["buy_lot"].sum() + d["sell_lot"].sum())
    val = float(d["buy_val"].sum() + d["sell_val"].sum())
    if lots <= 0 or val <= 0:
        return np.nan
    return val / (lots * 100.0)


def execution_edges(D: pd.DataFrame) -> pd.DataFrame:
    """Per row: buy and sell edge against a SELF-EXCLUDED visible VWAP, in bps.

    Positive buy edge = bought below what everyone else paid, i.e. patient,
    working the order. Negative = paying up, i.e. urgent and liquidity-taking.
    §9.4 calls this the most informative and most neglected metric, and the
    sign convention here is its.

    A4's exact formula, applied to the visible pool::

        VWAP_excluding_b = (total_value - b_value) / ((total_lot - b_lot) x 100)

    Average prices are DERIVED as value / (lots x 100) rather than read from a
    column: the cached window frame keeps values and lots, and the average is
    exactly their ratio, so re-reading 36,000 files to recover a column that is
    already implied would be waste.

    Vectorised with groupby transforms. The obvious per-row loop is 378,689
    iterations over 30,000 groups and takes longer than the whole rest of the
    analysis put together.
    """
    d = D.copy()
    for c in ("buy_lot", "buy_val", "sell_lot", "sell_val"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["_lot"] = d["buy_lot"] + d["sell_lot"]
    d["_val"] = d["buy_val"] + d["sell_val"]

    g = d.groupby(["ticker", "window_end"], sort=False)
    d["_n"] = g["_lot"].transform("size")
    tot_lot = g["_lot"].transform("sum")
    tot_val = g["_val"].transform("sum")

    rem_lot = tot_lot - d["_lot"]
    rem_val = tot_val - d["_val"]
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(rem_lot > 0, rem_val / (rem_lot * 100.0), np.nan)
        buy_avg = np.where(d["buy_lot"] > 0,
                           d["buy_val"] / (d["buy_lot"] * 100.0), np.nan)
        sell_avg = np.where(d["sell_lot"] > 0,
                            d["sell_val"] / (d["sell_lot"] * 100.0), np.nan)
        eb = np.where(np.isfinite(vwap) & (vwap > 0) & np.isfinite(buy_avg),
                      10000.0 * (vwap - buy_avg) / vwap, np.nan)
        es = np.where(np.isfinite(vwap) & (vwap > 0) & np.isfinite(sell_avg),
                      10000.0 * (sell_avg - vwap) / vwap, np.nan)

    # A window showing fewer than three brokers leaves almost nothing to
    # compare against once the broker itself is removed.
    thin = d["_n"].to_numpy() < 3
    eb[thin] = np.nan
    es[thin] = np.nan
    return pd.DataFrame({"broker": d["broker"].to_numpy(),
                         "ticker": d["ticker"].to_numpy(),
                         "window_end": d["window_end"].to_numpy(),
                         "edge_buy": eb, "edge_sell": es,
                         "buy_lot": d["buy_lot"].to_numpy(),
                         "sell_lot": d["sell_lot"].to_numpy()})


def _hhi(x: np.ndarray) -> float:
    s = x.sum()
    if s <= 0:
        return np.nan
    p = x / s
    return float((p ** 2).sum())


def _ar1_sign(x: np.ndarray) -> float:
    """AR(1) of the SIGN of net-buy: does the broker keep pushing one way?

    On the sign rather than the level because the level is dominated by size,
    and §9.4's question here is horizon — persistence versus alternation — not
    magnitude.
    """
    s = np.sign(x)
    s = s[np.isfinite(s)]
    if len(s) < 6 or s.std() == 0:
        return np.nan
    a, b = s[:-1], s[1:]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def fingerprints(D: pd.DataFrame, edges: pd.DataFrame,
                 period: str = "year") -> pd.DataFrame:
    """One fingerprint row per (broker, period).

    ``D`` is the per (broker, ticker, window) frame; ``edges`` is the output of
    :func:`execution_edges`. Both are joined rather than recomputed so the
    expensive self-excluded VWAP pass happens once.
    """
    d = D.copy()
    d[period] = pd.to_datetime(d["window_end"]).dt.year
    d["gross"] = d["buy_val"] + d["sell_val"]
    d["net"] = d["buy_val"] - d["sell_val"]
    d["two_sided"] = (d["buy_lot"] > 0) & (d["sell_lot"] > 0)

    e = edges.copy()
    if not e.empty:
        e[period] = pd.to_datetime(e["window_end"]).dt.year

    total = d.groupby(period)["gross"].sum()
    rows = []
    for (b, p), g in d.groupby(["broker", period]):
        n_win = g["window_end"].nunique()
        gross = float(g["gross"].sum())
        if n_win < MIN_WINDOWS or gross < MIN_GROSS:
            continue
        bv, sv = float(g["buy_val"].sum()), float(g["sell_val"].sum())
        per_win = g.groupby("window_end")["net"].sum().to_numpy()
        r = {
            "broker": b, period: p, "n_windows": n_win, "gross": gross,
            "cross": (min(bv, sv) / max(bv, sv)) if max(bv, sv) > 0 else np.nan,
            "hhi": _hhi(g.groupby("ticker")["gross"].sum().to_numpy()),
            "ar1": _ar1_sign(per_win),
            "share": float(np.log10(gross / total.get(p, np.nan)))
                     if total.get(p, 0) > 0 else np.nan,
            "censor": float(1.0 - g["two_sided"].mean()),
        }
        if not e.empty:
            ge = e[(e["broker"] == b) & (e[period] == p)]
            r["edge_buy"] = _wmean(ge, "edge_buy", "buy_lot")
            r["edge_sell"] = _wmean(ge, "edge_sell", "sell_lot")
        else:
            r["edge_buy"] = r["edge_sell"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def _wmean(g: pd.DataFrame, col: str, weight: str) -> float:
    """Lot-weighted mean, so one small print does not outrank a year of size."""
    if g.empty:
        return np.nan
    v = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(g[weight], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not m.any() or w[m].sum() <= 0:
        return np.nan
    return float((v[m] * w[m]).sum() / w[m].sum())


def standardise(F: pd.DataFrame, cols: Sequence[str] = METRICS,
                period: str = "year") -> pd.DataFrame:
    """Z-score each metric WITHIN each period.

    Within-period, not pooled: a pooled z-score would let a market-wide drift
    in, say, crossing show up as every broker changing style at once, and Q2 is
    specifically about whether brokers become less distinguishable FROM EACH
    OTHER. Standardising inside the year removes the common movement and leaves
    the cross-section, which is the thing being measured.
    """
    out = F.copy()
    for c in cols:
        if c not in out:
            continue
        g = out.groupby(period)[c]
        out[c + "_z"] = (out[c] - g.transform("mean")) / g.transform("std")
    return out


def distinctiveness(F: pd.DataFrame, cols: Sequence[str] = METRICS,
                    period: str = "year") -> pd.DataFrame:
    """Mean pairwise distance between brokers' standardised fingerprints.

    §9.5: "Plot fingerprint distinctiveness by year. If it decays as
    order-splitting spreads, that is a real and important finding about the
    dataset's shelf life. Report it prominently rather than averaging it away."

    Standardising within the period makes the scale comparable across years by
    construction, so a fall in mean distance is a fall in how separable brokers
    are from one another — not a fall in the units.
    """
    Z = standardise(F, cols, period)
    zc = [c + "_z" for c in cols if c + "_z" in Z]
    rows = []
    for p, g in Z.groupby(period):
        X = g[zc].to_numpy(dtype=float)
        ok = np.isfinite(X).all(axis=1)
        X = X[ok]
        if len(X) < 5:
            continue
        d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        iu = np.triu_indices(len(X), k=1)
        rows.append({period: p, "n_brokers": len(X),
                     "mean_distance": float(d[iu].mean()),
                     "median_distance": float(np.median(d[iu]))})
    return pd.DataFrame(rows).sort_values(period).reset_index(drop=True)
