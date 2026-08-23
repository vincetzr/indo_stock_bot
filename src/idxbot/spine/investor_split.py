"""§12 at INVESTOR-CLASS resolution: foreign against domestic.

WHY THE STATISTIC CHANGES SHAPE HERE
--------------------------------------
H11 asked whether a broker code's margin RANK persists, and rank correlation
was the right tool because there were 89 codes to rank. There are two investor
classes. A rank correlation on two items is meaningless, so the question has to
be restated — and restating it moves it much closer to what §12 actually claims:

    Is one class persistently on the profitable side of the other, by enough to
    trade after costs?

That is a LEVEL question with a PERSISTENCE condition attached, not a
cross-sectional ranking. It decomposes into three separable tests, and all
three have to pass before §12's strategy has an instrument:

    1. LEVEL       is the class's timing margin distinguishable from zero, and
                   from a null that destroys the flow-to-return pairing?
    2. PERSISTENCE does the SIGN hold across years, or is a pooled average
                   being carried by two good ones?
    3. SIZE        is |margin| larger than A5's 56 bps round-trip cost?

THE MEASURE
------------
Per ticker and window, using the next window's return so nothing is stamped
after the decision bar::

    timing_pnl  = net_value x forward_return
    margin_bps  = 10000 x sum(timing_pnl) / sum(gross_value)

``net_value`` is the class's buy value minus sell value over the window;
``gross_value`` is buy plus sell. The margin is therefore rupiah earned per
rupiah that class put through, which is §9.3's ``margin_bps`` and is comparable
across classes of wildly different size — foreign is about two thirds of gross
in the liquid names and a rounding error in the illiquid ones.

This is timing, NOT realised P&L. A class that earns the spread intraday reads
exactly zero here while being profitable. H11 carries the same limitation and
it is stated in both places rather than assumed remembered.

WHAT IS AND IS NOT KNOWN ABOUT THE INPUT
------------------------------------------
Both views are top-ten censored, so ``net_value`` from the visible rows is a
lower bound on the class's true net. Two things bound the error:

  - The foreign view is concentrated enough that its top ten is close to all of
    it — measured at 99% on the day store.
  - Foreign net and domestic net are structurally mirror images, because every
    rupiah bought is a rupiah sold. Measured on 28 BBCA sessions they correlate
    **-0.996** and their sum is **2.2% of gross at the median, 7.1% at worst**.
    That residual IS the censoring, and it is the honest error bar on any net.

The view footer's ``tval`` is deliberately not used. It matches neither the
visible buy total, the sell total, nor their mean (9.9%, 21.6%, 15.3% median
gaps), so what it counts for a filtered view is unresolved; a coverage ratio
built on it would look rigorous and mean nothing.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: A5's round-trip cost: 0.28% buy + 0.18% sell + 0.1% sell tax.
ROUND_TRIP_BPS = 56.0


def margin_bps(pnl: Sequence[float], gross: Sequence[float]) -> float:
    """Value-weighted rupiah earned per 10,000 rupiah traded.

    Value-weighted rather than a mean of per-window margins: a class's margin
    is what its whole book earned per rupiah it put through, not the average of
    its individual fortnights, which would let a tiny window count as much as
    the largest one.
    """
    p = np.asarray(pnl, dtype=float)
    g = np.asarray(gross, dtype=float)
    m = np.isfinite(p) & np.isfinite(g)
    tot = g[m].sum()
    if tot <= 0:
        return np.nan
    return float(10000.0 * p[m].sum() / tot)


def class_margin(df: pd.DataFrame, view: str, by: Optional[str] = None
                 ) -> pd.Series:
    """Margin for one class, pooled or split by a column."""
    d = df[df["view"] == view]
    if d.empty:
        return pd.Series(dtype=float)
    if by is None:
        return pd.Series({"all": margin_bps(d["timing_pnl"], d["gross_value"])})
    return d.groupby(by).apply(
        lambda g: margin_bps(g["timing_pnl"], g["gross_value"]))


def shuffle_forward(df: pd.DataFrame, rng, group: str = "window_end",
                    col: str = "fwd_ret") -> pd.DataFrame:
    """Permute forward returns across tickers WITHIN each window.

    This is the null the question needs. It preserves every window's flow
    distribution and every window's cross-section of returns exactly, and
    destroys only which ticker's flow was paired with which ticker's
    subsequent move — which is the entire claim. A null that shuffled returns
    across windows instead would also destroy the market's own time structure
    and would be far too easy to beat.
    """
    out = df.copy()
    out[col] = out.groupby(group)[col].transform(
        lambda s: rng.permutation(s.to_numpy()))
    out["timing_pnl"] = out["net_value"] * out[col]
    return out


def permutation_margin(df: pd.DataFrame, view: str, draws: int = 200,
                       seed: int = 0) -> Tuple[float, np.ndarray, float]:
    """Observed class margin, its null distribution, and a two-sided p.

    Two-sided here, unlike H11's one-sided test. §12 predicts a direction for
    RETAIL (it loses), but foreign-versus-domestic is not the same split as
    institution-versus-retail — domestic institutions exist and foreign retail
    exists — so there is no honest prior about which way this one should point.
    Claiming one after seeing the data is exactly the move §2 forbids.
    """
    d = df[df["view"] == view]
    obs = margin_bps(d["timing_pnl"], d["gross_value"])
    rng = np.random.default_rng(seed)
    nulls = np.full(draws, np.nan)
    for i in range(draws):
        s = shuffle_forward(d, rng)
        nulls[i] = margin_bps(s["timing_pnl"], s["gross_value"])
    v = nulls[np.isfinite(nulls)]
    if not len(v) or not np.isfinite(obs):
        return obs, nulls, np.nan
    p = (float((np.abs(v - v.mean()) >= abs(obs - v.mean())).sum() + 1)
         / (len(v) + 1))
    return obs, nulls, p


def sign_persistence(annual: pd.Series) -> Dict[str, float]:
    """Does the sign hold year to year, or is a pooled figure carried by a few?

    Reports the fraction of years sharing the pooled sign, the lag-1
    autocorrelation of the annual margin, and the pooled margin recomputed with
    the single largest-magnitude year removed. That last one is the check H11
    needed and did not have until a 13-broker year was found carrying its
    headline.
    """
    a = annual.dropna()
    if len(a) < 3:
        return {"n_years": float(len(a))}
    s = np.sign(a.sum())
    out = {
        "n_years": float(len(a)),
        "share_same_sign": float((np.sign(a) == s).mean()),
        "mean": float(a.mean()),
        "median": float(a.median()),
        "sd": float(a.std()),
    }
    if len(a) >= 4:
        x = a.to_numpy()
        out["lag1_autocorr"] = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    k = int(np.argmax(np.abs(a.to_numpy())))
    out["mean_drop_largest"] = float(a.drop(a.index[k]).mean())
    out["largest_year"] = float(a.index[k])
    return out


def mirror_residual(df: pd.DataFrame) -> pd.Series:
    """|F_net + D_net| / (F_gross + D_gross) per ticker-window.

    Structurally this is zero: every rupiah bought is a rupiah sold, so the two
    classes' nets are exact mirrors. Anything above zero is the top-ten
    censoring, which makes this the one honest, data-driven error bar on every
    net in this module — no assumption about coverage required.
    """
    w = df.pivot_table(index=["ticker", "window_end"], columns="view",
                       values=["net_value", "gross_value"], aggfunc="sum")
    if ("net_value", "F") not in w or ("net_value", "D") not in w:
        return pd.Series(dtype=float)
    num = (w[("net_value", "F")] + w[("net_value", "D")]).abs()
    den = (w[("gross_value", "F")].abs() + w[("gross_value", "D")].abs())
    return (num / den.replace(0, np.nan)).dropna()
