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


#: The two nulls answer DIFFERENT questions and a single one is misleading.
#:
#: This was caught by running the pipeline on partial data rather than waiting
#: for the full collection: the within-window null came back at +25.1 bps
#: against an observed +23.85, sitting exactly on top of the signal. That is
#: not a broken null — it is the correct answer to the question it asks.
#:
#:   SELECTION ("within_window"): permute forward returns across tickers inside
#:       each window. Preserves each window's flow and its whole cross-section
#:       of returns; destroys only WHICH ticker's flow met which ticker's move.
#:       A class that was merely net long into a rising market scores the same
#:       under this shuffle as it did in reality, because the market move is
#:       common to every ticker in the window and survives the permutation.
#:       So this null asks: did the class pick the right STOCKS?
#:
#:   DIRECTION ("block_window"): permute WHOLE WINDOWS — each window's flow is
#:       paired with a different window's entire cross-section of returns, the
#:       ticker identities kept intact. Destroys WHEN the class was long, which
#:       is what §12's question actually needs: persistently dumb flow is flow
#:       positioned wrongly, not flow that picks the wrong names.
#:
#: WHY THE DIRECTION NULL IS A BLOCK PERMUTATION AND NOT A PER-TICKER SHUFFLE.
#: The obvious version permutes each ticker's returns across windows
#: independently. It destroys timing correctly, but it also makes the tickers
#: independent of each other, which understates the aggregate margin's
#: volatility — the null then comes back too narrow and the test over-rejects.
#: Permuting whole windows as blocks keeps each window's co-movement, and
#: therefore the true spread, while still breaking the link between flow and
#: the move that followed it.
#:
#: THE CONDITION, because the first version of this note stated it too broadly:
#: the gap only opens when the FLOW is correlated across names within a window
#: as well as the returns. With independent per-ticker flow the two nulls are
#: indistinguishable — measured at sd 7.7 against 7.4 on a synthetic panel with
#: strongly co-moving returns but random flow signs. Add a common per-window
#: flow tilt, which is the realistic case since a risk-on class is net long
#: across the board rather than long one name and short the next, and the block
#: null is materially wider. Both cases are in the tests, and whether real IDX
#: flow meets the condition is measured rather than assumed.
#:
#: The block permutation is the conservative choice either way: where the
#: condition does not hold it costs nothing, and where it does it prevents a
#: false positive.
#:
#: Reporting only the selection null would have made a real directional result
#: look like nothing; reporting only the direction null would let a pure
#: market-beta effect read as skill. Both are reported, always.
NULLS = ("within_window", "block_window")


def shuffle_forward(df: pd.DataFrame, rng, kind: str = "within_window",
                    col: str = "fwd_ret") -> pd.DataFrame:
    """Permute forward returns, preserving one margin and destroying the other.

    See :data:`NULLS` for why the choice of permutation IS the choice of
    question. Rows whose partner window does not carry that ticker come back
    NaN and drop out of the margin, which loses a little sample and biases
    nothing.
    """
    if kind not in NULLS:
        raise ValueError(f"kind must be one of {NULLS}, got {kind!r}")
    out = df.copy()
    if kind == "within_window":
        out[col] = out.groupby("window_end")[col].transform(
            lambda s: rng.permutation(s.to_numpy()))
    else:
        wins = np.asarray(sorted(df["window_end"].unique()))
        mapped = dict(zip(wins, rng.permutation(wins)))
        lookup = df.set_index(["ticker", "window_end"])[col]
        lookup = lookup[~lookup.index.duplicated()]
        out[col] = [lookup.get((t, mapped[w]), np.nan)
                    for t, w in zip(df["ticker"], df["window_end"])]
    out["timing_pnl"] = out["net_value"] * out[col]
    return out


def permutation_margin(df: pd.DataFrame, view: str,
                       kind: str = "within_window", draws: int = 200,
                       seed: int = 0) -> Tuple[float, np.ndarray, float]:
    """Observed class margin, its null distribution, and a two-sided p.

    Two-sided, unlike H11's one-sided test. §12 predicts a direction for
    RETAIL, and foreign-versus-domestic is only an imperfect proxy for that
    split — domestic institutions are large in IDX and are pooled into
    "domestic". The prediction is still pre-registered in `hypotheses.md`; the
    p-value simply does not claim the extra power a one-sided test would take.
    """
    d = df[df["view"] == view]
    obs = margin_bps(d["timing_pnl"], d["gross_value"])
    rng = np.random.default_rng(seed)
    nulls = np.full(draws, np.nan)
    for i in range(draws):
        s = shuffle_forward(d, rng, kind=kind)
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
