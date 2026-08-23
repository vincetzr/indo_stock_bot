"""§12 — does a broker's margin rank carry over to the next period?

WHY THIS MODULE EXISTS SEPARATELY FROM THE SCRIPT
---------------------------------------------------
The statistic is a rank correlation between per-broker margins in adjacent
periods. Computing it once is easy; computing it two hundred times under a
label shuffle is not, and the shuffle is the whole point — H9 established in
this repo that a null run once is a decoration and a null run properly is the
only thing that caught two broken estimators.

So the statistic is written once, vectorised, and both the observed value and
every permutation draw go through the identical path. A readable pandas
reference lives in the tests and is asserted to agree with it.

THE GUARDS ARE PART OF THE STATISTIC
--------------------------------------
A broker is ranked in a period only if it traded in enough distinct windows and
enough gross value there. Those guards are applied INSIDE the permuted pipeline
too, not just to the observed data. If they were applied only once, the null
would be answering a different question — "what if labels were shuffled among a
set chosen using the real labels" — and would be biased toward whatever the real
selection did.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

#: Fewest brokers present in BOTH periods before a rank correlation is quoted.
#: Below this the correlation is a function of two or three points and swings
#: between +1 and -1 on noise alone.
MIN_BROKERS = 6


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not tilt the correlation."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    # average tied ranks
    s = x[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, NaN when it is not defined."""
    if len(a) < MIN_BROKERS:
        return np.nan
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def margin_matrix(bcode: np.ndarray, pcode: np.ndarray, wcode: np.ndarray,
                  pnl: np.ndarray, gross: np.ndarray,
                  n_brokers: int, n_periods: int, n_windows: int,
                  min_windows: int, min_gross: float) -> np.ndarray:
    """(broker x period) value-weighted margin_bps, NaN where guarded out.

    Value-weighted rather than an average of per-window margins, matching
    §9.3's ``margin_bps = 10000 * pnl / gross_traded_value``: a broker's margin
    is what its whole book earned per rupiah it put through, not the mean of
    its individual fortnights.
    """
    key = bcode.astype(np.int64) * n_periods + pcode
    size = n_brokers * n_periods
    s_pnl = np.bincount(key, weights=pnl, minlength=size)
    s_gross = np.bincount(key, weights=gross, minlength=size)

    # distinct windows per (broker, period) — a broker can appear in the same
    # window for several tickers, so a row count would overstate it
    uniq = np.unique(key * n_windows + wcode.astype(np.int64))
    n_win = np.bincount((uniq // n_windows).astype(np.int64), minlength=size)

    ok = (n_win >= min_windows) & (s_gross >= min_gross) & (s_gross > 0)
    bps = np.full(size, np.nan)
    np.divide(10000.0 * s_pnl, s_gross, out=bps, where=ok)
    bps[~ok] = np.nan
    return bps.reshape(n_brokers, n_periods)


def adjacent_corr(M: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean rank correlation over adjacent period pairs, and the pairs.

    With two periods this is the split-half statistic. With twelve it is a
    year-over-year autocorrelation, which is the same claim measured eleven
    more times.
    """
    out = []
    for j in range(M.shape[1] - 1):
        a, b = M[:, j], M[:, j + 1]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < MIN_BROKERS:
            continue
        r = spearman(a[m], b[m])
        if np.isfinite(r):
            out.append(r)
    arr = np.asarray(out, dtype=float)
    return (float(arr.mean()) if len(arr) else np.nan, arr)


def shuffle_within(group: np.ndarray, values: np.ndarray, rng) -> np.ndarray:
    """Permute ``values`` within each group, vectorised.

    Preserves each group's multiset of values exactly and destroys only which
    row held which — for a ticker-window group that means the window's total
    flow and its size distribution survive untouched while broker identity is
    scrambled, which is the null §12 needs.
    """
    key = rng.random(len(values))
    order = np.lexsort((key, group))
    out = np.empty_like(values)
    out[np.argsort(group, kind="mergesort")] = values[order]
    return out


def permutation_test(group: np.ndarray, bcode: np.ndarray, pcode: np.ndarray,
                     wcode: np.ndarray, pnl: np.ndarray, gross: np.ndarray,
                     n_brokers: int, n_periods: int, n_windows: int,
                     min_windows: int, min_gross: float,
                     draws: int = 200, seed: int = 0
                     ) -> Tuple[float, np.ndarray, np.ndarray, Optional[float]]:
    """Observed adjacent-period rank correlation and its null distribution.

    Returns (observed mean, observed per-pair correlations, null means,
    one-sided empirical p). The p-value is one-sided upward because §12
    predicts a specific direction: persistence means POSITIVE rank correlation.
    A significantly negative one would be a different and stranger finding, and
    is reported as such rather than folded into a two-sided number.
    """
    M = margin_matrix(bcode, pcode, wcode, pnl, gross,
                      n_brokers, n_periods, n_windows, min_windows, min_gross)
    obs, pairs = adjacent_corr(M)

    nulls = np.full(draws, np.nan)
    rng = np.random.default_rng(seed)
    for i in range(draws):
        sb = shuffle_within(group, bcode, rng)
        Mi = margin_matrix(sb, pcode, wcode, pnl, gross,
                           n_brokers, n_periods, n_windows,
                           min_windows, min_gross)
        nulls[i], _ = adjacent_corr(Mi)

    v = nulls[np.isfinite(nulls)]
    p = (float((v >= obs).sum() + 1) / (len(v) + 1)
         if len(v) and np.isfinite(obs) else None)
    return obs, pairs, nulls, p
