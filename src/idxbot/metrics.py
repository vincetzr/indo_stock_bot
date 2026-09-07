"""The §39 metric set, and the deflated Sharpe ratio §11 has demanded since day one.

WHY THIS MODULE EXISTS, AND IT IS NOT THE SIX MISSING RATIOS.
`docs/PLATFORM_ASSESSMENT.md` lists Sharpe, Sortino, profit factor, expectancy,
exposure and tail risk as uncomputed. That is half wrong — `portfolio._stats`
has computed Sharpe and Sortino all along — and the half that is right is the
less important half. The real gap is that **CLAUDE.md §11 names the deflated
Sharpe ratio as non-negotiable, `hypotheses.md` line 5 says the trial count
exists so that it CAN be computed, and across fifty-seven hypotheses it never
once was.** A running trial count that never enters a statistic is bookkeeping,
not a correction.

WHAT A DEFLATED SHARPE ACTUALLY DOES HERE. This repo has run **350 trials**
against one Indonesian price history. The largest Sharpe among 350 draws from a
zero-edge process is not near zero — it is around 2.9 standard errors up. The
deflated Sharpe asks the only question that matters after a search: *is this
Sharpe larger than the largest one the search would have produced from
nothing?* Every headline in this repo has been read against a permutation null,
which handles the label but not the SEARCH. The two corrections are different
and this repo has only ever applied one of them.

THREE THINGS THAT MAKE THE RATIOS HERE DIFFERENT FROM THE TEXTBOOK ONES.

  1. **Overlapping windows.** Almost every statistic in this project is built
     from k-session forward returns sampled every session, so the nominal n is
     ~k times the number of independent observations. `effective_n` deflates
     it and every function that consumes n takes the deflated value. A20 and
     A23 both record effective n being an order of magnitude below the row
     count; the Sharpe is where that does the most damage, because it enters
     as sqrt(n).

  2. **Skew and kurtosis are not decoration.** A13 measured kurtosis from 10 to
     **2,800** on the series this repo trades. The probabilistic Sharpe ratio
     is the version that takes both, and on a sample like that it is a
     different number from the naive one, not a refinement of it.

  3. **A ratio is not a benchmark.** A19 and A31 both record the missing
     comparison as the error class that manufactures results. `summary()`
     therefore refuses to be read alone: it carries `mean_log` beside `mean`
     (A36 measured the two disagreeing in SIGN), and `describe()` prints the
     power statement — how many periods it would take to distinguish this from
     zero — before it prints any ratio.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

EULER = 0.5772156649015329


# ----------------------------------------------------------------- utilities

def _clean(r: Sequence[float]) -> np.ndarray:
    a = np.asarray(r, dtype=float).ravel()
    return a[np.isfinite(a)]


def effective_n(n: int, overlap: int = 1) -> float:
    """Independent observations behind `n` rows of `overlap`-overlapping windows.

    A k-session forward return sampled every session shares k-1 of its k days
    with its neighbour, so n rows carry roughly n/k independent observations.
    This is the crude ratio deliberately: the exact factor depends on the
    autocorrelation of the underlying, and `lo_factor` is the version that
    estimates it from the data. Overstating independence inflates every
    t-statistic in the file, so the crude version is the conservative one.
    """
    if overlap <= 1:
        return float(max(n, 0))
    return float(max(n, 0)) / float(overlap)


def lo_factor(r: Sequence[float], q: int) -> float:
    """Lo (2002) scaling from a per-period Sharpe to a q-period one.

    The naive scaling is sqrt(q), which assumes independence. With positive
    autocorrelation the true factor is smaller, so sqrt(q) overstates the
    Sharpe — the same direction as every other error in this family.
    """
    a = _clean(r)
    q = int(max(q, 1))
    if q == 1 or len(a) < q + 2:
        return math.sqrt(q)
    acc = float(q)
    for k in range(1, q):
        x, y = a[:-k], a[k:]
        if len(x) < 3:
            break
        sx, sy = x.std(ddof=1), y.std(ddof=1)
        if sx <= 0 or sy <= 0:
            continue
        rho = float(np.corrcoef(x, y)[0, 1])
        if not np.isfinite(rho):
            continue
        acc += 2.0 * (q - k) * rho
    if acc <= 0:
        return float("nan")
    return q / math.sqrt(acc)


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _nppf(p: float) -> float:
    """Inverse normal CDF. Acklam's rational approximation, |err| < 1.15e-9."""
    if not 0.0 < p < 1.0:
        return float("-inf") if p <= 0.0 else float("inf")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ------------------------------------------------------------------- Sharpe

def sharpe(r: Sequence[float], periods_per_year: float = 252.0,
           rf: float = 0.0, overlap: int = 1) -> float:
    """Annualised Sharpe. `overlap` uses Lo's factor instead of sqrt(q)."""
    a = _clean(r) - rf
    if len(a) < 3:
        return float("nan")
    sd = a.std(ddof=1)
    if sd <= 0:
        return float("nan")
    per = a.mean() / sd
    if overlap > 1:
        f = lo_factor(a, int(overlap))
        if not np.isfinite(f) or f <= 0:
            return float("nan")
        #  scale to one independent period, then annualise on those
        per = per * f
        return float(per * math.sqrt(periods_per_year / max(overlap, 1)))
    return float(per * math.sqrt(periods_per_year))


def sortino(r: Sequence[float], periods_per_year: float = 252.0,
            mar: float = 0.0) -> float:
    """Sortino against a MINIMUM ACCEPTABLE RETURN, stated rather than assumed.

    The common implementation takes the standard deviation of the negative
    returns, which is a different quantity: it drops the zeros and the upside
    from the denominator's COUNT as well as its sum, inflating the ratio. The
    downside deviation below sums squared shortfalls over ALL periods.
    """
    a = _clean(r)
    if len(a) < 3:
        return float("nan")
    short = np.minimum(a - mar, 0.0)
    dd = math.sqrt(float((short ** 2).mean()))
    if dd <= 0:
        return float("nan")
    return float((a.mean() - mar) / dd * math.sqrt(periods_per_year))


def psr(r: Sequence[float], benchmark_sr: float = 0.0,
        periods_per_year: float = 252.0, overlap: int = 1) -> float:
    """Probabilistic Sharpe ratio: P(true SR > `benchmark_sr`).

    Bailey & Lopez de Prado. Takes skew and kurtosis, which matters here rather
    than in general: A13 measured kurtosis from 10 to 2,800 on the series this
    repo trades, and a fat right tail makes a high Sharpe LESS trustworthy, not
    more.

    `benchmark_sr` is annualised, same as the observed one.
    """
    a = _clean(r)
    n = effective_n(len(a), overlap)
    if n < 4:
        return float("nan")
    sd = a.std(ddof=1)
    if sd <= 0:
        return float("nan")
    sr = a.mean() / sd                              # per period
    ann = math.sqrt(periods_per_year / max(overlap, 1))
    sr_b = benchmark_sr / ann if ann > 0 else float("nan")
    z = (a - a.mean()) / sd
    g3 = float((z ** 3).mean())
    g4 = float((z ** 4).mean())                     # NOT excess
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        return float("nan")
    return float(_ncdf((sr - sr_b) * math.sqrt(n - 1.0) / math.sqrt(denom)))


def expected_max_sharpe(trials: int, sr_variance: float,
                        periods_per_year: float = 252.0) -> float:
    """The annualised Sharpe the SEARCH alone would be expected to produce.

    `sr_variance` is the variance of the annualised Sharpe ratios ACROSS the
    trials. This repo does not store one, so it must be supplied and the
    supplier has to say where it came from. A default would be the worst
    possible thing here: it would make the deflation look measured when it was
    assumed.
    """
    n = int(trials)
    if n < 2 or not np.isfinite(sr_variance) or sr_variance <= 0:
        return float("nan")
    a = _nppf(1.0 - 1.0 / n)
    b = _nppf(1.0 - 1.0 / (n * math.e))
    return float(math.sqrt(sr_variance) * ((1.0 - EULER) * a + EULER * b))


def deflated_sharpe(r: Sequence[float], trials: int, sr_variance: float,
                    periods_per_year: float = 252.0,
                    overlap: int = 1) -> Dict[str, float]:
    """P(true SR > 0) AFTER charging for having searched `trials` times.

    THIS IS A DIFFERENT CORRECTION FROM THE PERMUTATION NULL, AND THIS REPO HAS
    ONLY EVER APPLIED THE OTHER ONE. A clustered permutation null asks whether
    the LABEL carries information. It cannot ask whether the strategy is the
    best of 350 attempts, because it only ever sees the one attempt it is
    handed. The deflated Sharpe is the correction for the search.

    Returns the observed Sharpe, the benchmark the search alone would clear,
    the deflated probability, and — because a probability near one is exactly
    the number most likely to be over-read — the count of trials at which the
    deflation would take it below 0.95.
    """
    obs = sharpe(r, periods_per_year, overlap=overlap)
    sr_star = expected_max_sharpe(trials, sr_variance, periods_per_year)
    out = {"sharpe": obs, "trials": float(trials),
           "sr_variance": float(sr_variance), "sr_star": sr_star,
           "dsr": float("nan"), "trials_to_kill": float("nan")}
    if not np.isfinite(sr_star):
        return out
    out["dsr"] = psr(r, sr_star, periods_per_year, overlap=overlap)
    if np.isfinite(out["dsr"]) and out["dsr"] >= 0.95:
        t = int(trials)
        for _ in range(60):
            t *= 2
            s = expected_max_sharpe(t, sr_variance, periods_per_year)
            if not np.isfinite(s):
                break
            if psr(r, s, periods_per_year, overlap=overlap) < 0.95:
                out["trials_to_kill"] = float(t)
                break
    return out


# ------------------------------------------------------------ trade metrics

def profit_factor(trade_returns: Sequence[float]) -> float:
    """Gross wins over gross losses. Undefined with no losers, not infinite."""
    a = _clean(trade_returns)
    win = float(a[a > 0].sum())
    loss = float(-a[a < 0].sum())
    if loss <= 0:
        return float("nan")
    return win / loss


def expectancy(trade_returns: Sequence[float]) -> Dict[str, float]:
    """Per-trade expectation, with the mean LOG beside the mean.

    A36 measured a +5.77% average trade compounding at -5.3% a year, so a mean
    quoted alone is not a statement about an account. Which one applies is a
    question about the BOOK: an equal-weighted holder of many names at once is
    paid the mean (A18), a bot running positions one at a time is paid the mean
    log (A36). Both are returned and the caller has to choose.
    """
    a = _clean(trade_returns)
    if len(a) == 0:
        return {}
    win = a[a > 0]
    loss = a[a < 0]
    p = float(len(win)) / len(a)
    return {
        "trades": float(len(a)),
        "win_rate": p,
        "avg_win": float(win.mean()) if len(win) else float("nan"),
        "avg_loss": float(loss.mean()) if len(loss) else float("nan"),
        "expectancy": float(a.mean()),
        "median": float(np.median(a)),
        "mean_log": float(np.log1p(np.clip(a, -0.999, None)).mean()),
        "profit_factor": profit_factor(a),
    }


def exposure(in_market: Sequence[bool]) -> float:
    """Share of periods holding anything. A29 measured a rule at 33.5%.

    It belongs beside every return statistic because a rule that is out of the
    market two thirds of the time is not comparable to one that is not, and
    A29's whole finding was that the exposure gap explained the result.
    """
    a = np.asarray(in_market)
    if a.size == 0:
        return float("nan")
    return float(np.mean(a.astype(bool)))


def tail(r: Sequence[float], alpha: float = 0.05) -> Dict[str, float]:
    """Left-tail risk: VaR, conditional VaR, worst period, and the Ulcer index.

    The Ulcer index is the RMS of the drawdown path, which is the one measure
    here that charges for how LONG a drawdown lasts. A18 measured a per-position
    stop producing the deepest PORTFOLIO drawdown in its table, so depth alone
    has already misled this repo once.
    """
    a = _clean(r)
    if len(a) < 3:
        return {}
    var = float(np.quantile(a, alpha))
    below = a[a <= var]
    #  THE PATH STARTS AT THE INITIAL CAPITAL, NOT AT THE FIRST BAR'S CLOSE.
    #  Without the leading 1.0 the running maximum starts BELOW par whenever
    #  the first period is a loss, so a series that opens -30% reports a
    #  maximum drawdown of exactly zero. An impossible number is the cheapest
    #  bug detector available (A30) and this one was caught by the test that
    #  asserts a shallow-but-long drawdown scores worse on Ulcer than a deep
    #  one -- it failed on the DEPTH clause, not the length clause.
    eq = np.concatenate([[1.0], np.cumprod(1.0 + a)])
    dd = eq / np.maximum.accumulate(eq) - 1.0
    return {
        "var": var,
        "cvar": float(below.mean()) if len(below) else float("nan"),
        "worst_period": float(a.min()),
        "max_drawdown": float(dd.min()),
        "ulcer": float(math.sqrt(float((dd ** 2).mean()))),
        "skew": float(((a - a.mean()) ** 3).mean() / a.std(ddof=0) ** 3)
        if a.std(ddof=0) > 0 else float("nan"),
        "kurtosis": float(((a - a.mean()) ** 4).mean() / a.std(ddof=0) ** 4)
        if a.std(ddof=0) > 0 else float("nan"),
    }


def periods_to_detect(r: Sequence[float], t: float = 2.0,
                      against: float = 0.0) -> float:
    """How many periods to tell this mean from `against` at |t| = `t`.

    A31 computed this once by hand and it was the most useful number in that
    study: 104 months to distinguish the account from zero, 46,856 from random
    picking. It is stated SEPARATELY from the effect, because A19 records
    writing a power statement as an effect statement as its own error.
    """
    a = _clean(r)
    if len(a) < 3:
        return float("nan")
    sd = a.std(ddof=1)
    d = a.mean() - against
    if sd <= 0 or d == 0:
        return float("nan")
    return float((t * sd / abs(d)) ** 2)


# ---------------------------------------------------------------- the bundle

def summary(r: Sequence[float], periods_per_year: float = 252.0,
            overlap: int = 1, trade_returns: Optional[Sequence[float]] = None,
            in_market: Optional[Sequence[bool]] = None,
            trials: Optional[int] = None,
            sr_variance: Optional[float] = None) -> Dict[str, float]:
    """The whole §39 set from one per-period return series.

    `trials` and `sr_variance` are BOTH required for a deflated Sharpe, and
    supplying neither leaves the field absent rather than defaulted. A deflation
    against an assumed variance would read as a measurement.
    """
    a = _clean(r)
    out: Dict[str, float] = {"periods": float(len(a))}
    if len(a) < 3:
        return out
    eq = float(np.prod(1.0 + a))
    yrs = len(a) / periods_per_year
    out.update({
        "effective_n": effective_n(len(a), overlap),
        "total_return": eq - 1.0,
        "cagr": float(max(eq, 1e-9) ** (1.0 / max(yrs, 1e-9)) - 1.0),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "mean_log": float(np.log1p(np.clip(a, -0.999, None)).mean()),
        "volatility": float(a.std(ddof=1) * math.sqrt(periods_per_year)),
        "sharpe": sharpe(a, periods_per_year, overlap=overlap),
        "sortino": sortino(a, periods_per_year),
        "psr": psr(a, 0.0, periods_per_year, overlap=overlap),
        "hit_rate": float((a > 0).mean()),
        "periods_to_detect": periods_to_detect(a),
    })
    out.update(tail(a))
    if in_market is not None:
        out["exposure"] = exposure(in_market)
    if trade_returns is not None:
        out.update({f"trade_{k}": v for k, v in
                    expectancy(trade_returns).items()})
    if trials is not None and sr_variance is not None:
        d = deflated_sharpe(a, trials, sr_variance, periods_per_year,
                            overlap=overlap)
        out.update({"sr_star": d["sr_star"], "dsr": d["dsr"],
                    "trials_to_kill": d["trials_to_kill"]})
    return out


def describe(m: Dict[str, float]) -> str:
    """Human-readable, POWER FIRST.

    The order is deliberate. A31 records the power statement being the most
    useful line in that study, and A19 records it being conflated with the
    effect. Printing it above the ratios makes the reader meet "you could not
    tell this from zero in under N periods" before they meet a Sharpe.
    """
    if not m or "mean" not in m:
        return "insufficient data"
    L = []
    n = m.get("periods_to_detect", float("nan"))
    L.append(f"POWER      {n:,.0f} periods to tell this mean from zero at |t|=2"
             if np.isfinite(n) else "POWER      undefined (mean is zero)")
    L.append(f"           {m['periods']:,.0f} observed, "
             f"{m.get('effective_n', float('nan')):,.0f} independent")
    L.append(f"RETURN     CAGR {m['cagr']:+.2%}   mean {m['mean']:+.4f}   "
             f"median {m['median']:+.4f}   mean log {m['mean_log']:+.4f}")
    L.append(f"RISK       vol {m['volatility']:.2%}   maxDD "
             f"{m.get('max_drawdown', float('nan')):.2%}   ulcer "
             f"{m.get('ulcer', float('nan')):.2%}   "
             f"CVaR{5} {m.get('cvar', float('nan')):+.2%}")
    L.append(f"SHAPE      skew {m.get('skew', float('nan')):+.2f}   "
             f"kurtosis {m.get('kurtosis', float('nan')):.1f}   "
             f"hit {m['hit_rate']:.1%}")
    L.append(f"RATIO      Sharpe {m['sharpe']:+.2f}   "
             f"Sortino {m['sortino']:+.2f}   PSR {m['psr']:.3f}")
    if "exposure" in m:
        L.append(f"EXPOSURE   {m['exposure']:.1%} of periods in the market")
    if "trade_trades" in m:
        L.append(f"TRADES     {m['trade_trades']:,.0f}   win "
                 f"{m['trade_win_rate']:.1%}   expectancy "
                 f"{m['trade_expectancy']:+.2%}   mean log "
                 f"{m['trade_mean_log']:+.4f}   PF "
                 f"{m['trade_profit_factor']:.2f}")
    if "dsr" in m and np.isfinite(m["dsr"]):
        L.append(f"DEFLATED   {m['trials_to_kill']:,.0f} trials would kill it"
                 if np.isfinite(m.get("trials_to_kill", float("nan")))
                 else "DEFLATED   already below 0.95")
        L.append(f"           search alone clears Sharpe {m['sr_star']:+.2f}; "
                 f"DSR {m['dsr']:.3f}")
    return "\n".join(L)
