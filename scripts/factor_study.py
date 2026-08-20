#!/usr/bin/env python3
"""What actually pays on IDX: every price-derived factor, tested the hard way.

WHY START HERE
--------------
A desk that is handed client money does not begin with "how do I guarantee a
profit". It begins with two questions that have answers: what has a documented
positive expected return, and what reliably destroys one. The destroyer is
already measured here and is not in doubt - 0.56% a round trip, certain, paid
whether or not the trade works. The premium side is the part that has to be
earned, and so far every timing rule tried in this repo has lost to simply
holding (110, 111, 113, 115, 116).

Timing is one way to try to earn a premium. CROSS-SECTION is the other, and it
has never been tested here properly. The claim is not "buy when the chart turns"
but "of the names available today, some group has a higher expected return than
the rest, for a reason that persists". That is the question this script settles
for IDX, on every factor computable from price and volume alone.

WHY PRICE-DERIVED ONLY
----------------------
data/cache/fundamentals holds 59 names, and holds them as a SNAPSHOT of today -
one P/E, one ROE, no history. Ranking 2015 on a P/E published in 2026 is
look-ahead of the worst kind: it knows which companies turned out to be
profitable. Those fields therefore cannot be backtested here at all, and this
script does not touch them. Price and volume, by contrast, are point-in-time by
construction - the close on a given day was the close on that day.

WHAT IS TESTED
--------------
    mom12_1   12-month return skipping the last month   (momentum)
    mom6_1    6-month return skipping the last month
    rev1      minus the last month's return             (short-term reversal)
    lowvol    minus trailing 250d volatility            (low-volatility anomaly)
    lowbeta   minus beta to the IHSG
    lowidio   minus residual volatility to the IHSG
    lowmax    minus the mean of the 5 best days in a month  (lottery / MAX)
    trend     price over its 200-day average
    high52    price over its 52-week high
    illiq     Amihud |return| per rupiah traded         (illiquidity premium)
    small     minus median turnover                     (size proxy)
    large     plus median turnover

THREE INDEPENDENT READINGS, BECAUSE ONE CAN LIE
-----------------------------------------------
    1. RANK IC - the cross-sectional Spearman correlation between the score
       today and the return over the next month, averaged over every month.
       Portfolio-free: it cannot be flattered by a construction choice.
    2. DECILE SPREAD - top decile minus bottom decile. Diagnostic only; this
       account cannot short, so a factor that only works on the short side is
       information the account cannot spend.
    3. LONG-ONLY PORTFOLIO - the thing that could actually be held, after costs,
       scored against the equal-weight universe (the neutral portfolio you would
       hold with no view at all) and against the IHSG.

Significance is judged after Bonferroni across every factor tested, with
Newey-West standard errors on overlapping monthly series, and a stationary block
bootstrap that resamples whole blocks of months so a lucky run cannot pass as
skill.

THE BIAS THAT CANNOT BE FIXED, ONLY DECLARED
--------------------------------------------
Every one of the 843 cached names is still trading today. Zero delisted names
are present, so the universe is a SURVIVOR universe and every long-only level in
here is too high. This cuts one way that matters: any strategy that already
fails to beat the IHSG inside a survivor universe fails by strictly more in the
real one, because the IHSG is a real published index and carries its own
casualties. Losing results are therefore safe to trust; winning ones are not.

    python3 scripts/factor_study.py
    python3 scripts/factor_study.py --breadth 20 --rebalance quarterly
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from idxbot.config import load_config           # noqa: E402
from idxbot.data.cache import Cache             # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV        # noqa: E402
from account_sim import load_ohlc               # noqa: E402

FEE = 0.0056                     # 0.28% a side; a round trip is 0.56%
ONE_WAY = FEE / 2.0
TRADING_DAYS = 250
REBALANCE = {"monthly": "ME", "quarterly": "QE", "annual": "YE"}
DRAWS = 25                       # random books averaged in the breadth sweep


# --------------------------------------------------------------------------
# panel
# --------------------------------------------------------------------------
def total_return_series(loader: YahooOHLCV, ticker: str,
                        total: bool = True) -> Optional[pd.DataFrame]:
    """One name's price series, dividends included by default.

    THE OMISSION THIS EXISTS TO FIX. Every earlier study in this repo ran on the
    raw close, which is what a chart shows and is NOT what a holder earns: it
    excludes every dividend paid. Across 281 cached names the gap averages
    +1.75%/yr, and on the payers that dominate a liquid book it is far larger -
    ADRO +15.2%/yr, BSSR +15.5%, HEXA +16.0%. Measuring an equity strategy on
    the price line understates the holder by more than any signal in this repo
    has ever added, so it is not a rounding detail, it is the main term.

    Returns raw close as well, because turnover must be computed on the price
    actually traded and volume actually printed, not on an adjusted level.
    """
    d = loader.get(ticker, max_age=86400 * 30)
    if d is None or len(d) < 300:
        return None
    d = d.set_index("date").sort_index()
    px = d["adj_close"] if (total and "adj_close" in d) else d["close"]
    px = pd.to_numeric(px, errors="coerce")
    raw = pd.to_numeric(d["close"], errors="coerce")
    vol = pd.to_numeric(d["volume"], errors="coerce")
    out = pd.DataFrame({"px": px, "raw": raw, "volume": vol}).dropna()
    if out.empty or len(out) < 300:
        return None
    # Cap impossible prints: IDX auto-rejection makes a one-day move beyond 35%
    # impossible on the regular board, so anything larger is a bad tick.
    r = out["px"].pct_change().clip(-0.35, 0.35).fillna(0.0)
    out["px"] = float(out["px"].iloc[0]) * (1.0 + r).cumprod()
    return out


def build_panel(loader: YahooOHLCV, cache_dir: str, start: str,
                min_price: float = 50.0, min_bars: int = 400,
                total: bool = True
                ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Total-return, raw-price and rupiah-turnover panels for every IDX equity.

    The raw close is kept alongside the adjusted one because the RATIO between
    them is the accumulated dividend, and that ratio is what makes trailing
    yield a point-in-time factor rather than a fundamental nobody has history
    for.
    """
    names = sorted({os.path.basename(f).split(".")[0].upper()
                    for f in glob.glob(os.path.join(cache_dir, "ohlcv",
                                                    "*.JK.csv.gz"))})
    names = [n for n in names if n.isalpha() and len(n) == 4]
    px, rw, tv = {}, {}, {}
    for t in names:
        df = total_return_series(loader, t, total)
        if df is None or len(df) < min_bars:
            continue
        df = df[df.index >= pd.Timestamp(start)]
        if len(df) < min_bars or float(df["raw"].median()) < min_price:
            continue
        px[t] = df["px"]
        rw[t] = df["raw"]
        tv[t] = df["raw"] * df["volume"]
    close = pd.DataFrame(px).sort_index()
    raw = pd.DataFrame(rw).sort_index().reindex(columns=close.columns)
    turn = pd.DataFrame(tv).sort_index().reindex(columns=close.columns)
    return close, raw, turn


def load_index(loader: YahooOHLCV, symbol: str, index: pd.Index) -> pd.Series:
    """The IHSG close, aligned to the equity panel's calendar."""
    d = loader.get(symbol, max_age=86400 * 30)
    if d is None or d.empty:
        return pd.Series(dtype=float)
    s = d.set_index("date").sort_index()["close"].astype(float)
    return s.reindex(index).ffill()


def rebalance_positions(index: pd.Index, freq: str, first: int) -> List[int]:
    """Integer bar positions of each period end, from ``first`` onward."""
    rule = REBALANCE[freq]
    marks = pd.Series(np.arange(len(index)), index=index).resample(rule).last()
    return [int(p) for p in marks.dropna().to_numpy() if int(p) >= first]


# --------------------------------------------------------------------------
# factors — every one reads bars at position <= i and nothing after
# --------------------------------------------------------------------------
def _tail(a: np.ndarray, n: int) -> np.ndarray:
    return a[-n:] if len(a) >= n else a


def factor_score(name: str, close: np.ndarray, turn: np.ndarray,
                 idx: np.ndarray,
                 raw: Optional[np.ndarray] = None) -> np.ndarray:
    """Score one factor for every column. Higher score = expected to do better.

    ``close`` is (bars, names) up to and including the decision bar; ``idx`` is
    the index level over the same bars. Nothing later than the last row is
    visible to any branch here, which is the whole point of passing a slice
    rather than the full panel.
    """
    n = close.shape[0]
    last = close[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.diff(close, axis=0) / close[:-1]
    ret = np.where(np.isfinite(ret), ret, np.nan)

    if name == "mom12_1":
        if n < 253:
            return np.full(close.shape[1], np.nan)
        return close[-22] / close[-253] - 1.0
    if name == "mom6_1":
        if n < 127:
            return np.full(close.shape[1], np.nan)
        return close[-22] / close[-127] - 1.0
    if name == "rev1":
        if n < 22:
            return np.full(close.shape[1], np.nan)
        return -(last / close[-22] - 1.0)
    if name == "lowvol":
        return -np.nanstd(_tail(ret, TRADING_DAYS), axis=0)
    if name in ("lowbeta", "lowidio"):
        r = _tail(ret, TRADING_DAYS)
        m = np.diff(idx) / idx[:-1]
        m = np.where(np.isfinite(m), m, np.nan)
        m = _tail(m, r.shape[0])
        good = np.isfinite(m)
        m = m[good]
        r = r[good]
        if len(m) < 60:
            return np.full(close.shape[1], np.nan)
        mv = float(np.var(m))
        if mv <= 0:
            return np.full(close.shape[1], np.nan)
        rc = np.where(np.isfinite(r), r, np.nan)
        mu_r = np.nanmean(rc, axis=0)
        cov = np.nanmean((rc - mu_r) * (m - m.mean())[:, None], axis=0)
        beta = cov / mv
        if name == "lowbeta":
            return -beta
        resid = rc - beta[None, :] * m[:, None]
        return -np.nanstd(resid, axis=0)
    if name == "lowmax":
        r = _tail(ret, 21)
        if r.shape[0] < 10:
            return np.full(close.shape[1], np.nan)
        srt = np.sort(np.where(np.isfinite(r), r, -np.inf), axis=0)
        top = srt[-5:]
        top = np.where(np.isfinite(top), top, np.nan)
        return -np.nanmean(top, axis=0)
    if name == "trend":
        if n < 200:
            return np.full(close.shape[1], np.nan)
        return last / np.nanmean(close[-200:], axis=0) - 1.0
    if name == "high52":
        if n < TRADING_DAYS:
            return np.full(close.shape[1], np.nan)
        return last / np.nanmax(close[-TRADING_DAYS:], axis=0)
    if name == "illiq":
        # ret[i] belongs to bar i+1, so it pairs with turnover at bar i+1.
        t = turn[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(np.isfinite(t) & (t > 0), np.abs(ret) / t, np.nan)
        return np.nanmean(_tail(a, TRADING_DAYS), axis=0) * 1e12
    if name == "divyield":
        # adj_close/close is the accumulated dividend factor; how much it grew
        # over the trailing year IS the trailing yield, and it is knowable at
        # the decision bar. Yahoo rescales the whole history when a new dividend
        # lands, but a RATIO between two bars is invariant to that rescaling,
        # so this stays point-in-time.
        if raw is None or n < TRADING_DAYS + 1:
            return np.full(close.shape[1], np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(raw > 0, close / raw, np.nan)
        y = f[-1] / f[-TRADING_DAYS - 1] - 1.0
        return np.where(np.isfinite(y) & (y >= 0) & (y < 1.0), y, np.nan)
    if name in ("small", "large"):
        med = np.nanmedian(_tail(turn, TRADING_DAYS), axis=0)
        med = np.where(med > 0, med, np.nan)
        lg = np.log(med)
        return lg if name == "large" else -lg
    raise ValueError(f"unknown factor {name!r}")


FACTORS = ("mom12_1", "mom6_1", "rev1", "lowvol", "lowbeta", "lowidio",
           "lowmax", "trend", "high52", "illiq", "small", "large", "divyield")


def eligible_mask(close: np.ndarray, turn: np.ndarray,
                  min_turnover: float, min_hist: int) -> np.ndarray:
    """Tradable at the decision bar, judged only on bars up to it."""
    if close.shape[0] < min_hist:
        return np.zeros(close.shape[1], dtype=bool)
    alive = np.isfinite(close[-1]) & np.isfinite(close[-min_hist])
    med = np.nanmedian(_tail(turn, TRADING_DAYS), axis=0)
    return alive & np.isfinite(med) & (med >= min_turnover)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, ties averaged. Returns nan on fewer than 5 pairs."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def newey_west_t(x: Sequence[float], lag: int = 3) -> float:
    """t-statistic on the mean, with autocorrelation-robust standard errors.

    Monthly factor series are not independent - a crowded factor stays crowded
    for a while - and an ordinary t overstates significance when they are not.
    """
    v = np.asarray([float(z) for z in x if np.isfinite(z)])
    n = len(v)
    if n < 8:
        return np.nan
    e = v - v.mean()
    s = float(e @ e) / n
    for L in range(1, min(lag, n - 1) + 1):
        w = 1.0 - L / (lag + 1.0)
        s += 2.0 * w * float(e[L:] @ e[:-L]) / n
    if s <= 0:
        return np.nan
    return float(v.mean() / np.sqrt(s / n))


def block_bootstrap_p(x: Sequence[float], block: int = 6,
                      draws: int = 4000, seed: int = 20260820) -> float:
    """Two-sided p that the mean differs from zero, resampling whole blocks.

    Blocks keep the month-to-month dependence a naive shuffle would destroy, so
    a factor that was simply hot for one stretch does not clear the bar.
    """
    v = np.asarray([float(z) for z in x if np.isfinite(z)])
    n = len(v)
    if n < 12:
        return np.nan
    rng = np.random.default_rng(seed)
    centred = v - v.mean()
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block, 1), size=(draws, nb))
    means = np.empty(draws)
    for d in range(draws):
        chunks = [centred[s:s + block] for s in starts[d]]
        means[d] = np.concatenate(chunks)[:n].mean()
    return float((np.abs(means) >= abs(v.mean())).mean())


def bonferroni_alpha(n_tests: int, alpha: float = 0.05) -> float:
    return alpha / max(1, n_tests)


# --------------------------------------------------------------------------
# portfolio
# --------------------------------------------------------------------------
def turnover_fraction(old: Sequence[int], new: Sequence[int]) -> float:
    """Value traded as a fraction of the book, moving from one equal-weight
    holding to another.

    sum|dw| counts the sells and the buys, and both are charged: on IDX the buy
    side is 0.28% and the sell side 0.18% plus the 0.1% sales tax, so a uniform
    0.28% per rupiah traded is exact rather than an average. Opening from cash
    is one side and costs 0.28%; a full switch is two and costs 0.56%.
    """
    o, n = set(int(x) for x in old), set(int(x) for x in new)
    if not o and not n:
        return 0.0
    wo = 1.0 / len(o) if o else 0.0
    wn = 1.0 / len(n) if n else 0.0
    both = o & n
    return (len(both) * abs(wn - wo) + len(o - n) * wo + len(n - o) * wn)


def hold_returns(ret: np.ndarray, start: int, stop: int,
                 cols: np.ndarray) -> np.ndarray:
    """Compounded return of each named column over bars [start, stop)."""
    if stop <= start or len(cols) == 0:
        return np.zeros(len(cols))
    seg = ret[start:stop][:, cols]
    seg = np.where(np.isfinite(seg), seg, 0.0)
    return np.prod(1.0 + seg, axis=0) - 1.0


class Board:
    """Everything the panel knows at each rebalance, computed exactly once.

    Eligibility and every factor score depend only on bars up to the decision
    bar, so they are the same whatever portfolio is built on top of them.
    Computing them once and reusing them is not just speed: it guarantees that
    the momentum book, the momentum decile and the momentum IC are all reading
    the identical numbers, so a difference between them is a difference in
    construction and never a difference in inputs.
    """

    def __init__(self, close: pd.DataFrame, turn: pd.DataFrame,
                 idx: np.ndarray, rebal: List[int], factors: Sequence[str],
                 min_turnover: float, min_hist: int, delay: int = 1,
                 raw: Optional[pd.DataFrame] = None):
        self.index = close.index
        self.cv = close.to_numpy(float)
        self.tvv = turn.to_numpy(float)
        self.rw = (raw.to_numpy(float) if raw is not None
                   else close.to_numpy(float))
        self.rebal = list(rebal)
        self.delay = delay
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.diff(self.cv, axis=0) / self.cv[:-1]
        self.ret = np.vstack([np.full((1, self.cv.shape[1]), np.nan), r])
        self.cols: List[np.ndarray] = []
        self.scores: Dict[str, List[np.ndarray]] = {f: [] for f in factors}
        for b in self.rebal:
            m = eligible_mask(self.cv[:b + 1], self.tvv[:b + 1],
                              min_turnover, min_hist)
            self.cols.append(np.flatnonzero(m))
            for f in factors:
                self.scores[f].append(
                    factor_score(f, self.cv[:b + 1], self.tvv[:b + 1],
                                 idx[:b + 1], self.rw[:b + 1]))

    def window(self, k: int) -> Tuple[int, int]:
        """Bars the book put on at rebalance k is held across."""
        n = len(self.cv) - 1
        entry = min(self.rebal[k] + self.delay, n)
        exit_ = min(self.rebal[k + 1] + self.delay, n) \
            if k + 1 < len(self.rebal) else n
        return entry, exit_


def select(cols: np.ndarray, score: Optional[np.ndarray], breadth: int,
           decile: Optional[str] = None,
           rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Which of the eligible names the book holds this period.

    ``score`` None with no generator is the equal-weight universe - the
    portfolio somebody with no view holds, and the benchmark any factor must
    clear before it means anything. ``score`` None with a generator is a random
    draw of the same size, which isolates what breadth alone is worth.
    """
    if len(cols) < 10:
        return np.array([], dtype=int)
    if score is None:
        if rng is None:
            return cols
        k = min(breadth, len(cols))
        return np.sort(rng.choice(cols, size=k, replace=False))
    s = score[cols]
    good = np.isfinite(s)
    cg, sg = cols[good], s[good]
    if not len(cg):
        return np.array([], dtype=int)
    order = np.argsort(-sg, kind="stable")
    if decile == "top":
        return cg[order[:max(1, len(cg) // 10)]]
    if decile == "bottom":
        return cg[order[-max(1, len(cg) // 10):]]
    return cg[order[:breadth]]


def run_portfolio(board: Board, factor: Optional[str], breadth: int,
                  decile: Optional[str] = None,
                  rng: Optional[np.random.Generator] = None
                  ) -> Tuple[pd.Series, List[float], List[int], List[float]]:
    """Equal-weight long-only book.

    Returns (equity, period returns, sizes, cost paid per period). The cost is
    reported separately rather than only netted into the curve, because a
    concentrated book loses to a broad one through TWO channels - it pays more
    turnover and it suffers more variance drag - and lumping them together
    hides which is doing the damage.

    The score is computed on the close of bar r and the book is executed
    ``delay`` bars later, so no fill ever happens on the same print that
    produced the signal.
    """
    equity = 1.0
    curve: Dict[pd.Timestamp, float] = {}
    period, sizes, costs = [], [], []
    held = np.array([], dtype=int)
    for k in range(len(board.rebal)):
        entry, exit_ = board.window(k)
        sc = board.scores[factor][k] if factor is not None else None
        picks = select(board.cols[k], sc, breadth, decile, rng)
        c = ONE_WAY * turnover_fraction(held, picks)
        equity *= (1.0 - c)
        costs.append(c)
        held = picks
        sizes.append(len(picks))
        if not len(picks) or exit_ <= entry:
            period.append(0.0)
            curve[board.index[exit_]] = equity
            continue
        g = float(np.mean(hold_returns(board.ret, entry + 1, exit_ + 1, picks)))
        equity *= (1.0 + g)
        period.append(g)
        curve[board.index[exit_]] = equity
    return pd.Series(curve).sort_index(), period, sizes, costs


def compound(period: Sequence[float]) -> float:
    return float(np.prod([1.0 + r for r in period]))


def annualise(period: Sequence[float], per_year: float) -> float:
    n = len(period)
    if n == 0:
        return np.nan
    return float(compound(period) ** (per_year / n) - 1.0)


def walk_forward_pick(per_factor: Dict[str, List[float]],
                      ew: List[float]) -> Tuple[Optional[str], int]:
    """Pick the factor that led on the first half. Nothing after the split is
    visible to the choice - that is the whole point of making one."""
    n = len(ew)
    split = n // 2
    if split < 12:
        return None, split
    best, best_ex = None, -np.inf
    for f, p in per_factor.items():
        ex = float(np.mean([a - b for a, b in zip(p[:split], ew[:split])]))
        if np.isfinite(ex) and ex > best_ex:
            best, best_ex = f, ex
    return best, split


def split_halves(per_factor: Dict[str, List[float]], ew: List[float],
                 per_year: float) -> pd.DataFrame:
    """Each factor's edge over the neutral book in the first half and the second.

    The walk-forward picks one factor and reports one number, which cannot tell
    a factor that is genuinely persistent from a factor that simply had one good
    stretch. This shows both halves for all of them, so a winner that lives
    entirely in one half is visible as exactly that.
    """
    n = len(ew)
    split = n // 2
    rows = []
    for f, p in per_factor.items():
        a = annualise(p[:split], per_year) - annualise(ew[:split], per_year)
        b = annualise(p[split:], per_year) - annualise(ew[split:], per_year)
        rows.append({"factor": f, "first": a, "second": b,
                     "both_halves": bool(a > 0 and b > 0),
                     "worse_half": min(a, b)})
    return pd.DataFrame(rows).sort_values("worse_half", ascending=False)


def book_correlation(board: "Board", factor: Optional[str], breadth: int,
                     lookback: int = TRADING_DAYS,
                     rng: Optional[np.random.Generator] = None) -> float:
    """Average pairwise correlation among the names a book actually holds.

    A factor book that scores well can be a real cross-sectional effect or one
    sector wearing a factor's name. Twenty coal miners all rank high on trailing
    yield after a coal boom, and a book of twenty coal miners is one bet, not
    twenty. Correlation says which it is without needing a sector table, which
    this repo does not have.
    """
    vals = []
    for k in range(len(board.rebal)):
        sc = board.scores[factor][k] if factor is not None else None
        picks = select(board.cols[k], sc, breadth, None, rng)
        if len(picks) < 3:
            continue
        b = board.rebal[k]
        seg = board.ret[max(1, b - lookback + 1):b + 1][:, picks]
        seg = seg[np.isfinite(seg).all(axis=1)]
        if len(seg) < 60:
            continue
        c = np.corrcoef(seg, rowvar=False)
        iu = np.triu_indices_from(c, k=1)
        v = c[iu]
        v = v[np.isfinite(v)]
        if len(v):
            vals.append(float(v.mean()))
    return float(np.mean(vals)) if vals else np.nan


def name_persistence(board: "Board", factor: Optional[str], breadth: int
                     ) -> float:
    """Share of the book that survives from one rebalance to the next.

    Persistence is the difference between a factor that costs 0.03% a month to
    run and one that costs 0.5%, and at the returns available here that gap is
    most of the answer.
    """
    keep, prev = [], None
    for k in range(len(board.rebal)):
        sc = board.scores[factor][k] if factor is not None else None
        picks = set(select(board.cols[k], sc, breadth).tolist())
        if prev is not None and picks:
            keep.append(len(prev & picks) / len(picks))
        prev = picks
    return float(np.mean(keep)) if keep else np.nan


def variance_drag(period: Sequence[float]) -> Tuple[float, float, float]:
    """Arithmetic mean, geometric mean, and the gap between them per period.

    The gap is not an accounting curiosity: it is the price of concentration.
    Two books with the SAME average period return compound to different money,
    and the more volatile one always compounds to less. Breadth buys that gap
    back without needing to forecast anything, which makes it the one return
    improvement in this repo that does not depend on a signal being real.
    """
    v = np.asarray([float(x) for x in period if np.isfinite(x)])
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    a = float(v.mean())
    g = float(np.exp(np.mean(np.log1p(np.clip(v, -0.999, None)))) - 1.0)
    return a, g, a - g


def cagr(eq: pd.Series) -> float:
    if len(eq) < 2:
        return np.nan
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else np.nan


def max_dd(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1.0).min()) if len(eq) else np.nan


# --------------------------------------------------------------------------
def ic_series(board: Board, factor: str) -> List[float]:
    """One rank IC per rebalance: score today against the next period's return."""
    out = []
    for k in range(len(board.rebal) - 1):
        entry, exit_ = board.window(k)
        cols = board.cols[k]
        if len(cols) < 20 or exit_ <= entry:
            continue
        sc = board.scores[factor][k][cols]
        fwd = hold_returns(board.ret, entry + 1, exit_ + 1, cols)
        out.append(spearman(sc, fwd))
    return [v for v in out if np.isfinite(v)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--breadth", type=int, default=30)
    ap.add_argument("--rebalance", default="monthly",
                    choices=sorted(REBALANCE))
    ap.add_argument("--min-turnover", type=float, default=5e9)
    ap.add_argument("--min-hist", type=int, default=TRADING_DAYS + 30)
    ap.add_argument("--factors", default=",".join(FACTORS))
    ap.add_argument("--price-only", action="store_true",
                    help="run on the raw close, dividends excluded, to show "
                         "what the omission was worth")
    args = ap.parse_args()

    cfg = load_config()
    cache_dir = cfg.path("data.cache_dir", "data/cache")
    loader = YahooOHLCV(cfg, Cache(cache_dir))

    print(f"{'=' * 96}\n IDX CROSS-SECTION — every price-derived factor, "
          f"after costs\n{'=' * 96}")
    close, raw, turn = build_panel(loader, cache_dir, args.start,
                                   total=not args.price_only)
    idx_s = load_index(loader, "^JKSE", close.index)
    idx = idx_s.to_numpy(float) if len(idx_s) else np.full(len(close), np.nan)
    print(f" panel {close.shape[1]} names x {len(close)} bars, "
          f"{close.index[0]:%Y-%m-%d} to {close.index[-1]:%Y-%m-%d}")
    print(f" prices are {'RAW CLOSE, dividends excluded' if args.price_only else 'TOTAL RETURN, dividends reinvested'}"
          f"; the IHSG line below is a PRICE index either way")
    print(f" liquidity floor {args.min_turnover:,.0f} IDR/day median, "
          f"{args.rebalance} rebalance, {args.breadth} names, "
          f"{FEE:.2%} round trip, 1-bar execution delay")

    rebal = rebalance_positions(close.index, args.rebalance, args.min_hist)
    factors = [f for f in args.factors.split(",") if f]
    print(f" {len(rebal)} rebalances; computing eligibility and "
          f"{len(factors)} factor scores once")
    board = Board(close, turn, idx, rebal, factors, args.min_turnover,
                  args.min_hist, raw=raw)

    ew, ew_p, ew_sizes, ew_c = run_portfolio(board, None, args.breadth)
    bench = pd.Series(dtype=float)
    if len(idx_s):
        b = idx_s.reindex(ew.index).dropna()
        bench = b / b.iloc[0] if len(b) else bench

    print(f"\n{'=' * 96}\n BENCHMARKS\n{'=' * 96}")
    print(f" {'equal-weight universe':<26}{ew.iloc[-1]:>8.2f}x "
          f"{cagr(ew):>8.2%}/yr  DD {max_dd(ew):>7.1%}  "
          f"{int(np.median(ew_sizes)):>4d} names")
    if len(bench):
        print(f" {'IHSG (cap-weighted)':<26}{bench.iloc[-1]:>8.2f}x "
              f"{cagr(bench):>8.2%}/yr  DD {max_dd(bench):>7.1%}")

    alpha = bonferroni_alpha(len(factors))
    print(f"\n{'=' * 96}\n RANK IC — score today vs next period's return\n"
          f"{'=' * 96}")
    print(f" {'factor':<10}{'mean IC':>10}{'IC t':>8}{'boot p':>9}"
          f"{'months':>8}   {'verdict at Bonferroni alpha=' + f'{alpha:.4f}':<30}")
    rows = []
    for f in factors:
        ics = ic_series(board, f)
        t = newey_west_t(ics)
        p = block_bootstrap_p(ics)
        sig = (np.isfinite(p) and p < alpha)
        rows.append({"factor": f, "ic": float(np.mean(ics)) if ics else np.nan,
                     "t": t, "p": p, "n": len(ics), "sig": sig})
        print(f" {f:<10}{np.mean(ics) if ics else np.nan:>10.4f}{t:>8.2f}"
              f"{p:>9.4f}{len(ics):>8}   "
              f"{'SIGNIFICANT' if sig else 'not significant'}")

    print(f"\n{'=' * 96}\n LONG-ONLY BOOKS — what could actually be held\n"
          f"{'=' * 96}")
    print(f" {'factor':<10}{'final':>8}{'CAGR':>9}{'max DD':>9}"
          f"{'vs EW':>9}{'vs IHSG':>9}{'excess t':>10}{'boot p':>9}")
    port = []
    per_factor: Dict[str, List[float]] = {}
    for f in factors:
        eq, per, _, _ = run_portfolio(board, f, args.breadth)
        per_factor[f] = per
        ex = [a - b for a, b in zip(per, ew_p)]
        t = newey_west_t(ex)
        p = block_bootstrap_p(ex)
        v_ew = cagr(eq) - cagr(ew)
        v_ix = cagr(eq) - cagr(bench) if len(bench) else np.nan
        port.append({"factor": f, "cagr": cagr(eq), "dd": max_dd(eq),
                     "vs_ew": v_ew, "vs_index": v_ix, "t": t, "p": p,
                     "curve": eq})
        print(f" {f:<10}{eq.iloc[-1]:>8.2f}{cagr(eq):>9.2%}{max_dd(eq):>9.1%}"
              f"{v_ew:>+9.2%}{v_ix:>+9.2%}{t:>10.2f}{p:>9.4f}")

    print(f"\n{'=' * 96}\n DECILE SPREAD — diagnostic only, this account "
          f"cannot short\n{'=' * 96}")
    print(f" {'factor':<10}{'top decile':>12}{'bottom':>10}{'spread/yr':>12}")
    for f in factors:
        top, tp, _, _ = run_portfolio(board, f, args.breadth, decile="top")
        bot, bp, _, _ = run_portfolio(board, f, args.breadth, decile="bottom")
        print(f" {f:<10}{cagr(top):>12.2%}{cagr(bot):>10.2%}"
              f"{cagr(top) - cagr(bot):>+12.2%}")

    per_year = {"monthly": 12.0, "quarterly": 4.0, "annual": 1.0}[args.rebalance]
    print(f"\n{'=' * 96}\n WALK-FORWARD — pick the leader on the first half, "
          f"then live with it\n{'=' * 96}")
    pick, split = walk_forward_pick(per_factor, ew_p)
    if pick is None:
        print(" not enough periods to split.")
    else:
        a = annualise(per_factor[pick][:split], per_year)
        b = annualise(per_factor[pick][split:], per_year)
        ew_a = annualise(ew_p[:split], per_year)
        ew_b = annualise(ew_p[split:], per_year)
        print(f" first half ({split} periods) leader: {pick}")
        print(f"   in-sample   {pick:<10}{a:>8.2%}/yr   vs EW {ew_a:>7.2%}/yr"
              f"   edge {a - ew_a:>+7.2%}")
        print(f"   OUT of sample {pick:<8}{b:>8.2%}/yr   vs EW {ew_b:>7.2%}/yr"
              f"   edge {b - ew_b:>+7.2%}")
        second = {f: annualise(p[split:], per_year) - ew_b
                  for f, p in per_factor.items()}
        rank = sorted(second, key=lambda z: -second[z]).index(pick) + 1
        print(f"   the chosen factor placed {rank} of {len(second)} in the "
              f"second half; a coin flip averages {(len(second) + 1) / 2:.1f}")
        print(f"   average factor edge in the second half: "
              f"{np.mean(list(second.values())):+.2%}/yr")

    print(f"\n{'=' * 96}\n BOTH HALVES — a factor that only worked once is not "
          f"a factor\n{'=' * 96}")
    halves = split_halves(per_factor, ew_p, per_year)
    print(f" {'factor':<10}{'first half':>12}{'second half':>13}"
          f"{'held up?':>10}{'correlation':>13}{'persistence':>13}")
    rnd = np.random.default_rng(77)
    base_corr = book_correlation(board, None, args.breadth, rng=rnd)
    for _, r in halves.iterrows():
        cor = book_correlation(board, r["factor"], args.breadth)
        per = name_persistence(board, r["factor"], args.breadth)
        print(f" {r['factor']:<10}{r['first']:>+12.2%}{r['second']:>+13.2%}"
              f"{'both' if r['both_halves'] else 'one half':>10}"
              f"{cor:>13.3f}{per:>13.0%}")
    print(f" a random book of the same size correlates {base_corr:.3f} within "
          f"itself; a factor book\n far above that line is one sector wearing "
          f"a factor's name.")

    print(f"\n{'=' * 96}\n BREADTH — what diversification is worth without any "
          f"forecast\n{'=' * 96}")
    print(f" {'names':>7}{'arith/period':>14}{'geo/period':>12}"
          f"{'var drag':>10}{'cost':>8}{'CAGR':>9}{'max DD':>9}")
    print(" names are drawn AT RANDOM from the eligible set, averaged over "
          f"{DRAWS} draws,\n so nothing below is a selection effect - it is "
          "breadth and only breadth.")
    for n in (1, 3, 5, 10, 20, 30, 50, 100):
        aa, gg, dd_, cc, mm, kk = [], [], [], [], [], []
        for seed in range(DRAWS):
            rng = np.random.default_rng(1000 + seed)
            eqn, pern, _, cst = run_portfolio(board, None, n, rng=rng)
            a, g, d = variance_drag(pern)
            aa.append(a); gg.append(g); dd_.append(d); kk.append(np.mean(cst))
            cc.append(cagr(eqn)); mm.append(max_dd(eqn))
        print(f" {n:>7}{np.mean(aa):>14.3%}{np.mean(gg):>12.3%}"
              f"{np.mean(dd_):>10.3%}{np.mean(kk):>8.3%}{np.mean(cc):>9.2%}"
              f"{np.mean(mm):>9.1%}")
    a, g, d = variance_drag(ew_p)
    print(f" {'all':>7}{a:>14.3%}{g:>12.3%}{d:>10.3%}{np.mean(ew_c):>8.3%}"
          f"{cagr(ew):>9.2%}{max_dd(ew):>9.1%}   (equal-weight universe)")

    print(f"\n{'=' * 96}\n READING\n{'=' * 96}")
    survivors = [r for r in port if np.isfinite(r["p"]) and r["p"] < alpha
                 and r["vs_ew"] > 0 and r["vs_index"] > 0]
    ic_sig = [r["factor"] for r in rows if r["sig"]]
    print(f" factors with significant rank IC after Bonferroni: "
          f"{', '.join(ic_sig) if ic_sig else 'NONE'}")
    if survivors:
        for r in sorted(survivors, key=lambda z: -z["vs_index"]):
            print(f" {r['factor']} beats BOTH the equal-weight universe "
                  f"({r['vs_ew']:+.2%}/yr) and the IHSG ({r['vs_index']:+.2%}"
                  f"/yr), p={r['p']:.4f}")
    else:
        print(" no long-only factor book beats both the equal-weight universe "
              "and the IHSG\n at a Bonferroni-corrected level. In a SURVIVOR "
              "universe, which flatters\n every one of them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
