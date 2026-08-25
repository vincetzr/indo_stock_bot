"""Exit rules for the multiplier-cell entry — the half of the design that was
missing.

WHY THIS EXISTS
----------------
H16 ran the published entry rule on 2025-08-25 and held for a fixed year with
no stop. The attribution was unambiguous: the ten names reached a mean PEAK of
**+102.2%** and realised **+15.1%**, giving back 43.4 points. Four peaked above
+100% and two ended there. INET was a triple at session 77 and finished +27.9%.

The entry was not the failure. The absence of an exit was.

And it is structural rather than bad luck. The entry rule selects for
P(2x), which selects for path volatility, which produces large peaks AND large
give-backs by construction. **A selection rule that maximises tail width cannot
be held passively.** Entry and exit are one design and only half of it existed.

THE METHODOLOGICAL POINT THAT DECIDES EVERYTHING HERE
------------------------------------------------------
Exit parameters are trivially overfittable — on the H16 cohort a six-month hold
"returned" +41.4% and a nine-month hold −14.6%, from the same ten names. That
surface is noise, and any parameter read off it is worth nothing.

So every rule in this module is fitted and selected on **pre-holdout data
only** (through 2024-08-23), by walk-forward across cohorts: choose on the
years up to Y, score on Y+1, never the reverse. The 2025 cohort is used once at
the end as an illustration and is explicitly NOT the basis of any choice — the
holdout is already spent (H16) and cannot certify anything now.

WHAT A RULE IS SCORED ON
--------------------------
The COHORT is the unit, not the name. Ten names picked on one date share a
market, a regime and often a sector, so treating them as ten independent
observations overstates the sample by roughly the number of names. Cohort-level
statistics with a cohort-level bootstrap is the honest accounting, and it is
why the numbers here have wider intervals than a naive per-name view.

Costs are charged once per name per cohort — one round trip whether the exit
comes from a stop or from the horizon — at A5's 0.56% plus a fraksi-harga
half-spread at the entry price.
"""

from __future__ import annotations

import inspect
import weakref
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Longest a position is held under any rule, in sessions. One year.
HORIZON = 252

#: A5's round trip, before the spread term the caller adds per name.
FEE = 0.0056


# ==========================================================================
# the rules
# ==========================================================================
def hold(path: np.ndarray, n: int = HORIZON) -> Tuple[float, int]:
    """Buy and hold for ``n`` sessions. The rule H16 actually ran."""
    if not len(path):
        return (np.nan, 0)
    i = min(n, len(path)) - 1
    return (float(path[i]) - 1.0, i + 1)


def trailing(path: np.ndarray, drop: float, arm: float = 0.0,
             n: int = HORIZON) -> Tuple[float, int]:
    """Exit when price falls ``drop`` from its running peak.

    ``arm`` delays the trail until the position is up that much, which stops a
    tight trail from being knocked out by ordinary noise on day three. With
    ``arm=0`` it trails from entry.
    """
    peak = 1.0
    for i, p in enumerate(path[:n]):
        peak = max(peak, p)
        if peak >= 1.0 + arm and p <= peak * (1.0 - drop):
            return (float(p) - 1.0, i + 1)
    return hold(path, n)


def hard_stop(path: np.ndarray, drop: float, n: int = HORIZON
              ) -> Tuple[float, int]:
    """Exit when price falls ``drop`` below the ENTRY, never re-entering."""
    for i, p in enumerate(path[:n]):
        if p <= 1.0 - drop:
            return (float(p) - 1.0, i + 1)
    return hold(path, n)


def time_stop(path: np.ndarray, by: int, need: float, n: int = HORIZON
              ) -> Tuple[float, int]:
    """Exit at session ``by`` unless the position is up ``need`` by then.

    MOTIVATED BY A CLEAN PATTERN, NOT BY A GRID. In the H16 cohort the three
    names that never worked — BLOG, KRYA, MERI — peaked at −3.3%, −1.6% and
    +2.2% within sessions 10 to 14 and then collapsed to −34%, −75% and −55%.
    A name that has not moved in three weeks is a different animal from one
    that has, and this is the cheapest test of that.
    """
    if not len(path):
        return (np.nan, 0)
    k = min(by, len(path)) - 1
    if float(np.max(path[:k + 1])) < 1.0 + need:
        return (float(path[k]) - 1.0, k + 1)
    return hold(path, n)


def combined(path: np.ndarray, stop: float, drop: float, arm: float,
             by: int = 0, need: float = 0.0, n: int = HORIZON
             ) -> Tuple[float, int]:
    """Hard stop, then a trail once armed, with an optional time stop.

    Evaluated in the order a real book would: the hard stop is checked first
    because it is the one that caps the loss.
    """
    peak = 1.0
    for i, p in enumerate(path[:n]):
        if p <= 1.0 - stop:
            return (float(p) - 1.0, i + 1)
        peak = max(peak, p)
        if peak >= 1.0 + arm and p <= peak * (1.0 - drop):
            return (float(p) - 1.0, i + 1)
        if by and i + 1 == min(by, len(path)) and peak < 1.0 + need:
            return (float(p) - 1.0, i + 1)
    return hold(path, n)


def catalogue() -> Dict[str, Callable[[np.ndarray], Tuple[float, int]]]:
    """Every price-path candidate, named. Fixed before any scoring.

    Rules here take ``(path, F=None)`` and ignore ``F`` — the indicator
    catalogue below uses the same signature so both can go through
    ``apply_rule`` unchanged.
    """
    R: Dict[str, Callable] = {}
    for n in (63, 126, 189, 252):
        R[f"hold {n}"] = (lambda p, F=None, n=n: hold(p, n))
    for d in (0.15, 0.20, 0.25, 0.30, 0.40):
        R[f"trail {d:.0%}"] = (lambda p, F=None, d=d: trailing(p, d))
        R[f"trail {d:.0%} armed +50%"] = (
            lambda p, F=None, d=d: trailing(p, d, 0.50))
    for s in (0.15, 0.20, 0.25, 0.30):
        R[f"stop {s:.0%}"] = (lambda p, F=None, s=s: hard_stop(p, s))
    for by, need in ((21, 0.0), (21, 0.10), (42, 0.10)):
        R[f"time {by}d need +{need:.0%}"] = (
            lambda p, F=None, by=by, need=need: time_stop(p, by, need))
    for s in (0.20, 0.25, 0.30):
        for d in (0.25, 0.30, 0.40):
            R[f"stop {s:.0%} + trail {d:.0%} armed +50%"] = (
                lambda p, F=None, s=s, d=d: combined(p, s, d, 0.50))
    for s in (0.25,):
        for d in (0.25, 0.30):
            R[f"stop {s:.0%} + trail {d:.0%} + time 21d"] = (
                lambda p, F=None, s=s, d=d: combined(p, s, d, 0.50, by=21,
                                                     need=0.10))
    return R


# ==========================================================================
# indicator-conditioned rules
# ==========================================================================
#: Bars between observing a signal and filling. A close-to-close rule that
#: fills AT the close it triggered on assumes you knew the close before it
#: printed. One bar is what a retail EOD workflow can actually do, and the
#: price-path rules above are re-scored at the same delay for comparison.
DELAY = 1


def _exit_at(path: np.ndarray, i: int, delay: int = DELAY, n: int = HORIZON
             ) -> Tuple[float, int]:
    """Fill ``delay`` bars after the signal at ``i``, or hold to the horizon."""
    j = i + delay
    if j >= min(n, len(path)):
        return hold(path, n)
    return (float(path[j]) - 1.0, j + 1)


def _armed(path: np.ndarray, i: int, arm: float) -> bool:
    """Has the position been up ``arm`` at any point up to and including i?"""
    return arm <= 0.0 or float(np.max(path[:i + 1])) >= 1.0 + arm


def signal_exit(path: np.ndarray, fire: np.ndarray, arm: float = 0.0,
                warm: int = 0, delay: int = DELAY, n: int = HORIZON
                ) -> Tuple[float, int]:
    """Exit ``delay`` bars after the first armed bar where ``fire`` is True.

    ``fire`` is a boolean array aligned to ``path`` — one flag per forward
    session, each computed from bars at or before that session. ``warm`` skips
    the first few bars, which stops a rule from firing on the entry bar itself
    on a name that was already broken when it was picked.
    """
    m = min(n, len(path), len(fire))
    for i in range(warm, m):
        if bool(fire[i]) and _armed(path, i, arm):
            return _exit_at(path, i, delay, n)
    return hold(path, n)


def ema_break(path: np.ndarray, F, span: int = 20, arm: float = 0.0,
              warm: int = 5) -> Tuple[float, int]:
    """Exit when the adjusted close closes below its own EMA.

    MECHANISM, REGISTERED BEFORE SCORING: a run that is over stops making
    higher closes before it round-trips, and the EMA is the cheapest line that
    notices. Unlike a fixed percentage trail this adapts to the name's own
    speed rather than to a number chosen once for all names.
    """
    c, e = F["close"], F[f"ema{span}"]
    return signal_exit(path, np.less(c, e, where=np.isfinite(e),
                                     out=np.zeros(len(c), bool)), arm, warm)


def ema_cross(path: np.ndarray, F, fast: int = 10, slow: int = 30,
              arm: float = 0.0, warm: int = 5) -> Tuple[float, int]:
    """Exit when the fast EMA closes below the slow one."""
    a, b = F[f"ema{fast}"], F[f"ema{slow}"]
    ok = np.isfinite(a) & np.isfinite(b)
    return signal_exit(path, np.where(ok, a < b, False), arm, warm)


def chandelier(path: np.ndarray, F, k: float = 3.0, arm: float = 0.0,
               warm: int = 5, n: int = HORIZON) -> Tuple[float, int]:
    """Exit when close falls ``k`` ATRs below the highest high since entry.

    MECHANISM: this is the trailing stop H17 selected, but measured in the
    name's own volatility instead of in percent. The multiplier-cell entry
    selects for high realised vol, so a fixed 15% band is a different rule for
    every name it picks — tight enough to be noise on one, loose enough to give
    back half the move on another. A chandelier is the same rule for all of
    them, which is the whole reason to prefer it a priori.
    """
    c, hi, a = F["close"], F["high"], F["atr22"]
    m = min(n, len(path), len(c))
    peak = -np.inf
    for i in range(m):
        h = hi[i] if np.isfinite(hi[i]) else c[i]
        peak = max(peak, h if np.isfinite(h) else -np.inf)
        if i < warm or not np.isfinite(a[i]) or not np.isfinite(peak):
            continue
        if _armed(path, i, arm) and c[i] <= peak - k * a[i]:
            return _exit_at(path, i, DELAY, n)
    return hold(path, n)


def stoch_rollover(path: np.ndarray, F, hot: float = 80.0, look: int = 3,
                   arm: float = 0.0, warm: int = 5) -> Tuple[float, int]:
    """Exit when %K crosses below %D having been overbought recently.

    MECHANISM: the classic exhaustion read. An impulse that has pushed %K above
    ``hot`` and then rolls over has stopped closing near the top of its own
    range, which is the earliest observable sign the buyers have finished.
    """
    k, d = F["stoch_k"], F["stoch_d"]
    ok = np.isfinite(k) & np.isfinite(d)
    was_hot = (pd.Series(np.where(np.isfinite(k), k, -np.inf))
               .rolling(look, min_periods=1).max().to_numpy() >= hot)
    return signal_exit(path, ok & was_hot & (k < d), arm, warm)


def stoch_cool(path: np.ndarray, F, hot: float = 80.0, cool: float = 50.0,
               arm: float = 0.0, warm: int = 5) -> Tuple[float, int]:
    """Exit when %K drops through ``cool`` after having been above ``hot``.

    Slower and less twitchy than the cross: it waits for the midpoint rather
    than reacting to the first two-bar wobble.
    """
    k = F["stoch_k"]
    ok = np.isfinite(k)
    ever_hot = np.maximum.accumulate(np.where(ok, k, -np.inf)) >= hot
    return signal_exit(path, ok & ever_hot & (k < cool), arm, warm)


def volume_climax(path: np.ndarray, F, z: float = 2.0, arm: float = 0.50,
                  warm: int = 5) -> Tuple[float, int]:
    """Exit on an outsized-turnover DOWN bar after the position has run.

    MECHANISM: distribution. A day of far-above-normal rupiah turnover that
    closes lower, in a name already up ``arm``, is supply meeting the bid — the
    signature of someone large finishing. Requires the arm because the same bar
    early in a move is accumulation, not distribution, and the sign of the day
    is what separates them.
    """
    zz, c = F["tvz20"], F["close"]
    dn = np.concatenate([[False], c[1:] < c[:-1]])
    return signal_exit(path, np.isfinite(zz) & (zz >= z) & dn, arm, warm)


def indicator_catalogue() -> Dict[str, Callable]:
    """The indicator search space, fixed before any of it was scored.

    Includes a PRE-REGISTERED PREDICTED-NULL rule. A9 records that `squeeze`
    was registered as a null, came back at t = +3.55 on two million rows, and
    thereby proved that significance is nearly free at that sample size. The
    cheapest protection against a pipeline that manufactures its own signal is
    to always carry a rule that must not work.
    """
    R: Dict[str, Callable] = {}
    for span in (10, 20, 50):
        R[f"ema{span} break"] = (
            lambda p, F, s=span: ema_break(p, F, s, 0.0))
        R[f"ema{span} break armed +50%"] = (
            lambda p, F, s=span: ema_break(p, F, s, 0.50))
    R["ema 10/30 cross"] = (lambda p, F: ema_cross(p, F, 10, 30, 0.0))
    R["ema 10/30 cross armed +50%"] = (
        lambda p, F: ema_cross(p, F, 10, 30, 0.50))
    for k in (2.0, 3.0, 4.0, 5.0):
        R[f"chandelier {k:.0f}x ATR"] = (lambda p, F, k=k: chandelier(p, F, k))
        R[f"chandelier {k:.0f}x ATR armed +50%"] = (
            lambda p, F, k=k: chandelier(p, F, k, 0.50))
    R["stoch rollover"] = (lambda p, F: stoch_rollover(p, F))
    R["stoch rollover armed +50%"] = (
        lambda p, F: stoch_rollover(p, F, arm=0.50))
    R["stoch cool <50"] = (lambda p, F: stoch_cool(p, F))
    R["stoch cool <50 armed +50%"] = (lambda p, F: stoch_cool(p, F, arm=0.50))
    R["volume climax armed +50%"] = (lambda p, F: volume_climax(p, F))
    R["volume climax z3 armed +50%"] = (
        lambda p, F: volume_climax(p, F, z=3.0))
    # combinations: an indicator trail with a floor under it
    for k in (3.0, 4.0):
        R[f"chandelier {k:.0f}x + stop 25%"] = (
            lambda p, F, k=k: _first(chandelier(p, F, k, 0.50),
                                     hard_stop(p, 0.25)))
    R["ema20 break + stoch cool"] = (
        lambda p, F: _first(ema_break(p, F, 20, 0.50), stoch_cool(p, F)))
    # THE NEGATIVE CONTROL — must not work
    R["NULL random exit"] = (lambda p, F: _random_exit(p, F))
    return R


def _first(*results: Tuple[float, int]) -> Tuple[float, int]:
    """Whichever rule fired earliest wins — a book holds one position."""
    ok = [r for r in results if np.isfinite(r[0])]
    return min(ok, key=lambda r: r[1]) if ok else (np.nan, 0)


def _random_exit(path: np.ndarray, F, n: int = HORIZON) -> Tuple[float, int]:
    """PREDICTED NULL. Exit at a bar chosen with no reference to the data.

    Seeded from the path's own length and first value so it is deterministic
    and reproducible, and drawn over the same range the real rules exit in, so
    it carries a comparable holding period and NO information. If this scores
    like the real rules, the real rules are measuring holding period and not
    behaviour, and the whole table should be thrown away.
    """
    m = min(n, len(path))
    if m < 2:
        return hold(path, n)
    seed = int(abs(float(path[0]) * 1e6)) % (2 ** 31) + m
    i = int(np.random.default_rng(seed).integers(1, m))
    return (float(path[i]) - 1.0, i + 1)


# ==========================================================================
# scoring
# ==========================================================================
#: NOT keyed by ``id()``. CPython reuses the id of a garbage-collected object,
#: so a short-lived lambda's arity can be served for a completely different
#: rule created later at the same address — which is exactly what happened, and
#: it presented as "this rule takes one argument but was called with two".
#: A weak-keyed map is both correct and self-cleaning.
_ARITY: "weakref.WeakKeyDictionary[Callable, Tuple[bool, bool]]" = \
    weakref.WeakKeyDictionary()


def rule_arity(rule: Callable) -> Tuple[bool, bool]:
    """``(accepts_features, requires_features)`` for one rule, memoised.

    DECIDED BY INSPECTION, NOT BY CATCHING TypeError. The first version wrapped
    the call in ``except TypeError`` and turned any failure into a silent NaN —
    which meant a one-argument rule was dropped wholesale and, worse, that a
    genuine bug inside a rule would have been recorded as "this name had no
    data". A blanket except around the thing you are measuring is how a broken
    rule comes to look like a merely inapplicable one.

    "Accepts" is having a second positional parameter; "requires" is that
    parameter having no default. So ``lambda p, F=None, d=d: ...`` is handed
    the features and free to ignore them, while ``lambda p, F: ...`` is a rule
    that cannot run without them and whose name is dropped when they are absent.
    """
    try:
        got = _ARITY.get(rule)
    except TypeError:                                 # unhashable callable
        got = None
    if got is not None:
        return got
    try:
        ps = [p for p in inspect.signature(rule).parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    except (TypeError, ValueError):                   # builtins, C callables
        out = (False, False)
    else:
        accepts = len(ps) >= 2
        out = (accepts,
               accepts and ps[1].default is inspect.Parameter.empty)
    try:
        _ARITY[rule] = out
    except TypeError:
        pass
    return out


def apply_rule(paths: Sequence[np.ndarray], costs: Sequence[float],
               rule: Callable, feats: Optional[Sequence[dict]] = None
               ) -> pd.DataFrame:
    """Net return and holding period per position under one rule.

    ``feats[i]`` is the indicator frame for ``paths[i]`` — a dict of arrays
    aligned bar-for-bar with the path. Price-path rules take it and ignore it;
    indicator rules require it, and a name whose indicators are missing is
    DROPPED rather than silently held to the horizon, because a rule that
    cannot fire is a different rule and averaging the two together hides which
    one actually ran.
    """
    accepts, requires = rule_arity(rule)
    out = []
    for i, (p, c) in enumerate(zip(paths, costs)):
        F = feats[i] if feats is not None and i < len(feats) else None
        if requires and F is None:
            out.append({"gross": np.nan, "cost": c, "net": np.nan, "held": 0})
            continue
        r, held = rule(np.asarray(p, dtype=float), F) if accepts \
            else rule(np.asarray(p, dtype=float))
        out.append({"gross": r, "cost": c, "net": r - c, "held": held})
    return pd.DataFrame(out)


def score_cohorts(cohorts: Dict[pd.Timestamp, Dict[str, object]],
                  rule: Callable) -> pd.DataFrame:
    """One row per cohort: what the rule did to that date's ten names."""
    rows = []
    for day, c in cohorts.items():
        D = apply_rule(c["paths"], c["costs"], rule, c.get("feats"))
        D = D[np.isfinite(D["net"])]
        if D.empty:
            continue
        rows.append({
            "as_of": day, "n": len(D),
            "mean": float(D["net"].mean()),
            "median": float(D["net"].median()),
            "p2": float((D["gross"] >= 1.0).mean()),
            "pdn": float((D["gross"] <= -0.5).mean()),
            "held": float(D["held"].mean()),
            "uni_med": c.get("uni_med", np.nan)})
    return pd.DataFrame(rows)


def cohort_block(S: pd.DataFrame, horizon: int = HORIZON) -> int:
    """How many consecutive cohorts share a forward window.

    Monthly cohorts each holding a year overlap in eleven of their twelve
    months, so twelve consecutive cohorts are very nearly one observation.
    Inferred from the actual spacing of ``as_of`` rather than assumed, because
    the same function is used on weekly and quarterly cohort sets.
    """
    if "as_of" not in S.columns or len(S) < 3:
        return 1
    d = pd.DatetimeIndex(S["as_of"]).sort_values()
    # NOT ``.asi8``: this pandas builds date ranges as datetime64[us], so the
    # integer view is microseconds here and nanoseconds elsewhere, and a
    # hardcoded divisor was silently off by a thousand — it returned a block
    # of 11,783 for monthly cohorts, which then clipped to a degenerate value.
    step = float(np.median(np.diff(
        d.to_numpy().astype("datetime64[D]").astype("int64"))))
    if not np.isfinite(step) or step <= 0:
        return 1
    return int(max(1, np.ceil(horizon * 365.25 / 252 / step)))


def bootstrap_cohorts(S: pd.DataFrame, col: str = "mean", draws: int = 2000,
                      seed: int = 20260825, block: Optional[int] = None
                      ) -> Tuple[float, float]:
    """CI on the average cohort statistic, by MOVING-BLOCK resample.

    Two levels of dependence have to be respected and the first version of
    this function respected only one.

    *Within a cohort* — ten names chosen on one date share a market, a regime
    and often a sector, so the unit is the cohort, never the name.

    *Between cohorts* — and this is the one that was missed. Monthly cohorts
    holding for a year overlap in eleven months out of twelve, so 188 of them
    span about sixteen independent year-windows. An iid resample over them
    treats near-duplicates as fresh evidence and returns an interval roughly
    ``sqrt(188/16) ≈ 3.4x`` too narrow. H16 made exactly this point about its
    own twelve cohorts ("effective n is ~1, not 12") and then the exit study
    reintroduced it one layer up.

    Blocks are drawn WITH replacement and duplicates are kept — A11 records
    that filtering them with ``np.isin`` silently shrank every resample and
    made every interval too narrow, which is the failure mode a bootstrap is
    supposed to protect against.
    """
    T = S.dropna(subset=[col])
    if "as_of" in T.columns:
        T = T.sort_values("as_of")
    x = T[col].to_numpy(dtype=float)
    n = len(x)
    if n < 5:
        return (np.nan, np.nan)
    # A block that is a large fraction of the sample DEGENERATES: with only
    # three or four blocks covering the whole series each block mean is already
    # close to the sample mean, and the interval collapses back to something
    # narrower than the iid one. Measured here: widths 0.049 (b=1), 0.105
    # (b=13), 0.047 (b=63) on n=189. Cap at a fifth so that can't happen
    # quietly.
    b = int(block) if block is not None else cohort_block(T)
    b = int(np.clip(b, 1, max(1, n // 5)))
    rng = np.random.default_rng(seed)
    if b == 1:
        m = rng.choice(x, size=(draws, n), replace=True).mean(axis=1)
    else:
        k = int(np.ceil(n / b))
        start = rng.integers(0, n - b + 1, size=(draws, k))
        take = (start[:, :, None] + np.arange(b)[None, None, :]
                ).reshape(draws, k * b)[:, :n]
        m = x[take].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def walk_forward_select(cohorts: Dict[pd.Timestamp, Dict[str, object]],
                        rules: Dict[str, Callable],
                        min_train: int = 24,
                        purge: bool = True,
                        objective: str = "median") -> pd.DataFrame:
    """Choose the rule on SETTLED past cohorts only, score it on the next one.

    THIS IS THE WHOLE POINT OF THE MODULE. Picking the best rule over the full
    sample and quoting its full-sample return is how a six-month hold comes to
    look like +41%. Here the choice at cohort *t* sees only cohorts before *t*,
    so the reported series is what the procedure would actually have earned.

    THE PURGE IS NOT OPTIONAL AND THE FIRST VERSION OF THIS FUNCTION OMITTED
    IT. Cohorts are monthly and the horizon is a full year, so last month's
    cohort has eleven months still to run when this month's decision is made —
    its outcome is not known and training on it is a look-ahead of exactly the
    kind §11 forbids. With the purge on, the training set at cohort *t* holds
    only cohorts whose forward window CLOSED on or before *t*, which for a
    252-session horizon discards the most recent ~12 cohorts at every step.

    Each cohort may carry ``settles`` — the date its horizon ends. Without one
    the horizon is approximated on the calendar (252 sessions ≈ 366 days),
    which is conservative in the right direction because it over-purges.

    THE OBJECTIVE IS A CHOICE AND IT DECIDES THE ANSWER. ``objective="median"``
    picks the rule with the best average cohort median, which rewards being
    right often. ``"mean"`` rewards total return and therefore protects the
    right tail. ``"p2"`` maximises the doubling rate directly. On an entry rule
    SELECTED FOR P(2x) these disagree sharply — the median-optimal exit in H18
    cut the doubling rate from 7.3% to 2.3% — so the objective must be stated
    with any result, never left implicit.

    Returns one row per scored cohort: which rule was chosen, what it earned,
    and what plain buy-and-hold earned on the same names.
    """
    if objective not in ("median", "mean", "p2"):
        raise ValueError(f"objective must be median/mean/p2, got {objective!r}")
    days = sorted(cohorts)
    approx = pd.Timedelta(days=int(HORIZON * 365.25 / 252) + 1)
    settles = {d: (cohorts[d].get("settles") or d + approx) for d in days}
    per_rule = {name: score_cohorts(cohorts, fn).set_index("as_of")
                for name, fn in rules.items()}
    base = score_cohorts(cohorts,
                         lambda p, F=None: hold(p, HORIZON)
                         ).set_index("as_of")
    rows = []
    for i, d in enumerate(days):
        past = [p for p in days[:i] if not purge or settles[p] <= d]
        if len(past) < min_train:
            continue
        best, best_v = None, -np.inf
        for name, S in per_rule.items():
            v = S.reindex(past)[objective].mean()
            if np.isfinite(v) and v > best_v:
                best, best_v = name, v
        if best is None or d not in per_rule[best].index:
            continue
        r = per_rule[best].loc[d]
        rows.append({"as_of": d, "rule": best, "objective": objective,
                     "median": r["median"], "mean": r["mean"],
                     "p2": r["p2"], "pdn": r["pdn"], "held": r["held"],
                     "bh_median": base.loc[d, "median"] if d in base.index
                                  else np.nan,
                     "bh_mean": base.loc[d, "mean"] if d in base.index
                                else np.nan,
                     "uni_med": r.get("uni_med", np.nan)})
    return pd.DataFrame(rows)
