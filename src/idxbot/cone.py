"""The price×time cone: the two fitted laws the Pine panel prints.

This module is the SINGLE SOURCE OF THE COEFFICIENTS. `pine/IDX_Suite.pine`
carries the same numbers because Pine cannot import anything, and
`scripts/pine_cone_check.py` re-derives the Pine arithmetic here and scores it
against the measured cells so the two cannot drift apart silently.

WHAT THE LAWS ARE
-----------------
Fitted in H32b to 180 cells — 10 volatility deciles × 9 target levels × trend
state — over 623,126 eligible IDX entries, 2000-2024 pre-holdout.

    d      = |log(target / entry)|          the log distance to the target
    sigma  = vol60, the SAMPLE sd (ddof=1) of simple daily returns over 60 bars

    terms  = [1, log d, (log d)^2, log sigma, log d · log sigma] (+ stack)

    log(sessions to first touch) = terms · coefficients
    logit P(touch within 252)    = terms · coefficients + g·stack

THE QUADRATIC AND THE INTERACTION ARE NOT DECORATION. The first version of this
fit was linear in log d and it under-predicted P(+20%) by ELEVEN probability
points — at the exact target the panel ships as its default. Over a distance
range running from +5% to 2x the logit is visibly curved and a straight line
sags through the middle, which is where a user actually sets the dial. Adding
the two terms takes the median error from 4.2 points to 1.7 on the upside and
the median time error from 14% to 6%. A fit is only as good as the cell the
reader asks for, so check the fit AT the default before shipping it.

WHAT THEY ARE NOT
-----------------
Not a diffusion result. A driftless random walk reaches a log barrier d with
per-bar sd sigma in a time scaling as (d/sigma)^2; these fit near
d^0.89 / sigma^0.59. The series trends, its volatility clusters, and — the term
that matters most — the sample CONDITIONS ON TOUCHING WITHIN 252 SESSIONS. That
censoring is why the q3 exponent on distance (0.59) is so much flatter than
q1's (1.08): the top of the date band is partly the horizon, not the market.
Nothing here extrapolates past a year.

Not a forecast either. Every number is an in-sample frequency over a panel, and
the band is a quartile band: half the cases that reach the target do so between
the two dates, which leaves the other half outside them.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

#  [1, log d, (log d)^2, log sigma, log d * log sigma]
#  R2 / median error / p90 error:
#     q1  0.981 /  7.3% / 25.2%
#     med 0.984 /  6.1% / 17.2%
#     q3  0.985 /  4.6% / 12.4%
TIME_LAW: Dict[str, Tuple[float, float, float, float, float]] = {
    "q1":  (3.4462, 1.5154, -0.1484, -0.3646, 0.2492),
    "med": (4.2905, 1.2640, -0.1530, -0.2359, 0.2380),
    "q3":  (5.0526, 0.9246, -0.1383, -0.0827, 0.2134),
}

#  Same terms plus a stack dummy. Median absolute error 1.71 probability points
#  on the upside and 1.75 on the downside; p90 4.4 and 4.0.
#
#  THE STACK TERMS ARE THE FINDING. exp(+0.1750) = 1.19 and exp(-0.3388) = 0.71,
#  so a confirmed uptrend multiplies upside odds by 1.19 and downside odds by
#  0.71 — nearly twice the effect on the side nobody puts in the headline.
#  A trend filter is mostly a RISK filter.
PROB_LAW: Dict[str, Tuple[float, float, float, float, float, float]] = {
    "up":   (1.3567, -1.1390, -0.3009, 1.1850, 0.3405,  0.1750),
    "down": (1.5117, -1.3308, -0.3158, 1.0515, 0.2545, -0.3388),
}

#  H35. logit P(the target is reached BEFORE the stop | one of them is) on
#  [1, log tp, log sl, log sigma], fitted to 300 (tp, sl, volatility-decile)
#  cells. Median error 1.26 probability points, p90 3.67.
#
#  THE VOLATILITY TERM IS ALMOST EXACTLY ZERO (-0.0225), and that is the
#  finding: which barrier arrives first is a RATIO OF DISTANCES, not a question
#  about how fast the name moves. Volatility speeds both up equally. The two
#  distance coefficients are near mirror images (-0.764, +0.816), so the law is
#  very nearly a function of log(sl / tp) alone.
RACE_LAW: Tuple[float, float, float, float] = (0.0588, -0.7635, 0.8160, -0.0225)

#  The fitted range of sigma, from the 10 volatility deciles the cells came
#  from. Outside it the laws are extrapolation and callers must be told.
SIGMA_MIN, SIGMA_MAX = 0.0117, 0.0623
#  vol60 decile edges over 623,126 eligible IDX name-days.
VOL_DECILES = (0.0145, 0.0179, 0.0205, 0.0230, 0.0256,
               0.0289, 0.0331, 0.0393, 0.0509)
#  252 sessions a year is 1.448 calendar days each. An average, which is why the
#  output is a band of dates and never a day.
DAYS_PER_SESSION = 365.25 / 252.0
HORIZON = 252


def vol_decile(sigma: float) -> int:
    """1 = calmest tenth of IDX, 10 = wildest. Context for the date band: the
    median time to +20% runs 89 sessions in decile 1 and 30 in decile 10."""
    for i, edge in enumerate(VOL_DECILES, start=1):
        if sigma <= edge:
            return i
    return 10


def sessions_to(target_mult: float, sigma: float, quantile: str = "med",
                clamp: bool = True) -> float:
    """Sessions until first touch of entry × target_mult, at `quantile`."""
    d = abs(math.log(target_mult))
    if d <= 0.0 or not sigma > 0.0:
        return float("nan")
    if clamp:
        sigma = min(max(sigma, SIGMA_MIN), SIGMA_MAX)
    a, b, c, e, f = TIME_LAW[quantile]
    ld, ls = math.log(d), math.log(sigma)
    return math.exp(a + b * ld + c * ld * ld + e * ls + f * ld * ls)


def p_touch(target_mult: float, sigma: float, stack: bool = False,
            clamp: bool = True) -> float:
    """P(the path touches entry × target_mult within 252 sessions).

    The side is read off the multiplier, so p_touch(1.2, ...) and
    p_touch(0.8, ...) answer the two halves of the same trade. Quoting the
    first without the second is quoting half a trade.
    """
    d = abs(math.log(target_mult))
    if d <= 0.0 or not sigma > 0.0:
        return float("nan")
    if clamp:
        sigma = min(max(sigma, SIGMA_MIN), SIGMA_MAX)
    a, b, c, e, f, g = PROB_LAW["up" if target_mult > 1.0 else "down"]
    ld, ls = math.log(d), math.log(sigma)
    z = a + b * ld + c * ld * ld + e * ls + f * ld * ls + (g if stack else 0.0)
    return 1.0 / (1.0 + math.exp(-z))


def p_target_first(tp: float, sl: float, sigma: float,
                   clamp: bool = True) -> float:
    """P(the target is reached before the stop | one of them is reached).

    `tp` and `sl` are positive fractions — 0.20 means +20% and −20%. This is the
    question a bracket actually asks, and the two touch probabilities do not
    answer it: over a year both barriers are usually reachable, so what decides
    the trade is which arrives first.

    IT DOES NOT SAY THE BRACKET MAKES MONEY. H35 scored thirty (tp, sl) pairs
    on 17.6 million entries and **not one was positive in both halves** once
    fills were taken at the actual close, duration was matched, and the result
    was annualised. The best cell compounds at +2.4%/yr against an index at
    about +12.7%. This function prices a decision the user has already made.
    """
    if not (tp > 0 and sl > 0 and sigma > 0):
        return float("nan")
    if clamp:
        sigma = min(max(sigma, SIGMA_MIN), SIGMA_MAX)
    a, b, c, e = RACE_LAW
    z = (a + b * math.log(tp) + c * math.log(sl) + e * math.log(sigma))
    return 1.0 / (1.0 + math.exp(-z))


def in_domain(sigma: float) -> bool:
    return bool(SIGMA_MIN <= sigma <= SIGMA_MAX)


def cone(target_pct: float, sigma: float, stack: bool = False) -> Dict[str, float]:
    """The whole panel row: both sides, the odds ratio, and the date band."""
    up, dn = 1.0 + target_pct / 100.0, 1.0 - target_pct / 100.0
    pu, pd_ = p_touch(up, sigma, stack), p_touch(dn, sigma, stack)
    return {"p_up": pu, "p_down": pd_,
            "odds": pu / pd_ if pd_ else float("nan"),
            "q1": sessions_to(up, sigma, "q1"),
            "med": sessions_to(up, sigma, "med"),
            "q3": sessions_to(up, sigma, "q3"),
            "in_domain": float(in_domain(sigma))}


# ============================ H42 — what the SCANNER'S OWN LIST actually did ==
#  Measured, not fitted: 116,754 replayed signals over 706 names, 2000-2026,
#  each walked forward 252 sessions, filled at the ACTUAL close of the exit bar.
#  `picks` is the mean return of the scanner's row; `random` is the same
#  bracket distances applied to a randomly chosen eligible IDX name on the same
#  date; `hold` is simply owning the name for the year.
#
#  THE TABLE EXISTS SO NO ROW CAN BE QUOTED WITHOUT ITS OUTCOME. A11's rule —
#  a conditional result quoted without its condition is a wrong result, and the
#  fix belongs in the code that prints it.
#
#  The bin edges were fixed in `scripts/signal_backtest.py` BEFORE the study
#  ran, so the 1.5 cut below is a sign change at a pre-registered boundary and
#  not the maximum of a sweep.
#
#    r:r       n     share   picks   random    diff   both halves    hold
BRACKET_CELLS = (
    #  (rr_lo, rr_hi, share, picks, random, diff, replicates, hold)
    (0.00, 0.75, 0.54, -0.0027, +0.0045, -0.0072, False, +0.1501),
    (0.75, 1.50, 0.22, -0.0018, +0.0043, -0.0061, False, +0.1243),
    (1.50, 2.50, 0.11, +0.0031, +0.0023, +0.0008, True,  +0.1040),
    (2.50, 4.00, 0.07, +0.0085, +0.0056, +0.0029, False, +0.0779),
    (4.00, 1e9, 0.06, +0.0201, +0.0092, +0.0109, True,  +0.0926),
)
#  Below this the bracket is negative in absolute terms AND negative against a
#  random name in BOTH halves. It is 76% of everything the scanner used to
#  print, and it is the half that hits its target most often.
MIN_RR = 1.5
#  Bracketing at all, against owning the name for the year, paired and
#  (ticker, year) block-bootstrapped over 112,190 signals.
BRACKET_VS_HOLD = (-0.1306, -0.1625, -0.1022)


def bracket_cell(rr: float):
    """The measured outcome of the reward-to-risk cell a live row occupies."""
    for lo, hi, share, picks, rnd, diff, both, hold in BRACKET_CELLS:
        if lo <= rr < hi:
            return {"lo": lo, "hi": hi, "share": share, "picks": picks,
                    "random": rnd, "diff": diff, "replicates": both,
                    "hold": hold, "vs_hold": picks - hold}
    return None
