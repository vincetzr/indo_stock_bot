# H36/H37/H38 — how accurate is the indicator, on every stock

*891 names / 2,465,286 bars for the turn and placement studies; 777 names /
2,236,165 bars for the whole-board calibration. Code: `scripts/calibrate.py`,
`scripts/turns.py`. Raw: `reports/calibrate*.csv`, `reports/turns_*.csv`.*

Three different questions live inside "how accurate is it", and they have three
different answers. A single accuracy percentage would have hidden all of them.

| | question | answer |
|---|---|---|
| **H36** | are the printed probabilities honest? | **yes — well calibrated, thin skill** |
| **H37** | does the colour turn at the peak? | **near it, ~11% below it, and better than chance** |
| **H38** | where should the target and stop go? | **further out on both — and no placement wins** |

---

## 1. H36 — the probabilities are honest and the skill over the base rate is thin

**A win rate would have been a lie.** The panel emits probabilities, not calls,
so a hit rate is undefined until someone picks a threshold and whoever picks it
decides the answer. What *is* defined: calibration (when it says 60%, does it
happen 60% of the time), skill against the base rate, discrimination (AUC), and
coverage of the date band.

**Every number the panel prints was in-sample until this study.** The reserved
holdout was spent at H16 and cannot be un-spent, but a **purged walk-forward**
is genuinely out of sample: for test year Y the laws are refitted using only
bars whose 252-session window *closed* before Y began. Without that purge a bar
from December Y−1 is still resolving inside Y, and "training" contains the test
year's own outcomes.

| arm | scored | Brier | base-rate Brier | **skill** | AUC |
|---|---|---|---|---|---|
| shipped constants | 4,708,669 | 0.1861 | 0.1935 | **+0.0374** | 0.591 |
| **purged walk-forward** | 4,708,669 | 0.1912 | 0.1935 | **+0.0130** | 0.580 |
| **whole board, 777 names** | 17,000,000+ | 0.1917 | 0.1956 | **+0.0168** | 0.594 |

**Reliability, walk-forward** — the column that answers the question literally:

| predicted | 0.096 | 0.174 | 0.254 | 0.356 | 0.466 | 0.579 | 0.674 | 0.728 | 0.796 | 0.856 |
|---|---|---|---|---|---|---|---|---|---|---|
| **observed** | 0.119 | 0.158 | 0.262 | 0.338 | 0.478 | 0.575 | 0.666 | 0.726 | 0.759 | 0.803 |

Straight through the middle, with the top two bins over-confident by 4–5 points.
The shipped constants are tighter still (0.0936→0.0817 … 0.8274→0.8265).

**The date band does exactly what it claims.** It says half the arrivals land
between the two dates. Measured: **0.497 (shipped), 0.500 (walk-forward), 0.475
(whole board)**, with 25% arriving before the early date. That is the single
best-behaved number in this project.

**And it generalises off the liquid universe.** The laws were fitted on names
clearing Rp1bn/day. Run on the whole board — 777 names, four times the bars,
much thinner — skill is **+0.0168** and AUC **0.594**, both slightly *better*
than the liquid walk-forward. Calibration slips a little in the middle bins
(0.475 predicted against 0.430 observed) and the date band covers 0.475.

**What the skill number means, and it is the honest bad news.** A Brier skill of
+0.013 is the improvement over just quoting the base rate for that target. It is
real and it is small. AUC by target shows where it comes from:

| target | −50% | −33% | −20% | −10% | +5% | +10% | +20% | +50% | 2x |
|---|---|---|---|---|---|---|---|---|---|
| AUC | **0.648** | 0.614 | 0.588 | 0.565 | 0.502 | 0.524 | 0.551 | 0.597 | **0.632** |
| skill | 0.029 | 0.021 | 0.016 | 0.005 | **−0.006** | **−0.005** | 0.007 | 0.038 | 0.015 |

**The near targets carry no information at all** — at +5% and +10% AUC is 0.50
and the skill is *negative*, because almost everything touches +5% inside a year
and there is nothing to discriminate. All the skill is in the far targets, and
more of it on the **downside** than the upside.

Year by year the skill oscillates around zero (−0.074 in 2008, +0.051 in 2009,
+0.040 in 2017, −0.005 in 2019) and AUC runs 0.49 to 0.63. **It is not a stable
edge; it is an honest gauge.**

**The race law is calibrated and carries no discrimination whatsoever.**
P(target first | one arrives): observed 0.5022 against predicted 0.5048 at
10/10, 0.6425 against 0.6422 at 10/20 — near perfect on average. And the skill
is **0.000** and AUC **0.51** at every pair, dipping to **0.464** at 50/10.
Which barrier arrives first is a function of the two distances *you chose* and
nothing about the name. Use it to price a decision, never to make one.

---

## 2. H37 — the colour turns near the top, about 11% below it

Ground truth is a confirmed 10% ZigZag swing high — hindsight on purpose, since
the peak is the thing being predicted. The detectors see nothing they could not
have seen. "Caught" means a flip-down within 30 sessions after the peak.

**The matched null is the whole test.** A detector that flips every five bars
catches every top by construction and also flips two hundred other times, so
each one is scored against a random detector with **the same number of flips**.

| detector | peaks | flips | recall | precision | **F1** | null F1 | lag | **give-back** |
|---|---|---|---|---|---|---|---|---|
| **close over EMA34** | 47,001 | 107,218 | **0.852** | 0.583 | **0.692** | 0.554 | 6 | 10.9% |
| Hull-55 slope | 47,002 | 56,732 | 0.812 | 0.589 | 0.683 | 0.472 | 11 | 11.8% |
| close over EMA50 | 46,984 | 87,956 | 0.758 | **0.593** | 0.666 | 0.531 | 6 | 11.5% |
| HMA-21 over HMA-55 | 47,002 | 61,070 | 0.825 | 0.523 | 0.640 | 0.479 | 11 | 10.5% |
| price>50>100>200 | 46,042 | 37,469 | **0.348** | **0.614** | 0.445 | 0.396 | 9 | 12.5% |

**EMA34 is the most accurate flip of the five**, and every detector beats its
own matched null on F1 — the colour change is not decoration.

**T1 was half wrong.** I registered that they would all sit on one
recall/precision curve so the choice is only where to sit on it. EMA34 has both
higher recall *and* higher precision than the dual-Hull cross, so it dominates
rather than trading off. The trade-off is real between the fast and slow ends,
not everywhere.

**The EMA stack is the outlier and it fails in an interesting direction.** Its
recall of **0.348 is BELOW its own null of 0.367** — the full stack breaks long
after most tops and misses two thirds of them — while its precision of **0.614
is the highest in the table**. It is a confirmation, not a detector.

**T2 CONFIRMED AND IT IS THE NUMBER THAT MATTERS.** The median give-back — how
much of the peak is already gone when the flip fires — is **10.5% to 12.5%**,
against a random detector's **8.5–8.9%**. Read that twice: **the real detectors
surrender MORE of the peak than random bars do**, even though their median lag
is shorter in *time*. That is not a paradox, it is the mechanism — a
trend-following flip fires *because* price has fallen, so it is conditioned on
the drop having already happened. **The colour does not change at the peak. It
changes about a ninth of the way down from it**, and any read of these charts
that assumes otherwise is assuming away the cost of confirmation.

---

## 3. H38 — target and stop placement: further out on both, and it still loses

Entries where a confirmed swing high sits above price and a confirmed swing low
below. The target is placed at the resistance and the stop at the support, then
offset by ±2% and ±5% around each. Exits at the **breaching close**, never at
the level. Net of 56 bps.

**Marginal effect of the target offset** (averaged over every stop offset):

| target offset | −5% | −2% | at the level | +2% | +5% |
|---|---|---|---|---|---|
| median distance | 0.073 | 0.107 | 0.130 | 0.153 | 0.187 |
| P(target first) | 0.540 | 0.471 | 0.436 | 0.404 | 0.367 |
| **mean net** | −0.52% | −0.27% | −0.12% | +0.12% | **+0.35%** |
| mean log | −0.0167 | −0.0167 | −0.0167 | −0.0158 | **−0.0157** |

**T3 WAS WRONG, MONOTONICALLY.** I predicted that selling *into* resistance
would beat waiting for the break, reasoning from H34b's finding that two
breakouts in three fail. The opposite holds on both mean and mean log: **the
further out the target, the better**, even though it is reached less than
half as often. The false-break rate is real and the breaks that do work pay for
all of it — which is H35's "let winners run" arriving from a second direction.

**T4 WAS ALSO WRONG, and in the opposite way to T3.**

| stop offset | −5% (looser) | −2% | at the level | +2% | +5% (tighter) |
|---|---|---|---|---|---|
| median distance | 0.153 | 0.127 | 0.109 | 0.091 | 0.064 |
| mean net | −0.26% | +0.01% | −0.02% | −0.07% | −0.09% |
| **mean log** | **−0.0223** | −0.0172 | −0.0154 | −0.0144 | **−0.0124** |

I predicted the stop offset would matter *less* than the target offset. On mean
log it matters **ten times more** — a 0.0099 spread against the target's 0.0010
— and tighter is better, because a tight stop truncates the log-loss tail. On
the arithmetic **mean** it barely matters at all (0.0026 of spread). That is the
mean/mean-log wedge again, and A18's rule decides it: **an equal-weighted holder
is paid the mean**, so the stop offset is close to free and the target offset is
where the money is.

**And not one of the twenty-five placements is positive in both halves.** Best
cell is target +5% / stop +5%: mean log **−0.0117**, early −0.0056, late
−0.0172. Placement changes the shape of the outcome and does not change its
sign — the same wall H35 hit with fixed-percentage brackets.

---

## What this licenses

- **Believe the probabilities and the date band.** Calibration is good
  out-of-sample and on the whole board, and the band covers exactly the half it
  claims. This is a gauge and it is an honest one.
- **Do not mistake calibration for edge.** Brier skill over the base rate is
  **+0.013** walk-forward, AUC 0.58, and the near targets carry nothing at all.
- **Use EMA34 for the flip if you want the turn**, and know it hands back
  **10.9%** of the peak doing it — more than a random bar would, because
  confirmation is conditioned on the fall.
- **Do not use the flip as an entry.** H33: every flip rule in this family is
  net-negative against holding, and the dual-Hull cross is the worst.
- **Put the target beyond resistance, not in front of it**, and place the stop
  tight if you care about compounding and anywhere sensible if you care about
  the mean.
- **No placement and no bracket makes the round trip pay.** Fifty-five
  combinations across H35 and H38; zero positive in both halves.
