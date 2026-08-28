# H34/H35 — support, resistance, Fibonacci, and the take-profit / stop grid

*891 names, 2,465,286 pre-holdout bars, 2000-03-30 → 2024-08-23. Code:
`scripts/levels.py`. Raw: `reports/levels*.csv`, `reports/levels.txt`.*

Asked to add take-profit and stop-loss levels drawn from support/resistance and
Fibonacci, and to set them from history so as to maximise profit and minimise
loss. Three separate questions, three separate answers, and they do not agree
with each other.

| | question | answer |
|---|---|---|
| **H34a** | are the Fibonacci ratios special? | **no — and the curve is perfectly smooth** |
| **H34b** | is a prior swing high a real barrier? | **yes, and it replicates in both halves** |
| **H35** | does any (take-profit, stop) bracket beat holding? | **no — 0 of 30 cells survive the half-split** |

So the levels are worth drawing; the bracket is not worth trading. Those are
different claims and the chart now carries both.

---

## 1. H34a — Fibonacci is a smooth function read at five arbitrary points

**The design is the whole result.** "Price bounced off the 61.8%" is not
evidence, because a shallow retracement is reached by every pullback and a deep
one only by big ones, so any test comparing 0.618 to 0.90 compares *depths* and
will always find the shallow level stronger. The grid here is therefore **fine
and continuous — 0.15 to 0.95 in steps of 0.025** — and the question is not
whether 0.618 works but **whether it stands out from 0.60 and 0.65**. Those
neighbours are matched on depth almost exactly and are not Fibonacci numbers.

280,228 (up-leg, ratio) touches, levels gated on the ZigZag **confirmation** bar
so nothing uses a pivot before it was knowable.

| retracement | n | P(regain leg high in 60b) | fwd 20b mean | P(−10% more) | |
|---|---|---|---|---|---|
| 0.150 | 10,347 | 0.3964 | −0.0185 | 0.8831 | |
| 0.225 | 10,262 | 0.3933 | −0.0185 | 0.8354 | **≈ 0.236** |
| 0.300 | 10,048 | 0.3871 | −0.0174 | 0.7950 | |
| 0.375 | 9,707 | 0.3769 | −0.0172 | 0.7635 | **≈ 0.382** |
| 0.450 | 9,245 | 0.3650 | −0.0172 | 0.7426 | |
| 0.475 | 9,090 | 0.3631 | −0.0165 | 0.7329 | |
| **0.500** | 8,954 | **0.3611** | −0.0165 | 0.7239 | **Fibonacci** |
| 0.525 | 8,793 | 0.3571 | −0.0165 | 0.7182 | |
| 0.600 | 8,273 | 0.3446 | −0.0174 | 0.7060 | |
| 0.625 | 8,134 | 0.3414 | −0.0163 | 0.6997 | **≈ 0.618** |
| 0.775 | 7,154 | 0.3225 | −0.0179 | 0.6779 | **≈ 0.786** |
| 0.950 | 5,957 | 0.2911 | −0.0210 | 0.6807 | |

**Look at 0.500: 0.3611, sitting exactly between 0.475's 0.3631 and 0.525's
0.3571.** The curve declines monotonically from 0.3964 to 0.2911 with no bump
anywhere. Fitting a quadratic in the ratio and testing whether the five
Fibonacci nodes sit above the residual, against 2,000 draws of five randomly
chosen non-Fibonacci nodes from the same grid:

| statistic | observed residual | null mean | null sd | z | p |
|---|---|---|---|---|---|
| P(regain leg high) | +0.0005 | 0.0000 | 0.0006 | **+0.77** | 0.447 |
| fwd 20b mean | +0.0001 | 0.0000 | 0.0002 | +0.68 | 0.507 |
| fwd 20b median | +0.0001 | −0.0000 | 0.0001 | +0.41 | 0.691 |
| P(−10% further) | −0.0013 | −0.0000 | 0.0013 | −0.95 | 0.355 |

**L1 confirmed on all four.** The largest effect is five hundredths of a
percentage point on a probability of 0.36. There is nothing at the Fibonacci
ratios that is not at every other ratio.

**Two by-products worth more than the headline.** The forward 20-session mean
return after touching a retracement level is **negative at every depth**
(−0.016 to −0.021), before costs — buying a pullback is not a positive-
expectancy act on this market at any depth. And P(the leg high is regained
within 60 sessions) never exceeds **40%**: the "it'll come back" premise fails
three times in five even at the shallowest retracement.

---

## 2. H34b — a prior swing high IS a real barrier, and it replicates

Same construction, different level. The event is *the first close above a
confirmed prior swing high*; the placebo is the first close above **that same
high displaced ±7%** — an identical construction against the wrong level. 23,235
events.

The pooled rows cannot be read on their own, because a level displaced downward
is crossed early in a rally and one displaced upward late, so the two sit at
different points in the move. A linear probability model in the distance (and
its square) absorbs that, with a **ticker-clustered** bootstrap because one name
contributes many events from overlapping legs:

| statistic | true-level effect | 95% CI | early half | late half |
|---|---|---|---|---|
| **false break** (closes back under within 20b) | **−6.95 pp** | [−8.26, −5.70] | −8.60 | −7.11 |
| follow-through (+5% within 20b) | **+4.46 pp** | [+3.08, +5.58] | +5.27 | +4.04 |
| forward 20-session return | **+1.00%** | [+0.58, +1.37] | +1.33 | +1.40 |

**L2 confirmed, and larger than predicted.** I registered "a few percentage
points"; the false-break rate is seven points lower at the true level. All three
statistics replicate in both halves — **the first level-based result in this
repo that survives a half-split.**

Even so, the pooled false-break rate at the *true* level is **66.4%**: two
breakouts in three still close back under the level within twenty sessions. The
finding is that the true level is *better than a fake one*, not that breakouts
work.

**What this does and does not license.** It licenses drawing prior swing highs
and lows as meaningful reference points, and placing a target or a stop with
reference to them rather than at a round number. It does **not** license a
breakout strategy: +1.00% over twenty sessions is the *difference against a
placebo level*, measured on single events with no portfolio construction, no
capacity check, and no accounting for the fact that this repo has watched
conditional means of exactly this size vanish under portfolio accounting four
times (A18, A19).

---

## 3. H35 — the bracket grid, and three controls that each cut the result

Every eligible bar is an entry; 30 (take-profit, stop) pairs plus a plain hold,
252-session horizon, 17,593,275 entry-cells, net of 56 bps.

Three controls decide this table and each one was added because the previous
version was flattering:

1. **Fill at the actual close, not at the nominal level.** A bar that breaches
   −5% often closes at −8%, and on IDX a name can gap to ARB and be untradeable
   all day. Crediting a stop with its nominal price flatters exactly the tight
   stops that win the grid. Correcting it took the best cell's mean log from
   +0.0173 to **+0.0050**.
2. **Match the duration.** The best cell is invested **52 sessions out of 252**
   and holds cash for the other 200. On a market whose per-name yearly log
   return is −0.0587, being out of it is most of what a stop does.
3. **Annualise.** A per-trade figure is not a yearly one — a 52-session bracket
   is redeployed about five times a year. This is A21's lesson, where a ten-year
   doubling rate was quoted as if annual.

**Plain hold, 252 sessions:** mean **+9.58%**, median −3.78%, mean log −0.0587
(early +0.0031, late −0.1197).

| tp | sl | P(tp first) | P(sl first) | bars held | mean | mean log | **annualised** | early | late | both +? |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 0.05 | 0.137 | **0.806** | 52 | +2.48% | +0.0050 | **+2.40%** | +8.04% | **−3.16%** | no |
| 0.30 | 0.05 | 0.207 | 0.770 | 40 | +1.47% | +0.0017 | +1.09% | +6.73% | −4.47% | no |
| 0.50 | 0.10 | 0.190 | 0.700 | 82 | +3.10% | +0.0007 | +0.22% | +5.68% | −5.17% | no |
| 0.20 | 0.05 | 0.265 | 0.723 | 31 | +0.87% | −0.0005 | −0.40% | +5.07% | −5.79% | no |
| 0.15 | 0.05 | 0.309 | 0.682 | 26 | +0.57% | −0.0016 | −1.54% | +3.61% | −6.63% | no |
| … | | | | | | | | | | |
| 0.10 | 0.30 | 0.670 | 0.250 | 73 | −1.72% | −0.0297 | −10.3% | −6.2% | −14.6% | no |

**L3 confirmed and then some. Not one of the thirty cells is positive in both
halves.** The best annualises to **+2.4%/yr** against the index's ~+12.7%, and
its late half is **−3.2%**.

**L4 confirmed, with a structure I did not predict.** At tight symmetric
distances the stop is hit first more often — 5%/5% gives P(sl first) 0.508
against P(tp first) 0.489 — and the asymmetry **reverses at wide distances**:
20%/20% gives 0.472 tp against 0.444 sl. Tight brackets are dominated by the
negative short-horizon drift and the spread; wide ones start to catch the fat
right tail. Symmetry is not neutrality either way.

**The one shape that is real, and it is a shape, not a strategy.** Against a
hold of its *own* duration, the edge is monotone in the direction of the
folklore:

| | edge on mean log vs duration-matched hold |
|---|---|
| tp 0.50 / sl 0.10 | **+0.0262** |
| tp 0.50 / sl 0.05 | +0.0229 |
| tp 0.30 / sl 0.05 | +0.0165 |
| tp 0.15 / sl 0.15 | +0.0072 |
| tp 0.05 / sl 0.15 | −0.0003 |
| tp 0.10 / sl 0.30 | **−0.0064** |
| tp 0.05 / sl 0.30 | −0.0077 |

**Cut losses short and let winners run is measurably the right direction** — and
the whole width of that effect, from best to worst, is 3.4 points of log return
per trade, which the annualised column then hands back to costs and to the
regime break. Direction right, size insufficient. That has now been the answer
in this repo from flow, broker identity, investor class, price features, exit
rules, index timing, and now brackets.

---

## What this licenses

- **Draw prior swing highs and lows. Use them.** −6.95 pp on the false-break
  rate, replicating in both halves, is the strongest level result here.
- **Do not pay attention to Fibonacci retracements.** They are a smooth curve
  read at five arbitrary points; z ≤ 0.95 and p ≥ 0.35 on every statistic
  tested. Drawing them is harmless; believing them is not.
- **Do not run a fixed take-profit / stop bracket.** Thirty cells, none positive
  in both halves, best +2.4%/yr against an index at +12.7%.
- **If you use a stop anyway, use it as risk control and price it as such.**
  A15 already measured that frontier: a hard stop takes P(−50%) to near zero and
  costs 6–8 points of median return. That is a purchase, not an edge, and it is
  a legitimate thing to buy.
- **Place levels from structure, take odds from measurement.** The honest
  combination is a target and a stop placed at real swing levels, with H32's
  first-passage laws saying how likely each is to be reached and which is likely
  to come first. That is what the chart now prints, and it makes no claim that
  the bracket earns money.
