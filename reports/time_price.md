# H31/H32/H33 — the time-cycle premise, the price×time cone, and the flip rule

*891 names, 2,465,286 pre-holdout bars, 2000-03-30 → 2024-08-23. Index tests on
8,853 IHSG sessions from 1990. Code: `scripts/time_price.py`. Raw:
`reports/time_price*.csv`.*

The request was for an Astronacci-style read: technical analysis gives a
horizontal price target, a time method gives the date, and the two cross at
"this price on this date, or a narrow range around it". That is two separate
claims and they do not stand or fall together, so they are tested separately.

| | claim | result |
|---|---|---|
| **T1** | turning points recur on a schedule, so a future DATE is forecastable from past dates | **fails, three ways** |
| **T2** | from a defined state, the joint (how far, how long) distribution is tight enough to quote as a range with odds | **holds — and the band is wide** |

So the deliverable is a **cone**, not a cross: a price band and a date band with
a measured hit rate attached, which is what "exact date at exact price, maybe in
ranges" honestly reduces to once the cycle half is removed.

---

## 1. H31 — turning points do not recur on a schedule

ZigZag pivots at 5%, 10% and 20% reversal thresholds; the gaps between
consecutive pivots are the half-cycles a time method claims to project. The null
is a **circular moving-block resample of each name's own returns** — same
marginal distribution, same short-range volatility clustering, no cycle. 100
draws, 150 names, observed statistic computed on the same names the null uses.

### 1a. Pivot spacing is LESS regular than a random walk's, not more

| threshold | statistic | observed | null | z |
|---|---|---|---|---|
| 10% | median gap (sessions) | 12.0 | 14.1 | −9.36 |
| 10% | **coefficient of variation** | **2.246** | **1.340** | **+32.7** |
| 10% | AR(1) of log gap | 0.302 | 0.210 | +10.1 |
| 10% | R² forecasting the next gap from the last two | 0.145 | 0.062 | +17.5 |
| 20% | coefficient of variation | 1.888 | 1.308 | +18.6 |
| 20% | R² from the last two gaps | 0.251 | 0.150 | +7.9 |

**The CV result is the one that matters and it points the wrong way for the
claim.** A cycle makes spacing *more* regular; IDX pivot spacing is 68% more
dispersed than the scrambled control. Real prices turn sooner than the null
(12 sessions against 14) and far less predictably.

*A detector that cannot find a cycle proves nothing by not finding one, so it
was checked against a case with a known answer first.* On a clean sine of period
60 the detector returns a CV under 0.05 and a median gap of 30 — a half period,
since consecutive pivots alternate high and low. IDX's 2.25 is two orders of
magnitude away from that. **The first version of the detector failed this
check**: it tracked only the previous bar before the first leg was confirmed, so
it could not start until a *single* bar moved 10% and read zero turns in a pure
sine. It is fixed, the study is re-run on the fixed version, and the test that
caught it is in `tests/test_time_price.py`.

### 1b. The memory that IS there is volatility clustering, not a period

The R² of 0.143 is well above its null and it would be easy to sell as a cycle.
It is not one. Lengthen the bootstrap block — which preserves longer and longer
volatility regimes — and the excess evaporates:

| block | observed R² | null R² | excess | z |
|---|---|---|---|---|
| 5 | 0.119 | 0.028 | 0.091 | +29.6 |
| 21 | 0.119 | 0.049 | 0.070 | +15.7 |
| 63 | 0.119 | 0.076 | 0.042 | +6.4 |
| **252** | 0.119 | **0.103** | **0.016** | **+1.64** |

**At a one-year block the null reproduces 87% of the observed interval memory
and the residual is not significant.** The mechanism is mundane: a quiet regime
produces several long gaps in a row and a violent one several short gaps. That
is autocorrelation in *volatility* leaking into the spacing of turns, and it
carries no information about *when* the next turn lands beyond "recently it has
been slow, so probably still slow".

### 1c. There is no fixed period in the IHSG either

1b tests "the next turn arrives a predictable interval after the last". The
other form of the claim is a fixed frequency — this market turns every N — and
that is a periodogram question. Detrended log IHSG, 8,853 sessions, periods from
10 to 1,260 sessions, against 500 block-bootstrap draws:

> strongest period **885 sessions (3.51 years)**, peak/mean power **140.2**,
> null **151.6 ± 56.6**, **p = 0.499**.

**The observed dominant period is weaker than the median scrambled control.** A
random walk with the same volatility clustering produces a more convincing
"cycle" half the time.

### 1d. Nor is there a calendar

Pivot share by month divided by the panel's own share of trading days, so an
unbalanced panel cannot masquerade as seasonality: the whole range is **0.865
(February) to 1.097 (May)**. Nothing here supports an anniversary method, and
the one genuine Indonesian candidate — Ramadan and Idul Fitri liquidity — moves
about 11 days a year against the Gregorian calendar and so cannot produce a
fixed date effect in the first place.

### What P1 got right, and what it got wrong

P1 predicted the interval R² would be "under 0.05". It is **0.143**, so the
registered threshold was wrong and is logged as wrong. The prediction's
substance — that no useful date forecast survives a proper null — holds, and 1b
is why: the R² is real, it is volatility clustering, and it buys almost nothing.

**The practitioner's version of that sentence, which is the only one that
matters:** the 50% confidence band for the date of the next turn is **±140% of
the median gap** knowing nothing, and **±125%** knowing the entire past sequence
of gaps for that name. Full knowledge of the cycle history narrows the date band
by **about a tenth of its width**. There is no time method here to build.

---

## 2. H32 — the cone: what IS forecastable about price and time

Every eligible bar (Rp1bn/day, tradeable) is an entry. For nine target levels
the study records the **first passage time** — sessions until the path first
touches entry×m within 252 — and windows that run off the end of a name's life
are dropped as censored rather than counted as misses. 623,126 entries.

### 2a. P(touch within one year)

| state | n | +5% | +10% | +20% | +50% | **2x** | −10% | −20% | −33% | **−50%** |
|---|---|---|---|---|---|---|---|---|---|---|
| base (any eligible bar) | 623,126 | 0.824 | 0.725 | 0.567 | 0.293 | **0.122** | 0.741 | 0.560 | 0.363 | **0.177** |
| price>50>100>200 | 195,561 | 0.837 | 0.743 | 0.598 | 0.320 | **0.136** | 0.703 | 0.509 | 0.314 | **0.141** |
| not stacked | 427,565 | 0.818 | 0.717 | 0.552 | 0.280 | 0.115 | 0.758 | 0.584 | 0.387 | 0.194 |
| hull55 rising | 295,288 | 0.834 | 0.741 | 0.587 | 0.309 | 0.129 | 0.718 | 0.529 | 0.334 | 0.158 |
| strength+calm (H26) | 47,770 | 0.833 | 0.732 | 0.559 | 0.241 | 0.083 | 0.627 | 0.424 | 0.221 | **0.078** |

**A TREND FILTER IS MOSTLY A RISK FILTER.** The EMA stack moves P(+20%) from
0.567 to 0.598 — a 5% relative gain — and P(−20%) from 0.560 to 0.509, a 9%
relative cut. On the 2x/half pair it is +11% up against −20% down. Fitted
properly in §3 the asymmetry is explicit: the stack multiplies upside odds by
**1.19** and downside odds by **0.71**. Whatever a trend state is worth, most of
it is on the side nobody puts in the headline.

*One number here contradicts nothing but looks like it does.* The base up/down
ratio at 2x reads **0.69**, against H26's skew of 1.33. Different denominators:
H26 measured P(**end** ≤ half), this measures P(**touch** half. A name that
falls 50% and recovers scores as a halving here and not there. Touch-versus-end
is the difference between what a stop-loss experiences and what a buy-and-hold
holder experiences, and both are worth knowing.

### 2b. Sessions to touch, given it touches (q1 / median / q3)

| state | +10% | +20% | +50% | 2x | −20% | −50% |
|---|---|---|---|---|---|---|
| base | 11/28/70 | 24/**54**/110 | 55/104/166 | 80/**134**/190 | 32/68/128 | 83/137/190 |
| price>50>100>200 | 11/27/69 | 24/**54**/111 | 53/101/167 | 76/**132**/192 | 35/72/132 | 93/148/198 |
| not stacked | 11/28/70 | 24/**54**/110 | 56/105/165 | 82/**134**/189 | 31/66/126 | 79/133/187 |
| hull55 rising | 10/26/67 | 23/**53**/108 | 54/101/163 | 79/**132**/189 | 35/72/131 | 89/141/193 |
| strength+calm | 16/36/82 | 35/**70**/128 | 78/131/183 | 110/**163**/209 | 58/102/162 | 92/155/210 |

**P4 CONFIRMED, AND IT IS THE CENTRAL RESULT.** Across the four trend states the
median time to +20% is **53, 54, 54 and 54 sessions**. Being in a confirmed
uptrend changes the odds of getting there and does not move the clock by one
session. Direction and timing are separate questions and the technical state
answers only the first.

The one state that does move the clock is the volatility screen, and it moves it
the boring way: calm names take **70** sessions to +20% instead of 54, and
**163** to double instead of 134. It buys its asymmetry with time.

**And the band is wide in every row.** For +20% the interquartile range is
24→110 sessions, a factor of **4.6**. A date can be quoted as a quarter-wide
band. It cannot be quoted as a week.

### 2c. Volatility is the clock

| vol60 decile | vol60 | P(+20%) | q1/med/q3 sessions | P(−20%) | P(2x) | med to 2x |
|---|---|---|---|---|---|---|
| 0 (calmest) | 0.0117 | 0.415 | 47/**89**/148 | 0.421 | 0.057 | 166 |
| 3 | 0.0218 | 0.547 | 31/**62**/118 | 0.545 | 0.083 | 161 |
| 6 | 0.0309 | 0.610 | 22/**49**/100 | 0.574 | 0.129 | 143 |
| 9 (wildest) | 0.0623 | 0.632 | 11/**30**/75 | 0.700 | 0.242 | 94 |

Monotone on every column. The median time to +20% falls by a factor of **three**
across the deciles while P(+20%) rises from 0.415 to 0.632 and P(−20%) rises
from 0.421 to 0.700 — A22's finding in first-passage form: volatility raises the
chance of touching *any* level and carries no direction.

**Volatility places the band; it does not narrow it.** Scaling the time axis by
vol60 changes the interquartile ratio for +20% from 4.62 to 4.33 and makes it
*worse* for 2x (2.38 → 2.50). So the right use of volatility is to centre the
date band on the right number, not to hope it tightens.

---

## 3. H32b — the cone as two closed-form laws

A 180-cell grid cannot be shipped to a chart: it answers only the nine targets
it was tabulated for, and 360 hand-copied constants in Pine is a transcription
error waiting to happen. Fitted in `d = |log(target/entry)|` and `σ = vol60`
(daily sample sd of simple returns), with terms
`[1, log d, (log d)², log σ, log d·log σ]` and a stack dummy on the probability:

```
log(sessions to target)                                R²    median   p90 err
  q1       3.4462  +1.5154  −0.1484  −0.3646  +0.2492  0.981   7.3%     25.2%
  median   4.2905  +1.2640  −0.1530  −0.2359  +0.2380  0.984   6.1%     17.2%
  q3       5.0526  +0.9246  −0.1383  −0.0827  +0.2134  0.985   4.6%     12.4%

logit P(touch within 252)                                    median   p90 err
  up    1.3567 −1.1390 −0.3009 +1.1850 +0.3405  +0.1750     1.71 pp   4.40 pp
  down  1.5117 −1.3308 −0.3158 +1.0515 +0.2545  −0.3388     1.75 pp   4.03 pp
```

**THE QUADRATIC TERM IS NOT DECORATION, AND LEAVING IT OUT NEARLY SHIPPED A
WRONG PANEL.** The first version of this fit was linear in `log d`. Pooled it
looked respectable — R² 0.95, median error 4.2 probability points — and it
under-predicted P(+20%) by **eleven points**, at the exact target the chart
ships as its default. Over a distance range from +5% to 2x the logit visibly
bends and a straight line sags through the middle, which is precisely where a
user sets the dial. **A pooled fit statistic does not tell you the fit is good
at the cell the reader will actually ask for; check that cell explicitly.**
`scripts/pine_cone_check.py` now does, and it is what caught this.

Two properties of the fitted shape are worth stating rather than smoothing over.

**It is not diffusion.** A driftless random walk reaches a log barrier `d` with
per-bar sd `σ` in time scaling as `(d/σ)²`. This does not, because the series
trends, its volatility clusters, and — the big one — the sample **conditions on
touching within 252 sessions**.

**That censoring is why the band is asymmetric.** The upper quartile's
sensitivity to distance is roughly half the lower quartile's, because the
252-session ceiling truncates the slow tail. **The top of the date band is
partly the horizon and not the market**, and any use beyond a year is
extrapolation the data does not license.

**Where the fit is worst, stated so it can be avoided.** Every one of the five
largest residuals sits in volatility decile 10 (σ ≈ 0.063), at the very edge of
the fitted domain — up to 10 probability points. The shipped panel clamps σ to
[0.0117, 0.0623] and prints *"vol outside fitted range"* when it does.

---

## 4. H33 — the flip rule, which is what the requested chart draws

Every flip the chart would put a label on, entered on the flip bar, held 60
sessions, charged 56 bps:

| rule | n | mean | median | win rate | **mean log** | early | late |
|---|---|---|---|---|---|---|---|
| price>50>100>200 turns on | 11,875 | +1.96% | −1.62% | 45.6% | **−0.0124** | +0.0069 | −0.0311 |
| **base — any eligible bar, no toll** | 611,023 | +2.27% | −1.02% | 46.2% | **−0.0140** | +0.0042 | −0.0295 |
| hull55 slope turns up | 14,066 | +2.17% | −1.68% | 45.5% | **−0.0148** | +0.0048 | −0.0310 |
| **hma21 crosses over hma55** | 16,736 | +1.80% | −1.96% | 45.1% | **−0.0191** | −0.0019 | −0.0336 |

**P5 confirmed. The dual-Hull cross — the exact signal on the requested chart —
is the WORST of the four**, 51 bps of mean log below simply owning an eligible
name, and it is the only rule negative in both halves. Nothing here compounds.

This is consistent with what the repo already had: `reports/hullut_*.csv` scored
the published Hull Suite + UT Bot on 84 IDX names over 240 grid configurations
and the best cell's median excess CAGR was **−6.1%**; it beat buy-and-hold on
26% of names, and out-of-sample CAGR trailed buy-and-hold in **all five**
walk-forward folds.

### The state, however, is not the cross — and H30's constants were wrong

| state (not entry) | n | mean log | early | late | vs base |
|---|---|---|---|---|---|
| all eligible bars (base) | 611,023 | −0.0140 | +0.0042 | −0.0295 | — |
| **price>50>100>200** | 191,927 | **+0.0127** | +0.0249 | −0.0019 | **+267 bps** |
| ordered, price below | 88,333 | −0.0197 | +0.0000 | −0.0368 | −57 bps |
| not aligned | 330,763 | −0.0279 | −0.0104 | −0.0402 | −139 bps |

**H30 PUBLISHED +0.0218 FOR THIS STATE AGAINST A BASE OF +0.0021 AND BOTH
NUMBERS ARE WITHDRAWN.** `scripts/ema_cross_idx.py` computed the forward return
as `px.shift(-60)/px` on a **date × ticker pivot**, so the step was 60 rows of
the *panel's union index* rather than 60 of the name's own bars. For any name
that does not trade every session that is a shorter window, biasing a negative
mean log toward zero. Recomputed within each ticker, base is **−0.0140** and the
stack is **+0.0127**. Diagnosed by reproducing both numbers with each method:
the pivot method returns 0.00214 and 0.02203, matching H30 to three decimals.

This is **A11's own defect — "rolling windows on a pivot are indexed by the
union of trading days; group by ticker, never roll on a pivot" — recommitted by
the same author eight appendices later**, in a study whose constants were then
shipped into a Pine script and handed to the user. Writing the rule down does
not make it operate; the only thing that catches it is recomputing the number a
second way.

**The finding survives the correction and is cleaner for it.** The stack beats
the base by +207 bps of mean log in the early half and **+276 bps in the late
half** — positive in both halves *relative to the base*, which is the comparison
that matters, and no longer resting on an absolute figure that was an artefact.
What is withdrawn is any claim that the state compounds positively on its own:
in the late half it is −0.0019, essentially flat.

---

## What this licenses, and what it does not

- **There is no time method here.** Three independent tests — pivot-spacing
  regularity, block-length discrimination, and a periodogram on the index — say
  the same thing, and the calendar says it a fourth time. Do not buy a
  date-projection method for this market on this evidence.
- **A price target with a probability and a date BAND is real and is
  computable.** +20% within a year from a stacked name: **59.8%**, half the time
  between session 24 and session 111, median 54. That is honest and useful, and
  it is not a forecast of anything on any particular day.
- **State the downside in the same breath.** The same entry has a **50.9%**
  chance of touching −20% first-passage. A target quoted without its matching
  downside odds is half a trade.
- **Do not trade the flips.** The cross the requested chart draws is measurably
  worse than doing nothing, in this study and in the repo's earlier Hull work.
  Draw it, look at it, and take the trade for a reason the label does not supply.
