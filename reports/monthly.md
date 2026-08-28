# H41 — Rp 50 juta through the Hull rule, month by month, against two controls

*777 names, 2,438,621 name-days, 293 months, 2000-03-30 → 2024-08-23. Ten
independent account draws per arm. Code: `scripts/monthly.py`. Raw:
`reports/monthly.txt`, `reports/monthly.csv`.*

The question was "if I start trading with 50 jt, what is the expected return
per month". A single number would have been a lie in three separate ways — the
distribution is too skewed for a mean to describe, a single backtest reports
its own luck as the strategy, and no number means anything without the
alternative it is being compared to. So this runs an actual book: Rp 50,000,000
across at most five equal-weighted positions, entering on the measured rule,
exiting on H40's best stop (a −25% trail from the running peak, 252-session
cap), paying 0.56% a round trip, cash earning nothing while it waits.

---

## 1. The account

| draw | mean | median | sd | % up | worst | best | CAGR |
|---|---|---|---|---|---|---|---|
| 1 | 1.35% | 0.33% | 7.47% | 54% | −17.9% | +40.6% | 13.85% |
| 2 | 1.54% | 0.89% | 7.00% | 59% | −18.4% | +29.2% | 16.82% |
| 3 | 1.50% | 0.46% | 7.19% | 55% | −16.7% | +55.8% | 16.20% |
| 4 | 1.24% | 0.68% | 6.35% | 58% | −18.5% | +25.7% | 13.30% |
| 5 | 1.38% | 1.11% | 7.68% | 58% | −30.2% | +39.3% | 13.94% |
| 6 | 1.05% | 0.87% | 7.12% | 57% | −22.0% | +31.2% | 10.09% |
| 7 | 1.32% | 0.97% | 7.17% | 57% | −25.2% | +27.4% | 13.60% |
| 8 | 1.46% | 0.50% | 7.35% | 57% | −21.4% | +34.0% | 15.38% |
| 9 | 1.73% | 0.72% | 7.98% | 59% | −29.0% | +36.1% | 18.41% |
| 10 | 1.70% | 0.77% | 7.67% | 57% | −24.7% | +37.1% | 18.44% |
| **RANDOM, same exit** | **1.44%** | 1.09% | 8.39% | 57% | −29.2% | +53.8% | **14.19%** |
| **IHSG buy and hold** | **1.04%** | 1.25% | 5.71% | 62% | −31.9% | +20.1% | **11.05%** |

On Rp 50,000,000, averaged across the ten draws:

| | |
|---|---|
| mean month | **+1.43%** = Rp +714,457 |
| **median month** | **+0.73%** = Rp +365,527 ← *the typical month* |
| months positive | 57% |
| month-to-month sd | **7.30%**, i.e. ±Rp 3,648,964 of ordinary swing |
| worst single month | **−30.2%** = **−Rp 15,120,393** |
| invested | 87% of the time |
| draw-to-draw CAGR spread | **+10.09% to +18.44%** |

**That 8.35-point spread is luck, and it is what a single backtest would have
reported as the answer.** Which five names a small book happens to hold is a
draw, not a strategy, and one run of this simulation would have returned any
number in that range with equal confidence.

---

## 2. The control that decides the study, and it is negative

A book of five IDX names with a trailing stop, rotating constantly and invested
87% of the time, **is an equal-weighted IDX portfolio with an exit rule.** A19
measured that an equal-weighted IDX basket behaves nothing like the cap-weighted
index, so beating the index proves nothing about the signal. The comparison that
means something is the same machine — same universe, same slots, same stop, same
costs — picking names **at random**.

| | CAGR | mean month |
|---|---|---|
| Hull-filtered | **+15.00%** | +1.43% |
| random, identical machine | **+14.19%** | +1.44% |
| IHSG buy and hold | +11.05% | +1.04% |

**The Hull filter is worth +0.81% a year, against a draw-to-draw luck spread of
8.35 points.** The mean month is *lower* than the random arm's. The account beats
the index and so does picking at random, so the outperformance belongs to
equal weighting and the trailing stop, not to the signal.

### And the half-split kills what is left

A18: *"a within-sample consistency statistic over correlated units reads as
overwhelming and says almost nothing about whether an effect replicates. Only
the half-split does."*

| | early | late |
|---|---|---|
| Hull-filtered | 20.71% | 9.80% |
| random, same exit | 22.05% | 7.25% |
| **edge** | **−1.34% ± 4.22%** | **+2.55% ± 6.34%** |

**Negative in the early half, positive in the late half, both inside their own
error bars.** Draws beating the random mean in *both* halves: **1 of 10, against
2.5 expected from chance alone.** The filter does worse than a coin flip on the
one test this repo trusts.

---

## 3. The power statement, kept separate from the effect statement

A19 recorded writing a power claim as an effect claim as its own error, so both
are stated here explicitly.

| question | months needed at t = 2 |
|---|---|
| is this account's mean month different from **zero**? | **104 months — 8.7 years** |
| is it different from **picking at random**? | **46,856 months — 3,905 years** |

The second number is not a joke about precision; it is the correct
interpretation of a +0.81%/yr edge against a 7.30% monthly standard deviation.
**No live track record of any realistic length can distinguish this rule from
random selection.** Anyone trading it for a year and judging by the outcome will
be reading noise, in whichever direction it happens to point.

---

## 4. The pre-registered predictions

| | prediction | result |
|---|---|---|
| **M1** | median near zero or negative, mean positive | **PARTLY FAILED.** The skew is there — mean +1.43% against a median +0.73%, and 57% of months positive. But the median is clearly positive, so "most months lose a little" is wrong. Most months make a little; the mean is dragged up by a few large ones. |
| **M2** | the account underperforms the index | **FAILED.** +15.00% against +11.05%. And the failure is not evidence for the rule: the random arm returns +14.19%, so what beat the index was equal weighting, not selection. |
| **M3** | monthly sd several times the mean | **CONFIRMED**, at 5.1×, and quantified into the 8.7-year figure above. |

M2 failing while the study's conclusion stays negative is worth noting on its
own: **a prediction can fail in the direction that flatters the rule and still
tell you nothing, if the control moved with it.**

---

## 5. Two bugs, both found by disagreement with an earlier study

**The entry filled on its own signal bar.** Both conditions are computed from
the close of bar *t*, and the first version bought at that same close — a price
only known once the bar was finished. H39 filled at *t+1* throughout. Two
studies of one rule disagreeing is the signal that one of them is wrong; the
fix was a one-bar shift and it barely moved the headline, which is its own
useful information about how little of the return sits in that one bar.

**A delisted name was held forever.** The panel deliberately contains names that
stop printing — a survivorship-free universe is the whole point of the spine —
and a position whose ticker disappeared was carried at its last price for the
rest of the backtest, blocking a slot and inflating the book. Now realised after
five missing sessions.

---

## What this licenses

- **The honest answer to "what do I make a month" is a distribution, not a
  number:** typically +0.7%, on average +1.4%, ±7.3% of ordinary swing, with a
  −30% month in the sample. On Rp 50m that is a typical +Rp 366k, an average
  +Rp 714k, and a −Rp 15.1m month that has happened.
- **None of it is attributable to the Hull signal.** +0.81%/yr against random
  selection, negative in the early half, 1 of 10 draws replicating against 2.5
  by chance.
- **Everything above is in-sample.** The reserved holdout was spent at H16.
- **At Rp 50m across five names, capacity is not the binding constraint** —
  Rp 10m a position against a Rp 1bn/day filter is small. A23's impact,
  suspension and auto-rejection costs, which are absent from every cost figure
  in this repo, would bite at client scale and do not bite here.
