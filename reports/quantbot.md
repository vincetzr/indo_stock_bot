# H50 — the goal: >+4% per trade, 80%+ of the time. Built the way a desk builds it.

*2026-08-29. `scripts/quantbot.py`. Pre-registration Q0–Q6 in the module
docstring, written before any model was fit. 891 names, 2,848,482 bars,
2000–2026; the tradeable universe is 355 names screened **point-in-time** at
Rp5bn/day trailing median turnover and Rp500 minimum price. Net of 0.56% fees
plus each name's own fraksi-harga half-spread. The 24-month holdout was spent at
H16; everything here is in-sample except the walk-forward folds, which are
genuinely out of sample.*

---

## 0. The answer, in three lines

**Both halves of the target are individually reachable. Neither is reachable
together. And every configuration that reaches either one compounds negatively
or below the index.**

| target | reached? | where | what it costs |
|---|---|---|---|
| mean per trade ≥ **+4%** | **YES** | +5.77%, no take-profit, 2σ stop, 252-session clock | positive rate 47.1%, compounds at **−5.3%/yr** |
| positive rate ≥ **80%** | **YES** | 80.3%, tight target + far stop + the model | mean **−0.40%**, compounds at **−9.1%/yr** |
| **both at once** | **NO** | 0 of 130 cells tested | — |

---

## 1. Why this is different from the 49 studies before it

Every previous hypothesis here ranked names on a score and held for a fixed
horizon. This is the standard institutional construction, and none of its three
pieces existed in this repo:

**Triple-barrier labelling.** A trade is labelled by which of three barriers it
touches first — profit target, stop, or clock — using the actual intraday
**high and low**. Every earlier study labelled on a fixed-horizon *close*
return, which is a different and easier question than "does my order fill".

**Volatility-scaled barriers.** A fixed +10% target is a routine day on one name
and a once-a-year event on another. Barriers sit at multiples of the name's own
trailing 60-day volatility scaled to the horizon, so "1.5σ" means the same thing
across the cross-section.

**Meta-labelling.** A primary rule decides the side (long, when the trend filter
is live); a secondary gradient-boosted classifier predicts whether *that
particular signal* will hit its target first, and the book takes only the
confident ones. This is the technique that exists specifically to raise
**precision** — the win rate the goal asks about — and it is the honest way to
attack the target, as opposed to moving the barriers.

36 features plus 6 cross-sectional ranks, purged **and** embargoed walk-forward
by calendar year, a label-shuffled null, a random-selection control at matched
slot count, a half-split, and a predicted-null feature.

---

## 2. The arithmetic the goal runs into — measured, not asserted

Per trade: `E[r] = p·W − (1−p)·L − c`. Under no edge, with barriers at (+a, −b),
the price is a martingale and optional stopping fixes `p = b/(a+b)` with
`E[r] = −c`. **Any win rate is purchasable — target near, stop far — and buying
it changes nothing**, because W and L move against p by exactly enough to cancel.

Q0 tests that the labeller reproduces this on a market with a known answer:

| barrier width | +tp | −sl | theory p | **measured p** | z | mean (symmetric fill) | z |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.050 | 0.050 | 0.500 | **0.496** | 1.40 | −0.0004 | 1.40 |
| 0.05 | 0.025 | 0.100 | 0.800 | **0.793** | 2.70 | −0.0009 | 2.70 |
| 0.10 | 0.100 | 0.100 | 0.500 | **0.493** | 1.13 | −0.0013 | 1.13 |
| 0.10 | 0.050 | 0.200 | 0.800 | **0.794** | 1.19 | −0.0015 | 1.19 |
| 0.20 | 0.100 | 0.400 | 0.800 | **0.798** | 0.17 | −0.0012 | 0.24 |

**Q0 PASSES** — every cell within 2.7 se, scored against *effective* n rather
than row count, because trades opened on consecutive bars overlap almost
completely and 96,000 rows are nowhere near 96,000 independent draws.

Look at row two: at a target of +2.5% and a stop at −10% you get a **79.3% win
rate and an expectation of −0.0009**. That is the whole constraint, in one line,
on data where the answer is known.

### Three errors in this check before it passed — all in the test, not the thing tested

The first version printed **"instrument FAILS"** and it was wrong three times over.

1. **Log increments of mean zero make the *price* drift up** by
   `exp(σ²T/2)` = +5.2% over 252 bars. The labeller measured that drift
   correctly and was blamed for it. Fix: drift the logs by −σ²/2.
2. **`b/(a+b)` is a price-space formula.** Symmetric price barriers are
   *asymmetric* in log space (+0.276 against −0.382 at these widths), so the
   formula was being applied to the wrong process.
3. **Feeding `close=high=low`** forced every touch to be detected a day late and
   a whole daily move past the level. Simulating 20 intraday steps gives a true
   extreme and the overshoot collapses.

The third is a *simulation* artefact, and it is shown to be one by convergence
rather than by loosening a tolerance — at the tightest geometry the error runs
**0.0422 → 0.0330 → 0.0135 → 0.0114 → 0.0033** as the intraday path is refined
1 → 5 → 20 → 80 → 320 steps. Real bars carry the true intraday extreme, so on
IDX the trigger is exact and this residual does not exist at all.

*A fourth attempt shrank the daily step instead, and the error got **worse**
(0.014 → 0.038) because shrinking the step at a fixed horizon also shrinks total
volatility, so fixed barriers stop being reachable and timeouts appear — and a
timeout breaks the formula outright.*

**An instrument check is worth building only if you are willing to believe it
when it fails.** It failed four times and every failure was mine.

---

## 3. The frontier — 120 cells, swept over horizon as well as geometry

Fixing the horizon is the error A20 names as having project-wide blast radius,
so it is a swept dimension here: 21, 63, 126 and 252 sessions.

**The mean end:**

| cell | hor | arm | positive | **mean** | median | **ann** | bars | yrs ≥4% |
|---|---|---|---|---|---|---|---|---|
| NO tp / sl 2σ | 252 | trend | 47.1% | **+5.77%** | −2.45% | **−5.3%** | 249 | 60% |
| NO tp / sl 2σ | 252 | all | 46.5% | +5.47% | −2.96% | −5.6% | 248 | 65% |
| tp 2σ / sl 2σ | 252 | trend | 48.1% | +4.95% | −1.65% | −5.1% | 233 | 56% |
| NO tp / sl 1σ | 252 | trend | 45.1% | +4.46% | −4.76% | −6.4% | 222 | 56% |
| NO tp / sl 0.5σ | 252 | trend | 35.2% | +3.76% | −14.27% | −5.3% | 159 | 52% |

**The win-rate end:**

| cell | hor | arm | **positive** | mean | median | **ann** |
|---|---|---|---|---|---|---|
| tp 0.25σ / sl 2σ | 252 | all | **77.7%** | −0.44% | +7.16% | −10.7% |
| tp 0.25σ / sl 2σ | 126 | all | 77.5% | −1.07% | +4.80% | −16.0% |
| tp 0.25σ / sl 1σ | 252 | all | 75.2% | −1.30% | +7.05% | −14.1% |

**Q1 CONFIRMED** — 5 of 120 cells reach a mean of +4% or better, exactly where
predicted: no take-profit (right tail intact), wide stop, long clock.

**Q2 FAILED at base rates.** The best positive rate anywhere is **77.7%**, not
80%. It fails for an interesting reason: gross of cost a tight target clears 80%
easily, and the **1.4% toll alone** pushes enough just-positive trades to
just-negative to keep it under.

**Q3 CONFIRMED — 0 of 120 clear both.**

**Q6 — reading B fails, but narrowly.** Of the five cells at mean ≥ +4%, the best
share of calendar years also at ≥ +4% is **65%**, against the 80% target.

### The column that decides it

**`ann` is negative in every one of the 120 cells. Best −5.1%.**

A **+5.77% average trade** that runs 249 sessions with a **median of −2.45%**
compounds at **−5.3% a year**. The arithmetic mean and the growth rate of an
account disagree *in sign*. A18 established that an equal-weighted holder is
paid the mean; that is a statement about holding many names *at once*. A bot
running positions sequentially is paid the mean log, and here they point
opposite ways.

---

## 4. Q4/Q5 — meta-labelling, and it does work

| geometry | selection | arm | n | positive | target-first | mean | median | ann |
|---|---|---|---|---|---|---|---|---|
| win-rate corner | all signals | real | 157,176 | 77.1% | 76.2% | −0.63% | +6.97% | −10.3% |
| win-rate corner | model top 20% | **real** | 34,119 | **80.3%** | **79.7%** | −0.40% | +5.81% | −9.1% |
| win-rate corner | model top 20% | NULL | 34,119 | 76.9% | **66.2%** | −0.46% | +6.97% | −9.8% |
| balanced | all signals | real | 157,176 | 52.2% | 35.8% | +2.57% | +2.44% | −5.9% |
| balanced | model top 20% | **real** | 34,119 | **57.5%** | **42.1%** | **+3.32%** | +8.13% | **−3.5%** |
| balanced | model top 20% | NULL | 34,119 | 51.8% | 32.0% | +2.42% | +2.34% | −5.4% |

**Q4 — the model is real and it clears your 80%.** At the win-rate corner the
top model quintile reaches **80.3% positive**. On the target-first label it
reaches **79.7% against a label-shuffled null of 66.2%** — a thirteen-point lift
that the null does not produce, so this is not the pipeline manufacturing its
own signal. At the balanced geometry it lifts the mean from +2.57% to +3.32% and
the median from +2.44% to +8.13%, again clearly above its null.

**Q5 — the predicted null did not fire.** Good: the machinery is clean.

**And it still does not get you there.** At 80.3% positive the mean is
**−0.40%** and the account compounds at **−9.1%/yr**. **0 of 10 modelled cells
clear both halves of the target**, and the best `ann` in the entire modelled
table is **−3.5%**.

*One arm was degenerate and is reported as such: with no take-profit there is no
target to hit, so the label "hit target first" is identically zero and the
classifier had nothing to learn — it returned the unranked sample and printed a
row that looked like a model result. The meta-label for that arm is now "did
this trade end positive", which is the question a book actually asks.*

---

## 5. The book — because a frontier map is not a bot

Eight to thirty-two slots, each taking the highest-ranked candidate it is not
already holding and running it to its own barrier exit. Geometry: no
take-profit, 2σ stop, 252-session clock, trend filter. 157,176 out-of-sample
signals, 2007-01 → 2025-08. **Index over the same span on a total-return basis
(the measured 1.77% yield added back, per A19): +9.92%/yr.**

The first table read +12.88% at 8 slots, −0.38% at 4 and +4.37% at 16. **That
non-monotonicity is the tell**: a genuine selection edge cannot be strong at 8
slots, absent at 16, and strong again at 32. So the luck spread was measured —
30 random-selection draws at each slot count, identical machinery, identical
dates:

| slots | model | random mean | random sd | model percentile | reads as |
|---|---|---|---|---|---|
| 4 | −0.38% | +3.14% | 5.94% | 30% | noise |
| 8 | +12.88% | +5.40% | 4.93% | 97% | signal |
| 16 | +4.37% | +6.01% | 3.31% | 40% | noise |
| 32 | +11.51% | +4.92% | 2.66% | 100% | signal |

Two above the band, two below, non-monotone. So the half-split, which is the
only replication test this repo trusts:

| slots | half | model | random | percentile | index | both halves? |
|---|---|---|---|---|---|---|
| 4 | early / late | +2.99% / +28.15% | +7.78% / +1.78% | 23% / 100% | +13.39% / +5.29% | **NO** |
| 8 | early / late | +11.32% / +19.52% | +10.78% / +2.83% | 60% / 100% | | **YES** |
| 16 | early / late | +8.29% / +12.10% | +10.46% / +4.06% | 30% / 97% | | **NO** |
| 32 | early / late | +12.69% / +9.69% | +7.93% / +4.77% | 90% / 97% | | **YES** |

**The model beats random selection in both halves at 2 of 4 slot counts. It
beats the index in both halves at 0 of 4.**

And the pattern is diagnostic: in the **late** half the model beats random at
every single slot count (100th, 100th, 97th, 97th percentile); in the **early**
half only at 32 slots. The mundane explanation is the expanding window — early
folds train on less data — and it fits better than a regime story.

Note also the index columns: **+13.39% early, +5.29% late.** Every book beats
the index late and none beats it early. That is the index's own regime, not the
model's skill.

---

## 6. What I would actually hand you

**The selection carries information.** The meta-model beats its label-shuffled
null by thirteen points of precision, and beats random selection in both halves
at two slot counts. That is more than most things in this repo have managed.

**It does not turn into money.** Zero of 130 barrier cells clear your joint
target; every one of the 120 base cells compounds negatively; the best modelled
cell compounds at −3.5%; and the book never beats the index in both halves.

**The specific trap to avoid.** You can have your 80% win rate tomorrow — set a
target 2.5% away and a stop 10% away and 79% of your trades will be green. Q0
measures the expectation of exactly that trade at **−0.0009 before costs**, and
−1.4% after. The win rate is not a property of your skill, it is a property of
where you put the two lines, and it is free to buy and worth nothing.

**Where the remaining honest upside is**, and it is not in this machine: the
binding constraint from nine directions is a 1.4% round trip against effects
worth tenths of a percent. Nothing in the price/volume feature space closes that
gap. What has never been tested at panel scale here is **non-price
fundamentals** — earnings, book value, debt — and `data/cache/fundamentals`
holds 59 names of 725. That is the instrument A25 named as the next one and it
is still unbuilt.

---

## 7. Trials and standing caveats

Q0–Q6 is 7 registered tests; the 120-cell frontier and the 10-cell model table
are a **frontier map, not 130 hypotheses**, and the best cell of a sweep is a
maximum rather than a measurement. **Trials after H50: 302.** Bonferroni bar
α = 0.05/302 = **0.00017**. No positive claim is made against that bar: Q1, Q2
and Q4 report *reachability*, and Q3, Q6 and the book are negative.

The holdout was spent at H16. The cost model is fees plus a fraksi-harga
half-spread and contains **no impact, suspension or auto-rejection term** —
A23 measured all three biting, and all three run against the holder. Q0's stop
fills at the *close* of the breaching bar rather than at the level, which is
H35's finding and is the one place this study is deliberately pessimistic.
