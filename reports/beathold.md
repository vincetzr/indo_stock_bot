# H54 — beat buy-and-hold. Three bugs were mine, and the binding constraint was my own benchmark.

*2026-09-03. `scripts/bhbench.py`, `scripts/beathold.py`, `scripts/beathold_diag.py`.
Pre-registration D1–D4 in the `beathold.py` module docstring, written before any
arm was run. 21 arms × 6 rebalance calendars = 126 phase-verdicts. Cost = 0.56%
fee + half a fraksi-harga tick, quoted separately per A38. Index on a
total-return basis over each arm's own window.*

---

## 0. The answer, both ways

**Under the bar as I originally wrote it — beat the index, beat a never-touched
basket of the whole eligible universe, beat the strategy's own first basket held
untouched, and beat a random control, in both halves, in a majority of six
rebalance calendars — 0 of 21 arms pass. D1 confirmed.**

**Under a bar that demotes the third benchmark to a diagnostic, 4 of 21 pass at
4 of 6 calendars**, all in the strength-plus-calm family, at **+4.5% to +6.5% a
year over the index**. That change was made *after* seeing which benchmark was
binding, so those four are a lead and not a finding. §3 is the measurement that
motivates it; §6 is what it does and does not license.

What is not in doubt either way: **six of the twenty-one arms beat the index in
6 of 6 calendars**, and every strength or momentum arm beats a random basket
drawn from its own universe on its own calendar by 8 to 14 points a year.

---

## 1. This re-run exists because the previous one measured my harness

A fleet of strategy designs was run against the first version of `bhbench.py`
and three of them "passed". All three passes were properties of the harness.

*What is re-run here is the FAMILIES, not the fleet's scripts.* Each design is
re-implemented compactly in `beathold.py` so that all 21 arms share one
selector protocol, one cost model and one verdict — the fleet's scripts each
carried their own harness variations, and comparing across them would repeat the
same error this section is about. The originals are kept under
`scripts/strat_*.py` as the record of what was tried; their numbers are
superseded.

**CASH DRAG.** Every `continue` in `walk()` appended the unchanged equity, so a
mark the strategy could not act on earned it **zero** while all three
buy-and-hold benchmarks compounded through that window. Refusing to *buy* a
degenerate cross-section is right — it is what stops H52's one-name basket.
*Liquidating a book you already own* because the screen went quiet is not
something any holder would do, and it is not what the benchmark does.

**THE REBALANCE PHASE.** `marks` starts at `dates[offset]`, and offset 0 is the
panel's first bar for no reason but that it is first. All three passes were
properties of that start date. This is A20's lesson — a parameter fixed once by
convenience and inherited by every study — committed inside the harness written
to catch exactly that class of error.

**THE CONTROL'S CALENDAR.** `_verdict` passed `offset` to the strategy's walk
and not to the random control's, so at every phase but zero the control was
measured over a different window from the arm it exists to control for.

**And a fourth, found here rather than by the fleet.** Equalising the phases is
not enough: the window starts at the first mark that yields a tradeable basket,
and *that date moves with the calendar*. One strategy's six semiannual phases
began in **2005, 2007, 2007, 2008, 2009 and 2010**, spanning 15.9 to 21.2 years.
The phase that caught the 2005–07 boom was not replicating the one that did not,
so "4 of 6 phases" was counting six different experiments. `evaluate` now
scouts, then re-runs every phase over the **common** window. A19's error class,
surviving one fix and reappearing one level up.

---

## 2. D2 — what the drag was worth, and it failed in two of three clauses

`Bench(carry=False)` reinstates the bug on purpose, so its cost is measured
rather than asserted.

| | mean lift from the fix | n |
|---|---|---|
| quarterly arms | **+2.64%/yr** | 9 |
| semiannual arms | −0.08%/yr | 2 |
| annual arms | **+1.11%/yr** | 10 |

**D2a FAILED. The lift is not one-signed.** Three arms got *worse* when the bug
was fixed — `strength+calm 20, annual` −1.42%, `strength+calm 10, annual`
−1.19%, `strength+calm sticky, annual` −0.54%. Carrying the book through a
skipped window exposes it to whatever happened in that window, and sitting in
cash was accidentally *protective* where the market fell. So the drag was an
**unpriced cash position, not a systematic handicap** — and my earlier
"a handicap of 1.8 to 6.7 points a year" was a one-sided reading of a two-sided
error. The correction stands; the characterisation of it did not.

**D2b FAILED.** I predicted the lift would be largest where rebalances are
rarest, because skipped marks are longest there. It is largest at **quarterly**
(+2.64%) and smaller at annual (+1.11%). The reason is where the skips fall, not
how long they are: the quarterly arms skip in 2000–2005 while the universe is
below the 40-name floor, and a compounding gap that early costs the most.

**D2c held.** Spearman rank correlation between the two orderings is **+0.871** —
the drag mostly moved everything together, which is why the fleet's *relative*
conclusions largely survive even though its levels do not.

---

## 3. The binding constraint was my own benchmark, and it is a single draw

Across 126 phase-verdicts:

| gate | times failed |
|---|---|
| **beat its own picks, both halves** | **103 (82%)** |
| beat the index, both halves | 78 (62%) |
| beat the universe basket, both halves | 78 (62%) |
| beat its own picks | 75 (60%) |
| beat the index | 53 (42%) |
| beat the universe basket | 48 (38%) |
| beat the random control | **18 (14%)** |

`BH_PICKS` is **one ~10-name basket held for eighteen years**. Across six
calendars of the *same* rule it spans **0.49% to 15.82% a year** — a 15.3-point
spread produced by moving the start date a few weeks. `BH_UNIVERSE` does not
have this problem: it holds every eligible name on day one, 100–200 of them, so
it is a diversified benchmark rather than a draw.

**A gate whose sampling error exceeds every effect it is measuring is not
measuring.** It stays *reported* — a rule that loses to holding its own picks is
telling you the trading adds nothing, which is the whole ADRO question — but as
a diagnostic, not a gate. Both verdicts are printed on every run and
`tests/test_bhbench.py` asserts the strict one can never be true where the
weaker one is false, because CLAUDE.md §2 forbids replacing a criterion that
failed.

---

## 4. The table

Sorted on **excess over each arm's own index**, which is the only cross-row
comparable column: each row starts at its own first tradeable mark, so the index
column runs +6.21% to +11.31% over one panel and ranking on raw CAGR would rank
the decades.

| strategy | vs index | phase range | CAGR | index | picks | random | >idx | strict | divers |
|---|---|---|---|---|---|---|---|---|---|
| sticky tight buffer, quarterly | **+6.50%** | +1.97..+6.61 | 14.11% | 7.21% | 13.41% | 0.56% | **6/6** | 1/6 | **4/6** |
| sticky 20 names, quarterly | +4.81% | +1.36..+6.01 | 10.94% | 6.21% | 10.45% | −1.25% | **6/6** | 2/6 | **4/6** |
| sticky wide buffer, quarterly | +4.76% | +2.26..+9.00 | 12.98% | 8.47% | 3.29% | 2.75% | **6/6** | 2/6 | **4/6** |
| strength+calm 10, semiannual | +4.53% | −2.00..+9.12 | 13.25% | 8.91% | 12.04% | 8.81% | 5/6 | 1/6 | **4/6** |
| sticky no buffer (= hard screen) | +4.14% | +1.94..+7.57 | 11.81% | 7.21% | 13.41% | 0.56% | **6/6** | 1/6 | 3/6 |
| momentum top 10, quarterly | +4.00% | +1.57..+8.52 | 15.17% | 11.31% | 11.47% | 6.36% | **6/6** | 0/6 | 2/6 |
| strength+calm 10, quarterly | +3.76% | +1.61..+8.08 | 11.35% | 7.21% | 13.41% | 0.56% | **6/6** | 1/6 | 3/6 |
| strength+calm sticky, quarterly | +3.40% | +2.14..+7.29 | 11.80% | 8.47% | 3.29% | 2.75% | **6/6** | 1/6 | 2/6 |
| frozen everything, drift wts | +0.81% | −1.44..+1.26 | 9.31% | 7.62% | 6.71% | 4.76% | 5/6 | 3/6 | 3/6 |
| own everything, quarterly | −1.13% | −2.17..−0.69 | 9.59% | 11.31% | 10.48% | 6.36% | 0/6 | 0/6 | 0/6 |
| own everything, annual | −1.95% | −3.61..−1.26 | 6.78% | 7.62% | 6.71% | 4.76% | 0/6 | 0/6 | 0/6 |
| liquidity top 10, annual | −3.88% | −6.23..−2.76 | 5.04% | 7.62% | 8.50% | 4.76% | 0/6 | 0/6 | 0/6 |
| low vol top 10, quarterly | −4.20% | −5.49..−3.80 | 6.51% | 10.32% | 7.79% | 4.39% | 0/6 | 0/6 | 0/6 |
| liquidity top 20, annual | −5.54% | −8.81..−3.99 | 2.98% | 9.53% | 10.01% | 4.62% | 0/6 | 0/6 | 0/6 |

*(Full 21-row table in `reports/beathold.txt`, with each phase's verdict and
which benchmark it lost to.)*

**The buffer sweep is not monotone and therefore is not a validated parameter.**
Tight (keep top 20% of `hi52`, bottom 60% of vol) reads +6.50% at 56% turnover;
wide (top 60% / bottom 90%) reads +4.76% at **15%** turnover; no buffer at all
reads +4.14% at 75% turnover. Removing the buffer is clearly worse than either
buffered version, which is the family's actual claim — churn driven by rank
noise around a cut is not informative. Which buffer width is best is not
resolved by three points that do not order with turnover, and the range across
the family (+3.40% to +6.50%) is about as wide as the effect itself.

---

## 5. D3 and D4, on one explicitly fixed window

The table above equalises the six phases *of each strategy*. It does not
equalise *across* strategies. So D3 and D4 are run separately on one window,
2008-02-04 → 2025-07-17, chosen as the narrowest every arm can occupy.

| arm | CAGR | index | vs index | cost/yr | turnover |
|---|---|---|---|---|---|
| D3 own everything, quarterly | 4.71% | 7.42% | −2.79% | 0.54% | 17% |
| D3 own everything, annual | 6.42% | 7.62% | −2.32% | 0.25% | 35% |
| D4 re-select names + reset weights | 6.42% | 7.62% | −2.32% | 0.25% | 35% |
| D4 freeze names + reset weights | 8.14% | 9.60% | −1.13% | 0.11% | 15% |
| D4 freeze names + **drift** weights | 9.66% | 9.60% | **+0.51%** | 0.11% | 14% |
| D4 re-select names + drift weights | 7.42% | 9.60% | −1.75% | 0.25% | 32% |
| strength+calm, hard screen | 10.65% | 9.53% | +0.76% | 2.12% | 77% |
| strength+calm, tight buffer | 12.03% | 9.53% | **+3.18%** | 1.53% | 55% |

**D3 — the predicted null — is satisfied.** Quarterly and annual own-everything
differ by 0.47 points of excess against a toll gap of 0.29, and their phase
ranges overlap. The harness is not manufacturing or destroying return in
proportion to how often it is called, so the table above is readable.

**D4 FAILED, and it failed in the direction opposite to the claim it was testing.**
The fleet attributed the destruction of the equal-weight arm to *name churn*
from the eligibility filter, with weight rebalancing exonerated. Window-matched,
the 2×2 says both matter and **weight drift is the larger term**:

- freezing the names, holding weights fixed: **+1.19 points**
- letting weights drift, given frozen names: **+1.64 points**
- the two interact — drift is worth +0.57 when names churn and +1.64 when they
  do not
- doing neither → doing both: **−2.32% → +0.51%, +2.83 points**

**The cheapest way to beat the index measured anywhere in this project is to
stop trading.** Own the eligible universe, never re-select, never reset to
equal, and the arm goes from 2.3 points behind the index to half a point ahead
of it, at 0.11% a year in cost.

---

## 6. What this establishes, and what it does not

**Established.** The selection is real. Every strength or momentum arm beats a
random basket drawn from its own universe on its own calendar, in 108 of 126
phase-verdicts, by 8 to 14 points a year. That has never been in doubt in this
repo and it is not the question.

**Established.** Beating the index on full-period CAGR replicates across the
calendar: six arms do it in 6 of 6, at a median excess of +3.4% to +6.5% a year.
The previous four appendices' "no selection rule beats the index" was measured
on rules run at offset 0 through a harness with a cash drag in it, and that
statement does not survive the correction.

**Not established, and this is the load-bearing sentence.** Nothing clears the
bar as it was written. The four arms that clear the weaker bar do so at 4 of 6
calendars — and with six *heavily overlapping* windows, 4 of 6 is not a
significance statement about anything. The phases are a robustness check, not
independent trials; the effective number of independent observations over one
18-year Indonesian macro history is small and no resampling manufactures more.
**The holdout was spent at H16**, so every number here is in-sample on data
these rules have already been fitted through.

**And the cost model is still the small-order one.** A23 measured that impact,
suspension and auto-rejection are in no number in this project, and this table
inherits that. Two of today's ten picks trade under Rp 2bn a day.

---

## 7. Today's basket from the leading rule

Sticky, tight buffer, as of 2026-08-28 — 242 eligible names, 10 picked:

| ticker | close | % of 52w high | 60d vol | Rp bn/day |
|---|---|---|---|---|
| ASGR | 2,090 | 100.0% | 2.6% | 1.5 |
| AUTO | 3,400 | 100.0% | 1.8% | 4.4 |
| DMAS | 199 | 100.0% | 2.8% | 11.3 |
| BSSR | 4,540 | 100.0% | 1.1% | 2.0 |
| MPMX | 1,020 | 100.0% | 1.0% | 3.1 |
| TAPG | 2,030 | 99.0% | 2.7% | 33.9 |
| SRTG | 1,935 | 99.0% | 2.7% | 6.8 |
| ADRO | 2,670 | 98.9% | 2.0% | 102.6 |
| BFIN | 920 | 98.4% | 3.2% | 14.4 |
| ERAA | 488 | 98.4% | 2.7% | 12.2 |

Reproduce with `python scripts/beathold_diag.py`. This is a research output, not
advice, and A23 applies in full: the holdout is spent, there is no live track
record, and a suitability judgement about a specific client is not something
this repo can supply.

---

## 8. Trials

D1–D4 is 4 registered tests. The 21-arm table is a map; the sticky buffer sweep
is 4 cells registered as a sweep with every cell printed, because the family's
claim is about the shape of the curve rather than its maximum. **Counting the
sweep: 8. Trials after H54: 323** (H53's 5 registered tests are logged
alongside). Bonferroni bar α = 0.05/323 = **0.00015**. Nothing here is claimed
against it: the strict result is negative, and the positive result under the
weaker bar rests on a 4-of-6 count over overlapping windows, which carries no
p-value at all.
