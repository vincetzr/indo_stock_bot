# H23 — the horizon, and the first thing in this project that beats the index

*31,417 monthly name-cohorts, 725 names, 2000-05 → 2024-08, pre-holdout only.
Code: `scripts/horizon_sweep.py`. Tests: `tests/test_horizon_sweep.py` (11).
Raw: `reports/horizon.txt`.*

**The question: can 8 of 10 names double, and if not, what is the ceiling?**

Every P(2x) in this repo — H13, H16, H18, H21 — is measured at **252 sessions**,
because that is the horizon H16 happened to choose. A doubling rate is mostly a
function of how long you wait, and A9 named "a horizon long enough that turnover
stops mattering" as untested and then never tested it. So the question had only
ever been asked at one year.

Two definitions of a hit, and the gap between them is the point:

- **touched 2x** — the path reached 2x at some point. What a take-profit order
  captures. H16 measured a mean *peak* of +102.2% against a realised +15.1%.
- **ended 2x** — buy-and-hold. All a terminal-return study can see.

---

## 1. The ceiling, with survivorship handled

Deaths valued at last traded price. `IHSG 2x` is the tide: a hit rate near it is
the index in a costume.

| horizon | n | eff n | touched 2x | ended 2x | profit | P(−50%) | median | IHSG 2x |
|---|---|---|---|---|---|---|---|---|
| 1y | 15,771 | 1,314 | 9.5% | 5.0% | 47.3% | 8.5% | −2.2% | 0.8% |
| 2y | 15,655 | 652 | 18.9% | 10.1% | 46.8% | 14.6% | −3.7% | 6.7% |
| 3y | 14,057 | 390 | 27.0% | 14.4% | 46.8% | 19.1% | −5.2% | 11.1% |
| 5y | 10,934 | 182 | 39.0% | 22.2% | 49.8% | −0.3% | −0.3% | 28.8% |
| 7.5y | 8,689 | 97 | 46.7% | 26.7% | 50.0% | 27.9% | +0.0% | 40.3% |
| **10y** | 6,683 | **56** | **55.5%** | 32.3% | 52.3% | 28.6% | +6.7% | 48.6% |

**80% is not reached at any horizon the data can measure.** At 10 years the
unconditional touch rate is 55.5%, so 8 of 10 would need a **1.44× lift** — and
that is the smallest gap in the table. At one year it needs 8.4×, against the
3.24× that is the largest lift ever measured in this project.

**Two bugs made the first version of this table say 73.8% at 7.5 years.**
`MU.PX` is a list of *cut edges* for the price bucket, not a `[min, max]` pair,
so `close >= PX[0] & close <= PX[1]` restricted the universe to sub-Rp50 names —
the penny board, 336 tickers. And eligibility was applied to *every* bar, so a
name dropping out of the universe mid-hold had its path cut there; eligibility
is a condition for **buying**, and once held the path is whatever the name does.
Requiring a full window on top of that discarded **91% of 7.5-year cohorts** and
measured the doubling rate of the names that lived.

## 2. Does any signal deliver the 1.44×?

The one-year lift is not transferable and has to be measured where it would be
used. Top and bottom decile of every registered feature, `squeeze` kept as
H13's predicted-null control.

**At 3 years** (base 27.0%, 80% needs 2.97×) the best is `mom12_1` top at
**1.31×** — nowhere near, and its median is −15.7%.

**At 10 years** (base 55.5%, 80% needs 1.44×):

| feature | side | n | touched 2x | lift | P(−50%) | median |
|---|---|---|---|---|---|---|
| amihud60 | bottom | 561 | 71.8% | 1.29 | 16.4% | +174.4% |
| **log_turnover** | **top** | 789 | **69.1%** | **1.24** | **13.7%** | **+174.7%** |
| lowvol | top | 787 | 62.6% | 1.13 | 23.3% | +49.0% |
| vol60 | bottom | 562 | 61.7% | 1.11 | 26.5% | +32.0% |

**Nothing reaches 1.44×.** The best is 1.29×, giving ~7.2 of 10.

**And the direction inverts everything the one-year studies found.** At one
year, liquid names were the worst cell in every table (A19: the liquid tercile
trailed the index by 9.5%). At ten years the *most liquid* names are the best on
every axis at once — highest touch rate, **half** the loss rate, and a median
that nearly trebles. The top two rows are the same axis said twice.

## 3. The one candidate, tested properly

`log_turnover` top decile at 10y is the maximum of ~22 cells, so it gets the
treatment rather than the headline. It is preferred over the marginally higher
`amihud60` because they are the same axis and this one has a prior mechanism
already established here — A19 found the cap-weighted index beat equal-weighted
baskets *because a handful of mega-caps carried it*. **This is that finding from
the other side, not an independent discovery.**

The null permutes whole **(ticker, year) blocks**, not rows: a name contributes
~12 near-identical monthly cohorts a year, and a row shuffle leaves the null far
too tight (A17 records that inflating a z to −8.7 before it became a headline).

| | |
|---|---|
| base touch rate | 55.5% |
| liquid decile | **69.1%** (n = 789), lift 1.24× |
| permutation null | 56.6%, sd 4.6%, p95 65.0% |
| **z** | **+2.70**, empirical p = 0.000 on 200 draws |

| half | base | liquid decile | lift |
|---|---|---|---|
| early | 70.6% | 76.1% | 1.08 |
| late | 40.3% | 61.3% | 1.52 |

**Positive in both halves.** Note the base rate itself is 70.6% early against
40.3% late — the regime does more work than the selection.

**It does not clear this project's Bonferroni bar.** z = +2.70 is p ≈ 0.0035
one-sided; the bar after 70 trials is 0.0007, and 200 draws cannot resolve below
0.005 anyway. Suggestive, not established.

## 4. And the comparison that has killed everything else

6,332 matched 10-year windows. Index put on a total-return basis with the
measured top-decile yield of 1.77% (H21), which compounds to 1.192× over ten
years.

| | median | mean | P(2x end) | P(2x touch) |
|---|---|---|---|---|
| **liquid decile** | **+188.8%** | **+464.1%** | **62.9%** | **72.8%** |
| all liquid names | +12.8% | +200.8% | 33.7% | 57.9% |
| IHSG price | +75.0% | +172.7% | 36.4% | 45.8% |
| IHSG **total return** | +108.5% | +225.0% | 61.0% | 75.8% |

Paired per window, liquid decile minus index TR: **median +51.8%, mean +239.0%,
winning 57.7% of windows.** Half-split: **+32.9% early, +73.9% late — positive
in both.**

**This is the first construction in the entire project that beats the index.**
Every previous one — flow, broker identity, investor class, price/TA, the
multiplier entry, all 58 exit rules, all 9 timing rules — lost. The reason is
not subtle: at a one-year horizon you pay ~1.3% round trip every year, and at a
ten-year horizon you pay it once, which is 0.13% a year. **A9 flagged exactly
this and it was never tested.**

---

## What this licenses, and what it does not

**8 of 10 is not reachable.** The ceiling is 72.8% — about **7 of 10** — and it
takes a ten-year hold of the most liquid names to get there. There is no
horizon, and no signal in this project, that reaches 80%.

**7 of 10 IS reachable, and the recipe is boring:** buy the ~10 most liquid IDX
names, equal weight, hold ten years. Historically ~7.3 of 10 touched 2x, the
median name returned +188.8%, and P(−50%) was 13.7% against a 28.6% base.

**The honest limits, and they are severe:**

- **Effective n is 56 for the whole sample and ~6 for the decile.** A ten-year
  window over a twenty-four-year panel is about two independent observations per
  name. Overlapping cohorts produce rows, not information, and no permutation
  null can manufacture independent samples the panel does not contain.
- **Only 30 distinct names were ever in the top decile** — TLKM, ASII, BMRI,
  BBRI, BBCA, BUMI, BBNI, PGAS, SMGR, ADRO, UNTR, ANTM and eighteen others. The
  cross-section is a list, not a population.
- It **is** reassuring that BUMI is in it — a top-decile name that fell ~99% —
  so this is not a survivor list.
- **The base rate does more work than the selection**: 70.6% early against 40.3%
  late. A decade beginning in 2026 is not a draw from the same urn as one
  beginning in 2005.
- **The holdout is spent** (H16), so none of this is out of sample.
- **Ten years is the whole mechanism.** Sell at year three and every result in
  §1 says you are back in the regime where costs eat the effect.
