# §12 memo — is the losing cohort the *same* cohort next period?

**Date:** 2026-08-23
**Verdict: no. Broker-code margin rank does not persist, on either store.**
Reproduce with `python3 scripts/persistence.py` and
`python3 scripts/persistence_diag.py`.

---

## 1. Why this test and not another

Phase 2b (`reports/phase2b_cohort_pnl.md`) measured the median cohort round
trip at **−25 to −32 bps** and found a shuffled label losing about the same. It
closed by saying that number is not yet interesting, and §12 says why:

> *"That retail cohorts lose persistently while institutions gain is among the
> most robust results in the market-microstructure literature ... and it is
> durable precisely because the losing cohort continuously regenerates."*

The operative word is **persistently**. §12's whole strategic argument — that
taking the other side of persistent losers beats following winners — rests on
the losing cohort being *identifiable*, which means the same codes have to keep
losing. One period's ranking is a snapshot. Persistence is the claim, and it
was the claim Phase 2b left untested.

So: rank brokers by margin in one period, rank them again in the next,
correlate the ranks. High positive means last period's ranking tells you who to
fade. Near zero means it does not.

---

## 2. The answer

**Every statistic, on both stores, sits inside its own permutation null.**

### Track B — fortnightly, 89 codes, 217 tickers, 378,689 broker-windows, 2014–2026

| statistic | observed | null mean | null 95% band | one-sided p |
|---|---|---|---|---|
| split-half, all rows | −0.185 | +0.000 | [−0.222, +0.229] | 0.930 |
| split-half, two-sided | +0.076 | −0.001 | [−0.224, +0.276] | 0.239 |
| year-over-year, all rows | **+0.032** | −0.004 | [−0.079, +0.072] | 0.144 |
| year-over-year, two-sided | **+0.045** | −0.004 | [−0.089, +0.083] | 0.119 |

### Track A — daily, 27 codes, 9 names, 628 round-trip episodes, 1.5 years

| statistic | observed | null mean | null 95% band | one-sided p |
|---|---|---|---|---|
| split-half | −0.203 | **−0.207** | [−0.504, +0.108] | 0.498 |
| quarter-over-quarter | +0.223 | **+0.122** | [−0.070, +0.335] | 0.279 |

Nothing clears 0.05 even before correcting for the trial count, and with 22
prior trials plus 6 here the Bonferroni bar is **p < 0.0018**.

---

## 3. The one result that did clear 0.05, and why it is not real

The first run reported year-over-year **+0.078 with p = 0.025** on all rows.
That was the only sub-0.05 figure produced anywhere in this test, and it does
not survive.

**It is one year pair.** The thirteen adjacent-year correlations are:

```
2013→2014  +0.632      2018→2019  +0.110      2023→2024  +0.218
2014→2015  +0.112      2019→2020  −0.081      2024→2025  −0.106
2015→2016  +0.012      2020→2021  −0.159      2025→2026  +0.009
2016→2017  +0.174      2021→2022  −0.069
2017→2018  +0.095      2022→2023  +0.071
```

`2013→2014 = +0.632` is more than three times the next largest. And 2013 is
where the store is thinnest by an enormous margin:

| year | rows | tickers | windows | brokers clearing the guard |
|---|---|---|---|---|
| 2013 | **262** | 14 | 15 | **13** |
| 2014 | 26,900 | 98 | 38 | 82 |

The panel collection starts in 2014; everything before it is exploratory probes
— **454 rows out of 379,143, or 0.1% of the data.** Dropping them removes that
pair and the statistic goes from **+0.078 (p 0.025) to +0.032 (p 0.144)**, and
the two subsamples, which had disagreed, come into agreement at +0.032 and
+0.045.

**This filter was chosen after seeing the result, and that is stated rather
than hidden.** It is justified on sample size — 13 brokers from 15 windows —
not on the answer it gives, and it is a guard that would have been set a priori
had coverage been checked before the statistic. Both versions are printed by
the script so the effect of the line is visible.

---

## 4. My explanation for it was wrong, and the diagnostic said so

The obvious suspect was the **top-10 censor**. IndoPremier ranks buyers and
sellers independently and publishes ten of each, so a code appearing only among
buyers has its sell side recorded as **zero** when the truth is an unknown
lower bound — its net is forced long by construction. Only **51.1%** of rows
are two-sided. If some codes are systematically the censored ones, "which code
gets truncated" would be a persistent broker attribute masquerading as skill.

The first half of that chain is emphatically true:

| | year-over-year rank correlation |
|---|---|
| censor share | **+0.801** (range +0.27 to +0.89) |
| margin | +0.078 |

Censoring is *ten times* more persistent than margin. But the chain needs a
second link, and it is absent:

| | within-year cross-sectional correlation |
|---|---|
| corr(censor share, margin_bps) | **−0.037**, range [−0.17, +0.09], 6/14 positive |

Censoring does not predict the measured margin. **The artefact explanation
fails**, and the sparse-2013 explanation in §3 is what is left.

---

## 5. What the null did that a single shuffle could not

H9 in this repo nearly shipped with a broken null, and a later single draw was
briefly over-read as systematic bias before a second seed contradicted it. So
this test used a **200-draw permutation distribution** through the identical
pipeline, guards included. It earned its keep immediately:

**Track A's null is not centred on zero.** Split-half null mean **−0.207**;
quarter-over-quarter null mean **+0.122**. With 21 brokers and 3-episode guards
the statistic is biased, and in *both directions* depending on the period
grid. Read against zero, Track A's −0.203 would have been reported as negative
persistence and its +0.223 as positive persistence. Read against its own null,
both say nothing (p 0.498 and 0.279).

That is the whole argument for permutation testing in one table.

---

## 6. The level, which persistence deliberately ignores

Persistence asks whether the *ranking* carries over. It says nothing about
whether the flow makes money, so that is reported separately.

| | value |
|---|---|
| per broker-year timing margin | median **+0.13 bps**, mean −6.08, sd 366 |
| value-weighted, all flow pooled | **+2.15 bps** per fortnight |
| A5 round-trip cost | **56 bps** |

Even granting the pooled +2.15 bps as real, it is a fortnightly directional
margin roughly **26× smaller than the cost of the round trip that would capture
it**. And the persistence result says the cross-sectional part of it cannot be
attributed to identifiable codes in advance.

**The two combine multiplicatively and both factors are ~0.** A tradeable
version of §12 needs a persistent ranking *and* a persistent margin gap wider
than 56 bps. The ranking's persistent component is indistinguishable from zero.

---

## 7. What this test can and cannot say

**Track B measures directional timing, not profit.** `timing_pnl = net_shares ×
(next window's close − this window's close)`. It is not §9.3's realised cohort
P&L and is never called that. A market maker earning the spread intraday shows
**zero** timing_pnl while being consistently profitable, so a null here rules
out persistent *directional* skill, not persistent P&L. That gap is real and it
is the main thing a longer daily store would close.

**A broker code is not an investor class.** §6 point 1: one code aggregates
thousands of accounts — retail, institutional, prop, foreign nominee. The
literature §12 cites (Taiwan, Finland) identifies *account types*, which the
exchanges there publish and IDX does not. **So this result does not contradict
that literature.** It says the broker code is too coarse an instrument to
isolate the losing cohort at fortnightly resolution — a statement about the
available data, not about whether persistent losers exist.

**Fortnightly resolution.** A cohort that reliably buys high and sells low
*within* ten sessions is invisible here, exactly as it was in Phase 2b.

**Half the rows are censored.** The two-sided subsample is the honest one and
it agrees with the full one (+0.045 vs +0.032, both inside their nulls), which
is reassuring but does not undo the censoring.

---

## 8. What would have falsified this, and what would change the answer

**Falsification:** a rank correlation above its null's 97.5th percentile,
holding across both estimators and both subsamples, with a margin gap between
the top and bottom groups wider than 56 bps. Nothing came close.

**What would change it:**

- **Daily broker summary over 10+ years.** Would let the clean §9.3 round-trip
  measure run on the long history instead of the timing proxy, and would catch
  within-fortnight round trips. Needs the licensed feed or the sanctioned
  browser-download route (`docs/FULL_REKAP.md` §3), not a scraper.
- **Investor-type data rather than broker codes.** This is the one that
  actually matches §12's literature. IDX publishes a foreign/domestic split;
  that is a class, not a code, and it is the right unit for the question.
- **Nothing else.** Adding features to rescue this is the move §7 forbids and
  it is forbidden here too.

---

## 9. Where this leaves the project

Three results now point the same way, and they were reached independently:

| | result |
|---|---|
| Phase 1 (H9) | aggregate flow: faint in-sample rank tilt, sign-unstable, zero spread before costs |
| Phase 2b (H10) | broker identity adds nothing to cohort P&L over a label shuffle |
| **§12 (H11)** | **broker-code margin rank does not persist** |

Gate 2b asked whether broker identity carries information beyond aggregate
flow. It does not, on cohort P&L, and now the identity is not even stable
enough to be worth carrying. §12's strategic inversion — take the other side of
persistent losers — remains the right question, but **the broker code is not
the instrument that answers it.**

The honest next step is the investor-type split, not more work on broker codes.
