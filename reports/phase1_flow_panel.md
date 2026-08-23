# Phase 1 memo — does broker flow predict anything?

**Date:** 2026-08-23
**Gate 1: FAILS.**
Reproduce with `python3 scripts/flow_panel_build.py && python3 scripts/flow_ic.py`.

---

## 1. The verdict, first

§7's Gate 1 is a **conjunction**: *"Post-cost, liquidity-filtered,
control-neutralised IC is significantly non-zero **with a stable sign**, out of
sample."* Four conditions. The panel clears one of them.

| Gate 1 condition | result |
|---|---|
| significantly non-zero | **YES, in sample** — IC −0.0190, permutation p = 0.005 |
| stable sign | **NO** — Q4 +0.032, Q5 −0.047, sign flips between adjacent quintiles |
| post-cost | **NO** — the quintile spread is −0.215%/fortnight, t = −0.70, i.e. indistinguishable from zero **before** any cost |
| out of sample | **NOT TESTED** — the 24-month holdout is untouched, and stays that way |

**The one-line summary: there is a faint, statistically real rank tilt in the
in-sample data, and it is far too small to trade.** Those two facts are not in
tension — an |IC| of 0.019 across ~80 names is a whisper, and a portfolio built
on it never leaves the noise.

**No feature was added to rescue it**, per §7. The hypothesis is reported as
failing and the pivot is discussed in §8.

---

## 2. What was built to test it

A1 costed the §4 panel at ~2,000,000 requests and 27 days, and concluded Phase 1
had to run on 10 names. **That costing was wrong.** It priced IndoPremier's
*daily* mode; the same endpoint takes `start`/`end` and returns the rekap
aggregated over the whole window for one request.

| | before | now |
|---|---|---|
| names | 1 (BBCA) | **176**, of which **64 delisted** |
| cross-section per period | — | median **80** in sample |
| periods | — | **241** fortnights in sample, 2014-01 … 2024-08 |
| labelled rows | — | **30,234** (21,693 in sample, 5,703 reserved) |
| requests | — | 31,824, **zero failures** |

The panel is stratified 20-per-quintile on **point-in-time entry liquidity**
(first 250 traded bars), not full-sample liquidity, so §7's decile question is
not secretly a question about which names grew.

**It is substantially survivorship-corrected.** 64 of the 88 recovered delisted
names are served by IndoPremier and are in the panel with their real flow.
The 24 that are not are listed in `config/flow_panel_unserved.txt` — naming what
is missing is the measurement of the residual bias.

**What was given up:** fortnightly flow, not daily. §7's decay curve runs
k ∈ {1,3,5,10,20}; this panel speaks to k = 10 and 20 and **cannot** speak to
k = 1 or 3. That is a narrowing of the stated hypothesis and every number here
carries it. Aggregation runs one way — two fortnights make a month, no
arithmetic recovers a day.

---

## 3. The result in full

```
                       periods   mean IC   HAC se      t    raw IC  raw t
flow -> +10d               241   -0.0190   0.0066  -2.86   -0.0051  -0.74
flow -> +20d               241   -0.0190   0.0082  -2.32   -0.0025  -0.34
NULL -> +10d               241   -0.0032   0.0071  -0.45   -0.0006  -0.10
```

**The decay curve is flat**, which is itself informative: whatever this is, it
does not decay between 10 and 20 sessions the way a genuine information effect
usually does.

**The raw IC is not significant** (−0.0051, t = −0.74). The entire measured
effect appears only after neutralisation. That is suspicious on its face, so it
was tested rather than assumed:

```
controls                             flow t   NULL t
none (raw)                            -0.74    -0.07
4 controls only                       -1.88    -0.67
4 controls + 5 statistical factors    -2.86    -0.22
```

The null stays flat while the signal strengthens, so the controls are removing
variance that masks a weak negative relation rather than manufacturing one.

### The permutation test, which is the decisive statistic

A single null draw is one draw. **200 nulls were run through the identical
pipeline:**

```
null t : mean -0.10, sd 1.09, p5 -1.81, p95 +1.78
observed flow t = -2.86  ->  empirical p = 0.005 (two-sided)
```

The null's t has sd 1.09, so the HAC standard error is correctly calibrated —
a useful validation of the whole harness. The observed −2.86 lies outside all
200 draws. **The IC is real in-sample and is not an artefact.**

### And it still does not clear the trial count

§11 requires the number of trials to be tracked. `hypotheses.md` carried 8
pre-registered trials before this memo; this study adds ~12 specifications
(two horizons, a coverage cut, five quintiles, three control sets, a spread).
Bonferroni over 20 trials needs **p < 0.0025**. The observed p is **0.005**.

**It does not clear.** The honest reading is that a p of 0.005 on the twentieth
look is roughly what a null universe produces.

---

## 4. Liquidity: the sign is not stable

§7: *"Report IC by liquidity decile. If the effect lives only in the bottom two
deciles it is likely untradeable — say so."*

| quintile | median turnover/day | IC | t |
|---|---|---|---|
| Q1 | Rp 811,300 | −0.0139 | −0.54 |
| Q2 | Rp 23.2m | −0.0120 | −0.53 |
| Q3 | Rp 223.8m | −0.0013 | −0.07 |
| Q4 | Rp 1.55bn | **+0.0322** | +1.33 |
| Q5 | Rp 18.8bn | **−0.0466** | −2.13 |

The effect does **not** live only in the illiquid tail — which would have been
the expected failure. It does something worse: **the sign flips between adjacent
quintiles**, +0.032 in Q4 and −0.047 in Q5. A real cross-sectional effect does
not alternate direction between neighbouring liquidity buckets. This pattern is
what noise looks like when five buckets are drawn from a distribution centred
near zero.

---

## 5. Costs: it never gets far enough to pay them

Ranking on the **same neutralised score** the IC is measured on (the first
version ranked on the raw score, which is a different signal and made the
economics look worse than the statistics warranted):

```
gross spread per fortnight  -0.215%   (HAC se 0.307%, t -0.70, 241 periods)
less 2 x 0.56% round trip   -1.335%   =  -33.4% a year
```

**Gross, before a single rupiah of cost, the spread is not distinguishable from
zero.** Costs are not what kills this; there is nothing for them to kill.

A5 forbids shorting, so the spread is diagnostic only — the investable object is
the long leg alone, carrying one round trip rather than two.

---

## 6. Data quality: the result weakens where the data is verifiable

`coverage_ok` marks rows where the top-ten share of volume could be checked
against the spine (82% of rows).

```
                       periods   mean IC   HAC se      t
flow, coverage_ok          227   -0.0150   0.0077  -1.94
```

On the subset where the inputs are independently corroborated, t falls from
−2.86 to −1.94. A result that is strongest where the data is least checkable is
not one to build on.

---

## 7. Two mistakes made and caught, both worth recording

**The null was broken, and it certified the signal.** The first run reported
flow at t = −2.86 and the NULL at t = **−2.96**. A shuffled label cannot
predict anything, so that was a bug: the panel is sorted by ticker, the
permutations were concatenated in period order, and positional assignment
scattered every period's shuffled values onto rows from other periods. A
groupby transform fixed it; `tests/test_flow_ic.py` now reproduces the historical
bug and requires the fix. **This is the single most valuable thing the null did**
— §11 asks for a null run through the identical pipeline precisely because the
pipeline is where the error lives.

**Then I over-read one null draw.** With the bug fixed, one draw showed the null
at t = −1.53 on the coverage subset and I briefly suspected systematic bias in
the neutralisation. A second seed gave −0.07 / −0.67 / −0.22. It was sampling
noise in a single draw — the exact mistake this repo already has a test against.
That is what prompted the 200-draw permutation test, which is a better statistic
than anything that preceded it.

---

## 7b. Reproducibility, and the one place it is not exact

Rebuilding the panel from the same cache is **content-identical** (sha256 match),
and `flow_ic.py` reproduces IC −0.0190 / t −2.86 exactly. But rebuilding after a
price-cache refresh is not bit-identical, and it is worth naming why.

Refreshing prices moved `mom12_1` on **939 of 30,410 rows (3.1%)**, across four
names — BBRI, BRIS, TOWR, UNVR. The magnitude is **max 2.5e-06, mean 6.1e-09**,
i.e. about 1e-05 of a typical momentum value. Every other column, including
`imbalance`, `fwd_1w` and `coverage`, is bit-identical.

This is vendor float noise, not a data change: `mom12_1` is computed on
`adj_close`, and Yahoo re-derives the whole adjusted history when a name's
corporate-action record changes, which shifts the last bits. The four affected
names are the ones whose records moved.

**It does not touch the result** — the headline reproduces to four decimals
either way. It is recorded because "the build is deterministic" and "the study is
reproducible" are different claims, and only the first is exactly true. Anyone
re-running this months from now should expect the controls to drift at that
order, and should be suspicious if they drift by more.

---

## 8. What I believe, and with what confidence

**High:** the panel itself. 31,824 windows, zero failures, look-ahead controlled
by a tested rule (label starts at T+1, not T), 64 delisted names included.

**High:** the quintile spread is not tradeable. Gross, pre-cost, t = −0.70.

**High:** Gate 1 fails on the conjunction §7 states.

**Medium:** that the in-sample IC of −0.019 is a real feature of 2014-2024 IDX
rather than a twentieth-look artefact. The permutation p is 0.005; the
Bonferroni threshold is 0.0025. Genuinely on the line.

**Medium:** the negative sign. It is now the second independent sample to come
back negative — Protocol A's four hypotheses all failed with a negative sign on
BBCA (A3), and this 176-name panel agrees in direction. Two samples agreeing on
a sign they were not selected for is worth more than either alone. But the
magnitude is not distinguishable from noise after the required controls, so this
is a direction without a size.

**Low:** any claim about mechanism. Nothing here identifies *why* high top-ten
buying would precede underperformance, and §6 point 4 forbids inventing one.

---

## 9. What would have falsified this

A clean pass would have shown: a monotone IC across liquidity quintiles, a decay
curve that actually decays, a gross spread several standard errors above zero,
and a result that strengthened rather than weakened on the verifiable-coverage
subset. None of those appeared.

---

## 10. Where this goes next — pivot, not rescue

§7: *"If not — report it and we pivot the hypothesis rather than adding features
to rescue it."*

The obvious rescue moves are all forbidden and all recorded here so nobody
tries them later: adding features until the IC clears; dropping Q4 because it
has the wrong sign; reporting the best horizon; using the 82% of rows where
coverage cannot be verified because the number is bigger there.

The pivot §12 already points at is **not** "which flow predicts returns" but
**"which flow is persistently dumb, is it identifiable in real time, and is it
large enough to trade against after costs."** That is a different question about
the same data, it does not require the aggregate imbalance to predict anything,
and the panel just built — 176 names, 64 of them delisted, per-broker rekap over
241 fortnights — is the right input for it. Gate 2b then asks whether broker
identity adds anything over aggregate flow, and the aggregate flow it must beat
is now measured: **essentially nothing**, which is a low bar and an honest one.

**The 24-month holdout has not been touched.** It is worth one look, and only
after there is a candidate worth spending it on.
