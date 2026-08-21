# Phase 1 memo — does broker flow predict anything at all?

**Date:** 2026-08-21
**Protocols:** A `6b8e0a2c9d1f4e73`, B `b8c26de3a02d24f7`
**Verdict: Gate 1 FAILS.** 8 pre-registered trials, 0 survivors.

---

## 1. What was tested

CLAUDE.md §7's hypothesis, in both directions:

> Broker net-buy imbalance for ticker *i* on day *t* predicts the return of *i*
> over *t+1 … t+k*.

Protocol A froze four one-sided *positive* claims before any panel existed. It
failed, and failed with every sign reversed. Because a reversed sign discovered
in data is an observation and not a finding, the reversed claims were frozen
separately as Protocol B — **with BBCA, the name that generated them, excluded
from its own confirmatory sample** — and run on nine untouched names.

## 2. Sample and power

| | |
|---|---|
| names | 9 (ADRO, ANTM, ASII, BBNI, BBRI, BMRI, GOTO, MDKA, TLKM) |
| sessions | 361, 2025-02-10 → 2026-08-21 |
| ticker-days | 2,886 |
| design effect | 3.4 (ICC 0.30, day-level clustering) |
| effective n | ≈ 956 |
| smallest detectable effect | d = 0.100 |

Adequately powered for the effect sizes the protocol was written to detect.
This is **not** a null caused by too little data at these horizons.

## 3. Result

One-sided, Bonferroni α = 0.0125, Newey–West corrected for window overlap.

| id | signal | h | d | t | p | up-days | down-days | verdict |
|---|---|---|---|---|---|---|---|---|
| H5 | top-3 net | 5 | −0.062 | −0.93 | 0.177 | −0.08 | +0.04 | fail |
| H6 | concentration | 20 | −0.163 | −1.25 | 0.106 | −0.28 | −0.07 | fail |
| H7 | foreign net | 5 | −0.041 | −0.62 | 0.269 | −0.18 | −0.00 | fail |
| H8 | streak-3 | 10 | +0.099 | +0.89 | 0.812 | +0.12 | +0.00 | fail, wrong sign |

Largest effect anywhere: |d| = 0.163. Signs are inconsistent across hypotheses.

## 4. The near-miss, and what killed it

**H6 was reported as SURVIVING before the overlap was checked.** Under an iid
t-test it read t = −2.67, p = 0.0041, clearing Bonferroni, negative on both
control subsets, stable across censoring levels. It was the first thing in this
project to pass a pre-registered confirmatory test.

It does not survive. A 20-day forward return computed on consecutive days shares
19/20 of its window with its neighbour, so the day-level series is
autocorrelated **by construction** even under a true null. 266 day-level
observations are roughly **13 independent windows**.

| method | t | p |
|---|---|---|
| iid t-test | −2.67 | 0.0041 |
| Newey–West, L = 19 | −1.25 | 0.106 |
| Newey–West, L = 40 | −1.15 | 0.126 |
| non-overlapping, 20 offsets | — | median 0.315; 1 of 20 beat 0.0125 |
| moving-block bootstrap, block 20 | — | 95% CI **[−2.20%, +0.29%]** |

The iid test inflated t by ≈ 2.1×. This is CLAUDE.md §11's "purge overlapping
label windows" requirement, and it was the difference between a result and
nothing.

**A second defect found in the same pass.** H6 and H7 printed identical numbers
at all three censoring levels, and the runner read that as three independent
robustness checks. It is one check printed three times: `concentration` is
computed from observed buy lots and `foreign_net` from the published foreign
value, so neither depends on the bracket level at all. Only H5 and H8 genuinely
vary with censoring.

## 5. What would have falsified the null

A consistent sign across hypotheses, surviving HAC correction, present on both
up-days and down-days, at |d| ≥ 0.10 on the untouched names. H6 met three of
those four and failed the one that mattered.

## 6. What I now believe, and with what confidence

**High confidence:** the top-ten EOD broker summary carries no exploitable
forward-return signal on large-cap IDX names at 5, 10 or 20 days, in either
direction. Eight pre-registered trials, zero survivors, adequate power.

**Not established, and each needs its own trial:**

- **Small caps.** §7 asks for IC by liquidity decile. This panel is 10 large
  caps and structurally cannot answer it. If flow signal exists anywhere it is
  most likely where §6's "public and simultaneous" competition is weakest.
- **Full-depth rekap.** Everything here rests on a top-ten table missing 10–15%
  of volume. A complete rekap is a different measurement, not a cleaner one.
- **Shorter horizons.** h = 1 was not in either protocol.
- **Cohort P&L** (§9.3) — a different question entirely, untouched by this.

## 7. Consequence for the project

Per CLAUDE.md §2 and §7: **pivot the hypothesis, do not add features to rescue
it.** Phase 2 (feature expansion) is gated on Gate 1 and does not open.

The two directions the evidence actually supports:

1. **§12's strategic inversion.** The question stops being "does flow predict
   price" and becomes "which cohort persistently loses". Execution edge (§9.4)
   is the one measurement here that has survived scrutiny — it is descriptive
   rather than predictive, and it is zero-sum by construction, so it says who
   pays whom without needing a forecast. Cohort P&L (§9.3) is the natural next
   step and the brief's round-trip restriction is the right way to handle the
   unknown starting inventory.
2. **Small caps**, as a genuinely new Phase-1 trial with its own pre-registration
   — not as a re-run of these hypotheses on new data until one passes.

## 8. Standing caveat on the execution-edge work

It is exploratory, generated by looking at BBCA, and it must not be quoted as a
confirmatory result. Its one out-of-sample test (BBCA → BBRI) gives a rank
correlation of +0.476, p = 0.025, **but a regression slope of 0.200** — the
ranking transfers, about a fifth of the magnitude does. Anything acted on must
be shrunk by that slope first.
