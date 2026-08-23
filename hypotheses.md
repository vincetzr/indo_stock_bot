# Hypothesis log

Every pre-registered trial run in this repo, in order. Required by CLAUDE.md §8
and §11: the number of trials matters more than any single result, and a
deflated Sharpe ratio cannot be computed without an honest count.

**Rule: a hypothesis is added here BEFORE it is run, with its prediction.**
A row added after seeing the answer is not a trial, it is a description, and it
must be marked `EXPLORATORY` so it never enters the trial count.

**Running confirmatory trial count: 8. Survivors: 0.**

---

## Protocol A — does broker buying predict a RISE?

`scripts/layer2_protocol.py`, hash `6b8e0a2c9d1f4e73`. Frozen before any panel
existed. α = 0.05, Bonferroni to 0.0125, power target 80%, day-level clustering,
entry at the close of t+1, three censoring levels.

| id | date frozen | hypothesis | horizon | predicted | result |
|---|---|---|---|---|---|
| H1 | 2026-08 | top-3 net buying predicts POSITIVE excess return | t+1..t+5 | d > 0 | **FAIL** — negative |
| H2 | 2026-08 | top-3 buy concentration predicts POSITIVE excess return | t+1..t+20 | d > 0 | **FAIL** — negative |
| H3 | 2026-08 | net foreign buying predicts POSITIVE excess return | t+1..t+5 | d > 0 | **FAIL** — negative |
| H4 | 2026-08 | same top net buyer 3 sessions running predicts POSITIVE excess return | t+1..t+10 | d > 0 | **FAIL** — d = −1.34, t = −5.37 |

All four failed, and all four failed *with the sign reversed*. Run on BBCA only.

---

## Protocol B — does broker buying predict a FALL?

`scripts/layer2_protocol_b.py`, hash `b8c26de3a02d24f7`. Frozen while only BBCA
had been examined and the other nine names were still being collected.
**BBCA excluded from the confirmatory sample** — it generated these hypotheses
and cannot also test them. Mandatory same-day-return control.

Sample: 9 untouched names × 361 sessions = 2,886 ticker-days, 2025-02-10 to
2026-08-21. Design effect 3.4, effective n ≈ 956, MDE d = 0.100. Adequately
powered.

| id | hypothesis | horizon | predicted | d | t (HAC) | p (HAC) | result |
|---|---|---|---|---|---|---|---|
| H5 | top-3 net buying predicts NEGATIVE excess return | t+1..t+5 | d < 0 | −0.062 | −0.93 | 0.177 | **FAIL** |
| H6 | top-3 buy concentration predicts NEGATIVE excess return | t+1..t+20 | d < 0 | −0.163 | −1.25 | 0.106 | **FAIL** |
| H7 | net foreign flow predicts NEGATIVE excess return | t+1..t+5 | d < 0 | −0.041 | −0.62 | 0.269 | **FAIL** |
| H8 | same top net buyer 3 sessions running predicts NEGATIVE excess return | t+1..t+10 | d < 0 | +0.099 | +0.89 | 0.812 | **FAIL** — wrong sign |

### H6 is the one that nearly got through, and how it was caught

Under the iid t-test H6 read **t = −2.67, p = 0.0041**, which clears the
Bonferroni α of 0.0125. It was reported as SURVIVING before the overlap was
checked. It does not survive:

| method | t | p |
|---|---|---|
| iid t-test | −2.67 | 0.0041 |
| Newey–West, L = 19 | −1.25 | 0.106 |
| Newey–West, L = 40 | −1.15 | 0.126 |
| non-overlapping subsamples (20 offsets) | — | median 0.315; 1/20 beat 0.0125 |
| moving-block bootstrap, block = 20 | — | 95% CI **[−2.20%, +0.29%]**, straddles zero |

The cause: a 20-day forward return computed on consecutive days shares 19/20 of
its window with its neighbour. 266 day-level observations are about **13
independent windows**, not 266. The iid test inflated t by ≈ 2.1×.

Fixed permanently in `layer2_test.py` — `one_sided` now takes the horizon and
returns HAC-corrected `t`/`p` plus a block-bootstrap CI, keeping `t_iid`/`p_iid`
alongside so the size of the correction stays visible. `tests/test_overlap.py`
holds the calibration in place.

### Robustness: the verdict survived the Phase 0 spine repairs

Protocol B was re-run after Phase 0 corrected the return series - point-in-time
auto-rejection caps instead of a flat 35%, verified corporate actions adjusted
rather than clipped, quarantined windows exempted. **Every number is
unchanged.** The panel is 10 large caps with no corporate action inside the
2025-2026 window, so the corrections correctly had nothing to move. The failure
is a property of the data, not of the pipeline that was reading it.

### A second defect found in the same pass

H6 and H7 printed identical numbers at all three censoring levels and this was
being read as three independent robustness checks. It is one check printed three
times: `concentration` is computed from observed buy lots and `foreign_net` from
the published foreign value, so **neither depends on the bracket level at all**.
Only H5 and H8 genuinely vary with censoring. The runner now detects and states
this.

---

## Multiple-testing position

Protocols A and B are the same four questions with the sign flipped, so the
honest family is **4 two-sided hypotheses**, not 8 one-sided ones. Under that
reading each side gets α = 0.05 / 4 / 2 = **0.00625**. Nothing in either
protocol reaches it. Under the looser 8-independent-trials reading (α = 0.00625
Bonferroni) nothing reaches it either. The conclusion does not depend on which
correction is used.

---

## Standing conclusion

**Broker flow, as visible in the top-ten EOD broker summary, does not predict
forward returns on the 10 large-cap names tested, in either direction, at
horizons of 5, 10 or 20 days.**

Effect sizes are small and inconsistent in sign: |d| ≤ 0.163, and the largest of
them is the one that dies under a correct standard error. This is **Gate 1
failing**, not a null awaiting more features. Per CLAUDE.md §2 and §7, the
response is to pivot the hypothesis, not to add features to rescue it.

What is NOT established by this, and would need its own trial:

- the same test on **small caps**, where the effect might live (§7 asks for IC
  by liquidity decile; the current panel is 10 large caps and cannot answer it)
- the same test on a **full-depth** rekap rather than a top-ten one
- **cohort P&L** (§9.3), which is a different question from prediction and is
  not touched by this failure
- **execution edge** (§9.4), which is descriptive, is not a forecast, and is
  the one measurement here that has survived scrutiny — see `reports/`

---

## Exploratory observations — NOT trials, NOT in the count

Recorded so they are not later mistaken for confirmatory results.

| id | observation | status |
|---|---|---|
| E1 | execution edge vs VWAP is stable within BBCA (split-half Spearman +0.691, p = 0.0004) | exploratory, generated on BBCA |
| E2 | the edge ranking transfers BBCA → BBRI (Spearman +0.476, p = 0.025) but **slope only 0.200** | out-of-sample on ranking; magnitude ~5× overstated |
| E3 | flow has memory: median lag-1 autocorrelation of daily net lots +0.194 | exploratory |
| E4 | "big desks execute worst" is FALSE — no size effect | withdrawn, recomputed, **reinstated**. See below. |

### E4: the §9.4 bias was real, was fixed, and did not change the answer

E4 was withdrawn on the grounds that it was computed against a VWAP including
each broker's own trades — the bias CLAUDE.md §9.4 identifies. That was the
right call to make and the wrong outcome to assume. Recomputed against a
self-excluded VWAP on the 10-name panel:

| | corr(log V, edge) | p | corr(log V, \|edge\|) | p | mean \|edge\| |
|---|---|---|---|---|---|
| biased (VWAP includes self) | −0.002 | 0.99 | −0.234 | 0.11 | 0.199% |
| corrected (VWAP excludes self) | −0.001 | 1.00 | −0.213 | 0.15 | 0.220% |

**The correction is real and it is implemented.** Verified on a synthetic
closed market with known true edges: a broker holding 40% of the day measures
+0.43% biased against a true +0.83%, and +0.71% corrected. On the real panel it
de-shrinks mean |edge| by 11%, and broker shares are large enough to matter —
median 5.7% of a session, 26% of broker-sides above 10%, maximum 94%.

**It does not overturn the size null.** Neither the signed nor the absolute
correlation with volume approaches significance after correction. The reason
the correction moves the *correlation* so little is that it attenuates edge
magnitude roughly symmetrically about zero, so it barely disturbs a signed
relationship — it distorts how big an individual broker's edge looks, not
whether edge tracks size.

So E4 stands: **size does not predict execution quality on this panel.** It now
stands on a measurement that is not biased in the direction of that conclusion,
which is the only reason it is worth anything.

---

## H9 — broker-flow imbalance predicts the forward fortnight, cross-sectionally

**Pre-registered:** 2026-08-22, before the panel existed. Protocol: rank the
cross-section each fortnight on top-10 net-buy imbalance, neutralise on
momentum / short-term reversal / trailing turnover / trailing vol / 5
statistical factors, correlate with the next fortnight's return, Newey-West
the series of ICs, and compare against a shuffled-label null through the
identical pipeline. Holdout: the most recent 24 months, untouched.

**Prediction:** if the earlier BBCA-only result (A3, Protocol A — all four
hypotheses failed with a NEGATIVE sign) reflected something real rather than
one name's idiosyncrasy, a 176-name panel would show a negative IC that
survives controls, is stable across liquidity, and produces a tradeable spread.

**Result: the sign replicated, the magnitude did not.**

| | value |
|---|---|
| in-sample rows / periods / names | 21,693 / 241 / 176 (64 delisted) |
| IC (neutralised, +10d) | **−0.0190**, HAC se 0.0066, t −2.86 |
| IC (+20d) | −0.0190, t −2.32 — the decay curve is flat |
| raw IC, no controls | −0.0051, t −0.74 |
| 200-draw permutation p | **0.005** two-sided; null t sd = 1.09, so HAC is calibrated |
| Bonferroni threshold (8 prior + ~12 here) | **0.0025 — not cleared** |
| liquidity quintiles | Q4 **+0.032**, Q5 **−0.047** — sign flips |
| quintile spread, same neutralised score | −0.215%/fortnight, t −0.70, **gross** |
| on verifiable-coverage rows only | t falls −2.86 → −1.94 |

**Gate 1 FAILS on the conjunction §7 states.** Significantly non-zero in
sample, yes. Stable sign, no. Post-cost, no — it is not distinguishable from
zero *before* costs. Out of sample, untested and staying that way.

**Two errors made and caught, both in the harness rather than the data.**
First run: flow t = −2.86 and NULL t = **−2.96**. The null was
cross-period-scrambled by a positional assignment against a ticker-sorted
frame — a null that certified anything. Fixed and pinned by a regression test.
Then I over-read a single null draw at t = −1.53 as systematic bias; a second
seed said otherwise. That prompted the 200-draw permutation test, which is the
better statistic and should have been the first one.

**Trials after H9: 20.** Full memo: `reports/phase1_flow_panel.md`.

---

## H10 — does broker IDENTITY carry information about cohort profitability?

**Date:** 2026-08-23. §9.3 cohort P&L on the daily store, with §9.3's own
shuffled-broker-label null. Full memo: `reports/phase2b_cohort_pnl.md`.

**Prediction:** if broker identity means anything, the real-label distribution
of round-trip `margin_bps` separates from a distribution where the labels are
shuffled within each ticker-day.

**Result: it does not.**

| | median margin_bps | n | 95% CI |
|---|---|---|---|
| round-trip, real labels | **−25.3** | 147 broker-tickers, 628 episodes | [−395, −29] |
| round-trip, SHUFFLED | **−32.5** | 162, 883 episodes | [−278, −28] |

Indistinguishable. **No detectable broker-identity signal in cohort P&L.** With
Phase 1's aggregate baseline at essentially nothing, both halves of Gate 2b now
point the same way.

**A second finding, about the market rather than the brokers.** Both figures are
negative and similar: the median cohort round trip loses ~25–32 bps of what it
traded, whoever does it. That is the spread paid twice. Against A5's 56 bps
round-trip cost, the median cohort round trip is a losing trade on execution
alone, before commission or tax.

**The estimator was wrong twice; the null caught it both times.**
1. WAC starts at zero, so a cohort already long booked its opening sell's entire
   proceeds as profit — +Rp 10,000,000 on a synthetic true-zero case, and the
   null read **+6.3 bps with a CI excluding zero**. Fixed by attributing a cost
   basis only to shares actually recorded as held, and by computing round-trip
   P&L as sell value − buy value with no WAC at all.
2. `unrealized = inventory × (close − WAC)` on negative inventory gave full-path
   margins of **−13,000 bps** (−130% of gross). Now NaN there; full-path is
   computable on only **49%** of series.

**§9.2's limits, measured:** negative inventory **46.4%** of the time (median
broker-ticker), crossing ratio median **0.84**.

**Not tested and not claimed:** persistence. A cohort losing 25 bps is
uninteresting unless the *same* cohort keeps doing it, and 18 months on 9 names
cannot establish that. That is §12's actual question and it remains open.

**Trials after H10: 22.**
