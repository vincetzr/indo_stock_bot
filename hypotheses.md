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

---

## H11 — does a broker code's margin RANK persist from one period to the next?

**Date:** 2026-08-23. §12's actual question, and the one H10 explicitly left
open. Full memo: `reports/phase2b_persistence.md`.

**Prediction, stated before running.** §12 argues the losing cohort is durable
because it continuously regenerates. If that holds at broker-code resolution,
brokers ranked by margin in one period rank the same way in the next, and the
rank correlation sits **above** its label-shuffled null. The p-value is
one-sided upward for that reason.

**Six confirmatory statistics:** {split-half, period-over-period} ×
{all rows, two-sided only} on Track B, and the same two estimators on Track A.

**Result: all six sit inside their own permutation nulls.**

| track | statistic | observed | null mean | null 95% | one-sided p |
|---|---|---|---|---|---|
| B | split-half, all rows | −0.185 | +0.000 | [−0.222, +0.229] | 0.930 |
| B | split-half, two-sided | +0.076 | −0.001 | [−0.224, +0.276] | 0.239 |
| B | year-over-year, all rows | +0.032 | −0.004 | [−0.079, +0.072] | 0.144 |
| B | year-over-year, two-sided | +0.045 | −0.004 | [−0.089, +0.083] | 0.119 |
| A | split-half | −0.203 | −0.207 | [−0.504, +0.108] | 0.498 |
| A | quarter-over-quarter | +0.223 | +0.122 | [−0.070, +0.335] | 0.279 |

Track B: 89 codes, 217 tickers, 378,689 broker-windows, 2014–2026.
Track A: 27 codes, 628 round-trip episodes, 1.5 years. Null = 200 draws
shuffling broker labels within each ticker-window, guards inside the loop.

**The one sub-0.05 figure, and why it is not real.** Year-over-year on all rows
*including the pre-panel era* reads +0.078, p = 0.025. It is one pair:
`2013→2014 = +0.632`, three times the next largest, computed on **13 brokers
from 262 rows and 15 windows** against 2014's 82 brokers from 26,900 rows.
Dropping the 454 pre-panel rows — 0.1% of the data — gives +0.032, p = 0.144,
and brings the two subsamples into agreement.

**That filter is post-hoc and is marked as such.** It was chosen after seeing
the pair, justified on sample size rather than on the answer, and both versions
are printed by the script. It does not enter the trial count.

**The null earned its keep.** Track A's null is **not centred on zero** —
−0.207 for split-half, +0.122 for quarter-over-quarter. Read against zero,
Track A would have been reported as showing negative persistence *and* positive
persistence depending on the period grid. Read against its own null, neither
says anything.

**A wrong explanation, recorded because it was tested and failed.** The top-10
censor forces a one-sided code's net long by construction, and only 51.1% of
rows are two-sided, so censoring looked like the obvious artefact. Censoring is
indeed a strongly persistent broker attribute (year-over-year rank corr
**+0.801** against margin's +0.078) — but it does **not** predict the measured
margin within a year (**−0.037**, 6/14 pairs positive). The chain does not
close and the explanation is wrong.

**The level, reported separately because persistence ignores it.** Per
broker-year timing margin: median +0.13 bps, mean −6.08, sd 366. Value-weighted
pooled +2.15 bps a fortnight, against A5's 56 bps round-trip cost.

**What this does NOT say.** Track B measures directional timing, not profit — a
market maker earning spread shows zero timing_pnl while being profitable. And a
broker code is not an investor class (§6.1), so this does not contradict the
Taiwan/Finland literature §12 cites; it says the broker code is too coarse an
instrument to isolate the losing cohort at fortnightly resolution.

**Trials after H11: 28.** Bonferroni bar α = 0.05/28 = **0.0018**. Nothing here
is within an order of magnitude of it.

---

## H12 — is an INVESTOR CLASS persistently on the profitable side? (PRE-REGISTERED)

**Registered 2026-08-23, before the data was collected and before any statistic
was computed.** H11 closed by saying its finding was about the *instrument*:
a broker code aggregates thousands of accounts of mixed type (§6.1), while the
Taiwan and Finland results §12 cites identify account TYPES. IDX tags each
trade with the investor's domicile and IndoPremier serves that split through
the same endpoint (`fd=F` / `fd=D`), so §12's question can now be asked with an
instrument that matches it.

**The measure.** Per ticker and fortnight, using strictly forward returns:

```
timing_pnl = net_value x forward_return        (window_end close -> next window_end close)
margin_bps = 10000 x sum(timing_pnl) / sum(gross_value)
```

**Prediction.** Foreign participation in IDX is overwhelmingly institutional
and domestic flow contains essentially all of Indonesian retail, so the
literature's direction maps onto this split as:

> **foreign margin > 0 and domestic margin < 0.**

The mapping is imperfect — domestic institutions are large in IDX and are
pooled into "domestic" — so the p-value reported is **two-sided**, which is the
conservative choice. A one-sided test would be more powerful and is not taken.

**Three conditions, ALL of which must pass** before §12's strategy has an
instrument. Failing any one is the result, not a threshold to move (§2):

| | condition |
|---|---|
| 1. LEVEL | margin outside a 200-draw null that shuffles forward returns across tickers *within* each window |
| 2. PERSISTENCE | the sign holds across years, and the pooled figure survives dropping its largest single year |
| 3. SIZE | \|margin\| > A5's **56 bps** round trip |

**Trial count: this is 1 confirmatory trial per class, 2 in total.**
Trials after H12 will be **30**, so the Bonferroni bar is α = 0.05/30 = **0.0017**.

**What is already verified about the instrument, before any result** — these
are data facts, established against 28 BBCA sessions with all three views
cached plus live probes, and they hold whatever the test returns:

- `fd=F`/`fd=D` composes with `start`/`end`, so the split costs one request per
  (ticker, window, view) — the same economics as the combined panel, **not** the
  per-ticker-day figure. Verified live, not assumed.
- F + D reconcile to the combined footer total to **0.023%** median.
- The F view reproduces the published `F. NVal` to **1.1%**.
- **The broker-flag proxy is a different quantity.** The panel's `foreign_net`
  sums foreign-*flagged members*; against the domicile split it correlates
  +0.749 but **disagrees in sign on 29% of sessions**, and puts foreign share of
  gross at 65.0% against the domicile figure's 74.6%. `ipot.parse_totals` had a
  docstring asserting the opposite; it was wrong and is corrected.
- Censoring bound: F_net and D_net are structurally exact mirrors, so their
  residual measures the top-10 cut directly — **2.2% of gross at the median,
  7.1% at worst**.
- Foreign participation tracks liquidity hard (decile 9 **49.4%**, 8 30.0%,
  7 19.4%, 5 3.4%), so the universe is the top liquidity strata. That is a
  statement about where the test has power, not a convenience.

### H12 — RESULT

**Run 2026-08-23 on the complete panel.** 18 names (liquidity decile 9), 329
fortnights, 5,993 class-windows, 2014-01-14 … 2026-08-11; 2,935 ticker-windows
carry both views. Collection: 6,346 requests, **zero errors**, 220 empty.
Full memo: `reports/phase2b_investor_split.md`.

**Result: all three pre-registered conditions FAIL, for both classes.**

| | FOREIGN | DOMESTIC |
|---|---|---|
| margin | **−1.70 bps**/fortnight | **+1.02 bps** |
| direction null | +0.55 [−12.72, +11.29], **p 0.692** | +0.12 [−8.35, +11.86], **p 0.816** |
| selection null | −6.71, p 0.587 | +4.02, p 0.627 |
| years sharing pooled sign | 38% of 13 | 46% of 13 |
| lag-1 autocorr of annual margin | **−0.410** | **−0.331** |
| drop largest year | −2.53 → **+0.67** | +0.18 → **−1.89** |
| vs 56 bps round trip | fails by **33×** | fails by **55×** |

**THE PRE-REGISTERED PREDICTION WAS WRONG IN SIGN.** H12 predicted
foreign > 0 > domestic; observed is foreign −1.70, domestic +1.02. Both sit
deep inside their nulls so the reversal is not itself a finding, but the
registered direction failed and is logged as failed.

**This is not an underpowered null.** Null sd 6.34 (F) and 4.95 (D), so the
test resolves ±12.4 and ±9.7 bps at p<0.05 — and the 56 bps cost bar sits
**8.8 and 11.3 null-sds away**. An effect large enough to trade would have been
found comfortably. What is there instead is 1–2 bps in a sign that will not
hold still.

**Persistence fails by reversing, not by being weak.** Both classes show
NEGATIVE lag-1 autocorrelation of the annual margin: a good year tends to be
followed by a bad one. And dropping the single largest year flips the pooled
sign for both — the same detector that caught H11's thin-year headline, firing
a second time.

**Two structural checks passed, which is what licenses reading any of it.**
Annual foreign and domestic margins correlate **−0.896** with opposite signs in
10 of 13 years, as the zero-sum identity demands. And the censoring bound,
measured from the F/D mirror residual rather than assumed, is **2.29% of gross
at the median**. That per-window error is much larger than the margin, so a
POSITIVE result of this size could not have been trusted — which costs nothing
here, since the bar that matters is 56 bps and the resolution is ±10.

**Not claimed:** that the Taiwan/Finland literature is wrong. Foreign/domestic
is an imperfect proxy for institution/retail — domestic institutions are large
in IDX and pooled into "domestic" — and this measures directional timing, not
realised P&L.

**Trials after H12: 30.** Bonferroni bar α = 0.05/30 = **0.0017**; the best
p-value here is 0.587.

---

## H13 — do PRICE/TA and structural features predict IDX returns? (PRE-REGISTERED)

**Registered 2026-08-23, before any feature was computed and before any
statistic was run.** §8 requires that every feature be justified by a prior
mechanism *before* it is tested, with the prediction logged. That is what this
entry is.

**Why now.** §4 ordered the work by cost-to-falsify and the flow branch has been
run to its end: aggregate flow (H9), broker identity (H10/H11) and investor
class (H12) all returned nulls. Price and structural features have never been
tested, are free, need no broker data at all — and therefore run at **daily**
resolution on the **full ~1,000-name spine** rather than fortnightly on 176.
That is far more power than anything the flow branch had.

**Design.** Identical to Gate 1's, deliberately, so the results are comparable:
per-day cross-sectional Spearman IC of the control-neutralised feature against
the forward return; Newey–West HAC t on the IC series; a 200-draw permutation
null shuffling the feature within each day; IC by liquidity decile; costs
charged to the quintile spread at A5's **56 bps** round trip plus the
point-in-time fraksi-harga half-spread. Locked (ARA/ARB) and stale bars are
excluded — a day at ARA is a day you could not buy.

**Controls:** `mom12_1`, `rev1`, `log_turnover`, `vol60`, plus 5 statistical
factors standing in for the unavailable sector control. When a tested feature
*is* one of the controls it is left out of its own control set, and that is
stated in the output.

**Horizons:** k ∈ {1, 5, 10, 20} trading days. The **full decay curve is
reported**, never the best k. The confirmatory statistic is pre-specified at
**k = 5**.

**Holdout:** the most recent 24 months stay untouched, exactly as H9 left them.

### The eight features, their mechanisms, and their predicted signs

| # | feature | mechanism (stated before testing) | predicted IC |
|---|---|---|---|
| 1 | `rev5` = −return(5d) | short-horizon liquidity provision: demanders of immediacy push price away from value, which reverts as inventory unwinds | **+** |
| 2 | `mom12_1` | slow diffusion of information; under-reaction to news | **+** |
| 3 | `lowvol` = −vol60 | leverage-constrained investors bid up high-volatility names, so low vol earns the premium (betting-against-beta) | **+** |
| 4 | `amihud60` = mean(\|ret\|/value) | illiquidity premium — compensation for price impact | **+** |
| 5 | `volz20` = volume z-score | attention-induced buying: retail buys what is salient, and salience-driven demand reverses | **−** |
| 6 | `hi52` = close / 252d high | anchoring on the 52-week high causes under-reaction near it | **+** |
| 7 | `atr_mom20` = ret20 / ATR20 | risk-adjusted momentum: volatility-normalising makes the cross-section comparable | **+** |
| 8 | `squeeze` = ATR20 / ATR250 | **NEGATIVE CONTROL.** Range compression predicts the SIZE of the next move, not its sign, so it has no signed cross-sectional return prediction. | **0** |

Feature 8 is included deliberately as a feature that *should* come back null.
A pipeline that finds signal everywhere, including where no mechanism predicts
it, is finding its own artefacts. This is the check for that.

**Gate:** the same conjunction §7 states — post-cost, liquidity-filtered,
control-neutralised IC significantly non-zero, with a stable sign, out of
sample. All four, or it fails.

**Trial count: 8 confirmatory trials.** Trials after H13 will be **38**, so the
Bonferroni bar is α = 0.05/38 = **0.0013**.

### H13 — RESULT

**Run 2026-08-24.** 891 names (89 delisted), 5,909 days, **1,989,504** tradeable
in-sample rows, 2000-03-30 … 2024-08-21. Holdout untouched.
Full memo: `reports/phase3_price_features.md`.

**Result: Gate 1 fails on COST, for all eight — the opposite failure to the
flow branch, which failed on signal.**

| feature | pred | IC (k=5) | HAC t | liq sign | net %/yr k=5 | net %/yr k=20 |
|---|---|---|---|---|---|---|
| lowvol | + | +0.0398 | **+14.49** | stable | −116.7 | −36.7 |
| mom12_1 | + | +0.0274 | +10.75 | stable | −81.5 | −14.6 |
| rev5 | + | +0.0270 | **+14.09** | stable | −72.1 | −18.9 |
| hi52 | + | +0.0261 | +10.13 | flips Q1 | −92.6 | −21.0 |
| amihud60 | + | +0.0169 | +5.98 | flips Q1 | −96.4 | −25.1 |
| volz20 | **−** | +0.0136 | +7.18 | stable | −66.9 | −7.4 |
| squeeze | **0 ctrl** | +0.0083 | +3.55 | stable | −84.8 | −18.0 |
| atr_mom20 | + | +0.0061 | +3.32 | flips Q1 | −77.7 | −12.5 |

Every IC lies outside all 100 permutation draws. Five carry HAC t > 10 — H9's
flow signal managed −2.86 on the identical statistic. **And every net cell is
negative.** Cost per rebalance is 1.7–1.9% against a gross quintile spread of
0.15–0.36% per period.

**THE NEGATIVE CONTROL FIRED, and it is the most useful thing in the run.**
`squeeze` was registered as predicted-null (range compression forecasts the
SIZE of the next move, not its sign). It returns t = **+3.55**. That is not a
defect in the feature — it says that **at two million observations a
t-statistic is nearly free**, and an IC of 0.008 clears any conventional bar
while meaning nothing economically. Every t in the table must be read through
it. This is why the control was registered.

**`volz20` failed its registered sign** — predicted negative on an
attention-buying mechanism, observed +0.0136. Logged as failed.

**A rank tilt is not a return spread.** `lowvol` shows IC +0.0398 (t +14.49)
with a quintile spread of **−0.430%/period (t −5.24)**: robust rank correlation
across the whole cross-section, negative in the tails where a few high-vol names
deliver huge returns. When the two disagree the spread decides tradeability.

**The decay curves have the shapes their mechanisms predict** — reversal decays
with k (0.0270 → 0.0155), momentum and 52-week-high strengthen (0.0157 →
0.0417, 0.0162 → 0.0375). Evidence the features are real rather than fitted.

**EXPLORATORY, not in the trial count: the liquidity-restricted sweep.**
Restricting to the top 5% by turnover halves cost (1.77% → 1.02%) and collapses
the t five- to sevenfold. Two of 24 cells cross zero — `mom12_1` k=20 at
**+1.6%/yr (t +2.00)** and `hi52` k=20 at **+2.0%/yr (t +0.95)**. Not a
finding: a 24-cell post-hoc search, neither t clears the 3.21 bar, +2%/yr is
not a business, and the quintile spread is long-short while **A5 forbids
shorting**, so even those cells describe a portfolio this project cannot hold.

**Trials after H13: 38.** Bonferroni bar α = 0.05/38 = **0.0013**.

---

## H14 — does a broker's STYLE persist even though its EDGE does not? (PRE-REGISTERED)

**Registered 2026-08-24, before any fingerprint was computed.** H11 established
that a broker code's *margin rank* does not persist (year-over-year +0.032,
inside its null). It said nothing about whether a broker's **behaviour** is
stable, and those are different claims: a firm can have a completely stable
business model — always crossing, always patient, always concentrated in three
names — while having no stable edge at all. §9.4–9.6 ask for exactly that
fingerprint, and §9.5 attaches two falsifiable checks to it.

**The fingerprint,** per (broker, year), from the fortnightly range store —
89 codes, 217 tickers, 2014–2026. Each metric is in §9.4 and is computable
without the footer the range files lack:

| metric | §9.4 group | what it means |
|---|---|---|
| `cross` | Positioning | min(buy,sell)/max(buy,sell) value — high implies market-making churn, low implies directional conviction |
| `hhi` | Positioning | Herfindahl of gross value across tickers — specialist vs generalist |
| `edge_buy` | Execution | (visible VWAP − buy avg)/VWAP — positive means patient, negative means paying up |
| `edge_sell` | Execution | (sell avg − visible VWAP)/VWAP |
| `ar1` | Horizon | AR(1) of the sign of net-buy across windows — persistence vs alternation |
| `share` | size | log share of total visible gross |
| `censor` | data artefact | fraction of rows one-sided under the top-10 cut |

**The VWAP here is a VISIBLE-broker VWAP**, value over lots across the brokers
the top-10 cut shows, not IDX's published one — `pullback_flow.fetch_window`
never kept the footer (A7). It is self-excluded per §9.4's mandatory bias
correction. It is named `visible_vwap` everywhere so it is never mistaken for
the real thing, and `censor` is carried alongside precisely because it bounds
how wrong it can be.

### Three questions, three predictions, all registered now

**Q1 — does style persist?** Year-over-year rank correlation of each metric,
against the identical 200-draw within-ticker-window label shuffle H11 used.

> **Prediction: style persists far more strongly than margin's +0.078.** A
> fingerprint reflects a firm's business model rather than skill, and H11
> already found one structural attribute — `censor` — persisting at **+0.801**.
> I expect `cross`, `hhi` and `share` above +0.5, and the execution-edge metrics
> weaker but still above margin.

**Q2 — does distinctiveness DEGRADE?** §6.3 says large players split orders
across brokers precisely because the summary is public, and §9.5 says to plot
distinctiveness by year and "report it prominently rather than averaging it
away". Measure: mean pairwise Euclidean distance between brokers' standardised
fingerprints, per year, plus each metric's cross-sectional dispersion.

> **Prediction: distinctiveness declines over 2014–2026.** If it does, that is
> a real finding about the dataset's shelf life. If it does not, §6.3's premise
> is not visible at this resolution and that is worth saying too.

**Q3 — are archetypes stable?** §9.5's mandatory check: fit clusters on
**2014–2019**, assign **2020–2026**, and measure whether assignments persist.
Both HDBSCAN and GMM, compared. Also whether the *number* of stable clusters is
stable.

> **Prediction: assignments persist for the structural metrics and the cluster
> COUNT is not stable.** I expect broad separation (a churn/market-making group
> vs a directional group) to survive and finer splits not to.

**§9.6 dossiers are conditional on Q3.** If archetypes do not prove stable,
the honest output is "no stable archetype" and no dossier is written. §9.6's
own rule — below the minimum sample threshold, write "insufficient data" and
move on — governs. Fabricating a behavioural read is the specific failure mode
that section exists to prevent.

**Trial count: 3 confirmatory trials (Q1, Q2, Q3).** Trials after H14 will be
**41**, so the Bonferroni bar is α = 0.05/41 = **0.0012**.

### H14 — RESULT

**Run 2026-08-24.** 89 codes, 13 years, 1,015 broker-years, 2014–2026.
Full memo: `reports/phase2b_fingerprints.md`.

**Q1 — style persistence. THE NULL INVERTED THE ANSWER.**

| metric | observed | null | distance | verdict |
|---|---|---|---|---|
| `censor` | +0.845 | +0.484 ± 0.025 | **+14.6 sd** | real, but a data artefact |
| `cross` | +0.521 | +0.245 ± 0.038 | **+7.3 sd** | **real style persistence** |
| `edge_buy` | +0.104 | +0.009 ± 0.035 | +2.7 sd | weak |
| `hhi` | +0.603 | +0.575 ± 0.014 | +2.1 sd | weak — mostly artefact |
| `edge_sell` | +0.058 | +0.010 ± 0.031 | +1.6 sd | nothing |
| `ar1` | +0.009 | +0.002 ± 0.032 | +0.2 sd | nothing |
| `share` | **+0.912** | **+0.919 ± 0.002** | **−2.5 sd** | **BELOW its own null** |

**`share` persisting at +0.912 is not a finding — its null is +0.919.** The
shuffle permutes labels *within* each ticker-window, so every code keeps the
exact set of windows it appeared in; a broker present in 5,000 windows still
draws 5,000 times, so annual gross is driven by presence, which is conserved by
construction. `hhi` fails identically (+0.603 vs +0.575). **Reading either
against zero gives a confident wrong answer — the fourth such occasion in this
repo**, after H9's broken null, H10's WAC bug and H11's off-centre Track A null.

**Prediction partly failed.** I registered `cross`, `hhi`, `share` above +0.5,
which the raw numbers satisfy — but only `cross` survives its null. The honest
answer is narrower: **the shape of the book persists; nothing resembling skill
does.**

**Q2 — degradation. PREDICTION FAILED: there is no trend.** Distinctiveness
runs 3.41 (2014) → 3.48 (2026); slope −0.0026/yr, rank correlation with year
**+0.132** (wrong sign for a decline), last above first, full range 7.4% of the
level. §6.3's order-splitting is not visible at fortnightly top-10 resolution.
An earlier output labelled this "DECLINING" from the slope alone while printing
endpoints that contradicted it; the verdict now requires slope, rank
correlation and endpoints to agree.

**Q3 — archetypes. THEY DO NOT EXIST.** **HDBSCAN finds zero clusters on the
early era and labels 100% of codes noise.** Forced partitions confirm it: GMM
k=2 agrees across eras 77% against 63% chance, KMeans 83% against 67%, both
degrading toward chance as k rises. k-means and GMM always return k clusters
and cannot report that there are none; HDBSCAN can, and did. Prediction
CORRECT and stronger than registered.

**§9.6 dossiers: NOT WRITTEN**, per the pre-registered conditional. §9.6 is the
section most exposed to fabrication, and the conditional existed so the
decision could not be revisited after seeing the answer.

**Trials after H14: 41.** Bonferroni bar α = 0.05/41 = **0.0012**.

---

## O1 — 2026-08-24 — run-state conditioning beats its null. AN OBSERVATION, NOT A TEST.

**Logged as O rather than H on purpose.** This came out of building the daily
brief (`reports/daily_brief.md`), not out of a pre-registered question, and
nothing about it was predicted in advance. Recording it as a hypothesis tested
would be backdating a registration, which is the exact move §11's trial count
exists to make impossible.

**What was fixed before any cell was seen:** four conditioning dimensions —
leg direction, run-age tercile, extension tercile (`run_z`, cut separately per
leg), and index-vol tercile. 54 cells. Outcome: mean forward 20-session return
minus the equal-weighted mean over all liquid names on the same dates.
Reference sample 1,127,670 liquid **pre-holdout** bars, 2001–2024.

**What was NOT fixed in advance:** which cells would matter. That is the whole
problem with reading the result.

| | |
|---|---|
| largest cell excess over base | **+1.67%** / 20 sessions |
| null (state labels shuffled within date), 200 draws | **+0.37%** mean, +0.53% p95 |
| cell spread observed vs null | **0.68%** vs **0.13%** |
| p(null ≥ observed) | **0.000** |

**So the conditioning is real.** Old stretched advances continued (+1.0% to
+1.7% over base, intervals excluding zero in all three vol regimes); advances
old in time but shallow in price underperformed (−0.7% to −1.0%). Coherent,
and sign-consistent with H13's momentum features.

**Four reasons it is not a result:**

1. **It is the maximum of 54.** The null says *some* cell is extreme; it does
   not license reading the largest, which is biased upward by the selection
   that found it. ~3 of 54 uncorrected intervals clear zero by luck.
2. **The cells were not registered.** Only the dimensions were.
3. **Entirely in-sample.** The 24-month holdout is untouched and stays so — a
   brief running twice a day would otherwise spend it inside a week.
4. **H13 measured very nearly this and found it net-negative.** `mom12_1`,
   `hi52` and `atr_mom20` all encode "old stretched advance". The construction
   differs (long-only excess over the cross-sectional mean here; a
   control-neutralised quintile spread there) and that difference is exactly
   the kind that manufactures an effect. The burden is on the new one.

**What would turn this into H15:** pre-register the specific cells, the
long-only rule, the holding period and the cost model in writing; then spend
the holdout once. Registering after reading the table above does not count.

**Trial count unchanged at 41.** No hypothesis was tested here. The 54 cells
are recorded so that a future H15 inherits them rather than pretending its
cells were the first ones looked at.

---

## H15 — 2026-08-25 — the forecast question, asked properly, and answered

**"I want you to forecast" is two questions.** Whether the brief can produce a
calibrated probabilistic forecast, and whether that forecast is worth trading.
They have different answers and conflating them is how a good forecaster gets
mistaken for an edge. Both are settled here.

### The economic question: closed WITHOUT spending the holdout

O1 recorded a state-conditioning effect whose largest cell was +1.66% per 20
sessions. The pre-registered version — **the whole family `leg=up, extension
tercile=2`, nine cells, motivated a priori as "stretched advances continue"
rather than picked as the argmax** — measures:

| | |
|---|---|
| gross excess over the liquid equal-weighted universe | **+0.586%** / 20 sessions |
| non-overlapping rebalance periods, in sample | 271, sd 3.30%, **t +2.92** |
| round-trip cost at the observed median price | **0.90%** |
| **net** | **−0.314%** |

**The a-priori family is a third of the argmax cell.** That gap is the
selection bias O1 warned about, measured rather than asserted.

**And the holdout could not have rescued it.** 481 sessions = **24
non-overlapping periods**. At the in-sample effect size, power is **12.5%** at
nominal α and **0.5%** after Bonferroni over 42 trials. The minimum detectable
excess at 80% power is **+1.97%** per period nominal, **+3.07%** Bonferroni —
three to five times the effect that exists, and the effect that exists is
already net-negative before it gets there.

**So the holdout is NOT spent.** Running it would have burned the only clean
sample on a test that could not answer the question. `scripts/power.py`
reproduces the calculation; it touches no holdout row.

### The forecasting question: real resolution, no skill

Walk-forward with a 20-session embargo, 5 folds, 1,059,858 scored forecasts.

**The first attempt was WORSE THAN USELESS and the verification caught it.**
Raw cell frequencies scored a **Brier skill of −0.0093** — worse than saying
"45%" every day — with a reliability curve predicting **99.4% where reality was
58.9%**. Resolution was genuinely non-zero, so the conditioning knew something;
overconfidence threw it away, because a cell holding a few hundred bars was
quoted like one holding forty thousand.

**Shrinkage repairs it, and the prior it chooses is the finding.** Empirical
Bayes toward the base rate, prior strength selected by inner walk-forward on
training folds only:

| | raw | shrunk |
|---|---|---|
| Brier skill vs climatology | **−0.0093** | **−0.0008** |
| skill vs walk-forward base rate | −0.0077 | **+0.0008** |
| calibration error | 0.00230 | **0.00044** |
| resolution | 0.00041 | 0.00024 |
| resolution vs shuffle null | p = 0.000 | **p = 0.000** |
| predicted range | 25% → 100% | **40.8% → 51.1%** |

**The chosen prior was 10,000–30,000 pseudo-observations.** Given a free choice
across seven orders of magnitude, the data said *shrink almost all the way to
climatology*. An independent check: the optimal shrinkage weight
τ²/(τ²+σ²) at the panel's signal-to-noise is ~0.17, equivalent to a prior near
10,000. The cross-validation and the algebra agree.

**So: the forecasts carry real resolution (p = 0.000 against a within-date
shuffle) and essentially zero skill (+0.0008).** The most confident statement
this model will ever make is **51% against a 45% base rate**. That is an honest
forecast and it is not an edge.

**Trials after H15: 43.** Bonferroni bar α = 0.05/43 = **0.0012**.
**The 24-month holdout remains untouched.**

---

## H16 — 2026-08-25 — THE HOLDOUT IS NOW SPENT. Backtest of the multiplier-cell rule.

**§11 said touch it once. This is the once.** Asked what the published
multiplier-cell rule would have picked on 2025-08-25 and whether those names
doubled, the only honest way to answer was to run it on holdout data. The rule
was published in full the previous turn and could not be retro-tuned, and the
cells themselves are fit on pre-holdout rows ending 2024-08-23, so the
selection is genuinely out of sample. **Any future test on this window is now
compromised.**

### The cohort asked about: 2025-08-25 → 2026-08-24

| | | | |
|---|---|---|---|
| ENRG **+118.8%** | ARCI **+115.4%** | FORE +43.8% | MBMA +35.8% |
| INET +27.9% | DKFT −6.6% | KRAS −20.8% | BLOG −33.6% |
| MERI −54.8% | KRYA −74.6% | | |

**2 of 10 doubled = 20%.** 5 positive, 2 halved. Median +10.6%, mean +15.1%.
All ten still trading at the end, so no delisting censoring in this cohort.

**Calibration was good.** The cell predicted P(2x) = 15.2% → 1.5 expected, 2
realised; P(−50%) = 26.6% → 2.7 expected, 2 realised. The model said what
happened.

**THE EDGE OVER RANDOM IS NOT ESTABLISHED.** Against 20,000 random 10-name
draws from the same 267-name liquid universe:

| statistic | picks | null mean | p |
|---|---|---|---|
| count >100% | 2 | 0.86 | **0.211** |
| mean | +27.0% | +8.2% | 0.184 |
| median | +31.9% | −8.4% | **0.026** |

Only the median clears. **Picking 10 names at random from the same liquid
universe produced 2 or more doublers 21% of the time.** On this date the rule
is indistinguishable from a coin.

### Pooled across 12 monthly cohorts — and why it is still n ≈ 1

| | rule | universe |
|---|---|---|
| doubled | **27.5%** (33/120) | 14.1% |
| drop the best cohort | 23.6% (26/110) | 13.6% |
| mean of cohort medians | +28.9% | +4.9% |
| doubling edge positive in | **10 of 12 cohorts** | |

The drop-largest-cohort check — A8 says it belongs in every pooled statistic
here — leaves the gap intact, and the edge is positive in 10 of 12 cohorts.

**But the cohorts overlap almost completely.** Twelve monthly start dates each
holding one year share ~11/12 of their forward window, so the consistency
across cohorts is largely mechanical: they hold nearly the same names over
nearly the same period. **The holdout contains ONE independent year-window.
Effective n is ~1, not 12 and emphatically not 120.**

**And that year was generous.** The liquid universe doubled at 14.1% against a
long-run pre-holdout base rate of 9.6%. The whole test sits in one bull regime
for the speculative segment.

### What this does and does not license

**Does:** the cell base rates are calibrated out of sample. When the table says
15% double and 27% halve, roughly that happens.

**Does not:** any claim that the rule beats picking liquid names at random.
p = 0.21 on the date asked, and n_eff ≈ 1 pooled.

**One structural observation worth carrying.** At a one-year horizon the cost
wall that killed H9/H12/H13 is nearly irrelevant — a 1–2% round trip against a
+15% median is not the binding constraint it is at 20 sessions. That is the
"hold longer" lever, and it is the only one this repo has not run to its end.

**Trials after H16: 44.** Bonferroni bar α = 0.05/44 = **0.0011**.
**The 24-month holdout is SPENT.**

---

## H17 — does an exit rule improve the multiplier-cell entry? (2026-08-25)

**Registered before running the walk-forward:** the catalogue of 32 exit rules
was fixed in `exits.catalogue()` before any cohort was scored; the selection is
purged walk-forward, so the reported number is never scored on the sample that
chose it. Memo: `reports/exit_rules.md`.

**Prediction.** H16's attribution — mean peak +102.2%, realised +15.1% — says
the give-back is the loss. A trailing stop should recover part of it. It should
NOT reduce P(−50%), because a name that falls from entry never arms a trail.

**Result, 176 purged cohorts, 2008-09 → 2023-08, tie='all' baskets:**

| | cohort median | 95% CI (moving-block) |
|---|---|---|
| walk-forward selected rule | **+0.94%** | [−8.11%, +10.49%] |
| buy and hold 252 | −3.19% | [−13.94%, +6.38%] |
| **difference** | **+4.13%** | **[+1.87%, +6.33%]** |

Differed from buy-and-hold in 94 of 176 cohorts and **won 74 of those, 79%,
sign test p = 1.8 × 10⁻⁸**. Mean +14.24% when it wins, −16.34% when it loses,
so the median difference is exactly 0.00% — the win RATE is the effect, not a
tail. 153 of 176 cohorts chose `trail 15% armed +50%`; mean hold 210 sessions.

**The prediction about the downside held exactly.** P(−50%) 15.0% against
16.3%. Every armed-trail variant reads an identical 15.1% because the loss
happens before the arm. The frontier table in the memo shows a hard stop takes
P(−50%) to ~0% at a cost of 6–8 points of median — for a rule selected ON
P(2x), cutting the left tail cuts the premise.

**Trials after H17: 46** (the 32-rule catalogue is a pre-registered search
space resolved by walk-forward, not 32 separate trials; a reader who prefers to
count it as 32 gets a Bonferroni bar of 0.05/78 = 0.00064, which the sign test
still clears by four orders of magnitude). **The holdout remains SPENT.**

---

## H17b — is the multiplier-cell ENTRY reproducible at all? (2026-08-25)

**Not a prediction; a defect found and then measured.** Two independent
implementations of the same published entry rule returned different baskets on
2025-08-25 (IMPC vs MERI) and therefore different returns (+26.3% vs +15.1%)
for the identical buy-and-hold exit.

**Cause.** ≤125 cells against ~800 live names, so cell scores are massively
tied. On 2025-08-25 **17 names shared the 10th-place score** and the top 30
held **4 distinct scores**. "Top ten" was decided by frame order.

**Measured, 500 random tie-breaks on the 2025 cohort:**
p5 −16.1%, median **+6.9%**, p95 +26.9%, sd **13.0%**, range −29.2% to +36.6%.
H16's draw sits at the ~75th percentile, exit_study's near the 95th.

**Measured, 211 pre-holdout cohorts × 40 tie-breaks:** the cut falls inside a
tie in **64%** of cohorts; within-cohort sd **4.31%**; middle-90% span
**10.92%**.

**So H16's headline is revised down.** The centre of that cohort is +6.9%, not
+15.1%. The 2/10 doublers stands as an observation about one arbitrary draw.

**Fix:** `multiplier.select(tie="all")` holds the whole tied group. Measured
−4.42% against tie='first''s −2.54% over 211 cohorts, difference −1.47%
[−2.48%, −0.56%] — accepted, because a reproducible −4.4% is worth more than
an irreproducible −2.5%.

---

## H18 — do indicator-conditioned exits beat fixed price-path ones? (2026-08-25)

**Pre-registered before any scoring**, in `scripts/exit_indicators.py`'s
docstring and `exits.indicator_catalogue()`. Memo: `reports/exit_indicators.md`.
26 indicator rules added to H17's 32 price rules; same purged walk-forward,
same cohort-level moving-block bootstrap, 176 cohorts 2008-09 → 2023-08.

**News was excluded and cannot be included.** No point-in-time archive exists;
`tests/test_news.py` fails the build if `spine/` or `features/` imports the news
module. A news-conditioned exit in a backtest is look-ahead by construction.
It ships as a live-only overlay in `scripts/positions.py`, computed into nothing.

### H18a — a volatility-normalised trail beats a fixed percentage one

**SUPPORTED.** Best chandelier **+2.9%** cohort median against the best fixed
trail's **+0.9%**; the walk-forward chose a chandelier in 56 of 176 cohorts.
Mechanism as registered: the entry selects for high realised vol, so a fixed
15% band is a different rule for every name it picks.

### H18b — an indicator stop cuts P(−50%) more cheaply than a hard stop

**FAILED.** The undominated (P(−50%), P(2x)) frontier is *entirely* price
rules. Every armed indicator rule reads P(−50%) = 15.1%, identical to the armed
trails — a name that falls from entry never arms. Unarmed indicator rules reach
P(−50%) ≈ 0.1% only by exiting in 10–17 sessions with P(2x) at 0.7–1.1%.
Logged as failed, not reframed.

### NULL — a random exit should behave like a matched-length hold

**It did**, and it earned its keep twice:

| | median | P(2x) | days |
|---|---|---|---|
| NULL random exit | −4.6% | 6.0% | 122 |
| `hold 126` | −5.0% | 6.8% | 125 |

**And it beat every hard stop** (−4.6% vs −10.6% to −11.7%). A coin-flip exit
date does better than a 15–30% hard stop on this entry.

### THE RESULT — the objective decides the answer

Same walk-forward, same 58 rules, three selection targets:

| objective | median | mean | **P(2x)** | P(−50%) | days | most-chosen |
|---|---|---|---|---|---|---|
| median | **+3.16%** | +7.10% | **3.5%** | 14.8% | 195 | stoch rollover armed +50% |
| mean | −2.86% | **+18.42%** | 11.0% | 15.8% | 239 | volume climax armed +50% |
| p2 | −3.36% | +19.97% | **11.6%** | 16.1% | 247 | volume climax z3 armed +50% |
| buy&hold | −4.30% | +18.80% | 11.6% | 16.3% | 250 |

**No rule in the catalogue beats buy-and-hold on mean return or on P(2x).**
Optimising the median cuts the doubling rate from 11.6% to **3.5%**. The whole
measured improvement is a median effect. For an entry selected ON P(2x),
selecting its exit on median return optimises against its own premise — and
H17's headline made exactly that choice without stating it.

**Headline, objective stated:** on `median`, +3.16% vs buy-and-hold −3.19%,
difference **+6.35% [+3.57%, +9.08%]**, 85/176 cohorts (86% of the 99 that
differ), sign test p = 1.4e−13.

**Versus H17's incumbent** (`trail 15% armed +50%`, same cohorts): **+1.21%
[+0.07%, +2.42%]**, 37/55 differing, sign test **p = 0.014** — which does NOT
clear the Bonferroni bar. Suggestive, not established.

**On the 2025 cohort** (spent holdout, certifies nothing): the trail returned
mean +16.0% with **2 doublers**; the median-optimal indicator rule returned
+8.3% with **0 doublers**. Same trade-off, one date.

**Trials after H18: 49.** Bonferroni bar α = 0.05/49 = **0.001**.
**The 24-month holdout remains SPENT.**

---

## H19 — where does the recovery edge actually die, and do indicators know? (2026-08-25)

**Pre-registered in `scripts/recovery.py`'s docstring before any of it ran.**
243,977 armed liquid pre-holdout bars, 560 names, 279 months, 2001-04 → 2024-08,
forward horizon 60 sessions. Memo: `reports/recovery_curve.md`.

**Motivation:** H17 and H18 both chose a trail distance by GRID SEARCH and never
measured the conditional the distance is supposed to encode.

### H19a — P(new high) falls monotonically with give-back depth

**HOLDS.** 81.3% → 56.1% → 38.8% → 27.1% → 17.3% → **11.3% at −30%** → 5.9% at
−40% → 0.8% below −60%. Crosses one-half between −5% and −10%.

### H19b — there is a depth where the mean forward return turns negative, deeper than 15%

**FAILED, and instructively.** The mean NEVER turns significantly negative at
any depth: it stays between −2.2% and +7.1% and its interval covers zero from
−25% down. A thin tail of enormous recoveries keeps expected value roughly flat
all the way to −60%.

What turns negative is the **MEDIAN**, at **−10% to −15%** (−0.5%). So the
typical outcome sours at 10–15% while the average outcome never does — the H18
mean/median wedge arriving from a completely different direction.

**This independently validates the 15% nobody had verified.** The grid picked
15% by optimising cohort median; the conditional curve puts the median crossing
at −10 to −15%. Two unrelated routes, same number. In ATRs: −10% = 2.1 ATR,
−15% = 3.2 ATR, which is where `chandelier 2x ATR` sits.

Mean further drawdown over the next 60 sessions widens monotonically with depth
(−12.5% at −10%, −16.1% at −30%, −22.4% below −60%), so holding deeper costs
extra pain even where it does not cost expected value.

### H19c — at matched depth, indicators separate recoverers from non-recoverers

**SPLIT, and the split is the finding.**

| outcome | cells clearing Bonferroni (0.0009) | sign split | median effect |
|---|---|---|---|
| **P(new high)** | **14 / 53** | **13 pos / 1 neg** | **12.6 points** |
| forward return | 4 / 53 | 2 pos / 2 neg | 4.2% |

**Indicators predict WHETHER it gets back to the high, strongly and
consistently. They do not predict the RETURN in any stable direction.**

By indicator, sign consistency on P(new high): **above EMA50 9+/0−** (mean
+7.9pts), above EMA20 9+/1− (+3.7), give-back < 2 ATR 2+/0− (+15.4), turnover
z>0 7+/4− (+3.8), stoch %K>50 8+/2− (+1.4), **stoch %K>%D 5+/6− (+0.2 — noise)**.

Concentrated where it matters: average shift +17.7 pts at −10 to −5%, +11.9 at
−15 to −10%, decaying to +2–4 pts past −20%.

**Consequence:** `ema50 break armed +50%` is the rule the curve endorses — H18's
table gives it +0.8% median, **+15.0% mean, P(2x) 7.9%** against the trail's
+0.9% / +11.4% / 7.3%, same P(−50%). Best-motivated, NOT out-of-sample: it is
read off a full-sample table already seen, and the holdout is spent.

**Two method defects, both mine, both fixed.** The null permuted ROWS when the
information varies at (ticker, month) — 20 near-identical bars priced as 20
observations, inflating every z (one read −8.7). And a cell with a flag true for
0.2% of rows produced the first table's largest effect (−46.1%); both sides now
need 300 observations and a 5% share.

**Trials after H19: 52.** Bonferroni bar α = 0.05/52 = **0.00096**.
**The 24-month holdout remains SPENT.**

---

## H20 — what you actually earn, and does the entry earn any of it? (2026-08-25)

**Pre-registered in `scripts/portfolio.py` before scoring.** 212 cohorts,
27,451 priced name-cohorts, 658 names, 2004-12 → 2023-08, pre-holdout only.
Memo: `reports/portfolio.md`.

**Motivation — two critiques of H17–H19, both mine.** (1) The cohort MEDIAN is
not what an equal-weighted holder receives; the MEAN is. H17 and H18 both
selected on the median. (2) The random-entry control had never been run against
the exit layer.

### H20a — the exit gap largely survives random entry

**FAILED.** Best exit vs hold: **+3.18%** CAGR on the multiplier basket,
**+0.73%** on random baskets — only **23%** survives. The exit's apparent
benefit is basket-specific, not a general property of volatile names.

### H20b — no rule beats buy-and-hold on terminal wealth; all reduce drawdown

**FAILED on both halves of the prediction.** 3 of 6 beat hold on terminal
wealth, and only 4 of 6 reduce max drawdown. `stop 25%` has a per-name
P(−50%) of **0.2%** and the **worst portfolio drawdown of all, −68.7%**,
because it realises losses 29 times against hold's 18 and redeploys into the
same regime. **Per-position stops can increase portfolio drawdown.**

### H20c — the growth-optimal rule differs from mean- and median-optimal

**SUPPORTED.** Growth-optimal (mean log) `stop 25%`; mean-optimal `hold 252`;
median-optimal `chandelier 2x ATR armed +50%`. Three objectives, three answers.

### THE RESULT — the half-split ends it

Paired per slot vs buy-and-hold, inside each half independently:

| rule | early ΔCAGR | wins | late ΔCAGR | wins | both? |
|---|---|---|---|---|---|
| trail 15% armed +50% | −9.41% | 1/12 | +0.53% | 7/12 | no |
| trail 30% armed +50% | +3.85% | 9/12 | +1.69% | 6/12 | **YES** |
| chandelier 2x ATR armed +50% | −4.52% | 1/12 | −4.97% | 2/12 | no |
| stoch rollover armed +50% | −8.60% | 0/12 | −4.58% | 2/12 | no |
| ema50 break armed +50% | −3.16% | 5/12 | +6.73% | 10/12 | no |
| stop 25% | +2.83% | 10/12 | −2.26% | 2/12 | no |

**Exactly 1 of 6 survives, which is BELOW the 1.5 chance predicts** if each
half were a coin flip. `trail 30%` is not evidence. `stop 25%` and
`ema50 break` win in OPPOSITE halves — regime noise, not signal.

**The entry, one pre-specified comparison, paired per slot (12 × 20 draws):**

| half | mean ΔCAGR | wins | win rate |
|---|---|---|---|
| early 2004-12 → 2017-08 | **+5.86%** | 204/240 | **85%** |
| late 2017-08 → 2023-08 | **+0.36%** | 122/240 | **51%** |
| full | +3.99% | 197/240 | 82% |

**Since 2017 the entry is a coin flip.**

### RETRACTIONS

**H17's +4.13% and H18's +6.35% are withdrawn as improvements.** On portfolio
CAGR the same rules deliver **+2.4%** and **+1.8%** against buy-and-hold's
**+10.5%** — they are the two WORST rules tested. They won a statistic nobody
is paid.

### The bug that changed the winner

The slot scheduler converted held sessions to calendar days and searched the
date index; a 30-day lock opened 1 February ends 3 March and **skips the
1 March cohort**. The penalty scales with turnover, biasing short-holding rules
down. Before the fix `stop 25%` won at 12/12 and t = +7.05; after it, it fails
the late half. Locking is now in cohort-index space and two tests pin it.

**Only stable findings: two NEGATIVE ones.** `chandelier 2x` and
`stoch rollover` are worse than holding in both halves. The second is H18's
own walk-forward pick.

**Trials after H20: 55.** Bonferroni bar α = 0.05/55 = **0.00091**.
**The 24-month holdout remains SPENT.**

---

## H21 — the benchmark H20 left out, and two critiques of its own conclusions

**2026-08-25.** Not pre-registered. It is the control H20 should have carried
from the start, and it is recorded here because it is **negative**: a benchmark
added after the fact is only suspect when it rescues a result, and this one
destroys the memo's conclusion. Code `scripts/portfolio_critique.py`, tests
`tests/test_portfolio_critique.py` (15). Raw `reports/portfolio_critique.txt`.

### C3 — every number in H20 is picks-versus-picks

H20 compared the picks against buy-and-hold on the same picks and against
random draws from the same pool. The IHSG — the thing that can be bought
instead, for one round trip — appears nowhere in it. `_JKSE.csv.gz` was in
`data/cache/ohlcv/` the whole time.

**Making it like for like, and both corrections favour the picks.** The name
returns are on `adj_close`, so they are TOTAL returns; `^JKSE` is a PRICE
index. The adjustment identifies itself: `log(adj_close/close)` steps only at
corporate actions, and back-adjustment makes dividend steps positive going
forward. Across 1.75m steps: **3,707 small positive, ZERO small negative.**
Mean annual dividend 1.27%, and it rises monotonically with liquidity —
0.65% in decile 1 to **2.01% in decile 10** — so the cap-weighted index yields
what large caps yield (1.77%), more than the picks' 1.27%. Correcting the index
UP is the larger correction and cuts the same way.

| window | picks TR | index TR | **gap** | picks maxDD | IHSG maxDD |
|---|---|---|---|---|---|
| early | +13.5% | +16.8% | **−3.3%** | −41.0% | −60.7% |
| late | +3.2% | +4.7% | **−1.5%** | −46.3% | −41.1% |
| full | +10.5% | +12.7% | **−2.2%** | −49.8% | −60.7% |

**The picks lose to the index in both halves and buy no risk reduction for it**
— shallower drawdown than the index over the full sample, *deeper* in the
recent half. Three uncorrected biases run against the picks (their figures are
net of cost, the index's is gross; 1.45% of holds end when the name stops
trading and are marked at last price, not zero, worth ~1.3 pts/yr) and one runs
for them (index products cost 0.5–1.0%/yr), so the honest gap is **−1.2% to
−1.7%**.

### C3b — and my own C3 table had two window mismatches, both flattering

The twelve slots begin and end in twelve different months, so one global index
window compares each slot to a period it did not occupy; and a slot's last
position is open for its holding period after the final entry, so its span runs
a year past the last cohort date while the index stopped at it. `slots()` now
returns `start`/`end` and each slot is paired against the index over its own
span. **It makes the result stronger:**

| window | picks | index TR | mean Δ | sd | won | 95% CI |
|---|---|---|---|---|---|---|
| early | +14.2% | +16.4% | −2.18% | 4.86% | 3/12 | [−4.93%, +0.57%] |
| late | +1.4% | +4.0% | −2.65% | 6.56% | 6/12 | [−6.36%, +1.06%] |
| **full** | +9.8% | +12.3% | **−2.53%** | 3.73% | **3/12** | **[−4.64%, −0.42%]** |

Nine of twelve slots lose to the index, and the full-sample interval excludes
zero. Twelve overlapping slots are not twelve independent trials so that
interval is too narrow — quoted because it is the reading most favourable to a
significance claim and the picks still lose it.

**But the claim does not rest on significance.** Even read as a tie, the picks
cost 18 round trips, single-name concentration and a −50% drawdown to reach the
same place. **A tie against the cheap alternative is a loss for the expensive
one.**

### C3c — the shortfall is SELECTION, not the toll

| | picks | vs index TR |
|---|---|---|
| net of cost | +10.5% | −2.53% |
| gross, every round trip refunded | **+12.3%** | **−0.82%** |

The 18 round trips cost **1.76%/yr** — most of the gap — but refunding all of
them still leaves the picks behind. A lower-turnover version of the same rule
does not rescue it.

### C3d — and it is NOT a small-cap handicap; that answer came back backwards

The obvious rescue is that the picks are equal-weighted mid-caps against a
cap-weighted large-cap index. Split by within-cohort liquidity tercile. **The
picked baskets fragment and cannot answer** — the liquid cell scores 54/212
cohorts at a median of 4 names. An earlier draft printed **−13.9%** for that
cell; it is a degenerate-cell artefact of the kind already recorded twice here
(smallest cell, largest effect) and is **not reported**.

A random draw of twelve IS a full basket in every tercile, so the segment
handicap is readable:

| tercile | random pool CAGR | vs index TR |
|---|---|---|
| thin | +7.9% | −3.54% |
| middle | +4.8% | −6.69% |
| **liquid** | **+3.7%** | **−9.48%** |

**The size story predicted the liquid tercile would close the gap; it is the
worst of the three.** Every tercile trails the index and moving upmarket makes
it worse. The shortfall is that an EQUAL-WEIGHTED basket of IDX names lost to
the CAP-WEIGHTED index over this sample — a handful of mega-caps carried it,
and even the pool's liquid tercile sits far below those in capitalisation, so
it inherits none of that return and keeps the equal-weighting penalty. The rule
leans thin (51% vs 15%), which given this table is the better end and is not
the source of the gap.

### C3d(b) — the rebuild was priced at 20 minutes, so it was paid

`scripts/liquid_rerank.py`, tests `tests/test_liquid_rerank.py` (6). Universe
restricted to the liquid tercile BEFORE ranking, so the rule takes a full
basket from that segment on every cohort. **The prediction that it would still
fail was registered in the script docstring before scoring.** 179 cohorts
2005-04 → 2023-08, 8,696 name-cohorts, 265 names, 2,084 picks, **median basket
11 (min 10), all 179 scored** — the thin-basket problem is gone.

| | |
|---|---|
| re-ranked picks CAGR | **−5.0%** |
| random draw, same liquid universe | −2.4% |
| edge over its own segment | **−2.6%** |
| index TR, same window | **+9.0%** |
| picks − index, paired per slot | **−15.08%** [−16.70%, −13.46%] |
| slots beating the index | **0 of 12** |

**The size explanation is dead.** Applied upmarket the rule goes from beating
its own pool by 4.8 points to TRAILING it by 2.6.

**AND THE PER-NAME BREAKDOWN IS THE SHARPEST DESCRIPTION OF THIS RULE IN THE
PROJECT:**

| | picks | rest of liquid universe |
|---|---|---|
| 1-yr return mean | −2.03% | −0.12% |
| 1-yr return **median** | **−10.48%** | −3.27% |
| **P(2x)** | **4.89%** | 1.51% |
| round-trip cost | 1.349% | 1.019% |

**It does exactly what it was built to do and loses money doing it** — more
than triples the doubler rate while the median pick falls 10.5%. It is a
LOTTERY-TICKET SELECTOR: it buys convexity and the convexity costs more than it
is worth. That is H16 seen from the other side (2/10 doublers, mean peak
+102.2%, realised +15.1%) and it is why no exit rule rescued it — the frontier
cannot cut the left tail without cutting the premise. It also picks
wider-spread names *within* the liquid tercile, 1.35% a round trip against
1.02%.

### C1 — "a coin flip since 2017" was a POWER statement written as an EFFECT one

A8's exact distinction, and H20 made the wrong side of it. Taking the SLOT as
the unit (averaging the 20 redraws within each slot first, which H20 did not):

| window | slots | mean ΔCAGR | 95% CI | smallest detectable |
|---|---|---|---|---|
| early | 12 | +6.08% | [+3.94%, +8.22%] | 2.14% |
| late | 12 | −0.01% | [−4.15%, +4.14%] | 4.14% |

The late interval **excludes +6.08%**, so H20's break claim survives — but it
**cannot rule out an edge of +4.1% a year**. "Coin flip" is too strong.

### C2 — 2017 is a decay, not a cliff

A cut chosen at the median finds a break somewhere. Rolling six-year windows
stepped a year: **14 of 15 positive**, +12.1% (2007–13) down to −0.6%
(2018–24), trend −0.45% per year of start date. Overlapping windows share five
years of six, so this is a shape, not fifteen measurements.

### Verdicts

- **C3 — H20's conclusion RETRACTED.** "The remaining defensible position is
  buy-and-hold on the picks" is false; the picks trail the index in both halves.
- **C1 — H20's break claim STANDS, narrowed.** Excludes the early effect, not
  a smaller one.
- **C2 — H20's framing CORRECTED.** Persistent decaying edge over the pool, not
  a switch thrown in 2017.

### And the fifteen tests H20 deleted without failing anything

H20's tests were written to `tests/test_portfolio.py`, **which already existed**
— 15 tests for `src/idxbot/portfolio.py`, a CLI-exposed module — and replaced
them. Both files held exactly 15, so the suite total did not move and no run
said anything had gone. `git status` said ` M` not `??`; that character was the
only warning. A module can go from covered to uncovered without any test
failing, because the evidence of coverage is the tests themselves.
`tests/test_coverage_map.py` now asserts every `src/idxbot/` module is named by
some test, with the two genuinely uncovered ones listed rather than hidden.
Suite 1,892 → **1,910**.

**Trials after H21: 58.** Bonferroni bar α = 0.05/58 = **0.00086**.
**The 24-month holdout remains SPENT.**

---

## H22 — can a rule tell you to be out of IDX before a correction?

**2026-08-25.** Nine timing rules on the IHSG, fixed before scoring. Asked
because the user asked; never asked before, since H9–H21 all ask which NAMES
to hold and none asks whether to be in the market at all. Code
`scripts/market_timing.py`, tests `tests/test_market_timing.py` (12). Memo
`reports/market_timing.md`.

**NOT ONE RULE BEATS HOLDING.** Buy-and-hold +10.32% CAGR; the best rule
(`above 100d MA`) +8.59%. Two rules beat holding at ZERO cost — `above 100d`
+11.54%, `above 50d` +11.01% — and lose once switching is charged at 0.28% a
side. The signal is real and the toll eats it: the same wall as H13 and H19
from a new direction.

**THE MATCHED NULL IS WHAT DECIDES IT.** A rule out of the market a third of
the time dodges a third of the crashes by construction, so a shallow drawdown
proves nothing alone. Against 200 random switchers with the SAME trade count
and time-in-market, three rules carry genuine information (`above 100d`,
`above 50d`, `1m momentum`) — **and all three still lose to not switching at
all.** Beating a coin flip about *when* to trade is not the same as trading
being worth it.

**THE HALF-SPLIT IS UNANIMOUS: 18 cells, all 18 negative.** No rule, no era.
This is the cleanest negative in the project — everywhere else at least one
cell survived by chance.

**WHAT IS KNOWABLE IS THE CONDITIONAL, AND IT POINTS THE OPPOSITE WAY TO THE
INTUITION.** P(index falls a further 5% within 20 sessions), base rate 20.7%:

| state | n | P | vs base |
|---|---|---|---|
| vol20 bottom quartile | 1,322 | **11.6%** | −9.0 pp |
| at/near 52w high | 1,755 | 18.1% | −2.5 pp |
| dd −10% to −20% | 963 | 26.6% | +5.9 pp |
| dd worse than −20% | 483 | **31.5%** | +10.8 pp |
| vol20 top quartile | 1,322 | **32.5%** | +11.9 pp |

A factor of three, and real. **Risk is highest when you are already down, not
when you are at a high** — so "sell before the correction" means acting on an
11.6% probability, and the states that do predict further falls are ones you
can only occupy after the fall has begun. That is precisely why the timing
rules fail: two-thirds of even the worst cell does not fall further, nowhere
near enough to carry 56 bps a switch.

**A drawdown-limiting rule is a legitimate PURCHASE, not an edge.** The golden
cross takes max drawdown from −60.7% to −23.9% for about 1.7% a year.

**Trials after H22: 67.** Bonferroni bar α = 0.05/67 = **0.00075**.
**The 24-month holdout remains SPENT.**

---

## H23 — the horizon, and the first construction here that beats the index

**2026-08-25.** Asked because the user set "make 8 of 10 reachable" as the
goal. Every P(2x) in this repo is measured at 252 sessions because H16 chose
that horizon; A9 named a long horizon as untested and it stayed untested.
Code `scripts/horizon_sweep.py`, tests `tests/test_horizon_sweep.py` (11).
Memo `reports/horizon.md`.

**THE CEILING: 80% IS NOT REACHED AT ANY HORIZON.** Unconditional P(touch 2x),
deaths at last price: 9.5% (1y), 18.9% (2y), 27.0% (3y), 39.0% (5y), 46.7%
(7.5y), **55.5% (10y)**. So 8 of 10 needs an 8.4x lift at one year and a
**1.44x** lift at ten — the smallest gap available.

**TWO BUGS MADE THE FIRST TABLE READ 73.8%.** `MU.PX` is a list of CUT EDGES,
not a [min,max] pair, so `close >= PX[0] & close <= PX[1]` restricted the
universe to sub-Rp50 names — the penny board, 336 tickers. And eligibility was
applied to EVERY bar, cutting the forward path when a name left the universe;
eligibility is a condition for BUYING. Requiring a full window on top of that
discarded **91% of 7.5-year cohorts** and measured the survivors.

**NOTHING DELIVERS 1.44x.** Best at 10y is `amihud60` bottom decile at 1.29x,
then `log_turnover` top at 1.24x — the same axis twice. At 3y the best is 1.31x
against a 2.97x requirement.

**BUT THE DIRECTION INVERTS EVERY ONE-YEAR RESULT.** At one year liquid names
were the worst cell in every table (A19: liquid tercile −9.5% vs index). At ten
years the MOST liquid names are best on every axis at once: touch 69.1%,
P(−50%) **13.7%** against a 28.6% base, median **+174.7%**.

**THE CANDIDATE SURVIVES ITS NULL AND BOTH HALVES.** Clustered permutation over
(ticker, year) blocks: observed 69.1% against null 56.6% ± 4.6%, **z = +2.70**,
0 of 200 draws exceeded. Half-split lift 1.08 early, 1.52 late, positive in
both. **It does NOT clear the project's Bonferroni bar** — z = +2.70 is
p ≈ 0.0035 against a 0.0007 bar after 70 trials, and 200 draws cannot resolve
below 0.005. Suggestive, not established.

**AND FOR THE FIRST TIME THE INDEX COMPARISON IS WON.** 6,332 matched 10-year
windows, index on a total-return basis at the measured 1.77% top-decile yield:

| | median | mean | P(2x end) | P(2x touch) |
|---|---|---|---|---|
| liquid decile | **+188.8%** | +464.1% | 62.9% | **72.8%** |
| all liquid names | +12.8% | +200.8% | 33.7% | 57.9% |
| IHSG total return | +108.5% | +225.0% | 61.0% | 75.8% |

Paired per window: median **+51.8%**, 57.7% of windows won, **+32.9% early and
+73.9% late — positive in both halves.** Every previous construction here lost
to the index. The mechanism is not subtle: a one-year hold pays ~1.3% round
trip every year, a ten-year hold pays it once — 0.13% a year. **A9 flagged
exactly this and never tested it.**

### Verdict

**8 of 10: NOT REACHABLE.** Ceiling 72.8%, about 7 of 10, and only at ten years.

**7 of 10: reachable**, by buying the ~10 most liquid IDX names equal-weighted
and holding a decade.

**Limits, and they are severe.** Effective n is **56** for the sample and ~6 for
the decile — two independent decades. Only **30 distinct names** were ever in
the top decile (it does include BUMI, down ~99%, so it is not a survivor list).
The base rate does more work than the selection: 70.6% early against 40.3% late.
The holdout is spent. And ten years IS the mechanism — exiting at year three
returns you to the regime where costs eat everything.

**Trials after H23: 70.** Bonferroni bar α = 0.05/70 = **0.00071**.
**The 24-month holdout remains SPENT.**

---

## H24 — pushing on 7-of-10, and the cell where 8-of-10 appears

**2026-08-25.** The user's goal was "do 7 of 10, don't stop pushing". Code
`scripts/decade_push.py`, `scripts/decade.py`; tests `tests/test_decade.py`
(15). Memo `reports/horizon.md` §5–§6.

### The two costs H23 named and did not pay

**"200 draws cannot resolve below 0.005"** was left as a limitation when more
draws was the whole price of an answer. At 5,000 draws (with the +1
correction, since an empirical p can never honestly be zero): z = +2.87,
**p = 0.00140** against a bar of 0.00071. **Still does not clear.** Two
minutes to settle.

**P1 — the 2x take-profit.** Confirmed and brutal. Hold: 59.6% captured, mean
**+432.5%**. Sell all at 2x: 69.1% captured, mean **+58.1%**. The last 9.5
points of hit rate cost 374 points of mean.

**P4 WAS PRE-REGISTERED AND FAILED.** I predicted scaling out would dominate
both corners. It does not — the mean falls monotonically with every unit sold
(+432.5% / +338.9% / +245.3% / +151.7% / +58.1% at 0/25/50/75/100%). No free
lunch in the interior. `sell 50% at 3x` lifts the median to +186.9% against
+174.7% for 160 points of mean — the mean/median wedge a fourth time.

**A COLUMN IN THE FIRST DRAFT CONFLATED TWO QUESTIONS.** "Doubles realised" was
computed from the PEAK, crediting a hold-and-never-sell with captures it never
made: a name that doubled in year three and ended at 1.4x realised nothing.
Now split into **name doubled** (a property of the picking, constant at 69.1%
under every selling rule) and **I captured it** (what the rule decides). **So
"7 of 10 multi-baggers" is settled AT ENTRY; the exit only decides how much
reaches the account.**

### The tenure cell

Prior years already in the decile — backward-looking, so it passes A5:

| prior yrs | n | names | touched 2x | median | P(−50%) |
|---|---|---|---|---|---|
| 0 of 3 | 262 | 35 | 56.9% | +33.6% | 18.3% |
| 1 of 3 | 162 | 14 | 67.9% | +188.7% | 14.2% |
| 2 of 3 | 144 | 9 | 71.5% | +225.5% | 14.6% |
| **3 of 3** | 221 | **8** | **82.8%** | +197.2% | **7.2%** |

Monotone. **82.8% is eight of ten.** Names: ASII BBCA BBNI BBRI BMRI BUMI PGAS
TLKM. Clustered null z = +3.04, **p = 0.00100** vs bar 0.00069 — closer than
the decile, still short. Stable: **83.6% early, 82.4% late**, while the base
rate collapsed 70.6% → 40.3%.

**FOUR REASONS IT IS NOT THE ANSWER.** Effective n **1.8** — eight names, about
ONE independent observation, the thinnest cell in the project. It was found by
LOOKING after the decile result was in hand. It yields **four names today**
(BBCA BBRI BMRI TLKM) and **three are banks**. And AMMN scores 3 of 3 but has
been listed 3.1 years, so its tenure score is its whole life against the
historical eight's 20 — `MIN_LISTED_YEARS = 10` now enforces that a name whose
entire history is shorter than the hold is not the object measured.

### Verdict

**10 names at ~69%**, or **4 names at ~83%** on one effective observation and
concentrated in banks. **Ten names at eight-of-ten is not available from this
evidence**, and the two cells trade breadth against hit rate with no point on
the curve delivering both.

### And the horizon was dropped from a summary

The 69.1% was quoted without "over ten years" beside it and read as a one-year
figure. **The correction is not a rescaling — the tilt inverts.**

| hold | all liquid | decile | core | per 10 names |
|---|---|---|---|---|
| 1y | 9.5% | **4.2%** | **1.0%** | **0.4 of 10** |
| 3y | 27.0% | 24.6% | 19.5% | 2.5 of 10 |
| 5y | 39.0% | **49.3%** | 49.4% | 4.9 of 10 |
| 10y | 55.5% | **70.0%** | **82.8%** | **7.0 of 10** |

At one year the decile touches 2x LESS often than the liquid names it excludes
(4.2% vs 10.2%) — the wrong side of the trade, not a weaker version of the
right one, and A19's inversion seen a second time. `BY_HORIZON` in
`scripts/decade.py` prints it on every run with four tests pinning it: a
conditional result quoted without its condition is a wrong result, and the fix
belongs in the code that prints it.

**Trials after H24: 76.** Bonferroni bar α = 0.05/76 = **0.00066**.
**The 24-month holdout remains SPENT.**

---

## H25 — the fast-multiplier screen, and the only result here that clears the bar

**2026-08-25.** The user rejected the ten-year answer: "I want high multiplier
fast, that's why I play in emerging market." So the question is inverted —
maximise P(a name doubles WITHIN A YEAR). Code `scripts/fastmover.py`, tests
`tests/test_fastmover.py` (13). Memo `reports/fastmover.md`.

**THE SCREEN: most volatile 5% AND thinnest-traded 20%**, within names above
Rp1bn/day. P(touch 2x in one year) **21.22%** against a 12.05% base, lift
1.76x, on 443 name-years / 234 names.

| | |
|---|---|
| clustered null (5,000 draws) | 12.37% ± 1.56% |
| z | **+5.66** |
| p | **0.00020** vs bar **0.00064** after 78 trials |
| half-split | 1.51x early, 2.06x late — positive in both |

**IT CLEARS THE BONFERRONI BAR. It is the only result in this project that ever
has** — and §2 is why that is not the good news it looks like.

**IT IS VARIANCE, NOT SKILL.** Rank-corr with vol60: `lowvol` **−1.00** (it IS
vol60 negated), `amihud60` +0.34, `squeeze` +0.31 — one factor, and it is
volatility. **H13's PREDICTED-NULL control `squeeze` ranks THIRD** at 18.53%.
A19 records that a firing negative control means significance is not evidence;
here it does not merely fire, it places. P(touch 2x) 21.2% against P(end below
half) **18.7%** — **2.1 double and 1.9 halve per 10 names a year.** The
permutation null cannot see this, because it asks "is this cell different from
a random cell", to which the answer is trivially yes.

**AND IT DOES NOT COMPOUND — the number that withdrew H17 and H18, a fifth
time.** Arithmetic mean net of cost **+16.9%**, median **−19.1%**, **mean log
−0.1927 → −17.5% a year compounded.** Round trip on these names is 1.38%
median against 0.90% for the liquid decile. A ten-name basket rebalanced
annually recovers much of it and still returns **+5.1% against the index's
+14.6%: 3.1x against 22.8x over 23 years, with a 14.8% chance of ending below
where you started.**

**THE RESOLUTION IS POSITION SIZE, NOT SELECTION.** The NUMBER of names that
double does not depend on how much money is in them — ten screen names double
about twice a year whatever the sleeve weighs.

| sleeve | median CAGR | terminal | P(<1.0x) |
|---|---|---|---|
| 0% | +14.6% | 22.8x | 0.0% |
| **20%** | **+14.0%** | 20.2x | 0.0% |
| **30%** | **+13.7%** | 19.3x | 0.0% |
| 100% | +5.1% | 3.1x | **14.8%** |

Each 10% into the sleeve costs about **1 point of CAGR**. At 20–30% you keep
94–96% of the index's compounding and still see two doubles a year.

**THE LIVE SCREEN TIERS AND THE TIERS CARRY THEIR OWN ODDS.** The tight screen
yields only **four** names on 2026-08-24. Returning four when ten were asked
for, or quoting 21.2% for a loosened screen, are both quiet failures, so
`pick_tier` widens and states which tier it used. Only the tight tier has the
null behind it; wider tiers have measured odds and **inherited** significance.

**Trials after H25: 78.** Bonferroni bar α = 0.05/78 = **0.00064**.
**The 24-month holdout remains SPENT.**

---

## H26 — search on ASYMMETRY, and it retracts H25

**2026-08-25.** The user asked for "profit far more likely than loss, and still
a multi-bagger" — a request for a RATIO, which nothing here had ever ranked
cells by. Objective fixed before scoring: `skew = P(touch 2x) / P(end<=0.5)`,
300-obs floor, both legs estimable. Code `scripts/asymmetry.py`, memo
`reports/asymmetry.md`.

**THE FRONTIER IS MONOTONE — YOU CANNOT MAXIMISE BOTH.**

| cell | P(2x) | P(−50%) | skew | dbl/10 | CAGR/name |
|---|---|---|---|---|---|
| everything | 12.1% | 9.0% | 1.33 | 1.2 | −6.3% |
| **strength + calm** | 10.5% | **4.1%** | **2.60** | 1.1 | **+5.2%** |
| strength only | 13.6% | 6.3% | 2.15 | 1.4 | +1.1% |
| strength + some vol | 21.4% | 13.5% | 1.59 | 2.1 | −8.1% |
| H25 vol screen | 21.2% | 18.7% | 1.13 | 2.1 | −16.1% |

**THE WINNER: within ~2% of the 52-week high AND below-median vol.** n=2,022,
skew **2.60** vs null 1.20±0.15, **z=+9.44, p=0.00033 vs bar 0.00061 — CLEARS.**
Half-split 2.65 / 2.53. **10-name basket over 24 years: median 58.3x = +18.5%
CAGR against the index's 25.1x = +14.4%, beating it in 90.2% of draws, 10th
percentile +14.4%.** Momentum plus low-vol — a prior mechanism, not a mined cell.

**IT RETRACTS H25.** The volatility screen reads skew **1.13 against a null of
1.18: z = −0.19, p = 0.54** — indistinguishable from a random cell on the thing
that matters — and its basket returns 3.4x against the index's 22.8x, beating
it 5.0% of the time. It cleared the bar on P(2x) and that told us nothing,
exactly as its own §2 warned. **Optimising the upside alone finds variance;
optimising the RATIO finds something that survives.**

**Q2 PRE-REGISTERED AND FAILED, monotonically.** "Already fallen names are
asymmetric" is backwards: nearest the 52w high skew **2.15**, middle 1.53,
furthest below **0.80** — fallen names halve MORE often than they double.

**Q1 PRE-REGISTERED AND SUPPORTED.** Base skew rises monotonically with
horizon: 1.33, 1.48, 1.57, 1.80, 2.26 at 1/2/3/5/10 years. Time converts
diffusion into drift; the screen buys about six years of that at one year.

**NOT licensed: "multi-bagger fast".** 1.1 names in ten double per year, not
2.1. Holdout spent; best of ~30 cells.

**Trials after H26: 82.** Bonferroni bar α = 0.05/82 = **0.00061**.

---

## H27 — the cross-sectional model §8 asked for, and the null that beat it twice

**2026-08-25.** Pushed on "there must be patterns, it cannot be random". The
fair part of that: every sweep here ranked DISCRETE CELLS with hand-tuned cuts,
which is exactly what §8 forbids — "numeric features feeding a cross-sectional
model, not discrete buy/sell rules with hand-tuned parameters". **The model was
never built.** Code `scripts/xsection_model.py`, tests
`tests/test_xsection_model.py` (8). Memo `reports/xsection_model.md`.

Purged expanding walk-forward: a cohort dated t settles at t+252, so each test
year trains only on cohorts whose window CLOSED before it began. 15 folds,
27,561 test-fold observations.

| | P(2x) | P(−50%) | skew | mean log |
|---|---|---|---|---|
| **model top decile** | 9.2% | **4.0%** | **2.31** | +0.0132 |
| base | 10.6% | 8.7% | 1.22 | |
| H26 hand-cut cell | 10.5% | 4.1% | 2.60 | +0.0494 |
| + sector | 9.0% | 3.6% | 2.46 | +0.0034 |

**Null 1.15 ± 0.06 → z = +20.51.**

**R1 — the model does NOT beat the cell, and that IS the finding.** 2.31 vs
2.60, but the model's is OUT OF SAMPLE and the cell's is in-sample. Two
entirely different methods land within 0.3 of each other. **That convergence is
the strongest evidence in this project that the structure is real rather than
mined.**

**R2 SUPPORTED.** vol60 0.073, log_turnover 0.021, amihud60 0.012. `hi52`
scores low on the UP leg — it earns its keep by suppressing the DOWN leg, which
is why a ratio objective found it and a rate objective never did.

**Sector: +0.15 skew, −0.0098 mean log** at 99.6% coverage. Marginal. Its
`shares` column is deliberately unused — frozen at 2024-07-10, so applying it
to a 2010 bar is look-ahead and rights issues are what make that wrong.

**THE NULL BEAT THE MODEL TWICE AND BOTH TIMES IT WAS THE NULL.** First
version returned 3.06 against the model's 2.31.
*Bug one:* `up` and `down` permuted independently, breaking their real link —
a name that can double is the same name that can halve — inventing rows that
doubled with no halving risk.
*Bug two, the real cause:* permuting INSIDE (ticker, year) blocks is nearly a
no-op, because the ~12 monthly cohorts of one ticker-year hold near-identical
labels from eleven-month-overlapping windows. The null kept the structure it
existed to destroy.
*The fix:* reassign whole blocks' LABELS to other blocks' FEATURES. Null fell
3.06 → 1.15. **A null that beats the thing it tests is broken, not evidence of
a weak model.** Seventh time the null decided a result here, and the first time
it erred toward UNDERSTATING a real effect.

**What this settles.** There is a pattern, it is not random, and it is worth
about 2.3:1. And a flexible model finds no more than two hand-picked filters —
with eleven collinear price features over one macro history the interaction
space is empty. The next real instrument is non-price fundamentals, and
`data/cache/fundamentals` holds 59 names of 725.

**Trials after H27: 85.** Bonferroni bar α = 0.05/85 = **0.00059**.

---

## H30 — the triple-EMA golden cross on IDX: the state works, the cross does not

**2026-08-25.** Asked for a "triple EMA golden cross pointer trained on IDX".
Trained has to mean measured, so all 36 (fast, mid, slow) combinations were
gridded on 744 names and 5,879 pre-holdout sessions, 60-session hold. Code
`scripts/ema_cross_idx.py`, memo `reports/ema_cross_idx.md`.

**PRE-REGISTERED:** no configuration beats buy-and-hold after costs; what
survives is the alignment STATE, not the cross. **Half wrong, half exactly
right.**

**THE BENCHMARK.** Unconditional 60-session forward: mean +3.88%, median
+0.00%, **mean log +0.0021 → +0.9%/yr.** The mean is volatility drag in
disguise, which is why everything is scored on mean-log.

**THE CROSS IS NOT A TRIGGER.** Best of 36 (`20/21/100`): mean **+4.25%** net,
which does edge buy-and-hold — so the prediction was wrong there. But median
**−0.62%**, win rate **48.5%**, and half-split mean log **+0.0274 early,
−0.0145 late** — it does NOT compound in both halves. All 36 rows share that
shape: positive mean, negative median, sub-50% win rate. The winner also has
fast=20 mid=21, so the fast/mid axis is noise.

**THE STATE WORKS AND A SINGLE MA DOES NOT.**

| state | n | mean log | early | late | both>base |
|---|---|---|---|---|---|
| price>EMA100 | 253,136 | +0.0159 | +0.0308 | +0.0009 | no |
| 50>100>200 | 205,790 | +0.0142 | +0.0270 | +0.0014 | no |
| 13>34>100 | 195,393 | +0.0197 | +0.0331 | +0.0063 | YES |
| **price>50>100>200** | 142,797 | **+0.0218** | +0.0327 | **+0.0109** | **YES** |

`price>EMA100` alone and the bare `50>100>200` collapse to the base rate in the
recent half. **Requiring price ABOVE the stack as well as the stack ordered is
what holds in both eras.**

**A BUG COST A WHOLE TABLE.** The first run reported identical counts for
"cross" and "aligned" in every row. `DataFrame.shift()` on a boolean frame
returns OBJECT dtype; `~` on a Python bool is integer negation, `~True` is −2
and `~False` is −1, and **both are truthy** — so `aligned & ~prev` evaluated to
`aligned` and the table measured the state while labelling it the trigger.
`.astype(bool)` is the fix.

**Trials after H30: 95.** Bonferroni bar α = 0.05/95 = **0.00053**.
