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
