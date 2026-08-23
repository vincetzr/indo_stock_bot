# IDX Alpha Research — Project Brief

---

## 1. Mission

Test whether IDX broker-flow, price structure, and macro regime data contain
exploitable signal, and if so, build a system that generates tradeable signals from it.

**This is a research program, not a trading bot.** It becomes a bot only if the research
clears the gates in §5–§9. Building the bot first is the default failure mode and it is
explicitly forbidden here.

---

## 2. Operating stance

You are the quant research engineer. I have an ML background (control science /
robotics, Python, comfortable with the maths) but I am not a finance professional.
Do not simplify the statistics. Do explain finance-specific conventions the first time
they appear.

**Your job includes telling me I'm wrong.**

- If a hypothesis I propose is untestable as stated, say so in the first line.
- If a result is a multiple-testing artifact, say so even if I sound excited about it.
- If a sample is too small to support the claim, give the effective sample size.
- If a gate fails, report the failure. Do **not** loosen the criterion and continue.
  Silently relaxing a threshold to keep the project alive is the worst thing you can do
  in this repo.

A dead hypothesis found in week two is a win. Three months of elegant pipelines feeding
a signal that never had predictive power is the outcome we are structuring against.

When I ask for a result, give effect size and uncertainty, not a p-value alone.

---

## 3. Data sourcing — what is free, what is not

| Data | Availability | Notes |
|---|---|---|
| Daily OHLCV | Free | yfinance `.JK`, and IDX's own summary endpoints |
| **EOD broker summary** | **Free** | `idx.co.id` publishes it; endpoints reverse-engineered in `NeaByteLab/IDX-API`, `nichsedge/idx-bei`, `ExRonin/Stock-Scrapper-IDX` |
| Corporate actions | Free | IDX announcements; needs parsing |
| Foreign net flow | Free | IDX trading summary |
| Macro (USDIDR, BI rate, DXY, UST) | Free | BI, FRED |
| Real-time tick / running trade | **Licensed** | IDX Data Services. Free route is scraping my own broker session — brittle, ToS-grey |
| Order execution | **Does not exist** | No Indonesian retail broker offers a public order API |

Two consequences that shape the whole architecture:

1. **Execution is manual, always.** The end product is a signal generator plus alerting,
   not an auto-trader. Do not build execution plumbing.
2. **Live tick data is deferred, deliberately.** A signal operating on daily broker
   summary needs daily data. Do not build a live feed until a validated EOD signal
   exists. If one emerges and turns out to be intraday-sensitive, revisit then.

IDX Terms of Use item 5 bars redistributing site data commercially. Personal research
use is fine; do not build anything that republishes the feed.

---

## 4. Why the build order is not macro → flow → TA

My original instinct was macro first, then broker flow, then technical analysis. Macro,
flow and price structure are indeed the three feature families and all three probably
matter — but that is a bad *build* order.

Macro is the slowest family to reach a falsifiable claim. It moves roughly monthly, so
ten years is ~120 effective observations, and most of it is already priced. Building it
first means months of work before the first real test of anything.

Broker flow is daily and cross-sectional: ~2,500 days × ~800 names. It reaches a
falsifiable claim in about two weeks.

**Order by cost-to-falsify, ascending:**

```
spine → broker flow → price/TA features → macro (last, as regime conditioner)
```

Macro's job in this project is not prediction. It is answering *"in which regimes does
the flow signal work, and in which does it invert?"* That question is meaningless until
a flow signal exists.

---

## 5. Phase 0 — Data spine

A point-in-time database. If this is wrong, nothing downstream is real.

### Contents

- Daily OHLCV, all IDX names, maximum available history
- **Delisted and suspended names included.** A universe of currently-listed tickers is
  survivorship-biased and every backtest run on it is inflated.
- Daily broker summary per ticker: buy/sell volume, value, average price, per broker code
- Corporate actions: splits, reverse splits, rights issues, dividends, warrants
- Foreign net flow per ticker
- Suspension and ARA/ARB flags per ticker-day

### Point-in-time rules — the subtle part

IDX's **trading rules changed over the sample**, not just its prices. Encode the
historical schedule of:

- **Auto Rejection (ARA/ARB) bands** — the percentage limits over time, including the
  asymmetric-ARB period. A day where a stock is locked at ARA is a day you **could not
  buy**. Any backtest filling at close on an ARA day is fiction.
- **Fraksi harga (tick size) schedule** — revised several times. Applying today's
  schedule to 2016 data is a lookahead error that silently understates spread cost.
- Session times, pre-closing and random-closing mechanism.
- Full-lot definition (100 shares) and any changes.
- **Broker code master with effective date ranges.** Mergers and licence changes
  reassign codes. Without this you will attribute one firm's history to another.

### Adjustment

Splits are easy. **Rights issues are the trap.** A heavily dilutive rights issue needs a
proper theoretical ex-rights price adjustment or the return series shows a fake crash.
Use the CBRE-type extreme-dilution cases as test fixtures — if the adjustment handles
those, it handles most things.

### Gate 0

Reconstruct 20 random ticker-years and reconcile total traded value against IDX
published aggregates to within tolerance. Reconcile 5 known corporate-action events by
hand. If the spine doesn't reconcile, stop and fix it.

---

## 6. What broker summary actually is

Read before interpreting anything. These four limits go into every output.

1. **A broker code is not a client.** One code aggregates thousands of accounts —
   retail, institutional, prop desk, foreign nominee. "YP is accumulating" means YP's
   net book moved, which may be ten thousand retail buyers.
2. **It is public and simultaneous.** Every retail trader in Indonesia sees the same
   table at the same moment. A signal visible to everyone at once has its edge competed
   down fast. If there is alpha, it is more likely in *transformations* others aren't
   computing than in the raw net-buy column.
3. **It is adversarially obfuscated.** Large players split orders across brokers
   precisely *because* broker summary is public. Concentration metrics degrade over the
   sample as this spreads — check for that decay explicitly and report it.
4. **"Reverse engineering the bandar" is an inference, not an observation.** Frame every
   claim as *"this flow pattern precedes this return distribution."* Never as *"this
   broker is the bandar and is doing X."* The second invents a narrative the data cannot
   support, and it makes the model worse.

---

## 7. Phase 1 — Does broker flow predict anything at all?

The cheapest real question. Answer it before writing another line of infrastructure.

### Hypothesis

> Broker net-buy imbalance for ticker *i* on day *t* predicts the return of *i* over
> *t+1 … t+k*, for k ∈ {1, 3, 5, 10, 20}.

### Required controls

Report the information coefficient *after* neutralizing for: 1–12 month momentum, size
(market cap), turnover/liquidity, sector, and short-term reversal. Broker flow that
works only because it proxies momentum is not a discovery.

### Required honesty

- Report the **full decay curve**, not the best k.
- Report IC by liquidity decile. If the effect lives only in the bottom two deciles it
  is likely untradeable — say so.
- **Apply costs before claiming anything.** Indonesian retail is roughly 0.15–0.20% on
  buy and 0.25–0.30% on sell (sell side includes the 0.1% final tax). *Verify against my
  actual broker's schedule.* Then add half-spread from the point-in-time fraksi harga —
  on small caps this is often larger than the commission.
- Exclude fills that ARA/ARB would have blocked.

### Gate 1

Post-cost, liquidity-filtered, control-neutralized IC is significantly non-zero with a
stable sign, out of sample. If not — report it and we pivot the hypothesis rather than
adding features to rescue it.

---

## 8. Phase 2 — Feature expansion

Only past Gate 1. Cheap, because the spine exists.

- **Flow:** concentration (HHI across brokers), persistence, foreign vs domestic
  divergence, broker-set overlap across tickers, net-buy acceleration
- **Price/TA:** realized vol, ATR-normalized returns, range compression, volume
  Z-scores, distance to N-day extremes. Compute as **numeric features feeding a
  cross-sectional model** — not as discrete buy/sell rules with hand-tuned parameters.
- **Structural:** free float, index membership, days since corporate action

Rule: every feature must be justified by a prior mechanism *before* it is tested, and
logged in `hypotheses.md` with the date and the prediction. Post-hoc feature discovery
is how the multiple-testing problem eats the project.

---

## 9. Phase 2b — Broker behavioural reverse engineering

Produces: per-broker profitability, a behavioural fingerprint vector, archetypes, and a
written dossier per broker code.

### 9.1 "Profit margin" is two questions

**How does the brokerage firm make money?** Commission on turnover — proportional to
gross volume, direction-irrelevant, already published in their OJK filings. Nothing to
reverse-engineer. Read the statements.

**Is the flow behind broker code X profitable?** This is the real question and it *is*
estimable. But be precise about whose profit it is: the code aggregates thousands of
accounts, overwhelmingly agency, with prop desk a small slice. What you compute is
**the aggregate P&L of the client cohort behind that code.**

Name it `cohort_pnl` everywhere in code and output. Never `broker_profit`. The naming
discipline stops the conceptual error leaking into the dossiers.

### 9.2 Structural limits — state in every output

- **Starting inventory is unknown.** Broker summary is flow, not position. Needs burn-in
  and leaves a level ambiguity that never fully resolves.
- **Crossing inflates gross volume** without directional exposure. Detect and handle.
- **Foreign nominees (CS, CLSA, UBS, MS) are omnibus.** One code carries many
  uncorrelated end-clients. Flag low-confidence by construction; do not treat as one actor.
- **Codes are not stable over history** — see the broker code master in §5.

### 9.3 Cohort P&L estimation

Per (broker, ticker), walk the daily flow forward:

```
inventory_t   = inventory_{t-1} + buy_vol_t − sell_vol_t
WAC_t         = weighted-average cost, updated on buys only
realized_t    = sell_vol_t × (sell_avg_price_t − WAC_{t-1})
unrealized_t  = inventory_t × (close_t − WAC_t)
```

**Burn-in.** Discard the first 250 trading days of each broker-ticker series. Report
with and without, to show sensitivity.

**Round-trip restriction — the clean estimate.** Find episodes where inventory starts
near zero, rises, and returns near zero. Within those, P&L is unambiguous and
independent of the initial-position problem. Round-trip P&L is the primary estimate;
full-path is the noisy secondary. Report both.

**Headline metric — margin per rupiah traded:**

```
margin_bps = 10000 × cohort_pnl / gross_traded_value
```

Use this, not absolute rupiah. Comparable across brokers of wildly different size, and
it directly answers "how profitable is this flow." Report as a distribution across
tickers and time with bootstrapped CIs, never a point estimate. Include a null: the same
computation on shuffled broker labels.

**Costs.** Cohort P&L is gross. Retail cohorts pay commission plus the 0.1% sell tax;
prop and institutional pay far less, and you cannot observe which. State this, report
gross, give a sensitivity band.

### 9.4 Behavioural fingerprint vector

Per broker, rolling, so drift is visible. All computable from broker summary + OHLCV.

**Execution style — the most informative and most neglected metric**

Broker summary gives average buy price and average sell price. Compare to the day's VWAP:

```
buy_edge_bps  = 10000 × (VWAP_t − buy_avg_price_t)  / VWAP_t
sell_edge_bps = 10000 × (sell_avg_price_t − VWAP_t) / VWAP_t
```

Positive = executed better than the day's average: patient, working the order, earning
spread. Negative = paying up: urgent, liquidity-taking. This separates market-maker-like
flow from momentum-chasing flow better than volume ever will.

> **Mechanical bias — must correct.** A broker who is a large share of the day's volume
> pulls VWAP toward their own price, shrinking the measured edge toward zero. Recompute
> VWAP **excluding that broker's own trades** whenever their share exceeds ~10%.
> Skipping this makes every large broker look identically neutral. This is the most
> common error in this kind of analysis.

**Horizon**

- Inventory half-life: decay time after accumulation episodes
- AR(1) of daily net-buy sign — persistence vs alternation

**Timing**

- `corr(net_buy_t, return_t)` — same-day: do they push or absorb
- `corr(net_buy_t, return_{t−1})` — momentum-chasing vs contrarian
- `corr(net_buy_t, return_{t+1..t+k})` — the predictive one; overlaps Phase 1

**Positioning**

- HHI of net-buy value across tickers — generalist vs specialist
- Peak inventory as % of free float — capacity, and whether they *could* move price
- Crossing ratio `min(buy_val, sell_val) / max(buy_val, sell_val)` — high implies
  market-making/churn, low implies directional conviction

**Conditioning**

- All of the above split by IHSG realized-vol tercile
- Month-end and quarter-end effects (window dressing)
- Behaviour around corporate actions, rights issues especially

### 9.5 Archetypes, not names

Cluster on the standardized fingerprint vector (try HDBSCAN and GMM, compare).

**Label clusters from their statistics, never from market folklore.** A cluster is
described as *"negative buy_edge, 3-day half-life, momentum loading +0.4, low HHI"* and
only then given a working name like `impatient-momentum`. If the label cannot be derived
from the numbers in front of you, the cluster doesn't get a name.

**Stability check, mandatory.** Fit on 2015–2019, assign on 2020–2024. If assignments
don't persist, the archetypes are noise — report that. Also check whether the *number*
of stable clusters is stable; if not, stop and report that broker behaviour is not
cleanly separable.

**Expect degradation.** Plot fingerprint distinctiveness by year. If it decays as order-
splitting spreads, that is a real and important finding about the dataset's shelf life.
Report it prominently rather than averaging it away.

### 9.6 The dossier — "how they do it"

One per broker code with sufficient data. This is where fabrication is most likely, so
it is constrained.

```
BROKER CODE      + firm name, with code-history caveats
SAMPLE           n days, n tickers, date range, data-quality flags
COHORT P&L       margin_bps [95% CI], round-trip and full-path
FINGERPRINT      each metric with value and CI
ARCHETYPE        cluster, assignment confidence, stability across splits
BEHAVIOURAL READ ≤200 words, every sentence traceable to a number above
FALSIFICATION    what observation would overturn this read
CONFIDENCE       high / medium / low, with reason
DRIFT            has the fingerprint changed materially over the sample
```

Hard rules for BEHAVIOURAL READ:

- Every claim maps to a computed statistic. No inferred motive, no invented strategy
  description, no "accumulating ahead of news."
- State the observed regularity first — *"buys execute 12bps below VWAP, inventory
  half-life 9 days, net-buy loads +0.31 on prior-week return"* — then, separately and
  clearly marked as interpretation, *"consistent with patient momentum accumulation."*
- If the fingerprint is a blend (nominee codes), say so and stop. Do not resolve
  ambiguity by picking the more interesting story.
- Below the minimum sample threshold: write "insufficient data" and move on.

The pull toward narrative here is strong and is the main way this phase produces
confident nonsense. Prefer a short, boring, well-supported read over a rich one.

### Gate 2b

**Does broker identity carry information beyond aggregate flow?**

Baseline: the Phase-1 signal using only *total* net-buy imbalance, ignoring which broker.
Test: the same signal with broker-identity or archetype features added.

If the broker-aware model does not improve out-of-sample IC over baseline, identifying
individual brokers is decorative and the project drops back to aggregate flow. Report
that outcome plainly if it occurs — it is legitimate, useful, and saves months.

---

## 10. Phase 3 — Macro as regime conditioner

Macro enters here. Not to predict returns — to *segment* them.

Candidate regime variables: USDIDR level and momentum, BI rate and direction, US 10y and
DXY, Indonesian CDS, aggregate foreign net flow to IDX, commodity complex (coal, CPO,
nickel), IHSG realized vol.

**Test:** does the Phase-1/2b signal's IC change sign or magnitude conditional on regime?

Be sceptical. With ~15 years and a handful of regime switches you have very few
independent observations. Any regime split producing beautiful results on three episodes
is almost certainly overfit. Say so when it happens.

---

## 11. Validation protocol — non-negotiable

- **Purged, embargoed walk-forward.** Purge overlapping label windows; embargo after each
  test fold. Standard k-fold on financial panels leaks.
- **Track every hypothesis tested.** Maintain a running count and report the **deflated
  Sharpe ratio** — the number of trials matters more than any single result.
- **True holdout.** Reserve the most recent 18–24 months, untouched, until the end. Touch
  it once. If it fails, it fails; do not iterate against it.
- **Report the null.** Every result table includes a random-signal baseline run through
  the identical pipeline.
- Backtested edge is not live edge. Slippage, capacity, and the fact that the sample is
  one realization of history all cut the same direction.

---

## 12. The strategic inversion — read before optimizing anything

Flow is close to zero-sum: the counterparty of a profitable cohort is another cohort. So
there are two ways to use a profitability ranking, and they are not equally good.

**Following the winners** is the obvious move and the weak one. The signal is public and
simultaneous, the winning cohort is often winning *because* it executes better than you
can, and you would enter after them at a worse price on retail costs.

**Taking the other side of persistent losers** is the durable play. That retail cohorts
lose persistently while institutions gain is among the most robust results in the
market-microstructure literature (Taiwan, Finland, elsewhere), and it is durable
precisely because the losing cohort continuously regenerates — new retail keeps arriving.
A cohort that has lost for eight straight years is far more stable than one that has won
for two.

**So the primary output of this project is not "who is the bandar."** It is: *which flow
is persistently dumb, is it identifiable in real time, and is it large enough to trade
against after costs.* Structure the analysis around that question.

---

## 13. Repo layout

```
data/           raw pulls, immutable, gitignored
  spine/        point-in-time DB (parquet or duckdb)
  reference/    ARA/ARB schedule, fraksi harga schedule, broker code master
src/
  ingest/       scrapers, corporate action parsing
  spine/        adjustment, PIT joins, reconciliation tests
  features/     flow, price, structural, macro
  research/     hypothesis notebooks, one per hypothesis
  validation/   purged CV, deflated Sharpe, null baselines
reports/        one memo per phase — the actual deliverable
hypotheses.md   running log: date, hypothesis, prediction, result
config/         every experiment reproducible from a config + seed
```

---

## 14. Definition of done, per phase

Each phase ends with a written memo in `reports/`, not just code: what was tested, what
the null was, what the result was, what would have falsified it, and what I now believe
with what confidence.

If the memo cannot be written honestly, the phase isn't done.

---

## 15. Working style

- Commit per logical unit. Every experiment reproducible from a seed and a config file.
- Charts over tables for anything with a time or decay dimension.
- Effect size and uncertainty, not p-values alone.
- If I propose something that won't work, tell me in the first line of your reply.

---

# APPENDIX — repo reality as of 2026-08-21

*Added by Claude. The brief above is the standing instruction; this section only
records where the existing repo already meets it, already contradicts it, or cannot
reach it. Where the two disagree, the brief wins unless the disagreement is listed here
as a hard blocker.*

## A1. The data-shape blocker that turned out not to be one — read first

**This section was wrong, and it was wrong in the direction that stops work.** It is
kept rather than deleted because the error is instructive: a costing mistake read as a
law of nature, and it very nearly cost the project its central experiment.

The original claim was that IndoPremier costs **1 request per ticker-day**, so the §4
panel needs ~2,000,000 requests and ~27 days of fetching, and therefore *"Phase 1 runs
on a 10-name panel, not an 800-name one."*

That prices the **daily** mode. The same endpoint takes `start` and `end` and returns
the rekap **aggregated over the whole window for one request** — which
`scripts/pullback_flow.fetch_window` had already been using, and 1,703 range files were
sitting in the cache while the appendix said the panel was infeasible.

| route | real shape | real cost | status |
|---|---|---|---|
| IndoPremier, **range** mode | 1 request per **(ticker, window)**, top-10 | **31,824 requests** for 200 names × fortnightly × 2014-2026 | **done — collected, zero failures** |
| IndoPremier, daily mode | 1 request per ticker-day | the ~2,000,000 figure above | works, and is the wrong tool |
| `idx.co.id` broker summary | 1 file per **session, all stocks** | ~2,500 files | 403 from any script — Cloudflare bot check on TLS fingerprint |
| Sectors licensed API (`api.sectors.app/v2`) | 1 credit per ticker per **fortnight**, full depth | ~143,000 credits | needs a paid key |

**The panel exists.** 176 names (64 of them delisted), 329 fortnights, 30,234 labelled
rows. §7 has been run on it and Gate 1 fails — see `reports/phase1_flow_panel.md` and
A3 below.

What is genuinely given up on this route is **resolution, not size**: fortnightly flow,
so §7's decay curve reaches k = 10 and 20 sessions and cannot reach k = 1 or 3.
Aggregation runs one way — two fortnights make a month, no arithmetic recovers a day.

**The lesson worth keeping:** before recording a constraint as arithmetic, check that
the unit price is the cheapest one the source offers. This one was off by a factor of
sixty.

The 403 is a bot check, not a network block or a geo-block — proven, and documented in
`docs/FULL_REKAP.md`. The three repos §3 cites clear it with `curl_cffi` browser
impersonation. That is the specific act IDX's controls exist to prevent, so this repo
does not do it; `docs/FULL_REKAP.md` §3 has the full reasoning and the sanctioned
alternatives (you download in a browser → `broker_collect.py --ingest`, or the licensed
API).

## A2. What already exists, mapped onto the brief

| brief | repo today |
|---|---|
| §5 spine | **Gate 0 PASSES** on nine checks (`scripts/gate0.py`, exits 0/1). PIT ARA/ARB + fraksi harga + lot + halt schedules in `src/idxbot/spine/reference.py`; quality gates, corporate-action adjustment, three sourced repairs and a broker-code master alongside. Survivorship **partly repaired**: 121 vanished names recovered with history from a 2019 point-in-time snapshot, giving a measured 2.87%/yr attrition instead of an assumed 1–8% range. **What is left is one-sided** — the snapshot ends 2019-04-07, so the months in which a name actually died are still missing, and the bias figure stays a bound rather than a correction. See `reports/phase0_spine.md`. |
| §7 Phase 1 | **run and FAILED** — first on a 1-name panel, then properly on 176 names (64 delisted) × 241 fortnights. Gate 1 fails on the conjunction: significant in sample but sign-unstable across liquidity and a quintile spread indistinguishable from zero *before* costs. See A3 and `reports/phase1_flow_panel.md`. |
| §9.3 cohort P&L | **built and run; no broker-identity signal.** Round-trip median −25.3 bps against a shuffled-label −32.5: the null sits on top of the signal. The round-trip restriction was indeed the answer to the starting-inventory problem, which is severe here (inventory negative 46.4% of the time). See `reports/phase2b_cohort_pnl.md`. |
| §12 persistence | **run and negative.** Broker-code margin rank does not persist: six statistics across two stores, all inside their 200-draw permutation nulls, best p = 0.119 against a Bonferroni bar of 0.0018. So the losing cohort is not *identifiable* at broker-code resolution, which is what §12's strategy needs. See A6 and `reports/phase2b_persistence.md`. |
| §9.4 execution style | **built and the §9.4 bias is corrected** — VWAP now excludes each broker's own trades. See A4; the size null it threatened was recomputed and stands. |
| §9.5 archetypes | not built |
| §11 purged walk-forward | partial — walk-forward exists; overlapping forward windows are now Newey-West corrected (`layer2_test.one_sided`), but purge/embargo folds do not exist |
| §11 trial count | `hypotheses.md` now exists; **20 trials** logged through H9 |
| §13 layout | **conflicts.** Repo is `src/idxbot/`, `scripts/`, `docs/`, `tests/` with 1,574 passing tests. Restructuring wholesale would be destructive for no research gain, so the brief's layout is treated as the target for *new* work and `reports/` + `hypotheses.md` are adopted now. |

**One spine result worth carrying into every later phase.** Every IDX price is an
exact multiple of that day's fraksi harga, so a price that is not was never
traded — it is vendor arithmetic. `quality.off_tick` uses that, and two things
follow. First, an off-grid stretch that `level_shifts` also calls a break is the
SCCO defect: across 937 tickers it finds exactly three (SCCO, PYFA, SINI), all
now repaired. Second, **22.3% of the spine provably sits on a vendor-adjusted
basis** — a lower bound, since a whole-number factor is invisible to the test —
which means `reference.half_spread` is looking the tick up from the wrong price
band on those bars and **understating spread cost**. §7 requires costs from the
point-in-time fraksi harga; on a back-adjusted series that requirement is not yet
met. The test is **one-sided**: off-grid proves adjustment, on-grid proves
nothing. Two earlier versions forgot the second half and reported 18,300 and
2,760 defects respectively.

## A3. Phase 1 has already been run once, and it failed

Pre-registered as Protocol A (`scripts/layer2_protocol.py`, hash `6b8e0a2c9d1f4e73`),
frozen before the data existed. Four hypotheses, Bonferroni-corrected, day-level
clustering. **All four failed**, and all four failed with a *negative* sign — high
broker buying was followed by underperformance (H4: d = −1.34, t = −5.37).

That is an observation, not a finding: it came from looking at BBCA. So the reversed
claim was frozen as Protocol B (`layer2_protocol_b.py`, hash `b8c26de3a02d24f7`) with
**BBCA excluded from its own confirmatory sample**, and with a mandatory same-day-return
control because flow correlates +0.22 with the day's own move.

Anything that reports on H1–H8 must quote the protocol hash it ran under.

### Then it was run properly, on a real cross-section, and it failed again

**H9, 2026-08-23.** 176 names (64 delisted), 241 in-sample fortnights, 21,693 rows,
holdout reserved. Full memo: `reports/phase1_flow_panel.md`.

**Gate 1 is a conjunction and the panel clears one of its four conditions.**

| condition | result |
|---|---|
| significantly non-zero | **YES, in sample.** IC −0.0190, HAC t −2.86; 200 permutation draws through the identical pipeline put it outside every draw, empirical p 0.005. Not an artefact. |
| stable sign | **NO.** Liquidity Q4 +0.032, Q5 −0.047 — the sign flips between adjacent quintiles. |
| post-cost | **NO.** Quintile spread −0.215% a fortnight, t −0.70: indistinguishable from zero *before* any cost. |
| out of sample | **NOT TESTED.** The 24-month holdout is untouched and stays that way. |

It also does not clear the trial count: 8 prior trials plus ~12 here needs p < 0.0025
under Bonferroni, against an observed 0.005.

**So the honest summary is: a faint, statistically real rank tilt in-sample, far too
small to trade.** The sign replicated Protocol A's negative direction on a sample it was
not selected for, which is worth something — but it is a direction without a size.

**Two harness errors, both caught by the null, both now regression-tested.** The first
run had the NULL at t = −2.96 against the signal's −2.86: the permutations were
concatenated in period order and assigned positionally against a ticker-sorted frame, so
every period's shuffled values landed on other periods' rows. A null that certifies
anything is worse than no null. Then a single null draw at t = −1.53 was briefly
over-read as systematic bias; a second seed said otherwise. That is what prompted the
permutation test, which should have been the first statistic rather than the last.

**Do not attempt to rescue this by adding features** (§7). The memo lists the specific
rescue moves that are forbidden. The live question is now §12's: *which flow is
persistently dumb*, for which Gate 2b's baseline — aggregate flow — is now measured at
essentially nothing.

## A4. The bias §9.4 names WAS present, and is now corrected

`execution_edge()` compared each broker's average fill to the day's VWAP
*including that broker's own trades*. §9.4 is right that this shrinks large
brokers toward zero, and the shares here are large enough to matter: median 5.7%
of a session, 26% of broker-sides above 10%, maximum 94%.

It is now corrected exactly, since the footer totals are uncensored and each
broker's lots partition them:

```
VWAP_excluding_b = (total_value − b_value) / ((total_lot − b_lot) × 100)
```

Verified on a synthetic closed market with known true edges: a broker holding
40% of the day measures +0.43% biased against a true +0.83%, and +0.71%
corrected.

**The negative result it threatened was withdrawn, recomputed, and reinstated.**
"Size does not explain execution edge" now reads corr(log volume, edge) = −0.001
(p = 1.00) and corr with |edge| = −0.213 (p = 0.15) against a self-excluded
VWAP. The correction attenuates edge *magnitude* roughly symmetrically about
zero, so it changes how large an individual broker looks rather than whether
edge tracks size. Full detail in `hypotheses.md` under E4.

## A5. Standing constraints that predate this brief and still hold

- Fees: **0.28% buy, 0.18% sell, plus 0.1% sell tax = 0.56% round trip.** §7 quotes
  0.15–0.20% / 0.25–0.30%; the figures above are the user's actual Mandiri schedule and
  win.
- No leverage, no shorting.
- No look-ahead: never use data stamped after the decision bar.
- `data.broker_allowed_hosts` ships **empty** and a host is added only after the user
  has checked its licensing.
- IndoPremier is a public page on a licensed member's site: polite delay, permanent
  cache, no bulk harvesting, no redistribution.

## A6. §12's question has now been asked, and the answer is no at broker-code level

§12 is the strategic core of this brief: *"which flow is persistently dumb, is
it identifiable in real time, and is it large enough to trade against after
costs."* Phase 2b answered the middle of that badly by leaving it alone — a
cohort losing 25 bps a round trip is uninteresting unless the *same* cohort
keeps doing it.

**It does not.** Six statistics, two stores, 89 codes over 12.6 years and 27
codes over 18 months, every one inside its own 200-draw permutation null.
Best p = 0.119. Full memo: `reports/phase2b_persistence.md`, logged as H11.

Three things from it are worth carrying forward rather than re-deriving.

**One p-value cleared 0.05 and was one thin year.** Year-over-year across the
whole store reads +0.078, p = 0.025 — carried entirely by `2013→2014 = +0.632`,
computed on **13 brokers from 262 rows** against 2014's 82 from 26,900. The
panel collection starts in 2014; the 454 rows before it are exploratory probes.
Dropping 0.1% of the data takes the statistic to +0.032, p = 0.144. The floor
is post-hoc, is marked so, and both versions are printed.

**A null centred away from zero is normal in small samples, and it is invisible
without the distribution.** Track A's null means are **−0.207** and **+0.122**
depending only on whether periods are halves or quarters. Against zero, the
same store would have been reported as showing negative persistence and
positive persistence. This is now the third time in this repo that reading a
statistic against zero instead of against its own permutation null would have
produced a confident wrong answer. Treat the permutation null as the first
statistic, not the last.

**The censoring artefact is real and is not the explanation.** The top-10 cut
leaves only **51.1%** of rows two-sided, and which code gets truncated is a
strongly persistent broker attribute (year-over-year rank corr **+0.801**, an
order of magnitude above margin's). But it does not predict the measured margin
within a year (**−0.037**). A plausible mechanism that fails its second link is
still a failed explanation.

**What this does not say, and the distinction matters for what comes next.**
A broker code is not an investor class (§6 point 1). The Taiwan and Finland
results §12 cites identify *account types*, which those exchanges publish and
IDX does not. This finding is therefore about the instrument, not the
phenomenon: the broker code is too coarse to isolate a losing cohort at
fortnightly resolution. And Track B measures directional timing, not profit —
a market maker earning spread reads exactly zero on it while being profitable.

So §12's question stays open and its natural next instrument is the
**foreign/domestic investor-type split**, which is a class rather than a code
and which IDX does publish. More work on broker codes is not indicated.

## A7. The investor-type split exists, is free, and is NOT what this repo was computing

A6 closed by saying §12's natural next instrument is the foreign/domestic
investor split, because a broker code is not an investor class. That instrument
turns out to be available through the route already in use, and checking it
surfaced an error in this repo worth stating before any result.

**`fd=F` / `fd=D` is IDX's per-trade investor-domicile filter, and it composes
with `start`/`end`.** So the split costs one request per (ticker, window, view)
— the same economics A1 established for the combined panel, not the
per-ticker-day figure. Verified with a live request and a reconciliation
against 28 BBCA sessions, not assumed. This is A1's lesson applying a second
time: check the unit price before recording a constraint as arithmetic.

**The panel's `foreign_net` is a different quantity from net foreign flow.**
`flow_panel_build.flow_features` sums the brokers `config/brokers.yaml` flags
as foreign-*owned*. Against the domicile split on BBCA:

| | |
|---|---|
| correlation of the two nets | +0.749 |
| **sign disagrees** | **29% of sessions** |
| foreign share of gross | 74.6% by domicile vs 65.0% by member flag |

The cause is structural, not a config error: a foreign-owned member executes
for domestic clients all day, and YP (Mirae) is the largest RETAIL broker in
Indonesia while carrying a foreign flag. `ipot.parse_totals` carried a docstring
asserting that `F. NVal` "belongs to the F-broker definition, not to IDX's
per-trade foreign-investor flag" — the measurement says the opposite, the F
*view* reproduces `F. NVal` to **1.1%**, and the docstring is corrected. The
panel columns are kept so H9 stays reproducible, with the caveat at their
source.

**The footer was thrown away once and that cost a re-collection.**
`pullback_flow.fetch_window` calls `parse_table` and never `attach_totals`, so
none of the 31,824 combined windows carries its footer — which is why net
foreign value is unavailable for the existing panel without re-fetching all of
it. `investor_split_collect.py` keeps it.

**Two numbers that bound everything downstream.** F_net and D_net are
structurally exact mirrors, since every rupiah bought is a rupiah sold, so
their residual measures the top-10 censoring directly and needs no assumption:
**2.2% of gross at the median, 7.1% at worst**. And the filtered view's footer
`tval` matches neither the visible buy total, the sell total, nor their mean
(9.9% / 21.6% / 15.3%), so what it counts is unresolved — it is recorded as
unresolved and used in no statistic rather than dressed up as a coverage ratio.

**Where the test has power.** Foreign participation tracks liquidity hard —
decile 9 **49.4%**, decile 8 30.0%, decile 7 19.4%, decile 5 3.4% — so the
universe is the top strata. That is a statement about power, not convenience:
below it the foreign side is a rounding error and the comparison measures noise
at someone else's expense.
