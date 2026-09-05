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
| **News / narrative** | **Free** | *Added 2026-08-24, after this table's silence was misread as absence.* Public RSS: Google News (per-query and per-ticker), CNBC Indonesia, Kontan, Detik finance. Verified 200-OK with parseable items; Bisnis.com and idnfinancials 403. `src/idxbot/data/news.py`. **No point-in-time archive, so it may never enter a statistic** |
| **Overnight / global** | **Free** | *Added 2026-08-25.* The same unauthenticated Yahoo chart endpoint the `.JK` names use also serves ^GSPC, ^IXIC, DX-Y.NYB, IDR=X, ^TNX, BZ=F, CL=F, GC=F, HG=F, ALI=F, TIO=F, GLEN.L, BHP, RIO — and `^JKSE` itself. `src/idxbot/data/overnight.py`. **Palm oil `CPO=F` added 2026-08-25 after the claim that it did not exist was disproved (A14).** Thermal coal and LME nickel genuinely have no free front-month series; the listed miners stand in, labelled |
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

## A8. §12 is now closed on every free instrument, and the recommendation is to stop

A6 said the broker code was the wrong instrument for §12 and pointed at the
foreign/domestic investor split as the right one. That has now been collected
and tested, and it returns nothing.

**H12: all three pre-registered conditions fail, both classes.** 18 names,
329 fortnights, 5,993 class-windows, 2014–2026. Foreign margin **−1.70 bps** a
fortnight (p 0.692), domestic **+1.02** (p 0.816); 38% and 46% of years share
the pooled sign; both fail the 56 bps cost bar by 33× and 55×. Memo:
`reports/phase2b_investor_split.md`.

**The pre-registered prediction was wrong in sign** — foreign > 0 > domestic
was registered, foreign −1.70 / domestic +1.02 observed. Logged as failed
rather than reframed.

Three things worth carrying rather than re-deriving.

**This is a powered null, not an inconclusive one.** Null sd 6.34 and 4.95, so
the test resolves ±10–12 bps, and the 56 bps bar sits **8.8 to 11.3 null-sds
away**. A tradeable effect would have been found comfortably. That distinction
matters: "no effect large enough to trade" is a much stronger statement than
"no effect detected", and only the power calculation licenses it.

**Persistence failed by REVERSING.** Lag-1 autocorrelation of the annual margin
is **negative** for both classes (−0.410, −0.331) — a good year tends to be
followed by a bad one. And dropping the single largest year flips the pooled
sign for both. That drop-largest-year check exists because H11's headline was
carried by one thin year; it has now earned its place twice, and it should be
in every pooled statistic this repo reports.

**Structural checks are what license reading a number at all.** Annual foreign
and domestic margins correlate **−0.896**, opposite-signed in 10 of 13 years,
exactly as the zero-sum identity demands. And the censoring bound comes from
the F/D mirror residual rather than an assumption: **2.29% of gross at the
median**. That is far larger per window than the margin measured, so a POSITIVE
result of this size could not have been trusted — worth stating because the
same design would be tempting to reuse for a smaller question where it would
not be adequate.

### The recommendation

Three instruments, three nulls: aggregate flow (H9), broker identity (H10/H11),
investor class (H12). §12's premise may well be true of Indonesian retail, but
**no free instrument available here resolves it.** The remaining moves are an
account-type split IDX does not publish or a licensed daily feed; neither is
free and neither is worth buying on the strength of three negative results.

§4 ordered the work by cost-to-falsify and that ordering has been run to its
end on the flow branch. The unexplored cheap ground is §8's price/TA and
structural families, on a spine that Gate 0 passes — and they have not been
tested at all.

## A9. §8 has now been tested too, and the whole programme has one answer

A8 recommended stopping work on §12 and testing §8's price/TA and structural
families, which had never been tried. That is now done as H13, on a panel two
orders of magnitude larger than anything the flow branch had — 891 names,
daily, 1,989,504 in-sample rows, because price features need no broker data.

**The result inverts the flow branch's failure mode.** Flow had no signal.
Price has abundant signal and no economics: every one of the eight registered
features is significant, five carry HAC t above 10 against H9's −2.86, and
**every single one is net-negative after costs at every horizon**. The cost of
one rebalance is 1.7–1.9%; the gross quintile spread is 0.15–0.36% a period.
Memo: `reports/phase3_price_features.md`.

Three things to carry rather than re-derive.

**THE NEGATIVE CONTROL FIRED, and it changes how every t in this repo should be
read.** `squeeze` was registered as predicted-null on the grounds that range
compression forecasts the size of the next move, not its sign. It came back at
**t = +3.55**. At two million observations a t-statistic is nearly free: an IC
of 0.008 clears any conventional threshold while meaning nothing. **On panels
this size, significance is not evidence.** Effect size against cost is the only
thing that discriminates. Register a predicted-null feature in every future
sweep — it is the cheapest possible check that the pipeline is not manufacturing
its own signal.

**A rank tilt is not a return spread.** `lowvol` reads IC +0.0398 (t +14.49)
and a quintile spread of −0.430% a period (t −5.24) — a robust rank correlation
across the whole cross-section, negative in the tails where a few high-vol
names deliver enormous returns. H9 met the same wedge from the other side.
When the two disagree, **the spread decides tradeability**, because the spread
is what would actually be held.

**The binding constraint is the cost structure, and it is the same wall from
four directions.** 56 bps round trip plus a fraksi-harga half-spread that on
cheap names is several times the fee. Phase 2b found the median cohort round
trip already loses 25–32 bps to the spread before any fee. H13 finds real
price structure that the same spread swallows whole. Restricting to the most
liquid 5% halves the cost and collapses the signal's t fivefold; the best
post-hoc cell lands at +1.6%/yr with t = 2.00 out of a 24-cell sweep, and the
long-short construction it describes is one **A5 forbids** anyway.

### Where the programme stands

| | instrument | result |
|---|---|---|
| H9 | aggregate broker flow | no signal |
| H10/H11 | broker identity | no signal, no persistence |
| H12 | investor class | no signal, powered null |
| H13 | price/TA and structural | **strong signal, no economics** |

§4 ordered the work by cost-to-falsify and both branches have now been run to
the end. The honest summary of the research programme is that **the Indonesian
retail cost structure is wide enough to swallow every effect this data can
measure.** That is a finding about the market, reached four independent ways,
and it is the answer to §1's question.

What has not been tested is whether a lower cost base changes it — institutional
commissions, or a horizon long enough that turnover stops mattering. Both are
outside what §3's free data and A5's schedule can reach.

## A10. §9 is complete, and the null inverted its headline

A9 closed the price branch. §9's remaining sections — 9.4 fingerprints, 9.5
archetypes, 9.6 dossiers — are now done as H14, on 89 codes over 13 years.
Memo: `reports/phase2b_fingerprints.md`.

**The result: one style dimension persists, archetypes do not exist, and no
dossiers were written.**

**THE HEADLINE NUMBER WAS AN ARTEFACT, AND ONLY THE NULL SHOWED IT.** A
broker's share of gross persists year-over-year at **+0.912** — and its
label-shuffled null is **+0.919**, higher. The shuffle permutes labels *within*
each ticker-window, so every code keeps the exact set of windows it appeared
in; a code present in 5,000 windows still draws 5,000 times, so annual gross
measures **presence, not size**, and presence is conserved by construction.
`hhi` fails the same way (+0.603 against a +0.575 null). This is the fourth
time here that reading a statistic against zero rather than its own null would
have produced a confident wrong answer, and it is the most dramatic.

**What actually survives is `cross`** — the crossing ratio, at **+7.3 null-sds**.
How much of a broker's flow is matched on both sides is a genuine stable
property of the firm. `edge_buy` survives weakly (+2.7 sd); execution edge on
the sell side, and horizon (`ar1`), are noise. **The shape of the book persists;
nothing resembling skill does.**

**ARCHETYPES DO NOT EXIST — and the method mattered.** §9.5 asks for HDBSCAN
and GMM compared. **HDBSCAN finds zero clusters and labels 100% of codes
noise.** GMM and k-means, forced to k, return partitions that beat chance by
~15 points at k=2 and decay to chance by k=5. The lesson generalises: a method
that must return k clusters cannot tell you there are none, so always pair it
with one that can.

**§6.3's predicted degradation is not there.** Fingerprint distinctiveness is
flat 2014–2026 (slope −0.0026/yr, rank correlation with year **+0.132**, last
above first). Order-splitting may be spreading; it is not visible at
fortnightly top-10 resolution.

**No dossiers, by prior agreement.** H14 pre-registered that §9.6 dossiers
would be written only if archetypes proved stable. They did not. §9.6 is the
section most exposed to fabrication — a headed template invites filling in —
and the conditional existed precisely so the decision could not be revisited
after the answer was known.

§9 is therefore complete: **broker codes are stable identities whose stable
part carries no information about returns.**

## A11. The daily brief exists, and building it turned up one thing worth a real test

A9 and A10 closed the research programme. What was asked for next was not a
statistical edge but a twice-daily situational read: what the market is doing,
the narratives, potential candidates, is the run over or just started. Built as
`src/idxbot/report/brief.py` + `scripts/brief.py`, 26 tests. Memo:
`reports/daily_brief.md`.

**Three of the four are reachable; the fourth was written off and should not
have been — see A12.** The co-movement half of the narrative section is real
and stands: components fitted on the 250 sessions ending the day *before* the
bar separate banks, coal-and-metals and the Prajogo complex out of returns
alone, with no sector data anywhere in the project. They are printed as
constituent lists; naming them is the reader's interpretation, per §9.6's rule.
The news half was declared unavailable without a single request being made;
A12 records how that went.

**THE ONE NEW NUMBER, AND IT IS A LEAD, NOT A FINDING.** Bucketing bars on four
dimensions fixed before any cell was seen — leg, run age, extension (`run_z`,
cut per leg), index vol — gives 54 cells over 1,127,670 liquid pre-holdout
bars. Largest cell excess over the same-day base rate **+1.67% per 20
sessions**, against a label-shuffled null of **+0.37%** (p95 0.53%), cell
spread **0.68%** against a null **0.13%**, p = 0.000 on 200 draws. Old stretched
advances continued; advances old in time but shallow in price underperformed.

It is not reported as tradeable and must not be: it is the **maximum of 54**
in-sample post-hoc cells, and **H13 measured very nearly the same thing and
found it net-negative** — `mom12_1`, `hi52` and `atr_mom20` all encode "old
stretched advance". Logged as **O1** in `hypotheses.md`, not H15; the trial
count stays at 41 because no hypothesis was tested. Turning it into H15 means
registering the cells, the rule, the horizon and the cost model in writing
*first*, then spending the holdout once. **The 24-month holdout is untouched
and every reference table here is built on `holdout == False` rows** — a brief
running twice a day would otherwise spend it inside a week.

**Four defects surfaced, and all four printed believable output.** Worth
carrying because each generalises.

*A current date column is not a current cross-section.* The panel held bars
through 2026-08-21 with only 46 names in the last four sessions — a watchlist
refresh — and "71.7% of names above the 20-day" was forty large caps.
`resolve_asof` now falls back to the last representative session and says what
it skipped.

*Rolling windows on a pivot are indexed by the UNION of trading days.* One
suspended name inserts NaN rows it never had, and `min_periods` then fails for
every column at once: the first output read "0 of 830 names above the 200-day".
Group by ticker; never roll on a pivot.

*`np.argmax` returns the index of a NaN.* NaN compares False against
everything, so the scan never displaces it. Against 2,327 spine bars with a
non-positive adjusted close this produced **915 "advances" with a negative
return from their own anchor** — impossible by the definition, sitting quietly
in a conditional table. A further 164,627 bars carry `vol60 <= 0`, which makes
a motionless name infinitely extended.

*`np.isin` is a set test and has no place in a bootstrap.* Selecting drawn
blocks with it silently dropped duplicates, so every resample was smaller and
less variable than its sample and every interval came out too narrow. **A
bootstrap that understates uncertainty is worse than none, because it looks
like rigour.**

**One limitation kept rather than patched.** `give_back` — how far price has
come back from the leg's extreme — is arguably the best single "is it over"
variable and is deliberately NOT a bucket dimension, because the four were
fixed before any cell was seen and adding a fifth after noticing which cells
came out large is the trap the trial count exists to catch.

## A12. The news source that was declared absent without one request being made

A11 shipped a function whose whole job was to print *"there is no news,
filings or announcement source anywhere in this repo, and §3's data table
lists none."* Both halves were true. The conclusion drawn from them — that a
news narrative was unreachable — was false, and **no request was made before
writing it.** §3 listed none because nobody had looked.

Eight endpoints were then tested in about a minute:

| endpoint | result |
|---|---|
| Google News RSS, arbitrary query | **200, 100 items** — takes any query, so it works per-market AND per-ticker |
| CNBC Indonesia `/market/rss` | **200, 100 items** |
| Detik finance RSS | **200, 100 items** |
| Kontan investasi RSS | **200, 25 items** |
| Yahoo per-ticker `.JK` RSS | 200 but 0–1 items — useless for IDX |
| Bisnis.com RSS | 403 |
| idnfinancials RSS | 403 |
| idx.co.id announcements | 403, Cloudflare, exactly as `docs/FULL_REKAP.md` records |

On the first real run it returned an ADHI trading halt over a missed bond
coupon, a UMA flag on PACK/FUJI/BDKR, an SWAP IPO bookbuilding, and a Rp140.7bn
rights issue on BABY. **None of that is inferable from a price series**, and
ADHI's halt explains a missing bar the panel would otherwise treat as a data
gap.

**THIS IS THE THIRD TIME.** A1 priced IndoPremier per ticker-day and called an
800-name panel infeasible when the same endpoint returns a whole window per
request — off by sixty. A7 assumed the investor split needed a per-ticker-day
pull when `fd=F` composes with `start`/`end`. A11 assumed no news source
existed without issuing a request. The shape is identical: **a true observation
about what is present, converted into a false claim about what is possible.**

`docs/STANDING_ORDERS.md` is the corrective, and its short form belongs in
`~/.claude/CLAUDE.md` so it applies beyond this repo. The operative rules:
"not in X" is never "does not exist"; one query is a sample of size one;
testing an endpoint beats reading about it; name the tool you used before
declaring a limit; and a negative answer is unfinished until it says what
would change it and what that would cost.

**Two things about the news layer that are load-bearing.**

*It is quarantined, and a test enforces it.* There is no point-in-time news
archive, so a headline visible today cannot be reconstructed as it stood on a
past bar — anything under `spine/` or `features/` importing it would make every
downstream backtest look-ahead by construction. `tests/test_news.py` walks the
AST of both packages and fails if either imports it. It is for reading.

*The relevance filter is not optional.* Many IDX tickers are ordinary words —
GULA (sugar), KOTA (city), RAJA (king), CASH, BABY, COAL — so an unfiltered
query returns the commodity and the municipality. Requiring the ticker as a
standalone token cuts GULA from 100 items to 73 and KOTA from 100 to 40.

**And the refresh that was actually broken is fixed.** `daily_update.py` only
ever refreshed a 40–60 name watchlist, so the panel carried 830 names through
2026-08-14 and then 46 for four more sessions. The brief detected the ragged
edge and fell back, which was right but left it permanently stale.
`scripts/refresh.py` pulls the whole universe (~0.34 s/name, ~5 min for 843)
and rebuilds the panel and the brief's tables behind it: **829 names, 98%,
current.**

## A13. The pre-open brief that knew nothing, and the estimator that inverted its headline

A11's brief shipped a `--session pre` mode whose own banner admitted the
problem: *"a morning run and an evening run differ only in what has settled,
not in what is known."* That makes half the stated use case pointless, because
what a pre-open read is FOR is the overnight gap.

**The gap was free and had never been fetched.** Jakarta closes 15:50 WIB =
08:50 UTC; New York closes 20:00 UTC and London 15:30 UTC. `YahooOHLCV` —
already in this repo, already serving every `.JK` name — returns all of them
from the same unauthenticated endpoint. This is A12's lesson again, one section
later: the tool was in hand and was not tried.

**THE CLOCK IS THE WHOLE DIFFICULTY.** Wall Street's bar dated 2026-08-24 lands
eleven hours AFTER Jakarta's bar of the same name, so it is overnight news;
Tokyo's bar of that date closed two hours BEFORE Jakarta's and is not. A naive
`date > idx_day` test finds nothing and reports a silent NaN for the entire
board, which is what the first version did. `AFTER_JAKARTA` encodes which is
which, and the historical alignment deliberately uses the previous session for
every market rather than risk a leak.

**AND THE HEADLINE INVERTED WHEN THE ESTIMATOR CHANGED.** The first
sensitivity table used Pearson and reported that the S&P 500 has essentially no
relationship with IDX (r = **−0.001**), with Glencore the strongest link. On
rank correlation the S&P is the **strongest** at **+0.207 (z = +14.9)** and
Glencore third. The cause is measured, not guessed: these series carry kurtosis
from 10 to **2,800**, and Pearson on a sample like that is a statistic about
its four largest days.

Two data defects fell out of the same check. Yahoo's `IDR=X` carries
**decimal-shift errors** — 2010-11-01 prints 888.11 against a true ~8,881 and
reverses the next day, a +903% return followed by −90% — which are dropped and
counted, not winsorised into something plausible. And `^TNX` is a **rate**: it
fell 0.93 → 0.50 in March 2020, a real 43 bp move and a spurious −46% return,
so rates are differenced and prices are not.

**This is a different error from the null one.** A10 and A11 record four
occasions where reading a statistic against zero instead of its own
permutation null gave a confident wrong answer. This is the fifth confident
wrong answer but the first from a mis-specified *estimator* rather than a
missing *benchmark*. The generalisation: check the distribution before choosing
the statistic. The block bootstrap was verified unbiased against a synthetic
sample of known correlation, which is what let the blame be pinned on Pearson
rather than on the intervals.

**What the measured table actually says**, on 6,091 pre-holdout sessions:
S&P +0.207, Nasdaq +0.199, BHP +0.185, Rio +0.169, Glencore +0.160, Brent
+0.123, copper +0.096, DXY **−0.080**, USDIDR **−0.042**, and Asia
indistinguishable from zero. So the conventional reads — a strong dollar is a
headwind, the commodity complex matters — are real and signed as folklore says.
**None of it is tradeable.** The strongest explains about 4% of variance, a
fraction of the 56 bps round trip.

**The brief is also now one document rather than eight lists.** The candidate
section printed the top names on each registered feature in turn — forty
tickers in eight columns with no name appearing beside its own context. It is
replaced by a single fused watchlist: today's move, run state, the historical
excess of the cell it occupies, round-trip cost, net, an event-tag column from
the news layer, and a COUNT of how many registered features rank it top-decile.
A count, not a blend — a composite of eight separately-tested features is a new
signal wearing their credibility.

## A14. A parallel source sweep, and three defects it found in code written the same day

A13 closed the overnight board. A systematic probe of every free source the
brief does not use — 8 categories, tested rather than researched, then
adversarially re-verified — returned three corrections to code written hours
earlier, plus a shortlist worth building.

**PALM OIL WAS REACHABLE ALL ALONG.** `overnight.py` asserted *"there is no
Yahoo symbol for Bursa Malaysia CPO or for LME nickel."* **`CPO=F` carries
3,929 daily bars from 2010 to the current session**, from the same
unauthenticated endpoint the rest of the board uses. Indonesia is the world's
largest palm-oil exporter, so this was not a small omission. **This is the
fourth instance of the A12 shape, and the first committed on the same day as
`docs/STANDING_ORDERS.md` by its own author.** Writing the rule down does not
make it operate.

The nickel half of that sentence is now *properly* true: tested across 21
symbol roots and Yahoo's own search endpoint.

**Measured, palm oil does nothing at daily frequency:** r = **+0.026**,
z = +1.5, CI [−0.008, +0.061], 6.5% stale. The conventional link between CPO
and IDX is not visible in the daily cross-section. That is a real finding and
it was only obtainable by adding the series the docstring said did not exist.

**`MTF=F` WAS MISLABELLED.** The prose called it Newcastle coal. It is **API2
CIF ARA** — Rotterdam-delivered European coal, median ratio 0.781 against the
World Bank's `Coal, Australian` over 139 months. Wrong basin for an Indonesian
exporter panel, and dead since 2025-12-26 regardless.

**THE AUTO-REJECTION BAND WAS WRONG FOR THE THIN BOARD.** `limit_moves` called
`auto_rejection(p, day)` and took the default `board="main"` for every ticker.
`reference.py` has carried the thin-board ladder all along — Papan Pemantauan
Khusus and Akselerasi trade a flat **±10%** against the main ladder's
**+35%/−15%** — and `infer_board` derives membership from IDX's published
six-month-average-price rule. **41 of 818 live names sit on that board** and
every one was being tested against a ceiling three and a half times too high;
fixing it immediately surfaced a limit-up the main ladder had missed (ARA
1 → 2). The machinery existed; the caller did not use it.

Two notes on the fix. `infer_board` answers `"unknown"` for a sub-Rp-51 name
before 2023-06-12, because Papan Pemantauan Khusus did not exist then — those
names are skipped rather than banded on a guess, and stay in the denominator so
the printed ratio is honest about what was untestable. And a test asserting the
new behaviour failed until its synthetic panel was dated after that rule
existed: the point-in-time discipline working exactly as designed.

**What the sweep found and did NOT get built, with the reason.**

*The IDX-IC sector map is reachable*, contradicting a bounded "not found" this
session recorded. Eleven CSVs under `wildangunawan/Dataset-Saham-IDX`,
**934 tickers across the 11 official IDX-IC sectors**, plus listing board,
listing date and **shares outstanding** — the last being the series whose
absence forces the turnover-weighted index proxy. **CC BY-NC 4.0**, frozen at
2024-07-10, so it misses the 41 post-July-2024 listings.

*TradingView's screener* covers 838/838 with index membership for 18 IDX
indices verified exactly (LQ45 45, IDX30 30, IDX80 80, KOMPAS100 100). It is
**not built**, because A5 is explicit that a host is added only after the user
has checked its licensing, and this one is an undocumented internal endpoint
whose `/indonesia/scan` path its own robots.txt disallows. It is also *wrong*
on taxonomy — ASII, a top-ten name, comes back as "Technology Services". That
is the user's call to make, not this repo's.

## A15. The entry rule was not a rule, and the exit layer that was missing works

A11 logged the multiplier cell as O1, a lead not a finding. H16 spent the
holdout on it and measured 2/10 doublers with a mean PEAK of +102.2% against a
realised +15.1%. The obvious next move was an exit layer. Building it turned up
something bigger first. Memo: `reports/exit_rules.md`, logged H17 and H17b.

**THE ENTRY RULE WAS UNDER-DETERMINED AND NOBODY NOTICED FOR TWO HYPOTHESES.**
Re-implementing it to generate historical cohorts produced a DIFFERENT basket
from H16's on the same date — IMPC where H16 drew MERI — and therefore +26.3%
against +15.1% for the identical exit. Neither had a bug. The rule scores names
by the historical P(2x) of the cell they occupy, and there are ≤125 cells
against ~800 live names, so on 2025-08-25 **seventeen names shared the
tenth-place score and the top thirty held four distinct scores.** "Take the top
ten" was decided by `sort_values` stability.

Measured rather than argued: 500 random tie-breaks on that cohort span
**−29.2% to +36.6%, sd 13.0%**, median **+6.9%**. H16's draw sits near the 75th
percentile of baskets it had no reason to prefer. Across 211 pre-holdout
cohorts the cut falls inside a tie **64%** of the time, with a within-cohort sd
of **4.31%**. **That variance is in no interval this repo has ever printed.**
H16's headline is revised: the centre of that cohort is +6.9%, not +15.1%.

The fix is not a better tie-break but the absence of one — `select(tie="all")`
holds the entire tied group. It measures slightly WORSE (−4.42% against
−2.54%) and is still correct, because the −2.54% was never a property of the
rule. **A discrete-cell score over a large cross-section is tied by
construction; check the tie structure before quoting any top-N from one.**
The daily brief is unaffected — it ranks on continuous features and on today's
absolute move, neither of which ties.

**The exit layer itself works, modestly and measurably.** 32 rules fixed before
scoring, purged walk-forward over 176 cohorts 2008–2023, cohort-level moving-
block bootstrap: selected rule **+0.94%** against buy-and-hold **−3.19%**,
difference **+4.13% [+1.87%, +6.33%]**. It differed from buy-and-hold in 94
cohorts and **won 74 of them — 79%, p = 1.8e−8**. 153 of 176 cohorts chose
`trail 15% armed +50%`. Mean hold 210 sessions against 250.

**AND IT DOES NOT FIX THE DOWNSIDE, exactly as pre-registered.** P(−50%) is
15.0% against 16.3%; every armed-trail variant reads an identical 15.1%,
because a name that falls from entry never arms. The frontier is explicit: a
hard stop takes P(−50%) to ~0% and costs 6–8 points of median return. **For a
rule selected ON P(2x), cutting the left tail cuts the premise.** No rule in
the catalogue is better on both axes.

**Three method defects, all committed while writing the fix for the last one.**

*The walk-forward leaked.* Monthly cohorts holding a year means last month's
outcome is eleven months from being known, and the first version trained on it.
`purge=True` restricts training to settled cohorts.

*The cohort bootstrap treated overlapping cohorts as independent* — the SAME
error H16 named about its own twelve cohorts ("effective n is ~1, not 12"),
reintroduced one layer up inside the function written to avoid it. An iid
resample over 176 monthly year-holds returns an interval ~3.4x too narrow.
Every interval in the memo roughly doubled when it became a moving-block
resample. **The unit of resampling and the unit of independence are different
questions, and getting the first right does not settle the second.**

*`DatetimeIndex.asi8` is microseconds on a `datetime64[us]` index* and
nanoseconds on a `[ns]` one, so a hardcoded divisor returned a block length of
11,783 for monthly cohorts. And a long block **degenerates** — widths 0.049 at
b=1, 0.105 at b=13, 0.047 again at b=63 — so it is capped at a fifth of the
sample.

## A16. Indicators, and the objective that was deciding the answer unstated

H17 built exit rules that see only the price path. H18 added the indicator
layer — EMA, ATR/chandelier, stochastic, turnover z — and a live position
monitor. Memo: `reports/exit_indicators.md`, logged H18.

**NEWS CANNOT ENTER A BACKTEST HERE AND THAT IS STRUCTURAL.** No point-in-time
archive exists, and `tests/test_news.py` fails the build if `spine/` or
`features/` imports the news module. A news-conditioned exit rule in a backtest
would be look-ahead by construction. It ships as a live-only overlay in
`scripts/positions.py` — standing event tags printed beside the levels,
computed into nothing, with a test that a dead news feed changes no number.

**THE HIGH AND LOW WERE THERE ALL ALONG.** The spine panel carries no high or
low, so ATR and stochastic look uncomputable and close-only proxies look
mandatory. `data/cache/ohlcv/` has full OHLCV for **all 919 panel names**, with
the cached close matching the panel's on **100%** of sampled overlapping bars.
This is the A12 shape a fifth time — a true statement about one artefact read
as a fact about the world — and the check took one minute. Rebasing high/low
with the panel's own adjustment factor also surfaced SCCO and SINI, two of the
three repaired names, as the only tickers where vendor and panel close disagree.

**THE HEADLINE FINDING IS THAT THE OBJECTIVE DECIDES THE ANSWER.** The same
walk-forward over the same 58 rules and 176 cohorts:

| objective | median | mean | P(2x) | most-chosen |
|---|---|---|---|---|
| median | **+3.16%** | +7.10% | **3.5%** | stoch rollover armed +50% |
| mean | −2.86% | **+18.42%** | 11.0% | volume climax armed +50% |
| p2 | −3.36% | +19.97% | **11.6%** | volume climax z3 armed +50% |
| buy&hold | −4.30% | +18.80% | 11.6% | — |

**No rule beats buy-and-hold on mean return or on P(2x).** Optimising the
median cuts the doubling rate from 11.6% to 3.5%. The entire measured
improvement — H17's +4.13% and H18's +6.35% alike — is a median effect. For an
entry rule selected ON P(2x), choosing its exit on median return optimises
against its own premise, and H17 made that choice without stating it. **State
the objective with every rule-selection result; it is not a detail.**

Against H17's incumbent the indicator layer is **+1.21% [+0.07%, +2.42%],
sign test p = 0.014**, which does NOT clear the 49-trial Bonferroni bar of
0.001. Suggestive, not established.

**H18b was pre-registered and FAILED.** An indicator stop does not cut P(−50%)
more cheaply than a hard stop: the undominated frontier is entirely price
rules, and every armed indicator rule reads the same 15.1% because a name that
falls from entry never arms.

**THE NULL EARNED ITS KEEP TWICE.** A random exit date behaved exactly like a
matched-length hold (−4.6%/122d against `hold 126`'s −5.0%/125d), so the
pipeline is not manufacturing signal — and it **beat every hard stop**
(−10.6% to −11.7%). A coin flip does better than a 15–30% hard stop here.

**Three code defects, each found by a test written after the bug.**
*A blanket `except TypeError` around the thing being measured* recorded every
failure as "no data", silently dropping one-argument rules and disguising real
crashes. *`id()` is not a cache key* — CPython reuses it after collection, so a
dead lambda's arity was served for a live rule. *A label containing another
label as a substring* — `(unarmed)` contains "armed" — silently matched the
wrong rows in a filter.

## A17. The recovery curve — the conditional two studies had assumed

H17 and H18 both picked a trailing-stop distance by GRID SEARCH and never
measured the conditional that distance is meant to encode: given you are
already X% below the peak, what are the odds it comes back? H19 measures it on
243,977 armed liquid pre-holdout bars. Memo: `reports/recovery_curve.md`.

**P(new high within 60 sessions), by give-back:** 81.3% at −5%, 56.1% at −10%,
38.8% at −15%, 27.1% at −20%, 17.3% at −25%, **11.3% at −30%**, 5.9% at −40%,
0.8% below −60%. It crosses one-half between −5% and −10%.

**H19b WAS PRE-REGISTERED AND FAILED, INSTRUCTIVELY.** The prediction was a
depth where the MEAN forward return turns significantly negative. There is no
such depth — the mean never leaves [−2.2%, +7.1%] and its interval covers zero
from −25% down, because a thin tail of enormous recoveries keeps expected value
flat all the way to −60%. What turns negative is the **MEDIAN**, at **−10 to
−15%**. The mean/median wedge that decided H18 arrives here from a completely
different direction, and it is now the third time this repo has found the two
disagreeing on the same question.

**AND IT VALIDATES A NUMBER THAT HAD ONLY EVER BEEN SELECTED.** The grid chose
`trail 15%` by optimising cohort median; the conditional curve puts the median
crossing at −10 to −15%. Two unrelated routes, same answer. In ATRs that is
2.1–3.2, which is where `chandelier 2x ATR` sits — so H18a's support was the
same fact seen from the side.

**INDICATORS PREDICT RECOVERY, NOT RETURN.** 53 (indicator × depth) cells:
**14 clear Bonferroni on P(new high), 13 of them positive, median shift 12.6
points** — but only 4 clear on forward return and those split 2 positive / 2
negative. Being above the EMA50 in a shallow drawdown is worth up to **+22.4
points** of recovery probability and nothing reliable in return. Of the
indicators tested, **above EMA50 is 9+/0− across depths, and the stochastic
cross is 5+/6− — noise.** The help concentrates at −5 to −15% and decays to
nothing past −20%.

So `ema50 break armed +50%` is what the curve endorses: H18's table gives it
+0.8% median, **+15.0% mean, P(2x) 7.9%** against `trail 15%`'s +0.9% / +11.4%
/ 7.3% at identical P(−50%). Best-motivated, not out-of-sample — read off a
table already seen, holdout spent.

**THE NULL HAD TO PERMUTE NAMES, NOT ROWS.** A name contributes ~20
near-identical bars a month, so a row shuffle destroys the label while leaving
the null far too tight; every z came back inflated and one read **−8.7** before
becoming the headline. Permuting whole (ticker, month) blocks is the fix, and a
test now makes the difference visible on synthetic data instead of asserting
it. **Clustered data needs a clustered null — getting the unit of resampling
right in one place does not carry to the next.** Separately, a cell with a flag
true for 0.2% of rows produced the first table's largest effect (−46.1%): both
sides now need 300 observations and a 5% share.

## A18. Portfolio accounting retracts H17 and H18, and the half-split ends it

H17 and H18 both selected an exit rule by maximising the average cohort
MEDIAN. **An equal-weighted holder is paid the MEAN.** H20 redoes everything on
portfolio accounting — twelve month-offset slots, capital redeployed when the
exit fires — and runs the random-entry control that had never been run. Memo:
`reports/portfolio.md`, logged H20.

**H17 AND H18's HEADLINES ARE WITHDRAWN.** `trail 15% armed +50%` (+4.13% in
H17) and `stoch rollover armed +50%` (+6.35% in H18) turn a 6.4x buy-and-hold
into **1.6x and 1.4x**, at **+2.4% and +1.8% CAGR against hold's +10.5%**. They
are the two worst rules of the seven tested. They won the median by cutting the
right tail, and the right tail is where the return is. **State the objective
with every selection result, and check it is the one the investor receives.**

**THE HALF-SPLIT IS THE ONLY THING THAT CHANGED MY MIND.** Paired per slot
against buy-and-hold, run inside each half: exactly **1 of 6** rules is positive
in both — and with six rules and a coin flip per half you EXPECT **1.5**. So the
survivor (`trail 30%`) is not evidence. `stop 25%` (10/12 early) and
`ema50 break` (10/12 late) win in OPPOSITE halves, which is what regime noise
looks like. **No exit rule is established, including the one A17's recovery
curve endorsed on mechanism.**

**AND THE ENTRY IS A COIN FLIP SINCE 2017.** One pre-specified comparison,
paired per slot, picks vs random draws from the same pool: early half
**+5.86% CAGR, 204/240 slots (85%)**; late half **+0.36%, 122/240 (51%)**. The
full-sample +3.99% averages a regime where it worked with one where it did not.

**PER-POSITION STOPS CAN INCREASE PORTFOLIO DRAWDOWN.** `stop 25%` caps every
name at −25% and has a per-name P(−50%) of **0.2%** — and the **worst portfolio
drawdown in the table, −68.7%**, because it realises losses 29 times against
hold's 18 and redeploys straight back into the same bad regime. Position-level
risk control is not portfolio-level risk control.

**THE BUG THAT CHANGED THE WINNER.** The slot scheduler converted held sessions
to calendar days and searched the date index; a 30-day lock opened 1 February
ends 3 March and **skips the 1 March cohort entirely**. The penalty scales with
turnover, so it fell hardest on short-holding rules — the exact comparison the
study exists to make. Before the fix `stop 25%` looked best at 12/12 slots and
t = +7.05. Locking now happens in cohort-index space, which cannot skip.

**THE LESSON, NOW THREE TIMES OVER.** A within-sample consistency statistic over
correlated units — 12 overlapping slots, 20 redraws, 188 overlapping cohorts —
reads as overwhelming (12/12, t = 7.05) and says almost nothing about whether an
effect replicates. Only the half-split does. It is cheap and it should run
before any rule is reported, not after.

**The only findings that replicate are negative:** `chandelier 2x ATR armed
+50%` and `stoch rollover armed +50%` are worse than holding in both halves.
~~The defensible position is buy-and-hold on the picks~~ — **retracted by A19,
which added the benchmark this study did not contain.**

## A19. The benchmark that was never in the study, and it ends the entry too

A18 closed by naming buy-and-hold on the picks as the remaining defensible
position. **Every number it used to reach that was picks-versus-picks or
picks-versus-a-random-draw-from-the-same-pool. The IHSG is in none of it**, and
`_JKSE.csv.gz` had been sitting in `data/cache/ohlcv/` the whole time. Memo:
`reports/portfolio.md` §5–§6, logged H21.

**THIS IS THE A12 SHAPE FOR THE FIFTH TIME, in its worst form yet.** A1 priced
IndoPremier per ticker-day. A7 assumed the investor split needed a per-ticker-day
pull. A11 declared no news source existed without issuing a request. A14 said
palm oil had no Yahoo symbol on the same day `docs/STANDING_ORDERS.md` was
written. Each of those was a *missing source*. This one is a **missing
comparison**, which is worse: the data was already loaded, no request was
needed, and the omission did not block a result — it manufactured one. A study
can be internally impeccable, permutation-nulled, half-split, purged and
bootstrapped, and still answer the wrong question because the alternative the
reader would actually take was never priced.

**Make it like for like first, and both corrections favour the picks.** The
names run on `adj_close` and are TOTAL returns; `^JKSE` is a PRICE index. The
adjustment identifies itself — `log(adj_close/close)` steps only at corporate
actions, and back-adjustment makes dividend steps positive going forward: across
1.75m steps, **3,707 small positive and ZERO small negative**. Yield rises
monotonically with liquidity, 0.65% in decile 1 to **2.01% in decile 10**, so
the cap-weighted index yields the large-cap 1.77% against the picks' 1.27% and
correcting the index UP is the bigger move.

**Picks +10.5% against index +12.7% on a total-return basis — a −2.2% a year
shortfall, negative in BOTH halves** (−3.3% early, −1.5% late). It buys no risk
reduction: shallower drawdown than the index full-sample, *deeper* in the recent
half. Three uncorrected biases run against the picks and one for them, leaving
an honest **−1.2% to −1.7%**.

**AND THE FIRST VERSION OF THAT TABLE HAD TWO WINDOW MISMATCHES OF MY OWN,
both flattering the picks.** Twelve slots begin and end in twelve different
months, so one global index window compares each slot to a period it did not
occupy; and a slot's last position stays open for its holding period after the
final entry, so its span runs a year past the last cohort date while the index
stopped at it. Paired per slot over each slot's own span, the shortfall is
**−2.53% a year, nine of twelve slots losing, 95% CI [−4.64%, −0.42%]** — the
fix made it stronger. *Every* error in this section ran the same way: comparing
quantities measured over different windows.

**AND IT IS THE SELECTION, NOT THE TOLL, AND NOT SIZE.** Refunding every one
of the eighteen round trips — worth **1.76% a year** — still leaves the picks
behind by 0.82%, so a lower-turnover version of the same rule does not rescue
it. And the small-cap rescue came back **backwards**: by within-cohort
liquidity tercile the random pool reads +7.9% / +4.8% / **+3.7%** against the
index at −3.5% / −6.7% / **−9.5%**, so the LIQUID end is the worst and moving
upmarket makes it worse. The shortfall is that an EQUAL-WEIGHTED basket of IDX
names lost to the CAP-WEIGHTED index over this sample: a handful of mega-caps
carried it, and even this pool's liquid tercile sits far below those in
capitalisation, inheriting none of that return while keeping the equal-weighting
penalty.

**And the picked half of that test was refused rather than reported.** Splitting
a twelve-name basket three ways leaves the liquid cell scoring 54 of 212 cohorts
at a median of four names. A first draft printed **−13.9%** for it — the
smallest cell producing the largest effect, the degenerate-cell trap this repo
has already recorded twice. The readable half (random draws, full baskets in
every tercile) is what is quoted; the unreadable half said "insufficient data"
and priced answering it at ~20 minutes of rebuild.

**THEN THE 20 MINUTES WERE PAID, because naming a cost and not paying it is the
premature closure `docs/STANDING_ORDERS.md` exists to stop.**
`scripts/liquid_rerank.py` restricts the universe to the liquid tercile BEFORE
ranking, so the rule takes a full basket there — median 11, min 10, all 179
cohorts scored. Prediction of failure registered in the docstring first. Result:
picks **−5.0%** CAGR, random draw from the same universe −2.4%, index **+9.0%**,
paired shortfall **−15.08%**, **0 of 12 slots**. Applied upmarket the rule goes
from beating its own pool by 4.8 points to trailing it by 2.6. The size
explanation is dead.

**AND IT GIVES THE SHARPEST DESCRIPTION OF THIS ENTRY RULE IN THE PROJECT.**
Per name over the year: picks mean −2.03% and **median −10.48%** against the
rest of the liquid universe at −0.12% and −3.27% — while **P(2x) is 4.89%
against 1.51%.** The rule does exactly what it was built to do, more than
tripling the doubler rate, and loses money doing it. **It is a lottery-ticket
selector: it buys convexity, and on this cost structure the convexity costs more
than it is worth.** Every other result lines up behind that one sentence — H16's
2 doublers with a mean PEAK of +102.2% against a realised +15.1%, A15's exit
frontier that cannot cut the left tail without cutting the premise, and this
shortfall. It also selects wider-spread names *within* the liquid tercile,
paying 1.35% a round trip against 1.02%.

**It does not rest on significance either.** Slot dispersion [+6.7%, +13.8%]
contains the index, so read as a tie the picks still cost 18 round trips,
single-name concentration and a −50% drawdown to reach the same place. **A tie
against the cheap alternative is a loss for the expensive one.**

**Two of A18's own sentences were also wrong, in opposite directions.** "Since
2017 the entry is a coin flip" was a POWER statement written as an EFFECT
statement — A8's exact distinction, violated one appendix after citing it. Taking
the slot as the unit, the late half's interval is [−4.15%, +4.14%]: it excludes
the early half's +6.08%, so the break is real, but it **cannot rule out +4.1% a
year**. And the break is a decay, not a cliff — rolling six-year windows are
**positive in 14 of 15**, trending −0.45% per year of start date. The halves cut
where the decay bites.

**A18 ALSO DELETED FIFTEEN TESTS WITHOUT FAILING ANYTHING.** Its tests were
written to `tests/test_portfolio.py`, which already existed — 15 tests for
`src/idxbot/portfolio.py`, a CLI-exposed module — and replaced them. Both files
held exactly 15, so the suite total did not move and no run reported anything.
`git status` said ` M` rather than `??`; that one character was the entire
warning. **A module can go from covered to uncovered without any test failing,
because the evidence of coverage is the tests themselves.**
`tests/test_coverage_map.py` now asserts every module under `src/idxbot/` is
named by some test, listing the two genuinely uncovered ones rather than hiding
them. Suite 1,892 → **1,910**.

**AND THE SAME SHAPE WAS FOUND A SIXTH TIME, WITH A NEW TWIST.** A12 corrected
`brief.news_caveat()` to carry its own retraction. The **module docstring above
it still said** *"There is no news source anywhere in this repo and §3's data
table lists none"* — the exact refuted claim, sitting in the file A12 was about,
for a day after the function beneath it had been fixed. **Fixing the code is not
fixing the claim.** A reader opening the module met the retracted version first.
Both are now corrected and the sentence is kept in place, marked, rather than
deleted.

## A20. The horizon was never varied, and varying it inverts the answer

A19 closed the programme with "buying the index is the defensible position".
The user then set the goal "make 8 of 10 multi-baggers reachable", which forced
the one question §7's decay curve and every later study had left alone:
**every P(2x) in this repo is measured at 252 sessions, because H16 chose that
horizon.** A9 named "a horizon long enough that turnover stops mattering" as
untested and it stayed untested for five appendices. Memo `reports/horizon.md`,
logged H23.

**THE ANSWER TO THE GOAL IS NO, AND THE CEILING IS 7 OF 10.** Unconditional
P(touch 2x) runs 9.5% (1y), 27.0% (3y), 39.0% (5y), **55.5% (10y)**. So 8 of 10
needs an 8.4x lift at one year and **1.44x** at ten. Nothing delivers it: the
best long-horizon cell is 1.29x. The reachable ceiling is **72.8%** — about
seven of ten — by holding the most liquid names for a decade.

**TWO BUGS MADE THE FIRST TABLE READ 73.8% AT 7.5 YEARS.** `MU.PX` is a list of
CUT EDGES for the price bucket, not a `[min, max]` pair, so
`close >= PX[0] & close <= PX[1]` silently restricted the universe to sub-Rp50
names — the penny board, 336 tickers out of 725. And eligibility was applied to
**every bar**, cutting the forward path wherever a name left the universe;
eligibility is a condition for BUYING, and once held the path is whatever the
name does. Requiring a full window on top of that discarded **91% of 7.5-year
cohorts** and measured the doubling rate of the survivors. Three separate
errors, all pushing the same way, all printing a believable table.

**AND THE DIRECTION INVERTS EVERY ONE-YEAR RESULT IN THIS REPO.** A19 measured
the liquid tercile as the WORST cell, trailing the index by 9.5% a year. At a
ten-year horizon the most liquid names are the best on every axis at once:
touch rate **69.1%**, P(−50%) **13.7%** against a 28.6% base, median
**+174.7%**. That is not a contradiction, it is the horizon: a one-year hold
pays ~1.3% round trip every year and a ten-year hold pays it once, 0.13% a year.

**FOR THE FIRST TIME IN THE PROJECT THE INDEX COMPARISON IS WON.** 6,332
matched ten-year windows, index on a total-return basis at the measured 1.77%
top-decile yield: liquid decile median **+188.8%** against the index's
**+108.5%**, paired median **+51.8%**, 57.7% of windows, **+32.9% early and
+73.9% late — positive in both halves**, and the clustered permutation null
(whole (ticker, year) blocks, per A17) gives z = **+2.70** with 0 of 200 draws
exceeding.

**IT STILL DOES NOT CLEAR THIS REPO'S OWN BAR, and that is the honest place to
leave it.** z = +2.70 is p ≈ 0.0035 against a Bonferroni bar of 0.0007 after 70
trials, and 200 draws cannot resolve below 0.005. Effective n is **56** for the
whole sample and ~6 for the decile — a ten-year window over a twenty-four-year
panel is about two independent observations per name, and no null manufactures
independence the panel does not contain. Only **30 distinct names** were ever in
the top decile, so the cross-section is a list rather than a population. The
base rate does more work than the selection (70.6% early, 40.3% late). And the
holdout is spent, so none of it is out of sample.

One reassurance worth recording: the decile contains **BUMI**, which fell ~99%
from its peak. This is not a survivor list.

**The lesson, and it is a new one for this repo.** Every negative result here —
flow, broker identity, investor class, price/TA, the multiplier entry, 58 exit
rules, 9 timing rules — was measured at one horizon that nobody chose
deliberately. A parameter fixed once by convenience and then inherited by
twelve studies is not a constant; it is an untested assumption with a
project-wide blast radius. **Vary the thing every experiment holds fixed.**

## A21. The 8-of-10 cell exists, rests on one observation, and is four banks

A20 put the ceiling at ~7 of 10 and left two costs unpaid. Both are now paid
and a tighter cell was found. Memo `reports/horizon.md` §5–§6, logged H24.

**"200 DRAWS CANNOT RESOLVE BELOW 0.005" WAS A LIMITATION I WROTE AND LEFT.**
The bar is 0.00071 and more draws is the entire price of an answer. At 5,000
draws with the +1 correction: z = +2.87, **p = 0.00140. Still does not clear.**
Two minutes. Naming a cost is not paying it, and this repo has now recorded
that failure three times (A19's 20-minute rebuild, A19's liquid re-rank, this).

**THE TAKE-PROFIT CONFIRMS P1 AND THE PRICE IS BRUTAL.** Hold to ten years:
59.6% of positions captured a double, mean **+432.5%**. Sell everything at 2x:
69.1% captured, mean **+58.1%**. The last 9.5 points of hit rate cost 374
points of mean return.

**P4 WAS PRE-REGISTERED AND FAILED.** Scaling out does NOT dominate the
corners — the mean falls monotonically with every unit sold. No free lunch in
the interior.

**AND A COLUMN CONFLATED TWO QUESTIONS.** "Doubles realised" was computed from
the PEAK, crediting a never-sell with captures it never made. Split into
**name doubled** (constant at 69.1% under every selling rule — a property of
the picking) and **I captured it**. **"7 of 10 multi-baggers" is settled AT
ENTRY; the exit decides only how much reaches the account.** That distinction
should have been made the first time P(2x) was written down.

**THE 8-OF-10 CELL IS REAL AND IS ONE OBSERVATION.** Split the decile by how
many of the three prior years the name was ALREADY in it — backward-looking,
so A5-clean — and the gradient is monotone: 56.9% (new), 67.9%, 71.5%,
**82.8% (3 of 3)**. Null z = +3.04, p = 0.00100 against a 0.00069 bar. It is
strikingly stable — **83.6% early, 82.4% late, while the BASE rate collapsed
70.6% → 40.3%.** And it rests on **eight names and an effective n of 1.8**,
was found by LOOKING after the decile result was in hand, yields **four names
today of which three are banks**, and would have admitted AMMN — listed 3.1
years, so its tenure score is its entire life — until `MIN_LISTED_YEARS`
excluded it.

**The honest offer is two priced options, not one answer:** ten names at ~69%
doubling, or four names at ~83% on one effective observation and a single
sector. Ten names at eight-of-ten is not available from this evidence.

**AND THE HORIZON GOT DROPPED FROM A SUMMARY, WHICH IS ITS OWN LESSON.** The
69% was written in a summary table without "over ten years" beside it and was
read — reasonably — as a one-year number. The correction is not a rescaling,
because **the tilt INVERTS**: at one year the liquid decile touches 2x **4.2%**
of the time against **10.2%** for the liquid names it excludes, and the core
cell reads **1.0%**. A ten-name basket delivers **0.4 of 10 in a year, 2.5 in
three, 7.0 in ten**. Below roughly three to five years this basket is the wrong
side of the trade, not a weaker version of the right one — which is A19's
inversion seen a second time. `BY_HORIZON` now prints on every run of
`scripts/decade.py` and four tests pin it, because **a conditional result
quoted without its condition is a wrong result, and the fix belongs in the code
that prints it rather than in the discipline of whoever quotes it.**

## A22. The fast-multiplier screen clears the bar, and clearing it proves nothing

The user rejected the ten-year answer — "I want high multiplier fast, that's
why I play in emerging market" — which inverts the question to: maximise P(a
name doubles within a YEAR). Memo `reports/fastmover.md`, logged H25.

**IT EXISTS AND IT CLEARS.** Most volatile 5% and thinnest-traded 20% of names
above Rp1bn/day: P(touch 2x in a year) **21.22%** against a 12.05% base, lift
1.76x, clustered null z = **+5.66**, **p = 0.00020 against a bar of 0.00064**.
Positive in both halves. **The only result in this project that has ever
cleared the Bonferroni bar.**

**AND IT IS THE STRONGEST DEMONSTRATION HERE THAT CLEARING THE BAR IS NOT
ENOUGH.** Every top feature is the same axis — `lowvol` correlates **−1.00**
with vol60 because it IS vol60 negated, `amihud60` +0.34, `squeeze` +0.31 — so
there is one factor and it is volatility. **H13's predicted-null control
`squeeze` ranks THIRD** at 18.53%. A9 wrote that a firing negative control
means significance is not evidence; here it does not merely fire, it *places*.
A volatile name is mechanically likelier to touch ANY level: **2.1 double and
1.9 halve per ten names a year.** The permutation null is powerless against
this because it asks "is this cell different from a random cell", and the
answer is trivially yes.

**THE MEAN/MEDIAN/MEAN-LOG WEDGE DECIDES IT, A FIFTH TIME.** Arithmetic mean
net of cost **+16.9%**, median **−19.1%**, **mean log −0.1927 → −17.5% a year
compounded.** A ten-name basket rebalanced annually recovers much of the
diversification loss and still compounds at **+5.1% against the index's
+14.6% — 3.1x against 22.8x over 23 years, with a 14.8% chance of ending below
where it started.**

**AND THE RESOLUTION IS POSITION SIZE, NOT SELECTION — which is new here.**
The NUMBER of names that double does not depend on how much money is in them.
Ten screen names double about twice a year whatever the sleeve weighs; the
weight decides only what that does to the account. Each 10% of the account
moved into the sleeve costs about **one point of CAGR**, so at 20–30% you keep
94–96% of the index's compounding and still watch two names double a year.
**Every earlier study in this repo treated the choice as WHICH names to hold;
this is the first where the answer is HOW MUCH.**

**A packaging lesson too.** The tight screen yields only four names on
2026-08-24. Returning four when ten were asked for, or quoting the tight
screen's 21.2% for a loosened one, are both quiet failures — the same shape as
quoting a ten-year doubling rate as a one-year one (A21). `pick_tier` widens,
names the tier it used, and carries that tier's own odds; only the tight tier
has the null behind it and the wider ones inherit significance, which is
weaker and is said so.

## A23. "This is a customer's money" — what the cost model never contained

Asked directly whether the fast screen is technical analysis, narrative or
guessing, and told the capital belongs to a client. The first two answers are
easy and the third is the point. Memo `reports/fastmover.md` §6–§7.

**IT IS A VOLATILITY SORT.** Percentile rank on `vol60` plus a turnover filter.
Not TA — no pattern, indicator or trend rule enters the selection. Not
narrative — the news layer is quarantined and `tests/test_news.py` fails the
build if `spine/` or `features/` imports it. Not guessing — 443 name-years, a
clustered permutation null, p = 0.00020. **And it carries zero directional
information**, which is the honest headline: it raises the chance of touching
ANY level, which is why 2.1 double and 1.9 halve.

**THE COST MODEL HAS NO IMPACT, SUSPENSION OR AUTO-REJECTION TERM, AND ON
THESE NAMES ALL THREE BITE.** `cost_bar` is fees plus a fraksi-harga
half-spread and nothing else. Measured on the same 1,228 screen name-years:
median daily traded value **Rp1.77bn**, so a **Rp500m position is 28% of one
day's volume** and ~3 days to exit at a 10% participation cap; **4.54% of
sessions untradeable** in the year after entry with **28.1% of name-years above
5%**; and **1.40% of sessions down ≥9%**, about four a year per name where
selling may be impossible at any price. All three run against the holder, none
was in any earlier number, and the year you most want out is the year the name
is suspended.

**THE GENERALISATION.** Every cost figure in this repo is a FEE plus a SPREAD.
That is the cost of a small order. On the thin end it is not the cost of a
position, and the gap grows with size — so a result measured at retail scale
does not survive being repeated at client scale, and nothing here was built to
find out where it breaks.

**AND THE REPO'S OWN FRAMING ASSUMES PERSONAL CAPITAL.** §1 calls this a
research programme; A5 fixes costs to "the user's actual Mandiri schedule".
Third-party money is a different activity: the holdout is spent so every number
is in-sample, there is no live track record at all, and a suitability
assessment is a regulated judgement about a specific client that this repo
cannot supply. **Nothing here should be presented to a client as a validated
strategy.**

## A24. Ranking on the RATIO instead of the upside, and it retracts A22

The user asked for "profit far more likely than loss, and still a multi-bagger".
That is a request for a RATIO, and **nothing in this project had ever ranked
cells by one** — every sweep here optimised a rate. Memo `reports/asymmetry.md`,
logged H26. Objective fixed before scoring: `skew = P(touch 2x) / P(end<=0.5)`.

**THE WINNER IS STRENGTH PLUS CALM, WHICH IS WHERE NOBODY WAS LOOKING.** Within
~2% of the 52-week high AND below-median 60-day vol: skew **2.60** against a
null of 1.20 ± 0.15, **z = +9.44, p = 0.00033 against a bar of 0.00061 —
CLEARS**, half-split 2.65 / 2.53. P(a name halves) is **4.1%**. A ten-name
basket over 24 years returns a median **58.3x (+18.5% CAGR)** against the
index's 25.1x (+14.4%), **beating it in 90.2% of draws, with even the
10th-percentile draw at +14.4%.** Momentum plus low-volatility — a prior
mechanism from the global literature, not a mined cell.

**AND IT RETRACTS A22.** H25's volatility screen, scored on asymmetry, reads
skew **1.13 against a null of 1.18: z = −0.19, p = 0.54.** It is
**indistinguishable from a random cell** on the thing that matters, and its
basket returns 3.4x against the index's 22.8x, beating it 5.0% of the time.
A22 said clearing the bar proved nothing; this is what "nothing" looked like.
**Optimising the upside alone finds variance. Optimising the ratio finds
something that survives.** The generalisation: *a rate is not an objective —
check what the denominator is doing before you rank on the numerator.*

**Q2 PRE-REGISTERED AND FAILED, MONOTONICALLY.** "Already-fallen names are
asymmetric" is backwards: nearest the 52-week high skew **2.15**, middle 1.53,
furthest below **0.80** — fallen names halve MORE often than they double. A17's
recovery curve said the same thing from the other direction.

**Q1 PRE-REGISTERED AND SUPPORTED.** Base skew rises monotonically with
horizon — 1.33, 1.48, 1.57, 1.80, 2.26 at 1/2/3/5/10 years. Time converts
diffusion into drift, and the screen buys about six years of that at a one-year
horizon.

**A FIRST DRAFT CALLED THE FRONTIER "PERFECTLY MONOTONE" AND A TEST CAUGHT IT.**
The unscreened baseline is a reference point, not a point on the curve, and
H25's screen carries no strength filter — neither belongs in a monotonicity
claim. What those rows obscured is the more useful comparison: H25 and
`strength + some vol` double at the **same** rate (21.2%, 21.4%), and adding the
strength filter moves the ratio 1.13 → 1.59, halvings 18.7% → 13.5%, and
compounding −16.1% → −8.1%. **At an equal doubling rate, strength is free.**

**What is still NOT licensed: "multi-bagger fast".** The winner doubles **1.1
names in ten a year**, not 2.1. The frontier is a straight trade and buying the
higher doubling rate costs the asymmetry, the compounding, and on H25's version
everything.

## A25. The model §8 asked for, never built until now — and it confirms the cell

Pushed with "there must be patterns, it cannot be random". The fair part: every
sweep in this repo ranked DISCRETE CELLS with hand-tuned percentile cuts, which
is precisely what §8 forbids — "numeric features feeding a cross-sectional
model, not discrete buy/sell rules with hand-tuned parameters". **Twenty-seven
hypotheses in, the model had never been built.** Memo
`reports/xsection_model.md`, logged H27.

**IT IS REAL AND IT IS NOT RANDOM.** Purged expanding walk-forward — each test
year trains only on cohorts whose 252-session window CLOSED before it began —
15 folds, 27,561 test-fold observations. Model top decile: skew **2.31**
against a null of **1.15 ± 0.06**, **z = +20.51**, P(−50%) **4.0%** against a
base of 8.7%.

**AND IT DOES NOT BEAT THE HAND-CUT CELL, WHICH IS THE FINDING.** 2.31 against
H26's 2.60 — but the model's number is OUT OF SAMPLE and the cell's is in
sample. Two entirely different methods, eleven features through gradient
boosting and two percentile cuts chosen by hand, land within 0.3 of each other.
**That convergence is the strongest evidence in this project that the structure
is real rather than mined.** It also bounds the data: with eleven collinear
price features over one macro history, the interaction space is empty.

**THE NULL BEAT THE MODEL TWICE, AND BOTH TIMES IT WAS THE NULL.** First
versions returned 3.06 against the model's 2.31. *Bug one:* `up` and `down`
permuted independently, which breaks their real link — a name that can double
is the same name that can halve, both driven by volatility — inventing rows
that doubled with no halving risk. *Bug two, the actual cause:* permuting
INSIDE (ticker, year) blocks is nearly a NO-OP, because the ~12 monthly cohorts
of one ticker-year carry near-identical labels from eleven-month-overlapping
windows, so the null preserved the mapping it existed to destroy. *The fix:*
reassign whole blocks' LABELS to other blocks' FEATURES. Null fell 3.06 → 1.15.
**A null that beats the thing it tests is broken, not evidence of a weak
model** — the seventh time the null decided a result here, and the first time
it erred toward UNDERSTATING a real effect. A17's lesson was that too fine a
shuffle leaves the null too tight; this is the same lesson with the sign
reversed.

**SECTOR WAS FINALLY USED AND IS MARGINAL.** A14 found the IDX-IC map and no
study had touched it. At 99.6% coverage it moves the skew **+0.15** and costs
0.0098 of mean log. Its `shares` column is deliberately NOT used: the file is
frozen at 2024-07-10, so applying a 2024 share count to a 2010 bar is
look-ahead, and Indonesian rights issues are exactly what makes that wrong —
so market cap stays out rather than being smuggled in as a size factor.

**The honest next instrument is non-price fundamentals** — earnings, book
value, debt — which this repo does not have at panel scale:
`data/cache/fundamentals` holds 59 names of 725.

### Where the programme stands now

Every branch has been run to its end and the answer is the same from six
directions **at a one-year horizon**: flow (H9), broker identity (H10/H11),
investor class (H12), price/TA (H13), the multiplier entry plus every exit rule
tested (H16–H21), and index timing (H22). **At that horizon the Indonesian
retail cost structure is wide enough to swallow every effect this data can
measure, and no selection rule beats the index.**

**H23 changes the shape of that conclusion without softening any of it.** Vary
the horizon — the one parameter twelve studies inherited without choosing — and
a large-cap tilt held for a decade beats the index by a paired median of +51.8%,
positive in both halves. The cost structure is still the binding constraint;
it simply stops binding when you stop paying it annually. What does NOT change:
8 of 10 doublers is unreachable at any horizon (ceiling ~7 of 10), the result
does not clear this repo's Bonferroni bar, effective n is ~56, and the holdout
is spent. Buying the index remains defensible; buying the ten most liquid names
and not touching them for ten years is now the one measured alternative that
has ever beaten it here.

## A26. The time method does not exist, the cone does, and H30's constants were wrong

Asked for an Astronacci-style read — technical gives the horizontal price
target, a time method gives the date, and the two cross — plus a Hull ribbon
with buy/sell labels like the TradingView charts everyone posts. Memo
`reports/time_price.md`, logged H31/H32/H33, deliverable `pine/IDX_Suite.pine`.

**THE CLAIM SPLITS IN TWO AND ONLY ONE HALF SURVIVES.** T1, that turning points
recur on a schedule, is the strong claim and the one that makes the method
distinctive. T2, that from a defined state the joint (how far, how long)
distribution is quotable as a range with odds, needs no cycle at all. **T1 fails
four ways; T2 holds and is wide.**

**T1 FAILS, AND IT FAILS BY BEING MORE RANDOM THAN RANDOM.** ZigZag pivot
spacing has a coefficient of variation of **2.246** against **1.340** for a
block-bootstrap of the same returns, z = **+32.7**. A cycle makes spacing MORE
regular. The interval memory that does exist — R² 0.145 against a null of
0.062 — is volatility clustering: lengthen the bootstrap block to a year and
the null reproduces **87%** of it (excess z = **+1.64**, not significant). The
IHSG has no dominant period either: strongest 885 sessions, peak/mean power
140.2 against a null of **151.6 ± 56.6**, **p = 0.499**, weaker than the median
scrambled control. Month-of-year pivot share runs 0.865 to 1.097.

**The number that ends it:** the 50% band for the date of the next turn is
**±140%** of the median gap knowing nothing and **±125%** knowing that name's
entire history of gaps. Full cycle knowledge is worth about a tenth of the
band's width.

**A DETECTOR THAT CANNOT FIND A CYCLE PROVES NOTHING BY NOT FINDING ONE.** The
first ZigZag tracked only the previous bar before its first leg was confirmed,
so it could not start until a SINGLE bar moved 10%, and it read **zero turns in
a pure sine wave**. The whole negative result rested on it. The sine is now a
test. **Every negative result about a detector needs a positive control on a
case with a known answer** — this repo has recorded the null deciding a result
seven times, and this is the same discipline one level down: check the
instrument before believing what it fails to see.

**T2 HOLDS, AND THE CENTRAL FINDING IS THAT STATE MOVES THE ODDS AND NOT THE
CLOCK.** Median sessions to +20% by state: base **54**, EMA-stacked **54**, not
stacked **54**, Hull rising **53**. A confirmed uptrend changes the chance of
arriving and moves the timing by one session. **Volatility is the only clock**
— 89 sessions in the calmest decile against 30 in the wildest — and it places
the band rather than narrowing it. So the honest deliverable is a **cone**: a
price band crossed with a quartile date band, computed from the name's own
volatility, with a measured hit rate. The interquartile range for +20% is
24 → 110 sessions, a factor of **4.6**. A quarter, never a week.

**A TREND FILTER IS MOSTLY A RISK FILTER, which is not how anyone sells one.**
The stack multiplies upside odds by **1.19** and downside odds by **0.71**.
Twice the effect on the side nobody puts in the headline.

**THE FIT NEARLY SHIPPED WRONG AND POOLED STATISTICS HID IT.** The cone is two
closed-form laws so the chart can answer any target rather than nine. The first
version was linear in log(distance): R² 0.95, median error 4.2 probability
points — and it **under-predicted P(+20%) by ELEVEN POINTS, at the exact target
the chart ships as its default**. Adding a quadratic and a distance×volatility
interaction takes the median error to 1.71pp. **A pooled fit statistic does not
tell you the fit is good at the cell the reader will actually ask for.**

**AND H30's HEADLINE CONSTANTS ARE WITHDRAWN.** `ema_cross_idx.py` computed the
60-session forward return as `px.shift(-60)/px` on a **date × ticker pivot**, so
the step ran over the panel's union index instead of the name's own bars.
Recomputed within ticker, the base is **−0.0140** and the stacked state
**+0.0127**, not the published +0.0021 and +0.0218 — confirmed by reproducing
both numbers with each method. **This is A11's own rule — "group by ticker,
never roll on a pivot" — recommitted by its own author eight appendices later,
in a study whose constants had already been shipped into a Pine script and given
to the user.** The finding survives and improves: the stack beats the base by
+207 bps of mean log early and **+276 bps late**, positive in both halves
relative to the base. What dies is the claim that it compounds on its own.

**THE FLIP LABELS WERE ASKED FOR, ARE DRAWN, AND ARE MEASURED AS WORTHLESS.**
60-session hold net of 56 bps, mean log: base (no toll) −0.0140, EMA stack
−0.0124, Hull slope −0.0148, **HMA-21 over HMA-55 −0.0191 — the classic dual
Hull cross is the worst of the four.** That matches `reports/hullut_*.csv`,
where the published Hull Suite + UT Bot lost to buy-and-hold in 240 of 240 grid
cells and all five walk-forward folds. Drawing a thing a user asked for while
printing the measurement that says not to trade it is the right resolution;
refusing to draw it, or drawing it silently, are both worse.

**One drift guard worth keeping.** Pine cannot import anything, so the 27 fitted
coefficients exist twice. `tests/test_cone.py` parses the .pine file and asserts
every one appears verbatim, that the clamp matches, that the withdrawn H30
constants survive only in the retraction paragraph and never in code, and that
no statement is comma-chained (Pine has no comma operator and this file has made
that mistake before). **The only thing worse than an unvalidated constant is two
copies of it that stop matching.**

## A27. Support and resistance are real, Fibonacci is not, and no bracket survives

Asked to add take-profit and stop levels from support/resistance and Fibonacci,
set from history to maximise profit and minimise loss. Three questions, three
different answers. Memo `reports/levels.md`, logged H34/H35.

**THE PLACEBO IS THE ENTIRE DESIGN, AND IT IS WHAT MAKES BOTH ANSWERS
CREDIBLE.** "Price bounced off the 61.8%" cannot be tested against nothing: a
shallow retracement is reached by every pullback and a deep one only by big
ones, so comparing 0.618 to 0.90 compares DEPTHS and will always find the
shallow level stronger. The grid is therefore **continuous — 0.15 to 0.95 in
steps of 0.025** — and the question is whether 0.618 stands out from **0.60 and
0.65**, neighbours matched on depth that are not Fibonacci numbers. The same
logic runs the support test: the placebo is the same construction against the
same swing high **displaced 7%**.

**FIBONACCI IS A SMOOTH FUNCTION READ AT FIVE ARBITRARY POINTS.** 280,228
touches. P(the leg high is regained within 60 sessions) runs monotonically from
0.3964 at r=0.15 to 0.2911 at r=0.95 **with no bump anywhere** — 0.500 reads
**0.3611** between 0.475's 0.3631 and 0.525's 0.3571. Against 2,000 draws of
five random non-Fibonacci nodes: **z = +0.77, +0.68, +0.41, −0.95, every p above
0.35.** The largest effect is five hundredths of a percentage point.

**A PRIOR SWING HIGH IS REAL AND IT REPLICATES — the first level result here
that does.** Distance-adjusted, ticker-clustered: false-break rate **−6.95 pp
[−8.46, −5.77]**, follow-through **+4.46 pp**, forward 20-session return
**+1.00%**, and all three hold in both halves. I registered "a few percentage
points" and it is seven. **What it is NOT:** the false-break rate at the true
level is still **66.4%** — two breakouts in three close back under within a
month. A real level beats a fake one; breakouts still mostly fail.

**NO BRACKET SURVIVES, AND THREE CONTROLS KILLED IT ONE AT A TIME.** 30 (tp, sl)
pairs, 17.6m entry-cells. *Fill at the actual close, not the nominal level* — a
bar breaching −5% often closes at −8%, and on IDX can gap to ARB untradeable;
this alone took the best cell from +0.0173 to **+0.0050** and it flatters
exactly the tight stops that win. *Match the duration* — the best cell is
invested **52 sessions of 252**, and on a market whose per-name yearly log
return is −0.0587, being out of it is most of what a stop does. *Annualise* —
a 52-session bracket is redeployed five times a year. Result: **not one of the
thirty cells is positive in both halves**, best **+2.4%/yr** against an index at
~+12.7% with a late half of **−3.2%**.

**THE DIRECTION OF THE FOLKLORE IS RIGHT AND ITS SIZE IS INSUFFICIENT.** Against
a hold of its own duration the edge is monotone: tp0.50/sl0.10 **+0.0262** down
to tp0.05/sl0.30 **−0.0077**. Cut losses short and let winners run is measurably
correct and worth 3.4 points of log per trade across the whole grid, which the
annualised column hands straight back to costs and the regime break. That is now
the same answer from eight directions.

**AND A SYMMETRIC BRACKET IS NOT NEUTRAL, IN A WAY THAT FLIPS.** At 5%/5% the
stop arrives first more often (0.508 against 0.489); at 20%/20% the target does
(0.472 against 0.444). Tight brackets are dominated by the negative
short-horizon drift and the spread, wide ones start catching the fat right tail.
The fitted race law makes the mechanism explicit: **the volatility coefficient
is −0.0225 against distance coefficients near 0.8**, so which barrier arrives
first is a ratio of distances and says nothing about how fast the name moves —
volatility speeds both up equally.

**Two instrument checks that had to come before the results.** `fib_test` is
verified against a PLANTED bump (it returns z > 4 when one exists), because a
null test that cannot detect an effect proves nothing by finding none — the same
discipline A26's sine-wave control introduced, one section later. And the Pine
ZigZag's first version updated both running extremes on every bar, so the high
ratcheted through down legs and the detector **could only ever find higher
highs** — a swing finder that quietly turns every downtrend into "no resistance
above".

**The resolution shipped to the chart.** Levels are drawn from structure and
odds come from measurement: the target is the nearest confirmed swing high, the
stop the nearest confirmed swing low, and the panel prints P(touch) for each
plus P(target first) — with the row `bracket verdict: 0 of 30 beat hold in both
halves` beside them. Fibonacci is drawn on request and labelled *"measured
nothing"*. Drawing what was asked for while printing the measurement that says
not to trade it is the right resolution; refusing to draw it, or drawing it
silently, are both worse.

## A28. The accuracy question, answered three ways, and the give-back nobody quotes

Asked to make the chart dynamic, run it on every stock, and say how accurate it
is — then, mid-run, to be specific: accurate at *turning colour at the peak* and
at *TP/SL placement*. Three different questions with three different answers.
Memo `reports/accuracy.md`, logged H36/H37/H38.

**A WIN RATE WOULD HAVE BEEN A LIE AND IT IS WORTH SAYING WHY.** The panel emits
probabilities, not calls. A hit rate is undefined until someone picks a
threshold, and whoever picks the threshold decides the answer. Calibration,
skill against the base rate, AUC and band coverage are defined; "accuracy" is
not. Any future request for a single accuracy percentage should be met with this
paragraph rather than a number.

**EVERY NUMBER THE PANEL PRINTED WAS IN-SAMPLE UNTIL THIS.** The holdout is
spent and cannot be un-spent, but a **purged walk-forward** is genuinely out of
sample: for test year Y the laws are refitted only on bars whose 252-session
window CLOSED before Y began. Without the purge a December Y−1 bar is still
resolving inside Y and "training" contains the test year's outcomes.

**Calibration is good, skill is thin, and it generalises off the fitted
universe.** Shipped constants: Brier 0.1861 against a base-rate 0.1935, skill
**+0.0374**, AUC 0.591. Purged walk-forward: skill **+0.0130**, AUC 0.580. And
on the **whole board — 777 names, four times the bars, much thinner than the
Rp1bn/day names the laws were fitted on** — skill **+0.0168** and AUC **0.594**,
both slightly *better* than the liquid walk-forward.

**THE DATE BAND IS THE BEST-BEHAVED NUMBER IN THIS PROJECT.** It claims to
contain half the arrivals and contains **0.497 / 0.500 / 0.475** across the
three arms.

**WHERE THERE IS NO SKILL AT ALL, which the pooled number hides.** By target,
AUC runs 0.648 at −50% and 0.632 at 2x — and **0.502 at +5% and 0.524 at +10%,
with NEGATIVE skill**. Almost everything touches +5% inside a year, so there is
nothing to discriminate. All the skill is in the far targets and more of it on
the downside. **And the race law has NO discrimination anywhere**: calibrated to
within 0.3 points on average, BSS 0.000, AUC 0.51, dipping to 0.464. Which
barrier arrives first is a function of the two distances the user chose and
nothing about the name.

**THE COLOUR DOES NOT CHANGE AT THE PEAK. IT CHANGES A NINTH OF THE WAY DOWN.**
Against 47,002 confirmed swing highs, with every detector scored against a
random detector spending **the same number of flips**: EMA34 recall 0.852,
precision 0.583, **F1 0.692 against a null of 0.554** — the most accurate of the
five. Hull-55 slope 0.683 (null 0.472), EMA50 0.666, dual Hull 0.640, and the
EMA stack **0.445 with recall 0.348 BELOW its own null of 0.367** — it breaks
long after most tops and misses two thirds of them, while carrying the highest
precision in the table. It is a confirmation, not a detector.

**And the give-back is the number nobody quotes.** Median share of the peak
already surrendered when the flip fires: **10.5% to 12.5%**, against a random
detector's **8.5–8.9%**. **The real detectors give back MORE of the peak than
random bars do**, even with a shorter median lag in time — because a trend flip
fires *because* price fell, so it is conditioned on the drop having happened.
Confirmation costs about an eleventh of the top and that cost is invisible in
recall, precision and lag alike.

**BOTH PLACEMENT PREDICTIONS WERE WRONG, IN OPPOSITE DIRECTIONS.** I registered
that selling *into* resistance would beat waiting for the break, from H34b's
66.4% false-break rate. Monotonically false: mean net runs −0.52% at five
percent short of the level to **+0.35% at five percent beyond it**. The
false-break rate is real and the breaks that work pay for all of it. And I
registered that the stop offset would matter less than the target offset; on
mean log it matters **ten times more** (spread 0.0099 against 0.0010), while on
the arithmetic mean it barely matters at all. A18 decides which to read — an
equal-weighted holder is paid the mean — so **the stop offset is nearly free and
the target offset is where the money is.**

**Fifty-five bracket and placement combinations across H35 and H38, zero
positive in both halves.** Placement changes the shape of the outcome and never
its sign.

**One shipped change worth recording.** The levels now plot as a stepped SERIES
across all history rather than only at the right edge, so the chart can be
audited by eye: scroll back and the line sits where it would have sat, because
it is built from confirmed pivots and never repaints. That is what "dynamic"
should mean — not that it redraws, but that every past bar shows what was known
at that bar.

**A LATE ADDITION THAT FOUND A REAL BUG.** Asked to run the indicator on BBCA,
`scripts/paint_suite.py` reimplements the whole Pine file in Python and renders
the chart plus the panel. Its first run printed **"resistance: none above — at
new highs" next to "drawdown from peak −24.7%"** — two rows of the same panel
contradicting each other. The cause: both the replica and the .pine carried only
the LAST confirmed swing high, so after a deep fall followed by a lower high the
most recent high sits *below* price and the level disappears. Resistance is the
**nearest confirmed high above price**, which needs every pivot kept, not the
last one; fixed in both, and BBCA then reads Rp 7,158 (+11.8%) against support
at Rp 6,110 (−4.5%). **A panel that contradicts itself is the cheapest bug
detector available, and it only works if the panel prints enough rows to
contradict.**

## A29. The Hull round trip, and the benchmark column that found the real result

Asked to trade the chart's own signals across every IDX name — buy on Hull
green + BUY, sell on Hull red + SELL — and report accuracy and average return;
then, separately, whether to exit on the Hull alone since IDX is long-only.
Memo `reports/hull_trade.md`, logged H39.

**WHAT WAS ASKED FOR, MEASURED.** 15,327 round trips over 703 names, mean hold
32 sessions, net of 56 bps: **win rate 32.5%** against a matched hold's 42.7%,
**average return +5.54%** against +2.08%, median −2.58%. The textbook
trend-following shape, and **the best 1% of trades contribute 71% of the total
return.**

**THE PER-TRADE EDGE IS REAL AND THE MONEY IS NOT.** Mean log +0.0127 against a
duration-matched hold's −0.0022, positive in both halves. But the rule is in
the market **33.5%** of the time, and compounded per name over the span it was
active it returns a median **+1.13% CAGR against +9.88% for simply owning the
name** — beating it on 21.5% of names. **Zero of forty grid configurations beat
buy-and-hold on CAGR.**

**A STATISTIC THAT READS EXACTLY 0.0% IS A BUG, NOT A FINDING.** The first
version compared each trade to a "buy-and-hold" over that trade's OWN entry and
exit bars — the same trade minus the toll, so the rule could never win by
construction, and it dutifully printed "beat buy-and-hold on 0.0% of trades".
The number was so clean it was obviously definitional. The comparison the
question actually asks is the whole campaign: compound every trade a name
produced against owning it across the same span.

**THE EXIT QUESTION HAS A CLEAN AND STABLE ANSWER, unusually for this repo.**
Exiting the moment the Hull turns red beats waiting for both conditions on CAGR
(+1.45% against +1.00%), while waiting for both earns **more per trade** (+6.85%
against +4.90%) by holding winners 38 sessions instead of 29 and sitting through
the first leg down on every loser. **Faster out wins the compounding; slower out
wins the average trade.** The ordering holds across every Hull length and every
signal, which is why it is quoted at all — it is the one comparison here that is
not the maximum of a sweep.

**AND THE BENCHMARK COLUMN, ADDED AS A CONTROL, CONTAINED THE ACTUAL FINDING.**
For EMA-stack cells the hold benchmark runs **+9.9% to +11.2%/yr**; for the
EMA34, EMA50 and hull-only cells it runs **+2.0% to +3.0%**, against a panel
median of **+2.56%**. The stack entry filter therefore selects spans in which
merely owning the name returned about four times the typical rate — **and the
trading rule converts that 9.9% into 1.13%.** The signal carries real
information about WHAT TO OWN, and flipping in and out of it destroys roughly
nine tenths of that value: part the toll (32 sessions a trade at 56 bps is
~4.4%/yr), the larger part simply being out of a rising asset two thirds of the
time. **A control added to make a result readable turned out to be the result.**

**One packaging trap avoided.** 33 of the 40 cells are positive in both halves
on the *duration-matched* edge, which reads as an overwhelming validation and
is not one: that edge measures "better than a random-start hold of the same
length", not "better than holding". Two benchmarks that sound alike answering
opposite questions is the same shape as A19's missing index comparison, caught
this time before it was published rather than after.

## A30. The Hull's flip price is computable, is a WIDE stop, and still loses

Asked for a dynamic TP/SL "based on when the Hull will turn colour", then
whether that beats a classic stop from support or an EMA. Memo
`reports/hull_stop.md`, logged H40.

**THE REQUEST SPLITS AND ONLY ONE HALF IS POSSIBLE.** Forecasting the flip DATE
is A26's failed claim. Computing the PRICE at which it flips tomorrow is exact,
because **the Hull average is LINEAR in the next close** — one `x*` solves
HMA_{t+1} = HMA_t. Derived in closed form and **verified against a brute-force
recomputation to 1e-14**. It says where, never when, and it is now plotted.

**IT IS A VERY WIDE STOP, WHICH IS THE POINT OF PRINTING IT.** Every green
hull55 bar over 891 names: the level sits **13.5% below the close at the
median**, 29.8% at the lower quartile, **75.8% in the worst twentieth**. Anyone
treating "stop where the Hull turns" as a tight stop is wrong by an order of
magnitude. Same fact H37 measured from the other side as an 11% give-back.

**THE HEAD-TO-HEAD ANSWERS THE QUESTION AND THEN EMBARRASSES EVERY INDICATOR.**
Same entry, 252-session cap on every rule, net of 56 bps, on CAGR: trail −25%
from peak **+1.73%**, fixed −20% +1.47%, hull55 flip +1 bar +1.01%, trail −15%
+0.57%, **hull55 flip price −0.15%**, **confirmed swing support −1.27%**,
**EMA50 −1.81%**, **EMA34 −1.89%**, hull21 flip −4.59%. So the Hull flip beats
support and both EMAs — and **the best stop in the table is the dumbest one, a
plain wide percentage trail.** The ordering is close to monotone in time
invested, which is the same wall from a ninth direction. 0 of 13 beat
buy-and-hold.

**THE EMA STOPS FAIL FOR A REASON WORTH NAMING: a 22.5% win rate**, the lowest
in the table. Price crosses an EMA on noise, so a stop there is hit by nothing.

**S2 FAILED AND THE FAILURE IS THE FINDING.** I predicted exiting AT the flip
price would beat waiting one bar to confirm, since the bar after a flip should
be down. The opposite: **+1.01% against −0.15%**. **The bar after the Hull turns
red is on average an UP bar** — H13's `rev1` reversal effect. Confirming a bar
later is not just safer, it is better, and that inverts the usual "cut the lag"
instinct.

**S4 FAILED TOO.** A tighter trail was predicted to raise the win rate and lower
the mean; it lowers **both**. The only thing that raises the win rate is a
take-profit — the swing-high TP reaches **46.6%** while collapsing the mean to
+0.48% and the CAGR to −1.98%. *A high win rate is bought, and the price is the
right tail.*

**TWO BUGS, BOTH CAUGHT BY IMPOSSIBLE NUMBERS.** A win rate of **exactly 0.0%**:
the fixed stops had no other exit, so a level sitting below the entry closes
losers and never winners and every completed trade was a loss by construction —
fixed with a time cap on every rule, which is also what makes a fixed stop
comparable to a trailing one. And a CAGR of **nan**: `ret = ratio − 1 − cost`
can fall below −1 on a collapsing name, so the growth factor goes negative and a
fractional power is NaN. **A statistic that cannot occur is the cheapest bug
detector available**, and this is the second catch of that kind in two studies —
H39's "beat buy-and-hold on 0.0% of trades" was the first.

## A31. The account simulator, and the control that took the answer back

Asked for the TradingView script, a plain-English guide, a daily end-of-day
signal, and *"if i start trading with 50 jt what is the expected return per
month"*. The first three are packaging. The fourth is a research question and
it needed a control this project had never run for a trading rule: **the same
machine picking names at random.** Memo `reports/monthly.md`, logged H41.

**THE FIRST TWO RUNS SAID THE ACCOUNT BEATS THE INDEX, AND THAT SHOULD HAVE
BEEN THE WARNING.** H39 had measured the same rule compounding at +1.13% a year
against +9.88% for holding the same names. A portfolio built from a rule that
loses to holding, returning +15.0% against the index's +11.05%, is two studies
of one rule disagreeing — which is the signal that one of them is wrong, not
that the portfolio version found something.

**Both explanations turned out to be true at once.** One bug: the entry filled
on its own signal bar, a close only known once the bar was finished. And one
structural fact: **a five-name book rotating constantly with a trailing stop,
invested 87% of the time, IS an equal-weighted IDX portfolio with an exit
rule.** A19 had already established that such a basket behaves nothing like the
cap-weighted index. So the index was never the right comparison and beating it
was never evidence.

| | CAGR | mean month |
|---|---|---|
| Hull-filtered | +15.00% | +1.43% |
| **random, identical machine** | **+14.19%** | **+1.44%** |
| IHSG buy and hold | +11.05% | +1.04% |

**+0.81% a year against a draw-to-draw luck spread of 8.35 points, and a mean
month LOWER than random's.** Then the half-split: edge **−1.34%** early,
**+2.55%** late, both inside their own error bars, and **1 of 10 draws beating
the random mean in both halves against 2.5 expected by chance.** The filter
does worse than a coin flip on the one replication test this repo trusts.

**THE POWER STATEMENT IS THE MOST USEFUL NUMBER IN THE STUDY.** Months needed
at t = 2 to distinguish this account's mean from **zero**: 104, i.e. **8.7
years**. From **random picking**: **46,856, i.e. 3,905 years**. That is not a
joke about precision — it is what a +0.81%/yr edge against a 7.30% monthly sd
arithmetically means. **No live track record of any realistic length can tell
this rule from random selection**, so anyone trading it for a year and judging
by the outcome is reading noise in whichever direction it points. Stated
separately from the effect, because A19 recorded conflating the two as its own
error.

**AND M2 FAILED IN THE DIRECTION THAT FLATTERED THE RULE, WHICH TAUGHT
SOMETHING NEW.** The registered prediction was that the account would
underperform the index. It did not. That failure is worth exactly nothing,
because the control moved with it. **A prediction can fail in the flattering
direction and still be uninformative if the benchmark it failed against is not
the alternative the investor would actually take.** This is A19's missing-
comparison lesson arriving from the opposite side: there the omission
manufactured a result, here it would have manufactured a retraction of one.

**The delivered answer is therefore a distribution and a disclaimer, not a
rate:** typically +0.73% a month (Rp +366k on 50 juta), on average +1.43%
(Rp +714k), ±7.30% of ordinary swing (±Rp 3.65m), with a −30.2% month
(−Rp 15.1m) in the sample — **none of it attributable to the signal, and all of
it in-sample.**

**Two pieces of packaging shipped alongside.** `scripts/daily_signal.py` ranks
every green-ribbon name on **expectancy** — `P(target first)×target −
P(stop first)×stop − cost` — never on upside (H25→H26) and never on the odds
ratio, which promoted setups with a 3% target and a 24% stop. On a typical
evening **4 or 5 of 25 rows are positive**, and a scanner where everything looks
good is a scanner with the cost term missing. And a Routine fires the scan at
11:00 UTC on weekdays (19:00 Shanghai, two hours after the 15:50 WIB close),
carrying the H39/H40/H41 caveats in its own prompt so the list cannot arrive
without them.

## A32. The scanner's own list, replayed through 26 years, and the hit rate is the trap

A31 shipped `scripts/daily_signal.py` without ever checking it. Every number it
prints — target, stop, `P(target first)`, the `EV` it ranks on — is a FITTED
LAW; nothing had asked whether the rows it SELECTS behave as advertised. H42
replays the identical scan at every historical bar: 116,754 signals, 706 names,
2000-2026. Memo `reports/signal_backtest.md`.

**THE ANSWER INVERTS THE QUESTION. The target IS reliably hit, and that is
exactly why the list makes no money.** 68.5% touch it within a year, **59.1%
reach it before the stop**, and the mean signal returns **+0.08%** against
**+12.97%** for holding the same name for the year. Split by reward-to-risk it
is perfectly monotone: R:R below 0.75 hits **74.2%** of the time and returns
**−0.20%**; R:R above 4 hits **24.7%** and returns **+2.01%**. And **55% of all
signals sit in that first bucket** — near target, far stop, high hit rate, the
only cell that loses money. *A high win rate is bought and the price is the
right tail* (H40's S4), reached here from a completely different direction.

**THE CONTROL AGAIN, AND IT IS THE SAME ANSWER AS H41.** Same date, same bracket
distances, a random eligible name: P(target first) **0.591 against 0.591** — to
three decimals — and mean return **+0.50% against +0.08%**, so the random arm
makes six times as much. Paired per signal with (ticker, year) block resampling:
**−0.43% [−0.88%, +0.05%]**, negative in both halves of every arm. The interval
clips zero so "worse than random" is not established; "no better than random" is.

**AND THE NUMBER THAT ENDS IT: the bracket costs −13.06% a year
[−16.25%, −10.22%]** against owning the name, negative in ALL TEN EV deciles.

**B3 FAILED AND IT IS A SMALL GENUINE WIN.** I registered the EV column as a
predicted-null. It separates — **+1.93% against −0.15%, in both halves** — but
only in the top decile, and the `hold` column of the same table runs +21.55% in
the worst EV bin down to +11.07% in the best: **the scanner systematically
prefers names that, held, go up less.** Registering a predicted-null and having
it fire is A9's lesson working as designed, in the direction that finds a real
effect rather than a phantom one.

**B1 CONFIRMED, and it is the cleanest statement of what the panel is for.**
Pooled predicted 0.566 against realised 0.591, and under-confident rather than
over-. Realised runs 0.275 → 0.893 across predicted deciles, which LOOKS like
strong discrimination and is not — the deciles ARE the ratio of the two
distances the reader chose. That is H36's AUC 0.51 restated: **it prices a
decision already made and cannot make one.**

**THE FOURTH IMPOSSIBLE NUMBER IN FOUR STUDIES, AND A FIFTH AVOIDED.** The first
run printed an annualised return of **1.7e28** — annualising each trade's
arithmetic return separately means a +50% trade held one bar contributes 1.5^252
and one row swamps a hundred thousand. Annualise the MEAN LOG. And the trap that
was avoided rather than committed: **a duration-matched hold is not a benchmark
here**, because the bracket exits at the CLOSE of its exit bar and so does a hold
of that many bars — the edge is identically zero by construction, H39's "0.0% of
trades" waiting to happen. A test pins both.

**AND THE REPLAY IS TESTED RATHER THAN ASSERTED**, which is what makes any of it
readable. `tests/test_signal_backtest.py` truncates the panel to a past date,
runs the LIVE scanner on what was knowable then, and demands the replay return
the same row — at three distances into the past. One non-causal helper anywhere
in the chain (a ZigZag drawn at the pivot instead of the confirmation bar, a
centred window) would turn the whole backtest into a look-ahead, and nothing in
the output would look wrong.

**AND THE FIX, WHICH IS A REDUCTION IN HARM AND NOT AN EDGE.** Scored against
the control in both halves, the gate table is unambiguous about which half of
the list to delete: reward-to-risk **below 1.5 is 76% of the rows, is negative
in absolute terms, and loses to the same bracket on a random name in BOTH
halves** (−0.72% and −0.61%). `MIN_RR = 1.5` now rejects them — a sign change at
a bin edge fixed before the study ran, not the maximum of a sweep — and it
removed **24 of 33 rows** on the live board, matching the replay's 76%. Every
surviving row carries `cell/rand/hold`: what its reward-to-risk cell actually
returned, what the same bracket returned on a random name, and what holding
returned. **`tests/test_daily_signal.py` asserts that every cell of the shipped
table has the bracket losing to holding**, so an edit that broke that invariant
fails the build instead of quietly printing an edge.

**What the fix does NOT do, and the memo says so twice.** It does not make the
bracket profitable — no cell of H42 has a target and a stop beating a hold, and
the survivors still lose by 7 to 10 points a year. And it does not establish the
selection: at the surviving geometry the scanner beats a random name by +1.09%/yr
with a 95% interval of **[−0.44%, +2.77%]**, which contains zero.

**TWO BUGS THE CHANGE ITSELF CAUSED, BOTH CAUGHT BY TESTS WRITTEN EARLIER.**
Gating the live scanner without gating the replay made them different screens,
and the equivalence test failed immediately — so `signal_rows` took the gate as
a PARAMETER defaulting to zero, because **a backtest that inherits the filter it
is evaluating can only ever confirm it.** And an emptied scan returned
`pd.DataFrame([])`, which has no columns at all, turning "nothing qualified
today" into a `KeyError` in every caller. Both are now pinned, the second by a
test that also checks the declared column tuple against what a populated scan
produces so it cannot drift.

## A33. The rebalance frequency nobody chose, and the conflict it exposed

A32 closed the scanner. The next question was the one parameter the SURVIVING
screen had never had varied: H26's basket was rebalanced annually because H16
picked 252 sessions for an unrelated reason and twelve studies inherited it —
the exact shape A20 says has a project-wide blast radius. Memo
`reports/rebalance.md`, logged H43.

**P1 CONFIRMED: the curve is humped**, peaking between a quarter and six months
(+11.97%, +11.99%) against monthly +10.91% and three-yearly +6.67%.

**P2 FAILED AND IT IS THE FINDING.** I registered that the selection edge would
be INVARIANT to how often you act on it, since signal quality is a property of
the cross-section. It is not: the GROSS edge over a random basket decays
monotonically **+12.78% at a one-month hold to +2.01% at two years**. **The
screen is a SHORT-HORIZON signal, not a buy-and-hold-forever one** — which
corrects advice I had given one message earlier, extrapolating H23's ten-year
result on the LIQUID DECILE to a screen that behaves nothing like it.

**GROSS HAD TO BE SEPARATED FROM NET BECAUSE THE CONTROL CHURNS MORE THAN THE
TREATMENT.** A random basket redrawn monthly rotates ~95% of its book; the
screen rotates **58%**. A net-of-cost difference between them confounds
SELECTION with TURNOVER, and at monthly frequency the cost gap alone is several
points a year. The stickiness is itself new: cost is charged on TURNOVER here,
and every earlier study in this repo priced a rebalance as the whole book.

**P3, THE PREDICTED NULL, NEARLY FIRED.** Across-frequency spread 3.38% against
a within-frequency PHASE spread of 2.93% — ratio **1.15**, very nearly a tie. A
first version printed "frequency dominates, the result is readable", which is
far too generous for 1.15. With 8 phases the se of a frequency mean is 1.04%, so
**differences under ~2.08% are not readable: monthly, quarterly and six-month
are mutually indistinguishable.** The SHAPE is established, the optimum is not.
**A binary verdict on a ratio near one is a way of not reporting a null.**

**AND THE CONFLICT MATTERS MORE THAN THE FREQUENCY.** H26's memo claims the
screen returns **+18.5%/yr against the index's +14.4%**; this reproducible,
costed, window-matched portfolio walk says **+10.98% against +13.20% — the sign
flips on the central claim of the one surviving result.** H26's basket number is
a **hardcoded `EVIDENCE` dict whose simulation is not in the repo**, violating
§15 independently of which number is right. Its language — "median 58.3x",
"beats the index in 90.2% of draws", percentiles — is that of RESAMPLING rather
than a historical path, and **draws that sample names independently from a
pooled distribution would understate how correlated one year of this screen is**
(every name in it is a high-momentum low-vol liquid name), inflating the median
compounded outcome. That is a hypothesis, not a demonstration, because the code
is gone; reproducing it costs about an hour.

**AND BEATING THE RANDOM CONTROL IS NECESSARY AND NOWHERE NEAR SUFFICIENT.** The
same portfolio beats random by **+12.78%** and loses to the index by **0.61%**.
Both are true of one thing and only the second is a decision. A19 recorded the
missing index comparison as the error that manufactured a result; this is the
same comparison catching a second one, in the study that was supposed to be
tuning the winner.

## A34. Detecting the sell-off makes the give-back WORSE, and the Hull number I owed

The user's observation was that the Hull entry is good and the exit is late, so
detecting a sell-off should let them sell nearer the top. Both halves of the
observation are correct and already measured — H39's benchmark column found the
entry selects spans returning +9.9% to +11.2%/yr merely held against a panel
median of +2.56%, and H37 measured the give-back at 10.9–12.5%. Memo
`reports/selloff.md`, logged H47 and H48.

**THE FIX AS STATED IS STRUCTURALLY UNAVAILABLE, and the reason is worth
keeping.** Any rule that fires on a decline is DOWNSTREAM of the decline. Firing
before it is prediction, which H31 tested directly: ZigZag pivot spacing has a
CV of 2.246 against a block bootstrap's 1.340 — a cycle would make spacing more
regular and the data makes it less. So the reachable question is not "can I sell
at the top" but "for a given amount of give-back saved, what does it cost."

**S3, THE PREDICTED NULL, FIRED AND IT IS THE WHOLE ANSWER.** Eleven detectors
against a RANDOM exit drawn from each rule's own holding-period distribution —
matched trading rate, no information. **Every single real detector gives back
MORE of the peak than the coin flip of the same speed**: `hull55 +1bar` 6.9%
against 4.1%, `drop −4%` 7.3% against 2.8%, `trail 25%` 25.8% against 11.4%.
Eleven of eleven under both re-entry policies, 22 cells, no exceptions. A
detector fires BECAUSE price fell, so it is conditioned on the drop having
already happened. **Leaving sooner is what cuts the give-back; detecting the
sell-off is not.** This replicates H37 on a completely different construction —
there peak-detection framing, here portfolio framing, same sign.

**S2 FAILED and gives a genuinely new shape.** CAGR is NOT monotone in
give-back: the best cells are in the MIDDLE of the table (+8.37%, +8.22%), not
at either end. H40's "a tighter trail lowers BOTH the win rate and the mean"
does not generalise. **S4 confirmed** — nothing beats holding (+13.06%), now **169**
exit configurations with none beating a hold: H17's 32, H18's 58, H35/H38's 55
bracket and placement combinations, H40's 13, H47's 11.

**THE CONTROL WAS RIGGED AND FIXING IT MOVED NUMBERS I WOULD HAVE QUOTED.** The
first harness required a fresh RISING EDGE of the setup to re-enter. That
quietly handicapped the random arm: a real detector fires on a drop that usually
breaks the setup too, so it gets a fresh edge and rejoins the next leg, while the
random arm sells mid-trend with the setup still live and is locked out until the
whole trend dies and restarts — skipping the rest of every trend it sold into.
Under the symmetric policy the control's CAGR rises 1–3 points in every cell.
**A control that cannot play the same game as the treatment is a handicap, not a
null.** Both policies are now printed, and the give-back conclusion is quoted
because it survives both. This is a different failure from the seven times a
null has decided a result here: the null was not mis-specified statistically, it
was denied the treatment's own affordances.

**AND THE BOARD-WIDE HULL NUMBER, WHICH WAS OWED.** I asserted the Hull suite
loses to buy-and-hold; challenged with "really? are you high?", I found I had
measured **EMA34 breaks and labelled them the ribbon**, and that on BBCA the
actual HMA(55) colour returns −17.9% against a hold of −29.0% — it BEAT holding
on the chart being eyeballed. The pure colour rule is now run on 891 names and
2,850,959 name-bars: rule **−1.32%/yr against hold +6.42%**, edge **−7.74%**,
winning on **31.3%** of names; on liquid names +7.15% against +12.62%, edge
−5.47%, **negative in both halves of every universe**. **The claim survives, and
the route to it was still wrong.** BBCA is one of the 36.9% of liquid names
where the ribbon wins — a real minority, which is exactly why one chart could
not settle it in either direction, and why "I checked the board" is the only
form of that claim worth making. A right answer reached by measuring the wrong
thing is not a right answer yet.

## A35. The win rate that was challenged, and the control H48 did not contain

A34's board-wide table reported the ribbon beating buy-and-hold on 31.3% of
names. The user rejected it on central-limit grounds — *"price can only go up or
down, so it should be 50%"* — and the right response to an intuition like that is
a measurement, not an argument. Memo `reports/selloff.md` §7, logged H49.

**THE OBJECTION WAS HALF RIGHT AND IT LANDED ON A REAL GAP.** Two statistics
were both called "win", which was my sloppiness: per-TRADE (share of round trips
ending positive) genuinely should be ~50% without information, and per-NAME
(share of tickers where the rule beat owning the ticker) has no reason to be —
it pits 44% exposure and ~71 tolls against 100% exposure and one toll.

**W1 CONFIRMED — THE HARNESS REPRODUCES THE COIN FLIP.** Driftless log random
walk, zero cost, random timing: per-trade **50.2%**, per-name **47.7%**, se
2.9%. This is A26's sine-wave discipline applied to a win-rate harness, and it
is what licenses believing the rest: a negative-sounding number about IDX is
worthless until the instrument is shown to return the known answer on a known
case.

**W2 FAILED AND W3 IS THE WHOLE STORY.** I registered DRIFT as the largest term.
Adding IDX's own measured drift with no cost moves the per-name rate 47.7% →
**50.3%** — nothing. Adding nothing but a 1.44% round trip to the DRIFTLESS
market takes it to **28.3%**, against the observed 31.3%. **71 round trips ×
1.44% = 102% of the position's value paid in tolls over the span.** The rule
pays for its own position twice over in fees and spread.

**W4's PREDICTED NULL FAILED, IN THE DIRECTION THAT FAVOURS THE INDICATOR, AND
THAT IS THE FINDING.** H48 compared rule to hold with **nothing in between** —
a missing comparison, which A19 records as the error class that manufactures
results. Reordering the ribbon's own green and red runs (exposure, trade count
and run lengths preserved EXACTLY, only alignment with price destroyed) beats
holding on **21.7%** of names against the real ribbon's **31.3%**: **+9.6
points, se 0.8%, ~12 se**, and **+16.6 points** on liquid names. **The ribbon's
timing carries real information.** H48 is QUALIFIED, NOT OVERTURNED — the rule
still loses to buy-and-hold in both halves of every universe — but the reason is
now measured: not that the indicator is noise, which it demonstrably is not, but
that its information is smaller than the toll. The thin bucket is the one place
real and null coincide (20.3% vs 20.0%), exactly consistent with H44's
withdrawal.

**AND A SUB-50% PER-TRADE WIN RATE IS THE RULE WORKING.** The matched-speed
random exits win **40–51%** and make LESS per trade; the real detectors win
**24–47%** and make MORE (`close < EMA20` 24.3%/+2.05% against 40.2%/+0.72%).
Board-wide the ribbon's mean trade is **+3.54%** against a median of **−4.16%**.
H40 measured the same trade-off from the other side: a take-profit lifts the win
rate to 46.6% and takes CAGR to −1.98%. **Win rate and profitability are close
to orthogonal, and a high win rate is bought with the right tail.**

**THE LESSON, AND IT IS A19's FOR THE SECOND TIME.** A study can be
permutation-nulled, half-split, cost-charged and decile-split and still be
missing the one comparison that explains its own headline. A34 ran a matched
random EXIT against every sell-off detector and then reported a rule-versus-hold
number with no control at all, one section later, in the same memo. **Ask of
every headline: what is the thing in between, and did I price it.**

## A36. The desk build — both halves of the target reachable, never together

Asked to stop holding back and build a trading bot the way quants at big firms
do, with a goal: **>+4% average per trade, 80%+ of the time**. The right
response was not to argue the arithmetic but to build the standard
institutional construction and MAP the frontier. Memo `reports/quantbot.md`,
logged H50.

**WHAT WAS BUILT, AND NONE OF ITS THREE PIECES EXISTED HERE BEFORE.** Every one
of the 49 prior hypotheses ranked names on a score and held a fixed horizon.
H50 adds **triple-barrier labelling** on the true intraday high and low (every
earlier study labelled on a fixed-horizon CLOSE return, which is a different
and easier question than "does my order fill"), **volatility-scaled barriers**
so "1.5σ" means the same thing across the cross-section, and **meta-labelling**
— a gradient-boosted secondary model predicting whether each primary signal
will hit its target, which is the technique that exists specifically to raise
PRECISION, i.e. the win rate the goal asks about.

**THE ANSWER: BOTH HALVES ARE INDIVIDUALLY REACHABLE AND NEITHER TOGETHER.**
Mean per trade **+5.77%** (no take-profit, 2σ stop, 252-session clock).
Positive rate **80.3%** (tight target, far stop, model top quintile). **0 of
130 cells clear both**, and at 80.3% the mean is **−0.40%**.

**Q0 PASSED AFTER FAILING FOUR TIMES AND EVERY FAILURE WAS THE TEST.** The
check is whether the labeller reproduces the martingale on a driftless
synthetic where p = b/(a+b) exactly. It printed "instrument FAILS" because
(1) log increments of mean zero make the PRICE drift up by exp(σ²T/2) = +5.2%
over 252 bars and the labeller was blamed for measuring it; (2) b/(a+b) is a
PRICE-space formula and symmetric price barriers are asymmetric in log space
(+0.276 against −0.382); (3) feeding close=high=low detected every touch a day
late and a whole daily move past the level; (4) a version that shrank the daily
step made the error WORSE, because shrinking the step at a fixed horizon also
shrinks total volatility, so barriers stop being reachable and timeouts appear
— and a timeout breaks the formula outright. **An instrument check is worth
building only if you are willing to believe it when it fails.** The residual is
shown to be simulation granularity by CONVERGENCE (0.0422 → 0.0033 as the
intraday path refines 1 → 320 steps) rather than by loosening a tolerance, and
real bars carry the true intraday extreme so it does not exist on IDX at all.

**AND Q0 IS WHAT MAKES THE ANSWER SAYABLE.** At a target of +2.5% and a stop at
−10% the measured win rate is **79.3%** and the measured expectation is
**−0.0009** before costs. **Any win rate is purchasable — target near, stop far
— and buying it is worth nothing**, because W and L move against p by exactly
enough to cancel. That is the optional stopping theorem, and it is now measured
in this repo rather than asserted.

**META-LABELLING IS REAL, WHICH IS MORE THAN MOST THINGS HERE MANAGE.** The top
model quintile reaches **80.3% positive against a label-shuffled null of 76.9%**
— a **+3.4 point** lift the null does not produce. **A first version claimed
+13 points** by differencing the real arm's TRUE-label rate against the null
arm's SHUFFLED-label rate; those are different quantities, and the valid
comparison is the column computed from the unshuffled realised return in both
arms. **Two columns sharing a heading are not thereby comparable, and which arm
a statistic came from has to be checked before it is differenced.** At the balanced geometry it lifts the mean +2.57% → +3.32% and the
median +2.44% → +8.13%, both clearly above null. **It still does not get
anywhere: the best `ann` in the entire modelled table is −3.5%.**

**THE COLUMN THAT DECIDES EVERYTHING IS `ann`, AND IT IS NEGATIVE IN ALL 120
FRONTIER CELLS.** A **+5.77% average trade** running 249 sessions with a
**median of −2.45%** compounds at **−5.3% a year**. A18 established that an
equal-weighted holder is paid the MEAN — but that is a statement about holding
many names AT ONCE. **A bot running positions sequentially is paid the mean
LOG, and here the two disagree in SIGN.** That distinction should be attached
to every per-trade statistic this repo ever quotes again.

**THE BOOK'S NON-MONOTONICITY GAVE IT AWAY BEFORE ANY NULL DID.** 8 slots read
+12.88%, 4 slots −0.38%, 16 slots +4.37%, 32 slots +11.51%. A genuine selection
edge cannot be strong at 8, absent at 16 and strong again at 32 — the SHAPE was
the tell, and it prompted the luck sweep (30 random draws per slot count:
percentiles 30 / 97 / 40 / 100) and then the half-split. **The model beats
random in both halves at 2 of 4 slot counts and beats the INDEX in both halves
at 0 of 4.** It beats random at every slot count in the LATE half and only at
32 in the early one, which the expanding window explains more simply than a
regime story. And the index runs +13.39% early against +5.29% late, so every
book beats it late and none early: that is the index's regime, not the model's
skill.

**A DEGENERATE ARM PRINTED A BELIEVABLE MODEL ROW.** With no take-profit there
is no target to hit, so the meta-label "hit target first" is identically zero,
the classifier had nothing to learn, and it returned the unranked sample under
a "model top 50%" heading with numbers identical to "all signals" — visible
only because the two rows matched to the decimal. **Two identical rows under
different headings is a free bug detector, and it only works if both are
printed.**

**AND A SILENT SUCCESS THAT PRODUCED NOTHING.** A string edit to the runner
dropped the `if __name__ == "__main__"` block, so the script defined `main()`,
never called it, and exited 0 in under a second with an empty output file.
Exit code 0 is not evidence that anything ran.

## A37. The goal reached, and reaching it showed the goal selected for nothing

Told to make ">=+4% mean per trade AND >=80% positive" the goal and not stop
until it was reached. It is reached. Memo `reports/goal.md`, logged H51.

**H50 SAID IT WAS UNREACHABLE AND H50's GRID COULD NOT HAVE FOUND IT.** The
required edge is NOT constant across barrier geometries. With
E[r] = (p − p0)(a + b) − c and p0 = b/(a + b), a symmetric +15%/−15% bracket
needs **+0.300** of edge while +15%/−95% needs **+0.049** — because the payoff
per unit of edge is (a + b) and a far stop makes that large. **H50's grid
stopped at a 2σ stop and never entered the one region where the arithmetic is
not hostile.** Declaring a goal unreachable from a grid that never covered the
cheapest corner of the space is the A12 shape again, in its most expensive
form yet: not a missing source or a missing comparison, but a **missing region
of the search itself**, in a study that had just cited A20 for exactly this.

**THE ANSWER: buy any eligible name, target +30%, NO stop, ten-year clock —
83.5% positive, +12.52% mean per trade.** 14 of 40 cells clear both halves at
ten years, 0 of 216 at one and three years. **What moves the frontier is the
CLOCK, not the selection.**

**G4 CONFIRMED AND IT IS THE WHOLE FINDING.** Every eligible bar with NO
selection at all reads **83.0% / +12.08%**; random 20% subsamples read
82.8–83.0% / +11.93–12.15%. The trend filter adds half a point, inside
subsample noise. **The goal is a property of the two lines you draw, not of
anything you know about a stock** — the optional stopping theorem that Q0
measured on synthetic data, now visible on the real board.

**THE BENCHMARK THE GOAL ITSELF DID NOT CONTAIN.** The index over each trade's
OWN entry and exit dates returns **+37.59%** total-return against the trade's
+12.52% — **−25.07%**, and the index beats the trade in **61.5%** of cases. A
goal phrased as a win rate and a per-trade mean has no benchmark in it, so it
can be satisfied by something that loses to the alternative by 25 points.

**AND IT COMPOUNDS AT −2.43%, WHICH IS NOT THE AVERAGE LOSS DOING IT.** 83.5%
win at a median of +29%; the 16.5% that lose average −70% and **7.88% of all
trades lose 80% or more**. A synthetic test built with a flat −70% loser still
GREW at +0.39%/yr — only the near-total losses turn the geometric mean negative.

**G3 FAILED DEGENERATELY.** I predicted the goal-clearing cells would not be the
best-annualised cells. The best-annualised cell in the whole search IS the goal
cell, because every cell is negative and "best" ranks losses. Logged as a failed
prediction rather than reframed.

**THE GENERALISATION, AND IT IS THE MOST USEFUL THING IN THIS APPENDIX.** A win
rate and a mean per trade, specified **without a holding period and without a
benchmark**, do not jointly constrain a strategy to be good. Both halves are
purchasable with barrier placement alone. The three things that WOULD constrain
it — annualised growth above the index, positive in both halves, and a
random-selection control beaten — remain unsatisfied by any configuration in 306
trials. **Any future target should be stated in those terms, because a target
that can be met by drawing two lines on a chart of a randomly chosen stock is
not a target.**

**A37 addendum — does it work, and the window mismatch found while answering.**
No. Single position compounded **−2.71%/yr**; best equal-weighted book (4 slots)
**+13.34%**; index over the same 23.2-year span **+14.92%**. **0 of 6 slot
counts beat the index.** Rp 100 juta over ten years: Rp 76 juta held one at a
time, Rp 350 juta as the best book, **Rp 402 juta in the index.** The two CAGRs
are both true of the same rule — diversification recovers most of the volatility
drag — and neither reaches the index. The best book is a **tie inside noise**:
the spread across twelve random draws at 4 slots is **5.42%**, wider than the
1.58% gap, on **30 trades in 23 years**.

**AND `book()` MEASURED ITS SPAN TO THE LAST ENTRY, NOT THE LAST EXIT.** A
position opened on the final entry date keeps compounding through its whole
3.4-year holding period, so years of growth were credited to a window that did
not contain them — and the benchmark was then priced over that same wrong
window. It put the index at **+18.94%** and the 8-slot book at **+18.39%**;
corrected to first-entry → last-exit they are **+14.92%** and **+11.93%**. A19's
error class — comparing quantities measured over different windows — committed
inside the function written to price A19's comparison, and caught only because
an index CAGR of +18.94% for Indonesia looked too high to be true. **Two tests
now pin the span; the sanity check that caught it was a benchmark that did not
match the country it claimed to measure.**

## A38. The screen's alpha IS the size effect, and one-name baskets nearly proved otherwise

Told the point is to beat the index by a mile — which is the right target and
the one A37 had just recommended. The best-motivated remaining route was the
repo's own unjoined-up pair: H43's screen beats a random basket from its own
universe by **+12.78%/yr gross**, the largest selection margin in this project,
and ties the index; A19 measured why an equal-weighted IDX basket structurally
trails a cap-weighted one. Hypothesis: the alpha is being spent paying a size
penalty, so stop paying it. Memo `reports/beatindex.md`, logged H52.

**IT IS BACKWARDS, AND B3's PREDICTED NULL IS WHAT SHOWED IT.** The screen's
gross edge over its own random control was predicted to be INVARIANT to the
universe tier. It collapses instead: **+12.9% (all) → +8.4% (top 150) → +5.2%
(top 100) → +3.4% (top 60) → +0.7% (top 40).** **The alpha is not a victim of
the size penalty; the alpha IS a size effect.** Strength-plus-calm works among
small and mid names and does essentially nothing among the largest forty, so
moving upmarket removes the penalty and the edge together — and the edge goes
faster. B1 and B2 failed monotonically alongside it; **0 of 15 arms beat the
index.**

**AND THE ARITHMETIC IS NOW CLOSED.** Random equal-weighted basket **+2.72%**,
index **+11.15%** — a structural handicap of **~8.4 points** — screen adds
**+12.9% gross / ~10.9% net**, lands **half a point short**. Every study of this
screen has returned a tie because **its edge is almost exactly the size of the
handicap it runs under**, and the one lever that removes the handicap removes
the edge faster.

**A ONE-NAME PORTFOLIO NEARLY BECAME THE HEADLINE.** The first run reported
**+17.63% against the index's +11.15%, positive in both halves, 15 of 15 arms
winning.** `rebalance.py` measures the same screen at the same frequency at
+11.67%, and **two studies of one screen disagreeing is the signal that one is
wrong** — so they were run side by side on identical inputs. The picks were
**identical on 78 of 105 rebalance bars**. The entire six-point gap came from
**nine quarters where the eligible universe held 20–40 names and the
"portfolio" was one to three stocks**, landing at the start of IDX's biggest
bull run and compounding through a 26-year path. **The tell was that the RANDOM
CONTROL MOVED BY THE SAME SIX POINTS — a selection effect cannot lift the
control, so a shift in both arms is an accounting artefact by construction.**
That diagnostic is cheap, general, and should be the first thing checked
whenever a treatment improves: *did the control move too?*

`MIN_UNIV = 40` and `MIN_BASKET = 5` now make a degenerate basket impossible,
and `TIERS` is asserted never to cut below the universe floor. A19 records the
smallest-cell trap twice; this is the third, and the first where it produced a
POSITIVE headline rather than a negative one.

**A38 addendum — the cost figure was two things quoted as one.** Challenged with
"Wdym 1.4%? I pay 0.5%", and the challenge was right. The blended number mixed
**commission + sell tax (0.56%, the published schedule, exactly the 0.5%
quoted)** with **one fraksi-harga tick (0.25–1.00%)**, which is the bid-ask
spread and appears on no contract note. Charging a FULL tick assumes liquidity
is TAKEN on both sides of every trade — an execution assumption wearing a fee's
clothing. `fee` and `spread_mult` are now explicit arguments to
`beatindex.run()`. **The sensitivity flips the sign of H52's headline**: fees
only +0.55% vs index, fees + half a tick −0.01%, fees + full tick −0.57%. It
does not become a mile — the realistic middle row is flat — and the **gross edge
over random is +7.33% in every row**, since cost hits both arms equally, so none
of it touches whether the selection is real. **Execution is worth ~1.1%/yr,
larger than most effects measured in this project and, unlike the alpha,
entirely under the holder's control.** The general lesson: a blended cost figure
hides which part is a fact and which part is an assumption, so every result here
now quotes the fee and the spread multiplier separately.

## A39. Three bugs in the harness, and the gate that was a single draw

A38 closed with the screen half a point short of the index. The user then set
the goal plainly — *"beat buy and hold. if need be forget everything and start
over"* — so H54 fixed three benchmarks a holder could actually take (the index,
a never-touched basket of the whole eligible universe, and the strategy's own
first basket held untouched) plus a random control, and required all of them in
both halves. A fleet of strategy designs was run against it and three passed.
**All three passes were properties of my harness.** Memo `reports/beathold.md`,
logged H54.

**THE FLEET FOUND THE BUGS, NOT ME, AND THE FIRST WAS WORTH MORE THAN ANY EFFECT
THE HARNESS HAD MEASURED.** *Cash drag*: every `continue` in `walk()` appended
the unchanged equity, so a mark the strategy could not act on earned it **zero**
while all three buy-and-hold benchmarks compounded through that window.
Refusing to BUY a degenerate cross-section is right — it is what stops H52's
one-name basket — but LIQUIDATING a book you already own because the screen went
quiet is not what any holder does and not what the benchmark does. *The
rebalance phase*: `marks` starts at `dates[offset]` and offset 0 is the panel's
first bar for no reason but that it is first; all three passes vanished at every
other offset. **A20's lesson — a parameter fixed by convenience and inherited by
every study — committed inside the harness written to catch that class of
error.** *The control's calendar*: the random arm ran at offset 0 whatever the
strategy did.

**AND A FOURTH I FOUND ONLY AFTER FIXING THE SECOND.** Equalising the phases is
not enough, because the window starts at the first mark that yields a tradeable
basket and **that date moves with the calendar**: one strategy's six semiannual
phases began in **2005, 2007, 2007, 2008, 2009 and 2010**, spanning 15.9 to 21.2
years. "4 of 6 phases" was counting six different experiments. A19's error class,
surviving one fix and reappearing one level up. Phases now run over the common
window, found by a scout pass.

**D2 FAILED IN TWO OF THREE CLAUSES, AND ONLY BECAUSE THE BUG WAS REINSTATED ON
PURPOSE TO MEASURE IT.** `Bench(carry=False)` is the drag, deliberately. The
lift is **not one-signed** — three arms got WORSE when it was fixed, because
sitting in cash was accidentally protective in windows where the market fell.
**The drag was an unpriced cash position, not a systematic handicap**, and my own
"1.8 to 6.7 points a year" was a one-sided read of a two-sided error. It is also
largest at QUARTERLY (+2.64%/yr) rather than annual (+1.11%), because what
matters is WHERE the skips fall — the early years, when the universe is below
the floor — not how long they are. The ordering survives (Spearman **+0.871**),
which is why the fleet's relative conclusions mostly stand while its levels do
not. **Measuring a bug's cost by re-enabling it beats asserting the fix helped,
and it is two lines.**

**THE BINDING CONSTRAINT WAS MY OWN BENCHMARK, AND IT IS A SINGLE DRAW.**
`picks-halves` failed **103 of 126 phase-verdicts (82%)** — and `BH_PICKS` is
ONE ~10-name basket held eighteen years, spanning **0.49% to 15.82% a year
across six calendars of the SAME rule**. A 15-point spread from moving the start
date a few weeks. **A gate whose sampling error exceeds every effect it is
measuring is not measuring.** It is now reported and not gated — a rule that
loses to holding its own picks is still telling you the trading adds nothing,
which is the whole ADRO question — while `BH_UNIVERSE` (100–200 names) stays a
gate because it is diversified. The strict verdict still prints beside the
weaker one and a test asserts it can never be true where the weaker one is
false, per §2's ban on replacing a criterion that failed.

**THE RESULT, BOTH WAYS.** Strict bar: **0 of 21 arms pass — D1 confirmed.**
Diversified bar: **4 of 21 pass at 4 of 6 calendars**, all strength+calm, at
**+4.5% to +6.5%/yr over the index** — and that bar was defined after seeing
which gate bound, so they are a lead. **What is not in doubt either way: six
arms beat the index in 6 of 6 calendars, and every strength or momentum arm
beats a random basket from its own universe in 108 of 126 phase-verdicts by
8–14 points a year.** A34–A38's "no selection rule beats the index" was measured
at offset 0 through a harness with a cash drag in it and **does not survive the
correction**.

**D4 FAILED OPPOSITE TO THE CLAIM IT TESTED, and gives the cheapest result in
the project.** Window-matched, the 2x2 says name churn is worth **+1.19 points**
and letting weights DRIFT is worth **+1.64**, interacting (+0.57 when names
churn, +1.64 when frozen). Doing neither to doing both is **−2.32% → +0.51%**.
**The cheapest way to beat the index measured anywhere here is to stop trading**
— own the eligible universe, never re-select, never reset to equal, 0.11%/yr in
cost. The fleet had attributed the destruction to name churn alone; both matter
and drift is the larger.

**WHAT IS STILL NOT ESTABLISHED, and it is the load-bearing paragraph.** 4 of 6
**heavily overlapping** windows is a robustness check, not a significance
statement — the effective number of independent observations over one 18-year
Indonesian history is small and no resampling manufactures more. The holdout was
spent at H16, so every number is in-sample. The buffer sweep is **not monotone**
(+6.50% tight / +4.76% wide / +4.14% none) so the parameter is unresolved and
the family's spread is about as wide as its effect. And the cost model is still
A23's small-order one: impact, suspension and auto-rejection are in no number
here, and two of the ten current picks trade under Rp 2bn a day.

---

# STANDING INSTRUCTION — the deliverable is a signal with three levels

*Added 2026-09-04 at the user's direction: "i want you to be a trading bot that
can give accurate signals for winning trades so you need to have entry, tp and
sl". This section governs the FORM of every signal this repo emits from now on.
It does not override §2 or §11 — a level that is measured to lose money is still
reported as losing money, next to the level.*

## The contract

Every signal this repo hands the user carries **four** things, never fewer:

| | |
|---|---|
| **ENTRY** | a price and the condition that produced it |
| **SL** | a resting stop, from the user's own fill, live every session |
| **TP** | a target, or an explicit scale-out, or an explicit "none and here is what one costs" |
| **the measured cost of each level** | what that SL and that TP did to CAGR, drawdown and win rate in the backtest that produced them |

The fourth column is the part that is not negotiable. `scripts/rules.py` and
`scripts/daily_signal.py` must print it. A level without its measured
consequence is the thing this repo exists to avoid: A26 and A27 both record the
resolution, and it is the same one here — **draw what was asked for, and print
the measurement beside it. Refusing to draw it, or drawing it silently, are both
worse.**

## What is measured about each level, as of H56

- **SL: adopt it.** S1 was registered as "a hard stop will not cut the
  PORTFOLIO's maximum drawdown" and **FAILED**. Every level from −10% to −30%
  cut it in BOTH halves (−42.0% → −30.7%/−36.3%), worst single name −91% → −41%,
  at no measurable cost in return. `STOP = 0.20` is the middle of the family and
  deliberately not its argmax.
- **TP: it costs money, and the cost is monotone in tightness.** S3's predicted
  null CONFIRMED: +20%/+30%/+50% read 6.36%/7.83%/8.77% CAGR against 10.24% with
  none. A target truncates the right tail, and the right tail is the return.
- **So the TP that ships is the one that costs least, not a round number**, and
  its cost is printed next to it every time.

## The three things that make a target legitimate here

1. **Wide beats tight, always.** The whole grid is monotone. Any request for a
   tighter target gets the number for that target, not a refusal.
2. **A scale-out is not a target.** Selling a fraction and letting the rest run
   is the only form of profit-taking that does not cap the winner, and it is
   priced separately.
3. **A trailing stop rises with the position.** It is the honest way to "take
   profit" without naming a level, and it is in the same table.

## What must never be claimed

"Accurate signals for winning trades" is the goal, and these are the words the
repo may not use to describe what it has:

- **No win rate without its horizon and its threshold.** A28: the machinery
  emits probabilities, so a hit rate is undefined until someone picks a
  threshold, and whoever picks it decides the answer.
- **No per-trade mean without the mean LOG beside it.** A36 measured the two
  disagreeing in SIGN: a +5.77% average trade compounding at −5.3% a year.
- **No level quoted without the benchmark.** A19 and A31 both record the missing
  comparison as the error class that manufactures a result.
- **Nothing described as validated.** The holdout was spent at H16. Every number
  in this repo is in-sample, and A23 applies in full to third-party money.
