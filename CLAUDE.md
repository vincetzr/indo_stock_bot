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
