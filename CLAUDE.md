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

## A1. The data-shape blocker on Phase 1 — read first

§4 sizes Phase 1 at ~2,500 days × ~800 names. **The free route that actually works from
here cannot produce that panel**, and it is arithmetic rather than effort:

| route | shape | cost of the §4 panel | status |
|---|---|---|---|
| IndoPremier public module | 1 request per **ticker-day**, top-10 only | ~2,000,000 requests ≈ 27 days of continuous polite fetching | works today |
| `idx.co.id` broker summary | 1 file per **session, all stocks** | ~2,500 files | 403 from any script — Cloudflare bot check on TLS fingerprint |
| Sectors licensed API (`api.sectors.app/v2`) | 1 credit per ticker per **fortnight**, full depth | ~143,000 credits | needs a paid key |

So §3's data question is not a convenience issue. **The one-file-per-session shape is
the only thing that makes §7 feasible at all.** Until one of those three is resolved,
Phase 1 runs on a 10-name panel, not an 800-name one, and the effective sample must be
quoted accordingly.

The 403 is a bot check, not a network block or a geo-block — proven, and documented in
`docs/FULL_REKAP.md`. The three repos §3 cites clear it with `curl_cffi` browser
impersonation. That is the specific act IDX's controls exist to prevent, so this repo
does not do it; `docs/FULL_REKAP.md` §3 has the full reasoning and the sanctioned
alternatives (you download in a browser → `broker_collect.py --ingest`, or the licensed
API).

## A2. What already exists, mapped onto the brief

| brief | repo today |
|---|---|
| §5 spine | **partial.** OHLCV full history via `idxbot.data.ohlcv`. Broker summary 10 names × ~360 sessions in `data/cache/broker_daily`. **No delisted names, no ARA/ARB schedule, no fraksi harga schedule, no broker-code master.** Gate 0 not attempted. |
| §7 Phase 1 | **run and FAILED** on a 1-name panel. See A3. |
| §9.3 cohort P&L | **not built.** Previously declined on the starting-inventory problem; the brief's round-trip restriction is a better answer and supersedes that decision. |
| §9.4 execution style | **built** (`scripts/broker_economics.py`) and **carries the exact bias §9.4 warns about.** See A4. |
| §9.5 archetypes | not built |
| §11 purged walk-forward | partial — walk-forward exists, purge/embargo does not |
| §11 trial count | `hypotheses.md` now exists; 8 pre-registered trials logged |
| §13 layout | **conflicts.** Repo is `src/idxbot/`, `scripts/`, `docs/`, `tests/` with 1,297 passing tests. Restructuring wholesale would be destructive for no research gain, so the brief's layout is treated as the target for *new* work and `reports/` + `hypotheses.md` are adopted now. |

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

## A4. The bias §9.4 names is present in this repo's own code

`execution_edge()` compares each broker's average fill to the day's VWAP **including
that broker's own trades**. §9.4 is right that this shrinks large brokers toward zero.

It also means **a negative result already reported to you is unsafe**: "size does not
explain execution edge" (corr(log volume, edge) = +0.010, p = 0.96) is exactly what this
bias manufactures. That result is withdrawn pending recomputation against a
self-excluded VWAP, which is available from the footer totals:

```
VWAP_excluding_b = (total_value − b_buy_value) / (total_lot − b_buy_lot)
```

Every broker's buy side is exactly a partition of total volume, so this is exact
arithmetic on the top-10 table, not an approximation.

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
