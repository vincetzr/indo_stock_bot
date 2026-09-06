# §58 assessment — environment, sources, architecture A–G

*2026-09-06. Produced BEFORE implementation, as the master brief's §58 requires.
Every status code and number below was measured this session unless it cites a
hypothesis number, in which case it is quoted from that memo. Anything not
established is written DATA UNAVAILABLE with what was tried and what would
change it, per the brief's §0.*

---

## Headline

**Roughly half of this brief already exists, and three of its engines have
already been built, run, and returned nothing.** The repo is 243 commits, 328
Python files, ~70k lines, 2,393 passing tests, a Gate-0-passing point-in-time
spine of 2,866,877 bars over 919 tickers (89 delisted) from 2000-03-30, and 347
registered trials logged in `hypotheses.md`.

Four things decide what the platform can honestly become:

1. **The brief's §10 broker summary, §11 hitung barang and §12 money flow have
   been measured and are null.** Not "not built" — built, run, and negative
   against their own permutation nulls. The brief's §11 anticipates exactly
   this ("if they do not predict, reduce their model weight"). §D and the
   dead-ends table give the numbers.
2. **`sabarkaya.com`, the source §11 names, expressly disallows this crawler.**
   Its robots.txt carries `User-agent: ClaudeBot / Disallow: /` and
   `Content-Signal: ai-train=no, use=reference`. It is closed by the operator's
   own terms, not by a judgement call.
3. **Fundamentals are the one genuinely untested lever**, and the binding
   question is not availability but *point-in-time*: a restated statement
   applied to a historical bar is look-ahead, and without filing dates no
   fundamental backtest in §39 is valid. This is the single most important open
   question in the assessment.
4. **The holdout was spent at H16.** Every number this repo has is in-sample.
   No amount of new engineering changes that; only new data does.

**Two urgent operational findings**, both measured today and both outside the
brief's scope but affecting it:

- **The intraday refresh would destroy 1,092,171 hourly bars.** `Cache.write`
  replaces rather than merges, and 34.94% of the cached 1h bars are older than
  Yahoo's 730-day window, so they cannot be re-fetched. The cache is currently
  an archive that a refresh would silently truncate.
- **`CPO=F`'s `meta.regularMarketPrice` is 820.0 stamped 2024-03-11 — stale by
  2.5 years — while its array is live at 1,234.25.** Today's settled-close fix
  declined it, but only because of the date-match guard; the settled-session
  test alone returned True. Verified on `CPO=F` and `MTF=F`.

---

## A. SYSTEM ARCHITECTURE

```
                      ┌──────────────────────────────────────┐
                      │  SOURCES (§B)                        │
   Yahoo chart ───────┤  daily OHLCV 919 names 2000→   BUILT │
   Yahoo intraday ────┤  1h × 766 names, 3.1m bars    BUILT │
   IndoPremier ───────┤  broker windows, range mode   BUILT │
   RSS ×4 ────────────┤  news, live-only, QUARANTINED BUILT │
   World Bank ────────┤  Indonesian macro annual      ABSENT │
   BPS webapi ────────┤  needs free key               ABSENT │
   GDELT ─────────────┤  point-in-time narrative      ABSENT │
   idx.co.id ─────────┤  403 Cloudflare → manual DL   PARTIAL│
   sabarkaya.com ─────┤  DISALLOWED by robots.txt     CLOSED │
                      └───────────────┬──────────────────────┘
                                      ▼
        ┌───────────────── SPINE (Gate 0 PASSES) ─────────────── BUILT ┐
        │ point-in-time ARA/ARB, fraksi harga, lot, halt schedules      │
        │ corporate-action adjustment · off-tick vendor-adjust detector │
        │ survivorship: 121 delisted recovered, 2.87%/yr attrition      │
        │ ONE-SIDED: snapshot ends 2019-04-07, death months missing     │
        └───────────────────────────┬──────────────────────────────────┘
                                    ▼
   ┌──── FEATURES ────┐  ┌──── ENGINES ────┐  ┌──── VALIDATION ────┐
   │ price/TA  BUILT  │  │ regime   ABSENT │  │ purged WF    BUILT │
   │ flow      BUILT  │  │ sector   ABSENT │  │ clustered    BUILT │
   │ structural PARTIAL│ │ macro    PARTIAL│  │   perm nulls       │
   │ fundamental ABSENT│ │ narrative ABSENT│  │ half-split   BUILT │
   │ narrative  ABSENT │ │ multibag PARTIAL│  │ random ctrl  BUILT │
   └──────────────────┘  └─────────────────┘  │ calibration  BUILT │
                                    │          │ 3× BH bench  BUILT │
                                    ▼          └────────────────────┘
        ┌─────── SIGNAL / PORTFOLIO / REPORT ────────┐
        │ rules.py entry+SL+TP        BUILT (H54/56) │
        │ daily_signal.py scanner     BUILT (H42)    │
        │ brief.py twice-daily        BUILT (A11)    │
        │ Pine chart + cone           BUILT (H32)    │
        │ signal→outcome learning     ABSENT ←cheapest│
        │ research memory / theses    ABSENT          │
        └────────────────────────────────────────────┘
```

**Component status against the brief's numbered engines** (41 slots classified;
full table in the workflow probe, summarised here):

| brief § | engine | status | evidence |
|---|---|---|---|
| 5 | global macro | **PARTIAL** | `data/overnight.py`, 16 symbols; no US macro series |
| 6 | macro transmission | **PARTIAL** | A13 rank correlations measured; no sector-sensitivity map |
| 7 | market regime | **ABSENT** | no HMM/clustering; `market.py` has breadth only |
| 8 | sector rotation | **ABSENT** | sector map exists (A14), never used in a study |
| 9 | stock universe | **BUILT** | `bhbench.load()`, PIT eligibility |
| 10 | broker summary | **BUILT, NULL** | H9/H10/H11 — see dead ends |
| 11 | hitung barang | **BUILT, NULL** | H10 cohort P&L, inventory ledger; source now closed |
| 12 | money flow | **BUILT, NULL** | H12 investor split, powered null |
| 13 | technical | **BUILT** | H13, 8 features, `indicators.py` |
| 14 | chart vision | **PARTIAL** | `paint_suite.py` renders; no vision read-back |
| 15 | fundamental | **ABSENT** | 59 names of 725 cached |
| 16 | corporate action | **BUILT** | verified registry, spine repairs |
| 17 | news | **BUILT, QUARANTINED** | A12; AST test forbids import into spine/features |
| 18–21 | narrative | **ABSENT** | GDELT route newly found, untested at scale |
| 22–25 | multibagger | **PARTIAL** | H23/H24 base rates; no scenario model (needs fundamentals) |
| 26–28 | management / moat / TAM | **ABSENT** | needs fundamentals + filings |
| 29–31 | market-wrong / disconfirm / narrative-vs-price | **ABSENT** | analyst-judgement layer |
| 32 | historical multibagger DB | **PARTIAL** | base rates exist; false-positive comparison blocked by one-sided survivorship |
| 33 | value trap | **ABSENT** | needs fundamentals |
| 34–35 | signals, multi-horizon | **BUILT** | `rules.py`, `daily_signal.py`; horizons per H23/A21 |
| 36–37 | scoring, ML | **BUILT** | H27 gradient-boosted cross-sectional, purged WF |
| 38 | signal learning | **ABSENT** | **cheapest high-value item in the brief** |
| 39–40 | backtesting, adversarial | **BUILT** | `bhbench.py`, permutation nulls, half-splits |
| 41–42 | risk, portfolio | **PARTIAL** | H56 stop/TP, `portfolio.py`; no correlation limits |
| 43–45 | reports | **PARTIAL** | `brief.py` covers ~8 of 17 sections |
| 46 | alerts | **PARTIAL** | Routine fires a scan; no P0–P3 tiering |
| 47 | research memory | **ABSENT** | `hypotheses.md` is the manual version |
| 48 | database | **PARTIAL** | parquet; no relational store |
| 50–51 | testing, reproducibility | **BUILT** | 2,393 tests, config+seed |
| 53 | calibration | **BUILT** | A28: Brier skill, AUC, band coverage |
| 54 | ensemble | **ABSENT** | H27 is a single model |

**On §49's proposed directory layout:** A2 already settled this and the answer
stands — the brief's layout is the target for *new* work; existing code stays
put. Wholesale restructuring of 328 files and 2,393 tests is destructive for no
research gain.

---

## B. DATA SOURCE MATRIX

All statuses measured 2026-09-06. Agent-proxy `recentRelayFailures: []` after
every probe, so **every 403 below is destination-side, not egress policy.**

| source | availability | route | depth | freq | reliability | cost | auth | fallback |
|---|---|---|---|---|---|---|---|---|
| Yahoo chart, `.JK` | **200** (429 without UA) | JSON | ^JKSE 9,186 bars from 1990; BBCA 5,489 from 2004 | daily | high | free | none | none — this is the spine |
| Yahoo `meta.regularMarketPrice` | **200** | JSON | today's settled close ~20 min after the bell | intraday | **conditional** — stale 2.5yr on `CPO=F` | free | none | date-match guard mandatory |
| Yahoo intraday 1h | **cached** | JSON | 766 names, 3.1m bars, 2023-07→2026-08 | 730d rolling | high | free | none | **refresh destroys 1.09m bars — see §E** |
| IndoPremier broker windows | **200** | HTML | 217 tickers, 89 broker codes, 379,143 rows | fortnightly | medium | free | none | manual download |
| `idx.co.id` (7 paths) | **403** ×7, identical 4,545B Cloudflare | — | — | — | — | — | — | **user downloads in browser → `broker_collect.py --ingest`** |
| `sabarkaya.com` | **200 but DISALLOWED** | robots.txt names `ClaudeBot: Disallow: /` | — | — | — | — | — | user's own account export, or written permission. **Not a decision this repo makes (A5).** |
| TradingView scanner | robots.txt **disallows** `/indonesia/scan` | — | — | — | — | — | — | not built (A14); taxonomy also wrong (ASII → "Technology Services") |
| World Bank | **200** | JSON | Indonesian GDP, 66 obs | annual | high | free | none | IMF/OECD |
| BPS webapi | **200** `{"status":"Error","message":"Parameter Key is Missing."}` | JSON | unknown | quarterly | unknown | free | **free key needed — user must register** | World Bank |
| BPS website | **403** Cloudflare | — | — | — | — | — | — | webapi above |
| Bank Indonesia | **200**, 145,702B | HTML only | BI rate, inflation, reserves | monthly | medium | free | none | World Bank, Yahoo `IDR=X` |
| FRED api | **HTTP 400** — `"Variable api_key is not set"` | JSON | full US macro | daily/monthly | high | free | **free key needed — user must register** | none tested |
| ALFRED vintages | **ProxyError** on `alfred.stlouisfed.org` | — | — | — | — | — | — | **DATA UNAVAILABLE.** Tried: urllib + requests, both fail; `recentRelayFailures: []` so not policy. Would change it: a FRED key (the api host answered 400, so it IS reachable) — ALFRED vintages may be served via the keyed API. **This decides whether §6 macro can be backtested at all.** |
| GDELT doc API | **200** then **429** | JSON | per-entity daily volume 2015→; ADRO 2018–2024 returned 103,900B | daily | **rate-limited** | free | none | BigQuery (GDELT is on the free tier; `CLOUDSDK_AUTH_ACCESS_TOKEN` is set in env) |
| Google News / CNBC ID / Kontan / Detik RSS | **200** | RSS | live only | continuous | medium | free | none | **no point-in-time archive** |
| Bisnis.com, idnfinancials | **403** | — | — | — | — | — | — | — |
| Wayback CDX | **200** | JSON | archive-dependent | — | medium | free | none | — |
| Yahoo fundamentals | **probe in flight** | — | — | — | — | — | — | see §F stage 3 |
| PyPI | **200**, direct (in `NO_PROXY`) | — | — | — | — | free | none | installs persist across resumes; `requirements.txt` lists only 4 of 85 packages |

---

## C. DATABASE DESIGN

**Recommendation: DuckDB over the existing Parquet files. No migration.**

Measured this session on `price_panel.parquet` (2,866,877 × 23, 268 MB):

| operation | duckdb over existing parquet | pandas | sqlite |
|---|---|---|---|
| one-ticker last-250 | **0.050 s** | 0.650 s (filters) / 1.827 s (full load) | 0.003 s (indexed) |
| cross-sectional agg by date | **0.035 s** | — | 0.988 s (**28×**) |
| full table → pandas | **8.1 s** | — | 35.0 s (4.3×) |
| build cost / size | 8.9 s / 493 MB | — | 22.0 s / 542 MB |

Postgres client 16.13 is present and the server package *is* installed (started,
verified, stopped again), but a remote database is impossible — the agent proxy
does not carry raw TCP. DuckDB reads the parquet already on disk, so it costs
nothing to adopt and nothing to abandon.

Tables, following the brief's §48 (new ones marked ✚):

```
prices ─────────────< price_panel.parquet            BUILT
intraday_1h ────────< h1_panel                       BUILT (rebuild script)
broker_windows ─────< broker_windows.csv.gz          BUILT
investor_split ─────< investor_split.csv.gz          BUILT
reference ──────────< ARA/ARB, fraksi, lot, halts    BUILT
corporate_actions ──< verified registry              BUILT
sectors ────────────< idx_classification.parquet     BUILT (frozen 2024-07-10)
news ───────────────< 34 cached feeds                BUILT (quarantined)
✚ fundamentals      (ticker, period_end, FILING_DATE, field, value, restated_flag)
✚ macro             (series, date, RELEASE_DATE, value, vintage)
✚ narratives        (term, date, gdelt_volume, source)
✚ signals           (ts, ticker, features_json, signal, entry, sl, tp, model_ver)
✚ signal_outcomes   (signal_id, mfe, mae, ret, held, exit_reason, outcome)
✚ theses            (ticker, date, thesis, assumptions, milestones, status)
```

**The two ✚ date columns in bold are the whole design.** `filing_date` and
`release_date` are what make a fundamental or macro backtest legal; without them
those tables are decoration.

---

## D. MODEL ARCHITECTURE

| model | can honestly estimate | cannot | prior measured result |
|---|---|---|---|
| **macro** | contemporaneous rank correlation with IHSG | anything predictive without release-date vintages | A13: S&P +0.207, Nasdaq +0.199, BHP +0.185, DXY −0.080, Asia ~0. Strongest explains ~4% of variance against a 56 bp round trip |
| **regime** | a *descriptive* label with an honest cluster count | a validated regime count — GMM/k-means must return k | A10: HDBSCAN found **zero** clusters where GMM forced to k returned partitions decaying to chance by k=5. Pair any k-method with one that can say "none" |
| **technical** | cross-sectional ranks, calibrated touch probabilities | tradeable timing | H13: all 8 features significant, **all net-negative after cost**; the predicted-null `squeeze` fired at t=+3.55 — on 2m rows significance is free |
| **broker** | crossing ratio (a stable firm attribute) | returns | H9 IC −0.0190; H11 best p 0.119 vs a 0.0018 bar; H12 powered null, bar missed by 33–55× |
| **fundamental** | **untested** | nothing yet — blocked on filing dates | A25 named it the honest next instrument |
| **narrative** | live tagging today; **point-in-time volume if GDELT scales** | back-tested narrative alpha until the archive is collected | A12: quarantined by an AST test |
| **multibagger** | base rates by horizon and cell | which name, with any confidence | H23: P(2x) 9.5%/27.0%/55.5% at 1/3/10y. Effective n **56** whole-panel, ~6 per decile, 30 distinct names ever in the top decile |
| **ensemble** | — | — | H27: eleven collinear price features over one macro history — "the interaction space is empty" |

**The §55 ladder, applied.** §22–31 (multibagger thesis, TAM, moat, management,
why-the-market-is-wrong, disconfirmation) are **analyst judgement the system can
structure but never validate**. They belong at the THESIS rung, never at
PREDICTION. Saying so is what makes the measurable half trustworthy.

---

## E. BACKTESTING DESIGN

Every guard below already exists; the file is named so it is inherited rather
than rewritten.

| bias | guard | where |
|---|---|---|
| look-ahead | `select` sees only one bar's rows; the future is not in the frame | `bhbench.walk` |
| look-ahead, features | weekly/monthly state stamped at the first bar strictly *after* the period closed, with a test that mutating a future bar cannot change an earlier state | `mtf.features`, `test_mtf.py` |
| look-ahead, live-vs-replay | the replay must reproduce the live scanner's row at three past dates | `test_signal_backtest.py` |
| survivorship | 89 delisted names in the panel; delisting realised at the last real print | `Prices.exit_price` |
| survivorship, residual | **one-sided** — snapshot ends 2019-04-07, 105 of 121 names end at the snapshot not at their delisting. `FORCED_RETURN=-1.00` is an ASSUMPTION | `spine/universe.py` |
| overfitting | clustered permutation null as the **first** statistic, block LABELS reassigned across blocks | `mtf.block_null`, A17/A25 |
| overfitting | half-split — the only replication test this repo trusts | `bhbench.half_cagr`, A18 |
| overfitting | rebalance-phase majority, not offset 0 | `bhbench.evaluate`, A39 |
| overfitting | predicted-null features registered in advance | A9's `squeeze`, H55's M3 |
| instrument error | positive control on synthetic data with a known answer | `m0_positive_control`, A26/A36 |
| missing benchmark | three buy-and-hold arms + a matched random control | `bhbench`, A19 |
| **data loss** | **NOT GUARDED** — `Cache.write` replaces; a 1h refresh would destroy 1,092,171 bars older than Yahoo's 730-day window | `data/cache.py` ← **fix before any intraday work** |

Metrics the brief's §39 lists that the repo does **not** yet compute: Sharpe,
Sortino, profit factor, expectancy, exposure, tail risk. It computes CAGR, max
drawdown, win rate, turnover, mean/median/mean-log and half-split. Adding the
missing six is trivial and should happen in stage 1.

---

## F. IMPLEMENTATION ROADMAP — ordered by cost-to-falsify ascending

The repo's own §4 argues this ordering and it holds: the cheapest question that
could kill a branch goes first.

| # | stage | why first | assumption | limitation | test | cost |
|---|---|---|---|---|---|---|
| 0 | **Fix `Cache.write` to merge on `ts`** | it is destroying data now | none | — | a test that a refresh preserves out-of-window bars | 1 h |
| 1 | **Signal→outcome store (§38)** | costs nothing, compounds daily, and is the only thing that ever produces out-of-sample evidence now the holdout is spent | none | needs months to pay | schema + a test that every emitted signal is logged | 1 d |
| 2 | **§39 metric completion + DuckDB** | pure plumbing, 13–28× faster, no migration | none | — | numbers must match the existing ones | 1 d |
| 3 | **Fundamental point-in-time probe** | one question kills or opens §15, §22–28, §33 at once: *are filing dates available?* If no, six engines are unbuildable and we stop pretending otherwise | Yahoo or a filing source carries report dates | if absent, the branch dies here | reconstruct one known restatement | 1 d |
| 4 | **FRED + BPS keys, ALFRED vintages** | same shape: one question decides whether §5–§6 can be backtested | vintages retrievable | revised-only series ⇒ macro stays descriptive | reproduce a known revision | 1 d, needs **user to register two free keys** |
| 5 | **Regime engine (§7)** paired with a method that can return "no clusters" | cheap on existing data | regimes exist | A10 warns they may not | HDBSCAN alongside GMM; report if k=0 | 2 d |
| 6 | **Sector rotation (§8)** | sector map already on disk, never used | frozen 2024-07-10 map is adequate | misses 41 post-July-2024 listings | permutation null on sector ranks | 2 d |
| 7 | **GDELT narrative collection (§18–21)** | the only route to a *backtestable* narrative; rate-limited so collection must start early | GDELT scales via BigQuery | 429s from this IP; sustainable rate not established | a point-in-time replay test like `test_signal_backtest` | 1 w collection |
| 8 | **Multibagger scenario model (§24)** | blocked on stage 3 | fundamentals arrive | effective n ~56; cannot validate a name-level claim | scenario arithmetic unit-tested; **no accuracy claim** | 3 d |
| 9 | **Chart vision (§14)** | most speculative, and H36 already measured pattern skill at AUC 0.58 | vision adds over quantitative | likely marginal | every pattern needs a base rate before it is used | 3 d |
| — | **NOT ON THE ROADMAP** | broker/hitung-barang/money-flow rebuild | — | — | — | see dead ends |

---

## G. MVP

**It already exists.** The smallest useful daily IDX report is three commands:

```
python scripts/refresh.py --panel --max-age 0     # ~6 min, 843 names, 98% same-day
python scripts/brief.py --session post            # market state, groups, flow, news
python scripts/rules.py                           # entry / SL / TP with measured costs
```

The Routine `trig_01XFpgpTcoNEsCBx3ZAVAA2Z` already fires an end-of-day scan on
`0 11 * * 1-5`. The brief asks for 19:15 Shanghai; **19:15 Shanghai = 18:15 WIB
= 11:15 UTC exactly** (verified with `zoneinfo` for Jan/Jul/Sep 2026 — neither
zone observes DST), and IDX closes 15:50 WIB = 08:50 UTC, so the run sits
**2 h 25 m** after the bell. One change: cron `0 11` → `15 11`.

What the MVP does **not** yet cover of the brief's §43 seventeen report
sections: sector rotation (5), hitung barang (8), multibagger watchlist (12),
narrative watch (13), model performance (16). Five of seventeen, and stages 1,
5, 6 and 7 above fill four of them.

---

## Dead ends — do not rebuild

| brief asks for | measured | verdict |
|---|---|---|
| §10 broker summary | H9: IC **−0.0190**, HAC t −2.86, empirical p 0.005 against a 0.0025 Bonferroni bar. Quintile spread −0.215%/fortnight, t −0.70 — indistinguishable from zero **before costs**. Sign flips between adjacent liquidity quintiles | **NULL** |
| §10 broker identity | H11: six statistics, two stores, 89 codes over 12.6 years, every one inside its own 200-draw permutation null. Best p **0.119** against a 0.0018 bar | **NULL, and no persistence** |
| §11 hitung barang | H10 cohort P&L: round-trip median **−25.3 bps** against a shuffled-label null of **−32.5** — the null sits on top of the signal. Inventory negative 46.4% of the time | **NULL**; source also closed by robots.txt |
| §12 money flow | H12: foreign **−1.70 bps**/fortnight (p 0.692), domestic **+1.02** (p 0.816). Null sd 6.34/4.95, so the 56 bp cost bar is **8.8–11.3 null-sds away** — a *powered* null, not an inconclusive one | **NULL** |
| §13 as a timing tool | H13: all 8 features significant, **all net-negative at every horizon**. Rebalance costs 1.7–1.9%; gross spread 0.15–0.36% | **significant ≠ tradeable** |
| cycle/time methods | H31: ZigZag pivot spacing CV **2.246** against a block-bootstrap null of **1.340** — a cycle would make spacing *more* regular; the data makes it less | **NULL** |
| Fibonacci levels | H34: z = +0.77 / +0.68 / +0.41 / −0.95 across 280,228 touches; 0.500 reads 0.3611 between 0.475's 0.3631 and 0.525's 0.3571 | **NULL — a smooth function read at arbitrary points** |
| Hull ribbon as a rule | H48: **−1.32%/yr against hold's +6.42%**, winning on 31.3% of 891 names, negative in both halves of every universe. H49: its *timing* does carry information (+9.6 points vs a run-shuffled null) — the information is smaller than the toll | **rule NULL, signal real** |
| exit rules | 169 configurations across H17/H18/H35/H38/H40/H47, none beating a hold. H20 **withdrew** H17's and H18's headlines on portfolio accounting | **NULL** — but see H56: a stop cuts *drawdown* in both halves at no return cost |
| multi-timeframe confluence | H55: confluence z **+0.40**; the foreign-trend null (a *stranger's* weekly trend) z **+0.39** | **NULL — it measures the market's regime** |
| intraday trading | H55b: one round trip = **1.65×** the median hourly move; only 32.0% of 1h bars clear it. Perfect intraday hindsight is worth **+1.1%**, 1–2% of one σ | **closed by arithmetic** |
| volatility screens | H25 cleared Bonferroni at p 0.00020 and A24 **retracted it**: scored on asymmetry it reads z −0.19, indistinguishable from a random cell | **retracted** |

**What survived, and it is short:** H26 strength+calm, skew **2.60** vs a null of
1.20 ± 0.15, **p 0.00033** — the only result that ever cleared this repo's
Bonferroni bar. H54's sticky basket, beating the IHSG in **6 of 6** rebalance
calendars by a median +6.50%/yr. H23's ten-year liquid-decile tilt. H27's
cross-sectional model at **2.31 skew out of sample**. H56's stop.

---

## Binding constraints, stated once

- **The holdout was spent at H16.** Every number here is in-sample. Stage 1's
  signal store is the only mechanism that produces new out-of-sample evidence.
- **347 trials** put the Bonferroni bar at **0.05/347 = 0.00014**.
- **Effective n is small**: 56 at a ten-year horizon over the whole panel, ~6
  per decile, 30 distinct names ever in the top decile. No resampling
  manufactures independence that 24 years of one country's history does not
  contain.
- **The cost model is A23's small-order one**: fee + fraksi-harga half-spread,
  with **no impact, suspension or auto-rejection term**. On thin names a
  Rp 500m position is ~28% of a day's volume.
- **Survivorship is bounded, not corrected**: 105 of 121 recovered delisted
  names end at the 2019 snapshot rather than at their delisting, so
  `FORCED_RETURN = −1.00` remains an assumption.

---

## Postscript — the container was rebuilt cold, 2026-09-06

The assessment workflow's remaining six probes were killed by a container
restart. What the restart itself taught is worth more than the probes were.

**All code survived; all raw data did not.** The three most recent commits were
on the remote and not local — the container restored an older snapshot — and
`git reset --hard origin/...` recovered them intact. Nothing in git was lost,
because every unit of work had been pushed.

**`data/` went from 1.9 GB to 4.5 MB.** It is gitignored by design (CLAUDE.md
§13: "raw pulls, immutable, gitignored"), and that design assumes the data is
re-fetchable. Measured, it partly is not:

| store | before | after | recoverable? |
|---|---|---|---|
| `cache/ohlcv` | 901 files | **838 refetched in 3 min** | yes |
| `spine/price_panel` | 2,866,877 rows, 919 tickers | **2,605,914 rows, 827 tickers** | yes, minus below |
| `cache/delisted` | 121 names | 0 | **NO — see below** |
| `cache/intraday` | 1,004 files, 3.1m 1h bars | 0 | **partly — 34.94% were older than Yahoo's 730-day window** |
| `cache/ipot_broker` | 42,227 files | 0 | yes, but A1 prices it at 31,824 polite requests |
| `cache/news` | 34 | 0 | live-only anyway |
| `cache/fundamentals` | 59 | 0 | yes, and it was the gap regardless |

**THE SURVIVORSHIP REPAIR IS GONE AND ITS SOURCE IS NOT RECORDED.** The panel
now reads **827 tickers, 0 delisted**, against 919 with 89 before. Yahoo does
not serve the dead names: probed `MYRX.JK` 200 with **0 bars**, `SUGI.JK`,
`BTEL.JK`, `TRAM.JK` 200 with **36 bars** each, `SIAP.JK` **404**. And
`spine/universe.py` documents the recovery's provenance only in prose — "a
published point-in-time listing of 627 IDX tickers" as of 2019-04-07 — with **no
URL, no accession date and no checksum anywhere in the repo**. So the artefact
that turned three guesses into measurements (2.87%/yr attrition, 4.8pp
pre-delisting drag, a survivorship-free universe) cannot be re-obtained from the
repo's own record.

That is a documentation failure, not a data failure, and it is the more
embarrassing of the two: the repo was careful enough to keep the delisted names
in a separate directory so they could not silently change a study, and not
careful enough to write down where they came from.

**pip installs do NOT survive a cold rebuild.** The environment probe measured
them persisting across warm resumes (eight packages from 08-07..08-24 still
present on 09-06) and explicitly flagged the cold case as DATA UNAVAILABLE.
It is now answered: after the restart, `pandas`, `numpy`, `pyarrow`,
`scikit-learn`, `matplotlib`, `pytest`, `statsmodels`, `lightgbm`, `duckdb` and
`yfinance` were all gone, leaving 38 packages. `requirements.txt` listed **four**
of them. A manifest that is a subset of what the code imports is not a manifest.

### What this changes in the roadmap

Stage 0 was "fix `Cache.write` to merge rather than replace, because a refresh
would destroy 1,092,171 out-of-window 1h bars". The restart destroyed them
first, which does not make the fix less necessary — it makes it more, because
the *next* cache to accumulate beyond Yahoo's window will be lost the same way
unless the store is durable.

Three items are added ahead of everything else:

| # | item | why |
|---|---|---|
| **0a** | `requirements.txt` now lists the full runtime (done) | the rebuild blocker |
| **0b** | Record the 2019 snapshot's URL, accession date and checksum in `spine/universe.py`; re-obtain the 121 names | it is the only survivorship repair and it is currently unreproducible |
| **0c** | Decide what in `data/` is *derived* (re-buildable from a script) versus *irreplaceable* (a point-in-time artefact), and commit or externally back up the second class | 1.9 GB of gitignore made no distinction between the two, and the distinction is the whole point |

**Verified working after recovery:** 2,390 tests pass, 3 skipped; the panel
rebuilt to 2026-09-04; `scripts/rules.py` emits the live basket with entry, SL
and TP. What is degraded: every study that depended on the delisted names is now
running on a survivorship-biased universe until 0b is done, and `gate0.py`'s
survivorship check will fail.

---

## 0b done — the survivorship repair is back, wider, and reproducible

**The lost 2019-04-07 snapshot was not re-found and its source remains unknown.**
What replaced it is a different artefact, and both are now recorded as constants
in `spine/universe.py` so no study can silently swap one for the other.

| | lost snapshot | current recovery |
|---|---|---|
| source | **unknown — prose only** | `github.com/wildangunawan/Dataset-Saham-IDX` @ `bc0ac771` |
| names recovered | 121 | **145** |
| window | reached back before 2019-04-07 | **2019-07-29 → 2025-02-21**, 1,355 bars |
| dates each death? | **no** — "the months in which a name actually died are still missing" (A2) | **yes**, from the last bar carrying volume |
| reproducible? | **no** | `python scripts/delisted_collect.py --clone` |

Measured: 958 dataset names against 827 in the live spine, **145 the spine does
not have**, median total return **−54.5%**, and **68 stopped trading before
2025-02**. A median that deeply negative is the point — survivorship bias *is*
the absence of the losers, so a recovery whose median looked like the survivors'
would have recovered the wrong thing. A test asserts it stays below −20%.

Sample death dates, derived from volume: ARMY 2019-11-29, MYRX 2020-01-15,
TRAM and SMRU 2020-01-22, IIKP 2020-01-22, HOME 2020-01-31, RIMO 2020-02-11.

**Wider but shallower.** Names that died before 2019-07-29 are still gone, so
the recovery is one-sided in the *other* direction from the old one. That is
recorded rather than papered over.

**One trap worth naming.** The dataset has a `delisting_date` column. It is
**empty for all 958 names** — a header only. Its name invites exactly the
assumption that would produce a table full of silent NaTs, so the death date is
derived from volume and a test pins that.

### The licence is the user's call, and nothing imports this until they make it

CC BY-NC 4.0 on the compilation. The dataset's own README states the data is
taken from idx.co.id and "all data in the dataset belongs to PT Bursa Efek
Indonesia", pointing at IDX's Syarat Penggunaan — which CLAUDE.md §3 already
records as **barring commercial redistribution while permitting personal
research use**.

A23 records this project being pointed at a client's money. Whether managing
third-party money for a fee is "non-commercial" is not a question this repo
should answer for you. So, following the same pattern as
`data.broker_allowed_hosts` shipping empty (A5):

- the recovery writes to `data/cache/delisted/` and a manifest;
- **nothing under `spine/` or `features/` imports it**, enforced by an AST test
  identical in spirit to the one quarantining the news layer;
- `RECOVERY_LICENCE_PENDING = True` in `universe.py` until you rule.

**What I need from you:** whether CC BY-NC covers your use. If yes, the spine
rebuild includes the 145 names and `gate0.py`'s survivorship check passes again.
If no, the repo stays survivorship-biased and every affected result carries that
caveat — which is worse, but honest, and it is your call to make.

### 0c — the rule that stops this recurring

`data/` was 1.9 GB of gitignore that made no distinction between two kinds of
thing, and the distinction is the entire point:

| class | example | on loss | policy |
|---|---|---|---|
| **DERIVED** | `price_panel.parquet`, `ohlcv/`, indicator panels | rebuild by script in minutes | gitignore, and keep the script |
| **IRREPLACEABLE** | the 2019 snapshot, out-of-window 1h bars, any point-in-time artefact | **gone forever** | **must carry a collector with pinned provenance, or be backed up outside the container** |

Every irreplaceable store now needs four recorded fields — source, pinned
commit or accession date, licence, checksum — and a script that rebuilds it from
them. `scripts/delisted_collect.py` is the reference implementation and
`tests/test_delisted_collect.py` asserts the fields cannot regress to prose.

Still unprotected under this rule: `cache/intraday` (34.94% of its bars were
already past Yahoo's 730-day window and are gone) and `cache/ipot_broker`
(42,227 files, priced by A1 at 31,824 polite requests).
