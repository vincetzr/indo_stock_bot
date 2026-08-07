# idxbot — IDX broker-flow accumulation engine

Reverse-engineers what each exchange member is doing in an Indonesian stock from
broker summary, segments that flow into campaigns, profiles how each desk enters
and takes profit, screens a universe for accumulation, and turns a signal into an
executable trading plan with IDX-correct tick, lot and auto-rejection handling.

```bash
pip install -r requirements.txt

PYTHONPATH=src python3 -m idxbot.cli screen --universe lq45
PYTHONPATH=src python3 -m idxbot.cli plan --tickers BBCA --pool lq45
PYTHONPATH=src python3 -m idxbot.cli dashboard --universe lq45
```

---

## Read this first: what is and isn't verified

| Claim | Status |
|---|---|
| Price/volume history is real and deep | ✅ 169,553 daily bars / 30 names / up to 26.2 yrs; IHSG from **1990-04-06** |
| Ledger, cost basis, campaign maths | ✅ 60 unit tests, hand-computed expectations |
| Tick / lot / ARA-ARB / fee mechanics | ✅ unit tested against IDX rules |
| Score has no look-ahead | ✅ explicitly tested — scoring bar *i* is unchanged by appending future bars |
| Price-only score predicts forward returns | ❌ **REFUTED — it is inverted. See below.** |
| **How J.P. Morgan / UBS / CLSA actually trade** | ❌ **not verified — no real broker-summary data was obtainable** |

### ⚠️ The price-only signal does not work — it is backwards

Tested on **56,732 observations across 66 tickers, 2001–2026, all real data**:

| Score bucket | n | 20d | 60d |
|---|---|---|---|
| 0–39 (lowest) | 24,433 | **+2.44%** | **+6.86%** |
| 40–54 | 22,039 | +1.18% | +4.07% |
| 55–64 | 7,647 | +0.76% | +4.52% |
| 65+ (signal) | 2,613 | +0.63% | +4.49% |

Monotonically inverted, and it holds in both halves of the sample and ex-crisis
(t = −2.57 / −4.42 / −4.76). Wyckoff phase E — which the planner *blocks* as "a
chase" — was the best state at **+10.67% / 60d**, versus +3.38% for the spring
setup the engine rates highest.

**IDX 2001–2026 rewarded momentum, not mean reversion into bases.** Do not trade
the price-only score as a buy signal. Full analysis, caveats, and why "just
invert it" is a trap: **`docs/FINDINGS.md`**.

The broker-flow half — the actual thesis — remains untested, because the data is
paywalled. That is the experiment worth running, and the code is ready for it.

**There is no free public IDX broker-summary API.** `idx.co.id` is behind a WAF
(403), Stockbit needs an authenticated app session, and GoAPI's broker-summary
route exists but is gated on a paid key. So the repo ships a **simulator** to
make the pipeline runnable end to end.

The simulator *assumes* institutions buy weakness and retail chases momentum. Run
the playbook against it and it will report that institutions buy weakness —
because that was typed in by hand. **Circular, and worthless as evidence about the
real market.** Every report prints a warning banner when broker flow is simulated.

The machinery for the real answer is built and tested. Connect a real source
(`docs/LIVE_DATA.md`) and the same commands produce a genuine answer.

---

## Getting live broker flow

**Broker summary is an aggregation of running trade.** Every IDX print carries a
buyer and a seller member code; summing them by broker *is* the broker summary.
Running trade with broker codes is live on every Indonesian platform, so the
end-of-day delay applies to the aggregated view, not the underlying stream.

```bash
idxbot live --file ticks.jsonl --follow      # tail a live running-trade file
cat ticks.jsonl | idxbot live --stdin        # or pipe it
idxbot live --file session.csv --out data/broker_summary/BBCA.csv
```

Also: for a multi-week accumulation strategy, a two-hour delay costs nothing.
Full discussion, vendor options and ToS considerations in **`docs/LIVE_DATA.md`**.

---

## Commands

| Command | What it does |
|---|---|
| `screen` | Rank a universe by accumulation score, with evidence for each hit |
| `analyze TICKER` | Deep dive: score breakdown, Wyckoff phase, broker positions, campaigns |
| `plan` | Executable plan — entry band, stop, targets, lot-rounded size, R:R |
| `playbook` | Reverse-engineer each broker's entry/exit behaviour across a universe |
| `backtest` | Walk-forward: does the score actually predict forward returns? |
| `dashboard` | Offline self-contained HTML report |
| `live` | Reconstruct broker summary from a running-trade stream |
| `pine [TICKER]` | Pine Script source, or plan levels as paste-ready Pine inputs |
| `watchlist` | TradingView-importable watchlist, grouped by signal level |
| `brokers` | Exchange member registry |
| `data` | What history is actually retrievable per ticker |

Everything reads `config/config.yaml` — no thresholds, weights, broker codes or
universes are hard-coded.

---

## How the signal works

Nine weighted components. Four need broker data; five need only price and volume.
Without broker summary the engine drops the first four and **renormalises the
remaining weights**, so the score keeps its 0–100 scale and stays comparable.
That is *price-only mode*, and it is labelled on every output — a price-only 70
and a broker-confirmed 70 are not the same claim.

| Component | Weight | Broker data | Detects |
|---|---|---|---|
| `inventory_zscore` | 0.22 | ✅ | Bulge-desk inventory build vs its own history |
| `stealth` | 0.18 | ✅ | Heavy institutional buying while price barely moves |
| `smart_dumb_divergence` | 0.14 | ✅ | Institutions absorbing while retail distributes |
| `concentration` | 0.12 | ✅ | Few members taking most of the net supply |
| `wyckoff` | 0.15 | — | Phase structure (spring, sign of strength) |
| `obv_divergence` | 0.10 | — | Volume flow outpacing price |
| `volume_dryup` | 0.08 | — | Supply exhaustion in the base |
| `range_compression` | 0.08 | — | Coiling before expansion |
| `relative_strength` | 0.08 | — | Outperforming the IHSG |

### Reverse engineering, concretely

For each `(ticker, broker)` the ledger replays the daily buy/sell tape into a
moving-average-cost position: **inventory**, **cost basis**, **realised** and
**unrealised P/L**. That series is detrended against a 120-day EWMA (to strip
structural custody/index drift) and segmented by a zigzag into **campaigns**.

Per campaign it measures the four things that characterise a desk:

- `entry_percentile` — where inside the range their buy VWAP sat
- `stealth_ratio` — price move during accumulation vs the campaign's full move
- `markup_pct` — how far it ran above their entry before they unwound
- `exit_capture` — **share of the available move they actually kept**

`exit_capture` is the most useful number: it separates desks that sell near the
high from desks that scale out early into strength.

Two independent views are produced, because each has a weakness the other
doesn't: the **campaign profile** (rich, interpretable, depends on the
segmentation) and the **forward-return edge** (`--edge`: after this broker buys
unusually hard, what happens next, with a t-statistic — no segmentation
assumptions at all). *When they disagree, believe the forward-return test.*

---

## TradingView

TradingView is the charting surface, not a data source — it carries no broker
data. The integration pushes idxbot's conclusions *into* it:

- `pine/accumulation_score.pine` — the price-only score, live on any chart, with
  Wyckoff phase shading, spring/SOS markers and alert conditions
- `pine/broker_campaign.pine` — draws a plan: broker cost basis, entry band,
  stop, targets, anchored VWAP from the campaign start
- `idxbot pine BBCA` — emits that plan's numbers as a paste-ready input block
- `idxbot watchlist --screen` — importable watchlist grouped by signal level

---

## Data sources

| Source | Status |
|---|---|
| Yahoo Finance chart API | ✅ Full daily history. Note: `range=max` silently returns *monthly* bars — this code passes explicit `period1`/`period2` to get true daily depth |
| CSV import | ✅ Drop platform exports in `data/broker_summary/`. Handles English/Indonesian headers, `1.234.567,89` numbers, and lot- vs share-denominated volume (auto-detected) |
| Running trade | ✅ Live aggregation into broker summary |
| GoAPI.id | 🔑 Route exists, needs `IDXBOT_GOAPI_KEY` |
| `idx.co.id` | ❌ WAF-blocked |
| Synthetic | ⚠️ Simulator — clearly labelled, never for trading |

**Coverage limit:** Yahoo has no IDX data before 1990, and most individual stocks
start 2000–2005. "Since the start of the exchange" is not achievable from any
free source — the 1977–1990 era is not available.

---

## Layout

```
config/           config.yaml, brokers.yaml, universe.yaml  (all behaviour)
src/idxbot/
  market.py       tick grid, lots, ARA/ARB, fees, position sizing
  engine.py       orchestration + per-ticker analysis cache
  plan.py         trading plan generation
  backtest.py     walk-forward evaluation
  data/           ohlcv, broker_summary, running_trade, synthetic, cache
  analytics/      indicators, broker_flow, campaigns, playbook, wyckoff, accumulation
  tradingview/    links, watchlists, Pine scripts
  report/         offline HTML dashboard
tests/            60 tests
docs/             LIVE_DATA.md, TRADING_PLAN.md
```

---

## Known limitations

- **Broker summary is flow, not holdings.** Inventory is relative to the first day
  of your data window. The *changes* and their cost basis carry the signal; the
  absolute level does not.
- **A broker code is not a firm's opinion.** `BK` aggregates every client routing
  through J.P. Morgan's membership — pension funds, hedge funds, index trackers,
  delta hedges. Strong evidence of institutional intent; not a confession of it.
- **Broker codes change.** Verify `config/brokers.yaml` against IDX's member
  directory; entries carry a `confidence` field.
- **The backtest flatters itself.** Survivorship (today's constituents),
  overlapping forward windows (inflated t-stats), gross returns unless the
  net-cost column is shown.
- **Campaign segmentation is a model**, not a measurement. Different parameters
  find different campaigns.

---

*Educational tooling, not investment advice.*
