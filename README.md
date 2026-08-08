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
| Ledger, cost basis, campaign maths | ✅ 171 unit tests, hand-computed expectations |
| Tick / lot / ARA-ARB / fee mechanics | ✅ unit tested against IDX rules |
| Score has no look-ahead | ✅ explicitly tested — scoring bar *i* is unchanged by appending future bars |
| `accumulation` profile predicts forward returns | ❌ **REFUTED — inverted. See below.** |
| `momentum` profile predicts forward returns | ⚠️ **Holds, but most of the apparent edge was survivorship** — see below |
| **How J.P. Morgan / UBS / CLSA actually trade** | ❌ **not verified — no real broker-summary data was obtainable** |

### Two results on 55,699 real observations (66 tickers, 2001–2026)

Metrics are **cross-sectional** — ranked within each date on a shared exchange
calendar (median 52 names per cross-section), so market direction cancels — with
the t-statistic computed over dates. Train 2001–2016, **holdout 2016–2026
untouched until the profile was frozen.**

| Profile | Train 20d IC (t) | Holdout 20d IC (t) | Holdout 60d IC (t) |
|---|---|---|---|
| `accumulation` (contrarian) | −0.0379 (−5.30) | +0.0001 (+0.02) | −0.0103 (−1.33) |
| `momentum` (trend) | +0.0612 (+6.56) | **+0.0313 (+3.08)** | **+0.0462 (+4.92)** |

**1. The accumulation thesis is refuted.** The component split is perfectly clean
along family lines: every momentum component positive, every contrarian one zero
or negative — worst `volume_dryup` (t = −6.35). Wyckoff phase E, which the planner
*blocks* as "a chase", was the best-performing state.

**2. A momentum profile built from that diagnosis survives the holdout.** Top
quintile beat bottom by **+5.16% over 60 days (t = 5.90)**; a long-only top-10
basket beat the equal-weight universe by **+4.67% per period (t = 2.32)** with a
smaller drawdown (−16.9% vs −32.2% for IHSG).

### ⚠️ Survivorship audit cut these numbers by ~4x

Restricting to names already liquid at the *start* of the holdout (established
before any measured return happened) collapses CAGR from **35.5% to 8.9%** —
against IHSG's 4.1%. 49% of picks were names promoted into an index *later*, and
those returned +15.7% per 60 days vs +4.1% for established names. A separate
liquidity audit found trades in names with Rp26 juta/day turnover that could
never have been filled. Both are documented in `docs/FINDINGS.md` §5.

### Then it was re-run on the whole exchange — 724 names, not 66

`docs/FINDINGS.md` Part II repeats everything on every IDX ticker Yahoo returns
(838 symbols, 2.59M bars), after dropping rows below Rp50 and under Rp1bn/day of
turnover — **71% of the raw run**. Three results changed the picture:

| | |
|---|---|
| **The median IDX stock loses money** | Under half of all 60-day windows are profitable; the top 1% of observations supply *more than 100%* of total return at 5–20 days. Buying the average name is the trap, not the safe option. |
| **The composite decayed; one component did not** | Distance from the 52-week high (20d IC **+0.066, t=10.2**) beats the blend it sits inside, and is *strongest* in 2021–26 (60d IC +0.103, t=13.9) while the composite hit **zero** in 2018–20. |
| **Training to optimum made it worse** | Re-choosing the weights every 2 years out-of-sample returned **+4.97%**/60d against **+8.6–9.5%** for simply fixing a trend profile. No candidate won more than 3 of 9 folds. |

Survivorship bit exactly as predicted: the composite's holdout 60d IC fell from
**+0.0462 on 66 names to +0.0217 on 724**. More than half the apparent edge was
the universe, not the signal. What survives is crude and robust — the trend
family beat the equal-weight universe in **23 of 24** horizon/size combinations,
and the contrarian profile lost in **all 8**, turning 17 years into +9% total.

**Treat the honest expectation as high-single-digit CAGR, not 35–40%.**

**But: only 6 of 10 holdout years were positive** (2024: −6.65%), and **~8 points
of the headline CAGR is survivorship bias**, not skill — the equal-weight universe
alone beat IHSG by +7.96% CAGR purely from being built out of today's constituent
lists. Full analysis: **`docs/FINDINGS.md`**.

The broker-flow half — the actual thesis — remains untested because the data is
paywalled. `--profile momentum_plus_flow` is the experiment to run when you
connect a real source.

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

### Three horizons, very different evidence

| Horizon | Command | Hold | Measured edge | Verdict |
|---|---|---|---|---|
| **Day** | `idxbot daytrade` | hours | straddles zero | ⚠️ not established |
| **Swing** | `idxbot screen` | 20 days | IC +0.031 (t=3.08) | real but thin after costs |
| **Long** | `idxbot invest` | 60 days | IC +0.046 (t=4.92) | ✅ validated on holdout |

The slowest horizon has the best evidence. `idxbot invest --horizons` prints this
comparison. Day-trading detail: **`docs/DAYTRADE.md`**.

**Day trading, briefly:** a random IDX stock-day touches +5% only **6.9%** of the
time. The burst setup (volume >8× normal, up >7%, at a 20-day high) lifts that to
**38.7%** — but fires ~17×/year on 66 names. Buying at the open loses (−0.62%/trade,
measured on 5-minute bars); the opening-range-breakout filter refused 9 of 12
signals and turned it slightly positive — on n=3, which is not evidence.

| Command | What it does |
|---|---|
| `verify` | Acceptance test: can this broker data support the analysis? Run it first |
| `reverse` | Institutional plan: who leads, do they coordinate, when to join |
| `daytrade` | Intraday momentum scan + timed entry/exit execution plan |
| `invest` | Long-horizon basket from the validated 60-day momentum score |
| `screen` | Rank a universe by score, with evidence for each hit |
| `evaluate` | Cross-sectional rank IC, quantile spreads, per-component diagnosis, train/holdout |
| `analyze TICKER` | Deep dive: score breakdown, Wyckoff phase, broker positions, campaigns |
| `plan` | Executable plan — entry band, stop, targets, lot-rounded size, R:R |
| `playbook` | Reverse-engineer each broker's entry/exit behaviour across a universe |
| `backtest` | Sweep history: does the score actually predict forward returns? |
| `walkforward` | Choose the weighting on past data only, score it on unseen years — and check whether choosing beat not choosing |
| `dashboard` | Offline self-contained HTML report |
| `live` | Reconstruct broker summary from a running-trade stream |
| `pine [TICKER]` | Pine Script source, or plan levels as paste-ready Pine inputs |
| `watchlist` | TradingView-importable watchlist, grouped by signal level |
| `brokers` | Exchange member registry |
| `data` | What history is actually retrievable per ticker |

Everything reads `config/config.yaml` — no thresholds, weights, broker codes or
universes are hard-coded.

### Weight profiles

`--profile` selects a hypothesis. Each is a weight set in `config.yaml`:

| Profile | Components | Status |
|---|---|---|
| `momentum` *(default)* | 12-1 momentum, relative strength, trend persistence, near 52w high | validated out-of-sample; best at 20-day holds |
| `near_high` | distance from the 52-week high, alone | strongest single component and the only one that has not decayed; best at 60-day holds |
| `accumulation` | broker inventory, stealth, concentration, smart-vs-dumb, Wyckoff, volume dry-up | price-only half refuted; broker half untested |
| `momentum_plus_flow` | trend + institutional flow | **untested — needs real broker data** |

`near_high` is deliberately *not* the default. It won the 60-day test and lost
the 20-day one, and picking the per-horizon winner after seeing both results is
the selection bias this repo spends most of its effort avoiding. Choose by
holding period, not by which number is larger.

Evidence for each component only appears in the output if the active profile
actually weights it, so a flag never reads as support for a score it did not
contribute to.

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
  evaluate.py     cross-sectional IC, quantile spreads, train/test split
  daytrade.py     intraday burst scanner, ORB entry rule, day plans
  invest.py       long-horizon portfolio construction
  portfolio.py    long-only simulation vs universe and IHSG
tests/            171 tests
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
