# The trading plan

The strategy this repo implements, why each rule exists, and — importantly —
which parts are validated and which are not.

---

## 0. What is actually verified

Read this before the strategy. It determines how much weight each rule carries.

| Claim | Status |
|---|---|
| IDX price/volume history is real and deep | ✅ **Verified** — 169,553 daily bars across 30 names, up to 26.2 years each; IHSG from 1990-04-06 |
| Ledger, cost-basis and campaign maths are correct | ✅ **Verified** — 44 unit tests, hand-computed expectations |
| Tick / lot / ARA-ARB / fee mechanics are IDX-correct | ✅ **Verified** — unit tested |
| The price-only accumulation score predicts forward returns | ❌ **REFUTED — inverted on 56,732 real observations. See `docs/FINDINGS.md`** |
| **How J.P. Morgan, UBS and CLSA actually enter and take profit** | ❌ **NOT verified — no real broker-summary data was available** |

> **Before reading further:** the price/volume half of this strategy was tested
> on 25 years of real IDX data and **failed** — high scores underperformed low
> scores at every horizon, monotonically, in both sample halves. The rules below
> describe a coherent system, but the price-only evidence contradicts its core
> premise. Treat sections 2–6 as the specification of a hypothesis that is still
> waiting on broker-level data to be tested properly, not as a validated edge.
> Details: `docs/FINDINGS.md`.

That last row is the honest limit of this work, and it is the row you care most
about. I could not obtain real IDX broker summary: `idx.co.id` is behind a WAF
(403), Stockbit requires an authenticated app session, and GoAPI's broker-summary
route exists but is gated on a paid key. There is no free public source.

So the repo ships a **simulator** to make the pipeline runnable. The simulator
*assumes* institutions buy weakness and retail chases momentum. If you run the
playbook against it, it will report that institutions buy weakness — because that
was typed into `src/idxbot/data/synthetic.py` by hand. **That is circular and
proves nothing about the real market.**

The machinery for the real answer is built, tested, and waiting for real data.
Connect a source per `docs/LIVE_DATA.md` and the same commands produce a genuine
answer. Until then, treat every broker-level number as a format demonstration.

---

## 1. The thesis

Large positions cannot be built quickly without moving the price. A desk that
wants a meaningful position in an IDX name has to absorb supply patiently, over
weeks, without advertising itself — and that leaves a measurable footprint:

- inventory rising while price goes sideways (**absorption**)
- a small number of members taking most of the net supply (**concentration**)
- institutional members buying while retail members sell (**divergence**)
- volume drying up and the range compressing as sellers are exhausted
- volume flow rising faster than price (**OBV divergence**)

The trade is to find that footprint before the markup and position alongside it,
with risk defined by the same structure the operator is defending.

**What this is not.** It is not front-running, insider information, or copying a
signal you shouldn't have. Broker summary is published exchange data. And a
broker code is not a firm's opinion: `BK` is *every* client routing through
J.P. Morgan's membership — a pension fund, a hedge fund, an index tracker
rebalancing, a delta hedge against a warrant. "J.P. Morgan bought" means flow
crossed their membership. Strong evidence of institutional intent; not a
confession of it.

---

## 2. Signal generation

`idxbot screen` scores every name 0–100 from nine weighted components. Four need
broker data; five need only price and volume. Without broker summary the engine
drops the first four and renormalises — that is **price-only mode**, and it is
labelled on every output, because a price-only 70 and a broker-confirmed 70 are
not the same claim.

| Component | Weight | Needs broker data | What it detects |
|---|---|---|---|
| `inventory_zscore` | 0.22 | ✅ | Bulge-desk inventory build vs its own history in that name |
| `stealth` | 0.18 | ✅ | Heavy institutional buying while price barely moves |
| `smart_dumb_divergence` | 0.14 | ✅ | Institutions absorbing while retail distributes |
| `concentration` | 0.12 | ✅ | Few members taking most of the net supply |
| `wyckoff` | 0.15 | — | Phase structure (spring, sign of strength) |
| `obv_divergence` | 0.10 | — | Volume flow outpacing price |
| `volume_dryup` | 0.08 | — | Supply exhaustion in the base |
| `range_compression` | 0.08 | — | Coiling before expansion |
| `relative_strength` | 0.08 | — | Outperforming the IHSG during the base |

Thresholds: **≥78 STRONG**, **≥65 SIGNAL**, **≥50 WATCH**. All in `config/config.yaml`.

### Wyckoff phase gates the entry

Phase determines *whether* to act, not just how much:

| Phase | Meaning | Action |
|---|---|---|
| A | Decline being stopped | Too early. Watch. |
| B | Cause being built | Scale in, small size |
| **C** | **Spring / shakeout** | **Best risk/reward — supply tested and failed** |
| **D** | **Markup starting** | **Enter on pullbacks** |
| E | Markup extended | Too late — this is a chase, and the planner blocks it |

---

## 3. Entry

**Anchor on the lead institutional desk's reconstructed cost basis.** The ledger
rebuilds each member's volume-weighted cost from their daily buy/sell tape. If a
bulge desk accumulated at 5,974 and the stock is at 6,375, buying near their
basis means your risk is aligned with theirs. Buying far above it means you are
paying a premium for the same thesis.

```
entry_low  = min(broker_cost, close) - 0.25 x ATR
entry_high = max(broker_cost + 0.5 x ATR, close)
```

Sizing and reward:risk are measured from the **midpoint** of that band — the
realistic average fill of a scale-in. Measuring from the top overstates risk on
every plan.

Triggers by phase:
- **C** — buy the reclaim: enter once price closes back above the range low after the spring
- **D** — buy pullbacks into the zone while it holds above the breakout level
- **B** — scale in inside the zone; add on a close above the range high on above-average volume
- **none** — no confirmed structure, wait

---

## 4. Risk

**The stop is the tightest candidate that still leaves at least 1 ATR of room:**

```
structural = 60-day base low - 0.5 x ATR
atr_stop   = entry - 2.0 x ATR
stop       = max(candidates that sit at least 1 ATR below entry)
floor      = entry x 0.85          # never risk more than 15%
```

Taking the *widest* candidate — the obvious-looking choice — guarantees an
unworkable reward:risk, because learned targets are often only 5–15%. This was a
real bug during development, caught by running the plan end to end.

**Thesis invalidation is separate from the price stop**, and it matters more:

> The lead broker turns net seller for 3 consecutive sessions, or inventory falls
> 20% from its peak → **exit regardless of price**.

The reason you are in the trade is that a large operator is accumulating. If they
stop, the reason is gone even if price has not hit your stop yet.

**Position size** is risk-first, rounded *down* to whole lots so realised risk is
always at or below budget:

```
lots = floor( min(equity x risk% / (entry - stop), equity x max_position% / entry) / 100 )
```

Defaults: 1% risk per trade, 20% max single-name exposure.

---

## 5. Targets

Targets come from **the anchor broker's own realised markup distribution** — the
25th, 50th and 75th percentiles of how far price ran above their entry VWAP
across their detected campaigns — not from round numbers.

Then a behavioural adjustment: a desk whose `exit_capture` is low habitually
scales out well before the high, so its raw markup distribution overstates what
is actually gettable. Targets are scaled by `max(0.55, capture + 0.25)`.

When the anchor has no usable campaign record the plan says so explicitly
(`configured fallback`) and uses 8% / 15% / 25%. The label always states which
applies — a target labelled as broker-derived when it silently fell back was
another bug caught in testing.

Every target is checked against the **auto-rejection band**: a target needing
several consecutive limit-up days is flagged, because it cannot print this week.

**Reject the plan if reward:risk < 1.8** on the middle target.

---

## 6. Exits

| Exit | Rule |
|---|---|
| T1 | Sell ⅓, move stop to breakeven (`entry x 1.0015 / 0.9975`, fee-adjusted) |
| T2 | Sell ⅓, trail the remainder below the 20-day low |
| T3 | Close the rest, or trail if the broker is still accumulating |
| Stop | Full exit, no averaging down |
| Thesis | Lead broker turns seller → full exit regardless of price |
| Time | `min(1.5 x broker median hold, 120 days)` → exit, capital has an opportunity cost |

---

## 7. The daily routine

The two-hour delay on retail broker summary does not matter for this strategy —
campaigns run for weeks. See `docs/LIVE_DATA.md` §3.

```
17:30–18:00 WIB   broker summary lands → data/broker_summary/
18:00             idxbot screen --universe lq45 --out reports/screen.csv
18:05             idxbot plan --tickers <hits> --pool lq45 --equity 100000000
18:10             idxbot watchlist --screen        # push into TradingView
next session      work the limit orders inside the entry band
weekly            idxbot playbook --universe lq45 --edge
monthly           idxbot backtest --universe all --providers none
```

---

## 8. Position and portfolio limits

- Max 1% equity risk per position
- Max 20% equity in any single name
- Max ~6% total portfolio heat (`idxbot plan` prints it if all TAKEs filled)
- Never average down into a stop
- Skip names whose median daily value traded cannot absorb your size

---

## 9. Honest failure modes

**The signal is co-integrated with beta.** Accumulation scores rise across the
board in a market-wide base. The relative-strength component helps; it does not
eliminate the exposure.

**Broker summary is flow, not holdings.** Inventory is measured relative to the
first day of your data window. If UBS held 200m shares before your window opened,
the ledger starts them at zero and can show negative inventory while they remain
net long. The *changes* and the *cost basis of those changes* carry the signal;
the absolute level does not.

**Campaign segmentation is a model.** The zigzag detrends inventory against a
120-day EWMA to strip out structural custody/index drift, then requires a 25%
retracement. Different parameters find different campaigns. It is a lens, not a
measurement.

**The backtest flatters itself.** Today's index constituents (survivorship),
overlapping forward windows (inflated t-statistics), gross returns unless the
net-cost column is shown. Treat the numbers as a ranking device.

**Small samples.** Per-ticker, a broker may have 3–5 campaigns. Pool across a
universe (`--pool lq45`) before believing any profile.

---

*Educational tooling. Not investment advice. Trade your own risk.*
