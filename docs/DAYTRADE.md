# Day trading IDX for 5% a session

What the data says about the goal, and the plan that follows from it.

---

## 1. How often a 5% day even happens

Measured on **296,344 real IDX stock-days, 2001–2026, 66 names**:

| From the open, price reaches | % of all stock-days |
|---|---|
| +3% | 16.9% |
| **+5%** | **6.9%** |
| +7% | 3.3% |
| +10% | 1.4% |

A random name on a random day touches +5% about once every fourteen sessions.
Any strategy targeting 5% a day is therefore **entirely a selection problem** —
the move is common enough to exist, rare enough that picking is everything.

## 2. The setup that shifts the odds

Conditioning on the *previous* day's behaviour, then entering the next session:

| Condition on day T | n | P(+5% on T+1) | P(+10%) | median dip |
|---|---|---|---|---|
| any day (baseline) | 296,344 | 6.9% | 1.4% | −1.2% |
| volume > 3× normal | 27,758 | 13.2% | 4.1% | −1.4% |
| volume > 5× and up >5% | 3,269 | 28.9% | 11.5% | −2.5% |
| volume > 5×, up >5%, ATR >4% | 1,792 | 36.3% | 15.7% | −3.3% |
| **volume > 8×, up >7%, at 20-day high** | **416** | **38.7%** | **18.5%** | **−2.9%** |

The last row is the **burst** setup. It raises the odds of a 5% day by **5.6×**,
and the median adverse excursion is only −2.9%, which is what makes a −3% stop
viable.

Your DSSA example is exactly this pattern: on 2026-08-06 it opened 870 and hit
975 (**+12.1%**) on **6.5× normal volume**. Same signature on 2026-07-21:
+15.3% on 10× volume.

**But it fires ~17 times a year across 66 names.** Every loosening of the filter
tested worse. The lever for more trades is a **wider universe** — scan all ~900
IDX names instead of 66 — not a looser rule.

## 3. Entry is worth more than the setup

Two entry rules, same signals, measured on real 5-minute bars:

| Entry rule | Signals | Traded | Mean net | Win rate |
|---|---|---|---|---|
| Buy at the next open | 12 | 12 | **−0.62%** | 36% |
| **Opening-range breakout + volume pace** | 12 | **3** | **+0.15%** | 33% |

The ORB rule **refused 9 of 12 trades**, and the refusals are where the losses
were. This is the single most important line in this document: *most of the
edge is in not trading.*

⚠️ **n = 3.** That is not evidence. It is consistent with the design and nothing
more. Yahoo serves only ~60 days of 5-minute history, so a real intraday
backtest is not possible from free data.

### Why daily bars cannot settle this

With a +5% target and −3% stop, **13.5% of burst days touch both levels**. A
daily bar records that both happened, never which came first. That single
ambiguity spans the entire result:

- assume the stop always won → **−0.56%** per trade
- assume the target always won → **+0.52%** per trade

Walking real 5-minute bars, the target won **38%** of those races — close to the
pessimistic end. That is why the naive entry loses.

---

## 4. The plan

```bash
idxbot daytrade --universe all --plans 3        # after the close, for tomorrow
idxbot daytrade --study                          # resolve paths on 5m bars
```

**Times are WIB.**

| When | Do |
|---|---|
| **09:00–09:30** | Nothing. Let the opening range form; mark its high and low. Skip the day if it gaps up more than 3% — the stop would sit under the open. |
| **09:30–11:30** | Entry window. Buy **only** when all three hold: (1) a 5-min bar *closes* above the opening-range high, (2) price is above session VWAP, (3) volume pace ≥ 1.5× normal for the time of day. **No trigger by 11:30 → no trade.** |
| **On fill** | Stop = the **higher** of the opening-range low or −3% from entry. Place it as a real order. Never widen it. |
| **+3%** | Sell half. Move the stop to fee-adjusted breakeven. The trade can no longer lose. |
| **+5%** | Sell the rest. This is the goal — take it. |
| **15:00** | If +3% has not filled and price is below VWAP, exit. The burst failed. |
| **15:45** | **Flat, unconditionally.** Overnight gap risk is not in any number here. |

Sizing: 0.5% of equity at risk per trade (half the swing risk, because the odds
are worse and the frequency higher). Every price is snapped to the IDX tick grid
and checked against the ARA band, so a target that cannot print today is flagged.

### The broker-flow trigger

`daytrade.broker_trigger()` fires when bulge desks (AK, BK, …) dominate the buy
side of the live running trade — the DSSA pattern. Whether that confirmation
*improves* the burst's expectancy is **unmeasured**, because it needs live broker
data (`docs/LIVE_DATA.md`). It is wired in and ready to test. Until then, treat it
as a tiebreaker between candidates, not as an edge.

---

## 5. The honest summary

- The setup is **real**: 38.7% vs 6.9% is a large, robust conditioning effect on
  a 25-year sample.
- The entry filter is **probably** where the money is, and is **unproven** at n=3.
- Net expectancy for the naive version is **negative**. For the filtered version
  it is **unknown but plausibly slightly positive**.
- It fires **~17 times a year** on this universe. This is not a daily income.

**Compare with the other horizons in this repo:**

| Horizon | Hold | Measured edge | Verdict |
|---|---|---|---|
| **day** | hours | straddles zero | not established |
| swing | 20 days | IC +0.031 (t=3.08) | real but thin after costs |
| **long** | 60 days | IC +0.046 (t=4.92) | **validated on a holdout** |

The slowest horizon has by far the best evidence. That is the opposite of most
people's instinct, and it is what 25 years of IDX data says. If the goal is
compounding capital rather than the activity of trading, `idxbot invest` rests on
much firmer ground than `idxbot daytrade`.

Trade the day setup small, in a wide universe, with the ORB filter, and treat it
as an experiment you are funding — not as a salary.

---

## 6. Can a 1–2 day trade make +5% with an 80% win rate?

No. This section exists because it is the most natural thing to want, and
because "no" is worth much more when it comes with the measurement.

Scanned every liquid (ticker, day) on IDX — **761,137 observations, 792 tickers,
2000–2026** — entering at the next bar's open (`scripts/short_horizon_scan.py`).

### The base rate

How often does a stock touch the target *at all*?

| hold | +3% | +5% | +8% |
|---|---|---|---|
| 1 day | 22.3% | **10.6%** | 4.5% |
| 2 day | 34.7% | **19.3%** | 9.4% |
| 3 day | 42.6% | 25.9% | 13.8% |

Reaching an 80% hit rate from 19.3% needs a **4.1× lift**. Nothing lifts an
equity base rate fourfold. The strongest conditioning found comes close on the
touch rate and collapses on the tradeable one:

| cut | n | touches +5% | lift | **actually wins** |
|---|---|---|---|---|
| ATR>15% & yesterday >+15% | 224 | 73.7% | 3.8× | **33%** |
| ATR>10% & yesterday >+20% | 591 | 70.7% | 3.7× | **27%** |
| ATR top decile | 75,664 | 47.2% | 2.4× | 41% |

The gap between columns 3 and 5 is the whole trap. These names touch +5% often
*because* they are violent, and violence is symmetric: they break the stop first
at least as often. A 74% "hit rate" that pays 33% of the time is not a 74%
strategy.

### Why no rule can fix it

Before any strategy, the arithmetic:

| hold | gross drift | round-trip cost | net | cost vs edge |
|---|---|---|---|---|
| 1 day | **−0.085%** | 0.40% | −0.485% | 5× |
| 2 day | **−0.041%** | 0.40% | −0.441% | 10× |
| 3 day | −0.004% | 0.40% | −0.404% | 109× |

At these horizons the average IDX stock **drifts down**, and you pay 0.40% for
the privilege of holding it. There is no edge to harvest, only a cost to pay.
Compare the 60-day horizon, where the gross edge is +8–9% and the same 0.40%
is a rounding error.

And the two requirements pull against each other. Raising the win rate means
widening the stop, which deepens the losses:

| target | stop | win rate | expectancy |
|---|---|---|---|
| +5% | −10% | 53.7% | −1.65% |
| +5% | −20% | 67.1% | −0.42% |
| +2% | −20% | 77.6% | −1.24% |
| +1% | −20% | **82.1%** | **−1.54%** |

An 80% win rate *is* reachable in 2 days — at a +1% target with a −20% stop,
losing 1.54% per trade. That is the picking-up-pennies structure in its purest
form, and it is the only way the two constraints meet.

### The one short setup that is genuinely positive

Of 35 conditions searched, four had positive 2-day net expectancy, and only one
survives scrutiny: **5-day return > +30%**, held two days *flat*.

| era | n | net expectancy |
|---|---|---|
| 2000–12 | 1,120 | +0.56% |
| 2013–19 | 1,054 | +1.51% |
| 2020–26 | 6,099 | +0.39% |

Positive in all three eras. But note what it is not: **+0.4% to +1.5%, not +5%**,
and it only works with *no target and no stop* — every barrier version of it is
negative, for the same reason as Part III. 11% of entries gap more than +5%
above the prior close, so you are chasing.

(A cut on "gap down >2%" showed +0.73%, but it conditions on the fill price
itself — you cannot know the open before it opens. Discarded, not reported.)

### What to do instead: the horizon frontier

The 88% rule from FINDINGS Part III, run at progressively shorter maximum holds:

| max hold | win rate | expectancy | PF | **avg days held** | annualised |
|---|---|---|---|---|---|
| 2 days | 56% | −0.06% | 0.95 | 2.0 | −8.2% |
| 3 days | 61% | +0.05% | 1.03 | 2.8 | +4.1% |
| 5 days | 69% | +0.27% | 1.17 | 4.1 | +16.6% |
| 10 days | 77% | +0.53% | 1.31 | 6.6 | +20.4% |
| **20 days** | **85%** | **+1.27%** | 1.82 | **9.8** | **+32.7%** |
| 40 days | 88% | +1.75% | 2.11 | 13.9 | +31.8% |
| 60 days | 86% | +2.39% | 2.23 | 17.9 | +33.5% |

Monotonic, and it crosses 80% at a **20-day cap** — where the average trade
actually closes in **9.8 days**, not 20. Annualised it matches the 60-day
version (32.7% vs 33.5%) in **half the holding time**, which makes it the right
choice for anyone who wants speed. Two days is the one row that loses money.

**So: ~2 weeks is the floor for an 80% win rate on IDX.** Below it, costs exceed
the gross drift and no amount of selection closes the gap.

---

## 7. Intraday specifically: +5% at an 80% win rate is above the ceiling

§6 covered 1–2 day holds. This covers the strictly harder case — in and out
within one session — and the answer is not "very difficult". It is
**arithmetically unavailable**, and that can be shown without proposing a single
strategy.

### The bound, computed from the data alone

A same-session trade cannot return more than the session's own range. So the
ceiling follows from daily bars directly (`scripts/intraday_ceiling.py`), over
**761,420 liquid sessions, 792 tickers, 2000–2026**:

| target | reachable from the **open** | reachable from the **session low** |
|---|---|---|
| +2% | 34.6% | 77.3% |
| +3% | 22.5% | 57.2% |
| **+5%** | **10.8%** | **29.4%** |
| +8% | 4.7% | 12.2% |

The right column is what a trader who buys the **exact low of every session**
would achieve. That is perfect foresight — the low is only identifiable after
the close — so it bounds *every* intraday system that will ever be written. At
+5% it is **29.4%**. An 80% win rate is not 2.4× harder than the state of the
art; it is above a ceiling no skill can reach.

Filtering to the most violent sessions raises both columns and does not rescue
the realistic one:

| cut | n | +5% from open | +5% from low |
|---|---|---|---|
| all sessions | 761,420 | 10.8% | 29.4% |
| ATR top decile | 75,692 | 34.1% | 80.6% |
| ATR top 1% & volume >3× | 1,367 | **42.3%** | 87.6% |

The last row is the only place perfect foresight clears 80% — and it is 1,367
sessions out of 761,420, about **two opportunities a month across the entire
exchange**, still requiring you to buy the exact low. The implementable number
is 42.3%.

### Confirmed on real 5-minute bars

The bound above is inferred from daily OHLC. So it was re-run on genuine
intraday data — **245,199 five-minute bars, 70 liquid names, 3,818 sessions** —
with a real strategy: enter on the first 5m close above the 30-minute opening
range, exit on target/stop within the session, stop checked first.

| target | stop | trades | win rate | hit target | expectancy | PF |
|---|---|---|---|---|---|---|
| +5% | −3% | 1,614 | 30% | 10% | **−0.60%** | 0.55 |
| +5% | −5% | 1,614 | 33% | 12% | −0.56% | 0.59 |
| +3% | −5% | 1,614 | 37% | 22% | −0.58% | 0.55 |
| +2% | −5% | 1,614 | 42% | 33% | −0.55% | 0.48 |

Every configuration loses. The same sessions put the perfect-foresight ceiling
at 52.5% for +5% — higher than the 25-year figure only because these are the 70
most liquid names in a recent volatile stretch, and still nowhere near 80%.

### Why the two requirements are mutually exclusive

Shrinking the target raises the win rate, exactly as expected — until it doesn't:

| target | win rate | net expectancy |
|---|---|---|
| +2.0% | 42% | −0.60% |
| +1.0% | 57% | −0.53% |
| +0.5% | **70%** | −0.51% |
| +0.3% | **0%** | −0.53% |

The collapse from 70% to **zero** is the whole argument in one line. A 0.3%
target is smaller than the 0.40% round trip, so every trade that "hits its
target" still books a loss. The win rate can only rise while the target stays
above costs, and it runs out of room at ~70%.

Underneath it all:

    open-to-close drift  -0.42%      (the average IDX session drifts DOWN)
    round-trip cost      -0.40%
    ------------------------------
    before any strategy  -0.82%

You are paying 0.40% to enter a game with negative expected value. Selection
cannot fix a starting position that bad — it can only choose which −0.82% you
take.

### The honest bottom line

| | |
|---|---|
| **+5% intraday at 80% win rate** | **impossible** — ceiling is 29.4% (25y) / 52.5% (5m), even with perfect entry |
| Best real intraday win rate found | ~70%, at a +0.5% target, expectancy −0.51% |
| Best intraday expectancy found | −0.55%, i.e. every variant loses |
| Shortest hold that reaches 80% **and** makes money | **20-day cap, ~9.8 days actual** — 85% win, +1.27%/trade, 32.7% annualised |

Nothing intraday in this repo is profitable, and this section exists so that
stays visible rather than being rediscovered with real money.

---

## 8. Pushing harder: dip entries, one real finding, and a bug worth the trip

§7 tested only *breakout* entries, which buy strength. That was an incomplete
search, and the gap mattered: the ceiling table shows +5% is reachable from the
session low far more often than from the open, so entry timing is exactly where
the remaining room was. This section runs that search on **739,320 five-minute
bars, 216 names, 11,838 sessions**.

### Dip entries genuinely beat breakouts

Target +5%, stop −5%, all sessions:

| entry rule | fills | fill rate | win rate | expectancy |
|---|---|---|---|---|
| ORB breakout | 4,662 | 39% | 33% | −0.59% |
| limit −1% below open | 8,920 | 75% | 41% | −0.61% |
| limit −2% below open | 6,349 | 54% | 42% | −0.56% |
| limit −3% below open | 4,581 | 39% | 45% | −0.50% |
| VWAP dip | 11,747 | 99% | 37% | −0.68% |

Monotonic in depth, on both win rate and expectancy. Pushed further, into names
whose *prior* ten sessions were in the top volatility decile (trailing, no
look-ahead), a −7% limit with a +5% target and −10% stop reaches:

**61% win rate, +0.58% per trade, profit factor 1.35 — the first positive
intraday expectancy in this repo.**

Split in time, the second half is the stronger one (67% win, +1.03%, t=3.45
against 57%, +0.23%, t=0.68). That is not the signature of noise. But it is
**378 trades over three months in one regime, selected as the best of 24 grid
cells**, and 61% is not 80%. It is a lead worth more data, not capital.

### The bug this nearly became

Validating that finding on 25 years of daily bars produced a **96% win rate and
profit factor 9.44**. It was entirely false, and the mechanism is worth stating
because it is easy to reproduce by accident.

A daily bar reports a high and a low but not **which came first**. Simulating
"buy on a limit 7% below the open, then check whether the session high reached
my target" answers yes almost always — because the high had already happened
before the limit filled:

| limit depth | sessions where the high preceded the fill |
|---|---|
| −3% | 58% |
| −5% | 74% |
| −7% | **79%** |

The simulation credits the trade with a rally it could not have been in. Worse,
it scales the wrong way: a deeper limit fills nearer the session low, so more of
the day's range sits before the fill, so the fabricated edge *grows* exactly as
the rule starts to look irresistible.

Identical rule, identical sessions, only the time ordering differing:

| depth | correct (5m, ordered) | broken (daily, unordered) |
|---|---|---|
| −5% | 52% win | 93% win |
| −7% | **55% win** | **96% win** |

The 5-minute cross-check is the only reason this was caught. `tests/
test_ordering_trap.py` now pins the rule: once an entry is triggered by a
**price level** rather than a bar's open or close, only bars strictly after the
trigger may resolve the outcome — and daily bars cannot support that at all.

This does not touch the 20-day and 60-day results in `FINDINGS.md`. Those enter
at the **next bar's open**, which is unambiguously that bar's first price, so no
ordering question arises.

### Where this leaves the original question

| | |
|---|---|
| +5% intraday at 80% win | still not reached — best honest result is **61%** |
| Best intraday expectancy | **+0.58%/trade**, hi-vol names, −7% limit, n=378, unestablished |
| Improvement from harder searching | real: 33% → 61% win, −0.59% → +0.58% |
| Most valuable output | the ordering trap, now permanently tested |


---

## 9. Goal: 80% win rate at +5% intraday — the exhaustive attempt

Directive: keep going until 80% at +5% intraday. This section is the record of
that attempt, including the best rule it produced and why the target itself is
unreachable.

### The search

The earlier sections were limited by sample: only ~60 days of 5-minute history.
Yahoo serves **730 days of hourly bars**, so the search was rerun on
**168,586 sessions, 251 names, 2023-07 to 2026-08** — 46× the 5-minute sample,
and still time-ordered, which is what the daily-bar approach lacked.

**120 base configurations** — every entry depth from buy-at-open to a −10% limit,
every stop from −3% to −20%, unconditional and in the top 10% / 3% / 1% of
trailing volatility — plus ~20 stacked filter combinations.

| best of 120, target +5% | trades | win | expectancy | PF |
|---|---|---|---|---|
| −10% limit, −20% stop, top 3% volatility | 774 | **61.1%** | +0.63% | 1.36 |
| −10% limit, −20% stop, all sessions | 3,982 | 60.7% | +0.67% | 1.42 |

61.1%. The 5-minute dataset — three months, different years, twelve times finer
resolution — independently gave **61%**. Two samples that share almost nothing
landing on the same number is the strongest evidence in this document that 61%
is where the unfiltered rule actually sits.

### What lifted it, and how far

The dip that reverts is the **idiosyncratic** one — a stock down 10% while the
index is *up*. That is forced or panicked selling in a single name rather than a
market event, and market-wide selloffs do not bounce the same way:

| filter added | trades | win | expectancy | PF |
|---|---|---|---|---|
| none | 3,982 | 60.7% | +0.67% | 1.42 |
| index up at fill | 1,046 | 63.9% | +0.85% | 1.51 |
| index up >0.5% at fill | 350 | 66.0% | +1.16% | 1.80 |
| + prior 20-session trend positive | 200 | 68.0% | +1.12% | 1.70 |
| **+ dip within the first 3 hours** | **138** | **68.8%** | **+1.25%** | **1.71** |

Then it stops. Deeper dips (−12%, −15%), stricter index thresholds, more
recovery time — each either fails to improve or runs the sample below 50 trades.
The win rate asymptotes at **~70%** and the sample collapses before 80%.

### Why 80% at +5% cannot be reached

Lowering the target does not get there either:

| target | trades | win rate | expectancy |
|---|---|---|---|
| +5% | 138 | 68.8% | **+1.25%** |
| +4% | 138 | 70.3% | +0.78% |
| +3% | 138 | 71.7% | +0.25% |
| +2% | 138 | 72.5% | −0.40% |
| +1% | 138 | **74.6%** | **−1.01%** |

**74.6% is the maximum at any target**, and by then the trade loses money. The
two requirements move in opposite directions and never meet: the win rate climbs
only as the target shrinks, and expectancy turns negative at +2% — below the
point where the win rate would still need another six points.

Combined with §7's bound — even buying the *exact session low* reaches +5% in
only 29.4% of sessions across 25 years — the conclusion is arithmetic rather
than a failure of searching.

### What the search did produce

The first positive-expectancy intraday rule in this repo, shipped as
`src/idxbot/dipreversal.py`:

```
entry     limit at -10% from the session open, first three hours only
filter    index up >0.5% at the moment of fill, prior 20-session trend positive
target    +5% from the fill
stop      -20% from the fill
exit      the close, unconditionally
```

**138 trades, 68.8% win rate, +1.25% per trade, PF 1.71, t = 2.73.** Split
chronologically the second half is stronger (73.9%, +1.46%, t=2.26).

Read it with the caveats attached. **Roughly 140 configurations were compared to
arrive at 138 trades** — that is a great deal of searching for a thin sample, and
the holdout being stronger is reassuring rather than conclusive. The limit fills
on about 2% of sessions, preferentially on days that keep falling, so the index
filter is carrying heavy weight on little data. It is an experiment worth funding
in small size, not an established edge.

**The goal was not reached. 68.8% at +5% was, and it is real money rather than a
number produced by relaxing a definition.**
