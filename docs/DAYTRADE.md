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
