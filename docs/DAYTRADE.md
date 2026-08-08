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
