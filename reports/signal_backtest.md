# H42 — replaying the daily scan through 26 years: does the target get hit?

*116,754 signal rows, 706 names, 2000-07-27 → 2026-08-28, 252-session horizon.
2,793 censored (the window runs off the name's life) and excluded from every
rate. Code: `scripts/signal_backtest.py`. Raw: `reports/signal_backtest.txt`,
`reports/signal_backtest.csv.gz`.*

Every number `scripts/daily_signal.py` prints — the target, the stop,
`P(target first)`, the `EV` it ranks on — comes from a **fitted law**. Nothing
in this repo has ever checked whether the rows the scanner *selects* behave the
way it says they will. This replays the identical scan at every historical bar
and walks each signal forward a year.

**The replay is legitimate and it is tested, not asserted.** Every input is
causal — EMAs and HMAs by construction, the flip price solved from trailing
WMAs, the swing levels gated on their *confirmation* bar rather than the pivot
bar, vol and turnover trailing. `tests/test_signal_backtest.py` truncates the
panel to a past date, runs the **live scanner** on it, and demands the same
row back, at three different distances into the past.

**Still in-sample, said before the numbers.** The probability laws were fitted
on this data, so the calibration arm is a consistency check and nothing more.
What is *not* contaminated is the realised hit rate and the realised return —
those are facts about the price path, and they are what the question asks.

---

## 1. Does it hit the target? Yes — 68.5% of the time, and that is the trap

| | |
|---|---|
| **P(target touched within a year)** | **68.5%** |
| **P(target arrives BEFORE the stop)** | **59.1%** |
| mean return per signal, net of cost | **+0.08%** |
| median | +5.89% |
| mean bars held | 54 |
| **the same name simply held for the year** | **+12.97%** |

So the honest answer to "does it reliably hit the target" is **yes, and it does
not matter.** The target arrives first three times in five and the average
signal returns eight basis points, because the hit rate is bought with the
payoff. The mechanism is entirely visible once the population is split by
reward-to-risk — which is the thing the list varies most:

| reward:risk | n | **P(target first)** | predicted | **mean return** |
|---|---|---|---|---|
| < 0.75 | 62,220 | **74.2%** | 70.9% | **−0.20%** |
| 0.75 – 1.5 | 24,353 | 50.4% | 49.9% | −0.12% |
| 1.5 – 2.5 | 12,923 | 39.1% | 38.0% | +0.35% |
| 2.5 – 4 | 7,803 | 32.2% | 29.5% | +0.87% |
| **> 4** | 6,661 | **24.7%** | 19.8% | **+2.01%** |

**Fifty-five percent of all signals carry a near target and a far stop.** Those
are the ones that "reliably hit the target" — 74% of the time — and they are the
only bucket that loses money. Perfectly monotone: every step up in payoff costs
hit rate, and the money is entirely at the end nobody would call reliable.

*This is H40's S4 finding arriving from a completely different direction: a high
win rate is something you buy, and the price is the right tail.*

---

## 2. B1 — the probability is calibrated, and its discrimination is geometry

| predicted decile | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| predicted | 0.232 | 0.349 | 0.430 | 0.498 | 0.558 | 0.612 | 0.663 | 0.714 | 0.765 | 0.834 |
| **realised** | 0.275 | 0.361 | 0.438 | 0.503 | 0.566 | 0.630 | 0.685 | 0.748 | 0.812 | 0.893 |
| gap | +.043 | +.011 | +.008 | +.005 | +.008 | +.018 | +.022 | +.034 | +.046 | +.059 |

Pooled: **predicted 0.566 against realised 0.591.** The law is honest to about
two and a half points, and the realised curve is slightly *steeper* than the
fitted one — it is under-confident at both ends, not over-confident, which is
the safe direction to be wrong in.

**B1 CONFIRMED, in the sense that matters.** Realised running 0.275 → 0.893
across the deciles looks like strong discrimination and is not: the deciles
*are* the distance ratio, which the user chose when they picked a target. That
is precisely what H36 measured as AUC 0.51 — the race law encodes the geometry
correctly and knows nothing else. **It prices a decision you have already made
and cannot make one for you.**

---

## 3. B3 — the EV column does work, weakly, and it was the wrong prediction

I registered that EV would not predict realised return, citing H35's 0-of-30
brackets and H38's 0-of-25 placements, and flagged it as the closest thing here
to a predicted-null control.

| | n | predicted EV | **realised** |
|---|---|---|---|
| EV > 0 | 12,932 | +1.55% | **+1.93%** |
| EV ≤ 0 | 101,029 | −1.95% | −0.15% |

And it separates in **both halves** — early +4.28% against +0.26%, late −0.27%
against −0.57%. **B3 FAILED: the expectancy column carries real information.**

Two things stop that being good news. It is confined to the **top decile** —
bins 0 through 8 all sit within a few tens of basis points of zero, and only
bin 9 (+2.21%) moves. And the `hold` column of that same table runs from
**+21.55% in the worst EV bin down to +11.07% in the best**: the scanner
systematically prefers names that, held, would have gone up *less*.

---

## 4. B4 — the control, and it is worse than that

The comparison that has decided every result in this repo: the **same date**,
the **same bracket distances**, a **randomly chosen eligible name**.

| arm | P(target first) | mean return | ann. log | hold 252 |
|---|---|---|---|---|
| scanner signals | **0.591** | **+0.08%** | −10.5% | +12.97% |
| **random name, same bracket** | **0.591** | **+0.50%** | −7.0% | +12.03% |

Identical hit rate to three decimal places, and the random arm makes **six times
the return**. Paired signal-by-signal with the control drawn for it, resampling
whole **(ticker, year)** blocks:

| | n | picks − random | 95% CI | early | late |
|---|---|---|---|---|---|
| all signals | 112,190 | **−0.43%** | [−0.88%, +0.05%] | −0.49% | −0.37% |
| EV > 0 only | 12,812 | −0.48% | [−2.29%, +1.57%] | −0.23% | −0.73% |
| top 10 by EV | 2,370 | −0.42% | [−1.56%, +0.75%] | −0.41% | −0.43% |

**Negative in every arm and in both halves of every arm.** The interval clips
zero on the "all signals" row, so *"worse than random"* is not established at
95% — but *"no better than random"* is, comfortably, and the point estimate
never once comes out positive. **B4 CONFIRMED.**

---

## 5. The number that ends it

Bracketing the position — putting the scanner's target and stop on it — against
simply owning the same name for the year, paired and block-bootstrapped:

> **−13.06% a year, 95% CI [−16.25%, −10.22%].**

The `edge` column is negative in **all ten** EV deciles, from −20.80% in the
worst to **−8.85% in the best**. There is no cell of this study in which putting
a target and a stop on an IDX name beat holding it.

---

## 6. The bug, and it is the fourth of its kind in four studies

The first run printed an annualised return of **1.7 × 10²⁸**. Annualising each
trade's arithmetic return *separately* and averaging means a +50% trade held one
bar contributes 1.5²⁵², and a single row swamps a hundred thousand others. The
fix is to annualise the **mean log** and use the mean holding period.

That makes four consecutive studies in which an impossible printed value caught
a definitional error — H39's "beat buy-and-hold on 0.0% of trades", H40's
exactly-0.0% win rate and its `nan` CAGR, the scanner's green-ribbon count
disagreeing with itself, and this. **A statistic that cannot occur remains the
cheapest bug detector available.**

A second trap was avoided rather than committed, and is worth recording because
it would have printed a beautiful number: a **duration-matched hold is not a
benchmark here.** The bracket exits at the *close* of its exit bar, and so does
a hold of that many bars, so the difference is identically zero by construction
— H39's shape exactly. `tests/test_signal_backtest.py` pins it.

---

## What this licenses

- **"Does it hit the target?" is the wrong question, and answering it yes is
  how the list would mislead.** 68.5% touch it, 59.1% before the stop, and the
  average signal returns +0.08%.
- **The quoted probability is trustworthy and useless.** Calibrated to 2.5
  points; its entire discrimination is the ratio of the two distances the user
  chose.
- **The EV column is real and small** — +1.93% against −0.15%, replicating in
  both halves — and it is confined to the top decile and still loses to holding.
- **The name selection is worth nothing or less.** Same bracket on a random
  eligible IDX name matches the hit rate exactly and beats the return.
- **The bracket itself costs 13 points a year.** That is the finding, and it is
  the same wall this repo has now hit from ten directions.
