# H39 — Hull green + BUY in, Hull red + SELL out, on every IDX name

*891 names, 2,465,286 pre-holdout bars, 2000–2024. Code: `scripts/hull_trade.py`.
Raw: `reports/hull_trade.txt`, `reports/hull_trade.csv`.*

The rule, stated exactly, because the chart version is ambiguous: both flips
never land on the same bar, so it means the **conjunction of two states** —
enter the first bar where the Hull is rising AND the signal is on, exit the
first bar where both reverse. Every fill is the **next bar's close**, never the
bar that produced the condition.

---

## 1. The answer to the two questions asked

**Hull 55 + the EMA stack signal, exit on both — 15,327 round trips over 703
names, mean hold 32 sessions, net of 56 bps:**

| | rule | plain hold, same length |
|---|---|---|
| **win rate** | **32.5%** | 42.7% |
| **average return** | **+5.54%** | +2.08% |
| median | −2.58% | — |
| mean log | +0.0127 | −0.0022 |

**G1 confirmed exactly.** A win rate well below half and a strongly positive
average: many small losses, a few enormous winners. **The best 1% of trades
contribute 71% of the total return.** The distribution: p5 −18.2%, p25 −7.2%,
p50 −2.6%, p75 +2.6%, p95 **+48.7%**, p99 **+161.6%**.

**On a per-trade basis the rule genuinely beats a random-start hold of the same
length — +5.54% against +2.08%.** That is a real conditioning effect and it is
positive in both halves (+0.0254 early, +0.0052 late of mean-log edge).

---

## 2. And then the number that settles it

The rule is **in the market 33.5% of the time**. Compounding every trade a name
produced, over the span from its first entry to its last exit, against simply
owning that name across the same span:

| | |
|---|---|
| **strategy, median CAGR** | **+1.13%** |
| **owning the name over the same span** | **+9.88%** |
| trades where the strategy beat holding | **21.5% of names** |

**Across the whole 40-cell grid, ZERO configurations beat buy-and-hold on
CAGR**, and none beats it on more than **48.9%** of names. The best cell in the
project — `hull89 + EMA50 / exit hull only` — returns **+1.63% against a hold's
+2.76%**.

**G2 confirmed on the thing that matters.** The per-trade edge is real; the
money is not, because two thirds of the time the capital is in cash and the
market it is out of goes up.

**MY FIRST VERSION OF THIS COMPARISON WAS MEANINGLESS AND PRINTED 0.0%.** It
compared each trade to a "buy-and-hold" computed over that trade's *own* entry
and exit bars — which is the same trade minus the toll, so the rule could never
win by construction. A statistic that reads exactly 0.0% is not a finding, it is
a bug; the campaign comparison above is what the question actually asks.

---

## 3. The exit: sell on Hull red alone, or wait for both?

IDX is long-only, so the exit is the entire risk decision. Three exit rules,
same entries:

| exit | trades | win | mean | bars | in market | **CAGR** | vs hold |
|---|---|---|---|---|---|---|---|
| **Hull red alone** (hull89 + stack) | 14,596 | 34.5% | +4.90% | 29 | 29.6% | **+1.45%** | +10.72% |
| both reverse (hull89 + stack) | 12,589 | 33.9% | **+6.85%** | 38 | 33.4% | +1.00% | +9.88% |
| signal off alone (hull89 + stack) | 19,656 | 30.7% | +3.95% | 23 | 31.2% | +0.56% | +10.26% |

**Selling the moment the Hull turns red is the better of the three, and it is
better for a reason worth understanding.** Waiting for both conditions earns
**more per trade** (+6.85% against +4.90%) because it holds the winners longer —
but it also sits through the first leg down on every loser, holds 38 sessions
instead of 29, and ends up with a *lower* compounded return. Faster out wins on
CAGR; slower out wins on the average trade. Both lose to holding.

*The exit ordering is stable across every Hull length and every signal in the
grid, which is the only reason it is quoted at all: it is the one comparison
here that is not the maximum of a sweep.*

---

## 4. What the benchmark column accidentally revealed

Look at `vs hold` across signals. For the **EMA-stack** cells the hold benchmark
is **+9.9% to +11.2%** a year; for the EMA34, EMA50 and hull-only cells it is
**+2.0% to +3.0%**. The panel-wide median is **+2.56%**.

**So the EMA-stack entry condition is genuinely good at picking spans.** It
selects periods in which merely owning the name returned about **10% a year**
against a typical **2.6%** — a four-fold improvement, from a filter, before any
trading. And then the trading rule converts that 9.9% into **1.13%**.

That is the sharpest statement this study produces: **the signal carries real
information about which periods to be invested in, and the act of trading it
destroys roughly nine tenths of that information's value.** The toll is part of
it — 32 sessions a trade at 56 bps is about 4.4% a year in fees — but the larger
part is simply being out of a rising asset two thirds of the time.

---

## 5. Optimisation, and what it is worth

**G3 confirmed.** The grid is 40 cells and 33 of them are positive in both
halves on the *duration-matched* edge, which sounds like a strong result and is
not one: that edge measures "better than a random-start hold of the same
length", not "better than holding". On CAGR the ranking is unanimous and
negative.

By entry year the rule is a market-beta machine: +0.2275 mean in 2010,
+0.1210 in 2007, +0.1016 in 2021 — and **−0.0427 in 2008, −0.0148 in 2015,
−0.0280 in 2024**. It makes money when IDX rises and loses when it falls, with
extra turnover.

---

## What this licenses

- **The entry filter is worth keeping as a filter.** Names in the EMA-stack
  state during Hull uptrends returned ~10%/yr held, against a 2.6% panel median.
  Use it to decide *what to own*, not *when to flip*.
- **Do not trade the round trip.** 0 of 40 configurations beat owning the same
  names over the same spans, and the best manages +1.63% against +2.76%.
- **If you trade it anyway, exit on the Hull alone.** It is measurably better
  than waiting for both, in every Hull length and every signal tested.
- **Expect 1 trade in 3 to win.** That is the shape of the rule, not a defect —
  but it means the account depends on the top 1% of trades, and a year without
  one is a year of small losses.
