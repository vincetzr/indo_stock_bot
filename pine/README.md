# IDX Suite — a TradingView indicator

`IDX_Suite.pine` — paste into TradingView's Pine editor and add to any IDX chart.

Five layers: a Hull ribbon and EMA stack (the visual), IDX mechanics (exact
arithmetic), a measured price×time **projection**, **swing support/resistance
with target and stop odds**, and flip labels that carry their own measurement.
Studies: `reports/time_price.md`, `reports/levels.md`.

## Import

1. TradingView → open any IDX chart (e.g. `IDX:BBCA`).
2. Bottom panel → **Pine Editor** → **Open** → **New blank indicator**.
3. Select everything in the editor and replace it with `IDX_Suite.pine`.
4. **Save** (name it anything), then **Add to chart**.
5. Set the chart to **daily** — every number is measured on daily bars.
6. Chart settings → Symbol → tick **Adjust data for dividends** if available.
   The research runs on adjusted close; leaving it off is a small, one-signed
   error (~1.27%/yr, the measured IDX dividend).

## What the panel shows, and how much to trust each row

The panel is deliberately split into three blocks because they are **not**
equally reliable.

**`[exact]` — IDX mechanics.** Tick size (fraksi harga), maximum price step,
tomorrow's ARA/ARB auto-rejection levels, and the round-trip cost floor. These
are arithmetic from published IDX rules, not statistics. They are simply
correct, and they are the most useful thing on the chart.

* The **board** row matters. The thin board (Papan Pemantauan Khusus /
  Akselerasi) trades a flat ±10%; the main board is +35/25/20% up against a
  flat −15% down. Using the main ladder on a thin-board name allows moves that
  cannot happen — this was a real bug in the research repo affecting 41 of 818
  names. Auto-detection uses IDX's own six-month-average-price rule, which is a
  guess; override it if you know the board.
* **Round trip** is a *floor*: your commissions plus half a tick each way. Half
  a tick assumes a one-tick-wide book — roughly right on a large cap, generous
  on a small one. Real cost is worse, never better. Set the fee inputs to your
  own broker's schedule; the defaults are 0.28/0.18/0.10.
* A name locked at **ARB cannot be sold at all.** On the thin, volatile end of
  IDX that happens ~4 sessions a year per name.

**`[state]`** — where the name is. Measurement, no inference.

**`[measured]`** — what that state implied historically, over 31,394 name-years
of IDX data, 2000–2024. Frequencies, not forecasts. Read the caveats below.

## The numbers, and where they come from

Asymmetry by distance from the 52-week high (1-year horizon):

| bucket | P(touch 2×) | P(end ≤ half) | skew |
|---|---|---|---|
| near the high (top decile) | 13.6% | 6.3% | **2.15** |
| mid-range | 11.2% | 7.4% | 1.53 |
| far below the high | 15.1% | **18.9%** | **0.80** |
| *any IDX name, unscreened* | 12.1% | 9.0% | 1.33 |

**The gradient is monotone and it is the opposite of the usual intuition:
fallen names halve more often than they double.** "It's already down 70%, how
much lower can it go" is measurably backwards on IDX.

Recovery curve — P(a new 60-session high) *given* you are already X below the
peak, from 243,977 bars. It crosses one-half between −5% and −10%:

| −5% | −10% | −15% | −20% | −25% | −30% | −40% |
|---|---|---|---|---|---|---|
| 81.3% | 56.1% | 38.8% | 27.1% | 17.3% | 11.3% | 5.9% |

Being **above the EMA50 in a shallow drawdown** is worth up to +22.4 points
(9 positive / 0 negative across depths). The stochastic cross tested 5+/6− —
noise — so it is not in the script.

Doubling rate by holding period, unscreened: **9.5% at 1y, 27.0% at 3y, 39.0%
at 5y, 55.5% at 10y.** This row is on the panel so the one-year figure is never
read as the whole story.

## The optional screen

Off by default. `close ≥ 0.9625 × 252-bar high` **and** `60-bar return stdev ≤
0.0257`: P(touch 2×) 8.3%, P(halve) 4.1%, skew **2.01**. Clustered permutation
null over whole (ticker, year) blocks, 5,000 draws: 1.20 ± 0.13, **z = +6.06,
p = 0.00020** against a Bonferroni bar of 0.00057 — it clears. Half-split 2.44
early / 1.39 late against a base of 1.61 / 1.13.

**Note what it gives up:** it doubles *less* often than an unscreened name
(8.3% vs 12.1%). Its entire edge is the denominator.

## Why there are no `request.security` calls

The research ranks every IDX name against every other on the same day; a chart
sees one symbol. The obvious fix — pull 36 reference symbols and rank against
them — was built and backtested, and **it lost**: skew 1.86 and 53% of the real
rule's picks, against 2.01 and 72% for plain absolute thresholds. A percentile
ranked against ~24 live names is noisier than a fixed threshold is stale.

## What this is not

* **In-sample.** The out-of-sample holdout was spent once and is gone.
* **One year.** Every probability is a 252-session figure, and the same family
  of rules inverts below three years.
* **Not a forecast of magnitude.** Nothing in this research predicts how far a
  price goes. There is no price target anywhere in the script, deliberately.
* **Not a timing tool.** Nine index-timing rules were tested across two
  independent halves; all eighteen cells lost to buy-and-hold.
* **Per-name, not portfolio.** These are per-name frequencies, and per-name
  statistics and portfolio outcomes can disagree completely.
* **Not advice**, and not suitable as the basis of a discretionary
  recommendation for someone else's money.


## The projection, added 2026-08-28

Set a target percentage and the panel answers three measured questions: how
often an IDX name in this state touched `+X%` within a year, how often it
touched `−X%` instead, and — if it got there — when. The date band is a
**quartile** band drawn as a box on the chart: half the cases that reach the
target do so between the two dates.

**Why a band and not a date.** The premise of any time-projection method is
that turning points recur on a schedule. Tested four ways on 891 names and it
does not:

| test | result |
|---|---|
| ZigZag pivot-spacing regularity | CV **2.246** vs a block-bootstrap null of **1.340**, z **+32.7** — *less* regular than random |
| is the interval memory a cycle? | at a 252-day block the null reproduces **87%** of it, excess z **+1.64**, ns |
| a fixed period in the IHSG | strongest 885 sessions, power **140.2** vs null **151.6 ± 56.6**, **p = 0.499** |
| month-of-year | pivot share runs **0.865 – 1.097** |

Knowing a name's entire history of turn spacings narrows the 50% band for its
next turn from ±140% of the median gap to ±125%.

**And trend state moves the odds, not the clock.** Median sessions to +20%:
base 54, EMA-stacked 54, not stacked 54, Hull rising 53. The only thing that
moves the clock is volatility — 89 sessions in the calmest IDX decile against
30 in the wildest — which is why the band is computed from the name's own
`vol60`.

**A trend filter is mostly a risk filter.** The stack multiplies upside odds by
**1.19** and downside odds by **0.71**.

**What it is worth.** The two laws reproduce the 180 measured cells to a median
of **1.7 probability points** and **6% on the median time**; `python3
scripts/pine_cone_check.py` prints the residuals and fails if they drift. Every
number is in-sample, one-year, and the top of the date band is partly the
252-session ceiling rather than the market.

## The flip labels are drawn and are measured as worthless

60-session hold, net of 56 bps, mean log return:

| rule | mean log |
|---|---|
| any eligible bar, no toll — the benchmark | **−0.0140** |
| EMA stack turns on | −0.0124 |
| Hull-55 slope turns up | −0.0148 |
| **HMA-21 crosses over HMA-55** | **−0.0191** |

The classic dual-Hull cross is the worst of the four. That matches the repo's
earlier Hull Suite + UT Bot work — 84 names, 240 configurations, best median
excess CAGR −6.1%, lost to buy-and-hold in all five walk-forward folds. Set
**Label these flips** to `None` if the arrows tempt you.


## Target and stop, added 2026-08-28

Both were tested before either was drawn, and the two came back opposite.

**Prior swing highs and lows are real barriers.** The event is *the first close
above a confirmed prior swing high*; the placebo is the same event against that
same high **displaced ±7%** — identical construction, wrong level. 23,235
events, distance-adjusted, ticker-clustered 95% CI:

| statistic | true-level effect | 95% CI | early | late |
|---|---|---|---|---|
| false break (closes back under within 20b) | **−6.95 pp** | [−8.46, −5.77] | −8.60 | −7.11 |
| follow-through (+5% within 20b) | **+4.46 pp** | [+3.15, +5.59] | +5.27 | +4.04 |
| forward 20-session return | **+1.00%** | [+0.57, +1.46] | +1.33 | +1.40 |

All three replicate in both halves — the only level result in this research that
does. **What it is not:** the false-break rate at the *true* level is still
**66.4%**. Two breakouts in three close back under within a month. A real level
beats a fake one; breakouts still mostly fail.

**Fibonacci measured nothing, and is off by default.** On a continuous grid of
33 retracement depths, P(the leg high is regained within 60 sessions) declines
monotonically from 0.3964 at r=0.15 to 0.2911 at r=0.95 with no bump anywhere —
0.500 reads **0.3611**, sitting exactly between 0.475's 0.3631 and 0.525's
0.3571. Against 2,000 draws of five random non-Fibonacci nodes from the same
grid: **z = +0.77, +0.68, +0.41, −0.95; every p above 0.35.** The lines are
drawn on request and the panel says what they are worth.

**No fixed bracket beats holding.** 30 (take-profit, stop) pairs over 17.6
million entry-cells, filled at the actual close rather than the nominal level,
compared to a hold of the *same duration*, and annualised:

| tp | sl | P(tp first) | P(sl first) | bars held | annualised | early | late |
|---|---|---|---|---|---|---|---|
| 0.50 | 0.05 | 0.137 | **0.806** | 52 | **+2.40%** | +8.04% | **−3.16%** |
| 0.30 | 0.05 | 0.207 | 0.770 | 40 | +1.09% | +6.73% | −4.47% |
| 0.10 | 0.30 | 0.670 | 0.250 | 73 | −10.3% | −6.2% | −14.6% |

**Not one of the thirty is positive in both halves**; the best compounds at
+2.4%/yr against an index at about +12.7%. The panel says so in the
`bracket verdict` row, next to the levels it prints.

The direction of the folklore is right and its size is not enough: against a
duration-matched hold, `tp 0.50 / sl 0.10` is worth **+0.0262** of mean log per
trade and `tp 0.05 / sl 0.30` **−0.0077**. Cut losses short and let winners run
is real, and the annualised column hands it back to costs.

**Reading the target/stop block.** The target is the nearest confirmed swing
high above price, the stop the nearest confirmed swing low below. `P(touch it
within a year)` comes from the H32 first-passage laws, and `P(target first)`
from a race law fitted on 300 cells — whose volatility coefficient is −0.0225
against distance coefficients near 0.8, meaning **which barrier arrives first is
a ratio of distances and not a statement about speed.**


## How accurate is it, measured on every stock (2026-08-28)

Three questions, three answers. Full study: `reports/accuracy.md`.

**The probabilities are honest; the skill over the base rate is thin.** The
panel emits probabilities, not calls, so a win rate is undefined until someone
picks a threshold. What is defined — with the laws refitted year by year on a
**purged** walk-forward, so this is genuinely out of sample:

| arm | Brier | base rate | skill | AUC |
|---|---|---|---|---|
| shipped constants | 0.1861 | 0.1935 | +0.0374 | 0.591 |
| **purged walk-forward** | 0.1912 | 0.1935 | **+0.0130** | 0.580 |
| **whole board, 777 names** | 0.1917 | 0.1956 | **+0.0168** | 0.594 |

Reliability, walk-forward — predicted against observed:

| 0.096 | 0.174 | 0.254 | 0.356 | 0.466 | 0.579 | 0.674 | 0.728 | 0.796 | 0.856 |
|---|---|---|---|---|---|---|---|---|---|
| 0.119 | 0.158 | 0.262 | 0.338 | 0.478 | 0.575 | 0.666 | 0.726 | 0.759 | 0.803 |

**The date band contains exactly the half it claims: 0.497 / 0.500 / 0.475.**

**Where there is no skill at all:** at +5% and +10% the AUC is 0.502 and 0.524
and the skill is *negative* — almost everything touches +5% inside a year, so
there is nothing to discriminate. All the skill is in the far targets, and more
of it on the downside (AUC 0.648 at −50%). The `P(target first)` row is
calibrated to within 0.3 points on average and has **AUC 0.51** — it prices a
decision you have already made and cannot make one for you.

**The colour turns near the top, about 11% below it.** 47,002 confirmed 10%
swing highs, each detector against a random one spending the *same number of
flips*:

| detector | recall | precision | F1 | null F1 | lag | give-back |
|---|---|---|---|---|---|---|
| **close over EMA34** | **0.852** | 0.583 | **0.692** | 0.554 | 6 | 10.9% |
| Hull-55 slope | 0.812 | 0.589 | 0.683 | 0.472 | 11 | 11.8% |
| close over EMA50 | 0.758 | **0.593** | 0.666 | 0.531 | 6 | 11.5% |
| HMA-21 over HMA-55 | 0.825 | 0.523 | 0.640 | 0.479 | 11 | 10.5% |
| price>50>100>200 | 0.348 | **0.614** | 0.445 | 0.396 | 9 | 12.5% |

EMA34 is the most accurate flip and is the default. The EMA stack's recall is
**below its own null** — it misses two thirds of tops and is a confirmation
rather than a detector.

**Give-back is the number nobody quotes.** The real detectors surrender **more**
of the peak than random bars do (10.5–12.5% against 8.5–8.9%), because a trend
flip fires *because* price fell. The colour does not change at the peak.

**Placement: further out on the target, tight on the stop, and it still loses.**
Mean net by target offset from the resistance level: −0.52% at five percent
short, −0.12% at the level, **+0.35% five percent beyond it**. Waiting for the
break beats selling into it, even though two breakouts in three fail. On the
stop the mean barely moves; only the mean log does. Twenty-five placements,
**none positive in both halves**.

## Dynamic across all history

The support, resistance and 50% retracement lines plot as a **stepped series on
every bar**, not just at the right edge. Scroll back and each bar shows the
level that was known *then* — the levels are built from confirmed pivots and
never repaint, so the chart can be audited by eye.


---

# How to use it, in plain English

*Added 2026-08-28, because everything above is written for someone checking the
statistics. This part is written for someone with the chart open.*

## The one-minute version

The chart tells you **where you are**. It does not tell you what to do, and the
research says that every attempt to turn it into a what-to-do lost money.

1. **Green Hull ribbon = the name is in an uptrend.** Red = it is not.
2. **The orange line under the ribbon is where the ribbon turns red tomorrow.**
   Exact arithmetic, not a guess. It is usually *far* below price — 13.5% at the
   median — so treat it as "the trend is over here", not as a stop.
3. **The red line above price is the target, the green line below is the stop.**
   Both are prior swing levels the whole market can see. Real, and measured.
4. **The panel's `P(target first | one is hit)` row is the trade in one number.**
   Above 50% the target is the more likely of the two to arrive first.
5. **The `round trip (floor)` row is what you pay to find out.** On a Rp 100
   stock it is over 1.5%, which is most of a small target.

## Reading the panel top to bottom

**STATE — where the name is.**

| row | what to do with it |
|---|---|
| annualised vol / IDX vol decile | decile 9–10 is the wild end. Everything moves faster there — the good and the bad equally. |
| % of 52-week high | above 90% is the good half of the measured asymmetry (skew 2.15 near the high against 0.80 far below). |
| drawdown from peak | past −20%, the odds of a new high inside 60 sessions are 27% and falling. |
| turnover | under Rp 1bn/day, the spread eats a small target. This is the single most common way a good-looking setup is not one. |

**TREND — is it going up.** `price > EMA50 > EMA100 > EMA200` is the best of 36
gridded states. The `60b fwd mean log` row is what that state was historically
worth over the next 60 sessions, against a base of −0.0140 — **note that the
base is negative**: the average IDX name over the average 60 sessions lost
money. The stack beats it. It does not by itself make money.

**PROJECTION — set your target, read the odds and the date band.** Turn the
`Target move %` dial. The panel answers: how often a name that looked like this
touched `+X%` within a year, how often it touched `−X%` instead, and the
quartile date band for the arrival. Half the cases that get there land between
the two dates. **Half do not, which is the point of a band.**

**TARGET / STOP.** The nearest confirmed swing high above and swing low below,
each with the chance of touching it inside a year, plus `P(target first)`.

**HOW ACCURATE.** The panel's own report card, on the chart, permanently.

**IDX MECHANICS.** Tomorrow's ARA and ARB prices, the tick, the round-trip cost.
This block is arithmetic and is the most reliable thing on the screen.

## A worked read

> Hull green, stack `price>F>M>S 3/3`, 8% off the 52-week high, turnover Rp
> 4bn, resistance +11.8% away, support −4.5% away, `P(target first) 61%`,
> round trip 0.72%.

Uptrend confirmed, liquid enough that the spread is not the trade, and the
target is 2.6× as far away as the stop while being the *more likely* of the two
to arrive first. That is the shape you want. It is still a **coin flip with a
good payout**, not a prediction — and 39% of the time the stop arrives first.

> Hull green, 31% off the high, turnover Rp 0.4bn, resistance +3% away, stop
> −24% away, round trip 1.9%.

A 3% target that costs 1.9% to attempt, with a stop eight times further away
than the target, on a name too thin to exit. Green ribbon, terrible trade. **The
ribbon colour is the least informative thing on the chart.**

## The four things the measurements say not to do

1. **Do not trade the BUY/SELL arrows.** All four flip rules lose over 60
   sessions net of cost; the classic dual-Hull cross is the worst. They are
   drawn because a chart is for looking at.
2. **Do not treat the Hull flip as a stop.** It sits 13.5% below price at the
   median and 76% below in the worst twentieth.
3. **Do not use Fibonacci.** It is off by default and measured nothing.
4. **Do not take a target under about 5%.** The cost floor eats it, and the
   probability laws have no discrimination that close in (AUC 0.502 at +5%).

## The daily routine

```bash
python3 scripts/refresh.py --panel     # ~6 min, pulls the whole board
python3 scripts/daily_signal.py        # names whose ribbon just turned green
python3 scripts/daily_signal.py --fresh 9999 --top 25   # every green name
```

Run it after the close. IDX shuts at 15:50 WIB, so 19:00 Shanghai (11:00 UTC,
18:00 WIB) is two hours after the bell and the daily bar is final.

The list is ordered by **expectancy** — `P(target first) × target −
P(stop first) × stop − cost` — and not by upside. That is deliberate: H25 ranked
on the upside alone, cleared this project's significance bar, and H26 then
showed the same screen was indistinguishable from a random cell on the ratio and
compounded at −16%/yr. *A rate is not an objective; check what the denominator
is doing before ranking on the numerator.*

**Most rows come back negative.** On a typical evening 4 or 5 of 25 green names
have a positive expectancy. That is the honest output of the arithmetic, and a
list where everything looks good is a list with the cost term missing.
