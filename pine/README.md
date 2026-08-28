# IDX Suite — a TradingView indicator

`IDX_Suite.pine` — paste into TradingView's Pine editor and add to any IDX chart.

Four layers: a Hull ribbon and EMA stack (the visual), IDX mechanics (exact
arithmetic), a measured price×time **projection**, and flip labels that carry
their own measurement. Full study: `reports/time_price.md`.

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
