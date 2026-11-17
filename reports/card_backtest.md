# The card, year by year, against the IHSG

*2026-11-17. `scripts/card_backtest.py`; outputs `reports/card_backtest.json`
and `reports/card_backtest_curves.parquet` (both local). No hypothesis
registered and no trial added. This re-measures the rule H56/H63 already
measured and shows it the way a holder reads it. Every number below was
re-derived by an independent check with its own code (§7).*

## 1. What was run

The shipped quarterly card (`scripts/rules.py`): buy the top 10 by `hi52`
among names in the top 10% on `hi52` and the calmest half on `vol60`, keep a
name while it stays in the top 30% / calmest 60%, review every 63 sessions,
−20% stop from the fill, sell half at +100%. It is walked by `stoptest.walk` on
the six rebalance calendars H56/H63 use.

Every series is reported over **one common window**, 2008-03-25 → 2026-11-13
(18.64 years): the dates on which every calendar *has an equity curve*.
`summarise()` asserts that the dates are shared. On the window's first day
three of the six calendars held no stock at all (§4).

Three series are compared:

* the card;
* the **IHSG**: `^JKSE` with a 1.77%/yr dividend accrual (A19), read as of the
  card's dates;
* **every eligible stock in equal weight**, rebalanced on the card's own
  calendars at the same costs. This is the comparison between the card and a
  cap-weighted index. A19 measured an equal-weighted IDX basket and the
  cap-weighted index behaving as different animals, so the card-versus-index
  gap mixes selection with weighting, and this series separates them.

**Positive control.** With the harness as shipped, the medians of each
calendar's own-window CAGR are 13.1894% (no levels) and 11.7222% (card). These
are exactly `reports/stoptest.json`'s figures, and the script checks this
against the file on every run.

## 2. The cost corrections, beside the uncorrected numbers

The 2026-11-16 feature inventory found two cost errors in `stoptest.walk`. Both
are selectable, and neither changes the shipped default.

1. **One and a half round trips per position.** Every sale paid a full round
   trip on top of the half every purchase pays. `exit_cost_mult=0.5` charges
   exactly one. Tests pin the identity on the band-exit, stop and scale-out
   paths.
2. **Today's tick ladder on every bar.** The point-in-time ladder was coarser
   before 2014-01-06 and *finer* in two bands from 2014-01-06 to 2016-05-02, so
   the error runs both ways; the net is a small understatement. The lookahead
   is reduced, not removed: about a fifth of bars close off the point-in-time
   grid because they are vendor-adjusted (A2).

| over the common window | CAGR (median of 6) | calendar range | vs IHSG | vs equal-weight | max DD (median / worst) | first half / second half |
|---|---|---|---|---|---|---|
| IHSG total return | +6.19% | — | — | — | −55.3% | +11.80% / +0.83% |
| every eligible stock, equal weight | +5.03% | +4.62 .. +5.44 | −1.16 | — | −58.1% / −58.9% | +8.56% / +1.86% |
| card, as printed | +11.35% | +10.50 .. +12.93 | +5.16 (6/6) | +6.40 (6/6) | −30.8% / −42.1% | +11.22% / +11.30% |
| no levels, as printed | +12.17% | +10.28 .. +13.13 | +5.98 (6/6) | +7.15 (6/6) | −39.4% / −48.7% | +12.80% / +11.56% |
| **card, corrected** | **+11.83%** | +10.95 .. +13.36 | **+5.64 (6/6)** | **+6.86 (6/6)** | **−30.5% / −41.0%** | +11.55% / +11.87% |
| no levels, corrected | +12.63% | +10.69 .. +13.56 | +6.44 (6/6) | +7.60 (6/6) | −39.0% / −47.9% | +13.12% / +12.07% |
| card, corrected, 1 tick of spread | +11.33% | +10.44 .. +12.86 | +5.14 (6/6) | +6.34 (6/6) | −31.0% / −41.7% | |
| card, corrected, 1.6 ticks (OJK) | +10.73% | +9.82 .. +12.26 | +4.54 (6/6) | +5.72 (6/6) | −31.7% / −42.5% | |

The two corrections net to about **+0.5 points a year**: the fee fix adds about
+0.6 and the tick fix subtracts about 0.2. Pricing the spread realistically
moves the result more than either correction: −1.1 points at OJK's measured
~1.6 ticks. The card stays ahead of both benchmarks in every calendar at every
spread tested.

## 3. Year by year: the corrected card (median calendar, min..max)

| year | card | range across calendars | IHSG TR | calendars ahead |
|---|---|---|---|---|
| 2008 (from 25 Mar) | +0.0% | −8.1% .. +7.9% | −43.2% | 6 / 6 |
| 2009 | −0.5% | −9.5% .. +27.0% | +90.3% | 0 / 6 |
| 2010 | +46.6% | +32.6% .. +86.6% | +48.7% | 3 / 6 |
| 2011 | −2.2% | −24.7% .. +14.0% | +5.0% | 2 / 6 |
| 2012 | +13.6% | +6.3% .. +35.3% | +14.9% | 3 / 6 |
| 2013 | +5.8% | +0.2% .. +17.7% | +0.8% | 5 / 6 |
| 2014 | +25.8% | +11.6% .. +38.5% | +24.5% | 3 / 6 |
| 2015 | −6.7% | −14.9% .. −4.0% | −10.6% | 4 / 6 |
| 2016 | +16.6% | +6.8% .. +23.8% | +17.4% | 3 / 6 |
| 2017 | +31.7% | +22.1% .. +36.2% | +22.1% | 5 / 6 |
| 2018 | −6.2% | −16.9% .. −2.0% | −0.8% | 0 / 6 |
| 2019 | +12.1% | −8.9% .. +32.0% | +3.5% | 5 / 6 |
| 2020 | +2.9% | −1.6% .. +11.8% | −3.4% | 6 / 6 |
| 2021 | +29.4% | +17.6% .. +49.9% | +12.0% | 6 / 6 |
| 2022 | +7.2% | −0.9% .. +18.7% | +5.9% | 3 / 6 |
| 2023 | +6.8% | −4.6% .. +12.2% | +8.0% | 2 / 6 |
| 2024 | +2.6% | −1.7% .. +15.4% | −0.9% | 5 / 6 |
| 2025 | +45.8% | +33.4% .. +61.0% | +24.3% | 6 / 6 |
| 2026 (to 13 Nov) | +5.6% | −4.6% .. +17.8% | −37.2% | 6 / 6 |

The median calendar is ahead in 12 of 19 years. Five years are 3–3 splits. The
rows are medians per year, so they do not chain exactly to the headline (they
compound to 11.68%, not 11.83%).

## 4. What the headline does not show

**About half the lead over the index is 2026.** The window closes inside a
41.5% fall in the IHSG from its 2026-01-20 peak, concentrated in mega-caps.
Ending earlier:

| window | card | IHSG TR | equal weight | card max DD | IHSG max DD |
|---|---|---|---|---|---|
| 2008-03-25 → 2026-11-13 | +11.83% | +6.19% | +5.03% | −30.5% | −55.3% |
| 2008-03-25 → 2025-12-31 | +12.21% | +9.33% | +7.65% | −30.5% | −55.3% |
| 2008-03-25 → 2024-12-31 | +10.56% | +8.50% | +5.35% | −30.5% | −55.3% |
| 2010-01-01 → 2026-11-13 | +12.92% | +6.37% | +5.33% | −27.3% | −45.6% |
| 2012-01-01 → 2026-11-13 | +11.95% | +4.09% | +3.71% | −26.1% | −45.6% |

6 of 6 calendars are ahead of the IHSG in every row. Without 2026 the lead is
+2 to +3 points a year, not +5.6.

**2008–09 is mostly cash, and that is a backtest guard, not the rule.** From
2008-03-25 the median book was about 82% cash with about one funded name, and
71% cash in 2009. Most 2008–09 reviews could not form a five-name basket: the
entry band passed 2–6 names from 42–60 eligible, and `MIN_BASKET` refuses
anything under five. From late 2008 to mid-2009 the universe also fell under
the 40-name floor. Stops and band exits sat in that cash. So the card sat out
the 2008 crash and missed the 2009 rebound by construction, and the no-levels
arm did the same. Over the two years together it made about −0.3% against the
index's +8.0%. `rules.py` has no such guard in that form, so the backtest does
not say what a holder would have done then. The 2010 and 2012 rows above drop
those years.

**Against the index the lead is all second half; against the universe it is in
both.** Split at the window's date midpoint (July 2017), the card and the IHSG
tie in the first half: +11.55% against +11.80%, with 3 of 6 calendars ahead.
The card itself was flat across the halves. What changed is the index, which
compounded at +0.83% a year from 2017. Against every eligible stock held
equally, the card leads in both halves: +2.96 points in the first, with 5 of 6
calendars ahead, and +10.36 in the second, 6 of 6. On these numbers the regime
sits in the benchmark's cap-weighting more than in the card.

**The calendars disagree by more than the corrections.** Calendar CAGRs span
+10.95% to +13.36% and maximum drawdowns −24.9% to −41.0%, from moving the
review date by a few weeks.

## 5. Two more harness defects, found by the independent check, not fixed here

Both are pre-existing in `stoptest.walk`, so they are also in
`reports/stoptest.json` and in the card's fourth column.

* **A review on a non-trading row skips the quarter.** The panel carries rows
  for 59 exchange holidays inside the window, plus vendor zero-volume days.
  A review landing on one sees no eligible names and does nothing for 63
  sessions; this happens to the card eight times inside the window, and once
  just before it in a way that changes a path. Deferring those
  reviews to the next trading session leaves the corrected card's median
  almost unchanged (11.83% → 11.79%) and drops its weakest calendar (10.95% →
  10.12%). It **roughly halves the measured CAGR cost of the stop and
  scale-out**: from 1.47 to 0.70 points on `stoptest.json`'s convention, and
  from 0.80 to 0.43 on this window.
* **Unfunded placeholders keep a stale reference.** A name added at a review
  that could not form a basket keeps that review's price as the reference for
  its stop and target, and fires zero-value stops. Resetting the reference at
  funding moves the median by +0.09 points.

With both fixed, the corrected card reads 11.75% (10.33 .. 13.36). These
figures come from the checker's own code; they will enter the harness only
when the fixes are adopted.

## 6. What this still does not fix

* **Survivor-only.** The panel holds 828 live names and no delisted ones; the
  delisted store waits on the user's licence decision (A51). H54's partial
  correction moved this family by −2.64 to +0.90 points a year, and the bias is
  worst in the early years.
* **Same-close fills** at the signal bar. In the forward record, LSIP filled
  four points through its −20% stop.
* **No impact, suspension or auto-rejection term** (A23).
* **In-sample.** The holdout was spent at H16. Six overlapping calendars on one
  18.6-year history show robustness to review timing, not significance. The
  shipped buffer has no deflated Sharpe.

## 7. How it was checked

Two independent agents re-derived every figure with their own code, sharing
only `stoptest.walk` and the panel loader:

* the point-in-time cost column was rebuilt from `reference.tick_size`;
* every curve was replayed bit for bit;
* the IHSG was re-aligned;
* the cash shares were re-measured with an instrumented walk.

Every number above reproduced exactly, including the equal-weight benchmark.
They also corrected the first draft's explanation of the 2008–09 cash (it
blamed the liquidity floor; the main cause was `MIN_BASKET`). And they caught
that the first chart plotted the drawdown of the per-date median line, a path
no calendar followed, which understated drawdown by about 8 points.

## 8. Out of sample so far

The forward record holds one card basket, emitted 2026-09-04 (10 names). After
50 sessions it was at −0.85% (mean log −0.021), with two names stopped out. Over
the same dates the IHSG fell 19.5%. That is one observation of ten correlated
names: about one effective data point, and evidence of nothing yet.

## Reproduce

```bash
python scripts/card_backtest.py --chart reports/card_backtest.png
```

About 15 minutes on three worker processes. `--chart-only` redraws the chart
from the saved curves.
