# The card, year by year, against the IHSG

*2026-11-17. `scripts/card_backtest.py`; output `reports/card_backtest.json`
(local). No hypothesis registered and no trial added: this re-measures the
rule H56/H63 already measured and shows it the way a holder reads it.*

## 1. What was run

The shipped quarterly card (`scripts/rules.py`): buy the top 10 by `hi52`
among names in the top 10% on `hi52` and the calmest half on `vol60`, keep a
name while it stays in the top 30% / calmest 60%, review every 63 sessions,
−20% stop from the fill, sell half at +100%. Walked by `stoptest.walk` on the
six rebalance calendars H56/H63 use.

Every calendar and the index are reported over **one common window**,
2008-03-25 → 2026-11-13 (18.64 years): the latest calendar start to the panel
end. A19 records comparing quantities measured over different windows as this
repo's most repeated error; here the dates are shared by construction, and
`summarise()` asserts it.

The IHSG is `^JKSE` with a 1.77%/yr dividend accrual (A19's measured
top-decile yield), compounded in calendar time and read as of the card's own
dates.

**Positive control.** With the harness as shipped, the medians of each
calendar's own-window CAGR are 13.1894% (no levels) and 11.7222% (card),
which are exactly `reports/stoptest.json`'s figures. This is therefore the
harness the card quotes, not a re-implementation of it.

## 2. Two harness corrections, printed beside the uncorrected numbers

The 2026-11-16 feature inventory found two cost errors in `stoptest.walk`.
Both are now selectable; neither changes the shipped default.

1. **One and a half round trips per position.** Every sale paid the full
   round-trip cost, on top of the half every purchase already pays through
   the rebalance turnover. `exit_cost_mult=0.5` charges exactly one round
   trip; `tests/test_card_backtest.py` pins the identity on a flat panel.
2. **Today's tick ladder on every bar.** Before 2014-01-06 the fraksi-harga
   ladder was coarser, so the spread was understated there: the lookahead
   CLAUDE.md §5 names. The corrected variant uses the point-in-time ladder from
   `reference.py`, checked against `tick_size` on every regime.

| over the common window | CAGR (median of 6) | calendar range | vs IHSG | calendars ahead | max DD | first half / second half |
|---|---|---|---|---|---|---|
| IHSG total return | +6.19% | — | — | — | −55.3% | +11.80% / +0.83% |
| card, as printed | +11.35% | +10.50% .. +12.93% | +5.16 | 6 / 6 | −30.8% | +11.22% / +11.30% |
| no levels, as printed | +12.17% | +10.28% .. +13.13% | +5.98 | 6 / 6 | −39.4% | +12.80% / +11.56% |
| **card, corrected** | **+11.83%** | +10.95% .. +13.36% | **+5.64** | **6 / 6** | **−30.5%** | +11.55% / +11.87% |
| no levels, corrected | +12.63% | +10.69% .. +13.56% | +6.44 | 6 / 6 | −39.0% | +13.12% / +12.07% |

Net of the two corrections, the card reads about half a point a year **higher**
than printed. The fee double-count (+0.5) outweighs the tick lookahead (−0.2).
The cost of the stop and scale-out barely moves: 0.80 points a year corrected
against 0.82 printed on this window (differences of medians).

## 3. Year by year: the corrected card (median calendar, min..max)

| year | card | range across calendars | IHSG TR | ahead |
|---|---|---|---|---|
| 2008 (from 25 Mar) | +0.0% | −8.1% .. +7.9% | −43.2% | yes |
| 2009 | −0.5% | −9.5% .. +27.0% | +90.3% | |
| 2010 | +46.6% | +32.6% .. +86.6% | +48.7% | |
| 2011 | −2.2% | −24.7% .. +14.0% | +5.0% | |
| 2012 | +13.6% | +6.3% .. +35.3% | +14.9% | |
| 2013 | +5.8% | +0.2% .. +17.7% | +0.8% | yes |
| 2014 | +25.8% | +11.6% .. +38.5% | +24.5% | yes |
| 2015 | −6.7% | −14.9% .. −4.0% | −10.6% | yes |
| 2016 | +16.6% | +6.8% .. +23.8% | +17.4% | |
| 2017 | +31.7% | +22.1% .. +36.2% | +22.1% | yes |
| 2018 | −6.2% | −16.9% .. −2.0% | −0.8% | |
| 2019 | +12.1% | −8.9% .. +32.0% | +3.5% | yes |
| 2020 | +2.9% | −1.6% .. +11.8% | −3.4% | yes |
| 2021 | +29.4% | +17.6% .. +49.9% | +12.0% | yes |
| 2022 | +7.2% | −0.9% .. +18.7% | +5.9% | yes |
| 2023 | +6.8% | −4.6% .. +12.2% | +8.0% | |
| 2024 | +2.6% | −1.7% .. +15.4% | −0.9% | yes |
| 2025 | +45.8% | +33.4% .. +61.0% | +24.3% | yes |
| 2026 (to 13 Nov) | +5.6% | −4.6% .. +17.8% | −37.2% | yes |

Ahead in 12 of 19 calendar years.

## 4. Three readings the headline does not show

**2008–09 is mostly cash, and that is the filter, not a decision.** Median
cash share was 76% in 2008 and 71% in 2009, with about 1.3–1.4 funded names.
Too few names cleared the nominal Rp1bn/day floor to form a basket, and the
walk keeps what it holds rather than liquidating (A39). So the card sat out the
2008 crash and missed the 2009 rebound by construction. Over the two years
together it made about −0.5% against the index's +8%. Cash was 33% in 2010 and
at most about 11% from 2012, with 8–10 names held.

The comparison does not rest on that period. From 2010-01-01 the card returns
+12.92% a year against +6.37%, and from 2012-01-01 +11.95% against +4.09%: 6 of
6 calendars ahead both times.

**The whole lead is in the second half.** Split at the window's date midpoint
(July 2017), the card and the index tie in the first half (+11.55% vs +11.80%).
The card's lead is entirely 2017–2026, a stretch in which the IHSG total return
compounded at +0.83% a year, and the price index fell 41.5% from its
2026-01-20 peak to 2026-11-13.
A tie in one half and a lead in the other is the shape A18 and A43 called
regime-dependent. It is not evidence of a stable edge.

**The calendars disagree by more than the correction.** Calendar CAGRs span
+10.95% to +13.36% and maximum drawdowns −24.9% to −41.0%, from moving the
review date by a few weeks. That spread is larger than either harness
correction.

## 5. What this does not fix

* **Survivor-only.** The panel holds 828 live names and no delisted ones; the
  delisted store waits on the user's licence decision (A51). H54's partial
  correction moved this family by −2.64 to +0.90 points a year, so the sign of
  the bias on these figures is not known.
* **Same-close fills.** Every trade fills at the close of the bar that
  signalled it.
* **Optimistic spread.** Half a tick, against OJK's measured ~1.6 ticks; no
  impact, suspension or auto-rejection term (A23).
* **In-sample.** The holdout was spent at H16. Six overlapping calendars are a
  robustness check, not a significance test. The shipped buffer has no deflated
  Sharpe (H58's 0.974 belongs to a sibling arm).

## 6. Out of sample so far

The forward record holds one card basket, emitted 2026-09-04 (10 names). After
50 sessions it was at −0.85% (mean log −0.021), with two names stopped out. Over
the same dates the IHSG fell 19.5%. That is one observation of ten correlated
names: about one effective data point, and evidence of nothing yet.

## Reproduce

```bash
python scripts/card_backtest.py --chart reports/card_backtest.png
```

About 10 minutes on three worker processes.
