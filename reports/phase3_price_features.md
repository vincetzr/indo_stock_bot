# H13 memo — §8's price/TA features: real signal, and it still does not pay

**Date:** 2026-08-24
**Verdict: Gate 1 fails again, but for the OPPOSITE reason to the flow branch.**
The flow branch had no signal. This has plenty of signal and no economics.
Reproduce with `python3 scripts/price_ic.py --draws 100`.
Pre-registration with mechanisms and predicted signs: `hypotheses.md`, H13.

---

## 1. The sample is 130× the flow panel

| | H9 (flow) | H13 (price) |
|---|---|---|
| names | 176 | **891** (89 delisted) |
| resolution | fortnightly | **daily** |
| rows | 21,693 | **1,989,504** |
| span | 2014–2026 | **2000–2024** in sample |
| reachable horizons | k = 10, 20 | **k = 1, 5, 10, 20** |

Price features need no broker data, so nothing had to be fetched and the whole
spine was usable. Holdout: the most recent 24 months, untouched, as §11 requires.

---

## 2. Every feature is real. Every feature loses money.

Confirmatory horizon k = 5, pre-specified. Full decay curves in §3.

| feature | predicted | IC | HAC t | IC sign across liquidity | **net %/yr, k=5** | **net %/yr, k=20** |
|---|---|---|---|---|---|---|
| `lowvol` | + | **+0.0398** | **+14.49** | stable | −116.7 | −36.7 |
| `mom12_1` | + | +0.0274 | +10.75 | stable | −81.5 | −14.6 |
| `rev5` | + | +0.0270 | **+14.09** | stable | −72.1 | −18.9 |
| `hi52` | + | +0.0261 | +10.13 | flips in Q1 | −92.6 | −21.0 |
| `amihud60` | + | +0.0169 | +5.98 | flips in Q1 | −96.4 | −25.1 |
| `volz20` | **−** | +0.0136 | +7.18 | stable | −66.9 | −7.4 |
| `squeeze` | **0 (control)** | +0.0083 | +3.55 | stable | −84.8 | −18.0 |
| `atr_mom20` | + | +0.0061 | +3.32 | flips in Q1 | −77.7 | −12.5 |

Every IC sits outside all 100 permutation draws. Five of the eight carry a HAC
t above 10 — H9's flow signal managed −2.86 on the same statistic.

**And every single cell of the net column is negative.** The cost of one
rebalance is **1.7–1.9%**: A5's 56 bps round trip plus twice the point-in-time
fraksi-harga half-spread, charged at the prices of the names actually held.
The gross quintile spread at k=5 is 0.15–0.36% per period. The arithmetic is
not close.

### Two features did not do what I registered

**`volz20` came out with the wrong sign.** I registered **negative** on an
attention-induced-buying mechanism; observed **+0.0136**. Logged as a failed
prediction, not reframed.

**The negative control fired, and that is the most useful result here.**
`squeeze` was registered as predicted-null because range compression forecasts
the *size* of the next move, not its sign. It returns IC +0.0083 with
**t = +3.55** — nominally significant.

That is not a defect in `squeeze`; it is a statement about the test. **With two
million observations a t-statistic is nearly free.** An IC of 0.008 — eight
thousandths of a rank correlation — clears any conventional threshold at this
sample size while being economically meaningless. The negative control was put
in precisely to detect that, and it did. Every t-statistic in the table above
should be read through it.

---

## 3. The decay curves, in full

§7 requires the whole curve, never the best k. Net is on the curve too, because
turnover falls with k: a signal rebalanced every 5 days pays the round trip four
times as often as one held 20.

```
rev5      k= 1 IC +0.0201 (t +16.57)  gross  +50.2%/yr  net -396.4%
          k= 5 IC +0.0270 (t +14.09)  gross  +17.2%/yr  net  -72.1%
          k=10 IC +0.0218 (t +10.24)  gross   +6.9%/yr  net  -37.8%
          k=20 IC +0.0155 (t  +7.27)  gross   +3.4%/yr  net  -18.9%

mom12_1   k= 1 IC +0.0157 (t +13.08)  gross   -2.4%/yr  net -447.5%
          k= 5 IC +0.0274 (t +10.75)  gross   +7.5%/yr  net  -81.5%
          k=10 IC +0.0352 (t  +8.97)  gross   +8.6%/yr  net  -35.9%
          k=20 IC +0.0417 (t  +6.92)  gross   +7.7%/yr  net  -14.6%

lowvol    k=20 IC +0.0540 (t  +9.10)  gross  -12.9%/yr  net  -36.7%
hi52      k=20 IC +0.0375 (t  +6.15)  gross   +1.9%/yr  net  -21.0%
```

Two shapes worth naming. **Reversal decays with k** (IC 0.027 → 0.0155) —
short-horizon by construction. **Momentum and 52-week-high strengthen with k**
(0.0157 → 0.0417, 0.0162 → 0.0375) — slow-diffusion signals, exactly as their
mechanisms predict. That the curves have the shapes the registered mechanisms
imply is a point in favour of the features being real rather than fitted.

### A rank tilt is not a return spread

`lowvol` has IC **+0.0398 with t +14.49** and a quintile spread of
**−0.430% per period (t −5.24)** — positive rank correlation, negative extreme
spread. The two are not contradictory: rank IC is robust and covers the whole
cross-section, while a mean-return spread lives in the tails, where a handful
of high-volatility names deliver enormous positive returns. H9 hit the same
wedge from the other side ("a direction without a size"). **When they disagree,
the spread is the one that decides tradeability**, because the spread is what
you would actually hold.

---

## 4. The post-hoc check that came closest, and why it is not a finding

**Marked EXPLORATORY. This was not pre-registered and does not enter the trial
count.** The main run charges the half-spread on whatever the quintile sort
picks, including very cheap small caps where half a tick is hundreds of basis
points. The obvious question is whether the signal survives where the tick is
small.

| feature | universe | k | IC | t | cost/rebalance | **net %/yr** |
|---|---|---|---|---|---|---|
| `mom12_1` | top 5% liquidity | 20 | +0.0640 | **+2.00** | 1.02% | **+1.6** |
| `hi52` | top 5% liquidity | 20 | +0.0315 | **+0.95** | 1.02% | **+2.0** |
| `lowvol` | top 20% liquidity | 20 | +0.0656 | +7.35 | 1.34% | −3.6 |
| `rev5` | top 5% liquidity | 20 | +0.0067 | +0.45 | 0.99% | −5.8 |

Restricting to liquid names roughly **halves the cost** (1.77% → 1.02%) — and
**collapses the t-statistic five- to sevenfold**, because the top 5% is only a
few dozen names a day and because, for reversal at least, the signal was always
strongest where trading is dearest.

Two cells cross zero, at **+1.6% and +2.0% a year**. They are not a result:

- **It is a 24-cell search** (4 features × 3 universes × 2 horizons). Two
  marginal positives is what multiple testing produces from noise.
- **The t-statistics are +2.00 and +0.95.** Neither clears the Bonferroni bar
  of 3.21; `hi52` is not significant on any reading.
- **+2% a year is not a business** against the risk taken and the manual
  execution §3 requires.
- **The quintile spread is long-short, and A5 forbids shorting.** So even the
  positive cells describe a portfolio this project cannot hold. A long-only
  version earns top-quintile minus market, which is smaller again.

Taking +2%/yr from a 24-cell post-hoc sweep as a green light is precisely the
move §2 names as the worst thing that can be done in this repo.

---

## 5. What would have falsified this, and what did not

A feature clearing all four of Gate 1's conditions: significant, sign-stable
across liquidity, **net positive after costs**, out of sample. Six of eight
clear the first, five clear the second, **none clear the third**, and the
holdout was never touched because nothing earned the right to spend it.

---

## 6. What I now believe

**High confidence: IDX prices carry strong, real, mechanically-sensible
cross-sectional structure.** Reversal, momentum, low-volatility and 52-week-high
all behave as their literature predicts, with t-statistics an order of magnitude
above anything the flow branch produced, and with decay curves whose *shapes*
match the registered mechanisms.

**High confidence: none of it is tradeable at this cost structure.** The binding
constraint is not signal, it is the 56 bps round trip plus a fraksi-harga
half-spread that on cheap names exceeds the fee several times over. That is the
same wall Phase 2b hit from a different direction — the median cohort round trip
already loses 25–32 bps to the spread before any fee.

**High confidence, and new: at two million observations, significance is
uninformative.** The negative control clears t = 3.5. Effect size and cost are
the only things that discriminate here, and this repo should stop treating a
t-statistic as evidence of anything on panels this size.

**Medium: the liquid, long-horizon corner is not quite dead.** The best post-hoc
cells sit a whisker above zero rather than deep below it. That is worth
remembering, not acting on — and it would need pre-registration, a long-only
construction, and the holdout to become a claim.

---

## 7. Where this leaves the project

§4's cost-to-falsify ordering has now been run to its end on **both** branches:

| | instrument | result |
|---|---|---|
| H9 | aggregate broker flow | no signal |
| H10/H11 | broker identity | no signal, no persistence |
| H12 | investor class | no signal, powered null |
| **H13** | **price/TA and structural** | **strong signal, no economics** |

The honest summary of the whole research programme is that the Indonesian
retail cost structure — 56 bps plus a tick that is a percent or more of price
on anything cheap — is wide enough to swallow every effect this data can
measure. That is a finding about the market, and it is the same finding from
four independent directions.
