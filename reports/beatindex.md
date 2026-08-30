# H52 — beat the index by a mile. The diagnosis was wrong, and the way it was wrong is the finding.

*2026-08-30. `scripts/beatindex.py`. Pre-registration B1–B4 in the module
docstring, written before any arm was run. Quarterly rebalance, cost charged on
weight-space turnover at 0.56% plus each name's fraksi-harga half-spread, index
on a total-return basis (measured 1.77% yield added) over each arm's own window.*

---

## 0. The answer

**0 of 15 arms beat the index.** Best is the full universe, equal-weighted, at
**+10.58% against the index's +11.15% — −0.57%.**

---

## 1. The idea, which the repo's own numbers motivated

H43 measured the strength+calm screen beating a **random basket from its own
universe by +12.78%/yr gross** — the largest selection margin anywhere in this
project — and landing at +11.97% against the index's +11.64%. A tie.

A19 measured why an equal-weighted IDX basket should trail: a handful of
mega-caps carry the cap-weighted index and an equal-weighted basket inherits
none of that. H43's own table shows a random basket returning −5.21% to +5.90%
while the index returns +11.5% to +14.5%.

So the hypothesis was: **selection alpha ≈ +12.8%, universe penalty ≈ −12.5%,
and they cancel.** Stop paying the penalty — run the screen inside the large-cap
end, and weight by size rather than equally — and the alpha should survive.

---

## 2. It is backwards

| universe | weight | names | CAGR | index | **vs index** | random | **gross edge over random** |
|---|---|---|---|---|---|---|---|
| **all** | equal | 11 | **+10.58%** | +11.15% | **−0.57%** | +2.72% | **+12.9%** |
| all | sqrt_tv | 11 | +7.51% | +11.15% | −3.64% | +1.93% | +5.0% |
| all | tv | 11 | +6.42% | +11.15% | −4.73% | +1.46% | +4.4% |
| top 150 | equal | 9 | +6.30% | +11.15% | −4.85% | +0.57% | **+8.4%** |
| top 100 | equal | 7 | +3.60% | +11.15% | −7.56% | −1.08% | **+5.2%** |
| top 60 | equal | 5 | +2.70% | +11.15% | −8.46% | −0.12% | **+3.4%** |
| top 40 | equal | 5 | +0.62% | +11.15% | −10.53% | −0.06% | **+0.7%** |

**B1 FAILED, monotonically.** Narrowing the universe makes it *worse* at every
step: −0.57% → −4.85% → −7.56% → −8.46% → −10.53%.

**B2 FAILED.** Size-weighting loses to equal-weighting at every tier.

**B3 FAILED, and it is the whole finding.** The predicted null was that the
screen's gross edge over its own random control would be roughly *invariant* to
the tier. It is not — it **collapses**: **+12.9% → +8.4% → +5.2% → +3.4% →
+0.7%.**

> **The screen's alpha is not being spent paying a size penalty. The alpha IS a
> size effect.** Strength-plus-calm works among small and mid names and does
> essentially nothing among the largest forty. Restricting to large caps removes
> the penalty and the alpha together — and the alpha goes faster.

**B4 CONFIRMED, and then some.** 0 of 15.

---

## 3. Why this closes the route rather than narrowing it

The arithmetic is now fully determined and it is very tight:

- a random equal-weighted basket of the eligible universe returns **+2.72%**
- the index returns **+11.15%** — a structural advantage of **~8.4 points**
- the screen adds **+12.9% gross**, **~10.9% net** of its 2.02%/yr toll
- so the screen recovers almost exactly the structural gap, and lands **half a
  point short**

That is why every study of this screen keeps returning a tie: **its edge is
almost precisely the size of the handicap it is running under.** And the one
lever that would remove the handicap — moving upmarket — removes the edge
faster than the handicap.

---

## 4. The bug that nearly became the headline

The first run of this table reported **+17.63% against the index's +11.15%, a
+6.48% edge, positive in both halves, 15 of 15 arms beating the index.**

It was wrong. The check that caught it: `rebalance.py` measures the same screen
at the same frequency and got +11.67%. **Two studies of one screen disagreeing
is the signal that one of them is wrong**, so the two were run side by side on
identical inputs — and the picks were **identical on 78 of 105 rebalance bars**.

The gap came entirely from **9 quarters where the eligible universe held 20–40
names and the "portfolio" was one to three stocks.** Those land in 2003–04, the
start of IDX's largest bull run, and compound through the whole 26-year path.
The tell was that the **random control moved by the same six points** — a
selection effect cannot lift the control.

`MIN_UNIV = 40` and `MIN_BASKET = 5` now make a degenerate basket impossible.
A19 records this trap twice: the smallest cell producing the largest effect.
**A one-name portfolio is not a portfolio, and nine bars should not decide a
twenty-six-year CAGR.**

---

## 5. Where this leaves "beat the index by a mile"

It does not get there, and this was the best-motivated remaining route. The
honest state after 310 trials:

| | measured |
|---|---|
| best selection signal found here | strength+calm, **+12.9%/yr gross over random** |
| what it needs to beat, structurally | **~8.4%** equal-weight handicap + **~2.0%** toll |
| result | **−0.57% vs the index**, and 0 of 15 configurations positive |

Two things follow that are worth stating plainly. The screen **is** a real
signal — beating a matched random control by 12.9 points a year is the
strongest result in this project and it is not in doubt. And it is **not enough**,
because in this market an equal-weighted basket starts eight points behind the
benchmark and the toll takes two more.

The remaining untested instrument is unchanged: **non-price fundamentals**,
which `data/cache/fundamentals` holds for 59 names of 725. I would not expect
earnings and book value to add eight points a year on top of momentum and
volatility — but it is the only lever left that has never been pulled, and it is
the one a real desk would pull next.

---

## 6. Trials

B1–B4 is 4 registered tests; the 15-arm table is a map, not 15 hypotheses.
**Trials after H52: 310.** Bonferroni bar α = 0.05/310 = **0.00016**. Every
result here is negative, so nothing is claimed against it. The holdout was spent
at H16. Size is proxied by trailing 60-day turnover because no point-in-time
share count exists in this repo (A25: the only one available is frozen at
2024-07-10, which on a 2010 bar is look-ahead) — a turnover-weighted portfolio
is not a cap-weighted one, and that limit is real.
