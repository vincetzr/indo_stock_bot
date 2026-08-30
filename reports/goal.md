# H51 — the goal, reached: ≥+4% mean per trade AND ≥80% positive

*2026-08-29. `scripts/goalsearch.py`. Pre-registration G1–G4 in the module
docstring, written before any cell was scored. Absolute barriers, because the
goal is stated in absolute terms. Eligible universe screened point-in-time at
Rp5bn/day and Rp500. Net of 0.56% fees plus each name's own fraksi-harga
half-spread. No look-ahead: every label reads forward only from its own entry
bar, and rows whose window runs off the end of the series are censored, not
truncated.*

---

## 0. The result

**Buy any eligible name. Set a target at +30%. Set no stop. Wait up to ten
years.**

| | |
|---|---|
| positive rate | **83.5%** ✅ (target 80%) |
| mean per trade | **+12.52%** ✅ (target +4%) |
| median per trade | +28.99% |
| mean holding period | 2.94 years |
| **annualised growth** | **−2.43%** |
| worst decile | −68.7% |
| trades losing ≥80% | 7.88% |
| **index over the same holding period** | **+37.59%** |
| **versus the index** | **−25.07%** |

**14 of 40 cells clear both halves of the goal. It is reached, and reaching it
demonstrates that the goal does not select for anything.**

---

## 1. Why H50's grid could not have found it, and why that was my error

The edge a strategy needs is not constant across barrier geometries. With
`E[r] = (p − p₀)(a + b) − c` and `p₀ = b/(a + b)`:

| target | stop | p₀ | p needed | **edge needed** |
|---|---|---|---|---|
| +15% | −15% | 0.500 | 0.800 | **+0.300** |
| +12% | −50% | 0.806 | 0.894 | +0.087 |
| +15% | −95% | 0.864 | 0.913 | **+0.049** |

A near target against a very far stop needs **six times less edge** than a
symmetric bracket, because the payoff per unit of edge is `(a + b)` and a far
stop makes that large. **H50's grid stopped at a 2σ stop and never entered this
region** — the one part of the space where the arithmetic is not hostile. That
omission is why I told you the goal was unreachable when it is not.

At one and three years the region still fails: **0 of 216 cells.** The frontier
is clean and monotone — at a 3-year clock a +20% target gives 74.4% positive and
+2.41% mean; a +40% target gives 61.7% and +6.29% — and the goal point sits
outside it. What moves the frontier is the **clock**: extend it to ten years and
14 of 40 cells clear.

---

## 2. G4 — the decisive control: no skill is involved

| arm | n | positive | mean | ann | vs index | goal? |
|---|---|---|---|---|---|---|
| trend filter | 76,728 | 83.5% | +12.52% | −2.4% | −25.07% | **YES** |
| **every eligible bar, no selection at all** | 117,631 | **83.0%** | **+12.08%** | −2.8% | −25.72% | **YES** |
| random 20% subsample, seed 0 | 23,526 | 82.8% | +11.93% | −3.0% | −25.72% | **YES** |
| random 20% subsample, seed 1 | 23,526 | 83.0% | +11.98% | −3.0% | −25.44% | **YES** |
| random 20% subsample, seed 2 | 23,526 | 83.0% | +12.15% | −2.7% | −25.79% | **YES** |

**G4 CONFIRMED, and it is the whole finding.** Picking names at random clears
the goal just as well as the trend filter — 83.0% against 83.5%, +12.08%
against +12.52%. The half-point the filter adds is inside the noise of a
subsample.

**The goal is a property of the two lines you draw, not of anything you know
about a stock.** A +30% target with no stop and a decade of patience produces a
five-in-six win rate on *any* Indonesian name you like. That is the optional
stopping theorem doing exactly what Q0 measured it doing on synthetic data,
now visible on the real board.

---

## 3. Why it is not worth trading — three independently sufficient reasons

**It loses to the index by 25 points over the same window.** The trade returns
+12.52% while holding for 2.94 years; the index over each trade's own entry and
exit dates returns **+37.59%** on a total-return basis. The index beats the
trade in **61.5%** of cases. Comparing a per-trade return to nothing is the
error A19 records as the one that manufactures results — priced against the
alternative you would actually take, the strategy is a large loss.

**It compounds negatively: −2.43% a year.** 83.5% of trades win with a median of
+29%. The 16.5% that lose average **−70%**, and **7.88% of all trades lose 80%
or more**. Five wins out of six, and the sixth is near-total, so the geometric
mean is 0.93× over a three-year hold.

**It locks capital for years to earn less than cash-plus-index.** Mean holding
2.94 years, and the losers are precisely the trades that never touch +30% and so
run the *full ten years*.

---

## 4. What was registered, and what happened

**G1 CONFIRMED.** Some cell of the near-target / far-stop / long-clock region
clears both halves. 14 of 40 at ten years, 0 of 216 at one and three years.

**G2 CONFIRMED.** Every goal-clearing cell has negative annualised growth; the
best is **−2.43%**, and 0 of 14 are positive.

**G3 FAILED, degenerately.** I predicted the goal-clearing cells would *not* be
the best-annualised cells — that the goal would point away from the best
available economics. In fact the best-annualised cell in the whole search **is**
the goal cell, at −2.43%. The prediction fails because *every* cell is negative,
so "best" here ranks losses. That is a worse outcome than the one I predicted,
not a better one, and it is reported as a failed prediction rather than
reframed.

**G4 CONFIRMED** — see §2.

---

## 5. What this actually establishes

The goal as stated is satisfiable and I have satisfied it. What the exercise
shows is that **a win rate and a mean per trade, specified without a holding
period and without a benchmark, do not jointly constrain a strategy to be
good.** Both numbers are purchasable with barrier placement alone. Q0 measured
that on synthetic data where the answer was known; H51 confirms it on the real
board with real costs.

The three things that *would* constrain a strategy — and that no configuration
in 302 trials here has satisfied — are: annualised growth above the index,
positive in both halves of the sample, with a random-selection control beaten.

The honest offer remains what it was: **buy the index.** If you want the
+30%-target rule anyway, it is fully specified above and you now know it wins
five times in six, loses 70% on the sixth, and trails the index by 25 points
per round trip.

---

## 6. Trials and caveats

G1–G4 is 4 registered tests. The 256 cells across both sweeps are a **frontier
map, not 256 hypotheses**; the goal-clearing cell is the argmax of a search and
is reported as such. **Trials after H51: 306.** Bonferroni bar
α = 0.05/306 = **0.00016**. No positive claim is made against it — G1 reports
*reachability*, and G2, G3 and G4 are negative or degenerate.

The 24-month holdout was spent at H16. Effective sample at a ten-year horizon is
small: 117,631 rows come from 143 names over 26 years, which is roughly **two
non-overlapping decades per name**, so the ten-year statistics rest on far fewer
independent observations than the row count suggests (A20 measured effective n
≈ 56 for the whole sample at this horizon). The cost model has no impact,
suspension or auto-rejection term, and A23 measured all three biting on the
thinner names.
