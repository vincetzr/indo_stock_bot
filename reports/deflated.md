# H58 — the deflated Sharpe ratio §11 has demanded since day one

*2026-09-07. `src/idxbot/metrics.py`, `scripts/deflate.py`,
`tests/test_metrics.py`, `tests/test_deflate.py`, `reports/deflated.json`.
Registered in the script's docstring before the numbers existed.*

---

## 1. The gap, and it is not the six missing ratios

`docs/PLATFORM_ASSESSMENT.md` listed Sharpe, Sortino, profit factor,
expectancy, exposure and tail risk as uncomputed. That is half wrong —
`portfolio._stats` has computed Sharpe and Sortino all along — and the half
that is right is the less important half.

**The real gap: CLAUDE.md §11 names the deflated Sharpe ratio as
non-negotiable, `hypotheses.md` opens by saying the trial count exists *so
that* it can be computed, and across fifty-seven hypotheses and 350 trials it
was never computed once.** A running count that never enters a statistic is
bookkeeping, not a correction.

**It is a different correction from every null in this repo.** Seven times a
clustered permutation null has decided a result here, and it asks one question:
does the *label* carry information? It cannot ask whether a strategy is the
best of 350 attempts, because it is only ever handed one attempt. The deflated
Sharpe is the correction for the **search**. The two are not substitutes and
this project had only ever applied one of them.

---

## 2. What `src/idxbot/metrics.py` computes, and the three things that make it
## different from the textbook version

The §39 set is there — Sharpe, Sortino (against a stated MAR, using shortfall
over *all* periods rather than the standard deviation of the losers), profit
factor, expectancy, exposure, VaR, CVaR, Ulcer, skew, kurtosis. Three
departures matter:

**Overlapping windows.** Almost every statistic in this project is built from
k-session forward returns sampled every session, so nominal n is ~k times the
independent count. `effective_n` deflates it and everything consuming n takes
the deflated value; `lo_factor` replaces the naive √q with Lo (2002)'s
autocorrelation-aware scaling. A test asserts the adjustment can only make a
Sharpe **smaller** — a correction that can flatter a result is a bug wearing a
statistic's clothes.

**Skew and kurtosis are not decoration.** A13 measured kurtosis from 10 to
**2,800** on the series this repo trades. `psr` is the version that takes both,
and on a sample like that it is a different number, not a refinement.

**A ratio is not a benchmark.** `describe()` prints the **power statement
first** — how many periods it would take to tell this mean from zero at |t| = 2
— because A31 found that the most useful line in its study and A19 records
writing a power statement as an effect statement as its own error class.

One bug the tests caught: `tail()` built the drawdown path from the return
series alone, so the running maximum started **below par** whenever the first
period was a loss and a series opening −30% reported a maximum drawdown of
**exactly zero**. A30's rule again — a statistic that cannot occur is the
cheapest bug detector available.

---

## 3. The one input that could have been faked

DSR needs the **variance of the Sharpe ratios across trials**. A plausible
constant is easy to type and would make the deflation look measured when it was
assumed. It is estimated instead from the random-selection control arms
`bhbench` already runs through identical machinery — same universe, same
calendar, same costs, only the names drawn at random — each producing an equity
path and therefore a Sharpe.

**And that estimate is a LOWER BOUND, which changes how everything below
reads.** A control draw varies only *which names are bought*. This repo's 350
trials varied the whole hypothesis: broker flow, investor class, price
features, exit rules, horizons, timing rules. Dispersion across that family is
necessarily wider. A lower bound on `sr_variance` is an **upper bound on the
DSR**, so every number in §4 is the most flattering reading available.

The script's docstring originally asserted the control spread "IS ... exactly
the quantity DSR wants". That sentence is wrong; it is kept, marked, rather
than deleted, and a test pins both the retraction and the correction.

---

## 4. The result

Panel 2000-07-13 → 2026-09-04, 741 names, offset 0, 12 control draws per arm.

| arm | Sharpe | PSR | SR\* | DSR | kill@ | ctl DSR |
|---|---|---|---|---|---|---|
| own everything, quarterly | +0.43 | 0.976 | +0.25 | 0.796 | already | 0.748 |
| momentum top 10, quarterly | +0.46 | 0.992 | +0.25 | 0.865 | already | 0.748 |
| low vol top 10, quarterly | +0.53 | 0.992 | +0.25 | 0.901 | already | 0.748 |
| strength+calm 10, quarterly | +0.69 | 0.998 | +0.25 | 0.968 | 2,800 | 0.748 |
| **strength+calm sticky, quarterly** | **+0.73** | 0.999 | +0.25 | **0.974** | 11,200 | 0.748 |
| sticky tight buffer, quarterly | +0.61 | 0.995 | +0.25 | 0.936 | already | 0.748 |
| strength+calm 10, annual | +0.63 | 0.994 | +0.20 | 0.956 | 1,400 | 0.585 |

**D2 — the predicted null — PASSED.** A random basket run through the identical
deflation lands at **0.748** and 0.585. The deflation bites; without that, none
of the column above would be readable.

**D4 — and this is the finding.** DSR of the best arm as a function of the
assumed trial dispersion:

| sr_variance | sd of trial SR | SR\* | DSR |
|---|---|---|---|
| **0.0072** (control draws — lower bound) | 0.08 | +0.25 | **0.974** |
| **0.0127** (the 7 families run here) | 0.11 | +0.33 | **0.947** |
| 0.0500 | 0.22 | +0.66 | 0.608 |
| 0.1000 | 0.32 | +0.93 | 0.200 |
| 0.2500 | 0.50 | +1.47 | 0.001 |
| 1.0000 | 1.00 | +2.94 | 0.000 |

**It survives 1 of 7 dispersion assumptions, and the one it survives is the
single most flattering one available.** Move to the very next estimate in
hand — the spread of the seven *strategy families* actually run in this
script — and it reads 0.947 and fails. A spread of 0.11 in trial Sharpe is
narrower than 350 hypotheses across six instruments would plausibly produce.

**D1 — CONFIRMED in substance, failed on its literal wording.** I predicted the
surviving arms would *not* clear the deflated bar at 350 trials. Three of seven
do, on the lower bound. They do not on anything wider. Logged as predicted-in-
substance rather than reframed as a pass.

**D3 — FAILED.** I predicted `trials_to_kill` would be within an order of
magnitude of 350. It is **11,200** — thirty-two times the repo's own count.
That is a real miss, and it is the same fact as D4 seen from the other side:
on the flattering dispersion the result is very hard to kill by counting
trials, and very easy to kill by admitting the trials were not all alike.

---

## 5. What this does not say

- **`sr_variance` is not recoverable from this repo's log**, because no Sharpe
  was ever stored per trial. That is now a concrete, cheap thing to fix going
  forward — `signal_store` already records outcomes, and a Sharpe per logged
  hypothesis would make the deflation a measurement rather than a sweep.
- **A DSR is a probability about the true Sharpe of *this* series**, not a
  claim that the series would repeat.
- **The holdout was spent at H16**, so every number here is in-sample, and A23
  applies in full to third-party money.
- The arms are measured at **rebalance offset 0**. A39 records offset 0 as a
  free parameter nobody chose, and three strategies passed a harness at offset
  0 and at no other phase. This study inherits that exposure; the phase sweep
  belongs in any follow-up.

---

## 6. What changed in the repo

`src/idxbot/metrics.py` is now the one place these are computed, with 38 tests
including three positive controls: `psr` reduces exactly to the normal
one-sided test on a normal sample, `expected_max_sharpe` matches a simulated
order statistic to within 5%, and every correction is asserted to move the
answer in the pessimistic direction.

`scripts/deflate.py` reproduces the table above from the panel in about a
minute, writes `reports/deflated.json`, and prints its own verdict so the
numbers and the reading cannot drift apart.
