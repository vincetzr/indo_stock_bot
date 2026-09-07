# H59 / H60 — sector rotation and the regime engine

*2026-09-07. `scripts/sector.py`, `scripts/regime.py`, `tests/test_sector.py`,
`tests/test_regime.py`. Roadmap stages 6 and 5. Both registered in their
scripts' docstrings before any cell was scored.*

---

# H59 — sector rotation (§8)

## 1. What the frozen map actually costs — measured, not assumed

A14 found the IDX-IC sector map and recorded that it is **frozen at
2024-07-10**. Nobody had measured what the freeze costs, and the obvious worry
is fatal if true: a map naming only companies listed in 2024 cannot name a
company that delisted in 2012, so a study on the covered names would be
survivorship-biased by construction.

| | |
|---|---|
| panel names | 968 |
| mapped | 934 (96.5%) |
| still trading | 828, of which **798 mapped (96.4%)** |
| **no longer trading** | 140, of which **136 mapped (97.1%)** |
| bars covered | 2,774,156 of 2,788,474 (**99.5%**) |

**Dead names are covered as well as live ones.** The worry does not bite, and it
was worth an hour to find out rather than a caveat on every number. The map
evidently predates the current listing set by enough to carry recent
delistings.

Two limits that do stand:

- **IDX-IC launched 2021-01-25**, replacing JASICA. 437 of 666 sector-marks
  below predate it, so a label on a 2010 bar is a **backward projection of a
  taxonomy that did not exist**. That is not look-ahead in H57's sense — a
  company's industry does not respond to its own returns — but it is real.
- **The `shares` column is dropped at the source**, not merely unused. A25 and
  H57: a 2024 share count on a 2010 bar is look-ahead, and Indonesian rights
  issues move it by up to 41×. A test fails if that guard is removed.

## 2. R1 — does sector momentum rank next-quarter sector return?

94 quarterly marks, 11 sectors, 666 sector-marks, 2002-02-11 → 2026-04-28.

| | |
|---|---|
| rank IC | **+0.0798** |
| date-block null (500 draws) | +0.0077 ± 0.0480 |
| **z** | **+1.50** |
| half-split | early +0.1146, late +0.0329 — same sign |
| **POWER** | **114 marks needed** to tell this IC from zero at \|t\| = 2; **75 observed** |
| Bonferroni bar at 354 trials | 0.00014, needs \|z\| > 3.63 — **does not clear** |

**R3 is confirmed exactly and it is the honest headline: the sample is too
narrow to answer R1 either way.** The point estimate is respectable and the
same sign in both halves; the study is under-powered by roughly a third before
any multiple-testing correction is applied. Stated before the IC, per A31.

The null is a **whole-date reassignment**, not a within-date shuffle: eleven
sectors on one day share the market's move almost entirely, so shuffling inside
the date would leave most of the dependence intact and every z would be
inflated. A17 and A25 record the same error one level down.

## 3. R2 — does a sector tilt improve H26's screen?

Common window 2012-09-21 → 2026-07-24, scouted across all five arms, six
rebalance phases.

| arm | CAGR | its index | **excess** | beats index |
|---|---|---|---|---|
| strength+calm, no tilt | +11.07% | +5.24% | **+5.83%** | 6/6 |
| + top 3 sectors by momentum | +5.96% | +4.11% | +1.85% | 6/6 |
| + top 5 sectors by momentum | +9.98% | +4.54% | +5.44% | 5/6 |
| + BOTTOM 3 sectors (inverted) | +1.69% | +3.30% | −1.61% | 3/6 |
| + 3 RANDOM sectors (control) | +1.84% | +5.04% | −3.20% | 1/6 |

**R2's predicted null is CONFIRMED: no tilt arm beats the untilted screen.**

**But tilt-versus-no-tilt is the wrong comparison and the matched one says
something different.** Restricting to three sectors concentrates the book
whatever chooses them, so a top-3 arm losing to the untilted screen measures
*concentration*. Holding the number of sectors fixed and varying only the
choosing:

| at 3 sectors | excess |
|---|---|
| **momentum-chosen** | **+1.85%** |
| bottom-chosen | −1.61% |
| **random-chosen (control)** | **−3.20%** |
| momentum − random | **+5.05%** |
| top − bottom | **+3.46%** |

**Sector momentum does carry information — about five points a year against a
random sector choice at the same concentration — and it is worth less than the
diversification it destroys.** That is A38's shape a second time: the lever that
would exploit the effect costs more than the effect is worth.

## 4. The error this study committed and caught

A first run printed the untilted screen at **+12.01%** against a top-3 tilt at
**+6.65%** — with their index benchmarks at **+8.71%** and **+4.63%**. A
benchmark moving four points between arms is proof the windows differ: a tilt
arm cannot trade until enough sectors carry five names, so it starts later.
**The difference was the calendar, not the tilt.** A19 records this error class
three times; this is the fourth. Two guards now: a scout pass fixes one window
for every arm, and the excess over each arm's own index is printed beside the
raw CAGR.

---

# H60 — the regime engine (§7 / CLAUDE.md §10)

4,527 sessions, 2007-10-26 → 2026-09-04, twelve backward-looking state
features: IHSG realised vol, trend and drawdown, breadth, and the momentum of
DXY, USDIDR, UST 10y (**differenced — it is a rate**, A13), S&P, Brent and
copper, plus USDIDR and S&P realised vol. The window starts in 2007 because
`BZ=F` does; that is the binding series.

## 5. G1 — HDBSCAN, the method that can say "there are none"

| min_cluster_size | clusters | noise | episodes |
|---|---|---|---|
| 90 (2%) | 2 | 0.1% | 9 |
| **226 (5%)** | **0** | **100.0%** | 1 |
| **452 (10%)** | **0** | **100.0%** | 1 |

**G1's predicted null is CONFIRMED.** At any minimum size worth calling a
regime, HDBSCAN labels the entire sample noise. A10's rule is what makes this
sayable: a method forced to return *k* cannot tell you there are none, so both
are run and HDBSCAN's answer is reported first.

GMM forced to *k*, for the contrast:

| k | BIC | episodes | days/episode |
|---|---|---|---|
| 2 | 88,118 | 13 | 348 |
| 3 | 76,744 | 22 | 206 |
| 4 | 71,855 | 40 | 113 |
| 5 | 66,077 | 51 | 89 |
| 6 | **60,006** | 93 | 49 |

BIC falls monotonically to the largest *k* tried, which is what BIC does on
**autocorrelated** data: it counts each of 4,527 days as an independent
observation. It is printed and not obeyed.

## 6. G3 — the unit of independence, stated before the effect

| k | contiguous episodes | over 18.9 years |
|---|---|---|
| 2 | **13** | 0.7/yr |
| 3 | 22 | 1.2/yr |
| 6 | 93 | 4.9/yr |

**G3 CONFIRMED.** An episode is the observation, not a day. 4,527 days of two
regimes is thirteen observations.

## 7. G2 — and it FAILED, which is the outcome registered as the valuable one

The statistic is H26's screen edge — mean log 63-session forward return, screen
minus rest — computed separately inside each regime. The null **rotates** the
regime label series circularly rather than shuffling it: regimes are long
contiguous runs, and a day-level shuffle manufactures evenly-mixed
pseudo-regimes whose spread is far too small, inflating every z.

**Every k is reported, because trying two values and quoting the louder is a
search of size two.**

| k | spread | rotation null | z | sign inverts | clears 3.63 |
|---|---|---|---|---|---|
| 2 | +0.0074 | +0.0241 ± 0.0203 | −0.82 | no | no |
| **3** | **+0.1853** | +0.0507 ± 0.0334 | **+4.03** | **yes** | **YES** |
| 4 | +0.2157 | +0.0707 ± 0.0414 | +3.50 | yes | no |
| 5 | +0.2300 | +0.0931 ± 0.0498 | +2.75 | yes | no |
| 6 | +0.2130 | +0.0943 ± 0.0441 | +2.69 | yes | no |

At k=3 the inverting regime is stark: 258 days in **5 episodes**, screen
+0.0038 against rest **+0.1557** — the screen underperforms by fifteen points of
log return over a quarter.

**AND IT IS ONE HISTORICAL WINDOW.** The extreme regime's largest episode, at
every k where the sign inverts:

| k | largest episode of the inverting regime | spread after dropping it |
|---|---|---|
| 3 | 2008-09-04 → 2009-04-22, 134 d | +0.1853 → **+0.0996** |
| 4 | 2008-10-06 → 2009-03-12, 92 d | +0.2157 → +0.1466 |
| 5 | 2008-10-06 → 2009-03-12, 92 d | +0.2300 → +0.1444 |
| 6 | 2008-09-09 → 2009-03-12, 106 d | +0.2130 → +0.1177 |

**One window, four clusterings.** The "regime" effect is a label on the global
financial crisis and its rebound, and dropping that single episode removes
roughly half the spread every time. That drop-largest check exists because
H11's headline was carried by one thin year and A8 records it earning its place
twice; this is the third.

**What is true and what is not.** It is true and economically coherent that a
strength-plus-calm screen underperforms violently in a sharp post-crash rebound
— beaten-down names rip and a screen that avoids them by construction misses
it. It is **not** established that this is a regime effect in any usable sense:
one clustering of five clears the bar, its z is +4.03 against a threshold of
3.63, and the cell is five episodes of one crisis. CLAUDE.md §10 warns by name
that "any regime split producing beautiful results on three episodes is almost
certainly overfit". **This is a registered failure of a predicted null, and no
trading arm is built on it.**

---

## 8. What neither study establishes

- **The holdout was spent at H16.** Every number here is in-sample.
- **Neither result clears the 354-trial Bonferroni bar**, except k=3's regime
  spread, which does so on one clustering out of five and on one crisis.
- **No cost model enters H60.** Its statistic is a gross forward return, and
  A38's rule holds: fee and spread are separate quantities.
- H59's arms carry the standard cost model (0.56% fee, half a fraksi-harga
  tick) but are measured at six rebalance phases on one panel.
