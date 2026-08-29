# H46 — the best allocation this research supports, priced against the index

*247 names, 59 quarterly rebalances, 2010-12 → 2026-04, on the H44 tradeable
universe (close ≥ Rp500, 60-bar turnover ≥ Rp10bn). Code:
`scripts/allocator.py`. Raw: `reports/allocator.txt`, `reports/allocator.csv`.*

Asked for "the most comprehensive algorithm that works best for trading". No
such thing exists in this data — 158 exit configurations, none beat holding —
so this builds the only thing left standing: a **selection** and an
**allocation**, with **no exits at all**, priced against buying the index.

Every component is here because a measurement put it here: the universe gate
from H44, the score from H26 (the one cell that ever cleared Bonferroni) plus
H27 (which confirmed it out of sample by an unrelated method) plus `mom12_1`
(H44's only survivor on a tradeable universe), quarterly rebalancing from H43,
and no stops because A18 measured per-position stops making portfolio drawdown
*worse*.

---

## 1. The result

| sleeve | CAGR | vol | maxDD | early | late | vs index | random sleeve |
|---|---|---|---|---|---|---|---|
| 0% (all index) | +6.83% | 18.1% | −28.3% | +10.33% | +3.55% | — | +6.83% |
| 20% | +7.52% | 15.4% | −26.9% | +10.67% | +4.56% | +0.70% | +6.02% |
| 40% | +7.88% | 14.6% | −26.8% | +10.79% | +5.15% | +1.06% | +4.70% |
| **60%** | **+7.91%** | 16.0% | −27.2% | +10.68% | +5.30% | **+1.08%** | +2.90% |
| 80% | +7.61% | 19.1% | −28.8% | +10.32% | +5.04% | +0.78% | +0.65% |
| 100% (all sleeve) | +6.96% | 23.2% | −32.6% | +9.71% | +4.37% | +0.14% | **−2.03%** |

---

## 2. THE SELECTION IS REAL, AND IT IS THE LARGEST SUCH MARGIN IN THE PROJECT

The same machine — same universe, same book size, same schedule, same toll —
selecting at **random** returns **−2.03%** a year. The real score returns
**+6.96%**. That is **+9.00% a year of pure selection**, and it is the biggest
gap between a rule and its own random control anywhere in this repository.

**G3 CONFIRMED.** The signal is not an artefact of the machinery.

---

## 3. AND IT CONVERTS TO NOTHING AGAINST THE INDEX

| sleeve weight | annualised mean difference vs index | t | 95% CI (moving-block) |
|---|---|---|---|
| 20% | +0.20% | **+0.13** | [−1.63%, +1.87%] |
| 40% | +0.40% | +0.13 | [−3.21%, +3.76%] |
| 60% | +0.60% | +0.13 | [−4.71%, +5.41%] |
| 100% | +1.01% | +0.13 | [−8.34%, +9.52%] |

**t = +0.13 at every weight.** The edge over the index is indistinguishable
from zero, and the interval is an order of magnitude wider than the point
estimate. **G1 CONFIRMED** — the 100% sleeve ties the index at +0.14%.

**And the half-split kills what is left of the 100% sleeve:** early
**−0.61%**, late **+0.82%**. NOT BOTH.

---

## 4. G2 FAILED, AND THE FAILURE IS ARITHMETIC RATHER THAN ALPHA

I registered that the best blend would sit at or near 0% sleeve. It sits at
**60%**. But that is not a selection result — it is variance reduction:

| | index | 60% blend | 100% sleeve |
|---|---|---|---|
| volatility | 18.1% | **16.0%** | 23.2% |
| arithmetic edge over index | — | +0.60%/yr | +1.01%/yr |
| **CAGR** edge over index | — | **+1.08%/yr** | +0.14%/yr |

The CAGR edge exceeds the arithmetic edge by ~0.48%/yr at the 60% blend, and
the volatility drag `σ²/2` accounts for ~0.36% of that (1.64% at the index's
18.1% against 1.28% at 16.0%). **Mixing two imperfectly-correlated assets
raises the geometric mean even at zero arithmetic edge.** That is portfolio
arithmetic, available from any two assets, and it is not stock selection.

---

## 5. What the gate cost, and it is worth stating

H44's liquidity gate is what makes this study honest — and it cuts the sample
hard. Only **3 to 11 names** cleared Rp 10bn/day before 2004, so the panel
starts in **2010-12**, the book is **8 names**, and the index returns only
**+6.83%** here because IDX's great run (2003–2007) is outside the window.
59 quarterly periods with overlapping selection is a small effective sample,
and every component of the score was chosen because it worked in it.

---

## What this licenses

- **The selection works.** +9.00%/yr over its own random control, which is the
  strongest such result here and vindicates H26/H27.
- **It does not beat the index.** t = +0.13, CI [−4.71%, +5.41%] at the best
  blend, and the 100% sleeve fails the half-split.
- **The blend's advantage is diversification, not alpha**, and you can have
  that from any two uncorrelated assets without a signal.
- **Buy the index remains the defensible position.** A 20–40% satellite in
  this sleeve is defensible on variance grounds and cannot be claimed to add
  return.
