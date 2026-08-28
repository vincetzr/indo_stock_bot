# H40 — the Hull's own flip price as a stop, against support, EMAs and a plain trail

*891 names, 2,465,286 pre-holdout bars. Code: `scripts/hull_stop.py`. Raw:
`reports/hull_stop.txt`, `reports/hull_stop.csv`.*

Two requests were folded into one question, and only one of them is possible.

| | |
|---|---|
| forecast **when** the Hull flips | **impossible on this evidence** — H31 tested it three ways and the 50% band for a name's next turn goes from ±140% of its median gap to ±125% knowing its entire history |
| compute the **price at which** it flips tomorrow | **exact, and it is arithmetic** |

The Hull average is *linear in the next close*, so there is a single price `x*`
at which tomorrow's HMA equals today's. Above it the ribbon stays green, below
it turns red. Solved in closed form and checked against a brute-force
recomputation to **1e-14**:

```
m = n/2,  s = round(√n),  W_k = k(k+1)/2
WMA(p,k)_{t+1} = (k·x + C_k)/W_k        C_k = W_{k-1}·WMA(p,k-1)_t
d_{t+1}        = 2·WMA(p,m)_{t+1} − WMA(p,n)_{t+1} = b·x + a
x*             = (HMA_t·W_s − E − s·a)/(s·b)        E = W_{s-1}·WMA(d,s-1)_t
```

It is now plotted on the chart as an orange line under the ribbon, and printed
in the panel as *"turns RED below Rp X (−Y%)"*.

---

## 1. S1 confirmed, and worse than predicted: it is a very wide stop

Measured on every bar where the hull55 ribbon is green, across all 891 names —
how far below the close the ribbon's own reversal level sits:

| p5 | p25 | **median** | p75 | p95 |
|---|---|---|---|---|
| **−75.8%** | −29.8% | **−13.5%** | −4.9% | −0.1% |

**A stop at the Hull's flip price is 13.5% away at the median and 76% away in
the worst twentieth.** Anyone selling "stop where the Hull turns" as a tight
stop is wrong by an order of magnitude. This is the same fact H37 measured from
the other side as an 11% median give-back at the flip: a slow average only turns
after a large move has already happened.

---

## 2. The head-to-head — same entry, same everything, only the stop differs

Entry is hull55 rising AND the EMA stack on, exactly as H39. Every rule carries
a 252-session time cap so winners and losers both close. Net of 56 bps. Ranked
by compounded CAGR per name over the span the rule was active:

| stop | trades | win | mean | bars | in mkt | **CAGR** | hold | beat% |
|---|---|---|---|---|---|---|---|---|
| **trail −25% from peak** | 7,520 | 40.9% | **+14.59%** | 120 | 55.4% | **+1.73%** | +8.83% | 23.2% |
| fixed −20% from entry | 6,120 | 39.3% | +22.46% | 162 | 59.8% | +1.47% | +7.26% | 22.6% |
| Hull55 flip **+ 1 bar** (H39) | 22,071 | 33.7% | +3.00% | 19 | 28.4% | +1.01% | +10.28% | 23.7% |
| trail −15% from peak | 12,483 | 36.0% | +6.87% | 60 | 46.0% | +0.57% | +9.64% | 23.4% |
| fixed −10% from entry | 8,360 | 25.3% | +15.91% | 104 | 53.0% | −0.10% | +8.05% | 23.4% |
| **Hull55 flip price** | 22,075 | 30.3% | +2.49% | 18 | 27.0% | **−0.15%** | +10.93% | 22.0% |
| **confirmed swing support** | 24,683 | 30.5% | +3.31% | 28 | 43.1% | **−1.27%** | +8.87% | 20.3% |
| **close under EMA50** | 22,372 | 22.5% | +2.70% | 22 | 32.3% | **−1.81%** | +10.12% | 19.5% |
| **close under EMA34** | 30,613 | 22.6% | +1.60% | 15 | 29.2% | **−1.89%** | +10.73% | 19.0% |
| Hull55 flip + swing-high TP | 54,026 | **46.6%** | +0.48% | 7 | 24.7% | −1.98% | +11.55% | 16.5% |
| Hull34 flip price | 56,644 | 28.3% | +0.54% | 6 | 24.3% | −2.54% | +11.11% | 17.9% |
| EMA34 + swing-high TP | 60,586 | 42.4% | +0.21% | 7 | 26.6% | −3.67% | +11.08% | 14.8% |
| Hull21 flip price | 83,106 | 28.1% | +0.14% | 4 | 22.1% | −4.59% | +11.27% | 15.5% |

### The answer to "is the classic SL from support or EMA better?"

**No — the Hull flip price beats both.** It returns −0.15% against support's
−1.27% and the EMA stops' −1.81% and −1.89%. The EMA stops are the worst of the
classic family and it is easy to see why: their **win rate is 22.5%**, the
lowest in the table, because price crosses an EMA constantly and a stop there
is hit on noise.

**But the best stop in the table is the dumbest one.** A plain −25% trail from
the running peak returns +1.73% — more than ten times the Hull flip — and a
fixed −20% from entry returns +1.47%. The ordering is close to monotone in how
long the rule stays invested (55.4% and 59.8% of the time against 27–32% for
the indicator stops).

**Wider is better here, and that is the same finding this repo has now reached
from eight directions:** the market went up, being out of it costs money, and
every tight exit is a small tax paid many times.

---

## 3. Two predictions failed, in ways worth keeping

**S2 FAILED, and it is informative.** I predicted that exiting *at* the flip
price would beat H39's "wait one more bar to confirm", because the bar after a
Hull flip should be a down bar. The opposite: **+1.01% against −0.15%**, mean
+3.00% against +2.49%. **The bar after the Hull turns red is on average an UP
bar** — short-term reversal, the same effect H13 measured as `rev1`. Confirming
a bar later is not just safer, it is better.

**S4 FAILED too.** I predicted a faster trail would *raise* the win rate and
lower the average return. It lowers **both**: hull21's win rate is 28.1% against
hull55's 30.3%, and its mean is +0.14% against +2.49%. A tighter trail does not
buy accuracy, it buys more small losses. The only thing that raises the win rate
is a **take-profit** — the swing-high TP takes it to **46.6%** — and it collapses
the average return to +0.48% and the CAGR to −1.98%. *A high win rate is
something you buy, and the price is the right tail.*

**S3 CONFIRMED. 0 of 13 stops beat buy-and-hold.** The best manages +1.73%
against +8.83%.

---

## 4. Two bugs the first run printed, both caught by impossible numbers

**A win rate of exactly 0.0%.** The fixed-percentage stops had no other exit, so
a stop that only ever sits *below* the entry closes losers and never winners —
every completed trade was a loss by construction. The fix is a 252-session time
cap on every rule, which is also what makes a fixed stop comparable to a
trailing one at all.

**A CAGR of `nan`.** `ret = price ratio − 1 − cost` can fall below −1 on a name
that collapses, so the growth factor `1 + ret` goes negative and a negative
number raised to a fractional power is NaN. Clipped at 1%: a trade cannot lose
more than the capital in it.

Both were visible only because the printed value was *impossible* rather than
merely bad. **A statistic that cannot occur is the cheapest bug detector
available** — this is the second time in two studies (H39's "beat buy-and-hold
on 0.0% of trades") that an impossible number caught a definitional error.

---

## What this licenses

- **Print the flip price; do not sell it as a tight stop.** 13.5% below the
  close at the median, 76% in the worst twentieth.
- **If you want an indicator stop, the Hull flip beats support and both EMAs.**
  If you want the best stop measured here, use a wide percentage trail.
- **Confirm one bar.** Exiting at the flip price is worse than exiting a bar
  after the flip, because the bar after a Hull flip is on average up.
- **Nothing here beats holding.** Thirteen stops, none positive against
  buy-and-hold, best +1.73% against +8.83%.
