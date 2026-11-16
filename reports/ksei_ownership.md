# KSEI ownership, running trade, and the free BEI/OJK sources

**Date:** 2026-11-15
**Verdict:** the depository publishes the account-type split CLAUDE.md §12 said
was unavailable, and retail flow measured with it **does not lose**.
*Corrected 2026-11-16 after an independent review (§4.7): the retail tilt's p
is ≈ 0.012 under a null that respects dependence across months, not 0.0069,
and two attributions were wrong. No verdict changed.* H64's
registered prediction (what retail buys underperforms) failed with the sign
reversed. Neither side of retail flow is a long-only trade after cost.
Running trade: nothing new is free; the one new API found has shut down.

Reproduce:

```bash
python3 scripts/ksei_collect.py      # the archive, ~5 min cold, no network warm
python3 scripts/ksei_flow.py         # H64 as registered (amendment A), ~40 s
python3 scripts/ksei_flow.py --registered   # the pre-amendment spec
python3 scripts/ownership.py BBCA --market  # the descriptive report
```

Pre-registration: `hypotheses.md` H64, commit `f5b1ac5`, before any KSEI
number was joined to a price; amendment A, commit `979f4a8`, also before.

---

## 1. What was asked, and what each part turned out to be

| ask | what it is | answer |
|---|---|---|
| "KSEI can show ownership of stocks" | three different things: AKSes (your OWN holdings), the monthly investor-type composition of every security, and named holders above 1% | the second is free, reachable and now ingested; the third is published monthly via IDX (403 from here); the first is yours only |
| a connection to running trade, via Mandiri or another platform | a licensed real-time feed | **not free, and not obtainable without a licence or your credentials** — §4 |
| use all free BEI and OJK information | a source sweep | §2; one source materially new, two useful for checks, the rest aggregate |

---

## 2. The source sweep — tested, not researched

Every row was an actual request on 2026-11-15.

| source | status | shape | history | used |
|---|---|---|---|---|
| **KSEI `BalanceposEfekYYYYMMDD.zip`** | **200** | every security, depository-held units by 9 investor types × local/foreign, plus units on issue and month-end price | **2018-01 → 2026-10, 106 files, none missing.** Page lists 36; older ones are served but unlinked | **ingested, H64** |
| KSEI `StatisEfekYYYYMMDD.txt.zip` (master file) | 200 | ISIN, issuer, status, listing date, units, scripless %, sector | from 2019-01 (2018 candidates 302) | not built |
| KSEI announcements (`TICKER_DIV_YYYYMMDD_ID.pdf`) | 200 | corporate-action schedules: dividends, splits, rights | current | not built — would replace inferred dividend steps with stated dates and amounts |
| KSEI CMS API `webcms.ksei.co.id` | proxy 502 | the listing's back end | — | not needed: the files are at a stable path |
| IDX `idx.co.id` (incl. the >1% holders, HSC list, free-float pages) | **403** Cloudflare | — | — | not touched; `docs/FULL_REKAP.md` §3 explains why this repo does not go through the check |
| OJK weekly *Bid-Ask Spread & Turnover* xlsx | 200 | MARKET-WIDE average quoted spread and turnover ratio, daily within the month | annual from 2015 | used once, as a cost-model check (§5) |
| OJK weekly *Emisi Efek & Aksi Korporasi* xlsx/pdf | 200 | issuance and corporate-action overview | weekly | not built |
| archive.org (to date KSEI's 2018 files) | connection reset, twice | — | — | the gap is stated, §3.1 |

**KSEI's terms.** *Penafian Umum* and *Ketentuan dan Kebijakan* (read
2026-11-15) disclaim accuracy and govern K-CASH/AKSes; neither restricts the
public statistics, and there is no counterpart to IDX's explicit anti-crawling
clause. The host sits in `data.ksei_allowed_hosts` so you can withdraw it. **This
repository is public**, so the holdings are gitignored and only a hash manifest
(`data/reference/ksei_manifest.csv`) is tracked.

---

## 3. What the file actually measures — three findings before any return

### 3.1 The point-in-time trap: backfills

Each zip carries its member's creation stamp. From 2019-03 on it is 06:00,
1–12 days after the state date (median 3). But the **14 files for 2018-01 …
2019-02 were all created 2020-04-13**, and the 2020-12 file on **2021-10-19**.
Whether that data was public at the time cannot be established from here. The
test therefore treats every file as available only from its own stamp, which
makes the first usable signal April 2019 and drops two months around the 2020
regeneration.

### 3.2 "Held" is not the shareholder base — the controlling block is missing

`held / sec_num` is below 0.90 for **46.4%** of equity rows, and the gap is not
scrip: BBCA **0.4255**, TLKM **0.4786**, BBRI **0.4681**, UNVR **0.1453** —
one minus the parent's or the state's stake, to within a point. Blocks move in
and out of the balance (BBCA 0.9995 → 0.4255 by 2022; GOTO 0.043 at listing →
0.9551 as lock-ups expired). The registered coverage ≥ 0.90 filter rested on
the scrip assumption and would have dropped the most liquid names; **amendment
A removed it before any return was joined**, and the registered version is
still printed (§4.3).

### 3.3 Type-to-type transfers read as flow

In October 2026 BYAN's individual holding fell by **10,751,166,170** shares as
its corporate holding rose by **10,751,301,170** — a founder re-registering a
stake. Read as flow, it is retail selling **Rp 98 trillion** in one month, and
three names supplied 88% of that month's absolute retail flow market-wide.
`ksei.transfer_flags` marks a month where the individual change is mirrored by
another local type within 10% and exceeds 1% of the balance: 2,511 name-months,
2.84%. Found after H64 ran, so it is a robustness line, not a filter (§4.2).

---

## 4. H64 — does retail flow lose?

### 4.1 The registered test (amendment A)

Signal: individual-type net flow over month m ÷ the previous balance.
Decision: first session after the file's stamp day. Label: total return to 63
own-bars later. Controls: 12-1 momentum, 21-session reversal, liquidity,
volatility, log market cap, the retail share level, sector. 10,000
within-month permutations. **11,130 in-sample rows, 62 months
(2019-05-03 … 2024-08-02), median 190 names a month; the 24-month holdout
(6,664 rows) is untouched.**

| | H64 retail flow | H64b foreign flow | P0 parity (null) |
|---|---|---|---|
| predicted | **−** | **+** | 0 |
| IC, k = 63 | **+0.0275** | −0.0167 | +0.0137 |
| HAC t | +2.58 | −1.75 | +0.81 |
| permutation p, registered within-month null (bar 0.000136) | 0.0069 | 0.1075 | 0.1555 |
| **permutation p, names relabelled consistently across months** (review, §4.7) | **0.0120** | **0.1413** | **0.3535** |
| k = 21 / 126 | +0.0271 / +0.0256 | −0.0187 / +0.0143 | — |
| halves | +0.0238 (t 1.28) / +0.0313 (t 2.52) | −0.0114 / −0.0219 | +0.0455 / −0.0181 |
| liquidity terciles | +0.037 / +0.012 / +0.033 | −0.008 / +0.010 / −0.042 (flips) | flips |
| drop largest year | 2020: +0.0240 (t 1.96) | 2022: −0.0112 | — |
| registered long-only leg, net | hold what retail SOLD: **−7.9%/yr** | hold what foreigners BOUGHT: −3.5%/yr | — |
| lag 5 sessions | +0.0258 (t 2.34) | −0.0128 | — |
| **gate** | **FAILS** — wrong sign | **FAILS** | **silent — pipeline clean** |

### 4.2 After the result, labelled as such

- **Type-to-type transfers** (306 rows): scored out of the IC but still in the
  per-month fit, retail reads **+0.0280 (t 2.80)**; dropped from the fit as
  well, **+0.0203 (t 1.87)**. The transfers carry none of the tilt themselves
  — their extreme values move the OLS coefficients for every other name.
  ~~A quarter of the retail tilt was re-registration.~~ *(Retracted by the
  review, §4.7: that read a leverage effect as the rows' own contribution.)*
- **Neutralisation, bracketed** (review): rank-space retail **+0.0252
  (t 2.07)**, winsorised 1/99 **+0.0210 (t 2.05)**; foreign −0.0141 (t −1.24)
  and −0.0148 (t −1.49). The registered +0.0275 sits near the top of a
  +0.021 … +0.028 range; positive under every variant.
- **Holding what retail BOUGHT** (the opposite leg, exploratory): **−1.9%/yr**
  at 63 sessions (t −0.76), **+2.7%/yr** at 126 (t +1.04).

### 4.3 The pre-amendment spec (coverage ≥ 0.90), as promised

Retail IC **+0.0505 (t 4.07; permutation p 0.0009 within-month, 0.0012 with
names relabelled)** — ~~p 0.0007~~, a truncation of the first run's 0.0008 —
positive in both halves and every tercile; **+0.0532 (t 4.38)** with transfers
scored out and **+0.0411 (t 2.82)** with them refitted; rank-space +0.0498
(t 3.21); holding what retail sold
**−10.8%/yr**; holding what it bought +2.8%/yr (t 0.63). Same sign, larger, in
the narrower universe where the controlling holder sits inside the balance —
and still not under the permutation bar.

### 4.4 Mechanism, descriptive

- **Retail is contrarian within the month:** same-month rank correlation with
  the month's return **−0.249**, negative in **all 62** months (~~−0.253, 98%~~:
  the first version paired some flows with multi-month returns, §4.7).
- **Retail is the foreigners' counterparty:** −0.479 within month.
- Flow persists: AR(1) **+0.222** on 9,849 archive-adjacent pairs
  (~~+0.210~~ paired some months up to 50 apart).

That is the Kaniel–Saar–Titman picture — individuals buy what falls, sell what
rises, and are not punished for it at a quarter's horizon — not the
Barber–Odean one §12 assumed.

### 4.5 Survivorship, measured with the file itself

The delisted price store is deliberately not imported (`delisted_collect.py`,
pending your licensing call), so names without prices drop out. KSEI lists
them anyway: **197 of 996** in-sample codes have no price here, and their mean
within-month retail-flow percentile is **0.490 against 0.504** for priced names;
the 31 that left KSEI entirely sat at **0.474** in their last year. The missing
names did not lean toward retail buying, so survivorship is not what produced
the sign. It is stated, not corrected.

### 4.6 What H64 does and does not say

It says: at the account-type resolution the literature asked for, monthly
retail flow does not identify a losing cohort in Indonesia; a weak tilt points
the other way and does not survive cost on either side. **This is the fifth
instrument on §12** — aggregate flow (H9), broker identity (H10/H11), investor
domicile (H12), price structure (H13), investor type (H64) — and the fifth with
no tradeable answer. It does not say retail never loses: monthly holdings
cannot see intramonth round trips, which is where Barber et al. find most of
retail's losses.

**The holdout stays unspent.** A test of the reversed sign would be a new
hypothesis, and with ~24 months it has roughly a coin's chance of detecting the
in-sample effect, while the economics are already ~zero in-sample. Spending
the only clean data on that is the wrong trade.

---

### 4.7 Review, 2026-11-16 — five findings, all confirmed, no verdict changed

An independent reviewer reproduced every printed number and then found five
defects. Its own skeptic agents never ran (a usage limit), so each was
confirmed by re-implementing the fix here and recomputing; the registered
statistics reproduced exactly (IC +0.0275, t 2.58, p 0.00690).

1. **The registered null is anti-conservative for this design.** It reshuffles
   each month independently, so its draws are independent across months while
   the real ICs are not: 63-session labels span three monthly decisions and
   flows persist (AR(1) +0.22). A null that relabels names *consistently
   across months* keeps that dependence. Retail p **0.0069 → 0.0120**,
   foreign 0.1075 → 0.1413, and the persistent parity control's null widens
   from sd 0.0107 to **0.0172** (p 0.1555 → 0.3535) — the textbook signature.
   The reviewer's own construction gave 0.0165 / 0.1447 / 0.4064. **Quote the
   relabelled p.** A17 and A25 already said clustered data needs a clustered
   null; the registration did not apply it across time.
2. **OLS neutralisation of a heavy-tailed flow.** Per-month kurtosis of
   `f_retail` has median 25; a few rows set the coefficients. The transfer line
   therefore measured leverage, not the transfers' tilt (§4.2), and residual
   leakage onto the controls ran to |r| ≈ 0.06. Rank-space and winsorised
   variants are now printed beside the registered OLS.
3. **Mechanism windows.** `shift(1)` over filtered rows paired 806 of 10,655
   flows with returns spanning 2–50 months. Fixed with archive-adjacent
   predecessors: −0.249 (all 62 months negative), AR(1) +0.222.
4. **A rounding slip** in the pre-amendment p (written 0.0007).
5. **A latent unpivoted-QR rank drop** that would silently break the
   projection if a degenerate column ever preceded others. Never fired here;
   now a pivoted QR, with the failing case as a test.

None of these could pass a signal: both fail on sign, the gate also requires
|HAC t| > 3.81, and a too-narrow null can only make the parity control fire,
which voids rather than passes. What they change is how strong the wrong-sign
tilt looks — t ≈ 2 with p ≈ 0.012, not p 0.0069 — and two sentences that were
false.

---

## 5. One cheap check of the cost model

OJK publishes the market-wide average quoted spread. For October 2026 it reads
**1.517%** (daily 1.21–2.21%). The repo's model, one fraksi-harga tick as a
share of price averaged over the same traded names, reads **0.952%**
(0.855% above Rp 1bn/day). Methodology is not published, so this is a
direction rather than a calibration: **real spreads are wider than one tick on
average, which is what `price_ic.half_spread_frac` already says ("a floor, not
an estimate").** Every net figure in this repo is, if anything, optimistic.

---

## 6. Running trade

Recorded in `docs/LIVE_DATA.md` §4 (2026-08 audit) and
`docs/MANDIRI_REQUEST.md`; re-checked here:

- **A print with buyer and seller codes exists as a FEED in two places only:**
  IDX ITCH (Rp 17.9–44m a month, a signed licence, a company, a leased line)
  and licensed redistributors. Nothing has changed.
- **Growin' (Mandiri Sekuritas)** shows Smart Running Trade in-app. No API or
  export found in Mandiri's 2026 material. Reading your own screen is clean
  (`scripts/ocr_broker.py`); scripting your logged-in session is the thing
  their terms and IDX's market-data rules restrict.
- **New since August: idxpro.id**, an MCP server that sold live IDX prices,
  orderbook and running trade. Its homepage now reads *"idxpro telah tutup …
  Unofficial, tidak berafiliasi dengan PT Bursa Efek Indonesia"*. Unlicensed,
  and gone.
- **Do not give any automated system your securities login.** This container
  is a cloud machine; a trading account's credentials do not belong in it.

**And the brief says not to build it yet.** §3: no live feed until a validated
end-of-day signal turns out to be intraday-sensitive. None exists — H64 is one
more measured null. The only intraday quantity this repo has measured as
material is **execution**: A38 found fees-only versus a full tick worth ~1.1%
a year. That needs a limit order at the bid when the card rebalances, not a
feed.

---

## 7. What would change this, and what it would cost

| | would change | cost |
|---|---|---|
| the 2018 files' publication dates | four more months of usable signal | an archive.org route that answers |
| named holders above 1% (IDX monthly, since 2026-03) | a concentration / free-float risk display; too short for any statistic | you download in a browser → a parser written against one real file |
| KSEI corporate-action PDFs | exact dividend dates and amounts in the spine | a PDF parser; a few hours |
| a licensed tick feed | intraday execution study | Rp 17.9m+/month and a licence |
| the delisted store | survivorship-complete H64 | your CC BY-NC call (A5) |

## 8. What I now believe

- Retail flow, measured exactly from depository holdings, is **not** a
  persistently losing cohort at a quarter's horizon in Indonesia
  (confidence: moderate — 62 months, one regime-rich but single history).
- Neither following nor fading it is a long-only trade after 0.56% plus spread
  (confidence: high — the best leg is t ≈ 1 before any multiple-testing
  allowance).
- The ownership file is still worth having as **context**: who holds a name,
  how much of it actually trades, and when a block moved. `scripts/ownership.py`
  prints that, with H64's result read from its own output beneath every report.
