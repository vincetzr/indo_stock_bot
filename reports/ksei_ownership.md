# KSEI ownership, running trade, and the free BEI/OJK sources

**Date:** 2026-11-15
**Verdict:** the depository publishes the account-type split CLAUDE.md §12 said
was unavailable, and retail flow measured with it **does not lose**.
*Corrected 2026-11-16 after an independent review (§4.7): the retail tilt's p
is ≈ 0.012 under a null that respects dependence across months, not 0.0069,
and two attributions were wrong. Corrected again the same day after a second
review (§4.8): the in-sample cut is now pinned to the registered date, the
transfer flag searches both sides of the domicile line, and four written
figures were wrong. No verdict changed either time.* H64's
registered prediction (what retail buys underperforms) failed with the sign
reversed. Neither side of retail flow is a long-only trade after cost.
Running trade: no free FEED exists; a free way to WATCH it does — IPOT's app,
opened to any broker's customers in April 2026 (§6).

Reproduce:

```bash
python3 scripts/ksei_collect.py      # the archive, ~1 h cold (30 s courtesy delay), none warm
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
| use all free BEI and OJK information | a source sweep | §2. The first sweep (eight rows) was not "all": a second, wider one added KSEI's legacy server, OJK's broker and sharia lists, KPEI's haircut list, Bank Indonesia's series and others. One source is ingested; the rest are listed with what each would cost and who has to approve it |

---

## 2. The source sweep — tested, not researched

Every row was an actual request: on 2026-11-15, or on 2026-11-16 where marked
*(second sweep)*. Rows marked not built are listed, not used.

| source | status | shape | history | used |
|---|---|---|---|---|
| **KSEI `BalanceposEfekYYYYMMDD.zip`** | **200** | every security, depository-held units by 9 investor types × local/foreign, plus units on issue and month-end price | **2018-01 → 2026-10 on `www.ksei.co.id/storage/Download/`, 106 files, none missing.** Page lists 36; older ones are served but unlinked. *That is one path, not the archive — see the next row* | **ingested, H64** |
| KSEI legacy server `web.ksei.co.id/Download/` *(second sweep)* | 200 | the same files | HEAD 200 for 2009-12, 2012-12, 2017-12 (2007-12 404, 2008-12 500); **its 2018-01 copy has identical text, stamped 2018-02-01 06:00 against 2020-04-13** | not built — the host is not in `data.ksei_allowed_hosts` (your call); no pre-2018 file opened |
| KSEI `StatisEfekYYYYMMDD.txt.zip` (master file) | 200 | ISIN, issuer, listing date, units, warrant/rights terms (exercise price, expiry), sector; 28 columns; STATUS is "ACTIVE" on every row, so a dead code simply disappears | from 2019-01 on the new path; the legacy path was not tried before 2020 | not built |
| KSEI corporate-action letters, legacy index *(second sweep)* | 200 | 11 categories (cash and share dividends, bonus, rights, splits…), each letter dated, PDF fields parse cleanly (an ASII 2010 dividend: amount, cum/ex, recording, payment) | at least 2010-05 → now | not built — would replace the spine's inferred dividend steps with stated dates; no rights letter was opened, so the TERP fields are unverified |
| KSEI CMS `webcms.ksei.co.id` | ~~proxy 502~~ **does not resolve (NXDOMAIN)** — the 502 was a DNS failure, and "the listing's back end" was a guess | — | — | not needed |
| IDX `idx.co.id` (incl. the >1% holders, HSC list, free-float pages) | **403** Cloudflare | — | — | not touched; `docs/FULL_REKAP.md` §3 explains why this repo does not go through the check |
| OJK weekly *Bid-Ask Spread & Turnover* xlsx | 200 | MARKET-WIDE average quoted spread and turnover ratio, daily within the month | annual from 2015 | used once, as a cost-model check (§5) |
| OJK weekly *Emisi Efek & Aksi Korporasi* xlsx/pdf | 200 | issuance and corporate-action overview | weekly | not built, and not opened by either sweep |
| archive.org (to date KSEI's 2018 files) | connection reset, twice | — | — | superseded: KSEI's own legacy server dated 2018-01 (row 2) |
| **OJK *Data Perusahaan Efek*** *(second sweep)* | 200 | monthly xlsx of licensed securities firms: 2-letter code, name, **every former name**, exchange member or not, licences, and revocation decrees with dates | ~2021 onward (older pages need a postback) | not built — it is the §5 broker-code master with dates. 61 of `config/brokers.yaml`'s 66 codes appear; it names the ~30 the repo only knows by code, and lists CG, DB, ML, MS as no longer exchange members |
| OJK *Daftar Efek Syariah* *(second sweep)* | 200 | the semiannual sharia list (622 stocks in 2026-I, by IDX-IC sector), decree-dated | ~2007 onward, xlsx at least since 2021 | not built — an add/delete event study is possible but its size was not measured; sector labels agree 585/585 with the frozen map |
| OJK data portal `data.ojk.go.id` *(second sweep)* | 200 | 55 aggregate cubes; per-issuer P&L for only three quarters of 2024; no per-ticker trading or ownership | from 2023 | not useful: **every cube is reprocessed together with no vintages**, so history can change silently |
| KPEI/IDClear equity haircut list *(second sweep)* | 200 (JSON) | the clearing house's collateral haircut per stock, daily; 585 of 1,013 at 100% | **rows before 2026-09-26 were backfilled on that date** (the KSEI trap again) | not built — forward collection only; terms allow non-commercial use with attribution; the host needs your approval |
| KPEI + BEI *Efek Tidak Dijamin* *(second sweep)* | 200 | monthly list of stocks whose settlement KPEI does not guarantee, chosen on manipulation-type criteria | current month only | check by hand before acting on a card name; the November 2026 list is empty |
| PEFINDO rating actions *(second sweep)* | 200 | 7,310 dated actions since 2004, with IDX tickers | 2004 → now | **manual only: PEFINDO's terms forbid scraping.** One test call was made before the terms were read; nothing was collected |
| Bank Indonesia: BI-Rate, JISDOR, INDONIA, CPI *(second sweep)* | 200 | decision dates since 2016; the USD/IDR fixing since 2013; overnight interbank since 2018; CPI since 2003 | as stated | not built — §10 regime variables, last in §4's order; BIS's keyless API gives the policy rate back to 2005 |
| Kontan *Dana Asing* *(second sweep)* | 200 | market-wide net foreign flow, daily since 2000, one request | 6,508 sessions | **not built: robots.txt disallows AI crawlers by name**, the terms are unread, and its largest days look like transcription errors (378,649 where 378.649 fits KSEI's monthly holdings) |
| PHEI government yield curve *(second sweep)* | 200 | today and yesterday only | none | display-only; history is a paid product |
| FRED / BPS / DJPPR / e-IPO / TICMIDATA / reksadana.ojk.go.id | unreachable, 403, or TLS failure | — | — | recorded as attempted, not as absent |

**KSEI's terms.** *Penafian Umum* and *Ketentuan dan Kebijakan* (read
2026-11-15) disclaim accuracy and govern K-CASH/AKSes; neither restricts the
public statistics, and there is no counterpart to IDX's explicit anti-crawling
clause. The host sits in `data.ksei_allowed_hosts` so you can withdraw it. **This
repository is public**, so the holdings are gitignored and only a hash manifest
(`data/reference/ksei_manifest.csv`) is tracked. The manifest now also records
each file's member CRC and text hash, so a copy fetched from the legacy server
can be checked by content (its zip hash differs for identical text).

**What the second sweep's agents did that you should know.** 121 requests
across five categories, each capped at 25. It sent one request to PEFINDO's
internal table endpoint before reading terms that forbid crawling. It sent one
plain query to OJK's portal after the portal's firewall had rejected an
earlier, more complex one. It also requested `web.ksei.co.id`, which is not on
the allowlist. Nothing from any of these was collected or ingested.

**`www.ksei.co.id` reset every TLS handshake for a while** about eight hours
after the 106-file collection. One HEAD request on 2026-11-16 got a 200, so
the block was not lasting. The courtesy delay is now 30 s, not 2 s.

---

## 3. What the file actually measures — three findings before any return

### 3.1 The point-in-time trap: backfills

Each zip carries its member's creation stamp. From 2019-03 on it is 1–12
days after the state date, **median 2** (~~median 3~~: that was the median of
all 106 files, backfills included), stamped 06:00 on 87 of 91 files — the
other four at 06:41, 10:11, 11:16 and 11:37. But the **14 files for 2018-01 …
2019-02 were all created 2020-04-13**, and the 2020-12 file on **2021-10-19**.
~~Whether that data was public at the time cannot be established from here.~~
*KSEI's own legacy server answered it for one month* (second review): its copy
of 2018-01 has byte-identical text (CRC 0x0010c988, 263,916 bytes) stamped
**2018-02-01 06:00**, so the 2020 stamp is a re-zip, at least there. The other
13 months were not checked. That host is not on the allowlist, so nothing is
re-dated. The test therefore still treats every file as available only from
its own stamp. That makes the first usable signal April 2019 and drops two
months around the 2020 regeneration. A stamp is a lower bound on publication,
which is the convention the 2019-03+ files already rely on.

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
its corporate holding rose by **10,751,301,170** — ~~a founder re-registering a
stake~~ a type-to-type move consistent with one holder re-registering a block
(KSEI carries no holder identity, so who moved it is not observable). Read as
flow, it is retail selling **Rp 98 trillion** in one month, and three names
supplied 88% of that month's absolute retail flow market-wide.

The first flag looked for the mirror only among LOCAL types: 2,511 name-months,
2.84%. **It could not see a move across the domicile line**, and those exist
(second review). 122 individual-type moves are mirrored by a single foreign
type to within a millionth, against 0–1 when the partner is taken from the
month before or after. EMTK's foreign-individual line moved +4,990,264,500,
−4,989,570,900 and +4,994,405,900 in 2023-01, -02 and -07: the same block out,
back and out again. PNGO's whole individual holding (98.3% of the balance)
went to foreign corporate in 2026-05. MAPI's local-corporate block (51% of the
balance) went to foreign corporate the same month. `ksei.transfer_flags` now
searches all 17 other types (3,773 name-months), and
`ksei.foreign_transfer_flags` flags a foreign type mirrored by a local one
(2,141). Both are robustness lines, not filters (§4.2).

**The descriptive table was wrong in sign for its biggest month.** The first
`ownership.py --market` printed May 2026 foreign flow as **+Rp 6.94T**.
MAPI's block contributed +Rp 12.59T of that and PNGO's +Rp 2.68T. Corrected,
the month reads **−Rp 8.56T**: foreigners sold.

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

- **Type-to-type transfers.** With the first, local-only flag (306 rows),
  scoring the rows out of the IC while keeping them in the per-month fit gives
  retail **+0.0280 (t 2.80)**. Dropping them from the fit as well gives
  **+0.0203 (t 1.87)**. The transfers carry none of the tilt themselves; their
  extreme values move the OLS coefficients for every other name.
  ~~A quarter of the retail tilt was re-registration.~~ *(Retracted by the
  review, §4.7: that read a leverage effect as the rows' own contribution.)*
  **Each signal against its own flag** (second review, §4.8), scored out /
  dropped:

  | flag | rows | retail | foreign |
  |---|---|---|---|
  | first version, local partners only, 10% | 306 | +0.0280 / +0.0203 | −0.0175 / −0.0173 *(cannot touch foreign_total)* |
  | mirrored by any type on the other side, 10% | 558 / 413 | +0.0314 (t 3.05) / +0.0215 | −0.0223 (t −2.20) / −0.0119 (t −1.07) |
  | the same, within 1% | 119 / 85 | +0.0285 / +0.0250 | −0.0156 / −0.0126 |

  A band, not a number. Retail runs +0.020 … +0.031 and foreign −0.012 …
  −0.022 across definitions, every value wrong-signed and none near |t| 3.81.
- **Neutralisation, bracketed** (review): rank-space retail **+0.0252
  (t 2.07)**, winsorised 1/99 **+0.0210 (t 2.05)**; foreign −0.0141 (t −1.24)
  and −0.0148 (t −1.49). The registered +0.0275 sits near the top of a
  +0.021 … +0.028 range; positive under every variant.
- **Holding what retail BOUGHT** (the opposite leg, exploratory): **−1.9%/yr**
  at 63 sessions (t −0.76), **+2.7%/yr** at 126 (t +1.04).

### 4.3 The pre-amendment spec (coverage ≥ 0.90), as promised

Retail IC **+0.0504 (t 4.07; permutation p 0.0009 within-month, 0.0012 with
names relabelled)**, positive in both halves and every tercile. With the
local-only transfer flag, scoring the rows out gives **+0.0531 (t 4.38)** and
dropping them from the fit gives **+0.0410 (t 2.81)**. Rank-space +0.0497
(t 3.20). Holding what retail sold: **−10.8%/yr**; holding what it bought:
+2.8%/yr (t 0.63). Same sign, larger, in the narrower universe where the
controlling holder sits inside the balance, and still not under the
permutation bar.

*These figures moved at the fourth decimal between runs, and the first
explanation was wrong.* The first run printed p **0.00070**, a later one
0.0008, then 0.0009, and the IC moved +0.0506 → +0.0505. ~~A truncation of the
first run's 0.0008~~ / ~~a rounding slip~~: the second review found the real
cause. **45 rows are the only member of their sector that month.** Their own
sector dummy fits them exactly, so their residuals are zero in exact arithmetic
and about 1e-17 in floating point. That noise decided their ranks, and it
changes with the QR routine and the library build. Such residuals are now set
to exactly zero and ranked as ties, so the figures above are deterministic: the
same to 1e-17 under any row order. The confirmatory amendment-A spec has only
two such rows, never two in one month, and its numbers never moved.

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
the other way and does not survive cost on either side. **This is the fourth
instrument on §12** — aggregate flow (H9), broker identity (H10/H11), investor
domicile (H12), investor type (H64) — and the fourth with no tradeable answer.
~~Fifth … price structure (H13)~~: H13 tested §8's price features, which
contain no flow and cannot name a cohort, and A8/A9 file it there. It does not
say retail never loses: monthly holdings cannot see intramonth round trips,
which is where Barber et al. find most of retail's losses.

**The holdout stays unspent.** A test of the reversed sign would be a new
hypothesis. If the in-sample effect were exactly true, ~24 months would detect
it about half the time at a nominal one-sided 5% — ~~roughly a coin's
chance~~, a claim that left its level out — but **only ~1–3% of the time at
the Bonferroni-plus-|t| bar** every registered test here faces. The economics
are already ~zero in-sample. Only 21 of the 24 holdout months carry a
63-session label today. Spending the only clean data on that is the wrong
trade.

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
4. ~~**A rounding slip** in the pre-amendment p (written 0.0007).~~
   *Wrong diagnosis* (§4.8): the first run did print 0.00070; the figure
   moved because of floating-point ties.
5. **A latent unpivoted-QR rank drop** that would silently break the
   projection if a degenerate column ever preceded others. Never fired here;
   now a pivoted QR, with the failing case as a test.

None of these could pass a signal: both fail on sign, the gate also requires
|HAC t| > 3.81, and a too-narrow null can only make the parity control fire,
which voids rather than passes. What they change is how strong the wrong-sign
tilt looks — t ≈ 2 with p ≈ 0.012, not p 0.0069 — and two sentences that were
false.

### 4.8 Second review, 2026-11-16 — eight findings survived two skeptics each

The second review ran three finders (point-in-time, data layer, written
claims), with two adversarial verifiers per finding, plus five source sweeps
and a critic; 20 agents in all. Eight findings survived. Each was re-implemented
here before any fix. **The confirmatory numbers are bit-identical after every
fix**: IC +0.0275, t 2.58, p 0.0069 / 0.0120, every null sd, every leg.

1. **The in-sample cut rolled with the price panel.** `ksei_flow.py` read the
   panel's own `holdout` column, which is the panel's last date minus 24
   months. On the day it ran that was exactly the registered 2024-11-13, so
   nothing looked wrong. After a refresh past 2026-12-03, a re-run would have
   pulled registered-holdout labels into the in-sample; past 2027-03-11 it
   would have pulled in whole holdout months (642 rows by mid-2027), while
   still printing "holdout UNTOUCHED". The cut is now the constant
   `IN_SAMPLE_CUT`; the panel's rolling cut is printed beside it and never
   used; a test pins it. `price_panel_build.py`'s claim that the split "cannot
   drift between runs" was true within one build and false across rebuilds.
2. **The transfer flag saw one side of the domicile line** (§3.3, §4.2). For
   H64b, which reads foreign_total, the first robustness line removed rows
   that a local-to-local move cannot affect, so it tested nothing.
3. **A 503 counted as "no such file"**, and every consecutiveness check counted
   archive positions, not calendar months. So one failed fetch would have made
   a two-month flow enter as a one-month signal, with nothing failing. It never
   fired, since all 106 months are present. Absence is now a redirect, 404 or
   410 only; everything else is retried and then raised. Consecutiveness is by
   calendar month, and H64 refuses to run on a panel with a missing month.
4. **The pre-amendment figures' drift was floating-point ties**, not a
   rounding slip (§4.3).
5. **§7 said dating the 2018 files would add "four more months"; it adds 13.**
   Re-dating the 14 backfilled files on the regular schedule takes the
   in-sample from 62 to 75 months (IC +0.0229, t 2.41; same sign, still far
   from the bar). It would be 14, except that 2018-12's decision lands on
   2019-01-01. That is a zero-volume holiday filler in the panel's date union,
   181 of which exist. Decisions are now drawn only from dates with a tradeable
   bar; no current decision moved.
6. **"About a coin's chance" left its level out** (§4.6).
7. **The stamp-lag median is 2 days, not 3** (§3.1).
8. **"Fifth instrument on §12" counted H13**, a §8 study (§4.6). The BYAN
   sentence named an actor the data cannot identify (§3.3).

What survived the verifiers but is NOT a defect: the first transfer flag was
stated accurately everywhere it was printed. Its gap was what it could not see,
not a false number.

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
- **IPOT's app is free to any broker's customers** (reported April 2026 by
  Kontan, Emiten News and Infobank, quoting IPOT). It has a live orderbook,
  HAKA/HAKI and LADI accumulation indicators, needs no fund transfer, and your
  money stays at Mandiri. It is the cheapest legitimate way to WATCH running
  trade and the queue. It is not a feed: the stream runs over a logged-in
  websocket, and scripting it is the same restricted act. The same coverage
  says its broker summary is delayed up to 16 hours. IPOT X, the AI version
  launched 2026-09-24, may have changed the terms; check in the app store.
- **IndoPremier's anonymous stock page now redirects to the homepage** (301).
  The broker-summary module `ipot.py` calls directly was not requested, because
  `data.broker_allowed_hosts` is empty by your design. So whether the repo's
  only free per-ticker broker-summary route survived the redesign is
  **unknown**. One request with the fixture's parameters, after you approve
  the host, would settle it.
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
| the 2018 files' publication dates | ~~four~~ **13 more in-sample months** (62 → 75) for any new investor-type test | allow `web.ksei.co.id`, fetch 14 legacy copies at ≥ 30 s, adopt a stamp only on an exact CRC match |
| BalanceposEfek before 2018 (legacy server) | possibly 2009 onward: twice the span, and a monthly register that includes names that later died | the same host decision; ~100 files at ≥ 30 s; each opened and checked against the identities before use. Existence rests on three HEAD responses |
| OJK's monthly broker list | a dated broker-code master for §5 | ojk.go.id terms read and the host approved; a parser for one xlsx shape |
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
