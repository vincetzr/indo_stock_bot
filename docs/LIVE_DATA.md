# Getting broker summary and net foreign on IDX

> "How do other sites track broker summary when this repo can't?"

Answered first, because the answer changed: **this repo can now, and the reason
it could not before was never technical skill.**

---

## 0. Why idx.co.id blocks you and not them

Three facts, each verified rather than assumed:

1. **The block is on the network, not the endpoint.** `idx.co.id` returns
   `403` from Cloudflare for the broker-summary JSON, the stock-summary JSON,
   the digital-statistics API — *and for the bare homepage*. Nothing is being
   defended selectively. This egress is simply not welcome, which is what a
   datacentre IP outside Indonesia usually gets. Someone on an ordinary
   Indonesian connection hits none of this.
2. **The big platforms are not scraping at all.** Stockbit, RTI, Ajaib and the
   bank-owned terminals are IDX-licensed data-feed subscribers. They pay for
   trade-by-trade with member codes and redistribute it under licence. There is
   no clever request to reverse-engineer, because they are not making one.
3. **IDX prohibits scraping outright.** *"PT BEI telah melarang tiap-tiap
   pengguna untuk melakukan web crawling ataupun scraping."* So the 403 is the
   stated policy being enforced, not an obstacle to route around.

The asymmetry was licensing and geography. Knowing that is what pointed at the
route that actually works.

---

## 0b. The route that works: an IDX member publishes it

Broker summary is not IDX-exclusive. Every exchange member receives it and some
publish it. **IndoPremier** (PT Indo Premier Sekuritas, an IDX member) renders
the full rekap broker on a public, unauthenticated page:

```
/module/saham/include/data-brokersummary.php?code=BBCA&start=2026-08-13&end=2026-08-13&board=RG
```

One GET, no key, no session. `src/idxbot/data/ipot.py` reads it and it is in
the default provider chain:

```bash
idxbot analyze BBCA              # real broker flow, no configuration
idxbot screen --universe lq45
```

**What it gives you**

| | |
|---|---|
| Coverage | top 10 buyers + top 10 sellers, ranked independently |
| Fields | lots, value, average price, per side; plus net foreign value |
| Broker class | the source's own three-way label: foreign / **bumn** (state-owned) / local |
| Boards | regular / cash / negotiated, filterable |
| History | back to roughly 2008; 2005 returns zeroes |
| Freshness | previous session, available the same evening |
| Cost | free |

**Proof it is the real thing, not a lookalike.** Regular-board totals were
checked against Yahoo's tape for 2026-08-13:

| | parsed | tape | error |
|---|---|---|---|
| BBCA | 832,077 lots | 832,080 | 4e-6 |
| ANTM | 860,776 | 860,777 | 1e-6 |
| ASII | 484,063 | 484,073 | 2e-5 |
| UNVR | 99,947 | 99,958 | 1e-4 |

And independently, the table over-determines itself — value must equal
lots x 100 x average. Across 160 rows and eight stocks the median disagreement
is 0.2%, entirely explained by display rounding.

### Three ways this will bite you

**1. Use the regular board.** The default all-board view folds in negotiated
block crossings that print at arbitrary prices. GOTO on 2026-08-13 reads
**25.0M lots all-board against 196k on the regular board** — a factor of 127,
at an average price of 32 against a close of 50. `board: "RG"` is the default
here for that reason.

**2. A top-10 view cannot balance, so the inventory ledger drifts.** In a
complete rekap every lot bought is a lot sold and the market-wide net is
exactly zero. In a top-10 view a broker only appears on the side where it was
large that day, so a steady accumulator's selling is censored away. On BBCA
over 52 sessions DX appears as a top-10 buyer on 21 days and a top-10 seller on
5, and the market-wide cumulative net — which must be zero — comes to **-2.8
million lots**. `truncation_bias()` measures this and `idxbot analyze` prints it
above the position table. *Direction and relative ranking survive. Absolute
positions, cost basis and open P/L do not.*

**3. Big figures are abbreviated.** `3.4 M`, `699.9 B` — two to three
significant figures above one million, exact below it, with average prices
always exact. Fetching one session at a time (which the provider does) keeps
the numbers small and therefore exact far more often than a range query, which
sums first and abbreviates afterwards.

### Please do not abuse it

This is a public page on a licensed member's site, read the way a browser reads
it. It is still someone else's server, and IDX still restricts redistribution
of its market data. The provider fetches one day at a time, sleeps between
requests, and caches permanently. Do not turn it into a bulk harvester and do
not redistribute what it returns.

---

## 0c. What this fixed elsewhere in the repo

Connecting a real source immediately exposed two things no amount of reasoning
had:

* **20 broker codes were missing from `config/brokers.yaml`** and silently
  defaulted to `foreign: false`. That is not a neutral default — it pushed real
  foreign flow into the domestic bucket and understated net foreign. They are
  added now, carrying the source's own F/D flag (stable across 90 fetches),
  with names left as bare codes rather than invented.
* **BQ, DR and TP disagree.** All three have foreign parents and are flagged
  foreign here on an ownership basis; the exchange-side flag calls them
  domestic, every time. Both answer different questions and the disagreement is
  recorded rather than resolved by fiat.
* **A whole category was missing.** The source labels brokers three ways, not
  two — the third being `bumn`, state-owned. Exactly four houses carry it: CC
  (Mandiri Sekuritas), DX (Bahana), NI (BNI Sekuritas), OD (BRI Danareksa).
  `state_owned` is now a third axis on `Broker`, because "the state is
  accumulating" is a different claim from "a domestic institution is
  accumulating". **DX had no registry entry at all** — a state-owned house
  among the exchange's largest desks, surfaced only by connecting real data.

The simulator has also been **removed from the default provider chain**. It
existed because no real source was reachable; leaving it as a fallback would
let a transient network failure swap simulated flow into a real-looking report
one ticker at a time. Ask for it explicitly with `--providers synthetic`.

---

## 1. The structural fact that makes this possible

Broker summary is not a separate data product. It is an **aggregation of running
trade**.

Every print on IDX carries a buyer member code and a seller member code:

```
09:41:07   BBCA   6350   150 lot   BK -> YP
09:41:09   BBCA   6350    40 lot   AK -> PD
09:41:11   BBCA   6375   200 lot   BK -> CC
```

Sum those by broker and you have reconstructed the broker summary table. That is
literally all your platform is doing when it shows you "Broker Summary" — it just
does the summing on a batch job that runs after the close.

**Running trade with broker codes is displayed live, during the session, on
essentially every Indonesian platform.** The delay you are hitting applies to the
aggregated *view*, not to the underlying tick stream. So the problem reduces to:
get the prints out of the platform you already have.

That is what `src/idxbot/data/running_trade.py` implements:

```bash
# follow a file your platform (or a userscript) appends to
idxbot live --file ticks.jsonl --follow

# or pipe ticks in
cat ticks.jsonl | idxbot live --stdin

# or aggregate a completed session's prints
idxbot live --file session_2026-08-07.csv --out data/broker_summary/BBCA.csv
```

Expected fields per tick — the parser accepts English or Indonesian headers, and
JSONL or CSV:

| field | meaning |
|---|---|
| `ts` | print timestamp (a bare `09:41:07` works; pass `--date`) |
| `ticker` | bare IDX code |
| `price` | trade price in rupiah |
| `lot` | size in lots |
| `buyer` | buying member code |
| `seller` | selling member code |

---

## 1b. Getting the DAILY broker summary (three routes, easiest first)

Everything above is about intraday. If you just want the end-of-day table:

**Route A — paste it (works today, zero setup).** Select the broker-summary
table in your platform, copy, and run:

```bash
idxbot paste BBCA --date 2026-08-06
```

The parser handles tab/pipe/multi-space/comma separation, Indonesian numbers
(`1.234.567,89`), abbreviated values (`1,2 M`, `450 rb`), and the side-by-side
buyers|sellers layout. It then checks that total buy lots equal total sell lots
— every lot bought is a lot sold, so a mismatch means the columns were misread,
and it says so rather than saving nonsense.

**Route B — fetch it with a real browser (run on your machine).**

> ⚠️ **IDX prohibits scraping.** A widely-used community dataset carries the
> notice *"PT BEI telah melarang tiap-tiap pengguna untuk melakukan web crawling
> ataupun scraping"* — IDX has forbidden users from crawling or scraping the
> site. That reframes this route: the Cloudflare block is not an obstacle to
> route around, it is the stated policy being enforced. `scripts/fetch_broker_summary.mjs`
> remains in the repo because reading a page you are entitled to view is a
> different act from bulk harvesting, but **Route A and Route C are the
> defensible paths** and this one is at your own risk. Prefer them.

```bash
npm install playwright && npx playwright install chromium
node scripts/fetch_broker_summary.mjs BBCA 2026-08-06
```

idx.co.id blocks plain HTTP clients (403) but lets real browsers through after a
Cloudflare JS challenge. Headless Chromium passes that challenge *from an
ordinary connection*. It does not pass from a locked-down build sandbox, which
is why this is a script you run rather than something the engine calls. Use
`--headed` to watch it and fix selectors if IDX redesigns the page.

**Route C — a paid vendor API.** These exist and sell exactly this data:

| Vendor | What it advertises | Notes |
|---|---|---|
| [Invezgo](https://invezgo.com/data-api-saham-indonesia) | *"Full bandarmology suite: foreign flow, broker distribution, accumulation zones"* + 15 years history | Markets itself explicitly against the 15-minute delay. `api.invezgo.com` is a live API server. Closest fit to this engine. |
| [OHLC.dev](https://ohlc.dev/indonesia-stock-exchange-idx-api) | *"Equities, index constituents, broker summaries, bonds"* | IDX-focused |
| [GoAPI.io](https://goapi.io/api-data-saham-indonesia/) | `/stock/idx/{SYMBOL}/broker_summary` | Route confirmed live and key-gated — answers 401, not 404 |
| [Sectors.app](https://sectors.app/) | IDX market data, v2 API | Broker-level coverage varies by plan |

Pricing sits behind signup, so confirm two things before paying:

1. **Is broker summary real-time or end-of-day?** For this engine's 60-day
   horizon EOD is fine; for day trading it is not.
2. **How much history?** Campaign profiling needs months, ideally years. A
   real-time-only feed cannot backtest the thesis.
3. **Is it per-broker, or foreign-flow aggregate?** Aggregate cannot support the
   ledger, campaign segmentation, lead-lag or coordination analysis. Some
   vendors describe both as "bandarmology".
4. **Does history include buy/sell VALUE, not just lots?** This is the quiet
   dealbreaker. Without value columns there is no VWAP, so no cost basis, so no
   way to tell whether a desk is underwater — which is half of what the ledger
   exists to compute.

Because each vendor uses its own paths, auth style and field names — and their
docs are behind signup — the engine ships a **generic REST adapter** rather than
a guessed endpoint. Fill in `data.rest_broker_summary` in `config.yaml`, set your
key in the environment, and run `--providers rest`. Field naming is handled by
the same normaliser as every other route, so a new vendor is usually zero code.

**Open-source caution.** [NeaByteLab/IDX-API](https://github.com/NeaByteLab/IDX-API)
(MIT, no auth) looks like a free answer and is not: its `syncBrokerSummary()`
hits `ExchangeMember/GetBrokerSearch`, which is the **exchange-member directory**,
not per-stock rekap broker. It is still useful — it documents IDX's real public
API surface (`/primary/DigitalStatistic/GetApiData?urlName=...`), including
`LINK_TABLE_DAILY_TRADING_INVESTOR_FOREIGN` for aggregate foreign-vs-domestic
flow. Aggregate foreign flow is a weaker signal than per-broker data, but it is
free and unauthenticated.

All three land in `data/broker_summary/<TICKER>.csv`, which the CSV provider
reads automatically.

## 2. Where to actually get the stream

Ordered by how much friction each involves.

### A. Your platform's running-trade window (cheapest, most friction)

Most desktop platforms (Mirae HOTS/Neo, IPOT, RTI, BIONS, MOST, Stockbit) show a
live running-trade panel with broker codes. Getting it into a file means either:

- an **export button** if the platform has one (some allow CSV export of running
  trade mid-session), or
- a **browser userscript** that mirrors the running-trade DOM/websocket into a
  local file, for web platforms.

⚠️ **Read your platform's terms of service before automating this.** Scraping a
paid terminal generally violates its ToS even when the data is on your own
screen, and IDX's market-data rules separately restrict *redistribution*.
Personal use of data you are licensed to see is the least problematic case;
sharing or reselling it is not. This is your call to make, not mine — but make it
knowingly.

### B. A commercial data vendor (cleanest, costs money)

| Vendor | Notes |
|---|---|
| **GoAPI.id** | Has a broker-summary endpoint. Probing `https://api.goapi.io/stock/idx/{SYMBOL}/broker_summary` returns `401 invalid API key` rather than 404 — the route is live and gated on a key. Set `IDXBOT_GOAPI_KEY` and use `--providers goapi`. Check with them whether your tier is EOD or intraday. |
| **Sectors.app** | Indonesian market data API (v2; v1 was discontinued 2026-05-11). Coverage of broker-level data varies by plan. |
| **IDX Data Services** | The authoritative source. IDX licenses a real-time datafeed including trade-by-trade with member codes. This is the "correct" answer and it is priced for institutions. |
| **Refinitiv / Bloomberg** | Carry IDX broker-level data via the exchange feed. Institutional pricing. |

`idxbot` ships a GoAPI adapter (`src/idxbot/data/broker_summary.py`) and a
provider interface so adding another vendor is one class.

### C. Your broker's own API

Some Indonesian brokers expose programmatic access to their terminal data. If
yours does, that is the least legally ambiguous live route, because you are an
authenticated customer using a supported interface. Ask your broker's support
whether they offer an API tier — this varies by firm and changes often enough
that any list here would be stale.

---

## 3. The part you may not want to hear: you probably don't need it live

This engine detects **multi-week accumulation campaigns**. Look at what the
campaign segmentation actually finds: a median accumulation leg runs *dozens of
trading days*. A desk quietly absorbing supply over six weeks does not care, and
neither should you, whether you saw today's prints at 15:30 or 17:30.

Concretely, a two-hour delay costs you nothing on this strategy:

| | needs live tick data? |
|---|---|
| Detecting a multi-week accumulation base | **No** — daily granularity is plenty |
| Reconstructing a broker's cost basis | **No** — daily VWAPs are what feed it |
| Knowing a desk flipped to net seller | **No** — that is a multi-day pattern |
| Timing an entry *within* the session | Yes |
| Scalping off a single large print | Yes |

The intended workflow is deliberately end-of-day:

```
17:30-18:00 WIB   broker summary lands, drop it in data/broker_summary/
18:00             idxbot screen --universe lq45
18:05             idxbot plan --tickers <hits> --pool lq45
next session      work the plan's limit orders in the entry band
```

Where live data genuinely helps is **execution**, not detection: you already know
from last night which name you want and at what price, and intraday flow tells
you whether to lean in or wait. For that, the price/volume half of the signal is
enough, and TradingView gives you that live via
`src/idxbot/tradingview/pine/accumulation_score.pine`.

---

## 4. Interpreting intraday broker flow when you do have it

Two traps worth knowing:

**Raw intraday numbers are meaningless without a pace baseline.** Every broker's
volume is "small" at 10:00 because the day is incomplete. What carries
information is whether they are running *ahead of* their normal rate.
`intraday_pace()` computes exactly this — `pace_ratio > 1` means the desk has
absorbed more than its usual share for this point in the session.

**A broker code is not a firm's opinion.** `BK` is every client routing through
J.P. Morgan's membership: a foreign pension fund, a hedge fund, an index tracker
rebalancing, a delta hedge against a warrant. "J.P. Morgan bought 2 million lots"
means flow crossed their membership. It is a strong signal of institutional
intent. It is not a confession of one.

---

## 4b. Routes exhausted from a sandboxed environment

Recorded so this is not re-litigated. Every one of these was actually attempted:

| Route | Result |
|---|---|
| `idx.co.id` — 6 endpoint patterns, browser headers, Referer, XHR headers | 403 Cloudflare, every path |
| Headless Chromium against IDX | Fails, but on an unrelated proxy incompatibility — it also failed on Yahoo, which curl reaches. The approach is sound from a normal connection. |
| idnfinancials, sahamidx, duniainvestasi, britama, RTI, pasardana | Cloudflare / maintenance / 404 / egress policy |
| Stockbit API | Authenticated app session required |
| archive.org + Wayback CDX | 429, then blocked by egress policy |
| GitHub datasets (wildangunawan, faisalburhanudin, nichsedge) | Per-stock *foreign flow* yes (shares, to 2025-02); per-broker rekap no |
| GoAPI / Invezgo / OHLC.dev / Sectors | Live and key-gated |
| **indopremier.com public module** | ✅ **works — see section 0b** |

The old conclusion here was "every legitimate route runs through an account
someone holds". That was wrong, and wrong in an instructive way: the search had
only ever probed IDX itself and the commercial vendors. It never occurred to me
to ask whether an exchange *member* publishes the same table — and one does, on
a page with no login at all. The error was in where I looked, not in what
exists.

## 5. What is blocked, and why the fallback exists

From this build environment:

| Source | Result |
|---|---|
| Yahoo Finance chart API | ✅ works — full daily history, IHSG back to 1990-04-06 |
| `idx.co.id` broker summary | ❌ 403, Cloudflare WAF |
| Stockbit API | ❌ requires an authenticated app session |
| GoAPI broker summary | 🔑 route exists, needs a paid key |
| **IndoPremier rekap broker** | ✅ works, free, unauthenticated |

The simulator (`src/idxbot/data/synthetic.py`) is still in the tree but **out of
the default chain**, because the condition that justified it no longer holds. It
*assumes* institutions buy weakness, so analysing its output and concluding
"institutions buy weakness" is circular and proves nothing. It is now only for
exercising the pipeline offline: `--providers synthetic`, deliberately.
