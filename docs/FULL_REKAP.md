# Getting the FULL broker summary — the answer, and the line I won't cross

You were right that there is a way. Here it is, along with why one version of it
is fine and another is not.

---

## 1. IDX publishes the full broker summary themselves, free

**`https://www.idx.co.id/en/market-data/trading-summary/broker-summary`**

Official, authoritative, complete — every member, not a top ten — and it is the
source every other site is ultimately repeating. It costs nothing.

## 2. It is not my network that is blocked. It is automation

This container gets **403 on every idx.co.id path**, including the homepage.
Earlier in this project that was read as a geographic or network block. It is
not. The open-source scraper that does reach it (`nichsedge/idx-bei`) states
plainly what it uses:

> `curl_cffi` — *"browser impersonation, Cloudflare bypass"*

So idx.co.id sits behind **Cloudflare**, and the 403 is a bot check on the TLS
fingerprint. Defeating it means pretending to be a browser that isn't there.

**You showed me earlier that idx.co.id loads fine for you.** That is the whole
point: the site is not blocking *you*, it is blocking *scripts*.

## 3. Why I am not going to bypass it

Two reasons, and the first is enough:

- **IDX's terms prohibit scraping their market data.** Deliberately defeating a
  bot check on a source that forbids automated collection is not a grey area,
  and it is not a decision I should make on your behalf with your name and your
  client money attached.
- A banned IP or a complaint to a member firm is a real operational risk on a
  project whose entire premise is not losing other people's money.

The distinction that matters: **a person reading a public page in a browser is
the intended use. A script defeating a bot check is the thing being forbidden.**
Those are different acts even though the bytes are identical.

## 4. So: you download, I ingest

This is not a workaround, it is the sanctioned route, and the pipeline for it
already exists.

**One file a day covers the whole market.** Unlike the top-ten source — one
request per stock per day — IDX's broker summary is published per session. If it
downloads as one file covering every stock, then **~4,500 files is the entire
history of the exchange**, at full depth, with no censoring at all.

What that changes, concretely:

| today, from the free top-ten source | with the full rekap |
|---|---|
| 85–90% of volume visible | **100%** |
| positions are *brackets* | positions are **exact** |
| cost basis unavailable | cost basis **exact** |
| 63% of a year's flow has a proven direction | **all of it** |
| `idxbot.broker_bounds` is required | it becomes **unnecessary** |

### How to hand it over

```bash
# drop whatever you downloaded here
cp ~/Downloads/*.csv  data/inbox/
python3 scripts/broker_collect.py --ingest data/inbox
```

`scripts/import_broker_data.py` already reads CSV, Excel and saved HTML, and
already distinguishes a **running-trade** export (buyer and seller code on every
print) from a **broker-summary** table, flagging which it found.

**Send me one file first.** I do not know IDX's exact column layout and I am not
going to guess it — the last time you sent a real sample it exposed three parser
bugs in one go (dropped magnitude suffixes, two brokers merged into one row, and
the column order being Val/Lot/Avg rather than Lot/Val/Avg). One file and I will
have the importer matching it exactly, then you can bulk-drop the rest.

## 4b. THE ANSWER: buy it from someone licensed to sell it

You said there is always a way, and that sites with no live trade still publish
full broker summaries from previous days. You were right, and here is the way
they do it — none of them are defeating Cloudflare. They are **licensed
redistributors**, and at least one of them sells API access.

**`https://api.sectors.app/v2/broker-summary/{symbol}/`**
(Sectors, operated by Supertype Pte. Ltd. / PT Supertype Teknologi Nusantara)

Their own specification says the response

> *"lists **every broker active on that day** with buy/sell/net values, lots,
> frequency, and weighted avg price per share."*

Not a top ten. The worked example in their docs is a broker that traded **55
lots** — a number no top-ten table on this exchange would ever show. That is
the full rekap, and it arrives as JSON with an API key instead of as a file you
have to download by hand.

### What it gives that the free route cannot

| | free top-ten route | licensed API |
|---|---|---|
| brokers listed | 10 buyers, 10 sellers | **every one** |
| volume visible | 85–90% | **100%** |
| positions | brackets | **exact numbers** |
| rupiah values | 4 significant figures in billions, 2 in trillions (**4.55% error on a busy day**) | **exact integers** |
| trade counts (`bfreq`/`sfreq`) | not available | **included** |
| collection | 1 request per stock per day | 1 request per stock per **fortnight** |

That last row is not a convenience, it is the economics. **One call costs one
credit and returns up to 14 days.** A per-day fetcher would burn 14× the money
for identical data, so `idxbot.data.sectors` never makes one: it resolves any
requested day to a fixed fortnight, buys and caches the whole fortnight, and
returns the slice. Ask for 400 sessions of one ticker and it costs **about 40
credits, not 400**.

The trade-count field deserves its own line. `bfreq` is the number of trades
behind the volume, and it separates *one institutional block* from *a thousand
retail tickets* — which is the distinction most of the bandarmology folklore in
this market is actually reaching for and has never been able to measure.

### The catch, stated plainly

**It is not free.** The API is gated behind their Insider subscription. I could
not read the price from this container — their host returns 429 to it — so
**check `sectors.app/pricing` yourself before committing.** I am not going to
quote you a number I could not verify.

### What it costs us, in credits

```
  10 names x 400 sessions   ~400 credits      (the frozen protocol's panel)
   1 name  x 400 sessions   ~ 40 credits
  10 names x  10 sessions   ~ 10 credits      (enough to validate)
```

`scripts/broker_collect.py --sectors ...` prints that bill and then **stops**.
It will not spend a credit without `--yes`. A backfill that discovers its own
cost afterwards is a backfill nobody agreed to.

### It gets validated before it gets trusted

Buying data does not make it correct. A vendor can mis-map a column, quote
shares where the exchange quotes lots, or publish a different board — and every
one of those produces numbers that look completely normal. We already hold 416
sessions of BBCA from a wholly independent route, so:

```bash
SECTORS_API_KEY=...  python3 scripts/sectors_validate.py --ticker BBCA
```

runs four checks, in order of what they would catch:

1. **internal** — `value = lots x 100 x average` on the vendor's own rows.
   Catches a mis-mapped column with no second source needed.
2. **overlap** — every broker the free route names must show the same lots.
   Two unrelated pipelines landing on the same integer is not something a
   parsing bug does by accident.
3. **depth** — the paid route must list *more* brokers, and the free route's
   ten must be the ten largest of them. If the "full" rekap is also a top ten,
   the reason for paying evaporates and this says so.
4. **closure** — across all brokers, lots bought must equal lots sold. A
   complete rekap closes exactly; a censored one *cannot*. This is the check
   the free route can never pass, and passing it is what licenses collapsing
   every interval in this project to a point.

Until all four pass, nothing downstream is allowed to treat this route as
complete. One credit buys the whole validation.

## 5. The other sites — reachable, but not mine to enable

From this container these respond:

| site | reachable | note |
|---|---|---|
| sahamgain.com | yes | |
| new.sahamidx.com | yes | |
| pasardana.id | yes | |
| sabarkaya.com | yes | the bandarmology site you mentioned |
| britama.com | no | proxy error |
| **idx.co.id** | **no** | Cloudflare |

I have **not** wired a fetcher to any of them, and `data.broker_allowed_hosts`
in the config is still empty. That is deliberate and it is the repo's own
standing rule: a host goes on that list only after *you* have checked its
licensing, because every one of these is redistributing IDX market data under
terms I cannot see from here.

If you want one of them enabled, say which, and I will look at its terms with
you before writing a line of fetch code.

## 6. What I am doing in the meantime

The top-ten source is legitimate, works, and is already collecting. It is not a
substitute for the full rekap but it is not nothing either: 85–90% of volume,
with the remainder rigorously bracketed rather than guessed, and a panel sized
from the frozen protocol's own power requirement — 10 names × 400 sessions,
which is what it takes to detect a real flow edge at d = 0.10.

If the IDX file lands, that panel gets replaced by an exact one and every
bracket in this project collapses to a number.
