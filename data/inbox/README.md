# Drop broker data here

This folder is the only sanctioned way broker data enters this repo. Export it
from your own platform by hand, drop the file here, then run:

```
python3 scripts/import_broker_data.py --report
```

Any CSV / TSV / JSON / JSONL / XLSX works. Headers are matched against English
and Indonesian aliases the repo already knows (`kode`, `harga`, `lot`, `pembeli`,
`penjual`, `tanggal`, `volume beli`, ...), so a raw export usually needs no
editing. Name files `TICKER_YYYYMMDD_whatever.csv` if the date is not inside the
file — an undated broker row cannot be joined to a price bar and will be refused.

## Running trade beats broker summary, by a lot

Every print on IDX carries a **buyer** member code and a **seller** member code.
Broker summary is not a separate product — it is those prints summed by code. So
if you can export the prints, the repo reconstructs the **complete** rekap for
all ~90 members, at any resolution you like, including intraday.

| you export | you get |
|---|---|
| **running trade** (buyer + seller per print) | the full rekap, any resolution, balances exactly |
| broker summary | only the rows your platform chose to print, end of day |

The importer tells the two apart automatically and labels each file `FULL REKAP`
or `top-N only`.

**How it knows:** a complete rekap must balance — total buy lots equal total sell
lots, because every share bought was sold. A truncated top-N table does not
balance, and the importer reports the gap. Measured against the free IndoPremier
page for BBCA on 2026-08-18: it printed 10 rows per side summing to 943,424 buy
lots while its own footer reported 1.1M total, so ~12–14% of that session sat in
members the page never showed.

## What this will not do

It will not scrape. IDX prohibits scraping; Stockbit's terms forbid "data mining,
robots, spiders, or similar" without written consent; no Indonesian retail
platform publishes a documented API for this. There is no legitimate automated
route from a retail account, so the repo does not pretend there is one. A person
exporting their own data and a program reading the file is legitimate, and that
is what this is.

Nothing here is ever deleted, and nothing imported is re-fetched.
