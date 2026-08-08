#!/usr/bin/env node
/**
 * Fetch IDX broker summary with a real browser — run this on YOUR machine.
 *
 * Why this is a separate script and not part of the Python engine:
 *
 * idx.co.id sits behind Cloudflare, which blocks plain HTTP clients (curl,
 * requests) with a 403 but lets real browsers through after a JS challenge. A
 * headless Chromium *is* a real browser, so it passes — from an ordinary
 * connection. It does not pass from a locked-down build sandbox, which is why
 * this ships as something you run locally rather than something the engine
 * calls for you.
 *
 * Setup (once):
 *     npm install playwright
 *     npx playwright install chromium
 *
 * Use:
 *     node scripts/fetch_broker_summary.mjs BBCA
 *     node scripts/fetch_broker_summary.mjs BBCA 2026-08-06
 *     node scripts/fetch_broker_summary.mjs BBCA --headed     # watch it work
 *
 * Output goes straight to data/broker_summary/<TICKER>.csv, which the CSV
 * provider already reads. Then:
 *     idxbot analyze BBCA
 *     idxbot screen --providers csv
 *
 * A note on terms of service: this reads a page you could open yourself, for
 * your own use. Redistributing IDX market data is a different matter and is
 * restricted — see docs/LIVE_DATA.md. Your call, made knowingly.
 */

import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

const args = process.argv.slice(2);
const TICKER = (args[0] || '').toUpperCase();
const DATE = args.find(a => /^\d{4}-\d{2}-\d{2}$/.test(a)) || todayISO();
const HEADED = args.includes('--headed');
const OUT_DIR = path.join(process.cwd(), 'data', 'broker_summary');

if (!TICKER) {
  console.error('usage: node scripts/fetch_broker_summary.mjs TICKER [YYYY-MM-DD] [--headed]');
  process.exit(2);
}

function todayISO() {
  // IDX trades in WIB (UTC+7); use that day, not the machine's timezone.
  const now = new Date(Date.now() + 7 * 3600 * 1000);
  return now.toISOString().slice(0, 10);
}

/** Indonesian number formats: 1.234.567,89 and bare 1.234 (thousands). */
function toNumber(raw) {
  if (raw == null) return 0;
  let s = String(raw).trim().replace(/[()]/g, '-');
  if (!s || s === '-' || s === '—') return 0;
  s = s.replace(/[^\d,.\-]/g, '');
  if (!s) return 0;
  const lastDot = s.lastIndexOf('.'), lastComma = s.lastIndexOf(',');
  if (lastDot >= 0 && lastComma >= 0) {
    s = lastComma > lastDot ? s.replace(/\./g, '').replace(',', '.') : s.replace(/,/g, '');
  } else {
    const sep = lastDot >= 0 ? '.' : (lastComma >= 0 ? ',' : '');
    if (sep) {
      const idx = s.lastIndexOf(sep);
      const tail = s.slice(idx + 1), head = s.slice(0, idx);
      const grouped = tail.length === 3 && /^\d+$/.test(tail) && head !== '' && head !== '-';
      if (grouped || s.split(sep).length > 2) s = s.split(sep).join('');
      else if (sep === ',') s = s.replace(',', '.');
    }
  }
  const n = parseFloat(s);
  return Number.isFinite(n) ? n : 0;
}

const LOT = 100;

async function main() {
  console.log(`Fetching broker summary: ${TICKER} on ${DATE}`);
  const browser = await chromium.launch({ headless: !HEADED });
  const ctx = await browser.newContext({
    locale: 'id-ID',
    timezoneId: 'Asia/Jakarta',
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' +
               '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
  });
  const page = await ctx.newPage();

  // The rendered table is fed by an XHR. Capturing the JSON is far more robust
  // than scraping DOM cells, which change with every site redesign.
  const payloads = [];
  page.on('response', async (res) => {
    const url = res.url();
    if (!/[Bb]roker[Ss]ummary|broker-summary/.test(url)) return;
    try {
      const ct = res.headers()['content-type'] || '';
      if (ct.includes('json')) payloads.push(await res.json());
    } catch { /* non-JSON or already consumed */ }
  });

  console.log('  opening idx.co.id (Cloudflare challenge may take ~10s)...');
  await page.goto('https://www.idx.co.id/en/market-data/trading-summary/broker-summary/',
                  { waitUntil: 'domcontentloaded', timeout: 90000 });
  await page.waitForTimeout(12000);

  const title = await page.title();
  if (/just a moment|attention required/i.test(title)) {
    console.error('  ! Cloudflare did not clear. Re-run with --headed and solve it once;');
    console.error('    the cookie usually persists for the session.');
    await browser.close();
    process.exit(1);
  }
  console.log(`  page loaded: ${title}`);

  // Drive the form. Selectors are best-effort across redesigns; --headed lets
  // you see exactly where it fails.
  try {
    const codeInput = page.locator('input[placeholder*="ode" i], input[type="search"]').first();
    await codeInput.fill(TICKER, { timeout: 15000 });
    await page.keyboard.press('Enter');
    await page.waitForTimeout(2500);
    const dateInput = page.locator('input[type="date"]').first();
    if (await dateInput.count()) { await dateInput.fill(DATE); await page.keyboard.press('Enter'); }
    const go = page.locator('button:has-text("Cari"), button:has-text("Search"), button[type="submit"]').first();
    if (await go.count()) await go.click({ timeout: 10000 });
    await page.waitForTimeout(9000);
  } catch (e) {
    console.warn('  ! form interaction failed:', e.message.split('\n')[0]);
    console.warn('    Re-run with --headed to see the page and adjust selectors.');
  }

  let rows = payloads.flatMap(extractRows);
  if (!rows.length) rows = await scrapeDom(page);

  await browser.close();

  if (!rows.length) {
    console.error('\n  No broker rows captured.');
    console.error('  Fallback that always works: copy the table off the screen and run');
    console.error(`      idxbot paste ${TICKER} --date ${DATE}`);
    process.exit(1);
  }

  writeCsv(rows);
}

function extractRows(payload) {
  const list = findArray(payload);
  if (!list) return [];
  return list.map((r) => {
    const g = (...keys) => {
      for (const k of keys) {
        const hit = Object.keys(r).find((x) => x.toLowerCase() === k.toLowerCase());
        if (hit != null && r[hit] != null) return r[hit];
      }
      return null;
    };
    const code = String(g('BrokerCode', 'Broker', 'Kode', 'code') || '').trim().toUpperCase();
    if (!/^[A-Z]{2,3}$/.test(code)) return null;
    const buyLot = toNumber(g('BLot', 'BuyLot', 'BuyVolume', 'VolumeBeli'));
    const sellLot = toNumber(g('SLot', 'SellLot', 'SellVolume', 'VolumeJual'));
    const buyVal = toNumber(g('BVal', 'BuyValue', 'NilaiBeli'));
    const sellVal = toNumber(g('SVal', 'SellValue', 'NilaiJual'));
    return { broker: code, buy_lot: buyLot, buy_val: buyVal, sell_lot: sellLot, sell_val: sellVal };
  }).filter(Boolean);
}

/** Pull the first array-of-objects out of an arbitrary JSON envelope. */
function findArray(node, depth = 0) {
  if (depth > 6 || node == null) return null;
  if (Array.isArray(node)) return node.length && typeof node[0] === 'object' ? node : null;
  if (typeof node === 'object') {
    for (const key of ['data', 'Data', 'results', 'Results', 'rows', 'items']) {
      if (node[key]) { const f = findArray(node[key], depth + 1); if (f) return f; }
    }
    for (const v of Object.values(node)) { const f = findArray(v, depth + 1); if (f) return f; }
  }
  return null;
}

async function scrapeDom(page) {
  return page.evaluate(() => {
    const out = [];
    for (const table of document.querySelectorAll('table')) {
      for (const tr of table.querySelectorAll('tr')) {
        const cells = [...tr.querySelectorAll('td')].map((td) => td.innerText.trim());
        if (cells.length < 3) continue;
        const codes = cells.map((c, i) => ({ c, i })).filter(({ c }) => /^[A-Z]{2,3}$/.test(c));
        if (!codes.length) continue;
        out.push({ cells, codeIndexes: codes.map((x) => x.i) });
      }
    }
    return out;
  }).then((raw) => raw.flatMap(({ cells, codeIndexes }) => {
    // Side-by-side buyers|sellers is the common IDX layout.
    if (codeIndexes.length >= 2) {
      const [l, r] = codeIndexes;
      const bn = cells.slice(l + 1, r).map(toNumber).filter(Boolean);
      const sn = cells.slice(r + 1).map(toNumber).filter(Boolean);
      return [
        { broker: cells[l], buy_lot: bn[0] || 0, buy_val: Math.max(...bn, 0), sell_lot: 0, sell_val: 0 },
        { broker: cells[r], buy_lot: 0, buy_val: 0, sell_lot: sn[0] || 0, sell_val: Math.max(...sn, 0) },
      ];
    }
    const i = codeIndexes[0];
    const n = cells.slice(i + 1).map(toNumber);
    return [{ broker: cells[i], buy_lot: n[0] || 0, buy_val: n[1] || 0,
              sell_lot: n[2] || 0, sell_val: n[3] || 0 }];
  }));
}

function writeCsv(rows) {
  const merged = new Map();
  for (const r of rows) {
    const cur = merged.get(r.broker) || { buy_lot: 0, buy_val: 0, sell_lot: 0, sell_val: 0 };
    for (const k of ['buy_lot', 'buy_val', 'sell_lot', 'sell_val']) cur[k] += r[k] || 0;
    merged.set(r.broker, cur);
  }

  const header = 'date,ticker,broker,buy_lot,buy_val,buy_avg,sell_lot,sell_val,sell_avg,source';
  const lines = [...merged.entries()].map(([broker, v]) => {
    const bAvg = v.buy_lot > 0 ? v.buy_val / (v.buy_lot * LOT) : 0;
    const sAvg = v.sell_lot > 0 ? v.sell_val / (v.sell_lot * LOT) : 0;
    return [DATE, TICKER, broker, v.buy_lot, v.buy_val, bAvg.toFixed(2),
            v.sell_lot, v.sell_val, sAvg.toFixed(2), 'idx_browser'].join(',');
  });

  fs.mkdirSync(OUT_DIR, { recursive: true });
  const file = path.join(OUT_DIR, `${TICKER}.csv`);

  // Re-fetching a day should correct it, not duplicate it.
  let existing = [];
  if (fs.existsSync(file)) {
    existing = fs.readFileSync(file, 'utf8').split('\n')
      .filter((l) => l.trim() && !l.startsWith('date,'))
      .filter((l) => !l.startsWith(`${DATE},${TICKER},`));
  }
  fs.writeFileSync(file, [header, ...existing, ...lines].join('\n') + '\n');

  const totalBuy = [...merged.values()].reduce((a, v) => a + v.buy_lot, 0);
  const totalSell = [...merged.values()].reduce((a, v) => a + v.sell_lot, 0);
  const match = Math.max(totalBuy, totalSell) > 0
    ? 1 - Math.abs(totalBuy - totalSell) / Math.max(totalBuy, totalSell) : 0;

  console.log(`\n  brokers      : ${merged.size}`);
  console.log(`  total buy    : ${totalBuy.toLocaleString()} lots`);
  console.log(`  total sell   : ${totalSell.toLocaleString()} lots`);
  console.log(`  buy/sell fit : ${(match * 100).toFixed(1)}%  ` +
              `[${match > 0.98 ? 'OK' : 'CHECK — every lot bought is a lot sold'}]`);
  console.log(`\n  wrote -> ${file}`);
  console.log(`  next: idxbot analyze ${TICKER}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
