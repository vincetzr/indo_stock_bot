"""Render the brief as a page you can actually read twice a day.

WHY NOT JUST WRAP THE TERMINAL OUTPUT IN <pre>
------------------------------------------------
Because the brief is a dashboard, not a document. It is scanned, not read
top-to-bottom, and the thing a reader needs at 07:00 WIB is which rows deserve
attention — which is exactly what a wall of monospace hides. Real tables let
the net column carry a severity stripe, event tags become pills, and the
overnight column sit where the eye lands first.

THE ONE STRUCTURAL DEVICE, AND WHY IT IS NOT DECORATION
--------------------------------------------------------
Every section carries an epistemic stamp — *arithmetic*, *derived*,
*historical*, *not a signal*. That is the whole ethos of this repository
rendered as layout: four instruments were run to their end here and none
produced an edge that survived costs, so a page that presented a breadth
percentage and a conditional excess in the same visual register would be
lying about what it knows. The stamp is the design carrying a real
distinction, which is the only justification a structural device gets.
"""

from __future__ import annotations

import html as _h
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

FONTS = ("https://fonts.googleapis.com/css2?"
         "family=Archivo:wght@500;600;700&"
         "family=IBM+Plex+Mono:wght@400;500;600&"
         "family=IBM+Plex+Sans:wght@400;500;600&display=swap")

CSS = """
:root{
  --ground:#f7f8f8; --surface:#ffffff; --sunk:#eef1f1;
  --ink:#16191b; --muted:#5e6a70; --faint:#8a969c; --line:#dfe4e5;
  --accent:#0f766e; --accent-soft:#d7ece9;
  --up:#157f3d; --down:#b4342a; --warn:#a16207;
  --up-soft:#dcf0e3; --down-soft:#fae0dd; --warn-soft:#faeece;
  --shadow:0 1px 2px rgba(20,30,32,.06),0 8px 24px -16px rgba(20,30,32,.25);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#101315; --surface:#171b1e; --sunk:#1e2427;
    --ink:#e8ebec; --muted:#94a3aa; --faint:#6b797f; --line:#2a3236;
    --accent:#2dd4bf; --accent-soft:#12312f;
    --up:#4ade80; --down:#f87171; --warn:#fbbf24;
    --up-soft:#12291b; --down-soft:#2c1614; --warn-soft:#2a2110;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -16px rgba(0,0,0,.8);
  }
}
:root[data-theme="dark"]{
  --ground:#101315; --surface:#171b1e; --sunk:#1e2427;
  --ink:#e8ebec; --muted:#94a3aa; --faint:#6b797f; --line:#2a3236;
  --accent:#2dd4bf; --accent-soft:#12312f;
  --up:#4ade80; --down:#f87171; --warn:#fbbf24;
  --up-soft:#12291b; --down-soft:#2c1614; --warn-soft:#2a2110;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -16px rgba(0,0,0,.8);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
  font-size:15px; line-height:1.55;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1140px;margin:0 auto;padding:40px 22px 80px;
      display:flex;flex-direction:column;gap:26px}
h1,h2,h3{font-family:Archivo,ui-sans-serif,system-ui,sans-serif;
         text-wrap:balance;margin:0;letter-spacing:-.015em}
h1{font-size:clamp(30px,4.4vw,42px);font-weight:700;line-height:1.08}
h2{font-size:19px;font-weight:600}
.mono,td.n,th.n{font-family:"IBM Plex Mono",ui-monospace,monospace;
                font-variant-numeric:tabular-nums}

header.top{display:flex;flex-direction:column;gap:10px;
           border-bottom:2px solid var(--ink);padding-bottom:18px}
.kicker{font-family:"IBM Plex Mono",monospace;font-size:11px;
        letter-spacing:.16em;text-transform:uppercase;color:var(--accent);
        font-weight:600}
.sub{color:var(--muted);font-size:14px}

section.card{background:var(--surface);border:1px solid var(--line);
             border-radius:4px;box-shadow:var(--shadow);overflow:hidden}
.head{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;
      padding:16px 20px;border-bottom:1px solid var(--line)}
.head .num{font-family:"IBM Plex Mono",monospace;font-size:12px;
           color:var(--faint);font-weight:500}
/* the epistemic stamp — the page's only structural device, and it encodes a
   real distinction rather than decorating one */
.stamp{margin-left:auto;font-family:"IBM Plex Mono",monospace;font-size:10px;
       letter-spacing:.1em;text-transform:uppercase;padding:3px 8px;
       border-radius:2px;font-weight:600;white-space:nowrap}
.stamp.exact{background:var(--accent-soft);color:var(--accent)}
.stamp.derived{background:var(--sunk);color:var(--muted)}
.stamp.hist{background:var(--warn-soft);color:var(--warn)}
.stamp.nosig{background:var(--down-soft);color:var(--down)}
.body{padding:18px 20px}
.note{color:var(--muted);font-size:13.5px;max-width:68ch}
.note+.note{margin-top:8px}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th{text-align:left;font-weight:600;font-size:11px;letter-spacing:.06em;
   text-transform:uppercase;color:var(--faint);padding:8px 10px;
   border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
tr:last-child td{border-bottom:none}
td.n,th.n{text-align:right}
tbody tr:hover{background:var(--sunk)}
.up{color:var(--up)} .down{color:var(--down)} .warn{color:var(--warn)}
.dim{color:var(--faint)}
.tick{font-family:"IBM Plex Mono",monospace;font-weight:600}
.pill{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:10px;
      letter-spacing:.06em;padding:2px 6px;border-radius:2px;margin-right:4px;
      background:var(--sunk);color:var(--muted);font-weight:600}
.pill.hot{background:var(--warn-soft);color:var(--warn)}
.pill.bad{background:var(--down-soft);color:var(--down)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));
       gap:1px;background:var(--line);border:1px solid var(--line)}
.stat{background:var(--surface);padding:12px 14px}
.stat .lab{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
           color:var(--faint);font-weight:600}
.stat .val{font-family:"IBM Plex Mono",monospace;font-size:21px;
           font-weight:600;margin-top:3px;font-variant-numeric:tabular-nums}
.stat .sub2{font-size:11.5px;color:var(--muted);margin-top:1px}
.lede{background:var(--surface);border:1px solid var(--line);
      border-left:3px solid var(--accent);border-radius:4px;padding:20px 22px;
      box-shadow:var(--shadow);display:flex;flex-direction:column;gap:9px}
.lede p{margin:0;font-size:16px;line-height:1.5;max-width:74ch}
.callout{background:var(--sunk);border-left:3px solid var(--warn);
         padding:12px 15px;border-radius:2px;font-size:13.5px;
         color:var(--muted);max-width:74ch}
.callout strong{color:var(--ink)}
.grp{font-family:"IBM Plex Mono",monospace;font-size:10.5px;
     letter-spacing:.1em;text-transform:uppercase;color:var(--faint);
     font-weight:600;padding-top:14px}
.pc{display:flex;flex-direction:column;gap:5px;padding:12px 0;
    border-bottom:1px solid var(--line)}
.pc:last-child{border-bottom:none}
.pc .row1{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.pc .names{font-family:"IBM Plex Mono",monospace;font-size:12.5px}
.news{display:flex;flex-direction:column;gap:0}
.item{display:flex;gap:12px;padding:9px 0;border-bottom:1px solid var(--line);
      align-items:baseline;flex-wrap:wrap}
.item:last-child{border-bottom:none}
.item time{font-family:"IBM Plex Mono",monospace;font-size:11.5px;
           color:var(--faint);white-space:nowrap;min-width:96px}
.item .t{flex:1;min-width:260px;font-size:14px}
.item .src{font-size:11.5px;color:var(--faint)}
footer{color:var(--faint);font-size:12.5px;border-top:1px solid var(--line);
       padding-top:16px;max-width:74ch}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (prefers-reduced-motion:no-preference){
  tbody tr{transition:background .12s ease}
}
@media (max-width:640px){
  .wrap{padding:26px 14px 60px}
  .item time{min-width:0}
}
"""


def e(x) -> str:
    return _h.escape(str(x), quote=True)


def num(x, d: int = 2, pct: bool = True, sign: bool = True) -> str:
    if x is None or not np.isfinite(x):
        return '<span class="dim">—</span>'
    s = f"{x:{'+' if sign else ''}.{d}{'%' if pct else 'f'}}"
    cls = "up" if x > 0 else ("down" if x < 0 else "dim")
    return f'<span class="{cls}">{s}</span>'


def plain(x, d: int = 2, pct: bool = True) -> str:
    if x is None or not np.isfinite(x):
        return '<span class="dim">—</span>'
    return f"{x:.{d}{'%' if pct else 'f'}}"


STAMPS = {
    "exact": ("arithmetic", "exact"),
    "derived": ("derived from returns", "derived"),
    "hist": ("historical frequency", "hist"),
    "nosig": ("not a signal", "nosig"),
}


def card(n: str, title: str, stamp: str, inner: str) -> str:
    label, cls = STAMPS[stamp]
    return (f'<section class="card"><div class="head">'
            f'<span class="num">{e(n)}</span><h2>{e(title)}</h2>'
            f'<span class="stamp {cls}">{e(label)}</span></div>'
            f'<div class="body">{inner}</div></section>')


def _tags(v) -> str:
    if not isinstance(v, (list, tuple)) or not v:
        return ""
    hot = {"UMA", "SUSPEND", "DELIST"}
    return "".join(
        f'<span class="pill {"bad" if t in hot else "hot"}">{e(t)}</span>'
        for t in v)


# ==========================================================================
def render(ctx: Dict[str, object]) -> str:
    day, session = ctx["day"], ctx["session"]
    parts: List[str] = []

    parts.append(
        f'<header class="top"><div class="kicker">IDX · '
        f'{e(session).upper()} SESSION</div>'
        f'<h1>Jakarta, {e(day.strftime("%-d %B %Y"))}</h1>'
        f'<div class="sub">Bars through {e(day.date())}. '
        f'Generated {e(ctx["now"].strftime("%Y-%m-%d %H:%M"))} WIB. '
        f'A description plus historical frequencies — not a forecast, '
        f'not a buy list.</div></header>')

    if ctx.get("warn"):
        parts.append(f'<div class="callout"><strong>Partial refresh.</strong> '
                     f'{e(ctx["warn"])}</div>')

    # ---- the read
    parts.append('<div class="lede">'
                 + "".join(f"<p>{e(l)}</p>" for l in ctx["headline"])
                 + '<div class="callout">Every clause above is a printed '
                   'number from the sections below. Four instruments were run '
                   'to their end in this repository — aggregate broker flow, '
                   'broker identity, investor class and price structure — and '
                   '<strong>none produced an edge that survived costs</strong>.'
                   '</div></div>')

    # ---- 1 overnight
    if ctx.get("overnight") is not None and not ctx["overnight"].empty:
        parts.append(card("01", "Overnight — what moved while Jakarta was shut",
                          "exact", _overnight(ctx["overnight"],
                                              ctx.get("sens"))))
    # ---- 2 state
    parts.append(card("02", "Market state", "exact",
                      _state(ctx["breadth"], ctx["regime"], ctx["limits"],
                             ctx["movers"])))
    # ---- 3 co-movement
    if ctx.get("comovement") is not None and not ctx["comovement"].empty:
        parts.append(card("03", "What moved together", "derived",
                          _comovement(ctx["comovement"])))
    # ---- 4 news
    if ctx.get("market_news") is not None:
        parts.append(card("04", "What is being said", "derived",
                          _news(ctx["market_news"], ctx.get("ticker_news"),
                                ctx.get("news_caveat", ""))))
    # ---- 5 watchlist
    if ctx.get("watchlist") is not None and not ctx["watchlist"].empty:
        parts.append(card("05", "Watchlist", "hist",
                          _watchlist(ctx["watchlist"], ctx["k"])))
    # ---- 6 conditional
    if ctx.get("null"):
        parts.append(card("06", f"Is the run over — {ctx['k']} sessions on",
                          "nosig", _conditional(ctx["null"], ctx["table_meta"])))

    parts.append(
        '<footer>The conditional cells beat their permutation null, so the '
        'state conditioning is real. That does not make it tradeable: the '
        'cells are post-hoc, in-sample and uncorrected for 54 tests, and H13 '
        'measured very nearly the same thing net-negative after costs. The '
        '24-month holdout is untouched and every reference distribution here '
        'is estimated on pre-holdout rows only.</footer>')

    return (f'<title>Jakarta Session Brief</title>'
            f'<link rel="preconnect" href="https://fonts.googleapis.com">'
            f'<link rel="preconnect" href="https://fonts.gstatic.com" '
            f'crossorigin>'
            f'<link rel="stylesheet" href="{FONTS}">'
            f"<style>{CSS}</style>"
            f'<div class="wrap">{"".join(parts)}</div>')


def _overnight(Bd: pd.DataFrame, sens) -> str:
    rows = []
    for group in Bd["group"].unique():
        rows.append(f'<tr><td colspan="7" class="grp">{e(group)}</td></tr>')
        for _, r in Bd[Bd["group"] == group].iterrows():
            if not np.isfinite(r["overnight"]):
                ov = ('<span class="dim">closes before Jakarta</span>'
                      if not r["after_jakarta"]
                      else '<span class="dim">no print yet</span>')
            else:
                ov = f'<strong>{num(r["overnight"])}</strong>'
            note = f'<span class="dim">{e(r["proxy"])}</span>' \
                if r["proxy"] else ""
            rows.append(
                f'<tr><td class="tick">{e(r["name"])}</td>'
                f'<td class="n">{plain(r["last"], 2, False)}</td>'
                f'<td class="n">{ov}</td><td class="n">{num(r["d5"], 1)}</td>'
                f'<td class="n">{num(r["d21"], 1)}</td>'
                f'<td class="n dim">{e(str(r["asof"])[:10])}</td>'
                f'<td>{note}</td></tr>')
    tbl = (f'<div class="tablewrap"><table><thead><tr><th></th>'
           f'<th class="n">Last</th><th class="n">Overnight</th>'
           f'<th class="n">5d</th><th class="n">1m</th>'
           f'<th class="n">As of</th><th></th></tr></thead>'
           f'<tbody>{"".join(rows)}</tbody></table></div>')
    note = ('<p class="note">Jakarta closes 08:50 UTC. New York closes 20:00 '
            'and London 15:30 <em>on the same date</em>, so their bar landed '
            'hours after Jakarta\'s — that is the overnight column. Tokyo, '
            'Hong Kong and Shanghai close before Jakarta, so for IDX there is '
            'no overnight number, only context.</p>')
    s = ""
    if sens is not None and not sens.empty:
        srows = "".join(
            f'<tr><td class="tick">{e(r["name"])}</td>'
            f'<td class="n">{num(r["r"], 3, pct=False)}</td>'
            f'<td class="n dim">[{r["lo"]:+.2f}, {r["hi"]:+.2f}]</td>'
            f'<td class="n">{r["z"]:+.1f}</td>'
            f'<td class="n dim">{r["stale"]:.0%}</td></tr>'
            for _, r in sens.head(10).iterrows())
        s = (f'<h3 style="margin-top:22px;font-size:13px;'
             f'text-transform:uppercase;letter-spacing:.08em;'
             f'color:var(--faint)">Which of these IDX has actually tracked'
             f'</h3><div class="tablewrap"><table><thead><tr><th></th>'
             f'<th class="n">Rank corr</th><th class="n">95% CI</th>'
             f'<th class="n">vs null</th><th class="n">Stale</th></tr></thead>'
             f'<tbody>{srows}</tbody></table></div>'
             f'<div class="callout" style="margin-top:14px">'
             f'<strong>Rank correlation, not Pearson.</strong> These series '
             f'carry kurtosis from 10 to 2,800; computing it the other way '
             f'first reported the S&amp;P as uncorrelated with IDX '
             f'(&minus;0.001) when the rank figure is +0.207. '
             f'<strong>Even the strongest is context, not a signal</strong> — '
             f'0.21 explains about 4% of variance, a fraction of the 56 bps '
             f'round trip.</div>')
    return note + tbl + s


def _state(b, reg, L, mv) -> str:
    def stat(lab, val, sub=""):
        return (f'<div class="stat"><div class="lab">{e(lab)}</div>'
                f'<div class="val">{val}</div>'
                f'<div class="sub2">{sub}</div></div>')
    cells = [
        stat("Advancing", f"{b['advancing']:.0%}",
             f"{b['n_names']} names traded"),
        stat("Above 20-day", f"{b['above_20d']:.0%}",
             f"200-day {b['above_200d']:.0%}"),
        stat("Dispersion", f"{b['dispersion']:.2%}",
             "cross-sectional spread"),
        stat("20d index vol", f"{reg.get('equal_vol', float('nan')):.1%}",
             f"{reg.get('equal_vol_pct', float('nan')):.0%} pct of 5y"),
        stat("At the band", f"{L['ara']}/{L['arb']}",
             "ARA / ARB, close test"),
        stat("250-day extremes", f"{b['new_highs']}/{b['new_lows']}",
             "highs / lows"),
    ]
    grid = f'<div class="stats">{"".join(cells)}</div>'
    hrows = "".join(
        f'<tr><td class="tick">{e(lab)}</td>'
        + "".join(f'<td class="n">{num(reg.get(f"{k}_{h}"), 1)}</td>'
                  for h in ("1d", "1w", "1m", "3m", "ytd"))
        + "</tr>"
        for k, lab in (("equal", "Equal-weighted"),
                       ("turnover", "Turnover-weighted")))
    idx = (f'<div class="tablewrap" style="margin-top:18px"><table><thead><tr>'
           f'<th></th><th class="n">1d</th><th class="n">1w</th>'
           f'<th class="n">1m</th><th class="n">3m</th><th class="n">YTD</th>'
           f'</tr></thead><tbody>{hrows}</tbody></table></div>'
           f'<p class="note">Turnover-weighted is a <em>proxy</em> for '
           f'cap-weighted — no shares-outstanding series exists here. A gap '
           f'between the rows is the big names carrying the tape while the '
           f'median name does not, or the reverse.</p>')
    def side(D, cls):
        return "".join(f'<span class="pill">{e(x.ticker)} '
                       f'<span class="{cls}">{x.ret:+.1%}</span></span>'
                       for x in D.itertuples())
    movers = (f'<div style="margin-top:16px"><div class="lab grp">Movers</div>'
              f'<div style="margin-top:6px">{side(mv["up"], "up")}</div>'
              f'<div style="margin-top:6px">{side(mv["down"], "down")}</div>'
              f'</div>')
    return grid + idx + movers


def _comovement(cm) -> str:
    rows = []
    for _, x in cm.iterrows():
        rows.append(
            f'<div class="pc"><div class="row1">'
            f'<strong>PC{int(x["pc"])}</strong>'
            f'<span class="dim">{x["var_share"]:.1%} of history · '
            f'{x["today_share"]:.1%} of today</span>'
            f'<span class="mono">{x["score_z"]:+.2f}z</span></div>'
            f'<div class="names">{e(" ".join(x["with"]))}</div>'
            f'<div class="names dim">vs {e(" ".join(x["against"][:6]))}</div>'
            f'</div>')
    return ('<p class="note">Components fitted on 250 sessions ending the day '
            '<em>before</em> this one, then today\'s cross-section projected '
            'onto them. A group is named by its members and nothing else — '
            'whether eight coal names moving together is "the coal trade" is '
            'your reading, not a finding.</p>' + "".join(rows))


def _news(M, T, caveat) -> str:
    items = "".join(
        f'<div class="item"><time>{e(str(r["published"])[:16])}</time>'
        f'<div class="t">{_tags(r["tags"])}{e(r["title"])}</div>'
        f'<div class="src">{e(r["source"])}</div></div>'
        for _, r in M.iterrows()) if M is not None and not M.empty else \
        '<p class="note">Nothing in the window.</p>'
    per = ""
    if T is not None and not T.empty:
        blocks = []
        for tk, g in T.groupby("ticker", sort=False):
            li = "".join(
                f'<div class="item"><time>{e(str(r["published"])[:10])}'
                f'{"" if r["recent"] else " ·old"}</time>'
                f'<div class="t">{_tags(r["tags"])}{e(r["title"])}</div>'
                f'</div>' for _, r in g.iterrows())
            blocks.append(f'<div style="margin-top:14px">'
                          f'<span class="tick">{e(tk)}</span>{li}</div>')
        per = ("".join(blocks)
               + '<p class="note">"old" items are corporate actions kept '
                 'beyond the 14-day window because they change what the price '
                 'series <em>means</em> — a rights issue is the named '
                 'adjustment trap.</p>')
    return (f'<div class="news">{items}</div>{per}'
            f'<div class="callout" style="margin-top:16px">{e(caveat)}</div>')


def _watchlist(Wl, k) -> str:
    rows = []
    for _, x in Wl.iterrows():
        net = x.get("net", np.nan)
        stripe = ""
        if np.isfinite(net):
            stripe = ("border-left:3px solid var(--up)" if net > 0
                      else "border-left:3px solid transparent")
        rows.append(
            f'<tr><td class="tick" style="{stripe}">{e(x["ticker"])}</td>'
            f'<td class="n">{plain(x["close"], 0, False)}</td>'
            f'<td class="n">{num(x.get("ret1"), 1)}</td>'
            f'<td>{e(x["leg"])}</td>'
            f'<td class="n">{int(x["run_days"])}</td>'
            f'<td class="n">{num(x["run_pct"], 0)}</td>'
            f'<td class="n">{x["run_z"]:+.2f}</td>'
            f'<td class="n">{num(x["give_pct"], 0)}</td>'
            f'<td class="n">{num(x.get("diff"), 2)}</td>'
            f'<td class="n dim">{plain(x.get("cost"), 2)}</td>'
            f'<td class="n">{num(net, 2)}</td>'
            f'<td class="n dim">{int(x["n_feat"])}</td>'
            f'<td>{_tags(x["events"].split(",") if x["events"] else [])}</td>'
            f'</tr>')
    return (f'<p class="note">Sorted by absolute move today — '
            f'<strong>not</strong> by attractiveness. There is no measured '
            f'attractiveness here and sorting by one would invent it.</p>'
            f'<div class="tablewrap"><table><thead><tr>'
            f'<th></th><th class="n">Close</th><th class="n">Today</th>'
            f'<th>Leg</th><th class="n">Since</th><th class="n">Run</th>'
            f'<th class="n">run_z</th><th class="n">Off ext</th>'
            f'<th class="n">Excess</th><th class="n">Cost</th>'
            f'<th class="n">Net</th><th class="n">Feat</th><th>Events</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            f'<p class="note"><strong>Excess</strong> is the historical mean '
            f'return of the cell this name currently occupies, over {k} '
            f'sessions, minus the equal-weighted return of all liquid names on '
            f'the same dates. <strong>Cost</strong> is 0.56% round trip plus '
            f'half a tick each way — a floor, since it assumes a one-tick '
            f'book. <strong>Net</strong> is a historical average minus that '
            f'floor, not an expectation for this name. <strong>Feat</strong> '
            f'counts how many of the eight registered features rank it '
            f'top-decile — a count, not a blend.</p>')


def _conditional(N, meta) -> str:
    verdict = ("The state conditioning carries information beyond chance."
               if N["p_max"] < 0.05
               else "Indistinguishable from shuffled labels — read nothing "
                    "from the cells.")
    return (f'<div class="stats">'
            f'<div class="stat"><div class="lab">Largest cell excess</div>'
            f'<div class="val">{N["obs_max_abs"]:.2%}</div>'
            f'<div class="sub2">of {N["n_cells"]} cells</div></div>'
            f'<div class="stat"><div class="lab">Under shuffled labels</div>'
            f'<div class="val">{N["null_max_abs_mean"]:.2%}</div>'
            f'<div class="sub2">p95 {N["null_max_abs_p95"]:.2%}</div></div>'
            f'<div class="stat"><div class="lab">Cell spread</div>'
            f'<div class="val">{N["obs_spread"]:.2%}</div>'
            f'<div class="sub2">null {N["null_spread_mean"]:.2%}</div></div>'
            f'<div class="stat"><div class="lab">p(null ≥ observed)</div>'
            f'<div class="val">{N["p_max"]:.3f}</div>'
            f'<div class="sub2">200 draws</div></div></div>'
            f'<p class="note" style="margin-top:16px">Reference: '
            f'{meta["n_rows"]:,} liquid pre-holdout bars, {e(meta["date_min"])}'
            f' to {e(meta["date_max"])}. The permutation null comes first — '
            f'reading a statistic against zero rather than its own shuffled '
            f'null has produced a confident wrong answer four separate times '
            f'in this repository.</p>'
            f'<div class="callout"><strong>{e(verdict)}</strong> It is still '
            f'in-sample and post-hoc. Fifty-four cells were computed and the '
            f'intervals are uncorrected, so roughly '
            f'{N["expected_false_cells"]:.0f} clear zero by luck; the largest '
            f'is the largest <em>of fifty-four</em> and is biased upward by '
            f'exactly the selection that found it. Treat it as a lead for a '
            f'pre-registered test, not a result.</div>')
