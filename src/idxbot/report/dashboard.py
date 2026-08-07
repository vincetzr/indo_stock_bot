"""Self-contained HTML dashboard.

No CDN, no external fonts, no network at render time: the file opens from disk
and works offline, which matters because it is generated after the close and
read on whatever machine happens to be nearby.

Charts are hand-rolled inline SVG. Colours come from a validated categorical
palette with light and dark steps chosen per surface; series identity is carried
by a legend plus direct labels, never by colour alone.
"""

from __future__ import annotations

import html
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..market import format_idr

# Validated categorical palette: light / dark steps for the same eight hues.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]

CSS = """
:root {
  color-scheme: light;
  --surface-0: #f7f7f5;
  --surface-1: #fcfcfb;
  --surface-2: #f0efec;
  --border:    #dcdbd6;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #78776f;
  --good:    #008300;
  --warning: #eda100;
  --critical:#e34948;
  --accent:  #2a78d6;
  --grid:    #e6e5e1;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#008300;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-0: #131312;
    --surface-1: #1a1a19;
    --surface-2: #252523;
    --border:    #3a3a37;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #94938a;
    --good:    #4fb04f;
    --warning: #c98500;
    --critical:#e66767;
    --accent:  #3987e5;
    --grid:    #2e2e2b;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #131312;
  --surface-1: #1a1a19;
  --surface-2: #252523;
  --border:    #3a3a37;
  --text-primary:   #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted:     #94938a;
  --good:    #4fb04f;
  --warning: #c98500;
  --critical:#e66767;
  --accent:  #3987e5;
  --grid:    #2e2e2b;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300;
}

* { box-sizing: border-box; }
body {
  margin: 0; padding: 0;
  background: var(--surface-0);
  color: var(--text-primary);
  font: 14px/1.5 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.wrap { max-width: 1280px; margin: 0 auto; padding: 28px 20px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 16px; margin: 36px 0 12px; letter-spacing: -0.01em; }
h3 { font-size: 14px; margin: 0 0 10px; }
.sub { color: var(--text-secondary); font-size: 13px; margin: 0 0 20px; }
.mono { font-variant-numeric: tabular-nums;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }

.banner {
  border-radius: 8px; padding: 12px 14px; margin: 0 0 22px;
  border: 1px solid var(--border); background: var(--surface-2);
  font-size: 13px; line-height: 1.5;
}
.banner.warn   { border-color: var(--warning); background: color-mix(in srgb, var(--warning) 12%, var(--surface-1)); }
.banner.danger { border-color: var(--critical); background: color-mix(in srgb, var(--critical) 12%, var(--surface-1)); }
.banner strong { display: block; margin-bottom: 3px; }

.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.tile {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px;
}
.tile .label { color: var(--text-secondary); font-size: 12px; text-transform: uppercase;
               letter-spacing: 0.04em; }
.tile .value { font-size: 26px; font-weight: 600; margin-top: 4px; letter-spacing: -0.02em; }
.tile .note  { color: var(--text-muted); font-size: 12px; margin-top: 2px; }

.card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px 18px; margin-bottom: 16px;
}
.card-head { display: flex; align-items: baseline; justify-content: space-between;
             gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
.card-head .ticker { font-size: 18px; font-weight: 600; }
.card-head a { color: var(--accent); text-decoration: none; font-size: 13px; }
.card-head a:hover { text-decoration: underline; }

.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 7px 10px; text-align: right; white-space: nowrap;
         border-bottom: 1px solid var(--border); }
th { color: var(--text-secondary); font-weight: 600; font-size: 12px;
     text-transform: uppercase; letter-spacing: 0.03em; text-align: right;
     position: sticky; top: 0; background: var(--surface-1); }
th:first-child, td:first-child { text-align: left; }
th:nth-child(2), td:nth-child(2) { text-align: left; }
tbody tr:hover { background: var(--surface-2); }

.pill { display: inline-block; padding: 1px 8px; border-radius: 999px;
        font-size: 11px; font-weight: 600; letter-spacing: 0.02em; }
.pill.STRONG { background: color-mix(in srgb, var(--good) 20%, transparent); color: var(--good); }
.pill.SIGNAL { background: color-mix(in srgb, var(--accent) 20%, transparent); color: var(--accent); }
.pill.WATCH  { background: color-mix(in srgb, var(--warning) 22%, transparent); color: var(--warning); }
.pill.NONE   { background: var(--surface-2); color: var(--text-muted); }
.pill.TAKE   { background: color-mix(in srgb, var(--good) 20%, transparent); color: var(--good); }
.pill.SKIP   { background: var(--surface-2); color: var(--text-muted); }

.scorebar { display: inline-block; vertical-align: middle; width: 62px; height: 6px;
            background: var(--surface-2); border-radius: 3px; overflow: hidden;
            margin-left: 8px; }
.scorebar i { display: block; height: 100%; border-radius: 3px; background: var(--accent); }

.components { display: grid; grid-template-columns: 150px 1fr 46px; gap: 6px 10px;
              align-items: center; font-size: 12px; }
.components .name { color: var(--text-secondary); }
.components .track { height: 7px; background: var(--surface-2); border-radius: 4px; }
.components .track i { display: block; height: 100%; border-radius: 4px; }
.components .num { text-align: right; color: var(--text-secondary); }

.legend { display: flex; flex-wrap: wrap; gap: 12px; margin: 8px 0 4px; font-size: 12px;
          color: var(--text-secondary); }
.legend span { display: inline-flex; align-items: center; gap: 5px; }
.legend i { width: 10px; height: 2px; border-radius: 1px; display: inline-block; }
.legend i.sw { height: 10px; border-radius: 2px; }

.chart { position: relative; }
.chart svg { display: block; width: 100%; height: auto; }
.tip {
  position: absolute; pointer-events: none; opacity: 0; transition: opacity .08s;
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 6px;
  padding: 6px 9px; font-size: 12px; white-space: nowrap; z-index: 5;
  box-shadow: 0 2px 8px rgba(0,0,0,.18);
}
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 860px) { .grid2 { grid-template-columns: 1fr; } }

.flags { margin: 10px 0 0; padding-left: 18px; color: var(--text-secondary); font-size: 13px; }
.flags li { margin: 2px 0; }
.foot { color: var(--text-muted); font-size: 12px; margin-top: 40px;
        border-top: 1px solid var(--border); padding-top: 16px; }
details summary { cursor: pointer; color: var(--text-secondary); font-size: 13px;
                  margin-bottom: 8px; }
"""

JS = """
document.querySelectorAll('.chart').forEach(function (chart) {
  var tip = chart.querySelector('.tip');
  var svg = chart.querySelector('svg');
  if (!tip || !svg) return;
  var pts = JSON.parse(chart.dataset.points || '[]');
  if (!pts.length) return;
  var x0 = +chart.dataset.x0, x1 = +chart.dataset.x1;
  var cursor = svg.querySelector('.cursor');

  function move(evt) {
    var box = svg.getBoundingClientRect();
    var vb = svg.viewBox.baseVal;
    var px = (evt.clientX - box.left) / box.width * vb.width;
    var frac = (px - x0) / Math.max(x1 - x0, 1e-9);
    var i = Math.round(frac * (pts.length - 1));
    i = Math.max(0, Math.min(pts.length - 1, i));
    var p = pts[i];
    tip.innerHTML = p.l;
    tip.style.opacity = 1;
    var left = (p.x / vb.width) * box.width;
    tip.style.left = Math.min(Math.max(left - tip.offsetWidth / 2, 0),
                              box.width - tip.offsetWidth) + 'px';
    tip.style.top = '4px';
    if (cursor) { cursor.setAttribute('x1', p.x); cursor.setAttribute('x2', p.x);
                  cursor.style.opacity = 1; }
  }
  svg.addEventListener('mousemove', move);
  svg.addEventListener('touchmove', function (e) {
    if (e.touches.length) move(e.touches[0]);
  }, { passive: true });
  svg.addEventListener('mouseleave', function () {
    tip.style.opacity = 0;
    if (cursor) cursor.style.opacity = 0;
  });
});
"""


def _esc(value) -> str:
    return html.escape(str(value if value is not None else ""))


def _fmt_pct(value, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value * 100:.{digits}f}%"


def _fmt_num(value, digits: int = 0) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:,.{digits}f}"


# ---------------------------------------------------------------------------
# SVG chart primitives
# ---------------------------------------------------------------------------
def _line_chart(
    dates: Sequence,
    series: Dict[str, Sequence[float]],
    height: int = 190,
    width: int = 640,
    shading: Optional[List[tuple]] = None,
    value_fmt=lambda v: f"{v:,.0f}",
    zero_line: bool = False,
) -> str:
    """Multi-series line chart with an optional shaded-interval layer."""
    names = [n for n, values in series.items() if len(values)]
    if not names or not len(dates):
        return '<p class="sub">no data</p>'

    pad_l, pad_r, pad_t, pad_b = 8, 8, 10, 20
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    flat = [v for name in names for v in series[name] if v is not None and np.isfinite(v)]
    if not flat:
        return '<p class="sub">no data</p>'
    lo, hi = float(min(flat)), float(max(flat))
    if zero_line:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    lo -= span * 0.06
    hi += span * 0.06

    n = len(dates)
    def sx(i: int) -> float:
        return pad_l + (i / max(n - 1, 1)) * plot_w

    def sy(v: float) -> float:
        return pad_t + (1.0 - (v - lo) / (hi - lo)) * plot_h

    parts: List[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'preserveAspectRatio="none" aria-label="time series chart">'
    ]

    # Shaded campaign intervals sit behind everything else.
    for start, end, colour, opacity in (shading or []):
        x1, x2 = sx(start), sx(end)
        parts.append(
            f'<rect x="{x1:.1f}" y="{pad_t}" width="{max(x2 - x1, 1):.1f}" '
            f'height="{plot_h}" fill="{colour}" opacity="{opacity}"/>'
        )

    # Recessive gridlines.
    for frac in (0.0, 0.5, 1.0):
        y = pad_t + frac * plot_h
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
        )
    if zero_line and lo < 0 < hi:
        y = sy(0.0)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            f'stroke="var(--text-muted)" stroke-width="1" stroke-dasharray="3 3"/>'
        )

    for idx, name in enumerate(names):
        values = series[name]
        colour = f"var(--s{(idx % 6) + 1})"
        points = []
        for i, v in enumerate(values):
            if v is None or not np.isfinite(v):
                continue
            points.append(f"{sx(i):.1f},{sy(float(v)):.1f}")
        if points:
            parts.append(
                f'<polyline fill="none" stroke="{colour}" stroke-width="2" '
                f'stroke-linejoin="round" stroke-linecap="round" '
                f'points="{" ".join(points)}"/>'
            )

    parts.append(
        f'<line class="cursor" x1="0" y1="{pad_t}" x2="0" y2="{pad_t + plot_h}" '
        f'stroke="var(--text-muted)" stroke-width="1" style="opacity:0"/>'
    )

    # Date ticks at the ends.
    first, last = _label_date(dates[0]), _label_date(dates[-1])
    parts.append(
        f'<text x="{pad_l}" y="{height - 5}" font-size="10" fill="var(--text-muted)">{first}</text>'
    )
    parts.append(
        f'<text x="{width - pad_r}" y="{height - 5}" font-size="10" '
        f'fill="var(--text-muted)" text-anchor="end">{last}</text>'
    )
    parts.append("</svg>")

    # Hover payload.
    points_json = []
    for i in range(n):
        bits = []
        for name in names:
            values = series[name]
            if i < len(values) and values[i] is not None and np.isfinite(values[i]):
                bits.append(f"{_esc(name)} {value_fmt(float(values[i]))}")
        points_json.append(
            '{"x":%.1f,"l":"%s"}' % (sx(i), _esc(_label_date(dates[i])) + " &middot; " + " &middot; ".join(bits))
        )

    legend = ""
    if len(names) > 1:
        chips = "".join(
            f'<span><i style="background:var(--s{(i % 6) + 1})"></i>{_esc(name)}</span>'
            for i, name in enumerate(names)
        )
        legend = f'<div class="legend">{chips}</div>'

    return (
        legend
        + f'<div class="chart" data-x0="{pad_l}" data-x1="{width - pad_r}" '
        + "data-points='[" + ",".join(points_json) + "]'>"
        + '<div class="tip"></div>' + "".join(parts) + "</div>"
    )


def _label_date(value) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def _component_bars(components: Dict[str, float]) -> str:
    if not components:
        return ""
    rows = []
    for name, value in sorted(components.items(), key=lambda kv: -kv[1]):
        pct = max(0.0, min(1.0, float(value)))
        colour = "var(--good)" if pct >= 0.66 else "var(--accent)" if pct >= 0.4 else "var(--text-muted)"
        label = name.replace("_", " ")
        rows.append(
            f'<div class="name">{_esc(label)}</div>'
            f'<div class="track"><i style="width:{pct * 100:.0f}%;background:{colour}"></i></div>'
            f'<div class="num mono">{pct:.2f}</div>'
        )
    return f'<div class="components">{"".join(rows)}</div>'


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _provenance_banner(is_real: bool, provenance: str, mode: str) -> str:
    if not is_real:
        return (
            '<div class="banner danger"><strong>Broker flow is SIMULATED</strong>'
            f'Provenance: <span class="mono">{_esc(provenance)}</span>. '
            "The price and volume series are genuine exchange data, but every broker-level "
            "number on this page was generated by the simulator in "
            '<span class="mono">data/synthetic.py</span>, which assumes institutions buy '
            "weakness and retail chases momentum. Any conclusion about broker behaviour drawn "
            "from this page is circular. Connect a real broker-summary source before trading."
            "</div>"
        )
    return (
        '<div class="banner"><strong>Data provenance</strong>'
        f'Broker flow: <span class="mono">{_esc(provenance)}</span> &middot; '
        f'scoring mode: <span class="mono">{_esc(mode)}</span>. '
        "Broker summary reflects flow through each exchange member, which aggregates all "
        "clients of that member - it is a strong signal of institutional intent, not a "
        "statement of it.</div>"
    )


def _screener_table(results: pd.DataFrame, prefix: str = "IDX") -> str:
    if results is None or results.empty:
        return '<p class="sub">No results.</p>'

    head = (
        "<tr><th>#</th><th>Ticker</th><th>Score</th><th>Level</th><th>Phase</th>"
        "<th>Close</th><th>Lead broker</th><th>Range pos</th><th>ATR%</th><th>Mode</th></tr>"
    )
    rows = []
    for _, r in results.iterrows():
        score = float(r.get("score", 0))
        level = str(r.get("level", "NONE"))
        ticker = str(r.get("ticker", ""))
        url = f"https://www.tradingview.com/chart/?symbol={prefix}%3A{ticker}"
        rows.append(
            "<tr>"
            f'<td class="mono">{int(r.get("rank", 0))}</td>'
            f'<td><a href="{url}" target="_blank" rel="noopener">{_esc(ticker)}</a></td>'
            f'<td class="mono">{score:.1f}'
            f'<span class="scorebar"><i style="width:{min(score, 100):.0f}%"></i></span></td>'
            f'<td><span class="pill {_esc(level)}">{_esc(level)}</span></td>'
            f'<td class="mono">{_esc(r.get("wyckoff_phase", "-"))}</td>'
            f'<td class="mono">{_fmt_num(r.get("close"))}</td>'
            f'<td class="mono">{_esc(r.get("lead_broker", "") or "-")}</td>'
            f'<td class="mono">{_fmt_pct(r.get("range_pos"), 0)}</td>'
            f'<td class="mono">{_fmt_pct(r.get("atr_pct"), 1)}</td>'
            f'<td class="mono">{_esc(r.get("data_mode", ""))}</td>'
            "</tr>"
        )
    return f'<div class="scroll"><table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def _playbook_table(book: pd.DataFrame) -> str:
    if book is None or book.empty:
        return '<p class="sub">Not enough completed campaigns to profile any broker.</p>'

    head = (
        "<tr><th>Broker</th><th>Name</th><th>Tier</th><th>Campaigns</th><th>Acc days</th>"
        "<th>Entry pct</th><th>Markup</th><th>Realised</th><th>Capture</th><th>Win rate</th>"
        "<th>Hold days</th><th>Style</th></tr>"
    )
    rows = []
    for _, r in book.iterrows():
        rows.append(
            "<tr>"
            f'<td class="mono">{_esc(r["broker"])}</td>'
            f'<td>{_esc(r["name"])}</td>'
            f'<td>{_esc(r["tier"])}</td>'
            f'<td class="mono">{int(r["campaigns"])}</td>'
            f'<td class="mono">{_fmt_num(r.get("median_acc_days"))}</td>'
            f'<td class="mono">{_fmt_pct(r.get("median_entry_percentile"), 0)}</td>'
            f'<td class="mono">{_fmt_pct(r.get("median_markup_pct"))}</td>'
            f'<td class="mono">{_fmt_pct(r.get("median_realized_return"))}</td>'
            f'<td class="mono">{_fmt_pct(r.get("median_exit_capture"), 0)}</td>'
            f'<td class="mono">{_fmt_pct(r.get("win_rate"), 0)}</td>'
            f'<td class="mono">{_fmt_num(r.get("median_holding_days"))}</td>'
            f"<td>{_esc(r.get('style', ''))}</td>"
            "</tr>"
        )
    return (
        '<div class="scroll"><table><thead>' + head + "</thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )


def _ticker_card(analysis, plan, cfg, prefix: str = "IDX") -> str:
    bars = analysis.bars
    signal = analysis.signal
    ticker = analysis.ticker

    tail = bars.tail(320)
    dates = list(tail["date"])
    closes = [float(v) for v in tail["close"]]

    # Shade the accumulation legs of campaigns that overlap the visible window.
    shading = []
    if not analysis.campaigns.empty:
        index_of = {pd.Timestamp(d): i for i, d in enumerate(dates)}
        for _, camp in analysis.campaigns.iterrows():
            start = index_of.get(pd.Timestamp(camp["acc_start"]))
            end = index_of.get(pd.Timestamp(camp["acc_end"]))
            if start is None and end is None:
                continue
            start = 0 if start is None else start
            end = len(dates) - 1 if end is None else end
            if end > start:
                shading.append((start, end, "var(--s3)", 0.14))

    price_chart = _line_chart(
        dates, {"Close": closes}, shading=shading,
        value_fmt=lambda v: f"{v:,.0f}",
    )

    # Top institutional inventories, capped at four series for legibility.
    inventory_chart = '<p class="sub">No broker data.</p>'
    if not analysis.ledger.empty:
        positions = analysis.positions.copy()
        positions["tier"] = positions["broker"].map(lambda c: cfg.brokers.get(c).tier)
        institutional = positions[positions["tier"].isin(["bulge", "foreign", "local_inst"])]
        top = institutional.reindex(
            institutional["inventory_lot"].abs().sort_values(ascending=False).index
        ).head(4)

        wide = analysis.ledger.pivot_table(
            index="date", columns="broker", values="inventory_lot", aggfunc="last"
        ).ffill()
        wide = wide.reindex(pd.DatetimeIndex(dates)).ffill()
        series = {}
        for broker in top["broker"]:
            if broker in wide.columns:
                series[broker] = [float(v) if np.isfinite(v) else None for v in wide[broker]]
        if series:
            inventory_chart = _line_chart(
                dates, series, value_fmt=lambda v: f"{v:,.0f} lots", zero_line=True
            )

    flags = "".join(f"<li>{_esc(f)}</li>" for f in signal.flags)
    flags_html = f'<ul class="flags">{flags}</ul>' if flags else ""

    plan_html = ""
    if plan is not None:
        verdict_class = "TAKE" if plan.verdict.startswith("TAKE") else (
            "WATCH" if plan.verdict == "WATCH" else "SKIP")
        targets = " / ".join(f"{t:,.0f}" for t in plan.targets) or "-"
        plan_html = (
            '<div class="grid2" style="margin-top:14px">'
            "<div>"
            f'<h3>Plan <span class="pill {verdict_class}">{_esc(plan.verdict)}</span></h3>'
            '<table style="font-size:12px">'
            f'<tr><td>Entry zone</td><td class="mono">{_fmt_num(plan.entry_low)} - '
            f'{_fmt_num(plan.entry_high)}</td></tr>'
            f'<tr><td>Stop</td><td class="mono">{_fmt_num(plan.stop)} '
            f'({_fmt_pct(plan.risk_pct, 1)})</td></tr>'
            f'<tr><td>Targets</td><td class="mono">{_esc(targets)}</td></tr>'
            f'<tr><td>Reward:risk</td><td class="mono">{_fmt_num(plan.reward_risk, 2)}</td></tr>'
            f'<tr><td>Size</td><td class="mono">{plan.lots:,} lots '
            f'({_esc(format_idr(plan.notional))})</td></tr>'
            f'<tr><td>Anchor</td><td class="mono">{_esc(plan.anchor_broker or "-")} @ '
            f'{_fmt_num(plan.anchor_cost)}</td></tr>'
            f'<tr><td>Time stop</td><td class="mono">{plan.time_stop_days} days</td></tr>'
            "</table></div>"
            f"<div><h3>Score components</h3>{_component_bars(signal.components)}</div>"
            "</div>"
        )
    else:
        plan_html = f"<div style='margin-top:14px'><h3>Score components</h3>{_component_bars(signal.components)}</div>"

    url = f"https://www.tradingview.com/chart/?symbol={prefix}%3A{ticker}"
    return (
        '<div class="card">'
        '<div class="card-head">'
        f'<span class="ticker">{_esc(ticker)}</span>'
        f'<span class="pill {_esc(signal.level)}">{_esc(signal.level)} {signal.score:.1f}</span>'
        f'<span class="sub" style="margin:0">Wyckoff {_esc(signal.wyckoff_state.phase if signal.wyckoff_state else "-")} '
        f'&middot; close {_fmt_num(analysis.last_close)} &middot; {_label_date(analysis.last_date)}</span>'
        f'<a href="{url}" target="_blank" rel="noopener">Open in TradingView &rarr;</a>'
        "</div>"
        '<div class="grid2">'
        f"<div><h3>Price &amp; accumulation legs</h3>{price_chart}</div>"
        f"<div><h3>Institutional inventory (lots)</h3>{inventory_chart}</div>"
        "</div>"
        f"{flags_html}{plan_html}"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def render(
    results: pd.DataFrame,
    analyses: Sequence,
    plans: Optional[Dict[str, object]],
    playbook: pd.DataFrame,
    cfg,
    provenance: str = "",
    title: str = "IDX Accumulation Dashboard",
    detail_limit: int = 8,
) -> str:
    prefix = str(cfg.get("tradingview.exchange_prefix", "IDX"))
    is_real = bool(results["data_is_real"].any()) if (
        results is not None and not results.empty and "data_is_real" in results
    ) else False
    mode = str(results["data_mode"].mode().iloc[0]) if (
        results is not None and not results.empty and "data_mode" in results
    ) else "unknown"

    generated = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    strong = int((results["level"] == "STRONG").sum()) if not results.empty else 0
    signals = int((results["level"] == "SIGNAL").sum()) if not results.empty else 0
    watch = int((results["level"] == "WATCH").sum()) if not results.empty else 0

    tiles = (
        '<div class="tiles">'
        f'<div class="tile"><div class="label">Screened</div>'
        f'<div class="value mono">{len(results) if results is not None else 0}</div>'
        f'<div class="note">tickers</div></div>'
        f'<div class="tile"><div class="label">Strong</div>'
        f'<div class="value mono" style="color:var(--good)">{strong}</div>'
        f'<div class="note">score &ge; 78</div></div>'
        f'<div class="tile"><div class="label">Signal</div>'
        f'<div class="value mono" style="color:var(--accent)">{signals}</div>'
        f'<div class="note">score &ge; 65</div></div>'
        f'<div class="tile"><div class="label">Watch</div>'
        f'<div class="value mono" style="color:var(--warning)">{watch}</div>'
        f'<div class="note">score &ge; 50</div></div>'
        f'<div class="tile"><div class="label">Mode</div>'
        f'<div class="value mono" style="font-size:16px">{_esc(mode)}</div>'
        f'<div class="note">{"real broker data" if is_real else "simulated broker data"}</div></div>'
        "</div>"
    )

    cards = "".join(
        _ticker_card(a, (plans or {}).get(a.ticker), cfg, prefix)
        for a in list(analyses)[:detail_limit]
    )

    body = (
        '<div class="wrap">'
        f"<h1>{_esc(title)}</h1>"
        f'<p class="sub">Generated {generated} &middot; universe of '
        f'{len(results) if results is not None else 0} tickers &middot; '
        f"IDX broker-flow accumulation engine</p>"
        f"{_provenance_banner(is_real, provenance, mode)}"
        f"{tiles}"
        "<h2>Screener</h2>"
        f"{_screener_table(results, prefix)}"
        "<h2>Top candidates</h2>"
        f"{cards}"
        "<h2>Broker playbook</h2>"
        '<p class="sub">Behavioural profile of each exchange member, pooled across the '
        "universe. Entry pct = where inside the trailing range their buy VWAP sat. "
        "Capture = share of the available move actually realised.</p>"
        f"{_playbook_table(playbook)}"
        '<div class="foot">'
        "Educational tooling, not investment advice. Broker summary shows flow through an "
        "exchange member, aggregating all of that member's clients; inventory is measured "
        "relative to the start of the data window, not absolute holdings. "
        "Backtested edges are gross of costs unless stated and carry no survivorship "
        "adjustment."
        "</div>"
        "</div>"
    )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_esc(title)}</title>"
        f"<style>{CSS}</style></head><body>{body}"
        f"<script>{JS}</script></body></html>"
    )


def write(path: str, html_text: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html_text)
    return path
