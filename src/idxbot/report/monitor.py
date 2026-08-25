"""Position monitor — where each exit rule sits RIGHT NOW, as a price.

WHY A SEPARATE MODULE FROM THE BACKTEST
-----------------------------------------
`spine/exits.py` answers "what would this rule have earned". That is the right
question for research and the wrong one for a Tuesday evening. What a holder
needs is the level: **at what price does my stop fire tomorrow, and how far is
that from here.** Same rule definitions, same parameters, evaluated forward
instead of backward, so there is exactly one implementation of each rule and
the monitor cannot drift from the study that validated it.

THE NEWS COLUMN IS NOT A SIGNAL AND MUST NOT BECOME ONE
---------------------------------------------------------
There is no point-in-time news archive, so no news-conditioned rule can ever be
backtested here — `tests/test_news.py` fails the build if `spine/` or
`features/` so much as imports the news module. This file lives in `report/`
precisely because it is a READING SURFACE, not a signal path: it prints the
standing event tags (suspension, UMA, rights issue, delisting) beside the
levels so a holder sees them, and no number in this repo is conditioned on
them. A suspension is a fact about whether you can trade at all, which is worth
knowing whether or not it carries measurable alpha.

WHAT THE LEVELS MEAN
---------------------
Each rule reduces to "exit if tomorrow's close is at or below L" — except the
stochastic ones, which are conditions on an oscillator rather than on price and
are reported as state. Levels are on the ADJUSTED basis and converted back to
today's quoted rupiah so they can be typed into a broker screen.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..spine import exits as X
from . import brief as B

#: How far back to look for standing corporate-event headlines, in days.
NEWS_DAYS = 270


def position_frame(P: pd.DataFrame, I: pd.DataFrame,
                   positions: Sequence[Dict[str, object]],
                   day: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Current state of each held name: entry, peak, drawdown, indicators.

    ``positions`` is a sequence of ``{"ticker", "entry_date"}`` and optionally
    ``{"entry_price"}``. When the entry price is absent the adjusted close of
    the first session ON OR AFTER the entry date is used, which is the one-bar
    convention the backtest fills at.
    """
    if day is None:
        day = B.resolve_asof(P)
    rows = []
    for pos in positions:
        t = str(pos["ticker"]).upper()
        e = pd.Timestamp(pos["entry_date"])
        g = P[(P["ticker"] == t) & (P["date"] <= day)].sort_values("date")
        f = I[(I["ticker"] == t) & (I["date"] <= day)].sort_values("date")
        if g.empty or f.empty:
            rows.append({"ticker": t, "entry_date": e, "status": "no data"})
            continue
        held = g[g["date"] >= e]
        if held.empty:
            rows.append({"ticker": t, "entry_date": e, "status": "not yet held"})
            continue
        adj = held["adj_close"].astype(float).to_numpy()
        entry_adj = float(adj[0])
        if not np.isfinite(entry_adj) or entry_adj <= 0:
            rows.append({"ticker": t, "entry_date": e, "status": "bad entry"})
            continue
        fh = f[f["date"] >= held["date"].iloc[0]]
        hi = fh["adj_high"].astype(float).to_numpy()
        peak_adj = float(np.nanmax(np.where(np.isfinite(hi), hi, adj[:len(hi)])))
        last = f.iloc[-1]
        px = float(held["close"].iloc[-1])
        # adjusted -> quoted rupiah, so a level can be typed into a screen
        q = px / float(adj[-1]) if adj[-1] else np.nan
        rows.append({
            "ticker": t, "entry_date": held["date"].iloc[0], "asof": day,
            "sessions": int(len(held)), "price": px,
            "entry_price": float(pos.get("entry_price")
                                 or held["close"].iloc[0]),
            "gain": float(adj[-1] / entry_adj - 1.0),
            "peak_gain": float(peak_adj / entry_adj - 1.0),
            "give_back": float(adj[-1] / peak_adj - 1.0) if peak_adj else np.nan,
            "adj": float(adj[-1]), "peak_adj": peak_adj, "q": q,
            "ema20": float(last.get("ema20", np.nan)),
            "ema50": float(last.get("ema50", np.nan)),
            "atr22": float(last.get("atr22", np.nan)),
            "stoch_k": float(last.get("stoch_k", np.nan)),
            "stoch_d": float(last.get("stoch_d", np.nan)),
            "tvz20": float(last.get("tvz20", np.nan)),
            "status": "held"})
    return pd.DataFrame(rows)


def levels(row: pd.Series, arm: float = 0.50, trail: float = 0.15,
           chand_k: float = 3.0) -> pd.DataFrame:
    """The price at which each rule fires on the next close, for one position.

    ``armed`` is False when the position has never been up ``arm`` — and a rule
    that is not armed CANNOT FIRE, which is the whole reason H17 measured
    P(-50%) as unchanged. That is shown as "not armed" rather than as a level,
    because printing a level for a rule that cannot trigger is worse than
    printing nothing.
    """
    if row.get("status") != "held":
        return pd.DataFrame()
    q = row["q"] if np.isfinite(row.get("q", np.nan)) else 1.0
    armed = row["peak_gain"] >= arm
    out: List[Dict[str, object]] = []

    def add(name, lvl_adj, active=True, note=""):
        lvl = lvl_adj * q if (active and np.isfinite(lvl_adj)) else np.nan
        out.append({"rule": name, "level": lvl,
                    "distance": (lvl / row["price"] - 1.0)
                    if np.isfinite(lvl) and row["price"] else np.nan,
                    "active": bool(active), "note": note})

    add(f"trail {trail:.0%} armed +{arm:.0%}", row["peak_adj"] * (1 - trail),
        armed, "" if armed else f"not armed (peak +{row['peak_gain']:.0%})")
    add(f"chandelier {chand_k:.0f}x ATR armed +{arm:.0%}",
        row["peak_adj"] - chand_k * row["atr22"], armed,
        "" if armed else f"not armed (peak +{row['peak_gain']:.0%})")
    # NOT "(unarmed)": that string contains "armed", which misreads at a
    # glance and broke a filter written against it. Say what it does.
    add(f"chandelier {chand_k:.0f}x ATR (trails from entry)",
        row["peak_adj"] - chand_k * row["atr22"], True)
    add("ema20 break", row["ema20"], True)
    add("ema50 break", row["ema50"], True)
    add("hard stop 25%", row["adj"] / (1 + row["gain"]) * 0.75, True,
        "measured from entry")
    D = pd.DataFrame(out)
    return D.sort_values("level", ascending=False, na_position="last")


def oscillator_state(row: pd.Series) -> str:
    """The stochastic and volume rules, which are conditions not price levels."""
    if row.get("status") != "held":
        return ""
    k, d, z = row.get("stoch_k"), row.get("stoch_d"), row.get("tvz20")
    bits = []
    if np.isfinite(k) and np.isfinite(d):
        bits.append(f"%K {k:.0f} / %D {d:.0f}"
                    + (" — rolled over" if k < d else " — rising")
                    + (" from overbought" if k >= 80 or d >= 80 else ""))
    else:
        bits.append("stochastic undefined (flat or thin range)")
    if np.isfinite(z):
        bits.append(f"turnover z {z:+.1f}"
                    + (" — climax territory" if z >= 2 else ""))
    return "; ".join(bits)


def replay(P: pd.DataFrame, I: pd.DataFrame, row: pd.Series,
           rules: Dict[str, object]) -> pd.DataFrame:
    """When would each rule ALREADY have exited this position, and at what?

    A level sitting far above today's price does not mean "sell now" — it means
    the rule told you to sell weeks ago and you are looking at the aftermath.
    Printing only the level invites reading a stale trigger as a fresh one, so
    the fire date and the price on it are computed by running the SAME rule
    functions over the realised path.
    """
    t, e = row["ticker"], pd.Timestamp(row["entry_date"])
    g = P[(P["ticker"] == t) & (P["date"] >= e)].sort_values("date")
    f = I[(I["ticker"] == t) & (I["date"] >= e)].sort_values("date")
    if len(g) < 2 or len(f) < 2:
        return pd.DataFrame()
    adj = g["adj_close"].astype(float).to_numpy()
    path = adj[1:X.HORIZON + 1] / adj[0]
    F = {"close": f["close"].to_numpy(float)[1:X.HORIZON + 1],
         "high": f["adj_high"].to_numpy(float)[1:X.HORIZON + 1]}
    for c in ("ema10", "ema20", "ema30", "ema50", "atr22", "stoch_k",
              "stoch_d", "tvz20"):
        if c in f:
            F[c] = f[c].to_numpy(float)[1:X.HORIZON + 1]
    dates = pd.DatetimeIndex(g["date"])[1:X.HORIZON + 1]
    px = g["close"].astype(float).to_numpy()[1:X.HORIZON + 1]
    out = []
    for name, fn in rules.items():
        accepts, requires = X.rule_arity(fn)
        if requires and not F:
            continue
        r, held = fn(path, F) if accepts else fn(path)
        i = min(held, len(dates)) - 1
        fired = held < min(X.HORIZON, len(path))
        out.append({"rule": name, "fired": bool(fired),
                    "date": dates[i], "price": float(px[i]),
                    "gross": float(r), "sessions": int(held)})
    return pd.DataFrame(out).sort_values("sessions")


def nearest_trigger(L: pd.DataFrame) -> Optional[pd.Series]:
    """The active rule whose level is closest below the current price.

    This is the number that actually matters to a holder: the first stop that
    would fire. Rules already breached show a non-negative distance and are
    reported first by the caller.
    """
    A = L[L["active"] & np.isfinite(L["distance"])]
    if A.empty:
        return None
    breached = A[A["distance"] >= 0]
    return (breached.sort_values("distance", ascending=False).iloc[0]
            if not breached.empty
            else A.sort_values("distance", ascending=False).iloc[0])


def event_tags(tickers: Sequence[str], days: int = NEWS_DAYS
               ) -> Dict[str, List[str]]:
    """Standing corporate-event headlines per ticker — READ ONLY, never a rule.

    Import is local so that the static check in `tests/test_news.py` keeps
    seeing a clean separation, and so a network failure here degrades the
    monitor to "no tags" instead of taking the whole report down.
    """
    try:
        from ..data import news as N
        standing = set(N.STANDING_TAGS)
        df = N.ticker_news(list(tickers), per=3, days=30, standing_days=days)
    except Exception:                                            # noqa: BLE001
        return {}
    if df is None or df.empty or "ticker" not in df:
        return {}
    out: Dict[str, List[str]] = {}
    for t, g in df.groupby("ticker"):
        seen = set()
        for row in g.get("tags", []):
            if isinstance(row, (list, tuple, set, np.ndarray)):
                seen.update(row)
            elif row:
                seen.add(row)
        keep = sorted(seen & standing)
        if keep:
            out[str(t)] = keep
    return out
