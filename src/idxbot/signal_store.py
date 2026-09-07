"""§38 — every signal this repo emits, and what it later did.

WHY THIS IS THE MOST VALUABLE THING IN THE ROADMAP DESPITE BEING THE SIMPLEST.
The 24-month holdout was spent at H16. Every one of the 347 registered trials is
therefore IN-SAMPLE, and no amount of further engineering changes that — the
data has already been looked at. There is exactly one mechanism that produces
genuinely out-of-sample evidence from here: record what the system says BEFORE
the outcome exists, then score it. It costs nothing to start and it compounds
from the day it starts, which is why it is stage 1 and not stage 9.

THE STORE IS APPEND-ONLY AND THAT IS THE ENTIRE DESIGN.
A record that a later run can rewrite is not evidence, it is a draft. So:

  * `emit()` refuses to overwrite an existing signal_id and says so.
  * `signal_id` is a deterministic hash of (asof, ticker, rule, rule_version),
    so re-running the same scan on the same day is idempotent rather than
    duplicating — but changing the RULE changes the id, which is correct,
    because that is a different prediction.
  * `score()` only ever reads the log and writes to a SEPARATE outcomes file.
    Nothing scores a signal in the same process that emitted it.

WHAT MUST BE RECORDED, AND WHY EACH FIELD IS NOT OPTIONAL (§51):
  asof          the bar the decision was made ON. Not "today" — a scan run at
                19:00 acts on the 15:50 close, and conflating them is how a
                one-session look-ahead enters a live track record.
  code_version  the git SHA. A38 records this repo changing a cost model and a
                harness mid-stream; a signal whose generating code is unknown
                cannot be re-derived, and §51 requires that it can be.
  rule_version  the parameters that produced it. STOP=0.20 and TP=1.00 are
                H56/H56b's answers today and may not be tomorrow.
  features      what was knowable at `asof`. Without it an outcome can be
                counted but never attributed.
  horizon_days  fixed AT EMISSION. A20: the horizon is the parameter twelve
                studies inherited without choosing, and choosing it after
                seeing the outcome is the purest form of the error.

WHAT IS DELIBERATELY NOT HERE. No accuracy, no win rate, no score. This module
records and scores; it makes no claim. A28's rule stands — the machinery emits
probabilities, a hit rate is undefined until someone picks a threshold, and
whoever picks the threshold decides the answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

STORE_DIR = os.path.join("data", "signals")
EMITTED = os.path.join(STORE_DIR, "emitted.csv.gz")
OUTCOMES = os.path.join(STORE_DIR, "outcomes.csv.gz")

EMIT_COLUMNS = [
    "signal_id", "asof", "emitted_at", "ticker", "rule", "rule_version",
    "code_version", "direction", "entry", "sl", "tp", "tp_frac",
    "horizon_days", "features",
]
OUTCOME_COLUMNS = [
    "signal_id", "asof", "ticker", "rule", "horizon_days", "scored_at",
    "bars_seen", "settled", "exit_reason", "exit_date", "exit_px",
    "ret", "ret_net", "mfe", "mae", "hit_tp", "hit_sl", "last_px",
    "adj_factor",
]


def code_version() -> str:
    """The git SHA, or `unknown`. Never a lie: an unversioned signal says so."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def signal_id(asof, ticker: str, rule: str, rule_version: str) -> str:
    """Deterministic, so the same scan on the same day is idempotent.

    Changing the RULE or its VERSION changes the id, which is intended: a
    different rule on the same bar is a different prediction and must not
    silently overwrite the first one's record.
    """
    key = f"{pd.Timestamp(asof).date()}|{ticker}|{rule}|{rule_version}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _read(path: str, cols: Sequence[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=list(cols))
    try:
        d = pd.read_csv(path, parse_dates=["asof"])
    except Exception:
        return pd.DataFrame(columns=list(cols))
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan
    return d


def load_emitted() -> pd.DataFrame:
    return _read(EMITTED, EMIT_COLUMNS)


def load_outcomes() -> pd.DataFrame:
    return _read(OUTCOMES, OUTCOME_COLUMNS)


def emit(rows: Sequence[Dict], rule: str, rule_version: str,
         asof, horizon_days: int, emitted_at=None) -> Dict:
    """Append signals. Existing ids are SKIPPED, never overwritten.

    Returns a summary rather than raising on duplicates, because the daily job
    re-running is normal and must be a no-op rather than a failure.
    """
    if not rows:
        return {"written": 0, "skipped": 0, "reason": "no rows"}
    os.makedirs(STORE_DIR, exist_ok=True)
    old = load_emitted()
    have = set(old["signal_id"].astype(str)) if len(old) else set()
    cv = code_version()
    stamp = pd.Timestamp(emitted_at) if emitted_at is not None \
        else pd.Timestamp.now("UTC")

    out, skipped = [], 0
    for r in rows:
        tk = str(r["ticker"])
        sid = signal_id(asof, tk, rule, rule_version)
        if sid in have:
            skipped += 1
            continue
        have.add(sid)
        feats = {k: v for k, v in r.items()
                 if k not in ("ticker", "entry", "sl", "tp", "tp_frac",
                              "direction")}
        out.append({
            "signal_id": sid,
            "asof": pd.Timestamp(asof).normalize(),
            "emitted_at": stamp,
            "ticker": tk,
            "rule": rule,
            "rule_version": rule_version,
            "code_version": cv,
            "direction": r.get("direction", "long"),
            "entry": float(r["entry"]),
            "sl": float(r["sl"]) if r.get("sl") is not None else np.nan,
            "tp": float(r["tp"]) if r.get("tp") is not None else np.nan,
            #  1.0 = the target is a full exit. Below 1.0 it is a SCALE-OUT and
            #  the remainder runs on, which is what `rules.py` actually ships.
            "tp_frac": float(r.get("tp_frac", 1.0)),
            "horizon_days": int(horizon_days),
            #  Sorted keys so the same features serialise identically and a
            #  diff of two days' logs is readable.
            "features": json.dumps(feats, sort_keys=True, default=str),
        })
    if not out:
        return {"written": 0, "skipped": skipped, "reason": "all present"}
    new = pd.DataFrame(out)[EMIT_COLUMNS]
    allrows = pd.concat([old, new], ignore_index=True) if len(old) else new
    tmp = EMITTED + ".tmp"
    allrows.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, EMITTED)
    return {"written": len(new), "skipped": skipped,
            "total": len(allrows), "code_version": cv}


def _tp_frac(s) -> float:
    """The scale-out fraction for one recorded signal.

    RECOVERED FROM `rule_version` WHEN THE COLUMN IS ABSENT, RATHER THAN
    ASSUMED. `tp_frac` was added after signals had already been logged, and
    defaulting those rows to 1.0 would score them as full exits — a different
    rule from the one that produced them. But the information was never
    actually lost: `rule_version` carries `...tp1.0x0.5`, which IS the
    fraction. Reading it back is a recovery, not a rewrite of an append-only
    store, and it is what lets the earliest rows stay in the record instead of
    being silently reinterpreted.
    """
    v = s.get("tp_frac", np.nan)
    try:
        f = float(v)
    except (TypeError, ValueError):
        f = float("nan")
    if np.isfinite(f) and 0.0 < f <= 1.0:
        return f
    m = re.search(r"tp[0-9.]+x([0-9.]+)", str(s.get("rule_version", "")))
    if m:
        try:
            f = float(m.group(1))
            if 0.0 < f <= 1.0:
                return f
        except ValueError:
            pass
    return 1.0


def score(panel: pd.DataFrame, cost: float = 0.0056,
          scored_at=None) -> pd.DataFrame:
    """Walk every emitted signal forward and record what it did.

    THE OUTCOME IS COMPUTED FROM BARS STRICTLY AFTER `asof`. A signal decided on
    the 15:50 close cannot be filled before the next session, and scoring it
    from its own bar is the one-session look-ahead that would make a live track
    record look better than the thing it is tracking.

    `settled` is False while the horizon is still open. An unsettled signal is
    reported, never dropped and never counted as a win or a loss — A28's rule
    that a hit rate is undefined until someone picks a threshold applies twice
    as hard when some of the sample has not finished happening.
    """
    em = load_emitted()
    if em.empty:
        return pd.DataFrame(columns=OUTCOME_COLUMNS)
    stamp = pd.Timestamp(scored_at) if scored_at is not None \
        else pd.Timestamp.now("UTC")
    P = panel[["ticker", "date", "adj_close", "close"]].copy()
    P = P.sort_values(["ticker", "date"])
    by = {tk: g for tk, g in P.groupby("ticker", sort=False)}
    #  THE ADJUSTMENT FACTOR AT THE DECISION BAR IS WHAT MAKES THIS SCOREABLE
    #  AT ALL, AND ITS ABSENCE WAS THE WORST BUG IN THIS REPO'S FORWARD RECORD.
    #  `entry`, `sl` and `tp` are RAW prices — they have to be, they are the
    #  numbers you give a broker. The forward path is `adj_close`, which is
    #  BACK-adjusted: it anchors to the newest bar, so every dividend or split
    #  after emission divides the whole history before it, including the
    #  emission bar. Comparing a forward `adj_close` to a raw recorded `entry`
    #  therefore drifts further wrong with every corporate action.
    #
    #  Measured on BBCA: a signal that actually returned **+1.09%** scored as
    #  **-13.71%** — a 14.8-point error, from accumulated dividends alone. A
    #  split would print a fake near-total loss. And on the day of emission the
    #  two bases AGREE, so nothing looks wrong until months later. 59% of panel
    #  bars already differ by more than 0.1%.
    #
    #  The fix is to carry the recorded raw levels onto the panel's CURRENT
    #  adjusted basis with the factor at `asof`. Both sides then move together
    #  whenever the panel is rebuilt, so the recorded return is stable.

    rows: List[Dict] = []
    for _, s in em.iterrows():
        g = by.get(s["ticker"])
        rec = {c: np.nan for c in OUTCOME_COLUMNS}
        rec.update({"signal_id": s["signal_id"], "asof": s["asof"],
                    "ticker": s["ticker"], "rule": s["rule"],
                    "horizon_days": s["horizon_days"], "scored_at": stamp,
                    "bars_seen": 0, "settled": False,
                    "exit_reason": "no data"})
        if g is None:
            rows.append(rec)
            continue
        #  STRICTLY after the decision bar.
        fwd = g[g["date"] > s["asof"]]
        h = int(s["horizon_days"])
        fwd = fwd.head(h)
        if fwd.empty:
            rec["exit_reason"] = "no bars yet"
            rows.append(rec)
            continue
        px = fwd["adj_close"].to_numpy(float)
        dts = fwd["date"].to_numpy()
        entry = float(s["entry"])
        if not np.isfinite(entry) or entry <= 0:
            rec["exit_reason"] = "bad entry"
            rows.append(rec)
            continue
        #  factor = adj_close / close ON THE DECISION BAR, from today's panel.
        at = g[g["date"] == s["asof"]]
        if at.empty:
            #  No bar on the decision date (a halt, or a name the panel does
            #  not carry that day). Fall back to the nearest EARLIER bar rather
            #  than to 1.0: assuming no adjustment is the bug, not the default.
            at = g[g["date"] <= s["asof"]].tail(1)
        if at.empty or not np.isfinite(float(at["close"].iloc[0])) \
                or float(at["close"].iloc[0]) <= 0:
            rec["exit_reason"] = "no basis bar"
            rows.append(rec)
            continue
        f = float(at["adj_close"].iloc[0]) / float(at["close"].iloc[0])
        if not np.isfinite(f) or f <= 0:
            rec["exit_reason"] = "bad adjustment factor"
            rows.append(rec)
            continue
        rec["adj_factor"] = f
        entry_adj = entry * f
        rel = px / entry_adj - 1.0
        sl = float(s["sl"]) * f if np.isfinite(s["sl"]) else None
        tp = float(s["tp"]) * f if np.isfinite(s["tp"]) else None

        #  Exit on the CLOSE that breaches, not at the level (A27): a bar
        #  breaching -20% often closes lower, and on IDX it can gap to
        #  auto-rejection where nothing trades at any price.
        i_sl = int(np.argmax(px <= sl)) if sl and (px <= sl).any() else None
        i_tp = int(np.argmax(px >= tp)) if tp and (px >= tp).any() else None
        cands = [(i, r) for i, r in ((i_sl, "sl"), (i_tp, "tp"))
                 if i is not None]
        if cands:
            i, reason = min(cands)
        else:
            i, reason = len(px) - 1, "horizon"
        settled = reason in ("sl", "tp") or len(fwd) >= h

        #  A SCALE-OUT IS NOT AN EXIT, AND SCORING IT AS ONE MEASURES A RULE
        #  NOBODY IS TRADING. `rules.py` ships TP_FRAC = 0.5: sell half at the
        #  target and let the rest run to the stop or the horizon. Treating the
        #  target as a full exit caps the winner in the record while the live
        #  book does not, which is the exact asymmetry H56b priced at 8.77%
        #  CAGR against 10.24%. `tp_frac` defaults to 1.0 so a signal emitted
        #  without one still scores as the full exit it was.
        frac = _tp_frac(s)
        if reason == "tp" and frac < 1.0:
            #  half realised at the target, the remainder carried on to
            #  whichever of the stop or the horizon comes first
            tail = rel[i + 1:]
            if sl is not None and len(tail) and (px[i + 1:] <= sl).any():
                j = i + 1 + int(np.argmax(px[i + 1:] <= sl))
                reason, k = "tp_then_sl", j
            else:
                reason, k = "tp_then_horizon", len(px) - 1
            gross = frac * rel[i] + (1.0 - frac) * rel[k]
            settled = (reason == "tp_then_sl") or len(fwd) >= h
            i = k
        else:
            gross = rel[i]

        rec.update({
            "bars_seen": int(len(fwd)), "settled": bool(settled),
            "exit_reason": reason,
            "exit_date": pd.Timestamp(dts[i]),
            "exit_px": float(px[i]),
            "ret": float(gross),
            "ret_net": float(gross - cost),
            #  MFE/MAE over the bars actually seen, so they are meaningful on an
            #  unsettled signal too.
            "mfe": float(np.max(rel)), "mae": float(np.min(rel)),
            "hit_tp": bool(i_tp is not None), "hit_sl": bool(i_sl is not None),
            "last_px": float(px[-1]),
        })
        rows.append(rec)

    out = pd.DataFrame(rows)[OUTCOME_COLUMNS]
    os.makedirs(STORE_DIR, exist_ok=True)
    tmp = OUTCOMES + ".tmp"
    out.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, OUTCOMES)
    return out


def summary(outcomes: Optional[pd.DataFrame] = None) -> str:
    """What the store can honestly say, which at the start is 'not yet'.

    THE POWER STATEMENT IS THE POINT. A31 measured that distinguishing this
    kind of edge from random needs 46,856 months, and recorded conflating an
    EFFECT statement with a POWER statement as its own error. So this prints how
    much evidence exists BEFORE it prints any number derived from it.
    """
    o = outcomes if outcomes is not None else load_outcomes()
    if o.empty:
        return ("SIGNAL STORE EMPTY. Nothing has been emitted yet, so there is\n"
                "no out-of-sample evidence. This is the honest state on day one\n"
                "and it is why the store exists: it only ever gets later.")
    s = o[o["settled"].astype(bool)]
    L = [f"emitted   {len(o):,} signals over "
         f"{o['ticker'].nunique()} names, "
         f"{pd.to_datetime(o['asof']).min().date()} -> "
         f"{pd.to_datetime(o['asof']).max().date()}",
         f"settled   {len(s):,}  ({len(s) / max(len(o), 1):.0%})",
         f"open      {len(o) - len(s):,}"]
    if s.empty:
        L.append("")
        L.append("NOTHING HAS SETTLED YET. No rate, mean or win count is")
        L.append("defined on an unsettled sample, so none is printed.")
        return "\n".join(L)
    r = s["ret_net"].dropna()
    if len(r):
        #  Mean AND mean log, always together (A36: they disagreed in SIGN).
        pos = r[r > -1 + 1e-9]
        ml = float(np.mean(np.log1p(pos))) if len(pos) else float("nan")
        L += ["",
              f"mean net      {r.mean():+.2%}",
              f"median net    {r.median():+.2%}",
              f"mean log      {ml:+.4f}   "
              f"(an equal-weighted holder is paid the mean, a sequential",
              f"                        trader the mean log -- A36 measured "
              f"them disagreeing in SIGN)",
              f"positive      {(r > 0).mean():.1%}  of {len(r)} settled"]
        #  The power statement, before anything is concluded from the numbers.
        sd = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
        if np.isfinite(sd) and sd > 0 and abs(r.mean()) > 0:
            need = (2.0 * sd / abs(r.mean())) ** 2
            L += ["",
                  f"POWER: at t=2, distinguishing this mean from ZERO needs "
                  f"~{need:,.0f} settled signals.",
                  f"       {len(r)} exist. This is a POWER statement, not an "
                  f"effect statement (A31)."]
    return "\n".join(L)
