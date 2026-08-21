"""Point-in-time broker code master: which firm was behind a code, and when.

THE PROBLEM CLAUDE.md §5 NAMES
------------------------------
"Mergers and licence changes reassign codes. Without this you will attribute
one firm's history to another."

True, but the shape of the risk is not what it first looks like, and getting
that shape right decides how much of this matters.

**A rename is not a reassignment.** Code YP has been eTrading Securities (from
2003), Daewoo Securities Indonesia (2013) and Mirae Asset Sekuritas Indonesia
(2016). Three names, three owners - and one continuous business with one
continuous client base. For §9's behavioural fingerprints that continuity is
what matters, not the signage. Splitting YP's history at each rename would
destroy a real fifteen-year record to fix a problem that is not there.

**A merger IS a discontinuity**, because the client base changes underneath the
code in a single step. When UBS absorbed Credit Suisse the flow behind CS did
not gradually become UBS flow; it moved.

So this module tracks TWO things separately:

    entity   the continuing business. Survives renames and ownership changes.
             Fingerprints may be compared across an entity's whole history.
    naming   what the code was called when. Presentation only.

and marks the events that genuinely break comparability - mergers and
retirements - as ``discontinuity``.

CONFIDENCE IS PART OF THE DATA, NOT A FOOTNOTE
----------------------------------------------
Some of this is verified against sources read this session; some is widely
repeated in the market and not verified here. Every record carries its own
confidence and :func:`firm` returns it, so a caller cannot silently rely on a
date that was half-remembered. Nothing here is authoritative against IDX's own
Anggota Bursa directory, which is the only real master.

WHY THE EMPIRICAL AUDIT IS WEAKER THAN IT LOOKS
-----------------------------------------------
The obvious check is to watch the data for codes that appear or vanish. On the
2025-2026 panel three codes (RB, PI, MU) appear mid-sample - and none of them
is a new licence. The broker summary is a TOP TEN, so a code shows up only on
days it happens to be large, and absence is almost always smallness rather than
non-existence. :func:`activity_audit` reports the candidates and refuses to
call them code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: Confidence in a record. ``verified`` means a source was read for it in this
#: repo's history; ``reported`` means it is widely stated in the market and was
#: not independently checked here.
CONFIDENCE = ("verified", "reported", "current_only", "unknown")


@dataclass(frozen=True)
class Naming:
    """What a code was called over one interval. ``until=None`` means current."""

    code: str
    name: str
    since: Optional[pd.Timestamp]
    until: Optional[pd.Timestamp]
    entity: str
    confidence: str = "reported"
    source: str = ""
    #: True when the client base changed underneath the code - a merger or a
    #: reassignment. Fingerprints must NOT be compared across one of these.
    discontinuity: bool = False
    note: str = ""


def _ts(v) -> Optional[pd.Timestamp]:
    return None if v is None else pd.Timestamp(v).normalize()


#: The code history. Deliberately short: it holds the codes where a change is
#: known, not all 90-odd members, because inventing dates for the rest would
#: make the file look authoritative while being guesswork. A code absent here
#: is reported as unknown-history, which is honest and actionable.
_HISTORY: List[Naming] = [
    # -- the worked example, verified this session ---------------------------
    Naming("YP", "eTrading Securities", _ts("2003-01-01"), _ts("2013-01-01"),
           entity="mirae-id", confidence="verified",
           source="company history; rebranded eTrading 2003",
           note="same continuing business as Daewoo and Mirae below"),
    Naming("YP", "Daewoo Securities Indonesia", _ts("2013-01-01"),
           _ts("2016-01-01"), entity="mirae-id", confidence="verified",
           source="acquired by Daewoo 2013"),
    Naming("YP", "Mirae Asset Sekuritas Indonesia", _ts("2016-01-01"), None,
           entity="mirae-id", confidence="verified",
           source="Mirae Asset Financial Group from 2016",
           note="largest retail order flow on IDX"),

    # -- the merger, which IS a discontinuity --------------------------------
    Naming("CS", "Credit Suisse Sekuritas Indonesia", None, _ts("2023-06-12"),
           entity="cs-id", confidence="reported",
           source="UBS/Credit Suisse global merger completed 2023",
           discontinuity=True,
           note="flow moved to UBS (AK); do not compare CS before with AK "
                "after as one series"),
    Naming("AK", "UBS Sekuritas Indonesia", None, None, entity="ubs-id",
           confidence="reported",
           note="absorbed the Credit Suisse Indonesia business in 2023; its "
                "client base is not continuous across that date"),

    # -- ownership changes that are renames, not discontinuities -------------
    Naming("OD", "Danareksa Sekuritas", None, _ts("2021-01-01"),
           entity="danareksa-id", confidence="reported"),
    Naming("OD", "BRI Danareksa Sekuritas", _ts("2021-01-01"), None,
           entity="danareksa-id", confidence="reported",
           note="BRI took control; continuing business"),
    Naming("ZP", "Maybank Kim Eng Securities", None, _ts("2022-01-01"),
           entity="maybank-id", confidence="reported"),
    Naming("ZP", "Maybank Sekuritas Indonesia", _ts("2022-01-01"), None,
           entity="maybank-id", confidence="reported"),
    Naming("YU", "CIMB Securities Indonesia", None, _ts("2018-01-01"),
           entity="cgs-id", confidence="reported"),
    Naming("YU", "CGS-CIMB Sekuritas Indonesia", _ts("2018-01-01"),
           _ts("2024-01-01"), entity="cgs-id", confidence="reported"),
    Naming("YU", "CGS International Sekuritas Indonesia", _ts("2024-01-01"),
           None, entity="cgs-id", confidence="reported"),
    Naming("DR", "OSK Nusadana Securities Indonesia", None, _ts("2019-01-01"),
           entity="rhb-id", confidence="reported"),
    Naming("DR", "RHB Sekuritas Indonesia", _ts("2019-01-01"), None,
           entity="rhb-id", confidence="reported"),
]


def _covers(n: Naming, day: pd.Timestamp) -> bool:
    return ((n.since is None or n.since <= day)
            and (n.until is None or day < n.until))


def current_names() -> Dict[str, str]:
    """Today's code -> name map from ``config/brokers.yaml``, if readable.

    The registry knows what 66 codes are called NOW but nothing about when
    they were called it. That is still worth having: "current name known,
    history unknown" is a better answer than "nothing known", as long as the
    two are never confused - which is why it comes back with its own
    confidence rather than being folded into the dated history.
    """
    try:
        import yaml                                          # noqa: PLC0415
    except ImportError:
        return {}
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, os.pardir, os.pardir, os.pardir, "config",
                        "brokers.yaml")
    try:
        with open(os.path.normpath(path)) as fh:
            blob = yaml.safe_load(fh) or {}
    except Exception:                                        # noqa: BLE001
        return {}
    out: Dict[str, str] = {}
    for code, rec in (blob.get("brokers") or {}).items():
        name = rec.get("name") if isinstance(rec, dict) else rec
        if name:
            out[str(code).upper().strip()] = str(name)
    return out


def firm(code: str, day, registry: Optional[Dict[str, str]] = None
         ) -> Dict[str, object]:
    """Who was behind ``code`` on ``day``, with the confidence attached.

    Three distinct answers, and keeping them distinct is the whole point:

        ``verified`` / ``reported``   a dated record covers this date
        ``current_only``              the code's name today is known, but
                                      nothing about when it changed. Safe to
                                      LABEL a report, never safe to compare
                                      two eras with.
        ``unknown``                   nothing at all

    "We know nothing about this code's history" and "this code never changed"
    look identical in a result table and mean opposite things, so the second
    is never inferred from the first.
    """
    c = str(code).upper().strip()
    d = pd.Timestamp(day).normalize()
    hits = [n for n in _HISTORY if n.code == c and _covers(n, d)]
    if not hits:
        known = any(n.code == c for n in _HISTORY)
        reg = current_names() if registry is None else registry
        if not known and c in reg:
            return {"code": c, "name": reg[c], "entity": None,
                    "confidence": "current_only", "discontinuity": False,
                    "source": "config/brokers.yaml", "ambiguous": False,
                    "note": ("name as of today only; no dated history, so two "
                             "eras of this code must not be compared")}
        return {"code": c, "name": None, "entity": None,
                "confidence": "unknown",
                "note": ("code has recorded history but none covering this date"
                         if known else "no recorded history for this code"),
                "discontinuity": False}
    # If two records overlap, the more specific (later `since`) wins, and the
    # overlap is itself worth surfacing rather than silently resolving.
    hits.sort(key=lambda n: (n.since is not None, n.since or pd.Timestamp.min))
    n = hits[-1]
    return {"code": c, "name": n.name, "entity": n.entity,
            "confidence": n.confidence, "note": n.note,
            "discontinuity": n.discontinuity, "source": n.source,
            "ambiguous": len(hits) > 1}


def same_entity(code: str, a, b) -> bool:
    """Is a code's history comparable between two dates?

    False when the code spans a discontinuity - a merger or reassignment -
    because the clients behind it changed in one step. This is the guard §9's
    fingerprints need: comparing a broker with itself across a merger measures
    the merger.
    """
    fa, fb = firm(code, a), firm(code, b)
    if fa["entity"] is None or fb["entity"] is None:
        return False
    if fa["entity"] != fb["entity"]:
        return False
    lo, hi = sorted([pd.Timestamp(a).normalize(), pd.Timestamp(b).normalize()])
    c = str(code).upper().strip()
    for n in _HISTORY:
        if n.code != c or not n.discontinuity:
            continue
        edge = n.until or n.since
        if edge is not None and lo < edge <= hi:
            return False
    return True


def history(code: Optional[str] = None) -> pd.DataFrame:
    """The recorded history, as a frame - so it can be eyeballed and audited."""
    rows = [n for n in _HISTORY
            if code is None or n.code == str(code).upper().strip()]
    return pd.DataFrame([{
        "code": n.code, "name": n.name, "since": n.since, "until": n.until,
        "entity": n.entity, "confidence": n.confidence,
        "discontinuity": n.discontinuity, "source": n.source, "note": n.note}
        for n in rows]).sort_values(["code", "since"], na_position="first")


def coverage(codes: Sequence[str]) -> Dict[str, object]:
    """How much of an observed code set has any recorded history at all."""
    seen = sorted({str(c).upper().strip() for c in codes if str(c).strip()})
    known = sorted({n.code for n in _HISTORY})
    have = [c for c in seen if c in known]
    return {"observed": len(seen), "with_history": len(have),
            "fraction": len(have) / max(len(seen), 1),
            "missing": [c for c in seen if c not in known]}


def activity_audit(df: pd.DataFrame, min_sessions: int = 20,
                   edge_days: int = 90) -> pd.DataFrame:
    """Codes that appear or vanish mid-sample - CANDIDATES, not conclusions.

    On a top-ten broker summary a code is listed only on days it was large
    enough to rank, so absence is nearly always smallness rather than
    non-existence. Run on the 2025-2026 panel this flags RB, PI and MU, none of
    which is a new licence.

    It is kept because on a FULL-depth rekap the same check becomes strong:
    there, absence really does mean the member did not trade. The ``censored``
    column says which regime the caller is in, so the result cannot be read as
    stronger than it is.
    """
    if df is None or df.empty or "broker" not in df or "date" not in df:
        return pd.DataFrame(columns=["broker", "first", "last", "sessions",
                                     "event", "censored"])
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    complete = bool(d["complete"].all()) if "complete" in d else False
    start, end = d["date"].min(), d["date"].max()
    g = d.groupby("broker")["date"].agg(first="min", last="max",
                                        sessions="nunique").reset_index()
    g = g[g["sessions"] >= int(min_sessions)]
    lag = pd.Timedelta(days=int(edge_days))
    g["event"] = np.where(
        g["last"] < end - lag, "stopped appearing",
        np.where(g["first"] > start + lag, "started appearing", ""))
    g["censored"] = not complete
    return g[g["event"] != ""].sort_values("sessions", ascending=False)
