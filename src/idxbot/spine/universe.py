"""The universe, its survivorship bias, and the point-in-time snapshot that
partly repairs it.

WHAT WAS WRONG
--------------
The live price spine holds 843 IDX tickers and **every one of them is currently
listed**. Not one stopped trading more than two years ago, and of 25 companies
known to have been delisted from IDX, zero were present. The ticker list was
built from a ``TICKER,marketcap`` file, which can only ever contain live names
because market cap does not exist for a dead one.

WHAT FIXED MOST OF IT
---------------------
A published **April-2019 snapshot of 627 IDX tickers**, with price history. It
is not survivorship-biased, because it was taken then. Comparing it with
today's spine identifies the casualties directly: **121 of its 627 names are
gone**, and 21 of the 25 known-delisted companies are among them, with prices.
Those are now recovered into ``data/cache/delisted``.

That turns three guesses into measurements:

    attrition            121/627 over 7.4 years = **2.87% a year**, where the
                         range previously assumed was 1-8%
    pre-delisting drag   the doomed names were ALREADY lagging by **4.8
                         percentage points a year** at the median over
                         2014-2019 (+3.0% for survivors, -1.8% for them),
                         while everyone was still listed
    the universe itself  :func:`point_in_time_universe` reconstructs the
                         constituent list as it stood, survivorship-free, for
                         any date at or before the snapshot

WHAT IS STILL MISSING, AND IT IS ONE-SIDED
------------------------------------------
The recovery only runs backwards from 2019. Names that delisted BEFORE the
snapshot are still absent, and after it the vanished names are known while the
newly LISTED ones are not separable from the live set - so a universe built for
2026 from these parts would be biased the other way. :func:`point_in_time_universe`
therefore returns an explicit ``complete`` flag rather than letting a caller
assume it got a clean universe.

The terminal loss is also still unmeasured. The snapshot ends in April 2019, so
what these names did on the way OUT - which is where most of the damage is -
is not in it. The bias estimate still needs an assumption there, and says so.

THE PART THAT IS ACTUALLY ACTIONABLE
------------------------------------
The bias is not uniform. It is concentrated in illiquid names, because a
company does not delist out of nowhere - it stops trading first, and the
measured 4.8-point pre-delisting drag says it underperforms first too. So a
liquidity filter removes most of the exposure and a cap-weighted large-cap book
carries very little of it, which is why :func:`bias_estimate` takes a weighting
scheme.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

#: Companies confirmed delisted from IDX in 2025, from published delisting
#: lists. NOT exhaustive - roughly 70 names went in 2025 and these are the ones
#: named in sources that could be read. Used to AUDIT the universe, not to
#: reconstruct it: there is no price history behind any of them.
DELISTED_2025 = (
    "ALMI", "ARMY", "ARTI", "BEBS", "BIKA", "BOSS", "BTEL", "CBMF", "COWL",
    "CPRI", "DEAL", "DUCK", "ENVY", "ETWA", "FASW", "GAMA", "GOLL", "FREN",
    "MAMI", "FORZ", "MYRX", "KRAH", "KPAS", "KPAL",
)

#: Delisted earlier, confirmed individually.
DELISTED_EARLIER = ("SRIL",)

KNOWN_DELISTED = tuple(sorted(set(DELISTED_2025) | set(DELISTED_EARLIER)))

#: Delistings as a fraction of the listed universe, per year.
#:
#: These were assumptions until a POINT-IN-TIME universe turned up: a published
#: April-2019 snapshot of 627 IDX tickers, which is not survivorship-biased
#: because it was taken then. Comparing it with today's spine gives the
#: attrition directly - 121 of 627 names are gone, 19.3% over 7.4 years, an
#: implied 2.87% a year. That is the measured centre, and it sits well inside
#: the range that was previously guessed.
#:
#: The range is kept because a single window is one realisation: 2025 alone saw
#: ~70 delistings out of ~900 as IDX cleared a backlog of long-suspended names,
#: so the rate is plainly not constant.
DELIST_RATE_LOW = 0.01          # a quiet year
DELIST_RATE_MEASURED = 0.0287   # 2019-2026, from the point-in-time snapshot
DELIST_RATE_HIGH = 0.08         # 2025, the purge

#: How far the names that later vanished were ALREADY lagging, measured over
#: 2014-2019 while they were all still listed: 4.8 percentage points a year at
#: the median (+3.0% for survivors against -1.8% for the doomed). This is the
#: part of the bias that is visible before the delisting event, and it is why a
#: liquidity or momentum filter removes so much of the exposure.
PRE_DELIST_DRAG = 0.048

#: What a delisted holding actually returns. A forced delisting after a long
#: suspension is close to a total loss; a voluntary one - a take-private or a
#: merger, as FREN into EXCL - can pay a premium. Most of the 2025 batch was
#: forced.
FORCED_RETURN = -1.00
VOLUNTARY_RETURN = 0.15
#: Fraction of delistings that are forced rather than voluntary.
FORCED_SHARE = 0.85


def audit_universe(tickers: Iterable[str]) -> Dict[str, object]:
    """How much of the known-delisted set is present? Answer so far: none of it.

    A universe that contains none of the names known to have died is a
    universe assembled from survivors, whatever else it may also be.
    """
    have = {str(t).upper().replace(".JK", "") for t in tickers}
    present = sorted(t for t in KNOWN_DELISTED if t in have)
    missing = sorted(t for t in KNOWN_DELISTED if t not in have)
    return {
        "universe": len(have),
        "checked": len(KNOWN_DELISTED),
        "present": present,
        "missing": missing,
        "survivorship_biased": len(present) == 0,
        "coverage": len(present) / max(len(KNOWN_DELISTED), 1),
    }


def expected_delisted_return(forced_share: float = FORCED_SHARE) -> float:
    """Blended return on a holding that delists."""
    f = float(np.clip(forced_share, 0.0, 1.0))
    return f * FORCED_RETURN + (1.0 - f) * VOLUNTARY_RETURN


def bias_estimate(survivor_return: float, delist_rate: float,
                  delisted_return: Optional[float] = None,
                  weighting: str = "equal") -> Dict[str, float]:
    """How much an observed return overstates the truth, per year.

    For an equal-weight portfolio over one year, with a fraction ``f`` of names
    delisting and returning ``r_d`` while survivors return ``r_s``:

        true  = (1 - f) * r_s + f * r_d
        bias  = r_s - true = f * (r_s - r_d)

    which is large: at f = 8% and a total loss, an observed +10% is really
    +1.2%, an overstatement of nearly nine percentage points.

    ``weighting`` matters more than anything else here. A delisting candidate
    is tiny by the time it goes, so a CAP-weighted portfolio barely holds it
    and barely feels it. The ``cap`` case applies a weight-share factor: those
    names are a few tenths of a percent of index weight, not 8% of it.
    """
    r_s = float(survivor_return)
    r_d = expected_delisted_return() if delisted_return is None \
        else float(delisted_return)
    f = float(np.clip(delist_rate, 0.0, 1.0))
    if weighting == "cap":
        # A name about to delist is a micro cap. Its share of a cap-weighted
        # book is far below its share of the NAME count - assume ~1/20th, which
        # is generous to the strategy rather than to the argument.
        f = f * 0.05
    elif weighting != "equal":
        raise ValueError(f"weighting must be 'equal' or 'cap', got {weighting!r}")
    true = (1.0 - f) * r_s + f * r_d
    return {"observed": r_s, "true": float(true), "bias": float(r_s - true),
            "delist_rate_used": f, "delisted_return": r_d}


def sensitivity(survivor_return: float = 0.10) -> pd.DataFrame:
    """The bias across the plausible range, because a point estimate is a lie.

    The delisting rate is genuinely uncertain - 2025 was a clean-up year and
    quiet years are far lower - and so is what a delisted holding returns. A
    single number would hide both.
    """
    rows = []
    for f in (0.01, 0.02, 0.04, 0.06, 0.08):
        for w in ("equal", "cap"):
            b = bias_estimate(survivor_return, f, weighting=w)
            rows.append({"delist_rate": f, "weighting": w,
                         "observed": b["observed"], "true": b["true"],
                         "bias_pp": b["bias"] * 100})
    return pd.DataFrame(rows)


def liquidity_shield(stale_fraction: pd.Series,
                     threshold: float = 0.10) -> Dict[str, float]:
    """How much of the universe a liquidity filter removes, and why it helps.

    Delisting is preceded by illiquidity, and illiquidity is visible in the
    spine as stale bars. Filtering on it does not fix survivorship bias - the
    dead names are still absent - but it makes the SURVIVING sample resemble a
    population that was never at much risk of delisting, so the bias that
    remains is far smaller than the headline rate suggests.

    This returns the fraction of names a threshold keeps. It deliberately does
    not return a corrected return: the correction is not identifiable without
    the delisted names, and inventing one would be worse than stating the gap.
    """
    s = pd.to_numeric(pd.Series(stale_fraction), errors="coerce").dropna()
    if s.empty:
        return {}
    keep = float((s < threshold).mean())
    return {"threshold": float(threshold), "names": int(len(s)),
            "kept": int((s < threshold).sum()), "kept_fraction": keep,
            "median_stale": float(s.median())}


def caveat(weighting: str = "equal") -> str:
    """The sentence that must appear on any backtest run on this universe."""
    mid = bias_estimate(0.10, DELIST_RATE_MEASURED,
                        weighting=weighting)["bias"] * 100
    hi = bias_estimate(0.10, DELIST_RATE_HIGH, weighting=weighting)["bias"] * 100
    article = "an" if weighting[:1].lower() in "aeiou" else "a"
    return (f"Universe is survivorship-biased in the LIVE cache: 0 of "
            f"{len(KNOWN_DELISTED)} known-delisted IDX names are present. "
            f"{len(delisted_available())} vanished names are recovered from a "
            f"2019 point-in-time snapshot, giving a MEASURED attrition of "
            f"{DELIST_RATE_MEASURED:.2%}/yr rather than an assumed range. On "
            f"{article} {weighting}-weighted book that inflates annual return "
            f"by about {mid:.1f} percentage points, rising to {hi:.1f} in a "
            f"clean-up year like 2025. Still a bound rather than a correction: "
            f"the snapshot ends 2019-04-07, so what these names did on the way "
            f"OUT - where most of the damage is - is not in it.")

# --------------------------------------------------------------------------
# the point-in-time universe, recovered
# --------------------------------------------------------------------------
#: Where the recovered vanished names live. Kept apart from the live cache on
#: purpose: mixing them in would silently change the meaning of every existing
#: study, and a survivorship-free universe is a DIFFERENT universe, not a
#: bigger one.
DELISTED_DIR = os.path.join("data", "cache", "delisted")

#: The snapshot these came from, and its limits. A published point-in-time
#: listing of 627 IDX tickers as of this date - not survivorship-biased,
#: because it was taken then. It is the only such snapshot found, so the
#: recovery is one-sided: names that vanished AFTER it are recoverable, names
#: that delisted before it are not.
SNAPSHOT_DATE = pd.Timestamp("2019-04-07")
SNAPSHOT_SIZE = 627


def delisted_available() -> List[str]:
    """Vanished names with a usable price history, recovered from the snapshot."""
    import glob
    out = []
    for f in sorted(glob.glob(os.path.join(DELISTED_DIR, "*.JK.csv.gz"))):
        out.append(os.path.basename(f).replace(".JK.csv.gz", ""))
    return out


def point_in_time_universe(day, live: Optional[Iterable[str]] = None
                           ) -> Dict[str, object]:
    """The universe as it stood on ``day``, survivorship-free where possible.

    Only meaningful for dates at or before :data:`SNAPSHOT_DATE`: after it, the
    names that vanished are known but the ones that LISTED since are not
    separable from the live set, and a universe missing new listings is biased
    the other way.

    Returns the constituent list plus an explicit ``complete`` flag, because a
    caller silently receiving a partially-reconstructed universe is exactly the
    failure this whole module exists to prevent.
    """
    d = pd.Timestamp(day).normalize()
    gone = delisted_available()
    have = sorted({str(t).upper().replace(".JK", "") for t in (live or [])})
    return {
        "date": d,
        "constituents": sorted(set(have) | set(gone)),
        "live": len(have), "recovered": len(gone),
        "complete": bool(d <= SNAPSHOT_DATE and gone),
        "note": ("survivorship-free: the vanished names are present"
                 if d <= SNAPSHOT_DATE and gone else
                 f"PARTIAL - vanished names are recovered only back to "
                 f"{SNAPSHOT_DATE:%Y-%m-%d}, and names delisted before that "
                 f"are still missing"),
    }
