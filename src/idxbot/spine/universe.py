"""The universe, and the survivorship bias it is known to carry.

THE FINDING THIS MODULE EXISTS TO RECORD
----------------------------------------
The price spine holds 843 IDX tickers. **Every one of them is currently
listed.** Not one name in the spine stopped trading more than two years ago,
and checking 25 companies known to have been delisted from IDX in 2025 found
**0 of 25 present**.

That is not a small gap. IDX delisted roughly 70 companies in 2025 - about 8%
of the listed universe in a single year - and by construction those are the
names that went to zero, got suspended into oblivion, or were taken out. A
backtest run on this universe is a backtest run on the winners, and it is
inflated for exactly that reason.

CLAUDE.md §5 requires delisted names in the spine. They are not obtainable from
any free source reached so far: Yahoo answers ``possibly delisted; no timezone
found`` for every one of SRIL, MYRX, FREN and MAMI, and the ticker list the
spine was built from is a ``TICKER,marketcap`` file, which can only ever
contain live names because market cap does not exist for a dead one.

So the brief's fallback applies: measure the bias, state it, and carry it as a
KNOWN quantity rather than an unknown one. That is what :func:`bias_estimate`
and :func:`sensitivity` do.

THE PART THAT IS ACTUALLY ACTIONABLE
------------------------------------
The bias is not uniform. It is concentrated almost entirely in illiquid names,
because a company does not delist out of nowhere - it stops trading first. A
name heading for delisting spends months or years barely trading, which is
visible in the spine as stale bars long before the delisting itself.

So a liquidity filter removes most of the bias, and a large-cap strategy
carries very little of it. That is worth far more than a correction factor:
it means the repo's large-cap work is close to safe and its small-cap work is
not. :func:`bias_estimate` takes the weighting scheme for that reason.
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

#: Delistings as a fraction of the listed universe, per year. 2025 was roughly
#: 70 of ~900. It is quoted as a RANGE because 2025 was plainly a clean-up year
#: - IDX cleared a backlog of names suspended for six months or more - and
#: treating it as the steady state would overstate the long-run bias.
DELIST_RATE_LOW = 0.01          # a quiet year
DELIST_RATE_HIGH = 0.08         # 2025, the purge

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
    lo = bias_estimate(0.10, DELIST_RATE_LOW, weighting=weighting)["bias"] * 100
    hi = bias_estimate(0.10, DELIST_RATE_HIGH, weighting=weighting)["bias"] * 100
    article = "an" if weighting[:1].lower() in "aeiou" else "a"
    return (f"Universe is survivorship-biased: 0 of {len(KNOWN_DELISTED)} "
            f"known-delisted IDX names are present, and ~70 companies delisted "
            f"in 2025 alone. On {article} {weighting}-weighted book this inflates "
            f"annual return by roughly {lo:.1f} to {hi:.1f} percentage points. "
            f"The figure is a bound, not a correction - the delisted history is "
            f"not obtainable from any free source reached.")
