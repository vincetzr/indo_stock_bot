"""The hybrid: a weekly threshold, monitored daily.

What the measurement said
-------------------------
The daily 8% band anticipates **100%** of weekly 12% flips, by a median of 9
days. It is also wrong most of the time: only **46%** of its flips are followed
by a weekly flip the same way within 60 days. So the lead is real and the noise
is real, and a hybrid is only worth building if it takes one without the other.

Two ways to combine them, and only one is right
-----------------------------------------------
The obvious hybrid - "flip the weekly when the daily flips" - imports the 54%
false-alarm rate wholesale. It reacts sooner and is wrong more, which is not an
improvement, it is a smaller band wearing a disguise.

The hybrid that works keeps the weekly THRESHOLD and drops only the weekly
CLOCK:

    weekly painter   flip when a FRIDAY close is 12% from the running extreme
    hybrid           flip when ANY close is 12% from the running extreme

Identical rule, identical threshold, identical number of legs in the limit -
the hybrid simply does not wait until Friday to notice. It cannot manufacture
false alarms the weekly rule would not also make, because the trigger level is
the same; it can only reach the same conclusion sooner.

Everything is scored against the FINAL weekly segmentation, which is the picture
being reproduced. Reaching it earlier is the entire claim.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from idxbot.config import load_config          # noqa: E402
from idxbot.data.cache import Cache            # noqa: E402
from idxbot.data.ohlcv import YahooOHLCV       # noqa: E402
from leg_signals import market_caps            # noqa: E402
from legpaint import unadjusted_weekly, zigzag_labels   # noqa: E402
from paint_daily import unadjusted_daily       # noqa: E402
from paint_live import band_state              # noqa: E402


def hybrid_state(daily: pd.Series, band: float) -> pd.Series:
    """The weekly threshold checked on every session instead of only on Friday.

    This is deliberately just ``band_state`` on daily closes at the WEEKLY band
    width. The point is not a new rule - it is the same rule on a faster clock.
    """
    st, _ = band_state(daily.to_numpy(float), band)
    return pd.Series(st, index=daily.index)


def weekly_live_colour(w: pd.Series, band: float) -> pd.Series:
    """What the weekly painter shows at each Friday, as it goes."""
    px = w.to_numpy(float)
    out = np.full(len(px), np.nan)
    for i in range(30, len(px)):
        lab = zigzag_labels(px[:i + 1], band, drop_last=False)
        out[i] = lab[i]
    return pd.Series(out, index=w.index)


def compare(t: str, loader: YahooOHLCV, band: float,
            ages_wk: Tuple[int, ...] = (1, 2, 4, 6, 8, 13)) -> Optional[Dict]:
    w = unadjusted_weekly(loader, t, start="2015-01-01")
    d = unadjusted_daily(loader, t, start="2015-01-01")
    if w is None or d is None or len(w) < 200:
        return None
    px = w.to_numpy(float)
    final = zigzag_labels(px, band, drop_last=False)      # the picture to reproduce

    hyb = hybrid_state(d, band).reindex(w.index, method="ffill")
    wk_state, _ = band_state(px, band)

    out = {"ticker": t, "weeks": len(w)}
    for k in ages_wk:
        hit_w = tot_w = hit_h = tot_h = 0
        for now in range(40, len(px)):
            i = now - k
            if i < 0:
                continue
            lab = zigzag_labels(px[:now + 1], band, drop_last=False)
            if np.isfinite(lab[i]) and np.isfinite(final[i]):
                tot_w += 1
                hit_w += int(lab[i] == final[i])
            # the hybrid's answer for bar i, using only sessions up to `now`
            cut = d.index <= w.index[now]
            hs, _ = band_state(d[cut].to_numpy(float), band)
            hcol = pd.Series(hs, index=d.index[cut]).reindex(
                [w.index[i]], method="ffill").iloc[0]
            if np.isfinite(final[i]) and np.isfinite(hcol):
                tot_h += 1
                hit_h += int(hcol == final[i])
        out[f"wk_{k}w"] = hit_w / tot_w if tot_w else np.nan
        out[f"hy_{k}w"] = hit_h / tot_h if tot_h else np.nan
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=float, default=0.12)
    ap.add_argument("--names", type=int, default=12)
    ap.add_argument("--min-mcap", type=float, default=1e13)
    args = ap.parse_args()
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()
    loader = YahooOHLCV(cfg, Cache(cfg.path("data.cache_dir", "data/cache")))
    mc = market_caps()
    big = sorted(mc[mc >= args.min_mcap].index)

    rows = []
    for t in big:
        r = compare(t, loader, args.band)
        if r:
            rows.append(r)
            print(f"  {t} done ({len(rows)}/{args.names})")
        if len(rows) >= args.names:
            break
    R = pd.DataFrame(rows)
    R.to_csv("reports/hybrid_painter.csv", index=False)

    ages = (1, 2, 4, 6, 8, 13)
    print(f"\n{'=' * 84}\n HYBRID vs WEEKLY — colour already final, at the "
          f"{args.band:.0%} threshold\n{'=' * 84}")
    print(f" scored against the finished weekly picture, {len(R)} large caps\n")
    print(f" {'bar age':>9}{'weekly only':>14}{'hybrid':>10}{'gain':>9}")
    for k in ages:
        a, b = R[f"wk_{k}w"].median(), R[f"hy_{k}w"].median()
        print(f" {str(k) + 'w':>9}{a:>14.1%}{b:>10.1%}{b - a:>+9.1%}")

    def first90(prefix: str) -> Optional[int]:
        for k in ages:
            if R[f"{prefix}_{k}w"].median() >= 0.90:
                return k
        return None
    fw, fh = first90("wk"), first90("hy")
    print(f"\n reaches 90%:  weekly at {fw}w, hybrid at {fh}w")
    print("\n -> reports/hybrid_painter.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
