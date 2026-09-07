#!/usr/bin/env python3
"""H57 — a point-in-time share count, and the conclusion it lets us re-test.

WHAT WAS BLOCKED. A25 records that market capitalisation is NOT used anywhere in
this repo, because the only share-count source found was frozen at 2024-07-10
and applying a 2024 count to a 2010 bar is look-ahead. Indonesian rights issues
are exactly what makes that wrong. So SIZE has been proxied by trailing turnover
in every study that needed it -- including H52, whose conclusion is that the
strength+calm alpha IS a size effect because the edge collapses moving upmarket.

WHAT IS NOW AVAILABLE. The delisted-recovery dataset carries `listed_shares`
PER BAR. Measured: 405 of 949 names (43%) change their count, with ratios up to
41x -- PANI 410,000,000 -> 16,883,595,500, PYFA 21x, BBKP 16x over 26 distinct
values. That is not a rounding difference; a frozen count would misprice PANI's
2019 capitalisation by a factor of forty.

Window: 2019-07-29 to 2025-02-21, the dataset's own span. Outside it there is
still no point-in-time count and market cap stays unavailable -- which is why
this replaces nothing and adds a column.

--------------------------------------------------------------------- REGISTERED
Written before any cell was scored.

  P1  Turnover rank and market-cap rank agree closely but not perfectly, and
      the disagreement is SYSTEMATIC rather than noise: a large company that
      trades rarely ranks small on turnover and large on cap. PREDICTION:
      Spearman above 0.7, with the residual concentrated in low-turnover
      large caps.

  P2  PREDICTED NULL, AND THE ONE THAT MATTERS. H52 concluded that the
      strength+calm edge collapses as the universe moves upmarket -- +12.9% on
      the whole board down to +0.7% in the top 40 -- and therefore that the
      alpha IS a size effect. Every tier in that study was cut on TURNOVER.
      PREDICTION: the same collapse appears when the tiers are cut on real
      market cap. If it does NOT, H52's conclusion is a property of the proxy
      rather than of size, and it has to be withdrawn.

  P3  Market cap adds nothing to the screen that turnover did not already
      carry. PREDICTION: confirmed -- H27 found eleven collinear price features
      with an empty interaction space, and a twelfth correlated one should not
      change that.

Nothing here is claimed against the Bonferroni bar (0.05/347 = 0.00014). P2 is
a REPLICATION CHECK on an existing conclusion, not a new claim, and the honest
outcomes are "H52 survives the better instrument" or "H52 was measuring its
proxy".

------------------------------------------------------------------ WHAT HAPPENED
Added after the run. The registration above is unaltered.

  C0 IS THE REASON ANY OF THIS IS READABLE, AND IT FAILED FIRST. A positive
  control was written in before P2 could be scored: does the TURNOVER ruler --
  H52's own -- reproduce H52's own collapse on THIS statistic? On the
  arithmetic mean of the 60-session forward return it does NOT. The ladder is
  hump-shaped (-0.0046 at `all`, +0.0092 at top 100, +0.0024 at top 40), so a
  market-cap arm differing from it would have said nothing about market cap.
  On the MEAN LOG the collapse is there and monotone (+0.0253 -> +0.0164).
  A36 records the arithmetic mean and the mean log disagreeing in SIGN on this
  repo's data; here they disagree about whether an effect EXISTS. H52 measured
  a compounded portfolio, so the mean log is the arm P2 is scored on -- chosen
  because it is the quantity H52 measured, not because it passed.

  P2 CONFIRMED, AND ITS ANSWER IS "THE RULER DOES NOT MATTER". Row-matched on
  identical bars, turnover falls +0.0358 -> +0.0184 and point-in-time market
  cap falls +0.0358 -> +0.0185. H52's size conclusion is not an artefact of
  its proxy.

  BUT THE COLLAPSE IS AN EARLY-HALF PHENOMENON. Fall from `all` to `top 40`:
  turnover +0.0292 early against +0.0029 late, market cap +0.0278 against
  +0.0038. Same sign in both halves, a factor of TEN apart in size. And the
  two rulers agree only to within 0.0199, which is 68% of the largest fall
  being measured. So the ruler question is answered and the underlying effect
  is not independently established.

  P3 IS NOT CONFIRMED AND IS NOT OVERTURNED. Market cap does mark a cell
  turnover misses -- big companies that trade rarely (MYOR, CMRY, BYAN, BNGA,
  TBIG, GEMS) carry the screen's largest edge, +0.0460 at z +2.14 against a
  clustered null. +2.14 is nowhere near the 0.00014 bar and the cell was
  chosen after seeing P1, so it is a lead, not a finding.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from bhbench import load                                          # noqa: E402

SRC = "/home/user/wildangunawan/dataset-saham-idx/Saham/Semua"
OUT = os.path.join("data", "spine", "shares_pit.parquet")
FIRST = pd.Timestamp("2019-07-29")
LAST = pd.Timestamp("2025-02-21")


def build(src: str = SRC, out: str = OUT) -> pd.DataFrame:
    """(ticker, date, listed_shares) for every name the dataset carries."""
    rows = []
    for f in sorted(glob.glob(src + "/*.csv")):
        tk = os.path.basename(f)[:-4]
        try:
            d = pd.read_csv(f, usecols=["date", "listed_shares"],
                            parse_dates=["date"])
        except Exception:
            continue
        d = d.dropna(subset=["listed_shares"])
        d = d[d["listed_shares"] > 0]
        if d.empty:
            continue
        d["ticker"] = tk
        rows.append(d)
    S = pd.concat(rows, ignore_index=True)
    S = S.sort_values(["ticker", "date"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    S.to_parquet(out, index=False)
    return S


def attach(P: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    """Join the count as of each bar, carried FORWARD only.

    A share count is known from the day it takes effect, never before, so a
    backward fill would place a post-rights-issue count on pre-issue bars --
    the exact look-ahead this file exists to remove.
    """
    #  merge_asof requires the `on` key sorted GLOBALLY, not within `by` groups.
    #  Sorting by (ticker, date) raises "left keys must be sorted" -- the error
    #  is about the date column across the whole frame.
    P = P.sort_values("date", kind="mergesort").reset_index(drop=True)
    S = S.sort_values("date", kind="mergesort").reset_index(drop=True)
    out = pd.merge_asof(P, S, on="date", by="ticker", direction="backward")
    out["mktcap"] = out["listed_shares"] * out["close"]
    return out


TIERS = (None, 150, 100, 60, 40)     # H52's own tiers, verbatim
HI_PCT, VOL_PCT = 0.90, 0.50         # H26's cell, verbatim
MIN_N = 300                          # both sides, or the cell is unreadable


def screen_flag(F: pd.DataFrame) -> pd.Series:
    """H26's strength-plus-calm cell, cut WITHIN each date."""
    hi = F.groupby("date")["hi52"].transform(lambda s: s.quantile(HI_PCT))
    vo = F.groupby("date")["vol60"].transform(lambda s: s.quantile(VOL_PCT))
    return (F["hi52"] >= hi) & (F["vol60"] <= vo)


def tier_mask(F: pd.DataFrame, ruler: str, tier) -> np.ndarray:
    """Boolean mask for the top-`tier` names by `ruler`, WITHIN each date.

    `tier=None` is the whole universe. Ranking within the date is what makes
    "top 40" mean "the 40 largest that day" rather than "names that ended up
    large" -- the survivorship form of the same error.
    """
    if tier is None:
        return np.ones(len(F), dtype=bool)
    r = F.groupby("date")[ruler].rank(ascending=False, method="first")
    return (r <= tier).to_numpy()


def edge_at(val: np.ndarray, screen: np.ndarray, keep: np.ndarray):
    """Screen-minus-rest mean of `val` inside `keep`, NaNs dropped."""
    ok = keep & np.isfinite(val)
    s, o = val[ok & screen], val[ok & ~screen]
    if len(s) < MIN_N or len(o) < MIN_N:
        return None
    return {"n": int(len(s)), "screen": float(s.mean()), "rest": float(o.mean()),
            "edge": float(s.mean() - o.mean())}


def blocks_of(F: pd.DataFrame) -> list:
    """Row indices grouped by (ticker, calendar year)."""
    blk = (F["ticker"].astype(str) + "|"
           + F["date"].dt.year.astype(str)).to_numpy()
    order = np.argsort(blk, kind="mergesort")
    _keys, starts = np.unique(blk[order], return_index=True)
    return np.split(order, starts[1:])


def block_null(val: np.ndarray, screen: np.ndarray, keep: np.ndarray,
               groups: list, draws: int = 200, seed: int = 0) -> tuple:
    """Clustered null: reassign whole (ticker, year) blocks' LABELS.

    A17 and A25 between them fix the two ways to get this wrong. Shuffling ROWS
    leaves the null far too tight, because one name contributes ~20 nearly
    identical bars a month. Shuffling INSIDE a (ticker, year) block is nearly a
    no-op for the same reason. What destroys the mapping without destroying the
    dependence is moving one block's screen labels onto another block's rows.
    """
    rng = np.random.default_rng(seed)
    out = []
    n = len(screen)
    for _ in range(draws):
        perm = rng.permutation(len(groups))
        new = np.empty(n, dtype=bool)
        for gi, pi in enumerate(perm):
            src, dst = groups[pi], groups[gi]
            take = screen[src]
            if len(take) >= len(dst):
                new[dst] = take[:len(dst)]
            else:
                reps = int(np.ceil(len(dst) / max(len(take), 1)))
                new[dst] = np.tile(take, reps)[:len(dst)]
        r = edge_at(val, new, keep)
        if r is not None:
            out.append(r["edge"])
    if not out:
        return float("nan"), float("nan"), 0
    a = np.asarray(out)
    return float(a.mean()), float(a.std(ddof=1)), len(a)


def tier_table(F: pd.DataFrame, ruler: str, col: str, draws: int = 0,
               seed: int = 0, indent: str = "    ") -> list:
    """One tier ladder. Returns the edge sequence, prints it."""
    val = F[col].to_numpy(float)
    screen = F["screen"].to_numpy(bool)
    groups = blocks_of(F) if draws else None
    head = (f"{indent}{'tier':<10}{'n':>9}{'screen':>10}{'rest':>10}"
            f"{'edge':>9}")
    if draws:
        head += f"{'null':>9}{'sd':>8}{'z':>7}"
    print(head)
    print(indent + "-" * (len(head) - len(indent)))
    seq = []
    for t in TIERS:
        keep = tier_mask(F, ruler, t)
        r = edge_at(val, screen, keep)
        nm = "all" if t is None else f"top {t}"
        if r is None:
            print(f"{indent}{nm:<10}   insufficient")
            continue
        line = (f"{indent}{nm:<10}{r['n']:>9,}{r['screen']:>+10.4f}"
                f"{r['rest']:>+10.4f}{r['edge']:>+9.4f}")
        if draws:
            mu, sd, _ = block_null(val, screen, keep, groups, draws, seed)
            z = (r["edge"] - mu) / sd if sd and sd > 0 else float("nan")
            line += f"{mu:>+9.4f}{sd:>8.4f}{z:>+7.2f}"
        print(line)
        seq.append(r["edge"])
    return seq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--draws", type=int, default=200)
    args = ap.parse_args()

    if args.rebuild or not os.path.exists(OUT):
        S = build()
        print(f"built {OUT}: {len(S):,} rows, {S['ticker'].nunique()} names, "
              f"{S['date'].min().date()} -> {S['date'].max().date()}")
    else:
        S = pd.read_parquet(OUT)

    #  ------------------------------------------------------------------ C0
    #  THE POSITIVE CONTROL COMES FIRST, AND IT IS NOT OPTIONAL.
    #  P2 compares a turnover ruler with a market-cap ruler and asks whether
    #  the SHAPE differs. That question is only answerable if the turnover arm
    #  reproduces H52's known collapse ON THIS STATISTIC. If it does not, a
    #  difference between the two arms is a fact about the statistic or the
    #  window, not about the ruler -- and reading it as a ruler effect is
    #  exactly the error A26's sine wave and A36's Q0 exist to stop.
    #
    #  WHICH AVERAGE, AND IT DECIDES WHETHER THE CONTROL PASSES AT ALL.
    #  H52 measured a COMPOUNDED portfolio, and A36 records the arithmetic mean
    #  and the mean log disagreeing in SIGN on this repo's own data. Both are
    #  printed below. The arithmetic arm is reported first and is NOT the arm
    #  P2 is scored on, because it is not the quantity H52 measured.
    ALL = load()
    ALL = ALL[ALL["elig"]].copy().reset_index(drop=True)
    ALL["f60"] = (ALL.groupby("ticker")["adj_close"].shift(-60)
                  / ALL["adj_close"] - 1.0)
    ALL["l60"] = np.log1p(ALL["f60"].clip(lower=-0.999))
    ALL["screen"] = screen_flag(ALL)
    print("\nC0  POSITIVE CONTROL — does H52's collapse reproduce on this")
    print("    statistic, on the FULL panel, cut by turnover as H52 cut it?")
    print(f"    panel {ALL['date'].min().date()} -> {ALL['date'].max().date()}, "
          f"{len(ALL):,} eligible bars, {ALL['ticker'].nunique()} names")
    print("\n    arithmetic mean of the 60-session forward return")
    a0 = tier_table(ALL, "tv60", "f60")
    print(f"    edge {a0[0]:+.4f} -> {a0[-1]:+.4f}   "
          f"{'declines' if a0[-1] < a0[0] else 'does NOT decline'}"
          f"  (hump: {'yes' if max(a0) > max(a0[0], a0[-1]) else 'no'})")
    print("\n    MEAN LOG — the quantity a compounding holder is paid")
    c0 = tier_table(ALL, "tv60", "l60", draws=args.draws)
    c0_ok = len(c0) >= 3 and c0[-1] < c0[0]
    print(f"    edge {c0[0]:+.4f} -> {c0[-1]:+.4f}   "
          f"{'DECLINES — control passes' if c0_ok else 'does NOT decline'}")
    if not c0_ok:
        print("\n    C0 FAILS. P2 is reported below and is NOT scored.")

    #  ------------------------------------------------------------------ P1
    P = load()
    P = P[(P["date"] >= FIRST) & (P["date"] <= LAST)]
    F = attach(P, S)
    F = F[F["elig"] & F["mktcap"].notna() & (F["mktcap"] > 0)].copy()
    F = F.reset_index(drop=True)
    cov = len(F) / max(int((P["elig"]).sum()), 1)
    print(f"\npanel in window: {len(F):,} eligible bars, "
          f"{F['ticker'].nunique()} names with a point-in-time cap "
          f"({cov:.1%} of eligible bars in the window)")

    print("\nP1  turnover rank vs market-cap rank, WITHIN each date")
    F["r_tv"] = F.groupby("date")["tv60"].rank(pct=True)
    F["r_mc"] = F.groupby("date")["mktcap"].rank(pct=True)
    sp = F[["r_tv", "r_mc"]].corr(method="spearman").iloc[0, 1]
    F["gap"] = F["r_mc"] - F["r_tv"]
    print(f"    Spearman(turnover rank, cap rank) = {sp:+.3f}")
    print(f"    |rank gap| median {F['gap'].abs().median():.3f}, "
          f"p90 {F['gap'].abs().quantile(0.90):.3f}")
    big_thin = F[(F["r_mc"] > 0.8) & (F["r_tv"] < 0.5)]
    small_busy = F[(F["r_mc"] < 0.5) & (F["r_tv"] > 0.8)]
    print(f"    BIG but THIN  (cap top 20%, turnover bottom half): "
          f"{len(big_thin):,} bars, {big_thin['ticker'].nunique()} names")
    print(f"    SMALL but BUSY(cap bottom half, turnover top 20%): "
          f"{len(small_busy):,} bars, {small_busy['ticker'].nunique()} names")
    if len(big_thin):
        top = (big_thin.groupby("ticker").size().sort_values(ascending=False)
               .head(8))
        print(f"    most misranked by turnover: {', '.join(top.index)}")

    #  ------------------------------------------------------------------ P2
    F["f60"] = (F.groupby("ticker")["adj_close"].shift(-60)
                / F["adj_close"] - 1.0)
    F["l60"] = np.log1p(F["f60"].clip(lower=-0.999))
    F["screen"] = screen_flag(F)

    #  The window arm needs its OWN control, because the window is 5.6 years of
    #  one regime and the full panel is 26. Running the turnover ruler on the
    #  SAME ROWS is what separates "the ruler changed the answer" from "the
    #  window changed the answer" -- A19's error class, and the reason both
    #  arms below are computed on an identical frame.
    print("\nC1  the same turnover cut, restricted to the market-cap window "
          f"({F['date'].min().date()} -> {F['date'].max().date()})")
    c1 = tier_table(F, "tv60", "l60", draws=args.draws)
    c1_ok = len(c1) >= 3 and c1[-1] < c1[0]
    if c1:
        print(f"    edge {c1[0]:+.4f} -> {c1[-1]:+.4f}   "
              f"{'DECLINES' if c1_ok else 'does NOT decline'}")

    print("\nP2  the same cut, ruled by POINT-IN-TIME MARKET CAP")
    p2 = tier_table(F, "mktcap", "l60", draws=args.draws)
    p2_ok = len(p2) >= 3 and p2[-1] < p2[0]
    if p2:
        print(f"    edge {p2[0]:+.4f} -> {p2[-1]:+.4f}   "
              f"{'DECLINES' if p2_ok else 'does NOT decline'}")

    #  THE HALF-SPLIT COMES BEFORE THE VERDICT, not after it. A18 records it as
    #  the only replication test this repo trusts, and a ladder built from
    #  overlapping 60-session windows reads far more solid than it is. The
    #  split is on the RETURN period, so a bar belongs to the half its forward
    #  window closes in.
    print("\n    half-split of the same two ladders (edge, mean log)")
    mid = F["date"].quantile(0.5)
    print(f"    {'ruler':<10}{'half':<8}" + "".join(
        f"{('all' if t is None else 'top %d' % t):>10}" for t in TIERS))
    half: Dict[tuple, list] = {}
    for ruler in ("tv60", "mktcap"):
        for lab, m in (("early", F["date"] <= mid), ("late", F["date"] > mid)):
            G = F[m].reset_index(drop=True)
            val, sc = G["l60"].to_numpy(float), G["screen"].to_numpy(bool)
            cells, seq = [], []
            for t in TIERS:
                r = edge_at(val, sc, tier_mask(G, ruler, t))
                cells.append(f"{r['edge']:>+10.4f}" if r else f"{'--':>10}")
                seq.append(r["edge"] if r else float("nan"))
            half[(ruler, lab)] = seq
            print(f"    {ruler:<10}{lab:<8}" + "".join(cells))

    #  A BOOLEAN "does it decline" IS A WAY OF NOT REPORTING A NULL (A33). The
    #  fall is printed as a magnitude in each half, because a decline of 0.003
    #  and a decline of 0.029 are the same boolean and different findings.
    falls = {k: (v[0] - v[-1]) for k, v in half.items()}
    print("\n    fall from `all` to `top 40`, by half:")
    for r in ("tv60", "mktcap"):
        print(f"      {r:<8} early {falls[(r, 'early')]:+.4f}   "
              f"late {falls[(r, 'late')]:+.4f}")
    halves_ok = {r: min(falls[(r, "early")], falls[(r, "late")]) > 0
                 for r in ("tv60", "mktcap")}
    weak = min(abs(falls[(r, h)]) for r in ("tv60", "mktcap")
               for h in ("early", "late"))
    strong = max(abs(falls[(r, h)]) for r in ("tv60", "mktcap")
                 for h in ("early", "late"))

    #  How far apart are the two rulers ON THE SAME ROWS? This is P2's actual
    #  question, and it is answerable half by half even when the SHAPE is not
    #  stable across halves.
    gaps = [abs(half[("tv60", h)][i] - half[("mktcap", h)][i])
            for h in ("early", "late") for i in range(len(TIERS))
            if np.isfinite(half[("tv60", h)][i])
            and np.isfinite(half[("mktcap", h)][i])]
    print(f"\n    max |turnover edge - mktcap edge| over 10 (tier, half) cells: "
          f"{max(gaps):.4f}")
    print(f"    for scale, the largest ladder fall in any half:           "
          f"{strong:.4f}")

    #  The verdict is a CONJUNCTION and it is written out rather than implied,
    #  because "the two arms differ" is only evidence about the ruler when the
    #  turnover arm is known to work on this statistic in this window.
    print("\n    VERDICT")
    if not c0_ok:
        print("    C0 failed: the turnover ruler does NOT reproduce H52's")
        print("    collapse on this statistic even on the full panel. P2 is")
        print("    UNSCOREABLE — the instrument is not shown able to detect")
        print("    the effect it is being asked to compare against.")
    elif not c1_ok:
        print("    C0 passed, C1 failed: the decline is in the full panel and")
        print("    NOT in this 5.6-year window, on the SAME ruler. The window")
        print("    is the confound, so P2 would be measuring the window and")
        print("    not the ruler. UNSCOREABLE.")
    else:
        if p2_ok:
            print("    Pooled over the window, BOTH rulers decline, and they")
            print("    decline by the same amount. On the question P2 asks —")
            print("    is the tier collapse a property of the RULER — the")
            print("    answer is NO: a point-in-time market-cap ruler and a")
            print("    turnover ruler give the same ladder to within the noise.")
        else:
            print("    C0 and C1 decline on turnover; the market-cap arm does")
            print("    NOT on the same rows. P2 FAILED: the collapse is a")
            print("    property of the turnover proxy, and what decays with the")
            print("    tier is LIQUIDITY rather than capitalisation.")
        print("\n    WHAT THIS DOES NOT ESTABLISH.")
        if not (halves_ok["tv60"] and halves_ok["mktcap"]):
            print("    The shape does not even hold in sign in both halves")
            print(f"    (turnover {halves_ok['tv60']}, mktcap "
                  f"{halves_ok['mktcap']}).")
        else:
            print(f"    The fall holds in sign in every half, but its size does")
            print(f"    not: {weak:.4f} in the weakest half against {strong:.4f}")
            print(f"    in the strongest, a factor of {strong / max(weak, 1e-9):.0f}.")
        print(f"    And the two rulers agree only to within {max(gaps):.4f}, "
              f"which is {max(gaps) / max(strong, 1e-9):.0%}")
        print("    of the largest fall being measured.")
        print("    So this study says the ruler is not what produces")
        print("    the tier effect. It does NOT independently establish the")
        print("    effect, and it is not evidence for H52's LEVEL — a per-bar")
        print("    gross mean log is not a costed, quarterly-rebalanced,")
        print("    turnover-charged portfolio CAGR.")

    #  ------------------------------------------------------------------ P3
    print("\nP3  does cap carry anything turnover did not? "
          "(screen edge where the two rulers disagree)")
    F["r_mc_resid"] = F["r_mc"] - F["r_tv"]
    q_lo = F["r_mc_resid"].quantile(0.25)
    q_hi = F["r_mc_resid"].quantile(0.75)
    for nm, m in (("cap >> turnover (thin big)", F["r_mc_resid"] >= q_hi),
                  ("cap << turnover (busy small)", F["r_mc_resid"] <= q_lo)):
        G = F[m].reset_index(drop=True)
        val = G["l60"].to_numpy(float)
        sc = G["screen"].to_numpy(bool)
        keep = np.ones(len(G), dtype=bool)
        r = edge_at(val, sc, keep)
        if r is None:
            print(f"    {nm:<30} insufficient")
            continue
        mu, sd, _ = block_null(val, sc, keep, blocks_of(G), draws=args.draws)
        z = (r["edge"] - mu) / sd if sd and sd > 0 else float("nan")
        print(f"    {nm:<30} screen {r['screen']:+.4f}  rest {r['rest']:+.4f}  "
              f"edge {r['edge']:+.4f}  null {mu:+.4f}+-{sd:.4f}  z {z:+.2f}  "
              f"n={r['n']:,}")


if __name__ == "__main__":
    main()
