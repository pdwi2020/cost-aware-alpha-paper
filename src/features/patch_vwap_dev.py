"""Recompute the `vwap_dev` column of features_all.parquet on one price basis.

Why this is a patch and not a full rebuild
------------------------------------------
`vwap_dev` was built as (adjusted daily close - raw intraday VWAP) / adjusted
daily close. The two legs sat on different split bases, so for a stock that
later split N:1 the feature equalled 1 - N on every earlier date: a constant.
Measured mean |vwap_dev| per ticker included CMG 49.000 (50:1), AMZN 18.998
(20:1) and ORLY 14.000 (15:1), against a median ticker value of 0.0045, and
twelve tickers carried 76% of the total absolute signal.

Only this one column is affected. `vol_clock`, `vol_sig_ratio` and
`overnight_gap` are each ratios whose legs share a basis, no interaction in
`feature_spec.ADDED_INTERACTIONS` reads `vwap_dev`, and the panel stores raw
(not cross-sectionally standardised) values. `features_tier1.parquet` was
deleted to reclaim disk, and rebuilding tier 1 to change one tier-2 column
would recompute a great deal that is provably unchanged. So this script redoes
exactly the affected column and overwrites it in place, reproducing the
lag-1-per-ticker step that `build_tier2_features` applies to every tier-2
feature.

Run:
    python3 -u src/features/patch_vwap_dev.py            # writes the panel
    python3 -u src/features/patch_vwap_dev.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data import universe_v3
from src.data.recover_delisted_prices import month_files
from src.features.tier2_extended import (
    VWAP_DEV_SANITY_MAX,
    _intraday_agg_file,
)

FEATURES_PATH = ROOT / "data" / "processed" / "features_all.parquet"
UNIVERSE_START = "2010-01-01"
UNIVERSE_END = "2026-09-14"


def log(msg: str) -> None:
    print(msg, flush=True)


def recompute_vwap_dev(tickers: list[str]) -> pd.Series:
    """Lag-1 vwap_dev per (ticker, date), both legs from the raw 1-minute bars."""
    db = duckdb.connect()
    files = month_files(pd.Timestamp(UNIVERSE_START), pd.Timestamp(UNIVERSE_END))
    log(f"  querying {len(files)} monthly 1-minute files ...")

    parts, t0 = [], time.time()
    for i, path in enumerate(files, start=1):
        parts.append(
            _intraday_agg_file(db, path.as_posix(), tickers,
                               UNIVERSE_START, UNIVERSE_END)
        )
        if i % 24 == 0 or i == len(files):
            log(f"    {i}/{len(files)} files ({time.time() - t0:.0f}s)")

    intra = pd.concat(parts, ignore_index=True)
    intra["date"] = pd.to_datetime(intra["date"])
    dev = (
        (intra["close_rth"] - intra["vwap"])
        / intra["close_rth"].replace(0, np.nan)
    )
    intra = intra.assign(vwap_dev=dev)

    finite = intra["vwap_dev"].replace([np.inf, -np.inf], np.nan).dropna()
    p999 = float(finite.abs().quantile(0.999)) if len(finite) else float("nan")
    log(f"  recomputed |vwap_dev|: p50={finite.abs().median():.5f} "
        f"p999={p999:.5f} max={finite.abs().max():.5f}")
    if p999 > VWAP_DEV_SANITY_MAX:
        raise ValueError(
            f"vwap_dev still implausible (p99.9 = {p999:.3f}); the two legs "
            "are on different bases."
        )

    out = (
        intra.set_index(["ticker", "date"])["vwap_dev"]
        .sort_index()
    )
    # Same window filter and lag that build_tier2_features applies to every
    # tier-2 column, in that order.
    dates = out.index.get_level_values("date")
    out = out[(dates >= pd.Timestamp(UNIVERSE_START))
              & (dates <= pd.Timestamp(UNIVERSE_END))]
    return out.groupby(level="ticker").shift(1).sort_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log("=== Patching vwap_dev (split-basis fix) ===\n")
    t0 = time.time()

    tickers = universe_v3.pit_tickers(UNIVERSE_START, UNIVERSE_END)
    log(f"  PIT members: {len(tickers)}")

    new_dev = recompute_vwap_dev(sorted(tickers))

    log("\n  loading panel ...")
    panel = pd.read_parquet(FEATURES_PATH)
    old_dev = panel["vwap_dev"]

    aligned = new_dev.reindex(panel.index)
    log(f"  panel rows {len(panel):,} | old non-null {old_dev.notna().sum():,} "
        f"| new non-null {aligned.notna().sum():,}")

    worst_old = old_dev.abs().groupby(level="ticker").mean().nlargest(6)
    worst_new = aligned.abs().groupby(level="ticker").mean().nlargest(6)
    log("\n  largest mean |vwap_dev| by ticker")
    log(f"    before: {', '.join(f'{t}={v:.3f}' for t, v in worst_old.items())}")
    log(f"    after:  {', '.join(f'{t}={v:.5f}' for t, v in worst_new.items())}")

    if args.dry_run:
        log("\n  --dry-run: panel not written")
        return

    other_before = panel.drop(columns=["vwap_dev"])
    panel["vwap_dev"] = aligned.to_numpy()
    # The whole point is that nothing else moves.
    assert panel.drop(columns=["vwap_dev"]).equals(other_before), \
        "columns other than vwap_dev changed"

    panel.to_parquet(FEATURES_PATH)
    log(f"\nSaved → {FEATURES_PATH} "
        f"({FEATURES_PATH.stat().st_size / 1e6:.1f} MB)")
    log(f"Total elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
