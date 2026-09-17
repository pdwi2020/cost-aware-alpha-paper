"""Recompute the `overnight_gap` column of features_all.parquet on one price basis.

Why
---
`_sanitize_ohlcv_for_tier2` winsorises daily returns at +/-50% and rebuilds the
close as first_close * cumprod(1 + clipped_return). It used to rebuild only the
close, leaving `open` on the original basis, while `overnight_gap` is
open_t / close_{t-1} - 1. From a ticker's first clipped day onward the two legs
therefore differed by the constant cumulative clip factor, so every later
overnight gap for that ticker was multiplied by it:

    WY    1 clipped day (2010-07-20)  close_clean/close_raw = 1.3121
          -> stored gap = 0.7621 * (1 + true gap) - 1, a near-constant -23.8%
    VRTX  2 clipped days               factor 0.8945 -> +11.8%
    AAPL, XOM  no clipped day          exact

Correlation between stored and recomputed is +1.0000 with a constant
multiplicative offset, which is the signature. Screen 0 does not remove it,
because the affected names are liquid index members.

`_sanitize_ohlcv_for_tier2` now rescales the whole bar, so this script recomputes
the column from the same daily panel and the same lag the tier-2 builder applies.
`overnight_gap` needs no 1-minute data, so this is a cheap daily-only patch.

Run:
    python3 -u src/features/patch_overnight_gap.py            # writes the panel
    python3 -u src/features/patch_overnight_gap.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data import universe_v3
from src.features.build_features_all import (
    SANITIZE_CAP_T2,
    _sanitize_ohlcv_for_tier2,
)

FEATURES_PATH = ROOT / "data" / "processed" / "features_all.parquet"
UNIVERSE_START = "2010-01-01"
UNIVERSE_END = "2026-09-14"


def log(msg: str) -> None:
    print(msg, flush=True)


def recompute_overnight_gap() -> pd.Series:
    """Lag-1 overnight_gap per (ticker, date), open and close on one basis."""
    daily = universe_v3.load_panel(UNIVERSE_START, UNIVERSE_END)
    log(f"  panel rows: {len(daily):,}")
    daily = _sanitize_ohlcv_for_tier2(daily, SANITIZE_CAP_T2)

    close_w = daily.pivot(index="date", columns="ticker", values="close")
    open_w = daily.pivot(index="date", columns="ticker", values="open")
    close_w.index = pd.to_datetime(close_w.index)
    open_w.index = pd.to_datetime(open_w.index)
    close_w = close_w.sort_index()
    open_w = open_w.sort_index().reindex_like(close_w)

    gap_w = open_w / close_w.shift(1) - 1.0
    gap_w = gap_w.replace([np.inf, -np.inf], np.nan)

    out = (
        gap_w.stack(future_stack=True)
        .rename("overnight_gap")
        .rename_axis(index=["date", "ticker"])
        .reorder_levels(["ticker", "date"])
        .sort_index()
    )
    dates = out.index.get_level_values("date")
    out = out[(dates >= pd.Timestamp(UNIVERSE_START))
              & (dates <= pd.Timestamp(UNIVERSE_END))]
    # Same lag the tier-2 builder applies to every tier-2 column.
    return out.groupby(level="ticker").shift(1).sort_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log("=== Patching overnight_gap (open/close basis fix) ===\n")
    t0 = time.time()

    new_gap = recompute_overnight_gap()
    finite = new_gap.replace([np.inf, -np.inf], np.nan).dropna()
    log(f"  recomputed |overnight_gap|: p50={finite.abs().median():.5f} "
        f"p999={finite.abs().quantile(0.999):.5f} max={finite.abs().max():.3f}")

    panel = pd.read_parquet(FEATURES_PATH)
    old_gap = panel["overnight_gap"]
    aligned = new_gap.reindex(panel.index)

    log(f"  panel rows {len(panel):,} | old non-null {old_gap.notna().sum():,} "
        f"| new non-null {aligned.notna().sum():,}")

    worst_old = old_gap.abs().groupby(level="ticker").mean().nlargest(6)
    worst_new = aligned.abs().groupby(level="ticker").mean().nlargest(6)
    log("\n  largest mean |overnight_gap| by ticker")
    log(f"    before: {', '.join(f'{t}={v:.4f}' for t, v in worst_old.items())}")
    log(f"    after:  {', '.join(f'{t}={v:.5f}' for t, v in worst_new.items())}")

    if args.dry_run:
        log("\n  --dry-run: panel not written")
        return

    other_before = panel.drop(columns=["overnight_gap"])
    panel["overnight_gap"] = aligned.to_numpy()
    assert panel.drop(columns=["overnight_gap"]).equals(other_before), \
        "columns other than overnight_gap changed"

    panel.to_parquet(FEATURES_PATH)
    log(f"\nSaved → {FEATURES_PATH} "
        f"({FEATURES_PATH.stat().st_size / 1e6:.1f} MB)")
    log(f"Total elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
