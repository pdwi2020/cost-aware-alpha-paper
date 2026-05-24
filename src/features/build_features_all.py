"""Week 3: Full feature pipeline — Tier 1 + Tier 2 merged.

Loads Tier 1 from parquet, computes Tier 2 features (intraday microstructure,
cross-asset macro, crowding proxies), adds the term_spread × mom_12_1
interaction term, merges, and saves features_all.parquet.

Run directly:
    cd ~/ML_Paper && python3 src/features/build_features_all.py

Prerequisites:
    data/processed/features_tier1.parquet  (from build_features.py)
    data/processed/daily_ohlcv.parquet     (from build_features.py)
"""

import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.data.universe_builder import get_universe_tickers
from src.features.tier2_extended import build_tier2_features

CATALOG      = os.environ.get("DUCKDB_CATALOG", "/Volumes/Crucial X9/data/catalog.duckdb")
SP500_DIR    = ROOT / "datasets" / "sp500_holdings"
OUT_DIR      = ROOT / "data" / "processed"
TIER1_PATH   = OUT_DIR / "features_tier1.parquet"
OHLCV_PATH   = OUT_DIR / "daily_ohlcv.parquet"
OUT_PATH     = OUT_DIR / "features_all.parquet"

UNIVERSE_START = "2010-01-01"
UNIVERSE_END   = "2024-12-31"


def main() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Week 3: Full Feature Engineering (Tier 1 + Tier 2) ===\n")

    # ------------------------------------------------------------------
    # 1. Load Tier 1 features
    # ------------------------------------------------------------------
    print("Loading Tier 1 features...")
    tier1 = pd.read_parquet(TIER1_PATH)
    tier1.index.names = ["ticker", "date"]
    tier1.index = tier1.index.set_levels(
        [tier1.index.levels[0],
         pd.to_datetime(tier1.index.levels[1])],
    )
    print(f"  Tier 1 shape: {tier1.shape}")

    # ------------------------------------------------------------------
    # 2. Load daily OHLCV (for overnight gap + crowding base)
    # ------------------------------------------------------------------
    print("Loading daily OHLCV...")
    daily = pd.read_parquet(OHLCV_PATH)
    daily["date"] = pd.to_datetime(daily["date"])
    print(f"  OHLCV shape: {daily.shape}")

    # ------------------------------------------------------------------
    # 3. SP500 universe tickers
    # ------------------------------------------------------------------
    sp500_tickers = get_universe_tickers(str(SP500_DIR))
    print(f"  SP500 union tickers: {len(sp500_tickers)}")

    # ------------------------------------------------------------------
    # 4. Build Tier 2 features
    # ------------------------------------------------------------------
    db = duckdb.connect(CATALOG, read_only=True)

    t0 = time.time()
    tier2 = build_tier2_features(
        db=db,
        sp500_tickers=sp500_tickers,
        daily_ohlcv=daily,
        start_date=UNIVERSE_START,
        end_date=UNIVERSE_END,
        intraday_batch_size=150,
        crowding_window=20,
    )
    print(f"\nTier 2 total time: {time.time()-t0:.0f}s  →  {tier2.shape}")

    # ------------------------------------------------------------------
    # 5. Merge Tier 1 + Tier 2
    # ------------------------------------------------------------------
    print("\nMerging Tier 1 + Tier 2...")
    out = tier1.join(tier2, how="left")

    # ------------------------------------------------------------------
    # 6. Interaction term: term_spread × mom_12_1
    #    Both features are already lag-1 in tier1, so NO extra shift.
    # ------------------------------------------------------------------
    if "term_spread" in out.columns and "mom_12_1" in out.columns:
        out["term_spread_x_mom"] = out["term_spread"] * out["mom_12_1"]
        print("  Added interaction: term_spread × mom_12_1")

    # ------------------------------------------------------------------
    # 7. Filter to universe window and drop fully-NaN feature rows
    # ------------------------------------------------------------------
    date_idx = out.index.get_level_values("date")
    out = out[
        (date_idx >= pd.Timestamp(UNIVERSE_START)) &
        (date_idx <= pd.Timestamp(UNIVERSE_END))
    ]
    target_cols = [c for c in out.columns if c.startswith("target")]
    feat_cols   = [c for c in out.columns if c not in target_cols]
    out = out.dropna(subset=feat_cols, how="all")

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    dates = out.index.get_level_values("date")
    tier1_feats = [c for c in out.columns if c in tier1.columns and not c.startswith("target")]
    tier2_feats = [c for c in out.columns if c not in tier1.columns]

    print(f"\n=== Output ===")
    print(f"Shape:          {out.shape}")
    print(f"Date range:     {dates.min().date()} → {dates.max().date()}")
    print(f"Tickers:        {out.index.get_level_values('ticker').nunique()}")
    print(f"Tier 1 feats:   {len(tier1_feats)}")
    print(f"Tier 2 feats:   {len(tier2_feats)}")
    print(f"\nTier 2 NaN rates:")
    if tier2_feats:
        print(out[tier2_feats].isna().mean().round(3).to_string())
    print(f"\nTarget coverage:")
    print(out[target_cols].notna().mean().round(3).to_string())

    out.to_parquet(OUT_PATH)
    print(f"\nSaved → {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    main()
