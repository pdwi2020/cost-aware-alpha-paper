"""Week 3: Full feature pipeline — Tier 1 + Tier 2 merged.

Loads Tier 1 from parquet, computes Tier 2 features (intraday microstructure,
cross-asset macro, crowding proxies), adds the term_spread × mom_12_1
interaction term, merges, and saves features_all.parquet.

SANITIZATION NOTE:
    Tier-1 features (including return-derived columns and both targets) are
    already sanitized if features_tier1.parquet was built with build_features.py
    (default mode, ±50% cap).
    This script also sanitizes the close prices passed to build_tier2_features()
    so that overnight_gap uses the same artifact-free price series.

Run directly:
    cd ~/ML_Paper && python3 src/features/build_features_all.py

Prerequisites:
    data/processed/features_tier1.parquet  (from build_features.py --sanitize)
    data/processed/daily_ohlcv.parquet     (from build_features.py)
"""

import argparse
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
from src.data.screen0 import screen0_eligibility, trailing_adv_usd, survivorship_delta
from src.features.tier2_extended import build_tier2_features

# ±50% daily return cap for corporate-action artifact removal in Tier-2 features.
# Must match build_features.py SANITIZE_CAP for full coherence.
SANITIZE_CAP_T2 = 0.50


def _sanitize_ohlcv_for_tier2(daily: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Replace `close` in OHLCV with a sanitized proxy for Tier-2 feature building.

    Used specifically for overnight_gap (open_T / close_{T-1} - 1) in Tier-2.
    The original `close` is replaced with close_clean = first × cumprod(1 + r_clean)
    where r_clean = clip(pct_change(close), -cap, +cap).

    ADV / dollar-volume computations in crowding features also use close × volume,
    but those are structural features (not return-derived) so raw close is acceptable;
    however sanitized close is also fine since the change is small.
    """
    daily = daily.copy()
    close_w = daily.pivot(index="date", columns="ticker", values="close")
    r_raw = close_w.pct_change(fill_method=None)
    n_clipped = int((r_raw.abs() > cap).sum().sum())
    if n_clipped > 0:
        print(f"  [T2 sanitize] Clipping {n_clipped} (ticker,day) cells with |ret|>{cap:.0%}")
    r_clean = r_raw.clip(lower=-cap, upper=cap)
    first_close = close_w.iloc[0]
    growth = (1.0 + r_clean.fillna(0.0)).cumprod()
    close_clean_w = growth.multiply(first_close, axis="columns")
    close_clean_w[close_w.isna()] = np.nan

    # Long-format merge
    cc_long = (
        close_clean_w.stack(future_stack=True)
        .rename("close_clean")
        .reset_index()
    )
    cc_long.columns = ["date", "ticker", "close_clean"]
    cc_long["date"] = pd.to_datetime(cc_long["date"])
    daily = daily.merge(cc_long, on=["ticker", "date"], how="left")
    daily.rename(columns={"close": "close_raw", "close_clean": "close"}, inplace=True)
    return daily

CATALOG      = os.environ.get("DUCKDB_CATALOG", "/Volumes/Crucial X9/data/catalog.duckdb")
SP500_DIR    = ROOT / "datasets" / "sp500_holdings"
OUT_DIR      = ROOT / "data" / "processed"
TIER1_PATH   = OUT_DIR / "features_tier1.parquet"
OHLCV_PATH   = OUT_DIR / "daily_ohlcv.parquet"
OUT_PATH     = OUT_DIR / "features_all.parquet"

UNIVERSE_START = "2010-01-01"
UNIVERSE_END   = "2024-12-31"


def main() -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Week 3: Full Feature Engineering (Tier 1 + Tier 2)")
    parser.add_argument(
        "--universe", choices=["pit", "liquidity"], default="pit",
        help="Screen 0 universe: 'pit' (point-in-time S&P 500 annual, default) "
             "or 'liquidity' (top-N by trailing ADV, fully reconstructible)."
    )
    args = parser.parse_args()
    universe_choice = args.universe

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
    #    Sanitize close prices for Tier-2 feature building so overnight_gap
    #    uses the same artifact-free price series as Tier-1 features.
    # ------------------------------------------------------------------
    print("Loading daily OHLCV...")
    daily = pd.read_parquet(OHLCV_PATH)
    daily["date"] = pd.to_datetime(daily["date"])
    print(f"  OHLCV shape: {daily.shape}")
    print(f"  Sanitizing OHLCV for Tier-2 (cap=±{SANITIZE_CAP_T2:.0%}) ...")
    daily = _sanitize_ohlcv_for_tier2(daily, SANITIZE_CAP_T2)
    print(f"  Sanitized OHLCV shape: {daily.shape}")

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
    # 8. Screen 0: PIT membership + lagged price/ADV eligibility flags
    #    Uses the sanitized OHLCV (`daily`) which has close_raw after
    #    _sanitize_ohlcv_for_tier2 renamed columns; if not present falls
    #    back to close.  All criteria use only data ≤ t-1 (look-ahead free).
    #    See src/data/screen0.py for implementation details.
    # ------------------------------------------------------------------
    print(f"\n=== Screen 0 Eligibility (universe={universe_choice!r}) ===")
    t_s0 = time.time()
    daily_for_screen = daily.copy()
    daily_for_screen["date"] = pd.to_datetime(daily_for_screen["date"])

    s0_series  = screen0_eligibility(
        daily_for_screen, str(SP500_DIR), universe=universe_choice
    )
    adv_series = trailing_adv_usd(daily_for_screen)

    out["s0_eligible"] = s0_series.reindex(out.index)
    out["in_universe"] = out["s0_eligible"]
    out["adv_usd"]     = adv_series.reindex(out.index)

    n_eligible = int(out["s0_eligible"].sum())
    n_total    = len(out)
    print(f"  Eligible rows: {n_eligible:,} / {n_total:,} "
          f"({100*n_eligible/n_total:.1f}%)  [{time.time()-t_s0:.0f}s]")

    s_delta = survivorship_delta(daily_for_screen, str(SP500_DIR))
    print(f"  Survivorship delta (union vs PIT): {s_delta}")
    try:
        from src.manifest import record as _manifest_record
        _manifest_record(
            "universe.survivorship_delta",
            s_delta,
            stage="screen0",
            universe=universe_choice,
            meta={"adv_window": 21, "min_price_usd": 5.0, "min_adv_usd": 1_000_000},
        )
    except Exception as _e:
        print(f"  [warn] manifest record failed: {_e}")

    # ------------------------------------------------------------------
    # 9. Summary
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
