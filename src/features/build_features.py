"""Week 2: Tier 1 feature engineering pipeline.

Loads daily OHLCV (cached), builds 16 classic alpha features,
computes Track A (FF5+Mom residualized) and Track B (raw) 5-day
forward return targets, and saves the merged matrix.

Output: data/processed/features_tier1.parquet
        data/processed/daily_ohlcv.parquet  (cache)

Run directly:
    cd ~/ML_Paper && python3 src/features/build_features.py
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
from src.features.tier1_classic import build_tier1_features, compute_residualized_returns

CATALOG  = os.environ.get("DUCKDB_CATALOG", "/Volumes/Crucial X9/data/catalog.duckdb")
SP500_DIR = ROOT / "datasets" / "sp500_holdings"
OUT_DIR   = ROOT / "data" / "processed"

UNIVERSE_START = "2010-01-01"
UNIVERSE_END   = "2024-12-31"
FACTOR_WINDOW  = 252   # rolling OLS window (trading days) for beta estimation
FWD_HORIZON    = 5     # 5-day forward return target


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_daily_ohlcv(db: duckdb.DuckDBPyConnection, sp500_tickers: set) -> pd.DataFrame:
    cache = OUT_DIR / "daily_ohlcv.parquet"
    if cache.exists():
        print(f"  [cache] Loading daily OHLCV from {cache}")
        df = pd.read_parquet(cache)
        df["date"] = pd.to_datetime(df["date"])
        return df

    print("  [duckdb] Resampling 1-min → daily (first run, ~1 min)...")
    t0 = time.time()
    ticker_sql = "', '".join(sorted(sp500_tickers))
    query = f"""
        SELECT
            ticker,
            (timestamp AT TIME ZONE 'America/New_York')::DATE  AS date,
            FIRST(open  ORDER BY timestamp)                    AS open,
            MAX(high)                                          AS high,
            MIN(low)                                           AS low,
            LAST(close  ORDER BY timestamp)                    AS close,
            SUM(volume)                                        AS volume
        FROM equities_ohlcv_1m
        WHERE ticker IN ('{ticker_sql}')
          AND (timestamp AT TIME ZONE 'America/New_York')::DATE >= '{UNIVERSE_START}'
          AND (timestamp AT TIME ZONE 'America/New_York')::DATE <  '2025-01-01'
        GROUP BY ticker, (timestamp AT TIME ZONE 'America/New_York')::DATE
        ORDER BY ticker, date
    """
    df = db.execute(query).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"  Saved {cache}  ({time.time()-t0:.0f}s, {len(df):,} rows)")
    return df


def load_factors(db: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    ff5 = db.execute("SELECT * FROM ff_famafrench_ff5_daily").fetchdf()
    ff5["date"] = pd.to_datetime(ff5["date"]).dt.normalize()
    ff5 = ff5.rename(columns={"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml",
                                "RMW": "rmw", "CMA": "cma", "RF": "rf"})
    mom = db.execute("SELECT * FROM ff_famafrench_mom_daily").fetchdf()
    mom["date"] = pd.to_datetime(mom["date"]).dt.normalize()
    mom = mom.rename(columns={"Mom": "mom"})
    factors = ff5.merge(mom[["date", "mom"]], on="date", how="left")
    factors = factors.set_index("date").sort_index() / 100.0   # % → decimal
    return factors


def load_macro(db: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    fred = db.execute("SELECT * FROM fred_macro").fetchdf()
    fred["date"] = pd.to_datetime(fred["date"]).dt.normalize()
    macro = fred.pivot_table(index="date", columns="series_id", values="value", aggfunc="last")
    macro = macro.rename(columns={"VIXCLS": "vix", "T10Y2Y": "term_spread_2s10s"})
    return macro[["vix", "term_spread_2s10s"]].ffill()


# ---------------------------------------------------------------------------
# Target computation
# ---------------------------------------------------------------------------

def compute_track_b(close_wide: pd.DataFrame, horizon: int = FWD_HORIZON) -> pd.DataFrame:
    """Unadjusted h-day forward close-to-close return.

    At date T: (close[T+h] - close[T]) / close[T].
    Features are already lag-1, so this target is leak-free when joined on date.
    """
    return close_wide.pct_change(horizon, fill_method=None).shift(-horizon)


def compute_track_a(
    close_wide: pd.DataFrame,
    factors: pd.DataFrame,
    horizon: int = FWD_HORIZON,
    window: int = FACTOR_WINDOW,
) -> pd.DataFrame:
    """FF5+Mom residualized h-day forward return (idiosyncratic alpha target).

    Steps:
      1. Estimate daily betas via rolling OLS on daily close-to-close returns.
      2. Compound h consecutive daily idiosyncratic returns for the forward window.
    """
    daily_ret = close_wide.pct_change(fill_method=None)

    factor_cols = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
    f = factors[factor_cols].reindex(daily_ret.index, method="ffill").fillna(0.0)

    print("  Computing daily idio returns (rolling OLS)...")
    t0 = time.time()
    daily_idio = compute_residualized_returns(daily_ret, f, window=window)
    print(f"  Done in {time.time()-t0:.0f}s")

    # Compound h-day forward idio return: at T, product of (1+idio[T+1:T+h])
    one_plus = (1.0 + daily_idio.values).astype(float)   # (T, N)
    n_t, n_k = one_plus.shape
    result = np.full((n_t, n_k), np.nan)

    for t in range(n_t - horizon):
        window_fwd = one_plus[t + 1 : t + 1 + horizon]   # (h, N)
        # only compound where we have full h-day non-NaN data
        valid = ~np.isnan(window_fwd).any(axis=0)
        if valid.any():
            result[t, valid] = window_fwd[:, valid].prod(axis=0) - 1.0

    return pd.DataFrame(result, index=daily_idio.index, columns=daily_idio.columns)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "features_tier1.parquet"

    print("=== Week 2: Tier 1 Feature Engineering ===\n")

    db = duckdb.connect(CATALOG, read_only=True)

    # 1. Universe tickers
    sp500_tickers = get_universe_tickers(str(SP500_DIR))
    print(f"SP500 union tickers (2010-2024): {len(sp500_tickers)}")

    # 2. Daily OHLCV
    daily = load_daily_ohlcv(db, sp500_tickers)
    print(f"Daily OHLCV: {daily.shape}  ({daily['ticker'].nunique()} tickers)")

    # 3. Factors + macro
    factors = load_factors(db)
    macro   = load_macro(db)

    # 4. Build Tier 1 features (MultiIndex ticker, date)
    prices_mi = daily.set_index(["ticker", "date"]).sort_index()
    vix    = macro["vix"]
    spread = macro["term_spread_2s10s"]

    print("\nBuilding Tier 1 features...")
    t0 = time.time()
    features = build_tier1_features(prices_mi, factors, vix, spread)
    print(f"  Done in {time.time()-t0:.0f}s  →  {features.shape}")

    # 5. Wide close panel for target computation
    close_wide = daily.pivot(index="date", columns="ticker", values="close")
    close_wide.index = pd.to_datetime(close_wide.index)

    # 6. Track B: raw 5-day forward return
    print("\nComputing Track B (raw 5d forward return)...")
    track_b = compute_track_b(close_wide)           # date × ticker

    # 7. Track A: FF5+Mom residualized 5-day forward return
    print("\nComputing Track A (idiosyncratic 5d forward return)...")
    track_a = compute_track_a(close_wide, factors)  # date × ticker

    # 8. Stack targets to long format
    def _to_long(wide: pd.DataFrame, col: str) -> pd.DataFrame:
        long = wide.stack(future_stack=True).rename(col).reset_index()
        long.columns = ["date", "ticker", col]
        long["date"] = pd.to_datetime(long["date"])
        return long

    track_b_long = _to_long(track_b, "target_track_b")
    track_a_long = _to_long(track_a, "target_track_a")

    # 9. Merge features + targets
    feat_df = features.reset_index()
    out = feat_df.merge(track_b_long, on=["ticker", "date"], how="left")
    out = out.merge(track_a_long,    on=["ticker", "date"], how="left")
    out = out.set_index(["ticker", "date"]).sort_index()

    # 10. Filter to universe window; drop all-NaN feature rows (burn-in)
    feat_cols = [c for c in out.columns if c.startswith("target") is False]
    out = out.loc[
        (out.index.get_level_values("date") >= UNIVERSE_START) &
        (out.index.get_level_values("date") <= UNIVERSE_END)
    ]
    out = out.dropna(subset=feat_cols, how="all")

    # --- Summary ---
    dates = out.index.get_level_values("date")
    print(f"\n=== Output ===")
    print(f"Shape:      {out.shape}")
    print(f"Date range: {dates.min().date()} → {dates.max().date()}")
    print(f"Tickers:    {out.index.get_level_values('ticker').nunique()}")
    print(f"\nFeature NaN rates:")
    print(out[feat_cols].isna().mean().round(3).to_string())
    print(f"\nTarget coverage:")
    print(out[["target_track_b", "target_track_a"]].notna().mean().round(3).to_string())

    out.to_parquet(out_path)
    print(f"\nSaved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    main()
