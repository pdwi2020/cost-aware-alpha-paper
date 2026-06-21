"""Week 2: Tier 1 feature engineering pipeline.

Loads daily OHLCV (cached), builds 16 classic alpha features,
computes Track A (FF5+Mom residualized) and Track B (raw) 5-day
forward return targets, and saves the merged matrix.

SANITIZATION (±50% return cap):
    Corporate-action artifacts (unadjusted splits, bad ticks) in the
    S&P-500 universe produce impossible single-day moves (>±50%).  These
    contaminate every return-derived feature (ret_1d/5d/21d/63d,
    reversal_1w/4w) and both targets.

    Sanitization approach (mirrors run_baselines.py::build_clean_prices):
      1. For each ticker, compute raw daily returns r = close.pct_change()
      2. Clip to ±SANITIZE_CAP  (default 0.50 = ±50%)
      3. Reconstruct close_clean = first_close × cumprod(1 + r_clean)
      4. Replace `close` in the OHLCV panel with close_clean before all
         downstream feature and target computation.

    The original `close` column is preserved for dollar-volume (ADV) so
    that liquidity filtering is not distorted by the reconstruction.
    Pass --no-sanitize to reproduce the original raw behavior.

Output: data/processed/features_tier1.parquet
        data/processed/daily_ohlcv.parquet  (cache)

Run directly:
    cd ~/ML_Paper && python3 src/features/build_features.py
    cd ~/ML_Paper && python3 src/features/build_features.py --no-sanitize
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
from src.data.screen0 import screen0_eligibility, survivorship_delta
from src.features.tier1_classic import build_tier1_features, compute_residualized_returns

CATALOG  = os.environ.get("DUCKDB_CATALOG", "/Volumes/Crucial X9/data/catalog.duckdb")
SP500_DIR = ROOT / "datasets" / "sp500_holdings"
OUT_DIR   = ROOT / "data" / "processed"

UNIVERSE_START = "2010-01-01"
UNIVERSE_END   = "2025-08-01"   # extended to include the locked 2025 OOS window (data max 2025-08-01)
FACTOR_WINDOW  = 252   # rolling OLS window (trading days) for beta estimation
FWD_HORIZON    = 5     # 5-day forward return target

# ±50% daily return cap for corporate-action artifact removal.
# Set to None (or pass --no-sanitize) to reproduce original raw behavior.
SANITIZE_CAP = 0.50


# ---------------------------------------------------------------------------
# Price sanitization helpers
# ---------------------------------------------------------------------------

def build_clean_prices_wide(
    close_w: pd.DataFrame,
    cap: float,
) -> tuple:
    """Reconstruct a clean price series by cumulatively compounding clipped returns.

    Mirrors run_baselines.py::build_clean_prices exactly.

    Parameters
    ----------
    close_w : (date × ticker) raw close prices
    cap     : symmetric absolute cap (e.g. 0.50 = ±50%)

    Returns
    -------
    close_clean : (date × ticker) cleaned price proxy, same shape as close_w
    n_clipped   : int, number of (ticker, day) cells clipped
    """
    r_raw = close_w.pct_change(fill_method=None)
    n_clipped = int((r_raw.abs() > cap).sum().sum())

    r_clean = r_raw.clip(lower=-cap, upper=cap)

    # Reconstruct: start from the first valid close level per ticker,
    # then compound the cleaned returns forward.
    first_close = close_w.iloc[0]                         # Series (ticker → price)
    growth = (1.0 + r_clean.fillna(0.0)).cumprod()        # (date × ticker)
    close_clean = growth.multiply(first_close, axis="columns")
    # Restore NaN where original close was NaN (listing gap, delisting)
    close_clean[close_w.isna()] = np.nan

    return close_clean, n_clipped


def sanitize_ohlcv(daily: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Replace the `close` column in OHLCV with the sanitized price proxy.

    The original `close` is preserved under `close_raw` so that ADV
    computations (which use close × volume) remain on the original scale
    and are not distorted by the price reconstruction.

    Parameters
    ----------
    daily : long-format OHLCV DataFrame with columns [ticker, date, close, ...]
    cap   : return cap (default SANITIZE_CAP = 0.50)

    Returns
    -------
    Sanitized OHLCV DataFrame (close → close_clean; close_raw added)
    """
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    # Wide close panel: date × ticker
    close_w = daily.pivot(index="date", columns="ticker", values="close")

    close_clean_w, n_clipped = build_clean_prices_wide(close_w, cap)
    print(f"\n  [sanitize_ohlcv] Clipped {n_clipped:,} (ticker,day) cells "
          f"with |raw_ret| > {cap:.0%} to ±{cap:.0%}")

    # Long-format clean close
    clean_long = (
        close_clean_w
        .stack(future_stack=True)
        .rename("close_clean")
        .reset_index()
    )
    clean_long.columns = ["date", "ticker", "close_clean"]
    clean_long["date"] = pd.to_datetime(clean_long["date"])

    # Merge back, rename original close → close_raw, set close = close_clean
    daily = daily.merge(clean_long, on=["ticker", "date"], how="left")
    daily.rename(columns={"close": "close_raw"}, inplace=True)
    daily.rename(columns={"close_clean": "close"}, inplace=True)

    print(f"  [sanitize_ohlcv] Sanitized OHLCV shape: {daily.shape}")
    return daily


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
          AND (timestamp AT TIME ZONE 'America/New_York')::DATE <= '{UNIVERSE_END}'
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
    parser = argparse.ArgumentParser(description="Week 2: Tier 1 Feature Engineering")
    parser.add_argument(
        "--no-sanitize", action="store_true",
        help="Disable ±50%% return-cap sanitization (reproduce original raw behavior)"
    )
    parser.add_argument(
        "--universe", choices=["pit", "liquidity"], default="pit",
        help="Screen 0 universe: 'pit' (point-in-time S&P 500 annual, default) "
             "or 'liquidity' (top-N by trailing ADV, fully reconstructible)."
    )
    args = parser.parse_args()
    sanitize = not args.no_sanitize
    universe_choice = args.universe
    cap = SANITIZE_CAP if sanitize else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "features_tier1.parquet"

    if sanitize:
        print(f"=== Week 2: Tier 1 Feature Engineering  [SANITIZE_CAP=±{cap:.0%}] ===\n")
    else:
        print("=== Week 2: Tier 1 Feature Engineering  [RAW — no sanitization] ===\n")

    db = duckdb.connect(CATALOG, read_only=True)

    # 1. Universe tickers
    sp500_tickers = get_universe_tickers(str(SP500_DIR))
    print(f"SP500 union tickers (2010-2024): {len(sp500_tickers)}")

    # 2. Daily OHLCV
    daily = load_daily_ohlcv(db, sp500_tickers)
    print(f"Daily OHLCV: {daily.shape}  ({daily['ticker'].nunique()} tickers)")

    # 3. Sanitize prices BEFORE any feature computation
    #    This replaces the `close` column with close_clean (cumproduct of clipped returns).
    #    tier1_classic.py::build_tier1_features() and compute_track_b/a() both read
    #    `close` from the OHLCV, so a single replacement here cleans all downstream
    #    return-derived features AND both targets coherently.
    if sanitize:
        daily = sanitize_ohlcv(daily, cap)
        # Count residual artifacts for verification
        close_w_chk = daily.pivot(index="date", columns="ticker", values="close")
        r_chk = close_w_chk.pct_change(fill_method=None)
        n_residual = int((r_chk.abs() > cap).sum().sum())
        print(f"  [verify] Residual |ret|>{cap:.0%} cells after sanitization: {n_residual}")
        if n_residual > 0:
            print(f"  [WARN] Non-zero residuals — check for extreme close[t=0] values or NaN gaps")
    else:
        print("  Price sanitization DISABLED (raw mode)")

    # 4. Factors + macro
    factors = load_factors(db)
    macro   = load_macro(db)

    # 5. Build Tier 1 features (MultiIndex ticker, date)
    #    Uses sanitized `close` (or raw, if --no-sanitize)
    prices_mi = daily.set_index(["ticker", "date"]).sort_index()
    vix    = macro["vix"]
    spread = macro["term_spread_2s10s"]

    print("\nBuilding Tier 1 features...")
    t0 = time.time()
    features = build_tier1_features(prices_mi, factors, vix, spread)
    print(f"  Done in {time.time()-t0:.0f}s  →  {features.shape}")

    # 6. Wide close panel for target computation (uses sanitized close)
    close_wide = daily.pivot(index="date", columns="ticker", values="close")
    close_wide.index = pd.to_datetime(close_wide.index)

    # 7. Track B: 5-day forward return (from sanitized close)
    print("\nComputing Track B (5d forward return, sanitized close)...")
    track_b = compute_track_b(close_wide)           # date × ticker

    # 8. Track A: FF5+Mom residualized 5-day forward return (from sanitized close)
    print("\nComputing Track A (idiosyncratic 5d forward return, sanitized close)...")
    track_a = compute_track_a(close_wide, factors)  # date × ticker

    # 9. Stack targets to long format
    def _to_long(wide: pd.DataFrame, col: str) -> pd.DataFrame:
        long = wide.stack(future_stack=True).rename(col).reset_index()
        long.columns = ["date", "ticker", col]
        long["date"] = pd.to_datetime(long["date"])
        return long

    track_b_long = _to_long(track_b, "target_track_b")
    track_a_long = _to_long(track_a, "target_track_a")

    # 10. Merge features + targets
    feat_df = features.reset_index()
    out = feat_df.merge(track_b_long, on=["ticker", "date"], how="left")
    out = out.merge(track_a_long,    on=["ticker", "date"], how="left")
    out = out.set_index(["ticker", "date"]).sort_index()

    # 11. Filter to universe window; drop all-NaN feature rows (burn-in)
    feat_cols = [c for c in out.columns if c.startswith("target") is False]
    out = out.loc[
        (out.index.get_level_values("date") >= UNIVERSE_START) &
        (out.index.get_level_values("date") <= UNIVERSE_END)
    ]
    out = out.dropna(subset=feat_cols, how="all")

    # --- Screen 0: PIT membership + lagged price/ADV eligibility flags ---
    # Compute eligibility from the daily OHLCV panel (which has close_raw after
    # sanitization, or close if --no-sanitize).  All three criteria use only
    # data ≤ t-1; see src/data/screen0.py for details.
    print(f"\n=== Screen 0 Eligibility (universe={universe_choice!r}) ===")
    t_s0 = time.time()
    # Rebuild the daily panel subset that matches the filtered `out` index
    # (use `daily` which has close_raw from sanitize_ohlcv, or close if raw)
    daily_for_screen = daily.copy()
    daily_for_screen["date"] = pd.to_datetime(daily_for_screen["date"])

    s0_series = screen0_eligibility(
        daily_for_screen, str(SP500_DIR), universe=universe_choice
    )
    # trailing ADV (lagged) — reuse the same window for the column we attach
    from src.data.screen0 import trailing_adv_usd as _adv_fn
    adv_series = _adv_fn(daily_for_screen)

    # Attach columns to `out` (align on the (ticker, date) MultiIndex)
    out["s0_eligible"] = s0_series.reindex(out.index)
    out["in_universe"] = out["s0_eligible"]  # alias for clarity; same value
    out["adv_usd"]     = adv_series.reindex(out.index)

    n_eligible = int(out["s0_eligible"].sum())
    n_total    = len(out)
    print(f"  Eligible rows: {n_eligible:,} / {n_total:,} "
          f"({100*n_eligible/n_total:.1f}%)  [{time.time()-t_s0:.0f}s]")

    # Survivorship-bias delta: quantify what the union→PIT switch removes
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

    # --- Artifact count report (before = raw bak, after = this run) ---
    ret_feats_chk = ['ret_1d','ret_5d','ret_21d','ret_63d','reversal_1w','reversal_4w']
    tgt_chk = ['target_track_a','target_track_b']
    print(f"\n=== Sanitization Verification ===")
    print(f"{'Feature':<25s}  {'|val|>50% count':>17s}")
    for c in ret_feats_chk + tgt_chk:
        if c in out.columns:
            n = int((out[c].abs() > 0.50).sum())
            print(f"  {c:<23s}  {n:>15,}")

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
