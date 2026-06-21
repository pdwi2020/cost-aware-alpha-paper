"""Week 3: Full feature pipeline — Tier 1 + Tier 2 merged.

Loads Tier 1 from parquet, computes Tier 2 features (intraday microstructure,
cross-asset macro, crowding proxies), adds the three prespecified interactions
(beta_x_vix, beta_x_term_spread, credit_beta_x_credit), merges, and saves
features_all.parquet.

Interaction formulas (all look-ahead-free):
  market_beta_lagged  = rolling_cov(r_i, r_mkt) / rolling_var(r_mkt) over 252d,
                        then shifted 1 day per ticker (uses only t-1 returns).
  credit_beta_lagged  = rolling_cov(r_i, d_credit) / rolling_var(d_credit) over 252d,
                        then shifted 1 day per ticker.
  beta_x_vix          = market_beta_lagged * vix           (vix already lag-1 in frame)
  beta_x_term_spread  = market_beta_lagged * term_spread_chg_21d  (already lag-1)
  credit_beta_x_credit = credit_beta_lagged * credit_proxy_chg_5d (already lag-1)

Broadcast-only columns (vix, term_spread, etc.) are DROPPED from the final feature
set via feature_spec.feature_columns(), but vix and term_spread are retained as
regime labels (regime_vix / regime_term_spread).

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
from src.features.feature_spec import feature_columns as _feature_columns

# ±50% daily return cap for corporate-action artifact removal in Tier-2 features.
# Must match build_features.py SANITIZE_CAP for full coherence.
SANITIZE_CAP_T2 = 0.50

BETA_WINDOW = 252   # rolling OLS window for market / credit beta estimation


def _compute_rolling_beta(
    stock_ret_wide: pd.DataFrame,
    factor_series: pd.Series,
    window: int = BETA_WINDOW,
) -> pd.DataFrame:
    """Rolling OLS slope of each stock return on a single factor series.

    Uses the formula: beta(t) = rolling_cov(r_i, f) / rolling_var(f)
    computed over the `window` trailing days ending at t (inclusive).

    The result is then .shift(1) per ticker so that beta used on date t
    reflects only returns through t-1 — no look-ahead.

    Parameters
    ----------
    stock_ret_wide : pd.DataFrame, shape (dates, tickers)
        Daily stock returns, wide format.
    factor_series  : pd.Series, shape (dates,)
        Daily factor values (e.g. mkt_rf, daily credit-proxy change).
        Must share the same index as stock_ret_wide.
    window         : int
        Rolling window length in trading days.

    Returns
    -------
    pd.DataFrame, same shape as stock_ret_wide
        Rolling betas, shifted 1 day so position [t] uses data through t-1.
    """
    # Align factor to stock index
    f = factor_series.reindex(stock_ret_wide.index).ffill()

    # Rolling covariance of each stock with the factor
    # pandas rolling().cov(other) computes cov(self, other)
    rolling_cov = stock_ret_wide.rolling(window).cov(f)      # (dates, tickers)
    rolling_var = f.rolling(window).var()                     # (dates,)

    # beta = cov / var; broadcast var along ticker axis
    beta_wide = rolling_cov.div(rolling_var, axis=0)

    # Shift 1 day: beta at t now uses returns through t-1 only (no look-ahead).
    # beta_wide is (dates × tickers) with a DatetimeIndex shared by all tickers,
    # so a single .shift(1) uniformly lags every ticker by one date row.
    beta_lagged = beta_wide.shift(1)

    return beta_lagged


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
    # 6. Prespecified interactions (look-ahead-free)
    #
    # Three stock-level macro interaction features replace the old
    # term_spread_x_mom interaction.  Each multiplies a STOCK-SPECIFIC
    # rolling beta (lagged 1 day) by a broadcast macro level that is
    # already lag-1 in the feature frame (tier1 broadcast columns were
    # already .shift(1) in tier1_classic.py::build_tier1_features).
    #
    # Betas are computed from the sanitized daily OHLCV (same prices used
    # for all other return-derived features).
    #
    # market_beta_lagged = rolling_cov(r_i, r_mkt) / rolling_var(r_mkt),
    #                      shifted 1 day (uses returns ≤ t-1 only).
    # credit_beta_lagged = rolling_cov(r_i, d_credit) / rolling_var(d_credit),
    #                      shifted 1 day (uses returns ≤ t-1 only).
    # ------------------------------------------------------------------
    print("\n  Computing prespecified interaction features...")

    # Build wide daily return panel (sanitized close from daily_ohlcv)
    close_wide_beta = daily.pivot(index="date", columns="ticker", values="close")
    close_wide_beta.index = pd.to_datetime(close_wide_beta.index)
    stock_ret_wide = close_wide_beta.pct_change(fill_method=None)

    # ------------------------------------------------------------------
    # 6a. Market beta — factor = mkt_rf (Fama-French market excess return)
    # ------------------------------------------------------------------
    # Load factors from DuckDB for mkt_rf
    try:
        mkt_rf_series = None
        # Attempt to read from DuckDB (already open in the scope above as `db`)
        ff5_df = db.execute("SELECT * FROM ff_famafrench_ff5_daily").fetchdf()
        ff5_df["date"] = pd.to_datetime(ff5_df["date"]).dt.normalize()
        mkt_rf_col = "Mkt-RF" if "Mkt-RF" in ff5_df.columns else "mkt_rf"
        if mkt_rf_col in ff5_df.columns:
            mkt_rf_series = (
                ff5_df.set_index("date")[mkt_rf_col] / 100.0
            ).reindex(stock_ret_wide.index).ffill()
            print(f"    mkt_rf loaded: {mkt_rf_series.notna().sum()} valid days")
    except Exception as _e:
        print(f"    [warn] Could not load mkt_rf: {_e}; skipping market-beta interactions")
        mkt_rf_series = None

    if mkt_rf_series is not None:
        market_beta_lagged_wide = _compute_rolling_beta(
            stock_ret_wide, mkt_rf_series, window=BETA_WINDOW
        )
        # Long-format: stack (date, ticker) → Series
        mbl_long = (
            market_beta_lagged_wide.stack(future_stack=True)
            .rename("_market_beta_lagged")
            .reset_index()
        )
        mbl_long.columns = ["date", "ticker", "_market_beta_lagged"]
        mbl_long["date"] = pd.to_datetime(mbl_long["date"])
        out = out.reset_index().merge(mbl_long, on=["ticker", "date"], how="left")
        out = out.set_index(["ticker", "date"]).sort_index()

        # beta_x_vix: market_beta_lagged * vix (vix already lag-1 in frame)
        if "vix" in out.columns:
            out["beta_x_vix"] = out["_market_beta_lagged"] * out["vix"]
            print("    Added interaction: beta_x_vix = market_beta_lagged × vix")

        # beta_x_term_spread: market_beta_lagged * term_spread_chg_21d (already lag-1)
        if "term_spread_chg_21d" in out.columns:
            out["beta_x_term_spread"] = out["_market_beta_lagged"] * out["term_spread_chg_21d"]
            print("    Added interaction: beta_x_term_spread = market_beta_lagged × term_spread_chg_21d")

        # Drop the helper column
        out.drop(columns=["_market_beta_lagged"], inplace=True)

    # ------------------------------------------------------------------
    # 6b. Credit beta — factor = daily change in credit_proxy
    # ------------------------------------------------------------------
    credit_beta_ok = False
    if "credit_proxy" in out.columns:
        # Reconstruct the daily credit-proxy level (date-indexed, before lag-1 was
        # applied in tier2_extended).  The tier2 macro pipeline already created
        # credit_proxy = DGS10 - DFF; we can recover it from the feature frame by
        # un-shifting (the original unshifted series is the same per date for all
        # tickers, so we take the per-date median to undo NaN from edge tickers).
        # More robustly: re-derive from DuckDB directly.
        try:
            fred_df = db.execute("SELECT * FROM fred_macro").fetchdf()
            fred_df["date"] = pd.to_datetime(fred_df["date"]).dt.normalize()
            fred_pivot = fred_df.pivot_table(
                index="date", columns="series_id", values="value", aggfunc="last"
            ).ffill()
            if "DGS10" in fred_pivot.columns and "DFF" in fred_pivot.columns:
                credit_proxy_raw = (fred_pivot["DGS10"] - fred_pivot["DFF"]).reindex(
                    stock_ret_wide.index
                ).ffill()
                # Daily change in credit proxy (used as the factor)
                d_credit = credit_proxy_raw.diff(1)

                credit_beta_lagged_wide = _compute_rolling_beta(
                    stock_ret_wide, d_credit, window=BETA_WINDOW
                )
                cbl_long = (
                    credit_beta_lagged_wide.stack(future_stack=True)
                    .rename("_credit_beta_lagged")
                    .reset_index()
                )
                cbl_long.columns = ["date", "ticker", "_credit_beta_lagged"]
                cbl_long["date"] = pd.to_datetime(cbl_long["date"])
                out = out.reset_index().merge(cbl_long, on=["ticker", "date"], how="left")
                out = out.set_index(["ticker", "date"]).sort_index()

                # credit_beta_x_credit: credit_beta_lagged * credit_proxy_chg_5d (already lag-1)
                if "credit_proxy_chg_5d" in out.columns:
                    out["credit_beta_x_credit"] = (
                        out["_credit_beta_lagged"] * out["credit_proxy_chg_5d"]
                    )
                    print("    Added interaction: credit_beta_x_credit = credit_beta_lagged × credit_proxy_chg_5d")
                    credit_beta_ok = True

                out.drop(columns=["_credit_beta_lagged"], inplace=True)
        except Exception as _e:
            print(f"    [warn] Could not compute credit_beta: {_e}")

    if not credit_beta_ok:
        print("    [warn] credit_beta_x_credit not added (credit_proxy or DuckDB unavailable)")

    # ------------------------------------------------------------------
    # 6c. Retain broadcast columns as regime labels (renamed), then drop
    #     from the feature set via feature_spec.feature_columns() below.
    # ------------------------------------------------------------------
    if "vix" in out.columns:
        out["regime_vix"] = out["vix"]
    if "term_spread" in out.columns:
        out["regime_term_spread"] = out["term_spread"]
    print("  Retained regime labels: regime_vix, regime_term_spread")

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
    # 7b. Derive the canonical final feature set via feature_spec.
    #     DROPPED columns (broadcast-only, duplicates, old interaction)
    #     are excluded from the signal set but regime labels and universe
    #     flags are retained separately.
    # ------------------------------------------------------------------
    final_feat_list = _feature_columns(out)
    _non_feat_cols = set(target_cols) | {"s0_eligible", "in_universe", "adv_usd",
                                          "regime_vix", "regime_term_spread"}
    print(f"\n  Final pre-registered feature set ({len(final_feat_list)} features):")
    for _f in final_feat_list:
        print(f"    {_f}")
    print(f"  Non-feature cols retained: {sorted(_non_feat_cols & set(out.columns))}")

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
        from src.features.feature_spec import DROPPED as _SPEC_DROPPED, ADDED_INTERACTIONS as _ADDED
        _manifest_record(
            "features.final_list",
            final_feat_list,
            stage="features",
            meta={
                "n_features": len(final_feat_list),
                "n_kept": len([f for f in final_feat_list if f not in _ADDED]),
                "n_interactions": len([f for f in final_feat_list if f in _ADDED]),
                "dropped": sorted(_SPEC_DROPPED),
            },
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
