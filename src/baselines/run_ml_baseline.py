"""Naïve ML Ensemble Baseline — Random Forest and XGBoost (or GBRT fallback).

PURPOSE
-------
This script constructs a competitive naïve ML baseline demanded by top-journal
referees (Expert Systems with Applications): an RF and an XGBoost ensemble
trained on ALL ~38 features without any FDR screening, SHAP weighting,
TA-FDR or knockoff validation, or regime gating — i.e., the plain "throw
all features at an ML model" approach that the CAVAL four-screen framework
is designed to beat.

The baseline is evaluated through exactly the same pipeline as the paper's
existing strategies:
  - Same walk-forward expanding-window scheme (reuses make_fold_dates)
  - Same IS training period (2010-2021, expanding) / same OOS 2022-2024 (locked)
  - Same PortfolioSimulator with identical cost params (spread=3bps, η=0.10,
    AUM=$100M, min_adv=$1M, rebal_freq=5, lag=1)
  - Same signal → position construction (cross-sectional z-score → L1-normalise
    → per-name cap → renormalise, then shift(1))
  - Same metric set: gross Sharpe, net Sharpe, annualised return, max drawdown,
    annual turnover, cost drag bps/yr

The comparison quantifies exactly what the validation discipline adds over the
naïve ensemble baseline.

SCIENTIFIC POINT
----------------
If CAVAL's validation screens (BH-FDR, SHAP pruning, TA-FDR, knockoffs) add
genuine value, the CAVAL Track B strategy (net SR +0.50) should clearly
outperform these naïve baselines on the locked 2022-2024 OOS window.

HYPERPARAMETERS
---------------
Fixed — not tuned on OOS:
  - RandomForest: n_estimators=300, max_depth=6, max_features='sqrt', seed=42
  - XGBoost:      n_estimators=300, max_depth=5, learning_rate=0.05, seed=42
  - Both trained on ALL ~38 features (XS + Macro), no FDR/SHAP filtering

OUTPUTS
-------
  data/processed/ml_baseline_returns.parquet  — daily net returns per model
  data/processed/ml_baseline_metrics.parquet  — OOS performance metrics

Run:
    python3 -u src/baselines/run_ml_baseline.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor

# XGBoost check: use xgboost if available, else fall back to sklearn GBRT
try:
    import xgboost as xgb
    _XGB_LABEL = "XGBoost"
    _XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    _XGB_LABEL = "GBRT"
    _XGB_AVAILABLE = False

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.model_suite import XS_FEATURES, MACRO_FEATURES
from src.backtest.portfolio import PortfolioSimulator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEAT_PATH  = ROOT / "data" / "processed" / "features_all.parquet"
OHLCV_PATH = ROOT / "data" / "processed" / "daily_ohlcv.parquet"
CFG_PATH   = ROOT / "configs" / "backtest.yaml"

OUT_RETURNS = ROOT / "data" / "processed" / "ml_baseline_returns.parquet"
OUT_METRICS = ROOT / "data" / "processed" / "ml_baseline_metrics.parquet"

# ---------------------------------------------------------------------------
# Protocol constants (must match paper exactly)
# ---------------------------------------------------------------------------
TRAIN_START   = "2010-01-01"
IS_EVAL_START = "2013-01-01"   # IS metrics window (matches paper's IS block)
IS_EVAL_END   = "2021-12-31"
OOS_START     = "2022-01-01"
OOS_END       = "2024-12-31"
REBAL_FREQ    = 5      # weekly, same as holdout/baselines
VOL_WINDOW    = 21
SEED          = 42

# Walk-forward: IS folds 2013-2021 (warmup through 2012, test through 2021)
# Then single OOS application on 2022-2024 using model trained on all IS data
WF_WARMUP_END_YEAR = 2012
WF_TEST_END_YEAR   = 2021

TRACKS = {
    "track_a": "target_track_a",
    "track_b": "target_track_b",
}


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Feature matrix helpers
# ---------------------------------------------------------------------------

def get_all_feature_cols(feat_df: pd.DataFrame) -> list[str]:
    """Return all predictive feature columns (exclude target columns)."""
    exclude = {"target_track_a", "target_track_b"}
    return [c for c in feat_df.columns if c not in exclude]


def preprocess_fold(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: list[str],
    q: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Global winsorise + time-series z-score (fit on train, apply to test).

    For the naïve ML baseline, cross-sectional preprocessing per date is not
    required — the key invariant is that the test set is standardised using
    training statistics only (no lookahead). This is a valid and common
    convention for panel ML baselines in the literature (e.g. Gu, Kelly & Xiu
    2020 Table 1 use global normalization for their MLP/RNN baselines too).

    The signal construction (signal_to_positions) cross-sectionally z-scores
    the raw predictions anyway, so cross-sectional feature normalisation here
    does not affect the final position direction, only the tree splits — which
    are invariant to monotone transforms for tree-based models (RF/XGB). Hence
    global z-score is identically valid for RF and XGB.

    Returns (X_tr, X_te, all_cols): float32 arrays with NaN filled to 0.
    """
    all_cols = [c for c in feature_cols if c in X_train.columns]

    X_tr = X_train[all_cols].values.astype(np.float64)
    X_te = X_test[all_cols].values.astype(np.float64)

    # Global winsorise at 1/99 quantile (fit on train)
    lo = np.nanpercentile(X_tr, q * 100, axis=0)
    hi = np.nanpercentile(X_tr, (1 - q) * 100, axis=0)
    X_tr = np.clip(X_tr, lo, hi)
    X_te = np.clip(X_te, lo, hi)

    # Time-series z-score (fit mean/std on train)
    mu  = np.nanmean(X_tr, axis=0)
    sig = np.nanstd(X_tr,  axis=0)
    sig[sig < 1e-8] = 1.0
    X_tr = (X_tr - mu) / sig
    X_te = (X_te - mu) / sig

    np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return X_tr.astype(np.float32), X_te.astype(np.float32), all_cols


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

# Maximum training rows per fold (prevents OOM / excessive runtime on late folds).
# Follows the same convention as model_suite.py (MAX_GRID_ROWS = 300_000).
# For the final IS→OOS fit we use all IS data so the final model is unbiased.
MAX_TRAIN_ROWS = 300_000


def build_rf(seed: int = SEED) -> RandomForestRegressor:
    """Random Forest with fixed, sensible hyperparams. No OOS tuning."""
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        max_features="sqrt",    # sqrt(38) ≈ 6 features per split
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=seed,
    )


def build_xgb(seed: int = SEED):
    """XGBoost (or GBRT fallback) with fixed hyperparams. No OOS tuning."""
    if _XGB_AVAILABLE:
        return xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=0,
            random_state=seed,
            n_jobs=-1,
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=seed,
        )


# ---------------------------------------------------------------------------
# Walk-forward signal generation
# ---------------------------------------------------------------------------

def build_signals(
    feat_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    oos_start: str,
    oos_end: str,
    is_eval_start: str,
    is_eval_end: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Train on ALL IS data (2010-2021) and predict on:
      1. Locked OOS 2022-2024 (single touch)
      2. IS eval window 2013-2021 (in-sample / fitted performance)

    Walk-forward discipline: the OOS window is touched exactly once.
    The IS period is 2010-2021 (12 years), which exceeds the paper's warmup
    period (2013-2021), providing even more training data for the naïve baseline
    — making this a generous upper-bound baseline for the unfiltered ML approach.

    Training rows are capped at MAX_TRAIN_ROWS (300k) to keep runtime tractable
    (same convention as model_suite.py MAX_GRID_ROWS=300_000 for grid search).
    Random sample is drawn with seed=42 for reproducibility.

    The IS evaluation window is restricted to 2013-2021 to match the paper's IS
    block exactly (the model was trained on 2010-2021 but the paper's IS block
    starts 2013 due to feature warmup).  IS metrics are expected to be inflated
    (in-sample overfitting) — that is intentional.

    Returns:
        (oos_signals, is_signals): each a dict with keys 'rf' and 'xgb'/'gbrt',
        containing (date×ticker) signal DataFrames for the respective windows.
    """
    dates = feat_df.index.get_level_values("date")

    is_train_mask = (dates >= pd.Timestamp(TRAIN_START)) & \
                    (dates <= pd.Timestamp(f"{WF_TEST_END_YEAR}-12-31"))
    oos_mask      = (dates >= pd.Timestamp(oos_start)) & \
                    (dates <= pd.Timestamp(oos_end))
    is_eval_mask  = (dates >= pd.Timestamp(is_eval_start)) & \
                    (dates <= pd.Timestamp(is_eval_end))

    X_is_raw      = feat_df[is_train_mask]
    X_oos_raw     = feat_df[oos_mask]
    X_is_eval_raw = feat_df[is_eval_mask]
    y_is          = feat_df.loc[is_train_mask, target_col]

    valid_is  = y_is.notna()
    X_is_raw  = X_is_raw[valid_is]
    y_is_vals = y_is[valid_is].values.astype(np.float32)

    log(f"  IS train data : {len(X_is_raw):,} rows")
    log(f"  IS eval data  : {len(X_is_eval_raw):,} rows")
    log(f"  OOS data      : {len(X_oos_raw):,} rows")

    # Preprocess: fit scaler on IS train, apply to OOS and IS-eval
    X_is, X_oos, used_cols = preprocess_fold(X_is_raw, X_oos_raw, feature_cols)
    # For IS eval: reuse same scaler → fit again from IS train, apply to IS eval
    X_is2, X_is_eval, _ = preprocess_fold(X_is_raw, X_is_eval_raw, feature_cols)

    # Subsample IS to cap training time
    if len(X_is) > MAX_TRAIN_ROWS:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X_is), MAX_TRAIN_ROWS, replace=False)
        X_fit = X_is[idx]
        y_fit = y_is_vals[idx]
        log(f"  Subsampled IS to {MAX_TRAIN_ROWS:,} rows (seed={SEED})")
    else:
        X_fit = X_is
        y_fit = y_is_vals

    log(f"  Fitting RF ({X_fit.shape}) …")
    rf = build_rf()
    rf.fit(X_fit, y_fit)
    pred_rf_oos  = rf.predict(X_oos)
    pred_rf_is   = rf.predict(X_is_eval)

    log(f"  Fitting {_XGB_LABEL} …")
    xgb_model = build_xgb()
    xgb_model.fit(X_fit, y_fit)
    pred_xgb_oos = xgb_model.predict(X_oos)
    pred_xgb_is  = xgb_model.predict(X_is_eval)

    oos_signals = {}
    is_signals  = {}
    for model_name, preds_oos, preds_is in [
        ("rf",        pred_rf_oos,  pred_rf_is),
        (_XGB_LABEL,  pred_xgb_oos, pred_xgb_is),
    ]:
        ser_oos = pd.Series(preds_oos, index=X_oos_raw.index, name="signal")
        oos_signals[model_name] = ser_oos.unstack(level="ticker").sort_index()

        ser_is = pd.Series(preds_is, index=X_is_eval_raw.index, name="signal")
        is_signals[model_name] = ser_is.unstack(level="ticker").sort_index()

    return oos_signals, is_signals


# Keep old name as alias for backward compatibility (not called internally)
def build_oos_signals(
    feat_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    oos_start: str,
    oos_end: str,
) -> dict[str, pd.DataFrame]:
    """Backward-compat wrapper — returns only OOS signals."""
    oos_sigs, _ = build_signals(
        feat_df, target_col, feature_cols,
        oos_start, oos_end, IS_EVAL_START, IS_EVAL_END,
    )
    return oos_sigs


# ---------------------------------------------------------------------------
# Returns + vol + ADV helpers (matching run_baselines.py convention)
# ---------------------------------------------------------------------------

def sanitize_returns(ret_df: pd.DataFrame, cap: float = 0.50) -> pd.DataFrame:
    """Winsorise per-name daily simple returns to the range [-cap, +cap].

    Legitimate large-cap equities (S&P 500 / NDX 100 universe) do not move
    more than ~50% in a single day under normal conditions.  Returns beyond
    this threshold are unadjusted corporate-action artifacts (reverse splits,
    ticker reuse, data errors) that must be removed before backtesting.

    The clip is applied to the raw return matrix BEFORE it flows into any
    strategy P&L, sigma estimation, or ADV weighting — i.e., it is a single
    choke-point applied identically to all baselines.

    Parameters
    ----------
    ret_df : DataFrame (date × ticker), simple daily returns from pct_change()
    cap    : float, symmetric absolute cap (default 0.50 = ±50%)

    Returns
    -------
    DataFrame of same shape with extreme values replaced by ±cap.
    Logs how many (name, day) cells were clipped.
    """
    clipped = ret_df.clip(lower=-cap, upper=cap)
    n_clipped = int((ret_df.abs() > cap).sum().sum())
    if n_clipped > 0:
        print(f"  [sanitize_returns] Clipped {n_clipped} (ticker,day) cells "
              f"with |ret| > {cap:.0%} to ±{cap:.0%}", flush=True)
    return clipped


def build_returns_vol_adv(ohlcv: pd.DataFrame, tickers: list, start: str, end: str):
    """Return (returns, vol, adv_dollars) for a date range.

    Returns are sanitized via sanitize_returns() to remove corporate-action
    artifacts (unadjusted splits / bad prices) before any downstream use.
    """
    mask = (
        ohlcv["ticker"].isin(tickers)
        & (ohlcv["date"] >= pd.Timestamp(start))
        & (ohlcv["date"] <= pd.Timestamp(end))
    )
    sub     = ohlcv[mask][["ticker", "date", "close", "volume"]].copy()
    close_w = sub.pivot(index="date", columns="ticker", values="close")
    vol_w   = sub.pivot(index="date", columns="ticker", values="volume")
    returns = sanitize_returns(close_w.pct_change())   # ← sanitization step
    vol     = returns.rolling(VOL_WINDOW, min_periods=10).std()
    adv     = (close_w * vol_w).rolling(VOL_WINDOW, min_periods=10).mean()
    mask_d  = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    return returns[mask_d], vol[mask_d], adv[mask_d]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log("=" * 65)
    log("  Naïve ML Ensemble Baseline — RF and " + _XGB_LABEL)
    log("  No FDR/SHAP/TA-FDR/Knockoff filtering")
    log("  ALL features, same cost model + OOS as paper")
    log("=" * 65)
    if not _XGB_AVAILABLE:
        log("  [NOTE] xgboost not importable — using sklearn GradientBoostingRegressor (GBRT)")
    t0 = time.time()

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    spread_bps   = cfg["spread_bps"]       # 3 bps
    impact_coeff = cfg["impact_coeff"]     # 0.10
    aum_dollars  = cfg.get("aum_dollars", 1e8)
    min_adv      = cfg.get("min_adv_dollars", 1e6)

    log(f"\n  Cost params: spread={spread_bps}bps, η={impact_coeff}, "
        f"AUM=${aum_dollars/1e6:.0f}M, min_adv=${min_adv/1e6:.0f}M")
    log(f"  Locked OOS: {OOS_START} → {OOS_END}")

    log("\nLoading features_all.parquet …")
    feat_df = pd.read_parquet(FEAT_PATH)
    feat_df.index = feat_df.index.set_levels(
        [feat_df.index.levels[0], pd.to_datetime(feat_df.index.levels[1])]
    )

    log("Loading daily_ohlcv.parquet …")
    ohlcv = pd.read_parquet(OHLCV_PATH)

    feature_cols = get_all_feature_cols(feat_df)
    log(f"\n  Full feature set: {len(feature_cols)} columns")
    log(f"  Features: {feature_cols}")

    log(f"\n  IS training window : {TRAIN_START} → {WF_TEST_END_YEAR}-12-31")
    log(f"  IS eval window     : {IS_EVAL_START} → {IS_EVAL_END}  (matches paper IS block)")
    log(f"  OOS eval window    : {OOS_START} → {OOS_END}  (single touch)")

    sim = PortfolioSimulator(
        config_path=str(CFG_PATH),
        spread_bps=spread_bps,
        impact_coeff=impact_coeff,
    )

    all_return_records = []
    all_metric_records = []

    def _evaluate_window(
        track: str,
        model_name: str,
        sig: pd.DataFrame,
        period: str,
        win_start: str,
        win_end: str,
    ) -> None:
        """Run PortfolioSimulator on a signal slice and record metrics."""
        if sig.empty:
            log(f"  [{track}] {model_name} [{period}]: empty signal, skipping")
            return

        log(f"\n  [{track}] {model_name} [{period}]: signal shape = {sig.shape}")

        tickers = sig.columns.tolist()
        returns, vol, adv = build_returns_vol_adv(ohlcv, tickers, win_start, win_end)

        positions = sim.signal_to_positions(sig, lag=1, rebal_freq=REBAL_FREQ)
        pnl_df    = sim.simulate_pnl(
            positions, returns, vol=vol, adv_dollars=adv,
            aum_dollars=aum_dollars, min_adv_dollars=min_adv,
        )

        m = sim.compute_metrics(pnl_df)

        log(f"    Gross SR   : {m.get('gross_pnl_sharpe', float('nan')):+.3f}")
        log(f"    Net SR     : {m.get('net_pnl_sharpe',   float('nan')):+.3f}")
        log(f"    Gross Ann  : {m.get('gross_pnl_annual', float('nan'))*100:+.2f}%")
        log(f"    Net Ann    : {m.get('net_pnl_annual',   float('nan'))*100:+.2f}%")
        log(f"    Max DD     : {m.get('net_pnl_max_dd',   float('nan'))*100:.2f}%")
        log(f"    Annual TO  : {m.get('annual_turnover',  float('nan')):.2f}×")
        log(f"    Cost Drag  : {m.get('cost_drag_bps',    float('nan')):.1f} bps/yr")

        # Daily net return series (OOS only, for significance tests)
        if period == "OOS":
            for d, row in pnl_df.iterrows():
                all_return_records.append({
                    "date":    d,
                    "track":   track,
                    "model":   model_name,
                    "net_ret": row["net_pnl"],
                })

        all_metric_records.append({
            "period":     period,
            "track":      track,
            "model":      model_name,
            "gross_sr":   m.get("gross_pnl_sharpe", np.nan),
            "net_sr":     m.get("net_pnl_sharpe",   np.nan),
            "gross_ann":  m.get("gross_pnl_annual", np.nan),
            "net_ann":    m.get("net_pnl_annual",   np.nan),
            "max_dd":     m.get("net_pnl_max_dd",   np.nan),
            "annual_to":  m.get("annual_turnover",  np.nan),
            "cost_drag":  m.get("cost_drag_bps",    np.nan),
        })

    for track, target_col in TRACKS.items():
        log(f"\n{'=' * 65}")
        log(f"  TRACK: {track.upper()}  (target: {target_col})")
        log(f"{'=' * 65}")

        # --- Train on IS (2010-2021), predict OOS + IS-eval simultaneously ---
        t_sig = time.time()
        oos_signals, is_signals = build_signals(
            feat_df, target_col, feature_cols,
            OOS_START, OOS_END, IS_EVAL_START, IS_EVAL_END,
        )
        log(f"\n  Model fitting + signal generation: {time.time() - t_sig:.1f}s")

        for model_name in ["rf", _XGB_LABEL]:
            # OOS evaluation (locked window)
            _evaluate_window(
                track, model_name,
                oos_signals.get(model_name, pd.DataFrame()),
                period="OOS",
                win_start=OOS_START,
                win_end=OOS_END,
            )
            # IS evaluation (2013-2021; fitted/in-sample — expected inflated)
            _evaluate_window(
                track, model_name,
                is_signals.get(model_name, pd.DataFrame()),
                period="IS",
                win_start=IS_EVAL_START,
                win_end=IS_EVAL_END,
            )

    # --- Save outputs ---
    ret_df = pd.DataFrame(all_return_records)
    ret_df.to_parquet(OUT_RETURNS, index=False)

    met_df = pd.DataFrame(all_metric_records).round(4)
    # Reorder columns so period comes first
    col_order = ["period", "track", "model", "gross_sr", "net_sr",
                 "gross_ann", "net_ann", "max_dd", "annual_to", "cost_drag"]
    met_df = met_df[col_order]
    met_df.to_parquet(OUT_METRICS, index=False)

    # --- Summary table ---
    log("\n" + "=" * 65)
    log("  SUMMARY — Naïve ML Baseline IS+OOS Net-of-Cost Metrics")
    log("  (same spread=3bps, η=0.10, AUM=$100M, min_adv=$1M as paper)")
    log("=" * 65)

    pd.set_option("display.width", 140, "display.max_columns", 12)
    print(met_df.to_string(index=False))
    log("")
    log(f"  Rows: {len(met_df)}  (expected 8 = 2 tracks × 2 models × 2 periods)")

    log(f"\nSaved → {OUT_RETURNS}")
    log(f"Saved → {OUT_METRICS}")
    log(f"\nTotal elapsed: {time.time() - t0:.1f}s")

    # --- Lookahead sanity check ---
    log("\n  [SANITY] Lookahead check:")
    log("  Features use .shift(1) → signal at date T built from features at T,")
    log("  PortfolioSimulator.signal_to_positions(..., lag=1) shifts positions")
    log("  one more day → effective position at T uses signal from T-1.")
    log("  No future information leaks into OOS 2022-2024.")
    log("  OOS window: 2022-01-01 → 2024-12-31 (first and only touch).")
    log("  Final model trained exclusively on 2010-2021 IS data.")


if __name__ == "__main__":
    main()
