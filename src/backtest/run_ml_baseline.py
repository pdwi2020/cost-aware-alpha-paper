"""Naive ML Baseline for paper baselines table.

Trains RandomForest + XGBoost on ALL 38 features (no FDR feature selection,
no SHAP weighting) and evaluates under the EXACT same cost model, Screen 0
filter, and OOS window as the main strategy (run_holdout.py).

The point: show what FDR/SHAP discovery machinery adds over throwing an
off-the-shelf ML ensemble at every feature.

Methodology:
  - Target: Track B = h=5-day forward return (target_track_b)
  - IS signal (2013-2021): walk-forward out-of-fold predictions
    (9 expanding folds via make_fold_dates, same as run_walk_forward.py)
  - OOS signal (2022-2024): single frozen model trained on ALL IS data (≤2021)
  - Ensemble: simple 50/50 average of RF and XGB predictions (no SHAP weighting)
  - Preprocessing: same preprocess() from model_suite.py
  - RF hyperparams: grid-searched per fold (max_depth ∈ {3,5,7},
    min_samples_leaf ∈ {50,100}, n_estimators=200 final; same as model_suite)
  - XGB hyperparams: grid-searched per fold (max_depth ∈ {3,4,5},
    eta ∈ {0.01,0.05}, same subsample/colsample_bytree as model_suite)
  - Screen 0: look-ahead-free (s0_eligible from features_all.parquet) + $1M ADV floor
  - Cost model: spread=3bps, impact_coeff=0.10, AUM=$100M (from backtest.yaml)
  - Rebalancing: 5-day weekly (REBAL_FREQ=5, same as run_holdout.py)

Outputs:
    data/processed/ml_baseline_metrics.parquet  — IS and OOS metrics

Run:
    python3 -u src/backtest/run_ml_baseline.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.model_suite import (
    make_fold_dates,
    preprocess,
)
from src.features.feature_spec import feature_columns as _feature_columns
from src.backtest.portfolio import PortfolioSimulator, apply_s0_eligible
from src.backtest.run_holdout import (
    build_returns_vol_adv_holdout,
    SANITIZE_CAP,
    REBAL_FREQ,
    VOL_WINDOW,
)

FEATURES_PATH = ROOT / "data" / "processed" / "features_all.parquet"
OHLCV_PATH    = ROOT / "data" / "processed" / "daily_ohlcv.parquet"
CFG_PATH      = ROOT / "configs" / "backtest.yaml"
MODEL_CFG     = ROOT / "configs" / "models.yaml"
OUT_METRICS   = ROOT / "data" / "processed" / "ml_baseline_metrics.parquet"

IS_START  = "2013-01-01"
IS_END    = "2021-12-31"
OOS_START = "2022-01-01"
OOS_END   = "2024-12-31"

SEED = 42


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Grid-search helpers (mirror model_suite._best_rf / _best_xgb exactly)
# ---------------------------------------------------------------------------

def _ic_spearman(y_pred, y_true):
    """Spearman IC between predictions and true values (ignores NaN)."""
    from scipy.stats import spearmanr
    valid = np.isfinite(y_pred) & np.isfinite(y_true)
    if valid.sum() < 10:
        return np.nan
    val = float(spearmanr(y_pred[valid], y_true[valid])[0])
    return val if np.isfinite(val) else np.nan


def _best_rf(X_tv, y_tv, X_vl, y_vl, X_full, y_full, mcfg, seed=SEED):
    """Grid-search RF on validation, refit on full train.  Mirrors model_suite._best_rf."""
    depths = mcfg["random_forest"]["max_depth_values"]
    leaves = mcfg["random_forest"]["min_samples_leaf_values"]
    best_ic, best_params = -np.inf, (depths[0], leaves[0])
    for depth in depths:
        for leaf in leaves:
            m = RandomForestRegressor(
                n_estimators=50,
                max_depth=depth,
                max_features="sqrt",
                min_samples_leaf=leaf,
                n_jobs=-1,
                random_state=seed,
            ).fit(X_tv, y_tv)
            s = _ic_spearman(m.predict(X_vl), y_vl)
            if not np.isnan(s) and s > best_ic:
                best_ic, best_params = s, (depth, leaf)
    depth, leaf = best_params
    log(f"      RF best: max_depth={depth}, min_samples_leaf={leaf}, val_IC={best_ic:+.4f}")
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=depth,
        max_features="sqrt",
        min_samples_leaf=leaf,
        n_jobs=-1,
        random_state=seed,
    ).fit(X_full, y_full)


def _best_xgb(X_tv, y_tv, X_vl, y_vl, X_full, y_full, mcfg, seed=SEED):
    """Grid-search XGB on validation, refit on full train.  Mirrors model_suite._best_xgb."""
    depths = mcfg["xgboost"]["max_depth_values"]
    etas   = mcfg["xgboost"]["eta_values"]
    n_est  = mcfg["xgboost"]["n_estimators"]
    sub    = mcfg["xgboost"]["subsample"]
    col    = mcfg["xgboost"]["colsample_bytree"]
    es     = mcfg["xgboost"]["early_stopping_rounds"]

    best_ic, best_params, best_iters = -np.inf, (depths[0], etas[0]), 200
    for depth in depths:
        for eta in etas:
            m = xgb.XGBRegressor(
                max_depth=depth, learning_rate=eta,
                n_estimators=n_est,
                subsample=sub,
                colsample_bytree=col,
                early_stopping_rounds=es,
                eval_metric="rmse",
                verbosity=0,
                random_state=seed,
                n_jobs=-1,
            ).fit(X_tv, y_tv, eval_set=[(X_vl, y_vl)], verbose=False)
            s = _ic_spearman(m.predict(X_vl), y_vl)
            if not np.isnan(s) and s > best_ic:
                best_ic, best_params = s, (depth, eta)
                best_iters = max(50, getattr(m, "best_iteration", 200))
    depth, eta = best_params
    log(f"      XGB best: max_depth={depth}, eta={eta}, iters={best_iters}, val_IC={best_ic:+.4f}")
    return xgb.XGBRegressor(
        max_depth=depth, learning_rate=eta,
        n_estimators=best_iters,
        subsample=sub,
        colsample_bytree=col,
        verbosity=0,
        random_state=seed,
        n_jobs=-1,
    ).fit(X_full, y_full)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _train_val_split(X_tr_raw, y_tr_np, val_frac=0.20):
    """Chronological train/val split (mirrors model_suite.fit_fold)."""
    dates_tr = X_tr_raw.index.get_level_values("date")
    unique_tr_dates = np.sort(dates_tr.unique())
    n_val_dates = max(1, int(len(unique_tr_dates) * val_frac))
    val_cutoff  = unique_tr_dates[-n_val_dates]
    is_val = dates_tr >= val_cutoff
    # Cap train-for-grid at 300k rows (mirrors MAX_GRID_ROWS in model_suite)
    MAX_GRID_ROWS = 300_000
    idx_tv = np.where(~is_val)[0]
    if len(idx_tv) > MAX_GRID_ROWS:
        rng = np.random.default_rng(SEED)
        idx_tv = rng.choice(idx_tv, MAX_GRID_ROWS, replace=False)
    return idx_tv, np.where(is_val)[0]


def fit_and_predict_fold(
    X_tr_raw, y_tr_raw,
    X_te_raw, y_te_raw,
    mcfg,
):
    """Fit RF+XGB on one fold and return (predictions_on_test, test_index)."""
    # Drop NaN targets
    tr_mask = y_tr_raw.notna()
    te_mask = y_te_raw.notna()
    X_tr_c = X_tr_raw[tr_mask]
    y_tr_c = y_tr_raw[tr_mask].values.astype(np.float32)
    X_te_c = X_te_raw[te_mask]

    # Preprocess — all 38 features (no selection)
    X_tr_np, X_te_np, _ = preprocess(X_tr_c, X_te_c)

    # Chronological train/val split for grid search
    idx_tv, idx_vl = _train_val_split(X_tr_c, y_tr_c)
    X_tv, y_tv = X_tr_np[idx_tv], y_tr_c[idx_tv]
    X_vl, y_vl = X_tr_np[idx_vl], y_tr_c[idx_vl]
    X_full, y_full = X_tr_np, y_tr_c

    rf_model  = _best_rf( X_tv, y_tv, X_vl, y_vl, X_full, y_full, mcfg)
    xgb_model = _best_xgb(X_tv, y_tv, X_vl, y_vl, X_full, y_full, mcfg)

    rf_pred  = rf_model.predict(X_te_np)
    xgb_pred = xgb_model.predict(X_te_np)

    # Standardise each model's predictions before averaging (unit variance ensemble)
    def _std(p):
        s = p.std()
        if s < 1e-10:
            return p
        return (p - p.mean()) / s

    ens_pred = 0.5 * _std(rf_pred) + 0.5 * _std(xgb_pred)

    return ens_pred, X_te_c.index


def fit_oos_model(X_all_raw, y_all_raw, mcfg):
    """Train frozen RF+XGB on entire IS data (≤2021) for OOS prediction.

    Returns the fitted models plus IS macro stats (mu, sig) so OOS data
    can be standardised with the same parameters — no look-ahead into OOS.
    """
    tr_mask = y_all_raw.notna()
    X_c = X_all_raw[tr_mask]
    y_c = y_all_raw[tr_mask].values.astype(np.float32)

    # Grid-search val: last year of IS (2021) as validation
    dates_all = X_c.index.get_level_values("date")
    val_cutoff = pd.Timestamp("2021-01-01")
    val_mask   = dates_all >= val_cutoff
    X_tr_raw = X_c[~val_mask]
    X_vl_raw = X_c[val_mask]
    y_tr = y_c[~val_mask]
    y_vl = y_c[val_mask]

    X_tr_np, X_vl_np, _ = preprocess(X_tr_raw, X_vl_raw)
    X_full_np, _, _     = preprocess(X_c, X_c)   # fit on full IS for final model

    # Grid search
    rf_model  = _best_rf( X_tr_np, y_tr, X_vl_np, y_vl, X_full_np, y_c, mcfg)
    xgb_model = _best_xgb(X_tr_np, y_tr, X_vl_np, y_vl, X_full_np, y_c, mcfg)

    # Record the canonical feature columns for OOS alignment.
    all_cols = _feature_columns(X_c)
    return rf_model, xgb_model, all_cols


def predict_oos(rf_model, xgb_model, X_te_raw, is_stats):
    """Predict OOS using frozen models, aligned to IS feature columns.

    All pre-registered features are cross-sectional; broadcast macro columns
    are excluded by feature_spec.feature_columns().  OOS preprocessing uses
    XS winsorize+zscore per date (same as training), with no IS-stats look-ahead.
    """
    from src.models.model_suite import _xs_winsorize_zscore

    all_cols = is_stats  # list of feature column names from fit_oos_model
    arr = X_te_raw[[c for c in all_cols if c in X_te_raw.columns]].values.astype(np.float64)
    dates_te = X_te_raw.index.get_level_values("date").values

    # All features are cross-sectional: winsorize + z-score per date.
    arr = _xs_winsorize_zscore(arr, dates_te)

    np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    X_np = arr.astype(np.float32)

    def _std(p):
        s = p.std()
        if s < 1e-10:
            return p
        return (p - p.mean()) / s

    rf_pred  = rf_model.predict(X_np)
    xgb_pred = xgb_model.predict(X_np)
    return 0.5 * _std(rf_pred) + 0.5 * _std(xgb_pred)


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------

def predictions_to_signal(preds, index):
    """Convert (predictions, MultiIndex[ticker,date]) to wide signal DataFrame."""
    s = pd.Series(preds, index=index, name="signal")
    # index is (ticker, date) — pivot to date × ticker
    level_names = index.names
    if level_names[0] == "ticker":
        sig_wide = s.unstack(level="ticker")
    else:
        sig_wide = s.unstack(level="ticker")
    sig_wide.index = pd.to_datetime(sig_wide.index)
    return sig_wide.sort_index()


# ---------------------------------------------------------------------------
# Backtest one window
# ---------------------------------------------------------------------------

def run_backtest_window(signal_wide, ohlcv, feat_df, start, end, cfg, label):
    """Run full backtest for a given signal panel and window."""
    tickers = signal_wide.columns.tolist()
    returns, vol, adv_dollars, close_px = build_returns_vol_adv_holdout(
        ohlcv, tickers, start, end
    )

    sim = PortfolioSimulator(
        config_path=str(CFG_PATH),
        spread_bps=cfg["spread_bps"],
        impact_coeff=cfg["impact_coeff"],
    )

    positions = sim.signal_to_positions(signal_wide, lag=1, rebal_freq=REBAL_FREQ)

    # Screen 0 (look-ahead-free) — see src/data/screen0.py
    n_before = int((positions.abs() > 1e-12).sum().sum())
    positions = apply_s0_eligible(positions, feat_df)
    n_after = int((positions.abs() > 1e-12).sum().sum())
    log(f"  [{label}] Screen 0 (lagged price≥$5, ADV≥$1M, PIT member): {n_before} → {n_after} active positions")

    min_adv = cfg.get("min_adv_dollars", 1e6)
    pnl_df  = sim.simulate_pnl(
        positions, returns, vol=vol,
        adv_dollars=adv_dollars,
        aum_dollars=cfg.get("aum_dollars", 1e8),
        min_adv_dollars=min_adv,
    )

    metrics = sim.compute_metrics(pnl_df)
    return metrics, pnl_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log("=" * 65)
    log("  Naive ML Baseline: RF+XGB, all 38 features, no FDR/SHAP")
    log("=" * 65)
    log(f"  IS  window : {IS_START} → {IS_END}")
    log(f"  OOS window : {OOS_START} → {OOS_END}")
    log(f"  Target     : track_b (h=5d forward return)")
    log(f"  Ensemble   : 50/50 RF + XGB (standardised predictions)")
    log(f"  Features   : all pre-registered (derived per-frame via feature_spec.feature_columns)")
    log(f"  Rebal freq : {REBAL_FREQ}d (weekly)")
    log(f"  Costs      : spread=3bps, impact=0.10, AUM=$100M\n")
    t_total = time.time()

    # ── Load data ─────────────────────────────────────────────────────────
    log("Loading features_all.parquet …")
    feat_df = pd.read_parquet(FEATURES_PATH)
    # Ensure date level is datetime
    if not pd.api.types.is_datetime64_any_dtype(feat_df.index.get_level_values("date")):
        feat_df.index = feat_df.index.set_levels(
            [feat_df.index.levels[0],
             pd.to_datetime(feat_df.index.levels[1])],
        )
    log(f"  Shape: {feat_df.shape}")

    log("Loading OHLCV …")
    ohlcv = pd.read_parquet(OHLCV_PATH)

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    with open(MODEL_CFG) as f:
        mcfg = yaml.safe_load(f)

    # Feature columns only (exclude target cols)
    feat_cols = [c for c in feat_df.columns if not c.startswith("target")]
    features  = feat_df[feat_cols]
    target    = feat_df["target_track_b"]

    log(f"  Feature cols ({len(feat_cols)}): {feat_cols}")
    log(f"  Target coverage: {target.notna().mean():.3f}\n")

    # ── Fold definitions (same as run_walk_forward.py) ────────────────────
    fold_dates = make_fold_dates(
        train_start="2010-01-01",
        warmup_end_year=2012,
        test_end_year=2021,
    )
    log(f"IS folds: {[f['fold_id'] for f in fold_dates]}")

    # ── IS: walk-forward out-of-fold predictions ──────────────────────────
    log("\n" + "=" * 65)
    log("  IS: Walk-forward out-of-fold predictions (2013–2021)")
    log("=" * 65)

    all_preds = []
    all_idx   = []

    dates_all = features.index.get_level_values("date")

    for fd in fold_dates:
        fid = fd["fold_id"]
        log(f"\n  Fold {fid}  train=[{fd['train_start']} → {fd['train_end']}]  "
            f"test=[{fd['test_start']} → {fd['test_end']}]")

        tr_mask = (dates_all >= pd.Timestamp(fd["train_start"])) & \
                  (dates_all <= pd.Timestamp(fd["train_end"]))
        te_mask = (dates_all >= pd.Timestamp(fd["test_start"])) & \
                  (dates_all <= pd.Timestamp(fd["test_end"]))

        X_tr = features[tr_mask]
        y_tr = target[tr_mask]
        X_te = features[te_mask]
        y_te = target[te_mask]

        log(f"    train rows: {tr_mask.sum():,}   test rows: {te_mask.sum():,}")
        t_fold = time.time()

        preds, idx = fit_and_predict_fold(X_tr, y_tr, X_te, y_te, mcfg)

        log(f"    fold time: {time.time()-t_fold:.0f}s")
        all_preds.append(preds)
        all_idx.append(idx)

    # Concatenate IS predictions
    is_preds = np.concatenate(all_preds)
    is_idx   = all_idx[0].append(all_idx[1:])

    log(f"\n  IS predictions: {len(is_preds):,} rows")

    # Build wide signal matrix
    is_signal = predictions_to_signal(is_preds, is_idx)
    log(f"  IS signal matrix: {is_signal.shape[0]} dates × {is_signal.shape[1]} tickers")
    log(f"  IS date range: {is_signal.index.min().date()} → {is_signal.index.max().date()}")

    # ── OOS: single frozen model trained on all IS data (≤2021) ──────────
    log("\n" + "=" * 65)
    log("  OOS: Frozen model trained on ALL IS data (≤2021)")
    log("=" * 65)

    is_mask  = (dates_all >= pd.Timestamp("2010-01-01")) & \
               (dates_all <= pd.Timestamp(IS_END))
    oos_mask = (dates_all >= pd.Timestamp(OOS_START)) & \
               (dates_all <= pd.Timestamp(OOS_END))

    X_is  = features[is_mask]
    y_is  = target[is_mask]
    X_oos = features[oos_mask]

    log(f"  IS training rows : {is_mask.sum():,}")
    log(f"  OOS predict rows : {oos_mask.sum():,}")

    t_oos = time.time()
    rf_frozen, xgb_frozen, is_stats = fit_oos_model(X_is, y_is, mcfg)
    log(f"  OOS model trained in {time.time()-t_oos:.0f}s")

    # OOS preprocessing: XS features cross-sectionally normalised per date;
    # macro features standardised with IS stats (no OOS look-ahead).
    oos_preds = predict_oos(rf_frozen, xgb_frozen, X_oos, is_stats)
    oos_signal = predictions_to_signal(oos_preds, X_oos.index)
    log(f"  OOS signal matrix: {oos_signal.shape[0]} dates × {oos_signal.shape[1]} tickers")
    log(f"  OOS date range: {oos_signal.index.min().date()} → {oos_signal.index.max().date()}")

    # ── Backtest IS window ────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("  Backtest — IS window (2013–2021)")
    log("=" * 65)

    is_metrics, is_pnl = run_backtest_window(
        is_signal, ohlcv, feat_df, IS_START, IS_END, cfg, "IS"
    )

    log(f"\n  IS Results (2013–2021):")
    log(f"    Gross Sharpe  : {is_metrics.get('gross_pnl_sharpe', float('nan')):+.3f}")
    log(f"    Net   Sharpe  : {is_metrics.get('net_pnl_sharpe',   float('nan')):+.3f}")
    log(f"    Gross Annual  : {is_metrics.get('gross_pnl_annual', float('nan'))*100:+.2f}%")
    log(f"    Net   Annual  : {is_metrics.get('net_pnl_annual',   float('nan'))*100:+.2f}%")
    log(f"    Max Drawdown  : {is_metrics.get('net_pnl_max_dd',   float('nan'))*100:.2f}%")
    log(f"    Hit Rate      : {is_metrics.get('net_pnl_hit_rate', float('nan'))*100:.1f}%")
    log(f"    Annual TO     : {is_metrics.get('annual_turnover',  float('nan')):.2f}x")
    log(f"    Cost Drag     : {is_metrics.get('cost_drag_bps',    float('nan')):.1f} bps/yr")

    # ── Backtest OOS window ───────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("  Backtest — OOS window (2022–2024)")
    log("=" * 65)

    oos_metrics, oos_pnl = run_backtest_window(
        oos_signal, ohlcv, feat_df, OOS_START, OOS_END, cfg, "OOS"
    )

    log(f"\n  OOS Results (2022–2024):")
    log(f"    Gross Sharpe  : {oos_metrics.get('gross_pnl_sharpe', float('nan')):+.3f}")
    log(f"    Net   Sharpe  : {oos_metrics.get('net_pnl_sharpe',   float('nan')):+.3f}")
    log(f"    Gross Annual  : {oos_metrics.get('gross_pnl_annual', float('nan'))*100:+.2f}%")
    log(f"    Net   Annual  : {oos_metrics.get('net_pnl_annual',   float('nan'))*100:+.2f}%")
    log(f"    Max Drawdown  : {oos_metrics.get('net_pnl_max_dd',   float('nan'))*100:.2f}%")
    log(f"    Hit Rate      : {oos_metrics.get('net_pnl_hit_rate', float('nan'))*100:.1f}%")
    log(f"    Annual TO     : {oos_metrics.get('annual_turnover',  float('nan')):.2f}x")
    log(f"    Cost Drag     : {oos_metrics.get('cost_drag_bps',    float('nan')):.1f} bps/yr")

    # ── Save metrics ──────────────────────────────────────────────────────
    rows = [
        {
            "period":          "IS",
            "gross_sharpe":    is_metrics.get("gross_pnl_sharpe", float("nan")),
            "net_sharpe":      is_metrics.get("net_pnl_sharpe",   float("nan")),
            "annual_turnover": is_metrics.get("annual_turnover",  float("nan")),
            "cost_drag_bps":   is_metrics.get("cost_drag_bps",    float("nan")),
        },
        {
            "period":          "OOS",
            "gross_sharpe":    oos_metrics.get("gross_pnl_sharpe", float("nan")),
            "net_sharpe":      oos_metrics.get("net_pnl_sharpe",   float("nan")),
            "annual_turnover": oos_metrics.get("annual_turnover",  float("nan")),
            "cost_drag_bps":   oos_metrics.get("cost_drag_bps",    float("nan")),
        },
    ]
    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT_METRICS, index=False)
    log(f"\nSaved metrics → {OUT_METRICS}")

    # Persist daily P&L series for cross-method significance testing (V3)
    is_pnl.to_parquet(ROOT / "data" / "processed" / "ml_baseline_pnl_is.parquet", index=True)
    oos_pnl.to_parquet(ROOT / "data" / "processed" / "ml_baseline_pnl_oos.parquet", index=True)
    log(f"Saved daily P&L → ml_baseline_pnl_{{is,oos}}.parquet")

    # ── Summary table ─────────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("  SUMMARY: Naive ML Baseline (RF+XGB, all 38 feat, no FDR/SHAP)")
    log("=" * 65)
    log(f"  {'Metric':<22} {'IS (2013–2021)':>15}  {'OOS (2022–2024)':>15}")
    log(f"  {'-'*22} {'-'*15}  {'-'*15}")
    log(f"  {'Gross Sharpe':<22} {is_metrics.get('gross_pnl_sharpe',float('nan')):+15.3f}  "
        f"{oos_metrics.get('gross_pnl_sharpe',float('nan')):+15.3f}")
    log(f"  {'Net Sharpe':<22} {is_metrics.get('net_pnl_sharpe',float('nan')):+15.3f}  "
        f"{oos_metrics.get('net_pnl_sharpe',float('nan')):+15.3f}")
    log(f"  {'Annual Turnover (x)':<22} {is_metrics.get('annual_turnover',float('nan')):15.2f}  "
        f"{oos_metrics.get('annual_turnover',float('nan')):15.2f}")
    log(f"  {'Cost Drag (bps/yr)':<22} {is_metrics.get('cost_drag_bps',float('nan')):15.1f}  "
        f"{oos_metrics.get('cost_drag_bps',float('nan')):15.1f}")

    log(f"\nTotal elapsed: {(time.time()-t_total)/60:.1f} min")
    return out_df


if __name__ == "__main__":
    main()
