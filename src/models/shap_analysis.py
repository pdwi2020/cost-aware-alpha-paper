"""Week 5: SHAP feature importance analysis for XGB and LGBM.

For each fold × track, fits XGB and LGBM using fixed representative
hyperparameters (no grid search — importance ranking is HP-robust), then
computes TreeExplainer SHAP values on up to N_SHAP_SAMPLES test rows.

Aggregates mean |SHAP| across folds per (feature, model, track). Used to
pre-screen features before the BH FDR correction in Week 6.

Outputs:
    data/processed/shap_by_fold.parquet   — per-fold per-feature mean |SHAP|
    data/processed/shap_summary.parquet   — mean |SHAP| averaged across folds

Run:
    python3 -u src/models/shap_analysis.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import lightgbm as lgb
import yaml

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.model_suite import make_fold_dates, preprocess

FEATURES_PATH = ROOT / "data" / "processed" / "features_all.parquet"
CONFIG_PATH   = ROOT / "configs" / "models.yaml"
OUT_FOLD      = ROOT / "data" / "processed" / "shap_by_fold.parquet"
OUT_SUMMARY   = ROOT / "data" / "processed" / "shap_summary.parquet"

N_SHAP = 2000  # test rows sampled for SHAP (speed vs accuracy trade-off)
SEED   = 42


def _fit_xgb(X_tr: np.ndarray, y_tr: np.ndarray, cfg: dict, seed: int) -> xgb.XGBRegressor:
    xc = cfg["xgboost"]
    m = xgb.XGBRegressor(
        max_depth=xc["max_depth_values"][1],   # 4
        learning_rate=xc["eta_values"][0],      # 0.01
        n_estimators=200,
        subsample=xc["subsample"],
        colsample_bytree=xc["colsample_bytree"],
        verbosity=0, random_state=seed, n_jobs=-1,
    )
    m.fit(X_tr, y_tr)
    return m


def _fit_lgbm(X_tr: np.ndarray, y_tr: np.ndarray, cfg: dict, seed: int) -> lgb.LGBMRegressor:
    lc = cfg["lightgbm"]
    m = lgb.LGBMRegressor(
        num_leaves=lc["num_leaves_values"][1],      # 31
        learning_rate=lc["learning_rate_values"][0], # 0.01
        n_estimators=200,
        verbose=-1, random_state=seed, n_jobs=-1,
    )
    m.fit(X_tr.astype(np.float32), y_tr.astype(np.float32))
    return m


def compute_shap_importance(
    model,
    X_shap: np.ndarray,
    feat_names: list,
) -> dict:
    """Return {feature: mean_abs_shap} using TreeExplainer."""
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    sv = explainer.shap_values(X_shap)          # (N, D)
    mean_abs = np.abs(sv).mean(axis=0)           # (D,)
    return dict(zip(feat_names, mean_abs.tolist()))


def run_shap_analysis() -> pd.DataFrame:
    print("=== Week 5: SHAP Feature Importance ===\n")
    t_start = time.time()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_parquet(FEATURES_PATH)
    df.index = df.index.set_levels(
        [df.index.levels[0], pd.to_datetime(df.index.levels[1])]
    )
    feat_cols  = [c for c in df.columns if not c.startswith("target")]
    fold_dates = make_fold_dates()

    rng    = np.random.default_rng(SEED)
    records = []

    for track, target_col in [("track_b", "target_track_b"),
                               ("track_a", "target_track_a")]:
        print(f"\n{'='*55}")
        print(f"  Track: {track.upper()}  (target = {target_col})")
        print(f"{'='*55}")

        targets = df[target_col]

        for fd in fold_dates:
            fid = fd["fold_id"]
            print(f"  Fold {fid}...", end="", flush=True)
            t0 = time.time()

            dates = df.index.get_level_values("date")
            tr_mask = (dates >= pd.Timestamp(fd["train_start"])) & \
                      (dates <= pd.Timestamp(fd["train_end"]))
            te_mask = (dates >= pd.Timestamp(fd["test_start"])) & \
                      (dates <= pd.Timestamp(fd["test_end"]))

            tr_valid = tr_mask & targets.notna()
            te_valid = te_mask & targets.notna()

            X_tr_raw = df[feat_cols][tr_valid]
            y_tr     = targets[tr_valid].values.astype(np.float32)
            X_te_raw = df[feat_cols][te_valid]

            X_tr, X_te, feat_names = preprocess(X_tr_raw, X_te_raw)

            # Sample test rows for SHAP computation
            n_sample = min(N_SHAP, len(X_te))
            idx      = rng.choice(len(X_te), n_sample, replace=False)
            X_shap   = X_te[idx]

            for model_name, fit_fn in [("xgb", _fit_xgb), ("lgbm", _fit_lgbm)]:
                model = fit_fn(X_tr, y_tr, cfg, SEED)
                shap_dict = compute_shap_importance(model, X_shap, feat_names)
                for feat, imp in shap_dict.items():
                    records.append({
                        "fold":          fid,
                        "track":         track,
                        "model":         model_name,
                        "feature":       feat,
                        "mean_abs_shap": float(imp),
                    })

            print(f" {time.time()-t0:.0f}s", flush=True)

    # Save fold-level results
    df_fold = pd.DataFrame(records)
    df_fold.to_parquet(OUT_FOLD, index=False)
    print(f"\nSaved fold SHAP → {OUT_FOLD}  ({OUT_FOLD.stat().st_size/1e3:.0f} KB)")

    # Summary: mean |SHAP| across folds
    summary = (
        df_fold
        .groupby(["feature", "model", "track"])["mean_abs_shap"]
        .mean()
        .reset_index()
        .rename(columns={"mean_abs_shap": "mean_abs_shap"})
    )
    summary.to_parquet(OUT_SUMMARY, index=False)
    print(f"Saved SHAP summary → {OUT_SUMMARY}")

    # Print top-10 features per model × track
    for model in ["xgb", "lgbm"]:
        for track in ["track_b", "track_a"]:
            sub = summary[(summary.model == model) & (summary.track == track)]
            top = sub.nlargest(10, "mean_abs_shap")
            print(f"\n  Top-10 — {model.upper()} | {track.upper()}:")
            for _, row in top.iterrows():
                print(f"    {row.feature:<28s} {row.mean_abs_shap:.5f}")

    print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min")
    return summary


if __name__ == "__main__":
    run_shap_analysis()
