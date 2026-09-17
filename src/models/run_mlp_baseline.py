"""Reviewer revision R9: simple-MLP baseline (IS-only, 2013-2021).

Tests whether the tree models' nonlinearity is doing work beyond the linear models
+ feature engineering, by adding a simple MLP to the walk-forward IC comparison
(Table: tab:ic_summary). Reuses the exact folds, preprocessing (standardised), and
pooled-Spearman IC of model_suite.py so the row is directly comparable to the
existing Ridge/Lasso/RF/XGB/LGBM rows. Training rows are subsampled to 300k per
fold (matching the suite's grid cap) for tractability; test is the full fold.

Track A and Track B, IS 2013-2021 only; never touches the locked 2025 OOS.

Output: data/processed/mlp_ic_by_fold.parquet
Run:    python3 -u src/models/run_mlp_baseline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.model_suite import ModelSuite, make_fold_dates, preprocess, ic  # noqa: E402

FEATURES_PATH = ROOT / "data" / "processed" / "features_all.parquet"
CONFIG_PATH = ROOT / "configs" / "models.yaml"
OUT = ROOT / "data" / "processed" / "mlp_ic_by_fold.parquet"
MAX_TRAIN_ROWS = 300_000


def build_mlp(seed):
    return MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=1e-3, batch_size=4096, learning_rate_init=1e-3,
        early_stopping=True, n_iter_no_change=8, validation_fraction=0.1,
        max_iter=100, random_state=seed,
    )


def main():
    print("=== R9: simple-MLP baseline walk-forward IC (IS 2013-2021) ===\n", flush=True)
    # Lean load: the frozen-specification columns only. The full float64 panel
    # is 683 MB on disk and does not fit in 8 GB alongside the fold matrices.
    # lean_columns supplies the model-input list directly, so the non-feature
    # extras it carries (s0_eligible, regime_vix) cannot leak in as predictors.
    from src.data.lean_load import lean_columns, load_features_lean

    feat_cols, _ = lean_columns(FEATURES_PATH)
    df = load_features_lean(FEATURES_PATH)
    df.index = df.index.set_levels(
        [df.index.levels[0], pd.to_datetime(df.index.levels[1])],
    )
    features = df[feat_cols]
    fold_dates = make_fold_dates(train_start="2010-01-01", warmup_end_year=2012, test_end_year=2021)
    suite = ModelSuite(config_path=str(CONFIG_PATH))
    rng = np.random.default_rng(suite.seed)

    records = []
    for track, tcol in [("track_a", "target_track_a"), ("track_b", "target_track_b")]:
        targets = df[tcol]
        for fd in fold_dates:
            dates = features.index.get_level_values("date")
            tr = (dates >= pd.Timestamp(fd["train_start"])) & (dates <= pd.Timestamp(fd["train_end"]))
            te = (dates >= pd.Timestamp(fd["test_start"])) & (dates <= pd.Timestamp(fd["test_end"]))
            if "s0_eligible" in features.columns:
                elig = features["s0_eligible"].astype(bool)
                tr &= elig; te &= elig

            X_tr_raw, y_tr_s = features[tr], targets[tr]
            X_te_raw, y_te_s = features[te], targets[te]
            m_tr, m_te = y_tr_s.notna(), y_te_s.notna()
            X_tr_raw, y_tr = X_tr_raw[m_tr], y_tr_s[m_tr].values.astype(np.float32)
            X_te_raw, y_te = X_te_raw[m_te], y_te_s[m_te].values.astype(np.float32)

            X_tr, X_te, _ = preprocess(X_tr_raw, X_te_raw)

            if len(X_tr) > MAX_TRAIN_ROWS:
                idx = rng.choice(len(X_tr), MAX_TRAIN_ROWS, replace=False)
                X_fit, y_fit = X_tr[idx], y_tr[idx]
            else:
                X_fit, y_fit = X_tr, y_tr

            model = build_mlp(suite.seed).fit(X_fit, y_fit)
            pred = model.predict(X_te)
            fold_ic = ic(pred, y_te)
            records.append({"fold": fd["fold_id"], "track": track, "mlp": fold_ic})
            print(f"  {track}  fold {fd['fold_id']}: IC_mlp = {fold_ic:+.4f}", flush=True)

    out = pd.DataFrame(records)
    out.to_parquet(OUT, index=False)
    print("\n  MLP mean +/- std IC across 9 folds:")
    for track in ["track_a", "track_b"]:
        s = out[out.track == track]["mlp"]
        print(f"    {track}: {s.mean():+.3f} +/- {s.std():.3f}", flush=True)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
