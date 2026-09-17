"""Reviewer revision R8: ECE / Brier / Brier Skill Score for the classification head.

The manuscript already reports conformal *interval* coverage (Section: Conformal).
This adds probabilistic-classifier calibration for the logistic head (which predicts
P(idiosyncratic residual return > 0)), with the baselines the reviewer requested so
the Brier score is interpretable:

  * Climatological baseline  -- predict each fold's TRAINING base rate P(y_tr>0)
    (no look-ahead) for every test name.
  * Logistic (uncalibrated)  -- the raw predict_proba output.
  * Logistic + isotonic      -- post-hoc isotonic calibration fit on the validation
                                split, applied to the test fold.

Brier Skill Score is normalised against the climatological baseline:
    BSS = 1 - Brier_model / Brier_climatology.

Reuses the exact walk-forward folds, preprocessing, val split, and logistic C-grid of
model_suite.py (fits ONLY the logistic per fold -> fast). Track A, IS 2013-2021 only;
never touches the locked 2025 OOS.

Output: data/processed/brier_calibration.parquet
Run:    python3 -u src/calibration/run_brier_calibration.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.model_suite import ModelSuite, make_fold_dates, preprocess  # noqa: E402


def fit_logistic(X_tv, y_tv_bin, X_vl, y_vl, X_full, y_full_bin, c_values, max_iter):
    """Same l2-logistic + C-grid as model_suite._best_logistic, but with the lbfgs
    solver (identical convex objective, far faster convergence than saga at 30-D;
    predicted probabilities are numerically equivalent). Validation score = IC of
    P(y>0) against the continuous target, matching the deployed selection rule."""
    best_ic, best_c = -np.inf, c_values[0]
    for c in c_values:
        m = LogisticRegression(C=c, penalty="l2", max_iter=max_iter, solver="lbfgs").fit(X_tv, y_tv_bin)
        s = spearmanr(m.predict_proba(X_vl)[:, 1], y_vl).correlation
        if s is not None and not np.isnan(s) and s > best_ic:
            best_ic, best_c = s, c
    return LogisticRegression(C=best_c, penalty="l2", max_iter=max_iter,
                              solver="lbfgs").fit(X_full, y_full_bin)

FEATURES_PATH = ROOT / "data" / "processed" / "features_all.parquet"
CONFIG_PATH = ROOT / "configs" / "models.yaml"
OUT = ROOT / "data" / "processed" / "brier_calibration.parquet"
MAX_GRID_ROWS = 300_000
VAL_FRAC = 0.20


def ece(p, y, n_bins=10):
    """Expected Calibration Error (equal-width bins)."""
    p = np.asarray(p); y = np.asarray(y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        e += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return float(e)


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def main():
    print("=== R8: Brier / ECE / BSS for the logistic head (Track A, IS 2013-2021) ===\n", flush=True)
    df = pd.read_parquet(FEATURES_PATH)
    df.index = df.index.set_levels(
        [df.index.levels[0], pd.to_datetime(df.index.levels[1])],
    )
    feat_cols = [c for c in df.columns if not c.startswith("target")]
    features = df[feat_cols]
    targets = df["target_track_a"]

    fold_dates = make_fold_dates(train_start="2010-01-01", warmup_end_year=2012, test_end_year=2021)
    suite = ModelSuite(config_path=str(CONFIG_PATH))
    rng = np.random.default_rng(suite.seed)

    p_uncal_all, p_cal_all, y_all, p_clim_all = [], [], [], []

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

        # chronological validation split (last VAL_FRAC of training dates)
        d_tr = X_tr_raw.index.get_level_values("date")
        u = np.sort(d_tr.unique())
        cutoff = u[-max(1, int(len(u) * VAL_FRAC))]
        is_val = d_tr >= cutoff
        X_tv_full, y_tv_full = X_tr[~is_val], y_tr[~is_val]
        X_vl, y_vl = X_tr[is_val], y_tr[is_val]

        if len(X_tv_full) > MAX_GRID_ROWS:
            idx = rng.choice(len(X_tv_full), MAX_GRID_ROWS, replace=False)
            X_tv, y_tv = X_tv_full[idx], y_tv_full[idx]
        else:
            X_tv, y_tv = X_tv_full, y_tv_full

        y_tv_bin = (y_tv > 0).astype(int)
        X_full = np.vstack([X_tv_full, X_vl])
        y_full_bin = (np.concatenate([y_tv_full, y_vl]) > 0).astype(int)

        model = fit_logistic(
            X_tv, y_tv_bin, X_vl, y_vl, X_full, y_full_bin,
            suite.cfg["logistic"]["C_values"], suite.cfg["logistic"]["max_iter"],
        )

        p_te = model.predict_proba(X_te)[:, 1]
        y_te_bin = (y_te > 0).astype(int)

        # isotonic calibration fit on the validation split
        p_vl = model.predict_proba(X_vl)[:, 1]
        y_vl_bin = (y_vl > 0).astype(int)
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_vl, y_vl_bin)
        p_te_cal = iso.predict(p_te)

        base_rate = float((y_tv_full > 0).mean())   # training base rate (no look-ahead)

        p_uncal_all.append(p_te)
        p_cal_all.append(p_te_cal)
        y_all.append(y_te_bin)
        p_clim_all.append(np.full_like(p_te, base_rate))

        print(f"  Fold {fd['fold_id']}: n_test={len(p_te):,}  base_rate={base_rate:.3f}  "
              f"Brier_uncal={brier(p_te,y_te_bin):.4f}", flush=True)

    p_uncal = np.concatenate(p_uncal_all)
    p_cal = np.concatenate(p_cal_all)
    y = np.concatenate(y_all)
    p_clim = np.concatenate(p_clim_all)

    b_clim = brier(p_clim, y)
    rows = [
        {"model": "climatology", "brier": b_clim, "ece": ece(p_clim, y),
         "bss": 0.0},
        {"model": "logistic_uncalibrated", "brier": brier(p_uncal, y), "ece": ece(p_uncal, y),
         "bss": 1 - brier(p_uncal, y) / b_clim},
        {"model": "logistic_isotonic", "brier": brier(p_cal, y), "ece": ece(p_cal, y),
         "bss": 1 - brier(p_cal, y) / b_clim},
    ]
    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    print(f"\n  Pooled OOS test rows: {len(y):,}   base rate: {y.mean():.3f}\n")
    print(out.round(4).to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
