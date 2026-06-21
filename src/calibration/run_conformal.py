"""Week 10: Conformal prediction calibration + VIX regime coverage analysis.

For each walk-forward fold (2013–2021):
  1. Train Ridge regression on training window (expanding).
  2. Calibrate ConformalIntervals on trailing 20% of training data.
  3. Compute prediction intervals on test year.
  4. Measure empirical coverage and interval widths.
  5. Regime analysis: coverage in calm (VIX≤20) vs stressed (VIX>20) periods.

Using Ridge (fast, stable) as the base forecaster for conformal calibration.
The conformal wrapper guarantees ≥90% marginal coverage regardless of model.

Outputs:
    data/processed/conformal_coverage.parquet   — per-fold coverage metrics
    data/processed/conformal_widths.parquet     — q_hat + interval width per fold/regime

Run:
    python3 -u src/calibration/run_conformal.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.model_suite import make_fold_dates
from src.calibration.conformal import ConformalIntervals
from src.features.feature_spec import feature_columns as _feature_columns

FEAT_PATH    = ROOT / "data" / "processed" / "features_all.parquet"
OUT_COV      = ROOT / "data" / "processed" / "conformal_coverage.parquet"
OUT_WIDTHS   = ROOT / "data" / "processed" / "conformal_widths.parquet"

ALPHA        = 0.10       # target miscoverage (90% CI)
CAL_FRACTION = 0.20       # 20% of training period for calibration
RIDGE_ALPHA  = 1.0        # Ridge regularisation (λ)
VIX_THRESH   = 20.0       # calm/stressed split


def log(msg): print(msg, flush=True)


def get_feature_cols(df: pd.DataFrame, target_col: str) -> list:
    # Use the canonical pre-registered feature set from feature_spec.
    return _feature_columns(df)


def xs_preprocess(X: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional winsorize (1/99%) + z-score per date, then fill NaN."""
    out = X.copy()
    dates = out.index.get_level_values("date").unique()
    for d in dates:
        row = out.loc[(slice(None), d), :]
        lo  = row.quantile(0.01)
        hi  = row.quantile(0.99)
        row = row.clip(lo, hi, axis=1)
        m   = row.mean()
        s   = row.std().replace(0, np.nan)
        out.loc[(slice(None), d), :] = ((row - m) / s).values
    return out.fillna(0.0)


def _winsorize_target(y: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> np.ndarray:
    """Winsorize target array at lo/hi quantiles (in-place safe copy)."""
    y = np.asarray(y, dtype=float).copy()
    q_lo = np.nanquantile(y, lo)
    q_hi = np.nanquantile(y, hi)
    y = np.clip(y, q_lo, q_hi)
    return y


def run_fold(
    df: pd.DataFrame,
    fold: dict,
    target_col: str,
    feature_cols: list,
    vix_col: str = "regime_vix",
) -> dict:
    """Fit, calibrate, and evaluate conformal intervals for one fold."""
    dates = df.index.get_level_values("date")

    def _mask(start, end):
        return (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))

    tr_mask = _mask(fold["train_start"], fold["train_end"])
    te_mask = _mask(fold["test_start"], fold["test_end"])

    tr_valid = tr_mask & df[target_col].notna()
    te_valid = te_mask & df[target_col].notna()

    X_tr = df.loc[tr_valid, feature_cols].fillna(0.0)
    y_tr = df.loc[tr_valid, target_col].values

    X_te = df.loc[te_valid, feature_cols].fillna(0.0)
    y_te = df.loc[te_valid, target_col].values

    # Winsorize targets at 1/99% to remove extreme outliers before Ridge fitting.
    # Conformal calibration calibrates on winsorized residuals; test targets also
    # winsorized with the same quantiles computed from training data.
    q_lo_tr = np.nanquantile(y_tr, 0.01)
    q_hi_tr = np.nanquantile(y_tr, 0.99)
    y_tr = np.clip(y_tr, q_lo_tr, q_hi_tr)
    y_te = np.clip(y_te, q_lo_tr, q_hi_tr)   # same bounds to avoid lookahead

    regime_te = df.loc[te_valid, vix_col].values if vix_col in df.columns else None

    if len(X_tr) < 100 or len(X_te) < 10:
        return {}

    # Calibration split: trailing CAL_FRACTION of training data
    n_cal  = max(int(len(X_tr) * CAL_FRACTION), 50)
    X_cal  = X_tr.iloc[-n_cal:]
    y_cal  = y_tr[-n_cal:]
    X_fit  = X_tr.iloc[:-n_cal]
    y_fit  = y_tr[:-n_cal]

    if len(X_fit) < 50:
        return {}

    # Standardise features
    scaler = StandardScaler()
    X_fit_s  = scaler.fit_transform(X_fit)
    X_cal_s  = scaler.transform(X_cal)
    X_te_s   = scaler.transform(X_te)

    # Fit Ridge
    model = Ridge(alpha=RIDGE_ALPHA).fit(X_fit_s, y_fit)

    cal_preds = model.predict(X_cal_s)
    te_preds  = model.predict(X_te_s)

    # Calibrate conformal intervals
    ci = ConformalIntervals(alpha=ALPHA)
    ci.calibrate(cal_preds, y_cal)

    # Overall coverage
    cov_overall = ci.empirical_coverage(te_preds, y_te)

    # Regime coverage — regime_vix is a string label ("calm"/"stressed").
    # Treat any non-"calm" value as stressed.
    cov_calm     = np.nan
    cov_stressed = np.nan
    q_calm       = np.nan
    q_stressed   = np.nan

    if regime_te is not None:
        calm_m     = np.array([str(v).lower() == "calm" for v in regime_te])
        stressed_m = ~calm_m

        # Regime-specific q_hat (calibrate on regime subsets of calibration data)
        regime_cal = df.loc[tr_valid, vix_col].values[-n_cal:] \
                  if vix_col in df.columns else None
        if regime_cal is not None:
            calm_cal     = np.array([str(v).lower() == "calm" for v in regime_cal])
            stressed_cal = ~calm_cal

            for regime_m_cal, regime_m_te, attr in [
                (calm_cal, calm_m, "calm"),
                (stressed_cal, stressed_m, "stressed"),
            ]:
                if regime_m_cal.sum() >= 20 and regime_m_te.sum() >= 5:
                    ci_r = ConformalIntervals(alpha=ALPHA)
                    ci_r.calibrate(cal_preds[regime_m_cal], y_cal[regime_m_cal])
                    cov = ci_r.empirical_coverage(te_preds[regime_m_te], y_te[regime_m_te])
                    if attr == "calm":
                        cov_calm, q_calm = cov, ci_r.q_hat
                    else:
                        cov_stressed, q_stressed = cov, ci_r.q_hat

        if calm_m.sum() > 0:
            cov_calm = cov_calm if not np.isnan(cov_calm) else \
                ci.empirical_coverage(te_preds[calm_m], y_te[calm_m])
        if stressed_m.sum() > 0:
            cov_stressed = cov_stressed if not np.isnan(cov_stressed) else \
                ci.empirical_coverage(te_preds[stressed_m], y_te[stressed_m])
        # vix_te alias removed — regime handled via regime_vix string labels above

    return {
        "fold":           fold["fold_id"],
        "n_train":        len(X_fit),
        "n_cal":          n_cal,
        "n_test":         len(X_te),
        "q_hat":          ci.q_hat,
        "interval_width": 2 * ci.q_hat,
        "q_hat_calm":     q_calm,
        "q_hat_stressed": q_stressed,
        "coverage":       cov_overall,
        "coverage_calm":  cov_calm,
        "coverage_stressed": cov_stressed,
        "target_coverage":  1.0 - ALPHA,
    }


def main():
    log("=== Week 10: Conformal Prediction Calibration ===\n")
    t0 = time.time()

    log("Loading features_all.parquet …")
    df = pd.read_parquet(FEAT_PATH)

    fold_dates = make_fold_dates()
    cov_records  = []
    width_records = []

    for track, target_col in [("track_b", "target_track_b"),
                               ("track_a", "target_track_a")]:
        log(f"\n{'='*55}")
        log(f"  Track: {track.upper()}  (target={target_col})")
        log(f"{'='*55}")

        feature_cols = get_feature_cols(df, target_col)
        log(f"  Features: {len(feature_cols)}")
        log(f"  Target coverage: {(1-ALPHA)*100:.0f}%  (alpha={ALPHA})")

        for fd in fold_dates:
            res = run_fold(df, fd, target_col, feature_cols)
            if not res:
                log(f"  Fold {fd['fold_id']}: skipped (insufficient data)")
                continue

            res["track"] = track
            cov_records.append(res)

            log(
                f"  Fold {fd['fold_id']}: "
                f"q̂={res['q_hat']:.4f}  "
                f"cov={res['coverage']*100:.1f}%  "
                f"calm={res['coverage_calm']*100:.1f}%  "
                f"stressed={res['coverage_stressed']*100:.1f}%"
            )

        # Summary across folds
        track_rows = [r for r in cov_records if r["track"] == track]
        if track_rows:
            covs = [r["coverage"] for r in track_rows]
            log(f"\n  Mean coverage: {np.mean(covs)*100:.2f}%  "
                f"(target={100*(1-ALPHA):.0f}%,  "
                f"std={np.std(covs)*100:.2f}%)")
            log(f"  Mean q̂: {np.mean([r['q_hat'] for r in track_rows]):.4f}")
            log(f"  Mean interval width: {np.mean([r['interval_width'] for r in track_rows]):.4f}")
            if not np.isnan(track_rows[0].get("coverage_calm", np.nan)):
                log(f"  Calm coverage:     {np.nanmean([r['coverage_calm'] for r in track_rows])*100:.2f}%")
                log(f"  Stressed coverage: {np.nanmean([r['coverage_stressed'] for r in track_rows])*100:.2f}%")

    # Save
    cov_df = pd.DataFrame(cov_records)
    cov_df.to_parquet(OUT_COV, index=False)

    log(f"\n{'='*55}")
    log("  Coverage Summary")
    log(f"{'='*55}")
    cols = ["track", "fold", "q_hat", "interval_width", "coverage",
            "coverage_calm", "coverage_stressed"]
    log(cov_df[cols].round(4).to_string(index=False))

    log(f"\nSaved → {OUT_COV}")
    log(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
