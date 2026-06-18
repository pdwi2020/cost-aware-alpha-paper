"""E1: Adaptive Conformal Inference (ACI) under distribution shift.

Reference: Gibbs & Candès (2021), "Adaptive Conformal Inference Under
Distribution Shift", NeurIPS 34.

Static split conformal guarantees coverage only under exchangeability, which the
2020 COVID fold violates (documented undercoverage 73-77% vs 90% target). ACI
restores long-run coverage by re-estimating the miscoverage level online:

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t),

where err_t is the realised miscoverage on day t and the interval at day t uses
the (1 - alpha_t) empirical quantile of the calibration residuals. The interval
for day t is fixed BEFORE observing day t's outcomes (alpha_t depends only on
days < t), so there is no lookahead.

This is a calibration diagnostic on the walk-forward folds; it does NOT touch the
locked 2022-2024 OOS performance test.

Output: data/processed/aci_coverage.parquet
Run:    python3 -u src/calibration/run_aci.py
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

FEAT_PATH = ROOT / "data" / "processed" / "features_all.parquet"
OUT_PATH  = ROOT / "data" / "processed" / "aci_coverage.parquet"

ALPHA        = 0.10     # target miscoverage (90% interval)
CAL_FRACTION = 0.20
RIDGE_ALPHA  = 1.0
GAMMA        = 0.01     # ACI step size (Gibbs-Candès); result is insensitive in [0.005, 0.05]

SHAP_DROPPED = {"corr_XLRE", "corr_XLC", "term_spread_x_mom"}


def log(msg): print(msg, flush=True)


def get_feature_cols(df, target_col):
    exclude = {"target_track_a", "target_track_b"} | SHAP_DROPPED
    return [c for c in df.columns if c not in exclude and c != target_col]


def fit_fold(df, fold, target_col, feature_cols):
    """Fit Ridge, calibrate, and return (cal_resid, test_dates, test_resid)."""
    dates = df.index.get_level_values("date")

    def _mask(s, e):
        return (dates >= pd.Timestamp(s)) & (dates <= pd.Timestamp(e))

    tr_valid = _mask(fold["train_start"], fold["train_end"]) & df[target_col].notna()
    te_valid = _mask(fold["test_start"], fold["test_end"]) & df[target_col].notna()

    X_tr = df.loc[tr_valid, feature_cols].fillna(0.0)
    y_tr = df.loc[tr_valid, target_col].values
    X_te = df.loc[te_valid, feature_cols].fillna(0.0)
    y_te = df.loc[te_valid, target_col].values
    te_dates = df.loc[te_valid].index.get_level_values("date").values

    if len(X_tr) < 100 or len(X_te) < 10:
        return None

    q_lo, q_hi = np.nanquantile(y_tr, 0.01), np.nanquantile(y_tr, 0.99)
    y_tr = np.clip(y_tr, q_lo, q_hi)
    y_te = np.clip(y_te, q_lo, q_hi)

    n_cal = max(int(len(X_tr) * CAL_FRACTION), 50)
    X_cal, y_cal = X_tr.iloc[-n_cal:], y_tr[-n_cal:]
    X_fit, y_fit = X_tr.iloc[:-n_cal], y_tr[:-n_cal]
    if len(X_fit) < 50:
        return None

    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit)
    X_cal_s = scaler.transform(X_cal)
    X_te_s  = scaler.transform(X_te)

    model = Ridge(alpha=RIDGE_ALPHA).fit(X_fit_s, y_fit)
    cal_resid  = np.abs(y_cal - model.predict(X_cal_s))
    test_resid = np.abs(y_te - model.predict(X_te_s))
    return cal_resid, te_dates, test_resid


def static_coverage(cal_resid, test_resid):
    n = len(cal_resid)
    level = min((1.0 - ALPHA) * (1.0 + 1.0 / n), 1.0)
    q_hat = float(np.quantile(cal_resid, level))
    return float(np.mean(test_resid <= q_hat)), q_hat


def aci_coverage(cal_resid, te_dates, test_resid, gamma=GAMMA):
    """Online adaptive conformal inference, processing one trading day at a time."""
    n = len(cal_resid)
    order = np.argsort(te_dates, kind="stable")
    te_dates, test_resid = te_dates[order], test_resid[order]
    uniq_days = pd.unique(te_dates)

    alpha_t = ALPHA
    covered, total, alphas = 0, 0, []
    for d in uniq_days:
        r_day = test_resid[te_dates == d]
        # Interval at day t uses alpha_t fixed from days < t (no lookahead).
        if alpha_t <= 0.0:
            q_t = np.inf                      # cover everything
        else:
            level = (1.0 - alpha_t) * (1.0 + 1.0 / n)
            q_t = np.inf if level >= 1.0 else float(np.quantile(cal_resid, level))
        cov_day = (r_day <= q_t)
        covered += int(cov_day.sum())
        total   += len(r_day)
        err_t = 1.0 - float(cov_day.mean())
        alpha_t = alpha_t + gamma * (ALPHA - err_t)
        alphas.append(alpha_t)
    return covered / total, float(np.mean(alphas))


def main():
    log("=== E1: Adaptive Conformal Inference (Gibbs-Candes 2021) ===\n")
    t0 = time.time()
    df = pd.read_parquet(FEAT_PATH)
    folds = make_fold_dates()

    rows = []
    for track, target_col in [("track_b", "target_track_b"),
                              ("track_a", "target_track_a")]:
        feats = get_feature_cols(df, target_col)
        log(f"\n=== {track.upper()} ===")
        log(f"{'fold':>6} {'static_cov':>11} {'aci_cov':>9} {'mean_alpha_t':>13}")
        for fd in folds:
            out = fit_fold(df, fd, target_col, feats)
            if out is None:
                continue
            cal_resid, te_dates, test_resid = out
            stat_cov, _ = static_coverage(cal_resid, test_resid)
            aci_cov, mean_a = aci_coverage(cal_resid, te_dates, test_resid)
            flag = "  <-- COVID" if fd["fold_id"] == 2020 else ""
            log(f"{fd['fold_id']:>6} {stat_cov*100:>10.1f}% {aci_cov*100:>8.1f}% "
                f"{mean_a:>13.4f}{flag}")
            rows.append({
                "track": track, "fold": fd["fold_id"],
                "static_coverage": stat_cov, "aci_coverage": aci_cov,
                "mean_alpha_t": mean_a, "target_coverage": 1.0 - ALPHA,
                "gamma": GAMMA,
            })

    out = pd.DataFrame(rows)
    out.to_parquet(OUT_PATH, index=False)

    log(f"\n{'='*55}\n  2020 COVID fold: static vs ACI\n{'='*55}")
    covid = out[out.fold == 2020]
    for _, r in covid.iterrows():
        log(f"  {r['track']}: static={r['static_coverage']*100:.1f}%  "
            f"-> ACI={r['aci_coverage']*100:.1f}%  (target 90%)")
    log(f"\n  Mean |coverage-90%| error across all folds:")
    for track in ["track_b", "track_a"]:
        sub = out[out.track == track]
        se = np.mean(np.abs(sub.static_coverage - 0.9)) * 100
        ae = np.mean(np.abs(sub.aci_coverage - 0.9)) * 100
        log(f"    {track}: static={se:.2f}pp  ACI={ae:.2f}pp")
    log(f"\nSaved -> {OUT_PATH}\nElapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
