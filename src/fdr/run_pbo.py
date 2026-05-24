"""Week 7: Probability of Backtest Overfitting (PBO) + Deflated Sharpe Ratio.

Uses the 9-fold walk-forward IC matrix (ic_by_fold.parquet) as the performance
matrix. With T=9 folds and M=7 model variants, CSCV uses all C(9,4)=126
balanced IS/OOS splits.

Outputs:
    data/processed/pbo_results.parquet   — per-model DSR + bootstrap CI
    data/processed/pbo_summary.parquet   — PBO + λ distribution per track

Run:
    python3 -u src/fdr/run_pbo.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.fdr.pbo_cscv import compute_pbo, deflated_sharpe_ratio, block_bootstrap_ic

IC_PATH    = ROOT / "data" / "processed" / "ic_by_fold.parquet"
OUT_RESULTS = ROOT / "data" / "processed" / "pbo_results.parquet"
OUT_SUMMARY = ROOT / "data" / "processed" / "pbo_summary.parquet"

MODELS      = ["ridge", "lasso", "logistic", "rf", "xgb", "lgbm", "ensemble"]
N_BOOTSTRAP = 2000
CI_ALPHA    = 0.05     # 95% CI
BLOCK_LEN   = 2        # mean block length for bootstrap


def log(msg): print(msg, flush=True)


def build_perf_matrix(ic_df: pd.DataFrame, track: str) -> tuple:
    """Extract (n_folds × n_models) performance matrix for a given track."""
    sub     = ic_df[ic_df["track"] == track].set_index("fold")
    models  = [m for m in MODELS if m in sub.columns]
    folds   = sorted(sub.index.tolist())
    mat     = sub.loc[folds, models].values.astype(float)  # (T, M)
    return mat, models, folds


def run_pbo_analysis(ic_df: pd.DataFrame, track: str) -> dict:
    """Full PBO + DSR + bootstrap analysis for one track."""
    mat, models, folds = build_perf_matrix(ic_df, track)
    T, M = mat.shape

    log(f"\n  Performance matrix: {T} folds × {M} models")

    # ── PBO ────────────────────────────────────────────────────────────────
    pbo, lv = compute_pbo(mat)
    log(f"  PBO  = {pbo:.4f}  (n_paths = {len(lv)}, C({T},{T//2}))")
    log(f"  λ    mean={lv.mean():.3f}  std={lv.std():.3f}  "
        f"  P(λ>0) = {(lv>0).mean():.3f}")

    # ── DSR for each model ─────────────────────────────────────────────────
    # Sharpe-like metric = mean_IC / std_IC (IC information ratio)
    mean_ic  = np.nanmean(mat, axis=0)          # (M,)
    std_ic   = np.nanstd(mat, axis=0, ddof=1)   # (M,)
    ic_sr    = np.where(std_ic > 1e-8, mean_ic / std_ic, 0.0)

    # n_trials = number of models tried; sharpe_std = cross-model std of IC SR
    n_trials   = M
    sharpe_std = float(np.std(ic_sr))

    dsr_records = []
    for i, model in enumerate(models):
        sr  = float(ic_sr[i])
        dsr = deflated_sharpe_ratio(
            sharpe=sr, n_trials=n_trials,
            t_obs=T, sharpe_std=sharpe_std,
        )
        # Block bootstrap 95% CI for mean IC
        ic_col     = pd.Series(mat[:, i]).dropna()
        boot       = block_bootstrap_ic(ic_col, n_bootstrap=N_BOOTSTRAP, block_length=BLOCK_LEN)
        ci_lo      = float(np.percentile(boot, 100 * CI_ALPHA / 2))
        ci_hi      = float(np.percentile(boot, 100 * (1 - CI_ALPHA / 2)))
        p_positive = float(np.mean(boot > 0))

        dsr_records.append({
            "track":       track,
            "model":       model,
            "mean_ic":     float(mean_ic[i]),
            "std_ic":      float(std_ic[i]),
            "ic_sr":       sr,
            "dsr":         dsr,
            "boot_ci_lo":  ci_lo,
            "boot_ci_hi":  ci_hi,
            "p_positive":  p_positive,
        })
        log(f"    {model:<12s}  IC={mean_ic[i]:+.4f}  IC_SR={sr:+.3f}  "
            f"DSR={dsr:.3f}  CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  "
            f"P(IC>0)={p_positive:.3f}")

    return {
        "pbo":      pbo,
        "lambda":   lv,
        "results":  pd.DataFrame(dsr_records),
    }


def main():
    log("=== Week 7: PBO + DSR Analysis ===\n")
    t0 = time.time()

    ic_df = pd.read_parquet(IC_PATH)

    all_results = []
    summary_rows = []

    for track in ["track_b", "track_a"]:
        log(f"\n{'='*60}")
        log(f"  Track: {track.upper()}")
        log(f"{'='*60}")

        out = run_pbo_analysis(ic_df, track)
        all_results.append(out["results"])

        # Best model = highest mean IC
        best_idx  = out["results"]["mean_ic"].idxmax()
        best_row  = out["results"].iloc[best_idx]
        summary_rows.append({
            "track":         track,
            "pbo":           out["pbo"],
            "n_cscv_paths":  len(out["lambda"]),
            "lambda_mean":   float(out["lambda"].mean()),
            "lambda_p_pos":  float((out["lambda"] > 0).mean()),
            "best_model":    best_row["model"],
            "best_mean_ic":  best_row["mean_ic"],
            "best_dsr":      best_row["dsr"],
        })

        log(f"\n  Best model: {best_row['model']}  "
            f"(mean IC={best_row['mean_ic']:+.4f}, DSR={best_row['dsr']:.3f})")
        log(f"  PBO = {out['pbo']:.4f}  "
            f"({'LOW — selection is robust' if out['pbo'] < 0.3 else 'MODERATE — some overfitting risk' if out['pbo'] < 0.5 else 'HIGH — likely overfitting'})")

    # Save
    results_df = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    results_df.to_parquet(OUT_RESULTS, index=False)
    summary_df.to_parquet(OUT_SUMMARY, index=False)

    log(f"\n{'='*60}")
    log("  PBO Summary")
    log(f"{'='*60}")
    log(summary_df[["track","pbo","best_model","best_mean_ic","best_dsr"]].round(4).to_string(index=False))

    log(f"\nSaved → {OUT_RESULTS}")
    log(f"Saved → {OUT_SUMMARY}")
    log(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
