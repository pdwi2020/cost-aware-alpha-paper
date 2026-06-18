"""Objective-consistency robustness for the PBO / DSR model comparison.

Referee point: the logistic model is trained on binary sign labels (y>0) and
scored by Spearman IC against the *continuous* target, whereas the other six
models regress the continuous (heavy-tailed) target. On the idiosyncratic
Track~A target this gives the logistic an IC information ratio of ~6.1 -- a
large outlier -- which drives Track~A's published PBO to 0.000 and DSR to 0.986.
Because the deployed signal is feature-based (SHAP-weighted), not the logistic's
predictions, the honest overfitting measure restricts the CSCV comparison to the
models that share the continuous-return objective.

This script recomputes PBO + best-model DSR two ways and saves the result for
audit. It does NOT modify any published parquet.

Output: data/processed/pbo_objective_robustness.parquet
Run:    python3 -u src/fdr/pbo_objective_check.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.fdr.pbo_cscv import compute_pbo, deflated_sharpe_ratio

IC_PATH = ROOT / "data" / "processed" / "ic_by_fold.parquet"
OUT = ROOT / "data" / "processed" / "pbo_objective_robustness.parquet"

ALL = ["ridge", "lasso", "logistic", "rf", "xgb", "lgbm", "ensemble"]
# Models sharing the continuous-return regression objective (exclude the
# sign-classifier logistic and the IC-weighted ensemble it contaminates):
REG = ["ridge", "lasso", "rf", "xgb", "lgbm"]


def analyse(ic_df, track, models):
    sub = ic_df[ic_df.track == track].set_index("fold")
    folds = sorted(sub.index)
    mat = sub.loc[folds, models].values.astype(float)
    pbo, lv = compute_pbo(mat)
    mean_ic = np.nanmean(mat, axis=0)
    std_ic = np.nanstd(mat, axis=0, ddof=1)
    ic_sr = np.where(std_ic > 1e-8, mean_ic / std_ic, 0.0)
    ssd = float(np.std(ic_sr))
    M = len(models)
    dsr = np.array([deflated_sharpe_ratio(float(ic_sr[i]), M, mat.shape[0], ssd)
                    for i in range(M)])
    best = int(np.argmax(mean_ic))
    return {
        "pbo": float(pbo),
        "p_lambda_pos": float((lv > 0).mean()),
        "best_model": models[best],
        "best_ic": float(mean_ic[best]),
        "best_ic_sr": float(ic_sr[best]),
        "best_dsr": float(dsr[best]),
    }


def main():
    ic_df = pd.read_parquet(IC_PATH)
    rows = []
    for track in ["track_a", "track_b"]:
        for label, models in [("all7", ALL), ("regression5", REG)]:
            rec = {"track": track, "model_pool": label, "n_models": len(models)}
            rec.update(analyse(ic_df, track, models))
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(out.round(4).to_string(index=False))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
