"""P1.2 — BHY (Benjamini-Yekutieli) dependence-robustness analysis.

BHY controls FDR at level q under *arbitrary* dependence, unlike BH which
requires positive regression dependency (PRDS). BHY uses the threshold

    k * q / (m * H_m),   H_m = sum_{i=1}^{m} 1/i

which is approximately 3.8x more conservative than BH for m=35 features.

We report survivors under BH and BHY side-by-side for both tracks and
both the full test and the regime (calm/stressed) pooled test.

Outputs:
    data/processed/bhy_comparison.parquet

Run:
    python3 -u src/fdr/run_bhy.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

FDR_PATH = ROOT / "data" / "processed" / "fdr_results.parquet"
OUT      = ROOT / "data" / "processed" / "bhy_comparison.parquet"


def benjamini_yekutieli(p_values: np.ndarray, q: float = 0.10):
    """BHY procedure — valid under arbitrary dependence.

    Returns (rejected: bool array, bhy_adj_p: float array).
    """
    p_arr = np.asarray(p_values, dtype=float)
    m     = len(p_arr)
    H_m   = float(np.sum(1.0 / np.arange(1, m + 1)))   # harmonic number
    q_eff = q / H_m                                       # effective q

    sorted_idx = np.argsort(p_arr)
    sorted_p   = p_arr[sorted_idx]
    thresholds = np.arange(1, m + 1) * q_eff / m
    below      = sorted_p <= thresholds

    max_k = int(np.max(np.where(below)[0])) if below.any() else -1
    reject = np.zeros(m, dtype=bool)
    if max_k >= 0:
        reject[sorted_idx[:max_k + 1]] = True

    # BHY-adjusted p-values (analogous to BH but with q/H_m denominator)
    raw_adj      = sorted_p * m / (np.arange(1, m + 1) * q_eff / q)
    adj_monotone = np.minimum.accumulate(raw_adj[::-1])[::-1]
    adj_p        = np.empty(m)
    adj_p[sorted_idx] = np.minimum(adj_monotone, 1.0)

    return reject, adj_p, H_m


def main():
    fdr_df = pd.read_parquet(FDR_PATH)
    rows   = []

    for track in ["track_a", "track_b"]:
        sub = fdr_df[fdr_df.track == track].copy()
        m   = len(sub)
        p   = sub["p_value"].values
        feat = sub["feature"].values
        mean_ic = sub["mean_ic"].values
        t_stat  = sub["t_stat"].values
        bh_rej  = sub["rejected"].values

        bhy_rej, bhy_adj_p, H_m = benjamini_yekutieli(p, q=0.10)

        for i in range(m):
            rows.append({
                "track":     track,
                "feature":   feat[i],
                "mean_ic":   float(mean_ic[i]),
                "t_stat":    float(t_stat[i]),
                "p_value":   float(p[i]),
                "bh_reject": bool(bh_rej[i]),
                "bhy_reject": bool(bhy_rej[i]),
                "bhy_adj_p": float(bhy_adj_p[i]),
            })
        print(f"\n{'='*60}")
        print(f"  {track.upper()}  (m={m}, H_m={H_m:.3f}, BHY q_eff={0.10/H_m:.4f})")
        print(f"{'='*60}")
        print(f"  BH  survivors: {int(bh_rej.sum())} / {m}")
        print(f"  BHY survivors: {int(bhy_rej.sum())} / {m}")
        only_bh  = set(feat[bh_rej]) - set(feat[bhy_rej])
        only_bhy = set(feat[bhy_rej]) - set(feat[bh_rej])
        if only_bh:
            print(f"  Dropped by BHY (not by BH): {sorted(only_bh)}")
        if only_bhy:
            print(f"  Gained by BHY (not by BH): {sorted(only_bhy)}")

    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
