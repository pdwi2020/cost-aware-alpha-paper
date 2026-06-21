"""P1.2 — BHY (Benjamini-Yekutieli) dependence-robustness analysis.

BHY controls FDR at level q under *arbitrary* dependence, unlike BH which
requires positive regression dependency (PRDS).  BHY uses the threshold

    k * q / (m * H_m),   H_m = sum_{i=1}^{m} 1/i

which is approximately 3.8x more conservative than BH for m ≈ 32 features.

Phase-3 change: the input p-values are now BOOTSTRAP p-values from the
daily cross-sectional IC estimand (spec `fdr.dependence_robust`), stored in
fdr_results.parquet under the column "boot_p".  BH and BHY are applied to
the same bootstrap p-values via shared procedures from bh_correction.py.

Outputs:
    data/processed/bhy_comparison.parquet

Run:
    python3 -u src/fdr/run_bhy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.fdr.bh_correction import benjamini_hochberg, bhy_procedure

FDR_PATH = ROOT / "data" / "processed" / "fdr_results.parquet"
OUT      = ROOT / "data" / "processed" / "bhy_comparison.parquet"


def main() -> None:
    fdr_df = pd.read_parquet(FDR_PATH)

    # Determine which p-value column to use.  Phase-3 produces "boot_p";
    # fall back to "p_value" for backward compatibility with older runs.
    p_col = "boot_p" if "boot_p" in fdr_df.columns else "p_value"
    print(f"  Using p-value column: {p_col!r}")

    rows = []

    for track in ["track_a", "track_b"]:
        sub     = fdr_df[fdr_df.track == track].copy()
        m       = len(sub)
        p       = sub[p_col].values
        feat    = sub["feature"].values
        ic_col  = "ic_bar" if "ic_bar" in sub.columns else "mean_ic"
        mean_ic = sub[ic_col].values
        t_stat  = sub["t_stat"].values

        bh_rej,  bh_adj_p  = benjamini_hochberg(p, q=0.10)
        bhy_rej, bhy_adj_p = bhy_procedure(p, q=0.10)
        H_m = float(np.sum(1.0 / np.arange(1, m + 1)))

        for i in range(m):
            rows.append({
                "track":        track,
                "feature":      feat[i],
                "mean_ic":      float(mean_ic[i]),
                "t_stat":       float(t_stat[i]),
                "p_value":      float(p[i]),
                "bh_reject":    bool(bh_rej[i]),
                "bhy_reject":   bool(bhy_rej[i]),
                "bhy_adj_p":    float(bhy_adj_p[i]),
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
