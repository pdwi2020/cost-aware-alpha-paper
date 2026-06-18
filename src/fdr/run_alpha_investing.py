"""Cost-aware alpha-investing comparator for TA-FDR.

Implements Foster & Stine (2008) alpha-investing with two reward variants:
  - standard:    reward = alpha_0 per rejection (baseline)
  - cost_aware:  reward = alpha_0 * normalized_net_IS_sharpe per rejection

Features are ordered by descending |mean_IC| (IS IC magnitude).
p-values are the permutation p-values from ta_fdr.parquet (same null as TA-FDR),
ensuring the two procedures are directly comparable.

Payment function: uniform allocation alpha_j = W_{j-1} / (m - j + 1).
This spreads remaining wealth evenly over remaining tests; the first test
receives alpha_0 / m, matching the Bonferroni level.

Output: data/processed/alpha_investing.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"


def alpha_investing(
    features_df: pd.DataFrame,
    alpha_0: float = 0.10,
    mode: str = "standard",
) -> pd.DataFrame:
    """Run alpha-investing on features ordered by descending |mean_ic|.

    Args:
        features_df: DataFrame with columns [feature, p_perm, T_obs, mean_ic].
                     Must already be sorted by descending |mean_ic|.
        alpha_0:     Initial wealth (= FDR target q = 0.10).
        mode:        'standard' or 'cost_aware'.

    Returns:
        DataFrame with per-feature results including wealth trace.
    """
    m = len(features_df)
    W = alpha_0

    # Cost-aware normalisation: map T_obs to [0, 1] (0 = worst, 1 = best)
    T = features_df["T_obs"].values
    T_shifted = T - T.min()
    T_norm = T_shifted / T_shifted.max() if T_shifted.max() > 0 else np.ones(m) * 0.5

    rows = []
    for j, (_, row) in enumerate(features_df.iterrows()):
        remaining = m - j
        alpha_j = W / remaining  # uniform allocation
        alpha_j = min(alpha_j, W)  # cannot spend more than available

        W_before = W
        W -= alpha_j

        if row["p_perm"] <= alpha_j:
            rejected = True
            if mode == "cost_aware":
                reward = alpha_0 * float(T_norm[j])
            else:
                reward = alpha_0
            W += reward
        else:
            rejected = False
            reward = 0.0

        rows.append(
            {
                "feature": row["feature"],
                "mean_ic": row["mean_ic"],
                "T_obs": row["T_obs"],
                "p_perm": row["p_perm"],
                "alpha_j": round(alpha_j, 6),
                "wealth_before": round(W_before, 6),
                "wealth_after": round(W, 6),
                "reward": round(reward, 6),
                "ai_rejected": rejected,
                "mode": mode,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    ta_fdr = pd.read_parquet(DATA / "ta_fdr.parquet")
    fdr = pd.read_parquet(DATA / "fdr_results.parquet")

    results = []
    for track in ["track_b", "track_a"]:
        ta = ta_fdr[ta_fdr.track == track][["feature", "T_obs", "p_perm"]].copy()
        ic = fdr[fdr.track == track][["feature", "mean_ic"]].copy()
        merged = ta.merge(ic, on="feature")

        # Order by descending |mean_ic| (IS IC magnitude)
        merged = merged.assign(abs_ic=merged["mean_ic"].abs())
        merged = merged.sort_values("abs_ic", ascending=False).drop(columns="abs_ic")
        merged = merged.reset_index(drop=True)

        for mode in ("standard", "cost_aware"):
            df = alpha_investing(merged, alpha_0=0.10, mode=mode)
            df.insert(0, "track", track)
            results.append(df)

    out = pd.concat(results, ignore_index=True)
    out.to_parquet(DATA / "alpha_investing.parquet", index=False)

    # Summary
    for track in ["track_b", "track_a"]:
        for mode in ("standard", "cost_aware"):
            sub = out[(out.track == track) & (out["mode"] == mode)]
            n_rej = sub["ai_rejected"].sum()
            alpha_min = sub["alpha_j"].min()
            p_min = sub["p_perm"].min()
            print(
                f"{track} | {mode:12s} | rejections={n_rej}/35 "
                f"| min_alpha_j={alpha_min:.5f} | min_p_perm={p_min:.4f}"
            )


if __name__ == "__main__":
    main()
