"""Benjamini-Hochberg FDR correction and feature IC significance testing."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple


def compute_ic_tstats(
    ic_by_fold: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute IC t-statistics for each feature across walk-forward folds.

    Args:
        ic_by_fold: DataFrame of shape (n_folds, n_features) with IC values.

    Returns:
        Tuple of (mean_ic, t_stats, p_values) each of length n_features.
    """
    n       = len(ic_by_fold)
    mean_ic = ic_by_fold.mean()
    std_ic  = ic_by_fold.std(ddof=1)
    # Avoid division by zero for constant features
    std_ic  = std_ic.where(std_ic > 1e-10, other=np.nan)
    t_stats = mean_ic / (std_ic / np.sqrt(n))
    # Two-sided t-test with df = n - 1
    p_values = pd.Series(
        2 * stats.t.sf(np.abs(t_stats.fillna(0)), df=n - 1),
        index=mean_ic.index,
    )
    p_values[std_ic.isna()] = 1.0   # constant IC → no signal
    return mean_ic, t_stats, p_values


def benjamini_hochberg(
    p_values: np.ndarray,
    q: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply BH procedure to control FDR at level q.

    Args:
        p_values: Array of p-values (one per feature/hypothesis).
        q: Target FDR level (default 0.10).

    Returns:
        Tuple of (rejected: bool array, bh_adjusted_p: float array).
    """
    p_arr      = np.asarray(p_values, dtype=float)
    m          = len(p_arr)
    sorted_idx = np.argsort(p_arr)
    sorted_p   = p_arr[sorted_idx]

    # BH threshold for rank k (1-indexed): k * q / m
    thresholds = np.arange(1, m + 1) * q / m
    below      = sorted_p <= thresholds

    # Reject all hypotheses up to the largest rejected rank
    if not below.any():
        max_k = -1
    else:
        max_k = int(np.max(np.where(below)[0]))

    reject = np.zeros(m, dtype=bool)
    if max_k >= 0:
        reject[sorted_idx[: max_k + 1]] = True

    # BH-adjusted p-values: p_adj_(k) = min_{k'>=k} p_(k') * m/k'
    raw_adj     = sorted_p * m / np.arange(1, m + 1)
    adj_monotone = np.minimum.accumulate(raw_adj[::-1])[::-1]
    adj_p        = np.empty(m)
    adj_p[sorted_idx] = np.minimum(adj_monotone, 1.0)

    return reject, adj_p


def bh_with_regime(
    ic_calm: pd.DataFrame,
    ic_stressed: pd.DataFrame,
    q: float = 0.10,
) -> pd.DataFrame:
    """BH correction including regime-conditional ICs in a single pooled test.

    Tests each (feature, regime) pair as a separate hypothesis. The VIX ≤ 20 /
    > 20 split is pre-specified so it does not inflate the false discovery rate.

    Args:
        ic_calm:     DataFrame (n_folds, n_features) of ICs in calm regime.
        ic_stressed: DataFrame (n_folds, n_features) of ICs in stressed regime.
        q: FDR level.

    Returns:
        DataFrame with columns [feature, regime, mean_ic, std_ic, t_stat,
                                 p_value, bh_adj_p, rejected].
    """
    rows = []
    for regime, ic_df in [("calm", ic_calm), ("stressed", ic_stressed)]:
        mean_ic, t_stats, p_values = compute_ic_tstats(ic_df)
        for feat in ic_df.columns:
            rows.append({
                "feature": feat,
                "regime":  regime,
                "n_folds": int(ic_df[feat].notna().sum()),
                "mean_ic": float(mean_ic[feat]),
                "std_ic":  float(ic_df[feat].std(ddof=1)),
                "t_stat":  float(t_stats[feat]),
                "p_value": float(p_values[feat]),
            })

    result = pd.DataFrame(rows)
    reject, adj_p = benjamini_hochberg(result["p_value"].values, q=q)
    result["bh_adj_p"] = adj_p
    result["rejected"] = reject
    return result.sort_values(["rejected", "p_value"], ascending=[False, True])
