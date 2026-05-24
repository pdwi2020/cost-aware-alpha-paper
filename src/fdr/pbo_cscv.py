"""Probability of Backtest Overfitting via Combinatorial Symmetric Cross-Validation.

Reference: Bailey, Borwein, López de Prado & Zhu (2015).
           "The Probability of Backtest Overfitting."
           Journal of Computational Finance.

DSR reference: Bailey & López de Prado (2014).
               "The Deflated Sharpe Ratio."
               Financial Analysts Journal.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats
from typing import Tuple

EULER_MASCHERONI = 0.5772156649015329


def compute_pbo(
    performance_matrix: np.ndarray,
    n_splits: int = 16,
) -> Tuple[float, np.ndarray]:
    """Compute PBO via Combinatorial Symmetric Cross-Validation (CSCV).

    Uses all C(T, T//2) balanced IS/OOS splits regardless of n_splits (since
    T=9 walk-forward folds is too small for S=16 equal-block partitioning).
    This is equivalent to CSCV with maximum combinatorial coverage.

    Args:
        performance_matrix: Shape (T, M) — T time periods, M strategy variants.
                            Higher values = better performance.
        n_splits: Kept for API compatibility; actual splits = C(T, T//2).

    Returns:
        (pbo, lambda_vals):
          pbo        — fraction of paths where IS-best model is OOS below-median.
          lambda_vals — logit(OOS rank) for each CSCV path.
    """
    T, M          = performance_matrix.shape
    half          = T // 2
    all_combos    = list(combinations(range(T), half))

    # Replace NaN with slightly below the global minimum so NaN models rank last.
    # np.argmax and rankdata both have undefined/incorrect NaN behaviour.
    global_min = np.nanmin(performance_matrix)
    mat = np.where(np.isnan(performance_matrix),
                   global_min - 1e-6,
                   performance_matrix)

    lambda_vals = []
    for is_idx in all_combos:
        is_set  = set(is_idx)
        oos_idx = [i for i in range(T) if i not in is_set]

        is_perf  = mat[list(is_idx), :].mean(axis=0)   # (M,)
        oos_perf = mat[oos_idx, :].mean(axis=0)        # (M,)

        n_star = int(np.argmax(is_perf))

        # Normalized OOS rank of IS-selected strategy (1 = worst, M = best → [1/M, 1])
        ranks       = stats.rankdata(oos_perf, method="average")
        omega       = float(ranks[n_star]) / M
        omega_clip  = np.clip(omega, 1e-6, 1.0 - 1e-6)
        # λ < 0 ↔ ω < 0.5 ↔ selected strategy is OOS below-median → overfitting
        lambda_vals.append(np.log(omega_clip / (1.0 - omega_clip)))

    lv  = np.array(lambda_vals)
    pbo = float(np.mean(lv < 0.0))
    return pbo, lv


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    t_obs: int,
    sharpe_std: float = 1.0,
) -> float:
    """Deflated Sharpe Ratio (DSR) adjusted for selection bias from n_trials tests.

    Formula (simplified — omits skewness/kurtosis correction appropriate for
    small T):

        E[max SR] = σ_SR * [(1-γ) * Φ^{-1}(1 - 1/n) + γ * Φ^{-1}(1 - 1/(n·e))]
        DSR = Φ((SR - E[max SR]) / SE(SR))

    where SE²(SR) ≈ (1 + SR²/2) / (T - 1).

    Args:
        sharpe:     Observed Sharpe/IC ratio of the SELECTED strategy.
        n_trials:   Number of strategy variants evaluated before selection.
        t_obs:      Number of independent observations (here: walk-forward folds).
        sharpe_std: Std of Sharpe ratios across all n_trials strategies.

    Returns:
        DSR ∈ (0, 1). Values > 0.5 mean observed SR exceeds the data-mining
        expectation → strategy is likely genuine.
    """
    z1      = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2      = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    e_max   = sharpe_std * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)

    # SE of SR under iid returns (simplified; t_obs should be ≥ 3)
    se_sr   = np.sqrt(max((1.0 + 0.5 * sharpe ** 2) / max(t_obs - 1, 1), 1e-12))
    dsr     = float(stats.norm.cdf((sharpe - e_max) / se_sr))
    return dsr


def block_bootstrap_ic(
    ic_series: pd.Series,
    n_bootstrap: int = 2000,
    block_length: int = 2,
) -> np.ndarray:
    """Stationary block bootstrap CI for mean IC (Politis & Romano 1994).

    Block lengths are geometrically distributed with mean = block_length.
    For T=9 annual folds a block_length of 2 is conservative but appropriate.

    Args:
        ic_series:    IC values (one per walk-forward fold).
        n_bootstrap:  Number of bootstrap replicates.
        block_length: Mean block length (geometric distribution).

    Returns:
        Array of shape (n_bootstrap,) with bootstrap mean IC values.
    """
    x   = np.asarray(ic_series, dtype=float)
    T   = len(x)
    p   = 1.0 / block_length     # P(start new block) at each step
    rng = np.random.default_rng(42)

    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        indices = []
        while len(indices) < T:
            start = int(rng.integers(0, T))
            blen  = int(rng.geometric(p))          # geometric block length
            for k in range(blen):
                if len(indices) >= T:
                    break
                indices.append((start + k) % T)    # circular wrap
        boot_means[b] = x[np.array(indices[:T])].mean()

    return boot_means
