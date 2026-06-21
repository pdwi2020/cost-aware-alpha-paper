"""Online FDR via LORD++ applied to calibrated bootstrap p-values (fdr_results.parquet).

Primary procedure
-----------------
BH and BHY on calibrated block-bootstrap p-values (``boot_p`` column of
``fdr_results.parquet``) are the PRIMARY multiple-testing procedures for this
paper (see config/spec.yaml §fdr and R3 rewrite).  LORD++ is the ONLINE
complement: it processes features in the pre-registered spec order (fixed in
advance, not data-dependent) and provides a proven mFDR / FDR guarantee under
arbitrary dependence.  Its output is reported as a supporting check; do NOT
interpret LORD++ discoveries alone as the main confirmatory result.

LORD++ reference
----------------
Javanmard, A. & Montanari, A. (2018).
  "Online Rules for Control of False Discovery Rate and False Discovery
   Exceedance."  Annals of Statistics 46(2):526-554.
  arXiv:1603.09000

Ramdas, A., Yang, F., Wainwright, M. J., & Jordan, M. I. (2017).
  "Online control of the false discovery rate with decaying memory."
  NeurIPS 2017.  arXiv:1710.00499

The LORD++ recursion used here follows Algorithm 2 of Javanmard & Montanari
(2018).  With γ_j ∝ 1 / (j+1)^1.6 (normalised to sum to 1), the procedure
controls the mFDR (marginal FDR) at level α under arbitrary dependence among
the p-values, and controls FDR under independence.  The feature ordering is
the pre-registered canonical order from feature_spec.KEPT_FEATURES +
ADDED_INTERACTIONS, which is fixed before any data is seen.

Outputs
-------
data/processed/online_fdr.parquet
    Columns: track, feature, boot_p, lord_alpha_j, lord_rejected, order_idx
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.feature_spec import KEPT_FEATURES, ADDED_INTERACTIONS
import src.manifest as manifest

DATA = ROOT / "data" / "processed"
OUT  = DATA / "online_fdr.parquet"

# Pre-registered feature order (spec-fixed; must not be changed after registration).
_PREREGISTERED_ORDER: list[str] = KEPT_FEATURES + ADDED_INTERACTIONS


# ---------------------------------------------------------------------------
# LORD++ implementation
# ---------------------------------------------------------------------------

def _gamma_sequence(m: int, exponent: float = 1.6) -> np.ndarray:
    """Normalised γ weights: γ_j ∝ 1/(j+1)^exponent, summing to 1.

    j is 0-indexed here; in the paper it is 1-indexed.  The sum is over
    j = 0 .. m-1, i.e. the first m terms.  We normalise to make the weights
    proper even for finite m; for large m the tail is negligible.
    """
    j = np.arange(1, m + 1, dtype=float)   # 1-indexed j = 1..m
    w = 1.0 / (j ** exponent)
    return w / w.sum()


def lord_plus_plus(
    p_values: Sequence[float],
    alpha: float = 0.10,
    gamma: Sequence[float] | None = None,
) -> list[bool]:
    """LORD++ online FDR procedure (Javanmard & Montanari 2018, Alg. 2).

    Features are tested in the order they appear in ``p_values``.  The order
    must be pre-specified and data-independent (e.g. the canonical pre-
    registered feature order from feature_spec).

    Parameters
    ----------
    p_values:
        Ordered sequence of p-values to test.
    alpha:
        FDR target level (e.g. 0.10).
    gamma:
        Wealth-allocation sequence γ_1, …, γ_m (1-indexed).  Must be
        non-negative and sum to ≤ 1.  If None, uses 1/(j+1)^1.6 normalised.

    Returns
    -------
    list[bool]
        ``rejected[j]`` is True iff hypothesis j is rejected (0-indexed).

    Notes
    -----
    LORD++ (Algorithm 2, Javanmard & Montanari 2018) maintains an alpha-
    wealth W_t and allocates spending level α_t = γ_{t - τ_l} * W_{τ_l^-}
    where τ_l is the time of the l-th rejection and W_{τ_l^-} is the wealth
    just before that rejection.  The initial wealth is W_0 = α * γ_1.

    For practical implementation with a fixed-length sequence we compute
    α_j using the recursive form:
        W_0  = α * γ_1
        For each step j (1-indexed):
            α_j  = γ_{j - τ_{R(j)}} * W_{τ_{R(j)}^-}
            if p_j <= α_j: reject; W_j = W_{j-1} + α * γ_{j - τ_{R(j)} + 1}
            else:          do not reject; W_j = W_{j-1} - α_j

    where R(j) is the number of rejections up to j-1, and τ_{R(j)} is the
    time of the last rejection before j (or 0 if none).
    """
    p = list(p_values)
    m = len(p)
    if m == 0:
        return []

    if gamma is None:
        gam = _gamma_sequence(m + 1)   # need up to index m (1-indexed)
    else:
        gam = np.asarray(gamma, dtype=float)

    # W_0 = alpha * gamma_1  (LORD++ initial wealth)
    W = float(alpha * gam[0])

    rejected: list[bool] = []
    # Track wealth snapshots at each rejection time
    # tau_last = time of last rejection (1-indexed; 0 if none)
    # W_before_last = wealth just before last rejection
    tau_last: int = 0
    W_before_last: float = W   # W_0

    for j in range(1, m + 1):   # 1-indexed
        # Distance from last rejection event
        delta = j - tau_last

        if delta - 1 < len(gam):
            alpha_j = float(gam[delta - 1] * W_before_last)
        else:
            alpha_j = 0.0   # gamma exhausted; effectively no budget

        # Cannot spend more than current wealth (safeguard against rounding)
        alpha_j = min(alpha_j, W)
        alpha_j = max(alpha_j, 0.0)

        pj = float(p[j - 1])
        if pj <= alpha_j:
            rejected.append(True)
            # Wealth increases by alpha * gamma_{delta+1}
            bonus_idx = delta   # gamma[delta] = gamma_{delta+1} (0-indexed)
            bonus = float(alpha * gam[bonus_idx]) if bonus_idx < len(gam) else 0.0
            W_before_last = W   # capture wealth just BEFORE this rejection
            tau_last = j
            W = W - alpha_j + bonus
        else:
            rejected.append(False)
            W = W - alpha_j

        W = max(W, 0.0)   # wealth cannot go negative (numerical safeguard)

    return rejected


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def _preregistered_order_for(features: list[str]) -> list[str]:
    """Return the pre-registered order restricted to features actually present."""
    present = set(features)
    ordered = [f for f in _PREREGISTERED_ORDER if f in present]
    # Append any features not in the pre-registered list last (alphabetical)
    extras = sorted(present - set(ordered))
    return ordered + extras


def main() -> None:
    print(
        "=== LORD++ Online FDR (calibrated bootstrap p-values) ===\n"
        "NOTE: BH/BHY on boot_p (fdr_results.parquet) is the PRIMARY procedure.\n"
        "      LORD++ is the online complement with a proven mFDR/FDR guarantee.\n"
    )

    fdr = pd.read_parquet(DATA / "fdr_results.parquet")

    # Validate required columns
    required = {"track", "feature", "boot_p"}
    missing = required - set(fdr.columns)
    if missing:
        raise ValueError(
            f"fdr_results.parquet is missing columns: {missing}. "
            "Run the R3 FDR pipeline first."
        )

    all_results: list[pd.DataFrame] = []

    for track in sorted(fdr["track"].unique()):
        sub = fdr[fdr["track"] == track].copy()
        features = sub["feature"].tolist()

        # Sort into pre-registered canonical order (data-independent)
        ordered_features = _preregistered_order_for(features)
        order_map = {f: i for i, f in enumerate(ordered_features)}
        sub = sub.copy()
        sub["order_idx"] = sub["feature"].map(order_map)
        sub = sub.sort_values("order_idx").reset_index(drop=True)

        p_vals = sub["boot_p"].tolist()
        rejected_flags = lord_plus_plus(p_vals, alpha=0.10)

        sub["lord_alpha_j"] = _compute_alpha_sequence(p_vals, alpha=0.10)
        sub["lord_rejected"] = rejected_flags
        sub["track"] = track

        n_selected = int(sum(rejected_flags))
        print(
            f"  {track}: {n_selected}/{len(features)} selected by LORD++ "
            f"(vs BH primary: {int(sub['bh_rejected'].sum()) if 'bh_rejected' in sub.columns else '?'})"
        )

        # Record to manifest
        manifest.record(
            key=f"online_fdr.{track}.n_selected",
            value=n_selected,
            stage="online_fdr",
            track=track,
            meta={"procedure": "LORD++", "alpha": 0.10, "order": "pre_registered_feature_spec"},
        )

        all_results.append(sub[["track", "feature", "order_idx", "boot_p",
                                 "lord_alpha_j", "lord_rejected"]])

    out = pd.concat(all_results, ignore_index=True)
    out.to_parquet(OUT, index=False)
    print(f"\nSaved → {OUT}")
    print(
        "\nReminder: LORD++ discoveries are a supporting online-FDR check.\n"
        "The PRIMARY inference is BH/BHY on calibrated bootstrap p-values.\n"
        "Do not overclaim LORD++ as a standalone confirmatory result."
    )


def _compute_alpha_sequence(p_values: list[float], alpha: float = 0.10) -> list[float]:
    """Return the per-step spending levels α_j used by LORD++ (for logging/output)."""
    m = len(p_values)
    if m == 0:
        return []

    gam = _gamma_sequence(m + 1)
    W = float(alpha * gam[0])
    tau_last = 0
    W_before_last = W

    alphas: list[float] = []
    for j in range(1, m + 1):
        delta = j - tau_last
        if delta - 1 < len(gam):
            alpha_j = float(gam[delta - 1] * W_before_last)
        else:
            alpha_j = 0.0
        alpha_j = min(alpha_j, W)
        alpha_j = max(alpha_j, 0.0)
        alphas.append(alpha_j)

        pj = float(p_values[j - 1])
        if pj <= alpha_j:
            bonus_idx = delta
            bonus = float(alpha * gam[bonus_idx]) if bonus_idx < len(gam) else 0.0
            W_before_last = W
            tau_last = j
            W = W - alpha_j + bonus
        else:
            W = W - alpha_j
        W = max(W, 0.0)

    return alphas


if __name__ == "__main__":
    main()
