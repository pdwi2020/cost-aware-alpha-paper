"""R1 — Placebo FDR: BH should reject ≤1 of 10 i.i.d. Gaussian noise features.

Adds 10 Gaussian noise columns (seed 42) to the feature set before the BH
test. Under H0, the expected number of false discoveries at FDR q=0.10 is
≤ q × 10 = 1.0. Observing ≤1 rejection confirms the BH step is calibrated.
Observing >1 would flag inflated FDR (e.g., from cross-sectional or
temporal dependence that violates PRDS).

The noise features are cross-sectionally standardised per date (same
treatment as real features) to match the null exactly.

Outputs:
    data/processed/fdr_placebo.parquet — BH results including placebo features

Run:
    python3 -u src/fdr/run_fdr_placebo.py
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

from src.fdr.bh_correction import benjamini_hochberg
from src.fdr.run_fdr import (
    BLOCK_LENGTH_DAYS,
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
    compute_fold_ics,
    stationary_bootstrap_pvalue,
)
from src.models.model_suite import make_fold_dates
from src.features.feature_spec import feature_columns as _feature_columns
from src.data.lean_load import load_features_lean

FEATURES_PATH = ROOT / "data" / "processed" / "features_all.parquet"
OUT_PLACEBO   = ROOT / "data" / "processed" / "fdr_placebo.parquet"

FDR_Q        = 0.10
N_PLACEBO    = 10
PLACEBO_SEED = 42
PLACEBO_PREFIX = "PLACEBO_"


def log(msg): print(msg, flush=True)


def placebo_bh(daily_ic: dict, all_features: list, track: str) -> pd.DataFrame:
    """BH on the same estimand and null as the headline test.

    The point of this check is that the procedure the paper reports is
    calibrated, so it has to be that procedure: daily cross-sectional IC over
    Screen 0 eligible names, with stationary-block-bootstrap p-values. The
    earlier version instead used a t-test across the nine fold-mean ICs and
    skipped the Screen 0 mask, so a "PASS" certified an estimator that appears
    nowhere in the manuscript.
    """
    boot_stats = {
        f: stationary_bootstrap_pvalue(
            daily_ic[f],
            block_len=BLOCK_LENGTH_DAYS,
            n=N_BOOTSTRAP,
            seed=BOOTSTRAP_SEED,
        )
        for f in all_features
    }
    p_values = np.array([boot_stats[f]["boot_p"] for f in all_features], dtype=float)
    reject, adj_p = benjamini_hochberg(p_values, q=FDR_Q)

    return pd.DataFrame({
        "track":      track,
        "feature":    all_features,
        "is_placebo": [f.startswith(PLACEBO_PREFIX) for f in all_features],
        "mean_ic":    [boot_stats[f]["ic_bar"] for f in all_features],
        "t_stat":     [boot_stats[f]["t_stat"] for f in all_features],
        "p_value":    p_values,
        "bh_adj_p":   adj_p,
        "rejected":   reject,
    })


def inject_placebo_features(
    df: pd.DataFrame,
    n_placebo: int = N_PLACEBO,
    seed: int = PLACEBO_SEED,
) -> tuple:
    """Add n_placebo i.i.d. Gaussian columns, cross-sectionally standardised per date.

    Returns (augmented_df, placebo_feature_names).
    """
    rng = np.random.default_rng(seed)
    placebo_names = [f"{PLACEBO_PREFIX}{i:02d}" for i in range(n_placebo)]

    # Generate noise and add as columns
    n_rows = len(df)
    noise  = rng.standard_normal((n_rows, n_placebo))
    noise_df = pd.DataFrame(noise, index=df.index, columns=placebo_names)

    # Cross-sectional z-score per date to match feature treatment
    def xs_zscore(group):
        mu  = group.mean()
        sig = group.std()
        return (group - mu) / (sig + 1e-8)

    scaled_parts = []
    for col in placebo_names:
        scaled = noise_df[col].groupby(level="date").transform(xs_zscore)
        scaled_parts.append(scaled)
    scaled_df = pd.concat(scaled_parts, axis=1)
    scaled_df.columns = placebo_names

    augmented = pd.concat([df, scaled_df], axis=1)
    return augmented, placebo_names


def main():
    log(f"=== Placebo FDR (N={N_PLACEBO} Gaussian noise features, seed={PLACEBO_SEED}) ===\n")
    t0 = time.time()

    log("Loading features_all.parquet …")
    df = load_features_lean(FEATURES_PATH)
    df.index = df.index.set_levels(
        [df.index.levels[0], pd.to_datetime(df.index.levels[1])]
    )

    # Derive the canonical pre-registered feature set from the loaded frame.
    real_features = _feature_columns(df)
    fold_dates    = make_fold_dates()

    log(f"  Real features: {len(real_features)}")
    log(f"  Placebo features: {N_PLACEBO}  (prefix {PLACEBO_PREFIX!r})")
    log(f"  Total hypotheses: {len(real_features) + N_PLACEBO}")
    log(f"  BH q={FDR_Q} → expected placebo rejections ≤ {FDR_Q * N_PLACEBO:.1f}")
    log(f"  Estimand: daily cross-sectional IC over s0_eligible names; null: "
        f"stationary block bootstrap (block_len={BLOCK_LENGTH_DAYS}, "
        f"n={N_BOOTSTRAP}, seed={BOOTSTRAP_SEED}) — identical to run_fdr.py")

    log("Injecting placebo features …")
    df_aug, placebo_names = inject_placebo_features(df)
    log(f"  Injected: {placebo_names}")

    all_results = []

    for track, target_col in [("track_b", "target_track_b"), ("track_a", "target_track_a")]:
        log(f"\n{'='*60}")
        log(f"  Track: {track.upper()}")
        log(f"{'='*60}")

        all_feat = real_features + placebo_names
        daily_ic, _, _ = compute_fold_ics(df_aug, all_feat, target_col, fold_dates)
        result = placebo_bh(daily_ic, all_feat, track)

        n_real_rej    = result[~result["is_placebo"] & result["rejected"]].shape[0]
        n_placebo_rej = result[result["is_placebo"]  & result["rejected"]].shape[0]
        placebo_rej   = result[result["is_placebo"] & result["rejected"]]["feature"].tolist()

        log(f"\n  BH rejected real features: {n_real_rej}/{len(real_features)}")
        log(f"  BH rejected placebo features: {n_placebo_rej}/{N_PLACEBO}  ← KEY")
        if n_placebo_rej == 0:
            log(f"  ✓ PASS: 0 placebo rejected (FDR calibrated)")
        elif n_placebo_rej <= int(FDR_Q * N_PLACEBO):
            log(f"  ✓ PASS: {n_placebo_rej} ≤ {int(FDR_Q*N_PLACEBO)} expected (FDR calibrated)")
        else:
            log(f"  ✗ FAIL: {n_placebo_rej} > {FDR_Q*N_PLACEBO:.1f} expected (FDR inflated!)")
        if placebo_rej:
            log(f"  Rejected placebo names: {placebo_rej}")

        all_results.append(result)

    out_df = pd.concat(all_results, ignore_index=True)
    out_df.to_parquet(OUT_PLACEBO, index=False)
    log(f"\nSaved → {OUT_PLACEBO}")

    # Final summary
    log("\n=== SUMMARY: Placebo rejection counts ===")
    for track in ["track_a", "track_b"]:
        sub = out_df[out_df["track"] == track]
        nr = sub[~sub["is_placebo"] & sub["rejected"]].shape[0]
        np_ = sub[sub["is_placebo"] & sub["rejected"]].shape[0]
        status = "PASS" if np_ <= int(FDR_Q * N_PLACEBO) else "FAIL"
        log(f"  {track.upper()}: real_rej={nr}/{len(real_features)}, "
            f"placebo_rej={np_}/{N_PLACEBO}  [{status}]")

    log(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
