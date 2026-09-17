"""R2 — Purged 5-day embargo FDR robustness check.

Replicates the Week 6 BH FDR analysis with a 5-business-day gap between the
train end and test start of each walk-forward fold. The standard folds have
zero buffer (Dec 31 → Jan 1), which allows short-term autocorrelation in the
5-day-forward target to leak slightly across the fold boundary.

Adding a 5-BDay embargo removes this artifact. If the BH-rejected feature set
is stable under the embargo, the original FDR result is robust. If features
flip in/out, it flags a fragility worth reporting.

Outputs:
    data/processed/fdr_embargoed.parquet        — per-feature BH results
    data/processed/fdr_embargo_comparison.parquet — added/removed features vs baseline

Run:
    python3 -u src/fdr/run_fdr_embargoed.py
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

from src.fdr.bh_correction import benjamini_hochberg, bhy_procedure
from src.fdr.run_fdr import (
    BLOCK_LENGTH_DAYS,
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
    compute_fold_ics,
    stationary_bootstrap_pvalue,
)
from src.features.feature_spec import feature_columns as _feature_columns
from src.data.lean_load import load_features_lean

FEATURES_PATH = ROOT / "data" / "processed" / "features_all.parquet"
BASELINE_PATH = ROOT / "data" / "processed" / "fdr_results.parquet"
OUT_EMBARGOED = ROOT / "data" / "processed" / "fdr_embargoed.parquet"
OUT_COMPARE   = ROOT / "data" / "processed" / "fdr_embargo_comparison.parquet"

FDR_Q        = 0.10
EMBARGO_DAYS = 5       # business days


def log(msg): print(msg, flush=True)


def make_fold_dates_embargoed(
    train_start: str = "2010-01-01",
    warmup_end_year: int = 2012,
    test_end_year: int = 2021,
    embargo_bdays: int = EMBARGO_DAYS,
) -> list:
    """Same expanding-window folds but with embargo_bdays gap before test start."""
    folds = []
    for test_year in range(warmup_end_year + 1, test_end_year + 1):
        raw_train_end = pd.Timestamp(f"{test_year - 1}-12-31")
        raw_test_start = pd.Timestamp(f"{test_year}-01-01")
        # Actual test start: advance by embargo_bdays business days from Jan 1
        test_start = raw_test_start + pd.offsets.BDay(embargo_bdays)
        folds.append({
            "fold_id":     str(test_year),
            "train_start": train_start,
            "train_end":   raw_train_end.strftime("%Y-%m-%d"),
            "test_start":  test_start.strftime("%Y-%m-%d"),
            "test_end":    f"{test_year}-12-31",
            "embargo_bdays": embargo_bdays,
        })
    return folds


def run_bh_embargoed(daily_ic: dict, features: list, track: str) -> pd.DataFrame:
    """BH/BHY on the same estimand and null as the headline test.

    This must mirror run_fdr.py exactly: the daily cross-sectional IC series,
    with p-values from the stationary block bootstrap. An earlier version used
    a one-sample t-test across the nine fold-mean ICs (df = 8). That test is
    badly anti-conservative here, because averaging within a fold removes the
    day-to-day variation the block bootstrap correctly retains: it reported 24
    of 30 Track B rejections against the headline test's 1. Comparing that
    count to the bootstrap baseline measured the change of estimator, not the
    effect of the embargo, which is the only thing this check is for.
    """
    boot_stats = {
        f: stationary_bootstrap_pvalue(
            daily_ic[f],
            block_len=BLOCK_LENGTH_DAYS,
            n=N_BOOTSTRAP,
            seed=BOOTSTRAP_SEED,
        )
        for f in features
    }
    p_values = np.array([boot_stats[f]["boot_p"] for f in features], dtype=float)
    bh_reject, bh_adj_p   = benjamini_hochberg(p_values, q=FDR_Q)
    bhy_reject, bhy_adj_p = bhy_procedure(p_values, q=FDR_Q)

    return pd.DataFrame({
        "track":        track,
        "feature":      features,
        "n_days":       [boot_stats[f]["n_days"] for f in features],
        "ic_bar":       [boot_stats[f]["ic_bar"] for f in features],
        "t_stat":       [boot_stats[f]["t_stat"] for f in features],
        "boot_p":       p_values,
        "bh_adj_p":     bh_adj_p,
        "bh_rejected":  bh_reject,
        "bhy_adj_p":    bhy_adj_p,
        "bhy_rejected": bhy_reject,
    }).sort_values(["bh_rejected", "boot_p"], ascending=[False, True])


def main():
    log(f"=== FDR with {EMBARGO_DAYS}-BDay Embargo (Robustness Check) ===\n")
    t0 = time.time()

    log("Loading features_all.parquet …")
    df = load_features_lean(FEATURES_PATH)
    df.index = df.index.set_levels(
        [df.index.levels[0], pd.to_datetime(df.index.levels[1])]
    )

    # Derive the canonical pre-registered feature set from the loaded frame.
    features   = _feature_columns(df)
    fold_dates = make_fold_dates_embargoed()

    log(f"  Surviving features: {len(features)}")
    log(f"  Estimand: daily cross-sectional IC; null: stationary block "
        f"bootstrap (block_len={BLOCK_LENGTH_DAYS}, n={N_BOOTSTRAP}, "
        f"seed={BOOTSTRAP_SEED}) — identical to run_fdr.py")
    log(f"  Folds:")
    for fd in fold_dates:
        log(f"    {fd['fold_id']}: test {fd['test_start']} → {fd['test_end']}  "
            f"(embargo={fd['embargo_bdays']} BDays)")

    baseline_df = pd.read_parquet(BASELINE_PATH)

    all_results  = []
    compare_rows = []

    for track, target_col in [("track_b", "target_track_b"), ("track_a", "target_track_a")]:
        log(f"\n{'='*60}")
        log(f"  Track: {track.upper()}")
        log(f"{'='*60}")

        daily_ic, _, _ = compute_fold_ics(df, features, target_col, fold_dates)
        result   = run_bh_embargoed(daily_ic, features, track)
        n_rej    = result["bh_rejected"].sum()
        log(f"\n  BH rejected (embargoed, FDR q={FDR_Q}): {n_rej}/{len(features)}  "
            f"| BHY {int(result['bhy_rejected'].sum())}/{len(features)}")
        rej_feats = result[result["bh_rejected"]]["feature"].tolist()
        log(f"  Rejected: {rej_feats}")

        # Compare to baseline
        base_rej = set(baseline_df[(baseline_df["track"] == track) & baseline_df["bh_rejected"]]["feature"])
        emb_rej  = set(rej_feats)
        added    = sorted(emb_rej - base_rej)
        removed  = sorted(base_rej - emb_rej)
        stable   = sorted(emb_rej & base_rej)

        log(f"\n  vs. baseline (no embargo):")
        log(f"    Stable (in both):  {len(stable)} features  {stable}")
        log(f"    Gained (emb only): {len(added)} features  {added}")
        log(f"    Lost (base only):  {len(removed)} features  {removed}")
        log(f"    Stability rate:    {len(stable)/max(len(base_rej),1)*100:.1f}% of original set retained")

        all_results.append(result)
        compare_rows.append({
            "track": track,
            "baseline_n":  len(base_rej),
            "embargoed_n": len(emb_rej),
            "stable_n":    len(stable),
            "gained_n":    len(added),
            "lost_n":      len(removed),
            "stability_pct": len(stable) / max(len(base_rej), 1) * 100,
            "stable_features":   ",".join(stable),
            "gained_features":   ",".join(added),
            "lost_features":     ",".join(removed),
        })

    out_df  = pd.concat(all_results, ignore_index=True)
    comp_df = pd.DataFrame(compare_rows)

    out_df.to_parquet(OUT_EMBARGOED, index=False)
    comp_df.to_parquet(OUT_COMPARE, index=False)
    log(f"\nSaved → {OUT_EMBARGOED}")
    log(f"Saved → {OUT_COMPARE}")

    log("\n=== SUMMARY ===")
    log(comp_df[["track","baseline_n","embargoed_n","stability_pct","lost_features"]].to_string(index=False))
    log(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
