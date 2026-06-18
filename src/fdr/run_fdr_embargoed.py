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
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.fdr.bh_correction import compute_ic_tstats, benjamini_hochberg
from src.fdr.run_fdr import get_surviving_features

FEATURES_PATH = ROOT / "data" / "processed" / "features_all.parquet"
SHAP_PATH     = ROOT / "data" / "processed" / "shap_summary.parquet"
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


def compute_fold_ics_embargoed(
    df: pd.DataFrame,
    features: list,
    target_col: str,
    fold_dates: list,
) -> pd.DataFrame:
    """Per-fold Spearman IC with embargoed test windows."""
    dates   = df.index.get_level_values("date")
    targets = df[target_col]
    records = []
    for fd in fold_dates:
        te_mask = (dates >= pd.Timestamp(fd["test_start"])) & \
                  (dates <= pd.Timestamp(fd["test_end"]))
        valid   = te_mask & targets.notna()
        X_te    = df.loc[valid, features]
        y_te    = targets[valid].values
        row     = {"fold": fd["fold_id"], "n_test": int(valid.sum())}
        for feat in features:
            x = X_te[feat].values
            ok = ~np.isnan(x) & ~np.isnan(y_te)
            row[feat] = float(spearmanr(x[ok], y_te[ok])[0]) if ok.sum() >= 20 else np.nan
        records.append(row)
    return pd.DataFrame(records).set_index("fold")


def run_bh_embargoed(ic_df: pd.DataFrame, track: str) -> pd.DataFrame:
    mean_ic, t_stats, p_values = compute_ic_tstats(ic_df.drop(columns="n_test", errors="ignore"))
    reject, adj_p = benjamini_hochberg(p_values.values, q=FDR_Q)
    return pd.DataFrame({
        "track":    track,
        "feature":  ic_df.drop(columns="n_test", errors="ignore").columns.tolist(),
        "mean_ic":  mean_ic.values,
        "t_stat":   t_stats.values,
        "p_value":  p_values.values,
        "bh_adj_p": adj_p,
        "rejected": reject,
    }).sort_values(["rejected", "p_value"], ascending=[False, True])


def main():
    log(f"=== FDR with {EMBARGO_DAYS}-BDay Embargo (Robustness Check) ===\n")
    t0 = time.time()

    features   = get_surviving_features(SHAP_PATH)
    fold_dates = make_fold_dates_embargoed()

    log(f"  Surviving features: {len(features)}")
    log(f"  Folds:")
    for fd in fold_dates:
        log(f"    {fd['fold_id']}: test {fd['test_start']} → {fd['test_end']}  "
            f"(embargo={fd['embargo_bdays']} BDays, {fd['n_test'] if 'n_test' in fd else '?'} obs)")

    log("\nLoading features_all.parquet …")
    df = pd.read_parquet(FEATURES_PATH)
    df.index = df.index.set_levels(
        [df.index.levels[0], pd.to_datetime(df.index.levels[1])]
    )

    baseline_df = pd.read_parquet(BASELINE_PATH)

    all_results  = []
    compare_rows = []

    for track, target_col in [("track_b", "target_track_b"), ("track_a", "target_track_a")]:
        log(f"\n{'='*60}")
        log(f"  Track: {track.upper()}")
        log(f"{'='*60}")

        ic_df    = compute_fold_ics_embargoed(df, features, target_col, fold_dates)
        result   = run_bh_embargoed(ic_df, track)
        n_rej    = result["rejected"].sum()
        log(f"\n  BH rejected (embargoed, FDR q={FDR_Q}): {n_rej}/{len(features)}")
        rej_feats = result[result["rejected"]]["feature"].tolist()
        log(f"  Rejected: {rej_feats}")

        # Compare to baseline
        base_rej = set(baseline_df[(baseline_df["track"] == track) & baseline_df["rejected"]]["feature"])
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
