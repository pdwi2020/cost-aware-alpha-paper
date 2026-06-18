"""Week 6: Feature-level IC significance testing with BH FDR correction.

For each of the 35 SHAP-surviving features:
  1. Compute Spearman IC per walk-forward fold (test years 2013–2021) for
     both Track A (idiosyncratic) and Track B (raw) targets.
  2. t-test across 9 folds: H0: mean_IC = 0  (two-sided, df = 8).
  3. Benjamini-Hochberg at FDR q = 0.10.

Regime analysis: VIX ≤ 20 (calm) vs VIX > 20 (stressed) — pre-specified
split pooled into a single BH test across 2 × n_features hypotheses.

Note: Spearman IC is rank-based → monotone feature transforms (winsorize,
z-score) don't change IC values, so raw feature values are used directly.

Outputs:
    data/processed/fdr_results.parquet   — per-feature BH results (both tracks)
    data/processed/fdr_regime.parquet    — regime-conditional BH results

Run:
    python3 -u src/fdr/run_fdr.py
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

from src.models.model_suite import make_fold_dates
from src.fdr.bh_correction import compute_ic_tstats, benjamini_hochberg, bh_with_regime
from src.universe_paths import proc

FEATURES_PATH  = proc(ROOT, "features_all.parquet")
SHAP_PATH      = proc(ROOT, "shap_summary.parquet")
OUT_RESULTS    = proc(ROOT, "fdr_results.parquet")
OUT_REGIME     = proc(ROOT, "fdr_regime.parquet")

FDR_Q          = 0.10
VIX_CALM_THRESH = 20.0

# Features with near-zero SHAP across all models → excluded before BH
SHAP_DROPPED = {"corr_XLRE", "corr_XLC", "term_spread_x_mom"}


def log(msg): print(msg, flush=True)


def get_surviving_features(shap_path: Path) -> list:
    shap_df  = pd.read_parquet(shap_path)
    all_feat = shap_df["feature"].unique().tolist()
    surviving = [f for f in all_feat if f not in SHAP_DROPPED]
    return sorted(surviving)


def compute_fold_ics(
    df: pd.DataFrame,
    features: list,
    target_col: str,
    fold_dates: list,
    vix_col: str = "vix",
) -> tuple:
    """Compute per-fold Spearman IC for each feature against the target.

    Returns:
        ic_all:      pd.DataFrame (n_folds, n_features) — full test period IC
        ic_calm:     pd.DataFrame (n_folds, n_features) — calm days only
        ic_stressed: pd.DataFrame (n_folds, n_features) — stressed days only
    """
    dates   = df.index.get_level_values("date")
    targets = df[target_col]

    records_all      = []
    records_calm     = []
    records_stressed = []

    for fd in fold_dates:
        fid     = fd["fold_id"]
        te_mask = (dates >= pd.Timestamp(fd["test_start"])) & \
                  (dates <= pd.Timestamp(fd["test_end"]))
        valid   = te_mask & targets.notna()

        X_te = df.loc[valid, features]
        y_te = targets[valid].values

        # VIX-based regime classification (using lagged VIX in features)
        vix_vals     = df.loc[valid, vix_col].values if vix_col in df.columns else None
        calm_mask    = (vix_vals <= VIX_CALM_THRESH)    if vix_vals is not None else np.ones(len(y_te), dtype=bool)
        stressed_mask = ~calm_mask

        row_all = {"fold": fid}
        row_calm = {"fold": fid}
        row_stressed = {"fold": fid}

        for feat in features:
            x = X_te[feat].values

            def _ic(x_sub, y_sub):
                valid_sub = ~np.isnan(x_sub) & ~np.isnan(y_sub)
                if valid_sub.sum() < 20:
                    return np.nan
                return float(spearmanr(x_sub[valid_sub], y_sub[valid_sub])[0])

            row_all[feat]      = _ic(x, y_te)
            row_calm[feat]     = _ic(x[calm_mask],     y_te[calm_mask])
            row_stressed[feat] = _ic(x[stressed_mask], y_te[stressed_mask])

        records_all.append(row_all)
        records_calm.append(row_calm)
        records_stressed.append(row_stressed)

    mk = lambda recs: pd.DataFrame(recs).set_index("fold")
    return mk(records_all), mk(records_calm), mk(records_stressed)


def run_bh_on_ic(ic_all: pd.DataFrame, track: str, q: float) -> pd.DataFrame:
    """Run BH on per-feature IC t-statistics. Returns result DataFrame."""
    mean_ic, t_stats, p_values = compute_ic_tstats(ic_all)
    reject, adj_p              = benjamini_hochberg(p_values.values, q=q)

    return pd.DataFrame({
        "track":     track,
        "feature":   ic_all.columns.tolist(),
        "n_folds":   ic_all.notna().sum().values,
        "mean_ic":   mean_ic.values,
        "std_ic":    ic_all.std(ddof=1).values,
        "t_stat":    t_stats.values,
        "p_value":   p_values.values,
        "bh_adj_p":  adj_p,
        "rejected":  reject,
    }).sort_values(["rejected", "p_value"], ascending=[False, True])


def main():
    log("=== Week 6: Feature IC Significance (BH FDR) ===\n")
    t0 = time.time()

    features   = get_surviving_features(SHAP_PATH)
    fold_dates = make_fold_dates()
    log(f"  Surviving features after SHAP pre-screen: {len(features)}")
    log(f"  Dropped: {sorted(SHAP_DROPPED)}")
    log(f"  Folds: {[fd['fold_id'] for fd in fold_dates]}")
    log(f"  FDR q = {FDR_Q}\n")

    df = pd.read_parquet(FEATURES_PATH)
    df.index = df.index.set_levels(
        [df.index.levels[0], pd.to_datetime(df.index.levels[1])]
    )

    all_results = []
    all_regime  = []

    for track, target_col in [("track_b", "target_track_b"),
                               ("track_a", "target_track_a")]:
        log(f"{'='*55}")
        log(f"  Track: {track.upper()}")
        log(f"{'='*55}")

        ic_all, ic_calm, ic_stressed = compute_fold_ics(
            df, features, target_col, fold_dates
        )

        # --- Main BH test ---
        result = run_bh_on_ic(ic_all, track=track, q=FDR_Q)
        n_rejected = result["rejected"].sum()
        log(f"\n  BH rejected (FDR q={FDR_Q}): {n_rejected}/{len(features)} features")
        log("\n  Rejected features:")
        rej = result[result.rejected][["feature", "mean_ic", "t_stat", "p_value", "bh_adj_p"]]
        log(rej.round(4).to_string(index=False))
        log("\n  All features (sorted by p-value):")
        log(result[["feature","mean_ic","t_stat","p_value","bh_adj_p","rejected"]]
            .round(4).to_string(index=False))
        all_results.append(result)

        # --- Regime BH test ---
        regime_df = bh_with_regime(ic_calm, ic_stressed, q=FDR_Q)
        regime_df.insert(0, "track", track)
        n_reg_rej = regime_df["rejected"].sum()
        log(f"\n  Regime BH rejected: {n_reg_rej}/{2*len(features)} (feature×regime) pairs")
        calm_rej    = regime_df[regime_df.rejected & (regime_df.regime == "calm")]["feature"].tolist()
        stressed_rej = regime_df[regime_df.rejected & (regime_df.regime == "stressed")]["feature"].tolist()
        log(f"    Calm:     {calm_rej}")
        log(f"    Stressed: {stressed_rej}")
        all_regime.append(regime_df)

    # Save
    out_main   = pd.concat(all_results, ignore_index=True)
    out_regime = pd.concat(all_regime, ignore_index=True)
    out_main.to_parquet(OUT_RESULTS, index=False)
    out_regime.to_parquet(OUT_REGIME, index=False)
    log(f"\nSaved → {OUT_RESULTS}")
    log(f"Saved → {OUT_REGIME}")
    log(f"Total elapsed: {(time.time()-t0):.1f}s")


if __name__ == "__main__":
    main()
