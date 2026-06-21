"""Phase-3 FDR estimand: daily cross-sectional Spearman IC with
stationary block bootstrap inference (Politis-Romano).

For each feature f and each track (A, B):
  1. Daily cross-sectional IC: Spearman rank-corr across ELIGIBLE stocks on
     date t (s0_eligible == True), requiring >= min_names stocks.
     Yields daily series IC_f(t).
  2. Aggregate: IC_bar_f = mean over valid days.
  3. Stationary block bootstrap (block_length_days=21, n_samples=1000):
     resample the IC_f(t) series in blocks, recompute mean per draw → two-sided
     bootstrap p-value + 90% CI.  Also report naive t-stat for reference.
  4. BH and BHY at q=0.10 applied to the bootstrap p-values over the full
     pre-registered feature set.

Regime split uses `regime_vix` (pre-registered regime label, NOT the dropped
broadcast `vix` column).

Outputs:
    data/processed/fdr_results.parquet
    data/processed/fdr_regime.parquet

Run:
    python3 -u src/fdr/run_fdr.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.model_suite import make_fold_dates
from src.fdr.bh_correction import benjamini_hochberg, bhy_procedure
from src.universe_paths import proc
from src.features.feature_spec import feature_columns as _feature_columns
import src.manifest as manifest

FEATURES_PATH = proc(ROOT, "features_all.parquet")
OUT_RESULTS   = proc(ROOT, "fdr_results.parquet")
OUT_REGIME    = proc(ROOT, "fdr_regime.parquet")

FDR_Q              = 0.10
BLOCK_LENGTH_DAYS  = 21
N_BOOTSTRAP        = 1000
MIN_NAMES          = 20
VIX_CALM_THRESH    = 20.0   # used for regime_vix label
BOOTSTRAP_SEED     = 42


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Core estimand helpers
# ---------------------------------------------------------------------------

def daily_cross_sectional_ic(
    df: pd.DataFrame,
    feature: str,
    target: str,
    eligible_col: str = "s0_eligible",
    min_names: int = MIN_NAMES,
) -> pd.Series:
    """Compute daily cross-sectional Spearman IC for one feature.

    For each date t, compute Spearman rank correlation ACROSS stocks between
    feature[i,t] and target[i,t], using only rows where eligible_col is True.
    Requires >= min_names eligible rows on date t; otherwise IC(t) = NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with a MultiIndex (ticker, date) or DatetimeIndex level named
        "date".  Must contain `feature`, `target`, and `eligible_col`.
    feature : str
        Feature column name.
    target : str
        Target column name.
    eligible_col : str
        Column marking eligible rows (bool/int; truthy = eligible).
    min_names : int
        Minimum eligible names required to record a non-NaN IC.

    Returns
    -------
    pd.Series
        Daily IC values indexed by date (NaN where cross-section too small).
    """
    if "date" in df.index.names:
        dates = df.index.get_level_values("date")
    else:
        dates = df.index

    has_elig = eligible_col in df.columns

    records: dict = {}
    for d in np.unique(dates):
        day_mask = dates == d
        sub = df.loc[day_mask]

        if has_elig:
            elig_mask = sub[eligible_col].astype(bool)
            sub = sub.loc[elig_mask]

        feat_vals   = sub[feature].values.astype(float)
        target_vals = sub[target].values.astype(float)

        valid = ~np.isnan(feat_vals) & ~np.isnan(target_vals)
        n_valid = int(valid.sum())

        if n_valid < min_names:
            records[d] = np.nan
        else:
            ic_val, _ = spearmanr(feat_vals[valid], target_vals[valid])
            records[d] = float(ic_val)

    return pd.Series(records).sort_index()


def stationary_bootstrap_pvalue(
    ic_series: pd.Series,
    block_len: int = BLOCK_LENGTH_DAYS,
    n: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Stationary block bootstrap (Politis-Romano) for H0: E[IC] = 0.

    Resamples the daily IC series in blocks with geometrically distributed
    block lengths (mean = block_len), recomputes the mean for each draw, and
    returns the two-sided bootstrap p-value plus a 90% CI.

    Parameters
    ----------
    ic_series : pd.Series
        Daily IC values; NaN entries are dropped before bootstrapping.
    block_len : int
        Expected block length.  Geometric distribution with p = 1/block_len.
    n : int
        Number of bootstrap replications.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        ic_bar   : float  — mean of valid IC values
        boot_p   : float  — two-sided bootstrap p-value
        ci_low   : float  — 5th percentile of bootstrap distribution (90% CI)
        ci_high  : float  — 95th percentile of bootstrap distribution
        t_stat   : float  — naive t-statistic (IC_bar / (std / sqrt(n_days)))
        n_days   : int    — number of valid days used
    """
    vals = ic_series.dropna().values
    T = len(vals)

    if T < 2:
        nan = float("nan")
        return dict(ic_bar=nan, boot_p=nan, ci_low=nan, ci_high=nan,
                    t_stat=nan, n_days=T)

    ic_bar = float(np.mean(vals))
    std_ic = float(np.std(vals, ddof=1))
    t_stat = ic_bar / (std_ic / np.sqrt(T)) if std_ic > 1e-10 else float("nan")

    rng = np.random.default_rng(seed)
    p_geo = 1.0 / block_len   # geometric parameter

    boot_means = np.empty(n)
    for b in range(n):
        # Draw geometric block lengths until we have >= T observations.
        sample = []
        while len(sample) < T:
            # Geometric block length (at least 1)
            blen = int(np.clip(rng.geometric(p_geo), 1, T))
            start = int(rng.integers(0, T))
            for k in range(blen):
                sample.append(vals[(start + k) % T])
        boot_means[b] = np.mean(sample[:T])

    # Two-sided bootstrap p-value for H0: E[IC] = 0, via the RECENTERED (null)
    # bootstrap distribution. The resampled means are centered at the observed
    # ic_bar, so subtracting it yields draws under the null (mean 0); we then
    # ask how often a null draw is at least as extreme as the observed |ic_bar|.
    # The (1 + count)/(n + 1) plug-in keeps the p-value valid (never exactly 0)
    # and properly calibrated — a naive "fraction beyond -ic_bar" is
    # anti-conservative (~2*Phi(-2|Z|)) and would break FDR control.
    boot_null = boot_means - ic_bar
    n_extreme = int(np.sum(np.abs(boot_null) >= abs(ic_bar)))
    boot_p = float((1 + n_extreme) / (n + 1))

    ci_low  = float(np.percentile(boot_means, 5.0))
    ci_high = float(np.percentile(boot_means, 95.0))

    return dict(
        ic_bar=ic_bar,
        boot_p=boot_p,
        ci_low=ci_low,
        ci_high=ci_high,
        t_stat=t_stat,
        n_days=T,
    )


# ---------------------------------------------------------------------------
# Fold-level daily cross-sectional IC computation
# ---------------------------------------------------------------------------

def compute_fold_ics(
    df: pd.DataFrame,
    features: list,
    target_col: str,
    fold_dates: list,
    regime_col: str = "regime_vix",
) -> tuple:
    """Compute daily cross-sectional IC series for each feature across folds.

    For each fold's test window, computes the daily cross-sectional IC
    (across s0_eligible stocks) for each feature vs target_col.  Optionally
    splits by regime_col into calm/stressed sub-series.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature panel (MultiIndex ticker/date or date-level index).
    features : list
        Pre-registered feature columns.
    target_col : str
        "target_track_a" or "target_track_b".
    fold_dates : list
        Walk-forward fold definitions from make_fold_dates().
    regime_col : str
        Column with calm/stressed regime label.  Uses "regime_vix" per spec;
        NOT the dropped broadcast "vix".

    Returns
    -------
    Three dicts keyed by feature, each containing a pd.Series of daily ICs:
        daily_ic_all      : {feature: Series} over all eligible days in folds
        daily_ic_calm     : {feature: Series} calm days (regime_vix == "calm")
        daily_ic_stressed : {feature: Series} stressed days
    """
    if "date" in df.index.names:
        dates = df.index.get_level_values("date")
    else:
        dates = df.index

    has_regime = regime_col in df.columns

    # Initialise accumulators: list of daily ICs across all folds, per feature
    all_records: dict     = {f: {} for f in features}
    calm_records: dict    = {f: {} for f in features}
    stressed_records: dict = {f: {} for f in features}

    for fd in fold_dates:
        te_mask = (
            (dates >= pd.Timestamp(fd["test_start"]))
            & (dates <= pd.Timestamp(fd["test_end"]))
        )
        valid_target = df[target_col].notna()
        fold_mask = te_mask & valid_target

        fold_df = df.loc[fold_mask]
        if len(fold_df) == 0:
            continue

        fold_dates_unique = np.unique(
            fold_df.index.get_level_values("date")
            if "date" in fold_df.index.names
            else fold_df.index
        )

        # Determine regime for each date in this fold
        if has_regime:
            regime_per_date: dict = {}
            for d in fold_dates_unique:
                d_mask = (
                    fold_df.index.get_level_values("date") == d
                    if "date" in fold_df.index.names
                    else fold_df.index == d
                )
                vals = fold_df.loc[d_mask, regime_col].dropna()
                # Take first non-null regime value for that day
                regime_per_date[d] = vals.iloc[0] if len(vals) > 0 else None
        else:
            regime_per_date = {}

        for feat in features:
            if feat not in fold_df.columns:
                continue

            ic_series = daily_cross_sectional_ic(
                fold_df, feat, target_col, eligible_col="s0_eligible",
                min_names=MIN_NAMES,
            )

            for d, ic_val in ic_series.items():
                all_records[feat][d] = ic_val

                if has_regime:
                    reg = regime_per_date.get(d)
                    # "calm" when regime label is "calm" or numeric <= VIX_CALM_THRESH
                    try:
                        is_calm = (str(reg).lower() == "calm")
                    except Exception:
                        is_calm = False

                    if is_calm:
                        calm_records[feat][d] = ic_val
                    else:
                        stressed_records[feat][d] = ic_val

    # Convert to Series
    to_series = lambda d: pd.Series(d).sort_index() if d else pd.Series(dtype=float)
    daily_ic_all      = {f: to_series(all_records[f])      for f in features}
    daily_ic_calm     = {f: to_series(calm_records[f])     for f in features}
    daily_ic_stressed = {f: to_series(stressed_records[f]) for f in features}

    return daily_ic_all, daily_ic_calm, daily_ic_stressed


# ---------------------------------------------------------------------------
# BH/BHY applied to bootstrap p-values
# ---------------------------------------------------------------------------

def run_bh_bhy_on_boot_stats(
    boot_stats: dict,
    track: str,
    q: float = FDR_Q,
) -> pd.DataFrame:
    """Apply BH and BHY to per-feature bootstrap p-values.

    Parameters
    ----------
    boot_stats : dict
        {feature: {ic_bar, boot_p, ci_low, ci_high, t_stat, n_days}}
    track : str
    q : float

    Returns
    -------
    pd.DataFrame with one row per feature.
    """
    features = list(boot_stats.keys())
    p_values = np.array([boot_stats[f]["boot_p"] for f in features], dtype=float)

    bh_reject,  bh_adj_p  = benjamini_hochberg(p_values, q=q)
    bhy_reject, bhy_adj_p = bhy_procedure(p_values, q=q)

    rows = []
    for i, feat in enumerate(features):
        s = boot_stats[feat]
        rows.append({
            "track":       track,
            "feature":     feat,
            "n_days":      s["n_days"],
            "ic_bar":      s["ic_bar"],
            "t_stat":      s["t_stat"],
            "boot_p":      s["boot_p"],
            "ci_low":      s["ci_low"],
            "ci_high":     s["ci_high"],
            "bh_adj_p":    float(bh_adj_p[i]),
            "bh_rejected": bool(bh_reject[i]),
            "bhy_adj_p":   float(bhy_adj_p[i]),
            "bhy_rejected": bool(bhy_reject[i]),
        })

    return pd.DataFrame(rows).sort_values(
        ["bh_rejected", "boot_p"], ascending=[False, True]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log("=== Phase-3 FDR: Daily Cross-Sectional IC + Stationary Bootstrap ===\n")
    t0 = time.time()

    df = pd.read_parquet(FEATURES_PATH)
    df.index = df.index.set_levels(
        [df.index.levels[0], pd.to_datetime(df.index.levels[1])]
    )

    features   = _feature_columns(df)
    fold_dates = make_fold_dates()

    log(f"  Pre-registered features for BH/BHY test: {len(features)}")
    log(f"  Features: {features}")
    log(f"  Folds: {[fd['fold_id'] for fd in fold_dates]}")
    log(f"  FDR q = {FDR_Q}")
    log(f"  Bootstrap: stationary block, block_len={BLOCK_LENGTH_DAYS}, "
        f"n={N_BOOTSTRAP}, seed={BOOTSTRAP_SEED}\n")

    all_results = []
    all_regime  = []

    for track, target_col in [("track_b", "target_track_b"),
                               ("track_a", "target_track_a")]:
        log(f"{'='*60}")
        log(f"  Track: {track.upper()}  target: {target_col}")
        log(f"{'='*60}")

        daily_ic_all, daily_ic_calm, daily_ic_stressed = compute_fold_ics(
            df, features, target_col, fold_dates
        )

        # --- Bootstrap stats per feature (full test window) ---
        boot_stats_all = {}
        for feat in features:
            boot_stats_all[feat] = stationary_bootstrap_pvalue(
                daily_ic_all[feat],
                block_len=BLOCK_LENGTH_DAYS,
                n=N_BOOTSTRAP,
                seed=BOOTSTRAP_SEED,
            )

        result = run_bh_bhy_on_boot_stats(boot_stats_all, track=track, q=FDR_Q)
        n_bh_rej  = int(result["bh_rejected"].sum())
        n_bhy_rej = int(result["bhy_rejected"].sum())
        log(f"\n  BH  rejected: {n_bh_rej}/{len(features)} features")
        log(f"  BHY rejected: {n_bhy_rej}/{len(features)} features")
        log("\n  BH-selected features:")
        rej = result[result["bh_rejected"]][
            ["feature", "ic_bar", "t_stat", "boot_p", "bh_adj_p"]
        ]
        log(rej.round(4).to_string(index=False))
        log("\n  All features (sorted by boot_p):")
        log(result[["feature", "ic_bar", "t_stat", "boot_p",
                    "bh_adj_p", "bh_rejected", "bhy_rejected"]]
            .round(4).to_string(index=False))
        all_results.append(result)

        # Record to manifest
        bh_selected  = result[result["bh_rejected"]]["feature"].tolist()
        bhy_selected = result[result["bhy_rejected"]]["feature"].tolist()
        short_track  = track.split("_")[1].upper()   # "A" or "B"

        manifest.record(
            f"fdr.{track}.n_selected_bh", n_bh_rej,
            stage="fdr", track=short_track,
            meta={"estimand": "daily_cross_sectional_spearman_ic",
                  "bootstrap": "stationary_block",
                  "block_len": BLOCK_LENGTH_DAYS, "n_samples": N_BOOTSTRAP},
        )
        manifest.record(
            f"fdr.{track}.n_selected_bhy", n_bhy_rej,
            stage="fdr", track=short_track,
        )
        manifest.record(
            f"fdr.{track}.selected_features_bh", bh_selected,
            stage="fdr", track=short_track,
        )
        manifest.record(
            f"fdr.{track}.selected_features_bhy", bhy_selected,
            stage="fdr", track=short_track,
        )

        # Per-feature stats for manifest
        for feat in features:
            s = boot_stats_all[feat]
            manifest.record(
                f"fdr.{track}.feature.{feat}",
                {"ic_bar": s["ic_bar"], "boot_p": s["boot_p"],
                 "ci_low": s["ci_low"], "ci_high": s["ci_high"],
                 "n_days": s["n_days"]},
                stage="fdr", track=short_track,
            )

        # --- Regime bootstrap stats ---
        for regime_name, daily_ic_regime in [
            ("calm",     daily_ic_calm),
            ("stressed", daily_ic_stressed),
        ]:
            boot_stats_regime = {}
            for feat in features:
                boot_stats_regime[feat] = stationary_bootstrap_pvalue(
                    daily_ic_regime[feat],
                    block_len=BLOCK_LENGTH_DAYS,
                    n=N_BOOTSTRAP,
                    seed=BOOTSTRAP_SEED,
                )

            regime_result = run_bh_bhy_on_boot_stats(
                boot_stats_regime, track=track, q=FDR_Q
            )
            regime_result.insert(2, "regime", regime_name)
            all_regime.append(regime_result)

            n_reg_bh  = int(regime_result["bh_rejected"].sum())
            n_reg_bhy = int(regime_result["bhy_rejected"].sum())
            log(f"\n  Regime [{regime_name}]: BH {n_reg_bh}  BHY {n_reg_bhy} selected")

    # --- Save ---
    out_main   = pd.concat(all_results, ignore_index=True)
    out_regime = pd.concat(all_regime,  ignore_index=True)
    out_main.to_parquet(OUT_RESULTS, index=False)
    out_regime.to_parquet(OUT_REGIME, index=False)
    log(f"\nSaved → {OUT_RESULTS}")
    log(f"Saved → {OUT_REGIME}")
    log(f"Total elapsed: {(time.time()-t0):.1f}s")


if __name__ == "__main__":
    main()
