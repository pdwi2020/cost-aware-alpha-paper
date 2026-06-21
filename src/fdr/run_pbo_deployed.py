"""PBO via CSCV on the DEPLOYED Track A strategy's NET returns.

Background / reviewer motivation
---------------------------------
The existing run_pbo.py evaluates gross model ICs from walk-forward folds.
The reviewer correctly noted that CAVAL's deployed strategy is a cost-aware
tradable portfolio, so overfitting assessment must use NET-of-cost performance.

Design (Bailey, Borwein, López de Prado & Zhu 2017)
----------------------------------------------------
Candidate strategy set:
    The 30 pre-registered single-feature cross-sectional long-short books,
    each net of Almgren–Chriss (square-root participation) costs, oriented
    by their IS IC sign so each book is pointed at its in-sample edge.
    Source: same component construction as run_ta_fdr._precompute_pnl_components.

CSCV:
    Split T_IS trading days into S=16 contiguous equal blocks.
    Enumerate all C(16,8) = 12870 IS/OOS splits.
    For each split:
      - Rank 30 strategies by IS net Sharpe.
      - Take the IS-best; find its OOS net Sharpe rank r ∈ (0,1).
      - Logit λ = log(r / (1-r)).  λ < 0 ↔ IS-best below OOS median → overfit.
    PBO = fraction of splits with λ ≤ 0.

Also reported:
    - Logit distribution summary (mean, std, P(λ>0)).
    - Performance-degradation slope: OOS net Sharpe ~ IS net Sharpe (OLS slope).
    - DSR trial count note (30 single-feature strategies).

Outputs:
    data/processed/pbo_deployed.parquet  — λ distribution + summary
    Manifest keys (stage="pbo", track="track_a"):
        pbo.deployed.track_a.pbo
        pbo.deployed.track_a.n_strategies
        pbo.deployed.track_a.degradation_slope

Run:
    python3 -u src/fdr/run_pbo_deployed.py

For fast smoke test with fewer CSCV splits (random subset):
    python3 -u src/fdr/run_pbo_deployed.py --max-splits 500
"""

import argparse
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Reuse existing helpers — do NOT duplicate
from src.backtest.run_ta_fdr import (
    IS_END,
    IS_START,
    REBAL_FREQ,
    VOL_WINDOW,
    _precompute_pnl_components,
    build_single_feature_signal,
    load_ohlcv_matrices,
)
from src.features.feature_spec import feature_columns as get_feature_columns
from src.manifest import record as manifest_record

# ── Paths ─────────────────────────────────────────────────────────────────────

FEAT_PATH  = ROOT / "data" / "processed" / "features_all.parquet"
OHLCV_PATH = ROOT / "data" / "processed" / "daily_ohlcv.parquet"
FDR_PATH   = ROOT / "data" / "processed" / "fdr_results.parquet"
CFG_PATH   = ROOT / "configs" / "backtest.yaml"
OUT_PATH   = ROOT / "data" / "processed" / "pbo_deployed.parquet"

TRACK          = "track_a"
N_BLOCKS       = 16           # CSCV: split into 16 contiguous equal blocks
HALF_BLOCKS    = N_BLOCKS // 2  # 8 IS blocks out of 16


def log(msg: str) -> None:
    print(msg, flush=True)


# ── CSCV core ─────────────────────────────────────────────────────────────────

def cscv_pbo(
    returns_matrix: np.ndarray,
    n_blocks: int = 16,
    max_splits: int | None = None,
    rng_seed: int = 42,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Probability of Backtest Overfitting via CSCV.

    Implements Bailey, Borwein, López de Prado & Zhu (2017) exactly:
      - Split T rows into n_blocks contiguous equal-sized blocks.
      - Enumerate C(n_blocks, n_blocks//2) IS/OOS splits.
      - Rank M strategies by IS net Sharpe; find IS-best.
      - Record OOS net Sharpe rank → logit λ.
      - PBO = fraction of splits with λ ≤ 0.
      - Performance-degradation slope = OLS slope of OOS_SR ~ IS_SR.

    Parameters
    ----------
    returns_matrix : np.ndarray of shape (T, M)
        Daily NET return series for M candidate strategies over T trading days.
        Higher values = better.
    n_blocks : int
        Number of contiguous equal blocks to partition T rows into (default 16).
    max_splits : int or None
        If set, randomly subsample at most max_splits combinations (for speed).
        None = use all C(n_blocks, n_blocks//2) combinations.
    rng_seed : int
        Random seed for reproducible subsampling (only used if max_splits set).

    Returns
    -------
    pbo : float
        Fraction of CSCV paths with λ ≤ 0.
    lambda_vals : np.ndarray
        Logit of OOS rank for each CSCV path.
    is_sharpes : np.ndarray
        IS net Sharpe of the IS-best strategy for each path.
    oos_sharpes : np.ndarray
        OOS net Sharpe of the IS-best strategy for each path.
    """
    T, M = returns_matrix.shape
    half = n_blocks // 2

    # --- Partition T rows into n_blocks contiguous equal blocks ---------------
    # Each block spans exactly block_size rows; leftover rows are trimmed.
    block_size = T // n_blocks
    block_indices = []
    for b in range(n_blocks):
        start = b * block_size
        end   = start + block_size
        block_indices.append(np.arange(start, end))

    # --- Enumerate IS/OOS splits ----------------------------------------------
    all_combos = list(combinations(range(n_blocks), half))

    if max_splits is not None and len(all_combos) > max_splits:
        rng = np.random.default_rng(rng_seed)
        idx_sample = rng.choice(len(all_combos), size=max_splits, replace=False)
        all_combos = [all_combos[i] for i in idx_sample]

    # Replace NaN with just below global min so NaN strategies rank last.
    global_min = np.nanmin(returns_matrix)
    mat = np.where(np.isnan(returns_matrix), global_min - 1e-6, returns_matrix)

    lambda_vals  = np.empty(len(all_combos))
    is_sharpes   = np.empty(len(all_combos))
    oos_sharpes  = np.empty(len(all_combos))

    for path_idx, is_block_ids in enumerate(all_combos):
        oos_block_ids = [b for b in range(n_blocks) if b not in is_block_ids]

        is_rows  = np.concatenate([block_indices[b] for b in is_block_ids])
        oos_rows = np.concatenate([block_indices[b] for b in oos_block_ids])

        is_mat  = mat[is_rows, :]   # (T_IS × M)
        oos_mat = mat[oos_rows, :]  # (T_OOS × M)

        # Net Sharpe = mean / std (annualised by √252; but √252 cancels in rank)
        is_mean  = is_mat.mean(axis=0)
        is_std   = is_mat.std(axis=0, ddof=1)
        oos_mean = oos_mat.mean(axis=0)
        oos_std  = oos_mat.std(axis=0, ddof=1)

        is_sr  = np.where(is_std  > 1e-10, is_mean  / is_std,  0.0)
        oos_sr = np.where(oos_std > 1e-10, oos_mean / oos_std, 0.0)

        # IS-best strategy
        n_star = int(np.argmax(is_sr))

        # Normalized OOS rank (fraction of strategies beaten): r ∈ (1/M, 1]
        ranks  = stats.rankdata(oos_sr, method="average")
        omega  = float(ranks[n_star]) / M
        omega  = np.clip(omega, 1e-6, 1.0 - 1e-6)
        lambda_vals[path_idx] = np.log(omega / (1.0 - omega))

        is_sharpes[path_idx]  = float(is_sr[n_star])
        oos_sharpes[path_idx] = float(oos_sr[n_star])

    pbo = float(np.mean(lambda_vals <= 0.0))
    return pbo, lambda_vals, is_sharpes, oos_sharpes


# ── Net return matrix construction ────────────────────────────────────────────

def build_net_return_matrix(
    features: list[str],
    ic_signs: pd.Series,
    feat_df: pd.DataFrame,
    ohlcv: pd.DataFrame,
    cfg: dict,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build (T_IS × M) matrix of daily net returns for M single-feature strategies.

    Each column f = daily net return of the long-short book for feature f,
    oriented by ic_signs[f] (so the book is pointed at its in-sample edge).

    Parameters
    ----------
    features : list of str
        Feature names (the 30 pre-registered features).
    ic_signs : pd.Series
        Sign (+1/-1) keyed by feature name, from fdr_results.parquet ic_bar.
    feat_df, ohlcv : DataFrames
        The feature panel and OHLCV data.
    cfg : dict
        Backtest config (spread_bps, impact_coeff, aum_dollars, min_adv_dollars).

    Returns
    -------
    R : np.ndarray of shape (T, M_valid)
        T = number of IS trading days; M_valid ≤ M (features with valid components).
    valid_features : list[str]
        The M_valid feature names corresponding to columns of R.
    common_dates : np.ndarray
        The T IS trading date array.
    """
    spread_bps      = cfg["spread_bps"]
    impact_coeff    = cfg["impact_coeff"]
    aum_dollars     = cfg.get("aum_dollars", 1e8)
    min_adv_dollars = cfg.get("min_adv_dollars", 1e6)

    all_tickers = feat_df.index.get_level_values("ticker").unique().tolist()
    returns, sigma, adv = load_ohlcv_matrices(ohlcv, all_tickers, IS_START, IS_END)

    # Common IS grid rows — determined by _precompute_pnl_components internals
    grid_dates = returns.index[
        (returns.index >= pd.Timestamp(IS_START))
        & (returns.index <= pd.Timestamp(IS_END))
    ]
    T = len(grid_dates)
    log(f"  IS grid: {T} trading days  ({IS_START} → {IS_END})")

    net_cols      = []
    valid_features = []

    for i, feat in enumerate(features):
        ic_sign = float(ic_signs.get(feat, 1.0))
        sig = build_single_feature_signal(feat_df, feat, ic_sign, IS_START, IS_END)
        c = _precompute_pnl_components(
            sig, returns, sigma, adv,
            spread_bps=spread_bps,
            impact_coeff=impact_coeff,
            aum_dollars=aum_dollars,
            min_adv_dollars=min_adv_dollars,
        )
        if c is None:
            log(f"    [{i+1}/{len(features)}] {feat}: SKIPPED (insufficient data)")
            continue

        pos_, ret_, costs_ = c
        net = (pos_ * ret_).sum(axis=1) - costs_   # (T,) daily net returns
        net_cols.append(net)
        valid_features.append(feat)

        if (i + 1) % 5 == 0 or (i + 1) == len(features):
            log(f"    [{i+1}/{len(features)}] {feat}: mean_net={net.mean():+.6f}")

    R = np.column_stack(net_cols)   # (T, M_valid)
    return R, valid_features, grid_dates.values


# ── Degradation slope ─────────────────────────────────────────────────────────

def degradation_slope(is_sharpes: np.ndarray, oos_sharpes: np.ndarray) -> float:
    """OLS slope of OOS net Sharpe regressed on IS net Sharpe.

    Slope < 1 → performance degradation from IS to OOS.
    Slope ≈ 0 → IS Sharpe has zero predictive power for OOS (pure overfit).
    Slope < 0 → IS-best strategies actually underperform OOS (strong overfit).
    """
    if len(is_sharpes) < 2:
        return float("nan")
    slope, _intercept, _r, _p, _se = stats.linregress(is_sharpes, oos_sharpes)
    return float(slope)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PBO via CSCV on deployed Track A net returns."
    )
    parser.add_argument(
        "--max-splits", type=int, default=None,
        help="Subsample at most N CSCV splits (default: all 12870)."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log("=== PBO via CSCV — Deployed Track A Net Returns ===\n")
    t0 = time.time()

    # ── Load config ──────────────────────────────────────────────────────────
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    # ── Load feature panel + OHLCV ───────────────────────────────────────────
    log("Loading feature data ...")
    feat_df = pd.read_parquet(FEAT_PATH)

    log("Loading OHLCV data ...")
    ohlcv = pd.read_parquet(OHLCV_PATH)

    # ── Feature set (canonical 30 pre-registered features) ───────────────────
    features = get_feature_columns(feat_df)
    log(f"\nFeature set: {len(features)} features")

    # ── IC signs from fdr_results.parquet (IS ic_bar sign) ───────────────────
    fdr_df    = pd.read_parquet(FDR_PATH)
    fdr_track = fdr_df[fdr_df["track"] == TRACK].set_index("feature")
    ic_signs  = fdr_track["ic_bar"].apply(lambda x: float(np.sign(x)) if x != 0 else 1.0)
    # Features not in FDR results → default sign +1
    ic_signs_full = pd.Series(
        {f: float(ic_signs.get(f, 1.0)) for f in features}
    )
    log(f"IC signs loaded: {(ic_signs_full == 1).sum()} positive, "
        f"{(ic_signs_full == -1).sum()} negative")

    # ── Build net return matrix R (T × 30) ───────────────────────────────────
    log("\nBuilding daily net-return matrix R (T_IS × M) ...")
    R, valid_features, common_dates = build_net_return_matrix(
        features, ic_signs_full, feat_df, ohlcv, cfg
    )
    T, M = R.shape
    log(f"\nNet-return matrix: {T} days × {M} strategies")
    log(f"  Daily net Sharpe range: "
        f"[{(R.mean(0) / R.std(0)).min():.3f}, {(R.mean(0) / R.std(0)).max():.3f}]")

    # ── CSCV ─────────────────────────────────────────────────────────────────
    from math import comb
    total_splits = comb(N_BLOCKS, HALF_BLOCKS)
    n_splits_used = args.max_splits if args.max_splits else total_splits
    log(f"\nCSCV: S={N_BLOCKS} blocks, C({N_BLOCKS},{HALF_BLOCKS})={total_splits} paths "
        f"({'using all' if not args.max_splits else f'subsampled to {n_splits_used}'})")

    pbo, lambda_vals, is_srs, oos_srs = cscv_pbo(
        R,
        n_blocks=N_BLOCKS,
        max_splits=args.max_splits,
        rng_seed=args.seed,
    )

    slope = degradation_slope(is_srs, oos_srs)
    n_paths = len(lambda_vals)

    # ── Summary ──────────────────────────────────────────────────────────────
    log(f"\n{'='*60}")
    log(f"  PBO (Deployed Track A — Net Returns)")
    log(f"{'='*60}")
    log(f"  PBO                = {pbo:.4f}")
    log(f"  n_strategies (M)   = {M}   (30 pre-registered single-feature books)")
    log(f"  n_CSCV_paths       = {n_paths}")
    log(f"  λ mean             = {lambda_vals.mean():.4f}")
    log(f"  λ std              = {lambda_vals.std():.4f}")
    log(f"  P(λ > 0)           = {(lambda_vals > 0).mean():.4f}")
    log(f"  Degradation slope  = {slope:.4f}  (OOS_SR ~ IS_SR OLS slope)")
    log(f"\n  DSR trial count note: 30 single-feature strategies were searched.")
    log(f"  DSR already computed and fails (0.12–0.21); not recomputed here.")

    if pbo >= 0.5:
        log(f"\n  INTERPRETATION: PBO={pbo:.3f} ≥ 0.50 → selection is at or above "
            f"coin-flip in OOS (HIGH overfitting risk).")
    elif pbo >= 0.30:
        log(f"\n  INTERPRETATION: PBO={pbo:.3f} ∈ [0.30,0.50) → moderate overfitting risk.")
    else:
        log(f"\n  INTERPRETATION: PBO={pbo:.3f} < 0.30 → selection is robust OOS.")

    # ── Save output parquet ───────────────────────────────────────────────────
    out_df = pd.DataFrame({
        "lambda":    lambda_vals,
        "is_sharpe": is_srs,
        "oos_sharpe": oos_srs,
    })
    out_df.to_parquet(OUT_PATH, index=False)
    log(f"\nSaved → {OUT_PATH}")

    # Separate summary row
    summary_df = pd.DataFrame([{
        "track":             TRACK,
        "pbo":               pbo,
        "n_strategies":      M,
        "n_cscv_paths":      n_paths,
        "lambda_mean":       float(lambda_vals.mean()),
        "lambda_std":        float(lambda_vals.std()),
        "lambda_p_pos":      float((lambda_vals > 0).mean()),
        "degradation_slope": slope,
        "n_blocks":          N_BLOCKS,
        "block_size_days":   T // N_BLOCKS,
        "is_period":         f"{IS_START}/{IS_END}",
    }])
    summary_path = ROOT / "data" / "processed" / "pbo_deployed_summary.parquet"
    summary_df.to_parquet(summary_path, index=False)
    log(f"Saved → {summary_path}")

    # ── Record to manifest ────────────────────────────────────────────────────
    try:
        manifest_record(
            "pbo.deployed.track_a.pbo",
            float(pbo),
            stage="pbo",
            track=TRACK,
            meta={
                "method": "CSCV",
                "n_blocks": N_BLOCKS,
                "n_strategies": M,
                "n_cscv_paths": n_paths,
                "performance_metric": "net_sharpe",
                "IS_period": f"{IS_START}/{IS_END}",
                "reference": "Bailey et al. 2017",
            },
        )
        manifest_record(
            "pbo.deployed.track_a.n_strategies",
            M,
            stage="pbo",
            track=TRACK,
            meta={"description": "30 pre-registered single-feature books (oriented by IS IC sign)"},
        )
        manifest_record(
            "pbo.deployed.track_a.degradation_slope",
            slope,
            stage="pbo",
            track=TRACK,
            meta={
                "description": "OLS slope of OOS_net_Sharpe ~ IS_net_Sharpe across CSCV paths",
                "interpretation": "< 1 = degradation; ≈ 0 = no predictive power; < 0 = reversal",
            },
        )
        log("\nManifest records written:")
        log(f"  pbo.deployed.track_a.pbo               = {pbo:.4f}")
        log(f"  pbo.deployed.track_a.n_strategies      = {M}")
        log(f"  pbo.deployed.track_a.degradation_slope = {slope:.4f}")
    except Exception as exc:
        log(f"  WARNING: manifest write failed: {exc}")

    log(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
