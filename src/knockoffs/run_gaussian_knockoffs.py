"""E2: Second-order (Gaussian) model-X knockoffs as a CPU comparator to the
LSTM-VAE Cost-Aware Knockoffs (CAK).

Purpose: the LSTM-VAE knockoff generator collapsed to the marginal mean
(trivial reconstruction), producing near-constant knockoffs and an inflated W_j.
This experiment asks whether the FDR-gate failure was a property of that
degenerate generator or a structural barrier of the cost-aware importance
statistic. We generate NON-degenerate, covariance-matched Gaussian knockoffs
(Candès et al. 2018, equicorrelated construction):

    X ~ N(0, Σ);   X̃ | X ~ N(X(I - Σ⁻¹D),  2D - DΣ⁻¹D),   D = diag(s),
    s_j = min(1, 2·λ_min(Σ)),

then reuse the same cost-aware importance W_j = Z_j - Z̃_j, the knockoff(+)
filter, and the FDR-validity-gate design from run_cost_aware_knockoffs.py.

Output: data/processed/cak_gaussian_{importance,results,validity}.parquet
Run:    python3 -u src/knockoffs/run_gaussian_knockoffs.py [--n-sims 100]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.run_ta_fdr import (
    build_single_feature_signal, _fast_net_sharpe,
    load_ohlcv_matrices, estimate_ic_signs,
)
from src.fdr.run_fdr import get_surviving_features
from src.backtest.portfolio import PortfolioSimulator
from src.knockoffs.run_cost_aware_knockoffs import (
    compute_cost_aware_importance, knockoff_plus_filter,
    FEATURES_PATH, OHLCV_PATH, FDR_PATH, SHAP_PATH, CFG_PATH,
    IS_START, IS_END, FDR_Q, N_NULL_FEATS,
)

OUT_IMPORTANCE = ROOT / "data" / "processed" / "cak_gaussian_importance.parquet"
OUT_RESULTS    = ROOT / "data" / "processed" / "cak_gaussian_results.parquet"
OUT_VALIDITY   = ROOT / "data" / "processed" / "cak_gaussian_validity.parquet"


def log(m): print(m, flush=True)


def gaussian_knockoffs(X: np.ndarray, rng) -> np.ndarray:
    """Equicorrelated second-order model-X knockoffs for an (n×p) matrix.

    Columns are standardised internally; returns knockoffs in standardised space.
    """
    n, p = X.shape
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    Sigma = np.corrcoef(Xs, rowvar=False)
    Sigma = np.nan_to_num(Sigma, nan=0.0)
    np.fill_diagonal(Sigma, 1.0)

    lam_min = max(float(np.linalg.eigvalsh(Sigma).min()), 1e-4)
    s = min(1.0, 2.0 * lam_min) * np.ones(p)
    D = np.diag(s)
    Sinv = np.linalg.pinv(Sigma)

    M = np.eye(p) - Sinv @ D               # conditional-mean operator (row-vector form)
    V = 2 * D - D @ Sinv @ D               # conditional covariance
    V = (V + V.T) / 2.0
    w, Q = np.linalg.eigh(V)
    w = np.clip(w, 0.0, None)
    L = Q @ np.diag(np.sqrt(w))            # V = L Lᵀ

    Z = rng.standard_normal((n, p))
    return Xs @ M + Z @ L.T                # standardised knockoffs (n×p)


def build_knockoff_df(feat_df, features, rng):
    """Generate the Gaussian knockoff panel aligned to the IS feature index."""
    dates = feat_df.index.get_level_values("date")
    is_mask = (dates >= pd.Timestamp(IS_START)) & (dates <= pd.Timestamp(IS_END))
    feat_is = feat_df.loc[is_mask, features]
    X = feat_is.fillna(0.0).values
    X_tilde = gaussian_knockoffs(X, rng)
    return pd.DataFrame(
        X_tilde, index=feat_is.index, columns=[f + "_kn" for f in features]
    )


def gaussian_fdr_gate(feat_df, ohlcv, sim, cfg, n_sims, seed=42):
    """Empirical FDR over known-null Gaussian features with Gaussian knockoffs."""
    log(f"\n  FDR-validity gate (Gaussian): {n_sims} sims, {N_NULL_FEATS} null features …")
    rng = np.random.default_rng(seed)
    dates = feat_df.index.get_level_values("date")
    is_mask = (dates >= pd.Timestamp(IS_START)) & (dates <= pd.Timestamp(IS_END))
    feat_is_idx = feat_df.loc[is_mask].index
    null_names = [f"NULL_{i:02d}" for i in range(N_NULL_FEATS)]
    all_tickers = feat_is_idx.get_level_values("ticker").unique().tolist()
    returns, sigma, adv = load_ohlcv_matrices(ohlcv, all_tickers, IS_START, IS_END)

    n_rows = len(feat_is_idx)
    fdrs = []
    for sim_i in range(n_sims):
        noise = pd.DataFrame(
            rng.standard_normal((n_rows, N_NULL_FEATS)),
            index=feat_is_idx, columns=null_names,
        )
        noise = noise.groupby(level="date").transform(lambda g: (g - g.mean()) / (g.std() + 1e-8))
        kn = gaussian_knockoffs(noise.fillna(0.0).values, rng)
        kn_df = pd.DataFrame(kn, index=feat_is_idx, columns=null_names)

        w_null = []
        for nm in null_names:
            z_real = _fast_net_sharpe(
                build_single_feature_signal(noise, nm, 1.0, IS_START, IS_END),
                returns, sigma, adv, spread_bps=sim.spread_bps, impact_coeff=sim.impact_coeff,
                aum_dollars=cfg.get("aum_dollars", 1e8), min_adv_dollars=cfg.get("min_adv_dollars", 1e6),
            )
            z_kn = _fast_net_sharpe(
                build_single_feature_signal(kn_df, nm, 1.0, IS_START, IS_END),
                returns, sigma, adv, spread_bps=sim.spread_bps, impact_coeff=sim.impact_coeff,
                aum_dollars=cfg.get("aum_dollars", 1e8), min_adv_dollars=cfg.get("min_adv_dollars", 1e6),
            )
            w_null.append(z_real - z_kn if not (np.isnan(z_real) or np.isnan(z_kn)) else np.nan)

        w_arr = np.array(w_null)
        _, selected = knockoff_plus_filter(w_arr, q=FDR_Q)
        fdrs.append(selected.sum() / max(1, selected.sum()) if selected.sum() else 0.0)
        if (sim_i + 1) % 20 == 0:
            log(f"    sim {sim_i+1}/{n_sims}  running_FDR={np.mean(fdrs):.3f}")

    emp = float(np.mean(fdrs)) if fdrs else np.nan
    return {"empirical_fdr": emp, "q_nominal": FDR_Q,
            "gate_pass": (not np.isnan(emp)) and emp <= FDR_Q * 1.1, "n_sims": len(fdrs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="track_b")
    ap.add_argument("--n-sims", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    log("=== E2: Gaussian (second-order) Knockoff Comparator ===\n")
    rng = np.random.default_rng(args.seed)

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    features = get_surviving_features(SHAP_PATH)
    log(f"  Features: {len(features)}")
    feat_df = pd.read_parquet(FEATURES_PATH)
    ohlcv   = pd.read_parquet(OHLCV_PATH)
    sim = PortfolioSimulator(config_path=str(CFG_PATH))
    target_col = "target_track_b" if args.track == "track_b" else "target_track_a"
    ic_signs = estimate_ic_signs(feat_df, features, target_col, IS_START, IS_END)

    log("\n[1] Generating Gaussian knockoff panel …")
    knockoff_df = build_knockoff_df(feat_df, features, rng)
    log(f"    shape={knockoff_df.shape}")

    log("\n[2] Cost-aware importance W_j …")
    imp = compute_cost_aware_importance(feat_df, knockoff_df, features, ohlcv, sim, cfg, ic_signs, args.track)

    log("\n[3] Knockoff(+) filter (q=0.10) …")
    W = imp["W_j"].values
    tau, selected = knockoff_plus_filter(W, q=FDR_Q)
    imp["gaussian_cak_selected"] = selected
    imp["track"] = args.track
    n_pos = int((imp["W_j"] > 0).sum())
    log(f"    W_j range [{imp['W_j'].min():.3f}, {imp['W_j'].max():.3f}], {n_pos}/{len(imp)} positive")
    log(f"    tau={tau:.3f}, Gaussian-CAK selected: {int(selected.sum())}/{len(imp)}")
    imp.to_parquet(OUT_IMPORTANCE)
    imp.to_parquet(OUT_RESULTS)

    log("\n[4] FDR-validity gate …")
    gate = gaussian_fdr_gate(feat_df, ohlcv, sim, cfg, n_sims=args.n_sims, seed=args.seed)
    pd.DataFrame([gate]).to_parquet(OUT_VALIDITY)
    log(f"    empirical_FDR={gate['empirical_fdr']:.3f} (q={FDR_Q}) → "
        f"gate {'PASS' if gate['gate_pass'] else 'FAIL'}")

    # Compare to LSTM-VAE CAK
    try:
        vae = pd.read_parquet(ROOT / "data" / "processed" / "cak_importance.parquet")
        log(f"\n  Comparison — LSTM-VAE CAK: W_j range "
            f"[{vae['W_j'].min():.3f}, {vae['W_j'].max():.3f}], "
            f"{int((vae['W_j']>0).sum())}/{len(vae)} positive")
        log(f"  Comparison — Gaussian CAK: W_j range "
            f"[{imp['W_j'].min():.3f}, {imp['W_j'].max():.3f}], {n_pos}/{len(imp)} positive")
    except Exception as e:
        log(f"  (VAE comparison unavailable: {e})")

    log(f"\nElapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
