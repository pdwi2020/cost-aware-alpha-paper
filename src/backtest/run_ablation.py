"""P2.1 — Track B ablation study on OOS 2022-2024.

Ablation variants (signal reconstruction only, no model retraining):
  (A) Baseline  : BH-significant features × SHAP weights   (existing)
  (B) Equal wts : BH-significant features × equal weights
  (C) No FDR    : All 35 features × SHAP weights
  (D) No FDR/EW : All 35 features × equal weights

For each variant, run through PortfolioSimulator (weekly rebalance, base
cost model, $1M ADV filter) on 2022-2024 holdout. No model retraining.

The ADV filter ablation is not recomputed (it would require flagging bankrupt
stocks); the impact is documented inline from the CPWR example in the paper.

Outputs:
    data/processed/ablation_oos.parquet

Run:
    python3 -u src/backtest/run_ablation.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.portfolio import PortfolioSimulator, build_positions_screen0

FEAT_PATH  = ROOT / "data" / "processed" / "features_all.parquet"
FDR_PATH   = ROOT / "data" / "processed" / "fdr_results.parquet"
SHAP_PATH  = ROOT / "data" / "processed" / "shap_summary.parquet"
OHLCV_PATH = ROOT / "data" / "processed" / "daily_ohlcv_v3.parquet"
CFG_PATH   = ROOT / "configs" / "backtest.yaml"
OUT        = ROOT / "data" / "processed" / "ablation_oos.parquet"

OOS_START = "2022-01-01"
OOS_END   = "2024-12-31"
REBAL_FREQ = 5
VOL_WINDOW = 21
TRACK      = "track_b"


def log(msg): print(msg, flush=True)


def sharpe_ann(r):
    s  = np.array(r, dtype=float)
    sd = s.std(ddof=1)
    return float(s.mean() / sd * np.sqrt(252)) if sd > 1e-12 else np.nan


def build_signal(feat_df, features, shap_weights, fdr_sign, eq_weight=False):
    """Build composite signal as weighted sum of features."""
    # features: list of feature names; shap_weights: dict(feature → weight)
    # fdr_sign: dict(feature → sign of mean_ic); eq_weight: uniform 1/n
    sub = feat_df[[*features, "target_track_b"]].copy() if "target_track_b" in feat_df.columns \
          else feat_df[features].copy()
    # pivot to (date × ticker)
    sig = sub.reset_index()
    dates   = sig.iloc[:, 0] if isinstance(feat_df.index, pd.MultiIndex) else None

    # Work on MultiIndex (ticker, date) DataFrame
    for f in features:
        w = (1.0 / len(features)) if eq_weight else float(shap_weights.get(f, 1.0))
        s = float(fdr_sign.get(f, 1.0))
        sub[f] = sub[f] * w * s

    # Composite = sum across features per (date, ticker)
    composite = sub[features].sum(axis=1)
    composite.name = "signal"
    return composite


def unstack_signal(composite, feat_df):
    """Unstack MultiIndex composite series to (date × ticker) DataFrame."""
    # feat_df has MultiIndex (ticker, date) or (date, ticker)
    if isinstance(feat_df.index, pd.MultiIndex):
        idx = feat_df.index
        levels = idx.names
        if levels[0] != "date":
            # (ticker, date) → reset and pivot
            s = composite.copy()
            s.index = pd.MultiIndex.from_tuples(idx, names=levels)
            df = s.reset_index()
            return df.pivot(index=levels[1], columns=levels[0], values="signal")
        else:
            s = composite.copy()
            s.index = pd.MultiIndex.from_tuples(idx, names=levels)
            df = s.reset_index()
            return df.pivot(index=levels[0], columns=levels[1], values="signal")
    raise ValueError("Expected MultiIndex feat_df")


def main():
    t0 = time.time()
    log("=== P2.1 Track B OOS Ablation ===\n")

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    fdr_df  = pd.read_parquet(FDR_PATH)
    shap_df = pd.read_parquet(SHAP_PATH)
    # Lean load: the frozen-specification columns only. The full float64 panel
    # is 683 MB on disk and drove this 8 GB machine into swap.
    from src.data.lean_load import load_features_lean

    feat_df = load_features_lean(FEAT_PATH)
    ohlcv   = pd.read_parquet(OHLCV_PATH)

    # Normalise index
    if isinstance(feat_df.index, pd.MultiIndex):
        new_levels = []
        for lvl in feat_df.index.levels:
            try:
                new_levels.append(pd.to_datetime(lvl))
            except Exception:
                new_levels.append(lvl)
        feat_df.index = feat_df.index.set_levels(new_levels)

    # FDR-selected features for Track B
    fdr_b = fdr_df[fdr_df.track == TRACK]
    bh_features = fdr_b.loc[fdr_b.bh_rejected, "feature"].tolist()
    all_features = fdr_b["feature"].tolist()   # all 35 SHAP-surviving features

    if not all_features:
        log(f"  [SKIP] No features found for {TRACK} — aborting ablation.")
        return

    fdr_sign_bh  = {r.feature: np.sign(r.ic_bar) for _, r in fdr_b[fdr_b.bh_rejected].iterrows()}
    fdr_sign_all = {r.feature: np.sign(r.ic_bar) for _, r in fdr_b.iterrows()}

    # SHAP weights for Track B (average across models and folds)
    shap_b = shap_df[shap_df.track == TRACK] if "track" in shap_df.columns else shap_df
    shap_agg = shap_b.groupby("feature")["mean_abs_shap"].mean()
    shap_agg = shap_agg / shap_agg.sum()  # normalise
    shap_weights_bh  = {f: float(shap_agg.get(f, 1.0 / max(len(bh_features), 1)))  for f in bh_features}
    shap_weights_all = {f: float(shap_agg.get(f, 1.0 / max(len(all_features), 1))) for f in all_features}

    # OOS OHLCV data
    oos_mask = (
        (ohlcv["date"] >= pd.Timestamp(OOS_START)) &
        (ohlcv["date"] <= pd.Timestamp(OOS_END))
    )
    oos_ohlcv  = ohlcv[oos_mask]
    tickers    = oos_ohlcv["ticker"].unique().tolist()
    close_w    = oos_ohlcv.pivot(index="date", columns="ticker", values="close").sort_index()
    vol_w      = oos_ohlcv.pivot(index="date", columns="ticker", values="volume").fillna(0)
    returns    = close_w.pct_change()
    vol        = returns.rolling(VOL_WINDOW, min_periods=10).std()
    adv        = (close_w * vol_w).rolling(VOL_WINDOW, min_periods=10).mean()
    date_mask  = (returns.index >= pd.Timestamp(OOS_START)) & (returns.index <= pd.Timestamp(OOS_END))
    returns    = returns[date_mask]
    vol        = vol[date_mask]
    adv        = adv[date_mask]

    # OOS feature slice
    if isinstance(feat_df.index, pd.MultiIndex):
        lvl_names  = feat_df.index.names
        date_level = 0 if "date" in str(lvl_names[0]) else 1
        if date_level == 1:
            # (ticker, date)
            date_vals = feat_df.index.get_level_values(1)
        else:
            date_vals = feat_df.index.get_level_values(0)
        oos_feat = feat_df[
            (date_vals >= pd.Timestamp(OOS_START)) &
            (date_vals <= pd.Timestamp(OOS_END))
        ]
    else:
        raise ValueError("Expected MultiIndex")

    sim = PortfolioSimulator(config_path=str(CFG_PATH))
    min_adv = cfg.get("min_adv_dollars", 1e6)
    aum     = cfg.get("aum_dollars", 1e8)

    variants = [
        ("Baseline (BH + SHAP)",    bh_features,  shap_weights_bh,  fdr_sign_bh,  False),
        ("BH + Equal weights",       bh_features,  None,             fdr_sign_bh,  True),
        ("All features + SHAP",      all_features, shap_weights_all, fdr_sign_all, False),
        ("All features + Equal wts", all_features, None,             fdr_sign_all, True),
    ]

    results = []
    for label, feats, sw, fs, eq in variants:
        log(f"\n  {label}  ({len(feats)} features, {'equal wts' if eq else 'SHAP wts'})")
        available = [f for f in feats if f in oos_feat.columns]
        if not available:
            log("  ERROR: no available features!")
            continue

        # Build composite signal
        if isinstance(oos_feat.index, pd.MultiIndex):
            lvl_names = oos_feat.index.names
            date_level_name = next((n for n in lvl_names if "date" in str(n).lower()), lvl_names[0])
            ticker_level_name = [n for n in lvl_names if n != date_level_name][0]
        else:
            raise ValueError("Expected MultiIndex")

        col_sub = oos_feat[available].copy()
        for f in available:
            w = (1.0 / len(available)) if eq else float(sw.get(f, 1.0))
            sign = float(fs.get(f, 1.0))
            col_sub[f] = col_sub[f] * w * sign

        composite = col_sub.sum(axis=1)
        composite.name = "signal"
        df_sig = composite.reset_index()
        sig_wide = df_sig.pivot(index=date_level_name, columns=ticker_level_name, values="signal")
        sig_wide.index = pd.to_datetime(sig_wide.index)
        sig_wide = sig_wide.sort_index()

        # Align columns
        common_t = [t for t in sig_wide.columns if t in returns.columns]
        if not common_t:
            log("  ERROR: no common tickers!")
            continue

        positions = build_positions_screen0(sig_wide[common_t], feat_df, sim, REBAL_FREQ)
        pnl = sim.simulate_pnl(
            positions,
            returns[common_t],
            vol=vol[common_t],
            adv_dollars=adv[common_t],
            aum_dollars=aum,
            min_adv_dollars=min_adv,
        )
        m = sim.compute_metrics(pnl)
        res = {
            # The track was implicit in this file, so a consumer had to know
            # that TRACK is track_b to interpret the rows. Make it explicit.
            "track":     TRACK,
            "variant":   label,
            "n_features": len(available),
            "gross_sr":  m.get("gross_pnl_sharpe", np.nan),
            "net_sr":    m.get("net_pnl_sharpe", np.nan),
            "max_dd":    m.get("net_pnl_max_dd", np.nan),
            "annual_to": m.get("annual_turnover", np.nan),
            "cost_drag": m.get("cost_drag_bps", np.nan),
        }
        log(f"    gross SR={res['gross_sr']:+.3f}  net SR={res['net_sr']:+.3f}  "
            f"TO={res['annual_to']:.1f}x  drag={res['cost_drag']:.0f}bps/yr")
        results.append(res)

    out = pd.DataFrame(results)
    out.to_parquet(OUT, index=False)
    pd.set_option("display.width", 180)
    print("\n=== Ablation Summary ===")
    print(out.round(3).to_string(index=False))
    print(f"\nSaved → {OUT}")
    log(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
