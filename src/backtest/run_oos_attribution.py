"""B2b — OOS return attribution by feature group for Track B (2022-2024).

Addresses referee point M5: the IS→OOS sign-flip mechanism (§9.4) is narrative.
This script produces a quantified per-group P&L decomposition.

Feature groups (25 BH-selected Track B features):
  Macro / cross-asset  (2): vix, credit_proxy_chg_5d
  Sector-crowding     (10): corr_XLF, corr_XLB, corr_XLK, corr_XLI, corr_SPY,
                            corr_QQQ, corr_XLY, corr_XLV, corr_XLE, corr_XLP
  Liquidity / vol      (6): amihud, roll_spread, vol_21d, sharpe_21d,
                            vol_sig_ratio, vol_clock
  Reversal / momentum  (7): ret_1d, ret_5d, reversal_1w, ret_21d,
                            reversal_4w, ret_63d, vwap_dev

Two modes (both reported):
  (A) Standalone — each group's sub-signal run through the full pipeline.
      Individually valid but do not sum to full (due to cross-sectional
      normalisation and capping in position construction).
  (B) LOGO — full 25-feature signal minus each group; contribution =
      SR(full) − SR(full ∖ group).

Inputs:
    data/processed/features_all.parquet     (MultiIndex: ticker×date, 40 cols)
    data/processed/fdr_results.parquet      (feature, track, mean_ic, rejected)
    data/processed/shap_summary.parquet     (feature, model, track, mean_abs_shap)
    data/processed/daily_ohlcv.parquet      (ticker, date, close, volume, …)
    configs/backtest.yaml

Output:
    data/processed/oos_attribution.parquet
    (cols: group, mode, n_features, gross_sr, net_sr, ann_net_pct,
           cost_drag_bps, max_dd_pct, turnover)

Run:
    python3 -u src/backtest/run_oos_attribution.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.portfolio import PortfolioSimulator

# ── paths ────────────────────────────────────────────────────────────────────
FEAT_PATH  = ROOT / "data" / "processed" / "features_all.parquet"
FDR_PATH   = ROOT / "data" / "processed" / "fdr_results.parquet"
SHAP_PATH  = ROOT / "data" / "processed" / "shap_summary.parquet"
OHLCV_PATH = ROOT / "data" / "processed" / "daily_ohlcv.parquet"
CFG_PATH   = ROOT / "configs" / "backtest.yaml"
OUT        = ROOT / "data" / "processed" / "oos_attribution.parquet"

OOS_START  = "2022-01-01"
OOS_END    = "2024-12-31"
REBAL_FREQ = 5
VOL_WINDOW = 21
TRACK      = "track_b"
ANN        = 252.0

# Screen 0 — tradeable-universe filter (matches run_holdout.py verbatim)
MIN_PRICE    = 5.0
SANITIZE_CAP = 0.50

# ── feature group map (25 BH-selected Track B features) ─────────────────────
GROUP_MAP = {
    "Macro":    ["vix", "credit_proxy_chg_5d"],
    "Sector":   ["corr_XLF", "corr_XLB", "corr_XLK", "corr_XLI", "corr_SPY",
                 "corr_QQQ", "corr_XLY", "corr_XLV", "corr_XLE", "corr_XLP"],
    "Liq/Vol":  ["amihud", "roll_spread", "vol_21d", "sharpe_21d",
                 "vol_sig_ratio", "vol_clock"],
    "Rev/Mom":  ["ret_1d", "ret_5d", "reversal_1w", "ret_21d",
                 "reversal_4w", "ret_63d", "vwap_dev"],
    "Full":     None,   # filled after loading FDR results
}

GROUP_LONG = {
    "Macro":   "Macro / cross-asset",
    "Sector":  "Sector-crowding",
    "Liq/Vol": "Liquidity / volatility",
    "Rev/Mom": "Reversal / momentum",
    "Full":    "Full Track B (25)",
}


def log(msg):
    print(msg, flush=True)


def apply_min_price_filter(positions, close_w, min_price=MIN_PRICE):
    """Screen 0: zero positions in names priced < min_price on the trade date,
    then renormalise gross (L1) leverage to 1 per day."""
    pxa = close_w.reindex(index=positions.index, columns=positions.columns).ffill()
    positions = positions.where(pxa >= min_price, 0.0)
    l1 = positions.abs().sum(axis=1).replace(0, np.nan)
    return positions.div(l1, axis=0).fillna(0.0)


def sharpe_ann(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 1e-12 else np.nan


def run_variant(label: str, mode: str, feat_subset: list, oos_feat: pd.DataFrame,
                shap_weights: dict, fdr_sign: dict, sim: PortfolioSimulator,
                returns: pd.DataFrame, vol: pd.DataFrame, adv: pd.DataFrame,
                aum: float, min_adv: float,
                close_w: pd.DataFrame | None = None) -> dict | None:
    available = [f for f in feat_subset if f in oos_feat.columns]
    if not available:
        log(f"  SKIP {label}: no available features")
        return None

    lvl_names = oos_feat.index.names
    date_level_name   = next(n for n in lvl_names if "date" in str(n).lower())
    ticker_level_name = next(n for n in lvl_names if n != date_level_name)

    col_sub = oos_feat[available].copy()
    for f in available:
        w    = float(shap_weights.get(f, 1.0))
        sign = float(fdr_sign.get(f, 1.0))
        col_sub[f] = col_sub[f] * w * sign

    composite = col_sub.sum(axis=1)
    composite.name = "signal"
    df_sig = composite.reset_index()
    sig_wide = df_sig.pivot(index=date_level_name, columns=ticker_level_name, values="signal")
    sig_wide.index = pd.to_datetime(sig_wide.index)
    sig_wide = sig_wide.sort_index()

    common_t = [t for t in sig_wide.columns if t in returns.columns]
    if not common_t:
        log(f"  SKIP {label}: no common tickers")
        return None

    positions = sim.signal_to_positions(sig_wide[common_t], lag=1, rebal_freq=REBAL_FREQ)
    # Screen 0: exclude penny stocks (price < $5) on trade date
    if close_w is not None:
        positions = apply_min_price_filter(positions, close_w[common_t])
    pnl = sim.simulate_pnl(
        positions,
        returns[common_t].clip(lower=-SANITIZE_CAP, upper=SANITIZE_CAP),
        vol=vol[common_t],
        adv_dollars=adv[common_t],
        aum_dollars=aum,
        min_adv_dollars=min_adv,
    )
    m = sim.compute_metrics(pnl)
    return {
        "group":          GROUP_LONG.get(label, label),
        "mode":           mode,
        "n_features":     len(available),
        "gross_sr":       round(m.get("gross_pnl_sharpe", np.nan), 3),
        "net_sr":         round(m.get("net_pnl_sharpe",   np.nan), 3),
        "ann_net_pct":    round(m.get("net_pnl_annual", np.nan) * 100, 2),
        "cost_drag_bps":  round(m.get("cost_drag_bps", np.nan), 1),
        "max_dd_pct":     round(m.get("net_pnl_max_dd", np.nan) * 100, 2),
        "turnover":       round(m.get("annual_turnover", np.nan), 1),
    }


def main():
    t0 = time.time()
    log("=== B2b OOS Return Attribution ===\n")

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    fdr_df  = pd.read_parquet(FDR_PATH)
    shap_df = pd.read_parquet(SHAP_PATH)
    feat_df = pd.read_parquet(FEAT_PATH)
    ohlcv   = pd.read_parquet(OHLCV_PATH)

    # Normalise MultiIndex datetime levels
    if isinstance(feat_df.index, pd.MultiIndex):
        new_levels = []
        for lvl in feat_df.index.levels:
            try:
                new_levels.append(pd.to_datetime(lvl))
            except Exception:
                new_levels.append(lvl)
        feat_df.index = feat_df.index.set_levels(new_levels)

    fdr_b       = fdr_df[fdr_df.track == TRACK]
    bh_features = fdr_b.loc[fdr_b.rejected, "feature"].tolist()

    # Verify group map covers exactly the 25 BH features
    all_grouped = [f for g in ["Macro","Sector","Liq/Vol","Rev/Mom"] for f in GROUP_MAP[g]]
    assert set(all_grouped) == set(bh_features), (
        f"Group map mismatch: {set(all_grouped).symmetric_difference(set(bh_features))}"
    )
    GROUP_MAP["Full"] = bh_features

    fdr_sign = {r.feature: float(np.sign(r.mean_ic)) for _, r in fdr_b[fdr_b.rejected].iterrows()}

    shap_b   = shap_df[(shap_df.track == TRACK) & (shap_df.model.isin(["xgb", "lgbm"]))]
    shap_agg = shap_b.groupby("feature")["mean_abs_shap"].mean()
    shap_agg = shap_agg / shap_agg.sum()   # ℓ1-normalise over all 25 BH features
    shap_weights = {f: float(shap_agg.get(f, 1.0 / len(bh_features))) for f in bh_features}

    # OOS OHLCV slice
    oos_ohlcv = ohlcv[(ohlcv["date"] >= pd.Timestamp(OOS_START)) &
                      (ohlcv["date"] <= pd.Timestamp(OOS_END))]
    close_w = oos_ohlcv.pivot(index="date", columns="ticker", values="close").sort_index()
    vol_w   = oos_ohlcv.pivot(index="date", columns="ticker", values="volume").fillna(0)
    returns_raw = close_w.pct_change()
    n_clipped = int((returns_raw.abs() > SANITIZE_CAP).sum().sum())
    if n_clipped > 0:
        log(f"  [sanitize] Clipped {n_clipped} (ticker,day) cells with |ret|>{SANITIZE_CAP:.0%}")
    returns = returns_raw.clip(lower=-SANITIZE_CAP, upper=SANITIZE_CAP)
    vol     = returns.rolling(VOL_WINDOW, min_periods=10).std()
    adv     = (close_w * vol_w).rolling(VOL_WINDOW, min_periods=10).mean()
    date_mask = (returns.index >= pd.Timestamp(OOS_START)) & (returns.index <= pd.Timestamp(OOS_END))
    returns, vol, adv, close_w = returns[date_mask], vol[date_mask], adv[date_mask], close_w[date_mask]

    # OOS feature slice
    lvl_names  = feat_df.index.names
    date_level = 0 if "date" in str(lvl_names[0]).lower() else 1
    date_vals  = feat_df.index.get_level_values(date_level)
    oos_feat   = feat_df[(date_vals >= pd.Timestamp(OOS_START)) &
                         (date_vals <= pd.Timestamp(OOS_END))]

    sim     = PortfolioSimulator(config_path=str(CFG_PATH))
    aum     = cfg.get("aum_dollars",     1e8)
    min_adv = cfg.get("min_adv_dollars", 1e6)

    results = []

    # ── Mode A: standalone per-group ────────────────────────────────────────
    log("Mode A — Standalone group signals")
    full_net_sr = None
    for gname in ["Full", "Macro", "Sector", "Liq/Vol", "Rev/Mom"]:
        feats = GROUP_MAP[gname]
        log(f"  {gname} ({len(feats)} features)")
        res = run_variant(gname, "A_standalone", feats, oos_feat,
                          shap_weights, fdr_sign, sim, returns, vol, adv, aum, min_adv,
                          close_w=close_w)
        if res:
            if gname == "Full":
                full_net_sr = res["net_sr"]
            log(f"    → gross SR={res['gross_sr']:+.3f}  net SR={res['net_sr']:+.3f}  "
                f"drag={res['cost_drag_bps']:.0f} bps/yr")
            results.append(res)

    # ── Mode B: leave-one-group-out marginal ────────────────────────────────
    log("\nMode B — Leave-one-group-out marginal contributions")
    for gname in ["Macro", "Sector", "Liq/Vol", "Rev/Mom"]:
        complement = [f for f in bh_features if f not in GROUP_MAP[gname]]
        label_logo = f"{gname}_LOGO"
        log(f"  {gname} complement ({len(complement)} features)")
        res = run_variant(label_logo, "B_logo", complement, oos_feat,
                          shap_weights, fdr_sign, sim, returns, vol, adv, aum, min_adv,
                          close_w=close_w)
        if res and full_net_sr is not None:
            marginal = round(full_net_sr - res["net_sr"], 3)
            res["logo_marginal_sr"] = marginal
            log(f"    → full-LOGO net SR={res['net_sr']:+.3f}  marginal={marginal:+.3f}")
            results.append(res)

    out = pd.DataFrame(results)
    out.to_parquet(OUT, index=False)
    log(f"\n=== Attribution summary ===")
    pd.set_option("display.width", 160)
    print(out[["group","mode","n_features","gross_sr","net_sr","cost_drag_bps"]
              ].to_string(index=False))

    # Acceptance check (anchor: holdout_metrics Track B net SR = +0.357)
    full_row = out[(out["mode"] == "A_standalone") & (out["group"] == "Full Track B (25)")]
    if not full_row.empty:
        full_sr = full_row.iloc[0]["net_sr"]
        if abs(full_sr - 0.357) > 0.02:
            log(f"\n⚠ Acceptance check FAILED: full net SR={full_sr:.3f} ≠ +0.357 ± 0.02")
        else:
            log(f"\n✓ Acceptance check PASSED: full net SR={full_sr:.3f}")
    log(f"\nSaved → {OUT}")
    log(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
