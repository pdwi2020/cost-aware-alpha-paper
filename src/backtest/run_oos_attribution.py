"""B2b — OOS return attribution by feature group for Track B (2022-2024).

Addresses referee point M5: the IS→OOS sign-flip mechanism (§9.4) is narrative.
This script produces a quantified per-group P&L decomposition.

Feature groups (BH-selected Track B features, grouped by economic theme):
  Momentum            : mom_12_1, ret_1d, ret_5d, ret_21d, ret_63d, ret_252d
  Sector-crowding     : corr_SPY, corr_QQQ, corr_XLK, corr_XLE, corr_XLF, ...
  Liquidity / vol     : amihud, roll_spread, vol_21d, sharpe_21d, vol_sig_ratio,
                        vol_clock, vwap_dev, overnight_gap
  Macro-interaction   : beta_x_vix, beta_x_term_spread, credit_beta_x_credit
  (reversal_1w/4w dropped: == -ret_5d/-ret_21d; vix/credit_proxy_chg_5d dropped:
   broadcast-only; beta interactions carry macro signal stock-specifically)

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

from src.backtest.portfolio import PortfolioSimulator, apply_s0_eligible
from src.features.feature_spec import feature_columns as _feature_columns

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

# Screen 0 (look-ahead-free) — see src/data/screen0.py and portfolio.apply_s0_eligible.
# MIN_PRICE removed: apply_min_price_filter (which used trade-date close) was look-ahead.
SANITIZE_CAP = 0.50

# ── feature group templates (pattern-based; matched against actual BH features) ──
# reversal_1w / reversal_4w are DROPPED (== -ret_5d / -ret_21d); group absent.
# vix / credit_proxy_chg_5d are DROPPED broadcast; macro-group now uses interactions.
# The actual BH-selected feature list comes from fdr_results.parquet at runtime.
_GROUP_TEMPLATES = {
    "Momentum":     ["mom_12_1", "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_252d"],
    "Sector":       ["corr_SPY", "corr_QQQ",
                     "corr_XLK", "corr_XLE", "corr_XLF", "corr_XLY", "corr_XLP",
                     "corr_XLI", "corr_XLB", "corr_XLU", "corr_XLV", "corr_XLC", "corr_XLRE"],
    "Liq/Vol":      ["amihud", "roll_spread", "vol_21d", "sharpe_21d",
                     "vol_sig_ratio", "vol_clock", "vwap_dev", "overnight_gap"],
    "MacroInteract":["beta_x_vix", "beta_x_term_spread", "credit_beta_x_credit"],
    "Full":         None,   # filled after loading FDR results
}

_GROUP_LONG_TEMPLATES = {
    "Momentum":      "Momentum / return",
    "Sector":        "Sector-crowding",
    "Liq/Vol":       "Liquidity / volatility",
    "MacroInteract": "Macro-interaction (beta × macro)",
    "Full":          "Full Track B (BH-selected)",
}


def log(msg):
    print(msg, flush=True)


def sharpe_ann(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 1e-12 else np.nan


def run_variant(label: str, mode: str, feat_subset: list, oos_feat: pd.DataFrame,
                shap_weights: dict, fdr_sign: dict, sim: PortfolioSimulator,
                returns: pd.DataFrame, vol: pd.DataFrame, adv: pd.DataFrame,
                aum: float, min_adv: float,
                screen0_feat: pd.DataFrame | None = None) -> dict | None:
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
    # Screen 0 (look-ahead-free) — see src/data/screen0.py
    if screen0_feat is not None:
        positions = apply_s0_eligible(positions, screen0_feat)
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
        "group":          _GROUP_LONG_TEMPLATES.get(label, label),
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
    bh_set      = set(bh_features)

    # Build GROUP_MAP by intersecting templates with actual BH-selected features.
    # Dropped columns (reversal_1w, reversal_4w, vix, credit_proxy_chg_5d etc.)
    # will simply not appear in the intersection.  Empty groups are silently omitted.
    GROUP_MAP  = {}
    GROUP_LONG = {}
    for gname, candidates in _GROUP_TEMPLATES.items():
        if gname == "Full":
            GROUP_MAP["Full"]  = bh_features
            GROUP_LONG["Full"] = _GROUP_LONG_TEMPLATES["Full"]
            continue
        members = [f for f in candidates if f in bh_set]
        if members:
            GROUP_MAP[gname]  = members
            GROUP_LONG[gname] = _GROUP_LONG_TEMPLATES.get(gname, gname)

    # Any BH feature not covered by a named group goes into "Other".
    all_grouped = {f for g, ms in GROUP_MAP.items() if g != "Full" for f in ms}
    uncovered   = [f for f in bh_features if f not in all_grouped]
    if uncovered:
        GROUP_MAP["Other"]  = uncovered
        GROUP_LONG["Other"] = "Other (unclassified BH features)"
        log(f"  [attribution] Uncovered BH features → Other group: {uncovered}")

    log(f"  BH-selected features ({len(bh_features)}): {bh_features}")
    for gname, ms in GROUP_MAP.items():
        if gname != "Full":
            log(f"  Group {gname!r} ({len(ms)} features): {ms}")

    fdr_sign = {r.feature: float(np.sign(r.mean_ic)) for _, r in fdr_b[fdr_b.rejected].iterrows()}

    shap_b   = shap_df[(shap_df.track == TRACK) & (shap_df.model.isin(["xgb", "lgbm"]))]
    shap_agg = shap_b.groupby("feature")["mean_abs_shap"].mean()
    shap_agg = shap_agg / shap_agg.sum()   # ℓ1-normalise over all BH features
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
    for gname in ["Full"] + [g for g in GROUP_MAP if g != "Full"]:
        feats = GROUP_MAP[gname]
        log(f"  {gname} ({len(feats)} features)")
        res = run_variant(gname, "A_standalone", feats, oos_feat,
                          shap_weights, fdr_sign, sim, returns, vol, adv, aum, min_adv,
                          screen0_feat=oos_feat)
        if res:
            if gname == "Full":
                full_net_sr = res["net_sr"]
            log(f"    → gross SR={res['gross_sr']:+.3f}  net SR={res['net_sr']:+.3f}  "
                f"drag={res['cost_drag_bps']:.0f} bps/yr")
            results.append(res)

    # ── Mode B: leave-one-group-out marginal ────────────────────────────────
    log("\nMode B — Leave-one-group-out marginal contributions")
    for gname in [g for g in GROUP_MAP if g != "Full"]:
        complement = [f for f in bh_features if f not in GROUP_MAP[gname]]
        label_logo = f"{gname}_LOGO"
        log(f"  {gname} complement ({len(complement)} features)")
        res = run_variant(label_logo, "B_logo", complement, oos_feat,
                          shap_weights, fdr_sign, sim, returns, vol, adv, aum, min_adv,
                          screen0_feat=oos_feat)
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
    full_row = out[(out["mode"] == "A_standalone") & (out["group"] == _GROUP_LONG_TEMPLATES["Full"])]
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
