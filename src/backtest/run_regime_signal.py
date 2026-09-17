"""Regime-conditional signal construction from fdr_regime.parquet.

Builds 4 OOS (2022-2024) composites for both tracks:
  static         — SHAP-weighted BH-rejected features (same as Exp 7 baseline)
  regime_adaptive — calm features (VIX≤20) or stressed features (VIX>20), switched daily
  calm_only      — calm-regime BH-rejected features always
  stressed_only  — stressed-regime BH-rejected features always

For Track B, regime-BH has 0 rejections in both regimes → all variants == static.
This is reported explicitly as a finding (regime-agnostic features).

All OOS parameters locked from IS: spread=3bps, impact=0.10, AUM=$100M, weekly rebal.

Outputs:
    data/processed/regime_signal_results.parquet  — year-by-year + full-period metrics
    figures/fig7_regime_signal.png                 — net SR bar chart by variant × year

Run:
    python3 -u src/backtest/run_regime_signal.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.portfolio import PortfolioSimulator, build_positions_screen0

FDR_PATH    = ROOT / "data" / "processed" / "fdr_results.parquet"
REGIME_PATH = ROOT / "data" / "processed" / "fdr_regime.parquet"
SHAP_PATH   = ROOT / "data" / "processed" / "shap_summary.parquet"
FEAT_PATH   = ROOT / "data" / "processed" / "features_all.parquet"
OHLCV_PATH  = ROOT / "data" / "processed" / "daily_ohlcv_v3.parquet"
CFG_PATH    = ROOT / "configs" / "backtest.yaml"

OUT_RESULTS = ROOT / "data" / "processed" / "regime_signal_results.parquet"
OUT_FIG     = ROOT / "figures" / "fig7_regime_signal.png"

HOLDOUT_START  = "2022-01-01"
HOLDOUT_END    = "2024-12-31"
REBAL_FREQ     = 5
VOL_WINDOW     = 21
VIX_THRESHOLD  = 20.0
VARIANTS       = ["static", "regime_adaptive", "calm_only", "stressed_only"]


def log(msg): print(msg, flush=True)


def get_shap_avg(shap_df: pd.DataFrame, track: str) -> pd.Series:
    return (
        shap_df[shap_df["track"] == track]
        .groupby("feature")["mean_abs_shap"]
        .mean()
    )


def signed_shap_weights(features: list, mean_ic: pd.Series, shap_avg: pd.Series) -> pd.Series:
    """Sign(IC) × SHAP, L1-normalised. Empty feature list → empty Series."""
    if not features:
        return pd.Series(dtype=float)
    w = {}
    for feat in features:
        sign = np.sign(float(mean_ic.get(feat, 1.0)))
        w[feat] = sign * float(shap_avg.get(feat, shap_avg.mean() if len(shap_avg) > 0 else 1.0))
    ws = pd.Series(w)
    denom = ws.abs().sum()
    return ws / denom if denom > 1e-12 else ws


def build_signal_matrix(
    feat_df: pd.DataFrame,
    weights: pd.Series,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Weighted composite signal → (date × ticker). Returns empty df if no weights."""
    if weights.empty:
        return pd.DataFrame()
    features = [f for f in weights.index if f in feat_df.columns]
    if not features:
        return pd.DataFrame()
    dates = feat_df.index.get_level_values("date")
    mask  = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    sub   = feat_df.loc[mask, features]
    w_sub = weights.loc[features]
    all_dates   = sub.index.get_level_values("date").unique().sort_values()
    records = []
    for d in all_dates:
        day_data   = sub.xs(d, level="date")
        row_signal = pd.Series(0.0, index=day_data.index)
        for feat, w in w_sub.items():
            col = day_data[feat].fillna(day_data[feat].median())
            row_signal += w * col
        records.append(row_signal.rename(d))
    return pd.DataFrame(records)


def build_returns_vol_adv(ohlcv: pd.DataFrame, tickers: list, start: str, end: str):
    mask = (
        ohlcv["ticker"].isin(tickers)
        & (ohlcv["date"] >= pd.Timestamp(start))
        & (ohlcv["date"] <= pd.Timestamp(end))
    )
    sub = ohlcv[mask][["ticker", "date", "close", "volume"]].copy()
    close_w  = sub.pivot(index="date", columns="ticker", values="close")
    volume_w = sub.pivot(index="date", columns="ticker", values="volume")
    returns  = close_w.pct_change()
    vol      = returns.rolling(VOL_WINDOW, min_periods=10).std()
    adv      = (close_w * volume_w).rolling(VOL_WINDOW, min_periods=10).mean()
    ret_mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    return returns[ret_mask], vol[ret_mask], adv[ret_mask]


def compute_year_metrics(pnl_df: pd.DataFrame, sim: PortfolioSimulator) -> dict:
    """Year-by-year + full-period net SR and other metrics."""
    results = {}
    years = sorted(pnl_df.index.year.unique())
    for yr in years:
        sub = pnl_df[pnl_df.index.year == yr]
        if len(sub) < 20:
            continue
        m = sim.compute_metrics(sub)
        results[str(yr)] = {
            "net_sr":     m.get("net_pnl_sharpe", np.nan),
            "gross_sr":   m.get("gross_pnl_sharpe", np.nan),
            "net_annual": m.get("net_pnl_annual", np.nan),
            "max_dd":     m.get("net_pnl_max_dd", np.nan),
            "calmar":     m.get("net_pnl_calmar", m.get("net_pnl_annual", np.nan) /
                               abs(m.get("net_pnl_max_dd", 1e-6)) if m.get("net_pnl_max_dd", 0) else np.nan),
        }
    full = sim.compute_metrics(pnl_df)
    results["OOS (2022-24)"] = {
        "net_sr":     full.get("net_pnl_sharpe", np.nan),
        "gross_sr":   full.get("gross_pnl_sharpe", np.nan),
        "net_annual": full.get("net_pnl_annual", np.nan),
        "max_dd":     full.get("net_pnl_max_dd", np.nan),
        "calmar":     full.get("net_pnl_calmar", np.nan),
    }
    return results


def run_variant(
    variant: str,
    track: str,
    feat_df: pd.DataFrame,
    ohlcv: pd.DataFrame,
    weights_static: pd.Series,
    weights_calm: pd.Series,
    weights_stressed: pd.Series,
    vix_daily: pd.Series,
    sim: PortfolioSimulator,
    cfg: dict,
) -> dict:
    """Run one variant and return metrics dict keyed by period."""

    if variant == "static":
        sig = build_signal_matrix(feat_df, weights_static, HOLDOUT_START, HOLDOUT_END)
    elif variant == "calm_only":
        w = weights_calm if not weights_calm.empty else weights_static
        sig = build_signal_matrix(feat_df, w, HOLDOUT_START, HOLDOUT_END)
    elif variant == "stressed_only":
        w = weights_stressed if not weights_stressed.empty else weights_static
        sig = build_signal_matrix(feat_df, w, HOLDOUT_START, HOLDOUT_END)
    elif variant == "regime_adaptive":
        # Build two signal matrices, then splice by VIX daily
        sig_calm     = build_signal_matrix(feat_df, weights_calm if not weights_calm.empty else weights_static,
                                           HOLDOUT_START, HOLDOUT_END)
        sig_stressed = build_signal_matrix(feat_df, weights_stressed if not weights_stressed.empty else weights_static,
                                           HOLDOUT_START, HOLDOUT_END)
        # Align indices
        common_dates = sig_calm.index.intersection(sig_stressed.index)
        sig_c = sig_calm.reindex(common_dates)
        sig_s = sig_stressed.reindex(common_dates)
        # Splice: stressed when VIX > threshold
        vix_oos = vix_daily.reindex(common_dates).ffill()
        stressed_days = vix_oos > VIX_THRESHOLD
        sig = sig_c.copy()
        sig.loc[stressed_days] = sig_s.loc[stressed_days]
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if sig.empty:
        return {}

    tickers = sig.columns.tolist()
    returns, vol, adv = build_returns_vol_adv(ohlcv, tickers, HOLDOUT_START, HOLDOUT_END)

    positions = build_positions_screen0(sig, feat_df, sim, REBAL_FREQ)
    pnl_df    = sim.simulate_pnl(
        positions, returns, vol=vol,
        adv_dollars=adv,
        aum_dollars=cfg.get("aum_dollars", 1e8),
        min_adv_dollars=cfg.get("min_adv_dollars", 1e6),
    )
    return compute_year_metrics(pnl_df, sim)


def plot_results(records: list, out_path: Path):
    """Bar chart: net SR by variant and year for both tracks."""
    df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    years = ["2022", "2023", "2024", "OOS (2022-24)"]
    x = np.arange(len(years))
    width = 0.18

    colors = {"static": "#1f77b4", "regime_adaptive": "#ff7f0e",
              "calm_only": "#2ca02c", "stressed_only": "#d62728"}

    for ax_i, track in enumerate(["track_a", "track_b"]):
        ax = axes[ax_i]
        sub = df[df["track"] == track]
        for v_i, variant in enumerate(VARIANTS):
            vs = sub[sub["variant"] == variant]
            vals = []
            for yr in years:
                row = vs[vs["period"] == yr]
                vals.append(float(row["net_sr"].iloc[0]) if len(row) > 0 else np.nan)
            offset = (v_i - 1.5) * width
            ax.bar(x + offset, vals, width, label=variant,
                   color=colors[variant], alpha=0.85)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{track.upper()} — Regime Signal Variants", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(years, fontsize=9)
        ax.set_ylabel("Net Sharpe Ratio")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved figure → {out_path}")


def main():
    log("=== Regime-Conditional Signal Construction ===\n")
    t0 = time.time()

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    fdr_df     = pd.read_parquet(FDR_PATH)
    regime_df  = pd.read_parquet(REGIME_PATH)
    shap_df    = pd.read_parquet(SHAP_PATH)

    log("Loading features_all.parquet …")
    # Lean load: frozen-specification columns plus regime_vix, which this
    # script splits on. The full float64 panel does not fit in 8 GB.
    from src.data.lean_load import load_features_lean

    feat_df = load_features_lean(FEAT_PATH)
    log("Loading OHLCV …")
    ohlcv   = pd.read_parquet(OHLCV_PATH)

    # VIX daily series for regime switching (from features panel, VIX is a ticker-level col)
    dates_level = feat_df.index.get_level_values("date")
    oos_mask    = (dates_level >= pd.Timestamp(HOLDOUT_START)) & \
                  (dates_level <= pd.Timestamp(HOLDOUT_END))
    feat_oos    = feat_df.loc[oos_mask]
    if "vix" in feat_oos.columns:
        vix_daily = (
            feat_oos["vix"]
            .groupby(level="date").median()
            .rename("vix")
        )
    else:
        log("  WARNING: VIX column not found — regime_adaptive will use static signal only")
        vix_daily = pd.Series(dtype=float)

    sim = PortfolioSimulator(
        config_path=str(CFG_PATH),
        spread_bps=cfg["spread_bps"],
        impact_coeff=cfg["impact_coeff"],
    )

    all_records = []

    for track in ["track_a", "track_b"]:
        log(f"\n{'='*60}")
        log(f"  TRACK: {track.upper()}")
        log(f"{'='*60}")

        shap_avg = get_shap_avg(shap_df, track)

        # --- Static weights (pooled BH, from fdr_results.parquet) ---
        static_feats = fdr_df[(fdr_df["track"] == track) & fdr_df["bh_rejected"]]["feature"].tolist()
        mean_ic_pooled = fdr_df[fdr_df["track"] == track].set_index("feature")["ic_bar"]
        weights_static = signed_shap_weights(static_feats, mean_ic_pooled, shap_avg)
        log(f"  Static features ({len(static_feats)}): {static_feats}")

        # --- Regime-specific weights (from fdr_regime.parquet) ---
        calm_feats = regime_df[
            (regime_df["track"] == track) & (regime_df["regime"] == "calm") & regime_df["bh_rejected"]
        ]["feature"].tolist()
        stressed_feats = regime_df[
            (regime_df["track"] == track) & (regime_df["regime"] == "stressed") & regime_df["bh_rejected"]
        ]["feature"].tolist()

        mean_ic_regime = regime_df[regime_df["track"] == track].set_index(
            ["feature", "regime"])["ic_bar"]

        def _regime_ic(feats, reg):
            return pd.Series({
                f: float(mean_ic_regime.get((f, reg), mean_ic_pooled.get(f, 0.0)))
                for f in feats
            })

        weights_calm     = signed_shap_weights(calm_feats,     _regime_ic(calm_feats, "calm"),     shap_avg)
        weights_stressed = signed_shap_weights(stressed_feats, _regime_ic(stressed_feats, "stressed"), shap_avg)

        if not calm_feats and not stressed_feats:
            log(f"  [{track}] Regime-BH rejected 0 features in both regimes.")
            log(f"  → regime_adaptive / calm_only / stressed_only all fall back to static signal.")
        else:
            log(f"  Calm features ({len(calm_feats)}):     {calm_feats}")
            log(f"  Stressed features ({len(stressed_feats)}): {stressed_feats}")

        # --- Run all 4 variants ---
        for variant in VARIANTS:
            log(f"\n  Variant: {variant}")
            metrics = run_variant(
                variant, track, feat_df, ohlcv,
                weights_static, weights_calm, weights_stressed,
                vix_daily, sim, cfg,
            )
            if not metrics:
                log(f"    [SKIP] Empty signal")
                continue
            for period, m in metrics.items():
                log(f"    {period:>16s}  net_SR={m['net_sr']:+.3f}  gross_SR={m['gross_sr']:+.3f}")
                all_records.append({
                    "track": track, "variant": variant, "period": period, **m
                })

    results_df = pd.DataFrame(all_records)
    results_df.to_parquet(OUT_RESULTS, index=False)
    log(f"\nSaved → {OUT_RESULTS}")

    plot_results(all_records, OUT_FIG)

    # Summary table
    log("\n=== SUMMARY: Full OOS Net SR by Track × Variant ===")
    summ = results_df[results_df["period"] == "OOS (2022-24)"][
        ["track", "variant", "net_sr", "calmar"]
    ].pivot_table(index="track", columns="variant", values="net_sr")
    log(summ.round(3).to_string())

    log(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
