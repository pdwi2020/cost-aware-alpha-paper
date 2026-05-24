"""Week 11: Generate all paper figures.

Figures:
  1. Cumulative P&L (IS + OOS) — gross and net, both tracks
  2. Cost sensitivity heatmap — net Sharpe vs spread × impact (weekly, Track A)
  3. Conformal coverage per fold + regime breakdown
  4. Feature IC decay by holding period (signal decay analysis)
  5. IS vs OOS comparison bar chart (gross/net Sharpe)

Run:
    python3 -u src/figures/make_figures.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent.parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

PROC = ROOT / "data" / "processed"

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       10,
    "axes.labelsize":  10,
    "axes.titlesize":  11,
    "legend.fontsize": 9,
    "figure.dpi":      150,
})

COLORS = {
    "track_a": "#2166ac",
    "track_b": "#d6604d",
    "gross":   "#4d4d4d",
    "net":     "#1a9641",
}


# ─────────────────────────────────────────────────────────────────
# Figure 1: Cumulative P&L (IS + OOS, both tracks)
# ─────────────────────────────────────────────────────────────────

def fig_cumulative_pnl():
    # Load IS sensitivity at weekly, base costs
    sens = pd.read_parquet(PROC / "sensitivity_3d.parquet")
    base_is = sens[(sens.spread_bps == 3) & (sens.impact_coeff == 0.10) & (sens.rebal_freq == 5)]

    # We don't have daily IS P&L stored, use backtest_base (daily rebalancing) as proxy
    # Instead, reconstruct from backtest_base
    # Actually we need to re-derive cumulative PnL — let's use the holdout P&L files and
    # generate IS from sensitivity directly
    # For IS cumulative P&L, we need to re-run but that's expensive.
    # Use backtest_base.parquet metrics to annotate a schematic, or read stored PnL from OOS.

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax_idx, track in enumerate(["track_b", "track_a"]):
        ax = axes[ax_idx]
        pnl_oos_path = PROC / f"holdout_pnl_{track}.parquet"
        if pnl_oos_path.exists():
            pnl_oos = pd.read_parquet(pnl_oos_path)
            if "track" in pnl_oos.columns:
                pnl_oos = pnl_oos.drop(columns="track")
            cum_gross = (1 + pnl_oos["gross_pnl"]).cumprod() - 1
            cum_net   = (1 + pnl_oos["net_pnl"]).cumprod() - 1
            ax.plot(cum_gross.index, cum_gross * 100, color=COLORS["gross"],
                    alpha=0.7, lw=1.5, label="Gross P&L")
            ax.plot(cum_net.index,   cum_net * 100,   color=COLORS["net"],
                    lw=2.0, label="Net P&L (AC costs)")
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.set_xlabel("Date")
            ax.set_ylabel("Cumulative Return (%)")
            track_label = "Track B (Macro/Sector)" if track == "track_b" else "Track A (Idiosyncratic Reversal)"
            ax.set_title(f"OOS Holdout 2022–2024 — {track_label}")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
            # Annotate SR
            met = pd.read_parquet(PROC / "holdout_metrics.parquet")
            row = met[met.track == track].iloc[0]
            ax.text(0.98, 0.05, f"Gross SR: {row.gross_pnl_sharpe:+.2f}\nNet SR: {row.net_pnl_sharpe:+.2f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    fig.suptitle("OOS Holdout Cumulative P&L (2022–2024, Weekly Rebalancing)", fontsize=12, y=1.02)
    plt.tight_layout()
    path = FIG_DIR / "fig1_cumulative_pnl.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
# Figure 2: Cost sensitivity heatmap (Track B OOS — best performer)
# ─────────────────────────────────────────────────────────────────

def fig_sensitivity_heatmap():
    sens = pd.read_parquet(PROC / "sensitivity_3d.parquet")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    rebal_labels = {1: "Daily (1d)", 5: "Weekly (5d)", 21: "Monthly (21d)"}

    for ax, rf in zip(axes, [1, 5, 21]):
        sub = sens[(sens.track == "track_a") & (sens.rebal_freq == rf)]
        pivot = sub.pivot(index="spread_bps", columns="impact_coeff",
                          values="net_pnl_sharpe")
        # Diverging colormap centered at 0
        vabs = max(abs(pivot.values.min()), abs(pivot.values.max()), 0.5)
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        im = ax.imshow(pivot.values, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{i}" for i in pivot.index], fontsize=8)
        ax.set_xlabel("Impact coeff η")
        ax.set_ylabel("Spread (bps)" if rf == 1 else "")
        ax.set_title(rebal_labels[rf])
        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                color = "white" if abs(val) > 0.4 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Net Sharpe", shrink=0.8)

    fig.suptitle("Track A — Net Sharpe vs Cost Parameters & Rebalancing Frequency", fontsize=11)
    plt.tight_layout()
    path = FIG_DIR / "fig2_sensitivity_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
# Figure 3: Conformal coverage per fold
# ─────────────────────────────────────────────────────────────────

def fig_conformal_coverage():
    cov = pd.read_parquet(PROC / "conformal_coverage.parquet")
    target = 0.90

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    track_labels = {"track_b": "Track B (Raw Return)", "track_a": "Track A (Idiosyncratic)"}

    for ax, track in zip(axes, ["track_b", "track_a"]):
        sub = cov[cov.track == track].sort_values("fold")
        folds = sub["fold"].astype(str)
        x = np.arange(len(folds))
        width = 0.25

        ax.bar(x - width, sub["coverage"] * 100, width=width, color="#4393c3",
               alpha=0.8, label="Overall")
        ax.bar(x, sub["coverage_calm"] * 100, width=width, color="#92c5de",
               alpha=0.8, label="VIX≤20 (Calm)")
        ax.bar(x + width, sub["coverage_stressed"] * 100, width=width, color="#f4a582",
               alpha=0.8, label="VIX>20 (Stressed)")

        ax.axhline(target * 100, color="red", lw=1.5, ls="--", label=f"Target {target*100:.0f}%")
        ax.set_xticks(x)
        ax.set_xticklabels(folds, rotation=45, ha="right")
        ax.set_xlabel("Walk-forward Fold (Test Year)")
        ax.set_ylabel("Empirical Coverage (%)")
        ax.set_title(track_labels[track])
        ax.set_ylim(30, 102)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate mean
        mean_cov = sub["coverage"].mean()
        ax.text(0.98, 0.98, f"Mean: {mean_cov*100:.1f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Conformal Prediction Coverage (90% Target) — Split by VIX Regime", fontsize=11)
    plt.tight_layout()
    path = FIG_DIR / "fig3_conformal_coverage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
# Figure 4: IS vs OOS comparison bar chart
# ─────────────────────────────────────────────────────────────────

def fig_is_oos_comparison():
    # IS weekly from sensitivity_3d
    sens = pd.read_parquet(PROC / "sensitivity_3d.parquet")
    is_w = sens[(sens.spread_bps == 3) & (sens.impact_coeff == 0.10) & (sens.rebal_freq == 5)]
    is_a = is_w[is_w.track == "track_a"].iloc[0]
    is_b = is_w[is_w.track == "track_b"].iloc[0]

    oos = pd.read_parquet(PROC / "holdout_metrics.parquet")
    oos_a = oos[oos.track == "track_a"].iloc[0]
    oos_b = oos[oos.track == "track_b"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, label in zip(axes, ["Gross Sharpe", "Net Sharpe"]):
        col = "gross_pnl_sharpe" if "Gross" in label else "net_pnl_sharpe"
        categories = ["Track A IS", "Track A OOS", "Track B IS", "Track B OOS"]
        values = [is_a[col], oos_a[col], is_b[col], oos_b[col]]
        colors = [
            COLORS["track_a"] if v >= 0 else "#aec7e8" for v in [is_a[col], oos_a[col]]
        ] + [
            COLORS["track_b"] if v >= 0 else "#fdae6b" for v in [is_b[col], oos_b[col]]
        ]
        bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.7)
        ax.axhline(0, color="black", lw=1.0)
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title(label)
        ax.set_ylim(min(values) - 0.3, max(values) + 0.3)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.03 if val >= 0 else -0.06),
                    f"{val:+.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticklabels(categories, rotation=20, ha="right")

    fig.suptitle("IS (2013–2021) vs OOS (2022–2024) Performance — Weekly Rebalancing",
                 fontsize=11)
    # Annotate Track B OOS finding
    axes[1].annotate("Track B reversal:\nmacro signal works\nin rate-hike regime",
                     xy=(3, oos_b["net_pnl_sharpe"]), xytext=(2.2, 0.7),
                     fontsize=8, ha="center", color="darkred",
                     arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5))
    plt.tight_layout()
    path = FIG_DIR / "fig4_is_oos_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
# Figure 5: Signal decay (IC vs holding period)
# ─────────────────────────────────────────────────────────────────

def fig_signal_decay():
    """Compute Spearman IC between composite signal and forward return at horizons 1–21d."""
    import pandas as pd
    from scipy.stats import spearmanr

    sig_a = pd.read_parquet(PROC / "signals_track_a.parquet")
    sig_b = pd.read_parquet(PROC / "signals_track_b.parquet")
    feat  = pd.read_parquet(PROC / "features_all.parquet")

    horizons = [1, 5, 10, 15, 21]
    results  = {"track_a": {}, "track_b": {}}

    for h in horizons:
        for track, sig in [("track_a", sig_a), ("track_b", sig_b)]:
            target_col = f"target_{track}"
            # Forward h-day return from daily_ohlcv would be ideal; use features_all
            # target_track_b is 5d forward, target_track_a is 5d idiosyncratic
            # Proxy: compute rolling h-day return from ret_1d
            # Approximate: use ret_1d shifted back h days as the "h-day signal"
            # Instead, just compute IC at different forward windows from daily OHLCV
            # For speed: use the existing signals and the daily returns
            ohlcv = pd.read_parquet(PROC / "daily_ohlcv.parquet")
            close = ohlcv.pivot(index="date", columns="ticker", values="close")
            # h-day forward return
            fwd_h = close.shift(-h).pct_change(h)  # returns at horizon h
            fwd_h_test = fwd_h.loc[
                (fwd_h.index >= pd.Timestamp("2013-01-01")) &
                (fwd_h.index <= pd.Timestamp("2021-12-31"))
            ]
            sig_aligned = sig.reindex(columns=fwd_h_test.columns)

            ic_vals = []
            for d in sig_aligned.index[:100]:  # sample 100 dates for speed
                if d not in fwd_h_test.index:
                    continue
                s = sig_aligned.loc[d].dropna()
                r = fwd_h_test.loc[d].reindex(s.index).dropna()
                if len(r) > 50:
                    ic, _ = spearmanr(s.reindex(r.index), r)
                    if np.isfinite(ic):
                        ic_vals.append(ic)

            results[track][h] = np.mean(ic_vals) if ic_vals else np.nan
        break  # ohlcv loaded once — reuse below
    else:
        pass

    # Re-run cleanly
    ohlcv = pd.read_parquet(PROC / "daily_ohlcv.parquet")
    close = ohlcv.pivot(index="date", columns="ticker", values="close")

    for h in horizons:
        fwd_h = close.pct_change(h).shift(-h)
        fwd_test = fwd_h.loc[
            (fwd_h.index >= pd.Timestamp("2013-01-01")) &
            (fwd_h.index <= pd.Timestamp("2021-12-31"))
        ]
        for track, sig in [("track_a", sig_a), ("track_b", sig_b)]:
            sig_al = sig.reindex(columns=fwd_test.columns)
            ic_vals = []
            sample_dates = sig_al.index[::5][:50]  # every 5th date, 50 samples
            for d in sample_dates:
                if d not in fwd_test.index:
                    continue
                s = sig_al.loc[d].dropna()
                r = fwd_test.loc[d].reindex(s.index).dropna()
                if len(r) > 50:
                    from scipy.stats import spearmanr
                    ic, _ = spearmanr(s.reindex(r.index), r)
                    if np.isfinite(ic):
                        ic_vals.append(ic)
            results[track][h] = np.mean(ic_vals) if ic_vals else np.nan

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(horizons, [results["track_a"][h] for h in horizons],
            "o-", color=COLORS["track_a"], lw=2, ms=7, label="Track A (Idiosyncratic)")
    ax.plot(horizons, [results["track_b"][h] for h in horizons],
            "s-", color=COLORS["track_b"], lw=2, ms=7, label="Track B (Raw Return)")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.fill_between(horizons,
                    [results["track_a"][h] for h in horizons],
                    0, alpha=0.15, color=COLORS["track_a"])
    ax.fill_between(horizons,
                    [results["track_b"][h] for h in horizons],
                    0, alpha=0.15, color=COLORS["track_b"])
    ax.set_xlabel("Holding Period (Trading Days)")
    ax.set_ylabel("Mean IC (Spearman Rank Correlation)")
    ax.set_title("Signal IC Decay vs Holding Period")
    ax.set_xticks(horizons)
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Annotate weekly sweet spot
    ax.axvline(5, color="gray", lw=1.2, ls=":", alpha=0.7)
    ax.text(5.3, ax.get_ylim()[1] * 0.95, "Weekly\nrebal.", fontsize=8, color="gray")

    plt.tight_layout()
    path = FIG_DIR / "fig5_signal_decay.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
# Figure 6: SHAP feature importance (top 15 per track)
# ─────────────────────────────────────────────────────────────────

def fig_shap_importance():
    shap = pd.read_parquet(PROC / "shap_summary.parquet")
    fdr  = pd.read_parquet(PROC / "fdr_results.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, track in zip(axes, ["track_b", "track_a"]):
        sub  = shap[shap.track == track].groupby("feature")["mean_abs_shap"].mean()
        sub  = sub.sort_values(ascending=False).head(15)
        rej  = set(fdr[(fdr.track == track) & fdr.rejected]["feature"].tolist())

        colors = ["#1a9641" if f in rej else "#cccccc" for f in sub.index]
        bars = ax.barh(range(len(sub)), sub.values, color=colors, edgecolor="none")
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP| (XGB + LGBM average)")
        track_label = "Track B (Raw Return)" if track == "track_b" else "Track A (Idiosyncratic)"
        ax.set_title(track_label)
        ax.grid(True, alpha=0.3, axis="x")

        # Legend
        from matplotlib.patches import Patch
        legend_els = [Patch(facecolor="#1a9641", label="BH-significant (q=0.10)"),
                      Patch(facecolor="#cccccc", label="Not significant")]
        ax.legend(handles=legend_els, loc="lower right", fontsize=8)

    fig.suptitle("SHAP Feature Importance — BH FDR Significant Features Highlighted", fontsize=11)
    plt.tight_layout()
    path = FIG_DIR / "fig6_shap_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    print("=== Week 11: Generating Paper Figures ===\n")

    print("Figure 1: Cumulative P&L …")
    fig_cumulative_pnl()

    print("Figure 2: Cost sensitivity heatmap …")
    fig_sensitivity_heatmap()

    print("Figure 3: Conformal coverage …")
    fig_conformal_coverage()

    print("Figure 4: IS vs OOS comparison …")
    fig_is_oos_comparison()

    print("Figure 5: Signal decay …")
    fig_signal_decay()

    print("Figure 6: SHAP importance …")
    fig_shap_importance()

    print(f"\nAll figures saved to: {FIG_DIR}/")
    figs = sorted(FIG_DIR.glob("*.png"))
    for f in figs:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
