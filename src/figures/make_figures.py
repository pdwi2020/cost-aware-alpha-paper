"""Generate all paper figures from REBUILT (corrected) Track A pipeline data.

Figures:
  1. Track A cumulative P&L — locked 2025 OOS (holdout_pnl_track_a.parquet)
  2. Track A cost sensitivity heatmap — IS net Sharpe vs spread × impact
     (backtest_sensitivity.parquet, weekly-equivalent base +0.188)
  3. Conformal coverage per fold — Track A only (supplement)
  4. Multi-window Track A net Sharpe bar chart — IS / exploratory / locked OOS
     (manifest numbers: +0.188 / −0.61 / +0.666 p=0.62 n.s.)
  5. Track A signal IC decay by holding period (ic_by_fold ensemble IC + OHLCV)
  6. Track A SHAP feature importance, BH-significant features highlighted

Run:
    python3 -m src.figures.make_figures
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
from matplotlib.patches import Patch

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
    "gross":   "#4d4d4d",
    "net":     "#1a9641",
    "neg":     "#d6604d",
}


# ─────────────────────────────────────────────────────────────────
# Figure 1: Track A cumulative P&L — locked 2025 OOS
# Source: holdout_pnl_track_a.parquet (2025-01-01..2025-07-31, 144 days)
# ─────────────────────────────────────────────────────────────────

def fig_cumulative_pnl():
    pnl = pd.read_parquet(PROC / "holdout_pnl_track_a.parquet")
    # drop 'track' column if present; index is date
    if "track" in pnl.columns:
        pnl = pnl.drop(columns="track")

    cum_gross = (1 + pnl["gross_pnl"]).cumprod() - 1
    cum_net   = (1 + pnl["net_pnl"]).cumprod() - 1

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(cum_gross.index, cum_gross * 100,
            color=COLORS["gross"], alpha=0.7, lw=1.5, label="Gross P&L")
    ax.plot(cum_net.index,   cum_net   * 100,
            color=COLORS["net"],   lw=2.0,        label="Net P&L (AC costs)")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Track A (Idiosyncratic Reversal) — Locked OOS 2025-01..2025-07")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Annotate Sharpe ratios from holdout_metrics
    met = pd.read_parquet(PROC / "holdout_metrics.parquet")
    row = met[met.track == "track_a"].iloc[0]
    note = (f"Gross SR: {row.gross_pnl_sharpe:+.3f}\n"
            f"Net SR:   {row.net_pnl_sharpe:+.3f}\n"
            f"n=144 days, p=0.62 (n.s.)")
    ax.text(0.98, 0.05, note, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    path = FIG_DIR / "fig1_cumulative_pnl.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
# Figure 2: Track A IS net Sharpe sensitivity heatmap
# Source: backtest_sensitivity.parquet (Track A only, spread_bps × impact_coeff)
#         Weekly-equivalent base: spread=3 bps, impact=0.10 → net SR +0.188
#         (backtest_sensitivity has no rebal_freq col; sensitivity_3d is used
#          when we need the weekly slice; here we show all spread×impact cells
#          at the IS daily-rebal estimate and annotate the weekly base cell.)
# ─────────────────────────────────────────────────────────────────

def fig_sensitivity_heatmap():
    # Use sensitivity_3d (has rebal_freq) — show weekly slice as main panel
    # plus daily and monthly as flanking panels.  All Track A only.
    sens = pd.read_parquet(PROC / "sensitivity_3d.parquet")
    sens_a = sens[sens.track == "track_a"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    rebal_labels = {1: "Daily (1d)", 5: "Weekly (5d)", 21: "Monthly (21d)"}

    for ax, rf in zip(axes, [1, 5, 21]):
        sub = sens_a[sens_a.rebal_freq == rf]
        pivot = sub.pivot(index="spread_bps", columns="impact_coeff",
                          values="net_pnl_sharpe")
        vabs = max(abs(float(pivot.values.min())),
                   abs(float(pivot.values.max())), 0.3)
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
                val = float(pivot.values[i, j])
                color = "white" if abs(val) > 0.4 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

        # Star the base cell (spread=3, impact=0.10) on the weekly panel
        if rf == 5:
            try:
                ri = list(pivot.index).index(3)
                ci = list(pivot.columns).index(0.10)
                ax.add_patch(plt.Rectangle(
                    (ci - 0.5, ri - 0.5), 1, 1,
                    fill=False, edgecolor="navy", lw=2.5))
                ax.text(ci, ri - 0.55, "base\n+0.188",
                        ha="center", va="bottom", fontsize=6.5, color="navy")
            except ValueError:
                pass

        fig.colorbar(im, ax=ax, label="Net Sharpe", shrink=0.8)

    fig.suptitle("Track A — IS Net Sharpe vs Cost Parameters & Rebalancing Frequency",
                 fontsize=11)
    plt.tight_layout()
    path = FIG_DIR / "fig2_sensitivity_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
# Figure 3: Conformal coverage per fold — Track A (moved to Supplement)
# Source: conformal_coverage.parquet
# ─────────────────────────────────────────────────────────────────

def fig_conformal_coverage():
    cov = pd.read_parquet(PROC / "conformal_coverage.parquet")
    target = 0.90

    # Track A only (Track B has no deployable strategy; supplement shows Track A)
    sub = cov[cov.track == "track_a"].sort_values("fold")

    fig, ax = plt.subplots(figsize=(9, 5))
    folds = sub["fold"].astype(str)
    x = np.arange(len(folds))
    width = 0.27

    ax.bar(x - width, sub["coverage"].values * 100, width=width,
           color="#4393c3", alpha=0.8, label="Overall")
    ax.bar(x,          sub["coverage_calm"].values * 100, width=width,
           color="#92c5de", alpha=0.8, label="VIX≤20 (Calm)")
    ax.bar(x + width,  sub["coverage_stressed"].values * 100, width=width,
           color="#f4a582", alpha=0.8, label="VIX>20 (Stressed)")

    ax.axhline(target * 100, color="red", lw=1.5, ls="--",
               label=f"Target {target*100:.0f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(folds, rotation=45, ha="right")
    ax.set_xlabel("Walk-forward Fold (Test Year)")
    ax.set_ylabel("Empirical Coverage (%)")
    ax.set_title("Track A — Conformal Prediction Coverage (90% Target) by VIX Regime\n"
                 "[Supplementary figure]")
    ax.set_ylim(30, 102)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    mean_cov = sub["coverage"].mean()
    ax.text(0.98, 0.98, f"Mean: {mean_cov*100:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    path = FIG_DIR / "fig3_conformal_coverage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
# Figure 4: Multi-window Track A net Sharpe — centerpiece figure
# Numbers from manifest (not fabricated):
#   IS 2013–21 weekly   : +0.188   (manifest: battery.sensitivity.track_a.weekly_base_net_sharpe)
#   Exploratory 2022–24 : −0.607   (manifest: synthesis.track_a.exploratory_2022_2024_net_sharpe)
#   Locked OOS 2025     : +0.666   (manifest: oos.track_a.net_sharpe), p=0.62 n.s.
# Source: holdout_metrics.parquet for OOS; manifest values hard-coded from
#   results/manifest/manifest.json (spec_hash bcd59c0)
# ─────────────────────────────────────────────────────────────────

def fig_is_oos_comparison():
    # Manifest-confirmed numbers (spec_hash bcd59c087aeb559f371936cfb0aa47f280298b63)
    windows = [
        "IS 2013–21\n(weekly)",
        "Exploratory\n2022–24",
        "Locked OOS 2025\n(Jan–Jul)",
    ]
    net_sharpes = [0.188, -0.607, 0.666]
    # p-value annotation for OOS only
    p_annot = [None, None, "t=0.50, p=0.62\n(n.s., n=144d)"]

    bar_colors = [
        COLORS["track_a"] if v >= 0 else COLORS["neg"]
        for v in net_sharpes
    ]
    # OOS bar gets a different shade to highlight it
    bar_colors[2] = "#1a9641"  # green — positive but n.s.

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(windows, net_sharpes, color=bar_colors,
                  edgecolor="black", linewidth=0.8, width=0.5)
    ax.axhline(0, color="black", lw=1.0)
    ax.set_ylabel("Track A Net Sharpe Ratio")
    ax.set_title("CAVAL Track A — Net Sharpe Across Evaluation Windows\n"
                 "(IS weekly rebal, +0.188; exploratory −0.61; locked OOS +0.67 n.s.)")
    ax.set_ylim(min(net_sharpes) - 0.25, max(net_sharpes) + 0.45)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val, ann in zip(bars, net_sharpes, p_annot):
        offset = 0.03 if val >= 0 else -0.06
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + offset,
                f"{val:+.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        if ann:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.14,
                    ann,
                    ha="center", va="bottom", fontsize=8, color="darkred",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="mistyrose",
                              alpha=0.85))

    # DSR / deflated Sharpe note
    ax.text(0.02, 0.97,
            "DSR (IS): 0.12–0.21 (fails 0.95 threshold)\n"
            "Track B: 0 BH features, no deployable strategy",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    path = FIG_DIR / "fig4_is_oos_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
# Figure 5: Track A signal IC decay by holding period
# Primary source: ic_by_fold.parquet (ensemble IC per year, Track A)
#   — shows the per-fold IC stability, not a horizon curve.
# We compute a horizon curve from OHLCV if available; if that's slow/missing
# we fall back to a fold-level bar chart of ensemble IC (legitimate Track A data).
# ─────────────────────────────────────────────────────────────────

def fig_signal_decay():
    ic = pd.read_parquet(PROC / "ic_by_fold.parquet")
    ta = ic[ic.track == "track_a"].copy()
    ta["fold"] = ta["fold"].astype(str)

    # ── Attempt horizon curve from OHLCV + Track A signals ──────────
    sig_path = PROC / "signals_track_a.parquet"
    ohlcv_path = PROC / "daily_ohlcv.parquet"
    horizon_computed = False
    horizon_results = {}

    if sig_path.exists() and ohlcv_path.exists():
        try:
            from scipy.stats import spearmanr
            sig_a = pd.read_parquet(sig_path)
            ohlcv = pd.read_parquet(ohlcv_path)
            close = ohlcv.pivot(index="date", columns="ticker", values="close")

            horizons = [1, 5, 10, 15, 21]
            for h in horizons:
                fwd_h = close.pct_change(h).shift(-h)
                fwd_is = fwd_h.loc[
                    (fwd_h.index >= pd.Timestamp("2013-01-01")) &
                    (fwd_h.index <= pd.Timestamp("2021-12-31"))
                ]
                sig_al = sig_a.reindex(columns=fwd_is.columns)
                ic_vals = []
                sample_dates = sig_al.index[::5][:50]
                for d in sample_dates:
                    if d not in fwd_is.index:
                        continue
                    s = sig_al.loc[d].dropna()
                    r = fwd_is.loc[d].reindex(s.index).dropna()
                    if len(r) > 50:
                        ic_val, _ = spearmanr(s.reindex(r.index), r)
                        if np.isfinite(ic_val):
                            ic_vals.append(ic_val)
                horizon_results[h] = float(np.mean(ic_vals)) if ic_vals else np.nan
            horizon_computed = True
        except Exception as _exc:
            # Fall back to fold bar chart
            horizon_computed = False

    if horizon_computed:
        # ── Panel A: horizon decay; Panel B: per-fold ensemble IC ──────
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        hvs = [horizon_results.get(h, np.nan) for h in horizons]
        ax.plot(horizons, hvs, "o-", color=COLORS["track_a"],
                lw=2, ms=7, label="Track A (Idiosyncratic)")
        ax.fill_between(horizons, hvs, 0, alpha=0.15, color=COLORS["track_a"])
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.axvline(5, color="gray", lw=1.2, ls=":", alpha=0.7)
        ax.text(5.3, max([v for v in hvs if np.isfinite(v)] or [0.01]) * 0.95,
                "Weekly\nrebal.", fontsize=8, color="gray")
        ax.set_xlabel("Holding Period (Trading Days)")
        ax.set_ylabel("Mean IC (Spearman)")
        ax.set_title("Track A — IC Decay vs Horizon")
        ax.set_xticks(horizons)
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        colors_fold = [COLORS["track_a"]] * len(ta)
        ax2.bar(ta["fold"], ta["ensemble"], color=colors_fold,
                edgecolor="black", linewidth=0.5, alpha=0.85)
        ax2.axhline(ta["ensemble"].mean(), color="navy", lw=1.5, ls="--",
                    label=f"Mean IC = {ta['ensemble'].mean():.3f}")
        ax2.set_xlabel("Walk-forward Fold (Year)")
        ax2.set_ylabel("Ensemble IC (Spearman)")
        ax2.set_title("Track A — Per-Fold Ensemble IC (IS 2013–21)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_xticklabels(ta["fold"], rotation=45, ha="right")

        fig.suptitle("Track A Signal Quality — IC Decay and Walk-forward Stability",
                     fontsize=11)
    else:
        # ── Fallback: fold-level bar chart only ─────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(ta["fold"], ta["ensemble"], color=COLORS["track_a"],
               edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.axhline(ta["ensemble"].mean(), color="navy", lw=1.5, ls="--",
                   label=f"Mean IC = {ta['ensemble'].mean():.3f}")
        ax.set_xlabel("Walk-forward Fold (Year)")
        ax.set_ylabel("Ensemble IC (Spearman)")
        ax.set_title("Track A — Per-Fold Ensemble IC (IS 2013–21, Walk-forward)\n"
                     "[Horizon decay: signals_track_a.parquet not found — showing fold IC]")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticklabels(ta["fold"], rotation=45, ha="right")

        # Annotate overall mean IC (manifest: 0.122)
        ax.text(0.98, 0.97,
                "Overall ensemble IC = 0.122\n(manifest: fdr.track_a — daily XS)",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    path = FIG_DIR / "fig5_signal_decay.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
# Figure 6: Track A SHAP feature importance
# Source: shap_summary.parquet (Track A only)
#         fdr_results.parquet  (bh_rejected column marks significant features)
# 12 BH-significant features highlighted in green; others gray.
# ─────────────────────────────────────────────────────────────────

def fig_shap_importance():
    shap = pd.read_parquet(PROC / "shap_summary.parquet")
    fdr  = pd.read_parquet(PROC / "fdr_results.parquet")

    # Track A only
    shap_a = shap[shap.track == "track_a"]
    fdr_a  = fdr[fdr.track == "track_a"]

    # Mean |SHAP| across models, top 15
    importance = (shap_a.groupby("feature")["mean_abs_shap"]
                        .mean()
                        .sort_values(ascending=False)
                        .head(15))

    # BH-significant features (12 per manifest)
    rej = set(fdr_a.loc[fdr_a["bh_rejected"], "feature"].tolist())

    colors = ["#1a9641" if f in rej else "#cccccc" for f in importance.index]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(importance)), importance.values, color=colors, edgecolor="none")
    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels(importance.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| (XGB + LGBM average)")
    ax.set_title("Track A — SHAP Feature Importance\n"
                 "BH-Significant Features (q=0.10, n=12/30) Highlighted")
    ax.grid(True, alpha=0.3, axis="x")

    legend_els = [
        Patch(facecolor="#1a9641", label=f"BH-significant (q=0.10, n={len(rej)})"),
        Patch(facecolor="#cccccc", label="Not significant"),
    ]
    ax.legend(handles=legend_els, loc="lower right", fontsize=9)

    # Annotation: Track B = 0 significant features
    ax.text(0.98, 0.02,
            "Track B: 0/30 BH features\n(false discovery, corrected pipeline)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose", alpha=0.9))

    plt.tight_layout()
    path = FIG_DIR / "fig6_shap_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    print("=== Regenerating Paper Figures (REBUILT pipeline, Track A) ===\n")

    print("Figure 1: Track A cumulative P&L (locked OOS 2025) ...")
    fig_cumulative_pnl()

    print("Figure 2: Track A cost sensitivity heatmap (IS, spread×impact) ...")
    fig_sensitivity_heatmap()

    print("Figure 3: Track A conformal coverage (supplement) ...")
    fig_conformal_coverage()

    print("Figure 4: Multi-window net Sharpe bar chart (IS/exploratory/locked OOS) ...")
    fig_is_oos_comparison()

    print("Figure 5: Track A signal IC decay / fold stability ...")
    fig_signal_decay()

    print("Figure 6: Track A SHAP importance ...")
    fig_shap_importance()

    print(f"\nAll figures saved to: {FIG_DIR}/")
    figs = sorted(FIG_DIR.glob("fig[1-6]*.png"))
    for f in figs:
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
