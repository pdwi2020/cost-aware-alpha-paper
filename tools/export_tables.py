#!/usr/bin/env python3
"""
tools/export_tables.py — Export manuscript table data to results/tables/ as CSV
(and XLSX if openpyxl is available).

Run from the repo root:
    python3 tools/export_tables.py

Output directory: results/tables/
One CSV per table; one XLSX workbook (all_tables.xlsx) when openpyxl is importable.
Idempotent: safe to re-run, always overwrites.

Tables exported:
  fdr_full.csv        — Per-feature FDR results, Track A & B (fdr_results.parquet)
  fdr_regime.csv      — Per-feature × regime FDR (fdr_regime.parquet)
  ic_matrix.csv       — Per-fold IC by 7 models × 2 tracks (ic_by_fold.parquet)
  oos.csv             — OOS inference table (oos_inference.parquet)
  oos_subperiods.csv  — Sub-period OOS breakdown (oos_subperiods.parquet)
  baselines.csv       — Strategy comparison baselines (baselines_metrics.parquet)
  sensitivity.csv     — Cost / rebal sensitivity (backtest_sensitivity.parquet)
  ablation.csv        — Feature ablation OOS (ablation_oos.parquet)
  r2000.csv           — R2000 cross-universe FDR (fdr_results_r2000.parquet)
  shap_summary.csv    — Mean |SHAP| by feature × model (shap_summary.parquet)
  significance.csv    — OOS significance tests (significance_tests.parquet)

Any table whose source parquet is missing is skipped with a SKIP log line.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = ROOT / "results" / "tables"

# ---------------------------------------------------------------------------
# Optional XLSX support
# ---------------------------------------------------------------------------
try:
    import openpyxl  # noqa: F401
    XLSX_OK = True
except ImportError:
    XLSX_OK = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(name: str) -> pd.DataFrame | None:
    """Load a parquet from data/processed/; return None if missing."""
    p = PROCESSED / name
    if not p.exists():
        return None
    return pd.read_parquet(p)


def export(df: pd.DataFrame, name: str, sheets: dict) -> None:
    """Write df to results/tables/<name>.csv and register for XLSX."""
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    stem = Path(name).stem
    sheets[stem] = df
    print(f"  EXPORT  {name:<40s}  {len(df)} rows × {len(df.columns)} cols")


def skip(name: str, reason: str) -> None:
    print(f"  SKIP    {name:<40s}  {reason}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheets: dict[str, pd.DataFrame] = {}

    print(f"\nexport_tables.py — writing to {OUT_DIR}\n")

    # ------------------------------------------------------------------
    # 1. fdr_full.csv  —  per-feature FDR, Track A & B
    #    Source: data/processed/fdr_results.parquet
    #    Columns: track, feature, n_days, ic_bar, t_stat, boot_p,
    #             ci_low, ci_high, bh_adj_p, bh_rejected, bhy_adj_p, bhy_rejected
    # ------------------------------------------------------------------
    df = load("fdr_results.parquet")
    if df is not None:
        export(df.sort_values(["track", "boot_p"]).reset_index(drop=True),
               "fdr_full.csv", sheets)
    else:
        skip("fdr_full.csv", "fdr_results.parquet not found")

    # ------------------------------------------------------------------
    # 2. fdr_regime.csv  —  per-feature × regime FDR
    #    Source: fdr_regime.parquet
    #    Columns: track, feature, regime, n_days, ic_bar, t_stat, boot_p, ...
    # ------------------------------------------------------------------
    df = load("fdr_regime.parquet")
    if df is not None:
        export(df.sort_values(["track", "regime", "boot_p"]).reset_index(drop=True),
               "fdr_regime.csv", sheets)
    else:
        skip("fdr_regime.csv", "fdr_regime.parquet not found")

    # ------------------------------------------------------------------
    # 3. ic_matrix.csv  —  per-fold IC by 7 models × 2 tracks
    #    Source: ic_by_fold.parquet
    #    Columns: track, fold, ridge, lasso, logistic, rf, xgb, lgbm, ensemble
    # ------------------------------------------------------------------
    df = load("ic_by_fold.parquet")
    if df is not None:
        # Reorder columns for readability: track, fold first
        cols = ["track", "fold", "ridge", "lasso", "logistic", "rf", "xgb", "lgbm", "ensemble"]
        cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
        export(df[cols].sort_values(["track", "fold"]).reset_index(drop=True),
               "ic_matrix.csv", sheets)
    else:
        skip("ic_matrix.csv", "ic_by_fold.parquet not found")

    # ------------------------------------------------------------------
    # 4. oos.csv  —  OOS inference (Newey-West t-test + block bootstrap)
    #    Source: oos_inference.parquet
    #    Columns: track, pnl, T, sharpe_ann, lo_t, lo_p, nw_t, nw_p,
    #             boot_ci_lo, boot_ci_hi, boot_p, ...
    # ------------------------------------------------------------------
    df = load("oos_inference.parquet")
    if df is not None:
        export(df.sort_values("track").reset_index(drop=True),
               "oos.csv", sheets)
    else:
        skip("oos.csv", "oos_inference.parquet not found")

    # ------------------------------------------------------------------
    # 5. oos_subperiods.csv  —  sub-period performance breakdown
    #    Source: oos_subperiods.parquet
    # ------------------------------------------------------------------
    df = load("oos_subperiods.parquet")
    if df is not None:
        export(df.sort_values(["track", "period"]).reset_index(drop=True),
               "oos_subperiods.csv", sheets)
    else:
        skip("oos_subperiods.csv", "oos_subperiods.parquet not found")

    # ------------------------------------------------------------------
    # 6. baselines.csv  —  IS + OOS baseline comparison
    #    Source: baselines_metrics.parquet
    #    Columns: strategy, period, gross_sr, net_sr, gross_ann, net_ann,
    #             max_dd, annual_to, cost_drag
    # ------------------------------------------------------------------
    df = load("baselines_metrics.parquet")
    if df is not None:
        export(df.reset_index(drop=True), "baselines.csv", sheets)
    else:
        skip("baselines.csv", "baselines_metrics.parquet not found")

    # ------------------------------------------------------------------
    # 7. sensitivity.csv  —  transaction-cost sensitivity sweep
    #    Source: backtest_sensitivity.parquet
    #    Columns: track, spread_bps, impact_coeff, net_pnl_sharpe, ...
    # ------------------------------------------------------------------
    df = load("backtest_sensitivity.parquet")
    if df is not None:
        export(df.sort_values(["track", "spread_bps", "impact_coeff"]).reset_index(drop=True),
               "sensitivity.csv", sheets)
    else:
        skip("sensitivity.csv", "backtest_sensitivity.parquet not found")

    # ------------------------------------------------------------------
    # 8. ablation.csv  —  feature ablation OOS metrics
    #    Source: ablation_oos.parquet
    #    Columns: variant, n_features, gross_sr, net_sr, max_dd, annual_to, cost_drag
    # ------------------------------------------------------------------
    df = load("ablation_oos.parquet")
    if df is not None:
        export(df.reset_index(drop=True), "ablation.csv", sheets)
    else:
        skip("ablation.csv", "ablation_oos.parquet not found")

    # ------------------------------------------------------------------
    # 9. r2000.csv  —  Russell 2000 cross-universe FDR results
    #    Source: fdr_results_r2000.parquet  (same schema as fdr_results)
    # ------------------------------------------------------------------
    df = load("fdr_results_r2000.parquet")
    if df is not None:
        export(df.sort_values(["track", "boot_p"]).reset_index(drop=True),
               "r2000.csv", sheets)
    else:
        skip("r2000.csv", "fdr_results_r2000.parquet not found")

    # ------------------------------------------------------------------
    # 10. shap_summary.csv  —  mean |SHAP| by feature × model × track
    #     Source: shap_summary.parquet
    #     Columns: feature, model, track, mean_abs_shap
    # ------------------------------------------------------------------
    df = load("shap_summary.parquet")
    if df is not None:
        export(df.sort_values(["track", "model", "mean_abs_shap"],
                              ascending=[True, True, False]).reset_index(drop=True),
               "shap_summary.csv", sheets)
    else:
        skip("shap_summary.csv", "shap_summary.parquet not found")

    # ------------------------------------------------------------------
    # 11. significance.csv  —  Diebold-Mariano + Wilcoxon OOS significance
    #     Source: significance_tests.parquet
    # ------------------------------------------------------------------
    df = load("significance_tests.parquet")
    if df is not None:
        export(df.reset_index(drop=True), "significance.csv", sheets)
    else:
        skip("significance.csv", "significance_tests.parquet not found")

    # ------------------------------------------------------------------
    # XLSX — one sheet per exported table (only if openpyxl is available)
    # ------------------------------------------------------------------
    if not sheets:
        print("\nNo tables exported — nothing to write.")
        return

    if XLSX_OK:
        xlsx_path = OUT_DIR / "all_tables.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                # Sheet names are limited to 31 chars in Excel
                safe_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=safe_name, index=False)
        print(f"\n  XLSX    all_tables.xlsx  ({len(sheets)} sheets)")
    else:
        print("\n  NOTE    openpyxl not importable — XLSX skipped (CSV only)")

    print(f"\nDone. {len(sheets)} table(s) written to {OUT_DIR}\n")


if __name__ == "__main__":
    main()
    sys.exit(0)
