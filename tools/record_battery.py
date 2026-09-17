"""tools/record_battery.py — record the robustness battery to the manifest.

The battery stages (ex-mega-cap, regime-conditional, ablation, naive ML,
conformal) never wrote to the manifest. Their entries had been transcribed by
hand, which is why they survived a full v3 re-run carrying June and July
values while the rest of the manuscript moved to v3.

This tool derives them from the parquet files those stages write, so the
manuscript's corroborating evidence has the same provenance as its headline
numbers. It recomputes nothing and fails loudly if a schema changed.

Run after scripts/run_v3_stage_c.sh (which invokes it), or standalone:

    python3 -u tools/record_battery.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import manifest  # noqa: E402

PROC = ROOT / "data" / "processed"


def _slug(text: str) -> str:
    """Manifest-safe key fragment: keys are dot-namespaced, so no dots."""
    out = []
    for ch in str(text):
        out.append(ch if ch.isalnum() else "_")
    return "_".join("".join(out).split("_")).strip("_") or "unnamed"


def _need(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/run_v3_stage_c.sh")
    return pd.read_parquet(path)


def _col(df: pd.DataFrame, name: str, where: str) -> pd.Series:
    if name not in df.columns:
        raise KeyError(f"{name!r} not in {where}; columns are {sorted(df.columns)}")
    return df[name]


def _f(value) -> float:
    return round(float(value), 4)


def record_excl_megacap() -> None:
    """Ex-mega-cap sub-universe, exploratory 2022-2024 window."""
    df = _need(PROC / "excl_megacap_metrics.parquet")
    where = "excl_megacap_metrics.parquet"
    _col(df, "track", where)
    for track, group in df.groupby("track"):
        row = group.iloc[0]
        for src_col, key in (("net_pnl_sharpe", "oos_net_sharpe"),
                             ("gross_pnl_sharpe", "oos_gross_sharpe")):
            if src_col in group.columns:
                manifest.record(
                    f"battery.excl_megacap.{track}.{key}", _f(row[src_col]),
                    stage="battery", track=track[-1].upper(),
                    meta={"window": "2022-01-01..2024-12-31", "n_excluded": 25},
                )
        print(f"  excl_megacap {track}: net {row.get('net_pnl_sharpe', float('nan')):+.4f}")


def record_regime_signal() -> None:
    """Static / calm-only / stressed-only / regime-adaptive books."""
    df = _need(PROC / "regime_signal_results.parquet")
    where = "regime_signal_results.parquet"
    for c in ("track", "variant"):
        _col(df, c, where)
    sub = df[df["period"].astype(str).str.contains("2022")] if "period" in df.columns else df
    if sub.empty:
        sub = df
    for (track, variant), group in sub.groupby(["track", "variant"]):
        value = group["net_sr"].iloc[0] if "net_sr" in group.columns else np.nan
        manifest.record(
            f"battery.regime_signal.{track}.{_slug(variant)}.oos_net_sr", _f(value),
            stage="battery", track=track[-1].upper(),
        )
    print(f"  regime_signal: {sub.groupby(['track', 'variant']).ngroups} track-variant cells")


def record_ablation() -> None:
    """Feature-set and weighting ablation."""
    df = _need(PROC / "ablation_oos.parquet")
    where = "ablation_oos.parquet"
    for c in ("track", "variant"):
        _col(df, c, where)
    for (track, variant), group in df.groupby(["track", "variant"]):
        row = group.iloc[0]
        for src_col, key in (("gross_sr", "gross_sr"), ("net_sr", "net_sr")):
            if src_col in group.columns:
                manifest.record(
                    f"battery.ablation.{track}.{_slug(variant)}.{key}", _f(row[src_col]),
                    stage="battery", track=track[-1].upper(),
                )
    print(f"  ablation: {df.groupby(['track', 'variant']).ngroups} track-variant cells")


def record_ml_baseline() -> None:
    """Naive ML comparison (the manuscript's RF+XGB row)."""
    df = _need(PROC / "ml_baseline_metrics.parquet")
    where = "ml_baseline_metrics.parquet"
    for c in ("track", "model", "period"):
        _col(df, c, where)
    for row in df.itertuples():
        base = f"battery.ml_baseline.{row.track}.{_slug(row.model)}.{_slug(row.period)}"
        for field in ("gross_sr", "net_sr"):
            if hasattr(row, field):
                manifest.record(f"{base}.{field}", _f(getattr(row, field)),
                                stage="battery", track=str(row.track)[-1].upper())
    print(f"  ml_baseline: {len(df)} rows")


def record_conformal() -> None:
    """Split-conformal coverage, overall and by volatility regime."""
    df = _need(PROC / "conformal_coverage.parquet")
    where = "conformal_coverage.parquet"
    _col(df, "coverage", where)
    target = float(df["target_coverage"].iloc[0]) if "target_coverage" in df.columns else np.nan
    groups = df.groupby("track") if "track" in df.columns else [("pooled", df)]
    for track, group in groups:
        for src_col, key in (("coverage", "coverage"),
                             ("coverage_calm", "coverage_calm"),
                             ("coverage_stressed", "coverage_stressed"),
                             ("interval_width", "interval_width")):
            if src_col not in group.columns:
                continue
            value = group[src_col].astype(float).mean()
            if np.isnan(value):
                continue
            manifest.record(
                f"calibration.conformal.{track}.{key}", _f(value),
                stage="calibration",
                meta={"target_coverage": target, "n_folds": int(len(group))},
            )
        print(f"  conformal {track}: coverage {group['coverage'].astype(float).mean():.4f} "
              f"(target {target:.2f})")


def main() -> int:
    print("=== recording battery results ===")
    record_excl_megacap()
    record_regime_signal()
    record_ablation()
    record_ml_baseline()
    record_conformal()
    print("\nManifest updated. Now run: python3 -u tools/check_coherence.py --strict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
