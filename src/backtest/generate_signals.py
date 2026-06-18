"""Week 8: Build composite alpha signal from BH-significant features.

Signal construction:
  For each track, take BH-rejected features from fdr_results.parquet.
  Weight each feature by avg SHAP importance (XGB+LGBM) × sign(mean_IC).
  Normalise weights to sum to 1 in absolute value.
  Signal = weighted sum of raw feature values (cross-sectional z-score
  is applied later inside signal_to_positions).

Outputs:
    data/processed/signals_track_a.parquet   — (date × ticker) raw signal scores
    data/processed/signals_track_b.parquet   — (date × ticker) raw signal scores

Run:
    python3 -u src/backtest/generate_signals.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.universe_paths import proc

FDR_PATH  = proc(ROOT, "fdr_results.parquet")
SHAP_PATH = proc(ROOT, "shap_summary.parquet")
FEAT_PATH = proc(ROOT, "features_all.parquet")

OUT_A = proc(ROOT, "signals_track_a.parquet")
OUT_B = proc(ROOT, "signals_track_b.parquet")


def log(msg): print(msg, flush=True)


def build_signal_weights(track: str, fdr_df: pd.DataFrame, shap_df: pd.DataFrame) -> pd.Series:
    """Return signed SHAP weights for BH-significant features.

    Weight_f = sign(mean_IC_f) × avg_SHAP_f, then L1-normalised.
    """
    rejected = fdr_df[(fdr_df["track"] == track) & fdr_df["rejected"]].set_index("feature")
    if rejected.empty:
        raise ValueError(f"No BH-rejected features for {track}")

    # Average SHAP importance across XGB + LGBM
    shap_avg = (
        shap_df[shap_df["track"] == track]
        .groupby("feature")["mean_abs_shap"]
        .mean()
    )

    weights = {}
    for feat in rejected.index:
        sign = np.sign(rejected.loc[feat, "mean_ic"])
        shap_w = float(shap_avg.get(feat, 0.0))
        weights[feat] = sign * shap_w

    w = pd.Series(weights)
    # Use uniform weight for features not in SHAP summary (shouldn't happen but safety)
    zero_shap = w[w.abs() < 1e-12].index
    if len(zero_shap) > 0:
        log(f"  [{track}] Features with zero SHAP weight (using uniform): {zero_shap.tolist()}")
        w[zero_shap] = w[w.abs() >= 1e-12].mean() if (w.abs() >= 1e-12).any() else 1.0

    # L1-normalise so gross weight = 1
    denom = w.abs().sum()
    if denom > 1e-12:
        w = w / denom
    return w


def generate_composite_signal(
    feat_df: pd.DataFrame,
    weights: pd.Series,
    start_date: str = "2013-01-01",
    end_date: str = "2021-12-31",
) -> pd.DataFrame:
    """Compute signal matrix (date × ticker) for the IS+walk-forward period.

    Signal_t,i = sum_f (w_f × feature_{t,i,f})

    Missing feature values filled with cross-sectional median to avoid
    signal NaN propagation.
    """
    features = [f for f in weights.index if f in feat_df.columns]
    missing  = [f for f in weights.index if f not in feat_df.columns]
    if missing:
        log(f"  Warning: features not in data, skipping: {missing}")
    if not features:
        raise ValueError("No feature columns found in data")

    # Slice date range
    dates = feat_df.index.get_level_values("date")
    mask  = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
    sub   = feat_df.loc[mask, features]

    w_sub = weights.loc[features]

    # Pivot to (date × ticker) for each feature → compute weighted sum
    all_dates = sub.index.get_level_values("date").unique().sort_values()
    all_tickers = sub.index.get_level_values("ticker").unique()

    records = []
    for d in all_dates:
        day_data = sub.xs(d, level="date")          # ticker × features
        # Cross-sectional median fill per feature
        row_signal = pd.Series(0.0, index=day_data.index)
        for feat, w in w_sub.items():
            col = day_data[feat].copy()
            med = col.median()
            col = col.fillna(med)
            row_signal += w * col
        records.append(row_signal.rename(d))

    signal_df = pd.DataFrame(records)   # date × ticker
    return signal_df


def main():
    log("=== Week 8: Composite Signal Generation ===\n")
    t0 = time.time()

    fdr_df  = pd.read_parquet(FDR_PATH)
    shap_df = pd.read_parquet(SHAP_PATH)

    log("Loading features_all.parquet …")
    feat_df = pd.read_parquet(FEAT_PATH)

    for track, out_path in [("track_a", OUT_A), ("track_b", OUT_B)]:
        log(f"\n{'='*55}")
        log(f"  Building signal: {track.upper()}")
        log(f"{'='*55}")

        weights = build_signal_weights(track, fdr_df, shap_df)
        n_sig = (fdr_df[(fdr_df.track == track) & fdr_df.rejected]).shape[0]
        log(f"  BH-significant features: {n_sig}")
        log(f"  Features in signal:")
        for f, w in weights.sort_values(key=abs, ascending=False).items():
            log(f"    {f:<30s}  weight={w:+.4f}")

        t1 = time.time()
        signal_df = generate_composite_signal(feat_df, weights)
        log(f"\n  Signal matrix: {signal_df.shape}  ({time.time()-t1:.1f}s)")
        log(f"  Date range: {signal_df.index.min().date()} → {signal_df.index.max().date()}")
        log(f"  Tickers: {signal_df.shape[1]}")
        log(f"  Non-zero signals: {(signal_df.abs() > 1e-10).sum().sum()}")

        signal_df.to_parquet(out_path)
        log(f"  Saved → {out_path}")

    log(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
