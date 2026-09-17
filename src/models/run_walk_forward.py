"""Week 4: Walk-forward cross-validation over all 6 models.

Expanding window: warmup 2010-2012, step 1 year, 9 test folds (2013-2021).
Runs Track A (idiosyncratic) and Track B (raw) targets separately.
Saves fold-by-fold IC to data/processed/ic_by_fold.parquet.

Run:
    cd ~/ML_Paper && python3 -u src/models/run_walk_forward.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.models.model_suite import ModelSuite, make_fold_dates
from src.universe_paths import proc

FEATURES_PATH = proc(ROOT, "features_all.parquet")
OUT_PATH      = proc(ROOT, "ic_by_fold.parquet")
CONFIG_PATH   = ROOT / "configs" / "models.yaml"


def log(msg: str) -> None:
    print(msg, flush=True)


def main():
    log("=== Week 4: Walk-Forward Cross-Validation ===\n")
    t_start = time.time()

    # ------------------------------------------------------------------
    # 1. Load features
    # ------------------------------------------------------------------
    # Lean load: only the modelled columns, and float32 rather than float64.
    # The full 47-column float64 panel pushed this machine into ~13 GB of swap
    # on a volume with under 2 GB free, which is a hard failure rather than a
    # slow one. The models are insensitive to the dropped precision.
    import pyarrow.parquet as pq

    from src.features.feature_spec import ADDED_INTERACTIONS, KEPT_FEATURES

    log("Loading features_all.parquet (modelled columns only, float32)...")
    available = set(pq.read_schema(FEATURES_PATH).names)
    feat_cols = [c for c in list(KEPT_FEATURES) + list(ADDED_INTERACTIONS)
                 if c in available]
    target_cols = [c for c in ("target_track_a", "target_track_b") if c in available]

    # s0_eligible must come along: ModelSuite.fit_all_folds restricts training
    # and test rows to Screen 0-eligible names when the column is present, so
    # dropping it would silently widen the estimand.
    keep_cols = feat_cols + [c for c in ("s0_eligible",) if c in available]

    df = pd.read_parquet(FEATURES_PATH, columns=keep_cols + target_cols)
    df.index = df.index.set_levels(
        [df.index.levels[0], pd.to_datetime(df.index.levels[1])],
    )
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
    log(f"  Shape: {df.shape}  |  Feature cols: {len(feat_cols)}  "
        f"|  {df.memory_usage(deep=True).sum() / 1e9:.2f} GB in memory")

    features = df[keep_cols]

    # ------------------------------------------------------------------
    # 2. Fold dates
    # ------------------------------------------------------------------
    fold_dates = make_fold_dates(
        train_start="2010-01-01",
        warmup_end_year=2012,
        test_end_year=2021,
    )
    log(f"  Folds: {[f['fold_id'] for f in fold_dates]}")

    # ------------------------------------------------------------------
    # 3. Run both tracks
    # ------------------------------------------------------------------
    all_results = []

    for track, target_col in [("track_b", "target_track_b"),
                               ("track_a", "target_track_a")]:
        log(f"\n{'='*60}")
        log(f"  Track: {track.upper()}  (target = {target_col})")
        log(f"{'='*60}")

        targets = df[target_col]
        log(f"  Target coverage: {targets.notna().mean():.3f}")

        suite = ModelSuite(config_path=str(CONFIG_PATH))
        t0 = time.time()
        ic_df = suite.fit_all_folds(features, targets, fold_dates)
        elapsed = time.time() - t0

        log(f"\n  Track {track.upper()} done in {elapsed:.0f}s")
        log(f"\n  IC Summary ({track}):")
        log(ic_df.describe().round(4).to_string())
        log(f"\n  Mean IC across folds:")
        log(ic_df.mean().round(4).to_string())

        ic_df["track"] = track
        ic_df.index.name = "fold"
        all_results.append(ic_df.reset_index())

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    out = pd.concat(all_results, ignore_index=True)
    out.to_parquet(OUT_PATH, index=False)
    log(f"\nSaved → {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e3:.0f} KB)")
    log(f"Total elapsed: {(time.time() - t_start) / 60:.1f} min")
    return out


if __name__ == "__main__":
    main()
