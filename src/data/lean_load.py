"""Memory-lean loading of the feature panel.

Why this exists
---------------
The v3 panel is 2.74M rows x 47 columns. Loading all of it as float64 put this
8 GB machine into roughly 13 GB of swap, and because swap shares the volume with
the data, free disk fell to zero: the pipeline was killed twice, once by
out-of-space and once by the OS reclaiming memory.

No stage needs the whole panel. The models take their columns from
``feature_spec`` regardless of what is passed, so the extra columns (regime
labels, the dropped broadcast macros, adv_usd) were never inputs; they were only
ever ballast. float32 is ample for every statistic computed downstream.

Using this loader is a memory fix, not a methodological change: the columns it
keeps are exactly the ones the estimands are defined on.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.feature_spec import ADDED_INTERACTIONS, KEPT_FEATURES

DEFAULT_TARGETS = ("target_track_a", "target_track_b")
# regime_vix is not a model input, but run_fdr.py splits the daily IC series on
# it. Dropping it does not raise: the regime accumulators simply stay empty and
# the run reports "calm: BH 0, stressed: BH 0", which reads as a finding rather
# than a missing column. It is cheap to carry as a category, so it stays.
DEFAULT_EXTRAS = ("s0_eligible", "regime_vix")


def lean_columns(
    path: str | Path,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    extras: tuple[str, ...] = DEFAULT_EXTRAS,
) -> tuple[list[str], list[str]]:
    """Return (feature_columns, all_columns_to_read) present in the file."""
    import pyarrow.parquet as pq

    available = set(pq.read_schema(str(path)).names)
    features = [c for c in list(KEPT_FEATURES) + list(ADDED_INTERACTIONS)
                if c in available]
    keep = features + [c for c in targets if c in available] \
                    + [c for c in extras if c in available]
    return features, keep


def load_features_lean(
    path: str | Path,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    extras: tuple[str, ...] = DEFAULT_EXTRAS,
    float32: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load the feature panel with only the columns any stage actually uses.

    Keeps the frozen-specification features, the targets, and the Screen 0
    eligibility flag (which several stages use to restrict rows to the
    tradeable universe, so dropping it would silently widen the estimand).
    """
    features, keep = lean_columns(path, targets, extras)
    df = pd.read_parquet(path, columns=keep)

    if float32:
        for col in df.columns:
            if df[col].dtype == "float64":
                df[col] = df[col].astype("float32")

    # Label columns (regime_vix) are low cardinality; as objects they would cost
    # more than every float column put together on a 2.7M-row panel.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype("category")

    if verbose:
        print(
            f"  [lean_load] {df.shape} | {len(features)} features | "
            f"{df.memory_usage(deep=True).sum() / 1e9:.2f} GB in memory",
            flush=True,
        )
    return df
