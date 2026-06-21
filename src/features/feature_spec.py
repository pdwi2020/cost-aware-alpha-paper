"""src/features/feature_spec.py — Canonical pre-registered feature list.

Single source of truth for the feature set defined in config/spec.yaml.
All consumers (FDR, model training, portfolio) import from here.

The module reads spec.yaml once at import time (lightweight YAML parse).
No DuckDB, no parquet, no heavy dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import yaml

# ---------------------------------------------------------------------------
# Load spec.yaml (frozen pre-registration)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent.parent
_SPEC_PATH = _ROOT / "config" / "spec.yaml"


def _load_spec() -> dict:
    with open(_SPEC_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_spec = _load_spec()
_feat_spec = _spec["features"]

# ---------------------------------------------------------------------------
# ETF expansion
# The spec uses the placeholder "corr_sector_etfs" which expands to the
# actual corr_<ETF> columns produced by tier2_extended.py.
# These must match tier2_extended.CROWDING_ETFS exactly.
# ---------------------------------------------------------------------------

# All ETFs from tier2_extended.CROWDING_ETFS
_ALL_CORR_ETFS: list[str] = [
    "corr_SPY", "corr_QQQ",
    "corr_XLK", "corr_XLE", "corr_XLF", "corr_XLY", "corr_XLP",
    "corr_XLI", "corr_XLB", "corr_XLU", "corr_XLV", "corr_XLC", "corr_XLRE",
]

# ---------------------------------------------------------------------------
# KEPT_FEATURES: expand corr_sector_etfs placeholder
# Preserve order: non-placeholder columns first (in spec order), then ETF
# correlations in their canonical tier2 order.
# ---------------------------------------------------------------------------

def _expand_kept() -> list[str]:
    kept: list[str] = []
    for name in _feat_spec["kept"]:
        if name == "corr_sector_etfs":
            kept.extend(_ALL_CORR_ETFS)
        else:
            kept.append(name)
    return kept


KEPT_FEATURES: list[str] = _expand_kept()

# ---------------------------------------------------------------------------
# ADDED_INTERACTIONS: pre-specified stock-level macro interactions
# ---------------------------------------------------------------------------

ADDED_INTERACTIONS: list[str] = list(_feat_spec["added_interactions"])

# ---------------------------------------------------------------------------
# DROPPED: union of dropped_duplicates + dropped_broadcast
# Also include the old ad-hoc interaction term that is superseded.
# ---------------------------------------------------------------------------

DROPPED: set[str] = (
    set(_feat_spec["dropped_duplicates"])
    | set(_feat_spec["dropped_broadcast"])
    | {"term_spread_x_mom"}   # legacy interaction; replaced by prespecified set
)

# ---------------------------------------------------------------------------
# feature_columns(df) → ordered final feature list
#
# Returns:  KEPT (intersected with df columns, preserving order)
#         + ADDED_INTERACTIONS (those present in df)
#   minus:  anything in DROPPED
#
# Regime columns (regime_vix, regime_term_spread), targets, and universe
# flags (s0_eligible, in_universe, adv_usd) are NOT part of this list —
# callers retain them separately.
# ---------------------------------------------------------------------------

def feature_columns(df: "pd.DataFrame") -> list[str]:  # noqa: F821
    """Return the ordered final feature set for a feature DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Any feature frame whose columns include some subset of the
        pre-registered feature names.

    Returns
    -------
    list[str]
        Ordered list: KEPT features (intersected with df.columns, in spec
        order) followed by ADDED_INTERACTIONS (those present in df.columns),
        with DROPPED columns excluded from both.
    """
    df_cols = set(df.columns)

    kept_present = [
        c for c in KEPT_FEATURES
        if c in df_cols and c not in DROPPED
    ]
    added_present = [
        c for c in ADDED_INTERACTIONS
        if c in df_cols and c not in DROPPED
    ]

    return kept_present + added_present
