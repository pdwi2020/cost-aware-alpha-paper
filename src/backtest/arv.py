"""Alpha Robustness Volume (ARV): cost robustness as a single number.

What it is
----------
ARV(s) is the fraction of a cost-assumption grid on which a strategy's net
annualised Sharpe ratio still exceeds `s`. ARV(0) asks how often the strategy
is merely profitable; ARV(SR*) asks how often it clears the economic
threshold. A strategy whose verdict depends on one lucky corner of the grid has
a low ARV even if its base-case number looks fine, which is the situation the
measure exists to expose.

Why this module exists
----------------------
The manuscript reported ARV figures and recommended the measure to
practitioners, but nothing in the codebase computed it: the numbers had no
source and could not be checked. This module computes ARV from the sensitivity
grid that `run_sensitivity.py` already writes, so the reported values are
generated rather than asserted.

The grid is 5 half-spreads x 4 impact coefficients x 3 rebalancing
frequencies = 60 configurations (configs/backtest.yaml).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Thresholds reported in the paper: 0 (profitable at all) and the economic
# effect-size threshold with its sensitivity grid.
DEFAULT_THRESHOLDS = (0.0, 0.15, 0.3, 0.5, 0.75)


def alpha_robustness_volume(
    sharpes: np.ndarray | pd.Series,
    threshold: float,
) -> float:
    """Fraction of grid configurations whose net Sharpe exceeds `threshold`.

    NaN configurations are dropped rather than counted as failures, so the
    denominator is the number of configurations that actually produced a
    result.
    """
    s = pd.Series(sharpes, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return float("nan")
    return float((s > threshold).mean())


def arv_curve(
    grid: pd.DataFrame,
    sharpe_col: str = "net_pnl_sharpe",
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[float, float]:
    """ARV at each threshold, as {threshold: fraction}."""
    if sharpe_col not in grid.columns:
        raise KeyError(
            f"{sharpe_col!r} not in the sensitivity grid; columns are "
            f"{sorted(grid.columns)}"
        )
    return {t: alpha_robustness_volume(grid[sharpe_col], t) for t in thresholds}


def arv_report(
    grid: pd.DataFrame,
    track: str,
    sharpe_col: str = "net_pnl_sharpe",
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict:
    """ARV plus the grid facts a reader needs to interpret it."""
    sub = grid[grid["track"] == track] if "track" in grid.columns else grid
    s = pd.Series(sub[sharpe_col], dtype=float).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    curve = arv_curve(sub, sharpe_col=sharpe_col, thresholds=thresholds)
    return {
        "track": track,
        "n_configs": int(len(s)),
        "n_configs_total": int(len(sub)),
        "sharpe_min": float(s.min()) if len(s) else float("nan"),
        "sharpe_median": float(s.median()) if len(s) else float("nan"),
        "sharpe_max": float(s.max()) if len(s) else float("nan"),
        "arv": {f"{t:g}": curve[t] for t in thresholds},
    }
