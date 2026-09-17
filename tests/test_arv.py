"""tests/test_arv.py — Alpha Robustness Volume.

The manuscript reported ARV figures and recommended the measure to readers
while nothing computed it, so the numbers had no source. These tests pin the
definition that `src/backtest/arv.py` now implements.

Run:
    python3 -m pytest tests/test_arv.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.arv import (  # noqa: E402
    alpha_robustness_volume,
    arv_curve,
    arv_report,
)


def test_fraction_above_threshold():
    s = pd.Series([-0.2, 0.0, 0.1, 0.4, 0.9])
    assert alpha_robustness_volume(s, 0.0) == pytest.approx(3 / 5)
    assert alpha_robustness_volume(s, 0.5) == pytest.approx(1 / 5)
    assert alpha_robustness_volume(s, 1.0) == pytest.approx(0.0)


def test_threshold_is_strict():
    """A config exactly at the threshold does not count as clearing it."""
    s = pd.Series([0.5, 0.5, 0.6])
    assert alpha_robustness_volume(s, 0.5) == pytest.approx(1 / 3)


def test_arv_is_monotone_non_increasing_in_threshold():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0.2, 0.5, size=60))
    curve = arv_curve(pd.DataFrame({"net_pnl_sharpe": s}))
    vals = [curve[t] for t in sorted(curve)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:])), vals


def test_nan_configs_are_dropped_not_counted_as_failures():
    """A config that produced no result must not be scored against the strategy."""
    s = pd.Series([0.6, np.nan, 0.7, np.inf])
    # Two finite values, both above 0.5.
    assert alpha_robustness_volume(s, 0.5) == pytest.approx(1.0)


def test_all_nan_returns_nan_rather_than_zero():
    assert np.isnan(alpha_robustness_volume(pd.Series([np.nan, np.nan]), 0.0))


def test_report_carries_grid_facts_and_respects_track():
    grid = pd.DataFrame({
        "track": ["track_a"] * 4 + ["track_b"] * 4,
        "net_pnl_sharpe": [0.1, 0.2, 0.3, 0.4, -0.5, -0.4, -0.3, -0.2],
    })
    a = arv_report(grid, "track_a")
    b = arv_report(grid, "track_b")
    assert a["n_configs"] == 4 and b["n_configs"] == 4
    assert a["arv"]["0"] == pytest.approx(1.0)
    assert b["arv"]["0"] == pytest.approx(0.0)
    assert a["sharpe_max"] == pytest.approx(0.4)
    assert b["sharpe_min"] == pytest.approx(-0.5)


def test_missing_sharpe_column_raises():
    with pytest.raises(KeyError):
        arv_curve(pd.DataFrame({"something_else": [1.0]}))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
