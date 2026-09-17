"""tests/test_factor_alpha.py — synthetic tests for the factor-alpha module.

No network, no X9 access: factors are constructed in memory and passed in, so
these tests run anywhere.

    python3 -m pytest tests/test_factor_alpha.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.factor_alpha import (  # noqa: E402
    FACTOR_COLUMNS,
    _default_hac_lags,
    factor_regression,
)


def _synthetic_factors(T: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=T)
    data = {c: rng.normal(0.0002, 0.008, T) for c in FACTOR_COLUMNS}
    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


class TestRecovery:
    """Known alpha and betas must be recovered from synthetic data."""

    def test_recovers_known_alpha_and_betas(self):
        T = 3000
        factors = _synthetic_factors(T, seed=1)
        rng = np.random.default_rng(2)
        true_alpha = 0.0003
        true_betas = {"mkt_rf": 0.0, "smb": 0.35, "hml": -0.20,
                      "rmw": 0.0, "cma": 0.0, "mom": -0.45}
        pnl = (
            true_alpha
            + sum(b * factors[c] for c, b in true_betas.items())
            + rng.normal(0.0, 0.002, T)
        )

        res = factor_regression(pnl, factors=factors)

        assert res["n_obs"] == T
        # The estimator's own standard error is about resid_sd/sqrt(T) =
        # 0.002/sqrt(3000) ~= 3.7e-5, so anything tighter than a few standard
        # errors would fail on ordinary sampling noise. Allow ~4 SE.
        assert abs(res["alpha_daily"] - true_alpha) < 1.5e-4
        for c, b in true_betas.items():
            assert abs(res["betas"][c]["beta"] - b) < 0.05, c
        assert res["betas"]["mom"]["t"] < -5      # strong negative momentum load
        assert res["r2"] > 0.5

    def test_zero_alpha_is_not_significant(self):
        T = 2000
        factors = _synthetic_factors(T, seed=3)
        rng = np.random.default_rng(4)
        pnl = 0.4 * factors["mom"] + rng.normal(0.0, 0.003, T)

        res = factor_regression(pnl, factors=factors)
        assert abs(res["alpha_t"]) < 2.5
        assert res["alpha_p"] > 0.01

    def test_hedged_sharpe_strips_factor_variance(self):
        """A book that is pure factor exposure has ~zero residual Sharpe."""
        T = 2000
        factors = _synthetic_factors(T, seed=5)
        rng = np.random.default_rng(6)
        pnl = 1.0 * factors["mom"] + rng.normal(0.0, 1e-5, T)

        res = factor_regression(pnl, factors=factors)
        assert res["r2"] > 0.99
        assert abs(res["resid_sharpe_ann"]) < abs(res["raw_sharpe_ann"]) + 1e-9


class TestContract:
    """Input handling and the HAC lag rule."""

    def test_rejects_too_few_overlapping_days(self):
        factors = _synthetic_factors(500, seed=7)
        pnl = pd.Series(
            np.random.default_rng(8).normal(0, 0.01, 10), index=factors.index[:10]
        )
        with pytest.raises(ValueError, match="overlapping days"):
            factor_regression(pnl, factors=factors)

    def test_aligns_on_dates_and_drops_missing(self):
        factors = _synthetic_factors(400, seed=9)
        pnl = pd.Series(
            np.random.default_rng(10).normal(0.0, 0.01, 400), index=factors.index
        )
        pnl.iloc[:50] = np.nan                       # leading gap
        res = factor_regression(pnl, factors=factors)
        assert res["n_obs"] == 350
        assert res["start"] == str(factors.index[50].date())

    def test_hac_lag_rule(self):
        assert _default_hac_lags(100) == 4
        assert _default_hac_lags(2267) >= 7
        factors = _synthetic_factors(300, seed=11)
        pnl = pd.Series(
            np.random.default_rng(12).normal(0.0, 0.01, 300), index=factors.index
        )
        assert factor_regression(pnl, factors=factors, hac_lags=3)["hac_lags"] == 3
