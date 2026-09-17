"""tests/test_cost_calibration.py — synthetic tests for calibrated costs.

Prices are simulated with a KNOWN bid-ask spread, so the estimators can be
checked for magnitude, monotonicity and non-negativity. No network, no X9.

    python3 -m pytest tests/test_cost_calibration.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.cost_calibration import (  # noqa: E402
    BORROW_GC_BPS,
    BORROW_HTB_BPS,
    IMPACT_GRID,
    abdi_ranaldo_spread,
    borrow_bps_matrix,
    corwin_schultz_spread,
    cost_scenarios,
    half_spread_bps_matrix,
)


def _simulate_panel(tickers, T, spread, seed=0, start_price=100.0):
    """Daily OHLCV with a known proportional spread `spread` (round trip).

    An efficient log price follows a random walk; each day's observed high and
    low are the efficient intraday extremes widened by half the spread on each
    side, and the close is the efficient close plus a random half-spread bounce.
    """
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.bdate_range("2015-01-01", periods=T)
    for t in tickers:
        p = start_price * np.exp(np.cumsum(rng.normal(0.0, 0.012, T)))
        intraday = np.abs(rng.normal(0.0, 0.008, T))
        hi_eff = p * (1 + intraday)
        lo_eff = p * (1 - intraday)
        bounce = rng.choice([-0.5, 0.5], size=T) * spread
        rows.append(pd.DataFrame({
            "ticker": t,
            "date": dates,
            "open": p,
            "high": hi_eff * (1 + spread / 2),
            "low": lo_eff * (1 - spread / 2),
            "close": p * (1 + bounce),
            "volume": rng.uniform(1e6, 5e6, T),
        }))
    return pd.concat(rows, ignore_index=True)


def _wide(panel, col):
    return panel.pivot(index="date", columns="ticker", values=col).sort_index()


class TestEstimators:
    """Magnitude, monotonicity and non-negativity of both spread estimators."""

    @pytest.mark.parametrize("estimator", ["abdi_ranaldo", "corwin_schultz"])
    def test_estimates_are_non_negative(self, estimator):
        panel = _simulate_panel(["AAA", "BBB"], 400, spread=0.002, seed=1)
        h, l, c = (_wide(panel, x) for x in ("high", "low", "close"))
        est = (
            abdi_ranaldo_spread(h, l, c)
            if estimator == "abdi_ranaldo"
            else corwin_schultz_spread(h, l, c)
        )
        vals = est.to_numpy()
        assert np.nanmin(vals) >= 0.0

    @pytest.mark.parametrize("estimator", ["abdi_ranaldo", "corwin_schultz"])
    def test_estimate_increases_with_true_spread(self, estimator):
        """A wider true spread must produce a wider estimate."""
        medians = []
        for spread in (0.0005, 0.002, 0.01):
            panel = _simulate_panel(["AAA"], 600, spread=spread, seed=2)
            h, l, c = (_wide(panel, x) for x in ("high", "low", "close"))
            est = (
                abdi_ranaldo_spread(h, l, c)
                if estimator == "abdi_ranaldo"
                else corwin_schultz_spread(h, l, c)
            )
            medians.append(float(np.nanmedian(est.to_numpy())))
        assert medians[0] < medians[1] < medians[2], medians

    def test_abdi_ranaldo_magnitude_is_in_the_right_ballpark(self):
        """AR should land within a factor of ~3 of the true spread.

        These estimators are noisy and known to be biased for liquid names;
        the paper uses them as a conservative upper cost scenario, so the test
        checks order of magnitude rather than precision.
        """
        true_spread = 0.002
        panel = _simulate_panel([f"T{i}" for i in range(5)], 800, spread=true_spread, seed=3)
        h, l, c = (_wide(panel, x) for x in ("high", "low", "close"))
        est = float(np.nanmedian(abdi_ranaldo_spread(h, l, c).to_numpy()))
        assert true_spread / 3 < est < true_spread * 3, est


class TestHalfSpreadMatrix:
    """The matrix handed to simulate_pnl: units, lag, bounds, completeness."""

    def test_shape_units_and_completeness(self):
        panel = _simulate_panel(["AAA", "BBB"], 300, spread=0.002, seed=4)
        mat, diag = half_spread_bps_matrix(panel)
        close = _wide(panel, "close")
        assert mat.shape == close.shape
        assert list(mat.columns) == list(close.columns)
        assert not mat.isna().any().any()               # always complete
        assert (mat >= diag["fallback_bps"] * 0).all().all()
        assert diag["estimator"] == "abdi_ranaldo"
        assert diag["lag_days"] == 2
        # half-spread of a 20 bps round-trip spread is ~10 bps, not ~1000
        assert 0.5 <= diag["median_bps"] <= 250.0

    def test_bounds_are_applied(self):
        panel = _simulate_panel(["AAA"], 300, spread=0.05, seed=5)   # very wide
        mat, diag = half_spread_bps_matrix(panel, cap_bps=20.0, floor_bps=1.0)
        assert mat.to_numpy().max() <= 20.0 + 1e-9
        assert mat.to_numpy().min() >= 1.0 - 1e-9
        assert diag["pct_capped"] > 0.0

    @pytest.mark.parametrize(
        "estimator,expected_lag", [("corwin_schultz", 1), ("abdi_ranaldo", 2)]
    )
    def test_no_look_ahead(self, estimator, expected_lag):
        """A spike in the high-low range must not affect that day's own spread.

        The spread charged on day t may only depend on information available
        before t: for Corwin-Schultz that is through t-1, and for Abdi-Ranaldo
        through t-2 (its term at date s consumes date s+1's mid-range).
        """
        panel = _simulate_panel(["AAA"], 200, spread=0.002, seed=6)
        spike_pos = 150
        spike_date = sorted(panel["date"].unique())[spike_pos]

        base, base_diag = half_spread_bps_matrix(panel, estimator=estimator, window=10)
        assert base_diag["lag_days"] == expected_lag

        shocked = panel.copy()
        row = shocked["date"] == spike_date
        shocked.loc[row, "high"] *= 1.5
        shocked.loc[row, "low"] *= 0.5
        after, _ = half_spread_bps_matrix(shocked, estimator=estimator, window=10)

        dates = base.index
        # Nothing at or before the spike date may change.
        upto = dates[dates <= spike_date]
        assert np.allclose(
            base.loc[upto, "AAA"].to_numpy(), after.loc[upto, "AAA"].to_numpy(),
            equal_nan=True,
        ), f"{estimator}: spread on or before the spike date changed"
        # The shock must show up eventually, otherwise the test proves nothing.
        later = dates[dates > spike_date]
        assert not np.allclose(
            base.loc[later, "AAA"].to_numpy(), after.loc[later, "AAA"].to_numpy(),
            equal_nan=True,
        ), f"{estimator}: spike never propagated"


class TestBorrowMatrix:
    """Borrow tiers: defaults, the HTB proxy, and its lag."""

    def test_general_collateral_default(self):
        panel = _simulate_panel(["AAA", "BBB"], 120, spread=0.001, seed=7)
        adv = _wide(panel, "close") * _wide(panel, "volume")
        mat, diag = borrow_bps_matrix(panel, adv, adv_decile_cut=0.0, price_floor=0.0,
                                      crash_threshold=-10.0)
        assert (mat.to_numpy() == BORROW_GC_BPS).all()
        assert diag["pct_cells_htb"] == 0.0

    def test_low_price_name_is_flagged_htb(self):
        panel = _simulate_panel(["CHEAP"], 120, spread=0.001, seed=8, start_price=4.0)
        adv = _wide(panel, "close") * _wide(panel, "volume")
        mat, diag = borrow_bps_matrix(panel, adv, adv_decile_cut=0.0, price_floor=10.0,
                                      crash_threshold=-10.0)
        assert (mat.iloc[2:].to_numpy() == BORROW_HTB_BPS).all()
        assert diag["pct_cells_htb"] > 50.0

    def test_htb_proxy_is_lagged(self):
        """A crash on day t may only raise the fee from day t+1 onward."""
        panel = _simulate_panel(["AAA"], 120, spread=0.001, seed=9)
        dates = sorted(panel["date"].unique())
        crash_date = dates[60]
        after_crash = panel["date"] >= crash_date
        panel.loc[after_crash, "close"] *= 0.5          # -50% level shift

        adv = _wide(panel, "close") * _wide(panel, "volume")
        mat, _ = borrow_bps_matrix(panel, adv, adv_decile_cut=0.0, price_floor=0.0,
                                   crash_threshold=-0.30)
        assert mat.loc[crash_date, "AAA"] == BORROW_GC_BPS
        later = mat.loc[mat.index > crash_date, "AAA"]
        assert (later == BORROW_HTB_BPS).any()


def test_impact_grid_and_scenarios_are_documented():
    assert 0.10 in IMPACT_GRID and 0.142 in IMPACT_GRID
    assert all(isinstance(v, str) and v for v in IMPACT_GRID.values())
    scen = cost_scenarios()
    assert set(scen) == {"optimistic_fixed", "calibrated", "conservative"}
    assert scen["optimistic_fixed"]["impact_coeff"] == 0.10
