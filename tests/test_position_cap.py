"""Tests for the iterative water-fill position cap.

pytest tests/test_position_cap.py -q
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.portfolio import PortfolioSimulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_POS = 0.05
GROSS_TARGET = 1.0
TOL = 1e-9


def _waterfill(w, max_pos=MAX_POS, gross_target=GROSS_TARGET):
    return PortfolioSimulator._waterfill_cap(
        w, max_pos=max_pos, gross_target=gross_target
    )


def _make_simulator(tmp_path, n_names=100):
    """Build a minimal PortfolioSimulator backed by a temp backtest.yaml."""
    cfg = {
        "spread_bps": 3,
        "impact_coeff": 0.10,
        "borrow_cost_easy_bps": 0,
        "gross_exposure_target": GROSS_TARGET,
        "max_single_position": MAX_POS,
        "aum_dollars": 1e8,
    }
    import yaml
    cfg_path = tmp_path / "backtest.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return PortfolioSimulator(config_path=str(cfg_path))


# ---------------------------------------------------------------------------
# 1. Cap respected on every day
# ---------------------------------------------------------------------------

def test_cap_respected_every_day(tmp_path):
    """After signal_to_positions, |w_i| <= max_pos for every ticker on every date."""
    rng = np.random.default_rng(42)
    n_dates, n_names = 50, 200
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i:04d}" for i in range(n_names)]
    signals = pd.DataFrame(rng.standard_normal((n_dates, n_names)),
                           index=dates, columns=tickers)

    sim = _make_simulator(tmp_path, n_names=n_names)
    pos = sim.signal_to_positions(signals, lag=0, rebal_freq=1)

    max_weight = pos.abs().max().max()
    assert max_weight <= MAX_POS + TOL, (
        f"Position cap breached: max |w| = {max_weight:.6f} > {MAX_POS}"
    )


# ---------------------------------------------------------------------------
# 2. Gross preserved when feasible (many names)
# ---------------------------------------------------------------------------

def test_gross_preserved_when_feasible(tmp_path):
    """With n_names >> 1/max_pos, the gross exposure should equal gross_target."""
    rng = np.random.default_rng(7)
    n_dates, n_names = 30, 500   # 500 * 0.05 = 25 >> gross_target = 1
    dates = pd.date_range("2021-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i:04d}" for i in range(n_names)]
    signals = pd.DataFrame(rng.standard_normal((n_dates, n_names)),
                           index=dates, columns=tickers)

    sim = _make_simulator(tmp_path)
    pos = sim.signal_to_positions(signals, lag=0, rebal_freq=1)

    gross = pos.abs().sum(axis=1)
    # Allow 1 bps tolerance; on days with >=10 names and non-zero signals
    # the gross should be exactly gross_target
    feasible_days = gross > 1e-6
    assert feasible_days.sum() > 0, "No feasible days found"
    max_err = (gross[feasible_days] - GROSS_TARGET).abs().max()
    assert max_err < 1e-8, (
        f"Gross not preserved: max deviation = {max_err:.2e}"
    )


# ---------------------------------------------------------------------------
# 3. Infeasible case: n * max_pos < gross_target
# ---------------------------------------------------------------------------

def test_infeasible_all_capped_no_breach():
    """When n_active * max_pos < gross_target, every |w| == max_pos and cap is not breached."""
    # With 5 names and max_pos=0.05, gross_max = 0.25 < 1.0
    n = 5
    w = pd.Series([0.2, 0.3, -0.15, -0.25, 0.1],
                  index=[f"T{i}" for i in range(n)])
    # Normalise to gross_target first
    gross = w.abs().sum()
    w = w / gross * GROSS_TARGET

    result = _waterfill(w, max_pos=MAX_POS, gross_target=GROSS_TARGET)

    # No breach
    assert result.abs().max() <= MAX_POS + TOL, (
        f"Cap breached in infeasible case: {result.abs().max():.6f}"
    )
    # All capped at max_pos
    assert (result.abs() - MAX_POS).abs().max() < TOL, (
        f"Some names not at max_pos in infeasible case: {result.abs().values}"
    )


# ---------------------------------------------------------------------------
# 4a. Idempotency: applying waterfill twice yields the same result
# ---------------------------------------------------------------------------

def test_idempotent():
    """Applying the cap twice gives the same result as once."""
    rng = np.random.default_rng(99)
    w_raw = pd.Series(rng.standard_normal(100))
    w_raw = w_raw / w_raw.abs().sum() * GROSS_TARGET

    once  = _waterfill(w_raw)
    twice = _waterfill(once)

    pd.testing.assert_series_equal(once, twice, atol=1e-12, check_names=False)


# ---------------------------------------------------------------------------
# 4b. Sign-preserving: output signs match input signs
# ---------------------------------------------------------------------------

def test_sign_preserving():
    """The cap must not flip any sign."""
    rng = np.random.default_rng(55)
    w_raw = pd.Series(rng.standard_normal(80))
    w_raw = w_raw / w_raw.abs().sum() * GROSS_TARGET

    result = _waterfill(w_raw)

    sign_match = (np.sign(result) == np.sign(w_raw)).all()
    assert sign_match, "Sign flipped by water-fill cap"


# ---------------------------------------------------------------------------
# 4c. Order-preserving: larger |signal| gets >= weight (until cap)
# ---------------------------------------------------------------------------

def test_order_preserving():
    """A name with a larger absolute signal gets at least as large |w| (until both capped)."""
    rng = np.random.default_rng(13)
    w_raw = pd.Series(rng.standard_normal(100))
    w_raw = w_raw / w_raw.abs().sum() * GROSS_TARGET

    result = _waterfill(w_raw)

    # For each pair (i, j) where |w_raw[i]| > |w_raw[j]| + tol
    # assert |result[i]| >= |result[j]| - tol (or both at cap)
    abs_in  = w_raw.abs().values
    abs_out = result.abs().values

    violations = 0
    for i in range(len(w_raw)):
        for j in range(len(w_raw)):
            if abs_in[i] > abs_in[j] + 1e-10:
                # i had strictly larger signal
                if abs_out[i] < abs_out[j] - 1e-10:
                    # only a violation if i is not capped
                    if abs_out[i] < MAX_POS - 1e-10:
                        violations += 1

    assert violations == 0, (
        f"Order violated on {violations} pairs (uncapped name with larger signal got smaller weight)"
    )


# ---------------------------------------------------------------------------
# Direct unit tests on _waterfill_cap helper
# ---------------------------------------------------------------------------

def test_waterfill_no_clip_needed():
    """When all |w_i| <= max_pos already, output equals input (no change needed).

    Build a 40-name series with each weight = GROSS_TARGET/40 = 0.025 < MAX_POS.
    The algorithm should return values equal to the input (up to float tol).
    """
    n = 40  # gross_target / n = 0.025 < max_pos = 0.05 — no clipping needed
    vals = np.array([(-1) ** i * GROSS_TARGET / n for i in range(n)])
    w = pd.Series(vals)
    # Confirm input is already gross-normalised and under cap
    assert w.abs().sum() == pytest.approx(GROSS_TARGET, abs=1e-12)
    assert w.abs().max() <= MAX_POS

    result = _waterfill(w)

    pd.testing.assert_series_equal(result, w, atol=1e-12, check_names=False)


def test_waterfill_single_over_cap():
    """One dominant name gets capped; the rest absorb the freed gross.

    Use enough names so the feasibility condition holds: (n-1) * max_pos >= freed gross.
    """
    # 30 names; first carries 0.5 gross, remaining 29 share 0.5 evenly → 0.5/29 ≈ 0.017 each
    n = 30
    vals = [0.5] + [0.5 / (n - 1)] * (n - 1)  # gross = 1.0, first >> max_pos
    w = pd.Series(vals)
    assert w.abs().sum() == pytest.approx(GROSS_TARGET, abs=1e-12)

    result = _waterfill(w)

    assert result.abs().max() <= MAX_POS + TOL
    assert abs(result.abs().sum() - GROSS_TARGET) < 1e-9
    # First name should sit exactly at max_pos
    assert abs(abs(result.iloc[0]) - MAX_POS) < TOL


def test_waterfill_all_equal():
    """Uniform weights well below cap are returned unchanged."""
    n = 100
    w = pd.Series([GROSS_TARGET / n] * n)
    result = _waterfill(w)

    pd.testing.assert_series_equal(result, w, atol=1e-12, check_names=False)
