"""Tests for cap-safe Screen 0 portfolio application rules."""

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtest.portfolio import (
    PortfolioSimulator,
    apply_s0_daily_exit,
    mask_signal_screen0,
)


MAX_POS = 0.05


def _make_simulator(tmp_path):
    """Build a minimal PortfolioSimulator backed by a temp backtest.yaml."""
    cfg = {
        "spread_bps": 3,
        "impact_coeff": 0.10,
        "borrow_cost_easy_bps": 0,
        "gross_exposure_target": 1.0,
        "max_single_position": MAX_POS,
        "aum_dollars": 1e8,
    }
    cfg_path = tmp_path / "backtest.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return PortfolioSimulator(config_path=str(cfg_path))


def _make_features(eligibility):
    index = pd.MultiIndex.from_product(
        [eligibility.columns, eligibility.index],
        names=["ticker", "date"],
    )
    values = eligibility.T.to_numpy().ravel()
    return pd.DataFrame({"s0_eligible": values}, index=index)


def test_screen0_pipeline_preserves_position_cap(tmp_path):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-04", periods=30, freq="B")
    tickers = [f"T{i:03d}" for i in range(50)]
    signal = pd.DataFrame(
        rng.normal(size=(len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )
    eligibility = pd.DataFrame(
        rng.random(signal.shape) > 0.25,
        index=dates,
        columns=tickers,
    )
    feat_df = _make_features(eligibility)
    sim = _make_simulator(tmp_path)

    masked = mask_signal_screen0(signal, feat_df)
    positions = sim.signal_to_positions(masked, lag=0, rebal_freq=1)
    positions = apply_s0_daily_exit(positions, feat_df)

    assert (positions.abs().max(axis=1) <= MAX_POS + 1e-12).all()
    assert (positions.to_numpy()[~eligibility.to_numpy()] == 0.0).all()


def test_apply_s0_daily_exit_does_not_renormalise():
    dates = pd.date_range("2022-02-01", periods=2, freq="B")
    positions = pd.DataFrame(
        [[0.20, -0.15, 0.10], [0.18, -0.12, 0.08]],
        index=dates,
        columns=["A", "B", "C"],
    )
    eligibility = pd.DataFrame(
        [[True, False, True], [True, True, True]],
        index=dates,
        columns=positions.columns,
    )
    expected = positions.copy()
    expected.loc[dates[0], "B"] = 0.0

    actual = apply_s0_daily_exit(positions, _make_features(eligibility))

    pd.testing.assert_frame_equal(actual, expected)


def test_mask_signal_screen0_masks_only_ineligible_cells():
    dates = pd.date_range("2022-03-01", periods=2, freq="B")
    signal = pd.DataFrame(
        [[1.5, -2.0, 0.25], [3.0, 4.0, -1.0]],
        index=dates,
        columns=["A", "B", "C"],
    )
    eligibility = pd.DataFrame(
        [[True, False, True], [False, True, True]],
        index=dates,
        columns=signal.columns,
    )

    actual = mask_signal_screen0(signal, _make_features(eligibility))

    assert actual.isna().to_numpy()[~eligibility.to_numpy()].all()
    np.testing.assert_array_equal(
        actual.to_numpy()[eligibility.to_numpy()],
        signal.to_numpy()[eligibility.to_numpy()],
    )


def test_screen0_helpers_warn_and_return_unchanged_without_flag():
    dates = pd.date_range("2022-04-01", periods=2, freq="B")
    columns = ["A", "B"]
    signal = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=dates, columns=columns)
    positions = pd.DataFrame(
        [[0.2, -0.2], [0.1, -0.1]], index=dates, columns=columns
    )
    feat_df = pd.DataFrame(
        {"other": [True, False, True, False]},
        index=pd.MultiIndex.from_product(
            [columns, dates], names=["ticker", "date"]
        ),
    )

    with pytest.warns(UserWarning):
        masked = mask_signal_screen0(signal, feat_df)
    with pytest.warns(UserWarning):
        exited = apply_s0_daily_exit(positions, feat_df)

    assert masked is signal
    assert exited is positions
    pd.testing.assert_frame_equal(masked, signal)
    pd.testing.assert_frame_equal(exited, positions)
