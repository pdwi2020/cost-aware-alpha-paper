"""Synthetic tests for the shared strategy book-building pipeline."""

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtest.books import composite_book, load_market_panel, single_feature_book
from src.backtest.generate_signals import generate_composite_signal
from src.backtest.portfolio import (
    PortfolioSimulator,
    apply_s0_daily_exit,
    mask_signal_screen0,
)
from src.backtest.run_backtest import build_returns_vol_adv


PNL_COLUMNS = [
    "gross_pnl",
    "spread_cost",
    "impact_cost",
    "borrow_cost",
    "total_cost",
    "net_pnl",
    "turnover",
]
COST_COLUMNS = ["spread_cost", "impact_cost", "borrow_cost", "total_cost"]


def _make_simulator(tmp_path) -> PortfolioSimulator:
    """Build a PortfolioSimulator backed by a minimal temporary config."""
    config = {
        "spread_bps": 3,
        "impact_coeff": 0.10,
        "borrow_cost_easy_bps": 25,
        "gross_exposure_target": 1.0,
        "max_single_position": 0.10,
        "aum_dollars": 1e8,
        "min_adv_dollars": 1e6,
    }
    config_path = tmp_path / "backtest.yaml"
    config_path.write_text(yaml.dump(config))
    return PortfolioSimulator(config_path=str(config_path))


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str], str, str]:
    """Create reproducible OHLCV and feature panels entirely in memory."""
    rng = np.random.default_rng(20260915)
    tickers = [f"T{i:02d}" for i in range(15)]
    dates = pd.date_range("2020-01-02", periods=80, freq="B")

    close = 100.0 * np.exp(
        np.cumsum(rng.normal(0.0002, 0.012, size=(len(dates), len(tickers))), axis=0)
    )
    volume = rng.integers(750_000, 1_750_000, size=close.shape)
    market_index = pd.MultiIndex.from_product(
        [dates, tickers], names=["date", "ticker"]
    )
    ohlcv = market_index.to_frame(index=False)
    ohlcv["close"] = close.ravel()
    ohlcv["volume"] = volume.ravel()

    feat_x = rng.normal(size=close.shape)
    feat_y = 0.35 * feat_x + rng.normal(scale=0.9, size=close.shape)
    feat_x[12, 3] = np.nan
    feat_y[27, 8] = np.nan

    eligible = np.ones(close.shape, dtype=bool)
    eligible[20:24, 0] = False
    eligible[35, 4] = False
    eligible[52:55, 11] = False

    feat_df = pd.DataFrame(
        {
            "feat_x": feat_x.ravel(),
            "feat_y": feat_y.ravel(),
            "s0_eligible": eligible.ravel(),
        },
        index=market_index,
    ).reorder_levels(["ticker", "date"]).sort_index()

    return ohlcv, feat_df, tickers, str(dates[0].date()), str(dates[-1].date())


def test_load_market_panel_matches_build_returns_vol_adv(synthetic_data):
    """The wrapper returns exactly the first three canonical panel elements."""
    ohlcv, _, tickers, start, end = synthetic_data

    actual = load_market_panel(ohlcv, tickers, start, end)
    expected = build_returns_vol_adv(ohlcv, tickers, start, end)[:3]

    for actual_frame, expected_frame in zip(actual, expected):
        pd.testing.assert_frame_equal(actual_frame, expected_frame)


def test_single_feature_book_contract_and_costs(tmp_path, synthetic_data):
    """A single-feature book has the expected output and non-negative costs."""
    ohlcv, feat_df, tickers, start, end = synthetic_data
    panel = load_market_panel(ohlcv, tickers, start, end)
    sim = _make_simulator(tmp_path)

    result = single_feature_book(
        feat_df, "feat_x", 1.0, panel, sim, rebal_freq=5, start=start, end=end
    )

    assert result.columns.tolist() == PNL_COLUMNS
    assert result.index.min() >= pd.Timestamp(start)
    assert result.index.max() <= pd.Timestamp(end)
    assert (result["net_pnl"] <= result["gross_pnl"] + 1e-12).all()
    assert (result[COST_COLUMNS] >= -1e-12).all().all()


def test_partial_history_feature_spans_the_full_panel_grid(tmp_path, synthetic_data):
    """A feature that starts late still yields a full-length book of zeros.

    Regression test for the real failure this caused: corr_XLC exists only
    from 2018 (the XLC sector ETF launched then), so its book was 871 rows
    against a 2267-day grid and TA-FDR v3's joint bootstrap aborted, since
    that null requires every feature to share one date index. The book must
    instead hold nothing while the signal does not exist.
    """
    ohlcv, feat_df, tickers, start, end = synthetic_data
    panel = load_market_panel(ohlcv, tickers, start, end)
    sim = _make_simulator(tmp_path)

    # Blank the feature across every ticker for the first 30 dates, which is
    # what a not-yet-listed sector ETF looks like after the cross-sectional
    # median fill (an all-NaN row has no median to fill from).
    late = feat_df.copy()
    all_dates = sorted(late.index.get_level_values("date").unique())
    blank_dates = set(all_dates[:30])
    mask = late.index.get_level_values("date").isin(blank_dates)
    late.loc[mask, "feat_y"] = np.nan

    full = single_feature_book(
        feat_df, "feat_x", 1.0, panel, sim, rebal_freq=5, start=start, end=end
    )
    partial = single_feature_book(
        late, "feat_y", 1.0, panel, sim, rebal_freq=5, start=start, end=end
    )

    assert len(partial) == len(full)
    assert partial.index.equals(full.index)

    early = partial.loc[partial.index <= pd.Timestamp(all_dates[29])]
    assert np.allclose(early["gross_pnl"].to_numpy(), 0.0)
    assert np.allclose(early["turnover"].to_numpy(), 0.0)
    assert np.allclose(early["total_cost"].to_numpy(), 0.0)
    # ... and it does trade once the feature exists.
    assert partial["turnover"].sum() > 0.0


def test_single_feature_book_sign_symmetry(tmp_path, synthetic_data):
    """Flipping feature sign flips gross P&L but preserves turnover."""
    ohlcv, feat_df, tickers, start, end = synthetic_data
    panel = load_market_panel(ohlcv, tickers, start, end)
    sim = _make_simulator(tmp_path)

    positive = single_feature_book(
        feat_df, "feat_x", 1.0, panel, sim, rebal_freq=5, start=start, end=end
    )
    negative = single_feature_book(
        feat_df, "feat_x", -1.0, panel, sim, rebal_freq=5, start=start, end=end
    )

    assert np.allclose(
        positive["gross_pnl"], -negative["gross_pnl"], rtol=1e-10
    )
    assert np.allclose(
        positive["turnover"], negative["turnover"], rtol=1e-10
    )


def test_single_feature_book_rejects_missing_feature(tmp_path, synthetic_data):
    """A missing feature name raises a clear ValueError."""
    ohlcv, feat_df, tickers, start, end = synthetic_data
    panel = load_market_panel(ohlcv, tickers, start, end)
    sim = _make_simulator(tmp_path)

    with pytest.raises(ValueError, match="not found"):
        single_feature_book(
            feat_df,
            "absent_feature",
            1.0,
            panel,
            sim,
            rebal_freq=5,
            start=start,
            end=end,
        )


def test_composite_book_matches_manual_pipeline(tmp_path, synthetic_data):
    """The composite helper exactly matches the explicitly wired pipeline."""
    ohlcv, feat_df, tickers, start, end = synthetic_data
    panel = load_market_panel(ohlcv, tickers, start, end)
    sim = _make_simulator(tmp_path)
    weights = pd.Series({"feat_x": 0.6, "feat_y": 0.4})
    rebal_freq = 5

    actual = composite_book(
        feat_df,
        weights,
        panel,
        sim,
        rebal_freq=rebal_freq,
        start=start,
        end=end,
    )

    raw_signal = generate_composite_signal(feat_df, weights, start, end)
    masked = mask_signal_screen0(raw_signal, feat_df)
    positions = sim.signal_to_positions(masked, lag=1, rebal_freq=rebal_freq)
    positions = apply_s0_daily_exit(positions, feat_df)
    returns, sigma, adv = panel
    expected = sim.simulate_pnl(
        positions,
        returns,
        vol=sigma,
        adv_dollars=adv,
        aum_dollars=sim.cfg.get("aum_dollars", 1e8),
        min_adv_dollars=sim.cfg.get("min_adv_dollars", 1e6),
    ).loc[pd.Timestamp(start):pd.Timestamp(end)]

    assert actual.columns.tolist() == PNL_COLUMNS
    pd.testing.assert_frame_equal(actual, expected)
