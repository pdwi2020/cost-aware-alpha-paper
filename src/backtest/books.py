"""Build reusable strategy books through the full backtest pipeline.

This module is the shared entry point for converting single-feature or
composite alpha signals into cost-adjusted daily P&L.  It centralises market
panel construction and the signal, Screen 0, position, and simulation steps so
TA-FDR, PBO, and other backtest scripts use the same book-building contract.
"""

import numpy as np
import pandas as pd

from src.backtest.generate_signals import generate_composite_signal
from src.backtest.portfolio import (
    PortfolioSimulator,
    build_positions_screen0,
)
from src.backtest.run_backtest import build_returns_vol_adv


def load_market_panel(
    ohlcv: pd.DataFrame,
    tickers: list[str],
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the reusable returns, volatility, and ADV market panel.

    Returns are already sanitised by clipping them to +/-50%.  ``sigma`` is
    derived from those sanitised returns, while ``adv`` is derived from the raw
    close-price and volume panel, as documented by ``build_returns_vol_adv``.
    """
    returns, sigma, adv, _ = build_returns_vol_adv(ohlcv, tickers, start, end)
    return returns, sigma, adv


def _pivot_feature_signal(
    feat_df: pd.DataFrame,
    feature: str,
    sign: float,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Build a median-filled date-by-ticker signal for one feature."""
    if feature not in feat_df.columns:
        raise ValueError(f"Feature column {feature!r} not found in feat_df")

    dates = feat_df.index.get_level_values("date")
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    signal = feat_df.loc[mask, feature].unstack(level="ticker")

    values = signal.to_numpy(dtype=float, copy=True)
    row_medians = np.nanmedian(values, axis=1, keepdims=True)
    missing = np.isnan(values)
    values[missing] = np.broadcast_to(row_medians, values.shape)[missing]

    filled = pd.DataFrame(values, index=signal.index, columns=signal.columns)
    return (sign * filled).dropna(how="all")


def _run_book_pipeline(
    raw_signal: pd.DataFrame,
    feat_df: pd.DataFrame,
    panel: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    sim: "PortfolioSimulator",
    rebal_freq: int,
    start: str,
    end: str,
    spread_bps_matrix: pd.DataFrame | None = None,
    borrow_bps_matrix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply Screen 0, form positions, simulate costs, and slice the result."""
    positions = build_positions_screen0(raw_signal, feat_df, sim, rebal_freq)

    returns, sigma, adv = panel
    # Features with partial history (e.g. corr_XLC: the XLC sector ETF only
    # launched in 2018) would otherwise yield a shorter book than the panel.
    # Hold nothing on those dates instead: the book spans the full grid, which
    # is what the joint bootstrap requires (one shared date index across every
    # feature) and is also the honest treatment, since a signal that does not
    # exist yet cannot be traded.
    positions = positions.reindex(
        index=returns.index, columns=returns.columns
    ).fillna(0.0)
    pnl_df = sim.simulate_pnl(
        positions,
        returns,
        vol=sigma,
        adv_dollars=adv,
        aum_dollars=sim.cfg.get("aum_dollars", 1e8),
        min_adv_dollars=sim.cfg.get("min_adv_dollars", 1e6),
        spread_bps_matrix=spread_bps_matrix,
        borrow_bps_matrix=borrow_bps_matrix,
    )
    return pnl_df.loc[pd.Timestamp(start):pd.Timestamp(end)]


def single_feature_book(
    feat_df: pd.DataFrame,
    feature: str,
    sign: float,
    panel: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    sim: "PortfolioSimulator",
    rebal_freq: int = 5,
    start: str = "2013-01-01",
    end: str = "2021-12-31",
) -> pd.DataFrame:
    """Build one single-feature long-short book's daily P&L."""
    raw_signal = _pivot_feature_signal(feat_df, feature, sign, start, end)
    return _run_book_pipeline(
        raw_signal,
        feat_df,
        panel,
        sim,
        rebal_freq,
        start,
        end,
    )


def composite_book(
    feat_df: pd.DataFrame,
    weights: pd.Series,
    panel: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    sim: "PortfolioSimulator",
    rebal_freq: int = 5,
    start: str = "2013-01-01",
    end: str = "2021-12-31",
    spread_bps_matrix: pd.DataFrame | None = None,
    borrow_bps_matrix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one composite-alpha long-short book's daily P&L.

    ``spread_bps_matrix`` and ``borrow_bps_matrix`` accept per-stock-day costs
    (see ``src.backtest.cost_calibration``); omit them to use the simulator's
    scalar half-spread and borrow fee.
    """
    raw_signal = generate_composite_signal(feat_df, weights, start, end)
    return _run_book_pipeline(
        raw_signal,
        feat_df,
        panel,
        sim,
        rebal_freq,
        start,
        end,
        spread_bps_matrix=spread_bps_matrix,
        borrow_bps_matrix=borrow_bps_matrix,
    )
