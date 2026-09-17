"""Regression tests for vectorised portfolio P&L simulation."""

import numpy as np
import pandas as pd
import yaml

from src.backtest.almgren_chriss import compute_spread_cost
from src.backtest.portfolio import PortfolioSimulator


EXPECTED_COLUMNS = [
    "gross_pnl",
    "spread_cost",
    "impact_cost",
    "borrow_cost",
    "total_cost",
    "net_pnl",
    "turnover",
]


def _make_simulator(tmp_path):
    """Build a minimal PortfolioSimulator backed by a temp backtest.yaml."""
    cfg = {
        "spread_bps": 3,
        "impact_coeff": 0.10,
        "borrow_cost_easy_bps": 125,
        "gross_exposure_target": 1.0,
        "max_single_position": 0.05,
        "aum_dollars": 1e8,
    }
    cfg_path = tmp_path / "backtest.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return PortfolioSimulator(config_path=str(cfg_path))


def _simulate_pnl_reference_loop(
    sim,
    positions,
    returns,
    vol=None,
    adv_dollars=None,
    aum_dollars=1e8,
    min_adv_dollars=1e6,
):
    common_dates    = positions.index.intersection(returns.index)
    common_tickers  = positions.columns.intersection(returns.columns)
    pos  = positions.loc[common_dates, common_tickers]
    ret  = returns.loc[common_dates, common_tickers]

    delta_pos = pos.diff().fillna(pos)     # position changes (day 0: full position)

    # Fallback vol and ADV if not provided
    if vol is None:
        vol = pd.DataFrame(0.02, index=pos.index, columns=pos.columns)
    else:
        vol = vol.reindex(index=pos.index, columns=pos.columns).fillna(0.02)

    if adv_dollars is None:
        adv_dollars = pd.DataFrame(1e8, index=pos.index, columns=pos.columns)
    else:
        adv_dollars = adv_dollars.reindex(
            index=pos.index, columns=pos.columns
        ).fillna(1e8)

    records = []
    for d in common_dates:
        p_d     = pos.loc[d]
        r_d     = ret.loc[d]
        dp_d    = delta_pos.loc[d].abs()    # |Δw| per ticker
        v_d     = vol.loc[d]
        adv_d   = adv_dollars.loc[d]

        # Zero out positions in stocks with insufficient liquidity.
        # Prevents bankrupt/delisted stocks (ADV → 0) from generating
        # unrealistic AC impact costs.
        liquid = adv_d >= min_adv_dollars
        p_d  = p_d.where(liquid, 0.0)
        dp_d = dp_d.where(liquid, 0.0)

        gross    = float((p_d * r_d).sum())
        turnover = float(dp_d.sum())

        # Spread: paid on each trade (half spread per side = full spread round-trip)
        sc = float(dp_d.sum()) * compute_spread_cost(sim.spread_bps)

        # Market impact (Almgren-Chriss):
        #   trade_dollars_i = |Δw_i| × AUM
        #   participation_i = trade_dollars_i / ADV_i
        #   impact_frac_i   = η × vol_i × sqrt(participation_i)
        #   cost_i          = impact_frac_i × |Δw_i|   (as fraction of AUM)
        participation = (dp_d * aum_dollars) / adv_d.clip(lower=1.0)
        ic_per_stock  = sim.impact_coeff * v_d * np.sqrt(participation)
        ic = float((ic_per_stock * dp_d).sum())

        # Borrow cost: only for short positions held overnight
        bc = float(
            p_d.clip(upper=0.0).abs().sum()
            * (sim.cfg.get("borrow_cost_easy_bps", 0) / 10_000 / 252)
        )

        total_cost = sc + ic + bc
        net        = gross - total_cost

        records.append({
            "date":        d,
            "gross_pnl":   gross,
            "spread_cost": sc,
            "impact_cost": ic,
            "borrow_cost": bc,
            "total_cost":  total_cost,
            "net_pnl":     net,
            "turnover":    turnover,
        })

    return pd.DataFrame(records).set_index("date")


def _make_panels():
    rng = np.random.default_rng(2026)
    dates = pd.date_range("2020-01-02", periods=60, freq="B")
    tickers = [f"T{i:03d}" for i in range(40)]

    rebalance_rows = rng.normal(0.0, 0.025, size=(12, len(tickers)))
    positions = pd.DataFrame(
        np.repeat(rebalance_rows, 5, axis=0),
        index=dates,
        columns=tickers,
    )
    positions.iloc[10:15, 3] = np.nan
    positions.iloc[37, 9] = np.nan

    returns = pd.DataFrame(
        rng.normal(0.0002, 0.018, size=positions.shape),
        index=dates,
        columns=tickers,
    )
    returns.iloc[4, 2] = np.nan
    returns.iloc[22:25, 17] = np.nan

    vol = pd.DataFrame(
        rng.uniform(0.008, 0.045, size=positions.shape),
        index=dates,
        columns=tickers,
    )
    vol.iloc[8, 6] = np.nan
    vol.iloc[41:44, 21] = np.nan

    adv = pd.DataFrame(
        rng.uniform(2e6, 3e8, size=positions.shape),
        index=dates,
        columns=tickers,
    )
    adv.iloc[:, 0] = 4e5
    adv.iloc[:30, 1] = 8e5
    adv.iloc[30:, 2] = 5e5
    adv.iloc[7, 5] = np.nan
    adv.iloc[48:51, 12] = np.nan
    return positions, returns, vol, adv


def test_vectorized_simulate_pnl_matches_reference_loop(tmp_path):
    sim = _make_simulator(tmp_path)
    positions, returns, vol, adv = _make_panels()
    kwargs = {
        "vol": vol,
        "adv_dollars": adv,
        "aum_dollars": 7.5e7,
        "min_adv_dollars": 1e6,
    }

    actual = sim.simulate_pnl(positions, returns, **kwargs)
    expected = _simulate_pnl_reference_loop(sim, positions, returns, **kwargs)

    pd.testing.assert_index_equal(actual.index, expected.index)
    assert list(actual.columns) == list(expected.columns) == EXPECTED_COLUMNS
    for column in EXPECTED_COLUMNS:
        assert np.allclose(
            actual[column].values,
            expected[column].values,
            rtol=1e-12,
            atol=1e-15,
        ), column


def test_per_name_spread_and_borrow_matrices(tmp_path):
    sim = _make_simulator(tmp_path)
    positions, returns, vol, adv = _make_panels()
    rng = np.random.default_rng(77)
    spread_bps = pd.DataFrame(
        rng.uniform(0.5, 12.0, size=positions.shape),
        index=positions.index,
        columns=positions.columns,
    )
    borrow_bps = pd.DataFrame(
        rng.uniform(0.0, 500.0, size=positions.shape),
        index=positions.index,
        columns=positions.columns,
    )
    spread_bps.iloc[5, 4] = np.nan
    borrow_bps.iloc[9, 8] = np.nan
    kwargs = {
        "vol": vol,
        "adv_dollars": adv,
        "aum_dollars": 7.5e7,
        "min_adv_dollars": 1e6,
    }

    matrix_result = sim.simulate_pnl(
        positions,
        returns,
        spread_bps_matrix=spread_bps,
        borrow_bps_matrix=borrow_bps,
        **kwargs,
    )
    assert matrix_result.shape == (len(positions), len(EXPECTED_COLUMNS))
    assert list(matrix_result.columns) == EXPECTED_COLUMNS

    scalar_result = sim.simulate_pnl(positions, returns, **kwargs)
    constant_spread = pd.DataFrame(
        sim.spread_bps, index=positions.index, columns=positions.columns
    )
    constant_borrow = pd.DataFrame(
        sim.cfg.get("borrow_cost_easy_bps", 0),
        index=positions.index,
        columns=positions.columns,
    )
    constant_result = sim.simulate_pnl(
        positions,
        returns,
        spread_bps_matrix=constant_spread,
        borrow_bps_matrix=constant_borrow,
        **kwargs,
    )
    assert np.allclose(
        constant_result["spread_cost"],
        scalar_result["spread_cost"],
        rtol=1e-12,
        atol=1e-15,
    )
    assert np.allclose(
        constant_result["borrow_cost"],
        scalar_result["borrow_cost"],
        rtol=1e-12,
        atol=1e-15,
    )
