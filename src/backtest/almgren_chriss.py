"""Execution costs using a square-root participation impact model."""

import numpy as np
import pandas as pd


def compute_impact_cost(
    order_size: float,
    adv: float,
    vol: float,
    impact_coeff: float = 0.10,
) -> float:
    """Square-root participation impact cost per unit of order size.

    Cost per share = impact_coeff * vol * sqrt(order_size / ADV)

    Args:
        order_size: Dollar notional traded.
        adv: Average daily dollar volume.
        vol: Daily return volatility (e.g. 0.02 for 2%).
        impact_coeff: Proportionality constant η (default 0.10).

    Returns:
        Impact cost as fraction of price (e.g. 0.001 = 10 bps).
    """
    if adv <= 0:
        return 0.0
    return float(impact_coeff * vol * np.sqrt(np.abs(order_size) / adv))


def compute_spread_cost(spread_bps: float = 3.0) -> float:
    """Cost per unit of one-way portfolio-weight turnover.

    ``spread_bps`` is the half-spread in basis points. A full round trip pays
    this cost twice, once when buying and once when later selling, because the
    simulator charges every day's absolute change in portfolio weight.

    Args:
        spread_bps: Half-spread in basis points (default 3).

    Returns:
        Cost per unit of one-way turnover (spread_bps / 10000).
    """
    return spread_bps / 10_000


def compute_borrow_cost(
    position: float,
    borrow_tier: str = "easy",
    borrow_cost_bps: dict = None,
) -> float:
    """Daily borrow cost for short positions only.

    Args:
        position: Signed portfolio weight (negative = short).
        borrow_tier: 'easy', 'medium', or 'hard'.
        borrow_cost_bps: Override dict {tier: annual_bps}.

    Returns:
        Daily borrow cost as fraction of position notional.
    """
    defaults   = {"easy": 0, "medium": 150, "hard": 300}
    costs      = borrow_cost_bps if borrow_cost_bps else defaults
    annual_bps = costs.get(borrow_tier, 0)
    daily_cost = (annual_bps / 10_000) / 252
    return float(abs(min(position, 0.0)) * daily_cost)


def compute_total_cost(
    order_size: float,
    adv: float,
    vol: float,
    position: float,
    spread_bps: float = 3.0,
    impact_coeff: float = 0.10,
    borrow_tier: str = "easy",
) -> float:
    """Total cost associated with one-way turnover: spread + impact + borrow.

    The spread and impact components apply to a trade on a given day, while
    borrow applies to the short position held that day. Returns cost as a
    fraction of notional (always ≥ 0).
    """
    spread = compute_spread_cost(spread_bps)
    impact = compute_impact_cost(order_size, adv, vol, impact_coeff)
    borrow = compute_borrow_cost(position, borrow_tier)
    return spread + impact + borrow


def impact_cost_series(
    delta_weights: pd.Series,
    adv_dollars: pd.Series,
    vol: pd.Series,
    impact_coeff: float = 0.10,
) -> pd.Series:
    """Vectorised impact cost for a cross-section of stocks on one day.

    Args:
        delta_weights: Change in portfolio weights (|Δw|), indexed by ticker.
        adv_dollars:   Average daily dollar volume per ticker.
        vol:           Daily return vol per ticker.
        impact_coeff:  η parameter.

    Returns:
        Impact cost per ticker (as fraction of trade notional).
    """
    denom = adv_dollars.reindex(delta_weights.index).fillna(1e6)
    v     = vol.reindex(delta_weights.index).fillna(0.02)
    return impact_coeff * v * np.sqrt(delta_weights.abs() / denom.clip(lower=1.0))
