"""tests/test_screen0_position_rule.py — the Screen 0 position rule is one rule.

spec v3 (screen0.position_rule) says: mask ineligible names in the SIGNAL
before z-scoring, gross normalisation and the water-fill cap, then exit
ineligible holdings daily WITHOUT renormalising.

That rule lived in books.py while nine other stages called the deprecated
post-hoc path, which masks after sizing and renormalises the survivors. The two
disagreed by 24% on the in-sample gross P&L of the same book, and nothing
caught it because each stage was internally consistent. These tests pin the
properties that distinguish the rules, so a stage cannot drift back.

    python3 -m pytest tests/test_screen0_position_rule.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.portfolio import (  # noqa: E402
    PortfolioSimulator,
    apply_s0_eligible,
    build_positions_screen0,
)

CFG = ROOT / "configs" / "backtest.yaml"


def _panel(n_dates=60, n_tickers=12, ineligible=("T09", "T10", "T11"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_dates)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    signal = pd.DataFrame(
        rng.normal(size=(n_dates, n_tickers)), index=dates, columns=tickers
    )
    idx = pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"])
    elig = pd.Series(True, index=idx)
    for t in ineligible:
        elig.loc[t] = False
    feat_df = pd.DataFrame({"s0_eligible": elig})
    return signal, feat_df, tickers, ineligible


def _sim():
    return PortfolioSimulator(config_path=str(CFG), spread_bps=3.0, impact_coeff=0.10)


def test_ineligible_names_never_hold_a_position():
    signal, feat_df, _, ineligible = _panel()
    pos = build_positions_screen0(signal, feat_df, _sim(), rebal_freq=5)
    assert pos[list(ineligible)].abs().to_numpy().max() == pytest.approx(0.0)


def test_position_cap_holds_on_every_day():
    """The post-hoc path breaches the cap; the specified rule must not."""
    sim = _sim()
    cap = sim.cfg.get("max_single_position", 0.05)
    signal, feat_df, _, _ = _panel()
    pos = build_positions_screen0(signal, feat_df, sim, rebal_freq=5)
    assert pos.abs().to_numpy().max() <= cap + 1e-9


def test_the_two_rules_actually_differ():
    """Guard against a future refactor quietly making them the same call."""
    sim = _sim()
    signal, feat_df, _, _ = _panel()
    specified = build_positions_screen0(signal, feat_df, sim, rebal_freq=5)
    deprecated = apply_s0_eligible(
        sim.signal_to_positions(signal, lag=1, rebal_freq=5), feat_df
    )
    aligned = deprecated.reindex_like(specified).fillna(0.0)
    assert not np.allclose(specified.to_numpy(), aligned.to_numpy(), atol=1e-12)


def test_deprecated_path_can_breach_the_cap():
    """Records why the rule was changed, so the reason survives the refactor."""
    sim = _sim()
    cap = sim.cfg.get("max_single_position", 0.05)
    # Most of the book ineligible: renormalising the few survivors is what
    # pushes weights through the cap.
    ineligible = tuple(f"T{i:02d}" for i in range(3, 12))
    signal, feat_df, _, _ = _panel(ineligible=ineligible)
    deprecated = apply_s0_eligible(
        sim.signal_to_positions(signal, lag=1, rebal_freq=5), feat_df
    )
    assert deprecated.abs().to_numpy().max() > cap + 1e-9


def test_missing_eligibility_column_is_an_error_not_a_warning():
    signal, _, _, _ = _panel()
    empty = pd.DataFrame({"something_else": [1.0]})
    with pytest.raises(ValueError, match="s0_eligible"):
        build_positions_screen0(signal, empty, _sim(), rebal_freq=5)
