"""Reviewer revision R7: turnover decomposition (IS-only, 2013-2021).

Decomposes total portfolio turnover into two economically distinct components,
using the deployed Track-A weight matrix (weekly rebalancing). Because positions
are held at fixed fractions between rebalance dates (portfolio.signal_to_positions:
no drift renormalisation), all turnover occurs on rebalance dates and is driven
by the target weights changing. We split each name's weight change |Δw| by the
sign transition of its position:

  * Signal-driven (extensive margin): names that ENTER, EXIT, or FLIP sign
    -- genuine changes in which names are held long/short (new information).
  * Rebalancing-driven (intensive margin): names that stay on the SAME side and
    are merely RESIZED -- weight jitter near decision boundaries.

This directly answers "is high turnover genuine signal change or noise near
decision boundaries?". Reads only the IS window; never touches the locked 2025 OOS.

Output: data/processed/turnover_decomp.parquet
Run:    python3 -u src/backtest/run_turnover_decomp.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.portfolio import PortfolioSimulator, build_positions_screen0  # noqa: E402
from src.manifest import record  # noqa: E402
from src.backtest.run_backtest import (  # noqa: E402
    build_returns_vol_adv, SIG_A_PATH, OHLCV_PATH, FEAT_PATH, CFG_PATH,
    BACKTEST_START, BACKTEST_END,
)

REBAL_FREQ = 5
TRADING_DAYS = 252
OUT = ROOT / "data" / "processed" / "turnover_decomp.parquet"


def main():
    print("=== R7: turnover decomposition (Track A, weekly, IS 2013-2021) ===\n", flush=True)
    ohlcv = pd.read_parquet(OHLCV_PATH)
    # Lean load: frozen-specification columns only (683 MB panel, 8 GB machine).
    from src.data.lean_load import load_features_lean

    feat_df = load_features_lean(FEAT_PATH)
    signal_df = pd.read_parquet(SIG_A_PATH)
    tickers = signal_df.columns.tolist()
    returns, vol, adv_dollars, _ = build_returns_vol_adv(
        ohlcv, tickers, BACKTEST_START, BACKTEST_END
    )

    sim = PortfolioSimulator(config_path=str(CFG_PATH), spread_bps=3.0, impact_coeff=0.10)
    W = build_positions_screen0(signal_df, feat_df, sim, REBAL_FREQ).fillna(0.0)

    prev = W.shift(1).fillna(0.0)
    curr = W
    dabs = (curr - prev).abs()

    prev_nz = prev != 0
    curr_nz = curr != 0
    same_sign = np.sign(prev) == np.sign(curr)

    entry = (~prev_nz) & curr_nz
    exit_ = prev_nz & (~curr_nz)
    flip = prev_nz & curr_nz & (~same_sign)
    resize = prev_nz & curr_nz & same_sign

    signal_driven = dabs.where(entry | exit_ | flip, 0.0)     # extensive margin
    rebal_driven = dabs.where(resize, 0.0)                    # intensive margin

    # Daily one-sided-equivalent: sum |Δw| across names each day (= 2x one-way).
    daily_total = dabs.sum(axis=1)
    daily_signal = signal_driven.sum(axis=1)
    daily_rebal = rebal_driven.sum(axis=1)

    n_days = (daily_total > 0).sum() if (daily_total > 0).any() else len(daily_total)
    ann = TRADING_DAYS  # annualisation factor (mean daily * 252)

    tot = daily_total.mean() * ann
    sig = daily_signal.mean() * ann
    reb = daily_rebal.mean() * ann
    frac_sig = sig / tot if tot else float("nan")
    frac_reb = reb / tot if tot else float("nan")

    rows = [
        {"component": "total", "annual_turnover_x": tot, "fraction": 1.0},
        {"component": "signal_driven", "annual_turnover_x": sig, "fraction": frac_sig},
        {"component": "rebalancing_driven", "annual_turnover_x": reb, "fraction": frac_reb},
    ]
    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)

    # Record. This stage wrote only a parquet, so its manifest entries sat at a
    # July vintage (total 20.35x) while the stage itself produced 25.86x and the
    # manuscript quoted the latter. The numbers agreed by luck, not by wiring.
    record("backtest.track_a.turnover.total_annual", round(float(tot), 4),
           stage="turnover_decomp", track="A")
    record("backtest.track_a.turnover.signal_driven_annual", round(float(sig), 4),
           stage="turnover_decomp", track="A")
    record("backtest.track_a.turnover.rebalancing_driven_annual", round(float(reb), 4),
           stage="turnover_decomp", track="A")
    record("backtest.track_a.turnover.signal_driven_frac", round(float(frac_sig), 4),
           stage="turnover_decomp", track="A")
    record("backtest.track_a.turnover.rebalancing_driven_frac", round(float(frac_reb), 4),
           stage="turnover_decomp", track="A")

    print(f"  Rebalance days with trades: {int((daily_total>0).sum())}")
    print(f"  Total annualised turnover      : {tot:6.2f}x")
    print(f"  Signal-driven (enter/exit/flip): {sig:6.2f}x  ({frac_sig*100:4.1f}%)")
    print(f"  Rebalancing-driven (resize)    : {reb:6.2f}x  ({frac_reb*100:4.1f}%)")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
