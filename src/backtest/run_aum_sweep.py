"""Reviewer revision R6: capacity / AUM sensitivity sweep (IS-only, 2013-2021).

Re-runs the *deployed* Track-A composite (weekly rebalancing, REBAL_FREQ=5) through
the existing Almgren-Chriss simulator at four AUM levels ($10M / $50M / $100M /
$500M), holding every other parameter at the base case (spread=3bps, impact=0.10,
water-fill 5% cap, ADV>=$1M). It mirrors the deployed weekly pattern used in
run_ablation.py / run_ml_baseline.py (lag=1, rebal_freq=5), so the configuration
is identical to the paper's headline net-Sharpe figures. Positions are AUM-independent
(AUM enters only the market-impact term in simulate_pnl), so they are built once.

Reads only the IS window (2013-2021); never touches the locked 2025 OOS.

Output: data/processed/aum_sweep.parquet
Run:    python3 -u src/backtest/run_aum_sweep.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.portfolio import PortfolioSimulator, apply_s0_eligible  # noqa: E402
from src.backtest.run_backtest import (  # noqa: E402
    build_returns_vol_adv,
    SIG_A_PATH,
    OHLCV_PATH,
    FEAT_PATH,
    CFG_PATH,
    BACKTEST_START,
    BACKTEST_END,
)

AUM_LEVELS = [1e7, 5e7, 1e8, 5e8]   # $10M, $50M, $100M, $500M
BASE_SPREAD = 3.0
BASE_IMPACT = 0.10
REBAL_FREQ = 5          # weekly — the deployed configuration (matches run_holdout/run_ablation)
MIN_ADV = 1e6
OUT = ROOT / "data" / "processed" / "aum_sweep.parquet"


def main():
    print("=== R6: AUM capacity sweep (Track A, weekly rebal, IS 2013-2021) ===\n", flush=True)
    ohlcv = pd.read_parquet(OHLCV_PATH)
    # Lean load: frozen-specification columns only (683 MB panel, 8 GB machine).
    from src.data.lean_load import load_features_lean

    feat_df = load_features_lean(FEAT_PATH)
    signal_df = pd.read_parquet(SIG_A_PATH)
    tickers = signal_df.columns.tolist()

    returns, vol, adv_dollars, close_px = build_returns_vol_adv(
        ohlcv, tickers, BACKTEST_START, BACKTEST_END
    )

    sim = PortfolioSimulator(
        config_path=str(CFG_PATH), spread_bps=BASE_SPREAD, impact_coeff=BASE_IMPACT,
    )
    # Positions depend only on the signal + rebal schedule, not on AUM -> build once.
    positions = sim.signal_to_positions(signal_df, lag=1, rebal_freq=REBAL_FREQ)
    positions = apply_s0_eligible(positions, feat_df)

    rows = []
    for aum in AUM_LEVELS:
        pnl = sim.simulate_pnl(
            positions, returns, vol=vol, adv_dollars=adv_dollars,
            aum_dollars=aum, min_adv_dollars=MIN_ADV,
        )
        m = sim.compute_metrics(pnl)
        rows.append({
            "aum_dollars": aum,
            "aum_label": f"${int(aum / 1e6)}M",
            "gross_sharpe": m.get("gross_pnl_sharpe"),
            "net_sharpe": m.get("net_pnl_sharpe"),
            "net_annual": m.get("net_pnl_annual"),
            "cost_drag_bps": m.get("cost_drag_bps"),
            "annual_turnover": m.get("annual_turnover"),
        })
        print(
            f"  AUM ${int(aum / 1e6):>4d}M:  net SR {m.get('net_pnl_sharpe'):+.3f}   "
            f"gross {m.get('gross_pnl_sharpe'):+.3f}   "
            f"net ann {m.get('net_pnl_annual') * 100:+.2f}%   "
            f"drag {m.get('cost_drag_bps'):.0f} bps/yr   "
            f"TO {m.get('annual_turnover'):.1f}x",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
