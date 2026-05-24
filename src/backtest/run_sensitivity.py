"""Week 9: 2D cost sensitivity + turnover-constrained optimization.

For Track A (the signal with positive gross Sharpe):
  1. Full 3D grid: 5 spread_bps × 4 impact_coeff × 3 rebal_freq = 60 scenarios.
  2. Pareto frontier: net Sharpe vs annual turnover.
  3. Cost breakeven: what cost level kills profitability?
  4. Optimal rebalancing frequency under base-case costs.

Track B is included as a reference (gross-negative, dominated by Track A).

Outputs:
    data/processed/sensitivity_3d.parquet   — 60-scenario grid (Track A)
    data/processed/sensitivity_summary.parquet — Pareto-optimal configurations

Run:
    python3 -u src/backtest/run_sensitivity.py
"""

import sys
import time
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.portfolio import PortfolioSimulator
from src.backtest.run_backtest import build_returns_vol_adv

SIG_A_PATH = ROOT / "data" / "processed" / "signals_track_a.parquet"
SIG_B_PATH = ROOT / "data" / "processed" / "signals_track_b.parquet"
OHLCV_PATH = ROOT / "data" / "processed" / "daily_ohlcv.parquet"
CFG_PATH   = ROOT / "configs" / "backtest.yaml"

OUT_3D   = ROOT / "data" / "processed" / "sensitivity_3d.parquet"
OUT_SUMM = ROOT / "data" / "processed" / "sensitivity_summary.parquet"

BACKTEST_START = "2013-01-01"
BACKTEST_END   = "2021-12-31"

REBAL_FREQS    = [1, 5, 21]   # daily, weekly, monthly


def log(msg): print(msg, flush=True)


def run_scenario(
    signal_df, returns, vol, adv_dollars,
    spread_bps, impact_coeff, rebal_freq,
    aum_dollars, track, min_adv_dollars=1e6,
) -> dict:
    sim = PortfolioSimulator(
        config_path=str(CFG_PATH),
        spread_bps=spread_bps,
        impact_coeff=impact_coeff,
    )
    positions = sim.signal_to_positions(signal_df, lag=1, rebal_freq=rebal_freq)
    pnl_df    = sim.simulate_pnl(positions, returns, vol=vol,
                                  adv_dollars=adv_dollars, aum_dollars=aum_dollars,
                                  min_adv_dollars=min_adv_dollars)
    metrics   = sim.compute_metrics(pnl_df)
    return {
        "track":        track,
        "spread_bps":   spread_bps,
        "impact_coeff": impact_coeff,
        "rebal_freq":   rebal_freq,
        "n_days":       len(pnl_df),
        **metrics,
    }


def main():
    log("=== Week 9: 2D Cost Sensitivity + Turnover-Constrained Optimization ===\n")
    t0 = time.time()

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    sensitivity_spread = cfg["sensitivity_spread_bps"]    # [1, 3, 5, 7, 10]
    sensitivity_impact = cfg["sensitivity_impact_coeff"]  # [0.05, 0.10, 0.15, 0.20]
    aum_dollars        = cfg.get("aum_dollars", 1e8)
    min_adv_dollars    = cfg.get("min_adv_dollars", 1e6)

    log("Loading OHLCV …")
    ohlcv = pd.read_parquet(OHLCV_PATH)

    all_rows = []

    for track, sig_path in [("track_a", SIG_A_PATH), ("track_b", SIG_B_PATH)]:
        log(f"\n{'='*60}")
        log(f"  Track: {track.upper()}")
        log(f"{'='*60}")

        signal_df = pd.read_parquet(sig_path)
        tickers   = signal_df.columns.tolist()
        returns, vol, adv_dollars_df = build_returns_vol_adv(
            ohlcv, tickers, BACKTEST_START, BACKTEST_END
        )

        n_scenarios = len(sensitivity_spread) * len(sensitivity_impact) * len(REBAL_FREQS)
        log(f"  Running {n_scenarios} scenarios …")

        for rebal_freq in REBAL_FREQS:
            for sp, ic in product(sensitivity_spread, sensitivity_impact):
                row = run_scenario(
                    signal_df, returns, vol, adv_dollars_df,
                    sp, ic, rebal_freq, aum_dollars, track,
                    min_adv_dollars=min_adv_dollars,
                )
                all_rows.append(row)

        log(f"  Done ({n_scenarios} scenarios)")

    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_3D, index=False)

    # ── Summary tables ────────────────────────────────────────────────────────
    track_a = df[df.track == "track_a"].copy()

    log(f"\n{'='*60}")
    log("  TRACK A — Net Sharpe by rebal_freq (base costs: spread=3, impact=0.10)")
    log(f"{'='*60}")
    base = track_a[(track_a.spread_bps == 3) & (track_a.impact_coeff == 0.10)]
    log(base[["rebal_freq", "gross_pnl_sharpe", "net_pnl_sharpe",
              "net_pnl_annual", "cost_drag_bps", "annual_turnover"]].round(3).to_string(index=False))

    log(f"\n{'='*60}")
    log("  TRACK A — Net Sharpe 2D grid (rebal_freq=1, daily)")
    log(f"{'='*60}")
    for rf in REBAL_FREQS:
        sub = track_a[track_a.rebal_freq == rf]
        pivot = sub.pivot(index="spread_bps", columns="impact_coeff",
                          values="net_pnl_sharpe").round(3)
        freq_label = {1: "daily", 5: "weekly", 21: "monthly"}[rf]
        log(f"\n  Rebal = {freq_label} ({rf}d) — Net Sharpe:")
        log(pivot.to_string())

    # ── Pareto frontier: max net Sharpe at each turnover level ─────────────
    log(f"\n{'='*60}")
    log("  TRACK A — Pareto frontier (best net Sharpe per turnover bucket)")
    log(f"{'='*60}")

    pareto_rows = []
    for rf in sorted(track_a["rebal_freq"].unique()):
        sub = track_a[track_a.rebal_freq == rf]
        best = sub.loc[sub["net_pnl_sharpe"].idxmax()]
        pareto_rows.append({
            "rebal_freq":      rf,
            "best_spread_bps": best["spread_bps"],
            "best_impact":     best["impact_coeff"],
            "net_sharpe":      round(best["net_pnl_sharpe"], 3),
            "gross_sharpe":    round(best["gross_pnl_sharpe"], 3),
            "net_annual":      round(best["net_pnl_annual"] * 100, 2),
            "annual_to":       round(best["annual_turnover"], 1),
            "cost_drag_bps":   round(best["cost_drag_bps"], 1),
        })
    pareto_df = pd.DataFrame(pareto_rows)
    log(pareto_df.to_string(index=False))

    # ── Cost breakeven (at base impact 0.10) ─────────────────────────────
    log(f"\n{'='*60}")
    log("  TRACK A — Net Sharpe > 0 achievable? (impact=0.10 fixed)")
    log(f"{'='*60}")
    sub = track_a[track_a.impact_coeff == 0.10].copy()
    profitable = sub[sub.net_pnl_sharpe > 0][["rebal_freq", "spread_bps",
                                               "impact_coeff", "net_pnl_sharpe",
                                               "annual_turnover"]].sort_values("net_pnl_sharpe", ascending=False)
    if profitable.empty:
        log("  No profitable scenario at impact=0.10")
    else:
        log(profitable.round(3).to_string(index=False))

    pareto_df.to_parquet(OUT_SUMM, index=False)
    log(f"\nSaved → {OUT_3D}")
    log(f"Saved → {OUT_SUMM}")
    log(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
