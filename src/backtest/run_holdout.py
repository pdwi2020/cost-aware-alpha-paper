"""Exp 7 (Week 10): Holdout validation on 2022–2024 (FIRST TOUCH of holdout data).

This is the only authorized look at the holdout period. The strategy
parameters (signal weights, cost model, rebalancing frequency) are
fixed from IS analysis (Weeks 6–9) — zero degrees of freedom here.

Fixed parameters (from IS):
  - Signal: SHAP-weighted BH-significant features (same weights as IS)
  - Rebalancing: weekly (5d) — Pareto-optimal from Week 9
  - Costs: base case (spread=3bps, impact=0.10), AUM=$100M

Outputs:
    data/processed/holdout_pnl.parquet      — daily P&L 2022-2024
    data/processed/holdout_metrics.parquet  — performance metrics

Run:
    python3 -u src/backtest/run_holdout.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.portfolio import PortfolioSimulator
from src.backtest.generate_signals import (
    build_signal_weights,
    generate_composite_signal,
)

FDR_PATH    = ROOT / "data" / "processed" / "fdr_results.parquet"
SHAP_PATH   = ROOT / "data" / "processed" / "shap_summary.parquet"
FEAT_PATH   = ROOT / "data" / "processed" / "features_all.parquet"
OHLCV_PATH  = ROOT / "data" / "processed" / "daily_ohlcv.parquet"
CFG_PATH    = ROOT / "configs" / "backtest.yaml"

OUT_PNL     = ROOT / "data" / "processed" / "holdout_pnl.parquet"
OUT_METRICS = ROOT / "data" / "processed" / "holdout_metrics.parquet"

HOLDOUT_START = "2022-01-01"
HOLDOUT_END   = "2024-12-31"

# Fixed from IS: weekly rebalancing (Pareto-optimal from Week 9)
REBAL_FREQ = 5
VOL_WINDOW = 21


def log(msg): print(msg, flush=True)


def build_returns_vol_adv_holdout(ohlcv, tickers, start, end):
    mask = (
        ohlcv["ticker"].isin(tickers)
        & (ohlcv["date"] >= pd.Timestamp(start))
        & (ohlcv["date"] <= pd.Timestamp(end))
    )
    sub  = ohlcv[mask][["ticker", "date", "close", "volume"]].copy()
    close_w  = sub.pivot(index="date", columns="ticker", values="close")
    volume_w = sub.pivot(index="date", columns="ticker", values="volume")
    returns      = close_w.pct_change()
    vol          = returns.rolling(VOL_WINDOW, min_periods=10).std()
    dollar_vol   = close_w * volume_w
    adv_dollars  = dollar_vol.rolling(VOL_WINDOW, min_periods=10).mean()
    ret_mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    return returns[ret_mask], vol[ret_mask], adv_dollars[ret_mask]


def main():
    log("=== Exp 7: Holdout Validation 2022–2024 (FIRST TOUCH) ===\n")
    log(f"  Holdout window: {HOLDOUT_START} → {HOLDOUT_END}")
    log(f"  Rebalancing: {REBAL_FREQ}-day (weekly, Pareto-optimal from Week 9)")
    log(f"  Costs: base case (spread=3bps, impact=0.10, AUM=$100M)\n")
    t0 = time.time()

    fdr_df  = pd.read_parquet(FDR_PATH)
    shap_df = pd.read_parquet(SHAP_PATH)

    log("Loading features_all.parquet (holdout slice) …")
    feat_df = pd.read_parquet(FEAT_PATH)

    log("Loading OHLCV …")
    ohlcv = pd.read_parquet(OHLCV_PATH)

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    aum_dollars = cfg.get("aum_dollars", 1e8)

    all_metrics = []

    for track in ["track_a", "track_b"]:
        log(f"\n{'='*55}")
        log(f"  Track: {track.upper()}")
        log(f"{'='*55}")

        # Build signal weights from IS analysis (same as Week 8)
        weights = build_signal_weights(track, fdr_df, shap_df)

        # Generate holdout signals (2022-2024)
        sig = generate_composite_signal(
            feat_df, weights,
            start_date=HOLDOUT_START,
            end_date=HOLDOUT_END,
        )
        log(f"  Signal: {sig.shape[0]} dates × {sig.shape[1]} tickers")
        log(f"  Date range: {sig.index.min().date()} → {sig.index.max().date()}")

        tickers = sig.columns.tolist()
        returns, vol, adv_dollars = build_returns_vol_adv_holdout(
            ohlcv, tickers, HOLDOUT_START, HOLDOUT_END
        )

        sim = PortfolioSimulator(
            config_path=str(CFG_PATH),
            spread_bps=cfg["spread_bps"],
            impact_coeff=cfg["impact_coeff"],
        )
        positions = sim.signal_to_positions(sig, lag=1, rebal_freq=REBAL_FREQ)
        min_adv   = cfg.get("min_adv_dollars", 1e6)
        pnl_df    = sim.simulate_pnl(
            positions, returns, vol=vol,
            adv_dollars=adv_dollars, aum_dollars=aum_dollars,
            min_adv_dollars=min_adv,
        )
        pnl_df.insert(0, "track", track)

        metrics = sim.compute_metrics(pnl_df.drop(columns="track"))

        log(f"\n  Holdout Results (2022–2024):")
        log(f"    Gross Sharpe : {metrics.get('gross_pnl_sharpe', float('nan')):+.3f}")
        log(f"    Net   Sharpe : {metrics.get('net_pnl_sharpe', float('nan')):+.3f}")
        log(f"    Gross Annual : {metrics.get('gross_pnl_annual', float('nan'))*100:+.2f}%")
        log(f"    Net   Annual : {metrics.get('net_pnl_annual', float('nan'))*100:+.2f}%")
        log(f"    Max Drawdown : {metrics.get('net_pnl_max_dd', float('nan'))*100:.2f}%")
        log(f"    Hit Rate     : {metrics.get('net_pnl_hit_rate', float('nan'))*100:.1f}%")
        log(f"    Annual TO    : {metrics.get('annual_turnover', float('nan')):.2f}x")
        log(f"    Cost Drag    : {metrics.get('cost_drag_bps', float('nan')):.1f} bps/yr")

        all_metrics.append({"track": track, **metrics})

        if not hasattr(pnl_df, "_combined"):
            pnl_df.to_parquet(str(OUT_PNL).replace(".parquet", f"_{track}.parquet"), index=True)

    # ── IS vs OOS comparison ──────────────────────────────────────────────
    log(f"\n{'='*55}")
    log("  IS (2013-2021, weekly) vs OOS (2022-2024, weekly) Comparison")
    log(f"{'='*55}")

    # Load IS results from sensitivity_3d (weekly, base costs)
    try:
        sens = pd.read_parquet(ROOT / "data" / "processed" / "sensitivity_3d.parquet")
        is_base = sens[(sens.spread_bps == 3) & (sens.impact_coeff == 0.10) & (sens.rebal_freq == 5)]
        log(f"\n  IS (2013-2021):")
        for _, row in is_base.iterrows():
            log(f"    {row['track'].upper()}: "
                f"gross_SR={row['gross_pnl_sharpe']:+.3f}  "
                f"net_SR={row['net_pnl_sharpe']:+.3f}  "
                f"TO={row['annual_turnover']:.1f}x")
    except Exception:
        pass

    log(f"\n  OOS (2022-2024):")
    for m in all_metrics:
        log(f"    {m['track'].upper()}: "
            f"gross_SR={m.get('gross_pnl_sharpe', float('nan')):+.3f}  "
            f"net_SR={m.get('net_pnl_sharpe', float('nan')):+.3f}  "
            f"TO={m.get('annual_turnover', float('nan')):.1f}x")

    pd.DataFrame(all_metrics).to_parquet(OUT_METRICS, index=False)
    log(f"\nSaved → {OUT_METRICS}")
    log(f"Saved holdout P&L → {OUT_PNL.parent}/holdout_pnl_track_*.parquet")
    log(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
