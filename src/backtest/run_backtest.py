"""Week 8: Full backtest with Almgren-Chriss cost model + sensitivity grid.

Pipeline:
  1. Load composite signals (track_a, track_b) from generate_signals.py output.
  2. Build daily returns, vol, and ADV from daily_ohlcv.parquet (IS period only).
  3. Run PortfolioSimulator:
       signal_to_positions → simulate_pnl → compute_metrics
  4. Base-case backtest: spread=3bps, impact=0.10.
  5. 2D sensitivity: 5 spread_bps × 4 impact_coeff = 20 scenarios per track.

Outputs:
    data/processed/backtest_base.parquet       — base metrics per track
    data/processed/backtest_sensitivity.parquet — full 20-scenario grid per track

Run:
    python3 -u src/backtest/run_backtest.py
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

SIG_A_PATH  = ROOT / "data" / "processed" / "signals_track_a.parquet"
SIG_B_PATH  = ROOT / "data" / "processed" / "signals_track_b.parquet"
OHLCV_PATH  = ROOT / "data" / "processed" / "daily_ohlcv.parquet"
CFG_PATH    = ROOT / "configs" / "backtest.yaml"

OUT_BASE    = ROOT / "data" / "processed" / "backtest_base.parquet"
OUT_SENS    = ROOT / "data" / "processed" / "backtest_sensitivity.parquet"

# Walk-forward IS window (2013–2021 as per model suite)
BACKTEST_START = "2013-01-01"
BACKTEST_END   = "2021-12-31"

VOL_WINDOW = 21   # days for rolling vol


def log(msg): print(msg, flush=True)


SANITIZE_CAP = 0.50   # ±50% return cap, mirrors build_features.py and run_baselines.py

# Screen 0 — tradeable-universe filter (matches run_holdout.py verbatim)
MIN_PRICE = 5.0


def apply_min_price_filter(positions, close_w, min_price=MIN_PRICE):
    """Screen 0: zero positions in names priced < min_price on the trade date,
    then renormalise gross (L1) leverage to 1 per day."""
    pxa = close_w.reindex(index=positions.index, columns=positions.columns).ffill()
    positions = positions.where(pxa >= min_price, 0.0)
    l1 = positions.abs().sum(axis=1).replace(0, np.nan)
    return positions.div(l1, axis=0).fillna(0.0)


def _clip_returns(ret_df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Clip daily returns to ±cap, log how many cells were affected."""
    n_clipped = int((ret_df.abs() > cap).sum().sum())
    if n_clipped > 0:
        log(f"  [sanitize] Clipped {n_clipped} (ticker,day) cells with |ret|>{cap:.0%}")
    return ret_df.clip(lower=-cap, upper=cap)


def build_returns_vol_adv(
    ohlcv: pd.DataFrame,
    tickers: list,
    start: str,
    end: str,
) -> tuple:
    """Build daily returns, rolling vol, and ADV from OHLCV.

    Returns are sanitized (±50% cap) to remove corporate-action artifacts.
    ADV denominator uses the raw close × volume (dollar-volume scale preserved).

    Returns:
        returns    : (date × ticker) daily close-to-close returns (sanitized)
        vol        : (date × ticker) 21-day rolling return std
        adv_dollars: (date × ticker) 21-day rolling dollar volume
        close_w    : (date × ticker) raw close price matrix (for Screen 0)
    """
    # Filter tickers and date range
    mask = (
        ohlcv["ticker"].isin(tickers)
        & (ohlcv["date"] >= pd.Timestamp(start))
        & (ohlcv["date"] <= pd.Timestamp(end))
    )
    sub = ohlcv[mask][["ticker", "date", "close", "volume"]].copy()

    # Pivot to wide format
    close_w  = sub.pivot(index="date", columns="ticker", values="close")
    volume_w = sub.pivot(index="date", columns="ticker", values="volume")

    # Daily returns — sanitize to ±50% to remove corporate-action artifacts
    returns_raw = close_w.pct_change(fill_method=None)
    returns = _clip_returns(returns_raw, SANITIZE_CAP)

    # Rolling vol (std of returns)
    vol = returns.rolling(VOL_WINDOW, min_periods=10).std()

    # ADV in dollars (21-day rolling mean of close × volume; uses raw close scale)
    dollar_vol   = close_w * volume_w
    adv_dollars  = dollar_vol.rolling(VOL_WINDOW, min_periods=10).mean()

    # Restrict to backtest window
    ret_mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    return returns[ret_mask], vol[ret_mask], adv_dollars[ret_mask], close_w[ret_mask]


def run_single_backtest(
    signal_df: pd.DataFrame,
    returns: pd.DataFrame,
    vol: pd.DataFrame,
    adv_dollars: pd.DataFrame,
    spread_bps: float,
    impact_coeff: float,
    cfg_path: str,
    track: str,
    aum_dollars: float = 1e8,
    min_adv_dollars: float = 1e6,
    close_w: pd.DataFrame | None = None,
) -> dict:
    """Run one scenario: build positions, simulate P&L, compute metrics."""
    sim = PortfolioSimulator(
        config_path=str(cfg_path),
        spread_bps=spread_bps,
        impact_coeff=impact_coeff,
    )
    positions = sim.signal_to_positions(signal_df, lag=1)
    # Screen 0: exclude penny stocks (price < $5) on trade date
    if close_w is not None:
        positions = apply_min_price_filter(positions, close_w)
    pnl_df    = sim.simulate_pnl(positions, returns, vol=vol,
                                  adv_dollars=adv_dollars, aum_dollars=aum_dollars,
                                  min_adv_dollars=min_adv_dollars)
    metrics   = sim.compute_metrics(pnl_df)

    return {
        "track":        track,
        "spread_bps":   spread_bps,
        "impact_coeff": impact_coeff,
        "n_days":       len(pnl_df),
        **metrics,
        "_pnl_df":      pnl_df,   # keep for base-case printout (stripped before saving)
    }


def print_metrics(metrics: dict, label: str) -> None:
    log(f"\n  {label}")
    log(f"    Gross Sharpe : {metrics.get('gross_pnl_sharpe', float('nan')):+.3f}")
    log(f"    Net   Sharpe : {metrics.get('net_pnl_sharpe', float('nan')):+.3f}")
    log(f"    Gross Annual : {metrics.get('gross_pnl_annual', float('nan'))*100:+.2f}%")
    log(f"    Net   Annual : {metrics.get('net_pnl_annual', float('nan'))*100:+.2f}%")
    log(f"    Max Drawdown : {metrics.get('net_pnl_max_dd', float('nan'))*100:.2f}%")
    log(f"    Hit Rate     : {metrics.get('net_pnl_hit_rate', float('nan'))*100:.1f}%")
    log(f"    Daily TO     : {metrics.get('daily_turnover', float('nan')):.4f}")
    log(f"    Annual TO    : {metrics.get('annual_turnover', float('nan')):.2f}x")
    log(f"    Cost Drag    : {metrics.get('cost_drag_bps', float('nan')):.1f} bps/yr")


def main():
    log("=== Week 8: Backtest (Almgren-Chriss Cost Model) ===\n")
    t0 = time.time()

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    sensitivity_spread = cfg["sensitivity_spread_bps"]       # [1, 3, 5, 7, 10]
    sensitivity_impact = cfg["sensitivity_impact_coeff"]     # [0.05, 0.10, 0.15, 0.20]
    base_spread        = cfg["spread_bps"]                   # 3
    base_impact        = cfg["impact_coeff"]                 # 0.10
    aum_dollars        = cfg.get("aum_dollars", 1e8)         # $100M default
    min_adv_dollars    = cfg.get("min_adv_dollars", 1e6)    # $1M ADV filter

    log("Loading OHLCV …")
    ohlcv = pd.read_parquet(OHLCV_PATH)

    base_results = []
    sens_rows    = []

    for track, sig_path in [("track_b", SIG_B_PATH), ("track_a", SIG_A_PATH)]:
        log(f"\n{'='*60}")
        log(f"  Track: {track.upper()}")
        log(f"{'='*60}")

        signal_df = pd.read_parquet(sig_path)
        log(f"  Signal: {signal_df.shape[0]} dates × {signal_df.shape[1]} tickers")

        tickers = signal_df.columns.tolist()
        returns, vol, adv_dollars, close_px = build_returns_vol_adv(
            ohlcv, tickers, BACKTEST_START, BACKTEST_END
        )
        log(f"  Returns: {returns.shape[0]} dates, vol NaN={vol.isna().mean().mean():.3f}")

        # ── Base case ─────────────────────────────────────────────────────────
        log(f"\n  Base case (spread={base_spread}bps, impact={base_impact}) …")
        base = run_single_backtest(
            signal_df, returns, vol, adv_dollars,
            spread_bps=base_spread, impact_coeff=base_impact,
            cfg_path=CFG_PATH, track=track, aum_dollars=aum_dollars,
            close_w=close_px,
        )
        print_metrics(base, f"Base case — {track.upper()}")
        base_results.append({k: v for k, v in base.items() if k != "_pnl_df"})

        # ── Sensitivity grid ──────────────────────────────────────────────────
        log(f"\n  Running {len(sensitivity_spread)} × {len(sensitivity_impact)} sensitivity grid …")
        for sp, ic in product(sensitivity_spread, sensitivity_impact):
            res = run_single_backtest(
                signal_df, returns, vol, adv_dollars,
                spread_bps=sp, impact_coeff=ic,
                cfg_path=CFG_PATH, track=track, aum_dollars=aum_dollars,
                close_w=close_px,
            )
            sens_rows.append({k: v for k, v in res.items() if k != "_pnl_df"})

        log(f"  Grid complete: {len(sensitivity_spread) * len(sensitivity_impact)} scenarios")

    # ── Save ──────────────────────────────────────────────────────────────────
    base_df = pd.DataFrame(base_results)
    sens_df = pd.DataFrame(sens_rows)

    base_df.to_parquet(OUT_BASE, index=False)
    sens_df.to_parquet(OUT_SENS, index=False)

    log(f"\n{'='*60}")
    log("  Base-Case Summary (net Sharpe)")
    log(f"{'='*60}")
    cols = ["track", "spread_bps", "impact_coeff",
            "gross_pnl_sharpe", "net_pnl_sharpe", "net_pnl_annual",
            "cost_drag_bps", "annual_turnover"]
    log(base_df[cols].round(3).to_string(index=False))

    log(f"\n{'='*60}")
    log("  Sensitivity: Net Sharpe grid")
    log(f"{'='*60}")
    for track in ["track_b", "track_a"]:
        sub = sens_df[sens_df.track == track].copy()
        pivot = sub.pivot(index="spread_bps", columns="impact_coeff",
                          values="net_pnl_sharpe").round(3)
        log(f"\n  {track.upper()} — Net Sharpe (rows=spread_bps, cols=impact_coeff):")
        log(pivot.to_string())

    log(f"\nSaved → {OUT_BASE}")
    log(f"Saved → {OUT_SENS}")
    log(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
