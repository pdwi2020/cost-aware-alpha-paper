"""R3b — Cross-universe robustness: NASDAQ-100 + S&P-100 sub-universe.

Uses the 147/160 target tickers that are already in the existing S&P-500 universe
(features_all.parquet). Applies the LOCKED signal weights from IS (fdr_results +
shap_summary) to this sub-universe OOS 2022-2024. Tests whether Track B's net-positive
result holds outside the full S&P-500 universe.

No refit: feature set and SHAP weights are frozen from the S&P-500 IS analysis.

Outputs:
    data/processed/r3b_cross_universe_metrics.parquet  — per-track OOS metrics
    data/processed/r3b_cross_universe_year.parquet     — year-by-year breakdown

Run:
    python3 -u src/backtest/run_r3b_cross_universe.py
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

from src.backtest.portfolio import PortfolioSimulator, apply_s0_eligible
from src.backtest.generate_signals import build_signal_weights, generate_composite_signal

FDR_PATH    = ROOT / "data" / "processed" / "fdr_results.parquet"
SHAP_PATH   = ROOT / "data" / "processed" / "shap_summary.parquet"
FEAT_PATH   = ROOT / "data" / "processed" / "features_all.parquet"
OHLCV_PATH  = ROOT / "data" / "processed" / "daily_ohlcv.parquet"
CFG_PATH    = ROOT / "configs" / "backtest.yaml"
NDX_PATH    = ROOT / "data" / "processed" / "ndx100_tickers.csv"
SPX100_PATH = ROOT / "data" / "processed" / "spx100_tickers.csv"

OUT_METRICS = ROOT / "data" / "processed" / "r3b_cross_universe_metrics.parquet"
OUT_YEAR    = ROOT / "data" / "processed" / "r3b_cross_universe_year.parquet"

HOLDOUT_START = "2022-01-01"
HOLDOUT_END   = "2024-12-31"
REBAL_FREQ    = 5
VOL_WINDOW    = 21

# Screen 0 (look-ahead-free) — see src/data/screen0.py and portfolio.apply_s0_eligible.
# MIN_PRICE removed: apply_min_price_filter (which used trade-date close) was look-ahead.
SANITIZE_CAP = 0.50


def log(msg): print(msg, flush=True)


def build_returns_vol_adv(ohlcv: pd.DataFrame, tickers: list, start: str, end: str):
    mask = (
        ohlcv["ticker"].isin(tickers)
        & (ohlcv["date"] >= pd.Timestamp(start))
        & (ohlcv["date"] <= pd.Timestamp(end))
    )
    sub = ohlcv[mask][["ticker","date","close","volume"]].copy()
    close_w = sub.pivot(index="date", columns="ticker", values="close")
    vol_w   = sub.pivot(index="date", columns="ticker", values="volume")
    returns_raw = close_w.pct_change()
    n_clipped = int((returns_raw.abs() > SANITIZE_CAP).sum().sum())
    if n_clipped > 0:
        log(f"  [sanitize] Clipped {n_clipped} (ticker,day) cells with |ret|>{SANITIZE_CAP:.0%}")
    returns = returns_raw.clip(lower=-SANITIZE_CAP, upper=SANITIZE_CAP)
    sigma   = returns.rolling(VOL_WINDOW, min_periods=10).std()
    adv     = (close_w * vol_w).rolling(VOL_WINDOW, min_periods=10).mean()
    m = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    return returns[m], sigma[m], adv[m], close_w[m]


def main():
    log("=== R3b: NASDAQ-100 + S&P-100 Cross-Universe Robustness ===\n")
    t0 = time.time()

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    fdr_df  = pd.read_parquet(FDR_PATH)
    shap_df = pd.read_parquet(SHAP_PATH)

    # Build target ticker set (NDX-100 ∪ SPX-100 ∩ existing universe)
    ndx    = pd.read_csv(NDX_PATH)["ticker"].tolist()
    spx100 = pd.read_csv(SPX100_PATH)["ticker"].tolist()
    target = set(ndx + spx100)

    log("Loading features_all.parquet …")
    feat_df = pd.read_parquet(FEAT_PATH)
    existing = set(feat_df.index.get_level_values("ticker").unique())
    overlap  = sorted(target & existing)
    missing  = sorted(target - existing)
    log(f"  Target tickers (NDX-100 ∪ SPX-100): {len(target)}")
    log(f"  In existing universe (used):         {len(overlap)}")
    log(f"  Not in existing (excluded):          {len(missing)}  → {missing}\n")

    # Filter feature panel to cross-universe sub-set
    tickers_level = feat_df.index.get_level_values("ticker")
    feat_excl     = feat_df[tickers_level.isin(overlap)]

    log("Loading OHLCV …")
    ohlcv = pd.read_parquet(OHLCV_PATH)

    sim = PortfolioSimulator(
        config_path=str(CFG_PATH),
        spread_bps=cfg["spread_bps"],
        impact_coeff=cfg["impact_coeff"],
    )

    all_metrics = []
    all_year    = []

    for track in ["track_a", "track_b"]:
        log(f"\n{'='*60}")
        log(f"  Track: {track.upper()}")
        log(f"{'='*60}")

        weights = build_signal_weights(track, fdr_df, shap_df)
        sig = generate_composite_signal(
            feat_excl, weights,
            start_date=HOLDOUT_START,
            end_date=HOLDOUT_END,
        )
        log(f"  Signal: {sig.shape[0]} dates × {sig.shape[1]} tickers")

        tickers   = sig.columns.tolist()
        returns, sigma, adv, close_px = build_returns_vol_adv(ohlcv, tickers, HOLDOUT_START, HOLDOUT_END)

        positions = sim.signal_to_positions(sig, lag=1, rebal_freq=REBAL_FREQ)
        # Screen 0 (look-ahead-free) — see src/data/screen0.py
        n_before = int((positions.abs() > 1e-12).sum().sum())
        positions = apply_s0_eligible(positions, feat_excl)
        n_after = int((positions.abs() > 1e-12).sum().sum())
        log(f"  Screen 0 (lagged price≥$5, ADV≥$1M, PIT member): {n_before}→{n_after} active positions")
        pnl_df    = sim.simulate_pnl(
            positions, returns, vol=sigma,
            adv_dollars=adv,
            aum_dollars=cfg.get("aum_dollars", 1e8),
            min_adv_dollars=cfg.get("min_adv_dollars", 1e6),
        )

        full_m = sim.compute_metrics(pnl_df)
        log(f"\n  Cross-universe OOS (2022-2024):")
        log(f"    Gross SR : {full_m.get('gross_pnl_sharpe', float('nan')):+.3f}")
        log(f"    Net SR   : {full_m.get('net_pnl_sharpe', float('nan')):+.3f}")
        log(f"    Net Ann  : {full_m.get('net_pnl_annual', float('nan'))*100:+.2f}%")
        log(f"    Max DD   : {full_m.get('net_pnl_max_dd', float('nan'))*100:.2f}%")

        all_metrics.append({"track": track, "universe": "NDX100+SPX100", **full_m})

        # Year-by-year
        for yr in sorted(pnl_df.index.year.unique()):
            sub = pnl_df[pnl_df.index.year == yr]
            if len(sub) < 20:
                continue
            ym = sim.compute_metrics(sub)
            all_year.append({"track": track, "period": str(yr), **ym})
        all_year.append({"track": track, "period": "OOS (2022-24)", **full_m})

    # Summary table
    log("\n=== Year-by-year net SR (NDX-100 + SPX-100 universe) ===")
    yr_df = pd.DataFrame(all_year)
    for track in ["track_a", "track_b"]:
        sub = yr_df[yr_df["track"] == track][["period", "net_pnl_sharpe"]]
        log(f"  {track.upper()}:")
        log(sub.to_string(index=False))

    # vs. full-universe benchmark
    log("\n=== vs. full S&P-500 benchmark ===")
    try:
        bm = pd.read_parquet(ROOT / "data" / "processed" / "holdout_metrics.parquet")
        for track in ["track_a", "track_b"]:
            full_sr = float(bm[bm["track"] == track]["net_pnl_sharpe"].iloc[0])
            sub_sr  = float([m for m in all_metrics if m["track"] == track][0]["net_pnl_sharpe"])
            delta   = sub_sr - full_sr
            log(f"  {track.upper()}: SPX-500={full_sr:+.3f}  NDX+SPX100={sub_sr:+.3f}  Δ={delta:+.3f}")
    except Exception as e:
        log(f"  (benchmark comparison failed: {e})")

    pd.DataFrame(all_metrics).to_parquet(OUT_METRICS, index=False)
    pd.DataFrame(all_year).to_parquet(OUT_YEAR, index=False)
    log(f"\nSaved → {OUT_METRICS}")
    log(f"Saved → {OUT_YEAR}")
    log(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
