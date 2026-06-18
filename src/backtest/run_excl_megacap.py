"""R3a — Cross-universe robustness: ex-mega-cap S&P 500.

Drops the ~25 largest names by average market cap (price × volume proxy)
in the IS period (2013-2021) from the existing universe, then reruns the
OOS 2022-2024 backtest with the LOCKED signal weights from fdr_results.parquet.

Tests: "Is Track B's OOS alpha just the Magnificent-7 / mega-cap 2022 regime
reversal, or does it survive in the smaller-cap sub-universe?"

No refit: the BH-significant feature set and SHAP weights are fixed from the
full-universe IS analysis (fdr_results.parquet, shap_summary.parquet).

Outputs:
    data/processed/excl_megacap_metrics.parquet  — per-track OOS metrics
    data/processed/excl_megacap_dropped.parquet  — list of excluded tickers

Run:
    python3 -u src/backtest/run_excl_megacap.py
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
from src.backtest.generate_signals import build_signal_weights, generate_composite_signal

FDR_PATH    = ROOT / "data" / "processed" / "fdr_results.parquet"
SHAP_PATH   = ROOT / "data" / "processed" / "shap_summary.parquet"
FEAT_PATH   = ROOT / "data" / "processed" / "features_all.parquet"
OHLCV_PATH  = ROOT / "data" / "processed" / "daily_ohlcv.parquet"
CFG_PATH    = ROOT / "configs" / "backtest.yaml"

OUT_METRICS = ROOT / "data" / "processed" / "excl_megacap_metrics.parquet"
OUT_DROPPED = ROOT / "data" / "processed" / "excl_megacap_dropped.parquet"

HOLDOUT_START = "2022-01-01"
HOLDOUT_END   = "2024-12-31"
IS_START      = "2013-01-01"
IS_END        = "2021-12-31"
REBAL_FREQ    = 5
VOL_WINDOW    = 21
N_MEGA        = 25     # number of largest tickers to exclude

# Screen 0 — tradeable-universe filter (matches run_holdout.py verbatim)
MIN_PRICE    = 5.0
SANITIZE_CAP = 0.50


def log(msg): print(msg, flush=True)


def apply_min_price_filter(positions, close_w, min_price=MIN_PRICE):
    """Screen 0: zero positions in names priced < min_price on the trade date,
    then renormalise gross (L1) leverage to 1 per day."""
    pxa = close_w.reindex(index=positions.index, columns=positions.columns).ffill()
    positions = positions.where(pxa >= min_price, 0.0)
    l1 = positions.abs().sum(axis=1).replace(0, np.nan)
    return positions.div(l1, axis=0).fillna(0.0)


def identify_megacap_tickers(ohlcv: pd.DataFrame, n: int = N_MEGA) -> list:
    """Identify the n tickers with highest IS average market-cap proxy (price × volume).

    Uses daily dollar volume (close × volume) averaged over the IS period as a
    size proxy. This is a well-established free proxy for market cap when
    shares-outstanding data is unavailable.
    """
    mask = (
        (ohlcv["date"] >= pd.Timestamp(IS_START)) &
        (ohlcv["date"] <= pd.Timestamp(IS_END))
    )
    sub = ohlcv[mask].copy()
    sub["dollar_vol"] = sub["close"] * sub["volume"]
    avg_dv = sub.groupby("ticker")["dollar_vol"].mean().sort_values(ascending=False)
    top_n = avg_dv.head(n).index.tolist()
    return top_n, avg_dv


def build_returns_vol_adv(ohlcv: pd.DataFrame, tickers: list, start: str, end: str):
    mask = (
        ohlcv["ticker"].isin(tickers)
        & (ohlcv["date"] >= pd.Timestamp(start))
        & (ohlcv["date"] <= pd.Timestamp(end))
    )
    sub = ohlcv[mask][["ticker", "date", "close", "volume"]].copy()
    close_w  = sub.pivot(index="date", columns="ticker", values="close")
    volume_w = sub.pivot(index="date", columns="ticker", values="volume")
    returns_raw = close_w.pct_change()
    n_clipped = int((returns_raw.abs() > SANITIZE_CAP).sum().sum())
    if n_clipped > 0:
        log(f"  [sanitize] Clipped {n_clipped} (ticker,day) cells with |ret|>{SANITIZE_CAP:.0%}")
    returns  = returns_raw.clip(lower=-SANITIZE_CAP, upper=SANITIZE_CAP)
    vol      = returns.rolling(VOL_WINDOW, min_periods=10).std()
    adv      = (close_w * volume_w).rolling(VOL_WINDOW, min_periods=10).mean()
    ret_mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    return returns[ret_mask], vol[ret_mask], adv[ret_mask], close_w[ret_mask]


def year_metrics(pnl_df: pd.DataFrame, sim: PortfolioSimulator) -> list:
    records = []
    for yr in sorted(pnl_df.index.year.unique()):
        sub = pnl_df[pnl_df.index.year == yr]
        if len(sub) < 20:
            continue
        m = sim.compute_metrics(sub)
        records.append({"period": str(yr), **m})
    m_full = sim.compute_metrics(pnl_df)
    records.append({"period": "OOS (2022-24)", **m_full})
    return records


def main():
    log(f"=== R3a: ex-Mega-Cap Cross-Universe Robustness (drop top-{N_MEGA}) ===\n")
    t0 = time.time()

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    fdr_df  = pd.read_parquet(FDR_PATH)
    shap_df = pd.read_parquet(SHAP_PATH)

    log("Loading OHLCV …")
    ohlcv = pd.read_parquet(OHLCV_PATH)

    mega_tickers, avg_dv = identify_megacap_tickers(ohlcv, N_MEGA)
    log(f"\n  Excluded {N_MEGA} mega-cap tickers (by avg IS dollar volume):")
    for t in mega_tickers:
        log(f"    {t:8s}  avg_dv=${avg_dv[t]/1e6:.1f}M/day")

    dropped_df = pd.DataFrame({
        "ticker":       mega_tickers,
        "avg_dollar_vol_M": [avg_dv[t]/1e6 for t in mega_tickers],
        "rank":         range(1, N_MEGA + 1),
    })
    dropped_df.to_parquet(OUT_DROPPED, index=False)
    log(f"\n  Saved dropped list → {OUT_DROPPED}")

    log("\nLoading features_all.parquet (OOS slice) …")
    feat_df = pd.read_parquet(FEAT_PATH)

    # Filter feature panel: remove mega-cap tickers
    tickers_level = feat_df.index.get_level_values("ticker")
    feat_df_excl  = feat_df[~tickers_level.isin(mega_tickers)]
    log(f"  Universe after exclusion: {feat_df_excl.index.get_level_values('ticker').nunique()} tickers "
        f"(removed {len(mega_tickers)})")

    all_metrics = []
    all_year    = []

    sim = PortfolioSimulator(
        config_path=str(CFG_PATH),
        spread_bps=cfg["spread_bps"],
        impact_coeff=cfg["impact_coeff"],
    )

    for track in ["track_a", "track_b"]:
        log(f"\n{'='*60}")
        log(f"  Track: {track.upper()}")
        log(f"{'='*60}")

        weights = build_signal_weights(track, fdr_df, shap_df)
        sig = generate_composite_signal(
            feat_df_excl, weights,
            start_date=HOLDOUT_START,
            end_date=HOLDOUT_END,
        )
        log(f"  Signal: {sig.shape[0]} dates × {sig.shape[1]} tickers")

        # Exclude mega-cap from OHLCV as well
        ohlcv_excl = ohlcv[~ohlcv["ticker"].isin(mega_tickers)]
        tickers    = sig.columns.tolist()
        returns, vol, adv, close_px = build_returns_vol_adv(ohlcv_excl, tickers, HOLDOUT_START, HOLDOUT_END)

        positions = sim.signal_to_positions(sig, lag=1, rebal_freq=REBAL_FREQ)
        # Screen 0: exclude penny stocks (price < $5) on trade date
        n_before = int((positions.abs() > 1e-12).sum().sum())
        positions = apply_min_price_filter(positions, close_px)
        n_after = int((positions.abs() > 1e-12).sum().sum())
        log(f"  Screen 0 (price≥${MIN_PRICE:.0f}): {n_before}→{n_after} active positions")
        pnl_df    = sim.simulate_pnl(
            positions, returns, vol=vol,
            adv_dollars=adv,
            aum_dollars=cfg.get("aum_dollars", 1e8),
            min_adv_dollars=cfg.get("min_adv_dollars", 1e6),
        )

        full_m = sim.compute_metrics(pnl_df)
        log(f"\n  Ex-mega-cap OOS (2022-2024):")
        log(f"    Gross SR : {full_m.get('gross_pnl_sharpe', float('nan')):+.3f}")
        log(f"    Net SR   : {full_m.get('net_pnl_sharpe', float('nan')):+.3f}")
        log(f"    Net Ann  : {full_m.get('net_pnl_annual', float('nan'))*100:+.2f}%")
        log(f"    Max DD   : {full_m.get('net_pnl_max_dd', float('nan'))*100:.2f}%")

        all_metrics.append({"track": track, **full_m})
        for yr_m in year_metrics(pnl_df, sim):
            all_year.append({"track": track, **yr_m})

    # Year-by-year
    log("\n=== Year-by-year net SR (ex-mega-cap) ===")
    yr_df = pd.DataFrame(all_year)
    for track in ["track_a", "track_b"]:
        sub = yr_df[yr_df["track"] == track][["period", "net_pnl_sharpe"]]
        log(f"  {track.upper()}:")
        log(sub.to_string(index=False))

    out_df = pd.DataFrame(all_metrics)
    out_df.to_parquet(OUT_METRICS, index=False)
    log(f"\nSaved → {OUT_METRICS}")

    # Compare to full-universe benchmark
    log("\n=== vs. full-universe benchmark ===")
    try:
        bm = pd.read_parquet(ROOT / "data" / "processed" / "holdout_metrics.parquet")
        for track in ["track_a", "track_b"]:
            full_sr = float(bm[bm["track"] == track]["net_pnl_sharpe"].iloc[0])
            excl_sr = float(out_df[out_df["track"] == track]["net_pnl_sharpe"].iloc[0])
            delta   = excl_sr - full_sr
            log(f"  {track.upper()}: full={full_sr:+.3f}  ex-mega-cap={excl_sr:+.3f}  Δ={delta:+.3f}")
    except Exception as e:
        log(f"  (benchmark comparison failed: {e})")

    log(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
