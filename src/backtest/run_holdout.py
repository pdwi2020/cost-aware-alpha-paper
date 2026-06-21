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
from src.universe_paths import proc

FDR_PATH    = proc(ROOT, "fdr_results.parquet")
SHAP_PATH   = proc(ROOT, "shap_summary.parquet")
FEAT_PATH   = proc(ROOT, "features_all.parquet")
OHLCV_PATH  = proc(ROOT, "daily_ohlcv.parquet")
CFG_PATH    = ROOT / "configs" / "backtest.yaml"

OUT_PNL     = proc(ROOT, "holdout_pnl.parquet")
OUT_METRICS = proc(ROOT, "holdout_metrics.parquet")

HOLDOUT_START = "2025-01-01"   # LOCKED single-touch OOS (2022-2024 demoted to exploratory)
HOLDOUT_END   = "2025-07-31"   # data max

# Fixed from IS: weekly rebalancing (Pareto-optimal from Week 9)
REBAL_FREQ = 5
VOL_WINDOW = 21

# ±50% return cap for corporate-action artifact removal.
# NOTE: The ±50% winsorization that was previously applied here to build_returns
# is intentionally REMOVED.  build_features.py::sanitize_ohlcv already clips
# returns before writing features_tier1.parquet / features_all.parquet.
# Re-clipping here on the raw OHLCV would re-introduce the operation on a
# separate price series and risk inconsistency.  The holdout P&L returns come
# from the same sanitized price series embedded in the features parquet via
# the target columns, so no second winsorization is needed.
SANITIZE_CAP = 0.50  # kept for ADV computations only (raw close × volume)

# Screen 0 — REMOVED from this file.
# apply_min_price_filter (which used the *trade-date* close to gate a position
# that earns the trade-date return) was a look-ahead: it used price[t] to
# decide eligibility for a position whose return is also earned at t.
#
# The correct approach is upstream: build_features.py writes `s0_eligible`
# (a boolean flag derived from price[t-1] and trailing ADV through t-1) into
# features_all.parquet.  Below we load that flag and zero-out ineligible
# positions BEFORE the P&L simulation.  Eligibility is therefore known at
# t-1, which is when the position is sized — no look-ahead.
#
# Single source of truth: src/data/screen0.py


def log(msg): print(msg, flush=True)


def build_returns_vol_adv_holdout(ohlcv, tickers, start, end):
    """Build holdout returns, vol, ADV from raw OHLCV.

    Returns are NOT re-clipped here — the ±50% winsorization is applied once,
    upstream, in build_features.py::sanitize_ohlcv.  Re-clipping here would
    operate on a separate price series (the cached daily_ohlcv.parquet which
    stores raw close) and risk sign-inconsistency with the sanitized targets.
    ADV uses raw close × volume (scale preserved; not winsorized).
    """
    mask = (
        ohlcv["ticker"].isin(tickers)
        & (ohlcv["date"] >= pd.Timestamp(start))
        & (ohlcv["date"] <= pd.Timestamp(end))
    )
    sub  = ohlcv[mask][["ticker", "date", "close", "volume"]].copy()
    close_w  = sub.pivot(index="date", columns="ticker", values="close")
    volume_w = sub.pivot(index="date", columns="ticker", values="volume")
    returns      = close_w.pct_change(fill_method=None)
    vol          = returns.rolling(VOL_WINDOW, min_periods=10).std()
    dollar_vol   = close_w * volume_w   # raw close × volume for ADV scale
    adv_dollars  = dollar_vol.rolling(VOL_WINDOW, min_periods=10).mean()
    ret_mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    return returns[ret_mask], vol[ret_mask], adv_dollars[ret_mask], close_w[ret_mask]


def main():
    log("=== Locked single-touch OOS validation 2025 (FIRST TOUCH) ===\n")
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

        # Build signal weights from FROZEN IS analysis (BH-selected features × SHAP)
        try:
            weights = build_signal_weights(track, fdr_df, shap_df)
        except ValueError as e:
            log(f"  [SKIP] {e} — no deployable {track} strategy under the corrected FDR.")
            continue

        # Generate holdout signals (2022-2024)
        sig = generate_composite_signal(
            feat_df, weights,
            start_date=HOLDOUT_START,
            end_date=HOLDOUT_END,
        )
        log(f"  Signal: {sig.shape[0]} dates × {sig.shape[1]} tickers")
        log(f"  Date range: {sig.index.min().date()} → {sig.index.max().date()}")

        tickers = sig.columns.tolist()
        returns, vol, adv_dollars, close_px = build_returns_vol_adv_holdout(
            ohlcv, tickers, HOLDOUT_START, HOLDOUT_END
        )

        sim = PortfolioSimulator(
            config_path=str(CFG_PATH),
            spread_bps=cfg["spread_bps"],
            impact_coeff=cfg["impact_coeff"],
        )
        positions = sim.signal_to_positions(sig, lag=1, rebal_freq=REBAL_FREQ)

        # Screen 0 (look-ahead-free): load the pre-computed eligibility flag
        # from features_all.parquet.  s0_eligible at date t was computed from
        # price[t-1] and trailing ADV[t-window:t-1] in build_features.py via
        # src/data/screen0.py — all information known before the open at t.
        # We zero out positions for ineligible names and renormalise L1 gross
        # among the ELIGIBLE names only (so deployed capital is unchanged).
        n_before = int((positions.abs() > 1e-12).sum().sum())
        if "s0_eligible" in feat_df.columns:
            # Build a (date × ticker) boolean mask from the MultiIndex parquet
            elig_col = feat_df["s0_eligible"]
            if not isinstance(elig_col.index, pd.MultiIndex):
                elig_col = elig_col.set_index(["ticker", "date"]) if "ticker" in feat_df.columns else elig_col
            elig_wide = elig_col.unstack(level="ticker")  # date × ticker
            elig_wide = elig_wide.reindex(index=positions.index, columns=positions.columns)
            elig_wide = elig_wide.fillna(False).astype(bool)
            positions = positions.where(elig_wide, 0.0)
            # Renormalise L1 gross among eligible names so capital stays deployed
            l1 = positions.abs().sum(axis=1).replace(0.0, np.nan)
            positions = positions.div(l1, axis=0).fillna(0.0)
        else:
            log("  [warn] s0_eligible column not found in features parquet; "
                "rebuild with build_features.py to enable look-ahead-free Screen 0. "
                "Proceeding without Screen 0 filter.")
        n_after = int((positions.abs() > 1e-12).sum().sum())
        log(f"  Screen 0 (lagged price≥$5, ADV≥$1M, PIT member): "
            f"{n_before}→{n_after} active positions")
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
        sens = pd.read_parquet(proc(ROOT, "sensitivity_3d.parquet"))
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
