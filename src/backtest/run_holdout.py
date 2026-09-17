"""Locked single-touch OOS validation on the 2025 window (January to July).

This is the only authorized look at the holdout period. The strategy
parameters (signal weights, cost model, rebalancing frequency) are
fixed from IS analysis (Weeks 6–9) — zero degrees of freedom here.

Fixed parameters (from IS):
  - Signal: SHAP-weighted BH-significant features (same weights as IS)
  - Rebalancing: weekly (5d) — Pareto-optimal from Week 9
  - Costs: base case (spread=3bps, impact=0.10), AUM=$100M

Outputs:
    data/processed/holdout_pnl.parquet      — daily P&L over the locked window
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

import src.manifest as manifest
from src.backtest.portfolio import PortfolioSimulator, build_positions_screen0
from src.backtest.oos_inference import window_inference
from src.backtest.generate_signals import (
    build_signal_weights,
    generate_composite_signal,
)
from src.universe_paths import proc

FDR_PATH    = proc(ROOT, "fdr_results.parquet")
SHAP_PATH   = proc(ROOT, "shap_summary.parquet")
FEAT_PATH   = proc(ROOT, "features_all.parquet")
OHLCV_PATH  = proc(ROOT, "daily_ohlcv_v3.parquet")
CFG_PATH    = ROOT / "configs" / "backtest.yaml"

OUT_PNL     = proc(ROOT, "holdout_pnl.parquet")
INFERENCE_SEED = 42   # fixed so the window CI is reproducible
OUT_METRICS = proc(ROOT, "holdout_metrics.parquet")

HOLDOUT_START = "2025-01-01"   # LOCKED single-touch OOS (2022-2024 demoted to exploratory)
HOLDOUT_END   = "2025-07-31"   # end of the locked window as originally declared;
                               # the v3 panel now extends further, but this
                               # window is fixed by the single-touch rule and is
                               # NOT extended to use the newer data

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


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=HOLDOUT_START)
    ap.add_argument("--end", default=HOLDOUT_END)
    ap.add_argument("--label", default="locked_oos_2025",
                    help="manifest namespace and output suffix for this window")
    args = ap.parse_args(argv)
    start_w, end_w = args.start, args.end

    log(f"=== Window evaluation: {args.label} ===\n")
    log(f"  Window: {start_w} → {end_w}")
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
    pnl_by_track: dict[str, pd.DataFrame] = {}

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

        # Generate holdout signals over the locked window
        sig = generate_composite_signal(
            feat_df, weights,
            start_date=start_w,
            end_date=end_w,
        )
        log(f"  Signal: {sig.shape[0]} dates × {sig.shape[1]} tickers")
        log(f"  Date range: {sig.index.min().date()} → {sig.index.max().date()}")

        tickers = sig.columns.tolist()
        returns, vol, adv_dollars, close_px = build_returns_vol_adv_holdout(
            ohlcv, tickers, start_w, end_w
        )

        sim = PortfolioSimulator(
            config_path=str(CFG_PATH),
            spread_bps=cfg["spread_bps"],
            impact_coeff=cfg["impact_coeff"],
        )
        # Screen 0 (look-ahead-free): s0_eligible at date t was computed from
        # price[t-1] and trailing ADV[t-window:t-1] in build_features.py via
        # src/data/screen0.py, so no trade-date information is used.
        #
        # This block used to hold its own copy of the post-hoc rule: mask after
        # sizing, then renormalise the survivors. That is not the rule spec v3
        # states, and renormalising breached the position cap. It now calls the
        # one shared implementation, so this window and the forward window are
        # built the same way.
        n_before = int(
            (sim.signal_to_positions(sig, lag=1, rebal_freq=REBAL_FREQ).abs() > 1e-12).sum().sum()
        )
        positions = build_positions_screen0(sig, feat_df, sim, REBAL_FREQ)
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
        pnl_by_track[track] = pnl_df

        metrics = sim.compute_metrics(pnl_df.drop(columns="track"))

        log(f"\n  Holdout Results ({HOLDOUT_START} to {HOLDOUT_END}):")
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
            # The metrics file is suffixed by label (below) but this one was not,
            # so evaluating a second window overwrote the first window's daily
            # P&L in place. The per-year exploratory breakdown then had no
            # recoverable source. Keep the unsuffixed name for the locked window
            # so existing references resolve, and suffix every other window.
            stem = f"_{track}.parquet" if args.label == "locked_oos_2025" \
                else f"_{args.label}_{track}.parquet"
            pnl_df.to_parquet(str(OUT_PNL).replace(".parquet", stem), index=True)

    # ── IS vs OOS comparison ──────────────────────────────────────────────
    log(f"\n{'='*55}")
    log(f"  IS (2013-2021, weekly) vs {args.label} "
        f"({start_w} to {end_w}, weekly) Comparison")
    log(f"{'='*55}")

    # Load IS results from sensitivity_3d at the DEPLOYED configuration
    # (weekly rebalancing, base costs). These were previously logged and then
    # discarded, so the manifest's IS figures stayed at their pre-v3 values
    # while the window figures moved, and the manuscript compared a v3 window
    # against a v2 in-sample baseline. A failure here is reported rather than
    # swallowed: a missing IS leg makes the comparison below meaningless.
    try:
        sens = pd.read_parquet(proc(ROOT, "sensitivity_3d.parquet"))
        is_base = sens[(sens.spread_bps == 3) & (sens.impact_coeff == 0.10)
                       & (sens.rebal_freq == 5)]
        if is_base.empty:
            raise ValueError(
                "no (spread=3, impact=0.10, rebal=5) row in sensitivity_3d.parquet"
            )
        log(f"\n  IS (2013-2021, weekly = deployed):")
        for _, row in is_base.iterrows():
            track = row["track"]
            log(f"    {track.upper()}: "
                f"gross_SR={row['gross_pnl_sharpe']:+.3f}  "
                f"net_SR={row['net_pnl_sharpe']:+.3f}  "
                f"TO={row['annual_turnover']:.1f}x")
            for field, key in (("gross_pnl_sharpe", "is_gross_sharpe_weekly"),
                               ("net_pnl_sharpe", "is_net_sharpe_weekly"),
                               ("annual_turnover", "is_annual_turnover_weekly"),
                               ("cost_drag_bps", "is_cost_drag_bps_weekly")):
                if field in row.index:
                    manifest.record(
                        f"backtest.{track}.{key}", round(float(row[field]), 4),
                        stage="backtest", track=track.split("_")[1].upper(),
                        meta={"rebal_freq": 5, "spread_bps": 3, "impact_coeff": 0.10,
                              "window": "2013-01-01..2021-12-31",
                              "source": "sensitivity_3d.parquet"},
                    )
    except Exception as exc:
        log(f"  [WARN] IS baseline unavailable, not recorded: {exc}")

    log(f"\n  Window {args.label} ({start_w} to {end_w}):")
    for m in all_metrics:
        log(f"    {m['track'].upper()}: "
            f"gross_SR={m.get('gross_pnl_sharpe', float('nan')):+.3f}  "
            f"net_SR={m.get('net_pnl_sharpe', float('nan')):+.3f}  "
            f"TO={m.get('annual_turnover', float('nan')):.1f}x")

    for m in all_metrics:
        ns = f"window.{args.label}.{m['track']}"
        for field, key in (("gross_pnl_sharpe", "gross_sharpe"),
                           ("net_pnl_sharpe", "net_sharpe"),
                           ("annual_turnover", "annual_turnover"),
                           ("cost_drag_bps", "cost_drag_bps"),
                           ("n_days", "n_days")):
            if m.get(field) is not None:
                manifest.record(f"{ns}.{key}", float(m[field]), stage="window",
                                track=m["track"].split("_")[1].upper(),
                                meta={"start": start_w, "end": end_w,
                                      "label": args.label})

    # Inference on the window's daily net P&L: bootstrap CI, HAC t, and the
    # four-way economic classification. This was previously computed outside the
    # pipeline and pasted into the manifest, so the p-value, CI and class stayed
    # frozen at an earlier run's values while the Sharpe beside them moved.
    for track, pnl_df in sorted(pnl_by_track.items()):
        if "net_pnl" not in pnl_df.columns:
            log(f"  [WARN] {track}: no net_pnl column, inference skipped")
            continue
        series = pnl_df["net_pnl"].to_numpy(dtype=float)
        try:
            inf = window_inference(series, seed=INFERENCE_SEED)
        except Exception as exc:
            log(f"  [WARN] {track}: inference failed, not recorded: {exc}")
            continue
        ns = f"window.{args.label}.{track}"
        meta = {"start": start_w, "end": end_w, "label": args.label,
                "B": inf["T"], "seed": inf["seed"], "sr_star": inf["sr_star"]}
        manifest.record(f"{ns}.net_sharpe_boot_p", float(inf["boot_p"]),
                        stage="window", track=track.split("_")[1].upper(), meta=meta)
        manifest.record(f"{ns}.net_sharpe_nw_t", float(inf["newey_west"]["t"]),
                        stage="window", track=track.split("_")[1].upper(), meta=meta)
        manifest.record(f"{ns}.net_sharpe_ci95_lo", float(inf["ci_95"]["lo"]),
                        stage="window", track=track.split("_")[1].upper(), meta=meta)
        manifest.record(f"{ns}.net_sharpe_ci95_hi", float(inf["ci_95"]["hi"]),
                        stage="window", track=track.split("_")[1].upper(), meta=meta)
        # The classification is decided on the 90% CI (two one-sided tests at 5%),
        # so record that interval too: otherwise the paper reports a class whose
        # basis is absent from the manifest.
        manifest.record(f"{ns}.net_sharpe_ci90_lo", float(inf["ci_90"]["lo"]),
                        stage="window", track=track.split("_")[1].upper(), meta=meta)
        manifest.record(f"{ns}.net_sharpe_ci90_hi", float(inf["ci_90"]["hi"]),
                        stage="window", track=track.split("_")[1].upper(), meta=meta)
        manifest.record(f"{ns}.classification", str(inf["primary_classification"]),
                        stage="window", track=track.split("_")[1].upper(), meta=meta)
        manifest.record(f"{ns}.min_detectable_sharpe",
                        float(inf["min_detectable_sharpe"]),
                        stage="window", track=track.split("_")[1].upper(), meta=meta)
        log(f"  {track.upper()} inference: SR={inf['sharpe_ann']:+.3f} "
            f"95% CI [{inf['ci_95']['lo']:+.2f}, {inf['ci_95']['hi']:+.2f}] "
            f"90% CI [{inf['ci_90']['lo']:+.2f}, {inf['ci_90']['hi']:+.2f}] "
            f"boot p={inf['boot_p']:.4f} NW t={inf['newey_west']['t']:+.3f} "
            f"-> {inf['primary_classification']}")

    out_metrics = OUT_METRICS if args.label == "locked_oos_2025" else \
        OUT_METRICS.with_name(f"holdout_metrics_{args.label}.parquet")
    pd.DataFrame(all_metrics).to_parquet(out_metrics, index=False)
    log(f"\nSaved → {out_metrics}")
    log(f"Saved holdout P&L → {OUT_PNL.parent}/holdout_pnl_track_*.parquet")
    log(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
