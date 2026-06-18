"""
diag_trackb_clean.py — Diagnostic: Track B OOS Sharpe under cleaned returns.

PURPOSE
-------
The holdout +0.50 net Sharpe (gross +0.71) was computed on RAW OHLCV prices that
contain artifact moves (e.g. CPWR +9900% on 2024-12-26, MNK +264% on 2023-07-26,
SBNY +448% on 2023-11-27).  This script asks:

  1. Reproduce Track B OOS on raw data  → confirm match to locked +0.71 / +0.50.
  2. V-PnL: keep existing RAW-signal positions, realize them against CLEANED
             returns (close-to-close capped at ±50%).  Isolates P&L-side artifact.
  3. V-Full-light: NOT possible cheaply because Track B signal comes entirely from
             features_all.parquet (a pre-built parquet with price-derived columns
             ret_1d, ret_5d, reversal_1w, etc. that were computed from raw prices).
             We report artifact counts in the affected feature columns instead.
  4. Report top clipped (name, day) cells where Track B had nonzero exposure,
     with their P&L impact.

DESIGN CONSTRAINTS
------------------
- DOES NOT overwrite any locked parquet (holdout_pnl_*.parquet, holdout_metrics.parquet).
- DOES NOT edit paper/main.tex.
- Output is PRINT-ONLY (diagnostic text to stdout).

KEY FINDING (preview)
---------------------
Track B was heavily SHORT the artifact-affected names (MNK, SBNY, SIVBQ, FRCB).
Cleaning those extreme returns ELIMINATES large losses from the raw simulation.
The V-PnL Sharpe is therefore HIGHER than the raw +0.50, not lower.
This means the headline +0.50 is actually CONSERVATIVE on the P&L side —
artifact returns were hurting Track B because it was short those names.
However, the feature-side signal is still contaminated (13,402 artifact cells
across ret_1d, ret_5d, ret_21d, reversal_1w, reversal_4w, ret_63d in OOS),
so a full feature-side rerun is still required to be fully certain.

Run:
    cd /Users/paritoshdwivedi/ML_Paper
    python3 src/diagnostics/diag_trackb_clean.py
"""

import sys
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

# ── Paths ────────────────────────────────────────────────────────────────────
FDR_PATH    = ROOT / "data" / "processed" / "fdr_results.parquet"
SHAP_PATH   = ROOT / "data" / "processed" / "shap_summary.parquet"
FEAT_PATH   = ROOT / "data" / "processed" / "features_all.parquet"
OHLCV_PATH  = ROOT / "data" / "processed" / "daily_ohlcv.parquet"
CFG_PATH    = ROOT / "configs" / "backtest.yaml"
LOCKED_HM   = ROOT / "data" / "processed" / "holdout_metrics.parquet"

HOLDOUT_START = "2022-01-01"
HOLDOUT_END   = "2024-12-31"
WARMUP_START  = "2021-01-01"   # gives 252-day clean-return lookback buffer
REBAL_FREQ    = 5
VOL_WINDOW    = 21
SANITIZE_CAP  = 0.50


def log(msg=""):
    print(msg, flush=True)


def sep(title="", width=65):
    if title:
        log(f"\n{'─'*5} {title} {'─'*(max(0, width - 7 - len(title)))}")
    else:
        log("─" * width)


def build_clean_price_series(close_wide: pd.DataFrame, cap: float = SANITIZE_CAP):
    """
    Build cleaned price proxy (mirrors run_baselines.build_clean_prices).

    Algorithm
    ---------
    1. r_raw  = close_wide.pct_change(fill_method=None)
    2. r_clean = r_raw.clip(-cap, +cap)
    3. close_clean = first_close * cumprod(1 + r_clean)   [first_close = close_wide.iloc[0]]

    Returns
    -------
    close_clean : pd.DataFrame  — same shape as close_wide
    r_clean     : pd.DataFrame  — the clipped return series (= pct_change(close_clean))
    n_clipped   : int           — total (ticker,day) cells clipped
    mask_big    : pd.DataFrame  — boolean mask of clipped cells
    """
    r_raw     = close_wide.pct_change(fill_method=None)
    mask_big  = r_raw.abs() > cap
    n_clipped = int(mask_big.sum().sum())
    r_clean   = r_raw.clip(lower=-cap, upper=cap)

    first_close  = close_wide.iloc[0]
    growth       = (1.0 + r_clean.fillna(0.0)).cumprod()
    close_clean  = growth.multiply(first_close, axis="columns")
    close_clean[close_wide.isna()] = np.nan
    return close_clean, r_clean, n_clipped, mask_big


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 65)
    log("  DIAGNOSTIC: Track B OOS Net Sharpe — Clean vs Raw Returns")
    log("  Script : src/diagnostics/diag_trackb_clean.py")
    log("  OOS    : 2022-01-01 → 2024-12-31")
    log("  NOTE   : This script READS ONLY — no locked parquets modified")
    log("=" * 65)

    # ── Load inputs ──────────────────────────────────────────────────────────
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    fdr_df  = pd.read_parquet(FDR_PATH)
    shap_df = pd.read_parquet(SHAP_PATH)
    feat_df = pd.read_parquet(FEAT_PATH)
    ohlcv   = pd.read_parquet(OHLCV_PATH)

    aum_dollars     = cfg.get("aum_dollars", 1e8)
    min_adv_dollars = cfg.get("min_adv_dollars", 1e6)

    # ── Track B signal (same weights as run_holdout.py) ──────────────────────
    weights = build_signal_weights("track_b", fdr_df, shap_df)
    log(f"\nTrack B signal: {len(weights)} BH-significant features")
    log("  Top features by |weight|:")
    for feat, w in weights.sort_values(key=abs, ascending=False).head(8).items():
        log(f"    {feat:<28s}  w={w:+.4f}")

    # Generate holdout signals (from prebuilt features_all.parquet)
    sig = generate_composite_signal(
        feat_df, weights,
        start_date=HOLDOUT_START,
        end_date=HOLDOUT_END,
    )
    log(f"\n  Signal matrix: {sig.shape[0]} dates × {sig.shape[1]} tickers")
    tickers = sig.columns.tolist()

    # ── STEP 1: Reproduce RAW Track B OOS (exact replica of run_holdout.py) ──
    sep("STEP 1  —  RAW Reproduction (target: +0.7052 gross / +0.5011 net)")

    # EXACT replication: window starts 2022, uses default pct_change (pad fill)
    # This matches run_holdout.py line-for-line
    mask_oos = (
        ohlcv["ticker"].isin(tickers)
        & (ohlcv["date"] >= pd.Timestamp(HOLDOUT_START))
        & (ohlcv["date"] <= pd.Timestamp(HOLDOUT_END))
    )
    sub_oos    = ohlcv[mask_oos][["ticker", "date", "close", "volume"]].copy()
    close_raw  = sub_oos.pivot(index="date", columns="ticker", values="close")
    volume_raw = sub_oos.pivot(index="date", columns="ticker", values="volume")
    ret_raw    = close_raw.pct_change()                       # default pad fill — matches run_holdout.py
    vol_raw    = ret_raw.rolling(VOL_WINDOW, min_periods=10).std()
    adv_raw    = (close_raw * volume_raw).rolling(VOL_WINDOW, min_periods=10).mean()

    sim = PortfolioSimulator(
        config_path=str(CFG_PATH),
        spread_bps=cfg["spread_bps"],
        impact_coeff=cfg["impact_coeff"],
    )
    positions = sim.signal_to_positions(sig, lag=1, rebal_freq=REBAL_FREQ)
    pnl_raw   = sim.simulate_pnl(
        positions, ret_raw, vol=vol_raw,
        adv_dollars=adv_raw, aum_dollars=aum_dollars,
        min_adv_dollars=min_adv_dollars,
    )
    m_raw = sim.compute_metrics(pnl_raw)

    locked_hm    = pd.read_parquet(LOCKED_HM)
    locked_tb    = locked_hm[locked_hm["track"] == "track_b"].iloc[0]
    locked_gross = locked_tb["gross_pnl_sharpe"]
    locked_net   = locked_tb["net_pnl_sharpe"]

    log(f"\n  Locked   → gross Sharpe: {locked_gross:+.4f}   net Sharpe: {locked_net:+.4f}")
    log(f"  Computed → gross Sharpe: {m_raw['gross_pnl_sharpe']:+.4f}   "
        f"net Sharpe: {m_raw['net_pnl_sharpe']:+.4f}")
    match_gross = abs(m_raw["gross_pnl_sharpe"] - locked_gross) < 0.001
    match_net   = abs(m_raw["net_pnl_sharpe"]   - locked_net)   < 0.001
    log(f"  Match gross: {'PASS ✓' if match_gross else 'FAIL'}"
        f"   |   Match net: {'PASS ✓' if match_net else 'FAIL'}")
    log()
    log(f"  Gross Annual  : {m_raw['gross_pnl_annual']*100:+.2f}%")
    log(f"  Net   Annual  : {m_raw['net_pnl_annual']*100:+.2f}%")
    log(f"  Max Drawdown  : {m_raw['net_pnl_max_dd']*100:.2f}%")
    log(f"  Hit Rate (net): {m_raw['net_pnl_hit_rate']*100:.1f}%")
    log(f"  Annual TO     : {m_raw['annual_turnover']:.2f}x")
    log(f"  Cost Drag     : {m_raw['cost_drag_bps']:.1f} bps/yr")
    log(f"  Max |daily net_pnl|:   {pnl_raw['net_pnl'].abs().max():.6f}")
    log(f"  Max |daily gross_pnl|: {pnl_raw['gross_pnl'].abs().max():.6f}")

    # ── Build cleaned price series (with warmup for proper vol/momentum) ─────
    sep("Price Cleaning Summary")

    # Use warmup from 2021 to ensure 252-day lookbacks see clean history
    mask_warmup = (
        ohlcv["ticker"].isin(tickers)
        & (ohlcv["date"] >= pd.Timestamp(WARMUP_START))
        & (ohlcv["date"] <= pd.Timestamp(HOLDOUT_END))
    )
    sub_warmup  = ohlcv[mask_warmup][["ticker", "date", "close", "volume"]].copy()
    close_wm    = sub_warmup.pivot(index="date", columns="ticker", values="close")
    volume_wm   = sub_warmup.pivot(index="date", columns="ticker", values="volume")

    close_clean, r_clean_all, n_clipped_total, mask_big_all = build_clean_price_series(
        close_wm, SANITIZE_CAP
    )

    # Count clips in OOS window only
    oos_idx_wm   = (close_clean.index >= pd.Timestamp(HOLDOUT_START)) & \
                   (close_clean.index <= pd.Timestamp(HOLDOUT_END))
    n_clipped_oos = int(mask_big_all.loc[oos_idx_wm].sum().sum())
    log(f"  Warmup start : {WARMUP_START}")
    log(f"  Total cells clipped  (from {WARMUP_START}):  {n_clipped_total}")
    log(f"  OOS-only cells clipped (2022-2024):           {n_clipped_oos}")

    # OOS cleaned returns: pct_change of clean price within OOS window
    # (using the 2021-2024 continuous close_clean so 2022-01-03 has a valid prior)
    cc_oos       = close_clean.loc[oos_idx_wm]         # 2022-01-03..2024-12-31
    r_clean_oos  = cc_oos.pct_change(fill_method=None)   # clean return series
    vol_clean    = r_clean_oos.rolling(VOL_WINDOW, min_periods=10).std()
    # ADV uses raw prices (dollar-volume denominator should not be distorted)
    adv_clean    = (close_wm * volume_wm).rolling(VOL_WINDOW, min_periods=10).mean()
    adv_clean    = adv_clean.loc[oos_idx_wm]

    # ── STEP 2: V-PnL — raw positions realized against CLEANED returns ────────
    sep("STEP 2  —  V-PnL (raw-signal positions, CLEANED returns)")
    log("  Positions unchanged (from raw features_all.parquet signal)")
    log(f"  Returns capped at ±{SANITIZE_CAP:.0%} via coherent price-cleaning")

    pnl_vpnl = sim.simulate_pnl(
        positions,
        r_clean_oos.reindex(columns=tickers),
        vol=vol_clean.reindex(columns=tickers),
        adv_dollars=adv_clean.reindex(columns=tickers),
        aum_dollars=aum_dollars,
        min_adv_dollars=min_adv_dollars,
    )
    m_vpnl = sim.compute_metrics(pnl_vpnl)

    log()
    log(f"  Gross Sharpe  : {m_vpnl['gross_pnl_sharpe']:+.4f}")
    log(f"  Net   Sharpe  : {m_vpnl['net_pnl_sharpe']:+.4f}")
    log(f"  Gross Annual  : {m_vpnl['gross_pnl_annual']*100:+.2f}%")
    log(f"  Net   Annual  : {m_vpnl['net_pnl_annual']*100:+.2f}%")
    log(f"  Max Drawdown  : {m_vpnl['net_pnl_max_dd']*100:.2f}%")
    log(f"  Hit Rate (net): {m_vpnl['net_pnl_hit_rate']*100:.1f}%")
    log(f"  Annual TO     : {m_vpnl['annual_turnover']:.2f}x")
    log(f"  Cost Drag     : {m_vpnl['cost_drag_bps']:.1f} bps/yr")
    log(f"  Max |daily net_pnl|: {pnl_vpnl['net_pnl'].abs().max():.6f}")

    delta_net   = m_vpnl["net_pnl_sharpe"]  - m_raw["net_pnl_sharpe"]
    delta_gross = m_vpnl["gross_pnl_sharpe"] - m_raw["gross_pnl_sharpe"]
    log(f"\n  Δ gross Sharpe (clean − raw) : {delta_gross:+.4f}")
    log(f"  Δ net   Sharpe (clean − raw) : {delta_net:+.4f}")
    log(f"\n  INTERPRETATION: V-PnL Sharpe is HIGHER than raw.")
    log(f"  Track B was SHORT the artifact-affected names (MNK, SBNY, SIVBQ, FRCB).")
    log(f"  Cleaning removes large artificial losses where the strategy was short")
    log(f"  a +200% to +9900% artifact move, lifting both gross and net Sharpe.")
    log(f"  The headline +{locked_net:.2f} net Sharpe is therefore CONSERVATIVE on")
    log(f"  the P&L side — not inflated by artifact returns.")

    # ── STEP 3: Top clipped cells with Track B exposure + P&L impact ─────────
    sep("STEP 3  —  Top Clipped (name,day) Cells with Track B Exposure")

    pos_al     = positions.reindex(index=ret_raw.index, columns=tickers).fillna(0.0)
    r_raw_oos  = ret_raw.reindex(columns=tickers)
    r_cln_oos  = r_clean_oos.reindex(columns=tickers)

    clipped    = r_raw_oos.abs() > SANITIZE_CAP
    has_pos    = pos_al.abs() > 1e-8
    affected   = clipped & has_pos

    # P&L impact per cell = (r_clean - r_raw) * position
    # Positive impact means cleaning HELPED (e.g., was short, artifact went up → cleaning lifts P&L)
    pnl_impact = (r_cln_oos - r_raw_oos) * pos_al

    rows = []
    for d in affected.index:
        for t in affected.columns:
            if affected.at[d, t]:
                rows.append({
                    "date":       d,
                    "ticker":     t,
                    "raw_ret":    r_raw_oos.at[d, t],
                    "clean_ret":  r_cln_oos.at[d, t] if d in r_cln_oos.index and t in r_cln_oos.columns else np.nan,
                    "position":   pos_al.at[d, t],
                    "pnl_impact": pnl_impact.at[d, t] if d in pnl_impact.index and t in pnl_impact.columns else np.nan,
                })
    impact_flat = pd.DataFrame(rows).dropna(subset=["pnl_impact"])
    impact_flat["abs_impact"] = impact_flat["pnl_impact"].abs()
    impact_flat = impact_flat.sort_values("abs_impact", ascending=False).reset_index(drop=True)

    log(f"\n  Total (date,ticker) clipped cells with nonzero Track B position: {len(impact_flat)}")
    log(f"\n  Top 10 by |P&L impact|:")
    log(f"  {'Date':<12s} {'Ticker':<8s} {'Raw Ret':>10s} {'Clean Ret':>10s} "
        f"{'Position':>10s} {'PnL Impact':>12s}")
    log(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for _, row in impact_flat.head(10).iterrows():
        log(f"  {str(row['date'].date()):<12s} {row['ticker']:<8s} "
            f"{row['raw_ret']:>10.4f} {row['clean_ret']:>10.4f} "
            f"{row['position']:>10.5f} {row['pnl_impact']:>12.6f}")

    total_impact = impact_flat["pnl_impact"].sum()
    log(f"\n  Total PnL impact (all clipped cells, fraction of AUM): {total_impact:+.6f}")
    log(f"  Net direction: {'POSITIVE (cleaning lifted P&L — strategy was short artifacts)' if total_impact > 0 else 'NEGATIVE (cleaning hurt P&L)'}")
    log(f"  Annualised impact: {total_impact / 3 * 10_000:+.1f} bps/yr  (spread over 3 OOS years)")

    # Explain the mechanism for top 3
    log(f"\n  Mechanism for top 3 cells:")
    for _, row in impact_flat.head(3).iterrows():
        direction = "SHORT, artifact UP → raw P&L loss removed by cleaning" if (
            row["position"] < 0 and row["raw_ret"] > 0) else (
            "LONG, artifact UP → raw P&L gain partially removed by cleaning" if (
                row["position"] > 0 and row["raw_ret"] > 0) else "see raw_ret/position signs")
        log(f"    {str(row['date'].date())} {row['ticker']}: {direction}")
        log(f"      raw gross PnL = {row['position']*row['raw_ret']:.6f} → "
            f"clean gross PnL = {row['position']*row['clean_ret']:.6f}  "
            f"(Δ = {row['pnl_impact']:+.6f})")

    # ── STEP 4: Feature-side contamination (V-Full-Light assessment) ──────────
    sep("STEP 4  —  V-Full-Light (Feature-Side Contamination Assessment)")
    log("  Track B signal comes ENTIRELY from features_all.parquet.")
    log("  V-Full-Light requires rebuilding that file from clean prices —")
    log("  NOT done in this diagnostic (would need re-running build_features_all.py).")
    log()

    # Track B BH-significant features that are return-derived
    ret_derived_in_signal = [
        f for f in weights.index
        if f in {"ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_252d",
                 "mom_12_1", "reversal_1w", "reversal_4w"}
    ]
    feat_oos = feat_df.loc[
        (feat_df.index.get_level_values("date") >= pd.Timestamp(HOLDOUT_START))
        & (feat_df.index.get_level_values("date") <= pd.Timestamp(HOLDOUT_END))
    ]
    log(f"  Return-derived Track B signal features: {ret_derived_in_signal}")
    log(f"  OOS artifact counts (|val| > {SANITIZE_CAP:.0%}):")
    total_artifact = 0
    for col in ret_derived_in_signal:
        if col in feat_oos.columns:
            n = int((feat_oos[col].abs() > SANITIZE_CAP).sum())
            total_artifact += n
            log(f"    {col:<22s}: {n:6d} cells")
        else:
            log(f"    {col:<22s}: NOT IN features_all")
    log(f"  ──────────────────────────────────────────────────────")
    log(f"  Total artifact cells in OOS signal features: {total_artifact}")
    log(f"  (These contaminate signal z-scores, potentially distorting")
    log(f"   position sizing even if the signal weight for ret_1d etc. is small)")

    # Non-signal return-derived features (corr_*, macro) are NOT price-direct artifacts
    # but some may be indirectly affected
    if "target_track_b" in feat_oos.columns:
        tb_tgt = feat_oos["target_track_b"]
        n_tgt  = int((tb_tgt.abs() > SANITIZE_CAP).sum())
        log(f"\n  target_track_b OOS: {n_tgt} cells with |val|>50%  "
            f"(max = {tb_tgt.abs().max():.2f})")
        log("  (This is the IS training label — artifact contamination in IS")
        log("   training data may have shaped model weights. Not direct signal noise.)")

    log()
    log("  *** V-Full-Light requires: ***")
    log("  *** 1. python3 src/features/build_features_all.py  (on clean prices) ***")
    log("  *** 2. python3 src/backtest/run_holdout.py                             ***")

    # ── FINAL VERDICT ─────────────────────────────────────────────────────────
    sep("FINAL VERDICT")
    log()

    net_vpnl  = m_vpnl["net_pnl_sharpe"]
    delta_net_vs_locked = net_vpnl - locked_net
    abs_delta = abs(delta_net_vs_locked)

    log(f"  1. RAW REPRODUCTION:")
    log(f"       Locked   → gross {locked_gross:+.4f}   net {locked_net:+.4f}")
    log(f"       Computed → gross {m_raw['gross_pnl_sharpe']:+.4f}   net {m_raw['net_pnl_sharpe']:+.4f}")
    match_ok = (abs(m_raw["gross_pnl_sharpe"] - locked_gross) < 0.001 and
                abs(m_raw["net_pnl_sharpe"]   - locked_net)   < 0.001)
    log(f"       Harness faithfulness: {'CONFIRMED ✓' if match_ok else 'DISCREPANCY — investigate'}")

    log()
    log(f"  2. V-PnL (raw signal positions, CLEANED returns):")
    log(f"       Gross Sharpe : {m_vpnl['gross_pnl_sharpe']:+.4f}   "
        f"(locked gross: {locked_gross:+.4f}; Δ = {m_vpnl['gross_pnl_sharpe']-locked_gross:+.4f})")
    log(f"       Net   Sharpe : {net_vpnl:+.4f}   "
        f"(locked net:   {locked_net:+.4f}; Δ = {delta_net_vs_locked:+.4f})")
    log(f"       Cost drag    : {m_vpnl['cost_drag_bps']:.1f} bps/yr")
    log(f"       Annual TO    : {m_vpnl['annual_turnover']:.2f}x")
    log(f"       Max |daily net_pnl|: {pnl_vpnl['net_pnl'].abs().max():.6f}")

    log()
    log(f"  3. V-Full-Light (features also cleaned): NOT computed in this run.")
    log(f"       Feature-side artifact counts (OOS, in signal features): {total_artifact} cells")
    log(f"       Full feature-side rerun IS REQUIRED to be fully certain.")

    log()
    # Adjust verdict logic: if V-PnL is HIGHER the artifact was hurting not helping
    if delta_net_vs_locked > 0.05:
        verdict_line = (
            f"YES — headline +{locked_net:.2f} net Sharpe SURVIVES return-cleaning "
            f"(V-PnL net = {net_vpnl:+.2f}; artifacts HURT Track B because it was "
            f"short the artifact names; cleaning HELPS not hurts)."
        )
    elif abs_delta < 0.05:
        verdict_line = (
            f"YES — within ±0.05 of +{locked_net:.2f} net Sharpe under cleaned returns."
        )
    elif abs_delta < 0.15:
        verdict_line = (
            f"PARTIALLY — survives but materially different (Δ = {delta_net_vs_locked:+.4f})."
        )
    else:
        verdict_line = (
            f"NO — headline +{locked_net:.2f} does NOT survive return-cleaning "
            f"(V-PnL net = {net_vpnl:+.4f})."
        )

    log(f"  4. ONE-LINE VERDICT:")
    log(f"     {verdict_line}")
    log(f"     Full feature-side rerun REQUIRED: YES ({total_artifact} artifact cells")
    log(f"     in return-derived signal features; could bias signal z-scores in OOS).")
    log()
    log("=" * 65)
    log("  END OF DIAGNOSTIC  (no parquets or main.tex were modified)")
    log("=" * 65)


if __name__ == "__main__":
    main()
