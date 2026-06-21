"""P1.1 — Baseline strategy comparison.

Computes four baselines under the same Almgren-Chriss cost model:
  1. Buy-and-hold SPY (long-only, no rebalancing, no AC costs by convention)
  2. Equal-weight 12-1 momentum L/S  (weekly rebalance, same AC params)
  3. Short-term 5-day reversal L/S   (weekly rebalance, same AC params)
  4. Track B gross-only              (Track B signal, zero costs)

Periods: IS 2013-2021 and OOS 2022-2024.
All L/S baselines use the same PortfolioSimulator (spread=3bps, η=0.10)
and the same ADV filter ($1M) as the main strategy.

COHERENCE FIX (Option A)
------------------------
A single cleaned price series (close_clean) is derived by:
  1. Computing raw daily returns r = close.pct_change()
  2. Clipping to ±SANITIZE_CAP (default 50%)
  3. Reconstructing prices: close_clean = first_close * cumprod(1 + r_clean)
This ensures signals (momentum, reversal) and realized P&L both use the
SAME artifact-free data — eliminating the prior mismatch where signals
saw raw +445% jumps but P&L was capped at ±50%.

Use --no-sanitize to reproduce original raw behavior for A/B comparison.

Outputs:
    data/processed/baselines_pnl.parquet    — daily P&L per strategy × period
    data/processed/baselines_metrics.parquet — annualised metrics summary

Run:
    python3 -u src/backtest/run_baselines.py              # coherent-clean
    python3 -u src/backtest/run_baselines.py --no-sanitize  # raw (original)
"""

import argparse
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

OHLCV_PATH  = ROOT / "data" / "processed" / "daily_ohlcv.parquet"
FEAT_PATH   = ROOT / "data" / "processed" / "features_all.parquet"
FDR_PATH    = ROOT / "data" / "processed" / "fdr_results.parquet"
SHAP_PATH   = ROOT / "data" / "processed" / "shap_summary.parquet"
SPY_PATH    = ROOT / "data" / "processed" / "spy_returns.parquet"
CFG_PATH    = ROOT / "configs" / "backtest.yaml"

OUT_PNL     = ROOT / "data" / "processed" / "baselines_pnl.parquet"
OUT_METRICS = ROOT / "data" / "processed" / "baselines_metrics.parquet"

IS_START  = "2013-01-01"; IS_END  = "2021-12-31"
OOS_START = "2022-01-01"; OOS_END = "2024-12-31"
REBAL_FREQ = 5   # weekly
VOL_WINDOW = 21

# Module-level cap constant.  Set to None (or pass --no-sanitize) to reproduce
# the original raw behavior without any price cleaning.
SANITIZE_CAP = 0.50   # ±50%


def log(msg): print(msg, flush=True)


def build_clean_prices(close_w: pd.DataFrame, cap: float) -> tuple:
    """Reconstruct a clean price series by cumulatively compounding clipped returns.

    This is the COHERENT sanitization path (Option A):
      1. r_raw  = close_w.pct_change()           — raw daily simple returns
      2. r_clean = r_raw.clip(-cap, +cap)         — clip artifact moves
      3. close_clean[t] = close_w.iloc[0] * prod_{s=1..t}(1 + r_clean[s])

    Using close_clean for BOTH signal construction AND realized P&L guarantees
    that no position is opened based on an artifact price move that the P&L
    side has already capped.  This eliminates the prior signal-vs-P&L mismatch.

    Returns
    -------
    close_clean : pd.DataFrame   — cleaned price proxy (same shape as close_w)
    r_clean     : pd.DataFrame   — the clipped daily returns (= pct_change of close_clean)
    n_clipped   : int            — number of (ticker, day) cells clipped
    """
    r_raw = close_w.pct_change(fill_method=None)
    n_clipped = int((r_raw.abs() > cap).sum().sum())
    if n_clipped > 0:
        log(f"  [build_clean_prices] Clipped {n_clipped} (ticker,day) cells "
            f"with |raw_ret| > {cap:.0%} to ±{cap:.0%}")

    r_clean = r_raw.clip(lower=-cap, upper=cap)

    # Reconstruct price proxy: start from the first valid close level per ticker,
    # then compound the cleaned returns forward.
    # first_row is the row of raw close prices before any pct_change NaN.
    first_close = close_w.iloc[0]                     # Series (ticker → price)
    # growth factor from first valid row
    growth = (1.0 + r_clean.fillna(0.0)).cumprod()   # (date × ticker)
    # Restore first row to exact original close (growth[0] = 1.0 after fillna)
    close_clean = growth.multiply(first_close, axis="columns")
    # Keep NaN where original close was NaN (new ticker, delisted gap, etc.)
    close_clean[close_w.isna()] = np.nan

    return close_clean, r_clean, n_clipped


def build_returns_vol_adv(ohlcv, tickers, start, end, close_clean_full=None):
    """Return (returns, vol, adv_dollars) DataFrames for a date range.

    If close_clean_full is provided (coherent-clean mode), returns are taken
    directly from it (these equal the clipped r_clean for the relevant tickers).
    ADV denominator uses the original close prices to preserve dollar-volume scale.

    If close_clean_full is None (raw/--no-sanitize mode), falls back to the
    old behavior: raw pct_change with no clipping.
    """
    mask = (
        ohlcv["ticker"].isin(tickers)
        & (ohlcv["date"] >= pd.Timestamp(start))
        & (ohlcv["date"] <= pd.Timestamp(end))
    )
    sub = ohlcv[mask][["ticker", "date", "close", "volume"]].copy()
    close_w  = sub.pivot(index="date", columns="ticker", values="close")
    vol_w    = sub.pivot(index="date", columns="ticker", values="volume")

    if close_clean_full is not None:
        # Coherent path: extract returns from the pre-built cleaned price proxy
        tickers_avail = [t for t in tickers if t in close_clean_full.columns]
        cc = close_clean_full.loc[
            (close_clean_full.index >= pd.Timestamp(start)) &
            (close_clean_full.index <= pd.Timestamp(end)),
            tickers_avail
        ]
        returns = cc.pct_change(fill_method=None)
        # For tickers not in close_clean_full, fall back to raw (should be none)
        missing = [t for t in tickers if t not in close_clean_full.columns]
        if missing:
            log(f"  [WARN] {len(missing)} tickers missing from close_clean; using raw for them")
            raw_miss = close_w[missing].pct_change(fill_method=None)
            returns = pd.concat([returns, raw_miss], axis=1)[tickers_avail + missing]
    else:
        # Raw path (--no-sanitize): original behavior
        returns = close_w.pct_change(fill_method=None)

    # ADV uses original close × volume (dollar volume should not be distorted by cleaning)
    vol      = returns.rolling(VOL_WINDOW, min_periods=10).std()
    dollar_v = (close_w * vol_w).rolling(VOL_WINDOW, min_periods=10).mean()
    mask_d   = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    return returns[mask_d], vol[mask_d], dollar_v[mask_d]


def compute_sharpe(r):
    """Annualised Sharpe (ddof=1)."""
    s = r.dropna()
    sd = s.std(ddof=1)
    return float(s.mean() / sd * np.sqrt(252)) if sd > 1e-10 else np.nan


def run_ls_strategy(signals, returns, vol, adv, cfg, sim, label, period):
    """Run L/S strategy through PortfolioSimulator, return metrics dict."""
    positions = sim.signal_to_positions(signals, lag=1, rebal_freq=REBAL_FREQ)
    min_adv   = cfg.get("min_adv_dollars", 1e6)
    aum       = cfg.get("aum_dollars", 1e8)
    pnl = sim.simulate_pnl(positions, returns, vol=vol,
                            adv_dollars=adv, aum_dollars=aum,
                            min_adv_dollars=min_adv)
    m = sim.compute_metrics(pnl)
    return {
        "strategy": label,
        "period":   period,
        "gross_sr": m.get("gross_pnl_sharpe", np.nan),
        "net_sr":   m.get("net_pnl_sharpe",   np.nan),
        "gross_ann": m.get("gross_pnl_annual", np.nan),
        "net_ann":   m.get("net_pnl_annual",   np.nan),
        "max_dd":    m.get("net_pnl_max_dd",   np.nan),
        "annual_to": m.get("annual_turnover",  np.nan),
        "cost_drag": m.get("cost_drag_bps",    np.nan),
    }, pnl


def build_momentum_signal(close_px, start, end):
    """12-1 month momentum: 252d return - 21d return (cross-section z-scored).

    close_px must be the SAME price series used for realized P&L (close_clean
    in coherent mode, raw close in --no-sanitize mode).
    """
    mom = close_px.pct_change(252, fill_method=None) - close_px.pct_change(21, fill_method=None)
    mask = (mom.index >= pd.Timestamp(start)) & (mom.index <= pd.Timestamp(end))
    return mom[mask]


def build_reversal_signal(close_px, start, end):
    """Short-term reversal: -(5-day return).

    close_px must be the SAME price series used for realized P&L (close_clean
    in coherent mode, raw close in --no-sanitize mode).
    """
    rev = -close_px.pct_change(5, fill_method=None)
    mask = (rev.index >= pd.Timestamp(start)) & (rev.index <= pd.Timestamp(end))
    return rev[mask]


def main():
    parser = argparse.ArgumentParser(description="P1.1 Baseline Comparison")
    parser.add_argument(
        "--no-sanitize", action="store_true",
        help="Disable price cleaning — reproduce original raw behavior (for A/B comparison)"
    )
    args = parser.parse_args()

    sanitize = not args.no_sanitize
    cap = SANITIZE_CAP if sanitize else None

    log("=== P1.1 Baseline Comparison ===")
    if sanitize:
        log(f"  Mode: COHERENT-CLEAN  (cap=±{cap:.0%}; signals + P&L use same cleaned prices)")
    else:
        log("  Mode: RAW  (--no-sanitize; original behavior, no price cleaning)")
    log("")
    t0 = time.time()

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    sim_cost = PortfolioSimulator(config_path=str(CFG_PATH))
    sim_free = PortfolioSimulator(config_path=str(CFG_PATH),
                                  spread_bps=0.0, impact_coeff=0.0)

    ohlcv = pd.read_parquet(OHLCV_PATH)
    all_tickers = ohlcv["ticker"].unique().tolist()

    # Wide close table (full universe, full date range — includes warmup for momentum)
    log("Building wide close table …")
    full = ohlcv[ohlcv["date"] >= pd.Timestamp("2012-01-01")].copy()
    close_w_raw = full.pivot(index="date", columns="ticker", values="close").sort_index()

    # ── Build the single cleaned price proxy (Option A) ─────────────────────
    # This is computed ONCE over the full date range (from 2012 onwards) so that
    # the momentum lookback (252 days) is also computed on clean prices.
    total_clip_is = 0
    total_clip_oos = 0
    if sanitize:
        log(f"Building cleaned price proxy (clip ±{cap:.0%}) over full history …")
        close_clean, r_clean_full, total_clip_all = build_clean_prices(close_w_raw, cap)
        log(f"  Total cells clipped (full history from 2012): {total_clip_all}")
        # Count IS vs OOS clips from raw returns
        r_raw_full = close_w_raw.pct_change(fill_method=None)
        is_mask  = (r_raw_full.index >= pd.Timestamp(IS_START)) & \
                   (r_raw_full.index <= pd.Timestamp(IS_END))
        oos_mask = (r_raw_full.index >= pd.Timestamp(OOS_START)) & \
                   (r_raw_full.index <= pd.Timestamp(OOS_END))
        total_clip_is  = int((r_raw_full[is_mask].abs() > cap).sum().sum())
        total_clip_oos = int((r_raw_full[oos_mask].abs() > cap).sum().sum())
        log(f"  IS  clip count ({IS_START}→{IS_END}):  {total_clip_is}")
        log(f"  OOS clip count ({OOS_START}→{OOS_END}): {total_clip_oos}")
        # The price series used for signals — same as P&L source
        close_for_signals = close_clean
        close_clean_full  = close_clean
    else:
        log("  Price cleaning DISABLED (raw mode)")
        close_for_signals = close_w_raw
        close_clean_full  = None   # signals to use raw

    # Track B signal weights (fixed from IS analysis)
    fdr_df  = pd.read_parquet(FDR_PATH)
    shap_df = pd.read_parquet(SHAP_PATH)
    feat_df = pd.read_parquet(FEAT_PATH)
    try:
        weights_b = build_signal_weights("track_b", fdr_df, shap_df)
    except ValueError as e:
        log(f"  [INFO] Track B signal weights unavailable ({e}); Track B gross-only baseline will be skipped.")
        weights_b = None

    all_pnl     = []
    all_metrics = []

    for period, start, end in [("IS 2013-21", IS_START, IS_END),
                                ("OOS 2022-24", OOS_START, OOS_END)]:
        log(f"\n{'='*55}")
        log(f"  Period: {period}  ({start} → {end})")
        log(f"{'='*55}")

        # Universe tickers active in this period
        mask_period = (
            ohlcv["date"] >= pd.Timestamp(start)
        ) & (ohlcv["date"] <= pd.Timestamp(end))
        tickers = ohlcv.loc[mask_period, "ticker"].unique().tolist()
        log(f"  Universe: {len(tickers)} tickers")

        # Realized returns, vol, ADV — from cleaned prices (coherent) or raw
        returns, vol, adv = build_returns_vol_adv(
            ohlcv, tickers, start, end,
            close_clean_full=close_clean_full
        )

        # ── 1. Buy-and-hold SPY ──────────────────────────────────────────
        spy_df = pd.read_parquet(SPY_PATH)
        spy_col = spy_df.columns[0]
        spy_r = spy_df[spy_col].squeeze()
        spy_r.index = pd.to_datetime(spy_r.index)
        mask_spy = (spy_r.index >= pd.Timestamp(start)) & (spy_r.index <= pd.Timestamp(end))
        spy_slice = spy_r[mask_spy]
        spy_pnl = pd.DataFrame({
            "gross_pnl":   spy_slice.values,
            "spread_cost": 0.0,
            "impact_cost": 0.0,
            "borrow_cost": 0.0,
            "total_cost":  0.0,
            "net_pnl":     spy_slice.values,
            "turnover":    0.0,
        }, index=spy_slice.index)
        spy_sr = compute_sharpe(spy_slice)
        spy_ann = float(spy_slice.mean() * 252)
        cum_spy = (1 + spy_slice).cumprod()
        spy_dd  = float((cum_spy / cum_spy.cummax() - 1).min())
        spy_m = {
            "strategy": "Buy-and-Hold SPY",
            "period":   period,
            "gross_sr": spy_sr,
            "net_sr":   spy_sr,
            "gross_ann": spy_ann,
            "net_ann":   spy_ann,
            "max_dd":    spy_dd,
            "annual_to": 0.0,
            "cost_drag": 0.0,
        }
        spy_pnl["strategy"] = "Buy-and-Hold SPY"
        spy_pnl["period"]   = period
        all_pnl.append(spy_pnl)
        all_metrics.append(spy_m)
        log(f"\n  SPY B&H: gross SR={spy_sr:+.3f}  ann={spy_ann*100:+.2f}%  DD={spy_dd*100:.2f}%")

        # ── 2. Momentum L/S ─────────────────────────────────────────────
        # COHERENT: build signal from the SAME price series as P&L
        mom_sig = build_momentum_signal(close_for_signals, start, end)
        # Restrict to universe tickers
        common = [t for t in tickers if t in mom_sig.columns]
        mom_sig = mom_sig[common]
        mom_m, mom_pnl = run_ls_strategy(
            mom_sig, returns[common], vol[common], adv[common],
            cfg, sim_cost, "Momentum L/S (12-1)", period
        )
        mom_pnl["strategy"] = "Momentum L/S (12-1)"
        mom_pnl["period"]   = period
        all_pnl.append(mom_pnl)
        all_metrics.append(mom_m)
        log(f"  Momentum L/S: gross SR={mom_m['gross_sr']:+.3f}  net SR={mom_m['net_sr']:+.3f}")

        # ── 3. Short-term reversal L/S ───────────────────────────────────
        # COHERENT: build signal from the SAME price series as P&L
        rev_sig = build_reversal_signal(close_for_signals, start, end)
        common_r = [t for t in tickers if t in rev_sig.columns]
        rev_sig = rev_sig[common_r]
        rev_m, rev_pnl = run_ls_strategy(
            rev_sig, returns[common_r], vol[common_r], adv[common_r],
            cfg, sim_cost, "Reversal L/S (5d)", period
        )
        rev_pnl["strategy"] = "Reversal L/S (5d)"
        rev_pnl["period"]   = period
        all_pnl.append(rev_pnl)
        all_metrics.append(rev_m)
        log(f"  Reversal L/S: gross SR={rev_m['gross_sr']:+.3f}  net SR={rev_m['net_sr']:+.3f}")

        # ── 4. Track B gross-only (zero costs) ──────────────────────────
        if weights_b is not None:
            sig_b = generate_composite_signal(feat_df, weights_b,
                                               start_date=start, end_date=end)
            tickers_b = sig_b.columns.tolist()
            ret_b  = returns.reindex(columns=tickers_b).fillna(0)
            vol_b  = vol.reindex(columns=tickers_b).fillna(0.02)
            adv_b  = adv.reindex(columns=tickers_b).fillna(1e8)
            positions_b = sim_free.signal_to_positions(sig_b, lag=1, rebal_freq=REBAL_FREQ)
            pnl_b = sim_free.simulate_pnl(positions_b, ret_b, vol=vol_b,
                                           adv_dollars=adv_b,
                                           aum_dollars=cfg.get("aum_dollars", 1e8),
                                           min_adv_dollars=0.0)
            mb  = sim_free.compute_metrics(pnl_b)
            bm  = {
                "strategy": "Track B (gross, no cost)",
                "period":   period,
                "gross_sr": mb.get("gross_pnl_sharpe", np.nan),
                "net_sr":   mb.get("net_pnl_sharpe",   np.nan),
                "gross_ann": mb.get("gross_pnl_annual", np.nan),
                "net_ann":   mb.get("net_pnl_annual",   np.nan),
                "max_dd":    mb.get("net_pnl_max_dd",   np.nan),
                "annual_to": mb.get("annual_turnover",  np.nan),
                "cost_drag": 0.0,
            }
            pnl_b["strategy"] = "Track B (gross, no cost)"
            pnl_b["period"]   = period
            all_pnl.append(pnl_b)
            all_metrics.append(bm)
            log(f"  Track B gross-only: gross SR={bm['gross_sr']:+.3f}")
        else:
            log("  [SKIP] Track B gross-only: no BH-rejected features (weights unavailable)")

    # ── Save ─────────────────────────────────────────────────────────────
    pnl_out = pd.concat([p for p in all_pnl], axis=0)
    pnl_out.to_parquet(OUT_PNL, index=True)

    met_out = pd.DataFrame(all_metrics).round(4)
    met_out.to_parquet(OUT_METRICS, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 15)
    mode_label = f"coherent-clean (cap=±{cap:.0%})" if sanitize else "RAW (no sanitize)"
    print(f"\n=== Baseline Metrics Summary  [{mode_label}] ===")
    print(met_out[["strategy","period","gross_sr","net_sr","max_dd","annual_to","cost_drag"]]
          .to_string(index=False))

    # ── Clip count summary ───────────────────────────────────────────────
    if sanitize:
        print(f"\n  Clip counts: IS={total_clip_is}  OOS={total_clip_oos}  "
              f"(full history={total_clip_all})")

    # ── Sanity: no |net_pnl| > 0.5 remains ─────────────────────────────
    if "net_pnl" in pnl_out.columns:
        max_abs_net = pnl_out["net_pnl"].abs().max()
        print(f"\n  Sanity: max |daily net_pnl| in output = {max_abs_net:.4f}  "
              f"({'PASS' if max_abs_net <= 0.5 else 'FAIL — still has |pnl|>0.5'})")

    print(f"\nSaved → {OUT_PNL}")
    print(f"Saved → {OUT_METRICS}")
    log(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
