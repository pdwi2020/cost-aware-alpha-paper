"""IA1+IA2 — Tradable-Alpha FDR (TA-FDR) with permutation-calibrated null.

## Design (IA1 — Opus specification implemented here)

### Statistic
For each candidate feature f (IS 2013-2021 only):
  T_f = net Sharpe ratio of the single-feature strategy run through
        PortfolioSimulator with locked Almgren-Chriss costs.
  Signal = sign(mean_IC_f) × feature_f values (cross-sectionally z-scored per date).
  Almgren-Chriss impact is endogenous to position size — high-turnover features
  pay more impact and will have lower T_f under the null, even when noise.

### Permutation null
Repeat B times (default 1000):
  1. Block-permute the 5-day-forward target labels (block = 5 days to respect
     the 5-day return overlap). Permute blocks of rows within each cross-section
     date, not dates, to preserve cross-sectional structure.
  2. Rebuild the single-feature signal using permuted labels (IC sign re-estimated
     on permuted data to avoid sign peeking).
  3. Run through PortfolioSimulator → permuted net Sharpe T_f^(b).
The null distribution is the joint distribution of net-of-impact performance
under no genuine predictability. It automatically penalises high-turnover
features because their impact drags T_f^(b) downward even when labels are random.

### p-value + BH
  p_f = (1 + #{T_f^(b) ≥ T_f}) / (1 + B)    (one-sided: high T_f is signal)
  Apply BH at q=0.10 on {p_f} → TA-FDR accepted set.

### Prior-art delta (must cite in paper)
- Bajgrowicz & Scaillet (2012, JFE): rules / proportional cost / single time-series.
  Our delta: cross-sectional feature discovery / size-dependent AC market impact /
  permutation null / regime-aware walk-forward protocol.
- α-investing (Foster & Stine 2008, JRSS-B): data-collection cost, not market impact.
  Our delta: market-impact wealth penalty (the more you trade a feature, the more
  impact you pay, even under the null).

### OOS validation
After TA-FDR selection (IS only), rerun OOS 2022-2024 with the TA-FDR composite
and compare to the static-BH composite. Acceptance: static-BH OOS net SR = 0.50 ±0.02.

Outputs:
    data/processed/ta_fdr.parquet             — per-feature statistics + rejection flags
    data/processed/ta_fdr_oos_metrics.parquet — OOS comparison: static-BH vs TA-FDR
    data/processed/ta_fdr_null_dist.parquet   — permutation null distributions (optional)

Run (slow — ~30-60 min with B=1000):
    python3 -u src/backtest/run_ta_fdr.py

For a fast test with B=50:
    python3 -u src/backtest/run_ta_fdr.py --B 50
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
from src.fdr.bh_correction import benjamini_hochberg
from src.fdr.run_fdr import get_surviving_features

FDR_PATH   = ROOT / "data" / "processed" / "fdr_results.parquet"
SHAP_PATH  = ROOT / "data" / "processed" / "shap_summary.parquet"
FEAT_PATH  = ROOT / "data" / "processed" / "features_all.parquet"
OHLCV_PATH = ROOT / "data" / "processed" / "daily_ohlcv.parquet"
CFG_PATH   = ROOT / "configs" / "backtest.yaml"

OUT_TAFDR   = ROOT / "data" / "processed" / "ta_fdr.parquet"
OUT_OOS     = ROOT / "data" / "processed" / "ta_fdr_oos_metrics.parquet"
OUT_NULL    = ROOT / "data" / "processed" / "ta_fdr_null_dist.parquet"

IS_START   = "2013-01-01"
IS_END     = "2021-12-31"
HOLDOUT_START = "2022-01-01"
HOLDOUT_END   = "2024-12-31"
FDR_Q      = 0.10
BLOCK_LEN  = 5     # 5-day block for permutation (matches return horizon)
REBAL_FREQ = 5
VOL_WINDOW = 21


def log(msg): print(msg, flush=True)


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_ohlcv_matrices(ohlcv: pd.DataFrame, tickers: list, start: str, end: str):
    mask = (
        ohlcv["ticker"].isin(tickers)
        & (ohlcv["date"] >= pd.Timestamp(start))
        & (ohlcv["date"] <= pd.Timestamp(end))
    )
    sub = ohlcv[mask][["ticker","date","close","volume"]].copy()
    close_w  = sub.pivot(index="date", columns="ticker", values="close")
    vol_w    = sub.pivot(index="date", columns="ticker", values="volume")
    returns  = close_w.pct_change()
    sigma    = returns.rolling(VOL_WINDOW, min_periods=10).std()
    adv      = (close_w * vol_w).rolling(VOL_WINDOW, min_periods=10).mean()
    m = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    return returns[m], sigma[m], adv[m]


def build_single_feature_signal(
    feat_df: pd.DataFrame,
    feature: str,
    ic_sign: float,
    start: str,
    end: str,
) -> pd.DataFrame:
    """(date × ticker) signal matrix — vectorized pivot, no Python date loop."""
    dates = feat_df.index.get_level_values("date")
    mask  = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    sub   = feat_df.loc[mask, [feature]]
    sig = sub[feature].unstack(level="ticker")  # date × ticker
    # Vectorized cross-sectional median fill (no Python row loop)
    arr = sig.values.copy()
    row_med = np.nanmedian(arr, axis=1, keepdims=True)
    nan_mask = np.isnan(arr)
    arr[nan_mask] = np.broadcast_to(row_med, arr.shape)[nan_mask]
    sig = pd.DataFrame(arr, index=sig.index, columns=sig.columns)
    return (ic_sign * sig).dropna(how="all")


def _fast_net_sharpe(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    sigma: pd.DataFrame,
    adv: pd.DataFrame,
    spread_bps: float,
    impact_coeff: float,
    aum_dollars: float,
    min_adv_dollars: float,
    gross_target: float = 1.0,
    max_pos: float = 0.05,
    rebal_freq: int = REBAL_FREQ,
) -> float:
    """Vectorized single-feature net Sharpe — avoids Python date loops."""
    if signal.empty or len(signal) < 50:
        return np.nan

    # Cross-sectional z-score per date (vectorized)
    mu  = signal.mean(axis=1)
    std = signal.std(axis=1).replace(0, np.nan)
    z   = signal.sub(mu, axis=0).div(std, axis=0)

    # Normalize to gross_target
    gross = z.abs().sum(axis=1).replace(0, np.nan)
    w     = z.div(gross, axis=0).mul(gross_target)

    # Position cap
    w = w.clip(-max_pos, max_pos)
    gross2 = w.abs().sum(axis=1).replace(0, np.nan)
    w = w.div(gross2, axis=0).mul(gross_target)
    w = w.fillna(0.0)

    # Weekly rebalancing: hold positions for rebal_freq days (vectorized ffill)
    if rebal_freq > 1:
        arr  = w.values.copy()
        n    = len(arr)
        mask = np.arange(n) % rebal_freq != 0   # non-rebalance days
        for i in range(1, n):
            if mask[i]:
                arr[i] = arr[i - 1]
        w = pd.DataFrame(arr, index=w.index, columns=w.columns)

    # Lag 1
    pos = w.shift(1).fillna(0.0)

    # Align with returns
    common_dates   = pos.index.intersection(returns.index)
    common_tickers = pos.columns.intersection(returns.columns)
    pos_ = pos.loc[common_dates, common_tickers].values
    ret_ = returns.loc[common_dates, common_tickers].fillna(0.0).values
    sig_ = sigma.loc[common_dates, common_tickers].fillna(0.02).values
    adv_ = adv.loc[common_dates, common_tickers].fillna(aum_dollars).values

    # ADV filter: zero out positions in illiquid names
    liquid = (adv_ >= min_adv_dollars).astype(float)
    pos_  *= liquid

    delta = np.diff(np.vstack([np.zeros((1, pos_.shape[1])), pos_]), axis=0)
    delta_abs = np.abs(delta)

    # Gross P&L
    gross_pnl = (pos_ * ret_).sum(axis=1)

    # Spread cost (scalar per day)
    spread_cost_per_unit = spread_bps / 10_000 / 2
    spread_cost = delta_abs.sum(axis=1) * spread_cost_per_unit

    # AC impact cost (vectorized)
    part = (delta_abs * aum_dollars) / np.clip(adv_, 1.0, None)
    ic   = (impact_coeff * sig_ * np.sqrt(part) * delta_abs).sum(axis=1)

    net_pnl = gross_pnl - spread_cost - ic

    if net_pnl.std() < 1e-10:
        return np.nan
    return float(net_pnl.mean() / net_pnl.std() * np.sqrt(252))


def run_single_feature_backtest(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    sigma: pd.DataFrame,
    adv: pd.DataFrame,
    sim: PortfolioSimulator,
    cfg: dict,
) -> float:
    """Return net Sharpe for a single-feature strategy (vectorized fast path)."""
    return _fast_net_sharpe(
        signal, returns, sigma, adv,
        spread_bps=sim.spread_bps,
        impact_coeff=sim.impact_coeff,
        aum_dollars=cfg.get("aum_dollars", 1e8),
        min_adv_dollars=cfg.get("min_adv_dollars", 1e6),
    )


# ── Pre-compute positions + costs (called once per feature, not per permutation) ─

def _precompute_pnl_components(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    sigma: pd.DataFrame,
    adv: pd.DataFrame,
    spread_bps: float,
    impact_coeff: float,
    aum_dollars: float,
    min_adv_dollars: float,
    gross_target: float = 1.0,
    max_pos: float = 0.05,
    rebal_freq: int = REBAL_FREQ,
):
    """Return (pos_arr, ret_arr, costs_arr, T, N) as numpy arrays aligned on
    common (dates × tickers).  Costs are independent of realised returns, so
    they can be precomputed once and reused across all permutations.
    """
    if signal.empty or len(signal) < 50:
        return None

    mu  = signal.mean(axis=1)
    std = signal.std(axis=1).replace(0, np.nan)
    z   = signal.sub(mu, axis=0).div(std, axis=0)
    gross = z.abs().sum(axis=1).replace(0, np.nan)
    w = z.div(gross, axis=0).mul(gross_target)
    w = w.clip(-max_pos, max_pos)
    gross2 = w.abs().sum(axis=1).replace(0, np.nan)
    w = w.div(gross2, axis=0).mul(gross_target).fillna(0.0)

    if rebal_freq > 1:
        arr = w.values.copy()
        for i in range(1, len(arr)):
            if i % rebal_freq != 0:
                arr[i] = arr[i - 1]
        w = pd.DataFrame(arr, index=w.index, columns=w.columns)

    pos = w.shift(1).fillna(0.0)
    common_dates   = pos.index.intersection(returns.index)
    common_tickers = pos.columns.intersection(returns.columns)

    pos_ = pos.loc[common_dates, common_tickers].values
    ret_ = returns.loc[common_dates, common_tickers].fillna(0.0).values
    sig_ = sigma.loc[common_dates, common_tickers].fillna(0.02).values
    adv_ = adv.loc[common_dates, common_tickers].fillna(aum_dollars).values

    liquid = (adv_ >= min_adv_dollars).astype(float)
    pos_  *= liquid

    delta_abs = np.abs(np.diff(np.vstack([np.zeros((1, pos_.shape[1])), pos_]), axis=0))
    spread_cost = delta_abs.sum(axis=1) * (spread_bps / 10_000 / 2)
    part        = (delta_abs * aum_dollars) / np.clip(adv_, 1.0, None)
    impact_cost = (impact_coeff * sig_ * np.sqrt(part) * delta_abs).sum(axis=1)
    costs = spread_cost + impact_cost

    return pos_, ret_, costs


# ── Main IC sign estimation (pre-compute once for observed data) ───────────────

def estimate_ic_signs(
    feat_df: pd.DataFrame,
    features: list,
    target_col: str,
    start: str,
    end: str,
) -> pd.Series:
    """Sign of mean Spearman IC for each feature on IS data."""
    from scipy.stats import spearmanr
    dates  = feat_df.index.get_level_values("date")
    mask   = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    sub    = feat_df.loc[mask]
    signs  = {}
    for feat in features:
        y = sub[target_col].values
        x = sub[feat].values
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() < 50:
            signs[feat] = 1.0
            continue
        rho, _ = spearmanr(x[ok], y[ok])
        signs[feat] = float(np.sign(rho)) if not np.isnan(rho) else 1.0
    return pd.Series(signs)


# ── TA-FDR for one track ──────────────────────────────────────────────────────

def run_ta_fdr_track(
    track: str,
    target_col: str,
    features: list,
    feat_df: pd.DataFrame,
    ohlcv: pd.DataFrame,
    cfg: dict,
    B: int,
    rng: np.random.Generator,
    save_null: bool = False,
) -> tuple:
    """Returns (result_df, null_records).

    Permutation design: cross-sectional column shuffle of returns on each day.
    For each permutation b, randomly reassign which ticker's return is attributed
    to which ticker's position (permute columns of the returns matrix per row).
    This breaks the signal-return cross-sectional relationship while preserving:
      - The marginal return distribution of each ticker
      - The time-series autocorrelation structure
    Positions and costs are pre-computed once (they don't depend on realised
    returns), making each permutation a pure numpy operation.
    """
    log(f"\n  === Track: {track.upper()} ===")

    spread_bps       = cfg["spread_bps"]
    impact_coeff     = cfg["impact_coeff"]
    aum_dollars      = cfg.get("aum_dollars", 1e8)
    min_adv_dollars  = cfg.get("min_adv_dollars", 1e6)

    all_tickers = feat_df.index.get_level_values("ticker").unique().tolist()
    returns, sigma, adv = load_ohlcv_matrices(ohlcv, all_tickers, IS_START, IS_END)

    log(f"  Estimating IC signs on IS data …")
    ic_signs = estimate_ic_signs(feat_df, features, target_col, IS_START, IS_END)

    # --- Observed T_f + pre-compute positions/costs for all features ---
    log(f"  Computing observed T_f + pre-computing components for {len(features)} features …")
    t_obs  = {}
    comps  = {}   # feat → (pos_arr, ret_arr, costs_arr)

    for i, feat in enumerate(features):
        sig = build_single_feature_signal(feat_df, feat, ic_signs[feat], IS_START, IS_END)
        c = _precompute_pnl_components(
            sig, returns, sigma, adv,
            spread_bps=spread_bps, impact_coeff=impact_coeff,
            aum_dollars=aum_dollars, min_adv_dollars=min_adv_dollars,
        )
        if c is not None:
            pos_, ret_, costs_ = c
            comps[feat] = (pos_, ret_, costs_)
            net = (pos_ * ret_).sum(axis=1) - costs_
            t_obs[feat] = float(net.mean() / net.std() * np.sqrt(252)) \
                          if net.std() > 1e-10 else np.nan
        else:
            t_obs[feat] = np.nan
        if (i+1) % 5 == 0:
            log(f"    {i+1}/{len(features)} done")

    valid_t = [v for v in t_obs.values() if not np.isnan(v)]
    log(f"  Observed T_f range: [{min(valid_t):+.3f}, {max(valid_t):+.3f}]")

    # --- Permutation null: cross-sectional column shuffle per day (fast numpy) ---
    log(f"\n  Running {B} cross-section permutations …")
    null_dist    = {feat: [] for feat in features}
    null_records = []

    t_perm_start = time.time()
    for b in range(B):
        perm_row = {}
        for feat, (pos_, ret_, costs_) in comps.items():
            T, N = ret_.shape
            # Independently shuffle columns of ret_ for each row (date)
            rnd   = rng.random((T, N))
            cidx  = np.argsort(rnd, axis=1)          # (T × N) column permutation
            ridx  = np.arange(T)[:, np.newaxis]       # (T × 1) row broadcast
            ret_b = ret_[ridx, cidx]                  # shuffled returns
            net_b = (pos_ * ret_b).sum(axis=1) - costs_
            t_b   = float(net_b.mean() / net_b.std() * np.sqrt(252)) \
                    if net_b.std() > 1e-10 else np.nan
            null_dist[feat].append(t_b)
            perm_row[feat] = t_b

        for feat in features:
            if feat not in comps:
                null_dist[feat].append(np.nan)

        if save_null:
            null_records.append({"track": track, "perm": b, **perm_row})

        if (b+1) % 10 == 0:
            elapsed = time.time() - t_perm_start
            eta     = elapsed / (b+1) * (B - b - 1)
            log(f"    perm {b+1}/{B}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    # --- p-values and BH ---
    p_vals = {}
    for feat in features:
        null = np.array([v for v in null_dist[feat] if not np.isnan(v)])
        t_f  = t_obs[feat]
        if np.isnan(t_f) or len(null) == 0:
            p_vals[feat] = 1.0
        else:
            p_vals[feat] = (1 + (null >= t_f).sum()) / (1 + len(null))

    p_arr  = np.array([p_vals[f] for f in features])
    reject, adj_p = benjamini_hochberg(p_arr, q=FDR_Q)

    bh_rej = pd.read_parquet(FDR_PATH)
    bh_rej_set = set(
        bh_rej[(bh_rej["track"] == track) & bh_rej["rejected"]]["feature"].tolist()
    )

    result_df = pd.DataFrame({
        "track":             track,
        "feature":           features,
        "T_obs":             [t_obs[f] for f in features],
        "p_perm":            [p_vals[f] for f in features],
        "bh_adj_p":          adj_p,
        "ta_fdr_rejected":   reject,
        "bh_on_ic_rejected": [f in bh_rej_set for f in features],
    })

    n_ta = result_df["ta_fdr_rejected"].sum()
    n_bh = result_df["bh_on_ic_rejected"].sum()
    log(f"\n  TA-FDR rejected: {n_ta}/{len(features)}")
    log(f"  BH-on-IC rejected: {n_bh}/{len(features)}")

    ta_set   = set(result_df[result_df["ta_fdr_rejected"]]["feature"])
    only_ta  = sorted(ta_set - bh_rej_set)
    only_bh  = sorted(bh_rej_set - ta_set)
    in_both  = sorted(ta_set & bh_rej_set)
    log(f"\n  In both: {in_both}")
    log(f"  Only TA-FDR (tradable, cost-penalised): {only_ta}")
    log(f"  Only BH-on-IC (stat real, not tradable net-of-cost): {only_bh}")

    null_df = pd.DataFrame(null_records) if null_records else pd.DataFrame()
    return result_df, null_df


# ── OOS rerun comparison ──────────────────────────────────────────────────────

def run_oos_comparison(
    ta_fdr_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    ohlcv: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Compare OOS 2022-2024 performance: TA-FDR set vs static-BH set."""
    from src.backtest.generate_signals import build_signal_weights, generate_composite_signal
    fdr_df  = pd.read_parquet(FDR_PATH)
    shap_df = pd.read_parquet(SHAP_PATH)

    sim = PortfolioSimulator(
        config_path=str(CFG_PATH),
        spread_bps=cfg["spread_bps"],
        impact_coeff=cfg["impact_coeff"],
    )

    records = []

    for track in ["track_a", "track_b"]:
        sub_ta = ta_fdr_df[ta_fdr_df["track"] == track]

        for method, use_ta in [("static_bh", False), ("ta_fdr", True)]:
            if use_ta:
                # Build weights using TA-FDR feature set + IC signs from observed T_f
                ta_feats = sub_ta[sub_ta["ta_fdr_rejected"]]["feature"].tolist()
                if not ta_feats:
                    log(f"  [{track}] TA-FDR rejected 0 features — skipping OOS for ta_fdr")
                    continue
                # Use IC sign from bh_correction data
                bh_sub   = fdr_df[fdr_df["track"] == track].set_index("feature")
                shap_avg = shap_df[shap_df["track"] == track].groupby("feature")["mean_abs_shap"].mean()
                w = {}
                for f in ta_feats:
                    sign = float(np.sign(bh_sub.loc[f, "mean_ic"])) if f in bh_sub.index else 1.0
                    w[f] = sign * float(shap_avg.get(f, shap_avg.mean()))
                ws = pd.Series(w)
                ws = ws / ws.abs().sum()
                sig = generate_composite_signal(feat_df, ws, HOLDOUT_START, HOLDOUT_END)
            else:
                weights = build_signal_weights(track, fdr_df, shap_df)
                sig = generate_composite_signal(feat_df, weights, HOLDOUT_START, HOLDOUT_END)

            tickers = sig.columns.tolist()
            mask = (
                ohlcv["ticker"].isin(tickers)
                & (ohlcv["date"] >= pd.Timestamp(HOLDOUT_START))
                & (ohlcv["date"] <= pd.Timestamp(HOLDOUT_END))
            )
            sub_ohlcv = ohlcv[mask][["ticker","date","close","volume"]]
            close_w   = sub_ohlcv.pivot(index="date", columns="ticker", values="close")
            vol_w     = sub_ohlcv.pivot(index="date", columns="ticker", values="volume")
            returns   = close_w.pct_change()
            sigma     = returns.rolling(VOL_WINDOW, min_periods=10).std()
            adv       = (close_w * vol_w).rolling(VOL_WINDOW, min_periods=10).mean()
            ret_mask  = (returns.index >= pd.Timestamp(HOLDOUT_START)) & \
                        (returns.index <= pd.Timestamp(HOLDOUT_END))
            returns   = returns[ret_mask]
            sigma     = sigma[ret_mask]
            adv       = adv[ret_mask]

            positions = sim.signal_to_positions(sig, lag=1, rebal_freq=REBAL_FREQ)
            pnl_df    = sim.simulate_pnl(
                positions, returns, vol=sigma, adv_dollars=adv,
                aum_dollars=cfg.get("aum_dollars", 1e8),
                min_adv_dollars=cfg.get("min_adv_dollars", 1e6),
            )
            m = sim.compute_metrics(pnl_df)
            records.append({"track": track, "method": method, **m})

            log(f"  [{track}] {method}: net SR = {m.get('net_pnl_sharpe', float('nan')):+.3f}")

    return pd.DataFrame(records)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tracks", nargs="+", default=["track_b", "track_a"],
                        choices=["track_a", "track_b"])
    parser.add_argument("--save-null", action="store_true", default=False)
    args = parser.parse_args()

    log(f"=== Tradable-Alpha FDR (TA-FDR) — B={args.B}, block={BLOCK_LEN} ===\n")
    t0  = time.time()
    rng = np.random.default_rng(args.seed)

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    features = get_surviving_features(SHAP_PATH)
    log(f"  Features: {len(features)}")
    log(f"  IS period: {IS_START} → {IS_END}")
    log(f"  FDR q = {FDR_Q}")
    log("\nLoading data …")
    feat_df = pd.read_parquet(FEAT_PATH)
    ohlcv   = pd.read_parquet(OHLCV_PATH)

    track_targets = {"track_a": "target_track_a", "track_b": "target_track_b"}

    all_results = []
    all_nulls   = []

    for track in args.tracks:
        result_df, null_df = run_ta_fdr_track(
            track, track_targets[track],
            features, feat_df, ohlcv, cfg,
            B=args.B, rng=rng,
            save_null=args.save_null,
        )
        all_results.append(result_df)
        all_nulls.append(null_df)

    ta_fdr_df = pd.concat(all_results, ignore_index=True)
    ta_fdr_df.to_parquet(OUT_TAFDR, index=False)
    log(f"\nSaved → {OUT_TAFDR}")

    if args.save_null:
        null_full = pd.concat(all_nulls, ignore_index=True)
        null_full.to_parquet(OUT_NULL, index=False)
        log(f"Saved null distributions → {OUT_NULL}")

    log("\n=== OOS comparison: static-BH vs TA-FDR set ===")
    oos_df = run_oos_comparison(ta_fdr_df, feat_df, ohlcv, cfg)
    oos_df.to_parquet(OUT_OOS, index=False)
    log(f"Saved → {OUT_OOS}")

    log("\n=== SUMMARY ===")
    for track in args.tracks:
        sub = ta_fdr_df[ta_fdr_df["track"] == track]
        n_ta = sub["ta_fdr_rejected"].sum()
        n_bh = sub["bh_on_ic_rejected"].sum()
        log(f"  {track.upper()}: TA-FDR={n_ta}  BH-on-IC={n_bh}")

    log(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
