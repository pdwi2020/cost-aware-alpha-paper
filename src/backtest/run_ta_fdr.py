"""IA1+IA2 — Tradable-Alpha FDR (TA-FDR) with stationary-block-bootstrap null.

## Design (v2 — corrected statistic and joint null)

### Statistic
For each candidate feature f (IS 2013-2021 only):
  T_f = mean(net_return)  — the mean daily net return of the single-feature strategy
        run through a square-root participation cost model.
  Signal = sign(mean_IC_f) × feature_f values (cross-sectionally z-scored per date).

  Rationale: net = gross - cost with cost >= 0, so mean(net) <= mean(gross) holds
  POINTWISE.  This is the statistic for which the cost-aware conservativeness result
  is actually valid.  The Sharpe ratio is NOT valid here because its denominator
  changes under the null, which invalidates the comparison.

### Square-root participation cost model
  total impact ~ turnover^{3/2}
  (impact_coeff * sigma * sqrt(participation_fraction) * delta_abs)
  This is a square-root participation cost model, NOT the Almgren-Chriss (2001)
  linear transient-decay model.

### Joint null (stationary block bootstrap)
Repeat B times (default 1000):
  1. Draw ONE shared resample of date-block indices using the stationary block
     bootstrap (Politis-Romano; geometric block lengths with mean = block_length_days
     = 21 days).  Apply the SAME resampled date ordering to the RETURN/target panel
     for ALL features in that draw.
  2. For each feature, pair its ORIGINAL-ORDER position array with the BLOCK-
     RESAMPLED return array.  The feature/position panel keeps its original date
     order; pairing original-order signals with block-resampled returns breaks the
     predictive alignment while PRESERVING:
       (a) within-date cross-sectional dependence (whole columns/dates move together)
       (b) per-ticker autocorrelation (contiguous blocks maintain local time-series
           structure; a geometric-mean block length of 21 days is long relative to
           the 5-day return horizon)
  3. Recompute costs using the FIXED positions (turnover depends only on positions,
     not on realised returns), then compute net = pos * ret_resampled - costs.
  4. T_f^(b) = mean(net_b)   (same statistic as observed).

  The joint resampling ensures the null distribution properly reflects the
  cross-sectional dependence structure across features.

### p-value + BH + BHY
  p_f = (1 + #{T_f^(b) >= T_f}) / (1 + B)    (one-sided: high T_f is signal)
  Apply BH  at q=0.10 on {p_f} → BH TA-FDR accepted set.
  Apply BHY at q=0.10 on {p_f} → BHY TA-FDR accepted set (valid under arbitrary
  dependence, appropriate given the cross-sectional correlations among features).

### Prior-art delta (must cite in paper)
- Bajgrowicz & Scaillet (2012, JFE): rules / proportional cost / single time-series.
  Our delta: cross-sectional feature discovery / size-dependent participation cost /
  joint stationary block bootstrap null / regime-aware walk-forward protocol.
- alpha-investing (Foster & Stine 2008, JRSS-B): data-collection cost, not market
  impact.  Our delta: market-impact wealth penalty (the more you trade a feature,
  the more impact you pay, even under the null).

### OOS validation
After TA-FDR selection (IS only), rerun OOS 2022-2024 with the TA-FDR composite
and compare to the static-BH composite. Acceptance: static-BH OOS net SR = 0.50 +/-0.02.

Outputs:
    data/processed/ta_fdr.parquet             — per-feature statistics + rejection flags
    data/processed/ta_fdr_oos_metrics.parquet — OOS comparison: static-BH vs TA-FDR
    data/processed/ta_fdr_null_dist.parquet   — null distributions (optional)

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
from src.fdr.bh_correction import benjamini_hochberg, bhy_procedure
from src.features.feature_spec import feature_columns as get_feature_columns
from src.manifest import record as manifest_record

FDR_PATH   = ROOT / "data" / "processed" / "fdr_results.parquet"
SHAP_PATH  = ROOT / "data" / "processed" / "shap_summary.parquet"
FEAT_PATH  = ROOT / "data" / "processed" / "features_all.parquet"
OHLCV_PATH = ROOT / "data" / "processed" / "daily_ohlcv.parquet"
CFG_PATH   = ROOT / "configs" / "backtest.yaml"
SPEC_PATH  = ROOT / "config" / "spec.yaml"

OUT_TAFDR   = ROOT / "data" / "processed" / "ta_fdr.parquet"
OUT_OOS     = ROOT / "data" / "processed" / "ta_fdr_oos_metrics.parquet"
OUT_NULL    = ROOT / "data" / "processed" / "ta_fdr_null_dist.parquet"

IS_START   = "2013-01-01"
IS_END     = "2021-12-31"
HOLDOUT_START = "2022-01-01"
HOLDOUT_END   = "2024-12-31"
FDR_Q      = 0.10
REBAL_FREQ = 5
VOL_WINDOW = 21

# Read TA-FDR parameters from spec.yaml (authoritative)
with open(SPEC_PATH) as _f:
    _spec = yaml.safe_load(_f)
_ta_spec = _spec["ta_fdr"]
BLOCK_LEN  = int(_ta_spec.get("block_length_days", 21))   # stationary bootstrap mean block length
N_SAMPLES  = int(_ta_spec.get("n_samples", 1000))         # B permutation draws


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
    """(date x ticker) signal matrix — vectorized pivot, no Python date loop."""
    dates = feat_df.index.get_level_values("date")
    mask  = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    sub   = feat_df.loc[mask, [feature]]
    sig = sub[feature].unstack(level="ticker")  # date x ticker
    # Vectorized cross-sectional median fill (no Python row loop)
    arr = sig.values.copy()
    row_med = np.nanmedian(arr, axis=1, keepdims=True)
    nan_mask = np.isnan(arr)
    arr[nan_mask] = np.broadcast_to(row_med, arr.shape)[nan_mask]
    sig = pd.DataFrame(arr, index=sig.index, columns=sig.columns)
    return (ic_sign * sig).dropna(how="all")


def _compute_mean_net_return(
    pos_: np.ndarray,
    ret_: np.ndarray,
    costs_: np.ndarray,
) -> float:
    """Statistic T_f = mean(net_return) where net = gross - costs.

    mean(net) <= mean(gross) holds POINTWISE since costs >= 0.
    This is the statistic for which the cost-aware conservativeness result
    is valid (unlike the Sharpe ratio whose denominator changes).
    """
    net = (pos_ * ret_).sum(axis=1) - costs_
    return float(net.mean())


# ── Pre-compute positions + costs (called once per feature, not per draw) ─────

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
    """Return (pos_arr, ret_arr, costs_arr) as numpy arrays aligned on
    common (dates x tickers).

    Costs are computed from positions (turnover) and depend on sigma and
    adv but NOT on realised returns, so they can be precomputed once and
    reused across all bootstrap draws.  The joint null holds positions
    and costs fixed; only ret_arr is replaced by resampled returns.
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
    # Align EVERY feature to a FIXED common (date x ticker) grid (the IS trading
    # days present in `returns`), so the joint bootstrap's single shared date
    # index applies identically across all features. A feature contributes zero
    # position on dates/tickers where it has no signal. Without this, features
    # have different T and the shared date index goes out of bounds.
    grid_dates = returns.index[(returns.index >= pd.Timestamp(IS_START)) &
                               (returns.index <= pd.Timestamp(IS_END))]
    grid_tickers = returns.columns

    pos_ = pos.reindex(index=grid_dates, columns=grid_tickers).fillna(0.0).values
    ret_ = returns.reindex(index=grid_dates, columns=grid_tickers).fillna(0.0).values
    sig_ = sigma.reindex(index=grid_dates, columns=grid_tickers).fillna(0.02).values
    adv_ = adv.reindex(index=grid_dates, columns=grid_tickers).fillna(aum_dollars).values

    liquid = (adv_ >= min_adv_dollars).astype(float)
    pos_  *= liquid

    # Square-root participation cost model: total impact ~ turnover^{3/2}
    delta_abs = np.abs(np.diff(np.vstack([np.zeros((1, pos_.shape[1])), pos_]), axis=0))
    spread_cost = delta_abs.sum(axis=1) * (spread_bps / 10_000 / 2)
    part        = (delta_abs * aum_dollars) / np.clip(adv_, 1.0, None)
    impact_cost = (impact_coeff * sig_ * np.sqrt(part) * delta_abs).sum(axis=1)
    costs = spread_cost + impact_cost

    return pos_, ret_, costs


# ── Stationary block bootstrap ────────────────────────────────────────────────

def _draw_stationary_block_bootstrap_indices(
    T: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one resample of T indices using the stationary block bootstrap.

    Politis & Romano (1994): block lengths are geometrically distributed with
    mean = block_length.  Start positions are drawn uniformly from [0, T).
    The resample wraps around (circular) so every position is equally likely.

    Parameters
    ----------
    T            : number of time steps in the original series.
    block_length : mean block length (geometric distribution parameter).
    rng          : numpy random Generator (for reproducibility).

    Returns
    -------
    idx : np.ndarray of shape (T,), dtype int — resampled time indices.
    """
    # Geometric block lengths: P(L = k) = p * (1-p)^{k-1}, mean = 1/p
    p = 1.0 / block_length
    idx = np.empty(T, dtype=np.intp)
    i = 0
    while i < T:
        # Start of block: random position in [0, T) with wrap
        start = int(rng.integers(0, T))
        # Block length drawn from Geometric(p) — at least 1
        length = int(np.ceil(-np.log(rng.random()) / (-np.log(1.0 - p + 1e-12))))
        length = min(length, T - i)
        for j in range(length):
            idx[i] = (start + j) % T
            i += 1
            if i >= T:
                break
    return idx


# ── IC sign estimation ────────────────────────────────────────────────────────

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
    block_length: int = BLOCK_LEN,
    save_null: bool = False,
) -> tuple:
    """Returns (result_df, null_records).

    Joint null design (stationary block bootstrap):
    --------------------------------------------------
    For each draw b in 1..B:
      1. Draw ONE shared date-index resample via the stationary block bootstrap
         with geometric mean block length = block_length days.
      2. Apply the SAME resampled date ordering to the return matrix for ALL
         features.  The feature/position arrays keep their original date order.
         Pairing original-order signals with block-resampled returns breaks the
         predictive relationship while preserving:
           (a) cross-sectional dependence: entire columns (dates) move together
           (b) per-ticker autocorrelation: contiguous blocks maintain local
               time-series structure
      3. Recompute net = pos * resampled_ret - costs.  Costs are fixed because
         turnover depends only on positions, not on realised returns.
      4. T_f^(b) = mean(net_b)  — the same statistic as observed.
    """
    log(f"\n  === Track: {track.upper()} ===")

    spread_bps       = cfg["spread_bps"]
    impact_coeff     = cfg["impact_coeff"]
    aum_dollars      = cfg.get("aum_dollars", 1e8)
    min_adv_dollars  = cfg.get("min_adv_dollars", 1e6)

    all_tickers = feat_df.index.get_level_values("ticker").unique().tolist()
    returns, sigma, adv = load_ohlcv_matrices(ohlcv, all_tickers, IS_START, IS_END)

    log(f"  Estimating IC signs on IS data ...")
    ic_signs = estimate_ic_signs(feat_df, features, target_col, IS_START, IS_END)

    # --- Observed T_f (mean net return) + pre-compute positions/costs ---
    log(f"  Computing observed T_f (mean net return) + pre-computing components "
        f"for {len(features)} features ...")
    t_obs  = {}
    comps  = {}   # feat -> (pos_arr, ret_arr, costs_arr)

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
            # T_f = mean net return (NOT Sharpe — see module docstring)
            t_obs[feat] = _compute_mean_net_return(pos_, ret_, costs_)
        else:
            t_obs[feat] = np.nan
        if (i+1) % 5 == 0:
            log(f"    {i+1}/{len(features)} done")

    valid_t = [v for v in t_obs.values() if not np.isnan(v)]
    if valid_t:
        log(f"  Observed T_f (mean net return) range: "
            f"[{min(valid_t):+.6f}, {max(valid_t):+.6f}]")

    # --- Joint null: stationary block bootstrap on the returns panel ---
    log(f"\n  Running {B} joint stationary-block-bootstrap draws "
        f"(block_length={block_length}) ...")
    null_dist    = {feat: [] for feat in features}
    null_records = []

    # Determine T from the first available component's returns array
    T_common = None
    for feat in features:
        if feat in comps:
            T_common = comps[feat][1].shape[0]
            break

    if T_common is None:
        log("  WARNING: No valid components — skipping null.")
        T_common = 0

    t_perm_start = time.time()
    for b in range(B):
        # Draw ONE shared date resample for this draw (joint across all features)
        date_idx = _draw_stationary_block_bootstrap_indices(T_common, block_length, rng)

        perm_row = {}
        for feat, (pos_, ret_, costs_) in comps.items():
            # Apply the SAME resampled date ordering to the return array
            # Positions and costs remain in their original (unchanged) date order
            ret_b = ret_[date_idx, :]           # (T x N) resampled returns
            # Costs are fixed (depend on position turnover, not realised returns)
            net_b = (pos_ * ret_b).sum(axis=1) - costs_
            t_b   = float(net_b.mean())
            null_dist[feat].append(t_b)
            perm_row[feat] = t_b

        for feat in features:
            if feat not in comps:
                null_dist[feat].append(np.nan)

        if save_null:
            null_records.append({"track": track, "perm": b, **perm_row})

        if (b+1) % 100 == 0:
            elapsed = time.time() - t_perm_start
            eta     = elapsed / (b+1) * (B - b - 1)
            log(f"    draw {b+1}/{B}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    # --- p-values, BH, and BHY ---
    p_vals = {}
    for feat in features:
        null = np.array([v for v in null_dist[feat] if not np.isnan(v)])
        t_f  = t_obs[feat]
        if np.isnan(t_f) or len(null) == 0:
            p_vals[feat] = 1.0
        else:
            p_vals[feat] = (1 + (null >= t_f).sum()) / (1 + len(null))

    p_arr  = np.array([p_vals[f] for f in features])
    bh_reject, bh_adj_p   = benjamini_hochberg(p_arr, q=FDR_Q)
    bhy_reject, bhy_adj_p = bhy_procedure(p_arr, q=FDR_Q)

    # Load IC-based BH results for comparison (may not exist in test env)
    bh_rej_set = set()
    if FDR_PATH.exists():
        bh_rej = pd.read_parquet(FDR_PATH)
        bh_rej_set = set(
            bh_rej[(bh_rej["track"] == track) & bh_rej["bh_rejected"]]["feature"].tolist()
        )

    result_df = pd.DataFrame({
        "track":             track,
        "feature":           features,
        "T_obs":             [t_obs[f] for f in features],
        "p_perm":            [p_vals[f] for f in features],
        "bh_adj_p":          bh_adj_p,
        "bhy_adj_p":         bhy_adj_p,
        "bh_selected":       bh_reject,
        "bhy_selected":      bhy_reject,
        # Legacy column name kept for downstream compatibility
        "ta_fdr_rejected":   bh_reject,
        "bh_on_ic_rejected": [f in bh_rej_set for f in features],
    })

    n_bh  = result_df["bh_selected"].sum()
    n_bhy = result_df["bhy_selected"].sum()
    n_bh_ic = result_df["bh_on_ic_rejected"].sum()
    log(f"\n  BH TA-FDR selected:  {n_bh}/{len(features)}")
    log(f"  BHY TA-FDR selected: {n_bhy}/{len(features)}")
    log(f"  BH-on-IC selected:   {n_bh_ic}/{len(features)}")

    bh_set  = set(result_df[result_df["bh_selected"]]["feature"])
    bhy_set = set(result_df[result_df["bhy_selected"]]["feature"])
    only_bh  = sorted(bh_set - bh_rej_set)
    only_bh_ic = sorted(bh_rej_set - bh_set)
    in_both  = sorted(bh_set & bh_rej_set)
    log(f"\n  In both BH-TA-FDR and BH-on-IC: {in_both}")
    log(f"  Only BH-TA-FDR (tradable, cost-penalised): {only_bh}")
    log(f"  Only BH-on-IC (stat signal, not tradable net-of-cost): {only_bh_ic}")
    log(f"  BHY TA-FDR set: {sorted(bhy_set)}")

    # --- Record to manifest ---
    _record_to_manifest(track, features, t_obs, p_vals, bh_reject, bhy_reject,
                        B, block_length)

    null_df = pd.DataFrame(null_records) if null_records else pd.DataFrame()
    return result_df, null_df


def _record_to_manifest(
    track: str,
    features: list,
    t_obs: dict,
    p_vals: dict,
    bh_reject: np.ndarray,
    bhy_reject: np.ndarray,
    B: int,
    block_length: int,
) -> None:
    """Record per-feature stats and joint-null parameters to the manifest."""
    try:
        # Record joint-null parameters (once per track)
        manifest_record(
            f"ta_fdr.{track}.null_params",
            {"null": "stationary_block_bootstrap_JOINT",
             "block_length_days": block_length,
             "n_samples": B,
             "statistic": "mean_net_return",
             "preserves": ["ticker_autocorrelation", "cross_sectional_dependence"]},
            stage="ta_fdr",
            track=track,
            meta={"description": "Joint null parameters for TA-FDR permutation test"},
        )

        # Record per-feature statistics
        per_feature = []
        for i, feat in enumerate(features):
            per_feature.append({
                "feature":      feat,
                "t_obs":        float(t_obs[feat]) if not np.isnan(t_obs[feat]) else None,
                "p_value":      float(p_vals[feat]),
                "bh_selected":  bool(bh_reject[i]),
                "bhy_selected": bool(bhy_reject[i]),
            })

        manifest_record(
            f"ta_fdr.{track}.per_feature",
            per_feature,
            stage="ta_fdr",
            track=track,
            meta={"statistic": "mean_net_return",
                  "null": "stationary_block_bootstrap_JOINT"},
        )

        n_bh  = int(bh_reject.sum())
        n_bhy = int(bhy_reject.sum())
        manifest_record(
            f"ta_fdr.{track}.summary",
            {"n_features": len(features),
             "n_bh_selected": n_bh,
             "n_bhy_selected": n_bhy},
            stage="ta_fdr",
            track=track,
        )
    except Exception as e:
        log(f"  WARNING: manifest record failed: {e}")


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
                # Build weights using BH TA-FDR feature set + IC signs from observed T_f
                ta_feats = sub_ta[sub_ta["bh_selected"]]["feature"].tolist()
                if not ta_feats:
                    log(f"  [{track}] TA-FDR selected 0 features — skipping OOS for ta_fdr")
                    continue
                bh_sub   = fdr_df[fdr_df["track"] == track].set_index("feature")
                shap_avg = shap_df[shap_df["track"] == track].groupby("feature")["mean_abs_shap"].mean()
                w = {}
                for f in ta_feats:
                    sign = float(np.sign(bh_sub.loc[f, "ic_bar"])) if f in bh_sub.index else 1.0
                    w[f] = sign * float(shap_avg.get(f, shap_avg.mean()))
                ws = pd.Series(w)
                ws = ws / ws.abs().sum()
                sig = generate_composite_signal(feat_df, ws, HOLDOUT_START, HOLDOUT_END)
            else:
                try:
                    weights = build_signal_weights(track, fdr_df, shap_df)
                except ValueError:
                    log(f"  [{track}] no BH-selected features — skipping OOS for static_bh")
                    continue
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
    parser.add_argument("--B", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-length", type=int, default=BLOCK_LEN,
                        help="Mean block length for stationary bootstrap (default from spec.yaml)")
    parser.add_argument("--tracks", nargs="+", default=["track_b", "track_a"],
                        choices=["track_a", "track_b"])
    parser.add_argument("--save-null", action="store_true", default=False)
    args = parser.parse_args()

    log(f"=== Tradable-Alpha FDR (TA-FDR) — B={args.B}, "
        f"block_length={args.block_length} (stationary bootstrap) ===\n")
    t0  = time.time()
    rng = np.random.default_rng(args.seed)

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    log("\nLoading feature data to determine feature set ...")
    feat_df = pd.read_parquet(FEAT_PATH)

    # Use feature_spec.feature_columns — canonical pre-registered set
    # (drops dropped_broadcast and dropped_duplicates; includes KEPT + ADDED_INTERACTIONS)
    features = get_feature_columns(feat_df)
    log(f"  Features from feature_spec: {len(features)}")
    log(f"  IS period: {IS_START} -> {IS_END}")
    log(f"  FDR q = {FDR_Q}")

    log("\nLoading OHLCV data ...")
    ohlcv = pd.read_parquet(OHLCV_PATH)

    track_targets = {"track_a": "target_track_a", "track_b": "target_track_b"}

    all_results = []
    all_nulls   = []

    for track in args.tracks:
        result_df, null_df = run_ta_fdr_track(
            track, track_targets[track],
            features, feat_df, ohlcv, cfg,
            B=args.B, rng=rng,
            block_length=args.block_length,
            save_null=args.save_null,
        )
        all_results.append(result_df)
        all_nulls.append(null_df)

    ta_fdr_df = pd.concat(all_results, ignore_index=True)
    ta_fdr_df.to_parquet(OUT_TAFDR, index=False)
    log(f"\nSaved -> {OUT_TAFDR}")

    if args.save_null:
        null_full = pd.concat(all_nulls, ignore_index=True)
        null_full.to_parquet(OUT_NULL, index=False)
        log(f"Saved null distributions -> {OUT_NULL}")

    log("\n=== OOS comparison: static-BH vs TA-FDR set ===")
    oos_df = run_oos_comparison(ta_fdr_df, feat_df, ohlcv, cfg)
    oos_df.to_parquet(OUT_OOS, index=False)
    log(f"Saved -> {OUT_OOS}")

    log("\n=== SUMMARY ===")
    for track in args.tracks:
        sub = ta_fdr_df[ta_fdr_df["track"] == track]
        n_bh  = sub["bh_selected"].sum()
        n_bhy = sub["bhy_selected"].sum()
        n_ic  = sub["bh_on_ic_rejected"].sum()
        log(f"  {track.upper()}: BH-TA-FDR={n_bh}  BHY-TA-FDR={n_bhy}  BH-on-IC={n_ic}")

    log(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
