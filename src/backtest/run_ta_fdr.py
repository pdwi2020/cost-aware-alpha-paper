"""IA1+IA2 — Tradable-Alpha FDR (TA-FDR) with a recentred joint bootstrap null.

## Design (v3 — cost-aware net p-values via books.single_feature_book)

### Why v2 was invalid
The previous ("v2") design paired ORIGINAL-order positions with BLOCK-RESAMPLED
returns and subtracted the SAME FIXED cost vector from both the observed
statistic and every bootstrap draw:
    net_b = (pos_ * ret_b).sum(axis=1) - costs_
with `costs_` identical across draws and identical between the observed
computation and the null. Because the identical `costs_` is subtracted on
both sides of the `null >= t_obs` comparison, the comparison — and hence the
p-value — is invariant to the magnitude of `costs_`: v2's p-value was
mathematically a gross-return p-value in disguise, no matter how large costs
were (see `TestCostMonotonicity.test_old_design_p_is_cost_invariant` in
tests/test_ta_fdr_null.py for a machine-checked demonstration). v2 also built
books from raw `close.pct_change()` (skipping Screen 0's +/-50% return
winsorisation) and charged `spread_bps/2` per unit turnover, half of what the
deployed `PortfolioSimulator.simulate_pnl` actually charges.

v3 fixes all three problems by routing every book through
`src.backtest.books.single_feature_book`, which applies Screen 0
(`mask_signal_screen0` + `apply_s0_daily_exit`), the sanitised/clipped return
panel (`books.load_market_panel` -> `run_backtest.build_returns_vol_adv`), and
the deployed simulator's cost convention unmodified — and by resampling the
feature's own daily GROSS P&L series (positions and returns already paired at
each date) instead of pairing original-order positions with resampled
returns.

### Hypotheses & statistic
Per feature f (IS 2013-01-01..2021-12-31 only):
    H0_f: E[r_net,f] <= 0   vs   H1_f: E[r_net,f] > 0
where r_net,f,t = g_f,t - c_f,t is feature f's single-feature book's daily net
return from `books.single_feature_book` (weekly rebalance, REBAL_FREQ=5).
Statistic: T_f = mean_t r_net,f,t = mean(g_f) - mean(c_f).

### Orientation sign
sign_f = sign(ic_bar) read from data/processed/fdr_results.parquet for the
matching track/feature row; +1.0 if the feature is missing from that track's
rows, or ic_bar is 0/NaN (see `orientation_signs`).

### Joint null — recentred stationary block bootstrap
For each track, ONE shared (B, T) index matrix is drawn via
`_draw_stationary_block_bootstrap_indices` (Politis-Romano; geometric block
lengths, mean = block_length_days from config/spec.yaml, default 21) — the
SAME index matrix is reused across every feature, which is what makes the
null "joint": cross-feature dependence in the resampled panel is preserved
because every feature is resampled along the identical date permutation for a
given draw b.

For each feature, vectorised over draws (no Python loop over b):
    resampled_b = g_f[idx_matrix]                        # (B, T)
    D_b         = resampled_b.mean(axis=1) - mean(g_f)    # (B,) recentred null

    p_net   = (1 + #{b: D_b >= T_f})        / (1 + B)
    p_gross = (1 + #{b: D_b >= mean(g_f)})  / (1 + B)      # SAME D_b, gross bar

Since T_f <= mean(g_f) whenever costs >= 0, and the exceedance count is
non-increasing in the threshold, p_net >= p_gross ALWAYS. This is exactly
`pval_antitone` / `tafdr_conservative` from formal/CavalFormal/TAFdr.lean, and
is asserted in `compute_bootstrap_pvalues` on every feature.

### Studentised bootstrap-t (robustness)
t_f = T_f / se_f, se_f = bootstrap sd of {D_b}. A full double (nested)
bootstrap is too slow for B up to 2000 x 30 features x 2 tracks, so the
per-draw studentising denominator se_f^(b) uses a closed-form stationary-
bootstrap HAC-type (Newey-West, Bartlett kernel, maxlags=block_length)
standard error of the mean, computed directly on each draw's resampled series
`resampled_b[b, :]` (no further resampling) — see `_bartlett_hac_var_of_mean`.
    t_b    = D_b / se_f^(b)
    p_stud = (1 + #{b: t_b >= t_f}) / (1 + B)

### BH / BHY
Applied at q=0.10 (FDR_Q) separately per track and separately on {p_net},
{p_gross}, {p_stud} (six reject sets total per track).

### Outputs
    data/processed/ta_fdr_v3.parquet             — per track x feature statistics
    data/processed/ta_fdr_v3_null_summary.parquet — per track x feature null mean/sd
    results/staging/ta_fdr_v3_summary.json        — run metadata + rejection counts
The v2 outputs are renamed (not deleted):
    data/processed/ta_fdr.parquet             -> ta_fdr.parquet.v2_invalid_null
    data/processed/ta_fdr_oos_metrics.parquet -> ta_fdr_oos_metrics.parquet.v2_invalid_null
v3 records every quantity the manuscript's TA-FDR table prints (see
`record_manifest`), overwriting the superseded v2 entries, and does not
evaluate anything after 2021-12-31 (the old OOS 2022-2024 comparison is
removed, out of scope here).

Run (slow — B=2000 default, ~30-90 min):
    python3 -u src/backtest/run_ta_fdr.py --B 2000

For a fast smoke test:
    python3 -u src/backtest/run_ta_fdr.py --B 20 --tracks track_b
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import books
from src.backtest.portfolio import PortfolioSimulator
from src.fdr.bh_correction import benjamini_hochberg, bhy_procedure
from src.features.feature_spec import feature_columns as get_feature_columns
from src.manifest import record

FDR_PATH   = ROOT / "data" / "processed" / "fdr_results.parquet"
FEAT_PATH  = ROOT / "data" / "processed" / "features_all.parquet"
OHLCV_PATH = ROOT / "data" / "processed" / "daily_ohlcv_v3.parquet"
CFG_PATH   = ROOT / "configs" / "backtest.yaml"
SPEC_PATH  = ROOT / "config" / "spec.yaml"

# v2 (invalid-null) outputs — renamed, not deleted, at the start of main().
OLD_TAFDR_PATH = ROOT / "data" / "processed" / "ta_fdr.parquet"
OLD_OOS_PATH   = ROOT / "data" / "processed" / "ta_fdr_oos_metrics.parquet"

# v3 outputs.
OUT_TAFDR_V3      = ROOT / "data" / "processed" / "ta_fdr_v3.parquet"
OUT_NULL_V3       = ROOT / "data" / "processed" / "ta_fdr_v3_null_summary.parquet"
OUT_SUMMARY_JSON  = ROOT / "results" / "staging" / "ta_fdr_v3_summary.json"

IS_START   = "2013-01-01"
IS_END     = "2021-12-31"
FDR_Q      = 0.10
REBAL_FREQ = 5
VOL_WINDOW = 21

# Read TA-FDR parameters from spec.yaml (authoritative)
with open(SPEC_PATH) as _f:
    _spec = yaml.safe_load(_f)
_ta_spec = _spec["ta_fdr"]
BLOCK_LEN  = int(_ta_spec.get("block_length_days", 21))   # stationary bootstrap mean block length
N_SAMPLES  = int(_ta_spec.get("n_samples", 1000))         # B bootstrap draws


def log(msg): print(msg, flush=True)


# ═════════════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY LAYER — DO NOT MODIFY OR REMOVE
#
# src/fdr/run_pbo_deployed.py imports IS_START, IS_END, REBAL_FREQ,
# VOL_WINDOW, _precompute_pnl_components, build_single_feature_signal, and
# load_ohlcv_matrices directly and depends on their EXACT current behavior
# (including the pre-v3 spread/cost/Screen-0 conventions). src/knockoffs/
# run_gaussian_knockoffs.py and run_cost_aware_knockoffs.py additionally
# import estimate_ic_signs and load_ohlcv_matrices (those two files are
# currently broken for an unrelated reason — they also import a
# `_fast_net_sharpe` name that has never existed in this module — which is a
# pre-existing state on this branch, not caused by this change; out of scope
# here). A later task will migrate these callers onto src.backtest.books and
# retire this layer. Until then it must stay byte-for-byte identical to the
# v2 implementation.
# ═════════════════════════════════════════════════════════════════════════

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
    reused across all bootstrap draws. Legacy (v2) cost convention — see
    module-level LEGACY COMPATIBILITY LAYER note.
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


def _draw_stationary_block_bootstrap_indices(
    T: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one resample of T indices using the stationary block bootstrap.

    Politis & Romano (1994): block lengths are geometrically distributed with
    mean = block_length. Start positions are drawn uniformly from [0, T).
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


# ═════════════════════════════════════════════════════════════════════════
# v3 — recentred joint bootstrap
# ═════════════════════════════════════════════════════════════════════════

def orientation_signs(fdr_df: pd.DataFrame, track: str, features: list) -> pd.Series:
    """sign(ic_bar) per feature for `track`, from fdr_results.parquet.

    +1.0 if the feature is missing from that track's rows, or ic_bar is 0/NaN.
    """
    sub = fdr_df.loc[fdr_df["track"] == track].set_index("feature")["ic_bar"]
    signs = {}
    for feat in features:
        if feat not in sub.index:
            signs[feat] = 1.0
            continue
        ic_bar = sub.loc[feat]
        if pd.isna(ic_bar) or ic_bar == 0:
            signs[feat] = 1.0
        else:
            signs[feat] = float(np.sign(ic_bar))
    return pd.Series(signs)


def _bartlett_hac_var_of_mean(x: np.ndarray, maxlags: int) -> np.ndarray:
    """Newey-West (Bartlett-kernel) HAC variance of the sample mean.

    Operates along the LAST axis: `x` may be 1D (T,) -> returns a 0-d array
    (usable as a scalar), or 2D (B, T) -> returns a (B,) array (one HAC
    variance per row). Shared by `_newey_west_t` (single observed series) and
    `compute_bootstrap_pvalues` (batched over every bootstrap replicate, as a
    fast closed-form stand-in for a per-draw nested/double bootstrap).
    """
    x = np.asarray(x, dtype=float)
    T = x.shape[-1]
    demeaned = x - x.mean(axis=-1, keepdims=True)
    L = min(maxlags, T - 1)
    total = np.mean(demeaned ** 2, axis=-1)
    for k in range(1, L + 1):
        w = 1.0 - k / (maxlags + 1)          # Bartlett kernel weight
        gamma_k = np.mean(demeaned[..., :-k] * demeaned[..., k:], axis=-1)
        total = total + 2.0 * w * gamma_k
    return np.maximum(total, 1e-300) / T


def _newey_west_t(x: np.ndarray, maxlags: int = 21) -> float:
    """HAC t-statistic for H0: mean(x) = 0 (Bartlett kernel, fixed maxlags —
    default 21 per spec: 'nw_t_net (Newey-West, 21 lags)')."""
    x = np.asarray(x, dtype=float)
    se = float(np.sqrt(_bartlett_hac_var_of_mean(x, maxlags)))
    if se < 1e-15:
        return 0.0
    return float(x.mean() / se)


def compute_bootstrap_pvalues(
    gross_pnl: dict,
    total_cost: dict,
    B: int,
    block_length: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Joint stationary-block-bootstrap null for TA-FDR v3, decoupled from I/O.

    For each feature f with daily gross P&L g_f (T,) and daily total cost c_f
    (T,):
        T_obs      = mean(g_f) - mean(c_f)          # observed net statistic
        mean_gross = mean(g_f)
        mean_cost  = mean(c_f)

    ONE shared (B, T) index matrix is drawn via
    `_draw_stationary_block_bootstrap_indices` (called B times; the SAME
    index matrix is reused across every feature passed in this call, which is
    what makes the null "joint": cross-feature dependence in the resampled
    panel is preserved because every feature is resampled along the identical
    date permutation for a given draw b).

    For each feature, vectorised over draws (no Python loop over b):
        resampled_b = g_f[idx_matrix]                       # (B, T)
        D_b         = resampled_b.mean(axis=1) - mean_gross  # (B,) recentred

        p_net   = (1 + #{b: D_b >= T_obs})        / (1 + B)
        p_gross = (1 + #{b: D_b >= mean_gross})   / (1 + B)   # same D_b draws

    Since T_obs <= mean_gross (cost >= 0) and the exceedance count is
    non-increasing in the threshold, p_net >= p_gross ALWAYS — this is exactly
    `pval_antitone` / `tafdr_conservative` from formal/CavalFormal/TAFdr.lean.
    Asserted (raises AssertionError, naming the feature and both p-values, if
    ever violated) for every feature before returning.

    Studentised bootstrap-t robustness check:
        se_boot = std(D_b, ddof=1)     # bootstrap sd of D — if < 1e-15, set
                                        # t_stat=0.0, p_stud=1.0 and skip
        t_stat  = T_obs / se_boot
        Per-draw denominator se_boot^(b): a proper double (nested) bootstrap
        is too slow for B up to 2000 x 30 features x 2 tracks, so this uses a
        closed-form stationary-bootstrap HAC-type (Newey-West, Bartlett
        kernel, maxlags=block_length) standard error of the mean, computed
        directly on each draw's resampled series `resampled_b[b, :]` (no
        further resampling) — see `_bartlett_hac_var_of_mean`, vectorised
        over all B draws at once.
        t_b    = D_b / max(se_boot^(b), 1e-12)
        p_stud = (1 + #{b: t_b >= t_stat}) / (1 + B)

    Returns
    -------
    pd.DataFrame indexed by feature (same key order as `gross_pnl`) with
    columns: T_obs, mean_gross, mean_cost, p_net, p_gross, se_boot, t_stat,
    p_stud, null_mean_D, null_sd_D.

    Raises
    ------
    ValueError
        If `gross_pnl` and `total_cost` do not share the same feature keys,
        or if the per-feature arrays are not all the same length T.
    """
    features = list(gross_pnl.keys())
    if set(total_cost.keys()) != set(gross_pnl.keys()):
        raise ValueError("gross_pnl and total_cost must have the same feature keys")

    lengths = {len(np.asarray(v)) for v in gross_pnl.values()}
    if len(lengths) != 1:
        raise ValueError(f"All gross_pnl arrays must share one length T; got {lengths}")
    T = lengths.pop()
    for feat in features:
        if len(np.asarray(total_cost[feat])) != T:
            raise ValueError(
                f"total_cost[{feat!r}] length {len(total_cost[feat])} != T={T}"
            )

    idx_matrix = np.empty((B, T), dtype=np.intp)
    for b in range(B):
        idx_matrix[b] = _draw_stationary_block_bootstrap_indices(T, block_length, rng)

    rows = {}
    for feat in features:
        g = np.asarray(gross_pnl[feat], dtype=float)
        c = np.asarray(total_cost[feat], dtype=float)
        mean_gross = float(g.mean())
        mean_cost  = float(c.mean())
        T_obs      = mean_gross - mean_cost

        resampled = g[idx_matrix]                        # (B, T)
        D = resampled.mean(axis=1) - mean_gross           # (B,) recentred null

        p_net   = (1 + int(np.sum(D >= T_obs)))   / (1 + B)
        p_gross = (1 + int(np.sum(D >= mean_gross))) / (1 + B)
        if p_net < p_gross - 1e-9:
            raise AssertionError(
                f"p_net ({p_net:.6f}) < p_gross ({p_gross:.6f}) for feature "
                f"{feat!r}; this violates pval_antitone / tafdr_conservative "
                "(formal/CavalFormal/TAFdr.lean) and must never happen when "
                "T_obs <= mean_gross (costs >= 0)."
            )

        se_boot = float(D.std(ddof=1)) if B > 1 else 0.0
        if se_boot < 1e-15:
            t_stat = 0.0
            p_stud = 1.0
        else:
            t_stat = T_obs / se_boot
            se_b = np.sqrt(_bartlett_hac_var_of_mean(resampled, block_length))  # (B,)
            t_b  = D / np.maximum(se_b, 1e-12)
            p_stud = (1 + int(np.sum(t_b >= t_stat))) / (1 + B)

        rows[feat] = {
            "T_obs":       T_obs,
            "mean_gross":  mean_gross,
            "mean_cost":   mean_cost,
            "p_net":       p_net,
            "p_gross":     p_gross,
            "se_boot":     se_boot,
            "t_stat":      t_stat,
            "p_stud":      p_stud,
            "null_mean_D": float(D.mean()),
            "null_sd_D":   float(D.std(ddof=1)) if B > 1 else 0.0,
        }

    return pd.DataFrame.from_dict(rows, orient="index")[[
        "T_obs", "mean_gross", "mean_cost", "p_net", "p_gross",
        "se_boot", "t_stat", "p_stud", "null_mean_D", "null_sd_D",
    ]]


def run_ta_fdr_v3_track(
    track: str,
    features: list,
    feat_df: pd.DataFrame,
    fdr_df: pd.DataFrame,
    ohlcv: pd.DataFrame,
    cfg: dict,
    sim: PortfolioSimulator,
    B: int,
    block_length: int,
    rng: np.random.Generator,
) -> tuple:
    """Run TA-FDR v3 for one track. Returns (result_df, null_summary_df)."""
    log(f"\n  === Track: {track.upper()} ===")

    all_tickers = feat_df.index.get_level_values("ticker").unique().tolist()
    panel = books.load_market_panel(ohlcv, all_tickers, IS_START, IS_END)
    signs = orientation_signs(fdr_df, track, features)

    log(f"  Building {len(features)} single-feature books (rebal_freq={REBAL_FREQ}) ...")
    book_dict = {}
    for i, feat in enumerate(features):
        book_dict[feat] = books.single_feature_book(
            feat_df, feat, signs[feat], panel, sim,
            rebal_freq=REBAL_FREQ, start=IS_START, end=IS_END,
        )
        if (i + 1) % 5 == 0:
            log(f"    {i+1}/{len(features)} books built")

    first_feat = features[0]
    first_index = book_dict[first_feat].index
    for feat, book in book_dict.items():
        if not book.index.equals(first_index):
            raise AssertionError(
                f"Feature {feat!r}'s book date index ({len(book)} rows) does "
                f"not match {first_feat!r}'s shared grid ({len(first_index)} "
                "rows) — the joint bootstrap requires every feature to share "
                "one date index."
            )
    T = len(first_index)
    log(f"  Shared date grid: {T} trading days ({IS_START} -> {IS_END})")

    gross_pnl  = {f: book_dict[f]["gross_pnl"].to_numpy()  for f in features}
    total_cost = {f: book_dict[f]["total_cost"].to_numpy() for f in features}

    log(f"  Running {B} joint stationary-block-bootstrap draws "
        f"(block_length={block_length}) ...")
    t_boot = time.time()
    boot_df = compute_bootstrap_pvalues(
        gross_pnl, total_cost, B=B, block_length=block_length, rng=rng,
    )
    log(f"  Bootstrap complete in {time.time() - t_boot:.1f}s")

    ann_net_sharpe = {}
    nw_t_net = {}
    for feat in features:
        metrics = sim.compute_metrics(book_dict[feat])
        ann_net_sharpe[feat] = metrics["net_pnl_sharpe"]
        nw_t_net[feat] = _newey_west_t(book_dict[feat]["net_pnl"].to_numpy(), maxlags=21)

    bh_net,    bh_net_adj    = benjamini_hochberg(boot_df["p_net"].to_numpy(),   q=FDR_Q)
    bhy_net,   bhy_net_adj   = bhy_procedure(boot_df["p_net"].to_numpy(),        q=FDR_Q)
    bh_gross,  bh_gross_adj  = benjamini_hochberg(boot_df["p_gross"].to_numpy(), q=FDR_Q)
    bhy_gross, bhy_gross_adj = bhy_procedure(boot_df["p_gross"].to_numpy(),      q=FDR_Q)
    bh_stud,   bh_stud_adj   = benjamini_hochberg(boot_df["p_stud"].to_numpy(),  q=FDR_Q)
    bhy_stud,  bhy_stud_adj  = bhy_procedure(boot_df["p_stud"].to_numpy(),       q=FDR_Q)

    result_df = pd.DataFrame({
        "track":              track,
        "feature":            features,
        "sign":               [signs[f] for f in features],
        "mean_gross":         boot_df["mean_gross"].to_numpy(),
        "mean_cost":          boot_df["mean_cost"].to_numpy(),
        "mean_net":           boot_df["T_obs"].to_numpy(),
        "ann_net_sharpe":     [ann_net_sharpe[f] for f in features],
        "nw_t_net":           [nw_t_net[f] for f in features],
        "p_net":              boot_df["p_net"].to_numpy(),
        "p_gross":            boot_df["p_gross"].to_numpy(),
        "p_stud":             boot_df["p_stud"].to_numpy(),
        "bh_net_selected":    bh_net,
        "bh_net_adj_p":       bh_net_adj,
        "bhy_net_selected":   bhy_net,
        "bhy_net_adj_p":      bhy_net_adj,
        "bh_gross_selected":  bh_gross,
        "bh_gross_adj_p":     bh_gross_adj,
        "bhy_gross_selected": bhy_gross,
        "bhy_gross_adj_p":    bhy_gross_adj,
        "bh_stud_selected":   bh_stud,
        "bh_stud_adj_p":      bh_stud_adj,
        "bhy_stud_selected":  bhy_stud,
        "bhy_stud_adj_p":     bhy_stud_adj,
    })

    null_summary_df = pd.DataFrame({
        "track":        track,
        "feature":      features,
        "null_mean_D":  boot_df["null_mean_D"].to_numpy(),
        "null_sd_D":    boot_df["null_sd_D"].to_numpy(),
        "B":            B,
        "block_length": block_length,
    })

    n_bh_net,    n_bhy_net    = int(bh_net.sum()),    int(bhy_net.sum())
    n_bh_gross,  n_bhy_gross  = int(bh_gross.sum()),  int(bhy_gross.sum())
    n_bh_stud,   n_bhy_stud   = int(bh_stud.sum()),   int(bhy_stud.sum())
    log(f"  BH/BHY selected (net):   {n_bh_net}/{len(features)}  /  {n_bhy_net}/{len(features)}")
    log(f"  BH/BHY selected (gross): {n_bh_gross}/{len(features)}  /  {n_bhy_gross}/{len(features)}")
    log(f"  BH/BHY selected (stud):  {n_bh_stud}/{len(features)}  /  {n_bhy_stud}/{len(features)}")

    top = result_df.sort_values("p_net").head(10)
    log("\n  Top 10 by p_net:")
    for _, row in top.iterrows():
        log(f"    {row['feature']:<28s} sign={row['sign']:+.0f}  "
            f"mean_net={row['mean_net']:+.6f}  p_net={row['p_net']:.4f}  "
            f"p_gross={row['p_gross']:.4f}  p_stud={row['p_stud']:.4f}  "
            f"BH_net={bool(row['bh_net_selected'])}")

    return result_df, null_summary_df


# ── Entry point ───────────────────────────────────────────────────────────────

def _rename_v2_output(path: Path) -> None:
    """Rename a v2 (invalid-null) output out of the way, without clobbering."""
    if not path.exists():
        log(f"  [rename] {path} does not exist — nothing to rename")
        return
    target = path.with_name(path.name + ".v2_invalid_null")
    if target.exists():
        log(f"  [rename] {target} already exists — leaving {path} in place")
        return
    path.rename(target)
    log(f"  [rename] {path} -> {target}")


def record_manifest(df: pd.DataFrame, per_track_summary: dict, args) -> None:
    """Write the v3 results to the results manifest.

    v3 originally wrote only a parquet. The manifest kept the v2 entries from
    the superseded invalid-null run, so Table 6 of the manuscript went on
    printing v2 p-values beside a v3 parquet that disagreed with them, and the
    coherence check passed because it pinned only the rejection counts, which
    are 0/30 under both. Every quantity the table prints is recorded here.

    A partial run (--tracks track_b, or a small --B smoke test) must not
    overwrite a full run's entries, so recording is skipped unless both tracks
    ran at the configured B.
    """
    if set(args.tracks) != {"track_a", "track_b"}:
        log(f"\n[manifest] skipped: partial run (tracks={args.tracks})")
        return
    if args.B < N_SAMPLES:
        log(f"\n[manifest] skipped: smoke run (B={args.B} < {N_SAMPLES})")
        return

    n_features = len(df[df["track"] == "track_a"])
    threshold_rank1 = FDR_Q / n_features

    record("tafdr.B", int(args.B), stage="ta_fdr")
    record("tafdr.block_length", int(args.block_length), stage="ta_fdr")
    record("tafdr.bh_threshold_rank1", threshold_rank1, stage="ta_fdr",
           meta={"description": "q/m, the BH cutoff the smallest p-value faces"})

    for track in ("track_a", "track_b"):
        g = df[df["track"] == track]
        s = per_track_summary[track]
        label = track.split("_")[1].upper()

        # The net test is the headline; the cost-blind and studentised columns
        # are recorded beside it so the manuscript cannot quote one and mean
        # another.
        record(f"tafdr.{track}.n_selected_bh", s["n_bh_net"], stage="ta_fdr", track=label)
        record(f"tafdr.{track}.n_selected_bhy", s["n_bhy_net"], stage="ta_fdr", track=label)
        record(f"tafdr.{track}.n_selected_bh_gross", s["n_bh_gross"], stage="ta_fdr", track=label)
        record(f"tafdr.{track}.n_selected_bhy_gross", s["n_bhy_gross"], stage="ta_fdr", track=label)
        record(f"tafdr.{track}.n_selected_bh_stud", s["n_bh_stud"], stage="ta_fdr", track=label)
        record(f"tafdr.{track}.n_features", n_features, stage="ta_fdr", track=label)
        record(f"tafdr.{track}.min_p_net", float(g["p_net"].min()), stage="ta_fdr", track=label)
        record(f"tafdr.{track}.min_p_gross", float(g["p_gross"].min()), stage="ta_fdr", track=label)

        # The v2 namespace is overwritten rather than left beside the v3 one:
        # a stale key holding invalid p-values is a landmine for any later
        # reader, and check_coherence still reads ta_fdr.*.summary.
        record(f"ta_fdr.{track}.summary", {
            "n_features": n_features,
            "n_bh_selected": s["n_bh_net"],
            "n_bhy_selected": s["n_bhy_net"],
        }, stage="ta_fdr", track=label)
        record(f"ta_fdr.{track}.null_params", {
            "null": "recentred_stationary_block_bootstrap_JOINT",
            "statistic": "mean_net_return",
            "block_length_days": int(args.block_length),
            "n_samples": int(args.B),
            "h0": "E[net return] <= 0",
            "preserves": ["ticker_autocorrelation", "cross_sectional_dependence"],
        }, stage="ta_fdr", track=label)
        record(f"ta_fdr.{track}.per_feature", [
            {
                "feature": r.feature,
                "mean_gross": float(r.mean_gross),
                "mean_cost": float(r.mean_cost),
                "mean_net": float(r.mean_net),
                "ann_net_sharpe": None if pd.isna(r.ann_net_sharpe) else float(r.ann_net_sharpe),
                "p_net": float(r.p_net),
                "p_gross": float(r.p_gross),
                "p_stud": float(r.p_stud),
                "bh_selected": bool(r.bh_net_selected),
                "bhy_selected": bool(r.bhy_net_selected),
            }
            for r in g.itertuples()
        ], stage="ta_fdr", track=label)

    log("\n[manifest] recorded v3 TA-FDR entries (tafdr.* and ta_fdr.*)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-length", type=int, default=BLOCK_LEN,
                        help="Mean block length for stationary bootstrap (default from spec.yaml)")
    parser.add_argument("--tracks", nargs="+", default=["track_b", "track_a"],
                        choices=["track_a", "track_b"])
    args = parser.parse_args()

    log(f"=== Tradable-Alpha FDR (TA-FDR) v3 — B={args.B}, "
        f"block_length={args.block_length} (recentred joint bootstrap) ===\n")
    t0  = time.time()
    rng = np.random.default_rng(args.seed)

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    _rename_v2_output(OLD_TAFDR_PATH)
    _rename_v2_output(OLD_OOS_PATH)

    log("\nLoading feature data to determine feature set ...")
    feat_df = pd.read_parquet(FEAT_PATH)
    features = get_feature_columns(feat_df)
    log(f"  Features from feature_spec: {len(features)}")
    log(f"  IS period: {IS_START} -> {IS_END}")
    log(f"  FDR q = {FDR_Q}")

    log("\nLoading OHLCV data ...")
    ohlcv = pd.read_parquet(OHLCV_PATH)

    log("\nLoading fdr_results.parquet (for orientation signs) ...")
    fdr_df = pd.read_parquet(FDR_PATH)

    sim = PortfolioSimulator(
        config_path=str(CFG_PATH),
        spread_bps=cfg["spread_bps"],
        impact_coeff=cfg["impact_coeff"],
    )

    all_results = []
    all_nulls   = []
    per_track_summary = {}

    for track in args.tracks:
        result_df, null_summary_df = run_ta_fdr_v3_track(
            track, features, feat_df, fdr_df, ohlcv, cfg, sim,
            B=args.B, block_length=args.block_length, rng=rng,
        )
        all_results.append(result_df)
        all_nulls.append(null_summary_df)
        per_track_summary[track] = {
            "n_features":  len(features),
            "n_bh_net":    int(result_df["bh_net_selected"].sum()),
            "n_bhy_net":   int(result_df["bhy_net_selected"].sum()),
            "n_bh_gross":  int(result_df["bh_gross_selected"].sum()),
            "n_bhy_gross": int(result_df["bhy_gross_selected"].sum()),
            "n_bh_stud":   int(result_df["bh_stud_selected"].sum()),
            "n_bhy_stud":  int(result_df["bhy_stud_selected"].sum()),
        }

    ta_fdr_v3_df    = pd.concat(all_results, ignore_index=True)
    null_summary_df = pd.concat(all_nulls, ignore_index=True)

    ta_fdr_v3_df.to_parquet(OUT_TAFDR_V3, index=False)
    log(f"\nSaved -> {OUT_TAFDR_V3}")
    null_summary_df.to_parquet(OUT_NULL_V3, index=False)
    log(f"Saved -> {OUT_NULL_V3}")

    elapsed = time.time() - t0
    summary = {
        "B":               args.B,
        "block_length":    args.block_length,
        "seed":            args.seed,
        "is_start":        IS_START,
        "is_end":          IS_END,
        "fdr_q":           FDR_Q,
        "elapsed_seconds": elapsed,
        "tracks":          per_track_summary,
        "generated_at":    datetime.now(timezone.utc).isoformat(),
    }
    OUT_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    log(f"Saved -> {OUT_SUMMARY_JSON}")

    record_manifest(ta_fdr_v3_df, per_track_summary, args)

    log("\n=== SUMMARY ===")
    for track in args.tracks:
        s = per_track_summary[track]
        log(f"  {track.upper()}: BH_net={s['n_bh_net']}  BHY_net={s['n_bhy_net']}  "
            f"BH_gross={s['n_bh_gross']}  BHY_gross={s['n_bhy_gross']}  "
            f"BH_stud={s['n_bh_stud']}  BHY_stud={s['n_bhy_stud']}")

    log(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
