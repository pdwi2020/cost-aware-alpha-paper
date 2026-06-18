"""
Significance tests: Track B OOS daily net returns vs baselines.
OOS window: 2022-01-03 to 2024-12-31 (~753 trading days).

Tests:
  1. Diebold-Mariano (HAC, one-sided: Track B > baseline)
  2. Wilcoxon signed-rank (one-sided: Track B > baseline)
  3. Paired t-test (two-sided, as complement)

Outputs:
  - Printed table
  - data/processed/significance_tests.parquet
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

# ─── Load Track B OOS daily net returns ───────────────────────────────────────
hpnl = pd.read_parquet(DATA / "holdout_pnl_track_b.parquet")
# Index is date; column net_pnl
trackb = hpnl[["net_pnl"]].copy().rename(columns={"net_pnl": "track_b"})
trackb.index = pd.to_datetime(trackb.index)

# ─── Load ML baseline daily net returns (track_b only) ────────────────────────
ml = pd.read_parquet(DATA / "ml_baseline_returns.parquet")
ml_tb = ml[ml["track"] == "track_b"].copy()
ml_tb["date"] = pd.to_datetime(ml_tb["date"])

rf_ret = (
    ml_tb[ml_tb["model"] == "rf"]
    .set_index("date")[["net_ret"]]
    .rename(columns={"net_ret": "rf"})
)
xgb_ret = (
    ml_tb[ml_tb["model"] == "XGBoost"]
    .set_index("date")[["net_ret"]]
    .rename(columns={"net_ret": "xgboost"})
)

# ─── Load rule-based baseline OOS daily net returns ───────────────────────────
bp = pd.read_parquet(DATA / "baselines_pnl.parquet")
oos_bp = bp[bp["period"] == "OOS 2022-24"].copy()
oos_bp.index = pd.to_datetime(oos_bp.index)

mom_ret = (
    oos_bp[oos_bp["strategy"] == "Momentum L/S (12-1)"][["net_pnl"]]
    .rename(columns={"net_pnl": "momentum"})
)
rev_ret = (
    oos_bp[oos_bp["strategy"] == "Reversal L/S (5d)"][["net_pnl"]]
    .rename(columns={"net_pnl": "reversal"})
)

# ─── Merge all on common dates ────────────────────────────────────────────────
combined = (
    trackb
    .join(rf_ret, how="inner")
    .join(xgb_ret, how="inner")
    .join(mom_ret, how="inner")
    .join(rev_ret, how="inner")
)

print(f"Common OOS trading days (all 5 series aligned): {len(combined)}")
print(f"Date range: {combined.index.min().date()} to {combined.index.max().date()}\n")

ANNFACTOR = np.sqrt(252)


# ─── Helper: Annualised Sharpe ────────────────────────────────────────────────
def sharpe(ret: pd.Series) -> float:
    if ret.std() == 0:
        return np.nan
    return ret.mean() / ret.std() * ANNFACTOR


# ─── Helper: Diebold-Mariano (one-sided, HAC) ─────────────────────────────────
# Loss function: negative return (so lower is better).
# d_t = L(baseline) - L(trackb) = (-baseline_t) - (-trackb_t) = trackb_t - baseline_t
# H0: E[d] = 0  vs  H1: E[d] > 0  (Track B beats baseline)
# HAC variance uses Newey-West with lag = floor(1.5 * n^(1/3))

def newey_west_variance(d: np.ndarray, lag: int) -> float:
    """Newey-West HAC variance of the mean of d."""
    n = len(d)
    d_dm = d - d.mean()
    gamma0 = np.dot(d_dm, d_dm) / n
    hac_var = gamma0
    for j in range(1, lag + 1):
        gamma_j = np.dot(d_dm[j:], d_dm[:-j]) / n
        hac_var += 2 * (1 - j / (lag + 1)) * gamma_j
    return hac_var / n  # variance of the sample mean


def dm_test(trackb_r: np.ndarray, baseline_r: np.ndarray):
    """
    One-sided DM test: H1 = Track B mean daily return > baseline.
    Returns (dm_stat, p_value_one_sided).
    """
    d = trackb_r - baseline_r  # loss differential (positive = Track B better)
    n = len(d)
    lag = int(np.floor(1.5 * n ** (1 / 3)))
    hac_var = newey_west_variance(d, lag)
    if hac_var <= 0:
        return np.nan, np.nan
    dm_stat = d.mean() / np.sqrt(hac_var)
    # One-sided p-value: P(Z > dm_stat) under N(0,1), but use t(n-1) for small n
    p_one = stats.t.sf(dm_stat, df=n - 1)
    return dm_stat, p_one


# ─── Helper: Wilcoxon signed-rank (one-sided: Track B > baseline) ─────────────
def wilcoxon_one_sided(trackb_r: np.ndarray, baseline_r: np.ndarray):
    """One-sided Wilcoxon signed-rank test (alternative='greater')."""
    diff = trackb_r - baseline_r
    stat, p = stats.wilcoxon(diff, alternative="greater")
    return stat, p


# ─── Helper: Paired t-test (two-sided) ────────────────────────────────────────
def paired_ttest(trackb_r: np.ndarray, baseline_r: np.ndarray):
    diff = trackb_r - baseline_r
    t_stat, p_two = stats.ttest_1samp(diff, popmean=0)
    return t_stat, p_two


# ─── Run tests for each comparison ────────────────────────────────────────────
comparisons = [
    ("RF (track_b)",       "rf"),
    ("XGBoost (track_b)",  "xgboost"),
    ("Momentum L/S (12-1)", "momentum"),
    ("Reversal L/S (5d)",   "reversal"),
]

TRACK_B_SHARPE = sharpe(combined["track_b"])
print(f"Track B annualised OOS Sharpe: {TRACK_B_SHARPE:.3f}")
print(f"Track B mean daily net return: {combined['track_b'].mean()*10000:.2f} bps\n")

rows = []
for label, col in comparisons:
    tb_r   = combined["track_b"].values
    base_r = combined[col].values
    n      = len(tb_r)

    diff_bps = (tb_r - base_r).mean() * 10000  # mean daily differential in bps

    dm_stat, dm_p  = dm_test(tb_r, base_r)
    wx_stat, wx_p  = wilcoxon_one_sided(tb_r, base_r)
    t_stat,  t_p   = paired_ttest(tb_r, base_r)

    base_sharpe = sharpe(combined[col])

    rows.append({
        "comparison":        f"Track B vs {label}",
        "n_days":            n,
        "trackb_sharpe":     round(TRACK_B_SHARPE, 3),
        "baseline_sharpe":   round(base_sharpe, 3),
        "mean_diff_bps":     round(diff_bps, 2),
        "dm_stat":           round(dm_stat, 4) if not np.isnan(dm_stat) else np.nan,
        "dm_p_onesided":     round(dm_p,  6) if not np.isnan(dm_p)   else np.nan,
        "wilcoxon_stat":     round(wx_stat, 2),
        "wilcoxon_p_onesided": round(wx_p, 6),
        "paired_t_stat":     round(t_stat,  4),
        "paired_t_p_twosided": round(t_p,   6),
        "sig_5pct":          int(dm_p < 0.05 and wx_p < 0.05) if not np.isnan(dm_p) else 0,
        "sig_10pct":         int(dm_p < 0.10 and wx_p < 0.10) if not np.isnan(dm_p) else 0,
    })

results = pd.DataFrame(rows)

# ─── Print table ──────────────────────────────────────────────────────────────
print("=" * 100)
print(f"{'Comparison':<30} {'N':>5} {'Diff(bps)':>10} {'DM-stat':>8} {'DM-p(1)':>10} {'Wx-stat':>10} {'Wx-p(1)':>10} {'t-stat':>8} {'t-p(2)':>8} {'sig5%':>6} {'sig10%':>7}")
print("=" * 100)
for _, r in results.iterrows():
    print(
        f"{r['comparison']:<30} "
        f"{r['n_days']:>5} "
        f"{r['mean_diff_bps']:>10.2f} "
        f"{r['dm_stat']:>8.4f} "
        f"{r['dm_p_onesided']:>10.4f} "
        f"{r['wilcoxon_stat']:>10.0f} "
        f"{r['wilcoxon_p_onesided']:>10.4f} "
        f"{r['paired_t_stat']:>8.4f} "
        f"{r['paired_t_p_twosided']:>8.4f} "
        f"{'Yes' if r['sig_5pct'] else 'No':>6} "
        f"{'Yes' if r['sig_10pct'] else 'No':>7}"
    )
print("=" * 100)

# ─── Save ─────────────────────────────────────────────────────────────────────
out_path = DATA / "significance_tests.parquet"
results.to_parquet(out_path, index=False)
print(f"\nSaved: {out_path}")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\nSIGNIFICANCE SUMMARY")
print("-" * 60)
for _, r in results.iterrows():
    sig = "*** sig at 5%" if r["sig_5pct"] else ("** sig at 10%" if r["sig_10pct"] else "NOT significant")
    print(f"  {r['comparison']}: DM p={r['dm_p_onesided']:.4f}, Wx p={r['wilcoxon_p_onesided']:.4f} → {sig}")
