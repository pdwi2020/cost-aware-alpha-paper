"""V3 — Cross-method statistical significance testing (OOS 2022-2024).

Answers the referee question: are the differences between methods statistically
distinguishable on the locked holdout, or is the small-sample uncertainty large
enough that the method ranking is not significant?

For every pair of strategies we report, on aligned daily NET P&L over the
753-day OOS window:
  1. Jobson-Korkie-Memmel (JKM, Memmel 2003) test of equal Sharpe ratios
     (parametric, accounts for return correlation).
  2. Stationary-bootstrap (Politis-Romano 1994, mean block 10) 95% CI and
     two-sided p-value for the Sharpe difference (distribution-free).
  3. Wilcoxon signed-rank test on the paired daily net-return difference
     (distribution-free location test).
We also report each strategy's own annualised net Sharpe with the Lo (2002)
standard error and p-value vs zero.

Strategies (OOS net P&L):
  Track B, Track A           ← holdout_pnl_track_{a,b}.parquet
  Momentum, Reversal, SPY    ← baselines_pnl.parquet (period == 'OOS 2022-24')
  Naive ML (RF+XGB)          ← ml_baseline_pnl_oos.parquet

Output:
  data/processed/method_significance.parquet   (pairwise matrix, long form)
  printed tables for the manuscript.

Run:
  python3 -u src/diagnostics/method_significance.py
"""

import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "processed"

ANN = 252.0
SEED = 42
N_BOOT = 10_000
MEAN_BLOCK = 10.0   # stationary-bootstrap mean block length (days)


def log(m): print(m, flush=True)


def ann_sharpe(r):
    r = np.asarray(r, float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 1e-12 else np.nan


def lo_se(r):
    """Lo (2002) IID standard error of the annualised Sharpe ratio."""
    n = len(r)
    sr = ann_sharpe(r) / np.sqrt(ANN)   # per-period SR
    se_per = np.sqrt((1 + 0.5 * sr**2) / n)
    return se_per * np.sqrt(ANN)


def jkm_pvalue(ri, rj):
    """Jobson-Korkie test with Memmel (2003) correction: H0 SR_i = SR_j.

    Uses per-period (daily) Sharpe ratios; returns (delta_ann_SR, z, p_two_sided).
    """
    ri = np.asarray(ri, float); rj = np.asarray(rj, float)
    n = len(ri)
    mi, mj = ri.mean(), rj.mean()
    si, sj = ri.std(ddof=1), rj.std(ddof=1)
    sri, srj = mi / si, mj / sj                      # per-period Sharpe
    rho = np.corrcoef(ri, rj)[0, 1]
    # Memmel (2003) asymptotic variance of (sri - srj)
    theta = (1.0 / n) * (
        2 * (1 - rho)
        + 0.5 * (sri**2 + srj**2 - 2 * sri * srj * (rho**2))
    )
    z = (sri - srj) / np.sqrt(theta) if theta > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return (sri - srj) * np.sqrt(ANN), float(z), float(p)


def stationary_bootstrap_indices(n, n_boot, mean_block, rng):
    """Politis-Romano stationary bootstrap index matrix (n_boot x n)."""
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    for b in range(n_boot):
        i = rng.integers(0, n)
        for t in range(n):
            idx[b, t] = i
            if rng.random() < p:
                i = rng.integers(0, n)
            else:
                i = (i + 1) % n
    return idx


def boot_sharpe_diff(ri, rj, rng):
    """Stationary-bootstrap 95% CI + two-sided p for ann. Sharpe difference."""
    ri = np.asarray(ri, float); rj = np.asarray(rj, float)
    n = len(ri)
    idx = stationary_bootstrap_indices(n, N_BOOT, MEAN_BLOCK, rng)
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        bi = idx[b]
        diffs[b] = ann_sharpe(ri[bi]) - ann_sharpe(rj[bi])
    point = ann_sharpe(ri) - ann_sharpe(rj)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p: fraction of resamples on the opposite side of 0
    frac_pos = (diffs > 0).mean()
    p = 2 * min(frac_pos, 1 - frac_pos)
    return point, float(lo), float(hi), float(p)


def load_series():
    """Return dict name -> daily net-return Series on the OOS window."""
    s = {}
    tb = pd.read_parquet(DATA / "holdout_pnl_track_b.parquet")
    tb.index = pd.to_datetime(tb.index)
    s["Track B"] = tb["net_pnl"]
    ta = pd.read_parquet(DATA / "holdout_pnl_track_a.parquet")
    ta.index = pd.to_datetime(ta.index)
    s["Track A"] = ta["net_pnl"]

    base = pd.read_parquet(DATA / "baselines_pnl.parquet")
    base.index = pd.to_datetime(base.index)
    oos = base[base["period"] == "OOS 2022-24"]
    label_map = {
        "Momentum L/S (12-1)": "Momentum",
        "Reversal L/S (5d)":   "Reversal",
        "Buy-and-Hold SPY":    "SPY",
    }
    for raw, short in label_map.items():
        sub = oos[oos["strategy"] == raw]
        if len(sub):
            s[short] = sub["net_pnl"]

    ml_path = DATA / "ml_baseline_pnl_oos.parquet"
    if ml_path.exists():
        ml = pd.read_parquet(ml_path)
        ml.index = pd.to_datetime(ml.index)
        s["Naive ML"] = ml["net_pnl"]
    else:
        log("  [warn] ml_baseline_pnl_oos.parquet not found — skipping Naive ML")
    return s


def main():
    log("=" * 70)
    log("  V3 — Cross-method significance testing (OOS 2022-2024 net P&L)")
    log("=" * 70)
    rng = np.random.default_rng(SEED)

    series = load_series()
    # Align all on common dates
    df = pd.DataFrame(series).dropna(how="any")
    log(f"\n  Aligned series: {df.shape[1]} strategies × {df.shape[0]} common days")
    log(f"  Strategies: {list(df.columns)}\n")

    # ── Per-strategy Sharpe vs zero ──────────────────────────────────────────
    log("  Per-strategy annualised NET Sharpe (Lo 2002 SE):")
    log(f"    {'Strategy':<12} {'SR':>8} {'SE':>7} {'z':>7} {'p(vs 0)':>9}")
    one = []
    for c in df.columns:
        r = df[c].to_numpy()
        sr = ann_sharpe(r); se = lo_se(r)
        z = sr / se; p = 2 * (1 - norm.cdf(abs(z)))
        one.append({"strategy": c, "ann_sharpe": round(sr, 3),
                    "lo_se": round(se, 3), "z": round(z, 3), "p_vs_zero": round(p, 3)})
        log(f"    {c:<12} {sr:>+8.3f} {se:>7.3f} {z:>+7.2f} {p:>9.3f}")

    # ── Pairwise tests ───────────────────────────────────────────────────────
    log("\n  Pairwise tests (ΔSR = SR_row − SR_col, annualised):")
    log(f"    {'Pair':<24} {'ΔSR':>7} {'JKM p':>7} {'boot p':>7} "
        f"{'boot 95% CI':>20} {'Wilcoxon p':>11}")
    rows = []
    for a, b in combinations(df.columns, 2):
        ri, rj = df[a].to_numpy(), df[b].to_numpy()
        dsr_j, z, p_jkm = jkm_pvalue(ri, rj)
        dsr_b, lo, hi, p_boot = boot_sharpe_diff(ri, rj, rng)
        try:
            _, p_wil = wilcoxon(ri - rj)
        except ValueError:
            p_wil = np.nan
        rows.append({
            "strat_a": a, "strat_b": b,
            "delta_sr": round(dsr_b, 3),
            "jkm_z": round(z, 3), "jkm_p": round(p_jkm, 4),
            "boot_p": round(p_boot, 4),
            "boot_ci_lo": round(lo, 3), "boot_ci_hi": round(hi, 3),
            "wilcoxon_p": round(float(p_wil), 4),
        })
        log(f"    {a+' vs '+b:<24} {dsr_b:>+7.2f} {p_jkm:>7.3f} {p_boot:>7.3f} "
            f"[{lo:>+6.2f},{hi:>+6.2f}] {p_wil:>11.3f}")

    out = pd.DataFrame(rows)
    out.to_parquet(DATA / "method_significance.parquet", index=False)
    pd.DataFrame(one).to_parquet(DATA / "method_significance_solo.parquet", index=False)
    log(f"\n  Saved → method_significance.parquet (+ _solo)")
    log(f"  N_boot={N_BOOT}, mean_block={MEAN_BLOCK}, seed={SEED}")


if __name__ == "__main__":
    main()
