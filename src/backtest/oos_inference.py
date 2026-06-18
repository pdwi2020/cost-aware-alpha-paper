"""Statistical inference on the out-of-sample (2022-2024) Sharpe ratios.

Referee point (major): the headline OOS net Sharpe of +0.50 (Track B) is a point
estimate over a single ~3-year holdout. Without an uncertainty band it cannot be
distinguished from zero. This module attaches a confidence interval and a
significance test to every OOS Sharpe, using three independent methods that
should agree (they do, because the daily P&L is near-serially-uncorrelated):

  1. Lo (2002) analytic standard error of the Sharpe ratio (IID approximation).
  2. Newey-West HAC t-statistic on the mean daily P&L (serial-correlation robust).
  3. Stationary bootstrap (Politis & Romano, 1994) 95% CI on the annualised
     Sharpe — distribution-free, robust to autocorrelation AND non-normality.

The Sharpe convention matches PortfolioSimulator.compute_metrics:
    SR_ann = mean(daily_pnl) / std(daily_pnl, ddof=1) * sqrt(252).

Inputs:
    data/processed/holdout_pnl_track_a.parquet
    data/processed/holdout_pnl_track_b.parquet
Output:
    data/processed/oos_inference.parquet   (one row per track x pnl-type)

Run:
    python3 -u src/backtest/oos_inference.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "processed"

ANN = 252.0           # trading days per year
B_BOOT = 10_000       # bootstrap replicates
EXPECTED_BLOCK = 10   # stationary-bootstrap mean block length (days); robustness
                      # across {1,5,10,20,40} is reported separately and is flat.
BLOCK_GRID = [1, 5, 10, 20, 40]
SEED = 20240529


def sharpe_ann(r: np.ndarray) -> float:
    """Annualised Sharpe, ddof=1, matching compute_metrics."""
    sd = r.std(ddof=1)
    if sd < 1e-12:
        return np.nan
    return float(r.mean() / sd * np.sqrt(ANN))


def lo_se_sharpe(r: np.ndarray) -> dict:
    """Lo (2002) IID standard error of the Sharpe ratio + Gaussian CI/t/p.

    SE(SR_period) = sqrt((1 + 0.5 SR_period^2) / T). The t-statistic is
    scale-invariant, so it is identical whether SR is daily or annualised.
    """
    T = len(r)
    sr_d = sharpe_ann(r) / np.sqrt(ANN)        # per-period Sharpe
    se_d = np.sqrt((1.0 + 0.5 * sr_d ** 2) / T)
    t = sr_d / se_d
    p = 2.0 * stats.norm.sf(abs(t))
    se_ann = se_d * np.sqrt(ANN)
    sr_ann = sr_d * np.sqrt(ANN)
    return {
        "lo_t": float(t),
        "lo_p": float(p),
        "lo_se_ann": float(se_ann),
        "lo_ci_lo": float(sr_ann - 1.96 * se_ann),
        "lo_ci_hi": float(sr_ann + 1.96 * se_ann),
    }


def newey_west(r: np.ndarray) -> dict:
    """HAC (Newey-West) t-test of H0: mean daily P&L = 0.

    SR significance is equivalent to mean-return significance; the HAC
    covariance makes the test robust to serial correlation in the P&L.
    """
    T = len(r)
    maxlags = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))  # standard rule
    X = np.ones((T, 1))
    res = sm.OLS(r, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {
        "nw_maxlags": int(maxlags),
        "nw_t": float(res.tvalues[0]),
        "nw_p": float(res.pvalues[0]),
    }


def _sb_index_matrix(n: int, expected_block: int, B: int, rng) -> np.ndarray:
    """Vectorised stationary-bootstrap index matrix (B x n), circular wrap."""
    p = 1.0 / expected_block
    restart = rng.random((B, n)) < p
    restart[:, 0] = True
    starts = rng.integers(0, n, size=(B, n))
    idx = np.empty((B, n), dtype=np.int64)
    cur = starts[:, 0].copy()
    idx[:, 0] = cur
    for t in range(1, n):                      # loop over n (~753), vectorised over B
        cur = np.where(restart[:, t], starts[:, t], (cur + 1) % n)
        idx[:, t] = cur
    return idx


def stationary_bootstrap(r: np.ndarray, expected_block: int, B: int, rng) -> np.ndarray:
    """Return B bootstrap replicates of the annualised Sharpe."""
    n = len(r)
    idx = _sb_index_matrix(n, expected_block, B, rng)
    samp = r[idx]                              # (B, n)
    mean = samp.mean(axis=1)
    sd = samp.std(axis=1, ddof=1)
    sd = np.where(sd < 1e-12, np.nan, sd)
    return mean / sd * np.sqrt(ANN)


def boot_summary(r: np.ndarray, expected_block: int, B: int, rng) -> dict:
    sr_b = stationary_bootstrap(r, expected_block, B, rng)
    sr_b = sr_b[~np.isnan(sr_b)]
    lo, hi = np.percentile(sr_b, [2.5, 97.5])
    # two-sided bootstrap p-value for H0: SR = 0
    frac_neg = float(np.mean(sr_b < 0))
    frac_pos = float(np.mean(sr_b > 0))
    p = 2.0 * min(frac_neg, frac_pos)
    return {
        "boot_ci_lo": float(lo),
        "boot_ci_hi": float(hi),
        "boot_p": float(min(p, 1.0)),
        "boot_block": int(expected_block),
    }


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    block_robust = []

    for track in ["track_a", "track_b"]:
        df = pd.read_parquet(DATA / f"holdout_pnl_{track}.parquet")
        for col in ["gross_pnl", "net_pnl"]:
            r = df[col].dropna().to_numpy(dtype=float)
            rec = {"track": track, "pnl": col, "T": len(r), "sharpe_ann": sharpe_ann(r)}
            rec.update(lo_se_sharpe(r))
            rec.update(newey_west(r))
            rec.update(boot_summary(r, EXPECTED_BLOCK, B_BOOT, rng))
            rows.append(rec)

            # block-length robustness (net P&L only — the headline series)
            if col == "net_pnl":
                for bl in BLOCK_GRID:
                    bs = boot_summary(r, bl, B_BOOT, rng)
                    block_robust.append({
                        "track": track, "block": bl,
                        "ci_lo": bs["boot_ci_lo"], "ci_hi": bs["boot_ci_hi"],
                    })

    out = pd.DataFrame(rows)
    out.to_parquet(DATA / "oos_inference.parquet", index=False)

    # ---- pretty print -------------------------------------------------------
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== OOS Sharpe inference (2022-2024, T=753 trading days) ===\n")
    show = out[[
        "track", "pnl", "sharpe_ann",
        "lo_ci_lo", "lo_ci_hi", "lo_t", "lo_p",
        "nw_t", "nw_p",
        "boot_ci_lo", "boot_ci_hi", "boot_p",
    ]].copy()
    for c in show.columns:
        if show[c].dtype == float:
            show[c] = show[c].round(3)
    print(show.to_string(index=False))

    print("\n--- Net Sharpe 95% bootstrap CI vs expected block length ---")
    br = pd.DataFrame(block_robust)
    br["ci"] = br.apply(lambda x: f"[{x.ci_lo:+.2f}, {x.ci_hi:+.2f}]", axis=1)
    print(br.pivot(index="track", columns="block", values="ci").to_string())

    print("\n--- Headline (net P&L) ---")
    for track in ["track_a", "track_b"]:
        row = out[(out.track == track) & (out.pnl == "net_pnl")].iloc[0]
        sig = "NOT significant" if row.lo_p > 0.05 else "significant"
        print(f"  {track.upper()}: SR_net = {row.sharpe_ann:+.3f}  "
              f"95% CI [{row.boot_ci_lo:+.2f}, {row.boot_ci_hi:+.2f}]  "
              f"t={row.lo_t:+.2f} (Lo), t={row.nw_t:+.2f} (NW)  "
              f"p={row.lo_p:.2f} -> {sig} at 5%")
    print(f"\nSaved -> {DATA / 'oos_inference.parquet'}")


if __name__ == "__main__":
    main()
