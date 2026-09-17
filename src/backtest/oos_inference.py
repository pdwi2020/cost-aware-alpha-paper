"""Statistical inference on the out-of-sample (2022-2024) Sharpe ratios.

Referee point (major #1): the headline OOS net Sharpe of +0.50 (Track B) is a point
estimate over a single ~3-year holdout. Without an uncertainty band it cannot be
distinguished from zero. This module attaches a confidence interval and a
significance test to every OOS Sharpe, using three independent methods that
should agree (they do, because the daily P&L is near-serially-uncorrelated):

  1. Lo (2002) analytic standard error of the Sharpe ratio (IID approximation).
  2. Newey-West HAC t-statistic on the mean daily P&L (serial-correlation robust).
  3. Stationary bootstrap (Politis & Romano, 1994) 95%/90% percentile CIs on the
     annualised Sharpe -- distribution-free, robust to autocorrelation AND
     non-normality.

Referee point (major #2): "No economic effect-size threshold (the MCID analog):
state what net Sharpe counts as practically meaningful, independent of
significance, so readers can tell 'not significant' from 'too small to matter
even if it were'." The reusable core, ``window_inference``, answers both points
at once for any daily P&L window: a CI/significance test AND a classification
against a pre-specified economic-meaningfulness threshold SR* (see
results/staging/economic_threshold_note.md for the SR*=0.5 justification,
Grinold & Kahn 2000). ``window_inference`` is equivalent to two one-sided tests
at the 5% level applied to the 90% CI, so a reader can tell "not significant"
(CI spans 0) apart from "significant but too small to matter" (CI excludes 0 but
is entirely below SR*) apart from "significant and economically meaningful" (CI
lower bound clears SR*).

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

SR_STAR = 0.5                    # primary economic-meaningfulness threshold
SR_STAR_GRID = (0.3, 0.5, 0.75)  # sensitivity grid (see economic_threshold_note.md)


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


# ---------------------------------------------------------------------------
# window_inference: the reusable core (referee points #1 and #2 in one call)
# ---------------------------------------------------------------------------

def _classify(ci_lo: float, ci_hi: float, sr_star: float) -> str:
    """Classify a 90% Sharpe CI against an economic-meaningfulness threshold.

    Equivalent to two one-sided tests at the 5% level (TOST-style): a 90%
    two-sided CI excluding a value is equivalent to rejecting that value at
    the 5% one-sided level on each side. Checked in this exact priority
    order so every (ci_lo, ci_hi, sr_star) combination gets exactly one
    label:

      1. significantly_negative    : ci_hi < 0
           The entire CI is below zero -- unambiguous, checked first so it
           never falls through to "practically_null".
      2. significant_and_meaningful: ci_lo > sr_star
           The CI clears the economic-meaningfulness bar entirely.
      3. practically_null          : ci_lo <= 0 and ci_hi < sr_star
           Not distinguishable from zero AND too small to matter even if real.
      4. significant_but_small     : ci_lo > 0 and ci_hi < sr_star
           Distinguishable from zero, but the whole CI is below the
           meaningfulness bar -- "too small to matter even if it were real"
           in the reviewer's own words.
      5. inconclusive               : otherwise
           The CI straddles 0 and/or sr_star in a way not covered above
           (e.g. ci_lo <= 0 <= sr_star <= ci_hi, or 0 < ci_lo <= sr_star <= ci_hi):
           neither "clearly meaningful" nor "clearly too small" can be
           asserted from this window alone.
    """
    if ci_hi < 0:
        return "significantly_negative"
    if ci_lo > sr_star:
        return "significant_and_meaningful"
    if ci_lo <= 0 and ci_hi < sr_star:
        return "practically_null"
    if ci_lo > 0 and ci_hi < sr_star:
        return "significant_but_small"
    return "inconclusive"


def window_inference(
    daily_pnl: np.ndarray,
    sr_star: float = SR_STAR,
    sr_star_grid: tuple = SR_STAR_GRID,
    B: int = B_BOOT,
    block: int = EXPECTED_BLOCK,
    seed: int | None = None,
) -> dict:
    """Full inference + economic-meaningfulness classification for one P&L window.

    Parameters
    ----------
    daily_pnl : array-like
        Daily P&L (gross or net) for the window under test. NaNs are dropped.
    sr_star : float
        The primary economic-meaningfulness threshold (annualised net Sharpe).
        Always folded into ``sr_star_grid`` for the returned classification,
        even if not already present in it, and echoed back as
        ``result["sr_star"]`` / surfaced as ``result["primary_classification"]``.
    sr_star_grid : tuple of float
        Sensitivity grid of thresholds to classify against (default: the
        paper's (0.3, 0.5, 0.75) -- see results/staging/economic_threshold_note.md).
    B : int
        Number of stationary-bootstrap replicates (smaller values are fine for
        fast tests; the module default B_BOOT=10_000 is used for real runs).
    block : int
        Expected stationary-bootstrap block length in days (module default
        EXPECTED_BLOCK=10; robustness across other lengths is a separate,
        supplementary sweep in ``main()``, not part of this function).
    seed : int or None
        Seed for a private ``np.random.default_rng``. If None, a fresh seed is
        drawn from system entropy and reported back in the result so a caller
        can log it for exact reproducibility on rerun.

    Returns
    -------
    dict with keys:
        T                    : int, number of non-NaN observations used.
        sharpe_ann           : float, mean/sd(ddof=1)*sqrt(252) -- matches
                                PortfolioSimulator.compute_metrics exactly.
        ci_95 / ci_90         : {"lo": float, "hi": float} stationary-bootstrap
                                percentile CIs (2.5/97.5 and 5/95 percentiles of
                                the SAME B bootstrap replicates -- one draw,
                                nested CIs, not two independent bootstraps).
        boot_p               : float, two-sided bootstrap p-value for H0: SR=0,
                                from the same draw (2*min(P(SR_b<0),P(SR_b>0))).
        newey_west           : {"t", "p", "maxlags"} HAC t-test on mean daily
                                P&L; maxlags = floor(4*(T/100)**(2/9)) (unchanged
                                standard rule, see ``newey_west()`` above).
        lo_se                : {"se_ann", "t", "p", "ci_lo", "ci_hi"} from the
                                Lo (2002) analytic SE (``lo_se_sharpe()`` above);
                                ci_lo/ci_hi is the Gaussian sr_ann +/- 1.96*se_ann
                                interval (distinct from the bootstrap ci_95).
        min_detectable_sharpe : float, see derivation below.
        classification        : dict keyed by str(threshold) -> one of the five
                                 labels from ``_classify()``, for every value in
                                 the (sr_star-augmented) grid, using ci_90.
        primary_classification: classification[str(sr_star)], for convenience.
        sr_star / sr_star_grid: echoed back (grid includes sr_star, sorted).
        seed                   : int, the seed actually used (see above).

    Minimum detectable Sharpe (MDES) derivation
    --------------------------------------------
    Under H0: SR=0, the Lo (2002) per-period SE is sqrt((1+0.5*SR^2)/T) which,
    at SR=0, reduces to sqrt(1/T) -- i.e. SR^2 is negligible near the null. A
    two-sided test at size alpha=0.05 rejects H0 when |SR_hat| > z_{1-a/2}*SE.
    For power (1-beta)=0.80 at a true effect SR=MDES, the standard normal-test
    sample-size/power identity gives

        MDES = (z_{1-a/2} + z_{1-beta}) * SE
             = (z_0.975 + z_0.80) * SE  ,  z_0.975 = norm.ppf(0.975) ~= 1.9600,
                                            z_0.80  = norm.ppf(0.80)  ~= 0.8416.

    This function evaluates SE using ``lo_se["se_ann"]``, i.e. the Lo SE at the
    OBSERVED annualised Sharpe rather than re-deriving it exactly at SR=0. This
    is the standard, slightly-conservative practical shortcut used in the
    power-analysis literature (the SE is very flat near SR=0 for realistic
    Sharpe ratios, since the 0.5*SR^2 term is small relative to 1) -- it is an
    APPROXIMATION, not an exact null-evaluated MDES, and is documented as such
    here rather than presented as exact.
    """
    r = np.asarray(daily_pnl, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)

    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)

    sr_ann = sharpe_ann(r)
    lo = lo_se_sharpe(r)
    nw = newey_west(r)

    sr_boot = stationary_bootstrap(r, block, B, rng)
    sr_boot = sr_boot[~np.isnan(sr_boot)]
    ci_95_lo, ci_95_hi = np.percentile(sr_boot, [2.5, 97.5])
    ci_90_lo, ci_90_hi = np.percentile(sr_boot, [5.0, 95.0])
    frac_neg = float(np.mean(sr_boot < 0))
    frac_pos = float(np.mean(sr_boot > 0))
    boot_p = float(min(2.0 * min(frac_neg, frac_pos), 1.0))

    z_alpha = float(stats.norm.ppf(0.975))
    z_power = float(stats.norm.ppf(0.80))
    mdes = (z_alpha + z_power) * lo["lo_se_ann"]

    grid = tuple(sorted(set(sr_star_grid) | {sr_star}))
    classification = {str(s): _classify(ci_90_lo, ci_90_hi, s) for s in grid}

    return {
        "T": int(T),
        "sharpe_ann": sr_ann,
        "ci_95": {"lo": float(ci_95_lo), "hi": float(ci_95_hi)},
        "ci_90": {"lo": float(ci_90_lo), "hi": float(ci_90_hi)},
        "boot_p": boot_p,
        "newey_west": {"t": nw["nw_t"], "p": nw["nw_p"], "maxlags": nw["nw_maxlags"]},
        "lo_se": {
            "se_ann": lo["lo_se_ann"], "t": lo["lo_t"], "p": lo["lo_p"],
            "ci_lo": lo["lo_ci_lo"], "ci_hi": lo["lo_ci_hi"],
        },
        "min_detectable_sharpe": float(mdes),
        "classification": classification,
        "primary_classification": classification[str(sr_star)],
        "sr_star": sr_star,
        "sr_star_grid": list(grid),
        "seed": int(seed),
    }


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    block_robust = []

    for track in ["track_a", "track_b"]:
        df = pd.read_parquet(DATA / f"holdout_pnl_{track}.parquet")
        for col in ["gross_pnl", "net_pnl"]:
            r = df[col].dropna().to_numpy(dtype=float)

            # Derive a per-series seed from the shared, SEED-rooted RNG stream
            # so reruns of main() are exactly reproducible even though
            # window_inference() owns its own private RNG internally. Exact
            # bootstrap replicate values will differ slightly from any
            # pre-refactor run (different RNG substream construction) but are
            # statistically equivalent and stable across repeated runs.
            series_seed = int(rng.integers(0, 2**31 - 1))
            res = window_inference(
                r, sr_star=SR_STAR, sr_star_grid=SR_STAR_GRID,
                B=B_BOOT, block=EXPECTED_BLOCK, seed=series_seed,
            )

            rec = {
                "track": track, "pnl": col, "T": res["T"], "sharpe_ann": res["sharpe_ann"],
                "lo_t": res["lo_se"]["t"], "lo_p": res["lo_se"]["p"],
                "lo_se_ann": res["lo_se"]["se_ann"],
                "lo_ci_lo": res["lo_se"]["ci_lo"], "lo_ci_hi": res["lo_se"]["ci_hi"],
                "nw_maxlags": res["newey_west"]["maxlags"],
                "nw_t": res["newey_west"]["t"], "nw_p": res["newey_west"]["p"],
                "boot_ci_lo": res["ci_95"]["lo"], "boot_ci_hi": res["ci_95"]["hi"],
                "boot_p": res["boot_p"], "boot_block": EXPECTED_BLOCK,
                # additive: economic-effect-size threshold (referee point #2)
                "ci_90_lo": res["ci_90"]["lo"], "ci_90_hi": res["ci_90"]["hi"],
                "min_detectable_sharpe": res["min_detectable_sharpe"],
                "classification_sr0.3": res["classification"]["0.3"],
                "classification_sr0.5": res["classification"]["0.5"],
                "classification_sr0.75": res["classification"]["0.75"],
            }
            rows.append(rec)

            # block-length robustness (net P&L only -- the headline series)
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

    print("\n--- Economic effect-size threshold (SR* grid; 90% CI classification) ---")
    for track in ["track_a", "track_b"]:
        row = out[(out.track == track) & (out.pnl == "net_pnl")].iloc[0]
        print(f"  {track.upper()}: 90% CI [{row.ci_90_lo:+.2f}, {row.ci_90_hi:+.2f}]  "
              f"MDES(80% power)={row.min_detectable_sharpe:.2f}  "
              f"SR*=0.3 -> {row['classification_sr0.3']}  "
              f"SR*=0.5 -> {row['classification_sr0.5']}  "
              f"SR*=0.75 -> {row['classification_sr0.75']}")

    print(f"\nSaved -> {DATA / 'oos_inference.parquet'}")
    return out


if __name__ == "__main__":
    main()
