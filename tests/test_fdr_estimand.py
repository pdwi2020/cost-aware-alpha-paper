"""tests/test_fdr_estimand.py — Unit tests for Phase-3 daily cross-sectional
IC estimand and stationary block bootstrap inference.

All tests are synthetic and self-contained: no DuckDB, no parquet files.

Tests
-----
1. Cross-sectional (not pooled): a pure time-trend feature has zero daily
   cross-sectional IC but would produce a strong pooled IC.  The new estimand
   correctly returns IC_bar ≈ 0 and a non-significant bootstrap p-value.

2. Detects real cross-sectional signal: feature ≈ target + noise each day →
   IC_bar is high and bootstrap p is significant.

3. Bootstrap p-value sanity: under pure noise the p-value is not
   systematically small; under strong signal p < 0.05.

4. Eligibility respected: rows with s0_eligible=False are excluded from the
   daily cross-section.

Run:
    python3 -m pytest tests/test_fdr_estimand.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.fdr.run_fdr import (
    daily_cross_sectional_ic,
    stationary_bootstrap_pvalue,
)

# ---------------------------------------------------------------------------
# Helper: build synthetic panel
# ---------------------------------------------------------------------------

def _make_panel(
    n_dates: int = 60,
    n_stocks: int = 50,
    seed: int = 0,
) -> tuple[list[pd.Timestamp], list[str]]:
    """Return (dates, tickers) for constructing test panels."""
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-02", periods=n_dates, freq="B")
    tickers = [f"S{i:03d}" for i in range(n_stocks)]
    return dates, tickers


def _build_df(
    feature_vals: np.ndarray,   # (n_stocks, n_dates)
    target_vals:  np.ndarray,   # (n_stocks, n_dates)
    dates,
    tickers,
    eligible: np.ndarray | None = None,  # (n_stocks, n_dates) bool
) -> pd.DataFrame:
    """Build a MultiIndex (ticker, date) DataFrame."""
    n_stocks, n_dates = feature_vals.shape
    rows = []
    for di, d in enumerate(dates):
        for si, tkr in enumerate(tickers):
            row = {
                "feature":       feature_vals[si, di],
                "target":        target_vals[si, di],
                "s0_eligible":   True if eligible is None
                                 else bool(eligible[si, di]),
            }
            rows.append((tkr, d, row))

    idx = pd.MultiIndex.from_tuples(
        [(r[0], r[1]) for r in rows], names=["ticker", "date"]
    )
    data = {k: [r[2][k] for r in rows]
            for k in ["feature", "target", "s0_eligible"]}
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# Test 1: Pure time-trend feature has zero daily cross-sectional IC
#
# Design: feature[i,t] = t  (identical across all stocks for each day t)
#         target[i,t]  = noise_i_t  (IID noise)
#
# Pooled Spearman correlation:  feature ~ time-trend correlated with target
# would be HIGH if target also has a time trend — so we add a time trend to
# the target as well, creating a spurious pooled correlation.
# Daily cross-sectional IC must be ~0 because feature has zero cross-sectional
# dispersion (all stocks share the same value each day).
# ---------------------------------------------------------------------------

def test_time_trend_has_zero_cross_sectional_ic():
    """Pure time-trend feature: daily IC ≈ 0, bootstrap p not significant."""
    rng = np.random.default_rng(42)
    n_dates, n_stocks = 100, 80

    dates, tickers = _make_panel(n_dates, n_stocks, seed=0)
    time_idx = np.arange(n_dates, dtype=float)

    # Feature = time trend, identical across stocks
    feature_vals = np.tile(time_idx, (n_stocks, 1))   # shape (n_stocks, n_dates)

    # Target = noise + time trend  → strong pooled correlation but zero XS
    noise   = rng.standard_normal((n_stocks, n_dates))
    target_vals = noise + time_idx[np.newaxis, :]

    df = _build_df(feature_vals, target_vals, dates, tickers)

    ic_series = daily_cross_sectional_ic(df, "feature", "target")

    # The feature is constant across stocks each day → Spearman rank-corr
    # is undefined (zero variance); spearmanr may return NaN or 0.
    # Either way, the non-NaN values should be 0 (or NaN due to ties).
    non_nan = ic_series.dropna()

    # If all days have constant feature → all NaN (min_names check or all-ties)
    # If spearmanr returns 0 for ties, those should all be 0.
    if len(non_nan) > 0:
        ic_bar = float(non_nan.mean())
        # IC_bar must be essentially 0 (no cross-sectional signal)
        assert abs(ic_bar) < 0.05, (
            f"Time-trend feature should have near-zero daily IC_bar, got {ic_bar:.4f}"
        )

        # Bootstrap p-value should NOT be significant (not < 0.05)
        stats = stationary_bootstrap_pvalue(ic_series, block_len=10, n=500, seed=7)
        assert stats["boot_p"] > 0.05, (
            f"Time-trend feature must not be significant; boot_p={stats['boot_p']:.4f}"
        )
    else:
        # All NaN → confirmed zero cross-sectional dispersion (PASS)
        pass


# ---------------------------------------------------------------------------
# Test 2: Real cross-sectional signal is detected
#
# Design: target[i,t] = IID noise  (fresh each day, so no time-series trend)
#         feature[i,t] = target[i,t] + small_noise  (strong XS rank alignment)
#
# Daily cross-sectional IC should be high, bootstrap p should be significant.
# ---------------------------------------------------------------------------

def test_real_cross_sectional_signal_detected():
    """Feature ≈ target + small noise → high IC_bar, significant bootstrap p."""
    rng = np.random.default_rng(123)
    n_dates, n_stocks = 120, 80

    dates, tickers = _make_panel(n_dates, n_stocks, seed=1)

    target_vals  = rng.standard_normal((n_stocks, n_dates))
    feature_vals = target_vals + 0.15 * rng.standard_normal((n_stocks, n_dates))

    df = _build_df(feature_vals, target_vals, dates, tickers)

    ic_series = daily_cross_sectional_ic(df, "feature", "target")

    non_nan = ic_series.dropna()
    assert len(non_nan) >= n_dates * 0.9, (
        "Expected almost all days to have valid IC (large cross-section)"
    )

    ic_bar = float(non_nan.mean())
    assert ic_bar > 0.5, (
        f"Feature ≈ target + noise should yield IC_bar > 0.5, got {ic_bar:.4f}"
    )

    stats = stationary_bootstrap_pvalue(ic_series, block_len=10, n=500, seed=7)
    assert stats["boot_p"] < 0.01, (
        f"Strong signal must be significant; boot_p={stats['boot_p']:.4f}"
    )
    assert stats["ci_low"] > 0.0, (
        f"90%% CI lower bound should be positive for strong signal; "
        f"ci_low={stats['ci_low']:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3: Bootstrap p-value sanity under noise and under signal
# ---------------------------------------------------------------------------

def test_bootstrap_pvalue_sanity():
    """Under pure noise p > 0.10; under strong signal p < 0.05 (fixed seed)."""
    rng = np.random.default_rng(999)
    n_days = 200

    # Pure noise IC series
    noise_ic = pd.Series(rng.standard_normal(n_days) * 0.02)
    stats_noise = stationary_bootstrap_pvalue(
        noise_ic, block_len=21, n=1000, seed=42
    )
    assert stats_noise["boot_p"] > 0.10, (
        f"Noise IC series must not be significant at 0.10; "
        f"boot_p={stats_noise['boot_p']:.4f}"
    )

    # Strong signal IC series (consistently positive)
    signal_ic = pd.Series(0.06 + rng.standard_normal(n_days) * 0.02)
    stats_signal = stationary_bootstrap_pvalue(
        signal_ic, block_len=21, n=1000, seed=42
    )
    assert stats_signal["boot_p"] < 0.05, (
        f"Strong signal IC series must be significant; "
        f"boot_p={stats_signal['boot_p']:.4f}"
    )

    # CI ordering
    assert stats_signal["ci_low"] < stats_signal["ci_high"], (
        "ci_low must be less than ci_high"
    )

    # n_days count matches input
    assert stats_noise["n_days"] == n_days
    assert stats_signal["n_days"] == n_days


# ---------------------------------------------------------------------------
# Test 4: Eligibility filter is respected
#
# Design: n_stocks stocks, half eligible, half not.  For ineligible stocks
# the feature is perfectly anti-correlated with target (IC = -1 if included).
# With eligibility filtering, only the eligible half contributes.
# ---------------------------------------------------------------------------

def test_eligibility_respected():
    """Ineligible rows (s0_eligible=False) are excluded from daily IC."""
    rng = np.random.default_rng(77)
    n_dates, n_stocks = 40, 60
    half = n_stocks // 2

    dates, tickers = _make_panel(n_dates, n_stocks, seed=2)

    # Eligible half: feature strongly positively correlated with target
    target_vals  = np.zeros((n_stocks, n_dates))
    feature_vals = np.zeros((n_stocks, n_dates))

    target_vals[:half, :]   = rng.standard_normal((half, n_dates))
    feature_vals[:half, :]  = target_vals[:half, :] + 0.1 * rng.standard_normal((half, n_dates))

    # Ineligible half: feature perfectly NEGATIVELY correlated with target
    target_vals[half:, :]   = rng.standard_normal((half, n_dates))
    feature_vals[half:, :]  = -target_vals[half:, :]   # IC = -1 if included

    # Eligibility: only first half eligible
    eligible = np.zeros((n_stocks, n_dates), dtype=bool)
    eligible[:half, :] = True

    df = _build_df(feature_vals, target_vals, dates, tickers, eligible=eligible)

    ic_series = daily_cross_sectional_ic(df, "feature", "target",
                                          eligible_col="s0_eligible")

    # IC should be high positive (eligible half has strong +ve signal)
    ic_bar = float(ic_series.dropna().mean())
    assert ic_bar > 0.5, (
        f"IC_bar should be > 0.5 (only eligible positive-IC stocks counted); "
        f"got {ic_bar:.4f}"
    )

    # Now WITHOUT eligibility filter (include all) — should be dragged negative
    ic_series_unfiltered = daily_cross_sectional_ic(
        df, "feature", "target", eligible_col="__nonexistent__"
    )
    ic_bar_unfiltered = float(ic_series_unfiltered.dropna().mean())
    # Ineligible anti-correlated stocks drag IC toward 0 or negative
    assert ic_bar_unfiltered < ic_bar, (
        f"Unfiltered IC ({ic_bar_unfiltered:.4f}) should be lower than "
        f"filtered IC ({ic_bar:.4f}); ineligible anti-correlated stocks "
        f"should drag the IC down"
    )


# ---------------------------------------------------------------------------
# Test 5: min_names threshold enforced
# ---------------------------------------------------------------------------

def test_min_names_threshold():
    """Days with fewer than min_names eligible stocks yield NaN IC."""
    rng = np.random.default_rng(11)
    n_dates = 5
    n_stocks = 10   # only 10 stocks total

    dates = pd.date_range("2020-01-02", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_stocks)]

    feature_vals = rng.standard_normal((n_stocks, n_dates))
    target_vals  = rng.standard_normal((n_stocks, n_dates))

    # For day 0: only 5 eligible stocks  → should be NaN (min_names=20 default)
    # For day 1: all 10 eligible          → should be NaN (10 < 20)
    # min_names=6 → day 1 will be valid, day 0 will be NaN
    eligible = np.ones((n_stocks, n_dates), dtype=bool)
    eligible[:5, 0] = False   # only 5 eligible on day 0

    df = _build_df(feature_vals, target_vals, dates, tickers, eligible=eligible)

    ic_default = daily_cross_sectional_ic(df, "feature", "target",
                                           min_names=20)
    # All days have < 20 stocks → all NaN
    assert ic_default.isna().all(), (
        "All ICs should be NaN when n_stocks < min_names=20"
    )

    ic_small = daily_cross_sectional_ic(df, "feature", "target",
                                         min_names=6)
    # Day 0: 5 eligible < 6 → NaN
    # Days 1-4: 10 eligible >= 6 → valid
    day0 = dates[0]
    assert pd.isna(ic_small.loc[day0]), (
        f"Day 0 has only 5 eligible stocks (< min_names=6) → should be NaN"
    )
    valid_days = ic_small.dropna()
    assert len(valid_days) == n_dates - 1, (
        f"Expected {n_dates-1} valid days, got {len(valid_days)}"
    )


# ---------------------------------------------------------------------------
# Test 6: Bootstrap p-value is CALIBRATED under the null (regression guard)
#
# Generates many independent null IC series (iid, true mean 0) and checks the
# empirical type-I error rate at alpha=0.05 is near nominal. The earlier
# anti-conservative convention (2*Phi(-2|Z|)) inflated this to ~0.25-0.35; the
# recentered null bootstrap keeps it near 0.05.
# ---------------------------------------------------------------------------

def test_bootstrap_pvalue_calibrated_under_null():
    """Type-I error at alpha=0.05 over many null series must be near nominal."""
    rng = np.random.default_rng(20240101)
    n_series = 300
    n_days = 150
    alpha = 0.05

    rejections = 0
    for _ in range(n_series):
        # iid daily ICs with TRUE mean 0 (null); realistic IC scale ~0.03
        null_ic = pd.Series(rng.standard_normal(n_days) * 0.03)
        stats = stationary_bootstrap_pvalue(null_ic, block_len=21, n=300,
                                            seed=int(rng.integers(1, 1_000_000)))
        if stats["boot_p"] < alpha:
            rejections += 1

    rej_rate = rejections / n_series
    # Calibrated bootstrap: rej_rate ~ alpha. Allow slack for block bootstrap on
    # finite series, but it must be FAR below the broken version's ~0.25+.
    assert rej_rate <= 0.12, (
        f"Bootstrap p-value is anti-conservative: type-I error {rej_rate:.3f} "
        f"at alpha={alpha} (should be ~{alpha}); FDR control would be invalid."
    )
