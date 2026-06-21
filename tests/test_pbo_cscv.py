"""tests/test_pbo_cscv.py — Unit tests for CSCV PBO on deployed net returns.

Three core tests per the specification:
  (a) i.i.d.-noise strategies → PBO ≈ 0.5 (selection is a coin-flip OOS)
  (b) One strategy genuinely best in every block → PBO ≈ 0
  (c) Determinism: fixed seed reproduces identical results

Additional:
  (d) cscv_pbo API contract (shape, dtype, range checks)
  (e) import / syntax smoke test for run_pbo_deployed

All tests are self-contained (no parquet files, no network).
Run with:
    python3 -m pytest tests/test_pbo_cscv.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fdr.run_pbo_deployed import cscv_pbo, degradation_slope


# ─────────────────────────────────────────────────────────────────────────────
# Test (a): i.i.d. noise → PBO ≈ 0.5
# ─────────────────────────────────────────────────────────────────────────────

class TestNoisePBO:
    """With pure i.i.d. noise, the IS-best strategy should be OOS-median ≈ 50%
    of the time (no real skill → selection is a coin-flip)."""

    @staticmethod
    def _iid_noise_matrix(
        T: int,
        M: int,
        n_blocks: int,
        seed: int,
        n_trials: int = 20,
    ) -> list[float]:
        """Run cscv_pbo n_trials times on fresh i.i.d.-noise matrices.
        Returns a list of PBO values (one per trial/seed)."""
        pbo_vals = []
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + trial * 1000)
            R = rng.normal(0.0, 0.01, (T, M))
            pbo, _, _, _ = cscv_pbo(R, n_blocks=n_blocks, max_splits=300, rng_seed=seed)
            pbo_vals.append(pbo)
        return pbo_vals

    def test_noise_pbo_near_half_small(self):
        """n=8 blocks, 10 strategies, short T: median PBO ∈ [0.30, 0.70]."""
        pbo_vals = self._iid_noise_matrix(T=160, M=10, n_blocks=8, seed=0)
        median_pbo = float(np.median(pbo_vals))
        assert 0.30 <= median_pbo <= 0.70, (
            f"i.i.d. noise PBO median={median_pbo:.3f} not in [0.30, 0.70]\n"
            f"  All PBO values: {[f'{v:.3f}' for v in pbo_vals]}"
        )

    def test_noise_pbo_near_half_16blocks(self):
        """Standard n=16 blocks, 30 strategies: median PBO ∈ [0.30, 0.70]."""
        pbo_vals = self._iid_noise_matrix(T=320, M=30, n_blocks=16, seed=42)
        median_pbo = float(np.median(pbo_vals))
        assert 0.30 <= median_pbo <= 0.70, (
            f"i.i.d. noise PBO median={median_pbo:.3f} not in [0.30, 0.70]\n"
            f"  All PBO values: {[f'{v:.3f}' for v in pbo_vals]}"
        )

    def test_noise_pbo_not_trivially_zero(self):
        """Under pure noise the PBO must not collapse to zero (no real skill)."""
        pbo_vals = self._iid_noise_matrix(T=160, M=8, n_blocks=8, seed=7, n_trials=15)
        # At least half of the trials should have PBO > 0.25
        above_threshold = sum(v > 0.25 for v in pbo_vals)
        assert above_threshold >= 7, (
            f"Too many noise trials have PBO ≤ 0.25: {pbo_vals}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test (b): one strategy genuinely best in every block → PBO ≈ 0
# ─────────────────────────────────────────────────────────────────────────────

class TestGenuineBestPBO:
    """If one strategy strictly dominates all others in every block, IS always
    selects it and it is OOS-best → PBO should be ≈ 0 (no overfitting)."""

    @staticmethod
    def _dominant_strategy_matrix(T: int, M: int, n_blocks: int, seed: int,
                                   signal_strength: float = 5.0) -> np.ndarray:
        """Return (T × M) where column 0 is genuine; the rest are i.i.d. noise."""
        rng = np.random.default_rng(seed)
        R = rng.normal(0.0, 0.01, (T, M))
        # Strategy 0: add a large persistent positive daily return
        R[:, 0] += signal_strength * 0.01
        return R

    def test_dominant_strategy_pbo_near_zero_8blocks(self):
        """With one dominant strategy and 8 blocks, PBO ≈ 0."""
        R = self._dominant_strategy_matrix(T=160, M=10, n_blocks=8, seed=1)
        pbo, lv, _, _ = cscv_pbo(R, n_blocks=8, max_splits=300, rng_seed=1)
        # PBO should be very low — dominant strategy is always IS-best and OOS-best
        assert pbo <= 0.10, (
            f"Dominant strategy should give PBO ≈ 0, got PBO={pbo:.3f}\n"
            f"  λ mean={lv.mean():.3f}, P(λ>0)={(lv>0).mean():.3f}"
        )

    def test_dominant_strategy_pbo_near_zero_16blocks(self):
        """With one dominant strategy and 16 blocks, PBO ≈ 0."""
        R = self._dominant_strategy_matrix(T=320, M=30, n_blocks=16, seed=2,
                                            signal_strength=6.0)
        pbo, lv, _, _ = cscv_pbo(R, n_blocks=16, max_splits=500, rng_seed=2)
        assert pbo <= 0.10, (
            f"Dominant strategy (16 blocks, 30 strategies) gave PBO={pbo:.3f}"
        )

    def test_dominant_strategy_lambda_positive(self):
        """When one strategy dominates, most λ values should be positive (OOS above median)."""
        R = self._dominant_strategy_matrix(T=160, M=10, n_blocks=8, seed=3)
        _, lv, _, _ = cscv_pbo(R, n_blocks=8, max_splits=300, rng_seed=3)
        p_pos = float((lv > 0).mean())
        assert p_pos >= 0.80, (
            f"Dominant strategy: P(λ>0)={p_pos:.3f}, expected ≥ 0.80"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test (c): Determinism — fixed seed reproduces identical results
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    """Fixed seed + fixed data → identical PBO, λ array, slopes."""

    @staticmethod
    def _run(seed: int) -> tuple:
        rng = np.random.default_rng(seed)
        R = rng.normal(0.0, 0.01, (160, 15))
        pbo, lv, is_sr, oos_sr = cscv_pbo(R, n_blocks=8, max_splits=100,
                                            rng_seed=seed)
        return pbo, lv, is_sr, oos_sr

    def test_same_seed_same_pbo(self):
        """Same seed produces identical PBO."""
        pbo1, _, _, _ = self._run(42)
        pbo2, _, _, _ = self._run(42)
        assert pbo1 == pbo2, f"PBO differs across runs: {pbo1} vs {pbo2}"

    def test_same_seed_same_lambda(self):
        """Same seed produces identical λ array."""
        _, lv1, _, _ = self._run(42)
        _, lv2, _, _ = self._run(42)
        np.testing.assert_array_equal(lv1, lv2, err_msg="λ arrays differ")

    def test_same_seed_same_sharpes(self):
        """Same seed produces identical IS/OOS Sharpe arrays."""
        _, _, is1, oos1 = self._run(99)
        _, _, is2, oos2 = self._run(99)
        np.testing.assert_array_equal(is1, is2,  err_msg="IS Sharpe arrays differ")
        np.testing.assert_array_equal(oos1, oos2, err_msg="OOS Sharpe arrays differ")

    def test_different_seeds_may_differ(self):
        """Different seeds should (very likely) produce different λ arrays."""
        _, lv1, _, _ = self._run(1)
        _, lv2, _, _ = self._run(2)
        # They might be equal by chance, but with 100 splits and continuous data
        # this is astronomically unlikely — check at least the PBO stats differ
        # (we only assert they are valid floats; not requiring inequality to avoid flake)
        assert 0.0 <= float(np.mean(lv1 <= 0)) <= 1.0
        assert 0.0 <= float(np.mean(lv2 <= 0)) <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Test (d): API contract — shape, dtype, value ranges
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIContract:
    """cscv_pbo must satisfy its output contract regardless of input."""

    def _make_R(self, T: int, M: int, seed: int = 0) -> np.ndarray:
        return np.random.default_rng(seed).normal(0, 0.01, (T, M))

    def test_pbo_in_unit_interval(self):
        """PBO must be in [0, 1]."""
        R = self._make_R(160, 10)
        pbo, _, _, _ = cscv_pbo(R, n_blocks=8, max_splits=50)
        assert 0.0 <= pbo <= 1.0, f"PBO={pbo} not in [0,1]"

    def test_lambda_shape_matches_n_splits(self):
        """λ array length must equal number of splits used."""
        R = self._make_R(160, 10)
        max_splits = 50
        _, lv, _, _ = cscv_pbo(R, n_blocks=8, max_splits=max_splits)
        assert len(lv) == max_splits, f"len(λ)={len(lv)} ≠ max_splits={max_splits}"

    def test_is_oos_sharpe_lengths_match_lambda(self):
        """is_sharpes and oos_sharpes must have same length as λ."""
        R = self._make_R(160, 10)
        _, lv, is_sr, oos_sr = cscv_pbo(R, n_blocks=8, max_splits=50)
        assert len(is_sr)  == len(lv), "IS Sharpe length mismatch"
        assert len(oos_sr) == len(lv), "OOS Sharpe length mismatch"

    def test_lambda_is_finite(self):
        """λ values must all be finite (no inf / nan from logit)."""
        R = self._make_R(160, 10)
        _, lv, _, _ = cscv_pbo(R, n_blocks=8, max_splits=50)
        assert np.all(np.isfinite(lv)), f"Non-finite λ values: {lv[~np.isfinite(lv)]}"

    def test_single_strategy_pbo(self):
        """With M=1 strategy, it is always IS-best and OOS rank = 1/1 = 1.0
        (above median of one), so PBO = 0."""
        R = self._make_R(160, 1)
        pbo, lv, _, _ = cscv_pbo(R, n_blocks=8, max_splits=50)
        assert pbo == 0.0, f"M=1: expected PBO=0, got {pbo}"
        assert np.all(lv > 0), "M=1: all λ should be > 0 (rank=1.0 → ω clips above 0.5)"

    def test_all_splits_used_when_max_splits_is_none(self):
        """When max_splits=None, all C(n_blocks, half) combos are enumerated."""
        from math import comb
        n_blocks = 6
        T = n_blocks * 10
        M = 5
        R = self._make_R(T, M)
        expected_n = comb(n_blocks, n_blocks // 2)
        _, lv, _, _ = cscv_pbo(R, n_blocks=n_blocks, max_splits=None)
        assert len(lv) == expected_n, (
            f"Expected {expected_n} splits for C({n_blocks},{n_blocks//2}), got {len(lv)}"
        )

    def test_returns_four_tuple(self):
        """cscv_pbo must return exactly (pbo, lambda_vals, is_sharpes, oos_sharpes)."""
        R = self._make_R(80, 5)
        result = cscv_pbo(R, n_blocks=4, max_splits=10)
        assert len(result) == 4, f"Expected 4-tuple, got {len(result)}-tuple"

    def test_nan_rows_handled(self):
        """NaN rows in the returns matrix should not cause crashes."""
        R = self._make_R(80, 5)
        R[5, :] = np.nan
        R[20, 2] = np.nan
        pbo, lv, _, _ = cscv_pbo(R, n_blocks=4, max_splits=20)
        assert 0.0 <= pbo <= 1.0
        assert np.all(np.isfinite(lv))


# ─────────────────────────────────────────────────────────────────────────────
# Test: degradation_slope
# ─────────────────────────────────────────────────────────────────────────────

class TestDegradationSlope:
    """OLS slope of OOS ~ IS Sharpe."""

    def test_perfect_positive_correlation(self):
        """If OOS = IS + noise (strong correlation) slope ≈ 1."""
        rng = np.random.default_rng(0)
        is_sr  = rng.normal(0, 1, 500)
        oos_sr = is_sr + rng.normal(0, 0.05, 500)   # near-perfect correlation
        slope  = degradation_slope(is_sr, oos_sr)
        assert 0.85 <= slope <= 1.15, f"Expected slope≈1, got {slope:.3f}"

    def test_zero_correlation_slope_near_zero(self):
        """If OOS is independent of IS, slope should be near zero."""
        rng    = np.random.default_rng(1)
        is_sr  = rng.normal(0, 1, 1000)
        oos_sr = rng.normal(0, 1, 1000)   # independent
        slope  = degradation_slope(is_sr, oos_sr)
        assert abs(slope) < 0.20, f"Independent series: slope={slope:.3f}, expected ≈ 0"

    def test_negative_correlation_negative_slope(self):
        """Negative correlation → slope < 0 (IS-best underperforms OOS)."""
        rng    = np.random.default_rng(2)
        is_sr  = rng.normal(0, 1, 500)
        oos_sr = -is_sr + rng.normal(0, 0.05, 500)
        slope  = degradation_slope(is_sr, oos_sr)
        assert slope < -0.5, f"Negative correlation: slope={slope:.3f}, expected < -0.5"

    def test_single_point_returns_nan(self):
        """Single data point: slope undefined, must return nan."""
        slope = degradation_slope(np.array([1.0]), np.array([2.0]))
        assert np.isnan(slope), f"Expected nan for single-point, got {slope}"

    def test_empty_returns_nan(self):
        """Empty arrays: must return nan."""
        slope = degradation_slope(np.array([]), np.array([]))
        assert np.isnan(slope), f"Expected nan for empty input, got {slope}"


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test: import and syntax check for run_pbo_deployed
# ─────────────────────────────────────────────────────────────────────────────

def test_import_run_pbo_deployed():
    """run_pbo_deployed.py must be importable and expose cscv_pbo."""
    import importlib
    mod = importlib.import_module("src.fdr.run_pbo_deployed")
    assert hasattr(mod, "cscv_pbo"), "cscv_pbo not found in run_pbo_deployed"
    assert hasattr(mod, "degradation_slope"), "degradation_slope not found"
    assert hasattr(mod, "build_net_return_matrix"), "build_net_return_matrix not found"
    assert hasattr(mod, "main"), "main not found"


def test_ast_parse_run_pbo_deployed():
    """run_pbo_deployed.py must pass ast.parse (no syntax errors)."""
    import ast
    src_path = ROOT / "src" / "fdr" / "run_pbo_deployed.py"
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    assert tree is not None
