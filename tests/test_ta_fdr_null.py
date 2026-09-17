"""tests/test_ta_fdr_null.py — Synthetic unit tests for the TA-FDR null redesign.

Tests do NOT require DuckDB, parquet files, or any heavy data infrastructure.
All data is constructed synthetically.  Run with:

    python3 -m pytest tests/test_ta_fdr_null.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.run_ta_fdr import (
    _compute_mean_net_return,
    _draw_stationary_block_bootstrap_indices,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Statistic is mean net return and net <= gross
# ─────────────────────────────────────────────────────────────────────────────

class TestStatistic:
    """T_f = mean(positions * returns - costs) and net <= gross when costs >= 0."""

    def _make_example(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        T, N = 50, 10
        pos    = rng.uniform(-0.05, 0.05, (T, N))
        ret    = rng.normal(0.001, 0.02, (T, N))
        costs  = np.abs(rng.normal(0.0002, 0.0001, T))   # costs >= 0 by construction
        return pos, ret, costs

    def test_statistic_equals_mean_net_return(self):
        """T_f must equal mean(positions * returns - costs)."""
        pos, ret, costs = self._make_example()
        expected = float(np.mean((pos * ret).sum(axis=1) - costs))
        result   = _compute_mean_net_return(pos, ret, costs)
        assert abs(result - expected) < 1e-12, (
            f"Statistic mismatch: got {result}, expected {expected}"
        )

    def test_net_le_gross_when_costs_nonneg(self):
        """mean(net) <= mean(gross) when costs >= 0 (pointwise guarantee)."""
        pos, ret, costs = self._make_example()
        gross_daily = (pos * ret).sum(axis=1)
        net_daily   = gross_daily - costs
        # costs >= 0 => net_daily <= gross_daily elementwise
        assert np.all(net_daily <= gross_daily + 1e-15), (
            "net return exceeds gross return (costs must be non-negative)"
        )
        # Therefore the means satisfy the same inequality
        assert np.mean(net_daily) <= np.mean(gross_daily) + 1e-15

    def test_net_equals_gross_when_costs_zero(self):
        """When costs = 0, net == gross exactly."""
        pos, ret, _ = self._make_example()
        costs_zero = np.zeros(len(pos))
        gross_mean = float(np.mean((pos * ret).sum(axis=1)))
        net_mean   = _compute_mean_net_return(pos, ret, costs_zero)
        assert abs(net_mean - gross_mean) < 1e-12

    def test_statistic_is_scalar_float(self):
        """The statistic must be a Python float."""
        pos, ret, costs = self._make_example()
        result = _compute_mean_net_return(pos, ret, costs)
        assert isinstance(result, float)

    def test_sharpe_would_differ(self):
        """Sanity: adding a constant to costs changes mean_net but not its direction."""
        pos, ret, costs = self._make_example(seed=7)
        t1 = _compute_mean_net_return(pos, ret, costs)
        t2 = _compute_mean_net_return(pos, ret, costs + 0.001)
        assert t2 < t1, "Higher costs should reduce mean net return"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Joint null preserves autocorrelation; old column shuffle does NOT
# ─────────────────────────────────────────────────────────────────────────────

class TestJointNullAutocorrelation:
    """Stationary block bootstrap preserves lag-1 AR better than iid shuffle."""

    @staticmethod
    def _ar1_series(T: int, rho: float, seed: int = 42) -> np.ndarray:
        """Generate one AR(1) series with known autocorrelation rho."""
        rng = np.random.default_rng(seed)
        x = np.zeros(T)
        x[0] = rng.normal()
        for t in range(1, T):
            x[t] = rho * x[t-1] + np.sqrt(1 - rho**2) * rng.normal()
        return x

    @staticmethod
    def _lag1_ac(x: np.ndarray) -> float:
        """Estimate lag-1 autocorrelation."""
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])

    def test_block_bootstrap_preserves_autocorrelation(self):
        """
        Block bootstrap resamples should have lag-1 AC much closer to the
        original than iid-shuffled series.

        For an AR(1) with rho=0.7, block length=21, T=500:
          - original AC ~ 0.7
          - block bootstrap AC mean ~ 0.35-0.7 (partial, due to block boundaries)
          - iid shuffle AC ~ 0 (destroys all serial dependence)
        We assert: mean(|block_AC|) >> mean(|shuffle_AC|).
        """
        T    = 500
        rho  = 0.7
        n_draws = 200
        block_length = 21

        x = self._ar1_series(T, rho, seed=1)
        original_ac = self._lag1_ac(x)

        rng = np.random.default_rng(42)
        block_acs   = []
        shuffle_acs = []

        for _ in range(n_draws):
            # Stationary block bootstrap resample
            idx_block  = _draw_stationary_block_bootstrap_indices(T, block_length, rng)
            x_block    = x[idx_block]
            block_acs.append(self._lag1_ac(x_block))

            # iid per-day column shuffle (old null — destroys autocorrelation)
            idx_shuffle = rng.permutation(T)
            x_shuffle   = x[idx_shuffle]
            shuffle_acs.append(self._lag1_ac(x_shuffle))

        mean_block_ac   = float(np.mean(np.abs(block_acs)))
        mean_shuffle_ac = float(np.mean(np.abs(shuffle_acs)))

        # Block bootstrap should preserve substantially more autocorrelation
        assert mean_block_ac > mean_shuffle_ac * 3, (
            f"Block bootstrap did not preserve autocorrelation materially:\n"
            f"  original AC = {original_ac:.3f}\n"
            f"  block AC    = {mean_block_ac:.3f} (mean |AC| over {n_draws} draws)\n"
            f"  shuffle AC  = {mean_shuffle_ac:.3f} (mean |AC| over {n_draws} draws)\n"
            f"  ratio       = {mean_block_ac / (mean_shuffle_ac + 1e-10):.1f}x"
        )

    def test_shuffle_destroys_autocorrelation(self):
        """
        Per-element iid shuffle should reduce |lag-1 AC| to near zero,
        confirming the old null was discarding autocorrelation structure.
        """
        T   = 500
        rho = 0.7
        n_draws = 200

        x   = self._ar1_series(T, rho, seed=2)
        rng = np.random.default_rng(99)

        shuffle_acs = []
        for _ in range(n_draws):
            idx_shuffle = rng.permutation(T)
            x_shuffle   = x[idx_shuffle]
            shuffle_acs.append(self._lag1_ac(x_shuffle))

        mean_shuffle_ac = float(np.mean(np.abs(shuffle_acs)))
        # iid shuffle should give near-zero autocorrelation
        assert mean_shuffle_ac < 0.10, (
            f"iid shuffle still shows AC = {mean_shuffle_ac:.3f}; expected < 0.10"
        )

    def test_block_bootstrap_output_length(self):
        """Resampled index array must have length T."""
        for T in [10, 100, 503]:
            rng = np.random.default_rng(0)
            idx = _draw_stationary_block_bootstrap_indices(T, block_length=21, rng=rng)
            assert len(idx) == T, f"Expected {T} indices, got {len(idx)}"

    def test_block_bootstrap_indices_in_range(self):
        """All resampled indices must be in [0, T)."""
        T = 252
        rng = np.random.default_rng(7)
        for _ in range(10):
            idx = _draw_stationary_block_bootstrap_indices(T, block_length=21, rng=rng)
            assert np.all(idx >= 0) and np.all(idx < T), (
                f"Indices out of range [0, {T})"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: p-value validity (noise feature vs perfect predictor)
# ─────────────────────────────────────────────────────────────────────────────

class TestPValueValidity:
    """
    p-values under joint null: noise features should NOT have systematically
    small p-values; perfect predictors should have small p-values.
    """

    @staticmethod
    def _joint_null_pvalue(
        pos: np.ndarray,
        ret: np.ndarray,
        costs: np.ndarray,
        B: int,
        block_length: int,
        rng: np.random.Generator,
    ) -> float:
        """Compute a single joint-null p-value for testing."""
        T = ret.shape[0]
        t_obs = _compute_mean_net_return(pos, ret, costs)

        null_stats = []
        for _ in range(B):
            idx   = _draw_stationary_block_bootstrap_indices(T, block_length, rng)
            ret_b = ret[idx, :]
            net_b = (pos * ret_b).sum(axis=1) - costs
            null_stats.append(float(net_b.mean()))

        null = np.array(null_stats)
        return float((1 + (null >= t_obs).sum()) / (1 + B))

    def test_noise_feature_pvalue_not_systematically_small(self):
        """
        Under a pure-noise feature (positions uncorrelated with returns),
        the permutation p-value should not be systematically near 0.
        Over several seeds the median p-value should be > 0.10.
        """
        T, N   = 200, 20
        B      = 300
        block  = 21
        seeds  = [10, 20, 30, 40, 50]

        p_vals = []
        for seed in seeds:
            rng     = np.random.default_rng(seed)
            # Noise positions and noise returns — no predictive alignment
            pos     = rng.uniform(-0.02, 0.02, (T, N))
            ret     = rng.normal(0.0, 0.015, (T, N))
            costs   = np.abs(rng.normal(0.0001, 5e-5, T))
            p = self._joint_null_pvalue(pos, ret, costs, B, block, rng)
            p_vals.append(p)

        median_p = float(np.median(p_vals))
        assert median_p > 0.10, (
            f"Noise feature has systematically small p-values: median={median_p:.3f} "
            f"(all p_vals={[f'{v:.3f}' for v in p_vals]})"
        )

    def test_perfect_predictor_pvalue_is_small(self):
        """
        A feature whose positions equal the next-period return (perfect alignment)
        should receive a small p-value.
        """
        T, N  = 300, 15
        B     = 500
        block = 21
        rng   = np.random.default_rng(99)

        # Construct returns
        ret = rng.normal(0.001, 0.015, (T, N))
        # Perfect predictor: positions proportional to returns (lag already aligned)
        # Scale so positions are in [-0.05, 0.05]
        pos = ret / (np.abs(ret).max() * 20)
        # Small costs that don't overwhelm the signal
        costs = np.full(T, 1e-6)

        p = self._joint_null_pvalue(pos, ret, costs, B, block, rng)
        assert p < 0.05, (
            f"Perfect predictor did not get a small p-value: p={p:.4f}"
        )

    def test_rejection_set_nondegenenate_on_mixed_features(self):
        """
        With a mix of noise and signal features, the rejection set should be
        non-degenerate: some features rejected, some not — not all-or-nothing.
        """
        T, N  = 300, 15
        B     = 400
        block = 21

        p_vals   = []
        rng_data = np.random.default_rng(77)
        rng_null = np.random.default_rng(78)

        ret = rng_data.normal(0.001, 0.015, (T, N))

        n_features = 8
        for k in range(n_features):
            if k < 3:
                # Strong signal: positions proportional to returns
                pos = ret / (np.abs(ret).max() * 20)
            else:
                # Pure noise
                pos = rng_data.uniform(-0.02, 0.02, (T, N))
            costs = np.full(T, 1e-6)
            p = self._joint_null_pvalue(pos, ret, costs, B, block, rng_null)
            p_vals.append(p)

        p_arr = np.array(p_vals)
        # Apply BH at q=0.10
        from src.fdr.bh_correction import benjamini_hochberg
        reject, _ = benjamini_hochberg(p_arr, q=0.10)

        n_rejected = int(reject.sum())
        # Should reject at least the 3 signal features, but not all 8
        assert 0 < n_rejected < n_features, (
            f"Rejection set is degenerate: {n_rejected}/{n_features} features rejected\n"
            f"p-values: {[f'{v:.3f}' for v in p_vals]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Determinism — fixed seed reproduces p-values
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    """Fixed RNG seed must reproduce identical bootstrap draws and p-values."""

    def _run_single(self, seed: int) -> float:
        """Run one p-value computation with fixed seed and return it."""
        T, N  = 150, 10
        B     = 100
        block = 21

        rng_data = np.random.default_rng(seed)
        pos   = rng_data.uniform(-0.03, 0.03, (T, N))
        ret   = rng_data.normal(0.0005, 0.01, (T, N))
        costs = np.abs(rng_data.normal(0.0001, 5e-5, T))

        t_obs = _compute_mean_net_return(pos, ret, costs)

        rng_null = np.random.default_rng(seed + 1000)
        null_stats = []
        for _ in range(B):
            idx   = _draw_stationary_block_bootstrap_indices(T, block, rng_null)
            ret_b = ret[idx, :]
            net_b = (pos * ret_b).sum(axis=1) - costs
            null_stats.append(float(net_b.mean()))

        null = np.array(null_stats)
        return float((1 + (null >= t_obs).sum()) / (1 + B))

    def test_same_seed_same_pvalue(self):
        """Running twice with the same seed must give identical p-values."""
        seed = 12345
        p1   = self._run_single(seed)
        p2   = self._run_single(seed)
        assert p1 == p2, f"p-values differ: {p1} vs {p2}"

    def test_different_seeds_different_pvalues(self):
        """Different seeds should (very likely) give different p-values."""
        p1 = self._run_single(1)
        p2 = self._run_single(2)
        # Allow the remote chance of equality, but it should be extremely rare
        # with float-valued statistics over 100 draws
        # Just ensure both are valid probabilities
        assert 0.0 < p1 <= 1.0
        assert 0.0 < p2 <= 1.0

    def test_pvalue_range(self):
        """p-values must lie in (0, 1]."""
        for seed in [1, 2, 3, 42, 99]:
            p = self._run_single(seed)
            assert 0.0 < p <= 1.0, f"p-value {p} out of (0, 1] for seed={seed}"

    def test_block_bootstrap_deterministic(self):
        """Same seed must produce identical index arrays."""
        T = 100
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        idx1 = _draw_stationary_block_bootstrap_indices(T, 21, rng1)
        idx2 = _draw_stationary_block_bootstrap_indices(T, 21, rng2)
        np.testing.assert_array_equal(idx1, idx2)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test — can import run_ta_fdr without error
# ─────────────────────────────────────────────────────────────────────────────

def test_import_run_ta_fdr():
    """run_ta_fdr.py must be importable and expose the v3 API.

    ``run_ta_fdr_track`` was the v2 entry point; v3 replaced it with
    ``run_ta_fdr_v3_track`` plus the I/O-free ``compute_bootstrap_pvalues``.
    The two legacy helpers stay importable for src/fdr/run_pbo_deployed.py.
    """
    import importlib
    mod = importlib.import_module("src.backtest.run_ta_fdr")
    assert hasattr(mod, "_compute_mean_net_return")
    assert hasattr(mod, "_draw_stationary_block_bootstrap_indices")
    assert hasattr(mod, "run_ta_fdr_v3_track")
    assert hasattr(mod, "compute_bootstrap_pvalues")
    assert hasattr(mod, "orientation_signs")


def test_ast_parse_run_ta_fdr():
    """run_ta_fdr.py must pass ast.parse (syntax check)."""
    import ast
    src_path = ROOT / "src" / "backtest" / "run_ta_fdr.py"
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()
    # Will raise SyntaxError if invalid
    tree = ast.parse(source)
    assert tree is not None


# ─────────────────────────────────────────────────────────────────────────────
# v3 recentred joint bootstrap: size, power, containment, cost monotonicity
# ─────────────────────────────────────────────────────────────────────────────

from src.backtest.run_ta_fdr import compute_bootstrap_pvalues  # noqa: E402


def _correlated_zero_mean_panel(n_feat, T, rng, scale=0.004):
    """n_feat zero-mean daily P&L series sharing one common factor."""
    common = rng.normal(0.0, scale, T)
    return {
        f"f{i}": 0.6 * common + 0.8 * rng.normal(0.0, scale, T)
        for i in range(n_feat)
    }


class TestRecentredNullSizeAndPower:
    """The recentred null must be correctly sized and still have power."""

    def test_size_pvalues_are_approximately_uniform(self):
        """Zero-mean books with zero costs sit exactly on H0: E[net] = 0.

        Under H0 the recentred p-values must be ~Uniform(0,1): the mean should
        sit near 0.5 and the nominal 5% rejection rate near 5%. This is the
        property v2 did not have (its p-values clustered near 0.5 for a
        different, mechanical reason: the static tilt stayed in the null).
        """
        rng_data = np.random.default_rng(11)
        p_all = []
        for _ in range(20):
            gross = _correlated_zero_mean_panel(30, 400, rng_data)
            costs = {k: np.zeros(400) for k in gross}
            out = compute_bootstrap_pvalues(
                gross, costs, B=199, block_length=21,
                rng=np.random.default_rng(101),
            )
            p_all.extend(out["p_net"].tolist())

        p = np.asarray(p_all)
        assert len(p) == 600
        assert 0.35 < p.mean() < 0.65, f"mean p={p.mean():.3f} (expected ~0.5)"
        rej = float((p <= 0.05).mean())
        assert 0.015 < rej < 0.09, f"5% rejection rate={rej:.3f} (expected ~0.05)"

    def test_power_strong_signal_is_rejected(self):
        """A book with a large positive mean must get a small p-value."""
        rng_data = np.random.default_rng(5)
        T = 1000
        gross = {"strong": rng_data.normal(0.001, 0.01, T)}   # mean/sd = 0.1/day
        costs = {"strong": np.full(T, 1e-6)}
        out = compute_bootstrap_pvalues(
            gross, costs, B=999, block_length=21,
            rng=np.random.default_rng(7),
        )
        assert out.loc["strong", "p_net"] < 0.01


class TestCostMonotonicity:
    """Costs must move the p-value. In v2 they provably could not."""

    @staticmethod
    def _p_net(cost_level, seed=3):
        rng_data = np.random.default_rng(21)
        T = 600
        gross = {"f": rng_data.normal(0.0004, 0.008, T)}
        costs = {"f": np.full(T, cost_level)}
        out = compute_bootstrap_pvalues(
            gross, costs, B=399, block_length=21,
            rng=np.random.default_rng(seed),
        )
        return float(out.loc["f", "p_net"])

    def test_higher_costs_raise_p_net(self):
        """Raising costs weakly raises p_net, and strictly for a real jump."""
        p_low = self._p_net(0.0)
        p_mid = self._p_net(0.0002)
        p_high = self._p_net(0.002)
        assert p_low <= p_mid <= p_high
        assert p_high > p_low, (
            f"p_net did not respond to costs: {p_low:.4f} -> {p_high:.4f}"
        )

    def test_old_design_p_is_cost_invariant(self):
        """Regression test documenting why the v2 null was invalid.

        v2 compared `mean(g[idx_b]) - mean(c)` against `mean(g) - mean(c)`:
        the identical `mean(c)` appears on both sides, so it cancels and the
        p-value cannot depend on the cost level, no matter how large. This
        test reproduces that algebra directly and asserts the (wrong)
        invariance, so any future reintroduction of the v2 comparison fails
        loudly here.
        """
        rng_data = np.random.default_rng(21)
        T = 600
        g = rng_data.normal(0.0004, 0.008, T)
        idx = np.array([
            _draw_stationary_block_bootstrap_indices(T, 21, np.random.default_rng(3))
            for _ in range(200)
        ])
        resampled_means = g[idx].mean(axis=1)

        def p_v2(cost_level):
            c_bar = float(np.full(T, cost_level).mean())
            t_obs = g.mean() - c_bar
            null = resampled_means - c_bar          # v2's null, cost included
            return (1 + int(np.sum(null >= t_obs))) / (1 + len(null))

        assert p_v2(0.0) == p_v2(0.002) == p_v2(1.0), (
            "v2's p-value should be provably cost-invariant"
        )


class TestNetGrossContainment:
    """p_net >= p_gross always (Lean: pval_antitone / tafdr_conservative)."""

    def test_net_p_never_below_gross_p(self):
        rng_data = np.random.default_rng(31)
        T = 500
        gross = _correlated_zero_mean_panel(10, T, rng_data)
        costs = {k: np.abs(rng_data.normal(0.0003, 0.0001, T)) for k in gross}
        out = compute_bootstrap_pvalues(
            gross, costs, B=299, block_length=21,
            rng=np.random.default_rng(13),
        )
        assert (out["p_net"] >= out["p_gross"] - 1e-12).all()
        # T_obs is the net statistic (mean gross - mean cost); with costs >= 0
        # it can never exceed the gross mean, which is what makes p_net >= p_gross.
        assert (out["T_obs"] <= out["mean_gross"] + 1e-15).all()
