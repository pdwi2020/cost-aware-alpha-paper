"""Unit tests for LORD++ online FDR procedure.

Tests:
  (a) Under-null calibration  — all-Uniform(0,1) p-values should produce
      few discoveries (≤ expected level) over many independent trials.
  (b) Power                   — near-zero p-values (strong signals) should
      produce discoveries.
  (c) Determinism             — same input always yields same output.
"""

from __future__ import annotations

import random
import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import lord_plus_plus from the source file directly (avoids heavy imports
# triggered by main() which tries to read parquet files and the manifest).
# ---------------------------------------------------------------------------

_SRC = Path(__file__).resolve().parent.parent / "src" / "fdr" / "run_alpha_investing.py"


def _load_lord():
    """Load lord_plus_plus and _gamma_sequence without executing main()."""
    spec = importlib.util.spec_from_file_location("_run_alpha_investing", _SRC)
    mod = importlib.util.module_from_spec(spec)
    # Patch out heavy imports that don't exist in test env (manifest, feature_spec)
    import sys
    import types

    # Provide lightweight stubs so the module can be imported
    if "src.manifest" not in sys.modules:
        stub_manifest = types.ModuleType("src.manifest")
        stub_manifest.record = lambda *a, **kw: None
        sys.modules["src.manifest"] = stub_manifest

    if "src.features.feature_spec" not in sys.modules:
        stub_fs = types.ModuleType("src.features.feature_spec")
        stub_fs.KEPT_FEATURES = []
        stub_fs.ADDED_INTERACTIONS = []
        sys.modules["src.features.feature_spec"] = stub_fs

    # Also ensure parent packages exist
    for pkg in ("src", "src.fdr", "src.features"):
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)

    spec.loader.exec_module(mod)
    return mod


_mod = _load_lord()
lord_plus_plus = _mod.lord_plus_plus
_gamma_sequence = _mod._gamma_sequence


# ---------------------------------------------------------------------------
# (c) Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output(self):
        """LORD++ must be a pure function: identical inputs → identical outputs."""
        rng = np.random.default_rng(42)
        p = rng.uniform(0, 1, size=30).tolist()

        result_1 = lord_plus_plus(p, alpha=0.10)
        result_2 = lord_plus_plus(p, alpha=0.10)

        assert result_1 == result_2, "LORD++ is not deterministic for the same input"

    def test_empty_input(self):
        assert lord_plus_plus([], alpha=0.10) == []

    def test_single_element_rejected(self):
        """p=0 with alpha=0.10 → should reject (alpha_1 > 0)."""
        result = lord_plus_plus([0.0], alpha=0.10)
        assert result == [True]

    def test_single_element_not_rejected(self):
        """p=1 → should not reject."""
        result = lord_plus_plus([1.0], alpha=0.10)
        assert result == [False]

    def test_output_length_matches_input(self):
        p = [0.1, 0.05, 0.9, 0.3]
        result = lord_plus_plus(p, alpha=0.10)
        assert len(result) == len(p)

    def test_output_dtype_bool(self):
        p = [0.01, 0.5, 0.99]
        result = lord_plus_plus(p, alpha=0.10)
        assert all(isinstance(r, bool) for r in result)


# ---------------------------------------------------------------------------
# (a) Under-null: Uniform(0,1) p-values → few discoveries
# ---------------------------------------------------------------------------

class TestNullCalibration:
    """Under the global null (all p ~ Uniform), LORD++ should make very few
    discoveries in expectation.  We run many independent experiments and
    check that the average discovery rate is small (≤ alpha + slack).

    This is NOT a strict FDR theorem test (that would require oracle access to
    which hypotheses are null) but a sanity check: the procedure does not fire
    wildly under null inputs.
    """

    def test_null_discovery_rate_small(self):
        alpha = 0.10
        n_trials = 500
        m = 40   # number of hypotheses per trial
        rng = np.random.default_rng(0)
        total_discoveries = 0

        for _ in range(n_trials):
            p = rng.uniform(0.0, 1.0, size=m).tolist()
            disc = sum(lord_plus_plus(p, alpha=alpha))
            total_discoveries += disc

        mean_discoveries = total_discoveries / n_trials
        # Under the null LORD++ controls mFDR ≤ alpha.  With m=40, we expect
        # at most alpha * m = 4 discoveries on average; with Uniform p-values
        # and the LORD++ wealth schedule the actual number should be much
        # lower (< 2).  We allow a generous slack to avoid flaky tests.
        assert mean_discoveries <= alpha * m, (
            f"Mean discoveries under null ({mean_discoveries:.2f}) exceeds "
            f"alpha * m = {alpha * m:.2f}"
        )

    def test_null_many_hypotheses(self):
        """Even with m=200 null hypotheses, expected discoveries stay bounded."""
        alpha = 0.10
        rng = np.random.default_rng(7)
        m = 200
        n_trials = 200
        total_disc = 0
        for _ in range(n_trials):
            p = rng.uniform(0.0, 1.0, size=m).tolist()
            total_disc += sum(lord_plus_plus(p, alpha=alpha))
        mean_disc = total_disc / n_trials
        assert mean_disc <= alpha * m, (
            f"Null mean discoveries = {mean_disc:.2f} > {alpha * m}"
        )


# ---------------------------------------------------------------------------
# (b) Power: strong signals → LORD++ makes discoveries
# ---------------------------------------------------------------------------

class TestPower:
    def test_all_zero_p_values(self):
        """When all p-values are 0 (perfect signals), LORD++ should reject many."""
        m = 20
        p = [0.0] * m
        result = lord_plus_plus(p, alpha=0.10)
        n_disc = sum(result)
        assert n_disc >= 3, (
            f"Expected ≥ 3 discoveries with all-zero p-values, got {n_disc}"
        )

    def test_strong_signals_at_start(self):
        """Strong signals early in the sequence build wealth, enabling later rejections."""
        # First 5 features: p=0.001 (strong); next 15: p=0.5 (null)
        p = [0.001] * 5 + [0.5] * 15
        result = lord_plus_plus(p, alpha=0.10)
        n_early = sum(result[:5])
        assert n_early >= 3, (
            f"Expected ≥ 3 of the 5 strong signals rejected, got {n_early}"
        )

    def test_no_discoveries_all_p_one(self):
        """p=1 for all → no rejections (alpha_j budget never sufficient for p=1)."""
        m = 10
        p = [1.0] * m
        result = lord_plus_plus(p, alpha=0.10)
        assert sum(result) == 0

    def test_mixed_signal_null(self):
        """Mix of strong signals and nulls; at least the strong ones are picked up."""
        # Alternate strong/null pattern; the strong ones should be mostly rejected
        rng = np.random.default_rng(99)
        signal_idx = set(range(0, 20, 2))   # even indices: strong
        p = [
            0.0005 if i in signal_idx else float(rng.uniform(0.3, 1.0))
            for i in range(20)
        ]
        result = lord_plus_plus(p, alpha=0.10)
        n_signal_found = sum(result[i] for i in signal_idx)
        assert n_signal_found >= 5, (
            f"Expected ≥ 5 strong signals found, got {n_signal_found}"
        )


# ---------------------------------------------------------------------------
# Gamma sequence sanity
# ---------------------------------------------------------------------------

class TestGammaSequence:
    def test_sums_to_one(self):
        gam = _gamma_sequence(100)
        assert abs(gam.sum() - 1.0) < 1e-10

    def test_non_negative(self):
        gam = _gamma_sequence(50)
        assert (gam >= 0).all()

    def test_decreasing(self):
        gam = _gamma_sequence(20)
        assert (np.diff(gam) <= 0).all(), "Gamma weights should be non-increasing"

    def test_length(self):
        for m in (1, 10, 100):
            assert len(_gamma_sequence(m)) == m


# ---------------------------------------------------------------------------
# AST parse check (ensure no syntax errors in the module under test)
# ---------------------------------------------------------------------------

def test_source_parses_cleanly():
    src = _SRC.read_text(encoding="utf-8")
    ast.parse(src)   # raises SyntaxError on failure
