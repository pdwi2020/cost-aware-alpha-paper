"""tests/test_pbo_selection_path.py — the CSCV candidate set and DSR.

Pure-function tests for the selection-path grid that replaces the old
single-feature PBO (Reviewer 1, point 5). No data files needed.

    python3 -m pytest tests/test_pbo_selection_path.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fdr.run_pbo_selection_path import (  # noqa: E402
    REBAL_FREQS,
    SELECTION_RULES,
    WEIGHTINGS,
    deflated_sharpe,
    selected_features,
    signal_weights,
)


@pytest.fixture
def track_fdr():
    """Six features: three BH-significant, two of those BHY-significant."""
    return pd.DataFrame({
        "feature": ["a", "b", "c", "d", "e", "f"],
        "ic_bar": [-0.12, 0.09, -0.06, 0.03, -0.02, 0.01],
        "bh_rejected": [True, True, True, False, False, False],
        "bhy_rejected": [True, True, False, False, False, False],
    })


@pytest.fixture
def shap_summary():
    return pd.Series({"a": 0.5, "b": 0.3, "c": 0.2, "d": 0.1, "e": 0.05})
    # note: "f" deliberately missing, to exercise the fallback


class TestSelectionRules:

    def test_bh_and_bhy(self, track_fdr):
        assert selected_features("bh", track_fdr) == ["a", "b", "c"]
        assert selected_features("bhy", track_fdr) == ["a", "b"]

    def test_all(self, track_fdr):
        assert selected_features("all", track_fdr) == list("abcdef")

    def test_top_k_ranks_by_absolute_ic(self, track_fdr):
        assert selected_features("top5", track_fdr) == ["a", "b", "c", "d", "e"]
        # negative ICs must rank by magnitude, not by signed value
        assert selected_features("top20", track_fdr)[0] == "a"

    def test_unknown_rule_raises(self, track_fdr):
        with pytest.raises(ValueError, match="Unknown selection rule"):
            selected_features("nonsense", track_fdr)

    def test_grid_is_the_documented_size(self):
        assert len(SELECTION_RULES) * len(WEIGHTINGS) * len(REBAL_FREQS) == 54


class TestWeighting:

    def test_weights_are_l1_normalised_and_signed(self, track_fdr, shap_summary):
        for scheme in WEIGHTINGS:
            w = signal_weights(["a", "b", "c"], scheme, track_fdr, shap_summary)
            assert np.isclose(w.abs().sum(), 1.0), scheme
            # signs follow sign(ic_bar): a and c negative, b positive
            assert w["a"] < 0 and w["c"] < 0 and w["b"] > 0, scheme

    def test_equal_weighting_has_equal_magnitudes(self, track_fdr, shap_summary):
        w = signal_weights(["a", "b", "c"], "equal", track_fdr, shap_summary)
        assert np.allclose(w.abs().to_numpy(), 1.0 / 3.0)

    def test_shap_weighting_follows_shap_magnitudes(self, track_fdr, shap_summary):
        w = signal_weights(["a", "b", "c"], "shap", track_fdr, shap_summary).abs()
        assert w["a"] > w["b"] > w["c"]

    def test_ic_weighting_follows_ic_magnitudes(self, track_fdr, shap_summary):
        w = signal_weights(["a", "b", "c"], "ic", track_fdr, shap_summary).abs()
        assert np.isclose(w["a"] / w["b"], 0.12 / 0.09, rtol=1e-6)

    def test_missing_shap_value_falls_back_to_the_mean(self, track_fdr, shap_summary):
        w = signal_weights(["a", "f"], "shap", track_fdr, shap_summary)
        assert np.isfinite(w).all()
        assert np.isclose(w.abs().sum(), 1.0)

    def test_empty_selection_returns_empty(self, track_fdr, shap_summary):
        assert signal_weights([], "shap", track_fdr, shap_summary).empty

    def test_unknown_scheme_raises(self, track_fdr, shap_summary):
        with pytest.raises(ValueError, match="Unknown weighting"):
            signal_weights(["a"], "nonsense", track_fdr, shap_summary)


class TestDeflatedSharpe:
    """DSR must fall as the search widens: that is the whole point."""

    @staticmethod
    def _series(mean, sd, T=2000, seed=0):
        return np.random.default_rng(seed).normal(mean, sd, T)

    def test_dsr_decreases_with_trial_count(self):
        r = self._series(0.0004, 0.01)
        dsr_few = deflated_sharpe(r, n_trials=2)["dsr"]
        dsr_many = deflated_sharpe(r, n_trials=500)["dsr"]
        assert dsr_few > dsr_many

    def test_strong_signal_survives_a_small_search(self):
        r = self._series(0.0015, 0.008)
        assert deflated_sharpe(r, n_trials=2)["dsr"] > 0.95

    def test_noise_fails_even_a_small_search(self):
        r = self._series(0.0, 0.01, seed=3)
        assert deflated_sharpe(r, n_trials=54)["dsr"] < 0.95

    def test_reports_annualised_sharpe_and_trials(self):
        out = deflated_sharpe(self._series(0.0005, 0.01), n_trials=54)
        assert out["n_trials"] == 54
        assert np.isclose(out["sr_ann"], out["sr_ann"])          # finite
        assert out["T"] == 2000

    def test_degenerate_input_is_nan_not_a_crash(self):
        assert np.isnan(deflated_sharpe(np.zeros(500), n_trials=10)["dsr"])
        assert np.isnan(deflated_sharpe(np.array([0.01, 0.02]), n_trials=10)["dsr"])
