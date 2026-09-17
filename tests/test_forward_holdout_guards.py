"""tests/test_forward_holdout_guards.py — the single-touch enforcement.

A single-touch claim is only worth the guards behind it, so the guards are
tested: the window rule, the frozen-weight construction, the git-blob hash and
the touch log. No network, no X9; parquet inputs are synthesised in tmp_path.

    python3 -m pytest tests/test_forward_holdout_guards.py -q
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import run_forward_holdout as fh  # noqa: E402


SPEC_STUB = {
    "windows": {"forward_holdout": {"start": "2025-08-01", "end": "2026-08-31"}},
    "validation": {"economic_threshold": {"sr_star": 0.5}},
}


class TestWindowRule:
    """Composites using 1-minute-derived features get the truncated window."""

    def test_full_span_for_a_daily_computable_composite(self):
        start, end, note, intraday = fh.resolve_window(
            ["mom_12_1", "ret_252d", "overnight_gap"], SPEC_STUB
        )
        assert (start, end) == ("2025-08-01", "2026-08-31")
        assert intraday == []
        assert "daily-computable" in note

    def test_truncated_when_an_intraday_feature_is_selected(self):
        start, end, note, intraday = fh.resolve_window(
            ["mom_12_1", "vwap_dev"], SPEC_STUB
        )
        assert start == "2025-08-01"
        assert end == fh.INTRADAY_LAST_DATE == "2026-03-31"
        assert intraday == ["vwap_dev"]
        assert "secondary" in note

    def test_overnight_gap_does_not_trigger_the_rule(self):
        """overnight_gap comes from daily open/prev-close, not 1-minute bars."""
        assert "overnight_gap" not in fh.INTRADAY_FEATURES
        _, end, _, intraday = fh.resolve_window(["overnight_gap"], SPEC_STUB)
        assert end == "2026-08-31" and intraday == []


class TestFrozenWeights:
    """Weights come from committed in-sample artifacts, never forward data."""

    @pytest.fixture
    def artifacts(self, tmp_path, monkeypatch):
        fdr = pd.DataFrame({
            "track": ["track_a"] * 4,
            "feature": ["a", "b", "c", "d"],
            "ic_bar": [-0.12, 0.08, -0.04, 0.02],
            "bh_rejected": [True, True, True, False],
            "bhy_rejected": [True, True, False, False],
        })
        shap = pd.DataFrame({
            "track": ["track_a"] * 3,
            "feature": ["a", "b", "c"],
            "mean_abs_shap": [0.6, 0.3, 0.1],
        })
        fdr_path = tmp_path / "fdr.parquet"
        shap_path = tmp_path / "shap.parquet"
        fdr.to_parquet(fdr_path)
        shap.to_parquet(shap_path)
        monkeypatch.setattr(fh, "FDR_PATH", fdr_path)
        monkeypatch.setattr(fh, "SHAP_PATH", shap_path)
        return fdr_path, shap_path

    def test_only_bh_selected_features_with_signed_normalised_weights(self, artifacts):
        w, selected = fh.frozen_weights("track_a")
        assert selected == ["a", "b", "c"]          # "d" is not BH-selected
        assert np.isclose(w.abs().sum(), 1.0)
        assert w["a"] < 0 and w["c"] < 0 and w["b"] > 0   # signs follow ic_bar
        assert w.abs()["a"] > w.abs()["b"] > w.abs()["c"]  # SHAP magnitudes

    def test_refuses_when_nothing_is_selected(self, tmp_path, monkeypatch):
        fdr = pd.DataFrame({
            "track": ["track_b"] * 2,
            "feature": ["a", "b"],
            "ic_bar": [0.01, -0.01],
            "bh_rejected": [False, False],
            "bhy_rejected": [False, False],
        })
        shap = pd.DataFrame({"track": ["track_b"], "feature": ["a"], "mean_abs_shap": [0.1]})
        fdr_path, shap_path = tmp_path / "f.parquet", tmp_path / "s.parquet"
        fdr.to_parquet(fdr_path)
        shap.to_parquet(shap_path)
        monkeypatch.setattr(fh, "FDR_PATH", fdr_path)
        monkeypatch.setattr(fh, "SHAP_PATH", shap_path)
        with pytest.raises(RuntimeError, match="no BH-selected features"):
            fh.frozen_weights("track_b")


class TestSpecHashAndTouchLog:

    def test_blob_hash_matches_gits_definition(self, tmp_path):
        content = b"spec: value\n"
        path = tmp_path / "spec.yaml"
        path.write_bytes(content)
        expected = hashlib.sha1(
            b"blob " + str(len(content)).encode() + b"\0" + content
        ).hexdigest()
        assert fh.spec_blob_hash(path) == expected

    def test_touch_log_appends(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fh, "TOUCH_LOG", tmp_path / "touch.json")
        assert fh.load_touch_log() == []
        fh.record_touch({"spec_blob": "abc", "window": "2025-08-01..2026-08-31"})
        fh.record_touch({"spec_blob": "abc", "window": "2025-08-01..2026-08-31"})
        entries = fh.load_touch_log()
        assert len(entries) == 2
        assert entries[0]["spec_blob"] == "abc"

    def test_uncommitted_spec_is_detected(self, tmp_path):
        """A file outside git is not a pre-commitment."""
        path = tmp_path / "spec_v3.yaml"
        path.write_text("meta: {}\n")
        ok, why = fh.spec_is_committed_and_clean(path)
        assert ok is False
        assert "not tracked" in why or "uncommitted" in why
