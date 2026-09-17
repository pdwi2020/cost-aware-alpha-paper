"""tests/test_regenerate_numbers.py — manifest to LaTeX macros.

Every manuscript number should trace to a manifest key, so this emitter is the
gate that makes that possible: it must format by quantity, refuse to mangle
structured entries, and never let two keys silently share one macro.

    python3 -m pytest tests/test_regenerate_numbers.py -q
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.regenerate_numbers import (  # noqa: E402
    build_lines,
    format_value,
    main,
    sanitise_key,
)


class TestKeySanitisation:
    """Macro names must stay stable: existing references depend on them."""

    def test_known_mappings(self):
        # capitalize() lowercases the rest of each part, so an inner capital
        # does not survive. Pinned deliberately: changing it would silently
        # break every existing \Macro reference in the manuscript.
        assert sanitise_key("oos.trackB.net_sharpe") == "OosTrackbNetSharpe"
        assert sanitise_key("fdr.track_a.n_selected") == "FdrTrackANSelected"

    def test_digits_are_spelled_out_not_stripped(self):
        """Digits must survive as words, because a macro name cannot hold them.

        Stripping them collapsed distinct keys onto one macro: every ARV
        threshold became ``SensitivityTrackAArvp`` and every capacity tier
        became ``BacktestTrackACapacityAummNetSharpe``, so a reference meant
        for ARV at the economic threshold silently resolved to ARV at 0.15.
        """
        assert sanitise_key("ta_fdr.track_a.p95") == "TaFdrTrackAPNineFive"

    def test_distinct_numeric_keys_get_distinct_macros(self):
        names = {sanitise_key(f"sensitivity.track_a.arv_{s}")
                 for s in ("0", "0p15", "0p3", "0p5", "0p75")}
        assert len(names) == 5, f"collision among ARV thresholds: {sorted(names)}"

    def test_non_alphanumeric_is_still_stripped(self):
        assert sanitise_key("a.b-c_d") == "ABCD"

    def test_empty_key_has_a_fallback(self):
        assert sanitise_key("...") == "UnknownKey"


class TestFormatting:

    def test_sharpe_gets_a_sign_and_three_decimals(self):
        assert format_value(0.18812, "synthesis.track_a.is_net_sharpe") == "+0.188"
        assert format_value(-0.607, "synthesis.track_a.exploratory_net_sharpe") == "-0.607"

    def test_tiny_p_values_collapse(self):
        assert format_value(0.0000004, "fdr.track_a.p_value") == "< 0.001"
        assert format_value(0.615, "synthesis.track_a.locked_oos_pvalue") == "0.615"

    def test_percentages_get_two_decimals(self):
        assert format_value(96.2571, "universe.coverage_pct") == "96.26"

    def test_large_counts_get_separators(self):
        assert format_value(2119199, "universe.member_days") == "2,119,199"
        assert format_value(30, "features.n") == "30"

    def test_booleans_and_nan(self):
        assert format_value(True, "backtest.track_b.deployable") == "true"
        assert format_value(float("nan"), "x.y") == r"\textit{n/a}"

    def test_strings_are_latex_escaped(self):
        assert format_value("spec_hash & 100%", "note") == r"spec\_hash \& 100\%"


class TestBuildLines:

    def _manifest(self):
        return {
            "synthesis.track_a.is_net_sharpe": {"value": 0.188, "stage": "battery",
                                                "track": "track_a", "spec_hash": "abc12345"},
            "fdr.track_a.per_feature": {"value": {"a": 1, "b": 2}, "stage": "fdr"},
            "fdr.track_a.n_selected": {"value": 12, "stage": "fdr"},
            "nulled.entry": {"value": None, "stage": "fdr"},
        }

    def test_structured_entries_are_skipped_and_listed(self):
        lines, report = build_lines(self._manifest(), Path("m.json"))
        text = "\n".join(lines)
        assert report["skipped_structured"] == ["fdr.track_a.per_feature"]
        assert "fdr.track_a.per_feature" in text          # visible in the header
        assert "FdrTrackAPerFeature}" not in text          # but not emitted as a macro
        assert "{'a': 1" not in text                       # and never a printed dict

    def test_scalars_are_emitted_with_both_commands(self):
        lines, report = build_lines(self._manifest(), Path("m.json"))
        text = "\n".join(lines)
        assert r"\providecommand{\SynthesisTrackAIsNetSharpe}{+0.188}" in text
        assert r"\renewcommand{\SynthesisTrackAIsNetSharpe}{+0.188}" in text
        assert r"\renewcommand{\FdrTrackANSelected}{12}" in text
        assert report["n_emitted"] == 2
        assert report["skipped_null"] == ["nulled.entry"]

    def test_collisions_get_distinct_macros_not_silent_overwrite(self):
        manifest = {
            "fdr.track_a.n_selected": {"value": 12},
            "fdr.track-a.n-selected": {"value": 99},     # sanitises identically
        }
        lines, report = build_lines(manifest, Path("m.json"))
        text = "\n".join(lines)
        assert report["collisions"]
        # Both values survive under distinct macro names. Which key takes the
        # unsuffixed name follows sorted() order ("-" sorts before "_"), so the
        # test pins that both exist rather than which got which.
        assert r"\renewcommand{\FdrTrackANSelected}{99}" in text
        assert r"\renewcommand{\FdrTrackANSelectedB}{12}" in text

    def test_spec_hash_recorded_in_header(self):
        lines, _ = build_lines(self._manifest(), Path("m.json"))
        assert any("Spec hash(es): abc12345" in ln for ln in lines)


class TestCli:

    def test_writes_file_and_is_idempotent(self, tmp_path):
        manifest = {"a.b": {"value": 1.5, "stage": "s"}}
        mpath = tmp_path / "manifest.json"
        mpath.write_text(json.dumps(manifest))
        out = tmp_path / "generated.tex"

        assert main(["--manifest", str(mpath), "--out", str(out)]) == 0
        first = out.read_text().split("% Generated:")[1].split("\n", 1)[1]
        assert main(["--manifest", str(mpath), "--out", str(out)]) == 0
        second = out.read_text().split("% Generated:")[1].split("\n", 1)[1]
        assert first == second                    # only the timestamp differs

    def test_strict_fails_on_collisions(self, tmp_path):
        mpath = tmp_path / "m.json"
        mpath.write_text(json.dumps({"a.b_c": {"value": 1}, "a.b-c": {"value": 2}}))
        out = tmp_path / "g.tex"
        assert main(["--manifest", str(mpath), "--out", str(out), "--strict"]) == 2

    def test_strict_fails_on_empty_manifest(self, tmp_path):
        mpath = tmp_path / "m.json"
        mpath.write_text("{}")
        out = tmp_path / "g.tex"
        assert main(["--manifest", str(mpath), "--out", str(out), "--strict"]) == 2
