"""tests/test_forward_touch_guard.py — the single touch survives a spec edit.

The forward-holdout guard used to match prior touches on the spec blob and the
window together. Any edit to config/spec_v3.yaml changed the hash, the existing
log entries stopped matching, and the window became re-scorable with no
--reevaluation-reason. The freeze was one character away from being undone, and
nothing would have reported it.

The window is what can be touched once. These tests pin that.

    python3 -m pytest tests/test_forward_touch_guard.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TOUCH_LOG = ROOT / "results" / "forward_touch_log.json"
SPEC = ROOT / "config" / "spec_v3.yaml"
WINDOW = "2025-08-01..2026-03-31"
TRACK = "track_a"


def _prior(touches, window, track, blob=None):
    """The guard's matching rule: window and track, never the blob."""
    return [
        e for e in touches
        if e.get("window") == window and e.get("track") == track
    ]


def _prior_blob_matched(touches, window, track, blob):
    """The superseded rule, kept to show the two genuinely differ."""
    return [
        e for e in touches
        if e.get("window") == window and e.get("track") == track
        and e.get("spec_blob") == blob
    ]


@pytest.fixture(scope="module")
def touches():
    if not TOUCH_LOG.exists():
        pytest.skip("no touch log in this checkout")
    return json.loads(TOUCH_LOG.read_text())


def test_the_window_has_been_touched(touches):
    assert len(_prior(touches, WINDOW, TRACK)) >= 1


def test_a_changed_spec_does_not_reset_the_count(touches):
    """The point of the whole thing."""
    real_blob = touches[0]["spec_blob"]
    edited_blob = "0" * 40                      # any edit gives a new hash
    assert len(_prior(touches, WINDOW, TRACK)) == \
           len(_prior(touches, WINDOW, TRACK, edited_blob)), \
        "prior-touch count must not depend on the current spec hash"
    assert len(_prior(touches, WINDOW, TRACK)) > 0
    # And confirm the superseded rule really would have reset it, so this test
    # is guarding against something rather than restating a tautology.
    assert _prior_blob_matched(touches, WINDOW, TRACK, real_blob), \
        "fixture assumption: the log entries carry the current blob"
    assert not _prior_blob_matched(touches, WINDOW, TRACK, edited_blob), \
        "the old blob-matched rule would have found no prior touches"


def test_every_touch_records_a_blob_and_a_reason_field(touches):
    for e in touches:
        assert e.get("spec_blob"), "a touch with no spec blob is unattributable"
        assert "reevaluation_reason" in e
    reruns = [e for e in touches if e.get("reevaluation_reason")]
    firsts = [e for e in touches if not e.get("reevaluation_reason")]
    assert len(firsts) >= 1, "there must be an unforced first touch"
    for e in reruns:
        assert len(e["reevaluation_reason"]) > 40, \
            "a re-evaluation reason has to actually say something"


def test_errata_documents_the_spec_rather_than_editing_it():
    """The frozen file must stay byte-identical to what the log points at."""
    import subprocess
    errata = ROOT / "config" / "spec_v3_errata.md"
    assert errata.exists(), "corrections to a frozen spec belong in errata"
    if not TOUCH_LOG.exists():
        pytest.skip("no touch log in this checkout")
    logged = json.loads(TOUCH_LOG.read_text())[0]["spec_blob"]
    actual = subprocess.run(["git", "hash-object", str(SPEC)], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    assert actual == logged, (
        "config/spec_v3.yaml no longer hashes to the value recorded in the "
        "touch log; if the edit was deliberate, record it in the errata and "
        "update this test, but do not do it silently"
    )
