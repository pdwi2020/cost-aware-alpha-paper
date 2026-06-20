"""
src/manifest.py — Result manifest helper for the CAVAL pipeline.

Every pipeline stage records its reported quantities here so the manuscript
can be generated from a single source (results/manifest/manifest.json).

Usage (in a pipeline stage):
    from src.manifest import record, get, load

    record("oos.trackB.net_sharpe", 0.357,
           stage="holdout", universe="pit_sp500_annual", track="B")
    value = get("oos.trackB.net_sharpe", track="B")
    all_entries = load()

Keys are dot-namespaced: e.g. "oos.trackB.net_sharpe",
"fdr.trackA.n_selected", "pbo.trackB.probability".

Atomic write (tmp + os.replace) ensures the manifest is never corrupted
by a partial write.

Run as a module for a smoke test:
    python3 src/manifest.py
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_DIR = _ROOT / "results" / "manifest"
_MANIFEST_FILE = _MANIFEST_DIR / "manifest.json"
_SPEC_FILE = _ROOT / "config" / "spec.yaml"


# ── internal helpers ─────────────────────────────────────────────────────────

def _spec_hash() -> str:
    """Return the git blob hash of config/spec.yaml, or 'uncommitted'."""
    try:
        result = subprocess.run(
            ["git", "hash-object", str(_SPEC_FILE)],
            capture_output=True, text=True, check=True,
            cwd=_ROOT,
        )
        return result.stdout.strip()
    except Exception:
        return "uncommitted"


def _load_raw() -> dict:
    """Load the manifest JSON, returning {} if it does not exist."""
    if not _MANIFEST_FILE.exists():
        return {}
    with open(_MANIFEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_raw(data: dict) -> None:
    """Atomically write the manifest JSON."""
    _MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=_MANIFEST_DIR, prefix=".manifest_tmp_", suffix=".json"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, _MANIFEST_FILE)
    except Exception:
        # Clean up tmp file on failure; don't suppress the original error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── public API ───────────────────────────────────────────────────────────────

def record(
    key: str,
    value: Any,
    *,
    stage: str,
    universe: str | None = None,
    track: str | None = None,
    meta: dict | None = None,
) -> None:
    """Append or update a manifest entry.

    Parameters
    ----------
    key:      Dot-namespaced identifier, e.g. "oos.trackB.net_sharpe".
    value:    Scalar (float/int/str) or list.  Floats are stored as-is.
    stage:    Pipeline stage name, e.g. "holdout", "fdr", "backtest".
    universe: Universe name from spec.yaml, e.g. "pit_sp500_annual".
    track:    "A" or "B" (or None for stage-level metrics).
    meta:     Optional free-form metadata dict.
    """
    data = _load_raw()
    entry = {
        "value": value,
        "stage": stage,
        "universe": universe,
        "track": track,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "spec_hash": _spec_hash(),
    }
    if meta:
        entry["meta"] = meta
    data[key] = entry
    _save_raw(data)


def get(key: str, **filters: Any) -> Any:
    """Return the value for *key*, applying optional field filters.

    Filters are matched against the entry fields (stage, universe, track, …).
    Returns None if the key is absent or filters don't match.
    """
    data = _load_raw()
    entry = data.get(key)
    if entry is None:
        return None
    for field, expected in filters.items():
        if entry.get(field) != expected:
            return None
    return entry["value"]


def load() -> dict:
    """Return the full manifest as a dict keyed by metric name.

    Each value is the full entry dict (value, stage, universe, track,
    timestamp, spec_hash, optional meta).
    """
    return _load_raw()


# ── smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== manifest.py smoke test ===")
    print(f"Manifest file: {_MANIFEST_FILE}")
    print(f"Spec hash:     {_spec_hash()}")

    # Record a dummy entry
    TEST_KEY = "_smoke_test.dummy_sharpe"
    TEST_VALUE = 1.234

    print(f"\nRecording {TEST_KEY!r} = {TEST_VALUE} ...")
    record(
        TEST_KEY, TEST_VALUE,
        stage="smoke_test",
        universe="pit_sp500_annual",
        track="B",
        meta={"note": "auto-deleted after smoke test"},
    )

    # Read it back via get()
    retrieved = get(TEST_KEY)
    assert retrieved == TEST_VALUE, f"get() returned {retrieved!r}, expected {TEST_VALUE!r}"
    print(f"  get({TEST_KEY!r}) -> {retrieved}  [OK]")

    # Read it back via load()
    manifest = load()
    assert TEST_KEY in manifest, "Key missing from load() result"
    entry = manifest[TEST_KEY]
    assert entry["stage"] == "smoke_test"
    assert entry["track"] == "B"
    print(f"  load()[{TEST_KEY!r}]['stage'] = {entry['stage']!r}  [OK]")
    print(f"  load()[{TEST_KEY!r}]['spec_hash'] = {entry['spec_hash']!r}  [OK]")

    # Clean up dummy entry so it doesn't pollute generated_numbers.tex
    data = load()
    del data[TEST_KEY]
    _save_raw(data)
    print(f"\nSmoke entry cleaned up from manifest.")
    print(f"Manifest now has {len(data)} entries.")
    print("\n=== PASSED ===")
    sys.exit(0)
