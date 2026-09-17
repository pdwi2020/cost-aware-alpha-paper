"""Compare two manifest snapshots and print what moved.

Used when a pipeline-wide correction lands: the question is never "did anything
change" but "which reported quantities changed, and by how much", because every
one of them has to be chased into the manuscript.

    python3 tools/diff_manifest.py before.json [after.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AFTER = ROOT / "results" / "manifest" / "manifest.json"


def value_of(entry):
    return entry.get("value") if isinstance(entry, dict) else entry


def main() -> int:
    if len(sys.argv) < 2:
        return print(__doc__) or 2
    before = json.loads(Path(sys.argv[1]).read_text())
    after_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_AFTER
    after = json.loads(after_path.read_text())

    keys = sorted(set(before) | set(after))
    changed, added, removed = [], [], []
    for k in keys:
        if k not in after:
            removed.append(k)
            continue
        if k not in before:
            added.append(k)
            continue
        b, a = value_of(before[k]), value_of(after[k])
        if isinstance(b, (int, float)) and isinstance(a, (int, float)) \
                and not isinstance(b, bool) and not isinstance(a, bool):
            if abs(a - b) > 1e-9:
                rel = (a - b) / abs(b) * 100 if b else float("inf")
                changed.append((k, b, a, rel))
        elif b != a:
            changed.append((k, b, a, None))

    print(f"changed: {len(changed)}   added: {len(added)}   removed: {len(removed)}\n")
    for k, b, a, rel in changed:
        if rel is None:
            print(f"  {k}\n      {b!r}\n   -> {a!r}")
        else:
            print(f"  {k:<64} {b:>12.4f} -> {a:>12.4f}  ({rel:+.1f}%)")
    if added:
        print("\nadded:")
        for k in added:
            print(f"  {k} = {value_of(after[k])!r}")
    if removed:
        print("\nremoved:")
        for k in removed:
            print(f"  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
