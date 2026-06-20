#!/usr/bin/env python3
"""
tools/check_coherence.py — Numeric coherence checker for the manuscript.

Scans paper/main.tex for numeric tokens and reports any that do NOT appear
in results/manifest/manifest.json.  This makes it easy to track which numbers
in the paper are manifest-backed vs still hand-entered.

Expected output today: many unmatched numbers (that's fine — this is a
tracking tool, not a blocker).

Usage:
    python3 tools/check_coherence.py
    python3 tools/check_coherence.py --tex paper/main.tex
    python3 tools/check_coherence.py --strict          # exit 1 if unmatched
    python3 tools/check_coherence.py --show 20         # show first 20 unmatched

Exit codes:
    0 — all numbers matched (or --strict not set)
    1 — unmatched numbers found (only with --strict)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEX = ROOT / "paper" / "main.tex"
DEFAULT_MANIFEST = ROOT / "results" / "manifest" / "manifest.json"

# Numeric token pattern: integers and decimals (with optional leading minus/plus)
# Excludes pure year patterns like 2010, 2024 to reduce noise (configurable below)
NUMERIC_RE = re.compile(
    r"""
    (?<![A-Za-z\\])          # not preceded by a letter or backslash
    ([+-]?                    # optional sign
     (?:
       \d{1,3}(?:,\d{3})+    # comma-formatted: 1,234,567
       |\d+\.\d+             # decimal: 3.14, 0.357
       |\d{1,9}              # plain integer (up to 9 digits to skip LaTeX IDs)
     )
    )
    (?![A-Za-z_%])           # not followed by a letter, % or _
    """,
    re.VERBOSE,
)

# Year pattern — commonly skipped as they appear in data ranges, citations, etc.
YEAR_RE = re.compile(r"^(19|20)\d{2}$")

# LaTeX comment lines to skip
COMMENT_RE = re.compile(r"^\s*%")

# Very small or large integers that are structural LaTeX (e.g. line numbers,
# references) — tune SKIP_INTEGERS to control noise
SKIP_INTEGERS: set[str] = set()  # could add {"0","1","2"} to suppress trivially


def load_manifest_values(manifest_path: Path) -> set[str]:
    """Extract all numeric values from the manifest as normalised strings."""
    if not manifest_path.exists():
        return set()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    values: set[str] = set()
    for entry in manifest.values():
        v = entry.get("value") if isinstance(entry, dict) else entry
        if v is None:
            continue
        if isinstance(v, (int, float)):
            values.add(_normalise_number(str(v)))
            # Also add rounded variants
            if isinstance(v, float):
                for prec in [0, 1, 2, 3, 4]:
                    values.add(_normalise_number(f"{v:.{prec}f}"))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, (int, float)):
                    values.add(_normalise_number(str(item)))
    return values


def _normalise_number(s: str) -> str:
    """Normalise a numeric string for comparison (remove commas, leading zeros)."""
    s = s.strip().lstrip("+")
    s = s.replace(",", "")
    # Convert to float and back to remove trailing zeros for consistent comparison
    try:
        f = float(s)
        # Keep reasonable precision
        return f"{f:.6g}"
    except ValueError:
        return s


def extract_tex_numbers(tex_path: Path) -> list[tuple[int, str, str]]:
    """Return list of (line_number, raw_token, normalised) from the .tex file."""
    results: list[tuple[int, str, str]] = []
    with open(tex_path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            # Skip comment lines
            if COMMENT_RE.match(line):
                continue
            # Remove inline comments (everything after unescaped %)
            line_no_comment = re.sub(r"(?<!\\)%.*$", "", line)
            for m in NUMERIC_RE.finditer(line_no_comment):
                token = m.group(1)
                norm = _normalise_number(token)
                # Skip years
                if YEAR_RE.match(token.lstrip("+-")):
                    continue
                # Skip trivially small ints if configured
                if token in SKIP_INTEGERS:
                    continue
                results.append((lineno, token, norm))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check numeric coherence between manuscript and manifest."
    )
    parser.add_argument(
        "--tex",
        type=Path,
        default=DEFAULT_TEX,
        help=f"Path to the LaTeX manuscript (default: {DEFAULT_TEX})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to manifest.json (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any unmatched numbers are found.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=30,
        help="Maximum number of unmatched entries to display (default: 30).",
    )
    args = parser.parse_args()

    if not args.tex.exists():
        print(f"[ERROR] TeX file not found: {args.tex}", file=sys.stderr)
        return 1

    manifest_values = load_manifest_values(args.manifest)
    tex_numbers = extract_tex_numbers(args.tex)

    total = len(tex_numbers)
    matched: list[tuple[int, str]] = []
    unmatched: list[tuple[int, str]] = []

    for lineno, token, norm in tex_numbers:
        if norm in manifest_values:
            matched.append((lineno, token))
        else:
            unmatched.append((lineno, token))

    n_matched = len(matched)
    n_unmatched = len(unmatched)
    pct_matched = (100.0 * n_matched / total) if total > 0 else 0.0

    print("=" * 60)
    print("CAVAL Numeric Coherence Report")
    print("=" * 60)
    print(f"  Manuscript:        {args.tex}")
    print(f"  Manifest:          {args.manifest}")
    print(f"  Manifest entries:  {len(manifest_values)}")
    print(f"  Total tex numbers: {total}")
    print(f"  Manifest-backed:   {n_matched} ({pct_matched:.1f}%)")
    print(f"  Hand-entered:      {n_unmatched} ({100.0 - pct_matched:.1f}%)")
    print("=" * 60)

    if unmatched:
        show_n = min(args.show, n_unmatched)
        print(f"\nFirst {show_n} unmatched numbers (hand-entered):")
        print(f"  {'Line':>6}  {'Token':<20}")
        print(f"  {'----':>6}  {'-----':<20}")
        for lineno, token in unmatched[:show_n]:
            print(f"  {lineno:>6}  {token:<20}")
        if n_unmatched > show_n:
            print(f"  ... and {n_unmatched - show_n} more (use --show N to see more)")
    else:
        print("\nAll numbers in manuscript are manifest-backed. Excellent!")

    print()
    print(f"[SUMMARY] {n_matched}/{total} numbers are manifest-backed; "
          f"{n_unmatched} are hand-entered.")

    if args.strict and n_unmatched > 0:
        print(f"[STRICT] Exiting 1 because {n_unmatched} unmatched numbers found.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
