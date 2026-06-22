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

Beyond the loose token scan, a set of *headline assertions* cross-checks the
manuscript's FDR summary-table counts (BH-IC and TA-FDR rejections for each
track, plus the /N feature-set denominator) against the authoritative
manifest values. A mismatch there is a genuine contradiction and exits 2,
independent of --strict. This catches manifest-vs-manuscript drift (e.g. a
cross-universe run overwriting the primary FDR counts) that loose token
matching cannot — a wrong number still "matches" if it appears anywhere else.

Exit codes:
    0 — all headline assertions pass (and no unmatched numbers under --strict)
    1 — unmatched numbers found (only with --strict)
    2 — a headline assertion FAILED: the manuscript contradicts the manifest
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


# ---------------------------------------------------------------------------
# Headline assertions — structured, semantic cross-checks that specific
# manuscript claims equal the authoritative manifest values.  Unlike the loose
# token scan above, a failure here is a genuine contradiction (e.g. the manifest
# says Track~A rejects 12 features but the paper's summary table shows a
# different number), so it ALWAYS forces a non-zero exit, regardless of
# --strict.  This guards against the manifest-vs-manuscript drift that loose
# token matching cannot catch (a wrong number still "matches" if it happens to
# appear somewhere else in the manifest).
# ---------------------------------------------------------------------------

def load_manifest_raw(manifest_path: Path) -> dict:
    """Load the manifest as its raw {key: entry} dict (empty if absent)."""
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _manifest_number(manifest: dict, key: str, subpath: str | None = None):
    """Return a numeric value from a manifest entry, or None if absent.

    Without ``subpath`` the entry's ``value`` field is returned; with a dotted
    ``subpath`` (e.g. ``"value.n_features"``) the nested field is returned.
    """
    entry = manifest.get(key)
    if entry is None:
        return None
    if subpath:
        node = entry
        for part in subpath.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node
    if isinstance(entry, dict):
        return entry.get("value")
    return entry


# Candidate manifest keys for the pre-registered feature-set size (the "/N"
# denominator in the FDR summary table). First match wins.
FEATURE_COUNT_KEYS = [
    ("ta_fdr.track_a.summary", "value.n_features"),
    ("features.final_list", "meta.n_features"),
]

# Each assertion: (label, manifest_key, subpath, anchor_regex).  The regex must
# capture the numerator as group 1 and (optionally) the denominator as group 2,
# on the SAME manuscript line as the label (no DOTALL — avoids cross-line false
# matches).
HEADLINE_ASSERTIONS = [
    ("BH-IC rejections (Track A)", "fdr.track_a.n_selected_bh", None,
     r"BH-IC rejections\s*\(Track~A\).*?(\d+)\s*/\s*(\d+)"),
    ("BH-IC rejections (Track B)", "fdr.track_b.n_selected_bh", None,
     r"BH-IC rejections\s*\(Track~B\).*?(\d+)\s*/\s*(\d+)"),
    ("TA-FDR rejections (Track A)", "tafdr.track_a.n_selected_bh", None,
     r"TA-FDR rejections\s*\(Track~A\).*?(\d+)\s*/\s*(\d+)"),
    ("TA-FDR rejections (Track B)", "tafdr.track_b.n_selected_bh", None,
     r"TA-FDR rejections\s*\(Track~B\).*?(\d+)\s*/\s*(\d+)"),
]


def _feature_count(manifest: dict):
    for key, sub in FEATURE_COUNT_KEYS:
        v = _manifest_number(manifest, key, sub)
        if v is not None:
            return int(v)
    return None


def check_headline_assertions(
    tex_text: str, manifest: dict
) -> list[tuple[str, str, str]]:
    """Cross-check headline FDR counts in the manuscript against the manifest.

    Returns a list of ``(label, status, detail)`` where ``status`` is one of
    ``"PASS"``, ``"FAIL"`` (a real contradiction), or ``"WARN"`` (the manifest
    key or the manuscript anchor was not found, so the claim could not be
    checked — advisory, non-fatal).
    """
    results: list[tuple[str, str, str]] = []
    n_features = _feature_count(manifest)
    for label, key, sub, pattern in HEADLINE_ASSERTIONS:
        expected = _manifest_number(manifest, key, sub)
        if expected is None:
            results.append((label, "WARN", f"manifest key '{key}' absent"))
            continue
        expected = int(expected)
        m = re.search(pattern, tex_text)
        if not m:
            results.append((label, "WARN", "anchor not found in manuscript"))
            continue
        shown_num = int(m.group(1))
        detail = f"manuscript shows {shown_num}, manifest says {expected}"
        if shown_num != expected:
            results.append((label, "FAIL", detail))
            continue
        if m.lastindex and m.lastindex >= 2 and n_features is not None:
            shown_den = int(m.group(2))
            if shown_den != n_features:
                results.append((
                    label, "FAIL",
                    f"{detail}; denominator {shown_den} != n_features {n_features}",
                ))
                continue
            detail += f"; denom {shown_den}=={n_features}"
        results.append((label, "PASS", detail))
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

    # --- Headline assertion cross-checks (hard guard against manifest drift) ---
    manifest_raw = load_manifest_raw(args.manifest)
    tex_text = args.tex.read_text(encoding="utf-8", errors="replace")
    assertions = check_headline_assertions(tex_text, manifest_raw)
    n_fail = sum(1 for _, s, _ in assertions if s == "FAIL")
    n_warn = sum(1 for _, s, _ in assertions if s == "WARN")

    print()
    print("=" * 60)
    print("Headline FDR assertions (manuscript vs manifest)")
    print("=" * 60)
    marks = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN"}
    for label, status, detail in assertions:
        print(f"  [{marks[status]}] {label}: {detail}")
    if n_fail:
        print(f"\n[ASSERT] {n_fail} headline assertion(s) FAILED — "
              f"the manuscript contradicts the manifest.")
    elif n_warn:
        print(f"\n[ASSERT] checked assertions pass ({n_warn} warning(s) — "
              f"unverifiable, see above).")
    else:
        print("\n[ASSERT] all headline FDR counts match the manifest.")

    # A headline contradiction is a real error: fail hard regardless of --strict.
    if n_fail:
        print(f"[EXIT 2] {n_fail} headline assertion failure(s).")
        return 2
    if args.strict and n_unmatched > 0:
        print(f"[STRICT] Exiting 1 because {n_unmatched} unmatched numbers found.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
