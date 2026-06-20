#!/usr/bin/env python3
"""
tools/check_anonymity.py — Anonymity scanner for ESWA double-blind review.

Scans a LaTeX manuscript for identifying tokens:
  - Author names (configurable list)
  - Email addresses
  - Phone numbers
  - ORCID identifiers
  - GitHub repository URLs (github.com/...)
  - Known repository handles (e.g. pdwi2020)
  - \\acknowledgements / \\thanks / \\section*{Acknowledgements} blocks

Prints each hit with line number and exits non-zero if any are found.

Usage:
    python3 tools/check_anonymity.py
    python3 tools/check_anonymity.py --tex paper/main.tex
    python3 tools/check_anonymity.py --tex paper/submission/main.tex

Exit codes:
    0 — no identifying tokens found (manuscript is clean)
    1 — one or more identifying tokens found
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEX = ROOT / "paper" / "main.tex"

# ── Configurable name list ────────────────────────────────────────────────────
# Add any author names / institutional identifiers that should be absent
# from the anonymised submission.  Case-insensitive matching.
AUTHOR_NAMES: list[str] = [
    "Paritosh",
    "Dwivedi",
    "Ramanathan",
    "Lakshmanan",
    "Vellore",           # affiliation leakage
]

# Known repository / account handles to flag
REPO_HANDLES: list[str] = [
    "pdwi2020",
]

# ── Regex patterns ────────────────────────────────────────────────────────────
EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
    re.IGNORECASE,
)

PHONE_RE = re.compile(
    r"""
    (?:Tel|Fax|Phone|Ph)\.?\s*:?\s*  # optional label
    (?:\+?\d[\d\s\-().]{7,20}\d)     # number body
    |
    (?:\+\d{1,3}[\s\-])?             # country code without label
    (?:\(?\d{3,5}\)?[\s\-])?         # area code
    \d{3,5}[\s\-]\d{4,5}            # main number
    """,
    re.VERBOSE | re.IGNORECASE,
)

ORCID_RE = re.compile(
    r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]",
    re.IGNORECASE,
)

GITHUB_RE = re.compile(
    r"github\.com/[A-Za-z0-9_\-./]+",
    re.IGNORECASE,
)

# LaTeX acknowledgements / thanks — flags presence of these environments
ACKNOWLEDGE_RE = re.compile(
    r"\\(acknowledgements?|thanks|section\*\s*\{Acknowledgements?)",
    re.IGNORECASE,
)


def build_name_re(names: list[str]) -> re.Pattern | None:
    if not names:
        return None
    pattern = "|".join(re.escape(n) for n in names)
    return re.compile(pattern, re.IGNORECASE)


def scan_tex(tex_path: Path, name_re: re.Pattern | None) -> list[tuple[int, str, str]]:
    """Return list of (lineno, category, matched_text) for all hits."""
    hits: list[tuple[int, str, str]] = []
    with open(tex_path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            # Do NOT skip comment lines — a common mistake is leaving author
            # info in commented-out blocks

            # Author names
            if name_re:
                for m in name_re.finditer(line):
                    hits.append((lineno, "AUTHOR_NAME", m.group()))

            # Emails
            for m in EMAIL_RE.finditer(line):
                hits.append((lineno, "EMAIL", m.group()))

            # Phone numbers
            for m in PHONE_RE.finditer(line):
                text = m.group().strip()
                # Skip false positives: pure digit sequences from math
                if len(re.sub(r"\D", "", text)) >= 7:
                    hits.append((lineno, "PHONE", text))

            # ORCID
            for m in ORCID_RE.finditer(line):
                hits.append((lineno, "ORCID", m.group()))

            # GitHub URLs
            for m in GITHUB_RE.finditer(line):
                hits.append((lineno, "GITHUB_URL", m.group()))

            # Known handles
            for handle in REPO_HANDLES:
                if handle.lower() in line.lower():
                    idx = line.lower().index(handle.lower())
                    hits.append((lineno, "REPO_HANDLE", line[idx:idx+len(handle)]))

            # Acknowledgements / thanks
            for m in ACKNOWLEDGE_RE.finditer(line):
                hits.append((lineno, "ACKNOWLEDGEMENTS", m.group()))

    return hits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Anonymity scanner for ESWA double-blind review."
    )
    parser.add_argument(
        "--tex",
        type=Path,
        default=DEFAULT_TEX,
        help=f"Path to the LaTeX manuscript (default: {DEFAULT_TEX})",
    )
    args = parser.parse_args()

    if not args.tex.exists():
        print(f"[ERROR] TeX file not found: {args.tex}", file=sys.stderr)
        return 1

    name_re = build_name_re(AUTHOR_NAMES)
    hits = scan_tex(args.tex, name_re)

    print("=" * 60)
    print("CAVAL Anonymity Check — ESWA Double-Blind")
    print("=" * 60)
    print(f"  File: {args.tex}")
    print(f"  Checking {len(AUTHOR_NAMES)} name(s): {', '.join(AUTHOR_NAMES)}")
    print(f"  Checking handles: {', '.join(REPO_HANDLES)}")
    print("=" * 60)

    if not hits:
        print("\n[CLEAN] No identifying tokens found. Manuscript is anonymous.")
        return 0

    # Group by category for readability
    by_category: dict[str, list[tuple[int, str]]] = {}
    for lineno, cat, text in hits:
        by_category.setdefault(cat, []).append((lineno, text))

    print(f"\n[FAIL] {len(hits)} identifying token(s) found:\n")

    category_labels = {
        "AUTHOR_NAME":      "Author names",
        "EMAIL":            "Email addresses",
        "PHONE":            "Phone numbers",
        "ORCID":            "ORCID identifiers",
        "GITHUB_URL":       "GitHub URLs",
        "REPO_HANDLE":      "Repository handles",
        "ACKNOWLEDGEMENTS": "Acknowledgements/thanks blocks",
    }

    for cat, occurrences in sorted(by_category.items()):
        label = category_labels.get(cat, cat)
        print(f"  [{cat}] {label} — {len(occurrences)} occurrence(s):")
        for lineno, text in occurrences:
            # Truncate long matches for readability
            display = text[:80] + ("..." if len(text) > 80 else "")
            print(f"    line {lineno:>4}: {display}")
        print()

    print("-" * 60)
    print(f"[SUMMARY] {len(hits)} identifying token(s) across "
          f"{len(by_category)} category/categories.")
    print("[ACTION]  Remove or anonymise these before submitting to ESWA.")
    print("          Commonly needed:")
    print("          - Replace author block with \\author{{Anonymous}}")
    print("          - Remove/move \\acknowledgements to a separate file")
    print("          - Replace github.com/pdwi2020/... with [REPO ANONYMISED]")
    print("          - Remove ORCID and phone from corresponding-author block")

    return 1  # always exit 1 when hits are found


if __name__ == "__main__":
    sys.exit(main())
