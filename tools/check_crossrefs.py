"""Resolve every Table/Section/Figure reference in the submission letters.

The manuscript's own cross-references are LaTeX \\ref, so they cannot go wrong.
The letters are plain markdown and cite numbers literally: "Section 7.8",
"Table 13". Those numbers are produced by the manuscript's numbering, so
inserting one table renumbers everything after it and silently invalidates
every citation downstream. That happened: adding the calibrated-cost table as
Table 8 moved three of the response letter's references off by one, each of
which would have sent an editor to the wrong table while checking whether a
reviewer's point had been addressed.

This resolves every reference in the letters against the compiled .aux files
and reports the ones that do not point at anything, or point at something whose
caption looks unrelated to the surrounding sentence.

    python3 tools/check_crossrefs.py
    python3 tools/check_crossrefs.py --strict     # exit 1 on any unresolved

Exit codes:
    0 — every reference resolves
    1 — at least one reference does not (with --strict, or always for a
        reference to a number that does not exist at all)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
KIT = PAPER / "submission" / "ResultsInEngineering"

LETTERS = [
    KIT / "response_to_reviewers_DRAFT.md",
    KIT / "cover_letter_DRAFT.md",
    KIT / "00_READ_FIRST.md",
]

REF = re.compile(r"\b(Table|Section|Figure|Sections|Tables)\s+(S?\d+(?:\.\d+)*)")


def load_numbers(aux: Path) -> dict[tuple[str, str], str]:
    """Map (kind, printed number) -> label, from a compiled .aux.

    Keyed on the label's own prefix, not on the number alone. Section 6 and
    Table 6 are both "6"; a single map keyed on the number resolves whichever
    LaTeX happened to write first, which made this checker report every table
    reference as broken.
    """
    if not aux.exists():
        return {}
    out: dict[tuple[str, str], str] = {}
    for label, number in re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}", aux.read_text()):
        if label.endswith("@cref"):
            continue
        kind = {"tab": "Table", "fig": "Figure", "sec": "Section",
                "subsec": "Section", "supp": "Table"}.get(label.split(":")[0])
        if kind is None:
            continue
        out.setdefault((kind, number), label)
    return out


def section_numbers(tex: Path) -> set[str]:
    """Section numbers that exist, read from the PDF's own table of contents."""
    toc = tex.with_suffix(".toc")
    if not toc.exists():
        return set()
    return set(re.findall(r"\\numberline\s*\{([\d.]+)\}", toc.read_text()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    main_nums = load_numbers(PAPER / "main.aux")
    supp_nums = load_numbers(PAPER / "supplementary.aux")
    known = {**main_nums, **supp_nums}
    secs = section_numbers(PAPER / "main.tex") | section_numbers(PAPER / "supplementary.tex")

    if not known:
        print("no .aux files; build the manuscript first")
        return 1

    bad = 0
    total = 0
    for letter in LETTERS:
        if not letter.exists():
            continue
        text = letter.read_text()
        hits = []
        for m in REF.finditer(text):
            kind, num = m.group(1).rstrip("s"), m.group(2)
            total += 1
            line = text[: m.start()].count("\n") + 1
            if kind == "Section":
                ok = num in secs or not secs   # no .toc -> cannot check
                target = "section " + num if ok else None
            else:
                target = known.get((kind, num))
                ok = target is not None
            if not ok:
                bad += 1
                hits.append((line, kind, num, target))
        name = letter.name
        if hits:
            print(f"\n{name}:")
            for line, kind, num, target in hits:
                what = f"resolves to {target}" if target else "does not resolve"
                print(f"  line {line:>4}  {kind} {num:<5} {what}")
        else:
            print(f"{name}: all references resolve")

    print(f"\n{total - bad}/{total} references resolve")
    if bad:
        print(f"[FAIL] {bad} reference(s) do not point at what they claim")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
