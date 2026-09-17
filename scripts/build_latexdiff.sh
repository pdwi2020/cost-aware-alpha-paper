#!/usr/bin/env bash
# build_latexdiff.sh — marked-up manuscript against the Array-submitted source.
#
# latexdiff cannot be run and compiled blind here. It wraps float-level markup
# (\DIFaddbeginFL ... \DIFaddendFL) around whatever changed, and when the change
# is the last row of a booktabs table, or the tabular preamble itself, the
# closing marker lands between the final \\ and \bottomrule. TeX then reports
# "Misplaced \noalign" and the build dies with ~100 errors. The markers are
# cosmetic; the table structure is not, so we relocate or drop the markers that
# straddle table structure and keep everything else.
#
#   bash scripts/build_latexdiff.sh
#
# Output: paper/latexdiff_vs_array.pdf

set -euo pipefail
cd "$(dirname "$0")/../paper"

OLD="main.tex.array_submitted.bak"
NEW="main.tex"
DIFF="latexdiff_vs_array.tex"
FLAT="latexdiff_new_flat.tex"

[ -f "$OLD" ] || { echo "ABORT: $OLD missing (the Array baseline)"; exit 1; }

# The generated tables live in \input files now, and the Array baseline has them
# inline. Diffing a 30-line table against a one-line \input tells the reviewer
# nothing and produces markup wrapped around whole table environments, which is
# the one thing the repair pass below cannot fix. Splice the generated tables in
# first, so both sides are compared row by row as they were before.
echo "[0/3] flattening generated tables into the new source"
python3 - "$NEW" "$FLAT" <<'FLATTEN'
import re, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_text()
def splice(m):
    name = m.group(1)
    f = src.parent / (name if name.endswith(".tex") else name + ".tex")
    # Only the generated tables; macro files must stay as \input so the
    # preamble still defines them once.
    if name.startswith("tab_") and f.exists():
        return f.read_text().rstrip("\n")
    return m.group(0)
text = re.sub(r"\\input\{([^}]+)\}", splice, text)
dst.write_text(text)
print(f"  wrote {dst.name}")
FLATTEN
NEW="$FLAT"

echo "[1/3] latexdiff"
latexdiff --disable-citation-markup \
          --append-safecmd="toprule,midrule,bottomrule,cmidrule,addlinespace" \
          "$OLD" "$NEW" > "$DIFF"

echo "[2/3] repairing markup that straddles table structure"
python3 - "$DIFF" <<'PY'
import re
import sys

path = sys.argv[1]
lines = open(path, encoding="utf-8").read().split("\n")

RULE = re.compile(r"^\\DIF(?:add|del)endFL\s+(\\(?:top|mid|bottom)rule.*)$")
BEGIN_TABULAR = re.compile(r"\\DIF(?:add|del)beginFL\s+(\\begin\{tabular\})")
# A wholly-new table has every line wrapped, rules included, and \toprule inside
# the argument of \DIFaddFL is again a \noalign inside a group. Hoist it out.
TRAPPED_RULE = re.compile(r"(\\DIF(?:add|del)FL\{)(\\(?:top|mid|bottom)rule)\s*")
COMMENT = re.compile(r"(?<!\\)%.*$")
moved = dropped = 0


def live(text: str) -> str:
    """The part of a line TeX actually reads."""
    return COMMENT.sub("", text)


# Whether a trapped rule can be hoisted depends on whether its table still
# exists. A table that was deleted outright has its \begin{tabular} commented
# out by latexdiff (%DIFDELCMD), so hoisting the rule into open text produces a
# \noalign with no alignment to sit in. Track live tabular depth and drop the
# rule in that case instead.
hoisted = dropped_rules = 0
depth = 0
out = []
for line in lines:
    before = depth
    visible = live(line)
    depth += visible.count(r"\begin{tabular}") - visible.count(r"\end{tabular}")
    if TRAPPED_RULE.search(line):
        if before > 0 or depth > 0:
            line = TRAPPED_RULE.sub(r"\2\n\1", line)
            hoisted += 1
        else:
            line = TRAPPED_RULE.sub(r"\1", line)
            dropped_rules += 1
    out.append(line)
lines = "\n".join(out).split("\n")
if hoisted or dropped_rules:
    print(f"  hoisted {hoisted} rule(s) out of DIF groups, "
          f"dropped {dropped_rules} in deleted tables")

# \multicolumn must be the first token in its cell. latexdiff happily puts a
# \DIFaddendFL, or an empty \DIFaddFL{}, in front of one, and TeX then reports
# "Misplaced \omit" from inside \multispan. Clear the markers that precede the
# first \multicolumn on a line; the markup is cosmetic, the alignment is not.
MC_PREFIX = re.compile(
    r"^((?:\s*(?:\\DIF(?:add|del)(?:begin|end)FL|\\DIF(?:add|del)FL\{\s*\}))+)\s*"
    r"(?=\\multicolumn)"
)
MC_INLINE = re.compile(r"\\DIF(?:add|del)FL\{\s*\}\s*(?=\\multicolumn)")
OPEN_EMPTY = re.compile(r"\\DIF(?:add|del)FL\{\s*$")
mc_fixed = 0
for i, line in enumerate(lines):
    if r"\multicolumn" not in line:
        continue
    # The empty group can straddle a line break: "\DIFaddFL{" ends one line and
    # the matching "}" opens the next, immediately before \multicolumn.
    if i and line.lstrip().startswith("}") and OPEN_EMPTY.search(lines[i - 1]):
        lines[i - 1] = OPEN_EMPTY.sub("", lines[i - 1])
        line = line.lstrip()[1:]
        lines[i] = line
        mc_fixed += 1
    new_line = MC_PREFIX.sub("", line)
    new_line = MC_INLINE.sub("", new_line)
    if new_line != line:
        lines[i] = new_line
        mc_fixed += 1
if mc_fixed:
    print(f"  cleared DIF markers before {mc_fixed} \\multicolumn cell(s)")

for i, line in enumerate(lines):
    m = RULE.match(line)
    if not m or i == 0:
        continue
    prev = lines[i - 1]
    if BEGIN_TABULAR.search(prev):
        # The markup opens on the tabular preamble. Wrapping \begin{tabular} in
        # a DIF group leaves the environment opened inside a group it never
        # closes, so drop the pair rather than relocate it.
        lines[i - 1] = BEGIN_TABULAR.sub(r"\1", prev)
        lines[i] = m.group(1)
        dropped += 1
    elif prev.rstrip().endswith(r"\\"):
        # Last row of the table body. Close the markup before the row
        # terminator instead of after it.
        marker = line.split(None, 1)[0]
        stripped = prev.rstrip()
        lines[i - 1] = stripped[:-2].rstrip() + " " + marker + r" \\"
        lines[i] = m.group(1)
        moved += 1

open(path, "w", encoding="utf-8").write("\n".join(lines))
print(f"  relocated {moved} marker(s), dropped {dropped} preamble marker(s)")

remaining = sum(1 for ln in lines if RULE.match(ln))
if remaining:
    sys.exit(f"  {remaining} marker(s) still straddle a rule; inspect {path}")
PY

echo "[3/3] compiling"
latexmk -pdf -interaction=nonstopmode "$DIFF" > /tmp/build_latexdiff.log 2>&1 || true

errors=$(grep -cE '^! ' /tmp/build_latexdiff.log || true)
if [ "$errors" -ne 0 ]; then
    echo "FAIL: $errors TeX error(s); see /tmp/build_latexdiff.log"
    grep -E '^! ' /tmp/build_latexdiff.log | head -5
    exit 1
fi

pages=$(pdfinfo latexdiff_vs_array.pdf | awk '/^Pages:/ {print $2}')
markers=$(grep -c 'DIFadd\|DIFdel' "$DIFF")
echo "OK: latexdiff_vs_array.pdf, $pages pages, $markers change markers"
