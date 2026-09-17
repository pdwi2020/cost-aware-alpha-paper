#!/usr/bin/env bash
# Assemble the Results in Engineering submission kit and the guide review
# package, then prove both rebuild from a clean extraction.
#
# Written after the first kit shipped with a source tree that did not compile:
# four \input files and a figure were missing, and the figures directory held
# renamed PDFs rather than the .png files main.tex actually references. Nobody
# noticed because the kit was assembled by hand from a list. The list is now
# derived from the TeX itself, and a missing dependency is a hard error.
#
#   bash scripts/build_submission_kits.sh

set -uo pipefail
cd "$HOME/ML_Paper" || exit 1

PAPER="paper"
KIT="paper/submission/ResultsInEngineering"
STAMP="$(date +%Y-%m-%d)"
REVIEW_ZIP="$KIT/CAVAL_RiE_for_review_${STAMP}.zip"
SUBMIT_ZIP="$KIT/CAVAL_RiE_submission.zip"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

say() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { echo "FAIL: $*" >&2; exit 1; }

# --- 1. Derive what the manuscripts actually depend on --------------------
say "resolving TeX dependencies"
DEPS=$(python3 - <<'PY'
import re, pathlib
paper = pathlib.Path("paper")
want = set()
for root in ("main.tex", "supplementary.tex"):
    want.add(root)
    text = (paper / root).read_text()
    for name in re.findall(r"\\input\{([^}]+)\}", text):
        want.add(name if name.endswith(".tex") else name + ".tex")
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        # LaTeX resolves the extension; find whatever is on disk.
        if pathlib.Path(paper / name).exists():
            want.add(name)
            continue
        for ext in (".pdf", ".png", ".jpg", ".eps"):
            if (paper / (name + ext)).exists():
                want.add(name + ext)
                break
        else:
            raise SystemExit(f"MISSING FIGURE: {name}")
# The bibliography and its compiled form: Elsevier asks for the .bbl.
for extra in ("references.bib", "references_rie.bib",
              "main.bbl", "supplementary.bbl",
              "highlights.tex", "graphical_abstract.tex"):
    if (paper / extra).exists():
        want.add(extra)
print("\n".join(sorted(want)))
PY
) || fail "dependency resolution"

say "$(echo "$DEPS" | wc -l | tr -d ' ') source files"

# --- 2. Stage the source tree --------------------------------------------
SRC="$WORK/source"
mkdir -p "$SRC"
while IFS= read -r dep; do
    [ -n "$dep" ] || continue
    [ -f "$PAPER/$dep" ] || fail "declared dependency missing on disk: $dep"
    mkdir -p "$SRC/$(dirname "$dep")"
    cp "$PAPER/$dep" "$SRC/$dep"
done <<< "$DEPS"

# --- 3. Prove the staged source compiles on its own -----------------------
say "verifying the staged source rebuilds standalone"
VERIFY="$WORK/verify"
cp -R "$SRC" "$VERIFY"
for doc in main supplementary; do
    (cd "$VERIFY" && latexmk -pdf -interaction=nonstopmode "$doc.tex" \
        > "$WORK/$doc.build.log" 2>&1) || fail "$doc does not compile from the kit"
    pages=$(pdfinfo "$VERIFY/$doc.pdf" | awk '/^Pages/{print $2}')
    # Read LaTeX's own .log, which holds only the final pass. latexmk's stdout
    # accumulates every pass, so a first-pass warning that later resolves would
    # read as a failure here.
    undef=$(grep -c "undefined" "$VERIFY/$doc.log" || true)
    say "  $doc.pdf: $pages pages, $undef undefined-reference warnings"
    [ "$undef" -eq 0 ] || fail "$doc has undefined references in the kit build"
done

# --- 3b. The letters cite manuscript numbers literally --------------------
# Adding one table renumbers every table after it and silently invalidates the
# letters' citations. Three of them were wrong that way, each pointing an editor
# at the wrong table while checking whether a reviewer's point was addressed.
say "checking the letters' cross-references against the compiled manuscript"
python3 tools/check_crossrefs.py || fail "a letter cites a table, section or figure that does not resolve"

# --- 4. Refresh the kit directory's built artefacts -----------------------
say "refreshing built artefacts in $KIT"
for f in main.pdf supplementary.pdf latexdiff_vs_array.pdf; do
    [ -f "$PAPER/$f" ] || fail "$PAPER/$f not built"
    cp "$PAPER/$f" "$KIT/$f"
done
while IFS= read -r dep; do
    [ -n "$dep" ] || continue
    cp "$PAPER/$dep" "$KIT/$(basename "$dep")"
done <<< "$DEPS"

# --- 5. Zips --------------------------------------------------------------
say "building $REVIEW_ZIP"
REV="$WORK/review"
mkdir -p "$REV"
cp -R "$SRC" "$REV/source"
for f in 00_READ_FIRST.md 00_READ_FIRST.pdf 00_READ_FIRST.docx \
         main.pdf supplementary.pdf latexdiff_vs_array.pdf \
         response_to_reviewers.pdf response_to_reviewers.docx \
         cover_letter.pdf cover_letter.docx \
         highlights.pdf graphical_abstract.pdf \
         spec_v3.yaml forward_touch_log.json; do
    [ -f "$KIT/$f" ] || fail "review package: $f missing"
    cp "$KIT/$f" "$REV/$f"
done
rm -f "$REVIEW_ZIP"
(cd "$REV" && zip -q -r "$OLDPWD/$REVIEW_ZIP" .) || fail "review zip"

say "building $SUBMIT_ZIP"
SUB="$WORK/submit"
mkdir -p "$SUB"
cp "$SRC"/* "$SUB"/ 2>/dev/null
for f in main.pdf supplementary.pdf latexdiff_vs_array.pdf \
         response_to_reviewers.pdf response_to_reviewers.docx \
         response_to_reviewers_DRAFT.md \
         cover_letter.pdf cover_letter.docx cover_letter_DRAFT.md \
         highlights.pdf graphical_abstract.pdf \
         spec_v3.yaml forward_touch_log.json \
         README.md TASKS.md response_matrix.md prior_work_matrix.md \
         build_kit.sh; do
    [ -f "$KIT/$f" ] && cp "$KIT/$f" "$SUB/$f"
done
rm -f "$SUBMIT_ZIP"
(cd "$SUB" && zip -q -r "$OLDPWD/$SUBMIT_ZIP" .) || fail "submission zip"

# --- 6. Prove the shipped zips rebuild ------------------------------------
for z in "$REVIEW_ZIP" "$SUBMIT_ZIP"; do
    say "verifying $(basename "$z") from a clean extraction"
    X="$WORK/x$(basename "$z" .zip)"
    mkdir -p "$X"
    unzip -q "$z" -d "$X"
    D="$X"; [ -d "$X/source" ] && D="$X/source"
    (cd "$D" && latexmk -pdf -interaction=nonstopmode main.tex > /dev/null 2>&1) \
        || fail "$(basename "$z") does not rebuild"
    say "  OK: $(pdfinfo "$D/main.pdf" | awk '/^Pages/{print $2}') pages"
done

say "zips:"
ls -lh "$REVIEW_ZIP" "$SUBMIT_ZIP"
for z in "$REVIEW_ZIP" "$SUBMIT_ZIP"; do
    echo "  $(basename "$z")  $(shasum -a 256 "$z" | cut -c1-16)  $(unzip -l "$z" | tail -1 | awk '{print $2}') files"
done
say "DONE"
