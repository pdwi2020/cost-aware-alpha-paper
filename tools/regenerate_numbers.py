#!/usr/bin/env python3
"""
tools/regenerate_numbers.py — Emit LaTeX macros from the results manifest.

Every number in the manuscript should come from here, because the alternative
is what the Array submission did: numbers typed by hand, which drifted apart
from the tables (a momentum Sharpe labelled as the wrong window, an in-sample
row still carrying superseded cost and turnover figures).

    \\input{generated_numbers}
    ... the in-sample net Sharpe is \\SynthesisTrackAIsNetSharpeWeekly.

What changed in v3
------------------
1. **Scalars only.** The old version ran ``str()`` over whatever it found, so a
   manifest entry holding a dict produced a macro containing a printed Python
   dict. Structured entries are now skipped and listed in the header, so they
   are visible rather than silently mangled.
2. **Formatting by quantity.** Sharpe ratios carry an explicit sign and three
   decimals, p-values collapse to ``< 0.001`` when tiny, percentages get two
   decimals, counts get thousands separators.
3. **Collisions are not silently resolved.** Two manifest keys that sanitise to
   the same macro name previously let "the last key win". Each now gets a
   distinct suffixed macro, and ``--strict`` exits non-zero.

Usage:
    python3 tools/regenerate_numbers.py
    python3 tools/regenerate_numbers.py --strict
    python3 tools/regenerate_numbers.py --manifest path/to/manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "results" / "manifest" / "manifest.json"
DEFAULT_OUT = ROOT / "paper" / "generated_numbers.tex"

DEFAULT_PRECISION = 4
SUFFIXES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

LATEX_ESCAPES = [
    ("\\", r"\textbackslash{}"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("&", r"\&"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
]


def sanitise_key(key: str) -> str:
    """Convert a dot-namespaced manifest key to a valid LaTeX macro name.

    A LaTeX control sequence may not contain digits, so digits are spelled out
    rather than deleted. Deleting them collapsed every ARV threshold onto one
    name (``arv_0p15``, ``arv_0p3``, ``arv_0p5`` and ``arv_0p75`` all became
    ``SensitivityTrackAArvp``) and likewise every capacity tier, so the macro
    for "ARV at the economic threshold" silently resolved to ARV at 0.15.

    ``capitalize()`` lowercases the rest of each part, so an inner capital does
    not survive; that quirk is kept because it is harmless once digits are.

        "oos.trackB.net_sharpe"      -> "OosTrackbNetSharpe"  (lowercase b)
        "fdr.track_a.n_selected"     -> "FdrTrackANSelected"
        "sensitivity.track_a.arv_0p5"-> "SensitivityTrackAArvZeroPFive"
    """
    digit_words = {
        "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
        "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
    }
    parts = re.split(r"[._\-]+", key)
    camel = "".join(p.capitalize() for p in parts if p)
    camel = "".join(digit_words.get(ch, ch) for ch in camel)
    camel = re.sub(r"[^A-Za-z]", "", camel)
    return camel if camel else "UnknownKey"


def _is_scalar(value) -> bool:
    return isinstance(value, (bool, int, float, str))


def format_value(value, key: str = "") -> str:
    """Format one scalar for LaTeX, choosing precision from what it measures."""
    k = key.lower()

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, int):
        return f"{value:,}" if abs(value) >= 10_000 else str(value)

    if isinstance(value, float):
        if math.isnan(value):
            return r"\textit{n/a}"
        if math.isinf(value):
            return r"$\infty$" if value > 0 else r"$-\infty$"

        # p-values: readers care about the threshold, not the digits.
        if re.search(r"(^|[._])p(value|_val)?($|[._])|pval|_p$", k):
            if 0 <= value < 0.001:
                return "< 0.001"
            return f"{value:.3f}"

        # Sharpe ratios and returns: sign matters, three decimals is plenty.
        if any(t in k for t in ("sharpe", "_sr", "sr_", "alpha", "ic_bar", "slope")):
            return f"{value:+.3f}"

        # Percentages and coverage.
        if any(t in k for t in ("pct", "percent", "coverage", "rate")):
            return f"{value:.2f}"

        if abs(value) < 1e6 and (value == 0 or abs(value) >= 1e-4):
            return f"{value:.{DEFAULT_PRECISION}f}"
        return f"{value:.{DEFAULT_PRECISION}e}"

    s = str(value)
    for char, replacement in LATEX_ESCAPES:
        s = s.replace(char, replacement)
    return s


# Macros that render a mean and its dispersion as one "a \pm b" cell. The
# per-key macros above cannot do this, and the alternative was a hand-typed
# table cell, which is how the MLP row came to hold a pre-v3 value while the
# rest of its table had moved on.
PAIRED_MACROS = {
    "MLPicA": ("models.track_a.mlp_ic_mean", "models.track_a.mlp_ic_std"),
    "MLPicB": ("models.track_b.mlp_ic_mean", "models.track_b.mlp_ic_std"),
}


def _paired_macros(manifest: dict, emitted: dict[str, str]) -> list[str]:
    """Emit "$mean \\pm std$" macros, or a macro that fails the LaTeX build.

    A missing input must not silently render an empty cell: an undefined
    control sequence stops the build, which is the behaviour we want when the
    stage that produces the number has not been run.
    """
    lines: list[str] = []
    for macro, (mean_key, std_key) in sorted(PAIRED_MACROS.items()):
        mean = manifest.get(mean_key, {})
        std = manifest.get(std_key, {})
        mean_v = mean.get("value") if isinstance(mean, dict) else mean
        std_v = std.get("value") if isinstance(std, dict) else std
        lines.append("")
        if mean_v is None or std_v is None:
            missing = mean_key if mean_v is None else std_key
            lines.append(f"% {macro}: NOT EMITTED — {missing} absent from the manifest.")
            lines.append(f"% Run the stage that writes it; \\{macro} stays undefined so the build fails.")
            continue
        rendered = f"${float(mean_v):.3f} \\pm {abs(float(std_v)):.3f}$"
        lines.append(f"% {macro}: {mean_key} +/- {std_key}")
        lines.append(f"\\providecommand{{\\{macro}}}{{{rendered}}}")
        lines.append(f"\\renewcommand{{\\{macro}}}{{{rendered}}}")
        emitted[macro] = f"{mean_key} +/- {std_key}"
    return lines


def build_lines(manifest: dict, manifest_path: Path) -> tuple[list[str], dict]:
    """Return the .tex lines plus a report of what was emitted and skipped."""
    emitted: dict[str, str] = {}
    skipped_structured: list[str] = []
    skipped_null: list[str] = []
    collisions: dict[str, list[str]] = {}
    body: list[str] = []

    for key in sorted(manifest.keys()):
        entry = manifest[key]
        value = entry.get("value") if isinstance(entry, dict) else entry

        if value is None:
            skipped_null.append(key)
            continue
        if not _is_scalar(value):
            skipped_structured.append(key)
            continue

        macro = sanitise_key(key)
        collisions.setdefault(macro, []).append(key)
        if len(collisions[macro]) > 1:
            # Distinct name rather than silent overwrite.
            macro = f"{macro}{SUFFIXES[min(len(collisions[macro]) - 1, len(SUFFIXES) - 1)]}"

        meta_bits = [f"key={key}"]
        if isinstance(entry, dict):
            if entry.get("stage"):
                meta_bits.append(f"stage={entry['stage']}")
            if entry.get("track"):
                meta_bits.append(f"track={entry['track']}")
            if entry.get("universe"):
                meta_bits.append(f"universe={entry['universe']}")
            if entry.get("spec_hash"):
                meta_bits.append(f"spec={str(entry['spec_hash'])[:8]}")

        rendered = format_value(value, key)
        body.append("")
        body.append(f"% {' | '.join(meta_bits)}")
        body.append(f"\\providecommand{{\\{macro}}}{{{rendered}}}")
        body.append(f"\\renewcommand{{\\{macro}}}{{{rendered}}}")
        emitted[macro] = key

    body.extend(_paired_macros(manifest, emitted))

    spec_hashes = {
        str(e.get("spec_hash"))[:8]
        for e in manifest.values()
        if isinstance(e, dict) and e.get("spec_hash")
    }

    header = [
        "% generated_numbers.tex — AUTO-GENERATED by tools/regenerate_numbers.py",
        f"% Source: {manifest_path}",
        f"% Generated: {datetime.now(timezone.utc).isoformat()}",
        f"% Spec hash(es): {', '.join(sorted(spec_hashes)) or 'none recorded'}",
        "% DO NOT EDIT BY HAND — re-run tools/regenerate_numbers.py",
        "%",
        r"% Usage:  \input{generated_numbers}  then  \SynthesisTrackAIsNetSharpeWeekly",
        "%",
        f"% Emitted {len(emitted)} macro(s).",
    ]
    if skipped_structured:
        header.append(
            f"% Skipped {len(skipped_structured)} structured entry/entries "
            "(dict or list values cannot be a macro):"
        )
        header.extend(f"%   - {k}" for k in skipped_structured)
    if skipped_null:
        header.append(f"% Skipped {len(skipped_null)} null-valued entry/entries.")

    report = {
        "n_emitted": len(emitted),
        "skipped_structured": skipped_structured,
        "skipped_null": skipped_null,
        "collisions": {m: ks for m, ks in collisions.items() if len(ks) > 1},
    }
    return header + body + ["", "% end of generated_numbers.tex"], report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate paper/generated_numbers.tex from the manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero on macro-name collisions or an empty manifest.")
    args = parser.parse_args(argv)

    if not args.manifest.exists():
        print(f"[WARN] Manifest not found: {args.manifest}", file=sys.stderr)
        manifest: dict = {}
    else:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    lines, report = build_lines(manifest, args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] Wrote {report['n_emitted']} macro(s) to {args.out}")
    if report["skipped_structured"]:
        print(f"[INFO] Skipped {len(report['skipped_structured'])} structured entries "
              "(listed in the file header).")
    if report["skipped_null"]:
        print(f"[INFO] Skipped {len(report['skipped_null'])} null-valued entries.")
    if report["collisions"]:
        print("[WARN] Macro-name collisions (each key got a suffixed macro):")
        for macro, keys in report["collisions"].items():
            print(f"  \\{macro} <- {keys}")

    if args.strict:
        if report["collisions"]:
            print("[FAIL] --strict: collisions present.", file=sys.stderr)
            return 2
        if report["n_emitted"] == 0:
            print("[FAIL] --strict: no macros emitted.", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
