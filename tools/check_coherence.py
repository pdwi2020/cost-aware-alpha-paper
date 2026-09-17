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
    # Regime rows of tab:regime_fdr. These went to print computed over an empty
    # calm sub-sample, because nothing tied the table to an authoritative
    # source. Each row is now pinned to the manifest. The regex reads the
    # "<n features> & <BH-rejected>" pair from the row, so group 1 is the
    # DENOMINATOR and group 2 the rejection count; hence the swapped order.
    ("Regime BH (Track A, calm)", "fdr.track_a.regime.calm.n_selected_bh", None,
     r"Track~A\s*&\s*Calm\s*\(VIX\$\\leq\s*20\$\)\s*&\s*(\d+)\s*&\s*(\d+)",
     "swapped"),
    ("Regime BH (Track A, stressed)", "fdr.track_a.regime.stressed.n_selected_bh", None,
     r"Track~A\s*&\s*Stressed\(VIX\$>\s*20\$\)\s*&\s*(\d+)\s*&\s*(\d+)",
     "swapped"),
    ("Regime BH (Track B, calm)", "fdr.track_b.regime.calm.n_selected_bh", None,
     r"Track~B\s*&\s*Calm\s*\(VIX\$\\leq\s*20\$\)\s*&\s*(\d+)\s*&\s*(\d+)",
     "swapped"),
    ("Regime BH (Track B, stressed)", "fdr.track_b.regime.stressed.n_selected_bh", None,
     r"Track~B\s*&\s*Stressed\(VIX\$>\s*20\$\)\s*&\s*(\d+)\s*&\s*(\d+)",
     "swapped"),
]


# Headline *magnitudes*, as opposed to the rejection counts above. These are
# what drifted between the Array submission and its tables: a Sharpe ratio
# quoted in three places, updated in one. Each entry is
# (label, manifest key, subpath, regex capturing the number, decimals).
# The manuscript value is compared to the manifest value rounded to the same
# number of decimals, so "-0.196" matches -0.19625805.
SCALAR_ASSERTIONS = [
    ("IS net Sharpe (weekly, deployed)",
     "backtest.track_a.is_net_sharpe_weekly", None,
     r"IS net Sharpe is~\$\+([0-9.]+)\$", 3),
    ("IS gross Sharpe (weekly)",
     "backtest.track_a.is_gross_sharpe_weekly", None,
     r"IS net Sharpe is~\$\+[0-9.]+\$ \(gross~\$\+([0-9.]+)\$\)", 3),
    ("Exploratory 2022-2024 net Sharpe",
     "window.exploratory_2022_2024.track_a.net_sharpe", None,
     r"exploratory 2022--2024 window returns a net Sharpe of~\$\\mathbf\{(-[0-9.]+)\}\$", 3),
    ("Locked OOS 2025 net Sharpe",
     "window.locked_oos_2025.track_a.net_sharpe", None,
     r"locked OOS net Sharpe of~\$(-[0-9.]+)\$\s*\n?cannot be distinguished", 3),
    ("Locked OOS 2025 bootstrap p",
     "window.locked_oos_2025.track_a.net_sharpe_boot_p", None,
     r"\$t = -[0-9.]+\$, \$p = ([0-9.]+)\$\)", 2),
    ("Locked OOS 2025 CI lower",
     "window.locked_oos_2025.track_a.net_sharpe_ci95_lo", None,
     r"The 95\\% interval runs from~\$(-[0-9.]+)\$", 2),
    ("Locked OOS 2025 CI upper",
     "window.locked_oos_2025.track_a.net_sharpe_ci95_hi", None,
     r"The 95\\% interval runs from~\$-[0-9.]+\$\s*\n?to~\$\+([0-9.]+)\$", 2),
    ("Selection-path PBO",
     "pbo_selection_path.track_a.selection_path_54.pbo", None,
     r"A PBO of \$([0-9.]+)\$ sits", 3),
    ("Selection-path PBO (results section)",
     "pbo_selection_path.track_a.selection_path_54.pbo", None,
     r"\\PBO = \\mathbf\{([0-9.]+)\}\$: the IS-best", 3),
    ("Deployed DSR",
     "pbo_selection_path.track_a.selection_path_54.deployed_dsr", None,
     r"Deflated Sharpe Ratio of~\$([0-9.]+)\$ at the", 3),
    ("Ex-mega-cap exploratory net Sharpe",
     "battery.excl_megacap.track_a.oos_net_sharpe", None,
     r"ex-mega-cap~\$(-[0-9.]+)\$,\s*\n?mega-cap-only", 3),
    ("Forward window net Sharpe",
     "forward.track_a.net_sharpe", None,
     r"returns a net Sharpe of\s*\n?\$\\mathbf\{(-[0-9.]+)\}\$ \(gross", 3),
    ("Forward window days",
     "forward.track_a.n_days", None,
     r"Over ([0-9]+) trading days the frozen composite", 0),
    ("Ensemble IC (Track A)",
     "models.track_a.ensemble_ic_mean", None,
     r"ensemble averages \$\\mathbf\{([0-9.]+)\}\$ across the", 3),

    # tab:rebal. Every cell of this table was hand-typed and every cell drifted:
    # it still held pre-correction values after two full re-runs, while the
    # prose around it had been updated. Anchor each row on its own manifest key.
    ("tab:rebal daily turnover",
     "backtest.track_a.is_annual_turnover_daily", None,
     r"Daily  \(1d\) & ([0-9.]+) &", 1),
    ("tab:rebal daily gross Sharpe",
     "backtest.track_a.is_gross_sharpe_daily", None,
     r"Daily  \(1d\) & [0-9.]+ & \$\+([0-9.]+)\$", 3),
    ("tab:rebal daily net Sharpe",
     "backtest.track_a.is_net_sharpe_daily", None,
     r"Daily  \(1d\) & [0-9.]+ & \$\+[0-9.]+\$ & \$\+([0-9.]+)\$", 3),
    ("tab:rebal weekly turnover",
     "backtest.track_a.is_annual_turnover_weekly", None,
     r"Weekly \(5d\) & ([0-9.]+) &", 1),
    ("tab:rebal weekly gross Sharpe",
     "backtest.track_a.is_gross_sharpe_weekly", None,
     r"Weekly \(5d\) & [0-9.]+ & \$\\mathbf\{\+([0-9.]+)\}\$", 3),
    ("tab:rebal weekly net Sharpe",
     "backtest.track_a.is_net_sharpe_weekly", None,
     r"Weekly \(5d\) & [0-9.]+ & \$\\mathbf\{\+[0-9.]+\}\$ & \$\\mathbf\{\+([0-9.]+)\}\$", 3),
    ("tab:rebal monthly turnover",
     "backtest.track_a.is_annual_turnover_monthly", None,
     r"Monthly\(21d\)& ([0-9.]+) &", 1),
    ("tab:rebal monthly gross Sharpe",
     "backtest.track_a.is_gross_sharpe_monthly", None,
     r"Monthly\(21d\)& [0-9.]+ & \$\+([0-9.]+)\$", 3),
    ("tab:rebal monthly net Sharpe",
     "backtest.track_a.is_net_sharpe_monthly", None,
     r"Monthly\(21d\)& [0-9.]+ & \$\+[0-9.]+\$ & \$\+([0-9.]+)\$", 3),

    # tab:funnel. Same failure: the BHY row sat at 12 after the corr_* fix took
    # it to 13, and the search-adjusted row at 15 after it took 16.
    ("tab:funnel BH row (Track A)",
     "fdr.track_a.n_selected_bh", None,
     r"Screen 1: BH on daily cross-sectional IC  & ([0-9]+) &", 0),
    ("tab:funnel BH row (Track B)",
     "fdr.track_b.n_selected_bh", None,
     r"Screen 1: BH on daily cross-sectional IC  & [0-9]+ & ([0-9]+)", 0),
    ("tab:funnel BHY row (Track A)",
     "fdr.track_a.n_selected_bhy", None,
     r"\\quad under BHY \(dependence-robust\)       & ([0-9]+) &", 0),
    ("tab:funnel BHY row (Track B)",
     "fdr.track_b.n_selected_bhy", None,
     r"\\quad under BHY \(dependence-robust\)       & [0-9]+ & ([0-9]+)", 0),
]


def check_scalar_assertions(
    tex_text: str, manifest: dict
) -> list[tuple[str, str, str]]:
    """Cross-check headline magnitudes in the manuscript against the manifest."""
    results: list[tuple[str, str, str]] = []
    for label, key, sub, pattern, decimals in SCALAR_ASSERTIONS:
        expected = _manifest_number(manifest, key, sub)
        if expected is None:
            results.append((label, "WARN", f"manifest key '{key}' absent"))
            continue
        m = re.search(pattern, tex_text)
        if not m:
            results.append((label, "WARN", "anchor not found in manuscript"))
            continue
        shown = float(m.group(1))
        target = round(float(expected), decimals)
        detail = f"manuscript shows {shown}, manifest says {target}"
        if abs(shown - target) > 10 ** (-decimals) / 2:
            results.append((label, "FAIL", detail))
        else:
            results.append((label, "PASS", detail))
    return results


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
    for assertion in HEADLINE_ASSERTIONS:
        # A fifth element "swapped" marks a row whose regex captures the
        # denominator first (the regime table prints "n features & rejected").
        label, key, sub, pattern = assertion[:4]
        swapped = len(assertion) > 4 and assertion[4] == "swapped"
        expected = _manifest_number(manifest, key, sub)
        if expected is None:
            results.append((label, "WARN", f"manifest key '{key}' absent"))
            continue
        expected = int(expected)
        m = re.search(pattern, tex_text)
        if not m:
            results.append((label, "WARN", "anchor not found in manuscript"))
            continue
        num_group, den_group = (2, 1) if swapped else (1, 2)
        shown_num = int(m.group(num_group))
        detail = f"manuscript shows {shown_num}, manifest says {expected}"
        if shown_num != expected:
            results.append((label, "FAIL", detail))
            continue
        if m.lastindex and m.lastindex >= 2 and n_features is not None:
            shown_den = int(m.group(den_group))
            if shown_den != n_features:
                results.append((
                    label, "FAIL",
                    f"{detail}; denominator {shown_den} != n_features {n_features}",
                ))
                continue
            detail += f"; denom {shown_den}=={n_features}"
        results.append((label, "PASS", detail))
    return results


TA_FDR_ROW = re.compile(
    r"\\texttt\{([A-Za-z0-9\\_]+)\}\s*&\s*\$(-?[0-9.]+)\$\s*&\s*"
    r"\$([0-9.]+)\$\s*&\s*\$([0-9.]+)\$\s*&",
)


def check_ta_fdr_table(tex_text: str, manifest_raw: dict) -> list:
    """Pin every printed cell of the TA-FDR table to the run that produced it.

    The counts alone were pinned before, and they are 0/30 under both the valid
    and the invalid null, so the table went on printing the superseded run's
    per-feature p-values through a clean check. Each cell is compared here.
    """
    results = []
    entry = manifest_raw.get("ta_fdr.track_a.per_feature")
    if entry is None:
        return [("TA-FDR per-feature table", "WARN",
                 "ta_fdr.track_a.per_feature missing from the manifest")]
    by_feature = {r["feature"]: r for r in entry.get("value", [])}

    rows = TA_FDR_ROW.findall(tex_text)
    if not rows:
        return [("TA-FDR per-feature table", "WARN",
                 "no table rows matched; has the table format changed?")]

    for raw_name, mean_net, p_net, p_gross in rows:
        feature = raw_name.replace("\\_", "_")
        src = by_feature.get(feature)
        if src is None:
            continue  # a row from some other table that happens to match
        checks = (
            ("mean net (bps)", float(mean_net), src["mean_net"] * 1e4, 3),
            ("p_net", float(p_net), src["p_net"], 3),
            ("p_blind", float(p_gross), src["p_gross"], 3),
        )
        for what, shown, actual, decimals in checks:
            if round(shown, decimals) == round(actual, decimals):
                results.append((f"tab:ta_fdr {feature} {what}", "PASS",
                                f"manuscript shows {shown}, run says "
                                f"{round(actual, decimals)}"))
            else:
                results.append((f"tab:ta_fdr {feature} {what}", "FAIL",
                                f"manuscript shows {shown}, run says "
                                f"{round(actual, decimals)}"))
    return results


def read_tex_with_inputs(path: Path) -> str:
    """Return the manuscript with its \\input files spliced in.

    Assertions anchor on table rows, and the generated tables live in their own
    files. Reading main.tex alone left every generated table unguarded, which
    is how Table 6 kept printing a superseded vintage's p-values while the
    checker reported a clean run.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    for name in re.findall(r"^\s*\\input\{([^}]+)\}", text, flags=re.M):
        child = path.parent / (name if name.endswith(".tex") else name + ".tex")
        if child.exists():
            text += "\n% ---- spliced from " + child.name + " ----\n"
            text += child.read_text(encoding="utf-8", errors="replace")
    return text


GENERATED_TEX = [
    "tab_ta_fdr.tex", "tab_rebal.tex", "tab_oos.tex",
    "tab_capacity.tex", "tab_baselines.tex", "tab_search_adjusted.tex",
    "generated_numbers.tex", "ta_fdr_macros.tex", "search_adjusted_macros.tex",
]


def check_generated_freshness(tex_path: Path, manifest_path: Path) -> list:
    """Fail if a generated artefact predates the manifest it renders.

    Converting the hand-typed tables into generated ones removed the drift that
    the value assertions used to catch, and with it the assertions themselves,
    which now match nothing. The risk that remains is different: a re-run
    updates the manifest and nobody regenerates the tables, so the paper
    silently renders the previous vintage. That is what this checks.
    """
    results = []
    if not manifest_path.exists():
        return [("generated artefacts", "WARN", "no manifest to compare against")]
    m_mtime = manifest_path.stat().st_mtime
    for name in GENERATED_TEX:
        f = tex_path.parent / name
        if not f.exists():
            results.append((f"generated {name}", "WARN", "not present"))
            continue
        if f.stat().st_mtime >= m_mtime:
            results.append((f"generated {name}", "PASS", "newer than the manifest"))
        else:
            age = (m_mtime - f.stat().st_mtime) / 60.0
            results.append((f"generated {name}", "FAIL",
                            f"{age:.0f} min older than the manifest; "
                            f"re-run its generator"))
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
    tex_text = read_tex_with_inputs(args.tex)
    assertions = (check_headline_assertions(tex_text, manifest_raw)
                  + check_scalar_assertions(tex_text, manifest_raw)
                  + check_ta_fdr_table(tex_text, manifest_raw)
                  + check_generated_freshness(args.tex, args.manifest))
    n_fail = sum(1 for _, s, _ in assertions if s == "FAIL")
    n_warn = sum(1 for _, s, _ in assertions if s == "WARN")

    print()
    print("=" * 60)
    print("Headline assertions (manuscript vs manifest)")
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
        print("\n[ASSERT] all headline counts and magnitudes match the manifest.")

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
