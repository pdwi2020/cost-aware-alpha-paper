"""Search-adjusted false-discovery-rate sensitivity analysis.

Reviewer motivation:

    "The 30-feature set is described as pre-registered, but duplicate features
    and eight macro features were removed after evaluation, Polymarket
    variables were screened out, and an earlier analytic test and permutation
    procedure were replaced during a pipeline rebuild. These tried
    specifications are absent from the FDR denominator."

This module answers that concern with three analyses, each run separately for
Track A and Track B at q=0.10: reproduction of the frozen 30-feature BH/BHY
family, BH/BHY on an enlarged family containing every feature-level hypothesis
recorded in ``results/search_ledger.csv``, and a Harvey--Liu--Zhu-style
``|t| > 3`` hurdle on the original features.

The reproduction check verifies that recomputing BH/BHY from the stored
bootstrap p-values reproduces the stored rejection flags, which holds on any
panel. Specific counts are a property of a given panel, so they live in the
optional pin file ``config/expected_fdr_counts.json`` rather than in the code.

Two exact sign-flip duplicates borrow their source feature's two-sided
bootstrap p-value. Broadcast macros receive p=1 because daily cross-sectional
demeaning removes them by construction. The 14 Polymarket candidates and the
three other non-broadcast/non-duplicate candidates (``term_spread_x_mom``,
``vol_am``, and ``vol_pm``) also receive the conservative p=1 default because
no p-value under the current daily-cross-sectional-IC estimand is recoverable.

For the HLZ hurdle, ``t_stat`` is primary because it is the input file's own
analytic mean/SE t-statistic on the daily IC series, directly operationalizing
the screen. ``t_boot`` is an independent sanity check implied by the reported
90% bootstrap CI and an approximate-normal conversion.

Inputs:
    data/processed/fdr_results.parquet
    results/search_ledger.csv
Outputs:
    data/processed/fdr_search_adjusted.parquet
    results/staging/fdr_search_adjusted.json

Run:
    python3 -u src/fdr/run_search_adjusted_fdr.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.fdr.bh_correction import benjamini_hochberg, bhy_procedure


FDR_RESULTS = ROOT / "data" / "processed" / "fdr_results.parquet"
SEARCH_LEDGER = ROOT / "results" / "search_ledger.csv"
OUT_PARQUET = ROOT / "data" / "processed" / "fdr_search_adjusted.parquet"
OUT_JSON = ROOT / "results" / "staging" / "fdr_search_adjusted.json"

Q = 0.10
TRACKS = ("track_a", "track_b")
M_FROZEN = 30
EXPECTED_N_EXTRA = 27
EXPECTED_LEDGER_ROWS = 52
EXTRA_STATUSES = {"dropped", "never_built", "superseded"}
Z_90 = 1.6448536269514722

DUPLICATE_SOURCES = {
    "reversal_1w": "ret_5d",
    "reversal_4w": "ret_21d",
}
BROADCAST_MACROS = {
    "vix",
    "vix_chg_5d",
    "term_spread",
    "term_spread_chg_21d",
    "credit_proxy",
    "credit_proxy_chg_5d",
    "dxy_ret_5d",
    "wti_ret_21d",
}
CONSERVATIVE_DEFAULTS = {
    "term_spread_x_mom",
    "vol_am",
    "vol_pm",
    "total_trades",
    "buy_trades",
    "sell_trades",
    "total_usd_volume",
    "total_token_volume",
    "avg_price",
    "price_std",
    "min_price",
    "max_price",
    "vwap",
    "buy_sell_usd_ratio",
    "active_days",
    "first_trade_ts",
    "last_trade_ts",
}
EXPECTED_EXTRA_ITEMS = (
    set(DUPLICATE_SOURCES) | BROADCAST_MACROS | CONSERVATIVE_DEFAULTS
)

OUTPUT_COLUMNS = [
    "track",
    "variant",
    "feature",
    "is_ledger_extra",
    "p_value",
    "t_stat",
    "t_boot",
    "bh_adj_p",
    "bh_rejected",
    "bhy_adj_p",
    "bhy_rejected",
    "hlz_survivor",
]


def _tripwire(message: str) -> None:
    """Print a loud diagnostic and stop rather than emit invalid results."""
    print("\n*** SEARCH-ADJUSTED FDR HARD TRIP-WIRE FAILURE ***", file=sys.stderr)
    print(message, file=sys.stderr)
    print("No search-adjusted outputs were written.\n", file=sys.stderr)
    raise RuntimeError(message)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the frozen-family results and historical ledger."""
    fdr = pd.read_parquet(FDR_RESULTS)
    ledger = pd.read_csv(SEARCH_LEDGER)

    required_fdr = {
        "track",
        "feature",
        "ic_bar",
        "t_stat",
        "boot_p",
        "ci_low",
        "ci_high",
        "bh_rejected",
        "bhy_rejected",
    }
    missing = sorted(required_fdr - set(fdr.columns))
    if missing:
        _tripwire(f"Missing columns in {FDR_RESULTS}: {missing}")

    required_ledger = [
        "item",
        "category",
        "n_hypotheses_added",
        "status",
        "stage_evaluated",
        "evidence",
        "date",
    ]
    if ledger.columns.tolist() != required_ledger:
        _tripwire(
            "Ledger columns changed: expected "
            f"{required_ledger}, got {ledger.columns.tolist()}"
        )
    if len(ledger) != EXPECTED_LEDGER_ROWS:
        _tripwire(
            f"Ledger row count changed: expected {EXPECTED_LEDGER_ROWS}, "
            f"got {len(ledger)}"
        )

    for track in TRACKS:
        track_rows = fdr.loc[fdr["track"] == track]
        if len(track_rows) != M_FROZEN:
            _tripwire(
                f"Frozen-family size changed for {track}: expected "
                f"{M_FROZEN}, got {len(track_rows)}"
            )
        if track_rows["feature"].duplicated().any():
            duplicated = sorted(
                track_rows.loc[track_rows["feature"].duplicated(), "feature"]
            )
            _tripwire(f"Duplicate frozen features for {track}: {duplicated}")

    return fdr, ledger


def build_extra_p_values(
    track_fdr: pd.DataFrame,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    """Assign the specification-mandated p-value to every ledger extra.

    Duplicate reversals borrow a two-sided bootstrap p-value from the matching
    return feature. All other extras receive exactly 1.0, either by
    cross-sectional construction or as the documented conservative default.
    """
    extra_rows = ledger.loc[
        ledger["status"].isin(EXTRA_STATUSES)
        & (ledger["n_hypotheses_added"] > 0)
    ].copy()

    if not (extra_rows["n_hypotheses_added"] == 1).all():
        bad = extra_rows.loc[
            extra_rows["n_hypotheses_added"] != 1,
            ["item", "n_hypotheses_added"],
        ].to_dict("records")
        _tripwire(f"Extra rows must each contribute one hypothesis: {bad}")

    actual_items = set(extra_rows["item"])
    if actual_items != EXPECTED_EXTRA_ITEMS:
        _tripwire(
            "Ledger extra items changed: missing "
            f"{sorted(EXPECTED_EXTRA_ITEMS - actual_items)}, unexpected "
            f"{sorted(actual_items - EXPECTED_EXTRA_ITEMS)}"
        )

    source_p = track_fdr.set_index("feature")["boot_p"]
    records = []
    for item in extra_rows["item"]:
        if item in DUPLICATE_SOURCES:
            source = DUPLICATE_SOURCES[item]
            if source not in source_p.index:
                _tripwire(f"Missing duplicate source feature: {source}")
            p_value = float(source_p.loc[source])
        else:
            p_value = 1.0
        records.append({"feature": item, "p_value": p_value})

    return pd.DataFrame(records, columns=["feature", "p_value"])


def compute_t_boot(track_fdr: pd.DataFrame) -> pd.Series:
    """Infer a bootstrap-CI-based t-statistic for each frozen feature."""
    width = track_fdr["ci_high"] - track_fdr["ci_low"]
    se_boot = width / (2.0 * Z_90)
    return track_fdr["ic_bar"] / se_boot.where(se_boot > 0.0)


def _run_correction(p_values: np.ndarray) -> dict[str, np.ndarray]:
    """Run the repository's BH and BHY implementations at the fixed q."""
    bh_rejected, bh_adj_p = benjamini_hochberg(p_values, q=Q)
    bhy_rejected, bhy_adj_p = bhy_procedure(p_values, q=Q)
    return {
        "bh_rejected": bh_rejected,
        "bh_adj_p": bh_adj_p,
        "bhy_rejected": bhy_rejected,
        "bhy_adj_p": bhy_adj_p,
    }


def _validate_frozen_reproduction(
    track: str,
    track_fdr: pd.DataFrame,
    corrected: dict[str, np.ndarray],
) -> None:
    """Apply the frozen-family file and headline-count trip-wires."""
    for method in ("bh", "bhy"):
        column = f"{method}_rejected"
        expected = track_fdr[column].to_numpy(dtype=bool)
        actual = corrected[column]
        if not np.array_equal(actual, expected):
            _tripwire(
                f"{track} {method.upper()} reproduction mismatch: file has "
                f"{int(expected.sum())} rejections, recomputation has "
                f"{int(actual.sum())}; expected features="
                f"{sorted(track_fdr.loc[expected, 'feature'])}; actual "
                f"features={sorted(track_fdr.loc[actual, 'feature'])}"
            )

    # The invariant above (recomputation must match the stored flags) is
    # universal. Absolute counts are NOT: they are properties of a particular
    # panel, and hard-coding the Array-era values (12/11/0) turned a rebuild of
    # the universe into a crash. They are now an OPTIONAL pin: create
    # config/expected_fdr_counts.json once a run's counts are accepted, and any
    # later drift trips the wire.
    pins_path = ROOT / "config" / "expected_fdr_counts.json"
    if not pins_path.exists():
        return
    pins = json.loads(pins_path.read_text()).get(track, {})
    for method, expected_count in pins.items():
        if method not in corrected:
            continue
        actual_count = int(corrected[method].sum())
        if actual_count != int(expected_count):
            _tripwire(
                f"{track} {method.removesuffix('_rejected').upper()} count "
                f"changed: pinned {expected_count} in {pins_path.name}, got "
                f"{actual_count}. Update the pin deliberately if the change is real."
            )


def _fdr_rows(
    track: str,
    variant: str,
    features: list[str],
    p_values: np.ndarray,
    corrected: dict[str, np.ndarray],
    t_stat: np.ndarray,
    t_boot: np.ndarray,
    is_extra: np.ndarray,
) -> list[dict]:
    """Build output records for a frozen or enlarged FDR variant."""
    rows = []
    for index, feature in enumerate(features):
        rows.append(
            {
                "track": track,
                "variant": variant,
                "feature": feature,
                "is_ledger_extra": bool(is_extra[index]),
                "p_value": float(p_values[index]),
                "t_stat": float(t_stat[index]) if np.isfinite(t_stat[index]) else None,
                "t_boot": float(t_boot[index]) if np.isfinite(t_boot[index]) else None,
                "bh_adj_p": float(corrected["bh_adj_p"][index]),
                "bh_rejected": bool(corrected["bh_rejected"][index]),
                "bhy_adj_p": float(corrected["bhy_adj_p"][index]),
                "bhy_rejected": bool(corrected["bhy_rejected"][index]),
                "hlz_survivor": None,
            }
        )
    return rows


def _hlz_rows(
    track: str,
    track_fdr: pd.DataFrame,
    corrected: dict[str, np.ndarray],
    t_boot: np.ndarray,
) -> list[dict]:
    """Build output records for the HLZ hurdle variant."""
    rows = []
    for index, row in track_fdr.reset_index(drop=True).iterrows():
        t_stat = float(row["t_stat"])
        rows.append(
            {
                "track": track,
                "variant": "hlz_t3",
                "feature": row["feature"],
                "is_ledger_extra": False,
                "p_value": float(row["boot_p"]),
                "t_stat": t_stat,
                "t_boot": (
                    float(t_boot[index]) if np.isfinite(t_boot[index]) else None
                ),
                "bh_adj_p": float(corrected["bh_adj_p"][index]),
                "bh_rejected": bool(corrected["bh_rejected"][index]),
                "bhy_adj_p": float(corrected["bhy_adj_p"][index]),
                "bhy_rejected": bool(corrected["bhy_rejected"][index]),
                "hlz_survivor": bool(abs(t_stat) > 3.0),
            }
        )
    return rows


def _json_safe(obj):
    """Recursively replace non-finite floats with None so the summary is JSON.

    NaN and +/-Infinity have no JSON representation. json.dump would emit them
    as bare NaN/Infinity literals, which strict parsers reject, and with
    allow_nan=False it raises instead, which killed the run after the parquet
    had already been written. A missing statistic is real information, so it
    becomes null rather than disappearing or aborting the stage.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _find_t_stat_disagreements(
    track: str,
    track_fdr: pd.DataFrame,
    t_boot: np.ndarray,
) -> list[dict]:
    """Return threshold or relative-magnitude disagreements for t-statistics."""
    disagreements = []
    for index, row in track_fdr.reset_index(drop=True).iterrows():
        t_stat = float(row["t_stat"])
        t_boot_value = float(t_boot[index])
        if not np.isfinite(t_boot_value):
            disagreements.append(
                {
                    "track": track,
                    "feature": row["feature"],
                    "t_stat": t_stat,
                    "t_boot": None,
                    "threshold_disagreement": None,
                    "relative_magnitude_difference": None,
                    "reason": "non-positive or non-finite bootstrap-CI width",
                }
            )
            continue

        threshold_disagreement = (abs(t_stat) > 3.0) != (
            abs(t_boot_value) > 3.0
        )
        denominator = max(abs(t_stat), np.finfo(float).eps)
        relative_difference = abs(abs(t_boot_value) - abs(t_stat)) / denominator
        if threshold_disagreement or relative_difference > 0.20:
            disagreements.append(
                {
                    "track": track,
                    "feature": row["feature"],
                    "t_stat": t_stat,
                    "t_boot": t_boot_value,
                    "threshold_disagreement": bool(threshold_disagreement),
                    "relative_magnitude_difference": float(relative_difference),
                }
            )
    return disagreements


def _category_status_counts(ledger: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return deterministic nested ledger counts by category and status."""
    counts: dict[str, dict[str, int]] = {}
    grouped = ledger.groupby(["category", "status"], sort=True).size()
    for (category, status), count in grouped.items():
        counts.setdefault(str(category), {})[str(status)] = int(count)
    return counts


def run_analysis(
    fdr: pd.DataFrame,
    ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Compute all variants and return the row-level and summary outputs."""
    n_extra = int(
        ledger.loc[
            ledger["status"].isin(EXTRA_STATUSES), "n_hypotheses_added"
        ].sum()
    )
    if n_extra != EXPECTED_N_EXTRA:
        _tripwire(
            f"Ledger extra-hypothesis count changed: expected "
            f"{EXPECTED_N_EXTRA}, got {n_extra}"
        )
    m_prime = M_FROZEN + n_extra

    summary = {
        "ledger_summary": {
            "n_rows": int(len(ledger)),
            "by_category_status": _category_status_counts(ledger),
            "m": M_FROZEN,
            "n_extra": n_extra,
            "m_prime": m_prime,
        },
        "frozen_family": {},
        "enlarged_family": {},
        "hlz_hurdle": {},
        "t_stat_disagreements": [],
        "notes": [
            "The 14 Polymarket and 3 other non-broadcast/non-duplicate extra "
            "hypotheses (term_spread_x_mom, vol_am, vol_pm) were assigned the "
            "conservative p=1.0 default because no computable p-value exists "
            "under the current daily-cross-sectional-IC estimand; see "
            "results/search_ledger.csv evidence column for why.",
            "The Russell 2000 full re-instantiation "
            "(russell2000_full_reinstantiation, in_final, "
            "n_hypotheses_added=0) is a fully separate "
            "universe/feature-set/BH family and is NOT folded into m' here -- "
            "this is a judgment call documented in the ledger, flagged for "
            "author/reviewer confirmation.",
        ],
    }
    output_rows = []

    for track in TRACKS:
        track_fdr = fdr.loc[fdr["track"] == track].reset_index(drop=True)
        features = track_fdr["feature"].astype(str).tolist()
        p_frozen = track_fdr["boot_p"].to_numpy(dtype=float)
        t_stat = track_fdr["t_stat"].to_numpy(dtype=float)
        t_boot = compute_t_boot(track_fdr).to_numpy(dtype=float)

        frozen = _run_correction(p_frozen)
        _validate_frozen_reproduction(track, track_fdr, frozen)
        output_rows.extend(
            _fdr_rows(
                track,
                "frozen_m30",
                features,
                p_frozen,
                frozen,
                t_stat,
                t_boot,
                np.zeros(M_FROZEN, dtype=bool),
            )
        )

        frozen_bh_features = sorted(
            track_fdr.loc[frozen["bh_rejected"], "feature"].astype(str)
        )
        frozen_bhy_features = sorted(
            track_fdr.loc[frozen["bhy_rejected"], "feature"].astype(str)
        )
        summary["frozen_family"][track] = {
            "bh_rejected": int(frozen["bh_rejected"].sum()),
            "bhy_rejected": int(frozen["bhy_rejected"].sum()),
            "bh_rejected_features": frozen_bh_features,
            "bhy_rejected_features": frozen_bhy_features,
            "reproduction_check": "PASS",
        }

        extras = build_extra_p_values(track_fdr, ledger)
        enlarged_features = features + extras["feature"].astype(str).tolist()
        p_enlarged = np.concatenate(
            [p_frozen, extras["p_value"].to_numpy(dtype=float)]
        )
        if len(p_enlarged) != m_prime:
            _tripwire(
                f"{track} enlarged vector has {len(p_enlarged)} entries; "
                f"expected {m_prime}"
            )
        enlarged = _run_correction(p_enlarged)
        enlarged_t_stat = np.concatenate([t_stat, np.full(n_extra, np.nan)])
        enlarged_t_boot = np.concatenate([t_boot, np.full(n_extra, np.nan)])
        enlarged_is_extra = np.concatenate(
            [np.zeros(M_FROZEN, dtype=bool), np.ones(n_extra, dtype=bool)]
        )
        output_rows.extend(
            _fdr_rows(
                track,
                "enlarged_mprime",
                enlarged_features,
                p_enlarged,
                enlarged,
                enlarged_t_stat,
                enlarged_t_boot,
                enlarged_is_extra,
            )
        )

        enlarged_bh_original = sorted(
            track_fdr.loc[
                enlarged["bh_rejected"][:M_FROZEN], "feature"
            ].astype(str)
        )
        enlarged_bhy_original = sorted(
            track_fdr.loc[
                enlarged["bhy_rejected"][:M_FROZEN], "feature"
            ].astype(str)
        )
        bh_gained = sorted(set(enlarged_bh_original) - set(frozen_bh_features))
        bhy_gained = sorted(
            set(enlarged_bhy_original) - set(frozen_bhy_features)
        )
        if bh_gained or bhy_gained:
            _tripwire(
                f"{track} original features gained significance after family "
                f"enlargement: BH={bh_gained}, BHY={bhy_gained}"
            )

        bh_lost = sorted(set(frozen_bh_features) - set(enlarged_bh_original))
        bhy_lost = sorted(
            set(frozen_bhy_features) - set(enlarged_bhy_original)
        )
        bh_extra = sorted(
            extras.loc[
                enlarged["bh_rejected"][M_FROZEN:], "feature"
            ].astype(str)
        )
        bhy_extra = sorted(
            extras.loc[
                enlarged["bhy_rejected"][M_FROZEN:], "feature"
            ].astype(str)
        )
        summary["enlarged_family"][track] = {
            "m_prime": m_prime,
            "bh_rejected": int(enlarged["bh_rejected"].sum()),
            "bhy_rejected": int(enlarged["bhy_rejected"].sum()),
            "rejected_features": enlarged_bh_original,
            "lost_significance_vs_frozen": bh_lost,
            "bh_rejected_features": enlarged_bh_original,
            "bhy_rejected_features": enlarged_bhy_original,
            "bh_lost_significance_vs_frozen": bh_lost,
            "bhy_lost_significance_vs_frozen": bhy_lost,
            "bh_rejected_ledger_extras": bh_extra,
            "bhy_rejected_ledger_extras": bhy_extra,
        }

        output_rows.extend(_hlz_rows(track, track_fdr, frozen, t_boot))
        hlz_mask = np.abs(t_stat) > 3.0
        survivors = sorted(track_fdr.loc[hlz_mask, "feature"].astype(str))
        summary["hlz_hurdle"][track] = {
            "n_survivors": int(hlz_mask.sum()),
            "survivors": survivors,
            "agree_with_bh": int((hlz_mask & frozen["bh_rejected"]).sum()),
            "agree_with_bhy": int((hlz_mask & frozen["bhy_rejected"]).sum()),
        }
        summary["t_stat_disagreements"].extend(
            _find_t_stat_disagreements(track, track_fdr, t_boot)
        )

    output = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    output["is_ledger_extra"] = output["is_ledger_extra"].astype(bool)
    for column in ("bh_rejected", "bhy_rejected", "hlz_survivor"):
        output[column] = output[column].astype("boolean")
    return output, summary


def _print_summary(summary: dict) -> None:
    """Print a compact human-readable summary of all analyses."""
    ledger = summary["ledger_summary"]
    print("\n=== Search-adjusted FDR sensitivity (q=0.10) ===")
    print(
        f"Ledger: {ledger['n_rows']} rows; m={ledger['m']}; "
        f"n_extra={ledger['n_extra']}; m'={ledger['m_prime']}"
    )
    print("Frozen-family reproduction: PASS")
    for track in TRACKS:
        frozen = summary["frozen_family"][track]
        enlarged = summary["enlarged_family"][track]
        hlz = summary["hlz_hurdle"][track]
        print(
            f"  {track}: frozen BH={frozen['bh_rejected']}/30, "
            f"BHY={frozen['bhy_rejected']}/30; enlarged "
            f"BH={enlarged['bh_rejected']}/{ledger['m_prime']}, "
            f"BHY={enlarged['bhy_rejected']}/{ledger['m_prime']}"
        )
        print(
            f"           HLZ |t|>3: {hlz['n_survivors']}/30 "
            f"(agree BH={hlz['agree_with_bh']}, "
            f"BHY={hlz['agree_with_bhy']})"
        )
        print(
            "           rejected ledger extras: "
            f"BH={enlarged['bh_rejected_ledger_extras']}, "
            f"BHY={enlarged['bhy_rejected_ledger_extras']}"
        )
    print(
        "t_stat/t_boot disagreements flagged: "
        f"{len(summary['t_stat_disagreements'])}"
    )


def main() -> dict:
    """Run the analysis, write both output artifacts, and return the summary."""
    fdr, ledger = load_inputs()
    output, summary = run_analysis(fdr, ledger)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUT_PARQUET, index=False)
    with OUT_JSON.open("w", encoding="utf-8") as handle:
        # allow_nan=False stays: NaN/Infinity are not JSON and writing them as
        # bare literals produces a file that strict parsers reject. But a
        # missing t-statistic is a legitimate outcome (a dropped hypothesis has
        # no computable statistic), so those are mapped to null rather than
        # crashing the run after the parquet has already been written.
        json.dump(_json_safe(summary), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")

    _print_summary(summary)
    print(f"Saved -> {OUT_PARQUET.relative_to(ROOT)}")
    print(f"Saved -> {OUT_JSON.relative_to(ROOT)}")
    return summary


if __name__ == "__main__":
    main()
