"""tests/test_search_adjusted_fdr.py — Tests for the search-adjusted FDR
sensitivity analysis (src/fdr/run_search_adjusted_fdr.py).

Covers: the ledger's shape/content invariants, the frozen-family (m=30)
BH/BHY reproduction against the known headline counts, the enlarged-family
(m'=57) duplicate p-value borrowing and monotonicity property, the HLZ |t|>3
hurdle against an independent inline filter, and the two output artifacts
produced by main().

Run:
    python3 -m pytest tests/test_search_adjusted_fdr.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.fdr.bh_correction import benjamini_hochberg, bhy_procedure
from src.fdr.run_search_adjusted_fdr import (
    FDR_RESULTS,
    OUT_JSON,
    OUT_PARQUET,
    SEARCH_LEDGER,
    build_extra_p_values,
    load_inputs,
    main,
    run_analysis,
)

pytestmark = pytest.mark.skipif(
    not FDR_RESULTS.exists() or not SEARCH_LEDGER.exists(),
    reason="requires data/processed/fdr_results.parquet and results/search_ledger.csv",
)

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_LEDGER_COLUMNS = [
    "item",
    "category",
    "n_hypotheses_added",
    "status",
    "stage_evaluated",
    "evidence",
    "date",
]
EXTRA_STATUSES = {"dropped", "never_built", "superseded"}


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loaded():
    """Load the real fdr_results.parquet + search_ledger.csv once per module."""
    return load_inputs()


@pytest.fixture(scope="module")
def analysis(loaded):
    """Run the full analysis once per module (expensive-ish, so cache it)."""
    fdr, ledger = loaded
    return run_analysis(fdr, ledger)


# ---------------------------------------------------------------------------
# 1. Ledger shape / content invariants
# ---------------------------------------------------------------------------

def test_ledger_columns_and_row_count(loaded):
    """The ledger has exactly the 7 required columns and 52 data rows."""
    _, ledger = loaded
    assert ledger.columns.tolist() == REQUIRED_LEDGER_COLUMNS
    assert len(ledger) == 52


def test_ledger_m_prime_is_57(loaded):
    """sum(n_hypotheses_added) over dropped/never_built/superseded rows == 27,
    so m' = 30 + 27 = 57. This is a concrete regression value tied to the
    current ledger content: if someone edits the ledger without understanding
    the consequences, this test should fail.
    """
    _, ledger = loaded
    extra = ledger.loc[ledger["status"].isin(EXTRA_STATUSES), "n_hypotheses_added"]
    n_extra = int(extra.sum())
    assert n_extra == 27
    assert 30 + n_extra == 57


# ---------------------------------------------------------------------------
# 2. Frozen-family (m=30) reproduction of the known headline counts
# ---------------------------------------------------------------------------

def test_frozen_family_headline_counts(loaded):
    """Counts match the pin file when one exists, and are otherwise coherent.

    These counts are a property of a particular panel, not a universal
    invariant: the Array-era values (BH 12/30 track_a, 0/30 track_b, BHY 11/30)
    were hard-coded here and in the module, so rebuilding the universe turned a
    legitimate change into a failure. The pin lives in
    config/expected_fdr_counts.json and is updated deliberately when a run's
    counts are accepted.
    """
    import json

    fdr, _ = loaded
    counts = fdr.groupby("track")[["bh_rejected", "bhy_rejected"]].sum()

    # Universal: BHY is more conservative than BH, so it can never reject more.
    for track in counts.index:
        assert int(counts.loc[track, "bhy_rejected"]) <= int(counts.loc[track, "bh_rejected"])
        assert 0 <= int(counts.loc[track, "bh_rejected"]) <= len(fdr[fdr["track"] == track])

    pins_path = ROOT / "config" / "expected_fdr_counts.json"
    if pins_path.exists():
        pins = json.loads(pins_path.read_text())
        for track, methods in pins.items():
            # Keys starting with "_" are documentation (e.g. "_comment", which
            # records why the pinned counts changed), not track entries.
            if track.startswith("_"):
                continue
            for method, expected in methods.items():
                assert int(counts.loc[track, method]) == int(expected), (
                    f"{track}.{method}: pinned {expected}, got "
                    f"{int(counts.loc[track, method])}"
                )


def test_frozen_family_reproduction_matches_file(loaded, analysis):
    """The script's BH/BHY recomputation on boot_p reproduces the file's stored
    bh_rejected / bhy_rejected flags exactly.

    This is the invariant worth testing, and it holds on any panel: it catches a
    p-value column and a rejection column drifting out of step, which is the
    failure that would silently corrupt every downstream selection.
    """
    fdr, _ = loaded
    _, summary = analysis
    counts = fdr.groupby("track")[["bh_rejected", "bhy_rejected"]].sum()

    for track in ("track_a", "track_b"):
        assert summary["frozen_family"][track]["reproduction_check"] == "PASS"
        assert summary["frozen_family"][track]["bh_rejected"] == int(
            counts.loc[track, "bh_rejected"]
        )
        assert summary["frozen_family"][track]["bhy_rejected"] == int(
            counts.loc[track, "bhy_rejected"]
        )


# ---------------------------------------------------------------------------
# 3. Enlarged-family (m'=57) p-value assignment
# ---------------------------------------------------------------------------

def test_duplicate_features_borrow_source_p_value(loaded):
    """reversal_1w borrows ret_5d's boot_p; reversal_4w borrows ret_21d's
    boot_p -- exact sign-flip duplicates share their two-sided p-value,
    checked for both tracks."""
    fdr, ledger = loaded
    for track in ("track_a", "track_b"):
        track_fdr = fdr.loc[fdr["track"] == track]
        extras = build_extra_p_values(track_fdr, ledger).set_index("feature")
        source_p = track_fdr.set_index("feature")["boot_p"]

        assert extras.loc["reversal_1w", "p_value"] == pytest.approx(
            source_p.loc["ret_5d"]
        )
        assert extras.loc["reversal_4w", "p_value"] == pytest.approx(
            source_p.loc["ret_21d"]
        )


def test_broadcast_and_conservative_default_p_values_are_one(loaded):
    """Broadcast macros (p=1 by construction) and the conservative-default
    items (Polymarket columns, term_spread_x_mom, vol_am, vol_pm) all get
    exactly p_value == 1.0."""
    fdr, ledger = loaded
    broadcast = {
        "vix", "vix_chg_5d", "term_spread", "term_spread_chg_21d",
        "credit_proxy", "credit_proxy_chg_5d", "dxy_ret_5d", "wti_ret_21d",
    }
    conservative_defaults = {
        "term_spread_x_mom", "vol_am", "vol_pm",
        "total_trades", "buy_trades", "sell_trades", "total_usd_volume",
        "total_token_volume", "avg_price", "price_std", "min_price",
        "max_price", "vwap", "buy_sell_usd_ratio", "active_days",
        "first_trade_ts", "last_trade_ts",
    }
    track_fdr = fdr.loc[fdr["track"] == "track_a"]
    extras = build_extra_p_values(track_fdr, ledger).set_index("feature")

    for item in broadcast | conservative_defaults:
        assert extras.loc[item, "p_value"] == 1.0, item


def test_m_prime_equals_57(analysis):
    _, summary = analysis
    assert summary["ledger_summary"]["m_prime"] == 57
    assert summary["enlarged_family"]["track_a"]["m_prime"] == 57
    assert summary["enlarged_family"]["track_b"]["m_prime"] == 57


def test_enlarged_family_never_gains_significance_vs_frozen(analysis):
    """A feature that is significant under BH/BHY on the frozen family of 30
    must remain significant (or drop out), never NEWLY appear as significant
    only after adding p=1.0 (or duplicate) extra hypotheses -- BH/BHY
    rejection sets are monotonically non-increasing in the original
    features as m grows for fixed p-values. run_analysis() itself trip-wires
    on this, so reaching this point at all is already a partial check; we
    additionally verify explicitly here."""
    _, summary = analysis
    for track in ("track_a", "track_b"):
        frozen_bh = set(summary["frozen_family"][track]["bh_rejected_features"])
        enlarged_bh = set(summary["enlarged_family"][track]["bh_rejected_features"])
        frozen_bhy = set(summary["frozen_family"][track]["bhy_rejected_features"])
        enlarged_bhy = set(summary["enlarged_family"][track]["bhy_rejected_features"])
        assert enlarged_bh <= frozen_bh, f"{track}: BH gained significance"
        assert enlarged_bhy <= frozen_bhy, f"{track}: BHY gained significance"


# ---------------------------------------------------------------------------
# 4. BH/BHY monotonicity on a synthetic vector (independent of real data)
# ---------------------------------------------------------------------------

def test_bh_bhy_monotonic_under_added_null_hypotheses():
    """Adding p=1.0 hypotheses to a fixed p-value vector never increases the
    rejection count of the original hypotheses (synthetic sanity check,
    independent of any real data)."""
    base_p = np.array([0.001, 0.004, 0.02, 0.03, 0.5, 0.8])
    padded_p = np.concatenate([base_p, np.ones(20)])

    base_bh, _ = benjamini_hochberg(base_p, q=0.10)
    padded_bh, _ = benjamini_hochberg(padded_p, q=0.10)
    assert padded_bh[: len(base_p)].sum() <= base_bh.sum()

    base_bhy, _ = bhy_procedure(base_p, q=0.10)
    padded_bhy, _ = bhy_procedure(padded_p, q=0.10)
    assert padded_bhy[: len(base_p)].sum() <= base_bhy.sum()


# ---------------------------------------------------------------------------
# 5. HLZ |t|>3 hurdle vs. an independent inline filter
# ---------------------------------------------------------------------------

def test_hlz_hurdle_matches_independent_filter(loaded, analysis):
    """The script's reported HLZ survivor counts/lists match a direct,
    independently-written `fdr_results[|t_stat|>3.0]` filter grouped by
    track (this deliberately does NOT import the script's own hurdle
    logic)."""
    fdr, _ = loaded
    _, summary = analysis

    for track in ("track_a", "track_b"):
        track_fdr = fdr.loc[fdr["track"] == track]
        expected = sorted(
            track_fdr.loc[track_fdr["t_stat"].abs() > 3.0, "feature"].astype(str)
        )
        actual = summary["hlz_hurdle"][track]["survivors"]
        assert actual == expected
        assert summary["hlz_hurdle"][track]["n_survivors"] == len(expected)


# ---------------------------------------------------------------------------
# 6. End-to-end: main() writes both output artifacts with the expected shape
# ---------------------------------------------------------------------------

def test_main_writes_expected_artifacts():
    """Running main() (the script's real entry point) produces the parquet
    and JSON outputs with the documented schema/keys."""
    summary = main()

    assert OUT_PARQUET.exists()
    assert OUT_JSON.exists()

    output = pd.read_parquet(OUT_PARQUET)
    expected_columns = {
        "track", "variant", "feature", "is_ledger_extra", "p_value",
        "t_stat", "t_boot", "bh_adj_p", "bh_rejected", "bhy_adj_p",
        "bhy_rejected", "hlz_survivor",
    }
    assert expected_columns <= set(output.columns)
    assert set(output["variant"].unique()) == {
        "frozen_m30", "enlarged_mprime", "hlz_t3",
    }
    # 30 (frozen) + 57 (enlarged) + 30 (hlz) rows per track, x2 tracks.
    assert len(output) == 2 * (30 + 57 + 30)

    for key in (
        "ledger_summary", "frozen_family", "enlarged_family",
        "hlz_hurdle", "t_stat_disagreements", "notes",
    ):
        assert key in summary
    assert summary["ledger_summary"]["m_prime"] == 57
    assert len(summary["notes"]) >= 2
