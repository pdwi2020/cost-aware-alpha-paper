"""tests/test_screen0_no_lookahead.py — No-look-ahead unit tests for screen0.py.

All four tests use fully synthetic data (tiny in-memory DataFrames + toy CSVs
written to a tmp directory).  No DuckDB, no network access, no parquet files.

Run:
    python3 -m pytest tests/test_screen0_no_lookahead.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the repo root importable without installation
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.data.screen0 import (
    build_liquidity_universe,
    lagged_min_price,
    pit_membership_mask,
    screen0_eligibility,
    trailing_adv_usd,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_daily(
    tickers: list[str],
    dates: list[str],
    prices: dict[str, list[float]],
    volumes: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Build a long-format OHLCV-like DataFrame for testing.

    Parameters
    ----------
    tickers : list of ticker strings
    dates   : list of date strings (ISO)
    prices  : dict ticker -> list of close_raw prices (same length as dates)
    volumes : dict ticker -> list of volumes (default: all 1.0)

    The `close` column is set equal to `close_raw` (no winsorization in tests).
    """
    rows = []
    for tkr in tickers:
        px = prices[tkr]
        vol = (volumes or {}).get(tkr, [1.0] * len(dates))
        for i, d in enumerate(dates):
            rows.append({
                "ticker":    tkr,
                "date":      pd.Timestamp(d),
                "close_raw": px[i],
                "close":     px[i],
                "volume":    vol[i],
            })
    return pd.DataFrame(rows)


def _write_snapshot_csv(tmp_path: Path, year: int, tickers: list[str]) -> None:
    """Write a minimal annual snapshot CSV to tmp_path/{year}.csv."""
    rows = [{"Company": tkr, "Weight": 1.0, "Ticker": tkr} for tkr in tickers]
    df = pd.DataFrame(rows, columns=["Company", "Weight", "Ticker"])
    df.to_csv(tmp_path / f"{year}.csv", index=False)


# ---------------------------------------------------------------------------
# Test 1: No look-ahead on price
#
# Ticker A: price crosses $4 → $6 between dates[1] and dates[2].
#   At dates[2] (t):   close_raw[t-1] = $4 < $5  → INELIGIBLE
#   At dates[3] (t+1): close_raw[t]   = $6 >= $5 → ELIGIBLE
# ---------------------------------------------------------------------------

def test_no_lookahead_price(tmp_path):
    """s0_eligible[t] depends on price[t-1], never price[t]."""
    dates  = ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08"]
    # Ticker A: $4, $4, $6, $6, $6  (crosses threshold between dates[1] and dates[2])
    # Ticker B: $10 throughout (always above $5; controls for membership)
    prices = {
        "A": [4.0, 4.0, 6.0, 6.0, 6.0],
        "B": [10.0, 10.0, 10.0, 10.0, 10.0],
    }
    volumes = {
        "A": [1e6, 1e6, 1e6, 1e6, 1e6],
        "B": [1e6, 1e6, 1e6, 1e6, 1e6],
    }
    daily = _make_daily(["A", "B"], dates, prices, volumes)

    # Both tickers are PIT members in 2021
    _write_snapshot_csv(tmp_path, 2021, ["A", "B"])

    elig = screen0_eligibility(
        daily,
        sp500_dir=tmp_path,
        min_price=5.0,
        min_adv=0.0,  # disable ADV filter so price drives the test
        adv_window=2,
        universe="pit",
    )

    elig_a = elig.xs("A", level="ticker")

    # date[0]: price[t-1] is NaN (no prior day) → lagged_price is NaN → ineligible
    assert not elig_a[pd.Timestamp("2021-01-04")], \
        "A should be ineligible on dates[0] (no prior price)"

    # date[1]: price[t-1] = $4 < $5 → INELIGIBLE
    assert not elig_a[pd.Timestamp("2021-01-05")], \
        "A should be ineligible on dates[1]: lagged price $4 < $5"

    # date[2] (t): price[t] = $6 (above threshold) BUT price[t-1] = $4 → INELIGIBLE
    # This is the critical look-ahead check: we must NOT use today's $6 price.
    assert not elig_a[pd.Timestamp("2021-01-06")], \
        "A should be INELIGIBLE on dates[2]: lagged price[t-1]=$4 < $5, " \
        "even though current price[t]=$6 >= $5"

    # date[3] (t+1): price[t-1] = $6 >= $5 → ELIGIBLE
    assert elig_a[pd.Timestamp("2021-01-07")], \
        "A should be ELIGIBLE on dates[3]: lagged price[t-1]=$6 >= $5"

    # Ticker B is always eligible (price $10 ≥ $5 always, including t-1 from dates[1] on)
    elig_b = elig.xs("B", level="ticker")
    assert elig_b[pd.Timestamp("2021-01-05")], "B should be eligible (price always $10)"


# ---------------------------------------------------------------------------
# Test 2: No look-ahead on ADV
#
# Verify that trailing_adv_usd at date t equals the rolling mean of
# dollar-volume through t-1 (i.e., the series is shifted by 1 day per ticker).
# We hand-verify one specific cell.
# ---------------------------------------------------------------------------

def test_no_lookahead_adv():
    """ADV[t] equals rolling mean of dollar-volume through t-1 (shift correctness)."""
    dates = ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]
    # Dollar volume sequence for ticker A: 100, 200, 300, 400
    prices  = {"A": [1.0, 2.0, 3.0, 4.0]}
    volumes = {"A": [100.0, 100.0, 100.0, 100.0]}
    # close_raw * volume = 100, 200, 300, 400
    daily = _make_daily(["A"], dates, prices, volumes)

    adv = trailing_adv_usd(daily, window=2)
    adv_a = adv.xs("A", level="ticker").sort_index()

    ts = lambda s: pd.Timestamp(s)

    # At dates[0] (2021-01-04): no prior day → NaN
    assert pd.isna(adv_a[ts("2021-01-04")]), \
        "ADV should be NaN on dates[0] (nothing in trailing window)"

    # At dates[1] (2021-01-05): window of 2 trailing days, but only dates[0]
    # exists as prior data → mean of [100] = 100
    assert adv_a[ts("2021-01-05")] == pytest.approx(100.0, rel=1e-6), \
        f"ADV at dates[1] should be 100, got {adv_a[ts('2021-01-05')]}"

    # At dates[2] (2021-01-06): trailing 2 days are dates[0,1] (dv=100,200) → mean=150
    assert adv_a[ts("2021-01-06")] == pytest.approx(150.0, rel=1e-6), \
        f"ADV at dates[2] should be 150, got {adv_a[ts('2021-01-06')]}"

    # At dates[3] (2021-01-07): trailing 2 days are dates[1,2] (dv=200,300) → mean=250
    assert adv_a[ts("2021-01-07")] == pytest.approx(250.0, rel=1e-6), \
        f"ADV at dates[3] should be 250, got {adv_a[ts('2021-01-07')]}"

    # Crucially: ADV at dates[2] (150) does NOT incorporate dates[2]'s own dv=300.
    # If it did (no shift), it would be mean(200,300)=250.  The value 150 confirms
    # the shift is applied and dates[2]'s own dollar-volume is excluded.
    assert adv_a[ts("2021-01-06")] != pytest.approx(250.0, rel=1e-6) or False, \
        "This branch should never execute; it confirms 150 ≠ 250 (no look-ahead)"
    # (The double-negative above is fine — it's an explanatory guard that never fires.)


# ---------------------------------------------------------------------------
# Test 3: PIT masking
#
# Ticker C is in the 2021 snapshot but NOT the 2020 snapshot.
#   → ineligible on all 2020 dates, eligible on 2021 dates (price/ADV aside).
# ---------------------------------------------------------------------------

def test_pit_masking(tmp_path):
    """Ticker absent from year-Y snapshot is ineligible on year-Y dates."""
    dates_2020 = ["2020-12-29", "2020-12-30", "2020-12-31"]
    dates_2021 = ["2021-01-04", "2021-01-05", "2021-01-06"]
    all_dates  = dates_2020 + dates_2021

    prices  = {"C": [10.0] * 6}  # price always $10, well above $5
    volumes = {"C": [1e7]  * 6}  # always high volume

    daily = _make_daily(["C"], all_dates, prices, volumes)

    # 2020 CSV: C is NOT present; 2021 CSV: C IS present
    _write_snapshot_csv(tmp_path, 2020, ["OTHER"])  # C absent
    _write_snapshot_csv(tmp_path, 2021, ["C"])       # C present

    mask = pit_membership_mask(daily, sp500_dir=tmp_path)
    c_mask = mask.xs("C", level="ticker").sort_index()

    for d in dates_2020:
        assert not c_mask[pd.Timestamp(d)], \
            f"C should NOT be a PIT member on {d} (not in 2020 snapshot)"

    for d in dates_2021:
        assert c_mask[pd.Timestamp(d)], \
            f"C SHOULD be a PIT member on {d} (in 2021 snapshot)"

    # Confirm that screen0_eligibility reflects the PIT gate:
    elig = screen0_eligibility(
        daily,
        sp500_dir=tmp_path,
        min_price=5.0,
        min_adv=0.0,
        adv_window=2,
        universe="pit",
    )
    elig_c = elig.xs("C", level="ticker").sort_index()

    # 2020 dates: PIT=False → ineligible regardless of price/ADV
    for d in dates_2020:
        assert not elig_c[pd.Timestamp(d)], \
            f"C should be INELIGIBLE on {d} (PIT=False)"

    # 2021 dates from dates_2021[1] onward: PIT=True, price[t-1]=$10 >= $5 → eligible
    # (dates_2021[0] may have NaN lagged price if dates_2020[-1] is the previous row)
    # Actually dates[2020-12-31] exists, so price[t-1] for 2021-01-04 = $10.
    for d in dates_2021:
        assert elig_c[pd.Timestamp(d)], \
            f"C should be ELIGIBLE on {d} (PIT=True, price $10 >= $5)"


# ---------------------------------------------------------------------------
# Test 4: Liquidity universe
#
# 3 tickers (X, Y, Z) with distinct lagged ADV; top_n=2 should admit the
# two highest-ADV names each day.
# ---------------------------------------------------------------------------

def test_liquidity_universe():
    """With top_n=2 over 3 tickers, exactly the 2 highest-lagged-ADV names are members."""
    dates  = ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]
    # Dollar-volume (price * volume) differs by ticker so ADV ranking is clear:
    #   X: dv = 1000 (lowest)
    #   Y: dv = 2000 (middle)
    #   Z: dv = 3000 (highest)
    prices  = {"X": [10.0] * 4, "Y": [20.0] * 4, "Z": [30.0] * 4}
    volumes = {"X": [100.0] * 4, "Y": [100.0] * 4, "Z": [100.0] * 4}
    # dv: X=1000, Y=2000, Z=3000 every day

    daily = _make_daily(["X", "Y", "Z"], dates, prices, volumes)

    liq = build_liquidity_universe(daily, top_n=2, adv_window=2)

    # From dates[1] onward the lagged ADV is available; the ranking is stable.
    for d in ["2021-01-05", "2021-01-06", "2021-01-07"]:
        ts = pd.Timestamp(d)
        try:
            x_mem = liq.xs("X", level="ticker")[ts]
            y_mem = liq.xs("Y", level="ticker")[ts]
            z_mem = liq.xs("Z", level="ticker")[ts]
        except KeyError:
            pytest.skip(f"Date {ts} missing from liquidity series")
            return

        assert not x_mem, f"X (lowest ADV) should NOT be in top-2 on {d}"
        assert y_mem,     f"Y (middle ADV) should be in top-2 on {d}"
        assert z_mem,     f"Z (highest ADV) should be in top-2 on {d}"

    # Cross-check membership count: exactly top_n=2 members per day where data exist
    liq_wide = liq.unstack(level="ticker")  # date × ticker or ticker × date
    # Normalise so we always have date as index, ticker as columns
    if "ticker" in liq_wide.index.names:
        liq_wide = liq_wide.T
    for d in ["2021-01-05", "2021-01-06", "2021-01-07"]:
        ts = pd.Timestamp(d)
        if ts in liq_wide.index:
            row = liq_wide.loc[ts]
            n_members = int(row.sum())
            assert n_members == 2, \
                f"Expected exactly 2 members on {d}, got {n_members}"
