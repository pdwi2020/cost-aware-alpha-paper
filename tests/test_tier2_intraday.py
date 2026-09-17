"""tests/test_tier2_intraday.py — intraday aggregation uses the regular session.

The Array-era pipeline aggregated the last bar of ANY session, so its daily
close was a post-market print (ACN 2013-06-03: 81.28 from a 19:50 bar against
an official close of 82.10). These tests pin the regular-session behaviour of
the intraday features on a synthetic 1-minute file: no X9, no network.

    python3 -m pytest tests/test_tier2_intraday.py -q
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.tier2_extended import _intraday_agg_file  # noqa: E402


def _bars(day: str, ticker: str) -> pd.DataFrame:
    """One day of 1-minute bars: pre-market, RTH, and post-market.

    RTH volume is 100 per minute; the extended-hours bars carry a wildly
    different price and huge volume, so any leakage is unmistakable.
    """
    rows = []
    # Pre-market 08:00-08:04 ET at price 500 (should be ignored)
    for m in range(5):
        rows.append((f"{day} 08:0{m}:00", 500.0, 10_000.0))
    # RTH 09:30-15:59 ET at price 100, volume 100/min
    rth = pd.date_range(f"{day} 09:30:00", f"{day} 15:59:00", freq="1min")
    for ts in rth:
        rows.append((ts.strftime("%Y-%m-%d %H:%M:%S"), 100.0, 100.0))
    # Post-market 19:55-19:59 ET at price 900 (should be ignored)
    for m in range(55, 60):
        rows.append((f"{day} 19:{m}:00", 900.0, 10_000.0))

    df = pd.DataFrame(rows, columns=["ts_et", "close", "volume"])
    ts = pd.to_datetime(df.pop("ts_et")).dt.tz_localize("America/New_York")
    df["timestamp"] = ts.dt.tz_convert("UTC")
    df["ticker"] = ticker
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]
    return df[["timestamp", "open", "high", "low", "close", "volume", "ticker"]]


@pytest.fixture
def one_min_file(tmp_path):
    frames = [_bars("2015-03-02", t) for t in ("AAA", "BBB")]
    frames.append(_bars("2015-03-03", "AAA"))
    path = tmp_path / "ohlcv_2015-03.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    return path


def test_vwap_excludes_extended_hours(one_min_file):
    """VWAP must be the RTH price, not dragged by pre/post-market prints."""
    con = duckdb.connect()
    out = _intraday_agg_file(
        con, one_min_file.as_posix(), ["AAA", "BBB"], "2015-03-01", "2015-03-31"
    )
    assert len(out) == 3                      # 2 tickers x 1 day + 1 extra day
    assert np.allclose(out["vwap"].to_numpy(), 100.0), out["vwap"].tolist()


def test_volume_clock_is_the_first_hour_share_of_rth_volume(one_min_file):
    """60 of the 390 RTH minutes fall in the first hour, at equal volume."""
    con = duckdb.connect()
    out = _intraday_agg_file(
        con, one_min_file.as_posix(), ["AAA"], "2015-03-01", "2015-03-31"
    )
    expected = 60.0 / 390.0
    assert np.allclose(out["vol_clock"].to_numpy(), expected, atol=1e-6)


def test_ticker_and_date_filters_apply(one_min_file):
    con = duckdb.connect()
    out = _intraday_agg_file(
        con, one_min_file.as_posix(), ["AAA"], "2015-03-03", "2015-03-31"
    )
    assert out["ticker"].unique().tolist() == ["AAA"]
    assert out["date"].dt.strftime("%Y-%m-%d").tolist() == ["2015-03-03"]


def test_flat_prices_give_zero_intraday_vol(one_min_file):
    """Constant RTH prices: AM and PM realised vol are both zero."""
    con = duckdb.connect()
    out = _intraday_agg_file(
        con, one_min_file.as_posix(), ["AAA", "BBB"], "2015-03-01", "2015-03-31"
    )
    assert np.allclose(out["vol_am"].fillna(0.0).to_numpy(), 0.0)
    assert np.allclose(out["vol_pm"].fillna(0.0).to_numpy(), 0.0)


def test_close_rth_comes_from_the_rth_bars(one_min_file):
    """close_rth must be the last RTH print, not the post-market one.

    The fixture's post-market bars sit at 900 and the RTH bars at 100, so a
    close taken over all bars would be unmistakable.
    """
    db = duckdb.connect()
    out = _intraday_agg_file(db, str(one_min_file), ["AAA"], "2015-03-01", "2015-03-31")
    assert (out["close_rth"] == 100.0).all()


def test_vwap_dev_is_scale_free(one_min_file, tmp_path):
    """Scaling every price by a split factor must not move vwap_dev.

    This is the regression for the defect that made vwap_dev the strongest
    "alpha" in the study: the VWAP came from the raw 1-minute store while the
    close came from the split-ADJUSTED daily panel, so vwap_dev equalled
    (1 - split factor) on every date before a split. CMG read -49.000 for a
    50:1 split. Because both legs now come from the same bars, any common
    scaling cancels exactly.
    """
    db = duckdb.connect()
    base = _intraday_agg_file(db, str(one_min_file), ["AAA"], "2015-03-01", "2015-03-31")

    split = pd.read_parquet(one_min_file)
    for col in ("open", "high", "low", "close"):
        split[col] = split[col] / 20.0      # a 20:1 split, as AMZN had
    split_path = tmp_path / "ohlcv_2015-03_split.parquet"
    split.to_parquet(split_path, index=False)
    scaled = _intraday_agg_file(db, str(split_path), ["AAA"], "2015-03-01", "2015-03-31")

    def dev(df):
        return ((df["close_rth"] - df["vwap"]) / df["close_rth"]).to_numpy()

    np.testing.assert_allclose(dev(base), dev(scaled), atol=1e-12)
