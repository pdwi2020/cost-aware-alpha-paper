"""tests/test_recover_delisted_prices.py — split detection and adjustment.

Pure-function tests only: no DuckDB, no X9, no network.

    python3 -m pytest tests/test_recover_delisted_prices.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import recover_delisted_prices as rdp  # noqa: E402


def _series(closes, volumes, start="2015-01-05"):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({
        "ticker": "TEST",
        "date": dates,
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": volumes,
    })


class TestSplitDetection:
    """A split needs both a price jump and a matching volume jump."""

    def test_detects_two_for_one_split(self):
        closes = [100.0, 101.0, 99.0, 50.0, 50.5, 49.5]
        volumes = [1e6, 1.1e6, 0.9e6, 2.0e6, 2.1e6, 1.9e6]
        splits = rdp.detect_split_factors(_series(closes, volumes))
        assert len(splits) == 1
        assert splits[0]["factor"] == 2.0
        assert splits[0]["date"] == pd.Timestamp("2015-01-08")

    def test_ignores_crash_without_volume_corroboration(self):
        """A -50% day on flat volume is a crash, not a split."""
        closes = [100.0, 101.0, 99.0, 50.0, 50.5, 49.5]
        volumes = [1e6, 1.1e6, 0.9e6, 1.05e6, 1.0e6, 1.0e6]
        assert rdp.detect_split_factors(_series(closes, volumes)) == []

    def test_detects_reverse_split(self):
        """1-for-10 reverse: price x10, volume /10."""
        closes = [2.0, 2.1, 1.9, 19.0, 19.5, 18.5]
        volumes = [10e6, 11e6, 9e6, 0.9e6, 1.0e6, 0.95e6]
        splits = rdp.detect_split_factors(_series(closes, volumes))
        assert len(splits) == 1
        assert splits[0]["factor"] == 0.1

    def test_detects_three_for_two_split(self):
        """3-for-2 is a conventional ratio: price /1.5, volume x1.5."""
        closes = [100.0, 100.0, 66.67, 67.0]
        volumes = [1e6, 1e6, 1.5e6, 1.45e6]
        splits = rdp.detect_split_factors(_series(closes, volumes))
        assert len(splits) == 1
        assert splits[0]["factor"] == 1.5

    def test_ignores_odd_ratio_moves(self):
        """A -38% move matching no split ratio, on flat volume, is left alone."""
        closes = [100.0, 100.0, 62.0, 61.0]
        volumes = [1e6, 1e6, 1.05e6, 1.0e6]
        assert rdp.detect_split_factors(_series(closes, volumes)) == []

    def test_rejects_a_crash_in_volatile_context(self):
        """First Republic, March 2023: a crash that passes price and volume checks.

        The -47.5% day snaps to a 2-for-1 ratio (1.905) and its 1.42x volume
        sits inside the volume band, so only the surrounding days distinguish
        it from a corporate action. Real numbers from the 1-minute store.
        """
        closes = [115.01, 95.99, 81.01, 36.36, 39.44, 31.39, 34.80, 23.11, 12.13, 15.93]
        volumes = [1.25e6, 8.98e6, 29.2e6, 29.6e6, 23.1e6, 87.6e6,
                   159.5e6, 118.7e6, 168.9e6, 194.0e6]
        assert rdp.detect_split_factors(_series(closes, volumes)) == []

    def test_accepts_a_split_in_calm_context(self):
        """The same price and volume signature, but ordinary neighbours."""
        closes = [100.0, 100.5, 99.5, 50.0, 50.3, 49.8, 50.1]
        volumes = [1e6, 1.0e6, 1.0e6, 2.0e6, 2.0e6, 1.9e6, 2.0e6]
        splits = rdp.detect_split_factors(_series(closes, volumes))
        assert len(splits) == 1
        assert splits[0]["factor"] == 2.0
        assert splits[0]["neighbour_max_abs_ret"] < rdp.CONTEXT_MAX_ABS_RET

    def test_clean_series_has_no_splits(self):
        rng = np.random.default_rng(0)
        closes = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60))))
        volumes = list(rng.uniform(9e5, 1.1e6, 60))
        assert rdp.detect_split_factors(_series(closes, volumes)) == []


class TestSplitAdjustment:
    """Adjustment must remove the artificial jump and leave real returns alone."""

    def test_adjustment_removes_the_jump(self):
        closes = [100.0, 101.0, 99.0, 50.0, 50.5, 49.5]
        volumes = [1e6, 1.1e6, 0.9e6, 2.0e6, 2.1e6, 1.9e6]
        df = _series(closes, volumes)
        splits = rdp.detect_split_factors(df)
        adj = rdp.apply_split_adjustment(df, splits)

        rets = adj["close"].pct_change().to_numpy()
        assert abs(rets[3]) < 0.05, f"split-date return not neutralised: {rets[3]:+.3f}"
        # Pre-split prices halved, pre-split volume doubled.
        assert np.isclose(adj.loc[0, "close"], 50.0)
        assert np.isclose(adj.loc[0, "volume"], 2e6)
        # Post-split rows untouched.
        assert np.isclose(adj.loc[4, "close"], 50.5)

    def test_real_returns_are_preserved(self):
        closes = [100.0, 101.0, 99.0, 50.0, 55.0, 49.5]
        volumes = [1e6, 1.1e6, 0.9e6, 2.0e6, 2.1e6, 1.9e6]
        df = _series(closes, volumes)
        adj = rdp.apply_split_adjustment(df, rdp.detect_split_factors(df))
        # The +10% day after the split survives unchanged.
        assert np.isclose(adj["close"].pct_change().to_numpy()[4], 0.10, atol=1e-9)

    def test_no_splits_is_a_no_op(self):
        df = _series([10.0, 10.1, 10.2], [1e6, 1e6, 1e6])
        pd.testing.assert_frame_equal(rdp.apply_split_adjustment(df, []), df.copy())


class TestHelpers:

    def test_spells_pad_warmup_and_tail(self):
        membership = pd.DataFrame({
            "date": pd.to_datetime(["2015-01-05", "2016-06-01", "2018-03-02"]),
            "ticker": ["AAA", "AAA", "BBB"],
        })
        spells = rdp.spells_for(membership, {"AAA"})
        start, end = spells["AAA"]
        assert start == pd.Timestamp("2015-01-05") - pd.Timedelta(days=400)
        assert end == pd.Timestamp("2016-06-01") + pd.Timedelta(days=5)
        assert "BBB" not in spells

    def test_month_files_only_returns_existing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rdp, "ONE_MIN_DIR", tmp_path)
        (tmp_path / "ohlcv_2015-02.parquet").write_bytes(b"")
        (tmp_path / "ohlcv_2015-04.parquet").write_bytes(b"")
        files = rdp.month_files(pd.Timestamp("2015-01-15"), pd.Timestamp("2015-05-01"))
        assert [f.name for f in files] == ["ohlcv_2015-02.parquet", "ohlcv_2015-04.parquet"]

    def test_nearest_known_ratio(self):
        assert rdp._nearest_known_ratio(2.02) == 2.0
        assert rdp._nearest_known_ratio(0.4999) == 0.5
        assert rdp._nearest_known_ratio(1.27) is None
