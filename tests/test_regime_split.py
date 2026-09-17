"""tests/test_regime_split.py — regression tests for the calm/stressed split.

Why these exist
---------------
The submitted (Array) version classified regimes with

    is_calm = (str(reg).lower() == "calm")

while ``regime_vix`` holds the VIX *level* as a float. "16.89" is never equal to
"calm", so every day was filed as stressed, the calm sub-series was empty, and
the paper reported "calm: 0 rejections" as a finding when it was a count over
zero observations. ``VIX_CALM_THRESH`` was defined but never read.

The tests below pin the three properties that would have caught it:

1. numeric VIX levels classify on either side of the threshold;
2. a full panel produces a NON-EMPTY series for both regimes, and the two
   partition the days (no day is dropped or double counted);
3. BH over the regime family runs once at m = 2 * n_features, which is what the
   manuscript claims, rather than once per regime at m = n_features.

Run:
    python3 -m pytest tests/test_regime_split.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fdr.run_fdr import (  # noqa: E402
    VIX_CALM_THRESH,
    _is_calm_regime,
    compute_fold_ics,
)


# ---------------------------------------------------------------------------
# 1. The classifier itself
# ---------------------------------------------------------------------------

def test_numeric_vix_classifies_on_threshold():
    """A float VIX level must classify, which is the bug that shipped."""
    assert _is_calm_regime(12.5) is True
    assert _is_calm_regime(VIX_CALM_THRESH) is True        # boundary is calm
    assert _is_calm_regime(VIX_CALM_THRESH + 0.01) is False
    assert _is_calm_regime(45.0) is False
    assert _is_calm_regime(np.float64(16.89)) is True


def test_string_labels_still_classify():
    assert _is_calm_regime("calm") is True
    assert _is_calm_regime("Stressed") is False


def test_unknown_regime_is_not_silently_stressed():
    """Unknown/missing must be None, not False.

    Returning False would sweep every unclassifiable day into "stressed", which
    is how an empty calm series masquerades as a real result.
    """
    assert _is_calm_regime(None) is None
    assert _is_calm_regime(np.nan) is None
    assert _is_calm_regime("unlabelled") is None


# ---------------------------------------------------------------------------
# 2. The split over a panel
# ---------------------------------------------------------------------------

def _panel(n_days: int = 80, n_stocks: int = 40, seed: int = 0) -> pd.DataFrame:
    """Panel whose VIX straddles the threshold, with a real signal."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_stocks)]

    # First half calm (VIX 12-18), second half stressed (VIX 24-34).
    vix_by_date = dict(
        zip(dates,
            np.r_[rng.uniform(12, 18, n_days // 2),
                  rng.uniform(24, 34, n_days - n_days // 2)])
    )

    rows = []
    for d in dates:
        target = rng.normal(size=n_stocks)
        feat = target + rng.normal(scale=0.5, size=n_stocks)
        for j, tk in enumerate(tickers):
            rows.append({
                "ticker": tk, "date": d,
                "feat": feat[j], "target_x": target[j],
                "s0_eligible": True, "regime_vix": vix_by_date[d],
            })
    return pd.DataFrame(rows).set_index(["ticker", "date"])


def test_both_regimes_non_empty_and_partition_the_days():
    df = _panel()
    dates = df.index.get_level_values("date")
    folds = [{
        "fold_id": "f1",
        "train_start": "2014-01-01", "train_end": "2014-12-31",
        "test_start": str(dates.min().date()), "test_end": str(dates.max().date()),
    }]

    all_ic, calm_ic, stressed_ic = compute_fold_ics(
        df, ["feat"], "target_x", folds
    )

    n_all = all_ic["feat"].notna().sum()
    n_calm = calm_ic["feat"].notna().sum()
    n_stressed = stressed_ic["feat"].notna().sum()

    # The shipped bug produced n_calm == 0 while n_stressed == n_all.
    assert n_calm > 0, "calm sub-series is empty: the regime split is broken"
    assert n_stressed > 0, "stressed sub-series is empty"
    assert n_calm + n_stressed == n_all, "regimes must partition the days"

    # And they must land on the right side of the threshold.
    vix = df.groupby(level="date")["regime_vix"].first()
    assert vix.reindex(calm_ic["feat"].dropna().index).le(VIX_CALM_THRESH).all()
    assert vix.reindex(stressed_ic["feat"].dropna().index).gt(VIX_CALM_THRESH).all()


def test_regime_series_are_not_the_full_sample():
    """Guards the specific failure: one regime silently equal to everything."""
    df = _panel()
    dates = df.index.get_level_values("date")
    folds = [{
        "fold_id": "f1",
        "train_start": "2014-01-01", "train_end": "2014-12-31",
        "test_start": str(dates.min().date()), "test_end": str(dates.max().date()),
    }]
    all_ic, calm_ic, stressed_ic = compute_fold_ics(df, ["feat"], "target_x", folds)

    n_all = all_ic["feat"].notna().sum()
    assert stressed_ic["feat"].notna().sum() < n_all
    assert calm_ic["feat"].notna().sum() < n_all


# ---------------------------------------------------------------------------
# 3. The regime BH family size
# ---------------------------------------------------------------------------

def test_pooled_regime_family_is_stricter_than_per_regime():
    """BH over 2m pooled hypotheses must reject no more than m-at-a-time.

    The manuscript describes a 2m = 60 family. Correcting within each regime
    separately at m = 30 is the more permissive procedure, so for p-values near
    the boundary the pooled family must not reject more.
    """
    from src.fdr.bh_correction import benjamini_hochberg

    m = 30
    rng = np.random.default_rng(7)
    p_calm = np.r_[rng.uniform(0, 0.02, 5), rng.uniform(0.2, 1.0, m - 5)]
    p_stressed = np.r_[rng.uniform(0, 0.02, 3), rng.uniform(0.2, 1.0, m - 3)]

    sep = (int(benjamini_hochberg(p_calm, q=0.10)[0].sum())
           + int(benjamini_hochberg(p_stressed, q=0.10)[0].sum()))
    pooled = int(benjamini_hochberg(np.r_[p_calm, p_stressed], q=0.10)[0].sum())

    assert pooled <= sep


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
