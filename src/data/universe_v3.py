"""v3 universe access: daily point-in-time membership + the hybrid price panel.

One place for the three things the v3 rebuild changes, so the stage scripts
need only swap their loader calls:

1. **Daily, event-dated membership** (`membership_daily.parquet`, built by
   `src.data.build_daily_pit` from fja05680/sp500) instead of annual snapshots.
   The annual file treated a name as a member for every date in the year, which
   missed intra-year changes, and its 2025 snapshot was a late-2025 file applied
   to January 2025 (it listed constituents added that September).

2. **The hybrid price panel** (`daily_ohlcv_v3.parquet`): yfinance where Yahoo
   serves the symbol, plus session-filtered 1-minute bars for members Yahoo has
   dropped. Member-day coverage is about 96% overall and 99%+ from 2019, against
   the old panel's roughly 84%-to-98% measured against a smaller denominator.

3. **Correct closes.** The Array-era panel aggregated the last bar of any
   session, including extended hours, so its "close" was a post-market print
   (ACN 2013-06-03: 81.28 from a 19:50 bar against an official 82.10). Both v3
   sources carry official closes.

`close` is split-adjusted (Yahoo convention; the recovered series are adjusted
by the split detector in `src.data.recover_delisted_prices`), and `adj_close`
additionally carries dividends where the source provides them. Returns in this
project are price returns, so the pipeline reads `close`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

PANEL_PATH = ROOT / "data" / "processed" / "daily_ohlcv_v3.parquet"
MEMBERSHIP_PATH = ROOT / "datasets" / "sp500_daily_pit" / "membership_daily.parquet"

UNIVERSE_START = "2010-01-01"
UNIVERSE_END = "2026-09-14"


def load_panel(
    start: str = UNIVERSE_START,
    end: str = UNIVERSE_END,
    path: Path = PANEL_PATH,
) -> pd.DataFrame:
    """Long-format OHLCV for the v3 universe, restricted to [start, end]."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with:\n"
            "    python3 -m src.data.build_daily_pit --stages all"
        )
    panel = pd.read_parquet(path)
    panel["date"] = pd.to_datetime(panel["date"])
    mask = (panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))
    return panel.loc[mask].reset_index(drop=True)


def load_membership(
    start: str = UNIVERSE_START,
    end: str = UNIVERSE_END,
    path: Path = MEMBERSHIP_PATH,
) -> pd.DataFrame:
    """(date, ticker) rows, one per member-day."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with:\n"
            "    python3 -m src.data.build_daily_pit --stages raw,tickermap,membership"
        )
    mem = pd.read_parquet(path)
    mem["date"] = pd.to_datetime(mem["date"])
    mask = (mem["date"] >= pd.Timestamp(start)) & (mem["date"] <= pd.Timestamp(end))
    return mem.loc[mask].reset_index(drop=True)


def pit_tickers(start: str = UNIVERSE_START, end: str = UNIVERSE_END) -> set[str]:
    """Every symbol that is an index member at any point in the window."""
    return set(load_membership(start, end)["ticker"].unique())


def screen0_daily(daily: pd.DataFrame) -> pd.Series:
    """Screen 0 eligibility using DAILY point-in-time membership.

    Membership is lagged one trading day per ticker, alongside the existing
    lagged price and trailing-ADV filters, so eligibility on day t uses only
    information available before t.
    """
    from src.data.screen0 import screen0_eligibility

    return screen0_eligibility(
        daily,
        sp500_dir=None,
        universe="pit",
        granularity="daily",
        membership_daily_path=MEMBERSHIP_PATH,
    )


def survivorship_delta_daily(
    daily: pd.DataFrame,
    start: str = UNIVERSE_START,
    end: str = UNIVERSE_END,
) -> dict:
    """Quantify what point-in-time membership removes from a naive union panel.

    The union panel is what pooling every ever-member's full history gives you;
    the point-in-time panel keeps a name only while it is actually in the index.
    The difference is the survivorship and look-ahead contamination.
    """
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    window = (d["date"] >= pd.Timestamp(start)) & (d["date"] <= pd.Timestamp(end))
    d = d.loc[window]

    mem = load_membership(start, end)
    member_cells = set(zip(mem["ticker"], mem["date"]))
    union_cells = len(d)
    pit_cells = sum(1 for t, dt in zip(d["ticker"], d["date"]) if (t, dt) in member_cells)

    return {
        "union_cells": int(union_cells),
        "pit_member_cells": int(pit_cells),
        "non_pit_cells": int(union_cells - pit_cells),
        "pct_overstated": round(100.0 * (union_cells - pit_cells) / max(pit_cells, 1), 2),
        "distinct_tickers_union": int(d["ticker"].nunique()),
        "distinct_tickers_pit": int(mem["ticker"].nunique()),
    }
