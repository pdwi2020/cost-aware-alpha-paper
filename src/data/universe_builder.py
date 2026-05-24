"""Survivorship-bias-free S&P 500 universe builder.

Uses point-in-time annual membership snapshots (one CSV per year).
A ticker's membership for year Y is applied only to dates within year Y.
"""

import os
import pandas as pd
from pathlib import Path


def get_universe_tickers(sp500_dir: str, start_year: int = 2010, end_year: int = 2024) -> set:
    """Union of all S&P 500 tickers across years — used to pre-filter DuckDB queries."""
    tickers = set()
    for year in range(start_year, end_year + 1):
        path = Path(sp500_dir) / f"{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="latin-1", usecols=["Ticker"])
        tickers.update(df["Ticker"].dropna().str.strip().tolist())
    return tickers


def build_sp500_universe(
    sp500_dir: str,
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """Build point-in-time daily S&P 500 membership table.

    For each calendar year, loads the annual snapshot CSV and marks every
    trading date in that year. No future information leaks across year boundaries.

    Returns:
        DataFrame with columns [date, ticker, weight] covering [start_date, end_date].
        One row per (date, ticker) that was in the S&P 500 on that date.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    start_year = start.year
    end_year = end.year

    all_dates = pd.bdate_range(start=start, end=end)  # business days only

    records = []
    for year in range(start_year, end_year + 1):
        path = Path(sp500_dir) / f"{year}.csv"
        if not path.exists():
            continue

        snapshot = pd.read_csv(path, encoding="latin-1")
        snapshot.columns = snapshot.columns.str.strip()
        snapshot["Ticker"] = snapshot["Ticker"].str.strip()
        snapshot = snapshot[["Ticker", "Weight"]].dropna(subset=["Ticker"])

        year_dates = all_dates[(all_dates.year == year)]
        if len(year_dates) == 0:
            continue

        for _, row in snapshot.iterrows():
            records.append({
                "date": year_dates,
                "ticker": row["Ticker"],
                "weight": row["Weight"],
            })

    if not records:
        return pd.DataFrame(columns=["date", "ticker", "weight"])

    rows = []
    for r in records:
        for d in r["date"]:
            rows.append({"date": d, "ticker": r["ticker"], "weight": r["weight"]})

    universe = pd.DataFrame(rows)
    universe["date"] = pd.to_datetime(universe["date"]).dt.normalize()
    return universe.sort_values(["date", "ticker"]).reset_index(drop=True)


def get_daily_membership(sp500_dir: str, date: pd.Timestamp) -> list:
    """Return list of tickers in S&P 500 on a specific date."""
    path = Path(sp500_dir) / f"{date.year}.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path, encoding="latin-1", usecols=["Ticker"])
    return df["Ticker"].dropna().str.strip().tolist()
