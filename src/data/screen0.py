"""src/data/screen0.py — Screen 0 eligibility filters (PIT, price, ADV).

Single source-of-truth for all universe + liquidity screening.
All filters are LAGGED: eligibility at date t depends only on data ≤ t-1
(price/ADV and, when selected, daily PIT membership) or the annual PIT
membership snapshot for year(t).

Public API
----------
pit_membership_mask(daily, sp500_dir)          -> pd.Series[bool] (ticker, date)
pit_membership_mask_daily(daily, path)         -> pd.Series[bool] (ticker, date)
trailing_adv_usd(daily, window=21)             -> pd.Series[float] (ticker, date)
lagged_min_price(daily)                        -> pd.Series[float] (ticker, date)
screen0_eligibility(daily, sp500_dir, ...)     -> pd.Series[bool]  (ticker, date)
build_liquidity_universe(daily, top_n, ...)    -> pd.Series[bool]  (ticker, date)
survivorship_delta(daily, sp500_dir)           -> dict

spec.yaml authoritative defaults
---------------------------------
screen0.min_price_usd      = 5.0
screen0.min_adv_usd        = 1_000_000
universes.robustness.rule  = top-N, N=500
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Locate spec.yaml defaults
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent
_SPEC_FILE = _ROOT / "config" / "spec.yaml"


def _load_spec() -> dict:
    if not _SPEC_FILE.exists():
        return {}
    with open(_SPEC_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _spec_defaults() -> tuple[float, float, int, int]:
    """Return (min_price_usd, min_adv_usd, adv_window, top_n) from spec.yaml."""
    spec = _load_spec()
    s0 = spec.get("screen0", {})
    min_price = float(s0.get("min_price_usd", 5.0))
    min_adv = float(s0.get("min_adv_usd", 1_000_000))
    adv_window = int(s0.get("adv_window", 21))  # not in spec; use 21 as default
    # Parse top_n from universes.robustness.rule: "top-N names … N=500 …"
    rule = spec.get("universes", {}).get("robustness", {}).get("rule", "")
    m = re.search(r"N=(\d+)", rule)
    top_n = int(m.group(1)) if m else 500
    return min_price, min_adv, adv_window, top_n


# ---------------------------------------------------------------------------
# Internals: shared prep
# ---------------------------------------------------------------------------

def _ensure_long(daily: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with [ticker, date] as plain columns (not index)."""
    df = daily.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _raw_price(daily: pd.DataFrame) -> pd.Series:
    """Return the un-winsorized price column: `close_raw` if present, else `close`."""
    df = _ensure_long(daily)
    col = "close_raw" if "close_raw" in df.columns else "close"
    return df.set_index(["ticker", "date"])[col]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def pit_membership_mask(
    daily: pd.DataFrame,
    sp500_dir: Union[str, Path],
) -> pd.Series:
    """Point-in-time S&P 500 membership mask.

    A ticker found in `{sp500_dir}/{YEAR}.csv` is considered a member for
    ALL dates whose year equals YEAR.  No intra-year addition/removal timing
    is modelled (stated limitation in spec.yaml).

    Parameters
    ----------
    daily     : long OHLCV frame; must have columns [ticker, date].
    sp500_dir : directory containing annual CSVs named `{YEAR}.csv`.

    Returns
    -------
    pd.Series[bool] indexed by (ticker, date).
    True iff the ticker appears in the annual snapshot for date.year.
    """
    from src.data.universe_builder import build_sp500_universe

    df = _ensure_long(daily)
    start = df["date"].min().strftime("%Y-%m-%d")
    end   = df["date"].max().strftime("%Y-%m-%d")

    # Build the full PIT membership table from the annual snapshots
    pit = build_sp500_universe(str(sp500_dir), start_date=start, end_date=end)
    # pit has columns [date, ticker, weight]; normalise for safe merge
    pit["date"] = pd.to_datetime(pit["date"]).dt.normalize()
    pit["ticker"] = pit["ticker"].str.strip()
    pit["_member"] = True
    pit = pit[["ticker", "date", "_member"]].drop_duplicates(["ticker", "date"])

    merged = df[["ticker", "date"]].merge(pit, on=["ticker", "date"], how="left")
    merged["_member"] = merged["_member"].astype("boolean").fillna(False).astype(bool)
    merged = merged.set_index(["ticker", "date"])
    return merged["_member"].rename("pit_member")


def pit_membership_mask_daily(
    daily: pd.DataFrame,
    membership_path: Union[str, Path],
) -> pd.Series:
    """One-trading-day-lagged point-in-time S&P 500 membership mask.

    At date t, membership equals the same-day membership flag from ticker T's
    immediately preceding row in `daily`.  The per-ticker shift mirrors
    `lagged_min_price`, so the result uses membership known at the close of t-1
    and is look-ahead free.  A ticker's first row is always a non-member because
    it has no preceding row.

    Parameters
    ----------
    daily           : long OHLCV frame; must have columns [ticker, date].
    membership_path : parquet file with one [date, ticker] row per member day.

    Returns
    -------
    pd.Series[bool] indexed by (ticker, date), named 'pit_member_daily_lag1'.
    """
    df = _ensure_long(daily)
    membership = pd.read_parquet(membership_path, columns=["date", "ticker"])
    membership["date"] = pd.to_datetime(membership["date"]).dt.normalize()
    membership["_member"] = True
    membership = membership[["ticker", "date", "_member"]].drop_duplicates(
        ["ticker", "date"]
    )

    merged = df[["ticker", "date"]].merge(
        membership, on=["ticker", "date"], how="left"
    )
    merged["_member"] = merged["_member"].astype("boolean").fillna(False).astype(bool)
    merged = merged.sort_values(["ticker", "date"])
    lagged = (
        merged.groupby("ticker", group_keys=False)["_member"]
        .transform(lambda s: s.shift(1))
    )
    merged["_member_lag1"] = lagged.astype("boolean").fillna(False).astype(bool)
    return (
        merged.set_index(["ticker", "date"])["_member_lag1"]
        .rename("pit_member_daily_lag1")
    )


def trailing_adv_usd(
    daily: pd.DataFrame,
    window: int = 21,
) -> pd.Series:
    """Trailing average daily dollar volume (ADV), lagged by 1 day per ticker.

    At date t the value equals the rolling mean of (close_raw × volume) over
    the [t-window, t-1] window — i.e. data through t-1 only (look-ahead free).

    Parameters
    ----------
    daily  : long OHLCV frame with [ticker, date, volume] and either
             `close_raw` or `close` for price.
    window : number of trailing trading days for the rolling mean (default 21).

    Returns
    -------
    pd.Series[float] indexed by (ticker, date).
    """
    df = _ensure_long(daily).copy()
    price_col = "close_raw" if "close_raw" in df.columns else "close"
    df["dv"] = df[price_col] * df["volume"]  # dollar volume
    df = df.sort_values(["ticker", "date"])

    adv = (
        df.groupby("ticker", group_keys=False)["dv"]
        .transform(lambda s: s.rolling(window, min_periods=1).mean().shift(1))
    )
    df["adv_usd"] = adv
    return df.set_index(["ticker", "date"])["adv_usd"].rename("adv_usd")


def lagged_min_price(daily: pd.DataFrame) -> pd.Series:
    """Previous-day price (close_raw if present, else close), shifted 1 day per ticker.

    At date t, value = price[t-1].  Used to enforce the $5 minimum-price
    screen without look-ahead: we know yesterday's price before the open.

    Parameters
    ----------
    daily : long OHLCV frame with [ticker, date] and `close_raw` or `close`.

    Returns
    -------
    pd.Series[float] indexed by (ticker, date).
    """
    df = _ensure_long(daily).copy()
    price_col = "close_raw" if "close_raw" in df.columns else "close"
    df = df.sort_values(["ticker", "date"])
    lagged = (
        df.groupby("ticker", group_keys=False)[price_col]
        .transform(lambda s: s.shift(1))
    )
    df["price_lag1"] = lagged
    return df.set_index(["ticker", "date"])["price_lag1"].rename("price_lag1")


def screen0_eligibility(
    daily: pd.DataFrame,
    sp500_dir: Union[str, Path],
    min_price: float | None = None,
    min_adv: float | None = None,
    adv_window: int | None = None,
    universe: str = "pit",
    top_n: int | None = None,
    granularity: str = "annual",
    membership_daily_path: Union[str, Path, None] = None,
) -> pd.Series:
    """Compute Screen 0 eligibility flag for every (ticker, date) row.

    Eligibility at date t is determined ENTIRELY from data ≤ t-1 (price,
    ADV) plus either the annual PIT snapshot for year(t) or daily membership
    lagged by one row per ticker:

        eligible_t = member_as_of_t-1 AND (price[t-1] >= min_price)
                                        AND (adv_usd[t-1:t-window] >= min_adv)

    With the default `granularity='annual'`, annual snapshot membership is
    used exactly as before.  With `granularity='daily'`, membership is read
    from `membership_daily_path` and lagged one trading-day row per ticker.

    For `universe='liquidity'`, the PIT membership term is replaced by
    top-`top_n` membership by trailing ADV (also lagged, also look-ahead free),
    and both PIT membership paths are ignored.

    Parameters
    ----------
    daily      : long OHLCV frame (columns: ticker, date, close or close_raw,
                 volume, …).
    sp500_dir  : path to annual snapshot CSVs; required for universe='pit'.
    min_price  : minimum lagged price in USD (default from spec.yaml: $5).
    min_adv    : minimum trailing ADV in USD (default from spec.yaml: $1M).
    adv_window : trailing ADV window in days (default from spec.yaml: 21).
    universe   : 'pit' (default) or 'liquidity'.
    top_n      : for universe='liquidity', number of top-ADV names to admit
                 (default from spec.yaml: 500).
    granularity: for universe='pit', 'annual' (default) uses `sp500_dir`, while
                 'daily' uses one-row-lagged daily PIT membership.
    membership_daily_path : parquet [date, ticker] membership file; required
                            when universe='pit' and granularity='daily'.

    Returns
    -------
    pd.Series[bool] indexed by (ticker, date), named 's0_eligible'.
    """
    _d_min_price, _d_min_adv, _d_adv_window, _d_top_n = _spec_defaults()
    if min_price  is None: min_price  = _d_min_price
    if min_adv    is None: min_adv    = _d_min_adv
    if adv_window is None: adv_window = _d_adv_window
    if top_n      is None: top_n      = _d_top_n

    if granularity not in {"annual", "daily"}:
        raise ValueError(
            f"granularity must be 'annual' or 'daily', got {granularity!r}"
        )

    # --- 1. ADV and lagged price (both look-ahead free) ---
    adv  = trailing_adv_usd(daily, window=adv_window)   # indexed (ticker, date)
    lpx  = lagged_min_price(daily)                       # indexed (ticker, date)

    adv_ok  = adv  >= min_adv
    price_ok = lpx >= min_price

    # --- 2. Universe membership ---
    if universe == "pit":
        if granularity == "annual":
            member = pit_membership_mask(daily, sp500_dir)
        else:
            if membership_daily_path is None:
                raise ValueError(
                    "membership_daily_path is required when granularity='daily'"
                )
            member = pit_membership_mask_daily(daily, membership_daily_path)
    elif universe == "liquidity":
        member = build_liquidity_universe(daily, top_n=top_n, adv_window=adv_window)
    else:
        raise ValueError(f"universe must be 'pit' or 'liquidity', got {universe!r}")

    # --- 3. Combine; align on common (ticker, date) index ---
    idx = adv.index  # canonical index from trailing_adv_usd
    eligible = (
        member.reindex(idx).fillna(False)
        & adv_ok.reindex(idx).fillna(False)
        & price_ok.reindex(idx).fillna(False)
    )
    return eligible.rename("s0_eligible")


def build_liquidity_universe(
    daily: pd.DataFrame,
    top_n: int = 500,
    adv_window: int = 21,
) -> pd.Series:
    """Mark the top-`top_n` tickers by lagged trailing ADV as members each date.

    Membership is determined from the *lagged* ADV (data through t-1), so
    membership at date t is look-ahead free.

    Parameters
    ----------
    daily      : long OHLCV frame with [ticker, date, volume] and price col.
    top_n      : number of highest-ADV tickers to mark as members.
    adv_window : trailing window for ADV (same shift logic as trailing_adv_usd).

    Returns
    -------
    pd.Series[bool] indexed by (ticker, date), named 'liq_member'.
    """
    adv = trailing_adv_usd(daily, window=adv_window)  # (ticker, date) -> float

    # Reshape to wide (date × ticker) for cross-sectional ranking
    adv_wide = adv.unstack(level="ticker")  # date × ticker
    # Rank descending (highest ADV = rank 1); NaN tickers are excluded
    rank_wide = adv_wide.rank(axis=1, method="first", ascending=False, na_option="keep")
    member_wide = rank_wide <= top_n

    # Long format
    member_long = (
        member_wide.stack(future_stack=True)
        .rename("liq_member")
        .astype(bool)
    )
    member_long.index.names = ["date", "ticker"]
    # Reorder to (ticker, date) to match all other outputs
    member_long = member_long.reorder_levels(["ticker", "date"]).sort_index()
    return member_long


def survivorship_delta(
    daily: pd.DataFrame,
    sp500_dir: Union[str, Path],
) -> dict:
    """Quantify the survivorship-bias removed by switching union → PIT universe.

    Counts cells that exist in the historical union of all S&P 500 members
    (old behaviour) but are flagged as non-members under PIT masking.

    Returns
    -------
    dict with keys:
        union_cells          : total (ticker,date) rows in the input panel
        pit_member_cells     : cells where PIT membership = True
        non_pit_cells        : union_cells - pit_member_cells
        pct_removed          : fraction of union cells removed by PIT masking (0–1)
        union_tickers        : count of distinct tickers in the input panel
        tickers_never_member : count of tickers with zero PIT-membership rows
        tickers_partial      : count of tickers that appear before/after their
                               first/last PIT membership year (partial members)
    """
    df = _ensure_long(daily)
    membership = pit_membership_mask(daily, sp500_dir)

    union_cells = len(df)
    pit_member_cells = int(membership.sum())
    non_pit_cells = union_cells - pit_member_cells
    pct_removed = non_pit_cells / union_cells if union_cells > 0 else 0.0

    union_tickers = df["ticker"].nunique()

    # Per-ticker membership counts
    ticker_member_count = membership.groupby(level="ticker").sum()
    tickers_never_member = int((ticker_member_count == 0).sum())

    # Tickers that have ≥1 member day but also ≥1 non-member day
    # (i.e. appear in the union before their first or after their last snapshot year)
    member_df = membership.reset_index()
    member_df.columns = ["ticker", "date", "is_member"]
    member_df["year"] = member_df["date"].dt.year

    pit_years_by_ticker = (
        member_df[member_df["is_member"]]
        .groupby("ticker")["year"]
        .agg(first_pit_year="min", last_pit_year="max")
    )
    all_years_by_ticker = member_df.groupby("ticker")["year"].agg(
        first_year="min", last_year="max"
    )
    combined = all_years_by_ticker.join(pit_years_by_ticker, how="inner")
    tickers_partial = int(
        ((combined["first_year"] < combined["first_pit_year"]) |
         (combined["last_year"]  > combined["last_pit_year"])).sum()
    )

    return {
        "union_cells":          union_cells,
        "pit_member_cells":     pit_member_cells,
        "non_pit_cells":        non_pit_cells,
        "pct_removed":          round(pct_removed, 6),
        "union_tickers":        union_tickers,
        "tickers_never_member": tickers_never_member,
        "tickers_partial":      tickers_partial,
    }
