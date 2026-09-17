"""Week 3: Tier 2 features — intraday microstructure, cross-asset, crowding.

All features are lag-1 corrected (signal at T close used to trade at T+1 open).
Intraday aggregations use America/New_York timezone grouping.
"""

import time
import duckdb
import numpy as np
import pandas as pd


CROWDING_ETFS = [
    "SPY", "QQQ",
    "XLK", "XLE", "XLF", "XLY", "XLP",
    "XLI", "XLB", "XLU", "XLV", "XLC", "XLRE",
]


# ---------------------------------------------------------------------------
# A. Intraday microstructure
# ---------------------------------------------------------------------------

# The shared DuckDB catalog view `equities_ohlcv_1m` enumerates its source
# files explicitly, so months added afterwards (2025-08 onward) are invisible
# through it. Read the monthly files by glob instead and leave the shared
# catalog untouched, since other projects depend on it.
ONE_MIN_GLOB = (
    "/Volumes/Crucial X9/data/market_data/equities/ohlcv_1m_us/data/ohlcv_*.parquet"
)

# The close of a session sits within a fraction of a percent of that session's
# VWAP. A 99.9th percentile above this means the two legs are on different
# split bases, which is how a corporate action once became the strongest
# "alpha" in the study. Deliberately loose: it catches basis errors (values of
# order 1 and above), not fat tails.
VWAP_DEV_SANITY_MAX = 0.5


def _intraday_agg_file(
    db: duckdb.DuckDBPyConnection,
    source: str,
    tickers: list,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Aggregate ONE monthly 1-minute file to daily intraday features.

    Month-major, every ticker at once. The previous form scanned the whole
    store once per ticker batch, and the store is roughly 80 GB on an external
    drive, so six batches meant six full passes.

    Returned columns: ticker, date, vwap, vol_clock, vol_am, vol_pm
    VWAP deviation is computed later (needs EOD close from daily_ohlcv).
    """
    ticker_sql = "', '".join(tickers)
    # min_of_day = minute offset from midnight in NY time
    # Regular trading hours (RTH): 570 (9:30 AM) to 960 (4:00 PM)
    # First hour:                  570 (9:30 AM) to 630 (10:30 AM)
    # AM session for vol sig:      570 (9:30 AM) to 660 (11:00 AM)
    # PM session for vol sig:      840 (2:00 PM) to 960 (4:00 PM)
    query = f"""
        WITH base AS (
            SELECT
                ticker,
                (timestamp AT TIME ZONE 'America/New_York')::DATE  AS date_ny,
                timestamp,
                close,
                volume,
                (EXTRACT(HOUR   FROM timestamp AT TIME ZONE 'America/New_York') * 60
                 + EXTRACT(MINUTE FROM timestamp AT TIME ZONE 'America/New_York'))  AS mod,
                close / NULLIF(
                    LAG(close) OVER (
                        PARTITION BY ticker,
                                     (timestamp AT TIME ZONE 'America/New_York')::DATE
                        ORDER BY timestamp
                    ), 0
                ) - 1  AS bar_ret
            FROM read_parquet('{source}')
            WHERE ticker IN ('{ticker_sql}')
              AND (timestamp AT TIME ZONE 'America/New_York')::DATE >= '{start_date}'
              AND (timestamp AT TIME ZONE 'America/New_York')::DATE <= '{end_date}'
              -- Regular session only, BEFORE the LAG above is evaluated.
              -- Otherwise the 09:30 bar's return is measured from the last
              -- pre-market print, putting extended-hours noise into vol_am --
              -- the same defect that made the old daily close a post-market
              -- print. The CASE filters below are then redundant but kept as
              -- documentation of each window.
              AND (EXTRACT(HOUR   FROM timestamp AT TIME ZONE 'America/New_York') * 60
                   + EXTRACT(MINUTE FROM timestamp AT TIME ZONE 'America/New_York')) >= 570
              AND (EXTRACT(HOUR   FROM timestamp AT TIME ZONE 'America/New_York') * 60
                   + EXTRACT(MINUTE FROM timestamp AT TIME ZONE 'America/New_York')) < 960
        )
        SELECT
            ticker,
            date_ny  AS date,
            -- VWAP over RTH (9:30–16:00)
            SUM(CASE WHEN mod >= 570 AND mod < 960 THEN close * volume END)
            / NULLIF(SUM(CASE WHEN mod >= 570 AND mod < 960 THEN volume END), 0)  AS vwap,
            -- RTH closing print, taken from these SAME raw bars so that
            -- vwap_dev is a ratio of two prices on one basis. Pairing this
            -- VWAP with the split-ADJUSTED daily close made vwap_dev equal
            -- (1 - split factor) for every date before a split: CMG read
            -- -49.000 (50:1), ORLY -14.000 (15:1), AMZN -19 (20:1). Twelve
            -- tickers then held 76% of the signal, and the "alpha" was a
            -- post-2021 corporate action leaking into a 2013-2021 book.
            ARG_MAX(CASE WHEN mod >= 570 AND mod < 960 THEN close END,
                    CASE WHEN mod >= 570 AND mod < 960 THEN mod END)  AS close_rth,
            -- Volume clock: first-hour fraction of RTH volume
            SUM(CASE WHEN mod >= 570 AND mod < 630 THEN volume ELSE 0 END)
            / NULLIF(SUM(CASE WHEN mod >= 570 AND mod < 960 THEN volume END), 0)  AS vol_clock,
            -- AM realized vol (9:30–11:00)
            STDDEV(CASE WHEN mod >= 570 AND mod < 660  THEN bar_ret END)  AS vol_am,
            -- PM realized vol (14:00–16:00)
            STDDEV(CASE WHEN mod >= 840 AND mod < 960  THEN bar_ret END)  AS vol_pm
        FROM base
        GROUP BY ticker, date_ny
        ORDER BY ticker, date_ny
    """
    df = db.execute(query).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_tier2_intraday(
    db: duckdb.DuckDBPyConnection,
    sp500_tickers: set,
    daily_ohlcv: pd.DataFrame,
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """Intraday microstructure features.

    Features:
      overnight_gap   — open_T / close_{T-1} - 1  (from daily OHLCV)
      vwap_dev        — (eod_close - VWAP) / eod_close
      vol_clock       — first-hour RTH volume fraction
      vol_sig_ratio   — AM realized vol / PM realized vol

    Returns MultiIndex (ticker, date), NOT yet lag-1 shifted.
    """
    tickers = sorted(sp500_tickers)

    # ------------------------------------------------------------------
    # Overnight gap from daily OHLCV (no 1m bars needed)
    # ------------------------------------------------------------------
    close_wide = daily_ohlcv.pivot(index="date", columns="ticker", values="close")
    open_wide  = daily_ohlcv.pivot(index="date", columns="ticker", values="open")
    close_wide.index = pd.to_datetime(close_wide.index)
    open_wide.index  = pd.to_datetime(open_wide.index)
    overnight_wide = open_wide / close_wide.shift(1) - 1  # date × ticker

    # ------------------------------------------------------------------
    # 1-min intraday aggregation (batched DuckDB queries)
    # ------------------------------------------------------------------
    from src.data.recover_delisted_prices import month_files

    files = month_files(pd.Timestamp(start_date), pd.Timestamp(end_date))
    print(f"  Querying intraday features ({len(files)} monthly files × "
          f"{len(tickers)} tickers)...")
    parts = []
    t0 = time.time()
    for i, path in enumerate(files, start=1):
        part = _intraday_agg_file(db, path.as_posix(), tickers, start_date, end_date)
        parts.append(part)
        if i % 24 == 0 or i == len(files):
            print(f"    {i}/{len(files)} files  ({time.time() - t0:.0f}s elapsed)")
    if not parts:
        raise RuntimeError(
            f"No 1-minute files found for {start_date}..{end_date}; expected "
            "monthly parquet files next to ONE_MIN_GLOB."
        )
    intra = pd.concat(parts, ignore_index=True)

    # VWAP deviation: both legs come from the same raw 1-minute bars, so the
    # split-adjustment basis cancels instead of becoming the signal.
    intra["vwap_dev"]      = (intra["close_rth"] - intra["vwap"]) / intra["close_rth"].replace(0, np.nan)
    intra["vol_sig_ratio"] = intra["vol_am"] / intra["vol_pm"].replace(0, np.nan)

    # A same-day deviation of the close from that day's VWAP is a fraction of a
    # percent; anything near 1 means the two legs are on different bases again.
    _finite = intra["vwap_dev"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(_finite):
        _p999 = float(_finite.abs().quantile(0.999))
        if _p999 > VWAP_DEV_SANITY_MAX:
            worst = (
                intra.assign(_a=intra["vwap_dev"].abs())
                .nlargest(5, "_a")[["ticker", "date", "vwap", "close_rth", "vwap_dev"]]
            )
            raise ValueError(
                f"vwap_dev is implausible: 99.9th pct |vwap_dev| = {_p999:.3f} "
                f"exceeds {VWAP_DEV_SANITY_MAX}. The close and the VWAP are "
                f"probably on different split bases.\n{worst.to_string(index=False)}"
            )

    # Stack overnight gap to long
    og_long = (
        overnight_wide.stack(future_stack=True)
        .rename("overnight_gap")
        .reset_index()
    )
    og_long.columns = ["date", "ticker", "overnight_gap"]
    og_long["date"] = pd.to_datetime(og_long["date"])

    # Merge intraday features + overnight gap
    intra = intra.merge(og_long, on=["ticker", "date"], how="outer")
    intra = intra.set_index(["ticker", "date"]).sort_index()
    return intra[["overnight_gap", "vwap_dev", "vol_clock", "vol_sig_ratio"]]


# ---------------------------------------------------------------------------
# B. Cross-asset macro features
# ---------------------------------------------------------------------------

def build_tier2_macro(
    db: duckdb.DuckDBPyConnection,
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """Cross-asset macro signals from FRED.

    Features:
      dxy_ret_5d         — 5-day return on DXY (DTWEXBGS)
      wti_ret_21d        — 21-day return on WTI crude (DCOILWTICO)
      credit_proxy       — DGS10 - DFF (10yr rate minus Fed Funds as IG credit proxy)
      credit_proxy_chg_5d — 5-day change in credit_proxy

    Returns date-indexed DataFrame, NOT yet lag-1 shifted.
    """
    fred = db.execute("SELECT * FROM fred_macro").fetchdf()
    fred["date"] = pd.to_datetime(fred["date"]).dt.normalize()
    macro = fred.pivot_table(
        index="date", columns="series_id", values="value", aggfunc="last"
    ).ffill()

    out = pd.DataFrame(index=macro.index)
    out.index = pd.to_datetime(out.index)

    if "DTWEXBGS" in macro.columns:
        out["dxy_ret_5d"] = macro["DTWEXBGS"].pct_change(5, fill_method=None)

    if "DCOILWTICO" in macro.columns:
        out["wti_ret_21d"] = macro["DCOILWTICO"].pct_change(21, fill_method=None)

    if "DGS10" in macro.columns and "DFF" in macro.columns:
        out["credit_proxy"]       = macro["DGS10"] - macro["DFF"]
        out["credit_proxy_chg_5d"] = out["credit_proxy"].diff(5)

    date_mask = (out.index >= pd.Timestamp(start_date)) & (out.index <= pd.Timestamp(end_date))
    return out[date_mask]


# ---------------------------------------------------------------------------
# C. Crowding proxies
# ---------------------------------------------------------------------------

def build_tier2_crowding(
    db: duckdb.DuckDBPyConnection,
    daily_ohlcv: pd.DataFrame,
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    window: int = 20,
) -> pd.DataFrame:
    """Rolling 20-day correlation of each stock with SPY, QQQ, and sector ETFs.

    Returns MultiIndex (ticker, date) with columns corr_SPY, corr_QQQ, corr_XLK, ...
    NOT yet lag-1 shifted.
    """
    # ETF daily closes come from the SAME daily panel as the stocks.
    #
    # They used to be aggregated out of the 1-minute store with
    # LAST(close ORDER BY timestamp) and no session filter, which has two
    # consequences, both of which reached the published features:
    #
    #   1. That "close" is the last bar of the day INCLUDING after-hours, so it
    #      is a post-market print. Against the official close it differs by a
    #      mean of 11.9 bps and by as much as 70.5 bps (SPY, July 2025). This is
    #      the same defect the specification records as fixed for stocks; it was
    #      never fixed for the crowding ETFs.
    #   2. The 1-minute ETF history ends 2025-07-31, so every corr_* feature was
    #      undefined from 2025-08-01 onwards. Because one NaN feature nulls the
    #      whole weighted composite, this silently froze the forward-window book
    #      on its first day.
    #
    # These are 20-day correlations of DAILY returns, so the 1-minute store was
    # never needed. The daily panel carries all 13 ETFs with official closes and
    # runs to the end of the price data.
    etf_mask = daily_ohlcv["ticker"].isin(CROWDING_ETFS)
    etf_df = daily_ohlcv.loc[etf_mask, ["ticker", "date", "close"]].copy()
    etf_df["date"] = pd.to_datetime(etf_df["date"])
    etf_wide = etf_df.pivot(index="date", columns="ticker", values="close")
    available = [c for c in CROWDING_ETFS if c in etf_wide.columns]
    if not available:
        raise ValueError(
            "no crowding ETFs found in the daily panel; corr_* cannot be built. "
            f"Expected any of {CROWDING_ETFS}."
        )
    etf_ret = etf_wide[available].pct_change(fill_method=None)
    missing = [c for c in CROWDING_ETFS if c not in available]
    print(f"  Crowding ETFs available: {available}"
          + (f"  [MISSING: {missing}]" if missing else ""))
    print(f"  ETF close source: daily panel, {etf_wide.index.min().date()} "
          f"to {etf_wide.index.max().date()} (official closes)")

    # Stock daily returns (wide)
    close_wide = daily_ohlcv.pivot(index="date", columns="ticker", values="close")
    close_wide.index = pd.to_datetime(close_wide.index)
    stock_ret = close_wide.pct_change(fill_method=None)

    # Rolling correlation per ETF
    corr_dict = {}
    for etf in available:
        etf_s = etf_ret[etf].reindex(stock_ret.index)
        corr_dict[f"corr_{etf}"] = stock_ret.rolling(window).corr(etf_s)  # date × ticker

    if not corr_dict:
        return pd.DataFrame()

    # Stack to long (ticker, date) × feature
    long_parts = []
    for feat, wide_df in corr_dict.items():
        s = wide_df.stack(future_stack=True).rename(feat)
        long_parts.append(s)

    crowding = pd.concat(long_parts, axis=1)
    crowding.index.names = ["date", "ticker"]
    crowding = crowding.swaplevel(0, 1).sort_index()
    return crowding


# ---------------------------------------------------------------------------
# Full Tier 2 builder
# ---------------------------------------------------------------------------

def build_tier2_features(
    db: duckdb.DuckDBPyConnection,
    sp500_tickers: set,
    daily_ohlcv: pd.DataFrame,
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    crowding_window: int = 20,
) -> pd.DataFrame:
    """Construct Tier 2 feature library (intraday + macro + crowding).

    All returned features are lag-1 shifted (signal at T close → trade T+1 open).

    Returns MultiIndex (ticker, date) DataFrame.
    """
    # A. Intraday (not yet shifted)
    print("\n[Tier 2-A] Intraday microstructure features...")
    t0 = time.time()
    intraday = build_tier2_intraday(
        db, sp500_tickers, daily_ohlcv,
        start_date=start_date, end_date=end_date,
    )
    print(f"  Done in {time.time()-t0:.0f}s  →  {intraday.shape}")

    # B. Cross-asset macro (date-indexed, not yet shifted)
    print("\n[Tier 2-B] Cross-asset macro features...")
    macro_feats = build_tier2_macro(db, start_date=start_date, end_date=end_date)
    print(f"  Macro features: {list(macro_feats.columns)}")

    # C. Crowding (not yet shifted)
    print("\n[Tier 2-C] Crowding proxies...")
    t0 = time.time()
    crowding = build_tier2_crowding(
        db, daily_ohlcv,
        start_date=start_date, end_date=end_date,
        window=crowding_window,
    )
    print(f"  Done in {time.time()-t0:.0f}s  →  {crowding.shape}")

    # Assemble: start with intraday, merge crowding
    print("\n[Tier 2-D] Assembling feature matrix...")
    out = intraday.copy()

    if not crowding.empty:
        out = out.join(crowding, how="left")

    # Broadcast macro features (date-indexed) to all (ticker, date) rows
    dates = out.index.get_level_values("date")
    macro_aligned = macro_feats.reindex(dates).ffill()
    for col in macro_aligned.columns:
        out[col] = macro_aligned[col].values

    # Filter to date window
    date_idx = out.index.get_level_values("date")
    out = out[
        (date_idx >= pd.Timestamp(start_date)) &
        (date_idx <= pd.Timestamp(end_date))
    ]

    # Lag-1 all features per ticker
    out = out.groupby(level="ticker").shift(1)

    return out.sort_index()
