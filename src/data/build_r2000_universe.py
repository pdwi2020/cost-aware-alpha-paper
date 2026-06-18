"""Build Russell 2000 universe and download daily OHLCV (2010-01-01 to 2024-12-31).

Pipeline:
  Step 1 — Parse IWM/VTWO holdings CSV from datasets/russell2000_holdings/iwm_holdings.csv
            (or re-download from Vanguard VTWO public API if the file is absent/stale).
  Step 2 — Normalise tickers for yfinance: dots → dashes (e.g. "MOG.A" → "MOG-A").
            Keep the canonical (dot) form as the output ticker column value.
  Step 3 — Batch-download daily OHLCV via yfinance, batches of 40, 1-second sleep.
            auto_adjust=False → raw (unadjusted) prices, matching daily_ohlcv.parquet schema.
            Incremental parquet save after every batch.
  Step 4 — Filter: keep only tickers with earliest date ≤ 2013-01-31 AND >500 trading days.
  Step 5 — Write flat parquet [ticker, date, open, high, low, close, volume] + tickers CSV.

Usage:
    python src/data/build_r2000_universe.py

Outputs:
    datasets/russell2000_holdings/iwm_holdings.csv   (raw holdings, kept for audit)
    data/processed/r2000_tickers.csv                  (kept tickers, column "ticker")
    data/processed/daily_ohlcv_r2000.parquet          (flat OHLCV, raw prices)
"""

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
HOLDINGS_DIR = ROOT / "datasets" / "russell2000_holdings"
HOLDINGS_CSV = HOLDINGS_DIR / "iwm_holdings.csv"
OUT_PARQUET = ROOT / "data" / "processed" / "daily_ohlcv_r2000.parquet"
OUT_TICKERS_CSV = ROOT / "data" / "processed" / "r2000_tickers.csv"

HOLDINGS_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
BATCH_SIZE = 40
SLEEP_BETWEEN_BATCHES = 1.0
MIN_HISTORY_DATE = pd.Timestamp("2013-01-31")
MIN_TRADING_DAYS = 500
VANGUARD_VTWO_API = (
    "https://investor.vanguard.com/investment-products/etfs/profile/api/"
    "vtwo/portfolio-holding/stock"
)


# ---------------------------------------------------------------------------
# Step 1 — Load or re-download IWM/VTWO holdings
# ---------------------------------------------------------------------------

def _fetch_vtwo_from_vanguard() -> pd.DataFrame:
    """Fetch all VTWO (Russell 2000 proxy) holdings from Vanguard public API.

    Returns a DataFrame with columns [Ticker, Name, Asset Class, ...].
    """
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    all_entities = []
    start = 1
    count = 500
    total = None
    page = 0

    while True:
        url = f"{VANGUARD_VTWO_API}?start={start}&count={count}"
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            print(f"[holdings] Vanguard API error at start={start}: {exc}", flush=True)
            break

        if total is None:
            total = data.get("size", 0)
            print(f"[holdings] Vanguard VTWO reports {total} holdings", flush=True)

        entities = data.get("fund", {}).get("entity", [])
        all_entities.extend(entities)
        page += 1
        print(
            f"[holdings] page {page}: fetched {len(entities)}, cumulative {len(all_entities)}",
            flush=True,
        )

        if "next" not in data or len(all_entities) >= (total or len(all_entities) + 1):
            break
        start += count

    if not all_entities:
        raise RuntimeError("Vanguard VTWO API returned 0 entities — cannot build universe.")

    df = pd.DataFrame(all_entities)
    # Normalise to iShares-compatible column names
    holdings = pd.DataFrame(
        {
            "Ticker": df["ticker"],
            "Name": df["longName"],
            "Sector": "",
            "Asset Class": "Equity",
            "Market Value": df["marketValue"],
            "Weight (%)": df["percentWeight"],
            "Notional Value": df["notionalValue"],
            "Shares": df["sharesHeld"],
            "CUSIP": df["cusip"],
            "ISIN": df["isin"],
            "Exchange": "",
            "Currency": "USD",
            "FX Rate": "1.00",
            "Accrual Date": "",
        }
    )
    # Drop blank tickers (cash lines)
    holdings = holdings[holdings["Ticker"].str.strip() != ""].copy()
    return holdings


def load_holdings(force_refresh: bool = False) -> pd.DataFrame:
    """Load IWM/VTWO holdings from CSV, or download fresh if absent.

    The CSV file starts with a multi-line preamble.  The real header row begins
    with "Ticker," — we skip everything above it (mirrors iShares CSV format).

    Returns a DataFrame with at least columns [Ticker, Name, Asset Class].
    """
    if not HOLDINGS_CSV.exists() or force_refresh:
        print("[holdings] Downloading VTWO holdings from Vanguard public API …", flush=True)
        holdings = _fetch_vtwo_from_vanguard()
        # Persist with a human-readable preamble (iShares-compatible format)
        preamble = (
            "iShares Russell 2000 ETF (proxy: Vanguard VTWO)\n"
            "Source: Vanguard VTWO portfolio-holding API (asOf 2026-05-31)\n"
            "Note: VTWO tracks CRSP US Small Cap Index — constituent set matches Russell 2000\n"
            "Note: Dot notation for share classes  e.g. MOG.A  GEF.B  CRD.A\n"
            ",,,,,,,,,\n"
            "\n"
        )
        with open(HOLDINGS_CSV, "w") as fh:
            fh.write(preamble)
            holdings.to_csv(fh, index=False)
        print(f"[holdings] Saved {len(holdings)} rows → {HOLDINGS_CSV}", flush=True)
        return holdings

    # Parse existing file: skip preamble, find header row starting with "Ticker,"
    print(f"[holdings] Reading {HOLDINGS_CSV}", flush=True)
    with open(HOLDINGS_CSV, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Ticker,") or line.strip().startswith("Ticker\t"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            f"Could not find 'Ticker,' header row in {HOLDINGS_CSV}. "
            "Run with force_refresh=True to re-download."
        )

    from io import StringIO
    csv_text = "".join(lines[header_idx:])
    holdings = pd.read_csv(StringIO(csv_text))
    holdings.columns = holdings.columns.str.strip()
    print(
        f"[holdings] Loaded {len(holdings)} rows, columns: {holdings.columns.tolist()}",
        flush=True,
    )
    return holdings


# ---------------------------------------------------------------------------
# Step 2 — Parse and normalise tickers
# ---------------------------------------------------------------------------

EQUITY_CLASSES = {"Equity", "equity", "EQUITY"}
TICKER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,4}(\.[A-Za-z])?$")  # 1-6 chars, optional .X


def parse_equity_tickers(holdings: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Extract plausible equity tickers and build the dot→dash yfinance mapping.

    Args:
        holdings: DataFrame from load_holdings().

    Returns:
        canonical_tickers: list of canonical (dot-style) tickers, e.g. ["MOG.A", "AAPL"]
        yf_map: dict mapping canonical ticker → yfinance symbol ("MOG.A" → "MOG-A")
    """
    # Filter to equity rows
    asset_class_col = None
    for col in holdings.columns:
        if "asset" in col.lower() and "class" in col.lower():
            asset_class_col = col
            break

    if asset_class_col:
        equity_mask = holdings[asset_class_col].str.strip().isin(EQUITY_CLASSES)
        holdings = holdings[equity_mask].copy()
        print(
            f"[parse] {len(holdings)} equity rows after filtering on '{asset_class_col}'",
            flush=True,
        )
    else:
        print("[parse] No 'Asset Class' column found — keeping all rows.", flush=True)

    # Extract ticker column
    ticker_col = None
    for col in holdings.columns:
        if col.strip().lower() == "ticker":
            ticker_col = col
            break
    if ticker_col is None:
        raise ValueError(f"No 'Ticker' column in holdings. Columns: {holdings.columns.tolist()}")

    raw_tickers = holdings[ticker_col].dropna().str.strip().tolist()

    canonical_tickers: list[str] = []
    yf_map: dict[str, str] = {}
    skipped_invalid = 0

    for t in raw_tickers:
        if not t:
            continue
        if not TICKER_RE.match(t):
            skipped_invalid += 1
            continue
        canonical_tickers.append(t)
        # yfinance uses "-" for share-class separators, not "."
        yf_sym = t.replace(".", "-")
        yf_map[t] = yf_sym

    print(
        f"[parse] {len(canonical_tickers)} valid equity tickers "
        f"({skipped_invalid} skipped as non-equity/invalid symbol)",
        flush=True,
    )
    dot_tickers = [t for t in canonical_tickers if "." in t]
    print(f"[parse] Dot-class tickers ({len(dot_tickers)}): {dot_tickers}", flush=True)
    return canonical_tickers, yf_map


# ---------------------------------------------------------------------------
# Step 3 — Download OHLCV via yfinance
# ---------------------------------------------------------------------------

def _download_batch(
    batch_canonical: list[str],
    yf_map: dict[str, str],
    start: str,
    end: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Download one batch.  Returns (long_df, ok_canonical, failed_canonical)."""
    yf_batch = [yf_map[t] for t in batch_canonical]
    reverse_map = {v: k for k, v in yf_map.items()}  # yf_sym → canonical

    try:
        raw = yf.download(
            yf_batch,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        print(f"  [download] batch ERROR: {repr(exc)[:120]}", flush=True)
        return pd.DataFrame(), [], batch_canonical

    if raw.empty:
        return pd.DataFrame(), [], batch_canonical

    # Handle MultiIndex columns (multi-ticker download) vs single-ticker flat
    if isinstance(raw.columns, pd.MultiIndex):
        # Level 0 = price type (Open, High, Low, Close, Volume, Adj Close)
        # Level 1 = yfinance symbol
        price_types = {pt.lower(): pt for pt in raw.columns.get_level_values(0).unique()}
    else:
        # Single ticker — wrap in a MultiIndex-like structure
        raw.columns = pd.MultiIndex.from_tuples(
            [(col, yf_batch[0]) for col in raw.columns]
        )
        price_types = {pt.lower(): pt for pt in raw.columns.get_level_values(0).unique()}

    needed = ["open", "high", "low", "close", "volume"]
    for n in needed:
        if n not in price_types:
            print(f"  [download] missing '{n}' in downloaded columns", flush=True)
            return pd.DataFrame(), [], batch_canonical

    frames = []
    ok_canonical = []
    failed_canonical = []

    for yf_sym in raw.columns.get_level_values(1).unique():
        canonical = reverse_map.get(yf_sym, yf_sym)
        try:
            piece = pd.DataFrame(
                {
                    "open": raw[(price_types["open"], yf_sym)],
                    "high": raw[(price_types["high"], yf_sym)],
                    "low": raw[(price_types["low"], yf_sym)],
                    "close": raw[(price_types["close"], yf_sym)],
                    "volume": raw[(price_types["volume"], yf_sym)],
                }
            )
            piece = piece.dropna(subset=["close"])
            if len(piece) < 50:
                failed_canonical.append(canonical)
                continue
            piece = piece.reset_index().rename(columns={"Date": "date", "index": "date"})
            piece["ticker"] = canonical
            piece["date"] = pd.to_datetime(piece["date"]).dt.normalize()
            frames.append(piece[["ticker", "date", "open", "high", "low", "close", "volume"]])
            ok_canonical.append(canonical)
        except Exception as exc:
            print(f"  [download] {canonical} piece error: {repr(exc)[:80]}", flush=True)
            failed_canonical.append(canonical)

    # Tickers in request but not returned at all
    returned_yf = set(raw.columns.get_level_values(1).unique())
    for t_can, t_yf in yf_map.items():
        if t_can in batch_canonical and t_yf not in returned_yf:
            if t_can not in ok_canonical and t_can not in failed_canonical:
                failed_canonical.append(t_can)

    long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return long_df, ok_canonical, failed_canonical


def download_all_ohlcv(
    canonical_tickers: list[str],
    yf_map: dict[str, str],
) -> pd.DataFrame:
    """Download OHLCV for all tickers in batches; save incrementally to OUT_PARQUET.

    Returns the final combined long DataFrame.
    """
    all_frames: list[pd.DataFrame] = []
    all_ok: list[str] = []
    all_fail: list[str] = []
    n_batches = (len(canonical_tickers) + BATCH_SIZE - 1) // BATCH_SIZE

    print(
        f"[download] {len(canonical_tickers)} tickers, "
        f"{n_batches} batches of {BATCH_SIZE}",
        flush=True,
    )

    for batch_num, i in enumerate(range(0, len(canonical_tickers), BATCH_SIZE), start=1):
        batch = canonical_tickers[i : i + BATCH_SIZE]
        print(
            f"[download] batch {batch_num}/{n_batches} ({len(batch)} tickers) …",
            flush=True,
        )
        long_df, ok, fail = _download_batch(batch, yf_map, START_DATE, END_DATE)

        all_ok.extend(ok)
        all_fail.extend(fail)

        if not long_df.empty:
            all_frames.append(long_df)

        # Incremental save — survives crashes mid-pull
        if all_frames:
            combined = pd.concat(all_frames, ignore_index=True)
            combined.to_parquet(OUT_PARQUET, index=False)

        print(
            f"[download]   +{len(ok)} ok, +{len(fail)} fail "
            f"(cumulative ok={len(all_ok)}, fail={len(all_fail)})",
            flush=True,
        )
        time.sleep(SLEEP_BETWEEN_BATCHES)

    final = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    print(
        f"[download] DONE  total_ok={len(all_ok)}  total_fail={len(all_fail)}  "
        f"rows_before_filter={len(final)}",
        flush=True,
    )
    print(
        f"[download] sample failures (likely delisted/not in yf): "
        f"{sorted(all_fail)[:30]}",
        flush=True,
    )
    return final


# ---------------------------------------------------------------------------
# Step 4 — History filter
# ---------------------------------------------------------------------------

def apply_history_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only tickers with history back to MIN_HISTORY_DATE and >MIN_TRADING_DAYS rows.

    Returns (kept_df, dropped_df).
    """
    if df.empty:
        return df, df

    df["date"] = pd.to_datetime(df["date"])
    stats = df.groupby("ticker")["date"].agg(["min", "count"]).rename(
        columns={"min": "earliest", "count": "n_days"}
    )

    keep_mask = (stats["earliest"] <= MIN_HISTORY_DATE) & (stats["n_days"] > MIN_TRADING_DAYS)
    kept_tickers = set(stats.index[keep_mask])
    dropped_tickers = set(stats.index[~keep_mask])

    kept_df = df[df["ticker"].isin(kept_tickers)].copy()
    dropped_df = df[df["ticker"].isin(dropped_tickers)].copy()

    # Report top drop reasons
    dropped_stats = stats.loc[list(dropped_tickers)]
    n_too_new = int((dropped_stats["earliest"] > MIN_HISTORY_DATE).sum())
    n_too_sparse = int(
        (
            (dropped_stats["earliest"] <= MIN_HISTORY_DATE)
            & (dropped_stats["n_days"] <= MIN_TRADING_DAYS)
        ).sum()
    )
    n_both = len(dropped_tickers) - n_too_new - n_too_sparse

    print(
        f"[filter] kept={len(kept_tickers)}  dropped={len(dropped_tickers)} "
        f"(too_new={n_too_new}  too_sparse={n_too_sparse}  both={n_both})",
        flush=True,
    )
    return kept_df, dropped_df


# ---------------------------------------------------------------------------
# Step 5 — Write outputs and print summary
# ---------------------------------------------------------------------------

def write_outputs(kept_df: pd.DataFrame) -> None:
    """Sort, cast schema, write parquet + tickers CSV."""
    # Enforce schema column order and types
    kept_df = kept_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    kept_df["date"] = kept_df["date"].astype("datetime64[us]")
    for col in ["open", "high", "low", "close", "volume"]:
        kept_df[col] = kept_df[col].astype("float64")

    kept_df.to_parquet(OUT_PARQUET, index=False)
    print(f"[output] Parquet written: {OUT_PARQUET}  ({OUT_PARQUET.stat().st_size / 1e6:.1f} MB)", flush=True)

    tickers_kept = sorted(kept_df["ticker"].unique())
    pd.DataFrame({"ticker": tickers_kept}).to_csv(OUT_TICKERS_CSV, index=False)
    print(f"[output] Tickers CSV written: {OUT_TICKERS_CSV}  ({len(tickers_kept)} tickers)", flush=True)


def print_summary(
    n_parsed: int,
    n_attempted: int,
    kept_df: pd.DataFrame,
    dropped_df: pd.DataFrame,
) -> None:
    """Print the final human-readable pipeline summary."""
    n_kept = kept_df["ticker"].nunique() if not kept_df.empty else 0
    n_dropped = dropped_df["ticker"].nunique() if not dropped_df.empty else 0
    total_rows = len(kept_df)
    date_min = kept_df["date"].min() if not kept_df.empty else "N/A"
    date_max = kept_df["date"].max() if not kept_df.empty else "N/A"
    parquet_mb = OUT_PARQUET.stat().st_size / 1e6 if OUT_PARQUET.exists() else 0.0

    print("\n" + "=" * 70, flush=True)
    print("RUSSELL 2000 OHLCV BUILD — FINAL SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  IWM/VTWO equity constituents parsed  : {n_parsed}", flush=True)
    print(f"  Tickers attempted (download)         : {n_attempted}", flush=True)
    print(f"  Tickers KEPT (history >= 2013-01-31) : {n_kept}", flush=True)
    print(f"  Tickers DROPPED                      : {n_dropped}", flush=True)
    print(f"  Total rows in parquet                : {total_rows:,}", flush=True)
    print(f"  Date range                           : {date_min} → {date_max}", flush=True)
    print(f"  Output parquet                       : {OUT_PARQUET}  ({parquet_mb:.1f} MB)", flush=True)
    print(f"  Output tickers CSV                   : {OUT_TICKERS_CSV}", flush=True)
    print(f"  Holdings CSV                         : {HOLDINGS_CSV}", flush=True)
    print("=" * 70, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[r2000] Starting Russell 2000 OHLCV build …", flush=True)

    # Step 1
    holdings = load_holdings()

    # Step 2
    canonical_tickers, yf_map = parse_equity_tickers(holdings)
    n_parsed = len(canonical_tickers)
    n_attempted = n_parsed

    # Step 3
    raw_df = download_all_ohlcv(canonical_tickers, yf_map)

    if raw_df.empty:
        print("[r2000] ERROR: no data downloaded — aborting.", flush=True)
        sys.exit(1)

    # Step 4
    kept_df, dropped_df = apply_history_filter(raw_df)

    # Step 5
    write_outputs(kept_df)
    print_summary(n_parsed, n_attempted, kept_df, dropped_df)


if __name__ == "__main__":
    main()
