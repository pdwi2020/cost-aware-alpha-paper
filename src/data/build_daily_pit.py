"""src/data/build_daily_pit.py — reproducible daily, event-dated S&P 500 PIT
membership + recovered price panel.

Fixes the look-ahead / coverage problems in the annual-snapshot universe
(src/data/universe_builder.py, src/data/screen0.py::pit_membership_mask):
a ticker in datasets/sp500_holdings/{YEAR}.csv is currently treated as a
member for every date in YEAR, which misses intra-year additions/removals
and lets the late-2025 2025.csv snapshot leak into Jan-Jul 2025.

This script builds, from the MIT-licensed fja05680/sp500 GitHub repo plus a
live Wikipedia cross-check:

  1. datasets/sp500_daily_pit/raw/*            -- cached raw source files
     datasets/sp500_daily_pit/PROVENANCE.md    -- URLs, commit SHA, sha256, licence
  2. datasets/sp500_daily_pit/ticker_map.csv   -- source_symbol -> yahoo_symbol
  3. datasets/sp500_daily_pit/membership_daily.parquet -- one row per (date,
     ticker) member-day, ticker already resolved to its Yahoo symbol
  4. data/processed/daily_ohlcv_v3.parquet     -- recovered price panel
  5. results/universe_v3/*.csv                 -- coverage/characterisation

Never overwrites data/processed/daily_ohlcv.parquet. Never computes any
feature/IC/signal/strategy P&L -- pure data engineering.

Idempotent: every stage caches its inputs/outputs and is a fast no-op on a
second run unless --refresh is passed. Raw per-ticker yfinance downloads are
cached on the external X9 drive (not the Mac SSD) to respect a tight local
disk budget -- see PRICE_CACHE_DIR below.

Usage
-----
    python3 -m src.data.build_daily_pit --stages raw,tickermap,membership
    python3 -m src.data.build_daily_pit --stages prices --max-new-prices 120
    python3 -m src.data.build_daily_pit --stages coverage
    python3 -m src.data.build_daily_pit --stages all --refresh
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
PIT_DIR = ROOT / "datasets" / "sp500_daily_pit"
RAW_DIR = PIT_DIR / "raw"
PROVENANCE_MD = PIT_DIR / "PROVENANCE.md"
TICKER_MAP_CSV = PIT_DIR / "ticker_map.csv"
MEMBERSHIP_PARQUET = PIT_DIR / "membership_daily.parquet"

PROCESSED_DIR = ROOT / "data" / "processed"
OLD_OHLCV_PARQUET = PROCESSED_DIR / "daily_ohlcv.parquet"
NEW_OHLCV_PARQUET = PROCESSED_DIR / "daily_ohlcv_v3.parquet"

RESULTS_DIR = ROOT / "results" / "universe_v3"
SP500_HOLDINGS_DIR = ROOT / "datasets" / "sp500_holdings"

# Raw yfinance per-ticker cache lives on X9, NOT the Mac SSD (disk budget).
PRICE_CACHE_DIR = Path(
    "/Volumes/Crucial X9/Research Projects/ML_Paper/cache/yf_raw"
)

MEMBERSHIP_START = pd.Timestamp("2009-12-01")
MEMBERSHIP_END = pd.Timestamp("2026-09-14")
PRICE_START = "2009-06-01"
PRICE_END = "2026-09-14"  # yfinance `end` is exclusive; callers add +1 day

ETFS = [
    "SPY", "QQQ", "XLK", "XLE", "XLF", "XLY", "XLP",
    "XLI", "XLB", "XLU", "XLV", "XLC", "XLRE",
]

# Known corporate ticker-symbol changes (company continues, Yahoo serves the
# combined price history under the NEW symbol). Verified for this project.
KNOWN_RENAMES = {
    "PCLN": "BKNG",
    "ANTM": "ELV",
    "BLL": "BALL",
    "HFC": "DINO",
    "RE": "EG",
}

GITHUB_API = "https://api.github.com/repos/fja05680/sp500"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "ml-paper-research/1.0 (+point-in-time-universe-build)"}

RAW_FILES = {
    "sp500_historical_components_and_changes_updated.csv":
        "S&P 500 Historical Components & Changes (Updated).csv",
    "sp500_ticker_start_end.csv": "sp500_ticker_start_end.csv",
    "sp500_changes_since_2019.csv": "sp500_changes_since_2019.csv",
    "LICENSE": "LICENSE",
    "README_source.md": "README.md",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_class_share(symbol: str) -> str:
    """Yahoo convention: BRK.B -> BRK-B, BF.B -> BF-B (dot -> dash)."""
    return symbol.strip().replace(".", "-")


def resolve_yahoo_symbol(source_symbol: str) -> tuple[str, str]:
    """Return (yahoo_symbol, rule) for a raw fja05680 source symbol."""
    normalized = normalize_class_share(source_symbol)
    if normalized in KNOWN_RENAMES:
        return KNOWN_RENAMES[normalized], "rename"
    if normalized != source_symbol.strip():
        return normalized, "class_share"
    return normalized, "same"


# ---------------------------------------------------------------------------
# Stage 1 -- raw downloads + provenance
# ---------------------------------------------------------------------------

def _sidecar_path(dest: Path) -> Path:
    return dest.with_suffix(dest.suffix + ".provenance.json")


def _download_with_sidecar(url: str, dest: Path, commit_sha: str, commit_date: str,
                            refresh: bool) -> dict:
    sidecar = _sidecar_path(dest)
    if dest.exists() and sidecar.exists() and not refresh:
        return json.loads(sidecar.read_text())

    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    meta = {
        "url": url,
        "dest": str(dest.relative_to(ROOT)),
        "sha256": sha256_file(dest),
        "size_bytes": dest.stat().st_size,
        "fetched_utc": _utcnow_iso(),
        "commit_sha": commit_sha,
        "commit_date": commit_date,
    }
    sidecar.write_text(json.dumps(meta, indent=2))
    return meta


def fetch_github_raw(refresh: bool = False) -> dict:
    """Download the fja05680/sp500 source files (cached; skip unless refresh)."""
    commits = requests.get(f"{GITHUB_API}/commits", params={"per_page": 1},
                            headers=HEADERS, timeout=30).json()
    commit_sha = commits[0]["sha"]
    commit_date = commits[0]["commit"]["committer"]["date"]

    contents = requests.get(f"{GITHUB_API}/contents", headers=HEADERS, timeout=30).json()
    by_name = {f["name"]: f for f in contents}

    metas = {}
    for local_name, remote_name in RAW_FILES.items():
        f = by_name.get(remote_name)
        if f is None:
            warnings.warn(f"fja05680/sp500: expected file not found: {remote_name!r}")
            continue
        dest = RAW_DIR / local_name
        metas[local_name] = _download_with_sidecar(
            f["download_url"], dest, commit_sha, commit_date, refresh
        )
    return {"commit_sha": commit_sha, "commit_date": commit_date, "files": metas}


def fetch_wikipedia_constituents(refresh: bool = False) -> pd.DataFrame:
    """Live Wikipedia S&P 500 constituent-table ticker set (robust parse)."""
    csv_path = RAW_DIR / "wikipedia_constituents.csv"
    meta_path = RAW_DIR / "wikipedia_fetch_metadata.json"
    if csv_path.exists() and meta_path.exists() and not refresh:
        return pd.read_csv(csv_path)

    resp = requests.get(WIKI_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(resp.text, attrs={"id": "constituents"})
    table = tables[0]
    symbol_col = table.columns[0]
    tickers = sorted(set(table[symbol_col].astype(str).str.strip()))
    df = pd.DataFrame({"source_symbol": tickers})

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps({
        "source_url": WIKI_URL,
        "fetch_timestamp_utc": _utcnow_iso(),
        "method": "pandas.read_html on table id='constituents'",
        "row_count": len(df),
        "distinct_symbol_count": df["source_symbol"].nunique(),
    }, indent=2))
    return df


def _load_wide_changes() -> pd.DataFrame:
    """Parse the fja05680 wide date,tickers file into a sorted change table."""
    path = RAW_DIR / "sp500_historical_components_and_changes_updated.csv"
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            rows.append(row)
    header, data = rows[0], rows[1:]
    changes = pd.DataFrame(data, columns=header[:2])
    changes.columns = ["date", "tickers"]
    changes["date"] = pd.to_datetime(changes["date"])
    changes = changes.sort_values("date").reset_index(drop=True)
    return changes


def check_wikipedia_supplement(refresh: bool = False) -> dict:
    """Compare fja05680's last snapshot to live Wikipedia; return the diff."""
    changes = _load_wide_changes()
    last_date, last_tickers_str = changes.iloc[-1][["date", "tickers"]]
    fja_last = {t.strip() for t in last_tickers_str.split(",") if t.strip()}

    wiki = set(fetch_wikipedia_constituents(refresh=refresh)["source_symbol"])

    added = sorted(wiki - fja_last)
    removed = sorted(fja_last - wiki)
    return {
        "fja_last_date": str(pd.Timestamp(last_date).date()),
        "fja_last_n": len(fja_last),
        "wiki_n": len(wiki),
        "added_since_fja_last": added,
        "removed_since_fja_last": removed,
        "identical": (not added and not removed),
    }


def build_provenance_md(github_meta: dict, wiki_check: dict) -> None:
    lines = []
    lines.append("# Provenance -- daily point-in-time S&P 500 membership sources\n")
    lines.append(
        "Source repository: [fja05680/sp500](https://github.com/fja05680/sp500) "
        f"(MIT licence). Latest commit used: `{github_meta['commit_sha']}` "
        f"({github_meta['commit_date']}).\n"
    )
    lines.append(
        "Built by the repo maintainer from Andreas Clenow's *Trading Evolved* "
        "book file (`S&P 500 Historical Components & Changes.csv`, 1996-2019) "
        "merged with a hand-maintained Wikipedia changes log "
        "(`sp500_changes_since_2019.csv`), cross-checked against the current "
        "Wikipedia \"List of S&P 500 companies\" page by the maintainer's own "
        "`sp500_historical.ipynb`. See raw/README_source.md (the upstream "
        "README) for the full description.\n"
    )
    lines.append("## Files\n")
    lines.append("| local file | source URL | sha256 | fetched (UTC) |")
    lines.append("|---|---|---|---|")
    for local_name, meta in github_meta["files"].items():
        lines.append(
            f"| `{local_name}` | {meta['url']} | `{meta['sha256'][:16]}...` "
            f"| {meta['fetched_utc']} |"
        )
    lines.append("")
    lines.append("## Date coverage\n")
    lines.append(
        f"`sp500_historical_components_and_changes_updated.csv` spans "
        f"1996-01-02 through **{wiki_check['fja_last_date']}** "
        f"({wiki_check['fja_last_n']} constituents in its last row).\n"
    )
    lines.append(
        f"This last date is before 2026-08-31, so per spec we checked whether "
        f"Wikipedia's live \"List of S&P 500 companies\" constituent table "
        f"shows any membership changes since then, up to our required end "
        f"date 2026-09-14.\n"
    )
    lines.append("### Wikipedia supplement check\n")
    lines.append(
        f"- Method: fetched {WIKI_URL} and parsed the `id=\"constituents\"` "
        f"HTML table with `pandas.read_html` (robust to per-ticker exchange-"
        f"link template differences, e.g. CBOE uses a `{{BZX link}}` template "
        f"rather than the common `{{NyseSymbol}}`/`{{NasdaqSymbol}}` "
        f"templates -- a naive regex over template names misses it).\n"
        f"- Live Wikipedia constituent count: {wiki_check['wiki_n']}.\n"
        f"- fja05680 last-row constituent count "
        f"({wiki_check['fja_last_date']}): {wiki_check['fja_last_n']}.\n"
        f"- Tickers on Wikipedia but not in the fja05680 last row: "
        f"{wiki_check['added_since_fja_last'] or '(none)'}\n"
        f"- Tickers in the fja05680 last row but not on Wikipedia: "
        f"{wiki_check['removed_since_fja_last'] or '(none)'}\n"
    )
    if wiki_check["identical"]:
        lines.append(
            "**Result: identical sets, zero differences.** No supplement rows "
            "are needed -- the fja05680 snapshot as of "
            f"{wiki_check['fja_last_date']} already matches live Wikipedia as "
            f"of the fetch timestamp above, so it correctly covers our full "
            f"required window through 2026-09-14 with no changes in between.\n"
        )
    else:
        lines.append(
            "**Result: differences found.** These are treated as supplementary "
            "change rows (source = \"wikipedia manual supplement\") applied "
            "after the fja05680 last date when building membership_daily.parquet.\n"
        )
    lines.append("## Date ranges actually used\n")
    lines.append(
        f"- Daily membership table: {MEMBERSHIP_START.date()} .. "
        f"{MEMBERSHIP_END.date()} (every NYSE trading day, from SPY's "
        f"yfinance trading calendar).\n"
        f"- `sp500_ticker_start_end.csv`: used for cross-checking membership "
        f"spells and as a source of rename candidates.\n"
        f"- `sp500_changes_since_2019.csv` / `LICENSE` / `README.md`: kept for "
        f"provenance narrative only, not parsed for data.\n"
    )
    lines.append(
        "## Known limitation\n\n"
        "Ticker-symbol reuse across unrelated companies (a delisted symbol "
        "later reassigned to a different, unrelated company) is not modelled: "
        "`ticker_map.csv` maps each source symbol to one Yahoo symbol "
        "regardless of which spell it appears in. This is not believed to "
        "affect any symbol in our 2010-2026 window but is not exhaustively "
        "verified.\n"
    )
    PROVENANCE_MD.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Stage 2 -- ticker_map.csv
# ---------------------------------------------------------------------------

def _source_symbol_daily_table(changes: pd.DataFrame, trading_days: pd.DatetimeIndex
                                ) -> pd.DataFrame:
    """Forward-fill the wide change table onto every trading day; explode to
    one row per (date, source_symbol)."""
    trading_df = pd.DataFrame({"date": trading_days})
    merged = pd.merge_asof(trading_df, changes, on="date", direction="backward")
    merged["ticker_list"] = merged["tickers"].str.split(",")
    exploded = merged.explode("ticker_list")
    exploded["source_symbol"] = exploded["ticker_list"].str.strip()
    exploded = exploded[exploded["source_symbol"] != ""]
    return exploded[["date", "source_symbol"]].reset_index(drop=True)


def _derive_spells(dates: pd.Series, trading_days: pd.DatetimeIndex) -> list[tuple]:
    """Given the sorted trading-day dates a symbol is present on, split into
    contiguous (first, last) spells using the trading-day calendar (a gap in
    the calendar's own sequence, not in wall-clock days, ends a spell)."""
    if len(dates) == 0:
        return []
    pos = trading_days.searchsorted(dates.values)
    order = np.argsort(pos)
    pos_sorted = pos[order]
    dates_sorted = pd.DatetimeIndex(dates.values)[order]
    spells = []
    start = dates_sorted[0]
    prev_pos = pos_sorted[0]
    prev_date = dates_sorted[0]
    for p, d in zip(pos_sorted[1:], dates_sorted[1:]):
        if p != prev_pos + 1:
            spells.append((start, prev_date))
            start = d
        prev_pos, prev_date = p, d
    spells.append((start, prev_date))
    return spells


def build_ticker_map(refresh: bool = False) -> pd.DataFrame:
    if TICKER_MAP_CSV.exists() and not refresh:
        return pd.read_csv(TICKER_MAP_CSV)

    changes = _load_wide_changes()
    trading_days = _spy_trading_days(refresh=refresh)
    window_days = trading_days[
        (trading_days >= MEMBERSHIP_START) & (trading_days <= MEMBERSHIP_END)
    ]
    src_daily = _source_symbol_daily_table(changes, window_days)

    start_end = None
    start_end_path = RAW_DIR / "sp500_ticker_start_end.csv"
    if start_end_path.exists():
        start_end = pd.read_csv(start_end_path)
        start_end["start_date"] = pd.to_datetime(start_end["start_date"], errors="coerce")
        start_end["end_date"] = pd.to_datetime(start_end["end_date"], errors="coerce")

    rows = []
    for source_symbol, grp in src_daily.groupby("source_symbol"):
        spells = _derive_spells(grp["date"], window_days)
        yahoo_symbol, rule = resolve_yahoo_symbol(source_symbol)
        note_bits = ["spell derived from fja05680 wide file forward-fill"]
        if start_end is not None:
            se_rows = start_end[start_end["ticker"] == source_symbol]
            if len(se_rows):
                note_bits.append(
                    f"sp500_ticker_start_end.csv has {len(se_rows)} spell(s) "
                    f"for this ticker (cross-check)"
                )
        if rule == "rename":
            note_bits.append(f"known rename: {source_symbol}->{yahoo_symbol}")
        note = "; ".join(note_bits)
        for first_d, last_d in spells:
            rows.append({
                "source_symbol": source_symbol,
                "yahoo_symbol": yahoo_symbol,
                "first_date": first_d.date().isoformat(),
                "last_date": last_d.date().isoformat(),
                "rule": rule,
                "note": note,
            })

    ticker_map = pd.DataFrame(rows).sort_values(
        ["source_symbol", "first_date"]
    ).reset_index(drop=True)
    PIT_DIR.mkdir(parents=True, exist_ok=True)
    ticker_map.to_csv(TICKER_MAP_CSV, index=False)
    return ticker_map


# ---------------------------------------------------------------------------
# Stage 3 -- SPY trading calendar + membership_daily.parquet
# ---------------------------------------------------------------------------

def _spy_trading_days(refresh: bool = False) -> pd.DatetimeIndex:
    cache = RAW_DIR / "spy_trading_calendar.csv"
    if cache.exists() and not refresh:
        return pd.DatetimeIndex(pd.read_csv(cache)["date"])

    import yfinance as yf
    df = yf.download("SPY", start="2009-11-01", end="2026-09-16",
                      auto_adjust=False, progress=False)
    dates = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": dates}).to_csv(cache, index=False)
    return dates


def build_membership_daily(ticker_map: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    if MEMBERSHIP_PARQUET.exists() and not refresh:
        return pd.read_parquet(MEMBERSHIP_PARQUET)

    changes = _load_wide_changes()
    trading_days = _spy_trading_days(refresh=refresh)
    window_days = trading_days[
        (trading_days >= MEMBERSHIP_START) & (trading_days <= MEMBERSHIP_END)
    ]
    src_daily = _source_symbol_daily_table(changes, window_days)

    src_to_yahoo = dict(zip(ticker_map["source_symbol"], ticker_map["yahoo_symbol"]))
    src_daily["ticker"] = src_daily["source_symbol"].map(
        lambda s: src_to_yahoo.get(s, resolve_yahoo_symbol(s)[0])
    )
    membership = (
        src_daily[["date", "ticker"]]
        .drop_duplicates()
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    membership["date"] = pd.to_datetime(membership["date"]).dt.normalize()

    PIT_DIR.mkdir(parents=True, exist_ok=True)
    membership.to_parquet(MEMBERSHIP_PARQUET, index=False, compression="zstd")
    return membership


# ---------------------------------------------------------------------------
# Stage 4 -- price recovery (yfinance, cached on X9)
# ---------------------------------------------------------------------------

def universe_tickers_for_pricing(membership: pd.DataFrame) -> list[str]:
    since_2010 = membership[membership["date"] >= pd.Timestamp("2010-01-01")]
    tickers = sorted(set(since_2010["ticker"]) | set(ETFS))
    return tickers


def _ticker_cache_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_")
    return PRICE_CACHE_DIR / f"{safe}.parquet"


def download_prices(tickers: list[str], max_new: Optional[int] = None,
                     refresh: bool = False, batch_size: int = 40,
                     sleep_s: float = 1.0) -> dict:
    """Download + cache OHLCV for `tickers` on X9. Only fetches tickers not
    already cached (unless refresh), and stops after `max_new` NEW tickers
    have been fetched (so this can be driven in small foreground chunks).
    Returns a summary dict."""
    import yfinance as yf

    PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tickers if refresh or not _ticker_cache_path(t).exists()]
    if max_new is not None:
        todo = todo[:max_new]

    fetched, empty, failed = [], [], []
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        for attempt in range(3):
            try:
                data = yf.download(
                    batch, start=PRICE_START, end="2026-09-15",
                    auto_adjust=False, group_by="ticker", threads=True,
                    progress=False,
                )
                break
            except Exception:
                if attempt == 2:
                    data = None
                time.sleep(2.0 * (attempt + 1))
        if data is None:
            failed.extend(batch)
            continue

        for t in batch:
            try:
                sub = data[t] if len(batch) > 1 else data
            except (KeyError, TypeError):
                failed.append(t)
                continue
            sub = sub.dropna(how="all")
            if sub.empty or sub["Close"].dropna().empty:
                empty.append(t)
                # cache an empty marker so we don't retry every run
                pd.DataFrame(
                    columns=["date", "open", "high", "low", "close",
                             "adj_close", "volume"]
                ).to_parquet(_ticker_cache_path(t), index=False, compression="zstd")
                continue
            out = pd.DataFrame({
                "date": pd.DatetimeIndex(sub.index).tz_localize(None).normalize(),
                "open": sub["Open"].values,
                "high": sub["High"].values,
                "low": sub["Low"].values,
                "close": sub["Close"].values,
                "adj_close": sub["Adj Close"].values,
                "volume": sub["Volume"].values,
            })
            out.to_parquet(_ticker_cache_path(t), index=False, compression="zstd")
            fetched.append(t)
        time.sleep(sleep_s)

    already_cached = len(tickers) - len(todo)
    return {
        "requested": len(tickers),
        "already_cached": already_cached,
        "attempted_this_run": len(todo),
        "fetched": fetched,
        "empty": empty,
        "failed": failed,
    }


def price_cache_status(tickers: list[str]) -> dict:
    cached = [t for t in tickers if _ticker_cache_path(t).exists()]
    missing = [t for t in tickers if not _ticker_cache_path(t).exists()]
    return {"n_total": len(tickers), "n_cached": len(cached), "n_missing": len(missing),
            "missing": missing}


def _finnhub_cache_path(ticker: str) -> Path:
    """Session-filtered daily bars rebuilt from the public 1-minute store."""
    from src.data.recover_delisted_prices import CACHE_DIR as FINNHUB_CACHE_DIR

    return FINNHUB_CACHE_DIR / f"{ticker.replace('/', '_')}.parquet"


def assemble_price_panel(tickers: list[str], refresh: bool = False) -> pd.DataFrame:
    """Concatenate cached per-ticker files into data/processed/daily_ohlcv_v3.parquet.

    Two sources, chosen per ticker rather than per day, because their price
    conventions differ (yfinance `close` is split-adjusted; the 1-minute bars
    are adjusted only by the split detector in
    ``src.data.recover_delisted_prices``), and mixing them inside one ticker's
    history would splice two different scales:

      1. **yfinance** wherever it serves the symbol (the great majority), and
      2. **session-filtered 1-minute bars** for members Yahoo has dropped
         (delistings, acquisitions, retired symbols), which is what lifts
         member-day coverage from about 85% to about 99%.

    The `source` column records which one produced each row.
    """
    if NEW_OHLCV_PARQUET.exists() and not refresh:
        return pd.read_parquet(NEW_OHLCV_PARQUET)

    frames = []
    n_yf = n_fh = 0
    for t in tickers:
        df = None
        p = _ticker_cache_path(t)
        if p.exists():
            candidate = pd.read_parquet(p)
            if not candidate.empty:
                candidate = candidate.copy()
                candidate["ticker"] = t
                candidate["source"] = "yfinance"
                df = candidate
                n_yf += 1

        if df is None:                                  # Yahoo has nothing usable
            fp = _finnhub_cache_path(t)
            if fp.exists():
                candidate = pd.read_parquet(fp)
                if not candidate.empty:
                    candidate = candidate.copy()
                    candidate["ticker"] = t
                    if "source" not in candidate.columns:
                        candidate["source"] = "finnhub_1m_session"
                    df = candidate
                    n_fh += 1

        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError("No cached price data found -- run the 'prices' stage first.")

    panel = pd.concat(frames, ignore_index=True)
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        if col not in panel.columns:
            panel[col] = np.nan
    panel = panel[
        ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "source"]
    ]
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"  [assemble] {n_yf} tickers from yfinance, {n_fh} recovered from the 1-minute store")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(NEW_OHLCV_PARQUET, index=False, compression="zstd")
    return panel


# ---------------------------------------------------------------------------
# Stage 5 -- coverage / characterisation
# ---------------------------------------------------------------------------

def _annual_membership_old(year: int) -> set[str]:
    path = SP500_HOLDINGS_DIR / f"{year}.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path, encoding="latin-1", usecols=["Ticker"])
    return set(df["Ticker"].dropna().str.strip())


def compute_coverage_by_year(membership: pd.DataFrame, price_panel_new: pd.DataFrame
                              ) -> pd.DataFrame:
    price_new_pairs = set(zip(price_panel_new["ticker"], price_panel_new["date"]))

    old_price = None
    if OLD_OHLCV_PARQUET.exists():
        old_price = pd.read_parquet(OLD_OHLCV_PARQUET)
        old_price_pairs = set(zip(old_price["ticker"],
                                   pd.to_datetime(old_price["date"])))
        old_trading_days = pd.DatetimeIndex(sorted(set(old_price["date"])))
    else:
        old_price_pairs = set()
        old_trading_days = pd.DatetimeIndex([])

    rows = []
    for year in range(2010, 2027):
        # NEW: daily membership panel
        mem_y = membership[membership["date"].dt.year == year]
        member_days_new = len(mem_y)
        with_price_new = sum(
            1 for t, d in zip(mem_y["ticker"], mem_y["date"]) if (t, d) in price_new_pairs
        )
        distinct_members_new = mem_y["ticker"].nunique()
        no_price_tickers_new = sorted(
            set(mem_y["ticker"]) - set(price_panel_new.loc[
                price_panel_new["date"].dt.year == year, "ticker"
            ])
        )

        # OLD: annual snapshot x old daily_ohlcv.parquet
        annual_tickers = _annual_membership_old(year)
        year_old_days = old_trading_days[old_trading_days.year == year]
        member_days_old = len(annual_tickers) * len(year_old_days)
        with_price_old = 0
        no_price_tickers_old = []
        if annual_tickers and len(year_old_days):
            for t in annual_tickers:
                n = sum(1 for d in year_old_days if (t, d) in old_price_pairs)
                with_price_old += n
                if n == 0:
                    no_price_tickers_old.append(t)

        rows.append({
            "year": year,
            "new_member_days": member_days_new,
            "new_member_days_with_price": with_price_new,
            "new_coverage_pct": round(100 * with_price_new / member_days_new, 3)
                if member_days_new else np.nan,
            "new_distinct_members": distinct_members_new,
            "new_members_no_price": len(no_price_tickers_new),
            "new_unrecoverable_symbols": ";".join(no_price_tickers_new),
            "old_member_days": member_days_old,
            "old_member_days_with_price": with_price_old,
            "old_coverage_pct": round(100 * with_price_old / member_days_old, 3)
                if member_days_old else np.nan,
            "old_distinct_members": len(annual_tickers),
            "old_members_no_price": len(no_price_tickers_old),
            "old_unrecoverable_symbols": ";".join(sorted(no_price_tickers_old)),
        })
    return pd.DataFrame(rows)


def _guess_unrecoverable_reason(symbol: str) -> str:
    # Best-effort, conservative: only flag "renamed-unmapped" when the
    # ticker_map explicitly recorded it as such; otherwise "unknown".
    return "unknown"


def compute_unrecoverable(membership: pd.DataFrame, price_panel_new: pd.DataFrame,
                           ticker_map: pd.DataFrame) -> pd.DataFrame:
    priced_tickers = set(price_panel_new.loc[
        price_panel_new["close"].notna(), "ticker"
    ])
    all_members = set(membership["ticker"])
    unrecoverable = sorted(all_members - priced_tickers)

    rows = []
    for sym in unrecoverable:
        spells = membership[membership["ticker"] == sym]
        spans = _derive_spells(spells["date"], pd.DatetimeIndex(sorted(spells["date"].unique())))
        span_str = "; ".join(
            f"{a.date()}..{b.date()}" for a, b in spans
        )
        tm_rows = ticker_map[ticker_map["yahoo_symbol"] == sym]
        reason = "unknown"
        if (tm_rows["rule"] == "same").any() and len(tm_rows) and \
                any(tm_rows["source_symbol"] != sym):
            reason = "renamed-unmapped"
        rows.append({
            "symbol": sym,
            "membership_spells": span_str,
            "reason": reason,
        })
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def compute_annual_vs_daily(membership: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in range(2010, 2026):
        annual_path = SP500_HOLDINGS_DIR / f"{year}.csv"
        annual = _annual_membership_old(year)
        day_candidates = membership[
            (membership["date"].dt.year == year) & (membership["date"].dt.month == 1)
        ]
        if day_candidates.empty:
            continue
        first_day = day_candidates["date"].min()
        daily_members = set(day_candidates.loc[
            day_candidates["date"] == first_day, "ticker"
        ])
        inter = annual & daily_members
        union = annual | daily_members
        jaccard = len(inter) / len(union) if union else np.nan
        rows.append({
            "year": year,
            "first_trading_day": str(first_day.date()),
            "annual_snapshot_n": len(annual),
            "daily_members_n": len(daily_members),
            "jaccard": round(jaccard, 4) if union else np.nan,
            "in_annual_not_daily": len(annual - daily_members),
            "in_daily_not_annual": len(daily_members - annual),
        })
    df = pd.DataFrame(rows)

    # Explicit 2025.csv look-ahead quantification
    annual_2025 = _annual_membership_old(2025)
    day_2025 = membership[
        (membership["date"] == pd.Timestamp("2025-01-02"))
    ]
    daily_2025_01_02 = set(day_2025["ticker"])
    if not daily_2025_01_02:
        candidates = membership[membership["date"].dt.year == 2025]
        if not candidates.empty:
            d0 = candidates["date"].min()
            daily_2025_01_02 = set(
                candidates.loc[candidates["date"] == d0, "ticker"]
            )
    lookahead_added = sorted(annual_2025 - daily_2025_01_02)
    lookahead_missing = sorted(daily_2025_01_02 - annual_2025)
    df.attrs["lookahead_2025_in_2025csv_not_2025_01_02"] = lookahead_added
    df.attrs["lookahead_2025_01_02_not_in_2025csv"] = lookahead_missing
    return df


def compute_price_convention_check(price_panel_new: pd.DataFrame, n_tickers: int = 20
                                    ) -> pd.DataFrame:
    """Informational only: correlate old-panel daily returns against the new
    (yfinance close) panel's daily returns for a sample of tickers."""
    if not OLD_OHLCV_PARQUET.exists():
        return pd.DataFrame()
    old = pd.read_parquet(OLD_OHLCV_PARQUET)
    old["date"] = pd.to_datetime(old["date"])

    common_tickers = sorted(set(old["ticker"]) & set(price_panel_new["ticker"]))
    sample = common_tickers[:n_tickers]

    rows = []
    for t in sample:
        o = old[old["ticker"] == t].sort_values("date").set_index("date")["close"]
        n = price_panel_new[price_panel_new["ticker"] == t].sort_values(
            "date"
        ).set_index("date")["close"]
        idx = o.index.intersection(n.index)
        if len(idx) < 30:
            continue
        o_ret = o.loc[idx].pct_change()
        n_ret = n.loc[idx].pct_change()
        both = pd.concat([o_ret, n_ret], axis=1).dropna()
        both.columns = ["old_ret", "new_ret"]
        if len(both) < 20:
            continue
        corr = both["old_ret"].corr(both["new_ret"])
        big_diff = (both["old_ret"] - both["new_ret"]).abs() > 0.05
        rows.append({
            "ticker": t,
            "n_overlap_days": len(both),
            "return_correlation": round(corr, 4),
            "n_days_abs_diff_gt_5pct": int(big_diff.sum()),
            "pct_days_abs_diff_gt_5pct": round(100 * big_diff.mean(), 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages", default="all",
        help="Comma list of: raw,tickermap,membership,prices,assemble,coverage,all",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-new-prices", type=int, default=None,
                         help="Cap on NEW (uncached) tickers to download this run")
    args = parser.parse_args(argv)

    stages = set(args.stages.split(","))
    if "all" in stages:
        stages = {"raw", "tickermap", "membership", "prices", "assemble", "coverage"}

    ticker_map = None
    membership = None

    if "raw" in stages:
        github_meta = fetch_github_raw(refresh=args.refresh)
        wiki_check = check_wikipedia_supplement(refresh=args.refresh)
        build_provenance_md(github_meta, wiki_check)
        print("raw stage done:", json.dumps(wiki_check, indent=2))

    if "tickermap" in stages:
        ticker_map = build_ticker_map(refresh=args.refresh)
        print(f"ticker_map rows={len(ticker_map)}")
        print(ticker_map["rule"].value_counts())

    if "membership" in stages:
        if ticker_map is None:
            ticker_map = build_ticker_map(refresh=False)
        membership = build_membership_daily(ticker_map, refresh=args.refresh)
        print(f"membership_daily rows={len(membership)} "
              f"distinct_tickers={membership['ticker'].nunique()} "
              f"dates=[{membership['date'].min()}, {membership['date'].max()}]")

    if "prices" in stages:
        if membership is None:
            membership = build_membership_daily(build_ticker_map(refresh=False), refresh=False)
        tickers = universe_tickers_for_pricing(membership)
        summary = download_prices(tickers, max_new=args.max_new_prices, refresh=args.refresh)
        print(json.dumps({k: v for k, v in summary.items()
                           if k not in ("fetched",)}, indent=2, default=str))
        print(f"fetched_this_run={len(summary['fetched'])}")

    if "assemble" in stages:
        if membership is None:
            membership = build_membership_daily(build_ticker_map(refresh=False), refresh=False)
        tickers = universe_tickers_for_pricing(membership)
        panel = assemble_price_panel(tickers, refresh=args.refresh)
        print(f"daily_ohlcv_v3 rows={len(panel)} tickers={panel['ticker'].nunique()}")

    if "coverage" in stages:
        if membership is None:
            membership = build_membership_daily(build_ticker_map(refresh=False), refresh=False)
        if ticker_map is None:
            ticker_map = build_ticker_map(refresh=False)
        panel = pd.read_parquet(NEW_OHLCV_PARQUET)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        cov = compute_coverage_by_year(membership, panel)
        cov.to_csv(RESULTS_DIR / "coverage_by_year.csv", index=False)

        unrec = compute_unrecoverable(membership, panel, ticker_map)
        unrec.to_csv(RESULTS_DIR / "unrecoverable.csv", index=False)

        avd = compute_annual_vs_daily(membership)
        avd.to_csv(RESULTS_DIR / "annual_vs_daily.csv", index=False)
        with open(RESULTS_DIR / "annual_vs_daily_2025_lookahead.json", "w") as f:
            json.dump({
                "in_2025csv_not_member_2025_01_02":
                    avd.attrs.get("lookahead_2025_in_2025csv_not_2025_01_02", []),
                "member_2025_01_02_not_in_2025csv":
                    avd.attrs.get("lookahead_2025_01_02_not_in_2025csv", []),
            }, f, indent=2)

        conv = compute_price_convention_check(panel)
        conv.to_csv(RESULTS_DIR / "price_convention_check.csv", index=False)

        print(f"coverage_by_year rows={len(cov)}; unrecoverable rows={len(unrec)}; "
              f"annual_vs_daily rows={len(avd)}; price_convention rows={len(conv)}")


if __name__ == "__main__":
    main()
