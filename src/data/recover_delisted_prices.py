"""Recover daily bars for index members that Yahoo no longer serves.

Why this exists
---------------
Reviewer 1 (point 3) objected that "a lagged price filter cannot recover
observations that are missing from the downloaded panel", leaving the claimed
removal of survivorship bias unresolved. Rebuilding the universe daily and
point-in-time made the gap measurable: of 862 symbols that are S&P 500 members
at some point in 2010-2026, yfinance serves 654 and drops 208 (delistings,
acquisitions and old ticker symbols), which is 15.2% of all member-days.

191 of those 208 are present in the public 1-minute store (Hugging Face
`mito0o852/OHLCV-1m`, Finnhub-sourced; see datasets/OHLCV_1M_PROVENANCE.md), so
their daily bars can be rebuilt from a second free public source, lifting
member-day coverage from about 85% to about 99%.

Two corrections matter when reading that store.

**Regular session only.** The Array-era pipeline aggregated
`LAST(close ORDER BY timestamp)` over every bar, and the store carries extended
hours (roughly 04:00-20:00 ET), so its "daily close" was the last post-market
print. Verified 2026-09-15: ACN on 2013-06-03 came out as 81.28, a 19:50 print,
where the official close is 82.10 (which yfinance matches exactly). This module
aggregates 09:30-16:00 ET, which reproduces the official close.

**Splits.** These bars are unadjusted, so a 2-for-1 split looks like a -50% day.
`detect_split_factors` flags a candidate only when a large price jump is matched
by a compensating volume jump, and `apply_split_adjustment` rebuilds a
continuous series from it.

Run
---
    python3 -u -m src.data.recover_delisted_prices --limit 20     # trial
    python3 -u -m src.data.recover_delisted_prices                # all missing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

ONE_MIN_DIR = Path(
    "/Volumes/Crucial X9/data/market_data/equities/ohlcv_1m_us/data"
)
CACHE_DIR = Path(
    "/Volumes/Crucial X9/Research Projects/ML_Paper/cache/finnhub_daily"
)
MEMBERSHIP = ROOT / "datasets" / "sp500_daily_pit" / "membership_daily.parquet"
UNRECOVERABLE = ROOT / "results" / "universe_v3" / "unrecoverable.csv"
REPORT = ROOT / "results" / "universe_v3" / "recovered_from_1min.csv"
SPLITS_REPORT = ROOT / "results" / "universe_v3" / "splits_adjusted.csv"

SESSION_START = "09:30:00"
SESSION_END = "16:00:00"

# Split detection thresholds.
MIN_SPLIT_MOVE = 0.30        # |return| that triggers a candidate
VOL_RATIO_TOL = 0.40         # relative tolerance on the volume cross-check
KNOWN_RATIOS = (1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0)
RATIO_TOL = 0.06             # how close price ratio must be to a known ratio

# A genuine split happens on an ordinary trading day: the corporate action
# changes the share count, not the company's value, so the days around it are
# unremarkable. A collapsing stock is the opposite. Verified on this data:
# ABC's real 2-for-1 (2009-06-16) has neighbours moving about 1-2%, while
# First Republic's 2023-03-20 crash (-47.5% on 1.4x volume, which otherwise
# passes the price-ratio and volume checks) sits among -33.6%, -20.4% and
# +31.3% days. Without this guard that crash is "adjusted" away and the whole
# pre-crash history is rescaled -- the exact distressed-name corruption Screen 0
# exists to catch.
#
# Deliberately NOT a trailing-volatility guard: legitimate reverse splits
# happen in already-volatile names, so that would reject the real ones.
CONTEXT_WINDOW = 3           # trading days inspected either side
CONTEXT_MAX_ABS_RET = 0.15   # neighbours must all be calmer than this


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Session-filtered daily aggregation
# ---------------------------------------------------------------------------

def month_files(start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    """The monthly 1-minute files covering [start, end]."""
    months = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
    files = [ONE_MIN_DIR / f"ohlcv_{p.strftime('%Y-%m')}.parquet" for p in months]
    return [f for f in files if f.exists()]


def session_daily_bars(con, path: Path, tickers: list[str]) -> pd.DataFrame:
    """Aggregate one monthly file to daily bars, regular session only.

    Uses the 09:30-16:00 ET window, so the close is the official close rather
    than a post-market print.
    """
    if not tickers:
        return pd.DataFrame()
    quoted = ", ".join("'" + t.replace("'", "''") + "'" for t in tickers)
    query = f"""
        WITH b AS (
            SELECT ticker,
                   (timestamp AT TIME ZONE 'America/New_York') AS ts_et,
                   open, high, low, close, volume
            FROM read_parquet('{path.as_posix()}')
            WHERE ticker IN ({quoted})
        )
        SELECT ticker,
               ts_et::DATE                              AS date,
               arg_min(open, ts_et)                     AS open,
               max(high)                                AS high,
               min(low)                                 AS low,
               arg_max(close, ts_et)                    AS close,
               sum(volume)                              AS volume
        FROM b
        WHERE ts_et::TIME BETWEEN '{SESSION_START}' AND '{SESSION_END}'
        GROUP BY ticker, ts_et::DATE
        ORDER BY ticker, date
    """
    return con.execute(query).fetchdf()


# ---------------------------------------------------------------------------
# Split handling
# ---------------------------------------------------------------------------

def _nearest_known_ratio(ratio: float) -> float | None:
    """Snap a price ratio to the CLOSEST conventional split ratio, or None.

    Distance is relative, and every candidate is compared before choosing. An
    absolute tolerance would misbehave on reverse splits, where the candidates
    are fractions: a 1-for-10 ratio of 0.105 sits within 0.06 of both 0.1 and
    1/7 = 0.143, and first-match-wins would return whichever came first in the
    list. Relative distance picks 0.1 (5% away) over 1/7 (27% away).
    """
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    best, best_dist = None, np.inf
    for known in KNOWN_RATIOS:
        for candidate in (known, 1.0 / known):
            dist = abs(ratio - candidate) / candidate
            if dist < best_dist:
                best, best_dist = candidate, dist
    return best if best_dist <= RATIO_TOL else None


def detect_split_factors(df: pd.DataFrame) -> list[dict]:
    """Find likely splits in one ticker's unadjusted daily bars.

    A candidate needs both legs of the evidence: a large price jump, and a
    volume jump in the opposite direction of roughly the same ratio (a 2-for-1
    split halves the price and doubles the share count traded). Price moves
    without volume corroboration are left alone, so genuine crashes are not
    mistaken for splits.

    Returns a list of {date, price_ratio, volume_ratio, factor}, where `factor`
    is the number of new shares per old share (2.0 for a 2-for-1 split, 0.1 for
    a 1-for-10 reverse split).
    """
    d = df.sort_values("date").reset_index(drop=True)
    close = d["close"].to_numpy(dtype=float)
    volume = d["volume"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])

    out = []
    for i in range(1, len(d)):
        prev_c, cur_c = close[i - 1], close[i]
        if not (np.isfinite(prev_c) and np.isfinite(cur_c)) or prev_c <= 0 or cur_c <= 0:
            continue
        ret = cur_c / prev_c - 1.0
        if abs(ret) < MIN_SPLIT_MOVE:
            continue

        price_ratio = prev_c / cur_c            # 2.0 for a 2-for-1 split
        factor = _nearest_known_ratio(price_ratio)
        if factor is None:
            continue

        prev_v, cur_v = volume[i - 1], volume[i]
        if not (np.isfinite(prev_v) and np.isfinite(cur_v)) or prev_v <= 0:
            continue
        vol_ratio = cur_v / prev_v
        if abs(vol_ratio - factor) > VOL_RATIO_TOL * factor:
            continue                            # no volume corroboration

        lo = max(0, i - CONTEXT_WINDOW)
        hi = min(len(d), i + CONTEXT_WINDOW + 1)
        neighbours = np.concatenate([rets[lo:i], rets[i + 1:hi]])
        neighbour_max = (
            float(np.nanmax(np.abs(neighbours))) if np.isfinite(neighbours).any() else 0.0
        )
        if neighbour_max > CONTEXT_MAX_ABS_RET:
            continue                            # crash, not a corporate action

        out.append({
            "date": pd.Timestamp(d.loc[i, "date"]),
            "price_ratio": float(price_ratio),
            "volume_ratio": float(vol_ratio),
            "factor": float(factor),
            "neighbour_max_abs_ret": neighbour_max,
        })
    return out


def apply_split_adjustment(df: pd.DataFrame, splits: list[dict]) -> pd.DataFrame:
    """Rebuild a continuous series by dividing pre-split prices by the factor."""
    if not splits:
        return df.copy()

    d = df.sort_values("date").reset_index(drop=True).copy()
    dates = pd.to_datetime(d["date"])
    cum = np.ones(len(d), dtype=float)
    for s in splits:
        before = (dates < pd.Timestamp(s["date"])).to_numpy()
        cum[before] *= s["factor"]

    for col in ("open", "high", "low", "close"):
        if col in d.columns:
            d[col] = d[col].to_numpy(dtype=float) / cum
    if "volume" in d.columns:
        d["volume"] = d["volume"].to_numpy(dtype=float) * cum
    return d


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def spells_for(membership: pd.DataFrame, tickers: set[str]) -> dict[str, tuple]:
    """First and last membership date per ticker, padded for feature warmup."""
    sub = membership[membership["ticker"].isin(tickers)]
    grouped = sub.groupby("ticker")["date"].agg(["min", "max"])
    return {
        t: (row["min"] - pd.Timedelta(days=400), row["max"] + pd.Timedelta(days=5))
        for t, row in grouped.iterrows()
    }


def recover(tickers: list[str], refresh: bool = False) -> pd.DataFrame:
    """Rebuild daily bars for `tickers` from the 1-minute store."""
    import duckdb

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    membership = pd.read_parquet(MEMBERSHIP)
    membership["date"] = pd.to_datetime(membership["date"])
    spells = spells_for(membership, set(tickers))

    todo = [
        t for t in tickers
        if refresh or not (CACHE_DIR / f"{t.replace('/', '_')}.parquet").exists()
    ]
    log(f"{len(tickers)} requested; {len(tickers) - len(todo)} cached; {len(todo)} to build")
    if not todo:
        return pd.DataFrame()

    # Group work by month so each monthly file is scanned once.
    per_month: dict[Path, list[str]] = {}
    for t in todo:
        if t not in spells:
            continue
        start, end = spells[t]
        for f in month_files(start, end):
            per_month.setdefault(f, []).append(t)

    con = duckdb.connect()
    split_rows: list[dict] = []
    frames: dict[str, list[pd.DataFrame]] = {t: [] for t in todo}
    for i, (path, tick_list) in enumerate(sorted(per_month.items()), start=1):
        try:
            df = session_daily_bars(con, path, sorted(set(tick_list)))
        except Exception as exc:                       # keep going on one bad file
            log(f"  [warn] {path.name}: {exc}")
            continue
        for t, grp in df.groupby("ticker"):
            frames[t].append(grp)
        if i % 25 == 0:
            log(f"  scanned {i}/{len(per_month)} monthly files")

    rows = []
    for t, parts in frames.items():
        if not parts:
            rows.append({"ticker": t, "n_days": 0, "n_splits": 0, "status": "absent_from_store"})
            continue
        daily = pd.concat(parts, ignore_index=True)
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.drop_duplicates(["ticker", "date"]).sort_values("date")

        splits = detect_split_factors(daily)
        adjusted = apply_split_adjustment(daily, splits)
        adjusted["adj_close"] = adjusted["close"]      # no dividend data in this source
        adjusted["source"] = "finnhub_1m_session"
        adjusted = adjusted[
            ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "source"]
        ]
        adjusted.to_parquet(
            CACHE_DIR / f"{t.replace('/', '_')}.parquet", index=False, compression="zstd"
        )
        rows.append({
            "ticker": t,
            "n_days": int(len(adjusted)),
            "n_splits": len(splits),
            "split_dates": ";".join(str(pd.Timestamp(s["date"]).date()) for s in splits),
            "split_factors": ";".join(f"{s['factor']:g}" for s in splits),
            "first_date": str(adjusted["date"].min().date()),
            "last_date": str(adjusted["date"].max().date()),
            "status": "recovered",
        })
        for s in splits:
            split_rows.append({
                "ticker": t,
                "date": str(pd.Timestamp(s["date"]).date()),
                "factor": s["factor"],
                "price_ratio": round(s["price_ratio"], 4),
                "volume_ratio": round(s["volume_ratio"], 4),
                "neighbour_max_abs_ret": round(s["neighbour_max_abs_ret"], 4),
            })

    report = pd.DataFrame(rows).sort_values("ticker")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if REPORT.exists():
        # Merge, so rebuilding a subset does not discard the other tickers' rows.
        previous = pd.read_csv(REPORT)
        keep = previous[~previous["ticker"].isin(report["ticker"])]
        report = pd.concat([keep, report], ignore_index=True).sort_values("ticker")
    report.to_csv(REPORT, index=False)
    # Every adjustment is auditable: the paper reports how many bars were
    # rescaled and on what evidence, rather than silently altering prices.
    if split_rows:
        pd.DataFrame(split_rows).sort_values(["ticker", "date"]).to_csv(SPLITS_REPORT, index=False)
        log(f"split detail -> {SPLITS_REPORT.relative_to(ROOT)}")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N missing tickers (trial runs)")
    parser.add_argument("--tickers", default=None,
                        help="Comma-separated subset to (re)build, instead of the "
                             "unrecoverable list; use with --refresh to redo them")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)

    if args.tickers:
        missing = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        missing = sorted(pd.read_csv(UNRECOVERABLE)["symbol"].astype(str))
    if args.limit:
        missing = missing[: args.limit]

    report = recover(missing, refresh=args.refresh)
    if report.empty:
        log("nothing to do")
        return

    recovered = report[report["status"] == "recovered"]
    log(f"\nrecovered {len(recovered)}/{len(report)} tickers; "
        f"{int(recovered['n_days'].sum()):,} daily bars; "
        f"{int(recovered['n_splits'].sum())} splits adjusted")
    log(f"report -> {REPORT.relative_to(ROOT)}")
    log(json.dumps(report["status"].value_counts().to_dict(), indent=2))


if __name__ == "__main__":
    main()
