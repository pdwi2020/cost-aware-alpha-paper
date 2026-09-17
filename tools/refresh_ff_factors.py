"""tools/refresh_ff_factors.py — refresh the Fama-French daily factor mirror.

Why this exists
---------------
``build_features.compute_track_a`` reindexes the factor frame onto the price
calendar with ``method="ffill"``. Inside the historical sample that is
harmless, because the factors are complete. Past the mirror's last date it is
not: every factor is held at its final observed value, so the rolling OLS
residualisation runs against a constant factor vector and Track A's
"idiosyncratic" target degenerates toward the raw return.

The mirror ended 2026-02-27 while the forward holdout window runs into 2026,
which would have silently corrupted the one window the protocol allows us to
score exactly once. This tool refreshes the mirror from the source and reports
the new end date so the window can be set to match.

Source: Kenneth R. French Data Library (public, free).
  F-F_Research_Data_5_Factors_2x3_daily_CSV.zip
  F-F_Momentum_Factor_daily_CSV.zip

Usage:
    python3 -u tools/refresh_ff_factors.py [--zip-dir DIR] [--dry-run]

Writes data/fama_french_factors/famafrench_{ff5,mom}_daily.parquet, keeping a
.bak of each, and refuses to write a file that would shorten the series.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "fama_french_factors"

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
SOURCES = {
    "ff5": ("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
            ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
            "famafrench_ff5_daily.parquet"),
    "mom": ("F-F_Momentum_Factor_daily_CSV.zip",
            ["Mom"],
            "famafrench_mom_daily.parquet"),
}


def _read_zip(zip_bytes: bytes, columns: list[str]) -> pd.DataFrame:
    """Parse a French daily CSV: preamble, then YYYYMMDD rows, then copyright."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        text = z.read(name).decode("latin-1")

    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(columns) + 1:
            continue
        stamp = parts[0]
        # Daily rows are 8-digit dates; monthly/annual blocks are 6-digit.
        if not (stamp.isdigit() and len(stamp) == 8):
            continue
        try:
            values = [float(p) for p in parts[1:]]
        except ValueError:
            continue
        rows.append((pd.Timestamp(stamp), *values))

    if not rows:
        raise ValueError(f"no daily rows parsed from {name}")
    df = pd.DataFrame(rows, columns=["date", *columns]).set_index("date").sort_index()
    # French uses -99.99 / -999 as missing markers.
    return df.mask(df <= -99.0)


def _fetch(url: str) -> bytes:
    import urllib.request

    with urllib.request.urlopen(url, timeout=180) as resp:
        return resp.read()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-dir", type=Path, default=None,
                    help="read the two zips from here instead of downloading")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    ends = {}
    for key, (zip_name, columns, out_name) in SOURCES.items():
        if args.zip_dir:
            blob = (args.zip_dir / zip_name).read_bytes()
            origin = str(args.zip_dir / zip_name)
        else:
            origin = f"{BASE}/{zip_name}"
            blob = _fetch(origin)

        fresh = _read_zip(blob, columns)
        out_path = OUT_DIR / out_name

        if out_path.exists():
            current = pd.read_parquet(out_path)
            if fresh.index.max() < current.index.max():
                raise SystemExit(
                    f"REFUSED: fresh {key} ends {fresh.index.max().date()} but the "
                    f"existing mirror ends {current.index.max().date()}; refusing to "
                    f"shorten the series"
                )
            # The overlap must agree, or the vendor revised history and every
            # downstream number would move for a reason unrelated to this refresh.
            overlap = current.index.intersection(fresh.index)
            delta = (current.loc[overlap, columns] - fresh.loc[overlap, columns]).abs().max().max()
            print(f"  {key}: {len(overlap)} overlapping days, max |diff| = {delta:.4f}")
            if delta > 0.005:
                print(f"  [WARN] {key}: vendor history revised by more than 0.005; "
                      f"downstream results will move for that reason too")
            print(f"  {key}: {current.index.max().date()} -> {fresh.index.max().date()} "
                  f"(+{(fresh.index.max() - current.index.max()).days} days)")
        else:
            print(f"  {key}: new mirror ending {fresh.index.max().date()}")

        ends[key] = fresh.index.max()
        if not args.dry_run:
            if out_path.exists():
                shutil.copy2(out_path, out_path.with_suffix(".parquet.bak"))
            fresh.to_parquet(out_path)

    common_end = min(ends.values())
    print(f"\nFactor data now ends {common_end.date()} (both series).")
    print("Any window scored past that date residualises against forward-filled "
          "constants; set the forward window to end on or before it.")
    if args.dry_run:
        print("(dry run: nothing written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
