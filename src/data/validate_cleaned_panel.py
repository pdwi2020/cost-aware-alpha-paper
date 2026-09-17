"""Validate the cleaned price panel against an independently re-pulled source.

Why this exists
---------------
Section 4.4 of the manuscript claims the cleaned panel was checked against a
re-pulled, split- and dividend-adjusted series, and quotes per-name return
correlations, a count of confirmed >50% artefacts, and a count of unmatched
names. No script in the repository produced those figures. The claim was
therefore unverifiable from the artefacts we ship, which is the same defect the
rest of the pipeline was audited to remove: a number with no traceable source
is a number nobody can check, including us.

This regenerates the validation from scratch so the figures in the manuscript
are reproducible.

Method
------
For every ticker in the cleaned panel that yfinance still serves, pull the
auto-adjusted daily close over the in-sample window, align on common dates, and
compare daily returns against the cleaned series. Tickers yfinance no longer
serves are reported separately: they are delisted, and Screen 0's price floor
excludes them anyway.

The >50% single-day moves flagged by the sanitiser are cross-checked: a flagged
move is "confirmed as an artefact" when the adjusted source does not show a
comparable move on the same day, which is the signature of an unadjusted
corporate action in the raw feed.

Outputs
-------
    results/universe_v3/panel_validation.csv       per ticker
    results/universe_v3/panel_validation.json      headline figures
Manifest keys under ``validation.panel.*``.

Run (network; ~10-20 min for the full panel):
    python3 -u src/data/validate_cleaned_panel.py
    python3 -u src/data/validate_cleaned_panel.py --limit 50   # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.manifest import record  # noqa: E402

OHLCV = ROOT / "data" / "processed" / "daily_ohlcv_v3.parquet"
OUT_DIR = ROOT / "results" / "universe_v3"
OUT_CSV = OUT_DIR / "panel_validation.csv"
OUT_JSON = OUT_DIR / "panel_validation.json"

IS_START, IS_END = "2013-01-01", "2021-12-31"
SANITIZE_CAP = 0.50
BATCH = 40


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="validate only the first N tickers")
    args = ap.parse_args()

    import yfinance as yf

    panel = pd.read_parquet(OHLCV, columns=["ticker", "date", "close"])
    panel = panel[(panel.date >= IS_START) & (panel.date <= IS_END)]
    wide = panel.pivot(index="date", columns="ticker", values="close").sort_index()
    tickers = sorted(wide.columns)
    if args.limit:
        tickers = tickers[: args.limit]
    log(f"panel: {len(wide):,} dates x {len(tickers)} tickers over {IS_START}..{IS_END}")

    ours = wide[tickers].pct_change()

    rows, unmatched = [], []
    t0 = time.time()
    for i in range(0, len(tickers), BATCH):
        chunk = tickers[i : i + BATCH]
        try:
            px = yf.download(chunk, start=IS_START, end=IS_END, progress=False,
                             auto_adjust=True, threads=True)["Close"]
        except Exception as exc:                      # network or symbol errors
            log(f"  [warn] batch {i//BATCH}: {exc}")
            unmatched.extend(chunk)
            continue
        if isinstance(px, pd.Series):
            px = px.to_frame(chunk[0])
        theirs = px.pct_change()
        for t in chunk:
            if t not in theirs.columns or theirs[t].notna().sum() < 100:
                unmatched.append(t)
                continue
            a, b = ours[t].align(theirs[t], join="inner")
            both = pd.concat([a, b], axis=1).dropna()
            if len(both) < 100:
                unmatched.append(t)
                continue
            corr = float(both.iloc[:, 0].corr(both.iloc[:, 1]))
            flagged = both[both.iloc[:, 0].abs() > SANITIZE_CAP]
            confirmed = int((flagged.iloc[:, 1].abs() < SANITIZE_CAP / 2).sum()) if len(flagged) else 0
            rows.append({"ticker": t, "n_overlap": len(both), "return_correlation": corr,
                         "n_flagged_gt50pct": len(flagged), "n_confirmed_artefact": confirmed})
        log(f"  {min(i+BATCH, len(tickers)):>4}/{len(tickers)} tickers  "
            f"({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows).sort_values("return_correlation")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    c = df.return_correlation
    summary = {
        "n_validated": int(len(df)),
        "n_unmatched": int(len(unmatched)),
        "corr_min": float(c.min()), "corr_max": float(c.max()),
        "corr_p05": float(c.quantile(0.05)), "corr_p95": float(c.quantile(0.95)),
        "corr_median": float(c.median()),
        "n_flagged_total": int(df.n_flagged_gt50pct.sum()),
        "n_confirmed_artefact": int(df.n_confirmed_artefact.sum()),
        "unmatched_tickers": sorted(unmatched),
        "window": f"{IS_START}..{IS_END}",
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    for k in ("n_validated", "n_unmatched", "corr_p05", "corr_p95", "corr_median",
              "n_flagged_total", "n_confirmed_artefact"):
        record(f"validation.panel.{k}", summary[k], stage="validation")

    log("\n=== panel validation ===")
    log(f"  validated {summary['n_validated']} tickers, {summary['n_unmatched']} unmatched")
    log(f"  return correlation: median {summary['corr_median']:.3f}, "
        f"5th-95th {summary['corr_p05']:.2f}-{summary['corr_p95']:.2f}, "
        f"min {summary['corr_min']:.2f}")
    log(f"  flagged >50% moves: {summary['n_flagged_total']}, "
        f"confirmed as artefacts: {summary['n_confirmed_artefact']}")
    log(f"\nSaved -> {OUT_CSV}\nSaved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
