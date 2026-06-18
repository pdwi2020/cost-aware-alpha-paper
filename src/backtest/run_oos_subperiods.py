"""B1 — Year-by-year OOS sub-period breakdown for Track A and Track B.

Addresses referee point M4: a single 753-day window gives limited evidence of
temporal stability. This script splits the 2022-2024 holdout into annual
sub-periods and computes net Sharpe, annualised net return, max drawdown, and
cost drag for each year and for the full window.

Inputs:
    data/processed/holdout_pnl_track_a.parquet
    data/processed/holdout_pnl_track_b.parquet
    (columns: gross_pnl, spread_cost, impact_cost, borrow_cost, total_cost,
              net_pnl, turnover; date-indexed)

Output:
    data/processed/oos_subperiods.parquet
    (cols: track, period, n_days, gross_sr, net_sr, ann_net_pct,
           max_dd_pct, calmar, cost_drag_bps)

Run:
    python3 -u src/backtest/run_oos_subperiods.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "processed"
OUT  = DATA / "oos_subperiods.parquet"
ANN  = 252.0


def sharpe_ann(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 1e-12 else np.nan


def max_drawdown(r: np.ndarray) -> float:
    cum = np.cumsum(r)
    return float((cum - np.maximum.accumulate(cum)).min())


def calmar(r: np.ndarray) -> float:
    ann_ret = r.mean() * ANN
    mdd = abs(max_drawdown(r))
    return ann_ret / mdd if mdd > 1e-12 else np.nan


def compute_period(pnl: pd.DataFrame, label: str, track: str) -> dict:
    g = pnl["gross_pnl"].to_numpy(float)
    n = pnl["net_pnl"].to_numpy(float)
    if "total_cost" in pnl.columns:
        tc = pnl["total_cost"].to_numpy(float)
    else:
        tc = (pnl["spread_cost"] + pnl["impact_cost"] + pnl["borrow_cost"]).to_numpy(float)
    return {
        "track":          track,
        "period":         label,
        "n_days":         len(n),
        "gross_sr":       round(sharpe_ann(g), 3),
        "net_sr":         round(sharpe_ann(n), 3),
        "ann_net_pct":    round(float(n.mean() * ANN * 100), 2),
        "max_dd_pct":     round(max_drawdown(n) * 100, 2),
        "calmar":         round(calmar(n), 3),
        "cost_drag_bps":  round(float(tc.mean() * ANN * 10_000), 1),
    }


def main():
    rows = []
    for track in ["track_a", "track_b"]:
        pnl = pd.read_parquet(DATA / f"holdout_pnl_{track}.parquet")
        pnl.index = pd.to_datetime(pnl.index)
        pnl["year"] = pnl.index.year

        for yr in [2022, 2023, 2024]:
            sub = pnl[pnl["year"] == yr]
            rows.append(compute_period(sub, str(yr), track))

        rows.append(compute_period(pnl, "OOS (2022-24)", track))

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)

    pd.set_option("display.width", 160, "display.max_columns", 20)
    print("\n=== OOS Sub-Period Performance ===\n")
    for track in ["track_b", "track_a"]:
        sub = df[df["track"] == track]
        print(f"  {track.upper()}")
        print(sub[["period","n_days","gross_sr","net_sr","ann_net_pct",
                   "max_dd_pct","cost_drag_bps"]].to_string(index=False))
        print()

    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
