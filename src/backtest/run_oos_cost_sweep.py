"""P1.3 — OOS cost-robustness sweep for Track A and Track B.

Since the Almgren-Chriss cost components are linear in their parameters:
  spread_cost(h) = (h/h_base) × spread_cost_base      (linear in h)
  impact_cost(η) = (η/η_base) × impact_cost_base      (linear in η)

we can rescale the already-computed daily cost components from
holdout_pnl_track_*.parquet instead of re-running the full simulation.

This gives an exact OOS net Sharpe for every (spread, η) combination in the
sensitivity grid without any additional model evaluations.

Grid:
    spread ∈ {1, 3, 5, 10} bps   (base = 3)
    η      ∈ {0.05, 0.10, 0.20}  (base = 0.10)

Outputs:
    data/processed/oos_cost_sweep.parquet — net Sharpe for each cell

Run:
    python3 -u src/backtest/run_oos_cost_sweep.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

DATA     = ROOT / "data" / "processed"
OUT      = DATA / "oos_cost_sweep.parquet"

SPREADS  = [1, 3, 5, 10]   # bps
ETAS     = [0.05, 0.10, 0.20]
BASE_H   = 3.0
BASE_ETA = 0.10
ANN      = 252.0


def sharpe_ann(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 1e-12 else np.nan


def sweep_track(track: str):
    pnl = pd.read_parquet(DATA / f"holdout_pnl_{track}.parquet")
    gross  = pnl["gross_pnl"].to_numpy(float)
    sc_base = pnl["spread_cost"].to_numpy(float)
    ic_base = pnl["impact_cost"].to_numpy(float)
    bc      = pnl["borrow_cost"].to_numpy(float)   # independent of h and η

    rows = []
    for h in SPREADS:
        for eta in ETAS:
            sc = sc_base * (h / BASE_H)
            ic = ic_base * (eta / BASE_ETA)
            net = gross - sc - ic - bc
            sr  = sharpe_ann(net)
            rows.append({
                "track":       track,
                "spread_bps":  h,
                "eta":         eta,
                "net_sharpe":  sr,
                "gross_sharpe": sharpe_ann(gross),
                "cost_drag_bps": float((sc + ic + bc).mean() * ANN * 10_000),
            })
    return pd.DataFrame(rows)


def main():
    parts = []
    for track in ["track_a", "track_b"]:
        df = sweep_track(track)
        parts.append(df)

        print(f"\n=== {track.upper()} OOS net Sharpe (spread × η) ===")
        pivot = df.pivot(index="spread_bps", columns="eta", values="net_sharpe").round(3)
        pivot.index.name   = "spread_bps \\ η →"
        pivot.columns.name = None
        print(pivot.to_string())

    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(OUT, index=False)
    print(f"\nSaved → {OUT}")

    # --- Print break-even boundary for Track B ---
    tb = out[out.track == "track_b"].copy()
    print("\nTrack B: cells with net Sharpe > 0 (profitable at OOS costs):")
    pos = tb[tb.net_sharpe > 0][["spread_bps", "eta", "net_sharpe"]].sort_values("net_sharpe", ascending=False)
    print(pos.to_string(index=False))


if __name__ == "__main__":
    main()
