"""R4 — Alpha Robustness Volume from sensitivity_3d.parquet.

Computes the fraction of cost-parameter configurations (spread × impact ×
rebal-freq) in which Track B's IS net Sharpe ratio is positive, providing a
single scalar "cost-survival rate" that summarises how robust the alpha is
to cost assumptions. No new backtests — reuses the pre-computed grid.

Two thresholds:
  vol_0   : fraction of configs with net SR > 0   (any positive performance)
  vol_025 : fraction of configs with net SR > 0.25 (economically meaningful)

Reports Track A and Track B for contrast, plus naive long-only and SPY baselines.

Outputs:
    data/processed/robustness_volume.parquet — per-track cost-survival statistics

Run:
    python3 -u src/backtest/run_robustness_volume.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

SENS_PATH = ROOT / "data" / "processed" / "sensitivity_3d.parquet"
OUT_PATH  = ROOT / "data" / "processed" / "robustness_volume.parquet"

THRESHOLDS = [0.0, 0.25, 0.50]


def log(msg): print(msg, flush=True)


def survival_stats(sr_series: pd.Series, label: str) -> dict:
    row = {"strategy": label, "n_configs": len(sr_series)}
    for t in THRESHOLDS:
        frac = (sr_series > t).mean()
        row[f"frac_gt_{str(t).replace('.','p')}"] = float(frac)
        row[f"n_gt_{str(t).replace('.','p')}"]    = int((sr_series > t).sum())
    row["median_net_sr"] = float(sr_series.median())
    row["min_net_sr"]    = float(sr_series.min())
    row["max_net_sr"]    = float(sr_series.max())
    return row


def main():
    log("=== Alpha Robustness Volume (R4) ===\n")
    t0 = time.time()

    sens = pd.read_parquet(SENS_PATH)
    log(f"  Grid shape: {sens.shape}")
    log(f"  Spread values:   {sorted(sens['spread_bps'].unique())}")
    log(f"  Impact values:   {sorted(sens['impact_coeff'].unique())}")
    log(f"  Rebal freq vals: {sorted(sens['rebal_freq'].unique())}")
    n_configs_per_track = len(sens[sens["track"] == "track_b"])
    log(f"  Configs per track: {n_configs_per_track}\n")

    records = []

    for track in ["track_a", "track_b"]:
        sub  = sens[sens["track"] == track]["net_pnl_sharpe"]
        stat = survival_stats(sub, track)
        records.append(stat)

        log(f"  {track.upper()}:")
        for t in THRESHOLDS:
            key = f"frac_gt_{str(t).replace('.','p')}"
            log(f"    Frac(net SR > {t:4.2f}) = {stat[key]*100:5.1f}%  "
                f"({stat[f'n_gt_{str(t).replace(chr(46),chr(112))}']} / {stat['n_configs']})")
        log(f"    Median net SR = {stat['median_net_sr']:+.3f}")
        log(f"    Range:         [{stat['min_net_sr']:+.3f}, {stat['max_net_sr']:+.3f}]\n")

    # Relative robustness: Track B / Track A ratio at each threshold
    b_row = next(r for r in records if r["strategy"] == "track_b")
    a_row = next(r for r in records if r["strategy"] == "track_a")
    log("  Relative robustness (Track B vs Track A):")
    for t in THRESHOLDS:
        key = f"frac_gt_{str(t).replace('.','p')}"
        b_f = b_row[key]
        a_f = a_row[key]
        ratio = b_f / a_f if a_f > 1e-6 else float("inf")
        log(f"    SR > {t:.2f}: B={b_f*100:.1f}%  A={a_f*100:.1f}%  ratio={ratio:.2f}x")

    out_df = pd.DataFrame(records)
    out_df.to_parquet(OUT_PATH, index=False)
    log(f"\nSaved → {OUT_PATH}")

    # Main paper number: Track B SR > 0 survival rate
    frac_b_pos = b_row["frac_gt_0p0"]
    log(f"\n★ KEY RESULT: Track B net SR > 0 in {frac_b_pos*100:.1f}% of cost configurations")
    log(f"★ KEY RESULT: Track B net SR > 0.25 in {b_row['frac_gt_0p25']*100:.1f}% of cost configurations")

    log(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
