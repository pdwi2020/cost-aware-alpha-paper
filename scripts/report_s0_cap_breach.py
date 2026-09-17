"""Report position-cap breaches caused by the deployed Screen 0 rule.

The deployed rule removes ineligible names after position capping and then
renormalises the survivors. This script measures how often that old rule pushes
the real in-sample Track A composite book above its five-percent name cap.
"""

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.portfolio import PortfolioSimulator, apply_s0_eligible


SIGNAL_PATH = ROOT / "data" / "processed" / "signals_track_a.parquet"
FEATURES_PATH = ROOT / "data" / "processed" / "features_all.parquet"
CONFIG_PATH = ROOT / "configs" / "backtest.yaml"
OUTPUT_PATH = ROOT / "results" / "staging" / "s0_cap_breach_report.json"
START_DATE = pd.Timestamp("2013-01-01")
END_DATE = pd.Timestamp("2021-12-31")
CAP = 0.05


def log(msg):
    print(msg, flush=True)


def main() -> None:
    signal_df = pd.read_parquet(SIGNAL_PATH)
    feat_df = pd.read_parquet(FEATURES_PATH)
    sim = PortfolioSimulator(config_path=str(CONFIG_PATH))

    positions = sim.signal_to_positions(signal_df, lag=1)
    positions_old = apply_s0_eligible(positions, feat_df)
    in_sample = positions_old.loc[
        (positions_old.index >= START_DATE) & (positions_old.index <= END_DATE)
    ]

    max_weight = float(in_sample.abs().max().max())
    pct_days_breach = float(
        (in_sample.abs().max(axis=1) > CAP + 1e-9).mean() * 100
    )
    report = {
        "max_weight": max_weight,
        "pct_days_breach": pct_days_breach,
        "n_days": int(len(in_sample)),
        "cap": CAP,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n")

    log(f"max_weight: {max_weight}")
    log(f"pct_days_breach: {pct_days_breach}")
    log(f"n_days: {len(in_sample)}")
    log(f"cap: {CAP}")


if __name__ == "__main__":
    main()
