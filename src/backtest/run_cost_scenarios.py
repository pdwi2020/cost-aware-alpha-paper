"""Evaluate the deployed composite under calibrated execution costs.

Why this exists
---------------
Reviewer 1 (point 7): a fixed 3 bps half-spread, a single impact coefficient of
0.10, and zero borrow cost are applied across firms, dates, and market
conditions, so "an uncalibrated cost model cannot support the strength of the
stated verdict".

`src.backtest.cost_calibration` supplies the estimates; this script is the
runner that puts the deployed book through them and reports what changes.

The shape of the argument
-------------------------
The paper's verdict is NEGATIVE: a statistically real signal is not tradable.
A negative verdict is only strong if it survives *optimistic* costs, so the
manuscript's original parameters are kept as the optimistic bound and the
calibrated and conservative scenarios are reported beside them. If the book
fails to clear the economic threshold even under the optimistic bound, no
calibration can rescue it, and the calibrated numbers then say how much room
there was to begin with. The risk direction that would matter is the reverse
one (a book that clears optimistically but not when calibrated), which is why
every scenario is reported rather than only the primary one.

Outputs
-------
    data/processed/cost_scenarios.parquet    one row per scenario / impact point
    results/staging/cost_scenarios.json      headline numbers + diagnostics

Run
---
    python3 -u src/backtest/run_cost_scenarios.py
    python3 -u src/backtest/run_cost_scenarios.py --track track_a --B 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import books
from src.backtest.cost_calibration import (
    BORROW_GC_BPS,
    BORROW_HTB_BPS,
    BORROW_HTB_STRESS_BPS,
    IMPACT_GRID,
    borrow_bps_matrix,
    half_spread_bps_matrix,
)
from src.backtest.oos_inference import SR_STAR, window_inference
from src.backtest.portfolio import PortfolioSimulator
from src.fdr.run_pbo_selection_path import selected_features, signal_weights

FEAT_PATH = ROOT / "data" / "processed" / "features_all.parquet"
OHLCV_PATH = ROOT / "data" / "processed" / "daily_ohlcv_v3.parquet"
FDR_PATH = ROOT / "data" / "processed" / "fdr_results.parquet"
SHAP_PATH = ROOT / "data" / "processed" / "shap_summary.parquet"
CFG_PATH = ROOT / "configs" / "backtest.yaml"

OUT_PARQUET = ROOT / "data" / "processed" / "cost_scenarios.parquet"
OUT_JSON = ROOT / "results" / "staging" / "cost_scenarios.json"

IS_START = "2013-01-01"
IS_END = "2021-12-31"
DEPLOYED_RULE, DEPLOYED_WEIGHTING, DEPLOYED_REBAL = "bh", "shap", 5
TRADING_DAYS = 252


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def build_scenarios(
    panel_long: pd.DataFrame,
    adv: pd.DataFrame,
) -> tuple[list[dict], dict]:
    """The three reported scenarios, with their cost matrices materialised."""
    log("  building spread matrices ...")
    ar_spread, ar_diag = half_spread_bps_matrix(panel_long, estimator="abdi_ranaldo")
    cs_spread, cs_diag = half_spread_bps_matrix(panel_long, estimator="corwin_schultz")

    log("  building borrow matrices ...")
    borrow_gc, borrow_diag = borrow_bps_matrix(
        panel_long, adv, gc_bps=BORROW_GC_BPS, htb_bps=BORROW_HTB_BPS
    )
    borrow_stress, borrow_stress_diag = borrow_bps_matrix(
        panel_long, adv, gc_bps=BORROW_GC_BPS, htb_bps=BORROW_HTB_STRESS_BPS
    )

    scenarios = [
        {
            "scenario": "optimistic_fixed",
            "spread_source": "fixed 3 bps half-spread",
            "spread_matrix": None,
            "spread_bps": 3.0,
            "impact_coeff": 0.10,
            "borrow_matrix": None,
            "borrow_desc": "zero (config borrow_cost_easy_bps=0)",
            "note": "the original manuscript's parameters; the optimistic bound",
        },
        {
            "scenario": "calibrated",
            "spread_source": "Abdi-Ranaldo stock-day half-spread (21d, lagged 2)",
            "spread_matrix": ar_spread,
            "spread_bps": 3.0,          # fallback for cells the estimator misses
            "impact_coeff": 0.142,
            "borrow_matrix": borrow_gc,
            "borrow_desc": f"{BORROW_GC_BPS:.0f} GC / {BORROW_HTB_BPS:.0f} HTB",
            "note": "primary calibrated scenario",
        },
        {
            "scenario": "conservative",
            "spread_source": "Corwin-Schultz stock-day half-spread (21d median, lagged 1)",
            "spread_matrix": cs_spread,
            "spread_bps": 3.0,
            "impact_coeff": 0.50,
            "borrow_matrix": borrow_stress,
            "borrow_desc": f"{BORROW_GC_BPS:.0f} GC / {BORROW_HTB_STRESS_BPS:.0f} HTB stress",
            "note": "upper cost bound; square-root-law prefactor",
        },
    ]
    diagnostics = {
        "abdi_ranaldo": ar_diag,
        "corwin_schultz": cs_diag,
        "borrow_gc": borrow_diag,
        "borrow_stress": borrow_stress_diag,
    }
    return scenarios, diagnostics


# ---------------------------------------------------------------------------
# One book, one cost setting
# ---------------------------------------------------------------------------

def evaluate(
    feat_df: pd.DataFrame,
    weights: pd.Series,
    panel: tuple,
    scenario: dict,
    impact_coeff: float,
    B: int,
    seed: int,
) -> dict:
    """Build the deployed book under one cost setting and summarise it."""
    sim = PortfolioSimulator(
        config_path=str(CFG_PATH),
        spread_bps=scenario["spread_bps"],
        impact_coeff=impact_coeff,
    )
    book = books.composite_book(
        feat_df, weights, panel, sim,
        rebal_freq=DEPLOYED_REBAL, start=IS_START, end=IS_END,
        spread_bps_matrix=scenario["spread_matrix"],
        borrow_bps_matrix=scenario["borrow_matrix"],
    )

    net = book["net_pnl"].to_numpy(dtype=float)
    gross = book["gross_pnl"].to_numpy(dtype=float)
    inf = window_inference(net, B=B, seed=seed)

    def ann_sharpe(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        sd = x.std(ddof=1)
        return float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-15 else float("nan")

    return {
        "scenario": scenario["scenario"],
        "impact_coeff": impact_coeff,
        "spread_source": scenario["spread_source"],
        "borrow": scenario["borrow_desc"],
        "T": int(inf["T"]),
        "gross_sharpe": ann_sharpe(gross),
        "net_sharpe": float(inf["sharpe_ann"]),
        "net_ci95_lo": float(inf["ci_95"]["lo"]),
        "net_ci95_hi": float(inf["ci_95"]["hi"]),
        "boot_p": float(inf["boot_p"]),
        "nw_t": float(inf["newey_west"]["t"]),
        "classification": inf["primary_classification"],
        "mean_gross_bps": float(np.nanmean(gross) * 10_000),
        "mean_cost_bps": float(np.nanmean(book["total_cost"].to_numpy(float)) * 10_000),
        "mean_net_bps": float(np.nanmean(net) * 10_000),
        "spread_bps_of_cost": float(np.nanmean(book["spread_cost"].to_numpy(float)) * 10_000),
        "impact_bps_of_cost": float(np.nanmean(book["impact_cost"].to_numpy(float)) * 10_000),
        "borrow_bps_of_cost": float(np.nanmean(book["borrow_cost"].to_numpy(float)) * 10_000),
        "mean_turnover": float(np.nanmean(book["turnover"].to_numpy(float))),
        "clears_sr_star": bool(inf["sharpe_ann"] >= SR_STAR),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="track_a")
    ap.add_argument("--B", type=int, default=10_000,
                    help="bootstrap replicates for the CI")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    log("=== Cost scenarios for the deployed composite ===\n")
    t0 = time.time()

    from src.data.lean_load import load_features_lean

    feat_df = load_features_lean(FEAT_PATH)
    feat_df.index = feat_df.index.set_levels(
        [feat_df.index.levels[0], pd.to_datetime(feat_df.index.levels[1])]
    )

    fdr = pd.read_parquet(FDR_PATH)
    track_fdr = fdr[fdr["track"] == args.track].copy()
    if track_fdr.empty:
        raise SystemExit(f"No FDR rows for track {args.track!r}")

    shap_df = pd.read_parquet(SHAP_PATH)
    shap_summary = (
        shap_df[shap_df["track"] == args.track]
        .groupby("feature")["mean_abs_shap"].mean()
    )

    feats = selected_features(DEPLOYED_RULE, track_fdr)
    if not feats:
        raise SystemExit(
            f"The deployed rule {DEPLOYED_RULE!r} selects no features on "
            f"{args.track!r}: there is no book to cost."
        )
    weights = signal_weights(feats, DEPLOYED_WEIGHTING, track_fdr, shap_summary)
    log(f"  deployed book: {DEPLOYED_RULE}/{DEPLOYED_WEIGHTING}/"
        f"rebal={DEPLOYED_REBAL}, {len(feats)} features")
    log(f"  features: {sorted(feats)}\n")

    ohlcv = pd.read_parquet(OHLCV_PATH)
    tickers = sorted(feat_df.index.get_level_values("ticker").unique())
    returns, sigma, adv = books.load_market_panel(ohlcv, tickers, IS_START, IS_END)
    panel = (returns, sigma, adv)

    scenarios, diagnostics = build_scenarios(ohlcv, adv)
    for name, diag in diagnostics.items():
        log(f"    {name}: {json.dumps(diag, default=float)}")

    rows = []
    log("\n  --- named scenarios ---")
    for sc in scenarios:
        row = evaluate(feat_df, weights, panel, sc, sc["impact_coeff"],
                       args.B, args.seed)
        row["is_grid_point"] = False
        rows.append(row)
        log(f"    {row['scenario']:<18s} eta={row['impact_coeff']:<5.3f} "
            f"gross SR {row['gross_sharpe']:+.3f} -> net SR {row['net_sharpe']:+.3f} "
            f"[{row['net_ci95_lo']:+.3f}, {row['net_ci95_hi']:+.3f}]  "
            f"{row['classification']}")

    # Impact sweep under the calibrated spread and borrow model: isolates the
    # one parameter with no daily-data estimator of its own.
    calibrated = next(s for s in scenarios if s["scenario"] == "calibrated")
    log("\n  --- impact grid (calibrated spreads and borrow) ---")
    for eta, anchor in IMPACT_GRID.items():
        row = evaluate(feat_df, weights, panel, calibrated, eta, args.B, args.seed)
        row["scenario"] = "impact_grid"
        row["is_grid_point"] = True
        row["impact_anchor"] = anchor
        rows.append(row)
        log(f"    eta={eta:<5.3f} net SR {row['net_sharpe']:+.3f} "
            f"[{row['net_ci95_lo']:+.3f}, {row['net_ci95_hi']:+.3f}]  "
            f"{row['classification']:<22s} ({anchor})")

    out = pd.DataFrame(rows)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)

    optimistic = out[out["scenario"] == "optimistic_fixed"].iloc[0]
    headline = {
        "track": args.track,
        "deployed": {
            "rule": DEPLOYED_RULE, "weighting": DEPLOYED_WEIGHTING,
            "rebal": DEPLOYED_REBAL, "n_features": len(feats),
            "features": sorted(feats),
        },
        "window": {"start": IS_START, "end": IS_END},
        "sr_star": SR_STAR,
        "bootstrap_B": args.B,
        "seed": args.seed,
        "verdict_survives_optimistic_bound": not bool(optimistic["clears_sr_star"]),
        "any_scenario_clears_sr_star": bool(out["clears_sr_star"].any()),
        "scenarios": rows,
        "cost_diagnostics": diagnostics,
    }
    OUT_JSON.write_text(json.dumps(headline, indent=2, default=float))

    log(f"\nSaved → {OUT_PARQUET}")
    log(f"Saved → {OUT_JSON}")
    log(f"\n  Verdict holds at the optimistic bound: "
        f"{headline['verdict_survives_optimistic_bound']}")
    log(f"  Any scenario clears SR* = {SR_STAR}: "
        f"{headline['any_scenario_clears_sr_star']}")
    log(f"\nTotal elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
