"""PBO over the DEPLOYED SELECTION PATH (CSCV), not single features.

Why this exists
---------------
Reviewer 1 (point 5): "The PBO analysis is internally inconsistent and does not
audit the deployed strategy. Section 8 first defines a matrix with nine folds
and seven model variants, then switches to 16 blocks and single-feature
strategies. The reported PBO is described as belonging to the deployed Track A
strategy, although the deployed strategy is a BH-selected, SHAP-weighted
composite. The feature selection, model comparison, weighting, rebalancing,
cost-grid, and other tuning decisions are not represented in the reported CSCV
calculation."

That is correct on every count. This module makes the CSCV candidate set the
selection path that actually produced the deployed composite:

    selection rule   x   weighting scheme   x   rebalancing frequency
    (6 rules)            (3 schemes)            (3 frequencies)        = 54

The deployed strategy is one cell of that grid (BH selection, SHAP weighting,
weekly rebalancing). A conservative variant additionally treats the cost
assumption as a searched dimension (3 settings, 162 configurations), which is
arguably over-counting -- the cost model is an evaluation condition rather than
a strategy choice -- so both are reported.

Each configuration is a real composite book built through
`src.backtest.books.composite_book`, so it carries the same Screen 0, water-fill
cap and cost model as the deployed strategy.

Known limitation (stated in the paper, not hidden)
--------------------------------------------------
The selection rules and SHAP weights are estimated on the FULL in-sample window
before CSCV splits it, so every split's "in-sample" half contains information
from the other half. This biases CSCV towards UNDERSTATING overfitting: the
reported PBO is a lower bound. A nested version that re-estimates selection
inside each split is prohibitively expensive (12,870 splits x 54 configs).

Outputs
-------
    data/processed/pbo_selection_path.parquet          per-path lambda + Sharpes
    data/processed/pbo_selection_path_summary.parquet  per-variant summary
    results/staging/pbo_selection_path.json            headline numbers

Run
---
    python3 -u src/fdr/run_pbo_selection_path.py
    python3 -u src/fdr/run_pbo_selection_path.py --max-splits 500   # fast check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import books
from src.backtest.portfolio import PortfolioSimulator
import src.manifest as manifest
from src.fdr.run_pbo_deployed import cscv_pbo, degradation_slope

FEAT_PATH = ROOT / "data" / "processed" / "features_all.parquet"
OHLCV_PATH = ROOT / "data" / "processed" / "daily_ohlcv_v3.parquet"
FDR_PATH = ROOT / "data" / "processed" / "fdr_results.parquet"
SHAP_PATH = ROOT / "data" / "processed" / "shap_summary.parquet"
CFG_PATH = ROOT / "configs" / "backtest.yaml"

OUT_PATHS = ROOT / "data" / "processed" / "pbo_selection_path.parquet"
OUT_SUMMARY = ROOT / "data" / "processed" / "pbo_selection_path_summary.parquet"
OUT_JSON = ROOT / "results" / "staging" / "pbo_selection_path.json"

IS_START = "2013-01-01"
IS_END = "2021-12-31"
N_BLOCKS = 16

SELECTION_RULES = ("bh", "bhy", "top5", "top10", "top20", "all")
WEIGHTINGS = ("shap", "ic", "equal")
REBAL_FREQS = (1, 5, 21)
DEPLOYED = ("bh", "shap", 5)

# Conservative variant: treat the cost assumption as a searched dimension too.
COST_SETTINGS = (
    {"name": "base", "spread_bps": 3.0, "impact_coeff": 0.10},
    {"name": "low", "spread_bps": 1.0, "impact_coeff": 0.05},
    {"name": "high", "spread_bps": 5.0, "impact_coeff": 0.20},
)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# The selection path
# ---------------------------------------------------------------------------

def selected_features(rule: str, track_fdr: pd.DataFrame) -> list[str]:
    """Features admitted by one selection rule."""
    if rule == "bh":
        return track_fdr.loc[track_fdr["bh_rejected"], "feature"].tolist()
    if rule == "bhy":
        return track_fdr.loc[track_fdr["bhy_rejected"], "feature"].tolist()
    if rule == "all":
        return track_fdr["feature"].tolist()
    if rule.startswith("top"):
        k = int(rule[3:])
        ranked = track_fdr.reindex(
            track_fdr["ic_bar"].abs().sort_values(ascending=False).index
        )
        return ranked["feature"].head(k).tolist()
    raise ValueError(f"Unknown selection rule {rule!r}")


def signal_weights(
    features: list[str],
    weighting: str,
    track_fdr: pd.DataFrame,
    shap_summary: pd.Series,
) -> pd.Series:
    """Signed, L1-normalised weights for one weighting scheme.

    Every scheme carries the sign of the feature's mean IC, so the book is
    always pointed at its in-sample edge; the schemes differ only in magnitude.
    """
    if not features:
        return pd.Series(dtype=float)

    ic = track_fdr.set_index("feature")["ic_bar"]
    signs = np.sign(ic.reindex(features).fillna(1.0)).replace(0.0, 1.0)

    if weighting == "shap":
        mag = shap_summary.reindex(features)
        mag = mag.fillna(mag.mean() if np.isfinite(mag.mean()) else 1.0)
    elif weighting == "ic":
        mag = ic.reindex(features).abs()
    elif weighting == "equal":
        mag = pd.Series(1.0, index=features)
    else:
        raise ValueError(f"Unknown weighting {weighting!r}")

    w = signs * mag
    total = w.abs().sum()
    return w / total if total > 0 else w


def build_config_books(
    feat_df: pd.DataFrame,
    track_fdr: pd.DataFrame,
    shap_summary: pd.Series,
    panel: tuple,
    cfg: dict,
    cost_settings: tuple = (COST_SETTINGS[0],),
) -> tuple[pd.DataFrame, list[dict]]:
    """Daily net returns for every configuration on the selection path.

    Returns (matrix of shape T x M, config metadata list).
    """
    columns, meta = [], []
    for cost in cost_settings:
        sim = PortfolioSimulator(
            config_path=str(CFG_PATH),
            spread_bps=cost["spread_bps"],
            impact_coeff=cost["impact_coeff"],
        )
        for rule in SELECTION_RULES:
            feats = selected_features(rule, track_fdr)
            if not feats:
                log(f"    [skip] rule={rule} selects no features")
                continue
            for weighting in WEIGHTINGS:
                weights = signal_weights(feats, weighting, track_fdr, shap_summary)
                for rebal in REBAL_FREQS:
                    book = books.composite_book(
                        feat_df, weights, panel, sim,
                        rebal_freq=rebal, start=IS_START, end=IS_END,
                    )
                    columns.append(book["net_pnl"].rename(
                        f"{cost['name']}|{rule}|{weighting}|{rebal}"
                    ))
                    meta.append({
                        "cost": cost["name"], "rule": rule,
                        "weighting": weighting, "rebal": rebal,
                        "n_features": len(feats),
                        "is_deployed": (rule, weighting, rebal) == DEPLOYED
                                        and cost["name"] == "base",
                    })
                    log(f"    built {len(columns):3d}: cost={cost['name']:<4s} "
                        f"rule={rule:<5s} w={weighting:<5s} rebal={rebal:<2d} "
                        f"({len(feats)} features)")
    if not columns:
        raise RuntimeError("No configurations produced a book")
    return pd.concat(columns, axis=1), meta


# ---------------------------------------------------------------------------
# Deflated Sharpe with the ACTUAL trial count
# ---------------------------------------------------------------------------

def deflated_sharpe(returns: np.ndarray, n_trials: int) -> dict:
    """DSR for one daily return series against a search of `n_trials`.

    Uses the Bailey and Lopez de Prado (2014) expected maximum under the null,
    with the observed skew and kurtosis in the Sharpe standard error.
    """
    from scipy import stats

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    sd = r.std(ddof=1)
    if T < 30 or sd < 1e-15:
        return {"dsr": float("nan"), "sr_ann": float("nan"), "T": T}

    sr_daily = r.mean() / sd
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))

    gamma = 0.5772156649015329                       # Euler-Mascheroni
    n = max(int(n_trials), 2)
    e_max = (
        (1 - gamma) * stats.norm.ppf(1 - 1.0 / n)
        + gamma * stats.norm.ppf(1 - 1.0 / (n * np.e))
    )                                                # expected max Sharpe (per-period units of 1)
    se = np.sqrt((1 - skew * sr_daily + 0.25 * (kurt - 1) * sr_daily ** 2) / (T - 1))
    dsr = float(stats.norm.cdf((sr_daily - e_max * se) / se)) if se > 0 else float("nan")

    return {
        "dsr": dsr,
        "sr_ann": float(sr_daily * np.sqrt(252)),
        "expected_max_sr_ann": float(e_max * se * np.sqrt(252)),
        "T": int(T),
        "n_trials": n,
        "skew": skew,
        "kurtosis": kurt,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_variant(
    name: str,
    matrix: pd.DataFrame,
    meta: list[dict],
    max_splits: int | None,
) -> tuple[pd.DataFrame, dict]:
    """CSCV plus DSR for one candidate set."""
    R = matrix.to_numpy(dtype=float)
    total = comb(N_BLOCKS, N_BLOCKS // 2)
    log(f"\n  CSCV [{name}]: {R.shape[1]} configs, S={N_BLOCKS}, "
        f"{total if not max_splits else max_splits} paths")

    pbo, lam, is_sr, oos_sr = cscv_pbo(R, n_blocks=N_BLOCKS, max_splits=max_splits)
    slope = degradation_slope(is_sr, oos_sr)

    deployed_idx = next((i for i, m in enumerate(meta) if m["is_deployed"]), None)
    dsr = (
        deflated_sharpe(R[:, deployed_idx], n_trials=R.shape[1])
        if deployed_idx is not None else {}
    )

    paths = pd.DataFrame({
        "variant": name, "lambda": lam, "is_sharpe": is_sr, "oos_sharpe": oos_sr,
    })
    summary = {
        "variant": name,
        "n_configs": int(R.shape[1]),
        "n_paths": int(len(lam)),
        "pbo": float(pbo),
        "lambda_mean": float(lam.mean()),
        "prob_lambda_positive": float((lam > 0).mean()),
        "degradation_slope": float(slope),
        "deployed_config": (
            f"{meta[deployed_idx]['rule']}/{meta[deployed_idx]['weighting']}/"
            f"{meta[deployed_idx]['rebal']}d" if deployed_idx is not None else None
        ),
        "deployed_dsr": dsr,
    }
    log(f"    PBO={pbo:.4f}  slope={slope:+.4f}  "
        f"DSR(deployed, N={R.shape[1]})={dsr.get('dsr', float('nan')):.4f}")
    return paths, summary


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="track_a", choices=["track_a", "track_b"])
    parser.add_argument("--max-splits", type=int, default=None)
    args = parser.parse_args(argv)

    t0 = time.time()
    log("=== PBO over the deployed selection path ===\n")

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    feat_df = pd.read_parquet(FEAT_PATH)
    ohlcv = pd.read_parquet(OHLCV_PATH)
    fdr = pd.read_parquet(FDR_PATH)
    shap_df = pd.read_parquet(SHAP_PATH)

    track_fdr = fdr.loc[fdr["track"] == args.track].reset_index(drop=True)
    shap_summary = (
        shap_df.loc[shap_df["track"] == args.track]
        .groupby("feature")["mean_abs_shap"].mean()
    )

    tickers = feat_df.index.get_level_values("ticker").unique().tolist()
    panel = books.load_market_panel(ohlcv, tickers, IS_START, IS_END)

    log("  Building the 54-configuration grid (base costs) ...")
    matrix, meta = build_config_books(
        feat_df, track_fdr, shap_summary, panel, cfg,
        cost_settings=(COST_SETTINGS[0],),
    )
    paths_a, summary_a = run_variant("selection_path_54", matrix, meta, args.max_splits)

    log("\n  Building the conservative grid (+ cost dimension) ...")
    matrix_c, meta_c = build_config_books(
        feat_df, track_fdr, shap_summary, panel, cfg, cost_settings=COST_SETTINGS,
    )
    paths_b, summary_b = run_variant(
        "selection_path_plus_costs", matrix_c, meta_c, args.max_splits
    )

    all_paths = pd.concat([paths_a, paths_b], ignore_index=True)
    all_paths.to_parquet(OUT_PATHS, index=False)
    summary_df = pd.DataFrame([
        {k: v for k, v in s.items() if k != "deployed_dsr"}
        for s in (summary_a, summary_b)
    ])
    summary_df.to_parquet(OUT_SUMMARY, index=False)

    payload = {
        "track": args.track,
        "is_window": f"{IS_START}..{IS_END}",
        "n_blocks": N_BLOCKS,
        "variants": [summary_a, summary_b],
        "caveat": (
            "Selection rules and SHAP weights are estimated on the full IS "
            "window before CSCV splits it, so the reported PBO is a lower bound."
        ),
        "elapsed_seconds": time.time() - t0,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float) + "\n")

    # Record to the manifest so the manuscript's PBO numbers are generated
    # rather than transcribed. Without this the old single-feature entries
    # (pbo.deployed.*) stay in the manifest and the coherence checker keeps
    # validating the text against a superseded audit.
    for v in payload["variants"]:
        ns = f"pbo_selection_path.{args.track}.{v['variant']}"
        manifest.record(f"{ns}.pbo", float(v["pbo"]), stage="pbo",
                        track=args.track.split("_")[1].upper(),
                        meta={"n_configs": int(v["n_configs"]),
                              "n_blocks": N_BLOCKS,
                              "deployed": "/".join(str(x) for x in DEPLOYED),
                              "caveat": payload.get("caveat", "")})
        manifest.record(f"{ns}.degradation_slope",
                        float(v["degradation_slope"]), stage="pbo",
                        track=args.track.split("_")[1].upper())
        manifest.record(f"{ns}.n_configs", int(v["n_configs"]), stage="pbo",
                        track=args.track.split("_")[1].upper())
        dsr = v.get("deployed_dsr") or {}
        if dsr.get("dsr") is not None:
            manifest.record(f"{ns}.deployed_dsr", float(dsr["dsr"]),
                            stage="pbo", track=args.track.split("_")[1].upper(),
                            meta={"sr_ann": dsr.get("sr_ann"), "T": dsr.get("T"),
                                  "n_trials": int(v["n_configs"])})

    log(f"\nSaved -> {OUT_PATHS.relative_to(ROOT)}")
    log(f"Saved -> {OUT_SUMMARY.relative_to(ROOT)}")
    log(f"Saved -> {OUT_JSON.relative_to(ROOT)}")
    log(f"\nTotal elapsed: {time.time() - t0:.1f}s")
    return payload


if __name__ == "__main__":
    main()
