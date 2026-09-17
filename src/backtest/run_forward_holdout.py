"""The forward holdout: one scored look at data never used in development.

Why this exists
---------------
Reviewer 1 (point 6) questioned "the claim of a single-touch holdout", and the
2025 window the Array submission used is no longer pristine: it was scored once
under the annual-snapshot universe, and the pipeline has since been rebuilt.
The honest response is a window that has never been scored at all, so this
script evaluates 2025-08-01 onwards exactly once, with the specification frozen
beforehand.

Three guards, because a single-touch claim is only worth the enforcement
behind it:

1. **Frozen spec.** The specification file must be committed and clean. A
   specification that can still be edited is not a pre-commitment.
2. **Touch log.** `results/forward_touch_log.json` records every evaluation.
   A second run against the same spec and window is refused unless it is
   explicitly labelled a re-evaluation with a stated reason, and the label is
   written into the log so the paper can disclose it.
3. **Frozen inputs.** Feature selection and weights come from the committed
   in-sample artifacts; nothing is re-fit on forward data.

What it reports
---------------
Net and gross Sharpe, the stationary-bootstrap confidence intervals and HAC
test, the classification against the economic threshold SR* (so "not
significant" is distinguishable from "too small to matter"), the factor-adjusted
alpha (dollar-neutral is not beta-neutral), and the same pre-specified baselines
evaluated in the same run under the same cost model.

Run
---
    python3 -u src/backtest/run_forward_holdout.py --dry-run     # no scoring
    python3 -u src/backtest/run_forward_holdout.py --confirm-single-touch
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import books
from src.backtest.factor_alpha import factor_regression
from src.backtest.oos_inference import window_inference
import src.manifest as manifest
from src.backtest.portfolio import PortfolioSimulator

SPEC_PATH = ROOT / "config" / "spec_v3.yaml"
FEAT_PATH = ROOT / "data" / "processed" / "features_all.parquet"
OHLCV_PATH = ROOT / "data" / "processed" / "daily_ohlcv_v3.parquet"
FDR_PATH = ROOT / "data" / "processed" / "fdr_results.parquet"
SHAP_PATH = ROOT / "data" / "processed" / "shap_summary.parquet"
CFG_PATH = ROOT / "configs" / "backtest.yaml"

TOUCH_LOG = ROOT / "results" / "forward_touch_log.json"
OUT_PNL = ROOT / "data" / "processed" / "forward_pnl_track_a.parquet"
OUT_JSON = ROOT / "results" / "staging" / "forward_holdout.json"

INTRADAY_FEATURES = {"vwap_dev", "vol_clock", "vol_sig_ratio"}
INTRADAY_LAST_DATE = "2026-03-31"


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def spec_blob_hash(path: Path) -> str:
    """git blob hash of the spec file as it sits on disk."""
    out = subprocess.run(
        ["git", "hash-object", str(path)], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def spec_is_committed_and_clean(path: Path) -> tuple[bool, str]:
    """True when the spec is tracked by git and has no uncommitted changes."""
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if tracked.returncode != 0:
        return False, "spec file is not tracked by git"
    dirty = subprocess.run(
        ["git", "status", "--porcelain", str(path)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    if dirty.stdout.strip():
        return False, "spec file has uncommitted changes"
    return True, "committed and clean"


def load_touch_log() -> list[dict]:
    if TOUCH_LOG.exists():
        return json.loads(TOUCH_LOG.read_text())
    return []


def record_touch(entry: dict) -> None:
    entries = load_touch_log()
    entries.append(entry)
    TOUCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    TOUCH_LOG.write_text(json.dumps(entries, indent=2) + "\n")


# ---------------------------------------------------------------------------
# The frozen strategy
# ---------------------------------------------------------------------------

def frozen_weights(track: str) -> tuple[pd.Series, list[str]]:
    """Deployed weights: BH-selected features, SHAP-weighted, signed by mean IC.

    Read from the in-sample artifacts; nothing here looks at forward data.
    """
    fdr = pd.read_parquet(FDR_PATH)
    shap_df = pd.read_parquet(SHAP_PATH)
    track_fdr = fdr.loc[fdr["track"] == track]
    selected = track_fdr.loc[track_fdr["bh_rejected"], "feature"].tolist()
    if not selected:
        raise RuntimeError(
            f"{track} has no BH-selected features, so it has no deployable "
            "composite; report that rather than inventing one."
        )
    shap_mean = (
        shap_df.loc[shap_df["track"] == track]
        .groupby("feature")["mean_abs_shap"].mean()
    )
    ic = track_fdr.set_index("feature")["ic_bar"]
    signs = np.sign(ic.reindex(selected).fillna(1.0)).replace(0.0, 1.0)
    mag = shap_mean.reindex(selected)
    mag = mag.fillna(mag.mean() if np.isfinite(mag.mean()) else 1.0)
    w = signs * mag
    return w / w.abs().sum(), selected


def resolve_window(selected: list[str], spec: dict) -> tuple[str, str, str, list[str]]:
    """Apply the spec's rule for composites that use 1-minute-derived features."""
    fw = spec["windows"]["forward_holdout"]
    start, end = fw["start"], fw["end"]
    used_intraday = sorted(set(selected) & INTRADAY_FEATURES)
    if used_intraday:
        return (
            start, INTRADAY_LAST_DATE,
            f"composite uses {used_intraday}, which exist only to "
            f"{INTRADAY_LAST_DATE}; the full window is reported as a daily-only "
            "secondary variant per spec",
            used_intraday,
        )
    return start, end, "composite is daily-computable; full window used", []


# ---------------------------------------------------------------------------
# Baselines, scored in the same run under the same cost model
# ---------------------------------------------------------------------------

def baseline_books(feat_df, panel, sim, start, end) -> dict[str, pd.DataFrame]:
    """Pre-specified naive comparators: 12-1 momentum and 5-day reversal."""
    out = {}
    for name, feature, sign in (
        ("momentum_12_1", "mom_12_1", +1.0),
        ("reversal_5d", "ret_5d", -1.0),
    ):
        if feature in feat_df.columns:
            out[name] = books.single_feature_book(
                feat_df, feature, sign, panel, sim, rebal_freq=5, start=start, end=end
            )
    return out


def summarise(pnl: pd.DataFrame, sim: PortfolioSimulator, sr_star: float) -> dict:
    metrics = sim.compute_metrics(pnl)
    net = pnl["net_pnl"].to_numpy(dtype=float)
    inference = window_inference(net, sr_star=sr_star, B=10_000, seed=20260915)
    return {"metrics": metrics, "inference": inference}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="track_a", choices=["track_a", "track_b"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Check the guards and print the plan; score nothing.")
    parser.add_argument("--confirm-single-touch", action="store_true",
                        help="Required to actually score the forward window.")
    parser.add_argument("--reevaluation-reason", default=None,
                        help="Required to score a window that the log shows was already scored.")
    args = parser.parse_args(argv)

    t0 = time.time()
    spec = yaml.safe_load(SPEC_PATH.read_text())
    blob = spec_blob_hash(SPEC_PATH)
    committed, why = spec_is_committed_and_clean(SPEC_PATH)

    w, selected = frozen_weights(args.track)
    start, end, window_note, intraday_used = resolve_window(selected, spec)

    log("=== Forward holdout ===")
    log(f"  spec            : {SPEC_PATH.relative_to(ROOT)}  blob={blob[:12]}  ({why})")
    log(f"  track           : {args.track}")
    log(f"  selected        : {len(selected)} features -> {selected}")
    log(f"  window          : {start} .. {end}")
    log(f"  window rule     : {window_note}")

    # Prior touches are matched on window and track, NOT on the spec blob.
    #
    # Matching on the blob as well looked stricter and was in fact a hole: any
    # edit to config/spec_v3.yaml, down to a typo in a comment, changes the
    # hash, and the existing entries then stop matching. The log would still
    # hold them, the guard would no longer see them, and the window would be
    # re-scorable with no --reevaluation-reason required. A single-touch
    # mechanism that a one-character edit disarms is not one.
    #
    # The window is what can only be touched once. The blob is recorded so a
    # reader can tell which specification each touch was scored under, and a
    # touch under a different blob is reported rather than ignored.
    touches = load_touch_log()
    prior = [
        e for e in touches
        if e.get("window") == f"{start}..{end}" and e.get("track") == args.track
    ]
    if prior:
        log(f"  PRIOR TOUCHES   : {len(prior)} (this window has been scored before)")
        for e in prior:
            same = "same spec" if e.get("spec_blob") == blob else "DIFFERENT spec"
            log(f"     {e.get('scored_at_utc', '?')}  blob={str(e.get('spec_blob'))[:12]}"
                f"  ({same})")
        if any(e.get("spec_blob") != blob for e in prior):
            log("     note: the specification has changed since a previous touch; "
                "that does not reset the count.")

    if args.dry_run:
        log("\n  dry run: nothing scored.")
        return {"dry_run": True, "spec_blob": blob, "window": f"{start}..{end}"}

    if not committed:
        raise SystemExit(
            f"REFUSED: {why}. Commit config/spec_v3.yaml first: a specification "
            "that can still be edited is not a pre-commitment."
        )
    if not args.confirm_single_touch:
        raise SystemExit("REFUSED: pass --confirm-single-touch to score the window.")
    if prior and not args.reevaluation_reason:
        raise SystemExit(
            "REFUSED: the log shows this window was already scored under this "
            "spec. Pass --reevaluation-reason '...' to record a disclosed "
            "re-evaluation."
        )

    feat_df = pd.read_parquet(FEAT_PATH)
    ohlcv = pd.read_parquet(OHLCV_PATH)
    tickers = feat_df.index.get_level_values("ticker").unique().tolist()
    panel = books.load_market_panel(ohlcv, tickers, start, end)
    sim = PortfolioSimulator(config_path=str(CFG_PATH))
    sr_star = float(spec["validation"]["economic_threshold"]["sr_star"])

    log("\n  building the frozen composite book ...")
    pnl = books.composite_book(feat_df, w, panel, sim, rebal_freq=5, start=start, end=end)
    pnl.to_parquet(OUT_PNL)
    result = summarise(pnl, sim, sr_star)

    log("  factor-adjusted alpha ...")
    try:
        result["factor_alpha"] = factor_regression(pnl["net_pnl"])
    except Exception as exc:                       # factors may lag the window
        result["factor_alpha"] = {"error": str(exc)}

    # Record to the manifest here, not afterwards by hand: this is the single
    # pre-committed forward result and it must have the same provenance as
    # every other number in the paper.
    ns = f"forward.{args.track}"
    meta = {"window": f"{start}..{end}", "spec_blob": blob[:12],
            "window_rule": window_note, "n_features": len(selected)}
    inf = result["inference"]
    for key, value in (
        ("net_sharpe", result["metrics"]["net_pnl_sharpe"]),
        ("gross_sharpe", result["metrics"]["gross_pnl_sharpe"]),
        ("annual_turnover", result["metrics"]["annual_turnover"]),
        ("cost_drag_bps", result["metrics"]["cost_drag_bps"]),
        ("n_days", inf["T"]),
        ("net_sharpe_boot_p", inf["boot_p"]),
        ("net_sharpe_nw_t", inf["newey_west"]["t"]),
        ("net_sharpe_ci95_lo", inf["ci_95"]["lo"]),
        ("net_sharpe_ci95_hi", inf["ci_95"]["hi"]),
        ("net_sharpe_ci90_lo", inf["ci_90"]["lo"]),
        ("net_sharpe_ci90_hi", inf["ci_90"]["hi"]),
    ):
        manifest.record(f"{ns}.{key}", float(value), stage="forward",
                        track=args.track.split("_")[1].upper(), meta=meta)
    manifest.record(f"{ns}.classification", str(inf["primary_classification"]),
                    stage="forward", track=args.track.split("_")[1].upper(), meta=meta)

    log("  baselines ...")
    result["baselines"] = {
        name: summarise(book, sim, sr_star)
        for name, book in baseline_books(feat_df, panel, sim, start, end).items()
    }

    payload = {
        "spec_blob": blob,
        "spec_committed": committed,
        "track": args.track,
        "window": f"{start}..{end}",
        "window_rule": window_note,
        "intraday_features_used": intraday_used,
        "selected_features": selected,
        "weights": {k: float(v) for k, v in w.items()},
        "result": result,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "reevaluation_reason": args.reevaluation_reason,
        "elapsed_seconds": time.time() - t0,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    record_touch({k: payload[k] for k in
                  ("spec_blob", "track", "window", "scored_at_utc", "reevaluation_reason")})

    m, inf = result["metrics"], result["inference"]
    log(f"\n  net Sharpe      : {m['net_pnl_sharpe']:+.3f}  (gross {m['gross_pnl_sharpe']:+.3f})")
    log(f"  90% CI          : [{inf['ci_90']['lo']:+.2f}, {inf['ci_90']['hi']:+.2f}]  "
        f"HAC t={inf['newey_west']['t']:+.2f} p={inf['newey_west']['p']:.3f}")
    log(f"  vs SR*={sr_star}    : {inf['primary_classification']}")
    log(f"\nSaved -> {OUT_JSON.relative_to(ROOT)}")
    log(f"Touch logged -> {TOUCH_LOG.relative_to(ROOT)}")
    return payload


if __name__ == "__main__":
    main()
