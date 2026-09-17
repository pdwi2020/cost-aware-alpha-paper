"""tools/record_synthesis.py — derive the cross-stage manifest entries.

Most manifest keys are written by the stage that computes them. Three groups
were not, and were instead filled in by hand at various points:

  * ``synthesis.*``  — the headline three-window summary
  * ``battery.baselines.*`` — the naive-baseline comparison table
  * ``models.track_*.*_ic_*`` — walk-forward IC by model

Hand-filling is how ``synthesis.track_a.is_net_sharpe_weekly`` came to hold the
*daily*-rebalanced book's Sharpe (0.137) while the deployed book is weekly
(0.188), and how the baseline block survived a full re-run unchanged. This tool
derives all three groups from the on-disk stage outputs so they cannot drift
from the run that produced them.

It reads only outputs that already exist; it recomputes nothing. Run it after
Stage B, before ``tools/check_coherence.py``:

    python3 -u tools/record_synthesis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import manifest  # noqa: E402

PROC = ROOT / "data" / "processed"
IC_BY_FOLD = PROC / "ic_by_fold.parquet"
MLP_IC_BY_FOLD = PROC / "mlp_ic_by_fold.parquet"
BASELINES = PROC / "baselines_metrics.parquet"
BACKTEST_BASE = PROC / "backtest_base.parquet"
SENSITIVITY_3D = PROC / "sensitivity_3d.parquet"

# The cost setting the manuscript calls "base". The rebalancing-frequency table
# is reported at these costs, so the grid row that backs it has to be pinned to
# them and not to the optimistic corner the Pareto summary picks.
BASE_SPREAD_BPS = 3
BASE_IMPACT_COEFF = 0.10

# Pre-v3 keys whose names do not say which rebalancing frequency they measure.
# They were written before the daily/weekly distinction mattered and now sit
# next to explicitly-suffixed v3 keys, where an unqualified name is a trap:
# the deployed book is weekly, but these hold daily values.
SUPERSEDED_KEYS = (
    "backtest.track_a.is_gross_sharpe",
    "backtest.track_a.is_net_sharpe",
    "backtest.track_a.is_annual_turnover",
    # Pre-v3 walk-forward IC duplicates. models.<track>.<model>_ic_mean is the
    # live source; these stayed frozen at their June values through three
    # re-runs and disagreed with it (ensemble 0.1217 vs 0.1161).
    *[f"walkforward.{t}.mean_ic.{m}"
      for t in ("track_a", "track_b")
      for m in ("ridge", "lasso", "rf", "xgb", "lgbm", "ensemble")],
)

# Models as they appear in ic_by_fold.parquet, in the order the manuscript's
# IC table lists them.
IC_MODELS = ("ridge", "lasso", "logistic", "rf", "xgb", "lgbm", "ensemble")

# Baseline strategy names -> the manifest-safe slug used in the key.
BASELINE_SLUGS = {
    "Buy-and-Hold SPY": "Buy_and_Hold_SPY",
    "Momentum L/S (12-1)": "Momentum_L_S_12_1",
    "Reversal L/S (5d)": "Reversal_L_S_5d",
    "Track B (gross, no cost)": "Track_B_gross_no_cost",
}
PERIOD_SLUGS = {"IS 2013-21": "IS_2013_21", "OOS 2022-24": "OOS_2022_24"}


def _require(key: str) -> float:
    """Fetch a manifest value that an upstream stage must already have written."""
    value = manifest.get(key)
    if value is None:
        raise KeyError(
            f"{key!r} is absent from the manifest; run the stage that writes it "
            f"before tools/record_synthesis.py"
        )
    return float(value)


def record_walk_forward_ic() -> None:
    """Per-model IC mean and dispersion across the walk-forward folds."""
    if not IC_BY_FOLD.exists():
        raise FileNotFoundError(f"{IC_BY_FOLD} missing; run src/models/run_walk_forward.py")
    ic = pd.read_parquet(IC_BY_FOLD)
    for track, group in ic.groupby("track"):
        for model in IC_MODELS:
            if model not in group.columns:
                raise KeyError(f"{model!r} not in {IC_BY_FOLD.name}; columns are {sorted(group.columns)}")
            col = group[model].astype(float)
            manifest.record(
                f"models.{track}.{model}_ic_mean", round(float(col.mean()), 4),
                stage="walk_forward", track=track[-1].upper(),
                meta={"n_folds": int(col.notna().sum())},
            )
            # Dispersion across folds, not a standard error: the manuscript's
            # "+/-" column is the fold-to-fold spread.
            manifest.record(
                f"models.{track}.{model}_ic_std", round(float(col.std(ddof=1)), 4),
                stage="walk_forward", track=track[-1].upper(),
            )
        print(f"  {track}: ensemble IC {group['ensemble'].mean():+.4f} "
              f"+/- {group['ensemble'].std(ddof=1):.4f} over {len(group)} folds")


def record_mlp_ic() -> None:
    """MLP walk-forward IC, if the nonlinearity check has been run."""
    if not MLP_IC_BY_FOLD.exists():
        print(f"  (skipped) {MLP_IC_BY_FOLD.name} absent; "
              f"\\MLPicA/\\MLPicB stay undefined and the LaTeX build will fail")
        return
    ic = pd.read_parquet(MLP_IC_BY_FOLD)
    col = "mlp" if "mlp" in ic.columns else "ic"
    for track, group in ic.groupby("track"):
        series = group[col].astype(float)
        manifest.record(f"models.{track}.mlp_ic_mean", round(float(series.mean()), 4),
                        stage="walk_forward", track=track[-1].upper(),
                        meta={"n_folds": int(series.notna().sum())})
        manifest.record(f"models.{track}.mlp_ic_std", round(float(series.std(ddof=1)), 4),
                        stage="walk_forward", track=track[-1].upper())
        print(f"  {track}: MLP IC {series.mean():+.4f} +/- {series.std(ddof=1):.4f}")


def record_is_daily() -> None:
    """The daily-rebalanced in-sample book.

    This is not the deployed configuration (weekly is), but the manuscript
    compares the two to show what rebalancing frequency costs, so it needs a
    source of its own rather than an unqualified key.
    """
    if not BACKTEST_BASE.exists():
        raise FileNotFoundError(f"{BACKTEST_BASE} missing; run src/backtest/run_backtest.py")
    df = pd.read_parquet(BACKTEST_BASE)
    for row in df.itertuples():
        track = row.track
        for field, key in (("gross_pnl_sharpe", "is_gross_sharpe_daily"),
                           ("net_pnl_sharpe", "is_net_sharpe_daily"),
                           ("annual_turnover", "is_annual_turnover_daily"),
                           ("cost_drag_bps", "is_cost_drag_bps_daily")):
            manifest.record(
                f"backtest.{track}.{key}", round(float(getattr(row, field)), 4),
                stage="backtest", track=track.split("_")[1].upper(),
                meta={"rebal_freq": 1, "window": "2013-01-01..2021-12-31",
                      "source": "backtest_base.parquet"},
            )
    print(f"  daily IS recorded for {len(df)} track(s)")


def record_rebal_grid() -> None:
    """The monthly in-sample book, from the base-cost corner of the cost grid.

    The manuscript's rebalancing table (tab:rebal) has three rows. Daily and
    weekly are recorded by the backtest and holdout stages, but the monthly leg
    had no manifest source at all, so that row was hand-typed and drifted: it
    still carried pre-correction values through two full re-runs. Recording it
    here from the same grid the table is built on closes the gap.

    Daily and weekly are deliberately re-derived too, and disagreement with the
    keys the backtest stages wrote is an error rather than a warning: the two
    paths must be computing the same book at the same costs.
    """
    if not SENSITIVITY_3D.exists():
        raise FileNotFoundError(
            f"{SENSITIVITY_3D} missing; run src/backtest/run_sensitivity.py"
        )
    df = pd.read_parquet(SENSITIVITY_3D)
    base = df[(df["spread_bps"] == BASE_SPREAD_BPS)
              & (df["impact_coeff"] == BASE_IMPACT_COEFF)]
    if base.empty:
        raise ValueError(
            f"no rows at base costs (spread={BASE_SPREAD_BPS}, "
            f"impact={BASE_IMPACT_COEFF}) in {SENSITIVITY_3D.name}"
        )

    suffix = {1: "daily", 5: "weekly", 21: "monthly"}
    fields = (("gross_pnl_sharpe", "is_gross_sharpe"),
              ("net_pnl_sharpe", "is_net_sharpe"),
              ("annual_turnover", "is_annual_turnover"),
              ("cost_drag_bps", "is_cost_drag_bps"))
    written = 0
    for row in base.itertuples():
        freq = suffix.get(int(row.rebal_freq))
        if freq is None:
            continue
        for field, stem in fields:
            key = f"backtest.{row.track}.{stem}_{freq}"
            value = round(float(getattr(row, field)), 4)
            existing = manifest.get(key)
            if freq != "monthly" and existing is not None and existing != value:
                raise ValueError(
                    f"{key}: cost grid says {value} but the backtest stage "
                    f"recorded {existing}. The two paths disagree about the "
                    f"same book at the same costs; resolve before recording."
                )
            manifest.record(
                key, value, stage="backtest",
                track=row.track.split("_")[1].upper(),
                meta={"rebal_freq": int(row.rebal_freq),
                      "window": "2013-01-01..2021-12-31",
                      "spread_bps": BASE_SPREAD_BPS,
                      "impact_coeff": BASE_IMPACT_COEFF,
                      "source": "sensitivity_3d.parquet"},
            )
            written += 1
    print(f"  rebalancing grid recorded: {written} key(s) at base costs")


def drop_superseded() -> None:
    """Remove pre-v3 keys that do not name their rebalancing frequency."""
    import json

    path = ROOT / "results" / "manifest" / "manifest.json"
    raw = json.loads(path.read_text())
    removed = [k for k in SUPERSEDED_KEYS if raw.pop(k, None) is not None]
    if removed:
        path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    print(f"  dropped {len(removed)} superseded key(s): {removed}")


def record_forward_holdout() -> None:
    """Backfill the forward-window result from the run's saved JSON.

    run_forward_holdout.py now records these itself, but re-running it to
    populate the manifest would spend another entry in the single-touch log.
    Reading its output is equivalent and costs no touch.
    """
    import json

    path = ROOT / "results" / "staging" / "forward_holdout.json"
    if not path.exists():
        print("  (skipped) forward_holdout.json absent; window not scored yet")
        return
    d = json.loads(path.read_text())
    track = d["track"]
    res, inf = d["result"], d["result"]["inference"]
    meta = {"window": d["window"], "spec_blob": d["spec_blob"][:12],
            "window_rule": d["window_rule"]}
    pairs = {
        "net_sharpe": res["metrics"]["net_pnl_sharpe"],
        "gross_sharpe": res["metrics"]["gross_pnl_sharpe"],
        "annual_turnover": res["metrics"]["annual_turnover"],
        "cost_drag_bps": res["metrics"]["cost_drag_bps"],
        "n_days": inf["T"],
        "net_sharpe_boot_p": inf["boot_p"],
        "net_sharpe_nw_t": inf["newey_west"]["t"],
        "net_sharpe_ci95_lo": inf["ci_95"]["lo"],
        "net_sharpe_ci95_hi": inf["ci_95"]["hi"],
        "net_sharpe_ci90_lo": inf["ci_90"]["lo"],
        "net_sharpe_ci90_hi": inf["ci_90"]["hi"],
    }
    for key, value in pairs.items():
        manifest.record(f"forward.{track}.{key}", round(float(value), 4),
                        stage="forward", track=track.split("_")[1].upper(), meta=meta)
    manifest.record(f"forward.{track}.classification",
                    str(inf["primary_classification"]),
                    stage="forward", track=track.split("_")[1].upper(), meta=meta)
    print(f"  forward {track}: {res['metrics']['net_pnl_sharpe']:+.4f} over "
          f"{inf['T']} days -> {inf['primary_classification']}")


def record_baselines() -> None:
    """The naive-baseline comparison table."""
    if not BASELINES.exists():
        raise FileNotFoundError(f"{BASELINES} missing; run src/backtest/run_baselines.py")
    df = pd.read_parquet(BASELINES)
    for row in df.itertuples():
        strat = BASELINE_SLUGS.get(row.strategy)
        period = PERIOD_SLUGS.get(row.period)
        if strat is None or period is None:
            raise KeyError(
                f"unmapped baseline row ({row.strategy!r}, {row.period!r}); add it to "
                f"BASELINE_SLUGS/PERIOD_SLUGS so the key stays stable across runs"
            )
        for field in ("gross_sr", "net_sr", "annual_to", "cost_drag"):
            manifest.record(
                f"battery.baselines.{strat}.{period}.{field}",
                round(float(getattr(row, field)), 4),
                stage="baselines",
            )
    print(f"  {len(df)} baseline rows recorded")


def record_synthesis() -> None:
    """The three-window headline summary, each value taken from its own stage."""
    is_net = _require("backtest.track_a.is_net_sharpe_weekly")
    is_gross = _require("backtest.track_a.is_gross_sharpe_weekly")
    expl = _require("window.exploratory_2022_2024.track_a.net_sharpe")
    locked = _require("window.locked_oos_2025.track_a.net_sharpe")
    locked_gross = _require("window.locked_oos_2025.track_a.gross_sharpe")
    locked_p = _require("window.locked_oos_2025.track_a.net_sharpe_boot_p")
    locked_t = _require("window.locked_oos_2025.track_a.net_sharpe_nw_t")
    ci_lo = _require("window.locked_oos_2025.track_a.net_sharpe_ci95_lo")
    ci_hi = _require("window.locked_oos_2025.track_a.net_sharpe_ci95_hi")
    klass = manifest.get("window.locked_oos_2025.track_a.classification")

    pairs = {
        "synthesis.track_a.is_net_sharpe_weekly": is_net,
        "synthesis.track_a.is_gross_sharpe_weekly": is_gross,
        "synthesis.track_a.exploratory_2022_2024_net_sharpe": expl,
        "synthesis.track_a.locked_oos_2025_net_sharpe": locked,
        "synthesis.track_a.locked_oos_2025_pvalue": locked_p,
        # The locked-window gross Sharpe had been left at the pre-correction
        # +0.728 while its net had been updated, which made the pair incoherent.
        "oos.track_a.gross_sharpe": locked_gross,
        "oos.track_a.net_sharpe": locked,
        "oos.track_a.net_sharpe_pvalue": locked_p,
        "oos.track_a.net_sharpe_tstat": locked_t,
        "oos.track_a.net_sharpe_ci95_lo": ci_lo,
        "oos.track_a.net_sharpe_ci95_hi": ci_hi,
    }
    for key, value in pairs.items():
        manifest.record(key, value, stage="synthesis", track="A")

    manifest.record(
        "synthesis.note",
        f"Track A net Sharpe: IS {is_net:+.3f} (weekly) / exploratory 2022-24 "
        f"{expl:+.3f} / locked 2025 {locked:+.3f} ({klass}, p={locked_p:.2f}, "
        f"95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]). Both out-of-sample windows are "
        f"negative; no stable tradeable edge.",
        stage="synthesis", track="A",
    )
    print(f"  IS {is_net:+.3f} / exploratory {expl:+.3f} / locked {locked:+.3f} "
          f"(p={locked_p:.2f}, {klass})")


def main() -> int:
    print("=== recording walk-forward IC ===")
    record_walk_forward_ic()
    record_mlp_ic()
    print("=== recording in-sample books ===")
    record_is_daily()
    record_rebal_grid()
    drop_superseded()
    record_forward_holdout()
    print("=== recording baselines ===")
    record_baselines()
    print("=== recording synthesis ===")
    record_synthesis()
    print("\nManifest updated. Now run: python3 -u tools/check_coherence.py --strict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
