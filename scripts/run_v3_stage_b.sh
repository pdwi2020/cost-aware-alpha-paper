#!/usr/bin/env bash
# run_v3_stage_b.sh — screens and in-sample backtests of the v3 re-run.
#
# Launch only AFTER reviewing Stage A's discovery counts: the feature set it
# selects is the input to everything here.
#
# Order:
#   TA-FDR v3 (B=2000) -> PBO v2 (selection path) -> composite signals
#   -> in-sample backtest -> cost sensitivity -> naive baselines
#
# Deliberately NOT included, because they touch holdout windows and must be run
# with explicit intent rather than as part of a chain:
#   * src/backtest/run_holdout.py        (re-evaluates the previously-touched 2025 window)
#   * src/backtest/run_forward_holdout.py (the single forward touch; needs a committed spec)

set -uo pipefail
cd "$HOME/ML_Paper" || exit 1

LOG_DIR="results/staging"
MIN_FREE_MB=1500
mkdir -p "$LOG_DIR"

say() { echo "[$(date +%H:%M:%S)] $*"; }

disk_guard() {
    local free_mb
    free_mb=$(df -m "$HOME" | tail -1 | awk '{print $4}')
    if [ "$free_mb" -lt "$MIN_FREE_MB" ]; then
        say "ABORT: only ${free_mb}MB free (need ${MIN_FREE_MB}MB) before $1"
        exit 2
    fi
    say "disk ok (${free_mb}MB free) before $1"
}

# Everything here consumes the FDR selections, so an output produced before the
# current fdr_results.parquet describes a different feature set.
UPSTREAM="data/processed/fdr_results.parquet"

run_stage() {
    local name="$1" limit="$2" output="$3"; shift 3
    # Restart-safe, but only for output that is NEWER than the upstream input.
    # An existence-only check silently accepted a B=200 smoke file built from
    # the pre-correction feature set, which is worse than rerunning.
    if [ "$output" != "-" ] && [ -s "$output" ]; then
        if [ "$output" -nt "$UPSTREAM" ]; then
            say "skip $name (fresh output present: $output)"
            return 0
        fi
        say "$name: output is older than $UPSTREAM — rebuilding"
    fi
    disk_guard "$name"
    say "=== $name start"
    if ! timeout "$limit" python3 -u "$@" > "$LOG_DIR/${name}.log" 2>&1; then
        say "FAIL: $name, last lines:"
        tail -20 "$LOG_DIR/${name}.log"
        exit 1
    fi
    say "=== $name done"
}

# ---------------------------------------------------------------------------
# Preconditions: Stage A's outputs must exist.
# ---------------------------------------------------------------------------
for f in data/processed/fdr_results.parquet data/processed/shap_summary.parquet; do
    if [ ! -f "$f" ]; then
        say "ABORT: $f missing — run Stage A first"
        exit 1
    fi
done
say "Stage A outputs present"

# ---------------------------------------------------------------------------
run_stage ta_fdr_v3_B2000 7200 data/processed/ta_fdr_v3.parquet \
    src/backtest/run_ta_fdr.py --B 2000
run_stage pbo_selection_path 7200 data/processed/pbo_selection_path.parquet \
    src/fdr/run_pbo_selection_path.py
run_stage cost_scenarios_v3 7200 data/processed/cost_scenarios.parquet \
    src/backtest/run_cost_scenarios.py
run_stage signals_v3 1800 - src/backtest/generate_signals.py
run_stage backtest_is_v3 3600 data/processed/backtest_base.parquet \
    src/backtest/run_backtest.py
run_stage sensitivity_v3 5400 data/processed/sensitivity_arv.parquet \
    src/backtest/run_sensitivity.py
run_stage baselines_v3 3600 data/processed/baselines_metrics.parquet \
    src/backtest/run_baselines.py

# ---------------------------------------------------------------------------
say "=== STAGE B COMPLETE — headline screen results"
python3 - <<'PY'
from pathlib import Path
import json
import pandas as pd

def show(path, fn):
    p = Path(path)
    if not p.exists():
        print(f"  (missing) {path}")
        return
    try:
        fn(p)
    except Exception as exc:
        print(f"  (unreadable) {path}: {exc}")

def ta_fdr(p):
    df = pd.read_parquet(p)
    for track, g in df.groupby("track"):
        print(f"  TA-FDR {track}: BH_net {int(g['bh_net_selected'].sum())}/{len(g)}  "
              f"BH_gross {int(g['bh_gross_selected'].sum())}/{len(g)}  "
              f"BH_stud {int(g['bh_stud_selected'].sum())}/{len(g)}")
        top = g.nsmallest(3, "p_net")[["feature", "mean_net", "p_net", "p_gross"]]
        for r in top.itertuples():
            print(f"      {r.feature:<22s} mean_net={r.mean_net:+.6f} "
                  f"p_net={r.p_net:.4f} p_gross={r.p_gross:.4f}")

def pbo(p):
    for v in json.loads(p.read_text())["variants"]:
        dsr = v.get("deployed_dsr", {}) or {}
        print(f"  PBO {v['variant']}: {v['pbo']:.4f} over {v['n_configs']} configs, "
              f"slope {v['degradation_slope']:+.3f}, "
              f"DSR(deployed)={dsr.get('dsr', float('nan')):.4f}")

def base(p):
    df = pd.read_parquet(p)
    cols = [c for c in ("strategy", "period", "gross_sr", "net_sr", "cost_drag") if c in df.columns]
    print(df[cols].to_string(index=False))

show("data/processed/ta_fdr_v3.parquet", ta_fdr)
show("results/staging/pbo_selection_path.json", pbo)
show("data/processed/backtest_base.parquet",
     lambda p: print(pd.read_parquet(p).to_string(index=False)))
show("data/processed/baselines_metrics.parquet", base)
PY
say "logs: ta_fdr_v3_B2000 pbo_selection_path signals_v3 backtest_is_v3 sensitivity_v3 baselines_v3"
