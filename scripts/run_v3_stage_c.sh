#!/usr/bin/env bash
# run_v3_stage_c.sh — the robustness battery, on the v3 panel.
#
# Stage A produced the discoveries and Stage B the deployed backtest. This
# stage runs everything the manuscript cites as *corroborating* evidence:
# the ex-mega-cap sub-universe, the regime-conditional books, the weighting
# and feature-set ablation, the naive-ML comparison, and conformal
# calibration.
#
# None of these were re-run for v3. Their manifest entries dated from June and
# July, which meant the manuscript was citing pre-correction numbers (computed
# before the vwap_dev and overnight_gap basis fixes and before the daily
# point-in-time universe) as support for a v3 verdict. They are also the
# stages that never recorded to the manifest at all, so the numbers had been
# transcribed by hand; tools/record_battery.py now derives them from the
# outputs this script writes.
#
# Order is cheapest-first so a failure surfaces early.

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
}

# Every stage here consumes the Stage A selections, so an output produced
# before the current fdr_results.parquet describes a different feature set.
UPSTREAM="data/processed/fdr_results.parquet"

run_stage() {
    local name="$1" limit="$2" output="$3"; shift 3
    if [ "$output" != "-" ] && [ -s "$output" ] && [ "$output" -nt "$UPSTREAM" ]; then
        say "skip $name (fresh output present: $output)"
        return 0
    fi
    disk_guard "$name"
    say "=== $name start"
    if ! timeout "$limit" python3 -u "$@" > "$LOG_DIR/${name}.log" 2>&1; then
        say "FAIL: $name, last lines:"
        tail -25 "$LOG_DIR/${name}.log"
        return 1
    fi
    say "=== $name done"
}

for f in data/processed/fdr_results.parquet data/processed/shap_summary.parquet \
         data/processed/ic_by_fold.parquet data/processed/backtest_base.parquet; do
    if [ ! -f "$f" ]; then
        say "ABORT: $f missing — run Stages A and B first"
        exit 1
    fi
done
say "Stage A/B outputs present"

FAILED=""
run_stage excl_megacap_v3 3600 data/processed/excl_megacap_metrics.parquet \
    src/backtest/run_excl_megacap.py            || FAILED="$FAILED excl_megacap"
run_stage regime_signal_v3 5400 data/processed/regime_signal_results.parquet \
    src/backtest/run_regime_signal.py           || FAILED="$FAILED regime_signal"
run_stage ablation_v3 5400 data/processed/ablation_oos.parquet \
    src/backtest/run_ablation.py                || FAILED="$FAILED ablation"
run_stage ml_baseline_v3 7200 data/processed/ml_baseline_metrics.parquet \
    src/baselines/run_ml_baseline.py            || FAILED="$FAILED ml_baseline"
run_stage mlp_baseline_v3 7200 data/processed/mlp_ic_by_fold.parquet \
    src/models/run_mlp_baseline.py              || FAILED="$FAILED mlp_baseline"
run_stage conformal_v3 7200 data/processed/conformal_coverage.parquet \
    src/calibration/run_conformal.py            || FAILED="$FAILED conformal"

if [ -n "$FAILED" ]; then
    say "STAGE C INCOMPLETE — failed:$FAILED"
    say "Manifest NOT updated; the manuscript keeps its previous values for those stages."
    exit 1
fi

say "=== recording battery results to the manifest"
if ! python3 -u tools/record_battery.py > "$LOG_DIR/record_battery.log" 2>&1; then
    say "FAIL: record_battery"; tail -25 "$LOG_DIR/record_battery.log"; exit 1
fi
cat "$LOG_DIR/record_battery.log"

say "=== STAGE C COMPLETE"
say "next: python3 -u tools/check_coherence.py --strict"
