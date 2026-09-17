#!/usr/bin/env bash
# run_v3_full_chain.sh — everything downstream of the feature panel, in order.
#
# Written after a corr_* source fix forced a second full re-run: issuing the
# stages one at a time is error-prone, and the ordering constraints are real
# (Stage A waits on the walk-forward, Stage B consumes Stage A's selections,
# the windows consume Stage B's sensitivity grid).
#
# Assumes data/processed/features_all.parquet is already current. Each stage
# script has its own freshness guard, so completed work is skipped on a restart.
#
#   bash scripts/run_v3_full_chain.sh

set -uo pipefail
cd "$HOME/ML_Paper" || exit 1

LOG_DIR="results/staging"
mkdir -p "$LOG_DIR"
say() { echo "[$(date +%H:%M:%S)] $*"; }

PANEL="data/processed/features_all.parquet"
[ -f "$PANEL" ] || { say "ABORT: $PANEL missing"; exit 1; }

step() {
    local name="$1" limit="$2"; shift 2
    say "=== $name start"
    if ! timeout "$limit" "$@" > "$LOG_DIR/${name}.log" 2>&1; then
        say "FAIL: $name (see $LOG_DIR/${name}.log)"
        tail -20 "$LOG_DIR/${name}.log"
        exit 1
    fi
    say "=== $name done"
}

# ---------------------------------------------------------------------------
# 1. Walk-forward, then Stage A (which waits for the walk-forward to exit).
# The stale IC file is removed first: Stage A's presence check is
# existence-only, so a leftover file from the previous panel would pass it.
# ---------------------------------------------------------------------------
IC="data/processed/ic_by_fold.parquet"
if [ -s "$IC" ] && [ "$IC" -nt "$PANEL" ]; then
    say "skip walk_forward_v3 (fresh $IC present)"
else
    # Remove a stale file first: Stage A's presence check is existence-only, so
    # a leftover from a previous panel would pass it silently.
    rm -f "$IC"
    step walk_forward_v3 14400 python3 -u src/models/run_walk_forward.py
fi

say "=== stage_a start"
if ! bash scripts/run_v3_stage_a.sh > "$LOG_DIR/stage_a_driver.log" 2>&1; then
    say "FAIL: stage_a"; tail -20 "$LOG_DIR/stage_a_driver.log"; exit 1
fi
say "=== stage_a done"

step search_adjusted 3600 python3 -u src/fdr/run_search_adjusted_fdr.py
# The table and macros are rendered straight from the run's JSON. They were
# outside the chain once, and the result was tab_search_adjusted.tex sitting at
# the previous vintage's counts while the JSON beside it held the current ones.
step tab_search_adjusted 300 python3 -u tools/gen_tab_search_adjusted.py

# ---------------------------------------------------------------------------
# 2. Stage B (TA-FDR, PBO, costs, backtests) and Stage C (battery).
# ---------------------------------------------------------------------------
say "=== stage_b start"
if ! bash scripts/run_v3_stage_b.sh > "$LOG_DIR/stage_b_driver.log" 2>&1; then
    say "FAIL: stage_b"; tail -20 "$LOG_DIR/stage_b_driver.log"; exit 1
fi
say "=== stage_b done"

say "=== stage_c start"
if ! bash scripts/run_v3_stage_c.sh > "$LOG_DIR/stage_c_driver.log" 2>&1; then
    say "FAIL: stage_c"; tail -20 "$LOG_DIR/stage_c_driver.log"; exit 1
fi
say "=== stage_c done"

# ---------------------------------------------------------------------------
# 3. Windows and capacity. run_holdout writes the IS leg too, so it must come
# after the sensitivity grid that Stage B builds.
# ---------------------------------------------------------------------------
step holdout_locked 3600 python3 -u src/backtest/run_holdout.py
step holdout_exploratory 3600 python3 -u src/backtest/run_holdout.py \
    --start 2022-01-01 --end 2024-12-31 --label exploratory_2022_2024
step aum_sweep 3600 python3 -u src/backtest/run_aum_sweep.py
step turnover_decomp 3600 python3 -u src/backtest/run_turnover_decomp.py
step r3b_cross_universe 3600 python3 -u src/backtest/run_r3b_cross_universe.py
step aci 3600 python3 -u src/calibration/run_aci.py

# ---------------------------------------------------------------------------
# 4. Manifest, macros, figures.
# ---------------------------------------------------------------------------
step record_synthesis 1800 python3 -u tools/record_synthesis.py
step regenerate_numbers 600 python3 -u tools/regenerate_numbers.py
step make_figures 3600 python3 -u src/figures/make_figures.py

say "=== FULL CHAIN COMPLETE"
say "next: tools/check_coherence.py, then the manuscript sync, then the forward touch"
