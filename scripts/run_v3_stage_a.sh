#!/usr/bin/env bash
# run_v3_stage_a.sh — discovery stages of the v3 re-run, unattended.
#
# Waits for the walk-forward to finish, then runs, in order:
#   SHAP attribution -> FDR (BH) -> BHY -> placebo -> embargo
#
# Deliberately NOT included: src/fdr/run_search_adjusted_fdr.py. Its trip-wires
# assert the v2 headline counts (BH 12/30, BHY 11/30, Track B 0/30) and would
# abort this chain by design on the rebuilt panel. It runs once those expected
# counts are updated to the v3 values.
#
# Halts on the first failure, and refuses to start a stage when the disk is low:
# this machine has 8 GB of RAM and a nearly full volume, and swap lives on the
# same volume, so an out-of-space failure mid-write is a real risk.

set -uo pipefail
cd "$HOME/ML_Paper" || exit 1

LOG_DIR="results/staging"
MIN_FREE_MB=1200
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

# Every stage here reads the feature panel, so output older than the panel
# describes a superseded set of features.
UPSTREAM="data/processed/features_all.parquet"

run_stage() {
    local name="$1" limit="$2" script="$3" output="${4:-}"
    # Restart-safe (this chain has been killed twice), but only for output that
    # is NEWER than the panel. An existence-only check would silently keep
    # results built from the pre-fix vwap_dev column.
    if [ -n "$output" ] && [ -s "$output" ]; then
        if [ "$output" -nt "$UPSTREAM" ]; then
            say "skip $name (fresh output present: $output)"
            return 0
        fi
        say "$name: output is older than $UPSTREAM — rebuilding"
    fi
    disk_guard "$name"
    say "=== $name start"
    if ! timeout "$limit" python3 -u "$script" > "$LOG_DIR/${name}.log" 2>&1; then
        say "FAIL: $name (exit $?), last lines:"
        tail -20 "$LOG_DIR/${name}.log"
        exit 1
    fi
    say "=== $name done"
}

# ---------------------------------------------------------------------------
# 0. Wait for the walk-forward that is already running.
# ---------------------------------------------------------------------------
say "waiting for run_walk_forward to finish ..."
# Match the SCRIPT PATH, not the bare job name. The health monitor's command
# line contains the literal text "run_walk_forward" (it is part of the monitor's
# own search pattern), so both `pgrep -f run_walk_forward` and the bracketed
# variant matched the monitor and this loop waited forever on it. No other
# process mentions "src/models/run_walk_forward.py".
while pgrep -f 'src/models/run_walk_forward\.py' > /dev/null 2>&1; do sleep 30; done
say "walk-forward process has exited"

if [ ! -f data/processed/ic_by_fold.parquet ]; then
    say "ABORT: data/processed/ic_by_fold.parquet was never written"
    tail -20 "$LOG_DIR/walk_forward_v3.log" 2>/dev/null
    exit 1
fi
say "ic_by_fold.parquet present ($(stat -f%z data/processed/ic_by_fold.parquet) bytes)"
python3 - <<'PY'
import pandas as pd
ic = pd.read_parquet("data/processed/ic_by_fold.parquet")
print("  ic_by_fold rows:", len(ic), "| tracks:", sorted(ic["track"].unique()))
cols = [c for c in ic.columns if c not in ("fold", "track")]
print(ic.groupby("track")[cols].mean().round(4).to_string())
PY

# ---------------------------------------------------------------------------
# 1-5. Discovery stages.
# ---------------------------------------------------------------------------
run_stage shap_v3     5400 src/models/shap_analysis.py data/processed/shap_summary.parquet
run_stage fdr_v3      5400 src/fdr/run_fdr.py          data/processed/fdr_results.parquet
run_stage bhy_v3      1800 src/fdr/run_bhy.py
run_stage placebo_v3  3600 src/fdr/run_fdr_placebo.py
run_stage embargo_v3  3600 src/fdr/run_fdr_embargoed.py

# ---------------------------------------------------------------------------
# Summary the next step actually needs: how many features survived.
# ---------------------------------------------------------------------------
say "=== STAGE A COMPLETE — headline discovery counts"
python3 - <<'PY'
import pandas as pd
fdr = pd.read_parquet("data/processed/fdr_results.parquet")
for track, grp in fdr.groupby("track"):
    bh = int(grp["bh_rejected"].sum()) if "bh_rejected" in grp else -1
    bhy = int(grp["bhy_rejected"].sum()) if "bhy_rejected" in grp else -1
    print(f"  {track}: BH {bh}/{len(grp)}  BHY {bhy}/{len(grp)}")
    sel = grp.loc[grp.get("bh_rejected", False) == True, ["feature", "ic_bar"]]
    if len(sel):
        sel = sel.reindex(sel["ic_bar"].abs().sort_values(ascending=False).index)
        print("    selected:", ", ".join(
            f"{r.feature}({r.ic_bar:+.3f})" for r in sel.head(14).itertuples()))
PY
say "logs in $LOG_DIR: shap_v3 fdr_v3 bhy_v3 placebo_v3 embargo_v3"
