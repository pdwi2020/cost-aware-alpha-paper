#!/usr/bin/env bash
# =============================================================================
# run_all.sh — One-command CAVAL pipeline orchestrator
#
# Compatible with bash 3+ (macOS system bash).
#
# Reads config/spec.yaml for universe/track/split config, then runs every
# pipeline stage in order, logging each run.
#
# OOS GATE: before the holdout/OOS stage, asserts that config/spec.yaml
# is a committed git blob. Aborts if the spec is dirty/uncommitted.
#
# Usage:
#   ./run_all.sh                         # full run (all universes, tracks A+B)
#   ./run_all.sh --only holdout          # run a single stage only
#   ./run_all.sh --from fdr              # resume from a stage (inclusive)
#   ./run_all.sh --strict                # treat PENDING-REBUILD stages as errors
#   ./run_all.sh --help
#
# Stage names (--only / --from):
#   build_features  build_features_all  screen0
#   walk_forward  fdr  bhy  ta_fdr  backtest  pbo  conformal  aci  knockoffs
#   holdout  ablation  sensitivity  oos_subperiods  oos_attribution
#   oos_cost_sweep  excl_megacap  regime  baselines  r2000  figures
# =============================================================================
set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"
LOG_DIR="$ROOT/results/logs"
SPEC="$ROOT/config/spec.yaml"

mkdir -p "$LOG_DIR"

# ── colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
log_ok()      { echo -e "${GREEN}[OK]${NC}   $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_pending() { echo -e "${YELLOW}[PENDING-REBUILD]${NC} $*"; }

# ── argument parsing ──────────────────────────────────────────────────────────
ONLY_STAGE=""
FROM_STAGE=""
STRICT=0

usage() {
  cat <<EOF

${BOLD}CAVAL Pipeline Orchestrator${NC}

Usage:
  $(basename "$0") [options]

Options:
  --only <stage>   Run only this one stage (skip all others)
  --from <stage>   Resume from this stage (run it + all subsequent stages)
  --strict         Treat PENDING-REBUILD stages as fatal errors (default: skip with warning)
  --help, -h       Show this help

Stage names (in pipeline order):
  build_features    build_features_all   screen0
  walk_forward      fdr                  bhy
  ta_fdr            backtest             pbo
  conformal         aci                  knockoffs
  holdout           ablation             sensitivity
  oos_subperiods    oos_attribution      oos_cost_sweep
  excl_megacap      regime               baselines
  r2000             figures

OOS Gate:
  The 'holdout' stage is gated by an OOS lock check. config/spec.yaml must
  exist as a committed git blob. If the spec is uncommitted or dirty, the
  pipeline aborts with a clear message before touching holdout data.

Log files:
  results/logs/<stage>.<universe>.<track>.log

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)   ONLY_STAGE="$2"; shift 2 ;;
    --from)   FROM_STAGE="$2"; shift 2 ;;
    --strict) STRICT=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) log_error "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# ── read spec.yaml with Python ────────────────────────────────────────────────
PRIMARY_UNIVERSE=$(python3 -c "
import yaml, pathlib, sys
try:
    s = yaml.safe_load(open('$ROOT/config/spec.yaml'))
    print(s['universes']['primary']['name'])
except Exception as e:
    print('pit_sp500_annual', file=sys.stderr)
    print('pit_sp500_annual')
")
ROBUSTNESS_UNIVERSE=$(python3 -c "
import yaml, pathlib, sys
try:
    s = yaml.safe_load(open('$ROOT/config/spec.yaml'))
    print(s['universes']['robustness']['name'])
except Exception as e:
    print('liquidity_topN', file=sys.stderr)
    print('liquidity_topN')
")

UNIVERSES=("$PRIMARY_UNIVERSE" "$ROBUSTNESS_UNIVERSE")
TRACKS=("A" "B")

log_info "Universes: ${UNIVERSES[*]}"
log_info "Tracks:    ${TRACKS[*]}"
log_info "Log dir:   $LOG_DIR"

# ── stage registry (bash 3-compatible: use a function + case) ─────────────────
# Ordered list of stage names — this is the canonical pipeline order
STAGE_ORDER=(
  build_features
  build_features_all
  screen0
  walk_forward
  fdr
  bhy
  ta_fdr
  backtest
  pbo
  conformal
  aci
  knockoffs
  holdout
  ablation
  sensitivity
  oos_subperiods
  oos_attribution
  oos_cost_sweep
  excl_megacap
  regime
  baselines
  r2000
  figures
)

# Returns the shell command for a stage on stdout.
# Prefix "ONCE:" means it runs once (not per-universe/track loop).
# Prefix "PENDING:" means non-zero exit is tolerated (PENDING-REBUILD).
stage_cmd() {
  local stage="$1"
  case "$stage" in
    build_features)     echo "ONCE:python3 -u src/features/build_features.py" ;;
    build_features_all) echo "ONCE:python3 -u src/features/build_features_all.py" ;;
    screen0)            echo "python3 -u src/data/universe_builder.py" ;;
    walk_forward)       echo "python3 -u src/models/run_walk_forward.py" ;;
    fdr)                echo "python3 -u src/fdr/run_fdr.py" ;;
    bhy)                echo "python3 -u src/fdr/run_bhy.py" ;;
    ta_fdr)             echo "python3 -u src/backtest/run_ta_fdr.py" ;;
    backtest)           echo "python3 -u src/backtest/run_backtest.py" ;;
    pbo)                echo "python3 -u src/fdr/run_pbo.py" ;;
    conformal)          echo "python3 -u src/calibration/run_conformal.py" ;;
    aci)                echo "python3 -u src/calibration/run_aci.py" ;;
    knockoffs)          echo "python3 -u src/knockoffs/run_cost_aware_knockoffs.py" ;;
    holdout)            echo "python3 -u src/backtest/run_holdout.py" ;;
    ablation)           echo "python3 -u src/backtest/run_ablation.py" ;;
    sensitivity)        echo "python3 -u src/backtest/run_sensitivity.py" ;;
    oos_subperiods)     echo "python3 -u src/backtest/run_oos_subperiods.py" ;;
    oos_attribution)    echo "python3 -u src/backtest/run_oos_attribution.py" ;;
    oos_cost_sweep)     echo "python3 -u src/backtest/run_oos_cost_sweep.py" ;;
    excl_megacap)       echo "python3 -u src/backtest/run_excl_megacap.py" ;;
    regime)             echo "python3 -u src/backtest/run_regime_signal.py" ;;
    baselines)          echo "python3 -u src/backtest/run_baselines.py" ;;
    r2000)              echo "ONCE:python3 -u src/backtest/run_r3b_cross_universe.py" ;;
    figures)            echo "ONCE:python3 -u src/figures/make_figures.py" ;;
    *)                  echo "" ;;
  esac
}

# ── OOS gate ──────────────────────────────────────────────────────────────────
check_oos_gate() {
  log_info "OOS GATE: verifying config/spec.yaml is a committed git blob..."
  local spec_hash
  spec_hash=$(git -C "$ROOT" hash-object "$SPEC" 2>/dev/null) || {
    log_error "OOS GATE FAILED: could not compute git hash of $SPEC"
    log_error "Ensure git is available and $ROOT is a git repository."
    exit 1
  }
  if git -C "$ROOT" cat-file -e "$spec_hash" 2>/dev/null; then
    log_ok "OOS GATE: spec.yaml blob $spec_hash is committed in git. Proceeding."
  else
    log_error "==========================================================="
    log_error "OOS GATE FAILED — ABORTING"
    log_error "==========================================================="
    log_error ""
    log_error "config/spec.yaml has git hash: $spec_hash"
    log_error "but that blob is NOT found in the git object store."
    log_error ""
    log_error "This means your spec.yaml has uncommitted local changes,"
    log_error "or was never committed. The locked OOS holdout requires a"
    log_error "committed pre-registration spec (single-touch discipline)."
    log_error ""
    log_error "Fix: commit config/spec.yaml before running the OOS holdout:"
    log_error "  git add config/spec.yaml && git commit -m 'lock pre-reg spec'"
    log_error "==========================================================="
    exit 1
  fi
}

# ── stage runner ──────────────────────────────────────────────────────────────
run_stage() {
  local stage="$1"
  local universe="${2:-all}"
  local track="${3:-all}"
  local log_file="$LOG_DIR/${stage}.${universe}.${track}.log"

  local raw_cmd
  raw_cmd=$(stage_cmd "$stage")

  if [[ -z "$raw_cmd" ]]; then
    log_warn "No command registered for stage '$stage' — skipping."
    return 0
  fi

  # Strip ONCE: prefix (already handled by caller)
  local cmd="${raw_cmd#ONCE:}"

  # Check for PENDING prefix
  local is_pending=0
  if [[ "$cmd" == PENDING:* ]]; then
    is_pending=1
    cmd="${cmd#PENDING:}"
  fi

  # Pre-stage OOS gate
  if [[ "$stage" == "holdout" ]]; then
    check_oos_gate
  fi

  local start_ts
  start_ts=$(date +%s)
  local start_iso
  start_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  {
    echo "================================================================"
    echo "[START] stage=$stage universe=$universe track=$track"
    echo "[TIME]  $start_iso"
    echo "[CMD]   $cmd"
    echo "================================================================"
  } | tee -a "$log_file"

  log_info "Running stage=${BOLD}${stage}${NC} | universe=${universe} | track=${track}"
  log_info "  cmd: $cmd"
  log_info "  log: $log_file"

  local exit_code=0
  (cd "$ROOT" && eval "$cmd") >> "$log_file" 2>&1 || exit_code=$?

  local end_ts
  end_ts=$(date +%s)
  local elapsed=$(( end_ts - start_ts ))

  {
    echo "================================================================"
    echo "[END]   stage=$stage exit_code=$exit_code elapsed=${elapsed}s"
    echo "================================================================"
  } | tee -a "$log_file"

  if [[ $exit_code -eq 0 ]]; then
    log_ok "stage=$stage | universe=$universe | track=$track | elapsed=${elapsed}s"
  elif [[ $is_pending -eq 1 ]]; then
    log_pending "stage=$stage exited with code $exit_code (stage is PENDING-REBUILD, non-fatal)"
    if [[ $STRICT -eq 1 ]]; then
      log_error "--strict mode: treating PENDING-REBUILD failure as fatal."
      exit 1
    fi
  else
    log_error "stage=$stage FAILED with exit_code=$exit_code"
    log_error "  See: $log_file"
    exit $exit_code
  fi
}

# ── stage selection helpers ───────────────────────────────────────────────────
stage_index() {
  local target="$1"
  local i=0
  for s in "${STAGE_ORDER[@]}"; do
    if [[ "$s" == "$target" ]]; then
      echo $i
      return 0
    fi
    i=$(( i + 1 ))
  done
  echo -1
}

validate_stage_name() {
  local name="$1"
  local idx
  idx=$(stage_index "$name")
  if [[ $idx -lt 0 ]]; then
    log_error "Unknown stage name: '$name'"
    log_error "Valid stages: ${STAGE_ORDER[*]}"
    exit 1
  fi
}

# ── validate CLI args ─────────────────────────────────────────────────────────
if [[ -n "$ONLY_STAGE" ]]; then
  validate_stage_name "$ONLY_STAGE"
fi
if [[ -n "$FROM_STAGE" ]]; then
  validate_stage_name "$FROM_STAGE"
fi
if [[ -n "$ONLY_STAGE" && -n "$FROM_STAGE" ]]; then
  log_error "--only and --from are mutually exclusive."
  exit 1
fi

# ── main pipeline loop ────────────────────────────────────────────────────────
FROM_IDX=0
if [[ -n "$FROM_STAGE" ]]; then
  FROM_IDX=$(stage_index "$FROM_STAGE")
fi

log_info "=========================================="
log_info "CAVAL Pipeline — starting"
log_info "  Only: ${ONLY_STAGE:-<all>}"
log_info "  From: ${FROM_STAGE:-<beginning>}"
log_info "  Strict: $STRICT"
log_info "=========================================="

STAGE_IDX=0

for stage in "${STAGE_ORDER[@]}"; do
  # -- stage selection logic
  if [[ -n "$ONLY_STAGE" && "$stage" != "$ONLY_STAGE" ]]; then
    STAGE_IDX=$(( STAGE_IDX + 1 ))
    continue
  fi
  if [[ -n "$FROM_STAGE" && $STAGE_IDX -lt $FROM_IDX ]]; then
    STAGE_IDX=$(( STAGE_IDX + 1 ))
    continue
  fi

  # Determine if stage runs once or per-universe/track
  raw_cmd=$(stage_cmd "$stage")
  if [[ "$raw_cmd" == ONCE:* ]]; then
    run_stage "$stage" "all" "all"
  else
    for universe in "${UNIVERSES[@]}"; do
      for track in "${TRACKS[@]}"; do
        run_stage "$stage" "$universe" "$track"
      done
    done
  fi

  STAGE_IDX=$(( STAGE_IDX + 1 ))
done

log_info "=========================================="
log_ok "CAVAL Pipeline — COMPLETE"
log_info "=========================================="
