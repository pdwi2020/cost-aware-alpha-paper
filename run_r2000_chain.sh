#!/bin/zsh
# V13 Phase C: full CAVAL re-instantiation on the Russell 2000.
# Runs the 5 stage scripts in order with the universe suffix active, writing
# *_r2000 artifacts. Stops on the first failure.
set -e
cd /Users/paritoshdwivedi/ML_Paper
export CAVAL_UNIVERSE=r2000
echo "=== [stage 1/5] walk-forward IC ===" && python3 -u src/models/run_walk_forward.py
echo "=== [stage 2/5] SHAP ==="            && python3 -u src/models/shap_analysis.py
echo "=== [stage 3/5] FDR (re-derived) ===" && python3 -u src/fdr/run_fdr.py
echo "=== [stage 4/5] signals ==="          && python3 -u src/backtest/generate_signals.py
echo "=== [stage 5/5] locked OOS holdout ===" && python3 -u src/backtest/run_holdout.py
echo "=== R2000 CHAIN COMPLETE ==="
