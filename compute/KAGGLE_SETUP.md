# Kaggle Setup — ML Paper

## Overview

The ML paper uses Kaggle CPU notebooks (free tier). No GPU required — all models are
sklearn/XGBoost/LightGBM. This is the primary and only compute platform for this project.

## Prerequisites

1. Kaggle account at https://www.kaggle.com
2. API key: `https://www.kaggle.com/settings → Account → API → Create New Token`
3. Save to `~/.kaggle/kaggle.json` (chmod 600)

## Dataset Upload (One-Time Setup)

Upload feature data from X9 to a private Kaggle dataset:

```bash
# 1. Create upload directory
mkdir -p /tmp/ml_paper_upload

# 2. Export parquet features (from X9 via DuckDB)
python - <<'EOF'
import duckdb, os
db = duckdb.connect(os.environ["DUCKDB_CATALOG"])
db.execute("COPY (SELECT * FROM ml_paper_features) TO '/tmp/ml_paper_upload/features.parquet'")
db.execute("COPY (SELECT * FROM ff5_factors) TO '/tmp/ml_paper_upload/ff5_factors.parquet'")
print("Exported successfully")
EOF

# 3. Create dataset metadata
cat > /tmp/ml_paper_upload/dataset-metadata.json <<'EOF'
{
  "title": "ml-paper-features-private",
  "id": "paritoshdwi2019/ml-paper-features-private",
  "licenses": [{"name": "other"}]
}
EOF

# 4. Upload to Kaggle (private dataset)
kaggle datasets create -p /tmp/ml_paper_upload --dir-mode zip
```

## Running a Kaggle Notebook

1. Go to https://www.kaggle.com → Notebooks → New Notebook
2. Add dataset: search "ml-paper-features-private" (your private dataset)
3. Settings → Accelerator → **None** (CPU only)
4. Settings → Internet → On

Install packages at notebook start:
```python
!pip install duckdb xgboost lightgbm mapie shap -q
```

## Kaggle Notebook Template

```python
import os, sys

# Mount dataset
DATA_DIR = "/kaggle/input/ml-paper-features-private"

# Install extras
import subprocess
subprocess.run(["pip", "install", "mapie", "shap", "-q"])

import pandas as pd, numpy as np
import duckdb

# Load features
features = pd.read_parquet(f"{DATA_DIR}/features.parquet")
ff5 = pd.read_parquet(f"{DATA_DIR}/ff5_factors.parquet")

print(f"Features shape: {features.shape}")
print(f"Date range: {features.index.min()} → {features.index.max()}")
```

## Quota Management

- CPU sessions: 30h/week (no GPU needed → unlimited effectively)
- Session limit: 9h per session
- Output size: 20 GB limit (more than enough for parquet outputs)
- Save outputs to `/kaggle/working/` → download or commit to dataset

## Saving Results

```python
# Save checkpoint results
results_df.to_parquet("/kaggle/working/backtest_results.parquet")

# Commit to new dataset version (in notebook)
# Kaggle auto-saves /kaggle/working/ when notebook completes
```

## Local Workflow → Kaggle Handoff

For development: use local DuckDB catalog to build features → export parquet → upload.
For full walk-forward (slow): run on Kaggle for parallelism.

```bash
# Local: build features for one fold
python src/features/tier1_classic.py --fold 2013 --output data/processed/

# Kaggle: run all folds with walk-forward loop
# notebooks/02_baseline_models.ipynb handles fold loop
```

## colab-exec MCP Alternative

If Kaggle is slow, use colab-exec MCP tool:
```python
# Via Claude MCP tool: colab_execute_file
colab_execute_file("src/models/model_suite.py --fold 2013")
```
See `COLAB_SETUP.md` for details.
