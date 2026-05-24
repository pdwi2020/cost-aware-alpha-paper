# Google Colab Setup — ML Paper

## Overview

Colab is a fallback option for the ML paper (CPU-only tasks). Primary platform is Kaggle.

## MCP Tools Available

The following colab-exec MCP tools are available in Claude sessions:

| Tool | Description |
|------|-------------|
| `colab_execute` | Run inline Python code on Colab (T4 GPU) |
| `colab_execute_file` | Run a local `.py` file on Colab |
| `colab_execute_notebook` | Run code and collect artifacts (plots, CSVs, models) |

## Usage Examples

```python
# Test basic execution
colab_execute("import pandas as pd; print(pd.__version__)")

# Run a specific script
colab_execute_file("src/models/model_suite.py")

# Long-running task (increase timeout)
colab_execute("exec(open('src/backtest/portfolio.py').read())", timeout=600)
```

## Data Access on Colab

Upload parquet files using base64 encoding:
```python
import base64, os

# Encode local file
with open("data/processed/features_all.parquet", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

# In Colab script, decode and save
code = f"""
import base64
data = "{b64}"
with open("/tmp/features.parquet", "wb") as f:
    f.write(base64.b64decode(data))
import pandas as pd
df = pd.read_parquet("/tmp/features.parquet")
print(df.shape)
"""
colab_execute(code)
```

## Download Results from Colab

```python
# In your Colab script, base64-encode output files:
code = """
import base64, pandas as pd

# Your computation
result = pd.DataFrame({"sharpe": [1.2, 0.8, 1.5]})
result.to_parquet("/tmp/result.parquet")

# Encode for download
with open("/tmp/result.parquet", "rb") as f:
    print("BASE64_FILE_START")
    print(base64.b64encode(f.read()).decode())
    print("BASE64_FILE_END")
"""

output = colab_execute(code)

# Parse and decode
import base64
b64_data = output.split("BASE64_FILE_START\n")[1].split("\nBASE64_FILE_END")[0]
with open("data/results/result.parquet", "wb") as f:
    f.write(base64.b64decode(b64_data))
```

## Install Packages on Colab

```python
colab_execute("""
import subprocess
subprocess.run(["pip", "install", "mapie", "shap", "lightgbm", "-q"])
import mapie
print(f"mapie version: {mapie.__version__}")
""")
```

## Quota Notes

- Free tier: T4 GPU (not needed for ML paper), usage limits apply
- Colab Pro: L4 GPU, more RAM, priority access (~$10/month)
- For ML paper: free tier is sufficient (CPU tasks)
