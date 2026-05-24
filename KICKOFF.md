# ML Paper — Session Kickoff Guide

> **Read this first every session. This file is self-contained — no prior context needed.**

---

## Session Startup Checklist (do these in order)

1. **Verify X9 is mounted**: `ls "/Volumes/Crucial X9/data/market_data/equities/ohlcv_1m_us/" | head -3`
2. **Activate environment**: `conda activate ml_paper` (or `pip install -r requirements.txt` first time)
3. **Check DuckDB catalog**: `python3 -c "import duckdb; db=duckdb.connect('/Volumes/Crucial X9/data/catalog.duckdb',read_only=True); print(db.execute('SHOW TABLES').fetchdf())"`
4. **Open last notebook**: check `notebooks/` for where you left off
5. **Review experiment checklist** below — mark off what's done

---

## Paper Identity

**Full Title:** Cost-Aware Alpha Generation with False Discovery Control and Market-Impact-Aware Validation

**Research Question:** Which machine-learned equity signals survive the joint filter of multiple-testing correction, transaction costs, and market impact — and how do calibrated probability estimates improve position sizing under uncertainty?

**Unified Narrative Thread:**
> "I study reliable learning and decision-making under noisy, shifted, high-stakes environments: alpha discovery in markets, order-book prediction in microstructure, and calibrated infrared detection in machine vision."

### Abstract
Machine-learned alpha signals routinely appear profitable in backtests but fail in production due to three compounding problems: spurious discovery from testing many predictors simultaneously, ignored transaction costs and market impact, and miscalibrated probability estimates that lead to poor position sizing. This paper proposes a unified framework for execution-aware alpha generation that addresses all three. Using a survivorship-bias-controlled S&P 500 universe (2010–2024, true holdout 2022–2024) with US equity daily OHLCV data, we construct a two-tier feature library: Tier 1 comprises conventional momentum, reversal, volatility, liquidity, and macro-regime signals; Tier 2 adds less-commoditized microstructure, cross-asset lead-lag, and factor crowding signals derivable from 1-minute data. We run parallel tracks with both residualized (idiosyncratic) and raw forward return targets. False discovery is controlled via Benjamini-Hochberg correction, deflated Sharpe ratio, and PBO, with all regime-conditional tests in the same correction family. Execution realism is enforced through a square-root Almgren-Chriss impact model with borrow cost overlay and a 2D sensitivity surface. Conformal prediction intervals determine relative position sizing within a fixed gross exposure budget.

---

## Target Journals (ranked)

| Rank | Journal | IF | Why |
|---|---|---|---|
| 1 | Review of Financial Studies (RFS) | ~8 | Top empirical asset pricing; FDR + ML papers published here |
| 2 | Journal of Financial Economics (JFE) | ~9 | If microstructure angle is secondary |
| 3 | Journal of Portfolio Management (JPM) | ~2 | Applied quant; faster turnaround |
| 4 | Journal of Empirical Finance (JEF) | ~3 | Strong methods focus; accessible |

**Submission target:** Q1 2027
- Experiments complete: October 2026
- First draft: November 2026
- Revisions: December 2026
- Submission: January 2027

---

## QR Firm Targeting

**Primary targets:** Citadel GQS, D.E. Shaw, Two Sigma, AQR, Point72, Millennium, Schonfeld
**Secondary:** Systematic equity desks at any multi-strat (Balyasny, ExodusPoint, Eisler)
**PhD programs:** Columbia APAM, Princeton ORFE, NYU Courant (OR/Statistics track)

### Interview Talking Points
- "I combined PBO and Benjamini-Hochberg FDR control into a single signal validation pipeline — most ML backtests skip both."
- "The feature library is split into classic (expected to fail) and microstructure tiers — the survival rate difference is itself a publishable finding."
- "I use conformal prediction intervals to size positions — tighter interval = larger bet — within a fixed gross exposure budget so leverage constraints never bind."
- "The 2D cost sensitivity surface shows my net Sharpe is robust across realistic spread and impact assumptions, not just a point estimate."
- "True holdout: 2022–2024 never touched until final evaluation — walk-forward on 2010–2021 only."

---

## Datasets — Exact Paths on X9

All data is on `/Volumes/Crucial X9/`. Use DuckDB catalog for querying.

| Dataset | X9 Path | Size | DuckDB View |
|---|---|---|---|
| US equity OHLCV 1m | `data/market_data/equities/ohlcv_1m_us/` | 79 GB | `equities_ohlcv_1m` |
| FF5 + Momentum factors | `data/market_data/equities/aqr_factors/famafrench_ff5_daily.parquet` | 3.3 MB | `ff_famafrench_ff5_daily` |
| Momentum factor | `data/market_data/equities/aqr_factors/famafrench_mom_daily.parquet` | 1.1 MB | — |
| S&P 500 holdings | `data/market_data/equities/kaggle/s-and-p-500-holdings-and-weights-spy-2000-2024/` | 13 MB | — |
| Polymarket features | `data/market_data/polymarket_research_engine/features/market_features.parquet` | 57.6 MB | `poly_market_features` |
| FRED macro (VIX, yield) | `data/market_data/macro/fred/fred_macro.parquet` | 6.3 MB | `fred_macro` |

```python
# DuckDB connection
import duckdb
db = duckdb.connect('/Volumes/Crucial X9/data/catalog.duckdb', read_only=True)
df = db.execute("SELECT * FROM equities_ohlcv_1m LIMIT 5").fetchdf()
```

**Data preparation note:** Resample 1-min OHLCV to daily. Apply S&P 500 point-in-time membership filter (survivorship-bias control). Residualize returns against FF5+Mom for Track A; keep raw for Track B.

---

## Feature Library

### Tier 1 — Classic (conventional; expected low survival rate)
| Group | Features |
|---|---|
| Price momentum | 1m, 3m, 6m, 12m return |
| Short-term reversal | 1-week return |
| Volatility | 21-day realized vol, EWMA(0.94), GARCH(1,1) |
| Liquidity | Amihud ratio, bid-ask spread proxy, RVOL |
| Earnings | SUE (standardized unexpected earnings), earnings momentum |
| Sector | GICS sector dummies (11 sectors) |
| Macro regime | VIX level, 2s10s yield slope |
| Factor exposures | FF5 betas (rolling 12m OLS) |

### Tier 2 — Extended (microstructure + cross-asset + crowding)
| Group | Features | Source |
|---|---|---|
| Intraday microstructure | Vol 9:30–11am vs 2–4pm; overnight gap return; VWAP deviation; volume clock (ADV% by 11am) | 1-min OHLCV |
| Cross-asset lead-lag | IG/HY credit spread change; USD DXY 5d return; WTI oil 1m × energy sector; 2s10s × equity momentum | FRED parquet |
| Factor crowding proxy | 20d rolling correlation with SPY, QQQ, sector ETF | Daily OHLCV |

### Return Target Tracks
- **Track A:** FF5+Mom residualized forward returns (idiosyncratic alpha)
- **Track B:** Raw forward returns (factor-exposed signals)
- Horizons: 1-day, **5-day (primary)**, 21-day

**Implementation lag:** Signal at market close on day T → executed at next-day open T+1. P&L = open-to-open (T+1 to T+2).

---

## Experiment Checklist

- [ ] **Exp 1a** — Tier 1 feature survival: all models, Track A vs Track B, BH correction
- [ ] **Exp 1b** — Tier 1+2 feature survival: compare microstructure / cross-asset / crowding survival rates
- [ ] **Exp 2a** — SHAP beeswarm: top 20 surviving features (Tier 1 vs Tier 2 colour-coded)
- [ ] **Exp 2b** — Ensemble: equal-weight ensemble of FDR-surviving signals vs single-best vs XGBoost
- [ ] **Exp 3a** — Turnover-frequency-Sharpe surface: 3×2 grid (1d/5d/21d × quintile/z-score)
- [ ] **Exp 3b** — Cost sensitivity surface: 2D heatmap (spread_bps × impact_coeff)
- [ ] **Exp 4**  — Calibration benchmark: ECE + Brier + NLL for raw vs Platt vs isotonic vs conformal
- [ ] **Exp 5**  — Regime-conditional alpha: calm (VIX≤20) vs stressed (VIX>20), within BH pool
- [ ] **Exp 6**  — Cross-domain calibration robustness: Polymarket (standalone, not equity pipeline)
- [ ] **Exp 7**  — True holdout evaluation: 2022–2024 never-touch final report
- [ ] **Exp 8**  — Capacity curve: net Sharpe vs AUM ($10M → $1B) using impact model

---

## Implementation Roadmap (12 Weeks)

| Week | Task | Notebook |
|---|---|---|
| 1 | Data audit: verify X9 paths, resample OHLCV to daily, survivorship filter | 00_data_audit |
| 2 | Tier 1 feature engineering + factor residualization (Track A) | 01_feature_engineering |
| 3 | Tier 2 feature engineering (microstructure from 1-min, cross-asset from FRED) | 01_feature_engineering |
| 4 | Baseline models (Logistic, Ridge, Lasso, RF) — Exp 1a Tier 1 | 02_baseline_models |
| 5 | XGBoost/LightGBM + walk-forward CV — Exp 1a+1b complete | 02_baseline_models |
| 6 | BH correction, PBO/CSCV, DSR — Exp 1a/1b FDR analysis | 03_fdr_pbo_analysis |
| 7 | Execution backtest: Almgren-Chriss, borrow costs, 3×2 grid — Exp 3a, 3b | 04_execution_backtest |
| 8 | Conformal position sizing, Platt/isotonic calibration — Exp 4 | 05_calibration_conformal |
| 9 | Regime analysis (2 VIX states, within BH pool) — Exp 5 | 06_regime_analysis |
| 10 | Ensemble experiment, capacity curve — Exp 2b, 8 | 06_regime_analysis |
| 11 | Polymarket calibration robustness, true holdout — Exp 6, 7 | 07_final_results |
| 12 | All figures, tables, paper writing begins | 07_final_results |

---

## Key References

| # | Authors | Year | Title | Venue | Link |
|---|---|---|---|---|---|
| 1 | Harvey, Liu, Zhu | 2016 | …and the Cross-Section of Expected Returns | RFS 29(1) | https://doi.org/10.1093/rfs/hhv059 |
| 2 | Bailey et al. | 2015 | The Probability of Backtest Overfitting | JCF 20(4) | https://doi.org/10.21314/JCF.2015.316 |
| 3 | López de Prado | 2018 | Advances in Financial Machine Learning | Wiley | ISBN:9781119482086 |
| 4 | Gu, Kelly, Xiu | 2020 | Empirical Asset Pricing via Machine Learning | RFS 33(5) | https://doi.org/10.1093/rfs/hhaa009 |
| 5 | Almgren, Chriss | 2001 | Optimal Execution of Portfolio Transactions | JRisk 3(2) | https://doi.org/10.21314/JOR.2001.041 |
| 6 | Frazzini et al. | 2015 | Trading Costs of Asset Pricing Anomalies | arXiv | https://arxiv.org/abs/1506.05075 |
| 7 | Guo et al. | 2017 | On Calibration of Modern Neural Networks | ICML | https://arxiv.org/abs/1706.04599 |
| 8 | Angelopoulos, Bates | 2022 | A Gentle Introduction to Conformal Prediction | arXiv | https://arxiv.org/abs/2107.07511 |
| 9 | Benjamini, Hochberg | 1995 | Controlling the False Discovery Rate | JRSS-B 57(1) | https://doi.org/10.1111/j.2517-6161.1995.tb02031.x |
| 10 | López de Prado, Bailey | 2014 | The Deflated Sharpe Ratio | JPM 40(5) | https://doi.org/10.3905/jpm.2014.40.5.094 |

**Full proposal:** `../paper_proposals/ml_paper_proposal.md` (or `/Volumes/Crucial X9/Research Projects/paper_proposals/ml_paper_proposal.md` on X9)

---

## Compute Setup

**Platform:** Kaggle free CPU (primary) — no GPU needed for this paper.

### Kaggle Setup
```bash
# Install Kaggle CLI
pip install kaggle

# Place API key at ~/.kaggle/kaggle.json
# Get from: https://www.kaggle.com/settings → API → Create New Token
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Upload processed feature parquet files to Kaggle (once)
# After feature engineering locally, upload ~500 MB parquet to Kaggle dataset
kaggle datasets create -p ./kaggle_upload/ --dir-mode zip
```

### Running on Kaggle
1. Create new notebook → Settings → Accelerator: None (CPU) → Internet: On
2. Install packages: `!pip install xgboost lightgbm mapie shap duckdb pyarrow`
3. Upload feature parquets as a Kaggle dataset, attach to notebook
4. Quota: unlimited CPU runtime

### MCP Tools Available (when running Claude Code locally)
```python
# colab-exec MCP — run Python on Colab GPU if needed
# (not needed for this paper — CPU is sufficient)
# colab_execute("import sklearn; print(sklearn.__version__)")

# DuckDB catalog (local X9 access)
import duckdb
db = duckdb.connect('/Volumes/Crucial X9/data/catalog.duckdb', read_only=True)
```

### Local Environment Setup
```bash
conda create -n ml_paper python=3.11
conda activate ml_paper
pip install -r requirements.txt
```

---

## File Structure Guide

```
ml_paper/
├── KICKOFF.md          ← this file; read first every session
├── PLAN.md             ← week-by-week roadmap (more detail)
├── requirements.txt    ← pip install -r requirements.txt
├── .env.example        ← copy to .env, fill in paths
├── data/               ← symlinks to X9 datasets
├── notebooks/          ← work here; run in order (00 → 07)
│   ├── 00_data_audit.ipynb
│   ├── 01_feature_engineering.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_fdr_pbo_analysis.ipynb
│   ├── 04_execution_backtest.ipynb
│   ├── 05_calibration_conformal.ipynb
│   ├── 06_regime_analysis.ipynb
│   └── 07_final_results.ipynb
├── src/                ← importable modules (import from here)
│   ├── features/       ← tier1_classic.py, tier2_extended.py
│   ├── models/         ← model_suite.py (all 8 models)
│   ├── fdr/            ← bh_correction.py, pbo_cscv.py
│   ├── backtest/       ← almgren_chriss.py, portfolio.py
│   └── calibration/    ← conformal.py (Platt, isotonic, split conformal)
├── configs/            ← YAML files for hyperparameters
│   ├── models.yaml
│   ├── backtest.yaml   ← cost model params, borrow tiers
│   └── fdr.yaml        ← BH q, PBO splits, holdout dates
├── references/
│   └── REFERENCES.md   ← 10 key papers with DOIs
└── compute/
    ├── KAGGLE_SETUP.md
    └── COLAB_SETUP.md
```

---

## Critical Config Values (from configs/)

```yaml
# backtest.yaml
spread_bps: 3
impact_coeff: 0.10
borrow_cost_easy_bps: 0
borrow_cost_medium_bps: 150
borrow_cost_hard_bps: 300
gross_exposure_target: 1.0
max_single_position: 0.05
implementation_lag_days: 1
true_holdout_start: "2022-01-01"
vix_calm_threshold: 20
```
