# Cost-Aware Alpha Generation with FDR Control and Market-Impact-Aware Validation

**Paper:** `paper/main.tex` (IEEE Access format) · **Status:** Complete (Weeks 1–12) · **Universe:** S&P 500, 2010–2024

---

## Overview

This repository contains the full research pipeline for a quantitative finance paper that builds a statistically rigorous, execution-cost-aware equity alpha strategy. The key idea: most published factors fail when you apply honest multiple-testing correction *before* fitting any model, and the survivors collapse further once realistic trading costs are accounted for.

We show that a macro/sector signal (Track B) survives all three filters — FDR control, backtest overfitting analysis, and Almgren-Chriss cost modeling — and delivers **net Sharpe +0.50 on a true 2022–2024 holdout**, a period it had never touched during development.

### Key Results

| | Track A (Reversal/Microstructure) | Track B (Macro/Sector) |
|---|---|---|
| BH-significant features | 16 | 25 |
| PBO (CSCV, 126 paths) | 0.00 | 0.14 |
| IS gross Sharpe (2013–21) | +0.496 | -0.192 |
| IS net Sharpe (weekly, 3 bps) | -0.095 | -0.665 |
| **OOS gross Sharpe (2022–24)** | -0.346 | **+0.705** |
| **OOS net Sharpe (2022–24)** | -0.584 | **+0.501** |
| Conformal coverage (target 90%) | 88.8% | 89.1% |

**Main finding:** Short-term reversal (Track A) overfits to the 2013–2021 low-volatility regime. The macro/sector signal (Track B) completely reverses in the 2022–2024 rate-hike environment, emerging as genuinely predictive only out-of-sample.

---

## Pipeline Architecture

```
data/                          Raw OHLCV + factors + macro (X9 drive, not in repo)
│
src/data/universe_builder.py   Point-in-time S&P 500 membership (674 tickers)
src/features/                  38-feature library (Tier 1 classic + Tier 2 extended)
src/models/                    Walk-forward CV (9 folds, 2013–2021), 7 model variants
src/fdr/                       BH FDR correction (q=0.05) + PBO via CSCV C(9,4)=126
src/backtest/                  Almgren-Chriss cost model, PortfolioSimulator
src/calibration/               Split conformal prediction, per-fold calibration
src/figures/                   6 paper figures (PNG)
paper/                         IEEE Access LaTeX paper + compiled PDF
```

### Week-by-Week Pipeline

| Week | Module | Output |
|------|--------|--------|
| 1 | `src/data/universe_builder.py` | Point-in-time S&P 500 universe |
| 2 | `src/features/tier1_classic.py` | Momentum, reversal, vol, Amihud, Roll spread |
| 3 | `src/features/tier2_extended.py` | Micro-structure, macro (VIX, 2s10s, DXY), sector correlations |
| 4 | `src/models/run_walk_forward.py` | 9-fold IC table (Lasso, Ridge, OLS, LGBM, XGB, RF, ensemble) |
| 5 | `src/models/shap_analysis.py` | SHAP importance by fold, ensemble IC weights |
| 6 | `src/fdr/run_fdr.py` | BH-FDR correction — 25 Track B / 16 Track A features survive |
| 7 | `src/fdr/run_pbo.py` | PBO + Deflated Sharpe Ratio via CSCV |
| 8 | `src/backtest/run_backtest.py` | IS backtest (2013–21), base case + 2D sensitivity |
| 9 | `src/backtest/run_sensitivity.py` | 3D grid (spread × impact × rebal freq), Pareto frontier |
| 10 | `src/calibration/run_conformal.py` + `run_holdout.py` | Conformal calibration + **first OOS touch** (2022–24) |
| 11 | `src/figures/make_figures.py` | 6 paper figures |
| 12 | `paper/main.tex` | IEEE Access draft + compiled PDF |

---

## Repository Structure

```
.
├── configs/
│   ├── backtest.yaml       # spread, impact, AUM, ADV filter, sensitivity grid
│   ├── fdr.yaml            # BH q-threshold, fold definitions
│   └── models.yaml         # model hyperparameters, walk-forward params
├── figures/                # Final paper figures (PNG)
├── notebooks/              # 00–07 analysis notebooks (mirrors src/ pipeline)
├── paper/
│   ├── main.tex            # Full IEEE Access paper
│   ├── references.bib      # 20 BibTeX entries
│   └── main.pdf            # Compiled PDF
├── references/
│   └── REFERENCES.md       # Annotated reading list
├── src/
│   ├── backtest/
│   │   ├── almgren_chriss.py     # Square-root market impact model
│   │   ├── portfolio.py          # PortfolioSimulator (positions → P&L → metrics)
│   │   ├── generate_signals.py   # SHAP-weighted BH-significant composite signal
│   │   ├── run_backtest.py       # IS backtest + 2D sensitivity
│   │   ├── run_sensitivity.py    # 3D grid + Pareto/breakeven analysis
│   │   └── run_holdout.py        # OOS Exp 7 (2022–2024, first touch)
│   ├── calibration/
│   │   ├── conformal.py          # ConformalIntervals class (split CP)
│   │   └── run_conformal.py      # Per-fold calibration + regime coverage
│   ├── data/
│   │   └── universe_builder.py   # Point-in-time S&P 500 membership
│   ├── fdr/
│   │   ├── bh_correction.py      # Benjamini-Hochberg + BHY procedures
│   │   ├── pbo_cscv.py           # CSCV + Deflated Sharpe Ratio
│   │   ├── run_fdr.py            # Spearman IC → BH FDR → selected features
│   │   └── run_pbo.py            # 126-path PBO analysis
│   ├── features/
│   │   ├── tier1_classic.py      # Momentum, reversal, vol, Amihud, Roll
│   │   ├── tier2_extended.py     # Microstructure, macro, sector correlations
│   │   ├── build_features.py     # Tier 1 build script
│   │   └── build_features_all.py # Full 38-feature build (567 MB parquet)
│   ├── figures/
│   │   └── make_figures.py       # All 6 paper figures
│   └── models/
│       ├── model_suite.py        # 7 model wrappers (Lasso/Ridge/OLS/LGBM/XGB/RF/Ensemble)
│       ├── ensemble.py           # SHAP-IC weighted stacking
│       ├── run_walk_forward.py   # 9-fold expanding-window CV
│       └── shap_analysis.py      # Per-fold SHAP decomposition
├── environment.yml
└── requirements.txt
```

---

## Reproducing the Results

### Prerequisites

Data lives on an external drive (not in this repo). You need:
- `data/processed/features_all.parquet` — 38 features, 2010–2024 (567 MB)
- `data/processed/daily_ohlcv.parquet` — OHLCV, 2010–2024

```bash
pip install -r requirements.txt
```

### Run the pipeline in order

```bash
# Feature engineering (requires raw OHLCV + macro data)
python3 -u src/features/build_features_all.py

# Walk-forward model training (~86 min, 9 folds × 7 models)
python3 -u src/models/run_walk_forward.py

# SHAP importance analysis (~7 min)
python3 -u src/models/shap_analysis.py

# BH FDR correction
python3 -u src/fdr/run_fdr.py

# Probability of backtest overfitting (CSCV, 126 paths)
python3 -u src/fdr/run_pbo.py

# Generate composite signals
python3 -u src/backtest/generate_signals.py

# IS backtest (2013–2021)
python3 -u src/backtest/run_backtest.py

# 3D sensitivity grid (spread × impact × rebal_freq)
python3 -u src/backtest/run_sensitivity.py

# Conformal calibration + OOS holdout (2022–2024)
python3 -u src/calibration/run_conformal.py
python3 -u src/backtest/run_holdout.py

# Paper figures
python3 -u src/figures/make_figures.py
```

### Compile the paper

```bash
cd paper
latexmk -pdf main.tex
```

Requires TeX Live 2024+ with `IEEEtran` and `natbib` (standard in any full TL install).

---

## Design Decisions

**Why two tracks?** Track A (short-term reversal + microstructure) and Track B (macro/sector) use different *targets* — idiosyncratic returns vs. raw returns — intentionally separating the signal generation from the noise. They are never merged into one composite.

**Why FDR before model fitting?** Running BH correction on Spearman IC (rank-invariant, no distributional assumptions) uses *none of the training data* for model selection. This is the only way to avoid the "multiple comparison problem" that inflates reported Sharpe ratios in factor zoo papers.

**Why Almgren-Chriss?** The square-root model is the industry standard for permanent price impact. With $100M AUM, even a 3 bps half-spread and η=0.10 impact coefficient wipes out Track A's gross alpha at daily rebalancing (1,182 bps/yr drag vs. ~500 bps gross). **Weekly rebalancing is the Pareto-optimal choice**: reduces turnover from 104× to 36× while retaining 98.8% of gross IC.

**Why conformal prediction?** Unlike Bayesian credible intervals or bootstrap bands, split conformal prediction provides a *finite-sample coverage guarantee* (≥ 1−α) under exchangeability — no distributional assumptions. The 2020 COVID crash breaks exchangeability (coverage drops to 73–77%), which is itself an important finding about stress-regime limitations.

**The bankrupt stock problem:** SIVBQ (SVB), SBNY (Signature Bank), FRCB (First Republic) have near-zero ADV and extreme realized volatility post-failure, generating thousands of bps of spurious impact cost in a single day. The `min_adv_dollars=$1M` filter in `PortfolioSimulator.simulate_pnl` is required for a clean 2022–2024 backtest.

---

## Cost Model Parameters (Base Case)

| Parameter | Value | Notes |
|-----------|-------|-------|
| AUM | $100M | Reference portfolio size |
| Half-spread | 3 bps | Liquid large-cap assumption |
| Impact coefficient η | 0.10 | Almgren-Chriss square-root model |
| Rebalancing frequency | Weekly (5d) | Pareto-optimal for Track A |
| Min ADV filter | $1M | Excludes bankrupt/delisted stocks |
| Gross exposure target | 1.0× | Dollar-neutral long/short |
| Max single position | 5% | Per-stock weight cap |

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{dwivedi2025costaware,
  author  = {Dwivedi, Paritosh},
  title   = {Cost-Aware Alpha Generation with False Discovery Rate Control
             and Market-Impact-Aware Validation},
  journal = {IEEE Access},
  year    = {2025},
  note    = {Preprint}
}
```

---

## License

Private repository — all rights reserved. Contact the author before reuse.
