# ML Paper — 12-Week Implementation Plan

**Title:** Cost-Aware Alpha Generation with False Discovery Control and Market-Impact-Aware Validation  
**Target:** RFS / JFE / JFQA | QR internships at Jane Street, Citadel GQS, D.E. Shaw, Two Sigma, AQR

---

## Overview

| Phase | Weeks | Milestone |
|-------|-------|-----------|
| Data & Features | 1–3 | Clean dataset, Tier 1+2 features engineered, DuckDB catalog updated |
| Baseline Models | 4–5 | All 6 models (logistic → LightGBM) fitted on walk-forward splits |
| FDR / PBO | 6–7 | BH-corrected feature selection, CSCV PBO < 0.30 confirmed |
| Backtest | 8–9 | Almgren-Chriss execution, 2D cost surface, holdout performance |
| Calibration | 10 | Conformal intervals, regime-conditional sizing, ECE |
| Writing | 11–12 | Paper draft, tables/figures, SSRN preprint |

---

## Week-by-Week

### Week 1 — Data Audit & Pipeline

**Goal:** Reproducible data loading; confirm no lookahead leakage.

Tasks:
- [ ] Run `00_data_audit.ipynb`: verify ohlcv_1m_us parquet files (2010–2024), schema, NaN rates
- [ ] Confirm SP500 holdings snapshot aligns with price universe (S&P 500 as of each month)
- [ ] Load Fama-French 5 factors + momentum (AQR) for residualization
- [ ] Load FRED macro series: 2s10s slope, VIX (daily), USD DXY
- [ ] Establish DuckDB catalog view `ml_paper_universe` (symbol, date, adj_close, volume, sp500_member)
- [ ] Implement `src/data/universe_builder.py`: survivorship-bias-free rolling SP500 membership
- [ ] Unit test: no future data in train splits; all prices have valid adj close

Deliverable: `00_data_audit.ipynb` completed with summary statistics table.

---

### Week 2 — Tier 1 Feature Engineering

**Goal:** 25+ classic alpha features; zero lookahead.

Tasks:
- [ ] Implement `src/features/tier1_classic.py`:
  - Momentum: ret_1d, ret_5d, ret_21d, ret_63d, ret_252d; 12-1 momentum
  - Reversal: ret_1d sign-flip (1d reversal), 4-week reversal
  - Vol/Sharpe: rolling vol 21d, rolling Sharpe 21d, vol regime (calm/stressed)
  - Liquidity: Amihud illiquidity (daily abs_ret/dollar_volume), bid-ask proxy
  - Earnings: SUE score, PEAD signal (3-day post-earnings drift flag)
  - Macro: 2s10s slope change 1-month, VIX level, VIX change 1-week
- [ ] Lag all features by 1 bar (prevent lookahead into label period)
- [ ] Compute Track A targets: FF5+Mom residualized 5d forward returns
- [ ] Compute Track B targets: raw 5d forward open-to-open returns
- [ ] Save feature matrix to parquet: `data/processed/features_tier1.parquet`

Deliverable: `01_feature_engineering.ipynb` — Tier 1 feature summary + correlation heatmap.

---

### Week 3 — Tier 2 Feature Engineering

**Goal:** 15+ extended features (intraday microstructure + cross-asset + crowding).

Tasks:
- [ ] Implement `src/features/tier2_extended.py`:
  - Intraday: overnight gap return, VWAP deviation (1m bars), vol signature ratio (9:30–11am / 2–4pm vol), volume clock (fraction of daily volume in first hour)
  - Cross-asset: IG/HY credit spread change 1-week, USD DXY 1-week return, WTI × energy sector beta, 2s10s × equity momentum interaction
  - Crowding proxy: 20d rolling corr with SPY, QQQ, sector ETF (z-scored)
- [ ] Merge Tier 1 + Tier 2 into combined feature matrix
- [ ] Feature universe: ~40 signals total
- [ ] SHAP importance pre-screen: drop features with mean |SHAP| < 1e-4 across all models
- [ ] Save: `data/processed/features_all.parquet`

Deliverable: `01_feature_engineering.ipynb` — Tier 2 section + combined feature matrix shape.

---

### Week 4 — Walk-Forward Cross-Validation + Baseline Models

**Goal:** All 6 models fitted with correct temporal CV structure.

Tasks:
- [ ] Implement `src/models/model_suite.py`: Logistic, Ridge, Lasso (adaptive), RF, XGBoost, LightGBM
- [ ] Walk-forward expanding window: warmup 3 years (2010–2012), step 1 year, 9 test folds (2013–2021)
- [ ] Hyperparameter grid from `configs/models.yaml` (C, alpha, max_depth, etc.)
- [ ] For each fold × model: fit on train, predict on test, record IC (rank correlation with forward return)
- [ ] Track A and Track B run in parallel
- [ ] Save fold-by-fold IC time series: `data/processed/ic_by_fold.parquet`

Deliverable: `02_baseline_models.ipynb` — IC table by model and fold.

---

### Week 5 — SHAP Feature Attribution

**Goal:** Understand driver signals; prepare for FDR testing.

Tasks:
- [ ] Compute SHAP values for XGBoost + LightGBM on each test fold
- [ ] Aggregate: mean |SHAP| per feature across all folds
- [ ] Identify top-10 features per model; check cross-model overlap
- [ ] SHAP dependency plots for top 5 signals
- [ ] Partial dependence plots (ICE curves) for momentum/reversal features
- [ ] Implement `src/models/ensemble.py`: equal-weight IC-ensemble of XGB + LGBM + Logistic

Deliverable: `02_baseline_models.ipynb` — SHAP section + ensemble model results.

---

### Week 6 — False Discovery Rate Control

**Goal:** BH-corrected feature selection; FDR q ≤ 0.05.

Tasks:
- [ ] Implement `src/fdr/bh_correction.py`: BH procedure on mean IC t-statistics
- [ ] For each feature: t-stat from IC time series (T=9 folds); p-value (two-tailed t-test)
- [ ] Apply BH at q=0.05: identify significant features
- [ ] Separate by VIX regime (calm ≤20, stressed >20): all tests in single BH pool (not separate)
- [ ] Report: # of features surviving FDR, effective multiplicity
- [ ] Re-fit ensemble using only FDR-surviving features; compare IC before/after

Deliverable: `03_fdr_pbo_analysis.ipynb` — FDR table + feature count before/after.

---

### Week 7 — Probability of Backtest Overfitting

**Goal:** CSCV PBO < 0.30 on all model variants.

Tasks:
- [ ] Implement `src/fdr/pbo_cscv.py`: CSCV with S=16 combinatorial splits
- [ ] PBO metric: P(selected model is NOT best on OOS) across combinatorial train/test splits
- [ ] Compute Deflated Sharpe Ratio (DSR) for each model variant
- [ ] Bootstrap (block bootstrap, L=21 days, B=1000) for IC distribution confidence intervals
- [ ] Report: PBO table by model; target PBO < 0.30 for ensemble
- [ ] If PBO > 0.30: reduce model count or feature count (document decision)

Deliverable: `03_fdr_pbo_analysis.ipynb` — PBO/DSR table + bootstrap IC distribution.

---

### Week 8 — Backtest with Almgren-Chriss Execution

**Goal:** Realistic net-of-cost alpha; turnover-frequency surface.

Tasks:
- [ ] Implement `src/backtest/almgren_chriss.py`: square-root market impact model
  - `impact_cost = impact_coeff × σ × (order_size / ADV)^0.5`
  - spread cost per round-trip = `spread_bps / 10000`
  - borrow cost (short only): tiered by easy/medium/hard classification
- [ ] Implement `src/backtest/portfolio.py`: signal → position → execution → P&L
  - Implementation lag: signal at T close → execute at T+1 open
  - P&L window: T+1 open → T+2 open (open-to-open)
- [ ] Rebalancing frequency surface: daily, weekly (5d), monthly (21d)
- [ ] Turnover-penalty grid: λ ∈ {0, 0.001, 0.005, 0.010, 0.050}
- [ ] Save P&L series by frequency × lambda: `data/processed/backtest_results.parquet`

Deliverable: `04_execution_backtest.ipynb` — gross vs net Sharpe by rebalancing frequency.

---

### Week 9 — 2D Cost Sensitivity Analysis

**Goal:** Stress-test profitability across execution cost assumptions.

Tasks:
- [ ] 2D grid: spread_bps ∈ {1, 3, 5, 7, 10} × impact_coeff ∈ {0.05, 0.10, 0.15, 0.20}
- [ ] For each point: run full backtest (2010–2021 walk-forward), compute net Sharpe + annual return
- [ ] Plot: 5×4 heatmap of net Sharpe; overlay break-even contour (Sharpe = 0.5)
- [ ] Identify "profitable region" — parameter space where strategy remains viable
- [ ] Capacity curve: Sharpe vs AUM (vary impact_coeff ∝ sqrt(AUM/100M)); show AUM at Sharpe = 0.5

Deliverable: `04_execution_backtest.ipynb` — 2D cost surface heatmap + capacity curve.

---

### Week 10 — Calibration, Conformal Prediction, Regime Analysis

**Goal:** Calibrated uncertainty → conformal position sizing; regime-conditional P&L.

Tasks:
- [ ] Implement `src/calibration/conformal.py`: split conformal on calibration set (20% of walk-forward train)
- [ ] Conformal prediction intervals at α=0.10 for each signal
- [ ] Conformal-leverage position sizing:
  - raw_score = signal / interval_width
  - normalize to 1.0x gross exposure budget
  - apply 5% per-name cap, 2x gross hard cap
- [ ] ECE (Expected Calibration Error) with 10 bins: before/after conformal
- [ ] Regime analysis: P&L split by VIX calm (≤20) vs stressed (>20)
- [ ] True holdout evaluation: 2022–2024 (first and only touch)
- [ ] Polymarket robustness: re-run with polymarket_features added; compare IC Δ

Deliverable: `05_calibration_conformal.ipynb` + `06_regime_analysis.ipynb` + `07_final_results.ipynb`.

---

### Weeks 11–12 — Writing & Submission Prep

**Goal:** Submission-ready paper + SSRN preprint.

Tasks:
- [ ] Structure: Introduction → Literature → Data & Features → Methodology → Results → Conclusion
- [ ] Tables: feature IC table, model comparison (IC/Sharpe/PBO/DSR), cost surface, regime table, holdout
- [ ] Figures: SHAP bar chart, 2D cost heatmap, capacity curve, calibration plot (ECE), PnL time series
- [ ] Robustness appendix: Polymarket features, alternative feature sets
- [ ] Target journal: RFS (primary), JFE (secondary), JFQA (tertiary)
- [ ] SSRN preprint upload: include acknowledgments, data availability statement
- [ ] QR interview deck: 5 slides summarizing key results

---

## Key Milestones (Checkboxes)

- [ ] Week 3: Feature matrix complete (40+ signals, no leakage confirmed)
- [ ] Week 5: IC > 0.03 on at least 3 models (else revisit features)
- [ ] Week 7: PBO < 0.30, FDR survivors > 10 features
- [ ] Week 9: Profitable cost region confirmed at realistic assumptions (spread=3bps, impact=0.10)
- [ ] Week 10: True holdout IC > 0.02 (positive), Sharpe > 0.5 net of costs
- [ ] Week 12: First draft submitted to co-authors / advisor

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| FDR kills all features | Low | Expand Tier 2; relax q to 0.10 with disclosure |
| PBO > 0.30 | Medium | Reduce model count; use ensemble only |
| Net Sharpe < 0.5 after costs | Medium | Reduce rebalancing to weekly/monthly |
| Holdout fails | Low-Medium | Disclose honestly; paper = methodology, not alpha |
| OHLCV data gaps for pre-2012 | Low | Set universe start to 2012 if needed |
