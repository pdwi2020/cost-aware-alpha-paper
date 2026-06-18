# CAVAL: A Cost-Aware Validation System for Tradable-Alpha Discovery with FDR Control

**Target journal:** Expert Systems with Applications (Elsevier) — under review  
**Universe:** S&P 500, 2010–2024 (survivorship-bias-free, 9-fold expanding-window walk-forward)  
**Code status:** Complete (five screens + Russell 2000 re-instantiation, V13)

> This repository contains the **code** for the CAVAL pipeline.  
> The manuscript (`paper/main.tex`) is maintained separately and is **not included here**.

---

## Overview

CAVAL is a modular expert-system pipeline that admits an equity long-short signal only if it clears a data-integrity pre-screen (Screen 0) plus four sequential validation screens. The central thesis is that the deliverable is the **validation protocol**, not any single Sharpe ratio: a signal that naively looks strong in-sample should be systematically stress-tested for data artifacts, multiple-testing inflation, cost drag, cost robustness, and out-of-sample integrity before any deployment claim.

The pipeline operates on two independent alpha channels — Track A (idiosyncratic reversal) and Track B (macro/sector rotation) — which are never merged.

---

## The Five Screens

### Screen 0 — Data Integrity / Tradeable-Universe Pre-Screen

Applied uniformly to all artifacts before any statistical analysis:

- **±50 % daily-return winsorization** — removes corporate-action prints, delisting artifacts, and overnight-gap spikes that masquerade as alpha (a near-false-discovery that the statistical screens alone cannot detect).
- **Price ≥ $5 floor** — excludes penny stocks.
- **$1 M ADV filter** — excludes bankrupt/delisted names (e.g. SIVBQ, SBNY post-failure) that generate thousands of bps of spurious impact cost in a single day.

Screen 0 is the motivating case study: before it was applied, reversal OOS net Sharpe appeared as +0.53; after cleaning, the correct value is −0.22 — the entire apparent edge was a single +445% penny artifact on 2024-12-26.

### Screen 1 — Statistically Real (FDR Control)

Benjamini–Hochberg correction (q = 0.10) on per-feature Spearman information coefficients computed over 9-fold expanding-window walk-forward.  
- Run *before* any model fitting, using rank-invariant IC (no distributional assumptions).  
- S&P: 19/38 features survive per track (distinct sets for A and B).  
- R2000 re-derived independently: 19/32 features per track, distinct sets from S&P.

### Screen 2 — Economically Tradable (TA-FDR)

A novel **Tradable-Alpha FDR** test. The null distribution is calibrated by block-permutation of the *net-of-Almgren-Chriss* P&L at $100 M AUM — not the raw IC.  
This screen establishes that the surviving alpha is a **diversified-ensemble effect**: no single BH-significant feature is tradable at $100 M on its own.

### Screen 3 — Cost-Robust (Alpha Robustness Volume, ARV)

Track B must retain positive net Sharpe across a 3-D grid of 60 cost configurations (half-spread × impact η × rebalancing frequency).  
**Result (Screen 0 data):** ARV(SR > 0) = 83.3 % (50/60 configurations), ARV(SR > 0.25) = 75.0 %, median net SR = +0.439.

### Screen 4 — Locked Out-of-Sample Validation (2022–2024, single touch)

A single, locked OOS window (2022–2024) touched once:
- **Probability of Backtest Overfitting** via CSCV (Combinatorially Symmetric Cross-Validation).
- **Deflated Sharpe Ratio** (DSR) guard.
- **Split-conformal coverage** (finite-sample guarantee under exchangeability).
- **Embedded square-root market-impact model** (Almgren–Chriss) at $100 M AUM, 3 bps half-spread, η = 0.10.

---

## Two Alpha Channels

| | Track A — Idiosyncratic Reversal | Track B — Macro/Sector |
|---|---|---|
| Target | FF5 + UMD-residualized 5-day forward return | Raw 5-day forward return |
| Signal character | Short-term reversal, microstructure | Macro regime, sector rotation |
| Feature horizon | 5–21 d | 21–252 d, VIX, 2s10s, sector ETF correlations |

The two tracks are **never merged**. They use different regression targets and different BH-selected feature sets.

---

## Headline Results (S&P 500)

All numbers computed on Screen-0-cleaned data with pre-specified parameters: half-spread 3 bps, η = 0.10, AUM $100 M, 5-day rebalance, FDR q = 0.10. OOS = 2022–2024 (single touch, locked).

### Baselines vs. CAVAL Tracks

| Strategy | IS gross Sharpe | IS net Sharpe | IS cost drag (bps/yr) | OOS gross Sharpe | OOS net Sharpe | OOS cost drag (bps/yr) |
|---|---|---|---|---|---|---|
| Buy-and-Hold SPY | +1.00 | +1.00 | 0 | +0.58 | +0.58 | 0 |
| Momentum L/S (12-1) | +0.17 | +0.13 | 42 | **+0.93** | **+0.88** | 43 |
| Reversal L/S (5d) | +0.56 | +0.12 | 388 | +0.25 | −0.22 | 356 |
| Naive ML (RF+XGB, no FDR/SHAP) | +0.67 | +0.39 | 197 | −0.05 | −0.25 | 128 |
| **Track A (ours)** | −0.05 | −0.13 | 76 | −0.52 | −0.59 | 59 |
| **Track B (ours)** | **+0.64** | **+0.51** | 128 | **+0.47** | **+0.357** | 111 |

**Honest disclosure:**

- **Track A** is strong on IS IC in walk-forward but *fails* the locked OOS (net −0.59). CAVAL correctly rejects it.
- **Track B** is positive and cost-robust in both periods (+0.51 IS → +0.357 OOS). However, over the 3-year OOS window it is **statistically indistinguishable from zero**: Lo (2002) SE = 0.58, 95 % CI [−0.78, +1.49], p = 0.54; stationary-bootstrap 95 % CI [−0.62, +1.29], P(SR > 0) = 0.77.
- **Momentum beats Track B OOS** (net +0.88 vs. +0.357). This is disclosed. CAVAL's value is disciplined multi-feature validation and regime/cost robustness — not raw-Sharpe dominance. Note that momentum's IS net Sharpe is only +0.13, a single-factor regime fragility that multi-feature validation guards against.
- **Naive ML overfits**: without FDR+SHAP, a naive RF+XGB ensemble achieves +0.39 IS net but collapses to −0.25 OOS. FDR+SHAP converts it to Track B +0.357.

### Track B OOS Subperiods

| Year | Net Sharpe |
|---|---|
| 2022 | +1.12 |
| 2023 | +0.81 |
| 2024 | −1.04 |
| **Full 2022–2024** | **+0.357** |

### Ablation (Track B OOS, Screen 0)

| Variant | n_features | Gross Sharpe | Net Sharpe |
|---|---|---|---|
| Full CAVAL (FDR + SHAP) | 25 | +0.471 | **+0.357** |
| Remove FDR (all 35 features) | 35 | +0.067 | −0.080 |
| Remove SHAP (uniform weights) | 25 | +0.138 | +0.101 |

Both screening stages add measurable value.

---

## Russell 2000 Re-Instantiation (V13, 2026-06-18)

The entire CAVAL pipeline was re-run from scratch on US small-caps to demonstrate that the protocol re-instantiates mechanically on a wholly different universe — and correctly rejects a small-cap signal.

**Universe:** 986 survivors tracking the Russell 2000 (Vanguard VTWO constituents); daily OHLCV via yfinance. All universe-agnostic inputs (FF5+UMD, FRED macro, sector ETFs) re-sourced from public providers (Ken French / FRED CSV / yfinance). 3 intraday features unavailable on daily data → 35 features (not 38).

**Screen-0 finding on small-caps:** raw `overnight_gap` reached 2.4 × 10⁸ and `roll_spread` 1.0 × 10⁸ (vs. S&P maxima of ~3,200 / ~10,000). One uncapped artifact collapsed both tracks onto a degenerate identical signal (cross-track P&L correlation = 1.00, net SR ≈ 0). Fix: extend Screen-0 winsorization to overnight_gap from raw close (±50%), roll_spread clipped [0, 1], amihud winsorized at p99.9.

**Re-derived BH (q = 0.10):** 19/32 features per track, distinct sets:
- Track A: ret_252d, ret_63d, mom_12_1, ret_5d, reversals, sharpe_21d (momentum/reversal character)
- Track B: vix, corr_XLI, corr_XLF, corr_SPY, corr_XLY, corr_XLK, ret_21d (macro/sector character)

**Locked OOS 2022–2024 results (Screen 0, base costs, $100 M AUM):**

| Track | IS ensemble IC | Gross SR | Net SR | Net ann. | Max DD | Hit rate | Turnover | Cost drag |
|---|---|---|---|---|---|---|---|---|
| A (idio) | +0.133 | −0.648 | **−0.847** | −5.7 % | −20.3 % | 46.0 % | 20.0×/yr | 135 bps/yr |
| B (sector) | +0.035 | −0.145 | **−0.574** | −2.9 % | −9.6 % | 46.9 % | 31.6×/yr | 216 bps/yr |

**Honest reading:** the protocol mechanically re-instantiates and correctly **rejects** a small-cap signal that looked strong in-sample. Track A has strong IS IC (+0.133) but fails the locked OOS — a textbook IS→OOS collapse that CAVAL catches. Consistent with cross-universe evidence that the surviving Track B alpha is S&P-mid-cap-specific.

---

## Repository / Module Map

```
src/
├── universe_paths.py              # CAVAL_UNIVERSE env shim (see below)
│
├── data/
│   ├── universe_builder.py        # Point-in-time S&P 500 membership (674 tickers)
│   └── build_r2000_universe.py    # Russell 2000 universe via yfinance daily OHLCV
│
├── features/
│   ├── tier1_classic.py           # Momentum, reversal, volatility, Amihud, Roll spread
│   ├── tier2_extended.py          # Microstructure, macro (VIX, 2s10s, DXY), sector ETF correlations
│   ├── build_features.py          # Tier 1 build script
│   ├── build_features_all.py      # Full 38-feature build (S&P)
│   └── build_features_r2000.py    # 35-feature build for Russell 2000
│
├── models/
│   ├── model_suite.py             # 7 model wrappers (Lasso/Ridge/OLS/LGBM/XGB/RF/Ensemble)
│   ├── ensemble.py                # SHAP-IC weighted stacking
│   ├── run_walk_forward.py        # 9-fold expanding-window CV (CAVAL_UNIVERSE-aware)
│   └── shap_analysis.py           # Per-fold SHAP decomposition (CAVAL_UNIVERSE-aware)
│
├── fdr/
│   ├── bh_correction.py           # Benjamini–Hochberg + BHY procedures
│   ├── pbo_cscv.py                # CSCV + Deflated Sharpe Ratio
│   ├── run_fdr.py                 # Screen 1: Spearman IC → BH FDR → selected features
│   ├── run_pbo.py                 # Screen 4 PBO: 126-path CSCV analysis
│   ├── run_fdr_embargoed.py       # FDR with embargo gaps (robustness check)
│   ├── run_fdr_placebo.py         # Placebo FDR (shuffled targets, should reject all)
│   ├── run_alpha_investing.py     # Online alpha-investing sequential FDR
│   ├── run_bhy.py                 # BHY (negative-dependence) FDR
│   └── pbo_objective_check.py     # CSCV objective diagnostic
│
├── backtest/
│   ├── almgren_chriss.py          # Square-root market-impact model
│   ├── portfolio.py               # PortfolioSimulator: positions → P&L → metrics
│   ├── generate_signals.py        # SHAP-weighted BH-significant composite signal
│   ├── run_backtest.py            # IS backtest + 2D sensitivity (Screen 0 wired)
│   ├── run_sensitivity.py         # Screen 3 ARV: 3D cost grid (60 configs)
│   ├── run_holdout.py             # Screen 4: locked OOS 2022–2024 (single touch)
│   ├── run_ta_fdr.py              # Screen 2: Tradable-Alpha FDR (block-permutation null)
│   ├── run_robustness_volume.py   # ARV computation (83.3 % of 60 configs SR > 0)
│   ├── run_oos_attribution.py     # LOGO marginal attribution (Liq/Vol +0.45 marginal)
│   ├── run_oos_cost_sweep.py      # OOS cost sensitivity sweep
│   ├── run_oos_subperiods.py      # OOS subperiod breakdown (2022/2023/2024)
│   ├── run_r3b_cross_universe.py  # Cross-universe (NDX+SPX-100 large-cap, 147 names)
│   ├── run_excl_megacap.py        # Ex-mega-cap Track B (+0.589 net, stronger)
│   ├── run_regime_signal.py       # Regime-conditional signal analysis
│   ├── run_ablation.py            # Ablation: FDR-only / SHAP-only / both
│   ├── run_baselines.py           # Momentum, reversal, SPY baselines
│   ├── run_ml_baseline.py         # Naive ML baseline (RF+XGB, no FDR/SHAP)
│   ├── run_conformal_gate.py      # Screen 4 conformal gate (+0.357 → +0.184 gated)
│   ├── oos_inference.py           # DSR + 95 % bootstrap CI for OOS result
│   └── run_oos_attribution.py     # (see above)
│
├── calibration/
│   ├── conformal.py               # ConformalIntervals class (split conformal prediction)
│   ├── run_conformal.py           # Per-fold calibration + regime coverage
│   └── run_aci.py                 # Adaptive conformal inference (ACI)
│
├── baselines/
│   ├── run_ml_baseline.py         # Naive ML baseline driver
│   └── run_significance_tests.py  # Pairwise significance (JKM/Memmel, Wilcoxon, bootstrap)
│
├── knockoffs/
│   ├── run_gaussian_knockoffs.py  # Gaussian model-X knockoffs (FDR robustness check)
│   └── run_cost_aware_knockoffs.py # Cost-aware knockoffs (net-P&L null)
│
├── diagnostics/
│   ├── diag_d2b.py                # Screen-0 diagnostic: D2b re-runs
│   ├── diag_trackb_clean.py       # Track B cleaning diagnostics
│   ├── final_results_screen0.py   # Summary: all Screen-0 headline numbers
│   ├── make_supp_tables.py        # Supplementary tables (full FDR, IC matrix)
│   ├── method_significance.py     # 6-strategy pairwise significance table
│   └── pull_yf_validation.py      # yfinance independent data validation
│
└── figures/
    └── make_figures.py            # All paper figures (PNG → figures/)
```

Supporting files:
```
configs/
├── backtest.yaml                  # Half-spread, impact, AUM, ADV filter, sensitivity grid
├── fdr.yaml                       # BH q-threshold, fold definitions
└── models.yaml                    # Model hyperparameters, walk-forward params

run_r2000_chain.sh                 # End-to-end Russell 2000 re-instantiation (5 stages)
SCREEN0_RESULTS.md                 # Honest results record (all Screen-0 numbers, V13 state)
figures/                           # Generated figures (PNG)
notebooks/                         # Analysis notebooks mirroring src/ pipeline
```

---

## The `CAVAL_UNIVERSE` Environment Shim

`src/universe_paths.py` provides a `proc(root, filename)` helper that inserts a universe suffix into every data path:

```python
from src.universe_paths import proc
features_path = proc(ROOT, "features_all.parquet")
# CAVAL_UNIVERSE unset → data/processed/features_all.parquet   (S&P 500)
# CAVAL_UNIVERSE=r2000 → data/processed/features_all_r2000.parquet
```

Setting `CAVAL_UNIVERSE=r2000` reroutes every stage script's input/output paths to the `*_r2000` artifact set. The same five stage scripts run unchanged on either universe. `run_r2000_chain.sh` runs the full Russell 2000 re-instantiation end-to-end.

---

## Reproducing the Results

### Prerequisites

Raw OHLCV/factor/macro data is **not in this repository**. Sources:
- S&P 500 OHLCV: stored on external drive (reproduce via `src/data/universe_builder.py` + yfinance)
- Ken French daily FF5 + Momentum factors: [mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- FRED macro series (VIX, 2s10s, DXY, etc.): `fredgraph.csv?id=<SERIES>` endpoints
- Russell 2000 OHLCV: `src/data/build_r2000_universe.py` fetches via yfinance

```bash
pip install -r requirements.txt
```

### S&P 500 Pipeline (in order)

```bash
# 1. Feature engineering (requires raw OHLCV + macro data)
python3 -u src/features/build_features_all.py

# 2. Walk-forward model training (~34–40 min, 9 folds × 7 models)
python3 -u src/models/run_walk_forward.py

# 3. SHAP importance analysis
python3 -u src/models/shap_analysis.py

# 4. BH FDR correction (Screen 1)
python3 -u src/fdr/run_fdr.py

# 5. CSCV PBO analysis (Screen 4)
python3 -u src/fdr/run_pbo.py

# 6. Generate composite signals
python3 -u src/backtest/generate_signals.py

# 7. IS backtest + sensitivity (Screens 2 and 3)
python3 -u src/backtest/run_backtest.py
python3 -u src/backtest/run_sensitivity.py

# 8. Screen 2: Tradable-Alpha FDR
python3 -u src/backtest/run_ta_fdr.py

# 9. Screen 3: Alpha Robustness Volume (3D cost grid)
python3 -u src/backtest/run_robustness_volume.py

# 10. Conformal calibration + locked OOS (Screen 4, single touch)
python3 -u src/calibration/run_conformal.py
python3 -u src/backtest/run_holdout.py

# 11. Paper figures
python3 -u src/figures/make_figures.py
```

### Russell 2000 Re-Instantiation (end-to-end)

```bash
bash run_r2000_chain.sh
```

This runs five stages sequentially with `CAVAL_UNIVERSE=r2000` active: walk-forward IC → SHAP → FDR re-derivation → signal generation → locked OOS holdout.

---

## Locked Parameters

| Parameter | Value |
|---|---|
| FDR q-threshold | 0.10 |
| Half-spread (base) | 3 bps |
| Impact coefficient η | 0.10 |
| AUM | $100 M |
| Rebalancing frequency | Weekly (5 d) |
| Feature lag | 1 day |
| Sanitization cap (SANITIZE_CAP) | ±50 % daily return |
| Min price (MIN_PRICE) | $5.00 |
| Min ADV | $1 M |
| OOS window | 2022–2024 (single touch) |

---

## Citation

```bibtex
@article{dwivedi2026caval,
  author  = {Dwivedi, Paritosh},
  title   = {{CAVAL}: A Cost-Aware Validation System for Tradable-Alpha
             Discovery with {FDR} Control},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {Preprint / under review}
}
```

---

## License and Availability

Code is released for reproducibility review. The manuscript is maintained separately and is not included in this repository. Raw data artifacts (`.parquet` files) are not committed; they are produced by running the pipeline scripts from the publicly available sources listed above.
