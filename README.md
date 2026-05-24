# Cost-Aware Alpha Generation with False Discovery Control

**Status:** Pre-implementation | **Compute:** Kaggle CPU (free) | **Timeline:** 12 weeks

## Abstract

We construct a two-tier equity alpha library (classic momentum/reversal/vol/liquidity + intraday microstructure + cross-asset + crowding proxies) and apply rigorous false discovery rate control (Benjamini-Hochberg, q=0.05) and probability of backtest overfitting analysis (CSCV, S=16) before any capital allocation. The selected signals are combined in an ensemble model and evaluated under realistic execution costs via an Almgren-Chriss square-root impact model, producing a 2D cost-sensitivity surface. Calibrated uncertainty quantification via conformal prediction governs position sizing, with the final strategy validated on a true holdout period (2022–2024).

## Key Contributions

- Two-tier feature library (40+ signals) with strict lookahead prevention and FF5+Mom residualization
- Joint BH-FDR + CSCV-PBO framework applied before any hyperparameter optimization
- 2D execution cost surface (spread × impact) identifying the "profitable region"
- Conformal position sizing: signal-to-width ratio normalized to fixed gross budget

## Datasets

| Dataset | Path on X9 | Size |
|---------|-----------|------|
| Equity OHLCV 1m US | `data/equity_ohlcv_1m_us` (symlink) | 79 GB |
| Fama-French + AQR Factors | `data/fama_french_factors` | 3.3 MB |
| S&P 500 Holdings | `data/sp500_holdings` | 13 MB |
| Polymarket Features | `data/polymarket_features` | 57.6 MB |
| FRED Macro | `data/fred_macro` | 6.3 MB |

## Compute

- Platform: **Kaggle CPU** (free tier)
- No GPU required — all models are sklearn/XGBoost/LightGBM
- Estimated runtime: 2–4h per walk-forward fold; 20–30h total
- See [`compute/KAGGLE_SETUP.md`](compute/KAGGLE_SETUP.md)

## Quick Start

```bash
conda env create -f environment.yml
conda activate ml_paper
cp .env.example .env  # fill in X9 path
jupyter notebook notebooks/00_data_audit.ipynb
```

## Navigation

- [`KICKOFF.md`](KICKOFF.md) — comprehensive session guide (read first)
- [`PLAN.md`](PLAN.md) — week-by-week implementation roadmap
- [`notebooks/`](notebooks/) — numbered analysis notebooks (00–07)
- [`src/`](src/) — Python modules
- [`configs/`](configs/) — hyperparameter YAMLs
- [`references/REFERENCES.md`](references/REFERENCES.md) — key papers

## Target Journals

1. **Review of Financial Studies** (IF 8.1) — primary
2. **Journal of Finance** (IF 7.8) — secondary
3. **JFQA** (IF 3.4) — tertiary fallback
