# Key References — ML Paper

> **See [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md)** for the full referee-grade
> literature review (gap threads G1–G8, per-paper analysis, and the gap→`main.tex`
> insertion-point map). Downloaded PDFs + status are in [`pdfs/MANIFEST.md`](pdfs/MANIFEST.md).

## Core Papers

| # | Authors | Year | Title | Venue | Link |
|---|---------|------|-------|-------|------|
| 1 | Harvey, Liu, Zhu | 2016 | ...and the Cross-Section of Expected Returns | RFS 29(1) | https://doi.org/10.1093/rfs/hhv059 |
| 2 | Bailey, Borwein, López de Prado, Zhu | 2015 | The Probability of Backtest Overfitting | JCF 20(4) | https://doi.org/10.21314/JCF.2015.316 |
| 3 | López de Prado | 2018 | Advances in Financial Machine Learning | Wiley | ISBN 9781119482086 |
| 4 | Gu, Kelly, Xiu | 2020 | Empirical Asset Pricing via Machine Learning | RFS 33(5) | https://doi.org/10.1093/rfs/hhaa009 |
| 5 | Almgren, Chriss | 2001 | Optimal Execution of Portfolio Transactions | JRisk 3(2) | https://doi.org/10.21314/JOR.2001.041 |
| 6 | Frazzini, Israel, Moskowitz | 2015 | Trading Costs of Asset Pricing Anomalies | arXiv | https://arxiv.org/abs/1506.05075 |
| 7 | Guo, Pleiss, Sun, Weinberger | 2017 | On Calibration of Modern Neural Networks | ICML | https://arxiv.org/abs/1706.04599 |
| 8 | Angelopoulos, Bates | 2022 | A Gentle Introduction to Conformal Prediction | arXiv | https://arxiv.org/abs/2107.07511 |
| 9 | Benjamini, Hochberg | 1995 | Controlling the False Discovery Rate | JRSS-B 57(1) | https://doi.org/10.1111/j.2517-6161.1995.tb02031.x |
| 10 | López de Prado, Bailey | 2014 | The Deflated Sharpe Ratio | JPM 40(5) | https://doi.org/10.3905/jpm.2014.40.5.094 |

## One-Line Summaries

1. **Harvey et al. (2016)** — "…and the Cross-Section of Expected Returns": Shows that most published factor discoveries are false positives when adjusted for multiple testing; sets the bar for t-stat > 3.0.

2. **Bailey et al. (2015)** — "PBO via CSCV": Combinatorial Symmetric Cross-Validation gives a model-free probability of overfitting; PBO > 0.50 means the strategy is likely overfit.

3. **López de Prado (2018)** — "AFML": Practitioner textbook; Chapter 8 (feature importance), Chapter 10 (backtest methodology), Chapter 14 (cross-validation in finance) are directly relevant.

4. **Gu, Kelly, Xiu (2020)** — "Empirical Asset Pricing via ML": Large-scale horse race of ML methods vs OLS for return prediction; tree models + neural nets dominate; feature importance via SHAP.

5. **Almgren, Chriss (2001)** — "Optimal Execution": Square-root market impact model; impact cost = η·σ·(q/V)^0.5; closed-form optimal liquidation trajectory.

6. **Frazzini, Israel, Moskowitz (2015)** — "Trading Costs": Measures actual trading costs for quant strategies; finds momentum and size anomalies survive realistic cost assumptions; calibrates impact_coeff.

7. **Guo et al. (2017)** — "Calibration of Modern NNs": Temperature scaling as post-hoc calibration; ECE as calibration metric; reliability diagram methodology.

8. **Angelopoulos, Bates (2022)** — "Conformal Prediction": Exchange-validity guarantees; split conformal; RAPS for sets; applicable to prediction intervals for return forecasting.

9. **Benjamini, Hochberg (1995)** — "FDR Control": Original BH procedure; controls E[V/R] at level q regardless of dependency structure (positive dependency assumption).

10. **López de Prado, Bailey (2014)** — "Deflated SR": Adjusts Sharpe Ratio for number of trials; DSR = (SR − E[max SR]) / std[max SR] under Gaussian returns; complement to PBO.

## Additional Reading

- **Fama, French (2015)** — Five-Factor Model: https://doi.org/10.1016/j.jfineco.2014.10.010
- **Jegadeesh, Titman (1993)** — Returns to Buying Winners: https://doi.org/10.1111/j.1540-6261.1993.tb04702.x
- **Amihud (2002)** — Illiquidity and Stock Returns: https://doi.org/10.1016/S1386-4181(01)00024-6
- **Carhart (1997)** — On Persistence in Mutual Fund Performance: https://doi.org/10.1111/j.1540-6261.1997.tb03808.x
