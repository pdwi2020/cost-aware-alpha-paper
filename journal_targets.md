# ML Paper — Journal Target List

**Paper:** Cost-Aware Alpha Generation with FDR Control and Market-Impact-Aware Validation  
**Data:** yfinance + Kaggle SPY + Ken French + FRED (all free, Scopus-acceptable)  
**Target:** High IF · Scopus Q1 · Applied ML or Finance

---

## Tier 1 — Submit Now (in parallel)

| Priority | Journal | Publisher | IF | Scopus | Format |
|---|---|---|---|---|---|
| **#1** | **Expert Systems with Applications** | Elsevier | **10.48** | Q1 | Full paper (~30 pp manuscript) |
| **#2** | **Finance Research Letters** | Elsevier | 7.4 | Q1 | Short letter (3 500 words max) |

**#1 ESA** — Primary target. yfinance data accepted routinely. ML pipeline contributions (FDR + PBO + Almgren-Chriss + conformal CP) are core scope. Highest viable IF given data constraints.

**#2 FRL** — Run in parallel with #1. Focused 3 500-word letter on a single finding: *"Track B OOS net Sharpe +0.50 after joint FDR control and Almgren-Chriss cost adjustment."* Turnaround 6–8 weeks. Gets the OOS result into the literature while ESA review runs (~3–4 months).

---

## Tier 2 — Fallback if Tier 1 Rejects

| Priority | Journal | Publisher | IF | Scopus | Key Angle |
|---|---|---|---|---|---|
| **#3** | **Knowledge-Based Systems** | Elsevier | 7.6 | Q1 | ESA sibling; same data bar |
| **#4** | **Information Sciences** | Elsevier | 8.1 | Q1 | Broad ML applications; large readership |
| **#5** | **Decision Support Systems** | Elsevier | 7.5 | Q1 | Reframe as uncertainty-aware position sizing |
| **#6** | **Applied Soft Computing** | Elsevier | 7.2 | Q1 | Ensemble methods + applied ML |

---

## Tier 3 — Finance-Audience Journals

| Priority | Journal | Publisher | IF | Scopus | Key Angle |
|---|---|---|---|---|---|
| **#7** | **Quantitative Finance** | Taylor & Francis | 2.6 | Q1 | Reviewers understand FDR/PBO/AC without explanation |
| **#8** | **Journal of Portfolio Management** | Pageant Media | 2.0 | Q1 | Practitioner; OOS result + cost model |

---

## Do Not Target

| Journal | Reason |
|---|---|
| RFS / JoF / JFE | CRSP required — not accessible without WRDS/university affiliation |
| TNNLS / TPAMI | No neural network content — desk reject |
| ICML / NeurIPS / ICLR | No new ML algorithm — desk reject |
| International Journal of Forecasting | Comparable paper (Wolff & Echterling 2024) used Bloomberg; yfinance borderline |
| Neural Networks (Elsevier) | No neural nets in paper |

---

## Submission Timeline

```
Week 1–2   Expand paper: 8 IEEE pages → 30 manuscript pages
           Add: ablation study, extended related work, baseline comparisons
Week 2     Submit to ESA (mc.manuscriptcentral.com/eswa)
Week 2     Submit short version to FRL (in parallel)
Month 3    ESA decision → accept or route to KBS / Information Sciences
```

### What the ESA Expansion Needs

1. **Extended related work** — GKX2020, Wolff & Echterling (2024), Harvey et al. (2016)
2. **Ablation study** — net Sharpe with/without: FDR filter, conformal sizing, ADV filter
3. **Baseline table** — buy-and-hold, equal-weight momentum, gross-only (no cost model)
4. **Limitations** — yfinance data quality, SPY constituent approximation, S&P 500 universe only

---

## Key Results (for reference)

| | Track A (Reversal) | Track B (Macro/Sector) |
|---|---|---|
| BH-significant features | 16 | 25 |
| PBO (126 CSCV paths) | 0.00 | 0.14 |
| IS gross Sharpe 2013–21 | +0.496 | −0.192 |
| IS net Sharpe (weekly, 3 bps) | −0.095 | −0.665 |
| **OOS gross Sharpe 2022–24** | −0.346 | **+0.705** |
| **OOS net Sharpe 2022–24** | −0.584 | **+0.501** |
| Conformal coverage (target 90%) | 88.8% | 89.1% |

---

*Last updated: May 2026*
