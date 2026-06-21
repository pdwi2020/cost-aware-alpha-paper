# CAVAL — Rebuilt Results (manifest-backed, post methodological repair)

**Status:** the entire empirical pipeline was rebuilt and re-run on a corrected, look-ahead-free,
point-in-time pipeline (commits up to `5e4eae2`). Every number below is in
`results/manifest/manifest.json` (spec hash `bcd59c0…`) and emitted to
`paper/generated_numbers.tex`. The original manuscript's results are **superseded**.

## Headline: old (manuscript) vs rebuilt (corrected)

| Quantity | OLD paper | REBUILT | Note |
|---|---|---|---|
| Universe | survivorship union | annual point-in-time S&P 500 | **25.6%** of old (ticker,date) cells were contamination (302 partial-member tickers) |
| Screen 0 | downstream, trade-date look-ahead | upstream, lagged (t-1), `s0_eligible` | look-ahead removed in 7 backtest scripts |
| FDR estimand | pooled-fold IC | daily cross-sectional IC + calibrated stationary-block bootstrap | pooled IC let broadcast macros load via time-series |
| Track B BH-significant features | **25** | **0 / 30** | raw-return "macro alpha" was a false discovery |
| Track A BH-significant features | 16 | 12 / 30 (BHY 11) | idiosyncratic *reversal* (negative feature ICs) |
| Walk-forward ensemble IC | — | A: **0.122**, B: ~0.00 | Track A signal real & learnable; Track B noise |
| TA-FDR (cost-aware screen) | "rejects all" (invalid null) | **0 / 30 both tracks** (valid joint block-bootstrap null, mean-net-return stat) | Track A's IC-significant features are NOT tradeable after costs |
| IS net Sharpe | B +0.357 / A −0.59 | B: no strategy; A: **+0.098** (daily rebal) | high-turnover reversal; costs destroy the edge |
| **Locked OOS net Sharpe** | B +0.50 (2022–24) | **A +0.666 over 2025-01..07, but t=0.50, p=0.62** | **statistically indistinguishable from zero** (144-day window); B no strategy |
| 5% position cap | clip-then-reinflate (max 9.09%) | water-fill (max ≤ 5% every day) | |

## The honest conclusion (the new narrative)
**There is no reliable evidence of tradeable alpha.**
1. **Track B (raw return)** — the old +0.50 "macro/sector alpha" — is a **false discovery**: 0 significant
   features under the corrected cross-sectional FDR, no deployable strategy.
2. **Track A (FF5+UMD idiosyncratic residual)** has a genuinely strong *signal* (0.12 IC, 12 FDR
   features = idiosyncratic reversal) that **fails the cost screen**: 0/30 on TA-FDR, marginal IS net
   (+0.098), and an OOS point estimate (+0.666) that is **insignificant over the 7-month window**.
3. **CAVAL's contribution is the protocol** — it separates statistical significance from tradeable
   alpha, catches the raw-return false discovery, and counsels (correctly) that a single short-window
   positive OOS Sharpe is not evidence of alpha.

## Confirmatory evidence (the conclusion holds across 8 independent tests)
| Test | Track A result | Reading |
|---|---|---|
| IC-FDR | 12/30 significant | real cross-sectional signal |
| TA-FDR (cost-aware) | **0/30** | not tradable as single-feature strategies |
| IS net Sharpe (weekly) | **+0.188** | modest |
| **Deflated Sharpe Ratio** | **0.12–0.21 (fails 0.95)** | IS Sharpe within data-mining noise for the search |
| Exploratory 2022–24 | **−0.61** | corrected pipeline is NEGATIVE on the window the old paper claimed +0.357 |
| Locked OOS 2025 | +0.666, **p=0.62** | indistinguishable from zero (7-month window) |
| Ex-mega-cap | **−0.22** | apparent edge concentrated in the 25 largest names |
| Baselines | momentum +0.99 net OOS | plain 12-1 momentum dominates the reversal signal |

Track A net Sharpe sign **flips** across IS(+0.19)/exploratory(−0.61)/OOS(+0.67 insig.) — there is no
stable, deflation-surviving, cost-surviving edge. This is the protocol working as intended.

## Methodological fixes (all committed, 58 unit tests)
PIT universe + look-ahead-free Screen 0 (R1) · feature dedup/de-broadcast + β×macro interactions (R2) ·
daily-XS FDR with **recentered/calibrated** bootstrap p-values (R3; an anti-conservative
`2Φ(−2|Z|)` p-value was caught & fixed) · water-fill cap (R4) · TA-FDR joint stationary-block-bootstrap
null on a mean-net-return statistic (R5; the conservativeness result is now valid — Lean Prop 1
restated on mean-net-return) · custom α-investing → **LORD++** online-FDR (R5b) · stale-output
contamination purged (old +0.705/+0.501 signals).

## Single-touch integrity
The OOS rebuild extended the data to 2025-07-31; the IS (2013–2021) features were verified
**byte-identical** to the frozen pre-OOS backup (max abs diff 0.00e+00 over 1.29M rows). The 2025 OOS
was evaluated exactly once.

## Remaining work (for the manuscript rebuild, R7)
- **Confirmatory robustness** (won't change the conclusion): PBO/DSR on the *deployed* strategy;
  R2000 cross-universe re-run; ablation/sensitivity/sub-period battery; +2 schema reconciliations
  (`run_regime_signal`, `make_supp_tables`). One coherence note: report the **weekly-rebal** IS net
  Sharpe (matching the deployed/OOS), not the daily-rebal +0.098.
- **Manuscript** (`paper/main.tex`): the Track-B narrative inverts; regenerate every table/figure/
  abstract number from the manifest (`paper/generated_numbers.tex`); method/citation corrections
  (Roll = 2√(−cov); Almgren–Chriss is linear, ours is a √-participation model; refs 20/26); ESWA
  anonymisation + Highlights + AI declaration; move Lean/knockoffs/most-conformal to supplementary.
