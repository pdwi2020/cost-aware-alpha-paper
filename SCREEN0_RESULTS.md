# Screen-0 Honest Results (2026-06-16) — for D3 manuscript renumber

All numbers below are computed on **cleaned data**: ±50% daily-return sanitization
(corporate-action/penny-print artifacts removed) + **Screen 0 tradeable-universe
filter (price ≥ $5, $1M ADV)**, applied uniformly. Pre-specified params: spread 3 bps,
impact η=0.10, AUM $100M, rebal 5d (weekly), FDR q=0.10. OOS = 2022–2024 (single touch).

Validated/reproducible via: `src/backtest/run_holdout.py` (Screen 0 wired in,
`MIN_PRICE=5.0`, `apply_min_price_filter`), `src/diagnostics/diag_d2b.py`,
`src/diagnostics/final_results_screen0.py`. Backups: `*.raw.bak.parquet`.

## 1. Baselines table (tab:baselines) — REPLACE with these
| Strategy | IS gross | IS net | IS drag | OOS gross | OOS net | OOS drag |
|---|---|---|---|---|---|---|
| Buy-and-Hold SPY† | +1.00 | +1.00 | 0 | +0.58 | +0.58 | 0 |
| Momentum L/S (12-1) | +0.17 | +0.13 | 42 | **+0.93** | **+0.88** | 43 |
| Reversal L/S (5d) | +0.56 | +0.12 | 388 | +0.25 | −0.22 | 356 |
| Track A (ours) | −0.05 | −0.13 | 76 | −0.52 | −0.59 | 59 |
| **Track B (ours)** | **+0.64** | **+0.51** | 128 | **+0.47** | **+0.36** | 111 |

KEY CHANGES vs old (contaminated) table:
- Reversal OOS net +0.53 → **−0.22** (the +0.53 was a single +445% penny artifact on 2024-12-26).
- Momentum OOS net −0.01 → **+0.88** (clean momentum is strong in the 2022–24 regime).
- Track B IS net −0.64 → **+0.51** (now positive IS); OOS +0.50 → **+0.357**.
- **Momentum (+0.88) now BEATS Track B (+0.357) on OOS Sharpe — must disclose.**

## 2. Headline (abstract/intro/results/conclusion): +0.50 → **+0.357**
Track B is now positive in BOTH periods (IS +0.51 → OOS +0.357): consistent,
cost-robust market-neutral alpha (not the old "IS-negative, OOS-positive reversal").

## 3. Subperiods (Track B OOS net Sharpe by year)
2022 **+1.12**, 2023 **+0.81**, 2024 **−1.04**; full +0.357. (Old: 2022 +2.13.)

## 4. Cost sweep (Track B OOS net Sharpe; rows half-spread h, cols impact η)
| h \ η | 0.05 | 0.10 | 0.20 |
|---|---|---|---|
| 1 bps | +0.43 | +0.41 | +0.37 |
| 3 bps (base) | +0.37 | **+0.36** | +0.32 |
| 5 bps | +0.32 | +0.30 | +0.27 |
| 10 bps | +0.19 | +0.17 | +0.14 |
All positive → cost-robust.

## 5. Ablation (Track B OOS, Screen 0) — strong design justification
| Variant | n_feat | gross | net |
|---|---|---|---|
| Baseline (FDR + SHAP) | 25 | +0.471 | **+0.357** |
| Remove FDR (all 35 feats) | 35 | +0.067 | **−0.080** |
| Remove SHAP (uniform wts) | 25 | +0.138 | **+0.101** |
Both screens add real value (FDR removal → −0.08; SHAP removal → +0.10). Survives cleaning.

## 5b. Significance / CI for the +0.357 headline (from Screen-0 net P&L)
- Annualized net Sharpe +0.357, n=753. Lo (2002) SE = 0.58 → **95% CI [−0.78, +1.49], p=0.54**.
- Stationary-bootstrap (block≈10) 95% CI **[−0.62, +1.29], P(SR>0)=0.77**.
- Still **not statistically distinguishable from zero** (consistent with old +0.50, p=0.34) →
  reinforces repositioning: contribution is the validation methodology, not a statistically
  dominant alpha. Update tab/line ~1515, ~1541 (old "+0.50, CI [−0.45,+1.67], p=0.34").

## 6. yfinance independent validation (robustness)
Pulled split/div-adjusted Close for 610 listed names (`yf_adjclose_validation.parquet`):
per-name daily-return corr 0.85–0.98 (AAPL 0.91, JPM 0.98); 38+ of our >50% daily
moves confirmed as artifacts; 51 unmatched = delisted (Screen-0-excluded). Data sound
for the tradeable universe. NB: our close is price-return, yfinance total-return (div gap).

## 6b. D2d secondary re-runs (DONE 2026-06-16, Screen 0, all anchors pass)
- **ARV (`robustness_volume`/`sensitivity_3d` regenerated):** ARV(SR>0)=**83.3%** (50/60 cfgs, 3D grid), ARV(SR>0.25)=75.0%, ARV(SR>0.50)=**43.3%**, median net SR +0.439. (Old "100%/42%" was a 12-config 2D grid — not comparable.)
- **Attribution (LOGO marginals) — STORY CHANGED.** Standalone net SR: Sector +0.57, **Liq/Vol +0.65**, Macro −0.37, Rev/Mom +0.15, Full +0.37. LOGO marginals: **Liq/Vol +0.45** (was −0.54), Sector +0.04 (was +0.95), Macro −0.26, Rev/Mom −0.01. → Rewrite attribution narrative: liquidity/volatility is now the main marginal contributor; sector-crowding is NOT. (old +0.95/−0.54 at ~L1611-1626, L1541.)
- **Cross-universe (NDX+SPX-100 large-cap subset, 147 names):** Track B net **−0.065**, Track A −0.094. Screen 0 filters 0 here (all ≥$5); change is from return-sanitization. → Honest caveat: Track B alpha is ~0 on pure large-caps (it lives in the broader/smaller-cap names). NB this is a DIFFERENT universe from the +0.501/+0.357 full-S&P holdout (L1800).
- **Ex-mega-cap:** Track B net **+0.589** (was +0.489), gross +0.710. → Track B is STRONGER ex-mega-cap; alpha not from mega-caps (good robustness). (L205 ex-mega-cap mention.)
- **IS backtest (Screen 0):** Track B IS weekly net **+0.507** (anchor ✓), daily +0.195; Track A IS net −0.264.
- **DSR/inference:** Track B OOS +0.357, 95% bootstrap CI **[−0.62,+1.30]**, Lo p=**0.54**, NW t=+0.71 → not significant at 5% (consistent). (L1515,L1541.)
- Backups: `*.s0.bak.parquet` for backtest_base, backtest_sensitivity, excl_megacap_metrics, oos_attribution, oos_inference, r3b_cross_universe_{metrics,year}, robustness_volume, sensitivity_{3d,summary}. Scripts patched: run_backtest, run_sensitivity, run_oos_attribution, run_r3b_cross_universe, run_excl_megacap (+ run_holdout earlier).

## 7. (DONE — see §6b) Former pending list
Patch `apply_min_price_filter` (post `signal_to_positions`, pre `simulate_pnl`) into:
- `run_robustness_volume.py` — ARV (one of the four screens). OLD: ARV(0.50)=42%.
- `run_oos_attribution.py` — LOGO marginals. OLD: sector-crowding +0.95, liq/vol −0.54, macro ≈0.
- `run_r3b_cross_universe.py` — cross-universe / mid-cap. OLD: Track B +0.501 full, ex-mega +?.
- `run_excl_megacap.py` — ex-mega-cap Track B.
- single-feature T_f table (tab ~line 938) + 9×7 IC matrix.
- `oos_inference.py` — DSR + 95% CI (OLD: +0.50, CI [−0.45,+1.67], p=0.34) — recompute for +0.357.
- IS sensitivity_3d.parquet (stale; backtest IS under Screen 0).

## 8. Manuscript renumber map (old → new), key spots (grep done 2026-06-15)
- L112, L205, L911, L1479, L1515, L1550, L1573, L2006: headline +0.50 → +0.357.
- L1463/L1683 (baselines tables): replace rows per §1 above.
- L1681 reversal +0.53/+0.57 row + L1692 narrative ("happens to recover OOS"): now −0.22 (artifact removed).
- L1733-1734 (cost sweep grid): replace per §4.
- L1906-1925 (ablation): replace per §5; note FDR-removal collapse −0.08.
- L1394 (conformal gate +0.50→+0.33): recompute base from +0.357.
- L1800 (cross-universe +0.501): pending re-run (§7).
- "three observations" narrative (~L1688): rewrite — momentum now beats Track B; disclose honestly.

## 9. Repositioning (D3 narrative)
- Add **Screen 0: data-integrity + tradeable-universe pre-screen** as a formal screen.
- The artifact discovery = motivating case study (a near-false-discovery the 4 statistical
  screens missed — the paper's own thesis demonstrated live).
- Track B = consistent, cost-robust, validated multi-feature alpha (+0.51 IS → +0.357 OOS).
- Disclose clean momentum +0.88 > Track B; frame CAVAL's value as disciplined validation
  + regime/cost robustness, not raw-Sharpe dominance; note momentum's IS weakness (+0.13)
  = single-factor regime fragility that disciplined multi-feature validation guards against.

## 10. ESA revision board V-tasks (DONE 2026-06-16, all Screen 0)
- **V3 conformal gate (exact, re-run under Screen 0):** ungated +0.357 (Calmar 0.247); gated_v1 net +0.184 / Calmar 0.105 (DEMOTE); gated_v2 unchanged (never triggers). Manuscript now quotes "+0.357→+0.184, Calmar 0.247→0.105". Script: run_conformal_gate.py (assert updated 0.50→0.357).
- **V2 naive-ML baseline (RF+XGB, all 38 feat, NO FDR/NO SHAP, Screen 0, same costs):** IS gross +0.674/net **+0.387** (TO 44x, drag 197bps); OOS gross −0.046/net **−0.252** (TO 31x, drag 128bps). IS = 9-fold walk-forward OOF; OOS = frozen model ≤2021 (single touch). → naive ML overfits IS, FAILS OOS; FDR+SHAP converts it to Track B +0.357. New tab:baselines row + 5th observation. Script: run_ml_baseline.py; pnl: ml_baseline_pnl_{is,oos}.parquet.
- **V3 pairwise significance (6 strategies, OOS, n=752 common days):** JKM/Memmel + stationary bootstrap + Wilcoxon. Per-strategy vs 0: only Momentum marginal (p=0.082); Track B p=0.554. NO Track B pairwise comparison significant (all p≥0.15). Only 5%-significant pairs: Momentum>Reversal (boot 0.033), SPY>Track A (Wilcoxon 0.010), Momentum>NaiveML (Wilcoxon 0.023). New tab:pairwise + subsection. Script: method_significance.py.
- **V5** ablation promoted to standalone \section{Ablation: What Each Component Contributes} (outer=naive-ML, inner=FDR/SHAP) + roadmap.
- **V6** complexity/scalability subsec (Apple M2, holdout backtest ~50s, 2-model WF ~34-40min, validation screens asymptotically free).
- **V7** Threats-to-Validity (internal/external/construct/stat-conclusion) + Practical Implementation Guide subsecs.
- **V8** appendix: full 35-feat FDR table (tab:fdr_full, both tracks) + 9×7 IC matrix (tab:ic_matrix). Generator: make_supp_tables.py.
- **V9** Notation appendix table (tab:notation).
- **V10** +Jensen-Kelly-Pedersen 2023 JF (replication crisis overstated) into Related Work.
- **V11** CRediT statement + GenAI disclosure + firmed Data Availability (repo URL) + cover_letter.md (paper/submission/).
- **V4** graphical_abstract.{tex,pdf} (5-screen flow, 6:1 landscape) + elsarticle highlights & graphicalabstract envs in frontmatter.
- **V12** local: CITATION.cff + .zenodo.json prepared; PUBLISH (public push + Zenodo DOI) PENDING USER.
- **V13** new universe (Russell 2000): **DONE (2026-06-18)** — full re-instantiation, see §11 below.
- Manuscript: 79 pp, 0 undefined refs, abstract 249 words. Screen 0 formalized as 5th screen (Highlights/abstract/intro contribution #1/overview/algorithm/conclusion). ARV reconciled to 83.3%/60 in abstract+intro (was stale 100%/12).

---

## §11 — V13: Russell 2000 Full Re-Instantiation (2026-06-18)

**What:** entire CAVAL pipeline re-run from scratch on US small-caps — fresh universe,
features, walk-forward IC, *re-derived* BH selection, re-fit SHAP, single-touch OOS.

**Universe/data:** 986 survivors (Vanguard VTWO constituents tracking the Russell 2000;
iShares IWM endpoint bot-blocked), continuous history ≥ 2013-01. Raw daily OHLCV via
yfinance. All universe-agnostic inputs (FF5+UMD, FRED macro, sector ETFs) re-sourced
from PUBLIC providers (Ken French / FRED CSV / yfinance) → **no proprietary-data
dependency** (X9 DuckDB catalog was TCC-blocked at run time). 3 intraday features
(vwap_dev, vol_clock, vol_sig_ratio) unavailable on daily data → 35 features (not 38).

**Screen-0 finding (on-thesis):** small-cap overnight_gap reached 2.4e8 and roll_spread
1.0e8 (vs S&P maxima 3.2e3 / 1.0e4) — corporate-action artifacts the ±50% close-to-close
cap did not reach. Since the composite signal sums feature *levels*, one uncapped artifact
collapsed both tracks onto an identical degenerate signal (cross-track P&L corr = 1.00,
net SR ≈ 0). Root cause: overnight_gap used raw `open` ÷ clean-once-reconstructed `close`
(scales diverge after clipped splits). Fix = extend Screen-0 winsorization: overnight_gap
from RAW close clipped ±50%; roll_spread clipped [0,1]; amihud winsorized at p99.9. After
fix: two distinct channels (cross-track P&L corr 0.65). $5 floor removed 7.8% of positions.

**Re-derived BH (q=0.10):** 19/32 features EACH track, distinct sets:
- Track A: ret_252d, ret_63d, mom_12_1, ret_5d, reversals, sharpe_21d (momentum/reversal)
- Track B: vix, corr_XLI, corr_XLF, corr_SPY, corr_XLY, corr_XLK, ret_21d (macro/sector)

**Locked OOS 2022–2024 (single touch, Screen 0, base costs, AUM $100M):**

| Track | IS ens. IC | Gross SR | Net SR | Net ann. | Max DD | Hit | Turnover | Cost drag |
|-------|-----------:|---------:|-------:|---------:|-------:|----:|---------:|----------:|
| A (idio)   | +0.133 | −0.648 | **−0.847** | −5.7% | −20.3% | 46.0% | 20.0×/yr | 135 bps/yr |
| B (sector) | +0.035 | −0.145 | **−0.574** | −2.9% |  −9.6% | 46.9% | 31.6×/yr | 216 bps/yr |

**Honest reading:** protocol re-instantiates mechanically on a wholly different universe;
verdict is a REJECTION. Track A has strong IS IC (+0.13) but fails the locked OOS (−0.85) —
a textbook IS→OOS collapse that CAVAL correctly catches (a naïve IS-only analysis would
have championed it). Consistent with cross-universe evidence (alpha is S&P-mid-cap-specific)
and the "discipline not dominance" thesis.

**Artifacts:** data/processed/{daily_ohlcv_r2000,features_all_r2000,ic_by_fold_r2000,
shap_summary_r2000,fdr_results_r2000,signals_track_*_r2000,holdout_metrics_r2000,
holdout_pnl_r2000_track_*}.parquet ; r2000_inputs/ (cached public factors/macro/ETF).
**Code:** src/data/build_r2000_universe.py, src/features/build_features_r2000.py,
src/universe_paths.py (CAVAL_UNIVERSE=r2000 env shim across the 5 stage scripts; S&P
behavior unchanged when unset). **Manuscript:** §sec:r2000 + tab:r2000 + Limitations;
82 pp, 0 undefined refs.
