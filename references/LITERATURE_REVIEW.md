# Literature Review — Cost-Aware Alpha / FDR / Market-Impact Validation

Prepared for the ESA submission (primary) / Quantitative Finance (fallback).
Purpose: situate the manuscript's four-screen validation discipline (FDR → TA-FDR →
ARV → locked OOS) against the prior art, ground the Cost-Aware Knockoffs (CAK)
section in the knockoff lineage, and close the citation gaps a top-venue referee
would flag. PDFs for 9 papers are in `references/pdfs/` (verified); 12 are
citation-only (see `pdfs/MANIFEST.md`). Analysis grouped by gap thread G1–G8.

Legend: 📄 = PDF in repo (analysis grounded in the paper). 🔖 = citation-only.

---

## G1 — Regime-switching / non-stationarity
*Why it matters: the manuscript's headline empirical finding is that Track A (and the
in-sample/out-of-sample sign reversal of Track B) is driven by a macro-regime shift
(2013–21 low-vol → 2022–24 rate-hike). The paper currently asserts regime dependence
with **zero** regime-literature citations.*

- 📄 **Ang & Timmermann (2012), "Regime Changes and Financial Markets"** — survey of
  regime-switching models in finance; regimes parsimoniously capture the
  nonlinearities, fat tails, time-varying volatility and correlation, and persistent
  shifts in mean returns that single-regime models miss. **Use:** the canonical
  citation legitimising our "regime-contingent alpha" interpretation in the Discussion;
  positions our empirical observation within an established modelling tradition rather
  than as an ad-hoc post-hoc story.
- 📄 **Gibbs & Candès (2021), "Adaptive Conformal Inference Under Distribution Shift"**
  — online prediction sets when the data-generating distribution drifts in an unknown
  fashion; a single learnable parameter is re-estimated over time to provably restore
  long-run coverage irrespective of the DGP. Abstract explicitly motivates the method
  with *finance/economics regime shifts*. **Use:** the precise forward-reference for
  our COVID-2020 conformal-coverage breakdown — we document the failure; Gibbs-Candès
  is the principled fix (adaptive conformal), strengthening the Limitations/Conformal
  discussion and the future-work item. Complements `barber2023conformal` (already cited).
- 🔖 **Hamilton (1989), "A New Approach to the Economic Analysis of Nonstationary Time
  Series and the Business Cycle"** — the foundational Markov-regime-switching model.
  **Use:** one-line anchor citation for the concept of discrete regime shifts in the
  Discussion (cited without PDF; canonical).

## G2 — Factor-zoo / anomaly decay / replicability
*Why it matters: the feature library (reversal, momentum, illiquidity, sector
crowding) is anomaly-based; a referee expects engagement with the multiple-testing
and post-publication-decay literature beyond Harvey-Liu-Zhu (2016) and Hou-Xue-Zhang
(2020), both already cited.*

- 📄 **Feng, Giglio & Xiu (2020), "Taming the Factor Zoo: A Test of New Factors"** —
  a double-selection LASSO procedure to test whether a candidate factor contributes
  to pricing *beyond* a high-dimensional set of existing factors, explicitly
  correcting the omitted-variable/model-selection bias; most proposed factors are
  redundant, a few survive. **Use:** the state-of-the-art statement of the
  "factor-discovery-as-multiple-testing" problem our TA-FDR addresses; differentiate —
  Feng-Giglio-Xiu controls for *redundancy* among gross factors, TA-FDR adds the
  orthogonal *tradability* (net-of-impact) screen. Strengthens Related Work §Multiple
  Testing.
- 🔖 **McLean & Pontiff (2016), "Does Academic Research Destroy Stock Return
  Predictability?"** — anomaly returns decay ~58% post-publication. **Use:** the
  Limitations forward-risk sentence (our discovered alpha may decay) and motivation
  for the locked-OOS discipline.
- 🔖 **Chordia, Goyal & Saretto (2020), "Anomalies and False Rejections"** — applies
  multiple-testing/ FDR thinking to the anomaly cross-section, finding most are false
  rejections. **Use:** direct prior art for FDR-on-anomalies; differentiate on the
  cost/tradability axis.
- 🔖 **Harvey & Liu (2021), "Lucky Factors"** — bootstrap multiple-testing procedure
  for sequential factor selection. **Use:** completes the Harvey multiple-testing
  lineage already represented by `harvey2016and`.

## G3 — Transaction-cost factor investing
*Why it matters: this is the closest economic-finance neighbour to the paper's central
contribution (cost-aware discovery), yet the bib only had execution-side costs
(Almgren-Chriss, Frazzini). These papers study net-of-cost anomaly performance — the
exact question TA-FDR asks at the discovery stage.*

- 📄 **Novy-Marx & Velikov (2016), "A Taxonomy of Anomalies and Their Trading Costs"**
  — after-cost performance of a broad anomaly set; introduces the buy/hold-spread cost
  mitigation (stricter entry than exit); anomalies with <50%/month turnover retain
  significant net spreads. **Use:** the empirical-finance counterpart to TA-FDR — they
  show *which* anomalies survive costs ex-post; we build the survival test *into* the
  discovery null. Anchors Related Work §Transaction Cost Modelling and corroborates our
  diversified-ensemble finding (turnover/size drives net survival).
- 🔖 **Detzel, Novy-Marx & Velikov (2023), "Model Comparison with Transaction Costs"**
  — asset-pricing model comparison must be done net of trading costs; cost-aware
  comparison overturns gross-return rankings. **Use:** the strongest statement that
  cost-adjustment changes *which model/feature wins* — exactly our thesis. (PDF was
  mis-downloaded; cite from SSRN 3805379.)
- 🔖 **Chen & Velikov (2023), "Zeroing in on the Expected Returns of Anomalies"** —
  net-of-cost anomaly expected returns are close to zero for many strategies. **Use:**
  reinforces the "statistically real but not tradable" message of TA-FDR.

## G4 — Deep & time-series knockoffs  *(grounds the CAK section — fully covered)*
*Why it matters: the CAK subsection currently cites only Candès et al. (2018). The
LSTM-VAE knockoff and its trivial-reconstruction failure must be positioned against
the knockoff lineage so the failure reads as an informed negative result, not naïveté.*

- 📄 **Barber & Candès (2015), "Controlling the FDR via Knockoffs"** — the original
  (fixed-X) knockoff filter: exact finite-sample FDR control in the linear model with
  n≥p, no knowledge of noise level, "beyond what is possible with permutation-based
  methods," more power than existing rules when the null proportion is high. **Use:**
  the theoretical root of our CAK importance statistic; the antisymmetric W_j filter we
  use is theirs. Cite as the FDR-control foundation, and note our permutation TA-FDR is
  exactly the "permutation-based method" knockoffs improve on.
- 📄 **Romano, Sesia & Candès (2020), "Deep Knockoffs"** — a machine for sampling
  approximate model-X knockoffs for arbitrary/unspecified distributions via deep
  generative models, iteratively refining the sampler to minimise an MMD-style
  pairwise-exchangeability criterion; model-free. **Use:** **our LSTM-VAE knockoff sits
  squarely in this lineage.** Cite to (i) justify the deep-generative approach as
  principled, and (ii) frame our trivial-reconstruction collapse as a known risk of
  deep knockoff samplers on heavy-tailed/low-signal financial data — turning the
  negative CAK result into an informed contribution.
- 📄 **Sesia, Sabatti & Candès (2019), "Gene Hunting with HMM Knockoffs"** — exact
  knockoff sampling for hidden-Markov-structured covariates with guaranteed FDR
  control. **Use:** the precedent for *dependence-aware* (serial-structure) knockoffs;
  motivates why we attempted a temporal generative model and points to the HMM
  alternative as future work for the CAK failure mode.

## G5 — Data-snooping tests beyond FDR
*Why it matters: the paper uses Bonferroni/ FWER reasoning for the two-track OOS
comparison and CSCV-PBO, but never engages the reality-check / SPA lineage that
referees in this area expect alongside FDR.*

- 🔖 **White (2000), "A Reality Check for Data Snooping"** — bootstrap test for whether
  the best of many strategies beats a benchmark after accounting for the full search.
  **Use:** Related Work §Multiple Testing / §Backtest Overfitting — the FWER complement
  to our FDR approach; situates CSCV-PBO in the snooping-control tradition.
- 🔖 **Hansen (2005), "A Test for Superior Predictive Ability"** — a more powerful
  refinement of White's reality check. **Use:** cite together with White as the
  snooping-control pair.

## G6 — Time-series cross-validation + HAC inference
*Why it matters: the expanding-window walk-forward is the methodological centerpiece;
HAC is mentioned in the OOS-significance subsection but uncited.*

- 📄 **Bergmeir, Hyndman & Koo (2018), "A Note on the Validity of Cross-Validation for
  Evaluating Autoregressive Time Series Prediction"** — shows K-fold CV is valid for
  purely autoregressive models with uncorrelated errors (theory + simulation + real
  data), and is favourable vs OOS-only evaluation. **Use:** the theoretical
  justification that our walk-forward/CV design yields honest generalisation estimates
  on serially-dependent data; pre-empts the "CV is invalid on time series" referee
  objection. Strengthens Related Work §Backtest Overfitting and the walk-forward design
  section.
- 🔖 **Newey & West (1987), HAC covariance estimator** — **Use:** the missing citation
  for the HAC t-test already used in the OOS-significance subsection (pair with the
  already-present `politis1994stationary` stationary bootstrap).

## G7 — Deep asset pricing / venue currency
*Why it matters: shows engagement with the recent deep-learning-for-asset-pricing
frontier (the venue's own tradition), beyond the existing ESA/ASoC/EJOR cites.*

- 📄 **Chen, Pelger & Zhu (2024), "Deep Learning in Asset Pricing"** — deep NNs estimate
  a no-arbitrage conditional asset-pricing model; the no-arbitrage condition is the
  criterion function, test assets are built adversarially, and macro states are
  extracted from many time series via an RNN; outperforms OOS in Sharpe, explained
  variation, and pricing errors. **Use:** the deep-AP frontier reference in Related Work
  §Feature-Based Equity Models; contrast — they pursue maximal predictive flexibility,
  we pursue *validated, cost-aware* discovery; the two are complementary (flexibility vs
  discipline). Complements `gkx2020`.
- 🔖 **Gu, Kelly & Xiu (2021), "Autoencoder Asset Pricing Models"** — conditional
  autoencoder with characteristic-dependent latent factors/loadings. **Use:** second
  deep-AP anchor; cite alongside Chen-Pelger-Zhu.

## G8 — Portfolio-construction theory
*Why it matters: the position rule (z-score → L1-norm → cap → renormalise) is a
heuristic; a referee will ask why not principled optimisation.*

- 🔖 **Markowitz (1952), "Portfolio Selection"** — mean-variance optimisation, the
  baseline alternative. **Use:** acknowledge the principled benchmark in the position-
  construction section.
- 🔖 **López de Prado (2016), "Building Diversified Portfolios that Outperform Out of
  Sample" (HRP)** — shows quadratic optimisers (Markowitz CLA) are unstable,
  concentrated, and underperform OOS; HRP is a robust alternative. **Use:** the
  justification that our deliberately simple, robust heuristic is a *defensible* choice
  over unstable in-sample optimisation — turns a perceived weakness into a reasoned
  design decision (and a future-work pointer to HRP). Author already cited via
  `lopez2018advances`.

---

## Net effect on the paper
The downloaded evidence base fully grounds the two threads where grounding most
matters — **G4 (knockoff lineage, for the CAK negative result)** and **G1 (regime +
adaptive conformal, for the headline regime-dependence finding)** — plus the flagship
paper of each remaining thread. The 12 citation-only papers are canonical or have
abstracts in hand; their bib entries are accurate without the PDF. No claim in the
revised manuscript will rest on a paper not actually read or well-established.

See the **gap-closure map** (next section) for exact `main.tex` insertion points.

---

## Gap-closure map (LR5) — exact `main.tex` insertion points

Citation style: `elsarticle` numeric. Each new bibkey below MUST be cited at its
mapped location (LR9 orphan-check enforces this). 21 new entries; delete duplicate
`lundberg2017unified`.

| Gap | bibkeys to add | Insertion point in `main.tex` | What the cite establishes (1-line framing) |
|-----|----------------|-------------------------------|--------------------------------------------|
| G2 | `feng2020taming`, `harvey2021lucky` | Related Work §"Multiple Testing in Finance" (after the `hou2020replicating` sentence) | State-of-the-art factor-redundancy / multiple-testing selection; TA-FDR adds the orthogonal tradability axis. |
| G2 | `chordia2020anomalies` | Related Work §"Multiple Testing in Finance" (same paragraph) | FDR-on-anomalies prior art; differentiate on cost/tradability. |
| G5 | `white2000reality`, `hansen2005spa` | Related Work §"Multiple Testing in Finance" (new 2-sentence close) | FWER reality-check / SPA lineage complements our FDR + CSCV-PBO. |
| G6 | `bergmeir2018note` | Related Work §"Backtest Overfitting" (new sentence) AND §"Expanding-Window Design" (walk-forward justification) | K-fold/walk-forward CV is valid on serially-dependent AR data — pre-empts "CV invalid on time series". |
| G3 | `novymarx2016taxonomy`, `detzel2023model`, `chen2023zeroing` | Related Work §"Transaction Cost Modelling" (new 3-sentence paragraph) | Net-of-cost anomaly survival is the empirical-finance neighbour of TA-FDR; cost-adjustment changes which feature wins. |
| G7 | `chen2024deeplearning`, `gu2021autoencoder` | Related Work §"Feature-Based Equity Models" (after `gkx2020`) | Deep-AP frontier; contrast flexibility (theirs) vs validated cost-aware discipline (ours). |
| G8 | `markowitz1952portfolio`, `lopezdeprado2016hrp` | §"Position Construction" (`\subsection{Position Construction}`, ~line 1046) — 1-2 sentences justifying the heuristic | Quadratic optimisers are unstable OOS (HRP); our robust heuristic is a reasoned choice, not a shortcut. |
| G4 | `barber2015controlling`, `romano2020deep`, `sesia2019gene` | §"Cost-Aware Knockoffs" (`subsec:cak`) — extend the Construction paragraph | Roots CAK in the knockoff lineage; frames the LSTM-VAE trivial-reconstruction collapse as a known deep-knockoff failure mode, with HMM-knockoffs as the principled alternative. |
| G1 | `hamilton1989regime`, `ang2012regime` | Discussion §"Regime Dependence and the OOS Surprise" (~line 1700) | Legitimises the regime-contingent interpretation within an established modelling tradition (headline finding). |
| G1 | `gibbs2021adaptive` | Discussion §"Conformal Calibration Limitations" / Limitations conformal bullet | Adaptive conformal is the principled fix for the documented COVID coverage breakdown; future-work pointer. |
| G2 | `mclean2016does` | §"Limitations" (new forward-risk sentence) | Post-publication anomaly decay (~58%) as a forward risk → motivates the locked-OOS discipline. |
| G6 | `newey1987hac` | §"Statistical Significance of the OOS Sharpe" (~line 1363) | The missing citation for the HAC t-test already used (pair with present `politis1994stationary`). |

**Bib hygiene (LR6):** delete `lundberg2017unified` (keep `lundberg2017shap`); confirm
`politis1994stationary` is cited at the OOS-significance subsection (it will be, paired
with `newey1987hac`).

**Net:** references.bib 33 → ~53 entries (−1 duplicate). Expected page delta:
+2–3 pp (Related Work ≈ +0.7 pp, CAK ≈ +0.4 pp, Discussion/Position/Limitations ≈ +0.6 pp).
