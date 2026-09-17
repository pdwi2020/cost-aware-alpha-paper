# Reproducibility Package: CAVAL

This document maps the repository to the standard reproducibility layout
(`data_processing / training / backtesting / evaluation`) requested by reviewers
and records the single source of truth for every number in the manuscript.

Current vintage: **spec v3.0-rie**, daily point-in-time universe, re-run
2026-09-17 after the `corr_*` feature-source correction.

## Source of truth

- **Specification:** `config/spec_v3.yaml`, status `FROZEN`, committed as
  `64b9feb`. It fixes the universe, the 30 specified features (27
  cross-sectional + 3 macro interactions), Screen-0 thresholds, the cost model,
  FDR/TA-FDR settings, the locked-OOS rule and the forward-window rule. The
  older `config/spec.yaml` is the Array-era v2 specification, kept for
  provenance only. Do not run against it.
- **Results manifest:** `results/manifest/manifest.json`. Every headline number
  in the manuscript traces to a manifest key, and `tools/check_coherence.py`
  asserts 22 of them directly against the compiled LaTeX. A stage that does not
  write to the manifest will have its numbers hand-transcribed, and those
  survive a full re-run silently, so every stage records.
- **Discovery-count tripwire:** `config/expected_fdr_counts.json` pins the
  accepted BH/BHY rejection counts per track. Unintended drift trips the wire;
  a deliberate change is a one-line edit with a written reason. The file carries
  the reason for every change to date.
- **Single-touch ledger:** `results/forward_touch_log.json` is append-only and
  records every scoring of the forward window, including the void first touch
  and the reason the window was re-scored.
- **Processed artefacts:** `data/processed/*.parquet` (per-feature FDR, per-fold
  IC, TA-FDR, selection-path PBO, OOS inference, baselines, sensitivity,
  ablation, Russell 2000, SHAP). These are gitignored and live on the X9 mirror.

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # pandas, numpy, scikit-learn, xgboost,
                                       # lightgbm, statsmodels, mapie, shap, ...
```

Python 3.14.3 on macOS 26.5.2 (Apple M2) for the recorded run; Python 3.10 or
later is sufficient. All inputs are public and free (Yahoo Finance via
`yfinance`, the Kenneth French Data Library, FRED). No API keys or paid data are
required. Exact pinned versions are in the manuscript's computational
environment table.

The formal component needs Lean 4 v4.16.0 with mathlib (`formal/`, built
sorry-free; `#print axioms` reports the standard three only).

## Repository to standard tree

| Standard component | Location in this repo | Contents |
|--------------------|-----------------------|----------|
| `data_processing/` | `src/data/`, `src/features/`, `config/spec_v3.yaml` | daily PIT universe build (`build_daily_pit.py`, `universe_v3.py`), Screen-0 data-integrity filter, 30-feature engineering (Tier-1 classic, Tier-2 crowding/intraday/macro-interactions), delisted-price recovery |
| `training/`        | `src/models/`, `src/calibration/` | 9-fold expanding-window walk-forward model suite (Ridge, Lasso, Logit, RF, XGB, LGBM, ensemble), MLP baseline, split-conformal and adaptive-conformal calibration |
| `backtesting/`     | `src/backtest/`, `src/baselines/` | cost-aware portfolio construction, square-root participation impact, calibrated spread/borrow costs (`cost_calibration.py`), cost scenario grid and ARV, baselines (momentum, reversal, naive ML, buy-and-hold), locked/exploratory/forward windows |
| `evaluation/`      | `src/fdr/`, `src/knockoffs/`, `src/diagnostics/`, `src/figures/`, `tools/` | BH / BHY (`run_fdr.py`, `run_bhy.py`), TA-FDR (`run_ta_fdr.py`), search-adjusted FDR (`run_search_adjusted_fdr.py`), selection-path PBO-CSCV + DSR (`run_pbo_selection_path.py`), placebo/embargo, cost-aware knockoffs, factor-adjusted alpha, figures, table export, coherence gate |
| outputs            | `results/`, `data/processed/`, `paper/` | manifest, touch log, parquet artefacts, manuscript and supplement |

`paper/` and `results/` are gitignored: this is a code repository, and the
manuscript and generated outputs live alongside it rather than inside it.

## Reproducing the headline results

One command runs everything downstream of the feature panel, in dependency
order, with per-stage freshness guards so a restart skips completed work:

```bash
bash scripts/run_v3_full_chain.sh
```

That driver calls the three stages, which can also be run individually:

1. **Walk-forward IC** (`src/models/run_walk_forward.py`), then
   **Stage A** (`scripts/run_v3_stage_a.sh`): SHAP, BH FDR, BHY, placebo,
   embargo. Stage A waits on the walk-forward by file freshness, not by process
   name: a `pgrep` wait-loop matches its own command line and hangs forever.
2. **Search-adjusted FDR** (`src/fdr/run_search_adjusted_fdr.py`), run outside
   Stage A because its tripwires are meant to stop the chain.
3. **Stage B** (`scripts/run_v3_stage_b.sh`): TA-FDR, selection-path PBO, cost
   scenarios, signals, backtest, sensitivity grid, baselines.
4. **Stage C** (`scripts/run_v3_stage_c.sh`): ex-mega-cap, regime, ablation, ML
   and MLP baselines, conformal, then `tools/record_battery.py`.
5. **Windows and capacity:** locked 2025 and exploratory 2022-2024 via
   `src/backtest/run_holdout.py`, AUM sweep, turnover decomposition, Russell
   2000 cross-universe, adaptive conformal.
6. **Recording and rendering:** `tools/record_synthesis.py`,
   `tools/regenerate_numbers.py`, `src/figures/make_figures.py`.

The forward holdout is **not** in the chain. `src/backtest/run_forward_holdout.py`
refuses to run unless the spec blob is committed, and every scoring appends to
`results/forward_touch_log.json`; a repeat scoring requires an explicit
`--reevaluation-reason`.

Approximate wall time for the full chain on an 8 GB M2: 5 to 8 hours, dominated
by the walk-forward and the TA-FDR bootstrap.

## What the run should produce

The manifest is the authority; these are the verdict-level values for spec v3.0
so that a re-run can be checked at a glance.

| Screen | Track A | Track B |
|--------|---------|---------|
| BH / BHY discoveries (q=0.10, m=30) | 14 / 13 | 1 / 0 |
| Search-adjusted (enlarged m'=57) | 16 / 57 | 0 / 57 |
| TA-FDR tradable (3 nulls) | 0 / 30 | 0 / 30 |

Deployed book: in-sample net Sharpe +0.277 (gross +0.399, turnover 25.9x, cost
drag 92 bps). Exploratory 2022-2024 net Sharpe -0.540, classified practically
null. Locked 2025 net Sharpe -0.190 over 144 days, inconclusive. Forward
2025-08 to 2026-03 net Sharpe -0.204 over 167 days, inconclusive.
Selection-path PBO 0.458, DSR 0.041, ARV at SR* = 0.5 is 0%.

PBO over 54 configurations is unstable: a correction worth about 2% of composite
weight moved it from 0.672 to 0.458. Read it as an order of magnitude, not to
three decimals. The verdict does not rest on it.

## Verification gates

```bash
pytest -q                                   # 212 tests
python3 tools/check_coherence.py            # 22 assertions, manuscript == manifest
cd formal && lake build                     # Lean, 16 theorems, sorry-free
```

`tools/check_anonymity.py` applies only to the anonymised build used for the
earlier double-blind submissions. The Results in Engineering submission is
single-anonymized and named, so that gate is not part of the current path.

## Release policy

The repository is **public** at `github.com/pdwi2020/cost-aware-alpha-paper`.
A version-tagged, DOI-archived release accompanies acceptance (see the
data-availability statement in the manuscript). Raw 1-minute intraday inputs,
processed parquet artefacts and copyrighted reference PDFs are excluded from the
repository for size and licensing reasons; all results regenerate from the
public free data sources listed above.
