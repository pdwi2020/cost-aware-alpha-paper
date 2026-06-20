# CAVAL — Formal Verification Layer (Lean 4 / mathlib)

Machine-checked proofs of the core statistical guarantees behind the CAVAL protocol
(the manuscript *Cost-Aware Alpha Generation with FDR Control and
Market-Impact-Aware Validation*). Every theorem below is verified by the Lean 4
kernel and depends only on the three standard mathlib axioms
(`propext`, `Classical.choice`, `Quot.sound`) — there are no `sorry`s, `admit`s, or
extra axioms (see `Check.lean`).

## Toolchain
- Lean `v4.16.0`, mathlib `v4.16.0` (pinned in `lean-toolchain` / `lakefile.toml`).
- Reproduce: `lake exe cache get && lake build`.
- Audit the axiom base: `lake env lean Check.lean` (each theorem must report
  `depends on axioms: [propext, Classical.choice, Quot.sound]`).

> macOS 26 (Tahoe) note: the v4.16.0-built `cache` binary needs a one-time Mach-O
> fix (set `SG_READ_ONLY` on `__DATA_CONST`, then ad-hoc re-sign) before
> `lake exe cache get` will run. See the paper's reproducibility appendix.

## Theorem ↔ manuscript map

| Lean theorem (`CavalFormal/…`) | Manuscript object | What it certifies |
|---|---|---|
| `Costs.cSpread_nonneg`, `Costs.cImpact_nonneg` | Eqs. (spread/impact), §Cost-Aware Portfolio Construction | cost components are non-negative |
| `Costs.net_le_gross` | net/gross P&L identity | net P&L ≤ gross P&L when costs ≥ 0 |
| `Costs.impactAvg_mono` | §FDR, prose on Almgren–Chriss permutation null | **average market-impact cost is monotone in turnover** — high-turnover signals are penalised more (the test is "automatically more conservative for high-frequency signals"), now a theorem rather than an assertion |
| `ARV.arv_antitone` | Eq. (ARV), §Cost-Robustness | Alpha Robustness Volume is non-increasing in the Sharpe threshold |
| `FDR.bhy_thr_le_bh_thr`, `FDR.bhy_subset_bh` | Eqs. (BH), (BHY) | the BHY rejection set ⊆ the BH rejection set — BHY is uniformly more conservative |
| `TAFdr.net_stat_le_gross`, `TAFdr.pval_antitone`, `TAFdr.tafdr_reject_subset`, `TAFdr.tafdr_conservative` | §TA-FDR (novel) | **cost subtraction ⇒ smaller net statistic ⇒ larger permutation p-value ⇒ the cost-aware (TA-FDR) rejection set ⊆ the cost-blind rejection set** — the protocol's cost-awareness is provably conservative |
| `Conformal.split_conformal_coverage` | Eq. (conformal quantile), §Conformal Prediction Calibration | **split-conformal marginal coverage ≥ 1 − α** under exchangeability (the test point's rank is uniform on the `n+1` scores) |

## NOT formalised here (cited to the literature in the manuscript)
- Full Benjamini–Hochberg / Benjamini–Yekutieli FDR control `E[FDP] ≤ q` (the
  *value* of the bound; we verify only that BHY is more conservative than BH).
- Super-uniformity of the `(1+#)/(1+B)` permutation p-value.
- Deflated Sharpe Ratio and the `t`-statistic sampling distribution.

These remain ordinary citations — they are deliberately **not** labelled
"machine-checked" anywhere in the paper.

## Files
- `CavalFormal/Costs.lean`, `ARV.lean`, `FDR.lean`, `TAFdr.lean`, `Conformal.lean`
- `CavalFormal.lean` — root, imports all modules.
- `Check.lean` — `#print axioms` audit of every theorem.
