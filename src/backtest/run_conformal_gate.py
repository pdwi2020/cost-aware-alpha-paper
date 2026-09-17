"""B3b — Conformal risk-gate experiment for Track B OOS (2022-2024).

Tests whether conformal prediction can earn a load-bearing role as a
risk overlay. Two pre-specified variants (design spec: docs/design_conformal_gate.md).

Variant 1 — Regime-conditional leverage (deterministic):
    m_t = clip(q̂_calm / q̂(regime_t), 0.5, 1.0)
    regime_t = stressed if VIX_t > 20 else calm
    q̂_calm, q̂_stressed = IS-fold means from conformal_coverage.parquet (frozen)
    VIX same-day → applied to next-day positions via existing 1-day lag (no lookahead)

Variant 2 — Breach-rate distribution-shift gate:
    q̂_daily = IS-calibrated 90%-coverage threshold on daily portfolio net P&L
             = 1.645 × IS_daily_std  (from backtest_base.parquet, IS-only)
    breach_τ = 1 if |net_pnl_τ| > q̂_daily  else 0
    covrate_t = 1 − mean(breach over trailing W=63 days, lagged ≥5 days)
    m_t = r_reduce=0.5 if covrate_t < c_trigger=0.80 else 1.0
    All constants fixed IS; no OOS degrees of freedom.

P&L scaling: gated_pnl ≈ m_t × ungated_pnl (linear scaling approximation;
  de-risking also lowers turnover/cost, so this understates gating benefit).

Decision criterion (B3c — Opus):
    KEEP iff Calmar improves AND net SR not reduced by more than ~0.05.
    DEMOTE otherwise → standardise paper to "three disciplines + diagnostic".

Guardrail assertions:
    1. IS-only calibration: q̂, q̂_calm, q̂_stressed, and all constants from 2013-2021.
    2. No lookahead: Variant 1 uses same-day VIX→next-day exposure (1-day lag).
                    Variant 2 uses breach lagged ≥5 days.
    3. No OOS degrees of freedom: W, c_trigger, r_reduce, m_min are fixed here.
    4. Single-touch preserved: overlay applies to the already-locked signal.

Inputs:
    data/processed/holdout_pnl_track_b.parquet
    data/processed/conformal_coverage.parquet
    data/processed/features_all.parquet     (VIX column for Variant 1)
    data/processed/backtest_base.parquet    (IS daily P&L stats for Variant 2)

Output:
    data/processed/conformal_gate.parquet
    (cols: variant, net_sr, gross_sr, ann_net_return, max_dd, calmar,
           cost_drag_bps, frac_days_derisked)

Run:
    python3 -u src/backtest/run_conformal_gate.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.manifest import get  # noqa: E402

DATA = ROOT / "data" / "processed"
OUT  = DATA / "conformal_gate.parquet"

ANN            = 252.0
VIX_THRESHOLD  = 20.0   # VIX > 20 → stressed regime
M_MIN          = 0.5    # floor on leverage multiplier

# Variant 2 constants (pre-specified, IS-justified)
W          = 63    # trailing breach-rate window (trading days)
C_TRIGGER  = 0.80  # trigger de-risk when coverage drops below this
R_REDUCE   = 0.50  # exposure multiplier when triggered


def sharpe_ann(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 1e-12 else np.nan


def max_drawdown(r: np.ndarray) -> float:
    cum = np.cumsum(r)
    return float((cum - np.maximum.accumulate(cum)).min())


def calmar(r: np.ndarray) -> float:
    mdd = abs(max_drawdown(r))
    return float(r.mean() * ANN / mdd) if mdd > 1e-12 else np.nan


def compute_metrics(pnl: pd.DataFrame, m: pd.Series, label: str) -> dict:
    net   = (pnl["net_pnl"]   * m).to_numpy(float)
    gross = (pnl["gross_pnl"] * m).to_numpy(float)
    cost  = (pnl["total_cost"] * m).to_numpy(float)
    return {
        "variant":           label,
        "net_sr":            round(sharpe_ann(net),  3),
        "gross_sr":          round(sharpe_ann(gross), 3),
        "ann_net_return":    round(float(net.mean() * ANN * 100), 2),
        "max_dd":            round(max_drawdown(net) * 100, 2),
        "calmar":            round(calmar(net), 3),
        "cost_drag_bps":     round(float(cost.mean() * ANN * 10_000), 1),
        "frac_days_derisked": round(float((m < 1.0).mean()), 3),
    }


def main():
    # ── Load inputs ─────────────────────────────────────────────────────────
    pnl = pd.read_parquet(DATA / "holdout_pnl_track_b.parquet")
    pnl.index = pd.to_datetime(pnl.index)

    cov = pd.read_parquet(DATA / "conformal_coverage.parquet")
    feat = pd.read_parquet(DATA / "features_all.parquet")
    is_metrics = pd.read_parquet(DATA / "backtest_base.parquet")

    # ── Acceptance check: ungated must match the recorded headline ──────────
    # This compared against a hardcoded +0.357, the Array-era value. The
    # constant went stale and blocked the stage entirely rather than flagging
    # drift, which is the failure mode a guard is supposed to prevent. It now
    # reads the expectation from the manifest, so it tracks the pipeline.
    net_ungated = pnl["net_pnl"].to_numpy(float)
    sr_ungated  = sharpe_ann(net_ungated)
    expected = get("window.locked_oos_2025.track_b.net_sharpe")
    if expected is None:
        raise SystemExit("manifest has no window.locked_oos_2025.track_b.net_sharpe; "
                         "run src/backtest/run_holdout.py first")
    assert abs(sr_ungated - expected) < 0.02, (
        f"Ungated net SR={sr_ungated:.3f} != manifest {expected:.3f} +/- 0.02 "
        f"— the P&L file and the manifest describe different runs"
    )

    # ── Variant 1 parameters (IS-only, frozen) ───────────────────────────────
    tb_cov          = cov[cov.track == "track_b"]
    q_calm_mean     = float(tb_cov["q_hat_calm"].mean())
    q_stressed_mean = float(tb_cov["q_hat_stressed"].dropna().mean())
    m_stressed_v1   = float(np.clip(q_calm_mean / q_stressed_mean, M_MIN, 1.0))
    print(f"V1 IS params: q̂_calm={q_calm_mean:.4f}  q̂_stressed={q_stressed_mean:.4f}"
          f"  m_stressed={m_stressed_v1:.4f}")

    # VIX time series over the evaluated window — same-day, no lookahead via
    # the 1-day position lag.
    #
    # This was hardcoded to 2022-01-01..2024-12-31 while the P&L it gates is
    # the locked 2025 window. Reindexing a 2022-24 series onto 2025 dates gives
    # all-NaN, so `vix > 20` was False on every day and Variant 1 never fired.
    # The "Variant 1 lowers net Sharpe" result was an artefact of that, not a
    # finding. The slice now follows the P&L.
    dv = feat.index.get_level_values("date")
    win = feat[(dv >= pnl.index.min()) & (dv <= pnl.index.max())]
    vix = win.groupby(level="date")["vix"].first()
    vix.index = pd.to_datetime(vix.index)
    vix = vix.reindex(pnl.index).ffill()

    # Guardrail: VIX is same-day published (no OOS lookahead)
    # m_v1 applied to day-t positions → realised at day t+1 (existing 1-day lag)
    regime_stressed = (vix > VIX_THRESHOLD).astype(float)
    m_v1 = pd.Series(
        np.where(regime_stressed == 1.0, m_stressed_v1, 1.0),
        index=pnl.index,
        name="m_v1",
    ).clip(lower=M_MIN, upper=1.0)

    # ── Variant 2 parameters (IS-only, frozen) ───────────────────────────────
    is_tb = is_metrics[is_metrics.track == "track_b"].iloc[0]
    is_daily_mean = is_tb["net_pnl_annual"] / ANN
    is_daily_std  = abs(is_daily_mean) * np.sqrt(ANN) / abs(is_tb["net_pnl_sharpe"])
    q_hat_daily   = 1.645 * is_daily_std   # 90%-coverage threshold (two-sided)
    print(f"V2 IS params: is_daily_std={is_daily_std:.6f}  q̂_daily={q_hat_daily:.6f}")
    print(f"  W={W}  c_trigger={C_TRIGGER}  r_reduce={R_REDUCE}")

    # Guardrail: breach lagged ≥5 days
    breach = (pnl["net_pnl"].abs() > q_hat_daily).astype(float)
    assert breach.index.equals(pnl.index), "Index alignment error"

    breach_lagged = breach.shift(5)   # ≥5-day lag — no lookahead
    covrate = 1.0 - breach_lagged.rolling(W, min_periods=max(1, W // 4)).mean()
    m_v2 = covrate.apply(lambda c: R_REDUCE if c < C_TRIGGER else 1.0).rename("m_v2")

    # Guardrail: no OOS data was used to set W, C_TRIGGER, R_REDUCE, M_MIN
    assert all(v in (W, C_TRIGGER, R_REDUCE, M_MIN)
               for v in (W, C_TRIGGER, R_REDUCE, M_MIN))  # tautological but documents intent

    # ── Compute metrics ──────────────────────────────────────────────────────
    ungated_m = pd.Series(1.0, index=pnl.index)
    rows = [
        compute_metrics(pnl, ungated_m, "ungated"),
        compute_metrics(pnl, m_v1,      "gated_v1"),
        compute_metrics(pnl, m_v2,      "gated_v2"),
    ]
    results = pd.DataFrame(rows)
    results.to_parquet(OUT, index=False)

    pd.set_option("display.width", 160)
    print("\n=== Conformal Gate Results ===")
    print(results.to_string(index=False))

    # ── Decision summary ─────────────────────────────────────────────────────
    ungated = results[results.variant == "ungated"].iloc[0]
    print("\n=== B3c Decision ===")
    for var in ["gated_v1", "gated_v2"]:
        g = results[results.variant == var].iloc[0]
        calmar_improves = g["calmar"] > ungated["calmar"]
        sr_ok = (g["net_sr"] - ungated["net_sr"]) > -0.05
        keep = calmar_improves and sr_ok
        print(f"  {var}: Calmar {ungated['calmar']:.3f}→{g['calmar']:.3f} "
              f"({'↑' if calmar_improves else '↓'}), "
              f"net SR {ungated['net_sr']:.3f}→{g['net_sr']:.3f} "
              f"({'OK' if sr_ok else 'DEGRADED'}), "
              f"verdict={'KEEP' if keep else 'DEMOTE'}")
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
