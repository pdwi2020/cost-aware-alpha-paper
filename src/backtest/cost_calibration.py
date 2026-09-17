"""Calibrated execution costs: stock-day spreads, impact grid, borrow tiers.

Why this exists
---------------
Reviewer 1 (point 7): "The transaction-cost conclusion depends on unvalidated
assumptions. A fixed 3 bps half-spread, a single impact coefficient of 0.10,
and zero borrow cost are applied across firms, dates, and market conditions ...
an uncalibrated cost model cannot support the strength of the stated verdict."

This module replaces the three constants with estimates, using only free daily
data (no TAQ, no execution records):

* **Spreads** -- two published estimators that recover effective spreads from
  daily high/low/close prices: Abdi and Ranaldo (2017) as primary and Corwin
  and Schultz (2012) as a cross-check. Both are computed per stock-day, rolled
  over a window, and lagged one day so nothing uses trade-date information.
* **Impact** -- a grid of coefficients anchored in published estimates rather
  than a single value, from Frazzini et al.'s low realised costs through
  Almgren et al. (2005) to square-root-law prefactors of order one
  (Toth et al. 2011).
* **Borrow** -- a general-collateral fee for easy-to-borrow names with a
  hard-to-borrow tier, assigned by a lagged, observable proxy.

Direction of the argument
-------------------------
The paper's verdict is that a statistically real signal is NOT tradable. A
"not tradable" claim is only strong if it survives *optimistic* costs, so the
original fixed parameters (3 bps half-spread, eta = 0.10, zero borrow) are
retained as the optimistic bound, and the calibrated model is reported
alongside. Daily high-low estimators are known to overstate spreads for very
liquid large caps, which is why they are used as the conservative (upper) cost
scenario and never as the sole basis for a tradability claim.

All outputs are date x ticker frames aligned to the caller's panel, ready to
pass to ``PortfolioSimulator.simulate_pnl`` as ``spread_bps_matrix`` and
``borrow_bps_matrix``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Impact-coefficient grid with literature anchors (eta in the
# eta * sigma * sqrt(participation) participation model).
IMPACT_GRID: dict[float, str] = {
    0.05: "optimistic; below Frazzini et al. (2018) realised institutional costs",
    0.10: "paper base case (original manuscript)",
    0.142: "Almgren et al. (2005) direct estimation of equity market impact",
    0.30: "conservative",
    0.50: "lower end of square-root-law prefactors (Toth et al. 2011)",
    1.00: "upper end of square-root-law prefactors (Toth et al. 2011)",
}

# Annual borrow fees in basis points.
BORROW_GC_BPS = 25.0     # general collateral; D'Avolio (2002) value-weighted avg ~25bp
BORROW_HTB_BPS = 300.0   # hard-to-borrow tier
BORROW_HTB_STRESS_BPS = 500.0

# Spread post-processing bounds (bps, half-spread).
SPREAD_FLOOR_BPS = 0.5
SPREAD_CAP_BPS = 250.0
DEFAULT_WINDOW = 21


def _wide(panel: pd.DataFrame, column: str) -> pd.DataFrame:
    """Pivot a long OHLCV frame to date x ticker for one column."""
    return panel.pivot(index="date", columns="ticker", values=column).sort_index()


# ---------------------------------------------------------------------------
# Corwin & Schultz (2012) high-low spread estimator
# ---------------------------------------------------------------------------

def corwin_schultz_spread(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    overnight_adjust: bool = True,
) -> pd.DataFrame:
    """Two-day high-low proportional (round-trip) spread, per stock-day.

    Implements Corwin and Schultz (2012): with beta the sum of two consecutive
    daily squared log high-low ranges and gamma the squared log range over the
    two-day window,

        alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt 2)
                - sqrt(gamma / (3 - 2 sqrt 2))
        S     = 2 (exp(alpha) - 1) / (1 + exp(alpha))

    Negative estimates are set to zero, as the authors recommend. When
    ``overnight_adjust`` is true, day t+1's high and low are shifted by any
    overnight gap relative to day t's close, so the estimator measures
    intraday rather than overnight price movement.

    Returns a date x ticker frame of proportional round-trip spreads (decimal),
    with the estimate placed on the SECOND day of each pair.
    """
    h, l, c = high.astype(float), low.astype(float), close.astype(float)

    h1, l1 = h.shift(1), l.shift(1)
    c1 = c.shift(1)

    if overnight_adjust:
        # If the day-t close lies outside day t+1's range, shift t+1's range.
        gap_up = (l - c1).where(l > c1, 0.0)
        gap_dn = (c1 - h).where(h < c1, 0.0)
        h = h - gap_up + gap_dn
        l = l - gap_up + gap_dn

    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(h / l) ** 2
        hl1 = np.log(h1 / l1) ** 2
        beta = hl + hl1
        h2 = pd.DataFrame(np.maximum(h.to_numpy(), h1.to_numpy()), index=h.index, columns=h.columns)
        l2 = pd.DataFrame(np.minimum(l.to_numpy(), l1.to_numpy()), index=l.index, columns=l.columns)
        gamma = np.log(h2 / l2) ** 2

        k = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

    spread = spread.where(np.isfinite(spread))
    return spread.clip(lower=0.0)


# ---------------------------------------------------------------------------
# Abdi & Ranaldo (2017) close-high-low estimator
# ---------------------------------------------------------------------------

def abdi_ranaldo_spread(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Rolling close-high-low proportional (round-trip) spread, per stock-day.

    Implements Abdi and Ranaldo (2017): with c the log close and eta the log
    mid-range (log high + log low) / 2,

        S^2 = 4 E[(c_t - eta_t)(c_t - eta_{t+1})]

    estimated as a rolling mean over ``window`` days, with negative values set
    to zero. Returns a date x ticker frame of proportional round-trip spreads
    (decimal).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.log(close.astype(float))
        eta = (np.log(high.astype(float)) + np.log(low.astype(float))) / 2.0

    term = (c - eta) * (c - eta.shift(-1))
    term = term.where(np.isfinite(term))

    mp = min_periods if min_periods is not None else max(5, window // 2)
    rolled = term.rolling(window, min_periods=mp).mean()
    return (4.0 * rolled).clip(lower=0.0) ** 0.5


# ---------------------------------------------------------------------------
# Half-spread matrix in bps (lagged, smoothed, bounded)
# ---------------------------------------------------------------------------

def half_spread_bps_matrix(
    panel: pd.DataFrame,
    estimator: str = "abdi_ranaldo",
    window: int = DEFAULT_WINDOW,
    floor_bps: float = SPREAD_FLOOR_BPS,
    cap_bps: float = SPREAD_CAP_BPS,
    fallback_bps: float = 3.0,
) -> tuple[pd.DataFrame, dict]:
    """Stock-day HALF-spread in bps, lagged one day, for ``simulate_pnl``.

    The estimators return proportional round-trip spreads, so the half-spread
    is half of that. The series is rolled (median for Corwin-Schultz, which is
    noisy day to day), shifted one day, bounded, and missing cells fall back to
    ``fallback_bps`` so the matrix is always complete.

    Returns (matrix, diagnostics).
    """
    high, low, close = (_wide(panel, c) for c in ("high", "low", "close"))

    # `lag` is set by what each estimator's last usable observation needs:
    #   Corwin-Schultz at date t uses days t-1 and t, both known at t's close,
    #     so the estimate is usable from t+1: lag = 1.
    #   Abdi-Ranaldo's term at date t multiplies (c_t - eta_t) by (c_t -
    #     eta_{t+1}), so a window ending at t is only known at t+1's close and
    #     is usable from t+2: lag = 2. Shifting by 1 would let the spread paid
    #     on the trade date depend on that date's own high and low.
    if estimator == "abdi_ranaldo":
        proportional = abdi_ranaldo_spread(high, low, close, window=window)
        lag = 2
    elif estimator == "corwin_schultz":
        raw = corwin_schultz_spread(high, low, close)
        proportional = raw.rolling(window, min_periods=max(5, window // 2)).median()
        lag = 1
    else:
        raise ValueError(f"Unknown estimator {estimator!r}")

    half_bps = (proportional / 2.0) * 10_000.0
    half_bps = half_bps.shift(lag)                    # look-ahead free

    n_est = int(half_bps.notna().sum().sum())
    n_floor = int((half_bps < floor_bps).sum().sum())
    n_cap = int((half_bps > cap_bps).sum().sum())
    bounded = half_bps.clip(lower=floor_bps, upper=cap_bps)
    n_missing = int(bounded.isna().sum().sum())
    filled = bounded.fillna(fallback_bps)

    diagnostics = {
        "estimator": estimator,
        "lag_days": lag,
        "window": window,
        "n_estimated_cells": n_est,
        "n_missing_filled": n_missing,
        "pct_floored": round(100.0 * n_floor / max(n_est, 1), 3),
        "pct_capped": round(100.0 * n_cap / max(n_est, 1), 3),
        "median_bps": float(np.nanmedian(bounded.to_numpy())) if n_est else float("nan"),
        "p25_bps": float(np.nanpercentile(bounded.to_numpy(), 25)) if n_est else float("nan"),
        "p75_bps": float(np.nanpercentile(bounded.to_numpy(), 75)) if n_est else float("nan"),
        "fallback_bps": fallback_bps,
    }
    return filled, diagnostics


# ---------------------------------------------------------------------------
# Borrow-fee matrix in annual bps (lagged proxy for hard-to-borrow)
# ---------------------------------------------------------------------------

def borrow_bps_matrix(
    panel: pd.DataFrame,
    adv_dollars: pd.DataFrame,
    gc_bps: float = BORROW_GC_BPS,
    htb_bps: float = BORROW_HTB_BPS,
    adv_decile_cut: float = 0.10,
    price_floor: float = 10.0,
    crash_threshold: float = -0.30,
) -> tuple[pd.DataFrame, dict]:
    """Annual borrow fee per stock-day: general collateral plus an HTB tier.

    A name is flagged hard-to-borrow on day t when, using information known at
    t-1, any of the following holds: its trailing dollar volume is in the
    bottom ``adv_decile_cut`` of that day's cross-section, its close is below
    ``price_floor``, or its trailing 21-day return is below
    ``crash_threshold``. These are the observable conditions under which
    borrow gets expensive; without a securities-lending data feed they stand in
    for the fee itself, and the stress tier is reported separately.

    Returns (matrix in annual bps, diagnostics).
    """
    close = _wide(panel, "close")
    adv = adv_dollars.reindex(index=close.index, columns=close.columns)

    adv_lag = adv.shift(1)
    close_lag = close.shift(1)
    ret_21 = close.pct_change(21).shift(1)

    adv_rank = adv_lag.rank(axis=1, pct=True)
    htb = (
        (adv_rank <= adv_decile_cut)
        | (close_lag < price_floor)
        | (ret_21 < crash_threshold)
    ).fillna(False)

    matrix = pd.DataFrame(gc_bps, index=close.index, columns=close.columns)
    matrix = matrix.where(~htb, htb_bps)

    n_cells = int(htb.size)
    diagnostics = {
        "gc_bps": gc_bps,
        "htb_bps": htb_bps,
        "pct_cells_htb": round(100.0 * float(htb.to_numpy().sum()) / max(n_cells, 1), 3),
        "rule": (
            f"HTB if trailing-ADV pct-rank <= {adv_decile_cut} OR lagged close < "
            f"${price_floor} OR trailing 21d return < {crash_threshold:.0%} "
            "(all lagged one day)"
        ),
    }
    return matrix, diagnostics


def cost_scenarios() -> dict[str, dict]:
    """The scenarios the paper reports, from optimistic to conservative."""
    return {
        "optimistic_fixed": {
            "spread": "fixed 3 bps half-spread",
            "impact_coeff": 0.10,
            "borrow_bps": 0.0,
            "note": "the original manuscript's parameters; retained as the optimistic bound",
        },
        "calibrated": {
            "spread": "Abdi-Ranaldo stock-day half-spread (lagged, 21d window)",
            "impact_coeff": 0.142,
            "borrow_bps": f"{BORROW_GC_BPS} GC / {BORROW_HTB_BPS} HTB",
            "note": "primary calibrated scenario",
        },
        "conservative": {
            "spread": "Corwin-Schultz stock-day half-spread (lagged, 21d median)",
            "impact_coeff": 0.50,
            "borrow_bps": f"{BORROW_GC_BPS} GC / {BORROW_HTB_STRESS_BPS} HTB stress",
            "note": "upper cost bound; square-root-law prefactor",
        },
    }
