"""Split conformal prediction intervals for return forecasts.

Reference: Vovk, Gammerman & Shafer (2005). "Algorithmic Learning in a Random World."
           Angelopoulos & Bates (2023). "Conformal Risk Control." ICLR.

Split conformal (Papadopoulos et al. 2002):
  1. Fit model on training set.
  2. Calibrate: compute residuals on held-out calibration set.
  3. q_hat = (1-α)(1+1/n) quantile of |residuals|.
  4. Interval for new x*: [ŷ(x*) - q_hat, ŷ(x*) + q_hat].

Marginal coverage guarantee (exchangeability): P(Y ∈ CI) ≥ 1-α.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class ConformalIntervals:
    """Split conformal prediction intervals at coverage level 1-alpha.

    Calibration set: 20% of each walk-forward training window (held-out).
    Coverage guarantee: P(y ∈ [ŷ - q̂, ŷ + q̂]) ≥ 1 - alpha.
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha  = alpha
        self.q_hat: Optional[float] = None

    def calibrate(
        self,
        cal_predictions: np.ndarray,
        cal_targets: np.ndarray,
    ) -> float:
        """Fit conformal quantile on calibration set.

        Args:
            cal_predictions: Predicted returns on calibration set. Shape (n,).
            cal_targets:     True returns on calibration set. Shape (n,).

        Returns:
            q_hat: The ⌈(1-alpha)(1+1/n)⌉ quantile of absolute residuals.
        """
        residuals = np.abs(cal_targets - cal_predictions)
        n         = len(residuals)
        level     = min((1.0 - self.alpha) * (1.0 + 1.0 / n), 1.0)
        self.q_hat = float(np.quantile(residuals, level))
        return self.q_hat

    def predict_interval(
        self,
        predictions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return conformal prediction interval [ŷ - q̂, ŷ + q̂].

        Args:
            predictions: Point forecasts. Shape (n,).

        Returns:
            (lower, upper) arrays of shape (n,).
        """
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() before predict_interval()")
        lower = predictions - self.q_hat
        upper = predictions + self.q_hat
        return lower, upper

    def conformal_position_size(
        self,
        signals: np.ndarray,
        interval_widths: np.ndarray,
        gross_target: float = 1.0,
        max_single: float = 0.05,
        max_gross: float = 2.0,
    ) -> np.ndarray:
        """Conformal-leverage position sizing.

        Scales signal by prediction confidence (narrower interval → larger bet).
        Equivalent to Kelly-scaled positions where interval_width is uncertainty.

        raw_score_i = signal_i / interval_width_i
        w_i = raw_score_i / sum(|raw_score_j|) × gross_target, then capped.

        Args:
            signals:         Point-forecast returns (N,). Signed.
            interval_widths: 2 × q̂ (scalar or per-sample array).
            gross_target:    Target gross exposure (default 1.0x).
            max_single:      Max single-name position (default 5%).
            max_gross:       Hard gross cap (default 2x).

        Returns:
            Position weights (N,) with |w|.sum() ≤ max_gross.
        """
        widths = np.asarray(interval_widths, dtype=float)
        if widths.ndim == 0:
            widths = np.full(len(signals), float(widths))
        widths = np.where(widths > 1e-12, widths, 1e-12)   # guard divide-by-zero

        raw_score = signals / widths
        gross     = np.abs(raw_score).sum()
        if gross < 1e-12:
            return np.zeros_like(signals, dtype=float)

        w = raw_score / gross * gross_target
        w = np.clip(w, -max_single, max_single)

        # Rescale if caps caused gross to exceed max_gross
        gross2 = np.abs(w).sum()
        if gross2 > max_gross:
            w = w / gross2 * max_gross

        return w

    def empirical_coverage(
        self,
        test_predictions: np.ndarray,
        test_targets: np.ndarray,
    ) -> float:
        """Fraction of test samples where true target falls inside the interval."""
        lower, upper = self.predict_interval(test_predictions)
        return float(np.mean((lower <= test_targets) & (test_targets <= upper)))


def calibrate_per_fold(
    fold_preds: pd.DataFrame,
    fold_targets: pd.Series,
    cal_fraction: float = 0.2,
    alpha: float = 0.10,
) -> "ConformalIntervals":
    """Convenience: split trailing cal_fraction of a fold into calibration set.

    Args:
        fold_preds:   Predictions on training period, indexed by date.
        fold_targets: True targets on training period, indexed by date.
        cal_fraction: Fraction of training period reserved for calibration.
        alpha:        Miscoverage rate.

    Returns:
        Fitted ConformalIntervals object.
    """
    n_cal = max(int(len(fold_preds) * cal_fraction), 1)
    cal_preds  = fold_preds.values[-n_cal:]
    cal_tgts   = fold_targets.values[-n_cal:]
    valid      = ~(np.isnan(cal_preds) | np.isnan(cal_tgts))

    ci = ConformalIntervals(alpha=alpha)
    if valid.sum() >= 10:
        ci.calibrate(cal_preds[valid], cal_tgts[valid])
    else:
        # Fallback: set q_hat to the unconditional std of targets
        ci.q_hat = float(np.nanstd(cal_tgts))
    return ci
