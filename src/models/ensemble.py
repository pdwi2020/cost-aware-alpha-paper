"""Week 5: Ensemble weighting utilities.

Provides IC-weighted and shrinkage-adjusted ensemble weights derived from
walk-forward IC history (ic_by_fold.parquet). Weights are computed using an
expanding window so fold t only uses ICs from folds 1…t-1 (no lookahead).

Usage:
    from src.models.ensemble import EnsembleWeighter
    ew = EnsembleWeighter(track="track_b")
    weights_2017 = ew.weights_for_fold("2017")      # pd.Series(model → weight)
    pred = ew.blend(preds_dict, fold_id="2017")     # np.ndarray
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent.parent
IC_PATH = ROOT / "data" / "processed" / "ic_by_fold.parquet"

MODEL_NAMES = ["ridge", "lasso", "logistic", "rf", "xgb", "lgbm"]


# ---------------------------------------------------------------------------
# Weight computation helpers
# ---------------------------------------------------------------------------

def _ic_to_weights(ic_series: pd.Series, shrinkage: float = 0.5) -> pd.Series:
    """Convert past-fold mean ICs to ensemble weights with shrinkage.

    Only positive-IC models receive weight; weight is proportional to IC.
    shrinkage=0 → pure IC-weighted; shrinkage=1 → equal-weighted.
    """
    ic = ic_series.fillna(0.0).clip(lower=0.0)
    n  = len(ic)
    w_eq = pd.Series(1.0 / n, index=ic.index)
    if ic.sum() < 1e-8:
        return w_eq
    w_ic = ic / ic.sum()
    return (1.0 - shrinkage) * w_ic + shrinkage * w_eq


def expanding_weights(
    ic_df: pd.DataFrame,
    track: str = "track_b",
    shrinkage: float = 0.5,
    min_history: int = 2,
) -> pd.DataFrame:
    """Compute per-fold ensemble weights using only past ICs (no lookahead).

    Returns DataFrame indexed by fold_id, columns = model names.
    """
    sub    = ic_df[ic_df["track"] == track].set_index("fold")
    models = [c for c in MODEL_NAMES if c in sub.columns]
    folds  = sorted(sub.index.tolist())

    rows = []
    for i, fold in enumerate(folds):
        if i < min_history:
            w = pd.Series(1.0 / len(models), index=models)
        else:
            past_mean_ic = sub.loc[folds[:i], models].mean()
            w = _ic_to_weights(past_mean_ic, shrinkage=shrinkage)
        rows.append({"fold": fold, **w.to_dict()})

    return pd.DataFrame(rows).set_index("fold")


# ---------------------------------------------------------------------------
# EnsembleWeighter class
# ---------------------------------------------------------------------------

class EnsembleWeighter:
    """Stateful ensemble weighter loaded from ic_by_fold.parquet."""

    def __init__(
        self,
        track: str = "track_b",
        shrinkage: float = 0.5,
        ic_path: str | Path = IC_PATH,
    ):
        self.track     = track
        self.shrinkage = shrinkage
        ic_df          = pd.read_parquet(ic_path)
        self._weights  = expanding_weights(ic_df, track=track, shrinkage=shrinkage)
        self._ic_df    = ic_df[ic_df["track"] == track].set_index("fold")

    def weights_for_fold(self, fold_id: str) -> pd.Series:
        """Return model weights for a specific fold (based on all prior folds)."""
        if fold_id not in self._weights.index:
            raise KeyError(f"fold_id '{fold_id}' not in weight table")
        return self._weights.loc[fold_id]

    def blend(self, preds: dict, fold_id: str) -> np.ndarray:
        """IC-weighted blend of model predictions for given fold.

        preds: {model_name: np.ndarray of shape (N,)}
        """
        weights = self.weights_for_fold(fold_id)
        out     = np.zeros(len(next(iter(preds.values()))), dtype=np.float64)
        total   = 0.0
        for model, w in weights.items():
            if model in preds and w > 0:
                out  += w * preds[model].astype(np.float64)
                total += w
        return out / total if total > 0 else out

    def summary(self) -> pd.DataFrame:
        """Return weight table for inspection."""
        return self._weights.copy()

    def print_summary(self) -> None:
        print(f"\nEnsemble weights — {self.track.upper()}  (shrinkage={self.shrinkage})")
        print(self._weights.round(3).to_string())


# ---------------------------------------------------------------------------
# Rank-average ensemble (model-agnostic baseline)
# ---------------------------------------------------------------------------

def rank_average(preds: dict) -> np.ndarray:
    """Average cross-sectional ranks across all models (equal weight).

    More robust than averaging raw predictions when models have different scales.
    """
    from scipy.stats import rankdata
    stacked = np.stack([
        rankdata(p, method="average") for p in preds.values()
    ], axis=0)
    return stacked.mean(axis=0)


# ---------------------------------------------------------------------------
# IC stability analysis
# ---------------------------------------------------------------------------

def ic_stability_report(ic_path: str | Path = IC_PATH) -> pd.DataFrame:
    """Compute per-model IC statistics across folds for both tracks.

    Returns DataFrame with mean, std, hit_rate (% positive folds), IR (mean/std).
    """
    ic_df  = pd.read_parquet(ic_path)
    models = [c for c in MODEL_NAMES + ["ensemble"] if c in ic_df.columns]
    rows   = []

    for track in ["track_b", "track_a"]:
        sub = ic_df[ic_df["track"] == track][models]
        for model in models:
            vals = sub[model].dropna()
            if len(vals) == 0:
                continue
            rows.append({
                "track":    track,
                "model":    model,
                "mean_ic":  vals.mean(),
                "std_ic":   vals.std(),
                "hit_rate": (vals > 0).mean(),
                "ir":       vals.mean() / vals.std() if vals.std() > 1e-8 else np.nan,
                "n_folds":  len(vals),
            })

    return pd.DataFrame(rows).set_index(["track", "model"])


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    report = ic_stability_report()
    print("\n=== IC Stability Report ===\n")
    print(report.round(4).to_string())

    print("\n=== Expanding-Window Weights (Track B) ===")
    ew = EnsembleWeighter(track="track_b")
    ew.print_summary()

    print("\n=== Expanding-Window Weights (Track A) ===")
    ew_a = EnsembleWeighter(track="track_a")
    ew_a.print_summary()
