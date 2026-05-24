"""Walk-forward model suite: Logistic, Ridge, Lasso, RF, XGBoost, LightGBM.

Preprocessing follows Gu, Kelly & Xiu (2020, RFS):
  - Cross-sectional winsorize at 1/99% per date (xs features)
  - Cross-sectional z-score per date (xs features)
  - Time-series z-score using training stats (macro features)
  - NaN filled with 0 after normalization

IC = Spearman rank correlation between predicted signal and actual returns.
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")   # suppress all — warnings go to log and obscure progress

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
XS_FEATURES = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_252d", "mom_12_1",
    "reversal_1w", "reversal_4w", "vol_21d", "sharpe_21d", "amihud",
    "roll_spread", "overnight_gap", "vwap_dev", "vol_clock", "vol_sig_ratio",
    "corr_SPY", "corr_QQQ", "corr_XLK", "corr_XLE", "corr_XLF", "corr_XLY",
    "corr_XLP", "corr_XLI", "corr_XLB", "corr_XLU", "corr_XLV", "corr_XLC",
    "corr_XLRE", "term_spread_x_mom",
]
MACRO_FEATURES = [
    "vix", "vix_chg_5d", "term_spread", "term_spread_chg_21d",
    "dxy_ret_5d", "wti_ret_21d", "credit_proxy", "credit_proxy_chg_5d",
]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _xs_winsorize_zscore(X: np.ndarray, dates: np.ndarray, q: float = 0.01) -> np.ndarray:
    """In-place cross-sectional winsorize + z-score per unique date.

    X: (N, D) float array.  dates: (N,) array of date labels.
    """
    result = X.copy().astype(np.float64)
    for d in np.unique(dates):
        mask = dates == d
        x = result[mask]                           # (n_tickers, D)
        lo = np.nanpercentile(x, q * 100, axis=0)
        hi = np.nanpercentile(x, (1 - q) * 100, axis=0)
        x = np.clip(x, lo, hi)
        mu  = np.nanmean(x, axis=0)
        sig = np.nanstd(x, axis=0)
        sig[sig < 1e-8] = 1.0                      # avoid division by zero for macro cols
        result[mask] = (x - mu) / sig
    return result


def _ts_zscore(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """Time-series z-score: fit mean/std on train, apply to both."""
    mu  = np.nanmean(X_train, axis=0)
    sig = np.nanstd(X_train,  axis=0)
    sig[sig < 1e-8] = 1.0
    return (X_train - mu) / sig, (X_test - mu) / sig


def preprocess(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    q: float = 0.01,
) -> tuple:
    """Preprocess features for one fold.

    Returns (X_tr_np, X_te_np): float32 arrays, NaN → 0.
    """
    # Determine which columns are actually present
    xs_cols    = [c for c in XS_FEATURES    if c in X_train.columns]
    macro_cols = [c for c in MACRO_FEATURES if c in X_train.columns]
    all_cols   = xs_cols + macro_cols

    X_tr = X_train[all_cols].values.astype(np.float64)
    X_te = X_test[all_cols].values.astype(np.float64)

    n_xs = len(xs_cols)

    dates_tr = X_train.index.get_level_values("date").values
    dates_te = X_test.index.get_level_values("date").values

    # XS block: cross-sectional winsorize + z-score
    if n_xs > 0:
        X_tr[:, :n_xs] = _xs_winsorize_zscore(X_tr[:, :n_xs], dates_tr, q)
        X_te[:, :n_xs] = _xs_winsorize_zscore(X_te[:, :n_xs], dates_te, q)

    # Macro block: time-series z-score (fit on train)
    if macro_cols:
        X_tr[:, n_xs:], X_te[:, n_xs:] = _ts_zscore(
            X_tr[:, n_xs:], X_te[:, n_xs:]
        )

    np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    return X_tr.astype(np.float32), X_te.astype(np.float32), all_cols


def ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Spearman IC, ignoring NaN in y_true."""
    valid = ~np.isnan(y_true)
    if valid.sum() < 10:
        return np.nan
    pred_v = y_pred[valid]
    true_v = y_true[valid]
    if not np.all(np.isfinite(pred_v)) or np.std(pred_v) < 1e-8:
        return np.nan
    val = float(spearmanr(pred_v, true_v)[0])
    return val if np.isfinite(val) else np.nan


# ---------------------------------------------------------------------------
# Model suite
# ---------------------------------------------------------------------------

class ModelSuite:
    """Fits all 6 models on each walk-forward fold; returns IC per model per fold."""

    MODEL_NAMES = ["logistic", "ridge", "lasso", "rf", "xgb", "lgbm"]

    def __init__(self, config_path: str = "configs/models.yaml", seed: int = 42):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.seed = seed
        self._fitted: dict = {}            # {fold_id: {model_name: fitted_model}}
        self._feature_cols: list = []      # set during first fit_fold call
        self._fold_ics: dict = {}          # {fold_id: {model_name: IC}}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        fold_id: str,
    ) -> dict:
        """Fit all 6 models on one fold. Returns {model_name: IC}."""
        # Align: drop rows where target is NaN
        tr_mask = y_train.notna()
        te_mask = y_test.notna()
        X_tr_raw = X_train[tr_mask]
        y_tr     = y_train[tr_mask].values.astype(np.float32)
        X_te_raw = X_test[te_mask]
        y_te     = y_test[te_mask].values.astype(np.float32)

        # Preprocess
        X_tr, X_te, feat_cols = preprocess(X_tr_raw, X_te_raw)
        if not self._feature_cols:
            self._feature_cols = feat_cols

        # Validation split: last val_frac of training dates (chronological)
        val_frac = 0.20
        dates_tr = X_tr_raw.index.get_level_values("date")
        unique_tr_dates = np.sort(dates_tr.unique())
        n_val_dates = max(1, int(len(unique_tr_dates) * val_frac))
        val_cutoff  = unique_tr_dates[-n_val_dates]
        is_val = dates_tr >= val_cutoff
        X_tv_full, y_tv_full = X_tr[~is_val], y_tr[~is_val]   # train-minus-val
        X_vl,      y_vl      = X_tr[is_val],  y_tr[is_val]    # val

        # Cap grid-search training data at MAX_GRID_ROWS for speed on large folds.
        # Final model is always fit on the full data.
        MAX_GRID_ROWS = 300_000
        if len(X_tv_full) > MAX_GRID_ROWS:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(X_tv_full), MAX_GRID_ROWS, replace=False)
            X_tv, y_tv = X_tv_full[idx], y_tv_full[idx]
        else:
            X_tv, y_tv = X_tv_full, y_tv_full

        # Binary target for logistic
        y_tv_bin = (y_tv > 0).astype(int)

        # X_full = full train data for final model fits (may be larger than X_tv)
        X_full = np.vstack([X_tv_full, X_vl])
        y_full = np.concatenate([y_tv_full, y_vl])
        y_full_bin = (y_full > 0).astype(int)

        models = {
            "ridge":    self._best_ridge(X_tv, y_tv, X_vl, y_vl, X_full, y_full),
            "lasso":    self._best_lasso(X_tv, y_tv, X_vl, y_vl, X_full, y_full),
            "logistic": self._best_logistic(X_tv, y_tv_bin, X_vl, y_vl, X_full, y_full_bin),
            "rf":       self._best_rf(X_tv, y_tv, X_vl, y_vl, X_full, y_full),
            "xgb":      self._best_xgb(X_tv, y_tv, X_vl, y_vl, X_full, y_full),
            "lgbm":     self._best_lgbm(X_tv, y_tv, X_vl, y_vl, X_full, y_full),
        }

        # Predict on test
        fold_ic = {}
        preds = {}
        for name, model in models.items():
            if name == "logistic":
                pred = model.predict_proba(X_te)[:, 1]
            else:
                pred = model.predict(X_te)
            fold_ic[name] = ic(pred, y_te)
            preds[name]   = pred

        # IC-weighted ensemble
        valid_ics = {k: v for k, v in fold_ic.items() if not np.isnan(v)}
        if valid_ics:
            total = sum(abs(v) for v in valid_ics.values())
            weights = {k: abs(v) / total for k, v in valid_ics.items()} if total > 0 else {k: 1 / len(valid_ics) for k in valid_ics}
            ens_pred = sum(weights[k] * preds[k] for k in valid_ics)
            fold_ic["ensemble"] = ic(ens_pred, y_te)
        else:
            fold_ic["ensemble"] = np.nan

        self._fitted[fold_id] = models
        self._fold_ics[fold_id] = fold_ic
        return fold_ic

    def fit_all_folds(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        fold_dates: list,
    ) -> pd.DataFrame:
        """Walk-forward loop. Returns DataFrame: index=fold_id, columns=model names."""
        records = []
        for fd in fold_dates:
            fid = fd["fold_id"]
            print(f"\n  Fold {fid}  train={fd['train_start']}→{fd['train_end']}  "
                  f"test={fd['test_start']}→{fd['test_end']}", flush=True)

            dates = features.index.get_level_values("date")
            tr_mask = (dates >= pd.Timestamp(fd["train_start"])) & \
                      (dates <= pd.Timestamp(fd["train_end"]))
            te_mask = (dates >= pd.Timestamp(fd["test_start"])) & \
                      (dates <= pd.Timestamp(fd["test_end"]))

            X_tr = features[tr_mask]
            y_tr = targets[tr_mask]
            X_te = features[te_mask]
            y_te = targets[te_mask]

            t_fold = time.time()
            print(f"    train: {tr_mask.sum():,} rows  test: {te_mask.sum():,} rows", flush=True)
            fold_ic = self.fit_fold(X_tr, y_tr, X_te, y_te, fid)
            fold_elapsed = time.time() - t_fold
            for name, val in fold_ic.items():
                print(f"    IC {name:<12s} = {val:+.4f}", flush=True)
            print(f"    fold time: {fold_elapsed:.0f}s", flush=True)
            records.append({"fold": fid, **fold_ic})

        return pd.DataFrame(records).set_index("fold")

    def ensemble_predict(self, X: pd.DataFrame, fold_id: str = None) -> np.ndarray:
        """IC-weighted ensemble prediction using the last fitted fold."""
        if fold_id is None:
            fold_id = sorted(self._fitted)[-1]
        models = self._fitted[fold_id]
        fold_ic = self._fold_ics[fold_id]

        _, X_np, _ = _dummy_preprocess(X, self._feature_cols)

        valid = {k: v for k, v in fold_ic.items()
                 if k in models and not np.isnan(v)}
        total = sum(abs(v) for v in valid.values())
        if total == 0:
            return np.zeros(len(X))

        pred = np.zeros(len(X), dtype=np.float64)
        for k, v in valid.items():
            m = models[k]
            p = m.predict_proba(X_np)[:, 1] if k == "logistic" else m.predict(X_np)
            pred += (abs(v) / total) * p
        return pred

    # ------------------------------------------------------------------
    # Private fitters
    # ------------------------------------------------------------------

    def _best_ridge(self, X_tv, y_tv, X_vl, y_vl, X_full, y_full):
        best_ic, best_a = -np.inf, 1.0
        for a in self.cfg["ridge"]["alpha_values"]:
            m = Ridge(alpha=a).fit(X_tv, y_tv)
            s = ic(m.predict(X_vl), y_vl)
            if not np.isnan(s) and s > best_ic:
                best_ic, best_a = s, a
        return Ridge(alpha=best_a).fit(X_full, y_full)

    def _best_lasso(self, X_tv, y_tv, X_vl, y_vl, X_full, y_full):
        best_ic, best_a = -np.inf, self.cfg["lasso"]["alpha_values"][0]
        for a in self.cfg["lasso"]["alpha_values"]:
            m = Lasso(alpha=a, max_iter=5000).fit(X_tv, y_tv)
            s = ic(m.predict(X_vl), y_vl)
            if not np.isnan(s) and s > best_ic:
                best_ic, best_a = s, a
        final = Lasso(alpha=best_a, max_iter=5000).fit(X_full, y_full)
        # Guard: alpha selected on subsampled grid can over-regularize on larger X_full.
        # If all coefs zero, walk back through alphas in ascending order until non-trivial.
        if np.all(final.coef_ == 0) or np.std(final.predict(X_full)) < 1e-8:
            for a in sorted(self.cfg["lasso"]["alpha_values"]):
                final = Lasso(alpha=a, max_iter=5000).fit(X_full, y_full)
                if np.any(final.coef_ != 0) and np.std(final.predict(X_full)) > 1e-8:
                    break
        return final

    def _best_logistic(self, X_tv, y_tv_bin, X_vl, y_vl, X_full, y_full_bin):
        best_ic, best_c = -np.inf, 0.1
        for c in self.cfg["logistic"]["C_values"]:
            m = LogisticRegression(
                C=c, penalty="l2", max_iter=self.cfg["logistic"]["max_iter"],
                solver="saga", n_jobs=-1,      # saga scales better to large N
            ).fit(X_tv, y_tv_bin)
            s = ic(m.predict_proba(X_vl)[:, 1], y_vl)
            if not np.isnan(s) and s > best_ic:
                best_ic, best_c = s, c
        return LogisticRegression(
            C=best_c, penalty="l2", max_iter=self.cfg["logistic"]["max_iter"],
            solver="saga", n_jobs=-1,
        ).fit(X_full, y_full_bin)

    def _best_rf(self, X_tv, y_tv, X_vl, y_vl, X_full, y_full):
        cfg = self.cfg["random_forest"]
        best_ic, best_params = -np.inf, (5, 100)
        for depth in cfg["max_depth_values"]:
            for leaf in cfg["min_samples_leaf_values"]:
                m = RandomForestRegressor(
                    n_estimators=50,          # fast grid search
                    max_depth=depth,
                    max_features="sqrt",      # sqrt(38)≈6 vs default 38 → ~5× faster
                    min_samples_leaf=leaf, n_jobs=-1,
                    random_state=self.seed,
                ).fit(X_tv, y_tv)
                s = ic(m.predict(X_vl), y_vl)
                if not np.isnan(s) and s > best_ic:
                    best_ic, best_params = s, (depth, leaf)
        depth, leaf = best_params
        return RandomForestRegressor(
            n_estimators=200, max_depth=depth,
            max_features="sqrt",
            min_samples_leaf=leaf, n_jobs=-1,
            random_state=self.seed,
        ).fit(X_full, y_full)

    def _best_xgb(self, X_tv, y_tv, X_vl, y_vl, X_full, y_full):
        cfg = self.cfg["xgboost"]
        best_ic, best_params, best_iters = -np.inf, (3, 0.05), 200
        for depth in cfg["max_depth_values"]:
            for eta in cfg["eta_values"]:
                m = xgb.XGBRegressor(
                    max_depth=depth, learning_rate=eta,
                    n_estimators=cfg["n_estimators"],
                    subsample=cfg["subsample"],
                    colsample_bytree=cfg["colsample_bytree"],
                    early_stopping_rounds=cfg["early_stopping_rounds"],
                    eval_metric="rmse", verbosity=0,
                    random_state=self.seed, n_jobs=-1,
                ).fit(X_tv, y_tv, eval_set=[(X_vl, y_vl)], verbose=False)
                s = ic(m.predict(X_vl), y_vl)
                if not np.isnan(s) and s > best_ic:
                    best_ic, best_params = s, (depth, eta)
                    best_iters = max(50, getattr(m, "best_iteration", 200))
        depth, eta = best_params
        return xgb.XGBRegressor(
            max_depth=depth, learning_rate=eta,
            n_estimators=best_iters,
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            verbosity=0, random_state=self.seed, n_jobs=-1,
        ).fit(X_full, y_full)

    def _best_lgbm(self, X_tv, y_tv, X_vl, y_vl, X_full, y_full):
        cfg = self.cfg["lightgbm"]
        best_ic, best_params, best_iters = -np.inf, (31, 0.05), 200
        for leaves in cfg["num_leaves_values"]:
            for lr in cfg["learning_rate_values"]:
                m = lgb.LGBMRegressor(
                    num_leaves=leaves, learning_rate=lr,
                    n_estimators=cfg["n_estimators"],
                    verbose=-1, random_state=self.seed, n_jobs=-1,
                ).fit(
                    X_tv, y_tv,
                    eval_set=[(X_vl, y_vl)],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                                lgb.log_evaluation(period=-1)],
                )
                s = ic(m.predict(X_vl), y_vl)
                if not np.isnan(s) and s > best_ic:
                    best_ic, best_params = s, (leaves, lr)
                    # best_iteration_ is -1 when early stopping never triggered
                    best_iters = max(50, m.best_iteration_) if m.best_iteration_ > 0 else 200
        leaves, lr = best_params
        final = lgb.LGBMRegressor(
            num_leaves=leaves, learning_rate=lr,
            n_estimators=best_iters,
            verbose=-1, random_state=self.seed, n_jobs=-1,
        ).fit(X_full, y_full)
        # Guard: if final model predicts constant, fall back to stable defaults
        if np.std(final.predict(X_full)) < 1e-8:
            final = lgb.LGBMRegressor(
                num_leaves=31, learning_rate=0.05, n_estimators=200,
                verbose=-1, random_state=self.seed, n_jobs=-1,
            ).fit(X_full, y_full)
        return final


# ---------------------------------------------------------------------------
# Walk-forward fold generator
# ---------------------------------------------------------------------------

def make_fold_dates(
    train_start: str = "2010-01-01",
    warmup_end_year: int = 2012,
    test_end_year: int = 2021,
) -> list:
    """Expanding-window fold definitions.

    Each fold: train = [train_start, test_year-1], test = [test_year].
    """
    folds = []
    for test_year in range(warmup_end_year + 1, test_end_year + 1):
        folds.append({
            "fold_id":    str(test_year),
            "train_start": train_start,
            "train_end":  f"{test_year - 1}-12-31",
            "test_start": f"{test_year}-01-01",
            "test_end":   f"{test_year}-12-31",
        })
    return folds


def _dummy_preprocess(X: pd.DataFrame, feature_cols: list):
    """Light preprocessing for inference-only (no val split)."""
    xs_cols    = [c for c in XS_FEATURES    if c in X.columns and c in feature_cols]
    macro_cols = [c for c in MACRO_FEATURES if c in X.columns and c in feature_cols]
    all_cols   = xs_cols + macro_cols
    arr = X[all_cols].fillna(0).values.astype(np.float32)
    return arr, arr, all_cols
