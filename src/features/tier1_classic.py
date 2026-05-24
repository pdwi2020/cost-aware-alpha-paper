"""Classic alpha features: momentum, reversal, vol, liquidity, earnings, macro.

All features are lag-1 corrected (signal at T close used to trade at T+1 open).
Input prices DataFrame must be indexed by (ticker, date) with columns:
    open, high, low, close, volume
Factor DataFrames indexed by date.
"""

import numpy as np
import pandas as pd


def compute_momentum(returns: pd.Series, window: int, skip: int = 1) -> pd.Series:
    """Cumulative return over `window` bars, skipping the most recent `skip` bars."""
    cum = (1 + returns).rolling(window).apply(np.prod, raw=True) - 1
    return cum.shift(skip)


def compute_amihud(returns: pd.Series, dollar_volume: pd.Series, window: int = 21) -> pd.Series:
    """Amihud (2002) illiquidity: mean(|ret| / dollar_volume) over rolling window."""
    return (returns.abs() / dollar_volume.replace(0, np.nan)).rolling(window).mean()


def compute_residualized_returns(
    raw_returns: pd.DataFrame,
    factors: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """FF5+Mom rolling OLS residuals (Track A idiosyncratic return targets).

    Args:
        raw_returns: DataFrame (date × ticker) of daily returns.
        factors: DataFrame (date × factor) aligned to raw_returns index.
                 Factors must be in decimal form (divide by 100 if from Ken French).
        window: Rolling OLS window in days.

    Returns:
        DataFrame of same shape as raw_returns containing OLS residuals.
    """
    X_full = factors.copy()
    X_full["_const"] = 1.0

    residuals = pd.DataFrame(np.nan, index=raw_returns.index, columns=raw_returns.columns)

    for t in range(window, len(raw_returns)):
        y_window = raw_returns.iloc[t - window : t].values        # (window, n_tickers)
        X_window = X_full.iloc[t - window : t].values             # (window, n_factors+1)

        valid_rows = ~np.isnan(X_window).any(axis=1)
        if valid_rows.sum() < window // 2:
            continue

        X_w = X_window[valid_rows]
        y_w = y_window[valid_rows]

        # Fill NaN in y per-column with column mean so lstsq stays vectorized.
        # Tickers with sparse data get slightly biased betas but the residual
        # is masked to NaN wherever the actual return at time t is NaN anyway.
        all_nan_cols = np.all(np.isnan(y_w), axis=0)
        col_means = np.where(all_nan_cols, 0.0, np.nanmean(y_w, axis=0))
        nan_mask = np.isnan(y_w)
        y_filled = np.where(nan_mask, col_means, y_w)

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_w, y_filled, rcond=None)
        except np.linalg.LinAlgError:
            continue

        y_hat = X_full.iloc[t].values @ coeffs
        resid_t = raw_returns.iloc[t].values - y_hat
        # Restore NaN where the actual return is missing (not traded)
        resid_t[np.isnan(raw_returns.iloc[t].values)] = np.nan
        residuals.iloc[t] = resid_t

    return residuals


def build_tier1_features(
    prices: pd.DataFrame,
    ff5_factors: pd.DataFrame,
    vix: pd.Series,
    term_spread: pd.Series,
) -> pd.DataFrame:
    """Construct Tier 1 feature library from daily OHLCV + factor data.

    Args:
        prices: MultiIndex DataFrame (ticker, date) with columns
                [open, high, low, close, volume].
        ff5_factors: DataFrame (date × factor) — Mkt-RF, SMB, HML, RMW, CMA, RF, Mom.
                     Values in decimal (already divided by 100).
        vix: Daily VIX values indexed by date.
        term_spread: 2s10s yield spread indexed by date.

    Returns:
        MultiIndex DataFrame (ticker, date) × features, lag-1 applied to all signals.
    """
    tickers = prices.index.get_level_values("ticker").unique()
    all_features = []

    for ticker in tickers:
        px = prices.xs(ticker, level="ticker").sort_index()
        if len(px) < 63:
            continue

        ret = px["close"].pct_change(fill_method=None)
        dollar_vol = px["close"] * px["volume"]

        feats = pd.DataFrame(index=px.index)

        # Momentum
        feats["ret_1d"] = ret
        feats["ret_5d"] = compute_momentum(ret, 5, skip=0)
        feats["ret_21d"] = compute_momentum(ret, 21, skip=0)
        feats["ret_63d"] = compute_momentum(ret, 63, skip=0)
        feats["ret_252d"] = compute_momentum(ret, 252, skip=0)
        feats["mom_12_1"] = feats["ret_252d"] - feats["ret_21d"]

        # Short-term reversal
        feats["reversal_1w"] = -feats["ret_5d"]
        feats["reversal_4w"] = -feats["ret_21d"]

        # Volatility and Sharpe
        feats["vol_21d"] = ret.rolling(21).std() * np.sqrt(252)
        roll_mean = ret.rolling(21).mean()
        roll_std = ret.rolling(21).std().replace(0, np.nan)
        feats["sharpe_21d"] = (roll_mean / roll_std) * np.sqrt(252)

        # Liquidity
        feats["amihud"] = compute_amihud(ret, dollar_vol, window=21)
        feats["roll_spread"] = (px["high"] - px["low"]) / px["close"].replace(0, np.nan)

        # Macro features (broadcast from date-aligned series)
        feats["vix"] = vix.reindex(px.index, method="ffill")
        feats["vix_chg_5d"] = feats["vix"].diff(5)
        feats["term_spread"] = term_spread.reindex(px.index, method="ffill")
        feats["term_spread_chg_21d"] = feats["term_spread"].diff(21)

        # Lag all features by 1 bar (signal at T close → trade at T+1 open)
        feats = feats.shift(1)

        feats.index = pd.MultiIndex.from_product(
            [[ticker], feats.index], names=["ticker", "date"]
        )
        all_features.append(feats)

    if not all_features:
        return pd.DataFrame()

    return pd.concat(all_features).sort_index()
