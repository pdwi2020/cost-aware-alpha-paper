"""Factor-adjusted alpha of a strategy's daily P&L (FF5 + momentum, HAC).

Why this exists
---------------
Track A's signal is discovered on FF5+UMD *idiosyncratic residuals*, but the
deployed book trades *raw* returns. Residualising the target does not remove
the book's factor exposure: a book sorted on 12-month momentum is short UMD
whether or not its target was residualised. The Array-reviewed manuscript
claimed that "systematic factor and market-beta exposure is removed at the
signal level rather than through a portfolio constraint", which is not what the
construction delivers.

This module measures the exposure instead of asserting it away: regress the
book's daily P&L on the six daily factors and report the HAC alpha. The alpha
is the part of the book's performance a factor investor could not have
replicated with cheap passive exposures, and the residual Sharpe is the
strategy's performance after (costless, idealised) factor hedging.

Convention
----------
The book is dollar-neutral by construction (long and short gross exposures are
equal every day), so its P&L is already an excess return: no risk-free rate is
subtracted. The regression is therefore

    pnl_t = alpha + b' F_t + e_t,     F = (Mkt-RF, SMB, HML, RMW, CMA, Mom)

with Newey-West standard errors. Factors are converted from the Kenneth French
percent convention to decimals, matching the P&L units.

Data
----
Daily factors come from the project's X9 mirror of the Kenneth French Data
Library (the same source the pipeline uses elsewhere):
    .../aqr_factors/famafrench_ff5_daily.parquet   (Mkt-RF, SMB, HML, RMW, CMA, RF)
    .../aqr_factors/famafrench_mom_daily.parquet   (Mom)
Both are indexed by date and quoted in percent.

Run
---
    python3 -u src/backtest/factor_alpha.py --pnl data/processed/holdout_pnl_track_a.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

X9_FACTORS = Path(
    os.environ.get(
        "CAVAL_FACTOR_DIR",
        "/Volumes/Crucial X9/data/market_data/equities/aqr_factors",
    )
)
FF5_PARQUET = X9_FACTORS / "famafrench_ff5_daily.parquet"
MOM_PARQUET = X9_FACTORS / "famafrench_mom_daily.parquet"

FACTOR_COLUMNS = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
ANN = 252.0


def log(msg: str) -> None:
    print(msg, flush=True)


def load_ff6_daily(
    start: str | None = None,
    end: str | None = None,
    ff5_path: Path = FF5_PARQUET,
    mom_path: Path = MOM_PARQUET,
) -> pd.DataFrame:
    """Return daily FF5 + momentum factors in DECIMAL units, indexed by date.

    Columns: mkt_rf, smb, hml, rmw, cma, mom, rf. The source files are in
    percent (French's convention), so every column is divided by 100.
    """
    for path in (ff5_path, mom_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Factor file not found: {path}. Set CAVAL_FACTOR_DIR to the "
                "directory holding famafrench_ff5_daily.parquet and "
                "famafrench_mom_daily.parquet."
            )

    ff5 = pd.read_parquet(ff5_path)
    mom = pd.read_parquet(mom_path)
    ff5.index = pd.to_datetime(ff5.index).normalize()
    mom.index = pd.to_datetime(mom.index).normalize()

    factors = ff5.join(mom, how="inner")
    factors = factors.rename(
        columns={
            "Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml",
            "RMW": "rmw", "CMA": "cma", "RF": "rf", "Mom": "mom",
        }
    )
    factors = factors[FACTOR_COLUMNS + ["rf"]].astype(float) / 100.0
    factors.index.name = "date"

    if start is not None:
        factors = factors.loc[factors.index >= pd.Timestamp(start)]
    if end is not None:
        factors = factors.loc[factors.index <= pd.Timestamp(end)]
    return factors


def _default_hac_lags(n_obs: int) -> int:
    """Newey-West lag rule used elsewhere in this project."""
    return int(np.floor(4 * (n_obs / 100.0) ** (2.0 / 9.0)))


def factor_regression(
    daily_pnl: pd.Series,
    factors: pd.DataFrame | None = None,
    hac_lags: int | None = None,
) -> dict:
    """Regress daily P&L on FF5 + momentum with Newey-West standard errors.

    Parameters
    ----------
    daily_pnl : pd.Series indexed by date, in decimal return units.
    factors   : DataFrame from ``load_ff6_daily``; loaded on demand if None.
    hac_lags  : Newey-West lag truncation; the standard rule if None.

    Returns
    -------
    dict with alpha (daily and annualised) and its HAC t/p, per-factor betas
    with t-statistics, R^2, the residual ("hedged") annualised Sharpe, the raw
    annualised Sharpe, n_obs and hac_lags.
    """
    pnl = pd.Series(daily_pnl).dropna()
    pnl.index = pd.to_datetime(pnl.index).normalize()

    if factors is None:
        factors = load_ff6_daily()

    joined = pd.concat([pnl.rename("pnl"), factors[FACTOR_COLUMNS]], axis=1).dropna()
    if len(joined) < 30:
        raise ValueError(
            f"Only {len(joined)} overlapping days between the P&L and the "
            "factor file; cannot run a meaningful regression."
        )

    y = joined["pnl"].to_numpy(dtype=float)
    X = sm.add_constant(joined[FACTOR_COLUMNS].to_numpy(dtype=float))
    lags = _default_hac_lags(len(y)) if hac_lags is None else int(hac_lags)
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    resid = fit.resid
    resid_sd = float(np.std(resid, ddof=1))
    raw_sd = float(np.std(y, ddof=1))

    return {
        "n_obs": int(len(y)),
        "start": str(joined.index.min().date()),
        "end": str(joined.index.max().date()),
        "hac_lags": lags,
        "alpha_daily": float(fit.params[0]),
        "alpha_ann": float(fit.params[0] * ANN),
        "alpha_t": float(fit.tvalues[0]),
        "alpha_p": float(fit.pvalues[0]),
        "betas": {
            name: {"beta": float(fit.params[i + 1]), "t": float(fit.tvalues[i + 1])}
            for i, name in enumerate(FACTOR_COLUMNS)
        },
        "r2": float(fit.rsquared),
        "raw_sharpe_ann": float(np.mean(y) / raw_sd * np.sqrt(ANN)) if raw_sd > 1e-15 else np.nan,
        "resid_sharpe_ann": (
            float(np.mean(resid) / resid_sd * np.sqrt(ANN)) if resid_sd > 1e-15 else np.nan
        ),
    }


def print_result(res: dict, label: str) -> None:
    log(f"\n=== Factor-adjusted alpha: {label} ===")
    log(f"  window        : {res['start']} .. {res['end']}  (T={res['n_obs']}, HAC lags={res['hac_lags']})")
    log(f"  alpha (ann)   : {res['alpha_ann']:+.4f}   t={res['alpha_t']:+.2f}  p={res['alpha_p']:.3f}")
    log(f"  R^2           : {res['r2']:.3f}")
    log(f"  Sharpe raw    : {res['raw_sharpe_ann']:+.3f}")
    log(f"  Sharpe hedged : {res['resid_sharpe_ann']:+.3f}  (residual, costless hedge)")
    log("  betas:")
    for name, b in res["betas"].items():
        log(f"    {name:<8s} {b['beta']:+.4f}  (t={b['t']:+.2f})")


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pnl",
        default=str(ROOT / "data" / "processed" / "holdout_pnl_track_a.parquet"),
        help="Parquet with a daily P&L column, indexed or keyed by date.",
    )
    parser.add_argument("--column", default="net_pnl")
    parser.add_argument("--hac-lags", type=int, default=None)
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.pnl)
    if "date" in df.columns:
        df = df.set_index("date")
    if args.column not in df.columns:
        raise SystemExit(
            f"Column {args.column!r} not in {args.pnl}; available: {list(df.columns)}"
        )

    res = factor_regression(df[args.column], hac_lags=args.hac_lags)
    print_result(res, f"{Path(args.pnl).name}:{args.column}")
    return res


if __name__ == "__main__":
    main()
