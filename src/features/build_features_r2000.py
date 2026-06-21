"""V13 Phase B: Russell 2000 feature panel (daily-computable subset).

Re-instantiates the CAVAL feature library on the Russell 2000 universe using
ONLY daily data. It reuses the exact S&P feature formulas:

  * Tier 1  (src/features/tier1_classic.py::build_tier1_features)
  * Track A / Track B targets (src/features/build_features.py)
  * Tier 2 cross-asset macro + crowding (src/features/tier2_extended.py)

The ±50% Screen-0 price sanitisation (coherent clean-once price
reconstruction) is applied at source, identically to build_features.py.

THREE intraday-microstructure features are UNAVAILABLE for the Russell 2000:
``vwap_dev``, ``vol_clock``, ``vol_sig_ratio`` all require the 1-minute OHLCV
catalog, which only covers the 661 S&P tickers.  They are dropped by design.
``overnight_gap`` is retained because it is computable from daily open/close.

Result: ``data/processed/features_all_r2000.parquet`` — MultiIndex
(ticker, date), the same 35 daily features + 2 targets as the S&P panel minus
the three intraday columns (37 columns total), column names identical so the
downstream stage scripts can consume it unchanged.

Run:
    cd ~/ML_Paper && python3 src/features/build_features_r2000.py
"""

import io
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

# Reuse the EXACT S&P pure-pandas formulas (no DB dependency).
from src.features.build_features import (
    SANITIZE_CAP,
    build_clean_prices_wide,
    compute_track_a,
    compute_track_b,
)
from src.features.tier1_classic import build_tier1_features
from src.features.tier2_extended import CROWDING_ETFS
from src.features.build_features_all import _compute_rolling_beta

OUT_DIR = ROOT / "data" / "processed"
INPUT_CACHE = OUT_DIR / "r2000_inputs"
OHLCV_R2000 = OUT_DIR / "daily_ohlcv_r2000.parquet"
OUT_PATH = OUT_DIR / "features_all_r2000.parquet"

UNIVERSE_START = "2010-01-01"
UNIVERSE_END = "2024-12-31"

# Public macro series (FRED CSV endpoint, no API key required).
FRED_SERIES = ["VIXCLS", "T10Y2Y", "DTWEXBGS", "DCOILWTICO", "DGS10", "DFF"]
KENFRENCH_FF5 = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                 "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
KENFRENCH_MOM = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                 "F-F_Momentum_Factor_daily_CSV.zip")


# ---------------------------------------------------------------------------
# Public-source loaders (replace the DuckDB-catalog loaders; X9 is private).
# The factor / macro / ETF series are universe-agnostic, so sourcing them from
# Ken French + FRED + yfinance reproduces the S&P inputs and keeps the R2000
# re-instantiation fully reproducible from public data.
# ---------------------------------------------------------------------------

def _fetch_fred(series_id: str) -> pd.Series:
    cache = INPUT_CACHE / f"fred_{series_id}.parquet"
    if cache.exists():
        s = pd.read_parquet(cache)["value"]
        s.index = pd.to_datetime(s.index)
        return s.rename(series_id)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    raw = urllib.request.urlopen(url, timeout=60).read()
    df = pd.read_csv(io.BytesIO(raw), na_values=[".", ""])
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["value"].astype(float)
    INPUT_CACHE.mkdir(parents=True, exist_ok=True)
    s.to_frame("value").to_parquet(cache)
    return s.rename(series_id)


def _load_fred_panel() -> pd.DataFrame:
    """date-indexed FRED panel, columns = series ids, ffill'd."""
    cols = {sid: _fetch_fred(sid) for sid in FRED_SERIES}
    panel = pd.DataFrame(cols).sort_index().ffill()
    panel.index = pd.to_datetime(panel.index)
    return panel


def _parse_kenfrench_zip(url: str, value_cols_hint: int) -> pd.DataFrame:
    """Parse a Ken French daily factor CSV-in-zip into a date-indexed % frame."""
    raw = urllib.request.urlopen(url, timeout=90).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    name = zf.namelist()[0]
    text = zf.read(name).decode("latin-1")
    rows = []
    header = None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if header is None:
            # data rows begin with an 8-digit date token
            if parts and parts[0].isdigit() and len(parts[0]) == 8:
                # header is the most recent non-data line we saw; if none, synthesize
                header = header or []
                rows.append(parts)
            continue
        if parts and parts[0].isdigit() and len(parts[0]) == 8:
            rows.append(parts)
    # Re-scan to capture the header line (last comma-line before first data row).
    header_cols = None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0].isdigit() and len(parts[0]) == 8:
            break
        if len([p for p in parts if p]) >= value_cols_hint:
            header_cols = [p for p in parts if p]
    df = pd.DataFrame([r for r in rows])
    df = df.iloc[:, : 1 + len(header_cols)]
    df.columns = ["date"] + header_cols
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in header_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("date")


def load_factors_public() -> pd.DataFrame:
    """FF5 + Mom daily factors (decimal), columns mkt_rf,smb,hml,rmw,cma,rf,mom.

    Mirrors build_features.load_factors() output schema exactly.
    """
    cache = INPUT_CACHE / "factors_ff5_mom_daily.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    ff5 = _parse_kenfrench_zip(KENFRENCH_FF5, value_cols_hint=6)
    ff5 = ff5.rename(columns={"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml",
                              "RMW": "rmw", "CMA": "cma", "RF": "rf"})
    mom = _parse_kenfrench_zip(KENFRENCH_MOM, value_cols_hint=1)
    mom = mom.rename(columns={mom.columns[0]: "mom"})
    factors = ff5.join(mom[["mom"]], how="left").sort_index() / 100.0  # % -> decimal
    INPUT_CACHE.mkdir(parents=True, exist_ok=True)
    factors.to_parquet(cache)
    return factors


def load_macro_public(fred: pd.DataFrame) -> pd.DataFrame:
    """Tier-1 macro: vix (VIXCLS), term_spread_2s10s (T10Y2Y). Mirrors load_macro()."""
    macro = pd.DataFrame(index=fred.index)
    macro["vix"] = fred["VIXCLS"]
    macro["term_spread_2s10s"] = fred["T10Y2Y"]
    return macro.ffill()


def build_tier2_macro_public(fred: pd.DataFrame, start, end) -> pd.DataFrame:
    """Replicates tier2_extended.build_tier2_macro() from the public FRED panel."""
    out = pd.DataFrame(index=fred.index)
    out["dxy_ret_5d"] = fred["DTWEXBGS"].pct_change(5, fill_method=None)
    out["wti_ret_21d"] = fred["DCOILWTICO"].pct_change(21, fill_method=None)
    out["credit_proxy"] = fred["DGS10"] - fred["DFF"]
    out["credit_proxy_chg_5d"] = out["credit_proxy"].diff(5)
    mask = (out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))
    return out[mask]


def build_tier2_crowding_public(daily, start, end, window=20) -> pd.DataFrame:
    """Replicates tier2_extended.build_tier2_crowding() with yfinance ETF closes."""
    cache = INPUT_CACHE / "etf_daily_close.parquet"
    if cache.exists():
        etf_wide = pd.read_parquet(cache)
        etf_wide.index = pd.to_datetime(etf_wide.index)
    else:
        raw = yf.download(CROWDING_ETFS, start=start, end="2025-01-01",
                          progress=False, auto_adjust=True, threads=True)
        etf_wide = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        etf_wide = etf_wide.dropna(how="all", axis=1)
        INPUT_CACHE.mkdir(parents=True, exist_ok=True)
        etf_wide.to_parquet(cache)
    available = [c for c in CROWDING_ETFS if c in etf_wide.columns]
    print(f"  Crowding ETFs available: {available}")
    etf_ret = etf_wide[available].pct_change(fill_method=None)

    close_wide = daily.pivot(index="date", columns="ticker", values="close")
    close_wide.index = pd.to_datetime(close_wide.index)
    stock_ret = close_wide.pct_change(fill_method=None)

    corr_dict = {}
    for etf in available:
        etf_s = etf_ret[etf].reindex(stock_ret.index)
        corr_dict[f"corr_{etf}"] = stock_ret.rolling(window).corr(etf_s)
    long_parts = [w.stack(future_stack=True).rename(f) for f, w in corr_dict.items()]
    crowding = pd.concat(long_parts, axis=1)
    crowding.index.names = ["date", "ticker"]
    return crowding.swaplevel(0, 1).sort_index()

# Three intraday features that require the 1-minute catalog (S&P-only).
INTRADAY_ONLY = ["vwap_dev", "vol_clock", "vol_sig_ratio"]

# Target column order = canonical feature set (no dropped columns) + targets +
# non-feature columns, minus INTRADAY_ONLY (vwap_dev/vol_clock/vol_sig_ratio).
# Removed (DROPPED): reversal_1w, reversal_4w (sign-duplicates),
#   vix, vix_chg_5d, term_spread, term_spread_chg_21d, credit_proxy,
#   credit_proxy_chg_5d, dxy_ret_5d, wti_ret_21d (broadcast-only macros),
#   term_spread_x_mom (legacy ad-hoc interaction).
# Added: beta_x_vix, beta_x_term_spread, credit_beta_x_credit (prespecified).
COLUMN_ORDER = [
    # Tier-1 cross-sectional features (kept)
    "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_252d", "mom_12_1",
    "vol_21d", "sharpe_21d", "amihud", "roll_spread", "overnight_gap",
    # Targets
    "target_track_b", "target_track_a",
    # Tier-2 crowding proxies (stock-specific, kept)
    "corr_SPY", "corr_QQQ", "corr_XLK", "corr_XLE", "corr_XLF", "corr_XLY",
    "corr_XLP", "corr_XLI", "corr_XLB", "corr_XLU", "corr_XLV", "corr_XLC",
    "corr_XLRE",
    # Prespecified macro-interaction features (stock-specific, added)
    "beta_x_vix", "beta_x_term_spread", "credit_beta_x_credit",
    # Regime labels (retained for diagnostics, not signals)
    "regime_vix", "regime_term_spread",
]


def sanitize_close(daily: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Replace `close` with the ±cap clean-once reconstruction; keep `close_raw`.

    Mirrors build_features.py::sanitize_ohlcv exactly (coherent clean-once
    price reconstruction via cumprod of clipped returns).
    """
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    close_w = daily.pivot(index="date", columns="ticker", values="close")
    close_clean_w, n_clipped = build_clean_prices_wide(close_w, cap)
    print(f"  [sanitize] Clipped {n_clipped:,} (ticker,day) cells with "
          f"|raw_ret| > {cap:.0%} to ±{cap:.0%}")
    clean_long = (
        close_clean_w.stack(future_stack=True).rename("close_clean").reset_index()
    )
    clean_long.columns = ["date", "ticker", "close_clean"]
    clean_long["date"] = pd.to_datetime(clean_long["date"])
    daily = daily.merge(clean_long, on=["ticker", "date"], how="left")
    daily.rename(columns={"close": "close_raw", "close_clean": "close"}, inplace=True)
    return daily, n_clipped


def build_tier2_daily(fred, daily, start, end):
    """Daily-computable Tier-2 panel: overnight_gap + crowding + cross-asset macro.

    Replicates tier2_extended.build_tier2_features() EXACTLY for the
    daily-available features, omitting the three 1-minute intraday features.
    A single trailing groupby('ticker').shift(1) reproduces the lag-1 convention.
    """
    # overnight_gap = open_T / close_{T-1} - 1.  Use the RAW close (close_raw)
    # for the denominator: the clean-once reconstruction rescales close after
    # clipped splits, so open/close_clean diverges by orders of magnitude on
    # small-caps with corporate actions (max 2.4e8 vs ~real).  The true overnight
    # return open/raw_close is then winsorized to +/-SANITIZE_CAP — the same
    # Screen-0 artifact cap already applied to close-to-close returns.
    rawclose_w = daily.pivot(index="date", columns="ticker", values="close_raw")
    open_w = daily.pivot(index="date", columns="ticker", values="open")
    rawclose_w.index = pd.to_datetime(rawclose_w.index)
    open_w.index = pd.to_datetime(open_w.index)
    overnight_w = (open_w / rawclose_w.shift(1) - 1).clip(-SANITIZE_CAP, SANITIZE_CAP)
    og = (
        overnight_w.stack(future_stack=True).rename("overnight_gap").reset_index()
    )
    og.columns = ["date", "ticker", "overnight_gap"]
    og["date"] = pd.to_datetime(og["date"])
    out = og.set_index(["ticker", "date"]).sort_index()

    # Crowding: rolling 20d corr of each stock with SPY/QQQ/sector ETFs.
    print("\n[Tier 2] Crowding proxies (ETF corr)...")
    t0 = time.time()
    crowding = build_tier2_crowding_public(daily, start, end, window=20)
    print(f"  Done in {time.time()-t0:.0f}s  ->  {crowding.shape}")
    if not crowding.empty:
        out = out.join(crowding, how="left")

    # Cross-asset macro (date-indexed) broadcast to all (ticker, date) rows.
    print("\n[Tier 2] Cross-asset macro features...")
    macro_feats = build_tier2_macro_public(fred, start, end)
    print(f"  Macro features: {list(macro_feats.columns)}")
    dates = out.index.get_level_values("date")
    macro_aligned = macro_feats.reindex(dates).ffill()
    for col in macro_aligned.columns:
        out[col] = macro_aligned[col].values

    # Window filter, then lag-1 (identical to build_tier2_features).
    date_idx = out.index.get_level_values("date")
    out = out[(date_idx >= pd.Timestamp(start)) & (date_idx <= pd.Timestamp(end))]
    out = out.groupby(level="ticker").shift(1)
    return out.sort_index()


def main() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== V13 Phase B: Russell 2000 features  [SANITIZE_CAP=+/-{SANITIZE_CAP:.0%}] ===\n")

    # 1. Load R2000 daily OHLCV and sanitize close at source.
    daily = pd.read_parquet(OHLCV_R2000)
    daily["date"] = pd.to_datetime(daily["date"])
    print(f"R2000 OHLCV: {daily.shape}  ({daily['ticker'].nunique()} tickers)")
    daily, n_clipped = sanitize_close(daily, SANITIZE_CAP)

    # 2. Universe-agnostic factors + macro from PUBLIC sources (Ken French / FRED).
    print("\nFetching FF5+Mom factors (Ken French) and macro (FRED)...")
    factors = load_factors_public()
    fred = _load_fred_panel()
    macro = load_macro_public(fred)

    # 3. Tier 1 features (sanitized close).
    prices_mi = daily.set_index(["ticker", "date"]).sort_index()
    print("\nBuilding Tier 1 features...")
    t0 = time.time()
    tier1 = build_tier1_features(
        prices_mi, factors, macro["vix"], macro["term_spread_2s10s"]
    )
    print(f"  Done in {time.time()-t0:.0f}s  ->  {tier1.shape}")

    # 4. Targets (sanitized close).
    close_wide = daily.pivot(index="date", columns="ticker", values="close")
    close_wide.index = pd.to_datetime(close_wide.index)
    print("\nComputing Track B (5d forward return)...")
    track_b = compute_track_b(close_wide)
    print("Computing Track A (FF5+UMD residualized 5d forward return)...")
    track_a = compute_track_a(close_wide, factors)

    def _to_long(wide, col):
        long = wide.stack(future_stack=True).rename(col).reset_index()
        long.columns = ["date", "ticker", col]
        long["date"] = pd.to_datetime(long["date"])
        return long

    out = tier1.reset_index().merge(
        _to_long(track_b, "target_track_b"), on=["ticker", "date"], how="left"
    ).merge(
        _to_long(track_a, "target_track_a"), on=["ticker", "date"], how="left"
    ).set_index(["ticker", "date"]).sort_index()

    # 5. Tier 2 daily-computable features (no 1-min intraday).
    tier2 = build_tier2_daily(fred, daily, UNIVERSE_START, UNIVERSE_END)
    out = out.join(tier2, how="left")

    # 6. Prespecified stock-level macro interaction features.
    #    These replace the legacy term_spread_x_mom interaction.
    #    Reuses _compute_rolling_beta from build_features_all.py.
    #    Formulas mirror build_features_all.py §6 exactly, sourcing macro factors
    #    from the public FRED panel (loaded above as `fred`).
    #    All betas are lagged 1 day (shift-1) — no look-ahead.
    print("\n  Computing prespecified interaction features (beta_x_vix, "
          "beta_x_term_spread, credit_beta_x_credit)...")
    BETA_WINDOW = 252

    # Wide daily return panel (sanitized close)
    close_wide_beta = daily.pivot(index="date", columns="ticker", values="close")
    close_wide_beta.index = pd.to_datetime(close_wide_beta.index)
    stock_ret_wide = close_wide_beta.pct_change(fill_method=None)

    # 6a. Market beta — factor = FF5 mkt_rf (from Ken French public data)
    try:
        mkt_rf_series = factors["mkt_rf"].reindex(stock_ret_wide.index).ffill()
        market_beta_lagged_wide = _compute_rolling_beta(
            stock_ret_wide, mkt_rf_series, window=BETA_WINDOW
        )
        mbl_long = (
            market_beta_lagged_wide.stack(future_stack=True)
            .rename("_market_beta_lagged")
            .reset_index()
        )
        mbl_long.columns = ["date", "ticker", "_market_beta_lagged"]
        mbl_long["date"] = pd.to_datetime(mbl_long["date"])
        out = out.reset_index().merge(mbl_long, on=["ticker", "date"], how="left")
        out = out.set_index(["ticker", "date"]).sort_index()

        # vix is in `fred` as VIXCLS; use the broadcast value in `out` if present
        # (it may be there via tier1 broadcast columns), else derive from fred.
        if "vix" in out.columns:
            out["beta_x_vix"] = out["_market_beta_lagged"] * out["vix"]
        else:
            vix_series = fred["VIXCLS"].rename("vix").reindex(
                out.index.get_level_values("date")
            ).values
            out["beta_x_vix"] = out["_market_beta_lagged"].values * vix_series
        print("    Added: beta_x_vix = market_beta_lagged × vix")

        # term_spread_chg_21d: 21-day change in T10Y2Y
        ts_chg_21d = fred["T10Y2Y"].diff(21).rename("term_spread_chg_21d")
        ts_aligned  = ts_chg_21d.reindex(out.index.get_level_values("date")).values
        out["beta_x_term_spread"] = out["_market_beta_lagged"].values * ts_aligned
        print("    Added: beta_x_term_spread = market_beta_lagged × term_spread_chg_21d")

        out.drop(columns=["_market_beta_lagged"], inplace=True)
    except Exception as _e:
        print(f"    [warn] Market-beta interactions skipped: {_e}")

    # 6b. Credit beta — factor = daily change in credit_proxy (DGS10 - DFF)
    try:
        credit_proxy_raw = (fred["DGS10"] - fred["DFF"]).reindex(
            stock_ret_wide.index
        ).ffill()
        d_credit = credit_proxy_raw.diff(1)
        credit_beta_lagged_wide = _compute_rolling_beta(
            stock_ret_wide, d_credit, window=BETA_WINDOW
        )
        cbl_long = (
            credit_beta_lagged_wide.stack(future_stack=True)
            .rename("_credit_beta_lagged")
            .reset_index()
        )
        cbl_long.columns = ["date", "ticker", "_credit_beta_lagged"]
        cbl_long["date"] = pd.to_datetime(cbl_long["date"])
        out = out.reset_index().merge(cbl_long, on=["ticker", "date"], how="left")
        out = out.set_index(["ticker", "date"]).sort_index()

        # credit_proxy_chg_5d: 5-day change in credit proxy (already in fred-based macro)
        cp_chg_5d = credit_proxy_raw.diff(5).reindex(
            out.index.get_level_values("date")
        ).values
        out["credit_beta_x_credit"] = out["_credit_beta_lagged"].values * cp_chg_5d
        out.drop(columns=["_credit_beta_lagged"], inplace=True)
        print("    Added: credit_beta_x_credit = credit_beta_lagged × credit_proxy_chg_5d")
    except Exception as _e:
        print(f"    [warn] Credit-beta interaction skipped: {_e}")

    # 6c. Retain broadcast columns as regime labels; they are NOT features
    #     (dropped via feature_spec) but kept for diagnostics / regime analysis.
    if "vix" in out.columns:
        out["regime_vix"] = out["vix"]
    if "term_spread" in out.columns:
        out["regime_term_spread"] = out["term_spread"]

    # 6b. Screen-0 artifact winsorization for the remaining ratio features.
    #     On small-caps, (high-low)/close and Amihud illiquidity carry severe
    #     corporate-action / penny-price artifacts (roll_spread up to 1e8) that
    #     the close-to-close +/-50% cap does not reach.  Because the composite
    #     signal sums RAW feature values, an uncapped artifact would dominate
    #     and collapse the signal.  We clip them to economically admissible
    #     ranges — a direct extension of Screen 0 to the harder universe.
    if "roll_spread" in out.columns:
        # (high-low)/close > 1 (>100% intraday range) is a data artifact.
        n_rs = int((out["roll_spread"] > 1.0).sum())
        out["roll_spread"] = out["roll_spread"].clip(0.0, 1.0)
        print(f"  [Screen0] roll_spread: clipped {n_rs:,} cells to [0, 1]")
    if "amihud" in out.columns:
        cap = float(np.nanpercentile(out["amihud"].values, 99.9))
        n_am = int((out["amihud"] > cap).sum())
        out["amihud"] = out["amihud"].clip(upper=cap)
        print(f"  [Screen0] amihud: winsorized {n_am:,} cells at p99.9={cap:.3g}")

    # 7. Window filter + drop fully-NaN feature rows (burn-in).
    date_idx = out.index.get_level_values("date")
    out = out[(date_idx >= pd.Timestamp(UNIVERSE_START)) &
              (date_idx <= pd.Timestamp(UNIVERSE_END))]
    target_cols = [c for c in out.columns if c.startswith("target")]
    feat_cols = [c for c in out.columns if c not in target_cols]
    out = out.dropna(subset=feat_cols, how="all")

    # 7b. Screen 0 eligibility (look-ahead-free, R2000 universe).
    #     For R2000 there are no PIT membership snapshots, so we apply only
    #     the lagged price (≥$5) and lagged ADV (≥$1M) criteria — same thresholds
    #     as spec.yaml screen0.  The result is stored as `s0_eligible` to keep
    #     the downstream FDR / portfolio code working identically.
    print("\n[Screen0] Computing R2000 look-ahead-free eligibility flags ...")
    from src.data.screen0 import trailing_adv_usd as _adv_fn, lagged_min_price as _px_fn
    _MIN_PRICE = 5.0     # spec.yaml screen0.min_price_usd
    _MIN_ADV   = 1_000_000  # spec.yaml screen0.min_adv_usd
    adv_series  = _adv_fn(daily)          # indexed (ticker, date)
    lpx_series  = _px_fn(daily)           # indexed (ticker, date)
    adv_ok   = adv_series  >= _MIN_ADV
    price_ok = lpx_series  >= _MIN_PRICE
    s0_eligible = (adv_ok & price_ok).rename("s0_eligible")
    out["s0_eligible"] = s0_eligible.reindex(out.index).fillna(False)
    out["adv_usd"] = adv_series.reindex(out.index)
    n_elig = int(out["s0_eligible"].sum())
    print(f"  Eligible rows: {n_elig:,} / {len(out):,} ({100*n_elig/len(out):.1f}%)")

    # 8. Enforce canonical column order (S&P order minus the 3 intraday).
    #    Non-feature columns (s0_eligible, adv_usd) are appended after the
    #    COLUMN_ORDER block so downstream consumers can find them.
    NON_FEAT_EXTRA = [c for c in ["s0_eligible", "adv_usd"] if c in out.columns]
    missing = [c for c in COLUMN_ORDER if c not in out.columns]
    extra = [c for c in out.columns if c not in COLUMN_ORDER and c not in NON_FEAT_EXTRA]
    if missing:
        print(f"  [WARN] expected columns missing: {missing}")
    if extra:
        print(f"  [WARN] unexpected extra columns: {extra}")
    ordered_cols = [c for c in COLUMN_ORDER if c in out.columns] + NON_FEAT_EXTRA
    out = out[ordered_cols]

    # 9. Report + save.
    dates = out.index.get_level_values("date")
    print(f"\n=== Output ===")
    print(f"Shape:      {out.shape}")
    print(f"Columns:    {len(out.columns)}")
    print(f"Dropped intraday (by design): {INTRADAY_ONLY}")
    print(f"Dropped broadcast/duplicate (by feature_spec): reversal_1w, reversal_4w, "
          "vix, vix_chg_5d, term_spread, term_spread_chg_21d, credit_proxy, "
          "credit_proxy_chg_5d, dxy_ret_5d, wti_ret_21d, term_spread_x_mom")
    print(f"Date range: {dates.min().date()} -> {dates.max().date()}")
    print(f"Tickers:    {out.index.get_level_values('ticker').nunique()}")
    print(f"\nPer-column NaN fraction:")
    print(out.isna().mean().round(3).to_string())
    high_nan = out.columns[(out.isna().mean() > 0.50)].tolist()
    high_nan = [c for c in high_nan if not c.startswith("target")]
    if high_nan:
        print(f"\n  [WARN] feature columns >50% NaN: {high_nan}")
    # Track A must be a residual, not a copy of Track B.
    both = out[["target_track_a", "target_track_b"]].dropna()
    if len(both):
        corr_ab = both["target_track_a"].corr(both["target_track_b"])
        print(f"\ncorr(track_a, track_b) = {corr_ab:.3f}  "
              f"(should be <1.0 — Track A is FF5+UMD-residualized)")

    out.to_parquet(OUT_PATH)
    print(f"\nSaved -> {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    main()
