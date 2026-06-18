"""Robustness pull: fresh split/dividend-adjusted Close from yfinance for the
surviving universe, to cross-validate our cleaned prices. Delisted names will
fail (expected) — the tradeable-universe filter handles those regardless.
Saves long-format parquet: date, ticker, adj_close.
"""
import sys, time
from pathlib import Path
import pandas as pd, yfinance as yf
import warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
tickers = [t.strip() for t in open('/tmp/universe_tickers.txt') if t.strip()]
OUT = ROOT/"data"/"processed"/"yf_adjclose_validation.parquet"
START, END = "2013-01-01", "2025-01-01"
BATCH = 40
frames, ok, fail = [], [], []
print(f"[pull] {len(tickers)} tickers, batches of {BATCH}", flush=True)
for i in range(0, len(tickers), BATCH):
    b = tickers[i:i+BATCH]
    try:
        df = yf.download(b, start=START, end=END, progress=False, auto_adjust=True, threads=True)
        close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": b[0]})
        close = close.dropna(how="all", axis=1)
        for c in close.columns:
            s = close[c].dropna()
            if len(s) > 50:
                frames.append(pd.DataFrame({"date": s.index, "ticker": c, "adj_close": s.values})); ok.append(c)
        got = [c for c in close.columns]
        fail += [t for t in b if t not in got]
        print(f"[pull] batch {i//BATCH+1}/{(len(tickers)+BATCH-1)//BATCH}: +{len(got)} names (total ok={len(ok)})", flush=True)
    except Exception as e:
        fail += b; print(f"[pull] batch {i//BATCH+1} ERROR: {repr(e)[:120]}", flush=True)
    time.sleep(1.0)
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(OUT, index=False)  # incremental save
out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
out.to_parquet(OUT, index=False)
print(f"[pull] DONE ok={len(ok)} fail={len(fail)} rows={len(out)} -> {OUT}", flush=True)
print(f"[pull] sample failures (likely delisted): {sorted(fail)[:25]}", flush=True)
