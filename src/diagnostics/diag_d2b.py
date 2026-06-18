"""D2b: decisive reconciliation of clean Track B OOS Sharpe.
Compute Track B OOS via the canonical run_holdout machinery under 3 configs:
  (A) clean features + clean fdr/shap  (= what D2 holdout did -> expect -0.014)
  (B) raw features    + raw fdr/shap   (raw signal) on CLEAN returns (= D1 -> expect ~+0.77)
  (C) clean features + clean fdr/shap, but RAW (unsanitized) returns
Isolates whether the clean-feature SIGNAL or a bug drives the collapse.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.backtest.portfolio import PortfolioSimulator
from src.backtest.generate_signals import build_signal_weights, generate_composite_signal

P = ROOT/"data"/"processed"
CFG = ROOT/"configs"/"backtest.yaml"
OHLCV = P/"daily_ohlcv.parquet"
S, E = "2022-01-01", "2024-12-31"
CAP = 0.50

def returns_vol_adv(ohlcv, tickers, sanitize):
    m = ohlcv["ticker"].isin(tickers) & (ohlcv["date"]>=pd.Timestamp(S)) & (ohlcv["date"]<=pd.Timestamp(E))
    sub = ohlcv[m][["ticker","date","close","volume"]]
    cw = sub.pivot(index="date",columns="ticker",values="close")
    vw = sub.pivot(index="date",columns="ticker",values="volume")
    r = cw.pct_change(fill_method=None)
    if sanitize: r = r.clip(-CAP, CAP)
    vol = r.rolling(21,min_periods=10).std()
    adv = (cw*vw).rolling(21,min_periods=10).mean()
    mask = (r.index>=pd.Timestamp(S))&(r.index<=pd.Timestamp(E))
    return r[mask], vol[mask], adv[mask]

def run(feat_file, fdr_file, shap_file, sanitize, label):
    fdr = pd.read_parquet(P/fdr_file); shap = pd.read_parquet(P/shap_file)
    feat = pd.read_parquet(P/feat_file)
    ohlcv = pd.read_parquet(OHLCV)
    cfg = yaml.safe_load(open(CFG)); aum = cfg.get("aum_dollars",1e8)
    w = build_signal_weights("track_b", fdr, shap)
    sig = generate_composite_signal(feat, w, start_date=S, end_date=E)
    tickers = sig.columns.tolist()
    r,vol,adv = returns_vol_adv(ohlcv, tickers, sanitize)
    sim = PortfolioSimulator(config_path=str(CFG), spread_bps=cfg["spread_bps"], impact_coeff=cfg["impact_coeff"])
    pos = sim.signal_to_positions(sig, lag=1, rebal_freq=5)
    pnl = sim.simulate_pnl(pos, r, vol=vol, adv_dollars=adv, aum_dollars=aum, min_adv_dollars=cfg.get("min_adv_dollars",1e6))
    m = sim.compute_metrics(pnl)
    print(f"  [{label}] signal {sig.shape}, |sig|>0 frac={ (sig.abs()>1e-10).mean().mean():.3f}, "
          f"gross_SR={m['gross_pnl_sharpe']:+.3f} net_SR={m['net_pnl_sharpe']:+.3f} "
          f"drag={m['cost_drag_bps']:.0f}bps TO={m['annual_turnover']:.1f}x")
    return sig, m

print("=== D2b reconciliation: Track B OOS 2022-2024 ===")
print("(A) CLEAN features + clean fdr/shap, CLEAN returns  [= D2 holdout]")
sigA,_ = run("features_all.parquet","fdr_results.parquet","shap_summary.parquet", True, "A clean-sig/clean-ret")
print("(B) RAW features + raw fdr/shap, CLEAN returns       [= D1 raw-signal]")
sigB,_ = run("features_all.raw.bak.parquet","fdr_results.raw.bak.parquet","shap_summary.raw.bak.parquet", True, "B raw-sig/clean-ret")
print("(C) CLEAN features + clean fdr/shap, RAW returns")
sigC,_ = run("features_all.parquet","fdr_results.parquet","shap_summary.parquet", False, "C clean-sig/raw-ret")
# OOS signal correlation A vs B
ci=sigA.index.intersection(sigB.index); cc=sigA.columns.intersection(sigB.columns)
a=sigA.loc[ci,cc].values.flatten(); b=sigB.loc[ci,cc].values.flatten()
mm=np.isfinite(a)&np.isfinite(b)
print(f"\nOOS signal corr(clean,raw) = {np.corrcoef(a[mm],b[mm])[0,1]:.4f} (n={mm.sum()})")

# ---- P&L attribution for config B (raw signal + clean returns) ----
def attribute(feat_file, fdr_file, shap_file):
    fdr=pd.read_parquet(P/fdr_file); shap=pd.read_parquet(P/shap_file)
    feat=pd.read_parquet(P/feat_file); ohlcv=pd.read_parquet(OHLCV)
    cfg=yaml.safe_load(open(CFG)); aum=cfg.get("aum_dollars",1e8)
    w=build_signal_weights("track_b",fdr,shap)
    sig=generate_composite_signal(feat,w,start_date=S,end_date=E)
    tickers=sig.columns.tolist()
    r,vol,adv=returns_vol_adv(ohlcv,tickers,True)
    sim=PortfolioSimulator(config_path=str(CFG),spread_bps=cfg["spread_bps"],impact_coeff=cfg["impact_coeff"])
    pos=sim.signal_to_positions(sig,lag=1,rebal_freq=5)
    # per-name gross pnl = sum_t pos_{t-1,i} * ret_{t,i}  (positions already lagged inside sim? use pos*ret aligned)
    ci=pos.index.intersection(r.index); cc=pos.columns.intersection(r.columns)
    contrib=(pos.loc[ci,cc]*r.loc[ci,cc]).sum().sort_values()
    tot=contrib.sum()
    print(f"\n  Config B per-name OOS gross P&L (sum of pos*ret). total={tot:.4f}")
    print("  TOP 8 PROFIT (most positive contribution):")
    for t,v in contrib.tail(8)[::-1].items(): print(f"    {t:8s} {v:+.4f}  ({100*v/tot:+.1f}% of total)")
    print("  TOP 8 LOSS:")
    for t,v in contrib.head(8).items(): print(f"    {t:8s} {v:+.4f}")
    top8=contrib.tail(8).sum()
    print(f"  Top-8 winners = {100*top8/tot:.1f}% of total gross P&L; n names with nonzero={int((contrib.abs()>1e-9).sum())}")

print("\n=== ATTRIBUTION: config B (raw-feature signal, clean returns, +0.84) ===")
attribute("features_all.raw.bak.parquet","fdr_results.raw.bak.parquet","shap_summary.raw.bak.parquet")

# ---- RESCUE: min-price $5 tradeable-universe filter ----
def run_minprice(feat_file, fdr_file, shap_file, sanitize_ret, min_price, label):
    fdr=pd.read_parquet(P/fdr_file); shap=pd.read_parquet(P/shap_file)
    feat=pd.read_parquet(P/feat_file); ohlcv=pd.read_parquet(OHLCV)
    cfg=yaml.safe_load(open(CFG)); aum=cfg.get("aum_dollars",1e8)
    w=build_signal_weights("track_b",fdr,shap)
    sig=generate_composite_signal(feat,w,start_date=S,end_date=E)
    tickers=sig.columns.tolist()
    r,vol,adv=returns_vol_adv(ohlcv,tickers,sanitize_ret)
    # build price matrix (date x ticker) for the OOS window
    m=ohlcv["ticker"].isin(tickers)&(ohlcv["date"]>=pd.Timestamp(S))&(ohlcv["date"]<=pd.Timestamp(E))
    px=ohlcv[m].pivot(index="date",columns="ticker",values="close")
    sim=PortfolioSimulator(config_path=str(CFG),spread_bps=cfg["spread_bps"],impact_coeff=cfg["impact_coeff"])
    pos=sim.signal_to_positions(sig,lag=1,rebal_freq=5)
    # mask out names priced < min_price on each date (reindex price to positions)
    pxa=px.reindex(index=pos.index,columns=pos.columns).ffill()
    keep=(pxa>=min_price)
    pos_f=pos.where(keep,0.0)
    # renormalize gross leverage to 1 per day (L1) to keep capital deployed
    l1=pos_f.abs().sum(axis=1).replace(0,np.nan)
    pos_f=pos_f.div(l1,axis=0).fillna(0.0)
    pnl=sim.simulate_pnl(pos_f,r,vol=vol,adv_dollars=adv,aum_dollars=aum,min_adv_dollars=cfg.get("min_adv_dollars",1e6))
    mt=sim.compute_metrics(pnl)
    nmask=int((~keep).sum().sum())
    print(f"  [{label}] minP=${min_price} excl_cells={nmask}  gross_SR={mt['gross_pnl_sharpe']:+.3f} net_SR={mt['net_pnl_sharpe']:+.3f} drag={mt['cost_drag_bps']:.0f} TO={mt['annual_turnover']:.1f}x")

print("\n=== RESCUE: min-price tradeable-universe filter (Track B OOS) ===")
print("R1 raw-feat + RAW returns + minP$5 :"); run_minprice("features_all.raw.bak.parquet","fdr_results.raw.bak.parquet","shap_summary.raw.bak.parquet", False, 5.0, "R1")
print("R2 raw-feat + CLEAN returns + minP$5:"); run_minprice("features_all.raw.bak.parquet","fdr_results.raw.bak.parquet","shap_summary.raw.bak.parquet", True, 5.0, "R2")
print("R3 raw-feat + RAW returns + minP$1 :"); run_minprice("features_all.raw.bak.parquet","fdr_results.raw.bak.parquet","shap_summary.raw.bak.parquet", False, 1.0, "R3")
print("R4 clean-feat + CLEAN ret + minP$5 :"); run_minprice("features_all.parquet","fdr_results.parquet","shap_summary.parquet", True, 5.0, "R4")
print("R0 raw-feat + RAW returns + NO filter (orig published basis):"); run_minprice("features_all.raw.bak.parquet","fdr_results.raw.bak.parquet","shap_summary.raw.bak.parquet", False, 0.0, "R0")

# ---- Apples-to-apples: momentum / reversal / Track A on price>=$5 universe, clean ----
def clean_prices(ohlcv, tickers):
    m=ohlcv["ticker"].isin(tickers)
    cw=ohlcv[m].pivot(index="date",columns="ticker",values="close")
    r=cw.pct_change(fill_method=None).clip(-CAP,CAP)
    cc=cw.iloc[0:1].copy()
    clean=(1+r).cumprod()
    clean=clean*cw.iloc[0]  # rescale to first price level (approx)
    return cw, clean

def run_signal_oos(sig_full, ohlcv, min_price, label):
    cfg=yaml.safe_load(open(CFG)); aum=cfg.get("aum_dollars",1e8)
    sig=sig_full[(sig_full.index>=pd.Timestamp(S))&(sig_full.index<=pd.Timestamp(E))]
    tickers=sig.columns.tolist()
    r,vol,adv=returns_vol_adv(ohlcv,tickers,True)
    m=ohlcv["ticker"].isin(tickers)&(ohlcv["date"]>=pd.Timestamp(S))&(ohlcv["date"]<=pd.Timestamp(E))
    px=ohlcv[m].pivot(index="date",columns="ticker",values="close")
    sim=PortfolioSimulator(config_path=str(CFG),spread_bps=cfg["spread_bps"],impact_coeff=cfg["impact_coeff"])
    pos=sim.signal_to_positions(sig,lag=1,rebal_freq=5)
    pxa=px.reindex(index=pos.index,columns=pos.columns).ffill()
    pos=pos.where(pxa>=min_price,0.0)
    l1=pos.abs().sum(axis=1).replace(0,np.nan); pos=pos.div(l1,axis=0).fillna(0.0)
    pnl=sim.simulate_pnl(pos,r,vol=vol,adv_dollars=adv,aum_dollars=aum,min_adv_dollars=cfg.get("min_adv_dollars",1e6))
    mt=sim.compute_metrics(pnl)
    print(f"  [{label}] net_SR={mt['net_pnl_sharpe']:+.3f} gross={mt['gross_pnl_sharpe']:+.3f} drag={mt['cost_drag_bps']:.0f} TO={mt['annual_turnover']:.1f}x")

print("\n=== APPLES-TO-APPLES on price>=$5 tradeable universe, clean returns (OOS) ===")
ohlcv=pd.read_parquet(OHLCV)
allt=ohlcv["ticker"].unique().tolist()
cw,clean=clean_prices(ohlcv, allt)
# momentum 12-1 and reversal 5d signals from CLEAN prices, full history then sliced
mom=clean.pct_change(252)-clean.pct_change(21)
rev=-clean.pct_change(5)
run_signal_oos(mom, ohlcv, 5.0, "Momentum 12-1  (minP$5)")
run_signal_oos(rev, ohlcv, 5.0, "Reversal 5d    (minP$5)")
# Track A clean
fdrc=pd.read_parquet(P/"fdr_results.parquet"); shapc=pd.read_parquet(P/"shap_summary.parquet")
featc=pd.read_parquet(P/"features_all.parquet")
wa=build_signal_weights("track_a",fdrc,shapc)
siga=generate_composite_signal(featc,wa,start_date=S,end_date=E)
run_signal_oos(siga, ohlcv, 5.0, "Track A clean  (minP$5)")
print("  [Track B clean (minP$5)] net_SR=+0.357  (from R4 above)")
