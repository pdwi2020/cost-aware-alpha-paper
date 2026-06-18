"""Authoritative IS+OOS results under Screen 0 (price>=$5) + clean returns,
for the manuscript baselines table. All strategies, apples-to-apples."""
import sys, warnings; from pathlib import Path
import numpy as np, pandas as pd, yaml
warnings.filterwarnings("ignore")
ROOT=Path(__file__).resolve().parent.parent.parent; sys.path.insert(0,str(ROOT))
from src.backtest.portfolio import PortfolioSimulator
from src.backtest.generate_signals import build_signal_weights, generate_composite_signal
P=ROOT/"data"/"processed"; CFG=ROOT/"configs"/"backtest.yaml"; OHLCV=P/"daily_ohlcv.parquet"
CAP=0.50; MINP=5.0
WINDOWS={"IS":("2013-01-01","2021-12-31"), "OOS":("2022-01-01","2024-12-31")}
ohlcv=pd.read_parquet(OHLCV); cfg=yaml.safe_load(open(CFG)); aum=cfg.get("aum_dollars",1e8)

def rva(tickers,s,e):
    m=ohlcv["ticker"].isin(tickers)&(ohlcv["date"]>=pd.Timestamp(s))&(ohlcv["date"]<=pd.Timestamp(e))
    sub=ohlcv[m]; cw=sub.pivot(index="date",columns="ticker",values="close"); vw=sub.pivot(index="date",columns="ticker",values="volume")
    r=cw.pct_change(fill_method=None).clip(-CAP,CAP); vol=r.rolling(21,min_periods=10).std(); adv=(cw*vw).rolling(21,min_periods=10).mean()
    mk=(r.index>=pd.Timestamp(s))&(r.index<=pd.Timestamp(e)); return r[mk],vol[mk],adv[mk],cw[mk]
def clean_close(tickers):
    m=ohlcv["ticker"].isin(tickers); cw=ohlcv[m].pivot(index="date",columns="ticker",values="close")
    r=cw.pct_change(fill_method=None).clip(-CAP,CAP); cc=(1+r.fillna(0)).cumprod().multiply(cw.iloc[0],axis="columns"); cc[cw.isna()]=np.nan; return cc
def runp(sig_full,s,e,label):
    sig=sig_full[(sig_full.index>=pd.Timestamp(s))&(sig_full.index<=pd.Timestamp(e))]
    tk=sig.columns.tolist(); r,vol,adv,cw=rva(tk,s,e)
    sim=PortfolioSimulator(config_path=str(CFG),spread_bps=cfg["spread_bps"],impact_coeff=cfg["impact_coeff"])
    pos=sim.signal_to_positions(sig,lag=1,rebal_freq=5)
    pxa=cw.reindex(index=pos.index,columns=pos.columns).ffill(); pos=pos.where(pxa>=MINP,0.0)
    l1=pos.abs().sum(axis=1).replace(0,np.nan); pos=pos.div(l1,axis=0).fillna(0.0)
    pnl=sim.simulate_pnl(pos,r,vol=vol,adv_dollars=adv,aum_dollars=aum,min_adv_dollars=cfg.get("min_adv_dollars",1e6))
    mt=sim.compute_metrics(pnl); return mt['gross_pnl_sharpe'],mt['net_pnl_sharpe'],mt['cost_drag_bps']

fdr=pd.read_parquet(P/"fdr_results.parquet"); shap=pd.read_parquet(P/"shap_summary.parquet"); feat=pd.read_parquet(P/"features_all.parquet")
allt=ohlcv["ticker"].unique().tolist(); cc=clean_close(allt)
mom=cc.pct_change(252)-cc.pct_change(21); rev=-cc.pct_change(5)
wa=build_signal_weights("track_a",fdr,shap); wb=build_signal_weights("track_b",fdr,shap)
print(f"{'Strategy':<22}{'period':<5}{'grossSR':>9}{'netSR':>8}{'drag':>7}")
for label,sig in [("Momentum L/S (12-1)",mom),("Reversal L/S (5d)",rev)]:
    for per,(s,e) in WINDOWS.items():
        g,n,d=runp(sig,s,e,label); print(f"{label:<22}{per:<5}{g:>+9.3f}{n:>+8.3f}{d:>7.0f}")
for label,w in [("Track A (ours)",wa),("Track B (ours)",wb)]:
    for per,(s,e) in WINDOWS.items():
        sig=generate_composite_signal(feat,w,start_date=s,end_date=e)
        g,n,d=runp(sig,s,e,label); print(f"{label:<22}{per:<5}{g:>+9.3f}{n:>+8.3f}{d:>7.0f}")
# SPY
spy=pd.read_parquet(P/"spy_returns.parquet"); sc=spy.columns[0]; sr=spy[sc].squeeze(); sr.index=pd.to_datetime(sr.index)
for per,(s,e) in WINDOWS.items():
    sl=sr[(sr.index>=pd.Timestamp(s))&(sr.index<=pd.Timestamp(e))]; ssr=np.sqrt(252)*sl.mean()/sl.std()
    print(f"{'Buy-and-Hold SPY':<22}{per:<5}{ssr:>+9.3f}{ssr:>+8.3f}{0:>7.0f}")
