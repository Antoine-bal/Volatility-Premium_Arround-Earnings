# signal_generator.py (compact style)
import os, numpy as np, pandas as pd, statsmodels.api as sm

BASE_DIR = r"C:\Users\antoi\Documents\Antoine\Projets_Python\Trading Vol on Earnings"
IN_EXCEL = os.path.join(BASE_DIR, "outputs", "earnings_iv_analysis.xlsx")
OUT_SIGNALS = os.path.join(BASE_DIR, "outputs", "signals_all.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "signals_by_symbol")
OUT_XLS = os.path.join(BASE_DIR, "outputs", "signals_analysis.xlsx")
os.makedirs(OUT_DIR, exist_ok=True)

def zsc(s): m,sig=s.mean(),s.std(); return (s-m)/sig if sig and sig>0 else pd.Series(0.0,index=s.index)

def ols(df,y,x,n):
    df=df.dropna(subset=[y]+x);
    if df.empty: return None,None
    m=sm.OLS(df[y],sm.add_constant(df[x])).fit(cov_type="HC3")
    coef=pd.DataFrame({"model":n,"var":m.params.index,"coef":m.params.values,"t":m.tvalues.values,"p":m.pvalues.values})
    summ=pd.DataFrame({"model":[n],"nobs":[m.nobs],"rsq":[m.rsquared],"rsq_adj":[m.rsquared_adj],"f":[m.fvalue],"f_p":[m.f_pvalue]})
    return coef,summ

def buckets(df,sc,targets,n=10):
    df=df.dropna(subset=[sc]+targets)
    if df.empty: return None
    df["b"]=pd.qcut(df[sc],n,labels=False,duplicates="drop")+1
    return df.groupby("b")[targets].mean().assign(count=df.groupby("b").size()).reset_index().assign(signal=sc)

def build():
    ef=pd.read_excel(IN_EXCEL,"event_features")
    efa=pd.read_excel(IN_EXCEL,"event_features_all_tracks")
    for c in ["EventDate","AnchorDate","AnchorDate_plus1"]:
        if c in ef: ef[c]=pd.to_datetime(ef[c]).dt.normalize()
    efa["EventDate"]=pd.to_datetime(efa["EventDate"]).dt.normalize()
    piv=efa[efa["CaseID"]==1].pivot_table(index=["Symbol","EventDate"],columns="MnyTrackID",values="IV_pre")
    for k in [0,1,2]:
        if k not in piv: piv[k]=np.nan
    piv=piv[[0,1,2]].rename(columns={0:"IV_pre_m0",1:"IV_pre_m1",2:"IV_pre_m2"}).reset_index()
    df=ef.merge(piv,on=["Symbol","EventDate"],how="left")
    df["EM_skew_pre"]=0.5*(df["IV_pre_m1"]+df["IV_pre_m2"])-df["IV_pre_m0"]
    df["EM_skew_rel"]=df["EM_skew_pre"]/df["IV_pre_m0"]
    df["RealizedMove_1d"] = df["R_0_1"].abs()
    df["ImpliedMove_1d"] = df["EV_1d"]
    df["IVCrush_Impact"] = -df["Crush_1d"] * np.sqrt(1 / 252)
    df["PnL_proxy"] = df["IVCrush_Impact"] + df["ImpliedMove_1d"] - df["RealizedMove_1d"]
    for c in ["IV_abn_pre","TermSlope_pre","IV_pre","EM_skew_pre","EM_skew_rel"]:
        df[c+"_z"]=zsc(df[c])
    df["ShortScore_z"]=zsc(df["IV_abn_pre_z"]+df["TermSlope_pre_z"]-df["EM_skew_pre_z"])
    df["LongScore_z"]=zsc(df["EM_skew_pre_z"]-0.5*df["IV_abn_pre_z"])
    df["ShortScore_q"]=pd.qcut(df["ShortScore_z"].rank(method="first"),5,labels=False)+1
    df["LongScore_q"]=pd.qcut(df["LongScore_z"].rank(method="first"),5,labels=False)+1
    short_map={1:0.0,2:0.5,3:1.0,4:1.5,5:2.0}; long_map={1:2.0,2:1.5,3:1.0,4:0.5,5:0.0}
    df["VegaMult_short"]=df["ShortScore_q"].map(short_map).astype(float)
    df["VegaMult_long"]=df["LongScore_q"].map(long_map).astype(float)
    cols=[c for c in df.columns if c in [
        "Symbol","EventDate","AnchorDate","Maturity_pre","TTM_pre_yrs","IV_pre","IV_abn_pre","TermSlope_pre","EV_1d",
        "EM_skew_pre","EM_skew_rel","IV_pre_z","IV_abn_pre_z","TermSlope_pre_z","EM_skew_pre_z","EM_skew_rel_z",
        "ShortScore_z","ShortScore_q","VegaMult_short","LongScore_z","LongScore_q","VegaMult_long",
        "Crush_1d","IV_abn_post","RealizedMove_1d","ImpliedMove_1d","IVCrush_Impact","PnL_proxy"
    ]]
    sig=df[cols].sort_values(["Symbol","EventDate"]).reset_index(drop=True)
    sig.to_csv(OUT_SIGNALS,index=False)
    for s,g in sig.groupby("Symbol"): g.to_csv(os.path.join(OUT_DIR,f"{s}_signals.csv"),index=False)
    reg_df=df.dropna(subset=["Crush_1d","PnL_proxy","IV_abn_pre_z","TermSlope_pre_z","EM_skew_pre_z"]).copy()
    models=[("Crush_IVabn","Crush_1d",["IV_abn_pre_z"]),("Crush_Term","Crush_1d",["TermSlope_pre_z"]),
            ("Crush_EM","Crush_1d",["EM_skew_pre_z"]),("Crush_All","Crush_1d",["IV_abn_pre_z","TermSlope_pre_z","EM_skew_pre_z"]),
            ("PnL_IVabn","PnL_proxy",["IV_abn_pre_z"]),("PnL_Term","PnL_proxy",["TermSlope_pre_z"]),
            ("PnL_EM","PnL_proxy",["EM_skew_pre_z"]),("PnL_All","PnL_proxy",["IV_abn_pre_z","TermSlope_pre_z","EM_skew_pre_z"])]
    coefs=[]; sums=[]
    for n,y,x in models:
        c,s=ols(reg_df,y,x,n)
        if c is not None: coefs.append(c); sums.append(s)
    bucket_cols=["Crush_1d","PnL_proxy","EV_1d","RealizedMove_1d"]
    b1=buckets(df,"IV_abn_pre_z",bucket_cols); b2=buckets(df,"TermSlope_pre_z",bucket_cols)
    b3=buckets(df,"EM_skew_pre_z",bucket_cols); b4=buckets(df,"ShortScore_z",bucket_cols)
    b5=buckets(df,"LongScore_z",bucket_cols)
    corr=df[["IV_abn_pre_z","TermSlope_pre_z","EM_skew_pre_z","PnL_proxy","Crush_1d"]].corr()
    with pd.ExcelWriter(OUT_XLS) as w:
        sig.to_excel(w,"signals",index=False)
        (pd.concat(coefs) if coefs else pd.DataFrame()).to_excel(w,"reg_coef",index=False)
        (pd.concat(sums) if sums else pd.DataFrame()).to_excel(w,"reg_summary",index=False)
        for name,b in [("b_IVabn",b1),("b_Term",b2),("b_EMskew",b3),("b_Short",b4),("b_Long",b5)]:
            (b if b is not None else pd.DataFrame()).to_excel(w,name,index=False)
        corr.to_excel(w,"corr")
    print("[OK] signals and analysis built")

if __name__=="__main__": build()
