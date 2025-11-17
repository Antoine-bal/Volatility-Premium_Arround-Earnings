import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, statsmodels.api as sm
from datetime import datetime
from math import log, sqrt, exp, erf
from pandas.tseries.offsets import BDay
import pyarrow.dataset as ds
import pyarrow.compute as pc
from collections import defaultdict

aapl_path = r"C:\Users\antoi\Documents\Antoine\Projets_Python\Trading Vol on Earnings\alpha_options_raw\AAPL_options_2020-01-01_to_2025-11-16.parquet"
df = pd.read_parquet(aapl_path)
# ========= Preparation from raw data =========
path = r"C:\Users\antoi\Documents\Antoine\Projets_Python\Trading Vol on Earnings\exports\option_chain.parquet"

dataset = ds.dataset(path, format="parquet")

#######################  Creating usable csv from raw data  #######################################
#
# path = r"C:\Users\antoi\Documents\Antoine\Projets_Python\Trading Vol on Earnings\exports\option_chain.parquet"
#
tickers = [
    "NVDA", "MSFT", "AAPL", "AMZN", "META", "AVGO", "GOOGL", "GOOG", "BRK.B", "TSLA",
    "JPM", "V", "LLY", "NFLX", "XOM", "MA", "WMT", "COST", "ORCL", "JNJ",
    "HD", "PG", "ABBV", "BAC", "UNH",
    "CRM", "ADBE", "PYPL", "AMD", "INTC", "CSCO", "MCD", "NKE", "WFC", "CVX",
    "PEP", "KO", "DIS", "BA", "MRK", "MO", "IBM", "T", "GM", "CAT", "UPS", "DOW",
    "PLTR", "TXN", "LIN", "AMAT"
]
cols = [
    "date", "act_symbol", "expiration", "strike",
    "call_put", "bid", "ask", "vol", "delta",
    "gamma", "theta", "vega", "rho"
]

# dataset = ds.dataset(path, format="parquet")
#
# filt = pc.field("act_symbol").isin(tickers)
#
# table = dataset.to_table(columns=cols, filter=filt)
# df_all = table.to_pandas()
#
# print("Final:", df_all.shape)
#
# df_all.to_csv('my_data.csv')

#######################  LTTLE ANALYSIS #######################################
# Existing setup
# UNIVERSE = [
#     "NVDA","MSFT","AAPL","AMZN","META","AVGO","GOOGL","GOOG","BRK.B","TSLA","JPM","V","LLY","NFLX","XOM","MA","WMT",
#     "COST","ORCL","JNJ","HD","PG","ABBV","BAC","UNH","CRM","ADBE","PYPL","AMD","INTC","CSCO","MCD","NKE","WFC","CVX",
#     "PEP","KO","DIS","BA","MRK","MO","IBM","T","GM","CAT","UPS","DOW","PLTR","TXN","LIN","AMAT"
# ]
#
# cols = ["act_symbol", "date", "expiration"]
# filt = pc.field("act_symbol").isin(UNIVERSE)
#
# dataset = ds.dataset(path, format="parquet")
#
# # ⬇ keeps your original metric
# min_dte_per_symbol = defaultdict(lambda: None)
#
# # ⬇ NEW: for each symbol we track: min(# expiries on a given date)
# min_exp_count_per_symbol = defaultdict(lambda: None)
#
# for batch in dataset.to_batches(columns=cols, filter=filt):
#     df = batch.to_pandas()
#     if df.empty:
#         continue
#
#     df["date"] = pd.to_datetime(df["date"]).dt.normalize()
#     df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()
#     df["DTE"] = (df["expiration"] - df["date"]).dt.days
#     df = df[df["DTE"] > 0]
#     df = df[df['date']>= pd.Timestamp('2024-01-01')]
#
#     # ----- Update DTE metric -----
#     grouped_dte = df.groupby("act_symbol")["DTE"].min()
#     for sym, dte_min in grouped_dte.items():
#         current = min_dte_per_symbol[sym]
#         if current is None or dte_min < current:
#             min_dte_per_symbol[sym] = dte_min
#
#     # ----- NEW: expiration count per day -----
#     # For each (symbol, date), count distinct expiries
#     grouped_exp = df.groupby(["act_symbol", "date"])["expiration"].nunique()
#
#     # grouped_exp looks like:
#     #   act_symbol  date        -> number_of_expiries
#     # We update the minima per symbol
#     for (sym, _day), exp_count in grouped_exp.items():
#         current = min_exp_count_per_symbol[sym]
#         if current is None or exp_count < current:
#             min_exp_count_per_symbol[sym] = exp_count
#
# # ----- Final join into a single DataFrame -----
# rows = []
# for sym in sorted(min_dte_per_symbol.keys()):
#     rows.append({
#         "Symbol": sym,
#         "Min_DTE": min_dte_per_symbol[sym],
#         "Min_ExpiryCount_PerDay": min_exp_count_per_symbol.get(sym)
#     })
#
# result_df = pd.DataFrame(rows).sort_values(["Min_ExpiryCount_PerDay", "Min_DTE"]).reset_index(drop=True)
#
# print(result_df)


# df_all = table.to_pandas()
#
# print("Final:", df_all.shape)
#
# df_all.to_csv('my_data.csv')

# ========= Config =========
SPOT_CSV = "spot_data.csv"
EARNINGS_CSV = "earnings.csv"
OPTIONS_DIR = "options"
OPTIONS_SINGLE_CSV = "big_options_clean.csv"
CACHE_DIR = "cache_iv"
OUT_EXCEL = "earnings_iv_analysis.xlsx"
os.makedirs(CACHE_DIR, exist_ok=True)

UNIVERSE = [
    "NVDA","MSFT","AAPL","AMZN","META","AVGO","GOOGL","GOOG","BRK.B","TSLA","JPM","V","LLY","NFLX","XOM","MA","WMT",
    "COST","ORCL","JNJ","HD","PG","ABBV","BAC","UNH","CRM","ADBE","PYPL","AMD","INTC","CSCO","MCD","NKE","WFC","CVX",
    "PEP","KO","DIS","BA","MRK","MO","IBM","T","GM","CAT","UPS","DOW","PLTR","TXN","LIN","AMAT"
]

YEARS = 3
PRE, POST = 10, 10
BASELINE_PRE_TD, BASELINE_POST_TD = 30, 10
MAX_ANCHOR_LAG_DAYS = 3
BASE_PRE_MAT_DAYS = 30
R_RATE, Q_DIV = 0.03, 0.0
USE_CACHE, CACHE_PREFIX = True, "first_project"
KEEP_OPT_DATA_IN_RAM = True

plt.rcParams.update({"figure.figsize": (10,5), "axes.grid": True, "font.size": 12})

# ========= Black–Scholes / IV =========
def _norm_pdf(x): return (1.0/np.sqrt(2*np.pi))*np.exp(-0.5*x*x)
def _norm_cdf(x): return 0.5*(1.0+erf(x/np.sqrt(2.0)))
def _bs_price(S,K,T,r,q,sigma,is_call):
    if T <= 0: return max(0.0,S-K) if is_call else max(0.0,K-S)
    sigma = max(sigma,1e-8)
    d1 = (log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*sqrt(T)); d2 = d1 - sigma*sqrt(T)
    return (S*exp(-q*T)*_norm_cdf(d1) - K*exp(-r*T)*_norm_cdf(d2)) if is_call else (K*exp(-r*T)*_norm_cdf(-d2) - S*exp(-q*T)*_norm_cdf(-d1))
def _vega(S,K,T,r,q,sigma):
    if T<=0 or sigma<=1e-8: return 0.0
    d1 = (log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*sqrt(T))
    return S*exp(-q*T)*sqrt(T)*_norm_pdf(d1)
def _iv_from_price(target,S,K,T,is_call,r=R_RATE,q=Q_DIV,sigma0=0.3,tol=1e-6,max_iter=40):
    if T<=0 or target<=0 or S<=0 or K<=0: return np.nan
    intrinsic = max(0.0,(S*exp(-q*T)-K*exp(-r*T)) if is_call else (K*exp(-r*T)-S*exp(-q*T)))
    if target <= intrinsic+1e-10: return 0.01
    lo,hi = 1e-4,5.0; sigma = float(np.clip(sigma0,lo,hi))
    for _ in range(max_iter):
        px = _bs_price(S,K,T,r,q,sigma,is_call); diff = px-target
        if abs(diff)<tol: return float(sigma)
        v = _vega(S,K,T,r,q,sigma)
        if v<=1e-8 or not np.isfinite(v):
            hi,sigma = (sigma,0.5*(lo+sigma)) if diff>0 else (hi,0.5*(sigma+hi)); lo = lo if diff>0 else sigma
            continue
        sigma_new = sigma - diff/v
        if not np.isfinite(sigma_new) or sigma_new<=lo or sigma_new>=hi:
            hi,sigma = (sigma,0.5*(lo+sigma)) if diff>0 else (hi,0.5*(sigma+hi)); lo = lo if diff>0 else sigma
        else:
            sigma = sigma_new
    if sigma > 1.5:
        yo = 2+2
    return float(sigma if np.isfinite(sigma) else np.nan)

# ========= Spot loader =========
def load_spot_close_long(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]); df["date"] = df["date"].dt.normalize()
    px = "adj_close" if "adj_close" in df.columns else ("close" if "close" in df.columns else None)
    if px is None: raise ValueError("spot_data.csv must contain 'close' or 'adj_close'.")
    return (df[["symbol","date",px]].rename(columns={px:"spot"})
            .dropna(subset=["spot"]).set_index(["symbol","date"]).sort_index())

# ========= ATM helper =========
def pick_atm_row(sub: pd.DataFrame):
    if sub.empty: return None
    if "moneyness" not in sub.columns: sub = sub.assign(moneyness=sub["strike"]/sub["spot"])
    sub = sub.assign(abs_mny=(1.0-sub["moneyness"]).abs())
    return sub.loc[sub["abs_mny"].idxmin()]

# ========= Options loader + IV =========
def _load_raw_options_for_ticker(tkr: str) -> pd.DataFrame:
    if OPTIONS_SINGLE_CSV and os.path.exists(OPTIONS_SINGLE_CSV):
        df = pd.read_csv(OPTIONS_SINGLE_CSV)
        if "symbol" not in df.columns: raise ValueError("OPTIONS_SINGLE_CSV must have 'symbol'.")
        df = df[df["symbol"]==tkr].copy()
    else:
        path = os.path.join(OPTIONS_DIR,f"{tkr}_options.csv")
        if not os.path.exists(path): print(f"[{tkr}] no options file"); return pd.DataFrame()
        df = pd.read_csv(path)
    if df.empty: return df
    if "date" in df.columns: df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    elif "time" in df.columns: df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    else: raise ValueError(f"[{tkr}] no 'date'/'time'")
    if "expiry" not in df.columns and "expiration" in df.columns: df["expiry"] = pd.to_datetime(df["expiration"]).dt.normalize()
    elif "expiry" in df.columns: df["expiry"] = pd.to_datetime(df["expiry"]).dt.normalize()
    else: raise ValueError(f"[{tkr}] no 'expiry'/'expiration'")
    if "strike" not in df.columns: raise ValueError(f"[{tkr}] no 'strike'")
    df["strike"] = df["strike"].astype(float)
    if "is_call" in df.columns: df["is_call"] = df["is_call"].astype(bool)
    else:
        right_col = next((c for c in ["right","option_type","cp_flag"] if c in df.columns), None)
        if right_col is None: raise ValueError(f"[{tkr}] no call/put flag")
        df[right_col] = df[right_col].astype(str).str.upper().str.strip()
        df["is_call"] = df[right_col].isin(["C","CALL"])
    cols = {c.lower():c for c in df.columns}
    if "bid" in cols and "ask" in cols:
        df["mid"] = 0.5*(df[cols["bid"]].astype(float)+df[cols["ask"]].astype(float))
    else:
        price_col = next((cols[c] for c in ["last","lastprice","close","settle","price"] if c in cols), None)
        if price_col is None: raise ValueError(f"[{tkr}] no price cols")
        df["mid"] = df[price_col].astype(float)
    df = df[df["mid"]>0].copy()
    return df

def get_options_with_iv_local(tkr, close_long, start, end, r=R_RATE, q_div=Q_DIV,
                              use_cache=USE_CACHE, cache_prefix=CACHE_PREFIX):
    cache_key = os.path.join(CACHE_DIR,f"{cache_prefix}_opt_{tkr}_{start:%Y%m%d}_{end:%Y%m%d}.parquet")
    if use_cache and os.path.exists(cache_key):
        try:
            df = pd.read_parquet(cache_key)
            if not df.empty and "iv" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.normalize()
                df["expiry"] = pd.to_datetime(df["expiry"]).dt.normalize()
                return df
        except Exception as e: print(f"[{tkr}] cache read fail: {e}")
    df = _load_raw_options_for_ticker(tkr)
    if df.empty: print(f"[{tkr}] no raw options"); return pd.DataFrame()
    df = df[(df["date"]>=start) & (df["date"]<=end)].copy()
    if df.empty: print(f"[{tkr}] no options in range"); return pd.DataFrame()
    if not isinstance(close_long.index,pd.MultiIndex): raise ValueError("close_long must be MultiIndex")
    key = pd.MultiIndex.from_arrays([np.full(len(df),tkr),df["date"]],names=["symbol","date"])
    if (tkr,"spot") or (tkr,) in close_long.columns:
        pass
    try: df["spot"] = close_long.reindex(key)["spot"].values
    except:
        if tkr in close_long.columns: df["spot"] = close_long.reindex(df["date"])[tkr].values
        else: print(f"[{tkr}] no spot in close_long"); return pd.DataFrame()
    df = df.dropna(subset=["spot"]);
    if df.empty: print(f"[{tkr}] no rows after merge spot"); return pd.DataFrame()
    df["dte"] = (df["expiry"]-df["date"]).dt.days; df = df[df["dte"]>0].copy()
    if df.empty: print(f"[{tkr}] no positive DTE"); return pd.DataFrame()
    df["iv"] = [ _iv_from_price(float(rw["mid"]),float(rw["spot"]),float(rw["strike"]),
                                 max(1e-6,rw["dte"]/365.0),bool(rw["is_call"]),r,q_div)
                 for _,rw in df.iterrows() ]
    df = df.replace([np.inf,-np.inf],np.nan).dropna(subset=["iv"])
    if df.empty: print(f"[{tkr}] no valid IV"); return pd.DataFrame()
    df["moneyness"] = df["strike"]/df["spot"]; df["symbol_under"] = tkr
    df = df[["date","expiry","strike","is_call","spot","mid","iv","dte","moneyness","symbol_under"]]\
           .sort_values(["date","expiry","strike"]).reset_index(drop=True)
    if use_cache:
        try: df.to_parquet(cache_key,index=False); print(f"[{tkr}] cached -> {cache_key}")
        except Exception as e: print(f"[{tkr}] cache save fail: {e}")
    return df

# ========= Build 3-case panels for one ticker =========
def build_atm_panels_for_ticker(opt_df, earn_tkr, pre, post,
                                base_pre_mat_days, baseline_pre_td,
                                baseline_post_td, max_anchor_lag_days):
    if opt_df.empty or earn_tkr.empty: return {1:pd.DataFrame(),2:pd.DataFrame(),3:pd.DataFrame()}
    df = opt_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.normalize()
    trading_days = pd.DatetimeIndex(sorted(df["date"].unique()))
    if len(trading_days)==0: return {1:pd.DataFrame(),2:pd.DataFrame(),3:pd.DataFrame()}
    panels = {1:[],2:[],3:[]}
    for _,ev in earn_tkr.iterrows():
        symbol = ev["symbol"]; event_day = pd.to_datetime(ev["event_day"]).normalize()
        idx = trading_days.searchsorted(event_day)
        cand = [i for i in [idx,idx-1,idx+1] if 0<=i<len(trading_days)]
        if not cand: continue
        anchor = min((trading_days[i] for i in cand), key=lambda d: abs((d-event_day).days))
        if abs((anchor-event_day).days)>max_anchor_lag_days: continue
        anchor_idx = int(np.where(trading_days==anchor)[0][0])
        pre_idx = max(0,anchor_idx-pre); pre_date = trading_days[pre_idx]
        expiries_pre = sorted(df.loc[df["date"]==pre_date,"expiry"].unique())
        if not expiries_pre: continue
        future_exps = [e for e in expiries_pre if e>event_day]
        if not future_exps: continue
        exp1 = next((e for e in future_exps if e >= (event_day+BDay(6)).normalize()),None)
        exp2 = next((e for e in future_exps if (e-event_day).days>=30),None)
        exp3 = next((e for e in future_exps if (e-event_day).days>=90),None)
        expiry_by_case = {1:exp1,2:exp2,3:exp3}
        if all(v is None for v in expiry_by_case.values()): continue
        for case in [1,2,3]:
            target = expiry_by_case[case]
            if target is None: continue
            b_left,b_right = max(0,anchor_idx-baseline_pre_td),max(0,anchor_idx-baseline_post_td)
            baseline_iv = np.nan
            if b_right>b_left:
                ivs = []
                for d in trading_days[b_left:b_right]:
                    row = pick_atm_row(df[(df["date"]==d)&(df["expiry"]==target)])
                    if row is not None and np.isfinite(row["iv"]): ivs.append(float(row["iv"]))
                if ivs: baseline_iv = float(np.mean(ivs))
            w_left,w_right = max(0,anchor_idx-pre),min(len(trading_days)-1,anchor_idx+post)
            for i in range(w_left, w_right + 1):
                d = trading_days[i]
                row = pick_atm_row(df[(df["date"] == d) & (df["expiry"] == target)])
                if row is None or not np.isfinite(row["iv"]):
                    continue

                iv_val = float(row["iv"])
                rel = i - anchor_idx

                panels[case].append({
                    "symbol": symbol,
                    "event_day": event_day,
                    "case": case,
                    "target_expiry": target,
                    "date": d,
                    "rel": rel,
                    "iv": iv_val,
                    "baseline_IV": baseline_iv,
                    "abn_IV": iv_val - baseline_iv if np.isfinite(baseline_iv) else np.nan,
                    # <<< NEW FIELDS >>>
                    "spot": float(row["spot"]),
                    "strike": float(row["strike"]),
                    "moneyness": float(row["moneyness"]),
                    "is_call": bool(row["is_call"]),
                })
    for c in [1,2,3]:
        panels[c] = (pd.DataFrame(panels[c])
                     .sort_values(["symbol","event_day","date","rel"])
                     .reset_index(drop=True)) if panels[c] else pd.DataFrame()
    return panels

def build_opt_and_panels_for_ticker(tkr, earn_all, close_long,
                                    start, end, pre, post,
                                    baseline_pre_td, baseline_post_td,
                                    max_anchor_lag_days, base_pre_mat_days,
                                    keep_opt_data_in_ram, use_cache,
                                    cache_prefix, r, q_div):
    earn_tkr = earn_all[earn_all["symbol"]==tkr]
    if earn_tkr.empty:
        print(f"{tkr}: no earnings"); empty = {1:pd.DataFrame(),2:pd.DataFrame(),3:pd.DataFrame()}; return pd.DataFrame(),empty
    opt_df = get_options_with_iv_local(tkr,close_long,start,end,r,q_div,use_cache,cache_prefix)
    if opt_df.empty:
        print(f"{tkr}: no IV surface"); empty = {1:pd.DataFrame(),2:pd.DataFrame(),3:pd.DataFrame()}; return pd.DataFrame(),empty
    print(f"{tkr}: {len(opt_df):,} option rows")
    panels = build_atm_panels_for_ticker(opt_df,earn_tkr,pre,post,base_pre_mat_days,
                                         baseline_pre_td,baseline_post_td,max_anchor_lag_days)
    return (opt_df if keep_opt_data_in_ram else pd.DataFrame()), panels

# ========= Load spot & earnings =========
close_long = load_spot_close_long(SPOT_CSV)
END_DATE = close_long.index.get_level_values("date").max()
START_DATE = END_DATE - pd.Timedelta(days=365*YEARS)
print(f"Spot window: {START_DATE.date()} to {END_DATE.date()}")

earn = pd.read_csv(EARNINGS_CSV)
earn["symbol"] = earn["symbol"].astype(str).str.upper()
earn["event_day"] = pd.to_datetime(earn["event_day"]).dt.normalize()
earn = (earn[earn["symbol"].isin(UNIVERSE) &
             (earn["event_day"]>=START_DATE) &
             (earn["event_day"]<=END_DATE)]
        .drop_duplicates(subset=["symbol","event_day"])
        .reset_index(drop=True))
print(f"Earnings events: {len(earn)}")

idx = close_long.index
mask = idx.get_level_values("symbol").isin(UNIVERSE) & \
       (idx.get_level_values("date") >= START_DATE) & \
       (idx.get_level_values("date") <= END_DATE)
close_long = close_long[mask].sort_index()
print("close_long:", close_long.shape)

# ========= Build PANELS for all tickers =========
OPT_DATA, PANELS = {}, {}
for tkr in UNIVERSE:
    print(f"\nProcessing {tkr}...")
    opt_df_out, panels_tkr = build_opt_and_panels_for_ticker(
        tkr, earn, close_long, START_DATE, END_DATE,
        PRE, POST, BASELINE_PRE_TD, BASELINE_POST_TD,
        MAX_ANCHOR_LAG_DAYS, BASE_PRE_MAT_DAYS,
        KEEP_OPT_DATA_IN_RAM, USE_CACHE, CACHE_PREFIX, R_RATE, Q_DIV
    )
    OPT_DATA[tkr], PANELS[tkr] = opt_df_out, panels_tkr
print("\nAll tickers processed.")

# ========= panel_clean =========
clean_rows = []
for symbol,cases in PANELS.items():
    for case_id,df_case in cases.items():
        if df_case is None or df_case.empty: continue
        df = df_case.copy()
        df["Symbol"] = df["symbol"].astype(str)
        df["EventDate"] = pd.to_datetime(df["event_day"]).dt.normalize()
        df["CaseID"] = int(case_id)
        df["MaturityDate"] = pd.to_datetime(df["target_expiry"]).dt.normalize()
        df["ObsDate"] = pd.to_datetime(df["date"]).dt.normalize()
        df["t_rel"] = df["rel"].astype(int)
        df["IV"] = df["iv"]; df["IV_base"] = df["baseline_IV"]; df["IV_abn"] = df["abn_IV"]
        df["TTM_yrs"] = (df["MaturityDate"]-df["ObsDate"]).dt.days/365.0
        clean_rows.append(df[["Symbol","EventDate","CaseID","MaturityDate","ObsDate",
                              "t_rel","TTM_yrs","IV","IV_base","IV_abn"]])
panel_clean = (pd.concat(clean_rows,ignore_index=True)
               .sort_values(["Symbol","EventDate","CaseID","ObsDate","t_rel"])
               .reset_index(drop=True)) if clean_rows else \
              pd.DataFrame(columns=["Symbol","EventDate","CaseID","MaturityDate","ObsDate",
                                    "t_rel","TTM_yrs","IV","IV_base","IV_abn"])
print("panel_clean:", panel_clean.shape)

# ========= event_features =========
spot_df = pd.read_csv(SPOT_CSV,parse_dates=["date"])
spot_df["date"] = spot_df["date"].dt.normalize()
px = "adj_close" if "adj_close" in spot_df.columns else ("close" if "close" in spot_df.columns else None)
if px is None: raise ValueError("spot_data.csv must contain 'close' or 'adj_close'.")
spot_df = (spot_df[["symbol","date",px]]
           .rename(columns={"symbol":"Symbol","date":"Date",px:"Close"})
           .set_index(["Symbol","Date"]).sort_index())

anchors = (panel_clean[panel_clean["t_rel"]==0]
           .groupby(["Symbol","EventDate"])["ObsDate"]
           .min().rename("AnchorDate").reset_index())

pc1 = panel_clean[panel_clean["CaseID"]==1].copy()
pre_c1 = (pc1[pc1["t_rel"]==-1]
          .groupby(["Symbol","EventDate"],as_index=False)
          .agg({"MaturityDate":"first","IV":"mean","IV_base":"mean","IV_abn":"mean","TTM_yrs":"mean"})
          .rename(columns={"MaturityDate":"Maturity_pre","IV":"IV_pre",
                           "IV_base":"IV_base_pre","IV_abn":"IV_abn_pre","TTM_yrs":"TTM_pre_yrs"}))
post_c1 = (pc1[pc1["t_rel"]==1]
           .groupby(["Symbol","EventDate"],as_index=False)
           .agg({"IV":"mean","IV_abn":"mean"})
           .rename(columns={"IV":"IV_post","IV_abn":"IV_abn_post"}))

pc3 = panel_clean[panel_clean["CaseID"]==3].copy()
pre_c3 = (pc3[pc3["t_rel"]==-1]
          .groupby(["Symbol","EventDate"],as_index=False)
          .agg({"IV":"mean"})
          .rename(columns={"IV":"IV_pre_long"}))

event_features = (anchors
    .merge(pre_c1,on=["Symbol","EventDate"],how="left")
    .merge(post_c1,on=["Symbol","EventDate"],how="left")
    .merge(pre_c3,on=["Symbol","EventDate"],how="left"))
event_features = event_features[~event_features["IV_pre"].isna()].copy()
event_features["Crush_1d"] = event_features["IV_post"] - event_features["IV_pre"]
event_features["TermSlope_pre"] = event_features["IV_pre"] - event_features["IV_pre_long"]

def _take_close(symbols,dates):
    idx = pd.MultiIndex.from_arrays([symbols,dates],names=["Symbol","Date"])
    return spot_df.reindex(idx)["Close"].to_numpy()

event_features["S0"] = _take_close(event_features["Symbol"].values, event_features["AnchorDate"].values)
event_features["AnchorDate_plus1"] = event_features["AnchorDate"] + BDay(1)
event_features["S1"] = _take_close(event_features["Symbol"].values, event_features["AnchorDate_plus1"].values)
event_features["R_0_1"] = np.log(event_features["S1"]/event_features["S0"])
event_features["EV_1d"] = event_features["IV_pre"]*np.sqrt(1/252)
event_features["VRP_1d"] = event_features["EV_1d"] - event_features["R_0_1"].abs()
event_features = (event_features[[
    "Symbol","EventDate","AnchorDate","Maturity_pre","TTM_pre_yrs",
    "IV_base_pre","IV_pre","IV_abn_pre","IV_post","IV_abn_post",
    "Crush_1d","IV_pre_long","TermSlope_pre","S0","S1","R_0_1","EV_1d","VRP_1d"
]].sort_values(["Symbol","EventDate"]).reset_index(drop=True))
print("event_features:", event_features.shape)

# ========= Plot data for Excel =========
iv_abn = event_features["IV_abn_pre"].dropna()
bins_iv = np.linspace(iv_abn.min(),iv_abn.max(),51)
df_iv_hist = pd.DataFrame({"BinLeft":bins_iv[:-1],"BinRight":bins_iv[1:],
                           "Count":np.histogram(iv_abn,bins=bins_iv)[0]})
df_iv_vs_crush = event_features[["IV_abn_pre","Crush_1d"]].dropna()
df_termslope_vs_crush = event_features[["TermSlope_pre","Crush_1d"]].dropna()
vrp = event_features["VRP_1d"].dropna()
bins_vrp = np.linspace(vrp.min(),vrp.max(),51)
df_vrp_hist = pd.DataFrame({"BinLeft":bins_vrp[:-1],"BinRight":bins_vrp[1:],
                            "Count":np.histogram(vrp,bins=bins_vrp)[0]})



# ========= Single Excel export (all sheets) =========
with pd.ExcelWriter(OUT_EXCEL, engine="xlsxwriter") as w:
    # raw panels per ticker/case
    for tkr, cases in PANELS.items():
        for case_id, df_case in cases.items():
            if df_case is not None and not df_case.empty:
                sheet = f"{tkr}_c{case_id}"[:31]

                cols = [
                    "symbol",
                    "event_day",
                    "case",
                    "target_expiry",
                    "date",
                    "rel",
                    "iv",
                    "baseline_IV",
                    "abn_IV",
                    "spot",
                    "strike",
                    "moneyness",
                    "is_call",
                ]

                # keep only the requested columns in that order
                df_case[cols].to_excel(w, sheet_name=sheet, index=False)
    panel_clean.to_excel(w, sheet_name="panel_clean", index=False)
    event_features.to_excel(w, sheet_name="event_features", index=False)
    df_iv_hist.to_excel(w, sheet_name="plot_IVabn_hist", index=False)
    df_iv_vs_crush.to_excel(w, sheet_name="plot_IVabn_vs_Crush", index=False)
    df_termslope_vs_crush.to_excel(w, sheet_name="plot_TermSlope_vs_Crush", index=False)
    df_vrp_hist.to_excel(w, sheet_name="plot_VRP1d_hist", index=False)

    # ========= Regression + signals =========
    df_reg = event_features.dropna(subset=["Crush_1d", "IV_abn_pre", "TermSlope_pre", "VRP_1d", "R_0_1"]).copy()
    for c in ["IV_abn_pre", "TermSlope_pre", "VRP_1d"]:
        df_reg[c + "_z"] = (df_reg[c] - df_reg[c].mean()) / df_reg[c].std()
    y = df_reg["Crush_1d"]
    X = sm.add_constant(df_reg[["IV_abn_pre_z", "TermSlope_pre_z", "VRP_1d_z", "R_0_1"]])
    model = sm.OLS(y, X).fit(cov_type="HC3")
    print(model.summary())
    coef_table = model.summary2().tables[1].reset_index()
    coef_table = coef_table.rename(columns={"index": "Variable"})

    df_reg["Signal_VolBuildup"] = -df_reg["IV_abn_pre_z"]
    df_reg["Signal_TermSlope"] = -df_reg["TermSlope_pre_z"]
    df_reg["Signal_Composite"] = 0.6 * df_reg["Signal_VolBuildup"] + 0.4 * df_reg["Signal_TermSlope"]
    df_reg["ImpliedMove_1d"] = df_reg["EV_1d"]
    df_reg["RealizedMove_1d"] = df_reg["R_0_1"].abs()
    df_reg["IVCrush_Impact"] = -df_reg["Crush_1d"] * np.sqrt(1 / 252)
    df_reg["PnL_proxy"] = df_reg["IVCrush_Impact"] + df_reg["RealizedMove_1d"] - df_reg["ImpliedMove_1d"]
    df_reg["Signal_quintile"] = pd.qcut(df_reg["Signal_Composite"], 5, labels=False) + 1
    perf = df_reg.groupby("Signal_quintile")["PnL_proxy"].mean().rename("MeanPnL").to_frame()
    print(perf)
    df_reg.to_excel(w, sheet_name="regression_data", index=False)
    perf.to_excel(w, sheet_name="signal_perf")
    # <<< NEW: write OLS text summary >>>
    coef_table.to_excel(w, sheet_name="OLS_coef_table", index=False)

print(f"All outputs written to {OUT_EXCEL}")
