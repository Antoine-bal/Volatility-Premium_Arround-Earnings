# earnings_backtest.py

import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import datetime as dt

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from line_profiler import LineProfiler
except ImportError:
    LineProfiler = None

# line_profiler hook: no-op if not under kernprof
try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    def profile(func):
        return func


# ============================================================
# ===================== CONFIG SECTION =======================
# ============================================================

SYMBOLS = [
    "NVDA","MSFT","AAPL","AMZN","META","AVGO","GOOGL","GOOG","BRK.B","TSLA","JPM","V","LLY","NFLX","XOM","MA","WMT",
    "COST","ORCL","JNJ","HD","PG","ABBV","BAC","UNH","CRM","ADBE","PYPL","AMD","INTC","CSCO","MCD","NKE","WFC","CVX",
    "PEP","KO","DIS","BA","MRK","MO","IBM","T","GM","CAT","UPS","DOW","PLTR","TXN","LIN","AMAT"
]

# SYMBOLS = [
#     "NVDA","MSFT"
# ]

START_DATE = pd.Timestamp("2020-01-01")
END_DATE   = pd.Timestamp("2025-11-16")

# Paths
CORP_DIR     = pathlib.Path("alpha_corp_actions")   # *_daily_adjusted.parquet
OPTIONS_DIR  = pathlib.Path("alpha_options_raw")    # <SYM>.parquet
EARNINGS_CSV = "earnings.csv"                      # columns: symbol, event_day

OUT_PNL_EXCEL = r"C:\Users\antoi\Documents\Antoine\Projets_Python\Trading Vol on Earnings\outputs\Better_handle_Earnings_straddle_signal_S.xlsx"

BACKTEST_CONFIG = {
    "initial_equity_per_ticker": 100.0,
    "reinvest": True,
    "base_vega_target": 1.0 / 10,
    "multiplier": 100.0,

    # Strategy selection
    # "straddle" = ATM straddle (your original)
    # "strangle" = OTM strangle with ± strangle_mny_offset moneyness
    "strategy_type": "strangle",      # or "strangle", "straddle"
    "strangle_mny_offset": 0.03,      # 3% OTM for each leg when using strangle

    "use_signal": True,                     # activate signal-based sizing
    "signal_mode": "short",                # "short", "long", "long_short"
    "cost_model": {
        "option_spread_bps": 50,
        "stock_spread_bps": 1,
        "commission_per_contract": 0.0,
    },
    "delta_hedge": True,

    # Entry/exit lags in business days relative to event_day, per timing
    # BMO: enter previous day, exit event day  -> (-1, 0)
    # AMC: enter event day,   exit next day   -> (0, +1)
    "entry_lag": {
        "BMO": -1,
        "AMC": 0,
        "DURING": 0,
        "UNKNOWN": 0,
    },
    "exit_lag": {
        "BMO": 0,
        "AMC": 1,
        "DURING": 1,
        "UNKNOWN": 1,
    },

    "min_moneyness": 0.5,
    "max_moneyness": 1.5,

    "min_dte_for_entry": 5,
    "max_dte_for_entry": 30,
}


# ============================================================
# ===================== DATA STRUCTURES ======================
# ============================================================

def _flatten_config_dict(d: dict, prefix: str = "") -> List[Dict[str, Any]]:
    rows = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            rows.extend(_flatten_config_dict(v, prefix=key))
        else:
            rows.append({"Key": key, "Value": v})
    return rows


@dataclass
class OptionPosition:
    contract_id: str
    symbol: str
    expiry: pd.Timestamp
    strike: float
    opt_type: str  # "C" or "P"
    qty: float
    last_price: float
    iv: float
    delta: float
    gamma: float
    vega: float
    theta: float
    multiplier: float = 100.0

    # previous greeks for decomposition
    prev_iv: Optional[float] = None
    prev_delta: Optional[float] = None
    prev_gamma: Optional[float] = None
    prev_vega: Optional[float] = None
    prev_theta: Optional[float] = None
    prev_price: Optional[float] = None


@dataclass
class TickerState:
    equity: float
    cash: float

    positions: Dict[str, OptionPosition] = field(default_factory=dict)

    # Stock hedge - we distinguish "close" position for PnL semantics
    stock_pos_close: float = 0.0   # shares held at end-of-day (t-1) for PnL_t
    stock_pos_intraday: float = 0.0  # current position being traded today

    last_spot: Optional[float] = None

    mtm_options: float = 0.0
    mtm_stock: float = 0.0

    cum_pnl: float = 0.0
    cum_pnl_vega: float = 0.0
    cum_pnl_gamma: float = 0.0
    cum_pnl_theta: float = 0.0
    cum_pnl_delta_hedge: float = 0.0
    cum_pnl_tc: float = 0.0


# ============================================================
# ===================== MARKET DATA LAYER ====================
# ============================================================

class MarketData:
    """
    Loads and provides:
      - daily split-normalized spot
      - full options chain per (symbol, date)
      - earnings events + entry/exit mapping
    """

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.spot: Dict[str, pd.DataFrame] = {}
        self.options: Dict[str, pd.DataFrame] = {}
        self.earnings: pd.DataFrame = pd.DataFrame()

        self._load_spot()
        self._load_options()
        self._load_earnings()

    # ---------- Spot ----------
    @profile
    def _load_spot(self):
        for sym in self.symbols:
            path = CORP_DIR / f"{sym}_daily_adjusted.parquet"
            if not path.exists():
                print(f"[WARN] Spot file missing for {sym}: {path}")
                continue

            df = pd.read_parquet(path)
            if "date" not in df.columns:
                print(f"[WARN] No 'date' column in {path}, skipping {sym}")
                continue

            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()
            if df.empty:
                print(f"[WARN] No spot rows for {sym} in backtest window.")
                continue

            df = df.sort_values("date").set_index("date")

            if "close" in df.columns:
                close_raw = df["close"].astype(float)
            elif "adj_close" in df.columns:
                close_raw = df["adj_close"].astype(float)
            else:
                raise ValueError(f"{sym}: neither 'close' nor 'adj_close' in {path}")

            if "split_coefficient" in df.columns:
                split_raw = df["split_coefficient"].astype(float)
                split_raw = split_raw.replace(0.0, np.nan).fillna(1.0)
            else:
                split_raw = pd.Series(1.0, index=df.index)

            split_level = split_raw.cumprod()
            level_last = float(split_level.iloc[-1])
            if level_last <= 0:
                level_last = 1.0

            price_factor = level_last / split_level
            spot_norm = close_raw / price_factor

            out = pd.DataFrame(
                {
                    "spot": spot_norm.astype(float),
                    "price_factor": price_factor.astype(float),
                }
            )
            self.spot[sym] = out
            print(f"[INFO] Spot loaded for {sym}: {out.shape[0]} days")

    def get_spot(self, symbol: str, date: pd.Timestamp) -> Optional[float]:
        df = self.spot.get(symbol)
        if df is None:
            return None
        try:
            return float(df.loc[date, "spot"])
        except KeyError:
            return None

    def get_spot_calendar(self, symbol: str) -> pd.DatetimeIndex:
        df = self.spot.get(symbol)
        if df is None:
            return pd.DatetimeIndex([])
        return df.index

    # ---------- Options ----------
    @profile
    def _load_options(self):
        for sym in self.symbols:
            path = OPTIONS_DIR / f"{sym}.parquet"
            if not path.exists():
                print(f"[WARN] Options file missing for {sym}: {path}")
                continue

            df = pd.read_parquet(path)
            if "date" not in df.columns or "expiration" not in df.columns:
                print(f"[WARN] Missing date/expiration in options for {sym}, skipping.")
                continue

            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()
            df["strike"] = df["strike"].astype(float)
            df["type"] = df["type"].astype(str).str.upper().str[0]

            df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()
            if df.empty:
                print(f"[WARN] No option rows for {sym} in backtest window.")
                continue

            spot_df = self.spot.get(sym)
            if spot_df is not None and "price_factor" in spot_df.columns:
                pf = spot_df[["price_factor"]].reset_index()
                df = df.merge(pf, on="date", how="left")
            else:
                df["price_factor"] = 1.0

            df["price_factor"] = df["price_factor"].fillna(1.0).astype(float)

            # mid price from mark/bid-ask/last
            if "mark" in df.columns:
                df["mid"] = df["mark"].astype(float)
            else:
                bid = df.get("bid", np.nan).astype(float)
                ask = df.get("ask", np.nan).astype(float)
                df["mid"] = np.where(
                    np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > 0),
                    0.5 * (bid + ask),
                    df.get("last", np.nan).astype(float),
                )

            df = df[df["mid"] > 0]

            df["strike_eff"] = df["strike"] / df["price_factor"]

            self.options[sym] = df
            print(f"[INFO] Options loaded for {sym}: {df.shape[0]:,} rows")

    def get_chain(self, symbol: str, date: pd.Timestamp) -> pd.DataFrame:
        df = self.options.get(symbol)
        if df is None:
            return pd.DataFrame()
        return df[df["date"] == date].copy()

    # ---------- Earnings ----------
    def _load_earnings(self):
        if not os.path.exists(EARNINGS_CSV):
            print(f"[WARN] Earnings file {EARNINGS_CSV} missing.")
            self.earnings = pd.DataFrame(columns=["symbol", "event_day"])
            return

        df = pd.read_csv(EARNINGS_CSV)

        df["symbol"] = df["symbol"].astype(str).str.upper()
        df["event_day"] = pd.to_datetime(df["event_day"]).dt.normalize()

        # If timing exists in CSV (BMO/AMC/...), keep it; else default to UNKNOWN
        if "timing" in df.columns:
            df["timing"] = df["timing"].astype(str).str.upper()
        else:
            df["timing"] = "UNKNOWN"

        df = df[df["symbol"].isin(self.symbols)].copy()
        df = df[(df["event_day"] >= START_DATE) & (df["event_day"] <= END_DATE)].copy()
        df = df.drop_duplicates(subset=["symbol", "event_day"]).reset_index(drop=True)

        self.earnings = df
        print(f"[INFO] Earnings loaded: {len(df)} events")


# ============================================================
# ===================== STRATEGY LAYER =======================
# ============================================================

class StrategyEarningsATM:
    """
    Straddle / Strangle earnings strategy:

    - On entry date (T-1):
      - "straddle": short vega_target of ATM front-month straddle
      - "strangle": short vega_target of symmetric OTM strangle
                    with ± strangle_mny_offset around ATM
    - On exit date (T+1): flat
    """

    def __init__(self, config: dict):
        self.config = config
        self.entry_exit_map = {}
        self.signals_df = None
        self._short_rules = {}
        self._long_rules = {}
        self._ls_rules = {}
        if config.get("use_signal", False):
            self._load_signals_and_build_rules()

    def _load_signals_and_build_rules(self):
        cfg = self.config
        path = cfg.get("signal_csv_path", "outputs/signals_all.csv")
        if not os.path.exists(path):
            return
        d = pd.read_csv(path, parse_dates=["EventDate","AnchorDate"])
        d = d.dropna(subset=["ShortScore_z","LongScore_z","PnL_proxy"])
        d["year"] = d["AnchorDate"].dt.year
        self.signals_df = d

        min_years = cfg.get("signal_min_years", 2)
        n_bins    = cfg.get("signal_n_bins", 10)
        max_mult  = cfg.get("signal_max_vega_mult", 2.0)

        self._short_rules = {}
        self._long_rules  = {}
        self._ls_rules    = {}

        def quant_edges(s):
            qs = np.linspace(0,1,n_bins+1)
            e = np.quantile(s.dropna(), qs)
            e = np.unique(e)
            return e if len(e) > 1 else None

        def train_rule(train_df, score_col, pnl_col, flip=False):
            t = train_df[[score_col, pnl_col]].dropna()
            if t.empty: return None
            s = t[score_col]
            edges = quant_edges(s)
            if edges is None: return None
            bins = np.searchsorted(edges, s.values, side="right") - 1
            pnl = t[pnl_col].values
            if flip: pnl = -pnl
            g = pd.Series(pnl).groupby(bins).mean()
            g = g[g.index >= 0]
            good = g[g > 0.0]
            if good.empty: return None
            mm = good.clip(lower=0.0)
            vmax = mm.max()
            mult = (mm / vmax * max_mult) if vmax > 0 else mm*0.0
            vega_mult = {int(b): float(v) for b,v in mult.items()}
            return {"edges":edges, "good_bins":set(int(b) for b in good.index), "vega_mult":vega_mult}

        years = sorted(d["year"].unique())
        for y in years:
            y_start    = dt.datetime(y,1,1)
            train_start = y_start - dt.timedelta(days=365*min_years)
            train = d[(d["AnchorDate"]>=train_start) & (d["AnchorDate"]<y_start)]
            if train["year"].nunique() < min_years or len(train) < 50:
                self._short_rules[y] = None
                self._long_rules[y]  = None
                self._ls_rules[y]    = None
                continue
            short_rule = train_rule(train, "ShortScore_z", "PnL_proxy", flip=False)
            long_rule  = train_rule(train, "LongScore_z",  "PnL_proxy", flip=True)
            self._short_rules[y] = short_rule
            self._long_rules[y]  = long_rule
            ls_rule = None
            if short_rule and long_rule:
                ls_rule = {
                    "short_edges":short_rule["edges"],
                    "long_edges": long_rule["edges"],
                    "short_bins": short_rule["good_bins"],
                    "long_bins":  long_rule["good_bins"],
                    "short_mult": short_rule["vega_mult"],
                    "long_mult":  long_rule["vega_mult"],
                }
            self._ls_rules[y] = ls_rule

    def _assign_bin(self, x, edges):
        if edges is None or pd.isna(x): return None
        return int(np.searchsorted(edges, x, side="right") - 1)

    def _signal_decision_for_row(self, row: pd.Series, mode: str):
        y = int(row["AnchorDate"].year)
        if mode == "short":
            rule = self._short_rules.get(y)
            if not rule: return "flat",0.0
            s = float(row["ShortScore_z"])
            b = self._assign_bin(s, rule["edges"])
            if b is None or b not in rule["good_bins"]: return "flat",0.0
            v = rule["vega_mult"].get(b,0.0)
            return ("short", v if v>0 else 0.0)
        if mode == "long":
            rule = self._long_rules.get(y)
            if not rule: return "flat",0.0
            s = float(row["LongScore_z"])
            b = self._assign_bin(s, rule["edges"])
            if b is None or b not in rule["good_bins"]: return "flat",0.0
            v = rule["vega_mult"].get(b,0.0)
            return ("long", v if v>0 else 0.0)
        if mode == "long_short":
            rule = self._ls_rules.get(y)
            if not rule: return "flat",0.0
            s_s = float(row["ShortScore_z"]); s_l = float(row["LongScore_z"])
            b_s = self._assign_bin(s_s, rule["short_edges"])
            b_l = self._assign_bin(s_l, rule["long_edges"])
            cand = []
            if b_s is not None and b_s in rule["short_bins"]:
                v = rule["short_mult"].get(b_s,0.0)
                if v>0: cand.append(("short",v))
            if b_l is not None and b_l in rule["long_bins"]:
                v = rule["long_mult"].get(b_l,0.0)
                if v>0: cand.append(("long",v))
            if not cand: return "flat",0.0
            cand.sort(key=lambda x: x[1], reverse=True)
            if len(cand)>=2 and cand[0][1]==cand[1][1]:
                for side,v in cand:
                    if side=="short": return "short",v
            return cand[0]
        return "flat",0.0

    def compute_signal_vega(self, date: pd.Timestamp, symbol: str, base_vega: float) -> float:
        if (not self.config.get("use_signal", False)) or self.signals_df is None:
            return base_vega
        # only on entry dates
        mapping_sym = self.entry_exit_map.get(symbol, {})
        meta = mapping_sym.get(date)
        if not meta:
            return 0.0
        event_day = pd.to_datetime(meta["event_day"]).normalize()
        d = self.signals_df
        rows = d[(d["Symbol"]==symbol) & (d["EventDate"]==event_day)]
        if rows.empty:
            return 0.0
        row = rows.iloc[0]
        mode = self.config.get("signal_mode", "short")
        side, mult = self._signal_decision_for_row(row, mode)
        if side == "flat" or mult <= 0:
            return 0.0
        v = base_vega * mult
        return v if side == "short" else -v

    def build_entry_exit_map(self, market: MarketData) -> Dict[str, Dict[pd.Timestamp, Dict[str, Any]]]:
        """
        Build per-symbol entry/exit windows for earnings trades.

        Uses:
          - market.earnings: columns [symbol, event_day, timing, ...]
          - market.get_spot_calendar(symbol)
          - self.config['entry_lag'] / ['exit_lag']: dicts by timing

        Returns:
          { symbol -> { entry_date -> {'event_day', 'exit_date', 'timing'} } }
        """

        # Default lags if config doesn't provide full mapping
        default_entry_lag = {"BMO": -1, "AMC": 0, "DURING": 0, "UNKNOWN": 0}
        default_exit_lag  = {"BMO": 0,  "AMC": 1, "DURING": 1, "UNKNOWN": 1}

        cfg_entry = self.config.get("entry_lag", {})
        cfg_exit  = self.config.get("exit_lag", {})

        # Merge defaults with config (config overrides)
        entry_lag_cfg = {**default_entry_lag, **cfg_entry}
        exit_lag_cfg  = {**default_exit_lag,  **cfg_exit}

        mapping_global: Dict[str, Dict[pd.Timestamp, Dict[str, Any]]] = {}

        for sym in market.symbols:
            cal = market.get_spot_calendar(sym)
            if len(cal) == 0:
                continue

            evs = market.earnings[market.earnings["symbol"] == sym]
            if evs.empty:
                continue

            mapping_sym: Dict[pd.Timestamp, Dict[str, Any]] = {}

            for _, row in evs.iterrows():
                ev_day = row["event_day"]
                timing = row.get("timing", "UNKNOWN")

                # anchor index: first trading day >= event_day
                anchor_idx = cal.searchsorted(ev_day)
                if anchor_idx >= len(cal):
                    # event after last available trading day
                    continue

                entry_off = entry_lag_cfg.get(timing, entry_lag_cfg["UNKNOWN"])
                exit_off  = exit_lag_cfg.get(timing,  exit_lag_cfg["UNKNOWN"])

                entry_idx = anchor_idx + entry_off
                exit_idx  = anchor_idx + exit_off

                # skip if entry date out of calendar
                if entry_idx < 0 or entry_idx >= len(cal):
                    continue

                entry_date = cal[entry_idx]
                exit_date = cal[exit_idx] if 0 <= exit_idx < len(cal) else None

                mapping_sym[entry_date] = {
                    "event_day": ev_day,
                    "exit_date": exit_date,
                    "timing": timing,
                }

            mapping_global[sym] = mapping_sym

        self.entry_exit_map = mapping_global
        return mapping_global

    def compute_target_positions(
        self,
        date: pd.Timestamp,
        symbol: str,
        state: TickerState,
        market: MarketData,
        vega_target: float,
    ) -> List[Dict[str, Any]]:

        cfg = self.config
        strategy_type = cfg.get("strategy_type", "straddle")
        targets: List[Dict[str, Any]] = []

        meta = self.entry_exit_map.get(symbol, {}).get(date)
        if meta is None:
            return targets

        event_day = meta["event_day"]

        spot = market.get_spot(symbol, date)
        if spot is None or not np.isfinite(spot):
            return targets

        chain = market.get_chain(symbol, date)
        if chain.empty:
            return targets

        min_mny = cfg["min_moneyness"]
        max_mny = cfg["max_moneyness"]
        min_dte = cfg["min_dte_for_entry"]
        max_dte = cfg["max_dte_for_entry"]
        strangle_off = cfg.get("strangle_mny_offset", 0.03)

        chain = chain.copy()
        chain["dte"] = (chain["expiration"] - date).dt.days

        strike_col = "strike_eff" if "strike_eff" in chain.columns else "strike"
        chain["moneyness"] = chain[strike_col] / spot

        chain = chain[(chain["moneyness"] >= min_mny) &
                      (chain["moneyness"] <= max_mny)].copy()
        if chain.empty:
            return targets

        after_ev = chain[chain["expiration"] > event_day].copy()
        if after_ev.empty:
            return targets

        eligible = after_ev[(after_ev["dte"] >= min_dte) &
                            (after_ev["dte"] <= max_dte)].copy()
        if eligible.empty:
            eligible = after_ev.sort_values("dte").copy()

        eligible["abs_mny"] = (eligible["moneyness"] - 1.0).abs()
        eligible = eligible.sort_values(["dte", "abs_mny"])

        if strategy_type == "straddle":
            targets = self._build_straddle_targets(eligible, chain, symbol, vega_target)
        elif strategy_type == "strangle":
            targets = self._build_strangle_targets(
                eligible, chain, symbol, vega_target, strangle_off
            )
        else:
            raise ValueError(f"Unknown strategy_type: {strategy_type}")

        return targets

    def _build_straddle_targets(
        self,
        eligible: pd.DataFrame,
        chain: pd.DataFrame,
        symbol: str,
        vega_target: float,
    ) -> List[Dict[str, Any]]:

        for _, opt_row in eligible.iterrows():
            expiry = opt_row["expiration"]
            strike_raw = opt_row["strike"]
            sub = chain[(chain["expiration"] == expiry) &
                        (chain["strike"] == strike_raw)]
            calls = sub[sub["type"] == "C"]
            puts  = sub[sub["type"] == "P"]
            if calls.empty or puts.empty:
                continue

            call = calls.iloc[0]
            put  = puts.iloc[0]
            return self._size_vega_straddle(symbol, call, put, vega_target)

        return []

    def _build_strangle_targets(
        self,
        eligible: pd.DataFrame,
        chain: pd.DataFrame,
        symbol: str,
        vega_target: float,
        offset: float,
    ) -> List[Dict[str, Any]]:

        # pick closest expiry (first row in eligible)
        first = eligible.iloc[0]
        expiry = first["expiration"]

        sub = chain[chain["expiration"] == expiry].copy()
        if sub.empty:
            return []

        # we have moneyness column in chain
        target_call_mny = 1.0 + offset
        target_put_mny  = 1.0 - offset

        sub["mny_diff_call"] = (sub["moneyness"] - target_call_mny).abs()
        sub["mny_diff_put"]  = (sub["moneyness"] - target_put_mny).abs()

        call_candidates = sub[sub["type"] == "C"].sort_values("mny_diff_call")
        put_candidates  = sub[sub["type"] == "P"].sort_values("mny_diff_put")

        if call_candidates.empty or put_candidates.empty:
            return []

        call = call_candidates.iloc[0]
        put  = put_candidates.iloc[0]
        return self._size_vega_straddle(symbol, call, put, vega_target)

    def _size_vega_straddle(
        self,
        symbol: str,
        call: pd.Series,
        put: pd.Series,
        vega_target: float,
    ) -> List[Dict[str, Any]]:

        cfg = self.config
        multiplier = cfg["multiplier"]

        call_vega = float(call["vega"])
        put_vega  = float(put["vega"])
        tot_vega  = abs(call_vega) + abs(put_vega)
        if tot_vega <= 1e-8:
            return []

        vega_per_call = call_vega * multiplier
        vega_per_put  = put_vega * multiplier
        if abs(vega_per_call) <= 1e-8 or abs(vega_per_put) <= 1e-8:
            return []

        target_vega_each = -0.5 * vega_target
        qty_call = target_vega_each / vega_per_call
        qty_put  = target_vega_each / vega_per_put

        targets = [
            {
                "contract_id": str(call["contractID"]),
                "symbol": symbol,
                "expiry": call["expiration"],
                "strike": float(call["strike"]),
                "type": "C",
                "qty": qty_call,
            },
            {
                "contract_id": str(put["contractID"]),
                "symbol": symbol,
                "expiry": put["expiration"],
                "strike": float(put["strike"]),
                "type": "P",
                "qty": qty_put,
            },
        ]
        return targets


# ============================================================
# ====================== BACKTEST ENGINE =====================
# ============================================================

class Backtester:
    def __init__(self, config: dict, symbols: List[str]):
        self.config = config
        self.symbols = symbols
        self.market = MarketData(symbols)
        self.strategy = StrategyEarningsATM(config)
        self.strategy.build_entry_exit_map(self.market)


        self.daily_pnl_rows: List[Dict[str, Any]] = []
        self.trade_rows: List[Dict[str, Any]] = []

        self.state: Dict[str, TickerState] = {
            sym: TickerState(
                equity=config["initial_equity_per_ticker"],
                cash=config["initial_equity_per_ticker"],
            )
            for sym in symbols
        }

    def run(self):
        all_dates = self._build_global_calendar()
        print(f"[INFO] Backtest calendar: {len(all_dates)} days.")

        for sym in self.symbols:
            for date in tqdm(all_dates, desc=f"Processing {sym}", unit="day"):
                if date not in self.market.get_spot_calendar(sym):
                    continue
                self._process_symbol_date(sym, date)

        self._export_results()

    def _build_global_calendar(self) -> pd.DatetimeIndex:
        calendars = [self.market.get_spot_calendar(sym) for sym in self.symbols]
        if not calendars:
            return pd.DatetimeIndex([])
        all_dates = sorted(set().union(*[set(idx) for idx in calendars]))
        return pd.DatetimeIndex(all_dates)

    @profile
    def _process_symbol_date(self, symbol: str, date: pd.Timestamp):
        st = self.state[symbol]
        cfg = self.config
        equity_prev = st.equity

        spot = self.market.get_spot(symbol, date)
        if spot is None or not np.isfinite(spot):
            return

        # vega target scaling
        if cfg["reinvest"]:
            equity_scale = st.equity / cfg["initial_equity_per_ticker"]
        else:
            equity_scale = 1.0
        base_vega = cfg["base_vega_target"] * equity_scale

        if cfg.get("use_signal", False):
            vega_target = self.strategy.compute_signal_vega(date, symbol, base_vega)
        else:
            vega_target = base_vega

        target_lines = self.strategy.compute_target_positions(
            date, symbol, st, self.market, vega_target
        )

        # Exit logic: flat on any event exit date
        mapping_sym = self.strategy.entry_exit_map.get(symbol, {})
        need_flat = any(
            meta_ev.get("exit_date") == date
            for meta_ev in mapping_sym.values()
        )
        if need_flat:
            target_lines = []

        chain_today = self.market.get_chain(symbol, date)
        if chain_today.empty and st.positions:
            return

        if not chain_today.empty:
            chain_today = chain_today.copy()
            chain_today["cid"] = chain_today["contractID"].astype(str)
            chain_today_idx = chain_today.set_index("cid")
        else:
            chain_today_idx = None

        # carry prev greeks
        for pos in st.positions.values():
            pos.prev_iv = pos.iv
            pos.prev_delta = pos.delta
            pos.prev_gamma = pos.gamma
            pos.prev_vega = pos.vega
            pos.prev_theta = pos.theta
            pos.prev_price = pos.last_price

        # target qty map
        target_qty: Dict[str, float] = {}
        for t in target_lines:
            cid = t["contract_id"]
            target_qty[cid] = target_qty.get(cid, 0.0) + float(t["qty"])

        if need_flat:
            for cid in list(st.positions.keys()):
                if cid not in target_qty:
                    target_qty[cid] = 0.0

        cost_model = cfg["cost_model"]
        option_spread_bps = cost_model["option_spread_bps"]
        commission_per_contract = cost_model["commission_per_contract"]

        pnl_tc_options = 0.0

        contract_ids = set(st.positions.keys()) | set(target_qty.keys())
        for cid in contract_ids:
            cur_pos = st.positions.get(cid)
            tgt_qty = target_qty.get(cid, cur_pos.qty if cur_pos else 0.0)

            if chain_today_idx is None:
                continue

            try:
                row = chain_today_idx.loc[cid]
            except KeyError:
                continue

            mid_price = float(row["mid"])
            iv = float(row.get("implied_volatility", np.nan))
            delta = float(row.get("delta", 0.0))
            gamma = float(row.get("gamma", 0.0))
            vega = float(row.get("vega", 0.0))
            theta = float(row.get("theta", 0.0))
            opt_type = str(row["type"]).upper()[0]
            expiry = pd.to_datetime(row["expiration"]).normalize()
            strike = float(row["strike"])

            cur_qty = cur_pos.qty if cur_pos is not None else 0.0
            trade_qty = tgt_qty - cur_qty

            if abs(trade_qty) > 1e-10:
                spread = option_spread_bps / 1e4
                if trade_qty > 0:
                    trade_price = mid_price * (1 + spread)
                else:
                    trade_price = mid_price * (1 - spread)

                cash_change = -trade_qty * trade_price * cfg["multiplier"]
                commission = abs(trade_qty) * commission_per_contract

                st.cash += cash_change - commission
                pnl_tc_options += commission

                self.trade_rows.append({
                    "Date": date,
                    "Symbol": symbol,
                    "ContractID": cid,
                    "Expiry": expiry,
                    "Strike": strike,
                    "Type": opt_type,
                    "TradeQty": trade_qty,
                    "TradePrice": trade_price,
                    "TradeNotional": trade_qty * trade_price * cfg["multiplier"],
                    "Spot": spot,
                    "IV": iv,
                    "Delta": delta,
                    "Gamma": gamma,
                    "Vega": vega,
                    "Theta": theta,
                })

            if abs(tgt_qty) <= 1e-10:
                if cid in st.positions:
                    del st.positions[cid]
            else:
                if cur_pos is None:
                    st.positions[cid] = OptionPosition(
                        contract_id=cid,
                        symbol=symbol,
                        expiry=expiry,
                        strike=strike,
                        opt_type=opt_type,
                        qty=tgt_qty,
                        last_price=mid_price,
                        iv=iv,
                        delta=delta,
                        gamma=gamma,
                        vega=vega,
                        theta=theta,
                        multiplier=cfg["multiplier"],
                    )
                else:
                    cur_pos.qty = tgt_qty
                    cur_pos.last_price = mid_price
                    cur_pos.iv = iv
                    cur_pos.delta = delta
                    cur_pos.gamma = gamma
                    cur_pos.vega = vega
                    cur_pos.theta = theta

        # --- Delta hedge semantics ---
        pnl_delta_hedge = 0.0

        if cfg["delta_hedge"]:
            # target delta hedge using today's updated option deltas
            port_delta = sum(
                pos.delta * pos.qty * pos.multiplier
                for pos in st.positions.values()
            )
            tgt_stock_pos = -port_delta
            trade_shares = tgt_stock_pos - st.stock_pos_intraday

            if abs(trade_shares) > 1e-8:
                spread = cost_model["stock_spread_bps"] / 1e4
                if trade_shares > 0:
                    trade_price = spot * (1 + spread)
                else:
                    trade_price = spot * (1 - spread)

                cash_change = -trade_shares * trade_price
                st.cash += cash_change

            # PnL from holding stock between yesterday's close and today:
            if st.last_spot is not None:
                dS = spot - st.last_spot
                pnl_delta_hedge = st.stock_pos_close * dS

            # at EoD, we define "close" stock position as tgt_stock_pos
            st.stock_pos_intraday = tgt_stock_pos
            st.stock_pos_close = tgt_stock_pos
        else:
            # No active hedging, but if you have a static stock position, you still get PnL
            if st.last_spot is not None:
                dS = spot - st.last_spot
                pnl_delta_hedge = st.stock_pos_close * dS

        # --- PnL decomposition ---
        option_pnl = 0.0
        port_gamma_prev = 0.0
        port_vega_prev = 0.0
        port_theta_prev = 0.0
        dIV_weighted_num = 0.0
        dIV_weighted_den = 0.0

        port_delta_today = 0.0
        port_gamma_today = 0.0
        port_vega_today = 0.0
        port_theta_today = 0.0

        for pos in st.positions.values():
            if pos.prev_price is not None:
                dP = (pos.last_price - pos.prev_price) * pos.qty * pos.multiplier
                option_pnl += dP

            if pos.prev_gamma is not None:
                port_gamma_prev += pos.prev_gamma * pos.qty * pos.multiplier
            if pos.prev_vega is not None:
                port_vega_prev += pos.prev_vega * pos.qty * pos.multiplier
            if pos.prev_theta is not None:
                port_theta_prev += pos.prev_theta * pos.qty * pos.multiplier

            if (pos.prev_iv is not None and
                np.isfinite(pos.prev_iv) and np.isfinite(pos.iv)):
                dIV = pos.iv - pos.prev_iv
                w = abs(pos.prev_vega * pos.qty * pos.multiplier)
                dIV_weighted_num += w * dIV
                dIV_weighted_den += w

            port_delta_today += pos.delta * pos.qty * pos.multiplier
            port_gamma_today += pos.gamma * pos.qty * pos.multiplier
            port_vega_today += pos.vega * pos.qty * pos.multiplier
            port_theta_today += pos.theta * pos.qty * pos.multiplier

        dS = 0.0
        if st.last_spot is not None:
            dS = spot - st.last_spot

        dt = 1.0 / 252.0

        if dIV_weighted_den > 0:
            dIV_eff = dIV_weighted_num / dIV_weighted_den
        else:
            dIV_eff = 0.0

        pnl_gamma = 0.5 * port_gamma_prev * (dS ** 2)
        pnl_vega = port_vega_prev * dIV_eff
        pnl_theta = port_theta_prev * dt

        # --- MTM & equity ---
        mtm_options = sum(
            pos.last_price * pos.qty * pos.multiplier
            for pos in st.positions.values()
        )
        mtm_stock = st.stock_pos_close * spot

        st.mtm_options = mtm_options
        st.mtm_stock = mtm_stock

        st.equity = st.cash + mtm_options + mtm_stock
        pnl_total = st.equity - equity_prev

        st.cum_pnl += pnl_total
        st.cum_pnl_vega += pnl_vega
        st.cum_pnl_gamma += pnl_gamma
        st.cum_pnl_theta += pnl_theta
        st.cum_pnl_delta_hedge += pnl_delta_hedge
        st.cum_pnl_tc += pnl_tc_options

        row = {
            "Date": date,
            "Symbol": symbol,
            "Equity": st.equity,
            "Cash": st.cash,
            "MTM_Options": st.mtm_options,
            "MTM_Stock": st.mtm_stock,
            "Spot": spot,

            "DailyPnL": pnl_total,
            "OptionPnL": option_pnl,
            "PnL_gamma": pnl_gamma,
            "PnL_vega": pnl_vega,
            "PnL_theta": pnl_theta,
            "PnL_deltaHedge": pnl_delta_hedge,
            "PnL_TC": pnl_tc_options,

            "CumPnL": st.cum_pnl,
            "CumPnL_gamma": st.cum_pnl_gamma,
            "CumPnL_vega": st.cum_pnl_vega,
            "CumPnL_theta": st.cum_pnl_theta,
            "CumPnL_deltaHedge": st.cum_pnl_delta_hedge,
            "CumPnL_TC": st.cum_pnl_tc,

            "Delta": port_delta_today,
            "Gamma": port_gamma_today,
            "Vega": port_vega_today,
            "Theta": port_theta_today,
        }
        self.daily_pnl_rows.append(row)

        st.last_spot = spot

    # ---------- Export ----------
    def _export_results(self):
        if not self.daily_pnl_rows:
            print("[WARN] No PnL rows to export.")
            return

        df = pd.DataFrame(self.daily_pnl_rows)
        df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)
        df = df.bfill()

        pivot_equity = df.pivot(index="Date", columns="Symbol", values="Equity")
        pivot_equity["PortfolioEquity"] = pivot_equity.sum(axis=1)

        config_rows = _flatten_config_dict(self.config)
        df_config = pd.DataFrame(config_rows, columns=["Key", "Value"])

        df_earn = self.market.earnings.copy()
        if df_earn.empty:
            df_earn = pd.DataFrame(columns=["Symbol", "EventDay"])
        else:
            df_earn = df_earn.sort_values(["symbol", "event_day"]).copy()
            df_earn = df_earn.rename(
                columns={"symbol": "Symbol", "event_day": "EventDay"}
            )

        # --- Corporate actions table (reworked) ---
        def _build_corp_actions_df() -> pd.DataFrame:
            corp_rows = []

            for sym in self.symbols:
                path = CORP_DIR / f"{sym}_daily_adjusted.parquet"
                if not path.exists():
                    print(f"[CORP] Missing daily_adjusted file for {sym}: {path}")
                    continue

                df_c = pd.read_parquet(path)

                if "date" not in df_c.columns:
                    print(f"[CORP] No 'date' column in {path}, skipping {sym}")
                    continue

                df_c["date"] = pd.to_datetime(df_c["date"]).dt.normalize()
                df_c = df_c[(df_c["date"] >= START_DATE) &
                            (df_c["date"] <= END_DATE)].copy()
                if df_c.empty:
                    print(f"[CORP] No rows for {sym} in backtest window.")
                    continue

                # Try to detect dividend / split columns in a flexible way
                lower_cols = {c.lower(): c for c in df_c.columns}

                div_col = None
                split_col = None

                for lc, orig in lower_cols.items():
                    if "dividend" in lc:
                        div_col = orig
                    if "split" in lc:
                        split_col = orig

                if div_col is None and split_col is None:
                    print(f"[CORP] No dividend/split columns for {sym}. "
                          f"Columns: {list(df_c.columns)}")
                    continue

                # Build mask of rows where something actually happened
                mask = False
                if div_col is not None:
                    mask = (df_c[div_col].fillna(0).astype(float) != 0)
                if split_col is not None:
                    # split factor != 1 means actual split
                    split_mask = (df_c[split_col].fillna(1).astype(float) != 1)
                    mask = mask | split_mask if isinstance(mask, pd.Series) else split_mask

                df_c = df_c[mask]
                if df_c.empty:
                    print(f"[CORP] No non-zero dividends/splits for {sym} in window.")
                    continue

                cols = ["date"]
                if div_col is not None:
                    cols.append(div_col)
                if split_col is not None:
                    cols.append(split_col)

                df_c = df_c[cols].copy()
                df_c["Symbol"] = sym

                # Normalize column names
                rename_map = {"date": "Date"}
                if div_col is not None:
                    rename_map[div_col] = "Dividend"
                if split_col is not None:
                    rename_map[split_col] = "SplitFactor"

                df_c = df_c.rename(columns=rename_map)
                corp_rows.append(df_c)

            if not corp_rows:
                print("[CORP] No corporate actions found for any symbol.")
                return pd.DataFrame(columns=["Symbol", "Date", "Dividend", "SplitFactor"])

            df_corp = pd.concat(corp_rows, ignore_index=True)
            # Ensure standard column order
            for col in ["Dividend", "SplitFactor"]:
                if col not in df_corp.columns:
                    df_corp[col] = np.nan

            df_corp = df_corp[["Symbol", "Date", "Dividend", "SplitFactor"]]
            df_corp = df_corp.sort_values(["Symbol", "Date"])
            return df_corp

        df_corp = _build_corp_actions_df()

        df_trades_all = (
            pd.DataFrame(self.trade_rows)
            if self.trade_rows
            else pd.DataFrame(
                columns=[
                    "Date", "Symbol", "ContractID", "Expiry", "Strike",
                    "Type", "TradeQty", "TradePrice", "TradeNotional", "Spot",
                    "IV", "Delta", "Gamma", "Vega", "Theta",
                ]
            )
        )

        # --- Event-level PnL sheet (entry->exit window) ---
        event_pnl_rows = []
        for sym, mapping in self.strategy.entry_exit_map.items():
            for entry_date, meta in mapping.items():
                exit_date = meta["exit_date"]
                event_day = meta["event_day"]
                timing = meta.get("timing", "UNKNOWN")

                if exit_date is None:
                    continue

                mask = (
                        (df["Symbol"] == sym) &
                        (df["Date"] >= entry_date) &
                        (df["Date"] <= exit_date)
                )
                df_window = df[mask]
                if df_window.empty:
                    continue

                event_pnl_rows.append({
                    "Symbol": sym,
                    "EntryDate": entry_date,
                    "EventDay": event_day,
                    "ExitDate": exit_date,
                    "Timing": timing,
                    "EventWindowPnL": df_window["DailyPnL"].sum(),
                    "EventWindowPnL_vega": df_window["PnL_vega"].sum(),
                    "EventWindowPnL_gamma": df_window["PnL_gamma"].sum(),
                    "EventWindowPnL_theta": df_window["PnL_theta"].sum(),
                    "EventWindowPnL_deltaHedge": df_window["PnL_deltaHedge"].sum(),
                    "EventWindowPnL_TC": df_window["PnL_TC"].sum(),
                })

        df_event_pnl = (
            pd.DataFrame(event_pnl_rows)
            if event_pnl_rows
            else pd.DataFrame(
                columns=[
                    "Symbol", "EntryDate", "EventDay", "ExitDate", "Timing",
                    "EventWindowPnL", "EventWindowPnL_vega", "EventWindowPnL_gamma",
                    "EventWindowPnL_theta", "EventWindowPnL_deltaHedge", "EventWindowPnL_TC",
                ]
            )
        )

        # --- Per-symbol performance stats (EVENT_STATS) ---
        if df_event_pnl.empty:
            df_event_stats = pd.DataFrame(
                columns=[
                    "Symbol",
                    "N_events",
                    "Mean_EventPnL",
                    "Std_EventPnL",
                    "HitRatio",
                    "Best_EventPnL",
                    "Worst_EventPnL",
                    "BMO_N",
                    "BMO_Mean_EventPnL",
                    "AMC_N",
                    "AMC_Mean_EventPnL",
                ]
            )
        else:
            # base aggregates across all events
            base = (
                df_event_pnl
                .groupby("Symbol")["EventWindowPnL"]
                .agg(
                    N_events="count",
                    Mean_EventPnL="mean",
                    Std_EventPnL="std",
                    Best_EventPnL="max",
                    Worst_EventPnL="min",
                )
            )

            # hit ratio (fraction of events with positive pnl)
            hit = (
                df_event_pnl.assign(Positive=df_event_pnl["EventWindowPnL"] > 0)
                .groupby("Symbol")["Positive"]
                .mean()
                .rename("HitRatio")
            )

            # BMO stats
            bmo = (
                df_event_pnl[df_event_pnl["Timing"] == "BMO"]
                .groupby("Symbol")["EventWindowPnL"]
                .agg(
                    BMO_N="count",
                    BMO_Mean_EventPnL="mean",
                )
            )

            # AMC stats
            amc = (
                df_event_pnl[df_event_pnl["Timing"] == "AMC"]
                .groupby("Symbol")["EventWindowPnL"]
                .agg(
                    AMC_N="count",
                    AMC_Mean_EventPnL="mean",
                )
            )

            df_event_stats = (
                base
                .join(hit, how="left")
                .join(bmo, how="left")
                .join(amc, how="left")
                .reset_index()  # Symbol back as a column
            )

            # for symbols without BMO/AMC, counts = 0
            for col in ["BMO_N", "AMC_N"]:
                if col in df_event_stats.columns:
                    df_event_stats[col] = df_event_stats[col].fillna(0).astype(int)

        with pd.ExcelWriter(OUT_PNL_EXCEL, engine="xlsxwriter") as writer:
            # 1) PORTFOLIO
            out_portfolio = pivot_equity.reset_index()
            out_portfolio.to_excel(writer, sheet_name="PORTFOLIO", index=False)

            # 2) EVENT_PNL (right after portfolio)
            df_event_pnl.to_excel(writer, sheet_name="EVENT_PNL", index=False)

            # 2b) EVENT_STATS (per-name summary)
            df_event_stats.to_excel(writer, sheet_name="EVENT_STATS", index=False)

            # 3) CONFIG
            df_config.to_excel(writer, sheet_name="CONFIG", index=False)

            # 4) EARNINGS
            df_earn.to_excel(writer, sheet_name="EARNINGS", index=False)

            # 5) CORP_ACTIONS
            df_corp.to_excel(writer, sheet_name="CORP_ACTIONS", index=False)

            # 6) ALL_TRADES
            if not df_trades_all.empty:
                df_trades_all.sort_values(["Date", "Symbol"]).to_excel(
                    writer, sheet_name="ALL_TRADES", index=False
                )

            # 7) Per-symbol sheets
            for sym in self.symbols:
                df_sym = df[df["Symbol"] == sym].copy()
                if not df_sym.empty:
                    sheet = sym[:31]
                    df_sym.to_excel(writer, sheet_name=sheet, index=False)

                if not df_trades_all.empty:
                    df_tr_sym = df_trades_all[df_trades_all["Symbol"] == sym].copy()
                    if not df_tr_sym.empty:
                        sheet_tr = f"{sym}_TRADES"[:31]
                        df_tr_sym.to_excel(writer, sheet_name=sheet_tr, index=False)

        print(f"[INFO] Backtest results written to {OUT_PNL_EXCEL}")


# ============================================================
# ============================= MAIN =========================
# ============================================================

if __name__ == "__main__":
    bt = Backtester(BACKTEST_CONFIG, SYMBOLS)
    # If you want line_profiler:
    # if LineProfiler is not None:
    #     lp = LineProfiler()
    #     lp.add_function(Backtester._process_symbol_date)
    #     lp.add_function(StrategyEarningsATM.compute_target_positions)
    #     lp_wrapper = lp(bt.run)
    #     lp_wrapper()
    #     lp.print_stats()
    # else:
    bt.run()
