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

# SYMBOLS = [
#     "NVDA","MSFT","AAPL","AMZN","META","AVGO","GOOGL","GOOG","BRK.B","TSLA","JPM","V","LLY","NFLX","XOM","MA","WMT",
#     "COST","ORCL","JNJ","HD","PG","ABBV","BAC","UNH","CRM","ADBE","PYPL","AMD","INTC","CSCO","MCD","NKE","WFC","CVX",
#     "PEP","KO","DIS","BA","MRK","MO","IBM","T","GM","CAT","UPS","DOW","PLTR","TXN","LIN","AMAT"
# ]

SYMBOLS = [
"NVDA","MSFT"
]

START_DATE = pd.Timestamp("2020-01-01")
END_DATE   = pd.Timestamp("2025-11-16")

# Paths
CORP_DIR     = pathlib.Path("alpha_corp_actions")   # *_daily_adjusted.parquet
OPTIONS_DIR  = pathlib.Path("alpha_options_raw")    # <SYM>.parquet
EARNINGS_CSV = "earnings.csv"                      # columns: symbol, event_day

name_strat = r"\spy_ndq.xlsx"
OUT_PNL_EXCEL = rf"C:\Users\antoi\Documents\Antoine\Projets_Python\Trading Vol on Earnings\outputs{name_strat}"

BACKTEST_CONFIG = {
    "initial_equity_per_ticker": 100.0,
    "reinvest": True,
    "base_vega_target": 1.0 / 20 / 10,

    # ==========================
    # Strategy routing
    # ==========================
    # "earnings"  -> classic straddle/strangle around earnings
    # "rolling"   -> daily rolling single-leg option strategy
    "strategy_mode": "rolling",

    # Earnings sub-type
    #   "straddle": ATM C+P
    #   "strangle": OTM C+P at +/- moneyness offset
    "earnings_structure": "straddle",   # or "strangle"
    "strangle_mny_offset": 0.03,

    # Rolling daily option sub-type
    # Here we implement your “buy a 20δ put on 3rd maturity”
    "rolling_leg_type": "P",  # "P" or "C"
    "rolling_select_by": "delta",  # "delta" or "moneyness"
    "rolling_target_delta": -0.20,  # used if select_by == "delta"
    "rolling_target_mny": 1.0,  # used if select_by == "moneyness"
    "rolling_maturity_index": 2,  # 0=nearest, 1=2nd, 2=3rd...
    "rolling_min_dte": 1,  # optional extra filters
    "rolling_max_dte": 365,
    "rolling_holding_days": 20,  # hold each rolling lot for 20 business days
    "rolling_vega_per_lot": None,  # if None, we auto = base_vega_target / holding_days
    "rolling_reinvest": False,  # safer default for rolling
    "rolling_direction": -1,

    # Signal stuff (only meaningful for earnings)
    "use_signal": False,
    "signal_mode": "long",    # "short", "long", "ls"
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
class LotSpec:
    """
    Strategy-level description of ONE new lot to open today.
    The backtester will turn this into a concrete lot_id + positions.
    """
    symbol: str
    contract_id: str        # unique key for option
    expiry: pd.Timestamp
    strike: float
    opt_type: str           # "C" or "P"
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    qty: float              # signed
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LotInfo:
    lot_id: int
    symbol: str
    contract_id: str
    expiry: pd.Timestamp
    strike: float
    opt_type: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    qty: float
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RollingLot:
    lot_id: int
    contract_id: str
    expiry: pd.Timestamp
    strike: float
    opt_type: str  # "C" or "P"
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    qty: float

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
    split_factor_since_open: float

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
    # Open rolling lots (for strategy_mode="rolling")
    rolling_lots: List[RollingLot] = field(default_factory=list)
    # Lot registry per ticker
    lots: Dict[int, RollingLot] = field(default_factory=dict)
    next_lot_id: int = 1
    # Stock hedge positions
    stock_pos_close: float = 0.0   # shares held at end-of-day (t-1)
    stock_pos_intraday: float = 0.0  # current stock position being adjusted
    last_spot: Optional[float] = None
    mtm_options: float = 0.0
    mtm_stock: float = 0.0
    cum_pnl: float = 0.0
    cum_pnl_vega: float = 0.0
    cum_pnl_gamma: float = 0.0
    cum_pnl_theta: float = 0.0
    cum_pnl_delta_hedge: float = 0.0
    cum_pnl_tc: float = 0.0

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
        """
        Load spot as a split-normalized series and also keep a per-day split factor.

        - 'spot'         : split-adjusted price (continuous series)
        - 'price_factor' : used to normalize strikes / moneyness for options
        - 'split_factor' : raw split coefficient on that date (e.g. 4.0 on a 4-for-1 split,
                           1.0 on normal days). This is what we will use to rescale positions.
        """
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

            # Raw close (unadjusted)
            if "close" in df.columns:
                close_raw = df["close"].astype(float)
            elif "adj_close" in df.columns:
                close_raw = df["adj_close"].astype(float)
            else:
                raise ValueError(f"{sym}: neither 'close' nor 'adj_close' in {path}")

            # Split coefficient from Alpha Vantage:
            #  - 1.0 on normal days
            #  - k (>1) on split days, e.g. 4.0 for a 4-for-1
            if "split_coeff" in df.columns:
                split_raw = df["split_coeff"].astype(float)
                split_raw = split_raw.replace(0.0, np.nan).fillna(1.0)
            else:
                split_raw = pd.Series(1.0, index=df.index)

            # Build price_factor for *normalized* spot (continuous series)
            split_level = split_raw.cumprod()
            level_last = float(split_level.iloc[-1])
            if level_last <= 0:
                level_last = 1.0

            price_factor = level_last / split_level
            spot_norm = close_raw / price_factor

            out = pd.DataFrame({
                "spot": close_raw,
                # "price_factor": price_factor.astype(float),
                "split_factor": split_raw.astype(float),
            })
            out.index = split_level.index
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

    def get_split_factor(self, symbol: str, date: pd.Timestamp) -> float:
        """
        Return the raw split factor for this date:
          - 1.0 on normal days
          - k (>1) on split days (e.g. 4.0 for a 4-for-1)
        """
        df = self.spot.get(symbol)
        if df is None:
            return 1.0
        try:
            return float(df.loc[date, "split_factor"])
        except KeyError:
            return 1.0

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
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()
            df["strike"] = df["strike"].astype(float)
            df["type"] = df["type"].astype(str).str.upper().str[0]

            df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()
            if df.empty:
                print(f"[WARN] No option rows for {sym} in backtest window.")
                continue

            # Merge split_factor for strike normalization
            spot_df = self.spot.get(sym)
            if spot_df is not None and "split_factor" in spot_df.columns:
                pf = spot_df[["split_factor"]].reset_index()
                df = df.merge(pf, on="date", how="left")
            else:
                df["split_factor"] = 1.0
            df["split_factor"] = df["split_factor"].fillna(1.0).astype(float)

            # Compute mid price from bid/ask or mark/last
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
            df["mid"] = df["mid"]
            if "bid" in df.columns:
                df["bid"] = df["bid"].astype(float)
            if "ask" in df.columns:
                df["ask"] = df["ask"].astype(float)
            for greek in ["delta", "gamma", "vega", "theta"]:
                if greek in df.columns:
                    df[greek] = df[greek].astype(float)

            # Normalize strike for split-adjusted moneyness calculation
            df["strike_eff"] = df["strike"] / df["split_factor"]
            # Derive moneyness if not provided
            if "moneyness" not in df.columns:
                spot_merge = (
                    spot_df[["spot"]].reset_index().rename(columns={"spot": "_spot_for_mny"})
                    if spot_df is not None and not spot_df.empty else None
                )
                if spot_merge is not None:
                    df = df.merge(spot_merge, on="date", how="left")
                    df["_spot_for_mny"] = df["_spot_for_mny"].astype(float)
                    df["moneyness"] = np.where(
                        df["_spot_for_mny"] > 0,
                        df["strike_eff"] / df["_spot_for_mny"],
                        np.nan
                    )
                    df.drop(columns=["_spot_for_mny"], inplace=True)
                elif "underlying_price" in df.columns:
                    up = df["underlying_price"].astype(float)
                    df["moneyness"] = np.where(up > 0, df["strike_eff"] / up, np.nan)
                else:
                    df["moneyness"] = np.nan

            self.options[sym] = df.drop_duplicates()
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

class Strategy:
    """
    Straddle / Strangle earnings strategy:
    - On entry date (T-1):
      - "straddle": short vega_target of ATM front-month straddle
      - "strangle": short vega_target of symmetric OTM strangle with ± strangle_mny_offset around ATM
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
        self.backtester = None

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
        """
        Decide side ('short'/'long'/'flat') and vega multiplier for a single event row.

        mode in:
          - 'short'       : use ShortScore_z only
          - 'long'        : use LongScore_z only
          - 'ls','long_short' : pick the side (short vs long) with better expected PnL
                                based on training rules.
        """
        y = int(row["AnchorDate"].year)

        # Normalize mode aliases
        if mode in ("ls", "long_short", "long-short"):
            mode = "long_short"

        if mode == "short":
            rule = self._short_rules.get(y)
            if not rule:
                return "flat", 0.0
            s = float(row["ShortScore_z"])
            b = self._assign_bin(s, rule["edges"])
            if b is None or b not in rule["good_bins"]:
                return "flat", 0.0
            v = rule["vega_mult"].get(b, 0.0)
            return ("short", v if v > 0 else 0.0)

        if mode == "long":
            rule = self._long_rules.get(y)
            if not rule:
                return "flat", 0.0
            s = float(row["LongScore_z"])
            b = self._assign_bin(s, rule["edges"])
            if b is None or b not in rule["good_bins"]:
                return "flat", 0.0
            v = rule["vega_mult"].get(b, 0.0)
            return ("long", v if v > 0 else 0.0)

        if mode == "long_short":
            rule = self._ls_rules.get(y)
            if not rule:
                return "flat", 0.0

            s_s = float(row["ShortScore_z"])
            s_l = float(row["LongScore_z"])
            b_s = self._assign_bin(s_s, rule["short_edges"])
            b_l = self._assign_bin(s_l, rule["long_edges"])

            candidates = []

            if b_s is not None and b_s in rule["short_bins"]:
                v_s = rule["short_mult"].get(b_s, 0.0)
                if v_s > 0:
                    candidates.append(("short", v_s))

            if b_l is not None and b_l in rule["long_bins"]:
                v_l = rule["long_mult"].get(b_l, 0.0)
                if v_l > 0:
                    candidates.append(("long", v_l))

            if not candidates:
                return "flat", 0.0

            # Pick side with highest vega multiplier; tie → prefer short.
            candidates.sort(key=lambda x: x[1], reverse=True)
            if len(candidates) >= 2 and candidates[0][1] == candidates[1][1]:
                for side, v in candidates:
                    if side == "short":
                        return "short", v
            return candidates[0]

        # Unknown mode → flat
        return "flat", 0.0

    def compute_signal_vega(self, date: pd.Timestamp, symbol: str, base_vega: float) -> float:
        """
        Compute signal-adjusted vega target (sign + multiplier) for a given (date, symbol).
        If no signal or no rule, return 0 (no trade).
        """
        if (not self.config.get("use_signal", False)) or self.signals_df is None:
            return base_vega

        mapping_sym = self.entry_exit_map.get(symbol, {})
        meta = mapping_sym.get(date)
        if not meta:
            # only on entry dates
            return 0.0

        event_day = pd.to_datetime(meta["event_day"]).normalize()
        d = self.signals_df
        rows = d[(d["Symbol"] == symbol) & (d["EventDate"] == event_day)]
        if rows.empty:
            return 0.0

        row = rows.iloc[0]
        mode = self.config.get("signal_mode", "short")
        side, mult = self._signal_decision_for_row(row, mode)
        if side == "flat" or mult <= 0:
            return 0.0

        v = base_vega * mult
        # Earnings convention: short vol = positive vega_target
        return v if side == "short" else -v


    def build_entry_exit_map(self, market: MarketData) -> Dict[str, Dict[pd.Timestamp, Dict[str, Any]]]:
        """
        Build any pre-computed entry/exit metadata needed by the strategy.

        - earnings mode: build per-symbol entry/exit windows for earnings trades
        - rolling mode: currently no special schedule → empty map
        """

        cfg = self.config
        mode = cfg.get("strategy_mode", "earnings")

        if mode != "earnings":
            # No entry/exit concept for daily rolling (trade every day)
            self.entry_exit_map = {}
            return self.entry_exit_map

        # === existing earnings-specific logic below ===

        # Default lags if config doesn't provide full mapping
        default_entry_lag = {"BMO": -1, "AMC": 0, "DURING": 0, "UNKNOWN": 0}
        default_exit_lag = {"BMO": 0, "AMC": 1, "DURING": 1, "UNKNOWN": 1}

        cfg_entry = cfg.get("entry_lag", {})
        cfg_exit = cfg.get("exit_lag", {})

        entry_lag_cfg = {**default_entry_lag, **cfg_entry}
        exit_lag_cfg = {**default_exit_lag, **cfg_exit}

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

                anchor_idx = cal.searchsorted(ev_day)
                if anchor_idx >= len(cal):
                    continue

                entry_lag = entry_lag_cfg.get(timing, 0)
                exit_lag = exit_lag_cfg.get(timing, 1)

                entry_idx = anchor_idx + entry_lag
                exit_idx = anchor_idx + exit_lag

                if not (0 <= entry_idx < len(cal)):
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

        """
        Main public entry point.

        It routes to more specific builders based on config:
          - strategy_mode == "earnings":
                earnings_structure in {"straddle", "strangle"}
          - strategy_mode == "rolling":
                single-leg rolling option (put/call) with generic selection
        """

        cfg = self.config
        mode = cfg.get("strategy_mode", "earnings")

        if mode == "earnings":
            struct = cfg.get("earnings_structure", "straddle")
            return self._compute_earnings_targets(
                date=date,
                symbol=symbol,
                state=state,
                market=market,
                vega_target=vega_target,
                structure=struct,
            )

        elif mode == "rolling":
            return self._compute_rolling_option_targets(
                date=date,
                symbol=symbol,
                state=state,
                market=market,
                vega_target=vega_target,
            )

        else:
            # Unknown / inactive mode → no trade
            return []


    def _size_vega_legs(
        self,
        symbol: str,
        opt_rows: List[pd.Series],
        vega_target: float,
    ) -> List[Dict[str, Any]]:
        """
        Generic vega-based sizing:

        - opt_rows: list of chosen option rows (each with columns:
                    contractID, expiration, strike, type, vega, etc.).
        - vega_target: TOTAL vega magnitude requested at portfolio level.
                       Sign convention:
                         * earnings:
                             vega_target > 0  → SHORT vol (short options)
                             vega_target < 0  → LONG vol (long options)
                         * rolling:
                             we invert sign before calling, so that
                             vega_target > 0  → LONG vol (long options).

        Returns a list of target dicts with proper "qty".
        """

        cfg = self.config
        direction = cfg["rolling_direction"]

        if not opt_rows:
            return []

        if abs(vega_target) < 1e-12:
            return []

        mag = abs(vega_target)

        # Collect vegas for each leg
        legs_with_vega = []
        for row in opt_rows:
            v = float(row.get("vega", np.nan))
            if not np.isfinite(v) or abs(v) < 1e-12:
                continue
            legs_with_vega.append((row, v))

        if not legs_with_vega:
            return []

        denom = sum(abs(v) for (_, v) in legs_with_vega)
        if denom <= 1e-12:
            return []

        scale = mag / denom

        targets: List[Dict[str, Any]] = []
        for row, v_per_contract in legs_with_vega:
            qty = direction * scale  # same scale across legs

            targets.append(
                {
                    "contract_id": str(row["contractID"]),
                    "symbol": symbol,
                    "expiry": pd.to_datetime(row["expiration"]).normalize(),
                    "strike": float(row["strike"]),
                    "type": str(row["type"]).upper(),
                    "qty": qty,
                }
            )

        return targets

    def _compute_earnings_targets(
        self,
        date: pd.Timestamp,
        symbol: str,
        state: TickerState,
        market: MarketData,
        vega_target: float,
        structure: str,
    ) -> List[Dict[str, Any]]:

        """
        Earnings strategies:
          - structure == "straddle": ATM call + ATM put
          - structure == "strangle": OTM call/put at +/- offset
        """

        cfg = self.config

        # Only trade on entry dates
        meta = self.entry_exit_map.get(symbol, {}).get(date)
        if meta is None:
            return []

        chain = market.get_chain(symbol, date)
        if chain.empty:
            return []

        chain = chain.copy()
        chain["expiration"] = pd.to_datetime(chain["expiration"]).dt.normalize()
        chain["dte"] = (chain["expiration"] - date).dt.days

        min_dte = cfg.get("min_dte_for_entry", 5)
        max_dte = cfg.get("max_dte_for_entry", 30)
        eligible = chain[(chain["dte"] >= min_dte) & (chain["dte"] <= max_dte)]
        if eligible.empty:
            return []

        # Pick nearest eligible expiry
        first = eligible.sort_values("expiration").iloc[0]
        expiry = first["expiration"]
        sub = chain[chain["expiration"] == expiry].copy()
        if sub.empty:
            return []

        # We need a notion of moneyness; if missing or NaN, rebuild it
        if ("moneyness" not in sub.columns) or sub["moneyness"].isna().all():
            spot_today = market.get_spot(symbol, date)
            if spot_today is None or not np.isfinite(spot_today):
                return []

            if "strike_eff" in sub.columns:
                sub["moneyness"] = sub["strike_eff"] / spot_today
            else:
                sub["moneyness"] = sub["strike"] / spot_today

        calls = sub[sub["type"].str.upper().str[0] == "C"].copy()
        puts  = sub[sub["type"].str.upper().str[0] == "P"].copy()
        if calls.empty or puts.empty:
            return []

        legs: List[pd.Series] = []

        if structure == "strangle":
            offset = cfg.get("strangle_mny_offset", 0.03)
            target_call_mny = 1.0 + offset
            target_put_mny  = 1.0 - offset

            calls["mny_diff"] = (calls["moneyness"] - target_call_mny).abs()
            puts["mny_diff"]  = (puts["moneyness"]  - target_put_mny).abs()

        else:  # "straddle" default
            calls["mny_diff"] = (calls["moneyness"] - 1.0).abs()
            puts["mny_diff"]  = (puts["moneyness"]  - 1.0).abs()

        call = calls.sort_values("mny_diff").iloc[0]
        put  = puts.sort_values("mny_diff").iloc[0]

        legs = [call, put]

        # Earnings: vega_target > 0 → short vol (short options).
        # So we pass vega_target as-is to _size_vega_legs.
        return self._size_vega_legs(symbol, legs, vega_target)

    def _compute_rolling_option_targets(
            self,
            date: pd.Timestamp,
            symbol: str,
            state: TickerState,
            market: MarketData,
            vega_target: float,
    ) -> List[Dict[str, Any]]:
        """
        Rolling single-leg option strategy with *lot tracking*.

        Semantics:
          - Each trading day:
              * we keep existing lots whose exit_date >= today,
              * we drop lots whose exit_date < today,
              * we open ONE new lot (if possible) on a contract selected by:
                    leg_type (P/C), maturity_index, select_by (delta/moneyness)
              * new lot is sized to have vega ~= |vega_target| (per LOT),
                sign chosen so that we are LONG vol (buy put/call).
          - Target positions are the sum of all *open* lots per contract.
        """

        cfg = self.config
        holding_days = int(cfg.get("rolling_holding_days", 20))
        holding_days = max(1, holding_days)

        # 1) Keep only lots that are still live as of 'date'
        open_lots: List[RollingLot] = []
        for lot in state.rolling_lots:
            if date <= lot.exit_date:
                open_lots.append(lot)

        # 2) Try to create today's new lot
        chain = market.get_chain(symbol, date)
        if not chain.empty:
            chain = chain.copy()
            chain["expiration"] = pd.to_datetime(chain["expiration"]).dt.normalize()
            chain["dte"] = (chain["expiration"] - date).dt.days

            # Global DTE filter for rolling
            min_dte = cfg.get("rolling_min_dte", 1)
            max_dte = cfg.get("rolling_max_dte", 365)
            chain = chain[(chain["dte"] >= min_dte) & (chain["dte"] <= max_dte)]

            if not chain.empty:
                leg_type = cfg.get("rolling_leg_type", "P").upper()
                chain = chain[chain["type"].str.upper().str[0] == leg_type]

                if not chain.empty:
                    expiries = sorted(chain["expiration"].unique())
                    if expiries:
                        idx = cfg.get("rolling_maturity_index", 0)
                        if idx >= len(expiries):
                            idx = len(expiries) - 1
                        expiry = expiries[idx]

                        sub = chain[chain["expiration"] == expiry].copy()
                        if not sub.empty:
                            select_by = cfg.get("rolling_select_by", "delta")

                            if select_by == "delta":
                                if "delta" not in sub.columns:
                                    sub = pd.DataFrame()
                                else:
                                    target_delta = cfg.get("rolling_target_delta", -0.20)
                                    sub["sel_diff"] = (sub["delta"] - target_delta).abs()
                            elif select_by == "moneyness":
                                if "moneyness" not in sub.columns:
                                    sub = pd.DataFrame()
                                else:
                                    target_mny = cfg.get("rolling_target_mny", 1.0)
                                    sub["sel_diff"] = (sub["moneyness"] - target_mny).abs()
                            else:
                                sub = pd.DataFrame()

                            if not sub.empty:
                                row = sub.sort_values("sel_diff").iloc[0]

                                vega_for_sizing = vega_target
                                sized = self._size_vega_legs(symbol, [row], vega_for_sizing)

                                if sized:
                                    leg = sized[0]

                                    # Compute exit_date on spot calendar
                                    cal = market.get_spot_calendar(symbol)
                                    if len(cal) > 0:
                                        try:
                                            cur_idx = cal.get_loc(date)
                                        except KeyError:
                                            cur_idx = cal.searchsorted(date)
                                        exit_idx = cur_idx + holding_days - 1
                                        if exit_idx >= len(cal):
                                            exit_idx = len(cal) - 1
                                        exit_date = cal[exit_idx]
                                    else:
                                        exit_date = date + pd.tseries.offsets.BDay(
                                            holding_days - 1
                                        )

                                    # *** IMPORTANT: do NOT hold beyond expiry ***
                                    expiry_dt = pd.to_datetime(leg["expiry"]).normalize()
                                    exit_date = min(exit_date, expiry_dt)

                                    lot_id = self.backtester.register_new_lot(
                                        symbol=symbol,
                                        contract_id=leg["contract_id"],
                                        expiry=leg["expiry"],
                                        strike=leg["strike"],
                                        opt_type=leg["type"],
                                        entry_date=date,
                                        exit_date=exit_date,
                                        qty=leg["qty"],
                                        meta={"mode": "rolling"},
                                    )

                                    new_lot = RollingLot(
                                        lot_id=lot_id,
                                        contract_id=leg["contract_id"],
                                        expiry=leg["expiry"],
                                        strike=leg["strike"],
                                        opt_type=leg["type"],
                                        entry_date=date,
                                        exit_date=exit_date,
                                        qty=leg["qty"],
                                    )
                                    open_lots.append(new_lot)

                                    self.backtester.rolling_entries_log.append({
                                        "date": date,
                                        "symbol": symbol,
                                        "lot_id": lot_id,
                                        "contract_id": leg["contract_id"],
                                        "expiry": leg["expiry"],
                                        "days_to_expiry_on_entry": int((leg["expiry"] - date).days),
                                        "close_date": exit_date,
                                        "strike": float(leg["strike"]),
                                        "option_type": leg["type"],
                                        "qty": float(leg["qty"]),
                                        "delta_on_entry": float(row.get("delta", np.nan)),
                                        "vega_on_entry": float(row.get("vega", np.nan)),
                                        "vega_exposure": float(row.get("vega", np.nan))
                                                         * float(leg["qty"]),
                                        "note": "ROLL_ENTRY",
                                    })

        # Update state with currently open lots
        state.rolling_lots = open_lots

        # 3) Aggregate open lots per contract -> target positions
        by_cid: Dict[str, Dict[str, Any]] = {}
        for lot in open_lots:
            cid = lot.contract_id
            if cid not in by_cid:
                by_cid[cid] = {
                    "contract_id": cid,
                    "symbol": symbol,
                    "expiry": lot.expiry,
                    "strike": lot.strike,
                    "type": lot.opt_type,
                    "qty": 0.0,
                }
            by_cid[cid]["qty"] += lot.qty

        return list(by_cid.values())

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

        call_vega = float(call["vega"])
        put_vega = float(put["vega"])
        tot_vega = abs(call_vega) + abs(put_vega)
        if tot_vega <= 1e-8:
            return []

        vega_per_call = call_vega
        vega_per_put = put_vega
        if abs(vega_per_call) <= 1e-8 or abs(vega_per_put) <= 1e-8:
            return []

        target_vega_each = -0.5 * vega_target
        qty_call = target_vega_each / vega_per_call
        qty_put = target_vega_each / vega_per_put

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
        self.strategy = Strategy(config)
        self.strategy.backtester = self
        self.strategy.build_entry_exit_map(self.market)

        # Results storage
        self.daily_pnl_rows: List[Dict[str, Any]] = []
        self.trade_rows: List[Dict[str, Any]] = []
        self.rolling_entries_log: List[Dict[str, Any]] = []
        self.roll_portfolio_rows: List[Dict[str, Any]] = []

        # Global lot registry + live lots tracking
        self.all_lots: Dict[int, Dict[str, Any]] = {}
        self.live_lots_by_date: Dict[pd.Timestamp, List[int]] = {}
        self._next_lot_id: int = 1

        # Initialize state for each symbol
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

    def register_new_lot(
        self,
        symbol: str,
        contract_id: str,
        expiry: pd.Timestamp,
        strike: float,
        opt_type: str,
        entry_date: pd.Timestamp,
        exit_date: pd.Timestamp,
        qty: float,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Create a new lot_id, store it globally + per-ticker, and return lot_id."""
        lot_id = self._next_lot_id
        self._next_lot_id += 1

        # Global registry for lot (for PnL attribution or debugging)
        self.all_lots[lot_id] = {
            "lot_id": lot_id,
            "symbol": symbol,
            "contract_id": contract_id,
            "expiry": expiry,
            "strike": float(strike),
            "opt_type": opt_type,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "qty": float(qty),
            "meta": meta or {},
        }
        # Per-ticker registry of lot
        st = self.state[symbol]
        lot = RollingLot(
            lot_id=lot_id,
            contract_id=contract_id,
            expiry=expiry,
            strike=float(strike),
            opt_type=opt_type,
            entry_date=entry_date,
            exit_date=exit_date,
            qty=float(qty),
        )
        st.lots[lot_id] = lot
        return lot_id

    @profile
    def _process_symbol_date(self, symbol: str, date: pd.Timestamp):
        st = self.state[symbol]
        cfg = self.config
        equity_prev = st.equity

        # Spot price for this date (already split-adjusted)
        spot = self.market.get_spot(symbol, date)
        if spot is None or not np.isfinite(spot):
            return
        # Initialize option PnL breakdown variables for this day
        option_pnl_entry = 0.0
        option_pnl_close = 0.0
        option_pnl_expire = 0.0

        # ====== Corporate action handling (splits) ======
        split_factor = self.market.get_split_factor(symbol, date)
        if split_factor > 1:
                # Adjust all existing positions and lots for stock split
            for pos in st.positions.values():
                pos.qty *= split_factor
                pos.split_factor_since_open *= split_factor
            # Adjust any open rolling lots for split (track adjusted strike and qty)
            for lot in st.rolling_lots:
                lot.qty *= split_factor
                # lot.strike /= split_factor

        mode = cfg.get("strategy_mode", "earnings")

        # -------------------------------
        # 1) Compute vega_target for today
        # -------------------------------
        if mode == "earnings":
            equity_scale = st.equity / cfg["initial_equity_per_ticker"] if cfg["reinvest"] else 1.0
            base_vega = cfg["base_vega_target"] * equity_scale
            vega_target = self.strategy.compute_signal_vega(date, symbol, base_vega) if cfg.get("use_signal", False) else base_vega
        elif mode == "rolling":
            holding_days = max(1, int(cfg.get("rolling_holding_days", 20)))
            rolling_vega_per_lot = float(cfg.get("rolling_vega_per_lot") or (cfg["base_vega_target"] / holding_days))
            equity_scale = st.equity / cfg["initial_equity_per_ticker"] if cfg.get("rolling_reinvest", False) else 1.0
            vega_target = rolling_vega_per_lot * equity_scale
        else:
            vega_target = 0.0

        # -------------------------------
        # 2) Determine target option positions
        # -------------------------------
        target_lines = self.strategy.compute_target_positions(date, symbol, st, self.market, vega_target)

        # Earnings strategy: if today is an exit date, go flat
        if mode == "earnings":
            mapping_sym = self.strategy.entry_exit_map.get(symbol, {})
            need_flat = any(meta_ev.get("exit_date") == date for meta_ev in mapping_sym.values())
            if need_flat:
                target_lines = []
        else:
            need_flat = False

        # Get today's option chain and index by contract ID
        chain_today = self.market.get_chain(symbol, date)
        if not chain_today.empty:
            chain_today = chain_today.copy()
            chain_today["cid"] = chain_today["contractID"].astype(str)
            chain_today_idx = chain_today.set_index("cid")
            cids_today = set(chain_today["cid"])
        else:
            chain_today_idx = None
            cids_today = set()

        # -------------------------------
        # 3) Handle expired or delisted positions (drop zombies)
        # -------------------------------
        for cid in list(st.positions.keys()):
            pos = st.positions[cid]
            # If we've reached or passed expiry, and the option is still in the book, force expiry payoff.
            if pos.expiry <= date:
                spot_expiry = self.market.get_spot(symbol, date)
                if spot_expiry is None or not np.isfinite(spot_expiry):
                    continue  # defensively skip if we have no spot

                if pos.opt_type == "C":
                    intrinsic = max(0.0, spot_expiry / pos.split_factor_since_open - pos.strike )
                else:
                    intrinsic = max(0.0, pos.strike / pos.split_factor_since_open- spot_expiry )

                exp_cash = pos.qty * intrinsic  # short qty → negative if ITM
                option_pnl_expire += exp_cash
                st.cash += exp_cash
                print('expired on', date, 'with expiry =',pos.expiry)

                # Remove the position from the book (no MTM from tomorrow)
                del st.positions[cid]

        # Carry over previous Greeks and price for PnL decomposition
        for pos in st.positions.values():
            pos.prev_iv = pos.iv
            pos.prev_delta = pos.delta
            pos.prev_gamma = pos.gamma
            pos.prev_vega = pos.vega
            pos.prev_theta = pos.theta
            pos.prev_price = pos.last_price

        # -------------------------------
        # 4) Build target quantity map (desired end-of-day positions)
        # -------------------------------
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

        # Track transaction costs
        pnl_tc_opt = 0.0  # options total TC (spread + commission)
        pnl_tc_stock = 0.0  # stock hedge TC

        # Determine contracts we need to trade (union of current and target)
        contract_ids = set(st.positions.keys()) | set(target_qty.keys())

        # -------------------------------
        # 5) Trade options to reach target positions
        # -------------------------------
        for cid in contract_ids:
            cur_pos = st.positions.get(cid)
            tgt_qty = target_qty.get(cid, 0.0) if mode == "rolling" else target_qty.get(cid, cur_pos.qty if cur_pos else 0.0)
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
                # Determine trade execution price (mid +/- spread)
                spread = option_spread_bps / 1e4
                trade_price = mid_price * (1 + spread) if trade_qty > 0 else mid_price * (1 - spread)
                # Cash flow from trade (positive if cash received, negative if paid)
                cash_change = -trade_qty * trade_price
                st.cash += cash_change

                # Transaction costs (spread slippage + commission)
                tc_spread = abs(trade_price - mid_price) * abs(trade_qty)
                tc_comm = abs(trade_qty) * commission_per_contract
                pnl_tc_opt += (tc_spread + tc_comm)

                # Prepare trade record
                trade_record = {
                    "Date": date,
                    "Symbol": symbol,
                    "ContractID": cid,
                    "Expiry": expiry,
                    "Strike": strike,
                    "Type": opt_type,
                    "TradeQty": trade_qty,
                    "TradePrice": trade_price,
                    "TradeNotional": trade_qty * trade_price,
                    "Spot": spot,
                    "IV": iv,
                    "Delta": delta,
                    "Gamma": gamma,
                    "Vega": vega,
                    "Theta": theta,
                }
                # Allocate PnL to entry/close categories
                if cur_pos is None:
                    # Opened a new position
                    option_pnl_entry += cash_change
                    # Link CloseDate for this entry trade (from lot exit_date)
                    close_date = None
                    for lot in self.all_lots.values():
                        if lot["symbol"] == symbol and lot["contract_id"] == cid and lot["entry_date"] == date:
                            close_date = lot.get("exit_date")
                            break
                    if close_date is not None:
                        trade_record["CloseDate"] = close_date
                else:
                    # Adjusting an existing position
                    if cur_pos.qty * tgt_qty < 0:
                        # Position side flipped (closed current and opened opposite)
                        # Split cash_change into closing and opening parts by quantity ratio
                        close_qty = -cur_qty  # amount closed to flatten current
                        entry_qty = tgt_qty   # new position quantity after flip
                        if abs(trade_qty) > 1e-8:
                            cash_close = cash_change * (abs(close_qty) / abs(trade_qty))
                            cash_entry = cash_change * (abs(entry_qty) / abs(trade_qty))
                        else:
                            cash_close = cash_entry = 0.0
                        option_pnl_close += cash_close
                        option_pnl_entry += cash_entry
                        print('should not flip', date)
                    else:
                        if abs(tgt_qty) > abs(cur_pos.qty):
                            # Increased position in same direction
                            option_pnl_entry += cash_change
                            print('should not increase', date)
                        elif abs(tgt_qty) < abs(cur_pos.qty):
                            # Reduced position (partial or full close)
                            option_pnl_close += cash_change
                # Append trade record
                self.trade_rows.append(trade_record)

            # Update or remove position in the book
            if abs(tgt_qty) <= 1e-10:
                # Target is zero -> close the position
                if cid in st.positions:
                    del st.positions[cid]
            else:
                if cur_pos is None:
                    # New position opened
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
                        split_factor_since_open=1.0
                    )
                else:
                    # Position exists, just update its fields and quantity
                    cur_pos.qty = tgt_qty
                    cur_pos.last_price = mid_price
                    cur_pos.iv = iv
                    cur_pos.delta = delta
                    cur_pos.gamma = gamma
                    cur_pos.vega = vega
                    cur_pos.theta = theta

        # -------------------------------
        # 6) Delta hedging (stock) + TC on stock
        # -------------------------------
        pnl_delta_hedge = 0.0
        if cfg["delta_hedge"]:
            port_delta = sum(pos.delta * pos.qty for pos in st.positions.values())
            tgt_stock_pos = -port_delta
            trade_shares = tgt_stock_pos - st.stock_pos_intraday
            if abs(trade_shares) > 1e-8:
                spread = cost_model["stock_spread_bps"] / 1e4
                trade_price = spot * (1 + spread) if trade_shares > 0 else spot * (1 - spread)
                cash_change = -trade_shares * trade_price
                st.cash += cash_change
                # Transaction cost for stock trade (spread only, commission assumed negligible or included above)
                tc_stock = abs(trade_price - spot) * abs(trade_shares)
                pnl_tc_stock += tc_stock
            if st.last_spot is not None:
                dS = spot - st.last_spot
                pnl_delta_hedge = st.stock_pos_close * dS
            st.stock_pos_intraday = tgt_stock_pos
            st.stock_pos_close = tgt_stock_pos
        else:
            if st.last_spot is not None:
                dS = spot - st.last_spot
                pnl_delta_hedge = st.stock_pos_close * dS

        # -------------------------------
        # 7) Option PnL decomposition (Greek attribution for analysis only)
        # -------------------------------
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
                dP = (pos.last_price - pos.prev_price) * pos.qty
                option_pnl += dP

            if pos.prev_gamma is not None:
                port_gamma_prev += pos.prev_gamma * pos.qty
            if pos.prev_vega is not None:
                port_vega_prev += pos.prev_vega * pos.qty
            if pos.prev_theta is not None:
                port_theta_prev += pos.prev_theta * pos.qty

            if (pos.prev_iv is not None and
                    np.isfinite(pos.prev_iv) and np.isfinite(pos.iv)):
                dIV = pos.iv - pos.prev_iv
                w = abs(pos.prev_vega * pos.qty)
                dIV_weighted_num += w * dIV
                dIV_weighted_den += w

            port_delta_today += pos.delta * pos.qty
            port_gamma_today += pos.gamma * pos.qty
            port_vega_today += pos.vega * pos.qty
            port_theta_today += pos.theta * pos.qty

        dS = 0.0
        if st.last_spot is not None:
            dS = spot - st.last_spot
        dt = 1.0 / 252.0
        dIV_eff = dIV_weighted_num / dIV_weighted_den if dIV_weighted_den > 0 else 0.0

        pnl_gamma = 0.5 * port_gamma_prev * (dS ** 2)
        pnl_vega = port_vega_prev * dIV_eff
        pnl_theta = port_theta_prev * dt

        # -------------------------------
        # 8) Mark-to-market & equity update
        # -------------------------------
        mtm_options = sum(pos.last_price * pos.qty for pos in st.positions.values())
        mtm_stock = st.stock_pos_close * spot

        st.mtm_options = mtm_options
        st.mtm_stock = mtm_stock

        # Compute end-of-day equity from cash + MTM (realized + unrealized PnL)
        st.equity = st.cash + mtm_options + mtm_stock
        pnl_total = st.equity - equity_prev

        # Total transaction costs today
        pnl_tc_total = pnl_tc_opt + pnl_tc_stock

        # Update cumulative PnL trackers
        st.cum_pnl += pnl_total
        st.cum_pnl_vega += pnl_vega
        st.cum_pnl_gamma += pnl_gamma
        st.cum_pnl_theta += pnl_theta
        st.cum_pnl_delta_hedge += pnl_delta_hedge
        st.cum_pnl_tc += pnl_tc_total

        # Prepare daily PnL row with breakdown
        row = {
            "Date": date,
            "Symbol": symbol,
            "Equity": st.equity,
            "Cash": st.cash,
            "MTM_Options": st.mtm_options,
            "MTM_Stock": st.mtm_stock,
            "Spot": spot,
            "DailyPnL": pnl_total,
            "OptionPnL_Entry": option_pnl_entry,
            "OptionPnL_Close": option_pnl_close,
            "OptionPnL_Expire": option_pnl_expire,
            "PnL_gamma": pnl_gamma,
            "PnL_vega": pnl_vega,
            "PnL_theta": pnl_theta,
            "PnL_deltaHedge": pnl_delta_hedge,
            "PnL_TC": pnl_tc_total,
            "PnL_TC_Options": pnl_tc_opt,
            "PnL_TC_Stock": pnl_tc_stock,
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

        # Track live lot IDs for rolling strategy (for debugging or analysis)
        if mode != "earnings":
            live_ids = [lot.lot_id for lot in st.rolling_lots]
            self.live_lots_by_date.setdefault(date, []).extend(live_ids)

          # NEW: end-of-day portfolio snapshot per lot (rolling strategy)
            if chain_today_idx is not None:
                for lot in st.rolling_lots:
                    cid = str(lot.contract_id)
                    snap = {
                        "Date": date,
                        "Symbol": symbol,
                        "LotID": lot.lot_id,
                        "ContractID": cid,
                        "EntryDate": lot.entry_date,
                        "ExitDate": lot.exit_date,
                        "Expiry": lot.expiry,
                        "Strike": float(lot.strike),
                        "Type": lot.opt_type,
                        "Qty": float(lot.qty),
                    }
                    try:
                        row_opt = chain_today_idx.loc[cid]
                        mid = float(row_opt["mid"])
                        iv = float(row_opt.get("implied_volatility", np.nan))
                        delta = float(row_opt.get("delta", np.nan))
                        gamma = float(row_opt.get("gamma", np.nan))
                        vega = float(row_opt.get("vega", np.nan))
                        theta = float(row_opt.get("theta", np.nan))
                    except KeyError:
                        # Option not in today chain (delisted / missing data)
                        mid = np.nan
                        iv = delta = gamma = vega = theta = np.nan

                    snap.update(
                        {
                            "Mid": mid,
                            "MTM": mid * snap["Qty"] if np.isfinite(mid) else np.nan,
                            "IV": iv,
                            "Delta": delta,
                            "Gamma": gamma,
                            "Vega": vega,
                            "Theta": theta,
                            "Spot": spot,
                        }
                    )
                    self.roll_portfolio_rows.append(snap)


    def _export_results(self):
        if not self.daily_pnl_rows:
            print("[WARN] No PnL rows to export.")
            return
        df = pd.DataFrame(self.daily_pnl_rows).sort_values(["Date", "Symbol"]).reset_index(drop=True)
        df = df.bfill()  # forward-fill any NaNs for missing days per symbol

        # Pivot equity per symbol and compute total portfolio equity
        pivot_equity = df.pivot(index="Date", columns="Symbol", values="Equity")
        pivot_equity["PortfolioEquity"] = pivot_equity.sum(axis=1)

        # Flatten config dict to dataframe
        config_rows = _flatten_config_dict(self.config)
        df_config = pd.DataFrame(config_rows, columns=["Key", "Value"])

        # Prepare earnings events table
        df_earn = self.market.earnings.copy()
        if df_earn.empty:
            df_earn = pd.DataFrame(columns=["Symbol", "EventDay"])
        else:
            df_earn = df_earn.sort_values(["symbol", "event_day"]).copy()
            df_earn = df_earn.rename(columns={"symbol": "Symbol", "event_day": "EventDay"})

        # Build corporate actions table (splits/dividends) for reference
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
                df_c = df_c[(df_c["date"] >= START_DATE) & (df_c["date"] <= END_DATE)].copy()
                if df_c.empty:
                    continue
                lower_cols = {c.lower(): c for c in df_c.columns}
                div_col = None
                split_col = None
                for lc, orig in lower_cols.items():
                    if "dividend" in lc:
                        div_col = orig
                    if "split" in lc:
                        split_col = orig
                if div_col is None and split_col is None:
                    continue
                mask = False
                if div_col is not None:
                    mask = (df_c[div_col].fillna(0).astype(float) != 0)
                if split_col is not None:
                    split_mask = (df_c[split_col].fillna(1).astype(float) != 1)
                    mask = mask | split_mask if isinstance(mask, pd.Series) else split_mask
                df_c = df_c[mask]
                if df_c.empty:
                    continue
                cols = ["date"]
                if div_col is not None:
                    cols.append(div_col)
                if split_col is not None:
                    cols.append(split_col)
                df_c = df_c[cols].copy()
                df_c["Symbol"] = sym
                rename_map = {"date": "Date"}
                if div_col is not None:
                    rename_map[div_col] = "Dividend"
                if split_col is not None:
                    rename_map[split_col] = "SplitFactor"
                df_c = df_c.rename(columns=rename_map)
                corp_rows.append(df_c)
            if not corp_rows:
                return pd.DataFrame(columns=["Symbol", "Date", "Dividend", "SplitFactor"])
            df_corp = pd.concat(corp_rows, ignore_index=True)
            df_corp = df_corp[["Symbol", "Date", "Dividend", "SplitFactor"]].sort_values(["Symbol", "Date"])
            return df_corp

        df_corp = _build_corp_actions_df()

        # Prepare event-level PnL summary (for earnings mode, entry->exit window PnL)
        event_pnl_rows = []
        for sym, mapping in self.strategy.entry_exit_map.items():
            for entry_date, meta in mapping.items():
                exit_date = meta.get("exit_date")
                event_day = meta.get("event_day")
                timing = meta.get("timing", "UNKNOWN")
                if exit_date is None:
                    continue
                mask = (df["Symbol"] == sym) & (df["Date"] >= entry_date) & (df["Date"] <= exit_date)
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
        df_event_pnl = pd.DataFrame(event_pnl_rows) if event_pnl_rows else pd.DataFrame(columns=[
            "Symbol", "EntryDate", "EventDay", "ExitDate", "Timing",
            "EventWindowPnL", "EventWindowPnL_vega", "EventWindowPnL_gamma",
            "EventWindowPnL_theta", "EventWindowPnL_deltaHedge", "EventWindowPnL_TC",
        ])

        # Compute per-symbol event stats summary (hit ratios, etc.)
        if not df_event_pnl.empty:
            base = df_event_pnl.groupby("Symbol")["EventWindowPnL"].agg(
                N_events="count",
                Mean_EventPnL="mean",
                Std_EventPnL="std",
                Best_EventPnL="max",
                Worst_EventPnL="min",
            )
            hit = (df_event_pnl.assign(Positive=df_event_pnl["EventWindowPnL"] > 0)
                   .groupby("Symbol")["Positive"].mean().rename("HitRatio"))
            bmo = (df_event_pnl[df_event_pnl["Timing"] == "BMO"]
                   .groupby("Symbol")["EventWindowPnL"].agg(BMO_N="count", BMO_Mean_EventPnL="mean"))
            amc = (df_event_pnl[df_event_pnl["Timing"] == "AMC"]
                   .groupby("Symbol")["EventWindowPnL"].agg(AMC_N="count", AMC_Mean_EventPnL="mean"))
            df_event_stats = base.join(hit, how="left").join(bmo, how="left").join(amc, how="left").reset_index()
            for col in ["BMO_N", "AMC_N"]:
                if col in df_event_stats.columns:
                    df_event_stats[col] = df_event_stats[col].fillna(0).astype(int)
        else:
            df_event_stats = pd.DataFrame(columns=[
                "Symbol", "N_events", "Mean_EventPnL", "Std_EventPnL", "HitRatio",
                "Best_EventPnL", "Worst_EventPnL", "BMO_N", "BMO_Mean_EventPnL", "AMC_N", "AMC_Mean_EventPnL"
            ])

        # Write results to Excel with multiple sheets
        with pd.ExcelWriter(OUT_PNL_EXCEL, engine="xlsxwriter") as writer:
            pivot_equity.reset_index().to_excel(writer, sheet_name="PORTFOLIO", index=False)
            df_event_pnl.to_excel(writer, sheet_name="EVENT_PNL", index=False)
            df_event_stats.to_excel(writer, sheet_name="EVENT_STATS", index=False)
            df_config.to_excel(writer, sheet_name="CONFIG", index=False)
            df_earn.to_excel(writer, sheet_name="EARNINGS", index=False)
            df_corp[df_corp["SplitFactor"] != 1.0].to_excel(writer, sheet_name="CORP_ACTIONS", index=False)
            if not self.trade_rows:
                df_trades_all = pd.DataFrame(columns=[
                    "Date", "Symbol", "ContractID", "Expiry", "Strike", "Type",
                    "TradeQty", "TradePrice", "TradeNotional", "Spot", "IV", "Delta", "Gamma", "Vega", "Theta", "CloseDate"
                ])
            else:
                df_trades_all = pd.DataFrame(self.trade_rows)
            df_trades_all.sort_values(["Date", "Symbol"]).to_excel(writer, sheet_name="ALL_TRADES", index=False)
            # For rolling strategy, output entry log if available
            if self.config.get("strategy_mode") == "rolling" and self.rolling_entries_log:
                pd.DataFrame(self.rolling_entries_log).to_excel(writer, sheet_name="ROLL_ENTRIES", index=False)

            if self.config.get("strategy_mode") == "rolling" and self.roll_portfolio_rows:
                df_roll_all = pd.DataFrame(self.roll_portfolio_rows)
            else:
                df_roll_all = pd.DataFrame(
                    columns=[
                        "Date", "Symbol", "LotID", "ContractID",
                        "EntryDate", "ExitDate", "Expiry", "Strike", "Type", "Qty",
                        "Mid", "MTM", "IV", "Delta", "Gamma", "Vega", "Theta", "Spot",
                    ]
                )

            # Per-symbol sheets for daily PnL and trades
            for sym in self.symbols:
                df_sym = df[df["Symbol"] == sym].copy()
                if not df_sym.empty:
                    df_sym.to_excel(writer, sheet_name=sym[:31], index=False)
                df_tr_sym = df_trades_all[df_trades_all["Symbol"] == sym].copy()
                if not df_tr_sym.empty:
                    df_tr_sym.to_excel(writer, sheet_name=f"{sym}_TRADES"[:31], index=False)

                if not df_roll_all.empty:
                    df_roll_sym = df_roll_all[df_roll_all["Symbol"] == sym].copy()
                    if not df_roll_sym.empty:
                        df_roll_sym.to_excel(
                            writer,
                            sheet_name=f"{sym}_ROLL_PTF"[:31],  # e.g. TICKER_ROLL_PTF
                            index=False,
                        )
        print(f"[INFO] Backtest results written to {OUT_PNL_EXCEL}")

# If run as a script, execute backtest
if __name__ == "__main__":
    bt = Backtester(BACKTEST_CONFIG, SYMBOLS)
    bt.run()
