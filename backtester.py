import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import datetime as dt

import numpy as np
import pandas as pd
from tqdm import tqdm
from bs_pricing import bs_price, bs_greeks

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
# "NVDA","MSFT","AAPL"
# ]

START_DATE = pd.Timestamp("2020-01-01")
END_DATE   = pd.Timestamp("2025-11-16")

# Paths
CORP_DIR     = pathlib.Path("alpha_corp_actions")   # *_daily_adjusted.parquet
OPTIONS_DIR  = pathlib.Path("alpha_options_raw")    # <SYM>.parquet
EARNINGS_CSV = "earnings.csv"                      # columns: symbol, event_day

name_strat = r"\small_test.xlsx"
OUT_PNL_EXCEL = rf"C:\Users\antoi\Documents\Antoine\Projets_Python\Trading Vol on Earnings\outputs{name_strat}"

BACKTEST_CONFIG = {
    "initial_equity_per_ticker": 100.0,
    "reinvest": True,
    "base_vega_target": 1.0 / 20,

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
    "strangle_mny_offset": 0.035,

    # Rolling daily option sub-type
    # Here we implement your “buy a 20δ put on 3rd maturity”
    "rolling_leg_type": "P",  # "P" or "C"
    "rolling_select_by": "delta",  # "delta" or "moneyness"
    "rolling_target_delta": -0.20,  # used if select_by == "delta"
    "rolling_target_mny": 1.0,  # used if select_by == "moneyness"
    "rolling_maturity_index": 3,  # 0=nearest, 1=2nd, 2=3rd...
    "rolling_min_dte": 1,  # optional extra filters
    "rolling_max_dte": 365,
    "rolling_holding_days": 20,  # hold each rolling lot for 20 business days
    "rolling_vega_per_lot": None,  # if None, we auto = base_vega_target / holding_days
    "rolling_reinvest": False,  # safer default for rolling
    "rolling_direction": +1,

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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Canonical portfolio object
# ------------------------------------------------------------

@dataclass
class RollingPtf:
    """
    One 'portfolio slice' opened on entry_date and closed on exit_date.

    It may contain several contracts (legs), all opened at entry_date and
    all closed at exit_date (unless they hit expiry earlier).
    """
    ptf_id: int
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp

    contract_ids: List[str]        # option identifier in chains
    expiries: List[pd.Timestamp]
    strikes: List[float]
    opt_types: List[str]           # "C" or "P"
    qtys: List[float]              # signed quantities (>0 long, <0 short)

    # Previous-day pricing/greeks for decomposition
    prev_prices: List[float] = field(default_factory=list)
    prev_ivs: List[float] = field(default_factory=list)
    prev_deltas: List[float] = field(default_factory=list)
    prev_gammas: List[float] = field(default_factory=list)
    prev_vegas: List[float] = field(default_factory=list)
    prev_thetas: List[float] = field(default_factory=list)

    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TickerState:
    equity: float
    cash: float

    # Rolling portfolios for this symbol
    rolling_ptfs: Dict[int, RollingPtf] = field(default_factory=dict)
    next_ptf_id: int = 1

    # Stock hedge positions
    stock_pos_close: float = 0.0      # shares held at close of previous day
    stock_pos_intraday: float = 0.0   # intraday position today

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
        # self._load_options()
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
    def load_symbol_options(self, sym: str) -> None:
        """
        Lazily load options for a single symbol and store them in self.options[sym].

        Safe to call multiple times: if already loaded, it does nothing.
        """
        if sym in self.options:
            return

        path = OPTIONS_DIR / f"{sym}.parquet"
        if not path.exists():
            print(f"[WARN] Options file missing for {sym}: {path}")
            return

        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()
        df["strike"] = df["strike"].astype(float)
        df["type"] = df["type"].astype(str).str.upper().str[0]

        df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()
        if df.empty:
            print(f"[WARN] No option rows for {sym} in backtest window.")
            return

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
        spot_df = self.spot.get(sym)
        if ("moneyness" not in df.columns) or df["moneyness"].isna().all():
            if spot_df is not None and not spot_df.empty:
                spot_merge = (
                    spot_df[["spot"]]
                    .reset_index()
                    .rename(columns={"spot": "_spot_for_mny"})
                )
                df = df.merge(spot_merge, on="date", how="left")
                df["_spot_for_mny"] = df["_spot_for_mny"].astype(float)
                df["moneyness"] = np.where(
                    df["_spot_for_mny"] > 0,
                    df["strike_eff"] / df["_spot_for_mny"],
                    np.nan,
                )
                df.drop(columns=["_spot_for_mny"], inplace=True)
            elif "underlying_price" in df.columns:
                up = df["underlying_price"].astype(float)
                df["moneyness"] = np.where(up > 0, df["strike_eff"] / up, np.nan)
            else:
                df["moneyness"] = np.nan

        df = df.drop_duplicates()
        self.options[sym] = df
        print(f"[INFO] Options loaded for {sym}: {df.shape[0]:,} rows")

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

        IMPORTANT: in the RollingPtf framework, this function's job is to
        create RollingPtfs (portfolios) by calling
            self.backtester.register_new_ptf(...)
        when appropriate.

        The backtester then loops over live RollingPtfs to compute PnL.
        The *return value* is no longer used by the engine and is kept only
        for backwards compatibility / debugging.
        """

        cfg = self.config
        mode = cfg.get("strategy_mode", "earnings")

        # -----------------------------
        # 1) Earnings mode
        # -----------------------------
        if mode == "earnings":
            # Use the existing earnings selection logic to pick the legs,
            # then wrap them into a single RollingPtf whose entry/exit dates
            # come from entry_exit_map (precomputed elsewhere).
            struct = cfg.get("earnings_structure", "straddle")

            # If today is not an entry day, do nothing.
            meta = self.entry_exit_map.get(symbol, {}).get(date)
            if meta is None:
                return []

            exit_date = meta.get("exit_date")
            if exit_date is None:
                # No exit date → don't open anything.
                return []

            # Let the existing earnings logic pick & size the legs
            legs = self._compute_earnings_targets(
                date=date,
                symbol=symbol,
                state=state,
                market=market,
                vega_target=vega_target,
                structure=struct,
            )
            if not legs:
                return []

            self.backtester.register_new_ptf(
                symbol=symbol,
                entry_date=date,
                exit_date=exit_date,
                legs=legs,
                meta={
                    "mode": "earnings",
                    "event_day": meta.get("event_day"),
                },
            )

            # For debugging only – doesn't affect the engine
            return legs

        # -----------------------------
        # 2) Rolling mode
        # -----------------------------
        elif mode == "rolling":
            # Rolling: one new RollingPtf per day (when possible),
            # with exit_date = min(expiry, target_close_date).
            return self._build_rolling_ptf_for_date(
                date=date,
                symbol=symbol,
                state=state,
                market=market,
                vega_target=vega_target,
            )

        # -----------------------------
        # 3) Unknown / inactive mode
        # -----------------------------
        else:
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

    def _build_rolling_ptf_for_date(
        self,
        date: pd.Timestamp,
        symbol: str,
        state: TickerState,
        market: MarketData,
        vega_target: float,
    ) -> List[Dict[str, Any]]:
        """
        Build ONE RollingPtf for this (date, symbol) in rolling mode.

        Logic:
          - Look at today's option chain.
          - Filter maturities by DTE and pick the maturity_index-th expiry.
          - Within that expiry, select one leg (P or C) by delta or moneyness.
          - Size the quantity so that portfolio vega ~= |vega_target| * rolling_direction.
          - Set exit_date = min(expiry, target_close_date) where target_close_date
            is 'rolling_holding_days' business days after entry (using a simple
            business-day calendar).
          - Call backtester.register_new_ptf(...) with a single-leg RollingPtf.

        Returns a list of leg dicts (for debugging only); the backtester uses
        the RollingPtf registry, not the return value.
        """
        cfg = self.config

        # If no vega target → no trade
        if vega_target is None or abs(vega_target) < 1e-12:
            return []

        chain = market.get_chain(symbol, date)
        if chain is None or chain.empty:
            return []

        chain = chain.copy()
        # Normalize column names we rely on
        if "expiration" not in chain.columns:
            if "expiry" in chain.columns:
                chain["expiration"] = chain["expiry"]
            else:
                return []

        # Compute DTE
        chain["expiration"] = pd.to_datetime(chain["expiration"]).dt.normalize()
        dte = (chain["expiration"] - date.normalize()).dt.days
        chain["dte"] = dte

        min_dte = int(cfg.get("rolling_min_dte", 1))
        max_dte = int(cfg.get("rolling_max_dte", 365))
        mask_dte = (chain["dte"] >= min_dte) & (chain["dte"] <= max_dte)
        chain = chain[mask_dte]
        if chain.empty:
            return []

        # Filter by option type (P/C)
        leg_type = cfg.get("rolling_leg_type", "P").upper()[0]
        chain = chain[chain["type"].str.upper().str[0] == leg_type]
        if chain.empty:
            return []

        # Pick maturity by index
        expiries = sorted(chain["expiration"].unique())
        if not expiries:
            return []

        idx = int(cfg.get("rolling_maturity_index", 0))
        if idx >= len(expiries):
            idx = len(expiries) - 1
        expiry = expiries[idx]

        sub = chain[chain["expiration"] == expiry].copy()
        if sub.empty:
            return []

        # Ensure we have moneyness if needed
        select_by = cfg.get("rolling_select_by", "delta")
        spot_today = market.get_spot(symbol, date)
        if spot_today is None or not np.isfinite(spot_today):
            return []

        if select_by == "moneyness":
            if "moneyness" not in sub.columns:
                if "strike_eff" in sub.columns:
                    sub["moneyness"] = sub["strike_eff"] / spot_today
                else:
                    sub["moneyness"] = sub["strike"] / spot_today

        # Select the contract
        if select_by == "delta":
            if "delta" not in sub.columns:
                return []
            target_delta = float(cfg.get("rolling_target_delta", -0.20))
            sub["delta_diff"] = (sub["delta"] - target_delta).abs()
            chosen = sub.sort_values("delta_diff").iloc[0]
        elif select_by == "moneyness":
            target_mny = float(cfg.get("rolling_target_mny", 1.0))
            sub["mny_diff"] = (sub["moneyness"] - target_mny).abs()
            chosen = sub.sort_values("mny_diff").iloc[0]
        else:
            # Fallback: closest to ATM by moneyness
            if "moneyness" not in sub.columns:
                if "strike_eff" in sub.columns:
                    sub["moneyness"] = sub["strike_eff"] / spot_today
                else:
                    sub["moneyness"] = sub["strike"] / spot_today
            sub["mny_diff"] = (sub["moneyness"] - 1.0).abs()
            chosen = sub.sort_values("mny_diff").iloc[0]

        # Vega sizing
        if "vega" not in chosen.index:
            return []
        vega_leg = float(chosen["vega"])
        if not np.isfinite(vega_leg) or abs(vega_leg) < 1e-12:
            return []

        direction = float(cfg.get("rolling_direction", -1.0))  # e.g. -1 for short vol
        total_vega = direction * abs(vega_target)
        qty = total_vega / vega_leg  # vega_leg typically > 0

        if abs(qty) < 1e-8:
            return []

        # Holding period & exit date: min(expiry, target_close_date)
        holding_days = max(1, int(cfg.get("rolling_holding_days", 20)))

        # Simple business-day calendar (Mon–Fri); you can replace with your own if needed
        # Include 'date' itself as the first business day.
        bdays = pd.bdate_range(start=date.normalize(), periods=holding_days)
        target_close_date = bdays[-1]

        expiry_date = pd.to_datetime(chosen["expiration"]).normalize()
        exit_date = min(expiry_date, target_close_date)

        # Build RollingPtf leg definition
        leg_dict = {
            "contract_id": chosen["contractID"],
            "expiry": expiry_date,
            "strike": float(chosen["strike"]),
            "type": chosen["type"],
            "qty": float(qty),
        }

        # Register portfolio
        self.backtester.register_new_ptf(
            symbol=symbol,
            entry_date=date.normalize(),
            exit_date=exit_date,
            legs=[leg_dict],
            meta={
                "mode": "rolling",
                "leg_type": leg_type,
                "select_by": select_by,
                "target_delta": cfg.get("rolling_target_delta"),
                "target_mny": cfg.get("rolling_target_mny"),
                "holding_days": holding_days,
            },
        )

        # Optional: keep a log for Excel sheet "ROLL_ENTRIES"
        if hasattr(self.backtester, "rolling_entries_log"):
            self.backtester.rolling_entries_log.append(
                {
                    "Symbol": symbol,
                    "EntryDate": date.normalize(),
                    "ExitDate": exit_date,
                    "ContractID": chosen["contractID"],
                    "Expiry": expiry_date,
                    "Strike": float(chosen["strike"]),
                    "Type": chosen["type"],
                    "Qty": float(qty),
                    "DTE_Entry": int((expiry_date - date.normalize()).days),
                    "TargetCloseDate": target_close_date,
                }
            )

        # Return legs for debugging only
        return [leg_dict]

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

        self.daily_pnl_rows: List[Dict[str, Any]] = []
        self.trade_rows: List[Dict[str, Any]] = []
        self.roll_portfolio_rows: List[Dict[str, Any]] = []

        # Global registry of all portfolios (for Excel / debugging)
        self.all_ptfs: Dict[int, Dict[str, Any]] = {}
        self._next_ptf_id: int = 1

        self.state: Dict[str, TickerState] = {
            sym: TickerState(
                equity=config["initial_equity_per_ticker"],
                cash=config["initial_equity_per_ticker"],
            )
            for sym in symbols
        }
        self.rolling_entries_log = []

    def run(self):
        all_dates = self._build_global_calendar()
        print(f"[INFO] Backtest calendar: {len(all_dates)} days.")

        for sym in self.symbols:
            self.market.load_symbol_options(sym)

            for date in tqdm(all_dates, desc=f"Processing {sym}", unit="day"):
                if date not in self.market.get_spot_calendar(sym):
                    continue
                self._process_symbol_date(sym, date)

            if sym in self.market.options:
                del self.market.options[sym]

        self._export_results()

    def _build_global_calendar(self) -> pd.DatetimeIndex:
        calendars = [self.market.get_spot_calendar(sym) for sym in self.symbols]
        if not calendars:
            return pd.DatetimeIndex([])
        all_dates = sorted(set().union(*[set(idx) for idx in calendars]))
        return pd.DatetimeIndex(all_dates)

    def get_live_ptfs_between(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return meta rows for all RollingPtf that are live at any point in [start_date, end_date].

        A ptf (entry_date, exit_date) is live on this interval iff:
            entry_date <= end_date AND exit_date >= start_date.
        """
        if start_date > end_date:
            start_date, end_date = end_date, start_date

        rows: List[Dict[str, Any]] = []
        for ptf_id, meta in self.all_ptfs.items():
            if symbol is not None and meta["symbol"] != symbol:
                continue
            e1 = meta["entry_date"]
            e2 = meta["exit_date"]
            if (e1 <= end_date) and (e2 >= start_date):
                rows.append(meta.copy())
        return rows

    def get_live_ptfs_on_date(
        self,
        date: pd.Timestamp,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.get_live_ptfs_between(date, date, symbol)

    def register_new_ptf(
        self,
        symbol: str,
        entry_date: pd.Timestamp,
        exit_date: pd.Timestamp,
        legs: List[Dict[str, Any]],
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Register a new RollingPtf and store it both globally and in TickerState.

        legs: list of dicts with at least:
            - contract_id
            - expiry
            - strike
            - type ("C"/"P")
            - qty
        """
        ptf_id = self._next_ptf_id
        self._next_ptf_id += 1

        contract_ids = [str(leg["contract_id"]) for leg in legs]
        expiries = [pd.to_datetime(leg["expiry"]).normalize() for leg in legs]
        strikes = [float(leg["strike"]) for leg in legs]
        opt_types = [str(leg["type"]).upper()[0] for leg in legs]
        qtys = [float(leg["qty"]) for leg in legs]
        n = len(legs)

        meta = meta or {}

        # Global flat metadata
        meta_row = {
            "ptf_id": ptf_id,
            "symbol": symbol,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "contract_ids": contract_ids,
            "expiries": expiries,
            "strikes": strikes,
            "opt_types": opt_types,
            "qtys": qtys,
            "meta": meta,
        }
        self.all_ptfs[ptf_id] = meta_row

        st = self.state[symbol]
        ptf = RollingPtf(
            ptf_id=ptf_id,
            symbol=symbol,
            entry_date=entry_date,
            exit_date=exit_date,
            contract_ids=contract_ids,
            expiries=expiries,
            strikes=strikes,
            opt_types=opt_types,
            qtys=qtys,
            prev_prices=[0.0] * n,
            prev_ivs=[0.0] * n,
            prev_deltas=[0.0] * n,
            prev_gammas=[0.0] * n,
            prev_vegas=[0.0] * n,
            prev_thetas=[0.0] * n,
            meta=meta,
        )
        st.rolling_ptfs[ptf_id] = ptf
        return ptf_id


    def _apply_split_to_ptfs(self, symbol: str, date: pd.Timestamp) -> None:
        """
        Handle stock splits for:
          - all live RollingPtfs on this (symbol, date)
          - the stock hedge position
          - last_spot (so the split doesn't look like a huge price move)

        Conventions:
          - split_factor = 1.0 on normal days
          - split_factor = k (>1) on split days (e.g. 4.0 for 4-for-1)
        """
        split_factor = self.market.get_split_factor(symbol, date)

        # Nothing to do if no split on this date
        if not np.isfinite(split_factor) or abs(split_factor - 1.0) < 1e-12:
            return

        st = self.state[symbol]

        # ------------------------------
        # 1) Adjust all live option ptfs
        # ------------------------------
        for ptf_id, ptf in st.rolling_ptfs.items():
            # Only ptfs that are actually live on this date
            if not (ptf.entry_date <= date <= ptf.exit_date):
                continue

            if not ptf.qtys:
                continue

            # New strikes: divide by split_factor
            ptf.strikes = [float(s) / split_factor for s in ptf.strikes]
            # New quantities: multiply by split_factor
            ptf.qtys    = [float(q) * split_factor for q in ptf.qtys]

            # Keep global metadata in sync (for Excel / debugging)
            meta = self.all_ptfs.get(ptf_id)
            if meta is not None:
                meta["strikes"] = list(ptf.strikes)
                meta["qtys"]    = list(ptf.qtys)

            # Optional: we could also reset greek caches here to avoid
            # attributing any PnL to the mechanical split in the greek
            # decomposition. Minimal & safe behaviour:
            ptf.prev_deltas = [0.0] * len(ptf.prev_deltas)
            ptf.prev_gammas = [0.0] * len(ptf.prev_gammas)
            ptf.prev_vegas  = [0.0] * len(ptf.prev_vegas)
            ptf.prev_thetas = [0.0] * len(ptf.prev_thetas)
            # prev_prices / prev_ivs are not used in PnL_total, so we
            # can leave them as is or adjust later if needed.

        # ------------------------------
        # 2) Adjust stock hedge position
        # ------------------------------
        # If we were long 10 shares and we get a 4-for-1 split,
        # we must now be long 40 shares at quarter price. That’s how
        # we avoid creating fake PnL from the hedge.
        st.stock_pos_close     *= split_factor
        st.stock_pos_intraday  *= split_factor

        # ------------------------------
        # 3) Adjust last_spot
        # ------------------------------
        # last_spot is used to compute dS = spot_today - last_spot
        # for gamma / delta-hedge PnL. If we don't adjust it, the
        # split shows up as a huge "real" move.
        if st.last_spot is not None and np.isfinite(st.last_spot):
            st.last_spot = st.last_spot / split_factor

    def _apply_delta_hedge(
        self,
        symbol: str,
        date: pd.Timestamp,
        port_delta_today: float,
        spot: float,
    ) -> Dict[str, float]:
        cfg = self.config
        st = self.state[symbol]

        if not cfg.get("delta_hedge", True):
            st.stock_pos_intraday = st.stock_pos_close
            return {"pnl_delta_hedge": 0.0, "pnl_tc_stock": 0.0}

        tgt_stock_pos = -port_delta_today
        trade_shares = tgt_stock_pos - st.stock_pos_close
        pnl_delta_hedge = 0.0
        pnl_tc_stock = 0.0

        cost_model = cfg.get("cost_model", {})
        stock_spread_bps = float(cost_model.get("stock_spread_bps", 0.0))

        if abs(trade_shares) > 1e-10:
            spread = stock_spread_bps / 1e4
            trade_price = spot * (1 + spread) if trade_shares > 0 else spot * (1 - spread)
            cash_change = -trade_shares * trade_price
            st.cash += cash_change

            # TC from spread
            tc_stock = abs(trade_price - spot) * abs(trade_shares)
            pnl_tc_stock += tc_stock

        if st.last_spot is not None:
            dS = spot - st.last_spot
            pnl_delta_hedge = st.stock_pos_close * dS

        st.stock_pos_intraday = tgt_stock_pos
        st.stock_pos_close = tgt_stock_pos

        return {"pnl_delta_hedge": pnl_delta_hedge, "pnl_tc_stock": pnl_tc_stock}

    def _process_symbol_date(self, symbol: str, date: pd.Timestamp):
        st = self.state[symbol]
        cfg = self.config
        equity_prev = st.equity

        # 0) Spot
        spot = self.market.get_spot(symbol, date)
        if spot is None or not np.isfinite(spot):
            return

        # PnL diagnostics
        option_pnl_entry = 0.0
        option_pnl_close = 0.0
        option_pnl_expire = 0.0
        pnl_gamma = 0.0
        pnl_vega = 0.0
        pnl_theta = 0.0
        pnl_delta_hedge = 0.0
        pnl_tc_opt = 0.0
        pnl_tc_stock = 0.0

        # 1) Corporate actions (splits) – rescale strikes / qtys
        self._apply_split_to_ptfs(symbol, date)

        # 2) Compute vega target and let strategy create new PTFs
        mode = cfg.get("strategy_mode", "earnings")
        if mode == "earnings":
            equity_scale = (
                st.equity / cfg["initial_equity_per_ticker"]
                if cfg.get("reinvest", False)
                else 1.0
            )
            base_vega = cfg["base_vega_target"] * equity_scale
            if cfg.get("use_signal", False):
                vega_target = self.strategy.compute_signal_vega(date, symbol, base_vega)
            else:
                vega_target = base_vega
        elif mode == "rolling":
            holding_days = max(1, int(cfg.get("rolling_holding_days", 20)))
            rolling_vega_per_ptf = float(
                cfg.get("rolling_vega_per_lot") or (cfg["base_vega_target"] / holding_days)
            )
            equity_scale = (
                st.equity / cfg["initial_equity_per_ticker"]
                if cfg.get("rolling_reinvest", False)
                else 1.0
            )
            vega_target = rolling_vega_per_ptf * equity_scale
        else:
            vega_target = 0.0

        # Strategy must call backtester.register_new_ptf(.) when it wants new PTFs.
        self.strategy.compute_target_positions(date, symbol, st, self.market, vega_target)

        # 3) Get live portfolios and today's chain
        live_ptfs = [
            ptf for ptf in st.rolling_ptfs.values()
            if ptf.entry_date <= date <= ptf.exit_date
        ]

        chain_today = self.market.get_chain(symbol, date)
        if not chain_today.empty:
            chain_today = chain_today.copy()
            chain_today["cid"] = chain_today["contractID"].astype(str)
            chain_idx = chain_today.set_index("cid")
        else:
            chain_idx = None

        # 4) Entry / Close / Expiry cashflows from portfolios
        cost_model = cfg.get("cost_model", {})
        option_spread_bps = float(cost_model.get("option_spread_bps", 0.0))
        commission_per_contract = float(cost_model.get("commission_per_contract", 0.0))

        def _trade_price(mid: float, qty: float) -> float:
            spread = option_spread_bps / 1e4
            return mid * (1 + spread) if qty > 0 else mid * (1 - spread)

        for ptf in live_ptfs:
            n = len(ptf.contract_ids)
            for i in range(n):
                cid = ptf.contract_ids[i]
                expiry = ptf.expiries[i]
                strike = ptf.strikes[i]
                opt_type = ptf.opt_types[i]
                qty = ptf.qtys[i]

                if qty == 0.0:
                    # Already fully closed/expired
                    continue

                if chain_idx is None or cid not in chain_idx.index:
                    continue

                row_opt = chain_idx.loc[cid]
                mid_price = float(row_opt["mid"])

                # ---- Entry day: open qty ----
                if date == ptf.entry_date:
                    trade_qty = qty
                    trade_price = _trade_price(mid_price, trade_qty)
                    cash_change = -trade_qty * trade_price
                    st.cash += cash_change

                    # TC: spread + commission
                    tc_spread = abs(trade_price - mid_price) * abs(trade_qty)
                    tc_comm = commission_per_contract * abs(trade_qty)
                    pnl_tc_opt += tc_spread + tc_comm
                    st.cash -= tc_comm

                    # For diagnostics: premium received (short) or paid (long)
                    option_pnl_entry += -cash_change

                # ---- Exit day (non-expiry): close qty ----
                if (date == ptf.exit_date) and (expiry > date) and (qty != 0.0):
                    trade_qty = -qty
                    trade_price = _trade_price(mid_price, trade_qty)
                    cash_change = -trade_qty * trade_price
                    st.cash += cash_change

                    tc_spread = abs(trade_price - mid_price) * abs(trade_qty)
                    tc_comm = commission_per_contract * abs(trade_qty)
                    pnl_tc_opt += tc_spread + tc_comm
                    st.cash -= tc_comm

                    option_pnl_close += -cash_change

                    # VERY IMPORTANT: position is now flat
                    ptf.qtys[i] = 0.0

                # ---- Expiry payoff ----
                if (expiry == date) and (ptf.entry_date <= date <= ptf.exit_date) and (ptf.qtys[i] != 0.0):
                    qty_after_trade = ptf.qtys[i]

                    if opt_type == "C":
                        intrinsic = max(spot - strike, 0.0)
                    else:
                        intrinsic = max(strike - spot, 0.0)

                    payoff = intrinsic * qty_after_trade
                    st.cash += payoff
                    option_pnl_expire += payoff

                    # After expiry you no longer hold the option
                    ptf.qtys[i] = 0.0

        # 5) Option MtM and greek exposures (live portfolios, AFTER cashflows)
        mtm_options = 0.0
        port_delta_today = 0.0
        port_gamma_today = 0.0
        port_vega_today = 0.0
        port_theta_today = 0.0

        for ptf in live_ptfs:
            n = len(ptf.contract_ids)
            for i in range(n):
                cid = ptf.contract_ids[i]
                expiry = ptf.expiries[i]
                trade_date = ptf.entry_date
                close_date = ptf.exit_date
                strike = ptf.strikes[i]
                opt_type = ptf.opt_types[i]
                qty = ptf.qtys[i]

                # Legs closed/expired earlier are now qty=0
                if qty == 0.0:
                    continue

                if chain_idx is None or cid not in chain_idx.index:
                    continue

                row_opt = chain_idx.loc[cid]
                mid_price = float(row_opt["mid"])
                iv = float(row_opt.get("implied_volatility", np.nan))
                delta = float(row_opt.get("delta", 0.0))
                gamma = float(row_opt.get("gamma", 0.0))
                vega = float(row_opt.get("vega", 0.0))
                theta = float(row_opt.get("theta", 0.0))

                self.trade_rows.append({
                    "Date": date,
                    "Trade Date": trade_date,
                    "Exit Date": close_date,
                    "Symbol": symbol,
                    "ContractID": cid,
                    "Expiry": expiry,
                    "Strike": strike,
                    "Type": opt_type,
                    "TradeQty": qty,
                    "TradePrice": mid_price,
                    "TradeNotional": qty * mid_price,
                    "Spot": spot,
                    "IV": iv,
                    "Delta": delta,
                    "Gamma": gamma,
                    "Vega": vega,
                    "Theta": theta,
                })

                mtm_options += qty * mid_price
                port_delta_today += qty * delta
                port_gamma_today += qty * gamma
                port_vega_today += qty * vega
                port_theta_today += qty * theta

                # Previous day values for decomposition
                prev_price = ptf.prev_prices[i]
                prev_iv = ptf.prev_ivs[i]
                prev_gamma = ptf.prev_gammas[i]
                prev_vega = ptf.prev_vegas[i]
                prev_theta = ptf.prev_thetas[i]

                dS = 0.0 if st.last_spot is None else (spot - st.last_spot)

                # Gamma PnL
                pnl_gamma += 0.5 * prev_gamma * (dS ** 2) * qty

                # Vega PnL
                if np.isfinite(iv) and np.isfinite(prev_iv):
                    dIV = iv - prev_iv
                    pnl_vega += prev_vega * dIV * qty

                # Theta PnL (1 day)
                pnl_theta += prev_theta * (1.0 / 252.0) * qty

                # Update prev_* for next day
                ptf.prev_prices[i] = mid_price
                ptf.prev_ivs[i] = iv if np.isfinite(iv) else prev_iv
                ptf.prev_deltas[i] = delta
                ptf.prev_gammas[i] = gamma
                ptf.prev_vegas[i] = vega
                ptf.prev_thetas[i] = theta

        st.mtm_options = mtm_options

        # 6) Stock hedge & MTM
        #    IMPORTANT CHANGE vs your original version:
        #    - We DO NOT set st.mtm_stock before hedging.
        #    - We hedge first (which updates stock_pos_close & cash),
        #      then compute MTM based on the new stock_pos_close.
        hedge_res = self._apply_delta_hedge(symbol, date, port_delta_today, spot)
        pnl_delta_hedge = hedge_res["pnl_delta_hedge"]
        pnl_tc_stock = hedge_res["pnl_tc_stock"]

        # Now mark the stock with the *post-hedge* position
        st.mtm_stock = st.stock_pos_close * spot

        # 7) Equity & PnL (source of truth)
        st.equity = st.cash + st.mtm_options + st.mtm_stock
        pnl_total = st.equity - equity_prev

        st.cum_pnl += pnl_total
        st.cum_pnl_gamma += pnl_gamma
        st.cum_pnl_vega += pnl_vega
        st.cum_pnl_theta += pnl_theta
        st.cum_pnl_delta_hedge += pnl_delta_hedge
        st.cum_pnl_tc += (pnl_tc_opt + pnl_tc_stock)

        st.last_spot = spot

        # 8) Daily row (same schema as original backtester)
        self.daily_pnl_rows.append(
            {
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
                "PnL_TC": pnl_tc_opt + pnl_tc_stock,
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
        )
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
            # if not df_trades_all.empty:
            #     df_trades_all.sort_values(["Date", "Symbol"]).to_excel(
            #         writer, sheet_name="ALL_TRADES", index=False
            #     )

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

# If run as a script, execute backtest
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
