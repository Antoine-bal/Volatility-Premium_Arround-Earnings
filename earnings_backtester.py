import os
import time
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import numpy as np
import pandas as pd
from line_profiler import LineProfiler

# Line-profiler hook: if running with kernprof/line_profiler, a real
# `profile` symbol will be injected. Otherwise this is a no-op decorator.
try:
    profile  # type: ignore[name-defined]
except NameError:
    def profile(func):
        return func

# ============================================================
# ===================== CONFIG SECTION =======================
# ============================================================

API_KEY = ""  # only needed if you later add fetching here

# SYMBOLS = [
#     "NVDA","MSFT","AAPL","AMZN","META","AVGO","GOOGL","GOOG","BRK.B","TSLA","JPM","V","LLY","NFLX","XOM","MA","WMT",
#     "COST","ORCL","JNJ","HD","PG","ABBV","BAC","UNH","CRM","ADBE","PYPL","AMD","INTC","CSCO","MCD","NKE","WFC","CVX",
#     "PEP","KO","DIS","BA","MRK","MO","IBM","T","GM","CAT","UPS","DOW","PLTR","TXN","LIN","AMAT"
# ]

SYMBOLS = [
    "NVDA","MSFT","AAPL","META", "AVGO", "GOOGL", "PLTR", "JPM","V",
]

START_DATE = pd.Timestamp("2020-01-01")
END_DATE   = pd.Timestamp("2025-11-16")

# Paths
CORP_DIR    = pathlib.Path("alpha_corp_actions")   # where *_daily_adjusted.parquet live
OPTIONS_DIR = pathlib.Path("alpha_options_raw")    # where <SYM>.parquet live
EARNINGS_CSV = "earnings.csv"                     # must have: symbol, event_day

OUT_PNL_EXCEL = r"C:\Users\antoi\Documents\Antoine\Projets_Python\Trading Vol on Earnings\outputs\earnings_backtest_pnl.xlsx"

# Backtest configuration
BACKTEST_CONFIG = {
    "initial_equity_per_ticker": 100.0,
    "reinvest": True,                     # scale vega target with equity
    "base_vega_target": 1.0 / 10,              # vega when equity == initial
    "multiplier": 1.0,                  # option contract multiplier
    "use_signal": False,                  # placeholder: no signal gating for now
    "cost_model": {
        "option_spread_bps": 50,          # half-spread in bps of price (slippage)
        "stock_spread_bps": 1,           # half-spread on stock
        "commission_per_contract": 0.0,
    },
    "delta_hedge": True,                  # whether we hedge delta for this strategy
    "entry_lag": -1,                      # trade at day -1 vs earnings
    "exit_lag": +1,                       # flat after +1
    "min_moneyness": 0.5,                 # filter weird strikes
    "max_moneyness": 1.5,

    # NEW
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

    # previous greeks/price for PnL attribution
    prev_iv: Optional[float] = None
    prev_delta: Optional[float] = None
    prev_gamma: Optional[float] = None
    prev_vega: Optional[float] = None
    prev_theta: Optional[float] = None
    prev_price: Optional[float] = None


@dataclass
class TickerState:
    equity: float
    cash: float                          # NEW: cash / bank account
    positions: Dict[str, OptionPosition] = field(default_factory=dict)
    stock_pos: float = 0.0
    last_spot: Optional[float] = None

    mtm_options: float = 0.0            # NEW: MTM of options
    mtm_stock: float = 0.0              # NEW: MTM of stock hedge

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
      - earnings events and entry/exit dates
    """
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.spot: Dict[str, pd.DataFrame] = {}
        self.options: Dict[str, pd.DataFrame] = {}
        self.earnings: pd.DataFrame = pd.DataFrame()
        self.entry_exit_map: Dict[pd.Timestamp, Dict[str, Any]] = {}

        self._load_spot()
        self._load_options()
        self._load_earnings()
        self._build_entry_exit_maps()

    # ---------- Spot ----------
    # ---------- Spot ----------
    @profile
    def _load_spot(self):
        """
        Load spot and build a split-only normalization.

        For each symbol:
          - read AlphaVantage daily adjusted file
          - build a cumulative split_level from `split_coefficient`
          - define price_factor[t] = split_level[last] / split_level[t]
          - define spot[t] = close_raw[t] / price_factor[t]

        This gives a spot series that is continuous across splits and
        *does not* adjust for dividends. We also store price_factor per date
        so strikes can be normalized consistently in _load_options.
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

            # Use raw close if available, otherwise fall back to adj_close.
            if "close" in df.columns:
                close_raw = df["close"].astype(float)
            elif "adj_close" in df.columns:
                close_raw = df["adj_close"].astype(float)
            else:
                raise ValueError(f"{sym}: neither 'close' nor 'adj_close' in {path}")

            # Build split-only series: split_coefficient = split ratio on that date,
            # 10 for a 10-for-1 split, 1 otherwise. Missing → 1.
            if "split_coefficient" in df.columns:
                split_raw = df["split_coefficient"].astype(float)
                split_raw = split_raw.replace(0.0, np.nan).fillna(1.0)
            else:
                split_raw = pd.Series(1.0, index=df.index)

            # Ascending cumulative split level: e.g. 1 → 10 → 40 for 10-for-1 then 4-for-1.
            split_level = split_raw.cumprod()

            # Normalize prices so that the LAST date has factor = 1.
            # price_factor[t] = split_level[last] / split_level[t]
            # Then spot_norm[t] = close_raw[t] / price_factor[t].
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
        """
        Load options and attach a split-normalized strike:

          - merge per-symbol `price_factor` on date
          - define strike_eff = strike_raw / price_factor[date]

        Strategy will use strike_eff for moneyness, so that spot/strike
        live in the same split-adjusted universe as `spot`.
        """
        for sym in self.symbols:
            path = OPTIONS_DIR / f"{sym}.parquet"
            if not path.exists():
                print(f"[WARN] Options file missing for {sym}: {path}")
                continue

            df = pd.read_parquet(path)

            # normalize columns
            if "date" not in df.columns or "expiration" not in df.columns:
                print(f"[WARN] Missing date/expiration in options for {sym}, skipping.")
                continue

            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()
            df["strike"] = df["strike"].astype(float)
            df["type"] = df["type"].astype(str).str.upper().str[0]  # "C"/"P"

            # Restrict to backtest window early
            df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()
            if df.empty:
                print(f"[WARN] No option rows for {sym} in backtest window.")
                continue

            # Attach price_factor from spot (split-only normalization).
            spot_df = self.spot.get(sym)
            if spot_df is not None and "price_factor" in spot_df.columns:
                pf = spot_df[["price_factor"]].reset_index()  # index is date
                pf = pf.rename(columns={"date": "date"})
                df = df.merge(pf, on="date", how="left")
            else:
                # Fallback: no split info, assume factor = 1.
                df["price_factor"] = 1.0

            df["price_factor"] = df["price_factor"].fillna(1.0).astype(float)

            # choose mid price
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

            df = df[df["mid"] > 0].copy()

            # Effective (split-normalized) strike: raw strike divided by price_factor.
            # This ensures moneyness = strike_eff / spot uses the same split-adjusted scale.
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
            print(f"[WARN] Earnings file {EARNINGS_CSV} missing; no earnings strategy possible.")
            self.earnings = pd.DataFrame(columns=["symbol","event_day"])
            return
        df = pd.read_csv(EARNINGS_CSV)
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df["event_day"] = pd.to_datetime(df["event_day"]).dt.normalize()
        df = df[df["symbol"].isin(self.symbols)].copy()
        df = df[(df["event_day"] >= START_DATE) & (df["event_day"] <= END_DATE)].copy()
        df = df.drop_duplicates(subset=["symbol","event_day"]).reset_index(drop=True)
        self.earnings = df
        print(f"[INFO] Earnings loaded: {len(df)} events")

    def _build_entry_exit_maps(self):
        """
        For each symbol, build a mapping:
          entry_date -> {"event_day": ..., "exit_date": ...}
        using ticker's own spot calendar.
        """
        for sym in self.symbols:
            cal = self.get_spot_calendar(sym)
            if len(cal) == 0:
                continue
            evs = self.earnings[self.earnings["symbol"] == sym]
            mapping: Dict[pd.Timestamp, Dict[str, Any]] = {}

            for _, row in evs.iterrows():
                ev_day = row["event_day"]
                # entry date: previous trading day
                idx = cal.searchsorted(ev_day)
                if idx == 0:
                    continue
                entry = cal[idx - 1]
                # exit date: next trading day after event
                if idx >= len(cal):
                    continue
                exit_ = cal[idx + 1] if idx + 1 < len(cal) else None
                mapping[entry] = {"event_day": ev_day, "exit_date": exit_}
            self.entry_exit_map[sym] = mapping

    def get_earnings_meta(self, symbol: str, date: pd.Timestamp) -> Optional[Dict[str, Any]]:
        """
        If this date is an entry date for an earnings event, return its metadata.
        """
        return self.entry_exit_map.get(symbol, {}).get(date)


# ============================================================
# ===================== STRATEGY LAYER =======================
# ============================================================

class StrategyEarningsATM:
    """
    Strategy: earnings_short_dh_straddle

    - For each earnings event:
      - On entry_lag trading day before event: short vega_target of ATM front-month straddle
      - Delta-hedge daily if config["delta_hedge"] is True
      - Flatten on exit_lag trading day after event
    """

    def __init__(self, config: dict):
        self.config = config

    def compute_target_positions(self,
                                 date: pd.Timestamp,
                                 symbol: str,
                                 state: TickerState,
                                 market: MarketData,
                                 vega_target: float) -> List[Dict[str, Any]]:
        """
        Returns list of target positions (contractID, qty, etc.) for this date/symbol.
        If no trade, returns empty list (engine keeps current positions and will
        close them on exit date).

        Key point: moneyness is computed using split-normalized strike
        (`strike_eff`) divided by split-normalized spot.
        """
        targets: List[Dict[str, Any]] = []

        # Only trade on entry dates
        meta = market.get_earnings_meta(symbol, date)
        if meta is None:
            return targets

        event_day = meta["event_day"]
        # exit_date = meta["exit_date"]  # engine handles exit

        spot = market.get_spot(symbol, date)
        if spot is None or not np.isfinite(spot):
            return targets

        chain = market.get_chain(symbol, date)
        if chain.empty:
            return targets

        cfg = self.config
        min_mny = cfg["min_moneyness"]
        max_mny = cfg["max_moneyness"]
        min_dte = cfg["min_dte_for_entry"]
        max_dte = cfg["max_dte_for_entry"]

        chain = chain.copy()
        chain["dte"] = (chain["expiration"] - date).dt.days

        # Use split-normalized strike if available; fallback to raw strike.
        strike_col = "strike_eff" if "strike_eff" in chain.columns else "strike"
        chain["moneyness"] = chain[strike_col] / spot

        # Filter by moneyness
        chain = chain[(chain["moneyness"] >= min_mny) &
                      (chain["moneyness"] <= max_mny)].copy()
        if chain.empty:
            return targets

        # Keep expiries strictly after the earnings event
        after_ev = chain[chain["expiration"] > event_day].copy()
        if after_ev.empty:
            return targets

        # Prefer expiries within [min_dte, max_dte]
        eligible = after_ev[(after_ev["dte"] >= min_dte) &
                            (after_ev["dte"] <= max_dte)].copy()

        # If none in that window, fallback = closest expiry after event (smallest DTE)
        if eligible.empty:
            eligible = after_ev.sort_values("dte").copy()
        else:
            eligible = eligible.copy()

        # Sort by (DTE, |moneyness-1|) to get closest expiry then closest ATM
        eligible["abs_mny"] = (eligible["moneyness"] - 1.0).abs()
        eligible = eligible.sort_values(["dte", "abs_mny"])

        # Find first expiry/strike with both call and put
        straddle_contracts = None
        for _, opt_row in eligible.iterrows():
            expiry = opt_row["expiration"]
            strike_raw = opt_row["strike"]  # use raw strike for contract ID matching
            sub = chain[(chain["expiration"] == expiry) &
                        (chain["strike"] == strike_raw)]
            calls = sub[sub["type"] == "C"]
            puts  = sub[sub["type"] == "P"]
            if calls.empty or puts.empty:
                continue
            call = calls.iloc[0]
            put  = puts.iloc[0]
            straddle_contracts = (call, put)
            break

        if straddle_contracts is None:
            return targets

        call, put = straddle_contracts

        # --- Sizing by vega target ---
        call_vega = float(call["vega"])
        put_vega  = float(put["vega"])
        tot_vega  = abs(call_vega) + abs(put_vega)
        if tot_vega <= 1e-8:
            return targets

        multiplier = cfg["multiplier"]
        vega_per_call = call_vega * multiplier
        vega_per_put  = put_vega * multiplier
        if abs(vega_per_call) <= 1e-8 or abs(vega_per_put) <= 1e-8:
            return targets

        # Target total vega short = -vega_target split evenly between call and put
        target_vega_each = -0.5 * vega_target
        qty_call = target_vega_each / vega_per_call
        qty_put  = target_vega_each / vega_per_put

        targets.append({
            "contract_id": str(call["contractID"]),
            "symbol": symbol,
            "expiry": call["expiration"],
            "strike": float(call["strike"]),  # raw strike for reporting
            "type": "C",
            "qty": qty_call,
        })
        targets.append({
            "contract_id": str(put["contractID"]),
            "symbol": symbol,
            "expiry": put["expiration"],
            "strike": float(put["strike"]),
            "type": "P",
            "qty": qty_put,
        })

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
        """
        Main loop: for each date in global calendar, for each symbol:
          - build state (spot, chain, earnings meta)
          - compute vega_target given equity
          - ask strategy for target positions
          - engine trades from current positions to target
          - compute P&L decomposition
        """
        all_dates = self._build_global_calendar()
        print(f"[INFO] Backtest calendar: {len(all_dates)} days.")

        for sym in self.symbols:
            for date in tqdm(all_dates, desc=f"Processing {sym}", unit="day"):
                if date not in self.market.get_spot_calendar(sym):
                    # no spot, skip
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
        equity_prev = st.equity
        cfg = self.config

        spot = self.market.get_spot(symbol, date)
        if spot is None or not np.isfinite(spot):
            return

        # compute vega target based on equity and reinvest flag
        if cfg["reinvest"]:
            equity_scale = st.equity / cfg["initial_equity_per_ticker"]
        else:
            equity_scale = 1.0
        vega_target = cfg["base_vega_target"] * equity_scale

        # Check if today is an exit date for an event -> we want to be flat
        meta = self.market.get_earnings_meta(symbol, date)
        if meta is not None:
            exit_date = meta["exit_date"]
        else:
            exit_date = None

        # Strategy's target positions on *entry* dates.
        # On non-entry days, it returns empty -> we keep existing positions.
        target_lines = self.strategy.compute_target_positions(
            date, symbol, st, self.market, vega_target
        )

        # If today is an exit date for any event, we want to be flat.
        need_flat = False
        for entry, meta_ev in self.market.entry_exit_map.get(symbol, {}).items():
            if meta_ev["exit_date"] == date:
                need_flat = True
                break
        if need_flat:
            target_lines = []  # target zero on all contracts

        # --- Engine: from current positions + target -> trades ---
        chain_today = self.market.get_chain(symbol, date)
        if chain_today.empty and st.positions:
            # Cannot price, skip PnL / trading for today
            return

        if not chain_today.empty:
            # Build a fast index by *string* contract id (matches your positions dict keys)
            chain_today = chain_today.copy()
            chain_today["cid"] = chain_today["contractID"].astype(str)
            chain_today_idx = chain_today.set_index("cid")
        else:
            chain_today_idx = None

        # First, mark previous positions as prev_* for PnL decomposition
        for pos in st.positions.values():
            pos.prev_iv = pos.iv
            pos.prev_delta = pos.delta
            pos.prev_gamma = pos.gamma
            pos.prev_vega = pos.vega
            pos.prev_theta = pos.theta
            pos.prev_price = pos.last_price

        # Determine target qty per contractID
        target_qty: Dict[str, float] = {}
        for t in target_lines:
            cid = t["contract_id"]
            target_qty[cid] = target_qty.get(cid, 0.0) + float(t["qty"])

        # For contracts not in target, if we want flat (need_flat), target qty = 0
        if need_flat:
            for cid in list(st.positions.keys()):
                if cid not in target_qty:
                    target_qty[cid] = 0.0

        # Execute trades: adjust positions to match target_qty
        cost_model = cfg["cost_model"]
        option_spread_bps = cost_model["option_spread_bps"]
        commission_per_contract = cost_model["commission_per_contract"]

        pnl_tc_options = 0.0

        # Apply target on each contractID in target_qty ∪ existing positions
        contract_ids = set(st.positions.keys()) | set(target_qty.keys())
        for cid in contract_ids:
            cur_pos = st.positions.get(cid)
            tgt_qty = target_qty.get(cid, cur_pos.qty if cur_pos else 0.0)

            if chain_today_idx is None:
                # no chain data -> cannot trade / mark
                continue

            try:
                row = chain_today_idx.loc[cid]
            except KeyError:
                # contract not listed today -> skip
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

            if cur_pos is None:
                cur_qty = 0.0
            else:
                cur_qty = cur_pos.qty

            trade_qty = tgt_qty - cur_qty
            if abs(trade_qty) > 1e-10:
                spread = option_spread_bps / 1e4
                if trade_qty > 0:
                    trade_price = mid_price * (1 + spread)  # buy at ask
                else:
                    trade_price = mid_price * (1 - spread)  # sell at bid

                # Cash from trade (excluding commission)
                cash_change = -trade_qty * trade_price * cfg["multiplier"]

                # Commission cost
                commission = abs(trade_qty) * commission_per_contract

                # Cash moves (including commission), equity will be recomputed as cash + MTM
                st.cash += cash_change - commission
                pnl_tc_options += commission

                # Log trade with greeks + IV
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

            # ---- position update (IN PLACE) ----
            if abs(tgt_qty) <= 1e-10:
                if cid in st.positions:
                    del st.positions[cid]
            else:
                if cur_pos is None:
                    # new position: prev_* stay None (no PnL decomposition on first day)
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
                        multiplier=cfg["multiplier"]
                    )
                else:
                    # update existing object, preserving prev_* that we set earlier
                    cur_pos.qty = tgt_qty
                    cur_pos.last_price = mid_price
                    cur_pos.iv = iv
                    cur_pos.delta = delta
                    cur_pos.gamma = gamma
                    cur_pos.vega = vega
                    cur_pos.theta = theta

        # --- Delta hedge (simple daily hedge after options) ---
        pnl_delta_hedge = 0.0
        if cfg["delta_hedge"]:
            # Compute portfolio delta using today's greeks & qty.
            # If there are no positions, this is 0 and we will flatten stock_pos to 0.
            port_delta = 0.0
            for pos in st.positions.values():
                port_delta += pos.delta * pos.qty * pos.multiplier

            # Desired stock position is opposite of option delta
            tgt_stock_pos = -port_delta
            trade_shares = tgt_stock_pos - st.stock_pos

            if abs(trade_shares) > 1e-8:
                spread = cost_model["stock_spread_bps"] / 1e4
                if trade_shares > 0:
                    # buying shares (to increase stock_pos)
                    trade_price = spot * (1 + spread)
                else:
                    # selling shares (to reduce stock_pos)
                    trade_price = spot * (1 - spread)

                cash_change = -trade_shares * trade_price
                st.cash += cash_change
                # slippage is embedded in cash_change; not split out

            # Delta-hedging PnL from holding the stock between days:
            if st.last_spot is not None:
                dS = spot - st.last_spot
                # Use yesterday's stock_pos (before updating to tgt_stock_pos)
                pnl_delta_hedge = st.stock_pos * dS

            # Update stock position for next day
            st.stock_pos = tgt_stock_pos
        else:
            # No active hedging, but still compute PnL from any existing stock_pos
            if st.last_spot is not None:
                dS = spot - st.last_spot
                pnl_delta_hedge = st.stock_pos * dS

        # --- PnL decomposition using greeks ---
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
            if (pos.prev_iv is not None) and np.isfinite(pos.prev_iv) and np.isfinite(pos.iv):
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

        # --- Recompute MTM and equity at end-of-day ---
        mtm_options = 0.0
        for pos in st.positions.values():
            mtm_options += pos.last_price * pos.qty * pos.multiplier
        mtm_stock = st.stock_pos * spot

        st.mtm_options = mtm_options
        st.mtm_stock = mtm_stock

        st.equity = st.cash + mtm_options + mtm_stock

        # True economic PnL of the day
        pnl_total = st.equity - equity_prev

        st.cum_pnl += pnl_total
        st.cum_pnl_vega += pnl_vega
        st.cum_pnl_gamma += pnl_gamma
        st.cum_pnl_theta += pnl_theta
        st.cum_pnl_delta_hedge += pnl_delta_hedge
        st.cum_pnl_tc += pnl_tc_options

        # Save daily row
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

        # update last_spot
        st.last_spot = spot
    # ---------- Export ----------
    def _export_results(self):
        if not self.daily_pnl_rows:
            print("[WARN] No PnL rows to export.")
            return

        df = pd.DataFrame(self.daily_pnl_rows)
        df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)

        # Build portfolio equity curve
        pivot_equity = df.pivot(index="Date", columns="Symbol", values="Equity")
        pivot_equity["PortfolioEquity"] = pivot_equity.sum(axis=1)

        # Build config table
        config_rows = _flatten_config_dict(self.config)
        df_config = pd.DataFrame(config_rows, columns=["Key", "Value"])

        # Build earnings table (all earnings in period for these symbols)
        df_earn = self.market.earnings.copy()
        if df_earn.empty:
            df_earn = pd.DataFrame(columns=["Symbol", "EventDay"])
        else:
            df_earn = df_earn.sort_values(["symbol", "event_day"]).copy()
            df_earn = df_earn.rename(
                columns={
                    "symbol": "Symbol",
                    "event_day": "EventDay",
                }
            )

            # ---------- NEW: Corporate actions table ----------
        corp_rows = []
        for sym in self.symbols:
            path = CORP_DIR / f"{sym}_daily_adjusted.parquet"
            if not path.exists():
                continue

            df_c = pd.read_parquet(path)
            if "date" not in df_c.columns:
                continue

            df_c["date"] = pd.to_datetime(df_c["date"]).dt.normalize()
            df_c = df_c[(df_c["date"] >= START_DATE) & (df_c["date"] <= END_DATE)].copy()

            has_div = "dividend_amount" in df_c.columns
            has_split = "split_coefficient" in df_c.columns

            if not has_div and not has_split:
                continue

            mask = False
            if has_div:
                mask = mask | (df_c["dividend_amount"].fillna(0) != 0)
            if has_split:
                mask = mask | (df_c["split_coefficient"].fillna(1) != 1)

            df_c = df_c[mask]
            if df_c.empty:
                continue

            cols = ["date"]
            if has_div:
                cols.append("dividend_amount")
            if has_split:
                cols.append("split_coefficient")

            df_c = df_c[cols].copy()
            df_c["Symbol"] = sym
            corp_rows.append(df_c)

        if corp_rows:
            df_corp = pd.concat(corp_rows, ignore_index=True)
            df_corp = df_corp.rename(
                columns={
                    "date": "Date",
                    "dividend_amount": "Dividend",
                    "split_coefficient": "SplitFactor",
                }
            )
            df_corp = df_corp[["Symbol", "Date"] +
                              [c for c in ["Dividend", "SplitFactor"] if c in df_corp.columns]]
            df_corp = df_corp.sort_values(["Symbol", "Date"])
        else:
            df_corp = pd.DataFrame(columns=["Symbol", "Date", "Dividend", "SplitFactor"])
        # ---------------------------------------------------

        # Build trades table (full universe)
        df_trades_all = pd.DataFrame(self.trade_rows) if self.trade_rows else pd.DataFrame(
            columns=[
                "Date", "Symbol", "ContractID", "Expiry", "Strike",
                "Type", "TradeQty", "TradePrice", "TradeNotional", "Spot",
                "IV", "Delta", "Gamma", "Vega", "Theta",
            ]
        )

        with pd.ExcelWriter(OUT_PNL_EXCEL, engine="xlsxwriter") as writer:
            # Sheet 1: Portfolio + per-ticker equity
            out_portfolio = pivot_equity.reset_index()
            out_portfolio.to_excel(writer, sheet_name="PORTFOLIO", index=False)

            # Sheet 2: CONFIG (flattened BACKTEST_CONFIG)
            df_config.to_excel(writer, sheet_name="CONFIG", index=False)

            # Sheet 3: EARNINGS (all earnings in the period)
            df_earn.to_excel(writer, sheet_name="EARNINGS", index=False)

            # Sheet 4: CORP_ACTIONS (corporate actions in the period)
            df_corp.to_excel(writer, sheet_name="CORP_ACTIONS", index=False)

            # Optional global trades sheet
            if not df_trades_all.empty:
                df_trades_all.sort_values(["Date", "Symbol"]).to_excel(
                    writer, sheet_name="ALL_TRADES", index=False
                )

            # One pair of sheets per ticker:
            #   - <SYM>         : full backtest timeseries
            #   - <SYM>_TRADES  : trades only
            for sym in self.symbols:
                df_sym = df[df["Symbol"] == sym].copy()
                if not df_sym.empty:
                    sheet = sym[:31]
                    df_sym.to_excel(writer, sheet_name=sheet, index=False)

                if not df_trades_all.empty:
                    df_tr_sym = df_trades_all[df_trades_all["Symbol"] == sym].copy()
                    if not df_tr_sym.empty:
                        sheet_tr = f"{sym}_TRADES"
                        sheet_tr = sheet_tr[:31]  # Excel sheet name limit
                        df_tr_sym.to_excel(writer, sheet_name=sheet_tr, index=False)

        print(f"[INFO] Backtest results written to {OUT_PNL_EXCEL}")
# ============================================================
# ============================= MAIN ==========================
# ============================================================

if __name__ == "__main__":
    bt = Backtester(BACKTEST_CONFIG, SYMBOLS)
    # lp = LineProfiler()
    # lp.add_function(Backtester._process_symbol_date)
    # lp.add_function(StrategyEarningsATM.compute_target_positions)
    #
    # lp_wrapper = lp(bt.run)
    # lp_wrapper()
    #
    # lp.print_stats()
    bt.run()

