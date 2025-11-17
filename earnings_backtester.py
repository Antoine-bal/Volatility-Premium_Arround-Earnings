import os
import time
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import numpy as np
import pandas as pd

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
    "NVDA","MSFT","AAPL","META", "AVGO", "GOOGL"
]

START_DATE = pd.Timestamp("2023-01-01")
END_DATE   = pd.Timestamp("2025-11-16")

# Paths
CORP_DIR    = pathlib.Path("alpha_corp_actions")   # where *_daily_adjusted.parquet live
OPTIONS_DIR = pathlib.Path("alpha_options_raw")    # where <SYM>.parquet live
EARNINGS_CSV = "earnings.csv"                     # must have: symbol, event_day

OUT_PNL_EXCEL = "earnings_backtest_pnl.xlsx"

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
    "front_dte_min": 1,                   # at least 1 day to expiry
    "front_dte_max": 30,                  # front expiry <= 30D
    "back_dte_min": 60,                   # for calendar, not used in v1
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
      - daily adjusted spot
      - full options chain per (symbol, date)
      - earnings events and entry/exit dates
    """
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.spot: Dict[str, pd.DataFrame] = {}
        self.options: Dict[str, pd.DataFrame] = {}
        self.earnings: pd.DataFrame = pd.DataFrame()
        self.entry_exit_map: Dict[str, Dict[pd.Timestamp, Dict[str, Any]]] = {}

        self._load_spot()
        self._load_options()
        self._load_earnings()
        self._build_entry_exit_maps()

    # ---------- Spot ----------
    def _load_spot(self):
        for sym in self.symbols:
            path = CORP_DIR / f"{sym}_daily_adjusted.parquet"
            if not path.exists():
                print(f"[WARN] Spot file missing for {sym}: {path}")
                continue
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()
            df = df.set_index("date").sort_index()
            self.spot[sym] = df[["adj_close"]].rename(columns={"adj_close": "spot"})
            print(f"[INFO] Spot loaded for {sym}: {self.spot[sym].shape[0]} days")

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
    def _load_options(self):
        for sym in self.symbols:
            path = OPTIONS_DIR / f"{sym}.parquet"
            if not path.exists():
                print(f"[WARN] Options file missing for {sym}: {path}")
                continue
            df = pd.read_parquet(
                path
            )
            # normalize columns
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()
            df["strike"] = df["strike"].astype(float)
            df["type"] = df["type"].astype(str).str.upper().str[0]  # "C"/"P"
            # choose mid price
            if "mark" in df.columns:
                df["mid"] = df["mark"].astype(float)
            else:
                bid = df.get("bid", np.nan).astype(float)
                ask = df.get("ask", np.nan).astype(float)
                df["mid"] = np.where(
                    np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > 0),
                    0.5 * (bid + ask),
                    df.get("last", np.nan).astype(float)
                )
            df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()
            df = df[df["mid"] > 0].copy()

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
        If no trade, returns empty list (engine keeps current positions but will later
        close them when we hit exit date).
        """
        # Default: keep existing positions unless explicit flat signal around exit.
        targets: List[Dict[str, Any]] = []

        meta = market.get_earnings_meta(symbol, date)
        if meta is None:
            # If we are not on an entry date, and there are existing earnings
            # positions, we keep them until exit date; actual flattening will be
            # handled when date == exit_date of that event.
            return targets

        event_day = meta["event_day"]
        exit_date = meta["exit_date"]

        spot = market.get_spot(symbol, date)
        if spot is None or not np.isfinite(spot):
            return targets

        chain = market.get_chain(symbol, date)
        if chain.empty:
            return targets

        # Filter expiries that are after event_day and within DTE bounds
        cfg = self.config
        min_mny, max_mny = cfg["min_moneyness"], cfg["max_moneyness"]
        front_min, front_max = cfg["front_dte_min"], cfg["front_dte_max"]

        chain = chain.copy()
        chain["dte"] = (chain["expiration"] - date).dt.days
        chain = chain[chain["dte"] >= 1]

        # Filter by moneyness
        chain["moneyness"] = chain["strike"] / spot
        chain = chain[(chain["moneyness"] >= min_mny) &
                      (chain["moneyness"] <= max_mny)].copy()

        if chain.empty:
            return targets

        # Front expiry: min DTE >= front_min and <= front_max and > event_day
        after_ev = chain[chain["expiration"] > event_day]
        if after_ev.empty:
            return targets
        cand_front = after_ev[(after_ev["dte"] >= front_min) &
                              (after_ev["dte"] <= front_max)].copy()
        if cand_front.empty:
            # fallback: nearest expiry after event_day
            cand_front = after_ev.copy()

        # Pick ATM strike by abs(moneyness-1)
        cand_front["abs_mny"] = (cand_front["moneyness"] - 1.0).abs()
        # For straddle, we need both call and put at that strike & expiry
        # Start from nearest ATM candidate
        cand_front = cand_front.sort_values(["expiration", "abs_mny"])

        straddle_contracts = None
        for _, opt_row in cand_front.iterrows():
            expiry = opt_row["expiration"]
            strike = opt_row["strike"]
            sub = chain[(chain["expiration"] == expiry) &
                        (chain["strike"] == strike)]
            calls = sub[sub["type"] == "C"]
            puts  = sub[sub["type"] == "P"]
            if calls.empty or puts.empty:
                continue
            # choose first call/put
            call = calls.iloc[0]
            put  = puts.iloc[0]
            straddle_contracts = (call, put)
            break

        if straddle_contracts is None:
            return targets

        call, put = straddle_contracts

        # Size: short vega_target in total (call + put).
        # We use current greeks; if vega is degenerate, skip.
        call_vega = float(call["vega"])
        put_vega  = float(put["vega"])
        tot_vega  = abs(call_vega) + abs(put_vega)
        if tot_vega <= 1e-8:
            return targets

        # Target total vega short (negative) per straddle:
        # we split evenly across call and put.
        # So target_vega_call = -0.5 * vega_target; same for put.
        # Quantity = target_vega / vega_per_contract.
        vega_per_call = call_vega * self.config["multiplier"]
        vega_per_put  = put_vega * self.config["multiplier"]

        if abs(vega_per_call) <= 1e-8 or abs(vega_per_put) <= 1e-8:
            return targets

        target_vega_each = -0.5 * vega_target
        qty_call = target_vega_each / vega_per_call
        qty_put  = target_vega_each / vega_per_put

        # Build target lines (we want these positions in portfolio)
        targets.append({
            "contract_id": str(call["contractID"]),
            "symbol": symbol,
            "expiry": call["expiration"],
            "strike": float(call["strike"]),
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

        # Note: engine will handle closing on exit_date by setting target=0 then.
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

        # Build a dict for today's prices/greeks by contract_id
        todays = {
            str(row["contractID"]): row
            for _, row in chain_today.iterrows()
        }

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

            row = todays.get(cid)
            if row is None:
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
                    trade_price = mid_price * (1 + spread)
                    cash_change = -trade_qty * trade_price * cfg["multiplier"]
                else:
                    trade_price = mid_price * (1 - spread)
                    cash_change = -trade_qty * trade_price * cfg["multiplier"]

                # Cash moves, equity will be recomputed as cash + MTM
                st.cash += cash_change
                pnl_tc_options += abs(trade_qty) * commission_per_contract

                # ---- LOG TRADE WITH GREEKS + IV ----
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
                # slippage is embedded in cash_change; we do not split it out here

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
        # Option PnL from price changes
        option_pnl = 0.0
        port_gamma_prev = 0.0
        port_vega_prev = 0.0
        port_theta_prev = 0.0
        dIV_weighted_num = 0.0
        dIV_weighted_den = 0.0

        port_delta_today = 0.0
        port_gamma_today = 0.0
        port_vega_today  = 0.0
        port_theta_today = 0.0

        for pos in st.positions.values():
            # Need prev_* to compute gamma/vega/theta PnL; if missing, skip
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
            port_vega_today  += pos.vega  * pos.qty * pos.multiplier
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
        pnl_vega  = port_vega_prev * dIV_eff
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

            # PnL
            "DailyPnL": pnl_total,
            "OptionPnL": option_pnl,          # decomposition only
            "PnL_gamma": pnl_gamma,
            "PnL_theta": pnl_theta,
            "PnL_deltaHedge": pnl_delta_hedge,
            "PnL_TC": pnl_tc_options,

            # Cumulative PnL
            "CumPnL": st.cum_pnl,
            "CumPnL_gamma": st.cum_pnl_gamma,
            "CumPnL_vega": st.cum_pnl_vega,
            "CumPnL_theta": st.cum_pnl_theta,
            "CumPnL_deltaHedge": st.cum_pnl_delta_hedge,
            "CumPnL_TC": st.cum_pnl_tc,

            # Portfolio greeks (options)
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

        # Build trades table (full universe)
        df_trades_all = pd.DataFrame(self.trade_rows) if self.trade_rows else pd.DataFrame(
            columns=[
                "Date", "Symbol", "ContractID", "Expiry", "Strike",
                "Type", "TradeQty", "TradePrice", "TradeNotional", "Spot"
            ]
        )

        with pd.ExcelWriter(OUT_PNL_EXCEL, engine="xlsxwriter") as writer:
            # Sheet 1: Portfolio + per-ticker equity
            out_portfolio = pivot_equity.reset_index()
            out_portfolio.to_excel(writer, sheet_name="PORTFOLIO", index=False)

            # Sheet 2: CONFIG (flattened BACKTEST_CONFIG)
            df_config.to_excel(writer, sheet_name="CONFIG", index=False)

            # Optional global trades sheet (if you also want it aggregated)
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
    bt.run()

