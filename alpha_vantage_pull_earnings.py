import time, requests, pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
ALPHA_KEY = os.getenv('ALPHA_KEY')
OUT_CSV = "earnings.csv"
MIN_GAP_DAYS = 25
FROM, TO = "2019-01-01", "2025-11-10"  # adjust
UNIVERSE = [
    "NVDA","MSFT","AAPL","AMZN","META","AVGO","GOOGL","GOOG","BRK-B","TSLA","JPM","V","LLY","NFLX","XOM","MA","WMT",
    "COST","ORCL","JNJ","HD","PG","ABBV","BAC","UNH","CRM","ADBE","PYPL","AMD","INTC","CSCO","MCD","NKE","WFC","CVX",
    "PEP","KO","DIS","BA","MRK","MO","IBM","T","GM","CAT","UPS","DOW","PLTR","TXN","LIN","AMAT"
]


def map_timing(x):
    """
    Map Alpha Vantage 'reportTime' to:
    - timing: 'BMO', 'AMC', 'DURING', 'UNKNOWN'
    - a synthetic HH:MM:SS timestamp for earnings_datetime
    """
    s = (str(x) if x is not None else "").lower()

    # pre-market earnings: before cash open -> BMO
    if "pre" in s:
        return "BMO", "08:00:00"

    # post-market earnings: after cash close -> AMC
    if "post" in s:
        return "AMC", "16:10:00"

    # intraday releases
    if "during" in s or "market hours" in s or "intraday" in s:
        return "DURING", "12:00:00"

    # truly unknown / missing
    return "UNKNOWN", "12:00:00"

def _to_float(v):
    if v in (None, "None", ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def pull_alpha(symbol, from_date, to_date=None):
    """
    Fetch quarterly earnings from Alpha Vantage for one symbol.
    Filter by [from_date, to_date] on reportedDate.
    Return a list of dicts ready to feed into a DataFrame.
    """
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=EARNINGS&symbol={symbol}&apikey={ALPHA_KEY}"
    )
    r = requests.get(url, timeout=30)
    try:
        js = r.json()
    except Exception as e:
        print(symbol, "JSON decode error:", e)
        return []

    if not isinstance(js, dict):
        print(symbol, "unexpected payload type:", type(js), str(js)[:200])
        return []

    q = js.get("quarterlyEarnings") or []
    out = []
    for ev in q:
        d = ev.get("reportedDate")  # event date
        if not d:
            continue

        # date window filter
        if d < from_date:
            continue
        if to_date is not None and d > to_date:
            continue

        # timing field: NOTE: correct key is 'reportTime'
        t = ev.get("reportTime")
        timing, hhmm = map_timing(t)

        # Build full timestamp
        dt = pd.Timestamp(f"{d} {hhmm}")

        out.append(
            {
                "symbol": symbol,
                "event_day": pd.to_datetime(d),        # pure date
                "earnings_datetime": dt,               # with assumed time
                "timing": timing,                      # BMO / AMC / DURING / UNKNOWN
                "reportTime_raw": t,                   # raw text from AV
                "fiscalDateEnding": ev.get("fiscalDateEnding"),

                "reportedEPS": _to_float(ev.get("reportedEPS")),
                "estimatedEPS": _to_float(ev.get("estimatedEPS")),
                "surprise": _to_float(ev.get("surprise")),
                "surprisePercent": _to_float(ev.get("surprisePercentage")),
            }
        )

    return out
# ==== build earnings df for your UNIVERSE ====

FROM, TO = "2019-01-01", "2025-11-10"

rows = []
for sym in UNIVERSE:
    print(f"[EARNINGS] Fetching {sym} ...", end=" ", flush=True)
    evs = pull_alpha(sym, FROM, TO)
    print(f"{len(evs)} rows")
    rows.extend(evs)
    time.sleep(0.25)  # respect rate limits

earn = pd.DataFrame(rows)
if earn.empty:
    raise RuntimeError(
        "Alpha Vantage returned no earnings. "
        "Check ALPHA_KEY, FROM/TO dates, or rate limits."
    )

# De-duplicate on (symbol, event_day) just in case
earn = (
    earn.drop_duplicates(subset=["symbol", "event_day"])
        .sort_values(["symbol", "event_day"])
        .reset_index(drop=True)
)

print("Rows per ticker:")
print(earn.groupby("symbol").size())

earn.to_csv(OUT_CSV, index=False)
print(f"Saved {OUT_CSV} with {len(earn)} rows")