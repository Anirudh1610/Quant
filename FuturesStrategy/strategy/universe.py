"""
universe.py
-----------
NSE F&O universe: lot sizes, margin rates, sector mapping, ticker lookups.

Lot sizes are revised periodically by NSE.  Values here reflect the 2024 revision.
Check https://www.nseindia.com/products-services/equity-derivatives for updates.

NSE futures expiry
------------------
All monthly futures expire on the LAST THURSDAY of the contract month.
If that Thursday is a market holiday, expiry shifts to the preceding Wednesday.
"""

from __future__ import annotations


# ──────────────────────────────────────────────────────────────────────────────
# LOT SIZES  (shares per futures contract, 2024 revision)
# ──────────────────────────────────────────────────────────────────────────────

LOT_SIZES: dict[str, int] = {
    # ── Index futures ──────────────────────────────────────────────────────
    "NIFTY":        75,
    "BANKNIFTY":    30,
    "FINNIFTY":     65,
    "MIDCPNIFTY":   75,

    # ── Stock futures (liquid F&O names) ──────────────────────────────────
    "RELIANCE":    250,
    "HDFCBANK":    550,
    "ICICIBANK":   700,
    "INFY":        300,
    "TCS":         150,
    "SBIN":       1500,
    "AXISBANK":   1200,
    "WIPRO":      1500,
    "HINDUNILVR":  300,
    "BAJFINANCE":  125,
    "KOTAKBANK":   400,
    "LT":          275,
    "MARUTI":      100,
    "TATAMOTORS": 1425,
    "TATASTEEL":  5000,
    "HCLTECH":     350,
    "SUNPHARMA":   350,
    "TITAN":       375,
    "BHARTIARTL":  475,
    "ADANIENT":    250,
    "NTPC":       5250,
    "POWERGRID":  4700,
    "ONGC":       3850,
    "COALINDIA":  4200,
    "JSWSTEEL":    675,
    "HINDALCO":   2150,
    "DRREDDY":     125,
    "CIPLA":       650,
    "DIVISLAB":    200,
    "TECHM":       600,
    "BAJAJ-AUTO":  250,
    "GRASIM":      475,
    "ULTRACEMCO":  100,
    "ASIANPAINT":  200,
    "NESTLEIND":    40,
    "EICHERMOT":   175,
    "HEROMOTOCO":  300,
    "M&M":         700,
    "TATACONSUM":  875,
    "INDUSINDBK":  500,
    "BPCL":       3000,
    "APOLLOHOSP":  125,
    "SBILIFE":     750,
    "HDFCLIFE":   1100,
    "BAJAJFINSV":  125,
}

# ──────────────────────────────────────────────────────────────────────────────
# SPAN MARGIN RATES  (approximate fraction of contract value, updated daily by NSE)
# ──────────────────────────────────────────────────────────────────────────────

INDEX_MARGIN_RATES: dict[str, float] = {
    "NIFTY":      0.10,   # ~10%
    "BANKNIFTY":  0.12,   # ~12%
    "FINNIFTY":   0.12,
    "MIDCPNIFTY": 0.12,
}

# Most stock futures carry ~20% SPAN + exposure margin
DEFAULT_STOCK_MARGIN_RATE: float = 0.20

# Highly volatile stocks may have higher margins — override here if needed
CUSTOM_MARGIN_RATES: dict[str, float] = {
    "ADANIENT":  0.25,
    "TATASTEEL": 0.22,
    "HINDALCO":  0.22,
    "JSWSTEEL":  0.22,
}


def get_margin_rate(ticker: str) -> float:
    """Return the SPAN margin rate for a futures contract."""
    if ticker in INDEX_MARGIN_RATES:
        return INDEX_MARGIN_RATES[ticker]
    return CUSTOM_MARGIN_RATES.get(ticker, DEFAULT_STOCK_MARGIN_RATE)


# ──────────────────────────────────────────────────────────────────────────────
# YAHOO FINANCE TICKER MAPPING
# Index tickers → yfinance symbol for downloading spot OHLCV
# Stock tickers → append ".NS"
# ──────────────────────────────────────────────────────────────────────────────

INDEX_YF_MAP: dict[str, str] = {
    "NIFTY":       "^NSEI",
    "BANKNIFTY":   "^NSEBANK",
    "MIDCPNIFTY":  "^NSMIDCP",
}

INDIA_VIX_TICKER = "^INDIAVIX"


def to_yf_ticker(futures_ticker: str) -> str:
    """Convert an NSE futures ticker to its Yahoo Finance spot equivalent."""
    if futures_ticker in INDEX_YF_MAP:
        return INDEX_YF_MAP[futures_ticker]
    return futures_ticker + ".NS"


# ──────────────────────────────────────────────────────────────────────────────
# TRADING UNIVERSES
# ──────────────────────────────────────────────────────────────────────────────

INDEX_FUTURES: list[str] = ["NIFTY", "BANKNIFTY"]

# Liquid stock futures with tight bid-ask spreads and high OI
STOCK_FUTURES_UNIVERSE: list[str] = [
    "RELIANCE",  "HDFCBANK",  "ICICIBANK", "INFY",      "TCS",
    "SBIN",      "AXISBANK",  "WIPRO",     "BAJFINANCE","KOTAKBANK",
    "LT",        "MARUTI",    "TATAMOTORS","HCLTECH",   "SUNPHARMA",
    "TITAN",     "BHARTIARTL","ADANIENT",  "NTPC",      "ONGC",
    "JSWSTEEL",  "HINDALCO",  "DRREDDY",   "CIPLA",     "TECHM",
    "BAJAJ-AUTO","GRASIM",    "ASIANPAINT","INDUSINDBK","HINDUNILVR",
]

DEFAULT_UNIVERSE: list[str] = INDEX_FUTURES + STOCK_FUTURES_UNIVERSE


# ──────────────────────────────────────────────────────────────────────────────
# SECTOR MAPPING  (for concentration limits and sector-neutral construction)
# ──────────────────────────────────────────────────────────────────────────────

SECTOR_MAP: dict[str, str] = {
    "RELIANCE": "Energy",      "ONGC": "Energy",       "BPCL": "Energy",
    "COALINDIA": "Energy",     "NTPC": "Utilities",    "POWERGRID": "Utilities",
    "TCS": "IT",               "INFY": "IT",           "WIPRO": "IT",
    "HCLTECH": "IT",           "TECHM": "IT",
    "HDFCBANK": "Financials",  "ICICIBANK": "Financials","SBIN": "Financials",
    "KOTAKBANK": "Financials", "AXISBANK": "Financials","BAJFINANCE": "Financials",
    "BAJAJFINSV": "Financials","INDUSINDBK": "Financials","SBILIFE": "Financials",
    "HDFCLIFE": "Financials",
    "HINDUNILVR": "FMCG",      "NESTLEIND": "FMCG",
    "SUNPHARMA": "Pharma",     "CIPLA": "Pharma",      "DRREDDY": "Pharma",
    "DIVISLAB": "Pharma",      "APOLLOHOSP": "Healthcare",
    "TATAMOTORS": "Auto",      "MARUTI": "Auto",       "EICHERMOT": "Auto",
    "HEROMOTOCO": "Auto",      "BAJAJ-AUTO": "Auto",   "M&M": "Auto",
    "LT": "Industrials",       "ADANIENT": "Conglomerate",
    "GRASIM": "Conglomerate",  "ASIANPAINT": "Chemicals",
    "ULTRACEMCO": "Cement",
    "JSWSTEEL": "Metals",      "TATASTEEL": "Metals",  "HINDALCO": "Metals",
    "TITAN": "Consumer",       "BHARTIARTL": "Telecom",
    "NIFTY": "Index",          "BANKNIFTY": "Index",
}


def is_index_future(ticker: str) -> bool:
    """True if the ticker is an index futures contract."""
    return ticker in INDEX_FUTURES
