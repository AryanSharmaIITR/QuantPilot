"""Domain definitions and config-derived constants for QuantPilot.

The ticker universe and category maps live here as code (they are domain data,
not tunable config). Everything else — paths, windows, dimensions — is pulled
from ``config.yaml`` via :mod:`config`, but re-exported under the original
constant names so the rest of the package keeps importing them unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

from config import CONFIG, ROOT_DIR

# ---------------------------------------------------------------------------
# Ticker universe (domain data)
# ---------------------------------------------------------------------------
# These dicts are the built-in defaults. If a ``tickers.json`` registry exists at
# the project root (managed by the web app), it overrides these — letting the
# instrument universe be edited without code changes. Order is preserved and
# defines the category indices, so it must stay stable for a trained model.
DEFAULT_STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "Larsen & Toubro": "LT.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr Reddy's Laboratories": "DRREDDY.NS",
    "Cipla": "CIPLA.NS",
    "Wipro": "WIPRO.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Tech Mahindra": "TECHM.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Tata Steel": "TATASTEEL.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Power Grid Corporation": "POWERGRID.NS",
    "NTPC": "NTPC.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "TVS Motor": "TVSMOTOR.NS",
    "Ashok Leyland": "ASHOKLEY.NS",
    "Bharat Forge": "BHARATFORG.NS",
    "MRF": "MRF.NS",
    "Balkrishna Industries": "BALKRISIND.NS",
    "Bosch": "BOSCHLTD.NS",
    "Samvardhana Motherson": "MOTHERSON.NS",
    "Persistent Systems": "PERSISTENT.NS",
    "Coforge": "COFORGE.NS",
    "Mphasis": "MPHASIS.NS",
    "L&T Technology Services": "LTTS.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Bank of Baroda": "BANKBARODA.NS",
    "Punjab National Bank": "PNB.NS",
    "Canara Bank": "CANBK.NS",
    "Union Bank of India": "UNIONBANK.NS",
    "IDFC First Bank": "IDFCFIRSTB.NS",
    "Federal Bank": "FEDERALBNK.NS",
    "Bandhan Bank": "BANDHANBNK.NS",
    "AU Small Finance Bank": "AUBANK.NS",
    "Indian Bank": "INDIANB.NS",
    "Bank of India": "BANKINDIA.NS",
    "Cholamandalam Finance": "CHOLAFIN.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "Muthoot Finance": "MUTHOOTFIN.NS",
    "Power Finance Corporation": "PFC.NS",
    "REC": "RECLTD.NS",
    "Indian Railway Finance": "IRFC.NS",
    "LIC Housing Finance": "LICHSGFIN.NS",
    "HDFC AMC": "HDFCAMC.NS",
    "SBI Cards": "SBICARD.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "SBI Life": "SBILIFE.NS",
    "ICICI Prudential Life": "ICICIPRULI.NS",
    "ICICI Lombard": "ICICIGI.NS",
    "Life Insurance Corporation": "LICI.NS",
    "Aditya Birla Capital": "ABCAPITAL.NS",
    "Nestle India": "NESTLEIND.NS",
    "Britannia Industries": "BRITANNIA.NS",
    "Tata Consumer Products": "TATACONSUM.NS",
    "Dabur India": "DABUR.NS",
    "Godrej Consumer Products": "GODREJCP.NS",
    "Marico": "MARICO.NS",
    "Colgate-Palmolive": "COLPAL.NS",
    "Varun Beverages": "VBL.NS",
    "United Breweries": "UBL.NS",
    "Titan Company": "TITAN.NS",
    "Berger Paints": "BERGEPAINT.NS",
    "Pidilite Industries": "PIDILITIND.NS",
    "Havells India": "HAVELLS.NS",
    "Voltas": "VOLTAS.NS",
    "Polycab India": "POLYCAB.NS",
    "DLF": "DLF.NS",
    "Godrej Properties": "GODREJPROP.NS",
    "Oberoi Realty": "OBEROIRLTY.NS",
    "Prestige Estates": "PRESTIGE.NS",
    "Macrotech Developers": "LODHA.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Aurobindo Pharma": "AUROPHARMA.NS",
    "Lupin": "LUPIN.NS",
    "Zydus Lifesciences": "ZYDUSLIFE.NS",
    "Torrent Pharmaceuticals": "TORNTPHARM.NS",
    "Alkem Laboratories": "ALKEM.NS",
    "Mankind Pharma": "MANKIND.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Max Healthcare": "MAXHEALTH.NS",
    "Fortis Healthcare": "FORTIS.NS",
    "Biocon": "BIOCON.NS",
    "Oil & Natural Gas Corporation": "ONGC.NS",
    "Indian Oil Corporation": "IOC.NS",
    "Bharat Petroleum": "BPCL.NS",
    "Hindustan Petroleum": "HINDPETRO.NS",
    "GAIL India": "GAIL.NS",
    "Coal India": "COALINDIA.NS",
    "Petronet LNG": "PETRONET.NS",
    "Indraprastha Gas": "IGL.NS",
    "Gujarat Gas": "GUJGASLTD.NS",
    "Tata Power": "TATAPOWER.NS",
    "Adani Green Energy": "ADANIGREEN.NS",
    "Adani Power": "ADANIPOWER.NS",
    "Adani Energy Solutions": "ADANIENSOL.NS",
    "JSW Energy": "JSWENERGY.NS",
    "NHPC": "NHPC.NS",
    "Vedanta": "VEDL.NS",
    "Hindalco Industries": "HINDALCO.NS",
    "Hindustan Zinc": "HINDZINC.NS",
    "National Aluminium": "NATIONALUM.NS",
    "Steel Authority of India": "SAIL.NS",
    "Jindal Steel & Power": "JINDALSTEL.NS",
    "NMDC": "NMDC.NS",
    "Grasim Industries": "GRASIM.NS",
    "Shree Cement": "SHREECEM.NS",
    "Ambuja Cements": "AMBUJACEM.NS",
    "ACC": "ACC.NS",
    "Dalmia Bharat": "DALBHARAT.NS",
    "Siemens": "SIEMENS.NS",
    "ABB India": "ABB.NS",
    "BHEL": "BHEL.NS",
    "Bharat Electronics": "BEL.NS",
    "Hindustan Aeronautics": "HAL.NS",
    "Mazagon Dock Shipbuilders": "MAZDOCK.NS",
    "CG Power & Industrial": "CGPOWER.NS",
    "Cummins India": "CUMMINSIND.NS",
    "Container Corporation": "CONCOR.NS",
    "Indian Railway Catering (IRCTC)": "IRCTC.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Indus Towers": "INDUSTOWER.NS",
    "Trent": "TRENT.NS",
    "Avenue Supermarts (DMart)": "DMART.NS",
    "Info Edge": "NAUKRI.NS",
    "Aditya Birla Fashion": "ABFRL.NS",
    "PB Fintech": "POLICYBZR.NS",
    "SRF": "SRF.NS",
    "PI Industries": "PIIND.NS",
    "UPL": "UPL.NS",
    "InterGlobe Aviation (IndiGo)": "INDIGO.NS",
}

DEFAULT_NSE = {
    # Broad Market Indices (Most commonly tracked)
    "NIFTY 50": "^NSEI",
    "BSE Sensex": "^BSESN",
    "NIFTY BANK": "^NSEBANK",
    "INDIA VIX": "^INDIAVIX",
    "NIFTY NEXT 50": "^NSMIDCP",
    # Nifty Broad Market
    "NIFTY 100": "^CNX100",
    "NIFTY 200": "^CNX200",
    "NIFTY 500": "^CRSLDX",
    # Exchange
    "Gold Futures": "GC=F",
    "USDINR": "USDINR=X",
    "Crude Oil Futures": "CL=F",
}

# ---------------------------------------------------------------------------
# Active universe — loaded from the registry file if present, else defaults.
# ---------------------------------------------------------------------------
TICKERS_PATH = Path(ROOT_DIR) / "tickers.json"


def load_universe() -> tuple[dict[str, str], dict[str, str]]:
    """Return (stocks, market) ticker maps from tickers.json, or the defaults."""
    if TICKERS_PATH.exists():
        try:
            with open(TICKERS_PATH, "r") as f:
                reg = json.load(f)
            stocks = reg.get("stocks") or dict(DEFAULT_STOCKS)
            market = reg.get("market") or dict(DEFAULT_NSE)
            if stocks and market:
                return dict(stocks), dict(market)
        except (json.JSONDecodeError, OSError):
            pass  # fall back to defaults on a malformed/unreadable registry
    return dict(DEFAULT_STOCKS), dict(DEFAULT_NSE)


stocks_tickers, nse_tickers = load_universe()

# Category indices (stable ordering — must match the trained model).
stocks_cat = {name: idx for idx, name in enumerate(stocks_tickers)}
nse_cat = {name: idx for idx, name in enumerate(nse_tickers)}

# ---------------------------------------------------------------------------
# Config-derived constants (sourced from config.yaml)
# ---------------------------------------------------------------------------
_paths = CONFIG["paths"]
_data = CONFIG["data"]

# Training data directories
RAW_DIR_NSE = _paths["raw_market"]
RAW_DIR_STOCK = _paths["raw_stock"]
PREPROCESSED_DIR_NSE = _paths["preprocessed_market"]
PREPROCESSED_DIR_STOCK = _paths["preprocessed_stock"]

# Live-prediction data directories
PREDICT_RAW_DIR_NSE = _paths["predict_raw_market"]
PREDICT_RAW_DIR_STOCK = _paths["predict_raw_stock"]
PREDICT_PREPROCESSED_DIR_NSE = _paths["predict_preprocessed_market"]
PREDICT_PREPROCESSED_DIR_STOCK = _paths["predict_preprocessed_stock"]

ARTIFACTS_DIR = _paths["artifacts"]
PREDICTIONS_PATH = _paths["predictions"]

# Per-feature normalization stats computed from the TRAIN split only and reused
# at val/test/live inference, so no future statistics ever leak into a prediction.
NORM_STATS_PATH = str(Path(ARTIFACTS_DIR) / f"{CONFIG['project']['name']}_norm_stats.npz")

# Windows / dimensions
timeperiod = _data["train_timeperiod"]
PREDICT_TIMEPERIOD = _data["predict_timeperiod"]
SEQUENCE_LENGTH = _data["sequence_length"]
DIM = _data["dim"]

TRAIN_SPLIT = _data["splits"]["train"]
VAL_SPLIT = _data["splits"]["val"]

NSE_INDICES = len(nse_tickers)
STOCK_INDICES = len(stocks_tickers)


def sanitize(name: str) -> str:
    """Convert a human ticker name into the canonical CSV filename stem."""
    return name.lower().replace(" ", "").replace(":", "_").replace("/", "_")
