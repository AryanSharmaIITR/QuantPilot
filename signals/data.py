stocks_tickers={
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
    "Adani Ports": "ADANIPORTS.NS"
}

nse_tickers={
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

RAW_DIR_NSE = "./data/raw/market_data"
RAW_DIR_STOCK = "./data/raw/stock_data"

PREPROCESSED_DIR_NSE = "./data/preprocessed/market_data"
PREPROCESSED_DIR_STOCK = "./data/preprocessed/stock_data"

timeperiod="3y"