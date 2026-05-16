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

stocks_cat={
    "Reliance Industries": 0,
    "Tata Consultancy Services": 1,
    "Infosys": 2,
    "HDFC Bank": 3,
    "ICICI Bank": 4,
    "State Bank of India": 5,
    "Kotak Mahindra Bank": 6,
    "Axis Bank": 7,
    "Hindustan Unilever": 8,
    "ITC": 9,
    "Larsen & Toubro": 10,
    "Asian Paints": 11,
    "Bajaj Finance": 12,
    "Bajaj Finserv": 13,
    "Maruti Suzuki": 14,
    "Mahindra & Mahindra": 15,
    "Sun Pharma": 16,
    "Dr Reddy's Laboratories": 17,
    "Cipla": 18,
    "Wipro": 19,
    "HCL Technologies": 20,
    "Tech Mahindra": 21,
    "UltraTech Cement": 22,
    "Tata Steel": 23,
    "JSW Steel": 24,
    "Power Grid Corporation": 25,
    "NTPC": 26,
    "Adani Enterprises": 27,
    "Adani Ports": 28
}


nse_cat={
    # Broad Market Indices (Most commonly tracked)
    "NIFTY 50": 0,
    "BSE Sensex": 1,
    "NIFTY BANK": 2,
    "INDIA VIX": 3,
    "NIFTY NEXT 50": 4,
    # Nifty Broad Market
    "NIFTY 100": 5,
    "NIFTY 200": 6,
    "NIFTY 500": 7,
    # Exchange 
    "Gold Futures": 8,
    "USDINR": 9,
    "Crude Oil Futures": 10,
}
RAW_DIR_NSE = "./data/raw/market_data"
RAW_DIR_STOCK = "./data/raw/stock_data"

PREPROCESSED_DIR_NSE = "./data/preprocessed/market_data"
PREPROCESSED_DIR_STOCK = "./data/preprocessed/stock_data"

timeperiod="3y"

DIM = 18
NSE_INDICES = len(nse_tickers)
STOCK_INDICES = len(stocks_tickers)