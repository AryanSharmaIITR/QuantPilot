import pandas as pd
from pathlib import Path
import pandas as pd
import os
from data import BASE_DIR_NSE,nse_tickers,stocks_tickers,timeperiod,BASE_DIR_STOCK
import yfinance as yf

class dataIngestion:
    def __init__(self):
        pass

    def get_nse_data(self):
        os.makedirs(BASE_DIR_NSE, exist_ok=True)
        for name,ticker in nse_tickers.items():
            try:
                yfticker = yf.Ticker(ticker)
                hist = yfticker.history(period=timeperiod)
                # Handle empty or invalid responses
                if hist.empty:
                    print(f"[ERROR] No data returned for: {name} ({ticker})")
                    continue
                df = pd.DataFrame(hist).reset_index()
                df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

                file_name = name.lower()
                file_name=file_name.replace(" ","")
                file_name=file_name.replace(":","_")
                file_name=file_name.replace("/","_")
                file_path = os.path.join(BASE_DIR_NSE, f"{file_name}.csv")

                df.to_csv(file_path, index=False)
                print(f"{name} {ticker} Downloaded")
            except Exception as e:
                print(f"{name} ({ticker}) Failed: {e}")

    def get_stock_data(self):
        os.makedirs(BASE_DIR_STOCK, exist_ok=True)
        for name,ticker in stocks_tickers.items():
            try:
                yfticker = yf.Ticker(ticker)
                hist = yfticker.history(period=timeperiod)
                # Handle empty or invalid responses
                if hist.empty:
                    print(f"[ERROR] No data returned for: {name} ({ticker})")
                    continue
                df = pd.DataFrame(hist).reset_index()
                df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

                file_name = name.lower()
                file_name=file_name.replace(" ","")
                file_name=file_name.replace(":","_")
                file_name=file_name.replace("/","_")
                file_path = os.path.join(BASE_DIR_STOCK, f"{file_name}.csv")

                df.to_csv(file_path, index=False)

                print(f"{name} {ticker} Downloaded")
            except Exception as e:
                print(f"{name} ({ticker}) Failed: {e}")
    
    def startIngestion(self):
        print("Downloading NSE DATA")
        self.get_nse_data()
        print("-"*20)
        print("Downloading STOCKS DATA")
        self.get_stock_data()

if __name__=="__main__":
    di=dataIngestion()
    di.startIngestion()
