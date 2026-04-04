import pandas as pd
import numpy as np
import os
from data import nse_tickers,stocks_tickers,PREPROCESSED_DIR_NSE,PREPROCESSED_DIR_STOCK
from data import RAW_DIR_NSE, RAW_DIR_STOCK

class preProcessing:
    def __init__(self):
        self.epsilon=1e-8
    
    def set_nan(self,df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df
    
    def rsi(self,df,period=14):
        delta = df["day_return_scaled"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / (avg_loss + self.epsilon)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def rolling_sortino(self, window):
        window = np.asarray(window)
        downside = window[window < 0]
        if len(downside) == 0:
            return 0.0
        downside_std = downside.std(ddof=1)
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0
        return window.mean() / downside_std

    def add_technical_indicators(self,df):
        indicators_dict = {}
        series = df["day_return_scaled"]
        returns_1d = series.pct_change()


        indicators_dict["RSI"] = self.rsi(df)
        indicators_dict["Sortino"] = series.rolling(window=14).apply(self.rolling_sortino, raw=False)

        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd = ema26 - ema12
        indicators_dict["MACD"] = macd

        ema15 = series.ewm(span=15, adjust=False).mean()
        indicators_dict[f"emaslope_15d"] = ema15.diff()
        
        tr = series.diff().abs()
        indicators_dict[f"tr"] = tr
        indicators_dict[f"atr_15d"] = tr.rolling(15).mean()

        mean_15 = series.rolling(15).mean()
        std_15 = series.rolling(15).std()
        upper_15 = mean_15 + 2 * std_15
        lower_15 = mean_15 - 2 * std_15
            
        indicators_dict["bb_width_15d"] = (upper_15 - lower_15) / (mean_15 + self.epsilon)
        indicators_dict["zscore_15d"] = (series - mean_15) / (std_15 + self.epsilon)

        indicators_dict["ret_15d"] = series.pct_change(15)

        roll_max_15 = series.rolling(15).max()
        drawdown_15 = series / roll_max_15 - 1
        indicators_dict["max_dd_15d"] = drawdown_15.rolling(15).min()

        vol15 = series.rolling(15).std() * np.sqrt(252)
        mean15 = series.rolling(15).mean()
        rf_daily = 0.05 / 252

        indicators_dict[f"vol_15d"] = vol15
        indicators_dict["sharpe_ratio_15"] = (mean15 - rf_daily) * 15 / (vol15+self.epsilon)
        indicators_dict["var95_15"] = series.rolling(15).quantile(0.05)
        indicators_dict["cvar95_15"] = series.rolling(15).apply(
                lambda x: x[x <= x.quantile(0.05)].mean() if len(x) > 0 else np.nan
        )
        indicators_dict["beta_15"] = (
                    returns_1d.rolling(15).cov(series) /
                    (series.rolling(15).var()+self.epsilon)
        )   
        indicators_dict["alpha_5d"] = returns_1d.rolling(5).mean() - indicators_dict[f"beta_15"] * series.rolling(5).mean()

        indicators_dict_df = pd.DataFrame(indicators_dict)
        df = pd.concat([df, indicators_dict_df], axis=1)
        return df
    
    def startPreprocessing(self):
        os.makedirs(PREPROCESSED_DIR_NSE, exist_ok=True)
        for name in nse_tickers.keys():
            file_name = name.lower()
            file_name=file_name.replace(" ","")
            file_name=file_name.replace(":","_")
            file_name=file_name.replace("/","_")
            file_path = os.path.join(RAW_DIR_NSE, f"{file_name}.csv")
            df = pd.read_csv(file_path)
            df=self.set_nan(df)
            df=self.add_technical_indicators(df)
            df=self.set_nan(df)
            df.to_csv(os.path.join(PREPROCESSED_DIR_NSE, f"{file_name}.csv"), index=False)
            print(f"{name} NSE Preprocessed")
        
        print("------------------------------   ------------------------------")
        os.makedirs(PREPROCESSED_DIR_STOCK, exist_ok=True)
        for name in stocks_tickers.keys():
            file_name = name.lower()
            file_name=file_name.replace(" ","")
            file_name=file_name.replace(":","_")
            file_name=file_name.replace("/","_")
            file_path = os.path.join(RAW_DIR_STOCK, f"{file_name}.csv")
            df = pd.read_csv(file_path)
            df=self.set_nan(df)
            df=self.add_technical_indicators(df)
            df=self.set_nan(df)
            df["Target"] = df["day_return_scaled"].shift(-1)
            df.to_csv(os.path.join(PREPROCESSED_DIR_STOCK, f"{file_name}.csv"), index=False)
            print(f"{name} Stock Preprocessed")


if __name__ == "__main__":
    preprocessor = preProcessing()
    preprocessor.startPreprocessing()