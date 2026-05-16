from models import TimeSeriesTransformer,MarketAggregator,FusionV2, SurrogateHead
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import yfinance as yf
import numpy as np

from data import  nse_tickers, stocks_tickers, nse_cat, stocks_cat
RAW_DIR_NSE = "./Data_For_Prediction/market_data"
RAW_DIR_STOCK = "./Data_For_Prediction/stock_data"
PREPROCESSED_DIR_NSE = "./Data_For_Prediction/preprocessed_market_data"
PREPROCESSED_DIR_STOCK = "./Data_For_Prediction/preprocessed_stock_data"

timeperiod = "1mo"

class dataIngestion:
    def __init__(self):
        pass

    def get_nse_data(self):
        os.makedirs(RAW_DIR_NSE, exist_ok=True)
        for name,ticker in nse_tickers.items():
            try:
                yfticker = yf.Ticker(ticker)
                hist = yfticker.history(period=timeperiod)
                if hist.empty:
                    print(f"[ERROR] No data returned for: {name} ({ticker})")
                    continue
                df = pd.DataFrame(hist).reset_index()
                df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

                file_name = name.lower()
                file_name=file_name.replace(" ","")
                file_name=file_name.replace(":","_")
                file_name=file_name.replace("/","_")
                file_path = os.path.join(RAW_DIR_NSE, f"{file_name}.csv")

                df[f"day_return_scaled"] = ((df["Close"] - df["Open"]) / df["Open"]) * 1000
                df=df[["Date",f"day_return_scaled"]]
                df.to_csv(file_path, index=False)
                
                print(f"{name} {ticker} Downloaded")
            except Exception as e:
                print(f"{name} ({ticker}) Failed: {e}")

    def get_stock_data(self):
        os.makedirs(RAW_DIR_STOCK, exist_ok=True)
        for name,ticker in stocks_tickers.items():
            try:
                yfticker = yf.Ticker(ticker)
                hist = yfticker.history(period=timeperiod)
                if hist.empty:
                    print(f"[ERROR] No data returned for: {name} ({ticker})")
                    continue
                df = pd.DataFrame(hist).reset_index()
                df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

                file_name = name.lower()
                file_name=file_name.replace(" ","")
                file_name=file_name.replace(":","_")
                file_name=file_name.replace("/","_")
                file_path = os.path.join(RAW_DIR_STOCK, f"{file_name}.csv")
                
                df[f"day_return_scaled"] = ((df["Close"] - df["Open"]) / df["Open"]) * 1000
                df=df[["Date",f"day_return_scaled"]]
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
        print("------------------------------   ------------------------------")
        for name in nse_tickers.keys():
            temp_name=name
            file_name = name.lower()
            file_name=file_name.replace(" ","")
            file_name=file_name.replace(":","_")
            file_name=file_name.replace("/","_")
            file_path = os.path.join(RAW_DIR_NSE, f"{file_name}.csv")
            df = pd.read_csv(file_path)
            df=self.set_nan(df)
            df=self.add_technical_indicators(df)
            df=self.set_nan(df)
            df["category"] = nse_cat[temp_name]

            df = df.sort_values("Date").reset_index(drop=True)  # critical
            n = len(df)



            df.to_csv(os.path.join(PREPROCESSED_DIR_NSE, f"{file_name}.csv"), index=False)
            print(f"{name} NSE Preprocessed")
        
        print("------------------------------   ------------------------------")
        os.makedirs(PREPROCESSED_DIR_STOCK, exist_ok=True)
        for name in stocks_tickers.keys():
            temp_name=name
            file_name = name.lower()
            file_name=file_name.replace(" ","")
            file_name=file_name.replace(":","_")
            file_name=file_name.replace("/","_")
            file_path = os.path.join(RAW_DIR_STOCK, f"{file_name}.csv")
            df = pd.read_csv(file_path)
            df=self.set_nan(df)
            df=self.add_technical_indicators(df)
            df=self.set_nan(df)
            df["target"] = df["day_return_scaled"].shift(-1)
            df["target"] = (df["day_return_scaled"].shift(-1) > 0).astype(int)
            df["category"] = stocks_cat[temp_name]

            df = df.sort_values("Date").reset_index(drop=True)  # critical
            n = len(df)

            df.to_csv(os.path.join(PREPROCESSED_DIR_STOCK, f"{file_name}.csv"), index=False)
            print(f"{name} STOCK Preprocessed")

class dataset(Dataset):
    def __init__(self, market_dataset_path, stock_dataset_path, sequence_length=14):
        self.mkd = []
        self.skd = []
        self.targets = []
        
        # Load market data
        for path in market_dataset_path:
            df = pd.read_csv(path)
            df = df.sort_values("Date")
            self.mkd.append(df)
        
        # Load stock data
        for path in stock_dataset_path:
            df = pd.read_csv(path)
            df = df.sort_values("Date")
            self.targets.append(df[["Date","target"]])  # Store target values separately
            df=df.drop(columns=["target"])
            self.skd.append(df)

        # Find common dates
        common_dates_mkd = set(self.mkd[0]['Date'])
        for df in self.mkd[1:]:
            common_dates_mkd = common_dates_mkd.intersection(set(df['Date']))
            
        common_dates_skd = set(self.skd[0]['Date'])
        for df in self.skd[1:]:
            common_dates_skd = common_dates_skd.intersection(set(df['Date']))

        self.common_dates = common_dates_mkd.intersection(common_dates_skd)
        self.common_dates = sorted(list(self.common_dates))

        # Filter and set index for each market dataframe
        filtered_mkd = []
        for df in self.mkd:
            filtered_df = df[df['Date'].isin(self.common_dates)].copy()
            filtered_df = filtered_df.sort_values("Date")
            filtered_df = filtered_df.set_index('Date')
            filtered_mkd.append(filtered_df)
        self.mkd = filtered_mkd

        # Filter and set index for each stock dataframe
        filtered_skd = []
        for df in self.skd:
            filtered_df = df[df['Date'].isin(self.common_dates)].copy()
            filtered_df = filtered_df.sort_values("Date")
            filtered_df = filtered_df.set_index('Date')
            filtered_skd.append(filtered_df)
        self.skd = filtered_skd

        filtered_targets = []
        for df in self.targets:
            filtered_df = df[df['Date'].isin(self.common_dates)].copy()
            filtered_df = filtered_df.sort_values("Date")
            filtered_df = filtered_df.set_index('Date')
            filtered_targets.append(filtered_df)
        self.targets = filtered_targets

        # ===== ADD NORMALIZATION =====
        # Calculate mean and std from all data
        all_market = np.concatenate([df.values for df in self.mkd], axis=0)
        all_stock = np.concatenate([df.values for df in self.skd], axis=0)
        
        self.market_mean = torch.tensor(all_market.mean(axis=0), dtype=torch.float32)
        self.market_std = torch.tensor(all_market.std(axis=0) + 1e-8, dtype=torch.float32)
        self.stock_mean = torch.tensor(all_stock.mean(axis=0), dtype=torch.float32)
        self.stock_std = torch.tensor(all_stock.std(axis=0) + 1e-8, dtype=torch.float32)
        
        # Normalize the dataframes
        for i in range(len(self.mkd)):
            self.mkd[i] = (self.mkd[i] - self.market_mean.numpy()) / self.market_std.numpy()
        
        for i in range(len(self.skd)):
            self.skd[i] = (self.skd[i] - self.stock_mean.numpy()) / self.stock_std.numpy()
        
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.common_dates) - self.sequence_length - 1

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.sequence_length
        
        # Collect market data sequences
        market_data = []
        for df in self.mkd:
            seq_data = df.iloc[start_idx:end_idx].values
            market_data.append(seq_data)

        
        # Collect stock data sequences
        stock_data = []
        for df in self.skd:
            seq_data = df.iloc[start_idx:end_idx].values
            stock_data.append(seq_data)
        
        # Convert to tensors
        market_data = torch.tensor(np.array(market_data), dtype=torch.float32)
        stock_data = torch.tensor(np.array(stock_data), dtype=torch.float32)
        
        # ===== CRITICAL FIXES =====
        # 1. Replace any NaN/Inf
        market_data = torch.nan_to_num(market_data, nan=0.0, posinf=10.0, neginf=-10.0)
        stock_data = torch.nan_to_num(stock_data, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # 2. Clip extreme values to prevent overflow
        market_data = torch.clamp(market_data, min=-5.0, max=5.0)
        stock_data = torch.clamp(stock_data, min=-5.0, max=5.0)
        
        return market_data, stock_data
    
market_paths = [os.path.join(PREPROCESSED_DIR_NSE, f) for f in os.listdir(PREPROCESSED_DIR_NSE) if f.endswith(".csv")]
stock_paths = [os.path.join(PREPROCESSED_DIR_STOCK, f) for f in os.listdir(PREPROCESSED_DIR_STOCK) if f.endswith(".csv")]

DataLoader = dataset(market_paths, stock_paths)
print("Data Loaded Successfully")
print("Dataset Length:", len(DataLoader.common_dates))
n = len(DataLoader.common_dates)
print("Target: ", n-14)
market_data, stock_data = DataLoader.__getitem__(n-14)
print("Market Data Shape:", market_data.shape)
print("Stock Data Shape:", stock_data.shape)

market_model = TimeSeriesTransformer(
        input_dim=18,
        d_model=128,
        nhead=8,
        num_layers=3,
        num_market_indices=11
).to(torch.device("cpu"))

stock_model = TimeSeriesTransformer(
        input_dim=18,
        d_model=128,
        nhead=8,
        num_layers=3,
        num_market_indices=29
).to(torch.device("cpu"))

market_aggregator = MarketAggregator().to(torch.device("cpu"))

fusion_model = FusionV2().to(torch.device("cpu"))

head = SurrogateHead(dim=270).to(torch.device("cpu"))

path_prefix = "./artifacts/final_market_pipeline_new"
ckpt = torch.load(f"{path_prefix}_nn.pt", map_location="cpu")

market_aggregator.load_state_dict(ckpt["aggregator"])
fusion_model.load_state_dict(ckpt["fusion"])
head.load_state_dict(ckpt["head"])
market_model.load_state_dict(ckpt["market_model"])
stock_model.load_state_dict(ckpt["stock_model"])

print("Models Loaded Successfully")
market_aggregator.eval()
fusion_model.eval()
head.eval()
market_model.eval()
stock_model.eval()

market_data = market_data.unsqueeze(0)
stock_data = stock_data.unsqueeze(0)
mkd_output = market_model(market_data)
skd_output = stock_model(stock_data)

print("stage 1 done")
flt_mkd_out = mkd_output.view(1,-1)  # (B, 198)

agg_out = market_aggregator(mkd_output)  # (B, 128)

N_STOCKS = len(stock_paths)
fused_list = []

for stock_idx in range(N_STOCKS):
    s_vec = skd_output[:, stock_idx, :]           # (B, 18) - Transformer output
    ls_vec = stock_data[:, stock_idx, -1, :]      # (B, 18) - Last raw timestep
    fused = fusion_model(s_vec, agg_out)        # (B, 18) - Fused features

    input_vec = torch.cat([ls_vec, s_vec, agg_out, fused, flt_mkd_out], dim=1)

    fused_list.append(input_vec)

print("Final Fused Vector Shape:", fused_list[0].shape)  # Should be (B, 270)
features = torch.stack(fused_list, dim=1)
print("Stacked Features Shape:", features.shape)  # Should be (B, N_STOCKS, 270)

re = head(features)
print("Final Output Shape:", re.shape)  # Should be (B, N_STOCKS, 1)

y_pred_prob = torch.sigmoid(re)
y_pred_label = (y_pred_prob > 0.5).int()

y_pred_label = y_pred_label.reshape(-1)

print("-" * 50)
print("\n" * 2)

result_df = pd.DataFrame({
    "stock_path": stock_paths,
    "y_pred_label": y_pred_label
})
print(result_df)
result_df.to_csv("./predictions.csv", index=False)

# if __name__=="__main__":
#     # data_ingestion = dataIngestion()
#     # data_ingestion.startIngestion()

#     pre_processing = preProcessing()
#     pre_processing.startPreprocessing()
