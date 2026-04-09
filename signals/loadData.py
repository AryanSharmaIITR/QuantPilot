from torch.utils.data import DataLoader,Dataset
import pandas as pd
import torch
import numpy as np

from data import stocks_cat, nse_cat, PREPROCESSED_DIR_NSE, PREPROCESSED_DIR_STOCK

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
        target_idx = end_idx
        
        # Collect market data sequences
        market_data = []
        market_target_data = []
        for df in self.mkd:
            seq_data = df.iloc[start_idx:end_idx].values
            target_data = df.iloc[target_idx].values
            market_data.append(seq_data)
            market_target_data.append(target_data)

        target=[]
        for df in self.targets:
            target_data = df.iloc[target_idx].values
            target.append(target_data)
        
        # Collect stock data sequences
        stock_data = []
        stock_target_data = []
        for df in self.skd:
            seq_data = df.iloc[start_idx:end_idx].values
            target_data = df.iloc[target_idx].values
            stock_data.append(seq_data)
            stock_target_data.append(target_data)
        
        # Convert to tensors
        market_data = torch.tensor(np.array(market_data), dtype=torch.float32)
        stock_data = torch.tensor(np.array(stock_data), dtype=torch.float32)
        market_target_data = torch.tensor(np.array(market_target_data), dtype=torch.float32)
        stock_target_data = torch.tensor(np.array(stock_target_data), dtype=torch.float32)
        target = torch.tensor(np.array(target), dtype=torch.float32)
        
        # ===== CRITICAL FIXES =====
        # 1. Replace any NaN/Inf
        market_data = torch.nan_to_num(market_data, nan=0.0, posinf=10.0, neginf=-10.0)
        stock_data = torch.nan_to_num(stock_data, nan=0.0, posinf=10.0, neginf=-10.0)
        market_target_data = torch.nan_to_num(market_target_data, nan=0.0, posinf=10.0, neginf=-10.0)
        stock_target_data = torch.nan_to_num(stock_target_data, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # 2. Clip extreme values to prevent overflow
        market_data = torch.clamp(market_data, min=-5.0, max=5.0)
        stock_data = torch.clamp(stock_data, min=-5.0, max=5.0)
        market_target_data = torch.clamp(market_target_data, min=-5.0, max=5.0)
        stock_target_data = torch.clamp(stock_target_data, min=-5.0, max=5.0)
        
        return market_data, stock_data, market_target_data, stock_target_data ,target
    

def get_dataloader(market_dataset_paths=PREPROCESSED_DIR_NSE, stock_dataset_paths=PREPROCESSED_DIR_STOCK, batch_size=32, sequence_length=14):

    market_train_dataset_path = []
    market_test_dataset_path = []  
    market_val_dataset_path = []
    for name in nse_cat.keys():
        print(f"{name},")
        name = name.lower()
        name = name.replace(" ", "")
        name = name.replace(":", "_")
        name = name.replace("/", "_")
    
        market_train_dataset_path.append(f"{market_dataset_paths}/train/{name}.csv")
        market_test_dataset_path.append(f"{market_dataset_paths}/test/{name}.csv")
        market_val_dataset_path.append(f"{market_dataset_paths}/val/{name}.csv")

    stock_train_dataset_path = []
    stock_test_dataset_path = []  
    stock_val_dataset_path = []
    for name in stocks_cat.keys():
        print(f"{name},")
        name = name.lower()
        name = name.replace(" ", "")
        name = name.replace(":", "_")
        name = name.replace("/", "_")
        stock_train_dataset_path.append(f"{stock_dataset_paths}/train/{name}.csv")
        stock_test_dataset_path.append(f"{stock_dataset_paths}/test/{name}.csv")
        stock_val_dataset_path.append(f"{stock_dataset_paths}/val/{name}.csv")

    train_dataset = dataset(market_train_dataset_path, stock_train_dataset_path,sequence_length)
    test_dataset = dataset(market_test_dataset_path, stock_test_dataset_path,sequence_length)
    val_dataset = dataset(market_val_dataset_path, stock_val_dataset_path,sequence_length)

    print(f"Train: {len(train_dataset)} samples")
    print(f"Validation: {len(val_dataset)} samples") 
    print(f"Test: {len(test_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader