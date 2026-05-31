"""Stage 4 — live inference.

Loads the trained NN checkpoint (transformers + aggregator + fusion + surrogate
head) and produces a next-day directional signal (up/down) for every stock in
the universe from the most recent preprocessed sequence.

Inference uses the surrogate head directly (sigmoid -> threshold); the XGBoost
stage is training/evaluation-only.

Run order matters: market and stock instruments are loaded in the exact category
order used during training so each row maps to the correct learned embedding.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import data as D
from config import CONFIG
from logger import get_logger
from models import TimeSeriesTransformer, MarketAggregator, FusionV2, SurrogateHead

log = get_logger("predict")


class PredictionDataset(Dataset):
    """Builds aligned, normalized market/stock sequences for inference (no targets)."""

    def __init__(self, market_paths: list[str], stock_paths: list[str], sequence_length: int):
        self.mkd, self.skd = [], []

        for path in market_paths:
            self.mkd.append(pd.read_csv(path).sort_values("Date"))

        for path in stock_paths:
            df = pd.read_csv(path).sort_values("Date")
            # The model was trained on features with the target column dropped.
            if "target" in df.columns:
                df = df.drop(columns=["target"])
            self.skd.append(df)

        # Restrict to dates present across every instrument.
        common = set(self.mkd[0]["Date"])
        for df in self.mkd[1:] + self.skd:
            common &= set(df["Date"])
        self.common_dates = sorted(common)

        self.mkd = [self._align(df) for df in self.mkd]
        self.skd = [self._align(df) for df in self.skd]

        # Normalize per-feature across all instruments (matches training).
        all_market = np.concatenate([df.values for df in self.mkd], axis=0)
        all_stock = np.concatenate([df.values for df in self.skd], axis=0)
        m_mean, m_std = all_market.mean(0), all_market.std(0) + 1e-8
        s_mean, s_std = all_stock.mean(0), all_stock.std(0) + 1e-8
        self.mkd = [(df - m_mean) / m_std for df in self.mkd]
        self.skd = [(df - s_mean) / s_std for df in self.skd]

        self.sequence_length = sequence_length

    def _align(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["Date"].isin(self.common_dates)].copy()
        return df.sort_values("Date").set_index("Date")

    def __len__(self) -> int:
        return len(self.common_dates) - self.sequence_length - 1

    def __getitem__(self, idx: int):
        start, end = idx, idx + self.sequence_length
        market = torch.tensor(np.array([df.iloc[start:end].values for df in self.mkd]), dtype=torch.float32)
        stock = torch.tensor(np.array([df.iloc[start:end].values for df in self.skd]), dtype=torch.float32)
        market = torch.clamp(torch.nan_to_num(market, nan=0.0, posinf=10.0, neginf=-10.0), -5.0, 5.0)
        stock = torch.clamp(torch.nan_to_num(stock, nan=0.0, posinf=10.0, neginf=-10.0), -5.0, 5.0)
        return market, stock

    def latest_window(self):
        """Return the most recent (market, stock) sequence — the one we predict from."""
        idx = len(self.common_dates) - self.sequence_length
        return self.__getitem__(idx)


class Predictor:
    """Loads the trained pipeline and emits next-day directional signals."""

    def __init__(self, device: str | None = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        tcfg = CONFIG["transformer"]
        self.threshold = CONFIG["inference"]["threshold"]
        self.checkpoint_path = CONFIG["inference"]["nn_checkpoint"]
        self.seq_len = D.SEQUENCE_LENGTH

        self.market_model = TimeSeriesTransformer(
            input_dim=D.DIM, d_model=tcfg["d_model"], nhead=tcfg["nhead"],
            num_layers=tcfg["num_layers"], dim_feedforward=tcfg["dim_feedforward"],
            dropout=tcfg["dropout"], num_market_indices=D.NSE_INDICES,
        ).to(self.device)
        self.stock_model = TimeSeriesTransformer(
            input_dim=D.DIM, d_model=tcfg["d_model"], nhead=tcfg["nhead"],
            num_layers=tcfg["num_layers"], dim_feedforward=tcfg["dim_feedforward"],
            dropout=tcfg["dropout"], num_market_indices=D.STOCK_INDICES,
        ).to(self.device)
        self.aggregator = MarketAggregator(D.DIM).to(self.device)
        self.fusion = FusionV2(D.DIM).to(self.device)
        self.head = SurrogateHead(dim=270).to(self.device)

        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Inference checkpoint not found: {self.checkpoint_path}. "
                "Train the model (`python signals/pipeline.py train`) or point "
                "inference.nn_checkpoint in config.yaml at a valid checkpoint."
            )
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.aggregator.load_state_dict(ckpt["aggregator"])
        self.fusion.load_state_dict(ckpt["fusion"])
        self.head.load_state_dict(ckpt["head"])
        self.market_model.load_state_dict(ckpt["market_model"])
        self.stock_model.load_state_dict(ckpt["stock_model"])
        for m in (self.market_model, self.stock_model, self.aggregator, self.fusion, self.head):
            m.eval()
        log.info("Loaded inference checkpoint: %s", self.checkpoint_path)

    @torch.no_grad()
    def _forward(self, market_data: torch.Tensor, stock_data: torch.Tensor) -> torch.Tensor:
        """Return per-stock up-probabilities for a single batch element."""
        market_data = market_data.unsqueeze(0).to(self.device)  # (1, N_mkt, seq, dim)
        stock_data = stock_data.unsqueeze(0).to(self.device)    # (1, N_stk, seq, dim)

        mkd_out = self.market_model(market_data)   # (1, N_mkt, dim)
        skd_out = self.stock_model(stock_data)     # (1, N_stk, dim)
        flat_mkd = mkd_out.view(1, -1)
        market_vec = self.aggregator(mkd_out)      # (1, dim)

        feats = []
        for i in range(D.STOCK_INDICES):
            s_vec = skd_out[:, i, :]
            ls_vec = stock_data[:, i, -1, :]
            fused = self.fusion(s_vec, market_vec)
            feats.append(torch.cat([ls_vec, s_vec, market_vec, fused, flat_mkd], dim=1))

        features = torch.stack(feats, dim=1)       # (1, N_stk, 270)
        logits = self.head(features)               # (1, N_stk, 1)
        return torch.sigmoid(logits).reshape(-1)   # (N_stk,)

    def predict(self) -> pd.DataFrame:
        # Build ordered path lists (category order == training order).
        market_paths = [
            os.path.join(D.PREDICT_PREPROCESSED_DIR_NSE, f"{D.sanitize(n)}.csv")
            for n in D.nse_cat.keys()
        ]
        stock_paths = [
            os.path.join(D.PREDICT_PREPROCESSED_DIR_STOCK, f"{D.sanitize(n)}.csv")
            for n in D.stocks_cat.keys()
        ]

        ds = PredictionDataset(market_paths, stock_paths, self.seq_len)
        log.info("Aligned %d common dates; predicting from latest %d-day window",
                 len(ds.common_dates), self.seq_len)

        market_data, stock_data = ds.latest_window()
        probs = self._forward(market_data, stock_data).cpu().numpy()
        labels = (probs > self.threshold).astype(int)

        as_of = ds.common_dates[-1]
        result = pd.DataFrame({
            "stock": list(D.stocks_cat.keys()),
            "ticker": list(D.stocks_tickers.values()),
            "as_of_date": as_of,
            "up_probability": np.round(probs, 4),
            "signal": np.where(labels == 1, "UP", "DOWN"),
        })

        out_path = D.PREDICTIONS_PATH
        result.to_csv(out_path, index=False)
        log.info("Wrote %d predictions (as of %s) to %s", len(result), as_of, out_path)
        return result


def run_prediction() -> pd.DataFrame:
    """Entry point for the inference stage."""
    return Predictor().predict()


if __name__ == "__main__":
    df = run_prediction()
    log.info("\n%s", df.to_string(index=False))
