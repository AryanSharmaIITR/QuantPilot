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

        # Normalize per-feature using the TRAIN-only stats saved at training time.
        # This both matches training exactly AND avoids any look-ahead — the live/
        # backtest window is never normalized with its own (partly future) stats.
        stats = self._load_train_stats()
        if stats is not None:
            m_mean, m_std, s_mean, s_std = stats
        else:
            log.warning(
                "Train normalization stats not found (%s) — falling back to "
                "window stats. Retrain to generate them and remove this leakage.",
                D.NORM_STATS_PATH,
            )
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

    @staticmethod
    def _load_train_stats():
        """Load persisted TRAIN-only normalization stats, or None if absent."""
        if not os.path.exists(D.NORM_STATS_PATH):
            return None
        z = np.load(D.NORM_STATS_PATH)
        return z["market_mean"], z["market_std"], z["stock_mean"], z["stock_std"]

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

    def predict(self, target_date: str | None = None) -> pd.DataFrame:
        from market_calendar import next_market_open

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
        common = ds.common_dates
        if len(common) < self.seq_len:
            raise ValueError("Not enough history to build a prediction window")

        next_open = next_market_open(common[-1])

        # ------------------------------------------------------------------
        # Resolve the target trading day (the day the signal is FOR).
        #   - Forward/default: next NSE open after the latest data session.
        #   - Backtest: a past session in common_dates, predicting from the
        #     seq_len sessions ending the day before it.
        # ------------------------------------------------------------------
        if not target_date:
            market_data, stock_data = ds.latest_window()
            cutoff = common[-1]
            target = next_open
        else:
            picked = str(target_date)
            if picked >= next_open or picked > common[-1]:
                # At/after the next open (or beyond known data) -> forward.
                market_data, stock_data = ds.latest_window()
                cutoff = common[-1]
                target = next_open
            else:
                # Snap to the largest available session <= picked.
                idx = None
                for i, d in enumerate(common):
                    if d <= picked:
                        idx = i
                    else:
                        break
                if idx is None:
                    raise ValueError(
                        f"target_date {picked} is before the earliest available session"
                    )
                if idx < self.seq_len:
                    raise ValueError(f"not enough history before {common[idx]}")
                market_data, stock_data = ds[idx - self.seq_len]
                cutoff = common[idx - 1]
                target = common[idx]

        as_of = cutoff
        log.info("Predicting for target_date=%s (data cutoff=%s, %d common dates)",
                 target, cutoff, len(common))

        probs = self._forward(market_data, stock_data).cpu().numpy()
        labels = (probs > self.threshold).astype(int)

        result = pd.DataFrame({
            "Date": as_of,  # last date of the input window the prediction is based on
            "stock": list(D.stocks_cat.keys()),
            "ticker": list(D.stocks_tickers.values()),
            "as_of_date": as_of,
            "target_date": target,
            "up_probability": np.round(probs, 4),
            "signal": np.where(labels == 1, "UP", "DOWN"),
        })

        out_path = D.PREDICTIONS_PATH
        result.to_csv(out_path, index=False)
        log.info("Wrote %d predictions (target %s, as of %s) to %s",
                 len(result), target, as_of, out_path)
        return result


def run_prediction(target_date: str | None = None) -> pd.DataFrame:
    """Entry point for the inference stage."""
    return Predictor().predict(target_date)


if __name__ == "__main__":
    df = run_prediction()
    log.info("\n%s", df.to_string(index=False))
