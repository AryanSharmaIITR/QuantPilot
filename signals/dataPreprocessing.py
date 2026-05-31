"""Stage 2 — feature engineering & dataset construction.

Reads the raw per-instrument CSVs, engineers technical/risk indicators, cleans
NaN/inf, and writes preprocessed CSVs.

Two modes:
  * ``train``   — chronological train/val/test split (per config ``data.splits``)
                  written under ``preprocessed_{market,stock}/{train,val,test}/``.
  * ``predict`` — single flat directory (no split), for the live inference run.

The feature set and ordering are intentionally identical across modes so the
trained model sees the same ``dim``-wide vectors at inference time.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import data as D
from logger import get_logger

log = get_logger("preprocess")


class Preprocessing:
    """Engineer features for the market and stock universes."""

    def __init__(self, mode: str = "train"):
        if mode not in ("train", "predict"):
            raise ValueError(f"mode must be 'train' or 'predict', got {mode!r}")
        self.mode = mode
        self.epsilon = 1e-8
        if mode == "train":
            self.raw_market, self.raw_stock = D.RAW_DIR_NSE, D.RAW_DIR_STOCK
            self.out_market = D.PREPROCESSED_DIR_NSE
            self.out_stock = D.PREPROCESSED_DIR_STOCK
        else:
            self.raw_market = D.PREDICT_RAW_DIR_NSE
            self.raw_stock = D.PREDICT_RAW_DIR_STOCK
            self.out_market = D.PREDICT_PREPROCESSED_DIR_NSE
            self.out_stock = D.PREDICT_PREPROCESSED_DIR_STOCK

    # -- feature engineering -------------------------------------------------
    def set_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df

    def rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = df["day_return_scaled"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / (avg_loss + self.epsilon)
        return 100 - (100 / (1 + rs))

    def rolling_sortino(self, window) -> float:
        window = np.asarray(window)
        downside = window[window < 0]
        if len(downside) == 0:
            return 0.0
        downside_std = downside.std(ddof=1)
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0
        return window.mean() / downside_std

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        indicators = {}
        series = df["day_return_scaled"]
        returns_1d = series.pct_change()

        indicators["RSI"] = self.rsi(df)
        indicators["Sortino"] = series.rolling(window=14).apply(self.rolling_sortino, raw=False)

        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        indicators["MACD"] = ema26 - ema12

        ema15 = series.ewm(span=15, adjust=False).mean()
        indicators["emaslope_15d"] = ema15.diff()

        tr = series.diff().abs()
        indicators["tr"] = tr
        indicators["atr_15d"] = tr.rolling(15).mean()

        mean_15 = series.rolling(15).mean()
        std_15 = series.rolling(15).std()
        upper_15 = mean_15 + 2 * std_15
        lower_15 = mean_15 - 2 * std_15

        indicators["bb_width_15d"] = (upper_15 - lower_15) / (mean_15 + self.epsilon)
        indicators["zscore_15d"] = (series - mean_15) / (std_15 + self.epsilon)
        indicators["ret_15d"] = series.pct_change(15)

        roll_max_15 = series.rolling(15).max()
        drawdown_15 = series / roll_max_15 - 1
        indicators["max_dd_15d"] = drawdown_15.rolling(15).min()

        vol15 = series.rolling(15).std() * np.sqrt(252)
        mean15 = series.rolling(15).mean()
        rf_daily = 0.05 / 252

        indicators["vol_15d"] = vol15
        indicators["sharpe_ratio_15"] = (mean15 - rf_daily) * 15 / (vol15 + self.epsilon)
        indicators["var95_15"] = series.rolling(15).quantile(0.05)
        indicators["cvar95_15"] = series.rolling(15).apply(
            lambda x: x[x <= x.quantile(0.05)].mean() if len(x) > 0 else np.nan
        )
        indicators["beta_15"] = (
            returns_1d.rolling(15).cov(series) / (series.rolling(15).var() + self.epsilon)
        )
        indicators["alpha_5d"] = (
            returns_1d.rolling(5).mean() - indicators["beta_15"] * series.rolling(5).mean()
        )

        return pd.concat([df, pd.DataFrame(indicators)], axis=1)

    # -- per-instrument processing ------------------------------------------
    def _process_one(self, raw_dir: str, name: str, category: int, is_stock: bool) -> pd.DataFrame:
        file_path = os.path.join(raw_dir, f"{D.sanitize(name)}.csv")
        df = pd.read_csv(file_path)
        df = self.set_nan(df)
        df = self.add_technical_indicators(df)
        df = self.set_nan(df)
        if is_stock:
            # Next-day directional target (1 if next day's return > 0).
            df["target"] = (df["day_return_scaled"].shift(-1) > 0).astype(int)
        df["category"] = category
        return df.sort_values("Date").reset_index(drop=True)

    def _save_split(self, df: pd.DataFrame, out_dir: str, file_stem: str) -> None:
        """Chronologically split into train/val/test and persist (train mode)."""
        n = len(df)
        train_end = int(D.TRAIN_SPLIT * n)
        val_end = int(D.VAL_SPLIT * n)
        for split, frame in (
            ("train", df.iloc[:train_end]),
            ("val", df.iloc[train_end:val_end]),
            ("test", df.iloc[val_end:]),
        ):
            split_dir = os.path.join(out_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            frame.to_csv(os.path.join(split_dir, f"{file_stem}.csv"), index=False)

    def _process_universe(self, universe, cat_map, raw_dir, out_dir, is_stock, label):
        os.makedirs(out_dir, exist_ok=True)
        for name in universe.keys():
            df = self._process_one(raw_dir, name, cat_map[name], is_stock)
            stem = D.sanitize(name)
            if self.mode == "train":
                self._save_split(df, out_dir, stem)
            else:
                df.to_csv(os.path.join(out_dir, f"{stem}.csv"), index=False)
            log.info("%s %s preprocessed", name, label)

    # -- public API ----------------------------------------------------------
    def run(self) -> None:
        log.info("=== Preprocessing (mode=%s) ===", self.mode)
        self._process_universe(
            D.nse_tickers, D.nse_cat, self.raw_market, self.out_market,
            is_stock=False, label="NSE",
        )
        self._process_universe(
            D.stocks_tickers, D.stocks_cat, self.raw_stock, self.out_stock,
            is_stock=True, label="stock",
        )
        log.info("Preprocessing complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QuantPilot preprocessing")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    args = parser.parse_args()
    Preprocessing(mode=args.mode).run()
