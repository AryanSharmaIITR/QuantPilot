"""Stage 1 — financial data ingestion via yfinance.

Downloads daily OHLC history for the market-index and stock universe, derives
the scaled intraday return, and writes one CSV per instrument.

Two modes:
  * ``train``   — long history window (``data.train_timeperiod``), written to the
                  training raw dirs.
  * ``predict`` — short window (``data.predict_timeperiod``), written to the
                  live-prediction raw dirs.

Downloads are retried with linear backoff to tolerate transient yfinance/network
errors — important for unattended (scheduled) runs.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import yfinance as yf

import data as D
from config import CONFIG
from logger import get_logger

log = get_logger("ingest")


class DataIngestion:
    """Download and persist raw price data for a given pipeline mode."""

    def __init__(self, mode: str = "train"):
        if mode not in ("train", "predict"):
            raise ValueError(f"mode must be 'train' or 'predict', got {mode!r}")
        self.mode = mode
        self.timeperiod = D.timeperiod if mode == "train" else D.PREDICT_TIMEPERIOD
        if mode == "train":
            self.market_dir = D.RAW_DIR_NSE
            self.stock_dir = D.RAW_DIR_STOCK
        else:
            self.market_dir = D.PREDICT_RAW_DIR_NSE
            self.stock_dir = D.PREDICT_RAW_DIR_STOCK

        ing = CONFIG["ingestion"]
        self.max_retries = ing["max_retries"]
        self.retry_delay = ing["retry_delay_seconds"]

    # -- internals -----------------------------------------------------------
    def _download(self, name: str, ticker: str) -> pd.DataFrame | None:
        """Fetch history for one ticker, retrying transient failures."""
        for attempt in range(1, self.max_retries + 1):
            try:
                hist = yf.Ticker(ticker).history(period=self.timeperiod)
                if hist.empty:
                    log.warning("No data returned for %s (%s)", name, ticker)
                    return None
                return hist
            except Exception as exc:  # network / yfinance hiccup
                if attempt == self.max_retries:
                    log.error("%s (%s) failed after %d attempts: %s",
                              name, ticker, self.max_retries, exc)
                    return None
                wait = self.retry_delay * attempt
                log.warning("%s (%s) attempt %d/%d failed (%s); retrying in %ds",
                            name, ticker, attempt, self.max_retries, exc, wait)
                time.sleep(wait)
        return None

    def _fetch_universe(self, universe: dict[str, str], out_dir: str, label: str) -> int:
        """Download every ticker in ``universe`` into ``out_dir``. Returns count saved."""
        os.makedirs(out_dir, exist_ok=True)
        saved = 0
        for name, ticker in universe.items():
            hist = self._download(name, ticker)
            if hist is None:
                continue

            df = pd.DataFrame(hist).reset_index()
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            df["day_return_scaled"] = ((df["Close"] - df["Open"]) / df["Open"]) * 1000
            df = df[["Date", "day_return_scaled"]]

            file_path = os.path.join(out_dir, f"{D.sanitize(name)}.csv")
            df.to_csv(file_path, index=False)
            saved += 1
            log.info("%s %s downloaded (%s)", name, ticker, label)
        return saved

    # -- public API ----------------------------------------------------------
    def get_market_data(self) -> int:
        return self._fetch_universe(D.nse_tickers, self.market_dir, "market")

    def get_stock_data(self) -> int:
        return self._fetch_universe(D.stocks_tickers, self.stock_dir, "stock")

    def run(self) -> None:
        log.info("=== Ingestion (mode=%s, window=%s) ===", self.mode, self.timeperiod)
        n_market = self.get_market_data()
        n_stock = self.get_stock_data()
        log.info("Ingestion complete: %d/%d market, %d/%d stock instruments saved",
                 n_market, len(D.nse_tickers), n_stock, len(D.stocks_tickers))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QuantPilot data ingestion")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    args = parser.parse_args()
    DataIngestion(mode=args.mode).run()
