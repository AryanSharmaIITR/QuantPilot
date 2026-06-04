"""Stage 1 — financial data ingestion via yfinance.

Downloads daily OHLC history for the market-index and stock universe, derives
the scaled intraday return, and writes one CSV per instrument.

Two modes:
  * ``train``   — long history window (``data.train_timeperiod``), written to the
                  training raw dirs.
  * ``predict`` — short window (``data.predict_timeperiod``, default 2 months),
                  written to the live-prediction raw dirs. When an ``end_date``
                  (the requested prediction date) is supplied, the window is
                  anchored to it: ``predict_timeperiod`` of history ending the
                  session BEFORE ``end_date``, so any past trading day can be
                  predicted. Without it, the window simply ends today.

Downloads are retried with linear backoff to tolerate transient yfinance/network
errors — important for unattended (scheduled) runs.
"""
from __future__ import annotations

import os
import re
import time

import pandas as pd
import yfinance as yf

import data as D
from config import CONFIG
from logger import get_logger

log = get_logger("ingest")


def _period_to_offset(period: str) -> pd.DateOffset:
    """Convert a yfinance period string (e.g. '2mo', '1y', '45d') to a DateOffset.

    Used to anchor the predict window: start = end_date - offset. Falls back to
    2 months for an unrecognised value.
    """
    m = re.fullmatch(r"\s*(\d+)\s*(d|wk|mo|y)\s*", str(period).lower())
    if not m:
        return pd.DateOffset(months=2)
    n, unit = int(m.group(1)), m.group(2)
    return {
        "d": pd.DateOffset(days=n),
        "wk": pd.DateOffset(weeks=n),
        "mo": pd.DateOffset(months=n),
        "y": pd.DateOffset(years=n),
    }[unit]


class DataIngestion:
    """Download and persist raw price data for a given pipeline mode."""

    def __init__(self, mode: str = "train", end_date: str | None = None):
        if mode not in ("train", "predict"):
            raise ValueError(f"mode must be 'train' or 'predict', got {mode!r}")
        self.mode = mode
        # Only the predict window can be anchored to a requested date.
        self.end_date = end_date if mode == "predict" else None
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
    def _history(self, ticker: str) -> pd.DataFrame:
        """Pull raw OHLC history for one ticker.

        With an anchored ``end_date`` (predict mode), fetch ``predict_timeperiod``
        of history (default 2 months) ending the day before it — yfinance's
        ``end`` is exclusive, so the last bar is the session immediately preceding
        the requested prediction date. Otherwise fall back to the configured
        trailing ``period`` (ends today).
        """
        tk = yf.Ticker(ticker)
        if self.end_date:
            end = pd.Timestamp(self.end_date).normalize()
            start = (end - _period_to_offset(self.timeperiod)).strftime("%Y-%m-%d")
            return tk.history(start=start, end=end.strftime("%Y-%m-%d"))
        return tk.history(period=self.timeperiod)

    def _download(self, name: str, ticker: str) -> pd.DataFrame | None:
        """Fetch history for one ticker, retrying transient failures."""
        for attempt in range(1, self.max_retries + 1):
            try:
                hist = self._history(ticker)
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
        window = f"{self.timeperiod} ending {self.end_date}" if self.end_date else self.timeperiod
        log.info("=== Ingestion (mode=%s, window=%s) ===", self.mode, window)
        n_market = self.get_market_data()
        n_stock = self.get_stock_data()
        log.info("Ingestion complete: %d/%d market, %d/%d stock instruments saved",
                 n_market, len(D.nse_tickers), n_stock, len(D.stocks_tickers))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QuantPilot data ingestion")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--end-date", dest="end_date", default=None,
                        help="Anchor the predict window to end the session before "
                             "this date (ISO). Default: trailing window ending today.")
    args = parser.parse_args()
    DataIngestion(mode=args.mode, end_date=args.end_date).run()
