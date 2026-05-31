"""Pydantic request/response models for the QuantPilot API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Instrument(BaseModel):
    name: str
    ticker: str
    category: int


class InstrumentList(BaseModel):
    stocks: list[Instrument]
    market: list[Instrument]


class AddInstrument(BaseModel):
    kind: Literal["stocks", "market"] = "stocks"
    name: str = Field(..., min_length=1, examples=["Tata Motors"])
    ticker: str = Field(..., min_length=1, examples=["TATAMOTORS.NS"])


class JobStartRequest(BaseModel):
    stage: Literal[
        "ingest", "preprocess", "train", "predict",
        "train-pipeline", "predict-pipeline",
    ]
    mode: Optional[Literal["train", "predict"]] = None


class Prediction(BaseModel):
    stock: str
    ticker: str
    as_of_date: str
    up_probability: float
    signal: str
