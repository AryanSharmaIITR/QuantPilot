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
    as_of_date: Optional[str] = None


class Prediction(BaseModel):
    stock: str
    ticker: str
    as_of_date: str
    target_date: Optional[str] = None
    up_probability: float
    signal: str


class AdvisorRequest(BaseModel):
    budget: float = Field(..., gt=0, examples=[100000],
                          description="Amount to invest, in the configured currency")
    currency: Optional[str] = None
    include_news: Optional[bool] = None
    max_stocks: Optional[int] = Field(None, ge=0,
                                      description="Cap stocks to fetch news for (0/None = all)")
