<div align="center">

# ◤ QuantPilot

### AI-powered quantitative market intelligence

*Transformer-based temporal modeling · cross-market attention · hybrid deep learning + gradient boosting — for directional stock-movement prediction.*

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-017CEE?logo=xgboost&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-8A2BE2?logo=optuna&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

[Overview](#overview) · [Quickstart](#quickstart) · [Pipeline](#pipeline-architecture) · [Web App](#web-app) · [Automation](#automation)

</div>

---

# Quickstart

```bash
git clone https://github.com/AryanSharmaIITR/QuantPilot.git && cd QuantPilot
pip install -r requirements.txt

python signals/pipeline.py predict-pipeline   # ingest → features → predict
python APP/run.py                             # launch the web UI → http://127.0.0.1:8000
```

---

# Overview

QuantPilot is a hybrid quantitative AI framework designed for financial time-series modeling and directional stock movement prediction.

The system combines:

- transformer-based sequential representation learning
- cross-market contextual aggregation
- technical indicator engineering
- multi-source financial data ingestion
- hybrid neural + XGBoost prediction pipelines

to model both stock-specific behavior and broader market dynamics.

The project focuses on learning temporal relationships between:
- Indian equities
- market indices
- volatility indicators
- forex instruments
- commodity futures

to improve predictive robustness in financial markets.

---

# Features

- Automated financial data ingestion
- Multi-asset market modeling
- Transformer-based temporal encoders
- Cross-market attention aggregation
- Hybrid deep learning + XGBoost pipeline
- Advanced technical indicator engineering
- Risk-aware financial feature generation
- End-to-end training pipeline
- Multi-stage model pretraining
- Classification-based directional prediction

---

# Pipeline Architecture

```text
Market Data Sources
        ↓
Financial Data Ingestion
        ↓
Feature Engineering
        ↓
Sequence Construction
        ↓
Transformer Encoders
        ↓
Market Aggregation
        ↓
Cross-Market Fusion
        ↓
Feature Extraction
        ↓
XGBoost Classifier
        ↓
Directional Prediction
```

---

# Workflow

1. Market Data Collection
2. Data Cleaning & Alignment
3. Technical Indicator Engineering
4. Sequential Dataset Construction
5. Transformer Pretraining
6. Market Context Aggregation
7. Cross-Market Fusion
8. XGBoost Training
9. Evaluation & Inference

---

# Market Coverage

QuantPilot integrates multiple financial instruments including:

## Market Indices
- NIFTY 50
- NIFTY BANK
- NIFTY 100
- NIFTY 200
- NIFTY 500
- INDIA VIX
- BSE Sensex

## Commodities & Forex
- Gold Futures
- Crude Oil Futures
- USDINR

## Equities
- Reliance Industries
- TCS
- Infosys
- HDFC Bank
- ICICI Bank
- SBI
- Tata Steel
- Adani Enterprises
- Bajaj Finance
- Asian Paints
- and multiple large-cap Indian equities

---

# Feature Engineering

The preprocessing pipeline generates advanced technical and risk-sensitive indicators including:

- RSI
- MACD
- EMA Slope
- ATR
- Bollinger Band Width
- Z-Score
- Rolling Volatility
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Value-at-Risk (VaR)
- Conditional VaR (CVaR)
- Beta Estimation
- Alpha Estimation
- Return Momentum Features

---

# Model Architecture

## 1. Temporal Transformer Encoder

Transformer-based encoders are used for:
- sequential market representation learning
- temporal dependency extraction
- multi-timestep feature encoding

Separate encoders are trained for:
- market indices
- stock-specific data

---

## 2. Market Aggregator

A cross-attention aggregation module captures:
- inter-market relationships
- global market context
- cross-index dependencies

The aggregator produces a unified market representation vector.

---

## 3. Fusion Module

A gated fusion architecture combines:
- stock embeddings
- market embeddings
- interaction representations
- attention-derived contextual information

to generate market-aware stock representations.

---

## 4. XGBoost Classifier

The extracted deep representations are passed into an XGBoost classifier for final directional prediction.

The hybrid architecture improves:
- generalization
- classification robustness
- nonlinear decision modeling
- tabular feature learning

---

# Training Strategy

The training pipeline is divided into multiple stages:

## Phase 1 — Transformer Pretraining
Temporal transformer encoders are pretrained using sequential financial data.

## Phase 2 — Neural Representation Learning
Cross-market aggregation and fusion modules are trained using supervised objectives.

## Phase 3 — Feature Extraction
Learned embeddings are extracted from pretrained neural networks.

## Phase 4 — XGBoost Optimization
Extracted features are used to train an XGBoost classifier for final prediction.

---

# Evaluation Metrics

The model is evaluated using:

- Accuracy: 0.6984
- Precision: 0.6657
- Recall: 0.7183
- F1 Score: 0.6910
- ROC-AUC: 0.6995

---

# Tech Stack

## Languages & Frameworks
- Python
- PyTorch
- XGBoost
- Scikit-learn

## Data & Analytics
- Pandas
- NumPy
- yFinance

## Optimization & Experimentation
- Optuna

---

# Project Structure

```text
QuantPilot/
│
├── 🧠 signals/                    # Core ML pipeline (the engine room)
│   ├── __init__.py
│   ├── config.py                 # loads config.yaml, resolves paths
│   ├── logger.py                 # structured logging
│   ├── data.py                   # ticker universe + config-derived constants
│   ├── dataIngestion.py          # Stage 1 · yfinance download (train/predict)
│   ├── dataPreprocessing.py      # Stage 2 · feature engineering (train/predict)
│   ├── loadData.py               # sequence Dataset + dataloaders
│   ├── models.py                 # transformer · aggregator · fusion · head
│   ├── training.py               # Stage 3 · train + evaluate  (run_training)
│   ├── results.py                # Stage 4 · inference          (run_prediction)
│   ├── pipeline.py               # ⚙️  CLI orchestrator (entry point)
│   └── stack_stock_data.py
│
├── 🌐 APP/                        # FastAPI web control panel
│   ├── __init__.py               # wires signals/ onto sys.path
│   ├── main.py                   # FastAPI app + routes + static serving
│   ├── registry.py               # tickers.json read/write (add-stocks backend)
│   ├── jobs.py                   # subprocess pipeline runner + job tracking
│   ├── schemas.py                # Pydantic request/response models
│   ├── run.py                    # convenience launcher (python APP/run.py)
│   ├── README.md
│   ├── static/                   # vanilla SPA — no build step
│   │   ├── index.html
│   │   ├── style.css
│   │   └── app.js
│   └── logs/                     # per-job run logs (gitignored)
│
├── 🤖 .github/workflows/
│   └── pipeline.yml              # scheduled daily inference automation
│
├── 📦 data/                       # training data (gitignored)
│   ├── raw/{market_data, stock_data}/
│   └── preprocessed/{market_data, stock_data}/{train, val, test}/
│
├── 🔮 Data_For_Prediction/        # live-inference data (gitignored)
│   ├── market_data/  ·  stock_data/
│   └── preprocessed_market_data/  ·  preprocessed_stock_data/
│
├── 💾 artifacts/                  # trained model checkpoints
│   ├── final_market_pipeline_nn_model.pth      # combined NN (used for inference)
│   ├── final_market_pipeline_market_model.pth
│   ├── final_market_pipeline_stock_model.pth
│   └── final_market_pipeline_xgb.json
│
├── 📓 notebook/                   # research notebooks + training curves
│
├── ⚙️  config.yaml                # single source of truth for the whole pipeline
├── 🎯 tickers.json               # editable instrument universe (managed by APP)
├── 📄 predictions.csv            # latest directional signals (output)
├── 📋 requirements.txt
└── 📖 README.md
```

---

# Installation

```bash
git clone https://github.com/AryanSharmaIITR/QuantPilot.git

cd QuantPilot

pip install -r requirements.txt
```

---

# Configuration

All pipeline behaviour is driven by a single [config.yaml](config.yaml) at the
project root — ticker windows, filesystem paths, train/val/test splits, model
hyperparameters, and the inference checkpoint. Paths are auto-resolved relative
to the project root, so commands work from any directory. Point at an alternate
config with the `QUANTPILOT_CONFIG` environment variable.

---

# Usage

The pipeline is orchestrated through a single CLI entry point,
[signals/pipeline.py](signals/pipeline.py). Run every command from the project
root.

```bash
# Full training pipeline: ingest (3y) -> feature engineering -> train -> evaluate
python signals/pipeline.py train-pipeline

# Daily inference pipeline: ingest (1mo) -> feature engineering -> predict
python signals/pipeline.py predict-pipeline
```

Individual stages can also be run on their own:

```bash
python signals/pipeline.py ingest --mode train      # or --mode predict
python signals/pipeline.py preprocess --mode train  # or --mode predict
python signals/pipeline.py train
python signals/pipeline.py predict
```

Predictions are written to `predictions.csv` (stock, ticker, as-of date,
up-probability, and an UP/DOWN signal).

---

# Web App

A FastAPI control panel ([APP/](APP/)) provides a browser UI to manage the
instrument universe, run pipeline stages as background jobs (with live logs), and
view predictions.

```bash
python APP/run.py              # http://127.0.0.1:8000  (API docs at /docs)
# or: uvicorn APP.main:app --reload
```

- **Stocks** — add/remove stocks & market indices (name + ticker); persisted to
  `tickers.json`, which the pipeline reads.
- **Pipeline** — trigger `ingest` / `preprocess` / `train` / `predict` (or the full
  pipelines) and watch the job log stream.
- **Predictions** — view the latest signals as a table.

> Changing the instrument universe alters the model's input size — retrain before
> predicting. See [APP/README.md](APP/README.md) for details.

---

# Automation

A scheduled GitHub Actions workflow
([.github/workflows/pipeline.yml](.github/workflows/pipeline.yml)) runs the
inference pipeline every weekday after market close. It regenerates the
short-window prediction data and runs against the git-tracked NN checkpoint
(`artifacts/final_market_pipeline_nn_model.pth`), so no GPU or retraining is
needed in CI. The resulting `predictions.csv` is uploaded as a build artifact
(committing it back to the repo is available as an opt-in step). A manual
`workflow_dispatch` run can trigger the full training pipeline instead.

---

# Disclaimer

This project is intended for research and educational purposes only.

It should not be considered financial advice or an automated trading recommendation system.

---

# Author

Aryan Sharma  
IIT Roorkee