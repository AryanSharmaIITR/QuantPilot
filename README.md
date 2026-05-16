# QuantPilot

AI-powered quantitative market intelligence framework for stock movement prediction using transformer-based temporal modeling, cross-market attention mechanisms, and hybrid deep learning + gradient boosting architectures.

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
├── data/
│   ├── raw/
|   |    ├── market_data/
|   |    └── stock_data/
│   └── preprocessed/
|        ├── market_data/
|        └── stock_data/
|
│
├──Data_For_Prediction/
|   ├── market_data/
|   ├── stock_data/
|   ├── preprocessed_market_data/
|   └── preprocessed_stock_data/
|
├── artifacts/
|      ├── final_market_pipeline_nn.pt
|      ├── final_market_pipeline_market_model.pth
|      ├── final_market_pipeline_stock_model.pth
|      └── final_market_pipeline_xgb.json
│
├── notebooks/
|      ├── datacleaning.ipynb
|      ├── preprocessing.ipynb
|      ├── test1.ipynb
|      ├── test2.ipynb
|      ├── test4.ipynb
|      ├── test5.ipynb
|      ├── test6.ipynb
|      └── training_curves.png
│
├── signals/
|   ├── init.py
│   ├── dataIngestion.py
│   ├── dataPreprocessing.py
│   ├── loadData.py
│   ├── models.py
│   ├── training.py
│   ├── results.py
│   └── stack_stock_data.py
│
├── requirements.txt
│
└── README.md
```

---

# Installation

```bash
git clone https://github.com/your-username/QuantPilot.git

cd QuantPilot

pip install -r requirements.txt
```

---

# Data Ingestion

```bash
python src/dataIngestion.py
```

---

# Data Preprocessing

```bash
python src/dataPreprocessing.py
```

---

# Training

```bash
python src/training.py
```

---

# Disclaimer

This project is intended for research and educational purposes only.

It should not be considered financial advice or an automated trading recommendation system.

---

# Author

Aryan Sharma  
IIT Roorkee