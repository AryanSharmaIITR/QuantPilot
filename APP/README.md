# QuantPilot Web App

A FastAPI control panel for the QuantPilot pipeline. It lets you manage the
instrument universe, run pipeline stages, and view predictions from the browser.

## Features

- **Dashboard** — model name, universe size, checkpoint status, prediction freshness,
  live job state, and a compact view of the latest predictions.
- **Stocks** — add / remove stocks and market indices (name + ticker). Changes are
  persisted to `tickers.json` at the project root, which the pipeline reads.
- **Pipeline** — run any stage (`ingest`, `preprocess`, `train`, `predict`) or a full
  pipeline (`train-pipeline`, `predict-pipeline`) as a background job, with a live log
  view. Only one job runs at a time (stages share on-disk data dirs).
- **Predictions** — view the latest `predictions.csv` as a table with UP/DOWN signals
  and probability bars. Pick **any trading day** (past or the next session) and run a
  fresh, date-anchored predict pipeline for it.
- **AI Advisor** 🤖 — an agentic [LangGraph](https://langchain-ai.github.io/langgraph/)
  workflow that reads the predictions, pulls live market & per-stock news from
  [Tavily](https://tavily.com), and uses a free-tier LLM (Groq / Gemini) to draft
  **three budget-aware investment plans** — high-risk, low-risk, and optimal.

---

## How the project works (end to end)

QuantPilot is a four-stage ML pipeline with a web control panel and an AI advisor on
top. The data flows in one direction:

```
1. INGEST       yfinance → raw OHLC for 29 stocks + 11 market indices
      ↓
2. PREPROCESS   feature engineering (RSI, MACD, volatility, Sharpe/Sortino, …)
      ↓
3. TRAIN        2 Transformers (stock + market) → fusion → NN head → XGBoost
      ↓
4. PREDICT      next-day UP/DOWN signal + up-probability per stock → predictions.csv
      ↓
5. AI ADVISOR   predictions.csv + Tavily news + LLM → 3 budget-aware plans
```

The **web app does not contain ML logic** — it shells out to the pipeline
(`python signals/pipeline.py <stage>`) as background subprocesses and reads the
resulting files (`predictions.csv`, `tickers.json`). The **AI Advisor** lives in
`signals/advisor.py` and is called in-process by the API.

---

## How each feature works

### 📊 Dashboard
Polls `GET /api/status` every few seconds and renders status cards (model name,
#stocks, #indices, checkpoint Ready/Missing, prediction freshness, pipeline busy/idle)
plus a "Quick run" launcher and the latest predictions. The "Model Name" card shows the
`project.name` from `config.yaml`.

### 📈 Stocks
`GET/POST/DELETE /api/stocks` reads & writes `tickers.json` via `registry.py`. The order
of instruments defines the model's category indices, so **changing the universe changes
the model's input shape** (see the retraining warning below).

### ⚙️ Pipeline
`POST /api/pipeline/run` launches a stage as a `python signals/pipeline.py <stage>`
subprocess (`jobs.py`), streaming stdout to a per-job log file. The UI polls
`GET /api/pipeline/jobs[/{id}/log]` to show status and a live log tail. Only one job runs
at a time because stages share on-disk data directories.

### 🔮 Predictions
`GET /api/predictions` returns the latest `predictions.csv` as JSON. The date picker is
driven by `GET /api/predictions/dates`, which exposes the selectable range (back across
previous trading days, up to today/the next NSE open). When you pick a date and run the
predict pipeline, ingestion fetches **one month of history ending the session before that
date**, so you can predict any past trading day — not just the most recent window.

### 🤖 AI Advisor
`POST /api/advisor/plan` (with a `budget`) runs the LangGraph agent in
`signals/advisor.py`. The graph has four nodes:

```
load_predictions → fetch_news → draft_plans → validate_allocations
```

1. **load_predictions** — reads `predictions.csv`, sorted by up-probability.
2. **fetch_news** — Tavily: 5 broad-market headlines + 2 per stock (concurrent).
   Skipped gracefully if no Tavily key.
3. **draft_plans** — feeds the signals, the model's accuracy stats, and the news to the
   LLM, which returns three plans as JSON. Each plan has a strict up-probability cutoff
   (aggressive `>0.5`, conservative `>0.6`, optimal `>0.4`).
4. **validate_allocations** — the LLM proposes per-stock amounts; **code** validates and
   re-normalises them so every plan sums *exactly* to your budget (the model is never
   trusted to do the arithmetic).

`GET /api/advisor/status` reports readiness (deps installed, LLM/Tavily keys present) so
the UI can guide you. Everything degrades gracefully: no Tavily key → plans without news;
no LLM key/deps → a clear error.

## Run

From the project root (`QuantPilot/`):

```bash
pip install -r requirements.txt          # installs fastapi + uvicorn too

# option A — convenience launcher
python APP/run.py                        # http://127.0.0.1:8000
RELOAD=true PORT=8080 python APP/run.py   # dev mode with auto-reload

# option B — uvicorn directly
uvicorn APP.main:app --reload --port 8000
```

Open http://127.0.0.1:8000 and the API docs at http://127.0.0.1:8000/docs.

## AI Advisor setup (optional)

The advisor needs a (free-tier) LLM key, and optionally a Tavily key for news.
Add them to a `.env` file at the project root (or export them) — the app reads it
automatically:

```bash
# .env  (project root)
GROQ_API_KEY=your_groq_key       # free key: https://console.groq.com
TAVILY_API_KEY=your_tavily_key   # optional; free key: https://tavily.com
```

Provider, model, and news behaviour are configured under `agent:` in `config.yaml`:

```yaml
agent:
  currency: "INR"
  llm:
    provider: groq               # groq | gemini | openai | anthropic
    model: openai/gpt-oss-120b   # groq-hosted; gemini e.g. gemini-2.0-flash
    max_tokens: 3000
  news:
    enabled: true
    per_stock_results: 2         # news items per stock (shown in UI)
    market_results: 5            # broad-market headlines
    prompt_max_stocks: 12        # top-N signals' news fed to the LLM (0 = all)
```

> ⚠️ Free-tier note: Groq limits tokens-per-minute (~12k). The prompt is trimmed
> (`prompt_max_stocks`, `prompt_snippet_chars`) to stay under it; if you hit a
> rate limit, wait ~30–60s between runs or switch `provider` to `gemini`.

## ⚠️ Universe changes require retraining

The instrument universe defines the model's input dimensions and category order.
If you add or remove stocks/indices, the existing checkpoint no longer matches —
you must run the **train pipeline** before `predict` will work, otherwise inference
fails with a dimension mismatch.

## Architecture

```
APP/
├── __init__.py     # wires signals/ onto sys.path
├── main.py         # FastAPI app + routes + static serving
├── registry.py     # tickers.json read/write (the "add stocks" backend)
├── jobs.py         # subprocess pipeline runner + in-memory job tracking
├── schemas.py      # Pydantic request/response models (incl. AdvisorRequest)
├── run.py          # convenience launcher
├── static/         # index.html, style.css, app.js (vanilla SPA)
└── logs/           # per-job run logs (gitignored)

signals/
└── advisor.py      # 🤖 AI Advisor — LangGraph agent (called in-process by main.py)
```

Pipeline stages run as `python signals/pipeline.py <stage>` subprocesses, so the
web process stays responsive and heavy work is isolated. The **AI Advisor**, by
contrast, runs in-process (it's fast and I/O-bound on the news + LLM calls).

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | Dashboard summary |
| GET/POST/DELETE | `/api/stocks…` | Manage the instrument universe |
| POST | `/api/pipeline/run` | Start a pipeline stage (background job) |
| GET | `/api/pipeline/jobs[/{id}/log]` | Job status / live log |
| GET | `/api/predictions` | Latest predictions as JSON |
| GET | `/api/predictions/dates` | Selectable prediction-date range |
| GET | `/api/advisor/status` | Advisor readiness (deps + keys) |
| POST | `/api/advisor/plan` | Draft 3 budget-aware investment plans |
