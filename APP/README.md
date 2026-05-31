# QuantPilot Web App

A FastAPI control panel for the QuantPilot pipeline. It lets you manage the
instrument universe, run pipeline stages, and view predictions from the browser.

## Features

- **Dashboard** — universe size, checkpoint status, prediction freshness, live job state.
- **Stocks** — add / remove stocks and market indices (name + ticker). Changes are
  persisted to `tickers.json` at the project root, which the pipeline reads.
- **Pipeline** — run any stage (`ingest`, `preprocess`, `train`, `predict`) or a full
  pipeline (`train-pipeline`, `predict-pipeline`) as a background job, with a live log
  view. Only one job runs at a time (stages share on-disk data dirs).
- **Predictions** — view the latest `predictions.csv` as a table with UP/DOWN signals
  and probability bars, and trigger a fresh predict pipeline.

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
├── schemas.py      # Pydantic request/response models
├── run.py          # convenience launcher
├── static/         # index.html, style.css, app.js (vanilla SPA)
└── logs/           # per-job run logs (gitignored)
```

Pipeline stages run as `python signals/pipeline.py <stage>` subprocesses, so the
web process stays responsive and heavy work is isolated.
