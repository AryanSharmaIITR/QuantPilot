"""QuantPilot FastAPI application.

Endpoints
---------
  GET    /                          -> web UI (static SPA)
  GET    /api/health                -> liveness probe
  GET    /api/status                -> dashboard summary
  GET    /api/stocks                -> list instrument universe
  POST   /api/stocks                -> add an instrument
  DELETE /api/stocks/{kind}/{name}  -> remove an instrument
  POST   /api/stocks/reset          -> reset universe to defaults
  POST   /api/pipeline/run          -> start a pipeline stage (background)
  GET    /api/pipeline/jobs         -> list jobs
  GET    /api/pipeline/jobs/{id}    -> job status
  GET    /api/pipeline/jobs/{id}/log-> job log tail
  POST   /api/pipeline/jobs/{id}/stop
  GET    /api/predictions           -> latest predictions.csv as JSON
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import ROOT_DIR
from . import jobs, registry
from .schemas import AddInstrument, JobStartRequest

import data as D  # signals/data.py

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="QuantPilot",
    description="Web control panel for the QuantPilot quant ML pipeline.",
    version="1.0.0",
)


@app.on_event("startup")
def _startup() -> None:
    registry.ensure_seeded()


# ----------------------------------------------------------------------------
# System / dashboard
# ----------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict:
    universe = registry.list_instruments()
    checkpoint = D.CONFIG["inference"]["nn_checkpoint"]
    predictions_path = D.PREDICTIONS_PATH
    pred_count = 0
    pred_as_of = None
    if os.path.exists(predictions_path):
        rows = _read_predictions()
        pred_count = len(rows)
        pred_as_of = rows[0]["as_of_date"] if rows else None
    return {
        "project": D.CONFIG["project"]["name"],
        "num_stocks": len(universe["stocks"]),
        "num_market": len(universe["market"]),
        "checkpoint": checkpoint,
        "checkpoint_exists": os.path.exists(checkpoint),
        "predictions_exist": os.path.exists(predictions_path),
        "predictions_count": pred_count,
        "predictions_as_of": pred_as_of,
        "busy": jobs.is_busy(),
    }


# ----------------------------------------------------------------------------
# Instrument universe
# ----------------------------------------------------------------------------
@app.get("/api/stocks")
def get_stocks() -> dict:
    return registry.list_instruments()


@app.post("/api/stocks")
def add_stock(payload: AddInstrument) -> dict:
    try:
        registry.add_instrument(payload.kind, payload.name, payload.ticker)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return registry.list_instruments()


@app.delete("/api/stocks/{kind}/{name}")
def remove_stock(kind: str, name: str) -> dict:
    try:
        registry.remove_instrument(kind, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"'{name}' not found in {kind}")
    return registry.list_instruments()


@app.post("/api/stocks/reset")
def reset_stocks() -> dict:
    registry.reset_to_defaults()
    return registry.list_instruments()


# ----------------------------------------------------------------------------
# Pipeline control
# ----------------------------------------------------------------------------
@app.post("/api/pipeline/run")
def run_pipeline(req: JobStartRequest) -> dict:
    try:
        return jobs.start_job(req.stage, req.mode)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/pipeline/jobs")
def list_jobs() -> list[dict]:
    return jobs.list_jobs()


@app.get("/api/pipeline/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/pipeline/jobs/{job_id}/log")
def job_log(job_id: str, tail: int = 400) -> dict:
    log = jobs.read_log(job_id, tail=tail)
    if log is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"id": job_id, "log": log}


@app.post("/api/pipeline/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict:
    job = jobs.stop_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ----------------------------------------------------------------------------
# Predictions
# ----------------------------------------------------------------------------
def _read_predictions() -> list[dict]:
    path = D.PREDICTIONS_PATH
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@app.get("/api/predictions")
def get_predictions() -> dict:
    rows = _read_predictions()
    return {"count": len(rows), "predictions": rows}


# ----------------------------------------------------------------------------
# Static UI (mounted last so it doesn't shadow /api routes)
# ----------------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
