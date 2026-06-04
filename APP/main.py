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
from .schemas import AddInstrument, AdvisorRequest, JobStartRequest

import advisor as ADV  # signals/advisor.py
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
        return jobs.start_job(req.stage, req.mode, req.as_of_date)
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


# How far back the date picker lets you reach. Predicting a past day re-fetches
# the configured history window (2 months) ending the session before it, so the
# range is bounded by what is sensible to request, not by what is on disk.
PREDICTION_HISTORY_YEARS = 5


@app.get("/api/predictions/dates")
def prediction_dates() -> dict:
    """Selectable prediction-date range.

    Upper bound / default is today (if a trading day) or the next NSE open;
    lower bound reaches back across previous trading days. Because each run
    re-ingests the configured history window ending the day before the chosen date, the
    range no longer depends on the data currently on disk. Always returns HTTP
    200 — the UI degrades gracefully (disables the picker) on any error.
    """
    try:
        import pandas as pd
        from market_calendar import first_market_open, is_trading_day, next_market_open

        today = pd.Timestamp.now().normalize().date().isoformat()
        # The furthest forward you can predict: today's session if it is a
        # trading day, otherwise the next one to open.
        upper = today if is_trading_day(today) else next_market_open(today)

        earliest = (
            pd.Timestamp(today) - pd.DateOffset(years=PREDICTION_HISTORY_YEARS)
        ).date().isoformat()
        min_date = first_market_open(earliest)

        return {
            "available": True,
            "min_date": min_date,
            "max_date": upper,
            "default_date": upper,
            "next_open": upper,
        }
    except Exception as exc:  # noqa: BLE001 — degrade gracefully for the UI
        return {"available": False, "reason": str(exc)}


@app.get("/api/predictions")
def get_predictions() -> dict:
    rows = _read_predictions()
    return {"count": len(rows), "predictions": rows}


@app.get("/api/predictions/live")
def predictions_live() -> dict:
    """Current situation per predicted stock, referenced to its open price.

    For each stock we pull the prediction's target-day bar from yfinance and
    report how its price has moved from that session's OPEN:
      * target day is today  -> 'live' intraday move (open -> latest price)
      * target day is past    -> that day's realised move (open -> close)
      * target day not traded yet (future) -> 'pending'
    ``matched`` says whether the realised direction agrees with the signal.
    Always returns HTTP 200 — the UI degrades gracefully on any error.
    """
    rows = _read_predictions()
    if not rows:
        return {"available": False, "reason": "no predictions"}
    try:
        import pandas as pd
        import yfinance as yf

        target = rows[0].get("target_date") or rows[0].get("as_of_date")
        tgt = pd.Timestamp(target).normalize()
        today = pd.Timestamp.now().normalize()
        signal_by = {r.get("ticker"): (r.get("signal") or "").upper() for r in rows}
        tickers = [r.get("ticker") for r in rows if r.get("ticker")]
        if not tickers:
            return {"available": False, "reason": "no tickers"}

        # Small bounded window around the target day (fast regardless of how old it is).
        start = (tgt - pd.Timedelta(days=4)).strftime("%Y-%m-%d")
        end = (tgt + pd.Timedelta(days=6)).strftime("%Y-%m-%d")
        data = yf.download(tickers, start=start, end=end, group_by="ticker",
                           threads=True, progress=False, auto_adjust=False)

        out: dict[str, dict] = {}
        for tk in dict.fromkeys(tickers):  # de-dup, keep order
            try:
                df = data[tk].dropna(how="all")
                if df.empty:
                    out[tk] = {"status": "pending"}
                    continue
                idx = pd.to_datetime(df.index)
                if getattr(idx, "tz", None) is not None:
                    idx = idx.tz_localize(None)
                df = df.copy()
                df.index = idx.normalize()
                on_after = df[df.index >= tgt]
                if on_after.empty:
                    out[tk] = {"status": "pending"}
                    continue
                day = on_after.index[0]
                ref_open = float(df.loc[day, "Open"])
                current = float(df.loc[day, "Close"])
                if ref_open <= 0:
                    out[tk] = {"status": "error"}
                    continue
                change = (current - ref_open) / ref_open * 100.0
                actual = "UP" if change >= 0 else "DOWN"
                sig = signal_by.get(tk)
                out[tk] = {
                    "open": round(ref_open, 2),
                    "current": round(current, 2),
                    "change_pct": round(change, 2),
                    "actual": actual,
                    "status": "live" if day == today else "closed",
                    "matched": (actual == sig) if sig in ("UP", "DOWN") else None,
                }
            except Exception:  # noqa: BLE001 — one ticker must not break the rest
                out[tk] = {"status": "error"}

        return {"available": True, "target": target, "rows": out}
    except Exception as exc:  # noqa: BLE001 — degrade gracefully for the UI
        return {"available": False, "reason": str(exc)}


# ----------------------------------------------------------------------------
# Agentic investment advisor (LangGraph)
# ----------------------------------------------------------------------------
@app.get("/api/advisor/status")
def advisor_status() -> dict:
    """Readiness of the advisor (deps installed, LLM/Tavily keys present)."""
    return ADV.advisor_status()


@app.post("/api/advisor/plan")
def advisor_plan(req: AdvisorRequest) -> dict:
    """Draft three budget-aware investment plans from the latest predictions."""
    result = ADV.generate_plans(
        budget=req.budget,
        currency=req.currency,
        include_news=req.include_news,
        max_stocks=req.max_stocks,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Advisor failed"))
    return result


# ----------------------------------------------------------------------------
# Static UI (mounted last so it doesn't shadow /api routes)
# ----------------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    # no-store so the browser always re-fetches index.html and picks up the
    # latest ?v= asset references (otherwise a cached HTML pins old JS/CSS).
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
