"""Pipeline job manager.

Runs pipeline stages as subprocesses (``python signals/pipeline.py <stage>``)
so the heavy work is isolated from the web process, streaming each run's output
to a per-job log file. Jobs are tracked in memory and polled lazily.

Only one job runs at a time — the pipeline stages share on-disk data/artifact
directories, so concurrent runs could corrupt each other.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict

from . import ROOT_DIR

LOG_DIR = ROOT_DIR / "APP" / "logs"
PIPELINE = ROOT_DIR / "signals" / "pipeline.py"

# stage -> whether it accepts a --mode flag
STAGES = {
    "ingest": True,
    "preprocess": True,
    "train": False,
    "predict": False,
    "train-pipeline": False,
    "predict-pipeline": False,
}

_jobs: "OrderedDict[str, dict]" = OrderedDict()
_lock = threading.Lock()


def _public(job: dict) -> dict:
    """Strip non-serializable internals before returning a job to the client."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _refresh(job: dict) -> dict:
    if job["status"] == "running":
        rc = job["_proc"].poll()
        if rc is not None:
            job["returncode"] = rc
            job["status"] = "succeeded" if rc == 0 else "failed"
            job["ended"] = time.time()
            try:
                job["_logf"].close()
            except Exception:
                pass
    return job


def is_busy() -> bool:
    with _lock:
        return any(_refresh(j)["status"] == "running" for j in _jobs.values())


def start_job(stage: str, mode: str | None = None) -> dict:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Valid: {sorted(STAGES)}")
    if mode is not None and not STAGES[stage]:
        mode = None  # ignore mode for stages that don't take one
    if mode is not None and mode not in ("train", "predict"):
        raise ValueError("mode must be 'train' or 'predict'")
    if is_busy():
        raise RuntimeError("A pipeline job is already running. Wait for it to finish.")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    log_path = LOG_DIR / f"{job_id}.log"

    cmd = [sys.executable, str(PIPELINE), stage]
    if mode is not None:
        cmd += ["--mode", mode]

    logf = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT_DIR), stdout=logf, stderr=subprocess.STDOUT, text=True,
    )

    job = {
        "id": job_id,
        "stage": stage,
        "mode": mode,
        "command": " ".join(cmd),
        "status": "running",
        "pid": proc.pid,
        "started": time.time(),
        "ended": None,
        "returncode": None,
        "log_file": str(log_path),
        "_proc": proc,
        "_logf": logf,
    }
    with _lock:
        _jobs[job_id] = job
    return _public(job)


def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        return _public(_refresh(job)) if job else None


def list_jobs() -> list[dict]:
    with _lock:
        jobs = [_public(_refresh(j)) for j in _jobs.values()]
    return sorted(jobs, key=lambda j: j["started"], reverse=True)


def read_log(job_id: str, tail: int = 400) -> str | None:
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        return None
    try:
        with open(job["log_file"], "r") as f:
            lines = f.readlines()
        return "".join(lines[-tail:])
    except OSError:
        return ""


def stop_job(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        if job["status"] == "running":
            try:
                job["_proc"].terminate()
            except Exception:
                pass
            job["status"] = "cancelled"
            job["ended"] = time.time()
            try:
                job["_logf"].close()
            except Exception:
                pass
        return _public(job)
