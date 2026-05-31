"""Central configuration loader for QuantPilot.

Loads ``config.yaml`` once at import time and exposes it as a nested dict via
``CONFIG``. All entries under the ``paths`` section are resolved to absolute
paths relative to the project root so the pipeline runs identically regardless
of the current working directory.

Override the config file location with the ``QUANTPILOT_CONFIG`` env var.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Project root = parent of the ``signals`` package directory.
ROOT_DIR = Path(__file__).resolve().parents[1]

CONFIG_PATH = Path(os.environ.get("QUANTPILOT_CONFIG", ROOT_DIR / "config.yaml"))


def _resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve every value under ``paths`` to an absolute path string."""
    paths = cfg.get("paths", {})
    for key, value in paths.items():
        p = Path(value)
        if not p.is_absolute():
            p = ROOT_DIR / p
        paths[key] = str(p)
    # Resolve the inference checkpoint too (lives outside the paths section).
    ckpt = cfg.get("inference", {}).get("nn_checkpoint")
    if ckpt:
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = ROOT_DIR / ckpt_path
        cfg["inference"]["nn_checkpoint"] = str(ckpt_path)
    return cfg


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Read and parse the YAML config, resolving relative paths to absolute."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"QuantPilot config not found at {path}. "
            "Create config.yaml at the project root or set QUANTPILOT_CONFIG."
        )
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return _resolve_paths(cfg)


# Loaded once and shared across the package.
CONFIG: dict[str, Any] = load_config()
