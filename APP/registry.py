"""Instrument registry — backs the web app's stock/ticker management.

The registry is a JSON file (``tickers.json``) at the project root. It is the
source of truth for the instrument universe; ``signals/data.py`` reads it when
present and otherwise falls back to its built-in defaults. Editing it from the
web app therefore changes what the pipeline ingests and trains on.
"""
from __future__ import annotations

import json
import threading

from . import ROOT_DIR

import data as D  # signals/data.py (path wired in APP/__init__.py)

REGISTRY_PATH = ROOT_DIR / "tickers.json"
_lock = threading.Lock()

VALID_KINDS = ("stocks", "market")


def _default_registry() -> dict:
    return {"stocks": dict(D.DEFAULT_STOCKS), "market": dict(D.DEFAULT_NSE)}


def ensure_seeded() -> None:
    """Create tickers.json from the built-in defaults if it doesn't exist yet."""
    if not REGISTRY_PATH.exists():
        save_registry(_default_registry())


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return _default_registry()
    try:
        with open(REGISTRY_PATH, "r") as f:
            reg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return _default_registry()
    reg.setdefault("stocks", {})
    reg.setdefault("market", {})
    return reg


def save_registry(reg: dict) -> None:
    with _lock:
        with open(REGISTRY_PATH, "w") as f:
            json.dump(reg, f, indent=2)


def list_instruments() -> dict:
    """Return both universes as ordered lists with their category indices."""
    reg = load_registry()
    return {
        kind: [
            {"name": name, "ticker": ticker, "category": idx}
            for idx, (name, ticker) in enumerate(reg[kind].items())
        ]
        for kind in VALID_KINDS
    }


def add_instrument(kind: str, name: str, ticker: str) -> dict:
    if kind not in VALID_KINDS:
        raise ValueError(f"kind must be one of {VALID_KINDS}")
    name, ticker = name.strip(), ticker.strip()
    if not name or not ticker:
        raise ValueError("Both name and ticker are required")
    reg = load_registry()
    if name in reg[kind]:
        raise ValueError(f"'{name}' already exists in {kind}")
    reg[kind][name] = ticker
    save_registry(reg)
    return reg


def remove_instrument(kind: str, name: str) -> dict:
    if kind not in VALID_KINDS:
        raise ValueError(f"kind must be one of {VALID_KINDS}")
    reg = load_registry()
    if name not in reg[kind]:
        raise KeyError(name)
    del reg[kind][name]
    save_registry(reg)
    return reg


def reset_to_defaults() -> dict:
    reg = _default_registry()
    save_registry(reg)
    return reg
