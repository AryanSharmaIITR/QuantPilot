"""QuantPilot web application (FastAPI).

Importing this package wires the project root and the ``signals`` package onto
``sys.path`` so the app can reuse the existing pipeline modules (which use flat
imports like ``import data``).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SIGNALS_DIR = ROOT_DIR / "signals"

for _p in (str(ROOT_DIR), str(SIGNALS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ["ROOT_DIR", "SIGNALS_DIR"]
